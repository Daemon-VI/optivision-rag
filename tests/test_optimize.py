from __future__ import annotations

import numpy as np
import pytest

import optivision
from optivision.compose import Pipeline
from optivision.pareto import choose, dominates, pareto_front
from optivision.stages import AdaptiveMerge, BinaryQuantizer, Float16Quantizer, Int8Quantizer

pytestmark = [pytest.mark.filterwarnings("ignore:calibrating on"), pytest.mark.filterwarnings("ignore:Int8Quantizer")]


def _rows():
    return [
        {"name": "fp32", "bytes_per_doc": 100.0, "retention": 1.00},
        {"name": "int8", "bytes_per_doc": 25.0, "retention": 0.999},
        {"name": "bad", "bytes_per_doc": 30.0, "retention": 0.95},  # dominated by int8
        {"name": "bin", "bytes_per_doc": 3.0, "retention": 0.96},
        {"name": "bin2", "bytes_per_doc": 3.0, "retention": 0.96},  # duplicate point
    ]


def test_pareto_front_drops_dominated_and_duplicate_points():
    front = pareto_front(_rows())
    assert [r["name"] for r in front] == ["bin", "int8", "fp32"]
    assert dominates(_rows()[1], _rows()[2], ["bytes_per_doc"], "retention")


def test_choose_by_target_or_budget():
    assert choose(_rows(), quality_target=0.99)["name"] == "int8"
    assert choose(_rows(), quality_target=0.5)["name"] == "bin"
    assert choose(_rows(), max_cost=26.0)["name"] == "int8"
    assert choose(_rows(), quality_target=1.1) is None


def _task(seed=2):
    rng = np.random.default_rng(seed)
    dim = 16
    topics = rng.standard_normal((30, dim))
    docs, queries, qrels = [], [], {}
    for i in range(24):
        v = topics[np.repeat(rng.choice(30, 4, replace=False), 10)] + 0.3 * rng.standard_normal((40, dim))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        docs.append(v.astype(np.float32))
        q = v[rng.choice(40, 5, replace=False)] + 0.3 * rng.standard_normal((5, dim))
        queries.append((q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32))
        qrels[str(i)] = {str(i)}
    return docs, queries, qrels


def test_optimize_end_to_end():
    docs, queries, qrels = _task()
    space = {"m": [Pipeline(([AdaptiveMerge(radius=r)] if r else []) + [q()])
                   for r in (None, 0.8, 0.6) for q in (Float16Quantizer, Int8Quantizer, BinaryQuantizer)]}
    res = optivision.optimize(docs, queries=queries, qrels=qrels, quality_target=0.9, search_space=space,
                              assume_monotone=False, n_boot=100)
    assert res.compressed.num_vectors <= sum(d.shape[0] for d in docs)
    assert res.compression_ratio >= 2.0
    assert 0.0 < res.memory_saving < 1.0
    assert res.quality_retention is not None
    report = res.report()
    assert report["label"] == res.pipeline.label()


def test_optimize_needs_queries():
    docs, _, _ = _task()
    with pytest.raises(ValueError, match="sample queries"):
        optivision.optimize(docs)


def test_optimize_falls_back_loudly_when_nothing_qualifies():
    docs, queries, qrels = _task()
    space = {"b": [Pipeline([AdaptiveMerge(radius=0.1), BinaryQuantizer()])]}
    res = optivision.optimize(docs, queries=queries, qrels=qrels, quality_target=1.0, search_space=space, n_boot=0)
    assert res.target_met is None and res.quality_retention is None
    assert res.compression_ratio == pytest.approx(1.0)
