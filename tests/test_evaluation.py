from __future__ import annotations

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.evaluation import (
    mean_metrics,
    measure,
    parse_metric,
    per_query_metrics,
    relevant_from_baseline,
    relevant_from_qrels,
    retention,
    split_queries,
)
from optivision.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from optivision.representation import MultiVectorCorpus
from optivision.stages import BinaryQuantizer


def test_parse_metric():
    assert parse_metric("nDCG@5") == ("ndcg", 5)
    with pytest.raises(ValueError):
        parse_metric("map@10")


def test_per_query_metrics_match_the_reference_functions(rng):
    n_q, n_d = 30, 25
    scores = rng.standard_normal((n_q, n_d))
    relevant = [np.sort(rng.choice(n_d, size=rng.integers(1, 4), replace=False)) for _ in range(n_q)]
    got = per_query_metrics(scores, relevant, ["ndcg@5", "ndcg@10", "recall@5", "mrr@10"])
    for qi in range(n_q):
        ranked = [str(i) for i in np.argsort(-scores[qi], kind="stable")]
        rel = {str(i) for i in relevant[qi]}
        assert got["ndcg@5"][qi] == pytest.approx(ndcg_at_k(ranked, rel, 5))
        assert got["ndcg@10"][qi] == pytest.approx(ndcg_at_k(ranked, rel, 10))
        assert got["recall@5"][qi] == pytest.approx(recall_at_k(ranked, rel, 5))
        assert got["mrr@10"][qi] == pytest.approx(mrr_at_k(ranked, rel, 10))


def test_queries_without_relevant_documents_are_skipped():
    scores = np.array([[0.2, 0.9], [0.5, 0.1]])
    pq = per_query_metrics(scores, [np.array([1]), np.array([], dtype=np.int64)], ["ndcg@1"])
    assert pq["ndcg@1"][0] == 1.0 and np.isnan(pq["ndcg@1"][1])
    assert mean_metrics(pq)["ndcg@1"] == 1.0


def test_relevance_helpers():
    rel = relevant_from_qrels({"a": {"d2", "zz"}, "b": set()}, ["a", "b"], ["d1", "d2"])
    assert rel[0].tolist() == [1] and rel[1].size == 0
    base = relevant_from_baseline(np.array([[0.1, 0.9, 0.5]]), depth=2)
    assert base[0].tolist() == [1, 2]


def test_retention_ci_brackets_the_point(rng):
    b = rng.random(200)
    c = b * 0.9
    point, lo, hi = retention(c, b, n_boot=500)
    assert point == pytest.approx(0.9)
    assert lo <= point <= hi


def test_split_queries_is_disjoint_and_complete():
    cal, hold = split_queries(11, 0.5, seed=3)
    assert set(cal) | set(hold) == set(range(11)) and not set(cal) & set(hold)
    again, _ = split_queries(11, 0.5, seed=3)
    assert cal.tolist() == again.tolist()


def test_measure_float_baseline_is_self_consistent(rng):
    docs = [rng.standard_normal((n, 8)).astype(np.float32) for n in (4, 6, 5)]
    queries = MultiVectorCorpus.from_arrays([rng.standard_normal((3, 8)).astype(np.float32) for _ in range(5)])
    corpus = MultiVectorCorpus.from_arrays(docs)
    base, scores = measure(Pipeline(), corpus, queries, [np.array([0])] * 5)
    rel = relevant_from_baseline(scores, 1)
    again, _ = measure(Pipeline(), corpus, queries, rel, metrics=["ndcg@1"])
    assert again.means["ndcg@1"] == 1.0  # the baseline always agrees with itself
    binary, _ = measure(Pipeline([BinaryQuantizer()]), corpus, queries, rel, metrics=["ndcg@1"])
    assert binary.report["compression_vs_float32"] == pytest.approx(32.0)
    assert base.summary()["query_ms"] >= 0


def test_vector_dataset_round_trip(tmp_path):
    import json

    from optivision.benchmark import load_vector_dataset
    from optivision.representation import MultiVectorCorpus

    rng = np.random.default_rng(0)
    docs = MultiVectorCorpus.from_arrays([rng.standard_normal((5, 8)).astype(np.float32) for _ in range(3)],
                                         ids=["00000::p1", "00001::p1", "00002::p1"], attrs={"model": "m"})
    queries = MultiVectorCorpus.from_arrays([rng.standard_normal((2, 8)).astype(np.float32) for _ in range(2)],
                                            ids=["q0000", "q0001"])
    docs.save(tmp_path / "t_docs.npz")
    queries.save(tmp_path / "t_queries.npz")
    (tmp_path / "t_qrels.json").write_text(json.dumps({"q0000": ["00002::p1"], "q0001": ["00000::p1"]}))
    ds = load_vector_dataset(tmp_path / "t")
    assert ds.name == "t" and ds.attrs["model"] == "m"
    assert [r.tolist() for r in ds.relevant()] == [[2], [0]]

    (tmp_path / "t_qrels.json").write_text(json.dumps({"q9999": ["00000::p1"]}))
    with pytest.raises(ValueError, match="no vectors"):
        load_vector_dataset(tmp_path / "t")
