from __future__ import annotations

import numpy as np
import pytest

from optivision.calibration import calibrate
from optivision.compose import Pipeline
from optivision.representation import MultiVectorCorpus
from optivision.stages import AdaptiveMerge, BinaryQuantizer, Float16Quantizer, Int8Quantizer

pytestmark = pytest.mark.filterwarnings("ignore:calibrating on")


def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def retrieval_task():
    """Documents made of a few topics each; a query is a noisy fragment of one document."""
    rng = np.random.default_rng(1)
    dim, n_docs = 24, 40
    topics = _unit(rng.standard_normal((60, dim)))
    docs, queries, qrels = [], [], {}
    for i in range(n_docs):
        own = rng.choice(60, size=6, replace=False)
        v = topics[np.repeat(own, 12)] + 0.35 * rng.standard_normal((72, dim))
        docs.append(_unit(v).astype(np.float32))
        pick = rng.choice(72, size=6, replace=False)
        q = docs[-1][pick] + 0.3 * rng.standard_normal((6, dim))
        queries.append(_unit(q).astype(np.float32))
        qrels[f"q{i}"] = {f"d{i}"}
    return (
        MultiVectorCorpus.from_arrays(docs, ids=[f"d{i}" for i in range(n_docs)]),
        MultiVectorCorpus.from_arrays(queries, ids=[f"q{i}" for i in range(n_docs)]),
        qrels,
    )


def _space():
    radii = [None, 0.9, 0.7, 0.5, 0.3]
    return {
        f"merge+{name}": [Pipeline(([AdaptiveMerge(radius=r)] if r is not None else []) + [cls()]) for r in radii]
        for name, cls in (("float16", Float16Quantizer), ("int8", Int8Quantizer), ("binary", BinaryQuantizer))
    }


def test_selects_the_smallest_feasible_candidate(retrieval_task):
    corpus, queries, qrels = retrieval_task
    res = calibrate(corpus, queries, qrels, quality_target=0.9, search_space=_space(), n_boot=200)
    feasible = [c for c in res.candidates if c.feasible]
    assert res.selected is not None
    assert res.selected.bytes_per_doc == min(c.bytes_per_doc for c in feasible)
    assert all(c.retention >= 0.9 for c in feasible)


def test_lower_targets_never_select_larger_configurations(retrieval_task):
    corpus, queries, qrels = retrieval_task
    sizes = []
    for target in (0.99, 0.9, 0.7):
        res = calibrate(corpus, queries, qrels, quality_target=target, search_space=_space(), n_boot=0)
        sizes.append(res.selected.bytes_per_doc if res.selected else float("inf"))
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_holdout_is_measured_on_disjoint_queries(retrieval_task):
    corpus, queries, qrels = retrieval_task
    res = calibrate(corpus, queries, qrels, quality_target=0.8, search_space=_space(), n_boot=100)
    assert res.n_calibration_queries + res.n_holdout_queries == len(queries)
    assert res.holdout is not None and "retention_ci" in res.holdout
    assert isinstance(res.met_target_on_holdout, bool)
    assert res.summary()["selected"]["pipeline"] == res.pipeline.to_dict()


def test_label_free_reference(retrieval_task):
    corpus, queries, _ = retrieval_task
    res = calibrate(corpus, queries, None, quality_target=0.95, search_space=_space(), n_boot=0)
    assert res.reference == "baseline@1"
    # the float16 codec alone can never change the float ranking much
    assert res.candidates[0].retention > 0.95


def test_monotone_early_stop_skips_more_aggressive_steps(retrieval_task):
    corpus, queries, qrels = retrieval_task
    space = {"binary": [Pipeline([AdaptiveMerge(radius=r), BinaryQuantizer()]) for r in (0.3, 0.2, 0.1)]}
    res = calibrate(corpus, queries, qrels, quality_target=0.999, search_space=space, n_boot=0)
    assert len(res.candidates) == 1 and res.selected is None and res.holdout is None


def test_byte_budget_is_respected(retrieval_task):
    corpus, queries, qrels = retrieval_task
    res = calibrate(corpus, queries, qrels, quality_target=0.5, search_space=_space(), n_boot=0,
                    max_bytes_per_doc=1.0)
    assert res.selected is None


def test_argument_validation(retrieval_task):
    corpus, queries, qrels = retrieval_task
    with pytest.raises(ValueError):
        calibrate(corpus, queries, qrels, quality_target=1.5)
    with pytest.raises(ValueError):
        calibrate(corpus, queries, None, reference="labels")
    with pytest.raises(ValueError):
        calibrate(corpus, queries.select([0, 1]), qrels)


def test_select_reuses_scores_for_any_target(retrieval_task):
    from optivision.calibration import score_space, select
    from optivision.evaluation import relevant_from_qrels, split_queries
    from optivision.scoring import maxsim_matrix

    corpus, queries, qrels = retrieval_task
    scored = score_space(corpus, queries, _space())
    base = maxsim_matrix(queries, corpus)
    rel = relevant_from_qrels(qrels, queries.ids, corpus.ids)
    cal, hold = split_queries(len(queries), 0.5, 0)
    sizes = []
    for target in (0.99, 0.9, 0.7):
        judged, chosen, holdout = select(scored, base, rel, cal, hold, target, n_boot=0)
        assert len(judged) == len(scored)
        sizes.append(chosen.bytes_per_doc if chosen else float("inf"))
        if chosen:
            assert holdout["bytes_per_doc"] == chosen.bytes_per_doc
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_pseudo_queries_are_fragments_of_their_document(retrieval_task):
    from optivision.calibration import pseudo_queries

    corpus, _, _ = retrieval_task
    q, qrels = pseudo_queries(corpus, n_queries=10, tokens=5, seed=1)
    assert len(q) == 10 and all(len(v) == 1 for v in qrels.values())
    doc = corpus[corpus.ids.index(next(iter(qrels[q.ids[0]])))]
    rows = {tuple(r) for r in np.round(doc.vectors, 6)}
    assert all(tuple(r) in rows for r in np.round(q[0].vectors, 6))


def test_small_calibration_sets_warn(retrieval_task):
    corpus, queries, qrels = retrieval_task
    with pytest.warns(UserWarning, match="calibrating on 20 queries"):
        calibrate(corpus, queries, qrels, quality_target=0.9, search_space=_space(), n_boot=0)


def test_lower_bound_is_the_default_rule(retrieval_task):
    corpus, queries, qrels = retrieval_task
    res = calibrate(corpus, queries, qrels, quality_target=0.9, search_space=_space(), n_boot=200)
    assert res.safety == "lower_ci"
    if res.selected is not None:
        assert res.selected.retention_lo >= 0.9
