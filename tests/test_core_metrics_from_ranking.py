"""``metrics_from_ranking`` reproduces ``per_query_metrics`` from a ranking, exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optivision import evaluation
from optivision.core.evaluate import metrics as core_metrics
from optivision.core.evaluate.metrics import (
    DEFAULT_METRICS,
    mean_metrics,
    metrics_from_ranking,
    parse_metric,
    per_query_metrics,
)
from optivision.scoring import rank

ALL_METRICS = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "recall@1", "recall@5", "recall@10",
               "mrr@1", "mrr@5", "mrr@10", "hit@1", "hit@3", "hit@10"]


def same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    """Bitwise equality, NaN positions included."""
    return a.keys() == b.keys() and all(a[k].tobytes() == b[k].tobytes() for k in a)


def random_problem(rng, n_q, n_d, tie=False):
    if tie:
        scores = rng.integers(0, 4, size=(n_q, n_d)).astype(np.float32)  # many exact ties
    else:
        scores = rng.random((n_q, n_d)).astype(np.float32)
    relevant = []
    for _ in range(n_q):
        size = int(rng.choice([0, 1, 1, 2, 3, 6, min(n_d, 15)]))
        size = min(size, n_d)
        relevant.append(np.sort(rng.choice(n_d, size=size, replace=False)).astype(np.int64))
    return scores, relevant


def test_the_library_functions_are_re_exported_not_copied():
    assert core_metrics.per_query_metrics is evaluation.per_query_metrics
    assert core_metrics.parse_metric is evaluation.parse_metric
    assert core_metrics.mean_metrics is evaluation.mean_metrics
    assert DEFAULT_METRICS is evaluation.DEFAULT_METRICS
    assert parse_metric("ndcg@5") == ("ndcg", 5)
    assert mean_metrics({"m": np.array([1.0, np.nan, 0.0])}) == {"m": 0.5}


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("tie", [False, True])
def test_parity_with_per_query_metrics_on_random_score_matrices(seed, tie):
    rng = np.random.default_rng(seed)
    n_q, n_d = int(rng.integers(1, 40)), int(rng.integers(12, 60))
    scores, relevant = random_problem(rng, n_q, n_d, tie)
    kmax = max(parse_metric(m)[1] for m in ALL_METRICS)
    dense = per_query_metrics(scores, relevant, ALL_METRICS)
    ranked = metrics_from_ranking(rank(scores, kmax), relevant, ALL_METRICS)
    assert same(ranked, dense)
    # NaN exactly where a query has no relevant document
    nan_expected = np.array([r.size == 0 for r in relevant])
    for name in ALL_METRICS:
        assert np.array_equal(np.isnan(ranked[name]), nan_expected)


def test_parity_with_the_default_metric_set():
    rng = np.random.default_rng(99)
    scores, relevant = random_problem(rng, 25, 30)
    assert same(metrics_from_ranking(rank(scores, 10), relevant),
                per_query_metrics(scores, relevant))
    assert same(metrics_from_ranking(rank(scores, 10), relevant, DEFAULT_METRICS),
                per_query_metrics(scores, relevant, DEFAULT_METRICS))


def test_a_wider_ranking_than_the_largest_cutoff_changes_nothing():
    rng = np.random.default_rng(4)
    scores, relevant = random_problem(rng, 20, 50)
    dense = per_query_metrics(scores, relevant, ["ndcg@5", "mrr@5", "recall@5"])
    for width in (5, 6, 17, 50):
        assert same(metrics_from_ranking(rank(scores, width), relevant,
                                         ["ndcg@5", "mrr@5", "recall@5"]), dense)


@pytest.mark.parametrize("n_d", [1, 2, 3, 7, 9])
def test_a_corpus_smaller_than_the_cutoff_is_expressed_with_minus_one_pads(n_d):
    rng = np.random.default_rng(n_d)
    scores, relevant = random_problem(rng, 15, n_d)
    metrics = ["ndcg@10", "recall@10", "mrr@10", "hit@3"]
    dense = per_query_metrics(scores, relevant, metrics)  # rank() returns n_d columns here
    top = np.full((15, 10), -1, dtype=np.int64)
    top[:, :n_d] = rank(scores, 10)
    assert same(metrics_from_ranking(top, relevant, metrics), dense)
    with pytest.raises(ValueError, match="largest cut-off"):
        metrics_from_ranking(rank(scores, 10), relevant, metrics)  # cut short: refused


def test_rows_with_different_numbers_of_documents_are_scored_row_by_row():
    top = np.array([[4, 2, 0, 1, 3], [2, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])
    relevant = [np.array([2]), np.array([2, 7]), np.array([1])]
    out = metrics_from_ranking(top, relevant, ["ndcg@5", "recall@5", "mrr@5", "hit@1"])
    one = metrics_from_ranking(top[1:2], relevant[1:2], ["ndcg@5", "recall@5", "mrr@5", "hit@1"])
    for name, values in out.items():
        assert values[1] == one[name][0]
    assert out["hit@1"].tolist() == [0.0, 1.0, 0.0]  # an empty ranking has no hit
    assert out["mrr@5"][2] == 0.0 and out["ndcg@5"][2] == 0.0


def test_a_single_ranking_is_treated_like_a_single_query():
    scores = np.random.default_rng(2).random(20).astype(np.float32)
    relevant = [np.array([3, 8])]
    assert same(metrics_from_ranking(rank(scores, 10), relevant, ["ndcg@5", "mrr@10"]),
                per_query_metrics(scores, relevant, ["ndcg@5", "mrr@10"]))


def test_metric_definitions_on_a_hand_worked_example():
    top = np.array([[3, 7, 1, 9, 2]])
    relevant = [np.array([1, 2])]
    out = metrics_from_ranking(top, relevant, ["ndcg@5", "recall@5", "recall@2", "mrr@5",
                                               "mrr@2", "hit@2", "hit@3"])
    dcg = 1 / math.log2(4) + 1 / math.log2(6)  # relevant documents at ranks 3 and 5
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert out["ndcg@5"][0] == pytest.approx(dcg / ideal, rel=1e-15, abs=0)
    assert out["recall@5"][0] == 1.0 and out["recall@2"][0] == 0.0
    assert out["mrr@5"][0] == 1 / 3 and out["mrr@2"][0] == 0.0
    assert out["hit@2"][0] == 0.0 and out["hit@3"][0] == 1.0


def test_a_query_without_a_relevant_document_is_nan_for_every_metric():
    out = metrics_from_ranking(np.array([[0, 1, 2]]), [np.array([], dtype=np.int64)],
                               ["ndcg@3", "recall@3", "mrr@3", "hit@3"])
    assert all(np.isnan(v[0]) for v in out.values())


def test_the_input_is_not_modified_and_the_output_is_a_fresh_float64_array():
    top = np.array([[2, 1, 0]])
    relevant = [np.array([1])]
    copy = top.copy()
    out = metrics_from_ranking(top, relevant, ["ndcg@3"])
    assert np.array_equal(top, copy) and out["ndcg@3"].dtype == np.float64


def test_an_unsupported_metric_is_refused_like_the_library_does():
    with pytest.raises(ValueError, match="unsupported metric"):
        metrics_from_ranking(np.array([[0, 1]]), [np.array([0])], ["map@5"])
    with pytest.raises(ValueError):
        metrics_from_ranking(np.array([[0, 1]]), [np.array([0])], ["ndcg@0"])


@pytest.mark.parametrize(("top", "relevant", "error", "message"), [
    (np.array([[0.0, 1.0, 2.0]]), [np.array([0])], TypeError, "integer"),
    (np.zeros((1, 2, 3), dtype=np.int64), [np.array([0])], ValueError, "n_queries, k"),
    (np.array([[0, 1, 2]]), [np.array([0]), np.array([1])], ValueError, "relevant has 2"),
    (np.array([[0, 1]]), [np.array([0])], ValueError, "largest cut-off"),
    (np.array([[0, -2, 1]]), [np.array([0])], ValueError, "pads"),
    (np.array([[0, -1, 1]]), [np.array([0])], ValueError, "followed by a document"),
    (np.array([[0, 1, 1]]), [np.array([0])], ValueError, "twice"),
])
def test_malformed_rankings_are_refused(top, relevant, error, message):
    with pytest.raises(error, match=message):
        metrics_from_ranking(top, relevant, ["ndcg@3"])


def test_a_repeat_beyond_the_cutoff_is_not_read_and_not_refused():
    top = np.array([[0, 1, 2, 2, 2]])
    metrics_from_ranking(top, [np.array([0])], ["ndcg@3"])


def test_unsigned_and_small_integer_rankings_are_accepted():
    top = np.array([[2, 0, 1]])
    relevant = [np.array([0])]
    expected = metrics_from_ranking(top.astype(np.int64), relevant, ["ndcg@3"])
    for dtype in (np.int32, np.uint8, np.uint32):
        assert same(metrics_from_ranking(top.astype(dtype), relevant, ["ndcg@3"]), expected)
