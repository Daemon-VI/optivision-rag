"""References: labels versus agreement with the uncompressed ranking."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from optivision import evaluation
from optivision.calibration import reference_relevance
from optivision.core.evaluate import LabelReference, RankingReference, Reference, per_query_metrics
from optivision.core.evaluate import reference as reference_module
from optivision.core.identity import canonical_json

DOCS = [f"d{i}" for i in range(8)]
QUERIES = ["q0", "q1", "q2", "q3"]
QRELS = {"q0": ["d1", "d4"], "q1": {"d2": 1, "d3": 2}, "q2": ("d7",), "q3": []}


def base_scores(seed=0, n_q=4, n_d=8):
    return np.random.default_rng(seed).random((n_q, n_d)).astype(np.float32)


# ---------------------------------------------------------------------- LabelReference


def test_label_reference_delegates_to_the_librarys_qrels_function():
    ref = LabelReference(QRELS)
    got = ref.relevant(DOCS, QUERIES, None)
    want = evaluation.relevant_from_qrels(QRELS, QUERIES, DOCS)
    assert ref.construct == "labels" and isinstance(ref, Reference)
    assert len(got) == len(want) == 4
    for a, b in zip(got, want, strict=True):
        assert a.dtype == b.dtype == np.int64 and np.array_equal(a, b)
    assert [r.tolist() for r in got] == [[1, 4], [2, 3], [7], []]


def test_label_reference_matches_the_calibration_helper_for_labels():
    corpus, queries = SimpleNamespace(ids=DOCS), SimpleNamespace(ids=QUERIES)
    ref = LabelReference(QRELS)
    via_calibration = reference_relevance(corpus, queries, QRELS, "labels", base_scores())
    for a, b in zip(ref.relevant(DOCS, QUERIES, base_scores()), via_calibration, strict=True):
        assert np.array_equal(a, b)


def test_labels_are_binary_every_grade_above_zero_counts_as_relevant():
    ref = LabelReference({"q1": {"d2": 1, "d3": 0.25, "d4": np.int64(7), "d5": True}})
    assert ref.relevant(DOCS, ["q1"], None)[0].tolist() == [2, 3, 4, 5]


@pytest.mark.parametrize("grade", [0, 0.0, -1, -0.5, False, float("nan"), np.int64(0)])
def test_a_grade_of_zero_or_below_is_refused_with_the_remedy(grade):
    with pytest.raises(ValueError, match=r"qrels\['q1'\]\['d3'\].*Drop the entries that are not relevant"):
        LabelReference({"q0": ["d1"], "q1": {"d2": 1, "d3": grade}})


@pytest.mark.parametrize("grade", ["2", None, "high", [1]])
def test_a_grade_that_is_not_a_number_is_refused(grade):
    with pytest.raises(TypeError, match="must be a number"):
        LabelReference({"q1": {"d2": grade}})


def test_the_remedy_in_the_message_works():
    graded = {"q1": {"d2": 2, "d3": 0}, "q2": {"d7": 1}}
    with pytest.raises(ValueError):
        LabelReference(graded)
    binary = {q: {d: g for d, g in docs.items() if g > 0} for q, docs in graded.items()}
    assert [r.tolist() for r in LabelReference(binary).relevant(DOCS, ["q1", "q2"], None)] == [[2], [7]]


def test_lists_and_sets_are_not_graded_so_nothing_is_refused():
    ref = LabelReference({"q0": ["d1"], "q1": {"d2", "d3"}, "q2": ("d7",)})
    assert [r.tolist() for r in ref.relevant(DOCS, ["q0", "q1", "q2"], None)] == [[1], [2, 3], [7]]


def test_unknown_documents_are_ignored_and_unlabelled_queries_get_an_empty_set():
    ref = LabelReference({"q0": ["d1", "zzz"]})
    got = ref.relevant(DOCS, ["q0", "unlabelled"], None)
    assert got[0].tolist() == [1] and got[1].size == 0


def test_the_qrels_are_copied_at_construction():
    qrels = {"q0": ["d1"]}
    ref = LabelReference(qrels)
    before = ref.identity()
    qrels["q0"].append("d2")
    qrels["q9"] = ["d3"]
    assert ref.relevant(DOCS, ["q0"], None)[0].tolist() == [1]
    assert ref.identity() == before


def test_the_label_identity_does_not_depend_on_dictionary_or_collection_ordering():
    a = LabelReference({"q0": ["d1", "d4"], "q1": ["d2"]})
    b = LabelReference({"q1": ("d2",), "q0": {"d4", "d1"}})
    c = LabelReference({"q1": {"d2": 3}, "q0": {"d4": 1, "d1": 1}})  # grades are not read
    assert a.identity() == b.identity() == c.identity()


def test_the_label_identity_changes_with_any_change_to_what_the_metric_reads():
    base = LabelReference(QRELS).identity()["qrels_sha256"]

    def digest_of(qrels):
        return LabelReference(qrels).identity()["qrels_sha256"]

    assert digest_of({**QRELS, "q3": ["d0"]}) != base  # a label added
    assert digest_of({**QRELS, "q0": ["d1"]}) != base  # a label removed
    assert digest_of({**QRELS, "q0": ["d1", "d5"]}) != base  # a label changed
    assert digest_of({"q9" if k == "q0" else k: v for k, v in QRELS.items()}) != base
    assert digest_of({**QRELS, "q0": ["d1", "d4", "d4"]}) != base  # a repeat changes the count
    ids = LabelReference(QRELS).identity()
    assert ids["construct"] == "labels" and len(ids["qrels_sha256"]) == 64
    assert json.loads(json.dumps(ids)) == ids and canonical_json(ids)


def test_label_reference_refuses_malformed_qrels():
    with pytest.raises(TypeError, match="mapping"):
        LabelReference([("q0", ["d1"])])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not a single string"):
        LabelReference({"q0": "d1"})
    with pytest.raises(TypeError):
        LabelReference({"q0": 5})  # type: ignore[dict-item]


# ----------------------------------------------------------------- RankingReference


def test_ranking_reference_default_is_agreement_at_one():
    ref = RankingReference()
    assert ref.depth == 1 and ref.construct == "agreement@1" and isinstance(ref, Reference)
    assert RankingReference(depth=1).construct == "agreement@1"
    assert RankingReference(3).construct == "agreement@3"
    assert RankingReference(np.int64(2)).construct == "agreement@2"
    assert ref.identity() == {"construct": "agreement@1", "depth": 1}
    assert RankingReference(3).identity() == {"construct": "agreement@3", "depth": 3}


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "1", None])
def test_ranking_reference_refuses_a_bad_depth(bad):
    with pytest.raises(ValueError):
        RankingReference(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_ranking_reference_delegates_to_the_librarys_baseline_function(depth):
    scores = base_scores(3)
    got = RankingReference(depth).relevant(DOCS, QUERIES, scores)
    want = evaluation.relevant_from_baseline(scores, depth=depth)
    via_calibration = reference_relevance(SimpleNamespace(ids=DOCS), SimpleNamespace(ids=QUERIES),
                                          None, "baseline", scores, baseline_depth=depth)
    assert len(got) == 4
    for a, b, c in zip(got, want, via_calibration, strict=True):
        assert a.size == depth and np.array_equal(a, b) and np.array_equal(a, c)
        assert np.array_equal(a, np.sort(a))


def test_exact_score_ties_go_to_the_lower_document_index():
    scores = np.zeros((2, 8), dtype=np.float32)
    scores[1, [3, 5]] = 1.0
    got = RankingReference().relevant(DOCS, ["a", "b"], scores)
    assert [g.tolist() for g in got] == [[0], [3]]


def test_the_baseline_scores_one_on_every_query_by_construction():
    scores = base_scores(5, n_q=30, n_d=40)
    ids = [f"q{i}" for i in range(30)]
    docs = [f"d{i}" for i in range(40)]
    relevant = RankingReference().relevant(docs, ids, scores)
    assert (per_query_metrics(scores, relevant, ["ndcg@5"])["ndcg@5"] == 1.0).all()
    shifted = scores + 0.5 * np.random.default_rng(1).random(scores.shape).astype(np.float32)
    agreement = per_query_metrics(shifted, relevant, ["ndcg@5"])["ndcg@5"]
    top = np.array([r[0] for r in relevant])
    rank_of_top = (np.argsort(-shifted, axis=1, kind="stable") == top[:, None]).argmax(axis=1) + 1
    expected = np.where(rank_of_top <= 5, 1.0 / np.log2(1.0 + rank_of_top), 0.0)
    assert np.array_equal(agreement, expected)  # 1/log2(1+r) on the grid, else 0


@pytest.mark.parametrize("scores", [np.ones(8), np.ones((3, 8)), np.ones((4, 5)), np.ones((4, 8, 1))])
def test_ranking_reference_checks_the_score_matrix_against_the_ids(scores):
    with pytest.raises(ValueError):
        RankingReference().relevant(DOCS, QUERIES, scores)


def test_ranking_reference_needs_scores_and_a_depth_within_the_corpus():
    with pytest.raises(ValueError, match="baseline score matrix"):
        RankingReference().relevant(DOCS, QUERIES, None)
    with pytest.raises(ValueError, match="exceeds the corpus"):
        RankingReference(9).relevant(DOCS, QUERIES, base_scores())


def test_the_two_constructs_never_share_a_name_or_an_identity():
    labels, ranking = LabelReference(QRELS), RankingReference()
    assert labels.construct != ranking.construct
    assert labels.identity()["construct"] != ranking.identity()["construct"]
    assert RankingReference(1).construct != RankingReference(2).construct


def test_documentation_says_agreement_is_not_retrieval_quality():
    doc = " ".join(RankingReference.__doc__.split())
    assert "not retrieval quality" in doc and "including its mistakes" in doc
    assert "Only ``depth = 1`` was studied" in doc
    assert "never merged" in " ".join(reference_module.__doc__.split())


def test_ranking_reference_refuses_nan_and_names_the_first_query():
    scores = base_scores(1)
    scores[2, 5] = np.nan
    scores[3, 0] = np.nan
    with pytest.raises(ValueError, match=r"NaN \(first at query 2, id 'q2'\)"):
        RankingReference().relevant(DOCS, QUERIES, scores)


def test_ranking_reference_refuses_a_row_without_a_finite_score():
    scores = base_scores(1)
    scores[1, :] = -np.inf
    with pytest.raises(ValueError, match=r"no finite score for query 1 \(id 'q1'\)"):
        RankingReference().relevant(DOCS, QUERIES, scores)
    scores[1, :] = np.inf
    with pytest.raises(ValueError, match="no finite score for query 1"):
        RankingReference().relevant(DOCS, QUERIES, scores)


def test_ranking_reference_accepts_minus_infinity_next_to_finite_scores():
    scores = base_scores(1)
    scores[:, 0] = -np.inf  # an empty document
    got = RankingReference().relevant(DOCS, QUERIES, scores)
    assert all(g.size == 1 and g[0] != 0 for g in got)
    assert RankingReference().relevant(DOCS, QUERIES, scores.astype(np.float64))[0].size == 1
