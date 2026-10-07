"""The retrieval seam: Ranking, the protocols, ExactRetriever and CompressedRetriever.

Known-ranking cases T1-T9 and T16-T20 of docs/package/RETRIEVAL_BACKEND_AUDIT.md section 7 that
apply to the in-process retrievers (T10-T13 are in test_core_retrieve_tiered.py, T14-T15 in the
backend tests, T19 belongs to the metrics work). Small seeded synthetic data only.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from optivision.compose import Pipeline, load_compressed, save_compressed
from optivision.core.retrieve import (
    SCORE_MAXSIM_COSINE,
    SCORE_MAXSIM_IP,
    CompressedRetriever,
    DenseScorer,
    ExactRetriever,
    Ranking,
    Reranker,
    Retriever,
    TieredRetriever,
    TwoTierRetriever,
    as_queries,
    check_k,
    comparable,
    ranking_from_scores,
    require_comparable,
)
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, maxsim_pair, rank
from optivision.stages import (
    BinaryQuantizer,
    Float16Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
)
from optivision.storage import ExactIndex

SRC = Path(__file__).resolve().parents[1] / "src" / "optivision"


# ------------------------------------------------------------------ helpers


def _unit(v):
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _corpus(seed=0, n_docs=30, dim=16, lo=3, hi=12):
    rng = np.random.default_rng(seed)
    arrays = [_unit(rng.standard_normal((int(rng.integers(lo, hi)), dim))) for _ in range(n_docs)]
    return MultiVectorCorpus.from_arrays(arrays, ids=[f"doc{i}::p1" for i in range(n_docs)])


def _queries(seed=1, n=7, tokens=4, dim=16):
    rng = np.random.default_rng(seed)
    return MultiVectorCorpus.from_arrays([_unit(rng.standard_normal((tokens, dim))) for _ in range(n)])


def _docs(rows_per_doc, dim=2):
    arrays = [np.asarray(r, dtype=np.float32).reshape(-1, dim) for r in rows_per_doc]
    return MultiVectorCorpus.from_arrays(arrays, ids=[f"d{i}" for i in range(len(arrays))])


def _query_set(rows_per_query, dim=2):
    return _docs(rows_per_query, dim)


def assert_close_modulo_ties(a: Ranking, b: Ranking, rtol=1e-5, atol=1e-5):
    """Scores allclose; orders equal except inside runs of scores that differ by float32 noise."""
    assert a.scores.shape == b.scores.shape
    np.testing.assert_allclose(a.scores, b.scores, rtol=rtol, atol=atol)
    for q in range(a.n_queries):
        start = 0
        for j in range(1, a.width + 1):
            if j == a.width or not np.isclose(a.scores[q, j], a.scores[q, j - 1], rtol=rtol, atol=atol):
                assert set(a.indices[q, start:j]) == set(b.indices[q, start:j]), (q, start, j)
                start = j


def _good_ranking(**changes):
    fields = {
        "ids": [["a", "b"], ["b", "a"]],
        "scores": np.array([[2.0, 1.0], [3.0, 0.0]], dtype=np.float32),
        "indices": np.array([[0, 1], [1, 0]], dtype=np.int64),
    }
    fields.update(changes)
    return Ranking(**fields)


# ------------------------------------------------------------------ Ranking


class TestRanking:
    def test_valid_ranking_and_shape_properties(self):
        r = _good_ranking()
        assert (r.n_queries, r.width) == (2, 2)
        assert r.score_kind == SCORE_MAXSIM_IP and r.tie_break == "index"
        np.testing.assert_array_equal(r.lengths(), [2, 2])
        assert "n_queries=2" in repr(r) and "maxsim_ip" in repr(r)

    @pytest.mark.parametrize(
        "changes, error",
        [
            ({"scores": np.zeros((2, 2), dtype=np.float64)}, TypeError),
            ({"scores": [[2.0, 1.0], [3.0, 0.0]]}, ValueError),
            ({"scores": np.zeros(2, dtype=np.float32), "ids": [[], []], "indices": None}, ValueError),
            ({"ids": [["a", "b"]]}, ValueError),
            ({"ids": [["a", "b", "c"], ["b", "a"]]}, ValueError),
            ({"ids": [["a", 1], ["b", "a"]]}, TypeError),
            ({"indices": np.array([[0, 1], [1, 0]], dtype=np.int32)}, TypeError),
            ({"indices": [[0, 1], [1, 0]]}, TypeError),
            ({"indices": np.zeros((2, 3), dtype=np.int64)}, ValueError),
            ({"indices": np.array([[0, -1], [1, 0]], dtype=np.int64)}, ValueError),
            ({"scores": np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)}, ValueError),
            ({"scores": np.array([[2.0, np.nan], [3.0, 0.0]], dtype=np.float32)}, ValueError),
            ({"tie_break": "random"}, ValueError),
            ({"score_kind": ""}, ValueError),
        ],
    )
    def test_invalid_rankings_are_refused(self, changes, error):
        with pytest.raises(error):
            _good_ranking(**changes)

    def test_padding_is_told_apart_from_an_empty_document(self):
        padded = Ranking(
            ids=[["a"], ["b", "c"]],
            scores=np.array([[2.0, -np.inf], [1.0, -np.inf]], dtype=np.float32),
            indices=np.array([[0, -1], [1, 4]], dtype=np.int64),
        )
        np.testing.assert_array_equal(padded.lengths(), [1, 2])  # row 1: "c" is a real -inf document
        with pytest.raises(ValueError, match="padding"):
            Ranking(  # padding must be -inf and -1
                ids=[["a"]], scores=np.array([[2.0, 1.0]], dtype=np.float32),
                indices=np.array([[0, -1]], dtype=np.int64),
            )
        with pytest.raises(ValueError, match="indices"):
            Ranking(  # a listed document may not have the padding position
                ids=[["a", "b"]], scores=np.array([[2.0, 1.0]], dtype=np.float32),
                indices=np.array([[0, -1]], dtype=np.int64),
            )

    def test_ties_must_ascend_by_position_when_the_convention_is_index(self):
        tied = np.array([[1.0, 1.0]], dtype=np.float32)
        ok = Ranking(ids=[["a", "b"]], scores=tied, indices=np.array([[0, 1]], dtype=np.int64))
        assert ok.tie_break == "index"
        with pytest.raises(ValueError, match="ascending"):
            Ranking(ids=[["b", "a"]], scores=tied, indices=np.array([[1, 0]], dtype=np.int64))
        # an engine that does not promise an order, or an id-only ranking, may list ties any way
        Ranking(ids=[["b", "a"]], scores=tied, indices=np.array([[1, 0]], dtype=np.int64),
                tie_break="unspecified")
        Ranking(ids=[["b", "a"]], scores=tied, indices=None)

    def test_a_ranking_cannot_change_after_construction(self):
        scores = np.array([[2.0, 1.0]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int64)
        ids = [["a", "b"]]
        r = Ranking(ids=ids, scores=scores, indices=indices)
        scores[0, 0] = -5.0
        indices[0, 0] = 9
        ids[0][0] = "zzz"
        assert r.scores[0, 0] == 2.0 and r.indices[0, 0] == 0 and r.ids[0][0] == "a"
        with pytest.raises(ValueError):
            r.scores[0, 0] = 0.0
        with pytest.raises(ValueError):
            r.indices[0, 0] = 3
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.score_kind = "other"

    def test_equality_is_exact_and_rankings_are_unhashable(self):
        a, b = _good_ranking(), _good_ranking()
        assert a == b
        assert (a != b) is False
        assert a != _good_ranking(scores=np.array([[2.0, 1.0], [3.0, 0.5]], dtype=np.float32))
        assert a != _good_ranking(indices=None)
        assert a != _good_ranking(score_kind=SCORE_MAXSIM_COSINE)
        assert a != _good_ranking(tie_break="unspecified")
        assert a != "not a ranking"
        with pytest.raises(TypeError):
            hash(a)

    def test_top_takes_a_prefix_and_never_extends(self):
        r = _good_ranking()
        one = r.top(1)
        assert one.width == 1 and one.ids == [["a"], ["b"]]
        np.testing.assert_array_equal(one.indices, [[0], [1]])
        assert r.top(50) == r
        for bad in (0, -1):
            with pytest.raises(ValueError):
                r.top(bad)
        with pytest.raises(TypeError):
            r.top(1.5)

    def test_to_lists_drops_padding_and_keeps_infinities(self):
        r = Ranking(
            ids=[["a", "c"], ["b"]],
            scores=np.array([[2.0, -np.inf], [1.0, -np.inf]], dtype=np.float32),
            indices=np.array([[0, 3], [1, -1]], dtype=np.int64),
            tie_break="index",
        )
        out = r.to_lists()
        assert out["ids"] == [["a", "c"], ["b"]]
        assert out["scores"] == [[2.0, float("-inf")], [1.0]]
        assert out["indices"] == [[0, 3], [1]]
        assert out["score_kind"] == "maxsim_ip" and out["tie_break"] == "index"
        assert Ranking(ids=[["a"]], scores=np.zeros((1, 1), np.float32)).to_lists()["indices"] is None

    def test_empty_rankings_are_valid(self):
        none = Ranking(ids=[], scores=np.zeros((0, 3), dtype=np.float32))
        assert none.n_queries == 0 and none.lengths().size == 0
        no_docs = Ranking(ids=[[], []], scores=np.zeros((2, 0), dtype=np.float32),
                          indices=np.zeros((2, 0), dtype=np.int64))
        assert no_docs.width == 0 and no_docs.to_lists()["ids"] == [[], []]

    def test_comparable_guards_scores_across_kinds(self):
        a, b = _good_ranking(), _good_ranking()
        assert comparable(a, b)
        require_comparable(a, b)
        cosine = _good_ranking(score_kind=SCORE_MAXSIM_COSINE, tie_break="unspecified")
        assert not comparable(a, cosine)
        with pytest.raises(ValueError, match="not comparable"):
            require_comparable(a, cosine)
        assert not comparable(a, _good_ranking(tie_break="unspecified"))


# ---------------------------------------------------------------- protocols


class TestProtocols:
    def test_membership_of_the_library_retrievers(self):
        docs = _corpus()
        exact = ExactRetriever(docs)
        compressed = CompressedRetriever(Pipeline([BinaryQuantizer()]).compress(docs))
        tiered = TieredRetriever(compressed, exact, candidates=5)
        two_tier = TwoTierRetriever(compressed, exact, candidates=5)
        for r in (exact, compressed):
            assert isinstance(r, Retriever) and isinstance(r, DenseScorer) and isinstance(r, Reranker)
        assert isinstance(tiered, Retriever) and isinstance(tiered, DenseScorer)
        assert not isinstance(tiered, Reranker)
        assert isinstance(two_tier, Retriever)
        assert not isinstance(two_tier, DenseScorer) and not isinstance(two_tier, Reranker)

    def test_isinstance_checks_attribute_presence_only(self):
        class Imposter:
            n_docs = 3

            def search(self, *args):
                return "not a ranking"

        assert isinstance(Imposter(), Retriever)  # behaviour is pinned by tests, not by isinstance
        assert not isinstance(object(), Retriever)
        assert not isinstance(object(), DenseScorer)
        assert not isinstance(object(), Reranker)


# --------------------------------------------------- hand-computed rankings

T1_DOCS = [
    [[1, 0], [0, 1]],
    [[0.5, 0.5]],
    [[1, 0]],
    [],
    [[-1, 0], [0, -1]],
    [[-1, -1]],
]


class TestKnownRankings:
    def test_t1_hand_computed_scores_and_ranking(self):
        docs, q = _docs(T1_DOCS), _query_set([[[1, 0], [0, 1]]])
        retriever = ExactRetriever(docs)
        want = np.array([[2, 1, 1, -np.inf, 0, -2]], dtype=np.float32)
        np.testing.assert_array_equal(retriever.score(q), want)
        for i in range(6):  # the textbook definition agrees
            assert maxsim_pair(q[0].vectors, docs[i].vectors) == want[0, i]
        r = retriever.search(q, k=6)
        np.testing.assert_array_equal(r.indices, [[0, 1, 2, 4, 5, 3]])
        assert r.ids == [["d0", "d1", "d2", "d4", "d5", "d3"]]  # d1/d2 tie: by position
        np.testing.assert_array_equal(r.scores, [[2, 1, 1, 0, -2, -np.inf]])  # negative kept, empty last
        assert r.score_kind == "maxsim_ip" and r.tie_break == "index"

    def test_t2_copies_of_a_document_vector_do_not_change_scores(self):
        v = [[0.3, 0.9]]
        docs = _docs([v, v * 3, [[0.9, 0.3]], [[0.9, 0.3]] * 5])
        q = _query_set([[[1, 0], [0, 1]], [[0.2, 0.7]]])
        s = ExactRetriever(docs).score(q)
        np.testing.assert_array_equal(s[:, 0], s[:, 1])
        np.testing.assert_array_equal(s[:, 2], s[:, 3])

    def test_t3_inner_product_and_cosine_rank_differently(self):
        docs = _docs([[[1, 0]], [[0, 1]], [[0.7071, 0.7071]]])
        retriever = ExactRetriever(docs)
        raw = _query_set([[[10, 0], [0, 0.1]]])
        np.testing.assert_allclose(retriever.score(raw), [[10.0, 0.1, 7.1417]], rtol=1e-4)
        np.testing.assert_array_equal(retriever.search(raw, 3).indices, [[0, 2, 1]])
        unit = _query_set([[[1, 0], [0, 1]]])  # the same query with unit-norm tokens: cosine order
        np.testing.assert_allclose(retriever.score(unit), [[1.0, 1.0, 1.4142]], rtol=1e-4)
        np.testing.assert_array_equal(retriever.search(unit, 3).indices, [[2, 0, 1]])
        assert retriever.search(raw, 3).score_kind == "maxsim_ip"  # it never claims to be cosine

    def test_t4_zero_padding_a_query_is_neutral_but_padding_a_document_is_not(self):
        retriever = ExactRetriever(_docs([[[-1, 0]], [[-0.5, 0]]]))
        q = _query_set([[[1, 0]]])
        padded_q = _query_set([[[1, 0], [0, 0]]])
        np.testing.assert_array_equal(retriever.score(q), retriever.score(padded_q))
        np.testing.assert_array_equal(retriever.search(q, 2).indices, [[1, 0]])
        padded_doc = ExactRetriever(_docs([[[-1, 0], [0, 0]], [[-0.5, 0]]]))
        np.testing.assert_array_equal(padded_doc.search(q, 2).indices, [[0, 1]])  # the ranking flipped

    def test_t5_empty_queries(self):
        docs = _docs(T1_DOCS)
        retriever = ExactRetriever(docs)
        q = _query_set([[[1, 0], [0, 1]], []])  # the second query has no tokens
        s = retriever.score(q)
        np.testing.assert_array_equal(s[1], [0, 0, 0, -np.inf, 0, 0])
        np.testing.assert_array_equal(retriever.search(q, 6).indices[1], [0, 1, 2, 4, 5, 3])
        none = MultiVectorCorpus(np.zeros((0, 2), dtype=np.float32), [0, 0, 0])  # no tokens at all
        np.testing.assert_array_equal(retriever.score(none), [[0, 0, 0, -np.inf, 0, 0]] * 2)
        np.testing.assert_array_equal(retriever.search(none, 6).indices, [[0, 1, 2, 4, 5, 3]] * 2)


# ------------------------------------------------------------- equivalence


class TestEquivalence:
    @pytest.mark.parametrize("k", [1, 5, 29, 30, 100])
    def test_t6_exact_retriever_is_maxsim_matrix_plus_rank(self, k):
        docs, queries = _corpus(), _queries()
        r = ExactRetriever(docs).search(queries, k)
        full = maxsim_matrix(queries, docs)
        top = rank(full, k)
        assert r.width == min(k, len(docs))
        np.testing.assert_array_equal(r.indices, top)
        np.testing.assert_array_equal(r.scores, np.take_along_axis(full, top, axis=1))
        assert r.ids == [[docs.ids[i] for i in row] for row in top]
        assert r.scores.dtype == np.float32 and r.indices.dtype == np.int64

    def test_t6_it_is_exactly_exact_index(self):
        docs, queries = _corpus(), _queries()
        top, scores = ExactIndex(docs).search(queries, 10)
        r = ExactRetriever(docs).search(queries, 10)
        np.testing.assert_array_equal(r.indices, top)
        np.testing.assert_array_equal(r.scores, scores)
        np.testing.assert_array_equal(ExactRetriever(docs).score(queries), ExactIndex(docs).score(queries))

    def test_t6_a_list_of_arrays_is_accepted_as_queries(self):
        docs = _corpus()
        arrays = [q.vectors for q in _queries()]
        np.testing.assert_array_equal(
            ExactRetriever(docs).search(arrays, 5).indices,
            ExactRetriever(docs).search(MultiVectorCorpus.from_arrays(arrays), 5).indices,
        )

    def test_t7_compressed_float32_is_bitwise_the_exact_retriever(self):
        docs, queries = _corpus(), _queries()
        exact = ExactRetriever(docs).search(queries, 30)
        comp = CompressedRetriever(Pipeline().compress(docs)).search(queries, 30)
        assert comp == exact

    @pytest.mark.parametrize(
        "codec",
        [Float16Quantizer(), BinaryQuantizer(), Int8Quantizer("per_vector"), Int4Quantizer(),
         Lloyd2Quantizer(seed=3)],
        ids=["float16", "binary", "int8-per-vector", "int4", "lloyd2"],
    )
    def test_t7_compressed_retriever_is_exact_index_over_the_decoded_codes(self, codec):
        docs, queries = _corpus(), _queries()
        store = Pipeline([codec]).compress(docs)
        r = CompressedRetriever(store).search(queries, 12)
        top, scores = ExactIndex(store).search(queries, 12)
        np.testing.assert_array_equal(r.indices, top)
        np.testing.assert_array_equal(r.scores, scores)
        full = maxsim_matrix(queries, store)
        np.testing.assert_array_equal(r.indices, rank(full, 12))
        # exact over the stored representation: the decoded codes, not the float originals
        assert not np.array_equal(full, maxsim_matrix(queries, docs))

    def test_t7_queries_are_projected_with_the_pipelines_dimension_stages(self):
        docs, queries = _corpus(), _queries()
        pipe = Pipeline([PCAProjector(dim=8), Int8Quantizer("per_vector")]).fit(docs)
        store = pipe.compress(docs)
        retriever = CompressedRetriever(store, pipe)
        assert retriever.pipeline is pipe
        top, scores = ExactIndex(store, query_transform=pipe).search(queries, 10)
        r = retriever.search(queries, 10)
        np.testing.assert_array_equal(r.indices, top)
        np.testing.assert_array_equal(r.scores, scores)
        np.testing.assert_array_equal(
            r.indices, rank(maxsim_matrix(pipe.transform_queries(queries), store), 10)
        )
        np.testing.assert_array_equal(
            retriever.rescore(queries, top), ExactIndex(store, query_transform=pipe).rescore(queries, top)
        )
        with pytest.raises(ValueError, match="dimension"):  # unprojected queries do not fit the store
            CompressedRetriever(store).search(queries, 3)

    def test_t7_a_store_read_back_from_disk_scores_the_same(self, tmp_path):
        docs, queries = _corpus(), _queries()
        store = Pipeline([Int8Quantizer("per_vector")]).compress(docs)
        save_compressed(store, tmp_path / "cold.npz")
        again = CompressedRetriever(load_compressed(tmp_path / "cold.npz"))
        assert again.search(queries, 8) == CompressedRetriever(store).search(queries, 8)

    @pytest.mark.parametrize("budget", [64, 4096])
    def test_t8_the_block_budget_changes_bits_not_the_ranking(self, budget):
        docs, queries = _corpus(), _queries()
        reference = ExactRetriever(docs).search(queries, 30)
        small = ExactRetriever(docs, max_block_bytes=budget).search(queries, 30)
        assert_close_modulo_ties(small, reference)

    @pytest.mark.parametrize("tokens", [14, 20, 30])
    def test_t9_one_query_alone_or_in_a_batch_is_bitwise_equal_on_this_stack(self, tokens):
        # MEASURED-HERE in the audit and re-checked on this stack; BLAS kernel choice for other
        # shapes or machines can change the last bit, which is why only the ranking is promised.
        docs = _corpus()
        batch = _queries(n=5, tokens=tokens)
        retriever = ExactRetriever(docs)
        together = retriever.score(batch)
        for i in range(len(batch)):
            alone = retriever.score(MultiVectorCorpus.from_arrays([batch[i].vectors]))
            np.testing.assert_array_equal(alone[0], together[i])

    @pytest.mark.parametrize("tokens", [1, 2, 3])
    def test_t9_very_short_queries_agree_to_rounding(self, tokens):
        docs = _corpus()
        batch = _queries(n=5, tokens=tokens)
        retriever = ExactRetriever(docs)
        together = retriever.search(batch, 30)
        for i in range(len(batch)):
            alone = retriever.search(MultiVectorCorpus.from_arrays([batch[i].vectors]), 30)
            np.testing.assert_allclose(alone.scores[0], together.scores[i], rtol=1e-5, atol=1e-5)

    def test_ranking_from_scores_is_rank_plus_take_along_axis(self):
        scores = maxsim_matrix(_queries(), _corpus())
        r = ranking_from_scores(scores, [f"x{i}" for i in range(30)], 4, SCORE_MAXSIM_COSINE)
        np.testing.assert_array_equal(r.indices, rank(scores, 4))
        assert r.score_kind == SCORE_MAXSIM_COSINE and r.ids[0][0] == f"x{r.indices[0, 0]}"
        with pytest.raises(ValueError):
            ranking_from_scores(scores, ["a"], 4, SCORE_MAXSIM_IP)
        with pytest.raises(TypeError):
            ranking_from_scores(scores.astype(np.float64), ["a"] * 30, 4, SCORE_MAXSIM_IP)
        with pytest.raises(ValueError):
            ranking_from_scores(scores, ["a"] * 30, 0, SCORE_MAXSIM_IP)

    def test_ranking_from_scores_breaks_ties_by_position(self):
        tied = np.zeros((3, 200), dtype=np.float32)
        tied[1, 7] = 1.0  # one document ahead of 199 tied ones
        r = ranking_from_scores(tied, [str(i) for i in range(200)], 10, SCORE_MAXSIM_IP)
        np.testing.assert_array_equal(r.indices[0], np.arange(10))
        np.testing.assert_array_equal(r.indices[1], [7, 0, 1, 2, 3, 4, 5, 6, 8, 9])


def _random_stores(seed):
    """A random corpus and query set, with empty documents and empty queries mixed in."""
    rng = np.random.default_rng(1000 + seed)
    dim = int(rng.integers(2, 20))
    n_docs, n_queries = int(rng.integers(1, 26)), int(rng.integers(1, 9))
    counts = rng.integers(0, 9, size=n_docs)
    q_counts = rng.integers(0, 6, size=n_queries)
    docs = MultiVectorCorpus.from_arrays(
        [_unit(rng.standard_normal((max(c, 1), dim)))[:c] for c in counts],
        ids=[f"r{i}" for i in range(n_docs)])
    queries = MultiVectorCorpus.from_arrays(
        [_unit(rng.standard_normal((max(c, 1), dim)))[:c] for c in q_counts])
    return docs, queries, rng


class TestRandomCorpora:
    @pytest.mark.parametrize("seed", range(12))
    def test_exact_and_compressed_retrievers_are_the_scorer_on_random_inputs(self, seed):
        docs, queries, rng = _random_stores(seed)
        full = maxsim_matrix(queries, docs)
        for k in (1, int(rng.integers(1, len(docs) + 3)), len(docs) + 5):
            r = ExactRetriever(docs).search(queries, k)
            top = rank(full, k)
            np.testing.assert_array_equal(r.indices, top)
            np.testing.assert_array_equal(r.scores, np.take_along_axis(full, top, axis=1))
            assert r.width == min(k, len(docs))
        empties = np.flatnonzero(docs.counts == 0)
        if empties.size and docs.num_vectors:
            assert np.isneginf(full[:, empties]).all()
            store = Pipeline([BinaryQuantizer()]).compress(docs)
            both = CompressedRetriever(store).search(queries, len(docs))
            np.testing.assert_array_equal(both.indices, rank(maxsim_matrix(queries, store)))

    @pytest.mark.parametrize("seed", range(12))
    def test_rescore_is_the_scorer_on_random_candidates(self, seed):
        docs, queries, rng = _random_stores(seed)
        retriever = ExactRetriever(docs)
        m = int(rng.integers(0, 2 * len(docs) + 1))
        candidates = rng.integers(0, len(docs), size=(len(queries), m))  # duplicates allowed
        got = retriever.rescore(queries, candidates)
        np.testing.assert_array_equal(got, ExactIndex(docs).rescore(queries, candidates))
        want = np.take_along_axis(maxsim_matrix(queries, docs), candidates, axis=1)
        # ExactIndex.rescore gives a query without tokens 0 against a document without vectors,
        # where maxsim_matrix gives -inf (reported, not changed): that pair is compared apart.
        corner = (queries.counts == 0)[:, None] & (docs.counts[candidates] == 0)
        np.testing.assert_allclose(got[~corner], want[~corner], rtol=1e-5, atol=1e-6)
        assert np.isneginf(want[corner]).all()
        assert np.isin(got[corner], [0.0, -np.inf]).all()


# ------------------------------------------------- ties, empties, determinism


def _identical_docs(n=50):
    v = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    return MultiVectorCorpus.from_arrays([v] * n, ids=[f"t{i}" for i in range(n)])


def _tie_queries():
    return MultiVectorCorpus.from_arrays([np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)] * 3)


class TestTiesEmptiesDeterminism:
    def test_t16_fifty_identical_documents_rank_by_position(self):
        docs, q = _identical_docs(), _tie_queries()
        want = np.tile(np.arange(10), (3, 1))
        floats = ExactRetriever(docs)
        binary = CompressedRetriever(Pipeline([BinaryQuantizer()]).compress(docs))
        for retriever in (floats, binary):
            np.testing.assert_array_equal(retriever.search(q, 10).indices, want)
        tiered = TieredRetriever(binary, floats, candidates=5)
        np.testing.assert_array_equal(tiered.search(q, 10).indices, want)
        np.testing.assert_array_equal(TwoTierRetriever(binary, floats, candidates=5).search(q, 5).indices,
                                      want[:, :5])

    def test_t17_empty_documents_are_minus_infinity_and_last(self):
        rng = np.random.default_rng(7)
        arrays = [_unit(rng.standard_normal((4, 8))) for _ in range(10)]
        for empty in (2, 6):
            arrays[empty] = np.zeros((0, 8), dtype=np.float32)
        docs = MultiVectorCorpus.from_arrays(arrays, ids=[f"d{i}" for i in range(10)])
        q = _queries(n=4, tokens=3, dim=8)
        floats = ExactRetriever(docs)
        binary = CompressedRetriever(Pipeline([BinaryQuantizer()]).compress(docs))
        int8 = CompressedRetriever(Pipeline([Int8Quantizer("per_vector")]).compress(docs))
        tiered = TieredRetriever(binary, int8, candidates=10)
        for retriever in (floats, binary, int8, tiered):
            scores = retriever.score(q)
            assert np.isneginf(scores[:, [2, 6]]).all()
            assert np.isfinite(np.delete(scores, [2, 6], axis=1)).all()
            r = retriever.search(q, 10)
            np.testing.assert_array_equal(r.indices[:, -2:], [[2, 6]] * 4)
            assert np.isneginf(r.scores[:, -2:]).all() and r.lengths().tolist() == [10] * 4
        two = TwoTierRetriever(binary, int8, candidates=10).search(q, 10)
        np.testing.assert_array_equal(two.indices[:, -2:], [[2, 6]] * 4)
        assert two.scores[:, -2:].max() == -np.inf

    def test_t17_a_corpus_that_is_all_empty_ranks_by_position(self):
        docs = MultiVectorCorpus(np.zeros((0, 4), dtype=np.float32), [0, 0, 0, 0])
        r = ExactRetriever(docs).search(_queries(n=2, dim=4), 3)
        np.testing.assert_array_equal(r.indices, [[0, 1, 2]] * 2)
        assert np.isneginf(r.scores).all()

    def test_t18_repeating_a_call_gives_an_identical_ranking(self):
        docs, queries = _corpus(), _queries()
        store = Pipeline([BinaryQuantizer()]).compress(docs)
        for retriever in (
            ExactRetriever(docs),
            CompressedRetriever(store),
            TieredRetriever(CompressedRetriever(store), ExactRetriever(docs), candidates=6),
            TwoTierRetriever(CompressedRetriever(store), ExactRetriever(docs), candidates=6),
        ):
            k = 6
            assert retriever.search(queries, k) == retriever.search(queries, k)

    def test_t18_permuting_the_documents_permutes_the_ranking(self):
        docs, queries = _corpus(), _queries()
        perm = np.random.default_rng(5).permutation(len(docs))
        a = ExactRetriever(docs).search(queries, 30)
        b = ExactRetriever(docs.select(perm)).search(queries, 30)
        assert a.ids == b.ids  # the same documents in the same order, whatever their positions
        np.testing.assert_array_equal(perm[b.indices], a.indices)

    def test_t18_an_orthogonal_rotation_leaves_scores_and_ranking_alone(self):
        docs, queries = _corpus(), _queries()
        basis, _ = np.linalg.qr(np.random.default_rng(9).standard_normal((16, 16)))
        basis = basis.astype(np.float32)
        rotated_docs = docs.with_vectors(docs.vectors @ basis)
        rotated_queries = queries.with_vectors(queries.vectors @ basis)
        a = ExactRetriever(docs).search(queries, 30)
        b = ExactRetriever(rotated_docs).search(rotated_queries, 30)
        np.testing.assert_allclose(a.scores, b.scores, rtol=1e-4, atol=1e-5)
        assert_close_modulo_ties(a, b, rtol=1e-4, atol=1e-4)

    def test_t18_permuting_rows_inside_documents_and_query_tokens_leaves_scores_alone(self):
        rng = np.random.default_rng(11)
        docs, queries = _corpus(), _queries()
        shuffled = MultiVectorCorpus.from_arrays(
            [d.vectors[rng.permutation(d.num_vectors)] for d in docs], ids=docs.ids)
        shuffled_q = MultiVectorCorpus.from_arrays(
            [q.vectors[rng.permutation(q.num_vectors)] for q in queries])
        np.testing.assert_allclose(
            ExactRetriever(shuffled).score(shuffled_q), ExactRetriever(docs).score(queries),
            rtol=1e-5, atol=1e-5)

    def test_t18_scaling_every_query_keeps_the_ranking(self):
        docs, queries = _corpus(), _queries()
        base = ExactRetriever(docs).search(queries, 30)
        doubled = ExactRetriever(docs).search(queries.with_vectors(queries.vectors * 2.0), 30)
        np.testing.assert_array_equal(doubled.indices, base.indices)  # x2 is exact in float32
        np.testing.assert_array_equal(doubled.scores, base.scores * 2.0)
        scaled = ExactRetriever(docs).search(queries.with_vectors(queries.vectors * 3.7), 30)
        assert_close_modulo_ties(scaled, Ranking(
            ids=base.ids, scores=(base.scores * np.float32(3.7)).astype(np.float32),
            indices=base.indices), rtol=1e-4, atol=1e-4)


# ------------------------------------------------------ contract and checks


class TestContract:
    def test_t20_rescore_keeps_column_order_and_duplicates(self):
        docs, queries = _corpus(), _queries()
        retriever = ExactRetriever(docs)
        cand = np.tile(np.array([3, 3, 0, 7]), (len(queries), 1))
        got = retriever.rescore(queries, cand)
        assert got.dtype == np.float32 and got.shape == cand.shape
        np.testing.assert_array_equal(got[:, 0], got[:, 1])
        np.testing.assert_array_equal(got, ExactIndex(docs).rescore(queries, cand))
        full = maxsim_matrix(queries, docs)
        np.testing.assert_allclose(got, np.take_along_axis(full, cand, axis=1), rtol=1e-5)
        assert retriever.rescore(queries, np.zeros((len(queries), 0), dtype=np.int64)).shape == (7, 0)

    @pytest.mark.parametrize(
        "candidates, error",
        [
            (np.array([[-1, 0]] * 7), IndexError),
            (np.array([[0, 30]] * 7), IndexError),
            (np.array([[0.0, 1.0]] * 7), TypeError),
            (np.array([[True, False]] * 7), TypeError),
            (np.array([0, 1]), ValueError),
            (np.array([[0, 1]] * 6), ValueError),
        ],
        ids=["minus-one", "past-the-end", "floats", "booleans", "one-dimensional", "wrong-row-count"],
    )
    def test_t20_rescore_rejects_bad_candidates(self, candidates, error):
        with pytest.raises(error):
            ExactRetriever(_corpus()).rescore(_queries(), candidates)

    def test_the_position_error_names_the_problem_not_a_zero_division(self):
        with pytest.raises(IndexError, match="padding marker"):
            ExactRetriever(_corpus()).rescore(_queries(), np.full((7, 2), -1))

    @pytest.mark.parametrize("k, error", [(0, ValueError), (-1, ValueError), (1.5, TypeError),
                                          (True, TypeError), ("3", TypeError), (None, TypeError)])
    def test_t20_k_must_be_a_positive_integer(self, k, error):
        docs = _corpus()
        for retriever in (ExactRetriever(docs), CompressedRetriever(Pipeline().compress(docs))):
            with pytest.raises(error):
                retriever.search(_queries(), k)
        with pytest.raises(error):
            check_k(k)

    def test_k_may_be_a_numpy_integer(self):
        assert ExactRetriever(_corpus()).search(_queries(), np.int64(3)).width == 3

    def test_queries_with_nan_or_inf_are_refused_not_ranked_in_index_order(self):
        docs, queries = _corpus(), _queries()
        for bad in (np.nan, np.inf, -np.inf):
            vectors = np.array(queries.vectors)
            vectors[5, 3] = bad
            poisoned = queries.with_vectors(vectors)
            with pytest.raises(ValueError, match="non-finite"):
                ExactRetriever(docs).search(poisoned, 5)
            with pytest.raises(ValueError, match="non-finite"):
                as_queries(poisoned)

    def test_queries_of_the_wrong_dimension_or_type_are_refused(self):
        retriever = ExactRetriever(_corpus())
        with pytest.raises(ValueError, match="dimension 8 but the store holds 16"):
            retriever.search(_queries(dim=8), 3)
        with pytest.raises(TypeError):
            retriever.search(np.zeros((4, 16), dtype=np.float32), 3)  # a bare array is not a query set
        with pytest.raises(TypeError):
            retriever.search([np.zeros((4, 16), dtype=np.int64)], 3)  # integers are not narrowed

    def test_construction_is_validated(self):
        docs = _corpus()
        with pytest.raises(TypeError, match="vector store"):
            ExactRetriever(np.zeros((3, 3)))
        with pytest.raises(ValueError):
            ExactRetriever(docs, max_block_bytes=0)
        with pytest.raises(TypeError):
            ExactRetriever(docs, max_block_bytes=1.5)
        with pytest.raises(TypeError):
            CompressedRetriever(Pipeline().compress(docs), pipeline="pca")

    def test_an_empty_corpus_and_an_empty_query_batch_give_empty_rankings(self):
        nothing = MultiVectorCorpus(np.zeros((0, 16), dtype=np.float32), [0])
        r = ExactRetriever(nothing).search(_queries(), 5)
        assert r.width == 0 and r.ids == [[]] * 7 and ExactRetriever(nothing).n_docs == 0
        no_queries = MultiVectorCorpus(np.zeros((0, 16), dtype=np.float32), [0])
        r = ExactRetriever(_corpus()).search(no_queries, 5)
        assert r.n_queries == 0 and r.width == 5

    def test_ids_default_to_positions_for_a_store_without_ids(self):
        class Bare:
            offsets = np.array([0, 1, 2])

            def __len__(self):
                return 2

            def decode_rows(self, lo, hi):
                return np.eye(2, dtype=np.float32)[lo:hi]

        r = ExactRetriever(Bare()).search(_query_set([[[0, 1]]]), 2)
        assert r.ids == [["1", "0"]]

    def test_ids_must_be_strings_and_match_the_documents(self):
        docs = _corpus()

        class Numbered:
            offsets, decode_rows = docs.offsets, docs.decode_rows

            def __init__(self, ids):
                self.ids = ids

            def __len__(self):
                return 30

        with pytest.raises(TypeError, match="ids must be strings"):
            ExactRetriever(Numbered(list(range(30))))
        with pytest.raises(ValueError, match="30 documents but 1 ids"):
            ExactRetriever(Numbered(["a"]))
        assert ExactRetriever(Numbered([f"s{i}" for i in range(30)])).n_docs == 30

    def test_retrievers_expose_what_they_wrap(self):
        docs = _corpus()
        r = ExactRetriever(docs, max_block_bytes=4096)
        assert r.n_docs == 30 and r.ids == docs.ids and r.ids is not docs.ids
        assert r.store is docs and r.pipeline is None and r.index.max_block_bytes == 4096
        assert r.score_kind == SCORE_MAXSIM_IP


# ------------------------------------------------- source-level properties

_ALLOWED_FROM_OPTIVISION = {
    "optivision.scoring", "optivision.compose", "optivision.representation",
}


def _imports(path: Path):
    package = "optivision.core.retrieve" if path.parent.name == "retrieve" else "optivision.backends"
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, ()
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            yield module, tuple(a.name for a in node.names)


class TestSourceProperties:
    def test_core_retrieve_imports_only_what_the_brief_allows(self):
        import sys

        files = sorted((SRC / "core" / "retrieve").glob("*.py"))
        assert {f.name for f in files} >= {"__init__.py", "types.py", "exact.py", "tiered.py", "checks.py"}
        for path in files:
            for module, names in _imports(path):
                top = module.split(".")[0]
                if top in sys.stdlib_module_names or top in {"__future__", "numpy"}:
                    continue
                assert top == "optivision", f"{path.name}: third-party import {module}"
                if module == "optivision.core" or module.startswith("optivision.core."):
                    continue
                if module == "optivision.storage":
                    assert set(names) <= {"ExactIndex", "TieredIndex"}, f"{path.name}: {names}"
                    continue
                assert module in _ALLOWED_FROM_OPTIVISION, f"{path.name} imports {module}"

    def test_backends_import_only_index_types_and_the_seam(self):
        import sys

        files = sorted((SRC / "backends").glob("*.py"))
        assert {f.name for f in files} >= {"__init__.py", "base.py", "numpy_backend.py",
                                           "qdrant_backend.py", "factory.py"}
        for path in files:
            for module, _ in _imports(path):
                top = module.split(".")[0]
                if top in sys.stdlib_module_names or top in {"__future__", "numpy"}:
                    continue
                assert top == "optivision", f"{path.name}: third-party import {module}"
                assert (
                    module in {"optivision.core.retrieve", "optivision.index", "optivision.types",
                               "optivision.backends"}
                    or module.startswith(("optivision.index.", "optivision.backends."))
                ), f"{path.name} imports {module}"

    def test_no_new_code_reads_the_environment_or_names_a_device(self):
        for path in [*(SRC / "core" / "retrieve").glob("*.py"), *(SRC / "backends").glob("*.py")]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert not {"os", "torch"} & {a.name.split(".")[0] for a in node.names}, path.name
                if isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] not in {"os", "torch"}, path.name
                if isinstance(node, ast.Attribute):
                    assert node.attr not in {"environ", "getenv", "cuda", "to_device"}, path.name

    def test_wording_rules_hold_in_the_new_sources(self):
        banned = ["universal compression", "works with every model", "guaranteed", "state of the art",
                  "state-of-the-art", "novel", "better than", "lossless", "faster", "scales to"]
        for path in [*(SRC / "core" / "retrieve").glob("*.py"), *(SRC / "backends").glob("*.py")]:
            text = path.read_text(encoding="utf-8").lower()
            for phrase in banned:
                assert phrase not in text, f"{path.name}: {phrase!r}"


# --------------------------------------------- documents and query widths (D1, D2)


def _poison(corpus, value, docs=(4,), dtype=None):
    v = np.array(corpus.vectors if dtype is None else corpus.vectors.astype(dtype))
    for d in docs:
        v[corpus.offsets[d] + 1, 2] = value
    return MultiVectorCorpus(v, corpus.offsets, ids=corpus.ids)


class TestDocumentsAreScannedOnce:
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("make", [ExactRetriever, CompressedRetriever])
    def test_a_float_store_with_nan_or_infinity_is_refused_naming_the_first_document(
        self, make, value
    ):
        docs = _corpus()
        bad = _poison(docs, value, docs=(9, 4, 20))
        with pytest.raises(ValueError, match=r"non-finite.*document 4 \('doc4::p1'\)"):
            make(bad)

    def test_the_scan_runs_in_blocks_in_the_stored_dtype(self, monkeypatch):
        from optivision.core.retrieve import checks

        docs = _corpus()
        half = MultiVectorCorpus(docs.vectors.astype(np.float16), docs.offsets, ids=docs.ids)
        seen = []
        real = np.isfinite
        monkeypatch.setattr(
            checks.np, "isfinite", lambda a, *x, **k: (seen.append((a.dtype, a.shape[0])), real(a))[1]
        )
        monkeypatch.setattr(checks, "_FINITE_BLOCK_ROWS", 50)
        retriever = ExactRetriever(half)
        assert seen and {dt for dt, _ in seen} == {np.dtype(np.float16)}
        assert max(n for _, n in seen) <= 50 and sum(n for _, n in seen) == half.num_vectors
        calls = len(seen)
        retriever.search(_queries(), 3)
        retriever.score(_queries())
        assert len(seen) > calls  # the queries are scanned, once per call
        with pytest.raises(ValueError, match="non-finite"):
            ExactRetriever(_poison(docs, np.nan, dtype=np.float16))

    def test_the_documents_are_not_scanned_again_by_a_search(self, monkeypatch):
        from optivision.core.retrieve import checks

        retriever = ExactRetriever(_corpus())
        monkeypatch.setattr(checks, "check_finite_store", lambda *a: pytest.fail("scanned"))
        monkeypatch.setattr(
            "optivision.core.retrieve.exact.check_finite_store", lambda *a: pytest.fail("scanned")
        )
        retriever.search(_queries(), 3)
        retriever.score(_queries())
        retriever.rescore(_queries(), np.zeros((7, 2), dtype=np.int64))

    def test_a_store_of_integer_codes_is_not_scanned(self, monkeypatch):
        docs = _corpus()
        store = Pipeline([BinaryQuantizer()]).compress(docs)
        monkeypatch.setattr(
            "optivision.core.retrieve.checks.first_non_finite_row", lambda v: pytest.fail("scanned")
        )
        assert CompressedRetriever(store).n_docs == 30
        assert CompressedRetriever(store, Pipeline([BinaryQuantizer()])).n_docs == 30

    def test_a_clean_or_empty_store_and_a_store_without_vectors_pass(self):
        assert ExactRetriever(_corpus()).n_docs == 30
        empty = MultiVectorCorpus(np.zeros((0, 16), dtype=np.float32), [0])
        assert ExactRetriever(empty).n_docs == 0


class TestQueryWidthThroughAPipeline:
    @staticmethod
    def _projected(stages):
        docs = _corpus()
        pipeline = Pipeline(stages)
        return docs, pipeline, pipeline.compress(docs)

    @pytest.mark.parametrize("wrong", [8, 20])
    def test_a_fitted_projection_names_the_queries_and_both_dimensions(self, wrong):
        _docs, pipeline, store = self._projected([PCAProjector(6), BinaryQuantizer()])
        retriever = CompressedRetriever(store, pipeline)
        assert retriever.search(_queries(), 3).width == 3
        with pytest.raises(ValueError, match=rf"queries have dimension {wrong} .*fitted on 16"):
            retriever.search(_queries(dim=wrong), 3)
        with pytest.raises(ValueError, match="queries have dimension"):
            retriever.score(_queries(dim=wrong))

    def test_a_pipeline_without_a_projection_compares_with_the_store(self):
        _docs, pipeline, store = self._projected([BinaryQuantizer()])
        with pytest.raises(ValueError, match=r"dimension 8 .*16-dimensional"):
            CompressedRetriever(store, pipeline).search(_queries(dim=8), 3)

    def test_a_truncation_gives_a_projected_width_that_is_compared_with_the_store(self):
        from optivision.stages import TruncateProjector

        _docs, pipeline, store = self._projected([TruncateProjector(12), BinaryQuantizer()])
        retriever = CompressedRetriever(store, pipeline)
        assert retriever.search(_queries(), 3).width == 3
        with pytest.raises(ValueError, match=r"dimension 8 .*projects them to dimension 8.*12"):
            retriever.search(_queries(dim=8), 3)

    def test_a_projection_that_keeps_no_known_array_gets_its_shape_error_re_raised(self):
        from optivision.stages.base import DimensionReducer

        class Opaque(DimensionReducer):
            name = "test_opaque_projection"

            def params(self):
                return {}

            @property
            def fitted(self):
                return True

            def output_dim(self, d):
                return 4

            def project(self, vectors):
                return np.asarray(vectors, np.float32) @ np.ones((16, 4), np.float32)

        docs = _corpus()
        pipeline = Pipeline([Opaque()])
        retriever = CompressedRetriever(pipeline.compress(docs), pipeline)
        assert retriever.search(_queries(), 3).width == 3
        with pytest.raises(ValueError, match=r"queries have dimension 8, which the pipeline cannot"):
            retriever.search(_queries(dim=8), 3)

    def test_the_two_tier_and_tiered_paths_report_it_too(self):
        docs, pipeline, store = self._projected([PCAProjector(6), BinaryQuantizer()])
        first = CompressedRetriever(store, pipeline)
        for retriever in (
            TwoTierRetriever(first, ExactRetriever(docs), candidates=5),
            TieredRetriever(first, ExactRetriever(docs), candidates=5),
        ):
            with pytest.raises(ValueError, match=r"queries have dimension 8 .*fitted on 16"):
                retriever.search(_queries(dim=8), 3)

    def test_the_check_lets_a_query_set_without_vectors_pass(self):
        from optivision.core.retrieve.checks import check_projection

        _docs, pipeline, _store = self._projected([PCAProjector(6), BinaryQuantizer()])
        none = MultiVectorCorpus(np.zeros((0, 3), dtype=np.float32), [0])
        check_projection(none, pipeline, 6)
