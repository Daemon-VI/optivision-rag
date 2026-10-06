"""Two-tier retrieval: TieredRetriever (delegates to TieredIndex) and TwoTierRetriever.

Cases T10-T13 of docs/package/RETRIEVAL_BACKEND_AUDIT.md section 7, plus the tie-heavy,
padding, id-only and validation cases of the shortlist/rerank seam. Small seeded synthetic data.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.core.retrieve import (
    CompressedRetriever,
    ExactRetriever,
    Ranking,
    TieredRetriever,
    TwoTierRetriever,
)
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, rank
from optivision.stages import BinaryQuantizer, Int8Quantizer, PCAProjector
from optivision.storage import ExactIndex, TieredIndex


def _unit(v):
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _corpus(seed=0, n_docs=30, dim=16):
    rng = np.random.default_rng(seed)
    arrays = [_unit(rng.standard_normal((int(rng.integers(3, 12)), dim))) for _ in range(n_docs)]
    return MultiVectorCorpus.from_arrays(arrays, ids=[f"doc{i}::p1" for i in range(n_docs)])


def _queries(seed=1, n=7, tokens=4, dim=16):
    rng = np.random.default_rng(seed)
    return MultiVectorCorpus.from_arrays([_unit(rng.standard_normal((tokens, dim))) for _ in range(n)])


def _docs(rows_per_doc, dim=2, ids=None):
    arrays = [np.asarray(r, dtype=np.float32).reshape(-1, dim) for r in rows_per_doc]
    return MultiVectorCorpus.from_arrays(arrays, ids=ids or [f"d{i}" for i in range(len(arrays))])


def assert_close_modulo_ties(a: Ranking, b: Ranking, rtol=1e-5, atol=1e-5):
    assert a.scores.shape == b.scores.shape
    np.testing.assert_allclose(a.scores, b.scores, rtol=rtol, atol=atol)
    for q in range(a.n_queries):
        start = 0
        for j in range(1, a.width + 1):
            if j == a.width or not np.isclose(a.scores[q, j], a.scores[q, j - 1], rtol=rtol, atol=atol):
                assert set(a.indices[q, start:j]) == set(b.indices[q, start:j]), (q, start, j)
                start = j


@pytest.fixture(scope="module")
def stores():
    docs = _corpus()
    return {
        "docs": docs,
        "queries": _queries(),
        "hot": Pipeline([BinaryQuantizer()]).compress(docs),
        "cold": Pipeline([Int8Quantizer("per_vector")]).compress(docs),
    }


@pytest.fixture
def tiers(stores):
    return CompressedRetriever(stores["hot"]), CompressedRetriever(stores["cold"])


class _IdOnly:
    """A first stage that knows ids only, as a backend behind an engine does."""

    def __init__(self, inner):
        self.inner = inner

    @property
    def n_docs(self):
        return self.inner.n_docs

    def search(self, queries, k=10):
        return dataclasses.replace(self.inner.search(queries, k), indices=None)


class _Fixed:
    """A first stage that hands back a prepared ranking whatever it is asked."""

    def __init__(self, ranking, n_docs):
        self.ranking, self.n_docs = ranking, n_docs

    def search(self, queries, k=10):
        return self.ranking


class _Spy:
    """A reranker that records the candidates it is given."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    @property
    def ids(self):
        return self.inner.ids

    @property
    def n_docs(self):
        return self.inner.n_docs

    def rescore(self, queries, candidates):
        self.calls.append(np.array(candidates))
        return self.inner.rescore(queries, candidates)


# ----------------------------------------------------------- TieredRetriever


class TestTieredRetriever:
    @pytest.mark.parametrize("m", [1, 5, 12, 30, 100])
    def test_search_and_score_are_tiered_index_exactly(self, stores, tiers, m):
        hot, cold = tiers
        queries = stores["queries"]
        index = TieredIndex(ExactIndex(stores["hot"]), ExactIndex(stores["cold"]), candidates=m)
        retriever = TieredRetriever(hot, cold, candidates=m)
        np.testing.assert_array_equal(retriever.score(queries), index.score(queries))
        for k in (1, min(m, 30), 30, 50):
            top, scores = index.search(queries, k)
            r = retriever.search(queries, k)
            np.testing.assert_array_equal(r.indices, top)
            np.testing.assert_array_equal(r.scores, scores)
            assert r.ids == [[hot.ids[i] for i in row] for row in top]
            assert r.score_kind == "tiered" and r.tie_break == "index"

    @pytest.mark.parametrize("m", [3, 10])
    def test_final_order_is_shortlist_by_cold_score_then_the_rest_by_hot(self, stores, tiers, m):
        # Invariant I5, re-derived without TieredIndex.score.
        hot, cold = tiers
        q = stores["queries"]
        hot_scores = hot.score(q)
        shortlist = rank(hot_scores, m)
        cold_scores = cold.rescore(q, shortlist)
        got = TieredRetriever(hot, cold, candidates=m).search(q, 30).indices
        for row in range(len(q)):
            members = shortlist[row]
            by_cold = sorted(members, key=lambda d: (-cold_scores[row, list(members).index(d)], d))
            rest = [d for d in rank(hot_scores[row], 30) if d not in set(members)]
            assert got[row].tolist() == [int(d) for d in by_cold] + [int(d) for d in rest]

    def test_t10_a_shortlist_of_the_whole_corpus_is_the_cold_ranking(self, stores, tiers):
        hot, cold = tiers
        reference = ExactRetriever(stores["cold"]).search(stores["queries"], 30)
        for m in (30, 1000):
            r = TieredRetriever(hot, cold, candidates=m).search(stores["queries"], 30)
            assert_close_modulo_ties(r, dataclasses.replace(reference, score_kind="tiered"))

    def test_t10_one_candidate_leaves_the_hot_ranking(self, stores, tiers):
        hot, cold = tiers
        q = stores["queries"]
        r = TieredRetriever(hot, cold, candidates=1).search(q, 30)
        np.testing.assert_array_equal(r.indices, rank(hot.score(q)))
        np.testing.assert_array_equal(r.indices[:, 0], hot.search(q, 1).indices[:, 0])

    def test_t12_with_the_same_store_in_both_tiers_the_top_ranks_match_the_single_tier(self, stores):
        docs, q = stores["docs"], stores["queries"]
        one = ExactRetriever(docs)
        tiered = TieredRetriever(ExactRetriever(docs), ExactRetriever(docs), candidates=8)
        a = dataclasses.replace(tiered.search(q, 8), score_kind="maxsim_ip")
        assert_close_modulo_ties(a, one.search(q, 8))

    def test_t13_misaligned_tiers_raise_at_construction(self, stores):
        docs, hot_store = stores["docs"], stores["hot"]
        hot = CompressedRetriever(hot_store)
        backwards = ExactRetriever(docs.select(np.arange(len(docs))[::-1]))
        with pytest.raises(ValueError, match="same order.*position 0"):
            TieredRetriever(hot, backwards)
        with pytest.raises(ValueError, match="same documents: 30 vs 29"):
            TieredRetriever(hot, ExactRetriever(docs.select(np.arange(29))))
        renamed = MultiVectorCorpus(docs.vectors, docs.offsets, ids=[f"x{i}" for i in range(len(docs))])
        with pytest.raises(ValueError, match="'doc0::p1'.*'x0'"):
            TieredRetriever(hot, ExactRetriever(renamed))
        TieredRetriever(hot, ExactRetriever(docs))  # the aligned pair is accepted

    @pytest.mark.parametrize("candidates, error", [(0, ValueError), (-3, ValueError), (2.5, TypeError),
                                                   (True, TypeError), ("5", TypeError), (None, TypeError)])
    def test_candidates_must_be_a_positive_integer(self, tiers, candidates, error):
        with pytest.raises(error):
            TieredRetriever(*tiers, candidates=candidates)

    def test_tiers_must_be_retrievers_and_non_empty(self, stores, tiers):
        hot, cold = tiers
        with pytest.raises(TypeError, match="hot must be"):
            TieredRetriever(stores["hot"], cold)
        with pytest.raises(TypeError, match="cold must be"):
            TieredRetriever(hot, ExactIndex(stores["cold"]))
        nothing = MultiVectorCorpus(np.zeros((0, 16), dtype=np.float32), [0])
        with pytest.raises(ValueError, match="no documents"):
            TieredRetriever(ExactRetriever(nothing), ExactRetriever(nothing))

    def test_k_is_validated_and_clamped_like_every_other_retriever(self, stores, tiers):
        r = TieredRetriever(*tiers, candidates=5)
        for bad in (0, -1):
            with pytest.raises(ValueError):
                r.search(stores["queries"], bad)
        with pytest.raises(TypeError):
            r.search(stores["queries"], 2.0)
        assert r.search(stores["queries"], 500).width == 30
        assert r.shortlist_size == 5 and TieredRetriever(*tiers, candidates=99).shortlist_size == 30
        assert r.n_docs == 30 and r.ids == tiers[0].ids

    def test_last_timing_is_a_copy_of_the_tiered_index_state(self, stores, tiers):
        r = TieredRetriever(*tiers, candidates=5)
        assert r.last_timing == {}
        r.search(stores["queries"], 3)
        timing = r.last_timing
        assert set(timing) == {"hot_seconds", "rescore_seconds"}
        timing["hot_seconds"] = -1.0
        assert r.last_timing["hot_seconds"] >= 0.0

    def test_each_tier_projects_the_raw_queries_with_its_own_pipeline(self, stores):
        docs, q = stores["docs"], stores["queries"]
        pipe = Pipeline([PCAProjector(dim=8), BinaryQuantizer()]).fit(docs)
        hot_store = pipe.compress(docs)
        hot = CompressedRetriever(hot_store, pipe)  # 8-d tier, raw 16-d queries
        cold = ExactRetriever(docs)  # 16-d tier
        index = TieredIndex(ExactIndex(hot_store, query_transform=pipe), ExactIndex(docs), 6)
        top, scores = index.search(q, 10)
        r = TieredRetriever(hot, cold, candidates=6).search(q, 10)
        np.testing.assert_array_equal(r.indices, top)
        np.testing.assert_array_equal(r.scores, scores)

    def test_queries_are_checked_once_for_both_tiers(self, stores, tiers):
        r = TieredRetriever(*tiers, candidates=5)
        bad = np.array(stores["queries"].vectors)
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            r.search(stores["queries"].with_vectors(bad), 3)
        with pytest.raises(ValueError, match="dimension"):
            r.search(_queries(dim=8), 3)


# ----------------------------------------------------------- TwoTierRetriever


class TestTwoTierRetriever:
    def test_a_shortlist_of_the_whole_corpus_is_the_rerankers_exact_ranking(self, stores, tiers):
        hot, cold = tiers
        reference = cold.search(stores["queries"], 30)
        for m in (30, 31, 1000):
            r = TwoTierRetriever(hot, cold, candidates=m).search(stores["queries"], 30)
            assert_close_modulo_ties(r, reference)
        assert TwoTierRetriever(hot, cold, candidates=30).search(stores["queries"], 500).width == 30

    @pytest.mark.parametrize("m", [1, 4, 9, 30])
    def test_it_equals_tiered_retriever_for_k_up_to_the_shortlist(self, stores, tiers, m):
        hot, cold = tiers
        q = stores["queries"]
        tiered = TieredRetriever(hot, cold, candidates=m)
        two = TwoTierRetriever(hot, cold, candidates=m)
        for k in sorted({1, m}):
            a, b = two.search(q, k), tiered.search(q, k)
            np.testing.assert_array_equal(a.indices, b.indices)
            np.testing.assert_array_equal(a.scores, b.scores)
            assert a.ids == b.ids
        assert a.score_kind == "maxsim_ip" and b.score_kind == "tiered"

    def test_t11_exact_cold_ties_break_by_store_position_not_by_shortlist_order(self):
        q = _docs([[[1, 0]]])
        cold = ExactRetriever(_docs([[[0.5, 0]], [[0.2, 0]], [[0.5, 0]]]))  # d0 and d2 tie
        hot = ExactRetriever(_docs([[[0.6, 0]], [[0.1, 0]], [[0.9, 0]]]))  # hot prefers d2 over d0
        shortlist = hot.search(q, 3).indices
        np.testing.assert_array_equal(shortlist, [[2, 0, 1]])
        naive = np.take_along_axis(shortlist, rank(cold.rescore(q, shortlist)), axis=1)
        np.testing.assert_array_equal(naive, [[2, 0, 1]])  # ranking in first-stage order: wrong
        tiered = TieredRetriever(hot, cold, candidates=3).search(q, 3)
        two = TwoTierRetriever(hot, cold, candidates=3).search(q, 3)
        np.testing.assert_array_equal(tiered.indices, [[0, 2, 1]])
        np.testing.assert_array_equal(two.indices, [[0, 2, 1]])
        np.testing.assert_array_equal(
            TwoTierRetriever(hot, cold, candidates=2).search(q, 2).indices, [[0, 2]]
        )

    @pytest.mark.parametrize("m", [3, 5, 10, 20, 40])
    def test_tie_heavy_data_matches_tiered_index(self, m):
        # Cold: eight documents, each stored five times, so cold scores tie exactly. Hot: the same
        # documents with noise, so the hot order among the copies is unrelated to store position.
        rng = np.random.default_rng(21)
        base = [_unit(rng.standard_normal((4, 16))) for _ in range(8)]
        order = rng.permutation(np.repeat(np.arange(8), 5))
        ids = [f"t{i}" for i in range(40)]
        cold_docs = MultiVectorCorpus.from_arrays([base[i] for i in order], ids=ids)
        hot_docs = MultiVectorCorpus.from_arrays(
            [_unit(base[i] + 0.15 * rng.standard_normal(base[i].shape).astype(np.float32)) for i in order],
            ids=ids,
        )
        q = _queries(seed=22, n=9, dim=16)
        hot = ExactRetriever(hot_docs)
        cold_store = Pipeline([Int8Quantizer("per_vector")]).compress(cold_docs)
        cold = CompressedRetriever(cold_store)
        top, scores = TieredIndex(ExactIndex(hot_docs), ExactIndex(cold_store), candidates=m).search(q, m)
        got = TwoTierRetriever(hot, cold, candidates=m).search(q, m)
        np.testing.assert_array_equal(got.indices, top)
        np.testing.assert_array_equal(got.scores, scores)
        np.testing.assert_array_equal(TieredRetriever(hot, cold, candidates=m).search(q, m).indices, top)
        # the data discriminates: ranking in first-stage order would give a different answer
        shortlist = rank(hot.score(q), m)
        cold_scores = cold.rescore(q, shortlist)
        assert all(len(set(row.tolist())) < m for row in cold_scores)  # exact ties in every row
        naive = np.take_along_axis(shortlist, rank(cold_scores), axis=1)
        assert (naive != top).any()

    def test_a_first_stage_that_only_knows_ids_gives_the_same_ranking(self, stores, tiers):
        hot, cold = tiers
        q = stores["queries"]
        direct = TwoTierRetriever(hot, cold, candidates=7).search(q, 7)
        by_id = TwoTierRetriever(_IdOnly(hot), cold, candidates=7).search(q, 7)
        assert by_id == direct and by_id.indices is not None

    def test_unknown_ids_raise_key_error_and_point_at_the_id_convention(self, stores, tiers):
        _, cold = tiers
        proposal = Ranking(ids=[["doc0::p1", "ghost"]] * 7, scores=np.zeros((7, 2), dtype=np.float32))
        two = TwoTierRetriever(_Fixed(proposal, 30), cold, candidates=2)
        with pytest.raises(KeyError, match="ghost.*::p"):
            two.search(stores["queries"], 2)

    def test_ids_need_a_reranker_that_exposes_ids_and_unique_ones(self, stores):
        proposal = Ranking(ids=[["doc0::p1"]] * 7, scores=np.zeros((7, 1), dtype=np.float32))

        class NoIds:
            n_docs = 30

            def rescore(self, queries, candidates):
                raise AssertionError("not reached")

        with pytest.raises(TypeError, match="reranker must expose ids"):
            TwoTierRetriever(_Fixed(proposal, 30), NoIds(), candidates=1).search(stores["queries"], 1)
        duplicated = ExactRetriever(MultiVectorCorpus(
            stores["docs"].vectors, stores["docs"].offsets, ids=["same"] * 30))
        with pytest.raises(ValueError, match="not unique"):
            TwoTierRetriever(_Fixed(proposal, 30), duplicated, candidates=1).search(stores["queries"], 1)

    def test_short_rows_are_padded_and_never_reach_the_reranker(self, stores, tiers):
        _, cold = tiers
        spy = _Spy(cold)
        q = stores["queries"]
        n = len(q)
        ids = [[f"doc{3 + i}::p1" for i in (4, 0, 2)] if row % 2 == 0 else ["doc9::p1"] for row in range(n)]
        positions = np.full((n, 3), -1, dtype=np.int64)
        scores = np.full((n, 3), -np.inf, dtype=np.float32)
        for row, listed in enumerate(ids):
            positions[row, : len(listed)] = [int(i[3:].split("::")[0]) for i in listed]
            scores[row, : len(listed)] = np.linspace(2, 1, len(listed))
        proposal = Ranking(ids=ids, scores=scores, indices=positions, tie_break="unspecified")
        r = TwoTierRetriever(_Fixed(proposal, 30), spy, candidates=3).search(q, 3)
        (given,) = spy.calls
        assert given.shape == (n, 3) and given.min() >= 0  # the -1 padding was replaced
        np.testing.assert_array_equal(r.lengths(), [3 if row % 2 == 0 else 1 for row in range(n)])
        for row in range(n):
            listed = positions[row][positions[row] >= 0]
            want = cold.rescore(q, np.tile(listed, (n, 1)))[row]
            order = sorted(range(len(listed)), key=lambda j: (-want[j], listed[j]))
            assert r.indices[row, : len(listed)].tolist() == [int(listed[j]) for j in order]
            np.testing.assert_array_equal(r.scores[row, : len(listed)], want[order])
            assert (r.indices[row, len(listed):] == -1).all() and np.isneginf(r.scores[row, len(listed):]).all()
            assert r.ids[row] == [f"doc{p}::p1" for p in r.indices[row, : len(listed)]]

    def test_a_first_stage_may_return_fewer_documents_than_asked(self, stores, tiers):
        hot, cold = tiers

        class Five:  # proposes at most five documents, however many it is asked for
            n_docs = 30

            def search(self, queries, k=10):
                return hot.search(queries, 5)

        r = TwoTierRetriever(Five(), cold, candidates=10).search(stores["queries"], 10)
        assert r.width == 5
        np.testing.assert_array_equal(
            r.indices, TwoTierRetriever(hot, cold, candidates=5).search(stores["queries"], 5).indices
        )

    def test_proposals_are_checked(self, stores, tiers):
        _, cold = tiers
        q = stores["queries"]
        n = len(q)

        def fixed(ids, indices=None):
            scores = np.zeros((n, len(ids[0])), dtype=np.float32)
            return _Fixed(Ranking(ids=ids, scores=scores, indices=indices, tie_break="unspecified"), 30)

        twice = np.tile(np.array([4, 4], dtype=np.int64), (n, 1))
        with pytest.raises(ValueError, match="same document twice for query 0"):
            TwoTierRetriever(fixed([["doc4::p1"] * 2] * n, twice), cold, candidates=2).search(q, 2)
        too_far = np.tile(np.array([4, 30], dtype=np.int64), (n, 1))
        with pytest.raises(IndexError, match="position 30"):
            TwoTierRetriever(fixed([["doc4::p1", "x"]] * n, too_far), cold, candidates=2).search(q, 2)
        wide = Ranking(ids=[["doc0::p1", "doc1::p1", "doc2::p1"]] * n, scores=np.zeros((n, 3), np.float32),
                       tie_break="unspecified")
        with pytest.raises(ValueError, match="3 columns for a request of 2"):
            TwoTierRetriever(_Fixed(wide, 30), cold, candidates=2).search(q, 2)

    def test_proposed_ids_must_name_the_documents_at_the_proposed_positions(self, stores, tiers):
        _, cold = tiers
        n = len(stores["queries"])
        positions = np.full((n, 1), 2, dtype=np.int64)
        wrong = Ranking(ids=[["doc1::p1"]] * n, scores=np.zeros((n, 1), dtype=np.float32),
                        indices=positions, tie_break="unspecified")
        with pytest.raises(ValueError, match="position 2 'doc1::p1'.*'doc2::p1'"):
            TwoTierRetriever(_Fixed(wrong, 30), cold, candidates=1).search(stores["queries"], 1)
        right = Ranking(ids=[["doc2::p1"]] * n, scores=np.zeros((n, 1), dtype=np.float32),
                        indices=positions, tie_break="unspecified")
        r = TwoTierRetriever(_Fixed(right, 30), cold, candidates=1).search(stores["queries"], 1)
        assert r.ids == [["doc2::p1"]] * n and (r.indices == 2).all()

    def test_a_first_stage_that_answers_the_wrong_number_of_queries_is_refused(self, stores, tiers):
        _, cold = tiers
        short = Ranking(ids=[["doc0::p1"]] * 3, scores=np.zeros((3, 1), dtype=np.float32),
                        tie_break="unspecified")
        with pytest.raises(ValueError, match="3 rows for 7 queries"):
            TwoTierRetriever(_Fixed(short, 30), cold, candidates=1).search(stores["queries"], 1)

    def test_k_beyond_the_shortlist_is_refused_unless_the_shortlist_is_the_corpus(self, stores, tiers):
        hot, cold = tiers
        q = stores["queries"]
        two = TwoTierRetriever(hot, cold, candidates=5)
        with pytest.raises(ValueError, match="k=6 exceeds the shortlist size 5"):
            two.search(q, 6)
        assert two.search(q, 5).width == 5
        assert TwoTierRetriever(hot, cold, candidates=45).search(q, 40).width == 30
        for bad in (0, -2):
            with pytest.raises(ValueError):
                two.search(q, bad)
        with pytest.raises(TypeError):
            two.search(q, 1.5)

    def test_construction_is_validated(self, tiers):
        hot, cold = tiers
        for bad, error in ((0, ValueError), (2.5, TypeError), (True, TypeError)):
            with pytest.raises(error):
                TwoTierRetriever(hot, cold, candidates=bad)
        with pytest.raises(TypeError, match="first must provide search"):
            TwoTierRetriever(object(), cold)
        with pytest.raises(TypeError, match="reranker must provide rescore"):
            TwoTierRetriever(hot, object())

        class Anonymous:
            def search(self, queries, k=10):
                raise AssertionError("not reached")

            def rescore(self, queries, candidates):
                raise AssertionError("not reached")

        with pytest.raises(TypeError, match="n_docs"):
            TwoTierRetriever(Anonymous(), Anonymous())

    def test_misaligned_stores_that_both_expose_ids_are_refused(self, stores, tiers):
        hot, _ = tiers
        backwards = ExactRetriever(stores["docs"].select(np.arange(30)[::-1]))
        with pytest.raises(ValueError, match="same order"):
            TwoTierRetriever(hot, backwards)

    def test_the_reranker_contract_is_enforced(self, stores, tiers):
        hot, _ = tiers

        class Wrong:
            def __init__(self, result):
                self.result = result
                self.n_docs = 30

            def rescore(self, queries, candidates):
                return self.result

        q = stores["queries"]
        with pytest.raises(ValueError, match="expected"):
            TwoTierRetriever(hot, Wrong(np.zeros((7, 2), dtype=np.float32)), candidates=3).search(q, 3)
        with pytest.raises(TypeError, match="float32"):
            TwoTierRetriever(hot, Wrong(np.zeros((7, 3))), candidates=3).search(q, 3)

    def test_the_score_kind_is_the_rerankers_to_declare(self, stores, tiers):
        hot, cold = tiers
        default = TwoTierRetriever(hot, cold, candidates=4).search(stores["queries"], 4)
        assert default.score_kind == "maxsim_ip" and default.tie_break == "index"
        named = TwoTierRetriever(hot, cold, candidates=4, score_kind="maxsim_cosine")
        assert named.search(stores["queries"], 4).score_kind == "maxsim_cosine"

    def test_empty_stores_give_an_empty_ranking(self):
        nothing = ExactRetriever(MultiVectorCorpus(np.zeros((0, 16), dtype=np.float32), [0]))
        r = TwoTierRetriever(nothing, nothing, candidates=3).search(_queries(), 5)
        assert r.width == 0 and r.n_queries == 7 and r.indices.shape == (7, 0)

    def test_documented_difference_when_the_cold_store_lacks_a_shortlisted_document(self):
        # TieredIndex ranks the whole matrix, so documents outside the shortlist that have a finite
        # score sit above a shortlisted document that is empty in the cold store; the two-tier
        # retriever has nothing outside its shortlist and puts the empty document last within it.
        rng = np.random.default_rng(5)
        arrays = [_unit(rng.standard_normal((5, 8))) for _ in range(12)]
        ids = [f"d{i}" for i in range(12)]
        hot = ExactRetriever(MultiVectorCorpus.from_arrays(arrays, ids=ids))
        arrays[3] = np.zeros((0, 8), dtype=np.float32)
        cold = ExactRetriever(MultiVectorCorpus.from_arrays(arrays, ids=ids))
        q = _queries(seed=6, n=6, dim=8)
        m = 6
        tiered = TieredRetriever(hot, cold, candidates=m).search(q, 12)
        two = TwoTierRetriever(hot, cold, candidates=m).search(q, m)
        carries_it = 0
        for row in range(6):
            if 3 not in set(two.indices[row]):
                np.testing.assert_array_equal(two.indices[row], tiered.indices[row, :m])
                continue
            carries_it += 1
            assert two.indices[row, -1] == 3 and np.isneginf(two.scores[row, -1])
            np.testing.assert_array_equal(two.indices[row, : m - 1], tiered.indices[row, : m - 1])
            assert tiered.indices[row, m - 1] != 3  # TieredIndex put another document there
        assert carries_it > 0

    def test_it_works_with_a_ranking_of_any_shape_of_first_stage_scores(self, stores, tiers):
        # only the proposed documents matter; the first stage's own scores are never used
        hot, cold = tiers
        r = hot.search(stores["queries"], 6)
        shuffled = dataclasses.replace(
            r, scores=np.zeros_like(r.scores), indices=r.indices[:, ::-1].copy(),
            ids=[row[::-1] for row in r.ids], tie_break="unspecified")
        a = TwoTierRetriever(_Fixed(shuffled, 30), cold, candidates=6).search(stores["queries"], 6)
        b = TwoTierRetriever(hot, cold, candidates=6).search(stores["queries"], 6)
        assert a == b


def test_the_two_tier_ranking_rescoring_is_maxsim_over_the_cold_codes(stores, tiers):
    hot, cold = tiers
    q = stores["queries"]
    r = TwoTierRetriever(hot, cold, candidates=8).search(q, 8)
    full = maxsim_matrix(q, stores["cold"])
    np.testing.assert_allclose(r.scores, np.take_along_axis(full, r.indices, axis=1), rtol=1e-5)


@pytest.mark.parametrize("seed", range(16))
def test_random_tiers_two_tier_equals_tiered_index_within_the_documented_condition(seed):
    rng = np.random.default_rng(2000 + seed)
    dim, n_docs = int(rng.integers(3, 20)), int(rng.integers(4, 30))
    counts = rng.integers(1, 9, size=n_docs)
    empty = rng.random(n_docs) < 0.2  # empty in both tiers, at the same positions
    counts = np.where(empty, 0, counts)
    ids = [f"r{i}" for i in range(n_docs)]
    arrays = [_unit(rng.standard_normal((max(c, 1), dim)))[:c] for c in counts]
    docs = MultiVectorCorpus.from_arrays(arrays, ids=ids)
    if docs.num_vectors == 0:
        pytest.skip("nothing to compress")
    hot_codec = [BinaryQuantizer(), Int8Quantizer("per_vector")][seed % 2]
    cold_codec = [Int8Quantizer("per_vector"), None][(seed // 2) % 2]
    hot = CompressedRetriever(Pipeline([hot_codec]).compress(docs))
    cold_store = Pipeline([cold_codec] if cold_codec else []).compress(docs)
    cold = CompressedRetriever(cold_store)
    q = MultiVectorCorpus.from_arrays(
        [_unit(rng.standard_normal((int(rng.integers(1, 6)), dim))) for _ in range(int(rng.integers(1, 7)))])
    m = int(rng.integers(1, n_docs + 4))
    finite = int((counts > 0).sum())  # documents with a finite cold score
    k = max(1, min(m, finite, n_docs))
    reference = TieredIndex(ExactIndex(hot.store), ExactIndex(cold_store), candidates=m)
    top, scores = reference.search(q, k)
    two = TwoTierRetriever(hot, cold, candidates=m).search(q, k)
    np.testing.assert_array_equal(two.indices, top)
    np.testing.assert_array_equal(two.scores, scores)
    tiered = TieredRetriever(hot, cold, candidates=m).search(q, n_docs)
    np.testing.assert_array_equal(tiered.indices[:, :k], top)
    np.testing.assert_array_equal(tiered.indices, rank(reference.score(q)))

