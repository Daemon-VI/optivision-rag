"""The NumPy backend: deterministic batch search over NumpyIndex, and what it leaves unchanged.

Cases T15-T17 of docs/package/RETRIEVAL_BACKEND_AUDIT.md section 7 for the NumPy wrapper.
Equality with the in-process retrievers is rounding-level, never bitwise, so these tests compare
scores with tolerances and rankings outside float32 near-ties.
"""

from __future__ import annotations

import numpy as np
import pytest

from optivision.backends import NumpyBackend, comparable
from optivision.compose import Pipeline
from optivision.core.retrieve import CompressedRetriever, ExactRetriever, Ranking
from optivision.index.numpy_index import NumpyIndex
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix
from optivision.stages import BinaryQuantizer, Int8Quantizer
from optivision.storage import to_legacy_pages
from optivision.types import CompressedPage, PageRef

DIM = 32


def _unit(v):
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _corpus(seed=0, n_docs=30, dim=DIM):
    rng = np.random.default_rng(seed)
    arrays = [_unit(rng.standard_normal((int(rng.integers(3, 12)), dim))) for _ in range(n_docs)]
    return MultiVectorCorpus.from_arrays(arrays, ids=[f"doc{i}::p1" for i in range(n_docs)])


def _queries(seed=1, n=9, tokens=5, dim=DIM):
    rng = np.random.default_rng(seed)
    return MultiVectorCorpus.from_arrays([_unit(rng.standard_normal((tokens, dim))) for _ in range(n)])


_CODECS = {
    "none": ("none", []),
    "binary": ("binary", [BinaryQuantizer()]),
    "int8": ("int8", [Int8Quantizer("fixed")]),
}


def _stored(corpus, name):
    method, stages = _CODECS[name]
    return method, Pipeline(stages).compress(corpus)


def _backend(tmp_path, corpus, name="binary", **index_options):
    method, store = _stored(corpus, name)
    backend = NumpyBackend(NumpyIndex(tmp_path, dim=corpus.dimension, method=method, **index_options))
    backend.add(to_legacy_pages(store))
    return backend, store


def assert_close_modulo_ties(a: Ranking, b: Ranking, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(a.scores, b.scores, rtol=rtol, atol=atol)
    for q in range(a.n_queries):
        start = 0
        for j in range(1, a.width + 1):
            if j == a.width or not np.isclose(a.scores[q, j], a.scores[q, j - 1], rtol=rtol, atol=atol):
                assert set(a.ids[q][start:j]) == set(b.ids[q][start:j]), (q, start, j)
                start = j


class TestAgainstTheInProcessRetrievers:
    @pytest.mark.filterwarnings("ignore:Int8Quantizer:UserWarning")  # fixed scale clips a few components
    @pytest.mark.parametrize("name", ["none", "binary", "int8"])
    def test_t15_search_many_agrees_with_maxsim_matrix_over_the_same_codes(self, tmp_path, name):
        corpus, queries = _corpus(), _queries()
        backend, store = _backend(tmp_path, corpus, name)
        got = backend.search_many(queries, 30)
        reference = CompressedRetriever(store).search(queries, 30)
        assert comparable(got, reference)  # the same score kind and tie convention
        np.testing.assert_allclose(
            got.scores, np.take_along_axis(maxsim_matrix(queries, store), reference.indices, axis=1),
            rtol=1e-4, atol=1e-4)
        assert_close_modulo_ties(got, reference)
        assert got.ids == [[corpus.ids[i] for i in row] for row in got.indices]
        np.testing.assert_array_equal(got.top(3).ids, reference.top(3).ids)  # top-3 ids equal

    def test_t15_the_float_store_agrees_with_the_exact_retriever(self, tmp_path):
        corpus, queries = _corpus(), _queries()
        backend, _ = _backend(tmp_path, corpus, "none")
        assert_close_modulo_ties(backend.search_many(queries, 30), ExactRetriever(corpus).search(queries, 30))

    def test_the_streaming_regime_ranks_like_the_cached_regime(self, tmp_path):
        corpus, queries = _corpus(), _queries()
        cached, _ = _backend(tmp_path / "a", corpus)
        streamed, _ = _backend(tmp_path / "b", corpus, max_decoded_bytes=256)
        assert streamed.index.decoded_nbytes > streamed.index.max_decoded_bytes
        assert_close_modulo_ties(streamed.search_many(queries, 30), cached.search_many(queries, 30))

    def test_score_is_score_all_per_query_unchanged(self, tmp_path):
        corpus, queries = _corpus(), _queries()
        backend, _ = _backend(tmp_path, corpus)
        dense = backend.score(queries)
        assert dense.dtype == np.float32 and dense.shape == (len(queries), len(corpus))
        for i in range(len(queries)):
            np.testing.assert_array_equal(dense[i], backend.index.score_all(queries[i].vectors))


class TestPersistence:
    def test_t15_save_and_load_preserve_the_ranking(self, tmp_path):
        corpus, queries = _corpus(), _queries()
        backend, _ = _backend(tmp_path / "idx", corpus)
        before = backend.search_many(queries, 10)
        backend.save()
        reloaded = NumpyBackend.load(tmp_path / "idx")
        assert isinstance(reloaded, NumpyBackend) and reloaded.n_pages == len(corpus)
        assert reloaded.search_many(queries, 10) == before
        assert reloaded.info == backend.info
        small = NumpyBackend.load(tmp_path / "idx", max_decoded_bytes=256)
        assert small.index.max_decoded_bytes == 256
        assert_close_modulo_ties(small.search_many(queries, 10), before)


class TestTiesAndEmpties:
    def test_t16_identical_pages_rank_by_position_every_time(self, tmp_path):
        v = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        corpus = MultiVectorCorpus.from_arrays([v] * 50, ids=[f"t{i}::p1" for i in range(50)])
        backend = NumpyBackend(NumpyIndex(tmp_path, dim=4, method="none"))
        backend.add(to_legacy_pages(Pipeline().compress(corpus)))
        queries = MultiVectorCorpus.from_arrays([v] * 3)
        r = backend.search_many(queries, 12)
        np.testing.assert_array_equal(r.indices, np.tile(np.arange(12), (3, 1)))
        assert r.ids[0] == [f"t{i}::p1" for i in range(12)] and r.tie_break == "index"
        assert backend.search_many(queries, 12) == r
        # the legacy single-query search is unchanged and still reachable; its tie order is
        # unspecified (argpartition + argsort), so only its contract is asserted
        legacy = backend.search(v, top_k=12)
        assert [h.rank for h in legacy] == list(range(1, 13)) and len({h.ref.page_id for h in legacy}) == 12
        assert backend.search(v, 12) == backend.index.search(v, 12)

    def _with_empty_pages(self, tmp_path):
        corpus = _corpus(n_docs=6)
        pages = to_legacy_pages(Pipeline([BinaryQuantizer()]).compress(corpus))
        empty = CompressedPage(ref=PageRef("empty", 1), codes=np.zeros((0, DIM // 8), dtype=np.uint8),
                               dim=DIM, n_tokens_before=0, n_tokens_after=0)
        backend = NumpyBackend(NumpyIndex(tmp_path, dim=DIM, method="binary"))
        backend.add([*pages[:2], empty, *pages[2:]])
        return backend

    def test_t17_an_empty_page_scores_minus_infinity_and_ranks_last(self, tmp_path):
        backend = self._with_empty_pages(tmp_path)
        dense = backend.score(_queries())
        assert np.isneginf(dense[:, 2]).all() and np.isfinite(np.delete(dense, 2, axis=1)).all()
        r = backend.search_many(_queries(), 7)
        assert (r.indices[:, -1] == 2).all() and np.isneginf(r.scores[:, -1]).all()
        assert (r.ids[0][-1] == "empty::p1")

    def test_t17_an_all_empty_index_returns_minus_infinity_not_zero(self, tmp_path):
        backend = NumpyBackend(NumpyIndex(tmp_path, dim=DIM, method="binary"))
        pages = [CompressedPage(ref=PageRef(f"e{i}", 1), codes=np.zeros((0, DIM // 8), dtype=np.uint8),
                                dim=DIM, n_tokens_before=0, n_tokens_after=0) for i in range(2)]
        backend.add(pages)
        assert backend.n_pages == 2
        dense = backend.score(_queries())
        assert dense.shape == (9, 2) and np.isneginf(dense).all()
        r = backend.search_many(_queries(), 5)
        assert r.ids[0] == ["e0::p1", "e1::p1"] and np.isneginf(r.scores).all()

    def test_an_empty_query_scores_zero_against_non_empty_pages(self, tmp_path):
        backend = self._with_empty_pages(tmp_path)
        none = MultiVectorCorpus(np.zeros((0, DIM), dtype=np.float32), [0, 0, 0])  # two empty queries
        dense = backend.score(none)
        np.testing.assert_array_equal(dense, [[0, 0, -np.inf, 0, 0, 0, 0]] * 2)
        mixed = MultiVectorCorpus.from_arrays([_queries()[0].vectors, np.zeros((0, DIM), dtype=np.float32)])
        r = backend.search_many(mixed, 7)
        np.testing.assert_array_equal(r.indices[1], [0, 1, 3, 4, 5, 6, 2])  # ties by position, empty last

    def test_an_index_without_pages_scores_nothing(self, tmp_path):
        backend = NumpyBackend(NumpyIndex(tmp_path, dim=DIM, method="binary"))
        assert backend.score(_queries()).shape == (9, 0)
        r = backend.search_many(_queries(), 3)
        assert r.width == 0 and r.ids == [[]] * 9


class TestDelegationAndContract:
    def test_it_wraps_the_index_and_leaves_it_alone(self, tmp_path):
        corpus = _corpus()
        backend, _ = _backend(tmp_path, corpus)
        assert isinstance(backend.index, NumpyIndex) and backend.n_pages == len(corpus)
        assert backend.stats() == backend.index.stats()
        assert backend.info.score_kind == "maxsim_ip" and backend.info.tie_break == "index"
        with pytest.raises(TypeError):
            NumpyBackend("not an index")

    def test_adding_a_page_twice_stores_it_twice(self, tmp_path):
        corpus = _corpus(n_docs=5)
        method, store = _stored(corpus, "binary")
        pages = to_legacy_pages(store)
        backend = NumpyBackend(NumpyIndex(tmp_path, dim=DIM, method=method))
        backend.add(pages)
        backend.add(pages[:1])
        assert backend.n_pages == 6 and not backend.info.upsert
        r = backend.search_many(_queries(), 6)
        assert all(row.count("doc0::p1") == 2 for row in r.ids)

    def test_a_page_of_the_wrong_width_is_the_indexs_error_and_changes_nothing(self, tmp_path):
        backend, _ = _backend(tmp_path, _corpus(n_docs=5))
        odd = CompressedPage(ref=PageRef("odd", 1), codes=np.zeros((2, 7), dtype=np.uint8), dim=DIM,
                             n_tokens_before=2, n_tokens_after=2)
        with pytest.raises(ValueError):
            backend.add([odd])
        assert backend.n_pages == 5

    def test_queries_are_validated(self, tmp_path):
        backend, _ = _backend(tmp_path, _corpus(n_docs=5))
        bad = np.array(_queries().vectors)
        bad[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            backend.search_many(_queries().with_vectors(bad), 3)
        with pytest.raises(ValueError, match="dimension 8"):
            backend.score(_queries(dim=8))
        for k in (0, -1):
            with pytest.raises(ValueError):
                backend.search_many(_queries(), k)

    def test_repeating_a_call_is_bitwise_identical(self, tmp_path):
        backend, _ = _backend(tmp_path, _corpus())
        assert backend.search_many(_queries(), 30) == backend.search_many(_queries(), 30)
