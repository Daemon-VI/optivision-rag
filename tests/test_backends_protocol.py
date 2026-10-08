"""The backend interface: protocol, info, factory, re-exports, and a conformance suite.

The conformance tests at the bottom run against every backend registered in BACKEND_FACTORIES
(NumPy always, Qdrant when qdrant-client is installed). To add a backend, write its wrapper and
register a factory here; docs/package/BACKEND_ARCHITECTURE.md lists the checklist.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from optivision import backends
from optivision import index as legacy_index
from optivision.backends import (
    Backend,
    BackendInfo,
    BackendRetriever,
    NumpyBackend,
    Ranking,
    comparable,
    legacy_id_mismatches,
    legacy_page_id,
    make_backend,
    require_comparable,
)
from optivision.compose import Pipeline
from optivision.core.retrieve import ExactRetriever, Retriever, TwoTierRetriever
from optivision.index.numpy_index import NumpyIndex
from optivision.representation import MultiVectorCorpus
from optivision.stages import BinaryQuantizer
from optivision.storage import to_legacy_pages
from optivision.types import CompressedPage, PageRef

DIM = 16


def _unit(v):
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _corpus(seed=0, n_docs=12, dim=DIM, ids=None):
    rng = np.random.default_rng(seed)
    arrays = [_unit(rng.standard_normal((int(rng.integers(3, 9)), dim))) for _ in range(n_docs)]
    return MultiVectorCorpus.from_arrays(arrays, ids=ids or [f"doc{i}::p1" for i in range(n_docs)])


def _queries(seed=1, n=5, tokens=4, dim=DIM):
    rng = np.random.default_rng(seed)
    return MultiVectorCorpus.from_arrays([_unit(rng.standard_normal((tokens, dim))) for _ in range(n)])


def _pages(corpus, method="none"):
    stages = {"none": [], "binary": [BinaryQuantizer()]}[method]
    return to_legacy_pages(Pipeline(stages).compress(corpus))


def _empty_page(dim=DIM, name="empty", nbytes=None):
    nbytes = 4 * dim if nbytes is None else nbytes
    return CompressedPage(ref=PageRef(name, 1), codes=np.zeros((0, nbytes), dtype=np.uint8), dim=dim,
                          n_tokens_before=0, n_tokens_after=0)


# ------------------------------------------------------------- the registry


def _numpy_backend(tmp_path, dim, method):
    return make_backend(NumpyIndex(tmp_path / "numpy", dim=dim, method=method))


def _qdrant_backend(tmp_path, dim, method):
    pytest.importorskip("qdrant_client")
    from optivision.index.qdrant_index import QdrantIndex

    return make_backend(QdrantIndex(path=tmp_path / "qdrant", dim=dim, method=method, recreate=True))


BACKEND_FACTORIES = {"numpy": _numpy_backend, "qdrant": _qdrant_backend}
#: backends whose repeated identical calls return bitwise equal scores
BITWISE_REPEATABLE = {"numpy"}


@pytest.fixture(params=sorted(BACKEND_FACTORIES))
def empty_backend(request, tmp_path):
    backend = BACKEND_FACTORIES[request.param](tmp_path, DIM, "none")
    yield backend
    backend.close()


@pytest.fixture
def loaded(empty_backend):
    corpus = _corpus()
    pages = _pages(corpus)
    empty_backend.add(pages)
    return empty_backend, corpus, pages


# ------------------------------------------------------- info and protocol


class TestBackendInfo:
    def test_it_is_a_frozen_record(self):
        info = BackendInfo("numpy", True, "maxsim_ip", "index", True, False)
        assert info == BackendInfo("numpy", True, "maxsim_ip", "index", True, False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.exact = False

    @pytest.mark.parametrize("changes", [{"name": ""}, {"score_kind": ""}, {"tie_break": "random"}])
    def test_it_validates(self, changes):
        fields = {"name": "x", "exact": True, "score_kind": "maxsim_ip", "tie_break": "index",
                  "dense_scores": False, "upsert": False}
        fields.update(changes)
        with pytest.raises(ValueError):
            BackendInfo(**fields)

    def test_scores_of_different_kinds_are_not_comparable(self):
        numpy_info = BackendInfo("numpy", True, "maxsim_ip", "index", True, False)
        qdrant_info = BackendInfo("qdrant", True, "maxsim_cosine", "unspecified", False, True)
        assert comparable(numpy_info, numpy_info)
        assert not comparable(numpy_info, qdrant_info)
        with pytest.raises(ValueError, match="not comparable"):
            require_comparable(numpy_info, qdrant_info)


class TestProtocolAndFactory:
    def test_numpy_backend_is_a_backend(self, tmp_path):
        backend = _numpy_backend(tmp_path, DIM, "none")
        assert isinstance(backend, Backend) and isinstance(backend, NumpyBackend)
        assert backend.info.name == "numpy" and backend.info.exact
        assert backend.info.score_kind == "maxsim_ip" and backend.info.tie_break == "index"
        assert backend.info.dense_scores and not backend.info.upsert

    def test_isinstance_checks_attribute_presence_only(self):
        class Half:
            info = None
            n_pages = 0

            def add(self, pages): ...

        assert not isinstance(Half(), Backend)
        assert not isinstance(object(), Backend)

    def test_make_backend_wraps_the_index_it_is_given(self, tmp_path):
        index = NumpyIndex(tmp_path, dim=DIM, method="none")
        backend = make_backend(index)
        assert isinstance(backend, NumpyBackend) and backend.index is index
        assert make_backend(backend) is backend  # already wrapped

        class Sub(NumpyIndex):
            pass

        assert isinstance(make_backend(Sub(tmp_path / "sub", dim=DIM)), NumpyBackend)

    def test_make_backend_refuses_what_it_cannot_wrap(self):
        class Stranger:
            backend = "faiss"

        with pytest.raises(TypeError, match="faiss"):
            make_backend(Stranger())
        with pytest.raises(TypeError):
            make_backend(object())
        with pytest.raises(TypeError, match="NumpyIndex"):
            NumpyBackend(object())

    def test_the_canonical_import_place_exports_the_same_objects(self):
        assert backends.BaseIndex is legacy_index.BaseIndex
        assert backends.NumpyIndex is legacy_index.NumpyIndex is NumpyIndex
        assert backends.get_index is legacy_index.get_index
        assert backends.open_index is legacy_index.open_index
        from optivision.index.qdrant_index import QdrantIndex  # imports without qdrant_client

        assert backends.QdrantIndex is QdrantIndex
        assert "QdrantIndex" in dir(backends) and "QdrantIndex" not in backends.__all__
        with pytest.raises(AttributeError):
            backends.NoSuchThing  # noqa: B018
        for name in backends.__all__:
            assert hasattr(backends, name), name

    def test_importing_the_backends_needs_neither_the_engine_nor_torch(self):
        code = (
            "import sys, optivision.backends as b; b.QdrantIndex; "
            "print(sorted({'qdrant_client', 'torch'} & set(sys.modules)))"
        )
        src = str(Path(__file__).resolve().parents[1] / "src")
        env = {**os.environ, "PYTHONPATH": src}
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                             check=True)
        assert out.stdout.strip() == "[]"

    def test_the_old_import_paths_keep_working(self):
        from optivision.index import BaseIndex, get_index, open_index
        from optivision.index import NumpyIndex as old_numpy
        from optivision.index.base import BaseIndex as old_base
        from optivision.index.numpy_index import NumpyIndex as old_numpy_module

        assert BaseIndex is old_base is backends.BaseIndex
        assert old_numpy is old_numpy_module is backends.NumpyIndex
        assert get_index is backends.get_index and open_index is backends.open_index


# ----------------------------------------------------- the id convention


class TestIdConvention:
    IDS = ("doc0::p1", "abc", "a::pb", "x::p01", "a::p2::p3", "page::p0", "doc::p12")

    def test_the_mirror_agrees_with_to_legacy_pages(self):
        for corpus_id in self.IDS:
            store = Pipeline().compress(MultiVectorCorpus.from_arrays(
                [np.ones((1, 4), dtype=np.float32)], ids=[corpus_id]))
            (page,) = to_legacy_pages(store)
            assert legacy_page_id(corpus_id) == page.ref.page_id, corpus_id

    def test_the_mirror_reports_ids_the_bridge_cannot_convert(self):
        store = Pipeline().compress(MultiVectorCorpus.from_arrays(
            [np.ones((1, 4), dtype=np.float32)], ids=["z::p²"]))
        try:
            got = to_legacy_pages(store)[0].ref.page_id
        except ValueError:  # str.isdigit accepts a superscript that int() rejects
            got = None
        assert legacy_page_id("z::p²") == got

    def test_ids_that_do_not_survive_the_bridge_are_listed_not_repaired(self):
        assert legacy_id_mismatches(self.IDS) == [
            ("abc", "abc::p1"),
            ("a::pb", "a::p1"),
            ("x::p01", "x::p1"),
            ("a::p2::p3", "a::p1"),
        ]
        assert legacy_id_mismatches(["doc0::p1", "doc7::p3"]) == []

    def test_a_two_tier_search_does_not_hide_the_mismatch(self, tmp_path):
        corpus = _corpus(ids=[f"doc{i}" for i in range(12)])  # no "::p<N>" suffix
        backend = _numpy_backend(tmp_path, DIM, "none")
        backend.add(_pages(corpus))
        assert backend.search_many(_queries(), 3).ids[0][0].endswith("::p1")
        two = TwoTierRetriever(BackendRetriever(backend), ExactRetriever(corpus), candidates=4)
        with pytest.raises(KeyError, match="::p1"):
            two.search(_queries(), 3)


def test_a_raw_backend_as_the_first_stage_is_refused_at_construction(tmp_path):
    corpus = _corpus()
    backend = _numpy_backend(tmp_path, DIM, "none")
    backend.add(_pages(corpus))
    with pytest.raises(TypeError, match="BackendRetriever"):
        TwoTierRetriever(backend, ExactRetriever(corpus), candidates=4)
    TwoTierRetriever(BackendRetriever(backend), ExactRetriever(corpus), candidates=4)  # accepted


# ------------------------------------------------------- BackendRetriever


class TestBackendRetriever:
    def test_it_is_a_first_stage_retriever_without_positions(self, tmp_path):
        corpus = _corpus()
        backend = _numpy_backend(tmp_path, DIM, "none")
        backend.add(_pages(corpus))
        first = BackendRetriever(backend)
        assert isinstance(first, Retriever) and first.n_docs == 12
        r = first.search(_queries(), 4)
        assert r.indices is None and r.score_kind == "maxsim_ip"
        assert r.ids == backend.search_many(_queries(), 4).ids

    def test_two_tier_over_a_backend_with_a_shortlist_of_the_corpus_is_the_exact_ranking(self, tmp_path):
        corpus = _corpus()
        backend = _numpy_backend(tmp_path, DIM, "binary")
        backend.add(_pages(corpus, "binary"))
        cold = ExactRetriever(corpus)
        two = TwoTierRetriever(BackendRetriever(backend), cold, candidates=12).search(_queries(), 12)
        reference = cold.search(_queries(), 12)
        np.testing.assert_allclose(two.scores, reference.scores, rtol=1e-5, atol=1e-5)
        assert two.ids == reference.ids
        small = TwoTierRetriever(BackendRetriever(backend), cold, candidates=5).search(_queries(), 5)
        assert small.width == 5 and set(np.unique(small.indices)) <= set(range(12))

    def test_it_refuses_what_is_not_a_backend(self):
        with pytest.raises(TypeError, match="search_many"):
            BackendRetriever(object())


# ------------------------------------------------------ conformance suite


class TestConformance:
    def test_it_satisfies_the_backend_protocol(self, empty_backend):
        assert isinstance(empty_backend, Backend)
        assert isinstance(empty_backend.info, BackendInfo)
        assert empty_backend.info.name == empty_backend.index.backend

    def test_info_tells_the_truth_about_scores(self, loaded):
        backend, _, _ = loaded
        info = backend.info
        assert info.tie_break in {"index", "unspecified"} and isinstance(info.exact, bool)
        r = backend.search_many(_queries(), 3)
        assert (r.score_kind, r.tie_break) == (info.score_kind, info.tie_break)
        assert comparable(r, info)
        assert info.dense_scores == callable(getattr(backend, "score", None))

    def test_info_tells_the_truth_about_re_adding_a_page(self, loaded):
        backend, _, pages = loaded
        before = backend.n_pages
        backend.add(pages[:1])
        assert (backend.n_pages == before) == backend.info.upsert
        assert (backend.n_pages == before + 1) == (not backend.info.upsert)

    def test_search_many_returns_a_well_formed_ranking(self, loaded):
        backend, corpus, _ = loaded
        queries = _queries()
        r = backend.search_many(queries, 5)
        assert isinstance(r, Ranking) and r.n_queries == len(queries) and r.width == 5
        assert r.scores.dtype == np.float32 and (r.lengths() == 5).all()
        for row in r.ids:
            assert len(set(row)) == 5 and set(row) <= set(corpus.ids)

    def test_a_depth_beyond_the_index_gives_a_narrower_ranking(self, loaded):
        backend, corpus, _ = loaded
        r = backend.search_many(_queries(), 500)
        assert r.width == len(corpus) and all(sorted(row) == sorted(corpus.ids) for row in r.ids)

    def test_a_page_is_found_by_its_own_vectors(self, loaded):
        backend, corpus, _ = loaded
        own = MultiVectorCorpus.from_arrays([corpus[i].vectors for i in (2, 7, 11)])
        r = backend.search_many(own, 1)
        assert [row[0] for row in r.ids] == [corpus.ids[i] for i in (2, 7, 11)]
        legacy = backend.search(corpus[7].vectors, top_k=1)
        assert legacy[0].ref.page_id == corpus.ids[7]

    def test_the_same_call_twice_gives_the_same_ranking(self, loaded):
        backend, _, _ = loaded
        queries = _queries()
        a, b = backend.search_many(queries, 6), backend.search_many(queries, 6)
        assert a.ids == b.ids  # no near-ties in this data
        np.testing.assert_allclose(a.scores, b.scores, rtol=1e-6)
        if backend.info.name in BITWISE_REPEATABLE:
            assert a == b  # an in-process scan repeats to the bit
        # An engine's repeated calls were seen to differ in the last bit of a score (one ulp):
        # reproducible to float32 rounding, not bitwise (docs/package/BACKEND_ARCHITECTURE.md).

    @pytest.mark.parametrize("k, error", [(0, ValueError), (-1, ValueError), (1.5, TypeError),
                                          (True, TypeError), (None, TypeError)])
    def test_k_must_be_a_positive_integer(self, loaded, k, error):
        with pytest.raises(error):
            loaded[0].search_many(_queries(), k)

    def test_queries_are_validated_before_the_engine_sees_them(self, loaded):
        backend = loaded[0]
        nan = np.array(_queries().vectors)
        nan[2, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            backend.search_many(_queries().with_vectors(nan), 3)
        with pytest.raises(ValueError, match="dimension 8"):
            backend.search_many(_queries(dim=8), 3)
        with pytest.raises(TypeError):
            backend.search_many([np.zeros((4, DIM), dtype=np.int64)], 3)
        with pytest.raises(TypeError):
            backend.search_many(np.zeros((4, DIM), dtype=np.float32), 3)

    def test_an_empty_index_gives_empty_rankings(self, empty_backend):
        r = empty_backend.search_many(_queries(), 3)
        assert r.n_queries == 5 and r.width == 0 and r.ids == [[]] * 5

    def test_stats_and_counts_are_the_indexs_own(self, loaded):
        backend, corpus, _ = loaded
        assert backend.n_pages == len(corpus) == backend.stats()["n_pages"]
        assert backend.stats() == backend.index.stats()
        backend.save()  # a no-op for engines that persist on write; must not raise

    def test_the_legacy_single_query_search_is_still_there(self, loaded):
        backend, corpus, _ = loaded
        hits = backend.search(_queries()[0].vectors, top_k=3)
        assert [h.rank for h in hits] == [1, 2, 3]
        assert {h.ref.page_id for h in hits} <= set(corpus.ids)

    def test_it_works_as_a_context_manager(self, empty_backend):
        with empty_backend as b:
            assert b is empty_backend
