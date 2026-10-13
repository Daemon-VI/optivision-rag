"""The Qdrant backend (local path mode), and how its scores relate to the NumPy backend's.

Cases T14, T16, T17 and the T3 mechanism of docs/package/RETRIEVAL_BACKEND_AUDIT.md section 7.
Only Qdrant's local (path) mode is exercised: it is a brute-force scan, used here to test the
wrapper, not as a statement about the server. The two backends score differently (inner-product
MaxSim vs MaxSim of cosines), so their scores are never compared directly: ids are compared outside
tie groups, and scores only through the documented relations (see ``comparable``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optivision.backends import (
    NumpyBackend,
    QdrantBackend,
    comparable,
    make_backend,
    require_comparable,
)
from optivision.compose import Pipeline
from optivision.representation import MultiVectorCorpus
from optivision.stages import BinaryQuantizer, Int8Quantizer
from optivision.storage import to_legacy_pages
from optivision.types import CompressedPage, PageRef

DIM = 32


def _unit(v):
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _corpus(seed=0, n_docs=40, dim=DIM):
    rng = np.random.default_rng(seed)
    arrays = [_unit(rng.standard_normal((int(rng.integers(4, 13)), dim))) for _ in range(n_docs)]
    return MultiVectorCorpus.from_arrays(arrays, ids=[f"doc{i}::p1" for i in range(n_docs)])


def _queries(seed=1, n=30, tokens=6, dim=DIM):
    rng = np.random.default_rng(seed)
    return MultiVectorCorpus.from_arrays([_unit(rng.standard_normal((tokens, dim))) for _ in range(n)])


def _mixed_norm(queries, seed=2):
    scales = np.random.default_rng(seed).uniform(0.2, 5.0, size=(queries.num_vectors, 1))
    return queries.with_vectors(queries.vectors * scales.astype(np.float32))


@pytest.fixture
def engine(tmp_path):
    """Open QdrantIndex objects in local mode; closed at the end of the test."""
    pytest.importorskip("qdrant_client")
    from optivision.index.qdrant_index import QdrantIndex

    opened = []

    def make(dim, method, name="qdrant"):
        index = QdrantIndex(path=tmp_path / name, dim=dim, method=method, recreate=True)
        opened.append(index)
        return index

    yield make
    for index in opened:
        index.close()


def _pair(engine, tmp_path, corpus, name):
    """The same pages in a NumPy backend and a Qdrant backend."""
    from optivision.index.numpy_index import NumpyIndex

    stages = {"binary": [BinaryQuantizer()], "none": [], "int8": [Int8Quantizer("fixed")]}[name]
    store = Pipeline(stages).compress(corpus)
    pages = to_legacy_pages(store)
    method = {"binary": "binary", "none": "none", "int8": "int8"}[name]
    numpy_backend = NumpyBackend(NumpyIndex(tmp_path / f"np_{name}", dim=corpus.dimension, method=method))
    numpy_backend.add(pages)
    qdrant_backend = QdrantBackend(engine(corpus.dimension, method, name=f"qd_{name}"))
    qdrant_backend.add(pages)
    return numpy_backend, qdrant_backend, store


def assert_same_order_outside_ties(reference, other, rtol=1e-4, atol=1e-4):
    """Equal id order, except inside runs of reference scores that are equal to float32 noise."""
    for q in range(reference.n_queries):
        start = 0
        for j in range(1, reference.width + 1):
            if j == reference.width or not np.isclose(
                reference.scores[q, j], reference.scores[q, j - 1], rtol=rtol, atol=atol
            ):
                assert set(reference.ids[q][start:j]) == set(other.ids[q][start:j]), (q, start, j)
                start = j


class TestInfo:
    def test_a_local_store_is_exact_and_says_what_its_scores_are(self, engine):
        backend = QdrantBackend(engine(DIM, "none"))
        info = backend.info
        assert (info.name, info.exact) == ("qdrant", True)
        assert info.score_kind == "maxsim_cosine" and info.tie_break == "unspecified"
        assert not info.dense_scores and info.upsert
        assert make_backend(backend.index).info == info

    def test_a_server_is_not_exact(self):
        from optivision.index.qdrant_index import QdrantIndex

        server = object.__new__(QdrantIndex)  # the fields info reads; no client, no network
        server.location, server.path = "http://localhost:6333", Path("data/index/qdrant")
        assert QdrantBackend(server).info.exact is False
        server.location = str(server.path)
        assert QdrantBackend(server).info.exact is True

    def test_it_wraps_only_a_qdrant_index(self, tmp_path):
        from optivision.index.numpy_index import NumpyIndex

        with pytest.raises(TypeError, match="QdrantIndex"):
            QdrantBackend(NumpyIndex(tmp_path, dim=DIM))
        with pytest.raises(TypeError):
            QdrantBackend(object())

    def test_qdrant_scores_are_not_comparable_with_numpy_scores(self, engine, tmp_path):
        numpy_backend, qdrant_backend, _ = _pair(engine, tmp_path, _corpus(n_docs=6), "none")
        a, b = numpy_backend.search_many(_queries(n=2), 3), qdrant_backend.search_many(_queries(n=2), 3)
        assert not comparable(a, b) and not comparable(numpy_backend.info, qdrant_backend.info)
        with pytest.raises(ValueError, match="not comparable"):
            require_comparable(a, b)
        assert comparable(b, qdrant_backend.search_many(_queries(n=2), 3))


class TestAgainstTheNumpyBackend:
    @pytest.mark.parametrize("name, factor", [("binary", np.sqrt(DIM)), ("none", 1.0)])
    def test_t14_unit_norm_queries_give_one_ranking_and_a_known_scale(self, engine, tmp_path, name, factor):
        numpy_backend, qdrant_backend, _ = _pair(engine, tmp_path, _corpus(), name)
        queries = _queries()
        a, b = numpy_backend.search_many(queries, 40), qdrant_backend.search_many(queries, 40)
        assert a.width == b.width == 40 and b.indices is None
        assert_same_order_outside_ties(a, b)
        for q in range(len(queries)):  # Qdrant's score times sqrt(d) is NumPy's for binary codes
            by_id = dict(zip(b.ids[q], b.scores[q]))
            mine = np.array([by_id[i] for i in a.ids[q]], dtype=np.float64) * factor
            np.testing.assert_allclose(mine, a.scores[q], rtol=1e-4, atol=1e-4)

    @pytest.mark.filterwarnings("ignore:Int8Quantizer:UserWarning")  # fixed scale clips a few components
    def test_t14_decoded_int8_agrees_on_top_one_and_to_a_data_derived_ratio(self, engine, tmp_path):
        corpus = _corpus()
        numpy_backend, qdrant_backend, store = _pair(engine, tmp_path, corpus, "int8")
        norms = np.linalg.norm(store.decode_rows(0, store.num_vectors), axis=1)
        low, high = 1 / norms.max(), 1 / norms.min()  # cosine / inner product, per decoded vector
        queries = _queries()
        a, b = numpy_backend.search_many(queries, 40), qdrant_backend.search_many(queries, 40)
        decided = 0
        for q in range(len(queries)):
            # a top-one gap wider than the spread of the ratio cannot be reversed by it
            if a.scores[q, 0] > a.scores[q, 1] * high / low:
                decided += 1
                assert a.ids[q][0] == b.ids[q][0]
            by_id = dict(zip(b.ids[q], b.scores[q]))
            for page_id, score in zip(a.ids[q], a.scores[q]):
                if score > 1.0:
                    assert low * 0.999 <= by_id[page_id] / score <= high * 1.001
        assert decided >= 1  # the check was not vacuous
        assert sum(x != y for x, y in zip(a.ids, b.ids)) > 0  # decoded int8 is not unit-norm: orders differ

    @pytest.mark.parametrize("name", ["binary", "none"])
    def test_t3_query_token_norms_change_numpys_ranking_but_not_the_engines(self, engine, tmp_path, name):
        numpy_backend, qdrant_backend, _ = _pair(engine, tmp_path, _corpus(), name)
        unit_queries = _queries(n=12)
        mixed = _mixed_norm(unit_queries)
        cosine_order = numpy_backend.search_many(unit_queries, 40)  # inner product with unit tokens
        numpy_mixed = numpy_backend.search_many(mixed, 40)
        qdrant_mixed = qdrant_backend.search_many(mixed, 40)
        assert sum(x != y for x, y in zip(numpy_mixed.ids, cosine_order.ids)) >= 6  # IP is norm-sensitive
        assert_same_order_outside_ties(cosine_order, qdrant_mixed)  # cosine is not

    def test_t3_the_hand_example_through_both_backends(self, engine, tmp_path):
        a, b, c = [[1, 0]], [[0, 1]], [[0.7071, 0.7071]]
        corpus = MultiVectorCorpus.from_arrays(
            [np.array(x, dtype=np.float32) for x in (a, b, c)], ids=["A::p1", "B::p1", "C::p1"])
        numpy_backend, qdrant_backend, _ = _pair(engine, tmp_path, corpus, "none")
        q = MultiVectorCorpus.from_arrays([np.array([[10, 0], [0, 0.1]], dtype=np.float32)])
        ip, cosine = numpy_backend.search_many(q, 3), qdrant_backend.search_many(q, 3)
        assert ip.ids == [["A::p1", "C::p1", "B::p1"]]
        np.testing.assert_allclose(ip.scores, [[10.0, 7.1417, 0.1]], rtol=1e-4)
        assert cosine.ids[0][0] == "C::p1" and set(cosine.ids[0][1:]) == {"A::p1", "B::p1"}
        np.testing.assert_allclose(cosine.scores, [[1.4142, 1.0, 1.0]], rtol=1e-4)

    def test_t16_tied_pages_are_a_set_to_the_engine_not_a_sequence(self, engine, tmp_path):
        v = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        corpus = MultiVectorCorpus.from_arrays([v] * 10, ids=[f"t{i}::p1" for i in range(10)])
        numpy_backend, qdrant_backend, _ = _pair(engine, tmp_path, corpus, "none")
        queries = MultiVectorCorpus.from_arrays([v])
        in_process, engine_run = numpy_backend.search_many(queries, 10), qdrant_backend.search_many(queries, 10)
        assert in_process.ids == [[f"t{i}::p1" for i in range(10)]]  # by position
        assert set(engine_run.ids[0]) == set(in_process.ids[0])  # the same group...
        assert len(set(engine_run.scores[0].tolist())) == 1  # ...of exactly equal scores
        assert engine_run.tie_break == "unspecified"  # so no order is promised inside it


class TestWrapperBehaviour:
    def test_t17_an_empty_page_is_refused_before_anything_is_sent(self, engine):
        backend = QdrantBackend(engine(DIM, "none"))
        corpus = _corpus(n_docs=6)
        pages = to_legacy_pages(Pipeline().compress(corpus))
        backend.add(pages[:3])
        empty = CompressedPage(ref=PageRef("empty", 1), codes=np.zeros((0, 4 * DIM), dtype=np.uint8),
                               dim=DIM, n_tokens_before=0, n_tokens_after=0)
        with pytest.raises(ValueError, match="empty::p1.*no vectors.*nothing was added"):
            backend.add([pages[3], empty, pages[4]])
        assert backend.n_pages == 3  # not even the valid pages of the refused batch
        r = backend.search_many(_queries(n=2), 5)  # and the index is still usable
        assert r.width == 3 and set(r.ids[0]) == {f"doc{i}::p1" for i in range(3)}
        backend.add(pages[3:])
        assert backend.n_pages == 6

    def test_a_page_of_another_dimension_is_refused(self, engine):
        backend = QdrantBackend(engine(DIM, "none"))
        odd = CompressedPage(ref=PageRef("odd", 1), codes=np.zeros((2, 16), dtype=np.uint8), dim=4,
                             n_tokens_before=2, n_tokens_after=2)
        with pytest.raises(ValueError, match="dimension 4 but the index holds 32"):
            backend.add([odd])
        assert backend.n_pages == 0

    def test_queries_the_engine_cannot_score_are_refused_with_a_clear_message(self, engine):
        backend = QdrantBackend(engine(DIM, "none"))
        backend.add(to_legacy_pages(Pipeline().compress(_corpus(n_docs=5))))
        empty = MultiVectorCorpus.from_arrays([_queries(n=1).vectors, np.zeros((0, DIM), dtype=np.float32)])
        with pytest.raises(ValueError, match="query 1 has no vectors"):
            backend.search_many(empty, 3)
        nan = np.array(_queries(n=2).vectors)
        nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            backend.search_many(_queries(n=2).with_vectors(nan), 3)
        with pytest.raises(ValueError, match="dimension 8"):
            backend.search_many(_queries(dim=8), 3)
        for k in (0, -1):
            with pytest.raises(ValueError):
                backend.search_many(_queries(n=1), k)
        with pytest.raises(TypeError):
            backend.search_many(_queries(n=1), 2.5)

    def test_search_many_is_the_single_query_search_per_query(self, engine):
        backend = QdrantBackend(engine(DIM, "none"))
        backend.add(to_legacy_pages(Pipeline().compress(_corpus(n_docs=10))))
        queries = _queries(n=4)
        r = backend.search_many(queries, 5)
        for q in range(4):
            hits = backend.search(queries[q].vectors, top_k=5)
            assert r.ids[q] == [h.ref.page_id for h in hits]
            np.testing.assert_array_equal(r.scores[q], np.array([h.score for h in hits], dtype=np.float32))

    def test_a_page_added_again_overwrites_it(self, engine):
        backend = QdrantBackend(engine(DIM, "none"))
        pages = to_legacy_pages(Pipeline().compress(_corpus(n_docs=5)))
        backend.add(pages)
        backend.add(pages[:2])
        assert backend.n_pages == 5 and backend.info.upsert
        assert all(len(set(row)) == len(row) for row in backend.search_many(_queries(n=3), 5).ids)

    def test_a_reopened_local_collection_answers_the_same_ids(self, engine, tmp_path):
        from optivision.index.qdrant_index import QdrantIndex

        backend = QdrantBackend(engine(DIM, "none", name="reopen"))
        backend.add(to_legacy_pages(Pipeline().compress(_corpus(n_docs=8))))
        before = backend.search_many(_queries(n=3), 4)
        backend.save()
        backend.close()
        again = QdrantBackend(QdrantIndex(path=tmp_path / "reopen", dim=DIM, method="none"))
        try:
            assert again.n_pages == 8 and again.search_many(_queries(n=3), 4).ids == before.ids
            assert again.stats()["n_pages"] == 8
        finally:
            again.close()

    def test_the_wrappers_never_import_qdrant_client_themselves(self):
        import ast

        for path in sorted((Path(__file__).resolve().parents[1] / "src" / "optivision" / "backends").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                assert not any(n.split(".")[0] == "qdrant_client" for n in names), path.name
