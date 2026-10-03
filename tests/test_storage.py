from __future__ import annotations

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.index.numpy_index import NumpyIndex
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, rank, segment_reduce
from optivision.stages import AdaptiveMerge, BinaryQuantizer, Int8Quantizer, PCAProjector
from optivision.storage import ExactIndex, NpzStorage, TieredIndex, to_legacy_pages


def _unit(v):
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def data(rng):
    docs = MultiVectorCorpus.from_arrays([_unit(rng.standard_normal((rng.integers(3, 12), 16))) for _ in range(30)],
                                         ids=[f"doc{i}::p1" for i in range(30)])
    queries = MultiVectorCorpus.from_arrays([_unit(rng.standard_normal((4, 16))) for _ in range(7)])
    return docs, queries


def test_exact_index_matches_the_scorer(data):
    docs, queries = data
    top, scores = ExactIndex(docs).search(queries, k=5)
    full = maxsim_matrix(queries, docs)
    np.testing.assert_array_equal(top, rank(full, 5))
    np.testing.assert_allclose(scores, np.take_along_axis(full, top, axis=1))


def test_rescore_equals_full_scores_on_the_candidates(data, rng):
    docs, queries = data
    cand = np.stack([rng.choice(len(docs), 6, replace=False) for _ in range(len(queries))])
    got = ExactIndex(docs).rescore(queries, cand)
    np.testing.assert_allclose(got, np.take_along_axis(maxsim_matrix(queries, docs), cand, axis=1), rtol=1e-5)


def test_rescore_applies_the_query_projection(data):
    docs, queries = data
    pipe = Pipeline([PCAProjector(dim=8)]).fit(docs)
    comp = pipe.compress(docs)
    cand = np.tile(np.arange(5), (len(queries), 1))
    got = ExactIndex(comp, query_transform=pipe).rescore(queries, cand)
    want = maxsim_matrix(pipe.transform_queries(queries), comp)[:, :5]
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_tiered_with_every_document_shortlisted_is_the_cold_ranking(data):
    docs, queries = data
    hot_pipe = Pipeline([AdaptiveMerge(radius=0.2), BinaryQuantizer()])
    tiered = TieredIndex(ExactIndex(hot_pipe.compress(docs)), ExactIndex(docs), candidates=len(docs))
    np.testing.assert_array_equal(rank(tiered.score(queries)), rank(maxsim_matrix(queries, docs)))


def test_tiered_results_come_from_the_hot_shortlist(data):
    docs, queries = data
    hot = ExactIndex(Pipeline([BinaryQuantizer()]).compress(docs))
    tiered = TieredIndex(hot, ExactIndex(docs), candidates=4)
    top, _ = tiered.search(queries, k=4)
    shortlist = rank(hot.score(queries), 4)
    for a, b in zip(top, shortlist, strict=True):
        assert set(a) == set(b)
    assert set(tiered.last_timing) == {"hot_seconds", "rescore_seconds"}


def test_npz_storage_roundtrip(data, tmp_path):
    docs, _ = data
    store = NpzStorage(tmp_path)
    comp = Pipeline([Int8Quantizer("per_vector")]).compress(docs)
    store.write("a", comp)
    assert store.names() == ["a"]
    np.testing.assert_array_equal(store.read("a").codes, comp.codes)


def test_legacy_pages_feed_the_original_numpy_index(data, tmp_path):
    docs, queries = data
    comp = Pipeline([BinaryQuantizer()]).compress(docs)
    idx = NumpyIndex(tmp_path, dim=16, method="binary")
    idx.add(to_legacy_pages(comp))
    full = maxsim_matrix(queries, comp)
    for qi in range(len(queries)):
        np.testing.assert_allclose(idx.score_all(queries[qi].vectors), full[qi], rtol=1e-5)
    with pytest.raises(NotImplementedError):
        to_legacy_pages(Pipeline([Int8Quantizer("per_vector")]).compress(docs))


# ---------------------------------------------------------------- batched rescore
# ExactIndex.rescore decodes each candidate document once and scores every query that
# listed it. The previous body (one decode + product per query) is kept here as the
# reference: the batched version must give the same scores, bit for bit.


def _rescore_per_query(index, queries, candidates):
    q = index._q(queries)
    cand = np.asarray(candidates, dtype=np.int64)
    out = np.full(cand.shape, -np.inf, dtype=np.float32)
    offsets = index.store.offsets
    for qi in range(cand.shape[0]):
        lo_q, hi_q = int(q.offsets[qi]), int(q.offsets[qi + 1])
        if hi_q == lo_q:
            out[qi] = 0.0
            continue
        qv = q.decode_rows(lo_q, hi_q)
        spans = [(int(offsets[d]), int(offsets[d + 1])) for d in cand[qi]]
        blocks = [index.store.decode_rows(lo, hi) for lo, hi in spans]
        counts = np.array([b.shape[0] for b in blocks], dtype=np.int64)
        if counts.sum() == 0:
            continue
        local = np.concatenate([[0], np.cumsum(counts)])
        per_token = segment_reduce(qv @ np.concatenate(blocks, axis=0).T, local, np.maximum, axis=1, fill=-np.inf)
        out[qi] = per_token.sum(axis=0)
    return out


@pytest.mark.parametrize("store_kind", ["float", "int8", "binary"])
@pytest.mark.parametrize("m", [1, 6, 30])
def test_batched_rescore_is_bitwise_the_per_query_rescore(data, rng, store_kind, m):
    docs, queries = data
    store = {"float": docs, "int8": Pipeline([Int8Quantizer("per_vector")]).compress(docs),
             "binary": Pipeline([BinaryQuantizer()]).compress(docs)}[store_kind]
    idx = ExactIndex(store)
    cand = np.stack([rng.choice(len(docs), m, replace=False) for _ in range(len(queries))])
    got = idx.rescore(queries, cand)
    assert got.dtype == np.float32 and got.shape == cand.shape
    np.testing.assert_array_equal(got, _rescore_per_query(idx, queries, cand))


def test_batched_rescore_keeps_column_order_and_duplicates(data):
    docs, queries = data
    idx = ExactIndex(docs)
    cand = np.tile(np.array([3, 3, 0, 7]), (len(queries), 1))
    got = idx.rescore(queries, cand)
    np.testing.assert_array_equal(got[:, 0], got[:, 1])
    np.testing.assert_array_equal(got, _rescore_per_query(idx, queries, cand))
    np.testing.assert_array_equal(got[:, 2], idx.rescore(queries, cand[:, 2:3])[:, 0])


def test_batched_rescore_edge_cases(rng):
    d = 8
    docs = MultiVectorCorpus(_unit(rng.standard_normal((10, d))), np.array([0, 4, 4, 10]))  # doc 1 has no vectors
    qv = _unit(rng.standard_normal((5, d)))
    queries = MultiVectorCorpus(qv, np.array([0, 3, 3, 5]))  # query 1 has no tokens
    idx = ExactIndex(docs)
    cand = np.array([[0, 1, 2], [2, 0, 1], [1, 1, 1]])
    got = idx.rescore(queries, cand)
    np.testing.assert_array_equal(got, _rescore_per_query(idx, queries, cand))
    assert np.all(got[1] == 0.0)  # no tokens: 0, as before
    assert np.isneginf(got[0, 1]) and np.all(np.isneginf(got[2]))  # empty document: -inf
    empty = idx.rescore(queries, np.zeros((3, 0), dtype=np.int64))
    assert empty.shape == (3, 0) and empty.dtype == np.float32


def test_batched_rescore_chunks_within_the_block_budget(data, rng):
    docs, queries = data
    cand = np.stack([rng.choice(len(docs), 8, replace=False) for _ in range(len(queries))])
    small = ExactIndex(docs, max_block_bytes=64)  # forces one query token per product
    got, want = small.rescore(queries, cand), _rescore_per_query(small, queries, cand)
    # Not bitwise: BLAS takes a different kernel for products with very few rows (a
    # one-row product is a matrix-vector call), so the last bit can differ. The old
    # per-query code had the same shape dependence for very short queries.
    np.testing.assert_allclose(got, want, rtol=1e-6)
    np.testing.assert_array_equal(rank(got), rank(want))


def test_batched_rescore_with_projection_and_reload(data, tmp_path):
    docs, queries = data
    pipe = Pipeline([PCAProjector(dim=8)]).fit(docs)
    comp = pipe.compress(docs)
    cand = np.tile(np.arange(len(docs))[::-3][:6], (len(queries), 1))
    idx = ExactIndex(comp, query_transform=pipe)
    np.testing.assert_array_equal(idx.rescore(queries, cand), _rescore_per_query(idx, queries, cand))
    cold = Pipeline([Int8Quantizer("per_vector")]).compress(docs)
    NpzStorage(tmp_path).write("cold", cold)
    reloaded = ExactIndex(NpzStorage(tmp_path).read("cold"))
    np.testing.assert_array_equal(reloaded.rescore(queries, cand), ExactIndex(cold).rescore(queries, cand))


def test_tiered_ranking_and_ndcg_unchanged_by_batching(data):
    docs, queries = data
    from optivision.evaluation import per_query_metrics

    class PerQuery(ExactIndex):
        def rescore(self, q, c):
            return _rescore_per_query(self, q, c)

    hot = ExactIndex(Pipeline([BinaryQuantizer()]).compress(docs))
    cold = Pipeline([Int8Quantizer("per_vector")]).compress(docs)
    rel = [np.array([i % len(docs)]) for i in range(len(queries))]
    for m in (1, 5, len(docs)):
        new = TieredIndex(hot, ExactIndex(cold), candidates=m).score(queries)
        old = TieredIndex(hot, PerQuery(cold), candidates=m).score(queries)
        np.testing.assert_array_equal(rank(new), rank(old))
        np.testing.assert_array_equal(per_query_metrics(new, rel, ["ndcg@5"])["ndcg@5"],
                                      per_query_metrics(old, rel, ["ndcg@5"])["ndcg@5"])
