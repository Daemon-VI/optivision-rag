from __future__ import annotations

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.index.numpy_index import NumpyIndex
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, rank
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
