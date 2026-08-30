from __future__ import annotations

import numpy as np
import pytest

from optivision.compression import Compressor, fit_lloyd2, maxsim_asymmetric
from optivision.config import CompressionConfig
from optivision.index.numpy_index import NumpyIndex
from optivision.types import PageRef, PatchGrid, PrunedPage


def _unit(rng, n, d=32):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _page(rng, doc_id, n=10, d=32, before=100):
    vectors = _unit(rng, n, d)
    pruned = PrunedPage(
        ref=PageRef(doc_id=doc_id, page_no=1),
        embeddings=vectors,
        kept_token_index=np.arange(n, dtype=np.int32),
        keep_mask=np.ones((1, n), dtype=bool),
        grid=PatchGrid.contiguous(1, n),
        n_tokens_before=before,
    )
    return Compressor(CompressionConfig(method="binary")).compress(pruned), vectors


def _lloyd2_page(rng, doc_id, codec, n=10, d=32, before=100):
    vectors = _unit(rng, n, d)
    pruned = PrunedPage(
        ref=PageRef(doc_id=doc_id, page_no=1),
        embeddings=vectors,
        kept_token_index=np.arange(n, dtype=np.int32),
        keep_mask=np.ones((1, n), dtype=bool),
        grid=PatchGrid.contiguous(1, n),
        n_tokens_before=before,
    )
    return Compressor(CompressionConfig(method="lloyd2"), codec=codec).compress(pruned), vectors


class TestNumpyIndex:
    def test_scores_match_direct_maxsim(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        pages, _ = zip(*[_page(rng, f"d{i}") for i in range(6)])
        idx.add(list(pages))
        q = _unit(rng, 5, 32)
        scores = idx.score_all(q)
        expected = [maxsim_asymmetric(q, p.codes, 32) for p in pages]
        assert np.allclose(scores, expected, atol=1e-4)

    def test_variable_length_pages(self, rng, tmp_path):
        """reduceat segment boundaries are easy to get wrong with uneven pages."""
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        pages = [_page(rng, f"d{i}", n=n)[0] for i, n in enumerate([1, 17, 4, 33, 2])]
        idx.add(pages)
        q = _unit(rng, 3, 32)
        expected = [maxsim_asymmetric(q, p.codes, 32) for p in pages]
        assert np.allclose(idx.score_all(q), expected, atol=1e-4)

    def test_incremental_add_equals_single_add(self, rng, tmp_path):
        pages = [_page(rng, f"d{i}")[0] for i in range(5)]
        a = NumpyIndex(tmp_path / "a", dim=32)
        a.add(pages)
        b = NumpyIndex(tmp_path / "b", dim=32)
        for p in pages:
            b.add([p])
        q = _unit(rng, 4, 32)
        assert np.allclose(a.score_all(q), b.score_all(q))

    def test_search_is_ordered_and_ranked(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        idx.add([_page(rng, f"d{i}")[0] for i in range(10)])
        hits = idx.search(_unit(rng, 4, 32), top_k=4)
        assert [h.rank for h in hits] == [1, 2, 3, 4]
        assert all(hits[i].score >= hits[i + 1].score for i in range(3))

    def test_top_k_larger_than_corpus(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        idx.add([_page(rng, "only")[0]])
        assert len(idx.search(_unit(rng, 2, 32), top_k=50)) == 1

    def test_empty_index_returns_nothing(self, tmp_path, rng):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        assert idx.search(_unit(rng, 2, 32), top_k=5) == []
        assert idx.n_pages == 0

    def test_save_load_roundtrip(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        idx.add([_page(rng, f"d{i}")[0] for i in range(7)])
        q = _unit(rng, 4, 32)
        before = idx.score_all(q)
        idx.save()

        reloaded = NumpyIndex.load(tmp_path / "idx")
        assert reloaded.n_pages == 7
        assert np.allclose(reloaded.score_all(q), before)
        assert [r.page_id for r in reloaded.refs] == [r.page_id for r in idx.refs]

    def test_stats_accounting(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        idx.add([_page(rng, f"d{i}", n=10, d=32, before=100)[0] for i in range(4)])
        s = idx.stats()
        assert s["n_pages"] == 4
        assert s["n_vectors"] == 40
        assert s["tokens_before"] == 400
        assert s["token_reduction"] == pytest.approx(10.0)
        # 100 tokens x 32 dims x 4 bytes raw, vs 10 vectors x 4 bytes packed
        assert s["compression_ratio"] == pytest.approx((100 * 32 * 4) / (10 * 4))


class TestBoundedMemoryScoring:
    """The search path must not re-expand the index it just compressed.

    Decoding every packed code to float32 at once costs 32x the binary index,
    which is the entire saving the pipeline exists to produce. These tests pin
    that the streaming path stays bounded *and* returns identical scores.
    """

    def test_streaming_matches_cached_scores(self, rng, tmp_path):
        pages = [_page(rng, f"d{i}", n=n)[0] for i, n in enumerate([7, 3, 11, 5, 9, 2])]
        cached = NumpyIndex(tmp_path / "a", dim=32)
        cached.add(pages)
        streamed = NumpyIndex(tmp_path / "b", dim=32, max_decoded_bytes=256)
        streamed.add(pages)
        q = _unit(rng, 4, 32)
        assert np.array_equal(streamed.score_all(q), cached.score_all(q))

    def test_streaming_never_materialises_the_full_matrix(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32, max_decoded_bytes=256)
        idx.add([_page(rng, f"d{i}")[0] for i in range(8)])
        assert idx.decoded_nbytes > idx.max_decoded_bytes  # streaming regime
        idx.score_all(_unit(rng, 3, 32))
        assert idx._decoded is None

    def test_block_boundaries_do_not_split_a_page(self, rng, tmp_path):
        """A page wider than one block must still be scored as one segment."""
        pages = [_page(rng, f"d{i}", n=n)[0] for i, n in enumerate([1, 40, 2, 33, 1])]
        cached = NumpyIndex(tmp_path / "a", dim=32)
        cached.add(pages)
        q = _unit(rng, 3, 32)
        expected = cached.score_all(q)
        for budget in (128, 256, 512, 4096):
            idx = NumpyIndex(tmp_path / f"b{budget}", dim=32, max_decoded_bytes=budget)
            idx.add(pages)
            assert np.allclose(idx.score_all(q), expected, atol=1e-5)

    def test_search_agrees_across_both_regimes(self, rng, tmp_path):
        pages = [_page(rng, f"d{i}")[0] for i in range(12)]
        cached = NumpyIndex(tmp_path / "a", dim=32)
        cached.add(pages)
        streamed = NumpyIndex(tmp_path / "b", dim=32, max_decoded_bytes=256)
        streamed.add(pages)
        q = _unit(rng, 4, 32)
        assert [h.ref.page_id for h in streamed.search(q, top_k=5)] == [
            h.ref.page_id for h in cached.search(q, top_k=5)
        ]

    def test_budget_survives_save_load(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        idx.add([_page(rng, f"d{i}")[0] for i in range(5)])
        idx.save()
        q = _unit(rng, 3, 32)
        reloaded = NumpyIndex.load(tmp_path / "idx", max_decoded_bytes=256)
        assert reloaded.max_decoded_bytes == 256
        assert np.allclose(reloaded.score_all(q), idx.score_all(q))


class TestLloyd2Index:
    """The rotated 2-bit codec needs a fitted codec threaded through the index,
    unlike binary/int8 -- these pin that the extra state round-trips."""

    def test_scores_match_direct_maxsim(self, rng, tmp_path):
        from optivision.compression import maxsim_asymmetric_lloyd2

        codec = fit_lloyd2(_unit(rng, 300, 32), seed=7)
        idx = NumpyIndex(tmp_path / "idx", dim=32, method="lloyd2", codec=codec)
        pages, _ = zip(*[_lloyd2_page(rng, f"d{i}", codec) for i in range(6)])
        idx.add(list(pages))
        q = _unit(rng, 5, 32)
        scores = idx.score_all(q)
        expected = [maxsim_asymmetric_lloyd2(q, p.codes, 32, codec) for p in pages]
        assert np.allclose(scores, expected, atol=1e-4)

    def test_save_load_roundtrip_carries_the_codec(self, rng, tmp_path):
        codec = fit_lloyd2(_unit(rng, 300, 32), seed=7)
        idx = NumpyIndex(tmp_path / "idx", dim=32, method="lloyd2", codec=codec)
        idx.add([_lloyd2_page(rng, f"d{i}", codec)[0] for i in range(7)])
        q = _unit(rng, 4, 32)
        before = idx.score_all(q)
        idx.save()

        reloaded = NumpyIndex.load(tmp_path / "idx")
        assert reloaded.codec is not None
        assert np.array_equal(reloaded.codec.mu, codec.mu)
        assert reloaded.codec.sigma == pytest.approx(codec.sigma)
        assert np.allclose(reloaded.score_all(q), before)

    def test_stats_amortize_the_codec_overhead(self, rng, tmp_path):
        codec = fit_lloyd2(_unit(rng, 300, 32), seed=7)
        idx = NumpyIndex(tmp_path / "idx", dim=32, method="lloyd2", codec=codec)
        idx.add([_lloyd2_page(rng, f"d{i}", codec, n=10, d=32)[0] for i in range(4)])
        s = idx.stats()
        # 4 pages x 10 vectors x 8 bytes/vector (2 bits x 32 dims) + shared mu/sigma
        assert s["index_bytes"] == 4 * 10 * 8 + codec.overhead_bytes
