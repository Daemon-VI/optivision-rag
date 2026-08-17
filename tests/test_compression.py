from __future__ import annotations

import numpy as np
import pytest

from optivision.compression import (
    Compressor,
    code_nbytes,
    decode,
    maxsim_asymmetric,
    maxsim_asymmetric_batch,
    maxsim_float,
    pack_bits,
    unpack_signs,
)
from optivision.config import CompressionConfig
from optivision.types import PageRef, PatchGrid, PrunedPage


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestBitPacking:
    def test_roundtrip_preserves_signs(self, rng):
        v = _unit(rng, 32, 128)
        back = unpack_signs(pack_bits(v), 128)
        assert np.array_equal(np.sign(back), np.sign(np.where(v > 0, 1.0, -1.0)))

    def test_size_is_one_bit_per_dimension(self, rng):
        v = _unit(rng, 10, 128)
        codes = pack_bits(v)
        assert codes.shape == (10, 16)
        assert codes.nbytes * 32 == v.nbytes  # exactly 32x smaller

    @pytest.mark.parametrize("dim", [8, 96, 128, 130])
    def test_non_multiple_of_eight_dims(self, rng, dim):
        v = _unit(rng, 5, dim)
        back = unpack_signs(pack_bits(v), dim)
        assert back.shape == (5, dim)
        assert code_nbytes(dim) == pack_bits(v).shape[1]

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            pack_bits(np.zeros(8, dtype=np.float32))


class TestMaxSim:
    def test_float_maxsim_matches_manual(self, rng):
        q, d = _unit(rng, 4, 16), _unit(rng, 9, 16)
        expected = sum(max(float(qi @ dj) for dj in d) for qi in q)
        assert maxsim_float(q, d) == pytest.approx(expected, rel=1e-5)

    def test_empty_document_scores_zero(self, rng):
        q = _unit(rng, 4, 16)
        assert maxsim_float(q, np.zeros((0, 16), dtype=np.float32)) == 0.0
        assert maxsim_asymmetric(q, np.zeros((0, 2), dtype=np.uint8), 16) == 0.0

    def test_batch_matches_per_page(self, rng):
        pages = [_unit(rng, n, 32) for n in (5, 8, 3)]
        codes = np.concatenate([pack_bits(p) for p in pages])
        offsets = np.array([0, 5, 13, 16], dtype=np.int64)
        q = _unit(rng, 6, 32)
        batch = maxsim_asymmetric_batch(q, codes, 32, offsets)
        one_by_one = [maxsim_asymmetric(q, pack_bits(p), 32) for p in pages]
        assert np.allclose(batch, one_by_one, atol=1e-4)

    def test_binary_preserves_self_as_best_match(self, rng):
        """A page must still rank itself top after quantization."""
        pages = [_unit(rng, 40, 128) for _ in range(12)]
        query = pages[7][:8]  # a slice of page 7 acts as its own query
        scores = [maxsim_asymmetric(query, pack_bits(p), 128) for p in pages]
        assert int(np.argmax(scores)) == 7

    def test_binary_ranking_tracks_float_ranking(self, rng):
        """Quantization should shift scores far more than it reorders them.

        The pages must actually differ in relevance for there to be a ranking
        worth preserving: over i.i.d. random pages every MaxSim score is within
        noise of every other, so *any* perturbation reshuffles the order and the
        test would measure nothing. Here page i plants i query-derived vectors,
        which gives a known-correct ordering.
        """
        from optivision.metrics import rank_correlation

        n_query = 8
        query = _unit(rng, n_query, 128)
        pages = []
        # Plant each query vector at most once: MaxSim takes a max, so planting
        # duplicates of an already-planted vector adds nothing and would leave
        # the top pages genuinely tied.
        for i in range(n_query + 1):
            page = _unit(rng, 30, 128)
            for j in range(i):
                # Scale the noise as a *unit* vector, not per-dimension: in 128
                # dims a raw 0.35*randn has norm ~4 and would drown the signal.
                noise = rng.standard_normal(128).astype(np.float32)
                mixed = query[j] + 0.35 * (noise / np.linalg.norm(noise))
                page[j] = mixed / np.linalg.norm(mixed)
            pages.append(page)

        float_order = list(np.argsort([-maxsim_float(query, p) for p in pages]))
        assert float_order[0] == n_query  # most planted vectors wins, as constructed

        bin_order = list(np.argsort([-maxsim_asymmetric(query, pack_bits(p), 128) for p in pages]))
        assert len(set(float_order[:3]) & set(bin_order[:3])) >= 2
        tau = rank_correlation([str(i) for i in float_order], [str(i) for i in bin_order])
        assert tau > 0.6


class TestCompressor:
    def _page(self, rng, n=40, d=128):
        vectors = _unit(rng, n, d)
        return PrunedPage(
            ref=PageRef(doc_id="d", page_no=1),
            embeddings=vectors,
            kept_token_index=np.arange(n, dtype=np.int32),
            keep_mask=np.ones((1, n), dtype=bool),
            grid=PatchGrid.contiguous(1, n),
            n_tokens_before=1024,
        )

    def test_binary_is_32x_smaller_than_raw(self, rng):
        page = self._page(rng)
        out = Compressor(CompressionConfig(method="binary")).compress(page)
        assert out.nbytes == 40 * 16
        assert out.raw_nbytes() == 1024 * 128 * 4

    def test_int8_is_4x(self, rng):
        page = self._page(rng)
        out = Compressor(CompressionConfig(method="int8")).compress(page)
        assert out.nbytes == 40 * 128

    def test_none_keeps_float32(self, rng):
        page = self._page(rng)
        out = Compressor(CompressionConfig(enabled=False, method="none")).compress(page)
        assert out.nbytes == 40 * 128 * 4
        assert np.allclose(decode(out.codes, 128, "none"), page.embeddings)

    def test_int8_decode_is_close(self, rng):
        page = self._page(rng)
        out = Compressor(CompressionConfig(method="int8")).compress(page)
        assert np.allclose(decode(out.codes, 128, "int8"), page.embeddings, atol=0.01)

    def test_unknown_method_raises(self, rng):
        with pytest.raises(ValueError):
            Compressor(CompressionConfig(method="ternary")).compress(self._page(rng))

    def test_keep_norm_adds_a_float_per_vector(self, rng):
        page = self._page(rng)
        out = Compressor(CompressionConfig(method="binary", keep_norm=True)).compress(page)
        assert out.scale is not None and out.scale.shape == (40,)
        assert out.nbytes == 40 * 16 + 40 * 4
