from __future__ import annotations

import numpy as np
import pytest

from optivision.compression import (
    INT8_SCALE,
    Compressor,
    _int8_codes,
    code_nbytes,
    decode,
    decode_lloyd2,
    encode_lloyd2,
    fit_lloyd2,
    lloyd2_code_nbytes,
    maxsim_asymmetric,
    maxsim_asymmetric_batch,
    maxsim_asymmetric_lloyd2,
    maxsim_asymmetric_lloyd2_batch,
    maxsim_float,
    pack2,
    pack_bits,
    rotation_matrix,
    unpack2,
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


class TestInt8Scale:
    """int8 must spend the range it paid for.

    These vectors are L2-normalised, so components concentrate near
    1/sqrt(dim) and never approach 1.0. Quantizing as if they spanned
    [-1, 1] wastes most of the 255 available levels.
    """

    def _unit_vectors(self, rng, n=200, d=128):
        v = rng.standard_normal((n, d)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def test_uses_most_of_the_int8_range(self, rng):
        v = self._unit_vectors(rng)
        codes = _int8_codes(v).view(np.int8)
        # Naive round(v * 127) peaks near level 46 on unit-norm 128-d vectors.
        assert np.abs(codes).max() > 80

    def test_round_trip_beats_the_unscaled_quantizer(self, rng):
        v = self._unit_vectors(rng)
        scaled = decode(_int8_codes(v).view(np.uint8), 128, "int8")
        naive = np.clip(np.round(v * 127.0), -127, 127).astype(np.int8).astype(np.float32) / 127.0

        # Halving the full-scale value halves the quantization step, so the
        # mean absolute error halves and the squared error — what actually
        # moves a cosine — falls ~4x.
        assert np.abs(scaled - v).mean() < np.abs(naive - v).mean() / 1.9

        def one_minus_cos(x):
            return 1.0 - float(
                np.mean(np.sum(x * v, 1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(v, axis=1)))
            )

        assert one_minus_cos(scaled) < one_minus_cos(naive) / 3.5

    def test_does_not_clip_the_tail(self, rng):
        """A scale below the data's range would put a hard error on the
        largest — most informative — components."""
        v = self._unit_vectors(rng, n=2000)
        assert np.abs(v).max() < INT8_SCALE
        assert np.abs(decode(_int8_codes(v).view(np.uint8), 128, "int8") - v).max() < 0.01

    def test_storage_is_still_exactly_4x(self, rng):
        v = self._unit_vectors(rng, n=40)
        assert _int8_codes(v).view(np.uint8).nbytes == 40 * 128 == v.nbytes // 4


class TestCodebookSaliency:
    """Retrieval-space saliency: which patches win against probe directions."""

    def test_fitted_probes_are_unit_vectors_and_reproducible(self):
        from optivision.pruning import fit_codebook

        rng = np.random.default_rng(0)
        sample = _unit(rng, 500, 32)
        a = fit_codebook(sample, size=16, seed=7)
        b = fit_codebook(sample, size=16, seed=7)
        assert a.shape == (16, 32)
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
        assert np.array_equal(a, b), "same seed must give the same probes"

    def test_random_source_differs_from_kmeans(self):
        from optivision.pruning import fit_codebook

        rng = np.random.default_rng(0)
        sample = _unit(rng, 300, 32)
        km = fit_codebook(sample, size=8, seed=7, source="kmeans")
        rand = fit_codebook(sample, size=8, seed=7, source="random")
        assert not np.allclose(km, rand), "the control must not be the fitted codebook"

    def test_saliency_favours_patches_that_win_probes(self):
        """A patch that wins a probe must outscore every patch that wins none.

        Not that it outscores *everything*: another patch may win two probes and
        outrank it, which is the metric working rather than failing.
        """
        from optivision.pruning import codebook_saliency

        rng = np.random.default_rng(1)
        probes = _unit(rng, 4, 16)
        patches = _unit(rng, 9, 16)
        patches[3] = probes[0]  # exact match: wins probe 0 outright
        sal = codebook_saliency(patches, probes, 3, 3)
        assert sal.shape == (3, 3)

        flat = sal.reshape(-1)
        winners = set(np.asarray(probes @ patches.T).argmax(axis=1).tolist())
        assert 3 in winners
        losers = [i for i in range(9) if i not in winners]
        assert min(flat[i] for i in winners) > max(flat[i] for i in losers)

    def test_saliency_is_bounded_and_breaks_ties(self):
        from optivision.pruning import codebook_saliency

        rng = np.random.default_rng(2)
        sal = codebook_saliency(_unit(rng, 16, 16), _unit(rng, 3, 16), 4, 4).reshape(-1)
        assert sal.min() >= 0.0 and sal.max() <= 1.0
        # At most 3 patches can win a probe; the rest are separated only by the
        # tie-break, which must still order them rather than leaving a flat zero.
        losers = np.sort(sal)[:-3]
        assert len(set(losers.tolist())) > 1, "tie-break must order the non-winners"


def _fit(rng, n=2000, d=128, seed=7):
    v = _unit(rng, n, d)
    return v, fit_lloyd2(v, seed=seed)


class TestLloyd2Packing:
    """Rotated 2-bit Lloyd-Max: pack2/unpack2 round trip and byte size."""

    def test_roundtrip_preserves_indices(self, rng):
        idx = rng.integers(0, 4, size=(20, 37)).astype(np.uint8)
        back = unpack2(pack2(idx), 37)
        assert np.array_equal(back, idx)

    def test_size_is_two_bits_per_dimension(self, rng):
        v = _unit(rng, 10, 128)
        codec = fit_lloyd2(v, seed=1)
        codes = encode_lloyd2(v, codec)
        assert codes.shape == (10, 32)
        assert codes.nbytes * 16 == v.nbytes  # exactly 16x smaller

    @pytest.mark.parametrize("dim", [5, 6, 7, 96, 128, 130])
    def test_non_multiple_of_four_dims(self, rng, dim):
        v = _unit(rng, 5, dim)
        codec = fit_lloyd2(v, seed=1)
        codes = encode_lloyd2(v, codec)
        assert codes.shape == (5, lloyd2_code_nbytes(dim))
        decoded = decode_lloyd2(codes, dim, codec)
        assert decoded.shape == (5, dim)

    def test_rejects_1d_input(self, rng):
        codec = fit_lloyd2(_unit(rng, 10, 8), seed=1)
        with pytest.raises(ValueError):
            encode_lloyd2(np.zeros(8, dtype=np.float32), codec)


class TestLloyd2Fit:
    def test_reproducible_given_seed(self, rng):
        v = _unit(rng, 500, 32)
        a = fit_lloyd2(v, seed=7)
        b = fit_lloyd2(v, seed=7)
        assert np.array_equal(a.mu, b.mu)
        assert a.sigma == b.sigma
        assert np.array_equal(a.rotation, b.rotation)

    def test_rotation_is_orthonormal_and_needs_no_data(self):
        r7 = rotation_matrix(64, seed=7)
        r7_again = rotation_matrix(64, seed=7)
        r3 = rotation_matrix(64, seed=3)
        assert np.allclose(r7 @ r7.T, np.eye(64), atol=1e-4)
        assert np.array_equal(r7, r7_again), "same (dim, seed) must give the same rotation"
        assert not np.allclose(r7, r3), "different seeds must give different rotations"

    def test_overhead_is_mu_plus_one_scalar(self, rng):
        _, codec = _fit(rng, d=128)
        assert codec.overhead_bytes == 128 * 4 + 4


class TestLloyd2MaxSim:
    def test_preserves_self_as_best_match(self, rng):
        pages = [_unit(rng, 40, 128) for _ in range(12)]
        codec = fit_lloyd2(np.concatenate(pages), seed=7)
        query = pages[7][:8]
        scores = [
            maxsim_asymmetric_lloyd2(query, encode_lloyd2(p, codec), 128, codec) for p in pages
        ]
        assert int(np.argmax(scores)) == 7

    def test_batch_matches_per_page(self, rng):
        pages = [_unit(rng, n, 32) for n in (5, 8, 3)]
        codec = fit_lloyd2(np.concatenate(pages), seed=7)
        codes = np.concatenate([encode_lloyd2(p, codec) for p in pages])
        offsets = np.array([0, 5, 13, 16], dtype=np.int64)
        q = _unit(rng, 6, 32)
        batch = maxsim_asymmetric_lloyd2_batch(q, codes, 32, offsets, codec)
        one_by_one = [
            maxsim_asymmetric_lloyd2(q, encode_lloyd2(p, codec), 32, codec) for p in pages
        ]
        assert np.allclose(batch, one_by_one, atol=1e-4)

    def test_empty_document_scores_zero(self, rng):
        q = _unit(rng, 4, 16)
        codec = fit_lloyd2(_unit(rng, 50, 16), seed=7)
        assert maxsim_asymmetric_lloyd2(q, np.zeros((0, 4), dtype=np.uint8), 16, codec) == 0.0

    def test_ranking_tracks_float_ranking(self, rng):
        """Same construction as the binary codec's equivalent test: plant known
        signal, then check compression shifts scores more than it reorders them."""
        from optivision.metrics import rank_correlation

        n_query = 8
        query = _unit(rng, n_query, 128)
        pages = []
        for i in range(n_query + 1):
            page = _unit(rng, 30, 128)
            for j in range(i):
                noise = rng.standard_normal(128).astype(np.float32)
                mixed = query[j] + 0.35 * (noise / np.linalg.norm(noise))
                page[j] = mixed / np.linalg.norm(mixed)
            pages.append(page)
        codec = fit_lloyd2(np.concatenate(pages), seed=7)

        float_order = list(np.argsort([-maxsim_float(query, p) for p in pages]))
        assert float_order[0] == n_query

        lloyd2_order = list(
            np.argsort(
                [-maxsim_asymmetric_lloyd2(query, encode_lloyd2(p, codec), 128, codec) for p in pages]
            )
        )
        tau = rank_correlation([str(i) for i in float_order], [str(i) for i in lloyd2_order])
        assert tau > 0.6

    def test_reconstruction_error_beats_one_bit_sign(self, rng):
        """2-bit should approximate the float vector much more closely than the
        1-bit sign codec it sits above -- the whole point of spending 2 bits."""
        v = _unit(rng, 300, 128)
        codec = fit_lloyd2(v, seed=7)
        lloyd2_err = np.abs(decode(encode_lloyd2(v, codec), 128, "lloyd2", codec=codec) - v).mean()
        sign_err = np.abs(unpack_signs(pack_bits(v), 128) - v).mean()
        assert lloyd2_err < sign_err


class TestCompressorLloyd2:
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

    def test_lloyd2_is_16x_smaller_than_raw(self, rng):
        page = self._page(rng)
        codec = fit_lloyd2(page.embeddings, seed=7)
        out = Compressor(CompressionConfig(method="lloyd2"), codec=codec).compress(page)
        assert out.nbytes == 40 * 32
        assert out.raw_nbytes() == 1024 * 128 * 4

    def test_lloyd2_bytes_sit_between_binary_and_int8(self, rng):
        page = self._page(rng)
        codec = fit_lloyd2(page.embeddings, seed=7)
        binary = Compressor(CompressionConfig(method="binary")).compress(page)
        lloyd2 = Compressor(CompressionConfig(method="lloyd2"), codec=codec).compress(page)
        int8 = Compressor(CompressionConfig(method="int8")).compress(page)
        assert binary.nbytes < lloyd2.nbytes < int8.nbytes

    def test_lloyd2_without_a_fitted_codec_raises(self, rng):
        page = self._page(rng)
        with pytest.raises(ValueError, match="fitted codec"):
            Compressor(CompressionConfig(method="lloyd2")).compress(page)

    def test_decode_matches_module_function(self, rng):
        page = self._page(rng)
        codec = fit_lloyd2(page.embeddings, seed=7)
        out = Compressor(CompressionConfig(method="lloyd2"), codec=codec).compress(page)
        direct = encode_lloyd2(page.embeddings, codec)
        assert np.array_equal(out.codes, direct)
        assert np.allclose(decode(out.codes, 128, "lloyd2", codec=codec), decode(direct, 128, "lloyd2", codec=codec))

    def test_decode_without_codec_raises(self, rng):
        page = self._page(rng)
        codec = fit_lloyd2(page.embeddings, seed=7)
        out = Compressor(CompressionConfig(method="lloyd2"), codec=codec).compress(page)
        with pytest.raises(ValueError):
            decode(out.codes, 128, "lloyd2")
