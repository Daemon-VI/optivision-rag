from __future__ import annotations

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, rank
from optivision.stages import (
    BinaryQuantizer,
    Int4Quantizer,
    Int8Quantizer,
    PCAProjector,
    RandomProjector,
    TruncateProjector,
    stage_from_dict,
)


def _unit(v):
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _corpus(rng, n_docs=12, dim=32, anisotropic=True):
    scale = np.linspace(2.0, 0.1, dim) if anisotropic else np.ones(dim)
    docs = [_unit(rng.standard_normal((rng.integers(5, 15), dim)) * scale + 0.3) for _ in range(n_docs)]
    return MultiVectorCorpus.from_arrays(docs)


def _queries(rng, n=6, dim=32):
    return MultiVectorCorpus.from_arrays([_unit(rng.standard_normal((4, dim))) for _ in range(n)])


class TestPCA:
    def test_full_rank_preserves_scores_exactly(self, rng):
        c, q = _corpus(rng), _queries(rng)
        p = Pipeline([PCAProjector(dim=32)])
        comp = p.fit(c).compress(c)
        np.testing.assert_allclose(maxsim_matrix(p.transform_queries(q), comp), maxsim_matrix(q, c), atol=1e-4)

    def test_output_dimension_and_explained_variance(self, rng):
        c = _corpus(rng)
        small, big = PCAProjector(dim=4).fit(c), PCAProjector(dim=16).fit(c)
        assert small.project(c.vectors).shape[1] == 4
        assert 0 < small.explained < big.explained <= 1.0

    def test_centering_does_not_change_rankings(self, rng):
        c, q = _corpus(rng), _queries(rng)
        plain = Pipeline([PCAProjector(dim=32, center=False)])
        centred = Pipeline([PCAProjector(dim=32, center=True)])
        plain.fit(c)
        centred.fit(c)
        a = maxsim_matrix(plain.transform_queries(q), plain.compress(c))
        b = maxsim_matrix(centred.transform_queries(q), centred.compress(c))
        np.testing.assert_array_equal(rank(a), rank(b))

    def test_joint_basis_needs_queries(self, rng):
        c, q = _corpus(rng), _queries(rng)
        with pytest.raises(ValueError):
            PCAProjector(dim=8, basis="joint").fit(c)
        assert PCAProjector(dim=8, basis="joint").fit(c, q).fitted

    def test_unfitted_use_is_an_error(self, rng):
        with pytest.raises(RuntimeError):
            PCAProjector(dim=4).project(np.ones((2, 8), np.float32))


def test_random_projection_preserves_norms_on_average(rng):
    c = _corpus(rng, anisotropic=False)
    proj = RandomProjector(dim=16, seed=1).fit(c)
    x = _unit(rng.standard_normal((2000, 32)))
    ratio = np.linalg.norm(proj.project(x), axis=1) ** 2
    assert abs(ratio.mean() - 1.0) < 0.05


def test_truncation_keeps_leading_coordinates(rng):
    v = rng.standard_normal((3, 10)).astype(np.float32)
    np.testing.assert_array_equal(TruncateProjector(4).project(v), v[:, :4])


class TestInt8Scales:
    def test_per_vector_beats_fixed_on_wide_unit_vectors(self, rng):
        """Synthetic evidence for the audit's int8 finding (not a model measurement)."""
        v = _unit(rng.standard_normal((200, 3072)))
        err = {}
        for scale in ("fixed", "per_vector"):
            q = Int8Quantizer(scale)
            err[scale] = float(np.abs(q.decode(q.encode(v), 3072) - v).mean())
        assert err["per_vector"] < err["fixed"] / 2

    def test_per_vector_codes_carry_their_scale(self, rng):
        v = rng.standard_normal((5, 16)).astype(np.float32)
        q = Int8Quantizer("per_vector")
        codes = q.encode(v)
        assert codes.shape == (5, 18) == (5, q.code_bytes(16))
        np.testing.assert_allclose(q.decode(codes, 16), v, atol=np.abs(v).max() / 127 + 1e-3)

    def test_per_dimension_requires_fit_and_roundtrips(self, rng):
        c = _corpus(rng)
        q = Int8Quantizer("per_dimension")
        with pytest.raises(RuntimeError):
            q.encode(c.vectors)
        q.fit(c)
        back = q.decode(q.encode(c.vectors), c.dimension)
        assert np.abs(back - c.vectors).max() < 0.05
        assert q.overhead_bytes() == 4 * c.dimension


def test_int4_roundtrip_error_is_bounded(rng):
    v = rng.standard_normal((20, 15)).astype(np.float32)  # odd dimension on purpose
    q = Int4Quantizer()
    codes = q.encode(v)
    assert codes.shape[1] == q.code_bytes(15) == 10
    back = q.decode(codes, 15)
    step = np.abs(v).max(axis=1, keepdims=True) / 7.0
    assert np.all(np.abs(back - v) <= step / 2 + 1e-2)


def test_centred_binary_ranks_like_decoding_with_the_mean(rng):
    c, q = _corpus(rng), _queries(rng)
    bq = BinaryQuantizer(center="mean").fit(c)
    comp = Pipeline([bq]).compress(c, fit=False)
    plus_mean = MultiVectorCorpus(comp.decode_rows(0, comp.num_vectors) + bq.mean, comp.offsets)
    np.testing.assert_array_equal(rank(maxsim_matrix(q, comp)), rank(maxsim_matrix(q, plus_mean)))
    assert bq.overhead_bytes() == 4 * c.dimension


@pytest.mark.parametrize("stage", [PCAProjector(24, center=True, basis="joint"), RandomProjector(8, seed=3),
                                   TruncateProjector(12), Int8Quantizer("per_vector"), Int4Quantizer(),
                                   BinaryQuantizer(center="mean")])
def test_dict_roundtrip(stage):
    assert stage_from_dict(stage.to_dict()).params() == stage.params()


def test_dimension_projector_delegates(rng):
    from optivision import DimensionProjector

    c, q = _corpus(rng), _queries(rng)
    p = Pipeline([DimensionProjector(8)]).fit(c)
    direct = Pipeline([PCAProjector(8)]).fit(c)
    np.testing.assert_allclose(maxsim_matrix(p.transform_queries(q), p.compress(c)),
                               maxsim_matrix(direct.transform_queries(q), direct.compress(c)), rtol=1e-5)
    assert DimensionProjector(8, method="truncate").project(c.vectors).shape[1] == 8
    assert stage_from_dict(DimensionProjector(6, method="random", seed=2).to_dict()).params() == {
        "dim": 6, "method": "random", "seed": 2}
    with pytest.raises(ValueError):
        DimensionProjector(8, method="magic")



@pytest.mark.parametrize("centred", [True, False])
def test_centred_scalar_codecs_resolve_a_narrow_cone(rng, centred):
    common = _unit(rng.standard_normal((1, 64)))
    v = _unit(common + 0.05 * rng.standard_normal((400, 64)))  # every vector close to one direction
    c = MultiVectorCorpus.from_arrays([v])
    for make in (Int4Quantizer, lambda **kw: Int8Quantizer("per_vector", **kw)):
        plain = make()
        cen = make(center="mean").fit(c)
        err_plain = np.abs(plain.decode(plain.encode(v), 64) - v).mean()
        err_cen = np.abs(cen.decode(cen.encode(v), 64) - v).mean()
        assert err_cen < err_plain / 2
        assert stage_from_dict(cen.to_dict()).params() == cen.params()
        with pytest.raises(RuntimeError):
            make(center="mean").encode(v)
        if not centred:
            break
