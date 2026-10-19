"""``NormPreservingInt4Quantizer``: int4's levels and bytes, with a scale that keeps the norm."""

from __future__ import annotations

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.core.stages import fitted_state, traits_of
from optivision.representation import MultiVectorCorpus
from optivision.stages import STAGES, Int4Quantizer, NormPreservingInt4Quantizer, stage_from_dict


def _corpus(seed: int = 0, dim: int = 33, offset: float = 0.0) -> MultiVectorCorpus:
    rng = np.random.default_rng(seed)
    return MultiVectorCorpus.from_arrays(
        [(rng.normal(size=(30, dim)) + offset).astype(np.float32) for _ in range(5)])


def _levels(codes: np.ndarray, dim: int) -> np.ndarray:
    nb = (dim + 1) // 2
    packed = codes[:, :nb]
    out = np.empty((codes.shape[0], nb * 2), np.int16)
    out[:, 0::2], out[:, 1::2] = packed & 0x0F, packed >> 4
    return out[:, :dim]


@pytest.mark.parametrize("center", [None, "mean"])
def test_levels_and_code_size_are_int4s_and_only_the_scale_differs(center):
    docs = _corpus(offset=0.4)
    plain = Int4Quantizer(center=center).fit(docs)
    norm = NormPreservingInt4Quantizer(center=center).fit(docs)
    a, b = plain.encode(docs.vectors), norm.encode(docs.vectors)
    assert a.shape == b.shape and norm.code_bytes(33) == plain.code_bytes(33)
    assert np.array_equal(_levels(a, 33), _levels(b, 33))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("center", [None, "mean"])
def test_each_decoded_vector_keeps_the_norm_of_the_vector_it_codes(center):
    docs = _corpus(seed=1, offset=0.6)
    v = docs.vectors
    norm = NormPreservingInt4Quantizer(center=center).fit(docs)
    plain = Int4Quantizer(center=center).fit(docs)
    ratio = np.linalg.norm(norm.decode(norm.encode(v), 33), axis=1) / np.linalg.norm(v, axis=1)
    plain_ratio = np.linalg.norm(plain.decode(plain.encode(v), 33), axis=1) / np.linalg.norm(v, axis=1)
    assert np.abs(ratio - 1).max() < 2e-3  # the float16 scale is the only error left
    assert np.abs(ratio - 1).std() < np.abs(plain_ratio - 1).std() / 10


def test_zero_vectors_unfitted_centring_and_bad_centres():
    q = NormPreservingInt4Quantizer()
    zero = np.zeros((2, 15), np.float32)
    assert np.all(q.decode(q.encode(zero), 15) == 0)
    with pytest.raises(ValueError):
        NormPreservingInt4Quantizer(center="median")
    with pytest.raises(RuntimeError, match="must be fit"):
        NormPreservingInt4Quantizer(center="mean").encode(zero)


def test_it_is_a_registered_stage_with_labels_specs_and_traits():
    assert STAGES["int4norm"] is NormPreservingInt4Quantizer and STAGES["int4"] is Int4Quantizer
    assert Pipeline([NormPreservingInt4Quantizer()]).label() == "int4norm"
    assert Pipeline([NormPreservingInt4Quantizer(center="mean")]).label() == "int4norm(mean)"
    spec = {"stage": "int4norm", "center": "mean"}
    assert stage_from_dict(dict(spec)).to_dict() == spec
    assert traits_of(NormPreservingInt4Quantizer()).has_fitted_state is False
    assert traits_of(NormPreservingInt4Quantizer(center="mean")).has_fitted_state is True


def test_the_fitted_mean_is_captured_as_state():
    docs = _corpus(seed=2, offset=0.3)
    q = NormPreservingInt4Quantizer(center="mean").fit(docs)
    state = fitted_state(q)
    assert np.array_equal(state["mean"], q.mean)
    assert fitted_state(NormPreservingInt4Quantizer()) == {}
