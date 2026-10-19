"""The int4 norm study's two helpers (``scripts/universal_study``), which stay out of the package.

``int4norm_codec.py``: the same levels and bytes as int4, decoded norms restored.
``device_scoring.py``: exactly the scores of the pinned ``scoring.maxsim_matrix`` on the same device.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from optivision import scoring
from optivision.compose import Pipeline
from optivision.representation import MultiVectorCorpus
from optivision.stages import BinaryQuantizer, Int4Quantizer, Int8Quantizer
from optivision.stages.base import STAGES

STUDY = Path(__file__).resolve().parents[1] / "scripts" / "universal_study"

pytestmark = pytest.mark.skipif(not (STUDY / "int4norm_codec.py").exists(), reason="needs the repository's scripts")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ------------------------------------------------------------ int4norm_codec


@pytest.mark.parametrize("center", [None, "mean"])
def test_int4norm_keeps_the_levels_and_restores_the_norm(center):
    codec = _load("int4norm_codec")
    rng = np.random.default_rng(0)
    v = _unit(rng, 200, 15) + 0.5 / np.sqrt(15)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    c = MultiVectorCorpus.from_arrays([v])
    plain = Int4Quantizer(center=center).fit(c)
    kept = codec.NormPreservingInt4Quantizer(center=center).fit(c)
    a, b = plain.encode(v), kept.encode(v)
    assert b.shape == a.shape and kept.code_bytes(15) == plain.code_bytes(15)
    np.testing.assert_array_equal(b[:, :8], a[:, :8])  # the levels; only the float16 scale differs
    back = kept.decode(b, 15)  # the mean is added back: the whole vector keeps its norm
    np.testing.assert_allclose(np.linalg.norm(back, axis=1), 1.0, rtol=2e-3)
    old = np.abs(np.linalg.norm(plain.decode(a, 15), axis=1) - 1).std()
    assert np.abs(np.linalg.norm(back, axis=1) - 1).std() < old / 4


def test_int4norm_edge_cases_labels_and_the_registry():
    codec = _load("int4norm_codec")
    q = codec.NormPreservingInt4Quantizer()
    zero = np.zeros((1, 15), np.float32)
    assert np.all(q.decode(q.encode(zero), 15) == 0)
    with pytest.raises(ValueError):
        codec.NormPreservingInt4Quantizer(center="median")
    with pytest.raises(RuntimeError, match="must be fit"):
        codec.NormPreservingInt4Quantizer(center="mean").encode(zero)
    assert Pipeline([codec.NormPreservingInt4Quantizer()]).label() == "int4norm"
    assert Pipeline([codec.NormPreservingInt4Quantizer(center="mean")]).label() == "int4norm(mean)"
    assert "int4norm" not in STAGES and STAGES["int4"] is Int4Quantizer  # not a package stage


# ------------------------------------------------------------ device_scoring


@pytest.mark.parametrize("codec", [None, "int8", "binary", "int4norm(mean)"])
def test_device_scores_are_bit_identical_to_the_pinned_device_path(monkeypatch, codec):
    pytest.importorskip("torch")
    device = _load("device_scoring")
    stages = {"int8": lambda: Int8Quantizer("per_vector"), "binary": BinaryQuantizer,
              "int4norm(mean)": lambda: _load("int4norm_codec").NormPreservingInt4Quantizer(center="mean")}
    rng = np.random.default_rng(3)
    docs = MultiVectorCorpus.from_arrays([_unit(rng, n, 96) for n in (0, 7, 1, 0, 30, 2, 0, 11)])
    queries = MultiVectorCorpus.from_arrays([_unit(rng, n, 96) for n in (0, 4, 9, 0, 1)])
    store = docs if codec is None else Pipeline([stages[codec]()]).compress(docs, fit=True)
    monkeypatch.setenv("OPTIVISION_SCORE_DEVICE", "cpu")
    for budget in (1 << 10, 1 << 13, 1 << 20):  # one page per block up to all pages in one
        want = scoring.maxsim_matrix(queries, store, max_block_bytes=budget)
        got = device.maxsim_matrix(queries, store, max_block_bytes=budget)
        assert np.array_equal(got, want)
    assert np.all(np.isneginf(got[:, [0, 3, 6]])) and np.all(got[[0, 3], :][:, [1, 2, 4, 5, 7]] == 0)


def test_without_a_device_it_is_the_numpy_path(monkeypatch):
    device = _load("device_scoring")
    monkeypatch.delenv("OPTIVISION_SCORE_DEVICE", raising=False)
    rng = np.random.default_rng(4)
    docs = MultiVectorCorpus.from_arrays([_unit(rng, n, 16) for n in (3, 0, 5)])
    queries = MultiVectorCorpus.from_arrays([_unit(rng, n, 16) for n in (2, 1)])
    assert np.array_equal(device.maxsim_matrix(queries, docs), scoring.maxsim_matrix(queries, docs))


def test_calibration_scores_with_it_only_when_asked(monkeypatch):
    from optivision import calibration

    device = _load("device_scoring")
    monkeypatch.setattr(calibration, "maxsim_matrix", scoring.maxsim_matrix)
    device.use_for_calibration()
    assert calibration.maxsim_matrix is device.maxsim_matrix
