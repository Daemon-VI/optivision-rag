"""Wide (2,560-4,096-d) vectors: memory caps, the fast Ward path, GPU scoring, projection space."""

from __future__ import annotations

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix
from optivision.stages import BinaryQuantizer, Int8Quantizer
from optivision.stages.base import FIT_SAMPLE_ROWS, fit_sample_rows


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_fit_samples_are_unchanged_for_measured_widths_and_capped_for_wide_ones():
    for d in (96, 128):
        assert fit_sample_rows(d, 8) == FIT_SAMPLE_ROWS and fit_sample_rows(d, 4) == FIT_SAMPLE_ROWS
    assert fit_sample_rows(2560, 8) * 2560 * 8 <= 256 * 1024**2
    assert fit_sample_rows(4096, 4) < FIT_SAMPLE_ROWS


def test_transform_cache_respects_its_byte_budget(monkeypatch):
    from optivision import calibration

    rng = np.random.default_rng(0)
    corpus = MultiVectorCorpus.from_arrays([_unit(rng, 20, 64) for _ in range(5)])
    monkeypatch.setattr(calibration, "TRANSFORM_CACHE_BYTES", int(corpus.vectors.nbytes * 1.5))
    cache: dict = {}
    for k in ("a", "b", "c"):
        calibration._cache_put(cache, k, ([], corpus))
    assert list(cache) == ["c"]  # oldest entries evicted, newest kept


def test_fast_ward_path_matches_scipy_on_wide_unit_vectors():
    scipy = pytest.importorskip("scipy.cluster.hierarchy")
    from optivision.stages.merge import WIDE_DIM, _ward_input

    rng = np.random.default_rng(1)
    x = _unit(rng, 300, WIDE_DIM + 100).astype(np.float64)
    a = scipy.fcluster(scipy.linkage(x, method="ward"), t=80, criterion="maxclust")
    b = scipy.fcluster(scipy.linkage(_ward_input(x), method="ward"), t=80, criterion="maxclust")
    assert np.array_equal(a, b)
    narrow = _unit(rng, 10, 128).astype(np.float64)
    assert _ward_input(narrow) is narrow  # measured widths keep the original call


@pytest.mark.parametrize("codec", [None, Int8Quantizer("per_vector"), BinaryQuantizer()])
def test_torch_scoring_matches_numpy(monkeypatch, codec):
    pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    docs = MultiVectorCorpus.from_arrays([_unit(rng, int(rng.integers(0, 40)), 600) for _ in range(25)])
    queries = MultiVectorCorpus.from_arrays([_unit(rng, int(rng.integers(0, 9)), 600) for _ in range(9)])
    store = docs if codec is None else Pipeline([codec]).compress(docs)
    expected = maxsim_matrix(queries, store, max_block_bytes=1 << 15)
    monkeypatch.setenv("OPTIVISION_SCORE_DEVICE", "cpu")
    got = maxsim_matrix(queries, store, max_block_bytes=1 << 15)
    assert np.array_equal(np.isfinite(expected), np.isfinite(got))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(got[finite], expected[finite], atol=1e-4)


def test_projection_space_is_opt_in_and_ordered():
    from optivision.calibration import default_search_space, projection_targets, wide_search_space

    assert projection_targets(4096) == [2048, 1024, 512, 256, 128]
    assert projection_targets(2560) == [1280, 640, 320, 160, 128]
    default, wide = default_search_space(2560), wide_search_space(2560)
    assert not any("pca" in f for f in default)  # the measured default is untouched
    for family in (f for f in wide if "pca" in f):
        widths = [next((s.dim for s in p.stages if s.name == "project"), 2560) for p in wide[family]]
        assert widths == sorted(widths, reverse=True)  # least aggressive first
