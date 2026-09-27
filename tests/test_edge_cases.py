"""Every stage against degenerate inputs (release audit, 2026-09-27).

Empty, one-vector and duplicate documents, zero vectors, odd and wide
dimensions: shapes, dtypes, finite scores and no document emptied or grown.
Non-finite input is refused rather than silently leaking into fitted state.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RandomProjector,
    RandomPruner,
    RedundancyPruner,
    TruncateProjector,
)

pytest.importorskip("scipy")


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _stages(d):
    out = {
        "float32": [],
        "float16": [Float16Quantizer()],
        "int8 fixed": [Int8Quantizer()],
        "int8 per_vector": [Int8Quantizer("per_vector")],
        "int8 per_dimension": [Int8Quantizer("per_dimension")],
        "int8 centred": [Int8Quantizer("per_vector", center="mean")],
        "int4": [Int4Quantizer()],
        "int4 centred": [Int4Quantizer(center="mean")],
        "lloyd2": [Lloyd2Quantizer()],
        "binary": [BinaryQuantizer()],
        "binary centred": [BinaryQuantizer(center="mean")],
        "adaptive radius": [AdaptiveMerge(radius=0.6)],
        "adaptive ratio": [AdaptiveMerge(ratio=0.3)],
        "adaptive centred": [AdaptiveMerge(radius=0.6, center="mean")],
        "ward ratio": [HierarchicalMerge(ratio=0.3)],
        "ward distance": [HierarchicalMerge(max_distance=0.8)],
        "random": [RandomPruner(ratio=0.3)],
        "redundancy": [RedundancyPruner(0.8)],
        "ward > binary": [HierarchicalMerge(ratio=0.3), BinaryQuantizer()],
    }
    if d >= 4:
        k = d // 2
        out.update({"pca": [PCAProjector(k)], "random projection": [RandomProjector(k)],
                    "truncate": [TruncateProjector(k)], "pca > int8": [PCAProjector(k), Int8Quantizer("per_vector")]})
    return out


def _documents(rng, d):
    dup = np.repeat(_unit(rng, 1, d), 6, axis=0)
    return [
        np.zeros((0, d), np.float32),  # empty
        _unit(rng, 1, d),  # one vector
        _unit(rng, 2, d),
        dup,  # all duplicates
        np.vstack([dup, _unit(rng, 2, d)]),
        np.zeros((4, d), np.float32),  # all-zero vectors
        np.vstack([np.zeros((2, d), np.float32), _unit(rng, 3, d)]),
        _unit(rng, 30, d),
    ]


CASES = [(d, name) for d in (1, 3, 7, 96, 128) for name in _stages(d)] + [
    (4096, name) for name in ("int8 per_vector", "int4 centred", "binary", "ward > binary", "pca > int8")
]


@pytest.mark.filterwarnings(r"ignore:Int8Quantizer\(scale=.fixed.\) clips")
@pytest.mark.parametrize(("dim", "stage"), CASES, ids=[f"d{d}-{n}" for d, n in CASES])
def test_degenerate_documents(dim, stage):
    rng = np.random.default_rng(dim)
    arrays = _documents(rng, dim)
    docs = MultiVectorCorpus.from_arrays(arrays)
    queries = MultiVectorCorpus.from_arrays([_unit(rng, 4, dim), _unit(rng, 1, dim)])
    pipe = Pipeline(_stages(dim)[stage])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no invalid casts / divisions by zero
        compressed = pipe.fit(docs).compress(docs)
        rows = compressed.decode_rows(0, compressed.num_vectors)
        scores = maxsim_matrix(pipe.transform_queries(queries), compressed)
    assert len(compressed) == len(arrays)
    counts = np.diff(compressed.offsets)
    for i, a in enumerate(arrays):
        assert (counts[i] == 0) == (a.shape[0] == 0), f"document {i}: {a.shape[0]} -> {counts[i]} vectors"
        assert counts[i] <= a.shape[0]
    assert rows.dtype == np.float32 and np.isfinite(rows).all()
    assert scores.shape == (2, len(arrays))
    nonempty = counts > 0
    assert np.isfinite(scores[:, nonempty]).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("stage", ["int8 centred", "int8 per_dimension", "pca", "lloyd2", "float32"])
def test_non_finite_documents_are_refused(bad, stage):
    rng = np.random.default_rng(0)
    poisoned = _unit(rng, 4, 16)
    poisoned[2, 5] = bad
    docs = MultiVectorCorpus.from_arrays([_unit(rng, 5, 16), poisoned, _unit(rng, 6, 16)])
    with pytest.raises(ValueError, match=r"non-finite.*document 1"):
        Pipeline(_stages(16)[stage]).compress(docs)


def test_non_finite_queries_are_refused():
    rng = np.random.default_rng(0)
    q = _unit(rng, 3, 16)
    q[0, 0] = np.nan
    with pytest.raises(ValueError, match="queries contain non-finite"):
        Pipeline().transform_queries(MultiVectorCorpus.from_arrays([q]))


def test_fitting_on_a_corpus_without_vectors_is_refused():
    empty = MultiVectorCorpus.from_arrays([np.zeros((0, 8), np.float32)] * 3)
    with pytest.raises(ValueError, match="no vectors"):
        Pipeline([Int8Quantizer("per_dimension")]).compress(empty)
    # stateless stages still pass an all-empty corpus through
    assert Pipeline([BinaryQuantizer()]).compress(empty).num_vectors == 0


def test_zero_vector_codes_are_exact_zero():
    zeros = MultiVectorCorpus.from_arrays([np.zeros((3, 10), np.float32)])
    for q in (Int8Quantizer("per_vector"), Int4Quantizer()):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = Pipeline([q]).compress(zeros).decode_rows(0, 3)
        assert np.array_equal(out, np.zeros((3, 10), np.float32))


def test_fixed_int8_warns_when_it_clips():
    big = MultiVectorCorpus.from_arrays([np.full((2, 4), 0.9, np.float32)])
    with pytest.warns(UserWarning, match="clips components"):
        Pipeline([Int8Quantizer()]).compress(big)
    unit = MultiVectorCorpus.from_arrays([np.full((2, 4), 0.5, np.float32)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Pipeline([Int8Quantizer()]).compress(unit)
