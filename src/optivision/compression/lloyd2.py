"""Rotated 2-bit Lloyd-Max quantization of late-interaction vectors.

Sits between binary (32x, ``sign(d)``) and int8 (4x): 2 bits per dimension, so a
128-d vector costs 32 bytes instead of 16 (binary) or 128 (int8) -- a flat 16x on
everything the pruner let through.

Why rotate first
-----------------
A per-dimension sign split is the coarsest possible 1-bit quantizer; the natural
extension to 2 bits per dimension is a 4-level scalar quantizer, and the optimal
(mean-squared-error minimising) one for a Gaussian source is the Lloyd-Max
quantizer -- non-uniform levels/thresholds concentrated where the density is
highest. ColPali-family embeddings are not isotropic (``docs/REVIEW-2026-08-21.md``
measures 13-14 of 128 sign bits carrying under half a bit of entropy on ColSmol),
so quantizing raw dimensions spends levels on directions that barely vary. A
random orthonormal rotation spreads variance evenly across dimensions first,
which is what lets one fixed Lloyd-Max table serve every dimension: the rotation
needs no data to fit (unlike ITQ), only a seed and the dimension, so it costs
nothing to reproduce and nothing to store -- ``rotation_matrix`` regenerates it
from ``(dim, seed)`` on every load.

Levels and thresholds
----------------------
``_LEVELS_UNIT`` / ``_THRESHOLDS_UNIT`` are the standard 2-bit (4-level) Lloyd-Max
quantizer for a zero-mean, unit-variance Gaussian source, scaled per corpus by one
fitted scalar ``sigma`` -- the std of the corpus's centred embeddings. Centring
uses the corpus mean ``mu`` (a per-dimension vector), which must be fit once, the
same way :func:`optivision.pruning.fit_codebook` fits probe directions once and
every page reuses them (see ``TokenPruner(codebook=...)``). :func:`fit_lloyd2`
plays that role here; the caller fits it once over a sample of the corpus and
passes it to every :class:`~optivision.compression.Compressor` that needs it.

Byte accounting
----------------
32 bytes/vector for the packed 2-bit codes, plus ``mu`` (``dim`` float32s) and
``sigma`` (one float32) shared by the whole index rather than paid per page --
:meth:`Lloyd2Codec.overhead_bytes` reports that shared cost so callers can
amortize it the way ``NumpyIndex.stats`` does. The rotation itself is never
stored (regenerated from the seed), so it adds zero bytes.

Asymmetric scoring
-------------------
As with the binary codec, the query stays float32; only the document side is
quantized. :func:`decode_lloyd2` reconstructs an approximate *original-space*
float32 vector (``levels[index] @ R.T + mu``), so a raw float query scores
against it exactly the way it scores against a decoded binary or int8 code --
no query-side rotation needed, because reconstructing back through ``R.T``
before scoring keeps the interface identical to every other codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# Optimal 2-bit (4-level) Lloyd-Max quantizer for a zero-mean, unit-variance
# Gaussian source: the non-uniform levels/thresholds that minimise mean-squared
# reconstruction error at 4 levels. Textbook constants (e.g. Max, 1960), reused
# here scaled by one fitted sigma per corpus.
_LEVELS_UNIT = np.array([-1.510, -0.4528, 0.4528, 1.510], dtype=np.float32)
_THRESHOLDS_UNIT = np.array([-0.9816, 0.0, 0.9816], dtype=np.float32)

# How many vectors to draw when fitting mu/sigma. Both are low-variance
# statistics (a mean and a global std), so a fixed-size sample is enough
# regardless of corpus size -- matches the sample codec_ladder.py uses to fit
# the same quantities.
LLOYD2_FIT_SAMPLE = 50_000


@dataclass(frozen=True)
class Lloyd2Codec:
    """Corpus-fitted state the rotated 2-bit codec needs at encode and decode time.

    ``mu`` and ``sigma`` are fit once from data (:func:`fit_lloyd2`); the
    rotation is not stored here because it is a pure function of ``(dim, seed)``
    -- see :func:`rotation_matrix`.
    """

    mu: np.ndarray  # float32 [dim], corpus mean
    sigma: float  # std of the centred corpus, before rotation
    seed: int = 7
    dim: int = 0

    def __post_init__(self) -> None:
        if self.dim == 0:
            object.__setattr__(self, "dim", int(self.mu.shape[0]))

    @property
    def rotation(self) -> np.ndarray:
        return rotation_matrix(self.dim, self.seed)

    @property
    def levels(self) -> np.ndarray:
        return _LEVELS_UNIT * self.sigma

    @property
    def thresholds(self) -> np.ndarray:
        return _THRESHOLDS_UNIT * self.sigma

    @property
    def overhead_bytes(self) -> int:
        """Shared per-index cost: ``mu`` plus one scalar, paid once for the whole
        corpus rather than once per page. The rotation matrix is regenerated
        from ``seed`` and never stored, so it contributes nothing here."""
        return int(self.mu.nbytes) + 4


@lru_cache(maxsize=8)
def _rotation_matrix_cached(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return np.ascontiguousarray(r, dtype=np.float32)


def rotation_matrix(dim: int, seed: int = 7) -> np.ndarray:
    """Deterministic random orthonormal rotation for ``dim`` -- no data needed.

    Cached so repeated encode/decode calls (one per page) do not redo a QR
    decomposition every time; the array returned is shared, so callers must not
    mutate it in place.
    """
    return _rotation_matrix_cached(int(dim), int(seed))


def fit_lloyd2(
    vectors: np.ndarray,
    seed: int = 7,
    sample_size: int = LLOYD2_FIT_SAMPLE,
    rng: np.random.Generator | None = None,
) -> Lloyd2Codec:
    """Fit ``mu`` and ``sigma`` from a corpus sample. Fit once, reuse for every page.

    Mirrors :func:`optivision.pruning.fit_codebook`: this returns fitted state,
    it does not encode anything itself.
    """
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"expected 2-D vectors, got shape {v.shape}")
    mu = v.mean(axis=0).astype(np.float32)
    rng = rng or np.random.default_rng(seed)
    if v.shape[0] > sample_size:
        sample = v[rng.choice(v.shape[0], sample_size, replace=False)] - mu
    else:
        sample = v - mu
    sigma = float(sample.std()) if sample.size else 1.0
    return Lloyd2Codec(mu=mu, sigma=sigma, seed=seed, dim=v.shape[1])


def code_nbytes(dim: int) -> int:
    """Bytes needed to pack ``dim`` 2-bit codes, 4 per byte."""
    return (dim + 3) // 4


def pack2(indices: np.ndarray) -> np.ndarray:
    """uint8 [n, dim] of values in ``{0, 1, 2, 3}`` -> uint8 [n, ceil(dim/4)]."""
    idx = np.asarray(indices, dtype=np.uint8)
    if idx.ndim != 2:
        raise ValueError(f"expected 2-D indices, got shape {idx.shape}")
    n, dim = idx.shape
    nbytes = code_nbytes(dim)
    pad = nbytes * 4 - dim
    if pad:
        idx = np.pad(idx, ((0, 0), (0, pad)))
    idx = idx.reshape(n, nbytes, 4)
    return (idx[..., 0] | (idx[..., 1] << 2) | (idx[..., 2] << 4) | (idx[..., 3] << 6)).astype(
        np.uint8
    )


def unpack2(codes: np.ndarray, dim: int) -> np.ndarray:
    """Inverse of :func:`pack2`: uint8 [n, nbytes] -> uint8 [n, dim] in ``{0..3}``."""
    c = np.asarray(codes, dtype=np.uint8)
    n = c.shape[0]
    out = np.empty((n, c.shape[1] * 4), dtype=np.uint8)
    out[:, 0::4] = c & 0b11
    out[:, 1::4] = (c >> 2) & 0b11
    out[:, 2::4] = (c >> 4) & 0b11
    out[:, 3::4] = (c >> 6) & 0b11
    return out[:, :dim]


def encode_lloyd2(vectors: np.ndarray, codec: Lloyd2Codec) -> np.ndarray:
    """float32 [n, dim] -> packed 2-bit codes, uint8 [n, ceil(dim/4)]."""
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"expected 2-D vectors, got shape {v.shape}")
    rotated = (v - codec.mu) @ codec.rotation
    indices = np.digitize(rotated, codec.thresholds).astype(np.uint8)
    return pack2(indices)


def decode_lloyd2(codes: np.ndarray, dim: int, codec: Lloyd2Codec) -> np.ndarray:
    """Inverse of :func:`encode_lloyd2`: reconstructs an approximate original-space
    float32 [n, dim] array, so it scores against a raw float query exactly like
    every other codec's decode does."""
    indices = unpack2(codes, dim)
    rotated = codec.levels[indices].astype(np.float32)
    return rotated @ codec.rotation.T + codec.mu


def maxsim_asymmetric_lloyd2(
    query: np.ndarray, codes: np.ndarray, dim: int, codec: Lloyd2Codec
) -> float:
    """MaxSim between a float query and one page of 2-bit document codes."""
    doc = decode_lloyd2(codes, dim, codec)
    if doc.shape[0] == 0:
        return 0.0
    sims = np.asarray(query, dtype=np.float32) @ doc.T
    return float(sims.max(axis=1).sum())


def maxsim_asymmetric_lloyd2_batch(
    query: np.ndarray, codes: np.ndarray, dim: int, offsets: np.ndarray, codec: Lloyd2Codec
) -> np.ndarray:
    """Score many pages that share one concatenated 2-bit code matrix.

    Mirrors :func:`optivision.compression.binary.maxsim_asymmetric_batch`.
    """
    doc = decode_lloyd2(codes, dim, codec)
    sims = np.asarray(query, dtype=np.float32) @ doc.T
    n_pages = int(offsets.size - 1)
    scores = np.zeros(n_pages, dtype=np.float32)
    for i in range(n_pages):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if hi > lo:
            scores[i] = sims[:, lo:hi].max(axis=1).sum()
    return scores
