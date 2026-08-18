"""Binary quantization of late-interaction vectors.

A ColPali-family vector is 128 float32 dimensions = 512 bytes. Keeping only the
sign of each dimension gives 128 bits = 16 bytes: a flat 32x on everything the
pruner let through.

Why signs survive MaxSim
------------------------
These vectors are L2-normalised and their dimensions are close to
zero-centred, so ``sign(d)`` keeps the orthant of ``d`` and discards only its
within-orthant position. For random unit vectors the expected cosine between
``d`` and ``sign(d)/sqrt(dim)`` is sqrt(2/pi) ~= 0.798, and — crucially for
ranking — the distortion is almost the same for every document vector, so it
shifts scores far more than it reorders them.

Symmetric vs asymmetric
-----------------------
Symmetric scoring binarises the query too and reduces to popcount Hamming
distance: fastest, but it throws away the query magnitudes, which are the part
we can afford to keep (a query has ~20 vectors; a corpus has millions).
Asymmetric scoring keeps the query in float32 and scores it against +/-1
document codes. It costs one unpack and a float matmul, and recovers most of
the ranking quality. Asymmetric is the default here.
"""

from __future__ import annotations

import numpy as np


def pack_bits(vectors: np.ndarray) -> np.ndarray:
    """float32 [n, dim] -> uint8 [n, ceil(dim/8)] of sign bits (1 = positive)."""
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"expected 2-D vectors, got shape {v.shape}")
    return np.packbits(v > 0, axis=1)


# Two-entry lookup: bit 0 -> -1.0, bit 1 -> +1.0.
_SIGN_LUT = np.array([-1.0, 1.0], dtype=np.float32)


def unpack_signs(codes: np.ndarray, dim: int) -> np.ndarray:
    """uint8 [n, nbytes] -> float32 [n, dim] with entries in {-1, +1}.

    Written as a lookup rather than the obvious ``bits * 2 - 1`` because this is
    the one place the index is at its largest. That expression allocates the
    full float32 result three times over — once for the cast and once for each
    arithmetic step — so on a block sized to a memory budget the true peak came
    out at roughly three times the budget. Indexing a 2-element table produces
    the result in a single allocation.
    """
    bits = np.unpackbits(np.asarray(codes, dtype=np.uint8), axis=1)[:, :dim]
    return _SIGN_LUT[bits]


def code_nbytes(dim: int) -> int:
    return (dim + 7) // 8


def maxsim_asymmetric(query: np.ndarray, codes: np.ndarray, dim: int) -> float:
    """MaxSim between a float query and one page of binary document codes.

    score = sum over query vectors of max over document vectors of q . sign(d)
    """
    doc = unpack_signs(codes, dim)
    if doc.shape[0] == 0:
        return 0.0
    sims = np.asarray(query, dtype=np.float32) @ doc.T  # [n_query, n_doc]
    return float(sims.max(axis=1).sum())


def maxsim_asymmetric_batch(
    query: np.ndarray, codes: np.ndarray, dim: int, offsets: np.ndarray
) -> np.ndarray:
    """Score many pages that share one concatenated code matrix.

    Args:
        query: float32 [n_query, dim]
        codes: uint8 [total_vectors, nbytes] — all pages stacked
        offsets: int64 [n_pages + 1] — page i owns rows offsets[i]:offsets[i+1]

    One big matmul beats a Python loop of small ones by an order of magnitude,
    which is what makes the exhaustive numpy index usable as a reference.
    """
    doc = unpack_signs(codes, dim)  # [total, dim]
    sims = np.asarray(query, dtype=np.float32) @ doc.T  # [n_query, total]
    n_pages = int(offsets.size - 1)
    scores = np.zeros(n_pages, dtype=np.float32)
    for i in range(n_pages):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if hi > lo:
            scores[i] = sims[:, lo:hi].max(axis=1).sum()
    return scores


def maxsim_float(query: np.ndarray, doc: np.ndarray) -> float:
    """Full-precision MaxSim — the baseline every compressed score is judged against."""
    if doc.shape[0] == 0:
        return 0.0
    sims = np.asarray(query, dtype=np.float32) @ np.asarray(doc, dtype=np.float32).T
    return float(sims.max(axis=1).sum())


def maxsim_hamming(query_codes: np.ndarray, doc_codes: np.ndarray, dim: int) -> float:
    """Fully symmetric variant: both sides binary, scored by popcount.

    Kept for the ablation table — it is the cheapest option and shows what the
    asymmetric query buys.
    """
    q = unpack_signs(query_codes, dim)
    d = unpack_signs(doc_codes, dim)
    if d.shape[0] == 0:
        return 0.0
    sims = (q @ d.T) / dim  # agreements minus disagreements, normalised
    return float(sims.max(axis=1).sum())
