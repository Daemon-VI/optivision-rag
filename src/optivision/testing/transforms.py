"""Transformations that leave every MaxSim score unchanged, for checking basis dependence.

A late-interaction score is a sum of maxima of inner products. These changes of the stored
vectors keep every inner product between a query vector and a document vector (up to
rounding) and so keep the float scores:

* an orthogonal matrix applied to every document and query vector (``random_orthogonal``,
  ``signed_permutation`` with ``apply_basis``);
* reordering the vectors inside a document, together with their row annotations
  (``reorder_rows``).

What they change is what a coordinate codec sees: sign bits and fixed-step integers act on
the stored coordinates, so a pipeline can behave differently on a rotated copy of the same
index. Nothing here says whether it does; ``basis_probe`` measures that for one pipeline on
one data set.

The orthogonal matrices are built the way the rotation of the registered symmetry experiment
is (``reports/research/NEXT_PHASE_PREREGISTRATION.md`` section 12.3: the orthogonal factor of
a QR decomposition of a standard normal matrix from ``numpy.random.default_rng(seed)``, with
each column multiplied by the sign of the corresponding diagonal entry of R), and applied in
float64 and cast back to the stored dtype. That is a statement about the construction, not
about any result.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.vectors import MultiVectorCorpus

__all__ = ["apply_basis", "random_orthogonal", "reorder_rows", "signed_permutation"]

#: bytes of float64 temporaries ``apply_basis`` may hold at once
_BLOCK_BYTES = 64 * 1024 * 1024


def _dimension(dim: Any) -> int:
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim < 1:
        raise ValueError(f"dim must be a positive integer, got {dim!r}")
    return int(dim)


def random_orthogonal(dim: int, seed: int) -> np.ndarray:
    """A random orthogonal ``[dim, dim]`` float64 matrix, the same for the same ``seed``.

    The orthogonal factor of the QR decomposition of a standard normal matrix, with each
    column multiplied by the sign of the corresponding diagonal entry of R. LAPACK may
    return either sign on that diagonal, so without the last step the matrix would depend
    on the library build; with it the matrix is also Haar-distributed over seeds. Rows are
    vectors, so a batch ``x`` becomes ``x @ matrix``.
    """
    dim = _dimension(dim)
    g = np.random.default_rng(seed).standard_normal((dim, dim))
    q, r = np.linalg.qr(g)
    return np.ascontiguousarray(q * np.where(np.diag(r) < 0, -1.0, 1.0))


def signed_permutation(dim: int, seed: int) -> np.ndarray:
    """A random signed permutation as a ``[dim, dim]`` float64 matrix.

    Every column has exactly one non-zero entry, +1 or -1, so the matrix is orthogonal
    and exact in floating point: ``x @ matrix`` has coordinate ``j`` equal to coordinate
    ``perm[j]`` of ``x`` times a sign. The permutation and the signs come from
    ``numpy.random.default_rng(seed)``. Coordinate codecs that treat each axis alone
    (sign bits, fixed-step integers) can be invariant under this and not under a general
    rotation; that is a property of the codec, checked, not assumed, by ``basis_probe``.
    """
    dim = _dimension(dim)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dim)
    signs = np.where(rng.integers(0, 2, size=dim) == 0, -1.0, 1.0)
    matrix = np.zeros((dim, dim))
    matrix[perm, np.arange(dim)] = signs
    return matrix


def _copy(a: np.ndarray | None) -> np.ndarray | None:
    return None if a is None else a.copy()


def apply_basis(corpus: MultiVectorCorpus, matrix: Any) -> MultiVectorCorpus:
    """A new corpus whose vectors are ``vectors @ matrix``; the input is not modified.

    The product is taken in float64, in blocks of rows, and cast back to the corpus dtype
    (a float16 corpus is rounded to float16 again, which is a real change of the values).
    Offsets, ids, metadata and the aligned arrays (``protected``, ``positions``,
    ``weights``, ``importance``) are copied unchanged; ``images`` are the same objects.

    ``matrix`` must be square with the corpus dimension. It is **not** checked for
    orthogonality: only an orthogonal matrix keeps inner products, and with them MaxSim
    scores. Use ``random_orthogonal`` or ``signed_permutation``; ``basis_probe`` checks that
    the float scores really are unchanged.
    """
    if not isinstance(corpus, MultiVectorCorpus):
        raise TypeError(f"corpus must be a MultiVectorCorpus, got {type(corpus).__name__}")
    m = np.asarray(matrix, dtype=np.float64)
    d = corpus.dimension
    if m.shape != (d, d):
        raise ValueError(f"matrix must be [{d}, {d}] for a {d}-dimensional corpus, got {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("matrix has non-finite entries")
    out = np.empty(corpus.vectors.shape, dtype=corpus.dtype)
    step = max(1, _BLOCK_BYTES // (8 * d))
    for lo in range(0, out.shape[0], step):
        block = np.asarray(corpus.vectors[lo : lo + step], dtype=np.float64)
        out[lo : lo + step] = (block @ m).astype(corpus.dtype)
    return MultiVectorCorpus(
        out,
        corpus.offsets.copy(),
        ids=list(corpus.ids),
        metadata=corpus.metadata,
        protected=_copy(corpus.protected),
        positions=_copy(corpus.positions),
        weights=_copy(corpus.weights),
        importance=_copy(corpus.importance),
        images=corpus.images,
        attrs=corpus.attrs,
    )


def reorder_rows(corpus: MultiVectorCorpus, seed: int) -> MultiVectorCorpus:
    """A new corpus with the vectors of every document in a random order.

    Each document is permuted on its own, so the set of vectors of a document, and with it
    every MaxSim score, is unchanged. The aligned arrays are permuted with the vectors, so
    each row keeps its annotation. Offsets, ids, metadata and ``images`` are unchanged.
    The permutations come from ``numpy.random.default_rng(seed)``, one per document in
    corpus order.
    """
    if not isinstance(corpus, MultiVectorCorpus):
        raise TypeError(f"corpus must be a MultiVectorCorpus, got {type(corpus).__name__}")
    rng = np.random.default_rng(seed)
    order = np.arange(corpus.num_vectors)
    for i in range(len(corpus)):
        lo, hi = corpus.span(i)
        order[lo:hi] = lo + rng.permutation(hi - lo)

    def take(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else a[order]

    return MultiVectorCorpus(
        corpus.vectors[order],
        corpus.offsets.copy(),
        ids=list(corpus.ids),
        metadata=corpus.metadata,
        protected=take(corpus.protected),
        positions=take(corpus.positions),
        weights=take(corpus.weights),
        importance=take(corpus.importance),
        images=corpus.images,
        attrs=corpus.attrs,
    )
