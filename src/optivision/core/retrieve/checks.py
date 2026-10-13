"""Input checks shared by the retrievers and the backends.

They raise instead of converting. A float ``k``, a float candidate id, a NaN query token or a
query of the wrong dimension each become a message that says what was wrong, where the library
underneath would cast the value, return an index-order "ranking" built from NaN scores, or fail
inside a matrix product.

Nothing here touches a device or reads the environment.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from ...representation import MultiVectorCorpus, as_corpus

#: rows examined per call of ``np.isfinite``: bounds the boolean temporary
_FINITE_BLOCK_ROWS = 65_536


def as_int(value: Any, name: str) -> int:
    """``value`` as a Python int, or ``TypeError``. Booleans and floats are refused."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got the boolean {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__} {value!r}") from None


def check_k(k: Any) -> int:
    """A result depth: an integer ``>= 1``.

    ``rank(scores, -1)`` silently drops the last column and ``rank(scores, 0)`` returns nothing,
    so depths are validated here rather than passed on.
    """
    k = as_int(k, "k")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return k


def check_candidates(candidates: Any) -> int:
    """A shortlist size: an integer ``>= 1`` (larger than the corpus is clamped by the caller)."""
    candidates = as_int(candidates, "candidates")
    if candidates < 1:
        raise ValueError(f"candidates must be >= 1, got {candidates}")
    return candidates


def as_queries(queries: Any, dim: int | None = None) -> MultiVectorCorpus:
    """Queries as a :class:`~optivision.representation.MultiVectorCorpus`, checked.

    Accepts what :func:`~optivision.representation.as_corpus` accepts (a corpus, a list of
    ``[n_i, d]`` arrays, one representation). Integer arrays are refused by the corpus
    constructor; nothing is narrowed here. Raises ``ValueError`` when a query vector is NaN or
    infinite (the scorer would return an all-NaN row, which ``rank`` turns into index order) and,
    when ``dim`` is given, when the queries are not ``dim``-dimensional. A query set without any
    vector is accepted whatever its width: every score is then 0 (or ``-inf`` for an empty
    document) and no product is formed.
    """
    corpus = as_corpus(queries)
    if corpus.num_vectors == 0:
        return corpus
    if dim is not None and corpus.dimension != dim:
        raise ValueError(
            f"queries have dimension {corpus.dimension} but the store holds {dim}-dimensional "
            "vectors (if the documents were projected, pass the fitted pipeline so that queries "
            "are projected the same way)"
        )
    row = first_non_finite_row(corpus.vectors)
    if row is not None:
        query = int(np.searchsorted(corpus.offsets, row, side="right")) - 1
        raise ValueError(f"queries contain non-finite values (first at vector {row}, query {query})")
    return corpus


def first_non_finite_row(vectors: Any) -> int | None:
    """Index of the first row of a 2-D float array that holds NaN or infinity, else ``None``.

    The scan runs in blocks of ``_FINITE_BLOCK_ROWS`` rows in the array's own dtype: nothing is
    converted and no copy of the whole array is made.
    """
    for lo in range(0, vectors.shape[0], _FINITE_BLOCK_ROWS):
        block = np.asarray(vectors[lo : lo + _FINITE_BLOCK_ROWS])
        finite_rows = np.isfinite(block).all(axis=1)
        if not finite_rows.all():
            return lo + int(np.argmin(finite_rows))
    return None


def check_finite_store(store: Any, ids: Any) -> None:
    """Refuse a store of float vectors that holds NaN or infinity, naming the first document.

    Only a store with a floating-point ``vectors`` array is scanned (a ``MultiVectorCorpus``).
    A store of integer codes decodes to finite values and a store without ``vectors`` has
    nothing to scan, so neither is read. ``ValueError`` names the vector and the document.
    """
    vectors = getattr(store, "vectors", None)
    if not isinstance(vectors, np.ndarray) or vectors.dtype.kind != "f" or vectors.ndim != 2:
        return
    row = first_non_finite_row(vectors)
    if row is not None:
        doc = int(np.searchsorted(np.asarray(store.offsets), row, side="right")) - 1
        raise ValueError(
            "the store contains non-finite values (NaN or infinity): first at vector "
            f"{row}, document {doc} ({ids[doc]!r}); a NaN document would be scored NaN and "
            "silently never returned, an infinite one would rank first"
        )


def _fitted_input_dim(pipeline: Any) -> tuple[int, str] | None:
    """Width and name of the first dimension stage, when its fitted state shows the width.

    PCA components, a random-projection matrix and a rotation matrix are ``[d_in, ...]``
    arrays. ``None`` when that stage keeps no such array (truncation, a third-party stage).
    """
    for stage in getattr(pipeline, "stages", ()):
        if getattr(stage, "kind", None) != "dimension":
            continue
        inner = getattr(stage, "inner", stage)  # the ``project`` facade holds the real stage
        for attr in ("components", "matrix"):
            array = getattr(inner, attr, None)
            if isinstance(array, np.ndarray) and array.ndim == 2:
                return int(array.shape[0]), str(stage.name)
        return None
    return None


def check_projection(queries: MultiVectorCorpus, pipeline: Any, store_dim: int | None) -> None:
    """``ValueError`` that names the queries and both dimensions when they cannot be scored.

    The query dimension is compared with the width the pipeline's first dimension stage was
    fitted on when its fitted state shows it. Otherwise one query vector is projected and its
    dimension compared with the store's, and a shape error raised inside the projection is
    re-raised with a message that says what was compared. Queries without vectors pass.
    """
    if queries.num_vectors == 0:
        return
    fitted = _fitted_input_dim(pipeline)
    if fitted is not None and queries.dimension != fitted[0]:
        raise ValueError(
            f"queries have dimension {queries.dimension} but the pipeline's first dimension "
            f"stage ({fitted[1]}) was fitted on {fitted[0]}-dimensional vectors: pass queries "
            "in the same space as the documents the pipeline was fitted on"
        )
    if store_dim is None:
        return
    probe = MultiVectorCorpus(np.asarray(queries.vectors[:1]), [0, 1])
    try:
        width = pipeline.transform_queries(probe).dimension
    except ValueError as exc:
        raise ValueError(
            f"queries have dimension {queries.dimension}, which the pipeline cannot project "
            f"for a store of {store_dim}-dimensional vectors ({exc})"
        ) from exc
    if width != store_dim:
        raise ValueError(
            f"queries have dimension {queries.dimension} and the pipeline projects them to "
            f"dimension {width}, but the store holds {store_dim}-dimensional vectors"
        )


def check_positions(candidates: Any, n_queries: int, n_docs: int) -> np.ndarray:
    """Candidate document positions ``[n_queries, m]`` as int64, validated.

    Only integer arrays are accepted (a float array would be cast without a word), every
    position must lie in ``[0, n_docs)``, and ``-1`` is not a position. ``m == 0`` is allowed.
    """
    arr = np.asarray(candidates)
    if arr.ndim != 2:
        raise ValueError(f"candidates must be 2-D [n_queries, m], got shape {arr.shape}")
    if arr.dtype.kind not in "iu":
        raise TypeError(f"candidates must be an integer array of positions, got dtype {arr.dtype}")
    if arr.shape[0] != n_queries:
        raise ValueError(f"candidates has {arr.shape[0]} rows for {n_queries} queries")
    if arr.size:
        lo, hi = int(arr.min()), int(arr.max())
        if lo < 0 or hi >= n_docs:
            raise IndexError(
                f"candidates hold positions {lo}..{hi} but the store has {n_docs} documents "
                f"(valid positions are 0..{n_docs - 1}; -1 is a padding marker, not a position)"
            )
    return arr.astype(np.int64, copy=False)
