"""Seeded synthetic late-interaction data with planted relevance.

For people who add an adapter, a stage or a backend, and for this project's own tests: a
corpus of random multi-vector documents, and queries that are cut from a known document so
that the labelled relevant document is known.

Synthetic data exercise code paths. Results on them say nothing about any real encoder, and
no number computed on this data may be reported as a property of a model or of a pipeline
on real vectors. Every corpus and query set made here says so in its ``attrs``.

Both functions are deterministic: the same arguments give the same arrays. The random draws
happen in a fixed order that does not depend on the options (``anisotropy``,
``mean_shift``, ``noise``), so two calls that differ only in an option share their
underlying draws, which makes controlled comparisons possible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.vectors import MultiVectorCorpus

__all__ = ["make_corpus", "make_queries"]


def _span(tokens: int | tuple[int, int], what: str, lowest: int = 0) -> tuple[int, int]:
    """``(lo, hi)`` token counts, inclusive, from an int or a pair, with ``lo >= lowest``."""
    lo, hi = (tokens, tokens) if isinstance(tokens, (int, np.integer)) else tuple(tokens)
    if isinstance(lo, bool) or isinstance(hi, bool) or not (
        isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer))
    ):
        raise TypeError(f"{what} must be an int or a (lo, hi) pair of ints, got {tokens!r}")
    if lo < lowest or hi < max(lo, 1):
        raise ValueError(
            f"{what} must satisfy {lowest} <= lo <= hi and hi >= 1, got ({lo}, {hi})"
        )
    return int(lo), int(hi)


def _positive(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return int(value)


def make_corpus(
    n_docs: int,
    dim: int,
    *,
    tokens: int | tuple[int, int] = (8, 16),
    seed: int,
    anisotropy: float = 0.0,
    mean_shift: float = 0.0,
    dtype: Any = np.float32,
    unit_norm: bool = True,
) -> MultiVectorCorpus:
    """A corpus of ``n_docs`` random documents of ``dim``-dimensional vectors.

    Each document holds a number of vectors drawn uniformly from ``tokens`` (inclusive; an
    int means a fixed count). A vector is an independent Gaussian draw, so documents differ
    and vectors inside one document are unrelated: there is no redundancy for a token
    reducer to find unless you add it yourself.

    Args:
        seed: required; the draws come from ``numpy.random.default_rng(seed)``.
        anisotropy: 0 gives the same spread on every axis. A positive value gives axis ``j``
            a standard deviation of ``exp(-anisotropy * j / (dim - 1))``, so the leading
            coordinates carry the most variance and the ratio between the widest and the
            narrowest axis is ``exp(anisotropy)``. Data like this is in a special basis,
            which is what makes coordinate codecs sensitive to a change of basis.
        mean_shift: 0 adds nothing. A positive value adds that multiple of one fixed random
            unit direction to every vector, after the vectors have been scaled to unit
            expected squared norm (so 1.0 means the common part is as large as the random
            part), before any normalisation.
        dtype: a floating dtype; the vectors are drawn in float64 and cast at the end.
        unit_norm: L2-normalise every vector (as the measured encoders emit) before casting.

    Document ids are ``d000000``, ``d000001``, ... in order. ``attrs`` records the arguments
    and ``"synthetic": True``.

    Synthetic data exercise code paths; results on them say nothing about any real encoder.
    """
    n_docs, dim = _positive(n_docs, "n_docs"), _positive(dim, "dim")
    lo, hi = _span(tokens, "tokens")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError(f"dtype must be a floating dtype, got {dtype}")
    for what, value in ("anisotropy", anisotropy), ("mean_shift", mean_shift):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{what} must be finite and not negative, got {value!r}")

    rng = np.random.default_rng(seed)
    counts = rng.integers(lo, hi + 1, size=n_docs)
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)
    n_vectors = int(counts.sum())
    draws = rng.standard_normal((n_vectors, dim))

    scales = np.exp(-float(anisotropy) * np.arange(dim) / max(dim - 1, 1))
    x = draws * scales / np.sqrt(np.sum(scales**2)) + float(mean_shift) * direction
    if unit_norm and n_vectors:
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-300)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return MultiVectorCorpus(
        x.astype(dtype),
        offsets,
        ids=[f"d{i:06d}" for i in range(n_docs)],
        attrs={
            "synthetic": True,
            "generator": "optivision.testing.make_corpus",
            "seed": int(seed),
            "anisotropy": float(anisotropy),
            "mean_shift": float(mean_shift),
            "unit_norm": bool(unit_norm),
        },
    )


def make_queries(
    corpus: MultiVectorCorpus,
    n_queries: int,
    *,
    tokens: int | tuple[int, int] = (4, 8),
    noise: float = 0.0,
    seed: int,
) -> tuple[MultiVectorCorpus, dict[str, set[str]]]:
    """Queries cut from known documents, and the labels that say which document is relevant.

    Each query picks one target document, takes a few distinct vectors of it (a number
    drawn uniformly from ``tokens``, at most the document's size) and adds Gaussian noise,
    so the target document is the labelled relevant one. Targets are distinct while there
    are enough documents with vectors, and repeat after that.

    Args:
        tokens: how many vectors a query takes, drawn uniformly from this inclusive range (an
            int is a fixed count); at least 1, and capped by the size of the target document.
        noise: standard deviation of the added noise as a multiple of the vector's own
            norm (per coordinate it is ``noise * norm / sqrt(dim)``, so ``noise=1`` adds a
            vector about as long as the signal). The noisy vector is rescaled to the norm
            of the vector it came from. ``noise=0`` copies the vectors exactly, and a float
            index then retrieves the target for nearly every query (not provably for every
            one if the corpus is not unit-norm).
        seed: required; the draws come from ``numpy.random.default_rng(seed)``.

    Returns:
        ``(queries, qrels)``: queries as a corpus in the corpus's dtype with ids ``q00000``,
        ``q00001``, ...; ``qrels`` as ``{query id: {document id}}``, the form
        ``optivision.evaluation.relevant_from_qrels`` accepts.

    Synthetic data exercise code paths; results on them say nothing about any real encoder.
    """
    if not isinstance(corpus, MultiVectorCorpus):
        raise TypeError(f"corpus must be a MultiVectorCorpus, got {type(corpus).__name__}")
    n_queries = _positive(n_queries, "n_queries")
    lo, hi = _span(tokens, "tokens", lowest=1)  # an empty query scores 0 against everything
    if not np.isfinite(noise) or noise < 0:
        raise ValueError(f"noise must be finite and not negative, got {noise!r}")
    eligible = np.flatnonzero(corpus.counts > 0)
    if eligible.size == 0:
        raise ValueError("the corpus has no document with vectors to cut queries from")

    rng = np.random.default_rng(seed)
    targets = rng.choice(eligible, size=n_queries, replace=n_queries > eligible.size)
    dim = corpus.dimension
    arrays, ids, qrels = [], [], {}
    for j, target in enumerate(targets):
        size = int(corpus.counts[target])
        take = min(int(rng.integers(lo, hi + 1)), size)
        rows = int(corpus.offsets[target]) + rng.choice(size, size=take, replace=False)
        draws = rng.standard_normal((take, dim))
        v = np.asarray(corpus.vectors[rows], dtype=np.float64)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        if noise > 0:
            v = v + float(noise) * norms * draws / np.sqrt(dim)
            moved = np.linalg.norm(v, axis=1, keepdims=True)
            v = v * (norms / np.where(moved > 0, moved, 1.0))
        qid = f"q{j:05d}"
        arrays.append(v.astype(corpus.dtype))
        ids.append(qid)
        qrels[qid] = {corpus.ids[int(target)]}
    queries = MultiVectorCorpus.from_arrays(
        arrays, ids=ids,
        attrs={"synthetic": True, "generator": "optivision.testing.make_queries",
               "seed": int(seed), "noise": float(noise)},
    )
    return queries, qrels
