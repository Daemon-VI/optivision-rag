"""Dimension reduction stages. Every one of them projects queries with the same map.

``PCAProjector``
    Top principal directions of the corpus vectors (optionally including sample
    query vectors: queries and documents live in the same space but not
    necessarily the same *region* of it, and MaxSim needs the directions queries
    use). Uncentred by default -- the second-moment matrix, which is what
    preserves inner products; ``center=True`` projects documents after removing
    the corpus mean, which only shifts every query token's scores by a constant
    and so leaves MaxSim rankings unchanged.

``RandomProjector``
    A random orthonormal map scaled by sqrt(d / k), preserving inner products in
    expectation (Johnson-Lindenstrauss). Needs no data; the control for PCA.

``TruncateProjector``
    Keep the first k coordinates. Correct only for Matryoshka-trained models,
    whose leading coordinates are trained to stand alone; for others it is a
    control that shows what an untrained cut costs.

None of these has been shown to preserve quality on wide (2k-4k) embeddings:
the only models measured in this repository are 128-dimensional.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..representation import MultiVectorCorpus
from .base import DimensionReducer, fit_sample_rows, register


def _sample_rows(v: np.ndarray, n: int, seed: int) -> np.ndarray:
    if v.shape[0] <= n:
        return np.asarray(v, dtype=np.float32)
    idx = np.random.default_rng(seed).choice(v.shape[0], size=n, replace=False)
    return np.asarray(v[np.sort(idx)], dtype=np.float32)


@register
class PCAProjector(DimensionReducer):
    name: ClassVar[str] = "pca"

    def __init__(
        self,
        dim: int = 64,
        center: bool = False,
        basis: str = "documents",
        query_weight: float = 0.5,
        sample: int = 200_000,
        seed: int = 0,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        if basis not in {"documents", "joint"}:
            raise ValueError("basis must be 'documents' or 'joint'")
        self.dim = dim
        self.center = center
        self.basis = basis
        self.query_weight = query_weight
        self.sample = sample
        self.seed = seed
        self.components: np.ndarray | None = None  # [d_in, dim]
        self.mean: np.ndarray | None = None
        self.explained: float | None = None

    def params(self) -> dict[str, Any]:
        return {"dim": self.dim, "center": self.center, "basis": self.basis,
                "query_weight": self.query_weight, "sample": self.sample, "seed": self.seed}

    @property
    def fitted(self) -> bool:
        return self.components is not None

    def output_dim(self, input_dim: int) -> int:
        return min(self.dim, input_dim)

    def fit(self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None) -> PCAProjector:
        rows = fit_sample_rows(corpus.dimension, 8, rows=self.sample)
        x = _sample_rows(corpus.vectors, rows, self.seed).astype(np.float64)
        self.mean = x.mean(axis=0) if self.center else np.zeros(x.shape[1])
        xc = x - self.mean
        m = xc.T @ xc / max(1, xc.shape[0])
        if self.basis == "joint":
            if queries is None or queries.num_vectors == 0:
                raise ValueError("basis='joint' needs sample queries at fit time")
            q = _sample_rows(queries.vectors, rows, self.seed + 1).astype(np.float64)
            mq = q.T @ q / max(1, q.shape[0])
            m = (1.0 - self.query_weight) * m + self.query_weight * mq
        vals, vecs = np.linalg.eigh(m)
        order = np.argsort(vals)[::-1]
        k = self.output_dim(x.shape[1])
        self.components = np.ascontiguousarray(vecs[:, order[:k]], dtype=np.float32)
        self.explained = float(vals[order[:k]].sum() / max(vals.sum(), 1e-12))
        return self

    def project(self, vectors: np.ndarray) -> np.ndarray:
        if self.components is None:
            raise RuntimeError("PCAProjector must be fit before use")
        v = np.asarray(vectors, dtype=np.float32)
        if self.center:
            v = v - self.mean.astype(np.float32)
        return v @ self.components

    def transform_queries(self, queries: MultiVectorCorpus) -> MultiVectorCorpus:
        # Queries are never centred: the mean's contribution is a per-token
        # constant, so dropping it cannot change any ranking.
        if self.components is None:
            raise RuntimeError("PCAProjector must be fit before use")
        return queries.with_vectors(np.asarray(queries.vectors, np.float32) @ self.components)


@register
class RandomProjector(DimensionReducer):
    name: ClassVar[str] = "random_projection"

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.seed = seed
        self.matrix: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {"dim": self.dim, "seed": self.seed}

    @property
    def fitted(self) -> bool:
        return self.matrix is not None

    def output_dim(self, input_dim: int) -> int:
        return min(self.dim, input_dim)

    def fit(self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None) -> RandomProjector:
        d = corpus.dimension
        k = self.output_dim(d)
        g = np.random.default_rng(self.seed).standard_normal((d, k))
        q, _ = np.linalg.qr(g)
        self.matrix = np.ascontiguousarray(q * np.sqrt(d / k), dtype=np.float32)
        return self

    def project(self, vectors: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("RandomProjector must be fit before use")
        return np.asarray(vectors, dtype=np.float32) @ self.matrix


@register
class DimensionProjector(DimensionReducer):
    """``DimensionProjector(128, method="pca")`` -- one entry point for any projection.

    Delegates to :class:`PCAProjector` (default), :class:`RandomProjector` or
    :class:`TruncateProjector`; extra keyword arguments go to the chosen one.
    Future learned or Matryoshka-aware projections plug in here by name.
    """

    name: ClassVar[str] = "project"

    def __init__(self, dim: int = 128, method: str = "pca", **options: Any) -> None:
        methods = {"pca": PCAProjector, "random": RandomProjector, "truncate": TruncateProjector}
        if method not in methods:
            raise ValueError(f"unknown projection method {method!r}; choose from {sorted(methods)}")
        self.dim = dim
        self.method = method
        self.options = options
        self.inner = methods[method](dim=dim, **options)

    def params(self) -> dict[str, Any]:
        return {"dim": self.dim, "method": self.method, **self.options}

    @property
    def fitted(self) -> bool:
        return self.inner.fitted

    def fit(self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None) -> DimensionProjector:
        self.inner.fit(corpus, queries)
        return self

    def output_dim(self, input_dim: int) -> int:
        return self.inner.output_dim(input_dim)

    def project(self, vectors: np.ndarray) -> np.ndarray:
        return self.inner.project(vectors)

    def transform_queries(self, queries: MultiVectorCorpus) -> MultiVectorCorpus:
        return self.inner.transform_queries(queries)


@register
class TruncateProjector(DimensionReducer):
    name: ClassVar[str] = "truncate"

    def __init__(self, dim: int = 64) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim

    def params(self) -> dict[str, Any]:
        return {"dim": self.dim}

    def output_dim(self, input_dim: int) -> int:
        return min(self.dim, input_dim)

    def project(self, vectors: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(vectors, dtype=np.float32)[:, : self.dim])
