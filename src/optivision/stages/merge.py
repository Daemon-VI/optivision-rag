"""Training-free token reduction that needs nothing but the vectors.

Three reducers and one control:

``AdaptiveMerge``  (numpy only)
    Covers each document with as few representatives as possible. Greedy
    farthest-point seeding adds a centre where the document is least covered,
    until every vector has cosine >= ``radius`` to some centre (or a budget is
    hit); vectors then merge into their nearest centre. The count adapts per
    document -- a dense table keeps more vectors than a title page -- from one
    global knob, and the knob has a geometric meaning: for unit vectors, a
    query vector's best-match score moves by at most ``||d - c|| =
    sqrt(2 * (1 - cos))`` when ``d`` is replaced by a centre ``c`` it is
    within ``cos`` of. Replacing members by their *mean* rather than the centre
    loosens that bound slightly; it is a design rationale, not a guarantee.

``HierarchicalMerge``  (needs SciPy)
    Ward agglomerative clustering to a fixed fraction of the vectors, each
    cluster pooled to its mean -- the token-pooling recipe of Clavie et al.
    (2024), here on normalised vectors.

``RedundancyPruner``  (in :mod:`.prune`)
    The original greedy single-pass leader clustering at a cosine threshold.

``RandomPruner``  (control)
    Keeps a random fraction. Any method that does not beat it at a matched
    budget is not selecting anything.

All of them leave protected rows alone and keep a row's magnitude: a merged
vector is rescaled to its members' weighted mean norm, which for unit inputs is
plain renormalisation.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import DocView, Reduction, TokenReducer, fit_sample_rows, register


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _pool(vectors: np.ndarray, labels: np.ndarray, k: int, weights: np.ndarray) -> np.ndarray:
    """Weighted mean per cluster, rescaled to the members' weighted mean norm."""
    d = vectors.shape[1]
    w = weights.astype(np.float64)
    sums = np.zeros((k, d))
    np.add.at(sums, labels, vectors.astype(np.float64) * w[:, None])
    norms = np.linalg.norm(vectors, axis=1).astype(np.float64)
    norm_sum = np.bincount(labels, weights=norms * w, minlength=k)
    wsum = np.bincount(labels, weights=w, minlength=k)
    target = norm_sum / np.maximum(wsum, 1e-12)
    length = np.linalg.norm(sums, axis=1)
    scale = np.where(length > 0, target / np.maximum(length, 1e-12), 0.0)
    return (sums * scale[:, None]).astype(np.float32)


def _budget(n: int, ratio: float | None, max_vectors: int | None, min_vectors: int) -> int:
    k = n
    if ratio is not None:
        k = min(k, max(1, math.ceil(ratio * n)))
    if max_vectors is not None:
        k = min(k, int(max_vectors))
    return max(min(n, min_vectors), min(k, n))


@register
class AdaptiveMerge(TokenReducer):
    """Cover each document with representatives within ``radius`` cosine.

    Args:
        radius: stop adding centres once every vector has cosine >= radius to
            one. ``None`` means "use the budget only".
        ratio: cap on output vectors as a fraction of the reducible input.
        max_vectors: absolute cap per document.
        min_vectors: floor per document (a page must stay retrievable).
        refine: spherical k-means iterations after seeding (0 = none).
        representative: ``"mean"`` (pool members) or ``"center"`` (keep the
            seed vector itself, nothing synthesised).
        seeding: ``"farthest"`` (start from the vector closest to the document
            mean) or ``"importance"`` (start from the most important rows, when
            the corpus carries an ``importance`` signal such as attention).
        n_importance_seeds: how many top-importance rows to seed with.
        center: ``"mean"`` measures similarity after removing a fitted corpus
            mean (pooling still uses the original vectors). Needed for
            anisotropic encoders: the SciFact text ColBERT measured here has a
            median nearest-neighbour cosine of 0.991 inside a document, so an
            absolute cosine radius lumps most of a document together.

    At least one of ``radius`` / ``ratio`` / ``max_vectors`` must be set.
    """

    name: ClassVar[str] = "adaptive_merge"

    def __init__(
        self,
        radius: float | None = 0.8,
        ratio: float | None = None,
        max_vectors: int | None = None,
        min_vectors: int = 1,
        refine: int = 0,
        representative: str = "mean",
        seeding: str = "farthest",
        n_importance_seeds: int = 0,
        center: str | None = None,
    ) -> None:
        if radius is None and ratio is None and max_vectors is None:
            raise ValueError("set radius, ratio or max_vectors")
        if radius is not None and not -1.0 <= radius <= 1.0:
            raise ValueError("radius is a cosine similarity in [-1, 1]")
        if ratio is not None and not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        if representative not in {"mean", "center"}:
            raise ValueError("representative must be 'mean' or 'center'")
        if seeding not in {"farthest", "importance"}:
            raise ValueError("seeding must be 'farthest' or 'importance'")
        if representative == "center" and refine:
            raise ValueError("representative='center' keeps the seed vectors; it cannot be refined")
        if center not in (None, "mean"):
            raise ValueError("center must be None or 'mean'")
        self.radius = radius
        self.ratio = ratio
        self.max_vectors = max_vectors
        self.min_vectors = min_vectors
        self.refine = refine
        self.representative = representative
        self.seeding = seeding
        self.n_importance_seeds = n_importance_seeds
        self.center = center
        self.mean: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.center is None or self.mean is not None

    def fit(self, corpus: Any, queries: Any = None) -> AdaptiveMerge:
        if self.center == "mean":
            v = np.asarray(corpus.vectors)
            rows = fit_sample_rows(v.shape[1], 8)
            if v.shape[0] > rows:
                v = v[np.sort(np.random.default_rng(0).choice(v.shape[0], rows, replace=False))]
            self.mean = np.asarray(v, dtype=np.float64).mean(axis=0).astype(np.float32)
        return self

    def params(self) -> dict[str, Any]:
        return {
            "radius": self.radius,
            "ratio": self.ratio,
            "max_vectors": self.max_vectors,
            "min_vectors": self.min_vectors,
            "refine": self.refine,
            "representative": self.representative,
            "seeding": self.seeding,
            "n_importance_seeds": self.n_importance_seeds,
            "center": self.center,
        }

    def _seeds(self, u: np.ndarray, doc: DocView) -> list[int]:
        if self.seeding == "importance" and doc.importance is not None and self.n_importance_seeds > 0:
            top = np.argsort(-doc.importance, kind="stable")[: self.n_importance_seeds]
            return [int(i) for i in top]
        return [int(np.argmax(u @ u.mean(axis=0)))]

    def reduce(self, doc: DocView) -> Reduction:
        v = doc.vectors
        n = v.shape[0]
        if n <= 1:
            return Reduction(v.copy(), np.arange(n, dtype=np.int64))
        if self.center == "mean":
            if self.mean is None:
                raise RuntimeError("AdaptiveMerge(center='mean') must be fit before use")
            u = _unit(v - self.mean)
        else:
            u = _unit(v)
        cap = _budget(n, self.ratio, self.max_vectors, self.min_vectors)

        centres = self._seeds(u, doc)[:cap]
        sims = u @ u[centres].T  # [n, k]
        best = sims.max(axis=1)
        assign = sims.argmax(axis=1)
        while len(centres) < cap:
            if len(centres) >= self.min_vectors and self.radius is not None and best.min() >= self.radius:
                break
            nxt = int(np.argmin(best))
            if best[nxt] >= 1.0 - 1e-7:  # everything left is a duplicate of a centre
                break
            centres.append(nxt)
            s = u @ u[nxt]
            closer = s > best
            best[closer] = s[closer]
            assign[closer] = len(centres) - 1
        k = len(centres)

        for _ in range(max(0, self.refine)):
            c = _unit(_pool(u, assign, k, doc.weights))
            new_assign = (u @ c.T).argmax(axis=1)
            if np.array_equal(new_assign, assign):
                break
            assign = new_assign
            used = np.unique(assign)
            if used.size < k:  # a centre lost all its members: relabel densely
                remap = np.full(k, -1, dtype=np.int64)
                remap[used] = np.arange(used.size)
                assign, k = remap[assign], used.size
                centres = [centres[i] for i in used]

        if self.representative == "center":
            out = v[np.asarray(centres, dtype=np.int64)].astype(np.float32, copy=True)
        else:
            out = _pool(v, assign, k, doc.weights)
        return Reduction(out, assign.astype(np.int64))


@register
class HierarchicalMerge(TokenReducer):
    """Ward agglomerative clustering of each document's vectors (needs SciPy).

    Two ways to decide where to cut the tree:

    ``ratio``         a fixed fraction of the vectors per document. ``0.5`` is
                      Clavie et al.'s pool factor 2, ``1/3`` pool factor 3.
    ``max_distance``  a Ward merge height, so the count adapts per document.
                      For two single unit vectors the height is their Euclidean
                      distance ``sqrt(2 * (1 - cos))`` -- 0.447 at cosine 0.9 --
                      and it grows with cluster size. When both are set,
                      ``ratio`` caps the count.
    """

    name: ClassVar[str] = "hierarchical_merge"

    def __init__(self, ratio: float | None = 0.5, max_distance: float | None = None, min_vectors: int = 1) -> None:
        if ratio is None and max_distance is None:
            raise ValueError("set ratio, max_distance or both")
        if ratio is not None and not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        if max_distance is not None and max_distance <= 0:
            raise ValueError("max_distance must be positive")
        self.ratio = ratio
        self.max_distance = max_distance
        self.min_vectors = min_vectors

    def params(self) -> dict[str, Any]:
        return {"ratio": self.ratio, "max_distance": self.max_distance, "min_vectors": self.min_vectors}

    def reduce(self, doc: DocView) -> Reduction:
        v = doc.vectors
        n = v.shape[0]
        k = _budget(n, self.ratio, None, self.min_vectors)
        if n <= 1 or (self.max_distance is None and k >= n):
            return Reduction(v.copy(), np.arange(n, dtype=np.int64))
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("HierarchicalMerge needs SciPy: pip install 'optivision-rag[merge]'") from exc
        z = linkage(_ward_input(_unit(v).astype(np.float64)), method="ward")
        if self.max_distance is not None:
            raw = fcluster(z, t=self.max_distance, criterion="distance")
            n_clusters = int(raw.max())
            if n_clusters > k or n_clusters < min(self.min_vectors, n):
                raw = fcluster(z, t=min(max(n_clusters, self.min_vectors), k), criterion="maxclust")
        else:
            raw = fcluster(z, t=k, criterion="maxclust")
        _, labels = np.unique(raw, return_inverse=True)
        labels = labels.astype(np.int64)
        return Reduction(_pool(v, labels, int(labels.max()) + 1, doc.weights), labels)



#: From this width up, pairwise distances come from one matrix product instead
#: of SciPy's pair-by-pair loop (35 min -> ~1 min per 500 ViDoRe pages at 2,560-d).
#: Below it the original call is kept, so every measured 96-128-d result is unchanged.
WIDE_DIM = 512


def _ward_input(x: np.ndarray) -> np.ndarray:
    """What ``linkage(..., "ward")`` gets: the vectors, or for wide unit vectors
    their condensed Euclidean distances, ||a - b|| = sqrt(2 - 2 a.b)."""
    if x.shape[1] < WIDE_DIM:
        return x
    from scipy.spatial.distance import squareform

    d2 = 2.0 - 2.0 * (x @ x.T)
    np.fill_diagonal(d2, 0.0)
    return squareform(np.sqrt(np.clip(d2, 0.0, None)), checks=False)

@register
class RandomPruner(TokenReducer):
    """Keep a random ``ratio`` of each document's vectors. A control, not a method."""

    name: ClassVar[str] = "random_prune"

    def __init__(self, ratio: float = 0.5, seed: int = 0, min_vectors: int = 1) -> None:
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = ratio
        self.seed = seed
        self.min_vectors = min_vectors

    def params(self) -> dict[str, Any]:
        return {"ratio": self.ratio, "seed": self.seed, "min_vectors": self.min_vectors}

    def reduce(self, doc: DocView) -> Reduction:
        n = doc.vectors.shape[0]
        k = _budget(n, self.ratio, None, self.min_vectors)
        rng = np.random.default_rng((self.seed, doc.index))
        keep = np.sort(rng.choice(n, size=k, replace=False))
        labels = np.full(n, -1, dtype=np.int64)
        labels[keep] = np.arange(k)
        return Reduction(doc.vectors[keep].copy(), labels)
