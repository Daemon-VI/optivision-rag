"""The original two pruning stages, as composable token reducers.

Both wrap the tested implementations in :mod:`optivision.pruning` rather than
reimplementing them, and both reproduce the original ``TokenPruner`` exactly
when chained (``tests/test_stages.py`` checks this row for row):

    TokenPruner(spatial + redundancy)  ==  SpatialPruner() -> RedundancyPruner()
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..pruning.redundancy import prune_redundant
from ..pruning.saliency import patch_saliency
from ..pruning.spatial import build_keep_mask
from .base import DocView, Reduction, TokenReducer, labels_from_clusters, register


@register
class SpatialPruner(TokenReducer):
    """Drop patches that sit on blank paper, judged from the page pixels.

    Needs what only page-image models have: a page image per document, the
    patch grid shape (``grid_rows`` / ``grid_cols`` in metadata) and patch
    ``positions``. It raises on anything else rather than guessing, and it does
    little on dense pages (measured: 1000 of 1031 ColPali tokens kept on
    ViDoRe InfoVQA). For vector-only inputs use a merging stage instead.

    Kept rows come out in row-major grid order with the pixel saliency as their
    ``importance``, which is what the next stage (redundancy) visits them by.
    """

    name: ClassVar[str] = "spatial"
    produces_importance: ClassVar[bool] = True

    def __init__(
        self,
        ink_weight: float = 0.6,
        edge_weight: float = 0.4,
        blank_threshold: float = 0.02,
        keep_ratio: float | None = None,
        min_keep: int = 16,
        dilate: int = 1,
    ) -> None:
        self.ink_weight = ink_weight
        self.edge_weight = edge_weight
        self.blank_threshold = blank_threshold
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep
        self.dilate = dilate

    def params(self) -> dict[str, Any]:
        return {
            "ink_weight": self.ink_weight,
            "edge_weight": self.edge_weight,
            "blank_threshold": self.blank_threshold,
            "keep_ratio": self.keep_ratio,
            "min_keep": self.min_keep,
            "dilate": self.dilate,
        }

    def reduce(self, doc: DocView) -> Reduction:
        rows, cols = doc.metadata.get("grid_rows"), doc.metadata.get("grid_cols")
        if doc.image is None or rows is None or cols is None or doc.positions is None:
            raise ValueError(
                "SpatialPruner needs page images, a patch grid (grid_rows/grid_cols) and "
                "patch positions. This corpus has none of them for document "
                f"{doc.index}; use a vector-only stage such as AdaptiveMerge instead."
            )
        rows, cols = int(rows), int(cols)
        saliency = patch_saliency(doc.image, rows, cols, ink_weight=self.ink_weight, edge_weight=self.edge_weight)
        keep = build_keep_mask(
            saliency,
            blank_threshold=self.blank_threshold,
            keep_ratio=self.keep_ratio,
            min_keep=self.min_keep,
            dilate=self.dilate,
        ).reshape(-1)

        on_grid = np.isfinite(doc.positions).all(axis=1)
        cell = np.full(doc.vectors.shape[0], -1, dtype=np.int64)
        r = np.floor(doc.positions[on_grid, 0] * rows).astype(np.int64)
        c = np.floor(doc.positions[on_grid, 1] * cols).astype(np.int64)
        cell[on_grid] = np.clip(r, 0, rows - 1) * cols + np.clip(c, 0, cols - 1)

        # Rows that are not on the grid cannot be judged from pixels: keep them.
        survives = np.where(on_grid, keep[np.maximum(cell, 0)], True)
        kept_rows = np.flatnonzero(survives)
        # Row-major grid order (off-grid rows last), matching TokenPruner.
        sort_key = np.where(cell[kept_rows] >= 0, cell[kept_rows], rows * cols + kept_rows)
        kept_rows = kept_rows[np.argsort(sort_key, kind="stable")]

        labels = np.full(doc.vectors.shape[0], -1, dtype=np.int64)
        labels[kept_rows] = np.arange(kept_rows.size)
        flat = saliency.reshape(-1)
        importance = np.where(cell[kept_rows] >= 0, flat[np.maximum(cell[kept_rows], 0)], 1.0)
        return Reduction(doc.vectors[kept_rows], labels, importance=importance.astype(np.float32))


@register
class RedundancyPruner(TokenReducer):
    """Greedy single-pass clustering of near-duplicate vectors (cosine >= threshold).

    Visits rows by descending ``importance`` when there is one (so the most
    salient row leads its cluster) and in stored order otherwise; each cluster
    becomes its renormalised mean (``merge=True``) or its leader.
    """

    name: ClassVar[str] = "redundancy"

    def __init__(self, threshold: float = 0.92, merge: bool = True, max_pairwise: int = 4096) -> None:
        self.threshold = threshold
        self.merge = merge
        self.max_pairwise = max_pairwise

    def params(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "merge": self.merge, "max_pairwise": self.max_pairwise}

    def reduce(self, doc: DocView) -> Reduction:
        n = doc.vectors.shape[0]
        if n <= 1:
            return Reduction(doc.vectors.copy(), np.arange(n, dtype=np.int64))
        order = np.argsort(-doc.importance) if doc.importance is not None else None
        vectors, clusters = prune_redundant(
            doc.vectors, threshold=self.threshold, order=order, merge=self.merge,
            max_pairwise=self.max_pairwise,
        )
        return Reduction(vectors, labels_from_clusters(clusters, n))
