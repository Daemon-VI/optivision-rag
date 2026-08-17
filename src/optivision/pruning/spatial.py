"""Spatial pruning: drop the patch vectors that sit on blank paper."""

from __future__ import annotations

import numpy as np


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Grow a boolean grid mask by ``iterations`` cells in 4-connectivity.

    Glyphs rarely stop at a patch boundary: the top of a heading or the tail of
    a signature usually bleeds into a neighbouring cell whose own ink density is
    below threshold. Dilating buys that context back for a handful of vectors.
    """
    out = mask.copy()
    for _ in range(max(0, iterations)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def build_keep_mask(
    saliency: np.ndarray,
    blank_threshold: float = 0.02,
    keep_ratio: float | None = None,
    min_keep: int = 16,
    dilate: int = 1,
) -> np.ndarray:
    """Decide which grid cells survive.

    Two selection modes:

    ``blank_threshold``  absolute — keep whatever carries ink. The number of
        kept tokens then adapts to the page, which is the honest behaviour: a
        dense table keeps more vectors than a title page.

    ``keep_ratio``  relative — keep the top fraction of cells. Fixes the budget
        per page, which is what ablation curves and index-size guarantees need.
        ``dilate`` is ignored in this mode: growing the mask after selection
        would blow past the budget it exists to enforce (a scattered top-25%
        mask nearly triples under one dilation step), so a fixed budget means
        exactly k patches.

    ``min_keep`` is a floor for both: a nearly empty page must still be
    retrievable, and an empty vector list would break MaxSim.
    """
    rows, cols = saliency.shape
    n_cells = rows * cols

    if keep_ratio is not None:
        k = int(np.clip(round(keep_ratio * n_cells), 1, n_cells))
        flat = saliency.reshape(-1)
        # argpartition gives the top-k without a full sort
        top = np.argpartition(-flat, k - 1)[:k]
        mask = np.zeros(n_cells, dtype=bool)
        mask[top] = True
        mask = mask.reshape(rows, cols)
    else:
        mask = saliency > blank_threshold
        mask = dilate_mask(mask, dilate)

    n_kept = int(mask.sum())
    floor = min(min_keep, n_cells)
    if n_kept < floor:
        # Promote the most salient dropped cells until the floor is met.
        flat_sal = saliency.reshape(-1).copy()
        flat_mask = mask.reshape(-1)
        flat_sal[flat_mask] = -np.inf
        need = floor - n_kept
        extra = np.argpartition(-flat_sal, need - 1)[:need]
        flat_mask[extra] = True
        mask = flat_mask.reshape(rows, cols)

    return mask
