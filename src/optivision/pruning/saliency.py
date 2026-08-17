"""Image-space saliency: which patches of a scanned page actually carry ink.

The whole premise of OptiVision RAG is that a document page is mostly paper.
We measure that directly on the pixels — before the model ever sees the page —
so the estimate costs microseconds and needs no attention maps or extra passes.

Two signals, both computed per grid cell:

    ink density   how much of the cell is darker than the paper background
    edge energy   how much local contrast the cell has (glyph strokes, rules,
                  stamp edges, table lines)

Ink alone would drop a faint handwritten annotation; edges alone would keep
scanner noise on an empty margin. The weighted sum keeps both cheap and robust.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Cells are measured at CELL_SAMPLES x CELL_SAMPLES pixels each, so a 32x32 grid
# is analysed on a 256x256 image: enough to see strokes, cheap enough to ignore.
CELL_SAMPLES = 8


def _paper_level(gray: np.ndarray) -> float:
    """Estimate the background (paper) intensity of a scan.

    A clean scan peaks near 1.0, a yellowed or grey photocopy much lower, so a
    fixed white point mislabels whole pages as ink. The 95th percentile tracks
    the paper wherever it sits while ignoring specular highlights.
    """
    return float(np.percentile(gray, 95))


def _block_mean(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Average ``x`` down to a rows x cols grid (x's shape is an exact multiple)."""
    h, w = x.shape
    return x.reshape(rows, h // rows, cols, w // cols).mean(axis=(1, 3))


def _robust_unit(x: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """Scale to roughly [0, 1] using a high percentile instead of the max.

    A single dark speck (dust, a punch hole, a staple shadow) would otherwise
    squash the whole page toward zero and make every real glyph look blank.
    """
    hi = float(np.percentile(x, pct))
    if hi <= 1e-8:
        return np.zeros_like(x)
    return np.clip(x / hi, 0.0, 1.0)


def patch_saliency(
    image: Image.Image,
    rows: int,
    cols: int,
    ink_weight: float = 0.6,
    edge_weight: float = 0.4,
) -> np.ndarray:
    """Return a float32 [rows, cols] saliency map in [0, 1].

    Cell (r, c) covers the fractional region [c/cols, (c+1)/cols] x
    [r/rows, (r+1)/rows] of the page, which is exactly how a ViT patch grid maps
    back to the page whether or not the model preserved the aspect ratio.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"invalid grid {rows}x{cols}")

    target = (cols * CELL_SAMPLES, rows * CELL_SAMPLES)  # PIL wants (width, height)
    gray = np.asarray(image.convert("L").resize(target, Image.BILINEAR), dtype=np.float32) / 255.0

    paper = _paper_level(gray)
    ink = np.clip(paper - gray, 0.0, None)  # 0 on paper, positive on ink
    ink_cells = _block_mean(ink, rows, cols)

    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:, :] = np.abs(np.diff(gray, axis=0))
    gx[:, 1:] = np.abs(np.diff(gray, axis=1))
    edge_cells = _block_mean(gy + gx, rows, cols)

    total = max(ink_weight + edge_weight, 1e-8)
    saliency = (ink_weight * _robust_unit(ink_cells) + edge_weight * _robust_unit(edge_cells)) / total
    return saliency.astype(np.float32)
