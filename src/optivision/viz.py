"""Visualising what the pruner kept.

A picture of the keep-mask over a real page is the single most convincing
artefact this project produces: it shows at a glance that the discarded patches
are margins and blank paper, not content. Used by the demo app and by
``optivision explain`` to generate figures for the report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .types import PrunedPage

KEEP_TINT = (46, 160, 67)  # green: survives into the index
DROP_TINT = (200, 60, 60)  # red: dropped as blank


def overlay_mask(
    image: Image.Image,
    keep_mask: np.ndarray,
    alpha: float = 0.35,
    grid_lines: bool = True,
) -> Image.Image:
    """Tint each patch cell by whether it survived pruning."""
    rows, cols = keep_mask.shape
    base = image.convert("RGB")
    w, h = base.size

    tint = np.zeros((rows, cols, 3), dtype=np.uint8)
    tint[keep_mask] = KEEP_TINT
    tint[~keep_mask] = DROP_TINT
    layer = Image.fromarray(tint).resize((w, h), Image.NEAREST)

    out = Image.blend(base, layer, alpha)

    if grid_lines:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(out)
        for r in range(1, rows):
            y = int(r * h / rows)
            draw.line([(0, y), (w, y)], fill=(255, 255, 255), width=1)
        for c in range(1, cols):
            x = int(c * w / cols)
            draw.line([(x, 0), (x, h)], fill=(255, 255, 255), width=1)
    return out


def overlay_saliency(image: Image.Image, saliency: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Heat-map of the raw saliency scores (before thresholding)."""
    base = image.convert("RGB")
    w, h = base.size
    s = np.clip(saliency, 0.0, 1.0)
    # blue (cold, blank) -> yellow (hot, dense ink)
    heat = np.stack(
        [(s * 255).astype(np.uint8), (s * 220).astype(np.uint8), ((1 - s) * 200).astype(np.uint8)],
        axis=-1,
    )
    layer = Image.fromarray(heat).resize((w, h), Image.BILINEAR)
    return Image.blend(base, layer, alpha)


def explain_page(
    image: Image.Image, pruned: PrunedPage, out_path: str | Path | None = None
) -> Image.Image:
    """Side-by-side: original | saliency | keep-mask, with a caption line."""
    from PIL import ImageDraw

    thumb = image.convert("RGB")
    w, h = thumb.size
    panels = [thumb]
    if pruned.saliency is not None:
        panels.append(overlay_saliency(thumb, pruned.saliency))
    panels.append(overlay_mask(thumb, pruned.keep_mask))

    pad, caption_h = 8, 26
    canvas = Image.new(
        "RGB", (w * len(panels) + pad * (len(panels) + 1), h + caption_h + pad * 2), (250, 250, 250)
    )
    for i, panel in enumerate(panels):
        canvas.paste(panel, (pad + i * (w + pad), pad))

    draw = ImageDraw.Draw(canvas)
    kept = int(pruned.keep_mask.sum())
    total = pruned.keep_mask.size
    caption = (
        f"{pruned.ref.page_id}   patches {kept}/{total} kept "
        f"({kept / max(1, total):.0%})   vectors stored {pruned.n_kept} "
        f"of {pruned.n_tokens_before} ({pruned.keep_ratio:.0%})"
    )
    draw.text((pad, h + pad + 6), caption, fill=(30, 30, 30))

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
    return canvas
