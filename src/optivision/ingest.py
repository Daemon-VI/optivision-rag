"""Turn a folder of PDFs and scans into page images the encoder can read."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from .config import IngestConfig
from .types import PageRef

Image.MAX_IMAGE_PIXELS = 300_000_000  # large scans are normal here, not an attack


def downscale(image: Image.Image, max_side: int) -> Image.Image:
    """Cap the long edge. Saliency is computed on a 256px grid anyway, and the
    encoder resizes to its own resolution, so anything larger is wasted I/O."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def pdf_pages(path: Path, dpi: int = 150, max_side: int = 1536) -> Iterator[tuple[int, Image.Image]]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=dpi / 72.0)
            image = bitmap.to_pil().convert("RGB")
            yield i + 1, downscale(image, max_side)
    finally:
        doc.close()


def iter_pages(root: str | Path, cfg: IngestConfig | None = None) -> Iterator[tuple[PageRef, Image.Image]]:
    """Yield every page of every supported document under ``root``.

    ``root`` may be a single file or a directory tree. Document id is the path
    relative to root (without extension), so page ids stay stable across runs.
    """
    cfg = cfg or IngestConfig()
    root = Path(root)
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in cfg.formats)
        base = root

    for path in files:
        doc_id = path.relative_to(base).with_suffix("").as_posix()
        if path.suffix.lower() == ".pdf":
            for page_no, image in pdf_pages(path, dpi=cfg.dpi, max_side=cfg.max_side):
                yield PageRef(doc_id=doc_id, page_no=page_no, source_path=str(path)), image
        else:
            with Image.open(path) as im:
                image = downscale(im.convert("RGB"), cfg.max_side)
            yield PageRef(
                doc_id=doc_id, page_no=1, source_path=str(path), image_path=str(path)
            ), image


def count_pages(root: str | Path, cfg: IngestConfig | None = None) -> int:
    cfg = cfg or IngestConfig()
    root = Path(root)
    files = [root] if root.is_file() else [
        p for p in root.rglob("*") if p.suffix.lower() in cfg.formats
    ]
    total = 0
    for path in files:
        if path.suffix.lower() == ".pdf":
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(str(path))
            total += len(doc)
            doc.close()
        else:
            total += 1
    return total
