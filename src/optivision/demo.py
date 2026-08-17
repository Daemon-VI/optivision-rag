"""Single-page adapter for the demonstration UI.

This module contains **no compression logic of its own**. It calls the existing
research pipeline — :mod:`optivision.encoders`, :mod:`optivision.pruning`,
:mod:`optivision.compression`, :mod:`optivision.viz` — and packages one page's
result into a flat object the UI can render without knowing about tensors.

Everything reported here is read back off the real arrays the pipeline produced:

    original_tokens    PageEncoding.n_tokens          (what the model emitted)
    final_tokens       CompressedPage.n_vectors       (rows of the code matrix)
    original_bytes     CompressedPage.raw_nbytes()    tokens x dim x 4
    compressed_bytes   CompressedPage.nbytes          codes.nbytes

so a number on screen cannot drift from what was actually stored.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .compression import Compressor
from .config import Config
from .encoders import BaseEncoder, get_encoder
from .ingest import downscale
from .pruning import TokenPruner
from .types import PageRef
from .viz import overlay_mask, overlay_saliency

KIB = 1024.0

# A ColPali-family page is a few hundred KB of vectors; nothing here needs the
# full-resolution scan, and a 20 MP upload would only slow the preview down.
DEMO_MAX_SIDE = 1536


class DemoError(Exception):
    """A problem worth showing the user verbatim (bad file, wrong page, ...)."""


# --------------------------------------------------------------------- input


def pdf_page_count(path: str | Path) -> int:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        return len(doc)
    finally:
        doc.close()


def load_page(path: str | Path, page_number: int = 1, dpi: int = 150) -> tuple[Image.Image, int]:
    """Render one page of a PDF, or open an image file.

    Returns ``(image, total_pages)``. Raises :class:`DemoError` with a message
    meant for the screen — never a traceback.
    """
    path = Path(path)
    if not path.exists():
        raise DemoError("The uploaded file could not be read. Please try again.")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
        try:
            total = len(doc)
            if total == 0:
                raise DemoError("That PDF has no pages.")
            if not 1 <= page_number <= total:
                raise DemoError(
                    f"Page {page_number} does not exist — this PDF has {total} page(s)."
                )
            bitmap = doc[page_number - 1].render(scale=dpi / 72.0)
            image = bitmap.to_pil().convert("RGB")
        finally:
            doc.close()
        return downscale(image, DEMO_MAX_SIDE), total

    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}:
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
        except Exception as exc:
            raise DemoError("That image file could not be opened.") from exc
        return downscale(image, DEMO_MAX_SIDE), 1

    raise DemoError(
        f"Unsupported file type '{suffix or 'unknown'}'. Upload a PDF, PNG, JPG or TIFF."
    )


def looks_blank(image: Image.Image, threshold: float = 0.004) -> bool:
    """True when a page carries almost no ink.

    Not a failure — a blank page compresses spectacularly and says nothing. The
    UI warns rather than pretending the result is meaningful.
    """
    from .pruning import patch_saliency

    return float(patch_saliency(image, 16, 16).mean()) < threshold


# ------------------------------------------------------------------- device


@contextlib.contextmanager
def on_device(encoder: BaseEncoder, device: str):
    """Temporarily run a torch-backed encoder on another device.

    ZeroGPU only exposes CUDA *inside* a ``@spaces.GPU`` function, so the model
    is built on CPU at startup and moved for the duration of one encode. Encoders
    without a torch model (the synthetic one) pass straight through.
    """
    model = getattr(encoder, "model", None)
    if model is None or device == getattr(encoder, "device", None):
        yield encoder
        return

    previous = encoder.device
    try:
        encoder.model = model.to(device)
        encoder.device = device
        yield encoder
    finally:
        with contextlib.suppress(Exception):
            encoder.model = encoder.model.to(previous)
        encoder.device = previous


def pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------- result


@dataclass
class CompressionResult:
    """One page's journey through the pipeline, in numbers the UI can print."""

    # token counts
    original_tokens: int  # everything the model emitted for this page
    grid_patches: int  # of those, the ones laid out on the page grid
    non_grid_tokens: int  # thumbnail + instruction tokens, always kept
    spatial_tokens: int  # patches surviving blank-region pruning
    redundancy_tokens: int  # patches surviving duplicate collapsing
    final_tokens: int  # vectors actually stored (= rows of the code matrix)

    # representation
    embedding_dim: int
    original_bytes: int
    compressed_bytes: int
    original_dtype: str
    compressed_dtype: str

    # derived
    token_reduction_percent: float
    storage_reduction_factor: float
    bits_per_dimension: float

    # timing
    encoding_seconds: float
    pruning_ms: float
    quantization_ms: float

    # layout / provenance
    grid_rows: int
    grid_cols: int
    model_name: str
    device: str
    image_size: tuple[int, int]

    # images
    original_image: Image.Image | None = None
    retained_regions_image: Image.Image | None = None
    saliency_image: Image.Image | None = None

    blank_page: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    # The index-ready page itself, kept so the optional Qdrant step can store
    # exactly these bytes instead of recomputing them.
    compressed_page: Any = None

    @property
    def original_kib(self) -> float:
        return self.original_bytes / KIB

    @property
    def compressed_kib(self) -> float:
        return self.compressed_bytes / KIB

    def as_dict(self) -> dict[str, Any]:
        """Scalar view of the result — images and array payloads excluded."""
        skip = {"stats", "compressed_page", "original_image", "retained_regions_image", "saliency_image"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}


# ------------------------------------------------------------------ pipeline


class DemoPipeline:
    """Loads the encoder once and compresses one page at a time.

    Deliberately *not* a new pipeline: it holds the same ``TokenPruner`` and
    ``Compressor`` the benchmark uses, configured from the same YAML.
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._encoder: BaseEncoder | None = None
        self.pruner = TokenPruner(self.cfg.pruning)
        self.compressor = Compressor(self.cfg.compression)

    # The encoder is the expensive part (a checkpoint load), so it is built once
    # per process and reused across button clicks.
    @property
    def encoder(self) -> BaseEncoder:
        if self._encoder is None:
            self._encoder = get_encoder(self.cfg.encoder)
        return self._encoder

    @property
    def model_name(self) -> str:
        return str(getattr(self.encoder, "checkpoint", self.cfg.encoder.backend))

    def warm_up(self) -> None:
        """Force the checkpoint load now rather than on the first click."""
        _ = self.encoder

    def compress(
        self,
        image: Image.Image,
        page_id: str = "demo",
        device: str | None = None,
        progress=None,
    ) -> CompressionResult:
        """Run encode -> spatial prune -> redundancy prune -> quantize on one page."""
        ref = PageRef(doc_id=page_id, page_no=1)
        device = device or pick_device()

        def say(message: str) -> None:
            if progress is not None:
                progress(message)

        say("Encoding page with the vision-language model...")
        t0 = time.perf_counter()
        with on_device(self.encoder, device) as encoder:
            encoding = encoder.encode_pages([image], [ref])[0]
        encoding_seconds = time.perf_counter() - t0

        if encoding.n_tokens == 0:
            raise DemoError("The model returned no tokens for this page.")

        say(f"Generated {encoding.n_tokens} visual tokens. Applying spatial pruning...")
        t1 = time.perf_counter()
        pruned = self.pruner.prune(encoding, image)
        pruning_ms = (time.perf_counter() - t1) * 1000.0

        say("Applying binary quantization...")
        t2 = time.perf_counter()
        compressed = self.compressor.compress(pruned)
        quantization_ms = (time.perf_counter() - t2) * 1000.0

        # --- numbers read back off the real arrays -------------------------
        original_tokens = int(compressed.n_tokens_before)
        final_tokens = int(compressed.n_vectors)
        dim = int(compressed.dim)
        original_bytes = int(compressed.raw_nbytes())  # tokens x dim x 4
        compressed_bytes = int(compressed.nbytes)  # codes.nbytes

        stats = dict(pruned.stats)
        grid_patches = int(stats.get("n_patches", pruned.grid.n_patches))
        spatial_tokens = int(stats.get("n_after_spatial", grid_patches))
        redundancy_tokens = int(stats.get("n_after_redundancy", spatial_tokens))
        non_grid = int(encoding.text_token_index.size)

        method = self.cfg.compression.method if self.cfg.compression.enabled else "none"
        compressed_dtype = {"binary": "binary / 1-bit", "int8": "int8", "none": "float32"}.get(
            method, method
        )

        say("Rendering retained regions...")
        retained = overlay_mask(image, pruned.keep_mask)
        saliency_img = (
            overlay_saliency(image, pruned.saliency) if pruned.saliency is not None else None
        )

        say("Compression complete.")
        return CompressionResult(
            original_tokens=original_tokens,
            grid_patches=grid_patches,
            non_grid_tokens=non_grid,
            spatial_tokens=spatial_tokens,
            redundancy_tokens=redundancy_tokens,
            final_tokens=final_tokens,
            embedding_dim=dim,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            original_dtype="float32",
            compressed_dtype=compressed_dtype,
            token_reduction_percent=(original_tokens - final_tokens) / max(1, original_tokens) * 100.0,
            storage_reduction_factor=original_bytes / max(1, compressed_bytes),
            bits_per_dimension=compressed_bytes * 8.0 / max(1, final_tokens * dim),
            encoding_seconds=encoding_seconds,
            pruning_ms=pruning_ms,
            quantization_ms=quantization_ms,
            grid_rows=int(pruned.grid.rows),
            grid_cols=int(pruned.grid.cols),
            model_name=self.model_name,
            device=device,
            image_size=tuple(image.size),
            original_image=image,
            retained_regions_image=retained,
            saliency_image=saliency_img,
            blank_page=looks_blank(image),
            stats=stats,
            compressed_page=compressed,
        )


# ------------------------------------------------------------ optional store


def store_in_qdrant(
    page,
    collection: str = "optivision_demo",
    path: str = "data/index/demo_qdrant",
) -> dict[str, Any]:
    """Best-effort write of one compressed page into a local Qdrant collection.

    Optional by design: the compression demo must not depend on it, so every
    failure is returned as ``{"ok": False, "error": ...}`` rather than raised.
    """
    try:
        from .index.qdrant_index import QdrantIndex

        index = QdrantIndex(
            path=path,
            collection=collection,
            dim=page.dim,
            method="binary",
            recreate=True,
        )
        index.add([page])
        info = {
            "ok": True,
            "collection": collection,
            "n_points": index.n_pages,
            "n_vectors": int(page.n_vectors),
            "comparator": "MAX_SIM (multivector)",
        }
        index.close()
        return info
    except Exception as exc:  # noqa: BLE001 - optional feature, never fatal
        return {"ok": False, "error": str(exc)}
