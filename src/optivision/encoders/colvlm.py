"""Real late-interaction VLM backends (ColPali / ColQwen2 / ColSmol).

All of them come from ``colpali-engine`` and share one contract: a page image in,
one 128-d vector per visual patch out, plus a short tail of instruction tokens.
Retrieval is MaxSim late interaction over those vectors.

Backend choice is a hardware decision:

    colsmol   256M params, ~0.5 GB   runs on CPU / 8 GB laptops   (default)
    colqwen2    2B params, ~4 GB     needs a GPU for sane latency
    colpali     3B params, ~6 GB     reference model from the paper

The pruning and quantization stages never touch the model, so results transfer
between backends; only absolute quality moves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from ..types import PageEncoding, PageRef, PatchGrid
from .base import BaseEncoder, l2_normalise

# backend -> (default checkpoint, model class name, processor class name)
#
# ColPali is the one checkpoint here published *adapter-only*: `vidore/colpali-v1.3`
# holds `adapter_model.safetensors` and no `config.json`, so loading it makes
# transformers inject a LoRA adapter over the PaliGemma base. The ColSmol and
# ColQwen2 repos ship full weights alongside their adapter files, which is why only
# this backend ever hit the problem. Adapter injection is brittle across
# transformers/peft releases — when the checkpoint's key prefixes do not match what
# the installed build expects, the LoRA weights are silently left uninitialised
# rather than loaded. We use the pre-merged weights instead: same model, no
# injection step, nothing to get wrong.
BACKENDS: dict[str, tuple[str, str, str]] = {
    "colsmol": ("vidore/colSmol-256M", "ColIdefics3", "ColIdefics3Processor"),
    "colsmol-500m": ("vidore/colSmol-500M", "ColIdefics3", "ColIdefics3Processor"),
    "colpali": ("vidore/colpali-v1.3-merged", "ColPali", "ColPaliProcessor"),
    "colqwen2": ("vidore/colqwen2-v1.0", "ColQwen2", "ColQwen2Processor"),
}


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str, device: str):
    import torch

    if dtype != "auto":
        return getattr(torch, dtype)
    # bfloat16 on CPU is slower than float32 for these sizes and can be unstable
    # on older CPUs; float16 has no fast CPU kernels at all.
    return torch.float32 if device == "cpu" else torch.bfloat16


class ColVLMEncoder(BaseEncoder):
    def __init__(
        self,
        backend: str = "colsmol",
        model_name: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        query_batch_size: int = 16,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
        import torch
        from colpali_engine import models as cpm

        default_ckpt, model_cls_name, proc_cls_name = BACKENDS[backend]
        checkpoint = model_name or default_ckpt
        model_cls = getattr(cpm, model_cls_name)
        proc_cls = getattr(cpm, proc_cls_name)

        self.name = backend
        self.checkpoint = checkpoint
        self.query_batch_size = max(1, int(query_batch_size))
        self.device = _resolve_device(device)
        self.torch_dtype = _resolve_dtype(dtype, self.device)
        self._torch = torch

        # transformers>=5 renamed `torch_dtype` to `dtype`; support both.
        try:
            model = model_cls.from_pretrained(checkpoint, dtype=self.torch_dtype)
        except TypeError:
            model = model_cls.from_pretrained(checkpoint, torch_dtype=self.torch_dtype)
        # A parameter still on the meta device never received weights. That happens
        # when an adapter checkpoint's keys do not match the installed
        # transformers/peft build: the layers are created, the load silently skips
        # them, and the model would encode with randomly initialised projections —
        # producing a complete, plausible, meaningless benchmark table. `.to()`
        # would raise "Cannot copy out of meta tensor" here anyway; say why.
        unloaded = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if unloaded:
            raise RuntimeError(
                f"{checkpoint}: {len(unloaded)} parameters were never loaded "
                f"(e.g. {unloaded[0]}). The checkpoint's weights did not map onto this "
                "model, so encoding would return meaningless vectors. If this is an "
                "adapter-only repo, point --model at the merged checkpoint instead."
            )
        self.model = model.to(self.device).eval()
        self.processor = proc_cls.from_pretrained(checkpoint)
        self._dim: int | None = None
        self._image_token_id = self._find_image_token_id()

    # ------------------------------------------------------------- internals

    def _find_image_token_id(self) -> int | None:
        proc: Any = self.processor
        for attr in ("image_token_id", "image_seq_token_id"):
            tid = getattr(proc, attr, None)
            if isinstance(tid, int):
                return tid
        tokenizer = getattr(proc, "tokenizer", None)
        if tokenizer is not None:
            for token in ("<image>", "<|image_pad|>", "<image_soft_token>"):
                tid = tokenizer.convert_tokens_to_ids(token)
                if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
                    return tid
        return None

    def _image_positions(self, input_ids: np.ndarray, n_tokens: int) -> np.ndarray:
        """Positions in the embedding matrix that correspond to image patches."""
        if self._image_token_id is not None:
            pos = np.flatnonzero(input_ids == self._image_token_id)
            if pos.size:
                return pos.astype(np.int32)
        # Fallback: PaliGemma-style layouts put every image token first.
        n_text = max(0, n_tokens - self._largest_square_below(n_tokens))
        return np.arange(0, n_tokens - n_text, dtype=np.int32)

    @staticmethod
    def _largest_square_below(n: int) -> int:
        s = math.isqrt(n)
        return s * s

    @staticmethod
    def _runs(positions: np.ndarray) -> list[np.ndarray]:
        """Split sorted positions into maximal runs of consecutive indices."""
        if positions.size == 0:
            return []
        breaks = np.flatnonzero(np.diff(positions) != 1) + 1
        return np.split(positions, breaks)

    def _grid_for(
        self, positions: np.ndarray, image_size: tuple[int, int]
    ) -> tuple[PatchGrid, np.ndarray]:
        """Map image-token positions onto a rectangular page grid.

        Returns ``(grid, extra)`` where ``extra`` holds image tokens that are not
        part of the page grid and must be kept verbatim.

        Two layouts occur in practice:

        *Single image* (ColPali/PaliGemma, ColSmol with splitting disabled) — one
        contiguous run of ``rows*cols`` tokens in row-major order.

        *Tiled* (Idefics3/ColSmol default) — the page is cut into a ``cr x cc``
        grid of tiles, each encoded to ``s x s`` patches, emitted as equal-length
        runs in row-major tile order, followed by one final run for a
        thumbnail of the whole page. Flattening that into one rectangle would
        map saliency to the wrong patches, so the tiles are stitched into a
        ``cr*s x cc*s`` grid and the thumbnail run is set aside.
        """
        runs = self._runs(positions)
        n = int(positions.size)
        no_extra = np.zeros(0, dtype=np.int32)

        if len(runs) > 2:
            lengths = {len(r) for r in runs}
            # Tiled layouts emit equal-length runs; the trailing one is the
            # whole-page thumbnail, so tile count is len(runs) - 1.
            if len(lengths) == 1:
                per_tile = len(runs[0])
                s = math.isqrt(per_tile)
                n_tiles = len(runs) - 1
                if s * s == per_tile and n_tiles >= 1:
                    cr, cc = self._tile_shape(n_tiles, image_size)
                    if cr * cc == n_tiles:
                        token_index = np.zeros((cr * s, cc * s), dtype=np.int32)
                        for t in range(n_tiles):
                            tile = runs[t].reshape(s, s)
                            r0, c0 = (t // cc) * s, (t % cc) * s
                            token_index[r0 : r0 + s, c0 : c0 + s] = tile
                        return (
                            PatchGrid(rows=cr * s, cols=cc * s, token_index=token_index),
                            runs[-1].astype(np.int32),
                        )

        rows, cols = self._grid_shape(n, image_size)
        if rows * cols != n:  # unrecognised layout: degrade to a 1 x n strip
            rows, cols = 1, n
        return PatchGrid(rows=rows, cols=cols, token_index=positions.reshape(rows, cols)), no_extra

    @staticmethod
    def _tile_shape(n_tiles: int, image_size: tuple[int, int]) -> tuple[int, int]:
        """Factorise the tile count to match the page aspect ratio.

        The splitter tiles a resized page into fixed-size squares, so the tile
        grid is ceil(H/t) x ceil(W/t) — the factorisation closest to the page's
        own aspect ratio.
        """
        width, height = image_size
        target = height / max(width, 1)
        best, best_err = (1, n_tiles), float("inf")
        for r in range(1, n_tiles + 1):
            if n_tiles % r:
                continue
            c = n_tiles // r
            err = abs((r / c) - target)
            if err < best_err:
                best, best_err = (r, c), err
        return best

    def _grid_shape(self, n: int, image_size: tuple[int, int]) -> tuple[int, int]:
        # Ask the processor first — it knows the patch size and resize policy.
        getter = getattr(self.processor, "get_n_patches", None)
        if callable(getter):
            for kwargs in (
                {"image_size": image_size, "patch_size": getattr(self.model, "patch_size", 14)},
                {"image_size": image_size},
            ):
                try:
                    rows, cols = getter(**kwargs)
                    if int(rows) * int(cols) == n:
                        return int(rows), int(cols)
                except (TypeError, ValueError, AttributeError):
                    continue
        s = math.isqrt(n)
        if s * s == n:
            return s, s
        # Non-square token count: pick the factorisation closest to the page aspect.
        width, height = image_size
        target = (height / max(width, 1)) if width else 1.0
        best = (1, n)
        best_err = float("inf")
        for r in range(1, n + 1):
            if n % r:
                continue
            c = n // r
            err = abs((r / c) - target)
            if err < best_err:
                best, best_err = (r, c), err
        return best

    # ------------------------------------------------------------------ pages

    def encode_pages(
        self, images: Sequence[Image.Image], refs: Sequence[PageRef]
    ) -> list[PageEncoding]:
        torch = self._torch
        images = [im.convert("RGB") for im in images]
        batch = self.processor.process_images(list(images))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            emb = self.model(**batch)
        emb_np = emb.to(torch.float32).cpu().numpy()
        input_ids = batch["input_ids"].cpu().numpy()
        attn = batch.get("attention_mask")
        attn_np = attn.cpu().numpy() if attn is not None else np.ones_like(input_ids)

        out: list[PageEncoding] = []
        for i, ref in enumerate(refs):
            keep = np.flatnonzero(attn_np[i] == 1).astype(np.int32)
            vectors = l2_normalise(emb_np[i][keep])
            ids = input_ids[i][keep]
            positions = self._image_positions(ids, vectors.shape[0])
            grid, extra = self._grid_for(positions, images[i].size)
            # Everything outside the page grid — instruction tokens plus any
            # whole-page thumbnail run — is kept verbatim: it is a handful of
            # vectors and it summarises the page the pruner is thinning out.
            gridded = grid.token_index.reshape(-1)
            text_idx = np.setdiff1d(
                np.arange(vectors.shape[0], dtype=np.int32), gridded, assume_unique=False
            ).astype(np.int32)
            self._dim = int(vectors.shape[1])
            out.append(
                PageEncoding(
                    ref=ref,
                    embeddings=vectors,
                    grid=grid,
                    image_size=images[i].size,
                    text_token_index=text_idx,
                    meta={
                        "encoder": self.name,
                        "checkpoint": self.checkpoint,
                        "n_image_tokens": int(positions.size),
                        "n_thumbnail_tokens": int(extra.size),
                        "grid": f"{grid.rows}x{grid.cols}",
                    },
                )
            )
        return out

    # ---------------------------------------------------------------- queries

    def encode_queries(self, queries: Sequence[str]) -> list[np.ndarray]:
        """Encode queries in bounded chunks.

        A benchmark hands this the entire query set at once. Pushing all of them
        through the model in a single forward pass makes peak memory scale with
        the number of queries, which is fine for 10 and fatal for 5000, so the
        work is chunked regardless of how many arrive.
        """
        out: list[np.ndarray] = []
        for start in range(0, len(queries), self.query_batch_size):
            out.extend(self._encode_query_batch(list(queries[start : start + self.query_batch_size])))
        return out

    def _encode_query_batch(self, queries: list[str]) -> list[np.ndarray]:
        torch = self._torch
        batch = self.processor.process_queries(queries)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            emb = self.model(**batch)
        emb_np = emb.to(torch.float32).cpu().numpy()
        attn = batch.get("attention_mask")
        attn_np = attn.cpu().numpy() if attn is not None else np.ones(emb_np.shape[:2], dtype=int)
        out = []
        for i in range(emb_np.shape[0]):
            keep = np.flatnonzero(attn_np[i] == 1)
            vectors = l2_normalise(emb_np[i][keep])
            self._dim = int(vectors.shape[1])
            out.append(vectors)
        return out

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Every published Col* checkpoint projects to 128 dims.
            self._dim = int(getattr(self.model, "dim", 128))
        return self._dim
