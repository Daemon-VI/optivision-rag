"""Real late-interaction VLM backends (ColPali / ColQwen2 / ColSmol).

All of them come from ``colpali-engine`` and share one contract: a page image in,
one 128-d vector per visual patch out, plus a short tail of instruction tokens.
Retrieval is MaxSim late interaction over those vectors.

Backend choice is a hardware decision:

    colsmol   256M params, ~0.5 GB   runs on CPU / 8 GB laptops   (default)
    colqwen2    2B params, ~4 GB     needs a GPU for sane latency
    colqwen2.5  3B params, ~6 GB     needs a GPU; adapter-only checkpoint (merged on load)
    colpali     3B params, ~6 GB     reference model from the paper

The pruning and quantization stages never touch the model, so results transfer
between backends; only absolute quality moves.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from ..types import PageEncoding, PageRef, PatchGrid
from .base import BaseEncoder, l2_normalise

# backend -> (default checkpoint, model class name, processor class name)
#
# Several of these checkpoints are published *adapter-only*: `vidore/colpali-v1.3`,
# `vidore/colSmol-256M`, `vidore/colqwen2-v1.0` and `vidore/colqwen2.5-v0.2` hold
# `adapter_model.safetensors` and a pointer to a base model. (For ColSmol-256M the
# injected and the explicitly merged adapter give identical vectors, checked
# 2026-10-01 with transformers 5.15 / peft 0.19.) Loading such a repo through `from_pretrained` makes
# transformers inject the LoRA adapter itself, which is brittle across
# transformers/peft releases — when the checkpoint's key prefixes do not match what
# the installed build expects, the LoRA weights are silently left uninitialised
# rather than loaded. Where a pre-merged checkpoint exists we use it (same model, no
# injection step). Where none exists (ColQwen2.5), `_load_model` loads the base,
# applies the adapter with peft explicitly, merges it, and refuses to continue
# unless the merge actually changed the weights it targets.
BACKENDS: dict[str, tuple[str, str, str]] = {
    "colsmol": ("vidore/colSmol-256M", "ColIdefics3", "ColIdefics3Processor"),
    "colsmol-500m": ("vidore/colSmol-500M", "ColIdefics3", "ColIdefics3Processor"),
    "colpali": ("vidore/colpali-v1.3-merged", "ColPali", "ColPaliProcessor"),
    "colqwen2": ("vidore/colqwen2-v1.0-merged", "ColQwen2", "ColQwen2Processor"),
    "colqwen2.5": ("vidore/colqwen2.5-v0.2", "ColQwen2_5", "ColQwen2_5_Processor"),
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
    if device == "cpu":
        return torch.float32
    # GPUs without native bfloat16 (T4, P100 -- the free Kaggle/Colab cards) only
    # emulate it, slowly. float16 is not a safe substitute: Qwen2-VL activations can
    # overflow it. float32 fits a 2-3B model on a 16 GB card.
    if device == "cuda":
        try:
            native = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # older torch has no including_emulation argument
            native = torch.cuda.get_device_capability()[0] >= 8
        return torch.bfloat16 if native else torch.float32
    return torch.bfloat16


def _adapter_base(checkpoint: str) -> str | None:
    """The base model of an adapter-only checkpoint, or None for full weights."""
    import json
    import os

    if os.path.isdir(checkpoint):
        cfg = os.path.join(checkpoint, "adapter_config.json")
        if not os.path.isfile(cfg) or os.path.isfile(os.path.join(checkpoint, "config.json")):
            return None
        with open(cfg, encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path")
    from huggingface_hub import file_exists, hf_hub_download, try_to_load_from_cache

    try:
        has_adapter = file_exists(checkpoint, "adapter_config.json")
        has_config = file_exists(checkpoint, "config.json")
    except Exception:  # noqa: BLE001 -- offline or Hub unreachable (error types vary by hub version): use the cache
        has_adapter = isinstance(try_to_load_from_cache(checkpoint, "adapter_config.json"), str)
        has_config = isinstance(try_to_load_from_cache(checkpoint, "config.json"), str)
    if not has_adapter or has_config:
        return None
    with open(hf_hub_download(checkpoint, "adapter_config.json"), encoding="utf-8") as f:
        return json.load(f).get("base_model_name_or_path")


# colpali-engine 0.3.17's ColQwen2 / ColQwen2.5 rename `model.layers.*` to the
# transformers-5 name `language_model.layers.*` but not `model.embed_tokens` and
# `model.norm`. Those two then load as *newly initialised random* weights, with
# only a log line to say so: the text embedding table and the final norm of the
# language model. Measured on a Kaggle T4 (transformers 5.18): queries retrieved
# their own page at chance level and two loads of one checkpoint disagreed.
_EXTRA_KEY_MAPPING: dict[str, dict[str, str]] = {
    name: {r"^model\.embed_tokens\.": "language_model.embed_tokens.", r"^model\.norm\.": "language_model.norm."}
    for name in ("ColQwen2", "ColQwen2_5")
}

# Missing weights that are legitimately absent from a checkpoint: an output head
# tied to the input embeddings (re-tied after loading, never used for retrieval).
_ALLOWED_MISSING = re.compile(r"(^|\.)lm_head\.weight$")


def _key_mapping(model_cls) -> dict[str, str] | None:
    extra = _EXTRA_KEY_MAPPING.get(model_cls.__name__)
    if not extra:
        return None
    mapping: dict[str, str] = {}
    for klass in reversed(model_cls.__mro__):
        mapping.update(getattr(klass, "_checkpoint_conversion_mapping", None) or {})
    mapping.update(extra)
    return mapping


def _from_pretrained(model_cls, checkpoint: str, dtype):
    """Load and refuse any weight that the checkpoint did not provide.

    transformers fills a weight it finds no tensor for with random values and only
    logs it. For a retrieval encoder that yields complete, plausible, meaningless
    vectors, so a missing weight is an error here.
    """
    kwargs: dict[str, Any] = {"output_loading_info": True}
    mapping = _key_mapping(model_cls)
    if mapping is not None:
        kwargs["key_mapping"] = mapping
    # transformers>=5 renamed `torch_dtype` to `dtype`; support both.
    try:
        model, info = model_cls.from_pretrained(checkpoint, dtype=dtype, **kwargs)
    except TypeError:
        model, info = model_cls.from_pretrained(checkpoint, torch_dtype=dtype, **kwargs)
    missing = [k for k in info.get("missing_keys", []) if not _ALLOWED_MISSING.search(k)]
    unexpected = list(info.get("unexpected_keys", []))
    if missing:
        raise RuntimeError(
            f"{checkpoint}: {len(missing)} weights were not in the checkpoint and would be random "
            f"(e.g. {missing[:3]}); checkpoint tensors that matched nothing: {unexpected[:3]}. "
            "The checkpoint's key names do not map onto this model under the installed "
            "transformers / colpali-engine versions."
        )
    model.loading_report = {"missing_allowed": list(info.get("missing_keys", [])), "unexpected": unexpected}
    return model


def _load_model(model_cls, checkpoint: str, dtype):
    """Full checkpoints load directly; adapter-only ones are merged explicitly."""
    base_id = _adapter_base(checkpoint)
    if base_id is None:
        return _from_pretrained(model_cls, checkpoint, dtype)
    from peft import PeftModel

    base = _from_pretrained(model_cls, base_id, dtype)
    peft_model = PeftModel.from_pretrained(base, checkpoint)
    # LoRA initialises every B matrix to zero, so a B still at zero was never loaded.
    lora_b = {n: prm for n, prm in peft_model.named_parameters() if ".lora_B." in n}
    if not lora_b:
        raise RuntimeError(f"{checkpoint}: the adapter matched no module of {base_id}")
    unloaded = [n for n, prm in lora_b.items() if float(prm.detach().abs().max()) == 0.0]
    if unloaded:
        raise RuntimeError(
            f"{checkpoint}: {len(unloaded)} of {len(lora_b)} LoRA B matrices are still zero after loading "
            f"(e.g. {unloaded[0]}), so those adapter weights were not loaded."
        )
    # One module the adapter really targets: its base weight must change on merge.
    module = next(iter(lora_b)).split(".lora_B.")[0]
    inner = module.removeprefix("base_model.model.")
    before = peft_model.get_parameter(f"{module}.base_layer.weight").detach().float().clone()
    model = peft_model.merge_and_unload()
    changed = float((model.get_parameter(f"{inner}.weight").detach().float() - before).abs().max())
    if changed == 0.0:
        raise RuntimeError(f"{checkpoint}: merging the adapter left {inner}.weight unchanged")
    model.adapter_merge_check = {"base": base_id, "lora_modules": len(lora_b), "probe": f"{inner}.weight",
                                 "max_abs_change": changed}
    return model


def _retie_output_embeddings(model) -> None:
    """Re-tie the language-model head to the input embeddings after loading.

    PaliGemma ties `lm_head.weight` to the token embeddings, so a merged ColPali
    checkpoint does not store it — the weight is meant to be re-created by tying
    at load time. Nested one level down inside ColPali (`model.model.lm_head`),
    transformers does not always do that, which leaves exactly one parameter on
    the meta device and makes `.to(device)` fail with "Cannot copy out of meta
    tensor".

    Tying is safe here in a stronger sense than usual: ColPali reads
    `hidden_states[-1]`, which is computed *before* the head, and projects that
    through `custom_text_proj`. The head's values never reach a retrieval vector.
    It only has to exist on the right device, because the wrapped
    `...ForConditionalGeneration.forward` still computes logits and discards them.
    Tying also costs no extra memory, since the tensor is shared.
    """
    inner = getattr(model, "model", None)
    for target in (inner, model):
        if target is None:
            continue
        tie = getattr(target, "tie_weights", None)
        if callable(tie):
            tie()

    # Some builds only tie via config flags. If the head is still unmaterialised,
    # bind it to the embedding tensor directly — that is what tying does.
    for target in (inner, model):
        if target is None:
            continue
        head = getattr(target, "lm_head", None)
        get_input = getattr(target, "get_input_embeddings", None)
        if head is None or not callable(get_input):
            continue
        if getattr(head, "weight", None) is not None and head.weight.device.type == "meta":
            embeddings = get_input()
            if embeddings is not None and embeddings.weight.device.type != "meta":
                head.weight = embeddings.weight


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

        model = _load_model(model_cls, checkpoint, self.torch_dtype)
        self.adapter_merge_check = getattr(model, "adapter_merge_check", None)
        self.loading_report = getattr(model, "loading_report", None)
        _retie_output_embeddings(model)

        # A parameter still on the meta device never received weights. That happens
        # when a checkpoint's keys do not match the installed transformers build:
        # the layers are created, the load silently skips them, and the model would
        # encode with randomly initialised projections — producing a complete,
        # plausible, meaningless benchmark table. `.to()` would raise "Cannot copy
        # out of meta tensor" here anyway; say why.
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
