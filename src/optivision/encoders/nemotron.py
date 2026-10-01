"""NVIDIA Nemotron ColEmbed VL v2: wide (2,560 / 4,096-d) late-interaction encoders.

``nvidia/nemotron-colembed-vl-4b-v2`` and ``-8b-v2`` are Qwen3-VL models that
return, for every input token, the last decoder layer's hidden state (pre-norm),
masked and L2-normalised, with no projection head. A ViDoRe page becomes up to
8 tiles + 1 thumbnail x 256 tokens. Licence: CC-BY-NC-4.0 (research use).

This wrapper reproduces the model's own ``_extract_embeddings`` (forward, mask,
normalise, keep the non-padding tokens) while keeping control of devices and
precision, and it refuses to run unless:

* the checkpoint is the expected architecture (``model_type``, ``pooling``),
* the output width equals the registry's (``expected_dim``),
* every weight came from the checkpoint (see ``colvlm._from_pretrained``),
* the remote modelling code is a pinned revision (it executes on load).

Differences from the authors' reference pipeline, both deliberate: float32
instead of bfloat16 autocast (T4 GPUs have no native bfloat16, and ColQwen was
measured in float32), and ``sdpa`` attention instead of flash-attention 2
(unavailable on T4).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .colvlm import _from_pretrained, _resolve_device, _resolve_dtype, place_model

#: model id -> (output width, pinned revision of the checkpoint and its remote code)
KNOWN: dict[str, tuple[int, str]] = {
    "nvidia/nemotron-colembed-vl-4b-v2": (2560, "0ed152d91f8ad4c5d48296b51c220f686641a398"),
    "nvidia/nemotron-colembed-vl-8b-v2": (4096, "34b640612f311ed05a6c7c62c6564847ed555f5f"),
}
EXPECTED_MODEL_TYPE = "qwen3_vl_nemotron_embed"


class NemotronColEmbedEncoder:
    """Pages and queries in, one list of ``[n_tokens, dim]`` float32 arrays out."""

    def __init__(self, model_name: str = "nvidia/nemotron-colembed-vl-4b-v2", revision: str | None = None,
                 device: str = "auto", dtype: str = "auto", multi_gpu: bool = False,
                 expected_dim: int | None = None, attn_implementation: str = "sdpa",
                 page_batch_size: int = 1, query_batch_size: int = 8) -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor

        known_dim, known_rev = KNOWN.get(model_name, (None, None))
        self.checkpoint = model_name
        self.revision = revision or known_rev
        if self.revision is None:
            raise ValueError(f"{model_name}: pin a revision (its remote code runs on load)")
        self.expected_dim = expected_dim or known_dim

        config = AutoConfig.from_pretrained(model_name, revision=self.revision, trust_remote_code=True)
        model_type = getattr(config, "model_type", None)
        pooling = getattr(config, "pooling", None)
        width = getattr(getattr(config, "text_config", config), "hidden_size", None)
        if model_type != EXPECTED_MODEL_TYPE or pooling != "colbert":
            raise RuntimeError(f"{model_name}: expected a {EXPECTED_MODEL_TYPE} ColBERT-pooling model, "
                               f"got model_type={model_type!r}, pooling={pooling!r}")
        if self.expected_dim is not None and width != self.expected_dim:
            raise RuntimeError(f"{model_name}: hidden size {width} does not match the expected {self.expected_dim}")
        self.dim = int(width)

        self._torch = torch
        self.device = _resolve_device(device)
        self.torch_dtype = _resolve_dtype(dtype, self.device)
        model = _from_pretrained(AutoModel, model_name, self.torch_dtype, revision=self.revision,
                                 trust_remote_code=True, attn_implementation=attn_implementation)
        self.loading_report = getattr(model, "loading_report", None)
        self.adapter_merge_check = None
        unloaded = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if unloaded:
            raise RuntimeError(f"{model_name}: {len(unloaded)} parameters never loaded (e.g. {unloaded[0]})")
        model, self.device, self.placement = place_model(model, self.device, multi_gpu)
        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, revision=self.revision, trust_remote_code=True)
        self.page_batch_size = max(1, int(page_batch_size))
        self.query_batch_size = max(1, int(query_batch_size))
        self.attn_implementation = attn_implementation

    # ------------------------------------------------------------------ core

    def _embed(self, batch: dict[str, Any]) -> list[np.ndarray]:
        torch = self._torch
        import torch.nn.functional as F

        batch = {k: v for k, v in batch.items() if v is not None}
        mask = batch["attention_mask"].detach().cpu()
        batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        # the model's own recipe -- mask, then L2-normalise -- done on the CPU so a
        # model spread over several GPUs never mixes devices
        emb = hidden.detach().to("cpu", torch.float32) * mask.unsqueeze(-1).to(torch.float32)
        emb = F.normalize(emb, dim=-1)
        if not torch.isfinite(emb).all():
            raise ValueError(f"{self.checkpoint}: non-finite embeddings")
        if emb.shape[-1] != self.dim:
            raise RuntimeError(f"{self.checkpoint}: got {emb.shape[-1]}-d vectors, expected {self.dim}")
        keep = mask.bool()
        return [np.ascontiguousarray(emb[i][keep[i]].numpy()) for i in range(emb.shape[0])]

    def encode_images(self, images: Sequence[Any]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for start in range(0, len(images), self.page_batch_size):
            chunk = [{"image": im.convert("RGB"), "text": ""} for im in images[start : start + self.page_batch_size]]
            out.extend(self._embed(self.processor.process_documents(chunk)))
        return out

    def encode_queries(self, queries: Sequence[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for start in range(0, len(queries), self.query_batch_size):
            out.extend(self._embed(self.processor.process_queries(list(queries[start : start + self.query_batch_size]))))
        return out

    def info(self) -> dict[str, Any]:
        return {"model": self.checkpoint, "revision": self.revision, "dim": self.dim, "device": self.device,
                "dtype": str(self.torch_dtype), "attn_implementation": self.attn_implementation,
                "loading_report": self.loading_report, "placement": self.placement}

    def timed_encode_images(self, images: Sequence[Any]) -> tuple[list[np.ndarray], float]:
        t = time.perf_counter()
        out = self.encode_images(images)
        return out, time.perf_counter() - t
