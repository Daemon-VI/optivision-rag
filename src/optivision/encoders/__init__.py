"""Encoder registry."""

from __future__ import annotations

from ..config import EncoderConfig
from .base import BaseEncoder, l2_normalise
from .synthetic import SyntheticEncoder

__all__ = ["BaseEncoder", "SyntheticEncoder", "get_encoder", "l2_normalise"]


def get_encoder(cfg: EncoderConfig) -> BaseEncoder:
    backend = cfg.backend.lower()
    if backend == "synthetic":
        return SyntheticEncoder(
            dim=cfg.synthetic_dim,
            grid=cfg.synthetic_grid,
            layout_path=cfg.synthetic_layout,
        )
    from .colvlm import BACKENDS, ColVLMEncoder  # imported lazily: needs torch

    if backend not in BACKENDS:
        raise ValueError(
            f"unknown encoder backend {cfg.backend!r}; "
            f"choose from {sorted(BACKENDS) + ['synthetic']}"
        )
    return ColVLMEncoder(
        backend=backend,
        model_name=cfg.model_name,
        device=cfg.device,
        dtype=cfg.dtype,
        query_batch_size=cfg.batch_size,
    )
