"""Vector compression: PrunedPage -> CompressedPage."""

from __future__ import annotations

import numpy as np

from ..config import CompressionConfig
from ..types import CompressedPage, PrunedPage
from .binary import (
    code_nbytes,
    maxsim_asymmetric,
    maxsim_asymmetric_batch,
    maxsim_float,
    maxsim_hamming,
    pack_bits,
    unpack_signs,
)

__all__ = [
    "Compressor",
    "code_nbytes",
    "maxsim_asymmetric",
    "maxsim_asymmetric_batch",
    "maxsim_float",
    "maxsim_hamming",
    "pack_bits",
    "unpack_signs",
]


class Compressor:
    """Turns pruned float vectors into the bytes that go into the index."""

    def __init__(self, cfg: CompressionConfig) -> None:
        self.cfg = cfg

    @property
    def bytes_per_vector(self) -> int | None:
        """Index cost of one vector, or None when it depends on the dimension."""
        return None

    def compress(self, page: PrunedPage) -> CompressedPage:
        cfg = self.cfg
        vectors = np.ascontiguousarray(page.embeddings, dtype=np.float32)
        dim = int(vectors.shape[1])

        if not cfg.enabled or cfg.method == "none":
            # Store float32 verbatim, viewed as bytes, so the accounting in
            # CompressedPage stays uniform across methods.
            codes = vectors.view(np.uint8).reshape(vectors.shape[0], dim * 4)
        elif cfg.method == "binary":
            codes = pack_bits(vectors)
        elif cfg.method == "int8":
            codes = _int8_codes(vectors)
        else:
            raise ValueError(f"unknown compression method {cfg.method!r}")

        scale = None
        if cfg.keep_norm:
            scale = np.linalg.norm(vectors, axis=1).astype(np.float32)

        return CompressedPage(
            ref=page.ref,
            codes=np.ascontiguousarray(codes, dtype=np.uint8),
            dim=dim,
            n_tokens_before=page.n_tokens_before,
            n_tokens_after=int(vectors.shape[0]),
            scale=scale,
            stats={
                "method": cfg.method if cfg.enabled else "none",
                "keep_ratio": page.keep_ratio,
                **page.stats,
            },
        )


def _int8_codes(vectors: np.ndarray) -> np.ndarray:
    """Scalar quantization to int8 — the 4x middle ground, for the ablation."""
    q = np.clip(np.round(vectors * 127.0), -127, 127).astype(np.int8)
    return q.view(np.uint8)


def decode_int8(codes: np.ndarray, dim: int) -> np.ndarray:
    return codes.view(np.int8).reshape(-1, dim).astype(np.float32) / 127.0


def decode(codes: np.ndarray, dim: int, method: str) -> np.ndarray:
    """Inverse of :meth:`Compressor.compress` for one page's codes."""
    if method in ("none", None):
        return codes.view(np.float32).reshape(-1, dim)
    if method == "binary":
        return unpack_signs(codes, dim)
    if method == "int8":
        return decode_int8(codes, dim)
    raise ValueError(f"unknown compression method {method!r}")
