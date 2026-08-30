"""Index registry."""

from __future__ import annotations

from pathlib import Path

from ..compression import Lloyd2Codec
from ..config import IndexConfig
from .base import BaseIndex
from .numpy_index import NumpyIndex

__all__ = ["BaseIndex", "NumpyIndex", "get_index", "open_index"]


def get_index(
    cfg: IndexConfig,
    dim: int,
    method: str,
    recreate: bool = False,
    codec: Lloyd2Codec | None = None,
) -> BaseIndex:
    """``codec`` is the fitted :class:`Lloyd2Codec` for ``method == "lloyd2"``;
    every other method ignores it. Reopening an on-disk numpy index (``recreate``
    False, ``meta.json`` present) reloads its own fitted codec and ignores this
    argument."""
    backend = cfg.backend.lower()
    if backend == "numpy":
        path = Path(cfg.path)
        if not recreate and (path / "meta.json").exists():
            return NumpyIndex.load(path)
        return NumpyIndex(path=path, dim=dim, method=method, codec=codec)
    if backend == "qdrant":
        from .qdrant_index import QdrantIndex  # lazy: needs qdrant-client

        return QdrantIndex(
            path=Path(cfg.path) / "qdrant",
            collection=cfg.collection,
            dim=dim,
            method=method,
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
            on_disk=cfg.on_disk,
            recreate=recreate,
            codec=codec,
        )
    raise ValueError(f"unknown index backend {cfg.backend!r}; choose 'numpy' or 'qdrant'")


def open_index(cfg: IndexConfig, dim: int, method: str) -> BaseIndex:
    """Open an existing index for querying."""
    return get_index(cfg, dim=dim, method=method, recreate=False)
