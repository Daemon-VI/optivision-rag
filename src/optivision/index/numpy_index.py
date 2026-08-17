"""Exact brute-force MaxSim index over packed codes.

This is the reference implementation: no approximation, no ANN graph, no server.
Every quality number in the benchmark is produced here so that a change in the
score can only come from pruning or quantization — never from a recall miss in
an approximate index. It is also the fallback when Qdrant is unavailable.

Layout on disk (``<path>/``):

    codes.npy     uint8 [total_vectors, nbytes]  all pages concatenated
    offsets.npy   int64 [n_pages + 1]            page i owns [off[i], off[i+1])
    meta.json     refs, dim, method, per-page stats
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..compression import decode
from ..types import CompressedPage, PageRef, SearchHit
from .base import BaseIndex


class NumpyIndex(BaseIndex):
    backend = "numpy"

    def __init__(self, path: str | Path, dim: int, method: str = "binary") -> None:
        self.path = Path(path)
        self.dim = int(dim)
        self.method = method
        self._codes: np.ndarray | None = None
        self._offsets = np.zeros(1, dtype=np.int64)
        self._refs: list[PageRef] = []
        self._page_stats: list[dict] = []
        self._decoded: np.ndarray | None = None  # cache of +/-1 (or float) vectors

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def load(cls, path: str | Path) -> NumpyIndex:
        path = Path(path)
        with open(path / "meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        idx = cls(path, dim=meta["dim"], method=meta["method"])
        idx._codes = np.load(path / "codes.npy")
        idx._offsets = np.load(path / "offsets.npy")
        idx._refs = [PageRef(**r) for r in meta["refs"]]
        idx._page_stats = meta.get("page_stats", [])
        return idx

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        codes = self._codes if self._codes is not None else np.zeros((0, 0), dtype=np.uint8)
        np.save(self.path / "codes.npy", codes)
        np.save(self.path / "offsets.npy", self._offsets)
        meta = {
            "backend": self.backend,
            "dim": self.dim,
            "method": self.method,
            "n_pages": self.n_pages,
            "refs": [r.__dict__ for r in self._refs],
            "page_stats": self._page_stats,
        }
        with open(self.path / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    # ----------------------------------------------------------------- write

    def add(self, pages: Sequence[CompressedPage]) -> None:
        if not pages:
            return
        blocks = [p.codes for p in pages]
        if self._codes is not None and self._codes.size:
            blocks.insert(0, self._codes)
        self._codes = np.concatenate(blocks, axis=0)

        counts = np.array([p.n_vectors for p in pages], dtype=np.int64)
        new_offsets = self._offsets[-1] + np.cumsum(counts)
        self._offsets = np.concatenate([self._offsets, new_offsets])

        self._refs.extend(p.ref for p in pages)
        self._page_stats.extend(
            {
                "page_id": p.ref.page_id,
                "n_tokens_before": p.n_tokens_before,
                "n_tokens_after": p.n_tokens_after,
                "nbytes": p.nbytes,
                "raw_nbytes": p.raw_nbytes(),
            }
            for p in pages
        )
        self._decoded = None

    # ---------------------------------------------------------------- search

    def _decoded_matrix(self) -> np.ndarray:
        if self._decoded is None:
            if self._codes is None or self._codes.size == 0:
                self._decoded = np.zeros((0, self.dim), dtype=np.float32)
            else:
                self._decoded = np.ascontiguousarray(
                    decode(self._codes, self.dim, self.method), dtype=np.float32
                )
        return self._decoded

    def score_all(self, query: np.ndarray) -> np.ndarray:
        """MaxSim score of every page against one query. float32 [n_pages]."""
        doc = self._decoded_matrix()
        n_pages = self.n_pages
        if n_pages == 0 or doc.shape[0] == 0:
            return np.zeros(n_pages, dtype=np.float32)
        q = np.ascontiguousarray(query, dtype=np.float32)
        sims = q @ doc.T  # [n_query_tokens, total_vectors]
        per_token_max = np.maximum.reduceat(sims, self._offsets[:-1].astype(np.intp), axis=1)
        # reduceat emits a garbage column for empty segments; mask them out.
        empty = np.flatnonzero(np.diff(self._offsets) == 0)
        scores = per_token_max.sum(axis=0).astype(np.float32)
        if empty.size:
            scores[empty] = -np.inf
        return scores

    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        scores = self.score_all(query)
        if scores.size == 0:
            return []
        k = min(top_k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            SearchHit(ref=self._refs[int(i)], score=float(scores[int(i)]), rank=r)
            for r, i in enumerate(top, start=1)
        ]

    # ------------------------------------------------------------------ misc

    @property
    def n_pages(self) -> int:
        return len(self._refs)

    @property
    def refs(self) -> list[PageRef]:
        return list(self._refs)

    def stats(self) -> dict:
        total_vectors = int(self._offsets[-1]) if self._offsets.size else 0
        index_bytes = int(self._codes.nbytes) if self._codes is not None else 0
        raw_bytes = sum(s["raw_nbytes"] for s in self._page_stats)
        tokens_before = sum(s["n_tokens_before"] for s in self._page_stats)
        return {
            "backend": self.backend,
            "method": self.method,
            "dim": self.dim,
            "n_pages": self.n_pages,
            "n_vectors": total_vectors,
            "tokens_before": tokens_before,
            "vectors_per_page": total_vectors / max(1, self.n_pages),
            "index_bytes": index_bytes,
            "raw_bytes": raw_bytes,
            "bytes_per_page": index_bytes / max(1, self.n_pages),
            "compression_ratio": (raw_bytes / index_bytes) if index_bytes else 0.0,
            "token_reduction": (tokens_before / total_vectors) if total_vectors else 0.0,
        }
