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

from ..compression import Lloyd2Codec, decode
from ..types import CompressedPage, PageRef, SearchHit
from .base import BaseIndex

# Peak float32 working set allowed while scoring. 256 MB holds ~520k vectors at
# dim 128 — comfortably more than any laptop-scale corpus, so a small index keeps
# the single-matmul fast path while a million-page index stays bounded instead of
# expanding to tens of gigabytes.
DEFAULT_MAX_DECODED_BYTES = 256 * 1024 * 1024


class NumpyIndex(BaseIndex):
    backend = "numpy"

    def __init__(
        self,
        path: str | Path,
        dim: int,
        method: str = "binary",
        max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES,
        codec: Lloyd2Codec | None = None,
    ) -> None:
        self.path = Path(path)
        self.dim = int(dim)
        self.method = method
        self.max_decoded_bytes = int(max_decoded_bytes)
        # Corpus-fitted state ``method == "lloyd2"`` needs to decode -- see
        # Lloyd2Codec. Every other method ignores this.
        self.codec = codec
        self._codes: np.ndarray | None = None
        self._offsets = np.zeros(1, dtype=np.int64)
        self._refs: list[PageRef] = []
        self._page_stats: list[dict] = []
        self._decoded: np.ndarray | None = None  # cache of +/-1 (or float) vectors

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def load(
        cls,
        path: str | Path,
        max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES,
    ) -> NumpyIndex:
        path = Path(path)
        with open(path / "meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        codec = None
        if meta.get("codec") is not None:
            c = meta["codec"]
            codec = Lloyd2Codec(
                mu=np.array(c["mu"], dtype=np.float32), sigma=c["sigma"], seed=c["seed"]
            )
        idx = cls(
            path,
            dim=meta["dim"],
            method=meta["method"],
            max_decoded_bytes=max_decoded_bytes,
            codec=codec,
        )
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
            "codec": (
                {"mu": self.codec.mu.tolist(), "sigma": self.codec.sigma, "seed": self.codec.seed}
                if self.codec is not None
                else None
            ),
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
                    decode(self._codes, self.dim, self.method, codec=self.codec),
                    dtype=np.float32,
                )
        return self._decoded

    @property
    def decoded_nbytes(self) -> int:
        """Bytes a full float32 expansion of this index would occupy."""
        total = int(self._offsets[-1]) if self._offsets.size else 0
        return total * self.dim * 4

    @staticmethod
    def _segment_maxsim(
        query: np.ndarray, doc: np.ndarray, offsets: np.ndarray, n_pages: int
    ) -> np.ndarray:
        """MaxSim of one query against pages laid out contiguously in ``doc``.

        ``offsets`` is local to ``doc``: page i owns rows offsets[i]:offsets[i+1].
        """
        if doc.shape[0] == 0:
            return np.full(n_pages, -np.inf, dtype=np.float32)
        sims = query @ doc.T  # [n_query_tokens, n_vectors_in_block]
        starts = offsets[:-1].astype(np.intp)
        widths = np.diff(offsets)
        # reduceat rejects an index equal to the row length and emits a garbage
        # column for a zero-width segment, so clamp first and mask afterwards.
        safe = np.minimum(starts, doc.shape[0] - 1)
        per_token_max = np.maximum.reduceat(sims, safe, axis=1)
        scores = per_token_max.sum(axis=0).astype(np.float32)
        empty = np.flatnonzero(widths == 0)
        if empty.size:
            scores[empty] = -np.inf
        return scores

    def score_all(self, query: np.ndarray) -> np.ndarray:
        """MaxSim score of every page against one query. float32 [n_pages].

        The point of this project is that the index is small, so the search path
        must not quietly undo that. Expanding every packed code to float32 at
        once costs ``dim * 4`` bytes per vector — 32x the binary index it was
        just compressed into, which is precisely the memory the pipeline exists
        to save. Below ``max_decoded_bytes`` the full expansion is cached, so a
        laptop-scale corpus keeps the single-matmul fast path; above it the
        pages are scored in blocks and peak memory stays bounded however large
        the corpus grows. The arithmetic is identical either way.
        """
        n_pages = self.n_pages
        if n_pages == 0 or self._codes is None or self._codes.size == 0:
            return np.zeros(n_pages, dtype=np.float32)

        q = np.ascontiguousarray(query, dtype=np.float32)

        if self.decoded_nbytes <= self.max_decoded_bytes:
            return self._segment_maxsim(q, self._decoded_matrix(), self._offsets, n_pages)

        # Streaming path: decode a bounded window of pages at a time.
        self._decoded = None  # never hold the full expansion in this regime
        scores = np.empty(n_pages, dtype=np.float32)
        # Size the block against everything alive at once, not just the answer:
        # the float32 vectors (dim * 4), the uint8 bit expansion decoding walks
        # through (dim), and the query-by-vector similarity rows (n_query * 4).
        # Counting only the first lets real peak memory run to roughly twice the
        # budget, which defeats the point of having one.
        bytes_per_vector = self.dim * 5 + int(q.shape[0]) * 4
        per_block = max(1, self.max_decoded_bytes // bytes_per_vector)
        start = 0
        while start < n_pages:
            limit = self._offsets[start] + per_block
            end = int(np.searchsorted(self._offsets, limit, side="right")) - 1
            end = min(max(end, start + 1), n_pages)  # always make progress
            lo, hi = int(self._offsets[start]), int(self._offsets[end])
            block = np.ascontiguousarray(
                decode(self._codes[lo:hi], self.dim, self.method, codec=self.codec),
                dtype=np.float32,
            )
            local = self._offsets[start : end + 1] - lo
            scores[start:end] = self._segment_maxsim(q, block, local, end - start)
            # Released before the next block is allocated; without this the two
            # overlap and real peak lands at ~1.7x the budget instead of ~1x.
            del block
            start = end
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
        # lloyd2's mu/sigma are shared corpus-fitted state, not per-page cost --
        # charged once here so it amortizes across pages exactly the way
        # storage_summary's kb_per_page already divides index_bytes by n_pages.
        if self.codec is not None:
            index_bytes += self.codec.overhead_bytes
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
