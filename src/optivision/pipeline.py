"""The OptiVision RAG pipeline.

    page image
        -> VLM encoder      multi-vector page encoding      (unchanged model)
        -> spatial pruning  drop blank patches
        -> redundancy prune collapse duplicate patches
        -> binary quantize  128 floats -> 128 bits
        -> index            Qdrant MaxSim / exact numpy
                                                            <- query vectors

Only the middle two stages are ours. The model is used as published and the
query path is untouched, which is what makes the compression a drop-in change
for an existing ColPali deployment.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .compression import Compressor
from .config import Config
from .encoders import BaseEncoder, get_encoder
from .index import BaseIndex, get_index
from .ingest import iter_pages
from .pruning import TokenPruner
from .types import CompressedPage, PageRef, SearchResult


@dataclass
class IndexReport:
    n_pages: int
    n_tokens_before: int
    n_tokens_after: int
    index_bytes: int
    raw_bytes: int
    encode_seconds: float
    prune_seconds: float
    compress_seconds: float
    write_seconds: float

    @property
    def token_reduction(self) -> float:
        return self.n_tokens_before / max(1, self.n_tokens_after)

    @property
    def compression_ratio(self) -> float:
        return self.raw_bytes / max(1, self.index_bytes)

    @property
    def bytes_per_page(self) -> float:
        return self.index_bytes / max(1, self.n_pages)

    def as_dict(self) -> dict:
        return {
            "n_pages": self.n_pages,
            "n_tokens_before": self.n_tokens_before,
            "n_tokens_after": self.n_tokens_after,
            "tokens_per_page_before": self.n_tokens_before / max(1, self.n_pages),
            "tokens_per_page_after": self.n_tokens_after / max(1, self.n_pages),
            "token_reduction": self.token_reduction,
            "index_bytes": self.index_bytes,
            "raw_bytes": self.raw_bytes,
            "bytes_per_page": self.bytes_per_page,
            "compression_ratio": self.compression_ratio,
            "encode_seconds": self.encode_seconds,
            "prune_seconds": self.prune_seconds,
            "compress_seconds": self.compress_seconds,
            "write_seconds": self.write_seconds,
        }


class OptiVisionRAG:
    def __init__(self, cfg: Config | None = None, encoder: BaseEncoder | None = None) -> None:
        self.cfg = cfg or Config()
        self._encoder = encoder
        self.pruner = TokenPruner(self.cfg.pruning)
        self.compressor = Compressor(self.cfg.compression)
        self._index: BaseIndex | None = None

    # --------------------------------------------------------------- lazies

    @property
    def encoder(self) -> BaseEncoder:
        if self._encoder is None:
            self._encoder = get_encoder(self.cfg.encoder)
        return self._encoder

    @property
    def method(self) -> str:
        c = self.cfg.compression
        return c.method if c.enabled else "none"

    def index(self, recreate: bool = False) -> BaseIndex:
        if self._index is None:
            self._index = get_index(
                self.cfg.index, dim=self.encoder.dim, method=self.method, recreate=recreate
            )
        return self._index

    # ------------------------------------------------------------- indexing

    def process_page(self, image: Image.Image, ref: PageRef) -> CompressedPage:
        """Full per-page path, exposed for notebooks and the demo app."""
        enc = self.encoder.encode_pages([image], [ref])[0]
        pruned = self.pruner.prune(enc, image)
        return self.compressor.compress(pruned)

    def build(
        self,
        source: str | Path | Iterable[tuple[PageRef, Image.Image]],
        recreate: bool = True,
        progress=None,
        float_cache: str | Path | None = None,
    ) -> IndexReport:
        """Index a corpus.

        Args:
            source: a path to scan, or an iterable of (ref, image) pairs.
            recreate: wipe an existing index first.
            progress: optional callable(done, ref) for UI feedback.
            float_cache: if set, also write the pruned float32 vectors there so
                :meth:`search` can rerank. Costs 32x the index — use it to
                measure the quality ceiling, not in a deployment.
        """
        pages = iter_pages(source, self.cfg.ingest) if isinstance(source, (str, Path)) else source

        idx = self.index(recreate=recreate)
        t_enc = t_prune = t_comp = t_write = 0.0
        n_pages = tokens_before = tokens_after = 0
        index_bytes = raw_bytes = 0
        batch: list[CompressedPage] = []
        cache_vectors: list[np.ndarray] = []
        cache_refs: list[str] = []
        batch_size = max(1, self.cfg.encoder.max_pages_in_flight)

        for ref, image in pages:
            t0 = time.perf_counter()
            enc = self.encoder.encode_pages([image], [ref])[0]
            t1 = time.perf_counter()
            pruned = self.pruner.prune(enc, image)
            t2 = time.perf_counter()
            compressed = self.compressor.compress(pruned)
            t3 = time.perf_counter()

            t_enc += t1 - t0
            t_prune += t2 - t1
            t_comp += t3 - t2

            n_pages += 1
            tokens_before += compressed.n_tokens_before
            tokens_after += compressed.n_tokens_after
            index_bytes += compressed.nbytes
            raw_bytes += compressed.raw_nbytes()

            batch.append(compressed)
            if float_cache is not None:
                cache_vectors.append(pruned.embeddings.astype(np.float32))
                cache_refs.append(ref.page_id)

            if len(batch) >= batch_size:
                t4 = time.perf_counter()
                idx.add(batch)
                t_write += time.perf_counter() - t4
                batch = []

            if progress is not None:
                progress(n_pages, ref)

        if batch:
            t4 = time.perf_counter()
            idx.add(batch)
            t_write += time.perf_counter() - t4

        t4 = time.perf_counter()
        idx.save()
        t_write += time.perf_counter() - t4

        if float_cache is not None:
            _write_float_cache(float_cache, cache_refs, cache_vectors)

        self._save_manifest(idx)
        return IndexReport(
            n_pages=n_pages,
            n_tokens_before=tokens_before,
            n_tokens_after=tokens_after,
            index_bytes=index_bytes,
            raw_bytes=raw_bytes,
            encode_seconds=t_enc,
            prune_seconds=t_prune,
            compress_seconds=t_comp,
            write_seconds=t_write,
        )

    def _save_manifest(self, idx: BaseIndex) -> None:
        path = Path(self.cfg.index.path)
        path.mkdir(parents=True, exist_ok=True)
        manifest = {"config": self.cfg.to_dict(), "index": idx.stats()}
        with open(path / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)

    # --------------------------------------------------------------- search

    def search(self, query: str, top_k: int | None = None) -> SearchResult:
        return self.search_many([query], top_k)[0]

    def search_many(self, queries: Sequence[str], top_k: int | None = None) -> list[SearchResult]:
        k = top_k or self.cfg.search.top_k
        idx = self.index()
        t0 = time.perf_counter()
        query_vectors = self.encoder.encode_queries(list(queries))
        encode_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(queries))

        results = []
        for q, qv in zip(queries, query_vectors, strict=True):
            t1 = time.perf_counter()
            hits = idx.search(qv, top_k=max(k, self.cfg.search.prefilter_k if self.cfg.search.rerank else k))
            if self.cfg.search.rerank and self.cfg.search.rerank_cache:
                hits = _rerank(hits, qv, self.cfg.search.rerank_cache, k)
            hits = hits[:k]
            for rank, hit in enumerate(hits, start=1):
                hit.rank = rank
            search_ms = (time.perf_counter() - t1) * 1000.0
            results.append(
                SearchResult(
                    query=q,
                    hits=hits,
                    latency_ms=encode_ms + search_ms,
                    candidates_scored=idx.n_pages,
                    reranked=len(hits) if self.cfg.search.rerank else 0,
                )
            )
        return results

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None


# --------------------------------------------------------------- float cache


def _write_float_cache(path: str | Path, page_ids: list[str], vectors: list[np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.array([v.shape[0] for v in vectors], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    flat = (
        np.concatenate(vectors, axis=0)
        if vectors
        else np.zeros((0, 0), dtype=np.float32)
    )
    np.savez(path, vectors=flat, offsets=offsets, page_ids=np.array(page_ids, dtype=object))


def load_float_cache(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    flat, offsets, page_ids = data["vectors"], data["offsets"], data["page_ids"]
    return {
        str(pid): flat[int(offsets[i]) : int(offsets[i + 1])] for i, pid in enumerate(page_ids)
    }


def _rerank(hits, query_vectors: np.ndarray, cache_path: str, top_k: int):
    """Rescore prefilter candidates with full-precision vectors."""
    from .compression import maxsim_float

    cache = load_float_cache(cache_path)
    rescored = []
    for hit in hits:
        doc = cache.get(hit.ref.page_id)
        if doc is None:
            rescored.append(hit)
            continue
        hit.score = maxsim_float(query_vectors, doc)
        hit.stage = "rerank"
        rescored.append(hit)
    rescored.sort(key=lambda h: -h.score)
    return rescored[:top_k]
