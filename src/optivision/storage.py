"""Search over compressed corpora, including a two-tier (hot / cold) index.

``ExactIndex``
    Exact MaxSim over any store (float corpus or compressed codes): full search,
    or rescoring of a given candidate set.

``TieredIndex``
    A small *hot* representation (e.g. merged vectors as binary codes, meant for
    RAM) finds ``candidates`` documents per query; a *cold* representation
    (e.g. int8 or float, meant for SSD or object storage) rescores only those.
    Retrieval quality is then bounded by two things -- whether the right
    document reaches the shortlist, and how good the cold scores are -- and both
    are measured, not assumed (``docs/UNIVERSAL.md``).

``StorageBackend`` / ``NpzStorage``
    The interface a persistent store implements, with a local ``.npz``
    implementation. Vector databases (Qdrant, Milvus, Weaviate, Vespa, ...) are
    reached through :func:`to_legacy_pages` today -- which feeds the existing
    :mod:`optivision.index` backends -- and through dedicated backends later.
"""

from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import Any

import numpy as np

from .compose import CompressedCorpus, Pipeline, load_compressed, save_compressed
from .representation import MultiVectorCorpus
from .scoring import DEFAULT_MAX_BLOCK_BYTES, maxsim_matrix, rank, segment_reduce
from .types import CompressedPage, PageRef


class ExactIndex:
    """Exact MaxSim search over one store. ``query_transform`` is the pipeline's
    projection (queries must be projected exactly like the documents were)."""

    def __init__(self, store: Any, query_transform: Pipeline | None = None,
                 max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES) -> None:
        self.store = store
        self.pipeline = query_transform
        self.max_block_bytes = max_block_bytes

    def _q(self, queries: MultiVectorCorpus) -> MultiVectorCorpus:
        return self.pipeline.transform_queries(queries) if self.pipeline is not None else queries

    def score(self, queries: MultiVectorCorpus) -> np.ndarray:
        return maxsim_matrix(self._q(queries), self.store, max_block_bytes=self.max_block_bytes)

    def search(self, queries: MultiVectorCorpus, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score(queries)
        top = rank(scores, k)
        return top, np.take_along_axis(scores, top, axis=1)

    def rescore(self, queries: MultiVectorCorpus, candidates: np.ndarray) -> np.ndarray:
        """Exact scores of each query against its own candidate documents only."""
        q = self._q(queries)
        cand = np.asarray(candidates, dtype=np.int64)
        out = np.full(cand.shape, -np.inf, dtype=np.float32)
        offsets = self.store.offsets
        for qi in range(cand.shape[0]):
            lo_q, hi_q = int(q.offsets[qi]), int(q.offsets[qi + 1])
            if hi_q == lo_q:
                out[qi] = 0.0
                continue
            qv = q.decode_rows(lo_q, hi_q)
            docs = cand[qi]
            spans = [(int(offsets[d]), int(offsets[d + 1])) for d in docs]
            blocks = [self.store.decode_rows(lo, hi) for lo, hi in spans]
            counts = np.array([b.shape[0] for b in blocks], dtype=np.int64)
            if counts.sum() == 0:
                continue
            block = np.concatenate(blocks, axis=0)
            local = np.concatenate([[0], np.cumsum(counts)])
            per_token = segment_reduce(qv @ block.T, local, np.maximum, axis=1, fill=-np.inf)
            out[qi] = per_token.sum(axis=0)
        return out


class TieredIndex:
    """Shortlist on the hot store, rescore the shortlist on the cold store."""

    def __init__(self, hot: ExactIndex, cold: ExactIndex, candidates: int = 50) -> None:
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self.hot = hot
        self.cold = cold
        self.candidates = candidates
        self.last_timing: dict[str, float] = {}

    def score(self, queries: MultiVectorCorpus) -> np.ndarray:
        """A full score matrix: rescored shortlist on top, everything else below it.

        Shortlisted documents carry their cold score; the rest keep their hot
        score shifted below the lowest shortlisted score, so any ranking metric
        computed from this matrix sees exactly what a tiered search returns.
        """
        t0 = time.perf_counter()
        hot = self.hot.score(queries)
        t1 = time.perf_counter()
        m = min(self.candidates, hot.shape[1])
        shortlist = rank(hot, m)
        cold = self.cold.rescore(queries, shortlist)
        t2 = time.perf_counter()
        out = hot.copy()
        floor = np.nanmin(np.where(np.isfinite(cold), cold, np.nan), axis=1)
        top_hot = np.nanmax(np.where(np.isfinite(hot), hot, np.nan), axis=1)
        shift = np.where(np.isfinite(floor), top_hot - floor + 1.0, 0.0)
        out -= shift[:, None].astype(np.float32)
        np.put_along_axis(out, shortlist, cold, axis=1)
        self.last_timing = {"hot_seconds": t1 - t0, "rescore_seconds": t2 - t1}
        return out

    def search(self, queries: MultiVectorCorpus, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score(queries)
        top = rank(scores, k)
        return top, np.take_along_axis(scores, top, axis=1)


# ---------------------------------------------------------------- persistence


class StorageBackend(abc.ABC):
    """Where compressed corpora live between runs."""

    @abc.abstractmethod
    def write(self, name: str, compressed: CompressedCorpus) -> None: ...

    @abc.abstractmethod
    def read(self, name: str) -> CompressedCorpus: ...

    @abc.abstractmethod
    def names(self) -> list[str]: ...


class NpzStorage(StorageBackend):
    """One pickle-free ``.npz`` file per compressed corpus in a directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, name: str, compressed: CompressedCorpus) -> None:
        save_compressed(compressed, self.root / f"{name}.npz")

    def read(self, name: str) -> CompressedCorpus:
        return load_compressed(self.root / f"{name}.npz")

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.npz"))


_LEGACY_METHODS = {"float32": "none", "binary": "binary", "int8": "int8"}


def to_legacy_pages(compressed: CompressedCorpus) -> list[CompressedPage]:
    """Hand a pipeline's output to the original index backends (numpy, Qdrant).

    Only codecs those backends can decode are accepted: float32, plain binary
    and fixed-scale int8. Others raise rather than being silently re-encoded.
    """
    q = compressed.quantizer
    method = _LEGACY_METHODS.get(q.name)
    params = q.params()
    if method is None or (q.name == "int8" and params.get("scale") != "fixed") or (
        q.name == "binary" and params.get("center")
    ):
        raise NotImplementedError(f"the original index backends cannot decode {q.to_dict()}")
    before = int(compressed.stats.get("vectors_before", compressed.num_vectors))
    per_doc_before = before / max(1, len(compressed))
    pages = []
    for i, doc_id in enumerate(compressed.ids):
        lo, hi = int(compressed.offsets[i]), int(compressed.offsets[i + 1])
        ref_doc, _, page = doc_id.partition("::p")
        pages.append(CompressedPage(
            ref=PageRef(doc_id=ref_doc, page_no=int(page) if page.isdigit() else 1),
            codes=compressed.codes[lo:hi], dim=compressed.dim,
            n_tokens_before=round(per_doc_before), n_tokens_after=hi - lo,
            stats={"method": method},
        ))
    return pages
