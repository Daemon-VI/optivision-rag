"""Qdrant-backed multi-vector index.

Qdrant stores one point per page whose "vector" is the whole list of the page's
document vectors, and compares them with MaxSim natively. That is late-interaction
scoring like the numpy index's, with these differences: the similarity is cosine
(Qdrant normalises stored vectors and queries), so its scores are not comparable
with the numpy index's inner products; and on a server an HNSW graph over the
vectors and binary quantization are applied inside the engine (local path mode is
brute force and does not use the quantization setting).

What is sent is always ``decode(page.codes, ...)``, never the float vectors the
codes came from: +/-1 floats for ``binary``, dequantised floats for ``int8`` and the
stored float32 values only for ``method="none"``. Unless that is the method,
Qdrant holds no originals to rescore from, and its own rescoring cannot restore
float32 quality. Its recall, latency and quality were not measured here
(``docs/RESULTS.md``: the deployment path, not the measurement path).

Two modes, which differ only in the collection's quantization setting:

``server_quantized`` (default)
    Create the collection with ``BinaryQuantization`` + ``always_ram``, so a
    server keeps a binarised copy of the sent vectors in RAM. Whether and how it
    rescores is left to Qdrant (no search parameters are passed). This is the
    deployment-shaped path.

``client_binary``
    Create the collection without quantization. With ``method="binary"`` the index
    then holds exactly the +/-1 vectors this project produced — useful to prove
    the client-side pipeline in isolation, at the cost of Qdrant storing a float
    per bit.

Storage numbers reported by :meth:`stats` come from the pipeline's own byte
accounting, not from Qdrant's on-disk files, so they are comparable across
backends.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..compression import Lloyd2Codec, decode
from ..types import CompressedPage, PageRef, SearchHit
from .base import BaseIndex

_NAMESPACE = uuid.UUID("6f2a1f0e-4f8a-4d3e-9a1b-2c7d5e8f0a11")


class QdrantIndex(BaseIndex):
    backend = "qdrant"

    def __init__(
        self,
        path: str | Path = "data/index/qdrant",
        collection: str = "optivision",
        dim: int = 128,
        method: str = "binary",
        url: str | None = None,
        api_key: str | None = None,
        on_disk: bool = True,
        mode: str = "server_quantized",
        recreate: bool = False,
        codec: Lloyd2Codec | None = None,
    ) -> None:
        from qdrant_client import QdrantClient

        self.dim = int(dim)
        self.method = method
        self.codec = codec
        self.collection = collection
        self.mode = mode
        self.path = Path(path)

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
            self.location = url
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.path))
            self.location = str(self.path)

        self._page_stats: list[dict] = []
        self._ensure_collection(on_disk=on_disk, recreate=recreate)
        self._hydrate_page_stats()

    # ------------------------------------------------------------ collection

    def _ensure_collection(self, on_disk: bool, recreate: bool) -> None:
        from qdrant_client import models as qm

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            return

        quantization = None
        if self.mode == "server_quantized":
            quantization = qm.BinaryQuantization(
                binary=qm.BinaryQuantizationConfig(always_ram=True)
            )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=self.dim,
                distance=qm.Distance.COSINE,
                on_disk=on_disk,
                multivector_config=qm.MultiVectorConfig(
                    comparator=qm.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=quantization,
            ),
        )

    def _hydrate_page_stats(self) -> None:
        """Recover per-page byte accounting from the collection's own payloads.

        :meth:`add` fills ``_page_stats`` as it writes, but a process that only
        *opens* an existing collection has never seen those pages, so every
        storage figure — index bytes, compression ratio, token reduction —
        would report zero for a perfectly good index. The numbers are already
        stored on each point, so read them back rather than keeping a second
        copy on disk that could drift out of sync with the collection.
        """
        if self._page_stats:
            return
        try:
            if self.n_pages == 0:
                return
        except Exception:  # noqa: BLE001 - a fresh collection may not be queryable yet
            return

        recovered: list[dict] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                recovered.append(
                    {
                        "page_id": payload.get("page_id", "?"),
                        "n_tokens_before": int(payload.get("n_tokens_before", 0)),
                        "n_tokens_after": int(payload.get("n_tokens_after", 0)),
                        "nbytes": int(payload.get("nbytes", 0)),
                        "raw_nbytes": int(payload.get("raw_nbytes", 0)),
                    }
                )
            if offset is None:
                break
        self._page_stats = recovered

    # ----------------------------------------------------------------- write

    def _vectors_for(self, page: CompressedPage) -> list[list[float]]:
        vectors = decode(page.codes, page.dim, self.method, codec=self.codec)
        return np.ascontiguousarray(vectors, dtype=np.float32).tolist()

    def add(self, pages: Sequence[CompressedPage]) -> None:
        from qdrant_client import models as qm

        if not pages:
            return
        points = []
        for page in pages:
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid5(_NAMESPACE, page.ref.page_id)),
                    vector=self._vectors_for(page),
                    payload={
                        "page_id": page.ref.page_id,
                        "doc_id": page.ref.doc_id,
                        "page_no": page.ref.page_no,
                        "source_path": page.ref.source_path,
                        "image_path": page.ref.image_path,
                        "n_tokens_before": page.n_tokens_before,
                        "n_tokens_after": page.n_tokens_after,
                        "nbytes": page.nbytes,
                        "raw_nbytes": page.raw_nbytes(),
                    },
                )
            )
            self._page_stats.append(
                {
                    "page_id": page.ref.page_id,
                    "n_tokens_before": page.n_tokens_before,
                    "n_tokens_after": page.n_tokens_after,
                    "nbytes": page.nbytes,
                    "raw_nbytes": page.raw_nbytes(),
                }
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    # ---------------------------------------------------------------- search

    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        q = np.ascontiguousarray(query, dtype=np.float32).tolist()
        response = self.client.query_points(
            collection_name=self.collection,
            query=q,
            limit=top_k,
            with_payload=True,
        )
        hits = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    ref=PageRef(
                        doc_id=payload.get("doc_id", "?"),
                        page_no=int(payload.get("page_no", 0)),
                        source_path=payload.get("source_path"),
                        image_path=payload.get("image_path"),
                    ),
                    score=float(point.score),
                    rank=rank,
                )
            )
        return hits

    # ------------------------------------------------------------------ misc

    def save(self) -> None:
        # Qdrant persists on upsert; nothing to flush.
        pass

    def close(self) -> None:
        # Closing an already-closed client (double close, interpreter teardown)
        # raises from inside the client and is not worth surfacing.
        try:
            self.client.close()
        except Exception:  # noqa: BLE001, S110  # pragma: no cover
            pass

    @property
    def n_pages(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def stats(self) -> dict:
        n_pages = self.n_pages
        index_bytes = sum(s["nbytes"] for s in self._page_stats)
        raw_bytes = sum(s["raw_nbytes"] for s in self._page_stats)
        n_vectors = sum(s["n_tokens_after"] for s in self._page_stats)
        tokens_before = sum(s["n_tokens_before"] for s in self._page_stats)
        return {
            "backend": self.backend,
            "mode": self.mode,
            "location": self.location,
            "collection": self.collection,
            "method": self.method,
            "dim": self.dim,
            "n_pages": n_pages,
            "n_vectors": n_vectors,
            "tokens_before": tokens_before,
            "vectors_per_page": n_vectors / max(1, n_pages),
            "index_bytes": index_bytes,
            "raw_bytes": raw_bytes,
            "bytes_per_page": index_bytes / max(1, n_pages),
            "compression_ratio": (raw_bytes / index_bytes) if index_bytes else 0.0,
            "token_reduction": (tokens_before / n_vectors) if n_vectors else 0.0,
        }
