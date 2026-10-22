"""Export an ``optivision.easy`` Index to a Qdrant collection (switched off until its test passes).

    from optivision.easy.integrations import qdrant
    report = qdrant.export(index, "http://localhost:6333", "my_pdfs")
    hits = qdrant.search(client, "my_pdfs", query_vectors, k=10)

One point per page. Its multivector is the page's *stored* vectors as the index scores them (the decoded
codes), its distance is ``DOT`` with ``MultiVectorComparator.MAX_SIM``, so Qdrant computes the same
MaxSim inner product as ``Index.search_vectors``. Rankings can then agree with the index's own exact
search; ``tests/test_easy_qdrant.py`` checks top-10 equality on synthetic corpora, and the feature
``qdrant_export`` stays off in ``presets/easy_ledger.json`` until that result is recorded.

What Qdrant can hold exactly, and what is refused:

* ``datatype="float32"`` (the default) stores each decoded number as float32, which every layout of the
  ladder decodes to, and Qdrant scores a float32 query against float32 vectors. This is exact for every
  layout without a projection. It is not small: Qdrant cannot hold our 8-, 4- or 1-bit codes, so the
  collection holds 4 bytes per stored number (the merged layouts still hold fewer vectors). The report
  says how many bytes were sent.
* ``datatype="float16"`` halves that and is allowed only when every stored number is a float16 value
  (the float16 layouts, 1-bit layouts, and merges of those; checked on every vector). A Qdrant *server*
  rounds each query number to float16 before it scores float16 vectors, so pages that score within
  rounding of each other can swap places there. qdrant-client's local mode does not round.
* ``datatype="uint8"`` is refused: Qdrant's 8-bit storage holds whole numbers 0 to 255 and converts query
  numbers the same way, which cannot represent signed decoded codes or float queries.
* Layouts with a projection or rotation stage are refused: Qdrant would score the user's raw query vectors
  against projected page vectors.
* Pages without vectors are skipped and listed (Qdrant cannot store an empty multivector; the index ranks
  such pages last).

Search exactly on a server: :func:`search` passes ``SearchParams(exact=True)``; the collection is created
with ``hnsw_config.m = 0`` so no approximate graph is built unless asked for. Qdrant's own quantization is
not configured and was not measured by us. Payloads carry the page id, document, page, file (when known)
and the index's evidence under ``"optivision"``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _ledger
from .._claim import Block, claim, lit
from .._types import EasyError
from ._common import index_metadata, missing_extra, split_page_id

__all__ = ["ExportRefused", "QdrantExport", "export", "plan", "search"]

#: point ids: uuid5(namespace, page id), the same for every export of an index
NAMESPACE = uuid.UUID("2b0b1f6c-8d1e-4c3a-9f61-5a7e0c1d9e42")

_BYTES = {"float32": 4, "float16": 2}
_CHECK_ROWS = 65_536


class ExportRefused(EasyError, ValueError):
    """The layout cannot be held exactly in Qdrant with the requested datatype."""


@dataclass
class QdrantExport:
    """What :func:`export` wrote. ``lines`` is the printed summary (every number tagged)."""

    collection: str
    datatype: str
    dim: int
    n_points: int
    n_vectors: int
    bytes_sent: int
    index_vector_bytes: int
    skipped_empty: list[str] = field(default_factory=list)
    exact: str = ""
    lines: Block | None = None

    def __str__(self) -> str:
        return str(self.lines) if self.lines is not None else f"QdrantExport({self.collection})"


@dataclass(frozen=True)
class ExportPlan:
    segments: tuple[Any, ...]
    datatype: str
    dim: int
    exact: str


# ---------------------------------------------------------------------------- what is stored


def _segments_of(index: Any) -> list[Any]:
    """The index's stored segments (``index._store.segments()``), or a corpus / list given directly."""
    store = getattr(index, "_store", None)
    if store is not None and callable(getattr(store, "segments", None)):
        return list(store.segments())
    if isinstance(index, (list, tuple)):
        return list(index)
    if hasattr(index, "decode_rows") and hasattr(index, "offsets"):
        return [index]
    raise TypeError(f"cannot read stored vectors from {type(index).__name__}; pass an optivision.easy Index")


def _pipeline_of(index: Any, segments: Sequence[Any]) -> Any:
    from ...compose import Pipeline

    store = getattr(index, "_store", None)
    if store is not None and callable(getattr(store, "pipeline", None)):
        return store.pipeline()
    found = None if isinstance(index, (list, tuple)) else getattr(index, "pipeline", None)
    if found is None and segments:
        found = getattr(segments[0], "pipeline", None)  # a CompressedCorpus keeps its spec as a dict
    return Pipeline.from_dict(found) if isinstance(found, dict) else found


def _float16_exact(segment: Any) -> bool:
    for lo in range(0, int(segment.num_vectors), _CHECK_ROWS):
        block = segment.decode_rows(lo, min(lo + _CHECK_ROWS, int(segment.num_vectors)))
        if not np.array_equal(block.astype(np.float16).astype(np.float32), block):
            return False
    return True


def plan(index: Any, datatype: str = "float32") -> ExportPlan:
    """Check that the index's layout can be held exactly with ``datatype``; raise ``ExportRefused`` if not.

    Reads every stored vector once for ``datatype="float16"``. Sends nothing.
    """
    from ...stages.base import DimensionReducer

    segments = _segments_of(index)
    if not segments:
        raise ExportRefused("the index holds no segments")
    dt = str(datatype).lower()
    if dt == "uint8":
        raise ExportRefused(
            "Qdrant's uint8 storage holds whole numbers 0 to 255 and converts query numbers the same way; "
            "it cannot represent this index's decoded vectors or float queries. Use datatype=\"float32\".")
    if dt not in _BYTES:
        raise ExportRefused(f"datatype must be \"float32\" or \"float16\", got {datatype!r}")
    pipeline = _pipeline_of(index, segments)
    if pipeline is not None:
        projections = [s.name for s in pipeline.vector_stages if isinstance(s, DimensionReducer)]
        if projections:
            raise ExportRefused(
                f"the layout has a projection ({', '.join(projections)}): Qdrant would score raw query vectors "
                "against projected page vectors. Export an index built without a projection.")
    dims = {int(s.dim) for s in segments}
    if len(dims) != 1:
        raise ExportRefused(f"segments have different dimensions: {sorted(dims)}")
    if dt == "float16":
        for s in segments:
            if not _float16_exact(s):
                raise ExportRefused(
                    f"the stored {s.quantizer.name} values are not float16 numbers, so float16 storage would "
                    "round them. Use datatype=\"float32\" (exact, 4 bytes per stored number).")
        exact = ("stored values are float16 numbers; a Qdrant server rounds query numbers to float16 before "
                 "scoring them, so near-equal pages can swap there")
    else:
        exact = "float32 storage of the decoded vectors and float32 queries: the index's own MaxSim arithmetic"
    return ExportPlan(tuple(segments), dt, dims.pop(), exact)


# ---------------------------------------------------------------------------- client


def _client(client_or_url: Any, api_key: str | None) -> Any:
    if not isinstance(client_or_url, (str, bytes)) and hasattr(client_or_url, "upsert"):
        return client_or_url
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise missing_extra("Qdrant export", "qdrant-client", "qdrant") from exc
    target = str(client_or_url)
    if target == ":memory:":
        return QdrantClient(":memory:")
    if target.startswith(("http://", "https://")):
        return QdrantClient(url=target, api_key=api_key)
    return QdrantClient(path=target)


def _models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise missing_extra("Qdrant export", "qdrant-client", "qdrant") from exc
    return models


def point_id(page_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, str(page_id)))


def _files_of(index: Any) -> dict[str, str]:
    """Page id -> source file, when the Index exposes it (``index.files`` or the store manifest)."""
    files = getattr(index, "files", None)
    if isinstance(files, dict):
        return {str(k): str(v) for k, v in files.items()}
    return {}


# ---------------------------------------------------------------------------- export


def export(index: Any, client_or_url: Any = ":memory:", collection: str = "optivision", *,
           datatype: str = "float32", recreate: bool = False, batch_size: int = 64,
           api_key: str | None = None, payload_text: bool = False, on_disk: bool = False,
           hnsw_m: int = 0) -> QdrantExport:
    """Write the index's stored vectors to ``collection`` (switched off until its test passes).

    ``client_or_url``: a ``QdrantClient``, a server URL, ``":memory:"`` or a local folder. An existing
    collection is an error unless ``recreate=True``. ``payload_text=True`` adds each page's text layer.
    """
    _ledger.require("qdrant_export")
    return _export(index, client_or_url, collection, datatype=datatype, recreate=recreate,
                   batch_size=batch_size, api_key=api_key, payload_text=payload_text, on_disk=on_disk,
                   hnsw_m=hnsw_m)


def _export(index: Any, client_or_url: Any, collection: str, *, datatype: str = "float32",
            recreate: bool = False, batch_size: int = 64, api_key: str | None = None,
            payload_text: bool = False, on_disk: bool = False, hnsw_m: int = 0) -> QdrantExport:
    """:func:`export` without the switch (the equality test runs it while the feature is off)."""
    p = plan(index, datatype)
    models = _models()
    client = _client(client_or_url, api_key)
    if client.collection_exists(collection):
        if not recreate:
            raise ExportRefused(f"collection {collection!r} exists; pass recreate=True to replace it")
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=p.dim,
            distance=models.Distance.DOT,
            datatype=models.Datatype.FLOAT32 if p.datatype == "float32" else models.Datatype.FLOAT16,
            on_disk=on_disk,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
        ),
        hnsw_config=models.HnswConfigDiff(m=int(hnsw_m)),
    )
    shared = index_metadata(index)
    files = _files_of(index)
    store = getattr(index, "_store", None)
    text_of = getattr(store, "text", None) if payload_text else None
    skipped: list[str] = []
    n_points = n_vectors = 0
    batch: list[Any] = []

    def flush() -> None:
        if batch:
            client.upsert(collection_name=collection, points=list(batch), wait=True)
            batch.clear()

    for segment in p.segments:
        offsets = np.asarray(segment.offsets, dtype=np.int64)
        for i, page_id in enumerate(segment.ids):
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            if hi == lo:
                skipped.append(str(page_id))
                continue
            rows = segment.decode_rows(lo, hi)
            doc_id, page = split_page_id(page_id)
            payload: dict[str, Any] = {"page_id": str(page_id), "doc_id": doc_id, "page": page,
                                       "file": files.get(str(page_id)), "optivision": shared}
            if text_of is not None:
                payload["text"] = str(text_of(page_id) or "")
            batch.append(models.PointStruct(id=point_id(page_id), vector=rows.astype(np.float32).tolist(),
                                            payload=payload))
            n_points += 1
            n_vectors += hi - lo
            if len(batch) >= batch_size:
                flush()
    flush()

    sent = n_vectors * p.dim * _BYTES[p.datatype]
    index_bytes = int(sum(int(s.nbytes) for s in p.segments))
    lines = Block([], next=f'qdrant.search(client, "{collection}", query_vectors)')
    lines.add("Qdrant", "{0} pages written to {1} as {2} multivectors (MaxSim, dot product)",
              claim(n_points, "counted"), lit(collection), lit(p.datatype))
    lines.add("Size", "{0} of vector numbers sent; the index's own stored vectors are {1}",
              claim(sent, "counted", text=_size(sent)), claim(index_bytes, "counted", text=_size(index_bytes)))
    if skipped:
        lines.add("Skipped", "{0} pages without vectors (Qdrant cannot store an empty multivector)",
                  claim(len(skipped), "counted"))
    lines.add("Exactness", "{0}", lit(p.exact))
    return QdrantExport(collection=collection, datatype=p.datatype, dim=p.dim, n_points=n_points,
                        n_vectors=n_vectors, bytes_sent=sent, index_vector_bytes=index_bytes,
                        skipped_empty=skipped, exact=p.exact, lines=lines)


def _size(n: int) -> str:
    from .._claim import size

    return size(n)


# ---------------------------------------------------------------------------- search


def _query_rows(queries: Any) -> list[np.ndarray]:
    if hasattr(queries, "offsets") and hasattr(queries, "decode_rows"):
        off = np.asarray(queries.offsets, dtype=np.int64)
        return [queries.decode_rows(int(off[i]), int(off[i + 1])) for i in range(len(off) - 1)]
    if isinstance(queries, np.ndarray) and queries.ndim == 2:
        return [np.asarray(queries, dtype=np.float32)]
    return [np.asarray(q, dtype=np.float32) for q in queries]


def search(client: Any, collection: str, queries: Any, k: int = 10, *, exact: bool = True,
           pipeline: Any = None) -> list[list[tuple[str, float]]]:
    """Top ``k`` ``(page id, score)`` per query from an exported collection.

    ``queries``: a ``MultiVectorCorpus``, one ``[n, d]`` array or a list of them (the model's query
    vectors, as ``Index.search_vectors`` takes them). ``exact=True`` scans every point (no HNSW).
    ``pipeline``: optional; its projection stages are applied to the queries (export refuses projected layouts,
    so this is for collections written by other means).
    """
    models = _models()
    rows = _query_rows(queries)
    out = []
    local = type(getattr(client, "_client", None)).__name__ == "QdrantLocal"  # brute force already
    params = None if local else models.SearchParams(exact=bool(exact))
    for q in rows:
        if pipeline is not None:
            from ...representation import MultiVectorCorpus

            q = pipeline.transform_queries(MultiVectorCorpus(q, [0, q.shape[0]])).decode_rows(0, q.shape[0])
        if q.shape[0] == 0:
            raise ValueError("a query without vectors cannot be scored by Qdrant")
        res = client.query_points(collection_name=collection, query=np.asarray(q, dtype=np.float32).tolist(),
                                  limit=int(k), with_payload=["page_id"], search_params=params)
        out.append([(str(pt.payload["page_id"]), float(pt.score)) for pt in res.points])
    return out
