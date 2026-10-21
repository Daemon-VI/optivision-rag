"""A small stand-in for ``optivision.easy._api.Index`` (docs/easy/CONTRACT.md), for the integration tests.

It holds what the contract says other modules may use: ``search(query, k)``, ``search_vectors(q, k)``,
``_store`` (``segments()``, ``pipeline()``, ``text(id)``, ``ids``, ``n_pages``), ``_searcher``
(``search(queries, k) -> list[list[(id, score)]]``, exact top-k merged over segments, ties by store
order), ``_manifest`` and the attributes ``target``, ``layout``, ``evidence``, ``n_pages``, ``pipeline``.
Scoring is the library's own ``CompressedRetriever``, so "the index's own exact search" here is the
arithmetic the real index runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optivision.compose import Pipeline
from optivision.core.retrieve.exact import CompressedRetriever
from optivision.easy._claim import Block, claim, lit
from optivision.easy._types import Hit, LayoutRow, NoSource
from optivision.representation import MultiVectorCorpus


def layout_row(spec: dict, name: str = "layout", plain: str = "a layout") -> LayoutRow:
    return LayoutRow(name=name, plain=plain, spec=spec, chain="8-bit", bytes_per_page=None, ratio_f16=None,
                     ratio_f32=None, lowest_kept=0.978, worst_point=0.981)


def evidence_block() -> Block:
    b = Block([], next="index.search(...)")
    b.add("Layout", "{0}: lowest kept {1}", lit("8-bit"),
          claim(0.978, "catalogue", source="ViDoRe V1 DocVQA 500 pages", text="97.8%"))
    b.add("Size", "{0} on disk", claim(1_000_000, "counted", text="1.0 MB"))
    return b


class FakeStore:
    def __init__(self, segments: Sequence[Any], pipeline: Pipeline, texts: dict[str, str] | None = None) -> None:
        self._segments = list(segments)
        self._pipeline = pipeline
        self._texts = dict(texts or {})
        self.manifest = {"n_pages": sum(len(s) for s in self._segments)}

    def segments(self) -> list[Any]:
        return list(self._segments)

    def pipeline(self) -> Pipeline:
        return self._pipeline

    def text(self, page_id: str) -> str:
        return self._texts.get(page_id, "")

    @property
    def ids(self) -> list[str]:
        return [i for s in self._segments for i in s.ids]

    @property
    def n_pages(self) -> int:
        return len(self.ids)


class FakeSearcher:
    """Exact top-k per segment with ``CompressedRetriever``, merged, ties by store position."""

    def __init__(self, store: FakeStore) -> None:
        self.store = store
        self._retrievers = [CompressedRetriever(s, store.pipeline()) for s in store.segments()]

    def search(self, queries: MultiVectorCorpus, k: int) -> list[list[tuple[str, float]]]:
        per_query: list[list[tuple[float, int, str]]] = [[] for _ in range(len(queries))]
        base = 0
        for seg, retriever in zip(self.store.segments(), self._retrievers, strict=True):
            r = retriever.search(queries, k)
            for q, row in enumerate(r.ids):
                for j, doc in enumerate(row):
                    per_query[q].append((float(r.scores[q, j]), base + int(r.indices[q, j]), doc))
            base += len(seg)
        out = []
        for rows in per_query:
            rows.sort(key=lambda t: (-t[0], t[1]))
            out.append([(doc, score) for score, _, doc in rows[:k]])
        return out


@dataclass(frozen=True)
class FakeHit(Hit):
    """A Hit whose page image is a small PIL image when a file is known (no PDF rendering)."""

    def image(self, dpi: int = 150) -> Any:
        if self.file is None or self.page is None:
            raise NoSource(f"{self.id} was built from vectors; there is no page to render")
        from PIL import Image

        return Image.new("RGB", (40, 52), (255, 255, 255))


@dataclass
class FakeIndex:
    store: FakeStore
    encode: Callable[[list[str]], MultiVectorCorpus] | None = None
    files: dict[str, str] = field(default_factory=dict)
    target: float = 0.97
    layout: LayoutRow | None = None
    evidence: Any = None
    accept_log_keyword: bool = False

    def __post_init__(self) -> None:
        self._store = self.store
        self._searcher = FakeSearcher(self.store)
        self._manifest = dict(self.store.manifest)
        self._querylog: list[Any] | None = []  # an opt-in log that is switched on here, to prove it is bypassed
        self.pipeline = self.store.pipeline()
        if self.accept_log_keyword:
            self.search = self._search_with_log_keyword  # type: ignore[method-assign]

    @property
    def n_pages(self) -> int:
        return self.store.n_pages

    def _hit(self, doc: str, score: float) -> FakeHit:
        page = None
        if "::p" in doc and doc.rsplit("::p", 1)[1].isdigit():
            page = int(doc.rsplit("::p", 1)[1])
        file = self.files.get(doc)
        return FakeHit(doc, file, page if file else None, float(score), self.store.text(doc))

    def search_vectors(self, queries: MultiVectorCorpus, k: int = 5) -> list[list[FakeHit]]:
        return [[self._hit(d, s) for d, s in row] for row in self._searcher.search(queries, k)]

    def search(self, query: str | list[str], k: int = 5) -> Any:
        texts = [query] if isinstance(query, str) else list(query)
        if self._querylog is not None:
            self._querylog.extend(texts)
        rows = self.search_vectors(self.encode(texts), k)
        return rows[0] if isinstance(query, str) else rows

    def _search_with_log_keyword(self, query: str | list[str], k: int = 5, *, log: bool = True) -> Any:
        texts = [query] if isinstance(query, str) else list(query)
        if log and self._querylog is not None:
            self._querylog.extend(texts)
        rows = self.search_vectors(self.encode(texts), k)
        return rows[0] if isinstance(query, str) else rows


def segmented(corpus: MultiVectorCorpus, spec: dict, sizes: Sequence[int]) -> tuple[list[Any], Pipeline]:
    """Fit ``spec`` on the whole corpus, then compress consecutive segments with ``fit=False`` (as a build does)."""
    pipeline = Pipeline.from_dict(spec).fit(corpus)
    segments, lo = [], 0
    for n in sizes:
        segments.append(pipeline.compress(corpus.select(range(lo, lo + n)), fit=False))
        lo += n
    assert lo == len(corpus)
    return segments, pipeline


def hashing_encoder(queries: MultiVectorCorpus, texts: Sequence[str]) -> Callable[[list[str]], MultiVectorCorpus]:
    """Text query -> its row of ``queries`` (a lookup table standing in for a model)."""
    table = {t: i for i, t in enumerate(texts)}

    def encode(batch: list[str]) -> MultiVectorCorpus:
        return queries.select([table[t] for t in batch])

    return encode


def page_ids(n: int, doc: str = "my_pdfs/report.pdf") -> list[str]:
    return [f"{doc}::p{i + 1}" for i in range(n)]


def with_ids(corpus: MultiVectorCorpus, ids: Sequence[str]) -> MultiVectorCorpus:
    return MultiVectorCorpus(np.asarray(corpus.vectors), corpus.offsets, ids=list(ids))
