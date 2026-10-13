"""The Qdrant backend: ``QdrantIndex`` plus typed batch search.

``QdrantBackend`` holds a :class:`~optivision.index.qdrant_index.QdrantIndex` and delegates
``add``, ``search``, ``save``, ``stats``, ``n_pages`` and ``close`` to it. ``search_many`` loops the
existing single-query ``search`` and returns a ``Ranking`` of page ids.

What the numbers are (``info``):

* ``score_kind="maxsim_cosine"``: the engine's MaxSim over cosine similarity (stored vectors and
  queries are both L2-normalised by the engine). It equals the NumPy backend's inner-product
  score only for unit-norm query tokens against unit-norm stored vectors; for other queries the
  two rank differently. Never compare these scores with NumPy scores.
* ``tie_break="unspecified"``: the engine decides the order of equal scores, and which of several
  tied pages reach the top ``k``. The ``Ranking`` keeps the engine's order.
* ``exact`` is True only in Qdrant's local (path) mode, which is a brute-force scan over the
  stored float32 vectors; a server uses an HNSW graph and binary quantization, which are
  approximate and were not measured here. Local mode is for testing, not a performance statement.
* What is stored are the *decoded codes* (``QdrantIndex`` decodes before upserting): +/-1 floats
  for ``binary``, dequantized floats for ``int8``, the original values only for ``method="none"``.
  Server-side rescoring therefore cannot recover float32 quality that the codes already lost.

The wrapper refuses what the engine handles badly, before anything is sent: an empty page (the
engine raises ``IndexError`` after it has registered the point, which leaves a local collection
unusable until it is reopened), a page of the wrong dimension, a query without vectors, a
non-finite query, a wrong query dimension. ``Ranking.scores`` are the engine's Python floats
narrowed to float32.

``qdrant_client`` is never imported here; ``QdrantIndex`` imports it when it is constructed. Memory
and device: queries go to the engine as Python lists; no device is involved.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.retrieve import (
    SCORE_MAXSIM_COSINE,
    TIE_UNSPECIFIED,
    Ranking,
    as_queries,
    check_k,
)
from ..types import CompressedPage, SearchHit
from .base import BackendInfo


class QdrantBackend:
    """Typed batch search over a ``QdrantIndex`` (see the module docstring)."""

    def __init__(self, index: Any) -> None:
        from ..index.qdrant_index import QdrantIndex  # imports without qdrant_client

        if not isinstance(index, QdrantIndex):
            raise TypeError(f"QdrantBackend wraps a QdrantIndex, got {type(index).__name__}")
        self.index = index
        # QdrantIndex.location is the URL for a server and str(path) for a local store
        local = str(index.location) == str(index.path)
        self.info = BackendInfo(
            name="qdrant",
            exact=local,
            score_kind=SCORE_MAXSIM_COSINE,
            tie_break=TIE_UNSPECIFIED,
            dense_scores=False,
            upsert=True,
        )

    # ----------------------------------------------------- BaseIndex part

    @property
    def n_pages(self) -> int:
        return self.index.n_pages

    def add(self, pages: Sequence[CompressedPage]) -> None:
        """``QdrantIndex.add`` after checking every page; a repeated page id overwrites."""
        pages = list(pages)
        for page in pages:
            if page.n_vectors == 0:
                raise ValueError(
                    f"page {page.ref.page_id!r} has no vectors; Qdrant cannot store an empty "
                    "multivector (nothing was added)"
                )
            if page.dim != self.index.dim:
                raise ValueError(
                    f"page {page.ref.page_id!r} has dimension {page.dim} but the index holds "
                    f"{self.index.dim} (nothing was added)"
                )
        self.index.add(pages)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        """``QdrantIndex.search``, unchanged."""
        return self.index.search(query, top_k)

    def save(self) -> None:
        self.index.save()

    def stats(self) -> dict:
        return self.index.stats()

    def close(self) -> None:
        self.index.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------- typed batch

    def search_many(self, queries: Any, k: int = 10) -> Ranking:
        """The engine's top ``k`` pages per query as a ``Ranking`` of page ids (``k >= 1``).

        ``indices`` is ``None`` (an engine does not expose store positions). Rows shorter than the
        widest row (an index with fewer than ``k`` pages returns fewer) are padded with ``-inf``.
        """
        k = check_k(k)
        q = as_queries(queries, dim=self.index.dim)
        empty = np.flatnonzero(q.counts == 0)
        if empty.size:
            raise ValueError(
                f"query {int(empty[0])} has no vectors; the engine cannot score an empty "
                "multivector query (the in-process retrievers score it 0)"
            )
        hits = []
        for i in range(len(q)):
            lo, hi = q.span(i)
            hits.append(self.index.search(q.decode_rows(lo, hi), top_k=k))
        width = max((len(row) for row in hits), default=0)
        scores = np.full((len(hits), width), -np.inf, dtype=np.float32)
        for i, row in enumerate(hits):
            scores[i, : len(row)] = [h.score for h in row]  # float64 -> float32
        return Ranking(
            ids=[[h.ref.page_id for h in row] for row in hits],
            scores=scores,
            indices=None,
            score_kind=self.info.score_kind,
            tie_break=self.info.tie_break,
        )
