"""The NumPy backend: ``NumpyIndex`` plus typed batch search.

``NumpyBackend`` holds a :class:`~optivision.index.numpy_index.NumpyIndex` and delegates ``add``,
``search``, ``save``, ``stats``, ``n_pages`` and ``close`` to it. It adds ``info``, ``score`` (the
dense ``[n_queries, n_pages]`` matrix) and ``search_many``.

``search_many`` is ``NumpyIndex.score_all`` per query followed by ``scoring.rank``: scores are
float32 inner-product MaxSim over the decoded codes (exact over the stored representation, which
for ``binary`` and ``int8`` is not the original float vectors), and ties are broken by ascending
page position. The legacy single-query ``NumpyIndex.search`` (``argpartition`` plus an unstable
``argsort``) leaves ties unspecified; it is unchanged and stays reachable as ``search``, which is
why this class does not route ``search_many`` through it.

Conventions pinned here (invariant I2): an empty page scores ``-inf`` and an empty query 0, as in
``maxsim_matrix``. ``NumpyIndex.score_all`` returns 0 for every page when the whole index holds no
vectors; ``score`` maps that to ``-inf``.

Equality with the in-process retrievers is rounding-level, not bitwise: ``score_all`` forms the
products one query at a time and adds a page's per-token maxima with ``sum``, while ``maxsim_matrix``
forms them for a batch and adds with a segmented ``reduceat``, so the last bits of the float32 sums
differ. A near-tie can therefore swap places between the two.

Memory. ``score`` allocates the dense float32 ``[n_queries, n_pages]`` matrix. ``NumpyIndex`` caches
the decoded float32 index when it fits ``max_decoded_bytes`` and re-decodes it block by block for
every query otherwise. Device: CPU NumPy only; ``OPTIVISION_SCORE_DEVICE`` has no effect here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.retrieve import (
    SCORE_MAXSIM_IP,
    TIE_INDEX,
    Ranking,
    as_queries,
    check_k,
    ranking_from_scores,
)
from ..index.numpy_index import DEFAULT_MAX_DECODED_BYTES, NumpyIndex
from ..types import CompressedPage, SearchHit
from .base import BackendInfo


class NumpyBackend:
    """Typed batch search over a ``NumpyIndex`` (see the module docstring)."""

    def __init__(self, index: NumpyIndex) -> None:
        if not isinstance(index, NumpyIndex):
            raise TypeError(f"NumpyBackend wraps a NumpyIndex, got {type(index).__name__}")
        self.index = index
        self.info = BackendInfo(
            name="numpy",
            exact=True,
            score_kind=SCORE_MAXSIM_IP,
            tie_break=TIE_INDEX,
            dense_scores=True,
            upsert=False,
        )

    @classmethod
    def load(
        cls, path: str | Path, max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES
    ) -> NumpyBackend:
        """``NumpyBackend(NumpyIndex.load(path, max_decoded_bytes))``."""
        return cls(NumpyIndex.load(path, max_decoded_bytes=max_decoded_bytes))

    # ----------------------------------------------------- BaseIndex part

    @property
    def n_pages(self) -> int:
        return self.index.n_pages

    def add(self, pages: Sequence[CompressedPage]) -> None:
        """``NumpyIndex.add``: appends. A page added twice is stored twice (``info.upsert``)."""
        self.index.add(pages)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        """``NumpyIndex.search``, unchanged: one query, ties broken in an unspecified order."""
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

    def score(self, queries: Any) -> np.ndarray:
        """Float32 ``[n_queries, n_pages]`` inner-product MaxSim (``NumpyIndex.score_all`` per query)."""
        q = as_queries(queries, dim=self.index.dim)
        n_pages = self.index.n_pages
        if n_pages and self.index.decoded_nbytes == 0:
            return np.full((len(q), n_pages), -np.inf, dtype=np.float32)
        out = np.empty((len(q), n_pages), dtype=np.float32)
        for i in range(len(q)):
            lo, hi = q.span(i)
            out[i] = self.index.score_all(q.decode_rows(lo, hi))
        return out

    def search_many(self, queries: Any, k: int = 10) -> Ranking:
        """The ``k`` best pages per query as a ``Ranking`` (``k >= 1``; narrower if fewer pages).

        ``ids`` are page ids (``"<doc_id>::p<page_no>"``); ``indices`` are positions in this
        index's own page order, not in any other store.
        """
        k = check_k(k)
        page_ids = [ref.page_id for ref in self.index.refs]
        return ranking_from_scores(self.score(queries), page_ids, k, self.info.score_kind)
