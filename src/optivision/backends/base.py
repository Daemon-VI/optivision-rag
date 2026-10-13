"""The backend interface: a page index that also answers "queries in, ranked ids out".

``Backend`` is ``BaseIndex`` (``add``, ``search``, ``save``, ``stats``, ``n_pages``, ``close``)
plus two things: ``info``, a frozen record of what the backend's scores mean, and ``search_many``,
which takes a batch of queries and returns a :class:`~optivision.core.retrieve.Ranking`. The
wrappers in this package hold an existing ``NumpyIndex`` or ``QdrantIndex`` and delegate to it;
neither index class is changed, and the legacy single-query ``search`` stays reachable.

Nothing in ``optivision.core`` knows which backend sits behind a ``Backend``: it reads ``info``.

Scores of different backends are not comparable. NumPy scores are inner-product MaxSim; Qdrant
scores are MaxSim of cosine similarities. The two coincide only for unit-norm query vectors against
unit-norm stored vectors (binary codes decode to norm ``sqrt(d)``, so Qdrant's score times
``sqrt(d)`` equals the NumPy score there). ``BackendInfo.score_kind`` and ``tie_break`` record the
difference; :func:`~optivision.core.retrieve.comparable` is the guard.

Ids. A backend names a page by ``PageRef.page_id``, ``"<doc_id>::p<page_no>"``. The bridge from a
compressed corpus, ``optivision.storage.to_legacy_pages``, rebuilds each ``PageRef`` from the
corpus id by splitting at the first ``"::p"``: ``"abc"`` comes back as ``"abc::p1"``, ``"a::pb"``
as ``"a::p1"``, ``"x::p01"`` as ``"x::p1"``. Ids that do not already have the form
``"<doc>::p<N>"`` (decimal ``N`` without leading zeros, ``doc`` free of ``"::p"``) therefore do not
match the ids of the store they came from. Nothing here papers over that: a two-tier search that
receives an id its reranker's store does not hold raises ``KeyError``. Use
:func:`legacy_id_mismatches` to list the ids of a corpus that would not survive the bridge, and
give the corpus canonical ids before compressing it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.retrieve import TIE_INDEX, TIE_UNSPECIFIED, Ranking
from ..types import CompressedPage, PageRef, SearchHit

_TIE_BREAKS = (TIE_INDEX, TIE_UNSPECIFIED)


@dataclass(frozen=True)
class BackendInfo:
    """What a backend's answers mean. Compare scores only between equal ``score_kind``.

    ``name``: ``"numpy"`` or ``"qdrant"``. ``exact``: brute force over the stored representation
    (False for an engine that searches an approximate or quantized structure). ``score_kind`` and
    ``tie_break`` are those of the ``Ranking`` the backend returns. ``dense_scores``: the backend
    can produce a full ``[n_queries, n_pages]`` matrix (``score``). ``upsert``: ``add`` overwrites
    a page with the same id instead of appending a second copy.
    """

    name: str
    exact: bool
    score_kind: str
    tie_break: str
    dense_scores: bool
    upsert: bool

    def __post_init__(self) -> None:
        if not self.name or not self.score_kind:
            raise ValueError("name and score_kind must be non-empty strings")
        if self.tie_break not in _TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {_TIE_BREAKS}, got {self.tie_break!r}")


@runtime_checkable
class Backend(Protocol):
    """A page index with typed batch search. ``isinstance`` checks attribute presence only.

    (Before Python 3.12 that check reads every attribute, so ``isinstance(backend, Backend)``
    evaluates ``n_pages``, which asks the engine to count its points.)
    """

    info: BackendInfo

    @property
    def n_pages(self) -> int: ...

    def add(self, pages: Sequence[CompressedPage]) -> None: ...

    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]: ...

    def search_many(self, queries: Any, k: int = 10) -> Ranking: ...

    def save(self) -> None: ...

    def stats(self) -> dict: ...

    def close(self) -> None: ...


class BackendRetriever:
    """A backend as the first stage of a :class:`~optivision.core.retrieve.TwoTierRetriever`.

    ``search(queries, k)`` is ``backend.search_many`` with the store positions dropped
    (``indices=None``): a page's position inside a backend is the order pages were added, which is
    not a contract, so a two-tier search maps the returned ids onto its reranker's store instead.
    That works only when the backend's page ids equal the reranker store's document ids; see the
    id note in this module's docstring. ``n_docs`` is the backend's current page count.
    """

    def __init__(self, backend: Backend) -> None:
        if not callable(getattr(backend, "search_many", None)):
            raise TypeError(f"{type(backend).__name__} is not a Backend (no search_many)")
        self.backend = backend

    @property
    def n_docs(self) -> int:
        return int(self.backend.n_pages)

    def search(self, queries: Any, k: int = 10) -> Ranking:
        return dataclasses.replace(self.backend.search_many(queries, k), indices=None)


def legacy_page_id(store_id: str) -> str | None:
    """The page id a backend reports for a corpus id after ``to_legacy_pages``.

    Mirrors the split in ``optivision.storage.to_legacy_pages`` (kept in step by a test): the part
    before the first ``"::p"`` is the document, the rest is the page number when it is all digits
    and 1 otherwise. ``None`` when ``to_legacy_pages`` itself cannot convert the id (a "digit" that
    ``int`` rejects, such as a superscript).
    """
    doc_id, _, page = store_id.partition("::p")
    try:
        page_no = int(page) if page.isdigit() else 1
    except ValueError:
        return None
    return PageRef(doc_id=doc_id, page_no=page_no).page_id


def legacy_id_mismatches(ids: Iterable[str]) -> list[tuple[str, str | None]]:
    """Corpus ids that a backend fed by ``to_legacy_pages`` would report differently.

    Returns ``(corpus_id, backend_id)`` pairs (``backend_id`` is ``None`` when the bridge itself
    fails). An empty list means every id survives the round trip.
    """
    return [(i, b) for i in ids if (b := legacy_page_id(i)) != i]
