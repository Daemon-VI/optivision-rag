"""Page-index backends behind one interface, and the canonical import place for the indexes.

    from optivision.backends import make_backend, NumpyIndex

``Backend`` is the existing ``BaseIndex`` interface plus ``info`` (what the scores mean) and
``search_many`` (queries in, a :class:`~optivision.core.retrieve.Ranking` out). ``NumpyBackend`` and
``QdrantBackend`` wrap a ``NumpyIndex`` / ``QdrantIndex`` and delegate to it; ``make_backend``
picks the wrapper. Nothing in ``optivision.core`` knows which backend it is talking to.

``BaseIndex``, ``NumpyIndex``, ``QdrantIndex``, ``get_index`` and ``open_index`` are re-exported
here as the new canonical import location. They are the same objects as in ``optivision.index``,
which keeps working; nothing was moved. ``QdrantIndex`` is imported on first access, as
``get_index`` does, so importing this package never needs ``qdrant_client``.

Scores of different backends are not comparable; read ``info.score_kind`` and use
:func:`comparable`. See ``docs/package/BACKEND_ARCHITECTURE.md``, including the page-id convention
and how to add a backend.
"""

from __future__ import annotations

from typing import Any

from ..core.retrieve import Ranking, comparable, require_comparable
from ..index import BaseIndex, NumpyIndex, get_index, open_index
from .base import (
    Backend,
    BackendInfo,
    BackendRetriever,
    legacy_id_mismatches,
    legacy_page_id,
)
from .factory import make_backend
from .numpy_backend import NumpyBackend
from .qdrant_backend import QdrantBackend

__all__ = [
    "Backend",
    "BackendInfo",
    "BackendRetriever",
    "BaseIndex",
    "NumpyBackend",
    "NumpyIndex",
    "QdrantBackend",
    "Ranking",
    "comparable",
    "get_index",
    "legacy_id_mismatches",
    "legacy_page_id",
    "make_backend",
    "open_index",
    "require_comparable",
]


def __getattr__(name: str) -> Any:
    if name == "QdrantIndex":  # lazy, like get_index: not in __all__, so ``import *`` stays offline
        from ..index.qdrant_index import QdrantIndex

        return QdrantIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), "QdrantIndex"])
