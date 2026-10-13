"""``make_backend``: wrap an existing index in its backend."""

from __future__ import annotations

from ..index.base import BaseIndex
from ..index.numpy_index import NumpyIndex
from .base import Backend
from .numpy_backend import NumpyBackend
from .qdrant_backend import QdrantBackend


def make_backend(index: BaseIndex) -> Backend:
    """The backend wrapper for ``index``, keyed on what the index is.

    A ``NumpyIndex`` gives a :class:`NumpyBackend`; an index whose ``backend`` attribute is
    ``"qdrant"`` gives a :class:`QdrantBackend` (which checks it really is a ``QdrantIndex``).
    A backend that is already wrapped is returned as it is. Anything else raises ``TypeError``:
    a new backend is added by writing its wrapper, not by guessing.
    """
    if isinstance(index, (NumpyBackend, QdrantBackend)):
        return index
    if isinstance(index, NumpyIndex):
        return NumpyBackend(index)
    if getattr(index, "backend", None) == "qdrant":
        return QdrantBackend(index)
    raise TypeError(
        f"no backend wrapper for {type(index).__name__} "
        f"(backend={getattr(index, 'backend', None)!r}); known: numpy, qdrant"
    )
