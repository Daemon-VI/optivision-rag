"""Index interface: store compressed pages, answer MaxSim queries."""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np

from ..types import CompressedPage, SearchHit


class BaseIndex(abc.ABC):
    backend: str = "base"

    @abc.abstractmethod
    def add(self, pages: Sequence[CompressedPage]) -> None: ...

    @abc.abstractmethod
    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]: ...

    @abc.abstractmethod
    def save(self) -> None: ...

    @abc.abstractmethod
    def stats(self) -> dict: ...

    @property
    @abc.abstractmethod
    def n_pages(self) -> int: ...

    def close(self) -> None:  # pragma: no cover - default is a no-op
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
