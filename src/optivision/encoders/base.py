"""Encoder interface: page image -> multi-vector encoding, query text -> vectors."""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image

from ..types import PageEncoding, PageRef


class BaseEncoder(abc.ABC):
    """A late-interaction vision-language encoder.

    Implementations must return **L2-normalised** float32 vectors so that a dot
    product is a cosine similarity; every downstream stage assumes this.
    """

    name: str = "base"

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    @abc.abstractmethod
    def encode_pages(
        self, images: Sequence[Image.Image], refs: Sequence[PageRef]
    ) -> list[PageEncoding]: ...

    @abc.abstractmethod
    def encode_queries(self, queries: Sequence[str]) -> list[np.ndarray]:
        """Return one float32 [n_query_tokens, dim] matrix per query."""

    def encode_pages_iter(
        self, images: Iterable[Image.Image], refs: Iterable[PageRef], batch_size: int = 1
    ) -> Iterable[PageEncoding]:
        batch_imgs: list[Image.Image] = []
        batch_refs: list[PageRef] = []
        for img, ref in zip(images, refs, strict=True):
            batch_imgs.append(img)
            batch_refs.append(ref)
            if len(batch_imgs) >= batch_size:
                yield from self.encode_pages(batch_imgs, batch_refs)
                batch_imgs, batch_refs = [], []
        if batch_imgs:
            yield from self.encode_pages(batch_imgs, batch_refs)


def l2_normalise(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)
