"""Core data types shared across the OptiVision RAG pipeline.

A page moves through the system in three shapes:

    PageEncoding   raw multi-vector output of the vision-language model
    PrunedPage     after spatial + redundancy pruning (fewer vectors)
    CompressedPage after binary quantization (bit-packed, index-ready)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PageRef:
    """Identity of a single page inside a document collection."""

    doc_id: str
    page_no: int  # 1-based
    source_path: str | None = None
    image_path: str | None = None

    @property
    def page_id(self) -> str:
        return f"{self.doc_id}::p{self.page_no}"


@dataclass
class PatchGrid:
    """Layout of the image-token grid produced by the vision encoder.

    ``rows * cols`` must equal the number of image tokens the model emits for a
    page. ``token_index[r, c]`` gives the position of that patch inside the
    embedding matrix, so a patch can be mapped back to a pixel region.
    """

    rows: int
    cols: int
    token_index: np.ndarray  # int32 [rows, cols] -> row in the embedding matrix

    @property
    def n_patches(self) -> int:
        return self.rows * self.cols

    @staticmethod
    def contiguous(rows: int, cols: int, offset: int = 0) -> PatchGrid:
        """Grid whose patches occupy ``offset .. offset + rows*cols`` in order."""
        idx = (np.arange(rows * cols, dtype=np.int32) + offset).reshape(rows, cols)
        return PatchGrid(rows=rows, cols=cols, token_index=idx)


@dataclass
class PageEncoding:
    """Full multi-vector encoding of one page, straight from the model."""

    ref: PageRef
    embeddings: np.ndarray  # float32 [n_tokens, dim], L2-normalised rows
    grid: PatchGrid
    image_size: tuple[int, int]  # (width, height) of the encoded page image
    text_token_index: np.ndarray = field(  # non-image tokens (kept verbatim)
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def n_tokens(self) -> int:
        return int(self.embeddings.shape[0])

    def patch_embeddings(self) -> np.ndarray:
        """Image-token vectors in row-major grid order."""
        return self.embeddings[self.grid.token_index.reshape(-1)]


@dataclass
class PrunedPage:
    """Page after token pruning: a subset (or merge) of the original vectors."""

    ref: PageRef
    embeddings: np.ndarray  # float32 [n_kept, dim]
    kept_token_index: np.ndarray  # int32 [n_kept] -> original token positions
    keep_mask: np.ndarray  # bool [rows, cols] over the patch grid
    grid: PatchGrid
    n_tokens_before: int
    saliency: np.ndarray | None = None  # float32 [rows, cols], for visualisation
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_kept(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def keep_ratio(self) -> float:
        return self.n_kept / max(1, self.n_tokens_before)


@dataclass
class CompressedPage:
    """Index-ready page: bit-packed vectors plus the bytes they cost."""

    ref: PageRef
    codes: np.ndarray  # uint8 [n_vectors, ceil(dim/8)] packed sign bits
    dim: int
    n_tokens_before: int
    n_tokens_after: int
    scale: np.ndarray | None = None  # optional per-vector norm (asymmetric rerank)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vectors(self) -> int:
        return int(self.codes.shape[0])

    @property
    def nbytes(self) -> int:
        n = int(self.codes.nbytes)
        if self.scale is not None:
            n += int(self.scale.nbytes)
        return n

    def raw_nbytes(self, dtype_size: int = 4) -> int:
        """Bytes the *uncompressed* page would have taken in the index."""
        return self.n_tokens_before * self.dim * dtype_size


@dataclass
class SearchHit:
    ref: PageRef
    score: float
    rank: int
    stage: str = "final"  # "prefilter" | "rerank" | "final"


@dataclass
class SearchResult:
    query: str
    hits: list[SearchHit]
    latency_ms: float
    candidates_scored: int = 0
    reranked: int = 0
