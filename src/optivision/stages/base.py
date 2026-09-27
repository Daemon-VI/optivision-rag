"""The contract every compression stage implements.

Three kinds of stage, applied in this order by :class:`~optivision.compose.Pipeline`
(token and dimension stages may interleave; a quantizer is always last):

``TokenReducer``
    fewer vectors per document (prune or merge). Queries are untouched.
``DimensionReducer``
    fewer dimensions per vector (projection). Must be applied to queries too,
    or the inner products stop meaning anything.
``Quantizer``
    fewer bits per dimension. Documents only: queries stay float32 and are
    scored asymmetrically against decoded document codes.

Stages that carry fitted state (a PCA basis, a codec's mean) implement
``fit``; everything else is stateless. ``params()`` returns the constructor
arguments, so any pipeline can be written to JSON and rebuilt exactly.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..representation import MultiVectorCorpus

STAGES: dict[str, type[Stage]] = {}


def register(cls: type[Stage]) -> type[Stage]:
    """Class decorator: make a stage constructible from its ``to_dict()``."""
    if cls.name in STAGES and STAGES[cls.name] is not cls:
        raise ValueError(f"duplicate stage name {cls.name!r}")
    STAGES[cls.name] = cls
    return cls


def stage_from_dict(spec: dict[str, Any]) -> Stage:
    spec = dict(spec)
    name = spec.pop("stage")
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}; known: {sorted(STAGES)}")
    return STAGES[name](**spec)


class Stage(abc.ABC):
    kind: ClassVar[str] = "stage"
    name: ClassVar[str] = "stage"

    def fit(self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None) -> Stage:
        return self

    @property
    def fitted(self) -> bool:
        return True

    @abc.abstractmethod
    def params(self) -> dict[str, Any]: ...

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.name, **self.params()}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"


# ------------------------------------------------------------ token stages


@dataclass
class DocView:
    """What a token reducer may look at for one document.

    ``vectors`` are the reducible (non-protected) rows, float32. The aligned
    arrays are restricted to the same rows. ``image`` and ``metadata`` are the
    document's own, for stages that read pixels or grid shape.
    """

    index: int
    vectors: np.ndarray
    positions: np.ndarray | None
    importance: np.ndarray | None
    weights: np.ndarray
    metadata: dict[str, Any]
    image: Any = None


@dataclass
class Reduction:
    """A token reducer's answer for one document.

    ``vectors`` are the output rows. ``labels[i]`` says which output row input
    row ``i`` went into, or ``-1`` if it was dropped. Aligned arrays for the
    output (weights, positions, importance) are derived from ``labels``, unless
    the reducer supplies ``importance`` itself.
    """

    vectors: np.ndarray
    labels: np.ndarray
    importance: np.ndarray | None = None


class TokenReducer(Stage):
    """Fewer vectors per document. Protected rows always pass through verbatim."""

    kind: ClassVar[str] = "tokens"
    #: True if :meth:`reduce` returns its own per-row importance (a saliency map)
    produces_importance: ClassVar[bool] = False

    @abc.abstractmethod
    def reduce(self, doc: DocView) -> Reduction: ...

    def transform(self, corpus: MultiVectorCorpus) -> MultiVectorCorpus:
        dtype = corpus.dtype
        parts: list[np.ndarray] = []
        counts: list[int] = []
        prot_parts: list[np.ndarray] = []
        pos_parts: list[np.ndarray] = []
        w_parts: list[np.ndarray] = []
        imp_parts: list[np.ndarray] = []
        track_pos = corpus.positions is not None
        track_imp = corpus.importance is not None or self.produces_importance

        for i in range(len(corpus)):
            lo, hi = corpus.span(i)
            protected = (
                corpus.protected[lo:hi] if corpus.protected is not None else np.zeros(hi - lo, dtype=bool)
            )
            free = np.flatnonzero(~protected)
            fixed = np.flatnonzero(protected)
            weights = corpus.weights[lo:hi] if corpus.weights is not None else np.ones(hi - lo, np.float32)
            doc = DocView(
                index=i,
                vectors=np.asarray(corpus.vectors[lo:hi][free], dtype=np.float32),
                positions=corpus.positions[lo:hi][free] if track_pos else None,
                importance=corpus.importance[lo:hi][free] if corpus.importance is not None else None,
                weights=weights[free],
                metadata=corpus.metadata[i],
                image=corpus.images[i] if corpus.images is not None else None,
            )
            if free.size:
                red = self.reduce(doc)
                out_vec = np.asarray(red.vectors, dtype=np.float32).reshape(-1, corpus.dimension)
                labels = np.asarray(red.labels, dtype=np.int64)
                if labels.shape != (free.size,):
                    raise ValueError(f"{self.name}: labels must cover every reducible row")
            else:
                out_vec = np.zeros((0, corpus.dimension), np.float32)
                labels = np.zeros(0, np.int64)
                red = Reduction(out_vec, labels)
            k = out_vec.shape[0]
            if labels.size and labels.max() >= k:
                raise ValueError(f"{self.name}: a label points past the output rows")

            kept = labels >= 0
            w_out = np.bincount(labels[kept], weights=doc.weights[kept], minlength=k).astype(np.float32)

            parts.append(out_vec.astype(dtype, copy=False))
            parts.append(np.asarray(corpus.vectors[lo:hi][fixed], dtype=dtype))
            counts.append(k + fixed.size)
            prot_parts.append(np.concatenate([np.zeros(k, bool), np.ones(fixed.size, bool)]))
            w_parts.append(np.concatenate([w_out, weights[fixed]]))
            if track_pos:
                pos_parts.append(np.concatenate(
                    [_mean_positions(doc.positions, labels, doc.weights, k), corpus.positions[lo:hi][fixed]]
                ))
            if track_imp:
                if red.importance is not None:
                    imp_out = np.asarray(red.importance, dtype=np.float32).reshape(k)
                elif doc.importance is not None and kept.any():
                    # a merged row is as important as its most important member
                    imp_out = np.full(k, -np.inf, np.float32)
                    np.maximum.at(imp_out, labels[kept], doc.importance[kept])
                else:
                    imp_out = np.zeros(k, np.float32)
                imp_fixed = (
                    corpus.importance[lo:hi][fixed] if corpus.importance is not None
                    else np.zeros(fixed.size, np.float32)
                )
                imp_parts.append(np.concatenate([imp_out, imp_fixed]))

        vectors = np.concatenate(parts, axis=0) if parts else np.zeros((0, corpus.dimension), dtype)
        offsets = np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))])
        importance = np.concatenate(imp_parts) if track_imp and imp_parts else None
        return MultiVectorCorpus(
            vectors,
            offsets,
            ids=corpus.ids,
            metadata=corpus.metadata,
            protected=np.concatenate(prot_parts) if prot_parts else None,
            positions=np.concatenate(pos_parts) if track_pos and pos_parts else None,
            weights=np.concatenate(w_parts) if w_parts else None,
            importance=importance,
            images=corpus.images,
            attrs=corpus.attrs,
        )


def _mean_positions(positions: np.ndarray | None, labels: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    out = np.full((k, 2), np.nan, dtype=np.float32)
    if positions is None or k == 0:
        return out
    ok = (labels >= 0) & np.isfinite(positions).all(axis=1)
    if not ok.any():
        return out
    w = weights[ok].astype(np.float64)
    sums = np.zeros((k, 2))
    np.add.at(sums, labels[ok], positions[ok] * w[:, None])
    tot = np.bincount(labels[ok], weights=w, minlength=k)
    has = tot > 0
    out[has] = (sums[has] / tot[has, None]).astype(np.float32)
    return out


def labels_from_clusters(clusters: list[list[int]], n: int) -> np.ndarray:
    labels = np.full(n, -1, dtype=np.int64)
    for k, members in enumerate(clusters):
        labels[np.asarray(members, dtype=np.int64)] = k
    return labels


# -------------------------------------------------------- dimension stages


class DimensionReducer(Stage):
    """Fewer dimensions per vector. Applied to documents *and* queries."""

    kind: ClassVar[str] = "dimension"

    @abc.abstractmethod
    def project(self, vectors: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def output_dim(self, input_dim: int) -> int: ...

    def transform(self, corpus: MultiVectorCorpus) -> MultiVectorCorpus:
        if not self.fitted:
            raise RuntimeError(f"{self.name} must be fit before transform")
        return corpus.with_vectors(self.project(np.asarray(corpus.vectors, dtype=np.float32)))

    def transform_queries(self, queries: MultiVectorCorpus) -> MultiVectorCorpus:
        return self.transform(queries)


# -------------------------------------------------------------- quantizers


class Quantizer(Stage):
    """Fewer bits per dimension. The last stage of a pipeline, documents only."""

    kind: ClassVar[str] = "quantize"

    @abc.abstractmethod
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """float [n, dim] -> uint8 codes [n, code_bytes(dim)]."""

    @abc.abstractmethod
    def decode(self, codes: np.ndarray, dim: int) -> np.ndarray:
        """uint8 codes -> float32 [n, dim] for asymmetric scoring."""

    @abc.abstractmethod
    def code_bytes(self, dim: int) -> int: ...

    def overhead_bytes(self) -> int:
        """Shared, per-index state the codec must store (fitted means, scales)."""
        return 0

    def bits_per_dim(self, dim: int) -> float:
        return 8.0 * self.code_bytes(dim) / max(1, dim)
