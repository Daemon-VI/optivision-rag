"""The value type and the protocols of the retrieval seam: queries in, ranked ids out.

``Ranking``
    What a retriever returns: ranked document ids per query with their scores, and (for
    in-process retrievers) the store positions. It records the *kind* of score and the tie-break
    convention it was produced under, because scores of different kinds are not on one scale and
    must not be compared (see :func:`comparable`).

``Retriever``, ``DenseScorer``, ``Reranker``
    Three small capabilities. A retriever answers ``search``; a dense scorer also exposes the
    full ``[n_queries, n_docs]`` matrix; a reranker scores given candidates. ``runtime_checkable``
    checks that the attributes exist, nothing more: behaviour is pinned by the tests, not by
    ``isinstance``.

Nothing here computes a score. See ``docs/package/BACKEND_ARCHITECTURE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...representation import MultiVectorCorpus
from .checks import check_k

#: raw inner-product MaxSim over float32 queries and decoded document vectors
SCORE_MAXSIM_IP = "maxsim_ip"
#: MaxSim of cosine similarities (both sides L2-normalised by the engine)
SCORE_MAXSIM_COSINE = "maxsim_cosine"
#: cold-tier scores for a shortlist, hot-tier scores shifted below them for the rest
SCORE_TIERED = "tiered"

#: ties among equal scores are broken by ascending store position
TIE_INDEX = "index"
#: the producer does not say how ties are broken (an engine decides)
TIE_UNSPECIFIED = "unspecified"

_TIE_BREAKS = (TIE_INDEX, TIE_UNSPECIFIED)


def _checked_scores(scores: Any) -> np.ndarray:
    if not isinstance(scores, np.ndarray) or scores.ndim != 2:
        raise ValueError("scores must be a 2-D numpy array [n_queries, width]")
    if scores.dtype != np.float32:
        raise TypeError(f"scores must be float32, got {scores.dtype} (nothing is converted silently)")
    out = np.array(scores, copy=True)
    out.setflags(write=False)
    return out


def _checked_ids(ids: Any, n_queries: int, width: int) -> list[list[str]]:
    rows = [list(row) for row in ids]
    if len(rows) != n_queries:
        raise ValueError(f"ids has {len(rows)} rows but scores has {n_queries}")
    for q, row in enumerate(rows):
        if len(row) > width:
            raise ValueError(f"ids row {q} has {len(row)} entries but the width is {width}")
        for doc_id in row:
            if not isinstance(doc_id, str):
                raise TypeError(f"ids must be strings, got {type(doc_id).__name__} in row {q}")
    return rows


def _checked_indices(indices: Any, shape: tuple[int, int], lengths: np.ndarray) -> np.ndarray:
    if not isinstance(indices, np.ndarray) or indices.dtype != np.int64:
        dtype = getattr(indices, "dtype", type(indices).__name__)
        raise TypeError(f"indices must be an int64 numpy array or None, got {dtype}")
    if indices.shape != shape:
        raise ValueError(f"indices has shape {indices.shape} but scores has shape {shape}")
    inside = np.arange(shape[1])[None, :] < lengths[:, None]
    if (indices[inside] < 0).any() or (indices[~inside] != -1).any():
        raise ValueError("indices must be >= 0 for every listed document and -1 for padding")
    out = np.array(indices, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False, repr=False)
class Ranking:
    """Ranked documents for a batch of queries, best first.

    ``ids[q]`` lists the documents returned for query ``q``; row ``q`` of ``scores`` (float32,
    ``[n_queries, width]``) carries their scores, and positions beyond ``len(ids[q])`` are padding
    (``-inf``). ``indices`` (int64, same shape) holds each document's position in the producer's
    own store, ``-1`` for padding, and is ``None`` when the producer only knows ids (a backend
    behind an engine). A real empty document also scores ``-inf``; padding is told apart by
    ``len(ids[q])`` and by ``indices == -1``, never by the score.

    ``score_kind`` and ``tie_break`` say what the numbers mean. Scores are comparable only between
    rankings of the same kind and tie convention (:func:`comparable`); even then, scores of two
    different codecs or pipelines are on different scales. ``tie_break == "index"`` means equal
    scores are ordered by ascending store position, and the constructor checks it whenever
    ``indices`` is given.

    Construction validates shapes, dtypes, descending order and the padding convention, rejects
    NaN scores, and keeps private read-only copies, so a ranking cannot change after it was built.
    ``==`` compares every field exactly (arrays bitwise); a ranking is unhashable.
    """

    ids: list[list[str]]
    scores: np.ndarray
    indices: np.ndarray | None = None
    score_kind: str = SCORE_MAXSIM_IP
    tie_break: str = TIE_INDEX

    def __post_init__(self) -> None:
        scores = _checked_scores(self.scores)
        n_queries, width = scores.shape
        ids = _checked_ids(self.ids, n_queries, width)
        lengths = np.array([len(row) for row in ids], dtype=np.int64)
        indices = None
        if self.indices is not None:
            indices = _checked_indices(self.indices, scores.shape, lengths)
        if not isinstance(self.score_kind, str) or not self.score_kind:
            raise ValueError("score_kind must be a non-empty string")
        if self.tie_break not in _TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {_TIE_BREAKS}, got {self.tie_break!r}")
        _check_order(scores, indices, lengths, self.tie_break)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "indices", indices)

    # ------------------------------------------------------------ shape

    @property
    def n_queries(self) -> int:
        return int(self.scores.shape[0])

    @property
    def width(self) -> int:
        """Columns of ``scores``: the depth requested, or less when the store is smaller."""
        return int(self.scores.shape[1])

    def lengths(self) -> np.ndarray:
        """Documents actually returned per query (``len(ids[q])``), int64 ``[n_queries]``."""
        return np.array([len(row) for row in self.ids], dtype=np.int64)

    # ---------------------------------------------------------- helpers

    def top(self, k: int) -> Ranking:
        """The first ``k`` columns as a new ranking (never longer than ``width``; ``k >= 1``)."""
        k = check_k(k)
        w = min(k, self.width)
        return Ranking(
            ids=[row[:w] for row in self.ids],
            scores=self.scores[:, :w],
            indices=None if self.indices is None else self.indices[:, :w],
            score_kind=self.score_kind,
            tie_break=self.tie_break,
        )

    def to_lists(self) -> dict[str, Any]:
        """Plain Python values, padding removed (rows may differ in length).

        Keys ``ids``, ``scores``, ``indices`` (``None`` for id-only rankings), ``score_kind``,
        ``tie_break``. Scores are Python floats holding the float32 values exactly; ``-inf``
        (an empty document) stays ``float("-inf")``, which JSON cannot represent.
        """
        n = self.lengths()
        return {
            "ids": [list(row) for row in self.ids],
            "scores": [self.scores[q, : n[q]].tolist() for q in range(self.n_queries)],
            "indices": None if self.indices is None else [
                self.indices[q, : n[q]].tolist() for q in range(self.n_queries)
            ],
            "score_kind": self.score_kind,
            "tie_break": self.tie_break,
        }

    # --------------------------------------------------------- identity

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ranking):
            return NotImplemented
        if (self.score_kind, self.tie_break) != (other.score_kind, other.tie_break):
            return False
        if self.ids != other.ids or not np.array_equal(self.scores, other.scores):
            return False
        if (self.indices is None) != (other.indices is None):
            return False
        return self.indices is None or bool(np.array_equal(self.indices, other.indices))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"Ranking(n_queries={self.n_queries}, width={self.width}, "
            f"score_kind={self.score_kind!r}, tie_break={self.tie_break!r}, "
            f"indices={'yes' if self.indices is not None else 'no'})"
        )


def _check_order(
    scores: np.ndarray, indices: np.ndarray | None, lengths: np.ndarray, tie_break: str
) -> None:
    """Scores descend over the listed documents; with ``index`` ties, positions ascend."""
    width = scores.shape[1]
    inside = np.arange(width)[None, :] < lengths[:, None]
    if np.isnan(scores[inside]).any():
        raise ValueError("scores contain NaN (a NaN input reached the scorer)")
    if (scores[~inside] != -np.inf).any():
        raise ValueError("padding positions must score -inf")
    if width < 2:
        return
    a, b = scores[:, :-1], scores[:, 1:]
    both = inside[:, 1:]
    if (both & ~(b <= a)).any():
        raise ValueError("scores must be in descending order within each row")
    if tie_break == TIE_INDEX and indices is not None:
        tied = both & (a == b)
        if (tied & ~(indices[:, 1:] > indices[:, :-1])).any():
            raise ValueError("tie_break='index': equal scores must be ordered by ascending position")


@runtime_checkable
class Retriever(Protocol):
    """Queries in, ranked ids out.

    ``search`` takes raw queries (the retriever applies whatever query projection its store
    needs) and a depth ``k >= 1``; a corpus smaller than ``k`` returns a narrower ranking.
    """

    @property
    def n_docs(self) -> int: ...

    def search(self, queries: MultiVectorCorpus, k: int = 10) -> Ranking: ...


@runtime_checkable
class DenseScorer(Protocol):
    """Optional capability: the full score matrix, float32 ``[n_queries, n_docs]``.

    ``-inf`` marks a document that can never be retrieved (it holds no vectors). The matrix grows
    with the corpus; asking for it is not a scalability claim.
    """

    def score(self, queries: MultiVectorCorpus) -> np.ndarray: ...


@runtime_checkable
class Reranker(Protocol):
    """Scores each query against its own candidate documents, float32 ``[n_queries, m]``.

    ``candidates`` is an int64 ``[n_queries, m]`` array of store positions in ``[0, n_docs)``;
    column ``j`` of the result belongs to column ``j`` of ``candidates`` (duplicates included).
    """

    def rescore(self, queries: MultiVectorCorpus, candidates: np.ndarray) -> np.ndarray: ...


def comparable(a: Any, b: Any) -> bool:
    """True when the scores of ``a`` and ``b`` were produced under one definition.

    Works on anything with ``score_kind`` and ``tie_break`` attributes: two :class:`Ranking`
    objects, or two ``BackendInfo`` records. Scores of different kinds (inner-product MaxSim,
    cosine MaxSim, tiered scores) are not on one scale; ranks may be compared across kinds, scores
    may not. Necessary, not sufficient: two codecs or pipelines in the same kind still differ in
    scale (a binary code decodes to a vector of norm ``sqrt(d)``), and a ranking whose
    ``tie_break`` is ``"unspecified"`` has no meaningful order inside a group of equal scores.
    """
    return a.score_kind == b.score_kind and a.tie_break == b.tie_break


def require_comparable(a: Any, b: Any) -> None:
    """Raise ``ValueError`` unless :func:`comparable`."""
    if not comparable(a, b):
        raise ValueError(
            "scores are not comparable: "
            f"({a.score_kind!r}, tie_break={a.tie_break!r}) vs "
            f"({b.score_kind!r}, tie_break={b.tie_break!r})"
        )
