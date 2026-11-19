"""Exact search over an index's segments, and ranking with candidate layouts before any is stored
(docs/easy/CONTRACT.md; FINAL_DESIGN 5.7 and 9).

``Searcher(store)``
    Exact MaxSim top-k over every segment of a :class:`~optivision.easy._store.Store`. Each
    segment is scored by its own ``CompressedRetriever(segment, pipeline,
    max_block_bytes=32 MiB)``; the per-segment top-k lists are then merged exactly. A
    segment's top k holds every page of that segment that belongs to the overall top k, so
    the merge loses nothing, and ties are broken by store position, as ``scoring.rank`` breaks
    them inside one retriever. The result is therefore the result of one
    ``CompressedRetriever`` over the whole corpus compressed in one go: the same pages in the
    same order with the same scores. The scores are the same arithmetic on the same decoded
    vectors; the retriever's own documentation promises equality across block shapes only to
    float32 rounding (the tests allow 1e-6), and on the machine this was written on (NumPy 2.4,
    OpenBLAS 0.3.31) they were bitwise equal for every ladder layout.

``rank_float(corpus, queries, k, pipeline=None)``
    The build's practice check needs rankings from layouts it has not stored. This
    compresses a float16 corpus segment by segment with a fitted candidate pipeline (or
    scores the float vectors themselves when ``pipeline`` is None, the full-precision
    reference) and ranks exactly the way ``Searcher`` does, so a candidate scored here and
    then stored ranks the same. Only one segment's codes exist at a time.

Results are ``list[list[(page id, score)]]``, one list per query, best first; ``rank_all`` and
``as_rankings`` give ``{query id: [page ids]}`` for the practice check.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..compose import Pipeline
from ..core.retrieve.checks import as_int, as_queries, check_k
from ..core.retrieve.exact import CompressedRetriever, ExactRetriever
from ..representation import MultiVectorCorpus
from ..scoring import rank
from ._store import SEGMENT_PAGES, Store, _require_fitted, _segments_of, settle_many

__all__ = [
    "SEARCH_BLOCK_BYTES",
    "Searcher",
    "as_rankings",
    "merge_topk",
    "rank_float",
    "rank_float_many",
]

#: similarity-block budget of every per-segment retriever (FINAL_DESIGN 9, ``_index``)
SEARCH_BLOCK_BYTES = 32 * 1024 * 1024

Results = list[list[tuple[str, float]]]


def _check_block(max_block_bytes: Any) -> int:
    value = as_int(max_block_bytes, "max_block_bytes")
    if value < 1:
        raise ValueError(f"max_block_bytes must be >= 1, got {value}")
    return value


def merge_topk(
    best: tuple[np.ndarray, np.ndarray] | None, part: tuple[np.ndarray, np.ndarray], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two top lists: ``(positions int64 [n_queries, a], scores float32 [n_queries, a])``.

    The order is descending score, ties by ascending position, which is the order
    ``scoring.rank`` gives over one dense score matrix. ``-inf`` (a page without vectors)
    sorts last. Keeps at most ``k`` columns.
    """
    if best is None:
        positions, scores = part
    else:
        positions = np.concatenate([best[0], part[0]], axis=1)
        scores = np.concatenate([best[1], part[1]], axis=1)
    if positions.shape[1] == 0:
        return positions, scores
    order = np.lexsort((positions, -scores), axis=-1)[:, :k]
    return np.take_along_axis(positions, order, axis=1), np.take_along_axis(scores, order, axis=1)


def _segment_top(
    retriever: ExactRetriever, queries: MultiVectorCorpus, k: int, base: int
) -> tuple[np.ndarray, np.ndarray]:
    """One segment's top ``k``: positions shifted by the segment's store offset ``base``."""
    scores = retriever.score(queries)  # dense float32 [n_queries, n_pages_in_segment]
    top = rank(scores, min(k, scores.shape[1]))
    return np.asarray(top, dtype=np.int64) + base, np.take_along_axis(scores, top, axis=1)


def _to_results(positions: np.ndarray, scores: np.ndarray, ids: Sequence[str]) -> Results:
    return [[(ids[int(p)], float(s)) for p, s in zip(prow, srow, strict=True)]
            for prow, srow in zip(positions, scores, strict=True)]


def as_rankings(results: Results, queries: MultiVectorCorpus) -> dict[str, list[str]]:
    """``{query id: [page ids, best first]}`` from search results (what the practice check takes)."""
    q = as_queries(queries)
    if len(q) != len(results):
        raise ValueError(f"{len(results)} result lists for {len(q)} queries")
    return {qid: [pid for pid, _ in row] for qid, row in zip(q.ids, results, strict=True)}


class Searcher:
    """Exact top-k over every segment of ``store`` (see the module docstring).

    It holds no memory map of its own: each call takes ``store.segments()`` (the store's
    mapped codes, staged segments of a writer included), so ``store.close()`` is all it takes
    to release the files.
    """

    def __init__(self, store: Store, *, max_block_bytes: int = SEARCH_BLOCK_BYTES) -> None:
        if not isinstance(store, Store):
            raise TypeError(f"expected a Store, got {type(store).__name__}")
        self.store = store
        self.max_block_bytes = _check_block(max_block_bytes)

    @property
    def n_pages(self) -> int:
        return self.store.n_pages

    def _retrievers(self) -> Iterable[tuple[CompressedRetriever, int]]:
        pipeline = self.store.pipeline()
        for seg, base in zip(self.store.segments(), self.store.segment_bases(), strict=True):
            yield CompressedRetriever(seg, pipeline, max_block_bytes=self.max_block_bytes), base

    def search_positions(self, queries: Any, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Store positions int64 ``[n_queries, min(k, n_pages)]`` and their float32 scores."""
        k = check_k(k)
        q = as_queries(queries)
        best: tuple[np.ndarray, np.ndarray] | None = None
        for retriever, base in self._retrievers():
            best = merge_topk(best, _segment_top(retriever, q, k, base), k)
        if best is None:
            return np.zeros((len(q), 0), dtype=np.int64), np.zeros((len(q), 0), dtype=np.float32)
        return best

    def search(self, queries: Any, k: int = 5) -> Results:
        """The ``k`` best pages per query: ``[[(page id, score), ...], ...]``, best first.

        ``queries`` is a ``MultiVectorCorpus`` of query vectors (or a list of ``[n_i, d]``
        arrays), in the space the documents were encoded in; the pipeline projects them if it
        has a projection. Fewer than ``k`` pages give shorter lists.
        """
        positions, scores = self.search_positions(queries, k)
        return _to_results(positions, scores, self.store._all_ids())

    def rank_all(self, queries: Any, k: int = 10) -> dict[str, list[str]]:
        """``{query id: [page ids, best first]}`` for the practice check (top ``k``)."""
        q = as_queries(queries)
        return as_rankings(self.search(q, k), q)

    def score(self, queries: Any) -> np.ndarray:
        """Dense float32 ``[n_queries, n_pages]`` scores in store order (segment by segment)."""
        q = as_queries(queries)
        parts = [retriever.score(q) for retriever, _ in self._retrievers()]
        if not parts:
            return np.zeros((len(q), 0), dtype=np.float32)
        return np.concatenate(parts, axis=1)


def rank_float_many(
    corpus: MultiVectorCorpus | Iterable[MultiVectorCorpus],
    queries: Any,
    k: int,
    pipelines: Sequence[Pipeline | None],
    *,
    segment_pages: int = SEGMENT_PAGES,
    max_block_bytes: int = SEARCH_BLOCK_BYTES,
) -> list[Results]:
    """:func:`rank_float` for several candidate layouts in one pass over the corpus.

    ``pipelines`` holds fitted pipelines, or None for the float reference. Candidates that
    share their merge (same token and dimension stages and fitted state) share one transform
    per segment (``settle_many``). Returns one result list per pipeline, in order.
    """
    k = check_k(k)
    block = _check_block(max_block_bytes)
    q = as_queries(queries)
    pipelines = list(pipelines)
    compressed = [i for i, p in enumerate(pipelines) if p is not None]
    for i in compressed:
        _require_fitted(pipelines[i])
    best: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(pipelines)
    ids: list[str] = []
    base = 0
    for seg in _segments_of(corpus, segment_pages):
        n = len(seg)
        if n == 0:
            continue
        if len(compressed) < len(pipelines):
            top = _segment_top(ExactRetriever(seg, max_block_bytes=block), q, k, base)
            for i, p in enumerate(pipelines):
                if p is None:
                    best[i] = merge_topk(best[i], top, k)
        if compressed:
            settled = next(settle_many([seg], [pipelines[i] for i in compressed]))
            for i, codes in zip(compressed, settled, strict=True):
                retriever = CompressedRetriever(codes, pipelines[i], max_block_bytes=block)
                best[i] = merge_topk(best[i], _segment_top(retriever, q, k, base), k)
            del settled
        ids.extend(seg.ids)
        base += n
    out = []
    for entry in best:
        if entry is None:
            out.append([[] for _ in range(len(q))])
        else:
            out.append(_to_results(entry[0], entry[1], ids))
    return out


def rank_float(
    corpus: MultiVectorCorpus | Iterable[MultiVectorCorpus],
    queries: Any,
    k: int,
    pipeline: Pipeline | None = None,
    *,
    segment_pages: int = SEGMENT_PAGES,
    max_block_bytes: int = SEARCH_BLOCK_BYTES,
) -> Results:
    """Rank ``corpus`` for ``queries`` with a candidate layout, without storing it.

    ``corpus`` is the float16 copy: one corpus (memory-mapped, say; it is cut into
    ``segment_pages``-page views) or an iterable of segments (``store.float16_segments()``,
    the build's shards). With a fitted ``pipeline`` each segment is compressed with
    ``compress(segment, fit=False)`` and scored by a ``CompressedRetriever``; with None the
    float vectors are scored as they are. The ranking equals ``Searcher`` over a store
    holding the same pipeline's codes.
    """
    return rank_float_many(corpus, queries, k, [pipeline], segment_pages=segment_pages,
                           max_block_bytes=max_block_bytes)[0]
