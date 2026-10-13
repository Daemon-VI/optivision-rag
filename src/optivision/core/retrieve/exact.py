"""Exact MaxSim retrieval over one store, with a typed result.

``ExactRetriever`` (a float corpus) and ``CompressedRetriever`` (quantized codes, plus the
pipeline that projects queries) wrap :class:`optivision.storage.ExactIndex`. They add validation
and a :class:`~optivision.core.retrieve.types.Ranking`; they add no arithmetic. Every score comes
from ``scoring.maxsim_matrix`` (or ``ExactIndex.rescore``) and every ordering from
``scoring.rank``:

* Scoring is decode-first and asymmetric: the store decodes block by block to float32, float32
  queries are multiplied against it, the maximum is taken over a document's vectors and the
  products are summed over a query's tokens. The score is the raw inner product, so it is exact
  over the *stored representation* (the decoded codes), not over the float vectors the codes
  came from, and codes of different codecs score on different scales. No native compressed-domain
  kernel (Hamming, int8 matrix product) is provided; see ``docs/package/BACKEND_ARCHITECTURE.md``.
* Scores are float32. Ties are broken by ascending store position (``rank`` is stable). An empty
  document scores ``-inf`` and sorts last; an empty query scores 0 against non-empty documents.
* Repeating a call in one process returns bitwise equal results. Results are equal only to float32
  rounding across block budgets, batch sizes and machines.
* Dtype. Scoring is float32. A float16 store is widened (exact); float64 vectors, documents or
  queries are narrowed to float32 by ``decode_rows`` and ``maxsim_matrix``, as they always were
  (not changed here). Integer queries are refused, not cast.

Memory. ``score`` allocates the dense float32 ``[n_queries, n_docs]`` matrix, and ``search`` holds
it while it ranks: the interface does not avoid that matrix, so it is not a scalability claim.
``max_block_bytes`` bounds the decoded document block and the similarity block inside
``maxsim_matrix`` only; it does not bound the query chunk or the dense output. ``rescore``
materialises every query vector as float32 plus a float32 ``[n_query_vectors, m]`` array.

Device. Nothing in this module selects a device or reads the environment. The only device control
in the library is ``OPTIVISION_SCORE_DEVICE``, read inside ``maxsim_matrix`` (so by ``score`` and
``search``); ``rescore`` is CPU NumPy and ignores it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...compose import Pipeline
from ...representation import MultiVectorCorpus
from ...scoring import DEFAULT_MAX_BLOCK_BYTES, rank
from ...storage import ExactIndex
from .checks import (
    as_int,
    as_queries,
    check_finite_store,
    check_k,
    check_positions,
    check_projection,
)
from .types import SCORE_MAXSIM_IP, TIE_INDEX, Ranking


def _ids_array(ids: list[str]) -> np.ndarray:
    out = np.empty(len(ids), dtype=object)
    out[:] = ids
    return out


def ranking_from_positions(
    positions: np.ndarray, scores: np.ndarray, ids: Any, score_kind: str
) -> Ranking:
    """A :class:`Ranking` for full rows of store positions (no padding), ``tie_break='index'``.

    ``positions`` and ``scores`` are what ``rank`` and ``take_along_axis`` return (positions are
    widened to int64 if a platform's index type is narrower; ``scores`` must already be float32);
    ``ids`` lists the store's document ids, indexed by position.
    """
    top = np.asarray(positions, dtype=np.int64)
    names = ids if isinstance(ids, np.ndarray) else _ids_array(list(ids))
    return Ranking(
        ids=[names[row].tolist() for row in top],
        scores=scores,
        indices=top,
        score_kind=score_kind,
        tie_break=TIE_INDEX,
    )


def ranking_from_scores(scores: np.ndarray, ids: Any, k: int, score_kind: str) -> Ranking:
    """The ``k`` best columns of a dense float32 ``[n_queries, n_docs]`` matrix, as a ``Ranking``.

    Ordering is ``scoring.rank``: descending, ties by ascending column, ``-inf`` last. ``ids`` names
    the columns. Useful for any producer of a dense matrix (``k >= 1``; a matrix narrower than
    ``k`` gives a narrower ranking).
    """
    k = check_k(k)
    if not isinstance(scores, np.ndarray) or scores.ndim != 2 or scores.dtype != np.float32:
        raise TypeError("scores must be a 2-D float32 numpy array [n_queries, n_docs]")
    names = ids if isinstance(ids, np.ndarray) else _ids_array(list(ids))
    if len(names) != scores.shape[1]:
        raise ValueError(f"{len(names)} ids for {scores.shape[1]} score columns")
    top = rank(scores, k)
    return ranking_from_positions(top, np.take_along_axis(scores, top, axis=1), names, score_kind)


class ExactRetriever:
    """Exact MaxSim over a float corpus: a ``Retriever``, a ``DenseScorer`` and a ``Reranker``.

    Args:
        corpus: any store with ``offsets``, ``__len__`` and ``decode_rows(lo, hi)`` (a
            ``MultiVectorCorpus``); its ``ids`` name the documents (``"0"``, ``"1"``, ... when
            it has none).
        max_block_bytes: budget for the live similarity block (see the module docstring).

    ``index`` is the :class:`~optivision.storage.ExactIndex` that does the work; ``ids`` are the
    document ids in store order. Queries are validated (finite values, matching dimension) and
    otherwise passed through unchanged: vectors are neither normalised nor projected here.

    Documents are validated once, at construction: a store with a floating-point ``vectors``
    array (a ``MultiVectorCorpus``) is scanned in blocks, in its own dtype and without a copy,
    and a NaN or an infinite component raises ``ValueError`` that names the first document.
    The scan is not repeated by ``score``, ``search`` or ``rescore``. A store of integer codes
    decodes to finite values and is not scanned.
    """

    score_kind = SCORE_MAXSIM_IP

    def __init__(self, corpus: Any, max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES) -> None:
        self._setup(corpus, None, max_block_bytes)

    def _setup(self, store: Any, pipeline: Pipeline | None, max_block_bytes: int) -> None:
        missing = [a for a in ("offsets", "decode_rows") if not hasattr(store, a)]
        if missing or not hasattr(store, "__len__"):
            raise TypeError(
                f"{type(store).__name__} is not a vector store (needs offsets, decode_rows and len)"
            )
        if pipeline is not None and not isinstance(pipeline, Pipeline):
            raise TypeError(
                f"pipeline must be an optivision Pipeline or None, got {type(pipeline).__name__}"
            )
        max_block_bytes = as_int(max_block_bytes, "max_block_bytes")
        if max_block_bytes < 1:
            raise ValueError(f"max_block_bytes must be >= 1, got {max_block_bytes}")
        n_docs = len(store)
        ids = getattr(store, "ids", None)
        self.ids: list[str] = [str(i) for i in range(n_docs)] if ids is None else list(ids)
        if len(self.ids) != n_docs:
            raise ValueError(f"the store has {n_docs} documents but {len(self.ids)} ids")
        if not all(isinstance(i, str) for i in self.ids):
            raise TypeError("document ids must be strings")
        self._ids = _ids_array(self.ids)
        check_finite_store(store, self.ids)
        self.index = ExactIndex(store, pipeline, max_block_bytes)
        dim = getattr(store, "dim", getattr(store, "dimension", None))
        self._store_dim = None if dim is None else int(dim)
        self._query_dim = None if pipeline is not None else self._store_dim

    # ------------------------------------------------------------ shape

    @property
    def n_docs(self) -> int:
        return len(self._ids)

    @property
    def store(self) -> Any:
        return self.index.store

    @property
    def pipeline(self) -> Pipeline | None:
        """The pipeline whose dimension stages project queries (``None``: queries used as given)."""
        return self.index.pipeline

    # ------------------------------------------------------------- work

    def _prepare(self, queries: Any) -> MultiVectorCorpus:
        q = as_queries(queries, dim=self._query_dim)
        if self.pipeline is not None:
            check_projection(q, self.pipeline, self._store_dim)
        return q

    def score(self, queries: MultiVectorCorpus) -> np.ndarray:
        """Dense float32 ``[n_queries, n_docs]`` MaxSim scores (``ExactIndex.score``)."""
        return self.index.score(self._prepare(queries))

    def search(self, queries: MultiVectorCorpus, k: int = 10) -> Ranking:
        """The ``k`` best documents per query (``ExactIndex.search``); ``k >= 1``.

        A corpus with fewer than ``k`` documents gives a narrower ranking.
        """
        k = check_k(k)
        top, scores = self.index.search(self._prepare(queries), k)
        return ranking_from_positions(top, scores, self._ids, self.score_kind)

    def rescore(self, queries: MultiVectorCorpus, candidates: np.ndarray) -> np.ndarray:
        """Exact scores of each query against its own candidates (``ExactIndex.rescore``).

        ``candidates``: integer array ``[n_queries, m]`` of positions in ``[0, n_docs)``.
        Floats, ``-1`` and out-of-range positions are rejected (``TypeError``, ``IndexError``).
        Column ``j`` of the float32 result is candidate column ``j``, duplicates included.

        One corner differs from ``score``: a query without tokens scores 0 against every candidate
        here, a document without vectors included, where ``score`` gives that pair ``-inf``. That is
        ``ExactIndex.rescore`` as it is (reported, not changed), and ``TieredRetriever`` and
        ``TwoTierRetriever`` inherit it.
        """
        q = self._prepare(queries)
        cand = check_positions(candidates, len(q), self.n_docs)
        return self.index.rescore(q, cand)


class CompressedRetriever(ExactRetriever):
    """The same over a :class:`~optivision.compose.CompressedCorpus`, with query projection.

    Args:
        store: the compressed documents (decoded block by block at scoring time).
        pipeline: the *fitted* pipeline that produced ``store``, or ``None``. Its dimension stages
            project the queries exactly as ``ExactIndex`` does (``Pipeline.transform_queries``,
            which also rejects non-finite queries); token stages and the quantizer never touch
            queries. With ``None`` the queries are used as given, so they must already have the
            store's dimension. With a pipeline, queries of another dimension raise
            ``ValueError`` that names the queries and both dimensions: they are compared with
            the width the pipeline's first dimension stage was fitted on when its fitted state
            shows it, else one projected query vector is compared with the store's dimension.
        max_block_bytes: budget for the live similarity block.

    A float store given here (a ``MultiVectorCorpus``) is scanned once at construction for NaN
    and infinity, as in :class:`ExactRetriever`; a store of integer codes is not.

    Scores are inner products against the *decoded* codes, so they are exact over the stored
    representation and on the scale of that codec.
    """

    def __init__(
        self,
        store: Any,
        pipeline: Pipeline | None = None,
        max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
    ) -> None:
        self._setup(store, pipeline, max_block_bytes)
