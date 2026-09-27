"""Exact late-interaction (MaxSim) scoring over any multi-vector store.

    score(q, d) = sum over query vectors q_i of  max over document vectors d_j  of  q_i . d_j

This is the reference every compressed configuration is judged against, so it is
exact -- no ANN, no candidate pruning -- and it is written once, for every store:
anything with ``offsets`` and ``decode_rows(lo, hi) -> float32`` can be scored,
whether it holds float vectors (:class:`~optivision.representation.MultiVectorCorpus`)
or quantized codes that decode block by block.

Memory is bounded by ``max_block_bytes`` rather than by the corpus: documents
are decoded in page-aligned blocks and query tokens in chunks, so the similarity
matrix alive at any moment is at most one block by one chunk.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

DEFAULT_MAX_BLOCK_BYTES = 256 * 1024 * 1024


class VectorStore(Protocol):
    offsets: np.ndarray

    def __len__(self) -> int: ...

    def decode_rows(self, lo: int, hi: int) -> np.ndarray: ...


def _page_blocks(offsets: np.ndarray, max_rows: int) -> list[tuple[int, int]]:
    """Split documents into consecutive runs holding at most ``max_rows`` vectors.

    A single document larger than the budget still gets its own block -- a page
    is never split, because MaxSim needs all of a page's rows at once.
    """
    n_docs = int(offsets.size) - 1
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < n_docs:
        limit = offsets[start] + max_rows
        end = int(np.searchsorted(offsets, limit, side="right")) - 1
        end = min(max(end, start + 1), n_docs)
        blocks.append((start, end))
        start = end
    return blocks


def _query_chunks(q_offsets: np.ndarray, max_tokens: int) -> list[tuple[int, int]]:
    return _page_blocks(q_offsets, max_tokens)


def segment_reduce(values: np.ndarray, offsets: np.ndarray, ufunc: Any, axis: int, fill: float) -> np.ndarray:
    """Reduce ``values`` over consecutive segments along ``axis``; empty segments get ``fill``.

    ``np.ufunc.reduceat`` has two traps with empty segments: an index equal to
    the axis length is rejected, and clamping it to the last row -- the obvious
    fix -- silently shortens the *previous* segment by one row. Reducing only
    over the non-empty segments avoids both: their starts are strictly
    increasing, and each one's range runs exactly to the next non-empty start
    because the empty ones in between have no rows.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    counts = np.diff(offsets)
    nonempty = np.flatnonzero(counts > 0)
    shape = list(values.shape)
    shape[axis] = counts.size
    out = np.full(shape, fill, dtype=values.dtype)
    if nonempty.size:
        reduced = ufunc.reduceat(values, offsets[:-1][nonempty], axis=axis)
        index: list[Any] = [slice(None)] * values.ndim
        index[axis] = nonempty
        out[tuple(index)] = reduced
    return out


def maxsim_matrix(
    queries: VectorStore,
    docs: VectorStore,
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
    query_transform: Any = None,
) -> np.ndarray:
    """MaxSim score of every query against every document. float32 ``[n_queries, n_docs]``.

    Args:
        queries: a store of query vectors (one "document" per query).
        docs: a store of document vectors or codes.
        max_block_bytes: budget for the live similarity block.
        query_transform: optional callable applied to each float32 chunk of query
            vectors before scoring (a projection that must hit queries and
            documents alike).

    Empty documents score ``-inf`` (they can never be retrieved); an empty query
    scores 0 against everything.
    """
    n_q, n_d = len(queries), len(docs)
    out = np.zeros((n_q, n_d), dtype=np.float32)
    if n_q == 0 or n_d == 0:
        return out

    q_off = np.asarray(queries.offsets, dtype=np.int64)
    d_off = np.asarray(docs.offsets, dtype=np.int64)
    d_counts = np.diff(d_off)
    empty_docs = d_counts == 0
    total_q_tokens = int(q_off[-1])
    if total_q_tokens == 0:
        out[:, empty_docs] = -np.inf
        return out

    # Budget split: at most ~half the budget for the similarity block, with a
    # floor so tiny budgets still make progress one page at a time.
    q_chunk_tokens = max(1, min(total_q_tokens, max_block_bytes // (4 * 4096)))
    for q0, q1 in _query_chunks(q_off, q_chunk_tokens):
        t0, t1 = int(q_off[q0]), int(q_off[q1])
        if t1 == t0:
            continue  # every query in this chunk is empty: scores stay 0
        qv = np.ascontiguousarray(queries.decode_rows(t0, t1), dtype=np.float32)
        if query_transform is not None:
            qv = np.ascontiguousarray(query_transform(qv), dtype=np.float32)
        local_q = q_off[q0 : q1 + 1] - t0
        rows_per_block = max(1, max_block_bytes // (4 * qv.shape[0] + 4 * qv.shape[1] + 1))
        for d0, d1 in _page_blocks(d_off, rows_per_block):
            lo, hi = int(d_off[d0]), int(d_off[d1])
            if hi == lo:
                out[q0:q1, d0:d1] = -np.inf
                continue
            block = docs.decode_rows(lo, hi)
            sims = qv @ np.ascontiguousarray(block, dtype=np.float32).T  # [tokens, rows]
            local_d = d_off[d0 : d1 + 1] - lo
            # per query token, best document vector within each page
            per_token = segment_reduce(sims, local_d, np.maximum, axis=1, fill=-np.inf)
            del sims
            # per query, sum of its tokens' best matches (empty queries score 0)
            per_query = segment_reduce(per_token, local_q, np.add, axis=0, fill=0.0)
            out[q0:q1, d0:d1] = per_query
            del per_token, block
    out[:, empty_docs] = -np.inf
    return out


def rank(scores: np.ndarray, k: int | None = None) -> np.ndarray:
    """Document indices by descending score, ties broken by index (stable).

    Deterministic tie-breaking matters here: a quantizer that makes two pages
    tie must not look better or worse depending on the sort algorithm.
    """
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores[None, :]
        squeeze = True
    else:
        squeeze = False
    order = np.argsort(-scores, axis=1, kind="stable")
    if k is not None:
        order = order[:, :k]
    return order[0] if squeeze else order


def maxsim_pair(query: np.ndarray, doc: np.ndarray) -> float:
    """MaxSim of one query against one document, the textbook definition."""
    q = np.asarray(query, dtype=np.float32)
    d = np.asarray(doc, dtype=np.float32)
    if d.shape[0] == 0:
        return float("-inf")
    if q.shape[0] == 0:
        return 0.0
    return float((q @ d.T).max(axis=1).sum())
