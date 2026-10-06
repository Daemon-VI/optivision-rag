"""Per-query metrics: the library's own, plus the same metrics from a ranking.

``per_query_metrics`` computes metrics from a dense score matrix. A retriever that returns
ranked document indices (the top ``k`` per query, best-ranked at index 0) needs the same metrics
without materialising scores for every document, so :func:`metrics_from_ranking` reproduces
``per_query_metrics`` from a top-k index array with the same arithmetic: for any score
matrix ``s``, ``metrics_from_ranking(rank(s, kmax), relevant, metrics)`` equals
``per_query_metrics(s, relevant, metrics)`` exactly, NaN positions included.

Binary relevance. A query without a relevant document gets NaN for every metric; removing
such queries is the caller's job and must be done for every candidate and the baseline
together, never per candidate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from optivision.evaluation import DEFAULT_METRICS, mean_metrics, parse_metric, per_query_metrics

__all__ = ["DEFAULT_METRICS", "mean_metrics", "metrics_from_ranking", "parse_metric",
           "per_query_metrics"]


def metrics_from_ranking(indices: np.ndarray, relevant: Sequence[np.ndarray],
                         metrics: Sequence[str] = DEFAULT_METRICS) -> dict[str, np.ndarray]:
    """One value per query and metric from ranked document indices.

    ``indices`` is an integer array ``[n_queries, k]`` of document indices, best-ranked at index 0,
    padded at the end of a row with ``-1`` when fewer than ``k`` documents exist. It needs at
    least as many columns as the largest metric cut-off, so a ranking that was cut short is
    refused instead of silently scored on fewer documents. A pad that is followed by a
    document, a value below -1 and a document listed twice within the cut-off are errors.
    ``relevant`` holds one array of relevant document indices per query.

    Semantics are those of ``per_query_metrics``: NaN for a query with no relevant document;
    nDCG with ideal DCG over ``min(#relevant, k)`` positions; recall over
    ``min(#relevant, k)``; hit as any relevant document within ``k``; MRR as ``1 / rank`` of
    the highest-ranked relevant document within ``k``, else 0.
    """
    parsed = [(m, *parse_metric(m)) for m in metrics]
    kmax = max(k for _, _, k in parsed)
    top = np.atleast_2d(np.asarray(indices))
    if top.dtype.kind not in "iu":
        raise TypeError(f"indices must be an integer array, got dtype {top.dtype}")
    if top.ndim != 2:
        raise ValueError(f"indices must be [n_queries, k], got shape {top.shape}")
    n_q = top.shape[0]
    if len(relevant) != n_q:
        raise ValueError(f"relevant has {len(relevant)} entries for {n_q} queries")
    if top.shape[1] < kmax:
        raise ValueError(f"indices has {top.shape[1]} columns but the largest cut-off is "
                         f"{kmax}; rank at least {kmax} documents (pad with -1 if the corpus "
                         "is smaller)")
    top = top[:, :kmax].astype(np.int64, copy=False)
    if (top < -1).any():
        raise ValueError("indices must be document indices or -1 pads")
    is_doc = top >= 0
    if (is_doc[:, 1:] & ~is_doc[:, :-1]).any():
        raise ValueError("a -1 pad is followed by a document; pads must end each row")
    ordered = np.sort(top, axis=1)
    if ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any():
        raise ValueError("a document appears twice within the cut-off of one ranking")
    n_docs = is_doc.sum(axis=1)

    discounts = 1.0 / np.log2(np.arange(2, kmax + 2))
    out = {m: np.full(n_q, np.nan) for m, _, _ in parsed}
    for qi in range(n_q):
        rel = np.asarray(relevant[qi])
        if rel.size == 0:
            continue
        hits = np.isin(top[qi, : n_docs[qi]], rel)
        top_hit = int(np.argmax(hits)) if hits.any() else -1
        for name, kind, k in parsed:
            h = hits[:k]
            if kind == "ndcg":
                ideal = discounts[: min(rel.size, k)].sum()
                out[name][qi] = float((h * discounts[: h.size]).sum() / ideal)
            elif kind == "recall":
                out[name][qi] = float(h.sum() / min(rel.size, k))
            elif kind == "hit":
                out[name][qi] = float(h.any())
            else:  # mrr
                out[name][qi] = 1.0 / (top_hit + 1) if 0 <= top_hit < k else 0.0
    return out
