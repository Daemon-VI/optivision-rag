"""Retrieval quality metrics.

Everything here takes ranked page ids and a set of relevant page ids, so the
same functions score the numpy index, Qdrant, and the uncompressed baseline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & relevant)
    return hits / min(len(relevant), k)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for i, page_id in enumerate(ranked[:k], start=1):
        if page_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG — the standard ViDoRe metric."""
    dcg = sum(
        1.0 / math.log2(i + 1) for i, pid in enumerate(ranked[:k], start=1) if pid in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def rank_correlation(a: Sequence[str], b: Sequence[str]) -> float:
    """Kendall tau-b between two rankings over their shared items.

    This is the metric that isolates quantization damage: it asks whether the
    compressed index *orders* pages the way the float baseline did, independent
    of whether either ordering is correct.
    """
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return 1.0
    pos_a = {x: i for i, x in enumerate(a)}
    pos_b = {x: i for i, x in enumerate(b)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            sign = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def evaluate(
    run: dict[str, Sequence[str]],
    qrels: dict[str, set[str]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Aggregate metrics over a full query run.

    Args:
        run: query id -> ranked list of page ids.
        qrels: query id -> set of relevant page ids.
    """
    if not qrels:
        return {}
    out: dict[str, float] = {}
    n = 0
    sums: dict[str, float] = {}
    for qid, relevant in qrels.items():
        ranked = list(run.get(qid, []))
        n += 1
        for k in ks:
            sums[f"ndcg@{k}"] = sums.get(f"ndcg@{k}", 0.0) + ndcg_at_k(ranked, relevant, k)
            sums[f"recall@{k}"] = sums.get(f"recall@{k}", 0.0) + recall_at_k(ranked, relevant, k)
            sums[f"hit@{k}"] = sums.get(f"hit@{k}", 0.0) + hit_at_k(ranked, relevant, k)
        sums["mrr@10"] = sums.get("mrr@10", 0.0) + mrr_at_k(ranked, relevant, 10)
    for key, total in sums.items():
        out[key] = total / n
    out["n_queries"] = float(n)
    return out


def storage_summary(index_stats: dict) -> dict[str, float]:
    """Human-facing storage numbers derived from an index's own accounting."""
    n_pages = max(1, int(index_stats.get("n_pages", 0)))
    index_bytes = float(index_stats.get("index_bytes", 0))
    raw_bytes = float(index_stats.get("raw_bytes", 0))
    return {
        "index_mb": index_bytes / 1e6,
        "raw_mb": raw_bytes / 1e6,
        "kb_per_page": index_bytes / n_pages / 1e3,
        "raw_kb_per_page": raw_bytes / n_pages / 1e3,
        "compression_ratio": raw_bytes / index_bytes if index_bytes else 0.0,
        "gb_per_million_pages": index_bytes / n_pages * 1e6 / 1e9,
    }
