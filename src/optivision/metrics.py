"""Retrieval quality metrics.

Everything here takes ranked page ids and a set of relevant page ids, so the
same functions score the numpy index, Qdrant, and the uncompressed baseline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


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


def rank_correlation(
    a: Sequence[str], b: Sequence[str], pool: Sequence[str] | None = None
) -> float:
    """Kendall tau-b between two rankings, best-first.

    This is the metric that isolates quantization damage: it asks whether the
    compressed index *orders* pages the way the float baseline did, independent
    of whether either ordering is correct.

    ``pool`` is the set of candidates being ranked. Pass the whole corpus and
    the answer is rank agreement over the corpus, which is the quantity worth
    comparing across experiments. Omit it and the pool is the union of the two
    lists, which is the most that truncated hit lists can support.

    An item the pool contains but a list does not rank is placed after
    everything that list *does* rank, tied with the other absentees. That is the
    part :func:`rank_correlation_shared` gets wrong: a page one ranking puts
    first and the other drops entirely is a disagreement, not missing data.
    Two disjoint top-*k* lists score about $-0.71$ here and $1.0$ there.

    Values are comparable across corpora only when the pool is a comparable
    fraction of each. Two top-10 lists over 60 pages and over 500 are not the
    same measurement.
    """
    universe = list(pool) if pool is not None else list(dict.fromkeys([*a, *b]))
    if len(universe) < 2:
        return 1.0

    pos_a = {x: i for i, x in enumerate(a)}
    pos_b = {x: i for i, x in enumerate(b)}
    ra = np.array([pos_a.get(x, len(a)) for x in universe], dtype=np.int64)
    rb = np.array([pos_b.get(x, len(b)) for x in universe], dtype=np.int64)

    # Pairwise sign comparison. Quadratic, but vectorised and only ever run over
    # one query's candidate list, where n is hundreds rather than millions.
    iu = np.triu_indices(len(universe), k=1)
    sa = np.sign(ra[:, None] - ra[None, :])[iu]
    sb = np.sign(rb[:, None] - rb[None, :])[iu]

    agree = sa * sb
    concordant = int(np.count_nonzero(agree > 0))
    discordant = int(np.count_nonzero(agree < 0))
    ties_a = int(np.count_nonzero((sa == 0) & (sb != 0)))
    ties_b = int(np.count_nonzero((sb == 0) & (sa != 0)))

    ranked = concordant + discordant
    denom = math.sqrt((ranked + ties_a) * (ranked + ties_b))
    return (concordant - discordant) / denom if denom else 1.0


def rank_correlation_shared(a: Sequence[str], b: Sequence[str]) -> float:
    """Tau-b over only the ids the two rankings share. **Superseded.**

    This is the statistic reported in the paper (E1's 0.585 and E2's 0.527) and
    it is kept so those numbers stay reproducible, not because it should be
    used. It measures agreement *among the survivors*, so it is blind to the
    damage that matters most -- a page dropping out of the list entirely -- and
    it returns 1.0 when fewer than two ids survive, scoring two rankings that
    share nothing as identical. It is also cutoff-dependent in a way the paper
    did not report: the same E1 run gives 0.402 at top-5, 0.585 at top-10 and
    0.691 at top-20.

    Use :func:`rank_correlation` with an explicit ``pool``.
    """
    b_set = set(b)  # hoisted: rebuilding it per element made this quadratic
    common = [x for x in a if x in b_set]
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
