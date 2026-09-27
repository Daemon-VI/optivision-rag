"""Retrieval quality from score matrices, against labels or against the baseline.

Two ways to say what "relevant" means, and they answer different questions:

``labels``   the dataset's own qrels. Retention = metric(compressed) / metric(float):
             "is the compressed index still as *right* as the original?"

``baseline`` the uncompressed index's own top result(s) are treated as the
             relevant set. Needs no labels at all, so it works on any corpus a
             user brings, and it is stricter: it also counts changes of mind
             that happen to land on another correct page.

Every metric is computed per query, so retention can carry a paired bootstrap
confidence interval, and every ranking breaks ties by document index so that a
codec producing exact ties is neither rewarded nor punished by the sort.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .scoring import DEFAULT_MAX_BLOCK_BYTES, maxsim_matrix, rank

DEFAULT_METRICS = ("ndcg@5", "ndcg@10", "recall@5", "recall@10", "mrr@10")


def parse_metric(name: str) -> tuple[str, int]:
    kind, _, k = name.lower().partition("@")
    if kind not in {"ndcg", "recall", "mrr", "hit"} or not k.isdigit() or int(k) < 1:
        raise ValueError(f"unsupported metric {name!r}; use ndcg@k, recall@k, mrr@k or hit@k")
    return kind, int(k)


def relevant_from_qrels(
    qrels: dict[str, Any], query_ids: Sequence[str], doc_ids: Sequence[str]
) -> list[np.ndarray]:
    """qid -> iterable of relevant doc ids, as index arrays aligned with ``query_ids``."""
    pos = {d: i for i, d in enumerate(doc_ids)}
    return [
        np.array(sorted(pos[d] for d in qrels.get(q, ()) if d in pos), dtype=np.int64) for q in query_ids
    ]


def relevant_from_baseline(baseline_scores: np.ndarray, depth: int = 1) -> list[np.ndarray]:
    """The baseline's own top-``depth`` documents per query (label-free reference)."""
    top = rank(baseline_scores, depth)
    return [np.sort(row) for row in np.atleast_2d(top)]


def per_query_metrics(
    scores: np.ndarray, relevant: Sequence[np.ndarray], metrics: Sequence[str] = DEFAULT_METRICS
) -> dict[str, np.ndarray]:
    """One value per query and metric. Queries with no relevant document get NaN."""
    parsed = [(m, *parse_metric(m)) for m in metrics]
    kmax = max(k for _, _, k in parsed)
    top = np.atleast_2d(rank(scores, kmax))
    n_q = top.shape[0]
    discounts = 1.0 / np.log2(np.arange(2, kmax + 2))
    out = {m: np.full(n_q, np.nan) for m, _, _ in parsed}
    for qi in range(n_q):
        rel = relevant[qi]
        if rel.size == 0:
            continue
        hits = np.isin(top[qi], rel)
        first = int(np.argmax(hits)) if hits.any() else -1
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
                out[name][qi] = 1.0 / (first + 1) if 0 <= first < k else 0.0
    return out


def mean_metrics(per_query: dict[str, np.ndarray]) -> dict[str, float]:
    return {m: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for m, v in per_query.items()}


def retention(
    compressed: np.ndarray, baseline: np.ndarray, n_boot: int = 1000, seed: int = 0, ci: float = 0.95
) -> tuple[float, float, float]:
    """Ratio of means with a paired bootstrap interval over queries."""
    ok = np.isfinite(compressed) & np.isfinite(baseline)
    c, b = compressed[ok], baseline[ok]
    if c.size == 0 or b.mean() == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(c.mean() / b.mean())
    if n_boot <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c.size, size=(n_boot, c.size))
    bm = b[idx].mean(axis=1)
    ratios = np.where(bm > 0, c[idx].mean(axis=1) / np.where(bm > 0, bm, 1.0), np.nan)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.nanquantile(ratios, [alpha, 1.0 - alpha])
    return point, float(lo), float(hi)


def split_queries(n: int, calibration_fraction: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint calibration / held-out query indices (both sorted)."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    perm = np.random.default_rng(seed).permutation(n)
    cut = round(n * calibration_fraction)
    cut = min(max(cut, 1), n - 1) if n > 1 else n
    return np.sort(perm[:cut]), np.sort(perm[cut:])


@dataclass
class Measurement:
    """One pipeline, measured on one corpus and query set."""

    label: str
    pipeline: dict[str, Any]
    per_query: dict[str, np.ndarray]
    report: dict[str, Any]
    compress_seconds: float
    score_seconds: float
    n_queries: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def means(self) -> dict[str, float]:
        return mean_metrics(self.per_query)

    @property
    def bytes_per_doc(self) -> float:
        return float(self.report["bytes_per_doc"])

    @property
    def query_ms(self) -> float:
        """Exact brute-force scoring time per query over the whole corpus, this machine."""
        return 1000.0 * self.score_seconds / max(1, self.n_queries)

    def summary(self) -> dict[str, Any]:
        out = {
            "label": self.label,
            **{k: self.report[k] for k in ("vectors_per_doc", "dim", "bits_per_dim", "bytes_per_doc",
                                           "compression_vs_float32", "vector_reduction")},
            **self.means,
            "compress_seconds": self.compress_seconds,
            "query_ms": self.query_ms,
        }
        out.update(self.extra)
        return out


def measure(
    pipeline: Any,
    corpus: Any,
    queries: Any,
    relevant: Sequence[np.ndarray],
    metrics: Sequence[str] = DEFAULT_METRICS,
    label: str | None = None,
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
    fit: bool | None = None,
) -> tuple[Measurement, np.ndarray]:
    """Compress ``corpus`` with ``pipeline``, score every query exactly, evaluate.

    Returns the measurement and the raw score matrix (so callers can reuse it,
    e.g. as the label-free reference when this is the float baseline).
    """
    t0 = time.perf_counter()
    compressed = pipeline.compress(corpus, fit=fit)
    t1 = time.perf_counter()
    scores = maxsim_matrix(pipeline.transform_queries(queries), compressed, max_block_bytes=max_block_bytes)
    t2 = time.perf_counter()
    m = Measurement(
        label=label or pipeline.label(),
        pipeline=pipeline.to_dict(),
        per_query=per_query_metrics(scores, relevant, metrics),
        report=compressed.report(),
        compress_seconds=t1 - t0,
        score_seconds=t2 - t1,
        n_queries=len(queries),
    )
    return m, scores
