"""Q3 per-query and per-page features from existing vectors (no re-encoding).

For one ViDoRe store (model x split): recomputes the float32 score matrix and the
score matrices of two representative candidates (positions 27 `binary` and 30
`hierarchical_merge(0.25) > binary` of the default space) with the library's own
calls, checks that their per-query nDCG@5 equals the E1 store exactly (exit 1
otherwise), and writes small per-query / per-page feature arrays to
reports/research/Q3/features/<store>.npz. Score matrices are not kept.

    OPTIVISION_DATA=../optivision-rag-v2/data PYTHONPATH="src;scripts/research" \
        python scripts/research/q3_features.py colpali_docvqa
"""

from __future__ import annotations

import json
import sys
import time

import e1_common as E
import numpy as np

from optivision.calibration import recommended_search_space, score_candidate
from optivision.evaluation import per_query_metrics
from optivision.scoring import maxsim_matrix

REPRESENTATIVE_SCORED = (27, 30)
OUT = E.ROOT / "reports" / "research" / "Q3" / "features"


def full_rank(scores: np.ndarray, j: np.ndarray) -> np.ndarray:
    """1-based rank of document j[q] under rank()'s rule (descending, ties by index)."""
    s_rel = scores[np.arange(len(j)), j]
    higher = (scores > s_rel[:, None]).sum(axis=1)
    idx = np.arange(scores.shape[1])[None, :]
    tied_before = ((scores == s_rel[:, None]) & (idx < j[:, None])).sum(axis=1)
    return 1 + higher + tied_before


def query_features(scores: np.ndarray, j: np.ndarray, L: np.ndarray, unlabelled: np.ndarray) -> dict:
    n = len(j)
    s_rel = scores[np.arange(n), j]
    nonrel = scores.copy()
    nonrel[np.arange(n), j] = -np.inf
    best_nonrel = nonrel.max(axis=1)
    srt = -np.sort(-scores, axis=1)
    top1 = np.argsort(-scores, axis=1, kind="stable")[:, 0]
    return {
        "rank": full_rank(scores, j),
        "s_rel": s_rel, "s1": srt[:, 0], "s2": srt[:, 1],
        "margin_rel": (s_rel - best_nonrel) / L,
        "margin_12": (srt[:, 0] - srt[:, 1]) / L,
        "near_ties": ((nonrel - s_rel[:, None]) / L[:, None] > -0.01).sum(axis=1),
        "top1_unlabelled": unlabelled[top1],
    }


def main(name: str) -> int:
    t0 = time.time()
    s7b = E.load_s7b()
    ds, metric = s7b.dataset(E.dataset_arg(name))
    if metric != "ndcg@5":
        raise SystemExit("q3_features is for the ViDoRe stores (nDCG@5, one labelled page)")
    store = E.Store(E.load_store(name), "labels")
    corpus, queries = ds.corpus, ds.queries
    rel = ds.relevant()
    if any(len(r) != 1 for r in rel):
        raise SystemExit("expected exactly one labelled page per query")
    j = np.array([int(r[0]) for r in rel])
    labelled = set(j.tolist())
    unlabelled = np.array([i not in labelled for i in range(len(corpus))])
    L = np.diff(queries.offsets).astype(float)

    base = maxsim_matrix(queries, corpus)
    b = per_query_metrics(base, rel, [metric])[metric]
    checks = {"baseline_equals_store": bool(np.array_equal(b, store.base))}
    feats = {f"base_{k}": v for k, v in query_features(base, j, L, unlabelled).items()}
    feats["L"] = L
    pipes = [p for steps in recommended_search_space(corpus.dimension).values() for p in steps]
    for pos in REPRESENTATIVE_SCORED:
        sc = score_candidate(pipes[pos], corpus, queries)
        if sc.label != store.labels[pos]:
            raise SystemExit(f"candidate {pos} is {sc.label}, store says {store.labels[pos]}")
        c = per_query_metrics(sc.scores, rel, [metric])[metric]
        checks[f"candidate{pos}_equals_store"] = bool(np.array_equal(c, store.pq[pos]))
        for k, v in query_features(sc.scores, j, L, unlabelled).items():
            feats[f"c{pos}_{k}"] = v
    counts = np.diff(corpus.offsets)
    means = np.stack([corpus.vectors[corpus.offsets[i]:corpus.offsets[i + 1]].astype(np.float64).mean(axis=0)
                      for i in range(len(corpus))])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    sim = means @ means.T
    np.fill_diagonal(sim, -np.inf)
    feats["page_vectors"] = counts
    feats["page_nearest_mean_cosine"] = sim.max(axis=1)
    feats["page_unlabelled"] = unlabelled
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **feats)
    meta = {"store": name, "dataset": ds.name, "n_queries": len(queries), "n_pages": len(corpus),
            "unlabelled_pages": int(unlabelled.sum()), "checks": checks,
            "representative_scored": {p: store.labels[p] for p in REPRESENTATIVE_SCORED},
            "environment": E.environment(), "seconds": time.time() - t0}
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok = all(checks.values())
    print(json.dumps(meta["checks"]), "OK" if ok else "MISMATCH", f"{time.time() - t0:.0f}s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
