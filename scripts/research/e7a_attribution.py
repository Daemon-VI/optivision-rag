"""E7a: attribution of the OptiVision vs SAP disagreement (reports/research/Q7/E7a/PLAN.md).

Per page, the free (unprotected) vectors are reduced to exactly k = ceil(r * n_free)
vectors by four methods; protected vectors are kept unchanged by all of them:
  A  K-means (k-means++, Lloyd, <= 50 iterations), plain mean        seeds 0, 1, 2
  B  Ward (scipy linkage on unit vectors, maxclust k), plain mean
  C  Ward, the library's _pool (mean rescaled to members' mean norm)
  D  random pruning (k original vectors, no merging)                 seeds 0, 1, 2
Scored with exact MaxSim, nDCG@5. Writes reports/research/Q7/E7a/<store>.json|npz.

    OPTIVISION_DATA=../optivision-rag-v2/data PYTHONPATH="src;scripts/research" \
        python scripts/research/e7a_attribution.py colqwen2_docvqa
"""

from __future__ import annotations

import json
import math
import sys
import time

import e1_common as E
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from optivision.evaluation import per_query_metrics, retention
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, rank
from optivision.stages.merge import _pool, _unit, _ward_input

OUT = E.ROOT / "reports" / "research" / "Q7" / "E7a"
RATIOS = {"1/5": 0.2, "1/10": 0.1, "1/20": 0.05}
SEEDS = (0, 1, 2)
MAX_ITER = 50
NB = 1000


def budget(n, r):
    """The library's _budget for ratio r (min_vectors = 1)."""
    return max(min(n, 1), min(min(n, max(1, math.ceil(r * n))), n))


def kmeans(x, k, rng):
    """Lloyd's algorithm with k-means++ init; plain-mean centroids; exactly k non-empty clusters."""
    n = x.shape[0]
    x64 = x.astype(np.float64)
    sq = (x64 * x64).sum(1)
    centers = np.empty((k, x.shape[1]))
    centers[0] = x64[rng.integers(n)]
    d2 = np.maximum(sq + (centers[0] @ centers[0]) - 2 * x64 @ centers[0], 0.0)
    for j in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers[j] = x64[rng.choice(n, p=p)]
        d2 = np.minimum(d2, np.maximum(sq + (centers[j] @ centers[j]) - 2 * x64 @ centers[j], 0.0))
    labels = None

    def fix_empty(lab):
        counts = np.bincount(lab, minlength=k)
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            own = np.maximum(sq + (centers * centers).sum(1)[lab] - 2 * (x64 * centers[lab]).sum(1), 0.0)
            order = np.argsort(-own, kind="stable")
            used = 0
            for j in empty:  # move the farthest points (from clusters with > 1 member) into empty clusters
                while counts[lab[order[used]]] <= 1:
                    used += 1
                i = order[used]
                counts[lab[i]] -= 1
                lab[i] = j
                counts[j] += 1
                used += 1
        return lab

    for _ in range(MAX_ITER):
        d = sq[:, None] + (centers * centers).sum(1)[None, :] - 2 * x64 @ centers.T
        new = fix_empty(np.argmin(d, axis=1))
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        centers = np.zeros_like(centers)
        np.add.at(centers, labels, x64)
        centers /= np.bincount(labels, minlength=k)[:, None]
    return centers.astype(np.float32), labels


def plain_mean(x, labels, k):
    s = np.zeros((k, x.shape[1]))
    np.add.at(s, labels, x.astype(np.float64))
    return (s / np.bincount(labels, minlength=k)[:, None]).astype(np.float32)


def main(name):
    t0 = time.time()
    ds, _ = E.load_s7b().dataset(E.dataset_arg(name))
    corpus, queries, rel = ds.corpus, ds.queries, ds.relevant()
    n_docs = len(corpus)
    prot = corpus.protected if corpus.protected is not None else np.zeros(corpus.num_vectors, bool)
    base = per_query_metrics(maxsim_matrix(queries, corpus), rel, ["ndcg@5"])["ndcg@5"]
    store = E.Store(E.load_store(name), "labels")
    checks = {"baseline_equals_E1_store": bool(np.array_equal(base, store.base))}
    target = np.array([int(r[0]) for r in rel])
    pages = []
    for i in range(n_docs):
        lo, hi = corpus.span(i)
        v = np.asarray(corpus.vectors[lo:hi], dtype=np.float32)
        p = prot[lo:hi]
        free = v[~p]
        z = linkage(_ward_input(_unit(free).astype(np.float64)), method="ward") if free.shape[0] > 1 else None
        pages.append((free, v[p], z))
    print(name, "linkage done", f"{time.time() - t0:.0f}s", flush=True)

    def evaluate(parts):
        vecs = np.concatenate([np.concatenate(pp, axis=0) for pp in parts], axis=0)
        offs = np.concatenate([[0], np.cumsum([sum(a.shape[0] for a in pp) for pp in parts])])
        comp = MultiVectorCorpus(vecs, offs)
        scores = maxsim_matrix(queries, comp)
        pq = per_query_metrics(scores, rel, ["ndcg@5"])["ndcg@5"]
        top1 = float(np.mean(rank(scores, 1)[:, 0] == target))
        return pq, top1, int(comp.num_vectors)

    res = {"store": name, "dataset": ds.name, "n_docs": n_docs, "n_queries": len(queries),
           "vectors_total": int(corpus.num_vectors), "protected_total": int(prot.sum()),
           "protected_per_page": sorted({int(pp[1].shape[0]) for pp in pages}),
           "baseline_ndcg5": float(base.mean()),
           "baseline_top1": float(np.mean(rank(maxsim_matrix(queries, corpus), 1)[:, 0] == target)),
           "cells": {}}
    arrays = {"base": base}
    for rlab, r in RATIOS.items():
        ks = [budget(f.shape[0], r) for f, _, _ in pages]
        variants = {}
        # B and C: Ward labels
        b_parts, c_parts, nclust_ok = [], [], True
        for (free, fixed, z), k in zip(pages, ks, strict=True):
            if z is None or k >= free.shape[0]:
                b_parts.append((free, fixed))
                c_parts.append((free, fixed))
                continue
            lab = fcluster(z, t=k, criterion="maxclust") - 1
            kk = int(lab.max()) + 1
            nclust_ok &= kk == k
            b_parts.append((plain_mean(free, lab, kk), fixed))
            c_parts.append((_pool(free, lab, kk, np.ones(free.shape[0], np.float32)), fixed))
        variants["B_ward_plain"] = [evaluate(b_parts)]
        variants["C_ward_rescaled"] = [evaluate(c_parts)]
        for key, fn in (("A_kmeans_plain", "kmeans"), ("D_random_prune", "random")):
            runs = []
            for s in SEEDS:
                parts = []
                for pi, ((free, fixed, _), k) in enumerate(zip(pages, ks, strict=True)):
                    rng = np.random.default_rng([s, pi])
                    if k >= free.shape[0]:
                        parts.append((free, fixed))
                    elif fn == "kmeans":
                        parts.append((kmeans(free, k, rng)[0], fixed))
                    else:
                        parts.append((free[np.sort(rng.choice(free.shape[0], k, replace=False))], fixed))
                runs.append(evaluate(parts))
            variants[key] = runs
        cell = {"ratio": r, "k_free_total": int(sum(ks)), "ward_clusters_exact": bool(nclust_ok),
                "methods": {}}
        for key, runs in variants.items():
            pqs = np.stack([x[0] for x in runs])
            mean_pq = pqs.mean(axis=0)
            per_seed = [retention(p, base, n_boot=0)[0] for p in pqs]
            point, lo, hi = retention(mean_pq, base, n_boot=NB)
            cell["methods"][key] = {"vectors_kept_total": runs[0][2], "vectors_per_page": runs[0][2] / n_docs,
                                    "vector_compression_factor": corpus.num_vectors / runs[0][2],
                                    "ndcg5": float(mean_pq.mean()), "retention": point, "retention_ci": [lo, hi],
                                    "retention_per_seed": per_seed,
                                    "seed_range": float(max(per_seed) - min(per_seed)),
                                    "top1": float(np.mean([x[1] for x in runs])),
                                    "top1_per_seed": [x[1] for x in runs]}
            arrays[f"{rlab}|{key}"] = mean_pq
        if rlab == "1/10":
            z = np.load(E.ROOT / "reports" / "research" / "Q4" / "store" / f"{name}.npz")
            q4 = z["pq"][list(z["labels"]).index("hierarchical_merge(0.1)")]
            checks["C_1/10_equals_Q4_ward_0.1"] = bool(np.array_equal(arrays["1/10|C_ward_rescaled"], q4))
        res["cells"][rlab] = cell
        print(name, rlab, {k: round(v["retention"], 4) for k, v in cell["methods"].items()},
              f"{time.time() - t0:.0f}s", flush=True)
    res["checks"] = checks
    res["environment"] = E.environment()
    res["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **{k.replace("/", "_"): v for k, v in arrays.items()})
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    ok = all(checks.values())
    print(name, "checks", checks, "OK" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
