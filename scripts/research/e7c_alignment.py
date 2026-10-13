"""E7c: OptiVision evaluation vs the official ViDoRe evaluation (reports/research/Q7/E7c/PLAN.md).

Rebuilds E7a's random-pruning (D) and K-means/plain (A) indexes with E7a's code and
seeds, scores every query against every page (exact MaxSim), and evaluates the same
score matrices under six variants:
  E0  first-occurrence labels, OptiVision nDCG@5   (current; equals Q4/E7a)
  E0p first-occurrence labels, pytrec_eval ndcg_cut_5
  E1  last-occurrence labels,  OptiVision          E2 all-occurrence labels, OptiVision
  OL  last-occurrence labels,  pytrec_eval   (legacy vidore-benchmark ViDoReEvaluatorQA)
  OM  all-occurrence labels,   pytrec_eval   (MTEB, the current official route; primary)
Writes reports/research/Q7/E7c/<store>.json.

    OPTIVISION_DATA=../optivision-rag-v2/data PYTHONPATH="src;scripts/research" \
        python scripts/research/e7c_alignment.py colqwen2_docvqa
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import e1_common as E
import numpy as np
import pyarrow.parquet as pq
import pytrec_eval
from e7a_attribution import RATIOS, SEEDS, budget, kmeans

from optivision.evaluation import per_query_metrics, retention
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix

OUT = E.ROOT / "reports" / "research" / "Q7" / "E7c"
E7A = E.ROOT / "reports" / "research" / "Q7" / "E7a"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
RAW = {"docvqa": HUB / "datasets--vidore--docvqa_test_subsampled" / "snapshots"
       / "49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6" / "data" / "test-00000-of-00001.parquet",
       "infovqa": HUB / "datasets--vidore--infovqa_test_subsampled" / "snapshots"
       / "f793e830aaeae1ceb8a2df626fc555b3fd04d3db" / "data" / "test-00000-of-00001.parquet"}
VARIANTS = ("E0", "E0p", "E1", "E2", "OL", "OM")
LABELS = {"E0": "first", "E0p": "first", "E1": "last", "E2": "all", "OL": "last", "OM": "all"}


def labels_for(store, rel):
    """first / last / all relevance per OptiVision query, from the raw dataset rows."""
    rows = pq.read_table(RAW[store.split("_")[1]], columns=["query"]).to_pylist()
    texts = [r["query"] for r in rows]
    firsts, lasts, alls = {}, {}, {}
    for i, t in enumerate(texts):
        firsts.setdefault(t, i)
        lasts[t] = i
        alls.setdefault(t, []).append(i)
    order = list(firsts)  # unique texts in first-occurrence order
    cur = [int(r[0]) for r in rel]
    checks = {"one_label_per_query": all(len(r) == 1 for r in rel),
              "query_j_is_jth_unique_text_first_row": cur == [firsts[t] for t in order]}
    lab = {"first": [np.array([firsts[t]]) for t in order], "last": [np.array([lasts[t]]) for t in order],
           "all": [np.array(alls[t]) for t in order]}
    stats = {"queries": len(order), "repeated_texts": sum(len(v) > 1 for v in alls.values()),
             "queries_last_differs_from_first": sum(firsts[t] != lasts[t] for t in order),
             "relevant_pairs_all": sum(len(v) for v in alls.values())}
    return lab, checks, stats


def pytrec(scores, labels):
    qrels = {f"q{j}": {f"d{p}": 1 for p in labels[j]} for j in range(len(labels))}
    run = {f"q{j}": {f"d{p}": float(scores[j, p]) for p in range(scores.shape[1])} for j in range(scores.shape[0])}
    res = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.5"}).evaluate(run)
    return np.array([res[f"q{j}"]["ndcg_cut_5"] for j in range(len(labels))])


def evaluate(scores, lab):
    out = {}
    for v in VARIANTS:
        labels = lab[LABELS[v]]
        out[v] = pytrec(scores, labels) if v in ("E0p", "OL", "OM") else \
            per_query_metrics(scores, labels, ["ndcg@5"])["ndcg@5"]
    return out


def main(store):
    t0 = time.time()
    ds, _ = E.load_s7b().dataset(E.dataset_arg(store))
    corpus, queries, rel = ds.corpus, ds.queries, ds.relevant()
    lab, checks, stats = labels_for(store, rel)
    prot = corpus.protected if corpus.protected is not None else np.zeros(corpus.num_vectors, bool)
    pages = []
    for i in range(len(corpus)):
        lo, hi = corpus.span(i)
        v = np.asarray(corpus.vectors[lo:hi], dtype=np.float32)
        p = prot[lo:hi]
        pages.append((v[~p], v[p]))

    def build(parts):
        vecs = np.concatenate([np.concatenate(pp, axis=0) for pp in parts], axis=0)
        offs = np.concatenate([[0], np.cumsum([sum(a.shape[0] for a in pp) for pp in parts])])
        return maxsim_matrix(queries, MultiVectorCorpus(vecs, offs))

    base = evaluate(maxsim_matrix(queries, corpus), lab)
    e7a = np.load(E7A / f"{store}.npz")
    checks["E0_baseline_equals_E7a"] = bool(np.array_equal(base["E0"], e7a["base"]))
    res = {"store": store, "label_stats": stats, "baseline": {v: float(base[v].mean()) for v in VARIANTS},
           "cells": {}}
    for rlab, r in RATIOS.items():
        cell = {}
        for key, fn in (("A_kmeans_plain", "kmeans"), ("D_random_prune", "random")):
            per_seed = []
            for s in SEEDS:
                parts = []
                for pi, (free, fixed) in enumerate(pages):
                    k = budget(free.shape[0], r)
                    rng = np.random.default_rng([s, pi])
                    if k >= free.shape[0]:
                        parts.append((free, fixed))
                    elif fn == "kmeans":
                        parts.append((kmeans(free, k, rng)[0], fixed))
                    else:
                        parts.append((free[np.sort(rng.choice(free.shape[0], k, replace=False))], fixed))
                per_seed.append(evaluate(build(parts), lab))
            mean = {v: np.mean([ps[v] for ps in per_seed], axis=0) for v in VARIANTS}
            checks[f"E0_{rlab}_{key}_equals_E7a"] = bool(np.array_equal(mean["E0"],
                                                                        e7a[f"{rlab.replace('/', '_')}|{key}"]))
            ent = {v: {"retention": float(mean[v].mean() / base[v].mean()),
                       "retention_per_seed": [float(ps[v].mean() / base[v].mean()) for ps in per_seed]}
                   for v in VARIANTS}
            p, lo, hi = retention(mean["OM"], base["OM"], n_boot=1000)
            ent["OM"]["retention_ci"] = [lo, hi]
            cell[key] = ent
        res["cells"][rlab] = cell
        print(store, rlab, {k: {v: round(x[v]["retention"], 4) for v in ("E0", "OM", "OL")} for k, x in cell.items()},
              f"{time.time() - t0:.0f}s", flush=True)
    res["checks"] = checks
    res["environment"] = E.environment() | {"pytrec_eval": "pytrec-eval-terrier 0.5.10"}
    res["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{store}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    ok = all(checks.values())
    print(store, "checks", checks, "OK" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
