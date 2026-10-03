"""Validate the batched ExactIndex.rescore against the previous per-query body
(reports/engineering/batched_rescore/PLAN.md).

    correctness: six Q4 datasets x two hot tiers x candidates {1, 10, 50, 200, 500};
                 identical shortlists, final rankings and per-query nDCG@5; bitwise
                 rescore equality reported; NpzStorage reload checked bitwise.
    timing:      the Q4 / Q5/Q6 protocol (one untimed warm-up run, 5 timed rounds,
                 round-robin, rotating start), candidates = 50, Ward 1/4 > binary hot.

    OPTIVISION_DATA=../optivision-rag-v2/data PYTHONPATH="src;scripts/research" \
        python scripts/engineering/batched_rescore.py correctness
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import tracemalloc

import e1_common as E
import numpy as np

from optivision.compose import Pipeline
from optivision.evaluation import per_query_metrics
from optivision.scoring import rank, segment_reduce
from optivision.stages import BinaryQuantizer, HierarchicalMerge, Int8Quantizer
from optivision.storage import ExactIndex, NpzStorage, TieredIndex

OUT = E.ROOT / "reports" / "engineering" / "batched_rescore"
STORES = ("colpali_docvqa", "colpali_infovqa", "colqwen2_docvqa", "colqwen2_infovqa",
          "colqwen25_docvqa", "colqwen25_infovqa")
COUNTS = (1, 10, 50, 200, 500)
REPEATS = 5
TOPK = 10


class OldExactIndex(ExactIndex):
    """ExactIndex with the previous rescore body, copied verbatim (the reference)."""

    def rescore(self, queries, candidates):
        q = self._q(queries)
        cand = np.asarray(candidates, dtype=np.int64)
        out = np.full(cand.shape, -np.inf, dtype=np.float32)
        offsets = self.store.offsets
        for qi in range(cand.shape[0]):
            lo_q, hi_q = int(q.offsets[qi]), int(q.offsets[qi + 1])
            if hi_q == lo_q:
                out[qi] = 0.0
                continue
            qv = q.decode_rows(lo_q, hi_q)
            docs = cand[qi]
            spans = [(int(offsets[d]), int(offsets[d + 1])) for d in docs]
            blocks = [self.store.decode_rows(lo, hi) for lo, hi in spans]
            counts = np.array([b.shape[0] for b in blocks], dtype=np.int64)
            if counts.sum() == 0:
                continue
            block = np.concatenate(blocks, axis=0)
            local = np.concatenate([[0], np.cumsum(counts)])
            per_token = segment_reduce(qv @ block.T, local, np.maximum, axis=1, fill=-np.inf)
            out[qi] = per_token.sum(axis=0)
        return out


def load(name):
    ds, _ = E.load_s7b().dataset(E.dataset_arg(name))
    corpus = ds.corpus
    hots = {"ward_0.25_binary": Pipeline([HierarchicalMerge(ratio=0.25), BinaryQuantizer()]).compress(corpus),
            "binary": Pipeline([BinaryQuantizer()]).compress(corpus)}
    cold = Pipeline([Int8Quantizer("per_vector")]).compress(corpus)
    return ds, hots, cold


def correctness():
    res, ok_all = {}, True
    for name in STORES:
        t0 = time.time()
        ds, hots, cold = load(name)
        queries, rel = ds.queries, ds.relevant()
        new_i, old_i = ExactIndex(cold), OldExactIndex(cold)
        r = {"n_queries": len(queries), "n_docs": len(ds.corpus), "cells": []}
        for hname, hot in hots.items():
            hot_scores = ExactIndex(hot).score(queries)
            for m in COUNTS:
                m_eff = min(m, len(ds.corpus))
                sl = rank(hot_scores, m_eff)
                a, b = old_i.rescore(queries, sl), new_i.rescore(queries, sl)
                ta = TieredIndex(ExactIndex(hot), old_i, candidates=m)
                tb = TieredIndex(ExactIndex(hot), new_i, candidates=m)
                fa, fb = ta.score(queries), tb.score(queries)
                ra, rb = rank(fa), rank(fb)
                na = per_query_metrics(fa, rel, ["ndcg@5"])["ndcg@5"]
                nb = per_query_metrics(fb, rel, ["ndcg@5"])["ndcg@5"]
                cell = {"hot": hname, "candidates": m, "effective_candidates": m_eff,
                        "shortlist_identical": True,  # same hot scores feed both (checked below)
                        "final_ranking_identical_queries": int(np.all(ra == rb, axis=1).sum()),
                        "ndcg5_identical": bool(np.array_equal(na, nb)),
                        "rescore_bitwise_equal": bool(np.array_equal(a, b)),
                        "rescore_max_abs_diff": float(np.nanmax(np.abs(np.where(np.isfinite(a), a - b, 0.0)))),
                        "ndcg5_mean": float(nb.mean())}
                cell["shortlist_identical"] = bool(np.array_equal(rank(ExactIndex(hot).score(queries), m_eff), sl))
                cell["pass"] = (cell["final_ranking_identical_queries"] == len(queries) and cell["ndcg5_identical"]
                                and cell["shortlist_identical"])
                ok_all &= cell["pass"]
                r["cells"].append(cell)
                print(name, hname, m, "PASS" if cell["pass"] else "FAIL", "bitwise" if cell["rescore_bitwise_equal"] else "",
                      flush=True)
        with tempfile.TemporaryDirectory() as d:
            st = NpzStorage(d)
            st.write("cold", cold)
            reloaded = st.read("cold")
            sl = rank(ExactIndex(hots["ward_0.25_binary"]).score(queries), 50)
            r["reload_rescore_bitwise_equal"] = bool(np.array_equal(ExactIndex(reloaded).rescore(queries, sl),
                                                                    new_i.rescore(queries, sl)))
            ok_all &= r["reload_rescore_bitwise_equal"]
        r["seconds"] = time.time() - t0
        res[name] = r
        del ds, hots, cold
    out = {"criterion": "identical shortlists, final rankings and per-query nDCG@5 for every query (PLAN.md section 2)",
           "all_pass": bool(ok_all), "datasets": res, "environment": env()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "correctness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("ALL PASS" if ok_all else "FAILURE", flush=True)
    return 0 if ok_all else 1


def env():
    return E.environment() | {"processor": platform.processor(), "cpus": os.cpu_count(), "numpy": np.__version__,
                              "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                       cwd=E.ROOT, check=True).stdout.strip(),
                              "worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain", "src"],
                                                                    capture_output=True, text=True, cwd=E.ROOT,
                                                                    check=True).stdout.strip())}


def timing():
    res = {}
    for name in STORES:
        ds, hots, cold = load(name)
        queries = ds.queries
        n_q = len(queries)
        hot = hots["ward_0.25_binary"]
        old_i, new_i = OldExactIndex(cold), ExactIndex(cold)
        sl = rank(ExactIndex(hot).score(queries), 50)
        t_old, t_new = TieredIndex(ExactIndex(hot), old_i, 50), TieredIndex(ExactIndex(hot), new_i, 50)
        configs = {"old rescore": lambda o=old_i, q=queries, s=sl: o.rescore(q, s),
                   "batched rescore": lambda n=new_i, q=queries, s=sl: n.rescore(q, s),
                   "old two-tier": lambda t=t_old, q=queries: rank(t.score(q), TOPK),
                   "batched two-tier": lambda t=t_new, q=queries: rank(t.score(q), TOPK)}
        names = list(configs)
        warm = {}
        for n in names:
            t0 = time.perf_counter()
            configs[n]()
            warm[n] = time.perf_counter() - t0
        times = {n: [] for n in names}
        for rnd in range(REPEATS):
            order = names[rnd % len(names):] + names[:rnd % len(names)]
            for n in order:
                t0 = time.perf_counter()
                configs[n]()
                times[n].append(time.perf_counter() - t0)
        mem = {}
        for n in ("old rescore", "batched rescore"):
            tracemalloc.start()
            configs[n]()
            mem[n] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        ms = {n: 1000 * np.asarray(v) / n_q for n, v in times.items()}
        res[name] = {"n_queries": n_q,
                     "rows": {n: {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                                  "all": v.tolist(), "warmup": 1000 * warm[n] / n_q} for n, v in ms.items()},
                     "within_round_ratio_old_over_new": {
                         "rescore": (ms["old rescore"] / ms["batched rescore"]).tolist(),
                         "two_tier": (ms["old two-tier"] / ms["batched two-tier"]).tolist()},
                     "peak_traced_bytes": mem}
        print(name, {n: round(float(np.median(v)), 2) for n, v in ms.items()}, flush=True)
        del ds, hots, cold
    out = {"protocol": "one untimed warm-up run per configuration, then 5 timed rounds over the full query pool, "
                       "round-robin with rotating start; candidates 50; hot Ward 1/4 > binary; cold int8 per vector",
           "datasets": res, "environment": env()}
    (OUT / "timing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "correctness":
        sys.exit(correctness())
    timing()
