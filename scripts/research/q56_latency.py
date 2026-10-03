"""Q5/Q6 step 2: latency, RAM and disk (PLAN.md §5-6). Run only after q56_correctness.py
passed, alone on the machine.

Warm-up rule (fixed in PLAN.md before any measurement): every configuration runs once
untimed (recorded separately), then REPEATS timed rounds, round-robin with a rotating
start; median and min-max of the timed rounds are reported.

    OPTIVISION_DATA=../optivision-rag-v2/data PYTHONPATH="src;scripts/research" \
        python scripts/research/q56_latency.py
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path

import e1_common as E
import numpy as np
import q56_common as C
import torch

from optivision.compose import save_compressed
from optivision.scoring import rank
from optivision.storage import ExactIndex

OUT = E.ROOT / "reports" / "research" / "Q5Q6"
REPEATS = 5
TOPK = 10


def main():
    t_start = time.time()
    ds = C.load()
    corpus, queries = ds.corpus, ds.queries
    n_docs, n_q = len(corpus), len(queries)
    hot, hotp, cold = C.build(corpus)
    i8 = C.Int8Direct(cold)
    lut, ham = C.BinaryLUT(hot), C.BinaryHamming(hot)
    qv, qo = queries.vectors, queries.offsets
    comp = {}  # component timings of the last call, per configuration

    def full_f32():
        return rank(ExactIndex(corpus).score(queries), TOPK)

    def full_i8_cur():
        return rank(ExactIndex(cold).score(queries), TOPK)

    def full_i8_dir():
        return rank(C.maxsim(qv, qo, i8.offsets, i8.sims, n_docs), TOPK)

    def hot_cur():
        return rank(ExactIndex(hot).score(queries), C.CANDIDATES)

    def hot_b1():
        return rank(C.maxsim(qv, qo, lut.offsets, lut.sims, n_docs), C.CANDIDATES)

    def hot_b2():
        return rank(C.maxsim(qv, qo, ham.offsets, ham.sims, n_docs), C.CANDIDATES)

    def two_tier(name, hot_fn, rescore):
        def run():
            t0 = time.perf_counter()
            h = hot_fn()
            sl = rank(h, C.CANDIDATES)
            t1 = time.perf_counter()
            if rescore == "R-cur":
                cold_s = ExactIndex(cold).rescore(queries, sl)
            else:
                cold_s = C.rescore_grouped(queries, sl, i8, "direct" if rescore == "R-dir" else "decoded")
            t2 = time.perf_counter()
            top = rank(C.merge_tiers(h, sl, cold_s), TOPK)
            t3 = time.perf_counter()
            comp[name] = {"first_stage_incl_selection": t1 - t0, "rescoring": t2 - t1, "merge_and_final_rank": t3 - t2}
            return top
        return run

    hot_scores = {
        "H-cur": lambda: ExactIndex(hot).score(queries),
        "H-B1": lambda: C.maxsim(qv, qo, lut.offsets, lut.sims, n_docs),
        "H-B2": lambda: C.maxsim(qv, qo, ham.offsets, ham.sims, n_docs),
    }
    configs = {
        "F32 full scan": full_f32,
        "I8-cur full scan": full_i8_cur,
        "I8-dir full scan": full_i8_dir,
        "H-cur first stage": hot_cur,
        "H-B1 first stage (exact)": hot_b1,
        "H-B2 first stage (approximate)": hot_b2,
        "two-tier current (H-cur -> R-cur)": two_tier("two-tier current (H-cur -> R-cur)", hot_scores["H-cur"], "R-cur"),
        "two-tier native exact (H-B1 -> R-dir)": two_tier("two-tier native exact (H-B1 -> R-dir)", hot_scores["H-B1"], "R-dir"),
        "two-tier H-cur -> R-dir": two_tier("two-tier H-cur -> R-dir", hot_scores["H-cur"], "R-dir"),
        "two-tier control (H-cur -> R-bdec)": two_tier("two-tier control (H-cur -> R-bdec)", hot_scores["H-cur"], "R-bdec"),
        "two-tier approximate (H-B2 -> R-dir)": two_tier("two-tier approximate (H-B2 -> R-dir)", hot_scores["H-B2"], "R-dir"),
    }
    names = list(configs)
    warm, timed, comps = {}, {n: [] for n in names}, {n: [] for n in names}
    for n in names:  # warm-up round, untimed for the results
        t0 = time.perf_counter()
        configs[n]()
        warm[n] = time.perf_counter() - t0
    print("warm-up done", flush=True)
    for r in range(REPEATS):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            t0 = time.perf_counter()
            configs[n]()
            timed[n].append(time.perf_counter() - t0)
            if n in comp:
                comps[n].append(dict(comp[n]))
        print(f"round {r + 1}/{REPEATS}", flush=True)

    def ms(v):
        a = 1000 * np.asarray(v) / n_q
        return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "all": a.tolist()}

    rows = []
    for n in names:
        row = {"config": n, "ms_per_query": ms(timed[n]), "warmup_ms_per_query": 1000 * warm[n] / n_q,
               "throughput_queries_per_s": float(n_q / np.median(timed[n]))}
        if comps[n]:
            row["components_ms_per_query"] = {k: ms([c[k] for c in comps[n]]) for k in comps[n][0]}
        rows.append(row)

    # temporary allocations of the numpy paths (tracemalloc; torch tensors are not traced)
    temp = {}
    for n in ("F32 full scan", "I8-cur full scan", "I8-dir full scan", "H-cur first stage",
              "H-B1 first stage (exact)", "H-B2 first stage (approximate)", "two-tier current (H-cur -> R-cur)",
              "two-tier native exact (H-B1 -> R-dir)", "two-tier control (H-cur -> R-bdec)"):
        tracemalloc.start()
        configs[n]()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        temp[n] = peak
    disk = {}
    with tempfile.TemporaryDirectory() as d:
        for key, st in (("binary_hot", hot), ("binary_plain", hotp), ("int8_cold", cold)):
            disk[key] = save_compressed(st, Path(d) / f"{key}.npz").stat().st_size
    ram = {"float32_vectors": int(corpus.vectors.nbytes + corpus.offsets.nbytes),
           "binary_hot_codes_offsets": int(hot.codes.nbytes + hot.offsets.nbytes),
           "binary_plain_codes_offsets": int(hotp.codes.nbytes + hotp.offsets.nbytes),
           "int8_cold_codes_offsets": int(cold.codes.nbytes + cold.offsets.nbytes),
           "int8_direct_layout_codes_multipliers_offsets": int(i8.q.nbytes + i8.c.nbytes + i8.offsets.nbytes),
           "note": "Int8Direct keeps a float32 multiplier per vector (4 bytes) instead of the float16 scale (2 bytes): "
                   "+2 bytes/vector in this research layout"}
    env = E.environment() | {"processor": platform.processor(), "cpus": os.cpu_count(),
                             "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                             "numpy": np.__version__,
                             "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                      cwd=E.ROOT, check=True).stdout.strip()}
    out = {"label": "MEASURED wall time / array sizes on the Q4 laptop; temp peaks MEASURED by tracemalloc (numpy only)",
           "n_queries": n_q, "n_docs": n_docs, "candidates": C.CANDIDATES, "repeats": REPEATS,
           "warmup_rule": "one untimed run of every configuration before the timed rounds (PLAN.md §5)",
           "rows": rows, "temp_peak_bytes_tracemalloc": temp, "ram_bytes": ram, "disk_bytes": disk,
           "environment": env, "seconds": time.time() - t_start}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("done", f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
