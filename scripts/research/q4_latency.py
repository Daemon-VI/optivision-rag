"""Q4: scan latency and preprocessing cost of the existing implementation (PLAN.md §6).

For one 128-d DocVQA dataset: builds every single-axis and matched-budget
configuration with a fresh, uncached pipeline (its wall time is the MEASURED
preprocessing cost), then times ExactIndex.score -- query projection plus the
blockwise decode + MaxSim of the existing numpy implementation -- over all
queries, REPEATS rounds, round-robin over configurations with a rotating start.
The two-tier rows report the hot scan and the rescore separately. Nothing is
optimized. Run it alone on the machine.

    OPTIVISION_DATA=../optivision-rag-v2/data PYTHONPATH="src;scripts/research" \
        python scripts/research/q4_latency.py colqwen2_docvqa
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

import e1_common as E
import numpy as np
from q4_ablation import C, P, W

from optivision.compose import Pipeline
from optivision.stages import BinaryQuantizer, Int8Quantizer
from optivision.storage import ExactIndex, TieredIndex

OUT = E.ROOT / "reports" / "research" / "Q4" / "latency"
REPEATS = 5


def configs() -> list[tuple[str, list]]:
    g = [("baseline", [])]
    g += [("A", [W(r)]) for r in (0.5, 0.33, 0.25, 0.125, 0.1)]
    g += [("B", [P(w)]) for w in (64, 32, 16, 8)]
    g += [("C", [C(c)]) for c in ("float16", "int8", "int4", "int4c", "2bit", "binary")]
    g += [("A+B", [W(0.5), P(64)]), ("A+B", [W(0.25), P(64)]), ("A+B", [W(0.5), P(32)]), ("A+B", [W(0.25), P(16)]),
          ("A+C", [W(0.5), C("int8")]), ("A+C", [W(0.25), C("float16")]), ("A+C", [W(0.25), C("int4c")]),
          ("A+C", [W(0.25), C("int8")]), ("A+C", [W(0.25), C("binary")]),
          ("B+C", [P(64), C("int8")]), ("B+C", [P(32), C("float16")]), ("B+C", [P(64), C("int4c")]),
          ("B+C", [P(32), C("int8")]),
          ("A+B+C", [W(0.5), P(64), C("float16")]), ("A+B+C", [W(0.25), P(64), C("int8")])]
    return g


def main(name: str) -> None:
    t_start = time.time()
    ds, _ = E.load_s7b().dataset(E.dataset_arg(name))
    corpus, queries = ds.corpus, ds.queries
    n_q = len(queries)
    built = []
    for group, stages in configs():
        pipe = Pipeline(stages)
        t0 = time.perf_counter()
        comp = pipe.compress(corpus) if stages else corpus
        built.append({"group": group, "label": pipe.label() if stages else "float32", "pipe": pipe, "store": comp,
                      "preprocess_seconds": time.perf_counter() - t0, "scan_seconds": []})
        print(f"built {built[-1]['label']:50s} {built[-1]['preprocess_seconds']:6.1f}s", flush=True)
    cold = Pipeline([Int8Quantizer("per_vector")]).compress(corpus)
    tiers = []
    for hot_label, hot_pipe in (("binary", Pipeline([BinaryQuantizer()])),
                                ("hierarchical_merge(0.25) > binary", Pipeline([W(0.25), BinaryQuantizer()]))):
        tiers.append({"hot": hot_label, "index": TieredIndex(ExactIndex(hot_pipe.compress(corpus)), ExactIndex(cold), 50),
                      "hot_seconds": [], "rescore_seconds": []})
    for rnd in range(REPEATS):
        order = built[rnd % len(built):] + built[:rnd % len(built)]
        for b in order:
            idx = ExactIndex(b["store"], b["pipe"] if b["pipe"].stages else None)
            t0 = time.perf_counter()
            idx.score(queries)
            b["scan_seconds"].append(time.perf_counter() - t0)
        for t in tiers:
            t["index"].score(queries)
            t["hot_seconds"].append(t["index"].last_timing["hot_seconds"])
            t["rescore_seconds"].append(t["index"].last_timing["rescore_seconds"])
        print(f"round {rnd + 1}/{REPEATS} done", flush=True)
    base_med = float(np.median(built[0]["scan_seconds"]))

    def ms(v):
        v = 1000 * np.asarray(v) / n_q
        return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "all": v.tolist()}

    rows = [{"group": b["group"], "label": b["label"], "preprocess_seconds": b["preprocess_seconds"],
             "scan_ms_per_query": ms(b["scan_seconds"]),
             "scan_fraction_of_float32": float(np.median(b["scan_seconds"])) / base_med,
             "vectors_per_doc": (b["store"].num_vectors / len(corpus)),
             "dim": int(getattr(b["store"], "dim", corpus.dimension))} for b in built]
    tier_rows = [{"hot": t["hot"], "cold": "int8(per_vector), all vectors", "candidates": 50,
                  "hot_ms_per_query": ms(t["hot_seconds"]), "rescore_ms_per_query": ms(t["rescore_seconds"])}
                 for t in tiers]
    env = E.environment() | {"processor": platform.processor(), "cpus": os.cpu_count(),
                             "threads_env": {k: os.environ.get(k) for k in
                                             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    try:
        import threadpoolctl
        env["blas"] = threadpoolctl.threadpool_info()
    except ImportError:
        env["blas"] = None
    out = {"store": name, "n_docs": len(corpus), "n_queries": n_q, "repeats": REPEATS,
           "label": "MEASURED wall time of the existing implementation on this laptop CPU",
           "rows": rows, "tiered": tier_rows, "environment": env, "seconds": time.time() - t_start}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(name, f"done {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
