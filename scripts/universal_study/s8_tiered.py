"""Phase 8: does a two-tier index (small hot codes for candidates, cold vectors for rescoring)
recover float quality on real ColPali pages? Tests the earlier *projection* that binary + int8
rerank recovers 99-100% -- which had only been measured without merging and never on ViDoRe."""
import json
import os
import pathlib
import sys
import time

import numpy as np

from optivision.benchmark import load_legacy_dataset
from optivision.compose import Pipeline
from optivision.evaluation import per_query_metrics, retention
from optivision.scoring import maxsim_matrix
from optivision.stages import AdaptiveMerge, BinaryQuantizer, Float16Quantizer, Int8Quantizer
from optivision.storage import ExactIndex, TieredIndex

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
DATASETS = {
    "E2-colpali-infovqa": (D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json"),
    "E2-colpali-docvqa": (D + "/cache/colpali_docvqa_test_subsampled.npz", D + "/vidore_docvqa_test_subsampled/queries.json"),
    "E3-colpali-generated": (D + "/cache/colpali_generated.npz", D + "/corpus/queries.json"),
    "E1-colsmol-generated": (D + "/cache/colsmol.npz", D + "/corpus/queries.json"),
}
METRIC = "ndcg@5"


def run(name, radii):
    t0 = time.time()
    cache, qj = DATASETS[name]
    ds = load_legacy_dataset(cache, qj, name=name, with_images=False)
    corpus, queries, rel = ds.corpus, ds.queries, ds.relevant()
    base = per_query_metrics(maxsim_matrix(queries, corpus), rel, [METRIC])[METRIC]
    n_docs = len(corpus)
    print(f"== {name}: {n_docs} docs, {len(queries)} queries, float nDCG@5 {np.nanmean(base):.4f}", flush=True)
    colds = {
        "cold int8/vec (all vectors)": Pipeline([Int8Quantizer("per_vector")]),
        "cold float16 (all vectors)": Pipeline([Float16Quantizer()]),
    }
    cold_idx = {k: (ExactIndex(p.compress(corpus)), p.compress(corpus).report()["bytes_per_doc"]) for k, p in colds.items()}
    rows = []
    for r in radii:
        hot_pipe = Pipeline(([AdaptiveMerge(radius=r)] if r is not None else []) + [BinaryQuantizer()])
        hot_c = hot_pipe.compress(corpus)
        hot = ExactIndex(hot_c)
        hot_bytes = hot_c.report()["bytes_per_doc"]
        hot_only = per_query_metrics(hot.score(queries), rel, [METRIC])[METRIC]
        point, lo, hi = retention(hot_only, base, n_boot=1000)
        label = f"hot: {'merge ' + str(r) + ' > ' if r is not None else ''}binary"
        rows.append({"hot": label, "cold": None, "candidates": 0, "hot_bytes_per_doc": hot_bytes, "cold_bytes_per_doc": 0,
                     "retention": point, "ci": [lo, hi]})
        print(f"  {label:32s} alone                          hot {hot_bytes / 1e3:6.2f} KB/doc  ret {point:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
        merged_int8 = Pipeline(([AdaptiveMerge(radius=r)] if r is not None else []) + [Int8Quantizer("per_vector")])
        mc = merged_int8.compress(corpus)
        colds_here = dict(cold_idx)
        colds_here[f"cold int8/vec (same merge {r})"] = (ExactIndex(mc), mc.report()["bytes_per_doc"])
        for cname, (cidx, cbytes) in colds_here.items():
            for m in (10, 20, 50, 100):
                if m >= n_docs:
                    continue
                t = TieredIndex(hot, cidx, candidates=m)
                s = t.score(queries)
                pq = per_query_metrics(s, rel, [METRIC])[METRIC]
                point, lo, hi = retention(pq, base, n_boot=1000)
                rows.append({"hot": label, "cold": cname, "candidates": m, "hot_bytes_per_doc": hot_bytes,
                             "cold_bytes_per_doc": cbytes, "retention": point, "ci": [lo, hi],
                             "hot_ms_per_query": 1000 * t.last_timing["hot_seconds"] / len(queries),
                             "rescore_ms_per_query": 1000 * t.last_timing["rescore_seconds"] / len(queries)})
                print(f"  {label:32s} + {cname:30s} m={m:3d}  hot {hot_bytes / 1e3:6.2f} KB  cold {cbytes / 1e3:7.2f} KB  "
                      f"ret {point:.3f} [{lo:.3f},{hi:.3f}]  rescore {rows[-1]['rescore_ms_per_query']:.1f} ms/q", flush=True)
    out = {"dataset": name, "n_docs": n_docs, "n_queries": len(queries), "metric": METRIC,
           "float_ndcg5": float(np.nanmean(base)), "rows": rows, "seconds": time.time() - t0}
    pathlib.Path(S + "/phase8/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(S + f"/phase8/results/{name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    radii = [None if x == "none" else float(x) for x in sys.argv[1].split(",")]
    for n in (sys.argv[2:] or list(DATASETS)):
        run(n, radii)
