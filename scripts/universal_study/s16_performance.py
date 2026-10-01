"""Compression time and peak traced memory per stage on ColPali InfoVQA (500 pages x 1031 x 128)."""
import json
import os
import pathlib
import time
import tracemalloc

from optivision.benchmark import load_legacy_dataset
from optivision.compose import Pipeline
from optivision.scoring import maxsim_matrix
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RedundancyPruner,
)

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
ds = load_legacy_dataset(D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json",
                         name="infovqa", with_images=False)
corpus, queries = ds.corpus, ds.queries
input_mb = corpus.nbytes / 1e6
pipes = {
    "float16": Pipeline([Float16Quantizer()]),
    "int8 per-vector": Pipeline([Int8Quantizer("per_vector")]),
    "int4": Pipeline([Int4Quantizer()]),
    "lloyd2 (fit + encode)": Pipeline([Lloyd2Quantizer()]),
    "binary": Pipeline([BinaryQuantizer()]),
    "redundancy t=0.8": Pipeline([RedundancyPruner(threshold=0.8)]),
    "adaptive merge r=0.6": Pipeline([AdaptiveMerge(radius=0.6)]),
    "hierarchical merge 20%": Pipeline([HierarchicalMerge(ratio=0.2)]),
    "pca 64 (fit + project)": Pipeline([PCAProjector(dim=64)]),
    "adaptive r=0.6 > int8 per-vector": Pipeline([AdaptiveMerge(radius=0.6), Int8Quantizer("per_vector")]),
}
rows = []
for name, pipe in pipes.items():
    tracemalloc.start()
    t0 = time.perf_counter()
    comp = pipe.compress(corpus)
    secs = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t1 = time.perf_counter()
    maxsim_matrix(pipe.transform_queries(queries), comp)
    q_ms = 1000 * (time.perf_counter() - t1) / len(queries)
    rows.append({"pipeline": name, "compress_seconds": secs, "ms_per_page": 1000 * secs / len(corpus),
                 "peak_traced_mb": peak / 1e6, "input_mb": input_mb, "output_mb": comp.nbytes / 1e6,
                 "query_ms": q_ms})
    print(f"{name:34s} compress {secs:6.2f}s ({1000 * secs / len(corpus):5.1f} ms/page)  peak {peak / 1e6:7.1f} MB  "
          f"out {comp.nbytes / 1e6:7.2f} MB  scan {q_ms:5.1f} ms/query", flush=True)
t1 = time.perf_counter()
maxsim_matrix(queries, corpus)
float_ms = 1000 * (time.perf_counter() - t1) / len(queries)
print(f"float32 baseline scan {float_ms:.1f} ms/query; input {input_mb:.1f} MB", flush=True)
out = {"dataset": "ColPali ViDoRe InfoVQA", "n_docs": len(corpus), "n_queries": len(queries),
       "input_mb": input_mb, "float_query_ms": float_ms, "rows": rows}
pathlib.Path(S + "/perf/results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
