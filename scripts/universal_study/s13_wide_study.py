"""R12: fixed compression of wide (2,560-4,096-d) multi-vector encoders.

On one ``vec:<prefix>`` dataset (files from ``scripts/encode_vectors.py``) it measures:

* the float32 baseline: nDCG@5/@10, Recall@1/@5, vectors and bytes per page, exact
  scan latency;
* fixed pipelines, grouped to separate the compression axes:
    codec      float32 / float16 / int8 / int4 / centred int4 / 2-bit / binary
    token      Ward merging to 1/2, 1/3, 1/4, 1/10; centred adaptive merging r=0.6
    token+codec  Ward 1/3 and 1/4 with each codec
    dim        PCA to d/2, d/4, d/8, d/16, 128 (random projection to d/4 as a control)
    dim+codec  each PCA width with float16 / int8 / centred int4 / binary
    token+dim  Ward 1/3 and 1/4 with PCA to d/4, d/8, 128
    token+dim+codec  the same with int8 / centred int4 / binary
* two-tier rows: binary codes (plain and after Ward 1/4) in RAM, a 50-page
  shortlist rescored with per-vector int8 codes of every vector;
* the Pareto frontier of stored bytes per page against nDCG@5 retention.

Bytes are the actual stored bytes (codes, per-vector scales and shared codec
state), not vector counts. Retention is against the same encoder's float32 index,
with a paired bootstrap 95% interval over queries.

    OPTIVISION_DATA=data OPTIVISION_OUT=reports/universal/raw OPTIVISION_SCORE_DEVICE=cuda \
        python scripts/universal_study/s13_wide_study.py vec:nemotron-colembed-vl-4b-v2_docvqa_test_subsampled
"""

import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

from optivision.benchmark import load_vector_dataset
from optivision.calibration import projection_targets, score_space
from optivision.compose import Pipeline
from optivision.evaluation import per_query_metrics, retention
from optivision.pareto import pareto_front
from optivision.scoring import maxsim_matrix
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    DimensionProjector,
    Float16Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
)
from optivision.storage import ExactIndex, TieredIndex

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
METRICS = ["ndcg@5", "ndcg@10", "recall@1", "recall@5"]
CODECS = {
    "float16": Float16Quantizer, "int8/vec": lambda: Int8Quantizer("per_vector"), "int4": Int4Quantizer,
    "int4c": lambda: Int4Quantizer(center="mean"), "2-bit": Lloyd2Quantizer, "binary": BinaryQuantizer,
}


def grid(dim: int) -> dict[str, list[Pipeline]]:
    widths = projection_targets(dim)
    wards = (0.33, 0.25)
    sub = (dim // 4, dim // 8, 128)
    g: dict[str, list[Pipeline]] = {
        "codec": [Pipeline([]), *[Pipeline([c()]) for c in CODECS.values()]],
        "token": [Pipeline([HierarchicalMerge(ratio=r)]) for r in (0.5, 0.33, 0.25, 0.1)]
        + [Pipeline([AdaptiveMerge(radius=0.6, center="mean")])],
        "token+codec": [Pipeline([HierarchicalMerge(ratio=r), c()]) for r in wards for c in CODECS.values()],
        "dim": [Pipeline([DimensionProjector(w)]) for w in widths] + [Pipeline([DimensionProjector(dim // 4, "random")])],
        "dim+codec": [Pipeline([DimensionProjector(w), CODECS[c]()]) for w in widths
                      for c in ("float16", "int8/vec", "int4c", "binary")],
        "token+dim": [Pipeline([HierarchicalMerge(ratio=r), DimensionProjector(w)]) for r in wards for w in sub],
        "token+dim+codec": [Pipeline([HierarchicalMerge(ratio=r), DimensionProjector(w), CODECS[c]()])
                            for r in wards for w in sub for c in ("int8/vec", "int4c", "binary")],
    }
    return g


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
           "score_device": os.environ.get("OPTIVISION_SCORE_DEVICE") or "numpy-cpu"}
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return env


def run(name: str) -> None:
    t_start = time.time()
    ds = load_vector_dataset(f"{D}/vectors/{name[4:]}", name=name[4:]) if name.startswith("vec:") else None
    if ds is None:
        raise SystemExit("s13 takes vec:<prefix> datasets")
    corpus, queries, rel = ds.corpus, ds.queries, ds.relevant()
    n_docs, n_q, dim = len(corpus), len(queries), corpus.dimension
    print(f"== {ds.name}: {n_docs} docs, {n_q} queries, {dim}-d, {corpus.num_vectors / n_docs:.0f} vectors/page",
          flush=True)

    t = time.perf_counter()
    base_scores = maxsim_matrix(queries, corpus)
    base_scan_ms = 1000 * (time.perf_counter() - t) / n_q
    base = per_query_metrics(base_scores, rel, METRICS)
    fp32_bytes = corpus.num_vectors * dim * 4 / n_docs
    baseline = {"dim": dim, "vectors_per_doc": corpus.num_vectors / n_docs, "float32_bytes_per_doc": fp32_bytes,
                "float16_bytes_per_doc": fp32_bytes / 2, "scan_ms_per_query": base_scan_ms,
                **{m: float(np.nanmean(v)) for m, v in base.items()},
                "encode_seconds": corpus.attrs.get("encode_seconds"), "encoder": {
                    k: corpus.attrs.get(k) for k in ("model", "revision", "dtype", "device", "placement",
                                                     "loading_report", "attn_implementation")}}
    print(f"   float32: nDCG@5 {baseline['ndcg@5']:.4f}  {fp32_bytes / 1e6:.2f} MB/page  scan {base_scan_ms:.1f} ms/query",
          flush=True)

    space = grid(dim)
    group_of = {id(p): g for g, ps in space.items() for p in ps}

    def progress(sc):
        print(f"   {sc.label:62s} x{sc.report['compression_vs_float32']:7.1f}  "
              f"{sc.compress_seconds + sc.score_seconds:6.1f}s", flush=True)

    pipes = [p for ps in space.values() for p in ps]
    scored = score_space(corpus, queries, space, progress=progress)
    rows = []
    for p, sc in zip(pipes, scored, strict=True):
        pq = per_query_metrics(sc.scores, rel, METRICS)
        row = {"group": group_of[id(p)], "label": sc.label, "pipeline": sc.pipeline,
               "vectors_per_doc": float(sc.report["vectors_per_doc"]), "dim": int(sc.report["dim"]),
               "bits_per_dim": float(sc.report["bits_per_dim"]), "bytes_per_doc": sc.bytes_per_doc,
               "compression_vs_float32": float(sc.report["compression_vs_float32"]),
               "compress_seconds": sc.compress_seconds, "scan_ms_per_query": sc.query_ms}
        for m in METRICS:
            point, lo, hi = retention(pq[m], base[m], n_boot=1000)
            row[m] = float(np.nanmean(pq[m]))
            row[f"retention:{m}"] = point
            row[f"retention_ci:{m}"] = [lo, hi]
        rows.append(row)

    # two tiers: binary codes in RAM, int8 codes of every vector on disk, 50-page shortlist
    cold_pipe = Pipeline([Int8Quantizer("per_vector")])
    cold = cold_pipe.compress(corpus)
    cold_bytes = cold.report()["bytes_per_doc"]
    tiered = []
    for hot_label, hot_pipe in (("binary", Pipeline([BinaryQuantizer()])),
                                ("ward 1/4 > binary", Pipeline([HierarchicalMerge(ratio=0.25), BinaryQuantizer()]))):
        hot = hot_pipe.compress(corpus)
        idx = TieredIndex(ExactIndex(hot), ExactIndex(cold), candidates=50)
        s = idx.score(queries)
        pq = per_query_metrics(s, rel, ["ndcg@5"])["ndcg@5"]
        point, lo, hi = retention(pq, base["ndcg@5"], n_boot=1000)
        hot_bytes = hot.report()["bytes_per_doc"]
        tiered.append({"hot": hot_label, "cold": "int8/vec (all vectors)", "candidates": 50,
                       "ram_bytes_per_doc": hot_bytes, "disk_bytes_per_doc": hot_bytes + cold_bytes,
                       "ram_compression_vs_float32": fp32_bytes / hot_bytes,
                       "disk_compression_vs_float32": fp32_bytes / (hot_bytes + cold_bytes),
                       "retention:ndcg@5": point, "retention_ci:ndcg@5": [lo, hi],
                       "hot_ms_per_query": 1000 * idx.last_timing["hot_seconds"] / n_q,
                       "rescore_ms_per_query": 1000 * idx.last_timing["rescore_seconds"] / n_q})
        print(f"   tiered {hot_label:20s} RAM {hot_bytes / 1e3:8.1f} KB  ret {point:.3f}", flush=True)

    front = pareto_front(rows, costs=("bytes_per_doc",), gain="retention:ndcg@5")
    front_labels = [r["label"] for r in front]
    for r in rows:
        r["pareto"] = r["label"] in front_labels
    out = {"dataset": ds.name, "n_docs": n_docs, "n_queries": n_q, "baseline": baseline, "rows": rows,
           "tiered": tiered, "pareto_front": front_labels, "environment": environment(),
           "seconds": time.time() - t_start}
    out_dir = pathlib.Path(S) / "wide"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ds.name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"== {ds.name} done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
