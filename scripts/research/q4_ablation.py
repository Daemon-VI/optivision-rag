"""Q4: axis ablation on the 128-d encoders (reports/research/Q4/PLAN.md).

For one ViDoRe dataset (model x split), compresses the documents with every
configuration of the fixed Q4 grid through the library's own path
(calibration._compress, the same call score_candidate makes), scores all queries
with maxsim_matrix, and records per-query nDCG@5 plus the actual resident arrays:
codes, offsets, shared codec state and the projection basis. Configurations that
coincide with the E1 store's 40 candidates are checked equal to it per query
(exit 1 otherwise). The two R12 two-tier rows are rerun as well.

Writes reports/research/Q4/store/<dataset>.npz (per-query nDCG@5, one row per
configuration) and reports/research/Q4/store/<dataset>.json (rows, accounting,
checks). Timings here are incidental; latency is measured by q4_latency.py.

    OPTIVISION_DATA=../optivision-rag-v2/data OPTIVISION_CACHE_BYTES=1500000000 \
        PYTHONPATH="src;scripts/research" python scripts/research/q4_ablation.py colqwen2_docvqa
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import e1_common as E
import numpy as np

from optivision.calibration import _compress
from optivision.compose import Pipeline, save_compressed
from optivision.evaluation import per_query_metrics, retention
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

OUT = E.ROOT / "reports" / "research" / "Q4" / "store"
METRIC = "ndcg@5"
CODECS = {
    "float16": Float16Quantizer, "int8": lambda: Int8Quantizer("per_vector"), "int4": Int4Quantizer,
    "int4c": lambda: Int4Quantizer(center="mean"), "2bit": Lloyd2Quantizer, "binary": BinaryQuantizer,
}
ALL_STORES = ("colpali_docvqa", "colpali_infovqa", "colqwen2_docvqa", "colqwen2_infovqa",
              "colqwen25_docvqa", "colqwen25_infovqa")


def W(r):
    return HierarchicalMerge(ratio=r)


def P(w, method="pca"):
    return DimensionProjector(w, method) if method != "pca" else DimensionProjector(w)


def C(name):
    return CODECS[name]()


def grid() -> list[tuple[str, list]]:
    """(group, stages) in PLAN.md §4 order; stage prefixes kept contiguous for the cache."""
    g: list[tuple[str, list]] = [("baseline", [])]
    g += [("A", [W(r)]) for r in (0.5, 0.33, 0.25, 0.125, 0.1)]
    g += [("A-alt", [AdaptiveMerge(radius=0.6, center="mean")])]
    g += [("B", [P(w)]) for w in (64, 32, 16, 8)]
    g += [("B-control", [P(32, "random")])]
    g += [("C", [C(c)]) for c in CODECS]
    for r in (0.33, 0.25):
        g += [("A+C", [W(r), C(c)]) for c in CODECS]
    g += [("A+C", [W(0.5), C("int8")])]
    for w in (64, 32, 16, 8):
        g += [("B+C", [P(w), C(c)]) for c in ("float16", "int8", "int4c", "binary")]
    for r in (0.33, 0.25):
        for w in (64, 32, 16):
            g += [("A+B", [W(r), P(w)])]
            g += [("A+B+C", [W(r), P(w), C(c)]) for c in ("int8", "int4c", "binary")]
    g += [("A+B", [W(0.5), P(32)])]
    g += [("A+B", [W(0.5), P(64)]), ("A+B+C", [W(0.5), P(64), C("float16")])]
    return g


def accounting(pipe: Pipeline, comp, n_docs: int, fp32_bytes: int) -> dict:
    basis = 0
    for s in pipe.vector_stages:
        inner = getattr(s, "inner", None)
        for a in ("components", "mean", "matrix"):
            v = getattr(inner, a, None) if inner is not None else None
            if isinstance(v, np.ndarray) and not (a == "mean" and not getattr(inner, "center", False)):
                basis += v.nbytes
    code = int(comp.codes.nbytes)
    shared = int(comp.quantizer.overhead_bytes()) + basis
    index = code + int(comp.offsets.nbytes) + shared
    disk = None
    if not comp.quantizer.overhead_bytes():
        with tempfile.TemporaryDirectory() as d:
            path = save_compressed(comp, Path(d) / "x.npz")
            disk = path.stat().st_size + basis
    rep = comp.report()
    return {"vectors": int(comp.num_vectors), "vectors_per_doc": comp.num_vectors / n_docs, "dim": int(comp.dim),
            "bits_per_dim": float(rep["bits_per_dim"]), "bits_per_vector": 8 * code / max(1, comp.num_vectors),
            "code_bytes": code, "offsets_bytes": int(comp.offsets.nbytes), "codec_state_bytes": int(comp.quantizer.overhead_bytes()),
            "projection_basis_bytes": basis, "index_bytes": index, "index_bytes_per_doc": index / n_docs,
            "code_bytes_per_doc": code / n_docs, "library_bytes_per_doc": float(rep["bytes_per_doc"]),
            "disk_bytes": disk, "disk_bytes_per_doc": disk / n_docs if disk is not None else None,
            "persistable_by_library": disk is not None,
            "compression_vs_float32_codes": fp32_bytes / code, "compression_vs_float32_index": fp32_bytes / index,
            "library_compression_vs_float32": float(rep["compression_vs_float32"])}


def main(name: str) -> int:
    t_start = time.time()
    s7b = E.load_s7b()
    ds, metric = s7b.dataset(E.dataset_arg(name))
    assert metric == METRIC
    corpus, queries, rel = ds.corpus, ds.queries, ds.relevant()
    n_docs, n_q = len(corpus), len(queries)
    if corpus.dimension != 128:
        raise SystemExit("q4_ablation is for the 128-d encoders")
    fp32 = corpus.num_vectors * corpus.dimension * 4
    store = E.Store(E.load_store(name), "labels")
    base = per_query_metrics(maxsim_matrix(queries, corpus), rel, [METRIC])[METRIC]
    checks = {"baseline_equals_E1_store": bool(np.array_equal(base, store.base))}
    cache: dict = {}
    rows, pq_all = [], []
    for group, stages in grid():
        pipe = Pipeline(stages)
        t0 = time.perf_counter()
        comp = _compress(pipe, corpus, cache)
        t1 = time.perf_counter()
        pq = per_query_metrics(maxsim_matrix(pipe.transform_queries(queries), comp), rel, [METRIC])[METRIC]
        t2 = time.perf_counter()
        point, lo, hi = retention(pq, base, n_boot=1000)
        label = pipe.label() if stages else "float32"
        row = {"group": group, "label": label, "pipeline": pipe.to_dict(), "ndcg@5": float(pq.mean()),
               "retention": point, "retention_ci": [lo, hi], "quality_loss": 1 - point,
               **accounting(pipe, comp, n_docs, fp32),
               "incidental_compress_seconds_cached": t1 - t0, "incidental_scan_seconds": t2 - t1}
        if label in store.labels:
            k = store.labels.index(label)
            row["equals_E1_store"] = bool(np.array_equal(pq, store.pq[k]))
            checks[f"{label}_equals_E1_store"] = row["equals_E1_store"]
        rows.append(row)
        pq_all.append(pq)
        print(f"{group:6s} {label:55s} v/p {row['vectors_per_doc']:7.1f} d {row['dim']:3d} "
              f"x{row['compression_vs_float32_index']:7.1f} R {point:.4f}", flush=True)
    # two tiers (R12 parity): binary codes in RAM, 50-page shortlist rescored with int8 of every vector
    cold = Pipeline([Int8Quantizer("per_vector")]).compress(corpus)
    tiered = []
    for hot_label, hot_pipe in (("binary", Pipeline([BinaryQuantizer()])),
                                ("hierarchical_merge(0.25) > binary", Pipeline([W(0.25), BinaryQuantizer()]))):
        hot = hot_pipe.compress(corpus)
        idx = TieredIndex(ExactIndex(hot), ExactIndex(cold), candidates=50)
        pq = per_query_metrics(idx.score(queries), rel, [METRIC])[METRIC]
        point, lo, hi = retention(pq, base, n_boot=1000)
        ram = accounting(hot_pipe, hot, n_docs, fp32)["index_bytes"]
        disk = ram + accounting(Pipeline([Int8Quantizer("per_vector")]), cold, n_docs, fp32)["index_bytes"]
        tiered.append({"hot": hot_label, "cold": "int8(per_vector), all vectors", "candidates": 50,
                       "ram_index_bytes_per_doc": ram / n_docs, "disk_index_bytes_per_doc": disk / n_docs,
                       "ram_compression_vs_float32": fp32 / ram, "disk_compression_vs_float32": fp32 / disk,
                       "retention": point, "retention_ci": [lo, hi]})
        pq_all.append(pq)
        rows.append({"group": "tiered", "label": f"tiered[{hot_label} -> int8, 50]", "retention": point,
                     "retention_ci": [lo, hi], "ndcg@5": float(pq.mean())})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", base=base, pq=np.stack(pq_all),
                        labels=np.array([r["label"] for r in rows]))
    out = {"dataset": ds.name, "store": name, "n_docs": n_docs, "n_queries": n_q, "dim": corpus.dimension,
           "baseline": {"ndcg@5": float(base.mean()), "vectors": int(corpus.num_vectors),
                        "vectors_per_doc": corpus.num_vectors / n_docs, "float32_code_bytes": fp32,
                        "float32_code_bytes_per_doc": fp32 / n_docs},
           "compression_baseline": "float32 in-memory codes of the original vectors (num_vectors x d x 4 bytes)",
           "rows": rows, "tiered": tiered, "checks": checks, "environment": E.environment(),
           "seconds": time.time() - t_start}
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    ok = all(checks.values())
    print(name, "checks", "OK" if ok else "MISMATCH", sum(checks.values()), "/", len(checks),
          f"{time.time() - t_start:.0f}s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
