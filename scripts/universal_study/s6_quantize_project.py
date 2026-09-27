"""Phases 6-7: quantizers alone, projections alone, and combinations (exact MaxSim, labels).

Projection bases fitted on queries are fitted on one half of the queries and evaluated on the
other half only (no leakage). Everything else is evaluated on all queries.
"""
import os
import sys
import time

from optivision.benchmark import load_legacy_dataset, run_matrix, save_result
from optivision.compose import Pipeline
from optivision.evaluation import split_queries
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RandomProjector,
    TruncateProjector,
)

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
DATASETS = {
    "E2-colpali-infovqa": (D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json"),
    "E2-colpali-docvqa": (D + "/cache/colpali_docvqa_test_subsampled.npz", D + "/vidore_docvqa_test_subsampled/queries.json"),
    "E3-colpali-generated": (D + "/cache/colpali_generated.npz", D + "/corpus/queries.json"),
    "E1-colsmol-generated": (D + "/cache/colsmol.npz", D + "/corpus/queries.json"),
}
METRICS = ["ndcg@5", "ndcg@10", "recall@5", "mrr@10"]


def quant_and_proj():
    p = {
        "float16": Pipeline([Float16Quantizer()]),
        "int8 fixed (legacy)": Pipeline([Int8Quantizer("fixed")]),
        "int8 per-vector": Pipeline([Int8Quantizer("per_vector")]),
        "int8 per-dimension": Pipeline([Int8Quantizer("per_dimension")]),
        "int4 per-vector": Pipeline([Int4Quantizer()]),
        "lloyd2 (2-bit)": Pipeline([Lloyd2Quantizer(seed=7)]),
        "lloyd2 (2-bit, seed 1)": Pipeline([Lloyd2Quantizer(seed=1)]),
        "lloyd2 (2-bit, seed 2)": Pipeline([Lloyd2Quantizer(seed=2)]),
        "binary": Pipeline([BinaryQuantizer()]),
        "binary centred": Pipeline([BinaryQuantizer(center="mean")]),
    }
    for d in (96, 64, 48, 32):
        p[f"pca-docs {d}"] = Pipeline([PCAProjector(dim=d)])
    for d in (96, 64, 32):
        p[f"random-proj {d}"] = Pipeline([RandomProjector(dim=d, seed=0)])
        p[f"truncate {d}"] = Pipeline([TruncateProjector(dim=d)])
    return p


def merge_variants():
    p = {}
    for t in (0.6, 0.8, 1.0, 1.3, 1.6):
        p[f"ward distance={t} (per-doc count)"] = Pipeline([HierarchicalMerge(ratio=None, max_distance=t)])
    for r in (0.8, 0.7, 0.6):
        p[f"adaptive radius={r} refine=3"] = Pipeline([AdaptiveMerge(radius=r, refine=3)])
    return p


def combos(radius):
    m = lambda: AdaptiveMerge(radius=radius)
    return {
        f"merge {radius} > int8 per-vector": Pipeline([m(), Int8Quantizer("per_vector")]),
        f"merge {radius} > int4": Pipeline([m(), Int4Quantizer()]),
        f"merge {radius} > lloyd2": Pipeline([m(), Lloyd2Quantizer(seed=7)]),
        f"merge {radius} > binary": Pipeline([m(), BinaryQuantizer()]),
        f"merge {radius} > binary centred": Pipeline([m(), BinaryQuantizer(center="mean")]),
        f"merge {radius} > pca-docs 64 > int8 per-vector": Pipeline([m(), PCAProjector(dim=64), Int8Quantizer("per_vector")]),
        f"merge {radius} > pca-docs 64 > binary": Pipeline([m(), PCAProjector(dim=64), BinaryQuantizer()]),
    }


def joint_pca(ds):
    """Fit PCA on documents + calibration-half queries; evaluate on the held-out half only."""
    cal, hold = split_queries(len(ds.queries), 0.5, 0)
    q_cal = ds.queries.select(cal)
    held = ds.select_queries(hold)
    pipes = {}
    for d in (64, 32):
        joint = PCAProjector(dim=d, basis="joint", query_weight=0.5)
        joint.fit(ds.corpus, q_cal)
        pipes[f"pca-joint {d} (fit on cal queries)"] = Pipeline([joint])
        pipes[f"pca-docs {d} (same held-out queries)"] = Pipeline([PCAProjector(dim=d)])
    return held, pipes


def run(name, radius):
    t0 = time.time()
    cache, qj = DATASETS[name]
    ds = load_legacy_dataset(cache, qj, name=name, with_images=False)
    show = lambda label, row: print(f"  {label:44s} vec/doc {row['vectors_per_doc']:7.1f} dim {row['dim']:4d} "
                                    f"bits {row['bits_per_dim']:5.3g} x{row['compression_vs_float32']:7.1f} "
                                    f"nDCG@5 {row['ndcg@5']:.4f} ret {row['retention:ndcg@5']:.3f} "
                                    f"[{row['retention_ci:ndcg@5'][0]:.3f},{row['retention_ci:ndcg@5'][1]:.3f}]", flush=True)
    print(f"== {name}", flush=True)
    res = run_matrix(ds, {**quant_and_proj(), **merge_variants(), **combos(radius)}, metrics=METRICS, progress=show)
    save_result(res, S + "/phase67/results", name)
    held, pipes = joint_pca(ds)
    print(f"  -- joint-basis PCA, held-out half ({len(held.queries)} queries)", flush=True)
    res2 = run_matrix(held, pipes, metrics=METRICS, progress=show)
    save_result(res2, S + "/phase67/results", name + "-joint-pca")
    print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    radius = float(sys.argv[1])
    for n in (sys.argv[2:] or list(DATASETS)):
        run(n, radius)
