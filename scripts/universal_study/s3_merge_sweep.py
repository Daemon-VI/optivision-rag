import os

"""Phase 3: token reduction methods at matched budgets, float32 (no quantization), exact MaxSim."""
import sys
import time

from optivision.benchmark import load_legacy_dataset, run_matrix, save_result
from optivision.compose import Pipeline
from optivision.stages import (
    AdaptiveMerge,
    HierarchicalMerge,
    RandomPruner,
    RedundancyPruner,
    SpatialPruner,
)

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
DATASETS = {
    "E1-colsmol-generated": (D + "/cache/colsmol.npz", D + "/corpus/queries.json"),
    "E3-colpali-generated": (D + "/cache/colpali_generated.npz", D + "/corpus/queries.json"),
    "E2-colpali-infovqa": (D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json"),
    "E2-colpali-docvqa": (D + "/cache/colpali_docvqa_test_subsampled.npz", D + "/vidore_docvqa_test_subsampled/queries.json"),
}
def pipelines():
    p = {"spatial+redundancy (legacy)": Pipeline([SpatialPruner(), RedundancyPruner()])}
    for t in (0.95, 0.92, 0.88, 0.85, 0.80, 0.75):
        p[f"redundancy t={t}"] = Pipeline([RedundancyPruner(threshold=t)])
    for r in (0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5):
        p[f"adaptive radius={r}"] = Pipeline([AdaptiveMerge(radius=r)])
    for f in (0.5, 0.33, 0.25, 0.2, 0.1, 0.05):
        p[f"adaptive ratio={f}"] = Pipeline([AdaptiveMerge(radius=None, ratio=f)])
        p[f"hierarchical ratio={f}"] = Pipeline([HierarchicalMerge(ratio=f)])
    for f in (0.5, 0.25, 0.1):
        p[f"random ratio={f}"] = Pipeline([RandomPruner(ratio=f, seed=0)])
    return p
names = sys.argv[1:] or list(DATASETS)
for name in names:
    cache, qj = DATASETS[name]
    t = time.time()
    ds = load_legacy_dataset(cache, qj, name=name)
    def progress(label, row):
        print(f"  {label:32s} vec/doc {row['vectors_per_doc']:7.1f}  x{row['compression_vs_float32']:5.2f}  nDCG@5 {row['ndcg@5']:.4f}  ret {row['retention:ndcg@5']:.3f} [{row['retention_ci:ndcg@5'][0]:.3f},{row['retention_ci:ndcg@5'][1]:.3f}]  compress {row['compress_seconds']:.1f}s", flush=True)
    print(name, flush=True)
    res = run_matrix(ds, pipelines(), metrics=["ndcg@5", "ndcg@10", "recall@5", "mrr@10"], progress=progress)
    save_result(res, S + "/phase3/results", name)
    print(f"{name} done in {time.time()-t:.0f}s", flush=True)
