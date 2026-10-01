"""Does measuring similarity after removing the corpus mean fix AdaptiveMerge on anisotropic encoders?"""
import json
import os
import sys
import time
from pathlib import Path

from optivision.benchmark import Dataset, load_legacy_dataset, run_matrix, save_result
from optivision.compose import Pipeline
from optivision.representation import MultiVectorCorpus
from optivision.stages import AdaptiveMerge

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
LEGACY = {
    "E1-colsmol-generated": (D + "/cache/colsmol.npz", D + "/corpus/queries.json"),
    "E3-colpali-generated": (D + "/cache/colpali_generated.npz", D + "/corpus/queries.json"),
    "E2-colpali-infovqa": (D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json"),
    "E2-colpali-docvqa": (D + "/cache/colpali_docvqa_test_subsampled.npz", D + "/vidore_docvqa_test_subsampled/queries.json"),
}


def dataset(name):
    if name == "text-scifact":
        docs = MultiVectorCorpus.load(f"{D}/vectors/scifact_answerai-colbert-small-v1_docs.npz")
        qs = MultiVectorCorpus.load(f"{D}/vectors/scifact_answerai-colbert-small-v1_queries.npz")
        qrels = {k: set(v) for k, v in json.loads(Path(f"{D}/vectors/scifact_qrels_test.json").read_text(encoding="utf-8")).items()}
        return Dataset("text-answerai-colbert-small-v1-scifact", docs, qs, qrels)
    cache, qj = LEGACY[name]
    return load_legacy_dataset(cache, qj, name=name, with_images=False)


for name in sys.argv[1:]:
    t0 = time.time()
    ds = dataset(name)
    pipes = {f"adaptive radius={r} centred": Pipeline([AdaptiveMerge(radius=r, center="mean")])
             for r in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)}
    pipes.update({f"adaptive ratio={f} centred": Pipeline([AdaptiveMerge(radius=None, ratio=f, center="mean")])
                  for f in (0.5, 0.25, 0.1)})
    show = lambda label, row: print(f"  {label:34s} vec/doc {row['vectors_per_doc']:7.1f} x{row['compression_vs_float32']:6.1f} "
                                    f"ret@5 {row['retention:ndcg@5']:.3f} [{row['retention_ci:ndcg@5'][0]:.3f},"
                                    f"{row['retention_ci:ndcg@5'][1]:.3f}] ret@10 {row['retention:ndcg@10']:.3f}", flush=True)
    print(f"== {name}", flush=True)
    res = run_matrix(ds, pipes, metrics=["ndcg@5", "ndcg@10", "recall@10", "mrr@10"], progress=show)
    save_result(res, f"{S}/centred/results", ds.name)
    print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)
