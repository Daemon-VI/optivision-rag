"""Pareto frontier: token stage x codec grid, each token stage computed once.

Retention is against the ORIGINAL float32 baseline (labels, nDCG@5 for ViDoRe; nDCG@5 and @10 for text),
bytes are stored bytes of the final codes, compression is original float32 bytes / stored bytes.
"""
import json
import os
import pathlib
import sys
import time
from pathlib import Path

import numpy as np

from optivision.benchmark import Dataset, load_legacy_dataset, load_vector_dataset
from optivision.compose import Pipeline
from optivision.evaluation import per_query_metrics, retention
from optivision.pareto import pareto_front
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
)

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
LEGACY = {
    "E2-colpali-infovqa": (D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json"),
    "E2-colpali-docvqa": (D + "/cache/colpali_docvqa_test_subsampled.npz", D + "/vidore_docvqa_test_subsampled/queries.json"),
}
METRICS = ["ndcg@5", "ndcg@10"]


def dataset(name):
    if name.startswith("vec:"):  # files written by scripts/encode_vectors.py
        return load_vector_dataset(f"{D}/vectors/{name[4:]}", name=name[4:])
    if name == "text-scifact":
        docs = MultiVectorCorpus.load(f"{D}/vectors/scifact_answerai-colbert-small-v1_docs.npz")
        qs = MultiVectorCorpus.load(f"{D}/vectors/scifact_answerai-colbert-small-v1_queries.npz")
        qrels = {k: set(v) for k, v in json.loads(Path(f"{D}/vectors/scifact_qrels_test.json").read_text(encoding="utf-8")).items()}
        return Dataset("text-answerai-colbert-small-v1-scifact", docs, qs, qrels)
    cache, qj = LEGACY[name]
    return load_legacy_dataset(cache, qj, name=name, with_images=False)


def token_stages(name):
    stages = {"none": None}
    for t in (0.8, 1.0, 1.3, 1.6, 2.0):
        stages[f"ward d={t}"] = HierarchicalMerge(ratio=None, max_distance=t)
    for f in (0.5, 0.33, 0.25):
        stages[f"ward ratio={f}"] = HierarchicalMerge(ratio=f)
    for r in (0.7, 0.6, 0.5):
        stages[f"adaptive r={r}"] = AdaptiveMerge(radius=r)
    return stages


CODECS = {
    "float32": None,
    "float16": Float16Quantizer,
    "int8/vec": lambda: Int8Quantizer("per_vector"),
    "int4": Int4Quantizer,
    "binary": BinaryQuantizer,
}


def run(name):
    t0 = time.time()
    ds = dataset(name)
    corpus, queries = ds.corpus, ds.queries
    rel = ds.relevant()
    base_scores = maxsim_matrix(queries, corpus)
    base = per_query_metrics(base_scores, rel, METRICS)
    float_bytes = corpus.num_vectors * corpus.dimension * 4
    rows = []
    print(f"== {name}: {len(corpus)} docs, {len(queries)} queries", flush=True)
    for tname, stage in token_stages(name).items():
        t1 = time.time()
        merged = corpus if stage is None else stage.fit(corpus).transform(corpus)
        merge_s = time.time() - t1
        for cname, codec in CODECS.items():
            pipe = Pipeline([] if codec is None else [codec()])
            comp = pipe.compress(merged)
            scores = maxsim_matrix(queries, comp)
            pq = per_query_metrics(scores, rel, METRICS)
            row = {"token_stage": tname, "codec": cname, "label": f"{tname} > {cname}",
                   "pipeline": {"stages": ([] if stage is None else [stage.to_dict()]) + pipe.to_dict()["stages"]},
                   "vectors_per_doc": comp.num_vectors / len(comp), "bytes_per_doc": comp.nbytes / len(comp),
                   "compression": float_bytes / comp.nbytes, "merge_seconds": merge_s}
            for m in METRICS:
                point, lo, hi = retention(pq[m], base[m], n_boot=1000)
                row[f"retention:{m}"] = point
                row[f"ci:{m}"] = [lo, hi]
            rows.append(row)
            print(f"  {row['label']:28s} vec/doc {row['vectors_per_doc']:7.1f}  {row['bytes_per_doc'] / 1e3:8.2f} KB  "
                  f"x{row['compression']:7.1f}  ret@5 {row['retention:ndcg@5']:.3f} [{row['ci:ndcg@5'][0]:.3f},"
                  f"{row['ci:ndcg@5'][1]:.3f}]  ret@10 {row['retention:ndcg@10']:.3f}", flush=True)
    key = "retention:ndcg@10" if name.startswith("text") else "retention:ndcg@5"
    for r in rows:
        r["retention"] = r[key]
    front = pareto_front(rows, costs=("bytes_per_doc",), gain="retention")
    print("  -- Pareto frontier (" + key + ")", flush=True)
    for r in front:
        print(f"     {r['label']:28s} x{r['compression']:7.1f}  retention {r['retention']:.3f}", flush=True)
    out = {"dataset": ds.name, "n_docs": len(corpus), "n_queries": len(queries), "frontier_metric": key,
           "baseline": {m: float(np.nanmean(base[m])) for m in METRICS}, "rows": rows,
           "frontier": [r["label"] for r in front], "seconds": time.time() - t0}
    pathlib.Path(f"{S}/frontier/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{S}/frontier/results/{ds.name}.json").write_text(json.dumps(out, indent=1, default=float),
                                                                     encoding="utf-8")
    print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
