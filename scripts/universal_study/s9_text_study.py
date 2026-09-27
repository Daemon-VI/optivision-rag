"""Phase 9: the universal layer on a text late-interaction model (answerai-colbert-small-v1, BEIR SciFact).

Same measurements as for ColPali: token reduction and codecs against the float baseline (labels),
then the repeated-split calibration study.
"""
import json
import os
import pathlib
import sys
import time
from pathlib import Path

import numpy as np

from optivision.benchmark import Dataset, run_matrix, save_result
from optivision.calibration import default_search_space, score_space, select
from optivision.compose import Pipeline
from optivision.evaluation import (
    per_query_metrics,
    relevant_from_baseline,
    retention,
    split_queries,
)
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    RandomPruner,
    RedundancyPruner,
)

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
tag = sys.argv[1] if len(sys.argv) > 1 else "answerai-colbert-small-v1"
docs = MultiVectorCorpus.load(f"{D}/vectors/scifact_{tag}_docs.npz")
queries = MultiVectorCorpus.load(f"{D}/vectors/scifact_{tag}_queries.npz")
qrels = {k: set(v) for k, v in json.loads(Path(f"{D}/vectors/scifact_qrels_test.json").read_text(encoding="utf-8")).items()}
ds = Dataset(f"text-{tag}-scifact", docs, queries, qrels, attrs=dict(docs.attrs))
print(f"== {ds.name}: {len(docs)} docs ({docs.num_vectors} vectors, dim {docs.dimension}), {len(queries)} queries", flush=True)

show = lambda label, row: print(f"  {label:34s} vec/doc {row['vectors_per_doc']:6.1f} bits {row['bits_per_dim']:5.3g} "
                                f"x{row['compression_vs_float32']:7.1f} nDCG@10 {row['ndcg@10']:.4f} "
                                f"ret@5 {row['retention:ndcg@5']:.3f} [{row['retention_ci:ndcg@5'][0]:.3f},"
                                f"{row['retention_ci:ndcg@5'][1]:.3f}] ret@10 {row['retention:ndcg@10']:.3f}", flush=True)
pipes = {}
for r in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6):
    pipes[f"adaptive radius={r}"] = Pipeline([AdaptiveMerge(radius=r)])
for f in (0.5, 0.33, 0.25):
    pipes[f"hierarchical ratio={f}"] = Pipeline([HierarchicalMerge(ratio=f)])
    pipes[f"random ratio={f}"] = Pipeline([RandomPruner(ratio=f)])
for t in (0.92, 0.85):
    pipes[f"redundancy t={t}"] = Pipeline([RedundancyPruner(threshold=t)])
pipes.update({
    "float16": Pipeline([Float16Quantizer()]),
    "int8 fixed (legacy)": Pipeline([Int8Quantizer("fixed")]),
    "int8 per-vector": Pipeline([Int8Quantizer("per_vector")]),
    "int4 per-vector": Pipeline([Int4Quantizer()]),
    "lloyd2 (2-bit)": Pipeline([Lloyd2Quantizer()]),
    "binary": Pipeline([BinaryQuantizer()]),
    "adaptive 0.8 > int8 per-vector": Pipeline([AdaptiveMerge(radius=0.8), Int8Quantizer("per_vector")]),
    "adaptive 0.8 > binary": Pipeline([AdaptiveMerge(radius=0.8), BinaryQuantizer()]),
})
t0 = time.time()
res = run_matrix(ds, pipes, metrics=["ndcg@5", "ndcg@10", "recall@10", "mrr@10"], progress=show)
save_result(res, f"{S}/phase9/results", ds.name)
print(f"  matrix done in {time.time() - t0:.0f}s", flush=True)

# calibration study, same protocol as ColPali (metric nDCG@10, the BEIR convention)
METRIC = "ndcg@10"
base = maxsim_matrix(queries, docs)
rel_labels = ds.relevant()
rel_base = relevant_from_baseline(base, 1)
scored = score_space(docs, queries, default_search_space(docs.dimension),
                     progress=lambda sc: print(f"  scored {sc.label:36s} {sc.bytes_per_doc / 1e3:7.2f} KB/doc", flush=True))
by_json = {json.dumps(sc.pipeline, sort_keys=True): sc for sc in scored}


def ret_on(scores, rel, idx):
    r = [rel[i] for i in idx]
    return retention(per_query_metrics(scores[idx], r, [METRIC])[METRIC],
                     per_query_metrics(base[idx], r, [METRIC])[METRIC], n_boot=0)[0]


study = []
for reference, rel in (("labels", rel_labels), ("baseline@1", rel_base)):
    for safety in ("point", "lower_ci"):
        for target in (0.99, 0.97, 0.95):
            outcomes = []
            for seed in range(20):
                cal, hold = split_queries(len(queries), 0.5, seed)
                _, chosen, ho = select(scored, base, rel, cal, hold, target, METRIC, safety=safety,
                                       n_boot=300 if safety == "lower_ci" else 0, seed=seed)
                if chosen is None:
                    outcomes.append({"seed": seed, "selected": None})
                    continue
                sc = by_json[json.dumps(chosen.pipeline, sort_keys=True)]
                outcomes.append({"seed": seed, "selected": chosen.label, "compression": chosen.compression_vs_float32,
                                 "bytes_per_doc": chosen.bytes_per_doc, "cal_retention": chosen.retention,
                                 "holdout_retention": ho["retention"],
                                 "holdout_retention_labels": ret_on(sc.scores, rel_labels, hold)})
            ok = [o for o in outcomes if o["selected"]]
            s = {"reference": reference, "safety": safety, "target": target, "splits": 20, "selected_any": len(ok),
                 "met_on_holdout": float(np.mean([o["holdout_retention"] >= target for o in ok])) if ok else None,
                 "met_on_holdout_labels": float(np.mean([o["holdout_retention_labels"] >= target for o in ok]))
                 if ok else None,
                 "holdout_retention_mean": float(np.mean([o["holdout_retention"] for o in ok])) if ok else None,
                 "holdout_retention_min": float(np.min([o["holdout_retention"] for o in ok])) if ok else None,
                 "holdout_labels_mean": float(np.mean([o["holdout_retention_labels"] for o in ok])) if ok else None,
                 "compression_median": float(np.median([o["compression"] for o in ok])) if ok else None,
                 "bytes_per_doc_median": float(np.median([o["bytes_per_doc"] for o in ok])) if ok else None,
                 "selections": {k: sum(1 for o in ok if o["selected"] == k) for k in {o["selected"] for o in ok}},
                 "outcomes": outcomes}
            study.append(s)
            print(f"  {reference:10s} {safety:8s} target {target:.2f}: met {s['met_on_holdout']} (labels "
                  f"{s['met_on_holdout_labels']}), holdout mean {s['holdout_retention_mean']}, min "
                  f"{s['holdout_retention_min']}, median x{s['compression_median']}", flush=True)
out = {"dataset": ds.name, "n_docs": len(docs), "n_queries": len(queries), "metric": METRIC,
       "baseline_ndcg10": float(np.nanmean(per_query_metrics(base, rel_labels, [METRIC])[METRIC])),
       "study": study, "pseudo_query_proxy": [], "fixed": [],
       "candidates": [{"label": sc.label, "bytes_per_doc": sc.bytes_per_doc,
                       "compression": sc.report["compression_vs_float32"],
                       "retention_labels": ret_on(sc.scores, rel_labels, np.arange(len(queries))),
                       "retention_baseline": ret_on(sc.scores, rel_base, np.arange(len(queries)))} for sc in scored]}
pathlib.Path(f"{S}/phase9/results/{ds.name}-calibration.json").write_text(json.dumps(out, indent=1, default=float),
                                                                           encoding="utf-8")
print(f"== done in {time.time() - t0:.0f}s", flush=True)
