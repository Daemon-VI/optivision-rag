"""Phase 4 proof experiment: does calibration pick small configs that hold on unseen queries?

For each dataset: score every candidate of the default search space once against all
queries, then repeat the calibrate/hold-out split 20 times per (target, reference, safety)
and record the held-out retention of whatever was selected. Also: the fixed configurations
that exist today, and whether pseudo-queries (document fragments) predict real-query quality.
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

from optivision.benchmark import load_legacy_dataset
from optivision.calibration import (
    default_search_space,
    pseudo_queries,
    score_candidate,
    score_space,
    select,
)
from optivision.compose import Pipeline
from optivision.evaluation import (
    per_query_metrics,
    relevant_from_baseline,
    relevant_from_qrels,
    retention,
    split_queries,
)
from optivision.scoring import maxsim_matrix
from optivision.stages import BinaryQuantizer, Int8Quantizer, RedundancyPruner, SpatialPruner

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
DATASETS = {
    "E2-colpali-infovqa": (D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json"),
    "E2-colpali-docvqa": (D + "/cache/colpali_docvqa_test_subsampled.npz", D + "/vidore_docvqa_test_subsampled/queries.json"),
    "E3-colpali-generated": (D + "/cache/colpali_generated.npz", D + "/corpus/queries.json"),
    "E1-colsmol-generated": (D + "/cache/colsmol.npz", D + "/corpus/queries.json"),
}
FIXED = {
    "int8-only": Pipeline([Int8Quantizer()]),
    "binary-only": Pipeline([BinaryQuantizer()]),
    "prune+int8": Pipeline([SpatialPruner(), RedundancyPruner(), Int8Quantizer()]),
    "optivision (prune+binary)": Pipeline([SpatialPruner(), RedundancyPruner(), BinaryQuantizer()]),
}
TARGETS = (0.99, 0.97, 0.95)
SEEDS = range(20)
METRIC = "ndcg@5"


def ret_on(scores, base, rel, idx):
    r = [rel[i] for i in idx]
    return retention(per_query_metrics(scores[idx], r, [METRIC])[METRIC],
                     per_query_metrics(base[idx], r, [METRIC])[METRIC], n_boot=0)[0]


def run(name):
    t0 = time.time()
    cache, qj = DATASETS[name]
    ds = load_legacy_dataset(cache, qj, name=name)
    corpus, queries = ds.corpus, ds.queries
    everyone = np.arange(len(queries))
    print(f"== {name}: {len(corpus)} docs, {len(queries)} queries", flush=True)
    base = maxsim_matrix(queries, corpus)
    rel_labels = ds.relevant()
    rel_base = relevant_from_baseline(base, 1)
    space = default_search_space(corpus.dimension)

    def show(sc):
        print(f"  scored {sc.label:40s} {sc.bytes_per_doc / 1e3:8.2f} KB/doc  vec/doc "
              f"{sc.report['vectors_per_doc']:7.1f}  {sc.compress_seconds + sc.score_seconds:5.1f}s", flush=True)

    scored = score_space(corpus, queries, space, progress=show)
    by_json = {json.dumps(sc.pipeline, sort_keys=True): sc for sc in scored}
    table = [{"label": sc.label, "family": sc.family, "bytes_per_doc": sc.bytes_per_doc,
              "vectors_per_doc": sc.report["vectors_per_doc"], "compression": sc.report["compression_vs_float32"],
              "retention_labels": ret_on(sc.scores, base, rel_labels, everyone),
              "retention_baseline": ret_on(sc.scores, base, rel_base, everyone),
              "compress_seconds": sc.compress_seconds, "query_ms": sc.query_ms} for sc in scored]

    fixed_rows = []
    for fname, pipe in FIXED.items():
        sc = score_candidate(pipe, corpus, queries, label=fname)
        fixed_rows.append({"label": fname, "bytes_per_doc": sc.bytes_per_doc,
                           "compression": sc.report["compression_vs_float32"],
                           "vectors_per_doc": sc.report["vectors_per_doc"],
                           "retention_labels": ret_on(sc.scores, base, rel_labels, everyone),
                           "retention_baseline": ret_on(sc.scores, base, rel_base, everyone)})
        print(f"  fixed  {fname:30s} {sc.bytes_per_doc / 1e3:8.2f} KB/doc  ret(labels) "
              f"{fixed_rows[-1]['retention_labels']:.3f}", flush=True)

    study = []
    for reference, rel in (("labels", rel_labels), ("baseline@1", rel_base)):
        for safety in ("point", "lower_ci"):
            for target in TARGETS:
                outcomes = []
                for seed in SEEDS:
                    cal, hold = split_queries(len(queries), 0.5, seed)
                    _, chosen, ho = select(scored, base, rel, cal, hold, target, METRIC, safety=safety,
                                           n_boot=300 if safety == "lower_ci" else 0, seed=seed)
                    if chosen is None:
                        outcomes.append({"seed": seed, "selected": None})
                        continue
                    sc = by_json[json.dumps(chosen.pipeline, sort_keys=True)]
                    outcomes.append({"seed": seed, "selected": chosen.label, "bytes_per_doc": chosen.bytes_per_doc,
                                     "compression": chosen.compression_vs_float32,
                                     "cal_retention": chosen.retention, "holdout_retention": ho["retention"],
                                     "holdout_retention_labels": ret_on(sc.scores, base, rel_labels, hold)})
                ok = [o for o in outcomes if o["selected"]]

                def agg(fn, key, ok=ok):
                    return float(fn([o[key] for o in ok])) if ok else None

                summary = {
                    "reference": reference, "safety": safety, "target": target, "splits": len(outcomes),
                    "selected_any": len(ok),
                    "met_on_holdout": float(np.mean([o["holdout_retention"] >= target for o in ok])) if ok else None,
                    "met_on_holdout_labels": float(np.mean([o["holdout_retention_labels"] >= target for o in ok]))
                    if ok else None,
                    "holdout_retention_mean": agg(np.mean, "holdout_retention"),
                    "holdout_retention_min": agg(np.min, "holdout_retention"),
                    "holdout_labels_mean": agg(np.mean, "holdout_retention_labels"),
                    "compression_median": agg(np.median, "compression"),
                    "bytes_per_doc_median": agg(np.median, "bytes_per_doc"),
                    "selections": {k: sum(1 for o in ok if o["selected"] == k) for k in {o["selected"] for o in ok}},
                    "outcomes": outcomes,
                }
                study.append(summary)
                print(f"  {reference:10s} {safety:8s} target {target:.2f}: met {summary['met_on_holdout']} "
                      f"(labels {summary['met_on_holdout_labels']}), holdout mean {summary['holdout_retention_mean']}, "
                      f"min {summary['holdout_retention_min']}, median x{summary['compression_median']}", flush=True)

    pq, pqrels = pseudo_queries(corpus, n_queries=300, tokens=20, seed=0)
    pscored = score_space(corpus, pq, space)
    pbase = maxsim_matrix(pq, corpus)
    prel = relevant_from_qrels(pqrels, pq.ids, corpus.ids)
    pcal, phold = split_queries(len(pq), 0.5, 0)
    proxy = []
    for target in TARGETS:
        _, chosen, _ = select(pscored, pbase, prel, pcal, phold, target, METRIC, n_boot=0)
        if chosen is None:
            proxy.append({"target": target, "selected": None})
            continue
        sc = by_json[json.dumps(chosen.pipeline, sort_keys=True)]
        real = ret_on(sc.scores, base, rel_labels, everyone)
        proxy.append({"target": target, "selected": chosen.label, "pseudo_retention": chosen.retention,
                      "real_retention_labels": real, "compression": chosen.compression_vs_float32})
        print(f"  pseudo-query proxy target {target:.2f}: picks {chosen.label} (pseudo ret {chosen.retention:.3f})"
              f" -> real-query retention {real:.3f}", flush=True)

    out = {"dataset": name, "n_docs": len(corpus), "n_queries": len(queries), "metric": METRIC,
           "baseline_ndcg5": float(np.nanmean(per_query_metrics(base, rel_labels, [METRIC])[METRIC])),
           "candidates": table, "fixed": fixed_rows, "study": study, "pseudo_query_proxy": proxy,
           "seconds": time.time() - t0}
    pathlib.Path(S + "/phase4/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(S + f"/phase4/results/{name}.json").write_text(json.dumps(out, indent=1, default=float),
                                                                encoding="utf-8")
    print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(DATASETS)):
        run(n)
