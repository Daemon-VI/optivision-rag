"""R7b: which selection rule keeps held-out quality at the target with the default search space?

Scores every candidate of ``recommended_search_space`` once per dataset (all queries, transforms
cached), then for 20 random half/half query splits applies each rule on the calibration half and
records the retention of its choice on the held-out half.

The rule ``calibrate() default`` replays exactly what ``calibrate()`` does with its defaults: a
one-sided 0.999 lower bound, families walked in order and each family stopped at its first
infeasible step (``assume_monotone=True``), n_boot=1000. The other rules judge every candidate.

    OPTIVISION_DATA=data OPTIVISION_OUT=reports/universal/raw \
        python scripts/universal_study/s7b_selection_rules.py E2-colpali-docvqa E2-colpali-infovqa text-scifact
"""

import json
import os
import pathlib
import sys
import time

import numpy as np

from optivision.benchmark import Dataset, load_legacy_dataset, load_vector_dataset
from optivision.calibration import _judge, recommended_search_space, score_space, select
from optivision.evaluation import (
    per_query_metrics,
    relevant_from_baseline,
    retention,
    split_queries,
)
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
LEGACY = {
    "E2-colpali-infovqa": (D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json"),
    "E2-colpali-docvqa": (D + "/cache/colpali_docvqa_test_subsampled.npz",
                          D + "/vidore_docvqa_test_subsampled/queries.json"),
}
RULES = {
    "point": {"safety": "point"},
    "lower 0.975": {"safety": "lower_ci", "confidence": 0.975},
    "lower 0.99": {"safety": "lower_ci", "confidence": 0.99},
    "lower 0.999": {"safety": "lower_ci", "confidence": 0.999},
    "bonferroni 0.975": {"safety": "lower_ci", "confidence": 0.975, "multiplicity": "bonferroni"},
    "lower 0.975 + margin 0.01": {"safety": "lower_ci", "confidence": 0.975, "margin": 0.01},
}
DEFAULT_RULE = "calibrate() default"
TARGETS = (0.99, 0.97, 0.95)
SPLITS = 20


def dataset(name):
    if name.startswith("vec:"):  # files written by scripts/encode_vectors.py
        return load_vector_dataset(f"{D}/vectors/{name[4:]}", name=name[4:]), "ndcg@5"
    if name == "text-scifact":
        tag = "answerai-colbert-small-v1"
        docs = MultiVectorCorpus.load(f"{D}/vectors/scifact_{tag}_docs.npz")
        qs = MultiVectorCorpus.load(f"{D}/vectors/scifact_{tag}_queries.npz")
        qrels = {k: set(v) for k, v in
                 json.loads(pathlib.Path(f"{D}/vectors/scifact_qrels_test.json").read_text()).items()}
        return Dataset(f"text-{tag}-scifact", docs, qs, qrels), "ndcg@10"
    cache, qj = LEGACY[name]
    return load_legacy_dataset(cache, qj, name=name, with_images=False), "ndcg@5"


def calibrate_default(scored, base, rel, cal, hold, target, metric, seed):
    """calibrate()'s own path: 0.999 bound, early stop per family, n_boot=1000."""
    base_cal = per_query_metrics(base[cal], [rel[i] for i in cal], [metric])[metric]
    judged, stopped = [], set()
    for sc in scored:
        if sc.family in stopped:
            continue
        c = _judge(sc, rel, cal, metric, base_cal, target, "lower_ci", None, 1000, seed, confidence=0.999)
        judged.append((sc, c))
        if not c.feasible:
            stopped.add(sc.family)
    feasible = [(sc, c) for sc, c in judged if c.feasible]
    if not feasible:
        return None, None, len(judged)
    sc, c = min(feasible, key=lambda t: (t[1].bytes_per_doc, t[1].query_ms))
    r = [rel[i] for i in hold]
    ho = retention(per_query_metrics(sc.scores[hold], r, [metric])[metric],
                   per_query_metrics(base[hold], r, [metric])[metric], n_boot=0)[0]
    return c, ho, len(judged)


def run(name):
    t0 = time.time()
    ds, metric = dataset(name)
    corpus, queries = ds.corpus, ds.queries
    print(f"== {ds.name}: {len(corpus)} docs, {len(queries)} queries, metric {metric}", flush=True)
    base = maxsim_matrix(queries, corpus)
    rel_labels = ds.relevant()
    rel_base = relevant_from_baseline(base, 1)
    space = recommended_search_space(corpus.dimension)
    scored = score_space(corpus, queries, space, progress=lambda sc: print(
        f"  scored {sc.label:48s} x{sc.report['compression_vs_float32']:7.1f}  "
        f"{sc.compress_seconds + sc.score_seconds:5.1f}s", flush=True))
    by_label = {sc.label: sc for sc in scored}

    def ret_on(sc, rel, idx):
        r = [rel[i] for i in idx]
        return retention(per_query_metrics(sc.scores[idx], r, [metric])[metric],
                         per_query_metrics(base[idx], r, [metric])[metric], n_boot=0)[0]

    results = []
    for reference, rel in (("labels", rel_labels), ("baseline@1", rel_base)):
        for rule in [DEFAULT_RULE, *RULES]:
            for target in TARGETS:
                outcomes = []
                for seed in range(SPLITS):
                    cal, hold = split_queries(len(queries), 0.5, seed)
                    if rule == DEFAULT_RULE:
                        chosen, ho, n_judged = calibrate_default(scored, base, rel, cal, hold, target, metric, seed)
                    else:
                        _, chosen, h = select(scored, base, rel, cal, hold, target, metric, n_boot=500, seed=seed,
                                              **RULES[rule])
                        ho, n_judged = (None if h is None else h["retention"]), len(scored)
                    if chosen is None:
                        outcomes.append({"seed": seed, "selected": None, "judged": n_judged})
                        continue
                    sc = by_label[chosen.label]
                    outcomes.append({"seed": seed, "selected": chosen.label, "judged": n_judged,
                                     "compression": chosen.compression_vs_float32,
                                     "calibration": chosen.retention,
                                     "holdout": ho, "holdout_labels": ret_on(sc, rel_labels, hold)})
                ok = [o for o in outcomes if o["selected"]]
                row = {"reference": reference, "rule": rule, "target": target, "selected": len(ok),
                       "splits": SPLITS,
                       "met": float(np.mean([o["holdout"] >= target for o in ok])) if ok else None,
                       "met_labels": float(np.mean([o["holdout_labels"] >= target for o in ok])) if ok else None,
                       "holdout_mean": float(np.mean([o["holdout"] for o in ok])) if ok else None,
                       "holdout_min": float(np.min([o["holdout"] for o in ok])) if ok else None,
                       "compression_median": float(np.median([o["compression"] for o in ok])) if ok else None,
                       "outcomes": outcomes}
                results.append(row)
                print(f"  {reference:10s} {rule:26s} target {target:.2f}: met {row['met']} "
                      f"(labels {row['met_labels']})  median x{row['compression_median']}  "
                      f"min {row['holdout_min']}", flush=True)
    out = {"dataset": ds.name, "metric": metric, "n_docs": len(corpus), "n_queries": len(queries),
           "n_candidates": len(scored), "rules": {DEFAULT_RULE: "calibrate() defaults", **RULES},
           "results": results,
           "candidates": [{"label": sc.label, "family": sc.family,
                           "compression": sc.report["compression_vs_float32"],
                           "retention_labels_all": ret_on(sc, rel_labels, np.arange(len(queries)))}
                          for sc in scored],
           "seconds": time.time() - t0}
    out_dir = pathlib.Path(S) / "selection_rules"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ds.name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"== {ds.name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
