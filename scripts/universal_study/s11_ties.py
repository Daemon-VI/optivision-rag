"""Release audit: does breaking score ties by document index inflate retention?

The package ranks tied documents by index. This re-ranks every query twice more, with tied
relevant documents placed first (optimistic) or last (pessimistic), and compares retention.

    OPTIVISION_DATA=data OPTIVISION_OUT=reports/universal/raw \
        python scripts/universal_study/s11_ties.py docvqa infovqa scifact
"""

import json
import os
import pathlib
import sys

import numpy as np

from optivision.benchmark import Dataset, load_legacy_dataset
from optivision.compose import Pipeline
from optivision.evaluation import per_query_metrics
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix
from optivision.stages import AdaptiveMerge, BinaryQuantizer, HierarchicalMerge, Int4Quantizer

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
LEGACY = {"docvqa": ("colpali_docvqa_test_subsampled", "vidore_docvqa_test_subsampled"),
          "infovqa": ("colpali_infovqa_test_subsampled", "vidore_infovqa")}
CONFIGS = {
    "binary": lambda: [BinaryQuantizer()],
    "ward0.25>binary": lambda: [HierarchicalMerge(ratio=0.25), BinaryQuantizer()],
    "adaptive_c0.6>binary": lambda: [AdaptiveMerge(radius=0.6, center="mean"), BinaryQuantizer()],
    "ward0.25>int4c": lambda: [HierarchicalMerge(ratio=0.25), Int4Quantizer(center="mean")],
}


def dataset(name):
    if name == "scifact":
        tag = "answerai-colbert-small-v1"
        docs = MultiVectorCorpus.load(f"{D}/vectors/scifact_{tag}_docs.npz")
        qs = MultiVectorCorpus.load(f"{D}/vectors/scifact_{tag}_queries.npz")
        qrels = {k: set(v) for k, v in
                 json.loads(pathlib.Path(f"{D}/vectors/scifact_qrels_test.json").read_text()).items()}
        return Dataset("scifact", docs, qs, qrels), "ndcg@10"
    cache, qdir = LEGACY[name]
    return load_legacy_dataset(f"{D}/cache/{cache}.npz", f"{D}/{qdir}/queries.json", name=name,
                               with_images=False), "ndcg@5"


def ndcg_with_ties(scores, relevant, k, relevant_last):
    """nDCG@k with exact ties broken for (optimistic) or against (pessimistic) relevant documents."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = np.full(len(relevant), np.nan)
    for qi, rel in enumerate(relevant):
        if rel.size == 0:
            continue
        is_rel = np.isin(np.arange(scores.shape[1]), rel)
        tie_key = is_rel if relevant_last else ~is_rel
        order = np.lexsort((tie_key, -scores[qi]))[:k]  # last key is primary: score, then tie policy
        out[qi] = float((is_rel[order] * disc[: order.size]).sum() / disc[: min(rel.size, k)].sum())
    return out


def main(names):
    rows = []
    for name in names:
        ds, metric = dataset(name)
        k = int(metric.split("@")[1])
        rel = ds.relevant()
        base = per_query_metrics(maxsim_matrix(ds.queries, ds.corpus), rel, [metric])[metric]
        for label, make in CONFIGS.items():
            pipe = Pipeline(make())
            scores = maxsim_matrix(pipe.transform_queries(ds.queries), pipe.fit(ds.corpus).compress(ds.corpus))
            by_index = per_query_metrics(scores, rel, [metric])[metric]
            top = -np.sort(-scores, axis=1)[:, : k + 1]
            row = {"dataset": name, "config": label, "metric": metric,
                   "retention_index_ties": float(np.nanmean(by_index) / np.nanmean(base)),
                   "retention_pessimistic_ties": float(np.nanmean(ndcg_with_ties(scores, rel, k, True))
                                                       / np.nanmean(base)),
                   "retention_optimistic_ties": float(np.nanmean(ndcg_with_ties(scores, rel, k, False))
                                                      / np.nanmean(base)),
                   "queries_with_a_tie_in_top_k_plus_1": float(np.mean((np.diff(top, axis=1) == 0).any(axis=1)))}
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = pathlib.Path(S) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ties.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["docvqa", "infovqa", "scifact"])
