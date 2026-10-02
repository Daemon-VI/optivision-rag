"""Release audit: does retention hold as the corpus grows?

Retention of fixed configurations on nested random subsets of a corpus (every document relevant to
a kept query is always included, so each query keeps its answer; the rest is filled with random
distractors, nested across sizes, and the query set is fixed within a draw). Calibrating on a
small corpus says nothing about a large one unless this is flat.

    OPTIVISION_DATA=data OPTIVISION_OUT=reports/universal/raw \
        python scripts/universal_study/s12_corpus_size.py docvqa infovqa
"""

import json
import os
import pathlib
import sys

import numpy as np

from optivision.benchmark import load_legacy_dataset, load_vector_dataset
from optivision.compose import Pipeline
from optivision.evaluation import per_query_metrics, relevant_from_qrels, retention
from optivision.scoring import maxsim_matrix
from optivision.stages import AdaptiveMerge, BinaryQuantizer, Int4Quantizer, Int8Quantizer

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
LEGACY = {"docvqa": ("colpali_docvqa_test_subsampled", "vidore_docvqa_test_subsampled"),
          "infovqa": ("colpali_infovqa_test_subsampled", "vidore_infovqa")}
CONFIGS = {
    "binary": lambda: [BinaryQuantizer()],
    "int4": lambda: [Int4Quantizer()],
    "adaptive_c0.6>int8/vec": lambda: [AdaptiveMerge(radius=0.6, center="mean"), Int8Quantizer("per_vector")],
    "adaptive0.6>binary": lambda: [AdaptiveMerge(radius=0.6), BinaryQuantizer()],
}
SIZES = (100, 200, 300, 400, 500)
DRAWS = 5


def main(names):
    rows = []
    for name in names:
        if name.startswith("vec:"):  # files written by scripts/encode_vectors.py
            ds = load_vector_dataset(f"{D}/vectors/{name[4:]}", name=name[4:])
        else:
            cache, qdir = LEGACY[name]
            ds = load_legacy_dataset(f"{D}/cache/{cache}.npz", f"{D}/{qdir}/queries.json", name=name, with_images=False)
        corpus, queries = ds.corpus, ds.queries
        full_rel = ds.relevant()
        base_full = maxsim_matrix(queries, corpus)
        comp = {}
        for label, make in CONFIGS.items():
            pipe = Pipeline(make())
            comp[label] = maxsim_matrix(pipe.transform_queries(queries), pipe.fit(corpus).compress(corpus))
        rng = np.random.default_rng(0)
        for draw in range(DRAWS):
            # fixed queries whose relevant pages fit in the smallest corpus; distractors are nested
            docs, keep = set(), []
            for qi in rng.permutation(len(queries)):
                need = set(full_rel[qi].tolist())
                if need and len(docs | need) <= SIZES[0] // 2:
                    docs |= need
                    keep.append(qi)
            q_idx = np.array(sorted(keep))
            distractors = [i for i in rng.permutation(len(corpus)) if i not in docs]
            for size in SIZES:
                doc_idx = np.array(sorted(docs | set(distractors[: size - len(docs)])))
                ids = [corpus.ids[i] for i in doc_idx]
                rel = relevant_from_qrels(ds.qrels, [queries.ids[i] for i in q_idx], ids)
                b = per_query_metrics(base_full[np.ix_(q_idx, doc_idx)], rel, ["ndcg@5"])["ndcg@5"]
                for label, scores in comp.items():
                    c = per_query_metrics(scores[np.ix_(q_idx, doc_idx)], rel, ["ndcg@5"])["ndcg@5"]
                    rows.append({"dataset": name, "config": label, "corpus_size": int(doc_idx.size),
                                 "draw": draw, "queries": int(q_idx.size),
                                 "retention": retention(c, b, n_boot=0)[0],
                                 "baseline_ndcg@5": float(np.nanmean(b))})
        for label in CONFIGS:
            line = [f"{name:8s} {label:24s}"]
            for size in SIZES:
                vals = [r["retention"] for r in rows if r["dataset"] == name and r["config"] == label
                        and r["corpus_size"] == size]
                line.append(f"{size}:{np.mean(vals):.3f}")
            print("  ".join(line), flush=True)
    out = pathlib.Path(S) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus_size.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or list(LEGACY))
