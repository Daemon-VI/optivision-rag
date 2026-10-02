"""R3b: do centred scalar codecs fix int4 on the anisotropic text model, and do they hurt ColPali?

    OPTIVISION_DATA=data OPTIVISION_OUT=reports/universal/raw         python scripts/universal_study/s3c_centred_codecs.py E2-colpali-infovqa E2-colpali-docvqa text-scifact

With --geometry it instead writes the diagnostics behind R8/R3b: the median cosine
of each vector's nearest neighbour inside its own document, the norm of the corpus mean,
and centred binary scored with the query's component along the corpus mean removed
(if the collapse comes from that shared direction, this recovers it).
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

from optivision.benchmark import Dataset, load_legacy_dataset, load_vector_dataset
from optivision.compose import Pipeline
from optivision.evaluation import per_query_metrics, retention
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix
from optivision.stages import BinaryQuantizer, HierarchicalMerge, Int4Quantizer, Int8Quantizer

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
LEGACY = {
    "E2-colpali-infovqa": (D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json"),
    "E2-colpali-docvqa": (D + "/cache/colpali_docvqa_test_subsampled.npz", D + "/vidore_docvqa_test_subsampled/queries.json"),
}


def dataset(name):
    if name.startswith("vec:"):  # files written by scripts/encode_vectors.py
        return load_vector_dataset(f"{D}/vectors/{name[4:]}", name=name[4:]), "ndcg@5"
    if name == "text-scifact":
        docs = MultiVectorCorpus.load(f"{D}/vectors/scifact_answerai-colbert-small-v1_docs.npz")
        qs = MultiVectorCorpus.load(f"{D}/vectors/scifact_answerai-colbert-small-v1_queries.npz")
        qrels = {k: set(v) for k, v in json.loads(pathlib.Path(f"{D}/vectors/scifact_qrels_test.json").read_text()).items()}
        return Dataset("text-answerai-colbert-small-v1-scifact", docs, qs, qrels), "ndcg@10"
    cache, qj = LEGACY[name]
    return load_legacy_dataset(cache, qj, name=name, with_images=False), "ndcg@5"


CODECS = {
    "int8/vec": lambda: Int8Quantizer("per_vector"),
    "int8/vec centred": lambda: Int8Quantizer("per_vector", center="mean"),
    "int4": Int4Quantizer,
    "int4 centred": lambda: Int4Quantizer(center="mean"),
    "binary": BinaryQuantizer,
    "binary centred": lambda: BinaryQuantizer(center="mean"),
}

def geometry(name):
    ds, metric = dataset(name)
    corpus, queries = ds.corpus, ds.queries
    rel = ds.relevant()
    base = per_query_metrics(maxsim_matrix(queries, corpus), rel, [metric])[metric]
    rng = np.random.default_rng(0)
    nn = []
    for i in rng.choice(len(corpus), size=min(300, len(corpus)), replace=False):
        v = np.asarray(corpus[int(i)].vectors, dtype=np.float32)
        if v.shape[0] < 2:
            continue
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        s = v @ v.T
        np.fill_diagonal(s, -np.inf)
        nn.append(np.median(s.max(axis=1)))
    q = BinaryQuantizer(center="mean")
    comp = Pipeline([q]).fit(corpus).compress(corpus)
    mu = q.mean / np.linalg.norm(q.mean)
    qv = np.asarray(queries.vectors, dtype=np.float32)
    q_perp = queries.with_vectors((qv - np.outer(qv @ mu, mu)).astype(np.float32))
    rows = {}
    for label, qs in (("binary centred", queries), ("binary centred, query mean direction removed", q_perp)):
        pq = per_query_metrics(maxsim_matrix(qs, comp), rel, [metric])[metric]
        point, lo, hi = retention(pq, base, n_boot=1000)
        rows[label] = {"retention": point, "ci": [lo, hi]}
        print(f"  {label:48s} {point:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    out = {"dataset": ds.name, "metric": metric,
           "median_nn_cosine_within_doc": float(np.median(nn)),
           "corpus_mean_norm": float(np.linalg.norm(q.mean)),
           "query_cosine_to_corpus_mean_median": float(np.median((qv @ mu) / np.linalg.norm(qv, axis=1))),
           "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)
    pathlib.Path(f"{S}/geometry").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{S}/geometry/{ds.name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if "--geometry" in sys.argv:
    for n in [a for a in sys.argv[1:] if a != "--geometry"]:
        geometry(n)
    sys.exit(0)

for name in sys.argv[1:]:
    t0 = time.time()
    ds, metric = dataset(name)
    corpus, queries = ds.corpus, ds.queries
    rel = ds.relevant()
    base = per_query_metrics(maxsim_matrix(queries, corpus), rel, [metric])[metric]
    merged = HierarchicalMerge(ratio=0.33).transform(corpus)
    float_bytes = corpus.num_vectors * corpus.dimension * 4
    rows = []
    print(f"== {ds.name} ({metric})", flush=True)
    for tname, x in (("none", corpus), ("ward ratio=0.33", merged)):
        for cname, make in CODECS.items():
            comp = Pipeline([make()]).fit(x).compress(x)
            pq = per_query_metrics(maxsim_matrix(queries, comp), rel, [metric])[metric]
            point, lo, hi = retention(pq, base, n_boot=1000)
            rows.append({"label": f"{tname} > {cname}", "compression": float_bytes / comp.nbytes,
                         "retention": point, "ci": [lo, hi]})
            print(f"  {tname + ' > ' + cname:34s} x{float_bytes / comp.nbytes:6.1f}  retention {point:.3f} "
                  f"[{lo:.3f},{hi:.3f}]", flush=True)
    pathlib.Path(f"{S}/centred_codecs").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{S}/centred_codecs/{ds.name}.json").write_text(
        json.dumps({"dataset": ds.name, "metric": metric, "rows": rows}, indent=1), encoding="utf-8")
    print(f"== {ds.name} done in {time.time() - t0:.0f}s", flush=True)
