"""Follow-up: int4 with a norm-preserving scale, against the codecs and merges it would replace.

On one ``vec:<prefix>`` dataset (files from ``scripts/encode_vectors.py`` or
``scripts/encode_text_vectors.py``) it measures, against the float32 baseline:

* codecs: float16 / int8 per vector / int4 / int4norm / int4(mean) /
  int4norm(mean) / 2-bit / binary;
* Ward merging to 1/3 and 1/4, each followed by int8, the four int4 variants and binary;
* the decoded norms of each int4 variant over a fixed 20,000-vector sample (how far
  ``||decoded|| / ||v||`` strays from 1), and the norms of the vectors themselves;
* per-query nDCG@10 of the float32 index and of each codec row, with the number of
  vectors in each query, for analysis by query.

``int4norm`` keeps int4's levels and code size and stores, in the same float16 slot,
the scale that gives each decoded vector the norm of the vector it codes
(``NormPreservingInt4Quantizer`` in ``int4norm_codec.py``; ``PREDICTIONS_INT4NORM.md`` calls it ``int4(norm)``, and
``int4norm(mean)`` ``int4(mean, norm)``). Rows have the same fields as ``s13_wide_study.py``;
the output goes to ``int4norm/<dataset>.json``. With ``OPTIVISION_SCORE_DEVICE`` set, scoring
takes the per-page max on the device (``device_scoring.py``, the same scores).

    OPTIVISION_DATA=data OPTIVISION_OUT=reports/universal/raw OPTIVISION_SCORE_DEVICE=cuda \
        python scripts/universal_study/s17_int4_norm.py vec:LateOn_scifact
"""

import json
import os
import pathlib
import sys
import time

import numpy as np

from optivision.benchmark import load_vector_dataset
from optivision.calibration import score_space
from optivision.compose import Pipeline
from optivision.evaluation import per_query_metrics, retention
from optivision.representation import MultiVectorCorpus
from optivision.stages import (
    BinaryQuantizer,
    Float16Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from device_scoring import maxsim_matrix, use_for_calibration
from int4norm_codec import NormPreservingInt4Quantizer
from s13_wide_study import environment

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
METRICS = ["ndcg@5", "ndcg@10", "recall@1", "recall@5"]
INT4 = {
    "int4": lambda: Int4Quantizer(), "int4norm": lambda: NormPreservingInt4Quantizer(),
    "int4(mean)": lambda: Int4Quantizer(center="mean"),
    "int4norm(mean)": lambda: NormPreservingInt4Quantizer(center="mean"),
}
CODECS = {"float16": Float16Quantizer, "int8/vec": lambda: Int8Quantizer("per_vector"), **INT4,
          "2-bit": Lloyd2Quantizer, "binary": BinaryQuantizer}
MERGED = ("int8/vec", *INT4, "binary")
NORM_SAMPLE = 20_000


def grid() -> dict[str, list[Pipeline]]:
    return {
        "codec": [Pipeline([]), *[Pipeline([c()]) for c in CODECS.values()]],
        "token+codec": [Pipeline([HierarchicalMerge(ratio=r), CODECS[c]()]) for r in (0.33, 0.25) for c in MERGED],
    }


def norm_report(corpus: MultiVectorCorpus) -> dict:
    rows = np.random.default_rng(0).choice(corpus.num_vectors, min(NORM_SAMPLE, corpus.num_vectors), replace=False)
    v = corpus.vectors[np.sort(rows)]
    norms = np.linalg.norm(v, axis=1)
    out = {"sample": len(v), "vector_norm": {"mean": float(norms.mean()), "sd": float(norms.std()),
                                             "min": float(norms.min()), "max": float(norms.max())}}
    safe = np.where(norms > 0, norms, 1.0)
    for label, make in INT4.items():
        q = make().fit(corpus)
        ratio = np.linalg.norm(q.decode(q.encode(v), corpus.dimension), axis=1) / safe
        out[label] = {"ratio_mean": float(ratio.mean()), "ratio_sd": float(ratio.std()),
                      "ratio_p01": float(np.quantile(ratio, 0.01)), "ratio_p99": float(np.quantile(ratio, 0.99))}
    return out


def run(name: str) -> None:
    t_start = time.time()
    if not name.startswith("vec:"):
        raise SystemExit("s17 takes vec:<prefix> datasets")
    use_for_calibration()
    ds = load_vector_dataset(f"{D}/vectors/{name[4:]}", name=name[4:])
    corpus, queries, rel = ds.corpus, ds.queries, ds.relevant()
    n_docs, n_q, dim = len(corpus), len(queries), corpus.dimension
    print(f"== {ds.name}: {n_docs} docs, {n_q} queries, {dim}-d, {corpus.num_vectors / n_docs:.0f} vectors/doc",
          flush=True)

    base = per_query_metrics(maxsim_matrix(queries, corpus), rel, METRICS)
    baseline = {"dim": dim, "vectors_per_doc": corpus.num_vectors / n_docs,
                **{m: float(np.nanmean(v)) for m, v in base.items()},
                "encoder": {k: corpus.attrs.get(k) for k in ("model", "revision", "dtype", "device")}}
    norms = norm_report(corpus)
    print(f"   float32: nDCG@5 {baseline['ndcg@5']:.4f}; int4 norm ratio sd {norms['int4']['ratio_sd']:.4f}, "
          f"int4norm {norms['int4norm']['ratio_sd']:.5f}", flush=True)

    space = grid()
    group_of = {id(p): g for g, ps in space.items() for p in ps}
    pipes = [p for ps in space.values() for p in ps]

    def progress(sc):
        print(f"   {sc.label:44s} x{sc.report['compression_vs_float32']:7.1f}  "
              f"{sc.compress_seconds + sc.score_seconds:6.1f}s", flush=True)

    scored = score_space(corpus, queries, space, progress=progress)
    rows, per_query = [], {"float32": [float(x) for x in base["ndcg@10"]]}
    for p, sc in zip(pipes, scored, strict=True):
        pq = per_query_metrics(sc.scores, rel, METRICS)
        row = {"group": group_of[id(p)], "label": sc.label, "pipeline": sc.pipeline,
               "vectors_per_doc": float(sc.report["vectors_per_doc"]), "dim": int(sc.report["dim"]),
               "bits_per_dim": float(sc.report["bits_per_dim"]), "bytes_per_doc": sc.bytes_per_doc,
               "compression_vs_float32": float(sc.report["compression_vs_float32"]),
               "compress_seconds": sc.compress_seconds, "scan_ms_per_query": sc.query_ms}
        for m in METRICS:
            point, lo, hi = retention(pq[m], base[m], n_boot=1000)
            row[m] = float(np.nanmean(pq[m]))
            row[f"retention:{m}"] = point
            row[f"retention_ci:{m}"] = [lo, hi]
        rows.append(row)
        if row["group"] == "codec" and sc.label != "float32":
            per_query[sc.label] = [float(x) for x in pq["ndcg@10"]]

    out = {"dataset": ds.name, "n_docs": n_docs, "n_queries": n_q, "baseline": baseline, "rows": rows,
           "norms": norms, "per_query_ndcg@10": per_query,
           "query_vectors": [int(n) for n in np.diff(queries.offsets)],
           "environment": environment(), "seconds": time.time() - t_start}
    out_dir = pathlib.Path(S) / "int4norm"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ds.name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"== {ds.name} done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
