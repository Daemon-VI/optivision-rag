"""Geometry and codec-distortion statistics of one set of encoded vectors.

    python scripts/vector_geometry.py vec:<tag> --out <folder>/geometry

Reads ``$OPTIVISION_DATA/vectors/<tag>_docs.npz`` and ``_queries.npz`` (the files the encode
scripts write), samples up to ``--sample`` document vectors and query vectors with a fixed seed,
and writes ``<out>/<tag>.json``. It runs after every encode in ``scripts/measure_models.py``,
before the vectors are deleted, so that a model's retention can be set against the shape of its
vectors.

What it records, on the sample (no retrieval is run):

- the corpus mean vector's norm, and the cosine of each document and query vector to it (the
  shared offset that mean-centring removes);
- the peak-to-RMS ratio ``max|x_j| / (||x|| / sqrt(d))`` per vector, before and after centring
  (plain ``int4`` scales each vector by its largest component, so a few large coordinates
  coarsen every other one);
- the share of the second moment held by the largest 1, 4 and 1% of coordinates;
- for each codec (``int8(per_vector)``, ``int4``, ``int4(mean)``, ``binary``, ``binary(mean)``),
  the cosine between each vector and its decoded copy, and how well each sampled query token's
  dot products with sampled document tokens survive (Pearson correlation per query token; relative
  RMS error for the scalar codecs), using the package's own quantizers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optivision.representation import MultiVectorCorpus
from optivision.stages.quantize import BinaryQuantizer, Int4Quantizer, Int8Quantizer

SEED = 20261017


def _sample(vectors: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if vectors.shape[0] <= n:
        return np.asarray(vectors, dtype=np.float32)
    idx = np.sort(rng.choice(vectors.shape[0], size=n, replace=False))
    return np.asarray(vectors[idx], dtype=np.float32)


def _quantiles(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, [0.05, 0.5, 0.95])
    return {"mean": float(x.mean()), "p05": float(q[0]), "median": float(q[1]), "p95": float(q[2])}


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / np.maximum(na * nb, 1e-12)


def _pearson(true: np.ndarray, approx: np.ndarray) -> float:
    """Correlation of one query token's scores; 0 when the decoded scores are constant (up to
    rounding), since such a token no longer ranks the documents at all."""
    spread = approx.std()
    if spread <= 1e-6 * max(float(np.abs(approx).mean()), 1e-12):
        return 0.0
    r = float(np.corrcoef(true, approx)[0, 1])
    return r if np.isfinite(r) else 0.0


def _peak_to_rms(v: np.ndarray) -> np.ndarray:
    rms = np.linalg.norm(v, axis=1) / np.sqrt(v.shape[1])
    return np.abs(v).max(axis=1) / np.maximum(rms, 1e-12)


def codecs() -> dict[str, object]:
    return {"int8(per_vector)": Int8Quantizer("per_vector"), "int4": Int4Quantizer(),
            "int4(mean)": Int4Quantizer(center="mean"), "binary": BinaryQuantizer(),
            "binary(mean)": BinaryQuantizer(center="mean")}


def geometry(docs: np.ndarray, queries: np.ndarray | None, query_tokens: int = 64, doc_tokens: int = 2000,
             seed: int = SEED) -> dict:
    """Statistics of a document-vector sample (and a query-vector sample, if given)."""
    rng = np.random.default_rng(seed)
    d = docs.shape[1]
    mean = docs.mean(axis=0)
    centred = docs - mean
    second = (docs ** 2).mean(axis=0)
    order = np.sort(second)[::-1]
    top1pct = max(1, d // 100)
    out: dict = {
        "dim": int(d), "doc_vectors": int(docs.shape[0]),
        "query_vectors": int(queries.shape[0]) if queries is not None else 0,
        "doc_norm": _quantiles(np.linalg.norm(docs, axis=1)),
        "mean_vector_norm": float(np.linalg.norm(mean)),
        "doc_cos_to_mean": _quantiles(_cos_rows(docs, np.broadcast_to(mean, docs.shape))),
        "peak_to_rms": _quantiles(_peak_to_rms(docs)),
        "peak_to_rms_centred": _quantiles(_peak_to_rms(centred)),
        "second_moment_share": {"top1": float(order[0] / order.sum()), "top4": float(order[:4].sum() / order.sum()),
                                "top1pct": float(order[:top1pct].sum() / order.sum())},
        "max_abs_coordinate_mean": float(np.abs(mean).max()),
    }
    if queries is not None:
        out["query_cos_to_doc_mean"] = _quantiles(_cos_rows(queries, np.broadcast_to(mean, queries.shape)))
    # Scores of query tokens against document tokens, per query token: ranking depends on how
    # each query token's scores over documents survive, not on their absolute level (binary
    # decodes to +/-1, and binary(mean) drops the per-query constant q . mu).
    pool = queries if queries is not None else docs
    q = pool[rng.choice(pool.shape[0], size=min(query_tokens, pool.shape[0]), replace=False)]
    dsel = rng.choice(docs.shape[0], size=min(doc_tokens, docs.shape[0]), replace=False)
    true = q @ docs[dsel].T
    per_codec = {}
    for label, codec in codecs().items():
        if getattr(codec, "center", None) == "mean":
            codec.mean = mean.astype(np.float32)
        decoded = codec.decode(codec.encode(docs), d)
        approx = q @ decoded[dsel].T
        r = [_pearson(t, s) for t, s in zip(true, approx, strict=True) if t.std() > 1e-9]
        entry = {"vector_cosine": _quantiles(_cos_rows(docs, decoded)),
                 "score_pearson": _quantiles(np.asarray(r)) if r else None}
        if not label.startswith("binary"):  # scalar codecs decode to the original scale
            err = approx - true
            entry["score_rel_rmse"] = float(np.sqrt((err ** 2).mean()) / max(float(np.sqrt((true ** 2).mean())), 1e-12))
        per_codec[label] = entry
    out["codecs"] = per_codec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="vec:<tag>, the files under $OPTIVISION_DATA/vectors")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=20000, help="document vectors to sample")
    a = ap.parse_args()
    if not a.source.startswith("vec:"):
        raise SystemExit("source must be vec:<tag>")
    tag = a.source[4:]
    base = Path(os.environ.get("OPTIVISION_DATA", "data")) / "vectors"
    rng = np.random.default_rng(SEED)
    docs = MultiVectorCorpus.load(base / f"{tag}_docs.npz", mmap=True)
    qpath = base / f"{tag}_queries.npz"
    queries = MultiVectorCorpus.load(qpath, mmap=True) if qpath.exists() else None
    stats = geometry(_sample(docs.vectors, a.sample, rng),
                     _sample(queries.vectors, a.sample // 4, rng) if queries is not None else None)
    stats.update(tag=tag, n_docs=len(docs), seed=SEED, sample=a.sample)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{tag}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    summary = {k: (v["score_pearson"] or {}).get("median") for k, v in stats["codecs"].items()}
    print(json.dumps({"tag": tag, "mean_vector_norm": stats["mean_vector_norm"], "score_pearson_median": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
