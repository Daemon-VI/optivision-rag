"""Smoke test for a ColQwen checkpoint before any benchmark run.

Encodes a handful of real ViDoRe pages and their queries, then checks what a
benchmark would otherwise take on trust:

* every parameter received weights (the encoder refuses meta tensors), and an
  adapter-only checkpoint's merge actually changed the weights it targets;
* page and query vectors are finite, unit-norm and of the expected dimension;
* each query retrieves its own page among the sampled pages (a weak sanity
  check -- with N pages chance is 1/N -- not a quality measurement);
* time per page and per query on this machine.

With ``--compare-model`` it encodes the same pages and queries with a second
checkpoint and reports how closely the two agree, row by row. Comparing
``vidore/colqwen2-v1.0-merged`` (pre-merged by its authors) with
``vidore/colqwen2-v1.0`` (merged by our loader) tests the adapter path that
ColQwen2.5, which has no pre-merged release, depends on.

    python scripts/colqwen_smoke.py --model vidore/colqwen2-v1.0-merged --backend colqwen2 \
        --compare-model vidore/colqwen2-v1.0 --pages 6 --out smoke.json

The numbers are a correctness check, not a result.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optivision.encoders.colvlm import ColVLMEncoder
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix
from optivision.types import PageRef


def encode(model: str, backend: str, dtype: str, rows: list) -> tuple[MultiVectorCorpus, MultiVectorCorpus, dict]:
    t0 = time.perf_counter()
    enc = ColVLMEncoder(backend=backend, model_name=model, dtype=dtype)
    info = {"model": model, "device": enc.device, "dtype": str(enc.torch_dtype),
            "adapter_merge_check": enc.adapter_merge_check, "load_seconds": time.perf_counter() - t0}
    pages, page_s = [], []
    for i, row in enumerate(rows):
        t = time.perf_counter()
        pages.extend(enc.encode_pages([row["image"]], [PageRef(doc_id=f"{i:05d}", page_no=1)]))
        page_s.append(time.perf_counter() - t)
        print(f"  {model}: page {i} {pages[-1].vectors.shape} in {page_s[-1]:.2f}s", flush=True)
    t = time.perf_counter()
    qvecs = enc.encode_queries([r["query"] for r in rows])
    info["seconds_per_page"] = float(np.median(page_s))
    info["seconds_per_query"] = (time.perf_counter() - t) / len(rows)
    # free the model before a second one is loaded (two 2-3B models do not fit a 16 GB card)
    del enc
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return MultiVectorCorpus.from_page_encodings(pages), MultiVectorCorpus.from_arrays(qvecs), info


def check(docs: MultiVectorCorpus, queries: MultiVectorCorpus, dim: int) -> dict:
    scores = maxsim_matrix(queries, docs)
    norms = np.linalg.norm(np.asarray(docs.vectors, np.float32), axis=1)
    qnorms = np.linalg.norm(np.asarray(queries.vectors, np.float32), axis=1)
    return {
        "dimension": docs.dimension,
        "dimension_ok": docs.dimension == dim and queries.dimension == dim,
        "vectors_per_page": [int(c) for c in docs.counts],
        "tokens_per_query": [int(c) for c in queries.counts],
        "finite": bool(np.isfinite(docs.vectors).all() and np.isfinite(queries.vectors).all()),
        "unit_norm_max_error": float(max(np.abs(norms - 1).max(), np.abs(qnorms - 1).max())),
        "query_retrieves_own_page": int((scores.argmax(axis=1) == np.arange(len(docs))).sum()),
        "scores": np.round(scores, 3).tolist(),
    }


def agreement(a: MultiVectorCorpus, b: MultiVectorCorpus) -> dict:
    """Row-wise cosine between two encodings of the same inputs (same tokenisation)."""
    if not np.array_equal(a.counts, b.counts):
        return {"same_token_counts": False}
    va, vb = np.asarray(a.vectors, np.float32), np.asarray(b.vectors, np.float32)
    cos = (va * vb).sum(axis=1) / np.maximum(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12)
    return {"same_token_counts": True, "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--compare-model", default=None)
    ap.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from datasets import load_dataset

    # Distinct pages with their first query, streamed so only N rows are downloaded.
    rows, seen = [], set()
    for row in load_dataset(a.dataset, split="test", streaming=True):
        if row["query"] in seen:
            continue
        seen.add(row["query"])
        rows.append(row)
        if len(rows) == a.pages:
            break

    docs, queries, info = encode(a.model, a.backend, a.dtype, rows)
    report = {"backend": a.backend, "dataset": a.dataset, "pages": len(rows), **info, **check(docs, queries, a.dim)}
    ok = report["dimension_ok"] and report["finite"] and report["unit_norm_max_error"] < 1e-3
    if a.compare_model:
        docs2, queries2, info2 = encode(a.compare_model, a.backend, a.dtype, rows)
        report["compare"] = {**info2, **{k: v for k, v in check(docs2, queries2, a.dim).items() if k != "scores"},
                             "pages": agreement(docs, docs2), "queries": agreement(queries, queries2)}
        agree = report["compare"]["pages"].get("cosine_median", 0) > 0.99 and \
            report["compare"]["queries"].get("cosine_median", 0) > 0.99
        report["compare"]["agree"] = bool(agree)
        ok = ok and agree
    report["ok"] = bool(ok)
    print(json.dumps({k: v for k, v in report.items() if k != "scores"}, indent=1, default=str))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
