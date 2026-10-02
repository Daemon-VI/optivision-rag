"""Smoke test for a wide NVIDIA ColEmbed checkpoint before any benchmark run.

Encodes a handful of real ViDoRe pages and their queries and checks:

* identity and loading: architecture, ColBERT pooling, output width, a pinned
  revision, and no weight left to random initialisation (the encoder refuses
  otherwise);
* vectors: expected width, finite, unit norm; vectors per page;
* each query retrieves its own page among the sampled pages (a weak sanity check,
  chance is 1/N);
* our exact MaxSim against the model's own ``colbert_score`` on the same vectors.
  Theirs pads pages with zero vectors before the max, so the two differ only
  where a query token's best real similarity is negative; that count is
  reported;
* with ``--compare-fp16``: the same pages re-encoded in float16 on one GPU, and
  how far that moves the vectors and the scores. This decides whether a wider
  model that only fits in float16 can be trusted.

    python scripts/wide_smoke.py --model nvidia/nemotron-colembed-vl-4b-v2 --multi-gpu \
        --pages 6 --compare-fp16 --out smoke_nemotron4b.json
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

from optivision.encoders.nemotron import KNOWN, NemotronColEmbedEncoder
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix


def release_gpu() -> list[float]:
    """Collect and empty the CUDA cache; returns GiB still allocated per GPU.

    The caller must drop its own references to the encoder first: deleting a
    function argument only removes the function's name for it.
    """
    import torch

    gc.collect()
    if not torch.cuda.is_available():
        return []
    torch.cuda.empty_cache()
    return [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]


def encode(model: str, dtype: str, multi_gpu: bool, rows: list) -> tuple[list, list, dict, object]:
    t0 = time.perf_counter()
    enc = NemotronColEmbedEncoder(model, dtype=dtype, multi_gpu=multi_gpu)
    info = {**enc.info(), "load_seconds": time.perf_counter() - t0}
    pages, page_s = [], []
    for i, row in enumerate(rows):
        t = time.perf_counter()
        pages.extend(enc.encode_images([row["image"]]))
        page_s.append(time.perf_counter() - t)
        print(f"  {model} [{info['dtype']}]: page {i} {pages[-1].shape} in {page_s[-1]:.2f}s", flush=True)
    t = time.perf_counter()
    queries = enc.encode_queries([r["query"] for r in rows])
    info["seconds_per_page"] = float(np.median(page_s))
    info["seconds_per_query"] = (time.perf_counter() - t) / len(rows)
    return pages, queries, info, enc


def checks(pages: list, queries: list, dim: int) -> dict:
    docs, qs = MultiVectorCorpus.from_arrays(pages), MultiVectorCorpus.from_arrays(queries)
    scores = maxsim_matrix(qs, docs)
    allv = np.concatenate([*pages, *queries])
    return {
        "dimension": docs.dimension,
        "dimension_ok": docs.dimension == dim and qs.dimension == dim,
        "vectors_per_page": [int(p.shape[0]) for p in pages],
        "tokens_per_query": [int(q.shape[0]) for q in queries],
        "float32_bytes_per_page_mean": float(np.mean([p.shape[0] * p.shape[1] * 4 for p in pages])),
        "finite": bool(np.isfinite(allv).all()),
        "unit_norm_max_error": float(np.abs(np.linalg.norm(allv, axis=1) - 1).max()),
        "query_retrieves_own_page": int((scores.argmax(axis=1) == np.arange(len(pages))).sum()),
        "scores": scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="nvidia/nemotron-colembed-vl-4b-v2")
    ap.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--multi-gpu", action="store_true")
    ap.add_argument("--compare-fp16", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    import torch
    from datasets import load_dataset

    rows, seen = [], set()
    for row in load_dataset(a.dataset, split="test", streaming=True):
        if row["query"] in seen:
            continue
        seen.add(row["query"])
        rows.append(row)
        if len(rows) == a.pages:
            break

    dim = KNOWN[a.model][0]
    pages, queries, info, enc = encode(a.model, a.dtype, a.multi_gpu, rows)
    c = checks(pages, queries, dim)
    ours = c.pop("scores")
    theirs = enc.model.colbert_score([torch.from_numpy(q) for q in queries],
                                     [torch.from_numpy(p) for p in pages], device="cpu").float().numpy()
    neg = sum(int((q @ p.T).max(axis=1).__lt__(0).sum()) for q in queries for p in pages)
    report = {"dataset": a.dataset, "pages": len(rows), **info, **c,
              "scorer_check": {"max_abs_diff": float(np.abs(ours - theirs).max()),
                               "query_tokens_with_negative_best_similarity": neg}}
    del enc  # the only reference to the float32 model: it must go before a second model loads
    report["gpu_gib_allocated_after_release"] = release_gpu()
    ok = (report["dimension_ok"] and report["finite"] and report["unit_norm_max_error"] < 1e-3
          and (report["scorer_check"]["max_abs_diff"] < 1e-3 or neg > 0))

    if a.compare_fp16:
        pages16, queries16, info16, enc16 = encode(a.model, "float16", False, rows)
        del enc16
        release_gpu()
        c16 = checks(pages16, queries16, dim)
        s16 = c16.pop("scores")
        same_counts = [p.shape[0] for p in pages] == [p.shape[0] for p in pages16]
        cos = (np.concatenate([(a_ * b_).sum(1) for a_, b_ in zip(pages, pages16, strict=True)])
               if same_counts else np.array([np.nan]))
        report["fp16"] = {**{k: info16[k] for k in ("dtype", "device", "seconds_per_page")},
                          **{k: c16[k] for k in ("finite", "query_retrieves_own_page")},
                          "same_token_counts": same_counts,
                          "page_vector_cosine_min": float(np.nanmin(cos)),
                          "page_vector_cosine_median": float(np.nanmedian(cos)),
                          "score_max_abs_diff": float(np.abs(s16 - ours).max()),
                          "same_ranking": bool((s16.argsort(axis=1) == ours.argsort(axis=1)).all())}

    report["ok"] = bool(ok)
    print(json.dumps(report, indent=1, default=str))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
