"""Encode a ViDoRe (or any HF image-query) split into OptiVision vector files.

Produces what every universal-layer command consumes -- no model needed after this:

    <out>/<tag>_docs.npz      MultiVectorCorpus of page vectors
    <out>/<tag>_queries.npz   MultiVectorCorpus of query vectors
    <out>/<tag>_qrels.json    {qid: [doc ids]}

    python scripts/encode_vectors.py vidore/colqwen2-v1.0 vidore/infovqa_test_subsampled --out data/vectors
    optivision calibrate data/vectors/<tag>_docs.npz -q data/vectors/<tag>_queries.npz \
        --qrels data/vectors/<tag>_qrels.json --target 0.97

Models load through :func:`optivision.adapters.load_adapter`, so each one carries
its registry status (measured / untested / unverified). Needs a GPU for anything
larger than ColSmol, plus ``datasets`` and the model's own library. Query and page
ids follow ``optivision.corpus.load_vidore_subset``: page ``i`` is
``f"{i:05d}::p1"`` and each distinct query is relevant to the first row it
appears on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optivision.adapters import MODELS, PageEncoderAdapter, load_adapter
from optivision.types import PageRef


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="model id, e.g. vidore/colqwen2-v1.0")
    ap.add_argument("dataset", help="HF dataset id with 'image' and 'query' columns")
    ap.add_argument("--out", type=Path, default=Path("data/vectors"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", default=None,
                    help="colpali-engine backend (e.g. colqwen2, colqwen2.5) to load MODEL with directly, "
                         "for checkpoints not in the registry such as vidore/colqwen2-v1.0-merged")
    ap.add_argument("--dtype", default="auto", help="auto | float32 | bfloat16 | float16")
    ap.add_argument("--multi-gpu", action="store_true",
                    help="spread the model over all visible GPUs (keeps float32 for models too big for one card)")
    a = ap.parse_args()

    from datasets import load_dataset

    info = MODELS.get(a.model)
    print(f"{a.model}: registry status {info.status if info else 'unknown (not in registry)'}")
    ds = load_dataset(a.dataset, split=a.split)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    if a.backend == "nemotron":
        return encode_nemotron(a, info)
    if a.backend:
        from optivision.encoders.colvlm import ColVLMEncoder

        adapter = PageEncoderAdapter(
            ColVLMEncoder(backend=a.backend, model_name=a.model, dtype=a.dtype, multi_gpu=a.multi_gpu), info=info)
    else:
        adapter = load_adapter(a.model, dtype=a.dtype) if info and info.loader.startswith("colpali-engine:")             else load_adapter(a.model)
    encoder = getattr(adapter, "encoder", None)
    run_info = {
        "device": getattr(encoder, "device", None),
        "dtype": str(getattr(encoder, "torch_dtype", a.dtype)),
        "adapter_merge_check": getattr(encoder, "adapter_merge_check", None),
        "loading_report": getattr(encoder, "loading_report", None),
        "placement": getattr(encoder, "placement", None),
    }
    print(json.dumps({"loaded": a.model, **run_info}), flush=True)

    images, ids, queries, qrels, seen = [], [], [], {}, set()
    for i, row in enumerate(ds):
        pid = f"{i:05d}::p1"
        images.append(row["image"].convert("RGB"))
        ids.append(pid)
        q = row.get("query")
        if q and q not in seen:
            seen.add(q)
            qid = f"q{len(queries):04d}"
            queries.append((qid, q))
            qrels[qid] = [pid]

    t0 = time.time()
    if isinstance(adapter, PageEncoderAdapter):
        docs = adapter.encode_documents([(PageRef(pid.split("::")[0], 1), im) for pid, im in zip(ids, images)])
    else:
        docs = adapter.encode_documents(images, ids=ids)
    t1 = time.time()
    qs = adapter.encode_queries([q for _, q in queries], ids=[qid for qid, _ in queries])
    t2 = time.time()

    tag = f"{a.model.split('/')[-1]}_{a.dataset.split('/')[-1]}"
    a.out.mkdir(parents=True, exist_ok=True)
    docs.attrs.update({"model": a.model, "dataset": a.dataset, "encode_seconds": t1 - t0, **run_info})
    docs.save(a.out / f"{tag}_docs.npz")
    qs.save(a.out / f"{tag}_queries.npz")
    (a.out / f"{tag}_qrels.json").write_text(json.dumps(qrels), encoding="utf-8")
    print(json.dumps({"tag": tag, "docs": len(docs), "vectors": docs.num_vectors, "dim": docs.dimension,
                      "queries": len(qs), "doc_seconds": round(t1 - t0, 1), "query_seconds": round(t2 - t1, 1),
                      **run_info}))
    return 0


def encode_nemotron(a, info) -> int:
    """Wide NVIDIA ColEmbed models: arrays straight into the corpus (no patch grid)."""
    from datasets import load_dataset

    from optivision.encoders.nemotron import NemotronColEmbedEncoder
    from optivision.representation import MultiVectorCorpus

    enc = NemotronColEmbedEncoder(a.model, dtype=a.dtype, multi_gpu=a.multi_gpu,
                                  expected_dim=info.dimension if info else None)
    print(json.dumps({"loaded": a.model, **enc.info()}, default=str), flush=True)
    ds = load_dataset(a.dataset, split=a.split)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    vectors, ids, queries, qrels, seen = [], [], [], {}, set()
    t0 = time.time()
    for i, row in enumerate(ds):  # one page at a time: never more than one page of activations
        pid = f"{i:05d}::p1"
        vectors.extend(enc.encode_images([row["image"]]))
        ids.append(pid)
        q = row.get("query")
        if q and q not in seen:
            seen.add(q)
            qid = f"q{len(queries):04d}"
            queries.append((qid, q))
            qrels[qid] = [pid]
        if i % 50 == 0:
            print(f"  page {i}: {vectors[-1].shape} ({(time.time() - t0) / (i + 1):.2f} s/page)", flush=True)
    t1 = time.time()
    qvecs = enc.encode_queries([q for _, q in queries])
    t2 = time.time()
    run_info = {**enc.info(), "encode_seconds": t1 - t0, "query_seconds": t2 - t1}
    docs = MultiVectorCorpus.from_arrays(vectors, ids=ids, attrs={"dataset": a.dataset, **run_info})
    qs = MultiVectorCorpus.from_arrays(qvecs, ids=[qid for qid, _ in queries], attrs={"model": a.model})
    tag = f"{a.model.split('/')[-1]}_{a.dataset.split('/')[-1]}"
    a.out.mkdir(parents=True, exist_ok=True)
    docs.save(a.out / f"{tag}_docs.npz")
    qs.save(a.out / f"{tag}_queries.npz")
    (a.out / f"{tag}_qrels.json").write_text(json.dumps(qrels), encoding="utf-8")
    print(json.dumps({"tag": tag, "docs": len(docs), "vectors": docs.num_vectors, "dim": docs.dimension,
                      "queries": len(qs), "doc_seconds": round(t1 - t0, 1), "query_seconds": round(t2 - t1, 1),
                      **{k: v for k, v in run_info.items() if k not in ("encode_seconds", "query_seconds")}},
                     default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
