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

``--distractors`` adds pages of other image datasets (the first ``--distractor-pages`` rows,
in the order given) as pages no query is labelled with, so that only the corpus grows; they are
read and encoded a chunk at a time, never all held as images. The tag gets ``_plus<N>``.
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


def _gpu_budgets() -> dict[int, int]:
    """``max_memory`` for a sentence-transformers model spread with ``device_map="auto"``.

    GPU 0 also holds the inputs, the vision activations and every page's outputs, so it gets
    a smaller share of the weights (tencent/EVIE-8B ran GPU 0 out of memory in its forward pass
    with an even split, Kaggle 2026-10-16).
    """
    import torch

    return {i: int(torch.cuda.get_device_properties(i).total_memory * (0.45 if i == 0 else 0.85))
            for i in range(torch.cuda.device_count())}


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
    ap.add_argument("--loader", default=None,
                    help="loader for a model that is not in the registry, or to override the registry's "
                         "(see optivision.adapters.supported_loaders(), e.g. colpali-engine:colqwen3.5, pylate)")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="allow a remote-code model to run its own modelling code on this machine")
    ap.add_argument("--revision", default=None,
                    help="commit of a remote-code model to load ('latest' loads and records the current one)")
    ap.add_argument("--check-pairs", type=int, default=12,
                    help="pages on which check_retriever runs before the full encode (0 skips it); the run "
                         "stops if the vectors are not finite, unit-norm and retrieving their own page")
    ap.add_argument("--processor-kwargs", type=json.loads, default=None,
                    help="JSON for a sentence-transformers model's processor, e.g. the page budget "
                         "a model card's evaluation used")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="items per encode call on the sentence-transformers path (pages are large)")
    ap.add_argument("--distractors", default="", help="comma-separated HF image datasets to add unlabelled pages from")
    ap.add_argument("--distractor-pages", type=int, default=0, help="how many distractor pages to add")
    a = ap.parse_args()
    distractors = [d for d in a.distractors.split(",") if d]
    if bool(distractors) != bool(a.distractor_pages):
        raise SystemExit("--distractors and --distractor-pages go together")

    from datasets import load_dataset

    info = MODELS.get(a.model)
    print(f"{a.model}: registry status {info.status if info else 'unknown (not in registry)'}")
    ds = load_dataset(a.dataset, split=a.split)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    if a.backend == "nemotron":
        if distractors:
            raise SystemExit("--distractors is not supported on the --backend nemotron path")
        return encode_nemotron(a, info)
    if a.backend:
        from optivision.encoders.colvlm import ColVLMEncoder

        adapter = PageEncoderAdapter(
            ColVLMEncoder(backend=a.backend, model_name=a.model, dtype=a.dtype, multi_gpu=a.multi_gpu), info=info)
    else:
        loader = a.loader or (info.loader if info else "sentence-transformers")
        options: dict = {}
        if loader == "nemotron" or loader.startswith(("colpali-engine:", "remote-code:")):
            options.update(dtype=a.dtype, multi_gpu=a.multi_gpu)
        if loader.startswith("remote-code:"):
            options.update(trust_remote_code=a.trust_remote_code, revision=a.revision)
        if loader == "sentence-transformers":
            options.update(batch_size=a.batch_size, encode_kwargs={"show_progress_bar": False})
            if a.dtype != "auto":
                options["dtype"] = a.dtype
            if a.trust_remote_code:
                options["trust_remote_code"] = True
            # the registry's pin unless --revision says otherwise ("latest": the Hub's main branch)
            revision = a.revision or (info.revision if info else None)
            if revision and revision != "latest":
                options["revision"] = revision
            if a.multi_gpu:
                options["model_kwargs"] = {"device_map": "auto", "max_memory": _gpu_budgets()}
            if a.processor_kwargs:
                options["processor_kwargs"] = a.processor_kwargs
        adapter = load_adapter(a.model, loader=a.loader, **options)
    encoder = getattr(adapter, "encoder", adapter)
    run_info = {
        "device": getattr(encoder, "device", None),
        "dtype": str(getattr(encoder, "torch_dtype", a.dtype)),
        "adapter_merge_check": getattr(encoder, "adapter_merge_check", None),
        "loading_report": getattr(encoder, "loading_report", None),
        "placement": getattr(encoder, "placement", None),
        "max_num_visual_tokens": getattr(encoder, "max_num_visual_tokens", None),
        "revision": getattr(encoder, "revision", None),
        "processor_kwargs": a.processor_kwargs,
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

    if a.check_pairs:
        from optivision.adapters import check_retriever

        first = {pid: q for qid, q in queries for pid in qrels[qid]}
        pairs = [(im, first[pid]) for pid, im in zip(ids, images) if pid in first][: a.check_pairs]
        if len(pairs) >= 2:
            pages = [im for im, _ in pairs]
            if isinstance(adapter, PageEncoderAdapter):
                pages = [(PageRef(f"check{i}", 1), im) for i, im in enumerate(pages)]
            report = check_retriever(adapter, pages, [q for _, q in pairs],
                                     expected_dim=info.dimension if info else None)
            print(json.dumps({"check_retriever": report}, default=str), flush=True)
            if not report["ok"]:
                print("CHECK FAILED: the model does not behave like a retriever on these pages; nothing was written")
                return 1

    filenames = [row.get("image_filename") for row in ds.select_columns(["image_filename"])]         if "image_filename" in ds.column_names else []
    repeated = len(filenames) - len(set(filenames)) if filenames else 0
    if repeated:  # this loader takes each row as its own page: a repeated page would be a second copy
        print(json.dumps({"warning": f"{repeated} rows repeat an image_filename; each row is a page here"}), flush=True)

    t0 = time.time()
    docs = _encode_pages(adapter, images, ids)
    del images
    if distractors:
        docs = _with_distractors(adapter, docs, distractors, a.distractor_pages, a.split)
    t1 = time.time()
    qs = adapter.encode_queries([q for _, q in queries], ids=[qid for qid, _ in queries])
    t2 = time.time()

    tag = tag_for(a.model, a.dataset, a.distractor_pages)
    a.out.mkdir(parents=True, exist_ok=True)
    docs.attrs.update({"model": a.model, "dataset": a.dataset, "encode_seconds": t1 - t0,
                       "distractors": distractors, "distractor_pages": a.distractor_pages, **run_info})
    docs.save(a.out / f"{tag}_docs.npz")
    qs.save(a.out / f"{tag}_queries.npz")
    (a.out / f"{tag}_qrels.json").write_text(json.dumps(qrels), encoding="utf-8")
    print(json.dumps({"tag": tag, "docs": len(docs), "vectors": docs.num_vectors, "dim": docs.dimension,
                      "queries": len(qs), "doc_seconds": round(t1 - t0, 1), "query_seconds": round(t2 - t1, 1),
                      **run_info}))
    return 0


def tag_for(model: str, dataset: str, distractor_pages: int = 0) -> str:
    return f"{model.split('/')[-1]}_{dataset.split('/')[-1]}" + (f"_plus{distractor_pages}" if distractor_pages else "")


def _encode_pages(adapter, images: list, ids: list[str]):
    if isinstance(adapter, PageEncoderAdapter):
        return adapter.encode_documents([(PageRef(pid.split("::")[0], 1), im) for pid, im in zip(ids, images, strict=True)])
    return adapter.encode_documents(images, ids=ids)


def _with_distractors(adapter, docs, names: list[str], count: int, split: str, chunk: int = 32):
    """``docs`` followed by the first ``count`` pages of the named datasets (ids ``x<j>::p1``)."""
    from datasets import load_dataset

    from optivision.representation import MultiVectorCorpus

    arrays = [docs.vectors[docs.offsets[i]:docs.offsets[i + 1]] for i in range(len(docs))]
    ids = list(docs.ids)
    images, pids, t0 = [], [], time.time()

    def flush() -> None:
        extra = _encode_pages(adapter, images, pids)
        arrays.extend(extra.vectors[extra.offsets[i]:extra.offsets[i + 1]] for i in range(len(extra)))
        ids.extend(pids)
        images.clear()
        pids.clear()

    added = 0
    for name in names:
        for row in load_dataset(name, split=split):
            if added == count:
                break
            images.append(row["image"].convert("RGB"))
            pids.append(f"x{added:05d}::p1")
            added += 1
            if len(images) == chunk:
                flush()
                print(f"  distractor pages: {added}/{count} ({(time.time() - t0) / added:.2f} s/page)", flush=True)
        if added == count:
            break
    if images:
        flush()
    if added < count:
        raise SystemExit(f"only {added} distractor pages in {names}, {count} asked for")
    print(json.dumps({"distractors": names, "distractor_pages": added, "corpus": len(ids)}), flush=True)
    return MultiVectorCorpus.from_arrays(arrays, ids=ids, attrs=dict(docs.attrs))


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
