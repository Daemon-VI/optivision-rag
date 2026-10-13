"""Encode ViDoRe splits with a late-interaction retriever OptiVision has no adapter for.

Writes the same three files as ``scripts/encode_vectors.py`` (``<tag>_docs.npz``,
``<tag>_queries.npz``, ``<tag>_qrels.json``), so every study script reads them as
``vec:<tag>``. A smoke test on a few pages always runs first and stops the run if
the model does not behave like a retriever.

Loaders
    tomoro   remote-code AutoModel (TomoroAI/tomoro-colqwen3-embed-*)
    colqwen35  colpali-engine ColQwen3_5 (athrael-soju/colqwen3.5-*)
    st       sentence-transformers MultiVectorEncoder (fallback for the above)
    fake     random vectors, to test this script without a GPU
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

ALLOWED_MISSING = ("lm_head.weight",)


def _load(cls, model_id: str, rev: str, dtype: str, strict: bool = True, **kw):
    """from_pretrained over one GPU (half precision) or two (float32), refusing random weights."""
    import torch

    args = dict(revision=rev, attn_implementation="sdpa", **kw)
    if dtype == "float32" and torch.cuda.device_count() > 1:
        args.update(device_map="auto", max_memory={0: "10GiB", 1: "13GiB"})
    else:
        args.update(device_map="cuda:0")
    report = None
    for dtype_arg in ("dtype", "torch_dtype"):  # transformers>=5 renamed torch_dtype to dtype
        try:
            try:
                model, info = cls.from_pretrained(model_id, output_loading_info=True,
                                                  **{dtype_arg: getattr(torch, dtype)}, **args)
                missing = [k for k in info.get("missing_keys", []) if not k.endswith(ALLOWED_MISSING)]
                report = {"missing": missing[:20], "n_missing": len(missing),
                          "n_unexpected": len(info.get("unexpected_keys", []))}
            except (ValueError, KeyError, AttributeError) as e:  # loader without loading info
                print(f"  loading info unavailable ({type(e).__name__}: {e}); loading without it", flush=True)
                model = cls.from_pretrained(model_id, **{dtype_arg: getattr(torch, dtype)}, **args)
            break
        except TypeError as e:
            if dtype_arg == "torch_dtype":
                raise
            print(f"  retrying with torch_dtype ({e})", flush=True)
    if report and report["n_missing"] and not strict:
        print(f"  WARNING: {report['n_missing']} weights reported missing, e.g. {report['missing'][:3]}; "
              "the smoke test decides", flush=True)
    elif report and report["n_missing"]:
        raise RuntimeError(f"{model_id}: {report['n_missing']} weights were not in the checkpoint and would be "
                           f"random, e.g. {report['missing'][:3]}")
    return model.eval(), report


def _pin_head(model) -> list[str]:
    """Keep the projection head on GPU 0 when the model is spread over two GPUs.

    These models multiply the projected vectors by ``attention_mask``, which is on
    the input device, while the last decoder layers (and so the head, by default)
    are on the second GPU. Moving only the head changes no arithmetic.
    """
    if len({str(d) for d in (getattr(model, "hf_device_map", None) or {}).values()}) < 2:
        return []
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module

    pinned = []
    for name in ("embedding_proj_layer", "custom_text_proj"):
        head = getattr(model, name, None)
        if head is not None:
            remove_hook_from_module(head)
            head.to("cuda:0")
            add_hook_to_module(head, AlignDevicesHook(execution_device="cuda:0", io_same_device=False))
            pinned.append(name)
    return pinned


def _placement(model) -> dict:
    m = getattr(model, "hf_device_map", None)
    if not m:
        return {"devices": [str(next(model.parameters()).device)]}
    return {"devices": sorted({str(d) for d in m.values()}),
            "modules_per_device": dict(collections.Counter(str(d) for d in m.values()))}


def _rows(emb, mask) -> np.ndarray:
    """One item's [tokens, dim] float32 vectors, without padding rows."""
    import torch

    emb = emb.detach().to("cpu", torch.float32)
    if emb.ndim == 3:
        emb = emb[0]
    if mask is not None:
        mask = mask.detach().cpu()
        mask = mask[0] if mask.ndim == 2 else mask
        if mask.shape[0] == emb.shape[0]:
            emb = emb[mask.bool()]
    emb = emb[emb.norm(dim=-1) > 1e-6]
    return np.ascontiguousarray(emb.numpy())


class _HF:
    """Shared forward pass for the two transformers-based loaders."""

    def _embed(self, batch) -> np.ndarray:
        import torch

        batch = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in dict(batch).items()}
        with torch.inference_mode():
            out = self.model(**batch)
        return _rows(getattr(out, "embeddings", out), batch.get("attention_mask"))

    def info(self) -> dict:
        return {"loader": self.loader, "dtype": f"torch.{self.dtype}", "device": "cuda:0",
                "attn_implementation": "sdpa",
                "placement": {**_placement(self.model), "head_pinned_to_cuda0": self.pinned},
                "loading_report": self.loading_report}


class Tomoro(_HF):
    loader = "tomoro"

    def __init__(self, model_id: str, rev: str, dtype: str) -> None:
        from transformers import AutoModel, AutoProcessor

        self.dtype = dtype
        self.proc = AutoProcessor.from_pretrained(model_id, revision=rev, trust_remote_code=True,
                                                  max_num_visual_tokens=1280)
        self.model, self.loading_report = _load(AutoModel, model_id, rev, dtype, trust_remote_code=True)
        self.pinned = _pin_head(self.model)

    def page(self, image) -> np.ndarray:
        return self._embed(self.proc.process_images(images=[image]))

    def query(self, text: str) -> np.ndarray:
        return self._embed(self.proc.process_texts(texts=[text]))


class ColQwen35(_HF):
    loader = "colqwen35"

    def __init__(self, model_id: str, rev: str, dtype: str) -> None:
        from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

        self.dtype = dtype
        self.proc = ColQwen3_5Processor.from_pretrained(model_id, revision=rev)
        self.model, self.loading_report = _load(ColQwen3_5, model_id, rev, dtype, strict=False)  # adapter repo: base keys are reported
        self.pinned = _pin_head(self.model)

    def page(self, image) -> np.ndarray:
        return self._embed(self.proc.process_images([image]))

    def query(self, text: str) -> np.ndarray:
        self.model.rope_deltas = None  # the model card resets this before queries
        return self._embed(self.proc.process_queries([text]))


class SentenceTransformersEncoder:
    loader = "st"

    def __init__(self, model_id: str, rev: str, dtype: str) -> None:
        import torch
        from sentence_transformers import MultiVectorEncoder

        self.dtype = "float16" if dtype == "float32" else dtype  # one GPU only on this path
        self.model = MultiVectorEncoder(model_id, revision=rev, device="cuda",
                                        model_kwargs={"torch_dtype": getattr(torch, self.dtype)})

    @staticmethod
    def _np(x) -> np.ndarray:
        x = x[0]
        x = x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)
        return np.ascontiguousarray(x[np.linalg.norm(x, axis=1) > 1e-6], dtype=np.float32)

    def page(self, image) -> np.ndarray:
        return self._np(self.model.encode_document([image]))

    def query(self, text: str) -> np.ndarray:
        return self._np(self.model.encode_query([text]))

    def info(self) -> dict:
        return {"loader": self.loader, "dtype": f"torch.{self.dtype}", "device": "cuda",
                "attn_implementation": None, "placement": {"devices": ["cuda"]}, "loading_report": None}


class Fake:
    """Random unit vectors; a query shares its page's first vectors. No GPU, no model."""

    loader = "fake"

    def __init__(self, model_id: str, rev: str, dtype: str) -> None:
        pass

    def page(self, image) -> np.ndarray:  # ``image`` is the row number here
        v = np.random.default_rng(int(image)).standard_normal((20, 32)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def query(self, text: str) -> np.ndarray:
        return self.page(int(text.split()[-1]))[:5].copy()

    def info(self) -> dict:
        return {"loader": "fake", "dtype": "float32", "device": "cpu", "attn_implementation": None,
                "placement": {"devices": ["cpu"]}, "loading_report": None}


LOADERS = {"tomoro": Tomoro, "colqwen35": ColQwen35, "st": SentenceTransformersEncoder, "fake": Fake}


def maxsim(q: np.ndarray, p: np.ndarray) -> float:
    return float((q @ p.T).max(axis=1).sum())


def read(dataset: str, limit: int | None, fake: bool):
    """Rows as (page id, image, query or None for a repeated query)."""
    if fake:
        return [(f"{i:05d}::p1", i, f"question {i}") for i in range(limit or 30)]
    from datasets import load_dataset

    ds = load_dataset(dataset, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    seen, rows = set(), []
    for i, row in enumerate(ds):
        q = row.get("query")
        rows.append((f"{i:05d}::p1", row["image"].convert("RGB"), q if q and q not in seen else None))
        seen.add(q)
    return rows


def smoke(enc, rows, n: int, dim: int | None) -> dict:
    rows = [r for r in rows if r[2]][:n]
    t = time.perf_counter()
    pages = [enc.page(r[1]) for r in rows]
    seconds = (time.perf_counter() - t) / len(rows)
    queries = [enc.query(r[2]) for r in rows]
    allv = np.concatenate([*pages, *queries])
    scores = np.array([[maxsim(q, p) for p in pages] for q in queries])
    own = int((scores.argmax(axis=1) == np.arange(len(rows))).sum())
    rep = {"pages": len(rows), "dimension": int(pages[0].shape[1]), "vectors_per_page": [int(p.shape[0]) for p in pages],
           "tokens_per_query": [int(q.shape[0]) for q in queries], "finite": bool(np.isfinite(allv).all()),
           "unit_norm_max_error": float(np.abs(np.linalg.norm(allv, axis=1) - 1).max()),
           "query_retrieves_own_page": own, "seconds_per_page": seconds, **enc.info()}
    rep["ok"] = bool(rep["finite"] and rep["unit_norm_max_error"] < 1e-2 and own >= len(rows) - 2
                     and (dim is None or rep["dimension"] == dim))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--loader", required=True, choices=sorted(LOADERS))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--dim", type=int, default=None, help="expected output width")
    ap.add_argument("--smoke-pages", type=int, default=12)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--smoke-out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    fake = a.loader == "fake"
    rev = "none"
    if not fake:
        from huggingface_hub import model_info

        rev = model_info(a.model).sha  # pin: remote code and weights of exactly this commit
    print(f"{a.model} @ {rev} via {a.loader}, {a.dtype}", flush=True)
    t0 = time.perf_counter()
    enc = LOADERS[a.loader](a.model, rev, a.dtype)
    print(f"loaded in {time.perf_counter() - t0:.0f}s: {json.dumps(enc.info(), default=str)}", flush=True)

    first = read(a.datasets[0], max(a.smoke_pages * 3, 36) if not a.limit else a.limit, fake)
    rep = {"model": a.model, "revision": rev, "dataset": a.datasets[0], **smoke(enc, first, a.smoke_pages, a.dim)}
    print(json.dumps(rep, indent=1, default=str), flush=True)
    if a.smoke_out:
        a.smoke_out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    if not rep["ok"]:
        print("SMOKE FAILED", flush=True)
        return 1
    print("SMOKE OK", flush=True)
    if a.smoke_only:
        return 0

    from optivision.representation import MultiVectorCorpus

    a.out.mkdir(parents=True, exist_ok=True)
    for dataset in a.datasets:
        rows = read(dataset, a.limit, fake)
        vectors, ids, queries, qrels = [], [], [], {}
        t0 = time.time()
        for i, (pid, image, q) in enumerate(rows):
            vectors.append(enc.page(image))
            ids.append(pid)
            if q:
                qid = f"q{len(queries):04d}"
                queries.append((qid, q))
                qrels[qid] = [pid]
            if i % 50 == 0:
                print(f"  {dataset} page {i}: {vectors[-1].shape} ({(time.time() - t0) / (i + 1):.2f} s/page)", flush=True)
        t1 = time.time()
        qvecs = [enc.query(q) for _, q in queries]
        t2 = time.time()
        run_info = {"model": a.model, "revision": rev, **enc.info(), "encode_seconds": t1 - t0,
                    "query_seconds": t2 - t1}
        docs = MultiVectorCorpus.from_arrays(vectors, ids=ids, attrs={"dataset": dataset, **run_info})
        qs = MultiVectorCorpus.from_arrays(qvecs, ids=[qid for qid, _ in queries], attrs={"model": a.model})
        tag = f"{a.model.split('/')[-1]}_{dataset.split('/')[-1]}"
        docs.save(a.out / f"{tag}_docs.npz")
        qs.save(a.out / f"{tag}_queries.npz")
        (a.out / f"{tag}_qrels.json").write_text(json.dumps(qrels), encoding="utf-8")
        print(json.dumps({"tag": tag, "docs": len(docs), "vectors": docs.num_vectors, "dim": docs.dimension,
                          "queries": len(qs), "doc_seconds": round(t1 - t0, 1), "query_seconds": round(t2 - t1, 1)},
                         default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
