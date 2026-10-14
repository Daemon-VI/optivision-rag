"""Encode BEIR SciFact with any text late-interaction model into OptiVision vector files.

Writes ``<out>/<tag>_scifact_docs.npz``, ``<tag>_scifact_queries.npz`` and
``<tag>_scifact_qrels.json`` (``<tag>`` is the last part of the model id), the files every
study script reads as ``vec:<tag>_scifact``. Documents are ``title + " " + text``; the
queries are those of the test qrels with a relevance above zero, as in
``scripts/universal_study/s9_encode_scifact.py``.

    python scripts/encode_text_vectors.py lightonai/LateOn --out data/vectors
    python scripts/encode_text_vectors.py org/model --loader pylate --out data/vectors

Models load through :func:`optivision.adapters.load_adapter`. ``check_retriever`` runs on
the first labelled pairs before the full encode, and nothing is written if it fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optivision.adapters import MODELS, check_retriever, load_adapter


def read_scifact(limit: int | None) -> tuple[list[str], list[str], dict[str, str], dict[str, list[str]]]:
    from datasets import load_dataset

    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_rows = load_dataset("BeIR/scifact-qrels", split="test")
    qrels: dict[str, list[str]] = {}
    for row in qrels_rows:
        if int(row["score"]) > 0:
            qrels.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))
    ids = [str(r["_id"]) for r in corpus]
    texts = [(r["title"] + " " + r["text"]).strip() for r in corpus]
    if limit:
        keep = set(ids[:limit])
        ids, texts = ids[:limit], texts[:limit]
        qrels = {q: [d for d in docs if d in keep] for q, docs in qrels.items()}
        qrels = {q: docs for q, docs in qrels.items() if docs}
    query_text = {str(r["_id"]): r["text"] for r in queries}
    return ids, texts, query_text, qrels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, default=Path("data/vectors"))
    ap.add_argument("--loader", default=None, help="override the registry's loader (e.g. pylate)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N documents only (for a dry run)")
    ap.add_argument("--check-pairs", type=int, default=12)
    a = ap.parse_args()

    info = MODELS.get(a.model)
    loader = a.loader or (info.loader if info else "sentence-transformers")
    print(f"{a.model}: registry status {info.status if info else 'unknown'}, loader {loader}", flush=True)
    ids, texts, query_text, qrels = read_scifact(a.limit)
    options: dict = {}
    if loader == "sentence-transformers":
        options.update(batch_size=a.batch_size, encode_kwargs={"show_progress_bar": False})
    if loader == "pylate":
        options.update(batch_size=a.batch_size)
    if a.trust_remote_code:
        options["trust_remote_code"] = True
    adapter = load_adapter(a.model, loader=a.loader, **options)

    if a.check_pairs:
        by_id = dict(zip(ids, texts, strict=True))
        pairs, used = [], set()
        for qid in sorted(qrels, key=lambda q: int(q) if q.isdigit() else q):
            doc = qrels[qid][0]
            if doc not in used and qid in query_text:
                pairs.append((by_id[doc], query_text[qid]))
                used.add(doc)
            if len(pairs) == a.check_pairs:
                break
        report = check_retriever(adapter, [d for d, _ in pairs], [q for _, q in pairs],
                                 expected_dim=info.dimension if info else None)
        print(json.dumps({"check_retriever": report}, default=str), flush=True)
        if not report["ok"]:
            print("CHECK FAILED: the model does not behave like a retriever on these pairs; nothing was written")
            return 1

    qids = sorted((q for q in qrels if q in query_text), key=lambda q: int(q) if q.isdigit() else q)
    t0 = time.time()
    docs = adapter.encode_documents(texts, ids=ids)
    t1 = time.time()
    qs = adapter.encode_queries([query_text[q] for q in qids], ids=qids)
    t2 = time.time()
    tag = f"{a.model.split('/')[-1]}_scifact"
    a.out.mkdir(parents=True, exist_ok=True)
    docs.attrs.update({"model": a.model, "dataset": "BeIR/scifact", "encode_seconds": t1 - t0})
    docs.save(a.out / f"{tag}_docs.npz")
    qs.save(a.out / f"{tag}_queries.npz")
    (a.out / f"{tag}_qrels.json").write_text(json.dumps({q: qrels[q] for q in qids}), encoding="utf-8")
    print(json.dumps({"tag": tag, "docs": len(docs), "vectors": docs.num_vectors, "dim": docs.dimension,
                      "queries": len(qs), "doc_seconds": round(t1 - t0, 1), "query_seconds": round(t2 - t1, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
