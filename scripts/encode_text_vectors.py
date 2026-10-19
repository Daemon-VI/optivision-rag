"""Encode a BEIR dataset (SciFact by default) with any text late-interaction model into OptiVision vector files.

Writes ``<out>/<model>_<dataset>_docs.npz``, ``_queries.npz`` and ``_qrels.json`` (``<model>``
is the last part of the model id), the files every study script reads as
``vec:<model>_<dataset>``. Documents are ``title + " " + text``; the queries are those of the
test qrels with a relevance above zero (graded relevance counts as relevant), as in
``scripts/universal_study/s9_encode_scifact.py``.

    python scripts/encode_text_vectors.py lightonai/LateOn --out data/vectors
    python scripts/encode_text_vectors.py org/model --dataset nfcorpus --out data/vectors
    python scripts/encode_text_vectors.py org/model --distractors fiqa --distractor-docs 10000

``--distractors`` adds the first ``--distractor-docs`` documents of other BEIR corpora, in the
order given, as documents no query is labelled with: the queries and their labels stay the
dataset's own, and only the corpus grows. The tag gets ``_plus<N>``.

Models load through :func:`optivision.adapters.load_adapter`. ``check_retriever`` runs on
the first labelled SciFact pairs (``--check-dataset``) before the full encode, whatever the
dataset, and nothing is written if it fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optivision.adapters import MODELS, check_retriever, load_adapter


def read_beir(name: str, limit: int | None) -> tuple[list[str], list[str], dict[str, str], dict[str, list[str]]]:
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels_rows = load_dataset(f"BeIR/{name}-qrels", split="test")
    graded: dict[str, list[tuple[int, str]]] = {}
    for row in qrels_rows:
        if int(row["score"]) > 0:
            graded.setdefault(str(row["query-id"]), []).append((int(row["score"]), str(row["corpus-id"])))
    # most relevant first (a stable sort keeps the file's order among equal grades)
    qrels = {q: [d for _, d in sorted(docs, key=lambda sd: -sd[0])] for q, docs in graded.items()}
    ids = [str(r["_id"]) for r in corpus]
    texts = [(r["title"] + " " + r["text"]).strip() for r in corpus]
    if limit:
        keep = set(ids[:limit])
        ids, texts = ids[:limit], texts[:limit]
        qrels = {q: [d for d in docs if d in keep] for q, docs in qrels.items()}
        qrels = {q: docs for q, docs in qrels.items() if docs}
    query_text = {str(r["_id"]): r["text"] for r in queries}
    return ids, texts, query_text, qrels


def choose_check_pairs(qrels: dict[str, list[str]], query_text: dict[str, str], n: int) -> list[tuple[str, str]]:
    """Up to ``n`` (document id, query id) pairs in which each query is relevant to its own
    document and to none of the others chosen, so that "the query ranks its own document first"
    is the right answer even where a query has many relevant documents (NFCorpus). Each query
    takes its most relevant document."""
    chosen: list[tuple[str, str]] = []
    docs: set[str] = set()
    for qid in sorted(qrels, key=lambda q: int(q) if q.isdigit() else q):
        if qid not in query_text:
            continue
        doc, relevant = qrels[qid][0], set(qrels[qid])
        if doc in docs or relevant & docs or any(doc in qrels[q] for _, q in chosen):
            continue
        chosen.append((doc, qid))
        docs.add(doc)
        if len(chosen) == n:
            break
    return chosen


def read_distractors(names: list[str], count: int) -> tuple[list[str], list[str]]:
    """The first ``count`` documents of the named BEIR corpora, in order, with ids that cannot
    collide with the dataset's own (``x:<corpus>:<id>``)."""
    from datasets import load_dataset

    ids: list[str] = []
    texts: list[str] = []
    for name in names:
        for r in load_dataset(f"BeIR/{name}", "corpus", split="corpus"):
            if len(ids) == count:
                return ids, texts
            ids.append(f"x:{name}:{r['_id']}")
            texts.append((r["title"] + " " + r["text"]).strip())
    if len(ids) < count:
        raise SystemExit(f"only {len(ids)} distractor documents in {names}, {count} asked for")
    return ids, texts


def tag_for(model: str, dataset: str = "scifact", distractor_docs: int = 0) -> str:
    return f"{model.split('/')[-1]}_{dataset}" + (f"_plus{distractor_docs}" if distractor_docs else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, default=Path("data/vectors"))
    ap.add_argument("--loader", default=None, help="override the registry's loader (e.g. pylate)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N documents only (for a dry run)")
    ap.add_argument("--check-pairs", type=int, default=12)
    ap.add_argument("--check-dataset", default="scifact",
                    help="BEIR dataset whose labelled pairs check_retriever uses (SciFact: one clearly relevant "
                         "abstract per claim; NFCorpus's keyword queries with many loosely relevant documents "
                         "make 'ranks its own document first' a poor loader check)")
    ap.add_argument("--dataset", default="scifact", help="BEIR dataset name (scifact, nfcorpus, fiqa, ...)")
    ap.add_argument("--distractors", default="", help="comma-separated BEIR corpora to add unlabelled documents from")
    ap.add_argument("--distractor-docs", type=int, default=0, help="how many distractor documents to add")
    a = ap.parse_args()
    names = [n for n in a.distractors.split(",") if n]
    if bool(names) != bool(a.distractor_docs):
        raise SystemExit("--distractors and --distractor-docs go together")

    info = MODELS.get(a.model)
    loader = a.loader or (info.loader if info else "sentence-transformers")
    print(f"{a.model}: registry status {info.status if info else 'unknown'}, loader {loader}", flush=True)
    ids, texts, query_text, qrels = read_beir(a.dataset, a.limit)
    options: dict = {}
    if loader == "sentence-transformers":
        options.update(batch_size=a.batch_size, encode_kwargs={"show_progress_bar": False})
    if loader == "pylate":
        options.update(batch_size=a.batch_size)
    if a.trust_remote_code:
        options["trust_remote_code"] = True
    if loader == "sentence-transformers" and info is not None and info.revision:
        options["revision"] = info.revision  # measure the registry's pin, not whatever main is today
    adapter = load_adapter(a.model, loader=a.loader, **options)
    print(json.dumps({"loaded": a.model, "revision": options.get("revision")}), flush=True)

    if a.check_pairs:
        if a.check_dataset == a.dataset:
            c_ids, c_texts, c_query, c_qrels = ids, texts, query_text, qrels
        else:
            c_ids, c_texts, c_query, c_qrels = read_beir(a.check_dataset, None)
        by_id = dict(zip(c_ids, c_texts, strict=True))
        pairs = [(by_id[d], c_query[q]) for d, q in choose_check_pairs(c_qrels, c_query, a.check_pairs)]
        report = check_retriever(adapter, [d for d, _ in pairs], [q for _, q in pairs],
                                 expected_dim=info.dimension if info else None)
        print(json.dumps({"check_retriever": report}, default=str), flush=True)
        if not report["ok"]:
            print("CHECK FAILED: the model does not behave like a retriever on these pairs; nothing was written")
            return 1

    qids = sorted((q for q in qrels if q in query_text), key=lambda q: int(q) if q.isdigit() else q)
    if names:  # after check_retriever, which runs on the dataset's own labelled pairs
        extra_ids, extra_texts = read_distractors(names, a.distractor_docs)
        ids, texts = ids + extra_ids, texts + extra_texts
        print(json.dumps({"distractors": names, "distractor_docs": len(extra_ids), "corpus": len(ids)}), flush=True)
    t0 = time.time()
    docs = adapter.encode_documents(texts, ids=ids)
    t1 = time.time()
    qs = adapter.encode_queries([query_text[q] for q in qids], ids=qids)
    t2 = time.time()
    tag = tag_for(a.model, a.dataset, a.distractor_docs)
    a.out.mkdir(parents=True, exist_ok=True)
    docs.attrs.update({"model": a.model, "dataset": f"BeIR/{a.dataset}", "encode_seconds": t1 - t0,
                       "distractors": names, "distractor_docs": a.distractor_docs})
    docs.save(a.out / f"{tag}_docs.npz")
    qs.save(a.out / f"{tag}_queries.npz")
    (a.out / f"{tag}_qrels.json").write_text(json.dumps({q: qrels[q] for q in qids}), encoding="utf-8")
    print(json.dumps({"tag": tag, "docs": len(docs), "vectors": docs.num_vectors, "dim": docs.dimension,
                      "queries": len(qs), "doc_seconds": round(t1 - t0, 1), "query_seconds": round(t2 - t1, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
