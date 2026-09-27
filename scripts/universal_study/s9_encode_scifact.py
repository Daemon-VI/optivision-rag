"""Encode BEIR SciFact with a text ColBERT through optivision's SentenceTransformersAdapter.

Run in the isolated st_env (sentence-transformers >= 6) with PYTHONPATH pointing at the
worktree's src, so the adapter under test is the package's own code.
"""
import json
import os
import sys
import time
from pathlib import Path

from optivision.adapters import SentenceTransformersAdapter

D = os.environ.get("OPTIVISION_DATA", "data")  # expects D/scifact/{corpus.json,queries.json,qrels_test.tsv}
os.makedirs(f"{D}/vectors", exist_ok=True)
model_id = sys.argv[1] if len(sys.argv) > 1 else "answerdotai/answerai-colbert-small-v1"
tag = model_id.split("/")[-1]
corpus = json.loads(Path(f"{D}/scifact/corpus.json").read_text(encoding="utf-8"))
queries = {r["_id"]: r["text"] for r in json.loads(Path(f"{D}/scifact/queries.json").read_text(encoding="utf-8"))}
qrels: dict[str, list[str]] = {}
for line in Path(f"{D}/scifact/qrels_test.tsv").read_text(encoding="utf-8").splitlines()[1:]:
    qid, did, score = line.split("\t")
    if int(score) > 0:
        qrels.setdefault(qid, []).append(did)

adapter = SentenceTransformersAdapter(model_id)
texts = [(r["title"] + " " + r["text"]).strip() for r in corpus]
ids = [r["_id"] for r in corpus]
t0 = time.time()
docs = adapter.encode_documents(texts, ids=ids)
t1 = time.time()
qids = sorted(qrels, key=int)
qs = adapter.encode_queries([queries[q] for q in qids], ids=qids)
t2 = time.time()
docs.attrs.update({"model": model_id, "dataset": "BEIR/scifact", "encode_seconds": t1 - t0})
docs.save(f"{D}/vectors/scifact_{tag}_docs.npz")
qs.save(f"{D}/vectors/scifact_{tag}_queries.npz")
Path(f"{D}/vectors/scifact_qrels_test.json").write_text(json.dumps(qrels, indent=0), encoding="utf-8")
print(json.dumps({"model": model_id, "docs": len(docs), "doc_vectors": docs.num_vectors, "dim": docs.dimension,
                  "dtype": str(docs.dtype), "queries": len(qs), "doc_seconds": round(t1 - t0, 1),
                  "query_seconds": round(t2 - t1, 1), **{k: round(v, 3) for k, v in docs.summary().items()
                                                           if k.startswith("norm") or k == "unit_fraction"}}))
