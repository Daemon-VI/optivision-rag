"""Margin-triggered rescoring: the decision-margin rule as a query-time tool.

At query time a binary index only has binary scores. Compute the top-1/top-2
margin of the *binary* ranking; when it is below a threshold, rescore the binary
top-k with int8 vectors (stored cold) and return that order; otherwise return the
binary order. Measure nDCG@5 against the fraction of queries rescored, and compare
with (a) rescoring every query (the two-tier ceiling) and (b) rescoring the same
fraction chosen at random (the control: is the margin picking the right queries?).
Runs on the local caches in under a minute.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Rishi/optivision-rag/src")
from optivision.bench import EncodedCorpus  # noqa: E402
from optivision.metrics import ndcg_at_k  # noqa: E402

ROOT = Path("C:/Users/Rishi/optivision-rag")
SCR = Path("C:/Users/MPPSKA~1/AppData/Local/Temp/claude/C--Users-Rishi/2d9f7724-afc4-44a7-8036-3b2977a3e11b/scratchpad")
os.chdir(ROOT)
rng = np.random.default_rng(7)
K = 10  # binary shortlist depth

RUNS = [
    ("ColSmol 1x (E1)", ROOT / "data/cache/colsmol.npz", ROOT / "data/corpus"),
    ("ColSmol 3x", ROOT / "data/cache/colsmol_code3x.npz", ROOT / "data/corpus_code3x"),
    ("ColSmol 0.4x", ROOT / "data/cache/colsmol_code04x.npz", ROOT / "data/corpus_code04x"),
    ("ColPali 1x (E3)", ROOT / "data/cache/colpali_generated.npz", ROOT / "data/corpus"),
    ("ColPali 3x shifted", ROOT / "data/cache/colpali_generated_code3x_shifted.npz", ROOT / "data/corpus"),
    ("ColSmol infovqa", ROOT / "data/cache/colsmol_infovqa_test_subsampled.npz", ROOT / "data/vidore_infovqa_test_subsampled"),
]


def load(cache, corpus_dir):
    corpus = EncodedCorpus.load(cache)
    zq = np.load(cache.with_suffix(".queries.npz"), allow_pickle=True)
    texts = json.loads(str(zq["texts"]))
    queries = [np.asarray(zq[f"q_{i}"], np.float32) for i in range(len(texts))]
    rel = {r["query"]: r for r in json.loads((corpus_dir / "queries.json").read_text())}
    rel = [rel[t] for t in texts]
    ids = [e.ref.page_id for e in corpus.encodings]
    return [e.embeddings for e in corpus.encodings], queries, [set(r["relevant"]) for r in rel], ids


def int8(p):
    return (np.clip(np.round(p / 0.5 * 127), -127, 127) * 0.5 / 127).astype(np.float32)


def scores(pages, q):
    return np.array([float((q @ d.T).max(1).sum()) for d in pages])


def ndcg(order, rel, ids):
    return ndcg_at_k([ids[i] for i in order], rel, 5)


out = {}
print(f"{'cache':<20} {'float':>6} {'binary':>6} {'all rescored':>12} | margin-triggered: fraction rescored -> nDCG@5 (random control at same fraction)")
for label, cache, corpus_dir in RUNS:
    if not cache.exists():
        continue
    pages, queries, rels, ids = load(cache, corpus_dir)
    B = [np.sign(p) for p in pages]
    I8 = [int8(p) for p in pages]
    n = len(queries)
    nd_float, nd_bin, nd_all, m12, orders_bin, orders_resc = [], [], [], [], [], []
    for qi, q in enumerate(queries):
        f = scores(pages, q); b = scores(B, q)
        ob = np.argsort(-b)
        top = ob[:K]
        s8 = np.array([float((q @ I8[i].T).max(1).sum()) for i in top])
        o_resc = np.concatenate([top[np.argsort(-s8)], ob[K:]])
        nd_float.append(ndcg(np.argsort(-f), rels[qi], ids))
        nd_bin.append(ndcg(ob, rels[qi], ids))
        nd_all.append(ndcg(o_resc, rels[qi], ids))
        m12.append((b[ob[0]] - b[ob[1]]) / abs(b[ob[0]]))
        orders_bin.append(ob); orders_resc.append(o_resc)
    nd_float, nd_bin, nd_all, m12 = map(np.array, (nd_float, nd_bin, nd_all, m12))
    base = nd_float.mean()
    line = f"{label:<20} {base:6.4f} {nd_bin.mean():6.4f} {nd_all.mean():12.4f} |"
    res = {"float": float(base), "binary": float(nd_bin.mean()), "all_rescored": float(nd_all.mean()), "curve": []}
    for frac in (0.2, 0.4, 0.6):
        k = int(round(frac * n))
        pick = np.argsort(m12)[:k]  # smallest binary margins first
        nd_trig = nd_bin.copy(); nd_trig[pick] = nd_all[pick]
        rnd = np.mean([np.where(np.isin(np.arange(n), rng.choice(n, k, replace=False)), nd_all, nd_bin).mean() for _ in range(200)])
        res["curve"].append({"fraction": frac, "triggered": float(nd_trig.mean()), "random": float(rnd)})
        line += f"  {int(100*frac)}%: {nd_trig.mean():.4f} ({rnd:.4f})"
    print(line)
    out[label] = res
print(f"\nbinary shortlist depth K={K}; 'all rescored' = two-tier ceiling (every query's top-{K} reordered by int8).")
print("A trigger that works beats the random control at the same fraction and approaches the ceiling well before 100%.")
(ROOT / "reports/margin_rescoring.json").write_text(json.dumps(out, indent=1))
