"""Residual codec, properly: K sweep, centroids fitted on held-out pages,
paired bootstrap, and stacked on the real pruning pipeline."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Rishi/optivision-rag/src")
from optivision.bench import EncodedCorpus  # noqa: E402
from optivision.config import Config  # noqa: E402
from optivision.metrics import ndcg_at_k, rank_correlation  # noqa: E402
from optivision.pruning import TokenPruner  # noqa: E402

ROOT = Path("C:/Users/Rishi/optivision-rag")
os.chdir(ROOT)
rng = np.random.default_rng(7)
corpus = EncodedCorpus.load(ROOT / "data/cache/colsmol.npz")
zq = np.load(ROOT / "data/cache/colsmol.queries.npz", allow_pickle=True)
qtexts = json.loads(str(zq["texts"]))
queries = [np.asarray(zq[f"q_{i}"], np.float32) for i in range(len(qtexts))]
rel = json.loads((ROOT / "data/corpus/queries.json").read_text())
ids = [e.ref.page_id for e in corpus.encodings]
qrels = [set(r["relevant"]) for r in rel]
pages = [e.embeddings for e in corpus.encodings]
allv = np.concatenate(pages)


def per_query(mats, q_fn=lambda q: q):
    nd, r1, od = [], [], []
    for qi, q in enumerate(queries):
        qq = q_fn(q)
        sc = np.array([float((qq @ d.T).max(1).sum()) for d in mats])
        ranked = [ids[i] for i in np.argsort(-sc)]
        nd.append(ndcg_at_k(ranked, qrels[qi], 5)); r1.append(ranked[0] in qrels[qi]); od.append(ranked)
    return np.array(nd), np.array(r1, float), od


def boot(a, b, n=4000):
    d = b - a; idx = rng.integers(0, len(d), (n, len(d))); m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean()


def skmeans(v, k, iters=20, seed=0):
    r = np.random.default_rng(seed)
    c = v[r.choice(v.shape[0], k, replace=False)].copy()
    for _ in range(iters):
        a = (v @ c.T).argmax(1)
        sums = np.zeros_like(c); np.add.at(sums, a, v)
        cnt = np.bincount(a, minlength=k)
        empty = cnt == 0
        c = np.where(empty[:, None], v[r.integers(v.shape[0], size=k)], sums / np.maximum(cnt, 1)[:, None])
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return c.astype(np.float32)


class Residual:
    def __init__(self, fit, K, seed=1):
        self.C = skmeans(fit, K, seed=seed)
        a = (fit @ self.C.T).argmax(1)
        res = fit - self.C[a]
        self.beta = float(np.abs(res).mean())
        self.K = K
    def enc(self, p):
        k = (p @ self.C.T).argmax(1); r = p - self.C[k]
        return (self.C[k] + self.beta * np.sign(r)).astype(np.float32)
    @property
    def bytes(self):
        return 16 + np.ceil(np.log2(self.K)) / 8


base = per_query(pages)
sign = per_query([np.sign(p) for p in pages])
print(f"float nDCG@5 {base[0].mean():.4f}   sign 32x {sign[0].mean():.4f} ({100*sign[0].mean()/base[0].mean():.1f}%)\n")
print("=== K sweep, centroids fitted on a 30k sample of ALL pages (what a deployment does at index time) ===")
print(f"{'K':>6} {'B/vec':>6} {'compr':>6}  nDCG@5  retain   R@1   tau60   vs sign: mean [95% CI] P(>0)")
fit_all = allv[rng.choice(allv.shape[0], 30000, replace=False)]
for K in (256, 1024, 2048, 4096, 8192):
    cb = Residual(fit_all, K)
    nd, r1, od = per_query([cb.enc(p) for p in pages])
    tau = np.mean([rank_correlation(a, b, pool=ids) for a, b in zip(base[2], od)])
    m, lo, hi, p = boot(sign[0], nd)
    print(f"{K:>6} {cb.bytes:>6.1f} {512/cb.bytes:>5.1f}x  {nd.mean():.4f} {100*nd.mean()/base[0].mean():6.1f}%  {r1.mean():.3f}  {tau:.3f}   {m:+.4f} [{lo:+.4f},{hi:+.4f}] {p:.3f}")

print("\n=== centroids fitted on pages 0-29 only, codec applied to all 60 (held-out pages 30-59 never seen by k-means) ===")
fit_half = np.concatenate(pages[:30])
for K in (1024, 4096):
    cb = Residual(fit_half, K)
    nd, r1, od = per_query([cb.enc(p) for p in pages])
    # restrict to queries whose gold pages are all in the held-out half
    held_q = [qi for qi in range(len(queries)) if all(ids.index(g) >= 30 for g in qrels[qi])]
    tau = np.mean([rank_correlation(a, b, pool=ids) for a, b in zip(base[2], od)])
    print(f"K={K:>5}: all queries {nd.mean():.4f} ({100*nd.mean()/base[0].mean():5.1f}%)  R@1 {r1.mean():.3f} tau60 {tau:.3f} | "
          f"{len(held_q)} queries with gold in held-out half: float {base[0][held_q].mean():.4f}  sign {sign[0][held_q].mean():.4f}  residual {nd[held_q].mean():.4f}")

print("\n=== residual K=4096 (1-bit) stacked on the real pruning pipeline ===")
cfg = Config(); cfg.pruning.min_keep = 8
cb = Residual(fit_all, 4096)
def prune_all(**over):
    pr = TokenPruner(cfg.with_overrides(pruning={"enabled": True, "spatial": True, "redundancy": True, **over}).pruning)
    return [pr.prune(e, img).embeddings for e, img in zip(corpus.encodings, corpus.images)]
print(f"{'config':<28} {'tok/pg':>6} {'bytes/pg':>8} {'compr':>7}   float     sign (retain)     residual (retain)")
for label, over in (("optivision (threshold)", {}), ("keep-50%", {"keep_ratio": 0.5}), ("keep-30%", {"keep_ratio": 0.3}), ("keep-10%", {"keep_ratio": 0.1})):
    mats = prune_all(**over)
    tok = np.mean([m.shape[0] for m in mats])
    f = per_query(mats)[0].mean(); s = per_query([np.sign(m) for m in mats])[0].mean(); r = per_query([cb.enc(m) for m in mats])[0].mean()
    b = base[0].mean()
    print(f"{label:<28} {tok:6.0f} {tok*cb.bytes:8.0f} {875*512/(tok*cb.bytes):6.1f}x   {f:.4f}   {s:.4f} ({100*s/b:5.1f}%)   {r:.4f} ({100*r/b:5.1f}%)")
print("\n(for scale: E3 ColPali on these pages loses 1.6 points to raw sign codes; E1's prune+int8 row is 96.0% at 14.2x)")
