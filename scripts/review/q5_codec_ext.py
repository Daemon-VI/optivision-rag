"""Codec ladder extension: ColBERTv2-style residual codes, ITQ stacked on the
real pruning pipeline, blank-vs-ink anisotropy, paired bootstrap CIs."""
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
from optivision.pruning.saliency import patch_saliency  # noqa: E402

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
family = np.array([r["type"] for r in rel])
pages = [e.embeddings for e in corpus.encodings]
allv = np.concatenate(pages)
D = allv.shape[1]
mu = allv.mean(0)


def per_query(doc_mats, q_fn=lambda q: q):
    nd, r1, orders = [], [], []
    for qi, q in enumerate(queries):
        qq = q_fn(q)
        sc = np.array([float((qq @ d.T).max(1).sum()) for d in doc_mats])
        order = np.argsort(-sc)
        ranked = [ids[i] for i in order]
        nd.append(ndcg_at_k(ranked, qrels[qi], 5))
        r1.append(1.0 if ranked[0] in qrels[qi] else 0.0)
        orders.append(ranked)
    return np.array(nd), np.array(r1), orders


def boot(a, b, n=4000):
    """paired bootstrap of mean(b - a) over queries"""
    d = b - a
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean()


def itq(v, iters=50, seed=7):
    r = np.random.default_rng(seed)
    R, _ = np.linalg.qr(r.standard_normal((v.shape[1], v.shape[1])))
    for _ in range(iters):
        B = np.sign(v @ R)
        U, _, Wt = np.linalg.svd(B.T @ v)
        R = (U @ Wt).T
    return R


cov = np.cov((allv - mu).T)
_, V = np.linalg.eigh(cov); V = V[:, ::-1]
sample = (allv - mu)[rng.choice(allv.shape[0], 20000, replace=False)]
Vq = V @ itq(sample @ V)
Rr, _ = np.linalg.qr(rng.standard_normal((D, D)))

base_nd, base_r1, base_ord = per_query(pages)
sign_nd, sign_r1, sign_ord = per_query([np.sign(p) for p in pages])
itq_nd, itq_r1, itq_ord = per_query([np.sign((p - mu) @ Vq) for p in pages], q_fn=lambda q: q @ Vq)


def skmeans(v, k, iters=20, seed=0):
    r = np.random.default_rng(seed)
    c = v[r.choice(v.shape[0], k, replace=False)].copy()
    for _ in range(iters):
        a = (v @ c.T).argmax(1)
        for j in range(k):
            mem = v[a == j]
            if mem.shape[0]:
                c[j] = mem.mean(0)
            else:
                c[j] = v[r.integers(v.shape[0])]
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return c


def lloyd2(sigma):
    return np.array([-1.510, -0.4528, 0.4528, 1.510]) * sigma, np.array([-0.9816, 0.0, 0.9816]) * sigma


print("=== ColBERTv2-style residual codec on E1 (centroid id + quantised residual), no pruning ===")
print(f"{'codec':<46} {'bytes/vec':>9} {'compr':>6}  nDCG@5  retain   R@1   tau60")
print(f"{'float32':<46} {512:>9} {1.0:>5.1f}x  {base_nd.mean():.4f} {100:6.1f}%  {base_r1.mean():.3f}  1.000")
print(f"{'sign(d) [pipeline]':<46} {16:>9} {32.0:>5.1f}x  {sign_nd.mean():.4f} {100*sign_nd.mean()/base_nd.mean():6.1f}%  {sign_r1.mean():.3f}  "
      f"{np.mean([rank_correlation(a, b, pool=ids) for a, b in zip(base_ord, sign_ord)]):.3f}")
print(f"{'sign(ITQ(d-mu))':<46} {16:>9} {32.0:>5.1f}x  {itq_nd.mean():.4f} {100*itq_nd.mean()/base_nd.mean():6.1f}%  {itq_r1.mean():.3f}  "
      f"{np.mean([rank_correlation(a, b, pool=ids) for a, b in zip(base_ord, itq_ord)]):.3f}")
fit_sample = allv[rng.choice(allv.shape[0], 30000, replace=False)]
for K, idbytes in ((256, 1), (4096, 1.5)):
    C = skmeans(fit_sample, K, seed=1)
    for bits in (1, 2):
        # residual statistics for the global scale
        a = (fit_sample @ C.T).argmax(1)
        res = fit_sample - C[a]
        if bits == 1:
            beta = np.abs(res).mean()  # E|r_j|: the MSE-optimal scale for a sign code
            def enc(p):
                k = (p @ C.T).argmax(1); r = p - C[k]
                return (C[k] + beta * np.sign(r)).astype(np.float32)
        else:
            lv, th = lloyd2(res.std())
            def enc(p):
                k = (p @ C.T).argmax(1); r = p - C[k]
                return (C[k] + lv[np.digitize(r, th)]).astype(np.float32)
        nd, r1, od = per_query([enc(p) for p in pages])
        bpv = 16 * bits + idbytes
        tau = np.mean([rank_correlation(x, y, pool=ids) for x, y in zip(base_ord, od)])
        print(f"{f'residual K={K}, {bits}-bit residual':<46} {bpv:>9.1f} {512/bpv:>5.1f}x  {nd.mean():.4f} {100*nd.mean()/base_nd.mean():6.1f}%  {r1.mean():.3f}  {tau:.3f}")

print("\n=== paired bootstrap over 72 queries (nDCG@5 difference, 95% CI, P(>0)) ===")
for label, a, b in (("sign vs float", base_nd, sign_nd), ("ITQ vs sign", sign_nd, itq_nd), ("ITQ vs float", base_nd, itq_nd)):
    m, lo, hi, p = boot(a, b)
    print(f"  {label:<16} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  P(>0)={p:.3f}")
for f in ("precise", "topical"):
    msk = family == f
    m, lo, hi, p = boot(sign_nd[msk], itq_nd[msk])
    print(f"  ITQ vs sign, {f:<8} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  P(>0)={p:.3f}  (n={msk.sum()})")

print("\n=== ITQ stacked on the real pruning pipeline (pixel saliency + redundancy on original vectors, codec after) ===")
cfg = Config(); cfg.pruning.min_keep = 8
rows_, cols_ = corpus.encodings[0].grid.rows, corpus.encodings[0].grid.cols
def prune_all(**over):
    pr = TokenPruner(cfg.with_overrides(pruning={"enabled": True, "spatial": True, "redundancy": True, **over}).pruning)
    return [pr.prune(e, img).embeddings for e, img in zip(corpus.encodings, corpus.images)]
configs = [("optivision (threshold prune)", {}), ("keep-50%", {"keep_ratio": 0.5}), ("keep-30%", {"keep_ratio": 0.3}), ("keep-10%", {"keep_ratio": 0.1})]
print(f"{'config':<30} {'tok/pg':>6}   float    sign (retain)    ITQ-sign (retain)   2bit+rot (retain)")
lv, th = lloyd2((allv - mu).std())
for label, over in configs:
    mats = prune_all(**over)
    tok = np.mean([m.shape[0] for m in mats])
    f_nd, _, _ = per_query(mats)
    s_nd, _, _ = per_query([np.sign(m) for m in mats])
    i_nd, _, _ = per_query([np.sign((m - mu) @ Vq) for m in mats], q_fn=lambda q: q @ Vq)
    t_nd, _, _ = per_query([lv[np.digitize((m - mu) @ Rr, th)].astype(np.float32) for m in mats], q_fn=lambda q: q @ Rr)
    b = base_nd.mean()
    print(f"{label:<30} {tok:6.0f}   {f_nd.mean():.4f}   {s_nd.mean():.4f} ({100*s_nd.mean()/b:5.1f}%)   {i_nd.mean():.4f} ({100*i_nd.mean()/b:5.1f}%)    {t_nd.mean():.4f} ({100*t_nd.mean()/b:5.1f}%)")

print("\n=== blank vs ink patches: where the anisotropy lives ===")
blank_v, ink_v = [], []
for e, img in zip(corpus.encodings, corpus.images):
    s = patch_saliency(img, rows_, cols_).reshape(-1)
    g = e.embeddings[e.grid.token_index.reshape(-1)]
    blank_v.append(g[s <= 0.02]); ink_v.append(g[s > 0.02])
blank_v = np.concatenate(blank_v); ink_v = np.concatenate(ink_v)
for name, v in (("blank", blank_v), ("ink", ink_v)):
    pick = v[rng.choice(v.shape[0], 3000, replace=False)]
    s = pick @ pick.T; iu = np.triu_indices(3000, 1)
    print(f"  {name:<5} {v.shape[0]:>6} patches  mean pairwise cos {s[iu].mean():.3f}  p50 {np.median(s[iu]):.3f}  ||mean|| {np.linalg.norm(v.mean(0)):.3f}")
pb = blank_v[rng.choice(blank_v.shape[0], 3000, replace=False)]; pi = ink_v[rng.choice(ink_v.shape[0], 3000, replace=False)]
print(f"  blank-vs-ink mean cos {(pb @ pi.T).mean():.3f};  cos(mean_blank, mean_ink) {np.dot(blank_v.mean(0), ink_v.mean(0))/np.linalg.norm(blank_v.mean(0))/np.linalg.norm(ink_v.mean(0)):.3f}")
# winner's-curse check: under sign codes, how often is the per-token arg max on a blank patch vs float
fl = bl = 0; tot = 0
for q in queries[:36]:
    for e, img in zip(corpus.encodings, corpus.images):
        s = patch_saliency(img, rows_, cols_).reshape(-1)
        g = e.embeddings[e.grid.token_index.reshape(-1)]
        blank = s <= 0.02
        fl += blank[(q @ g.T).argmax(1)].sum(); bl += blank[(q @ np.sign(g).T).argmax(1)].sum(); tot += q.shape[0]
print(f"  query tokens whose grid arg max lands on a BLANK patch: float {100*fl/tot:.1f}%   sign {100*bl/tot:.1f}%   (distractor promotion)")
