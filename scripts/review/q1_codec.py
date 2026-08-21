"""Q1: is the one-bit loss a property of per-vector geometry (rho) or of the
*distribution* of vectors (anisotropy / dead bits)? And does fixing the latter
recover the loss? All on E1's cache, no pruning, every token kept, so the codec
is the only variable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Rishi/optivision-rag/src")
from optivision.metrics import evaluate, rank_correlation  # noqa: E402

ROOT = Path("C:/Users/Rishi/optivision-rag")
rng = np.random.default_rng(7)

z = np.load(ROOT / "data/cache/colsmol.npz", allow_pickle=True)
zq = np.load(ROOT / "data/cache/colsmol.queries.npz", allow_pickle=True)
meta = json.loads(str(z["meta"]))
n = len(meta)
pages = [np.asarray(z[f"emb_{i}"], np.float32) for i in range(n)]
grids = [np.asarray(z[f"grid_{i}"]) for i in range(n)]
texts = [np.asarray(z[f"text_{i}"]) for i in range(n)]
ids = [f"{m['ref']['doc_id']}::p{m['ref']['page_no']}" for m in meta]
qtexts = json.loads(str(zq["texts"]))
queries = [np.asarray(zq[f"q_{i}"], np.float32) for i in range(len(qtexts))]
rel = json.loads((ROOT / "data/corpus/queries.json").read_text())
assert [r["query"] for r in rel] == qtexts
qrels = {f"q{i}": set(r["relevant"]) for i, r in enumerate(rel)}
family = np.array([r["type"] for r in rel])
D = pages[0].shape[1]

allv = np.concatenate(pages)
grid_mask = np.zeros(allv.shape[0], bool)
off = 0
for p, g in zip(pages, grids):
    grid_mask[off + g.reshape(-1)] = True
    off += p.shape[0]
gridv = allv[grid_mask]

print(f"E1: {n} pages x {pages[0].shape[0]} tokens ({gridv.shape[0]} grid patches), dim {D}")

# ------------------------------------------------------------------ geometry
def describe(v: np.ndarray, label: str) -> np.ndarray:
    mu = v.mean(0)
    sd = v.std(0)
    p_pos = (v > 0).mean(0)
    h = -(p_pos * np.log2(np.clip(p_pos, 1e-9, 1)) + (1 - p_pos) * np.log2(np.clip(1 - p_pos, 1e-9, 1)))
    rho = np.abs(v).sum(1) / (np.linalg.norm(v, axis=1) * np.sqrt(D))
    pick = rng.choice(v.shape[0], 4000, replace=False)
    s = v[pick] @ v[pick].T
    iu = np.triu_indices(4000, 1)
    cov = np.cov((v - mu).T)
    ev = np.linalg.eigvalsh(cov)[::-1]
    pr = ev.sum() ** 2 / (ev**2).sum()
    r = v - mu
    rho_c = np.abs(r).sum(1) / (np.linalg.norm(r, axis=1) * np.sqrt(D))
    print(f"\n[{label}]")
    print(f"  rho(d)             mean {rho.mean():.4f}   (random ref {np.sqrt(2/np.pi):.4f})")
    print(f"  rho(d - mu)        mean {rho_c.mean():.4f}")
    print(f"  ||mu||             {np.linalg.norm(mu):.4f}   (unit vectors: 0 = isotropic, 1 = all identical)")
    print(f"  mean pairwise cos  {s[iu].mean():.4f}   p95 {np.percentile(s[iu],95):.4f}")
    print(f"  |mu_k|/sigma_k     median {np.median(np.abs(mu)/sd):.3f}   max {np.max(np.abs(mu)/sd):.3f}")
    print(f"  sign entropy/bit   mean {h.mean():.3f} bits   sum {h.sum():.1f} of {D}")
    print(f"  dead bits (H<0.5)  {(h < 0.5).sum()}   near-dead (H<0.8) {(h < 0.8).sum()}")
    print(f"  p(sign=+) extremes {np.sort(p_pos)[:5].round(3)} ... {np.sort(p_pos)[-5:].round(3)}")
    print(f"  cov eigen: top-1 share {ev[0]/ev.sum():.3f}  top-8 share {ev[:8].sum()/ev.sum():.3f}  participation ratio {pr:.1f} of {D}")
    print(f"  residual norm ||d-mu||  mean {np.linalg.norm(r,axis=1).mean():.3f}  p05 {np.percentile(np.linalg.norm(r,axis=1),5):.3f}  p95 {np.percentile(np.linalg.norm(r,axis=1),95):.3f}")
    return mu

mu_all = describe(allv, "all 875 tokens/page")
mu_grid = describe(gridv, "768 grid patches only")
describe(allv[~grid_mask], "107 thumbnail/marker/instruction tokens only")

# ------------------------------------------------------------------ codecs
def lloyd_max_2bit(sigma: float):
    # optimal 4-level quantiser for N(0, sigma^2): thresholds +-0.9816s, levels +-0.4528s, +-1.510s
    return np.array([-1.510, -0.4528, 0.4528, 1.510]) * sigma, np.array([-0.9816, 0.0, 0.9816]) * sigma

def itq(v: np.ndarray, iters: int = 50, seed: int = 7) -> np.ndarray:
    """Return orthogonal R such that sign(v @ R) is the ITQ code (v already centred/rotated)."""
    r = np.random.default_rng(seed)
    R, _ = np.linalg.qr(r.standard_normal((v.shape[1], v.shape[1])))
    for _ in range(iters):
        B = np.sign(v @ R)
        U, _, Wt = np.linalg.svd(B.T @ v)
        R = (U @ Wt).T
    return R

def page_scores(qs, docs):
    """MaxSim of one query against every page given per-page doc matrices."""
    return np.array([float((qs @ d.T).max(1).sum()) for d in docs], dtype=np.float64)

def run(label, doc_mats, q_fn=lambda q: q, weights=None):
    """doc_mats: per page matrix used on the doc side; q_fn: transform of query."""
    runs, deep = {}, {}
    for qi, q in enumerate(queries):
        qq = q_fn(q)
        if weights is None:
            sc = page_scores(qq, doc_mats)
        else:
            sc = np.array([float(((qq @ d.T) * w[None, :]).max(1).sum()) for d, w in zip(doc_mats, weights)])
        order = np.argsort(-sc)
        deep[f"q{qi}"] = [ids[i] for i in order]
        runs[f"q{qi}"] = [ids[i] for i in order[:10]]
    m = evaluate(runs, qrels, ks=(1, 5))
    return m, deep

# baseline float
base_m, base_deep = run("float", pages)

def report(label, m, deep, extra=""):
    taus = np.mean([rank_correlation(base_deep[q], deep[q], pool=ids) for q in deep])
    top1 = np.mean([base_deep[q][0] == deep[q][0] for q in deep])
    ov5 = np.mean([len(set(base_deep[q][:5]) & set(deep[q][:5])) / 5 for q in deep])
    # per family
    fam = {}
    for f in ("precise", "topical"):
        qs_f = [f"q{i}" for i in np.flatnonzero(family == f)]
        mf = evaluate({q: deep[q][:10] for q in qs_f}, {q: qrels[q] for q in qs_f}, ks=(1, 5))
        fam[f] = (mf["ndcg@5"], mf["recall@1"])
    print(f"{label:<34} nDCG@5 {m['ndcg@5']:.4f} ({100*m['ndcg@5']/base_m['ndcg@5']:5.1f}%)  R@1 {m['recall@1']:.3f}  "
          f"tau60 {taus:.3f}  top1= {top1:.2f}  ov@5 {ov5:.2f} | precise {fam['precise'][0]:.3f}/{fam['precise'][1]:.3f}  "
          f"topical {fam['topical'][0]:.3f}/{fam['topical'][1]:.3f}{extra}")

print("\n=== codec ladder, no pruning, 875 tokens/page (E1 ColSmol) ===")
print(f"{'codec':<34} {'quality':<40} {'agreement with float':<28} | per family nDCG@5/R@1")
report("float32", base_m, base_deep)

# sign (what the pipeline does)
sgn = [np.sign(p).astype(np.float32) for p in pages]
report("sign(d)            [pipeline]", *run("sign", sgn))

# centred sign, global mean over all tokens
mu = mu_all
sgn_c = [np.sign(p - mu).astype(np.float32) for p in pages]
report("sign(d - mu)       centred", *run("sign-c", sgn_c))

# centred sign with per-vector residual norm (2 extra bytes/vector as fp16)
norms = [np.linalg.norm(p - mu, axis=1).astype(np.float16).astype(np.float32) for p in pages]
report("sign(d - mu) * ||d-mu||  (+2B/vec)", *run("sign-c-norm", sgn_c, weights=norms))

# per-token-type mean: grid tokens vs non-grid tokens get their own mu
sgn_c2 = []
for p, g in zip(pages, grids):
    gm = np.zeros(p.shape[0], bool); gm[g.reshape(-1)] = True
    c = np.where(gm[:, None], p - mu_grid, p - allv[~grid_mask].mean(0))
    sgn_c2.append(np.sign(c).astype(np.float32))
report("sign(d - mu_type)  two means", *run("sign-c2", sgn_c2))

# centred + random rotation (balances variance across bits in expectation)
Rr, _ = np.linalg.qr(rng.standard_normal((D, D)))
sgn_rr = [np.sign((p - mu) @ Rr).astype(np.float32) for p in pages]
report("sign(R_rand (d - mu))", *run("sign-rr", sgn_rr, q_fn=lambda q: q @ Rr))

# centred + PCA rotation (unbalanced bits, expected to be worse)
cov = np.cov((allv - mu).T)
_, V = np.linalg.eigh(cov)
V = V[:, ::-1]
sgn_pca = [np.sign((p - mu) @ V).astype(np.float32) for p in pages]
report("sign(PCA (d - mu))", *run("sign-pca", sgn_pca, q_fn=lambda q: q @ V))

# ITQ on the PCA-projected centred data
sample = (allv - mu)[rng.choice(allv.shape[0], 20000, replace=False)] @ V
Ritq = itq(sample)
Vq = V @ Ritq
sgn_itq = [np.sign((p - mu) @ Vq).astype(np.float32) for p in pages]
report("sign(ITQ (d - mu))", *run("sign-itq", sgn_itq, q_fn=lambda q: q @ Vq))

# ITQ + residual norm
report("sign(ITQ (d - mu)) * ||d-mu||", *run("sign-itq-norm", sgn_itq, q_fn=lambda q: q @ Vq, weights=norms))

# 2-bit Lloyd-Max on centred components (global sigma, nothing stored per vector)
sig = (allv - mu).std()
levels, thr = lloyd_max_2bit(sig)
def q2(v):
    return levels[np.digitize(v, thr)].astype(np.float32)
two = [q2(p - mu) for p in pages]
report("2-bit Lloyd-Max (d - mu)", *run("2bit", two))
two_r = [q2((p - mu) @ Rr) for p in pages]
report("2-bit Lloyd-Max (R_rand (d - mu))", *run("2bit-rr", two_r, q_fn=lambda q: q @ Rr))

# int8 as the repo does it
INT8 = 0.5
i8 = [(np.clip(np.round(p / INT8 * 127), -127, 127) * INT8 / 127).astype(np.float32) for p in pages]
report("int8 (repo codec)", *run("int8", i8))

# ----------------------------------------------------------- SNR analysis
print("\n=== why the ranking moves: per-query signal-to-noise of the codec ===")
def snr_table(label, doc_mats, q_fn=lambda q: q):
    rows = []
    for qi, q in enumerate(queries):
        s = page_scores(q, pages)
        b = page_scores(q_fn(q), doc_mats)
        A = np.vstack([s, np.ones_like(s)]).T
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = b - A @ coef
        order_s = np.argsort(-s); order_b = np.argsort(-b)
        tau = rank_correlation([ids[i] for i in order_s], [ids[i] for i in order_b], pool=ids)
        gap = s[order_s[0]] - s[order_s[1]]
        rows.append((s.std(), resid.std(), tau, gap, coef[0]))
    r = np.array(rows)
    snr = r[:, 0] / r[:, 1]
    c = np.corrcoef(np.log(snr), r[:, 2])[0, 1]
    print(f"{label:<28} signal std(s) {r[:,0].mean():.3f}  noise std(resid) {r[:,1].mean():.3f}  "
          f"SNR {np.median(snr):.2f}  mean tau {r[:,2].mean():.3f}  corr(log SNR, tau) {c:.2f}  "
          f"slope {r[:,4].mean():.2f}")
    for f in ("precise", "topical"):
        m = family == f
        print(f"    {f:<8} signal {r[m,0].mean():.3f} noise {r[m,1].mean():.3f} SNR {np.median(snr[m]):.2f} tau {r[m,2].mean():.3f}  top1-gap {r[m,3].mean():.3f}")
snr_table("sign(d)", sgn)
snr_table("sign(d - mu)", sgn_c)
snr_table("sign(ITQ(d - mu))", sgn_itq, q_fn=lambda q: q @ Vq)
snr_table("int8", i8)

# --------------------------------------------------- where the score lives
print("\n=== what the float score is made of (E1) ===")
# share of a page's score contributed by query tokens whose argmax is a non-grid token
share_nongrid, share_const = [], []
for q in queries:
    for p, g in zip(pages, grids):
        gm = np.zeros(p.shape[0], bool); gm[g.reshape(-1)] = True
        sims = q @ p.T
        am = sims.argmax(1)
        mx = sims.max(1)
        share_nongrid.append(mx[~gm[am]].sum() / mx.sum())
print(f"  share of MaxSim score from query tokens whose arg max is a thumbnail/marker/instruction token: "
      f"{np.mean(share_nongrid):.3f}")
# how much of the score is page-invariant: per query, min over pages of each token's max, summed
inv = []
for q in queries:
    per_tok = np.array([(q @ p.T).max(1) for p in pages])  # [pages, tokens]
    inv.append(per_tok.min(0).sum() / per_tok.mean(0).sum())
print(f"  page-invariant floor of the score (sum over tokens of min-over-pages max) / mean score: {np.mean(inv):.3f}")
# near-max multiplicity: on the gold page, how many patches are within 0.05 of the max, per query token
mult = {"precise": [], "topical": []}
for qi, q in enumerate(queries):
    for pid in qrels[f"q{qi}"]:
        p = pages[ids.index(pid)]
        sims = q @ p.T
        mx = sims.max(1, keepdims=True)
        mult[family[qi]].append((sims >= mx - 0.05).sum(1))
for f, v in mult.items():
    v = np.concatenate(v)
    print(f"  {f:<8}: patches within 0.05 of the max on the gold page, per query token: median {np.median(v):.0f}  mean {v.mean():.1f}  p90 {np.percentile(v,90):.0f}")
