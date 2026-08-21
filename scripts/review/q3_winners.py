"""Q3: the winner set. (a) which token types win; (b) winner fraction as a
coverage curve in the number of queries; (c) does held-out retention survive a
split that does not leak subjects; (d) uniform-random selection as the missing
control; (e) pooling-to-budget against selection-to-budget at matched k.
Proper nDCG@5 (multi-relevant) via metrics.evaluate throughout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, "C:/Users/Rishi/optivision-rag/src")
from optivision.bench import EncodedCorpus  # noqa: E402
from optivision.metrics import evaluate  # noqa: E402
from optivision.pruning import codebook_saliency, fit_codebook  # noqa: E402
from optivision.pruning.saliency import patch_saliency  # noqa: E402

ROOT = Path("C:/Users/Rishi/optivision-rag")
rng = np.random.default_rng(7)
corpus = EncodedCorpus.load(ROOT / "data/cache/colsmol.npz")
zq = np.load(ROOT / "data/cache/colsmol.queries.npz", allow_pickle=True)
qtexts = json.loads(str(zq["texts"]))
queries = [np.asarray(zq[f"q_{i}"], np.float32) for i in range(len(qtexts))]
rel = json.loads((ROOT / "data/corpus/queries.json").read_text())
ids = [e.ref.page_id for e in corpus.encodings]
qrels = {i: set(r["relevant"]) for i, r in enumerate(rel)}
family = np.array([r["type"] for r in rel])
subject = np.array([r["query"] if r["type"] == "topical" else r["query"].rsplit(" ", 1)[0] for r in rel])
full = [e.embeddings for e in corpus.encodings]
grids = [e.grid.token_index.reshape(-1) for e in corpus.encodings]
textidx = [e.text_token_index for e in corpus.encodings]
gridv = [e.embeddings[g] for e, g in zip(corpus.encodings, grids)]
textv = [e.embeddings[t] for e, t in zip(corpus.encodings, textidx)]
n_grid = gridv[0].shape[0]
NQ = len(queries)

# token types inside the 107 non-grid tokens: thumbnail = the 64-run, markers = the rest
def token_type(e):
    t = np.zeros(e.embeddings.shape[0], dtype="<U6")
    t[:] = "grid"
    ti = e.text_token_index
    runs = np.split(ti, np.flatnonzero(np.diff(ti) != 1) + 1)
    for r in runs:
        t[r] = "thumb" if len(r) >= 64 else "marker"
    if len(runs[-1]) >= 64:
        t[runs[-1][64:]] = "instr"  # trailing text after the thumbnail
    return t
types = [token_type(e) for e in corpus.encodings]
counts = {k: int((types[0] == k).sum()) for k in ("grid", "thumb", "marker", "instr")}
print("token types per page:", counts)


def ndcg(doc_mats, qidx, ks=(1, 5)):
    runs, qr = {}, {}
    for qi in qidx:
        q = queries[qi]
        sc = np.array([-1e9 if d.shape[0] == 0 else float((q @ d.T).max(1).sum()) for d in doc_mats])
        runs[qi] = [ids[i] for i in np.argsort(-sc)[:10]]
        qr[qi] = qrels[qi]
    m = evaluate(runs, qr, ks=ks)
    return m["ndcg@5"], m["recall@1"]


# ------------------------------------------------------ (a) who wins, by type
print("\n=== (a) which token types win the MaxSim arg max (all 72 queries x 60 pages) ===")
win_type = {k: 0 for k in counts}
score_type = {k: 0.0 for k in counts}
for q in queries:
    for d, t in zip(full, types):
        sims = q @ d.T
        am = sims.argmax(1); mx = sims.max(1)
        for k in counts:
            sel = t[am] == k
            win_type[k] += int(sel.sum()); score_type[k] += float(mx[sel].sum())
tw = sum(win_type.values()); ts = sum(score_type.values())
for k in counts:
    print(f"  {k:<7} {counts[k]:>4} tokens ({100*counts[k]/875:4.1f}% of page)  wins {100*win_type[k]/tw:5.1f}%  score share {100*score_type[k]/ts:5.1f}%")

# which *query* tokens do the winning: by position (the augmentation tail)
print("\n  per query-token position: mean max-sim and whether its arg max is a non-grid token (first 72 queries)")
L = max(q.shape[0] for q in queries)
pos_ng = np.zeros(L); pos_n = np.zeros(L); pos_mx = np.zeros(L)
for q in queries:
    for d, t in zip(full, types):
        sims = q @ d.T; am = sims.argmax(1); mx = sims.max(1)
        for j in range(q.shape[0]):
            pos_ng[j] += (t[am[j]] != "grid"); pos_n[j] += 1; pos_mx[j] += mx[j]
print("  pos :", " ".join(f"{j:>4d}" for j in range(L)))
print("  nong:", " ".join(f"{100*pos_ng[j]/max(pos_n[j],1):4.0f}" for j in range(L)))
print("  mxsm:", " ".join(f"{pos_mx[j]/max(pos_n[j],1):4.2f}" for j in range(L)))

# ------------------------------------------------- (b) coverage curve
print("\n=== (b) winner fraction (grid patches) vs number of queries used ===")
def winners(qidx, mats):
    w = [np.zeros(m.shape[0], bool) for m in mats]
    for qi in qidx:
        q = queries[qi]
        for pi, m in enumerate(mats):
            w[pi][(q @ m.T).argmax(1)] = True
    return w
for k in (1, 2, 4, 8, 16, 36, 72):
    fr = []
    for _ in range(5 if k < 72 else 1):
        sub = rng.choice(NQ, k, replace=False)
        w = winners(sub, gridv)
        fr.append(np.mean([x.mean() for x in w]))
    print(f"  {k:>3} queries: {100*np.mean(fr):5.1f}% of grid patches ever win")
# distinct winners per query token: how concentrated are wins
w1 = winners(range(NQ), gridv)
print(f"  all 72 queries, {sum(q.shape[0] for q in queries)} query tokens: {100*np.mean([x.mean() for x in w1]):.1f}% of grid patches; "
      f"tokens per winner {sum(q.shape[0] for q in queries)*60/sum(x.sum() for x in w1):.1f}")

# ------------------------------------------------- (c) leakage split
print("\n=== (c) held-out retention of a winner-only index under different splits (grid winners + all 107 text tokens kept) ===")
def heldout(fit, held, label):
    w = winners(fit, gridv)
    mats_w = [np.concatenate([g[m], t]) for g, m, t in zip(gridv, w, textv)]
    mats_f = [np.concatenate([g, t]) for g, t in zip(gridv, textv)]
    b = ndcg(mats_f, held); o = ndcg(mats_w, held)
    # and without the text tokens, to see what the grid winners alone carry
    b2 = ndcg(gridv, held); o2 = ndcg([g[m] for g, m in zip(gridv, w)], held)
    kept = np.mean([m.mean() for m in w])
    print(f"  {label:<44} winners {100*kept:4.1f}%  full {b[0]:.4f}  winners-only {o[0]:.4f} ({100*o[0]/b[0]:5.1f}%) | grid-only: full {b2[0]:.4f} winners {o2[0]:.4f} ({100*o2[0]/b2[0]:5.1f}%)")
perm = rng.permutation(NQ)
heldout(list(perm[:36]), list(perm[36:]), "random half split (the ROADMAP number)")
pre = list(np.flatnonzero(family == "precise")); top = list(np.flatnonzero(family == "topical"))
heldout(pre, top, "fit on 60 precise -> test on 12 topical")
heldout(top, pre, "fit on 12 topical -> test on 60 precise")
subs = sorted(set(subject)); half = set(rng.permutation(subs)[: len(subs) // 2])
fit = [i for i in range(NQ) if subject[i] in half]; held = [i for i in range(NQ) if subject[i] not in half]
heldout(fit, held, f"subject-disjoint: fit {len(fit)} q / test {len(held)} q")
# leave-one-document-kind-out is not possible (subjects cycle over kinds); do two more subject draws
for s in (1, 2):
    r2 = np.random.default_rng(s); half = set(r2.permutation(subs)[: len(subs) // 2])
    fit = [i for i in range(NQ) if subject[i] in half]; held = [i for i in range(NQ) if subject[i] not in half]
    heldout(fit, held, f"subject-disjoint, draw {s}")

# -------------------------------------- (d)+(e) selectors vs random vs pooling
print("\n=== (d)/(e) selection vs random vs pooling at matched k, held-out half, grid tokens (+text tokens kept) ===")
fit, held = list(perm[:36]), list(perm[36:])
orc = winners(fit, gridv)
sample = np.concatenate(gridv); sample = sample[rng.choice(sample.shape[0], 20000, replace=False)]
books = {s: fit_codebook(sample, size=256, seed=7, source=s) for s in ("farthest", "random")}
rows, cols = corpus.encodings[0].grid.rows, corpus.encodings[0].grid.cols
sal = {
    "pixel": [patch_saliency(img, rows, cols).reshape(-1) for img in corpus.images],
    "probe:farthest": [codebook_saliency(g, books["farthest"], rows, cols).reshape(-1) for g in gridv],
    "probe:random": [codebook_saliency(g, books["random"], rows, cols).reshape(-1) for g in gridv],
    "probe:4096rand": None,
}
# many random probes -> a less noisy estimate of the normal-cone measure
big = fit_codebook(sample, size=4096, seed=11, source="random")
sal["probe:4096rand"] = [codebook_saliency(g, big, rows, cols).reshape(-1) for g in gridv]

def topk(s, k):
    m = np.zeros(s.shape[0], bool); m[np.argpartition(-s, k - 1)[:k]] = True; return m

def skmeans(v, k, iters=15, seed=0):
    r = np.random.default_rng(seed)
    c = v[r.choice(v.shape[0], k, replace=False)].copy()
    for _ in range(iters):
        a = (v @ c.T).argmax(1)
        for j in range(k):
            mem = v[a == j]
            if mem.shape[0]:
                c[j] = mem.mean(0)
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return c

def agglo(v, k):
    Z = linkage(v, method="average", metric="cosine")
    lab = fcluster(Z, t=k, criterion="maxclust")
    c = np.stack([v[lab == j].mean(0) for j in np.unique(lab)])
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)

def ev(mats, label, k, with_text=True, binary=False):
    if with_text:
        mats = [np.concatenate([m, t]) for m, t in zip(mats, textv)]
    if binary:
        mats = [np.sign(m).astype(np.float32) for m in mats]
    nd, r1 = ndcg(mats, held)
    return nd

for frac in (0.3, 0.1):
    k = round(frac * n_grid)
    print(f"\n  budget keep {int(frac*100)}% = {k} of {n_grid} grid patches        float, +text | float, grid-only | sign, +text")
    ref_t = ev(gridv, "full", k); ref_g = ev(gridv, "full", k, with_text=False); ref_b = ev(gridv, "full", k, binary=True)
    print(f"  {'full grid (no budget)':<30} {ref_t:.4f}        {ref_g:.4f}            {ref_b:.4f}")
    o = [g[m] for g, m in zip(gridv, orc)]
    print(f"  {'oracle winners (fit half)':<30} {ev(o,'o',k):.4f}        {ev(o,'o',k,False):.4f}            {ev(o,'o',k,binary=True):.4f}   ({100*np.mean([m.mean() for m in orc]):.1f}% kept, not budget-matched)")
    for name, s in sal.items():
        m = [g[topk(x, k)] for g, x in zip(gridv, s)]
        print(f"  {name:<30} {ev(m,name,k):.4f}        {ev(m,name,k,False):.4f}            {ev(m,name,k,binary=True):.4f}")
    vals = []
    for seed in range(3):
        rr = np.random.default_rng(100 + seed)
        m = [g[rr.choice(g.shape[0], k, replace=False)] for g in gridv]
        vals.append((ev(m, "rand", k), ev(m, "rand", k, False), ev(m, "rand", k, binary=True)))
    v = np.mean(vals, 0)
    print(f"  {'uniform random subset (3 seeds)':<30} {v[0]:.4f}        {v[1]:.4f}            {v[2]:.4f}")
    m = [skmeans(g, k, seed=0) for g in gridv]
    print(f"  {'pool: spherical k-means -> k':<30} {ev(m,'km',k):.4f}        {ev(m,'km',k,False):.4f}            {ev(m,'km',k,binary=True):.4f}")
    m = [agglo(g, k) for g in gridv]
    print(f"  {'pool: agglomerative avg-link -> k':<30} {ev(m,'ag',k):.4f}        {ev(m,'ag',k,False):.4f}            {ev(m,'ag',k,binary=True):.4f}")
    # pool only the non-winners, keep the oracle winners: what a selector+pool hybrid could reach
    # (oracle information, so a ceiling not a method)
