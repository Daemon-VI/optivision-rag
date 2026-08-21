"""Codec ladder + embedding geometry on any encode cache, no GPU needed.

Runs on E1 (ColSmol/generated), E3 (ColPali/generated), the E2 ViDoRe caches,
and the code-scale corpora. Everything is scored with exact MaxSim over the
whole corpus, proper multi-relevant nDCG@5, and paired bootstrap CIs against the
plain sign codec, because at 72 queries most differences are noise and the table
should say so.

    python scripts/review/codec_ladder.py --cache data/cache/colsmol.npz --corpus data/corpus
    python scripts/review/codec_ladder.py --cache data/cache/colpali_infovqa_test_subsampled.npz \
        --corpus data/vidore_infovqa_test_subsampled --label "E2 infovqa"

Codecs (bytes per 128-d vector):
    float32 512 | int8 128 | 2-bit Lloyd-Max after a random rotation 32 |
    sign(d) 16 (the pipeline) | sign(d - mu) 16 | sign(ITQ(d - mu)) 16 |
    centroid id + 1-bit residual 17-18 (ColBERTv2-style, K fitted on this corpus)

Geometry: per-vector rho (uninformative, kept for the record), ||mean||, mean
pairwise cosine, dead sign bits, covariance participation ratio, blank-vs-ink
patch cosine, and the distractor-promotion rate (share of query tokens whose best
grid match is a blank patch, float vs sign). See docs/REVIEW-2026-08-21.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from optivision.bench import EncodedCorpus  # noqa: E402
from optivision.config import Config  # noqa: E402
from optivision.metrics import ndcg_at_k, rank_correlation  # noqa: E402
from optivision.pruning import TokenPruner  # noqa: E402
from optivision.pruning.saliency import patch_saliency  # noqa: E402


# ----------------------------------------------------------------- scoring
class Scorer:
    """Exact MaxSim of every query against every page, one matmul per query."""

    def __init__(self, mats: list[np.ndarray]):
        self.D = np.ascontiguousarray(np.concatenate(mats), dtype=np.float32)
        counts = np.array([m.shape[0] for m in mats])
        self.off = np.concatenate([[0], np.cumsum(counts)])[:-1].astype(np.intp)
        self.empty = counts == 0
        # reduceat needs strictly valid starts; an empty page borrows the next start and is masked
        self.starts = np.minimum(self.off, self.D.shape[0] - 1)

    def scores(self, q: np.ndarray) -> np.ndarray:
        sims = q @ self.D.T
        s = np.maximum.reduceat(sims, self.starts, axis=1).sum(0)
        if self.empty.any():
            s[self.empty] = -np.inf
        return s


def evaluate(mats, queries, ids, qrels, q_fn=lambda q: q):
    sc = Scorer(mats)
    nd, r1, h5, top = [], [], [], []
    for qi, q in enumerate(queries):
        s = sc.scores(np.ascontiguousarray(q_fn(q), dtype=np.float32))
        order = np.argsort(-s, kind="stable")
        ranked = [ids[i] for i in order]
        nd.append(ndcg_at_k(ranked, qrels[qi], 5))
        r1.append(float(ranked[0] in qrels[qi]))
        h5.append(float(bool(set(ranked[:5]) & qrels[qi])))
        top.append(order)
    return {"ndcg": np.array(nd), "r1": np.array(r1), "hit5": np.array(h5), "order": top}


def tau_ap(ref_order: np.ndarray, other_order: np.ndarray) -> float:
    n = ref_order.size
    pos = np.empty(n, dtype=np.int64)
    pos[other_order] = np.arange(n)
    p = pos[ref_order]  # rank in `other` of the item at each reference rank
    below = np.tril(p[None, :] < p[:, None], k=-1).sum(1)[1:]  # C(i) for i = 2..n
    return float(2 * np.mean(below / np.arange(1, n)) - 1)


def agreement(base, other, ids):
    taus, taps, top1, ov5 = [], [], [], []
    pool = list(ids)
    for a, b in zip(base["order"], other["order"]):
        taus.append(rank_correlation([ids[i] for i in a], [ids[i] for i in b], pool=pool))
        taps.append(tau_ap(a, b))
        top1.append(float(a[0] == b[0]))
        ov5.append(len(set(a[:5]) & set(b[:5])) / 5)
    return float(np.mean(taus)), float(np.mean(taps)), float(np.mean(top1)), float(np.mean(ov5))


def boot(a, b, rng, n=2000):
    d = b - a
    idx = rng.integers(0, d.size, (n, d.size))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------------ codecs
def itq_rotation(v: np.ndarray, iters: int = 50, seed: int = 7) -> np.ndarray:
    r = np.random.default_rng(seed)
    R, _ = np.linalg.qr(r.standard_normal((v.shape[1], v.shape[1])))
    for _ in range(iters):
        B = np.sign(v @ R)
        U, _, Wt = np.linalg.svd(B.T @ v)
        R = (U @ Wt).T
    return R


def lloyd_max_2bit(sigma: float):
    return (np.array([-1.510, -0.4528, 0.4528, 1.510]) * sigma,
            np.array([-0.9816, 0.0, 0.9816]) * sigma)


def skmeans(v: np.ndarray, k: int, iters: int = 15, seed: int = 1, chunk: int = 16384) -> np.ndarray:
    r = np.random.default_rng(seed)
    c = v[r.choice(v.shape[0], k, replace=False)].copy()
    for _ in range(iters):
        sums = np.zeros_like(c); cnt = np.zeros(k, dtype=np.int64)
        for s in range(0, v.shape[0], chunk):
            blk = v[s:s + chunk]
            a = (blk @ c.T).argmax(1)
            np.add.at(sums, a, blk); cnt += np.bincount(a, minlength=k)
        empty = cnt == 0
        c = sums / np.maximum(cnt, 1)[:, None]
        if empty.any():
            c[empty] = v[r.integers(v.shape[0], size=int(empty.sum()))]
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return c.astype(np.float32)


def assign(v: np.ndarray, c: np.ndarray, chunk: int = 16384) -> np.ndarray:
    out = np.empty(v.shape[0], dtype=np.int64)
    for s in range(0, v.shape[0], chunk):
        out[s:s + chunk] = (v[s:s + chunk] @ c.T).argmax(1)
    return out


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True, help="folder holding queries.json")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--residual-k", type=int, default=None, help="centroids; default min(4096, 16*sqrt(n_vectors))")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--pruned", action="store_true", help="also run the optivision prune config under each codec")
    ap.add_argument("--out", type=Path, default=None, help="write a JSON of every number printed")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    t0 = time.time()

    corpus = EncodedCorpus.load(a.cache)
    qcache = a.cache.with_suffix("").with_suffix(".queries.npz")
    zq = np.load(qcache, allow_pickle=True)
    qtexts = json.loads(str(zq["texts"]))
    queries = [np.asarray(zq[f"q_{i}"], np.float32) for i in range(len(qtexts))]
    rel = json.loads((a.corpus / "queries.json").read_text(encoding="utf-8"))
    by_text = {r["query"]: r for r in rel}
    ids = [e.ref.page_id for e in corpus.encodings]
    idset = set(ids)
    keep = [i for i, t in enumerate(qtexts) if t in by_text and set(by_text[t]["relevant"]) & idset]
    if a.max_queries and len(keep) > a.max_queries:
        keep = sorted(rng.choice(keep, a.max_queries, replace=False))
    queries = [queries[i] for i in keep]
    qrels = [set(by_text[qtexts[i]]["relevant"]) & idset for i in keep]
    family = np.array([by_text[qtexts[i]].get("type", "all") for i in keep])
    pages = [e.embeddings for e in corpus.encodings]
    allv = np.concatenate(pages)
    D = allv.shape[1]
    label = a.label or a.cache.stem
    print(f"{label}: {len(pages)} pages x {pages[0].shape[0]} tokens, {allv.shape[0]:,} vectors, "
          f"{len(queries)} queries" + (f" ({ {str(k): int(v) for k, v in zip(*np.unique(family, return_counts=True))} })" if len(set(family)) > 1 else ""))

    # ------------------------------------------------------------ geometry
    mu = allv.mean(0)
    p_pos = (allv > 0).mean(0)
    H = -(p_pos * np.log2(np.clip(p_pos, 1e-9, 1)) + (1 - p_pos) * np.log2(np.clip(1 - p_pos, 1e-9, 1)))
    pick = allv[rng.choice(allv.shape[0], min(3000, allv.shape[0]), replace=False)]
    iu = np.triu_indices(pick.shape[0], 1)
    cov = np.cov((allv[rng.choice(allv.shape[0], min(100_000, allv.shape[0]), replace=False)] - mu).T)
    ev = np.linalg.eigvalsh(cov)[::-1]
    rho = (np.abs(allv).sum(1) / (np.linalg.norm(allv, axis=1) * np.sqrt(D))).mean()
    geo = {
        "rho": float(rho), "mu_norm": float(np.linalg.norm(mu)), "mean_pairwise_cos": float((pick @ pick.T)[iu].mean()),
        "dead_bits_H<0.5": int((H < 0.5).sum()), "sign_entropy_sum": float(H.sum()),
        "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()), "top8_var_share": float(ev[:8].sum() / ev.sum()),
    }
    print("\ngeometry (all tokens)")
    print(f"  rho = cos(d, sign d)      {geo['rho']:.4f}   (random reference {np.sqrt(2/np.pi):.4f}; uninformative by construction)")
    print(f"  ||mean vector||           {geo['mu_norm']:.3f}   mean pairwise cos {geo['mean_pairwise_cos']:.3f}")
    print(f"  dead sign bits (H<0.5)    {geo['dead_bits_H<0.5']} of {D}   total sign entropy {geo['sign_entropy_sum']:.1f} bits")
    print(f"  covariance                participation ratio {geo['participation_ratio']:.1f} of {D}, top-8 share {geo['top8_var_share']:.2f}")

    # blank vs ink on the grid, and distractor promotion
    blank_v, ink_v, blank_masks, gridv = [], [], [], []
    for e, img in zip(corpus.encodings, corpus.images):
        s = patch_saliency(img, e.grid.rows, e.grid.cols).reshape(-1)
        g = e.embeddings[e.grid.token_index.reshape(-1)]
        b = s <= 0.02
        blank_v.append(g[b]); ink_v.append(g[~b]); blank_masks.append(b); gridv.append(g)
    blank_v = np.concatenate(blank_v); ink_v = np.concatenate(ink_v)
    geo["blank_fraction"] = float(blank_v.shape[0] / (blank_v.shape[0] + ink_v.shape[0]))
    def pcos(v):
        if v.shape[0] < 10:
            return float("nan")
        p = v[rng.choice(v.shape[0], min(2000, v.shape[0]), replace=False)]
        return float((p @ p.T)[np.triu_indices(p.shape[0], 1)].mean())
    geo["blank_pairwise_cos"] = pcos(blank_v); geo["ink_pairwise_cos"] = pcos(ink_v)
    sub = queries[: min(48, len(queries))]
    fl = sg = tot = 0
    sgn_grid = [np.sign(g) for g in gridv]
    for q in sub:
        for g, sgg, b in zip(gridv, sgn_grid, blank_masks):
            fl += int(b[(q @ g.T).argmax(1)].sum()); sg += int(b[(q @ sgg.T).argmax(1)].sum()); tot += q.shape[0]
    geo["blank_argmax_float"] = fl / tot; geo["blank_argmax_sign"] = sg / tot
    print(f"  grid patches blank (pixel) {100*geo['blank_fraction']:.1f}%   pairwise cos: blank {geo['blank_pairwise_cos']:.3f}  ink {geo['ink_pairwise_cos']:.3f}")
    print(f"  query tokens whose best grid match is BLANK: float {100*geo['blank_argmax_float']:.1f}%  sign {100*geo['blank_argmax_sign']:.1f}%   (distractor promotion)")

    # -------------------------------------------------------------- codecs
    print(f"\nfitting codecs ... ", end="", flush=True)
    sample = (allv - mu)[rng.choice(allv.shape[0], min(50_000, allv.shape[0]), replace=False)]
    _, V = np.linalg.eigh(cov); V = np.ascontiguousarray(V[:, ::-1], dtype=np.float32)
    Vq = np.ascontiguousarray(V @ itq_rotation(sample @ V), dtype=np.float32)
    Rr, _ = np.linalg.qr(rng.standard_normal((D, D))); Rr = Rr.astype(np.float32)
    lv, th = lloyd_max_2bit(float(sample.std()))
    codecs = [
        ("float32", 512, lambda p: p, lambda q: q),
        ("int8 (repo)", 128, lambda p: (np.clip(np.round(p / 0.5 * 127), -127, 127) * 0.5 / 127).astype(np.float32), lambda q: q),
        ("2-bit LM, rotated", 32, lambda p: lv[np.digitize((p - mu) @ Rr, th)].astype(np.float32), lambda q: q @ Rr),
        ("sign(d)  [pipeline]", 16, lambda p: np.sign(p).astype(np.float32), lambda q: q),
        ("sign(d - mu)", 16, lambda p: np.sign(p - mu).astype(np.float32), lambda q: q),
        ("sign(ITQ(d - mu))", 16, lambda p: np.sign((p - mu) @ Vq).astype(np.float32), lambda q: q @ Vq),
    ]
    if not a.no_residual:
        K = a.residual_k or int(min(4096, 16 * np.sqrt(allv.shape[0])))
        fitv = allv[rng.choice(allv.shape[0], min(60_000, allv.shape[0]), replace=False)]
        C = skmeans(fitv, K)
        beta = float(np.abs(fitv - C[assign(fitv, C)]).mean())
        def res_enc(p, C=C, beta=beta):
            k = assign(p, C); return (C[k] + beta * np.sign(p - C[k])).astype(np.float32)
        codecs.append((f"centroid(K={K})+1-bit residual", 16 + np.ceil(np.log2(K)) / 8, res_enc, lambda q: q))
    print(f"done ({time.time()-t0:.0f}s)")

    results = {}
    base = None; sign = None
    hdr = f"{'codec':<32} {'B/vec':>5} {'compr':>6}  {'nDCG@5':>7} {'retain':>7} {'R@1':>5} {'hit@5':>5} | {'tau':>5} {'tauAP':>5} {'top1=':>5} {'ov@5':>4} | vs sign: diff [95% CI]"
    print("\n" + hdr); print("-" * len(hdr))
    for name, bpv, enc, qf in codecs:
        r = evaluate([enc(p) for p in pages], queries, ids, qrels, qf)
        if base is None:
            base = r
        if name.startswith("sign(d)"):
            sign = r
        results[name] = r
        row = {"bytes_per_vec": bpv, "ndcg5": float(r["ndcg"].mean()), "retain": float(r["ndcg"].mean() / base["ndcg"].mean()),
               "r1": float(r["r1"].mean()), "hit5": float(r["hit5"].mean())}
        row["tau"], row["tau_ap"], row["top1"], row["ov5"] = agreement(base, r, ids)
        for f in sorted(set(family)):
            m = family == f
            row[f"ndcg5_{f}"] = float(r["ndcg"][m].mean()); row[f"r1_{f}"] = float(r["r1"][m].mean())
        r["row"] = row
        print(f"{name:<32} {bpv:>5.1f} {512/bpv:>5.1f}x  {row['ndcg5']:7.4f} {100*row['retain']:6.1f}% {row['r1']:5.3f} {row['hit5']:5.3f} | "
              f"{row['tau']:5.3f} {row['tau_ap']:5.3f} {row['top1']:5.2f} {row['ov5']:4.2f} |", end="")
        if sign is not None and r is not sign:
            m_, lo, hi = boot(sign["ndcg"], r["ndcg"], rng); row["vs_sign"] = [m_, lo, hi]
            print(f" {m_:+.4f} [{lo:+.4f},{hi:+.4f}]", end="")
        print()
    if len(set(family)) > 1:
        print("\nper query family, nDCG@5 / R@1:")
        for name, r in results.items():
            print(f"  {name:<32}" + "  ".join(f"{f}: {r['row'][f'ndcg5_{f}']:.3f}/{r['row'][f'r1_{f}']:.3f}" for f in sorted(set(family))))

    pruned_rows = {}
    if a.pruned:
        cfg = Config()
        pr = TokenPruner(cfg.with_overrides(pruning={"enabled": True, "spatial": True, "redundancy": True}).pruning)
        mats = [pr.prune(e, img).embeddings for e, img in zip(corpus.encodings, corpus.images)]
        tok = float(np.mean([m.shape[0] for m in mats]))
        print(f"\noptivision prune config ({tok:.0f} tok/pg, {pages[0].shape[0]/tok:.2f}x fewer vectors) under each codec, retention vs float full index:")
        for name, bpv, enc, qf in codecs:
            r = evaluate([enc(m) for m in mats], queries, ids, qrels, qf)
            ret = float(r["ndcg"].mean() / base["ndcg"].mean())
            pruned_rows[name] = {"ndcg5": float(r["ndcg"].mean()), "retain": ret, "compression": float(pages[0].shape[0] * 512 / (tok * bpv))}
            print(f"  {name:<32} nDCG@5 {r['ndcg'].mean():.4f} ({100*ret:5.1f}%)  {pruned_rows[name]['compression']:6.1f}x")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"label": label, "n_pages": len(pages), "n_queries": len(queries), "geometry": geo,
                                     "codecs": {k: v["row"] for k, v in results.items()}, "pruned": pruned_rows}, indent=1),
                         encoding="utf-8")
        print(f"\nwrote {a.out}")
    print(f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
