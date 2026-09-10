"""Does the label-free margin predict codec flips as well as the gold margin?

The decision-margin rule (REVIEW section 2) uses m = (s_gold - s_best_other)/s_gold,
which needs relevance labels. A deployment only has float scores, so the quantity
it can compute is the top-1 / top-2 margin m12 = (s_1 - s_2)/s_1. If m12 predicts
which queries a codec flips about as well as m does, then codec cost can be
forecast on any corpus from a sample of queries with no labels, and rescoring can
be triggered per query by m12. Runs on the local caches in seconds.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Rishi/optivision-rag/src")
from optivision.bench import EncodedCorpus  # noqa: E402

ROOT = Path("C:/Users/Rishi/optivision-rag")
SCR = Path("C:/Users/MPPSKA~1/AppData/Local/Temp/claude/C--Users-Rishi/2d9f7724-afc4-44a7-8036-3b2977a3e11b/scratchpad")
os.chdir(ROOT)

RUNS = [
    ("ColSmol 1x (E1)", ROOT / "data/cache/colsmol.npz", ROOT / "data/corpus"),
    ("ColSmol 3x", ROOT / "data/cache/colsmol_code3x.npz", ROOT / "data/corpus_code3x"),
    ("ColSmol 0.4x", ROOT / "data/cache/colsmol_code04x.npz", ROOT / "data/corpus_code04x"),
    ("ColSmol untiled", SCR / "colsmol_notile.npz", ROOT / "data/corpus"),
    ("ColPali 1x (E3)", ROOT / "data/cache/colpali_generated.npz", ROOT / "data/corpus"),
    ("ColPali 3x shifted", ROOT / "data/cache/colpali_generated_code3x_shifted.npz", ROOT / "data/corpus"),
]


def load(cache, corpus_dir):
    corpus = EncodedCorpus.load(cache)
    zq = np.load(cache.with_suffix(".queries.npz"), allow_pickle=True)
    texts = json.loads(str(zq["texts"]))
    queries = [np.asarray(zq[f"q_{i}"], np.float32) for i in range(len(texts))]
    rel = {r["query"]: r for r in json.loads((corpus_dir / "queries.json").read_text())}
    rel = [rel[t] for t in texts]
    ids = [e.ref.page_id for e in corpus.encodings]
    gold = [np.array([i for i, p in enumerate(ids) if p in set(r["relevant"])]) for r in rel]
    return [e.embeddings for e in corpus.encodings], queries, gold


def scores(pages, q):
    return np.array([float((q @ d.T).max(1).sum()) for d in pages])


def auc(pos, neg):
    return float((pos[:, None] > neg[None, :]).mean()) if pos.size and neg.size else float("nan")


print(f"{'cache':<20} {'sigma':>6} {'flips':>5} | gold margin: AUC  flip<0.5s  flip>2s | top1-top2 margin: AUC  flip<0.5s  flip>2s | corr(m, m12)")
rows = {}
for label, cache, corpus_dir in RUNS:
    if not cache.exists():
        continue
    pages, queries, gold = load(cache, corpus_dir)
    sign = [np.sign(p) for p in pages]
    F = np.stack([scores(pages, q) for q in queries])
    S = np.stack([scores(sign, q) for q in queries])
    n = len(queries)
    m = np.zeros(n); m12 = np.zeros(n); sig = np.zeros(n); flip = np.zeros(n, bool); won = np.zeros(n, bool)
    for qi in range(n):
        f, s, g = F[qi], S[qi], gold[qi]
        others = np.setdiff1d(np.arange(len(f)), g)
        gi = g[f[g].argmax()]
        m[qi] = (f[gi] - f[others].max()) / f[gi]
        top = np.argsort(-f)[:2]
        m12[qi] = (f[top[0]] - f[top[1]]) / f[top[0]]
        a, b = np.polyfit(f, s, 1)
        sig[qi] = np.std(s - (a * f + b)) / abs(a) / f[gi]
        won[qi] = f.argmax() in g
        flip[qi] = (f.argmax() in g) != (s.argmax() in g)
    # a flip under the sign code changes the top-1 page; for the label-free version
    # count any change of top-1 identity, which is what a deployment can observe
    flip_any = np.array([F[qi].argmax() != S[qi].argmax() for qi in range(n)])

    def bins(x):
        r = np.abs(x) / np.maximum(sig, 1e-9)
        lo, hi = r < 0.5, r > 2
        return (100 * flip_any[lo].mean() if lo.any() else np.nan, 100 * flip_any[hi].mean() if hi.any() else np.nan)

    a_gold = auc(m[~flip_any], m[flip_any]) if flip_any.any() else np.nan
    a_12 = auc(m12[~flip_any], m12[flip_any]) if flip_any.any() else np.nan
    g_lo, g_hi = bins(m); t_lo, t_hi = bins(m12)
    corr = float(np.corrcoef(m[won], m12[won])[0, 1]) if won.sum() > 2 else np.nan
    rows[label] = dict(sigma=float(np.median(sig)), flips=int(flip_any.sum()), auc_gold=a_gold, auc_m12=a_12,
                       gold_lo=g_lo, gold_hi=g_hi, m12_lo=t_lo, m12_hi=t_hi, corr_won=corr)
    print(f"{label:<20} {np.median(sig):6.3f} {int(flip_any.sum()):>5} |   {a_gold:4.2f}    {g_lo:5.0f}%    {g_hi:5.0f}% |   {a_12:4.2f}    {t_lo:5.0f}%    {t_hi:5.0f}% |  {corr:+.2f}")

print("\nflip = the float top-1 page differs from the sign-code top-1 page (label-free, what a deployment can see).")
print("AUC = how well the margin ranks non-flipped above flipped queries. corr = correlation of m and m12 on queries the float retriever wins (there m12 == m when the runner-up is the best distractor).")
(ROOT / "reports/labelfree_margin.json").write_text(json.dumps(rows, indent=1))
