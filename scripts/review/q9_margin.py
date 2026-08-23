"""Is the one-bit cost a decision-margin story?

Hypothesis: a codec costs a query when the float retriever's margin between the
gold page and its best distractor is smaller than the score noise the codec
injects. On the generated corpus a precise query is decided by a 3-digit
difference inside a 9-character code, i.e. by a few hundredths of similarity
on two or three query tokens out of ~20 -> margin of order 0.1% of the score.
Any 1-bit code perturbs page scores by more than that.

For each query: float margin m = (s_gold - s_best_other) / s_gold, the codec's
score perturbation sigma (std over pages of the sign score after a linear fit
to the float score, normalised), and whether R@1 flipped. If the story holds,
flips concentrate where m / sigma < ~1, for every variant and every codec.
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
    ("E1 tiled 1x", ROOT / "data/cache/colsmol.npz", ROOT / "data/corpus"),
    ("tiled 3x", ROOT / "data/cache/colsmol_code3x.npz", ROOT / "data/corpus_code3x"),
    ("tiled 0.4x", ROOT / "data/cache/colsmol_code04x.npz", ROOT / "data/corpus_code04x"),
    ("3x shifted", ROOT / "data/cache/colsmol_code3x_shifted.npz", ROOT / "data/corpus_code3x_shifted"),
    ("0.4x incl. label", ROOT / "data/cache/colsmol_code04x_label.npz", ROOT / "data/corpus_code04x_label"),
    ("untiled", SCR / "colsmol_notile.npz", ROOT / "data/corpus"),
    # ColPali caches from the Kaggle archive (queries.json is identical across the
    # generated corpora, so data/corpus serves all of them)
    ("ColPali generated (E3)", ROOT / "data/cache/colpali_generated.npz", ROOT / "data/corpus"),
    ("ColPali 3x shifted", ROOT / "data/cache/colpali_generated_code3x_shifted.npz", ROOT / "data/corpus"),
    ("ColPali 3x clean", ROOT / "data/cache/colpali_generated_code3x.npz", ROOT / "data/corpus"),
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
    return [e.embeddings for e in corpus.encodings], queries, gold, np.array([r["type"] for r in rel])


def scores(pages, q):
    return np.array([float((q @ d.T).max(1).sum()) for d in pages])


def int8(p):
    return (np.clip(np.round(p / 0.5 * 127), -127, 127) * 0.5 / 127).astype(np.float32)


def lloyd2(p, sigma=0.088):  # 2-bit per dim, Lloyd-Max levels for a Gaussian with this std
    lv = np.array([-1.51, -0.45, 0.45, 1.51]) * sigma
    th = np.array([-0.98, 0.0, 0.98]) * sigma
    return lv[np.digitize(p, th)].astype(np.float32)


summary = {}
for label, cache, corpus_dir in RUNS:
    if not cache.exists():
        continue
    pages, queries, gold, family = load(cache, corpus_dir)
    allv = np.concatenate(pages)
    mu = allv.mean(0)
    sig = float(allv.std())
    codecs = {
        "sign(d)": [np.sign(p) for p in pages],
        "sign(d-mu)": [np.sign(p - mu) for p in pages],
        "2-bit": [lloyd2(p, sig) for p in pages],
        "int8": [int8(p) for p in pages],
    }
    F = np.stack([scores(pages, q) for q in queries])  # queries x pages
    res = {}
    for cname, mats in codecs.items():
        C = np.stack([scores(mats, q) for q in queries])
        rows = []
        for qi in range(len(queries)):
            f, c, g = F[qi], C[qi], gold[qi]
            others = np.setdiff1d(np.arange(len(f)), g)
            gi = g[f[g].argmax()]
            m = (f[gi] - f[others].max()) / f[gi]
            # codec noise: residual of the codec score around its linear fit to the float score
            a, b = np.polyfit(f, c, 1)
            sigma = float(np.std(c - (a * f + b)) / abs(a) / f[gi])
            won_f = f.argmax() in g
            won_c = c.argmax() in g
            rows.append((m, sigma, won_f, won_c))
        rows = np.array(rows, dtype=float)
        res[cname] = rows
    summary[label] = (res, family)

    print(f"\n=== {label}: {len(pages)} pages x {pages[0].shape[0]} tokens ===")
    r = res["sign(d)"]
    pm = family == "precise"
    print(f"  float margin m (gold - best other, / gold score): precise median {np.median(r[pm, 0]):+.4f}  "
          f"IQR [{np.percentile(r[pm, 0], 25):+.4f}, {np.percentile(r[pm, 0], 75):+.4f}]   topical median {np.median(r[~pm, 0]):+.4f}")
    print(f"  {'codec':<12} {'noise sigma':>11} {'R@1 float':>9} {'R@1 codec':>9} {'won->lost':>9} {'lost->won':>9}  flip rate by |m|/sigma bin: <0.5  0.5-1  1-2  >2")
    for cname, r in res.items():
        won_f, won_c = r[:, 2] > 0, r[:, 3] > 0
        ratio = np.abs(r[:, 0]) / np.maximum(r[:, 1], 1e-9)
        bins = [(0, 0.5), (0.5, 1), (1, 2), (2, np.inf)]
        flip = won_f != won_c
        cells = []
        for lo, hi in bins:
            sel = (ratio >= lo) & (ratio < hi)
            cells.append(f"{100*flip[sel].mean():4.0f}%({sel.sum():2d})" if sel.any() else "   -    ")
        print(f"  {cname:<12} {np.median(r[:, 1]):>11.4f} {won_f.mean():>9.3f} {won_c.mean():>9.3f} {int((won_f & ~won_c).sum()):>9} {int((~won_f & won_c).sum()):>9}  " + " ".join(cells))
    # margin vs flip: rank-biserial / AUC of |m| for predicting "not flipped" among float winners
    r = res["sign(d)"]
    wf = r[:, 2] > 0
    kept = r[wf & (r[:, 3] > 0), 0]
    lost = r[wf & (r[:, 3] == 0), 0]
    if kept.size and lost.size:
        auc = (kept[:, None] > lost[None, :]).mean()
        print(f"  sign(d): among {int(wf.sum())} float winners, AUC(margin separates kept from lost) = {auc:.2f}; "
              f"median margin kept {np.median(kept):+.4f} vs lost {np.median(lost):+.4f}")

# cross-run view: codec noise is a property of the codec + encoder, margin of the corpus/resolution
print("\n=== summary: median precise-query margin vs sign-codec noise (both as a fraction of the gold score) ===")
print(f"{'run':<18} {'margin(precise)':>16} {'margin(topical)':>16} {'sigma sign':>11} {'sigma 2-bit':>11} {'sigma int8':>11}   one-bit R@1 loss on precise")
for label, (res, family) in summary.items():
    pm = family == "precise"
    r = res["sign(d)"]
    loss = r[pm, 2].mean() - r[pm, 3].mean()
    print(f"{label:<18} {np.median(r[pm, 0]):>+16.4f} {np.median(r[~pm, 0]):>+16.4f} {np.median(r[:, 1]):>11.4f} "
          f"{np.median(res['2-bit'][:, 1]):>11.4f} {np.median(res['int8'][:, 1]):>11.4f}   {loss:+.3f}")
