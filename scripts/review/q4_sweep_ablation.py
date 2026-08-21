"""Finding 2 check: is E1's flat keep-N% sweep a property of the sparse corpus,
or of the 107 thumbnail/marker/instruction tokens that no variant prunes?
Re-run the real pipeline from the cache with those tokens removed. Also: does
pixel saliency survive a polarity flip (light text on dark background)?
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, "C:/Users/Rishi/optivision-rag/src")
from optivision.bench import EncodedCorpus, Variant, run_variant  # noqa: E402
from optivision.config import Config  # noqa: E402
from optivision.corpus import load_queries  # noqa: E402
from optivision.pruning.saliency import patch_saliency  # noqa: E402
from optivision.types import PageEncoding, PatchGrid  # noqa: E402

ROOT = Path("C:/Users/Rishi/optivision-rag")
os.chdir(ROOT)

cfg = Config()
cfg.pruning.min_keep = 8
corpus = EncodedCorpus.load(ROOT / "data/cache/colsmol.npz")
qids, texts, qrels = load_queries(ROOT / "data/corpus/queries.json")
zq = np.load(ROOT / "data/cache/colsmol.queries.npz", allow_pickle=True)
qv = [zq[f"q_{i}"] for i in range(len(texts))]


def strip(enc: PageEncoding, keep: str) -> PageEncoding:
    """keep = 'grid' (drop all 107), 'grid+thumb' (drop markers/instr), 'grid+instr' (drop thumbnail+markers)."""
    ti = enc.text_token_index
    runs = np.split(ti, np.flatnonzero(np.diff(ti) != 1) + 1)
    thumb = runs[-1][:64] if len(runs[-1]) >= 64 else np.zeros(0, int)
    instr = runs[-1][64:] if len(runs[-1]) >= 64 else runs[-1]
    markers = np.concatenate(runs[:-1]) if len(runs) > 1 else np.zeros(0, int)
    g = enc.grid.token_index.reshape(-1)
    extra = {"grid": [], "grid+thumb": [thumb], "grid+instr": [instr, markers], "all": [thumb, instr, markers]}[keep]
    keep_idx = np.concatenate([g] + [np.asarray(x, int) for x in extra]).astype(int)
    emb = enc.embeddings[keep_idx]
    # remap: grid occupies the first len(g) rows, extras follow
    grid = PatchGrid(rows=enc.grid.rows, cols=enc.grid.cols,
                     token_index=np.arange(len(g), dtype=np.int32).reshape(enc.grid.rows, enc.grid.cols))
    text = np.arange(len(g), len(keep_idx), dtype=np.int32)
    return PageEncoding(ref=enc.ref, embeddings=emb, grid=grid, image_size=enc.image_size,
                        text_token_index=text, meta=enc.meta)


def variants():
    out = [
        Variant("baseline-float32", pruning={"enabled": False}, compression={"enabled": False, "method": "none"}),
        Variant("binary-only", pruning={"enabled": False}, compression={"enabled": True, "method": "binary"}),
        Variant("spatial+redundancy", pruning={"enabled": True, "spatial": True, "redundancy": True},
                compression={"enabled": False, "method": "none"}),
    ]
    for r in (0.5, 0.3, 0.2, 0.1, 0.05):
        out.append(Variant(f"keep-{int(r*100)}pct-float", pruning={"enabled": True, "spatial": True, "redundancy": True, "keep_ratio": r},
                           compression={"enabled": False, "method": "none"}))
        out.append(Variant(f"keep-{int(r*100)}pct-binary", pruning={"enabled": True, "spatial": True, "redundancy": True, "keep_ratio": r},
                           compression={"enabled": True, "method": "binary"}))
    return out


print("E1 sweep through the real pipeline, with different subsets of the 107 unprunable tokens kept")
print("(nDCG@5 retention vs that configuration's own float baseline; tok/pg after both pruning stages)\n")
hdr = f"{'variant':<22}" + "".join(f"{k:>22}" for k in ("all (as published)", "grid+thumb", "grid+instr", "grid only"))
print(hdr)
results = {}
for keep in ("all", "grid+thumb", "grid+instr", "grid"):
    c2 = EncodedCorpus([strip(e, keep) for e in corpus.encodings], corpus.images)
    rows = {}
    for v in variants():
        r = run_variant(c2, cfg, v, qv, qids, qrels, top_k=10, workdir=ROOT / "data/tmp/q4")["row"]
        rows[v.name] = r
    results[keep] = rows
for v in variants():
    cells = ""
    for keep in ("all", "grid+thumb", "grid+instr", "grid"):
        r = results[keep][v.name]; b = results[keep]["baseline-float32"]
        cells += f"{r['ndcg@5']:8.4f} {100*r['ndcg@5']/b['ndcg@5']:5.1f}% {r['tokens_per_page']:6.0f}"
    print(f"{v.name:<22}{cells}")

# ------------------------------------------------------------ polarity check
print("\n=== pixel saliency under a polarity flip (same page, inverted) ===")
rows_, cols_ = corpus.encodings[0].grid.rows, corpus.encodings[0].grid.cols
img = corpus.images[0]
s_pos = patch_saliency(img, rows_, cols_).reshape(-1)
s_neg = patch_saliency(ImageOps.invert(img), rows_, cols_).reshape(-1)
ink_truth = s_pos > 0.02  # what the detector itself calls ink on the normal page
k = round(0.1 * rows_ * cols_)
top_pos = np.argsort(-s_pos)[:k]; top_neg = np.argsort(-s_neg)[:k]
print(f"  ink cells on the normal page (s>0.02): {ink_truth.sum()} of {rows_*cols_}")
print(f"  keep-10% on normal page:   {100*ink_truth[top_pos].mean():.0f}% of kept cells are ink cells")
print(f"  keep-10% on inverted page: {100*ink_truth[top_neg].mean():.0f}% of kept cells are ink cells")
print(f"  spearman-ish: corr(s_pos, s_neg) = {np.corrcoef(s_pos, s_neg)[0,1]:.2f}")
# ink-only and edge-only components
s_ink = patch_saliency(ImageOps.invert(img), rows_, cols_, ink_weight=1.0, edge_weight=0.0).reshape(-1)
s_edge = patch_saliency(ImageOps.invert(img), rows_, cols_, ink_weight=0.0, edge_weight=1.0).reshape(-1)
print(f"  inverted, ink term only:  keep-10% hits ink cells {100*ink_truth[np.argsort(-s_ink)[:k]].mean():.0f}%")
print(f"  inverted, edge term only: keep-10% hits ink cells {100*ink_truth[np.argsort(-s_edge)[:k]].mean():.0f}%")
