"""Score patch selectors against the oracle, on cached embeddings, in seconds.

Stage-II's premise held and its first proxy did not. The oracle -- keep only the
patches that actually win a MaxSim -- retains 100% of nDCG@5 on held-out queries
at 12.4x smaller (``scripts/winner_stats.py``). The codebook proxy for it lost to
pixel saliency at every budget, and random probes beat k-means ones, which says
the gap is in *how the probes are chosen*, not in the premise.

Iterating on that with the full 13-variant benchmark is the wrong loop: it costs
minutes and answers a question about the whole pipeline when the open question is
about one step. This scores selection alone, directly against the oracle:

    oracle recall   how much of the winner set a selector finds at a fixed budget
    nDCG@5          held-out retrieval with only the selected patches

The oracle row is the ceiling. Pixel saliency is the incumbent to beat. A proxy
worth putting on a GPU has to beat the incumbent here first.

    python scripts/probe_eval.py --cache data/cache/colsmol.npz --corpus data/corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from optivision.bench import EncodedCorpus
from optivision.pruning import codebook_saliency, fit_codebook
from optivision.pruning.saliency import patch_saliency

PROBE_SOURCES = ("kmeans", "random", "farthest")


def page_patches(enc) -> np.ndarray:
    """Patch vectors in grid order, [rows*cols, dim]."""
    return enc.embeddings[enc.grid.token_index.reshape(-1)]


def oracle_winners(pages: list[np.ndarray], queries: list[np.ndarray],
                   qidx) -> list[np.ndarray]:
    """Patches that are the arg max for some query token of some query."""
    w = [np.zeros(p.shape[0], bool) for p in pages]
    for qi in qidx:
        q = queries[qi]
        for pi, page in enumerate(pages):
            w[pi][(q @ page.T).argmax(1)] = True
    return w


def top_k_mask(score: np.ndarray, budget: int) -> np.ndarray:
    """Keep the highest-scoring ``budget`` patches. Ties broken by index."""
    mask = np.zeros(score.shape[0], bool)
    if budget >= score.shape[0]:
        return ~mask
    mask[np.argpartition(-score, budget - 1)[:budget]] = True
    return mask


def ndcg5(pages, queries, gold, qidx, masks=None) -> float:
    out = []
    for qi in qidx:
        q = queries[qi]
        scores = np.empty(len(pages), dtype=np.float32)
        for pi, page in enumerate(pages):
            sub = page[masks[pi]] if masks is not None else page
            scores[pi] = -1e9 if sub.shape[0] == 0 else float((q @ sub.T).max(1).sum())
        top = np.argsort(-scores)[:5]
        rank = next((k for k, i in enumerate(top) if i in gold[qi]), None)
        out.append(0.0 if rank is None else 1.0 / np.log2(rank + 2))
    return float(np.mean(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/colsmol.npz"))
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.30, 0.10])
    ap.add_argument("--probes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    if not a.cache.exists():
        raise SystemExit(f"no encode cache at {a.cache}")
    corpus = EncodedCorpus.load(a.cache)
    qcache = a.cache.with_suffix("").with_suffix(".queries.npz")
    if not qcache.exists():
        raise SystemExit(f"no query cache at {qcache}")
    zq = np.load(qcache, allow_pickle=True)
    queries = [np.asarray(zq[f"q_{i}"], np.float32)
               for i in range(len([k for k in zq.files if k.startswith("q_")]))]

    pages = [page_patches(e) for e in corpus.encodings]
    ids = [e.ref.page_id for e in corpus.encodings]
    pos = {pid: i for i, pid in enumerate(ids)}
    rel = json.loads((a.corpus / "queries.json").read_text())
    gold = [{pos[r] for r in q["relevant"] if r in pos} for q in rel]
    n_patch = pages[0].shape[0]

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(queries))
    fit, held = list(perm[: len(queries) // 2]), list(perm[len(queries) // 2:])
    print(f"{len(pages)} pages x {n_patch} patches, {len(queries)} queries "
          f"({len(fit)} fit / {len(held)} held out)\n")

    oracle = oracle_winners(pages, queries, fit)
    o_size = sum(int(w.sum()) for w in oracle)
    total = sum(p.shape[0] for p in pages)
    full = ndcg5(pages, queries, gold, held)
    print(f"full index        nDCG@5 {full:.4f}")
    print(f"oracle winners    nDCG@5 {ndcg5(pages, queries, gold, held, oracle):.4f}   "
          f"{100 * o_size / total:.1f}% of patches  <- the ceiling\n")

    # --- the selectors, all at the same budget ----------------------------
    sample = np.concatenate(pages, axis=0)
    take = min(20_000, sample.shape[0])
    sample = sample[rng.choice(sample.shape[0], take, replace=False)]
    books = {s: fit_codebook(sample, size=a.probes, seed=a.seed, source=s)
             for s in PROBE_SOURCES}

    grids = [(e.grid.rows, e.grid.cols) for e in corpus.encodings]
    scores: dict[str, list[np.ndarray]] = {
        "pixel": [
            patch_saliency(img, r, c).reshape(-1)
            for img, (r, c) in zip(corpus.images, grids, strict=True)
        ]
    }
    for name, book in books.items():
        scores[f"probe:{name}"] = [
            codebook_saliency(p, book, r, c).reshape(-1)
            for p, (r, c) in zip(pages, grids, strict=True)
        ]

    hdr = f"{'selector':<16}" + "".join(f"{f'keep {int(b * 100)}%':>22}" for b in a.budgets)
    print(hdr)
    print(f"{'':<16}" + "".join(f"{'nDCG@5  oracle recall':>22}" for _ in a.budgets))
    print("-" * len(hdr))
    for name, per_page in scores.items():
        cells = ""
        for b in a.budgets:
            budget = max(1, round(b * n_patch))
            masks = [top_k_mask(s, budget) for s in per_page]
            found = sum(int((m & o).sum()) for m, o in zip(masks, oracle, strict=True))
            recall = 100 * found / max(o_size, 1)
            cells += f"{ndcg5(pages, queries, gold, held, masks):>14.4f}{recall:>7.1f}%"
        print(f"{name:<16}{cells}")

    print("\nnDCG@5 is on held-out queries; oracle recall is of the winner set "
          "fitted on the\nother half. A proxy has to beat `pixel` here before it "
          "is worth a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
