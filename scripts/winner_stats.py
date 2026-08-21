"""How much of a late-interaction index is never read?

A late-interaction score is a sum over query tokens of a max over patches. Only
the argmax patch of each query token contributes anything; every other patch on
the page is encoded, quantized, stored and shipped without ever being consulted.
This measures the size of that dead set, and -- the part that decides whether it
is exploitable -- whether membership transfers to queries the selector never saw.

The headline statistic (what fraction of patches ever win) needs no relevance
labels at all, so it runs on any cached corpus:

    python scripts/winner_stats.py --cache data/cache/colsmol.npz

Pass a corpus to also score retrieval with a winner-only index, fitted on half
the queries and evaluated on the other half:

    python scripts/winner_stats.py --cache data/cache/colsmol.npz \
        --corpus data/corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(cache: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    z = np.load(cache, allow_pickle=True)
    zq = np.load(cache.with_suffix("").with_suffix(".queries.npz"), allow_pickle=True)
    n = len([k for k in z.files if k.startswith("emb_")])
    pages = [np.asarray(z[f"emb_{i}"], np.float32) for i in range(n)]
    nq = len([k for k in zq.files if k.startswith("q_")])
    queries = [np.asarray(zq[f"q_{i}"], np.float32) for i in range(nq)]
    return pages, queries


def winners(pages, queries, qidx) -> list[np.ndarray]:
    """Boolean mask per page: is this patch the argmax for any query token?"""
    w = [np.zeros(len(p), bool) for p in pages]
    for qi in qidx:
        q = queries[qi]
        for pi, page in enumerate(pages):
            w[pi][(q @ page.T).argmax(1)] = True
    return w


def ndcg5(pages, queries, gold, qidx, mask=None) -> float:
    out = []
    for qi in qidx:
        q = queries[qi]
        scores = np.array([
            float((q @ (page[mask[pi]] if mask else page).T).max(1).sum())
            if (mask is None or mask[pi].any()) else -1e9
            for pi, page in enumerate(pages)
        ])
        top = np.argsort(-scores)[:5]
        rank = next((k for k, i in enumerate(top) if i in gold[qi]), None)
        out.append(0.0 if rank is None else 1.0 / np.log2(rank + 2))
    return float(np.mean(out))


def page_ids(corpus: Path) -> list[str]:
    """Reproduce ingest.iter_pages: sorted pdfs, pages numbered from one."""
    spec = json.loads((corpus / "manifest.json").read_text())["spec"]
    stems = sorted(p.stem for p in (corpus / "pdfs").glob("*.pdf"))
    return [f"{s}::p{k + 1}" for s in stems for k in range(spec["pages_per_doc"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/colsmol.npz"))
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    pages, queries = load(a.cache)
    total = sum(len(p) for p in pages)
    print(f"{len(pages)} pages, {len(queries)} queries, {total:,} patches "
          f"({total / len(pages):.0f} per page)")

    # --- the label-free statistic -----------------------------------------
    w_all = winners(pages, queries, range(len(queries)))
    kept = sum(int(x.sum()) for x in w_all)
    per = np.array([100 * x.mean() for x in w_all])
    print(f"\npatches that ever win a MaxSim, over all {len(queries)} queries: "
          f"{kept:,} / {total:,} = {100 * kept / total:.1f}%")
    print(f"  per page: median {np.median(per):.1f}%  "
          f"min {per.min():.1f}%  max {per.max():.1f}%")
    print(f"  headroom if the dead set were free to drop: {total / max(kept, 1):.1f}x")

    if a.corpus is None:
        print("\n(pass --corpus to measure whether the winner set transfers)")
        return 0

    # --- does the set transfer to unseen queries? -------------------------
    ids = page_ids(a.corpus)
    if len(ids) != len(pages):
        print(f"\ncorpus has {len(ids)} pages but the cache has {len(pages)}; "
              "skipping the retrieval half")
        return 1
    pos = {pid: i for i, pid in enumerate(ids)}
    rel = json.loads((a.corpus / "queries.json").read_text())
    gold = [{pos[r] for r in q["relevant"] if r in pos} for q in rel]

    perm = np.random.default_rng(a.seed).permutation(len(queries))
    fit, held = list(perm[: len(queries) // 2]), list(perm[len(queries) // 2 :])
    w_fit = winners(pages, queries, fit)
    kept_fit = sum(int(x.sum()) for x in w_fit)

    print(f"\nwinner set fitted on {len(fit)} queries: {kept_fit:,} patches "
          f"({100 * kept_fit / total:.1f}%, {total / kept_fit:.1f}x smaller index)")
    base = ndcg5(pages, queries, gold, held)
    orac = ndcg5(pages, queries, gold, held, w_fit)
    print(f"nDCG@5 on the {len(held)} held-out queries")
    print(f"  full index         {base:.4f}")
    print(f"  winners only       {orac:.4f}   ({100 * orac / max(base, 1e-9):.1f}% retained)")

    # Stacked with the one-bit codec -- the stage that carries the loss in the
    # paper. Sign vectors collide, so binarization promotes patches that were
    # never competitive; dropping the dead set removes those distractors.
    binary = [np.sign(p).astype(np.float32) for p in pages]
    print("\nwith the one-bit codec on top")
    print(f"  binary, full index   {ndcg5(binary, queries, gold, held):.4f}")
    print(f"  binary, winners only {ndcg5(binary, queries, gold, held, w_fit):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
