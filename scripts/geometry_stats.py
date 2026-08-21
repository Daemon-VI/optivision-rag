"""How much of a patch vector survives sign-thresholding?

Section VI-A of the paper now claims the binary floor is a property of the
encoder's patch geometry rather than of the codec: ColPali loses far less to
one-bit codes than ColSmol does on identical pages, so its patch vectors must
carry more of their norm far enough from zero for the sign to preserve it. That
claim is currently a prediction. This measures it.

The quantity is the cosine between a vector and its own sign pattern,

    rho(d) = <d, sign(d)> / (||d|| sqrt(D)) = ||d||_1 / (||d||_2 sqrt(D))

which is exactly the distortion term the paper already quotes: for a random unit
vector rho is sqrt(2/pi) ~ 0.798. Higher means more of the vector survives.

Mean rho says how much is lost on average. The paper's own argument in VI-A is
that *uniform* distortion is safe and non-uniform distortion reorders, so the
spread of rho matters at least as much as its mean -- and the argmax flip rate
measures the reordering directly.

    python scripts/geometry_stats.py --cache data/cache/colsmol.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(cache: Path):
    if not cache.exists():
        raise SystemExit(f"no encode cache at {cache}")
    z = np.load(cache, allow_pickle=True)
    qpath = cache.with_suffix("").with_suffix(".queries.npz")
    zq = np.load(qpath, allow_pickle=True) if qpath.exists() else None
    n = len([k for k in z.files if k.startswith("emb_")])
    pages = [np.asarray(z[f"emb_{i}"], np.float32) for i in range(n)]
    queries = []
    if zq is not None:
        queries = [np.asarray(zq[f"q_{i}"], np.float32)
                   for i in range(len([k for k in zq.files if k.startswith("q_")]))]
    return pages, queries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/colsmol.npz"))
    ap.add_argument("--label", default=None, help="name for the printed row")
    ap.add_argument("--max-patches", type=int, default=60_000,
                    help="patches sampled for the mass-concentration statistic")
    ap.add_argument("--max-queries", type=int, default=64,
                    help="queries used for the argmax flip rate")
    a = ap.parse_args()
    rng = np.random.default_rng(7)

    pages, queries = load(a.cache)
    label = a.label or a.cache.stem
    allv = np.concatenate(pages, axis=0)
    dim = allv.shape[1]
    print(f"{label}: {len(pages)} pages, {allv.shape[0]:,} patches, dim {dim}\n")

    # rho = cos(d, sign(d)). The random-unit-vector reference is sqrt(2/pi).
    l1 = np.abs(allv).sum(axis=1)
    l2 = np.linalg.norm(allv, axis=1)
    rho = l1 / (np.maximum(l2, 1e-12) * np.sqrt(dim))
    print(f"  rho = cos(d, sign(d))   mean {rho.mean():.4f}   std {rho.std():.4f}")
    print(f"                          p05  {np.percentile(rho, 5):.4f}   "
          f"p95 {np.percentile(rho, 95):.4f}")
    print(f"  random-unit reference   {np.sqrt(2 / np.pi):.4f}")
    print(f"  above reference         {100 * (rho > np.sqrt(2 / np.pi)).mean():.1f}% of patches")

    # Concentration: how many dimensions carry half the L1 mass. A vector whose
    # mass sits in a few dimensions loses more of it to a sign. Sorting every
    # patch of a 500-page split means three copies of a 250 MB array for a
    # median, so sample.
    if allv.shape[0] > a.max_patches:
        pick = rng.choice(allv.shape[0], a.max_patches, replace=False)
        print(f"  (mass concentration sampled over {a.max_patches:,} of "
              f"{allv.shape[0]:,} patches)")
    else:
        pick = slice(None)
    sample = np.abs(allv[pick])
    srt = np.sort(sample, axis=1)[:, ::-1]
    cum = np.cumsum(srt, axis=1) / np.maximum(srt.sum(axis=1)[:, None], 1e-12)
    half = (cum < 0.5).sum(axis=1) + 1
    print(f"  dims holding half the L1 mass   median {np.median(half):.0f} of {dim}")

    if not queries:
        print("\n(no query cache beside this one; skipping the argmax flip rate)")
        return 0

    # The reordering the paper's arg max argument is about: how often does
    # binarising the page change which patch wins a query token? This is one
    # small matmul per (query, page) pair, so 451 queries over 500 pages is a
    # quarter of a million of them; a subset of queries measures the same rate.
    used = queries
    if len(queries) > a.max_queries:
        idx = rng.choice(len(queries), a.max_queries, replace=False)
        used = [queries[i] for i in idx]
        print(f"\n  (flip rate over {a.max_queries} of {len(queries)} queries)")
    signs = [np.sign(page) for page in pages]
    flips = total = 0
    for q in used:
        for page, sgn in zip(pages, signs, strict=True):
            a_f = (q @ page.T).argmax(1)
            a_b = (q @ sgn.T).argmax(1)
            flips += int(np.count_nonzero(a_f != a_b))
            total += a_f.size
    print(f"\n  argmax flips under one-bit codes   {100 * flips / total:.1f}% "
          f"of {total:,} (query token, page) pairs")
    print("  ^ this is the quantity Section VI-A's arg max argument is about")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
