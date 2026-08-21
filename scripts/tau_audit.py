"""What does the reported Kendall tau actually measure?

The paper builds an argument on tau: that one-bit codes reorder results just as
severely under both encoders, and only the stronger encoder absorbs it. That
argument needs tau to mean what `metrics.rank_correlation` says it means --
"whether the compressed index orders pages the way the float baseline did".

It does not, quite. `bench` passes `run[q]`, which is the top-`k` hit list with
`k = 10`, so tau-b is taken over the *intersection of two top-10 lists*, not
over the corpus. Two consequences follow, and this script measures both:

  * the value depends on the cutoff, so quoting a tau without naming k is
    under-specified;
  * `rank_correlation` returns 1.0 when fewer than two ids are common, so the
    worst case -- the two rankings sharing nothing -- scores as perfect
    agreement.

Runs on a cached corpus, no GPU:

    python scripts/tau_audit.py --cache data/cache/colsmol.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from optivision.compression.binary import pack_bits, unpack_signs
from optivision.metrics import rank_correlation

BENCH_TOP_K = 10  # bench.run_variant's default, and what the paper's tau uses


def maxsim(pages: list[np.ndarray], q: np.ndarray) -> np.ndarray:
    return np.array([float((q @ p.T).max(1).sum()) for p in pages])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache/colsmol.npz"))
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20])
    a = ap.parse_args()

    z = np.load(a.cache, allow_pickle=True)
    zq = np.load(a.cache.with_suffix("").with_suffix(".queries.npz"), allow_pickle=True)
    n = len([k for k in z.files if k.startswith("emb_")])
    pages = [np.asarray(z[f"emb_{i}"], np.float32) for i in range(n)]
    queries = [np.asarray(zq[f"q_{i}"], np.float32) for i in
               range(len([k for k in zq.files if k.startswith("q_")]))]
    binary = [unpack_signs(pack_bits(p), p.shape[1]).astype(np.float32) for p in pages]
    print(f"{n} pages, {len(queries)} queries, one-bit codes vs float32 baseline\n")

    float_scores = [maxsim(pages, q) for q in queries]
    bin_scores = [maxsim(binary, q) for q in queries]

    for k in sorted({*a.cutoffs, n}):
        taus, common, fallback = [], [], 0
        for sf, sb in zip(float_scores, bin_scores, strict=True):
            top_f = [str(i) for i in np.argsort(-sf)[:k]]
            top_b = [str(i) for i in np.argsort(-sb)[:k]]
            taus.append(rank_correlation(top_f, top_b))
            shared = len(set(top_f) & set(top_b))
            common.append(shared)
            fallback += shared < 2
        label = f"top-{k}" + (" (whole corpus)" if k >= n else "")
        mark = "  <- what the paper reports" if k == BENCH_TOP_K else ""
        print(f"{label:<24} tau {np.mean(taus):.3f}   "
              f"ids compared {np.mean(common):4.1f}/{k}   "
              f"tau=1.0 fallbacks {fallback:>3}{mark}")

    # The quantity the prose actually describes: agreement over the full ranking.
    full = [kendalltau(np.argsort(np.argsort(-sf)),
                       np.argsort(np.argsort(-sb))).statistic
            for sf, sb in zip(float_scores, bin_scores, strict=True)]
    print(f"\ntrue rank correlation over all {n} pages: {np.mean(full):.3f}")
    print("\nThe cutoff moves the number, so a tau quoted without its k is "
          "under-specified,\nand top-k covers a different fraction of a "
          "60-page corpus than of a 500-page one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
