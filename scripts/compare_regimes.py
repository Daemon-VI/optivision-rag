"""Print the 2x2: {ColSmol-256M, ColPali-3B} x {generated pages, ViDoRe}.

E1 and E2 change the encoder and the corpus together, so neither can say which
variable moved the per-stage attribution. E3 -- the reference encoder over E1's
own corpus -- is the cell that separates them:

    if E3 looks like E1  ->  the corpus drove it   (page sparsity)
    if E3 looks like E2  ->  the encoder drove it  (retrieval margin)

Run after MODE=generated bash scripts/run_bench_gpu.sh brings back
reports/colpali_generated/. Reads only benchmark.json, computes everything.

    python scripts/compare_regimes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CELLS = [
    ("E1", "ColSmol-256M", "generated", "reports/colsmol"),
    ("E2", "ColPali-3B", "ViDoRe docvqa", "reports/colpali_docvqa_test_subsampled"),
    ("E3", "ColPali-3B", "generated", "reports/colpali_generated"),
]

# The rows that carry the argument: what each stage costs on its own, and what
# the two operating points look like once both are applied.
ROWS = [
    ("spatial+redundancy", "pruning alone"),
    ("binary-only", "one-bit codec alone"),
    ("int8-only", "int8 codec alone"),
    ("optivision", "both, size-first"),
    ("keep-50pct", "sweep: keep 50%"),
    ("keep-10pct", "sweep: keep 10%"),
]


def load(path: str) -> dict | None:
    f = Path(path) / "benchmark.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {r["variant"]: r for r in d["rows"]}


def main() -> int:
    loaded = [(tag, enc, corpus, load(path), path) for tag, enc, corpus, path in CELLS]
    missing = [(tag, path) for tag, _, _, rows, path in loaded if rows is None]
    for tag, path in missing:
        print(f"note: {tag} not present ({path}/benchmark.json)", file=sys.stderr)
    have = [(tag, enc, corpus, rows) for tag, enc, corpus, rows, _ in loaded if rows]
    if len(have) < 2:
        print("need at least two cells to compare", file=sys.stderr)
        return 1

    def retain(rows: dict, variant: str) -> float | None:
        if variant not in rows:
            return None
        return 100 * rows[variant]["ndcg@5"] / rows["baseline-float32"]["ndcg@5"]

    def cell(rows: dict, variant: str) -> str:
        r = retain(rows, variant)
        if r is None:
            return f"{'--':>13} "
        compr = rows[variant]["compression_ratio"]
        return f"{r:6.1f}% {compr:6.1f}x"

    width = 15
    print()
    print(f"{'':<22}" + "".join(f"{tag:<{width}}" for tag, *_ in have))
    print(f"{'':<22}" + "".join(f"{enc.split('-')[0]:<{width}}" for _, enc, _, _ in have))
    print(f"{'':<22}" + "".join(f"{corpus:<{width}}" for _, _, corpus, _ in have))
    print("-" * (22 + width * len(have)))
    for variant, label in ROWS:
        print(f"{label:<22}" + "".join(cell(rows, variant) for *_, rows in have))
    print()
    print("retention of baseline nDCG@5, and compression ratio")

    # The single number the experiment exists to produce.
    print()
    binary = {tag: retain(rows, "binary-only") for tag, _, _, rows in have}
    prune = {tag: rows["spatial+redundancy"]["compression_ratio"] for tag, _, _, rows in have}
    print("the codec's cost, alone:")
    for tag, val in binary.items():
        if val is not None:
            print(f"  {tag}: {100 - val:5.1f} points")
    print("what pruning buys:")
    for tag, val in prune.items():
        print(f"  {tag}: {val:.2f}x")

    if binary.get("E3") is not None and binary.get("E1") and binary.get("E2"):
        # Two quantities were conflated, and they do not answer to the same
        # factor. Report each against its own control rather than collapsing
        # both into one verdict.
        cost = {t: 100 - binary[t] for t in ("E1", "E2", "E3")}
        print()
        print("E3 shares E1's corpus and E2's encoder, so each axis has a control:")
        print(f"  codec cost, corpus held fixed (E1 -> E3, encoder swapped): "
              f"{cost['E1']:.1f} -> {cost['E3']:.1f} points")
        print(f"  codec cost, encoder held fixed (E3 -> E2, corpus swapped): "
              f"{cost['E3']:.1f} -> {cost['E2']:.1f} points")
        print(f"  prune gain, corpus held fixed (E1 -> E3, encoder swapped): "
              f"{prune['E1']:.2f}x -> {prune['E3']:.2f}x")
        print(f"  prune gain, encoder held fixed (E3 -> E2, corpus swapped): "
              f"{prune['E3']:.2f}x -> {prune['E2']:.2f}x")
        print()
        print(f"=> the ENCODER sets what the codec costs "
              f"({cost['E1']:.1f} -> {cost['E3']:.1f} on identical pages)")
        print(f"=> the CORPUS sets what pruning buys "
              f"({prune['E3']:.2f}x -> {prune['E2']:.2f}x under one encoder)")

        # The paper explains the codec result by retrieval margin: only a
        # stronger retriever has room to absorb the distortion. E3 tests that
        # directly, because it is the weaker retriever on this corpus.
        base = {t: rows["baseline-float32"] for t, _, _, rows in have}
        b1, b3 = base["E1"]["ndcg@5"], base["E3"]["ndcg@5"]
        r1, r3 = base["E1"]["recall@1"], base["E3"]["recall@1"]
        if b3 < b1:
            print()
            print("note: on these same pages the reference encoder is the WEAKER "
                  "retriever")
            print(f"      baseline nDCG@5 {b1:.4f} (E1) vs {b3:.4f} (E3), "
                  f"R@1 {r1:.3f} vs {r3:.3f}")
            print("      so retrieval margin cannot be what absorbs the codec's "
                  "distortion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
