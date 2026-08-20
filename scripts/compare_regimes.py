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
    for tag in binary:
        if binary[tag] is not None:
            print(f"  {tag}: {100 - binary[tag]:5.1f} points")
    print("what pruning buys:")
    for tag in prune:
        print(f"  {tag}: {prune[tag]:.2f}x")

    if "E3" in binary and binary["E3"] is not None:
        e1, e2, e3 = (100 - binary[t] if binary.get(t) is not None else None
                      for t in ("E1", "E2", "E3"))
        if e1 is not None and e2 is not None:
            verdict = ("the ENCODER drove the reversal (E3 tracks E2)"
                       if abs(e3 - e2) < abs(e3 - e1)
                       else "the CORPUS drove the reversal (E3 tracks E1)")
            print()
            print(f"=> {verdict}")
            print(f"   codec cost: E1 {e1:.1f}, E2 {e2:.1f}, E3 {e3:.1f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
