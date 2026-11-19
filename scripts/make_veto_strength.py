"""Build ``src/optivision/presets/veto_strength.json``: how often the practice check fires under
toy damage models (FINAL_DESIGN 5.5.1, the table behind the ``Strength`` line).

    python scripts/make_veto_strength.py            # rewrite the table
    python scripts/make_veto_strength.py --check    # exit 1 if it is out of date

What is simulated
-----------------
The practice check of ``optivision.easy._veto``: ``n`` hard practice questions; for each,
``b`` is the full-precision nDCG@5 and ``c`` the candidate layout's; ``R = sum(c) / sum(b)``;
``U`` is the one-sided 95% upper bound of ``R`` from ``B = 2,000`` bootstrap resamples of the
questions (percentile method, ``numpy.quantile``, linear); the check fires when ``U < T``.
There is one hard question per page, so the page-cluster bootstrap resamples single
questions here.

Damage models (both speculative; they size the check, they do not describe any encoder):

``loss``
    Every question has its relevant page at rank 1 at full precision (``b = 1``). The
    candidate layout loses each question independently with probability ``1 - rho`` (its
    relevant page leaves the top 5, ``c = 0``) and keeps it otherwise (``c = b``).
``mixed``
    The full-precision rank of the relevant page is 1, 2, 3, 4 or 5 with probabilities
    0.60, 0.15, 0.10, 0.08 and 0.07 (``b = 1 / log2(rank + 1)``); losses as in ``loss``.

``rho`` is the true retention. Each cell runs ``REPS`` independent folders: a folder's
questions are drawn from the model, then the check is run on them. ``fire_rate`` is the share
of folders where it fired: the false-alarm rate when ``rho = T`` and the catch rate when
``rho`` is below ``T``. Falls of 0.04 and 0.07 below ``T`` are simulated (at ``T = 0.97``:
0.93 and 0.90).

How it is computed
------------------
Each folder is resampled the way ``_veto.bootstrap_upper`` does it: ``B`` rows of ``n``
question indices drawn uniformly with replacement, and the sums of ``c`` and ``b`` over each
row. The sums are taken as a product with the matrix of index counts (``c[idx].sum(1)`` is
``counts @ c``), and one index matrix is shared by the folders of a block of ``BLOCK``, which
makes the run fast enough for a test. Sharing it leaves each folder's bound exactly as
``_veto`` computes it; it only correlates folders of a block through the resampling noise of
``U`` (standard error about 0.002 at ``B = 2,000``), which is small next to the spread of
``R`` between folders (about 0.03 at ``n = 60``). Every cell has its own seed from
``numpy.random.SeedSequence([SEED, model, T, rho, n])``, so one cell can be recomputed alone.

What it leaves out: partial rank shifts, losses that are correlated within a file, and the
fact that practice questions are easier than real ones. Whether real losses show up in
practice questions at all is test V5 (reports/easy/PREREG_EASY.md); until then the build
summary tags this table ``[estimate: simulation]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "optivision" / "presets" / "veto_strength.json"

SEED = 0
B = 2000
REPS = 1000
LEVEL = 0.95
TARGETS = (0.90, 0.95, 0.97, 0.98)
NS = (60, 100, 137, 200, 300)
FALLS = (0.0, 0.04, 0.07)
BLOCK = 100  # folders that share one matrix of resample indices

#: damage model -> (rank probabilities, full-precision gain per rank)
MODELS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "loss": ((1.0,), (1.0,)),
    "mixed": ((0.60, 0.15, 0.10, 0.08, 0.07), tuple(1.0 / np.log2(r + 1.0) for r in range(1, 6))),
}


def _categories(model: str, rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Category probabilities and the gains ``c`` and ``b`` of each category (kept, then lost,
    for each rank)."""
    ranks, gains = MODELS[model]
    probs, gain_c, gain_b = [], [], []
    for p, g in zip(ranks, gains):
        probs += [p * rho, p * (1.0 - rho)]
        gain_c += [g, 0.0]
        gain_b += [g, g]
    return np.array(probs), np.array(gain_c), np.array(gain_b)


def resample_counts(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    """``[b, n]`` float counts: how often each of ``n`` questions is drawn in each of ``b``
    resamples of ``n`` indices (``rng.integers(0, n, size=(b, n))``, as ``_veto`` draws them)."""
    idx = rng.integers(0, n, size=(b, n))
    flat = (idx + (np.arange(b, dtype=np.int64) * n)[:, None]).ravel()
    return np.bincount(flat, minlength=b * n).reshape(b, n).astype(np.float64)


def upper_bounds(c: np.ndarray, bvals: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The bound ``U`` of each folder (rows of ``c`` and ``bvals``) under shared resample counts."""
    num = counts @ c.T
    den = counts @ bvals.T
    ratio = np.divide(num, den, out=np.ones_like(num), where=den > 0)
    return np.quantile(ratio, LEVEL, axis=0)


def simulate_cell(model: str, target: float, rho: float, n: int, *, reps: int = REPS, b: int = B,
                  seed: int = SEED) -> float:
    """Share of ``reps`` simulated folders on which the check fires (``U < target``)."""
    m = list(MODELS).index(model)
    rng = np.random.default_rng(np.random.SeedSequence([seed, m, round(target * 1000), round(rho * 1000), n]))
    probs, gain_c, gain_b = _categories(model, rho)
    fired = 0
    for lo in range(0, reps, BLOCK):
        folders = min(BLOCK, reps - lo)
        cat = rng.choice(probs.size, size=(folders, n), p=probs)
        upper = upper_bounds(gain_c[cat], gain_b[cat], resample_counts(rng, n, b))
        fired += int(np.count_nonzero(upper < target))
    return fired / reps


def _digest(path: Path) -> str:
    """Hash of the file with LF line endings, so a CRLF checkout gives the same value."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build() -> dict:
    cells = []
    for model in MODELS:
        for target in TARGETS:
            for fall in FALLS:
                rho = round(target - fall, 4)
                for n in NS:
                    rate = simulate_cell(model, target, rho, n)
                    cells.append({"model": model, "target": target, "retention": rho, "n": n,
                                  "fire_rate": round(rate, 3)})
    summary = []
    for target in TARGETS:
        for n in NS:
            def rates(fall: float, target: float = target, n: int = n) -> list[float]:
                vals = [c["fire_rate"] for c in cells
                        if c["target"] == target and c["n"] == n and c["retention"] == round(target - fall, 4)]
                return [min(vals), max(vals)]

            summary.append({
                "target": target, "n": n, "false_alarm": rates(0.0),
                "falls": [{"retention": round(target - f, 4), "caught": rates(f)} for f in FALLS if f > 0],
            })
    return {
        "schema": 1,
        "note": ("Toy damage-model simulation of the practice check (FINAL_DESIGN 5.5.1). It sizes the check; "
                 "it does not show that real losses appear in practice questions (test V5). Shown to users "
                 "tagged [estimate: simulation]."),
        "generator": "scripts/make_veto_strength.py",
        "script_sha256": _digest(Path(__file__)),
        "settings": {"B": B, "reps": REPS, "level": LEVEL, "seed": SEED, "targets": list(TARGETS), "n": list(NS),
                     "falls": list(FALLS), "statistic": "U = one-sided 95% bootstrap upper bound of sum(c)/sum(b), "
                     "nDCG@5, percentile method; fires when U < T"},
        "damage_models": {
            "loss": "b = 1 for every question; each is lost (c = 0) with probability 1 - retention",
            "mixed": ("full-precision rank 1-5 with probabilities 0.60/0.15/0.10/0.08/0.07, b = 1/log2(rank+1); "
                      "each is lost (c = 0) with probability 1 - retention"),
        },
        "cells": cells,
        "summary": summary,
    }


def render(table: dict) -> str:
    return json.dumps(table, indent=1, sort_keys=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed file instead of writing")
    a = ap.parse_args()
    text = render(build())
    if a.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current.replace("\r\n", "\n") != text:
            print(f"{OUT.relative_to(ROOT)} is out of date; run scripts/make_veto_strength.py")
            return 1
        print("strength table is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(json.loads(text)['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
