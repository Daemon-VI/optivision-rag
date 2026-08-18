"""Generate the paper figures from reports/colsmol/benchmark.json.

Run from the repo root:  python paper/make_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "colsmol" / "benchmark.json"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

rows = {r["variant"]: r for r in json.loads(REPORT.read_text())["rows"]}


def fig_tradeoff() -> None:
    """Compression ratio vs nDCG@5 retention, coloured by quantizer."""
    groups = {
        "no quantization": (["spatial-only", "spatial+redundancy"], "#1b6ca8", "o"),
        "int8": (["int8-only", "prune+int8"], "#2e8b57", "s"),
        "binary": (
            ["binary-only", "optivision", "optivision-aggressive",
             "keep-50pct", "keep-40pct", "keep-30pct", "keep-20pct", "keep-10pct"],
            "#c0392b", "^",
        ),
    }
    # variant -> (text, x-offset pt, y-offset pt, horizontal alignment)
    labels = {
        "spatial-only": ("spatial", 5, 2, "left"),
        "spatial+redundancy": ("spatial+redund.", 5, -9, "left"),
        "int8-only": ("int8", 5, 2, "left"),
        "prune+int8": ("prune+int8", 5, 2, "left"),
        "binary-only": ("binary, no pruning", 0, 6, "center"),
        "optivision": ("OptiVision", 0, 6, "center"),
        "keep-10pct": ("keep-10%", 5, -2, "left"),
    }

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for name, (variants, colour, marker) in groups.items():
        xs = [rows[v]["compression_ratio"] for v in variants]
        ys = [100 * rows[v]["ndcg5_retention"] for v in variants]
        ax.scatter(xs, ys, s=22, c=colour, marker=marker, label=name,
                   edgecolors="white", linewidths=0.4, zorder=3)

    base = rows["baseline-float32"]
    ax.scatter([base["compression_ratio"]], [100.0], s=30, c="black", marker="*",
               label="float32 baseline", zorder=4)

    for v, (text, dx, dy, ha) in labels.items():
        ax.annotate(text, (rows[v]["compression_ratio"], 100 * rows[v]["ndcg5_retention"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=5.6, color="#333333")

    ax.axhspan(85, 88.5, color="#c0392b", alpha=0.07, zorder=0)
    ax.text(0.82, 87.4, "binary floor: every configuration\nwith 1-bit codes lands here",
            fontsize=5.4, color="#c0392b", va="center", ha="left", linespacing=1.3)

    ax.set_xscale("log")
    ax.set_xlabel("index compression ratio (log scale)")
    ax.set_ylabel("nDCG@5 retained (%)")
    ax.set_ylim(78, 103)
    ax.set_xlim(0.7, 460)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=5.8, loc="lower left", framealpha=0.95,
              borderpad=0.4, handletextpad=0.4)
    fig.savefig(OUT / "tradeoff.pdf")
    plt.close(fig)


def fig_sweep() -> None:
    """The keep-ratio sweep: token budget barely moves quality once binary is on."""
    sweep = ["keep-10pct", "keep-20pct", "keep-30pct", "keep-40pct", "keep-50pct"]
    xs = [rows[v]["tokens_per_page"] for v in sweep]
    ys = [100 * rows[v]["ndcg5_retention"] for v in sweep]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(xs, ys, marker="o", ms=3.5, color="#c0392b", label="prune + binary")

    pruned = rows["spatial+redundancy"]
    ax.axhline(100 * pruned["ndcg5_retention"], ls="--", lw=0.8, color="#1b6ca8",
               label="prune only (float32)")
    ax.axhline(100 * rows["binary-only"]["ndcg5_retention"], ls=":", lw=0.8, color="#7f8c8d",
               label="binary only, no pruning")

    ax.set_xlabel("vectors stored per page")
    ax.set_ylabel("nDCG@5 retained (%)")
    ax.set_ylim(82, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5.8, loc="center right")
    fig.savefig(OUT / "sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_tradeoff()
    fig_sweep()
    print(f"wrote {OUT}/tradeoff.pdf and {OUT}/sweep.pdf")
