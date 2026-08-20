"""Generate the paper figures from the benchmark reports.

Run from the repo root:  python paper/make_figs.py

Two experiments are plotted. E1 is the 256M encoder on the generated corpus
(reports/colsmol); E2 is ColPali-v1.3 on ViDoRe (reports/colpali_docvqa_*), the
split with the most queries. Figures for E2 carry a `_colpali` suffix, and
scale.pdf overlays the two -- which is the point of the paper, since the
per-stage attribution does not survive the change of encoder and corpus.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

REPORTS = {
    "": ROOT / "reports" / "colsmol" / "benchmark.json",
    "_colpali": ROOT / "reports" / "colpali_docvqa_test_subsampled" / "benchmark.json",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

BINARY_VARIANTS = [
    "binary-only", "optivision", "optivision-aggressive",
    "keep-50pct", "keep-40pct", "keep-30pct", "keep-20pct", "keep-10pct",
]


def load(path: Path) -> dict:
    return {r["variant"]: r for r in json.loads(path.read_text())["rows"]}


def retention(row: dict) -> float:
    return 100 * row["ndcg5_retention"]


def fig_tradeoff(rows: dict, suffix: str, floor: tuple[float, float] | None) -> None:
    """Compression ratio vs nDCG@5 retention, coloured by quantizer."""
    groups = {
        "no quantization": (["spatial-only", "spatial+redundancy"], "#1b6ca8", "o"),
        "int8": (["int8-only", "prune+int8"], "#2e8b57", "s"),
        "binary": (BINARY_VARIANTS, "#c0392b", "^"),
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
        ys = [retention(rows[v]) for v in variants]
        ax.scatter(xs, ys, s=22, c=colour, marker=marker, label=name,
                   edgecolors="white", linewidths=0.4, zorder=3)

    base = rows["baseline-float32"]
    ax.scatter([base["compression_ratio"]], [100.0], s=30, c="black", marker="*",
               label="float32 baseline", zorder=4)

    for v, (text, dx, dy, ha) in labels.items():
        ax.annotate(text, (rows[v]["compression_ratio"], retention(rows[v])),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=5.6, color="#333333")

    if floor is not None:
        ax.axhspan(floor[0], floor[1], color="#c0392b", alpha=0.07, zorder=0)
        ax.text(0.82, floor[1] - 1.1,
                "binary floor: every configuration\nwith 1-bit codes lands here",
                fontsize=5.4, color="#c0392b", va="center", ha="left", linespacing=1.3)

    lo = min(retention(rows[v]) for v in rows)
    ax.set_xscale("log")
    ax.set_xlabel("index compression ratio (log scale)")
    ax.set_ylabel("nDCG@5 retained (%)")
    ax.set_ylim(min(78, lo - 4), 103)
    ax.set_xlim(0.7, 460)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=5.8, loc="lower left", framealpha=0.95,
              borderpad=0.4, handletextpad=0.4)
    fig.savefig(OUT / f"tradeoff{suffix}.pdf")
    plt.close(fig)


def fig_sweep(rows: dict, suffix: str) -> None:
    """The keep-ratio sweep against the two references it sits between."""
    sweep = ["keep-10pct", "keep-20pct", "keep-30pct", "keep-40pct", "keep-50pct"]
    xs = [rows[v]["tokens_per_page"] for v in sweep]
    ys = [retention(rows[v]) for v in sweep]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(xs, ys, marker="o", ms=3.5, color="#c0392b", label="prune + binary")

    ax.axhline(retention(rows["spatial+redundancy"]), ls="--", lw=0.8, color="#1b6ca8",
               label="prune only (float32)")
    ax.axhline(retention(rows["binary-only"]), ls=":", lw=0.8, color="#7f8c8d",
               label="binary only, no pruning")

    ax.set_xlabel("vectors stored per page")
    ax.set_ylabel("nDCG@5 retained (%)")
    ax.set_ylim(min(82, min(ys) - 4), 103)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5.8, loc="center right")
    fig.savefig(OUT / f"sweep{suffix}.pdf")
    plt.close(fig)


def fig_scale(e1: dict, e2: dict) -> None:
    """The reversal: the same variants under a 256M and a 3B encoder.

    One axes, two experiments. The binary configurations are the ones that move:
    they sit in a band below 88% under the small encoder and above 94% under the
    reference one, at the same compression ratios.
    """
    series = [
        ("E1: ColSmol-256M, generated pages", e1, "#c0392b", "^", 0.35),
        ("E2: ColPali-3B, ViDoRe docvqa", e2, "#1b6ca8", "o", 1.0),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for label, rows, colour, marker, alpha in series:
        xs = [rows[v]["compression_ratio"] for v in BINARY_VARIANTS]
        ys = [retention(rows[v]) for v in BINARY_VARIANTS]
        ax.scatter(xs, ys, s=24, c=colour, marker=marker, label=label, alpha=alpha,
                   edgecolors="white", linewidths=0.4, zorder=3)

    # Join the same variant across experiments, so the shift is legible as motion
    # rather than as two unrelated clouds.
    for v in ("binary-only", "optivision", "keep-10pct"):
        ax.annotate(
            "", xy=(e2[v]["compression_ratio"], retention(e2[v])),
            xytext=(e1[v]["compression_ratio"], retention(e1[v])),
            arrowprops=dict(arrowstyle="->", lw=0.5, color="#7f8c8d",
                            shrinkA=3, shrinkB=3, alpha=0.8), zorder=2,
        )
    for v, text, off, ha in (("binary-only", "binary, no pruning", (7, 1), "left"),
                             ("optivision", "OptiVision", (0, -11), "center"),
                             ("keep-10pct", "keep-10%", (-6, 6), "right")):
        ax.annotate(text, (e2[v]["compression_ratio"], retention(e2[v])),
                    textcoords="offset points", xytext=off, ha=ha,
                    fontsize=5.6, color="#333333")

    ax.set_xscale("log")
    # The data spans barely more than a decade, so matplotlib's log formatter
    # labels the minor ticks as well and they overlap into a smear. Pin the
    # ticks we actually want and silence the rest.
    ax.xaxis.set_major_locator(FixedLocator([30, 50, 100, 200, 400]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}$\\times$"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(24, 520)
    ax.set_xlabel("index compression ratio (log scale)")
    ax.set_ylabel("nDCG@5 retained (%)")
    ax.set_ylim(60, 104)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=5.8, loc="lower left", framealpha=0.95,
              borderpad=0.4, handletextpad=0.4)
    fig.savefig(OUT / "scale.pdf")
    plt.close(fig)


if __name__ == "__main__":
    loaded = {}
    for suffix, path in REPORTS.items():
        if not path.exists():
            print(f"skipping{suffix or ' colsmol'}: {path} not found")
            continue
        loaded[suffix] = load(path)
        # The shaded "binary floor" band is an E1 claim; drawing it on E2 would
        # assert something the E2 numbers contradict.
        fig_tradeoff(loaded[suffix], suffix, (85, 88.5) if suffix == "" else None)
        fig_sweep(loaded[suffix], suffix)
        print(f"wrote {OUT}/tradeoff{suffix}.pdf and {OUT}/sweep{suffix}.pdf")

    if "" in loaded and "_colpali" in loaded:
        fig_scale(loaded[""], loaded["_colpali"])
        print(f"wrote {OUT}/scale.pdf")
