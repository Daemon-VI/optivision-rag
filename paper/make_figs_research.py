"""Figures for paper/optivision_research.tex, generated only from frozen reports on
branch research-investigation. Run from the repo root:

    python paper/make_figs_research.py

  research_targets.pdf   median compression chosen out of sample vs retention target (R7b, R11, R12)
  research_axes.pdf      retention vs byte factor per compression axis (Q4 analysis.json)
  research_q3.pdf        DocVQA/InfoVQA certification-SE ratio: raw, margin-matched, positive-margin half (Q3)
  research_scan.pdf      scan time vs corpus size on tiled ColQwen2 DocVQA (Q8 validation.json; timing only)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200})
W = 3.45  # IEEE column width (in)


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def fig_targets():
    sel = ROOT / "reports" / "universal" / "selection_rules"
    cells = [("ColPali DocVQA", "E2-colpali-docvqa"), ("ColPali InfoVQA", "E2-colpali-infovqa"),
             ("ColQwen2 DocVQA", "colqwen2-v1.0-merged_docvqa_test_subsampled"),
             ("ColQwen2 InfoVQA", "colqwen2-v1.0-merged_infovqa_test_subsampled"),
             ("ColQwen2.5 DocVQA", "colqwen2.5-v0.2_docvqa_test_subsampled"),
             ("ColQwen2.5 InfoVQA", "colqwen2.5-v0.2_infovqa_test_subsampled"),
             ("ColEmbed 4B DocVQA", "nemotron-colembed-vl-4b-v2_docvqa_test_subsampled"),
             ("ColEmbed 4B InfoVQA", "nemotron-colembed-vl-4b-v2_infovqa_test_subsampled"),
             ("ColBERT-small SciFact", "text-answerai-colbert-small-v1-scifact")]
    fig, ax = plt.subplots(figsize=(W, 2.7))
    for name, f in cells:
        d = json.loads((sel / f"{f}.json").read_text(encoding="utf-8"))
        rows = sorted((r for r in d["results"] if r["reference"] == "labels" and r["rule"] == "calibrate() default"),
                      key=lambda r: -r["target"])
        ls = "--" if "DocVQA" in name else (":" if "SciFact" in name else "-")
        ax.plot([r["target"] for r in rows], [r["compression_median"] for r in rows], ls, marker="o", ms=3, lw=1,
                label=name)
    ax.set_yscale("log")
    ax.set_ylim(1.5, 6000)
    ax.set_xlabel("retention target T")
    ax.set_ylabel("median compression chosen (x)")
    ax.set_xticks([0.95, 0.97, 0.99])
    ax.invert_xaxis()
    ax.legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "research_targets.pdf")


def fig_axes():
    nc = load("reports/research/Q4/analysis.json")["normalized_curves"]
    panels = [("ColQwen2 DocVQA (128-d)", "ColQwen2_docvqa"), ("ColPali DocVQA (128-d)", "ColPali_docvqa"),
              ("ColEmbed 4B DocVQA (2,560-d)", "ColEmbed4B_docvqa")]
    fig, axs = plt.subplots(1, 3, figsize=(W * 2.05, 2.0), sharey=True)
    style = {"A": ("merge (Ward)", "o"), "B": ("PCA", "s"), "C": ("codec", "^")}
    for ax, (title, key) in zip(axs, panels, strict=True):
        for axis, (lab, mk) in style.items():
            pts = [(x["factor"], x["R"]) for x in nc[key][axis]]
            if axis != "C":
                pts.sort()
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], marker=mk, ms=3,
                    lw=1 if axis != "C" else 0, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 8, 16, 32])
        ax.set_xticklabels(["2", "4", "8", "16", "32"])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("byte factor vs float32 (x)")
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_ylim(0, 104)
    axs[0].set_ylabel("nDCG@5 retention (%)")
    axs[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "research_axes.pdf")


def fig_q3():
    models = ["ColPali", "ColQwen2", "ColQwen2.5"]
    raw, matched, half = [], [], []
    for m in models:
        iv = load(f"reports/research/Q3/intervention/{m}_margin.json")
        sp = load(f"reports/research/Q3/split/{m}_high.json")
        raw.append(math.exp(iv["median_log_se_ratio_unmatched"]))
        matched.append(math.exp(iv["median_log_se_ratio_matched"]))
        half.append(math.exp(sp["median_log_se_ratio_half_over_info"]))
    fig, ax = plt.subplots(figsize=(W, 1.9))
    x = range(len(models))
    wdt = 0.26
    ax.bar([i - wdt for i in x], raw, wdt, label="DocVQA / InfoVQA")
    ax.bar(list(x), matched, wdt, label="InfoVQA reweighted to DocVQA margins")
    ax.bar([i + wdt for i in x], half, wdt, label="DocVQA positive-margin half / InfoVQA")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xticks(list(x), models)
    ax.set_ylabel("certification SE ratio\n(median over candidates)")
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.set_ylim(0, 4.6)
    fig.tight_layout()
    fig.savefig(OUT / "research_q3.pdf")


def fig_scan():
    v1 = load("reports/research/Q8/validation.json")["V1_scan"]
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for lab, v in v1.items():
        ms = v["ms_per_query_median"]
        xs = [v["pages"][k] for k in sorted(ms, key=int)]
        ys = [ms[k] for k in sorted(ms, key=int)]
        ax.plot(xs, ys, marker="o", ms=3, lw=1, label=lab.replace("hierarchical_merge", "Ward").replace("(per_vector)", ""))
    ax.set_xlabel("corpus pages (ColQwen2 DocVQA tiled; timing only)")
    ax.set_ylabel("scan time (ms / query)")
    ax.set_xticks([500, 1000, 2000, 4000])
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "research_scan.pdf")


if __name__ == "__main__":
    fig_targets()
    fig_axes()
    fig_q3()
    fig_scan()
    print("written:", sorted(p.name for p in OUT.glob("research_*.pdf")))
