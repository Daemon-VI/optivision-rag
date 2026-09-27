"""Curated summary tables for docs/UNIVERSAL.md, generated from reports/universal/*.

    python scripts/universal_summary.py reports/universal

Every figure is read from the result JSON; the only thing chosen by hand is which
rows to show. The full per-row tables are in reports/universal/TABLES.md
(scripts/universal_tables.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VIDORE = [("E2-colpali-infovqa", "ColPali · ViDoRe InfoVQA"), ("E2-colpali-docvqa", "ColPali · ViDoRe DocVQA")]
GENERATED = [("E3-colpali-generated", "ColPali · generated"), ("E1-colsmol-generated", "ColSmol · generated")]
TEXT = [("text-answerai-colbert-small-v1-scifact", "ColBERT-small · SciFact")]


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["label"]: r for r in json.loads(path.read_text(encoding="utf-8"))["rows"]}


def cell(row, metric="ndcg@5", show_vec=True, show_x=False):
    if row is None:
        return "-"
    lo, hi = row[f"retention_ci:{metric}"]
    head = f"{row['vectors_per_doc']:.0f} vec · " if show_vec else ""
    x = f"{row['compression_vs_float32']:.0f}x · " if show_x else ""
    return f"{head}{x}{pct(row[f'retention:{metric}'])} [{pct(lo)}, {pct(hi)}]"


def table(root: Path, sub: str, labels: list[str], datasets, title: str, metric="ndcg@5", show_vec=True,
          show_x=False) -> str:
    data = {name: load_rows(root / sub / f"{name}.json") for name, _ in datasets}
    head = "| " + title + " | " + " | ".join(t for _, t in datasets) + " |"
    lines = [head, "|" + "---|" * (len(datasets) + 1)]
    for label in labels:
        cells = [cell(data[name].get(label), metric, show_vec, show_x) for name, _ in datasets]
        if all(c == "-" for c in cells):
            continue
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def calibration(root: Path) -> str:
    out = []
    names = [(n, t) for n, t in VIDORE + GENERATED if (root / "calibration" / f"{n}.json").exists()]
    text = root / "text" / "text-answerai-colbert-small-v1-scifact-calibration.json"
    if text.exists():
        names.append(("__text__", "ColBERT-small · SciFact (nDCG@10)"))
    lines = ["| reference · rule · target | " + " | ".join(t for _, t in names) + " |",
             "|" + "---|" * (len(names) + 1)]
    studies = {}
    for n, _ in names:
        path = text if n == "__text__" else root / "calibration" / f"{n}.json"
        studies[n] = {(s["reference"], s["safety"], s["target"]): s
                      for s in json.loads(path.read_text(encoding="utf-8"))["study"]}
    for ref in ("labels", "baseline@1"):
        for rule in ("point", "lower_ci"):
            for target in (0.99, 0.97, 0.95):
                cells = []
                for n, _ in names:
                    s = studies[n].get((ref, rule, target))
                    if s is None or s["met_on_holdout"] is None:
                        cells.append("-")
                        continue
                    labels = "" if ref == "labels" else f" (labels {pct(s['met_on_holdout_labels'])})"
                    cells.append(f"{pct(s['met_on_holdout'])}{labels} · x{s['compression_median']:.1f}")
                lines.append(f"| {ref} · {rule} · {target:.2f} | " + " | ".join(cells) + " |")
    out.append("\n".join(lines) + "\n")
    proxy = ["| dataset | target | chosen on fragments | fragment retention | real-query retention |",
             "|---|---|---|---|---|"]
    for n, t in names:
        if n == "__text__":
            continue
        for p in json.loads((root / "calibration" / f"{n}.json").read_text(encoding="utf-8"))["pseudo_query_proxy"]:
            if p.get("selected"):
                proxy.append(f"| {t} | {p['target']:.2f} | {p['selected']} | {pct(p['pseudo_retention'])} | "
                             f"{pct(p['real_retention_labels'])} |")
    out.append("\n".join(proxy) + "\n")
    return "\n".join(out)


def frontier(root: Path) -> str:
    """Per dataset: the smallest configuration whose retention (point, and CI
    lower bound) meets each target, plus the Pareto frontier."""
    out = []
    for path in sorted((root / "frontier").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        key = r["frontier_metric"]
        ci_key = key.replace("retention:", "ci:")
        metric = key.split(":")[1]
        out.append(f"**{r['dataset']}** — {r['n_docs']} docs, {r['n_queries']} queries, retention of {metric}\n")
        header = ("| target | smallest config, point estimate (in-sample) | x float32 | retention [95% CI] | "
                  "smallest config, 95% CI lower end (in-sample) | x float32 | retention [95% CI] |")
        lines = [header, "|---|---|---|---|---|---|---|"]
        for target in (0.99, 0.97, 0.95, 0.90):
            cells = []
            for use_lo in (False, True):
                ok = [row for row in r["rows"] if (row[ci_key][0] if use_lo else row[key]) >= target]
                if not ok:
                    cells += ["-", "-", "-"]
                    continue
                best = min(ok, key=lambda row: row["bytes_per_doc"])
                lo, hi = best[ci_key]
                cells += [best["label"], f"{best['compression']:.1f}x",
                          f"{pct(best[key])} [{pct(lo)}, {pct(hi)}]"]
            lines.append(f"| {target:.2f} | " + " | ".join(cells) + " |")
        out.append("\n".join(lines) + "\n")
        rows = {row["label"]: row for row in r["rows"]}
        front = ["Pareto frontier (bytes/doc vs retention): " + ", ".join(
            f"{lab} ({rows[lab]['compression']:.0f}x, {pct(rows[lab][key])})" for lab in r["frontier"])]
        out.append(front[0] + "\n")
    return "\n".join(out)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


SELECTION = [("E2-colpali-docvqa", "DocVQA"), ("E2-colpali-infovqa", "InfoVQA"),
             ("text-answerai-colbert-small-v1-scifact", "SciFact")]


def selection_rules(root: Path) -> str:
    """R7b: the shipped default (calibrate() replayed exactly) and the alternative rules."""
    data = {}
    for name, title in SELECTION:
        path = root / "selection_rules" / f"{name}.json"
        if path.exists():
            data[title] = json.loads(path.read_text(encoding="utf-8"))
    out = [("`calibrate()` with its defaults, labels as the reference, 20 random half/half splits. "
            "*met* = share of splits whose choice reached the target on the held-out half (95% Wilson "
            "interval over the 20 splits); *x* = median compression chosen; *held-out* = mean and worst "
            "held-out retention of the choice.\n"),
           "| dataset | queries | target | met [95% interval] | median x | held-out mean · worst |",
           "|---|---|---|---|---|---|"]
    for title, d in data.items():
        rows = {(r["reference"], r["rule"], r["target"]): r for r in d["results"]}
        for target in (0.99, 0.97, 0.95):
            r = rows.get(("labels", "calibrate() default", target))
            if r is None or r["met"] is None:
                continue
            ok = [o for o in r["outcomes"] if o["selected"]]
            k = sum(o["holdout"] >= target for o in ok)
            lo, hi = wilson(k, len(ok))
            out.append(f"| {title} | {d['n_queries']} | {target:.2f} | {k}/{len(ok)} [{pct(lo)}, {pct(hi)}] | "
                       f"{r['compression_median']:.1f}x | {pct(r['holdout_mean'])} · {pct(r['holdout_min'])} |")
    out.append("")
    rules = ["calibrate() default", "point", "lower 0.975", "lower 0.99", "lower 0.999", "bonferroni 0.975",
             "lower 0.975 + margin 0.01"]
    cols = [(t, x) for t in data for x in (0.99, 0.97, 0.95)]
    out.append("All rules (labels): met · median x.\n")
    out.append("| rule | " + " | ".join(f"{t} {x:.2f}" for t, x in cols) + " |")
    out.append("|" + "---|" * (len(cols) + 1))
    for rule in rules:
        cells = []
        for t, x in cols:
            r = {(q["reference"], q["rule"], q["target"]): q for q in data[t]["results"]}.get(("labels", rule, x))
            cells.append("-" if r is None or r["met"] is None else
                         f"{100 * r['met']:.0f}% · x{r['compression_median']:.1f}")
        out.append(f"| {rule} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/universal")
    merge_labels = [
        "spatial+redundancy (legacy)", "redundancy t=0.92", "redundancy t=0.8",
        "adaptive radius=0.8", "adaptive radius=0.7", "adaptive radius=0.6", "adaptive radius=0.5",
        "hierarchical ratio=0.5", "hierarchical ratio=0.25", "hierarchical ratio=0.1", "hierarchical ratio=0.05",
        "adaptive ratio=0.5", "adaptive ratio=0.25", "adaptive ratio=0.1", "adaptive ratio=0.05",
        "random ratio=0.5", "random ratio=0.25", "random ratio=0.1",
    ]
    print("### Token reduction (float32 vectors, nDCG@5 retention vs float, labels)\n")
    print(table(root, "merge", merge_labels, VIDORE + GENERATED, "method"))
    variant_labels = [f"ward distance={t} (per-doc count)" for t in (0.6, 0.8, 1.0, 1.3, 1.6)] + \
        [f"adaptive radius={r} refine=3" for r in (0.8, 0.7, 0.6)]
    print("### Merge variants\n")
    print(table(root, "quantize_project", variant_labels, VIDORE + GENERATED, "method"))
    centred_labels = [f"adaptive radius={r} centred" for r in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)] + \
        [f"adaptive ratio={f} centred" for f in (0.5, 0.25, 0.1)]
    print("### Adaptive merge with the corpus mean removed\n")
    print(table(root, "centred", centred_labels, VIDORE + GENERATED + TEXT, "method"))
    quant_labels = ["float16", "int8 fixed (legacy)", "int8 per-vector", "int8 per-dimension", "int4 per-vector",
                    "lloyd2 (2-bit)", "lloyd2 (2-bit, seed 1)", "lloyd2 (2-bit, seed 2)", "binary", "binary centred"]
    print("### Quantization alone (all vectors, nDCG@5 retention)\n")
    print(table(root, "quantize_project", quant_labels, VIDORE + GENERATED, "codec", show_vec=False, show_x=True))
    proj_labels = [f"pca-docs {d}" for d in (96, 64, 48, 32)] + [f"random-proj {d}" for d in (96, 64, 32)] + \
        [f"truncate {d}" for d in (96, 64, 32)]
    print("### Dimension reduction alone (float32, nDCG@5 retention)\n")
    print(table(root, "quantize_project", proj_labels, VIDORE + GENERATED, "projection", show_vec=False))
    combo_labels = [f"merge 0.6 > {c}" for c in ("int8 per-vector", "int4", "lloyd2", "binary", "binary centred",
                                                 "pca-docs 64 > int8 per-vector", "pca-docs 64 > binary")]
    print("### Merge + projection + quantization (nDCG@5 retention)\n")
    print(table(root, "quantize_project", combo_labels, VIDORE + GENERATED, "pipeline", show_x=True))
    if (root / "frontier").is_dir():
        print("### Token stage x codec frontier\n")
        print(frontier(root))
    print("### Calibration reliability (20 random half/half query splits per cell)\n")
    print("Share of splits whose chosen configuration met the target on the held-out half, "
          "and the median compression chosen.\n")
    print(calibration(root))
    if (root / "selection_rules").is_dir():
        print("### Selection rules with the default search space (R7b)\n")
        print(selection_rules(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
