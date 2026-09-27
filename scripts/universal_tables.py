"""Render the measured universal-layer results as Markdown tables.

Every number in docs/UNIVERSAL.md comes from here, read straight from the JSON the
benchmark scripts wrote -- nothing is typed by hand.

    python scripts/universal_tables.py reports/universal > reports/universal/TABLES.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def ci(row: dict, metric: str = "ndcg@5") -> str:
    lo, hi = row[f"retention_ci:{metric}"]
    return f"{pct(row[f'retention:{metric}'])} [{pct(lo)}, {pct(hi)}]"


def matrix_table(path: Path, keep=None) -> str:
    r = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"**{r['dataset']}** — {r['n_docs']} docs, {r['n_queries']} queries, reference: {r['reference']}", "",
             "| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for row in r["rows"]:
        if keep and not keep(row["label"]):
            continue
        lines.append(f"| {row['label']} | {row['vectors_per_doc']:.1f} | {row['dim']} | {row['bits_per_dim']:.3g} | "
                     f"{row['bytes_per_doc'] / 1e3:.2f} | {row['compression_vs_float32']:.1f}x | {row['ndcg@5']:.4f} | "
                     f"{ci(row)} |")
    return "\n".join(lines) + "\n"


def study_table(path: Path) -> str:
    r = json.loads(path.read_text(encoding="utf-8"))
    title = (f"**{r['dataset']}** — {r['n_docs']} docs, {r['n_queries']} queries, "
             "20 random half/half splits per row")
    header = ("| reference | rule | target | held-out met target | held-out mean | held-out worst | "
              "met (labels) | median compression |")
    lines = [title, "", header, "|---|---|---|---|---|---|---|---|"]
    for s in r["study"]:
        med = s["compression_median"]
        med_txt = "-" if med is None else f"{med:.1f}x"
        lines.append(f"| {s['reference']} | {s['safety']} | {s['target']:.2f} | "
                     f"{pct(s['met_on_holdout'])} ({s['selected_any']}/{s['splits']} selected) | "
                     f"{pct(s['holdout_retention_mean'])} | {pct(s['holdout_retention_min'])} | "
                     f"{pct(s['met_on_holdout_labels'])} | {med_txt} |")
    lines += ["", "Pseudo-query proxy (choose on document fragments, judge on real labelled queries):", "",
              "| target | chosen on fragments | fragment retention | real-query retention |", "|---|---|---|---|"]
    for p in r["pseudo_query_proxy"]:
        if p.get("selected"):
            lines.append(f"| {p['target']:.2f} | {p['selected']} | {pct(p['pseudo_retention'])} | "
                         f"{pct(p['real_retention_labels'])} |")
    lines += ["", "Fixed configurations that exist today (all queries):", "",
              "| configuration | KB/doc | x float32 | retention (labels) |", "|---|---|---|---|"]
    for f in r["fixed"]:
        lines.append(f"| {f['label']} | {f['bytes_per_doc'] / 1e3:.2f} | {f['compression']:.1f}x | "
                     f"{pct(f['retention_labels'])} |")
    return "\n".join(lines) + "\n"


def tiered_table(path: Path) -> str:
    r = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"**{r['dataset']}** — {r['n_docs']} docs, {r['n_queries']} queries", "",
             "| hot (RAM) | cold (disk) | shortlist | hot KB/doc | cold KB/doc | retention [95% CI] | rescore ms/q |",
             "|---|---|---|---|---|---|---|"]
    for row in r["rows"]:
        lo, hi = row["ci"]
        lines.append(f"| {row['hot']} | {row['cold'] or '-'} | {row['candidates'] or '-'} | "
                     f"{row['hot_bytes_per_doc'] / 1e3:.2f} | {row['cold_bytes_per_doc'] / 1e3:.2f} | "
                     f"{pct(row['retention'])} [{pct(lo)}, {pct(hi)}] | "
                     f"{row.get('rescore_ms_per_query', 0):.1f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/universal")
    for sub, render in (("merge", matrix_table), ("quantize_project", matrix_table), ("calibration", study_table),
                        ("tiered", tiered_table), ("text", matrix_table)):
        d = root / sub
        if not d.is_dir():
            continue
        print(f"## {sub}\n")
        for path in sorted(d.glob("*.json")):
            print(render(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
