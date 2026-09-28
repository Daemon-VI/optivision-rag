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


def render(path: Path) -> str | None:
    """Pick the renderer from the file's own shape; None for formats shown elsewhere."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None  # audit files are JSON lists; see docs/RELEASE-AUDIT-2026-09-27.md
    if "study" in data:
        return study_table(path)
    rows = data.get("rows") or []
    if not isinstance(rows, list):
        return None  # geometry files keep a dict of rows; shown in docs/UNIVERSAL.md (R8)
    if rows and "hot" in rows[0]:
        return tiered_table(path)
    if rows and "retention:ndcg@5" in rows[0] and "label" in rows[0] and "reference" in data:
        return matrix_table(path)
    return None  # frontier / centred-codec / performance files: see docs/UNIVERSAL.md


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/universal")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        blocks = [b for b in (render(f) for f in sorted(d.glob("*.json"))) if b]
        if blocks:
            print(f"## {d.name}\n")
            print("\n".join(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
