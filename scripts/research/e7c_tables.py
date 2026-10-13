"""E7c report tables (reports/research/Q7/E7c/REPORT.md): writes tables.md next to the results."""

from __future__ import annotations

import json

import e1_common as E
import numpy as np

D = E.ROOT / "reports" / "research" / "Q7" / "E7c"
NAMES = {"colpali_docvqa": "ColPali DocVQA", "colpali_infovqa": "ColPali InfoVQA",
         "colqwen2_docvqa": "ColQwen2 DocVQA", "colqwen2_infovqa": "ColQwen2 InfoVQA"}
# SAP Tables 7/8 (arXiv 2601.20107v3 HTML transcription, Q7/REPORT.md section 4): baseline nDCG@5, retention %
SAP = {"colpali_docvqa": (0.59, {"1/5": 93.44, "1/10": 86.09, "1/20": 76.05}, {"1/5": 91.65, "1/10": 85.03, "1/20": 73.58}),
       "colpali_infovqa": (0.85, {"1/5": 98.04, "1/10": 94.64, "1/20": 87.10}, {"1/5": 96.37, "1/10": 93.16, "1/20": 89.42}),
       "colqwen2_docvqa": (0.59, {"1/5": 94.74, "1/10": 87.41, "1/20": 75.37}, {"1/5": 89.04, "1/10": 77.75, "1/20": 62.83}),
       "colqwen2_infovqa": (0.91, {"1/5": 95.89, "1/10": 89.70, "1/20": 79.25}, {"1/5": 93.53, "1/10": 89.14, "1/20": 82.52})}
V = ("E0", "E0p", "E1", "E2", "OL", "OM")


def main():
    data = {s: json.loads((D / f"{s}.json").read_text(encoding="utf-8")) for s in NAMES}
    out = ["### Baselines (nDCG@5, uncompressed)", "",
           "| dataset | queries changed by label rule | " + " | ".join(V) + " | SAP |", "|---|---|" + "---|" * (len(V) + 1)]
    for s, d in data.items():
        b = d["baseline"]
        out.append(f"| {NAMES[s]} | {d['label_stats']['queries_last_differs_from_first']} (of {d['label_stats']['queries']}) | "
                   + " | ".join(f"{b[v]:.4f}" for v in V) + f" | {SAP[s][0]:.2f} |")
    for key, title in (("D_random_prune", "Random pruning (primary control)"), ("A_kmeans_plain", "K-means, plain mean")):
        out += ["", f"### {title}: retention % by evaluation variant (mean of seeds 0, 1, 2)", "",
                "| dataset | ratio | " + " | ".join(V) + " | OM 95% interval |", "|---|---|" + "---|" * (len(V) + 1)]
        for s, d in data.items():
            for r, c in d["cells"].items():
                e = c[key]
                ci = e["OM"]["retention_ci"]
                out.append(f"| {NAMES[s]} | {r} | " + " | ".join(f"{100 * e[v]['retention']:.2f}" for v in V)
                           + f" | {100 * ci[0]:.1f}–{100 * ci[1]:.1f} |")
    out += ["", "### A (current) / B (aligned) / C (SAP published)", "",
            "| dataset | ratio | baseline A / B (OM) / C | random A / B (OM) / C | B − A | K-means A / B (OM) / SAP Cluster | B − A |",
            "|---|---|---|---|---|---|---|"]
    gaps = {"ColQwen2": {"E0": [], "OM": [], "OL": []}, "ColPali": {"E0": [], "OM": [], "OL": []}}
    for s, d in data.items():
        b = d["baseline"]
        for r, c in d["cells"].items():
            Dr, Ar = c["D_random_prune"], c["A_kmeans_plain"]
            out.append(f"| {NAMES[s]} | {r} | {b['E0']:.4f} / {b['OM']:.4f} / {SAP[s][0]:.2f} | "
                       f"{100 * Dr['E0']['retention']:.2f} / {100 * Dr['OM']['retention']:.2f} / {SAP[s][2][r]:.2f} | "
                       f"{100 * (Dr['OM']['retention'] - Dr['E0']['retention']):+.2f} | "
                       f"{100 * Ar['E0']['retention']:.2f} / {100 * Ar['OM']['retention']:.2f} / {SAP[s][1][r]:.2f} | "
                       f"{100 * (Ar['OM']['retention'] - Ar['E0']['retention']):+.2f} |")
            grp = "ColQwen2" if "colqwen2" in s else "ColPali"
            for v in gaps[grp]:
                gaps[grp][v].append(abs(100 * Dr[v]["retention"] - SAP[s][2][r]))
    out += ["", "### Random-control gap to SAP Random, |D − SAP| in points (decision-rule quantity)", "",
            "| model | E0 (current) median / max | OL (legacy official) median / max | OM (MTEB official) median / max | change in median, OM against E0 |",
            "|---|---|---|---|---|"]
    for g, x in gaps.items():
        e0, ol, om = (np.median(x[v]) for v in ("E0", "OL", "OM"))
        out.append(f"| {g} | {e0:.2f} / {max(x['E0']):.2f} | {ol:.2f} / {max(x['OL']):.2f} | "
                   f"{om:.2f} / {max(x['OM']):.2f} | {100 * (om - e0) / e0:+.1f}% |")
    (D / "tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
