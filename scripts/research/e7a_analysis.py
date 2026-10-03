"""E7a analysis (reports/research/Q7/E7a/PLAN.md sections 5-7): paired attribution
comparisons and the relation to SAP's published values. Writes
reports/research/Q7/E7a/analysis.json.

    PYTHONPATH="src;scripts/research" python scripts/research/e7a_analysis.py
"""

from __future__ import annotations

import json

import e1_common as E
import numpy as np

OUT = E.ROOT / "reports" / "research" / "Q7" / "E7a"
STORES = ("colpali_docvqa", "colpali_infovqa", "colqwen2_docvqa", "colqwen2_infovqa",
          "colqwen25_docvqa", "colqwen25_infovqa")
RATIOS = ("1/5", "1/10", "1/20")
M = ("A_kmeans_plain", "B_ward_plain", "C_ward_rescaled", "D_random_prune")
PAIRS = (("B_ward_plain", "A_kmeans_plain"), ("C_ward_rescaled", "B_ward_plain"),
         ("A_kmeans_plain", "D_random_prune"), ("B_ward_plain", "D_random_prune"),
         ("C_ward_rescaled", "D_random_prune"))
# SAP Tables 7/8 (arXiv 2601.20107v3 HTML, transcribed in Q7/REPORT.md section 4; not PDF-verified), retention %
SAP_GAMMA = {"1/5": "0.20", "1/10": "0.10", "1/20": "0.05"}
SAP = {
    "colpali_docvqa": {"Cluster": {"0.20": 93.44, "0.10": 86.09, "0.05": 76.05},
                       "Random": {"0.20": 91.65, "0.10": 85.03, "0.05": 73.58}},
    "colpali_infovqa": {"Cluster": {"0.20": 98.04, "0.10": 94.64, "0.05": 87.10},
                        "Random": {"0.20": 96.37, "0.10": 93.16, "0.05": 89.42}},
    "colqwen2_docvqa": {"Cluster": {"0.20": 94.74, "0.10": 87.41, "0.05": 75.37},
                        "Random": {"0.20": 89.04, "0.10": 77.75, "0.05": 62.83}},
    "colqwen2_infovqa": {"Cluster": {"0.20": 95.89, "0.10": 89.70, "0.05": 79.25},
                         "Random": {"0.20": 93.53, "0.10": 89.14, "0.05": 82.52}},
}


def paired(a, b, base, nb=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(base), (nb, len(base)))
    bm = base[idx].mean(1)
    d = a[idx].mean(1) / bm - b[idx].mean(1) / bm
    point = float(a.mean() / base.mean() - b.mean() / base.mean())
    lo, hi = np.quantile(d, [0.025, 0.975]).tolist()
    return {"diff": point, "ci": [lo, hi], "material": bool((lo > 0 or hi < 0) and abs(point) >= 0.01)}


def main():
    out = {"datasets": {}, "sap": {}}
    for s in STORES:
        res = json.loads((OUT / f"{s}.json").read_text())
        z = np.load(OUT / f"{s}.npz")
        base = z["base"]
        d = {"checks": res["checks"], "protected_per_page": res["protected_per_page"], "cells": {}}
        for r in RATIOS:
            cell = res["cells"][r]
            get = lambda m, z=z, r=r: z[f"{r.replace('/', '_')}|{m}"]
            d["cells"][r] = {"methods": {m: {k: cell["methods"][m][k] for k in
                                             ("retention", "retention_ci", "retention_per_seed", "seed_range", "top1",
                                              "vectors_per_page", "vector_compression_factor", "vectors_kept_total")}
                                         for m in M},
                             "ward_clusters_exact": cell["ward_clusters_exact"], "k_free_total": cell["k_free_total"],
                             "comparisons": {f"{a} - {b}": paired(get(a), get(b), base) for a, b in PAIRS}}
        out["datasets"][s] = d
        if s in SAP:
            rows = {}
            for r in RATIOS:
                g = SAP_GAMMA[r]
                A = 100 * d["cells"][r]["methods"]["A_kmeans_plain"]["retention"]
                C = 100 * d["cells"][r]["methods"]["C_ward_rescaled"]["retention"]
                D = 100 * d["cells"][r]["methods"]["D_random_prune"]["retention"]
                sc, sr = SAP[s]["Cluster"][g], SAP[s]["Random"][g]
                ga, gc = abs(A - sc), abs(C - sc)
                rows[r] = {"gamma": g, "E7a_A_kmeans_plain": A, "E7a_C_ward_rescaled": C, "E7a_D_random": D,
                           "SAP_Cluster": sc, "SAP_Random": sr, "A_minus_SAP_Cluster": A - sc,
                           "C_minus_SAP_Cluster": C - sc, "D_minus_SAP_Random": D - sr,
                           "discrepancy_vs_Q7": ("reduced" if ga <= gc - 1 else "increased" if ga >= gc + 1
                                                 else "unchanged")}
            out["sap"][s] = rows
    # recommendation inputs (PLAN section 7)
    cells = [(s, r) for s in SAP for r in ("1/5", "1/10")]
    close = sum(abs(out["sap"][s][r]["A_minus_SAP_Cluster"]) <= 2 for s, r in cells)
    rnd_gaps = [abs(out["sap"][s][r]["D_minus_SAP_Random"]) for s in SAP for r in RATIOS]
    out["recommendation_inputs"] = {"A_within_2pts_of_SAP_Cluster_cells_gamma_0.10_0.20": f"{close}/{len(cells)}",
                                    "median_abs_D_minus_SAP_Random": float(np.median(rnd_gaps)),
                                    "max_abs_D_minus_SAP_Random": float(np.max(rnd_gaps))}
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["recommendation_inputs"]))


if __name__ == "__main__":
    main()
