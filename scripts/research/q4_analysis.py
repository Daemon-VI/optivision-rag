"""Q4 analysis (reports/research/Q4/PLAN.md §7): axis curves, matched budgets,
H4a-H4e criteria, interactions, separate Pareto frontiers (in-sample and
held-out), cross-model and 128-d vs 2,560-d comparison.

Reads reports/research/Q4/store/*.json|npz, reports/research/Q4/latency/*.json and
R12's committed ColEmbed rows (reports/universal/wide/). Writes
reports/research/Q4/analysis.json and reports/research/Q4/TABLES.md.

    PYTHONPATH="src;scripts/research" python scripts/research/q4_analysis.py
"""

from __future__ import annotations

import json

import e1_common as E
import numpy as np

from optivision.evaluation import split_queries
from optivision.pareto import pareto_front

Q4 = E.ROOT / "reports" / "research" / "Q4"
R12 = E.ROOT / "reports" / "universal" / "wide"
MODELS = {"ColPali": "colpali", "ColQwen2": "colqwen2", "ColQwen2.5": "colqwen25"}
SPLITS = ("docvqa", "infovqa")
NB = 1000

LEVELS = {  # H4a / H4b matched single-axis levels (PLAN §2)
    "2x": {"A": "hierarchical_merge(0.5)", "B": "project(64)", "C": "float16"},
    "4x": {"A": "hierarchical_merge(0.25)", "B": "project(32)", "C": "int8(per_vector)"},
    "8x": {"A": "hierarchical_merge(0.125)", "B": "project(16)", "C": "int4(mean)"},
}
LEVELS_R12 = {
    "2x": {"A": "hierarchical_merge(0.5)", "B": "project(1280)", "C": "float16"},
    "4x": {"A": "hierarchical_merge(0.25)", "B": "project(640)", "C": "int8(per_vector)"},
    "8x": {"A": "hierarchical_merge(0.1)", "B": "project(320)", "C": "int4(mean)"},
}
BUDGETS = {  # PLAN §4
    "~4x": {"A": ["hierarchical_merge(0.25)"], "B": ["project(32)"], "C": ["int8(per_vector)"],
            "A+B": ["hierarchical_merge(0.5) > project(64)"]},
    "~8x": {"A": ["hierarchical_merge(0.125)"], "B": ["project(16)"], "C": ["int4(mean)"],
            "A+B": ["hierarchical_merge(0.25) > project(64)", "hierarchical_merge(0.5) > project(32)"],
            "A+C": ["hierarchical_merge(0.5) > int8(per_vector)", "hierarchical_merge(0.25) > float16"],
            "B+C": ["project(64) > int8(per_vector)", "project(32) > float16"],
            "A+B+C": ["hierarchical_merge(0.5) > project(64) > float16"]},
    "~32x": {"C": ["binary"], "A+B": ["hierarchical_merge(0.25) > project(16)"],
             "A+C": ["hierarchical_merge(0.25) > int4(mean)"],
             "B+C": ["project(64) > int4(mean)", "project(32) > int8(per_vector)"],
             "A+B+C": ["hierarchical_merge(0.25) > project(64) > int8(per_vector)"]},
}
STRUCT = ["hierarchical_merge(0.5)", "hierarchical_merge(0.33)", "hierarchical_merge(0.25)",
          "project(64)", "project(32)", "project(16)", "project(8)"] + [
    f"hierarchical_merge({r}) > project({w})" for r in (0.33, 0.25) for w in (64, 32, 16)] + [
    "hierarchical_merge(0.5) > project(64)"]
CODEC_LABELS = ["float16", "int8(per_vector)", "int4", "int4(mean)", "lloyd2(7)", "binary"]


# ------------------------------------------------------------------ loading

class DS:
    """One 128-d dataset: rows by label, per-query matrix, bootstrap indices."""

    def __init__(self, store):
        self.name = store
        self.meta = json.loads((Q4 / "store" / f"{store}.json").read_text())
        z = np.load(Q4 / "store" / f"{store}.npz")
        self.base = z["base"]
        self.pq = {str(lab): z["pq"][i] for i, lab in enumerate(z["labels"])}
        self.rows = {r["label"]: r for r in self.meta["rows"]}
        n = len(self.base)
        self.idx = np.random.default_rng(0).integers(0, n, (NB, n))
        self.bm = self.base[self.idx].mean(1)
        self.cal, self.hold = split_queries(n, 0.5, seed=0)

    def R(self, lab):
        return float(self.pq[lab].mean() / self.base.mean())

    def Rboot(self, lab):
        return self.pq[lab][self.idx].mean(1) / self.bm

    def diff(self, a, b):
        d = self.Rboot(a) - self.Rboot(b)
        return [self.R(a) - self.R(b), *np.quantile(d, [0.025, 0.975]).tolist()]

    def factor(self, lab):
        return self.meta["baseline"]["float32_code_bytes"] / self.rows[lab]["index_bytes"]


def r12(split):
    d = json.loads((R12 / f"nemotron-colembed-vl-4b-v2_{split}_test_subsampled.json").read_text())
    fp32 = d["baseline"]["float32_bytes_per_doc"] * d["n_docs"]
    # latency ratios use the float32 *row* (same score_space path as every other row),
    # not the separately timed baseline pass
    fp32_scan = next(r["scan_ms_per_query"] for r in d["rows"] if r["label"] == "float32")
    rows = {}
    for r in d["rows"]:
        lab = r["label"]
        if any(s.get("method") == "random" for s in r["pipeline"]["stages"]):
            lab += " [random]"
        basis = sum(2560 * s["dim"] * 4 for s in r["pipeline"]["stages"] if s["stage"] == "project")
        index = r["bytes_per_doc"] * d["n_docs"] + (d["n_docs"] + 1) * 8 + basis
        rows[lab] = {**r, "label": lab, "index_bytes": index, "index_bytes_per_doc": index / d["n_docs"],
                     "projection_basis_bytes": basis, "factor_index": fp32 / index,
                     "factor_codes_library": r["compression_vs_float32"],
                     "retention": r["retention:ndcg@5"], "retention_ci": r["retention_ci:ndcg@5"],
                     "scan_fraction_of_float32": r["scan_ms_per_query"] / fp32_scan}
    return d, rows


# ------------------------------------------------------------------ analyses

def axis_tables(ds: DS):
    out = []
    for r in ds.rows.values():
        if r["group"] == "tiered":
            continue
        out.append({k: r[k] for k in ("group", "label", "vectors_per_doc", "dim", "bits_per_vector", "index_bytes_per_doc",
                                      "code_bytes_per_doc", "projection_basis_bytes", "codec_state_bytes",
                                      "disk_bytes_per_doc", "persistable_by_library", "compression_vs_float32_index",
                                      "compression_vs_float32_codes", "ndcg@5", "retention", "retention_ci",
                                      "quality_loss", "incidental_compress_seconds_cached")})
    return out


def h4b_level(ds: DS, lv):
    L = LEVELS[lv]
    res = {ax: {"label": L[ax], "R": ds.R(L[ax]), "factor": ds.factor(L[ax])} for ax in L}
    best = max(L, key=lambda ax: res[ax]["R"])
    diffs = {o: ds.diff(L[best], L[o]) for o in L if o != best}
    res["best"] = best
    res["best_minus_others"] = diffs
    res["best_clear"] = all(v[1] > 0 for v in diffs.values())
    return res


def h4c(ds: DS):
    out = []
    for s in STRUCT:
        if s not in ds.rows:
            continue
        for c in CODEC_LABELS:
            lab = f"{s} > {c}"
            if lab in ds.rows:
                out.append({"structural": s, "codec": c, "added_factor": ds.rows[s]["index_bytes"] / ds.rows[lab]["index_bytes"],
                            "delta_R": ds.diff(lab, s), "R_struct": ds.R(s), "R_with_codec": ds.R(lab)})
    return out


def interactions(ds: DS):
    out = []
    for lab, r in ds.rows.items():
        parts = lab.split(" > ")
        if r["group"] not in ("A+B", "A+C", "B+C", "A+B+C") or not all(p in ds.rows for p in parts):
            continue
        prod_b = np.prod([ds.Rboot(p) for p in parts], axis=0)
        prod = float(np.prod([ds.R(p) for p in parts]))
        I = ds.Rboot(lab) - prod_b
        f_prod = float(np.prod([ds.factor(p) for p in parts]))
        out.append({"label": lab, "group": r["group"], "parts": parts, "R": ds.R(lab), "R_product_of_parts": prod,
                    "interaction": [ds.R(lab) - prod, *np.quantile(I, [0.025, 0.975]).tolist()],
                    "factor": ds.factor(lab), "factor_product_of_parts": f_prod,
                    "factor_ratio_actual_over_product": ds.factor(lab) / f_prod})
    return out


def interactions_r12(rows):
    out = []
    for lab, r in rows.items():
        parts = lab.split(" > ")
        if len(parts) < 2 or not all(p in rows for p in parts) or "int4" == parts[-1]:
            continue
        prod = float(np.prod([rows[p]["retention"] for p in parts]))
        fprod = float(np.prod([rows[p]["factor_index"] for p in parts]))
        out.append({"label": lab, "group": r["group"], "R": r["retention"], "R_product_of_parts": prod,
                    "interaction_point": r["retention"] - prod, "factor": r["factor_index"],
                    "factor_ratio_actual_over_product": r["factor_index"] / fprod})
    return out


def frontiers(ds: DS, lat=None):
    pts = [r for r in ds.rows.values() if r["group"] != "tiered"]
    tiered = ds.meta["tiered"]

    def front(items, cost, gain_key):
        return [x["label"] for x in pareto_front(items, costs=(cost,), gain=gain_key)]

    def entries(sub):
        es = []
        for r in pts:
            pq = ds.pq[r["label"]]
            es.append({"label": r["label"], "group": r["group"], "storage": r["index_bytes_per_doc"],
                       "ram": r["index_bytes_per_doc"], "R_all": ds.R(r["label"]),
                       "R_cal": float(pq[ds.cal].mean() / ds.base[ds.cal].mean()),
                       "R_hold": float(pq[ds.hold].mean() / ds.base[ds.hold].mean())})
        for t in tiered:
            lab = f"tiered[{t['hot']} -> int8, 50]"
            pq = ds.pq[lab]
            es.append({"label": lab, "group": "tiered", "storage": t["disk_index_bytes_per_doc"],
                       "ram": t["ram_index_bytes_per_doc"], "R_all": float(pq.mean() / ds.base.mean()),
                       "R_cal": float(pq[ds.cal].mean() / ds.base[ds.cal].mean()),
                       "R_hold": float(pq[ds.hold].mean() / ds.base[ds.hold].mean())})
        return es

    es = entries(pts)
    by = {e["label"]: e for e in es}
    out = {}
    for obj in ("storage", "ram"):
        ins = front(es, obj, "R_all")
        cal = front(es, obj, "R_cal")
        out[obj] = {"in_sample_selection": [{**by[x], "cost": by[x][obj]} for x in ins],
                    "held_out": [{"label": x, "group": by[x]["group"], "cost": by[x][obj], "R_cal": by[x]["R_cal"],
                                  "R_hold": by[x]["R_hold"]} for x in cal]}
    if lat is not None:
        le = [{**by[r["label"]], "latency": r["scan_ms_per_query"]["median"]} for r in lat["rows"] if r["label"] in by]
        lby = {e["label"]: e for e in le}
        out["latency"] = {"in_sample_selection": [{**lby[x], "cost": lby[x]["latency"]} for x in front(le, "latency", "R_all")],
                          "held_out": [{"label": x, "group": lby[x]["group"], "cost": lby[x]["latency"],
                                        "R_cal": lby[x]["R_cal"], "R_hold": lby[x]["R_hold"]}
                                       for x in front(le, "latency", "R_cal")],
                          "note": "latency subset only (PLAN §6), DocVQA"}
    return out


def h4a(lat, levels):
    by = {r["label"]: r for r in lat["rows"]}
    res = {}
    for lv, L in levels.items():
        m = {ax: by[L[ax]]["scan_ms_per_query"] for ax in L}
        spread = {ax: m[ax]["max"] - m[ax]["min"] for ax in L}
        a_wins = all(m["A"]["median"] + max(spread["A"], spread[o]) < m[o]["median"] for o in ("B", "C"))
        a_loses = any(m[o]["median"] <= m["A"]["median"] for o in ("B", "C"))
        res[lv] = {ax: {"label": L[ax], "median_ms": m[ax]["median"], "spread_ms": spread[ax],
                        "fraction_of_float32": by[L[ax]]["scan_fraction_of_float32"]} for ax in L}
        res[lv]["verdict"] = "A fastest beyond spread" if a_wins else ("B or C as fast as A" if a_loses else "A fastest within spread")
        # POST HOC (round 1 was a warm-up for every configuration; see REPORT): rounds 2-5 only,
        # and a within-round comparison over all five rounds
        a2 = {ax: np.asarray(by[L[ax]]["scan_ms_per_query"]["all"][1:]) for ax in L}
        sp2 = {ax: float(a2[ax].max() - a2[ax].min()) for ax in L}
        win2 = all(np.median(a2["A"]) + max(sp2["A"], sp2[o]) < np.median(a2[o]) for o in ("B", "C"))
        allr = {ax: np.asarray(by[L[ax]]["scan_ms_per_query"]["all"]) for ax in L}
        rounds_a_fastest = int(np.sum((allr["A"] < allr["B"]) & (allr["A"] < allr["C"])))
        res[lv]["post_hoc"] = {"rounds_2_5_medians": {ax: float(np.median(a2[ax])) for ax in L},
                               "rounds_2_5_spreads": sp2, "A_fastest_beyond_spread_rounds_2_5": bool(win2),
                               "rounds_where_A_fastest_of_5": rounds_a_fastest}
    return res


def h4a_r12(rows, levels):
    res = {}
    for lv, L in levels.items():
        res[lv] = {ax: {"label": L[ax], "scan_ms": rows[L[ax]]["scan_ms_per_query"],
                        "fraction_of_float32": rows[L[ax]]["scan_fraction_of_float32"]} for ax in L}
        res[lv]["verdict"] = ("A fastest" if all(rows[L["A"]]["scan_ms_per_query"] < rows[L[o]]["scan_ms_per_query"]
                                                  for o in ("B", "C")) else "B or C as fast as A")
    return res


def h4b_r12(rows):
    res = {}
    for lv, L in LEVELS_R12.items():
        r = {ax: {"label": L[ax], "R": rows[L[ax]]["retention"], "ci": rows[L[ax]]["retention_ci"],
                  "factor": rows[L[ax]]["factor_index"]} for ax in L}
        best = max(L, key=lambda ax: r[ax]["R"])
        r["best"] = best
        r["best_ci_above_others_point"] = all(r[best]["ci"][0] > r[o]["R"] for o in L if o != best)
        res[lv] = r
    return res


def normalized_curves(dss, r12rows):
    """Retention by axis at relative levels (fraction of own float32 bytes), per model."""
    levels = {"A": [("1/2", "hierarchical_merge(0.5)"), ("1/3", "hierarchical_merge(0.33)"),
                    ("1/4", "hierarchical_merge(0.25)"), ("1/10", "hierarchical_merge(0.1)")],
              "B_128": [("1/2", "project(64)"), ("1/4", "project(32)"), ("1/8", "project(16)"), ("1/16", "project(8)")],
              "B_2560": [("1/2", "project(1280)"), ("1/4", "project(640)"), ("1/8", "project(320)"), ("1/16", "project(160)")],
              "C": [(c, c) for c in CODEC_LABELS]}
    out = {}
    for name, ds in dss.items():
        out[name] = {ax: [{"level": lv, "label": lab, "R": ds.R(lab), "factor": ds.factor(lab)} for lv, lab in levels[k]]
                     for ax, k in (("A", "A"), ("B", "B_128"), ("C", "C"))}
    for split, rows in r12rows.items():
        out[f"ColEmbed4B_{split}"] = {ax: [{"level": lv, "label": lab, "R": rows[lab]["retention"],
                                            "factor": rows[lab]["factor_index"]} for lv, lab in levels[k] if lab in rows]
                                      for ax, k in (("A", "A"), ("B", "B_2560"), ("C", "C"))}
    return out


FRONTIER = E.ROOT / "reports" / "universal" / "frontier"
FRONTIER_FILES = {"colpali": "E2-colpali-{s}", "colqwen2": "colqwen2-v1.0-merged_{s}_test_subsampled",
                  "colqwen25": "colqwen2.5-v0.2_{s}_test_subsampled"}


def r5b_check(ds: DS):
    """Retention of rows shared with the committed R5b/R11 frontier files (PLAN section 7.5)."""
    p, s = ds.name.rsplit("_", 1)
    f = json.loads((FRONTIER / f"{FRONTIER_FILES[p].format(s=s)}.json").read_text())
    key = {json.dumps(r["pipeline"], sort_keys=True): lab for lab, r in ds.rows.items() if "pipeline" in r}
    shared = [(key[k], r["retention:ndcg@5"], ds.R(key[k])) for r in f["rows"]
              if (k := json.dumps(r["pipeline"], sort_keys=True)) in key]
    worst = max((abs(a - b) for _, a, b in shared), default=None)
    return {"shared_rows": len(shared), "max_abs_difference": worst, "ok": worst is not None and worst < 1e-9}


def main():
    dss = {f"{m}_{s}": DS(f"{p}_{s}") for m, p in MODELS.items() for s in SPLITS}
    lats = {m: json.loads((Q4 / "latency" / f"{p}_docvqa.json").read_text()) for m, p in MODELS.items()
            if (Q4 / "latency" / f"{p}_docvqa.json").exists()}
    r12d = {s: r12(s) for s in SPLITS}
    r12rows = {s: v[1] for s, v in r12d.items()}
    out = {"checks": {n: ds.meta["checks"] for n, ds in dss.items()},
           "r5b_r11_check": {n: r5b_check(ds) for n, ds in dss.items()}, "datasets": {}, "ColEmbed4B": {}}
    for n, ds in dss.items():
        model = n.rsplit("_", 1)[0]
        bud = {b: {ax: [{"label": lab, "R": ds.R(lab), "ci": ds.rows[lab]["retention_ci"], "factor": ds.factor(lab)}
                        for lab in labs if lab in ds.rows] for ax, labs in axes.items()} for b, axes in BUDGETS.items()}
        out["datasets"][n] = {
            "baseline": ds.meta["baseline"], "axis_rows": axis_tables(ds), "tiered": ds.meta["tiered"],
            "H4b": {lv: h4b_level(ds, lv) for lv in LEVELS}, "H4c": h4c(ds), "interactions": interactions(ds),
            "budgets": bud,
            "frontiers": frontiers(ds, lats.get(model) if n.endswith("docvqa") else None)}
    for m, lat in lats.items():
        out["datasets"][f"{m}_docvqa"]["H4a"] = h4a(lat, LEVELS)
        out["datasets"][f"{m}_docvqa"]["latency"] = lat["rows"]
        out["datasets"][f"{m}_docvqa"]["latency_tiered"] = lat["tiered"]
    for s, (d, rows) in r12d.items():
        out["ColEmbed4B"][s] = {"baseline": d["baseline"], "rows": list(rows.values()),
                                "H4a_gpu": h4a_r12(rows, LEVELS_R12), "H4b": h4b_r12(rows),
                                "interactions_point": interactions_r12(rows), "tiered": d["tiered"]}
    out["normalized_curves"] = normalized_curves(dss, r12rows)
    (Q4 / "analysis.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("written analysis.json")


if __name__ == "__main__":
    main()
