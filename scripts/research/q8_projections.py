"""Q8 projections (reports/research/Q8/PLAN.md section 6): arithmetic on MEASURED primitives.

Storage / RAM: index(N) = per-page variable bytes x N + shared state (codec state + PCA basis).
Search cost:   PROJECTED only where the V1 validation supports linearity in stored vectors.
Writes reports/research/Q8/projections.json and reports/research/Q8/tables.md.

    PYTHONPATH="src;scripts/research" python scripts/research/q8_projections.py
"""

from __future__ import annotations

import json

import e1_common as E
import numpy as np

R = E.ROOT / "reports" / "research"
OUT = R / "Q8"
SIZES = (500, 1_000, 10_000, 100_000, 1_000_000)
RAM_BUDGET = 8 * 1024**3  # PLAN section 6: about half of this laptop's 15.8 GB, an assumption
STORES = ("colpali_docvqa", "colpali_infovqa", "colqwen2_docvqa", "colqwen2_infovqa",
          "colqwen25_docvqa", "colqwen25_infovqa")
CONFIGS_128 = ["float32", "float16", "int8(per_vector)", "int4(mean)", "binary", "hierarchical_merge(0.25)",
               "hierarchical_merge(0.25) > int8(per_vector)", "hierarchical_merge(0.25) > binary",
               "project(64) > int8(per_vector)", "hierarchical_merge(0.25) > project(64) > binary"]
CONFIGS_WIDE = ["float32", "int8(per_vector)", "binary", "hierarchical_merge(0.25) > int8(per_vector)",
                "project(640) > int8(per_vector)", "hierarchical_merge(0.25) > project(640) > binary"]


def gb(x):
    return x / 1e9


def per_page_128(store):
    d = json.loads((R / "Q4" / "store" / f"{store}.json").read_text())
    n = d["n_docs"]
    rows = {r["label"]: r for r in d["rows"]}
    fp32_pp = d["baseline"]["float32_code_bytes_per_doc"]
    out = {}
    for lab in CONFIGS_128:
        r = rows[lab]
        variable_pp = (r["code_bytes"] + r["offsets_bytes"]) / n  # offsets: 8 bytes per page (+8 once; negligible)
        shared = r["codec_state_bytes"] + r["projection_basis_bytes"]
        out[lab] = {"vectors_per_page": r["vectors_per_doc"], "dim": r["dim"], "bits_per_vector": r["bits_per_vector"],
                    "variable_bytes_per_page": variable_pp, "shared_bytes": shared,
                    "measured_index_bytes_500": r["index_bytes"], "factor_vs_fp32_codes": fp32_pp / (r["code_bytes"] / n)}
    tiers = {t["hot"]: t for t in d["tiered"]}
    t = tiers["hierarchical_merge(0.25) > binary"]
    out["two-tier (Ward 1/4 > binary hot, int8 cold)"] = {
        "ram_bytes_per_page": t["ram_index_bytes_per_doc"], "disk_bytes_per_page": t["disk_index_bytes_per_doc"],
        "shared_bytes": 0}
    return out, fp32_pp


def per_page_wide(split):
    d = json.loads((E.ROOT / "reports" / "universal" / "wide" /
                    f"nemotron-colembed-vl-4b-v2_{split}_test_subsampled.json").read_text())
    rows = {r["label"]: r for r in d["rows"] if not any(s.get("method") == "random" for s in r["pipeline"]["stages"])}
    out = {}
    for lab in CONFIGS_WIDE:
        r = rows[lab]
        basis = sum(2560 * s["dim"] * 4 for s in r["pipeline"]["stages"] if s["stage"] == "project")
        out[lab] = {"vectors_per_page": r["vectors_per_doc"], "dim": r["dim"],
                    "variable_bytes_per_page": r["bytes_per_doc"] + 8, "shared_bytes": basis,
                    "factor_vs_fp32_codes": r["compression_vs_float32"]}
    for t in d["tiered"]:
        out[f"two-tier ({t['hot']} hot, int8 cold)"] = {"ram_bytes_per_page": t["ram_bytes_per_doc"],
                                                         "disk_bytes_per_page": t["disk_bytes_per_doc"], "shared_bytes": 0}
    return out, d["baseline"]["float32_bytes_per_doc"]


def project_storage(cfg):
    if "ram_bytes_per_page" in cfg:
        return {N: {"ram_GB": gb(cfg["ram_bytes_per_page"] * N), "disk_GB": gb(cfg["disk_bytes_per_page"] * N)}
                for N in SIZES}
    return {N: {"index_GB": gb(cfg["variable_bytes_per_page"] * N + cfg["shared_bytes"]),
                "shared_share": cfg["shared_bytes"] / (cfg["variable_bytes_per_page"] * N + cfg["shared_bytes"])}
            for N in SIZES}


def ram_crossover(cfg):
    pp = cfg.get("ram_bytes_per_page", cfg.get("variable_bytes_per_page"))
    return int((RAM_BUDGET - cfg.get("shared_bytes", 0)) // pp)


def main():
    val = json.loads((OUT / "validation.json").read_text())
    res = {"ram_budget_bytes": RAM_BUDGET, "sizes": SIZES, "models_128": {}, "colembed_4b": {}, "latency": {}}
    for s in STORES:
        cfgs, fp32 = per_page_128(s)
        res["models_128"][s] = {"float32_bytes_per_page": fp32, "configs": {
            k: {**v, "storage": project_storage(v), "pages_within_8GB_ram": ram_crossover(v)} for k, v in cfgs.items()}}
    for split in ("docvqa", "infovqa"):
        cfgs, fp32 = per_page_wide(split)
        res["colembed_4b"][split] = {"float32_bytes_per_page": fp32, "configs": {
            k: {**v, "storage": project_storage(v), "pages_within_8GB_ram": ram_crossover(v)} for k, v in cfgs.items()}}
    # search cost: PROJECTED from the measured Q4 CPU ms/query at N = 500 (DocVQA), only for configurations whose
    # scan scaling was validated as linear in V1, using the V1 linear fit's slope relative to its N=500 value
    v1 = val["V1_scan"]
    linear = {k: v["linear_by_plan_criterion"] for k, v in v1.items()}
    for model, store in (("ColPali", "colpali_docvqa"), ("ColQwen2", "colqwen2_docvqa"), ("ColQwen2.5", "colqwen25_docvqa")):
        lat = json.loads((R / "Q4" / "latency" / f"{store}.json").read_text())
        rows = {r["label"]: r for r in lat["rows"]}
        out = {}
        for lab, v in v1.items():
            if lab not in rows:
                continue
            ms500 = rows[lab]["scan_ms_per_query"]["median"]
            if not linear[lab]:
                out[lab] = {"measured_ms_per_query_500": ms500, "projection": "not projected (V1 did not support linearity)"}
                continue
            fit = v["linear_fit_ms_per_query"]
            n500 = v["vectors"]["1"]
            fit500 = fit["slope_per_vector"] * n500 + fit["intercept"]
            q4 = json.loads((R / "Q4" / "store" / f"{store}.json").read_text())
            idx_pp = next(r for r in q4["rows"] if r["label"] == lab)["index_bytes"] / q4["n_docs"]
            # scale the model's own measured N=500 value by the validated relative growth (intercept kept);
            # PLAN section 6: no projection once the index no longer fits the 8 GB RAM assumption (no paging model)
            proj = {N: (ms500 * (fit["slope_per_vector"] * n500 * N / 500 + fit["intercept"]) / fit500
                        if idx_pp * N <= RAM_BUDGET else None) for N in SIZES}
            out[lab] = {"measured_ms_per_query_500": ms500, "projected_ms_per_query": proj,
                        "index_GB": {N: gb(idx_pp * N) for N in SIZES},
                        "assumption": "linear in stored vectors (V1, ColQwen2 DocVQA tiled to 4,000 pages); "
                                      "same per-page vectors; index resident in RAM (<= 8 GB); no paging; "
                                      "batch of all queries per call (ms/query amortized)"}
        res["latency"][model] = out
    # two-tier, ColQwen2 DocVQA only (V2 measured there): hot stage PROJECTED linear (V1); rescoring MEASURED
    hot = res["latency"]["ColQwen2"]["hierarchical_merge(0.25) > binary"]["projected_ms_per_query"]
    v2 = val["V2_rescore"]
    old_rs = float(np.mean([x["ms_per_query_median"]["old"] for x in v2.values()]))
    batched_low_overlap = v2["8"]["ms_per_query_median"]["batched"]
    res["two_tier_projection_colqwen2_docvqa"] = {
        N: {"hot_projected_ms": hot[N], "rescore_old_measured_ms": old_rs,
            "total_with_old_rescore_ms": None if hot[N] is None else hot[N] + old_rs,
            "batched_rescore": "MEASURED 1.53-4.03 ms over 500-3,984 distinct pages; NOT projected below that overlap",
            "hot_share_with_old_rescore": None if hot[N] is None else hot[N] / (hot[N] + old_rs)} for N in SIZES}
    res["batched_low_overlap_ms"] = batched_low_overlap
    res["two_tier_rescore"] = {m: {"old_ms": v["ms_per_query_median"]["old"], "batched_ms": v["ms_per_query_median"]["batched"],
                                   "distinct_pages": v["distinct_candidate_pages_per_batch"], "pages": v["pages"]}
                               for m, v in val["V2_rescore"].items()}
    (OUT / "projections.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    # tables
    L = []
    for s, d in res["models_128"].items():
        L += [f"#### {s} (float32 {d['float32_bytes_per_page'] / 1e3:.1f} KB per page)", "",
              "| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, c in d["configs"].items():
            st = c["storage"]
            if "ram_bytes_per_page" in c:
                L.append(f"| {k} | – | RAM {c['ram_bytes_per_page'] / 1e3:.1f} KB / disk {c['disk_bytes_per_page'] / 1e3:.1f} KB | 0 | – | "
                         f"RAM {st[500]['ram_GB'] * 1e3:.1f} MB / disk {st[500]['disk_GB'] * 1e3:.0f} MB | "
                         f"{st[10_000]['ram_GB']:.3f} / {st[10_000]['disk_GB']:.2f} GB | {st[100_000]['ram_GB']:.2f} / {st[100_000]['disk_GB']:.1f} GB | "
                         f"{st[1_000_000]['ram_GB']:.1f} / {st[1_000_000]['disk_GB']:.0f} GB | {c['pages_within_8GB_ram']:,} |")
            else:
                L.append(f"| {k} | {c['vectors_per_page']:.1f} × {c['dim']} | {c['variable_bytes_per_page'] / 1e3:.2f} KB | "
                         f"{c['shared_bytes'] / 1e3:.1f} KB | {c['factor_vs_fp32_codes']:.1f}x | {st[500]['index_GB'] * 1e3:.1f} MB | "
                         f"{st[10_000]['index_GB']:.2f} GB | {st[100_000]['index_GB']:.1f} GB | {st[1_000_000]['index_GB']:.1f} GB | "
                         f"{c['pages_within_8GB_ram']:,} |")
        L.append("")
    for split, d in res["colembed_4b"].items():
        L += [f"#### ColEmbed 4B {split} (float32 {d['float32_bytes_per_page'] / 1e6:.2f} MB per page)", "",
              "| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor (codes) | 500 | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, c in d["configs"].items():
            st = c["storage"]
            if "ram_bytes_per_page" in c:
                L.append(f"| {k} | – | RAM {c['ram_bytes_per_page'] / 1e3:.1f} KB / disk {c['disk_bytes_per_page'] / 1e6:.2f} MB | 0 | – | "
                         f"RAM {st[500]['ram_GB'] * 1e3:.0f} MB / disk {st[500]['disk_GB']:.2f} GB | {st[10_000]['ram_GB']:.2f} / {st[10_000]['disk_GB']:.1f} GB | "
                         f"{st[100_000]['ram_GB']:.1f} / {st[100_000]['disk_GB']:.0f} GB | {st[1_000_000]['ram_GB']:.0f} / {st[1_000_000]['disk_GB']:.0f} GB | "
                         f"{c['pages_within_8GB_ram']:,} |")
            else:
                L.append(f"| {k} | {c['vectors_per_page']:.1f} × {c['dim']} | {c['variable_bytes_per_page'] / 1e3:.1f} KB | "
                         f"{c['shared_bytes'] / 1e6:.2f} MB | {c['factor_vs_fp32_codes']:.1f}x | {st[500]['index_GB']:.3f} GB | "
                         f"{st[10_000]['index_GB']:.2f} GB | {st[100_000]['index_GB']:.1f} GB | {st[1_000_000]['index_GB']:.0f} GB | "
                         f"{c['pages_within_8GB_ram']:,} |")
        L.append("")
    L += ["#### Scan latency, laptop CPU (MEASURED at 500 pages; PROJECTED beyond under the V1 linearity assumption)", "",
          "| model | configuration | 500 MEASURED ms/query | 10k PROJECTED | 100k PROJECTED | 1M PROJECTED |", "|---|---|---|---|---|---|"]
    for model, d in res["latency"].items():
        for k, v in d.items():
            if "projected_ms_per_query" in v:
                p, ig = v["projected_ms_per_query"], v["index_GB"]
                f = lambda N, p=p, ig=ig: (f"{p[N]:.0f} ms" if p[N] < 1000 else f"{p[N] / 1e3:.2f} s") if p[N] is not None \
                    else f"not projected (index {ig[N]:.0f} GB > 8 GB)"
                L.append(f"| {model} | {k} | {v['measured_ms_per_query_500']:.2f} | {f(10_000)} | {f(100_000)} | {f(1_000_000)} |")
            else:
                L.append(f"| {model} | {k} | {v['measured_ms_per_query_500']:.2f} | {v['projection']} | | |")
    L += ["", "#### Two-tier, ColQwen2 DocVQA (hot Ward 1/4 > binary PROJECTED linear; rescoring MEASURED)", "",
          ("| pages | hot stage ms/query (PROJECTED) | old rescoring ms/query (MEASURED, flat 500-4,000) | "
           "total with old rescoring | hot share |"), "|---|---|---|---|---|"]
    for N, x in res["two_tier_projection_colqwen2_docvqa"].items():
        if x["hot_projected_ms"] is None:
            L.append(f"| {N:,} | not projected | {x['rescore_old_measured_ms']:.1f} | – | – |")
        else:
            L.append(f"| {N:,} | {x['hot_projected_ms']:.1f} | {x['rescore_old_measured_ms']:.1f} | "
                     f"{x['total_with_old_rescore_ms']:.1f} | {100 * x['hot_share_with_old_rescore']:.0f}% |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-25:]))


if __name__ == "__main__":
    main()
