"""Q3H: within-DocVQA relevance-margin split (reports/research/Q3/SPLIT.md).

One fixed split per model at the median of that model's DocVQA margins: high half
(margin > median) and low half (margin <= median). Per half: baseline, sensitivity,
pool retention against InfoVQA, the n = 225 certification SE and selection with the
frozen E1 code (e1_1_study.run_cell). Writes reports/research/Q3/split/<model>_<half>.json;
`summary` reads those and E1's selection rows and writes split/summary.json.

    PYTHONPATH="src;scripts/research" python scripts/research/q3_split.py ColPali high
    PYTHONPATH="src;scripts/research" python scripts/research/q3_split.py summary
"""

from __future__ import annotations

import json
import sys
import time

import e1_1_study as M
import e1_common as E
import numpy as np
from q3_analysis import REP, cert, loss_profile, quant, ratio
from q3_intervention import PAIRS

Q3 = E.ROOT / "reports" / "research" / "Q3"
OUT = Q3 / "split"
B = 2000
E1_ROWS = {"colpali": "real", "colqwen2": "real", "colqwen25": "confirm"}


def halves(model):
    doc, info = PAIRS[model]
    fd, fi = np.load(Q3 / "features" / f"{doc}.npz"), np.load(Q3 / "features" / f"{info}.npz")
    m = fd["base_margin_rel"]
    thr = float(np.median(m))
    edges = np.quantile(np.concatenate([m, fi["base_margin_rel"]]), [0.2, 0.4, 0.6, 0.8])
    return {"high": m > thr, "low": m <= thr}, thr, fd, fi, edges


def describe(b, rank, margin, edges):
    qi = np.searchsorted(edges, margin, side="right")
    return {"n": len(b), "b": quant(b), "hit1": float((rank == 1).mean()),
            "relevant_outside_top5": float((rank > 5).mean()), "margin_rel": quant(margin),
            "pooled_quintile_share": [float((qi == i).mean()) for i in range(5)]}


def main(model, half):
    t0 = time.time()
    doc, info = PAIRS[model]
    masks, thr, fd, fi, edges = halves(model)
    sel = masks[half]
    D, I = E.Store(E.load_store(doc), "labels"), E.Store(E.load_store(info), "labels")
    bh, ph = D.base[sel], D.pq[:, sel]
    sets = {"half": (bh, ph), "doc": (D.base, D.pq), "info": (I.base, I.pq)}
    desc = {"half": describe(bh, fd["base_rank"][sel], fd["base_margin_rel"][sel], edges),
            "doc": describe(D.base, fd["base_rank"], fd["base_margin_rel"], edges),
            "info": describe(I.base, fi["base_rank"], fi["base_margin_rel"], edges)}
    rng = np.random.default_rng(0)
    ia, ib = rng.integers(0, len(bh), (B, len(bh))), rng.integers(0, len(I.base), (B, len(I.base)))
    cands = []
    for k in range(D.K):
        ce = {s: cert(b, p[k]) for s, (b, p) in sets.items()}
        diff = ph[k][ia].sum(1) / bh[ia].sum(1) - I.pq[k][ib].sum(1) / I.base[ib].sum(1)
        cands.append({"position": k, "label": D.labels[k], "compression": float(D.compression[k]),
                      **{f"cert_{s}": v for s, v in ce.items()},
                      "R_half_minus_info_ci": [ratio(ph[k], bh) - ratio(I.pq[k], I.base),
                                               *np.quantile(diff, [0.025, 0.975]).tolist()]})
    sens = {k: {s: loss_profile(b, p[k]) for s, (b, p) in sets.items()} | {"label": D.labels[k]} for k in REP}

    def med_log(a, b):
        v = [np.log(c[f"cert_{a}"]["se_n225"] / c[f"cert_{b}"]["se_n225"]) for c in cands
             if all(c[f"cert_{s}"]["se_n225"] > 0 for s in sets)]
        return float(np.median(v)), len(v)

    gap, ng = med_log("doc", "info")
    hgap, _ = med_log("half", "info")
    prob = M.Problem(D.families, D.bytes, D.vectors, D.compression, ph.sum(axis=1) / bh.sum())
    rows = []
    for n in M.NS:
        def draw(seed, n=n):
            idx = np.random.default_rng(seed).integers(0, len(bh), size=n)
            return ph[:, idx], bh[idx]
        rows += M.summarise(prob, M.run_cell(prob, draw, n, "library"), n)
    out = {"model": model, "half": half, "threshold_docvqa_median_margin": thr,
           "label": "MEASURED per-query results split by a MEASURED feature; selection rows are SIMULATED resampling",
           "n_half": int(sel.sum()), "n_doc": len(D.base), "n_info": len(I.base), "pooled_quintile_edges": edges.tolist(),
           "describe": desc, "sensitivity": sens,
           "median_log_se_ratio_doc_over_info": gap, "median_log_se_ratio_half_over_info": hgap,
           "candidates_used": ng, "share_of_log_se_gap_closed": 1 - hgap / gap if gap else None,
           "R_half_below_info_beyond_ci": sum(1 for c in cands if c["R_half_minus_info_ci"][2] < 0),
           "R_half_above_info_beyond_ci": sum(1 for c in cands if c["R_half_minus_info_ci"][1] > 0),
           "candidates": cands, "selection_rows": rows, "environment": E.environment(), "seconds": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}_{half}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(model, half, "n", out["n_half"], "gap closed", out["share_of_log_se_gap_closed"], f"{time.time() - t0:.0f}s",
          flush=True)


def pick(rows, method, n=225, T=0.95):
    r = next(x for x in rows if x["n"] == n and x["target"] == T and x["method"] == method)
    return {"compression_median": r["compression_median"], "deployed_rate": r["deployed_rate"],
            "miss_rate": r["miss_rate"], "oracle_compression": r["oracle_compression"]}


def summary():
    res = {}
    for model, (doc, info) in PAIRS.items():
        e1 = {s: json.loads((E.E1 / E1_ROWS[s.split("_")[0]] / f"{s}.json").read_text())["rows"] for s in (doc, info)}
        h, lo = (json.loads((OUT / f"{model}_{x}.json").read_text()) for x in ("high", "low"))
        sel = {m: {"InfoVQA": pick(e1[info], m), "DocVQA": pick(e1[doc], m),
                   "high": pick(h["selection_rows"], m), "low": pick(lo["selection_rows"], m)}
               for m in ("A", "C@0.05")}
        closed = h["share_of_log_se_gap_closed"]
        within2 = all(sel[m]["high"]["compression_median"] * 2 >= sel[m]["InfoVQA"]["compression_median"] for m in sel)
        below2 = all(sel[m]["high"]["compression_median"] * 2 < sel[m]["InfoVQA"]["compression_median"] for m in sel)
        reading = ("A" if closed >= 0.75 and within2 else
                   "B" if closed <= 0.25 or (below2 and closed < 0.75) else "C")
        res[model] = {"threshold": h["threshold_docvqa_median_margin"], "n_high": h["n_half"], "n_low": lo["n_half"],
                      "share_closed_high": closed, "share_closed_low": lo["share_of_log_se_gap_closed"],
                      "R_high_below_info_beyond_ci": h["R_half_below_info_beyond_ci"],
                      "R_high_above_info_beyond_ci": h["R_half_above_info_beyond_ci"],
                      "selection_n225_T095": sel, "reading": reading}
    reads = [r["reading"] for r in res.values()]
    overall = ("A" if reads.count("A") >= 2 and "B" not in reads else
               "B" if reads.count("B") >= 2 and "A" not in reads else "C")
    (OUT / "summary.json").write_text(json.dumps({"models": res, "overall": overall}, indent=1), encoding="utf-8")
    print(json.dumps({m: (r["share_closed_high"], r["reading"]) for m, r in res.items()}), "overall", overall)


if __name__ == "__main__":
    summary() if sys.argv[1] == "summary" else main(sys.argv[1], sys.argv[2])
