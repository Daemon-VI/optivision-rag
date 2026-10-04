"""X1: label-free reference agreement (NEXT_PHASE_PREREGISTRATION.md, section 10).

The frozen procedure P0 (Method C over the 40 registered candidates) is run with
reference agreement in place of labelled retention. What is certified is agreement with
the uncompressed index: where a candidate places the document that index ranks first.
That is a different construct from labelled retention, and no number produced here is a
statement about relevance. Labelled values enter only as the registered cross-check
(conditions CP1 and CP2 of H4a) and the paired comparison; nothing is chosen with them.

    python scripts/research/x1_agreement.py                    dry run: validates, computes nothing
    python scripts/research/x1_agreement.py --run              the registered experiment
    python scripts/research/x1_agreement.py --run --store scifact    one store's unit of work

Outputs go to reports/research/X1/. The summary and the report are written once all
seven stores have a valid unit from the same commit.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e1_common as E  # noqa: E402
import x_common as X  # noqa: E402

EXPERIMENT, PREFIX = "X1", "x1"
N8 = ("Certifying agreement did not protect labelled retention on store {store}. "
      "Label-free certification certifies agreement only.")
N9 = ("Under agreement at 0.95, with every query the pool can supply, P0 did not reach non-trivial "
      "deployment on store {store}.")
N10 = ("Agreement is materially stricter than labelled retention. Using it cost at least a factor of 2 "
       "in compression.")
H4B_IF_FAILS = ("On every store and threshold the two free oracles were within a factor of 2 and CP1 held: "
                "the two constructs led to the same decisions on these stores. That would not establish "
                "equivalence elsewhere.")
MARGINS = {"relevance margin": "base_margin_rel", "top-2 margin": "base_margin_12"}  # A6, open point OP6
DIAGNOSTIC_READING = "more than half of the resamples deploy non-trivially"  # N10, Amendment 1, D1


def ranks_from_agreement(a: np.ndarray, cutoff: int) -> np.ndarray:
    """Rank the candidate gives to the reference's top document: 1..cutoff, or 0 beyond the cut-off.

    a = 1 / log2(1 + r) for r <= cutoff and 0 otherwise (section 6), so the rank is read back
    from the value. A value that is not on that grid means the array is not the registered construct.
    """
    ranks = np.zeros(a.shape, dtype=np.int64)
    matched = a == 0.0
    for r, v in enumerate(1.0 / np.log2(np.arange(2, cutoff + 2)), start=1):
        hit = np.abs(a - v) <= 1e-12
        ranks[hit] = r
        matched |= hit
    if not matched.all():
        raise X.GateFailure("agreement values that are not 1 / log2(1 + r): not the registered construct")
    return ranks


def oracles(construct: str, prob, threshold: float, meta: dict) -> dict:
    out = {}
    for kind, fn in (("free", X.free_oracle), ("chain", X.chain_oracle)):
        k = fn(prob, threshold)
        out[f"{kind}_oracle_{construct}"] = {
            "position": k, "label": None if k is None else meta["candidates"][k]["label"],
            "compression": X.oracle_compression(prob, k)}
    return out


def pool_tables(d: dict, spec: X.StoreSpec, prob: dict) -> tuple:
    """A1 and A2: pool-level quantities, exact functions of the stored per-query values. MEASURED."""
    meta = d["meta"]
    distinct = X.distinct_candidates(meta)
    truth_l, truth_a = prob["labels"].truth, prob["agreement"].truth
    g1 = np.asarray(d["base_labels"]) == 1.0
    a1 = []
    for k in distinct:
        c, a = meta["candidates"][k], d["cand_b1"][k]
        ranks = ranks_from_agreement(a, spec.cutoff)
        row = {"position": k, "label": c["label"], "family": c["family"],
               "compression": float(c["compression_vs_float32"]),
               "R_L": float(truth_l[k]), "R_A": float(truth_a[k]),
               "top1_agreement": float((ranks == 1).mean()),
               "containment@5": float(((ranks >= 1) & (ranks <= 5)).mean())}
        if spec.cutoff >= 10:
            row["containment@10"] = float(((ranks >= 1) & (ranks <= 10)).mean())
        if spec.vidore:  # G1: the reference document is the labelled page; G0: all other queries
            row["R_A_G1"] = float(a[g1].mean()) if g1.any() else None
            row["R_A_G0"] = float(a[~g1].mean()) if (~g1).any() else None
        a1.append(row)
    a2 = []
    for thr in X.THRESHOLDS["agreement"]:
        row = {"threshold": thr, "passing_agreement": [k for k in distinct if truth_a[k] >= thr],
               **oracles("agreement", prob["agreement"], thr, meta)}
        if thr in X.PRIMARY_THRESHOLDS:  # 0.90 is registered for agreement only
            row["passing_labels"] = [k for k in distinct if truth_l[k] >= thr]
            row.update(oracles("labels", prob["labels"], thr, meta))
            row["cp1_violations"] = [k for k in distinct
                                     if truth_a[k] >= thr and truth_l[k] < thr - X.CP1_TOLERANCE]
            row["cp1_holds"] = not row["cp1_violations"]
            ca, cl = row["free_oracle_agreement"]["compression"], row["free_oracle_labels"]["compression"]
            row["free_oracle_factor"] = float(max(ca / cl, cl / ca))
            row["free_oracles_differ_by_2"] = bool(row["free_oracle_factor"] >= X.ORACLE_FACTOR)
        a2.append(row)
    groups = {"queries_G1": int(g1.sum()), "queries_G0": int((~g1).sum())} if spec.vidore else {}
    return {"evidence": X.MEASURED, "distinct_candidates": len(a1), **groups, "rows": a1}, \
           {"evidence": X.MEASURED, "rows": a2}


def margin_analysis(name: str, d: dict, spec: X.StoreSpec, reg: X.Registry) -> dict:
    """A6, exploratory: agreement loss by quartile of the baseline score margin (open point OP6)."""
    if not spec.vidore:
        return {"computed": False, "reason": "Q3/features covers the ViDoRe stores only"}
    path = None if reg.q3_features is None else pathlib.Path(reg.q3_features) / f"{name}.npz"
    if path is None or not path.is_file():
        return {"computed": False, "reason": "no Q3 feature file for this store"}
    z = np.load(path, allow_pickle=False)
    rank = z["base_rank"]  # the features must be in the store's query order: they reproduce base_labels
    if rank.shape != d["base_labels"].shape or np.abs(
            np.where(rank <= spec.cutoff, 1.0 / np.log2(1.0 + rank), 0.0) - d["base_labels"]).max() > 1e-12:
        return {"computed": False, "reason": "the feature file is not aligned with the store's queries"}
    out = {"computed": True, "evidence": X.MEASURED, "exploratory": True, "features_file": path.name,
           "features_sha256": X.sha256_file(path), "margins": {}}
    for title, key in MARGINS.items():
        m = np.asarray(z[key], dtype=np.float64)
        ok = np.isfinite(m)
        edges = np.quantile(m[ok], [0.25, 0.5, 0.75])
        quartile = np.searchsorted(edges, m, side="right")  # 0 = smallest margins
        members = [ok & (quartile == q) for q in range(4)]
        rows = [{"position": k, "label": d["meta"]["candidates"][k]["label"],
                 "agreement_loss_by_quartile": [float(1.0 - d["cand_b1"][k][sel].mean()) if sel.any() else None
                                                for sel in members]}
                for k in X.distinct_candidates(d["meta"])]
        out["margins"][title] = {"array": key, "edges": [float(e) for e in edges],
                                 "queries_per_quartile": [int(sel.sum()) for sel in members],
                                 "queries_excluded_non_finite": int((~ok).sum()), "rows": rows}
    return out


def prepare(name: str, reg: X.Registry) -> dict:
    """Gates G2, G3 and G4 for one store. G1, the file hashes, is checked before any file is read."""
    spec = reg.stores[name]
    d = X.load_store(name, reg)
    gates = {"G2": X.require("G2", X.gate_values(d), name)}
    gates["G3"] = (X.require("G3", X.gate_identity(d), name) if spec.vidore
                   else {"applicable": False, "reason": "defined on the six ViDoRe stores"})
    stores = {c: E.Store(d, X.CONSTRUCT_REFERENCE[c]) for c in X.CONSTRUCTS}
    prob = {c: X.problem_for(stores[c]) for c in X.CONSTRUCTS}
    idx, _ = X.sample_indices(spec.n_queries, range(reg.n_seeds), reg.e1_budgets, reg)
    labels_e1 = X.simulate(stores["labels"].pq, stores["labels"].base, prob["labels"], idx,
                           X.PRIMARY_THRESHOLDS, methods=("C",))
    gates["G4"] = X.require("G4", X.gate_reproduction(name, labels_e1, reg, [f"C@{delta}" for delta in X.DELTAS]),
                            name)
    return {"d": d, "stores": stores, "prob": prob, "gates": gates, "labels_e1": labels_e1}


def analyse(name: str, prep: dict, reg: X.Registry, validation: dict) -> tuple:
    """One store: pool-level tables (MEASURED) and P0 under both constructs on the same resamples (SIMULATED)."""
    spec, d, st, prob = reg.stores[name], prep["d"], prep["stores"], prep["prob"]
    n_pool = spec.n_queries
    budgets, cap = reg.budgets(n_pool), reg.pool_capped_budget(n_pool)
    f16 = X.first_position(d["meta"], X.FLOAT16_PIPELINE)
    idx, nested = X.sample_indices(n_pool, range(reg.n_seeds), budgets, reg)

    rest = {n: idx[n] for n in budgets if n not in reg.e1_budgets}
    sel = {"labels": {**prep["labels_e1"],
                      **X.simulate(st["labels"].pq, st["labels"].base, prob["labels"], rest,
                                   X.PRIMARY_THRESHOLDS, methods=("C",))},
           "agreement": X.simulate(st["agreement"].pq, st["agreement"].base, prob["agreement"], idx,
                                   X.THRESHOLDS["agreement"], methods=("C",))}
    rep = {c: X.Replication(st[c].pq, st[c].base, prob[c], n_pool, reg) for c in X.CONSTRUCTS}
    monotone = {c: X.check_monotone(sel[c], prob[c], budgets, X.THRESHOLDS[c], nested, f"{name}, {c}")
                for c in X.CONSTRUCTS}
    rows = {c: X.simulated_rows(c, X.THRESHOLDS[c], sel[c], prob[c], rep[c], budgets, n_pool, f16, name)
            for c in X.CONSTRUCTS}

    def labelled_miss(k, target):  # the deployed candidate's R_L is below the same numeric threshold
        return (k >= 0) & (prob["labels"].truth[np.maximum(k, 0)] < target)

    cross = []  # A4; CP2 is its registered subset
    for target in X.THRESHOLDS["agreement"]:
        for delta in X.DELTAS:
            for n in budgets:
                method = f"C@{delta}"
                k = sel["agreement"][X.sel_key(n, target, method)]
                x = labelled_miss(k, target)
                r_l = prob["labels"].truth[k[k >= 0]]  # R_L of what the agreement arm deployed, per resample
                row = {**X.cell_identity("agreement", target, delta, n, n_pool, "P0"),
                       "deployments": int(r_l.size),
                       "R_L_of_deployed_min": float(r_l.min()) if r_l.size else None,
                       "R_L_of_deployed_median": float(np.median(r_l)) if r_l.size else None,
                       "cross_construct_miss": X.rate_with_interval(int(x.sum()), x.shape[0]),
                       "cp2_cell": bool(delta == X.PRIMARY_DELTA and target in X.PRIMARY_THRESHOLDS)}
                if row["cp2_cell"]:
                    row["cp2"] = X.check_miss_rate(
                        x, X.CP2_LEVEL,
                        lambda n=n, target=target, method=method: labelled_miss(
                            rep["agreement"].get(n, target, method), target))
                cross.append(row)

    paired = []
    n10 = {str(delta): {"eligible_cells": 0, "agreement_at_most_half": 0} for delta in X.DELTAS}  # OP7: decides N10
    n10_other = {str(delta): {"eligible_cells": 0, "agreement_at_most_half": 0} for delta in X.DELTAS}  # diagnostic
    for target in X.PRIMARY_THRESHOLDS:  # A5: the thresholds both arms share
        for delta in X.DELTAS:
            for n in budgets:
                key = X.sel_key(n, target, f"C@{delta}")
                ca = X.compression_of(sel["agreement"][key], prob["agreement"])
                cl = X.compression_of(sel["labels"][key], prob["labels"])
                labels_rate = X.deployed_flags(sel["labels"][key], prob["labels"], target, f16)["nontrivial"].mean()
                row = {**X.cell_identity("agreement against labels", target, delta, n, n_pool, "P0"),
                       "median_log2_compression_ratio": float(np.median(np.log2(ca / cl))),
                       "share_agreement_deploys_less": float((ca < cl).mean()),
                       "share_same": float((ca == cl).mean()),
                       "share_agreement_deploys_more": float((ca > cl).mean()),
                       "median_compression_agreement": float(np.median(ca)),
                       "median_compression_labels": float(np.median(cl)),
                       "labels_nontrivial_rate": float(labels_rate)}
                if row["budget_class"] == "pool-supported":
                    at_most_half = int(np.median(ca) <= 0.5 * np.median(cl))
                    # OP7: the labels arm deploys non-trivially when its rate is at least 0.90. The count with
                    # "more than half of the resamples" is diagnostic only and decides nothing (Amendment 1, D1).
                    for counts, eligible in ((n10, labels_rate >= X.RATE_THRESHOLD), (n10_other, labels_rate > 0.5)):
                        counts[str(delta)]["eligible_cells"] += int(eligible)
                        counts[str(delta)]["agreement_at_most_half"] += int(eligible) * at_most_half
                paired.append(row)

    method = f"C@{X.PRIMARY_DELTA}"
    h4c = X.decide_rate(
        X.deployed_flags(sel["agreement"][X.sel_key(cap, X.H4C_THRESHOLD, method)], prob["agreement"],
                         X.H4C_THRESHOLD, f16)["nontrivial"],
        lambda: X.deployed_flags(rep["agreement"].get(cap, X.H4C_THRESHOLD, method), prob["agreement"],
                                 X.H4C_THRESHOLD, f16)["nontrivial"])
    h4c.update(X.cell_identity("agreement", X.H4C_THRESHOLD, X.PRIMARY_DELTA, cap, n_pool, "P0"))

    a1, a2 = pool_tables(d, spec, prob)
    simulated = "SIMULATED: resamples of a MEASURED pool; each row carries its budget class"
    result = {
        "header": X.header(EXPERIMENT, reg, validation),
        "store": {"name": name, "encoder": spec.encoder, "pool": spec.pool, "pool_queries": n_pool,
                  "documents": spec.n_docs, "dim": spec.dim, "metric": spec.metric},
        "constructs": X.CONSTRUCT_NOTE,
        "budgets": X.budget_table(n_pool, reg), "pool_capped_budget": cap,
        "gates": {"G1": {"passed": True, "files": validation["stores"][name]}, **prep["gates"]},
        "sampling": {"seeds": reg.n_seeds, "nested_across_budgets": nested},
        "A1": a1, "A2": a2,
        "A3": {"evidence": simulated, "truth": "R_A over the pool", "rows": rows["agreement"]},
        "labels_arm": {"evidence": simulated, "truth": "R_L over the pool",
                       "note": "P0 under labels on the same resamples: the paired reference for A4 and A5",
                       "rows": rows["labels"]},
        "A4": {"evidence": simulated, "rows": cross},
        "A5": {"evidence": simulated, "rows": paired},
        "A6": margin_analysis(name, d, spec, reg),
        "H4c": h4c,
        "N10": {"by_delta": n10, "eligible_cells": sum(v["eligible_cells"] for v in n10.values()),
                "agreement_at_most_half": sum(v["agreement_at_most_half"] for v in n10.values()),
                "diagnostic_only": {
                    "reading": DIAGNOSTIC_READING, "by_delta": n10_other,
                    "eligible_cells": sum(v["eligible_cells"] for v in n10_other.values()),
                    "agreement_at_most_half": sum(v["agreement_at_most_half"] for v in n10_other.values())}},
        "checks": {"monotonicity": monotone},
        "replicated_cells": {c: sorted(rep[c].selections) for c in X.CONSTRUCTS},
    }
    arrays = {f"{c[0].upper()}_{k}": v for c in X.CONSTRUCTS for k, v in sel[c].items()}
    arrays.update({f"rep_{c[0].upper()}_{k}": v for c in X.CONSTRUCTS for k, v in rep[c].selections.items()})
    return result, arrays


def summarise(stores: dict, reg: X.Registry) -> dict:
    """H4a, H4b, H4c and the negative-result sentences of section 19, from the per-store files."""
    cp1 = [{"store": n, "threshold": r["threshold"], "candidates": r["cp1_violations"]}
           for n, s in stores.items() for r in s["A2"]["rows"] if r.get("cp1_holds") is False]
    cp2_rows = [(n, r) for n, s in stores.items() for r in s["A4"]["rows"] if r["cp2_cell"]]
    cp2 = [{"store": n, "threshold": r["threshold"], "n": r["n"], "budget_class": r["budget_class"],
            "evidence": r["evidence"], **r["cp2"]} for n, r in cp2_rows if r["cp2"]["confirmed_failure"]]
    h4a = {"CP1_holds": not cp1, "CP1_violations": cp1, "CP2_cells": len(cp2_rows),
           "CP2_cells_replicated": sum(r["cp2"]["replicated"] for _, r in cp2_rows),
           "CP2_holds": not cp2, "CP2_confirmed_failures": cp2,
           "verdict": "supported" if not cp1 and not cp2 else "fails"}
    differ = [{"store": n, "threshold": r["threshold"], "factor": r["free_oracle_factor"]}
              for n, s in stores.items() for r in s["A2"]["rows"] if r.get("free_oracles_differ_by_2")]
    h4b = {"free_oracles_differ_by_2": differ, "verdict": "supported" if differ or cp1 else "fails"}
    if h4b["verdict"] == "fails":
        h4b["conclusion"] = H4B_IF_FAILS
    failing = [n for n, s in stores.items() if not s["H4c"]["at_least_0.90"]]
    h4c = {"cells": {n: s["H4c"] for n, s in stores.items()}, "failing_stores": failing,
           "verdict": "supported" if not failing else "fails"}
    h4c["by_pool"] = X.worst_by_pool(h4c["cells"], reg)
    eligible = sum(s["N10"]["eligible_cells"] for s in stores.values())
    half = sum(s["N10"]["agreement_at_most_half"] for s in stores.values())
    n10 = {"eligible_cells": eligible, "agreement_at_most_half": half, "applies": bool(2 * half > eligible),
           # reported beside the count that decides; it enters no verdict, sentence or decision (Amendment 1, D1)
           "diagnostic_only": {
               "reading": DIAGNOSTIC_READING,
               "eligible_cells": sum(s["N10"]["diagnostic_only"]["eligible_cells"] for s in stores.values()),
               "agreement_at_most_half": sum(s["N10"]["diagnostic_only"]["agreement_at_most_half"]
                                             for s in stores.values())}}
    violating = [n for n in stores if any(v["store"] == n for v in cp1 + cp2)]
    sentences = ([N8.format(store=n) for n in violating] + [N9.format(store=n) for n in failing]
                 + ([N10] if n10["applies"] else []))
    return {"hypotheses": {"H4a": h4a, "H4b": h4b, "H4c": h4c}, "N10": n10,
            "negative_result_sentences": sentences,
            "scope": "development encoders only; agreement is not retrieval quality; nothing here bears on "
                     "model-blind transfer (sections 10.1 and 15.5)"}


def report(summary: dict, stores: dict) -> str:
    h = summary["hypotheses"]
    out = [
        "# X1: label-free reference agreement", "",
        f"Specification: `{X.PREREG_PATH}` section 10, commit `{X.PREREG_COMMIT}`. "
        f"Code commit `{summary['header']['code_commit']}`. Written by `scripts/research/x1_agreement.py`; "
        "every number is in the JSON files beside this report.", "",
        "**What is measured.** Reference agreement R_A: where a candidate places the document the uncompressed "
        "index ranks first. It is not retrieval quality, and nothing below is a statement about relevance. "
        "Labelled retention R_L appears only in the registered cross-checks. All seven stores are development "
        "data; nothing here bears on transfer to other encoders.", "",
        "## Summary", "",
        f"- **H4a (conservative proxy): {h['H4a']['verdict']}.** CP1 (MEASURED) "
        f"{'holds' if h['H4a']['CP1_holds'] else 'is violated'}; CP2 (SIMULATED) "
        f"{'holds' if h['H4a']['CP2_holds'] else 'is violated'} over {h['H4a']['CP2_cells']} cells, "
        f"{h['H4a']['CP2_cells_replicated']} of them re-run with seeds 2000-3999.",
        f"- **H4b (not equivalent): {h['H4b']['verdict']}.** "
        + (h["H4b"].get("conclusion") or f"{len(h['H4b']['free_oracles_differ_by_2'])} (store, threshold) pairs "
           "have free oracles a factor of 2 or more apart, or CP1 is violated."),
        f"- **H4c (usable with the pool; SIMULATED, pool-supported): {h['H4c']['verdict']}.**",
    ]
    out += [f"- {s}" for s in summary["negative_result_sentences"]]
    for v in h["H4a"]["CP2_confirmed_failures"]:
        out.append(f"- CP2 violated on {v['store']} at T_A = {v['threshold']}, n = {v['n']} ({v['budget_class']}).")
    out += ["", "## Gates", "",
            X.md_table(["store", "G1 hashes", "G2 values", "G3 identity", "G4 reproduces E1", "samples nested"],
                       [[n, s["gates"]["G1"]["passed"], s["gates"]["G2"]["passed"],
                         s["gates"]["G3"].get("passed", "not applicable"), s["gates"]["G4"]["passed"],
                         s["sampling"]["nested_across_budgets"]] for n, s in stores.items()]),
            "", "## A2. Candidates passing, oracles and CP1 (MEASURED)", "",
            "Compression of the free oracle and of the chain oracle under each construct; 1.0 means none.", "",
            X.md_table(["store", "threshold", "pass R_L", "pass R_A", "free oracle R_L", "free oracle R_A",
                        "chain oracle R_L", "chain oracle R_A", "CP1 holds"],
                       [[n, r["threshold"], len(r["passing_labels"]) if "passing_labels" in r else None,
                         len(r["passing_agreement"]),
                         r["free_oracle_labels"]["compression"] if "free_oracle_labels" in r else None,
                         r["free_oracle_agreement"]["compression"],
                         r["chain_oracle_labels"]["compression"] if "chain_oracle_labels" in r else None,
                         r["chain_oracle_agreement"]["compression"], r.get("cp1_holds")]
                        for n, s in stores.items() for r in s["A2"]["rows"]]),
            "", "## H4c. Non-trivial deployment under agreement at the pool-capped budget "
            "(SIMULATED, pool-supported)", "", "T_A = 0.95, delta = 0.05.", "",
            X.md_table(["store", "n", "rate [95% interval]", "seeds deciding", "borderline rule used", "at least 0.90"],
                       [[n, c["n"], X.fmt_rate(c), c.get("pooled_seeds", c["seeds"]), c["borderline_rule_used"],
                         c["at_least_0.90"]] for n, c in h["H4c"]["cells"].items()]),
            "", "Each pool by its worst store: "
            + "; ".join(f"{pool}: {w['worst_store']} ({w['deciding_rate']:.3f})"
                        for pool, w in h["H4c"]["by_pool"].items()) + "."]
    for target in X.THRESHOLDS["agreement"]:
        def cell(s, n, target=target):
            row = next(r for r in s["A3"]["rows"]
                       if r["threshold"] == target and r["delta"] == X.PRIMARY_DELTA and r["n"] == n)
            return f"{row['nontrivial']['rate']:.3f}"
        out += ["", f"## A3. Non-trivial deployment rate of P0 under agreement, T_A = {target}, delta = 0.05 "
                "(SIMULATED)", "",
                "An extrapolated column says what P0 would do if queries were drawn from the pool's own N "
                "queries. It is not what that many independent queries would give.", "",
                X.rate_by_budget_table(stores, cell)]

    def at_cap(s, block, target):
        return next(r for r in s[block]["rows"] if r["threshold"] == target and r["delta"] == X.PRIMARY_DELTA
                    and r["n"] == s["pool_capped_budget"])
    out += ["", "## A3 to A5 at the pool-capped budget, delta = 0.05 (SIMULATED, pool-supported)", "",
            "Miss rate against R_A; cross-construct miss rate against R_L (condition CP2, an empirical test "
            "with no statistical statement behind it).", "",
            X.md_table(["store", "threshold", "median compression, agreement", "median compression, labels",
                        "median log2 ratio", "miss rate vs R_A", "cross-construct miss rate vs R_L"],
                       [[n, t, at_cap(s, "A3", t)["compression_median"],
                         at_cap(s, "A5", t)["median_compression_labels"],
                         at_cap(s, "A5", t)["median_log2_compression_ratio"],
                         X.fmt_rate(at_cap(s, "A3", t)["miss"]), X.fmt_rate(at_cap(s, "A4", t)["cross_construct_miss"])]
                        for n, s in stores.items() for t in X.PRIMARY_THRESHOLDS]),
            "", "## N10 and A6", "",
            f"N10: in {summary['N10']['agreement_at_most_half']} of {summary['N10']['eligible_cells']} "
            "pool-supported cells where the labels arm deploys non-trivially, the median certified compression "
            "under agreement is at most half that under labels.", "",
            f"Diagnostic only, deciding nothing: with \"{DIAGNOSTIC_READING}\" in place of a rate of at least 0.90, "
            f"the count is {summary['N10']['diagnostic_only']['agreement_at_most_half']} of "
            f"{summary['N10']['diagnostic_only']['eligible_cells']}.", "",
            "A6 (exploratory, MEASURED): agreement loss by quartile of the baseline score margin is in each "
            "store's JSON file under `A6`. No hypothesis depends on it.", ""]
    return "\n".join(out)


def run(reg: X.Registry = X.REGISTERED, stores=None) -> dict:
    X.run_store_units(EXPERIMENT, reg, stores or list(reg.stores), prepare, analyse, PREFIX)
    return X.finish_experiment(EXPERIMENT, reg, PREFIX, summarise, report)


def dry_run(reg: X.Registry = X.REGISTERED) -> dict:
    rep = X.dry_run_common(EXPERIMENT, reg)  # includes the E1 selection files and the Q3 feature files
    rep["analyses"] = ["gates G1-G4", "A1 pool table (37 distinct candidates)", "A2 CP1 and oracles",
                       "A3 P0 under agreement at every registered budget", "A4 cross-construct miss (CP2)",
                       "A5 paired comparison with the labels arm", "A6 exploratory margin analysis",
                       "H4a, H4b, H4c; N8, N9, N10; the diagnostic-only N10 count"]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="X1: label-free reference agreement (preregistered).")
    ap.add_argument("--run", action="store_true", help="run the registered experiment (default: dry run)")
    ap.add_argument("--store", action="append", choices=list(X.REGISTERED.stores),
                    help="with --run: only this store's unit of work (repeatable)")
    ap.add_argument("--json", action="store_true", help="print the full dry-run report as JSON")
    args = ap.parse_args(argv)
    if not args.run:
        rep = dry_run()
        X.print_dry_run(rep, as_json=args.json)
        return 0 if rep["inputs_ok"] else 1
    run(X.REGISTERED, args.store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
