"""X2: query budget and certification (NEXT_PHASE_PREREGISTRATION.md, section 11).

How the frozen procedure P0 and its comparators behave as the number of calibration
queries grows, under labels and under reference agreement, on the seven registered stores.

Two classes of budget are never mixed (section 11.1). A budget that does not exceed the
pool size is pool-supported. A larger one (900, 1800, 3600 on every store; 450 on SciFact)
is extrapolated: its samples redraw the pool's own queries, so a result there says what
P0 would do if queries came from exactly those N queries. It is not what that many
independent queries would give, and it decides no primary hypothesis. Both are SIMULATED.
The only MEASURED outcome is the single full-pool decision per store, construct and threshold.

    python scripts/research/x2_budget.py                     dry run: validates, computes nothing
    python scripts/research/x2_budget.py --run               the registered experiment
    python scripts/research/x2_budget.py --run --store scifact     one store's unit of work

Outputs go to reports/research/X2/.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import e1_common as E  # noqa: E402
import x_common as X  # noqa: E402

EXPERIMENT, PREFIX = "X2", "x2"
PROCEDURES = ("P0", "I8", "D", "C-split", "B", "A", "A-bonf")
N6 = ("With every query the pool can supply ({n} queries), P0 did not reach non-trivial deployment at "
      "T = 0.97 on store {store}.")
N7 = ("Even {top:,} queries drawn from the pool's own {pool} queries did not produce non-trivial deployment "
      "at T = 0.97 on store {store}. Within the registered range, more queries did not remove the abstention.")
#: the cells that decide H3 at the pool-capped budget: (name, threshold, kind of deployment)
H3_CELLS = (("primary", X.H3_THRESHOLD, "nontrivial"),
            ("secondary_T0.99", X.H3_SECONDARY_THRESHOLD, "nontrivial"),
            ("secondary_substantial", X.H3_THRESHOLD, "substantial"),
            ("secondary_T0.99_substantial", X.H3_SECONDARY_THRESHOLD, "substantial"))


def prepare(name: str, reg: X.Registry) -> dict:
    """Gates G2 and G4 for one store. G1, the file hashes, is checked before any file is read.

    G4 as registered is Method C under labels at the E1 budgets. A, A-bonf, B, C-split and D
    are registered there "exactly as in E1", so their stored selections are compared too.
    """
    spec = reg.stores[name]
    d = X.load_store(name, reg)
    gates = {"G2": X.require("G2", X.gate_values(d), name)}
    stores = {c: E.Store(d, X.CONSTRUCT_REFERENCE[c]) for c in X.CONSTRUCTS}
    prob = {c: X.problem_for(stores[c]) for c in X.CONSTRUCTS}
    idx, _ = X.sample_indices(spec.n_queries, range(reg.n_seeds), reg.e1_budgets, reg)
    labels_e1 = X.simulate(stores["labels"].pq, stores["labels"].base, prob["labels"], idx, X.PRIMARY_THRESHOLDS,
                           i8_index=X.first_position(d["meta"], X.INT8_PIPELINE),
                           ab_seeds={n: reg.ab_seed_count(n, spec.n_queries) for n in reg.e1_budgets},
                           a_bonf_budgets=reg.e1_budgets)
    gates["G4"] = X.require("G4", X.gate_reproduction(name, labels_e1, reg, [f"C@{delta}" for delta in X.DELTAS]),
                            name)
    gates["E1_comparators"] = X.require("G4 (E1 comparators)",
                                        X.gate_reproduction(name, labels_e1, reg, X.E1_METHODS), name)
    return {"d": d, "stores": stores, "prob": prob, "gates": gates, "labels_e1": labels_e1}


def analyse(name: str, prep: dict, reg: X.Registry, validation: dict) -> tuple:
    """One store: every registered cell (SIMULATED) and the full-pool decision (MEASURED)."""
    spec, d, st, prob = reg.stores[name], prep["d"], prep["stores"], prep["prob"]
    n_pool = spec.n_queries
    budgets, cap = reg.budgets(n_pool), reg.pool_capped_budget(n_pool)
    f16, i8 = X.first_position(d["meta"], X.FLOAT16_PIPELINE), X.first_position(d["meta"], X.INT8_PIPELINE)
    labels = [c["label"] for c in d["meta"]["candidates"]]
    idx, nested = X.sample_indices(n_pool, range(reg.n_seeds), budgets, reg)
    ab = {n: reg.ab_seed_count(n, n_pool) for n in budgets}

    sel = {}
    for c in X.CONSTRUCTS:
        done = prep["labels_e1"] if c == "labels" else {}
        todo = {n: idx[n] for n in budgets if not (c == "labels" and n in reg.e1_budgets)}
        sel[c] = {**done, **X.simulate(st[c].pq, st[c].base, prob[c], todo, X.THRESHOLDS[c], i8_index=i8,
                                       ab_seeds=ab, a_bonf_budgets=reg.e1_budgets)}
    rep = {c: X.Replication(st[c].pq, st[c].base, prob[c], n_pool, reg, i8) for c in X.CONSTRUCTS}
    monotone = {c: X.check_monotone(sel[c], prob[c], budgets, X.THRESHOLDS[c], nested, f"{name}, {c}")
                for c in X.CONSTRUCTS}
    rows = [r for c in X.CONSTRUCTS
            for r in X.simulated_rows(c, X.THRESHOLDS[c], sel[c], prob[c], rep[c], budgets, n_pool, f16, name,
                                      PROCEDURES)]

    full = []  # the only observed budget: the whole pool once, in stored order
    for c in X.CONSTRUCTS:
        for target in X.THRESHOLDS[c]:
            for delta in X.DELTAS:
                k = X.full_pool_decision(st[c].pq, st[c].base, prob[c], target, delta)
                full.append({"construct": c, "threshold": target, "delta": delta, "evidence": X.MEASURED,
                             "queries": n_pool, "procedure": "P0", "position": k,
                             "label": None if k is None else labels[k],
                             "compression": X.oracle_compression(prob[c], k),
                             "nontrivial": bool(k is not None and k != f16),
                             "substantial": bool(k is not None
                                                 and prob[c].compression[k] >= X.SUBSTANTIAL_COMPRESSION)})

    def decide(c, target, delta, n, kind):
        method = f"C@{delta}"
        return X.decide_rate(
            X.deployed_flags(sel[c][X.sel_key(n, target, method)], prob[c], target, f16)[kind],
            lambda: X.deployed_flags(rep[c].get(n, target, method), prob[c], target, f16)[kind])

    h3 = {label: {**X.cell_identity("labels", target, X.PRIMARY_DELTA, cap, n_pool, "P0"), "deployment": kind,
                  **decide("labels", target, X.PRIMARY_DELTA, cap, kind)}
          for label, target, kind in H3_CELLS}
    n_star = [{"construct": c, "threshold": target, "delta": delta, "deployment": kind, "procedure": "P0",
               **X.n_star(budgets, n_pool, lambda n, c=c, target=target, delta=delta, kind=kind:
                          decide(c, target, delta, n, kind))}
              for c in X.CONSTRUCTS for target in X.THRESHOLDS[c] for delta in X.DELTAS
              for kind in ("nontrivial", "substantial")]

    checked = [r for r in rows if "validity" in r]
    result = {
        "header": X.header(EXPERIMENT, reg, validation),
        "store": {"name": name, "encoder": spec.encoder, "pool": spec.pool, "pool_queries": n_pool,
                  "documents": spec.n_docs, "dim": spec.dim, "metric": spec.metric},
        "constructs": X.CONSTRUCT_NOTE,
        "budgets": X.budget_table(n_pool, reg), "pool_capped_budget": cap,
        "gates": {"G1": {"passed": True, "files": validation["stores"][name]}, **prep["gates"]},
        "sampling": {"seeds": reg.n_seeds, "nested_across_budgets": nested,
                     "rule_A_and_B_seeds": {str(n): v for n, v in ab.items()}},
        "cells": {"evidence": "SIMULATED: resamples of a MEASURED pool; each row carries its budget class",
                  "truth": "pool truth of the row's own construct", "rows": rows},
        "full_pool": {"evidence": X.MEASURED,
                      "note": "P0 given the whole pool once, each query once, in stored order. A single "
                              "realisation with no truth to be checked against.", "rows": full},
        "n_star": {"evidence": "SIMULATED; each row carries the class of the budget that decides it",
                   "rows": n_star},
        "H3": h3,
        "checks": {"monotonicity": monotone,
                   "validity": {"procedures": [p for p in PROCEDURES if X.PROCEDURES[p][2]],
                                "cells_checked": len(checked),
                                "cells_replicated": sum(r["validity"]["replicated"] for r in checked),
                                "confirmed_failures": 0}},
        "replicated_cells": {c: sorted(rep[c].selections) for c in X.CONSTRUCTS},
    }
    arrays = {f"{c[0].upper()}_{k}": v for c in X.CONSTRUCTS for k, v in sel[c].items()}
    arrays.update({f"rep_{c[0].upper()}_{k}": v for c in X.CONSTRUCTS for k, v in rep[c].selections.items()})
    return result, arrays


def summarise(stores: dict, reg: X.Registry) -> dict:
    """H3, its extension beyond the pools, and the negative-result sentences of section 19."""
    def at_cap(label):
        cells = {n: s["H3"][label] for n, s in stores.items()}
        failing = [n for n, c in cells.items() if not c["at_least_0.90"]]
        return {"cells": cells, "by_pool": X.worst_by_pool(cells, reg), "failing_stores": failing,
                "verdict": "supported" if not failing else "fails"}

    def extension(target, kind):
        rows = {n: next(r for r in s["n_star"]["rows"] if r["construct"] == "labels" and r["threshold"] == target
                        and r["delta"] == X.PRIMARY_DELTA and r["deployment"] == kind)
                for n, s in stores.items()}
        without = [n for n, r in rows.items() if r["n_star"] is None]
        return {"n_star": {n: {k: r[k] for k in ("n_star", "decided_by_budget", "budget_class", "evidence")}
                           for n, r in rows.items()},
                "stores_without_n_star": without,
                "verdict": "holds in simulation" if not without else "fails in simulation"}

    primary, ext = at_cap("primary"), extension(X.H3_THRESHOLD, "nontrivial")
    top = max(reg.common_budgets)
    sentences = [N6.format(n=primary["cells"][n]["n"], store=n) for n in primary["failing_stores"]]
    sentences += [N7.format(top=top, pool=stores[n]["store"]["pool_queries"], store=n)
                  for n in ext["stores_without_n_star"]]
    return {
        "hypotheses": {"H3": {
            "primary": {"evidence": X.SIM_POOL, **primary},
            "extension_beyond_the_pools": {
                "evidence": "SIMULATED; extrapolated wherever the deciding budget exceeds the pool",
                "reading": "a statement about samples drawn repeatedly from the pool's own N queries; never "
                           "evidence of what that many independent queries would give", **ext},
            "secondary": {label: at_cap(label) for label, _, _ in H3_CELLS[1:]},
            "secondary_extension": {"T0.99": extension(X.H3_SECONDARY_THRESHOLD, "nontrivial"),
                                    "substantial": extension(X.H3_THRESHOLD, "substantial")}}},
        "negative_result_sentences": sentences,
        "validity": {"cells_checked": sum(s["checks"]["validity"]["cells_checked"] for s in stores.values()),
                     "cells_replicated": sum(s["checks"]["validity"]["cells_replicated"] for s in stores.values()),
                     "confirmed_failures": 0},
        "scope": "development encoders only; three pools; every simulated result is about the pool's own "
                 "empirical distribution (section 23)",
    }


def report(summary: dict, stores: dict) -> str:
    h3 = summary["hypotheses"]["H3"]
    out = [
        "# X2: query budget and certification", "",
        f"Specification: `{X.PREREG_PATH}` section 11, commit `{X.PREREG_COMMIT}`. "
        f"Code commit `{summary['header']['code_commit']}`. Written by `scripts/research/x2_budget.py`; "
        "every number is in the JSON files beside this report.", "",
        "**How to read the budgets.** Every rate is SIMULATED: resamples of one MEASURED pool. A "
        "pool-supported budget does not exceed the pool size. An extrapolated budget does: it says what P0 "
        "would do if queries were drawn from this pool's own N queries, and it is not evidence of what that "
        "many independent queries would give. The only MEASURED outcome is the full-pool decision.", "",
        "## Summary", "",
        f"- **H3 (primary; SIMULATED, pool-supported): {h3['primary']['verdict']}.** Non-trivial deployment "
        "rate of P0 under labels at T = 0.97, delta = 0.05, at the pool-capped budget, required to be at "
        "least 0.90 on all seven stores.",
        f"- **Extension beyond the pools (secondary; simulation only): "
        f"{h3['extension_beyond_the_pools']['verdict']}.**",
        f"- Validity: {summary['validity']['cells_checked']} cells checked against their error level, "
        f"{summary['validity']['cells_replicated']} re-run with seeds 2000-3999, "
        f"{summary['validity']['confirmed_failures']} confirmed failures.",
    ]
    out += [f"- {s}" for s in summary["negative_result_sentences"]]
    out += ["", "## Gates and checks", "",
            X.md_table(["store", "G1 hashes", "G2 values", "G4 reproduces E1 (Method C)", "E1 comparators reproduced",
                        "samples nested", "monotonicity violations (labels / agreement)"],
                       [[n, s["gates"]["G1"]["passed"], s["gates"]["G2"]["passed"], s["gates"]["G4"]["passed"],
                         s["gates"]["E1_comparators"]["passed"], s["sampling"]["nested_across_budgets"],
                         " / ".join(str(s["checks"]["monotonicity"][c].get("violations", "not checked"))
                                    for c in X.CONSTRUCTS)] for n, s in stores.items()]),
            "", "## H3. Non-trivial deployment at the pool-capped budget (SIMULATED, pool-supported)", "",
            "P0 under labels, T = 0.97, delta = 0.05.", "",
            X.md_table(["store", "pool", "n", "rate [95% interval]", "seeds deciding", "borderline rule used",
                        "at least 0.90"],
                       [[n, stores[n]["store"]["pool_queries"], c["n"], X.fmt_rate(c), c.get("pooled_seeds", c["seeds"]),
                         c["borderline_rule_used"], c["at_least_0.90"]] for n, c in h3["primary"]["cells"].items()]),
            "", "Each pool by its worst store: "
            + "; ".join(f"{pool}: {w['worst_store']} ({w['deciding_rate']:.3f})"
                        for pool, w in h3["primary"]["by_pool"].items()) + ".",
            "", "## n*, the smallest registered budget from which the rate stays at or above 0.90", "",
            "P0, delta = 0.05, non-trivial deployment. Each entry carries the class of the budget that decides "
            "it; an extrapolated entry is a statement about redrawing the pool's own queries.", "",
            X.md_table(["store", "construct", "threshold", "n*", "decided at", "class"],
                       [[n, r["construct"], r["threshold"], r["n_star"] if r["n_star"] is not None else "none",
                         r["decided_by_budget"], r["budget_class"]]
                        for n, s in stores.items() for r in s["n_star"]["rows"]
                        if r["delta"] == X.PRIMARY_DELTA and r["deployment"] == "nontrivial"])]
    borderline = [[n, r["construct"], r["threshold"], r["delta"], r["deployment"], budget, f"{c['rate']:.3f}",
                   f"{c['pooled_rate']:.3f}", c["at_least_0.90"]]
                  for n, s in stores.items() for r in s["n_star"]["rows"]
                  for budget, c in r["deciding_cells"].items() if c["borderline_rule_used"]]
    borderline += [[n, c["construct"], c["threshold"], c["delta"], c["deployment"], c["n"], f"{c['rate']:.3f}",
                    f"{c['pooled_rate']:.3f}", c["at_least_0.90"]]
                   for n, s in stores.items() for c in s["H3"].values() if c["borderline_rule_used"]]
    borderline = list({tuple(str(v) for v in row): row for row in borderline}.values())  # an H3 cell can decide an n* too
    out += ["", "## Cells that used the borderline rule", "",
            "A rate whose 95% interval contained 0.90 was re-run with seeds 2000-3999; the pooled rate decides.", "",
            X.md_table(["store", "construct", "threshold", "delta", "deployment", "n", "rate, 2,000 seeds",
                        "rate, 4,000 seeds", "at least 0.90"], borderline) if borderline else "None."]
    for c in X.CONSTRUCTS:
        for target in X.THRESHOLDS[c]:
            def cell(s, n, c=c, target=target):
                row = next(r for r in s["cells"]["rows"]
                           if r["construct"] == c and r["threshold"] == target and r["delta"] == X.PRIMARY_DELTA
                           and r["n"] == n and r["procedure"] == "P0")
                return f"{row['nontrivial']['rate']:.3f}"
            out += ["", f"## Non-trivial deployment rate of P0 by budget: {c}, threshold {target}, delta = 0.05 "
                    "(SIMULATED)", "", X.rate_by_budget_table(stores, cell)]
    out += ["", "## Comparators at the pool-capped budget, labels, delta = 0.05 (SIMULATED, pool-supported)", "",
            "Each cell: non-trivial deployment rate / median compression / miss rate. Rule A carries no error "
            "level; A and B use seeds 0-499 at this budget where it is not an E1 budget.", "",
            X.md_table(["store", "threshold", *PROCEDURES[:-1]],
                       [[n, t] + [next((f"{r['nontrivial']['rate']:.2f} / {r['compression_median']:.1f} / "
                                        f"{r['miss']['rate']:.3f}" for r in s["cells"]["rows"]
                                        if r["construct"] == "labels" and r["threshold"] == t
                                        and r["n"] == s["pool_capped_budget"] and r["procedure"] == p
                                        and r["delta"] in (X.PRIMARY_DELTA, None)), None) for p in PROCEDURES[:-1]]
                        for n, s in stores.items() for t in X.PRIMARY_THRESHOLDS]),
            "", "## The full-pool decision (MEASURED)", "",
            "P0 given the whole pool once, in stored order, delta = 0.05. One realisation per row; it has no "
            "truth to be checked against.", "",
            X.md_table(["store", "construct", "threshold", "queries", "deployed candidate", "compression"],
                       [[n, r["construct"], r["threshold"], r["queries"], r["label"] or "none (abstains)",
                         r["compression"]] for n, s in stores.items() for r in s["full_pool"]["rows"]
                        if r["delta"] == X.PRIMARY_DELTA]), ""]
    return "\n".join(out)


def run(reg: X.Registry = X.REGISTERED, stores=None) -> dict:
    X.run_store_units(EXPERIMENT, reg, stores or list(reg.stores), prepare, analyse, PREFIX)
    return X.finish_experiment(EXPERIMENT, reg, PREFIX, summarise, report)


def dry_run(reg: X.Registry = X.REGISTERED) -> dict:
    rep = X.dry_run_common(EXPERIMENT, reg)
    rep["procedures"] = {
        "P0, I8, D, C-split": "every registered budget, seeds 0-1999",
        "A, A-bonf, B": f"budgets {list(reg.e1_budgets)} exactly as in E1, seeds 0-1999",
        "A, B": f"the pool-capped budget and {reg.ab_extra_budget}, seeds 0-{reg.ab_extra_seeds - 1}",
    }
    rep["analyses"] = ["gates G1, G2, G4", "outcomes per cell (section 11.4)", "monotonicity and validity checks",
                       "the MEASURED full-pool decision", "H3 at the pool-capped budget", "n* per cell with its class",
                       "N6, N7, N13"]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="X2: query budget and certification (preregistered).")
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
