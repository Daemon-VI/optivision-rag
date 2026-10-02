"""Render the E1.1 tables in reports/research/E1/RESULTS_TABLES.md from the JSON results."""

import glob
import json
import pathlib

E1 = pathlib.Path(__file__).resolve().parents[2] / "reports" / "research" / "E1"
REAL = ["colpali_docvqa", "colpali_infovqa", "colqwen2_docvqa", "colqwen2_infovqa"]
NAMES = {"colpali_docvqa": "ColPali · DocVQA", "colpali_infovqa": "ColPali · InfoVQA",
         "colqwen2_docvqa": "ColQwen2 · DocVQA", "colqwen2_infovqa": "ColQwen2 · InfoVQA"}
MAIN = ["A", "B@0.05", "C@0.05", "C-split@0.05", "D@0.05"]
out = []


def pct(x):
    return f"{100 * x:.1f}%"


def cell(r):
    return f"{r['compression_median']:.1f}x · {pct(r['deployed_rate'])} · {pct(r['miss_rate'])}"


real = {n: json.loads((E1 / "real" / f"{n}.json").read_text(encoding="utf-8")) for n in REAL}
for n_q in (225, 100, 50):
    out.append(f"\n### Real data, n = {n_q} calibration queries (SIMULATED resampling of MEASURED per-query results)\n")
    out.append("Each cell: median selected compression (float32 = 1x when nothing is deployed) · "
               "share of resamples that deploy a compressed candidate · miss rate (deployed and finite "
               "query-pool truth retention < T). 2,000 resamples per cell.\n")
    out.append("| dataset · T | oracle | " + " | ".join(MAIN) + " |")
    out.append("|---|---|" + "---|" * len(MAIN))
    for n in REAL:
        for T in (0.99, 0.97, 0.95):
            rows = {r["method"]: r for r in real[n]["rows"] if r["n"] == n_q and r["target"] == T}
            out.append(f"| {NAMES[n]} · {T} | {rows['A']['oracle_compression']:.1f}x | "
                       + " | ".join(cell(rows[m]) for m in MAIN) + " |")

out.append("\n### Real data: miss rates of the current rule A (finite query-pool truth)\n")
out.append("| dataset | T | n=50 | n=100 | n=225 | worst miss (any n) | A's held-out flag said 'met' while truth missed (n=50) |")
out.append("|---|---|---|---|---|---|---|")
for n in REAL:
    for T in (0.99, 0.97, 0.95):
        rows = {r["n"]: r for r in real[n]["rows"] if r["method"] == "A" and r["target"] == T}
        w = [rows[k]["worst_miss_retention"] for k in rows if rows[k]["worst_miss_retention"] is not None]
        lo = lambda r: f"{pct(r['miss_rate'])} [{pct(r['miss_rate_ci95'][0])}, {pct(r['miss_rate_ci95'][1])}]"
        out.append(f"| {NAMES[n]} | {T} | {lo(rows[50])} | {lo(rows[100])} | {lo(rows[225])} | "
                   f"{pct(min(w)) if w else '—'} | {pct(rows[50]['flag_true_but_truth_miss'])} |")

out.append("\n### Real data: δ = 0.01 and A-bonf at n = 225\n")
out.append("| dataset · T | A-bonf | B@0.01 | C@0.01 | C-split@0.01 | D@0.01 |")
out.append("|---|---|---|---|---|---|")
for n in REAL:
    for T in (0.99, 0.97, 0.95):
        rows = {r["method"]: r for r in real[n]["rows"] if r["n"] == 225 and r["target"] == T}
        out.append(f"| {NAMES[n]} · {T} | " + " | ".join(cell(rows[m]) for m in
                   ("A-bonf", "B@0.01", "C@0.01", "C-split@0.01", "D@0.01")) + " |")

viol = [(n, r["n"], r["target"], r["method"]) for n in REAL for r in real[n]["rows"]
        if r.get("consistent_with_claim") is False]
out.append(f"\nFormal methods whose observed miss rate contradicts their claimed δ (real data): {viol or 'none'}.")

out.append("\n### Synthetic stress test, T = 0.95 (SIMULATED; truth known exactly)\n")
out.append("Each cell: miss rate at n = 50 / 100 / 225. Formal methods claim ≤ 5%.\n")
out.append("| K · scenario · dependence | oracle | A | A-bonf | B@0.05 | C@0.05 | C-split@0.05 | D@0.05 | C@0.05 median x at n=225 |")
out.append("|---|---|---|---|---|---|---|---|---|")
sviol = []
for f in sorted(glob.glob(str(E1 / "synthetic" / "*.json")), key=lambda p: (int(pathlib.Path(p).stem.split("_")[0][1:]), p)):
    d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
    rows = {(r["method"], r["n"]): r for r in d["rows"]}
    mr = lambda m, rows=rows: " / ".join(pct(rows[(m, k)]["miss_rate"]) for k in (50, 100, 225))
    out.append(f"| {d['K']} · {d['scenario']} · {d['dependence']} | {rows[('A', 225)]['oracle_compression']:.1f}x | "
               + " | ".join(mr(m) for m in ("A", "A-bonf", "B@0.05", "C@0.05", "C-split@0.05", "D@0.05"))
               + f" | {rows[('C@0.05', 225)]['compression_median']:.1f}x |")
    sviol += [(pathlib.Path(f).stem, r["n"], r["method"]) for r in d["rows"] if r.get("consistent_with_claim") is False]
out.append(f"\nFormal methods whose observed miss rate contradicts their claimed δ (synthetic): {sviol or 'none'}.")
CONF = {"colqwen25_docvqa": "ColQwen2.5 · DocVQA", "colqwen25_infovqa": "ColQwen2.5 · InfoVQA",
        "scifact": "SciFact (nDCG@10)"}
if all((E1 / "confirm" / f"{n}.json").exists() for n in CONF):
    out.append("\n### E1.2 confirmation on held-back datasets (frozen C at δ = 0.05; "
               "SIMULATED resampling of MEASURED per-query results)\n")
    out.append("Cells as above. Criterion column: C's 95% Clopper–Pearson miss interval and "
               "whether its lower end is ≤ 0.05.\n")
    out.append("| dataset · T · n | oracle | A | B@0.05 | C@0.05 | D@0.05 | C miss interval · consistent |")
    out.append("|---|---|---|---|---|---|---|")
    for n, label in CONF.items():
        d = json.loads((E1 / "confirm" / f"{n}.json").read_text(encoding="utf-8"))
        for T in (0.99, 0.97, 0.95):
            for n_q in (50, 100, 225):
                rows = {r["method"]: r for r in d["rows"] if r["n"] == n_q and r["target"] == T}
                c = rows["C@0.05"]
                out.append(f"| {label} · {T} · {n_q} | {rows['A']['oracle_compression']:.1f}x | "
                           + " | ".join(cell(rows[m]) for m in ("A", "B@0.05", "C@0.05", "D@0.05"))
                           + f" | [{pct(c['miss_rate_ci95'][0])}, {pct(c['miss_rate_ci95'][1])}] · {c['consistent_with_claim']} |")
(E1 / "RESULTS_TABLES.md").write_text("# E1.1 tables (generated by scripts/research/e1_tables.py)\n" + "\n".join(out) + "\n",
                                       encoding="utf-8")
print("\n".join(out[:40]))
