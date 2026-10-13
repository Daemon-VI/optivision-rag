"""E1.1: selection and certification among the 40 default candidates.

Real data (development datasets only): ``python e1_1_study.py real <store>``
Synthetic stress test:                ``python e1_1_study.py synth <K> <scenario> <dependence>``

Resampling (real data). The query pool of N queries is fixed. For seed s in
0..1999, a calibration sample of n queries is drawn i.i.d. WITH replacement:
idx = np.random.default_rng(s).integers(0, N, size=n). The same seed gives nested
samples across n (the n=50 sample is the first 50 of the n=225 one). Sampling
with replacement makes the pool's empirical distribution Q_pool the sampling
distribution, so R_Qpool(k) = sum(c_k) / sum(b) over the pool (the "finite
query-pool truth") is the exact quantity every formal statement below refers to.
The 2,000 samples are not independent datasets: they are resamples of one pool.

Methods (n = total labelled queries available to the procedure):
  A        calibrate() as implemented (seed=0): selection on the half
           split_queries(n, 0.5, 0) of the sample, one-sided 0.999 bootstrap bound
           (n_boot=1000), family walk with early stop, smallest size wins; the
           other half is only reported (no abstention).
  A-bonf   the same with multiplicity="bonferroni" (per-candidate 1 - 0.001/40).
  B        A's selection on the first half, then certify ONLY that candidate on
           the other half with the exact betting p-value at delta; deploy it if
           p <= delta, else fall back to float32 (abstain).
  C        all n queries; exact betting p-values; within each family, fixed-
           sequence testing in the search space's own order (least to most
           aggressive) at delta / F (F = number of families); deploy the smallest
           rejected candidate, else float32.
  C-split  order candidates by point retention on the first half (descending; ties
           by larger size first), fixed-sequence test on the second half at delta.
  D        Bonferroni: all n queries, reject H0_k if p_k <= delta / K; smallest
           rejected candidate, else float32.
A miss is: a compressed candidate is deployed and its truth retention is < T.
"""

from __future__ import annotations

import json
import os
import sys
import time
from statistics import NormalDist

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e1_common as E
import e1_stats as S

from optivision.evaluation import retention, retention_lower_bound, split_queries

SEEDS = 2000
NS = (50, 100, 225)
TARGETS = (0.99, 0.97, 0.95)
DELTAS = (0.05, 0.01)
A_CONF = 0.999
A_BONF_CONF = 1.0 - (1.0 - 0.999) / 40
METHODS = ["A", "A-bonf"] + [f"{m}@{d}" for d in DELTAS for m in ("B", "C", "C-split", "D")]


class Problem:
    """Candidate metadata + truth, independent of any sample."""

    def __init__(self, families, bytes_, vectors, compression, truth):
        self.families = list(families)
        self.bytes = np.asarray(bytes_, float)
        self.vectors = np.asarray(vectors, float)
        self.compression = np.asarray(compression, float)
        self.truth = np.asarray(truth, float)
        self.K = len(self.families)
        self.F = len(dict.fromkeys(self.families))


def a_bounds(pq_cal, base_cal, conf, mode):
    if mode == "library":
        return np.array([retention_lower_bound(pq_cal[k], base_cal, conf, n_boot=1000, seed=0)
                         for k in range(pq_cal.shape[0])])
    return S.lower_bound_all(pq_cal, base_cal, conf, n_boot=1000, seed=0)


def run_cell(prob: Problem, draw, n: int, a_mode: str) -> dict:
    """draw(seed) -> (pq (K, n), base (n)). Returns per-seed selections for every T and method."""
    cal, hold = split_queries(n, 0.5, 0)
    sel = {T: {m: np.full(SEEDS, -1, np.int16) for m in METHODS} for T in TARGETS}
    flag = {T: np.zeros(SEEDS, bool) for T in TARGETS}
    secs = {m: 0.0 for m in METHODS}
    samples = [draw(s) for s in range(SEEDS)]
    pq_all = np.stack([p for p, _ in samples])  # (S, K, n)
    b_all = np.stack([b for _, b in samples])  # (S, n)
    for s in range(SEEDS):
        pq, b = pq_all[s], b_all[s]
        t0 = time.perf_counter()
        if a_mode == "vector":
            # one bootstrap serves both levels: above 0.99 the bound is point - z * sd(ratios)
            point, ratios = S.lower_bound_stats(pq[:, cal], b[cal])
            sd = np.nanstd(ratios, axis=1)
            bnd = point - NormalDist().inv_cdf(A_CONF) * sd
            t1 = time.perf_counter()
            bnd_bonf = point - NormalDist().inv_cdf(A_BONF_CONF) * sd
        else:
            bnd = a_bounds(pq[:, cal], b[cal], A_CONF, a_mode)
            t1 = time.perf_counter()
            bnd_bonf = a_bounds(pq[:, cal], b[cal], A_BONF_CONF, a_mode)
        t2 = time.perf_counter()
        secs["A"] += t1 - t0
        secs["A-bonf"] += t2 - t1
        for T in TARGETS:
            k = S.walk_select(bnd, prob.families, prob.bytes, prob.vectors, T)
            sel[T]["A"][s] = -1 if k is None else k
            if k is not None:
                flag[T][s] = retention(pq[k, hold], b[hold], n_boot=0)[0] >= T
            kb = S.walk_select(bnd_bonf, prob.families, prob.bytes, prob.vectors, T)
            sel[T]["A-bonf"][s] = -1 if kb is None else kb
    for T in TARGETS:
        m0 = T / (1 + T)
        t0 = time.perf_counter()
        p_full = S.betting_pvalues(S.to_y(pq_all, b_all[:, None, :], T), m0)  # (S, K)
        t1 = time.perf_counter()
        p_hold = S.betting_pvalues(S.to_y(pq_all[:, :, hold], b_all[:, None, hold], T), m0)
        t2 = time.perf_counter()
        point_cal = pq_all[:, :, cal].mean(axis=2) / b_all[:, cal].mean(axis=1)[:, None]
        for d in DELTAS:
            for s in range(SEEDS):
                ka = sel[T]["A"][s]
                sel[T][f"B@{d}"][s] = ka if ka >= 0 and p_hold[s, ka] <= d else -1
                rej = S.fixed_sequence_by_family(p_full[s], prob.families, d / prob.F)
                k = S.smallest(rej, prob.bytes, prob.vectors)
                sel[T][f"C@{d}"][s] = -1 if k is None else k
                order = np.lexsort((np.arange(prob.K), -prob.bytes, -point_cal[s]))
                rej = S.fixed_sequence(p_hold[s], order, d)
                k = S.smallest(rej, prob.bytes, prob.vectors)
                sel[T][f"C-split@{d}"][s] = -1 if k is None else k
                rej = [int(j) for j in np.flatnonzero(p_full[s] <= d / prob.K)]
                k = S.smallest(rej, prob.bytes, prob.vectors)
                sel[T][f"D@{d}"][s] = -1 if k is None else k
            secs[f"C@{d}"] += t1 - t0
            secs[f"D@{d}"] += t1 - t0
            secs[f"B@{d}"] += (t2 - t1) / prob.K  # one candidate tested
            secs[f"C-split@{d}"] += t2 - t1
    return {"sel": sel, "flag": flag, "seconds_per_selection": {m: v / SEEDS for m, v in secs.items()},
            "cal_size": len(cal), "hold_size": len(hold)}


def clopper_pearson(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def summarise(prob: Problem, res: dict, n: int) -> list[dict]:
    rows = []
    for T in TARGETS:
        feas = np.flatnonzero(prob.truth >= T)
        oracle = None if feas.size == 0 else S.smallest([int(j) for j in feas], prob.bytes, prob.vectors)
        ox = 1.0 if oracle is None else float(prob.compression[oracle])
        for m in METHODS:
            k = res["sel"][T][m]
            dep = k >= 0
            comp = np.where(dep, prob.compression[np.maximum(k, 0)], 1.0)
            r = np.where(dep, prob.truth[np.maximum(k, 0)], 1.0)
            miss = dep & (r < T)
            nm = int(miss.sum())
            delta = float(m.split("@")[1]) if "@" in m else None
            row = {
                "n": n, "target": T, "method": m, "delta": delta,
                "deployed_rate": float(dep.mean()), "abstain_rate": float(1 - dep.mean()),
                "compression_median": float(np.median(comp)), "compression_mean_log2": float(np.mean(np.log2(comp))),
                "compression_p10_p90": [float(np.quantile(comp, .1)), float(np.quantile(comp, .9))],
                "oracle_label_index": oracle, "oracle_compression": ox,
                "median_compression_over_oracle": float(np.median(comp / ox)),
                "miss_rate": nm / SEEDS, "miss_rate_ci95": clopper_pearson(nm, SEEDS),
                "worst_miss_retention": float(r[miss].min()) if nm else None,
                "truth_retention_mean_deployed": float(r[dep].mean()) if dep.any() else None,
                "excess_over_target_median_deployed": float(np.median(r[dep] - T)) if dep.any() else None,
                "success_rate": float((dep & (r >= T)).mean()),
                "most_common_selection": [int(x) for x in np.bincount(k + 1).argsort()[::-1][:3] - 1],
                "seconds_per_selection": res["seconds_per_selection"][m],
            }
            if delta is not None:
                row["claimed_max_miss_rate"] = delta
                row["consistent_with_claim"] = bool(row["miss_rate_ci95"][0] <= delta)
            if m == "A":
                f = res["flag"][T]
                row["holdout_flag_rate"] = float(f.mean())
                row["flag_true_but_truth_miss"] = float((f & miss).mean())
            rows.append(row)
    return rows


# ------------------------------------------------------------------ real data

def real(name: str, out_dir: str = "real") -> None:
    if out_dir == "real" and name not in E.DATASETS:
        raise SystemExit(f"{name} is not a development dataset (confirmation sets are held back)")
    if out_dir == "confirm" and (name not in E.CONFIRMATION or not E.FREEZE.exists()):
        raise SystemExit("confirmation runs need a confirmation dataset and the E1.2 freeze record")
    d = E.load_store(name)
    st = E.Store(d, "labels")
    if not (np.isfinite(st.pq).all() and np.isfinite(st.base).all()):
        raise SystemExit("non-finite per-query values")
    N = st.pq.shape[1]
    truth = st.pq.sum(axis=1) / st.base.sum()
    prob = Problem(st.families, st.bytes, st.vectors, st.compression, truth)
    out = {"experiment": "E1.1", "kind": "SIMULATED resampling of MEASURED per-query results",
           "dataset": d["meta"]["dataset"], "store": name, "pool_queries": N, "K": prob.K, "F": prob.F,
           "baseline_mean": float(st.base.mean()),
           "truth_note": "finite query-pool truth: retention over all pool queries",
           "candidates": [{"index": k, "label": st.labels[k], "family": st.families[k],
                           "compression": float(st.compression[k]), "pool_retention": float(truth[k])}
                          for k in range(prob.K)],
           "environment": E.environment(), "rows": [], "selections_file": f"{name}_selections.npz"}
    sels = {}
    for n in NS:
        t0 = time.time()
        def draw(seed, n=n):
            idx = np.random.default_rng(seed).integers(0, N, size=n)
            return st.pq[:, idx], st.base[idx]

        res = run_cell(prob, draw, n, "library")
        out["rows"] += summarise(prob, res, n)
        for T in TARGETS:
            for m in METHODS:
                sels[f"n{n}_T{T}_{m}"] = res["sel"][T][m]
        print(f"{name} n={n} done in {time.time() - t0:.0f}s", flush=True)
    (E.E1 / out_dir).mkdir(parents=True, exist_ok=True)
    (E.E1 / out_dir / f"{name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    np.savez_compressed(E.E1 / out_dir / f"{name}_selections.npz", **sels)


# ------------------------------------------------------------------ synthetic

NDCG_VALUES = np.array([1.0, 1 / np.log2(3), 0.5, 1 / np.log2(5), 1 / np.log2(6), 0.0])  # rank 1..5, miss
NDCG_PROBS = np.array([0.50, 0.08, 0.05, 0.03, 0.02, 0.32])  # mean 0.596, near the DocVQA baselines
DOWN = np.array([1, 2, 3, 4, 5, 5])  # index of the value one rank lower (rank 5 -> miss; miss stays)
T_SYN = 0.95
GEN_SEED = 2026
SCENARIOS = ("separated", "near_tied", "many_below", "one_above", "several_above")


def synth_problem(K: int, scenario: str):
    """Candidates: a lossless floor (position 0, 2x) + K-1 in families of 5.

    Family f (1-based), step j (1..5): compression (1 + 0.1 f) * 2**j. Odd families
    lose whole queries (c = 0, "total"); even families move the relevant page one
    rank down ("down"). Truth retention by scenario (T = 0.95):
      separated     R = 1 - 0.015 j               (1.5-point steps)
      near_tied     R = T + U(-0.005, 0.005)
      many_below    R = T - U(0.005, 0.010)
      one_above     R = T - 0.02, except one random non-floor candidate at T + 0.01
      several_above 25% of non-floor at T + U(0.005, 0.03), the rest T - U(0.005, 0.03)
    Random draws use np.random.default_rng(GEN_SEED + K).
    """
    rng = np.random.default_rng(GEN_SEED + K)
    fams, comp, mode, steps = ["floor"], [2.0], ["none"], [0]
    for i in range(K - 1):
        f, j = i // 5 + 1, i % 5 + 1
        fams.append(f"family{f}")
        comp.append((1 + 0.1 * f) * 2 ** j)
        mode.append("total" if f % 2 else "down")
        steps.append(j)
    m = K - 1
    if scenario == "separated":
        R = np.array([1.0 - 0.015 * j for j in steps[1:]])
    elif scenario == "near_tied":
        R = T_SYN + rng.uniform(-0.005, 0.005, m)
    elif scenario == "many_below":
        R = T_SYN - rng.uniform(0.005, 0.010, m)
    elif scenario == "one_above":
        R = np.full(m, T_SYN - 0.02)
        R[rng.integers(0, m)] = T_SYN + 0.01
    elif scenario == "several_above":
        R = T_SYN - rng.uniform(0.005, 0.03, m)
        up = rng.choice(m, size=max(1, m // 4), replace=False)
        R[up] = T_SYN + rng.uniform(0.005, 0.03, up.size)
    else:
        raise ValueError(scenario)
    R = np.concatenate([[1.0], R])
    eb = float(NDCG_VALUES @ NDCG_PROBS)
    loss_down = float(((NDCG_VALUES - NDCG_VALUES[DOWN]) * NDCG_PROBS).sum()) / eb
    p = np.array([0.0 if md == "none" else (1 - r) if md == "total" else (1 - r) / loss_down
                  for md, r in zip(mode, R, strict=True)])
    if (p > 1).any():
        raise ValueError("a retention below what the loss mode can reach")
    bytes_ = 1.0 / np.array(comp)
    return Problem(fams, bytes_, bytes_, comp, R), np.array(mode), p


def synth_draw(prob, mode, p, dependence, n):
    def draw(s):
        rng = np.random.default_rng(s)
        r = rng.choice(6, size=n, p=NDCG_PROBS)
        b = NDCG_VALUES[r]
        u = np.repeat(rng.random(n)[None, :], prob.K, 0) if dependence == "nested" else rng.random((prob.K, n))
        hit = u < p[:, None]
        c = np.repeat(b[None, :], prob.K, 0)
        tot = hit & (mode == "total")[:, None]
        dn = hit & (mode == "down")[:, None]
        c[tot] = 0.0
        c[dn] = NDCG_VALUES[np.broadcast_to(DOWN[r], c.shape)[dn]]
        return c, b
    return draw


def synth(K: int, scenario: str, dependence: str) -> None:
    prob, mode, p = synth_problem(K, scenario)
    out = {"experiment": "E1.1-synthetic", "kind": "SIMULATED", "K": K, "F": prob.F, "scenario": scenario,
           "dependence": dependence, "target_for_truth": T_SYN, "gen_seed": GEN_SEED + K,
           "truth": prob.truth.tolist(), "compression": prob.compression.tolist(), "loss_mode": mode.tolist(),
           "loss_prob": p.tolist(), "environment": E.environment(), "rows": []}
    # check the vectorised bootstrap against the library on a few samples
    draw = synth_draw(prob, mode, p, dependence, 112)
    diffs = []
    for s in range(20):
        c, b = draw(10_000 + s)
        lib = np.array([retention_lower_bound(c[k], b, A_CONF, n_boot=1000, seed=0) for k in range(K)])
        diffs.append(float(np.nanmax(np.abs(lib - S.lower_bound_all(c, b, A_CONF)))))
    out["vectorised_bootstrap_max_abs_diff_vs_library"] = max(diffs)
    for n in NS:
        res = run_cell(prob, synth_draw(prob, mode, p, dependence, n), n, "vector")
        rows = summarise(prob, res, n)
        out["rows"] += [r for r in rows if r["target"] == T_SYN]
    (E.E1 / "synthetic").mkdir(parents=True, exist_ok=True)
    (E.E1 / "synthetic" / f"K{K}_{scenario}_{dependence}.json").write_text(
        json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"synthetic K={K} {scenario} {dependence} done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "real":
        real(sys.argv[2])
    elif sys.argv[1] == "confirm":
        real(sys.argv[2], out_dir="confirm")
    else:
        synth(int(sys.argv[2]), sys.argv[3], sys.argv[4])
