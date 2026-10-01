"""R10: what the one-sided lower bound does and does not guarantee (release audit).

    OPTIVISION_DATA=data python scripts/universal_study/s10_bound_audit.py reports/universal/audit/bound_audit.json


A. Coverage of retention_lower_bound for ONE fixed candidate: treat the full query set's
   per-query (baseline, compressed) pairs as the population, draw calibration samples of n
   queries, and count how often the bound is <= the population retention.
B. Selection among K candidates whose true retention is just below the target: how often
   does at least one pass (=> a wrong configuration is chosen), for several K and rules.
C. Adversarial: zero-variance bootstrap with few queries; one query dominating; shift.
"""
import json
import os
import sys

import numpy as np

from optivision.benchmark import load_legacy_dataset
from optivision.compose import Pipeline
from optivision.evaluation import per_query_metrics, retention, retention_lower_bound
from optivision.scoring import maxsim_matrix
from optivision.stages import AdaptiveMerge, BinaryQuantizer, Int4Quantizer, Int8Quantizer

D = os.environ.get("OPTIVISION_DATA", "data")
out = {}
ds = load_legacy_dataset(f"{D}/cache/colpali_docvqa_test_subsampled.npz", f"{D}/vidore_docvqa_test_subsampled/queries.json", name="docvqa", with_images=False)
rel = ds.relevant(); base = maxsim_matrix(ds.queries, ds.corpus)
b = per_query_metrics(base, rel, ["ndcg@5"])["ndcg@5"]
cfgs = {"binary": [BinaryQuantizer()], "int4": [Int4Quantizer()], "adaptive_c0.6>int8": [AdaptiveMerge(radius=0.6, center="mean"), Int8Quantizer("per_vector")],
        "adaptive0.5>binary": [AdaptiveMerge(radius=0.5), BinaryQuantizer()]}
pq = {}
for k, st in cfgs.items():
    p = Pipeline(st); c = p.fit(ds.corpus).compress(ds.corpus)
    pq[k] = per_query_metrics(maxsim_matrix(p.transform_queries(ds.queries), c), rel, ["ndcg@5"])["ndcg@5"]
ok = np.isfinite(b)
rng = np.random.default_rng(1)

print("A. coverage of the one-sided bound for a fixed candidate (population = 451 DocVQA queries)")
A = []
for k, c in pq.items():
    bb, cc = b[ok], c[ok]; true = cc.mean() / bb.mean()
    for n in (25, 50, 100, 225):
        for conf in (0.975, 0.999):
            hits = []; zero_var = 0
            for t in range(400):
                idx = rng.integers(0, bb.size, n)
                lb = retention_lower_bound(cc[idx], bb[idx], conf, n_boot=1000, seed=t)
                hits.append(lb <= true)
            cov = float(np.mean(hits))
            A.append({"candidate": k, "true": true, "n": n, "confidence": conf, "coverage": cov})
            print(f"  {k:22s} true {true:.4f} n={n:3d} conf={conf}: coverage {cov:.3f}", flush=True)
out["A"] = A

print("B. K independent candidates, each with true retention = target - delta; P(at least one passes)")
# Candidate j loses query i entirely with probability p = 1 - (target - delta), independently
# (independence is the worst case for multiplicity; real candidates are positively correlated).
B = []
bb = b[ok]
for target, delta in ((0.97, 0.01), (0.97, 0.005), (0.95, 0.01)):
    p_loss = 1 - (target - delta)
    for K in (1, 10, 40):
        for conf in sorted({0.975, 0.999, 1 - 0.025 / K}):
            if K == 1 and conf not in (0.975, 0.999):
                continue
            wrong = 0
            for t in range(300):
                idx = rng.integers(0, bb.size, 225)
                bs = bb[idx]
                for j in range(K):
                    cs = bs * (rng.random(bs.size) >= p_loss)
                    if retention_lower_bound(cs, bs, conf, n_boot=500, seed=t * 100 + j) >= target:
                        wrong += 1
                        break
            B.append({"target": target, "true": target - delta, "K": K, "confidence": conf, "p_wrong_choice": wrong / 300})
            print(f"  target {target} true {target-delta:.3f} K={K:2d} conf={conf:.5f}: P(some infeasible candidate passes) {wrong/300:.3f}", flush=True)
out["B"] = B

print("C1. zero-variance bootstrap: compressed identical to baseline on every sampled query")
c = np.array([1.0] * 10 + [0.0]); bq = np.array([1.0] * 11)  # true: 10/11 = 0.909 if the 11th query is in the population
hits = 0
for t in range(2000):
    idx = rng.integers(0, 11, 10)
    if retention_lower_bound(c[idx], bq[idx], 0.999, seed=t) >= 0.97: hits += 1
print(f"  population retention {c.mean()/bq.mean():.3f}; 10-query calibration passes a 0.97 target at 0.999 in {hits/2000:.3f} of draws")
out["C1"] = {"population_retention": c.mean() / bq.mean(), "n": 10, "pass_rate": hits / 2000}

print("C2. resamples with zero baseline mean are dropped (NaN) rather than counted")
bq = np.array([1.0] + [0.0] * 29); c = np.array([1.0] + [0.0] * 29)
print("  one of 30 queries has a relevant hit; LB =", retention_lower_bound(c, bq, 0.999), "point =", retention(c, bq, n_boot=0)[0])
out["C2"] = {"lb": retention_lower_bound(c, bq, 0.999)}
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1, default=float)
