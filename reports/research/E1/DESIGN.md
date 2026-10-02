# E1 design and pre-registration

Written before any real-data E1.1 run. Branch `research-investigation`.
E1.0 (reproduction) was already running when this was written; it does not
compute anything E1.1 uses beyond the per-query store.

Labels: **MEASURED** (computed from real encoder outputs), **SIMULATED**
(resampling or synthetic generators), **INFERENCE** (reasoning, including
mathematical derivations, which are marked "proof").

## Hypotheses (recorded verbatim before execution)

- **H1a.** The current point/bound-based selection procedure loses substantial
  compression because selection among many candidates is not explicitly
  accounted for.
- **H1b.** Independent certification can provide a defensible post-selection
  statement but will reduce usable compression because of the split in queries.
- **H1c.** A properly constructed simultaneous/ordered-testing method can
  recover some compression relative to naive independent certification while
  controlling the intended error quantity.

These are hypotheses, not expected conclusions.

## Data and protocol

- Development datasets (method design and E1.1): ColPali-v1.3 and ColQwen2-v1.0
  on ViDoRe V1 DocVQA (451 queries) and InfoVQA (494 queries), 500 pages each.
- Confirmation datasets (held back, not loaded by any E1.0/E1.1 script):
  ColQwen2.5-v0.2 (both splits) and SciFact.
- Candidates: the 40 of `recommended_search_space` (v0.3.0), in their order,
  F = 7 families (float16; Ward and centred-adaptive merging x {int8, centred
  int4, binary}).
- Per-query quantity: nDCG@5 with the dataset labels (one relevant page per query).
- **Finite query-pool truth**: for candidate k, R_pool(k) = Σ_i c_ki / Σ_i b_i over
  all pool queries. It is the retention under Q_pool, the empirical distribution of
  the pool. It is not a statement about future queries from any other distribution.
- **Resampling**: seed s ∈ {0, …, 1999}; calibration sample
  `np.random.default_rng(s).integers(0, N, size=n)`, i.i.d. **with replacement**
  from the pool, n ∈ {50, 100, 225}; targets T ∈ {0.99, 0.97, 0.95}. Sampling with
  replacement is what makes Q_pool the exact sampling distribution, so the formal
  claims below can be checked exactly against R_pool, up to Monte-Carlo error.
  The 2,000 samples are resamples of one fixed pool, not 2,000 independent
  datasets. Monte-Carlo intervals (Clopper–Pearson, 95%) describe only the
  resampling error for this pool.

## Statistical definitions

For a query drawn from a distribution Q, with b the float32 index's nDCG@5 and c_k
candidate k's (both in [0, 1]), define R_Q(k) = E_Q[c_k] / E_Q[b] with E_Q[b] > 0.

**Identity.** R_Q(k) ≥ T ⇔ E_Q[c_k − T b] ≥ 0, and c_k − T b ∈ [−T, 1]. With
Y = (c − T b + T)/(1 + T) ∈ [0, 1] and m0 = T/(1 + T):

    H0_k:  E_Q[Y_k] ≤ m0     (R_Q(k) ≤ T: candidate k does not exceed the target)

**Exact p-value (testing by betting; Waudby-Smith & Ramdas 2023).**
K_t = Π_{s≤t} (1 + λ_s (Y_s − m0)), with λ_s ∈ [0, c/m0] chosen from Y_1..Y_{s−1}
only (aGRAPA bet, c = 0.9). Under H0 and i.i.d. draws, K is a non-negative
supermartingale, so by Ville's inequality P(max_t K_t ≥ 1/δ) ≤ δ.
p = min(1, 1/max_t K_t) satisfies P(p ≤ δ) ≤ δ for every n and every bounded
distribution in H0. No normal approximation is involved.

**Methods and the quantity each controls.** "Deploy" means returning a
compressed candidate; otherwise the float32 index is returned.

| method | procedure | controlled quantity | status |
|---|---|---|---|
| A (current) | `calibrate()` defaults on the sample: 0.999 bootstrap bound per candidate on half the sample, family walk, smallest wins; the other half only reported | none for the selected candidate. Approximate (bootstrap) per-candidate level only | as implemented in v0.3.0 |
| A-bonf | A with `multiplicity="bonferroni"` | approximate per-candidate level 1 − 0.001/40. The bootstrap approximation is not a valid finite-sample bound (R10 shows under-coverage when losses are rare and large) | existing API option |
| B | A selects on the first half S1; only the selected k̂ is tested on the second half S2 at δ | P(deploy ∧ R_Q(k̂) < T) ≤ δ | proof below |
| C | all n queries; within each family, fixed-sequence testing in the pre-specified order at δ/F; deploy the smallest rejected | FWER ≤ δ, hence P(deploy ∧ R_Q(k̂) < T) ≤ δ | proof below |
| C-split | order by S1 point retention; fixed-sequence on S2 at δ; deploy the smallest rejected | the same, conditional on S1, hence marginally | proof below |
| D | all n queries; reject H0_k if p_k ≤ δ/K; deploy the smallest rejected | FWER ≤ δ, hence P(deploy ∧ R_Q(k̂) < T) ≤ δ | proof below |

δ ∈ {0.05, 0.01}; 0.05 is primary.

**Proofs (INFERENCE, proof).**
- *B.* S1 and S2 are independent, and k̂ is a function of S1. Conditional on S1,
  p_{k̂}(S2) is a valid p-value for the fixed hypothesis H0_{k̂}. So
  P(p_{k̂} ≤ δ ∧ R(k̂) < T) = E[1{R(k̂) < T} · P(p_{k̂}(S2) ≤ δ | S1)] ≤ δ.
- *C.* Within one family, fixed-sequence testing at level α controls the FWER at
  α under arbitrary dependence: a false rejection requires rejecting the first
  true null in the order, which happens with probability ≤ α. A union bound over
  F families at α = δ/F gives FWER ≤ δ. If a deployed k̂ has R(k̂) < T, then H0_{k̂}
  is true and was rejected, so the event lies inside the FWER event.
- *C-split.* The order is a function of S1 only. Apply the C argument on S2
  conditional on S1.
- *D.* Union bound over the K tests at δ/K.

**Assumptions under which B, C, C-split and D hold.**
1. Calibration queries are i.i.d. draws from Q.
2. The corpus, the candidate list, its order and families, and every fitted stage
   are fixed before the calibration queries are seen. They are: every stage is
   fit on corpus vectors only (`score_candidate` → `Pipeline.compress(corpus)`).
3. Per-query metric values lie in [0, 1], and E_Q[b] > 0.
4. The labels are taken as correct.

**What these statements do not say.**
- Nothing about queries from a distribution other than Q, or about a different
  or growing corpus (R10: retention moves with corpus size).
- Nothing about the realised retention on a finite batch of future queries, which
  fluctuates around R_Q.
- Nothing per query.
- They are marginal over calibration samples, not conditional on the sample at
  hand.
- In this experiment Q = Q_pool, so the simulation tests the claims about the
  pool only.

## A limit that applies to every valid method (INFERENCE, proof)

Let Q give a candidate with c = b, which is lossless, so R_Q = 1, and let
β = E_Q[b]. Let Q' mix Q with probability 1 − ε and the point (b = 1, c = 0) with
probability ε, where ε = β(1 − T) / (T + β(1 − T)), so that R_{Q'} = T and Q' is in
H0. A sample from Q' contains no contaminated draw with probability (1 − ε)^n, and
then it is distributed exactly as a sample from Q. Any test valid at level δ
therefore has P_Q(reject) ≤ δ (1 − ε)^{−n}. Rejecting a *lossless* candidate with
probability ≥ 1/2 requires n ≥ log(1/(2δ)) / (−log(1 − ε)):

| pool (float nDCG@5 mean β) | level | T=0.99 | T=0.97 | T=0.95 |
|---|---|---|---|---|
| ColPali DocVQA (0.584) | 0.05 | 392 | 129 | 77 |
| ColPali DocVQA (0.584) | 0.05/7 (C) | 723 | 238 | 141 |
| ColPali DocVQA (0.584) | 0.05/40 (D) | 1019 | 335 | 198 |
| ColPali InfoVQA (0.846) | 0.05 | 271 | 90 | 53 |
| ColPali InfoVQA (0.846) | 0.05/7 (C) | 500 | 165 | 98 |
| ColPali InfoVQA (0.846) | 0.05/40 (D) | 705 | 233 | 138 |
| ColQwen2 DocVQA (0.607) | 0.05 | 378 | 124 | 74 |
| ColQwen2 DocVQA (0.607) | 0.05/7 (C) | 696 | 229 | 136 |
| ColQwen2 DocVQA (0.607) | 0.05/40 (D) | 981 | 323 | 191 |
| ColQwen2 InfoVQA (0.920) | 0.05 | 250 | 83 | 49 |
| ColQwen2 InfoVQA (0.920) | 0.05/7 (C) | 460 | 152 | 90 |
| ColQwen2 InfoVQA (0.920) | 0.05/40 (D) | 648 | 214 | 127 |

**Predictions made from this table before the run.**
- With n ≤ 225, no valid distribution-free method can certify T = 0.99.
- B (112 certification queries at n = 225) can certify T = 0.95 on all four pools,
  and T = 0.97 only on InfoVQA.
- C and D at T = 0.97 are limited to InfoVQA.
- Real candidates are not lossless, so actual power is lower still.
- Every valid method abstains often at n = 50.

## Design-stage checks (SIMULATED, synthetic only; before any real-data E1.1 run)

Type-I error of the betting p-value at the null boundary, 20,000 replicates:

| null distribution | n | P(p ≤ 0.05) | P(p ≤ 0.01) |
|---|---|---|---|
| two-point at m0 | 50 / 225 | 0.016 / 0.022 | 0.002 / 0.005 |
| uniform with mean m0 | 50 / 225 | 0.009 / 0.020 | 0.001 / 0.004 |
| nDCG-like, rare total losses, R = T | 50 / 225 | 0.000 / 0.002 | 0.000 / 0.000 |

(Bet cap c = 0.5 in this check; the cap does not affect validity.)

Power, which motivated the cap. Probability of rejecting at 0.05 (and at 0.05/7)
with n = 225 and a DocVQA-like baseline:

| candidate | c = 0.5 | c = 0.75 | c = 0.9 | c = 0.99 |
|---|---|---|---|---|
| lossless, T = 0.97 | 0.00 / 0.00 | 0.08 / 0.00 | 0.93 / 0.00 | 1.00 / 0.00 |
| lossless, T = 0.95 | 0.98 / 0.00 | 1.00 / 0.44 | 1.00 / 1.00 | 1.00 / 1.00 |
| rare total losses, R = 0.99, T = 0.97 | — | 0.02 / 0.00 | 0.23 / 0.00 | 0.27 / 0.00 |
| rare total losses, R = 0.98, T = 0.95 | — | 0.29 / 0.02 | 0.33 / 0.09 | 0.32 / 0.12 |

With c = 0.5, even a lossless candidate cannot be certified at T = 0.97 with 225
queries. c = 0.9 is close to c = 0.99 and keeps 10% of the wealth after a
worst-case observation. **Frozen: c = 0.9.** Null rejection with c = 0.9 (rare
total losses, R = T, n = 225) was 0.029 ≤ 0.05.

## Synthetic stress test (generator in `scripts/research/e1_1_study.py`)

- K ∈ {10, 40, 82}: a lossless floor (2x) plus K − 1 candidates in families of 5,
  so F = 1 + ⌈(K − 1)/5⌉.
- Baseline nDCG@5 distribution: P(rank 1..5, miss) = (0.50, 0.08, 0.05, 0.03,
  0.02, 0.32), mean 0.596.
- Loss modes alternate by family. "Total" (odd families): a hit query drops to 0,
  so the losses are rare and large. "Down" (even families): the relevant page
  moves one rank, so the losses are frequent and small. This makes the variance
  differ by candidate.
- Dependence: "nested" (one shared fragility per query, so more aggressive
  candidates fail on supersets of queries, like real merges) or "independent".
- Scenarios at T = 0.95: separated, near-tied, many below, one above, several
  above. Exact definitions are in `synth_problem`; truth retention is known in
  closed form.
- The generator was written to cover the cases the plan lists. It was not tuned
  after any run.

## What is reported

For each dataset (or scenario) × n × T × method: deployment and abstention rate,
median and p10/p90 selected compression (abstain = 1x), oracle compression (the
smallest candidate with truth ≥ T), the median ratio to the oracle, miss rate with a
Clopper–Pearson interval, worst miss, truth retention of the deployed choice, the
excess over target, the most common selections, and seconds per selection. For
formal methods, also whether the observed miss rate is consistent with the
claimed δ (the lower end of the 95% interval ≤ δ). For A, also how often its
reported held-out flag said "met", and how often that flag was wrong.
