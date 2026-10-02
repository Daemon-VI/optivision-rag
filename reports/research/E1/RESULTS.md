# E1.0 + E1.1 results

Design and pre-registration: `DESIGN.md`, committed before any real-data E1.1 run
(`0ad40ba`). All tables: `RESULTS_TABLES.md`, generated from the JSON by
`scripts/research/e1_tables.py`. Labels: **MEASURED**, **SIMULATED**,
**INFERENCE**.

## 1. E1.0 reproduction (MEASURED)

For ColPali-v1.3 and ColQwen2-v1.0 on ViDoRe V1 DocVQA and InfoVQA, the 40
default candidates were re-scored with the committed s7b protocol.

- **Candidate lists and compression ratios are identical** to the committed ones.
- **All 3,360 committed outcomes were reproduced exactly**: 840 per dataset =
  2 references × 7 rules × 3 targets × 20 splits. This holds both when s7b's rules
  are re-run from fresh scores and when they are replayed from the compact per-query
  store alone. Zero mismatches; the maximum float difference is 0.0.
  Files: `e1_0_*.json`.
- **Store** (`store/*.npz` + `.json`, 8–15 KB each): per-query nDCG@5 of the float32
  baseline and of each candidate, under the labels reference and the baseline@1
  reference, plus candidate metadata. sha256 values are in `e1_0_*.json`.
- The run records `dirty_tree: true` because the E1.0 output files were
  uncommitted while it ran; the scripts are those of `0ad40ba`.

## 2. Statistical definitions

Exactly as pre-registered in `DESIGN.md`:
- the target condition is the bounded mean E[c − T b] ≥ 0;
- exact betting p-values (Ville's inequality, valid for every n);
- methods A, A-bonf, B, C, C-split and D, with the quantity each controls and
  proofs for B, C, C-split and D;
- finite query-pool truth, with resampling with replacement (seeds 0–1999).

## 3. Results on the development datasets (SIMULATED resampling of MEASURED per-query results)

"Selected compression under the procedure's stated error/coverage criterion", at n
= 225, T ∈ {0.99, 0.97, 0.95}, δ = 0.05. Each cell is the median compression
(float32 = 1x when nothing is deployed) · deployment rate · miss rate. The miss
rate counts resamples where the deployed candidate's finite query-pool truth is
below T.

| dataset · T | finite-pool oracle | A (current) | B | C | C-split | D |
|---|---|---|---|---|---|---|
| ColPali · DocVQA · 0.99 | 11.8x | 3.9x · 97.7% · 0.7% | 1.0x · 0% · 0% | 1.0x · 0% · 0% | 1.0x · 0% · 0% | 1.0x · 0% · 0% |
| ColPali · DocVQA · 0.97 | 49.7x | 3.9x · 100% · 0.1% | 1.0x · 0.1% · 0% | 1.0x · 0.7% · 0% | 1.0x · 0.3% · 0.1% | 1.0x · 0.1% · 0% |
| ColPali · DocVQA · 0.95 | 125.4x | 7.8x · 100% · 0.1% | 1.0x · 9.8% · 0% | 3.9x · 99.0% · 0% | 1.0x · 7.8% · 0.1% | 1.0x · 11.3% · 0% |
| ColPali · InfoVQA · 0.99 | 72.7x | 7.8x · 100% · 0.1% | 1.0x · 0% · 0% | 1.0x · 0% · 0% | 1.0x · 0% · 0% | 1.0x · 0% · 0% |
| ColPali · InfoVQA · 0.97 | 178.9x | 30.4x · 100% · 0.2% | 1.0x · 3.7% · 0% | 1.0x · 49.8% · 0% | 1.0x · 3.2% · 0% | 1.0x · 5.0% · 0% |
| ColPali · InfoVQA · 0.95 | 299.9x | 72.7x · 100% · 0% | 1.0x · 42.4% · 0% | **72.7x · 100% · 0%** | 23.2x · 69.3% · 0% | 43.4x · 100% · 0% |
| ColQwen2 · DocVQA · 0.99 | 58.9x | 2.0x · 100% · 0% | 1.0x · 0% · 0% | 1.0x · 0% · 0% | 1.0x · 0.4% · 0% | 1.0x · 0.1% · 0% |
| ColQwen2 · DocVQA · 0.97 | 85.3x | 3.9x · 100% · 0.1% | 1.0x · 0.9% · 0% | 1.0x · 0.8% · 0% | 1.0x · 3.0% · 0% | 1.0x · 3.3% · 0% |
| ColQwen2 · DocVQA · 0.95 | 280.9x | 11.6x · 100% · 0% | 1.0x · 24.6% · 0% | 7.8x · 100% · 0% | 1.0x · 21.2% · 0% | 7.8x · 58.8% · 0% |
| ColQwen2 · InfoVQA · 0.99 | 47.5x | 7.8x · 100% · 1.9% | 1.0x · 0% · 0% | 1.0x · 0% · 0% | 1.0x · 0% · 0% | 1.0x · 0% · 0% |
| ColQwen2 · InfoVQA · 0.97 | 161.2x | 27.6x · 100% · 0.7% | 1.0x · 3.2% · 0% | 7.8x · 99.9% · 0% | 1.0x · 5.5% · 0% | 1.0x · 5.6% · 0% |
| ColQwen2 · InfoVQA · 0.95 | 280.7x | 63.0x · 100% · 0% | 1.0x · 33.2% · 0% | **63.0x · 100% · 0%** | 15.3x · 77.3% · 0% | 47.5x · 100% · 0% |

The **finite-pool oracle** is the smallest candidate whose retention over the whole
pool is ≥ T. It is chosen with knowledge of every pool query, so it is an
unattainable upper reference, not a method.

**The current rule A**, across n:
- its miss rate falls with n, from up to 21.6% (ColPali DocVQA, T = 0.99) and
  25.7% (ColQwen2 InfoVQA, T = 0.99) at n = 50, to at most 1.9% at n = 225;
- its worst miss is a deployed candidate with 89.3% truth retention
  (`hierarchical_merge(0.1) > binary`, ColPali DocVQA, n = 50; one resample in
  2,000);
- its reported held-out flag said "target met" while the truth missed in up to
  11.3% of resamples at n = 50.

At n = 50 and n = 100, **no valid method deploys anything** in more than 0.25% of
resamples, on any dataset or target.

**Cost.** A takes about 43 ms per selection (40 bootstraps, library code). The
betting-test methods take 0.02–1.3 ms per selection, on top of the same scoring
of candidates, which dominates in practice.

## 4. Synthetic stress test (SIMULATED; truth known exactly; T = 0.95)

24 of the 30 planned cells are complete: K ∈ {10, 40} for all 5 scenarios × 2
dependence structures, and K = 82 for the separated and near-tied scenarios. Full table in `RESULTS_TABLES.md`.

- **B, C, C-split and D: every observed miss rate is at most 0.3%,** against a
  claimed 5%. There are zero contradictions of the claimed level in 576 checks
  (24 cells × 3 n × 4 methods × 2 δ).
- **A misses far more often when many candidates sit just below the target,**
  and more with K = 40 than with K = 10. Miss rates of A:

| scenario (independent losses) | K=10, n=50 / 225 | K=40, n=50 / 225 |
|---|---|---|
| many below | 40.5% / 3.1% | 88.4% / 13.0% |
| one above | 22.7% / 1.5% | 70.7% / 5.2% |
| several above | 15.5% / 0.6% | 53.3% / 5.0% |
| near-tied | 16.8% / 0.3% | 37.8% / 5.9% |

  These reproduce R10's warning: the per-candidate 0.999 bound does not control
  the error of the selected candidate. The real candidates are strongly
  correlated (nested losses), and their measured miss rates are much lower.
- **A-bonf barely changes A** (for example 13.0% → 10.4%). It applies Bonferroni to
  an approximate bootstrap bound, which under-covers when losses are rare and
  large.
- **The error of A grows with the number of candidates.** In the near-tied,
  independent-loss scenario, A's miss rate at n = 225 is 0.3% (K = 10), 5.9%
  (K = 40) and 25.8% (K = 82). The valid methods stay at 0%.
- **K = 82, many-below / one-above / several-above (6 cells): not run.** The run
  driver was stopped by the host for low memory, and those cells were never
  launched. Nothing above depends on them.

## 5. Compression and conservatism trade-off (INFERENCE from 3–4)

- **Every valid procedure is far more conservative than A, and A is far more
  conservative than the oracle.** A reaches 3–25% of the finite-pool oracle's
  compression, and the valid methods reach the same or less.
- **The gap between A and the oracle is not caused by ignoring selection.**
  Accounting for selection validly (C, D) never chose more than A. The gap comes
  from statistical uncertainty with 112–225 calibration queries: A's bound on half
  the sample, plus its family walk.
- **Among the valid methods, C (family-ordered fixed-sequence testing on all n
  queries) dominates.**
  - It matches A exactly at T = 0.95 on InfoVQA (72.7x and 63.0x, deployed in 100%
    of resamples, 0 misses).
  - It is the only valid method that deploys reliably at T = 0.97: 7.8x on ColQwen2
    InfoVQA (99.9%), and 50% of resamples on ColPali InfoVQA.
  - On DocVQA at T = 0.95 it reaches 3.9–7.8x where A reaches 7.8–11.6x.
- **B** (A plus independent certification on half the queries) deploys in at most
  42% of resamples at T = 0.95, and almost never at T = 0.97.
- **D** (Bonferroni over 40) loses to C wherever both deploy (for example 43.4x
  against 72.7x).
- **C-split** loses to C: half the queries are spent on ordering.

## 6. Observed target misses

- **Valid methods:** at most 0.05% on real data and ≤ 0.3% on synthetic data, always
  below δ. They are conservative in practice: the betting bound and the multiplicity
  corrections are worst-case.
- **A on real data:** a miss rate that decreases with n (above). Misses are
  concentrated in merged-binary candidates at small n. At n = 225, A's misses
  (≤ 1.9%) are small but not covered by any statement A makes.
- **A on synthetic data:** up to 88% at n = 50 and 13% at n = 225 when many
  candidates are near-misses.

## 7. Formal claims justified by this work

**Method C (and B, C-split, D) at level δ.** Assumptions:
1. calibration queries are i.i.d. draws from a query distribution Q;
2. the corpus, the candidate list with its families and order, and all fitted
   stages are fixed before the queries are seen (they are fit on the corpus only);
3. per-query nDCG@5 is in [0, 1], E_Q[b] > 0, and the labels are taken as given.

Under these, the procedure satisfies, for every sample size n,

    P( a compressed candidate k̂ is deployed  and  E_Q[c_k̂] / E_Q[b] < T ) ≤ δ.

Here the probability is over the draw of the calibration queries. This is a
finite-sample, distribution-free statement about the selected configuration
(INFERENCE, proof in `DESIGN.md`). It is consistent with every simulation here:
0 contradictions in 288 real and 576 synthetic checks.

On the word "guarantee":
- Under the three assumptions above, this is a guarantee in the usual
  statistical sense for C, B, C-split and D.
- No such statement exists for A or A-bonf.
- The simulations do not prove the claim; they are consistent with the proof for
  Q = the pool's empirical distribution.

## 8. What cannot be claimed

- **Nothing for A or A-bonf.** The current `calibrate()` / `optimize()` makes no
  valid statement about the selected configuration, and its miss rate can be large
  when many candidates are near the target (synthetic) or n is small (real, n = 50).
- **No valid method can certify T = 0.99 with ≤ 225 queries.** This holds even for
  a lossless candidate, by the sample-size lower bound in `DESIGN.md`. It is a
  property of the problem, not of the betting test.
- **Formal certification at T = 0.97 on DocVQA-like pools needs more than 225
  queries.** C deployed in under 1% of resamples there, as the lower bound
  predicted (229–238 queries for C, before any real loss).
- The claims concern R_Q, the ratio of expected nDCG@5. They are not about any
  finite batch of future queries, about a different corpus or corpus size (R10),
  about a different query distribution, or about individual queries.
- They do not transfer to the 2,560-d model, ColQwen2.5 or SciFact until those
  are run (the confirmation sets are untouched).
- The miss rates reported for A are relative to the finite query pools. They are
  not deployment error rates.

## 9. Limitations

- **One query pool per dataset.** The 2,000 resamples are resamples of that pool,
  not independent datasets. Monte-Carlo intervals cover resampling error only.
  ColPali and ColQwen2 share the same queries, so the four datasets are two query
  pools.
- **i.i.d. with replacement** was chosen so the formal model holds exactly. A real
  calibration set is a sample without replacement from an unknown population;
  near-duplicate queries would weaken i.i.d.
- **Synthetic generators** reflect choices (DocVQA-like baselines, two loss modes,
  nested or independent losses). They were not tuned after any run, but they do
  not span every possible structure.
- **The bet cap c = 0.9** was chosen at the design stage from synthetic power
  checks. Other valid tests (other bets, Hoeffding–Bentkus) could be somewhat
  more powerful. None can beat the sample-size lower bound.
- **6 of 30 synthetic cells (K = 82) were not run** (see §4).
- **Only labels-based nDCG@5** was studied, not the baseline@1 reference.

## 10. Hypotheses

- **H1a — partly supported, and its causal part rejected.**
  - A does lose substantial compression against the finite-pool oracle: median
    3–25% of it.
  - But not *because* selection is ignored. Every procedure that accounts for
    selection validly chose the same or less.
  - Ignoring selection is what lets A be *less* conservative, and it is also why A
    has no valid statement and misses often at n = 50 and in near-miss synthetic
    configurations.
- **H1b — supported, more strongly than stated.**
  - B's statement is valid: 0 observed misses.
  - But B is close to unusable at n ≤ 225. It deploys in at most 42% of resamples
    at T = 0.95, and almost never at T ≥ 0.97.
- **H1c — supported where any valid method can work.**
  - C recovers compression relative to B with the same controlled error: 72.7x
    deployed in 100% of resamples against B's 42% (ColPali InfoVQA, T = 0.95), and
    0 contradictions of δ.
  - At T = 0.99, and at T = 0.97 on DocVQA, nothing can be recovered at these sample
    sizes.

## 11. Recommendation for the next step (not started)

1. **E1.2 confirmation, as pre-registered.** Freeze C at δ = 0.05 (family-ordered
   fixed-sequence, exact betting p-values, c = 0.9) exactly as in
   `scripts/research/e1_stats.py`. Run it with A on the held-back ColQwen2.5
   (both splits) and SciFact. This is CPU only, about 1–2 h including building
   their per-query stores, and needs no method change.
2. **Finish the 6 remaining K = 82 synthetic cells.** CPU, about 1 h, run with
   fewer parallel processes to stay within memory.
3. Only after review: decide whether a valid selection mode belongs in the library
   as a new opt-in option. That would be a library change, which is a stop
   condition, so it is not done here. Q3 (DocVQA vs InfoVQA) can also reuse these
   per-query stores.
