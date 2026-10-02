# Q3G: the controlled intervention (stated before it is run)

Written after the observational analyses 3A–3F (`observational.json`) and before
any intervention computation.

## What the observational results suggest (INFERENCE from MEASURED 3A–3F)

The two strongest competing explanations are:

- **X: composition (H3c, acting through H3b).** DocVQA has many more queries whose
  labelled page sits at or near the decision boundary.
  - 56–58% of DocVQA queries fall in the two lowest relevance-margin quintiles
    (quintile edges pooled per model), against 23–26% on InfoVQA.
  - Within a margin quintile, loss rates are similar between the splits, or higher
    on InfoVQA.
  - Boundary queries both lose and gain under compression. That raises the
    per-query spread of c − R b. Together with the lower mean baseline nDCG, the
    certification SE is 1.6–3.8x larger on DocVQA, while pool retention for the
    same pipeline differs only marginally (0, 3 and 12 of 40 candidates beyond the
    intervals for ColQwen2, ColQwen2.5 and ColPali).
- **Y: intrinsic fragility (H3a).** DocVQA representations are more fragile under
  compression *at equal difficulty*. There is some support: among baseline-rank-1
  queries, the binary codes lose the top hit in 6.7–7.1% of DocVQA queries against
  2.5–4.2% of InfoVQA queries. ColPali also shows 12 of 40 candidates with lower
  pool retention beyond the intervals.

## The intervention: margin-matched InfoVQA

Reweight InfoVQA queries so that their distribution over the five pooled
relevance-margin quintiles (fixed in 3C, per model) equals DocVQA's. Weight per
InfoVQA query in quintile i: w = p_Doc(i) / p_Info(i). Nothing else changes: the
same queries, labels, candidates and scores.

**Hypothesis.**
- Under X, matched InfoVQA reproduces DocVQA's certification difficulty and
  selection: a similar SE and similar selected compression.
- Under Y, a clear gap remains, with DocVQA still less retentive and harder to
  certify at equal margin.

**Measured on matched InfoVQA, compared with DocVQA, per model:**
1. Pool retention R_w = Σ w c / Σ w b for all 40 candidates, with bootstrap
   intervals (weighted resampling), and the count of candidates with R_Doc < R_w
   beyond the intervals.
2. Certification SE at n = 225, with weighted moments, for all 40 candidates.
   **Primary quantity:** the share of the log SE gap closed,
   1 − median log(SE_Doc / SE_matched) / median log(SE_Doc / SE_Info).
3. Selection under the frozen E1 code (methods A and C at δ = 0.05, plus the
   others as computed). Draws: `np.random.default_rng(s).choice(N, n, p = w/Σw)`
   for s = 0..1999, so the formal claims apply to the matched distribution with
   truth R_w. n ∈ {50, 100, 225}, T ∈ {0.99, 0.97, 0.95}. Compared with DocVQA's
   E1/E1.2 results at the same n and T.
4. The effective sample size of the weights, (Σw)²/Σw².

**Reading (fixed now).**
- ≥ 75% of the log SE gap closed *and* the median selected compression of C and A
  at n = 225 within a factor of 2 of DocVQA's → "consistent with X; composition
  explains most of the difference".
- ≤ 25% closed → "consistent with Y, or with factors not captured by margin".
- In between → "partial; both contribute". This is reported as H3d.
- This is an observational reweighting, not causal proof.

**Robustness (same design, a different matching variable, reported but not used
for the reading):** match on baseline-rank strata (rank 1; 2–5; > 5) instead of
margin quintiles.

**Limits known in advance.**
- Matching on one variable cannot rule out others correlated with it.
- InfoVQA has few queries in the lowest quintiles (8–18% of 494), so weights there
  are large and the effective sample size drops.
- Both splits share each model's encoder but not pages or queries.
