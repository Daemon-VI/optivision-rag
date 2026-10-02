# E1.2 freeze record (written before any confirmation data was loaded)

Date 2026-10-02, branch `research-investigation`. The confirmation datasets
(ColQwen2.5-v0.2 on ViDoRe V1 DocVQA and InfoVQA, and answerai-colbert-small-v1
on BEIR SciFact) have not been loaded by any E1 script before this record. The
scripts refuse to load them unless this file exists.

## Frozen procedure

**Method C at δ = 0.05.** Exactly as implemented in `scripts/research/e1_stats.py`
and `e1_1_study.py` at the commit that adds this file:
- exact one-sided betting p-values for H0_k: E[c_k − T b] ≤ 0, using
  Y = (c − T b + T)/(1 + T), aGRAPA bets and cap c = 0.9;
- fixed-sequence testing inside each family of `recommended_search_space`, in its
  own order, at δ/F, where F is the number of families (7);
- deploy the smallest stored size among the rejected candidates, otherwise float32.

No parameter, ordering, bet, level or tie-break may change after this point.

## Protocol (identical to E1.1)

- **Per-query store**, built with `e1_0_rebuild.py --replay-only`. The store replay
  must reproduce every committed s7b outcome of that dataset in
  `reports/universal/selection_rules/` exactly. The re-run of s7b from fresh
  scores is skipped to save time; the replay comparison is kept.
- **Metric:** the one s7b used (nDCG@5 for ViDoRe, nDCG@10 for SciFact), labels
  reference.
- **Resampling:** seeds 0–1999, i.i.d. with replacement from the pool,
  n ∈ {50, 100, 225}, T ∈ {0.99, 0.97, 0.95}.
- **Truth:** finite query-pool truth.
- **Reported alongside C, as computed by the same code, with no decision depending
  on them:** A, A-bonf, B, C-split, D, and every method at δ = 0.01.

## Confirmation criterion (primary)

For C at δ = 0.05, in each of the 27 cells (3 datasets × 3 n × 3 T): the observed
miss rate must be consistent with the claim, meaning the lower end of its 95%
Clopper–Pearson interval is ≤ 0.05. **Confirmed** if all 27 cells are consistent;
otherwise **not confirmed**, and the failing cells are reported and investigated.
Compression is descriptive, not part of the criterion.

## Predictions (from the sample-size lower bound in `DESIGN.md`; recorded before the run)

Minimum n for a valid test to reject even a *lossless* candidate with
probability ≥ 1/2:

| pool (baseline mean) | level | T=0.99 | T=0.97 | T=0.95 |
|---|---|---|---|---|
| ColQwen2.5 DocVQA, nDCG@5 (0.618) | 0.05 | 371 | 122 | 72 |
| ColQwen2.5 DocVQA, nDCG@5 (0.618) | 0.05/7 (C) | 683 | 225 | 133 |
| ColQwen2.5 InfoVQA, nDCG@5 (0.917) | 0.05 | 250 | 83 | 49 |
| ColQwen2.5 InfoVQA, nDCG@5 (0.917) | 0.05/7 (C) | 461 | 152 | 91 |
| SciFact, nDCG@10 (0.746) | 0.05 | 307 | 101 | 60 |
| SciFact, nDCG@10 (0.746) | 0.05/7 (C) | 567 | 187 | 111 |

- C deploys nothing at T = 0.99 on any dataset.
- At n ≤ 100, C deploys in at most a few percent of resamples.
- At n = 225 and T = 0.97, C rarely deploys on ColQwen2.5 DocVQA (225 needed even
  for a lossless candidate), and may deploy on InfoVQA and SciFact.
- At n = 225 and T = 0.95, C deploys on all three, if candidates near-lossless at
  that target exist.
- SciFact compresses less than the image pools (R8), so its compression should be
  lower.
