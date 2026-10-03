# Q3H: within-DocVQA relevance-margin split (stated before it is run)

Written after Q3A–Q3G (`REPORT.md`, commit 9704803) and before any split
computation. This is a diagnostic check of the existing Q3 conclusion, not a new
hypothesis search.

## Question

Does the high-margin half of DocVQA behave substantially more like InfoVQA in
compression and certification?

## Data (unchanged)

- The E1 per-query store for colpali_docvqa, colqwen2_docvqa and colqwen25_docvqa
  (451 queries each). Same 40 candidates, nDCG@5 and labels.
- The Q3 feature arrays (`features/*.npz`), which were checked against the store.
- No vectors are regenerated, nothing is re-encoded, and no candidate or scoring
  change is made. InfoVQA is used only as the fixed comparison and never to
  choose the split.

## The split (one split, fixed now)

- Variable: the Q3 relevance margin `base_margin_rel` = (s_rel − max s_nonrel) / L,
  per query token.
- Threshold: the median of that model's **DocVQA** margins (451 values, so the
  median is one query's margin).
- **High half:** margin > median (225 queries). **Low half:** margin ≤ median (226
  queries).
- No other threshold is computed or reported. The exact threshold values are
  recorded in the output.

## Measured per half, per model (with full DocVQA and full InfoVQA alongside)

1. Baseline mean nDCG@5, top-1 accuracy, and margin quantiles. Also the share of
   each half in the pooled Q3 margin quintiles (edges from Q3C, unchanged).
2. Per-query compression sensitivity: for the Q3 representative candidates
   (positions 1, 16, 27, 30), the share of queries that lose, drop out or gain.
3. Pool retention R = Σc/Σb for all 40 candidates. Bootstrap intervals for
   R_half − R_InfoVQA (2,000 independent query resamples, seed 0), with counts
   beyond the intervals in each direction.
4. The certification SE at n = 225, sd(c − R b) / (√225 · mean b), for all 40
   candidates.
   - **Primary quantity:** the share of the log SE gap closed by the high half,
     1 − median log(SE_high / SE_Info) / median log(SE_Doc / SE_Info), over the
     candidates with nonzero SE on all three sets.
5. Selection under the frozen E1 code (`e1_1_study.run_cell`, library mode, all
   methods as computed). The half is a finite query pool. Draws are
   `np.random.default_rng(s).integers(0, N_half, n)` for s = 0..1999 and
   n ∈ {50, 100, 225}, with truth = the half's R, exactly as in E1. The targets
   are E1's (0.99, 0.97, 0.95); T = 0.95 is the reported focus. Full-DocVQA and
   InfoVQA selection rows are read unchanged from E1 (`real/`, `confirm/`).

## Reading (fixed now, per model, then overall)

Primary comparison: the high half against InfoVQA.

- **A (approaches InfoVQA):** ≥ 75% of the log SE gap closed *and*, at n = 225 and
  T = 0.95, the median selected compression of both A and C@0.05 is within a
  factor of 2 of InfoVQA's.
- **B (remains different):** ≤ 25% of the log SE gap closed, *or* the median
  selected compression of both A and C@0.05 is more than a factor of 2 below
  InfoVQA's while the SE gap is not ≥ 75% closed.
- **C (ambiguous):** anything else.

Overall:
- A if at least two models read A and none reads B.
- B if at least two read B and none reads A.
- Otherwise C.

The low half is reported as a contrast and is not part of the reading.

## Limits known now

- This is an observational split of one query pool. The halves are not
  independent datasets and the result is not causal. Selecting on margin also
  selects on everything correlated with it: page, question type and label quality.
- Each half has 225–226 queries. At n = 225, selection resamples a pool of about
  its own size with replacement. This is valid for the half's finite-pool truth,
  but the half is not a population.
- The high half need not match InfoVQA's margin distribution. Its quintile shares
  are reported so this can be judged.
