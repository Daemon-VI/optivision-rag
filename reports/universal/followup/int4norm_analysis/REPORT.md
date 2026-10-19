# int4 norm batches: exploratory analysis

## Status

- **Exploratory and post hoc.** Everything here was looked for after the results of `f9b` to
  `f13b` were seen. None of it changes the verdicts in `../INT4NORM_SUMMARY.md`: 5 held, 5 failed.
  The statistics below were chosen with the data in view, so they are hypotheses for a later test,
  not evidence.
- **Data.**
  - The 46 s17 units that Q1 to Q9 read: 24 text, 14 page and 8 SciDocs, from 19 models.
  - The 2 `f13b` scale units, where stated.
  - Their geometry files.
  - The earlier s13 files of the same units.
- **Retention.** ndcg@5 over the same unit's float32 run, unless stated otherwise.
- **Not independent.** Units are clustered by model, and most models appear on 2 or 3 datasets, so
  there are effectively about 19 independent points, not 46.
- **How it was made.** Four analysts each took one angle. A separate verifier then recomputed each
  claim with its own script and tried to refute it.
  - Of 59 claims: 43 confirmed, 9 corrected and 7 unverifiable. None was refuted outright.
  - Corrections are applied below. Anything left unverifiable is marked as such.
  - The four verified write-ups are in `angles/`, and every script is in `scripts/`. The scripts
    were run from a scratch folder, so their output paths name it.

## Main findings

1. **int4norm(mean) is close to int8 at half the bytes on these units.**
   - It keeps point retention ≥ 0.95 on all 46 units, and its lower bound ≥ 0.95 on 45.
   - Its mean shortfall against int8 is 0.001 on text, 0.001 on pages and 0.004 on SciDocs.
   - Only one unit has the two intervals apart, and there it favours int8.
2. **After Ward merging, the codec matters more.** At 1/3 and 1/4 of the vectors, merging
   then int4norm(mean) tracks merging then int8, at half the bytes (mean differences within ±0.005
   in every family).
   - Merging then int4(mean) falls behind instead: by 0.013 to 0.016 on text and 0.023 to 0.031 on
     SciDocs.
   - Why merging should favour storing the norm is not tested.
3. **Two regimes, and a label-free statistic that separates them on this data.** The statistic is
   `nsr = (peak_to_rms² / 588) / (1 − doc_cos_to_mean²)`. Its numerator is int4's expected relative
   noise energy, and its denominator is the off-mean part of a document vector. Both inputs come
   from the geometry file.
   - **Low regime.** At nsr ≤ 0.5 (36 units), int4norm alone gives at least 0.944, and below 0.95
     on only one unit.
   - **High regime.** Above 0.5 (10 units: nemotron, LateOn and mLateOn), int4norm alone is below
     0.95 on 9. Centring is needed there, and with it every one of the 10 is at 0.988 or higher.
   - **Caveats.**
     - nsr was built after seeing the table, from about 18 statistics tried. Its in-sample AUC of
       0.997 is optimistic.
     - Leave-one-model-out accuracy was 0.957, the same as the int4 score correlation already in
       the geometry file.
     - The 10 positives come from 3 model families.
4. **Q3 and Q7 failed because storing the norm repaired more than expected.**
   - nsr orders the four nemotron units exactly as their int4norm retention: 0.743, 0.409, 0.152
     and 0.082. That is 4 points.
   - The 4b models have a weaker dominant coordinate than the 8b models (top-4 second-moment share
     0.61 to 0.63, against 0.76 to 0.78).
   - On LateOn, the norm closed 0.74 to 0.83 of the gap that centring closes.
   - That the norm fixes norm scatter but not lost direction is a reading of these numbers, not
     something an experiment here isolated.
5. **No harm from storing the norm was detected, which is not the same as no harm.**
   - Paired per-query tests at ndcg@10 found no unit, out of 48, with an interval wholly below
     zero for either pair.
   - Intervals on healthy units are 0.01 to 0.04 wide, so a harm of about 0.01 would not show.
   - About 96 comparisons were made, with no correction for multiplicity.
6. **The Q6 tolerance was about the size of the noise.**
   - colpali-v1.3 on DocVQA lost 0.012 at ndcg@5. At ndcg@10, per query, the difference is −0.002
     with interval −0.009 to +0.006, and 18 queries went up against 17 down, of 451.
   - The median interval width per dataset is 0.016 to 0.027, against Q6's 0.01 margin.
   - Only ndcg@10 is stored per query, so this uses @10 as a proxy for the @5 headline. The proxy
     may understate the @5 gap.
7. **EVIE-4.5B on TabFQuAD remains unexplained.**
   - Its intervals are about three times the TabFQuAD median width.
   - int8 alone changes the ndcg@10 of 51 of its 280 queries, against 0 to 8 for most other page
     units (nemotron DocVQA: 45 to 50).
   - The pattern fits a cost from centring on this unit: int4norm(mean) is 0.033 below int4norm
     at @10, with interval −0.056 to −0.011. Storing the norm adds little on top.
   - But it is one unit among many looks, and its sign tests give p = 0.08 to 0.2. The cause is not
     established.

## Reproducibility and consistency

- **s17 and s13 agree exactly.** 41 s17 units have an s13 twin on the same corpus. Across 68 file
  comparisons, the baselines and every shared row value are identical to 0.0.
  - This is the same hardware class and deterministic code. It does not show reproducibility on
    other hardware.
  - The only differences are `f13b` against `f3`, which used FiQA distractors, not TREC-COVID. That
    is a different corpus.
- **The same models break on all three text datasets.** With a cut at plain int4 below 0.90, the
  broken set is the same on SciFact, NFCorpus and SciDocs: LateOn, answerai, mLateOn and both
  mxbai-edge models.
  - The cut is post hoc and close calls sit at it: answerai SciFact 0.899, and GTE SciDocs 0.904.
  - The 4 models not run on SciDocs are near 1.0 elsewhere.
  - Severity varies a lot by dataset. answerai is 0.455 on NFCorpus and about 0.89 on the other
    two. LateOn is 0.134 on SciDocs and 0.638 on NFCorpus.
- **Ranks within int4norm(mean) are noise.** int4norm(mean) lies between 0.984 and 1.017 on all
  32 text units, so its rank agreement across datasets is mostly noise (Spearman −0.29 to 0.33).
- **More same-domain documents changed little (`f13b`).** That is SciFact plus 10,000 TREC-COVID
  documents, against SciFact alone.
  - The baselines fell by 0.033.
  - int4norm and int4norm(mean) moved by 0.007 or less on both models, and all intervals overlap.
  - Plain int4 on answerai fell 0.027, within wide intervals.
  - Two models, one distractor set: weak evidence.
- **Timing.** On SciDocs, s13 averaged 358 s per row for answerai and 231 s for LateOn, over 49 and
  54 rows. s17 took 54 s and 43 s, over 21 rows.
  - So the GPU page max is about 5 to 7 times faster per row, on top of s17 having fewer rows.
  - Rows differ in cost (projection rows are cheaper), and Kaggle load was not controlled.

## Bytes against retention

- **Units meeting the bar, per family.** Each cell is point retention ≥ 0.95 / lower bound ≥ 0.95.

| config | bytes of float32 | text (24) | pages (14) | SciDocs (8) |
|---|---|---|---|---|
| int8 (per_vector) | 0.255 | 24 / 24 | 14 / 14 | 8 / 8 |
| int4norm(mean) | 0.13 | 24 / 24 | 14 / 13 | 8 / 8 |
| int4(mean) | 0.13 | 23 / 22 | 14 / 13 | 7 / 6 |
| int4norm | 0.13 | 20 / 19 | 10 / 10 | 6 / 5 |
| int4 | 0.13 | 13 / 12 | 10 / 10 | 2 / 2 |
| merge 1/3 > int4norm(mean) | 0.043 | 20 / 14 | 14 / 13 | 5 / 2 |
| merge 1/3 > int8 | 0.085 | 19 / 14 | 14 / 13 | 6 / 2 |
| lloyd2(7) | 0.0625 | 9 / 5 | 14 / 12 | 1 / 1 |
| binary | 0.031 | 6 / 2 | 14 / 12 | 1 / 0 |

- **Pages are robust to everything.** They are near ceiling for most codecs, which limits what
  they can tell apart.
- **On text and SciDocs, neither lloyd2(7) nor binary comes near int4norm(mean).** Merging to 1/3
  then int4norm(mean) is both smaller than lloyd2(7) and higher on 17 of 24 text units.
- **One collapse after merging.** Merging to 1/4 collapses on one LFM2-ColBERT-350M unit (0.509 with
  int8, 0.516 with int4norm(mean)). The other LFM2 unit and one answerai unit are also weak there
  (0.77 and 0.79).
- **Not recomputed.** The Pareto-frontier counts in `angles/frontier.md` were not recomputed by the
  verifier. They use point estimates, so near-ties count as dominance.

## What would test these

These are suggestions and have not been run. Each would need predictions written and committed
before it ran.

- **nsr as a screen.**
  - Compute nsr on models not yet measured with int4norm, from their geometry files alone.
  - Predict, before running, which need centring.
  - New datasets for the measured models, and models measured in the campaign but not in s17,
    are candidates.
- **Merging with a stored norm.** Record decoded-norm statistics after merging, to see whether
  merging is what makes storing the norm matter more.
- **EVIE TabFQuAD.** Store per-query ndcg@5 and the scores, to tell a centring cost from
  near-tied rankings.
- **Package use.** If int4norm(mean) became a package stage, it would be a new registered stage, so
  the byte-pinned files need not change. That needs a decision; nothing here makes it a default.
