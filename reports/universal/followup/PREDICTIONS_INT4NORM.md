# Predictions for the int4 norm batches (written before they ran)

This file is committed together with the notebooks in `notebooks/kaggle_followup_int4norm`
(batches `f9` to `f13`), before any of them ran. The commit's date is the record of that.

- **Source.** The predictions come from `analysis/REPORT.md` (sections 1, 3, 4 and 5). That
  analysis was exploratory and post hoc. They also come from a local check of the package's
  `Int4Quantizer(norm="preserve")` on a CPU re-encode of answerai-colbert-small-v1.
  - That check gave plain int4 0.414 and 0.888 ndcg@10 retention on NFCorpus and SciFact.
  - `int4(norm)` gave 0.968 and 0.999.
  - `int4(mean, norm)` gave 0.998 and 0.999.
- **Development units.** answerai on SciFact and on NFCorpus are where the idea came from. They are
  scored like the others, but they are not independent evidence.
- **How results are reported.** Each prediction is reported as held, failed or not testable (when
  the runs it needs did not complete). Thresholds and unit definitions are not changed after the
  results are seen. Anything analysed beyond this list is labelled exploratory.

## What `norm="preserve"` is

- **What it changes.** int4 keeps its levels (-7..7) and its code size. Its one float16 per-vector
  scale is stored so that the decoded vector has the norm of the vector it codes.
  - With `center="mean"`, that is the decoded vector with the mean added back.
- **What it leaves alone.** Decoding is unchanged, and so is the default, `norm=None`. Its bytes
  are pinned by `tests/test_golden_formats.py`.
- **Labels.** In the study files the four variants are `int4`, `int4(norm)`, `int4(mean)` and
  `int4(mean, norm)`.

## Definitions

- **Metric.** As in `campaign_analysis/PREDICTIONS.md`: retention is ndcg@5 over the float32
  baseline of the same model and dataset, on all queries. The "lower bound" is the lower end of
  the 95% query-bootstrap interval.
  - The values come from `s17_int4_norm.py` (`int4norm/` in each model folder), and from
    `s13_wide_study.py` (`wide/`) where a prediction says so.
- **Unit.** A unit is one model on one dataset.

| Units | Batches | What they are |
|---|---|---|
| 24 text units | `f9` | 12 text models on SciFact and NFCorpus |
| 14 page units | `f10`, `f11` | 7 page models on ViDoRe DocVQA and TabFQuAD |
| 8 SciDocs units | `f12` | 8 text models on BEIR SciDocs, a dataset none of them was measured on |
| 2 scale units | `f13` | colbertv2.0 and answerai on SciFact plus 10,000 TREC-COVID documents |

- **Comparisons for the scale units.** Each is compared with the same model on SciFact alone: the
  campaign's colbert-ir/colbertv2.0 run, and `f3b` for answerai.
- **Dominant-coordinate models.** The models where the top coordinate's second-moment share is at
  least 0.25 in their earlier geometry files:
  - LateOn (0.29).
  - nemotron-colembed-vl-4b-v2 (0.30).
  - nemotron-colembed-vl-8b-v2 (0.41).
- **Norm-noise candidates.** The other models whose plain int4 lost retention earlier (campaign or
  `f1`/`f2` plain int4 below 0.96): answerai-colbert-small-v1, mLateOn, mxbai-edge-colbert 17m and
  32m, and GTE-ModernColBERT-v1. This gives 10 text units.
- **sm4.** The second-moment share of the top 4 coordinates, from each model's SciFact geometry
  file (`f8`, or `f1` for answerai). Above 0.15 counts as flagged.
  - Flagged in `f12`: answerai 0.20, LateOn 0.72, mLateOn 0.32, mxbai-edge-17m 0.35, mxbai-edge-32m
    0.31 and GTE 0.16.
  - Not flagged: colbertv2.0 0.07 and jina-colbert-v2 0.10.

## A. The fix on text (24 text units)

- **Q1. It reproduces on the Kaggle vectors.**
  - On answerai NFCorpus, plain `int4` point retention is below 0.6 and `int4(norm)` is at least
    0.95.
  - On answerai SciFact, `int4(norm)` is at least 0.98.
- **Q2. It repairs norm noise.** On the 10 norm-noise candidate units, `int4(norm)` point retention
  is at least 0.95 on at least 8.
- **Q3. It does not repair a dominant coordinate.** On LateOn's 2 text units, `int4(norm)` point
  retention stays below 0.9, while `int4(mean, norm)` is at least 0.95.
- **Q4. With centring, the text bar is met almost everywhere.** `int4(mean, norm)` has a lower
  bound of at least 0.95 on at least 23 of the 24 text units.
- **Q5. Plain int4's norm error predicts its loss.** Over the 24 text units, the Spearman
  correlation is at most -0.5 between:
  - the standard deviation of plain int4's decoded-norm ratio (`norms.int4.ratio_sd`); and
  - plain `int4` point retention.

## B. No harm (all 46 units of f9 to f12)

- **Q6.** `int4(norm)` point retention is at least plain `int4`'s minus 0.01 on every unit.
  `int4(mean, norm)` is at least `int4(mean)`'s minus 0.01 on every unit.

## C. Pages (14 page units)

- **Q7. nemotron.** On its 4 units, plain `int4` and `int4(norm)` both stay below 0.5, and
  `int4(mean, norm)` is at least 0.97.
- **Q8. The bar on pages.** `int4(mean, norm)` has a lower bound of at least 0.95 on all 14 page
  units, including EVIE-4.5B on TabFQuAD, where `int4(mean)` had 0.929 in `f5`.

## D. A third text dataset (8 SciDocs units)

- **Q9. The ladder and the flag hold out of sample.**
  - `int8(per_vector)` has a lower bound of at least 0.95 on all 8.
  - Every unit where plain `int4` point retention is below 0.90 belongs to a flagged model.
  - `int4(mean, norm)` has a lower bound of at least 0.95 on at least 7 of the 8.

## E. Distractors from the same domain (2 scale units)

- **Q10. Corpus size with competing documents.**
  - For both models, the float32 baseline ndcg@5 is at least 0.02 lower than on SciFact alone.
  - In `wide/`, the mean change in retention, against SciFact alone, is at most -0.05 for the rows
    that project to 25% of the dimension or less (`project(d)` with d ≤ dim/4, random projection
    excluded).
  - It is at least -0.015 for the codec-only rows (group `codec`).
