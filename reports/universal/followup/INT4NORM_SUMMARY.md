# int4 norm batches: summary

## What ran and how it was judged

- **Runs.** Five Kaggle batches, `f9b` to `f13b`, from `notebooks/kaggle_followup_int4norm`, on
  2026-10-19 at `248ae2b`. They ran `s17_int4_norm.py` only.
  - The first versions, `f9` to `f13`, ran the wrong study; `INT4NORM_RUN_NOTES.md` has the history.
- **What they measured.**
  - 46 units for Q1 to Q9: 24 text units, 14 page units and 8 SciDocs units.
  - 2 scale units (`f13b`), which no prediction reads.
- **Outcomes.** Every model loaded, passed `check_retriever` and completed s17. No unit is missing,
  so no prediction is "not testable".
- **How it was judged.** Against Q1 to Q10 in `PREDICTIONS_INT4NORM.md`, with the thresholds as
  written. `scripts/int4norm_predictions.py` scores them into `PREDICTIONS_INT4NORM_STATUS.md`.
  The labels `int4norm` and `int4norm(mean)` are the predictions' `int4(norm)` and
  `int4(mean, norm)`.
- **Result: 5 held, 5 failed.**
- **Evidence label.** The idea came from a post hoc analysis of one model (`analysis/REPORT.md`).
  This is a test of it on new units, with predictions written in advance. The timing of that file
  rests on its file time, not on a commit (`INT4NORM_RUN_NOTES.md`). It was not a registered
  experiment in the sense of `reports/research`.

## Held

| | Prediction | Result |
|---|---|---|
| Q1 | On answerai, plain int4 < 0.6 and `int4norm` ≥ 0.95 on NFCorpus; `int4norm` ≥ 0.98 on SciFact | NFCorpus 0.455 and 0.984; SciFact 0.980 |
| Q2 | `int4norm` ≥ 0.95 on at least 8 of the 10 norm-noise units | 8 of 10 (the minimum). The misses are mLateOn (0.941) and mxbai-edge-17m (0.944), both on NFCorpus |
| Q4 | `int4norm(mean)` lower bound ≥ 0.95 on at least 23 of 24 text units | 24 of 24; the lowest is 0.971 (LateOn SciFact) |
| Q5 | Spearman(plain int4 norm-ratio sd, plain int4 retention) ≤ −0.5 over the 24 text units | −0.723 |
| Q9 | On SciDocs: int8 lower bound ≥ 0.95 on all 8; every plain int4 below 0.90 is a flagged model; `int4norm(mean)` lower bound ≥ 0.95 on at least 7 | 8 of 8; yes (answerai, LateOn, mLateOn, both mxbai-edge); 8 of 8 |

## Failed

| | Prediction | Result | By how much |
|---|---|---|---|
| Q3 | On LateOn's 2 text units, `int4norm` stays below 0.9 and `int4norm(mean)` ≥ 0.95 | SciFact 0.870 and 0.988; NFCorpus **0.9006** and 0.993 | 0.0006 over the line on one unit |
| Q6 | `int4norm` ≥ plain int4 − 0.01, and `int4norm(mean)` ≥ `int4(mean)` − 0.01, on all 46 units | 45 of 46 for the first clause, 46 of 46 for the second. colpali-v1.3 on DocVQA: int4 1.003, `int4norm` **0.991** | −0.012 against −0.01 |
| Q7 | On nemotron's 4 units, plain int4 and `int4norm` both below 0.5, `int4norm(mean)` ≥ 0.97 | int4 0.009 to 0.032; `int4norm(mean)` 0.997 to 1.003; `int4norm` 0.082, 0.152, 0.409 and **0.743** (4b on TabFQuAD) | 0.243 over on one unit |
| Q8 | `int4norm(mean)` lower bound ≥ 0.95 on all 14 page units, EVIE-4.5B TabFQuAD included | 13 of 14. EVIE-4.5B TabFQuAD: **0.927** (point 0.959; `int4(mean)` 0.966, lower 0.929) | the named unit, 0.023 short |
| Q10 | Same-domain distractors: baseline drop ≥ 0.02, projection rows ≤ −0.05, codec rows ≥ −0.015 | baseline and codec parts held; projection rows −0.039 and −0.040 | scored from `f13`'s s13 rows; see `INT4NORM_RUN_NOTES.md` |

## What the failures say

- **Q3 and Q7: storing the norm did more than predicted on dominant-coordinate models.** It was
  expected to leave them broken. On nemotron-4b TabFQuAD it took plain int4 from 0.032 to 0.743,
  and on LateOn NFCorpus from 0.638 to 0.901. Centring is still what repairs them: with the mean
  removed first, all 6 of those units are at 0.988 or higher.
- **Q6: one unit got worse.** On colpali-v1.3 DocVQA, `int4norm` lost 0.012 against plain int4,
  where plain int4 had nothing to repair (1.003). Its interval (0.981 to 1.001) overlaps plain
  int4's (0.985 to 1.020). The centred clause held everywhere.
- **Q8: EVIE-4.5B on TabFQuAD stays the hard unit.** `int4(mean)` already missed the bar there in
  `f5` (0.929) and does again (0.929). Storing the norm did not help: 0.927. int8 per-vector keeps
  it (lower bound 0.998).

## Exploratory: across the 46 units

None of this was predicted; it is description, not evidence.

| | lower bound ≥ 0.95 | point ≥ 0.95 |
|---|---|---|
| int8 (per_vector) | 46 | — |
| `int4norm(mean)` | 45 | 46 (lowest 0.959, EVIE-4.5B TabFQuAD) |
| `int4(mean)` | 41 | — |
| plain `int4` | — | 25 |
| `int4norm` | — | 36 |

- **Centred.** `int4norm(mean)` minus `int4(mean)`: median +0.004, range −0.007 to +0.063.
- **After Ward merging to 1/4.** Point ≥ 0.95 on 37 units with `int4norm(mean)`, 35 with
  `int4(mean)`, 37 with int8.
- **Bytes.** `int4norm` and `int4norm(mean)` have the same code size as `int4` and `int4(mean)`:
  only the stored scale changes.
- **Not done.** `NormPreservingInt4Quantizer` stays study-only in `scripts/universal_study/`. Making
  it a package stage would mean a new stage, not a change to the byte-pinned files, and needs a
  decision first.

## Cost and speed

| Batch | Wall time | Encoding | s17 |
|---|---|---|---|
| `f9b` | 2.31 h | 1.27 h | 1.03 h |
| `f11b` | 3.38 h | 2.44 h | 0.91 h |
| `f10b` | 1.60 h | 1.45 h | 0.15 h |
| `f12b` | 3.82 h | 1.62 h | 2.20 h |
| `f13b` | 0.29 h | 0.10 h | 0.19 h |
| **Total** | **11.4 h** | | |

- **The GPU page max worked.** On SciDocs, s17 took 697 to 1,421 s per model for 21 rows: about 33
  to 68 s per row. On the old path, s13 averaged 358 s per row for answerai and 231 s for LateOn
  (over 49 and 54 rows of mixed cost).
- **Exploratory analysis of these results:** `int4norm_analysis/REPORT.md`.
- **The estimate.** 11.4 h is at the low end of the README's 11.5 to 26 h.
