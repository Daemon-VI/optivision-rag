# The int4 norm batches: what ran, and the rerun

This note goes with `PREDICTIONS_INT4NORM.md`. That file is unchanged, so its thresholds are the
ones written before any result was seen.

## The first run ran the wrong study

- **What happened.** The five batches `f9` to `f13` ran on Kaggle on 2026-10-18/19.
  - Each notebook clones `kaggle-campaign`, which was still at `0ea1c02`.
  - That commit has no `s17_int4_norm.py`, and its `measure_models.py` ignores the spec's `study`
    key.
  - So every batch ran the default `s13_wide_study.py`, plus geometry, and no s17.
- **What it gave.** 39 s13 units, ingested under the batch names: `f9_int4norm_text/` to
  `f13_text_scale_same_domain/`.
- **They reproduce.** 27 of these units were measured before, by the campaign or by `f1` to
  `f5`, with the same model and dataset.
  - All 27 match to 4 decimals: every row's ndcg@5 retention and the float32 baseline.
  - In this comparison, the second campaign folder named `colbertv2.0` belongs to
    `lightonai/colbertv2.0`, a different model, and was set aside.
- **Not measured.** All of these ended at the 11-hour session deadline unless stated otherwise:
  - `f9`: jina-colbert-v2 never started, and the s13 run of mxbai-colbert-large-v1 on NFCorpus was
    cut off.
  - `f12`: answerai and LateOn finished on SciDocs (293 and 208 min for s13).
    - mLateOn ran out of GPU memory while encoding at batch size 32.
    - mxbai-edge-17m's s13 run was cut off.
    - mxbai-edge-32m, GTE-ModernColBERT-v1, colbertv2.0 and jina-colbert-v2 never started.

## Timing of the predictions

- **When it was written.** The local file time of `PREDICTIONS_INT4NORM.md` is 2026-10-18
  12:07:05 (sha256 `8d80df6c…20d9eff2`).
  - The first results zip is from 22:26 that day, and the other four from 10:12–10:14 on
    2026-10-19.
  - The file was not committed before the run, so this rests on the file time, not on git.
- **What is no longer blind.** These parts have values in the ingested s13 files:
  - the `int4` part of Q1 and Q7;
  - Q5's retention side;
  - Q9's second clause;
  - Q10.

  Two sets of values had been looked at:
  - the `f13` rows;
  - the `dim` (projection-only) rows of every unit.

  No `int4` value outside `f13` has been read.
- **What is still blind.** Every value of `int4(norm)` and `int4(mean, norm)` (labelled
  `int4norm` and `int4norm(mean)`; see the rerun section). No run has
  produced them yet.

## Q10, scored from s13

- **Q10 is s13-only.** It reads `wide/` rows, so the `f13` results score it as written.
- **The comparisons.**
  - colbertv2.0 is compared with `campaign/colbert-ir--colbertv2.0`.
  - answerai is compared with `f3b`.
- **Projection rows.** These are the 16 rows with `project(d)`, d ≤ dim/4, excluding random
  projection.

| | colbertv2.0 | answerai | Threshold | |
|---|---|---|---|---|
| float32 ndcg@5, SciFact alone | 0.6227 | 0.7308 | | |
| with 10,000 TREC-COVID documents | 0.5898 | 0.6985 | | |
| baseline drop | 0.033 | 0.032 | ≥ 0.02 | held |
| projection rows, mean retention change | −0.039 | −0.040 | ≤ −0.05 | **failed** |
| codec rows, mean retention change | +0.000 | −0.011 | ≥ −0.015 | held |

- **Verdict.** **Q10 failed**: the projection part missed on both models.
- **Other readings give the same verdict.**
  - Excluding float32 from the codec group: +0.000 and −0.013.
  - Pooling both models: projection rows −0.040, codec rows −0.005.
- **The scorer.** `scripts/int4norm_predictions.py` scores Q10 this way, and Q1 to Q9 once `f9b`
  to `f12b` are in.
- **Exploratory, for comparison.** In `f3`, FiQA distractors barely moved the baseline: by 0.002
  and 0.000.
  - Yet they moved the projection rows more: by −0.112 and −0.065.
  - So same-domain distractors lowered the baseline but cost projection less than off-domain ones
    did.
  - Why is untested.

## The rerun: f9b to f13b

- **What runs.** The same units, with s17 only. The s13 results above are kept, so these
  batches have new names and ingest into their own folders.
- **How the units map.** The units the predictions call `f9` … `f12` are scored from
  `f9b` … `f12b`. The batch names in `PREDICTIONS_INT4NORM.md` refer to those.
- **Changes from the notebooks first written.**
  - **f12b.** mLateOn encodes at batch size 8.
  - **f13b.** It runs s17 only, since its s13 is done. No prediction depends on its s17 rows (Q6
    covers `f9` to `f12`), so it is exploratory.
  - **The stale-branch guard.** A notebook whose models name a study stops before any GPU work if
    the cloned branch does not have that study.
  - **The codec's name.** `PREDICTIONS_INT4NORM.md` describes `Int4Quantizer(norm="preserve")`,
    labelled `int4(norm)` and `int4(mean, norm)`.
    - That version changed `stages/quantize.py` and `compose.py`, which are byte-pinned for X1-X3,
      so it was never committed.
    - The same method is now `NormPreservingInt4Quantizer`, in
      `scripts/universal_study/int4norm_codec.py`, labelled `int4norm` and `int4norm(mean)`. It is
      not a package stage.
    - On the local check its s17 rows equal those of the first version exactly.
    - The scorer reads `int4norm` for `int4(norm)` and `int4norm(mean)` for `int4(mean, norm)`.
  - **Scoring.** With `OPTIVISION_SCORE_DEVICE=cuda`, s17 takes the per-page max on the GPU
    (`scripts/universal_study/device_scoring.py`). Nothing under `src/` changed.
    - Scores are bit-identical to the pinned device path (`tests/test_followup_study_helpers.py`), so this
      does not change any value a prediction reads.
    - It changes only how long a row takes. The old path spent about 650 s per SciDocs row
      copying similarity blocks back to the CPU.
    - The CUDA speed-up has not been measured yet; the step times in `run.json` will show it.
