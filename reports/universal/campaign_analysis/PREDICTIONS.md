# Predictions for the follow-up batches (written before they ran)

This file is committed together with the follow-up notebooks (`notebooks/kaggle_followup`,
batches `f1` to `f8`), before any of them ran. The commit's date is the record of that.

- **Source.** Each prediction comes from a pattern in `REPORT.md`, which was found post hoc on the
  campaign data.
- **How results are reported.** Each prediction is reported as held, failed or not testable (when
  the runs it needs did not complete). Thresholds and unit definitions are not changed after the
  results are seen. Anything analysed beyond this list is labelled exploratory.

## Definitions

- **Metric.** Retention is the pipeline's ndcg@5 over the float baseline of the same model and
  dataset, on all queries. The "lower bound" is the 95% query-bootstrap interval's lower end from
  `s13_wide_study.py`.
- **ndcg@5 for text too.** Text rows use ndcg@5 here, unlike the catalogue's ndcg@10, so that every
  unit is on one metric.
- **NFCorpus labels.** NFCorpus queries are short topic keywords with a median of 16 relevant
  documents each, mostly at grade 1. Any grade above 0 counts as relevant, as for SciFact. Its
  baseline will be lower than SciFact's, and that is part of what it tests.
- **Retrieval check.** Text models' `check_retriever` gate runs on SciFact pairs for every dataset.
  On NFCorpus pairs it fails even for colbertv2.0 (2 of 12 in a local check), because "the query
  ranks its own document first" is not the right answer when a keyword has many relevant
  documents.
- **Unit.** A unit is one model on one dataset.
- **New-dataset units (18):** 8 text models on NFCorpus (`f1`, `f2`), plus 5 image models on two
  page sets each (`f4`, `f5`): ViDoRe arXivQA and TabFQuAD.
- **Corpus-size units (4):** colbertv2.0 and answerai-colbert-small on SciFact with 10,000 FiQA
  distractors (`f3`), and colSmol-500M and colpali-v1.3 on DocVQA with 2,000 distractor pages
  (`f6`, `f7`). Each is compared with the same model's campaign run on the original corpus.
- **Plain-int4 degraded set.** These are the models whose campaign plain-int4 retention is below
  0.95: nemotron-colembed-vl 4b and 8b, LateOn, mLateOn, mxbai-edge-colbert 17m and 32m, and
  GTE-ModernColBERT.
- **Controls.** Reason-ModernColBERT, ColBERT-Zero, mxbai-colbert-large, colbert-ir/colbertv2.0,
  LFM2-ColBERT-350M, EVIE-4.5B, EVIE-8B and colpali-v1.3.

## A. Codecs on new datasets (18 new-dataset units)

- **P1.** `int8(per_vector)` has a lower bound of at least 0.95 on all 18 units, and point
  retention of at least 0.97 on all 18.
- **P2.** `int4(mean)` has point retention of at least 0.93 on all 18 units, and a lower bound of at
  least 0.95 on at least 16.
- **P3. Plain-int4 behaviour transfers across datasets.**
  - On NFCorpus, plain `int4` point retention is below 0.6 for LateOn and mxbai-edge-colbert-17m.
  - It is at least 0.9 for colbert-ir/colbertv2.0 and jina-colbert-v2.
  - It is at least 0.97 for all five image models on both page sets.
- **P4.** The median `binary` retention over the 8 NFCorpus units is lower than the median over
  the 10 image page-set units. That is, the text-side fragility seen on SciFact is not specific to
  SciFact.

## B. Token merging (new-dataset units)

- **P5.** `hierarchical_merge(0.25) > int8(per_vector)` has a lower bound of at least 0.95 on at
  least 8 of the 10 image page-set units.
- **P6.** Retention under `hierarchical_merge(0.1)` is below that under `hierarchical_merge(0.25)`
  on all 18 units.

## C. Metrics

- **P7.** On each new dataset (NFCorpus, arXivQA, TabFQuAD), over pipelines with compression of
  16x or more, mean recall@1 retention is below mean ndcg@5 retention.

## D. Selection (units where `s7b_selection_rules.py` ran)

- **P8.** Pooled over these units, the `calibrate()` default's held-out met rate is at least 0.95
  at each target (0.95, 0.97, 0.99). The point rule's met rate is below 0.80 at target 0.99.

## E. Corpus size (4 corpus-size units, against each model's campaign run)

- **P9.** `int8(per_vector)` and `int4(mean)` retention change by less than 0.01 in absolute terms
  between the original and the enlarged corpus, for each of the 4 models.
- **P10.** `binary` and `hierarchical_merge(0.25) > binary` retention are lower on the enlarged
  corpus than on the original, for each of the 4 models. This is a prediction of direction only.
- **P11.** Tiered retention (binary hot index, int8 rescoring of 50 candidates) is lower on the
  enlarged corpus than on the original, for each of the 4 models.

## F. Geometry (`f8`, plus the geometry files of every follow-up run)

The statistic is the median per-query-token Pearson correlation of plain `int4` scores
(`codecs.int4.score_pearson.median` in `geometry/<tag>.json`).

- **P12.** On the `f8` runs, this statistic separates the degraded set from the controls. The
  prediction holds if the AUC of "lower statistic means degraded" is at least 0.9.
- **P13. The two failure signatures from REPORT section 1.2.**
  - For nemotron 4b and 8b and LateOn, uncentred `peak_to_rms.median` is higher than for every
    control.
  - For mxbai-edge-colbert 17m and 32m and mLateOn, `doc_cos_to_mean.median` is higher than for
    every control.
- **P14.** Over the 18 new-dataset units, Spearman's rank correlation between the statistic and the
  measured plain-`int4` retention is at least 0.5.

## Not predicted

Exploratory only:
- projection results on the new page sets;
- the DocVQA/InfoVQA-type asymmetry;
- adaptive merging;
- anything about models not listed above.
