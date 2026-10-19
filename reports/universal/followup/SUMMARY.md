# Follow-up batches: summary

## What ran and how it was judged

- **Runs.** Nine Kaggle batches (`f1` to `f8`, plus `f3b`) ran from `notebooks/kaggle_followup`,
  on 2026-10-17 and 2026-10-18.
- **What they measured.**
  - 18 new-dataset units: 8 text models on BEIR NFCorpus, and 5 image models on ViDoRe arXivQA and
    TabFQuAD.
  - 4 corpus-size units.
  - The vector geometry of 15 models.
- **Outcomes.** Every model loaded, passed `check_retriever` (10 to 12 of 12) and completed every
  study.
- **How it was judged.** Against the 14 predictions in
  `reports/universal/campaign_analysis/PREDICTIONS.md`, committed with the notebooks before any of
  them ran. `scripts/followup_predictions.py` scores them, with the thresholds as written, into
  `PREDICTIONS_STATUS.md`.
- **Result: 9 held, 5 failed.**
- **Evidence label.** The patterns behind the predictions were found post hoc on the campaign
  data. This is a test of them on new data, with predictions fixed in advance. It was not a
  registered experiment in the sense of `reports/research`.

## Held

| | Prediction | Result |
|---|---|---|
| P1 | int8 per-vector keeps lower bound ≥ 0.95 and point ≥ 0.97 on all 18 new-dataset units | 18 of 18 |
| P2 | int4(mean) point ≥ 0.93 on all 18, and lower bound ≥ 0.95 on at least 16 | 18 of 18, and 16 of 18 (the minimum). The misses are answerai on NFCorpus and EVIE-4.5B on TabFQuAD |
| P4 | binary is weaker on NFCorpus than on the page sets | median 0.856 against 0.995 |
| P5 | merge 0.25 then int8 keeps lower bound ≥ 0.95 on at least 8 of 10 page-set units | 9 of 10 |
| P8 | `calibrate()` default meets its target in ≥ 95% of held-out splits; the point rule meets 0.99 in < 80% | 0.989 / 0.989 / 0.972 at 0.95 / 0.97 / 0.99; the point rule 0.423, over 22 units |
| P9 | int8 and int4(mean) change by less than 0.01 on a corpus 3 to 5 times larger | largest change 0.009 (int4(mean) on colSmol) |
| P12 | the int4 score-correlation statistic separates the degraded set from the controls (AUC ≥ 0.9) | AUC 0.964 |
| P13 | uncentred peak-to-RMS is highest for nemotron 4b/8b and LateOn; doc cosine to the corpus mean is highest for mxbai-edge 17m/32m and mLateOn | both, against every control |
| P14 | the statistic correlates with measured plain-int4 retention (Spearman ≥ 0.5) | 0.854 |

## Failed

| | Prediction | Result |
|---|---|---|
| P3 | plain int4 below 0.6 for LateOn and mxbai-edge-17m on NFCorpus | LateOn 0.638. The other parts held: mxbai-edge-17m 0.590; colbertv2.0 and jina at or above 0.9; all 10 image units at or above 0.97 |
| P6 | merging to 10% always keeps less than merging to 25% | inverted on 3 of 18 units. The campaign already had 4 inversions in 80 units, so the prediction was stronger than its data |
| P7 | under compression of 16x or more, recall@1 drops more than ndcg@5 on each new dataset | arXivQA only. Not on TabFQuAD (0.914 against 0.898) or NFCorpus (0.526 against 0.525) |
| P10 | binary and merge-then-binary are lower on the enlarged corpus, for each of 4 models | held for colSmol, colpali and answerai; failed for colbertv2.0 |
| P11 | binary hot index with int8 rescoring is lower on the enlarged corpus, for each of 4 models | held for colSmol and colpali; failed for colbertv2.0 (1.000 to 1.000) and answerai (0.994 to 0.994) |

On P10 and P11:
- **SciFact plus FiQA.** The 10,000 FiQA documents barely competed with SciFact's: colbertv2.0's
  baseline moved from 0.623 to 0.621. That makes this a weak test of corpus size on text.
- **DocVQA plus 2,000 pages.** The extra pages did compete: baselines fell from 0.575 to 0.505
  (colpali) and 0.575 to 0.473 (colSmol). Both predictions held there.

## Caveats

- **The P12/P14 statistic is not independent of int4.** It is computed by encoding a sample with
  plain int4 and correlating its scores, so it is a cheap proxy for the codec's damage rather than
  an independent property of the vectors.
- **P13 is the independent geometric part.** The two failure signatures, a large shared offset
  (cosine to the corpus mean 0.94 to 0.98) and a few dominant coordinates (peak-to-RMS 6 to 41;
  second-moment share of the top coordinate 0.29 to 0.41 for LateOn and nemotron, against at most
  0.06 for the controls), both held.
- **Centring removes the uncentred peak-to-RMS gap.** For LateOn, centred it is 2.79, the same as
  the controls. For nemotron it falls from 28 to 41 down to 5. That fits int4(mean) repairing
  these models, but it is a description, not a tested mechanism.
- **Small geometry samples.** They used 1,000 documents (text) and 100 pages (images). The
  degraded set and the controls were fixed from campaign retention, before f8 ran.

## Not predicted (exploratory)

- **answerai-colbert-small-v1 contradicts the statistic.** Its plain int4 keeps 0.455 on NFCorpus
  and 0.899 on SciFact. Its geometry is the same on both: the statistic is 0.982 and 0.984, cosine
  to the mean 0.93 and 0.92, peak-to-RMS 2.6 and 2.7. Whatever makes it collapse on NFCorpus is
  not in these vector statistics.
  - One possibility, untested: NFCorpus's many near-tied relevant documents make the ranking
    sensitive to small score errors.
- **Token merging is dataset-sensitive for some text models.**
  - LFM2-ColBERT-350M keeps 0.509 / 0.194 at merge 0.25 / 0.1 on NFCorpus, against 0.755 / 0.560
    on SciFact.
  - jina-colbert-v2 keeps 0.548 at merge 0.1, against 0.785.
  - The image models keep at least 0.98 at merge 0.1 on both page sets.
- **Binary is very weak on mxbai-edge-colbert-v0-17m on NFCorpus (0.241).**
