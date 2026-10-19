# Follow-up measurements

Written by `scripts/make_kaggle_batches.py`; do not edit the notebooks by hand. Each notebook runs
on its own Kaggle session, with *GPU T4 x2* and *Internet on*, then *Save & Run All (Commit)*.
Download `results_<batch>.zip` from the version's *Output* tab.

These batches test the patterns in `reports/universal/campaign_analysis/REPORT.md`. The
predictions they are judged against were written first, in
`reports/universal/campaign_analysis/PREDICTIONS.md`. Results are ingested into
`reports/universal/followup/<batch>/`, and the campaign's record and catalogue do not change.

The estimates below come from the campaign's step times on the same models. The selection study
is skipped for a model when the session would run short.

| Order | Notebook | What it measures | Estimate |
|---|---|---|---|
| 1 | `batch_f8_geometry` | geometry of 15 models, encode only (P12, P13) | 1.5-2.5 h |
| 2 | `batch_f1_text_nfcorpus_a` | 4 text models on NFCorpus | 5-7 h |
| 3 | `batch_f4_pages_128d` | colSmol-500M, colpali-v1.3, colqwen2.5-v0.2 on arXivQA + TabFQuAD | 4.5-6 h |
| 4 | `batch_f6_scale_colsmol` | colSmol-500M, DocVQA plus 2,000 distractor pages | 4-5 h |
| 5 | `batch_f2_text_nfcorpus_b` | 4 more text models on NFCorpus | 6-8 h |
| 6 | `batch_f5_pages_wide` | VultronRetrieverCore (320-d) and EVIE-4.5B (2048-d) on the two page sets | 6-8 h |
| 7 | `batch_f7_scale_colpali` | colpali-v1.3, DocVQA plus 2,000 distractor pages | 4.5-5.5 h |
| 8 | `batch_f3_text_scale` | colbertv2.0 and answerai-colbert-small, SciFact plus 10,000 FiQA documents | 7-10 h |
| 9 | `batch_f3b_answerai_scifact` | answerai-colbert-small on SciFact alone, the original corpus f3 compares against | 1.5-2 h |

All nine together come to about 42-52 GPU hours (`f1` took 3.9 h, under its estimate). If the
account's weekly GPU quota is smaller, run them in the order above: each notebook stands alone,
and the first four test the most predictions per hour.

`f3b` is only needed with `f3`. P9 to P11 compare each enlarged corpus with the same model on the
original one, and answerai's earlier SciFact run came from an older script, without the rows they
compare. `f3b` was added after `f1` came back; the predictions did not change.

Progress against the predictions: `reports/universal/followup/PREDICTIONS_STATUS.md`, written by
`scripts/followup_predictions.py`.
