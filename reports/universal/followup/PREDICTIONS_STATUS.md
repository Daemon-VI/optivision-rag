# Follow-up predictions: status

Written by `scripts/followup_predictions.py` from the ingested follow-up results; do not edit by hand.
The predictions and their thresholds are in `reports/universal/campaign_analysis/PREDICTIONS.md`.

Batches in: `f1_text_nfcorpus_a`.
Still out: `f2_text_nfcorpus_b`, `f3_text_scale`, `f4_pages_128d`, `f5_pages_wide`, `f6_scale_colsmol`, `f7_scale_colpali`, `f3b_answerai_scifact`, `f8_geometry`.

| Prediction | Status | Detail |
|---|---|---|
| P1 | pending | 4 of 4 units in so far meet it (need 18 of 18) |
| P2 | pending | 4 of 4 units in so far meet it (need 18 of 18) | 3 of 4 units in so far meet it (need 16 of 18); misses: answerai-colbert-small-v1 / nfcorpus |
| P3 | failed | 1 of 2 units in so far meet it (need 2 of 2); misses: LateOn / nfcorpus | 1 of 1 units in so far meet it (need 2 of 2) | 0 of 0 units in so far meet it (need 10 of 10) |
| P4 | pending | NFCorpus median binary retention = 0.8925 over 4 of 8 units (interim) | no unit in yet |
| P5 | pending | 0 of 0 units in so far meet it (need 8 of 10) |
| P6 | failed | 3 of 4 units in so far meet it (need 18 of 18); misses: answerai-colbert-small-v1 / nfcorpus |
| P7 | pending | nfcorpus: (mean recall@1, mean ndcg@5, rows) = (0.5181, 0.5218, 110) over 4 of 8 units (interim) | no unit in yet | no unit in yet |
| P8 | pending | default met rate {0.95: 0.9875, 0.97: 0.9875, 0.99: 1.0}, point met rate at 0.99 = 0.2, over 4 units (interim) |
| P9 | pending | colbertv2.0: f3_text_scale pending | answerai-colbert-small-v1: f3_text_scale pending, f3b_answerai_scifact pending | colSmol-500M: f6_scale_colsmol pending | colpali-v1.3: f7_scale_colpali pending |
| P10 | pending | colbertv2.0: f3_text_scale pending | answerai-colbert-small-v1: f3_text_scale pending, f3b_answerai_scifact pending | colSmol-500M: f6_scale_colsmol pending | colpali-v1.3: f7_scale_colpali pending |
| P11 | pending | colbertv2.0: f3_text_scale pending | answerai-colbert-small-v1: f3_text_scale pending, f3b_answerai_scifact pending | colSmol-500M: f6_scale_colsmol pending | colpali-v1.3: f7_scale_colpali pending |
| P12 | pending | f8 has not come back |
| P13 | pending | f8 has not come back |
| P14 | pending | Spearman(statistic, plain int4 retention) = 0.2 over 4 of 18 units (interim) |

## New-dataset units

Retention of ndcg@5 against the same model's float32 run; the lower end of the 95% interval in brackets.
The int4 statistic is the median per-query-token Pearson correlation of plain int4 scores.

| Unit | Batch | ndcg@5 baseline | int8 (lower) | int4 | int4(mean) (lower) | binary | merge 0.25 | merge 0.1 | int4 statistic |
|---|---|---|---|---|---|---|---|---|---|
| colbertv2.0 / nfcorpus | f1_text_nfcorpus_a | 0.371 | 0.999 (0.996) | 0.999 | 0.997 (0.988) | 0.971 | 0.956 | 0.877 | 0.993 |
| answerai-colbert-small-v1 / nfcorpus | f1_text_nfcorpus_a | 0.413 | 0.995 (0.983) | 0.455 | 0.951 (0.924) | 0.845 | 0.797 | 0.858 | 0.982 |
| LateOn / nfcorpus | f1_text_nfcorpus_a | 0.428 | 0.996 (0.985) | 0.638 | 0.989 (0.975) | 0.940 | 0.987 | 0.961 | 0.597 |
| mxbai-edge-colbert-v0-17m / nfcorpus | f1_text_nfcorpus_a | 0.405 | 0.998 (0.988) | 0.590 | 0.987 (0.964) | 0.241 | 0.981 | 0.918 | 0.830 |
| mLateOn / nfcorpus | f2_text_nfcorpus_b | pending | | | | | | | |
| GTE-ModernColBERT-v1 / nfcorpus | f2_text_nfcorpus_b | pending | | | | | | | |
| jina-colbert-v2 / nfcorpus | f2_text_nfcorpus_b | pending | | | | | | | |
| LFM2-ColBERT-350M / nfcorpus | f2_text_nfcorpus_b | pending | | | | | | | |
| colSmol-500M / arxivqa_test_subsampled | f4_pages_128d | pending | | | | | | | |
| colSmol-500M / tabfquad_test_subsampled | f4_pages_128d | pending | | | | | | | |
| colpali-v1.3 / arxivqa_test_subsampled | f4_pages_128d | pending | | | | | | | |
| colpali-v1.3 / tabfquad_test_subsampled | f4_pages_128d | pending | | | | | | | |
| colqwen2.5-v0.2 / arxivqa_test_subsampled | f4_pages_128d | pending | | | | | | | |
| colqwen2.5-v0.2 / tabfquad_test_subsampled | f4_pages_128d | pending | | | | | | | |
| VultronRetrieverCore-Qwen3.5-4.5B / arxivqa_test_subsampled | f5_pages_wide | pending | | | | | | | |
| VultronRetrieverCore-Qwen3.5-4.5B / tabfquad_test_subsampled | f5_pages_wide | pending | | | | | | | |
| EVIE-4.5B / arxivqa_test_subsampled | f5_pages_wide | pending | | | | | | | |
| EVIE-4.5B / tabfquad_test_subsampled | f5_pages_wide | pending | | | | | | | |
