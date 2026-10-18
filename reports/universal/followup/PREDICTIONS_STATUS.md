# Follow-up predictions: status

Written by `scripts/followup_predictions.py` from the ingested follow-up results; do not edit by hand.
The predictions and their thresholds are in `reports/universal/campaign_analysis/PREDICTIONS.md`.

Batches in: `f1_text_nfcorpus_a`, `f2_text_nfcorpus_b`, `f3_text_scale`, `f4_pages_128d`, `f5_pages_wide`, `f7_scale_colpali`.
Still out: `f6_scale_colsmol`, `f3b_answerai_scifact`, `f8_geometry`.

| Prediction | Status | Detail |
|---|---|---|
| P1 | held | 18 of 18 units in so far meet it (need 18 of 18) |
| P2 | held | 18 of 18 units in so far meet it (need 18 of 18) | 16 of 18 units in so far meet it (need 16 of 18); misses: answerai-colbert-small-v1 / nfcorpus, EVIE-4.5B / tabfquad_test_subsampled |
| P3 | failed | 1 of 2 units in so far meet it (need 2 of 2); misses: LateOn / nfcorpus | 2 of 2 units in so far meet it (need 2 of 2) | 10 of 10 units in so far meet it (need 10 of 10) |
| P4 | held | NFCorpus median binary retention = 0.8556 over 8 of 8 units | page-set median binary retention = 0.9954 over 10 of 10 units |
| P5 | held | 9 of 10 units in so far meet it (need 8 of 10); misses: EVIE-4.5B / tabfquad_test_subsampled |
| P6 | failed | 15 of 18 units in so far meet it (need 18 of 18); misses: answerai-colbert-small-v1 / nfcorpus, colSmol-500M / tabfquad_test_subsampled, EVIE-4.5B / tabfquad_test_subsampled |
| P7 | failed | nfcorpus: (mean recall@1, mean ndcg@5, rows) = (0.5262, 0.5248, 222) over 8 of 8 units | arxivqa_test_subsampled: (mean recall@1, mean ndcg@5, rows) = (0.8536, 0.8819, 161) over 5 of 5 units | tabfquad_test_subsampled: (mean recall@1, mean ndcg@5, rows) = (0.9135, 0.8975, 161) over 5 of 5 units |
| P8 | pending | default met rate {0.95: 0.9881, 0.97: 0.9905, 0.99: 0.9707}, point met rate at 0.99 = 0.4262, over 21 units (interim) |
| P9 | pending | colbertv2.0: change {'int8(per_vector)': 0.0, 'int4(mean)': -0.0078} | answerai-colbert-small-v1: f3b_answerai_scifact pending | colSmol-500M: f6_scale_colsmol pending | colpali-v1.3: change {'int8(per_vector)': -0.0043, 'int4(mean)': -0.0022} |
| P10 | failed | colbertv2.0: (original, enlarged) {'binary': (1.0114, 1.0118), 'hierarchical_merge(0.25) > binary': (0.9523, 0.9526)} | answerai-colbert-small-v1: f3b_answerai_scifact pending | colSmol-500M: f6_scale_colsmol pending | colpali-v1.3: (original, enlarged) {'binary': (0.9644, 0.9554), 'hierarchical_merge(0.25) > binary': (0.9431, 0.9154)} |
| P11 | failed | colbertv2.0: tiered 1.0 -> 1.0 | answerai-colbert-small-v1: f3b_answerai_scifact pending | colSmol-500M: f6_scale_colsmol pending | colpali-v1.3: tiered 1.0037 -> 0.9978 |
| P12 | pending | f8 has not come back |
| P13 | pending | f8 has not come back |
| P14 | held | Spearman(statistic, plain int4 retention) = 0.8535 over 18 of 18 units |

## New-dataset units

Retention of ndcg@5 against the same model's float32 run; the lower end of the 95% interval in brackets.
The int4 statistic is the median per-query-token Pearson correlation of plain int4 scores.

| Unit | Batch | ndcg@5 baseline | int8 (lower) | int4 | int4(mean) (lower) | binary | merge 0.25 | merge 0.1 | int4 statistic |
|---|---|---|---|---|---|---|---|---|---|
| colbertv2.0 / nfcorpus | f1_text_nfcorpus_a | 0.371 | 0.999 (0.996) | 0.999 | 0.997 (0.988) | 0.971 | 0.956 | 0.877 | 0.993 |
| answerai-colbert-small-v1 / nfcorpus | f1_text_nfcorpus_a | 0.413 | 0.995 (0.983) | 0.455 | 0.951 (0.924) | 0.845 | 0.797 | 0.858 | 0.982 |
| LateOn / nfcorpus | f1_text_nfcorpus_a | 0.428 | 0.996 (0.985) | 0.638 | 0.989 (0.975) | 0.940 | 0.987 | 0.961 | 0.597 |
| mxbai-edge-colbert-v0-17m / nfcorpus | f1_text_nfcorpus_a | 0.405 | 0.998 (0.988) | 0.590 | 0.987 (0.964) | 0.241 | 0.981 | 0.918 | 0.830 |
| mLateOn / nfcorpus | f2_text_nfcorpus_b | 0.422 | 1.001 (0.991) | 0.407 | 0.984 (0.964) | 0.583 | 0.980 | 0.927 | 0.776 |
| GTE-ModernColBERT-v1 / nfcorpus | f2_text_nfcorpus_b | 0.426 | 0.998 (0.994) | 0.966 | 0.999 (0.985) | 0.715 | 0.981 | 0.954 | 0.965 |
| jina-colbert-v2 / nfcorpus | f2_text_nfcorpus_b | 0.400 | 1.004 (0.998) | 0.996 | 1.010 (0.996) | 0.906 | 0.986 | 0.548 | 0.993 |
| LFM2-ColBERT-350M / nfcorpus | f2_text_nfcorpus_b | 0.401 | 1.003 (0.995) | 0.982 | 1.012 (0.994) | 0.867 | 0.509 | 0.194 | 0.908 |
| colSmol-500M / arxivqa_test_subsampled | f4_pages_128d | 0.748 | 1.000 (1.000) | 0.995 | 1.002 (0.995) | 0.956 | 0.994 | 0.986 | 0.996 |
| colSmol-500M / tabfquad_test_subsampled | f4_pages_128d | 0.423 | 1.000 (1.000) | 1.006 | 0.996 (0.985) | 0.958 | 1.009 | 1.016 | 0.996 |
| colpali-v1.3 / arxivqa_test_subsampled | f4_pages_128d | 0.821 | 1.000 (1.000) | 1.000 | 1.001 (0.996) | 0.984 | 0.999 | 0.993 | 0.996 |
| colpali-v1.3 / tabfquad_test_subsampled | f4_pages_128d | 0.506 | 1.000 (1.000) | 1.005 | 1.004 (1.000) | 1.007 | 1.024 | 1.020 | 0.996 |
| colqwen2.5-v0.2 / arxivqa_test_subsampled | f4_pages_128d | 0.887 | 1.001 (1.000) | 0.999 | 0.999 (0.994) | 0.986 | 0.985 | 0.983 | 0.996 |
| colqwen2.5-v0.2 / tabfquad_test_subsampled | f4_pages_128d | 0.530 | 1.000 (1.000) | 1.004 | 0.999 (0.985) | 0.991 | 0.997 | 0.994 | 0.996 |
| VultronRetrieverCore-Qwen3.5-4.5B / arxivqa_test_subsampled | f5_pages_wide | 0.915 | 1.000 (1.000) | 0.998 | 1.000 (0.998) | 1.001 | 0.996 | 0.993 | 0.995 |
| VultronRetrieverCore-Qwen3.5-4.5B / tabfquad_test_subsampled | f5_pages_wide | 0.602 | 1.000 (1.000) | 1.000 | 0.996 (0.989) | 1.000 | 0.998 | 0.981 | 0.996 |
| EVIE-4.5B / arxivqa_test_subsampled | f5_pages_wide | 0.928 | 1.000 (1.000) | 0.999 | 0.999 (0.996) | 1.004 | 1.004 | 0.993 | 0.999 |
| EVIE-4.5B / tabfquad_test_subsampled | f5_pages_wide | 0.605 | 1.020 (0.998) | 1.008 | 0.966 (0.929) | 1.006 | 0.997 | 0.997 | 0.999 |
