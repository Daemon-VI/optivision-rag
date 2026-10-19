# int4 norm predictions: status

Written by `scripts/int4norm_predictions.py` from the ingested results; do not edit by hand.
The predictions and their thresholds are in `PREDICTIONS_INT4NORM.md`; the run history is in `INT4NORM_RUN_NOTES.md`.

Batches in: `f9b_int4norm_text`, `f10b_int4norm_pages_128d`, `f11b_int4norm_pages_wide`, `f12b_text_scidocs`, `f13_text_scale_same_domain`.
Still out: none.

| Prediction | Status | Detail |
|---|---|---|
| Q1 | held | 1 of 1 units in so far meet it (need 1 of 1) | 1 of 1 units in so far meet it (need 1 of 1) |
| Q2 | held | 8 of 10 units in so far meet it (need 8 of 10); misses: mLateOn / nfcorpus, mxbai-edge-colbert-v0-17m / nfcorpus |
| Q3 | failed | 1 of 2 units in so far meet it (need 2 of 2); misses: LateOn / nfcorpus |
| Q4 | held | 24 of 24 units in so far meet it (need 23 of 24) |
| Q5 | held | Spearman(int4 norm-ratio sd, plain int4 retention) = -0.7226 over 24 of 24 units |
| Q6 | failed | 45 of 46 units in so far meet it (need 46 of 46); misses: colpali-v1.3 / docvqa_test_subsampled |
| Q7 | failed | 3 of 4 units in so far meet it (need 4 of 4); misses: nemotron-colembed-vl-4b-v2 / tabfquad_test_subsampled |
| Q8 | failed | 13 of 14 units in so far meet it (need 14 of 14); misses: EVIE-4.5B / tabfquad_test_subsampled |
| Q9 | held | 8 of 8 units in so far meet it (need 8 of 8) | 8 of 8 units in so far meet it (need 8 of 8) | 8 of 8 units in so far meet it (need 7 of 8) |
| Q10 | failed | colbertv2.0: baseline drop 0.0329 (>= 0.02), projection rows -0.0394 (<= -0.05), codec rows +0.0003 (>= -0.015) | answerai-colbert-small-v1: baseline drop 0.0324 (>= 0.02), projection rows -0.0401 (<= -0.05), codec rows -0.0109 (>= -0.015) |

## Units (s17)

Retention of ndcg@5 against the same model's float32 run; the lower end of the 95% interval in brackets.

| Unit | Batch | ndcg@5 baseline | int8 (lower) | int4 | int4norm | int4(mean) | int4norm(mean) (lower) | int4 norm-ratio sd |
|---|---|---|---|---|---|---|---|---|
| answerai-colbert-small-v1 / scifact | f9b_int4norm_text | 0.731 | 0.998 (0.992) | 0.899 | 0.980 | 0.990 | 0.991 (0.981) | 0.0110 |
| answerai-colbert-small-v1 / nfcorpus | f9b_int4norm_text | 0.413 | 0.995 (0.983) | 0.455 | 0.984 | 0.951 | 0.991 (0.980) | 0.0107 |
| LateOn / scifact | f9b_int4norm_text | 0.742 | 0.990 (0.969) | 0.282 | 0.870 | 0.948 | 0.988 (0.971) | 0.0197 |
| LateOn / nfcorpus | f9b_int4norm_text | 0.428 | 0.996 (0.985) | 0.638 | 0.901 | 0.989 | 0.993 (0.980) | 0.0191 |
| mLateOn / scifact | f9b_int4norm_text | 0.746 | 0.993 (0.986) | 0.752 | 0.968 | 1.001 | 0.995 (0.981) | 0.0121 |
| mLateOn / nfcorpus | f9b_int4norm_text | 0.422 | 1.001 (0.991) | 0.407 | 0.941 | 0.984 | 0.991 (0.977) | 0.0119 |
| mxbai-edge-colbert-v0-17m / scifact | f9b_int4norm_text | 0.718 | 1.001 (0.991) | 0.235 | 0.989 | 0.987 | 0.997 (0.989) | 0.0140 |
| mxbai-edge-colbert-v0-17m / nfcorpus | f9b_int4norm_text | 0.405 | 0.998 (0.988) | 0.590 | 0.944 | 0.987 | 0.991 (0.978) | 0.0143 |
| mxbai-edge-colbert-v0-32m / scifact | f9b_int4norm_text | 0.723 | 0.994 (0.986) | 0.780 | 0.982 | 0.988 | 0.996 (0.991) | 0.0132 |
| mxbai-edge-colbert-v0-32m / nfcorpus | f9b_int4norm_text | 0.405 | 1.004 (0.995) | 0.865 | 0.991 | 0.990 | 1.000 (0.987) | 0.0131 |
| GTE-ModernColBERT-v1 / scifact | f9b_int4norm_text | 0.745 | 1.002 (0.996) | 0.927 | 0.991 | 0.994 | 1.000 (0.994) | 0.0094 |
| GTE-ModernColBERT-v1 / nfcorpus | f9b_int4norm_text | 0.426 | 0.998 (0.994) | 0.966 | 0.993 | 0.999 | 0.993 (0.981) | 0.0094 |
| LFM2-ColBERT-350M / scifact | f9b_int4norm_text | 0.717 | 0.999 (0.996) | 0.980 | 0.990 | 1.004 | 0.997 (0.991) | 0.0099 |
| LFM2-ColBERT-350M / nfcorpus | f9b_int4norm_text | 0.401 | 1.003 (0.995) | 0.982 | 0.996 | 1.012 | 1.017 (1.003) | 0.0102 |
| colbertv2.0 / scifact | f9b_int4norm_text | 0.623 | 1.000 (1.000) | 1.005 | 1.004 | 1.001 | 1.000 (0.993) | 0.0100 |
| colbertv2.0 / nfcorpus | f9b_int4norm_text | 0.371 | 0.999 (0.996) | 0.999 | 0.999 | 0.997 | 0.998 (0.989) | 0.0100 |
| ColBERT-Zero / scifact | f9b_int4norm_text | 0.752 | 0.998 (0.993) | 1.000 | 0.999 | 0.992 | 0.990 (0.980) | 0.0099 |
| ColBERT-Zero / nfcorpus | f9b_int4norm_text | 0.425 | 1.000 (0.997) | 0.985 | 0.992 | 1.003 | 1.004 (0.994) | 0.0100 |
| Reason-ModernColBERT / scifact | f9b_int4norm_text | 0.723 | 1.001 (0.998) | 1.002 | 1.002 | 1.001 | 0.998 (0.991) | 0.0097 |
| Reason-ModernColBERT / nfcorpus | f9b_int4norm_text | 0.420 | 0.998 (0.995) | 1.006 | 1.003 | 0.997 | 0.999 (0.988) | 0.0096 |
| mxbai-colbert-large-v1 / scifact | f9b_int4norm_text | 0.602 | 0.998 (0.994) | 0.983 | 1.007 | 0.993 | 1.003 (0.990) | 0.0100 |
| mxbai-colbert-large-v1 / nfcorpus | f9b_int4norm_text | 0.342 | 0.998 (0.994) | 0.980 | 0.993 | 0.999 | 1.002 (0.987) | 0.0100 |
| jina-colbert-v2 / scifact | f9b_int4norm_text | 0.675 | 1.001 (1.000) | 0.978 | 0.982 | 1.002 | 0.999 (0.991) | 0.0100 |
| jina-colbert-v2 / nfcorpus | f9b_int4norm_text | 0.400 | 1.004 (0.998) | 0.996 | 0.999 | 1.010 | 1.016 (1.003) | 0.0100 |
| colpali-v1.3 / docvqa_test_subsampled | f10b_int4norm_pages_128d | 0.575 | 1.002 (0.997) | 1.003 | 0.991 | 1.001 | 1.015 (1.003) | 0.0101 |
| colpali-v1.3 / tabfquad_test_subsampled | f10b_int4norm_pages_128d | 0.506 | 1.000 (1.000) | 1.005 | 1.005 | 1.004 | 1.009 (1.000) | 0.0101 |
| colSmol-500M / docvqa_test_subsampled | f10b_int4norm_pages_128d | 0.575 | 1.001 (0.999) | 1.005 | 1.000 | 0.997 | 1.004 (0.996) | 0.0101 |
| colSmol-500M / tabfquad_test_subsampled | f10b_int4norm_pages_128d | 0.423 | 1.000 (1.000) | 1.006 | 1.006 | 0.996 | 1.000 (0.989) | 0.0102 |
| colqwen2.5-v0.2 / docvqa_test_subsampled | f10b_int4norm_pages_128d | 0.618 | 0.998 (0.995) | 1.003 | 1.003 | 0.998 | 0.999 (0.993) | 0.0100 |
| colqwen2.5-v0.2 / tabfquad_test_subsampled | f10b_int4norm_pages_128d | 0.530 | 1.000 (1.000) | 1.004 | 1.008 | 0.999 | 1.007 (1.000) | 0.0100 |
| nemotron-colembed-vl-4b-v2 / docvqa_test_subsampled | f11b_int4norm_pages_wide | 0.664 | 0.997 (0.986) | 0.009 | 0.409 | 0.999 | 1.002 (0.991) | 0.0199 |
| nemotron-colembed-vl-4b-v2 / tabfquad_test_subsampled | f11b_int4norm_pages_wide | 0.617 | 1.001 (0.995) | 0.032 | 0.743 | 1.001 | 1.003 (0.992) | 0.0180 |
| nemotron-colembed-vl-8b-v2 / docvqa_test_subsampled | f11b_int4norm_pages_wide | 0.673 | 0.989 (0.978) | 0.011 | 0.082 | 0.996 | 0.997 (0.988) | 0.0198 |
| nemotron-colembed-vl-8b-v2 / tabfquad_test_subsampled | f11b_int4norm_pages_wide | 0.612 | 1.008 (0.996) | 0.017 | 0.152 | 1.004 | 1.003 (0.993) | 0.0223 |
| EVIE-4.5B / docvqa_test_subsampled | f11b_int4norm_pages_wide | 0.673 | 1.000 (1.000) | 1.003 | 1.002 | 1.000 | 1.004 (0.999) | 0.0039 |
| EVIE-4.5B / tabfquad_test_subsampled | f11b_int4norm_pages_wide | 0.605 | 1.020 (0.998) | 1.008 | 1.015 | 0.966 | 0.959 (0.927) | 0.0040 |
| VultronRetrieverCore-Qwen3.5-4.5B / docvqa_test_subsampled | f11b_int4norm_pages_wide | 0.656 | 1.001 (1.000) | 0.997 | 0.997 | 1.002 | 1.001 (0.996) | 0.0073 |
| VultronRetrieverCore-Qwen3.5-4.5B / tabfquad_test_subsampled | f11b_int4norm_pages_wide | 0.602 | 1.000 (1.000) | 1.000 | 1.000 | 0.996 | 0.996 (0.989) | 0.0073 |
| answerai-colbert-small-v1 / scidocs | f12b_text_scidocs | 0.153 | 1.004 (0.994) | 0.885 | 0.990 | 0.979 | 0.995 (0.983) | 0.0107 |
| LateOn / scidocs | f12b_text_scidocs | 0.180 | 0.993 (0.979) | 0.134 | 0.822 | 0.934 | 0.997 (0.982) | 0.0196 |
| mLateOn / scidocs | f12b_text_scidocs | 0.169 | 1.004 (0.993) | 0.361 | 0.938 | 0.988 | 0.995 (0.981) | 0.0120 |
| mxbai-edge-colbert-v0-17m / scidocs | f12b_text_scidocs | 0.145 | 1.006 (0.993) | 0.172 | 0.983 | 0.962 | 0.998 (0.982) | 0.0140 |
| mxbai-edge-colbert-v0-32m / scidocs | f12b_text_scidocs | 0.138 | 0.993 (0.983) | 0.827 | 0.971 | 0.990 | 0.994 (0.980) | 0.0125 |
| GTE-ModernColBERT-v1 / scidocs | f12b_text_scidocs | 0.164 | 0.998 (0.992) | 0.904 | 0.994 | 0.991 | 0.999 (0.989) | 0.0093 |
| colbertv2.0 / scidocs | f12b_text_scidocs | 0.133 | 0.997 (0.991) | 0.987 | 0.988 | 1.000 | 0.998 (0.988) | 0.0101 |
| jina-colbert-v2 / scidocs | f12b_text_scidocs | 0.160 | 0.998 (0.995) | 0.988 | 0.992 | 0.983 | 0.984 (0.972) | 0.0099 |
