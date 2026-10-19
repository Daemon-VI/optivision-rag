# int4 norm predictions: status

Written by `scripts/int4norm_predictions.py` from the ingested results; do not edit by hand.
The predictions and their thresholds are in `PREDICTIONS_INT4NORM.md`; the run history is in `INT4NORM_RUN_NOTES.md`.

Batches in: `f13_text_scale_same_domain`.
Still out: `f9b_int4norm_text`, `f10b_int4norm_pages_128d`, `f11b_int4norm_pages_wide`, `f12b_text_scidocs`.

| Prediction | Status | Detail |
|---|---|---|
| Q1 | pending | 0 of 0 units in so far meet it (need 1 of 1) | 0 of 0 units in so far meet it (need 1 of 1) |
| Q2 | pending | 0 of 0 units in so far meet it (need 8 of 10) |
| Q3 | pending | 0 of 0 units in so far meet it (need 2 of 2) |
| Q4 | pending | 0 of 0 units in so far meet it (need 23 of 24) |
| Q5 | pending | no unit in yet |
| Q6 | pending | 0 of 0 units in so far meet it (need 46 of 46) |
| Q7 | pending | 0 of 0 units in so far meet it (need 4 of 4) |
| Q8 | pending | 0 of 0 units in so far meet it (need 14 of 14) |
| Q9 | pending | 0 of 0 units in so far meet it (need 8 of 8) | 0 of 0 units in so far meet it (need 8 of 8) | 0 of 0 units in so far meet it (need 7 of 8) |
| Q10 | failed | colbertv2.0: baseline drop 0.0329 (>= 0.02), projection rows -0.0394 (<= -0.05), codec rows +0.0003 (>= -0.015) | answerai-colbert-small-v1: baseline drop 0.0324 (>= 0.02), projection rows -0.0401 (<= -0.05), codec rows -0.0109 (>= -0.015) |

## Units (s17)

Retention of ndcg@5 against the same model's float32 run; the lower end of the 95% interval in brackets.

| Unit | Batch | ndcg@5 baseline | int8 (lower) | int4 | int4norm | int4(mean) | int4norm(mean) (lower) | int4 norm-ratio sd |
|---|---|---|---|---|---|---|---|---|
| answerai-colbert-small-v1 / scifact | f9b_int4norm_text | pending | | | | | | |
| answerai-colbert-small-v1 / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| LateOn / scifact | f9b_int4norm_text | pending | | | | | | |
| LateOn / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| mLateOn / scifact | f9b_int4norm_text | pending | | | | | | |
| mLateOn / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| mxbai-edge-colbert-v0-17m / scifact | f9b_int4norm_text | pending | | | | | | |
| mxbai-edge-colbert-v0-17m / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| mxbai-edge-colbert-v0-32m / scifact | f9b_int4norm_text | pending | | | | | | |
| mxbai-edge-colbert-v0-32m / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| GTE-ModernColBERT-v1 / scifact | f9b_int4norm_text | pending | | | | | | |
| GTE-ModernColBERT-v1 / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| LFM2-ColBERT-350M / scifact | f9b_int4norm_text | pending | | | | | | |
| LFM2-ColBERT-350M / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| colbertv2.0 / scifact | f9b_int4norm_text | pending | | | | | | |
| colbertv2.0 / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| ColBERT-Zero / scifact | f9b_int4norm_text | pending | | | | | | |
| ColBERT-Zero / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| Reason-ModernColBERT / scifact | f9b_int4norm_text | pending | | | | | | |
| Reason-ModernColBERT / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| mxbai-colbert-large-v1 / scifact | f9b_int4norm_text | pending | | | | | | |
| mxbai-colbert-large-v1 / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| jina-colbert-v2 / scifact | f9b_int4norm_text | pending | | | | | | |
| jina-colbert-v2 / nfcorpus | f9b_int4norm_text | pending | | | | | | |
| colpali-v1.3 / docvqa_test_subsampled | f10b_int4norm_pages_128d | pending | | | | | | |
| colpali-v1.3 / tabfquad_test_subsampled | f10b_int4norm_pages_128d | pending | | | | | | |
| colSmol-500M / docvqa_test_subsampled | f10b_int4norm_pages_128d | pending | | | | | | |
| colSmol-500M / tabfquad_test_subsampled | f10b_int4norm_pages_128d | pending | | | | | | |
| colqwen2.5-v0.2 / docvqa_test_subsampled | f10b_int4norm_pages_128d | pending | | | | | | |
| colqwen2.5-v0.2 / tabfquad_test_subsampled | f10b_int4norm_pages_128d | pending | | | | | | |
| nemotron-colembed-vl-4b-v2 / docvqa_test_subsampled | f11b_int4norm_pages_wide | pending | | | | | | |
| nemotron-colembed-vl-4b-v2 / tabfquad_test_subsampled | f11b_int4norm_pages_wide | pending | | | | | | |
| nemotron-colembed-vl-8b-v2 / docvqa_test_subsampled | f11b_int4norm_pages_wide | pending | | | | | | |
| nemotron-colembed-vl-8b-v2 / tabfquad_test_subsampled | f11b_int4norm_pages_wide | pending | | | | | | |
| EVIE-4.5B / docvqa_test_subsampled | f11b_int4norm_pages_wide | pending | | | | | | |
| EVIE-4.5B / tabfquad_test_subsampled | f11b_int4norm_pages_wide | pending | | | | | | |
| VultronRetrieverCore-Qwen3.5-4.5B / docvqa_test_subsampled | f11b_int4norm_pages_wide | pending | | | | | | |
| VultronRetrieverCore-Qwen3.5-4.5B / tabfquad_test_subsampled | f11b_int4norm_pages_wide | pending | | | | | | |
| answerai-colbert-small-v1 / scidocs | f12b_text_scidocs | pending | | | | | | |
| LateOn / scidocs | f12b_text_scidocs | pending | | | | | | |
| mLateOn / scidocs | f12b_text_scidocs | pending | | | | | | |
| mxbai-edge-colbert-v0-17m / scidocs | f12b_text_scidocs | pending | | | | | | |
| mxbai-edge-colbert-v0-32m / scidocs | f12b_text_scidocs | pending | | | | | | |
| GTE-ModernColBERT-v1 / scidocs | f12b_text_scidocs | pending | | | | | | |
| colbertv2.0 / scidocs | f12b_text_scidocs | pending | | | | | | |
| jina-colbert-v2 / scidocs | f12b_text_scidocs | pending | | | | | | |
