# Follow-up measurements: int4 with a norm-preserving scale

Written by `scripts/make_kaggle_batches.py`; do not edit the notebooks by hand. Each notebook runs
on its own Kaggle session, with *GPU T4 x2* and *Internet on*, then *Save & Run All (Commit)*.
Download `results_<batch>.zip` from the version's *Output* tab.

## What they test

- **The idea.** `reports/universal/followup/analysis/REPORT.md`, section 1, found that plain
  int4's collapse on answerai-colbert-small-v1 is per-vector norm noise. Storing int4's float16
  scale so that each decoded vector keeps its norm removed it in a local re-encode, with no extra
  bytes. It is `NormPreservingInt4Quantizer`, labelled `int4norm` and `int4norm(mean)`; the
  predictions call these `int4(norm)` and `int4(mean, norm)`.
- **What the batches check.**
  - Whether that holds on other models and datasets.
  - Whether it hurts anywhere.
  - A third text dataset.
- **The predictions.** They are in `reports/universal/followup/PREDICTIONS_INT4NORM.md`.
- **Where results go.** They are ingested into `reports/universal/followup/<batch>/`, as for
  `f1` to `f8`.
- **Why the names end in `b`.** The first versions, `f9` to `f13`, cloned a branch without s17 and
  ran the s13 study instead. Their results are kept under those names.
  `reports/universal/followup/INT4NORM_RUN_NOTES.md` has the details and Q10, which those results
  score.

## How they run

- **The study.** `s17_int4_norm.py` runs the codecs (including the four int4 variants) and Ward
  merging to 1/3 and 1/4 followed by them. It also records decoded-norm statistics and per-query
  nDCG@10. Its results go to `int4norm/`.
- **No other study runs:** no s13 and no selection study.
- **The guard.** If the cloned branch has no s17, the notebook stops before using the GPU.

| Order | Notebook | What it measures | Estimate |
|---|---|---|---|
| 1 | `batch_f9b_int4norm_text` | 12 text models on SciFact and NFCorpus (Q1-Q6) | 2.5-7 h |
| 2 | `batch_f11b_int4norm_pages_wide` | nemotron 4b and 8b, EVIE-4.5B, VultronRetrieverCore on DocVQA and TabFQuAD (Q6-Q8) | 3-4.5 h |
| 3 | `batch_f10b_int4norm_pages_128d` | colpali-v1.3, colSmol-500M, colqwen2.5-v0.2 on DocVQA and TabFQuAD (Q6, Q8) | 2-2.5 h |
| 4 | `batch_f12b_text_scidocs` | 8 text models on BEIR SciDocs, 25,657 documents (Q6, Q9) | 3.5-11 h |
| 5 | `batch_f13b_text_scale_same_domain` | colbertv2.0 and answerai, SciFact plus 10,000 TREC-COVID documents (exploratory) | 0.5-1.2 h |

- **Where the estimates come from.** The encode times and s13 times of the first run.
- **The low end** assumes the GPU page max (`scripts/universal_study/device_scoring.py`) speeds scoring up as
  expected.
- **The high end** is the old scoring speed, at about 45% of s13 per unit.
- **`f12b`** is the one that cannot finish at the old speed: SciDocs took about 2 h per unit then.
  - Its models run in prediction order, flagged ones first.
  - A session that runs short loses the controls (colbertv2.0, jina-colbert-v2) last.
- **Order.** Each notebook stands alone. If the weekly GPU quota is short, run them in the order
  above. `f13b` tests no prediction.
