# Kaggle measurement campaign

Written by `scripts/ingest_campaign.py` from the batches' `results_<batch>.zip`; do not edit by hand.

Each model ran in a `notebooks/kaggle_batches` notebook on two Kaggle T4s: encode ViDoRe V1
DocVQA + InfoVQA (500 pages each; BEIR SciFact for text models) with the package's own
loader after `check_retriever` passed on 12 pairs, then the fixed-pipeline study
(`s13_wide_study.py`, in `wide/`) and, when the session had time, the selection study
(`s7b_selection_rules.py`, in `selection_rules/`). The figures are development measurements
on those datasets, not a guarantee for another corpus. The vectors were not kept.

14 of 18 models attempted are measured.

| model | batch | outcome | inference dtype | GPU hours | selection study |
|---|---|---|---|---|---|
| `vidore/colpali` | 01_colpali | measured | float32 | 1.3 | no |
| `vidore/colpali-v1.1` | 01_colpali | measured | float32 | 1.97 | yes |
| `vidore/colpali-v1.2` | 01_colpali | measured | float32 | 1.98 | yes |
| `vidore/colpali-v1.2-merged` | 01_colpali | measured | float32 | 1.96 | yes |
| `vidore/colpali-v1.3` | 01_colpali | measured | float32 | 2.0 | yes |
| `nomic-ai/colnomic-embed-multimodal-3b` | 02_colqwen_small | measured | float32 | 3.13 | yes |
| `vidore/colqwen2-v0.1` | 02_colqwen_small | measured | float32 | 1.85 | yes |
| `vidore/colqwen2.5-v0.1` | 02_colqwen_small | incomplete: encode_docvqa failed (exit 1) | float32 | 0.04 | no |
| `vidore/colSmol-500M` | 02_colqwen_small | measured | float32 | 1.04 | no |
| `vidore/colsmolvlm-v0.1` | 02_colqwen_small | measured | float32 | 2.17 | no |
| `Cognitive-Lab/ColNetraEmbed` | 03_colqwen_multilingual | incomplete: encode_docvqa failed (exit 1) | float32 | 0.02 | no |
| `Metric-AI/ColQwen2.5-3b-multilingual-v1.0` | 03_colqwen_multilingual | incomplete: encode_docvqa failed (exit 1) | float32 | 0.04 | no |
| `tsystems/colqwen2.5-3b-multilingual-v1.0` | 03_colqwen_multilingual | measured | float32 | 1.83 | yes |
| `tsystems/colqwen2.5-3b-multilingual-v1.0-merged` | 03_colqwen_multilingual | measured | float32 | 1.22 | no |
| `vidore/colqwen-omni-v0.1` | 03_colqwen_multilingual | measured | float32 | 3.76 | yes |
| `ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1` | 04_colqwen_7b | measured | float16 | 1.82 | yes |
| `Metric-AI/ColQwen2.5-7b-multilingual-v1.0` | 04_colqwen_7b | incomplete: encode_docvqa failed (exit 1) | float16 | 0.0 | no |
| `nomic-ai/colnomic-embed-multimodal-7b` | 04_colqwen_7b | measured | float16 | 1.57 | yes |

Why a run failed is in the model's `logs/` and in its registry evidence (`optivision models <id>`).
