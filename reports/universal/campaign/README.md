# Kaggle measurement campaign

Written by `scripts/ingest_campaign.py` from the batches' `results_<batch>.zip`; do not edit by hand.

Each model ran in a `notebooks/kaggle_batches` notebook on two Kaggle T4s: encode ViDoRe V1
DocVQA + InfoVQA (500 pages each; BEIR SciFact for text models) with the package's own
loader after `check_retriever` passed on 12 pairs, then the fixed-pipeline study
(`s13_wide_study.py`, in `wide/`) and, when the session had time, the selection study
(`s7b_selection_rules.py`, in `selection_rules/`). The figures are development measurements
on those datasets, not a guarantee for another corpus. The vectors were not kept.

28 of 41 models attempted are measured.

| model | batch | outcome | inference dtype | GPU hours | selection study |
|---|---|---|---|---|---|
| `vidore/colpali` | 01_colpali | measured | float32 | 1.3 | no |
| `vidore/colpali-v1.1` | 01_colpali | measured | float32 | 1.97 | yes |
| `vidore/colpali-v1.2` | 01_colpali | measured | float32 | 1.98 | yes |
| `vidore/colpali-v1.2-merged` | 01_colpali | measured | float32 | 1.96 | yes |
| `vidore/colpali-v1.3` | 01_colpali | measured | float32 | 2.0 | yes |
| `nomic-ai/colnomic-embed-multimodal-3b` | 02_colqwen_small | measured | float32 | 3.13 | yes |
| `vidore/colqwen2-v0.1` | 02_colqwen_small | measured | float32 | 1.85 | yes |
| `vidore/colSmol-500M` | 02_colqwen_small | measured | float32 | 1.04 | no |
| `vidore/colsmolvlm-v0.1` | 02_colqwen_small | measured | float32 | 2.17 | no |
| `tsystems/colqwen2.5-3b-multilingual-v1.0` | 03_colqwen_multilingual | measured | float32 | 1.83 | yes |
| `tsystems/colqwen2.5-3b-multilingual-v1.0-merged` | 03_colqwen_multilingual | measured | float32 | 1.22 | no |
| `vidore/colqwen-omni-v0.1` | 03_colqwen_multilingual | measured | float32 | 3.76 | yes |
| `ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1` | 04_colqwen_7b | measured | float16 | 1.82 | yes |
| `Metric-AI/ColQwen2.5-7b-multilingual-v1.0` | 04_colqwen_7b | incomplete: encode_docvqa failed (exit 1) | float16 | 0.0 | no |
| `nomic-ai/colnomic-embed-multimodal-7b` | 04_colqwen_7b | measured | float16 | 1.57 | yes |
| `Hcompany/NeoMME-260M-Retriever-ST-late` | 05_st_small | incomplete: encode_docvqa failed (exit 1) | float32 | 0.01 | no |
| `Hcompany/NeoMME-800M-Retriever-ST-late` | 05_st_small | incomplete: encode_docvqa failed (exit 1) | float32 | 0.0 | no |
| `ModernVBERT/colmodernvbert` | 05_st_small | measured | float32 | 1.69 | yes |
| `vidore/colpali-v1.2-hf` | 05_st_small | measured | float16 | 1.46 | yes |
| `vidore/colpali-v1.3-hf` | 05_st_small | measured | float16 | 1.58 | yes |
| `vidore/colqwen2-v1.0-hf` | 05_st_small | measured | float16 | 1.31 | yes |
| `tencent/EVIE-4.5B` | 06_st_qwen3 | measured | float16 | 4.62 | yes |
| `VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1` | 06_st_qwen3 | measured | float16 | 1.93 | yes |
| `VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1` | 06_st_qwen3 | measured | float16 | 2.19 | yes |
| `webAI-Official/webAI-ColVec1.1-4b` | 06_st_qwen3 | check failed | float16 | 0.04 | no |
| `vultr/VultronRetrieverCore-Qwen3.5-4.5B` | 07_qwen35_vultr | measured | float32 | 3.91 | yes |
| `vultr/VultronRetrieverFlash-Qwen3.5-0.8B` | 07_qwen35_vultr | measured | float32 | 2.88 | yes |
| `vultr/VultronRetrieverPrime-Qwen3.5-8B` | 07_qwen35_vultr | measured | float16 | 1.14 | no |
| `jinaai/jina-embeddings-v4` | 08_remote_code | incomplete: encode_docvqa failed (exit 1) | float16 | 0.02 | no |
| `nvidia/llama-nemotron-colembed-vl-3b-v2` | 08_remote_code | incomplete: encode_docvqa failed (exit -9) | float32 | 2.3 | no |
| `TomoroAI/tomoro-colqwen3-embed-8b` | 08_remote_code | measured | float16 | 3.08 | yes |
| `webAI-Official/webAI-ColVec1-4b` | 08_remote_code | incomplete: encode_docvqa failed (exit 1) | float32 | 0.07 | no |
| `tencent/EVIE-8B` | 09_st_8b | incomplete: encode_docvqa failed (exit 1) | float16 | 0.05 | no |
| `webAI-Official/webAI-ColVec1.1-8b` | 09_st_8b | incomplete: encode_docvqa failed (exit 1) | float16 | 0.04 | no |
| `5CD-AI/Vintern-Embedding-1B` | 10_pinned_transformers | incomplete: encode_docvqa failed (exit 1) | float32 | 0.0 | no |
| `ibm-granite/granite-vision-3.3-2b-embedding` | 10_pinned_transformers | incomplete: encode_docvqa failed (exit 1) | float32 | 0.03 | no |
| `nvidia/llama-nemoretriever-colembed-1b-v1` | 10_pinned_transformers | incomplete: encode_docvqa failed (exit 1) | float32 | 0.0 | no |
| `nvidia/llama-nemoretriever-colembed-3b-v1` | 10_pinned_transformers | incomplete: encode_docvqa failed (exit 1) | float32 | 0.0 | no |
| `Cognitive-Lab/ColNetraEmbed` | 12_retry | measured | float32 | 1.46 | yes |
| `Metric-AI/ColQwen2.5-3b-multilingual-v1.0` | 12_retry | measured | float32 | 1.9 | yes |
| `vidore/colqwen2.5-v0.1` | 12_retry | measured | float32 | 1.9 | yes |

Why a run failed is in the model's `logs/` and in its registry evidence (`optivision models <id>`).
