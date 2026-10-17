# Kaggle measurement campaign

Written by `scripts/ingest_campaign.py` from the batches' `results_<batch>.zip`; do not edit by hand.

Each model ran in a `notebooks/kaggle_batches` notebook on two Kaggle T4s: encode ViDoRe V1
DocVQA + InfoVQA (500 pages each; BEIR SciFact for text models) with the package's own
loader after `check_retriever` passed on 12 pairs, then the fixed-pipeline study
(`s13_wide_study.py`, in `wide/`) and, when the session had time, the selection study
(`s7b_selection_rules.py`, in `selection_rules/`). The figures are development measurements
on those datasets, not a guarantee for another corpus. The vectors were not kept.

47 of 55 models attempted are measured.

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
| `ModernVBERT/colmodernvbert` | 05_st_small | measured | float32 | 1.69 | yes |
| `vidore/colpali-v1.2-hf` | 05_st_small | measured | float16 | 1.46 | yes |
| `vidore/colpali-v1.3-hf` | 05_st_small | measured | float16 | 1.58 | yes |
| `vidore/colqwen2-v1.0-hf` | 05_st_small | measured | float16 | 1.31 | yes |
| `tencent/EVIE-4.5B` | 06_st_qwen3 | measured | float16 | 4.62 | yes |
| `VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1` | 06_st_qwen3 | measured | float16 | 1.93 | yes |
| `VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1` | 06_st_qwen3 | measured | float16 | 2.19 | yes |
| `vultr/VultronRetrieverCore-Qwen3.5-4.5B` | 07_qwen35_vultr | measured | float32 | 3.91 | yes |
| `vultr/VultronRetrieverFlash-Qwen3.5-0.8B` | 07_qwen35_vultr | measured | float32 | 2.88 | yes |
| `vultr/VultronRetrieverPrime-Qwen3.5-8B` | 07_qwen35_vultr | measured | float16 | 1.14 | no |
| `jinaai/jina-embeddings-v4` | 08_remote_code | incomplete: encode_docvqa failed (exit 1) | float16 | 0.02 | no |
| `nvidia/llama-nemotron-colembed-vl-3b-v2` | 08_remote_code | incomplete: encode_docvqa failed (exit -9) | float32 | 2.3 | no |
| `TomoroAI/tomoro-colqwen3-embed-8b` | 08_remote_code | measured | float16 | 3.08 | yes |
| `5CD-AI/Vintern-Embedding-1B` | 10_pinned_transformers | incomplete: encode_docvqa failed (exit 1) | float32 | 0.0 | no |
| `ibm-granite/granite-vision-3.3-2b-embedding` | 10_pinned_transformers | incomplete: encode_docvqa failed (exit 1) | float32 | 0.03 | no |
| `nvidia/llama-nemoretriever-colembed-1b-v1` | 10_pinned_transformers | incomplete: encode_docvqa failed (exit 1) | float32 | 0.0 | no |
| `nvidia/llama-nemoretriever-colembed-3b-v1` | 10_pinned_transformers | incomplete: encode_docvqa failed (exit 1) | float32 | 0.0 | no |
| `colbert-ir/colbertv2.0` | 11_text | measured | auto | 0.37 | no |
| `lightonai/ColBERT-Zero` | 11_text | measured | auto | 0.6 | no |
| `lightonai/GTE-ModernColBERT-v1` | 11_text | measured | auto | 0.43 | no |
| `lightonai/LateOn` | 11_text | measured | auto | 1.47 | yes |
| `lightonai/mLateOn` | 11_text | measured | auto | 0.74 | no |
| `lightonai/Reason-ModernColBERT` | 11_text | measured | auto | 0.59 | no |
| `LiquidAI/LFM2-ColBERT-350M` | 11_text | measured | auto | 0.85 | no |
| `mixedbread-ai/mxbai-colbert-large-v1` | 11_text | measured | auto | 1.72 | no |
| `mixedbread-ai/mxbai-edge-colbert-v0-17m` | 11_text | measured | auto | 1.25 | yes |
| `mixedbread-ai/mxbai-edge-colbert-v0-32m` | 11_text | measured | auto | 0.38 | no |
| `NeuML/colbert-bert-tiny` | 11_text | measured | auto | 0.37 | no |
| `Cognitive-Lab/ColNetraEmbed` | 12_retry | measured | float32 | 1.46 | yes |
| `Metric-AI/ColQwen2.5-3b-multilingual-v1.0` | 12_retry | measured | float32 | 1.9 | yes |
| `vidore/colqwen2.5-v0.1` | 12_retry | measured | float32 | 1.9 | yes |
| `Hcompany/NeoMME-260M-Retriever-ST-late` | 13_retry | measured | float32 | 3.01 | no |
| `webAI-Official/webAI-ColVec1-4b` | 13_retry | measured | float32 | 4.81 | yes |
| `jinaai/jina-colbert-v2` | 14a_text_retry | measured | auto | 1.81 | yes |
| `lightonai/colbertv2.0` | 14a_text_retry | measured | auto | 1.64 | yes |
| `LiquidAI/LFM2.5-ColBERT-350M` | 14a_text_retry | measured | auto | 2.08 | yes |
| `Hcompany/NeoMME-800M-Retriever-ST-late` | 14b_neomme_retry | measured | float32 | 4.52 | yes |
| `webAI-Official/webAI-ColVec1.1-4b` | 15a_colvec4b_retry | check failed | float32 | 0.06 | no |
| `webAI-Official/webAI-ColVec1.1-8b` | 15b_colvec8b_retry | measured | float16 | 5.79 | yes |
| `tencent/EVIE-8B` | 16_evie_retry | measured | float16 | 7.93 | yes |

Why a run failed is in the model's `logs/` and in its registry evidence (`optivision models <id>`).
