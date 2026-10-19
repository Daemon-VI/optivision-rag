"""Write the Kaggle notebooks that measure the registry's unmeasured models, a batch per session.

    python scripts/make_kaggle_batches.py            # writes notebooks/kaggle_batches/

Each notebook clones the repository at ``BRANCH``, installs what its batch needs, writes the
batch file and runs ``scripts/measure_models.py`` within Kaggle's 12-hour limit, then zips the
results. Batches group models that share an environment and fit one session.
"""

from __future__ import annotations

import json
import pprint
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_batches"
BRANCH = "kaggle-campaign"

ST = "sentence-transformers[image]>=6.0.0"
F32 = {"dtype": "float32", "multi_gpu": True}
F16 = {"dtype": "float16", "multi_gpu": True}
ST16 = {"dtype": "float16", "batch_size": 2}

BATCHES: list[tuple[str, str, list[str], list[dict]]] = [
    ("01_colpali", "ColPali checkpoints (PaliGemma 3B, float32 over two T4s)", [], [
        {"model": m, **F32} for m in ("vidore/colpali", "vidore/colpali-v1.1", "vidore/colpali-v1.2",
                                      "vidore/colpali-v1.2-merged", "vidore/colpali-v1.3")]),
    ("02_colqwen_small", "ColQwen2 / ColQwen2.5 3B and the ColSmol family", [], [
        {"model": "vidore/colsmolvlm-v0.1", **F32}, {"model": "vidore/colSmol-500M", **F32},
        {"model": "vidore/colqwen2-v0.1", **F32}, {"model": "vidore/colqwen2.5-v0.1", **F32},
        {"model": "nomic-ai/colnomic-embed-multimodal-3b", **F32}]),
    ("03_colqwen_multilingual", "multilingual ColQwen2.5 3B, ColQwen-Omni, ColNetraEmbed", [], [
        {"model": "tsystems/colqwen2.5-3b-multilingual-v1.0-merged", **F32},
        {"model": "tsystems/colqwen2.5-3b-multilingual-v1.0", **F32},
        {"model": "Metric-AI/ColQwen2.5-3b-multilingual-v1.0", **F32},
        {"model": "vidore/colqwen-omni-v0.1", **F32},
        {"model": "Cognitive-Lab/ColNetraEmbed", **F32}]),
    ("04_colqwen_7b", "7B ColQwen2.5 models (float16 over two T4s)", [], [
        {"model": "nomic-ai/colnomic-embed-multimodal-7b", **F16},
        {"model": "Metric-AI/ColQwen2.5-7b-multilingual-v1.0", **F16},
        {"model": "ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1", **F16}]),
    ("05_st_small", "sentence-transformers image models up to 3B", [ST], [
        {"model": "Hcompany/NeoMME-260M-Retriever-ST-late", "dtype": "float32", "batch_size": 4},
        {"model": "Hcompany/NeoMME-800M-Retriever-ST-late", "dtype": "float32", "batch_size": 4},
        {"model": "ModernVBERT/colmodernvbert", "dtype": "float32", "batch_size": 4},
        {"model": "vidore/colpali-v1.2-hf", **ST16}, {"model": "vidore/colpali-v1.3-hf", **ST16},
        {"model": "vidore/colqwen2-v1.0-hf", **ST16}]),
    ("06_st_qwen3", "sentence-transformers Qwen3 / Qwen3.5 models, 2B to 4.5B (float16, one T4)", [ST], [
        {"model": "VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1", **ST16},
        {"model": "VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1", **ST16},
        {"model": "webAI-Official/webAI-ColVec1.1-4b", **ST16, "trust_remote_code": True},
        {"model": "tencent/EVIE-4.5B", **ST16}]),
    ("07_qwen35_vultr", "Vultron ColQwen3.5 models (colpali-engine)", [], [
        {"model": "vultr/VultronRetrieverFlash-Qwen3.5-0.8B", **F32},
        {"model": "vultr/VultronRetrieverCore-Qwen3.5-4.5B", **F32},
        {"model": "vultr/VultronRetrieverPrime-Qwen3.5-8B", **F16}]),
    ("08_remote_code", "remote-code models: Tomoro 8B, ColVec1 4B, Nemotron ColEmbed 3B v2, jina v4", [], [
        {"model": "webAI-Official/webAI-ColVec1-4b", **F32, "trust_remote_code": True, "revision": "latest"},
        {"model": "TomoroAI/tomoro-colqwen3-embed-8b", **F16, "trust_remote_code": True},
        {"model": "nvidia/llama-nemotron-colembed-vl-3b-v2", **F32, "trust_remote_code": True, "revision": "latest"},
        {"model": "jinaai/jina-embeddings-v4", **F16, "trust_remote_code": True, "revision": "latest",
         "pip": ["peft>=0.15.2"]}]),
    ("09_st_8b", "8B sentence-transformers models (float16 spread over two T4s)", [ST], [
        {"model": "tencent/EVIE-8B", **ST16, "multi_gpu": True},
        {"model": "webAI-Official/webAI-ColVec1.1-8b", **ST16, "multi_gpu": True, "trust_remote_code": True}]),
    ("10_pinned_transformers", "models whose cards pin an old transformers (expected to fail on T4)", [], [
        {"model": "ibm-granite/granite-vision-3.3-2b-embedding", **F32, "trust_remote_code": True,
         "revision": "latest"},
        {"model": "5CD-AI/Vintern-Embedding-1B", **F32, "trust_remote_code": True, "revision": "latest"},
        {"model": "nvidia/llama-nemoretriever-colembed-1b-v1", **F32, "trust_remote_code": True,
         "revision": "latest"},
        {"model": "nvidia/llama-nemoretriever-colembed-3b-v1", **F32, "trust_remote_code": True,
         "revision": "latest"}]),
    ("11_text", "the text models, on BEIR SciFact", [ST, "einops"], [
        {"model": m, "kind": "text"} for m in (
            "NeuML/colbert-bert-tiny", "mixedbread-ai/mxbai-edge-colbert-v0-17m",
            "mixedbread-ai/mxbai-edge-colbert-v0-32m", "lightonai/colbertv2.0", "colbert-ir/colbertv2.0",
            "lightonai/GTE-ModernColBERT-v1", "lightonai/Reason-ModernColBERT", "lightonai/ColBERT-Zero",
            "lightonai/LateOn", "lightonai/mLateOn", "mixedbread-ai/mxbai-colbert-large-v1",
            "LiquidAI/LFM2-ColBERT-350M")] + [
        {"model": "LiquidAI/LFM2.5-ColBERT-350M", "kind": "text", "trust_remote_code": True},
        {"model": "jinaai/jina-colbert-v2", "kind": "text", "trust_remote_code": True}]),
    # after batches 01-04 (2026-10-14): two ColQwen2.5 checkpoints ran out of memory at their
    # 16384-token page limit (now capped at the 768 their cards state) and ColNetraEmbed's
    # nested key names were refused (now mapped); see reports/universal/campaign
    ("12_retry", "re-runs after loader fixes: two ColQwen2.5 3B checkpoints, ColNetraEmbed", [], [
        {"model": "vidore/colqwen2.5-v0.1", **F32},
        {"model": "Metric-AI/ColQwen2.5-3b-multilingual-v1.0", **F32},
        {"model": "Cognitive-Lab/ColNetraEmbed", **F32}]),
    # after batches 05, 08-10 (2026-10-15): NeoMME needs transformers>=5.17 (Kaggle had 5.16.1),
    # sentence-transformers moved the two 8B models off their two-GPU split (now kept) and
    # ColVec1's from_pretrained returned no loading info (now checked against the checkpoint)
    ("13_retry", "re-runs: NeoMME (newer transformers), two 8B sentence-transformers models, ColVec1 4B",
     [ST, "transformers>=5.17"], [
        {"model": "Hcompany/NeoMME-260M-Retriever-ST-late", "dtype": "float32", "batch_size": 4},
        {"model": "Hcompany/NeoMME-800M-Retriever-ST-late", "dtype": "float32", "batch_size": 4},
        {"model": "webAI-Official/webAI-ColVec1-4b", **F32, "trust_remote_code": True, "revision": "latest"},
        {"model": "tencent/EVIE-8B", **ST16, "multi_gpu": True},
        {"model": "webAI-Official/webAI-ColVec1.1-8b", **ST16, "multi_gpu": True, "trust_remote_code": True}]),
    # after batches 06, 11 and 13 (2026-10-16): the text models batch 11 had no time for, and
    # lightonai/colbertv2.0, whose results colbert-ir/colbertv2.0 overwrote there (same short name);
    # NeoMME-800M one page at a time; EVIE-8B with less of its weights on GPU 0; ColVec1.1-4b in
    # float32 over two T4s (in float16 it failed check_retriever, 9 of 12; its card states bfloat16)
    # split into four notebooks to run side by side (one would need ~12-18 h against an 11 h session)
    ("14a_text_retry", "re-runs: three text models", [ST, "einops"], [
        {"model": "lightonai/colbertv2.0", "kind": "text"},
        {"model": "LiquidAI/LFM2.5-ColBERT-350M", "kind": "text", "trust_remote_code": True},
        {"model": "jinaai/jina-colbert-v2", "kind": "text", "trust_remote_code": True}]),
    ("14b_neomme_retry", "re-run: NeoMME 800M, one page at a time", [ST, "transformers>=5.17"], [
        {"model": "Hcompany/NeoMME-800M-Retriever-ST-late", "dtype": "float32", "batch_size": 1}]),
    ("14c_colvec_retry", "re-run: ColVec1.1 4B in float32 over two T4s", [ST], [
        {"model": "webAI-Official/webAI-ColVec1.1-4b", "dtype": "float32", "batch_size": 1, "multi_gpu": True,
         "trust_remote_code": True}]),
    ("14d_evie_retry", "re-run: EVIE 8B, float16 over two T4s, less of it on GPU 0", [ST], [
        {"model": "tencent/EVIE-8B", **ST16, "batch_size": 1, "multi_gpu": True}]),
    # after batch 14c (2026-10-16): ColVec1.1's own forward multiplies its projection head's
    # output (last GPU) by the attention mask (first GPU); the adapter now moves the head to the
    # input GPU, which also lets the 8B (float16, too large for one T4) try the two-GPU split
    ("15a_colvec4b_retry", "re-run: ColVec1.1 4B in float32 over two T4s, head on the input GPU", [ST], [
        {"model": "webAI-Official/webAI-ColVec1.1-4b", "dtype": "float32", "batch_size": 1, "multi_gpu": True,
         "trust_remote_code": True}]),
    ("15b_colvec8b_retry", "re-run: ColVec1.1 8B in float16 over two T4s, head on the input GPU", [ST], [
        {"model": "webAI-Official/webAI-ColVec1.1-8b", **ST16, "batch_size": 1, "multi_gpu": True,
         "trust_remote_code": True}]),
    # after batch 14d (2026-10-16): EVIE-8B passed check_retriever (11 of 12) but a page of ~8500
    # visual tokens needed 4.4 GiB for one attention call on GPU 0; its processor allows 16,384
    # tokens a page, and its card gives the evaluation protocol's 1,024-token budget, used here
    ("16_evie_retry", "re-run: EVIE 8B over two T4s at its card's 1,024-token evaluation page budget", [ST], [
        {"model": "tencent/EVIE-8B", **ST16, "batch_size": 1, "multi_gpu": True,
         "processor_kwargs": {"size": {"longest_edge": 1024 * 32 * 32, "shortest_edge": 65536}}}]),
]

#: Follow-up measurements of models the campaign already holds (reports/universal/campaign_analysis,
#: section 6; the predictions they test are in reports/universal/campaign_analysis/PREDICTIONS.md,
#: committed before they ran). Written to notebooks/kaggle_followup; their results are ingested
#: into reports/universal/followup/<batch>/ and do not change the campaign's record or catalogue.
NFCORPUS = {"kind": "text", "datasets": ["nfcorpus"]}
PAGESETS = {"datasets": ["vidore/arxivqa_test_subsampled", "vidore/tabfquad_test_subsampled"]}
SCALE_PAGES = {"datasets": ["vidore/docvqa_test_subsampled"], "distractor_count": 2000,
               "distractors": ["vidore/syntheticDocQA_energy_test", "vidore/syntheticDocQA_healthcare_industry_test"]}
SCALE_TEXT = {"kind": "text", "datasets": ["scifact"], "distractors": ["fiqa"], "distractor_count": 10000}
GEOMETRY_TEXT = {"kind": "text", "studies": False, "limit": 1000}
GEOMETRY_PAGES = {"datasets": ["vidore/docvqa_test_subsampled"], "studies": False, "limit": 100}
FOLLOWUP: list[tuple[str, str, list[str], list[dict]]] = [
    ("f1_text_nfcorpus_a", "text models on a second text dataset, BEIR NFCorpus (1 of 2)", [ST, "einops"], [
        {"model": m, **NFCORPUS} for m in ("colbert-ir/colbertv2.0", "answerdotai/answerai-colbert-small-v1",
                                           "lightonai/LateOn", "mixedbread-ai/mxbai-edge-colbert-v0-17m")]),
    ("f2_text_nfcorpus_b", "text models on a second text dataset, BEIR NFCorpus (2 of 2)", [ST, "einops"], [
        {"model": "lightonai/mLateOn", **NFCORPUS}, {"model": "lightonai/GTE-ModernColBERT-v1", **NFCORPUS},
        {"model": "jinaai/jina-colbert-v2", **NFCORPUS, "trust_remote_code": True},
        {"model": "LiquidAI/LFM2-ColBERT-350M", **NFCORPUS}]),
    ("f3_text_scale", "SciFact with 10,000 FiQA distractor documents (corpus 5,183 -> 15,183)", [ST], [
        {"model": m, **SCALE_TEXT} for m in ("colbert-ir/colbertv2.0", "answerdotai/answerai-colbert-small-v1")]),
    # the original corpus for f3's second model: answerai was measured before the campaign, with an
    # older script whose rows lack int4(mean) and the tiered index that P9 and P11 compare
    ("f3b_answerai_scifact", "SciFact without distractors, the original corpus f3 compares against", [ST], [
        {"model": "answerdotai/answerai-colbert-small-v1", "kind": "text", "datasets": ["scifact"]}]),
    ("f4_pages_128d", "image models on two other ViDoRe page sets (arXiv figures, French tables), 128-d", [], [
        {"model": m, **F32, **PAGESETS} for m in ("vidore/colSmol-500M", "vidore/colpali-v1.3",
                                                  "vidore/colqwen2.5-v0.2")]),
    ("f5_pages_wide", "image models on two other ViDoRe page sets, 320-d and 2048-d", [ST], [
        {"model": "vultr/VultronRetrieverCore-Qwen3.5-4.5B", **F32, **PAGESETS},
        {"model": "tencent/EVIE-4.5B", **ST16, **PAGESETS}]),
    ("f6_scale_colsmol", "DocVQA with 2,000 distractor pages (corpus 500 -> 2,500): ColSmol 500M", [], [
        {"model": "vidore/colSmol-500M", **F32, **SCALE_PAGES}]),
    ("f7_scale_colpali", "DocVQA with 2,000 distractor pages (corpus 500 -> 2,500): ColPali v1.3", [], [
        {"model": "vidore/colpali-v1.3", **F32, **SCALE_PAGES}]),
    ("f8_geometry", "vector geometry of models where plain int4 collapses, and of controls (encode only)",
     [ST, "einops"], [
        *({"model": m, **GEOMETRY_TEXT} for m in (
            "lightonai/LateOn", "lightonai/mLateOn", "mixedbread-ai/mxbai-edge-colbert-v0-17m",
            "mixedbread-ai/mxbai-edge-colbert-v0-32m", "lightonai/GTE-ModernColBERT-v1", "LiquidAI/LFM2-ColBERT-350M",
            "lightonai/Reason-ModernColBERT", "lightonai/ColBERT-Zero", "mixedbread-ai/mxbai-colbert-large-v1",
            "colbert-ir/colbertv2.0")),
        {"model": "nvidia/nemotron-colembed-vl-4b-v2", **F32, **GEOMETRY_PAGES},
        {"model": "nvidia/nemotron-colembed-vl-8b-v2", **F16, **GEOMETRY_PAGES},
        {"model": "tencent/EVIE-4.5B", **ST16, **GEOMETRY_PAGES},
        {"model": "tencent/EVIE-8B", **ST16, "batch_size": 1, "multi_gpu": True, **GEOMETRY_PAGES,
         "processor_kwargs": {"size": {"longest_edge": 1024 * 32 * 32, "shortest_edge": 65536}}},
        {"model": "vidore/colpali-v1.3", **F32, **GEOMETRY_PAGES}]),
]
OUT_FOLLOWUP = ROOT / "notebooks" / "kaggle_followup"

#: The int4 norm round (reports/universal/followup/analysis/REPORT.md, section 1): the same follow-up
#: layout, with the predictions in reports/universal/followup/PREDICTIONS_INT4NORM.md, committed before
#: these ran. s17 is the int4 norm study; no selection study.
#: f9 to f13 first ran on 2026-10-18 at 0ea1c02, a commit without s17, and so ran s13 only; those
#: results are kept under their names, and these "b" batches run s17 on the same units
#: (reports/universal/followup/INT4NORM_RUN_NOTES.md).
S17 = {"study": ["s17"]}
TWO_TEXT = {"kind": "text", "datasets": ["scifact", "nfcorpus"], **S17}
TWO_PAGES = {"datasets": ["vidore/docvqa_test_subsampled", "vidore/tabfquad_test_subsampled"], **S17}
SCIDOCS = {"kind": "text", "datasets": ["scidocs"], **S17}
SCALE_SAME_DOMAIN = {"kind": "text", "datasets": ["scifact"], "distractors": ["trec-covid"],
                     "distractor_count": 10000, **S17}
FOLLOWUP_INT4NORM: list[tuple[str, str, list[str], list[dict]]] = [
    ("f9b_int4norm_text", "int4 with a norm-preserving scale: 12 text models on SciFact and NFCorpus",
     [ST, "einops"], [
        {"model": m, **TWO_TEXT} for m in (
            "answerdotai/answerai-colbert-small-v1", "lightonai/LateOn", "lightonai/mLateOn",
            "mixedbread-ai/mxbai-edge-colbert-v0-17m", "mixedbread-ai/mxbai-edge-colbert-v0-32m",
            "lightonai/GTE-ModernColBERT-v1", "LiquidAI/LFM2-ColBERT-350M", "colbert-ir/colbertv2.0",
            "lightonai/ColBERT-Zero", "lightonai/Reason-ModernColBERT", "mixedbread-ai/mxbai-colbert-large-v1")]
        + [{"model": "jinaai/jina-colbert-v2", **TWO_TEXT, "trust_remote_code": True}]),
    ("f10b_int4norm_pages_128d", "int4 with a norm-preserving scale: 128-d page models on DocVQA and TabFQuAD", [], [
        {"model": m, **F32, **TWO_PAGES} for m in ("vidore/colpali-v1.3", "vidore/colSmol-500M",
                                                   "vidore/colqwen2.5-v0.2")]),
    ("f11b_int4norm_pages_wide", "int4 with a norm-preserving scale: wide page models (plain int4 collapses on nemotron)",
     [ST], [
        {"model": "nvidia/nemotron-colembed-vl-4b-v2", **F32, **TWO_PAGES},
        {"model": "nvidia/nemotron-colembed-vl-8b-v2", **F16, **TWO_PAGES},
        {"model": "tencent/EVIE-4.5B", **ST16, **TWO_PAGES},
        {"model": "vultr/VultronRetrieverCore-Qwen3.5-4.5B", **F32, **TWO_PAGES}]),
    ("f12b_text_scidocs", "a third text dataset, BEIR SciDocs (25,657 documents, short title queries)",
     [ST, "einops"], [
        {"model": "answerdotai/answerai-colbert-small-v1", **SCIDOCS}, {"model": "lightonai/LateOn", **SCIDOCS},
        {"model": "lightonai/mLateOn", **SCIDOCS, "batch_size": 8},  # f12: out of GPU memory at 32
        *[{"model": m, **SCIDOCS} for m in (
            "mixedbread-ai/mxbai-edge-colbert-v0-17m", "mixedbread-ai/mxbai-edge-colbert-v0-32m",
            "lightonai/GTE-ModernColBERT-v1", "colbert-ir/colbertv2.0")],
        {"model": "jinaai/jina-colbert-v2", **SCIDOCS, "trust_remote_code": True}]),
    ("f13b_text_scale_same_domain", "SciFact with 10,000 TREC-COVID documents, distractors from the same domain",
     [ST], [
        {"model": m, **SCALE_SAME_DOMAIN} for m in ("colbert-ir/colbertv2.0", "answerdotai/answerai-colbert-small-v1")]),
]
OUT_FOLLOWUP_INT4NORM = ROOT / "notebooks" / "kaggle_followup_int4norm"

#: in a batch that ran, but no batch can measure them (the reason is in the registry evidence):
#: a missing base model, remote code that fails under transformers 5, or flash_attn on a T4
UNLOADABLE = {"Metric-AI/ColQwen2.5-7b-multilingual-v1.0", "jinaai/jina-embeddings-v4",
              "ibm-granite/granite-vision-3.3-2b-embedding", "5CD-AI/Vintern-Embedding-1B",
              "nvidia/llama-nemoretriever-colembed-1b-v1", "nvidia/llama-nemoretriever-colembed-3b-v1"}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(name: str, title: str, packages: list[str], models: list[dict], selection: bool = True) -> dict:
    listing = "\n".join(f"- `{m['model']}`" for m in models)
    remote = any(m.get("trust_remote_code") for m in models)
    # f9 to f13 once cloned a branch without s17 and ran s13 instead; stop before any GPU work if so
    study_check = '''
studies = sorted({s for spec in BATCH for s in spec.get("study") or []})
known = open(f"{REPO}/scripts/measure_models.py", encoding="utf-8").read()
if missing := [s for s in studies if f'"{s}":' not in known]:
    raise SystemExit(f"the {BRANCH} branch on GitHub has no study {missing}; push it, then run again")
''' if any(m.get("study") for m in models) else ""
    cells = [
        md(f"""
# OptiVision measurement batch {name}: {title}

Measures {len(models)} models one after the other with the package's own loaders, then zips
the results. Each model is checked on 12 pages first (`check_retriever`); a model that fails
the check or a step is recorded and the batch goes on.

{listing}

**Run it:** *Settings -> Accelerator*: **GPU T4 x2**, *Internet*: **On**. Then
*Save Version -> Save & Run All (Commit)* and close the tab; it runs on its own for up to 12 h.
Download `results_{name}.zip` from the version's *Output* tab and give it to Claude.
""" + ("\n**Remote code:** some models here run Python from their own repositories inside this "
       "session (`trust_remote_code`).\n" if remote else "")),
        code(f'''
BRANCH = "{BRANCH}"
BATCH = {pprint.pformat(models, width=100, sort_dicts=False)}
EXTRA_PIP = {packages!r}
DEADLINE_HOURS = 11.0
SELECTION = {selection!r}   # also run the selection study for a model when the session has time to spare
'''),
        code('''
import json, os, shutil, subprocess, sys, time
if shutil.which("nvidia-smi") is None:
    raise SystemExit("NO GPU IN THIS SESSION. Open the notebook editor, Settings (right panel) -> Accelerator -> "
                     "GPU T4 x2, then Save Version -> Save & Run All (Commit) again. If GPU is greyed out, the "
                     "account needs phone verification (kaggle.com -> Settings -> Phone verification).")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
WORK = "/kaggle/working"
REPO = f"{WORK}/optivision-rag"
NAME = "''' + name + '''"
RESULTS = f"{WORK}/results_{NAME}"
os.environ["HF_HOME"] = "/tmp/hf"
!rm -rf {REPO} && git clone -q --depth 1 --branch {BRANCH} https://github.com/Daemon-VI/optivision-rag.git {REPO}
!cd {REPO} && git log --oneline -1''' + study_check + '''
!pip install -q -e "{REPO}[bench,merge,vlm]" 2>&1 | tail -2
!pip install -q -U colpali-engine 2>&1 | tail -1
for package in EXTRA_PIP:
    !pip install -q "{package}" 2>&1 | tail -1
# Kaggle preinstalls torchao 0.10; peft refuses to work while an incompatible torchao is importable
!pip uninstall -y -q torchao
os.makedirs(RESULTS, exist_ok=True)
json.dump(BATCH, open(f"{WORK}/batch.json", "w"), indent=1)
'''),
        code('''
cmd = [sys.executable, f"{REPO}/scripts/measure_models.py", f"{WORK}/batch.json", "--out", RESULTS,
       "--data", "/tmp/optdata", "--deadline-hours", str(DEADLINE_HOURS)] + (["--selection"] if SELECTION else [])
subprocess.run(cmd, check=False)
'''),
        code('''
!pip freeze > {RESULTS}/pip_freeze.txt
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > {RESULTS}/gpu.txt
!cd {REPO} && git rev-parse HEAD > {RESULTS}/commit.txt
!cat {RESULTS}/batch_summary.json
!cd {WORK} && rm -f results_{NAME}.zip && zip -q -r results_{NAME}.zip results_{NAME} && ls -la results_{NAME}.zip
'''),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seen = []
    for name, title, packages, models in BATCHES:
        seen += [m["model"] for m in models]
        (OUT / f"batch_{name}.ipynb").write_text(json.dumps(notebook(name, title, packages, models), indent=1) + "\n",
                                                 encoding="utf-8", newline="\n")
    sys_path = str(ROOT / "src")
    import sys

    sys.path.insert(0, sys_path)
    from optivision.adapters import MODELS

    unmeasured = {m.model_id for m in MODELS.values() if m.status != "measured"}
    retried = {m["model"] for name, _, _, models in BATCHES if name.endswith("_retry") for m in models}
    repeated = {m for m in seen if seen.count(m) > 1}
    missing, unknown = unmeasured - set(seen) - UNLOADABLE, set(seen) - set(MODELS)
    assert repeated <= retried, f"in two batches without a retry: {repeated - retried}"
    for name, _, _, models in BATCHES:  # measure_models.py keeps one folder per short name
        if name == "11_text":  # ran before this check: colbert-ir/colbertv2.0 overwrote lightonai/colbertv2.0
            continue
        shorts = [m["model"].split("/")[-1] for m in models]
        assert len(set(shorts)) == len(shorts), f"batch {name}: two models share a short name"
    assert not missing and not unknown, (missing, unknown)
    print(f"wrote {len(BATCHES)} notebooks covering {len(set(seen))} models to {OUT.relative_to(ROOT)}; "
          f"{len(unmeasured)} of the registry still unmeasured")

    for out, batches, selection in ((OUT_FOLLOWUP, FOLLOWUP, True), (OUT_FOLLOWUP_INT4NORM, FOLLOWUP_INT4NORM, False)):
        out.mkdir(parents=True, exist_ok=True)
        for name, title, packages, models in batches:
            shorts = [m["model"].split("/")[-1] for m in models]
            assert len(set(shorts)) == len(shorts), f"batch {name}: two models share a short name"
            assert {m["model"] for m in models} <= set(MODELS), f"batch {name}: a model is not in the registry"
            assert name[0] == "f" and name.split("_")[0][1:].rstrip("abc").isdigit(), name  # ingest_campaign.is_followup
            (out / f"batch_{name}.ipynb").write_text(json.dumps(
                notebook(name, title, packages, models, selection), indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {len(batches)} follow-up notebooks to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
