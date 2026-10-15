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
]

#: in a batch that ran, but no batch can measure them (the reason is in the registry evidence)
UNLOADABLE = {"Metric-AI/ColQwen2.5-7b-multilingual-v1.0"}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(name: str, title: str, packages: list[str], models: list[dict]) -> dict:
    listing = "\n".join(f"- `{m['model']}`" for m in models)
    remote = any(m.get("trust_remote_code") for m in models)
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
SELECTION = True   # also run the selection study for a model when the session has time to spare
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
!cd {REPO} && git log --oneline -1
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
    assert not missing and not unknown, (missing, unknown)
    print(f"wrote {len(BATCHES)} notebooks covering {len(set(seen))} models to {OUT.relative_to(ROOT)}; "
          f"{len(unmeasured)} of the registry still unmeasured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
