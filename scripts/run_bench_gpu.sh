#!/usr/bin/env bash
# Unattended ColPali / ViDoRe benchmark for a rented GPU box.
#
# Written for RunPod or Vast.ai, but there is nothing host-specific in here: it
# needs Ubuntu, a CUDA-capable GPU with ~8 GB free, and network access. The
# Colab/Kaggle path is notebooks/vidore_colpali_bench.ipynb; this is the same
# run without a notebook UI.
#
#   bash scripts/run_bench_gpu.sh                    # all four splits
#   SPLITS="vidore/docvqa_test_subsampled" bash scripts/run_bench_gpu.sh
#
# Run it under tmux. An SSH drop kills the foreground process otherwise, and
# these runs are long enough that it will happen:
#
#   tmux new -s bench 'bash scripts/run_bench_gpu.sh 2>&1 | tee run.log'
#
# Results are archived after *every* split, not just at the end, because a
# preempted pod should not cost you the splits that already finished.
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO="${REPO:-https://github.com/Daemon-VI/optivision-rag.git}"
CHECKOUT="$WORKDIR/optivision"
LIMIT="${LIMIT:-500}"
SPLITS="${SPLITS:-vidore/docvqa_test_subsampled vidore/syntheticDocQA_energy_test vidore/infovqa_test_subsampled vidore/tabfquad_test_subsampled}"

# Keep the Hub cache on the persistent volume. The merged ColPali weights are a
# ~6 GB download; on an ephemeral container root you pay it again every restart.
export HF_HOME="${HF_HOME:-$WORKDIR/hf_cache}"
export HF_HUB_ETAG_TIMEOUT=10
export HF_HUB_DOWNLOAD_TIMEOUT=30
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

say() { printf '\n=== %s\n' "$*"; }

say "workdir $WORKDIR   hf cache $HF_HOME"
mkdir -p "$WORKDIR" "$HF_HOME"

# ---------------------------------------------------------------- code
if [ -d "$CHECKOUT/.git" ]; then
    say "updating $CHECKOUT"
    # reset --hard leaves untracked files alone, so data/ and the encode caches
    # survive a code update. Losing those means re-encoding the corpus.
    git -C "$CHECKOUT" fetch --depth 1 origin main
    git -C "$CHECKOUT" reset --hard FETCH_HEAD
else
    say "cloning into $CHECKOUT"
    git clone --depth 1 "$REPO" "$CHECKOUT"
fi
cd "$CHECKOUT"
say "at commit $(git rev-parse --short HEAD)"

# ---------------------------------------------------------------- deps
say "installing"
pip install -q -e ".[bench]"
pip install -q "colpali-engine>=0.3.10" "transformers>=4.46"

# PEFT's LoRA dispatcher version-gates on torchao and raises rather than
# declining when the installed version is below its floor. Nothing here is
# torchao-quantized. Harmless if it was never installed.
pip uninstall -y -q torchao || true

# ---------------------------------------------------------------- preflight
# Fail now, loudly, rather than after a 6 GB download and ten minutes of encoding.
python - <<'PYEOF'
import sys
from importlib.metadata import version

import torch

from optivision.config import Config

if not torch.cuda.is_available():
    sys.exit("no CUDA device visible - this run needs a GPU")

name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"gpu              {name}  {total:.1f} GB")
if total < 8:
    print("warning: ColPali in bfloat16 wants ~6 GB free; this card is tight")

checkpoint = Config.load("configs/colpali.yaml").encoder.model_name
if not checkpoint.endswith("-merged"):
    sys.exit(f"stale checkout: configs/colpali.yaml names {checkpoint}")

# Record what resolved. "Whatever the image shipped" is not a reproducibility
# section; this goes next to the commit hash in the paper.
for pkg in ("torch", "transformers", "peft", "colpali-engine", "datasets"):
    print(f"{pkg:<16} {version(pkg)}")
print(f"{'checkpoint':<16} {checkpoint}")
PYEOF

# ---------------------------------------------------------------- config
# Same override the notebook applies: exact brute-force index, so no ANN
# structure can contribute its own recall loss to the measurement.
python - <<'PYEOF'
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path("configs/colpali.yaml").read_text())
cfg["index"] = {"backend": "numpy", "path": "data/index/colpali", "on_disk": True}
cfg["encoder"]["batch_size"] = 4
cfg["encoder"]["max_pages_in_flight"] = 8
Path("configs/colpali_bench.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(yaml.safe_dump(cfg, sort_keys=False))
PYEOF

# ---------------------------------------------------------------- run
mkdir -p data/cache reports
for split in $SPLITS; do
    tag="${split##*/}"
    say "$split"

    optivision fetch-vidore --dataset "$split" --out "data/vidore_$tag" --limit "$LIMIT"
    optivision bench \
        "data/vidore_$tag/images" "data/vidore_$tag/queries.json" \
        -c configs/colpali_bench.yaml \
        --out "reports/colpali_$tag" \
        --sweep \
        --cache "data/cache/colpali_$tag.npz" \
        2>&1 | tee "reports/colpali_$tag.log"

    # Archive now. A pod that dies during split 3 must not cost you splits 1-2.
    tar czf "$WORKDIR/optivision_reports.tar.gz" reports
    say "archived $WORKDIR/optivision_reports.tar.gz after $tag"
done

say "done - copy $WORKDIR/optivision_reports.tar.gz off this box before terminating it"
ls -la "$WORKDIR/optivision_reports.tar.gz"
