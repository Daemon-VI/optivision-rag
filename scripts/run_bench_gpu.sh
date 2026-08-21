#!/usr/bin/env bash
# Unattended ColPali / ViDoRe benchmark for a rented GPU box.
#
# Written for RunPod or Vast.ai, but there is nothing host-specific in here: it
# needs Ubuntu, a CUDA-capable GPU with ~8 GB free, and network access. The
# Colab/Kaggle path is notebooks/vidore_colpali_bench.ipynb; this is the same
# run without a notebook UI.
#
#   bash scripts/run_bench_gpu.sh                    # E2: all four ViDoRe splits
#   SPLITS="vidore/docvqa_test_subsampled" bash scripts/run_bench_gpu.sh
#   MODE=generated bash scripts/run_bench_gpu.sh     # E3: the E1 corpus, ColPali
#   KEEP_CACHE=1 bash scripts/run_bench_gpu.sh       # also ship the encode caches
#   CODEBOOK=1 bash scripts/run_bench_gpu.sh         # add the Stage-II rows
#   LADDER=1 bash scripts/run_bench_gpu.sh           # codec ladder + geometry on each cache
#   MODE=generated CODE_SCALE=3 bash scripts/run_bench_gpu.sh   # E3 on the big-code corpus
#
# LADDER=1 runs scripts/review/codec_ladder.py after every bench: sign vs centred
# vs ITQ vs 2-bit vs centroid+residual codecs, with bootstrap CIs, plus the
# embedding-geometry statistics that replace rho (docs/REVIEW-2026-08-21.md).
# CODE_SCALE renders the generated corpus with the unique code N times larger
# (or smaller) and nothing else changed, so the reference encoder can (or cannot)
# read it at 448px. Outputs land in reports/colpali_generated_code<N>x/.
#
# CODEBOOK=1 adds retrieval-space saliency at matched token budgets, with
# random and k-means probes as controls. On E1 it is +2.5 points over pixel
# saliency at keep-10% and behind it at keep-30%; the claim is that the
# advantage grows on dense pages, where the pixel detector returns 1.03x and
# has nothing to remove. infovqa is the split that decides it. It roughly
# doubles the replay, not the encode, so it is cheap next to the download.
#
# The encode caches are the raw patch embeddings, about a gigabyte per split.
# We derive what we need from them on-box -- winner_stats.py and
# geometry_stats.py both write a few lines of text -- so the default archive
# stays small. KEEP_CACHE=1 tars the caches too, for work that needs the
# vectors themselves rather than statistics over them.
#
# MODE=generated is the experiment that separates encoder scale from corpus
# provenance. E1 (ColSmol, generated pages) and E2 (ColPali, ViDoRe) change both
# variables at once, so neither can say which one moved the attribution. This
# runs the reference encoder over the *same* corpus E1 used, which pins it: if
# the reversal follows the encoder the story is about retrieval margin, if it
# follows the corpus it is about page sparsity. It is 60 pages, so it costs
# minutes rather than the hour E2 takes.
#
# Run it under tmux. An SSH drop kills the foreground process otherwise, and
# these runs are long enough that it will happen:
#
#   tmux new -s bench 'bash scripts/run_bench_gpu.sh 2>&1 | tee run.log'
#
# Results are archived after *every* split, not just at the end, because a
# preempted pod should not cost you the splits that already finished.
set -euo pipefail

# Rented boxes mount a volume at /workspace; Kaggle gives /kaggle/working and
# Colab /content. Detect rather than make the caller pass it in.
if [ -z "${WORKDIR:-}" ]; then
    if [ -d /kaggle/working ]; then WORKDIR=/kaggle/working
    elif [ -d /content ]; then WORKDIR=/content
    else WORKDIR=/workspace
    fi
fi
REPO="${REPO:-https://github.com/Daemon-VI/optivision-rag.git}"
CHECKOUT="$WORKDIR/optivision-rag"
LIMIT="${LIMIT:-500}"
# Overridable so the control flow can be exercised without a GPU:
#   CONFIG=configs/synthetic.yaml MODE=generated bash scripts/run_bench_gpu.sh
CONFIG="${CONFIG:-configs/colpali_bench.yaml}"
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
# The CLI is invoked as `python -m optivision.cli`, not via the `optivision`
# console script: pip can install that script somewhere off PATH, and the
# failure then reads as a missing program rather than a missing PATH entry.

# Derive the statistics that would otherwise need the encode cache shipped
# home. Neither is a deliverable, so neither is allowed to end the run: under
# `set -e` an unguarded failure here would discard a benchmark that already
# finished. Errors are reported and stepped over.
stats() {
    local cache="$1" tag="$2" label="$3" corpus="${4:-}"
    local with_corpus=()
    [ -n "$corpus" ] && with_corpus=(--corpus "$corpus")

    python scripts/winner_stats.py --cache "$cache" "${with_corpus[@]}" 2>&1 \
        | tee "reports/winner_stats_$tag.txt" \
        || say "winner_stats failed on $tag - continuing"
    python scripts/geometry_stats.py --cache "$cache" --label "$label" 2>&1 \
        | tee "reports/geometry_$tag.txt" \
        || say "geometry_stats failed on $tag - continuing"

    # Selection scored against the oracle winner set. Needs labels, so it only
    # runs where a corpus was passed; it is the diagnostic that says *why* a
    # codebook row won or lost, which the benchmark table cannot.
    if [ -n "$corpus" ]; then
        python scripts/probe_eval.py --cache "$cache" --corpus "$corpus" 2>&1 \
            | tee "reports/probe_eval_$tag.txt" \
            || say "probe_eval failed on $tag - continuing"
    fi

    # Codec ladder and the geometry statistics that actually differ between
    # encoders (||mean||, dead bits, participation ratio, distractor promotion).
    # CPU-only, a few minutes on a 500-page split; --pruned adds the optivision
    # prune config under every codec.
    if [ -n "${LADDER:-}" ] && [ -n "$corpus" ]; then
        python scripts/review/codec_ladder.py --cache "$cache" --corpus "$corpus" \
            --label "$label" --pruned --out "reports/ladder_$tag.json" 2>&1 \
            | tee "reports/ladder_$tag.txt" \
            || say "codec_ladder failed on $tag - continuing"
    fi
}

# One archive step for every exit path, so KEEP_CACHE is honoured everywhere
# rather than in whichever branch remembered it.
archive() {
    if [ -n "${KEEP_CACHE:-}" ]; then
        say "KEEP_CACHE set - including data/cache (this makes the archive large)"
        tar czf "$WORKDIR/optivision_reports.tar.gz" "$@" data/cache
    else
        tar czf "$WORKDIR/optivision_reports.tar.gz" "$@"
    fi
}

mkdir -p data/cache reports

if [ "${MODE:-vidore}" = "generated" ]; then
    # Exactly the corpus E1 measured: 30 documents x 2 pages at seed 7 is the
    # spec recorded in reports/colsmol/benchmark.json. The generator is
    # deterministic, so this reproduces those pages rather than resembling them.
    # CODE_SCALE=3 renders the same pages with only the unique code three times
    # larger, under its own corpus/report/cache names so runs do not collide.
    if [ -n "${CODE_SCALE:-}" ]; then
        SCALE_TAG="code$(printf '%s' "$CODE_SCALE" | tr -d '.')x"
        CORPUS="data/corpus_$SCALE_TAG"; GEN_TAG="generated_$SCALE_TAG"
        SCALE_ARG=(--code-scale "$CODE_SCALE")
        say "E3 at code scale $CODE_SCALE: ColPali-v1.3 over the generated corpus with the code rendered ${CODE_SCALE}x"
    else
        CORPUS="data/corpus"; GEN_TAG="generated"; SCALE_ARG=()
        say "E3: ColPali-v1.3 over the generated corpus (E1's corpus, encoder swapped)"
    fi
    python -m optivision.cli make-corpus "$CORPUS" --docs 30 --pages 2 --seed 7 "${SCALE_ARG[@]}"

    python -m optivision.cli bench \
        "$CORPUS/pdfs" "$CORPUS/queries.json" \
        -c "$CONFIG" \
        --out "reports/colpali_$GEN_TAG" \
        --sweep ${CODEBOOK:+--codebook} \
        --cache "data/cache/colpali_$GEN_TAG.npz" \
        2>&1 | tee "reports/colpali_$GEN_TAG.log"

    # The encode cache is ~30 MB for 60 pages, so KEEP_CACHE=1 is cheap here.
    # Compute the derived statistics while the cache is warm, but bank the
    # benchmark before running anything optional over it.
    archive reports

    stats "data/cache/colpali_$GEN_TAG.npz" "$GEN_TAG" \
        "E3 ColPali-3B, $GEN_TAG" "$CORPUS"

    archive reports
    say "done - archived $WORKDIR/optivision_reports.tar.gz"
    ls -la "$WORKDIR/optivision_reports.tar.gz"
    exit 0
fi

for split in $SPLITS; do
    tag="${split##*/}"
    say "$split"

    python -m optivision.cli fetch-vidore --dataset "$split" --out "data/vidore_$tag" --limit "$LIMIT"
    python -m optivision.cli bench \
        "data/vidore_$tag/images" "data/vidore_$tag/queries.json" \
        -c "$CONFIG" \
        --out "reports/colpali_$tag" \
        --sweep ${CODEBOOK:+--codebook} \
        --cache "data/cache/colpali_$tag.npz" \
        2>&1 | tee "reports/colpali_$tag.log"

    # Label-free: what fraction of this split's patches ever wins a MaxSim.
    # Real pages are denser than the generated corpus, so this is the number
    # that says whether retrieval-space rate allocation has room to work.
    # Archive now. A pod that dies during split 3 must not cost you splits 1-2.
    archive reports

    stats "data/cache/colpali_$tag.npz" "$tag" "E2 ColPali-3B, $tag" \
        "data/vidore_$tag"

    archive reports
    say "archived $WORKDIR/optivision_reports.tar.gz after $tag"
done

say "done - copy $WORKDIR/optivision_reports.tar.gz off this box before terminating it"
ls -la "$WORKDIR/optivision_reports.tar.gz"
