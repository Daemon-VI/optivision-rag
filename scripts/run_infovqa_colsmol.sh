#!/usr/bin/env bash
# Overnight, CPU-only: the E1 encoder (ColSmol-256M, tiled) over the real
# infovqa split (500 pages, 494 queries), then every analysis the review needs
# on real pages -- bench with codebook controls, winner/geometry/probe stats,
# the codec ladder with the decision-margin block, the margin analysis and the
# two-tier rescoring simulation. ~12 s/page on this laptop => ~2.5 h end to end.
#
#   powershell: Start-Process bash.exe scripts/run_infovqa_colsmol.sh   (see scripts/keepawake.py)
set -uo pipefail
export PATH="/usr/bin:/mingw64/bin:/c/Windows/System32:$PATH"
cd "$(dirname "$0")/.."
export HF_HUB_ETAG_TIMEOUT=10 HF_HUB_DOWNLOAD_TIMEOUT=30 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=./.venv/Scripts/python.exe
TAG=infovqa_test_subsampled
CORPUS=data/vidore_$TAG
CACHE=data/cache/colsmol_$TAG.npz
OUT=reports/colsmol_$TAG
mkdir -p "$OUT"

say() { echo "=== $* $(date '+%Y-%m-%d %H:%M:%S')"; }

say "bench start"
$PY -m optivision.cli bench "$CORPUS/images" "$CORPUS/queries.json" \
    -c configs/colsmol.yaml --out "$OUT" --sweep --codebook --cache "$CACHE" \
    2>&1 | tee "$OUT/run.log" | grep -v "Loading weights"
say "bench done"

[ -f "$CACHE" ] || { say "no cache written - stopping"; exit 1; }

say "winner_stats"
$PY scripts/winner_stats.py --cache "$CACHE" --corpus "$CORPUS" 2>&1 | tee "reports/winner_stats_colsmol_$TAG.txt" \
    || say "winner_stats failed - continuing"
say "geometry_stats"
$PY scripts/geometry_stats.py --cache "$CACHE" --label "ColSmol-256M on $TAG" 2>&1 | tee "reports/geometry_colsmol_$TAG.txt" \
    || say "geometry_stats failed - continuing"
say "probe_eval"
$PY scripts/probe_eval.py --cache "$CACHE" --corpus "$CORPUS" 2>&1 | tee "reports/probe_eval_colsmol_$TAG.txt" \
    || say "probe_eval failed - continuing"
say "codec_ladder"
$PY scripts/review/codec_ladder.py --cache "$CACHE" --corpus "$CORPUS" \
    --label "ColSmol-256M, $TAG (E1 encoder on real pages)" --pruned \
    --out "reports/ladder_colsmol_$TAG.json" 2>&1 | tee "reports/ladder_colsmol_$TAG.txt" \
    || say "codec_ladder failed - continuing"
say "margin analysis (q9) and two-tier rescoring (q11)"
$PY scripts/review/q9_margin.py 2>&1 | tee reports/margin_summary.txt || say "q9 failed - continuing"
$PY scripts/review/q11_margin_rescoring.py 2>&1 | tee reports/margin_rescoring.txt || say "q11 failed - continuing"
say "all done"
