# Running the ColPali / ViDoRe benchmark on a rented GPU

The reference numbers need a GPU that the free tiers would not reliably give us:
Colab's quota ran out mid-run and Kaggle gates both the accelerator and network
access behind phone verification. A rented box costs about $0.25 for this entire
benchmark and has neither queue nor quota.

Two paths produce identical numbers — same config, same code:

| path | use when |
|---|---|
| `notebooks/vidore_colpali_bench.ipynb` | Colab or Kaggle, interactive, one split at a time |
| `scripts/run_bench_gpu.sh` | any Ubuntu + CUDA box, unattended, all four splits |

## What the run needs

- A CUDA GPU with **~8 GB free**. ColPali-v1.3 is ~3B parameters in bfloat16
  (~6 GB). A T4, A4000, L4 or 3090 all work; anything cheaper than a 3090 is
  usually the better value here because the run is short.
- **~40 GB disk.** The merged weights are ~6 GB, the four ViDoRe splits about
  1 GB, and the encode caches a few hundred MB per split.
- Network access, for the Hub and for `datasets`.

Encoding is the only slow part: roughly 1–3 pages/second, so 500 pages is a few
minutes. The ablation rows replay off the cached vectors and are near-instant.
Budget one hour end to end including setup and downloads.

## RunPod

1. **Deploy a Pod** with a PyTorch template (torch and CUDA preinstalled). Give
   it a 40 GB volume mounted at `/workspace` — the script keeps the Hub cache
   there, so a restarted pod does not re-download 6 GB of weights.
2. Open the **Web Terminal**, or SSH in.
3. Run it under `tmux`. The run outlives an SSH drop that way, which matters
   more than it sounds: a dropped connection kills a foreground process and
   takes the session's work with it.

   ```bash
   cd /workspace
   git clone --depth 1 https://github.com/Daemon-VI/optivision-rag.git optivision-rag
   tmux new -s bench 'bash optivision-rag/scripts/run_bench_gpu.sh 2>&1 | tee /workspace/run.log'
   ```

   Detach with `Ctrl-b d`, reattach with `tmux attach -t bench`.

   The checkout must not be named `optivision`. If the directory above it is on
   `sys.path` — which is the default for a notebook kernel, and permanent on
   Kaggle — a directory of that name shadows the installed package as an empty
   namespace package, and imports fail with `No module named 'optivision.config'`.
4. **Copy the results off before terminating the pod.** `runpodctl` avoids
   setting up SSH keys — on the pod:

   ```bash
   runpodctl send /workspace/optivision_reports.tar.gz
   ```

   It prints a one-time code; on your own machine, `runpodctl receive <code>`.
   Or plain `scp -P <port> root@<pod-ip>:/workspace/optivision_reports.tar.gz .`
5. **Terminate the pod.** It bills while it exists, running or not.

## Vast.ai

Identical, with a `pytorch/pytorch` CUDA image. Vast instances are
interruptible, which is why the script re-archives `reports/` after *every*
split rather than at the end — an instance reclaimed during split 3 still
leaves splits 1 and 2 on disk.

## Options

```bash
bash scripts/run_bench_gpu.sh                                  # all four splits
SPLITS="vidore/docvqa_test_subsampled" bash scripts/run_bench_gpu.sh
LIMIT=100 bash scripts/run_bench_gpu.sh                        # quick smoke run
WORKDIR=/data bash scripts/run_bench_gpu.sh                    # different volume
```

Set `HF_TOKEN` before running if you have one. It only lifts Hub rate limits,
but the weights are a 6 GB download and unauthenticated fetches are throttled.

## What it checks before doing any work

The script fails fast rather than after a long download:

- CUDA is actually visible, and the card has enough memory.
- `configs/colpali.yaml` resolves to a `-merged` checkpoint. The adapter-only
  `vidore/colpali-v1.3` loads a randomly initialised projection head on current
  transformers builds, which produces a complete and entirely meaningless
  results table. See `src/optivision/encoders/colvlm.py`.
- It prints the resolved `torch` / `transformers` / `peft` / `colpali-engine`
  versions next to the commit hash. Put those in the paper — they are what the
  numbers were produced with, and they are not recoverable afterwards.

## Results

Each split writes `reports/colpali_<tag>/benchmark.json` and `.md`, plus a
`.log`. Section 6 of the notebook turns `benchmark.json` into the LaTeX rows for
Table I; run it locally against the extracted archive.

Commit `reports/colpali_*/` to the repo. The paper claims the benchmark is
reproducible, so a reviewer who clones should find the numbers printed in it.

## Review follow-ups in one Kaggle cell

The three runs `docs/REVIEW-2026-08-21.md` asks for, in one `%%bash` cell. Needs a GPU
session (T4 x2 is enough), **Internet on**, and the repo pushed to GitHub `main` (the
runner clones it; nothing local is used). Optional: an `HF_TOKEN` secret under
Add-ons -> Secrets, which only lifts Hub rate limits on the 6 GB ColPali download.
Roughly 60-90 minutes on a T4; the two generated runs are minutes each, the ViDoRe
runs ~20-30 minutes per split. Output is one archive in `/kaggle/working`.

```bash
%%bash
# OptiVision RAG -- review follow-ups: E3 with its cache kept, E3 on the big-code
# corpus, and the dense ViDoRe splits with the Stage-II rows and the codec ladder.
set -euo pipefail
cd /kaggle/working
TOKEN="$(python -c 'from kaggle_secrets import UserSecretsClient as C; print(C().get_secret("HF_TOKEN"))' 2>/dev/null || true)"
[ -n "$TOKEN" ] && export HF_TOKEN="$TOKEN"
rm -rf optivision                      # a dir of this name shadows the package on Kaggle's sys.path
if [ -d optivision-rag/.git ]; then git -C optivision-rag fetch --depth 1 origin main && git -C optivision-rag reset --hard FETCH_HEAD
else git clone --depth 1 https://github.com/Daemon-VI/optivision-rag.git; fi
cd optivision-rag

# 1. E3 on E1's corpus. KEEP_CACHE is 30 MB here; LADDER runs the codec ladder and the
#    geometry statistics (||mean||, dead bits, participation ratio, distractor promotion)
#    on the ColPali cache -- the controlled comparison against E1 the paper lacks.
MODE=generated LADDER=1 KEEP_CACHE=1 bash scripts/run_bench_gpu.sh

# 2. Same pages with only the unique code rendered 3x larger, so ColPali can read it at
#    448 px. If its one-bit loss rises from 1.6 points towards E1's 12, the codec's cost
#    follows evidence legibility, not the encoder.
MODE=generated CODE_SCALE=3 LADDER=1 KEEP_CACHE=1 bash scripts/run_bench_gpu.sh

# 3. The two dense splits with enough queries to resolve 1-2 point differences: Stage-II
#    rows (CODEBOOK) and the codec ladder (does ITQ / 2-bit / residual close ColPali's
#    remaining 2.6-3.7 points?). Caches stay on the box; only text and JSON come home.
CODEBOOK=1 LADDER=1 SPLITS="vidore/infovqa_test_subsampled vidore/docvqa_test_subsampled" bash scripts/run_bench_gpu.sh

# Everything in one archive: every reports/ folder and the two small generated caches.
tar czf /kaggle/working/optivision_review.tar.gz reports data/cache/colpali_generated*.npz
ls -la /kaggle/working/optivision_review.tar.gz
```

What comes back, and what to read first:

| file | what it answers |
|---|---|
| `reports/ladder_generated.txt`, `ladder_generated_code3x.txt` | ColPali's geometry next to E1's (`reports/ladder_colsmol.json` locally), and its codec ladder when it can vs cannot read the code |
| `reports/colpali_generated_code3x/benchmark.md` + `benchmark.json["runs"]` | E3 on the legible corpus; `runs` holds top-10 ids per query so precise/topical R@1 can be split |
| `reports/ladder_infovqa_test_subsampled.txt`, `ladder_docvqa_test_subsampled.txt` | whether centring / ITQ / 2-bit / centroid+residual make the one-bit codec free on ColPali, with CIs over 494/451 queries |
| `reports/colpali_docvqa_test_subsampled/benchmark.md` | the Stage-II rows on the second dense split |
| `data/cache/colpali_generated*.npz` | 30 MB each; replay anything in `scripts/review/` against ColPali locally |
