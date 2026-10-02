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

## ColQwen2 / ColQwen2.5 vectors on a free Kaggle GPU

`notebooks/kaggle_colqwen_encode.ipynb` produced the vectors behind
`docs/UNIVERSAL.md`, R11. It needs a Kaggle account with phone verification,
**Accelerator: GPU T4 x2** and **Internet: On**. It runs in about 2 hours and
stops at the first failed check:

1. Install from `main` (R11 ran at commit `46adfd9`), pinning `colpali-engine==0.3.17`,
   and remove Kaggle's preinstalled `torchao`: peft 0.19 refuses to apply LoRA
   adapters while an incompatible version is importable.
2. Smoke-test ColQwen2 twice: pre-merged checkpoint against loader-merged
   adapter. They must agree (median cosine > 0.99), and each query must retrieve
   its own page.
3. Encode ViDoRe V1 DocVQA and InfoVQA (500 pages each) with ColQwen2, in float32,
   at about 2.3 s/page on a T4.
4. Smoke-test and encode ColQwen2.5 with `--multi-gpu`. At 14 GiB in float32 it is
   split across both T4s at a decoder-layer boundary.
5. Zip `vectors/` with `pip freeze`, the GPU name and the commit.

If the zip is too slow to download, **Save Version → Quick Save** keeps the
outputs on Kaggle. Then put the `*_docs.npz`, `*_queries.npz` and `*_qrels.json`
files in `data/vectors/` and run the studies as
`vec:<name>` datasets (see `scripts/universal_study/README.md`).

## A wide (2,560-d) encoder on Kaggle: encode and analyse in one session

`notebooks/kaggle_wide_model.ipynb` produced `docs/UNIVERSAL.md`, R12, for
`nvidia/nemotron-colembed-vl-4b-v2` (CC-BY-NC-4.0, non-commercial use only;
pinned revision `0ed152d9`). One split is about 3.8 GB of float32 vectors, too large to download
at home-connection speeds. Each session therefore encodes **and** analyses, and
exports only a few MB.

1. One notebook per split. Settings: **GPU T4 x2**, **Internet On**. Parameters:
   `SPLIT = "docvqa"` or `"infovqa"`; `RUN_SMOKE = True` in the first session only.
   Two accounts can run the two splits at the same time.
2. Run the smoke cell interactively first (about 5 minutes after install). It must
   report:
   - width 2560;
   - 6/6 own-page retrieval;
   - a `scorer_check` difference around 1e-5;
   - near-zero GPU memory after release.

   It also measures float16 against float32 encodings.
3. Then **Save Version -> Save & Run All (Commit)** with `RUN_SMOKE = False`. It
   needs no open tab or awake laptop. Measured on two T4s:
   - encoding: about 26 min (3.1 s/page, float32, layers split 15/21);
   - `s13_wide_study.py`: 46–52 min;
   - `s7b_selection_rules.py --space=both`: about 2.2 h.
4. Download `wide_results_<split>.zip` from the version's Output tab. Its JSON goes
   to `reports/universal/wide/`, `selection_rules/` and `selection_rules_wide/`.

Interactive sessions stop when the browser sleeps: use commit runs for anything
long. If an interactive cell was started by hand, it cannot be converted.

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

# 2. Same pages with only the unique code's glyphs rendered 3x larger (33 pt, drawn in
#    the blank right half of the field block; nothing else moves). On tiled ColSmol this
#    did NOT change the one-bit cost (docs/REVIEW-2026-08-21.md, section 2): the cost
#    follows each query's float decision margin vs the codec's score noise. Here the
#    question is only whether ColPali's precise-query R@1 leaves the 0.2 floor at all;
#    split precise/topical from benchmark.json["runs"] before reading the codec rows.
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
| `reports/colpali_generated_code3x/benchmark.md` + `benchmark.json["runs"]` | E3 on the big-code corpus; `runs` holds top-10 ids per query so precise/topical R@1 can be split |
| `reports/ladder_infovqa_test_subsampled.txt`, `ladder_docvqa_test_subsampled.txt` | whether centring / ITQ / 2-bit / centroid+residual make the one-bit codec free on ColPali, with CIs over 494/451 queries |
| same files, `decision margin` block | per-query float margin vs each codec's score noise and the R@1 flip rate by margin/noise; the review's prediction is 0% flips above 2 sigma (docs/REVIEW-2026-08-21.md, sections 2 and 8) |
| `reports/colpali_docvqa_test_subsampled/benchmark.md` | the Stage-II rows on the second dense split |
| `data/cache/colpali_generated*.npz` | 30 MB each; replay anything in `scripts/review/` against ColPali locally |
