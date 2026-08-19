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
   git clone --depth 1 https://github.com/Daemon-VI/optivision-rag.git optivision
   tmux new -s bench 'bash optivision/scripts/run_bench_gpu.sh 2>&1 | tee /workspace/run.log'
   ```

   Detach with `Ctrl-b d`, reattach with `tmux attach -t bench`.
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
