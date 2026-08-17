# Deploying the demo to Hugging Face Spaces

The demo (`app.py`) is a single-page presentation of the Stage-I pipeline. It is
deliberately small: one page in, compression numbers and a keep-mask figure out.

## Run it locally first

```bash
cd C:\Users\Rishi\optivision-rag
.venv\Scripts\python.exe -m pip install gradio          # once
.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:7860>. The first **Compress Document** click loads the
ColSmol checkpoint (~0.5 GB, cached afterwards) and takes ~30 s/page on this CPU;
later clicks reuse the loaded model.

To point the demo at a different config:

```bash
set OPTIVISION_CONFIG=configs\colpali.yaml   && .venv\Scripts\python.exe app.py
```

## Create the Space

1. Go to <https://huggingface.co/new-space>.
2. **Owner** your account · **Space name** `optivision-rag` · **License** MIT.
3. **SDK** → *Gradio*. (The metadata block at the top of `README.md` already sets
   `sdk: gradio`, `sdk_version`, and `app_file: app.py`.)
4. **Hardware** → see the note below. Start with **CPU basic (free)**; it works.
5. **Visibility** → Public (required for free ZeroGPU, and fine for a college demo).
6. Create the Space. It gives you a git remote like
   `https://huggingface.co/spaces/<user>/optivision-rag`.

## Push

This project is not a git repository yet, so initialise it and push:

```bash
cd C:\Users\Rishi\optivision-rag
git init
git add .
git commit -m "OptiVision RAG: Stage-I pipeline and single-page demo"

git remote add space https://huggingface.co/spaces/<user>/optivision-rag
git push space HEAD:main
```

If the Space was created with a starter commit, push over it once:

```bash
git push --force space HEAD:main
```

Authentication: when prompted, use your HF **username** and an access token with
*write* scope from <https://huggingface.co/settings/tokens> as the password.

`.gitignore` already excludes `.venv/`, `data/` and the benchmark indexes, so the
push stays small. `configs/`, `src/`, `app.py` and `requirements.txt` are what the
Space actually needs.

## Space settings

| setting | value | why |
|---|---|---|
| SDK | Gradio (from README metadata) | |
| App file | `app.py` | |
| Hardware | CPU basic works; ZeroGPU if available to you | see below |
| Visibility | Public | free ZeroGPU requires it |
| Secrets | **none** | the demo needs no credentials |
| Env vars | `OPTIVISION_CONFIG` *(optional)* | defaults to `configs/colsmol.yaml` |

No secrets or tokens are required. Qdrant is optional, runs embedded on the
Space's local disk, and is off by default.

## About ZeroGPU — read this before assuming it is free

`app.py` imports `spaces` and decorates only the encode step with
`@spaces.GPU(duration=120)`. Pruning, quantization, byte accounting and all
rendering stay on CPU, as intended.

Two honest caveats:

1. **ZeroGPU hardware selection has historically required a PRO subscription.**
   Hugging Face has changed this policy more than once, so check the Hardware
   dropdown on your own Space rather than trusting this document. If ZeroGPU is
   not offered to your account, you do **not** need it — see below.
2. **The app runs unchanged on free CPU-basic hardware.** `spaces` is imported
   inside a `try`, the decorator degrades to a no-op, and `pick_device()` returns
   `cpu`. The only difference is speed: roughly 30–60 s per page instead of a few
   seconds. For a single-page viva demo that is perfectly usable — just click
   *Compress Document* a moment before you need the result.

Do not point this Space at a benchmark dataset. It is sized for one page at a
time, and the free ZeroGPU quota is not meant for bulk encoding.

## First-boot behaviour

- The Space installs `requirements.txt`, then starts `app.py`.
- The ColSmol checkpoint (~0.5 GB) downloads on the **first compression**, not at
  build time, so the Space appears "Running" before the model is ready. The first
  click is therefore slower than the rest; the status panel says so.
- `concurrency_limit=1` on the compress button keeps two visitors from encoding
  simultaneously and exhausting memory.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| Build fails resolving `colpali-engine` | transformers/torch conflict | pin `transformers` in `requirements.txt` to the version your local venv resolved |
| "This demo requires the real ColSmol encoder" | `OPTIVISION_CONFIG` points at `configs/synthetic.yaml` | unset it, or point at `configs/colsmol.yaml` |
| First click times out | checkpoint still downloading | wait and click again; it is cached afterwards |
| Out-of-memory on CPU basic | very large scan | the demo already downscales to 1536 px; try a smaller page |
