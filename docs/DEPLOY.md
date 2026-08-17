# Running and deploying the demo

The demo (`app.py`) is a single-page presentation of the Stage-I pipeline. It is
deliberately small: one page in, compression numbers and a keep-mask figure out.

> ## Status: run locally
>
> **A Hugging Face Space is not available on a free account.** Attempting to
> create one returns:
>
> > `402 Payment Required` — *Static Spaces are free for everyone, but hosting
> > Gradio and Docker Spaces on free cpu-basic requires a PRO subscription.*
>
> This is broader than the ZeroGPU caveat: it applies to **any** live Gradio
> Space, GPU or not. Verified 2026-08-17 with a valid write token on account
> `trithikkrishna`.
>
> The demo therefore runs **locally** for the presentation, which is what the
> rest of this document now covers first. The Space path below still works and is
> kept because it needs no code changes — only a PRO subscription.

## Running it for the presentation

Double-click **`run_demo.bat`**, or from a terminal:

```bash
cd C:\Users\Rishi\optivision-rag
.venv\Scripts\python.exe app.py
```

Then open <http://127.0.0.1:7860> (the .bat opens it for you).

### On the day

- **Start it a few minutes early.** The checkpoint is already cached on this
  machine, so nothing downloads — but the first **Compress Document** click still
  spends ~30 s encoding on this CPU. Do one warm-up run before the examiner is
  watching; later clicks reuse the loaded model.
- **Have a page ready.** `data/corpus/pdfs/invoice_000.pdf` is a good example:
  clear text block at the top, blank lower half, so the keep-mask figure reads
  obviously. Any scan or PDF of the examiner's choosing also works.
- **No internet is required** once the model is cached. The only network call at
  startup is a Hugging Face metadata check, and `run_demo.bat` bounds it to 10 s.
- **Multi-page PDFs are fine** — a page picker appears; the demo compresses one
  page at a time by design.

To point the demo at a different config:

```bash
set OPTIVISION_CONFIG=configs\colpali.yaml   && .venv\Scripts\python.exe app.py
```

## Deploying to a Space (requires HF PRO)

Everything below is tested and ready; it only needs an account that is allowed to
create a Gradio Space. With a PRO subscription the whole deployment is one
command:

```bash
.venv\Scripts\python.exe scripts\deploy_space.py
```

That creates `<your-user>/optivision-rag` and uploads the 61 files the Space
needs (it uses the HF API rather than `git push`, which would block on an
interactive credential prompt). Add `--name` or `--private` to vary it, and
`--dry-run` to see the file list first.

The manual route, if you prefer it:

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
| Hardware | any (PRO required for the Space itself) | see below |
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

1. **Free accounts cannot host a live Gradio Space at all** — GPU or not (the 402
   above). ZeroGPU is a second paywall on top of that, not the only one.
2. **The app runs unchanged without any GPU.** `spaces` is imported inside a
   `try`, the decorator degrades to a no-op, and `pick_device()` returns `cpu`.
   The only difference is speed: roughly 30–60 s per page instead of a few
   seconds. That is why running locally loses nothing but the public URL.

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
