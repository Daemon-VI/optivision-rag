# OptiVision RAG for VS Code

Run the [OptiVision RAG](https://github.com/Daemon-VI/optivision-rag) CLI — extreme token
compression for vision-language document retrieval — from a control panel in the editor,
instead of hand-typing `optivision` commands in a terminal.

## Getting started

Run **OptiVision RAG: Open Control Panel** from the Command Palette (or click the status bar
item). It detects the CLI automatically: an installed `optivision` on PATH, else
`python`/`python3`/`py -3 -m optivision.cli`. If neither is found, the panel offers **Install
via pip** — this runs `pip install optivision-rag` in a terminal you can watch; nothing installs
silently.

## What it does

A single panel with a form per CLI command, each running the real `optivision` process (never
through a shell) against your open workspace folder, with output streamed live into the panel's
log:

- **Init config** — write a config file for a backend (`colsmol` / `colpali` / `colqwen2` /
  `synthetic`).
- **Make corpus** — generate a synthetic scanned-document corpus with ground-truth queries. No
  downloads.
- **Index** — encode, prune, quantize and index a folder of PDFs or images.
- **Search** — query the index; results print as the CLI's own table.
- **Stats** — show what is currently indexed.
- **Explain** — render original / saliency / keep-mask figures per page, shown inline in the
  panel once the run finishes.
- **Advanced**: **Bench** (the full ablation table — can run long) and **Fetch a real ViDoRe
  split** (downloads from Hugging Face) — present, but out of the way of the everyday flow.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `optivision.command` | *(auto-detect)* | How to run the CLI. Empty finds `optivision` on PATH, else a Python launcher running `optivision.cli`. |
| `optivision.configPath` | *(none)* | Default `--config` for commands whose own Config field is left blank. |
| `optivision.cwd` | *(workspace root)* | Working directory for the CLI. |

## What it does not do

It does not manage Python environments, download models on your behalf beyond what the CLI
itself does, or run anything through a shell — every call is `spawn(exe, argv, { shell: false
})`, so a query or path typed into the panel is never shell-interpreted.

MIT licensed. Part of the OptiVision RAG project.
