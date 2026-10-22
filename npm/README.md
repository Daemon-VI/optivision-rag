# OptiVision RAG

**Extreme token compression for vision-language document retrieval.**

ColPali-style models search scanned documents as images, with no OCR. They store
one 128-dimensional float vector per image patch, so a single page costs about
512 KB of index and a million pages cost about half a terabyte.

OptiVision RAG shrinks the **index**, not the model. The published checkpoint runs
unmodified and retrieval is still late-interaction MaxSim. Only the number and the
size of the stored vectors change:

```
page image
   ├─ VLM encoder ─────────► ~1000 patch vectors        (model unchanged)
   ├─ 1. spatial pruning ──► drop patches on blank paper
   ├─ 2. redundancy prune ─► collapse near-duplicate patches
   ├─ 3. binary quantize ──► 128 floats (512 B) -> 128 bits (16 B)
   └─ index ───────────────► Qdrant MaxSim, or an exact numpy index
```

| setting | encoder / corpus | compression | nDCG@5 retained |
|---|---|---|---|
| prune + binary | ColSmol-256M, 60 generated pages (72 queries) | 113.5x | 86.7% |
| prune + int8 | ColSmol-256M, 60 generated pages (72 queries) | 14.2x | 96.0% |
| prune + binary | ColPali-v1.3, 4 ViDoRe splits (1,780 pages, 1,325 queries) | 53-60x | 94.6-103.4% |

Each row is one fixed pipeline scored on every query of its corpus, with no choice made
from those queries. The 60-page corpus has 72 queries, so its figures carry intervals of
about 6 points (a sanity check, not evidence). Ratios and retention are specific to the
encoder and corpus named in each row.

Full results, the ablations and the analysis are in the
[repository](https://github.com/Daemon-VI/optivision-rag).

> **This npm package is a launcher.** OptiVision RAG is written in Python. The first
> time you run `npx optivision-rag`, it creates a private virtual environment in your
> user cache folder and installs the matching release from
> [PyPI](https://pypi.org/project/optivision-rag/). Later runs start straight away.
> You need **Python 3.10 or newer** on the machine.
>
> | variable | effect |
> |---|---|
> | `OPTIVISION_EXTRAS` | pip extras to install, default `corpus`. Use `vlm,corpus` for the real encoders. |
> | `OPTIVISION_PYTHON` | the Python interpreter to use |
> | `OPTIVISION_REINSTALL=1` | rebuild the environment |
> | `OPTIVISION_HOME` | where the environments are kept |
>
> Every `optivision` command below also works as `npx optivision-rag <command>`.
>
> The launcher installs the `corpus` extra by default, which does not include SciPy. The
> `optivision certify` command (added in 0.4.0) needs SciPy for its
> default candidate list, so through npx set `OPTIVISION_EXTRAS=corpus,merge` (the `merge` extra
> installs SciPy), or pass `--allow-reduced` to use the smaller candidate list. Each set of extras
> gets its own environment. Running `certify` from a pip install is the more direct route.

## Install

Python 3.10 or newer.

```bash
pip install optivision-rag                  # core pipeline + CLI
pip install "optivision-rag[corpus]"        # + synthetic test-corpus generator
pip install "optivision-rag[vlm]"           # + real encoders (torch, colpali-engine)
pip install "optivision-rag[vlm,corpus,bench]"
```

Or run the CLI through npm without touching pip yourself (it still needs Python
3.10+ on the machine):

```bash
npx optivision-rag --help
```

## Quick start (offline, no model download)

The `synthetic` preset uses a deterministic stand-in encoder, so the whole pipeline
runs in seconds on any laptop. Use it to try the tool, not for real results.

```bash
optivision make-corpus data/corpus --docs 10 --pages 2
optivision index data/corpus/pdfs -c synthetic
optivision search "renewal of vehicle insurance policy" -c synthetic
optivision stats -c synthetic
```

## With a real model

Install the `vlm` extra, then switch the preset. `colsmol` is a 256M-parameter
model that runs on a CPU laptop; `colpali` wants a GPU.

```bash
optivision index path/to/your/pdfs -c colsmol
optivision search "fire safety audit memorandum" -c colsmol
optivision explain path/to/your/pdfs -c colsmol --out figures
```

`-c` takes a bundled preset name (`synthetic`, `colsmol`, `colpali`, `qdrant`) or a
path to your own YAML file. `optivision init-config my.yaml` writes one pre-filled
with every default.

## Easy mode

Give a model and a target; get a compressed, searchable index of a folder of PDFs. `target`
is the share of the model's full-precision search quality (nDCG@5) to keep.

```bash
npx optivision-rag options vidore/colqwen2-v1.0-hf --target 0.97 --pages my_pdfs/
npx optivision-rag build my_pdfs/ --model vidore/colqwen2-v1.0-hf --target 0.97
npx optivision-rag find my_pdfs.ovx "What drove operating costs in 2023?"
npx optivision-rag check my_pdfs.ovx
npx optivision-rag quickstart my_pdfs/        # guided; prints the build line when run without a terminal
npx optivision-rag doctor --model vidore/colqwen2-v1.0-hf
```

`options` and `doctor` run in the environment described above. `build`, `find`, `check` and
`quickstart` load a model, so the launcher keeps a second environment for them with PyTorch and the
model's loader: the extra `easy-st` (sentence-transformers models) or `easy-colpali` (colpali-engine
models), chosen from the model's loader. The model is read from `--model`, or for `find` and `check`
from the index folder. Before the first install the launcher prints the download size (roughly 3 to
4 GB on Linux, 0.3 to 0.5 GB on Windows and macOS) and pip's output stays visible.

On Windows, pip installs the CPU-only build of PyTorch, so encoding runs on the CPU. For an NVIDIA GPU,
install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) into that
environment afterwards; `optivision doctor` reports which build you have. `OPTIVISION_EXTRAS` still
names the extras of the first environment (default `corpus`); the second one adds its extra to
whatever is set. If `options` asks for SciPy, set `OPTIVISION_EXTRAS=corpus,easy`.

## Python API

```python
from optivision import Config, OptiVisionRAG

rag = OptiVisionRAG(Config.load("colsmol"))
report = rag.build("path/to/pdfs")
print(report.compression_ratio)
print(rag.search("office memorandum on fire safety audit").hits[0].ref.page_id)
```

## Authors

T. Rithik Krishna, Amgovath Navanitha and Badavath Akhila, Mahatma Gandhi Institute
of Technology, Hyderabad. MIT licensed.
