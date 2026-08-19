---
title: OptiVision RAG
emoji: 🗜️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.24.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Visual-token pruning and binary quantization for document retrieval
---

# OptiVision RAG

**Extreme token compression for vision-language document retrieval.**

> The YAML block above is Hugging Face Space metadata. It is ignored by GitHub
> and by every tool in this repo; see [docs/DEPLOY.md](docs/DEPLOY.md) for how
> the Space is set up.

Project Stage-I (CD753PC) · B.Tech CSE (Data Science) · Dept. of Emerging Technologies
T. Rithik Krishna (23261A6753) · Amgovath Navanitha (23261A6704) · Badavath Akhila (23261A6709)
Guide: Ms. E. Sathiya Lakshmi

---

## The problem

ColPali-style models search scanned documents *as images* — no OCR, so they work on
old copies, seals, stamps and handwriting that text pipelines fail on. They do it by
splitting each page into patches and keeping **one vector per patch**.

That is also the problem. A single page becomes ~800–1000 vectors of 128 float32
dimensions:

```
1 page  =  1024 vectors x 128 dims x 4 bytes  =  512 KB
1 million pages                               =  512 GB
```

Half a terabyte of RAM-resident vector index for a corpus that a records office
would consider small. The accuracy is excellent; the storage is what stops it
being deployable.

## What this project does

OptiVision RAG shrinks the **index**, not the model. The published checkpoint runs
unmodified, the query path is untouched, and the retrieval is still late-interaction
MaxSim — only the number and the size of the stored vectors change.

```
page image
   │
   ├─ VLM encoder ─────────► ~1000 patch vectors        (model unchanged)
   │
   ├─ 1. spatial pruning ──► drop patches on blank paper
   │
   ├─ 2. redundancy prune ─► collapse near-duplicate patches
   │
   ├─ 3. binary quantize ──► 128 floats (512 B) → 128 bits (16 B)
   │
   └─ index ───────────────► Qdrant MaxSim, or an exact numpy index
```

**1. Spatial pruning.** Most of a document page is paper. Per patch we measure ink
density (how much is darker than the estimated paper background) and edge energy
(local contrast — glyph strokes, rules, stamp borders), and drop the cells that carry
neither. The estimate runs on the pixels in microseconds, before the model, and needs
no attention maps or second forward pass. The paper background is estimated per page
(95th percentile) rather than assumed white, so a yellowed photocopy is not read as a
fully inked page.

**2. Redundancy pruning.** Spatial pruning removes patches with *no* content;
this removes patches with *duplicate* content — the interior of a filled table cell,
the middle of a thick rule. They look salient to a pixel detector but add nothing to a
MaxSim score, because MaxSim already takes a max over near-identical vectors. A greedy
single pass collapses each cluster to its renormalised centroid.

**3. Binary quantization.** These vectors are L2-normalised and roughly zero-centred
per dimension, so the sign bits keep the orthant and discard only the within-orthant
position. Crucially, the distortion is nearly the same for every document vector, so
it moves scores much more than it moves *ranking*. Queries stay in float32 and are
scored against ±1 document codes (asymmetric scoring): a corpus has millions of
vectors, a query has twenty, so query precision is the cheapest thing to keep.

The three stages multiply. Pruning ~3.5× × quantization 32× ≈ **100×+** off the index.

## Results

Measured on 60 pages / 72 queries with **ColSmol-256M** on CPU. Full table and
analysis in [docs/RESULTS.md](docs/RESULTS.md); reproduce with `make bench`.

| variant | tok/pg | KB/pg | compression | nDCG@5 | retained | tau |
|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7823 | 100.0% | 1.000 |
| spatial-only | 356.1 | 182.32 | 2.5x | 0.7602 | 97.2% | 0.935 |
| spatial+redundancy | 246.8 | 126.34 | 3.5x | 0.7519 | 96.1% | 0.866 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.7787 | 99.5% | 0.973 |
| **prune+int8** | **246.8** | **31.59** | **14.2x** | **0.7596** | **97.1%** | 0.866 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6875 | 87.9% | 0.585 |
| **optivision** | **246.8** | **3.95** | **113.5x** | **0.6782** | **86.7%** | 0.606 |
| optivision-aggressive | 186.3 | 2.98 | 150.3x | 0.6680 | 85.4% | 0.602 |

**The two halves of the proposal do not contribute equally, and saying so is the
honest result.** Pruning is nearly free — 3.5x fewer vectors for 3.9% of nDCG@5.
Binary quantization is where the quality goes — 12.1% lost *without dropping a single
token*, while int8 gives 4x for 0.5%. Pruning harder barely matters once binary is in
play (keeping the top 10% of patches scores the same as keeping the top 50%).

So the pipeline has two defensible operating points:

- **smallest index** — prune + binary: **113.5x** at 86.7% of baseline nDCG@5
- **best quality per byte** — prune + int8: **14.2x** at 97.1%, with a Kendall tau of
  0.866 — identical to pruning alone, so int8 adds no measurable ranking distortion on
  top of the pruning

448 KB/page becomes 3.95 KB/page: a million-page archive drops from ~448 GB to ~4 GB.
At the quality-first setting it is ~32 GB.

**And it stays small at query time.** A compressed index is worth nothing if
searching it expands the vectors back to float32, which costs 32x the index and
is what the obvious implementation does. Scoring here runs in blocks bounded by
a memory budget, so peak RAM is set by that budget rather than by the size of
the corpus — 239 MB at the 256 MB default, whether the index holds sixty pages
or a million. See [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

The columns that matter:

| column | what it means |
|---|---|
| `Tok/pg` | vectors actually stored per page |
| `KB/pg` | index bytes per page |
| `Compr.` | vs. the uncompressed float32 index |
| `nDCG@5`, `R@1` | absolute retrieval quality against ground truth |
| `Retain` | nDCG@5 as a fraction of the uncompressed baseline's |
| `Tau` | Kendall tau against the **baseline's own ranking** |

`Tau` is the honest measure of compression damage: it does not ask whether the model
was right, only whether compressing changed its mind.

## Demo app

A single-page Gradio demo (`app.py`) shows the compression happening on one
uploaded document — built for the Stage-I presentation, not as a production RAG
service.

Double-click `run_demo.bat`, or:

```bash
pip install -e ".[vlm,app]" && pip install gradio
python app.py                       # http://127.0.0.1:7860
```

Upload a PDF or scan, press **Compress Document**, and it reports the real token
counts, the real byte sizes, and renders the pipeline's actual keep-mask over the
page. Every figure is read back off the arrays the pipeline produced.

It runs **locally**: Hugging Face now requires a PRO subscription to host any live
Gradio Space, free CPU hardware included. `app.py` and `scripts/deploy_space.py`
are Space-ready for whenever that changes — see [docs/DEPLOY.md](docs/DEPLOY.md).

The demo refuses to run on the `synthetic` backend: showing a regression harness's
output as a demonstration result would misrepresent the project.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[vlm,bench,app,dev]"
```

`torch` is CPU-only by default. For a GPU box install the CUDA build first
(`pip install torch --index-url https://download.pytorch.org/whl/cu121`).

## Quick start

```bash
# 1. build a corpus of scanned-looking documents with ground-truth queries
optivision make-corpus data/corpus --docs 30 --pages 2

# 2. index it  (colsmol.yaml = real 256M model, runs on a CPU laptop)
optivision index data/corpus/pdfs -c configs/colsmol.yaml

# 3. search
optivision search "renewal of vehicle insurance policy" -c configs/colsmol.yaml

# 4. see exactly which patches were dropped
optivision explain data/corpus/pdfs -c configs/colsmol.yaml --out reports/figures

# 5. run the full ablation table
#    --cache stores the encode pass, so re-running to add or change a row
#    takes seconds instead of re-encoding the whole corpus
optivision bench data/corpus/pdfs data/corpus/queries.json \
    -c configs/colsmol.yaml --out reports/colsmol --sweep \
    --cache data/cache/colsmol.npz

# 6. demo UI
streamlit run app/streamlit_app.py -- --config configs/colsmol.yaml
```

Point step 2 at your own folder of PDFs or scans to index real documents.

For the reference numbers — ColPali-v1.3 over ViDoRe splits — you need a GPU.
Two paths, same config and same code:
`notebooks/vidore_colpali_bench.ipynb` (Colab or Kaggle, interactive) and
`scripts/run_bench_gpu.sh` (any Ubuntu + CUDA box, unattended, all four splits).
See [docs/GPU_RUN.md](docs/GPU_RUN.md) — a rented GPU costs about $0.25 for the
whole benchmark, which is less trouble than the free tiers' quotas.

## Configurations

| config | encoder | needs | use for |
|---|---|---|---|
| `configs/synthetic.yaml` | hashed stand-in | nothing | tests, CI, first smoke run |
| `configs/colsmol.yaml` | ColSmol-256M | ~0.5 GB download, CPU ok | **default** — real results on a laptop |
| `configs/colqwen2.yaml`* | ColQwen2-2B | GPU | stronger quality |
| `configs/colpali.yaml` | ColPali-v1.3 | GPU (~6 GB) | reference model from the paper |
| `configs/qdrant.yaml` | ColSmol + Qdrant | optional server | deployment-shaped storage |

\* generate with `optivision init-config configs/colqwen2.yaml --backend colqwen2`.

The **synthetic** encoder deserves a warning: it makes the whole pipeline runnable
with no downloads and its retrieval genuinely works, so it is a real correctness
harness — but its hashed word vectors are near-orthogonal, which makes MaxSim behave
like exact matching. Quality metrics there saturate at 1.0 and **must not be reported
as results**. Use it to test plumbing; use ColSmol or ColPali for numbers.

## Layout

```
src/optivision/
  types.py            PageEncoding → PrunedPage → CompressedPage
  config.py           every experimental knob, YAML-loadable
  encoders/
    colvlm.py         ColPali / ColQwen2 / ColSmol via colpali-engine
    synthetic.py      offline stand-in (tests only)
  pruning/
    saliency.py       ink density + edge energy per patch
    spatial.py        keep-mask construction, dilation, budgets
    redundancy.py     greedy near-duplicate collapsing
  compression/
    binary.py         bit packing, asymmetric/symmetric MaxSim
  index/
    numpy_index.py    exact brute-force MaxSim (the reference)
    qdrant_index.py   Qdrant multivector + binary quantization
  pipeline.py         OptiVisionRAG.build() / .search()
  bench.py            encode-once, replay-every-variant ablation harness
  corpus.py           synthetic corpus generator + ViDoRe loader
  metrics.py          nDCG / recall / MRR / Kendall tau / storage
  viz.py              keep-mask and saliency figures
app/streamlit_app.py  demo UI
docs/                 architecture, results, viva notes
notebooks/            Colab / Kaggle runner for the ColPali benchmark
```

## Testing

```bash
pytest              # 75 tests, no model download needed
ruff check src tests app
```

The suite covers saliency behaviour on blank/grey/inked pages, keep-mask budgets,
redundancy clustering invariants, bit-packing round-trips, MaxSim segment maths on
variable-length pages, index save/load, Qdrant multivector round-trips, and a full
index→search→evaluate loop.

## Improvements

[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) records five changes made after the
pipeline was working — bounded query-time memory, an int8 quantizer that uses
the range it pays for, a single-allocation decode path, a Qdrant stats fix, and
one quadratic loop — each with the measurement behind it, plus what was
reviewed and deliberately left alone.

## Honest limitations

- **Encoder speed on CPU.** ColSmol-256M takes ~30 s/page on this laptop (no GPU).
  Pruning and quantization together take ~15 ms/page — the compression is free
  relative to encoding, but building a large index needs a GPU.
- **The bundled corpus is generated**, not scanned. It has the right whitespace
  profile and gives exact ground truth, but real scans have noise, skew and bleed-through.
  `optivision fetch-vidore` pulls the real ViDoRe benchmark for that reason.
- **Redundancy merging changes the vectors**, not just their count. It is nearly free
  under MaxSim but it is not lossless — the ablation reports both stages separately so
  the cost is visible.
- **Qdrant local mode** is convenient but not a performance claim; latency numbers
  come from the exact numpy index so they are not confounded by ANN recall.

## References

- Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models* (2024)
- Khattab & Zaharia, *ColBERT* (2020) — late interaction / MaxSim
- Qdrant multivector + binary quantization documentation
