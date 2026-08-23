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

The three stages multiply — under E1, pruning ~3.5× × quantization 32× ≈ **100×+** off
the index. On real ViDoRe pages pruning buys less, and the product lands in the 53-60× range; both
figures are in *Results* below.

## Results

We ran the ablation twice, and the two runs disagree. That disagreement is the
result — read both tables before drawing a conclusion from either.

### E1 — ColSmol-256M, generated pages

60 pages / 72 queries on CPU. Full table and analysis in
[docs/RESULTS.md](docs/RESULTS.md); reproduce with `make bench`.

| variant | tok/pg | KB/pg | compression | nDCG@5 | retained | tau |
|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7823 | 100.0% | 1.000 |
| spatial-only | 356.1 | 182.32 | 2.5x | 0.7602 | 97.2% | 0.935 |
| spatial+redundancy | 246.8 | 126.34 | 3.5x | 0.7519 | 96.1% | 0.866 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.7877 | 100.7% | 0.972 |
| **prune+int8** | **246.8** | **31.59** | **14.2x** | **0.7511** | **96.0%** | 0.864 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6875 | 87.9% | 0.585 |
| **optivision** | **246.8** | **3.95** | **113.5x** | **0.6782** | **86.7%** | 0.606 |
| optivision-aggressive | 186.3 | 2.98 | 150.3x | 0.6680 | 85.4% | 0.602 |

**Under this encoder, the two halves of the proposal do not contribute equally.**
Pruning is nearly free — 3.5x fewer vectors for 3.9% of nDCG@5.
Binary quantization is where the quality goes — 12.1% lost *without dropping a single
token*, while int8 gives 4x for 0.5%. Pruning harder barely matters once binary is in
play (keeping the top 10% of patches scores the same as keeping the top 50%).

So under E1 the pipeline has two defensible operating points:

- **smallest index** — prune + binary: **113.5x** at 86.7% of baseline nDCG@5
- **best quality per byte** — prune + int8: **14.2x** at 96.0%, with a Kendall tau of
  0.864 against 0.866 for pruning alone, so int8 adds no measurable ranking distortion on
  top of the pruning

448 KB/page becomes 3.95 KB/page: a million-page archive drops from ~448 GB to ~4 GB.
At the quality-first setting it is ~32 GB.

### E2 — ColPali-v1.3, real ViDoRe pages

**The reference model disagrees with every headline above.** The same ablation on
ColPali-v1.3 over four ViDoRe splits (1,780 pages, 1,325 queries) reaches 53-60x at
94.6-103.4% of baseline nDCG@5 — less compression, far less loss. Two of the three findings above do not transfer:
pruning buys only 1.7-1.9x on real pages rather than 3.5x, and binary quantization
costs 0-3.7% rather than 12.1%. Full table and analysis in
[docs/RESULTS.md](docs/RESULTS.md#the-same-table-on-colpali-3b-and-real-vidore-pages);
raw reports in [reports/](reports/); reproduce with `bash scripts/run_bench_gpu.sh` on
any free T4.

The takeaway is not "E2 supersedes E1". It is that **the per-stage attribution is a
property of the encoder and the corpus, not of the compression layer** — the same code
paid its quality in different places under the two. A third run separates those two
variables; see below. A compression ratio reported without naming the encoder and the
corpus it was measured on is not enough information to act on, which is what the paper
argues.

### E3 — ColPali-3B, generated pages

The missing cell: the reference encoder over E1's own corpus, regenerated at seed 7 so
the pages are the same ones. It splits the reversal in two.

| | corpus fixed, encoder swapped | encoder fixed, corpus swapped |
|---|---|---|
| one-bit codec costs | E1 → E3: **12.1 → 1.6 points** | E3 → E2: 1.6 → 3.7 points |
| pruning buys | E1 → E3: 3.55x → 4.20x | E3 → E2: **4.20x → 1.85x** |

**The corpus sets what pruning buys.** Run `python scripts/compare_regimes.py` to
print it from the benchmark files. E3 is also the best operating point in the project
(98.7% retention at 134.5x) and the least representative one.

The codec half is not an encoder property. Per query, a sign code adds the same ~3% of
score noise under both encoders, and it flips exactly the queries whose float margin over
the best competitor is smaller than that — none above twice the noise, on eight caches.
E1's precise queries are decided by three digits at a 0.05% margin, which is why the
small encoder looks fragile; ColPali cannot read those digits at 448 px and never wins
the queries it is credited with not losing (precise R@1 0.250, floor 0.200). Enlarge the
code and its one-bit cost appears. See [RESULTS.md](docs/RESULTS.md#e3---colpali-3b-on-the-generated-corpus)
and [REVIEW-2026-08-21.md](docs/REVIEW-2026-08-21.md); `scripts/tau_audit.py` explains why
E1's tau of 0.585 and E2's 0.527 were never the same measurement.

On dense pages, selecting tokens by embedding coverage beats pixel saliency below a 50%
budget — and 256 random probe directions do as well as the designed selector
([RESULTS.md](docs/RESULTS.md#token-selection-on-dense-pages---the-stage-ii-controls)).

### Memory at query time

This part holds under both experiments. A compressed index is worth nothing if
searching it expands the vectors back to float32, which costs 32x the index and
is what the obvious implementation does. Scoring here runs in blocks bounded by
a memory budget, so peak RAM is set by that budget rather than by the size of
the corpus — 239 MB at the 256 MB default, whether the index holds sixty pages
or a million. See [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

### Reading either table

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
