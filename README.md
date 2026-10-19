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

**Quality-constrained multi-vector compression behind a model-agnostic interface.** (Model-agnostic here means one interface for any encoder's `[n, d]` float vectors; how well a pipeline compresses depends on the encoder and is measured each time.)

> The YAML block above is Hugging Face Space metadata. It is ignored by GitHub
> and by every tool in this repo; see [docs/DEPLOY.md](docs/DEPLOY.md) for how
> the Space is set up.

---

OptiVision compresses the index of a late-interaction (multi-vector, MaxSim)
retriever — ColPali-style page encoders or ColBERT-style text encoders — **to a
quality target you set**. Give it the vectors your encoder produced and a few
hundred real sample queries. It chooses token merging and quantization (dimension
reduction is an experimental opt-in, measured on one encoder), and reports the
retention it measured **on queries that played no part in the choice**. The model
and the query path are unchanged.

```python
from optivision import MultiVectorCorpus, optimize

docs = MultiVectorCorpus.from_arrays(doc_vectors)       # your encoder's output: a list of [n_i, d] arrays
queries = MultiVectorCorpus.from_arrays(query_vectors)  # a few hundred real queries (required)
result = optimize(docs, queries=queries, qrels=labels_or_None, quality_target=0.97)
result.pipeline.label(), result.compression_ratio, result.quality_retention
```

### What the optimizer chose, measured on held-out queries

`calibrate()` / `optimize()` with default settings, labels as the reference,
repeated over 20 random calibration / held-out splits of each query set. This is
what it did on these three encoder and corpus pairs (500, 500 and 5,183 documents;
451, 494 and 300 queries), not a prediction for another encoder or corpus
([docs/UNIVERSAL.md](docs/UNIVERSAL.md), R7b):

| target | ColPali · ViDoRe DocVQA | ColPali · ViDoRe InfoVQA | ColBERT-small · SciFact (text) |
|---|---|---|---|
| 0.99 | 3.9x · met 20/20 · held-out 100.1% | 3.9x · met 20/20 · held-out 99.9% | 3.9x · met 14/14 · held-out 99.9% |
| 0.97 | 3.9x · met 19/20 · held-out 99.7% | 43.4x · met 20/20 · held-out 99.6% | 8.0x · met 20/20 · held-out 99.4% |
| 0.95 | 11.8x · met 20/20 · held-out 98.6% | 72.7x · met 20/20 · held-out 98.9% | 12.4x · met 20/20 · held-out 98.7% |

Each cell: median compression chosen (vs float32) · splits whose choice met the target on the held-out half · mean held-out retention.

At SciFact 0.99 no configuration qualified in 6 of 20 splits; `optimize()` then
returns the float corpus and says so. The default trades compression for
reliability on purpose: choosing by the point estimate instead picked 2.3–17x
more compression and met the target in only 20–85% of splits (R7b). Twenty
splits cannot pin down a rate near 95%, and none of this is a guarantee (R10).

### Fixed configurations, measured on all queries

One named pipeline each, scored on every query of the same three corpora, with no
choice made from these queries. They are only unbiased for a pipeline you pick
*in advance*, and reading the best row off the table is itself a selection:

| pipeline | ColPali · ViDoRe DocVQA | ColPali · ViDoRe InfoVQA | ColBERT-small · SciFact (text) |
|---|---|---|---|
| int8 per-vector (3.9x) | 100.0% | 100.0% | 100.1% |
| Ward merge to 1/3 of the tokens + int8 per-vector (11.8x) | 99.2% [97.2, 101.1] | 99.4% [98.6, 100.2] | 100.0% [98.5, 101.6] |
| binary codes (32x) | 96.3% | 97.4% | 94.1% |
| adaptive merge r=0.6 + int4 (65–69x on ColPali) | 95.0% | 99.2% | **7.2%**: the absolute radius collapses this encoder |
| merged binary codes in RAM + int8 rescoring of 50 | 99.1–101%, RAM 110–376x smaller | 99.8–100%, RAM 117–406x smaller | not measured |

Retention = nDCG@5 (ColPali) or nDCG@10 (SciFact) of the compressed index ÷ the
float32 index, exact MaxSim, with 95% bootstrap intervals over queries. The
two-tier row saves **RAM**: its disk tier keeps int8 codes of every vector, so
total storage is about 3.9x smaller, not 110x. The same pipeline can be nearly
free on one corpus and costly on another (row 4). That is why the package
measures instead of prescribing.

### Tested coverage

`optimize()` has been evaluated end to end, on held-out queries, on exactly six encoders:

- **ColPali-v1.3**: ViDoRe V1 DocVQA and InfoVQA (500 pages each, 451 and 494
  queries) and a generated corpus (60 pages).
- **ColQwen2-v1.0** and **ColQwen2.5-v0.2**: the same two ViDoRe V1 splits, on the
  same queries. Their float baselines are within 0.5–1.8 points of the published
  ViDoRe scores. Out of sample, the default met its target in 236 of 240 choices,
  for example 114–122x at a 0.95 target on InfoVQA
  ([docs/UNIVERSAL.md](docs/UNIVERSAL.md), R11).
- **NVIDIA Nemotron ColEmbed 4B** (2,560-d, licensed CC-BY-NC-4.0, so
  non-commercial use only): the same two ViDoRe V1 splits. The default met its
  target in 119 of 120 held-out choices
  ([docs/UNIVERSAL.md](docs/UNIVERSAL.md), R12). OptiVision has thus been
  evaluated empirically on a 2,560-dimensional multi-vector retrieval model, and
  its compression and optimization machinery kept working at that
  width. That is one model on two 500-page splits; nothing wider is measured.
- **ColSmol-256M**: the generated corpus only (60 pages, 72 queries, about ±6
  points); a sanity check, not evidence.
- **answerai-colbert-small-v1** (text): BEIR SciFact (5,183 abstracts, 300 queries).

Beyond those six, as of 0.5.0 the catalogue (`optivision models`) holds fixed-pipeline
measurements for 55 models: 40 page models on ViDoRe V1 DocVQA and InfoVQA and 15 text models on
BEIR SciFact, 48 to 4,096 dimensions, ColQwen3, ColNomic and 4,096-d models among them. Those
rows say how fixed pipelines kept retention on those corpora; they show what to try, not what
`optimize()` or `certify()` will choose and hold on yours. Follow-up batches added NFCorpus,
SciDocs, ArxivQA and TabFQuAD for some models, and held-out checks of `calibrate()` on 22 units
(`reports/universal/followup/SUMMARY.md`, P8).

Any other model's vectors go in through `from_arrays`, but how they compress is
**unmeasured** until someone runs `scripts/universal_study/`. Corpora so far
have 500–25,657 documents, and retention measurably falls as distractors are
added (release audit, §7). ViDoRe V2/V3, million-page corpora and database
connectors beyond numpy/Qdrant are **future work**, not results.

### What is not claimed

- No guarantee that the target is met: the 0.999 lower bound is an approximate
  bootstrap bound for one configuration, and it does not cover choosing among
  many, queries unlike your sample, or a corpus that grows. The held-out
  numbers above are the evidence.
- No distribution-free or model-independent result. Every figure is for the
  encoder, corpus and queries named next to it.
- The largest ratios in the write-up (R5b) are **in-sample**: the best of 60
  configurations, chosen and scored on the same queries. They are upper
  references, not predictions.

Write-up with every table: [docs/UNIVERSAL.md](docs/UNIVERSAL.md). Release audit:
[docs/RELEASE-AUDIT-2026-09-27.md](docs/RELEASE-AUDIT-2026-09-27.md). CLI:
`optivision inspect | compress | calibrate | benchmark | compare`.

## Model-agnostic core and certified selection (0.4.0)

`pip install "optivision-rag[merge]"` installs it (the extra adds SciPy, which the default
candidate list needs). It is engineering capability, tested on synthetic data and against the
research code for numerical equality; no registered experiment evaluated it as a package.

What was added, for any late-interaction encoder that hands over one `[n_vectors, dim]`
float array per document and per query, scored by MaxSim (the research record covers
L2-normalised vectors only):

- **a vector interface** (`optivision.core`): validation and content identities, the same
  for every encoder;
- **composable stages** (token reducers, projections, codecs), each with declared
  structural traits, including which changes of coordinate basis leave its scores
  unchanged;
- **a frozen, hashed candidate list** of pipelines (the default list was shaped by
  results on the development encoders);
- **a decision procedure** that selects one pipeline from that list or abstains, under an
  error level you state, against relevance labels or against agreement with the
  uncompressed ranking (agreement is not retrieval quality);
- **a certificate** that records the decision and its assumptions and can be verified
  (`optivision certify`, `optivision verify`), and an artifact directory holding the
  codes and the fitted state;
- **a retrieval seam** (queries in, ranked ids out) and wrappers over the NumPy and Qdrant
  indexes (`optivision.backends`).

`optimize()` and `calibrate()` are unchanged. A model-agnostic interface is not
model-independent compression quality: whether a pipeline keeps retrieval quality depends
on the encoder, the coordinate basis of its vectors, the corpus and the queries, and is
measured each time; the procedure may abstain. A certificate carries the selection rule's
own statement, claimed only if the assumptions and conditions it lists hold. It is not a statement that the
compressed index keeps retrieval quality, and it does not carry over to another encoder,
corpus or coordinate basis.

To try it:

```python
from optivision import certify, verify
from optivision.testing import make_corpus, make_queries

# Synthetic vectors exercise the code path; they say nothing about any encoder.
docs = make_corpus(300, 128, tokens=(10, 20), seed=0)
queries, qrels = make_queries(docs, 200, noise=0.3, seed=1)

result = certify(docs, queries, qrels, target=0.97, error=0.05)   # default list needs SciPy
result.outcome, result.certified   # SELECT on this run; ABSTAIN is a normal outcome too
result.save("artifact/")           # certificate, codes, fitted state, evidence
verify("artifact/").ok             # integrity, replay from stored evidence (built-in rules), binding to stored codes
```

`certify` needs SciPy for its default candidate list: `pip install "optivision-rag[merge]"` (from a
clone, `pip install -e ".[merge]"`). In the API, `space=CandidateSpace.default(dim, allow_reduced=True)` and, on the command line, the flag
`--allow-reduced` use a smaller candidate list that does not need SciPy; the certificate records which
list was used. The command line reads `.npz` files (a corpus saved with `MultiVectorCorpus.save`, the
same for the queries) and a JSON file of `{query id: [relevant document ids]}`; section 6 of the
[quickstart](docs/package/guide/QUICKSTART.md) writes them and runs `optivision certify` and
`optivision verify` on them.

### Model coverage

The registry (`optivision.adapters.MODELS`, `supported_models()`) lists 65 late-interaction
models: 50 that take page images and 15 that take text. Each has a status that describes the
loading path and the measurements, not how well a pipeline compresses that model's vectors:

- `measured` (10): retrieval quality with and without compression was measured by this
  repository on the model's real output. These are in-sample development measurements on ViDoRe
  V1 DocVQA and InfoVQA (500 pages each) or SciFact; they are not a statement about another
  corpus.
- `untested` (23): the loading code is the one a measured sibling uses, or the model passed a
  `check_retriever` run here; compression of this checkpoint's vectors was never measured here.
- `unverified` (32): the loader is written against the model card and was never run here.

`optivision models` prints the registry and the measured catalogue. `encode_and_certify(model,
documents, queries, qrels)` encodes with a model and then runs `certify` on the vectors;
`load_adapter(model_id, loader=...)` accepts a checkpoint that is not in the registry, and
`check_retriever` encodes a dozen matched pairs and reports whether an adapter behaves like a
retriever (it shows the model was driven correctly, not that compression works). The measured
catalogue (`measured_models`, `measured_results`, `measured_shortlist`) ships with the package.
Which models are covered, how, and what was measured: [models guide](docs/package/guide/MODELS.md).

What the research record shows, and does not show:

- X1-X3 were run on the seven development stores (three visual encoders on DocVQA and
  InfoVQA, one text encoder on SciFact), one query pool per corpus, and the registered
  list of 40 pipelines. H4a-c (agreement with the uncompressed ranking as a reference) and
  the primary setting of H3 (non-trivial compression at T = 0.97, delta = 0.05) were
  supported, mostly as
  resamples of one query pool per corpus and within that pool; **H5, invariance of the
  outcome under the registered maps of the vectors, failed**: 13 of 16 primary pairs
  hold, and the three that do not are rotations. At a 0.99 target the registered
  deployment criterion was not met on any store at pool-sized budgets.
- Transfer to encoders that took no part in the design has **not** been demonstrated:
  X1-X3 reuse the encoders that shaped the candidate list.
- XH, the held-out encoder protocol, has not been registered and has not been run.

See the [user guide](docs/package/guide/README.md), the
[API design](docs/package/API_DESIGN.md), the
[boundary between the package and the research](docs/package/FINAL_RESEARCH_BOUNDARY.md)
and the [certificate format](docs/package/CERTIFICATE_SPEC.md).

## The original recipe (v0.1): pruning and binary codes for ColPali pages

Everything below this point describes the first release: one fixed recipe for
page-image encoders. It still ships unchanged, and the universal layer wraps its
stages. ("Universal" in this README names the encoder-independent interface of that
layer, not a compression result that holds for every encoder.)

### The problem

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

### What this project does

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

## Results of the original recipe

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
token*, while int8 gives 4x with no measured loss (100.7%, table above). Pruning harder
barely matters once binary is in play (keeping the top 10% of patches scores the same as
keeping the top 50%).

So under E1 the pipeline has two defensible operating points:

- **smallest index** — prune + binary: **113.5x** at 86.7% of baseline nDCG@5
- **best quality per byte** — prune + int8: **14.2x** at 96.0%, with a Kendall tau of
  0.864 against 0.866 for pruning alone, so int8 adds no measurable ranking distortion on
  top of the pruning

448 KB/page becomes 3.95 KB/page: multiplied out, a million-page archive would drop from
~448 GB to ~4 GB, and to ~32 GB at the quality-first setting. That is arithmetic from a
60-page corpus (PROJECTED), not a measurement at that size, and retention at that size
was not measured.

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
print it from the benchmark files (its closing line about what the codec costs is the
earlier reading, superseded by the next paragraph). E3 is also the best operating point
in the project (98.7% retention at 134.5x, on 60 pages and 72 queries, so about 6
points either way) and the least representative one.

The codec half is not an encoder property. Per query, a sign code adds the same ~3% of
score noise under both encoders, and it flips only queries whose float margin over
the best competitor is smaller than that noise — under 1% of 945 real ViDoRe queries
above twice the noise, against a third of those below half.
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
a memory budget, so peak RAM follows that budget instead of the decoded size of
the index: 239 MB at the 256 MB default, measured on a 4,000-page index (1M
vectors) scored one query at a time. The stored codes still grow with the corpus,
and so does the queries-by-pages score matrix of batched scoring; an index of a
million pages was not measured. See [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

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

From PyPI, as a library and the `optivision` CLI:

Python 3.10 or newer.

```bash
pip install optivision-rag             # CLI + the offline synthetic backend, no model download
pip install "optivision-rag[corpus]"   # + make-corpus (reportlab)
pip install "optivision-rag[vlm]"      # + the real ColSmol / ColPali / ColQwen2 / ColQwen2.5 encoders (torch)
pip install "optivision-rag[merge]"    # + Ward merging (SciPy): the measured default search space of optimize()
```

The example configs ship inside the package as presets, so `-c synthetic`, `-c colsmol`,
`-c colpali` and `-c qdrant` work without a clone.

What was tested: the full test suite ran on Python 3.11 on one Windows laptop. The tests of the
new layer ran against the installed wheel on Python 3.10 (NumPy 1.26.4, SciPy 1.13.1), 3.12
(NumPy 2.0.2) and 3.13 (NumPy 2.5.3), all on the same Windows machine, and 3.14 had a smoke test
only. Not tested: Linux, macOS, big-endian machines, GPUs, a Qdrant server, and real encoder output
through the new layer. `qdrant-client` is a declared dependency of the package; what is optional is
using it: nothing imports it until a Qdrant index or backend is constructed, and no server is needed
otherwise.

Without installing anything, `npx optivision-rag --help` runs the same CLI: it installs the
matching PyPI release into a private virtualenv on first use (Python 3.10+ required; it is a
launcher, not a JS port).

A VS Code extension, `daemon-vi.optivision-rag` (Marketplace and Open VSX), puts the CLI's
commands in a control panel. It drives whichever `optivision` is on your PATH, so install the
package first. Its source is in [vscode-extension/](vscode-extension/).

Release steps for maintainers are in [docs/RELEASING.md](docs/RELEASING.md).

For development, from a clone:

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
| `configs/colqwen2.yaml`* | ColQwen2-2B | GPU | encoder measured in the universal layer (R11); this config's index/search path not run here |
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
    colvlm.py         ColPali / ColQwen2 / ColQwen2.5 / ColSmol via colpali-engine
    synthetic.py      offline stand-in (tests only)
  pruning/
    saliency.py       ink density + edge energy per patch
    spatial.py        keep-mask construction, dilation, budgets
    redundancy.py     greedy near-duplicate collapsing
  compression/
    binary.py         bit packing, asymmetric/symmetric MaxSim
  index/
    numpy_index.py    exact brute-force MaxSim (reference of the original recipe)
    qdrant_index.py   Qdrant multivector + binary quantization
  pipeline.py         OptiVisionRAG.build() / .search()
  bench.py            encode-once, replay-every-variant ablation harness
  corpus.py           synthetic corpus generator + ViDoRe loader
  metrics.py          nDCG / recall / MRR / Kendall tau / storage
  viz.py              keep-mask and saliency figures

  # universal layer (see docs/UNIVERSAL.md)
  representation.py   MultiVectorCorpus: any model's vectors, contiguous + offsets
  adapters/           model registry (measured / untested / unverified) + adapters
  stages/             merge.py, prune.py, project.py, quantize.py — composable stages
  compose.py          Pipeline, CompressedCorpus
  scoring.py          exact MaxSim over any store; decode and similarity blocks memory-bounded
  evaluation.py       per-query metrics, labels or float-baseline reference, bootstrap
  calibration.py      calibrate(): choose on some queries, report on others
  optimize.py         optimize(): the one-call entry point
  pareto.py           non-dominated configurations
  storage.py          ExactIndex, TieredIndex, StorageBackend
  benchmark.py        datasets + result matrices
  cli_universal.py    inspect / compress / calibrate / benchmark / compare

  # model-agnostic core and certified selection (0.4.0)
  core/               vector contract, stages with declared traits, candidate list and
                      selection rule, certificates, retrieval seam
  backends/           Backend protocol; wrappers over NumpyIndex and QdrantIndex
  api.py              certify(), verify()
  cli_certify.py      optivision certify / verify
  testing/            synthetic data and conformance checks for new adapters, stages, backends
  ext/, legacy/       namespaces that say which layer a name belongs to (no code moved)
app/streamlit_app.py  demo UI
docs/                 architecture, results, viva notes, AUDIT-2026-09-27, UNIVERSAL,
                      package/ (design, specifications and guide of the core)
notebooks/            Colab / Kaggle runner for the ColPali benchmark
reports/universal/    every measured number of the universal layer (JSON + tables)
scripts/universal_study/  the scripts that produced them
```

## Testing

```bash
pytest              # no model download needed
ruff check src tests app
```

The suite covers saliency behaviour on blank/grey/inked pages, keep-mask budgets,
redundancy clustering invariants, bit-packing round-trips, MaxSim segment maths on
variable-length pages, index save/load, Qdrant multivector round-trips, and a full
index→search→evaluate loop. For the universal layer it adds: the refactored
stages reproducing the original pipeline byte for byte, exact MaxSim against
brute force (empty pages and queries included), merge and quantizer invariants,
projection of queries, calibration on disjoint splits, serialisation, the CLI,
and a guard that no model is marked *measured* without results.

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
