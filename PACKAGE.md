# OptiVision RAG

**Quality-constrained, model-agnostic multi-vector compression.**

OptiVision compresses the index of a late-interaction (multi-vector, MaxSim)
retriever, such as a ColPali-style page encoder or a ColBERT-style text encoder,
to a quality target you set. Give it your encoder's vectors and a few hundred
real sample queries. It chooses token merging, dimension reduction and
quantization, and reports the retention it measured on queries that played no
part in the choice.

```python
from optivision import MultiVectorCorpus, optimize

docs = MultiVectorCorpus.from_arrays(doc_vectors)       # list of [n_i, d] arrays
queries = MultiVectorCorpus.from_arrays(query_vectors)  # real sample queries (required)
result = optimize(docs, queries=queries, qrels=labels_or_None, quality_target=0.97)
```

Measured on held-out queries with default settings (20 random splits, labels):

{{PACKAGE_HELDOUT_TABLE}}

Tested coverage is exactly three encoders: ColPali-v1.3 (ViDoRe V1 DocVQA and
InfoVQA, 500 pages each), ColSmol-256M (a 60-page generated corpus only) and
answerai-colbert-small-v1 (BEIR SciFact). Other models' vectors are accepted,
but their compression behaviour is unmeasured. There is no guarantee that a
target is met. The measurements, their limits and a release audit are in the
[repository](https://github.com/Daemon-VI/optivision-rag).

## The original recipe: pruning and binary codes for page images

ColPali-style models search scanned documents as images, with no OCR. They store
one 128-dimensional float vector per image patch, so a single page costs about
512 KB of index.

The first release shrinks the **index**, not the model. The published checkpoint runs
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
| prune + binary | ColSmol-256M, generated pages | 113.5x | 86.7% |
| prune + int8 | ColSmol-256M, generated pages | 14.2x | 96.0% |
| prune + binary | ColPali-v1.3, ViDoRe (4 splits) | 53-60x | 94.6-103.4% |

Full results, the ablations and the analysis are in the
[repository](https://github.com/Daemon-VI/optivision-rag).

## Install

Python 3.10 or newer.

```bash
pip install optivision-rag                  # core pipeline + CLI
pip install "optivision-rag[corpus]"        # + synthetic test-corpus generator
pip install "optivision-rag[vlm]"           # + real encoders (torch, colpali-engine)
pip install "optivision-rag[merge]"         # + Ward merging (SciPy), used by optimize()'s default search
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
