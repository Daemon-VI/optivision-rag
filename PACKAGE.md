# OptiVision RAG

**Quality-constrained multi-vector compression behind a model-agnostic interface.** (Model-agnostic here means one interface for any encoder's `[n, d]` float vectors; how well a pipeline compresses depends on the encoder and is measured each time.)

OptiVision compresses the index of a late-interaction (multi-vector, MaxSim)
retriever, such as a ColPali-style page encoder or a ColBERT-style text encoder,
to a quality target you set. Give it your encoder's vectors and a few hundred
real sample queries. It chooses token merging and quantization (dimension
reduction is an experimental opt-in, measured on one encoder), and reports the
retention it measured on queries that played no part in the choice.

```python
from optivision import MultiVectorCorpus, optimize

docs = MultiVectorCorpus.from_arrays(doc_vectors)       # list of [n_i, d] arrays
queries = MultiVectorCorpus.from_arrays(query_vectors)  # real sample queries (required)
result = optimize(docs, queries=queries, qrels=labels_or_None, quality_target=0.97)
```

Measured on held-out queries with default settings (20 random splits, labels; 500, 500
and 5,183 documents with 451, 494 and 300 queries):

| target | ColPali · ViDoRe DocVQA | ColPali · ViDoRe InfoVQA | ColBERT-small · SciFact (text) |
|---|---|---|---|
| 0.99 | 3.9x · met 20/20 · held-out 100.1% | 3.9x · met 20/20 · held-out 99.9% | 3.9x · met 14/14 · held-out 99.9% |
| 0.97 | 3.9x · met 19/20 · held-out 99.7% | 43.4x · met 20/20 · held-out 99.6% | 8.0x · met 20/20 · held-out 99.4% |
| 0.95 | 11.8x · met 20/20 · held-out 98.6% | 72.7x · met 20/20 · held-out 98.9% | 12.4x · met 20/20 · held-out 98.7% |

Each cell: median compression chosen (vs float32) · splits whose choice met the target on the held-out half · mean held-out retention. At SciFact 0.99 no configuration qualified in 6 of 20 splits.

Tested coverage is exactly six encoders:
- ColPali-v1.3, ColQwen2-v1.0 and ColQwen2.5-v0.2 (ViDoRe V1 DocVQA and InfoVQA,
  500 pages each);
- NVIDIA Nemotron ColEmbed 4B, 2,560-d (the same two ViDoRe V1 splits; licensed
  CC-BY-NC-4.0, non-commercial use only);
- ColSmol-256M (a 60-page generated corpus only);
- answerai-colbert-small-v1 (BEIR SciFact).

Other models' vectors are accepted, but their compression behaviour is
unmeasured. ColQwen3, ColNomic, 4,096-d models, ViDoRe V2/V3, million-page
corpora and database connectors beyond numpy/Qdrant are not validated.

There is no guarantee that a target is met. The confidence bound applies to each
configuration individually; choosing among many configurations is a
multiple-testing problem that it does not correct for. Measured corpora have
500–5,183 documents, and retention falls as a corpus grows. The measurements,
their limits and a release audit are in the
[repository](https://github.com/Daemon-VI/optivision-rag).

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
[quickstart](https://github.com/Daemon-VI/optivision-rag/blob/main/docs/package/guide/QUICKSTART.md) writes them and runs `optivision certify` and
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
Which models are covered, how, and what was measured: [models guide](https://github.com/Daemon-VI/optivision-rag/blob/main/docs/package/guide/MODELS.md).

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

Documentation, in the repository:

- [User guide](https://github.com/Daemon-VI/optivision-rag/blob/main/docs/package/guide/README.md)
- [Models: what is covered, and how far](https://github.com/Daemon-VI/optivision-rag/blob/main/docs/package/guide/MODELS.md)
- [API design](https://github.com/Daemon-VI/optivision-rag/blob/main/docs/package/API_DESIGN.md)
- [What the package and the research record do and do not support](https://github.com/Daemon-VI/optivision-rag/blob/main/docs/package/FINAL_RESEARCH_BOUNDARY.md)
- [Certificate format](https://github.com/Daemon-VI/optivision-rag/blob/main/docs/package/CERTIFICATE_SPEC.md)

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
| prune + binary | ColSmol-256M, 60 generated pages (72 queries) | 113.5x | 86.7% |
| prune + int8 | ColSmol-256M, 60 generated pages (72 queries) | 14.2x | 96.0% |
| prune + binary | ColPali-v1.3, 4 ViDoRe splits (1,780 pages, 1,325 queries) | 53-60x | 94.6-103.4% |

Each row is one fixed pipeline scored on every query of its corpus, with no choice made
from those queries. The 60-page corpus has 72 queries, so its figures carry intervals of
about 6 points (a sanity check, not evidence). Ratios and retention are specific to the
encoder and corpus named in each row.

Full results, the ablations and the analysis are in the
[repository](https://github.com/Daemon-VI/optivision-rag).

## Install

Python 3.10 or newer.

What was tested: the full test suite ran on Python 3.11 on one Windows laptop. The tests of the
new layer ran against the installed wheel on Python 3.10 (NumPy 1.26.4, SciPy 1.13.1), 3.12
(NumPy 2.0.2) and 3.13 (NumPy 2.5.3), all on the same Windows machine, and 3.14 had a smoke test
only. Not tested: Linux, macOS, big-endian machines, GPUs, a Qdrant server, and real encoder output
through the new layer. `qdrant-client` is a declared dependency of the package; what is optional is
using it: nothing imports it until a Qdrant index or backend is constructed, and no server is needed
otherwise.

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
