# OptiVision universal layer

> **Thesis.** OptiVision is a model-agnostic optimization layer for multi-vector
> retrieval: given the vectors a retriever produced and a sample of queries, it
> chooses token reduction, dimensionality reduction and quantization subject to
> a retrieval-quality target, and reports the quality it actually measured on
> queries that played no part in the choice.

This document separates three kinds of statement and never mixes them:

| label | meaning |
|---|---|
| **MEASURED** | produced by this repository's code on real encoder output; the JSON is in `reports/universal/` |
| **EXTERNAL** | reported by a paper or model card, not reproduced here |
| **PROJECTED / FUTURE** | not measured by anyone here; a hypothesis or a plan |

Everything under *Results* is MEASURED. Tested model coverage is exactly the
models listed as measured below — nothing else is claimed.

A MEASURED number is one of three kinds, and they are not interchangeable. Each
results section says which kind it holds:

| kind | what it is | how to read it | where |
|---|---|---|---|
| **fixed configuration** | one named pipeline, scored on every query | an unbiased estimate for that pipeline *if you had picked it in advance*; reading the best row off a table of many is itself a selection | R1–R4, R5, R6, R8, R9, R11, R12 |
| **in-sample selection** | the best of many configurations, chosen and reported **on the same queries** | optimistic by construction (winner's curse); an upper reference, not a prediction | R5b, the R12 Pareto frontier |
| **out-of-sample** | chosen on calibration queries, reported on **disjoint held-out** queries, over 20 random splits | what `calibrate()` / `optimize()` will do for you | R7, R7b, R10, R11, R12 |

For planning, use the **out-of-sample** numbers. The release audit
([RELEASE-AUDIT-2026-09-27.md](RELEASE-AUDIT-2026-09-27.md)) traces every
headline figure to its experiment.

## Architecture

```
 any encoder ──► vectors ──► MultiVectorCorpus ──► Pipeline ─────────────────────────────► CompressedCorpus ──► index / DB
 (ColPali, ColSmol,          [N, d] + offsets      token reducers  (merge / prune)          uint8 codes          ExactIndex
  text ColBERT, your own)    + optional:           dimension reducers (PCA / random / cut)  + byte accounting    TieredIndex (hot/cold)
                             protected rows,       one quantizer (fp16 / int8 / int4 /                           legacy numpy / Qdrant
                             grid positions,         2-bit / binary)
                             importance, images
                                    │
            sample queries ─────────┴──► calibrate(): score candidates once, choose on a calibration split,
            (+ labels, optional)          report on a held-out split  ──►  optimize() returns the compressed corpus
```

| module | role |
|---|---|
| `representation.py` | `MultiVectorRepresentation` (one document) and `MultiVectorCorpus` (contiguous matrix + offsets, zero-copy views, pickle-free `.npz`) |
| `adapters.py` | model registry with a status per model; `PageEncoderAdapter` (ColPali family), `SentenceTransformersAdapter` |
| `stages/` | `SpatialPruner`, `RedundancyPruner` (original recipe), `AdaptiveMerge`, `HierarchicalMerge`, `RandomPruner` (control), `PCAProjector`, `RandomProjector`, `TruncateProjector`, `Float32/Float16/Int8/Int4/Lloyd2/BinaryQuantizer` |
| `compose.py` | `Pipeline` (quantizer last; queries pass only through projections), `CompressedCorpus` |
| `scoring.py` | exact MaxSim over any store, memory-bounded blocks |
| `evaluation.py` | per-query nDCG / recall / MRR / hit, labels or float-baseline reference, paired bootstrap retention, query splits |
| `calibration.py` | `calibrate()`, `score_space()`, `select()` |
| `optimize.py` | `optimize()` |
| `pareto.py` | non-dominated configurations |
| `storage.py` | `ExactIndex`, `TieredIndex`, `StorageBackend` / `NpzStorage`, bridge to the original index backends |
| `benchmark.py` | datasets from encode caches, the original table rebuilt from stages, result files |

The original pipeline (`OptiVisionRAG`, `bench`, the Qdrant backend, every
existing CLI command) is untouched and still works; the universal layer wraps
its pruners and codecs rather than replacing them.

## Quick start

```python
from optivision import MultiVectorCorpus, Pipeline, AdaptiveMerge, Int8Quantizer, optimize

docs = MultiVectorCorpus.from_arrays(list_of_doc_arrays)       # any model's [n_i, d] outputs
queries = MultiVectorCorpus.from_arrays(list_of_query_arrays)  # a few hundred real queries

# explicit pipeline
pipe = Pipeline([AdaptiveMerge(radius=0.7), Int8Quantizer("per_vector")])
compressed = pipe.fit(docs).compress(docs)
print(compressed.report())

# or let OptiVision choose, against a target measured on held-out queries
result = optimize(docs, queries=queries, qrels=labels_or_None, quality_target=0.97, metric="ndcg@5")
print(result.pipeline.label(), result.compression_ratio, result.quality_retention, result.memory_saving)
```

```bash
optivision inspect   data/cache/colpali_infovqa_test_subsampled.npz
optivision compress  docs.npz -p "adaptive_merge(radius=0.7) > int8(scale=per_vector)" -o docs.opt.npz
optivision calibrate docs.npz -q queries.npz --qrels qrels.json --target 0.97
optivision benchmark docs.npz -q queries.npz --qrels qrels.json --suite merge
optivision compare   docs.npz -q queries.npz -p int8 -p "adaptive_merge(radius=0.6) > binary"
```

Legacy encode caches (`optivision bench --cache ...`) work anywhere a corpus
file does; pass the cache's `queries.json` as `-q`.

## Supported models — exact coverage

| model | modality | dim | status | measured on |
|---|---|---|---|---|
| `vidore/colpali-v1.3-merged` | page images | 128 | **MEASURED** | ViDoRe InfoVQA and DocVQA (500 pages each), generated corpus (60 pages) |
| `vidore/colSmol-256M` | page images | 128 | **MEASURED** | generated corpus (60 pages) |
| `answerdotai/answerai-colbert-small-v1` | text | 96 | **MEASURED** | BEIR SciFact (5,183 abstracts, 300 queries) via sentence-transformers 6.1.0; float nDCG@10 74.56 vs 74.77 on the model card |
| `vidore/colqwen2-v1.0-merged` | page images | 128 | **MEASURED** | ViDoRe InfoVQA and DocVQA (500 pages each); float nDCG@5 within 0.8 points of the published scores (R11) |
| `vidore/colqwen2-v1.0` | page images | 128 | **MEASURED** | the adapter-only repo, merged on load; same vectors as the merged checkpoint (median cosine 0.99999, R11) |
| `vidore/colqwen2.5-v0.2` | page images | 128 | **MEASURED** | ViDoRe InfoVQA and DocVQA (500 pages each); DocVQA float nDCG@5 1.8 points below the published score (R11) |
| `nvidia/nemotron-colembed-vl-4b-v2` | page images | 2560 | **MEASURED** | ViDoRe InfoVQA and DocVQA (500 pages each, ~750 vectors/page); float nDCG@5 0.9 points below the published DocVQA score (R12); CC-BY-NC-4.0 |
| `nvidia/nemotron-colembed-vl-8b-v2` | page images | 4096 | untested — loader exists (pinned revision) | — |
| `vidore/colSmol-500M` | page images | 128 | untested — loader exists | — |
| `nomic-ai/colnomic-embed-multimodal-7b`, `colbert-ir/colbertv2.0`, `lightonai/GTE-ModernColBERT-v1` | | | unverified — adapter written, never run | — |
| anything else | any | any | vectors in via `from_arrays`; compression behaviour unmeasured | — |

`optivision.adapters.supported_models()` returns this table; a test pins the
*measured* set to exactly the models that have results in `reports/universal/`.

Any other multi-vector model works through `MultiVectorCorpus.from_arrays` —
the layer only needs vectors — but its compression behaviour is **unmeasured**
until someone runs the benchmark on it. `scripts/encode_vectors.py` produces
the vector files on a GPU machine.

## Methods

| stage | kind | needs | notes |
|---|---|---|---|
| `AdaptiveMerge(radius, center=None)` | token | vectors only | farthest-point cover to cosine `radius`, members pooled; count adapts per document. `center="mean"` measures similarity after removing the corpus mean -- needed for anisotropic encoders (R8) |
| `HierarchicalMerge(ratio, max_distance)` | token | SciPy | Ward clustering to a fraction of each document (Clavie et al. 2024 token pooling) or cut at a height (count adapts per document) |
| `RedundancyPruner(threshold)` | token | vectors only | original greedy leader clustering |
| `SpatialPruner` | token | page images + grid | original pixel-saliency pruner; does little on dense pages |
| `RandomPruner(ratio)` | token | -- | control |
| `DimensionProjector(dim, method)` | dimension | fit on documents | `pca` (default), `random`, `truncate`; queries are projected with the same map |
| `PCAProjector` / `RandomProjector` / `TruncateProjector` | dimension | PCA can also fit on documents + calibration queries | truncation is only meaningful for Matryoshka-trained models |
| `Float16Quantizer` | quantize | -- | 2x |
| `Int8Quantizer(scale, center)` | quantize | `fixed` (original), `per_vector` (+2 B/vector), `per_dimension` (fitted); optional fitted `center="mean"` | ~4x |
| `Int4Quantizer(center)` | quantize | per-vector scale; optional `center="mean"` | ~8x |
| `Lloyd2Quantizer` | quantize | fitted mean/scale | random rotation + 2-bit Lloyd-Max, 16x |
| `BinaryQuantizer(center)` | quantize | optional fitted `center="mean"` | sign bits, 32x |

## Methodology

- **Exact scoring.** Every number is exact MaxSim of every query against every
  document (`scoring.maxsim_matrix`); no ANN, so a change can only come from
  compression.
- **Retention** is `metric(compressed) / metric(float32 baseline)` on the same
  queries, with a paired bootstrap 95% interval over queries.
- **Two references.** *labels*: the dataset's qrels. *baseline@1*: the float
  index's own top result is the relevant document — no labels needed, and
  stricter, because it also counts a changed answer that happens to be right.
- **No tuning on the reported queries — for out-of-sample results.**
  Calibration chooses on one part of the queries and reports on the disjoint
  rest; the calibration study repeats that over 20 random splits and reports how
  often the held-out figure met the target. The fixed-configuration tables do
  not choose anything, and R5b is in-sample on purpose (see the table above).
- **Ties.** Rankings break exact score ties by document index. Placing a tied
  relevant document first or last instead changes no reported retention
  (checked on binary, merged-binary and merged-int4 codes on all three real
  corpora in the release audit): float queries scored against quantized
  documents give essentially no exact ties.
- **The compressed index and the baseline are scored identically**: the same
  queries (changed only by a fitted projection stage, applied to both sides),
  every document a candidate, the same exact MaxSim, descending rank.
- **Bytes** are stored bytes per document including per-vector scales and shared
  codec state, against a float32 baseline (the repository's convention). Divide
  by 2 for a bf16/fp16 baseline.
- **Latency** is the exact brute-force scan on this laptop (Intel i7-1165G7,
  4 cores, numpy), a relative number only.
- **Equivalence.** Identity pipelines reproduce the float scores exactly, and the
  original nine-variant table rebuilt from stages reproduces the committed
  reports on all four caches (max |Δ nDCG| = 4.4e-16).

## Results (MEASURED)

All figures: exact MaxSim over the whole corpus; **retention = nDCG@5 of the
configuration / nDCG@5 of the float32 index on the same queries**, with a 95%
paired bootstrap interval; labels = the dataset's qrels. Full per-row tables:
`reports/universal/TABLES.md`; raw JSON: `reports/universal/*/`.

| dataset | encoder | documents | queries | float nDCG@5 |
|---|---|---|---|---|
| ViDoRe InfoVQA (test, subsampled) | ColPali-v1.3 | 500 pages | 494 | 0.8458 |
| ViDoRe DocVQA (test, subsampled) | ColPali-v1.3 | 500 pages | 451 | 0.5841 |
| generated office pages (E3) | ColPali-v1.3 | 60 pages | 72 | 0.6954 |
| generated office pages (E1) | ColSmol-256M | 60 pages | 72 | 0.7823 |
| BEIR SciFact | answerai-colbert-small-v1 (text) | 5,183 abstracts | 300 | 0.7308 (nDCG@10 0.7456) |
| ViDoRe InfoVQA (test, subsampled) | ColQwen2-v1.0 | 500 pages | 494 | 0.9197 |
| ViDoRe DocVQA (test, subsampled) | ColQwen2-v1.0 | 500 pages | 451 | 0.6065 |
| ViDoRe InfoVQA (test, subsampled) | ColQwen2.5-v0.2 | 500 pages | 494 | 0.9169 |
| ViDoRe DocVQA (test, subsampled) | ColQwen2.5-v0.2 | 500 pages | 451 | 0.6180 |

The two generated corpora have 72 queries, so their intervals are about ±6
points; read them as sanity checks. The ViDoRe splits carry the conclusions.

### R1 · The refactor preserves retrieval exactly

*Fixed configurations.*

The original nine-variant table rebuilt from stages reproduces the committed
reports on all four caches: 36 rows, maximum |Δ nDCG@5 / @10| = 4.4e-16
(`reports/universal/equivalence/`). Identity pipelines reproduce float scores.

### R2 · Token reduction: merge, don't drop

*Fixed configurations, all queries.*

| method | ColPali · ViDoRe InfoVQA | ColPali · ViDoRe DocVQA | ColPali · generated | ColSmol · generated |
|---|---|---|---|---|
| spatial+redundancy (legacy) | 618 vec · 99.7% [99.1%, 100.3%] | 557 vec · 97.4% [95.6%, 99.1%] | 245 vec · 102.2% [94.6%, 110.4%] | 247 vec · 96.1% [90.1%, 101.5%] |
| redundancy t=0.92 | 633 vec · 100.0% [99.6%, 100.5%] | 650 vec · 97.5% [95.8%, 99.0%] | 606 vec · 99.3% [95.5%, 103.3%] | 327 vec · 97.3% [92.9%, 101.9%] |
| redundancy t=0.8 | 277 vec · 100.1% [99.1%, 101.1%] | 293 vec · 97.1% [94.9%, 99.4%] | 215 vec · 100.1% [92.8%, 107.7%] | 200 vec · 92.1% [84.7%, 99.9%] |
| adaptive radius=0.8 | 283 vec · 99.8% [99.0%, 100.8%] | 300 vec · 97.0% [95.1%, 98.6%] | 224 vec · 99.0% [92.9%, 105.7%] | 196 vec · 95.6% [88.8%, 102.7%] |
| adaptive radius=0.7 | 172 vec · 99.7% [98.6%, 100.8%] | 182 vec · 96.7% [94.3%, 98.9%] | 126 vec · 99.8% [93.0%, 107.6%] | 164 vec · 88.6% [81.3%, 95.7%] |
| adaptive radius=0.6 | 116 vec · 99.4% [98.1%, 100.6%] | 122 vec · 96.2% [93.6%, 99.0%] | 82 vec · 101.7% [93.6%, 111.0%] | 148 vec · 86.5% [78.7%, 94.2%] |
| adaptive radius=0.5 | 82 vec · 97.5% [95.7%, 99.4%] | 88 vec · 94.1% [91.1%, 97.3%] | 60 vec · 101.5% [94.5%, 109.8%] | 136 vec · 84.3% [77.3%, 91.7%] |
| hierarchical ratio=0.5 | 519 vec · 99.9% [99.2%, 100.4%] | 519 vec · 99.9% [98.6%, 101.2%] | 519 vec · 99.7% [94.8%, 105.0%] | 491 vec · 98.1% [94.5%, 101.5%] |
| hierarchical ratio=0.25 | 263 vec · 99.9% [99.0%, 100.8%] | 263 vec · 98.9% [96.9%, 100.8%] | 263 vec · 99.4% [94.5%, 104.1%] | 299 vec · 99.6% [94.9%, 104.9%] |
| hierarchical ratio=0.1 | 110 vec · 99.0% [97.7%, 100.2%] | 110 vec · 95.4% [92.8%, 98.2%] | 110 vec · 97.6% [91.6%, 104.0%] | 184 vec · 91.6% [85.0%, 98.5%] |
| hierarchical ratio=0.05 | 59 vec · 97.3% [95.7%, 98.9%] | 59 vec · 89.7% [85.9%, 93.2%] | 59 vec · 99.2% [93.1%, 106.2%] | 146 vec · 87.2% [79.9%, 94.0%] |
| adaptive ratio=0.5 | 519 vec · 100.2% [99.7%, 100.8%] | 519 vec · 98.6% [97.0%, 100.1%] | 519 vec · 100.1% [96.3%, 104.0%] | 491 vec · 98.4% [95.0%, 101.7%] |
| adaptive ratio=0.25 | 263 vec · 99.9% [99.1%, 100.9%] | 263 vec · 97.6% [95.6%, 99.7%] | 263 vec · 100.0% [95.4%, 104.8%] | 299 vec · 97.8% [92.6%, 103.5%] |
| adaptive ratio=0.1 | 110 vec · 98.5% [97.0%, 99.9%] | 110 vec · 95.4% [92.8%, 98.2%] | 110 vec · 102.3% [95.2%, 110.6%] | 184 vec · 90.9% [84.8%, 97.5%] |
| adaptive ratio=0.05 | 59 vec · 94.6% [92.5%, 96.4%] | 59 vec · 93.5% [90.0%, 97.2%] | 59 vec · 102.2% [94.7%, 110.3%] | 146 vec · 86.8% [78.7%, 94.9%] |
| random ratio=0.5 | 519 vec · 98.6% [97.4%, 99.8%] | 519 vec · 97.9% [95.9%, 100.1%] | 519 vec · 97.1% [92.4%, 102.2%] | 491 vec · 93.6% [87.0%, 101.1%] |
| random ratio=0.25 | 263 vec · 97.8% [96.2%, 99.3%] | 263 vec · 94.1% [90.8%, 97.2%] | 263 vec · 98.9% [91.7%, 106.7%] | 299 vec · 88.1% [80.5%, 96.6%] |
| random ratio=0.1 | 110 vec · 94.5% [92.2%, 96.5%] | 110 vec · 84.7% [80.5%, 89.1%] | 110 vec · 91.9% [83.4%, 101.6%] | 184 vec · 82.7% [75.3%, 91.5%] |

- **Merging beats dropping.** At about 9x fewer vectors, random dropping keeps
  84.7% (DocVQA) and 94.5% (InfoVQA); merging keeps 95.4–99.4%.
- **No single merger wins everywhere.** The coverage-radius `AdaptiveMerge`
  is best on InfoVQA at ~9x (99.4%); Ward clustering is best on DocVQA at
  2–4x (99.9% / 98.9%). This is why the layer selects rather than fixes a method.
- **Real scanned pages (DocVQA) are the hard case**: about 4x at ≥99% and 9x at
  ~96% are what training-free merging delivers there.
- The original pixel-space pruner reaches only 1.7–1.8x on real pages.

**Merge variants** — Ward cut at a height (count adapts per page) and k-means
refinement of the adaptive cover:

| method | ColPali · ViDoRe InfoVQA | ColPali · ViDoRe DocVQA | ColPali · generated | ColSmol · generated |
|---|---|---|---|---|
| ward distance=0.6 (per-doc count) | 418 vec · 99.8% [99.1%, 100.4%] | 425 vec · 98.9% [97.5%, 100.3%] | 372 vec · 101.2% [97.3%, 105.6%] | 276 vec · 97.4% [92.7%, 102.3%] |
| ward distance=0.8 (per-doc count) | 270 vec · 100.1% [99.3%, 101.1%] | 275 vec · 99.1% [97.3%, 100.9%] | 230 vec · 98.3% [93.5%, 103.3%] | 220 vec · 94.1% [88.3%, 99.8%] |
| ward distance=1.0 (per-doc count) | 192 vec · 99.5% [98.5%, 100.5%] | 196 vec · 97.5% [95.2%, 99.7%] | 161 vec · 98.9% [92.1%, 105.6%] | 190 vec · 92.1% [84.6%, 99.5%] |
| ward distance=1.3 (per-doc count) | 128 vec · 98.9% [97.7%, 99.9%] | 134 vec · 97.0% [94.4%, 99.6%] | 110 vec · 98.7% [92.6%, 104.7%] | 167 vec · 86.7% [80.1%, 93.3%] |
| ward distance=1.6 (per-doc count) | 93 vec · 98.6% [97.3%, 99.8%] | 100 vec · 95.8% [93.3%, 98.4%] | 84 vec · 101.8% [93.9%, 110.7%] | 152 vec · 86.3% [78.9%, 93.8%] |
| adaptive radius=0.8 refine=3 | 283 vec · 99.7% [98.9%, 100.7%] | 300 vec · 97.2% [95.4%, 98.8%] | 224 vec · 98.6% [92.9%, 104.7%] | 196 vec · 94.4% [87.5%, 101.8%] |
| adaptive radius=0.7 refine=3 | 172 vec · 99.5% [98.4%, 100.5%] | 182 vec · 96.3% [93.6%, 98.7%] | 126 vec · 101.9% [94.7%, 109.8%] | 164 vec · 87.9% [80.3%, 95.7%] |
| adaptive radius=0.6 refine=3 | 116 vec · 100.1% [99.0%, 101.3%] | 122 vec · 96.2% [93.7%, 98.8%] | 82 vec · 103.5% [96.5%, 111.2%] | 148 vec · 86.4% [78.2%, 94.8%] |

Ward with a per-page count gives the best DocVQA points measured (3.7x at 99.1%,
7.7x at 97.0%); refinement does not help the adaptive cover.

### R3 · Codecs

*Fixed configurations, all queries.*

| codec | ColPali · ViDoRe InfoVQA | ColPali · ViDoRe DocVQA | ColPali · generated | ColSmol · generated |
|---|---|---|---|---|
| float16 | 2x · 100.0% [100.0%, 100.0%] | 2x · 100.0% [99.8%, 100.0%] | 2x · 99.7% [99.2%, 100.0%] | 2x · 100.0% [100.0%, 100.0%] |
| int8 fixed (legacy) | 4x · 100.1% [100.0%, 100.3%] | 4x · 99.9% [99.1%, 100.7%] | 4x · 100.0% [97.9%, 102.3%] | 4x · 100.7% [99.8%, 102.3%] |
| int8 per-vector | 4x · 100.0% [100.0%, 100.0%] | 4x · 100.0% [99.8%, 100.4%] | 4x · 98.9% [97.1%, 100.1%] | 4x · 99.3% [98.0%, 100.0%] |
| int8 per-dimension | 4x · 100.0% [99.9%, 100.0%] | 4x · 100.0% [99.6%, 100.4%] | 4x · 99.7% [97.7%, 101.8%] | 4x · 99.3% [98.0%, 100.0%] |
| int4 per-vector | 8x · 99.9% [99.4%, 100.4%] | 8x · 97.5% [95.4%, 99.4%] | 8x · 101.9% [98.2%, 106.0%] | 8x · 95.9% [92.2%, 99.8%] |
| lloyd2 (2-bit) | 16x · 99.4% [98.5%, 100.4%] | 16x · 95.2% [92.5%, 97.7%] | 16x · 98.5% [93.1%, 105.0%] | 16x · 95.8% [89.8%, 102.2%] |
| lloyd2 (2-bit, seed 1) | 16x · 99.6% [98.4%, 100.7%] | 16x · 95.5% [92.6%, 98.1%] | 16x · 101.5% [96.9%, 106.3%] | 16x · 94.6% [88.4%, 101.1%] |
| lloyd2 (2-bit, seed 2) | 16x · 99.3% [98.3%, 100.3%] | 16x · 95.2% [92.5%, 97.7%] | 16x · 101.1% [95.1%, 107.4%] | 16x · 93.8% [88.8%, 98.7%] |
| binary | 32x · 97.4% [96.0%, 98.7%] | 32x · 96.3% [93.7%, 98.8%] | 32x · 98.4% [91.9%, 105.4%] | 32x · 87.9% [80.9%, 95.7%] |
| binary centred | 32x · 98.5% [97.1%, 99.7%] | 32x · 95.2% [92.2%, 98.0%] | 32x · 99.1% [93.4%, 105.1%] | 32x · 91.1% [83.8%, 99.5%] |

- int8 is lossless everywhere; float16 trivially so.
- int4 (8x) is nearly free on InfoVQA and costs 2.5 points on DocVQA.
- The rotated 2-bit codec lands at 95.2–95.5% on DocVQA across three rotation
  seeds (so the 97.0% in the older codec ladder is not a seed artifact; the two
  scripts differ in fitting and tie handling).
- Centring the binary codec helps on InfoVQA (+1.1) and hurts on DocVQA (−1.1).

### R3b · Centring the codecs

*Fixed configurations, all queries.*

Removing a fitted corpus mean before quantizing (and adding it back on decode):

| configuration | x float32 | ColPali · InfoVQA (nDCG@5) | ColPali · DocVQA (nDCG@5) | ColBERT-small · SciFact (nDCG@10) |
|---|---|---|---|---|
| none > int8/vec | 3.9x | 100.0% [100.0%, 100.0%] | 100.0% [99.8%, 100.4%] | 100.1% [99.6%, 100.6%] |
| none > int8/vec centred | 3.9x | 100.0% [99.9%, 100.0%] | 99.8% [99.4%, 100.0%] | 99.6% [99.0%, 100.0%] |
| none > int4 | 7.8x | 99.9% [99.4%, 100.4%] | 97.5% [95.4%, 99.4%] | 89.7% [86.0%, 93.2%] |
| none > int4 centred | 7.8x | 100.1% [99.6%, 100.6%] | 99.0% [97.1%, 100.9%] | 98.6% [97.2%, 100.0%] |
| none > binary | 32.0x | 97.4% [96.0%, 98.7%] | 96.3% [93.7%, 98.8%] | 94.1% [91.4%, 96.8%] |
| none > binary centred | 32.0x | 98.5% [97.1%, 99.7%] | 95.2% [92.2%, 98.0%] | 1.1% [0.3%, 2.2%] |
| ward ratio=0.33 > int4 | 23.2x | 99.1% [98.1%, 99.9%] | 98.4% [95.9%, 100.9%] | 83.8% [79.2%, 88.2%] |
| ward ratio=0.33 > int4 centred | 23.2x | 99.8% [98.9%, 100.6%] | 98.1% [96.0%, 100.2%] | 98.7% [96.8%, 100.7%] |
| ward ratio=0.33 > binary | 95.6x | 97.6% [96.3%, 99.0%] | 95.4% [92.4%, 98.2%] | 92.8% [89.9%, 95.7%] |
| ward ratio=0.33 > binary centred | 95.6x | 97.2% [95.8%, 98.7%] | 93.4% [90.1%, 96.6%] | 0.9% [0.1%, 2.0%] |

- **Centred int4 is never worse beyond noise and rescues the text model**
  (89.7% -> 98.6% alone, 83.8% -> 98.7% after Ward merging at 23x). The default
  search space uses it.
- **Centred binary is a hazard.** It helps InfoVQA a little and collapses the
  text model to 1.1%: the queries' shared direction multiplies the sign code's
  error on each small residual (confirmed by the geometry diagnostic under R8).
  Plain binary stays in the default space and the option carries a measured
  warning. All of this is one text model on one corpus.

### R4 · Dimension reduction (128-d vectors)

*Fixed configurations, all queries (the joint-PCA comparison is on a held-out half).*

| projection | ColPali · ViDoRe InfoVQA | ColPali · ViDoRe DocVQA | ColPali · generated | ColSmol · generated |
|---|---|---|---|---|
| pca-docs 96 | 99.8% [99.2%, 100.3%] | 99.4% [98.2%, 100.6%] | 99.8% [96.6%, 103.2%] | 100.5% [98.2%, 103.0%] |
| pca-docs 64 | 97.2% [95.9%, 98.5%] | 97.4% [94.9%, 99.7%] | 100.9% [96.2%, 106.2%] | 100.2% [96.1%, 104.2%] |
| pca-docs 48 | 95.5% [93.6%, 97.2%] | 92.8% [89.7%, 95.6%] | 99.9% [94.6%, 105.8%] | 96.0% [89.6%, 101.6%] |
| pca-docs 32 | 87.1% [84.3%, 89.8%] | 80.3% [76.2%, 84.4%] | 98.4% [91.2%, 105.6%] | 90.8% [83.3%, 98.5%] |
| random-proj 96 | 98.3% [97.2%, 99.3%] | 98.1% [95.9%, 100.4%] | 99.2% [94.5%, 103.9%] | 96.2% [90.4%, 102.1%] |
| random-proj 64 | 95.2% [93.1%, 97.4%] | 92.2% [89.0%, 95.2%] | 98.5% [93.1%, 104.1%] | 89.8% [83.2%, 96.8%] |
| random-proj 32 | 80.8% [77.7%, 84.0%] | 64.7% [59.4%, 69.5%] | 103.6% [96.0%, 112.1%] | 81.1% [72.9%, 89.2%] |
| truncate 96 | 98.8% [97.5%, 100.0%] | 97.7% [95.4%, 99.9%] | 98.7% [94.4%, 102.9%] | 98.5% [94.3%, 103.0%] |
| truncate 64 | 96.4% [94.3%, 98.1%] | 89.8% [86.2%, 93.3%] | 100.7% [95.1%, 106.8%] | 92.9% [87.2%, 99.2%] |
| truncate 32 | 82.4% [78.8%, 85.6%] | 61.6% [56.4%, 67.0%] | 87.0% [77.7%, 95.8%] | 77.0% [68.1%, 86.6%] |

PCA on the documents dominates random projection and truncation; 96 dimensions
are nearly free, 64 cost ~3 points, 32 are unusable. Adding calibration queries
to the PCA basis did not help (DocVQA held-out: 97.9% joint vs 98.5% documents
only at 64 d). ColPali is not Matryoshka-trained, so truncation is a control.

### R5 · Combinations

*Fixed configurations, all queries.*

| pipeline | ColPali · ViDoRe InfoVQA | ColPali · ViDoRe DocVQA | ColPali · generated | ColSmol · generated |
|---|---|---|---|---|
| merge 0.6 > int8 per-vector | 116 vec · 35x · 99.4% [98.1%, 100.6%] | 122 vec · 33x · 95.9% [93.2%, 98.6%] | 82 vec · 49x · 101.6% [93.6%, 110.9%] | 148 vec · 23x · 86.6% [78.7%, 94.4%] |
| merge 0.6 > int4 | 116 vec · 69x · 99.2% [97.9%, 100.5%] | 122 vec · 65x · 95.0% [92.1%, 98.0%] | 82 vec · 97x · 100.4% [92.5%, 109.5%] | 148 vec · 46x · 85.5% [77.7%, 93.3%] |
| merge 0.6 > lloyd2 | 116 vec · 142x · 98.9% [97.3%, 100.4%] | 122 vec · 135x · 94.5% [91.5%, 97.7%] | 82 vec · 200x · 101.9% [93.5%, 111.4%] | 148 vec · 95x · 85.6% [77.5%, 94.0%] |
| merge 0.6 > binary | 116 vec · 285x · 96.6% [95.1%, 98.2%] | 122 vec · 270x · 89.3% [86.0%, 92.6%] | 82 vec · 401x · 100.5% [92.6%, 109.2%] | 148 vec · 190x · 81.6% [74.0%, 90.7%] |
| merge 0.6 > binary centred | 116 vec · 285x · 97.0% [95.4%, 98.4%] | 122 vec · 270x · 90.5% [87.0%, 93.9%] | 82 vec · 398x · 101.1% [91.5%, 111.2%] | 148 vec · 189x · 84.4% [76.6%, 92.5%] |
| merge 0.6 > pca-docs 64 > int8 per-vector | 116 vec · 69x · 96.5% [94.8%, 98.2%] | 122 vec · 65x · 93.8% [90.8%, 96.7%] | 82 vec · 97x · 98.8% [91.4%, 107.4%] | 148 vec · 46x · 84.7% [77.1%, 92.1%] |
| merge 0.6 > pca-docs 64 > binary | 116 vec · 570x · 91.4% [89.0%, 93.8%] | 122 vec · 540x · 85.7% [81.3%, 89.8%] | 82 vec · 802x · 96.3% [88.3%, 106.2%] | 148 vec · 380x · 83.2% [74.8%, 91.9%] |

Merging and a 4–8-bit codec compose almost additively on InfoVQA (35x at 99.4%,
69x at 99.2%); on DocVQA the same pipelines hold 95–96%. One-bit codes on top of
merging drop to 89–97%, and PCA-64 on top of merging costs another 2–3 points.

### R5b · The token-stage × codec frontier

*In-sample selection: the best of 60, chosen and reported on the same queries.*

Every token stage (none; Ward cut at a height or to a fraction; adaptive merge)
crossed with every codec (float32, float16, per-vector int8, int4, binary),
each merge computed once. For each retention target: the smallest configuration
whose retention over **all** queries meets it (left), and the smallest whose
bootstrap **lower bound** does (right).

**E2-colpali-docvqa** — 500 docs, 451 queries, retention of ndcg@5

| target | smallest config (point estimate) | x float32 | retention [95% CI] | smallest config (CI lower bound) | x float32 | retention [95% CI] |
|---|---|---|---|---|---|---|
| 0.99 | ward ratio=0.33 > int8/vec | 11.8x | 99.2% [97.2%, 101.1%] | none > int8/vec | 3.9x | 100.0% [99.8%, 100.4%] |
| 0.97 | ward ratio=0.25 > int4 | 30.4x | 97.6% [94.9%, 100.3%] | ward d=0.8 > int8/vec | 14.7x | 98.9% [97.0%, 100.7%] |
| 0.95 | ward ratio=0.25 > binary | 125.4x | 95.4% [92.4%, 98.1%] | ward ratio=0.33 > int4 | 23.2x | 98.4% [95.9%, 100.9%] |
| 0.90 | ward d=1.6 > binary | 330.3x | 90.1% [86.6%, 93.6%] | ward ratio=0.25 > binary | 125.4x | 95.4% [92.4%, 98.1%] |

Pareto frontier (bytes/doc vs retention): ward d=2.0 > binary (454x, 87.8%), ward d=1.6 > binary (330x, 90.1%), ward d=1.3 > binary (246x, 92.3%), ward d=1.0 > binary (169x, 92.4%), ward ratio=0.25 > binary (125x, 95.4%), ward d=1.3 > int4 (60x, 95.7%), adaptive r=0.7 > int4 (44x, 96.0%), ward d=1.0 > int4 (41x, 96.0%), none > binary (32x, 96.3%), ward ratio=0.25 > int4 (30x, 97.6%), ward ratio=0.33 > int4 (23x, 98.4%), ward ratio=0.25 > int8/vec (15x, 98.6%), ward d=0.8 > int8/vec (15x, 98.9%), ward ratio=0.33 > int8/vec (12x, 99.2%), ward ratio=0.5 > int8/vec (8x, 99.7%), ward ratio=0.5 > float16 (4x, 99.8%), none > int8/vec (4x, 100.0%)

**E2-colpali-infovqa** — 500 docs, 494 queries, retention of ndcg@5

| target | smallest config (point estimate) | x float32 | retention [95% CI] | smallest config (CI lower bound) | x float32 | retention [95% CI] |
|---|---|---|---|---|---|---|
| 0.99 | adaptive r=0.6 > int4 | 69.0x | 99.2% [97.9%, 100.5%] | ward ratio=0.25 > int4 | 30.4x | 100.1% [99.1%, 101.1%] |
| 0.97 | adaptive r=0.7 > binary | 191.8x | 97.5% [96.0%, 99.0%] | adaptive r=0.6 > int4 | 69.0x | 99.2% [97.9%, 100.5%] |
| 0.95 | ward d=1.6 > binary | 354.6x | 95.8% [93.9%, 97.5%] | adaptive r=0.6 > binary | 284.8x | 96.6% [95.1%, 98.2%] |
| 0.90 | ward d=2.0 > binary | 506.4x | 94.5% [92.3%, 96.6%] | ward d=2.0 > binary | 506.4x | 94.5% [92.3%, 96.6%] |

Pareto frontier (bytes/doc vs retention): ward d=2.0 > binary (506x, 94.5%), ward d=1.6 > binary (355x, 95.8%), adaptive r=0.6 > binary (285x, 96.6%), adaptive r=0.7 > binary (192x, 97.5%), adaptive r=0.5 > int4 (98x, 97.7%), ward d=1.6 > int4 (86x, 98.3%), adaptive r=0.6 > int4 (69x, 99.2%), ward d=1.3 > int4 (63x, 99.3%), adaptive r=0.7 > int4 (46x, 100.0%), ward ratio=0.25 > int4 (30x, 100.1%), ward d=0.8 > float16 (8x, 100.1%)

**Both columns are in-sample selections**: the best of 60 configurations,
chosen and reported on the same queries. The right column only requires the
lower end of each configuration's 95% interval (a one-sided 97.5% bound, not
corrected for choosing among 60) to clear the target. That makes it less
optimistic than the left column, not unbiased. The table shows what the grid
contains, not what calibration will choose: the out-of-sample default (R7b)
chose a median of 3.9x at the 0.97 target on DocVQA where this table shows 14.7x.


### R6 · Two tiers: tiny codes in RAM, int8 on disk

*Fixed configurations, all queries.*

The hot tier (binary codes, optionally of merged vectors) scores every page and
keeps a shortlist; the cold tier rescores only the shortlist. RAM figures are
DocVQA's (InfoVQA is within 10%); float32 is 527.9 KB/page.

| hot tier (RAM) | RAM KB/page | vs float32 | cold tier (disk) | disk KB/page | shortlist | DocVQA | InfoVQA | rescore ms/query |
|---|---|---|---|---|---|---|---|---|
| binary | 16.50 | 32x | none | - | - | 96.3% [93.7%, 98.8%] | 97.4% [96.0%, 98.7%] | - |
| binary | 16.50 | 32x | int8/vec (all vectors) | 134.0 | 10 | 99.6% [98.1%, 100.8%] | 99.9% [99.7%, 100.1%] | 8 / 6 |
| binary | 16.50 | 32x | int8/vec (all vectors) | 134.0 | 50 | 100.1% [99.5%, 100.9%] | 100.0% [100.0%, 100.0%] | 42 / 37 |
| adaptive merge 0.8 > binary | 4.80 | 110x | none | - | - | 91.7% [88.5%, 94.8%] | 97.1% [95.7%, 98.4%] | - |
| adaptive merge 0.8 > binary | 4.80 | 110x | int8/vec (all vectors) | 134.0 | 50 | 101.0% [100.1%, 102.0%] | 100.0% [100.0%, 100.1%] | 40 / 33 |
| adaptive merge 0.8 > binary | 4.80 | 110x | int8/vec (same merge 0.8) | 39.0 | 50 | 97.3% [95.5%, 98.9%] | 99.9% [99.1%, 100.9%] | 9 / 12 |
| adaptive merge 0.6 > binary | 1.96 | 270x | none | - | - | 89.3% [86.0%, 92.6%] | 96.6% [95.1%, 98.2%] | - |
| adaptive merge 0.6 > binary | 1.96 | 270x | int8/vec (all vectors) | 134.0 | 10 | 95.6% [92.8%, 98.2%] | 99.8% [99.1%, 100.5%] | 7 / 8 |
| adaptive merge 0.6 > binary | 1.96 | 270x | int8/vec (all vectors) | 134.0 | 50 | 99.1% [97.5%, 100.5%] | 100.0% [100.0%, 100.1%] | 25 / 33 |
| adaptive merge 0.6 > binary | 1.96 | 270x | int8/vec (same merge 0.6) | 15.9 | 50 | 95.9% [93.3%, 98.6%] | 99.4% [98.1%, 100.6%] | 5 / 6 |
| adaptive merge 0.5 > binary | 1.40 | 376x | none | - | - | 87.8% [84.1%, 91.3%] | 93.7% [91.6%, 95.7%] | - |
| adaptive merge 0.5 > binary | 1.40 | 376x | int8/vec (all vectors) | 134.0 | 50 | 99.5% [97.8%, 101.0%] | 99.8% [99.4%, 100.1%] | 33 / 46 |

- **A 50-page shortlist rescored from int8 restores 99.1–101.0% on both splits**
  while RAM holds 1.4–4.8 KB per page — 110–376x less than float32. This was a
  projection before; it is now measured, including merged hot tiers.
- **The cold tier has to keep the unmerged vectors.** Rescoring with the same
  merged vectors recovers only 95.9–97.3% on DocVQA: merging loses information
  a rerank cannot put back.
- **Total bytes are not the win.** RAM + disk is ~136 KB/page (3.9x); the design
  moves bytes from scarce RAM to cheap disk. The one-tier frontier (R5b) is the
  answer when total storage is what matters.
- Rescoring cost here is an unoptimised Python loop (25–46 ms/query for 50
  pages); it is not a latency claim.

### R7 · Can calibration be trusted on unseen queries?

*Out-of-sample: 20 random calibration / held-out splits.*

For every cell below, the query set was split in half 20 times at random; the
calibrator chose on one half, and the chosen configuration was scored on the
other. Search space: adaptive-merge radii × {float16, int8, binary}.

Share of splits whose chosen configuration met the target on the held-out half, and the median compression chosen.

| reference · rule · target | ColPali · ViDoRe InfoVQA | ColPali · ViDoRe DocVQA | ColPali · generated | ColSmol · generated |
|---|---|---|---|---|
| labels · point · 0.99 | 80.0% · x32.6 | 80.0% · x4.0 | 60.0% · x551.2 | 35.0% · x8.5 |
| labels · point · 0.97 | 75.0% · x191.8 | 30.0% · x27.4 | 80.0% · x551.2 | 55.0% · x12.3 |
| labels · point · 0.95 | 100.0% · x284.8 | 45.0% · x41.1 | 90.0% · x551.2 | 45.0% · x17.9 |
| labels · lower_ci · 0.99 | 95.0% · x10.8 | 100.0% · x2.0 | 66.7% · x5.2 | 95.0% · x4.0 |
| labels · lower_ci · 0.97 | 85.0% · x35.6 | 90.0% · x4.0 | 70.0% · x25.1 | 80.0% · x4.0 |
| labels · lower_ci · 0.95 | 100.0% · x171.9 | 80.0% · x10.1 | 70.0% · x151.1 | 100.0% · x4.0 |
| baseline@1 · point · 0.99 | 60.0% (labels 100.0%) · x5.1 | 100.0% (labels 100.0%) · x2.0 | 90.0% (labels 100.0%) · x2.0 | 85.0% (labels 100.0%) · x2.0 |
| baseline@1 · point · 0.97 | 50.0% (labels 100.0%) · x19.0 | 80.0% (labels 100.0%) · x4.0 | 80.0% (labels 100.0%) · x5.2 | 100.0% (labels 100.0%) · x4.0 |
| baseline@1 · point · 0.95 | 50.0% (labels 100.0%) · x86.0 | 100.0% (labels 100.0%) · x4.0 | 100.0% (labels 100.0%) · x5.2 | 100.0% (labels 100.0%) · x4.0 |
| baseline@1 · lower_ci · 0.99 | 95.0% (labels 100.0%) · x5.1 | 100.0% (labels 100.0%) · x2.0 | 90.0% (labels 100.0%) · x2.0 | 85.0% (labels 100.0%) · x2.0 |
| baseline@1 · lower_ci · 0.97 | 100.0% (labels 100.0%) · x10.8 | 90.0% (labels 100.0%) · x2.0 | 90.0% (labels 100.0%) · x2.0 | 100.0% (labels 100.0%) · x2.0 |
| baseline@1 · lower_ci · 0.95 | 85.0% (labels 100.0%) · x21.5 | 100.0% (labels 100.0%) · x4.0 | 100.0% (labels 100.0%) · x2.0 | 100.0% (labels 100.0%) · x4.0 |

| dataset | target | chosen on fragments | fragment retention | real-query retention |
|---|---|---|---|---|
| ColPali · ViDoRe InfoVQA | 0.99 | adaptive_merge(0.5) > binary | 100.0% | 93.7% |
| ColPali · ViDoRe InfoVQA | 0.97 | adaptive_merge(0.5) > binary | 100.0% | 93.7% |
| ColPali · ViDoRe InfoVQA | 0.95 | adaptive_merge(0.5) > binary | 100.0% | 93.7% |
| ColPali · ViDoRe DocVQA | 0.99 | adaptive_merge(0.5) > binary | 100.0% | 87.8% |
| ColPali · ViDoRe DocVQA | 0.97 | adaptive_merge(0.5) > binary | 100.0% | 87.8% |
| ColPali · ViDoRe DocVQA | 0.95 | adaptive_merge(0.5) > binary | 100.0% | 87.8% |
| ColPali · generated | 0.99 | adaptive_merge(0.5) > binary | 100.0% | 100.1% |
| ColPali · generated | 0.97 | adaptive_merge(0.5) > binary | 100.0% | 100.1% |
| ColPali · generated | 0.95 | adaptive_merge(0.5) > binary | 100.0% | 100.1% |
| ColSmol · generated | 0.99 | adaptive_merge(0.6) > binary | 100.0% | 81.6% |
| ColSmol · generated | 0.97 | adaptive_merge(0.5) > binary | 97.1% | 83.2% |
| ColSmol · generated | 0.95 | adaptive_merge(0.5) > binary | 97.1% | 83.2% |

- **Choosing by the point estimate is a winner's-curse selection.** It picks the
  most aggressive configuration that clears the target on the calibration half,
  and on the held-out half that configuration misses as often as not: 30–80%
  success on DocVQA with labels, 50–60% on InfoVQA without them.
- **The bootstrap lower bound fixes most of it** — 80–100% on both ViDoRe splits,
  with labels or without — and is now the default.
- **With 36 calibration queries nothing is reliable** (the generated corpora):
  as low as 67% even with the lower bound. `calibrate()` warns below 100.
- **The label-free reference is conservative in the right direction**: the
  configurations it chose met the *label-based* target in 100% of splits on
  every dataset, at the price of less compression (2–22x median instead of
  2–172x).
- **Document fragments are not queries.** Choosing on pseudo-queries picked
  `adaptive_merge(0.5) > binary` and reported ~100%; real queries got 81.6–93.7%
  on three of four corpora. `optimize()` therefore requires real queries.

### R7b · The shipped default, out of sample

*Out-of-sample: 20 random calibration / held-out splits per cell
(`s7b_selection_rules.py`; `reports/universal/selection_rules/`).*

The default search space has 40 candidates: Ward and centred adaptive merging,
each combined with per-vector int8, centred int4 or binary codes, plus float16.
Many of them sit just above any target, so how the winner is chosen matters. The
row `calibrate() default` replays `calibrate()` exactly: the one-sided 0.999
bound, each family stopped at its first infeasible step, and n_boot = 1000. The
other rows judge every candidate with n_boot = 500.

`calibrate()` with its defaults, labels as the reference, 20 random half/half splits. *met* = share of splits whose choice reached the target on the held-out half (95% Wilson interval over the 20 splits); *x* = median compression chosen; *held-out* = mean and worst held-out retention of the choice.

| dataset | queries | target | met [95% interval] | median x | held-out mean · worst |
|---|---|---|---|---|---|
| DocVQA | 451 | 0.99 | 20/20 [83.9%, 100.0%] | 3.9x | 100.1% · 99.7% |
| DocVQA | 451 | 0.97 | 19/20 [76.4%, 99.1%] | 3.9x | 99.7% · 96.5% |
| DocVQA | 451 | 0.95 | 20/20 [83.9%, 100.0%] | 11.8x | 98.6% · 95.2% |
| InfoVQA | 494 | 0.99 | 20/20 [83.9%, 100.0%] | 3.9x | 99.9% · 99.3% |
| InfoVQA | 494 | 0.97 | 20/20 [83.9%, 100.0%] | 43.4x | 99.6% · 97.8% |
| InfoVQA | 494 | 0.95 | 20/20 [83.9%, 100.0%] | 72.7x | 98.9% · 96.2% |
| SciFact | 300 | 0.99 | 14/14 [78.5%, 100.0%] | 3.9x | 99.9% · 99.6% |
| SciFact | 300 | 0.97 | 20/20 [83.9%, 100.0%] | 8.0x | 99.4% · 97.2% |
| SciFact | 300 | 0.95 | 20/20 [83.9%, 100.0%] | 12.4x | 98.7% · 96.2% |

All rules (labels): met · median x.

| rule | DocVQA 0.99 | DocVQA 0.97 | DocVQA 0.95 | InfoVQA 0.99 | InfoVQA 0.97 | InfoVQA 0.95 | SciFact 0.99 | SciFact 0.97 | SciFact 0.95 |
|---|---|---|---|---|---|---|---|---|---|
| calibrate() default | 100% · x3.9 | 95% · x3.9 | 100% · x11.8 | 100% · x3.9 | 100% · x43.4 | 100% · x72.7 | 100% · x3.9 | 100% · x8.0 | 100% · x12.4 |
| point | 25% · x23.2 | 20% · x66.9 | 50% · x125.4 | 70% · x57.0 | 55% · x178.9 | 80% · x299.9 | 40% · x21.0 | 85% · x23.1 | 50% · x29.0 |
| lower 0.975 | 90% · x3.9 | 75% · x7.8 | 70% · x30.4 | 95% · x11.6 | 85% · x64.3 | 100% · x178.9 | 85% · x6.6 | 90% · x21.0 | 95% · x23.1 |
| lower 0.99 | 90% · x3.9 | 75% · x7.8 | 75% · x24.1 | 90% · x7.8 | 95% · x49.7 | 100% · x117.5 | 89% · x6.6 | 95% · x11.8 | 100% · x23.1 |
| lower 0.999 | 100% · x3.9 | 95% · x3.9 | 95% · x14.5 | 100% · x3.9 | 100% · x43.4 | 100% · x72.7 | 100% · x3.9 | 100% · x11.8 | 100% · x21.0 |
| bonferroni 0.975 | 100% · x3.9 | 95% · x3.9 | 100% · x11.8 | 100% · x3.9 | 100% · x43.4 | 100% · x72.7 | 100% · x3.9 | 100% · x9.4 | 100% · x21.0 |
| lower 0.975 + margin 0.01 | 100% · x2.0 | 80% · x3.9 | 80% · x23.2 | 100% · x3.9 | 100% · x43.4 | 95% · x81.7 | 100% · x2.0 | 100% · x11.8 | 100% · x22.1 |

Read the "met" column with its interval: 20 splits of one query pool cannot pin
a rate near 95% (19/20 is consistent with anything from 76% to 99%). Each "met"
also compares a held-out *point estimate* (150–250 queries, noisy by 1–2 points)
with the target.

- **The default met the target in 173 of 174 split × target cells that chose a
  configuration** across the three corpora. The worst held-out retention of any
  choice was 96.5% against a 0.97 target (DocVQA). At SciFact 0.99, no
  configuration qualified in 6 of 20 splits; `optimize()` then returns the float
  corpus and says the target was not met.
- **Compression follows the data.** At a 0.97 target the default chose a median
  of 3.9x on DocVQA, 8.0x on SciFact and 43x on InfoVQA.
- **Choosing by the point estimate is unreliable**: it met the target in only
  20–85% of splits, while choosing 2.3–17x more compression than the default.
  Looser bounds (0.975, 0.99) sit in between.
- **An earlier version of this table was measured on a different search space**
  (uncentred int4) and was never re-run after the default changed. The release
  audit found this; the table above is the shipped default.

`confidence`, `multiplicity="bonferroni"` and `margin` stay available in the API
and on the CLI. Why the 0.999 level is not the confidence of the *choice* is
explained in R10.

### R8 · A text late-interaction model

*Fixed configurations, all queries.*

`answerdotai/answerai-colbert-small-v1` (33M parameters, 96-dimensional tokens),
encoded through the package's sentence-transformers adapter on BEIR SciFact:
5,183 abstracts (1.22 M vectors, 236 per document), 300 test queries, float
nDCG@10 0.7456 (the model card reports 0.7477). Retention of nDCG@10.

| configuration | vectors/doc | x float32 | nDCG@10 retention [95% CI] |
|---|---|---|---|
| adaptive radius=0.95, uncentred | 85 | 2.8x | 91.6% [88.5%, 94.8%] |
| adaptive radius=0.9, uncentred | 51 | 4.6x | 98.2% [96.2%, 100.3%] |
| adaptive radius=0.8, uncentred | 9 | 24.9x | 76.3% [71.6%, 80.7%] |
| adaptive radius=0.7, uncentred | 2 | 130.4x | 62.4% [57.0%, 67.2%] |
| adaptive radius=0.9 centred | 140 | 1.7x | 100.1% [99.4%, 100.8%] |
| adaptive radius=0.6 centred | 86 | 2.7x | 99.7% [98.4%, 101.0%] |
| adaptive radius=0.5 centred | 69 | 3.4x | 97.7% [95.9%, 99.5%] |
| adaptive radius=0.4 centred | 49 | 4.8x | 96.4% [93.9%, 98.8%] |
| hierarchical ratio=0.5 | 118 | 2.0x | 99.8% [98.7%, 100.9%] |
| hierarchical ratio=0.33 | 78 | 3.0x | 99.8% [98.4%, 101.2%] |
| hierarchical ratio=0.25 | 59 | 4.0x | 93.9% [90.9%, 97.2%] |
| random ratio=0.5 | 118 | 2.0x | 91.2% [87.7%, 94.5%] |
| random ratio=0.25 | 59 | 4.0x | 78.7% [73.6%, 83.9%] |
| float16 | 236 | 2.0x | 99.9% [99.6%, 100.0%] |
| int8 per-vector | 236 | 3.9x | 100.1% [99.6%, 100.6%] |
| int4 per-vector | 236 | 7.7x | 89.7% [86.0%, 93.2%] |
| lloyd2 (2-bit) | 236 | 16.0x | 85.3% [81.3%, 89.5%] |
| binary | 236 | 32.0x | 94.1% [91.4%, 96.8%] |
| ward ratio=0.5 > int8/vec | 118 | 7.8x | 99.8% [98.6%, 100.9%] |
| ward ratio=0.33 > int8/vec | 78 | 11.8x | 100.0% [98.5%, 101.6%] |
| ward ratio=0.33 > binary | 78 | 96.3x | 92.8% [89.9%, 95.7%] |
| ward d=0.8 > float32 | 29 | 8.1x | 95.7% [92.5%, 98.8%] |
| ward d=0.8 > int8/vec | 29 | 31.7x | 95.9% [92.5%, 99.0%] |

- **What transferred from ColPali:** merging beats dropping by a wide margin
  (Ward to half the tokens keeps 99.8%, random half keeps 91.2%); int8 is
  lossless; Ward to a third plus int8 keeps 100.0% at 11.8x.
- **What did not:** text tokens compress less (about 3x at 99.8%, against 9x on
  ColPali pages), and two ColPali defaults break. This model's vectors sit in a
  narrow cone -- the median nearest neighbour inside a document is at cosine
  0.989 (0.93 for ColPali; table below) -- so an absolute cosine radius or an absolute Ward height lumps whole
  documents together (uncentred radius 0.8 leaves 9 vectors and 76% retention;
  the radius-0.95 row scoring below radius 0.9 is the same instability).
  Removing the corpus mean first (`center="mean"`) makes the adaptive merge
  behave, and budgets relative to each document (Ward fractions) are safe.
- **Codecs are model-specific too:** int4 and the 2-bit codec lose 10-15 points
  here while binary loses 6 -- the reverse of ColPali. See R3b for centred
  scalar codecs.
- This is why the default search space uses relative budgets, and why the layer
  measures instead of assuming.

**Geometry, measured** (`s3c_centred_codecs.py --geometry`, 300 sampled
documents; `reports/universal/geometry/`):

| | ColPali · DocVQA | ColPali · InfoVQA | ColBERT-small · SciFact |
|---|---|---|---|
| median cosine of each vector's nearest neighbour in its own document | 0.927 | 0.934 | 0.989 |
| norm of the corpus mean (unit-norm tokens) | 0.25 | 0.23 | 0.90 |
| median cosine of a query token to the corpus mean | 0.13 | 0.10 | 0.94 |
| centred binary | 95.2% | 98.5% | 1.1% |
| centred binary, query component along the mean removed | 94.7% | 98.3% | 96.9% [94.4%, 99.2%] |

**How far this generalises.** It is *one* text model on *one* corpus (SciFact,
300 queries). Within it the effects are large and reproducible: the centred-int4
gain and the centred-binary collapse are far outside their intervals, and the
last row confirms the mechanism (the collapse comes from the queries' shared
direction; removing it recovers the codec). Whether other text ColBERTs are this
anisotropic is **not measured**. What is supported is the narrower statement
that *some* encoders are, that absolute thresholds and centred sign codes then
fail, and that the default search space therefore avoids both. Scoring queries
with the mean direction removed is a diagnostic here, not a package feature.

### R9 · Cost of compressing

*Fixed configurations; one laptop.*

ColPali InfoVQA, 500 pages, 264 MB of float32 input, one process on the
laptop in *Methodology* while other benchmark jobs shared the CPU (so treat times
as upper bounds). Peak traced memory is numpy's allocations during `compress`,
via `tracemalloc`.

| stage | ms / page | peak traced MB (before block encoding) | output MB |
|---|---|---|---|
| float16 | 0.7 | 149 (132) | 132.0 |
| int8 per-vector | 1.8 | 134 (530) | 67.0 |
| int4 | 2.3 | 101 (530) | 34.0 |
| lloyd2 (fit + encode) | 4.1 | 126 (859) | 16.5 |
| binary | 0.1 | 18 (74) | 8.2 |
| redundancy t=0.8 | 59.2 | 146 | 70.8 |
| adaptive merge r=0.6 | 18.7 | 62 | 29.7 |
| hierarchical merge 20% | 95.9 | 131 | 54.3 |
| pca 64 (fit + project) | 1.8 | 410 | 132.0 |
| adaptive r=0.6 > int8 per-vector | 18.2 | 98 (90) | 7.5 |

- Merging is the expensive stage: 19 ms/page (adaptive) and 96 ms/page (Ward,
  SciPy) against well under 5 ms/page for any codec -- still far below the
  encoder itself (~30 s/page for ColSmol on this CPU, per the original README).
- Encoding in 65,536-row blocks (numbers in parentheses are before) cut peak
  memory 4-7x for the scalar codecs; codes are byte-identical.
- PCA fitting still holds a float64 sample (410 MB peak); not yet optimised.
- Query time is dominated by how many vectors remain: an exact scan of the
  merged corpus (116 vectors/page) took 5-7 ms/query against ~48-68 ms/query for
  the float32 original on the same machine.


### R10 · What the lower bound does and does not guarantee

*Out-of-sample and simulation; release audit (`s10_bound_audit.py`,
`s12_corpus_size.py`, `s11_ties.py`; `reports/universal/audit/`).*

The bound is `R − 3.09·sd*` at 0.999, where `R = Σc/Σb` over the calibration
queries and `sd*` is its bootstrap standard error. It is an approximate bound
on **one** configuration's retention, for queries drawn like the calibration
queries, on **this** corpus.

**Coverage for one configuration.** Treat DocVQA's 451 per-query results for a
real configuration as the population, and draw calibration sets from it (400
draws per cell, four configurations):

| calibration queries | nominal 0.975 | nominal 0.999 |
|---|---|---|
| 25 | 77–90% | 87–97% |
| 50 | 90–96% | 97–99.5% |
| 100 | 94.5–97% | 99–99.5% |
| 225 | 95–97% | 99.5–100% |

Close to nominal from about 100 queries. `calibrate()` warns below 100
calibration queries *with a nonzero baseline score*: only those carry
information, and a handful of queries that happen to agree exactly give a
zero-width interval (10 such queries passed a 0.97 target in 40% of draws when
the true retention was 90.9%).

**Choosing among many configurations.** The per-configuration level is not the
confidence of the choice. In a synthetic worst case — `K` independent
configurations, each truly 0.5–1 point *below* the target, each losing whole
queries at random — the probability that at least one passes, and is therefore
chosen:

| truth vs target | K = 1 at 0.999 | K = 10 at 0.999 | K = 40 at 0.999 | K = 40, Bonferroni (0.975 / K) |
|---|---|---|---|---|
| 0.960 vs 0.97 | 0.3% | 9.7% | 34% | 35% |
| 0.965 vs 0.97 | 2.0% | 21% | 55% | — |
| 0.940 vs 0.95 | — | 7.3% | 26% | 25% |

Bonferroni does not rescue this regime because the per-configuration bound
under-covers when losses are rare and large. Real configurations share merges
and codecs, so they are strongly correlated and the measured held-out rates
(R7b) are far better. The worst case exists all the same, and the error grows
with the size of the search space.

**The corpus changes the answer.** Retention of fixed configurations, with each
draw's queries held fixed while random distractor pages are added (5 draws):

| configuration | DocVQA 100 → 500 pages | InfoVQA 100 → 500 pages |
|---|---|---|
| binary | 98.6% → 97.5% | 99.3% → 97.4% |
| int4 | 99.2% → 97.7% | 99.6% → 99.6% |
| centred adaptive merge r=0.6 + int8 | 97.4% → 95.6% | 99.8% → 99.0% |
| adaptive merge r=0.6 + binary | 91.7% → 90.3% | 99.8% → 96.7% |

Lossier configurations lose more as the corpus grows. Calibrating on a sample of
a corpus overstates retention on the whole of it, and nothing here measures
corpora beyond 5,183 documents.

**Ties do not flatter any number.** Placing tied relevant documents first or
last instead of by index leaves every tested retention unchanged to four
decimals (binary, merged binary and merged int4 on DocVQA, InfoVQA and SciFact).

**What would be needed for a real guarantee** (not implemented): a bound that
is valid for the *selected* configuration (for example, a held-out test
of the chosen configuration alone, which the result already reports, or
conformal / union-bound methods sized to the number of configurations
actually judged), plus calibration queries drawn from the deployment
distribution against the deployment-sized corpus.

### R11 · ColQwen2 and ColQwen2.5

*Encoder check, fixed configurations, and out-of-sample (20 splits); the same
scripts and the same 451 / 494 labelled queries as ColPali
(`reports/universal/*/colqwen2*`).*

**Encoding.** `notebooks/kaggle_colqwen_encode.ipynb` on a Kaggle Tesla T4 in
float32: torch 2.10, transformers 5.18, colpali-engine 0.3.17, peft 0.19,
commit `46adfd9`.

- **ColQwen2.** Encoded from the authors' pre-merged
  `vidore/colqwen2-v1.0-merged`. The adapter-only `vidore/colqwen2-v1.0`, merged
  by our loader, gives the same vectors: median cosine 0.99999, minimum 0.9986
  on 6 pages, and at least 0.99998 on their queries.
- **ColQwen2.5.** It has no pre-merged release, so its 253 LoRA modules were
  merged explicitly over `vidore/colqwen2.5-base`. At 14 GiB in float32 it was
  split across both T4s at a decoder-layer boundary (22 / 14 layers).
- **Labels and token counts.** Every page and query label is identical to the
  ColPali files'. The two encoders produce the same number of vectors per page
  (same image processing), but their vectors are unrelated (median cosine about 0).

**A loading bug caught before any encoding.** Under transformers 5,
colpali-engine 0.3.17 maps the checkpoint's `model.layers.*` onto
`language_model.layers.*` but not `model.embed_tokens` or `model.norm`.
transformers then fills those two weights with random values and only logs it:
queries found their own page at chance level, and two loads of one checkpoint
disagreed. The smoke test stopped the run. The loader now supplies the complete
key mapping, and it refuses any load that leaves a weight uninitialised (only a
tied `lm_head` is exempt). Every vector file records an empty loading report.

**Float baselines against the published scores** (EXTERNAL: MTEB results,
mteb 1.38.34):

| encoder | DocVQA nDCG@5, ours / published | InfoVQA nDCG@5, ours / published |
|---|---|---|
| ColPali-v1.3 | 0.5841 / 0.5837 | 0.8458 / 0.8553 |
| ColQwen2-v1.0 | 0.6065 / 0.6148 | 0.9197 / 0.9251 |
| ColQwen2.5-v0.2 | 0.6180 / 0.6363 | 0.9169 / 0.9252 |

- **The ordering matches, and every encoder is within 1.8 points.** ColPali,
  validated earlier, sits 1.0 point low on InfoVQA as well, which points to the
  evaluation protocol: here each distinct query has exactly one labelled page.
- **ColQwen2.5 on DocVQA has the largest gap (−1.8) and is not explained.**
  Candidate causes are float32 here against the authors' precision, or a
  different image-resolution cap. Retention below is always measured against our
  own float index, so the gap does not enter it.

**Fixed configurations** (each pipeline scored on every query; retention of
nDCG@5 against that encoder's float32 index):

| pipeline | ColPali · DocVQA | ColPali · InfoVQA | ColQwen2 · DocVQA | ColQwen2 · InfoVQA | ColQwen2.5 · DocVQA | ColQwen2.5 · InfoVQA |
|---|---|---|---|---|---|---|
| int8 per-vector | 4x · 100.0% | 4x · 100.0% | 4x · 100.1% | 4x · 100.0% | 4x · 99.8% | 4x · 100.0% |
| Ward 1/3 + int8 per-vector | 12x · 99.2% | 12x · 99.4% | 12x · 99.7% | 12x · 99.5% | 12x · 99.4% | 12x · 99.9% |
| Ward 1/3 + int4 | 23x · 98.4% | 23x · 99.1% | 23x · 100.3% | 23x · 99.5% | 23x · 98.8% | 23x · 100.0% |
| binary | 32x · 96.3% | 32x · 97.4% | 32x · 97.2% | 32x · 99.0% | 32x · 99.5% | 32x · 99.1% |
| Ward 1/4 + binary | 125x · 95.4% | 125x · 97.5% | 123x · 96.1% | 122x · 97.6% | 123x · 98.0% | 122x · 98.2% |
| adaptive r=0.6 + int4 | 65x · 95.0% | 69x · 99.2% | 34x · 98.9% | 30x · 99.3% | 38x · 96.5% | 29x · 99.3% |

- **Relative budgets transfer.** Ward to a third of the tokens plus int8 keeps
  99.4–99.9% on every ColQwen split, as it did on ColPali.
- **ColQwen is more robust to sign codes:** binary keeps 97.2–99.5% against
  96.3–97.4% for ColPali, and Ward 1/4 + binary keeps 96.1–98.2% at about 122x.
- **An absolute merge radius does not transfer.** The same cosine radius (0.6)
  merges about half as much on ColQwen as on ColPali (29–38x against 65–69x),
  because neighbouring ColQwen vectors on a page are less alike (below). This is
  the third encoder where an absolute threshold behaves differently, and it is
  why the default search space uses per-page budgets.

**Geometry** (`s3c_centred_codecs.py --geometry`):

| | ColQwen2 · DocVQA | ColQwen2 · InfoVQA | ColQwen2.5 · DocVQA | ColQwen2.5 · InfoVQA |
|---|---|---|---|---|
| median nearest-neighbour cosine within a page | 0.886 | 0.873 | 0.904 | 0.863 |
| norm of the corpus mean | 0.28 | 0.22 | 0.31 | 0.23 |
| median query-token cosine to the corpus mean | −0.02 | 0.02 | 0.06 | 0.03 |
| centred binary | 96.5% | 99.3% | 98.6% | 99.1% |

ColQwen sits with ColPali (corpus-mean norm 0.23–0.25, NN cosine 0.93) and far
from the anisotropic text ColBERT (0.90 and 0.99). Its queries share no dominant
direction, and centred binary works, as the R8 mechanism predicts.

**What `calibrate()` / `optimize()` chose, out of sample** (defaults, labels, 20
random half/half splits; median compression · splits whose choice met the target
on the held-out half · worst held-out retention):

| encoder · split | target 0.99 | target 0.97 | target 0.95 | point-estimate rule met (0.99 / 0.97 / 0.95) |
|---|---|---|---|---|
| ColQwen2 · DocVQA | 3.9x · 18/20 · worst 97.4% | 5.9x · 20/20 · worst 98.6% | 20.7x · 20/20 · worst 97.3% | 45% / 10% / 85% |
| ColQwen2 · InfoVQA | 7.8x · 20/20 · worst 99.1% | 29.8x · 19/20 · worst 96.2% | 114.1x · 20/20 · worst 95.8% | 30% / 55% / 85% |
| ColQwen2.5 · DocVQA | 3.9x · 20/20 · worst 99.6% | 7.8x · 20/20 · worst 98.5% | 22.8x · 20/20 · worst 95.6% | 20% / 25% / 60% |
| ColQwen2.5 · InfoVQA | 9.6x · 19/20 · worst 98.7% | 47.5x · 20/20 · worst 97.0% | 122.3x · 20/20 · worst 95.6% | 55% / 40% / 100% |

- **The default met its target in 236 of 240 ColQwen choices.** Including the
  ColPali and SciFact cells (R7b), it met the target in 355 of 360 choices. The
  splits share one query pool per corpus, so that count is a summary, not a rate
  with a confidence interval; per-cell intervals are in
  `reports/universal/TABLES.md` and R7b.
- **The misses are small.** The worst was ColQwen2 DocVQA at a 0.99 target,
  97.4%. This is the multiple-configuration limitation of R10 at work, not a
  new failure.
- **The point-estimate rule fails here as it did on ColPali,** meeting the
  target in 10–100% of splits.
- **The label-free reference** chose less (2–30x) and met the label-based target
  in every split.
- **InfoVQA compresses further on ColQwen than on ColPali:** 114–122x against
  73x at a 0.95 target. DocVQA stays at 4–23x for every encoder.

**Corpus size** (`s12_corpus_size.py`, 100 → 500 pages with fixed queries,
`reports/universal/audit/colqwen_corpus_size.*`). int4 stays at about 100%. The
lossiest pipeline, adaptive merge + binary, loses 0–2 points (for example
ColQwen2.5 DocVQA 93.6% → 91.7%, ColQwen2 InfoVQA 99.8% → 98.1%), and binary
alone is noisy rather than clearly declining. The trend is weaker than on
ColPali, and the caution in R10 stands.

**Scope.** These results are two ViDoRe V1 splits of 500 pages each, encoded once
in float32 on one GPU type. Nothing here covers ViDoRe V2/V3, other resolutions
or precisions, ColQwen3, ColNomic or 2k–4k-dimensional models.

### R12 · A 2,560-dimensional encoder: NVIDIA Nemotron ColEmbed 4B

*Encoder check, fixed configurations, out-of-sample (20 splits); the same 451 /
494 labelled queries and 500-page corpora as ColPali and ColQwen
(`reports/universal/wide/`, `reports/universal/selection_rules*/nemotron-*`).*

**Why this model.** `nvidia/nemotron-colembed-vl-4b-v2` (Qwen3-VL-4B) returns, for
every token, the last decoder layer's hidden state, masked and L2-normalised, at
the full **2,560** dimensions, with no 128-d projection head. That makes it a
genuine wide late-interaction encoder.
- **Licence:** CC-BY-NC-4.0, **non-commercial use only**. It was evaluated here
  for research. Nothing in R12 says the model may be used, benchmarked or deployed
  commercially. Weights were only downloaded inside Kaggle sessions, and only
  metrics are committed.
- **Pinned revision:** `0ed152d9`. Its modelling code is remote and executes on
  load.

The NVIDIA Llama-based ColEmbed models (2,048 / 3,072-d) need transformers 4.45
remote code and were not tried.

**`nvidia/nemotron-colembed-vl-8b-v2` (4,096-d) was not benchmarked.** Its loader
exists (pinned revision `34b64061`, the same code as the 4B), but it is not part of
the measured set, and nothing here is a performance or compression claim for it.
The 2,560-d result does not establish 4,096-d behaviour.

**Encoding.** `notebooks/kaggle_wide_model.ipynb` on two Kaggle Tesla T4s:
- float32, the 36 decoder layers split 15 / 21 at a layer boundary;
- `sdpa` attention (flash-attention 2 does not run on T4s);
- torch 2.10.0+cu128, transformers 5.0.0, accelerate 1.13.0, numpy 2.0.2,
  scipy 1.16.3 (Kaggle image, October 2026), commit `ec36937`.

The encoder refuses to load unless the checkpoint is a `qwen3_vl_nemotron_embed`
ColBERT-pooling model of width 2,560, with every weight provided. Every vector
file records an empty loading report.

Two deliberate differences from the authors' pipeline: float32 instead of their
bfloat16 autocast (T4s have no native bfloat16), and `sdpa` instead of
flash-attention.

The smoke test on 6 DocVQA pages:
- each query found its own page **6/6**;
- our exact MaxSim matched the model's own `colbert_score` to **5.7e-6**;
- float16 against float32: median vector cosine 0.999998, but **minimum 0.923**,
  with identical rankings on those pages. That minimum is the reason a float16-only
  run of the 8B would need its own check.

**Pages are about 750 vectors, not 2,300.** The model card describes 8 tiles + 1
thumbnail x 256 tokens. The released processor resizes pages the Qwen-VL way
instead, giving **758** (DocVQA) and **738** (InfoVQA) vectors per page, about the
same count as ColQwen, each 20x wider.

**Float baselines against the published scores** (EXTERNAL: the model card,
ViDoRe V1 nDCG@5): DocVQA **0.6645** here against 67.39 published; InfoVQA
**0.9344** against 93.31. Same ordering. DocVQA sits 0.9 points low, the offset
ColPali and ColQwen show under this repository's one-labelled-page-per-query
protocol.

| | DocVQA | InfoVQA |
|---|---|---|
| nDCG@5 | 0.6645 | 0.9344 |
| nDCG@10 | 0.6853 | 0.9392 |
| Recall@1 | 0.5831 | 0.8947 |
| Recall@5 | 0.7361 | 0.9656 |
| vectors per page | 758 | 738 |
| float32 bytes per page | 7,765,094 | 7,556,792 |
| float16 bytes per page | 3,882,547 | 3,778,396 |
| exact scan, ms per query (T4) | 52 | 62 |
| encode time for 500 pages, s | 1587 | 1546 |

**What one page costs.** 7.6–7.8 MB in float32 (3.8–3.9 MB in float16): 15x
ColPali's 0.53 MB and 20x ColQwen's, for the same number of vectors. A
million-page float32 index would be about 7.6–7.8 TB (an extrapolation: nothing
near a million pages was run). Encoding took 3.1 s per page on two T4s.

**Fixed configurations** (retention of nDCG@5 against this encoder's float32
index, 95% bootstrap intervals; bytes are the stored codes, scales and shared
state):

*Codec only (all vectors, 2,560-d)*

| pipeline | KB/page | DocVQA | InfoVQA |
|---|---|---|---|
| float16 | 3,883 | 2x · 100.0% [99.8%, 100.3%] | 2x · 100.1% [100.0%, 100.2%] |
| int8(per_vector) | 1,943 | 4x · 99.7% [98.6%, 100.8%] | 4x · 100.2% [99.9%, 100.7%] |
| int4 | 972 | 8x · 0.9% [0.1%, 1.8%] | 8x · 0.4% [0.1%, 0.8%] |
| int4(mean) | 972 | 8x · 99.7% [98.5%, 100.9%] | 8x · 100.2% [99.8%, 100.7%] |
| lloyd2(7) | 485 | 16x · 98.7% [97.2%, 100.4%] | 16x · 100.0% [99.4%, 100.6%] |
| binary | 243 | 32x · 98.9% [97.1%, 100.6%] | 32x · 99.8% [99.2%, 100.4%] |

*Token reduction only (float32)*

| pipeline | KB/page | DocVQA | InfoVQA |
|---|---|---|---|
| hierarchical_merge(0.5) | 3,883 | 2x · 99.5% [98.3%, 100.6%] | 2x · 100.0% [99.6%, 100.4%] |
| hierarchical_merge(0.33) | 2,571 | 3x · 99.1% [97.9%, 100.3%] | 3x · 99.6% [99.1%, 100.0%] |
| hierarchical_merge(0.25) | 1,946 | 4x · 98.8% [97.5%, 100.1%] | 4x · 99.5% [98.9%, 100.0%] |
| hierarchical_merge(0.1) | 781 | 10x · 98.4% [96.3%, 100.2%] | 10x · 99.6% [98.9%, 100.3%] |
| adaptive_merge(0.6) | 756 | 10x · 97.9% [95.9%, 99.8%] | 7x · 99.3% [98.5%, 100.1%] |

*Dimension reduction only (float32)*

| pipeline | KB/page | DocVQA | InfoVQA |
|---|---|---|---|
| project(1280) | 3,883 | 2x · 100.2% [99.7%, 100.7%] | 2x · 99.9% [99.5%, 100.4%] |
| project(640) | 1,941 | 4x · 99.3% [98.1%, 100.4%] | 4x · 99.6% [99.0%, 100.2%] |
| project(320) | 971 | 8x · 99.1% [97.7%, 100.5%] | 8x · 99.4% [98.6%, 100.3%] |
| project(160) | 485 | 16x · 96.7% [94.4%, 98.8%] | 16x · 98.8% [97.7%, 99.8%] |
| project(128) | 388 | 20x · 95.2% [92.7%, 97.7%] | 20x · 97.5% [96.3%, 98.7%] |
| project(640) (random projection, control) | 1,941 | 4x · 87.6% [84.4%, 91.0%] | 4x · 97.9% [96.5%, 99.3%] |

*Token + codec*

| pipeline | KB/page | DocVQA | InfoVQA |
|---|---|---|---|
| hierarchical_merge(0.33) > int8(per_vector) | 643 | 12x · 99.2% [97.7%, 100.7%] | 12x · 99.4% [98.8%, 100.1%] |
| hierarchical_merge(0.25) > int4 | 244 | 32x · 0.7% [0.0%, 1.6%] | 32x · 0.5% [0.1%, 1.1%] |
| hierarchical_merge(0.25) > int4(mean) | 244 | 32x · 98.6% [96.9%, 100.2%] | 32x · 99.6% [98.9%, 100.2%] |
| hierarchical_merge(0.25) > lloyd2(7) | 122 | 64x · 97.3% [95.5%, 99.1%] | 64x · 99.6% [98.7%, 100.6%] |
| hierarchical_merge(0.25) > binary | 61 | 128x · 97.4% [95.3%, 99.1%] | 128x · 99.3% [98.7%, 99.9%] |

*Dimension + codec*

| pipeline | KB/page | DocVQA | InfoVQA |
|---|---|---|---|
| project(1280) > int8(per_vector) | 972 | 8x · 100.4% [99.3%, 101.5%] | 8x · 100.0% [99.7%, 100.3%] |
| project(640) > int8(per_vector) | 487 | 16x · 100.1% [98.8%, 101.3%] | 16x · 99.6% [99.1%, 100.1%] |
| project(640) > int4(mean) | 244 | 32x · 99.6% [97.2%, 102.1%] | 32x · 99.5% [98.8%, 100.2%] |
| project(320) > int4(mean) | 123 | 63x · 98.2% [95.6%, 100.7%] | 63x · 99.2% [98.4%, 100.2%] |
| project(640) > binary | 61 | 128x · 94.7% [92.6%, 97.0%] | 128x · 99.9% [99.0%, 100.8%] |
| project(320) > binary | 30 | 256x · 92.8% [90.4%, 95.4%] | 256x · 99.2% [98.2%, 100.0%] |
| project(128) > binary | 12 | 640x · 90.4% [87.7%, 93.3%] | 640x · 98.1% [97.0%, 99.2%] |

*Token + dimension (float32)*

| pipeline | KB/page | DocVQA | InfoVQA |
|---|---|---|---|
| hierarchical_merge(0.33) > project(640) | 643 | 12x · 98.5% [97.0%, 99.8%] | 12x · 99.3% [98.6%, 99.8%] |
| hierarchical_merge(0.33) > project(320) | 321 | 24x · 97.3% [95.5%, 99.0%] | 24x · 98.5% [97.4%, 99.3%] |
| hierarchical_merge(0.25) > project(128) | 97 | 80x · 92.5% [89.7%, 95.0%] | 80x · 96.2% [94.8%, 97.5%] |

*Token + dimension + codec*

| pipeline | KB/page | DocVQA | InfoVQA |
|---|---|---|---|
| hierarchical_merge(0.33) > project(640) > int8(per_vector) | 161 | 48x · 97.8% [96.3%, 99.4%] | 48x · 99.2% [98.5%, 99.8%] |
| hierarchical_merge(0.33) > project(640) > int4(mean) | 81 | 96x · 95.7% [92.8%, 98.3%] | 96x · 98.4% [97.4%, 99.4%] |
| hierarchical_merge(0.33) > project(640) > binary | 20 | 387x · 95.9% [93.6%, 98.4%] | 387x · 99.6% [98.4%, 100.6%] |
| hierarchical_merge(0.33) > project(320) > binary | 10 | 773x · 94.7% [92.2%, 97.2%] | 774x · 98.4% [97.3%, 99.3%] |
| hierarchical_merge(0.25) > project(128) > binary | 3 | 2553x · 91.1% [87.9%, 94.0%] | 2554x · 97.8% [96.6%, 98.9%] |

**Pareto frontier** (stored bytes per page against nDCG@5 retention, over all 74
fixed configurations). *These are in-sample:* each point is the best of the
evaluated grid, chosen and reported on the same queries, so it is optimistic. It
shows what the grid contains, not what `optimize()` delivers on unseen queries;
for that, see the held-out table below.

- **DocVQA:** hierarchical_merge(0.25) > project(128) > binary (2553x, 91.1%), hierarchical_merge(0.33) > project(128) > binary (1933x, 91.2%), hierarchical_merge(0.25) > project(320) > binary (1021x, 93.6%), hierarchical_merge(0.33) > project(320) > binary (773x, 94.7%), hierarchical_merge(0.25) > project(640) > binary (511x, 94.9%), hierarchical_merge(0.33) > project(640) > binary (387x, 95.9%), hierarchical_merge(0.25) > binary (128x, 97.4%), project(320) > int4(mean) (63x, 98.2%), binary (32x, 98.9%), project(640) > int4(mean) (32x, 99.6%), project(640) > int8(per_vector) (16x, 100.1%), project(1280) > int8(per_vector) (8x, 100.4%)
- **InfoVQA:** hierarchical_merge(0.25) > project(128) > binary (2554x, 97.8%), hierarchical_merge(0.33) > project(128) > binary (1934x, 98.3%), hierarchical_merge(0.25) > project(320) > binary (1022x, 99.0%), hierarchical_merge(0.25) > project(640) > binary (511x, 99.0%), hierarchical_merge(0.33) > project(640) > binary (387x, 99.6%), project(640) > binary (128x, 99.9%), project(1280) > binary (64x, 100.1%), int4(mean) (8x, 100.2%), int8(per_vector) (4x, 100.2%)

**Two tiers** (binary codes searched in RAM, a 50-page shortlist rescored from
per-vector int8 codes of every vector on disk):

| RAM tier | RAM KB/page | RAM vs float32 | disk KB/page | disk vs float32 | DocVQA | InfoVQA | rescore ms/query |
|---|---|---|---|---|---|---|---|
| binary + int8 rescoring of 50 | 243 | 32x | 2,185 | 3.6x | 99.7% | 100.2% | 343 / 351 |
| ward 1/4 > binary + int8 rescoring of 50 | 61 | 128x | 2,004 | 3.9x | 99.7% | 100.2% | 346 / 349 |

**Latency** (exact scan on a T4 through `OPTIVISION_SCORE_DEVICE=cuda`; the
per-page max and per-query sum run on the CPU):
- The float32 scan takes 46–62 ms per query (two measurements per split: 46 / 56
  ms inside the grid, 52 / 62 ms for the baseline pass).
- Merging to a quarter of the vectors cuts it to 12–14 ms; PCA to 320 dims only to
  35–42 ms. Scan time follows the number of vectors more than their width.
- Codes are decoded back to floats before the product, so **binary codes are not
  faster** here (55–63 ms). A bit-level Hamming search would be; that is not what
  is measured.
- Compression of 500 pages: Ward merging 106–112 s (about 0.2 s/page, with the
  matrix-product distances of the wide path); PCA fit and projection 6–26 s; any
  codec under 15 s, except the 2-bit codec (about 40 s).
- The two-tier rescoring took about 350 ms per query, an unoptimised Python loop
  over 50 pages x 750 vectors x 2,560 dims, so not a latency claim.

**What `calibrate()` / `optimize()` chose, out of sample** (defaults, labels, 20
random half/half splits; median compression · splits meeting the target ·
worst held-out retention). Both spaces were judged from one scoring of the 82
candidates; the default space is its 40-candidate subset:

| space · split | target 0.99 | target 0.97 | target 0.95 | point-estimate rule met (0.99 / 0.97 / 0.95) |
|---|---|---|---|---|
| default (40) · DocVQA | 2.0x · 20/20 · worst 99.8% | 8.7x · 19/20 · worst 96.9% | 32.0x · 20/20 · worst 95.8% | 25% / 60% / 55% |
| default (40) · InfoVQA | 8.0x · 20/20 · worst 99.2% | 277.7x · 20/20 · worst 98.3% | 354.8x · 20/20 · worst 96.6% | 35% / 85% / 100% |
| with projection (82) · DocVQA | 2.0x · 20/20 · worst 99.8% | 9.4x · 19/20 · worst 96.9% | 32.0x · 19/20 · worst 94.8% | 10% / 60% / 55% |
| with projection (82) · InfoVQA | 8.0x · 20/20 · worst 99.2% | 287.0x · 20/20 · worst 97.0% | 640.0x · 20/20 · worst 97.0% | 30% / 100% / 100% |

**Findings**

1. **Merging still works at 2,560-d.** Ward to a quarter of the vectors keeps
   98.8% / 99.5%, and to a tenth 98.4% / 99.6%, comparable to ColPali and ColQwen.
2. **PCA projection works; a random projection did much worse.**
   - PCA to 320 dimensions (8x) keeps 99.1% / 99.4%.
   - It degrades below that: 160 keeps 96.7% / 98.8%, 128 keeps 95.2% / 97.5%.
     128 dimensions are not enough for this model on DocVQA.
   - A random projection to 640 keeps 87.6% on DocVQA (PCA: 99.3%) and 97.9% on
     InfoVQA (PCA: 99.6%).
3. **Quantization works, with one new failure.**
   - Per-vector int8 and centred int4 show no measurable loss (99.7–100.2%, every
     interval covering 100%).
   - Binary keeps 98.9% / 99.8% at 32x, better than ColPali (96.3% / 97.4%).
   - **Plain (uncentred) int4 collapses to 0.4–0.9%**, alone and after merging.
4. **Which axis matters most depends on what is being saved.** The figures in this
   item are fixed configurations read off the in-sample frontier above, so they are
   optimistic, not held-out optimizer results.
   - *Storage:* width is the most efficient axis at moderate compression. PCA to
     320 reaches 8x at 99.1%, where merging needs 10x for 98.4%. PCA + codec owns
     the frontier from 8x to 63x on DocVQA (`project(640) > int8` 16x at 100.1%,
     `project(320) > int4(mean)` 63x at 98.2%).
   - *Beyond about 100x:* all three axes together (Ward + PCA + binary: 387x at
     95.9% / 99.6%; about 2,550x at 91.1% / 97.8%).
   - *Scan time:* token reduction is what cuts it.
   - Treating token count and width as separate axes pays off: the two-axis and
     three-axis pipelines dominate the frontier.
5. **`optimize()` stays reliable and conservative.**
   - It met the target in **119 of 120** choices with the default space and
     **118 of 120** with projection families added (worst miss 94.8% against 0.95,
     DocVQA).
   - It adapts to the split: on InfoVQA it chose **278–355x** (default) and up to
     **640x** (with projection) at 0.97 / 0.95, all 20 splits meeting the target.
   - On DocVQA it stayed at 2–32x, where the in-sample frontier shows 16–63x at
     ≥98%.
     The 0.999 bound on about 225 calibration queries is too wide to certify those.
   - Adding projection changed DocVQA little and raised InfoVQA's 0.95 choice from
     355x to 640x, at the cost of one more miss on DocVQA.
   - The point-estimate rule again missed often: per cell it met the target in
     only 10–100% of splits (as few as 2 of 20).
   - These are empirical held-out counts over 20 splits per cell, not a guarantee
     and not an independent "99% success probability". The splits reuse one query
     pool per corpus, and the 0.999 bound covers each configuration individually,
     not the one selected among 40 or 82 candidates (R10).
6. **New failure modes at this width.**
   - **Plain int4 collapse (above). The cause is unresolved.** What was observed:
     per-vector uncentred int4 retains 0.4–0.9%, while centred int4, per-vector int8
     and binary codes on the same vectors retain 98.9–100.2%. Diagnosing it needs
     the raw vectors, which were not kept. The default search space contains only
     centred int4, so `optimize()` never chose the failing codec; an explicit
     pipeline can still use it.
   - **Size:** 7.6–7.8 MB per page in float32 (7.6–7.8 TB per million pages, by
     extrapolation), and 3.1 s per page to encode on two T4s. The raw index, not
     the encoder, is the bottleneck.
   - **Fitting at this width needed engineering changes before any run.** The
     200,000-row fitting samples (PCA, centring, per-dimension int8) would have held
     about 4 GB in float64 at 2,560-d, and the calibration cache kept whole transformed corpora.
     Both were capped by bytes. Ward distances moved to a matrix product (17x
     faster at 2,560-d, identical clusterings).
   - **float16 encoding is not free** (minimum vector cosine 0.923).
   - **The model card overstated vectors per page** (2,304 vs about 750 measured).

**Scope.** One 2,560-d model, two ViDoRe V1 splits of 500 pages each, encoded
once in float32 on T4s. Not measured: the 4,096-d 8B sibling, the 2,048 / 3,072-d
Llama-based ColEmbed models, ViDoRe V2/V3, corpora beyond 500 pages (nothing near
a million), and a bit-level binary search kernel.

**Reproduction.**
- **Model:** `nvidia/nemotron-colembed-vl-4b-v2` at revision `0ed152d91f8ad4c5d48296b51c220f686641a398`.
- **Datasets:** `vidore/docvqa_test_subsampled` and `vidore/infovqa_test_subsampled`
  (test split, 500 rows each). Page ids are `00000::p1`..`00499::p1`; the first
  occurrence of each distinct query is labelled with its row, giving 451 and 494
  queries. The labels are identical to the ColPali/ColQwen files.
- **Code:** `notebooks/kaggle_wide_model.ipynb` (one split per session),
  `scripts/wide_smoke.py`, `scripts/encode_vectors.py --backend nemotron`,
  `scripts/universal_study/s13_wide_study.py` and
  `s7b_selection_rules.py --space=both`.
- **Hardware:** two NVIDIA Tesla T4 (15 GB each). The float32 model needs about
  18 GB across them. A CPU-only run is not practical (encoding alone was 3.1 s per
  page on GPUs), and the vectors (3.8–3.9 GB of float32 page vectors per split) were not kept.
- **Outputs:** `reports/universal/wide/` (fixed study, smoke report, logs, commit
  and GPU records), plus `reports/universal/selection_rules/` and
  `selection_rules_wide/` (`nemotron-*`).
- **Steps:** `docs/GPU_RUN.md`.

## Limits of what is measured

- **Model coverage is six encoders in four families**:
  - the PaliGemma family: ColPali-v1.3 (two ViDoRe V1 splits and the generated
    corpus) and ColSmol-256M (the 60-page generated corpus only, so a sanity
    check rather than evidence);
  - the Qwen-VL family: ColQwen2-v1.0 and ColQwen2.5-v0.2 (the same two ViDoRe V1
    splits, encoded on a Kaggle T4, R11);
  - one text ColBERT (SciFact);
  - one wide encoder: NVIDIA Nemotron ColEmbed 4B at 2,560 dimensions (Qwen3-VL,
    the same two ViDoRe V1 splits, CC-BY-NC-4.0, R12).

  The other five are 96–128-dimensional. Nothing between 128 and 2,560 dimensions,
  and nothing above 2,560 (in particular no 4,096-d model), is measured.
- **Calibration does not guarantee the target, and nothing in this package is a
  distribution-free guarantee.** The 0.999 lower bound is an approximate
  bootstrap bound for *one* configuration, for *the query distribution the
  calibration queries were drawn from*, on *this* corpus. It does not account
  for choosing among many configurations, for queries unlike the calibration
  set, or for a corpus that grows or changes (R10). What is measured is how
  often the choice met the target on held-out queries (R7b).
- **Corpora are small**: 60 to 5,183 documents, and retention measurably falls
  as distractors are added (R10). At a million pages each query faces far more
  near-duplicates.
- **Dimension reduction was measured on 96–128-d vectors and on one 2,560-d model**
  (R12); nothing at 4,096-d.
- **The text-model findings are one model on one corpus** (R8).
- **Document fragments are not a substitute for real queries.** Calibrating on
  them chose configurations that lost 6–18% on real queries while reporting
  97–100% (R7), so `optimize()` refuses to run without sample queries.
- **Latency is a relative, single-machine number** from an exact numpy scan, not
  an ANN index or a database.

## External evidence (reported elsewhere, not reproduced here)

- The text ColBERT's model card reports SciFact nDCG@10 0.7477. We measured
  0.7456 with the float index, which is the only external figure we checked.
- Ward token pooling for ColBERT-style indexes is from Clavie et al. (2024).
  Their compression / quality figures were not reproduced; ours are in R2.
- Trained merging networks (MarginMerge-style), attention-guided merging
  (AnchorFold) and trained projection heads report stronger compression than
  anything here. None were reproduced, and none of their numbers appear in this
  document. `docs/AUDIT-2026-09-27.md` §4 lists them with their status.

## Unvalidated: in the code, not benchmarked

- Adapters for ColSmol-500M (*untested*: the loader exists) and for ColNomic,
  ColBERTv2 and GTE-ModernColBERT (*unverified*: written, never run). `optivision.adapters.supported_models()` returns the status of
  each.
- Attention-guided merging: `AdaptiveMerge` can seed on an `importance`
  signal, but no encoder here exposed attention.
- The default search space without SciPy (no Ward families); `calibrate()` warns.
- `Lloyd2Quantizer` and `PCAProjector` inside `optimize()`: available for
  explicit pipelines, but not in the default search space. The opt-in PCA
  families (`calibration.wide_search_space`) were measured only on the R12 encoder.
- The Qdrant bridge for new codecs: only float32, plain binary and fixed int8
  codes map onto the original numpy / Qdrant backends (`to_legacy_pages`).

## Future work

- 4,096-d (nemotron-colembed-vl-8b-v2: loader ready, not benchmarked; float16 only
  on two T4s, so it would first need its own precision check), the Llama-based
  NVIDIA ColEmbed models (2,048 / 3,072-d), ColQwen3, ColNomic and Jina v4.
- Explaining the plain-int4 collapse on wide hidden states (R12; the cause is
  unresolved).
- ColQwen2/2.5 beyond ViDoRe V1 DocVQA and InfoVQA, at other precisions or image
  resolutions, and an explanation of ColQwen2.5's 1.8-point DocVQA baseline gap.
- ViDoRe V2/V3, long documents, multilingual and non-document images.
- Million-page corpora, and calibration against a deployment-sized corpus.
- Database connectors behind `StorageBackend`: Milvus, Weaviate, Vespa,
  Elasticsearch/OpenSearch, LanceDB.
- A selection rule with a guarantee for the *selected* configuration (R10).
- Learned components: trained merging, Matryoshka or distilled projections,
  per-model adapters.

**Release status:** v0.3.0 (2026-10-02, PyPI and npm) contains R11 and R12; v0.2.0 was published on 2026-10-01.
