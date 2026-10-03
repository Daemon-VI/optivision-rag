# Q4 plan: which compression axis drives quality-preserving compression? (stated before any Q4 computation)

Branch `research-investigation`, from `ed80fe1`. Nothing here changes the library,
`main`, the defaults, Method C or any release. Q4 is an **axis-ablation study**. It
is not a selection-method study: no optimizer and no Method C are used. Every
configuration is fixed in advance, below.

## 1. Question

Separate the effects of three axes, alone and combined:
- **A**, vector-count reduction;
- **B**, dimension reduction;
- **C**, quantization;

on five objectives, each reported separately and never folded into one score:
1. retrieval quality;
2. storage;
3. RAM;
4. search latency;
5. preprocessing (compression) cost.

## 2. Hypotheses (recorded before running; any may be rejected)

- **H4a.** Vector-count reduction contributes disproportionately to search-latency
  reduction.
- **H4b.** Dimension reduction contributes disproportionately to storage/RAM
  reduction.
- **H4c.** Quantization provides substantial additional compression after
  structural reduction, but codec behaviour is model-dependent.
- **H4d.** Combining axes produces interactions and cannot be accurately predicted
  by multiplying independent compression factors.
- **H4e.** The qualitative axis trade-offs generalize from narrow models to the
  2,560-d ColEmbed model.

### Operational criteria (fixed now)

**H4a: latency.** At matched single-axis storage factors (2x, 4x, 8x), compare:
- A, Ward to 1/2, 1/4, 1/8 of the vectors;
- B, PCA to d/2, d/4, d/8;
- C, float16, int8, centred int4.

Supported if, for every 128-d model at every level, A's median scan time is below
both B's and C's by more than the larger of the two configurations' repeat spreads
(max − min over repeats). Rejected if B or C is at least as fast as A at any level.
Otherwise it is mixed.

**H4b: storage and RAM efficiency.** "Disproportionate" means the most quality per
byte saved. At the same matched factors, compare retention for A, B and C.
- Supported if B's retention is the highest of the three at ≥ 2 of the 3 levels, on
  ≥ 2 of 3 models, with the paired 95% interval of (R_B − R_other) excluding 0.
- Rejected if another axis meets that criterion instead.
- Otherwise it is mixed.
- R12's ColEmbed rows are judged the same way, on point estimates and their
  published intervals.

**H4c: quantization after structural reduction.** For each structural point (A, B
and A+B rows without a codec), compare it with the same point plus a codec.
- **Part 1 (additional compression).** Supported if int8 adds ≥ 3.5x at a retention
  change ≥ −0.01 at every structural point, on every model.
- **Part 2 (model-dependent behaviour).** Supported if any of these differ across
  models:
  - the codec ordering by retention, at an equal structural point;
  - which codecs fail (retention < 0.90);
  - the retention of one codec, by more than 2 points.
- Plain int4 is included as a known failure mode and kept visible.

**H4d: interactions.**
- For a combination X+Y (and X+Y+Z), the independence prediction is the product of
  the single-axis retentions measured on the same queries (R_X · R_Y). The
  **interaction** is I = R_{X+Y} − R_X · R_Y, with a paired bootstrap 95% interval
  over queries (1,000 resamples, seed 0, the same indices for every term).
- Supported (quality) if, on ≥ 2 of 3 models, ≥ 1/4 of the two- and three-axis
  combinations have |I| ≥ 0.01 with the interval excluding 0.
- Separately, the **byte factor** of every combination is compared with the product
  of its single-axis byte factors. Exact multiplicativity is not assumed, because
  per-vector scales do not shrink with width.

**H4e: generalization.** Supported if ColEmbed 4B (R12 rows, 2,560-d) gives the same
qualitative answer as the majority of the 128-d models on:
- H4a, comparing latency within each model only;
- H4b;
- H4c's failure pattern;
- the H4d direction (point estimates only, since R12 has no per-query data).

It is judged on normalized curves (fraction of the model's own float32 bytes and
own scan time), never on raw "Nx" numbers.

## 3. Data (no new encoding, no new models)

**Inventory (MEASURED):**
- **ColPali v1.2:** DocVQA and InfoVQA vectors on disk (`data/cache`), 128-d.
- **ColQwen2 v1.0:** DocVQA and InfoVQA vectors on disk (`data/vectors`), 128-d.
- **ColQwen2.5 v0.2:** DocVQA and InfoVQA vectors on disk, 128-d. E1.2 is
  complete, so these are no longer held back.
- **ColEmbed 4B (nemotron-colembed-vl-4b-v2):** **no vectors on this machine.** The
  only artifacts are R12's result files: 74 rows per split, each with an aggregate
  nDCG@5, a paired bootstrap 95% interval and a GPU scan time.
  - ColEmbed is therefore reused **as measured in R12**, and nothing about it is
    recomputed.
  - Cells the R12 grid lacks stay missing and are not projected.
  - No per-query data exist for it, so it has no interaction intervals.
  - Its scan times are GPU timings (Kaggle T4 x2) and are compared only within
    ColEmbed.
- **ColSmol and SciFact:** not used, because they are not among the four models
  named for Q4.

**Metric:** nDCG@5 with one labelled page per query, the established ViDoRe
protocol. Retention is mean(c) / mean(b) against the same encoder's float32 index,
with the library's paired bootstrap interval (`evaluation.retention`, 1,000
resamples, seed 0).

## 4. Configurations (128-d models; one grid for all six model × split datasets)

**Stages (library calls only):**
- `HierarchicalMerge(ratio=r)`: Ward merging within each page.
- `DimensionProjector(w)`: uncentred PCA on the documents' second-moment matrix. It
  is fitted on document vectors only; queries are projected with the same map.
  `DimensionProjector(32, "random")` is the control.
- Codecs: `Float16Quantizer`, `Int8Quantizer("per_vector")`, `Int4Quantizer()`
  (plain), `Int4Quantizer(center="mean")`, `Lloyd2Quantizer` (2-bit),
  `BinaryQuantizer`.

**Order:** merge, then project, then codec, as in R12. Each stage is fitted on the
previous stage's output. So the PCA basis in an A+B row is fitted on merged vectors.
That refit is part of what combining the axes means, and it is recorded.

| group | configurations |
|---|---|
| baseline | float32 |
| A count only (float32, 128-d) | Ward 1/2, 1/3, 1/4, 1/8, 1/10; centred adaptive merge r = 0.6 (a second method, reported but not used for criteria) |
| B dimension only (float32, all vectors) | PCA 64, 32, 16, 8; random projection 32 |
| C codec only (all vectors, 128-d) | float16, int8/vec, int4 (plain), int4c, 2-bit, binary |
| A+C | Ward 1/3 and 1/4 × each of the six codecs; Ward 1/2 > int8 |
| A+B | Ward 1/3 and 1/4 × PCA 64, 32, 16; Ward 1/2 > PCA 32 |
| B+C | PCA 64, 32, 16, 8 × float16, int8/vec, int4c, binary |
| A+B+C | Ward 1/3 and 1/4 × PCA 64, 32, 16 × int8/vec, int4c, binary; Ward 1/2 > PCA 64 > float16 |
| two-tier (R12 parity) | binary (and Ward 1/4 > binary) in RAM; a 50-page shortlist rescored with int8 of every vector |

**Matched budgets for the interaction comparison** (the factor is relative to
float32 bytes; actual factors are reported, and per-vector scales make int8 about
3.9x and centred int4 about 7.5x):
- **about 4x:** A = Ward 1/4; B = PCA 32; C = int8; A+B = Ward 1/2 > PCA 64.
- **about 8x:**
  - A = Ward 1/8; B = PCA 16; C = int4c;
  - A+B = Ward 1/4 > PCA 64 and Ward 1/2 > PCA 32;
  - A+C = Ward 1/2 > int8 and Ward 1/4 > float16;
  - B+C = PCA 64 > int8 and PCA 32 > float16;
  - A+B+C = Ward 1/2 > PCA 64 > float16.
- **about 32x:**
  - C = binary;
  - A+B = Ward 1/4 > PCA 16;
  - A+C = Ward 1/4 > int4c;
  - B+C = PCA 64 > int4c and PCA 32 > int8;
  - A+B+C = Ward 1/4 > PCA 64 > int8.
  - A alone (1/32 of the vectors) and B alone (PCA 4) are not run: they are outside
    any useful range. Those cells are reported as **not run**, not projected.

## 5. Accounting (one convention for every row)

**Baseline:** the float32 **in-memory** representation of the original vectors,
num_vectors × d × 4 bytes. This is the library's `compression_vs_float32`, and
every compression factor is relative to it.

**Storage, per page (MEASURED from the actual arrays):**
- **code bytes:** the codes array, including per-vector scales;
- **index bytes:** code bytes + the offsets array + shared codec state (fitted
  means and codebooks) + the projection basis (d × w float32, plus the mean if
  centred). Shared state is amortized over the corpus's pages, so it is reported
  in absolute bytes as well as per page.
- bits per vector, vectors per page and dimension.

**RAM:**
- The current `ExactIndex` keeps the codes resident and decodes page-aligned blocks
  to float32 during a scan, with the live block capped at `max_block_bytes`
  (256 MiB).
- So the **resident index RAM** equals the index bytes above. The **transient scan
  memory** is bounded by the same block cap for every configuration, and is
  reported as a bound, not measured per row.

**Disk:**
- The library's `save_compressed` writes codes + offsets + a JSON header. It refuses
  codecs with fitted state (int4c, 2-bit), and it stores the projection
  *parameters*, not the fitted basis.
- **Disk** is therefore the MEASURED `.npz` size where the library can write the
  configuration, with the basis bytes added and stated.
- For configurations the library cannot persist, disk is marked **not persistable
  by the current library**: a finding, not fixed here, because that would be a
  library change. Its index bytes are still reported.

**Preprocessing:** wall time to fit and apply the pipeline (`compress_seconds`), on
one CPU run (MEASURED, noisy). Encoder forward time does not depend on the
compression choice and is not part of any comparison.

## 6. Latency protocol (existing implementation only; nothing optimized)

- **Scope:** the existing `ExactIndex.score` (query projection plus a blockwise
  decode and MaxSim in numpy float32). There is no new backend, no native binary
  scoring and no tuning.
- **Runs:**
  - one dataset per 128-d model (DocVQA);
  - every single-axis row and every matched-budget row;
  - all 451 queries per timing;
  - 5 repeats, round-robin over configurations to spread thermal drift;
  - reported as the median, with the min–max spread.
  - Index construction is outside the timed region.
- Same machine, same process and same thread settings, all recorded. Values are
  milliseconds per query and also a fraction of the model's own float32 scan.
- **The two-tier rows** report the hot scan and the rescore separately.
- **Laptop CPU timings are not comparable to R12's GPU timings.** Only within-model
  ratios are compared across widths.
- Python overhead is part of the existing implementation and is not reported as an
  algorithmic result.

## 7. Analyses

1. **Per-axis curves:** retention against the fraction of own float32 bytes,
   resident RAM and scan time, for A, B and C separately (normalized; no raw "Nx"
   comparisons across widths).
2. **Matched budgets (§4) and interactions (H4d),** with paired intervals.
3. **Pareto frontiers, separate for storage, RAM and latency:**
   - **IN-SAMPLE SELECTION:** frontier membership decided on all queries. This
     describes the grid and is not unbiased.
   - **Held-out:** membership decided on the calibration half
     (`split_queries(n, 0.5, seed=0)`, the library's own split) and retention
     reported on the other half.
   - For each frontier region, the axis composition of its members.
4. **Cross-model comparison** of the qualitative answers (H4e).
5. **Reproduction checks (stop if any fails):**
   - Q4 rows that coincide with the 40 E1 candidates must give per-query nDCG@5
     identical to the E1 store, for every dataset.
   - Retention of the overlapping R5b/R11 frontier rows must match within
     floating-point tolerance.

## 8. Stop conditions (from the Q4 brief)

Stop and report, without working around it, if:
- a "single-axis" row changes a second axis;
- representations use different storage conventions;
- a codec changes scoring semantics;
- an artifact is missing for a planned cell;
- latency is not comparable within a model;
- a library change is needed;
- a result contradicts R11/R12;
- or something is chosen after seeing the test queries.

ColEmbed's missing vectors are known now and are handled by §3's rule: R12 rows as
measured, and missing cells left missing.

## 9. Labels

**MEASURED** (computed here, or in R12 within this repository) · **EXTERNAL**
(reported elsewhere) · **EXTRAPOLATED** (beyond the measured range; none planned;
no million-page claims) · **INFERENCE** (our reading).

## 10. Clarifications added before any Q4 result was inspected (commit after `709f421`)

1. **Model-level verdicts with two splits.** H4b, H4c and H4d are evaluated per
   dataset. A criterion counts for a model only if it holds on **both** its DocVQA
   and InfoVQA splits. H4a uses DocVQA only, the latency subset of §6.
2. **ColEmbed storage accounting.**
   - R12's `bytes_per_doc` counts codes plus codec state, but not offsets or the
     projection basis.
   - For ColEmbed rows Q4 adds both: offsets of 501 × 8 bytes, and a basis of
     2,560 × w × 4 bytes, which follows from the recorded shapes. That puts every
     model on the same index-bytes convention.
   - Shared state is amortized over each corpus's 500 pages and is also reported
     separately. No other corpus size is extrapolated.
3. **H4a matched levels for ColEmbed.** R12 has no Ward 1/8, so the 8x level uses
   Ward 1/10 (9.9x), the nearest measured row. This is marked in the table.
