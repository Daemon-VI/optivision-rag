# Q8 plan: how resources scale with corpus size, and what can legitimately be projected (frozen before any Q8 computation)

Branch `research-investigation`, after Q7 closed (`f9087b0`).
- No library change, no GPU.
- No retrieval-quality claims at scale (see §8).

**Labels:**
- **MEASURED:** from artifacts in this repository;
- **EXTERNAL:** none planned;
- **PROJECTED:** arithmetic on measured primitives under a stated assumption;
- **INFERENCE:** our reading.

## 1. What is scaled

**Corpus size N (pages)**, at 1,000, 10,000, 100,000 and 1,000,000, alongside the
measured 500.

Held fixed per configuration, at their measured values:
- vectors per page (Q4 store, actual means);
- dimension;
- bytes per vector (actual codes, including per-vector scales).

**Accounting** (the Q4 convention):
- index bytes = codes + offsets (8 bytes per page) + shared state;
- shared state = codec state + PCA basis, a fixed cost independent of N;
- compression ratio against the float32 in-memory codes of the same model.

**Per query, held fixed:** query tokens, at the measured mean per dataset, and the
two-tier candidate count of 50 (the Q4 / R12 policy).

**Temporary working memory:** the library's scan block cap (256 MiB) bounds the live
similarity block independently of N. The output score matrix
(n_queries × N × 4 bytes) grows with N for batch scoring.

## 2. Configurations (fixed now, not chosen after projections)

**128-d models** (ColPali v1.3, ColQwen2 v1.0, ColQwen2.5 v0.2; DocVQA and InfoVQA kept
separate), Q4 labels:
- float32 (baseline), float16, int8(per_vector), int4(mean), binary;
- hierarchical_merge(0.25);
- hierarchical_merge(0.25) > int8(per_vector);
- hierarchical_merge(0.25) > binary;
- project(64) > int8(per_vector);
- hierarchical_merge(0.25) > project(64) > binary;
- two-tier: hot Ward 1/4 > binary, cold int8(per_vector) of all vectors, 50
  candidates.

**ColEmbed 4B, 2,560-d** (R12 rows; DocVQA and InfoVQA separate):
- float32, int8(per_vector), binary;
- hierarchical_merge(0.25) > int8(per_vector);
- project(640) > int8(per_vector);
- hierarchical_merge(0.25) > project(640) > binary;
- R12's two-tier rows (binary hot and Ward 1/4 > binary hot, int8 cold, 50
  candidates).

**Not projected:** 4,096-d or any other untested model or width.

## 3. Measured primitives (reused; no rerun)

| source | what it provides |
|---|---|
| `Q4/store/*.json` | per-page vectors, dimension, code / offset / shared-state bytes, disk bytes |
| `Q4/latency/*_docvqa.json` | CPU scan ms/query at N = 500 for 31 configurations on 3 models |
| `Q5Q6/latency.json` | two-tier component times (first stage, rescoring) on ColQwen2 DocVQA |
| `reports/engineering/batched_rescore/timing.json` | old and batched rescoring, 6 datasets |
| R12 (`reports/universal/wide/`) | ColEmbed bytes per page and GPU scan ms/query at N = 500 |
| R10 (`reports/universal/audit/*corpus_size*`) | retention from 100 to 500 pages; quality context only |

## 4. Scaling classes (assigned with stated justification)

Each resource gets one class: approximately linear in N, sublinear, fixed per query
(independent of N), implementation-dependent, or unknown.

Expected classes are listed here, to be confirmed or rejected:
- **A. Index storage:** linear in N, plus a fixed shared state.
- **B. Resident RAM:** in the current implementation, equal to A.
- **C. Full-scan compute:** expected linear in total vectors (N × vectors per page).
  **Not established by any existing measurement** (only N = 500 was timed), so V1
  tests it.
- **D. Decode work:** linear in scanned vectors.
- **E. First-stage binary work:** linear in hot vectors (test in V1).
- **F. Two-tier rescoring:**
  - per query, depends on the candidate count, not on N, for the old per-query loop;
  - for the batched implementation, depends on how candidates are shared across the
    queries in a batch, which falls as N grows.
  - Implementation-dependent, so V2 tests it.
- **G. Query-independent metadata:** offsets are linear in N but negligible; shared
  state is fixed.
- **H. Temporary working memory:** the scan block is fixed; the output score matrix is
  linear in N × the number of queries in the batch.

## 5. Small validation experiment (needed because no timing exists at more than one N)

**Data.** Synthetic multiples of the real ColQwen2 DocVQA corpus: the 500 pages tiled
m ∈ {1, 2, 4, 8} times (500–4,000 pages). Each copy is an identical duplicate.
- The tiled corpora are used **for timing only**.
- **No retrieval metric is computed on them**, and none is reported.

**V1: scan time against N.**
- Configurations: float32, int8(per_vector), binary, hierarchical_merge(0.25),
  hierarchical_merge(0.25) > binary, project(64).
- Each is compressed once at m = 1 and tiled for larger m, so the codes are
  byte-identical copies.
- Timing: `ExactIndex.score` with all 451 queries, under the Q4 / Q5/Q6 protocol (one
  untimed warm-up run per configuration and m, then 5 timed rounds, round-robin over m
  within each configuration).
- **Linear** if the median ms/query at m = 8 lies within ±15% of 8 × the m = 1 value
  and a straight line through the four medians has R² ≥ 0.98. Otherwise the relation
  is reported as observed.

**V2: rescoring cost against candidate overlap.**
- Candidate sets: 50 pages per query, drawn uniformly at random without replacement
  from the m-tiled corpus (seed 0), for m ∈ {1, 2, 4, 8}. Overlap between queries
  falls as N grows.
- Old per-query rescore against batched `ExactIndex.rescore` on the int8 cold tier,
  same protocol.
- The mean number of distinct pages per batch is reported.
- This measures how the batched gain depends on overlap. Random candidates are a
  stand-in for "low overlap" and are not real shortlists.

**Run conditions:** alone on the machine, sizes capped at 8× (about 1.5 GB float32)
because of the 15.8 GB RAM and earlier low-memory kills.

## 6. Projections (only after §4–5)

- **Storage and RAM (PROJECTED):** per-page bytes × N + shared state, for each
  configuration.
- **Practical-RAM crossover (PROJECTED):** the N at which each configuration's
  resident index exceeds 8 GB, about half of this laptop's RAM. The threshold is
  stated as an assumption.
- **Search cost (PROJECTED):** used only if V1 supports linearity.
  - ms/query(N) = measured ms/query(500) × (N / 500), using the V1 slope where it
    differs.
  - These are labelled "PROJECTED from measured X under assumption Y".
  - No projection is made beyond the regime where the index fits in RAM (no disk or
    paging model).
- **Two-tier (PROJECTED):** first stage linear (V1); rescoring as measured per query
  under V2's low-overlap condition.
- **ColEmbed:** storage only from its measured bytes. Its latency is GPU (R12) and is
  not projected onto the CPU model.

## 7. Model-family sensitivity

The 128-d models are compared with each other and with ColEmbed 4B on per-page bytes
and on the projected-RAM crossover. Nothing is extrapolated to other widths.

## 8. Explicitly out of scope

- **Retrieval quality at larger N** is not measured and not claimed. R10 shows
  retention falling 0–2 points from 100 to 500 pages for lossy pipelines; that is the
  only quality-versus-size evidence.
- **No claim of "validated", "demonstrated" or "production-ready" at any size beyond
  500 real pages.** The V1 and V2 timings cover up to 4,000 duplicated pages.
