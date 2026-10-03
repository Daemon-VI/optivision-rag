# Q5/Q6 plan: scoring compressed codes directly (stated before any Q5/Q6 measurement)

Branch `research-investigation`, from `0e126ee`. All code lives in
`scripts/research/q56_*.py`, outside the package. The library is used read-only and
nothing in `src/` changes. No new model, encoding, dataset or GPU.

## 1. Setting

- **Data:** ColQwen2 v1.0 on ViDoRe V1 DocVQA (`colqwen2_docvqa`): 500 pages and
  374,154 document vectors of 128-d, 451 queries with 9,111 query tokens (14–45 per
  query). The vectors are the existing float32 files.
- **Primary two-tier configuration** (Q4 / R12 policy):
  - hot tier in RAM: `HierarchicalMerge(0.25) > BinaryQuantizer()`, 195.4 vectors
    per page;
  - cold tier: `Int8Quantizer("per_vector")` over all vectors;
  - **candidate policy: 50 pages**. This is the `TieredIndex(..., candidates=50)`
    setting of R12 and Q4, fixed and not tuned.
- **Secondary** (reported, not used for the hypotheses): the Q4 row with plain
  `BinaryQuantizer()` over all vectors as the hot tier, same cold tier, same 50.
- **Metric:** nDCG@5, with retention against the float32 index (the library's
  paired bootstrap, 1,000 resamples, seed 0).
- **Ranking rule:** `optivision.scoring.rank`, a stable descending sort with ties
  broken by page index.

## 2. Exact mathematics

**Per-vector int8.**
- Each document vector v is stored as codes q ∈ {−127..127}^128 (int8) and a scale
  s (float16), the vector's max |component|.
- Let c = fl32(fl32(s) / 127), the decode multiplier the library uses.
- **Current score** (decode-first, `Int8Quantizer.decode` followed by `x @ D.T`):

      sim(x, v) = Σ_j x_j · fl32(q_j · c)        (float32 products; BLAS accumulation order)

- **Direct score:**

      sim'(x, v) = c · Σ_j x_j · q_j              (the q_j stay int8; the scale is applied once per vector)

- In exact arithmetic sim = sim'. In float32 they **differ by rounding**:
  - the current path rounds every product q_j · c to float32 before the dot product;
  - the direct path rounds the dot product and then multiplies by c once.
- Bitwise-equal scores are therefore not guaranteed by any non-decoding formulation
  unless c is a power of two. Ranking equality is what can be tested; score
  equality cannot be expected.
- MaxSim is unchanged in both: the max over each page's vectors, then the sum over
  query tokens.

**Binary.**
- Codes are sign bits, with bit 1 meaning v_j > 0, decoded to s_j = ±1.
- **Current hot score:** Σ_j x_j · s_j, computed as a GEMM against the decoded ±1
  matrix.
- **B1, exact float-query scoring on packed bits:**
  - For each query token and each byte position b (16 bytes for 128-d), build a
    256-entry table LUT_b[k] = Σ_{i=0..7} x_{8b+i} · (bit i of k ? +1 : −1), using
    `np.packbits` order (big-endian bits).
  - Then sim = Σ_b LUT_b[code_b].
  - This is the same mathematical function: an exact rearrangement of the ±1 sum.
    Only the float rounding differs.
- **B2, approximate binarized query:**
  - The query bit is t_j = (x_j > 0), the same rule as the documents.
  - sim_B2 = Σ_j t'_j · s_j = d − 2 · popcount(t XOR code), with integer scores.
  - This is a **different similarity**, not an approximation of B1's arithmetic, and
    it is never called exact.

## 3. Implementations (all outside the library)

| id | path | what it computes | memory traffic |
|---|---|---|---|
| F32 | float32 full scan | `ExactIndex(corpus).score` | float32 vectors |
| I8-cur | current int8 full scan | `ExactIndex(int8 store).score`: blockwise decode + BLAS | decoded float32 blocks (≤ 256 MiB) |
| **I8-dir** | direct int8 full scan | per page block, `torch.ops.aten._weight_int8pack_mm(x, q_int8, c)` (float32 queries, int8 codes, per-row scale); MaxSim with the library's `segment_reduce` | **no decoded float32 document matrix** is formed |
| R-cur | current shortlist rescoring | `ExactIndex(int8).rescore`: a per-query Python loop, decoding the 50 candidate pages per query | decodes about 37k vectors per query |
| **R-dir** | direct batched rescoring | grouped **per page**: every query token whose query shortlisted page p, against p's int8 codes, in one `_weight_int8pack_mm` call per page; per-token max, then summed per query | no decoded document matrix; no per-query loop |
| R-bdec (control) | batched decode-first rescoring | the same per-page grouping, but decode p's codes and use a float32 GEMM | decoded single page (≤ about 1,000 vectors) |
| H-cur | current binary hot scan | `ExactIndex(binary store).score` (decode ±1 + BLAS) | decoded ±1 blocks |
| **H-B1** | exact LUT hot scan | per-byte LUT gather and sum (numpy), then MaxSim | LUT [tokens, 16, 256] + accumulators |
| **H-B2** | binarized-query hot scan | XOR + `np.bitwise_count` on uint64 words (numpy), then MaxSim | integer accumulators |

- **R-bdec** is a control. It separates the effect of batching from the effect of
  native scoring, and is not a "direct" path.
- The torch kernel is the only mixed float × int8 matrix kernel available on this
  machine: no numba and no compiler. Small tiles are converted inside the kernel,
  but no decoded matrix is materialized.
- Plain numpy `x @ q.astype(float32)` would materialize float32 codes and is
  therefore **not** used as a direct path.

## 4. Correctness criteria (checked before any timing)

**Exact paths:**
- **I8-dir against I8-cur.** For every query, the full ranking of all 500 pages under
  `rank()` must be identical. Also report top-k agreement (k = 1, 5, 10, 50),
  nDCG@5 and retention.
- **Two-tier, R-dir and R-bdec against R-cur.** With the same hot tier (H-cur), the
  final ranking of every query must be identical.
- **H-B1 against H-cur.** The 50-page shortlist must be identical for every query,
  and so must the final two-tier ranking after R-dir rescoring.

**If an exact path fails equality: STOP.**
- Optimize nothing; no timing of that path.
- Diagnose each discordant pair. Compute the float64 reference of both scores, and
  the gap relative to float32's unit roundoff (u = 2^−24) times Σ|terms|.
- If every discordance is a tie at rounding level, that is reported as outcome C:
  exact equality is impossible in float arithmetic, with the precise reason.
- **The criterion is not relaxed in-run.** Continuing to latency after such a
  finding needs the reviewer's decision.

**Approximate path (B2):** no equality criterion. Report:
- shortlist overlap with H-cur, |S_B2 ∩ S_cur| / 50;
- recall of the labelled page in the shortlist (B2 against H-cur);
- top-5 and top-10 overlap of the final ranking, against current two-tier;
- nDCG@5 and retention, with the paired 95% interval of (R_B2 − R_cur).

## 5. Latency protocol (Q4's, with the warm-up rule stated now)

- **Machine:** the same laptop as Q4 (Intel 11th-gen mobile, 8 logical CPUs, Windows
  11), Python 3.11, numpy 2.4.6 with OpenBLAS 0.3.31, torch 2.11.0+cpu. Default
  threads: OpenBLAS as configured; torch reports 4. Thread counts are recorded.
  Nothing else runs during timing.
- **Unit:** one timed call scores all 451 queries end to end, reported as ms per
  query.
- **Warm-up rule (fixed now):** every configuration runs once, untimed, before the
  first timed round. That run is discarded and its time recorded separately. Then 5
  timed rounds, round-robin over configurations with a rotating start. Report the
  median and min–max of the 5 timed rounds. Per-query p95 is not available: the Q4
  protocol times whole batches.
- **End-to-end two-tier** time covers:
  1. query preparation (projection: none; LUT building for B1; binarization for B2);
  2. the hot scan;
  3. top-50 selection;
  4. rescoring;
  5. the final ranking (top-10 per query).

  Component times are reported too.
- **Full-scan int8** (I8-cur against I8-dir) and float32 (F32) are timed the same way.

## 6. Resource accounting

- **Resident RAM:**
  - binary hot codes (16 bytes per vector) plus offsets;
  - int8 cold codes (130 bytes per vector) plus offsets;
  - float32 for F32.

  MEASURED from array sizes.
- **Temporary buffers:**
  - numpy paths: peak traced allocation during one full run (`tracemalloc`, which
    numpy reports to), MEASURED;
  - torch paths: DERIVED from tensor shapes, since torch allocations are not
    traced;
  - the library's 256 MiB block cap is noted where it applies.
- **Disk:** MEASURED `save_compressed` file sizes for the binary and int8 stores
  (both persistable).

## 7. Hypotheses (recorded before running)

- **H5a.** Direct int8 scoring preserves the existing retrieval ranking closely
  enough to be practically equivalent, and removes the decode overhead.
  - **Supported** if I8-dir (and R-dir) reproduce the current rankings exactly
    (§4), **and** I8-dir's median scan time is below I8-cur's by more than the
    larger min–max spread.
  - **Rejected** if rankings differ beyond rounding-level ties, or I8-dir is
    slower.
  - Otherwise, mixed.
- **H5b.** Native binary scoring (H-B1) cuts first-stage latency substantially
  against H-cur: median ≤ 0.5x H-cur's.
- **H5c.** The native two-tier pipeline (H-B1 → R-dir, the exact path) cuts the
  **total** end-to-end latency against the current two-tier pipeline (H-cur →
  R-cur, `TieredIndex`).
  - The median must be lower beyond the larger spread, and nDCG@5 retention against
    float32 must stay ≥ 0.95 (the established target).
- **H5d.** Binarized-query scoring (B2) introduces measurable retrieval loss.
  - **Supported** if the paired 95% interval of R(H-B2 → R-dir) − R(H-cur → R-cur)
    lies below 0, **or** the labelled page's shortlist recall is lower than with
    H-cur.
  - Otherwise, no measurable loss at n = 451. It is still an approximate path.

## 8. Stop conditions (from the brief)

Stop and report if any of these happen:
- the exact int8 ranking is not reproduced;
- binary scoring changes semantics unexpectedly;
- no fair latency comparison is possible;
- a library change is needed;
- hardware differs from Q4;
- a result contradicts Q4 without an identifiable cause.

For example, R-cur must reproduce Q4's two-tier retention (0.9997 for Ward 1/4 >
binary → int8, 50) and roughly its rescoring cost (12.4 ms/query).

## 9. Labels

**MEASURED** (timed or computed here) · **DERIVED** (from shapes or array sizes) ·
**INFERENCE** (our reading).
