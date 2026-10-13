# Q5/Q6 report: scoring compressed codes directly

Branch `research-investigation`.
- Pre-registration: `PLAN.md` (`a8db31b`).
- Reviewed amendment to the exact-path criterion: `PLAN.md` §10 (`3a1e3dc`), adopted
  after the strict criterion failed and before any latency was measured.
- Correctness, stopped at the strict criterion: `CORRECTNESS.md` (`d0bbc34`).
- Numbers: `correctness.json`, `b1_diagnosis.json`, `latency.json`.
- Code: `scripts/research/q56_*.py`, outside the package. No library change, nothing
  merged.

Labels: **MEASURED** (timed or computed here) · **DERIVED** (from shapes or array
sizes) · **INFERENCE**.

**Setting.**
- ColQwen2 v1.0 on ViDoRe V1 DocVQA: 500 pages, 374,154 vectors of 128-d, 451
  queries (9,111 tokens).
- Two-tier configuration: Ward 1/4 > binary as the hot tier (97,683 vectors), per-vector
  int8 of all vectors as the cold tier, **50 candidates** (the Q4 / R12 policy, not
  tuned).
- Hardware: the Q4 laptop (Intel 11th-gen mobile, 8 logical CPUs). Software: Python
  3.11, numpy 2.4.6 (OpenBLAS 0.3.31), torch 2.11.0+cpu with 4 threads.
- Runs at commit `3a1e3dc`.

## 1. Exact int8 correctness

- **Direct path:** float32 query tokens against the stored int8 codes, in
  `torch.ops.aten._weight_int8pack_mm(x, q, c)`. The per-vector multiplier is
  c = float32(scale) / 127, the library's own decode multiplier. **No decoded float32
  document matrix is formed.**
- **Arithmetic:**
  - current score = Σ_j x_j · fl32(q_j · c);
  - direct score = c · fl32(Σ_j x_j q_j).
- They are equal in exact arithmetic and round differently in float32 (`PLAN.md` §2).

**Strict criterion (identical full rankings for every query): FAILED (MEASURED).**
- Full int8 scan: 407 of 451 queries.
- Two-tier rescoring: 450 of 451.
- **All 45 discordances are float32 near-ties.**
  - Each pair's float64 gap is ≤ 2.7e−6, against an error bound of ≥ 1.7e−4.
  - They appear at rank 35 or deeper.
  - The float64 reference sides with the current path in 24 of 44 cases, so neither
    path is more accurate.
- Bit-identical tie-breaking is impossible without reproducing the decode rounding
  itself (`CORRECTNESS.md` §2).

**Amended criterion (reviewer decision, `PLAN.md` §10): PASSED (MEASURED).**
- Top-50 identical for every query (one order swap within one query's top-50).
- Per-query nDCG@5 identical.
- Every discordance is within the rounding bound.

## 2. Ranking agreement (MEASURED)

| | full ranking identical | top-1 / 5 / 10 identical | top-50 identical (set) | maximum score difference |
|---|---|---|---|---|
| direct int8 full scan against current | 407 / 451 | all 451 | 451 | 5.7e−6 (3.6e−7 relative) |
| direct int8 rescoring against current | 450 / 451 | all 451 | 451 | 1.5e−5 |
| control: batched decode-first rescoring | **451 / 451** | all 451 | 451 | 1.5e−5 |
| exact binary LUT hot stage against current | 407 / 451 | 451 / 450 / 450 (order) | 451 (shortlist set) | 4.6e−5 |
| exact binary LUT → direct int8, final | 406 / 451 | all 451 | 451 | – |

## 3. nDCG@5 and retention (MEASURED; against float32, nDCG@5 0.6065)

| path | nDCG@5 | retention [95% interval] |
|---|---|---|
| int8 full scan: current = direct | 0.6070 | 1.0009 [0.9962, 1.0058] |
| two-tier: current, direct rescoring, batched control, exact binary first stage | 0.6063 | 0.9997 [0.9937, 1.0056] (identical per query) |
| two-tier with binarized-query first stage (APPROXIMATE) | 0.6050 | 0.9976 [0.9885, 1.0061] |

## 4. Current against direct int8 latency (MEASURED)

**Protocol.**
- 451 queries per timed call, 5 timed rounds, round-robin with a rotating start.
- One untimed warm-up run of every configuration first, discarded by the rule fixed in
  `PLAN.md` §5.
- Each per-round value is the batch wall time divided by 451, in ms/query. Medians
  and min–max come from the 5 timed rounds; there is no per-query p95, because the
  protocol times batches.

**Machine-state note.**
- Rounds 1–4 ran about 1.35x slower than round 5 for **every** configuration
  together. For example, float32: 21.6, 19.4, 21.2, 23.3, then 15.4 ms.
- Round 5 reproduces Q4's values: float32 15.4 against Q4's 15.7 ms, and the current
  hot stage 4.3 against 4.1.
- So the median absolute times are higher than Q4's because of a machine-wide state
  in rounds 1–4. INFERENCE: power or thermal state; not determined.
- **Within-round ratios are stable**, and they are what the conclusions rest on.

| configuration | median ms/query [min–max] | ratio to the current equivalent, per round |
|---|---|---|
| float32 full scan | 21.2 [15.4–23.3] | – |
| int8 full scan, current (decode-first) | 22.5 [15.6–25.4] | 1.01–1.19x float32 |
| **int8 full scan, direct** | **604 [404–688]** | **23.8–34.1x slower** than the current int8 scan |
| rescoring 50 candidates, current (per-query loop + decode) | 19.2 | – |
| **rescoring, direct (batched per page)** | **66.7** | about 3.5x slower |
| rescoring, control (batched per page, decode-first) | **3.6** | **about 5.3x faster** |

**INFERENCE: why the direct kernel loses on this machine.**
- The only mixed float × int8 matrix kernel available (torch's
  `_weight_int8pack_mm`, built for LLM weight-only quantization) is far slower here
  than OpenBLAS's float32 GEMM after decoding.
- Microbenchmarks before the plan showed the same thing: 2–30x slower at every
  shape.
- Decoding int8 to float32 is cheap next to a well-tuned GEMM. The kernel's lack of
  optimization for these shapes dominates.

## 5. Binary first stage, exact (B1; MEASURED)

- **Method:** per-byte lookup tables of ±x partial sums in numpy. This is the same
  function as the decoded ±1 GEMM.
- **Correctness:** shortlist identical (as a set) for all 451 queries. All 44
  full-ranking discordances are within rounding.
- **Latency:** 172 ms/query [123–208] against **5.8 [4.3–6.9]** for the current
  decode-first binary scan, **28–30x slower** in every round.
- **INFERENCE:** numpy cannot express the SIMD byte-shuffle kernel that makes LUT
  scoring fast in compiled code. Gathers cost more than a BLAS GEMM over decoded ±1.

## 6. Binarized-query first stage (B2; APPROXIMATE; MEASURED)

| | current hot stage | B2 (binarized query, XOR + popcount) |
|---|---|---|
| hot-stage latency | 5.8 ms/query | 26.0 [20.2–31.8], **4.4–4.7x slower** |
| hot-stage-only retention | 0.961 | 0.948 |
| labelled page in the 50-page shortlist | 82.9% | **81.2%** |
| shortlist overlap with current | – | 64.1% on average |
| final two-tier retention after int8 rescoring | 0.9997 | 0.9976; difference −0.0021 [−0.0092, +0.0023] |
| final top-1 / 5 / 10 identical to current | – | 424 / 301 / 171 of 451 |

Candidates passed to rescoring: 50 in every case. The count is fixed, so latency
and memory differences come from the scoring and not from the shortlist size.

## 7. Complete two-tier, end to end (MEASURED)

**Total** = query preparation + hot scan + top-50 selection + rescoring + merge and
final top-10.

| pipeline | total ms/query, median [min–max] | first stage | rescoring | ratio to current, per round |
|---|---|---|---|---|
| **current** (decode-first binary → per-query int8 rescoring) | **23.5 [16.5–27.9]** | 4.3 | 19.2 | 1 |
| **native exact** (binary LUT → direct int8) | 250 [174–270] | 181.3 | 63.3 | **9.7–11.7x slower** |
| current binary → direct int8 | 71.4 [47.4–86.3] | 4.8 | 66.7 | 2.6–3.5x slower |
| binarized query → direct int8 (APPROXIMATE) | 91.4 [63.0–111.3] | 24.2 | 67.2 | 3.3–4.6x slower |
| **control:** current binary → batched decode-first int8 | **8.6 [6.3–9.8]** | 4.9 | **3.6** | **0.31–0.46x (2.2–3.2x faster)** |

The float32 full scan is 21.2 ms/query in the same rounds. So the control two-tier
runs at **0.37–0.46 of a float32 scan** with identical rankings. The current
two-tier runs at 1.01–1.44 of one, which matches Q4's finding that it buys RAM, not
speed.

## 8. RAM and storage

| item | bytes | per page | label |
|---|---|---|---|
| float32 vectors + offsets | 191.6 MB | 383 KB | MEASURED (array sizes) |
| binary hot tier (Ward 1/4) codes + offsets | **1.57 MB** | 3.1 KB | MEASURED |
| int8 cold tier codes + offsets (library layout, float16 scale) | 48.6 MB | 97.3 KB | MEASURED |
| int8, research layout used by the direct path (float32 multiplier) | 49.4 MB (+2 bytes per vector) | 98.8 KB | MEASURED |
| disk: binary hot `.npz` | 1.57 MB | – | MEASURED (file) |
| disk: int8 cold `.npz` | 48.7 MB | – | MEASURED (file) |

Peak temporary allocation for one run over all queries:

| path | peak |
|---|---|
| current paths (library blocks capped at 256 MiB) | 266–272 MB (MEASURED, tracemalloc) |
| direct int8 full scan, numpy side | 3.6 MB (MEASURED) |
| direct int8 full scan, torch side | ≤ 134 MB per 1,024 × 32,768 similarity block (DERIVED) |
| B1 lookup tables + accumulators | 403 MB (MEASURED) |
| B2 XOR and popcount temporaries | 567 MB (MEASURED) |
| batched control | 272 MB, dominated by the hot scan's block (MEASURED) |
| direct rescoring, torch side | ≤ about 8 MB per page call (DERIVED) |

**Totals.**
- Resident two-tier with both tiers in memory: 50.2 MB, 3.8x less than float32.
- With the cold tier on disk: 1.57 MB resident, 122x less.
- The batched rescoring reads only the shortlisted pages. Disk-backed rescoring
  latency was **not** measured.

## 9. Hypotheses

- **H5a (direct int8 is equivalent and removes the decode overhead): REJECTED.**
  - Equivalence holds under the reviewed, rounding-aware criterion. It fails the
    original strict one.
  - The direct path is 24–34x slower on the full scan and about 3.5x slower for
    rescoring, so it does not remove overhead.
- **H5b (native binary first stage is substantially faster): REJECTED.**
  - The exact LUT is 28–30x slower than the current decode-first scan.
  - The binarized-query path is also 4.4–4.7x slower in numpy.
- **H5c (native two-tier cuts total latency at the quality target): REJECTED.**
  - Quality holds at 0.9997.
  - Total latency is 9.7–11.7x higher.
- **H5d (binarized queries introduce measurable loss): SUPPORTED, by the pre-stated
  rule.**
  - The labelled page is in the shortlist less often (81.2% against 82.9%).
  - The final retention difference is −0.2 points, with an interval including 0.
  - Shortlist overlap is only 64%.
  - It is a separate approximate trade-off, and it is not faster here either.

**Outcome (brief's classes):**
- **B for direct int8:** correct (amended criterion) but not faster; much slower on
  this stack.
- **C for the strict reading:** bit-exact equality is impossible without the decode
  rounding.
- **D does not apply:** binary direct scoring gave no first-stage acceleration in
  numpy.

**The most useful measured finding comes from the control.** Batching the rescoring
per page with the *existing* decode arithmetic makes the rescoring step 5.3x faster
and the whole two-tier query 2.2–3.2x faster. It keeps every ranking bit-identical
(451 / 451). Q4's two-tier latency penalty came from the per-query rescoring loop,
not from decoding.

## 10. Limitations

- **One model and dataset, one laptop CPU, numpy and torch only.** No numba, no
  compiler and no SIMD intrinsics, so no hand-written popcount or LUT kernel. The
  negative native-kernel results apply to this software stack, not to native scoring
  in general. INFERENCE: compiled int8 or LUT kernels are known to beat
  decode + GEMM in other systems; that was not testable here.
- **Machine-wide timing drift** between rounds (§4). Ratios are stable, but absolute
  medians are about 1.35x Q4's.
- **Thread counts differ:** torch uses 4 threads, OpenBLAS its default. Not
  equalized, because the plan kept defaults.
- **Batch timing only:** no per-query latency distribution and no p95.
- **The int8 cold tier was held in RAM.** Disk-backed rescoring was not timed.
- **The bit-exact criterion was amended by the reviewer after it failed.** Both
  results are reported.

## 11. Status

**Q5/Q6 is CLOSED.**
- The question "can compressed codes be scored directly, preserving quality and
  improving latency?" has a measured answer on this stack.
  - Quality: yes, up to float32 near-ties.
  - Latency: no. Both native paths are slower than decode + BLAS.
- The two-tier latency problem has a different, measured cause (per-query
  rescoring), and an exact fix exists outside the library (the batched control).

## 12. Recommended next experiment (not started)

- **Make batched per-page rescoring a candidate change for `TieredIndex.rescore`.**
  - It is exact (bit-identical rankings) and 5.3x faster at rescoring.
  - It needs a library change, so it needs your approval. It would be validated with
    the Q4 two-tier rows on all six datasets, with identical rankings required.
- **A native-kernel test only if a compiled toolchain becomes acceptable as a
  research dependency.** For example, numba or a small C extension for a SIMD LUT
  binary kernel and an int8 dot-product kernel. The same correctness criterion and
  timing protocol would apply.
