# Q5/Q6 correctness: STOPPED at the exact-ranking criterion

Pre-registration: `PLAN.md` (`a8db31b`). Data: ColQwen2 v1.0, ViDoRe V1 DocVQA, 500
pages, 451 queries.

| Item | Artifact |
|---|---|
| Correctness results | `correctness.json` |
| B1 diagnosis | `b1_diagnosis.json` |
| Scripts | `scripts/research/q56_common.py`, `q56_correctness.py`, `q56_diagnose_b1.py` (outside the package) |

**Status.** The pre-registered primary criterion is identical full rankings for every
query. It **failed** for both exact direct paths. As `PLAN.md` §4 requires, nothing
was optimized and **no latency was measured**. Continuing needs the reviewer's
decision. Everything below is MEASURED unless marked otherwise.

## 1. Sanity checks (passed)

- The float32 baseline per-query nDCG@5 equals the E1 store exactly.
- The current two-tier path (`TieredIndex`, Ward 1/4 > binary → int8, 50 candidates)
  reproduces Q4's retention exactly: 0.99968 [0.99366, 1.00563].

## 2. Experiment A: direct int8 scoring

- **Direct path (I8-dir):** float32 query tokens against the stored int8 codes,
  using `torch.ops.aten._weight_int8pack_mm(x, q, c)`. The per-vector multiplier
  c = float32(scale) / 127 is applied inside the kernel, and no decoded float32
  document matrix is formed.
- **Current path (I8-cur):** the library's blockwise decode, then a float32 GEMM.

| check | full int8 scan (I8-dir against I8-cur) | two-tier rescoring (R-dir against R-cur) | control: batched decode-first (R-bdec against R-cur) |
|---|---|---|---|
| identical full ranking of all 500 pages | **407 / 451** | **450 / 451** | 451 / 451 |
| identical top-1 / 5 / 10 / 50 (ordered) | 451 / 451 / 451 / 451 | 451 / 451 / 451 / 450 | all 451 |
| per-query nDCG@5 identical | yes | yes | yes |
| retention against float32 | 1.00094 (both paths) | 0.99968 (both) | 0.99968 |
| maximum score difference | 5.7e−6 absolute (3.6e−7 relative) | 1.5e−5 | 1.5e−5 |
| bitwise-equal scores | no | no | no |
| where rankings first differ | ranks 66–488 | rank 35 (one query) | – |

**Diagnosis of every discordance.**
- There are 44 (full scan) and 1 (rescoring).
- For each one, both scores of the first swapped page pair were recomputed in
  float64 from the same int8 codes and multipliers.
- **All 45 are rounding-level near-ties.** The largest float64 gap between the two
  pages is 2.7e−6, against a deterministic float32 dot-product error bound of at
  least 1.7e−4 for those scores.
- In 18 of the 44 the *current* path produces an exact float32 tie, which `rank()`
  then breaks by page index. The direct path produces 22 such ties.
- The float64 reference orders the pair as the current path does in **24 of 44**
  cases and as the direct path does in 20. **Neither path is the more accurate
  one.**

**Precise reason (from the plan's §2 mathematics, confirmed here).**
- The current score is Σ_j x_j · fl32(q_j · c): every code is rounded to float32
  *after* multiplying by c.
- The direct score is c · fl32(Σ_j x_j q_j): the scale is applied once, after the
  dot product.
- In exact arithmetic they are equal. In float32 they round differently whenever c
  is not a power of two, which is always here. The BLAS and torch kernels also
  accumulate in different orders.
- So **bit-identical scores, and therefore identical tie-breaking among pages whose
  scores agree to about 1e−7 relative, are impossible for any formulation that does
  not reproduce the decode-then-multiply rounding**, which is the decode the
  experiment is meant to remove.
- Away from such near-ties the rankings agree everywhere: every top-50 is
  identical, and so is every per-query nDCG@5.

**The control (R-bdec) is identical on all 451 queries.** It batches the rescoring
per page but keeps the library's decode arithmetic. So the reorganization itself
(no per-query loop) preserves exact rankings; only the change of arithmetic does
not.

## 3. Experiment B1: exact float-query scoring on packed bits (LUT)

- **Method:** per-byte lookup tables of ±x partial sums. This is the same
  mathematical function as the decoded ±1 GEMM.
- **Hot ranking:**
  - 407 / 451 queries have an identical full ranking;
  - 44 discordances, first at rank 5 or deeper;
  - **all within float32 rounding;**
  - the float64 reference agrees with the current path in 22 of 44.
- **Shortlist:** the 50-page shortlist is the **identical set for all 451
  queries**, and identical in order for 449.
- **Final two-tier ranking** (B1 → direct int8 rescoring) against current:
  - 406 / 451 full rankings identical;
  - top-10 identical for all 451;
  - retention 0.99968, the same.
- The **same cause** applies: the LUT sums partial sums of 8 terms, a different
  rounding order from the GEMM.

## 4. Experiment B2: binarized query (APPROXIMATE; separate, no equality criterion)

| | current hot tier (decoded ±1, float query) | B2 hot tier (binarized query, Hamming) |
|---|---|---|
| hot-tier-only retention | 0.961 [0.934, 0.987] | 0.948 [0.915, 0.979] |
| labelled page inside the 50-page shortlist | 82.9% | 81.2% |
| mean shortlist overlap with current | – | 64.1% |

After int8 rescoring of the 50 candidates:

| | current two-tier | B2 two-tier |
|---|---|---|
| retention | 0.99968 | 0.99758 [0.98854, 1.00606] |
| difference (B2 − current), paired 95% interval | – | −0.0021 [−0.0092, +0.0023] |
| top-1 / top-5 / top-10 identical to current | – | 424 / 301 / 171 of 451 |

- Under the pre-stated H5d rule, the shortlist recall of the labelled page is lower
  (81.2% against 82.9%), which counts as measurable loss.
- The retention difference itself is not distinguishable from 0 at n = 451.
- The secondary configuration (plain binary hot tier, all vectors) shows the same
  pattern:
  - B1: shortlist set identical for all 451 queries, retention unchanged;
  - B2: retention 0.99596.

## 5. What was not done

- **No latency, RAM-during-scan or throughput measurements** (`q56_latency.py` is
  written but was not run), and no optimization.
- **H5a–H5d not evaluated** beyond what is shown above. H5d's quality part is
  computable now, but its latency context is not.

## 6. Decision needed (INFERENCE: the options)

1. **Keep the strict criterion.** Q5/Q6 closes with outcome **C**: exact direct
   scoring is not bit-equivalent under the current representation. The near-ties
   above are the precise and only reason.
2. **Adopt a rounding-aware criterion, explicitly as a reviewed amendment, then
   measure latency.**
   - Example criterion: identical top-50 for every query, identical per-query
     nDCG@5, and every full-ranking discordance shown to be a pair whose float64
     gap is below the float32 error bound.
   - Both exact paths meet that today.
   - The latency step would then run as pre-registered, with no other changes.
3. **Ask for a bit-exact variant.** The only route is to reproduce fl32(q_j · c)
   per component, which is the decode itself. It would be bit-exact by construction
   and would not test native scoring.
