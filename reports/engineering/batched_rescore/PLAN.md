# Batched `ExactIndex.rescore`: validation plan (frozen before the change is measured)

This is an engineering follow-up to Q5/Q6 (`reports/research/Q5Q6/REPORT.md`,
`33b04d3`). The approved scope is one library change: batched execution of the
existing decode-based rescoring. Q5/Q6's research conclusions are not touched.
Nothing is merged to `main`.

## 1. What changes

- `TieredIndex.score` calls `self.cold.rescore(queries, shortlist)`, which is
  `ExactIndex.rescore`. Only the body of `ExactIndex.rescore` changes. Its signature,
  return shape and dtype, and `TieredIndex`, are unchanged.
- **Old body:** for each query,
  1. decode its m candidate pages and concatenate them;
  2. `qv @ block.T`, then the max per page (`segment_reduce`);
  3. `per_token.sum(axis=0)`.
- **New body:** for each distinct candidate page,
  1. decode it once;
  2. score every query token whose query listed that page, with one GEMM;
  3. take the max over the page's vectors, then sum per query.
- **Same arithmetic:** the same decode (`store.decode_rows`), the same float32
  products, the same max, and per-query sums over the same tokens.
- **Preserved:**
  - **slot semantics:** output column j is candidate column j, duplicates included;
  - **edge cases:** an empty query gives 0.0, and a page with no vectors gives −inf;
  - **query transform:** applied as before;
  - **memory:** the similarity block is capped at `max_block_bytes`, by chunking
    tokens.
- **Not introduced:** native int8 scoring, binary approximations, or any change to
  candidate generation, compression format or public API.
- **Batch size:** there is no new parameter. The only size control is the existing
  `max_block_bytes` (default 256 MiB), which is not tuned.

## 2. Correctness criterion (frozen now)

**Data and configurations.**
- The six Q4 datasets: ColPali, ColQwen2 and ColQwen2.5 × DocVQA and InfoVQA, with
  the full query pools of 451 and 494 queries.
- Cold tier: `Int8Quantizer("per_vector")` of all vectors.
- Hot tiers: Q4's two rows, `HierarchicalMerge(0.25) > BinaryQuantizer` and
  `BinaryQuantizer`.
- Candidate counts:
  - **50**, the Q4 / R12 policy, used for timing;
  - **1, 10, 200 and 500** (500 = every page), for correctness only.

**Requirements.** For every dataset, hot tier, candidate count and query, the old
rescoring (an exact copy of the previous method body, kept in the validation script
and in the tests) and the new library rescoring must give:
1. identical shortlisted page IDs (the hot tier is unchanged, so this checks the
   harness);
2. identical final rankings of all pages from `TieredIndex.score`;
3. identical per-query nDCG@5.

Bitwise equality of the rescored scores is reported as a secondary result.

**Persistence.** For each dataset, the int8 cold store is written with `NpzStorage`
and read back. Rescoring from the reloaded store must equal rescoring from the
in-memory store bitwise.

**If any ranking or nDCG@5 difference appears, timing does not run, and the change
is diagnosed and contained.** The Q5/Q6 amended (rounding-aware) criterion is
**not** used: this change is meant to be execution-equivalent.

## 3. Timing (the Q4 / Q5/Q6 protocol, unchanged)

- **Machine:** the same laptop and environment (Python 3.11, numpy 2.4.6 with
  OpenBLAS 0.3.31). Nothing else runs.
- **Per dataset:** candidates = 50, hot tier Ward 1/4 > binary (the Q5/Q6 primary
  configuration).
- **Four configurations per dataset:**
  - old rescore and new rescore, timed alone on the same shortlist;
  - old two-tier end to end (`TieredIndex` with the old rescore) and new two-tier
    end to end (`TieredIndex` with the library rescore).

  End to end covers the hot scan, top-50 selection, rescoring, merge and the final
  top-10.
- **Rounds:** one untimed warm-up run of every configuration, then 5 timed rounds
  over the whole query pool, round-robin with a rotating start.
- **Reported:** median and min–max ms/query, plus the within-round ratio of old to
  new.
- **Memory:** peak traced allocation (`tracemalloc`) for one old and one new rescore
  call.
- The datasets run one after another in a single process. Each is loaded, timed and
  released before the next.

## 4. Decision rules (from the brief)

- **A.** All six are ranking-identical, and end-to-end latency improves or holds →
  keep the change, documented as an engineering optimization.
- **B.** Correct, but some dataset regresses materially (new end-to-end median above
  old by more than the larger min–max spread) → do not keep automatically; report
  and stop.
- **C.** Correctness fails → revert or contain the change; no timing.
