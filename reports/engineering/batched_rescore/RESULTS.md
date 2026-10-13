# Batched `ExactIndex.rescore`: six-dataset validation

An engineering follow-up to Q5/Q6, separate from it. The validation plan was
frozen in `PLAN.md` (`12628a7`) before the change was measured. The library change
is in `src/optivision/storage.py` (`ExactIndex.rescore`).

| Item | Location |
|---|---|
| Regression tests | `tests/test_storage.py` |
| Harness | `scripts/engineering/batched_rescore.py` |
| Numbers | `correctness.json`, `timing.json` |

All results are MEASURED on the Q4 / Q5/Q6 laptop (Intel 11th-gen mobile, 8 logical
CPUs; Python 3.11; numpy 2.4.6 with OpenBLAS 0.3.31). They come from the working
tree on top of `12628a7`, before the change was committed.

## 1. What changed

**Old body.** For each query:
1. decode all m candidate pages and concatenate them;
2. one product `qv @ block.T`;
3. the max per page;
4. the sum over the query's tokens.

**New body.** For each distinct candidate page:
1. decode the page once;
2. one product of the tokens of every query that listed it against the page, chunked
   to `max_block_bytes`;
3. the max over the page's vectors, written into a per-token × candidate-column table;
4. the per-query sum over the same tokens, using the same `sum(axis=0)` as before.

**Unchanged:**
- the decode, the float32 products and the max;
- column order, including duplicate candidates;
- empty query → 0, and page with no vectors → −inf;
- the query transform, dtype and shape;
- `TieredIndex`, candidate generation, the compression format and the public API.

There is no native int8 or binary scoring, and no new parameter. The block size is
the existing `max_block_bytes` (256 MiB), untuned.

**One measured limit on bit-equality.** BLAS uses a different kernel for products
with very few rows (1–3 here), so the last bit can depend on the matrix shape. The
old code had the same dependence for very short queries.
- Bit-identical scores are guaranteed only when the products have enough rows. They
  do in all six datasets (every query has at least 14 tokens), and every score
  there was bit-identical.
- The unit test that forces one-row products checks identical rankings and
  agreement within float32 tolerance instead.

## 2. Correctness (criterion frozen in `PLAN.md` §2)

| dataset | queries | cells passed (2 hot tiers × candidates 1 / 10 / 50 / 200 / 500) | rescored scores bitwise-equal | reload (NpzStorage) bitwise-equal |
|---|---|---|---|---|
| ColPali DocVQA | 451 | 10 / 10 | 10 / 10 | yes |
| ColPali InfoVQA | 494 | 10 / 10 | 10 / 10 | yes |
| ColQwen2 DocVQA | 451 | 10 / 10 | 10 / 10 | yes |
| ColQwen2 InfoVQA | 494 | 10 / 10 | 10 / 10 | yes |
| ColQwen2.5 DocVQA | 451 | 10 / 10 | 10 / 10 | yes |
| ColQwen2.5 InfoVQA | 494 | 10 / 10 | 10 / 10 | yes |

Every cell has, for every query:
- identical shortlisted page IDs;
- identical final rankings of all 500 pages;
- identical nDCG@5.

The full query pools were used, with no sampling.

**Regression tests** (`tests/test_storage.py`) compare against a verbatim copy of the
old body:
- bitwise equality for float32, int8 and binary stores at 1, 6 and 30 candidates;
- column order and duplicates;
- an empty query, a page with no vectors and an empty candidate set;
- forced chunking under a tiny block budget;
- query projection;
- an `NpzStorage` reload;
- `TieredIndex` rankings and nDCG@5 at 1, 5 and all candidates.

The full suite passes: 455 tests.

## 3. Timing (the Q4 / Q5/Q6 protocol; candidates = 50; hot tier Ward 1/4 > binary; cold tier per-vector int8)

**Protocol.**
- One untimed warm-up run of every configuration, then 5 timed rounds over the full
  query pool, round-robin with a rotating start.
- The batch size is the fixed existing block budget; nothing was tuned.
- The values are ms/query, as the median and min–max over the 5 rounds. The speedups
  are old/new ratios within each round, as a min–max range.

| dataset | old rescore | batched rescore | rescore speedup | old end-to-end | batched end-to-end | end-to-end speedup | ranking identical? |
|---|---|---|---|---|---|---|---|
| ColPali DocVQA | 66.4 [64.6–66.7] | 7.19 [7.15–7.21] | 8.99–9.34x | 80.3 [80.0–80.7] | 21.8 [21.7–22.0] | 3.63–3.72x | yes (451 / 451) |
| ColPali InfoVQA | 25.3 [25.0–25.7] | 3.80 [3.74–3.84] | 6.51–6.81x | 31.8 [31.7–32.8] | 9.70 [9.61–10.21] | 3.12–3.36x | yes (494 / 494) |
| ColQwen2 DocVQA | 44.4 [44.1–45.8] | 6.57 [6.35–6.59] | 6.72–7.02x | 55.3 [54.9–55.4] | 17.3 [17.2–17.5] | 3.16–3.20x | yes (451 / 451) |
| ColQwen2 InfoVQA | 15.9 [15.8–45.9] | 2.88 [2.71–7.20] | 5.51–6.41x | 57.4 [20.5–58.3] | 17.1 [7.1–19.4] | 2.88–3.40x | yes (494 / 494) |
| ColQwen2.5 DocVQA | 16.0 [15.7–16.3] | 2.38 [2.34–2.45] | 6.44–6.88x | 20.0 [19.6–20.2] | 6.21 [6.18–6.28] | 3.15–3.26x | yes (451 / 451) |
| ColQwen2.5 InfoVQA | 45.0 [44.2–45.1] | 7.13 [6.91–7.28] | 6.14–6.52x | 56.9 [56.5–57.7] | 19.4 [19.2–19.5] | 2.90–2.99x | yes (494 / 494) |

**Across the six datasets:**
- median rescore speedup 6.7x (range 5.5–9.3x);
- median end-to-end speedup 3.25x (range 2.9–3.7x);
- **no dataset regressed in any round.**

**Machine state (MEASURED; cause not determined).**
- The laptop ran in two speed states during this run, about 2.8x apart for every
  configuration.
- ColQwen2.5 DocVQA was timed entirely in the fast state. Its old two-tier at
  20.0 ms/query is in the range of Q5/Q6's fast-state timings.
- ColQwen2 InfoVQA switched state between rounds 3 and 4: old two-tier 57.4 → 20.5,
  new 17.1 → 7.1.
- The other four datasets ran in the slow state.
- **The old/new ratios are the same in both states** (ColQwen2 InfoVQA end to end: 2.96–3.39x in
  the slow rounds, 2.89x in the fast ones), so the speedups do not depend on it.
- Absolute milliseconds are not comparable across datasets in this table, nor with
  Q4.

**Memory (peak traced allocation of one rescore call over all queries, numpy):**

| | ColPali | ColQwen |
|---|---|---|
| old | 79.4 MB | 57.7–58.2 MB |
| batched | 17.8–20.5 MB | 9.9–13.9 MB |

- The old body concatenates about 37–52k decoded vectors per query.
- The new one decodes one page at a time and keeps one per-token × candidate table
  (tokens × 50 × 4 bytes, under 2 MB).

## 4. Decision: rule A, keep the change

All six datasets are ranking-identical (in fact bitwise-identical scores), and
end-to-end latency improves on every dataset and in every round. Recommendation:
**keep the batched `ExactIndex.rescore` in the library as an engineering
optimization.**

**Qualifications:**
- **Scope of the measurement.** The speedups were measured on one laptop CPU with
  numpy/OpenBLAS, on 500-page corpora, with 50 candidates and 451–494 queries scored
  as one batch. They are not claimed for other hardware, corpus sizes, candidate
  counts or single-query serving. With one query per call, each page is scored for
  one query only, so most of the batching gain would disappear. That case was not
  measured.
- **Bit-equality** holds whenever the products have enough rows (§1), as in every
  tested dataset.
- **The Q5/Q6 research conclusion is unchanged.** Native direct int8 and binary
  scoring preserved retrieval quality under the amended criterion, but was much
  slower on the tested software stack. This change does not score compressed codes
  natively; it batches the existing decode-based path. The two findings are
  separate, and this one does not overturn Q5/Q6.

Nothing is merged to `main`. Q7 and Q8 have not been started.
