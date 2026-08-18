# Improvements

Five changes made after a review of the working pipeline. Every claim below was
measured on this laptop (16 GB, no GPU, ColSmol-256M), not estimated.

These are memory, fidelity and correctness fixes, not attempts to move nDCG.
Their effect on the retrieval table is recorded in [RESULTS.md](RESULTS.md).

---

## 1. Search no longer re-expands the index it just compressed

**The problem.** `NumpyIndex.score_all` decoded every packed code into one
float32 matrix and cached it. An index storing 16 bytes per vector therefore
cost `dim * 4` = 512 bytes per vector to *query* — a flat 32x, exactly the
saving the pipeline exists to produce.

Measured on the 60-page corpus: 2.82 KB/page on disk, 90.12 KB/page in RAM.
Extrapolated to the million-page archive the README talks about, the index is
2.82 GB and searching it wants **90 GB** of RAM. The storage claim was true;
the system-level claim behind it was not.

**The fix.** Below `max_decoded_bytes` (default 256 MB) the full expansion is
still cached, so laptop-scale corpora keep the single-matmul fast path. Above
it, pages are scored in blocks sized against the whole working set — the float32
vectors, the uint8 bit expansion, and the similarity rows — and each block is
released before the next is allocated. The arithmetic is unchanged.

Measured on a 4,000-page index (1M vectors, 16 MB packed):

| | peak RSS while scoring |
|---|---|
| before | 640 MB |
| after, 256 MB budget | 239 MB |
| after, 64 MB budget | 60 MB |
| after, 32 MB budget | 30 MB |

Peak now tracks the budget (~0.93x) and is **independent of corpus size**.
Scores are bit-identical to the old path — `test_streaming_matches_cached_scores`
asserts exact equality, and three further tests cover block boundaries splitting
pages, ranking agreement, and the budget surviving save/load.

## 2. int8 quantization was spending a quarter of its range

These vectors are L2-normalised, so a component cannot approach 1.0 — the mass
spreads over `dim` dimensions and `|c|` concentrates near `1/sqrt(dim)`. On real
ColSmol output (dim 128) the mean `|component|` is 0.071 and the largest seen is
0.363. Quantizing with `round(v * 127)` — as if the data spanned [-1, 1] — sent
that largest component to level 46 and left roughly two thirds of the 255
available levels permanently unused.

Rescaling by the range the data actually occupies costs nothing: no extra byte is
stored, so int8 remains exactly 4x.

| | 1 − cosine, real vectors |
|---|---|
| `round(v * 127)` | 3.28e-4 |
| `INT8_SCALE = 0.50` | 8.20e-5 |

0.5 rather than the observed maximum because the cost is asymmetric: a scale
slightly too large loses a little resolution, one slightly too small *clips* the
largest and most informative components. At 0.5 (~5.7σ for unit-norm 128-d
vectors) worst-case error over 5,000 random unit vectors is 0.0020, against
0.0125 at scale 0.4 where the tail does clip.

**Expect little or no movement in the retrieval table.** int8's reconstruction
error was already far below the score gaps between pages, so 4x less of it has
little room to change a ranking at 60 pages. This is headroom for harder corpora
and shorter vectors rather than a quality win, and the measured effect is
reported in [RESULTS.md](RESULTS.md).

> **Rebuild required.** An int8 index written before this change decodes at the
> old scale. Binary and float32 indexes are unaffected.

## 3. The hottest decode path allocated three copies of its own output

`unpack_signs` built the result with `bits.astype(np.float32) * 2.0 - 1.0`,
which allocates the full float32 array once for the cast and once per arithmetic
step. This runs on whole index blocks, so on a block sized to a memory budget the
true peak came out at ~3x that budget. Replaced with a two-entry lookup table,
which produces the result in a single allocation; `decode_int8` scales in place
for the same reason. This is what took improvement 1's measured peak from
1152 MB down to 640 MB before block sizing was even applied.

## 4. Qdrant reported an empty index after reopening

`QdrantIndex` filled `_page_stats` as it wrote, but a process that merely
*opened* an existing collection had never seen those pages — so every storage
figure read as zero:

```
after build   : pages=5 index_bytes=240 compression=2666.7x
after REOPEN  : pages=5 index_bytes=0   compression=0.0x     <- before
after REOPEN  : pages=5 index_bytes=240 compression=2666.7x  <- after
```

This matters because Qdrant is the backend the project abstract names as the
storage layer, and "compression 0.0x" is exactly the number a reviewer would
ask about. The figures were already on each point's payload, so they are read
back from the collection rather than kept in a sidecar file that could drift
out of sync.

## 5. Kendall tau rebuilt a set once per element

`rank_correlation` computed `[x for x in a if x in set(b)]`, which re-evaluates
`set(b)` for every element of `a`. Hoisted. It runs once per query per variant
across the whole ablation sweep.

---

## What was looked at and deliberately left alone

- **The saliency, spatial and redundancy stages.** The pruning maths is sound and
  the ablation already separates the two stages honestly.
- **The `keep_ratio` path ignoring `dilate`.** Documented and correct — dilating
  after a top-k selection would break the budget it exists to enforce.
- **`recall_at_k` dividing by `min(len(relevant), k)`.** Non-standard but
  deliberate and documented.
- **The synthetic encoder's saturating metrics.** Already carries a warning
  against reporting them.
