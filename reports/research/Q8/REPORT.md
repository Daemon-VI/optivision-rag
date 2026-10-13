# Q8 report: how compression, storage and search cost scale with corpus size, and what can be projected

| Item | Location |
|---|---|
| Frozen plan | `PLAN.md` (`3a39ba3`), with a run-validity amendment in §9 (`ae67a46`) |
| Validation script | `scripts/research/q8_scaling.py` (`457fa3f` + keep-awake guard), output in `validation.json` |
| Projections | `scripts/research/q8_projections.py`, output in `projections.json` and `tables.md` |

No library change, no GPU and no new retrieval-quality measurement.

**Labels:**
- **MEASURED:** from artifacts in this repository.
- **PROJECTED:** arithmetic on measured primitives under the stated assumption.
- **INFERENCE:** our reading.
- **EXTERNAL:** none used.

## 1. Research question

How do OptiVision's measured compression, storage and search-cost relationships behave
as the corpus grows, and what can legitimately be projected to larger corpora?

## 2. Frozen assumptions (`PLAN.md`)

- **Scaled quantity:** the number of pages N, at 1,000, 10,000, 100,000 and 1,000,000,
  alongside the measured 500.
- **Held at measured values per configuration:** vectors per page, dimension, bytes per
  vector, query tokens, and 50 two-tier candidates.
- **Storage:** index bytes = codes + offsets (8 bytes per page) + shared state. The
  shared state (codec state and PCA basis) is fixed.
- **RAM:** practical limit assumed at **8 GB**, about half of this laptop's 15.8 GB
  (Intel i7-1165G7, 4 cores / 8 threads).
- **Configurations:** fixed before any projection (`PLAN.md` §2). There are 10 per
  128-d dataset plus two-tier, and 6 for ColEmbed 4B plus its two two-tier rows. They
  are kept separate per model and dataset.
- **Latency projections** apply only to configurations whose scan scaling the
  validation experiment found linear. None is made beyond the point where the index
  exceeds 8 GB, because there is no paging or disk model.

**Run validity.**
- The first validation run (2026-10-03) was **invalidated**: the laptop entered standby
  overnight mid-run.
- The rerun (2026-10-04 08:57–09:42) used a keep-awake request. The Windows System log
  shows **no** Kernel-Power or Power-Troubleshooter events in that window.

## 3. Measured primitives

**Existing artifacts (MEASURED):**
- **Per-page storage:** Q4's store gives actual vectors per page and code, offset and
  shared bytes for six 128-d datasets. R12 gives ColEmbed 4B's bytes per page.
- **CPU scan time at N = 500** (Q4 latency, DocVQA, laptop numpy/OpenBLAS):

  | configuration | ColPali | ColQwen2 | ColQwen2.5 |
  |---|---|---|---|
  | float32 | 22.2 | 15.7 | 15.8 |
  | Ward 1/4 | 6.2 | 4.3 | 4.2 |
  | Ward 1/4 > binary | 6.2 | 4.4 | 4.4 |

  ms per query, amortized over a batch of all queries.
- **Two-tier components** (Q5/Q6), **batched rescoring** (engineering follow-up) and
  **retention from 100 to 500 pages** (R10, quality context only).

**New: the validation experiment** (MEASURED wall time). ColQwen2 DocVQA tiled m = 1,
2, 4, 8 times (500–4,000 pages, identical duplicate pages), timing only.

**V1, scan time** (ms per query, median of 5 rounds after a warm-up):

| configuration | 500 | 1,000 | 2,000 | 4,000 | t(4,000) / 8·t(500) | R² of linear fit | slope (ms per query per million stored vectors) |
|---|---|---|---|---|---|---|---|
| float32 | 13.47 | 27.51 | 54.92 | 116.38 | 1.080 | 0.9992 | 39.3 |
| int8 (per vector) | 14.28 | 28.34 | 56.36 | 112.25 | 0.982 | 1.0000 | 37.4 |
| binary | 14.21 | 28.11 | 56.18 | 112.24 | 0.987 | 1.0000 | 37.4 |
| Ward 1/4 | 3.76 | 7.57 | 15.09 | 30.14 | 1.001 | 1.0000 | 38.5 |
| Ward 1/4 > binary | 3.87 | 7.67 | 15.38 | 30.65 | 0.991 | 1.0000 | 39.2 |
| PCA 64 | 11.45 | 22.85 | 45.98 | 92.06 | 1.005 | 1.0000 | 30.8 |

**V2, rescoring 50 candidates per query against candidate overlap.** Candidates are
random pages from the tiled corpus, standing in for falling overlap; they are not real
shortlists.

| corpus pages | distinct candidate pages in the 451-query batch | old per-query rescoring (ms/query) | batched rescoring (ms/query) | old / batched |
|---|---|---|---|---|
| 500 | 500 | 11.39 | 1.53 | 7.42x |
| 1,000 | 1,000 | 11.15 | 1.98 | 5.64x |
| 2,000 | 2,000 | 11.23 | 2.66 | 4.22x |
| 4,000 | 3,984 | 11.17 | 4.03 | 2.78x |

The outputs were bitwise identical between old and batched rescoring in every case.

## 4. Scaling relationships

| resource | class | justification |
|---|---|---|
| **A. Index storage** | linear in N, plus a fixed shared state | arithmetic: bytes per page are a measured constant per configuration; offsets add 8 bytes per page; the shared state (PCA basis, codec state) does not depend on N |
| **B. Resident RAM** | linear (= A for a single-tier index; = the hot tier for two-tier) | the current `ExactIndex` keeps the codes resident (Q4 §10) |
| **C. Full-scan compute** | **approximately linear in stored vectors** over 500–4,000 pages (MEASURED, V1: R² ≥ 0.999, ratios 0.98–1.08); **unknown** beyond 4,000 pages and beyond RAM | float32 shows a slight upturn at 4,000 pages (1.08); not tested at larger N |
| **D. Decode work** | linear in scanned vectors (part of C) | int8 and binary have the same per-vector slope as float32 (37.4 against 39.3); decoding to float32 keeps their cost per vector equal to float32 (Q4, Q5/Q6) |
| **E. First-stage binary work** | linear in hot vectors (MEASURED, V1, Ward 1/4 > binary) | same as C |
| **F. Two-tier rescoring** | **fixed per query** for the old per-query loop (MEASURED flat, 11.2 ms over 500–4,000 pages); **implementation-dependent** for batched rescoring (MEASURED growth with distinct candidate pages; **unknown** below the lowest overlap measured) | batched rescoring decodes each distinct candidate page once per batch, so its gain comes from candidate overlap, which falls as N grows |
| **G. Query-independent metadata** | offsets linear but negligible (8 bytes per page); shared state fixed | PCA basis: 32.8 KB (128 → 64), 6.55 MB (2,560 → 640) |
| **H. Temporary working memory** | scan block **fixed** (256 MiB cap; about 270 MB peak measured in Q5/Q6); output score matrix **linear in N × queries per batch** | `maxsim_matrix` and `TieredIndex.score` return a dense queries × pages float32 matrix: about 1.8 GB for 451 queries at 1M pages (PROJECTED), 4 MB for one query |

## 5. Storage projections (MEASURED at 500 pages; PROJECTED beyond: measured bytes per page × N + shared state)

#### colpali_docvqa (float32 527.9 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 1031.0 × 128 | 527.88 KB | 0.0 KB | 1.0x | 263.9 MB | 5.28 GB | 52.8 GB | 527.9 GB | 16,272 |
| float16 | 1031.0 × 128 | 263.94 KB | 0.0 KB | 2.0x | 132.0 MB | 2.64 GB | 26.4 GB | 263.9 GB | 32,544 |
| int8(per_vector) | 1031.0 × 128 | 134.04 KB | 0.0 KB | 3.9x | 67.0 MB | 1.34 GB | 13.4 GB | 134.0 GB | 64,085 |
| int4(mean) | 1031.0 × 128 | 68.05 KB | 0.5 KB | 7.8x | 34.0 MB | 0.68 GB | 6.8 GB | 68.1 GB | 126,222 |
| binary | 1031.0 × 128 | 16.50 KB | 0.0 KB | 32.0x | 8.3 MB | 0.17 GB | 1.7 GB | 16.5 GB | 520,475 |
| hierarchical_merge(0.25) | 263.0 × 128 | 134.66 KB | 0.0 KB | 3.9x | 67.3 MB | 1.35 GB | 13.5 GB | 134.7 GB | 63,787 |
| hierarchical_merge(0.25) > int8(per_vector) | 263.0 × 128 | 34.20 KB | 0.0 KB | 15.4x | 17.1 MB | 0.34 GB | 3.4 GB | 34.2 GB | 251,182 |
| hierarchical_merge(0.25) > binary | 263.0 × 128 | 4.22 KB | 0.0 KB | 125.4x | 2.1 MB | 0.04 GB | 0.4 GB | 4.2 GB | 2,037,453 |
| project(64) > int8(per_vector) | 1031.0 × 64 | 68.05 KB | 32.8 KB | 7.8x | 34.1 MB | 0.68 GB | 6.8 GB | 68.1 GB | 126,221 |
| hierarchical_merge(0.25) > project(64) > binary | 263.0 × 64 | 2.11 KB | 32.8 KB | 250.9x | 1.1 MB | 0.02 GB | 0.2 GB | 2.1 GB | 4,067,157 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 4.2 KB / disk 138.3 KB | 0 | – | RAM 2.1 MB / disk 69 MB | 0.042 / 1.38 GB | 0.42 / 13.8 GB | 4.2 / 138 GB | 2,037,453 |

#### colpali_infovqa (float32 527.9 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 1031.0 × 128 | 527.88 KB | 0.0 KB | 1.0x | 263.9 MB | 5.28 GB | 52.8 GB | 527.9 GB | 16,272 |
| float16 | 1031.0 × 128 | 263.94 KB | 0.0 KB | 2.0x | 132.0 MB | 2.64 GB | 26.4 GB | 263.9 GB | 32,544 |
| int8(per_vector) | 1031.0 × 128 | 134.04 KB | 0.0 KB | 3.9x | 67.0 MB | 1.34 GB | 13.4 GB | 134.0 GB | 64,085 |
| int4(mean) | 1031.0 × 128 | 68.05 KB | 0.5 KB | 7.8x | 34.0 MB | 0.68 GB | 6.8 GB | 68.1 GB | 126,222 |
| binary | 1031.0 × 128 | 16.50 KB | 0.0 KB | 32.0x | 8.3 MB | 0.17 GB | 1.7 GB | 16.5 GB | 520,475 |
| hierarchical_merge(0.25) | 263.0 × 128 | 134.66 KB | 0.0 KB | 3.9x | 67.3 MB | 1.35 GB | 13.5 GB | 134.7 GB | 63,787 |
| hierarchical_merge(0.25) > int8(per_vector) | 263.0 × 128 | 34.20 KB | 0.0 KB | 15.4x | 17.1 MB | 0.34 GB | 3.4 GB | 34.2 GB | 251,182 |
| hierarchical_merge(0.25) > binary | 263.0 × 128 | 4.22 KB | 0.0 KB | 125.4x | 2.1 MB | 0.04 GB | 0.4 GB | 4.2 GB | 2,037,453 |
| project(64) > int8(per_vector) | 1031.0 × 64 | 68.05 KB | 32.8 KB | 7.8x | 34.1 MB | 0.68 GB | 6.8 GB | 68.1 GB | 126,221 |
| hierarchical_merge(0.25) > project(64) > binary | 263.0 × 64 | 2.11 KB | 32.8 KB | 250.9x | 1.1 MB | 0.02 GB | 0.2 GB | 2.1 GB | 4,067,157 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 4.2 KB / disk 138.3 KB | 0 | – | RAM 2.1 MB / disk 69 MB | 0.042 / 1.38 GB | 0.42 / 13.8 GB | 4.2 / 138 GB | 2,037,453 |

#### colqwen2_docvqa (float32 383.1 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 748.3 × 128 | 383.14 KB | 0.0 KB | 1.0x | 191.6 MB | 3.83 GB | 38.3 GB | 383.1 GB | 22,419 |
| float16 | 748.3 × 128 | 191.57 KB | 0.0 KB | 2.0x | 95.8 MB | 1.92 GB | 19.2 GB | 191.6 GB | 44,838 |
| int8(per_vector) | 748.3 × 128 | 97.29 KB | 0.0 KB | 3.9x | 48.6 MB | 0.97 GB | 9.7 GB | 97.3 GB | 88,293 |
| int4(mean) | 748.3 × 128 | 49.40 KB | 0.5 KB | 7.8x | 24.7 MB | 0.49 GB | 4.9 GB | 49.4 GB | 173,898 |
| binary | 748.3 × 128 | 11.98 KB | 0.0 KB | 32.0x | 6.0 MB | 0.12 GB | 1.2 GB | 12.0 GB | 716,966 |
| hierarchical_merge(0.25) | 195.4 × 128 | 100.04 KB | 0.0 KB | 3.8x | 50.0 MB | 1.00 GB | 10.0 GB | 100.0 GB | 85,868 |
| hierarchical_merge(0.25) > int8(per_vector) | 195.4 × 128 | 25.41 KB | 0.0 KB | 15.1x | 12.7 MB | 0.25 GB | 2.5 GB | 25.4 GB | 338,111 |
| hierarchical_merge(0.25) > binary | 195.4 × 128 | 3.13 KB | 0.0 KB | 122.6x | 1.6 MB | 0.03 GB | 0.3 GB | 3.1 GB | 2,740,997 |
| project(64) > int8(per_vector) | 748.3 × 64 | 49.40 KB | 32.8 KB | 7.8x | 24.7 MB | 0.49 GB | 4.9 GB | 49.4 GB | 173,897 |
| hierarchical_merge(0.25) > project(64) > binary | 195.4 × 64 | 1.57 KB | 32.8 KB | 245.1x | 0.8 MB | 0.02 GB | 0.2 GB | 1.6 GB | 5,467,987 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 3.1 KB / disk 100.4 KB | 0 | – | RAM 1.6 MB / disk 50 MB | 0.031 / 1.00 GB | 0.31 / 10.0 GB | 3.1 / 100 GB | 2,740,997 |

#### colqwen2_infovqa (float32 375.0 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 732.5 × 128 | 375.03 KB | 0.0 KB | 1.0x | 187.5 MB | 3.75 GB | 37.5 GB | 375.0 GB | 22,904 |
| float16 | 732.5 × 128 | 187.52 KB | 0.0 KB | 2.0x | 93.8 MB | 1.88 GB | 18.8 GB | 187.5 GB | 45,807 |
| int8(per_vector) | 732.5 × 128 | 95.23 KB | 0.0 KB | 3.9x | 47.6 MB | 0.95 GB | 9.5 GB | 95.2 GB | 90,202 |
| int4(mean) | 732.5 × 128 | 48.35 KB | 0.5 KB | 7.8x | 24.2 MB | 0.48 GB | 4.8 GB | 48.4 GB | 177,657 |
| binary | 732.5 × 128 | 11.73 KB | 0.0 KB | 32.0x | 5.9 MB | 0.12 GB | 1.2 GB | 11.7 GB | 732,456 |
| hierarchical_merge(0.25) | 191.6 × 128 | 98.10 KB | 0.0 KB | 3.8x | 49.0 MB | 0.98 GB | 9.8 GB | 98.1 GB | 87,566 |
| hierarchical_merge(0.25) > int8(per_vector) | 191.6 × 128 | 24.91 KB | 0.0 KB | 15.1x | 12.5 MB | 0.25 GB | 2.5 GB | 24.9 GB | 344,795 |
| hierarchical_merge(0.25) > binary | 191.6 × 128 | 3.07 KB | 0.0 KB | 122.3x | 1.5 MB | 0.03 GB | 0.3 GB | 3.1 GB | 2,795,052 |
| project(64) > int8(per_vector) | 732.5 × 64 | 48.35 KB | 32.8 KB | 7.8x | 24.2 MB | 0.48 GB | 4.8 GB | 48.4 GB | 177,656 |
| hierarchical_merge(0.25) > project(64) > binary | 191.6 × 64 | 1.54 KB | 32.8 KB | 244.7x | 0.8 MB | 0.02 GB | 0.2 GB | 1.5 GB | 5,575,541 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 3.1 KB / disk 98.3 KB | 0 | – | RAM 1.5 MB / disk 49 MB | 0.031 / 0.98 GB | 0.31 / 9.8 GB | 3.1 / 98 GB | 2,795,052 |

#### colqwen25_docvqa (float32 383.1 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 748.3 × 128 | 383.14 KB | 0.0 KB | 1.0x | 191.6 MB | 3.83 GB | 38.3 GB | 383.1 GB | 22,419 |
| float16 | 748.3 × 128 | 191.57 KB | 0.0 KB | 2.0x | 95.8 MB | 1.92 GB | 19.2 GB | 191.6 GB | 44,838 |
| int8(per_vector) | 748.3 × 128 | 97.29 KB | 0.0 KB | 3.9x | 48.6 MB | 0.97 GB | 9.7 GB | 97.3 GB | 88,293 |
| int4(mean) | 748.3 × 128 | 49.40 KB | 0.5 KB | 7.8x | 24.7 MB | 0.49 GB | 4.9 GB | 49.4 GB | 173,898 |
| binary | 748.3 × 128 | 11.98 KB | 0.0 KB | 32.0x | 6.0 MB | 0.12 GB | 1.2 GB | 12.0 GB | 716,966 |
| hierarchical_merge(0.25) | 195.4 × 128 | 100.04 KB | 0.0 KB | 3.8x | 50.0 MB | 1.00 GB | 10.0 GB | 100.0 GB | 85,868 |
| hierarchical_merge(0.25) > int8(per_vector) | 195.4 × 128 | 25.41 KB | 0.0 KB | 15.1x | 12.7 MB | 0.25 GB | 2.5 GB | 25.4 GB | 338,111 |
| hierarchical_merge(0.25) > binary | 195.4 × 128 | 3.13 KB | 0.0 KB | 122.6x | 1.6 MB | 0.03 GB | 0.3 GB | 3.1 GB | 2,740,997 |
| project(64) > int8(per_vector) | 748.3 × 64 | 49.40 KB | 32.8 KB | 7.8x | 24.7 MB | 0.49 GB | 4.9 GB | 49.4 GB | 173,897 |
| hierarchical_merge(0.25) > project(64) > binary | 195.4 × 64 | 1.57 KB | 32.8 KB | 245.1x | 0.8 MB | 0.02 GB | 0.2 GB | 1.6 GB | 5,467,987 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 3.1 KB / disk 100.4 KB | 0 | – | RAM 1.6 MB / disk 50 MB | 0.031 / 1.00 GB | 0.31 / 10.0 GB | 3.1 / 100 GB | 2,740,997 |

#### colqwen25_infovqa (float32 375.0 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 732.5 × 128 | 375.03 KB | 0.0 KB | 1.0x | 187.5 MB | 3.75 GB | 37.5 GB | 375.0 GB | 22,904 |
| float16 | 732.5 × 128 | 187.52 KB | 0.0 KB | 2.0x | 93.8 MB | 1.88 GB | 18.8 GB | 187.5 GB | 45,807 |
| int8(per_vector) | 732.5 × 128 | 95.23 KB | 0.0 KB | 3.9x | 47.6 MB | 0.95 GB | 9.5 GB | 95.2 GB | 90,202 |
| int4(mean) | 732.5 × 128 | 48.35 KB | 0.5 KB | 7.8x | 24.2 MB | 0.48 GB | 4.8 GB | 48.4 GB | 177,657 |
| binary | 732.5 × 128 | 11.73 KB | 0.0 KB | 32.0x | 5.9 MB | 0.12 GB | 1.2 GB | 11.7 GB | 732,456 |
| hierarchical_merge(0.25) | 191.6 × 128 | 98.10 KB | 0.0 KB | 3.8x | 49.0 MB | 0.98 GB | 9.8 GB | 98.1 GB | 87,566 |
| hierarchical_merge(0.25) > int8(per_vector) | 191.6 × 128 | 24.91 KB | 0.0 KB | 15.1x | 12.5 MB | 0.25 GB | 2.5 GB | 24.9 GB | 344,795 |
| hierarchical_merge(0.25) > binary | 191.6 × 128 | 3.07 KB | 0.0 KB | 122.3x | 1.5 MB | 0.03 GB | 0.3 GB | 3.1 GB | 2,795,052 |
| project(64) > int8(per_vector) | 732.5 × 64 | 48.35 KB | 32.8 KB | 7.8x | 24.2 MB | 0.48 GB | 4.8 GB | 48.4 GB | 177,656 |
| hierarchical_merge(0.25) > project(64) > binary | 191.6 × 64 | 1.54 KB | 32.8 KB | 244.7x | 0.8 MB | 0.02 GB | 0.2 GB | 1.5 GB | 5,575,541 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 3.1 KB / disk 98.3 KB | 0 | – | RAM 1.5 MB / disk 49 MB | 0.031 / 0.98 GB | 0.31 / 9.8 GB | 3.1 / 98 GB | 2,795,052 |

#### ColEmbed 4B docvqa (float32 7.77 MB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor (codes) | 500 | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 758.3 × 2560 | 7765.1 KB | 0.00 MB | 1.0x | 3.883 GB | 77.65 GB | 776.5 GB | 7765 GB | 1,106 |
| int8(per_vector) | 758.3 × 2560 | 1942.8 KB | 0.00 MB | 4.0x | 0.971 GB | 19.43 GB | 194.3 GB | 1943 GB | 4,421 |
| binary | 758.3 × 2560 | 242.7 KB | 0.00 MB | 32.0x | 0.121 GB | 2.43 GB | 24.3 GB | 243 GB | 35,398 |
| hierarchical_merge(0.25) > int8(per_vector) | 190.1 × 2560 | 487.0 KB | 0.00 MB | 15.9x | 0.243 GB | 4.87 GB | 48.7 GB | 487 GB | 17,639 |
| project(640) > int8(per_vector) | 758.3 × 640 | 486.8 KB | 6.55 MB | 16.0x | 0.250 GB | 4.87 GB | 48.7 GB | 487 GB | 17,630 |
| hierarchical_merge(0.25) > project(640) > binary | 190.1 × 640 | 15.2 KB | 6.55 MB | 510.7x | 0.014 GB | 0.16 GB | 1.5 GB | 15 GB | 564,191 |
| two-tier (binary hot, int8 cold) | – | RAM 242.7 KB / disk 2.19 MB | 0 | – | RAM 121 MB / disk 1.09 GB | 2.43 / 21.9 GB | 24.3 / 219 GB | 243 / 2185 GB | 35,399 |
| two-tier (ward 1/4 > binary hot, int8 cold) | – | RAM 60.8 KB / disk 2.00 MB | 0 | – | RAM 30 MB / disk 1.00 GB | 0.61 / 20.0 GB | 6.1 / 200 GB | 61 / 2004 GB | 141,229 |

#### ColEmbed 4B infovqa (float32 7.56 MB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor (codes) | 500 | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 738.0 × 2560 | 7556.8 KB | 0.00 MB | 1.0x | 3.778 GB | 75.57 GB | 755.7 GB | 7557 GB | 1,136 |
| int8(per_vector) | 738.0 × 2560 | 1890.7 KB | 0.00 MB | 4.0x | 0.945 GB | 18.91 GB | 189.1 GB | 1891 GB | 4,543 |
| binary | 738.0 × 2560 | 236.2 KB | 0.00 MB | 32.0x | 0.118 GB | 2.36 GB | 23.6 GB | 236 GB | 36,373 |
| hierarchical_merge(0.25) > int8(per_vector) | 184.9 × 2560 | 473.8 KB | 0.00 MB | 16.0x | 0.237 GB | 4.74 GB | 47.4 GB | 474 GB | 18,131 |
| project(640) > int8(per_vector) | 738.0 × 640 | 473.8 KB | 6.55 MB | 16.0x | 0.243 GB | 4.74 GB | 47.4 GB | 474 GB | 18,116 |
| hierarchical_merge(0.25) > project(640) > binary | 184.9 × 640 | 14.8 KB | 6.55 MB | 510.8x | 0.014 GB | 0.15 GB | 1.5 GB | 15 GB | 579,901 |
| two-tier (binary hot, int8 cold) | – | RAM 236.1 KB / disk 2.13 MB | 0 | – | RAM 118 MB / disk 1.06 GB | 2.36 / 21.3 GB | 23.6 / 213 GB | 236 / 2127 GB | 36,374 |
| two-tier (ward 1/4 > binary hot, int8 cold) | – | RAM 59.2 KB / disk 1.95 MB | 0 | – | RAM 30 MB / disk 0.97 GB | 0.59 / 19.5 GB | 5.9 / 195 GB | 59 / 1950 GB | 145,164 |

#### Scan latency, laptop CPU (MEASURED at 500 pages; PROJECTED beyond under the V1 linearity assumption)

| model | configuration | 500 MEASURED ms/query | 10k PROJECTED | 100k PROJECTED | 1M PROJECTED |
|---|---|---|---|---|---|
| ColPali | float32 | 22.23 | 516 ms | not projected (index 53 GB > 8 GB) | not projected (index 528 GB > 8 GB) |
| ColPali | int8(per_vector) | 22.75 | 445 ms | not projected (index 13 GB > 8 GB) | not projected (index 134 GB > 8 GB) |
| ColPali | binary | 22.66 | 449 ms | 4.49 s | not projected (index 17 GB > 8 GB) |
| ColPali | hierarchical_merge(0.25) | 6.22 | 124 ms | not projected (index 13 GB > 8 GB) | not projected (index 135 GB > 8 GB) |
| ColPali | hierarchical_merge(0.25) > binary | 6.18 | 123 ms | 1.23 s | 12.25 s |
| ColPali | project(64) | 17.84 | 361 ms | not projected (index 26 GB > 8 GB) | not projected (index 264 GB > 8 GB) |
| ColQwen2 | float32 | 15.69 | 364 ms | not projected (index 38 GB > 8 GB) | not projected (index 383 GB > 8 GB) |
| ColQwen2 | int8(per_vector) | 15.91 | 311 ms | not projected (index 10 GB > 8 GB) | not projected (index 97 GB > 8 GB) |
| ColQwen2 | binary | 15.99 | 317 ms | 3.17 s | not projected (index 12 GB > 8 GB) |
| ColQwen2 | hierarchical_merge(0.25) | 4.27 | 85 ms | not projected (index 10 GB > 8 GB) | not projected (index 100 GB > 8 GB) |
| ColQwen2 | hierarchical_merge(0.25) > binary | 4.35 | 86 ms | 863 ms | 8.62 s |
| ColQwen2 | project(64) | 12.40 | 251 ms | not projected (index 19 GB > 8 GB) | not projected (index 192 GB > 8 GB) |
| ColQwen2.5 | float32 | 15.79 | 366 ms | not projected (index 38 GB > 8 GB) | not projected (index 383 GB > 8 GB) |
| ColQwen2.5 | int8(per_vector) | 16.06 | 314 ms | not projected (index 10 GB > 8 GB) | not projected (index 97 GB > 8 GB) |
| ColQwen2.5 | binary | 16.00 | 317 ms | 3.17 s | not projected (index 12 GB > 8 GB) |
| ColQwen2.5 | hierarchical_merge(0.25) | 4.22 | 84 ms | not projected (index 10 GB > 8 GB) | not projected (index 100 GB > 8 GB) |
| ColQwen2.5 | hierarchical_merge(0.25) > binary | 4.39 | 87 ms | 870 ms | 8.70 s |
| ColQwen2.5 | project(64) | 12.39 | 250 ms | not projected (index 19 GB > 8 GB) | not projected (index 192 GB > 8 GB) |

#### Two-tier, ColQwen2 DocVQA (hot Ward 1/4 > binary PROJECTED linear; rescoring MEASURED)

| pages | hot stage ms/query (PROJECTED) | old rescoring ms/query (MEASURED, flat 500-4,000) | total with old rescoring | hot share |
|---|---|---|---|---|
| 500 | 4.4 | 11.2 | 15.6 | 28% |
| 1,000 | 8.7 | 11.2 | 19.9 | 44% |
| 10,000 | 86.3 | 11.2 | 97.5 | 88% |
| 100,000 | 862.5 | 11.2 | 873.8 | 99% |
| 1,000,000 | 8625.0 | 11.2 | 8636.2 | 100% |

## 6. RAM projections

**Single-tier indexes.** In the current implementation, RAM is the index size in §5.
The last column of each table gives the number of pages that fit within the 8 GB
assumption (PROJECTED).

**Examples:**

| dataset | float32 | int8 | binary | Ward 1/4 > binary | Ward 1/4 > PCA 64 > binary |
|---|---|---|---|---|---|
| ColQwen2 DocVQA | 22.4k | 88.3k | 717k | 2.74M | 5.47M |
| ColPali | 16.3k | 64.1k | 520k | 2.04M | 4.07M |

**Two-tier.** RAM holds only the hot tier.
- 128-d, Ward 1/4 > binary: 3.1–4.2 KB per page, so 2.0–2.8M pages within 8 GB.
- The int8 cold tier is on disk: 98–138 KB per page, about 98–138 GB at 1M pages
  (PROJECTED).

## 7. Latency and search-cost projections

**Assumptions** (PROJECTED from the measured N = 500 value):
- linear in stored vectors, as V1 measured up to 4,000 pages;
- the same vectors per page;
- the index resident within 8 GB;
- a batch of all queries per call.

These cells extrapolate up to 250 times beyond the measured 4,000 pages, so they are
arithmetic, not measurements. The scan-latency table at the end of §5 gives the values.

**Examples (ColQwen2 DocVQA):**

| configuration | 10k pages | 100k pages | 1M pages |
|---|---|---|---|
| float32 | 364 ms | not projected (38 GB index) | not projected |
| binary | 317 ms | 3.2 s | not projected (12 GB index) |
| Ward 1/4 > binary | 86 ms | 863 ms | 8.6 s |

ColPali and ColQwen2.5 behave the same way, scaled by their vectors per page.

## 8. Two-tier projections (ColQwen2 DocVQA; hot stage PROJECTED; rescoring MEASURED)

See the two-tier table at the end of §5.
- **The hot first stage equals the measured old rescoring cost (11.2 ms) at about
  1,300 pages** (PROJECTED). Beyond that, the exhaustive hot scan dominates:
  - 88% of the total at 10,000 pages;
  - 99% at 100,000;
  - about 8.6 s per query at 1M pages.
- **Two-tier lowers RAM by about 30x against int8, but it does not change the scaling
  class.** Search stays linear in N because the hot stage still scans every page.
- **Batched rescoring:**
  - It is measured at 1.5–4.0 ms per query, falling to 2.8x faster than the old loop
    at the lowest overlap tested.
  - With no overlap at all (451 × 50 = 22,550 distinct pages per batch), it was **not
    measured** and is **not projected**.
  - INFERENCE: its per-distinct-page cost grew roughly linearly over the measured
    range, so at very low overlap it may approach or exceed the old per-query loop.
    That is unresolved.
  - Above about 1,300 pages, rescoring is a minor share of total time either way.

## 9. Model-family sensitivity

**float32 per-page size** (MEASURED):

| model | float32 per page |
|---|---|
| ColPali | 528 KB (1,031 × 128) |
| ColQwen2 / 2.5 | 375–383 KB (733–748 × 128) |
| ColEmbed 4B | 7.56–7.77 MB (738–758 × 2,560), about 15–20x a 128-d page |

**The same configuration fits about 15–20x fewer ColEmbed pages within 8 GB**
(PROJECTED). Pages within 8 GB:

| model | float32 | int8 | binary | Ward 1/4 > PCA 640 > binary |
|---|---|---|---|---|
| ColEmbed 4B | 1,106 | 4,421 | 35.4k | 564k |
| ColQwen2 DocVQA (for comparison) | 22.4k | 88.3k | 717k | – |

**ColEmbed's codec factors match the 128-d ones** (int8 4.0x against 3.9x; binary 32x).
The 128-d int8 factor is slightly lower because its per-vector scale is a larger share
of a narrower vector.

**ColEmbed's fixed PCA basis (6.55 MB) matters at small N only.** For Ward 1/4 > PCA
640 > binary it is about 46% of the index at 500 pages and under 0.05% at 1M pages
(PROJECTED).

**ColEmbed latency is not projected.** Its only timings are GPU (R12) and cannot be
placed on the CPU model. **Nothing is projected for 4,096-d or any other untested
width.**

## 10. Bottleneck and crossover analysis (INFERENCE from §4–9)

- **RAM stops being practical early for uncompressed or lightly compressed 128-d
  indexes:**

  | configuration (128-d) | pages within 8 GB |
  |---|---|
  | float32 | 16–23k |
  | int8 | 64–90k |
  | binary | 0.5–0.7M |
  | Ward 1/4 > binary | 2–2.8M |
  | Ward 1/4 > PCA 64 > binary | 4–5.6M |

  For ColEmbed 4B it is about 20 times earlier.
- **Full CPU scanning dominates every exhaustive configuration as N grows.**
  - Even the smallest validated index projects to about 8.6 s per query at 1M pages
    on this laptop (Ward 1/4 > binary, ColQwen2, batch-amortized).
  - **Exhaustive search is not interactive at that scale on this hardware,** under the
    linear assumption.
- **Vector count stays the dominant search-cost lever.** Per-vector scan cost is
  similar across codecs (37–39 ms per million vectors). Ward 1/4 cuts it by 3.6–3.9x
  at every measured size.
- **Dimension reduction mainly saves storage.**
  - PCA 64 halves storage, but cuts the per-vector cost by only about 22% (30.8
    against 39.3 ms per million vectors).
  - So it barely changes the latency curve (consistent with Q4).
- **Two-tier changes RAM, not the scaling class.** Rescoring is fixed per query, and
  the linear hot scan dominates beyond about 1,300 pages.
- **Batched rescoring matters only while rescoring is a large share,** which means
  small corpora or heavy candidate overlap. Its advantage measurably falls as overlap
  falls.
- **Corpus-size-dependent versus query-dependent resources:**
  - **Dependent on N:** storage, RAM, scan and decode work, the hot stage, and the
    output score matrix.
  - **Per query (independent of N):** query encoding (not measured here) and old-style
    rescoring of a fixed candidate count.
  - **Overlap-dependent:** batched rescoring.
  - **Fixed:** the PCA basis and codec state.
- **What would change the scaling class:** sublinear candidate generation (an ANN
  index over tokens or pages, or MUVERA-style encodings). It is **untested** in
  OptiVision (Q5/Q6, Q7 context).

## 11. Measured against projected evidence

| claim | status |
|---|---|
| bytes per page, vectors per page and compression factors for each configuration at 500 pages | MEASURED |
| scan time at 500 pages (3 models, 31 configurations, CPU) | MEASURED |
| scan time linear in stored vectors from 500 to 4,000 pages (6 configurations, ColQwen2, duplicated pages) | MEASURED |
| old rescoring flat in N from 500 to 4,000 pages; batched rescoring rising with distinct candidates | MEASURED |
| index size at 1k–1M pages | PROJECTED (exact arithmetic on measured bytes per page) |
| pages that fit in 8 GB | PROJECTED (arithmetic + the stated RAM assumption) |
| scan time at 10k–1M pages | PROJECTED (linear assumption beyond 4,000 pages; only within 8 GB) |
| two-tier share and crossover at about 1,300 pages | PROJECTED (linear hot stage + measured flat rescoring) |
| batched rescoring at very low overlap | **unresolved** (not projected) |
| scan time when the index exceeds RAM (paging, disk) | **unresolved** (not projected) |
| retrieval quality at any N above 500 | **not established** |

## 12. Limitations

- **Linearity was measured only up to 4,000 pages,** and on duplicated pages with one
  model (ColQwen2). Cache and memory effects at larger sizes are untested; float32
  already shows a slight upturn (1.08) at 4,000 pages.
- **Timing environment:**
  - One laptop, numpy/OpenBLAS.
  - Times are batch-amortized ms per query, not single-query latency.
  - The laptop has two CPU speed states (Q5/Q6 and the batched-rescore follow-up),
    which can move absolute times by up to about 2.8x
    between states.
  - The first Q8 run was invalidated by standby.
- **V2 used random candidates** as a proxy for low overlap; real shortlists may overlap
  more or less.
- **No paging, disk-backed or distributed model.** Anything beyond the 8 GB assumption
  is storage arithmetic only.
- **ColEmbed is covered by storage arithmetic only;** it has no CPU latency.

## 13. Implications for million-page scenarios (all PROJECTED unless stated)

**Storage at 1M pages, from measured bytes per page:**

| configuration | ColQwen2 DocVQA | ColEmbed 4B DocVQA |
|---|---|---|
| float32 | 383 GB | 7.8 TB |
| int8 | 97 GB | 1.9 TB |
| binary | 12 GB | 243 GB |
| Ward 1/4 > binary | 3.1 GB | – |
| Ward 1/4 > PCA 64 / 640 > binary | 1.6 GB (64) | 15 GB (640) |
| two-tier | 3.1 GB RAM + 100 GB disk | 61 GB RAM + 2.0 TB disk (Ward 1/4 hot) |

**Search at 1M pages on this laptop (exhaustive, linear assumption, Ward 1/4 > binary):**
about 8.6 s per query for ColQwen2 and 12.3 s for ColPali. None of the uncompressed or
lightly compressed configurations fits in RAM.

**What these statements are not.** "At the measured 3.13 KB per page, 1M ColQwen2
pages project to 3.1 GB" is an arithmetic statement. **"OptiVision handles 1M pages" is
not established:** nothing has been run above 4,000 (duplicated) pages, and no quality
has been measured above 500 real pages.

## 14. What Q8 does NOT establish

- **That OptiVision was validated, demonstrated, benchmarked or is production-ready at
  any size above 500 real pages,** or above 4,000 duplicated pages for timing.
- **That retrieval quality holds at larger N.** The only quality-versus-size evidence
  (R10) shows lossy pipelines losing 0–2 retention points from 100 to 500 pages. At 1M
  pages each query faces far more near-duplicates.
- **Latency once the index exceeds RAM,** single-query latency, multi-node behaviour,
  or GPU behaviour for the 128-d models.
- **Anything about 4,096-d or other untested models.**
- **Batched-rescoring performance at very low candidate overlap.**

## 15. Conclusion

**Storage and RAM scale linearly with pages,** at measured per-page constants plus a
small fixed shared state, so they can be projected exactly. Using the numbers from §6:
- uncompressed 128-d indexes exceed 8 GB at 16–23k pages;
- Ward 1/4 > binary fits about 2–2.8M pages;
- ColEmbed 4B needs about 15–20x more per page.

**Full-scan search cost is linear in stored vectors over 500–4,000 pages** (MEASURED,
six configurations). Vector count is the lever that moves it; codecs and dimension
reduction mostly move storage.

**Two-tier changes RAM, not the scaling class.** Its fixed per-query rescoring becomes
negligible beyond about 1,300 pages, where the exhaustive hot scan dominates. The
batched-rescoring advantage shrinks as candidate overlap falls (7.4x → 2.8x, MEASURED).

**A million-page index is projectable as storage arithmetic, but not as a demonstrated
capability.** Exhaustive CPU search projects to several seconds per query at that
scale on this laptop. Retrieval quality at that scale is unmeasured. Sublinear
candidate generation is what would be needed, and it is untested.
