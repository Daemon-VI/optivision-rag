# Q4 report: which compression axis drives quality-preserving compression?

Branch `research-investigation`. Pre-registration: `PLAN.md` (`709f421`), with
clarifications in `1153f17` added before any result was inspected. No library change,
no Method C, no optimizer, nothing merged.

Labels:
- **MEASURED**: computed here, or in R12 within this repository;
- **EXTERNAL**: reported elsewhere (none used);
- **EXTRAPOLATED**: none in this report;
- **INFERENCE**: our reading of measured evidence.

| Item | Artifact |
|---|---|
| Per-query results | `store/<dataset>.npz` and `.json` (6 datasets × 74 configurations + 2 two-tier rows) |
| Latency | `latency/<model>_docvqa.json` |
| Analysis | `analysis.json` |
| Scripts | `scripts/research/q4_ablation.py`, `q4_latency.py`, `q4_analysis.py` |
| ColEmbed 4B rows | R12's committed results (`reports/universal/wide/`), reused unchanged |

**Reproduction checks (MEASURED): all passed.**
- On every dataset, the float32 baseline and the 11 configurations shared with the
  E1 store give per-query nDCG@5 **identical** to E1 (72 of 72 checks).
- The 17 configurations shared with the R5b/R11 frontier files reproduce their
  retention exactly (maximum difference 0.0, on all 6 datasets).

## 1. Research question

How much do three axes, alone and combined, contribute to each of five objectives?
- **Axes:** vector-count reduction (**A**, Ward merging within each page),
  dimension reduction (**B**, uncentred PCA fitted on the documents), and
  quantization (**C**).
- **Objectives:** retrieval quality, storage, RAM, search latency and preprocessing
  cost. They are not folded into one score.

## 2. Hypotheses (as pre-registered)

- **H4a:** A contributes disproportionately to latency reduction.
- **H4b:** B contributes disproportionately to storage/RAM efficiency.
- **H4c:** C adds substantial compression after structural reduction, with
  model-dependent codec behaviour.
- **H4d:** Combinations interact and are not predicted by multiplying single-axis
  effects.
- **H4e:** The qualitative trade-offs carry over from 128-d to the 2,560-d ColEmbed
  model.

The operational criteria are in `PLAN.md` §2 and are applied as written in §15.

## 3. Experimental design

**Grid.** One fixed grid of 74 configurations per 128-d dataset (`PLAN.md` §4). Every
configuration is fixed in advance and fitted on documents only (no labels, no
queries), so reporting it on all queries involves no selection.

| group | configurations |
|---|---|
| A | Ward 1/2, 1/3, 1/4, 1/8, 1/10 (adaptive merge r = 0.6 as a second method) |
| B | PCA to 64, 32, 16, 8 (random projection to 32 as a control) |
| C | float16, int8, plain int4, centred int4, 2-bit, binary |
| combinations | A+C, A+B, B+C, A+B+C at representative points |
| two-tier | R12's two two-tier rows, rerun |

**Order and fitting.**
- Stages run merge, then project, then codec. Each is fitted on the previous stage's
  output, so the PCA basis inside A+B rows is refitted on the merged vectors (the
  only intended coupling).
- Single-axis rows change exactly one axis. The recorded vector counts and widths
  confirm this: A rows keep d = 128, and B and C rows keep the original vector count.

**Scoring semantics.**
- Every codec is scored by decoding to float32 and computing exact MaxSim, as the
  library does today.
- PCA projects the queries with the same map.
- So scores change only through the compression itself. Nothing changes the scoring
  rule.

**Accounting (one convention, MEASURED from the arrays).**
- **Index bytes** = codes (including per-vector scales) + offsets + codec state +
  projection basis.
- Compression factors are relative to the **float32 in-memory codes** of the
  original vectors (num_vectors × d × 4 bytes).
- **ColEmbed rows** use the same convention: R12's codes + codec state, with offsets
  and the 2,560 × w × 4-byte basis added from the recorded shapes.

**Latency.**
- The existing `ExactIndex.score`, untouched.
- On this laptop (Intel 11th-gen mobile, 8 logical CPUs), with nothing else running.
- 5 round-robin rounds over all 451 DocVQA queries, for 31 configurations per model.

**Held-out frontiers** use the library's own query split
(`split_queries(n, 0.5, seed = 0)`): membership is chosen on one half and reported
on the other.

## 4. Available model/dataset matrix (MEASURED)

| model | width | vectors on disk | DocVQA (451 q) | InfoVQA (494 q) | per-query results | latency |
|---|---|---|---|---|---|---|
| ColPali v1.3 (`vidore/colpali-v1.3-merged`; corrected from "v1.2", see note) | 128 | yes | full grid | full grid | yes | CPU, this laptop |
| ColQwen2 v1.0 | 128 | yes | full grid | full grid | yes | CPU, this laptop |
| ColQwen2.5 v0.2 | 128 | yes | full grid | full grid | yes | CPU, this laptop |
| ColEmbed 4B (nemotron v2) | 2,560 | **no** | R12's 74 rows | R12's 74 rows | **no** (aggregates + intervals) | GPU (R12, Kaggle 2×T4) |

**Correction (2026-10-03, found during Q7).** The ColPali checkpoint is
`vidore/colpali-v1.3-merged` (`docs/UNIVERSAL.md`), not "v1.2" as `PLAN.md` and an
earlier version of this table said. The data and results are unchanged; only the
label was wrong.

**Missing cells (not projected):**
- **ColEmbed 4B:**
  - Ward 1/8 does not exist, so the nearest row, Ward 1/10 (9.9x), stands in at the
    8x level.
  - Several combinations are absent: Ward 1/2 combinations, PCA 64/32/16-equivalent
    widths inside A+B+C, and B+C at d/16 with a 2-bit codec.
  - With no per-query data, interactions are point estimates only.
  - With no CPU timings, latency is compared within ColEmbed only.
- **All models:** A-only at 1/32 of the vectors and B-only at d/32 were not run (out
  of useful range, as the plan stated).
- **Planning error:** the two B+C rows the plan listed at "about 32x" (PCA 64 >
  centred int4, PCA 32 > int8) are actually 15x, because per-vector scales and
  4-bit codes do not reach 32x at those widths. They are reported at their actual
  factor. The nearest 32x-class B+C row is PCA 32 > centred int4 (28.4x).

## 5. Vector-count ablation (A only; float32, 128-d; MEASURED)

Retention of nDCG@5 against each model's own float32 index (95% intervals are in
`analysis.json`):

| vectors kept | ColPali Doc / Info | ColQwen2 Doc / Info | ColQwen2.5 Doc / Info | ColEmbed 4B Doc / Info |
|---|---|---|---|---|
| 1/2 (2.0x) | 0.999 / 0.999 | 1.003 / 0.996 | 0.987 / 0.996 | 0.995 / 1.000 |
| 1/3 (2.9x) | 0.992 / 0.994 | 0.997 / 0.995 | 0.995 / 1.000 | 0.991 / 0.996 |
| 1/4 (3.8–3.9x) | 0.989 / 0.999 | 1.006 / 0.991 | 0.989 / 0.996 | 0.988 / 0.995 |
| 1/8 (7.2–7.6x) | 0.951 / 0.990 | 0.986 / 0.984 | 0.977 / 0.991 | not run |
| 1/10 (8.8–9.9x) | 0.954 / 0.990 | 0.977 / 0.973 | 0.972 / 0.992 | 0.984 / 0.996 |

- Actual vectors per page at Ward 1/4:
  - ColPali: 263 of 1,031.
  - ColQwen2 and ColQwen2.5: 195 of 748 (DocVQA) and 190 of 733 (InfoVQA).
- Merging costs ≤ 1.3 points down to 1/4 of the vectors everywhere. At 1/8–1/10 the
  cost is 0.4–4.9 points, largest on ColPali DocVQA.

## 6. Dimension ablation (B only; all vectors, float32; MEASURED)

| width kept | ColPali | ColQwen2 | ColQwen2.5 | ColEmbed 4B |
|---|---|---|---|---|
| 1/2 | 0.974 / 0.972 | 0.972 / 0.991 | 0.975 / 0.987 | 1.002 / 0.999 |
| 1/4 | 0.803 / 0.871 | 0.875 / 0.899 | 0.842 / 0.907 | 0.993 / 0.996 |
| 1/8 | 0.379 / 0.494 | 0.410 / 0.425 | 0.356 / 0.473 | 0.991 / 0.994 |
| 1/16 | 0.094 / 0.146 | 0.087 / 0.085 | 0.069 / 0.116 | 0.967 / 0.988 |
| control: random projection to 1/4 | – | 0.752 (Doc) | – | 0.876 / 0.979 |

- **Projection fit.** Uncentred PCA on the document second-moment matrix, sampled up
  to 200,000 vectors (seed 0). Queries are projected with the same basis.
- **Basis size.** 128 × w float32 for the 128-d models. For ColEmbed it is
  2,560 × w; at 640 wide that is 6.5 MB, or 13.1 KB per page over 500 pages.
- **The dimension axis is completely different at the two widths.**
  - At 128-d, halving the width already costs 0.9–2.8 points, and a quarter of the
    width costs 9–20 points.
  - At 2,560-d, an eighth of the width costs ≤ 0.9 points.
  - PCA beats random projection at both widths.

## 7. Quantization ablation (C only; all vectors, full width; MEASURED)

| codec (bits per dimension, 128-d) | ColPali | ColQwen2 | ColQwen2.5 | ColEmbed 4B |
|---|---|---|---|---|
| float16 (16) | 1.000 / 1.000 | 1.000 / 1.000 | 1.001 / 1.000 | 1.000 / 1.001 |
| int8 per vector (8.25) | 1.000 / 1.000 | 1.001 / 1.000 | 0.998 / 1.000 | 0.997 / 1.002 |
| **plain int4** (4.25) | 0.975 / 0.999 | 1.001 / 1.004 | 1.003 / 1.000 | **0.009 / 0.004** |
| centred int4 (4.25) | 0.990 / 1.001 | 0.994 / 1.000 | 0.998 / 1.000 | 0.997 / 1.002 |
| 2-bit Lloyd (2) | 0.952 / 0.994 | 1.003 / 0.996 | 1.002 / 0.998 | 0.987 / 1.000 |
| binary (1) | 0.963 / 0.974 | 0.972 / 0.990 | 0.995 / 0.991 | 0.989 / 0.998 |

- **Plain int4 is a measured failure mode at 2,560-d only.** At 128-d it retains
  97.5–100.4%; on ColEmbed it collapses to 0.4–0.9%, which R12 already reported.
  The cause is still unresolved (no ColEmbed vectors available).
- **Codec behaviour depends on the model.** On ColPali DocVQA, 2-bit and binary lose
  3.7–4.8 points; on ColQwen2.5 DocVQA they lose ≤ 0.5.

## 8. Combination ablation (MEASURED)

**Retention at matched budgets, 128-d** (DocVQA / InfoVQA). Factors are actual index
bytes against float32:

| budget | A | B | C | A+B | A+C | B+C | A+B+C |
|---|---|---|---|---|---|---|---|
| **about 4x**, ColPali | 0.989 / 0.999 | 0.803 / 0.871 | 1.000 / 1.000 | 0.962 / 0.977 | – | – | – |
| ColQwen2 | 1.006 / 0.991 | 0.875 / 0.899 | 1.001 / 1.000 | 0.979 / 0.991 | – | – | – |
| ColQwen2.5 | 0.989 / 0.996 | 0.842 / 0.907 | 0.998 / 1.000 | 0.954 / 0.987 | – | – | – |
| **about 8x**, ColPali | 0.951 / 0.990 | 0.379 / 0.494 | 0.990 / 1.001 | 0.956 / 0.975 | 0.997 / 0.999 | 0.973 / 0.973 | 0.963 / 0.977 |
| ColQwen2 | 0.986 / 0.984 | 0.410 / 0.425 | 0.994 / 1.000 | 0.973 / 0.984 | 1.004 / 0.995 | 0.974 / 0.991 | 0.978 / 0.991 |
| ColQwen2.5 | 0.977 / 0.991 | 0.356 / 0.473 | 0.998 / 1.000 | 0.942 / 0.987 | 0.988 / 0.996 | 0.976 / 0.987 | 0.954 / 0.987 |
| **about 30x**, ColPali | not run | not run | 0.963 / 0.974 | 0.361 / 0.484 | 0.978 / 0.998 | (15x: 0.943 / 0.966) | 0.960 / 0.975 |
| ColQwen2 | not run | not run | 0.972 / 0.990 | 0.356 / 0.401 | 1.004 / 0.991 | (15x: 0.966 / 0.991) | 0.977 / 0.984 |
| ColQwen2.5 | not run | not run | 0.995 / 0.991 | 0.304 / 0.428 | 0.987 / 0.996 | (15x: 0.963 / 0.983) | 0.941 / 0.987 |

Configurations behind the matched-budget cells:

| budget | A+B | A+C | B+C | A+B+C |
|---|---|---|---|---|
| about 4x | Ward 1/2 > PCA 64 | – | – | – |
| about 8x | Ward 1/4 > PCA 64 | Ward 1/2 > int8 | PCA 64 > int8 | Ward 1/2 > PCA 64 > fp16 |
| about 30x | Ward 1/4 > PCA 16 | Ward 1/4 > centred int4 | PCA 64 > centred int4 (actually 15x) | Ward 1/4 > PCA 64 > int8 |

- At every matched budget on the 128-d models, **A+C is at or near the top** and
  everything involving narrow PCA is at the bottom.
- C alone wins at ≤ 8x.

## 9. Storage results (MEASURED)

- **Storage on the 128-d models is governed by codec and vector count.** Per-vector
  int8 costs 132 bytes per vector (4 bytes of scale per vector). A float32 page of
  1,031 vectors (ColPali) is 528 KB; Ward 1/4 > binary is 4.2 KB, i.e. 125x.
- **Per-vector scales stop the factors from multiplying.** At 128-d an int8 scale is
  3% of a vector's bytes; at PCA 8 it is 33%. So adding int8 after PCA 8 yields 3.2x,
  not 4x.
- **The shared state matters at small corpora (ColEmbed, 500 pages).**
  - The PCA basis is 2.6–13.1 KB per page.
  - For `Ward 1/4 > PCA 128 > binary` that is 2.6 KB of basis against 3.0 KB of
    codes per page: the factor is 2,553x by R12's codes-only count and 1,369x
    including the basis.
  - The basis is a one-off cost and its share falls with corpus size. That trend is
    not measured beyond 500 pages and is not extrapolated.
- **Disk (MEASURED `.npz` size where the library can write it).**
  - The library's `save_compressed` writes codes + offsets + a JSON header, about
    15 bytes per page more than the index bytes.
  - It **refuses centred int4 and 2-bit**, because they have fitted codec state.
  - It **does not store the PCA basis**, so a saved projected index cannot be queried
    after reloading.
  - These are library limitations found here and not fixed (that would be a library
    change). Every row's index bytes are still reported.

## 10. RAM results (MEASURED)

- **In the current implementation, RAM is the same as storage for single-tier
  indexes.** `ExactIndex` keeps the codes resident and decodes page-aligned blocks to
  float32 during a scan, with the live block capped at 256 MiB for every
  configuration. So the resident RAM equals the index bytes in §9, and a single-tier
  index's RAM and storage frontiers coincide.
- **The two-tier rows are the only configurations where RAM and disk differ.**
  - Binary codes are held in RAM, and a 50-page shortlist is rescored from int8 codes
    of every vector:

    | hot tier in RAM | RAM vs float32 | disk vs float32 | retention, 128-d (all six datasets) | ColEmbed |
    |---|---|---|---|---|
    | binary codes | 32x | 3.5–3.8x | 0.999–1.005 | 99.7% / 100.3% |
    | Ward 1/4 > binary | 122–125x | 3.5–3.8x | 0.999–1.005 | 99.7% / 100.3% |

  - They dominate the RAM frontier up to about 125x at full retention.
  - They do so at a latency cost (§11).

## 11. Latency results (MEASURED, laptop CPU, existing implementation)

Median scan time as a fraction of the model's own float32 scan (DocVQA, 451 queries,
5 rounds). Float32 itself takes 22.2 ms/query on ColPali, 15.7 on ColQwen2 and 15.8
on ColQwen2.5. ColEmbed fractions are taken against R12's float32 *row* (46.4 ms DocVQA),
which goes through the same scoring path as the other rows. R12's separately timed
baseline pass (52.0 ms) would make every ColEmbed fraction about 11% smaller.

| | ColPali | ColQwen2 | ColQwen2.5 | ColEmbed 4B (GPU, R12) |
|---|---|---|---|---|
| A, 1/2 of the vectors | 0.53 | 0.53 | 0.52 | 0.47 / 0.44 |
| A, 1/4 | 0.28 | 0.27 | 0.27 | 0.26 / 0.25 |
| A, 1/8 (ColEmbed: 1/10) | 0.16 | 0.16 | 0.16 | 0.12 / 0.11 |
| B, d/2 | 0.80 | 0.79 | 0.79 | 0.89 / 0.87 |
| B, d/4 | 0.74 | 0.74 | 0.74 | 0.80 / 0.78 |
| B, d/16 | 0.70 | 0.71 | 0.70 | 0.73 / 0.73 |
| C: float16 / int8 / centred int4 / binary | 1.01–1.04 | 1.01–1.04 | 1.01–1.03 | 1.06–1.25 |
| A+B: Ward 1/4 > PCA 64 | 0.22 | 0.22 | 0.22 | – |
| A+C: Ward 1/4 > any codec | 0.28–0.29 | 0.27–0.29 | 0.27–0.28 | 0.27–0.30 |

- **Vector count drives scan time almost proportionally.** Scan time falls to about
  0.53, 0.28 and 0.16 of float32 at 1/2, 1/4 and 1/8 of the vectors.
- **Width has a floor.** Even 1/16 of the width leaves 70% of the scan time. INFERENCE:
  the per-vector and per-page work (the segment max and the block handling) does not
  shrink with d.
- **Codecs make scans 1–7% slower,** because codes are decoded to float32 before the
  product.
- **Two-tier.** Rescoring the 50-page shortlist costs **12.4–20.5 ms/query on the CPU
  (ColEmbed GPU: 343–351 ms)**, as much as a full float scan. So two-tier buys RAM,
  not speed, in the current implementation.
- **Warm-up artefact.** The first of the five ColPali rounds was about 1.5x slower for
  *every* configuration (for example float32 33.1 ms against 21.8–22.3). The two
  ColQwen models show no such round. It is reported, not removed (§15, H4a).
- **Preprocessing (MEASURED wall time, one run, uncached pipeline):**
  - Ward merging dominates: 8.4–9.5 s per ColQwen corpus and 24–29 s per ColPali
    corpus (500 pages).
  - PCA takes 0.2–0.4 s; codecs ≤ 2 s (2-bit is the slowest).
  - Merge followed by projection or centred int4 took about twice as long as merging
    alone. The cause was not investigated.
  - Encoder time does not depend on the compression choice and is not compared.

## 12. Pareto frontiers (separate; MEASURED)

**Storage (index bytes) frontier, 128-d.** Members with retention ≥ 0.90, from
low to high compression. IN-SAMPLE SELECTION, membership on all queries:
- ≤ 8x: **C alone** (int8 / int4 / centred int4), with float32 retention.
- 8–125x: **A+C** (Ward 1/3–1/4 with int8, int4, 2-bit or binary).
- 180–240x: **A+B+C** (Ward 1/3–1/4 > PCA 64 > binary), at 0.90–0.97.
- B alone and B+C are essentially absent: one held-out B+C point (ColQwen2
  InfoVQA, 15x).

**Held-out check.** With membership chosen on the calibration half and retention
taken on the held-out half, the same axis composition reappears:
- Retention stays within about 2 points, with one exception: ColPali DocVQA's 186x
  A+B+C point drops from 0.913 to 0.864.
- So the composition is not a selection artefact, but the far end of the frontier is
  fragile.

**Storage frontier, ColEmbed 4B** (R12 rows; IN-SAMPLE only, since no per-query data
exist):

| region | DocVQA | InfoVQA |
|---|---|---|
| 8–16x | B+C (`project(1280) > int8`, 7.8x at 1.004; `project(640) > int8`, 15.5x at 1.001) | B+C (`project(1280) > binary`, 52x at 1.001) |
| 30–60x | B+C (`project(640) > centred int4`, 30x at 0.996; `project(320) > centred int4`, 60x at 0.982) | B+C (`project(640) > binary`, 105x at 0.999) |
| about 128x | A+C (Ward 1/4 > binary, 0.974) | – |
| > 230x | A+B+C | A+B+C |

On InfoVQA, C alone covers ≤ 8x.

**RAM frontier:**
- Single-tier members are the storage members, since RAM equals storage (§10).
- The **two-tier rows** add points at 32x and 122–125x with 0.999–1.005 retention,
  dominating everything at or below their RAM. Their disk footprint is only
  3.5–3.8x.

**Latency frontier, 128-d** (DocVQA latency subset):
- **A alone** dominates: Ward 1/2 to 1/10 at 0.13–0.53 of the float scan, with
  0.95–1.00 retention.
- A+C points tie with A at the same vector count, because codecs do not change scan
  time.
- A+B (Ward 1/4 > PCA 64) is on the ColPali frontier at 0.22 of the scan (retention
  0.956).
- B and C alone are never on it, except at float32-like cost.

**Which axis dominates where (INFERENCE from the frontiers).**
- **128-d:** codecs give the first about 8x of storage for free. Merging then
  multiplies on top up to about 125x. PCA joins only beyond about 180x, at a visible
  quality cost. Latency is governed by merging alone.
- **2,560-d:** projection with a codec dominates storage from 8x to about 63x
  (DocVQA) or 105x (InfoVQA). All three axes are needed beyond about 230x. Latency is
  again governed by merging (GPU timings).

## 13. Interaction analysis (MEASURED; intervals are paired bootstraps over queries)

**Definition.** The interaction is I = R_{X+Y(+Z)} − Π R_single, the measured
retention minus the product of the single-axis retentions on the same queries.

| dataset | combinations with \|I\| ≥ 0.01 and interval excluding 0 | median I | most negative I | largest positive I |
|---|---|---|---|---|
| ColPali DocVQA | 12 / 56 (21%) | −0.007 | −0.154 | +0.017 |
| ColPali InfoVQA | 8 / 56 (14%) | −0.004 | −0.090 | +0.014 |
| ColQwen2 DocVQA | 19 / 56 (34%) | −0.005 | −0.184 | +0.013 |
| ColQwen2 InfoVQA | 9 / 56 (16%) | −0.004 | −0.048 | +0.030 |
| ColQwen2.5 DocVQA | 17 / 56 (30%) | −0.012 | −0.160 | +0.006 |
| ColQwen2.5 InfoVQA | 15 / 56 (27%) | −0.002 | −0.061 | +0.020 |
| ColEmbed DocVQA (points only) | 25 / 54 with \|I\| ≥ 0.01 | – | −0.053 | +0.011 |
| ColEmbed InfoVQA (points only) | 7 / 54 | – | −0.040 | +0.015 |

**Where interactions occur.**
- **Every significant interaction at 128-d is negative.**
- They concentrate in **PCA combined with a coarse codec**. On ColQwen2 DocVQA,
  `project(32) > binary` has I = −0.105 [−0.156, −0.055] and
  `Ward 1/3 > PCA 32 > binary` has I = −0.159. They also appear in Ward combined with
  PCA at ≤ 32 dimensions.
- On ColEmbed the large negative points are likewise PCA + binary and PCA +
  centred int4.
- **Merging + codec combinations are close to multiplicative.** For Ward 1/3–1/4 with
  float16, int8, int4, centred int4, 2-bit or binary, there is no material
  interaction (|I| ≥ 0.01 with the interval excluding 0) on any 128-d dataset. One
  sub-point interaction has an interval excluding 0: ColQwen2.5 DocVQA,
  Ward 1/4 > int8, |I| < 0.01.

**Answers to the brief's questions (INFERENCE from the table and §8):**
- **Is A+B better than either axis alone?**
  - At about 4x, A+B (Ward 1/2 > PCA 64, 0.954–0.991) is better than B alone
    (0.80–0.91) but worse than A alone (0.989–1.006).
  - At about 8x, A+B beats B alone by 50+ points and roughly matches A alone.
  - At 128-d, B never helps A at a matched budget.
- **Is B+C better than B alone?**
  - At matched *bytes*, yes: PCA 64 > int8 at 7.7x keeps 0.973–0.991, against
    0.36–0.49 for PCA 16 alone at 8x.
  - The codec is the cheap way to buy the bytes.
- **Does C preserve quality after A or B?**
  - After merging: int8, centred int4 and plain int4 change retention by −0.013 to
    +0.006 against the merged float row at 1/3–1/4 of the vectors. The worst case is
    ColPali DocVQA, Ward 1/4 > plain int4; int8 alone stays within ±0.004.
  - After narrow PCA: binary and centred int4 lose substantially more than the
    product predicts.
- **Does binary remain useful after merging?**
  - Yes. Ward 1/4 > binary keeps 0.954–0.982 at 122–125x, about what the product of
    the two predicts.
  - It is the storage-frontier point near 125x on every 128-d dataset.
- **Diminishing returns?**
  - Yes, along A beyond 1/4 of the vectors (losses accelerate at 1/8–1/10).
  - Along B at 128-d the loss is immediate.
- **Do two individually safe steps become unsafe together?**
  - Only partly. The large, clearly significant cases (−5 to −18 points) involve PCA
    to ≤ 32 dimensions, which is already lossy alone. Combining it with binary or
    centred int4 makes it worse than the product predicts.
  - Among near-safe steps (each single-axis retention ≥ 0.97), PCA 64 > binary has
    interactions from −0.036 to +0.014, and every interval includes 0.
  - Ward 1/4 > PCA 64 > binary has a significant negative interaction on 2 of 6
    datasets: ColQwen2.5 DocVQA, −0.057 [−0.106, −0.009], and ColQwen2 InfoVQA,
    −0.021 [−0.040, −0.002].
- **Compression factors.** The actual factor of a combination is 0.69–1.00 times the
  product of its single-axis factors at 128-d, and 0.53–1.00 for ColEmbed index bytes
  (per-vector scales; the basis at 500 pages). So byte factors do not simply multiply
  either.

## 14. Cross-model comparison and 128-d against 2,560-d (MEASURED; reading is INFERENCE)

| question | ColPali | ColQwen2 | ColQwen2.5 | ColEmbed 4B (2,560-d) |
|---|---|---|---|---|
| Axis that drives latency | A | A | A | A (GPU) |
| Single axis with the most quality per byte at 2x/4x/8x | C (A ties at ≤ 4x) | A at 2x/4x on DocVQA, C on InfoVQA; C at 8x | C | no clear winner; all three ≥ 0.98 at ≤ 8x |
| Dimension axis | collapses beyond d/2 | collapses beyond d/2 | collapses beyond d/2 | nearly lossless to d/8 |
| Plain int4 | works (0.975–0.999) | works | works | **fails** (≤ 0.9%) |
| Coarse codecs (2-bit, binary) | weakest (DocVQA −3.7 to −4.8) | moderate | strong | strong |
| Interactions | negative, PCA + coarse codec | same | same | same direction (points) |
| Storage frontier mid-range | A+C | A+C | A+C | **B+C** (8–63x / 105x) |

**128-d against 2,560-d, on normalized curves:**
- The **latency** conclusion persists across widths: merging is the only axis that
  cuts scan time roughly in proportion, and both width and codecs have floors.
- The **storage-efficiency** conclusion does **not** persist:
  - At 2,560-d, PCA to a quarter or an eighth of the width is nearly free, so B+C
    owns the mid-range frontier.
  - At 128-d, PCA to half the width already costs 1–3 points, so A+C owns it.
- **Codec failure modes differ by width or model family.** Plain int4 fails only on
  the 2,560-d model. Coarse codecs are weakest on ColPali.
- **Model family matters within 128-d** for coarse codecs (ColPali against the
  ColQwen models), but not for the ranking of the axes.

## 15. Supported conclusions (with the pre-stated criteria)

- **H4a (count drives latency): supported in substance; the pre-stated rule returns
  "mixed".**
  - A is faster than B and C at every level on every model.
  - It is beyond the max−min repeat spread in 8 of 9 cells. The exception is ColPali
    2x, where the spread is inflated by the warm-up round of §11.
  - POST HOC checks:
    - excluding round 1, A is faster beyond the spread in all 9 cells;
    - within each round, A was the fastest in 5 of 5 rounds in all 9 cells.
  - ColEmbed (GPU) agrees at all three levels.
- **H4b (width is the most storage-efficient axis): not supported.**
  - At 128-d, B has the lowest retention of the three axes at every matched level on
    every dataset. Every best-minus-B interval excludes 0, by 0.9–64 points.
  - At 2,560-d, B is best only at 2x on DocVQA, and the three axes are within 1.4
    points of each other.
  - The formal rule says "mixed", because no other single axis cleared the
    clear-winner bar at ≥ 2 levels on ≥ 2 models in both splits.
  - **Qualification of R12 §4.** R12 said "width is the most efficient axis at
    moderate compression" (PCA 320 at 8x and 99.1%, against merging at 10x and 98.4%).
    The numbers are reproduced, but the statement holds against merging only:
    centred int4 alone reaches 8x at 99.75% / 100.2%. This qualifies R12's reading;
    it does not contradict its measurements.
- **H4c (codecs after structure; codec behaviour model-dependent): partially
  supported.**
  - Part 1: int8 adds 3.54–3.94x at a retention change of −0.004 to +0.012 at 12 of
    13 structural points, on all six datasets. The exception is PCA 8, where per-vector
    scales cap the added factor at 3.2x. Retention there was unaffected, but the
    pre-stated "every point" criterion fails.
  - Part 2 is supported: plain int4 works at 128-d and fails at 2,560-d. Binary and
    2-bit differ by up to 4.8 points between ColPali and the ColQwen models on DocVQA.
- **H4d (interactions; non-multiplicative factors): partially supported.**
  - Quality: the share of combinations with a material, interval-excluding-0
    interaction is 14–34%. It meets ≥ 25% on both splits only for ColQwen2.5, so 1 of
    3 models, and the quality part is **not** supported by the pre-stated threshold.
  - What is measured: interactions are real, almost always negative, and confined to
    one pattern, PCA with coarse codecs. Merging + codec is near-multiplicative.
  - The byte-factor part is supported: 0.69–1.00 at 128-d and 0.53–1.00 on ColEmbed.
- **H4e (the trade-offs carry over to 2,560-d): not supported as a whole.**
  - The latency axis and the direction of interactions do carry over.
  - The storage-efficient axis (A+C against B+C) and the plain-int4 failure do not.
  - That is 2 of the 4 pre-stated items.

**Strongest measured conclusions:**
1. Vector count is the only axis that cuts search time in the current
   implementation: about proportionally, 0.16 of float32 at 1/8 of the vectors.
   Width has a floor of about 70%, and codecs make scans 1–7% slower.
2. At 128-d, per-vector int8 and centred int4 stay within about 1 point alone and after
   merging. Merging to 1/4 costs ≤ 1.3 points. Their combination reaches 15–30x
   within about 2 points on every dataset.
3. At 128-d, PCA is the wrong axis below half the width (9–20 points lost at d/4). At
   2,560-d it is nearly free to d/8. So the best storage pipeline depends on the
   width: A+C at 128-d, B+C at 2,560-d.
4. PCA combined with binary or centred int4 interacts negatively (up to −18 points),
   while merging combined with any codec does not.
5. In the current implementation RAM equals storage. Two-tier is the only way to
   separate them, and its rescoring costs as much as a full float scan.

## 16. Unsupported conclusions (what this study does not show)

- Nothing here shows latency on a native integer or binary kernel. Every codec is
  decoded to float32 (Q5/Q6).
- ColEmbed latency is GPU-only and cannot be compared with the CPU numbers. Its
  ranking of the axes is within-model only.
- No result holds for corpora much larger than 500 pages. That matters most for the
  PCA basis, whose share of the bytes falls with corpus size.
- There is no held-out evidence for ColEmbed: its frontier is IN-SAMPLE.
- Merging may not be the best vector-count method. Ward was the tested method;
  adaptive merging is reported but not compared formally.
- Nothing here bears on the selection rules or Method C. Q4 did not use an optimizer.
- The cause of the plain int4 collapse at 2,560-d remains unknown.

## 17. Limitations

- **Coverage.** Two corpora (DocVQA, InfoVQA) and three 128-d models plus one 2,560-d
  model.
- **Laptop timing.** One laptop, Python and numpy; a warm-up round on ColPali (§11).
  Absolute milliseconds will not transfer to other hardware.
- **ColEmbed is reused from R12.** It has aggregates only, no per-query data and GPU
  timings, and some grid cells are missing (§4).
- **Fixed grid.** The axis levels are fixed, so a matched budget only matches
  approximately (actual factors are reported).
- **Planning error.** The "about 32x" B+C rows are 15x (§4).
- **Disk measurement.** Disk could be measured as a file only for codecs the library
  can save, and the saved projected index lacks its basis.
- **One pipeline order.** Merge → project → codec. Other orders were not tested.

## 18. Recommended Q5/Q6 experiment (not started)

The smallest experiment that addresses Q4's two latency findings:
- codecs do not cut scan time, because they are decoded to float32;
- two-tier rescoring costs as much as a full float scan.

**Design.**
- **Setting:** one model and dataset (ColQwen2 DocVQA, where the latency is cleanest),
  the existing two-tier configuration (hot tier Ward 1/4 > binary, cold tier int8,
  50 candidates), and research-only code outside the library.
- **Step 1, exact int8 cold tier.** Score the shortlist directly from per-vector int8
  codes as scale · (q · code) with float accumulation, instead of decoding blocks.
  Rescore all 50 candidates of all queries in one batched operation instead of the
  per-query Python loop. Verify: identical rankings, scores within float32 rounding.
- **Step 2, binary hot tier.** Score directly on packed bits:
  - with float queries (asymmetric, sign-weighted sums), which is exact;
  - with binarized queries (symmetric Hamming), which is approximate.

  Measure shortlist recall against the decoded binary scores.
- **Measure:** end-to-end latency (hot scan + rescore) with the Q4 protocol (5
  rounds, median, spread, nothing else running), RAM, and nDCG@5 retention against
  float32.
- **Criterion, stated in advance:** the exact paths must reproduce the current
  rankings exactly. An approximate path is reported with its recall loss.

This needs no new encoding, no new model and no library change. It answers whether
the RAM savings of two-tier can come without the current latency penalty.
