# E7a report: attributing the OptiVision against SAP disagreement

Frozen plan: `PLAN.md` (`47e98fe`). Script: `scripts/research/e7a_attribution.py`
(`f8b6c2b`, committed before running). Analysis: `scripts/research/e7a_analysis.py`.
Data: `<store>.json` / `.npz` and `analysis.json`.

- CPU only, existing vectors.
- No library change, no new dependency, no GPU, no re-encoding.
- About 43 minutes of compute in total.
- Results are **MEASURED** unless labelled EXTERNAL (SAP's published values) or
  INFERENCE.

**Checks (all passed, six datasets).**
- The uncompressed per-query nDCG@5 equals the E1 store.
- Method C at 1/10 reproduces Q4's `hierarchical_merge(0.1)` per-query nDCG@5
  exactly.
- Ward produced exactly k clusters on every page.

## 1. Frozen methodology (summary of `PLAN.md`)

**Four methods**, each reducing a page's free (unprotected) vectors to exactly k:
- **A. K-means, plain mean.** numpy Lloyd's algorithm, k-means++ initialization, at
  most 50 iterations; empty clusters re-seeded with the farthest point; seeds
  {0, 1, 2} per page.
- **B. Ward, plain mean.** scipy Ward linkage on the unit vectors, maxclust k.
- **C. Ward with OptiVision rescaling.** The labels from B, then the library's
  `_pool`, unchanged.
- **D. Random pruning.** k original vectors, seeds {0, 1, 2}.

**Everything else is identical across methods:** protected tokens are kept unchanged,
and scoring, the query vectors, the corpus and the evaluation code are the same.
- Scoring: exact MaxSim.
- Quality: nDCG@5 retention against the uncompressed index (paired bootstrap,
  1,000 resamples, seed 0), plus top-1 accuracy.
- A and D: the per-query nDCG@5 is averaged over the three seeds, and per-seed values
  are reported.
- Attribution comparisons: paired intervals. A difference is "material" (**M**) if its
  interval excludes 0 and its size is ≥ 1 point.

## 2. Vector-count convention (exact)

- **Budget:** k = ceil(r × n_free) per page, with r ∈ {1/5, 1/10, 1/20}. This is the
  library's own `_budget` convention, so C at 1/10 equals Q4's Ward 1/10.
- **Protected (non-patch) tokens** are kept in addition: 7 per page for ColPali,
  11 for ColQwen2 and ColQwen2.5.
- **Kept vectors per page** (all methods identical within a dataset):

  | ratio | ColPali | ColQwen2 / 2.5 DocVQA | ColQwen2 / 2.5 InfoVQA |
  |---|---|---|---|
  | 1/5 | 212.0 (205 + 7) | 158.7 | 155.6 |
  | 1/10 | 110.0 (103 + 7) | 85.2 | 83.5 |
  | 1/20 | 59.0 (52 + 7) | 48.5 | 47.5 |

  The vector-count factors are in the table in §5. Storage was not measured.
- **This is OptiVision's definition, not SAP's.** SAP's γ is a fraction of visual
  tokens, and its handling of non-visual tokens, preprocessing and evaluation set are
  unknown. E7a tests the compression mechanisms at OptiVision's vector-count
  definition. It **does not reproduce SAP's operating point**.

## 3. Configurations

6 datasets × 3 ratios × (A with 3 seeds, B, C, D with 3 seeds) = 144 scored indexes.

## 4. Datasets and evaluation

- **Data:** the six Q4 datasets (ColPali v1.3, ColQwen2 v1.0, ColQwen2.5 v0.2 ×
  ViDoRe V1 DocVQA and InfoVQA), with 500 pages and 451 / 494 labelled queries each,
  as in Q4.
- **Evaluation:** the same loader, exact MaxSim scorer and nDCG@5 code as Q4.

## 5. Complete results (retention [95% interval]; seed ranges for A and D; top-1)

| dataset | ratio | vectors/page (factor) | A K-means plain | B Ward plain | C Ward rescaled | D random |
|---|---|---|---|---|---|---|
| ColPali DocVQA | 1/5 | 212.0 (4.9x) | 0.948 [0.921–0.975], seeds 0.939–0.956; top-1 0.458 | 0.963 [0.937–0.990]; top-1 0.468 | 0.976 [0.951–1.000]; top-1 0.486 | 0.922 [0.897–0.949], seeds 0.909–0.937; top-1 0.448 |
| ColPali DocVQA | 1/10 | 110.0 (9.4x) | 0.905 [0.872–0.940], seeds 0.898–0.913; top-1 0.447 | 0.929 [0.896–0.964]; top-1 0.461 | 0.954 [0.928–0.982]; top-1 0.468 | 0.839 [0.807–0.871], seeds 0.818–0.852; top-1 0.401 |
| ColPali DocVQA | 1/20 | 59.0 (17.5x) | 0.814 [0.774–0.856], seeds 0.797–0.823; top-1 0.393 | 0.851 [0.809–0.896]; top-1 0.410 | 0.897 [0.859–0.932]; top-1 0.437 | 0.746 [0.706–0.786], seeds 0.741–0.752; top-1 0.360 |
| ColPali InfoVQA | 1/5 | 212.0 (4.9x) | 0.989 [0.979–0.999], seeds 0.980–0.994; top-1 0.767 | 0.997 [0.986–1.006]; top-1 0.775 | 0.998 [0.989–1.008]; top-1 0.775 | 0.972 [0.958–0.984], seeds 0.966–0.981; top-1 0.744 |
| ColPali InfoVQA | 1/10 | 110.0 (9.4x) | 0.970 [0.954–0.983], seeds 0.967–0.972; top-1 0.743 | 0.981 [0.964–0.995]; top-1 0.765 | 0.990 [0.977–1.002]; top-1 0.765 | 0.944 [0.925–0.961], seeds 0.938–0.950; top-1 0.721 |
| ColPali InfoVQA | 1/20 | 59.0 (17.5x) | 0.950 [0.931–0.967], seeds 0.946–0.953; top-1 0.723 | 0.959 [0.939–0.978]; top-1 0.731 | 0.973 [0.957–0.989]; top-1 0.755 | 0.910 [0.888–0.931], seeds 0.908–0.915; top-1 0.681 |
| ColQwen2 DocVQA | 1/5 | 158.7 (4.7x) | 0.970 [0.947–0.995], seeds 0.964–0.981; top-1 0.507 | 0.993 [0.970–1.018]; top-1 0.517 | 0.998 [0.979–1.018]; top-1 0.514 | 0.931 [0.905–0.957], seeds 0.897–0.956; top-1 0.473 |
| ColQwen2 DocVQA | 1/10 | 85.2 (8.8x) | 0.936 [0.908–0.964], seeds 0.911–0.961; top-1 0.478 | 0.968 [0.939–0.998]; top-1 0.506 | 0.977 [0.954–1.001]; top-1 0.499 | 0.853 [0.820–0.886], seeds 0.839–0.875; top-1 0.432 |
| ColQwen2 DocVQA | 1/20 | 48.5 (15.4x) | 0.877 [0.845–0.909], seeds 0.859–0.888; top-1 0.438 | 0.897 [0.859–0.933]; top-1 0.448 | 0.938 [0.906–0.969]; top-1 0.481 | 0.765 [0.727–0.801], seeds 0.746–0.779; top-1 0.383 |
| ColQwen2 InfoVQA | 1/5 | 155.6 (4.7x) | 0.974 [0.962–0.985], seeds 0.970–0.976; top-1 0.850 | 0.985 [0.974–0.995]; top-1 0.864 | 0.989 [0.980–0.998]; top-1 0.864 | 0.963 [0.951–0.975], seeds 0.959–0.971; top-1 0.835 |
| ColQwen2 InfoVQA | 1/10 | 83.5 (8.8x) | 0.951 [0.935–0.965], seeds 0.949–0.953; top-1 0.821 | 0.961 [0.944–0.976]; top-1 0.838 | 0.973 [0.959–0.986]; top-1 0.852 | 0.927 [0.910–0.943], seeds 0.924–0.930; top-1 0.796 |
| ColQwen2 InfoVQA | 1/20 | 47.5 (15.4x) | 0.918 [0.897–0.937], seeds 0.916–0.920; top-1 0.780 | 0.928 [0.907–0.947]; top-1 0.796 | 0.960 [0.941–0.975]; top-1 0.834 | 0.881 [0.860–0.901], seeds 0.872–0.894; top-1 0.732 |
| ColQwen2.5 DocVQA | 1/5 | 158.7 (4.7x) | 0.954 [0.930–0.976], seeds 0.932–0.969; top-1 0.500 | 0.960 [0.934–0.985]; top-1 0.510 | 0.985 [0.962–1.006]; top-1 0.517 | 0.936 [0.907–0.961], seeds 0.913–0.955; top-1 0.487 |
| ColQwen2.5 DocVQA | 1/10 | 85.2 (8.8x) | 0.924 [0.894–0.951], seeds 0.909–0.942; top-1 0.483 | 0.956 [0.924–0.987]; top-1 0.514 | 0.972 [0.944–0.995]; top-1 0.514 | 0.871 [0.840–0.901], seeds 0.860–0.880; top-1 0.451 |
| ColQwen2.5 DocVQA | 1/20 | 48.5 (15.4x) | 0.888 [0.854–0.918], seeds 0.887–0.890; top-1 0.466 | 0.920 [0.884–0.953]; top-1 0.486 | 0.955 [0.925–0.984]; top-1 0.501 | 0.775 [0.737–0.810], seeds 0.765–0.791; top-1 0.400 |
| ColQwen2.5 InfoVQA | 1/5 | 155.6 (4.7x) | 0.971 [0.959–0.982], seeds 0.967–0.978; top-1 0.839 | 0.984 [0.972–0.995]; top-1 0.856 | 0.994 [0.985–1.003]; top-1 0.868 | 0.967 [0.955–0.978], seeds 0.961–0.971; top-1 0.836 |
| ColQwen2.5 InfoVQA | 1/10 | 83.5 (8.8x) | 0.964 [0.949–0.976], seeds 0.961–0.970; top-1 0.828 | 0.976 [0.962–0.989]; top-1 0.838 | 0.992 [0.980–1.003]; top-1 0.868 | 0.935 [0.918–0.950], seeds 0.930–0.939; top-1 0.800 |
| ColQwen2.5 InfoVQA | 1/20 | 47.5 (15.4x) | 0.933 [0.915–0.949], seeds 0.929–0.937; top-1 0.788 | 0.943 [0.925–0.960]; top-1 0.794 | 0.977 [0.962–0.990]; top-1 0.850 | 0.883 [0.863–0.903], seeds 0.872–0.892; top-1 0.738 |

## 6. Clustering against rescaling (paired differences in points; **M** = material)

| dataset | ratio | B − A (clustering) | C − B (rescaling) | A − D | B − D | C − D |
|---|---|---|---|---|---|---|
| ColPali DocVQA | 1/5 | +1.5 [-0.4, +3.4] | +1.3 [-0.3, +2.8] | +2.5 [+0.0, +5.1] **M** | +4.1 [+1.5, +6.9] **M** | +5.3 [+2.7, +8.2] **M** |
| ColPali DocVQA | 1/10 | +2.3 [+0.3, +4.6] **M** | +2.6 [+0.6, +4.5] **M** | +6.6 [+3.9, +9.3] **M** | +9.0 [+6.0, +12.1] **M** | +11.5 [+8.6, +14.5] **M** |
| ColPali DocVQA | 1/20 | +3.7 [+1.0, +6.5] **M** | +4.6 [+2.3, +7.0] **M** | +6.8 [+3.6, +10.2] **M** | +10.5 [+6.5, +14.5] **M** | +15.0 [+11.6, +18.8] **M** |
| ColPali InfoVQA | 1/5 | +0.8 [-0.1, +1.7] | +0.1 [-0.7, +1.1] | +1.8 [+0.7, +2.9] **M** | +2.5 [+1.3, +3.8] **M** | +2.7 [+1.3, +4.2] **M** |
| ColPali InfoVQA | 1/10 | +1.1 [+0.1, +2.0] **M** | +1.0 [-0.0, +2.0] | +2.6 [+1.2, +4.0] **M** | +3.7 [+2.2, +5.1] **M** | +4.6 [+3.0, +6.3] **M** |
| ColPali InfoVQA | 1/20 | +0.9 [-0.3, +2.2] | +1.4 [+0.1, +2.6] **M** | +3.9 [+2.4, +5.5] **M** | +4.8 [+2.9, +6.8] **M** | +6.2 [+4.2, +8.3] **M** |
| ColQwen2 DocVQA | 1/5 | +2.3 [+0.7, +4.1] **M** | +0.5 [-1.0, +2.4] | +3.9 [+1.4, +6.3] **M** | +6.2 [+3.6, +8.9] **M** | +6.7 [+4.5, +9.3] **M** |
| ColQwen2 DocVQA | 1/10 | +3.1 [+0.8, +5.6] **M** | +0.9 [-1.0, +2.6] | +8.4 [+5.6, +11.2] **M** | +11.5 [+8.2, +15.2] **M** | +12.4 [+9.1, +15.7] **M** |
| ColQwen2 DocVQA | 1/20 | +2.0 [-0.4, +4.3] | +4.0 [+2.0, +6.2] **M** | +11.2 [+8.2, +14.5] **M** | +13.2 [+9.6, +16.9] **M** | +17.2 [+13.8, +20.8] **M** |
| ColQwen2 InfoVQA | 1/5 | +1.1 [+0.3, +1.9] **M** | +0.4 [-0.3, +1.3] | +1.1 [+0.1, +2.1] **M** | +2.2 [+1.0, +3.3] **M** | +2.6 [+1.5, +3.7] **M** |
| ColQwen2 InfoVQA | 1/10 | +1.0 [+0.0, +2.0] | +1.2 [+0.3, +2.2] **M** | +2.4 [+1.1, +3.8] **M** | +3.4 [+1.8, +5.1] **M** | +4.6 [+3.0, +6.2] **M** |
| ColQwen2 InfoVQA | 1/20 | +1.0 [-0.1, +2.2] | +3.1 [+1.9, +4.4] **M** | +3.7 [+2.1, +5.5] **M** | +4.8 [+3.0, +6.6] **M** | +7.9 [+6.1, +9.7] **M** |
| ColQwen2.5 DocVQA | 1/5 | +0.7 [-1.1, +2.5] | +2.4 [+0.8, +4.2] **M** | +1.8 [-0.6, +3.9] | +2.5 [-0.4, +5.0] | +4.9 [+2.1, +7.5] **M** |
| ColQwen2.5 DocVQA | 1/10 | +3.2 [+1.0, +5.4] **M** | +1.5 [-0.5, +3.6] | +5.3 [+2.9, +8.0] **M** | +8.6 [+5.4, +11.7] **M** | +10.1 [+6.9, +13.2] **M** |
| ColQwen2.5 DocVQA | 1/20 | +3.2 [+0.9, +5.3] **M** | +3.5 [+1.3, +5.7] **M** | +11.3 [+8.5, +14.4] **M** | +14.4 [+10.8, +18.0] **M** | +18.0 [+14.4, +21.6] **M** |
| ColQwen2.5 InfoVQA | 1/5 | +1.3 [+0.2, +2.4] **M** | +1.0 [+0.3, +1.8] | +0.5 [-0.6, +1.5] | +1.8 [+0.6, +3.1] **M** | +2.7 [+1.6, +4.0] **M** |
| ColQwen2.5 InfoVQA | 1/10 | +1.3 [+0.3, +2.3] **M** | +1.6 [+0.7, +2.4] **M** | +2.9 [+1.6, +4.2] **M** | +4.1 [+2.7, +5.7] **M** | +5.7 [+4.1, +7.3] **M** |
| ColQwen2.5 InfoVQA | 1/20 | +1.0 [-0.3, +2.3] | +3.4 [+2.2, +4.5] **M** | +5.0 [+3.4, +6.7] **M** | +6.0 [+4.1, +7.7] **M** | +9.4 [+7.5, +11.3] **M** |

**Reading** (MEASURED differences, controlled comparisons only):
- **Clustering algorithm (B − A).** Ward beats K-means, both with plain means, by
  +0.7 to +3.7 points.
  - It is material in 10 of 18 cells.
  - It is largest on DocVQA at 1/10 and 1/20 (+2.0 to +3.7).
- **Rescaling (C − B).** The norm rescaling adds +0.1 to +4.6 points.
  - It is material in 10 of 18 cells.
  - It grows with compression: at 1/20 it is material in all 6 datasets (+1.4 to
    +4.6).
- **Size of each effect.** At 1/10 and 1/20 the two effects are of similar size. At
  1/5 both are small (≤ 2.4 points).
- **Together (C − A),** they account for +0.9 to +8.2 points of retention at the same
  vector count.

## 7. Structured merging against random pruning

- **Every merging method retains more than random pruning.** In 51 of 54
  merging-minus-random comparisons the interval excludes 0, and all 54 point
  estimates are positive.
- **The gap grows with compression:**
  - A − D: +0.5 to +8.4 points at 1/5 and 1/10; up to +11.3 at 1/20;
  - C − D: +2.6 to +18.0 points.
- **The three non-material cells** are on ColQwen2.5 at 1/5:
  - A − D on InfoVQA, +0.5 [−0.6, +1.5];
  - A − D on DocVQA, +1.8 [−0.6, +3.9];
  - B − D on DocVQA, +2.5 [−0.4, +5.0].
- **Seed variability is material** in several cells. The across-seed range exceeds 1
  point for:
  - **A:** up to 5.0 points (ColQwen2 DocVQA 1/10: 0.911–0.961);
  - **D:** up to 5.9 points (ColQwen2 DocVQA 1/5).

  Single-seed results for K-means or random pruning can therefore differ by several
  points from a seed average. All three seeds are reported; none was selected.

## 8. Comparison with SAP's published values

**Sources.** SAP's values are EXTERNAL: Tables 7 and 8 of arXiv 2601.20107v3,
transcribed from the HTML (`Q7/REPORT.md` §4) and not PDF-verified. SAP has no
ColQwen2.5 rows. "change vs Q7" compares |A − SAP Cluster| with Q7's |C − SAP
Cluster| (reduced or increased if they differ by ≥ 1 point).

| dataset | ratio (SAP γ) | E7a A K-means plain | SAP Cluster | A − SAP | Q7 gap (C − SAP) | change vs Q7 | E7a D random | SAP Random | D − SAP |
|---|---|---|---|---|---|---|---|---|---|
| ColPali DocVQA | 1/5 (0.20) | 94.76 | 93.44 | +1.32 | +4.15 | reduced | 92.24 | 91.65 | +0.59 |
| ColPali DocVQA | 1/10 (0.10) | 90.51 | 86.09 | +4.42 | +9.35 | reduced | 83.89 | 85.03 | -1.14 |
| ColPali DocVQA | 1/20 (0.05) | 81.44 | 76.05 | +5.39 | +13.61 | reduced | 74.63 | 73.58 | +1.05 |
| ColPali InfoVQA | 1/5 (0.20) | 98.93 | 98.04 | +0.89 | +1.78 | unchanged | 97.17 | 96.37 | +0.80 |
| ColPali InfoVQA | 1/10 (0.10) | 96.98 | 94.64 | +2.34 | +4.39 | reduced | 94.41 | 93.16 | +1.25 |
| ColPali InfoVQA | 1/20 (0.05) | 94.97 | 87.10 | +7.87 | +10.17 | reduced | 91.04 | 89.42 | +1.62 |
| ColQwen2 DocVQA | 1/5 (0.20) | 97.00 | 94.74 | +2.26 | +5.08 | reduced | 93.11 | 89.04 | +4.07 |
| ColQwen2 DocVQA | 1/10 (0.10) | 93.65 | 87.41 | +6.24 | +10.26 | reduced | 85.30 | 77.75 | +7.55 |
| ColQwen2 DocVQA | 1/20 (0.05) | 87.73 | 75.37 | +12.36 | +18.40 | reduced | 76.52 | 62.83 | +13.69 |
| ColQwen2 InfoVQA | 1/5 (0.20) | 97.38 | 95.89 | +1.49 | +3.05 | reduced | 96.32 | 93.53 | +2.79 |
| ColQwen2 InfoVQA | 1/10 (0.10) | 95.07 | 89.70 | +5.37 | +7.56 | reduced | 92.69 | 89.14 | +3.55 |
| ColQwen2 InfoVQA | 1/20 (0.05) | 91.83 | 79.25 | +12.58 | +16.70 | reduced | 88.09 | 82.52 | +5.57 |

**This does NOT reproduce SAP.** The checkpoint handling, query set, vector-count
definition (visual tokens against OptiVision's free vectors), evaluation code,
preprocessing, K-means variant and seeds are not known to match.

**What the comparison shows:**
- **Using K-means with plain means instead of Ward with rescaling** reduces the gap to
  SAP's Cluster row in 11 of 12 cells (unchanged in 1). At γ = 0.10 the gap falls from
  +4.4 to +10.3 points to +2.3 to +6.2.
- **A gap remains:**
  - only 3 of the 8 cells at γ = 0.20 and 0.10 are within 2 points;
  - at γ = 0.05, A is still 5.4–12.6 points above SAP's Cluster.
- **The method-independent control (random pruning, D) behaves differently by model:**
  - **ColPali:** D is within 0.6–1.6 points of SAP's Random in all 6 cells.
  - **ColQwen2:** D is 2.8–13.7 points above SAP's Random (DocVQA: +4.1, +7.6,
    +13.7; InfoVQA: +2.8, +3.6, +5.6).

## 9. What E7a does and does not explain (INFERENCE from §6–8)

**It explains part of the gap.**
- Within OptiVision's data, the two mechanism choices together are worth +0.9 to +8.2
  points at equal vector count (clustering +0.7 to +3.7; rescaling +0.1 to +4.6).
- Replacing them with K-means and plain means removes roughly 30–55% of the Q7 gap at
  γ = 0.10.
- So part of the original disagreement is attributable to centroid construction and
  the clustering algorithm.

**For ColPali,** the evaluation appears aligned: the baselines agree within 0.6 points
and random pruning agrees within 1.6 points. A **remaining 0.9–7.9-point gap** between
E7a's K-means and SAP's Cluster row is therefore probably specific to the
mechanism or its implementation. Candidates, not testable here:
- the K-means variant (spherical or Euclidean, initialization, iterations);
- SAP's seeds (E7a shows a seed spread of up to 2.6 points on ColPali);
- SAP's token selection.

**For ColQwen2,** the random-pruning control itself disagrees, by up to 13.7 points.
A method-independent control cannot disagree because of the compression mechanism, so
something upstream must differ, for example:
- the evaluation set;
- the number and resolution of visual tokens per page (ColQwen2 uses dynamic
  resolution);
- the treatment of non-visual tokens;
- the evaluation code.

The 1–1.6-point gap in the uncompressed baselines (Q7) points the same way.

**It does not explain** whether SAP's own pruning method behaves differently from
anything measured here. SAP itself was not run.

## 10. Limitations

- **Not a reproduction.** E7a uses OptiVision's vector-count definition, OptiVision's
  K-means implementation and three seeds, not SAP's.
- **SAP's values are transcribed** from the arXiv HTML and not PDF-verified.
- **Coverage:** two corpora, ViDoRe V1 DocVQA and InfoVQA only; no ColQwen2.5 values
  on SAP's side.
- **Seed variability** for K-means and random pruning is material in some cells. E7a
  reports means over three seeds.
- **Storage** was not measured, only vector counts.

## 11. Recommendation

The rule was fixed in `PLAN.md` §7 before the results:
- the A gap mostly closes in only **3 of 8** cells;
- the random-control gap has a **median of 2.2 points and a maximum of 13.7**, above
  the 2-point threshold.

The rule therefore gives **E7c (official ViDoRe evaluation alignment)**, not E7b and not
yet "stop Q7".

**Why this follows from the evidence:**
- The ColQwen2 disagreement survives a method-independent control. That is a property
  of the evaluation pipeline, not of any compression method.
- Running SAP itself (E7b, GPU) before aligning the evaluation could not separate
  "SAP's method" from "different data or evaluation".
- E7c is CPU-only. It would determine which part of the ColQwen2 gap disappears under
  the official toolkit's query sets and preprocessing.
- E7b is justified only if E7c aligns the evaluation and a question about SAP's
  pruning method itself remains.

**Alternative:** if aligning with SAP's exact operating point is not worth the effort,
**stopping Q7** is defensible on what E7a has established:
- the mechanism attribution (clustering and rescaling each matter);
- the merging-over-random result;
- for ColPali, an evaluation that agrees on the method-independent control.

The ColQwen2 discrepancy would then stay documented as unresolved.

E7b, E7c and Q8 have not been started.
