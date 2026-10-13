# E7a plan: attribution of the OptiVision against SAP disagreement (frozen before any E7a result)

Specified in `reports/research/Q7/REPORT.md` §7 (Q7.8). Branch
`research-investigation`, after `24d2707`.
- CPU only, existing vectors.
- No library change, no new dependency (numpy and scipy are already used), no GPU, no
  re-encoding, no external code.
- Not a reproduction of SAP, and not a leaderboard.

## 1. Data and evaluation (unchanged from Q4)

- **Datasets:** the six Q4 datasets (ColPali, ColQwen2 and ColQwen2.5 × ViDoRe V1
  DocVQA and InfoVQA), loaded through the same `e1_common` / s7b loader as Q4. That is
  500 pages, with 451 (DocVQA) and 494 (InfoVQA) labelled queries.
- **Scoring:** exact MaxSim (`optivision.scoring.maxsim_matrix`) over the compressed
  page vectors, with unchanged float32 query vectors.
- **No other changes:** no quantization, projection or candidate pruning.
- **Metric:** nDCG@5 (`per_query_metrics`).
  - Retention = mean(compressed) / mean(uncompressed), using the library's
    `retention` (paired bootstrap 95% interval, 1,000 resamples, seed 0).
  - Top-1 accuracy = the share of queries whose labelled page ranks first, using
    `rank()`'s tie rule.
- **Baseline:** the uncompressed float32 index. It must equal the E1 store's
  per-query baseline exactly (a check).

## 2. Vector-count convention (exact)

All stored vectors have unit L2 norm. Each page's vectors are split by the corpus's
existing `protected` mask.
- **Protected (non-patch, special) tokens:** 7 per page for ColPali, 11 for ColQwen2.
  For ColQwen2.5 the count is recorded.
- **Free (patch) vectors:** the remaining n_free.

For a target ratio r ∈ {1/5, 1/10, 1/20}, every method produces **exactly k = ceil(r ×
n_free) vectors from the free vectors** (minimum 1), and all four methods keep the
protected vectors unchanged.
- This is the library's own convention (`TokenReducer` with `_budget`, as in Q4's
  `HierarchicalMerge(ratio)`).
- It is the same denominator for every method. The kept count per page is
  k + n_protected, and the exact totals are recorded.

**Relation to SAP.** SAP's γ is defined as a fraction of *visual* tokens. Its handling
of non-visual tokens, its preprocessing and its evaluation set are not known. E7a
therefore tests compression *mechanisms* at OptiVision's vector-count definition, and
**does not reproduce SAP's operating point**.

## 3. Methods (applied to each page's free vectors; protected vectors appended unchanged)

**A. K-means, plain mean.**
- Lloyd's algorithm in numpy, Euclidean distance on the stored (unit) vectors.
- k-means++ initialization with `np.random.default_rng([seed, page_index])`.
- At most 50 iterations, stopping early when the assignments do not change.
- An empty cluster is re-seeded with the point farthest from its current centroid
  (deterministic, applied before the means are recomputed).
- The output vector is the arithmetic mean of the members, with no rescaling.

**B. Ward, plain mean.**
- `scipy.cluster.hierarchy.linkage(method="ward")` on the unit vectors, exactly the
  library's input at 128-d, then `fcluster(..., k, "maxclust")`.
- The output vector is the arithmetic mean of the members.

**C. Ward, OptiVision rescaling.**
- The same labels as B, followed by the library's `stages.merge._pool`, called
  unchanged (unit weights): the mean rescaled to the members' mean norm.
- Check: at r = 1/10, C must reproduce Q4's `hierarchical_merge(0.1)` per-query
  nDCG@5 exactly on every dataset. Otherwise stop.

**D. Random pruning.**
- Keep k of the free vectors, chosen uniformly without replacement with
  `np.random.default_rng([seed, page_index])`. No merging.

**Seeds (fixed now):** {0, 1, 2} for A and D. Each A and D cell reports the
mean retention over the three seeds, plus the per-seed values and their range. If
the across-seed range exceeds 1 point, it is reported as material. B and C are
deterministic.

## 4. Measurements per dataset × ratio × method (× seed)

- the exact total and per-page vector counts (kept free + protected);
- the vector compression factor = original vectors / kept vectors;
- nDCG@5, retention with its interval, and top-1 accuracy.

Storage bytes are **not** reported. Only vector counts change, and no storage was
measured.

## 5. Attribution comparisons (paired bootstrap 95% intervals over queries, seed 0, 1,000 resamples)

For A and D, the per-query nDCG@5 is averaged over the three seeds.

| comparison | what it isolates |
|---|---|
| B − A | clustering algorithm (Ward against K-means), with plain means |
| C − B | the rescaling of merged vectors |
| A − D, B − D, C − D | structured merging against keeping random original vectors |

A difference is called **material** if its interval excludes 0 and its size is
≥ 1 point.

## 6. Relation to SAP's published values (only after §4–5 are complete)

**SAP's values.** From SAP Tables 7 and 8, transcribed in `Q7/REPORT.md` §4 (arXiv
HTML v3, not PDF-verified). They cover ColPali and ColQwen2 × DocVQA and InfoVQA, at
γ ∈ {0.20, 0.10, 0.05}, for the "Cluster" (K-means) and "Random" rows. SAP has no
ColQwen2.5 rows.

**For each cell, report:**
- E7a's K-means/plain retention (A) against SAP's Cluster;
- E7a's random pruning (D) against SAP's Random;
- the absolute difference.

**Reduced / unchanged / increased.** Q7 compared OptiVision's Ward-rescaled row (here
C) with SAP's Cluster row. Compare |A − SAP Cluster| with |C − SAP Cluster|:
- **reduced** if smaller by ≥ 1 point;
- **increased** if larger by ≥ 1 point;
- **unchanged** otherwise.

**This is not a reproduction of SAP.** The checkpoint conventions, query set,
vector-count definition, evaluation code and preprocessing are not known to match.

## 7. Recommendation rule (fixed now, applied to the §6 outcome)

- **The A–SAP Cluster gap mostly closes** (|A − SAP| ≤ 2 points in at least 6 of the 8
  ColPali / ColQwen2 cells at γ = 0.10 and 0.20).
  - The disagreement is then largely attributable to centroid construction and
    clustering.
  - Recommendation: **stop Q7 and move to Q8**, unless the SAP method itself is
    wanted.
- **The gap persists for A, and also for D against SAP Random** (median absolute gap
  > 2 points). Then even the method-independent control disagrees, which points to
  evaluation or preprocessing differences. Recommendation: **E7c** (evaluation
  alignment, CPU) before any E7b.
- **Mixed outcome:** report it as mixed, and recommend E7c if the random-control gap
  is > 2 points. Otherwise recommend stopping Q7.
- **E7b** (GPU re-encoding with SAP itself) is recommended only if E7c aligns the
  evaluation and a question about SAP's own method remains.

## 8. Stop conditions

Stop and report if:
- K-means cannot meet the definition;
- k cannot be matched exactly;
- data for any dataset is missing;
- GPU or re-encoding would be needed;
- the C reproduction check fails;
- the runtime grows well beyond about 2 hours of CPU.
