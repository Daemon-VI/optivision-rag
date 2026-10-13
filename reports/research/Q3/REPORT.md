# Q3 report: why does OptiVision behave differently on DocVQA and InfoVQA?

Branch `research-investigation`. Pre-registration: `PLAN.md` (`dc9cc90`).
Intervention, pre-stated: `INTERVENTION.md` (`cce335f`, one wording fix in
`cc4ec11` before it ran).

Data:
- the E1 per-query store at `b18ae19`, read only;
- per-query and per-page features recomputed from the existing vectors
  (`features/`). For all six VLM datasets, the per-query nDCG from the recomputed
  scores equals the E1 store exactly, for the baseline and both scored
  representative candidates.

Within-DocVQA margin split, pre-stated: `SPLIT.md` (`5734903`), results in §18;
Q3 status in §19.

Numbers: `observational.json`, `intervention/*.json`, `split/*.json`. Labels: **MEASURED**
(computed from encoder outputs), **SIMULATED** (resampling), **INFERENCE**.

## 1. Research question

The optimizer selects much less compression on DocVQA than on InfoVQA, for every
encoder (R7b, R11, E1). For example, frozen method C at T = 0.95 and n = 225 chose
3.9x against 72.7x on ColPali. Is that because DocVQA is intrinsically less
compressible, because it is harder to certify, or because of measurable
query/corpus properties?

## 2. Competing hypotheses

H3a (intrinsically less compressible), H3b (noisier, harder to certify), H3c
(query/corpus characteristics), H3d (several factors, none sufficient alone).
The operational definitions were fixed in `PLAN.md` before any computation.

## 3. Existing evidence (before Q3)

- **Baseline nDCG@5:** DocVQA 0.58–0.62, InfoVQA 0.85–0.92 (R11).
- **Within-page nearest-neighbour cosine:** similar on the two splits (R8, R11).
- **Selected compression:** consistently lower on DocVQA (E1).

## 4. Dataset and model characteristics (MEASURED)

| | ColPali Doc / Info | ColQwen2 Doc / Info | ColQwen2.5 Doc / Info |
|---|---|---|---|
| queries / pages | 451 / 494, 500 pages each | same | same |
| labelled pages per query | 1 / 1 | 1 / 1 | 1 / 1 |
| pages with no labelled query | 49 / 6 | 49 / 6 | 49 / 6 |
| vectors per page (median) | 1031 / 1031 | 755 / 747 | 755 / 747 |
| query token vectors (median) | 20 / 24 | 19 / 23 | 19 / 23 |

**Resource objectives.**
- Storage, RAM and the bytes per page of a pipeline are the same on the two
  splits, up to the ≤ 1% difference in vectors per page. So the difference
  studied here is entirely in **quality retention and how well it can be
  certified**, not in storage or RAM accounting.
- Search latency was not measured in Q3.

## 5. Per-query baseline difficulty (3A, MEASURED)

| | mean nDCG@5 [95% CI] | hit@1 | labelled page outside top 5 | rank: median / q75 / q90 | relevance margin, median (per token) | top-2 margin, median |
|---|---|---|---|---|---|---|
| ColPali · DocVQA | 0.584 [0.546, 0.624] | 49.4% | 33.9% | 2 / 20 / 141 | −0.0007 | 0.017 |
| ColPali · InfoVQA | 0.846 [0.818, 0.873] | 77.5% | 9.7% | 1 / 1 / 5 | 0.085 | 0.086 |
| ColQwen2 · DocVQA | 0.606 [0.565, 0.647] | 52.3% | 32.4% | 1 / 11 / 115 | 0.010 | 0.038 |
| ColQwen2 · InfoVQA | 0.920 [0.898, 0.939] | 88.7% | 5.1% | 1 / 1 / 2 | 0.148 | 0.148 |
| ColQwen2.5 · DocVQA | 0.618 [0.576, 0.660] | 53.4% | 30.6% | 1 / 9 / 106 | 0.013 | 0.027 |
| ColQwen2.5 · InfoVQA | 0.917 [0.895, 0.936] | 87.2% | 4.5% | 1 / 1 / 2 | 0.134 | 0.134 |

DocVQA − InfoVQA, with bootstrap 95% intervals:

| | mean nDCG@5 | hit@1 | median relevance margin |
|---|---|---|---|
| ColPali | −0.262 [−0.313, −0.212] | −0.281 [−0.343, −0.221] | −0.086 [−0.102, −0.071] |
| ColQwen2 | −0.313 [−0.358, −0.265] | −0.363 [−0.417, −0.308] | −0.138 [−0.156, −0.109] |
| ColQwen2.5 | −0.299 [−0.348, −0.253] | −0.338 [−0.396, −0.282] | −0.122 [−0.143, −0.105] |

DocVQA has a markedly different difficulty distribution on every encoder. About
a third of its queries already miss the top 5 uncompressed, and half of them sit
at a relevance margin near zero. Margins are compared between splits within a
model only (per-token cosine units); they are not compared across models.

## 6. Compression sensitivity per query (3B, MEASURED)

Pool retention R (finite query-pool truth) for the same pipeline, with the
certification standard error at n = 225 and its parts. The SE ratio is DocVQA /
InfoVQA; it is the product of the numerator-spread ratio and InfoVQA/DocVQA mean
baseline nDCG.

| model · candidate | R DocVQA | R InfoVQA | Doc − Info [95% CI] | SE ratio | spread ratio | denominator ratio | drop-out rate D / I | gain rate D / I |
|---|---|---|---|---|---|---|---|---|
| ColPali · Ward ⅓ + int4 | 0.981 | 0.998 | −0.017 [−0.042, 0.008] | 2.66 | 1.83 | 1.45 | 3.1% / 0.4% | 5.3% / 2.6% |
| ColPali · binary | 0.963 | 0.975 | −0.012 [−0.043, 0.017] | 1.91 | 1.32 | 1.45 | 4.4% / 2.2% | 5.5% / 3.0% |
| ColPali · Ward ¼ + binary | 0.954 | 0.975 | −0.021 [−0.054, 0.011] | 2.04 | 1.41 | 1.45 | 4.2% / 2.0% | 5.3% / 3.6% |
| ColQwen2 · Ward ⅓ + int4 | 0.990 | 0.995 | −0.005 [−0.024, 0.017] | 1.83 | 1.20 | 1.52 | 1.8% / 0.4% | 5.3% / 2.8% |
| ColQwen2 · binary | 0.972 | 0.990 | −0.019 [−0.048, 0.011] | 2.98 | 1.96 | 1.52 | 4.2% / 0.4% | 6.2% / 1.8% |
| ColQwen2 · Ward ¼ + binary | 0.961 | 0.976 | −0.015 [−0.046, 0.013] | 2.08 | 1.37 | 1.52 | 4.4% / 1.6% | 6.7% / 2.2% |
| ColQwen2.5 · Ward ⅓ + int4 | 0.993 | 0.998 | −0.005 [−0.024, 0.015] | 2.91 | 1.96 | 1.48 | 1.6% / 0.2% | 6.2% / 2.6% |
| ColQwen2.5 · binary | 0.995 | 0.991 | +0.004 [−0.028, 0.038] | 3.22 | 2.17 | 1.48 | 4.0% / 0.8% | 9.3% / 3.2% |
| ColQwen2.5 · Ward ¼ + binary | 0.981 | 0.982 | −0.001 [−0.031, 0.029] | 2.15 | 1.45 | 1.48 | 4.4% / 2.0% | 8.6% / 3.0% |

Over all 40 candidates:

| | candidates with R_Doc < R_Info beyond CI | with R_Doc > R_Info beyond CI | SE ratio, median (range) | spread ratio, median |
|---|---|---|---|---|
| ColPali | 12 | 0 | 2.07 (1.59–3.70) | 1.43 |
| ColQwen2 | 0 | 0 | 2.12 (1.62–8.28) | 1.40 |
| ColQwen2.5 | 3 | 0 | 2.63 (1.98–3.79) | 1.77 |

**Failure mode** (rule fixed in `PLAN.md`):
- DocVQA is "C: mixture" for 35–37 of 40 candidates. The worst 10% of losing
  queries carry only about 20–26% of the loss mass, so losses are **not** a few
  catastrophic queries.
- InfoVQA is "A: widespread small" for 13–22 of 40 candidates, and "C: mixture"
  for most of the rest.
- DocVQA queries **both lose and gain more often** under compression: they churn.

## 7. Difficulty against compression sensitivity (3C, MEASURED, descriptive)

**By baseline rank of the labelled page.** Loss rate under `binary`, DocVQA /
InfoVQA:

| | rank 1 | ranks 2–5 |
|---|---|---|
| ColPali | 6.7% / 4.2% | 42.7% / 34.9% |
| ColQwen2 | 6.8% / 2.5% | 40.6% / 19.4% |
| ColQwen2.5 | 7.1% / 3.0% | 29.2% / 29.3% |

A query whose page is not first is roughly 4–10 times more likely to lose under
compression, on both splits.

**By relevance-margin quintile** (edges pooled per model):
- Losses sit almost entirely in the two lowest quintiles on both splits. They are
  0–3% in the top three quintiles.
- DocVQA puts **56–58%** of its queries in the two lowest quintiles, against
  **23–26%** for InfoVQA.
- *Within* a quintile, DocVQA is not consistently more fragile. For example,
  `binary` in quintile 2 loses 18.7% vs 43.6% (ColPali), 21.0% vs 15.4%
  (ColQwen2) and 16.8% vs 24.1% (ColQwen2.5).

**Rank correlation** between the per-query margin and the loss, among queries
with b > 0: weak. Spearman ρ is between −0.22 and +0.12, and most intervals
include 0. The relationship is a threshold (only near-boundary queries lose), not
a monotone trend.

## 8. Score margins before and after compression (3D, MEASURED)

**A comparability problem, found and handled.** Binary codes decode to ±1
vectors, so post-compression score magnitudes are not on the float baseline's
scale. Margin *changes* are therefore not reported; that computation was removed
from the analysis. Only scale-free quantities are compared: the share of queries
whose labelled page leaves rank 1, among baseline hits.

| | `binary` DocVQA / InfoVQA | `Ward ¼ + binary` DocVQA / InfoVQA |
|---|---|---|
| ColPali | 6.7% / 4.2% | 9.4% / 5.5% |
| ColQwen2 | 6.8% / 2.5% | 8.5% / 4.8% |
| ColQwen2.5 | 7.1% / 3.0% | 7.1% / 3.7% |

## 9. Relevance structure (3E, MEASURED)

- Every ViDoRe query has exactly one labelled page by construction, so relevance
  is never spread over several labelled pages. The nDCG@5 behaviour is the same
  on both splits.
- **Pages with no labelled query:** 49 on DocVQA against 6 on InfoVQA. These are
  pages whose question repeats an earlier one, so they may be relevant but
  unlabelled.
- **How often such a page is ranked first when the labelled page is not:**
  DocVQA 6.1% / 7.4% / 12.9% (ColPali / ColQwen2 / ColQwen2.5), InfoVQA 2.7% /
  0.0% / 1.6%.
- Label incompleteness therefore explains at most a small share of DocVQA's
  baseline misses.
- SciFact (several relevant documents, nDCG@10) is reported separately in
  `observational.json` and is not pooled.

## 10. Vector and page geometry (3F, MEASURED)

| | ColPali Doc / Info | ColQwen2 Doc / Info | ColQwen2.5 Doc / Info |
|---|---|---|---|
| within-page nearest-neighbour cosine (existing reports) | 0.927 / 0.934 | 0.886 / 0.873 | 0.904 / 0.863 |
| page-to-nearest-page cosine of mean vectors, median | 0.718 / 0.678 | 0.769 / 0.681 | 0.801 / 0.690 |

Within-page redundancy is similar on the two splits. **DocVQA pages are more
alike each other** (a more homogeneous corpus: forms and letters), and most
clearly so for the ColQwen encoders. Ward merging retention is in §6. The 2,560-d
model is unavailable here: its vectors were not kept.

## 11. Controlled intervention (3G)

**Chosen from the observations.** The two strongest explanations were:
- **X, composition:** DocVQA has many more near-boundary queries, and these lose
  and gain under compression, inflating the certification SE.
- **Y, intrinsic fragility at equal difficulty.**

**Intervention:** reweight InfoVQA queries to DocVQA's distribution over the
pooled relevance-margin quintiles. Then re-measure pool retention, the SE, and
selection with the frozen E1 code, on 2,000 resamples drawn from the reweighted
pool.

## 12. Results

**Margin matching** (primary). The SE ratio is the median over candidates of SE_Doc / SE_Info.

| | effective InfoVQA queries | SE ratio before → after | log SE gap closed | candidates with R_Doc ≠ R_matched beyond CI (lower / higher) |
|---|---|---|---|---|
| ColPali | 252 of 494 | 2.07 → 1.15 | **81%** | 1 / 2 |
| ColQwen2 | 292 of 494 | 2.12 → 1.20 | **76%** | 0 / 2 |
| ColQwen2.5 | 288 of 494 | 2.63 → 1.46 | **61%** | 0 / 0 |

**Selection at n = 225** (SIMULATED resampling of MEASURED per-query results).
Median compression (deployment rate):

| | T | A: DocVQA / InfoVQA / matched | C@0.05: DocVQA / InfoVQA / matched |
|---|---|---|---|
| ColPali | 0.95 | 7.8x / 72.7x / 14.0x | 3.9x / 72.7x / 15.4x |
| ColPali | 0.97 | 3.9x / 30.4x / 7.8x | 1.0x (1%) / 1.0x (50%) / 1.0x (14%) |
| ColQwen2 | 0.95 | 11.6x / 63.0x / 19.6x | 7.8x / 63.0x / 22.8x |
| ColQwen2 | 0.97 | 3.9x / 27.6x / 7.8x | 1.0x (1%) / 7.8x (100%) / 1.0x (25%) |
| ColQwen2.5 | 0.95 | 10.6x / 76.0x / 22.8x | 7.8x / 76.0x / 22.8x |
| ColQwen2.5 | 0.97 | 7.8x / 29.7x / 9.6x | 1.0x (3%) / 9.6x (100%) / 1.0x (19%) |

Matched-InfoVQA miss rates stayed at or below 0.3% for every method shown. C's
claimed level held on the reweighted distribution as well.

**Pre-stated reading.**
- ColPali and ColQwen2 close ≥ 75% of the SE gap. A's selection lands within a
  factor of 2 of DocVQA's at both targets, but C's at T = 0.95 does not (3.9x and
  2.9x apart).
- ColQwen2.5 closes 61%.
- By the rule fixed in advance, all three models are **"partial; both
  contribute"** (H3d).

In log terms, matching removes:
- 61–74% of the A selection gap at T = 0.95: ColPali 9.3x → 1.8x, ColQwen2
  5.4x → 1.7x, ColQwen2.5 7.2x → 2.2x;
- 49–53% of the C gap: 18.6x → 3.9x, 8.1x → 2.9x, 9.7x → 2.9x.

**Robustness: rank matching** (strata: rank 1, 2–5, > 5).
- It closes 60–88% of the SE gap, but leaves matched-InfoVQA selection close to
  unmatched InfoVQA (for example A: 49.7x against 72.7x and DocVQA's 7.8x).
- It also *raises* matched-InfoVQA retention above DocVQA's beyond the intervals
  for 12–23 candidates.
- Why (INFERENCE): rank matching upweights InfoVQA queries whose page is outside
  the top 5 (baseline nDCG = 0). Those contribute nothing to the denominator and
  can only gain, so they inflate retention.
- Rank strata are therefore too coarse a matching variable. Within rank 1,
  margins differ widely, and the relevance margin captures what rank strata miss.
  This divergence is reported, not used for the reading.

## 13. Interpretation (INFERENCE)

1. **The optimizer's difference is driven far more by certification difficulty
   than by retention.**
   - For the same pipeline, pool retention is lower on DocVQA by 0–2 points,
     usually within the intervals (0–12 of 40 candidates beyond them).
   - The standard error with which retention can be estimated from 225 queries is
     about 2–2.6x larger on DocVQA, on every encoder.
   - Both `calibrate()`'s bound and method C pay for that SE directly.
2. **The SE gap has two measured parts of similar size.**
   - A lower mean baseline nDCG, the denominator (factor 1.45–1.52).
   - A larger per-query spread, the numerator (median factor 1.40–1.77).
   - The spread comes from churn: near-boundary queries both lose and gain under
     compression.
3. **Both parts trace largely to one measured property: the relevance-margin
   distribution.**
   - DocVQA has 56–58% of its queries in the two lowest margin quintiles, against
     23–26% for InfoVQA.
   - At equal margin, compression losses are similar between the splits.
   - Reweighting InfoVQA to DocVQA's margins reproduces 61–81% of the SE gap, and
     most of the selection gap for the current rule.
4. **A residual remains** (SE ratio 1.15–1.46 after matching; C still 2.9–3.9x
   apart at T = 0.95).
   - Candidates, none tested: finer margin structure within a quintile; DocVQA's
     more homogeneous corpus (pages closer to each other); its unlabelled
     duplicate pages; and, among rank-1 queries, a higher top-1 loss rate under
     binary codes (6.7–7.1% against 2.5–4.2%).
   - These are not separated by the existing data.

## 14. What the evidence supports

- **"DocVQA receives less compression" is MEASURED and robust**, on all three
  encoders.
- **H3b is supported.** DocVQA's quality retention is substantially harder to
  certify at a fixed number of calibration queries: SE about 2–2.6x, on every
  encoder.
- **H3c is supported for one property in particular.** The relevance-margin
  distribution, a property of the queries and corpus under the uncompressed
  model, accounts for most of the certification gap and most of the current
  rule's selection gap (observational matching, not causal proof).
- **H3d is supported by the pre-stated reading.** Margin composition explains
  most but not all of the difference.
- **The pattern is model-independent in direction** across ColPali, ColQwen2 and
  ColQwen2.5. The size of the residual varies by encoder (largest for
  ColQwen2.5 by matching; the within-DocVQA split places it on ColPali instead, so
  the ordering is not established, §18.6).

## 15. What the evidence does NOT support

- **"DocVQA is intrinsically less compressible"** (H3a) as the main explanation.
  - Retention differences for the same pipeline are small and mostly within the
    intervals, and after margin matching they almost vanish (0–1 candidates lower
    beyond the intervals).
  - Some residual fragility is not excluded (ColPali's 12 unmatched candidates;
    the rank-1 binary losses).
- **"InfoVQA is easier"** in a sense beyond the precise one measured here:
  higher baseline nDCG@5, hit@1 and relevance margins under these encoders and
  labels.
- **Any causal claim.** Matching on one variable cannot exclude variables
  correlated with it.
- Anything about storage, RAM or latency differences. Those are the same per
  pipeline (bytes) or were not measured (latency).
- Anything about other datasets, other targets of the analysis, or the 2,560-d
  model.

## 16. Limitations

- **Two corpora**, so dataset-level conclusions rest on one contrast. "Model
  independence" means three encoders on the same two corpora.
- **One labelled page per query.** The unlabelled duplicate pages on DocVQA (49)
  are a known label-noise source. They were counted, not corrected.
- **Margins are per-token cosine units**, compared within a model only.
  Post-compression margin magnitudes were not comparable (binary scale) and were
  dropped.
- **Matching reduces the effective sample to 252–292 InfoVQA queries.** Quintile
  matching is coarse.
- **The rank-matching robustness check diverged** (§12), so the result depends on
  the matching variable.
- **SciFact is not used in the reading** (different metric and relevance
  structure); its per-candidate numbers are in `observational.json`.
- **Latency was not measured.**

## 17. Recommended next experiment (run as Q3H, §18)

The cheapest way to resolve the residual is a **within-DocVQA split by relevance
margin**, using the existing store, CPU only, minutes:
- run frozen C and A on DocVQA restricted to queries with margin above and below
  InfoVQA's median (as run, the threshold is DocVQA's own median, so InfoVQA does
  not place the split; `SPLIT.md`);
- if DocVQA's high-margin half certifies like InfoVQA, the residual is margin
  structure;
- if not, it points to corpus homogeneity or labels.

That is still the same two corpora. A decisive test of the corpus-homogeneity and
label explanations needs a third document-VQA corpus with complete labels, which
is a new-dataset decision (Q8/Q9 territory) and needs your approval.

**Practical implication (INFERENCE, not a recommendation to change the library).**
The number of calibration queries a corpus needs depends strongly on its
relevance-margin distribution, which is measurable *before* compression from the
float index alone. That could predict how many queries a user must label.

## 18. Within-DocVQA margin split (Q3H, MEASURED; selection SIMULATED)

Pre-stated in `SPLIT.md` (`5734903`) before any computation. There is one split per
model, at the median of that model's DocVQA relevance margins:
- **High half:** 225 queries.
- **Low half:** 226 queries.

Same store, candidates, scoring and frozen E1 selection code (`run_cell`, 2,000
resamples per n). InfoVQA and full-DocVQA selection rows are E1's, unchanged.
Script: `scripts/research/q3_split.py`. Output: `split/<model>_<half>.json` and
`split/summary.json`.

**The threshold is about zero on all three models** (−0.0007, +0.0097 and +0.0128 per
token for ColPali, ColQwen2 and ColQwen2.5). So the split is, almost exactly,
"labelled page ranked first by a positive margin" against "not":
- the high half has top-1 accuracy of 0.99, 1.00 and 1.00;
- the low half has top-1 accuracy of 0.00, 0.05 and 0.07.

This was not anticipated in `SPLIT.md`, and it shapes everything below.

### 18.1 What the halves look like (MEASURED)

| model | set | n | mean nDCG@5 | top-1 | outside top 5 | median margin | in lowest pooled quintile |
|---|---|---|---|---|---|---|---|
| ColPali | DocVQA high | 225 | 0.996 | 0.99 | 0.00 | 0.099 | 0% |
| | DocVQA (all) | 451 | 0.584 | 0.49 | 0.34 | −0.001 | 23% |
| | InfoVQA | 494 | 0.846 | 0.78 | 0.10 | 0.085 | 18% |
| ColQwen2 | DocVQA high | 225 | 1.000 | 1.00 | 0.00 | 0.170 | 0% |
| | DocVQA (all) | 451 | 0.606 | 0.52 | 0.32 | 0.010 | 30% |
| | InfoVQA | 494 | 0.920 | 0.89 | 0.05 | 0.148 | 11% |
| ColQwen2.5 | DocVQA high | 225 | 1.000 | 1.00 | 0.00 | 0.134 | 0% |
| | DocVQA (all) | 451 | 0.618 | 0.53 | 0.31 | 0.013 | 29% |
| | InfoVQA | 494 | 0.917 | 0.87 | 0.05 | 0.134 | 11% |

- The high half's median margin is close to InfoVQA's.
- The high half has none of InfoVQA's low-margin tail, and its baseline is at the
  nDCG ceiling. So it is **not** an InfoVQA-like sample: it is a cleaner one.
- Because every high-half query already scores b ≈ 1, it cannot gain under
  compression. Its measured gain rate is 0 for every representative candidate.

### 18.2 Compression sensitivity per query (MEASURED)

Share of queries that lose (rank shift + drop-out), shown as high half / DocVQA
all / InfoVQA:

| model | int8 (1) | Ward 1/3 + int8 (16) | binary (27) | Ward 1/4 + binary (30) |
|---|---|---|---|---|
| ColPali | 0.4 / 0.7 / 0.0% | 5.3 / 6.2 / 3.4% | 7.6 / 10.4 / 7.7% | 10.2 / 10.6 / 9.1% |
| ColQwen2 | 0.0 / 0.4 / 0.0% | 2.2 / 7.3 / 3.4% | 4.9 / 9.8 / 3.4% | 6.2 / 11.8 / 6.5% |
| ColQwen2.5 | 0.0 / 0.4 / 0.0% | 2.2 / 7.5 / 2.8% | 3.6 / 8.4 / 5.1% | 4.0 / 9.8 / 6.3% |

- On ColQwen2 and ColQwen2.5, high-half loss rates are mostly at or below InfoVQA's.
- On ColPali they are at or slightly above InfoVQA's. This repeats the ColPali
  residual seen in §12.
- The low half loses more often than DocVQA overall (1.1–1.7x for candidates 16–30)
  and gains about twice as often
  (gains come only from imperfect queries; `split/*_low.json`).

### 18.3 Certification and retention (MEASURED)

| model | median log SE ratio, Doc / Info | high half / Info | share of gap closed (primary) | numerator sd, high / Info (median) | candidates with R_high < R_Info beyond 95% CI |
|---|---|---|---|---|---|
| ColPali | 0.73 | 0.12 | **84%** | 1.33 | 28 of 40 |
| ColQwen2 | 0.74 | −0.15 | **120%** | 0.93 | 1 of 40 |
| ColQwen2.5 | 0.99 | 0.14 | **86%** | 1.25 | 10 of 40 |

- The SE gap closes mainly through the denominator: mean b is 1.00 in the high half,
  against 0.85–0.92 on InfoVQA.
- On ColPali and ColQwen2.5 the per-query spread stays 1.25–1.33x InfoVQA's.
- The low half has a median log SE ratio of 2.0–2.2 against InfoVQA. Essentially all
  of DocVQA's certification difficulty sits in the half with non-positive margin.
- Pool retention in the high half is **lower** than InfoVQA's for many candidates
  on ColPali (28 of 40) and ColQwen2.5 (10 of 40). That is partly built into the
  split: high-half retention is pure loss, because no query can gain from b = 1.
  InfoVQA's retention, by contrast, nets losses against gains from imperfect
  queries. So this is not a like-for-like retention comparison.

### 18.4 Selection at n = 225, T = 0.95 (SIMULATED resampling, frozen E1 code)

Median selected compression. Deployment was ≥ 98.6% wherever the median is above 1x.

| model | method | InfoVQA | DocVQA (all) | DocVQA high | DocVQA low |
|---|---|---|---|---|---|
| ColPali | A | 72.7x | 7.8x | 23.2x | 3.9x |
| | C@0.05 | 72.7x | 3.9x | 15.4x | 1.0x (deploys 0.8%) |
| ColQwen2 | A | 63.0x | 11.6x | 47.6x | 3.9x |
| | C@0.05 | 63.0x | 7.8x | 29.7x | 1.0x (deploys 0.1%) |
| ColQwen2.5 | A | 76.0x | 10.6x | 47.6x | 3.9x |
| | C@0.05 | 76.0x | 7.8x | 47.6x | 1.0x (deploys 5.6%) |

- Gap to InfoVQA:
  - A: from 5.4–9.3x for all of DocVQA down to 1.3–3.1x for the high half.
  - C: from 8.1–18.6x down to 1.6–4.7x.
- Method C's miss rate was at most 0.05% in every half.
- A's miss rate in the ColPali high half was 1.85%. That fits E1: A does not control
  its miss rate.

### 18.5 Pre-stated reading

| model | gap closed | A within 2x of InfoVQA | C within 2x of InfoVQA | reading |
|---|---|---|---|---|
| ColPali | 84% | no (3.1x) | no (4.7x) | C (ambiguous) |
| ColQwen2 | 120% | yes (1.3x) | no (2.1x, just outside) | C (ambiguous) |
| ColQwen2.5 | 86% | yes (1.6x) | yes (1.6x) | A |

**Overall: C, ambiguous**, by the rule fixed in `SPLIT.md`: one model reads A, none
reads B.

### 18.6 Interpretation (INFERENCE)

- **What the split shows.** Within DocVQA, almost all of the certification
  difficulty, and most of the selection gap, sits in the queries whose labelled page
  is not ranked first (non-positive margin).
  - Restricted to the other half, DocVQA's certification SE matches or beats
    InfoVQA's (84–120% of the log gap closed).
  - The selected compression comes within 1.3–4.7x of InfoVQA's, from 5.4–18.6x.
  - The within-DocVQA analysis gives additional evidence that relevance-margin
    structure accounts for most of the certification gap (84–120% in the high
    half) and most, but not all, of the selection gap.
- **Why it is not read as A.**
  - Splitting at DocVQA's median makes the high half a ceiling subset (b ≈ 1, no
    gains). It is easier than InfoVQA in its denominator and different from it in
    composition, so part of "approaches InfoVQA" comes from how the split was made.
  - On ColPali a clear gap remains: 3.1–4.7x in selection and 28 of 40 candidates
    with lower retention. The ceiling effect explains part of that retention gap,
    but it cannot be separated here.
- **Residual model-dependence is not established.**
  - §12 placed the largest residual on ColQwen2.5 (the least gap closed by
    matching).
  - The split places it on ColPali.
  - The two analyses condition on different things, so neither ordering is
    established. Read the statement in §14 that the residual is largest for
    ColQwen2.5 as unconfirmed.
- **Effect on the existing conclusions.**
  - §14 stands: certification difficulty is the main driver, margin composition
    explains much of it, and H3a is not the main explanation.
  - The split strengthens the margin part. It does not settle the remainder, which
    shows mainly on ColPali.
- **Not causal.** The halves are two parts of one query pool, not independent
  datasets, and conditioning on margin also conditions on everything correlated
  with it.

## 19. Q3 status

**Q3 closed: conclusion supported with residual uncertainty.**

- Supported: DocVQA's lower selected compression comes mainly from certification
  difficulty, not retention.
  - Relevance-margin composition accounts for most of it: 61–81% of the SE gap by
    cross-dataset matching (§12) and 84–120% by the within-DocVQA split (§18).
- Residual uncertainty, which further experiments on the same two corpora cannot
  resolve:
  - the remaining selection gap, mainly on ColPali;
  - whether the residual is model-dependent (§18.6);
  - corpus homogeneity and label noise as contributors.
- Separating these needs a third document-VQA corpus with complete labels, which
  is a new-dataset decision.
