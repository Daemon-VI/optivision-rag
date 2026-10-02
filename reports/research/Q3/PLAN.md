# Q3 plan and pre-registration: why DocVQA and InfoVQA behave differently

Written before any Q3 computation. Branch `research-investigation`, starting at
`b18ae19`. Labels: **MEASURED**, **SIMULATED**, **EXTERNAL**, **INFERENCE**.

## Question

`calibrate()` / `optimize()` choose much less compression on DocVQA than on
InfoVQA (R7b, R11, E1). For example, method C at T = 0.95, n = 225: ColPali 3.9x
against 72.7x. Why?

## Competing hypotheses (as given)

- **H3a.** DocVQA is intrinsically less compressible than InfoVQA.
- **H3b.** DocVQA is not necessarily less compressible, but its per-query quality
  is noisier or harder to certify, so the optimizer selects more conservatively.
- **H3c.** The difference is caused mainly by query/corpus characteristics: score
  margins, relevant-document structure, query difficulty, vector redundancy.
- **H3d.** Several factors contribute, and no single measured property explains
  the difference.

## Operational definitions (fixed now)

- **Compressibility of a configuration on a pool** (used for H3a): its pool
  retention R = Σc/Σb (finite query-pool truth, E1 definition), with a bootstrap
  95% interval over queries. "Less compressible" means lower R for the *same*
  pipeline, beyond the intervals.
- **Certification difficulty** (used for H3b): the standard error of the retention
  estimate from n = 225 calibration queries, by the delta method:
  SE = sd(c − R b) / (√n · mean(b)). It splits into a **numerator spread**
  sd(c − R b) and a **denominator** mean(b). Higher SE means wider bounds and more
  conservative selection, at equal true retention.
- **Per-query loss** d = b − c. Each query falls in one category (ViDoRe: one
  labelled page, nDCG@5):
  - *unchanged* (c = b);
  - *rank shift* (0 < c < b: the page moves down but stays in the top 5);
  - *drop-out* (b > 0, c = 0: the page leaves the top 5);
  - *gain* (c > b).
- **Failure-mode rule** (fixed now), for 3B, per dataset and candidate:
  - **"B: few catastrophic queries"** if drop-outs carry ≥ 70% of the total loss
    mass Σ max(d, 0) and the worst 10% of losing queries carry ≥ 50% of it;
  - **"A: widespread small"** if drop-outs carry < 30% of the loss mass;
  - **"C: mixture"** otherwise.
- **Score margins** use one definition across datasets and models. With s the
  MaxSim score and L the number of query token vectors:
  - relevance margin m = (s_rel − max s_non-relevant) / L (positive iff the
    labelled page ranks first);
  - top margin m12 = (s_1 − s_2) / L.

  Dividing by L puts margins in per-token cosine units. They are compared
  **between splits within one model**, never across models (each encoder has its
  own score scale).
- **Representative candidates**, fixed by position in the default space and
  identical pipelines on every dataset:
  - 1 `int8(per_vector)` (≈ 4x);
  - 16 `hierarchical_merge(0.33) > int4(mean)` (≈ 23x);
  - 27 `binary` (32x);
  - 30 `hierarchical_merge(0.25) > binary` (≈ 125x).

  All 40 candidates are summarised as well; nothing is chosen afterwards.

## Data (reused, not regenerated)

- **Per-query nDCG for all 40 candidates and the baseline:** the E1 store at
  `b18ae19` (`reports/research/E1/store/`; sha256 in `e1_0_*.json`): ColPali,
  ColQwen2, ColQwen2.5 × DocVQA, InfoVQA; SciFact separately (nDCG@10, several
  relevant documents).
- **Baseline score matrices and two representative compressed score matrices**
  (`binary`, `hierarchical_merge(0.25) > binary`): recomputed on CPU from the
  existing vectors with the library's `maxsim_matrix` / `score_candidate`, no
  re-encoding. **Consistency check:** the per-query nDCG from these recomputed
  scores must equal the E1 store exactly. Otherwise STOP.
- **Labels:** each distinct query has one labelled page. Pages carrying no label
  (DocVQA 49, InfoVQA 6 of 500) are rows whose question duplicates an earlier one.
  They may be relevant but unlabelled. They are counted, and so is how often the
  baseline ranks one first.
- **Existing geometry reports:** `reports/universal/geometry/` (nearest-neighbour
  cosine, corpus-mean norm), reused as they are.

## Analyses (each stated before running)

**3A. Baseline difficulty (descriptive).**
- Per query: b, hit@1, rank of the labelled page, s_1, m, m12, L, the near-tie
  count (non-relevant pages with (s − s_rel)/L > −0.01), and whether the top page
  is unlabelled.
- Per dataset: mean, sd, median and quantiles; DocVQA − InfoVQA differences with
  bootstrap 95% intervals (2,000 query resamples, seed 0).
- *Expected, if H3c (difficulty) matters:* DocVQA has lower b and smaller m. No
  causal reading.

**3B. Compression sensitivity.**
- Per dataset × candidate: pool retention with a 95% interval, the loss category
  shares, loss-mass concentration (share carried by the worst 10% of losing
  queries), the 5th-percentile loss, and the failure-mode label under the rule
  above.
- *Expected under H3a:* lower R on DocVQA for the same pipeline.
- *Expected under H3b:* similar R but larger SE.

**3C. Difficulty against sensitivity.**
- Per dataset, for each representative candidate: the loss rate P(d > 0) and the
  drop-out rate by baseline-rank stratum (1; 2–5; > 5 has no loss under nDCG@5),
  and by relevance-margin quintile (quintiles pooled over both splits of a model).
  Also Spearman ρ(m, d) among queries with b > 0, with a bootstrap 95% interval.
- That is 6 datasets × 4 candidates. It is descriptive: no p-values, effect sizes
  with intervals.
- *Expected under H3c:* small-margin queries lose more, in both splits.

**3D. Score margins before and after compression.**
- Distributions of m and m12 at baseline, and of m after the two representative
  compressed matrices. Also the change Δm and the share of queries whose m
  changes sign.
- Compared between splits within each model.

**3E. Relevance structure.**
- Labelled relevant pages per query (1 by construction for ViDoRe), unlabelled
  pages per corpus, and the rate at which the baseline's top page is an unlabelled
  page.
- SciFact separately (several relevant documents, nDCG@10).

**3F. Geometry.**
- Vectors per page (distribution), reuse of the existing nearest-neighbour cosine
  and corpus-mean norm, and page near-duplication: the cosine of each page's
  normalised mean vector to its most similar other page.
- Ward merging effect: pool retention of the Ward candidates, from the store.
- Vectors are available locally for all six VLM datasets. Nemotron vectors were
  not kept, so 2,560-d is excluded and marked unavailable.

**Certification decomposition (H3a against H3b).**
- For every candidate: R and SE at n = 225, split into numerator spread and
  denominator.
- The counterfactual SE if DocVQA had InfoVQA's mean(b): SE × mean(b_Doc) /
  mean(b_Info).
- This is arithmetic on measured quantities, labelled INFERENCE, not an
  experiment.

**3G. One controlled intervention.** Chosen only after 3A–3F, written into
`INTERVENTION.md` with its own hypothesis before it is run.

## Discipline

- No formal hypothesis tests are planned. If any is added, Holm correction across
  all of them.
- SciFact is reported separately and never pooled with the ViDoRe analysis
  (different metric and relevance structure).
- Storage, RAM and latency stay separate. This study is about quality retention
  and the selection it drives; byte compression per pipeline is identical across
  splits up to vectors per page, and that is reported.
- The E1 artifacts are read only.
