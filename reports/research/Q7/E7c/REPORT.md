# E7c report: aligning OptiVision's evaluation with the official ViDoRe evaluation

| Item | Location |
|---|---|
| Frozen plan | `PLAN.md` (`308490e`) |
| Script | `scripts/research/e7c_alignment.py` (`fcf8403`, committed before running) |
| Tables | `scripts/research/e7c_tables.py`, output in `tables.md` |
| Data | `<store>.json` |

- CPU only, existing stored vectors.
- No library change, no re-encoding, no SAP reproduction, nothing tuned.
- Research-only addition: `pytrec-eval-terrier 0.5.10` in the research virtual
  environment.
- Results are **MEASURED** unless labelled EXTERNAL (SAP's published values,
  transcribed from the arXiv HTML) or INFERENCE.

**Checks (all passed, four datasets).**
- Each query `qj` is the j-th unique question text, labelled at its first row (the
  identity mapping between OptiVision and the raw dataset).
- The rebuilt uncompressed and E7a-configured indexes (random pruning and K-means with
  plain means, seeds 0, 1, 2, all three ratios) give per-query nDCG@5 under the current
  evaluation that equals E7a's saved arrays **exactly**.

## 1. Frozen alignment plan (summary)

**Six evaluation variants** are applied to the same score matrices:

| id | relevance labels | metric engine |
|---|---|---|
| E0 | first occurrence (current) | OptiVision `per_query_metrics` |
| E0p | first occurrence | pytrec_eval `ndcg_cut_5` |
| E1 | last occurrence | OptiVision |
| E2 | all occurrences | OptiVision |
| **OL** | last occurrence | pytrec_eval (legacy `vidore-benchmark` semantics) |
| **OM** | all occurrences | pytrec_eval (MTEB semantics, the **primary** aligned evaluation) |

- **Datasets:** ColPali v1.3 and ColQwen2 v1.0 × DocVQA and InfoVQA.
- **Configurations:** uncompressed; random pruning and K-means/plain at 1/5, 1/10 and
  1/20, with E7a's seeds and vector-count convention.
- **No query or page exclusions.**
- The decision rule is in `PLAN.md` §5.

## 2. Official toolkits and revisions

- **Raw datasets:** `vidore/docvqa_test_subsampled` @ `49bf8f13` and
  `vidore/infovqa_test_subsampled` @ `f793e830` (500 rows each).
- **Legacy official:** `vidore-benchmark` `ViDoReEvaluatorQA` (main branch; the
  repository marks it deprecated and kept for reproducibility). Its metric runs through
  MTEB's retrieval evaluator (pytrec_eval).
- **Current official:** MTEB tasks with data from `mteb/VidoreDocVQARetrieval` @
  `7237bd4b` and `mteb/VidoreInfoVQARetrieval` @ `f7393ade`, identical in content to
  `vidore/*_test_subsampled_beir`. Metric: pytrec_eval `ndcg_cut_5`, computed with
  `pytrec-eval-terrier 0.5.10`.

## 3. Differences between the pipelines (established before the run)

| component | OptiVision (current) | legacy official | MTEB official |
|---|---|---|---|
| query set | 451 / 494 unique texts | same (deduplicated by text) | same, same order |
| corpus | 500 pages, page i = row i | 500, keyed by filename (all unique) | 500, `corpus-test-i` = row i |
| relevance for a question repeated on several pages | **first** page | **last** page | **all** pages |
| scoring | exact MaxSim, float32 | MaxSim via the model processor | model-defined |
| metric | OptiVision nDCG@5, ties by page index | pytrec_eval | pytrec_eval |
| exclusions | none | none | none |

- **Affected queries:** DocVQA has 22 repeated question texts (49 extra rows; one query
  has 12 relevant pages under MTEB). InfoVQA has 3.
- **Not alignable here:** encoding and preprocessing (resolution, visual tokens per
  page, processor version, dtype). This would need GPU re-encoding and is untested.

## 4. Datasets and exact evaluation population

- DocVQA: 451 queries; InfoVQA: 494 queries; 500 pages each, in every variant.
- Relevant pairs: first and last rules 451 / 494; MTEB rule 500 / 500.
- The last-occurrence rule changes the labelled page for exactly the 22 (DocVQA) and 3
  (InfoVQA) repeated texts.

## Results tables (MEASURED; SAP columns EXTERNAL)

### Baselines (nDCG@5, uncompressed)

| dataset | queries changed by label rule | E0 | E0p | E1 | E2 | OL | OM | SAP |
|---|---|---|---|---|---|---|---|---|
| ColPali DocVQA | 22 (of 451) | 0.5841 | 0.5841 | 0.5790 | 0.5825 | 0.5790 | 0.5825 | 0.59 |
| ColPali InfoVQA | 3 (of 494) | 0.8458 | 0.8458 | 0.8460 | 0.8461 | 0.8460 | 0.8461 | 0.85 |
| ColQwen2 DocVQA | 22 (of 451) | 0.6065 | 0.6065 | 0.6052 | 0.6068 | 0.6052 | 0.6068 | 0.59 |
| ColQwen2 InfoVQA | 3 (of 494) | 0.9197 | 0.9197 | 0.9198 | 0.9199 | 0.9198 | 0.9199 | 0.91 |

### Random pruning (primary control): retention % by evaluation variant (mean of seeds 0, 1, 2)

| dataset | ratio | E0 | E0p | E1 | E2 | OL | OM | OM 95% interval |
|---|---|---|---|---|---|---|---|---|
| ColPali DocVQA | 1/5 | 92.24 | 92.24 | 92.55 | 92.39 | 92.55 | 92.39 | 89.9–94.9 |
| ColPali DocVQA | 1/10 | 83.89 | 83.89 | 84.33 | 84.10 | 84.33 | 84.10 | 81.0–87.3 |
| ColPali DocVQA | 1/20 | 74.63 | 74.63 | 75.18 | 74.86 | 75.18 | 74.86 | 70.7–78.8 |
| ColPali InfoVQA | 1/5 | 97.17 | 97.17 | 97.17 | 97.19 | 97.17 | 97.19 | 95.8–98.4 |
| ColPali InfoVQA | 1/10 | 94.41 | 94.41 | 94.40 | 94.41 | 94.40 | 94.41 | 92.5–96.1 |
| ColPali InfoVQA | 1/20 | 91.04 | 91.04 | 91.02 | 91.03 | 91.02 | 91.03 | 88.8–93.1 |
| ColQwen2 DocVQA | 1/5 | 93.11 | 93.11 | 93.19 | 93.39 | 93.19 | 93.39 | 90.8–95.9 |
| ColQwen2 DocVQA | 1/10 | 85.30 | 85.30 | 85.57 | 85.76 | 85.57 | 85.76 | 82.5–89.0 |
| ColQwen2 DocVQA | 1/20 | 76.52 | 76.52 | 76.72 | 77.01 | 76.72 | 77.01 | 73.3–80.6 |
| ColQwen2 InfoVQA | 1/5 | 96.32 | 96.32 | 96.33 | 96.33 | 96.33 | 96.33 | 95.1–97.5 |
| ColQwen2 InfoVQA | 1/10 | 92.69 | 92.69 | 92.72 | 92.70 | 92.72 | 92.70 | 91.0–94.3 |
| ColQwen2 InfoVQA | 1/20 | 88.09 | 88.09 | 88.17 | 88.13 | 88.17 | 88.13 | 86.1–90.2 |

### K-means, plain mean: retention % by evaluation variant (mean of seeds 0, 1, 2)

| dataset | ratio | E0 | E0p | E1 | E2 | OL | OM | OM 95% interval |
|---|---|---|---|---|---|---|---|---|
| ColPali DocVQA | 1/5 | 94.76 | 94.76 | 95.29 | 95.12 | 95.29 | 95.12 | 92.6–97.8 |
| ColPali DocVQA | 1/10 | 90.51 | 90.51 | 90.92 | 90.79 | 90.92 | 90.79 | 87.5–94.1 |
| ColPali DocVQA | 1/20 | 81.44 | 81.44 | 81.98 | 81.72 | 81.98 | 81.72 | 77.9–85.9 |
| ColPali InfoVQA | 1/5 | 98.93 | 98.93 | 98.98 | 98.95 | 98.98 | 98.95 | 98.0–99.9 |
| ColPali InfoVQA | 1/10 | 96.98 | 96.98 | 96.91 | 96.94 | 96.91 | 96.94 | 95.4–98.3 |
| ColPali InfoVQA | 1/20 | 94.97 | 94.97 | 94.88 | 94.93 | 94.88 | 94.93 | 93.0–96.7 |
| ColQwen2 DocVQA | 1/5 | 97.00 | 97.00 | 97.42 | 97.59 | 97.42 | 97.59 | 95.3–100.0 |
| ColQwen2 DocVQA | 1/10 | 93.65 | 93.65 | 94.02 | 94.30 | 94.02 | 94.30 | 91.4–97.0 |
| ColQwen2 DocVQA | 1/20 | 87.73 | 87.73 | 87.85 | 88.26 | 87.85 | 88.26 | 85.1–91.4 |
| ColQwen2 InfoVQA | 1/5 | 97.38 | 97.38 | 97.48 | 97.45 | 97.48 | 97.45 | 96.2–98.6 |
| ColQwen2 InfoVQA | 1/10 | 95.07 | 95.07 | 95.27 | 95.19 | 95.27 | 95.19 | 93.6–96.6 |
| ColQwen2 InfoVQA | 1/20 | 91.83 | 91.83 | 92.04 | 91.95 | 92.04 | 91.95 | 89.8–93.8 |

### A (current) / B (aligned) / C (SAP published)

| dataset | ratio | baseline A / B (OM) / C | random A / B (OM) / C | B − A | K-means A / B (OM) / SAP Cluster | B − A |
|---|---|---|---|---|---|---|
| ColPali DocVQA | 1/5 | 0.5841 / 0.5825 / 0.59 | 92.24 / 92.39 / 91.65 | +0.16 | 94.76 / 95.12 / 93.44 | +0.36 |
| ColPali DocVQA | 1/10 | 0.5841 / 0.5825 / 0.59 | 83.89 / 84.10 / 85.03 | +0.20 | 90.51 / 90.79 / 86.09 | +0.28 |
| ColPali DocVQA | 1/20 | 0.5841 / 0.5825 / 0.59 | 74.63 / 74.86 / 73.58 | +0.24 | 81.44 / 81.72 / 76.05 | +0.28 |
| ColPali InfoVQA | 1/5 | 0.8458 / 0.8461 / 0.85 | 97.17 / 97.19 / 96.37 | +0.02 | 98.93 / 98.95 / 98.04 | +0.02 |
| ColPali InfoVQA | 1/10 | 0.8458 / 0.8461 / 0.85 | 94.41 / 94.41 / 93.16 | +0.00 | 96.98 / 96.94 / 94.64 | -0.04 |
| ColPali InfoVQA | 1/20 | 0.8458 / 0.8461 / 0.85 | 91.04 / 91.03 / 89.42 | -0.00 | 94.97 / 94.93 / 87.10 | -0.04 |
| ColQwen2 DocVQA | 1/5 | 0.6065 / 0.6068 / 0.59 | 93.11 / 93.39 / 89.04 | +0.28 | 97.00 / 97.59 / 94.74 | +0.59 |
| ColQwen2 DocVQA | 1/10 | 0.6065 / 0.6068 / 0.59 | 85.30 / 85.76 / 77.75 | +0.47 | 93.65 / 94.30 / 87.41 | +0.65 |
| ColQwen2 DocVQA | 1/20 | 0.6065 / 0.6068 / 0.59 | 76.52 / 77.01 / 62.83 | +0.49 | 87.73 / 88.26 / 75.37 | +0.53 |
| ColQwen2 InfoVQA | 1/5 | 0.9197 / 0.9199 / 0.91 | 96.32 / 96.33 / 93.53 | +0.00 | 97.38 / 97.45 / 95.89 | +0.06 |
| ColQwen2 InfoVQA | 1/10 | 0.9197 / 0.9199 / 0.91 | 92.69 / 92.70 / 89.14 | +0.02 | 95.07 / 95.19 / 89.70 | +0.12 |
| ColQwen2 InfoVQA | 1/20 | 0.9197 / 0.9199 / 0.91 | 88.09 / 88.13 / 82.52 | +0.04 | 91.83 / 91.95 / 79.25 | +0.12 |

### Random-control gap to SAP Random, |D − SAP| in points (decision-rule quantity)

| model | E0 (current) median / max | OL (legacy official) median / max | OM (MTEB official) median / max | change in median, OM against E0 |
|---|---|---|---|---|
| ColQwen2 | 4.82 / 13.69 | 4.90 / 13.89 | 4.98 / 14.18 | +3.4% |
| ColPali | 1.09 / 1.62 | 1.07 / 1.60 | 1.09 / 1.61 | +0.1% |

## 5. Baseline comparison

- **Metric engine: no effect.** pytrec_eval reproduces OptiVision's nDCG@5 exactly
  under the same labels (E0p = E0 on every dataset and configuration).
- **Label rules: small baseline changes.** The largest is on ColPali DocVQA:
  - last-occurrence rule: −0.51 points;
  - all-occurrence rule (MTEB): −0.16 points.
  - On every other dataset the change is ≤ 0.13 points.
- **No dataset changes by more than 1 point,** so decision rule 4 (stop at the
  baseline) is not triggered. The decomposition accounts for every change: E1 = OL and
  E2 = OM to the fourth decimal.
- **Relative to SAP's baselines** (EXTERNAL; 0.59, 0.85, 0.59, 0.91), the aligned
  baselines are:
  - ColPali: −0.8 and −0.4 points;
  - ColQwen2: **+1.7 and +1.0 points.**

  Alignment does not change these offsets.

## 6. Random-pruning control (primary diagnostic)

- **Alignment changes random-pruning retention by at most +0.49 points** (ColQwen2
  DocVQA 1/20). On InfoVQA the change is ≤ 0.04 points.
- **The gap to SAP's Random row (decision-rule quantity):**
  - **ColQwen2:** median 4.82 → **4.98** points under MTEB alignment, maximum 13.69 →
    **14.18**. The median moves by +3.4%: no reduction.
  - **ColPali:** median 1.09 points and maximum 1.6 under every variant.

## 7. K-means comparison

- **Alignment moves K-means retention by −0.04 to +0.65 points,** in the same direction
  as random pruning.
- **The gaps to SAP's Cluster row stay as in E7a:**
  - ColPali: +0.9 to +7.8 points;
  - ColQwen2: +1.6 to +12.9 points.

## 8. Current / aligned / SAP

See the "A (current) / B (aligned) / C (SAP published)" table in §4. The three
categories are kept separate, and B does not replace A.

## 9. What alignment explains (INFERENCE)

- **Very little.** Two effects can be attributed to specific components:
  1. **The label rule for repeated questions** shifts baselines by up to 0.5 points
     (ColPali DocVQA) and retention by up to about 0.6.
  2. **The metric engine and tie handling** have no measurable effect.
- **Neither moves the SAP comparison materially.**

## 10. What remains unexplained

**ColQwen2: unresolved, and not caused by the evaluation pipeline.**
- The random-control discrepancy is unchanged: up to 14.2 points against SAP's Random
  row.
- So are the K-means gaps and the 1.0–1.7-point baseline offset.
- **What E7c rules out as the main cause:** the evaluation pipeline steps tested here
  (query set, page identity, relevance labels, nDCG implementation, ties).
- **What remains, untested** (INFERENCE):
  - encoding and preprocessing differences, e.g. ColQwen2's dynamic image resolution,
    which changes how many visual tokens a page has (OptiVision's stored ColQwen2
    vectors average about 748 per page);
  - SAP's exact query set or evaluation code, which its paper does not specify;
  - errors in the HTML transcription of SAP's tables.

**ColPali: largely consistent with SAP.** The evaluation agrees on the random control.
The remaining K-means gap is specific to the mechanism or its implementation (E7a §9),
and E7c does not change that.

## 11. Limitations

- **A misstatement in the frozen plan.** `PLAN.md` §5 quoted E7a's ColQwen2 median
  random-control gap as 5.6 points. The value computed from E7a's data is 4.82, the E0
  column above. The decision rule uses the relative change under alignment, so the
  outcome is unaffected.
- **The encoding side could not be aligned** without GPU re-encoding.
- **SAP's evaluation details are not published,** so this is alignment with the
  *official ViDoRe* evaluation, not with SAP's.
- **SAP's numbers are HTML transcriptions.**
- **Coverage:** four datasets and two ViDoRe V1 subsets.
- **pytrec-eval-terrier was used as MTEB's metric engine,** not the full MTEB or
  `vidore-benchmark` packages. Their data loading and qrels logic was reproduced from
  the official sources and datasets cited in §2–3, and the identity mapping was
  verified.

## 12. Decision on E7b

- **Pre-stated rule:** case 2, the random-control discrepancy stays **substantially
  intact** (the ColQwen2 median rose by 3.4%, against the ≥ 25% reduction needed for a
  partial result). It is documented as unresolved.
- **E7b is not justified by this evidence.** The rule allows E7b "only if the remaining
  question concerns SAP's pruning implementation", but the remaining ColQwen2 question
  shows up in **random pruning**, which involves no SAP method at all.
  - Running SAP's pruning (E7b) could not explain a disagreement that exists before any
    pruning method is applied.
  - The only experiment that could address it is **re-encoding** the ColQwen2 pages
    with documented processor settings, compared against SAP's (unpublished) settings.
    That would be a new GPU experiment, and it would need SAP's settings to be
    meaningful.
- **Recommendation: stop Q7 and move to Q8,** with the record as it stands:
  - OptiVision's evaluation agrees with the official ViDoRe evaluation to within 0.5
    points;
  - E7a's mechanism attribution stands;
  - ColPali is consistent with SAP on the random control;
  - the ColQwen2 discrepancy against SAP is documented as unresolved and upstream of
    the evaluation pipeline.
- **If the ColQwen2 question is ever pursued,** the first step is asking SAP's authors
  for their ColQwen2 image settings and evaluation code, not E7b.

E7b and Q8 have not been started.
