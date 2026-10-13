# E7c plan: aligning OptiVision's evaluation with the official ViDoRe evaluation (frozen before any E7c result)

Follows E7a (`946c816`), whose pre-stated rule selected E7c.
- CPU only, existing stored vectors.
- No library change, no re-encoding, no GPU, no SAP reproduction, no tuning.
- Research-only addition: `pytrec-eval-terrier==0.5.10`, the metric engine MTEB uses,
  installed into the research virtual environment. It is not a library dependency.

## 1. Inspected pipelines (facts established before freezing)

**Raw datasets** (Hugging Face):
- `vidore/docvqa_test_subsampled` @ `49bf8f13`: 500 rows.
- `vidore/infovqa_test_subsampled` @ `f793e830`: 500 rows.
- Every `image_filename` is unique, and there are no null queries.
- **Repeated question texts:** in DocVQA, 451 unique texts, of which 22 occur on 2–12
  different pages (49 extra rows). In InfoVQA, 494 unique texts, of which 3 repeat.

**OptiVision** (current, as in Q3, Q4 and E7a; `scripts/encode_vectors.py` and
`data/*/queries.json`):
- One query per unique text, in first-occurrence order (`q0000` …).
- Relevance: the page of the **first** row carrying that text, only.
- Pages: all 500 rows, with page i = row i.
- Metric: `optivision.evaluation.per_query_metrics` nDCG@5 (binary gain, discount
  1/log2(rank+1), ideal DCG over min(#relevant, 5)), with the ranking from `rank()`
  (stable descending sort; ties broken by page index).

**Official, legacy `vidore-benchmark` `ViDoReEvaluatorQA`** (main branch; the
repository now marks it deprecated and kept for reproducibility):
- Queries are deduplicated by text.
- `queries2filename` is a dict comprehension over the **full** dataset, so a repeated
  text maps to the **last** row's page.
- Relevance: that single page.
- Passages are keyed by `image_filename`, with a max over rows sharing a filename
  (no effect here: filenames are unique).
- Metric: MTEB's retrieval evaluator, which uses pytrec_eval.

**Official, current route: MTEB** (the `vidore-benchmark` README: V1 evaluation
"moved to MTEB").
- Datasets: `mteb/VidoreDocVQARetrieval` @ `7237bd4b` and `mteb/VidoreInfoVQARetrieval`
  @ `f7393ade`, identical in content to `vidore/*_test_subsampled_beir`.
- Queries: the same 451 / 494 unique texts, in first-occurrence order.
- Corpus: `corpus-test-i` = row i = OptiVision page i (verified).
- Relevance (`qrels`): **every** row whose query equals the text is relevant, with
  score 1 (verified). DocVQA has 22 queries with 2–12 relevant pages; InfoVQA has 3.
- Metric: pytrec_eval `ndcg_cut_5`.

**Not alignable without re-encoding** (stop condition for this component only):
image preprocessing, the number of visual tokens and resolution (ColQwen2 dynamic
resolution), special-token handling inside the encoder, and model dtype or processor
version. E7c reuses the stored vectors and therefore **cannot test the encoding side**.
Any difference there stays unresolved.

## 2. What is evaluated

**Datasets:** ColPali v1.3 and ColQwen2 v1.0 × DocVQA and InfoVQA (the four with SAP
values). ColQwen2.5 is out of scope.

**Score matrices.** Exact MaxSim, float32, all 451 / 494 queries × all 500 pages.
- **Uncompressed.**
- **D, random pruning:** seeds {0, 1, 2} at 1/5, 1/10 and 1/20, rebuilt with E7a's
  code and seeds.
- **A, K-means with plain means:** seeds {0, 1, 2}, same ratios, rebuilt with E7a's
  code.
- Vector-count convention as in E7a: k = ceil(r × n_free), protected tokens kept.
- Ward and rescaling are not rerun.

**Check.** Under the current evaluation (E0), the rebuilt A and D per-query nDCG@5
(seed-mean) must equal E7a's saved arrays exactly. Otherwise stop.

## 3. Evaluation variants (all applied to the same score matrices)

| id | relevance labels | metric engine | purpose |
|---|---|---|---|
| E0 | first occurrence (OptiVision) | OptiVision `per_query_metrics` | current evaluation (= Q4 / E7a) |
| E0p | first occurrence | pytrec_eval `ndcg_cut_5` | isolates the metric engine and tie handling |
| E1 | last occurrence (legacy official) | OptiVision | isolates the legacy label rule |
| E2 | all occurrences (MTEB official) | OptiVision | isolates the MTEB label rule |
| **OL** | last occurrence | pytrec_eval | **official legacy `vidore-benchmark` semantics** |
| **OM** | all occurrences | pytrec_eval | **official current MTEB semantics: the primary aligned evaluation** |

**pytrec_eval input.** `{query_id: {page_id: float(score)}}` over all 500 pages, as
MTEB passes it, with qrels `{query_id: {page_id: 1}}`. pytrec_eval's own ordering
applies (trec_eval convention for ties). nDCG@5 is the mean over queries.

**Retention** = mean(compressed) / mean(uncompressed) under the **same** variant.
Paired bootstrap 95% intervals (seed 0, 1,000 resamples) are given for OM retention.

**Exclusions:** none, for queries or pages. All 451 / 494 queries are evaluated in
every variant.

## 4. Comparison with SAP (only after §3 is complete)

Three separate categories:
- **A.** Current OptiVision evaluation (E0, = E7a).
- **B.** ViDoRe-aligned evaluation: OM primary, OL secondary.
- **C.** SAP published values (EXTERNAL; Tables 7 and 8 of arXiv 2601.20107v3, HTML
  transcription).

For each dataset and ratio, report:
- the baseline nDCG@5 under A and B, and SAP's;
- random-pruning retention under A and B, and SAP's Random;
- K-means retention under A and B, and SAP's Cluster;
- the change caused by alignment (B − A).

## 5. Decision rule (fixed now)

The random-control gap is g = |D retention − SAP Random| in points, over the 6 ColQwen2
cells (E7a: median 5.6, maximum 13.7) and the 6 ColPali cells (E7a: maximum 1.6).

1. **Alignment substantially closes it.** Under OM, the ColQwen2 median g falls by
   ≥ 50% **and** its maximum is ≤ 5 points.
   - An evaluation-pipeline mismatch is then an important source.
   - The decomposition (E0p, E1, E2) names the component.
   - The whole discrepancy is not called solved unless the K-means gap also closes.
2. **It stays substantially intact.** The ColQwen2 median g falls by < 25%.
   - The disagreement is then upstream of clustering but not explained by the
     evaluation differences tested here. It is documented as unresolved.
   - E7b becomes justifiable only if the remaining question concerns SAP's pruning
     implementation. The encoding side (§1) would still be untested.
3. **Partial** (between 1 and 2): reported as such, with the same E7b condition.
4. **Material baseline change.** If the uncompressed nDCG@5 under OM differs from E0
   by > 1 point on any dataset:
   - the decomposition must account for it, i.e. E2 must agree with OM within 0.5
     points, and the label rule must be the identified cause;
   - only then are compressed results interpreted.
   - If it is not accounted for, stop at the baseline.
5. **No undocumented assumptions.** If any official component cannot be reproduced
   without one, stop and document it. Nothing approximate is called "official".

## 6. Not changed

The compression algorithms, seeds, target vector counts, K-means settings and the
query vectors. Evaluation parameters are fixed above and not tuned after results.
