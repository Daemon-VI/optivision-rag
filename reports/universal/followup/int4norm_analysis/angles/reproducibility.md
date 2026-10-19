Exploratory, post hoc verification. Scripts: C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/reproducibility_verify/v1.py to v5.py.

## 1. Reproducibility of s17 vs earlier s13 (v1.py)
- s17: 48 units; 41 have an s13 twin on the same corpus; 7 have none (6 SciDocs units plus mxbai-colbert-large-v1 NFCorpus). That gives 68 same-corpus file comparisons.
- In those 68, baselines and every shared row value I compared (ndcg@5/10, bytes_per_doc, retention:ndcg@5 and the other numeric fields except timings) are identical to 0.0. I did not separately recompute the CI lists; they are likely identical but this is not independently confirmed. Per-query values cannot be compared, since s13 has none.
- The only differences are f13b vs f3_text_scale (2 pairs). f3 used FiQA distractors, f13/f13b use TREC-COVID, so the corpora differ (this is not a reproducibility failure). Max baseline difference I measured: 0.0325 (the original 0.0367 figure was not reproduced).
- Identical results are on the same hardware class (2x T4 per the analyst; I did not check the hardware). They do not show cross-hardware reproducibility, and I did not check whether vectors are bitwise identical.

## 2. SciDocs as a third text dataset (v2.py)
- Sizes: SciFact 5183 docs/300 queries (12 models), NFCorpus 3633/323 (12), SciDocs 25657/1000 (8 models).
- The int4 retention ndcg@5 table is reproduced exactly for the 8 shared models. Spearman for int4: 0.87 (n=12), 0.95 (n=8), 0.69 (n=8). int4norm: 0.71, 0.67, 0.79. int4(mean): 0.57, 0.55, 0.48. int4norm(mean): 0.28, 0.33, -0.29.
- With the post hoc threshold int4 retention < 0.90, the broken set (LateOn, answerai, mLateOn, mxbai-edge-17m, mxbai-edge-32m) is the same on all three datasets. This is threshold-sensitive (answerai SciFact 0.899, GTE SciDocs 0.904) and the 4 models absent from SciDocs are all near 1.0 elsewhere, so it should not be read as "the dataset never changes which models break". Severity varies a lot (answerai 0.455 on NFCorpus vs about 0.89 on the other two; LateOn 0.134 on SciDocs vs 0.638 on NFCorpus).
- int4norm(mean) retention lies between 0.984 and 1.017 on all 32 units; rank agreement within that band is mostly noise. On SciDocs only jina-colbert-v2 has an upper CI bound below 1.0 ([0.972, 0.996]). The same check on SciFact/NFCorpus was not recomputed.

## 3. f13b vs f9b (v3.py)
- Same 300 queries; corpus 5183 -> 15183 docs (+10k TREC-COVID). Baselines ndcg@5: colbertv2.0 0.623 -> 0.590; answerai 0.731 -> 0.698.
- Retention: colbertv2.0 int4 1.005/1.002, int4norm 1.004/0.999, int4norm(mean) 1.000/1.002; answerai int4 0.899/0.872, int4norm 0.980/0.987, int4norm(mean) 0.991/0.996, binary 0.931/0.937. All CIs overlap. CI overlap is a weak test, and answerai int4 moved 0.027 within wide CIs.
- Decoded-norm ratio_sd is essentially unchanged. Paired ndcg@10 deltas for answerai: int4 -0.077 vs -0.091 (se about 0.014), int4norm -0.014 vs -0.012, int4norm(mean) -0.007 vs -0.004.
- Two models, one distractor setup, 300 queries: weak evidence about scale.

## 4. Timings (v4.py, v5.py)
- Per-model log-log slope of s17 seconds per row vs n_docs: 1.14 (answerai), 1.20 (colbertv2.0), reproduced. This is confounded by query count (SciDocs 1000 vs about 300) and uncontrolled Kaggle load; with 4 points per model it should not be read as a clean scaling law in n_docs.
- s17 vs s13 over paired units: 41 units, s17 about 9,979 s vs s13 about 95,400 to 95,800 s, median ratio about 0.10 (the analyst's 39 units / 9,645 / 91,705 / 0.11 were not exactly reproduced). s17 has 21 rows per unit vs 49 to 74 in s13, so the speedup is mostly fewer rows scored.
- SciDocs: answerai 17,551 s (s13) vs 1,124 s (s17); LateOn 12,473 s vs about 901 s.
- The f3b s17 = 0 s anomaly and the full per-model s/row ranges were not verified.