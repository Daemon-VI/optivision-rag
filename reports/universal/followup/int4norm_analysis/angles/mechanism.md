All numbers below were re-run independently (mechanism_verify/v.py and w.py) on the 46 units that Q1 to Q9 read: 24 text, 14 page and 8 SciDocs units from 19 models, taken from f9b..f13b. The 2 f13b scale units are excluded. Retention is ndcg@5 over float32. This is exploratory, post hoc analysis. It makes no novelty or generality claim, and no fitted threshold below is a validated rule.

**nsr heuristic (label-free, built after seeing the unit table, about 18 statistics tried):** nsr = (peak_to_rms^2/588)/(1-doc_cos_to_mean^2). It is a ranking heuristic and saturates above 1 for the nemotron units.
- Regime split at 0.5: 36 units with nsr <= 0.5 and 10 units above it (4 nemotron, 3 LateOn, 3 mLateOn).
  - Low group: lowest int4norm is 0.944 (mxbai-edge-17m NFCorpus, the only one of 36 below 0.95). Lowest int4norm(mean) is 0.959.
  - High group: 9 of 10 are below 0.95 with int4norm. The exception is mLateOn SciFact at 0.9675. All 10 have int4norm(mean) at 0.988 or higher.
  - 11 of the 36 low-nsr units have plain int4 below 0.95. All except mxbai-17m NFCorpus reach 0.971 or higher once the norm is stored.
  - The cut at 0.5 is the natural gap between the groups (low max 0.17, high min 0.55). It is not the in-sample optimum, which is nsr 0.164 for 0.978 accuracy.
- Spearman, n=46 units (19 models), against int4norm, int4 and gain (int4norm minus int4); AUC is for "int4norm < 0.95", in-sample, 10 positives:

| statistic | vs int4norm | vs int4 | vs gain | AUC |
|---|---|---|---|---|
| nsr | -0.82 | -0.87 | +0.81 | 0.997 |
| `norms.int4.ratio_sd` | -0.76 | -0.75 | +0.69 | 0.967 |
| top1 second-moment share | -0.86 | -0.90 | +0.83 | 0.969 |
| top4 second-moment share | -0.87 | -0.91 | +0.85 | 0.983 |
| doc cos to mean | -0.79 | -0.86 | +0.84 | 0.933 |
| peak_to_rms | -0.18 | -0.07 | -0.03 | 0.892 |
| int4 score Pearson (geometry) | +0.81 | +0.83 | -0.76 | 0.003 |

- Gain of int4norm(mean) over int4(mean) is weakly related to geometry: top4 +0.25, nsr +0.12, ratio_sd +0.38.
- Model-level (n=19): top4 against int4norm rho -0.92. Doc cos to mean against gain rho +0.86. Units cluster by model, so the effective n is about 19.
- Leave-one-model-out (19 folds), accuracy / false alarms / misses for "int4norm < 0.95":

| statistic | accuracy | false alarms | misses |
|---|---|---|---|
| nsr | 0.957 | 1 | 1 |
| int4 score Pearson | 0.957 | 1 | 1 |
| `ratio_sd` | 0.935 | 0 | 3 |
| top1 | 0.935 | 0 | 3 |
| top4 | 0.848 | 5 | 2 |
| doc cos to mean | 0.783 | 6 | 4 |
| peak_to_rms | 0.826 | 2 | 6 |

  - In-sample best-threshold accuracy: nsr 0.978, ratio_sd 0.957 (threshold about 0.0142), Pearson 0.978 (threshold about 0.845).
- Fraction of the repair closed by the norm: (int4norm minus int4) divided by (int4norm(mean) minus int4), over the 21 units where plain int4 is under 0.95 and the repair exceeds 0.05. The values run from 0.74 to 0.99 for nsr up to 1.24, and rho(nsr, fraction closed) = -0.73.
- Norm scatter: `norms.int4.ratio_sd` has a median of 0.0101 over units. It is 0.018 to 0.022 for LateOn and nemotron, about 0.012 for mLateOn, and 0.004 (EVIE) to 0.007 (Vultron) for the other two families. After storing the norm it is at most 0.00028 on every unit.

**Nemotron (n=4, rank-only, suggestive):**

| unit | nsr | int4norm | share of repair closed |
|---|---|---|---|
| 4b TabFQuAD | 10.1 | 0.743 | 0.73 |
| 4b DocVQA | 13.3 | 0.409 | 0.40 |
| 8b TabFQuAD | 25.0 | 0.152 | 0.14 |
| 8b DocVQA | 32.3 | 0.082 | 0.07 |

- p2r is 27 to 28 for 4b and 40 to 41 for 8b. top4 is 0.61 to 0.63 for 4b and 0.76 to 0.78 for 8b. int4 score Pearson is 0.68 to 0.77 and does not explain the 4b versus 8b difference.
- 4b TabFQuAD (0.743) against 4b DocVQA (0.409):
  - Geometry differs only a little: doc cos to mean 0.935 vs 0.949, p2r 27.3 vs 27.9, top4 0.61 vs 0.63.
  - 280 pages and 280 queries for TabFQuAD, 500 pages and 451 queries for DocVQA. 20% of DocVQA queries are already at zero at float32.
  - Per-query ndcg@10 under int4norm, share unchanged from float32: 0.71 vs 0.41. Share that fell to zero while float32 > 0: 0.19 vs 0.43.
  - Geometry and dataset effects cannot be separated with 2 datasets per model.

**LateOn (per-query ndcg@10):**
- NFCorpus: int4 0.638, int4norm 0.9006. Share of queries scoring zero where float32 > 0 is 0.17 for int4. Mean ratio over those queries is 0.91 for int4norm. Norm closed 0.74 of the repair.
- SciFact: int4 0.282, int4norm 0.870. Zero-when-float32>0 share is 0.59 for int4 and 0.11 for int4norm. Norm closed 0.83.
- SciDocs: int4 0.134, norm closed 0.80.
- Interpretation language such as "the norm repairs scatter but not lost direction" is a hypothesis. No experiment here isolates the two effects.

**Other:** colpali-v1.3 DocVQA int4norm 0.991 against int4 1.003 is a 0.012 loss on a low-nsr unit. Whether that is noise was not tested.

Treat nsr as a post hoc screen, not a rule. There are 10 positive units, 9 of them in 3 model families, and the class boundary rests on one unit (mLateOn SciFact 0.968).