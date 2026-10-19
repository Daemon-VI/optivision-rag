# Harm and noise: does storing the norm cost retention? (verified, corrected)

**Status.** Exploratory, post hoc analysis. No novelty claim. All numbers below were recomputed by the verifier in `C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/harm_verify/` (`v1.py` to `v4.py`).

**Sample.** 48 units (46 prediction units plus 2 scale units). Query counts: SciFact 300, NFCorpus 323, SciDocs 1000, DocVQA 451, TabFQuAD 280. Units share queries across models, so they are not independent.

**Limitation.** Only `per_query_ndcg@10` is stored, so every paired test and flip simulation uses @10 as a proxy for the @5 headline. For colpali DocVQA the @5 gap is -0.0116 and the @10 gap is -0.003, so the proxy may understate it.

## 1. Differences (int4norm minus int4), confirmed
- int4norm - int4: 10 of 48 units negative at @5, 8 at @10. Median +0.0194 / +0.0191, mean +0.159 / +0.159, min -0.0116 (colpali DocVQA, @5).
- int4norm(mean) - int4(mean): 11 negative at @5, 13 at @10. Median +0.0039 / +0.0028, mean +0.0066 / +0.0069, min -0.0109 (EVIE TabFQuAD, @10).
- The mean is dominated by units where plain int4 collapses.
- Where the comparison codec keeps at least 0.95 retention, the mean gain is small: +0.0037 (26 units, 10 negative) and +0.0047 (46 units, 11 negative).
- 8 of 192 cells are at -0.005 or lower.

## 2. Paired per-query tests at @10, confirmed
- colpali DocVQA: mean -0.0018, interval [-0.0093, +0.0056], 18 up and 17 down of 451, sign p 1.0. The statement that the @5 gap is "indistinguishable from noise" is suggestive only, because of the @10 proxy.
- No unit has an interval fully below 0 (0 of 48 for each pair), and none has a negative mean with sign p below 0.05.
- int4norm beats int4 with an interval above 0 on 25 of 48 units, and int4norm(mean) beats int4(mean) on 8 of 48.
- This means no harm was detected at @10 with these sample sizes. It is not evidence of zero harm. About 96 looks were made with no multiplicity correction.

## 3. EVIE-4.5B TabFQuAD (Q8), confirmed with a correction
- The 0.929 is the lower bound of the int4(mean) interval in f5 (point 0.966). int4norm(mean) is 0.959 with interval [0.927, 0.990].
- All 15 shared rows between f5 and s17 are identical (0.0 difference), so the batch is deterministic. It is not an independent replicate.
- float32 puts the relevant page at ranks 1 to 4 in 62, 65, 68 and 70 queries. int8 changes 51 of 280 queries' @10 value, against 0 to 8 for the other page units (nemotron DocVQA 45 to 50). EVIE's decoded int4 norm-ratio sd is 0.0040 (TabFQuAD) against 0.0039 (DocVQA).
- Paired @10 differences (mean, interval):
  - int4(mean) vs int4: -0.0225 [-0.049, +0.003].
  - int4norm(mean) vs int4norm: -0.0334 [-0.056, -0.011], sign p 0.08.
  - int4norm(mean) vs float32: -0.0234 [-0.043, -0.004].
  - int4norm(mean) vs int4(mean): -0.0067 [-0.027, +0.014].
- Correction: the pattern is compatible with a centring cost on this unit, and storing the norm adds little on top. It is one unit and one dataset among many looks, and the sign tests are not significant (p 0.08 to 0.2). The cause is not established, and the "near-tied scores" explanation is unverified because scores are not stored.

## 4. Interval widths and flip risk
- Median 95% ndcg@5 retention interval widths for int4norm(mean): SciFact 0.016, NFCorpus 0.025, SciDocs 0.027, DocVQA 0.017, TabFQuAD 0.022. These are comparable to the 0.01 Q6 tolerance. EVIE TabFQuAD int4norm(mean) is 0.063 wide, about 3 times the TabFQuAD median.
- Confirmed: the colpali Q6 check fails in the hybrid bootstrap with P about 0.60.
- Confirmed point values: Q2 misses mLateOn NFCorpus 0.9406 and mxbai-edge-17m NFCorpus 0.9445. Q3 LateOn is 0.8702 and 0.9006. Q7 nemotron-4b TabFQuAD is 0.743 (lower bound 0.692).
- Unverified: the other flip probabilities, namely P(any of 92 fails)=0.96, Q2 P(exactly 8)=0.53, Q3 P(<0.9)=0.47 and Q8 P(pass)=0.08. They depend on independence and normal-approximation assumptions. "Q6 was close to guaranteed to fail on noise alone" overstates what the data supports. Say instead that the Q6 tolerance of 0.01 is no wider than the typical noise on the point estimates.

## 5. Summary
Within these 48 units and @10 per-query data, no harm from storing the norm was detected. The two highlighted gaps (colpali DocVQA @5 -0.0116, EVIE TabFQuAD) are within the noise of their units, or in EVIE's case possibly a centring effect that this data cannot establish. The Q7 and Q8 failures are not resampling artefacts, while Q2 and Q3 sit near their thresholds.