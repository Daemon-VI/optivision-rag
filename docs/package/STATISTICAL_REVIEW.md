# Statistical review of the selection procedure: internal second reader

**What this document is and is not.** This is an INTERNAL second-reader review of the selection procedure of the OptiVision package (betting p-values, fixed-sequence testing inside chains, a union bound over chains, smallest-size choice among the certified candidates). It was written by an automated reviewer that did not write the derivation or the code, from the files in the worktree `optivision-package` on 2026-10-06. It is NOT an external peer review, it is not a substitute for one, and it must never be cited as external validation of the method, of the code or of any guarantee. It does not widen any claim of the package or of the research record: "correct" below always means "correct under the stated conditions, as far as this reader could check". The sentence "the derivation is this project's own and has not been externally reviewed" must stay in every user-facing text until a real external review exists.

## 1. Verdict table

| Q | Subject | Verdict | In one line |
|---|---|---|---|
| A | Betting p-value | CORRECT UNDER STATED CONDITIONS | Positive test supermartingale whenever E[Y_t given the past] <= m0 (i.i.d. is sufficient, not necessary); p = 1/max K is valid for the composite null and every n. Sampling without replacement: the supermartingale property FAILS (explicit example, size 1.6e-4); validity there CANNOT BE DETERMINED (simulations conservative, no proof); no without-replacement version is implemented. |
| B | Ratio to bounded mean | CORRECT | R <= T iff E[c - T b] <= 0 exactly when E_Q[b] > 0; Y in [0,1] exactly in floating point; the sample zero-mean guard cannot affect the joint event; R > 1 is harmless. |
| C | Fixed sequence + union bound | CORRECT | "Selected candidate has R < T" is contained in "some true null rejected", which is contained in the union over chains of "first true null of the chain rejected" (each <= delta/F). No monotone order, no independence, no restriction on the choice rule is needed. |
| D | What is controlled | CORRECT AS A JOINT STATEMENT; WORDING DEFECTS | P(selected and R < T) <= delta. P(R < T given selected) is not controlled (example: 1.00 while the joint probability is 0.011). One logically wrong phrase; the recorded statement omits the disclaimers that the certificate spec attributes to it. |
| E | Engine additions | CORRECT UNDER STATED CONDITIONS, item by item | (1) seed: yes if not shopped; (2) budget: yes, even chosen after looking (certified sets are nested); (3) dropping: statement is about the labelled sub-population; (4) fallback: never certified; (5) Bonferroni: valid; (6) raise = abstention; (7) several looks: NOT covered, inflation measured up to x14 and above 0.05; (8) no data path into list or order, but guard gaps. |
| F | Assumptions A1-A6 | NOT SUFFICIENT AS WORDED | Missing: one look with fixed T/delta/rule/metric/reference, per-query metric on the same queries, reference fixed independently, deployed = evaluated. A1 and A5 are fragile: every query counted twice gives 0.21 against 0.05; an order sorted by a covariate gives 0.28 at a test level of 0.05. "i.i.d." and "documents only" are stronger than needed. |
| G | Power | CORRECT AND CONSISTENT | n_best = (ln(F/delta) + kappa^2/8) / E[ln(1 + rho b)]; the first-passage computation reproduces all 16 deterministic numbers of METHOD_C_SPEC TV3 exactly, the closed form is within 1 query of them, and within 1% of Monte Carlo medians. Documented guidance is consistent; three wording remarks. |
| H | Monte Carlo of the implementation | NO EXCEEDANCE, NO BLOCKER | 71 cases, about 612,000 replicates of the 40-candidate, 7-chain procedure; largest false-selection frequency 0.0377 [0.0321, 0.0441] against 0.05. |

## 2. Findings index

BLOCKER: none. Severity definitions are in section 2A.

| ID | Sev | Where | Finding |
|---|---|---|---|
| SR-1 | MAJOR | `rules.py:51-52`; `guide/SELECTION_AND_CERTIFICATION.md:163`; `core/evaluate/evidence.py:193-220`; `core/evaluate/collect.py:113-122` | Duplicated or clustered queries break A1 and inflate the error several-fold (every query counted twice: 0.211 against 0.05); the docs say only that near-duplicates "weaken independence"; nothing detects duplicate query vectors and `Evidence.subset` can create repeated positions. |
| SR-2 | MAJOR | `rules.py:60-61`; `engine.py:54-77` (default order "given"); `guide/SELECTION_AND_CERTIFICATION.md:167` | The default order is the order as given and A5 says "fixed before the outcomes are seen". An order that depends on a covariate correlated with the outcome (sorted by difficulty, date, source) inflates the error (0.279 at test level 0.05; 0.0795 at 0.05/7); a user-chosen seed can be shopped (best of 20 seeds: 0.076; of 100: 0.169). |
| SR-3 | MAJOR | `rules.py:40-48`; `core/artifact/record.py:76-78`; `CERTIFICATE_SPEC.md:244-274` | `STATEMENT` says it holds "only under assumptions A1-A6" (logically wrong) and lacks the "joint, not conditional / not the sample retention / not a future batch" sentences that CERTIFICATE_SPEC section 5 attributes to the certificate; the certificate stores `STATEMENT` verbatim. |
| SR-4 | MAJOR | `rules.py:50-65`, `67-73` | A1-A6 are not a sufficient list: the proof also needs one (T, delta, rule, metric, reference) chosen before the outcomes and used once on the sample, per-query metrics that are functions of the single query on the same queries for baseline and candidates, a reference fixed independently of the candidates, and (for the statement to concern the deployed index) deployed = evaluated. |
| SR-5 | MINOR | `record.py:76-78`; `engine.py:149` | Any rule object with `certifying = True` is stamped with the Method C statement and `certified=True`. |
| SR-6 | MINOR | `statistics.py:60-100`; `rules.py:51-52` | Finite pool sampled without replacement is not covered (no proof; not stated in A1); no without-replacement test is implemented. |
| SR-7 | MINOR | `METHOD_C_SPEC.md:334`; `E1/DESIGN.md:135`; `E1/RESULTS.md:171`; `guide/SELECTION_AND_CERTIFICATION.md:181-208`; `api.py:421` | Budget guidance: the quoted floors (392/723, 129/238, 77/141) are for any valid test at 50% power; the implemented bet needs 1.3-1.7 times more even for a lossless candidate; the literal sentence "no valid method can certify T = 0.99 with <= 225 queries" is imprecise; `certify` documents "at least one" query. |
| SR-8 | MINOR | `guide/SELECTION_AND_CERTIFICATION.md:166`; `statistics.py:271-281` | A4 "can the software check it? yes" overclaims: the checks are on the sample; E_Q[b] > 0 is a population condition. |
| SR-9 | MINOR | `guide/SELECTION_AND_CERTIFICATION.md:169, 230-232`; `api.py:80-86, 453-458`; `engine.py:11-15` | The warning about several looks has no numbers; section 7.7 supplies them. |
| SR-10 | MINOR | `FINAL_RESEARCH_BOUNDARY.md:157-166`; `cli_certify.py:103, 125`; `api.py:24, 110, 201, 414`; `rules.py:104, 138` | "conditional on" is used for assumptions in sentences that also discuss conditional probability; CLI headline "select (certified)". |
| SR-11 | MINOR | `rules.py:55-57`; `core/evaluate/collect.py:243-282`; `docs/package/guide/LIMITATIONS.md:43-52` | A3 is violated in letter when the calibration queries come from the pools that shaped the default list; the deployed index and retrieval procedure must be the evaluated ones. |
| SR-12 | NOTE | `guide/SELECTION_AND_CERTIFICATION.md:175-179` | Optional stopping, post hoc budget and continuation of the same order are valid (anytime validity); the guide is more conservative than needed. |
| SR-13 | NOTE | `rules.py:76-95` | The whole certified set carries the statement simultaneously; any rule that picks inside it is covered. |
| SR-14 | NOTE | section 10 | The union bound is within about 25% of the worst observed frequency (0.0377 of 0.05). |

## 2A. Scope, method, conventions

- Read: `reports/research/E1/DESIGN.md`, `E1/RESULTS.md` (sections 2, 6-9 and the E1.2 confirmation), `NEXT_PHASE_PREREGISTRATION.md` sections 16.2-16.3, `docs/package/METHOD_C_SPEC.md` (all), `scripts/research/e1_stats.py`, `src/optivision/core/selection/{statistics,rules,engine,decision,spaces}.py`, `core/evaluate/{reference,evidence,collect}.py`, `core/artifact/record.py`, `src/optivision/api.py`, `src/optivision/cli_certify.py`, `docs/package/{CERTIFICATE_SPEC,FINAL_RESEARCH_BOUNDARY}.md` and `docs/package/guide/{SELECTION_AND_CERTIFICATION,LIMITATIONS}.md`. Other engineers were editing the worktree while this review was written (HEAD 1d68db4 plus uncommitted files); line numbers are as read on 2026-10-06 and may drift.
- Run: Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1 (`../optivision-rag-v2/.venv`), `PYTHONPATH=src`, SYNTHETIC data only, the package functions called unchanged. No research experiment script (`x1_agreement.py`, `x2_budget.py`, `x3_symmetry.py`, `e1_*.py` mains) was run; `scripts/research/e1_stats.py` (pure functions, no main) was imported once for a bitwise parity check on synthetic arrays. Nothing was installed, no other file was modified, no git state was changed. Scratch scripts and raw outputs are in `C:\Users\Lenovo\AppData\Local\Temp\claude\C--Users-Lenovo\1a357e2a-547c-499d-8080-bbcf9187bf4e\scratchpad\statreview\` and are not part of the repository (seeds are listed in section 13).
- Notation. T target, delta error level, m0 = T/(1+T), Y = (c - T b + T)/(1+T), F number of chains, K number of candidates, n queries, R_Q(k) = E_Q[c_k]/E_Q[b]. "Boundary" means R_Q = T exactly. g = (1-T)/(1+T), rho = 0.9(1-T)/T, kappa = 0.9(1+T)/T. Intervals are 95% Clopper-Pearson and describe Monte Carlo error only.
- Severity: BLOCKER = the stated error level is false for the implementation under its stated assumptions. MAJOR = the stated level can be false in plausible use with no warning, or a user-facing statement is wrong or contradicts the code. MINOR = imprecision, missing guard or missing number. NOTE = observation, no change required.

## 3. A. The betting p-value

**Verdict: CORRECT UNDER STATED CONDITIONS.**

As implemented (`statistics.py:89-99`; with 1-based t the loop's `t+1` and `t+2` are t and t+1):

- mu_t = (1/2 + sum_{s<t} Y_s)/t and v_t = (1/4 + sum_{s<t} (Y_s - mu~_s)^2)/t with mu~_s = (1/2 + sum_{r<=s} Y_r)/(s+1): prior mean 1/2 and prior variance 1/4, each one pseudo-observation.
- lam_t = clip((mu_t - m0)/(v_t + (mu_t - m0)^2), 0, 0.9/m0).
- K_t = prod_{s<=t} (1 + lam_s (Y_s - m0)), K_0 = 1; p = min(1, 1/max_{0<=t<=n} K_t), the maximum including the starting wealth 1.

Argument.

1. Predictable: `s1` and `s2` are updated after the multiplier of observation t is applied, so lam_t uses Y_1..Y_{t-1} only. The squared deviation added to `s2` uses a mean that includes Y_s but affects bets for t > s only.
2. Positive: lam_t is in [0, 0.9/m0] and Y_t in [0,1], so every multiplier is >= 1 - lam_t m0 >= 0.1; K_t >= 0.1^t > 0 and `log1p` never receives an argument <= -1. v_t >= 1/(4t) > 0: no division by zero.
3. Supermartingale: E[1 + lam_t (Y_t - m0) | Y_1..Y_{t-1}] = 1 + lam_t (E[Y_t | past] - m0) <= 1 whenever E[Y_t | past] <= m0, because lam_t >= 0.
4. Ville (1939): P(sup_t K_t >= 1/a) <= a. Since p <= a iff max K >= 1/a, P(p <= a) <= a for every a in (0,1] and every n, and P(exists n: p_n <= a) <= a.

Composite null: yes. The null is E[Y] <= m0, not only the boundary: step 3 holds for every conditional mean below m0 because the bet is never negative (the clip at 0). Allowing negative bets would break exactly this.

What is needed: E[Y_t | Y_1..Y_{t-1}] <= m0 for every t, not i.i.d. Independent, non-identically distributed Y_t with E[Y_t] <= m0 each satisfy it. Any dependence under which the conditional mean of the next value rises above m0 after an informative past does not (duplicates, clusters, sampling without replacement, an order correlated with outcomes; section 8).

Edge cases. Y exactly 0 or 1: allowed (Bernoulli(m0), two-point and heavy-mass laws are simulated). m0 = 1/2 (T = 1): the first bet is 0, no problem. T small: cap 0.9/m0 large, multiplier still positive. n = 1: p = 1/(1 + lam_1 (Y_1 - m0)), lam_1 = (1/2 - m0)/(1/4 + (1/2 - m0)^2); p(Y=1) = 0.974 at T = 0.95, 0.995 at T = 0.99, 0.714 at T = 0.5, p(Y=0) = p(Y=m0) = 1; so for T >= 0.5 no level below 0.71 is reachable at n = 1. Y is exactly inside [0,1] in floating point (section 4).

Without replacement from a finite pool. The wealth is NOT a supermartingale in general. For a population of N values with mean mu <= m0, E[Y_t | past] = (N mu - S_{t-1})/(N - t + 1), which exceeds mu after a low past. The bet is positive when (1/2 + S_{t-1})/t > m0, that is also when the past mean is below m0 by up to (1/2 - m0)/(t-1) = g/(2(t-1)): the pseudo-observation 1/2 pulls mu_t up. Explicit example, T = 0.95 (m0 = 0.487179): population {0.48, 0.494359}, mean m0. After Y_1 = 0.48 the next bet is lam_2 = 0.02255 > 0 and the other value is certain: E[Y_2 | Y_1] = 0.494359 > m0, so E[K_2/K_1 | Y_1] = 1.000162 > 1. In general the one-step excess is at most rho/(2(N-t+1)) (0.047/(2(N-t+1)) at T = 0.95), and only on paths whose running mean sits in the narrow band just below m0. This is a failure of the proof device, not necessarily of validity. Evidence on validity:

- Exact enumeration of the whole population (n = N <= 7) for every population on the grid {0, .25, .5, .75, 1} with mean <= m0 (358 populations at T = 0.95, 176 at T = 0.5): largest exact P(p <= 0.5) = 0.071 (0.075), P(p <= 0.3) = 0.007 (0), and no rejection at any level <= 0.2.
- Monte Carlo (seed 20261011): N in {20, 50, 100, 225}, n in {N, N/2, N/10}, three boundary populations (Bernoulli-like, rare-low, nDCG-like), 20,000 random permutations against 20,000 with-replacement draws per cell, levels 0.05, 0.05/7, 0.01: the without-replacement rate is below the with-replacement rate in every cell; largest without-replacement rate/level 0.379, largest with-replacement 0.72 (rare-low population, N = n = 225: 0.0174 against 0.0354 at level 0.05).
- Hoeffding (1963, section on sampling without replacement) shows that a without-replacement sum is dominated in the convex order by the with-replacement sum, which gives validity of fixed-n Hoeffding-type bounds. I did not find an argument that carries this over to a product of data-adaptive bets. Verdict for this sub-question: CANNOT DETERMINE (no proof, no violation found).
- The without-replacement betting construction of Waudby-Smith and Ramdas replaces m0 in the multiplier by the conditional mean of the remaining population, (N m0 - S_{t-1})/(N - t + 1) (NeurIPS 2020, confidence sequences for sampling without replacement; the JRSSB paper has a finite-population extension; I am not certain of section or theorem numbers). It is the right test when the queries are the whole population. The package does NOT implement it. `METHOD_C_SPEC.md:126-140` and `E1/RESULTS.md:191-193` already say that sampling without replacement is not covered; `rules.py` A1 and the guide do not (SR-6).

Simulation, p-value level (`sim_A.py`, seed = 20261006 + 7*int(1000 T) + n). 14 laws per cell, each with E[Y] = m0 exactly except two COMPOSITE laws with E[Y] < m0: Bernoulli(m0) (Y in {0,1}), two-point {0, 2 m0}, rare-high (Y = 1 w.p. 0.02, 0.1), rare-low (Y = 0 w.p. 0.02, 0.1, 0.4), uniform[0, 2 m0], Beta(0.3, .) U-shape, symmetric steps, nDCG-like rare total loss (R = T), nDCG-like frequent small loss (R = T). Cells: T = 0.95 with n = 5, 20, 50, 225, 1000; T = 0.99 with n = 225, 1000; T = 0.90, 0.5, 1.0 with n = 225. 100,000 replicates per cell (40,000 at n = 1000). Levels 0.9, 0.5, 0.1, 0.05, 0.05/7, 0.01, 0.05/40. Result: no cell has a lower 95% bound above its level, at any of the seven levels. Largest P(p <= a)/a over the 14 laws:

| T, n | a = 0.05 | 0.05/7 | 0.01 | 0.05/40 |
|---|---|---|---|---|
| 0.95, 5 | 0 | 0 | 0 | 0 |
| 0.95, 20 | 0.45 | 0.29 | 0.32 | 0.26 |
| 0.95, 50 | 0.53 | 0.44 | 0.47 | 0.55 |
| 0.95, 225 | 0.70 | 0.72 | 0.70 | 0.62 |
| 0.95, 1000 | 0.74 | 0.79 | 0.77 | 0.76 |
| 0.99, 225 | 0.69 | 0.68 | 0.67 | 0.74 |
| 0.99, 1000 | 0.75 | 0.76 | 0.79 | 0.94 |
| 0.90, 225 | 0.68 | 0.72 | 0.69 | 0.62 |
| 0.5, 225 | 0.65 | 0.63 | 0.65 | 0.59 |
| 1.0, 225 | 0.67 | 0.72 | 0.70 | 0.74 |

The ratio grows with n (at T = 0.95 and level 0.05/7: 0.44 at n = 50, 0.72 at n = 225, 0.79 at n = 1000); the 0.94 is the two-point law {0, 2 m0} at T = 0.99, n = 1000: 47 rejections in 40,000 (interval [0.00086, 0.00156] against the level 0.00125). Adversarial search (`sim_A2.py`, seeds 20261007/20261008): 150 random three-point laws with mean exactly m0 (support random, 0 and 1 included with probability 1/2 each) per n in {50, 225} at T = 0.95, screened at 20,000 replicates, top 4 re-run at 200,000: largest ratio 0.58 (n = 50) and 0.66 (n = 225). A random search, not an optimisation.

Parity: the package's `to_y`, `betting_pvalues`, `fixed_sequence_by_family`, `smallest` against the frozen `scripts/research/e1_stats.py` on 400 random synthetic problems (K 1-44, n 1-399, many size ties, seed 20261901): p-values bitwise equal 400/400, rejected sets and selections identical 400/400.

Required changes: none to code. Docs: SR-6.

## 4. B. Reduction from the ratio to a bounded mean

**Verdict: CORRECT.**

- Identity. For E_Q[b] > 0: R_Q(k) <= T iff E[c_k] <= T E[b] iff E[c_k - T b] <= 0 iff E[Y_k] <= m0 (multiply by 1+T, subtract T). Rejecting H0_k therefore claims R_Q(k) > T, strictly. Exact rational check on 20,000 random finite populations: 0 disagreements (`sim_B.py`, seed 20261801).
- Range. c - T b is in [-T, 1] for c, b in [0,1], T in (0,1], so Y is in [0,1]. In floating point `to_y` gave min 0.0 and max 1.0 over 2,000,000 random and extreme (c, b) pairs (0, 1, nextafter values, 1e-300) at 47 targets including 1e-12, 1.0 and nextafter(1,0) (seed 20261801); rounding is monotone, so the bounds hold exactly. The package's metric values cannot leave [0,1] by rounding either: 90,000 per-query values (nDCG@5/10, recall, MRR, hit; ties; multi-relevant queries; seed 20262001) had min 0.0, max exactly 1.0, so the range guard (`statistics.py:271-275`) cannot raise on a legitimate nDCG.
- E_Q[b] = 0. The identity needs E_Q[b] > 0. Since b >= 0, E[b] = 0 means b = 0 almost surely, so the sample baseline mean is 0 almost surely and `validate_evidence_arrays` raises (`statistics.py:279-281`): a population violation is caught with probability 1. The guard also fires, harmlessly, when E[b] > 0 is small and every sampled b happens to be 0: probability P(b = 0)^n, e.g. 0.313 / 0.100 / 0.035 / 0.0024 at n = 1 / 2 / 3 / 5 for a DocVQA-like baseline (P(b = 0) = 0.32), measured in the small-n cells of section 10. A raise is not a selection, so the joint event is unaffected (E6). The guard is a check of the sample; it cannot verify the population condition (SR-8).
- Retention above 1: no issue (Y stays in [0,1], the identity is unchanged); it means the compressed index beats the baseline on average. Under the agreement construct R_A <= 1 and Y <= 1/(1+T).
- R_Q is a ratio of means: queries are weighted by b, it is not the mean of per-query ratios, and it says nothing per query (documented). Because the ratio is unchanged by adding or removing queries on which c = b = 0, dropping unlabelled queries (E3) is the same as scoring them 0 for the baseline and for every candidate.

Required changes: none in code (SR-8 wording).

## 5. C. Fixed-sequence testing inside a chain and the union bound

**Verdict: CORRECT.**

Proof. Let N = {k : R_Q(k) <= T} be the true nulls. For chain f containing a true null let j*_f be the first one in list order; it is not random (it depends on Q, the list and the order only, which A3 fixes). The chain rejects (f, j) iff p_(f,i) <= delta/F for all i <= j. If chain f rejects any true null (f, j), j >= j*_f, it has rejected (f, j*_f), so {chain f rejects a true null} is contained in {p_(f,j*_f) <= delta/F}, which has probability <= delta/F by section 3. The union over at most F chains gives P(some true null is rejected) <= delta. The selected candidate is a function of the rejected set, so {selected has R_Q < T} is contained in {selected is in N} which is contained in {some true null rejected}. The bound is the family-wise error of all K hypotheses (strong control).

- Data-dependent choice among rejections (smallest stored size over all chains): irrelevant; the bound is on a superset event. The same holds for ANY rule that picks a member of the certified set, simultaneously (SR-13: a user may choose by latency or by anything else among `certified_positions`; the winner's-curse bias of the sample retention remains).
- Monotone order: not needed. A non-monotone chain costs power only: it stops at its first non-rejection, so later candidates are unreachable however good (chain 6 of the "mixed" profile has retentions 0.99, 0.90, 0.98, 0.99, 0.70, 0.50 at T = 0.95; the 0.98 and 0.99 members are never tested). It does not raise the error: the first true null is the same.
- Same queries for every candidate (dependence): not a problem for the union bound (distribution-free) nor for fixed-sequence (one marginal p-value per chain is used). Positive dependence lowers the actual error: boundary for all 40 candidates at n = 800, independent chains 0.0333 and shared fragility inside chains 0.0377 (same law for the chain heads, difference is noise), all chains sharing one fragility draw 0.0047 (section 10).
- Equal split of delta over chains is one valid choice (any weights summing to 1 are valid). F is the number of distinct labels in the list passed (`rules.py:179`); interleaved chains behave as separate chains (the `stopped` set is per label). Duplicated pipelines in two chains are tested twice; the union bound counts them twice (conservative).
- Tightness: with F independent chain heads at the boundary the error is 1 - (1 - q)^F with q the per-test rejection probability; q is at most 0.8 of delta/F in the simulations (section 3), so the worst observed total is about 0.04.

Required changes: none.

## 6. D. What exactly is controlled

**Verdict: CORRECT AS A JOINT STATEMENT; WORDING DEFECTS (SR-3, SR-10).**

Controlled event. Fix the corpus, the candidate list with its order, T, delta, the rule, the metric and the reference. The calibration queries X_1..X_n are i.i.d. draws from Q (each X_i carries its reference and gives (b_i, c_1i, ..., c_Ki)); a raise counts as no selection. Then

E = { the procedure selects a compressed candidate k-hat } and { R_Q(k-hat) < T },   P_{Q^n}(E) <= delta,

and the proof bounds the larger event {selected and R_Q(k-hat) <= T}. The probability is over the draw of the queries only; the truth R_Q(k) is fixed, not random. This is a joint probability. It is NOT P(R_Q(k-hat) < T | selected) = P(E)/P(selected), which can be 1.

Numeric example (`sim_D.py`, seed 20262101, 20,000 samples per row). One candidate with R_Q = 0.94, T = 0.95, delta = 0.05, one chain (level 0.05):

| n | P(selected) = P(selected and R < T) | P(R < T given selected) |
|---|---|---|
| 225 | 0.0113 [0.0099, 0.0129] | 1.0000 |
| 450 | 0.0118 [0.0104, 0.0134] | 1.0000 |
| 800 | 0.0110 [0.0096, 0.0125] | 1.0000 |
| 1500 | 0.0120 [0.0105, 0.0136] | 1.0000 |

The statement holds (0.011 <= 0.05) and every selection is wrong. The 40-candidate default-shaped list with all candidates at 0.94 gives 0.0070 (n = 450) and 0.0093 (n = 800) with conditional probability 1. Across deployments: if one deployment in ten has a candidate at R = 0.99 that is selected with probability 0.5 and nine in ten have only candidates at 0.94 (selected with probability 0.0113), then 0.9 x 0.0113 / (0.9 x 0.0113 + 0.1 x 0.5) = 17% of all selections are below the target, 3.4 times delta, although the statement holds in every deployment. Conversely a rule that never selects satisfies the statement. The statement says how often the procedure goes wrong, not how reliable one selection is.

Wording audit (new sentences in section 11).

| Location | Sentence | Problem |
|---|---|---|
| `rules.py:40-48` | "It holds for every sample size, with no large-sample approximation, and only under assumptions A1-A6." | "only under" says the statement is false if an assumption fails; the correct claim is that it is proved under them and nothing is claimed otherwise. T, delta and Q are not defined in the string. |
| `rules.py:40-48` (copied verbatim by the certificate, `record.py:76-78`, and the guide `SELECTION_AND_CERTIFICATION.md:109-114`) | whole `STATEMENT` | The joint wording ("is selected and its retention ... is below T") is correct, but the string does not say that it is not the conditional probability, not the sample retention and not a future batch. `CERTIFICATE_SPEC.md:244-274` says the certificate carries these sentences and then admits (lines 272-274) that its text "paraphrases" the recorded one; the recorded one lacks them. |
| `FINAL_RESEARCH_BOUNDARY.md:157-166` | "More than the certifying rule's own statement: the probability ... is at most the error level, ..., **conditional on** independent queries ..." | The heading reads as "the package does not guarantee more than"; "conditional on" in a sentence about a probability reads as a conditional probability; the next sentence then says it is not one. |
| `cli_certify.py:125` | "what the rule states (conditional on every assumption below):" | Same double use of "conditional". |
| `cli_certify.py:103` | headline "select (certified)" in bold green | Can be read as "this candidate is certified to retain T". |
| `api.py:24, 110, 201, 414` | "conditional on the assumptions it lists" | Same double use (correct in meaning, ambiguous in a document about conditional probability). |
| `guide/SELECTION_AND_CERTIFICATION.md:146-149`, `CERTIFICATE_SPEC.md:259-262` | "Not a probability conditional on a selection ..." | CORRECT as written. |
| `api.py:411-416` | "the probability that a compressed candidate is selected and its retention under the query distribution is below target is at most error" | CORRECT as written. |

Statement versus proof: the proof covers "R <= T"; "below T" is therefore safe. In the Monte Carlo the boundary cases R = T are counted in the "R <= T" column.

## 7. E. The additions in the engine

### 7.1 (1) `order_seed`

**Verdict: CORRECT UNDER STATED CONDITIONS.** `_query_indices` (`engine.py:54-77`) applies `np.random.default_rng(seed).permutation(n)` to the n kept queries (n depends on labels only). Under A1 the evidence columns are i.i.d., a permutation chosen independently of the data leaves their law unchanged, so the law of every p-value is unchanged and the statement holds at the same delta, conditional on the seed. Conditions: (i) the seed is chosen before the outcomes and used once; (ii) it is not a function of outcomes (a hash of the evidence arrays is a function of the outcomes; a hash of the query ids is not, provided the ids carry no outcome information); (iii) it repairs order-outcome dependence, not dependence among queries.

Simulation (`sim_E.py`, seed 20261203): K = 7 independent chain heads at the boundary (R = T = 0.95), n = 225, level delta/7, 2,000 samples, report "any head rejected under any of S permutations": S = 1: 0.0120 [0.0077, 0.0178]; S = 2: 0.0170; S = 5: 0.0285 [0.0217, 0.0368]; S = 20: 0.0760 [0.0648, 0.0885]; S = 100: 0.1690 [0.1528, 0.1862] (bound 0.05). Seed shopping is therefore a real inflation: 20 seeds already exceed delta. The permutation helps when the given order is outcome-dependent (section 8: drift, adverse order 0.104 against 0.024 permuted; covariate-sorted 0.279 against 0.013 in random order).

Engine equivalence (`sim_E.py`, seed 20261201): 40 mixed samples, four paths (given, `order_seed`, `query_budget`, both) through `select()`: certified positions, selection and p-values equal `FixedSequenceBetting.decide` on the explicitly permuted or truncated arrays in 160/160.

Required changes: SR-2 (default to a seeded permutation whose seed is derived from the query ids, or document that user-chosen seeds can be shopped).

### 7.2 (2) `query_budget`

**Verdict: CORRECT, EVEN IF m IS CHOSEN AFTER LOOKING.** p_k(m) = 1/max_{t<=m} K_{k,t} is non-increasing in m and `fixed_sequence_by_family` is monotone in p, so the certified set at a budget m is contained in the certified set at any larger budget m' (same order). Hence {a false candidate is selected at some budget m <= n} is contained in {a true null is certified at n}, which has probability <= delta: a budget picked after looking, or several budgets tried and one reported, give no inflation, and reporting the "best" budget is reporting the full-n decision. Verified through `select()` on 40 samples at budgets 25, 50, 100, 150, 225: certified sets nested 40/40. Conditions: a PREFIX of the same order (same seed), not a subset chosen by outcome; same T, delta, list.

Consequences. Optional stopping and continuation are valid too (Ville applies to the whole path): appending new queries at the END of the same order and re-running is a continuation of the same test and keeps level delta. `certify()` is stateless: passing the old queries followed by the new ones in the same order, without `order_seed`, is such a continuation; with `order_seed` the whole enlarged sample is re-permuted, which is a different order and a data-dependent n (not a continuation). The package neither records nor checks whether a call is a continuation. The guide (`SELECTION_AND_CERTIFICATION.md:175-179`) advises to decide the number of queries before looking; this is conservative and the review does not widen it (SR-12). In `certify()` the budget saves no scoring: evidence is collected on all queries before it is applied.

### 7.3 (3) Dropping queries with no relevant document

**Verdict: CORRECT UNDER STATED CONDITIONS; THE STATEMENT IS ABOUT THE LABELLED SUB-POPULATION.** The rule (`collect.py:178-184`: relevant set empty, label-only; the same set for baseline and every candidate) depends on the reference only. Given i.i.d. draws from Q, the kept queries are i.i.d. from Q_lab = Q(. | at least one relevant document present in this corpus); n is random but the statement holds conditionally on n, hence marginally (n = 0 raises, a non-selection). So the statement is about R_{Q_lab}. It says nothing about the dropped queries (out-of-scope queries, queries whose labelled ids are absent from the corpus: `LIMITATIONS.md:100-103` notes an id mismatch silently removes queries). `report()` shows `n_queries_without_relevant_document` but no statement sentence mentions that the statement concerns the labelled queries (SR-4, W5). Because R is a ratio of means, R_{Q_lab} equals the retention of the full stream under the convention "unlabelled queries score 0 for the baseline and for every candidate".

Agreement construct (b = 1 by construction; c = 1/log2(1+r) if the candidate ranks the reference's top document at r <= k, else 0): nothing is dropped (every query has a top-1 reference; `RankingReference.relevant`); A4 holds trivially (E[b] = 1); Y = c/(1+T) mathematically (the code keeps the frozen operation order) lies in [0, 1/(1+T)] and R_A <= 1; the reference is a deterministic function of the query and the corpus including the tie-break by document index; the arithmetic is identical. The wealth is capped by data (Y - m0 <= (1-T)/(1+T)), so after the ramp-up the bet sits at the cap and TV3 gives the exact best case; with the cap at 0.99 instead of 0.9 about 9% fewer queries would be needed (validity unaffected; not a recommendation to change a frozen constant). The statement is about agreement, not retrieval quality (documented).

### 7.4 (4) The explicit `fallback`

**Verdict: CORRECT.** `engine.py:148-153` has three branches: SELECT (`certified = bool(rule.certifying)`), FALLBACK (`certified = False`, spec with position -1) and ABSTAIN; `decision.py:128-134` raises if a non-SELECT outcome carries `certified` or certified positions; `verification.py:411-419` re-checks this on a stored certificate; `certificate.py:556` and `api.CertifiedResult.certified` read `decision.certified`. The statement's event is "a compressed candidate is selected"; a fallback is returned only when none was selected, so it is outside the event and nothing is claimed for it: its retention may be below T with probability 1. Test (`sim_E.py`, seed 20261201): 60 boundary samples, 60 abstained; with a caller pipeline each gave FALLBACK with `certified=False`, empty certified positions, `selected.position == -1` and p-values identical to the abstain decision; and on 40 further samples in which a candidate was certified (mixed profile, n = 800, seed 20262301) the fallback was ignored and the decision was identical, 40/40. The only path that sets `certified=True` is SELECT under a rule that declares `certifying` (SR-5).

### 7.5 (5) `BonferroniBetting`

**Verdict: CORRECT.** Every candidate is tested at delta/K (`rules.py:235`, K counts all candidates including duplicates), `p <= level` inclusive, no chain, no stopping: the union bound over the true nulls gives P(some true null rejected) <= (number of true nulls) delta/K <= delta. No order is needed and the choice rule is irrelevant. Monte Carlo (`sim_H2.py`, seeds 20261165-20261169, 4,000-10,000 replicates): boundary for all candidates n = 225 (independent and shared fragility), mixed profile n = 800: false selection 0.0000 [0, 0.0004] to [0, 0.0009] in all four cases; mean certified count 13.3 against 12.7 for the fixed-sequence rule on the mixed profile (the mixed profile has non-monotone chains).

### 7.6 (6) Validation that raises

**Verdict: CORRECT; A RAISE IS AN ABSTENTION.** The outputs are select, abstain or raise, and E requires a selection. Formally, for the zero-mean sample baseline (where p is defined and valid) put p~ = 1 on the raising set: p~ >= p pointwise, so P(p~ <= a) <= P(p <= a) <= a and the proof is unchanged. NaN, out-of-range values and n = 0 are outside the model (A4, the data contract); they are raised before any decision, so nothing is decided on them. The only effects are lost power (a raise at n = 1 occurs in 31% of DocVQA-like samples, 10% at n = 2) and workflow selection: a user who reacts to a raise by dropping the offending queries or clipping values makes a data-dependent edit that is outside the statement; the package itself never does that (`collect.py:18-23`, `statistics.py:232-238`). A raise on a prefix budget (all b = 0 in the first m queries) is data-dependent but harmless. Monte Carlo small-n cells treat raises as abstentions (section 10).

### 7.7 (7) Several looks on the same sample

**Verdict: NOT COVERED BY THE STATEMENT (correctly so documented); inflation quantified.** Each extra look adds hypotheses or orders to the effective family. All numbers are false-selection frequencies, nominal bound 0.05.

| Action on the same queries | Honest single look | After the action (95% interval) | Script, seed, replicates |
|---|---|---|---|
| Best of S order seeds, S = 2 / 5 / 20 / 100 (7 chain heads at the boundary, n = 225) | 0.0120 | 0.0170 / 0.0285 / 0.0760 [0.065, 0.089] / 0.1690 [0.153, 0.186] | `sim_E.py`, 20261203, 2,000 |
| Delete the first non-rejected candidate of every chain and rerun until stable (plateau of independent boundary candidates, 40 candidates, n = 800) | 0.0120 | 0.0423 [0.0354, 0.0502] (x3.5) | `sim_E3.py`, 20261701, 3,000 |
| Keep the best 3 chains by the first look and rerun at delta/3 (same plateau) | 0.0120 | 0.0193 | same |
| Keep the best chain only and rerun at delta | 0.0120 (independent) / 0.0080 (nested within chain) | 0.0597 [0.0515, 0.0687] / 0.0707 [0.0618, 0.0804] (x5 to x9) | same |
| Same editing on a plateau of near-misses (R = 0.945 and below) | 0.0030 | drop blockers 0.0120; keep best chain 0.0290 / 0.0343 | `sim_E.py`, 20261205, 3,000 |
| Scan 18 targets 0.863-0.953 on the same queries (full procedure at each, candidates exactly at the boundary of one look each), report any wrong look | per-look mean 0.0019, max 0.0037 | 0.0100 [0.0043, 0.0196] (x5 over the mean look); "highest target that certified something" 0.0025 | `sim_E2.py`, 20261601, 800 |
| Several targets at or above the true retention of all candidates (0.95, 0.955, ... 0.99) | 0.031 | 0.031 (in these samples every rejection at a higher target was also a rejection at 0.95; no inflation) | `sim_E.py`, 20261204, 2,000 |

So: seed shopping can multiply the error by 14, list editing by 5-9 (and above delta), a target scan by about 5 in a design where each look has a boundary candidate; the upper bound in general is the number of looks, and with adaptive list editing the effective number of tested hypotheses can grow to K instead of F (K/F = 5.7 for the default list). Several error levels are different: the p-values do not depend on delta, rejections are nested in delta, so reporting the smallest level at which the reported selection is made is an adjusted p-value and is valid in the sense P(false selection reported with level <= d_j) <= d_j for each fixed d_j; what is invalid is claiming a level smaller than the one at which the decision was made.

Required changes: SR-9 (add these numbers where the warning is).

### 7.8 (8) Lists, order and the level

**Verdict: CORRECT; NO DATA PATH BY CODE, TWO GUARD GAPS.** F is counted from the list passed (`rules.py:179`, `spaces.py:259-261`); passing a sub-list after seeing results changes the level and the hypotheses (7.7: keeping only the best chain gave 0.06-0.07). Audit:

| Item | Can data enter? | Note |
|---|---|---|
| `CandidateSpace.default/wide(dim)` (`spaces.py:201-236`) | only via `dim` (a corpus property) and SciPy availability | recorded in `identity`; the reduced list needs `allow_reduced=True` and has another F |
| `from_families`, constructor (`spaces.py:176-199`) | caller | order = mapping order; positions are the tie-break key |
| `certify(space=None)` | list depends on `corpus.dimension` only | defaults `target=0.97`, `error=0.05` are constants |
| `collect_evidence` failure (`collect.py:200-205`) | raises for the whole list and advises "fix the candidate or the space and collect again" | an edit after a failure; harmless if the failure does not depend on the queries |
| `Evidence.subset` (`evidence.py:193-220`) | caller can reorder, drop, or repeat queries | repeated positions allowed (SR-1); `select` does not check ids for duplicates |
| `select` default order (`engine.py:54-77`) | the evidence order, i.e. the caller's | SR-2 |
| `order_seed`, `query_budget` | integers chosen by the caller, recorded | shopping not detectable (7.1); budget harmless (7.2) |
| `rule` | any object with the protocol; `certifying` is trusted | SR-5 |
| `fallback` | not in the space, not in F, not tested | fine |
| the same `Evidence` object | can be passed to `select` any number of times | no counter; the package cannot see several looks |

Suggested hardening (not required for validity): record `space.identity`, target, error and seed in a commitment before collecting queries, so that A3 and A7 can be audited later.

## 8. F. Assumptions

**Verdict: A1-A6 ARE NOT SUFFICIENT AS WORDED** (SR-4); some are stronger than needed.

Missing from `rules.py:50-65`:

1. One look. T, delta, the rule, the metric and cutoff, the reference construct, the list and the order are chosen before the outcomes and used once on the sample. `engine.py:11-15` and the guide table (`SELECTION_AND_CERTIFICATION.md:169`) say this; the six assumptions that every certificate carries do not.
2. Per-query evidence. The metric of query i is a deterministic function of query i alone (plus the fixed corpus, fitted pipeline and reference), baseline and candidates are scored on the same queries, and nothing uses batch statistics of the queries. The code satisfies this (queries never reach `fit`; transforms are per vector; the shared transform cache stores deterministic stage outputs) but the assumption is not stated.
3. The reference. Fixed independently of the candidates' outputs (not pooled from them), queries without a relevant document dropped by a rule that looks only at the reference, so the statement is about the labelled queries (7.3). A6 says "taken as given" only.
4. Deployed = evaluated. Same corpus (A2, partly), same fitted pipeline (`compress_selected(expected_fitted=...)` is optional; `certify` passes it), same retrieval procedure including exact scoring and tie-break (the certified R_Q is the retention of the evaluated procedure; an approximate backend serves a different one).
5. Independence of the queries from the list. The default list was shaped by results on development pools (`LIMITATIONS.md:43-52`); calibration queries taken from those pools violate A3 in letter.
6. The sample size is not chosen by outcomes except through a prefix (7.2).

Stronger than needed: "independent draws" (the proof needs E[Y_t | past] <= m0); "fitted on documents only" (needs: not fitted on these queries or their outcomes; the collector refuses even independent query data via `fit_uses_queries`, which is a safe over-restriction); A2 matters for meaning, not for the proof. A5's wording is too weak in one respect: "fixed before the outcomes are seen" does not exclude an order that was fixed in advance as a function of a covariate correlated with the outcome (sorted by difficulty, date, source); the requirement is independence of the order from the outcomes.

Is "i.i.d. draws from the deployment distribution" what a hand-collected finite set has? Usually not: (a) curated or hard-case selection is a different Q; (b) paraphrases, sessions, shared documents and duplicates are dependent; (c) a finite pool used without replacement is not i.i.d. (section 3); (d) chronological or sorted files carry order-outcome dependence. The documents say "honest sample" and "near-duplicates weaken independence"; the numbers say more (single test at the boundary R = T, n = 225, 20,000 replicates, `sim_F.py`, seeds 20261401-20261403):

| Departure | P(p <= 0.05) | P(p <= 0.05/7) |
|---|---|---|
| i.i.d., order as given | 0.0236 | 0.0015 |
| every query counted twice: adjacent / randomly permuted | 0.144 / 0.077 | 0.0345 / 0.0233 |
| counted 3 times: adjacent / permuted | 0.268 / 0.140 | 0.0987 / 0.0712 |
| counted 5 times | 0.419 / 0.241 | 0.2233 / 0.1794 |
| counted 9 times | 0.599 / 0.423 | 0.3767 / 0.3327 |
| failure rate drifts, mean fixed (a = 0.9): chronological / reversed / permuted | 0.0095 / 0.104 / 0.0239 | 0.0013 / 0.0039 / 0.0019 |
| failure probability depends on the baseline score: random order / sorted high-to-low / low-to-high | 0.0129 / 0.279 / 0.0100 | 0.0001 / 0.0795 / 0.0001 |
| order sorted by Y itself (outcome-dependent) | 1.000 | 0.956 |

Through the whole procedure (`sim_F2.py`, seed 20261501, 4,000 samples, 7 independent chain heads at the boundary, n = 225, level 0.05/7): false selection x1 0.0105; x2 0.211 [0.199, 0.224] adjacent, 0.148 permuted; x3 0.514 / 0.399; x5 0.784 / 0.715 (bound 0.05). Sorting by the baseline score alone is harmless when failures are independent of it (0.028 / 0.024 against 0.027 as drawn, seed 20261403); it matters when failure depends on difficulty.

What the docs should say (W7, W8): sample uniformly from a production log; remove exact and near-duplicate queries before collecting evidence (and by need, not by string); use a seeded permutation unless the given order is the arrival order of an i.i.d. stream; the statement is then about the distribution of that log; a curated, synthetic or sorted set defines another Q. Required changes: SR-1, SR-2, SR-4, SR-11.

## 9. G. Power

**Verdict: CORRECT AND CONSISTENT.**

Derivation (perfect candidate c = b). Then Y - m0 = g b, so a query with b = 0 multiplies the wealth by exactly 1. With the prior counted as one pseudo-observation, mu_t - m0 = g (1/2 + S_b)/t and v_t ~ 1/(4t) (the prior variance dominates g^2 Var(b) at the sample sizes of interest), so lam_t ~ 4 g (1/2 + beta t) grows linearly until it hits the cap kappa = 0.9/m0, after about kappa/(4 g beta) queries (18 at T = 0.95 and b = 1, 90 at T = 0.99). The cap binds asymptotically for every b-law when T >= 0.474 (the unconstrained bet beta/(g E[b^2]) is at least 1/g, and 1/g >= kappa iff T/(1-T) >= 0.9). After the ramp every informative query multiplies the wealth by 1 + rho b, with rho = kappa g = 0.9(1-T)/T. The log-wealth lost in the ramp is kappa^2/8, independent of beta and g. Hence, with alpha' = delta/F,

n_best(T, alpha', beta) = (ln(1/alpha') + kappa^2/8) / E[ln(1 + rho b)],   and E[ln(1 + rho b)] >= beta ln(1 + rho) with equality for b in {0, 1}.

Hard floor, for any candidate with c_i <= b_i on every query: each factor is at most 1 + kappa g = 1 + rho, so p_n >= (1 + rho)^(-n) and n >= ln(1/alpha')/ln(1 + rho) is necessary (property test, 840,000 random sequences, min p_n/(1+rho)^(-n) = 1.000000, `sim_floor.py`, seed 20262201): at T = 0.99, 0.97, 0.95, 0.90 it is 547, 180, 107, 52 queries for F = 7 (alpha' = 0.05/7) and 332, 110, 65, 32 for F = 1 (rounded up).

Verification (`sim_G.py`; the instrumented recursion is bitwise equal to `betting_pvalues` at n = 1, 2, 17, 100, 300; seed 20261301 for the check, 20261302 for the tables). Deterministic best case b = c = 1: the first n with p <= alpha' for the 4 targets x 4 levels of METHOD_C_SPEC TV3 equals TV3 in all 16 cases (377, 592, 555, 770 at T = 0.99; 125, 196, 183, 254; 74, 116, 109, 151; 37, 57, 54, 74) and the closed form is within 1 query of all 20 deterministic values (including alpha' = 0.05/40: 784, 259, 154, 75). Lossless candidate with b ~ Bernoulli(E[b]) (c = b), median n over 1,000 samples (closed form in brackets):

| T | alpha' | E[b] = 1 | 0.9 | 0.6 | 0.3 | floor, any c <= b | DESIGN floor, E[b] = 0.596, 50% power |
|---|---|---|---|---|---|---|---|
| 0.99 | 0.05/7 | 592 (591) | 658 (657) | 985 (985) | 1973 (1971) | 547 | 708 |
| 0.99 | 0.05 | 377 (376) | 419 (418) | 627 (627) | 1254 (1254) | 332 | 384 |
| 0.97 | 0.05/7 | 196 (195) | 218 (217) | 326 (325) | 652 (651) | 180 | 233 |
| 0.97 | 0.05 | 125 (124) | 139 (138) | 208 (207) | 415 (414) | 110 | 126 |
| 0.95 | 0.05/7 | 116 (116) | 129 (129) | 193 (193) | 385 (387) | 107 | 138 |
| 0.95 | 0.05 | 74 (74) | 82 (82) | 122 (123) | 245 (247) | 65 | 75 |

With the DocVQA-like discrete baseline (mean 0.596): n50 / n80 = 991 / 1011 (T = 0.99, 0.05/7), 631 / 647 (0.05), 327 / 338 (T = 0.97, 0.05/7), 208 / 218 (0.05), 195 / 204 (T = 0.95, 0.05/7), 124 / 131 (0.05); formula 991.5, 630.9, 327.1, 208.3, 194.3, 123.8. The implemented bet needs 1.32-1.67 times the information-theoretic floor of `E1/DESIGN.md` (approximately (ln(1/alpha') + kappa^2/8) / (0.9 ln(1/(2 alpha')))).

Consistency with the documents.

- `guide/SELECTION_AND_CERTIFICATION.md:181-208` (floor 392/723, 129/238, 77/141 at E[b] = 0.584): numerically the DESIGN formula; but it is a floor for ANY valid test at 50% power. The implemented rule's lossless median at E[b] = 0.6 and alpha' = 0.05/7 is 985 / 326 / 193, i.e. 1.4 times the quoted floor. "A real candidate loses something, so the number needed is larger" is true and understates (SR-7).
- `LIMITATIONS.md:65-72` and `FINAL_RESEARCH_BOUNDARY.md` section 5: T = 0.99 fails at pool-sized budgets (at most 0.010 at 450 or 300 queries). This is forced by arithmetic: a candidate that is never better than the baseline needs at least 547 queries at F = 7, and a perfect candidate needs a median of about 990 at E[b] = 0.6. The documents call it a pool-sized-budget result, which is right; it is not a finding about the encoders.
- T = 0.97 non-trivial >= 0.90 at 450 (300 on SciFact) queries: a lossless candidate needs a median of 326 at E[b] = 0.6, so n = 450 suffices with probability close to 1, and n = 300 needs E[b] around 0.65 or more (196/300): consistent with "lowest 0.917 on SciFact, close to the threshold".
- "No valid method can certify T = 0.99 with <= 225 queries" (`DESIGN.md:135`, `RESULTS.md:171`, `METHOD_C_SPEC.md:334`): the derivation gives that no valid test rejects a lossless candidate with probability >= 1/2 with fewer than 229 queries at E[b] = 1, delta = 0.05 (E[b] = 1: 229 at delta and 423 at delta/7; E[b] = 0.584: 392 and 723). Read literally "cannot certify" is false: a valid test may reject with probability up to delta/(1-eps)^n, 0.48 at n = 225. For the implemented rule the literal sentence is true for any candidate with c <= b: it needs at least 547 queries at F = 7 and 332 at F = 1. Wording W12.
- `api.py:421` documents "at least one" query; for a candidate that is never better than the baseline nothing can be certified at T >= 0.9 with fewer than 32 queries (F = 1, delta = 0.05; 52 at F = 7), and at T = 0.99 with fewer than 332 (547) (SR-7).

Required changes: wording only (SR-7).

## 10. H. Monte Carlo check of the implementation

**Verdict: NO CASE EXCEEDS delta; NO BLOCKER.**

Design. The registered 40-candidate, 7-chain list (`CandidateSpace.default(128)`, chain sizes 1, 6, 7, 6, 7, 6, 7, identity equal to the registered hash by `is_registered_default`), synthetic evidence, T = 0.95 unless stated, delta = 0.05 unless stated, the package's `to_y`, `betting_pvalues`, `fixed_sequence_by_family`, `smallest` on stacked replicates; on the first 100 replicates of every case (50 at small n) the same data went through `FixedSequenceBetting().decide` or `BonferroniBetting().decide`: decisions equal and p-values bitwise equal in every case. Truth is known exactly. Event "ERR<=": a candidate is selected and R <= T (what the proof controls); "R<T" is the statement's strict version (zero at the boundary by definition); "FWER": some candidate with R <= T is rejected. Generators: G1 (nDCG@5-like baseline, mean 0.596, candidate keeps b with probability 1 - loss else 0, so R_k = 1 - loss_k exactly) with dependence independent / shared inside a chain / shared by all candidates; G2x (heavy point masses: b ~ Bernoulli(1/2), c_k = 1 w.p. R_k on b = 0 queries and 0 on b = 1, so Y in {0, m0, 1} with about 98% of the mass on 0 and 1 at the boundary; R_k = retention_k). Profiles: "boundary-all" (every R_k = 0.95); "mixed" (R by chain: [0.999], [0.998, 0.995, 0.988, 0.97, 0.95, 0.88], [0.998, 0.996, 0.992, 0.98, 0.951, 0.93, 0.8], [0.997, 0.99, 0.975, 0.94, 0.9, 0.7], [0.996, 0.99, 0.955, 0.95, 0.95, 0.92, 0.8], [0.99, 0.9, 0.98, 0.99, 0.7, 0.5] (non-monotone), [0.94, 0.93, 0.92, 0.9, 0.85, 0.8, 0.6] (first member already a true null)); "duplicates" (positions 1/7, 14/20, 27/33 identical, as in the default list). Sizes decrease along each chain and differ across chains.

Result. 71 cases, 302,000 replicates at n >= 50 and 310,000 at n <= 20 (seeds 20261101-20261169, 20261201-20261202; one seed per case). No case has a lower 95% bound for ERR<= above delta. Largest ERR<= / delta: 0.754, boundary-all, shared fragility inside chains, n = 800: 0.0377 [0.0321, 0.0441]. Summary (full rows in the appendix):

| Scenario (delta = 0.05, T = 0.95) | n | replicates | ERR<= (largest over dependence variants) |
|---|---|---|---|
| (i) boundary-all, G1 | 50 / 225 / 800 / 3000 | 10,000 / 10,000 / 4,000 / 3,000 | 0.0000 / 0.0105 / 0.0377 / 0.0337 [0.0275, 0.0408] |
| (i) boundary-all, T = 0.90 / 0.99 | 225 | 10,000 | 0.0273 [0.0242, 0.0307] / 0.0000 |
| (i) boundary-all, delta = 0.01 | 225 | 10,000 | 0.0000 |
| (ii) mixed, G1 | 225 / 800 | 10,000 / 4,000 | 0.0029 [0.0019, 0.0042] / 0.0005 (FWER 0.0112) |
| (iii) duplicates + one fragility | 225 / 800 | 10,000 / 4,000 | 0.0017 / 0.0035 (boundary-all), 0.0001 / 0.0008 (mixed) |
| (iv) heavy masses, G2x | 50 / 225 / 800 / 3000 | 10,000 / 10,000 / 4,000 / 3,000 | 0.0029 / 0.0050 / 0.0080 / 0.0043 |
| (v) small n, delta in {0.05, 0.5, 0.9, 0.99} | 1-20 | 10,000 per cell | 0 for G1; G2x 0.0182 (n = 10, delta 0.5), 0.0431 (n = 10, delta 0.99), 0.0581 (n = 20, delta 0.99) against 0.5 and 0.99 |
| (vi) Bonferroni | 225 / 800 / 5 | 10,000 / 4,000 / 10,000 | 0.0000 in all |

Reading. (a) The error stays well under delta because per-test rejection at the boundary is at most about 0.8 of its level (section 3) and chains are tested at delta/7; the long-run value for 7 independent boundary chains is about 0.034-0.038. (b) In "mixed" the selected candidate is rarely a true null even when one is falsely rejected (FWER 0.0112 against ERR<= 0.0005) because a smaller false null elsewhere is usually selected; the proof's bound is on the superset event. (c) Small n: nothing is selected for G1 at any delta up to 0.99 at n <= 20 (the nDCG-like law did not reject in any of these cells), G2x selects only at n >= 10 with large delta, and the raise fraction for G1 is 0.313 / 0.100 / 0.035 / 0.0024 at n = 1 / 2 / 3 / 5, treated as abstention. (d) Dependence lowers the error (all candidates sharing one fragility: 0.0047 at n = 800).

## 11. Wording changes required

Old -> new. ASCII only; sentences are proposals and keep the claims unchanged.

- W1 (`rules.py:40-48`, SR-3). Old: "P(a compressed candidate is selected and its retention under Q is below T) <= delta, where the probability is over the draw of the calibration queries and retention under Q is E_Q[metric of the candidate] / E_Q[metric of the uncompressed index]. It holds for every sample size, with no large-sample approximation, and only under assumptions A1-A6. It makes no statement about power: it also holds if the rule always abstains. It is an application of published tools; the derivation is this project's own and has not been externally reviewed." New: "For the target T and error level delta of this decision: P(a compressed candidate is selected AND its retention under Q is below T) <= delta. The probability is over the random draw of the calibration queries; the corpus, the candidate list and its order, T and delta are fixed, not random. Q is the distribution the calibration queries are drawn from, and retention under Q is E_Q[metric of the candidate] / E_Q[metric of the uncompressed index]. This is a joint probability. It is NOT the probability, given that a candidate was selected, that its retention is below T: a rule that rarely selects can satisfy it while most of its selections are below T. It is not a statement about the retention measured on the calibration queries, about a finite batch of future queries, or about power (it also holds if the rule always abstains). The project's derivation covers every sample size, without a large-sample approximation, under assumptions A1-A7; nothing is claimed if one of them fails. It is an application of published tools; the derivation is this project's own and has not been externally reviewed."
- W2 (`rules.py:51-52`, SR-1). Old: "A1: the calibration queries are independent draws from one distribution Q, the deployment query distribution." New: "A1: the calibration queries are independent, identically distributed draws from one distribution Q, the distribution of the queries the deployed index will serve (a random sample of them, not a hand-picked, sorted, clustered or duplicated set). Duplicated or near-duplicate queries break this assumption and can inflate the error several-fold."
- W3 (`rules.py:55-57`, SR-4, SR-11). Old: "A3: ... are fixed before the calibration queries are seen, and every fitted stage is fitted on documents only." New: append "The list was not designed or edited after looking at these queries or at any result computed on them, and no stage was fitted on them."
- W4 (`rules.py:60-61`, SR-2). Old: "A5: the order in which queries enter the sequential test is fixed before their outcomes are seen." New: "A5: the order in which queries enter the sequential test does not depend on any outcome, nor on any query property correlated with the outcomes (difficulty, date, source, baseline score). The arrival order of an i.i.d. stream, or a seeded random permutation chosen before the outcomes are seen, qualifies; trying several orders or seeds and reporting one does not."
- W5 (`rules.py:62-64`, SR-4). Append to A6: "The reference was fixed independently of the candidates' outputs. Queries without a relevant document are removed by a rule that looks only at the reference, so the statement is about the labelled queries."
- W6 (new A7, `rules.py:50-65` and `67-73`, SR-4). "A7: one target, one error level, one rule, one metric and one reference were chosen before any outcome was seen and used once on this sample; the index that is deployed is the index that was evaluated (same corpus, same fitted pipeline, same retrieval procedure)." Replace "A1-A6" by "A1-A7" in `STATEMENT` and docstrings.
- W7 (`guide/SELECTION_AND_CERTIFICATION.md:163`, SR-1). Old: "near-duplicates weaken independence". New: "remove exact and near-duplicate queries before collecting evidence: in a simulation, counting every query twice raised the false-selection probability at the null boundary from 0.011 to 0.21 against a stated 0.05. A random sample of a production log is the intended protocol; a curated, sorted or synthetic set defines another Q".
- W8 (`guide/SELECTION_AND_CERTIFICATION.md:167`, SR-2). Old: "keep the order of the queries as given, or pass `order_seed` ...; do not order by outcome or try several orders and keep the best". New: "pass `order_seed` unless the given order is the arrival order of an i.i.d. stream; never use an order that depends on difficulty, date, source or any outcome (in a simulation an order sorted by the baseline score raised the error from 0.013 to 0.28 where 0.05 was stated); choose the seed before looking and use it once (the best of 20 seeds had error 0.076)".
- W9 (`guide/SELECTION_AND_CERTIFICATION.md:166`, SR-8). Old: "yes: the rule raises on values outside [0, 1], non-finite values and a zero baseline mean". New: "partly: the rule checks the sample (values in [0, 1], finite, positive mean baseline); it cannot check the population condition E_Q[b] > 0".
- W10 (`FINAL_RESEARCH_BOUNDARY.md:157-166`, SR-10). Old: "More than the certifying rule's own statement: the probability that ... is at most the error level, over the draw of the calibration queries, **conditional on** independent queries ...". New: "Anything beyond the certifying rule's own statement, which is: the probability that a compressed candidate is selected and its retention under the query distribution is below the target is at most the error level, over the draw of the calibration queries. The derivation holds only if the assumptions hold: independent, identically distributed queries from the deployment distribution, a fixed corpus, a candidate list and order fixed before the queries were seen with stages fitted on documents only, a bounded metric with a positive baseline mean, a query order independent of the outcomes, the reference taken as given, one look per sample."
- W11 (`cli_certify.py:125, 103`, SR-10). "what the rule states (conditional on every assumption below):" -> "what the rule states (claimed only if every assumption below holds):"; "select (certified)" -> "select (certified at the stated target and error level)". Same replacement of "conditional on its assumptions" by "claimed only under its assumptions" in `api.py:24, 110, 201, 414`.
- W12 (`METHOD_C_SPEC.md:334`, `E1/DESIGN.md:135`, `E1/RESULTS.md:171`, SR-7). Old: "No valid method can certify T = 0.99 with <= 225 queries". New: "No valid test can reject a lossless candidate with probability >= 1/2 at T = 0.99, delta = 0.05 with fewer than 229 queries (E[b] = 1: 229 at delta and 423 at delta/7; E[b] = 0.584: 392 and 723). The implemented rule needs at least 547 queries at F = 7 for any candidate that is never better than the baseline, 592 for a perfect candidate when the baseline scores 1 on every query, and a median of about 990 when it scores 0.6 on average".
- W13 (`guide/SELECTION_AND_CERTIFICATION.md:207-208`, SR-7). After "The columns are the level 0.05 and the level 0.05 / 7 of the seven-family default list." add: "For the implemented bet a candidate that loses nothing needs a median of about 1.4 times these numbers (985, 326 and 193 queries at level 0.05 / 7 and mean baseline 0.6); a candidate that is never better than the baseline on any query cannot be certified with fewer than 547, 180 and 107 queries".
- W14 (`api.py:421`, SR-7). Old: "queries: sample queries, same forms; at least one." New: "queries: sample queries, same forms; one is enough to run, but certification needs far more (a perfect candidate at target 0.95, error 0.05 and 7 families needs about 116 queries if the uncompressed index scores 1 on every query and about 195 if it scores 0.6 on average; target 0.99 needs about 600 to 1,000)".
- W15 (`api.py:80-86`, SR-9). Old: "with more queries, or a lower target, a later run may certify one. Decide on that before looking again: every further look at the same queries, with another target, error level or list, is outside the statement of the rule." New: "with new, independent queries, or a lower target chosen beforehand, a later run may certify one. Every further look at the same queries, with another target, error level, list, order or seed, is outside the statement of the rule (in simulations such looks multiplied the false-selection probability by 3 to 14)".
- W16 (`record.py:76-78`, SR-5). `statement_for(certifying)` should return `STATEMENT` only for the built-in rules (by identity of the rule class), and `STATEMENT_NOT_CERTIFYING` or the rule's own statement otherwise.

## 12. What remains unverified

- No external review. This is an internal check by a second reader; it may share blind spots with the authors, it is not independent in the sense of a statistician outside the project, and it does not change the status "not externally reviewed".
- The proof steps (supermartingale property, Ville's inequality applied to the running maximum including the start, fixed-sequence argument, union bound) were verified by derivation and by simulation, not mechanised. Citations are given from memory and were not re-checked against the sources: Waudby-Smith and Ramdas, JRSSB 86(1), 2024 (the aGRAPA bet, prior 1/2 and 1/4, and the finite-population extension: structure compared, constants and section numbers not); Waudby-Smith and Ramdas, NeurIPS 2020 (without replacement); Ville, 1939; Hoeffding, JASA 58, 1963 (sampling without replacement); Westfall and Krishen, J. Statist. Plann. Inference 99, 2001, and Maurer, Hothorn and Lehmacher, 1995 (fixed sequence); Angelopoulos et al., arXiv:2110.01052; Ramdas, Grunwald, Vovk and Shafer, Statistical Science, 2023 (anytime-valid inference). I am not certain of page or theorem numbers.
- Behaviour outside i.i.d. was sampled (duplicates, drift, covariate order, small finite pools), not analysed; no theory was derived. Validity of the implemented test for a finite pool sampled without replacement is CANNOT DETERMINE.
- All simulations use synthetic generators with exactly known truth and no real per-query data (by instruction). Monte Carlo intervals describe simulation error only. The adversarial search is a random search over three-point laws, not an optimisation; near-tight cases at very large n (the ratio 0.94 at level 0.05/40) were seen for the two-point law {0, 2 m0} only.
- Floating point and platform: parity with the frozen functions is bitwise on one platform (Windows, NumPy 2.4.6); no cross-platform or GPU check; the package test suite was not run.
- Deployment: the equality of the evaluated and the deployed index, approximate backends (Qdrant), tie-breaking in other retrieval paths and corpus drift are outside what was checked.
- The package, the guide and the specifications were still being edited while this review was written; findings refer to the text read on 2026-10-06.

## 13. Seeds and scripts (scratch directory, not in the repository)

| Block | Script | Seeds | Replicates |
|---|---|---|---|
| A p-value grid | `sim_A.py` | 20261006 + 7*int(1000 T) + n | 100,000 per cell (40,000 at n = 1000), 14 laws x 10 cells |
| A adversarial, without replacement, exact | `sim_A2.py` | 20261007, 20261008 (+1), 20261011 | 20,000 screen / 200,000 refine; 20,000 per SWOR cell; enumeration exact |
| B range, identity, n = 1 | `sim_B.py` | 20261801 | 2,000,000 pairs x 47 targets; 20,000 populations |
| D joint vs conditional | `sim_D.py` | 20262101 | 20,000 per n; 3,000 |
| E engine, seeds, budgets, fallback, lists | `sim_E.py`, `sim_E2.py`, `sim_E3.py` | 20261201, 20261203-20261205, 20261601, 20261701 | 40-60 selects; 2,000-3,000 |
| F departures from i.i.d. | `sim_F.py`, `sim_F2.py` | 20261401-20261403, 20261501-20261502 | 20,000; 4,000 |
| G power | `sim_G.py`, `sim_floor.py` | 20261301, 20261302, 20262201 | 1,000 per cell; 840,000 |
| H implementation | `sim_H.py`, `sim_H2.py`, `sim_H3.py` | 20261101-20261169, 20261201-20261202 | see appendix |
| Metric range, parity | `sim_metric.py`, `sim_parity.py` | 20262001, 20261901 | 90,000 values; 400 problems |

## Appendix H. Monte Carlo of the implementation, all cases

```text
Columns: case | T | delta | n | replicates | seed | P(select) | P(select a candidate with R<=T) [95% Clopper-Pearson] | same with R<T | P(some R<=T candidate rejected) | flag

G1 boundary-all, dep=indep | 0.95 | 0.05 | 50 | 10000 | 20261101 | 0.0000 | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | ok
G1 boundary-all, dep=indep | 0.95 | 0.05 | 225 | 10000 | 20261102 | 0.0105 | 0.0105 [0.0086, 0.0127] | 0.0000 | 0.0105 | ok
G1 boundary-all, dep=indep | 0.95 | 0.05 | 800 | 4000 | 20261103 | 0.0333 | 0.0333 [0.0279, 0.0393] | 0.0000 | 0.0333 | ok
G1 boundary-all, dep=nested_chain | 0.95 | 0.05 | 50 | 10000 | 20261104 | 0.0000 | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | ok
G1 boundary-all, dep=nested_chain | 0.95 | 0.05 | 225 | 10000 | 20261105 | 0.0098 | 0.0098 [0.0080, 0.0119] | 0.0000 | 0.0098 | ok
G1 boundary-all, dep=nested_chain | 0.95 | 0.05 | 800 | 4000 | 20261106 | 0.0377 | 0.0377 [0.0321, 0.0441] | 0.0000 | 0.0377 | ok
G1 boundary-all, dep=nested_all | 0.95 | 0.05 | 50 | 10000 | 20261107 | 0.0000 | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | ok
G1 boundary-all, dep=nested_all | 0.95 | 0.05 | 225 | 10000 | 20261108 | 0.0012 | 0.0012 [0.0006, 0.0021] | 0.0000 | 0.0012 | ok
G1 boundary-all, dep=nested_all | 0.95 | 0.05 | 800 | 4000 | 20261109 | 0.0047 | 0.0047 [0.0029, 0.0074] | 0.0000 | 0.0047 | ok
G1 boundary-all, dep=indep | 0.99 | 0.05 | 225 | 10000 | 20261110 | 0.0000 | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | ok
G1 boundary-all, dep=indep | 0.9 | 0.05 | 225 | 10000 | 20261111 | 0.0273 | 0.0273 [0.0242, 0.0307] | 0.0000 | 0.0273 | ok
G1 boundary-all, dep=indep, delta=0.01 | 0.95 | 0.01 | 225 | 10000 | 20261112 | 0.0000 | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | ok
G1 mixed, dep=indep | 0.95 | 0.05 | 225 | 10000 | 20261113 | 0.9962 | 0.0002 [0.0000, 0.0007] | 0.0002 | 0.0005 | ok
G1 mixed, dep=indep | 0.95 | 0.05 | 800 | 4000 | 20261114 | 1.0000 | 0.0000 [0.0000, 0.0009] | 0.0000 | 0.0013 | ok
G1 mixed, dep=nested_chain | 0.95 | 0.05 | 225 | 10000 | 20261115 | 0.9969 | 0.0029 [0.0019, 0.0042] | 0.0010 | 0.0047 | ok
G1 mixed, dep=nested_chain | 0.95 | 0.05 | 800 | 4000 | 20261116 | 1.0000 | 0.0005 [0.0001, 0.0018] | 0.0000 | 0.0112 | ok
G1 mixed, dep=nested_all | 0.95 | 0.05 | 225 | 10000 | 20261117 | 0.8885 | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0014 | ok
G1 mixed, dep=nested_all | 0.95 | 0.05 | 800 | 4000 | 20261118 | 1.0000 | 0.0003 [0.0000, 0.0014] | 0.0003 | 0.0053 | ok
G1 mixed+duplicates, dep=nested_all | 0.95 | 0.05 | 225 | 10000 | 20261119 | 0.8863 | 0.0001 [0.0000, 0.0006] | 0.0001 | 0.0023 | ok
G1 boundary-all+duplicates, dep=nested_all | 0.95 | 0.05 | 225 | 10000 | 20261120 | 0.0017 | 0.0017 [0.0010, 0.0027] | 0.0000 | 0.0017 | ok
G1 mixed+duplicates, dep=nested_all | 0.95 | 0.05 | 800 | 4000 | 20261121 | 1.0000 | 0.0008 [0.0002, 0.0022] | 0.0008 | 0.0070 | ok
G1 boundary-all+duplicates, dep=nested_all | 0.95 | 0.05 | 800 | 4000 | 20261122 | 0.0035 | 0.0035 [0.0019, 0.0059] | 0.0000 | 0.0035 | ok
G2x boundary-all, dep=indep | 0.95 | 0.05 | 50 | 10000 | 20261123 | 0.0029 | 0.0029 [0.0019, 0.0042] | 0.0000 | 0.0029 | ok
G2x mixed, dep=indep | 0.95 | 0.05 | 50 | 10000 | 20261124 | 0.0038 | 0.0017 [0.0010, 0.0027] | 0.0017 | 0.0033 | ok
G2x boundary-all, dep=indep | 0.95 | 0.05 | 225 | 10000 | 20261125 | 0.0050 | 0.0050 [0.0037, 0.0066] | 0.0000 | 0.0050 | ok
G2x mixed, dep=indep | 0.95 | 0.05 | 225 | 10000 | 20261126 | 0.0059 | 0.0017 [0.0010, 0.0027] | 0.0017 | 0.0043 | ok
G2x boundary-all, dep=indep | 0.95 | 0.05 | 800 | 4000 | 20261127 | 0.0080 | 0.0080 [0.0055, 0.0113] | 0.0000 | 0.0080 | ok
G2x mixed, dep=indep | 0.95 | 0.05 | 800 | 4000 | 20261128 | 0.0063 | 0.0020 [0.0009, 0.0039] | 0.0018 | 0.0030 | ok
G2x boundary-all, dep=nested_all | 0.95 | 0.05 | 50 | 10000 | 20261129 | 0.0016 | 0.0016 [0.0009, 0.0026] | 0.0000 | 0.0016 | ok
G2x mixed, dep=nested_all | 0.95 | 0.05 | 50 | 10000 | 20261130 | 0.0028 | 0.0013 [0.0007, 0.0022] | 0.0013 | 0.0024 | ok
G2x boundary-all, dep=nested_all | 0.95 | 0.05 | 225 | 10000 | 20261131 | 0.0041 | 0.0041 [0.0029, 0.0056] | 0.0000 | 0.0041 | ok
G2x mixed, dep=nested_all | 0.95 | 0.05 | 225 | 10000 | 20261132 | 0.0055 | 0.0020 [0.0012, 0.0031] | 0.0020 | 0.0033 | ok
G2x boundary-all, dep=nested_all | 0.95 | 0.05 | 800 | 4000 | 20261133 | 0.0040 | 0.0040 [0.0023, 0.0065] | 0.0000 | 0.0040 | ok
G2x mixed, dep=nested_all | 0.95 | 0.05 | 800 | 4000 | 20261134 | 0.0090 | 0.0030 [0.0016, 0.0052] | 0.0030 | 0.0050 | ok
BONF G1 boundary-all, dep=indep | 0.95 | 0.05 | 225 | 10000 | 20261165 | 0.0000 | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | ok
BONF G1 mixed, dep=indep | 0.95 | 0.05 | 800 | 4000 | 20261166 | 1.0000 | 0.0000 [0.0000, 0.0009] | 0.0000 | 0.0030 | ok
BONF G1 boundary-all, dep=nested_all | 0.95 | 0.05 | 225 | 10000 | 20261167 | 0.0000 | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | ok
BONF G1 mixed, dep=nested_all | 0.95 | 0.05 | 800 | 4000 | 20261168 | 1.0000 | 0.0000 [0.0000, 0.0009] | 0.0000 | 0.0020 | ok
G1 boundary-all, dep=indep (long run) | 0.95 | 0.05 | 3000 | 3000 | 20261201 | 0.0337 | 0.0337 [0.0275, 0.0408] | 0.0000 | 0.0337 | ok
G2x boundary-all, dep=indep (long run) | 0.95 | 0.05 | 3000 | 3000 | 20261202 | 0.0043 | 0.0043 [0.0023, 0.0074] | 0.0000 | 0.0043 | ok

Small n (all candidates at R = T; a replicate whose baseline is 0 on every query makes the package raise and counts as an abstention):
G1 boundary-all, dep=indep | n=1 | R=10000 | seeds 20261135..20261138 | raised fraction 0.3129 | delta=0.05: P(select)=0.0000, upper95=0.0004; delta=0.5: P(select)=0.0000, upper95=0.0004; delta=0.9: P(select)=0.0000, upper95=0.0004; delta=0.99: P(select)=0.0000, upper95=0.0004
G1 boundary-all, dep=indep | n=2 | R=10000 | seeds 20261139..20261142 | raised fraction 0.0995 | delta=0.05: P(select)=0.0000, upper95=0.0004; delta=0.5: P(select)=0.0000, upper95=0.0004; delta=0.9: P(select)=0.0000, upper95=0.0004; delta=0.99: P(select)=0.0000, upper95=0.0004
G1 boundary-all, dep=indep | n=3 | R=10000 | seeds 20261143..20261146 | raised fraction 0.0353 | delta=0.05: P(select)=0.0000, upper95=0.0004; delta=0.5: P(select)=0.0000, upper95=0.0004; delta=0.9: P(select)=0.0000, upper95=0.0004; delta=0.99: P(select)=0.0000, upper95=0.0004
G1 boundary-all, dep=indep | n=5 | R=10000 | seeds 20261147..20261150 | raised fraction 0.0024 | delta=0.05: P(select)=0.0000, upper95=0.0004; delta=0.5: P(select)=0.0000, upper95=0.0004; delta=0.9: P(select)=0.0000, upper95=0.0004; delta=0.99: P(select)=0.0000, upper95=0.0004
G1 boundary-all, dep=indep | n=10 | R=10000 | seeds 20261151..20261154 | raised fraction 0.0000 | delta=0.05: P(select)=0.0000, upper95=0.0004; delta=0.5: P(select)=0.0000, upper95=0.0004; delta=0.9: P(select)=0.0000, upper95=0.0004; delta=0.99: P(select)=0.0000, upper95=0.0004
G1 boundary-all, dep=indep | n=20 | R=10000 | seeds 20261155..20261158 | raised fraction 0.0000 | delta=0.05: P(select)=0.0000, upper95=0.0004; delta=0.5: P(select)=0.0000, upper95=0.0004; delta=0.9: P(select)=0.0000, upper95=0.0004; delta=0.99: P(select)=0.0000, upper95=0.0004
G2x boundary-all, dep=indep | n=3 | R=10000 | seeds 20261159..20261160 | raised fraction 0.1256 | delta=0.5: P(select)=0.0000, upper95=0.0004; delta=0.99: P(select)=0.0000, upper95=0.0004
G2x boundary-all, dep=indep | n=10 | R=10000 | seeds 20261161..20261162 | raised fraction 0.0009 | delta=0.5: P(select)=0.0182, upper95=0.0210; delta=0.99: P(select)=0.0431, upper95=0.0473
G2x boundary-all, dep=indep | n=20 | R=10000 | seeds 20261163..20261164 | raised fraction 0.0000 | delta=0.5: P(select)=0.0283, upper95=0.0317; delta=0.99: P(select)=0.0581, upper95=0.0629
BONF G1 boundary-all, dep=indep, n=5, delta=0.99 | n=5 | R=10000 | seeds 20261169..20261169 | raised fraction 0.0034 | delta=0.99: P(select)=0.0000, upper95=0.0004

Totals: 71 cases; 302000 replicates at n >= 50 and 310000 at n <= 20; cases with a lower 95% bound for P(select a candidate with R<=T) above delta: 0
Largest observed ratio P(select a candidate with R<=T)/delta: 0.754 (G1 boundary-all, dep=nested_chain, n=800, T=0.95, delta=0.05)
```
