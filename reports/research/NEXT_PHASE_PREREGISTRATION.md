# Next-phase preregistration

## 1. Title

**Model-blind selection and certification of compression for late-interaction retrieval:
label-free reference agreement (X1), query-budget curve (X2), coordinate symmetry (X3),
and the protocol for a future held-out-encoder experiment (XH).**

## 2. Status

**DRAFT. This document is not yet a preregistration.**

- Written 2026-10-04 on branch `research-investigation` at `7d87e33`, working tree
  clean, `main` at `4d52060` (v0.3.0) untouched. It has not been committed.
- It becomes **PREREGISTERED** only when all three hold:
  1. it is committed unchanged to `research-investigation`;
  2. the commit hash is recorded in `reports/research/README.md`;
  3. no X1, X2 or X3 computation has been run before that commit.
- After that commit the text above Appendix C is frozen. A change is made only as a
  dated amendment appended to Appendix C and committed before the analysis it affects.
  An amendment made after the affected results exist is not a preregistration and must
  be labelled a deviation in the report.

**What was done while drafting.** Existing reports, code, git history and file metadata
were read. For the per-query stores, only array names, shapes, data types, metadata
fields and file hashes were read. No per-query value was read or summarised, no
experiment was run, and no result was computed from per-query data.

**Pre-commit audit.** The draft was audited once, on 2026-10-04, before any commit, and
corrected in place: query budgets were split into pool-supported and extrapolated; H2,
H3 and H5 were restated; the held-out registration was tightened; immutable identifiers
were added. No result existed before or after the audit.

**Evidence labels used throughout.**

| label | meaning in this phase |
|---|---|
| MEASURED | computed directly from real encoder outputs: per-query values, and pool-level quantities that are exact functions of them |
| SIMULATED | any rate, quantile or curve over resamples of a measured pool. Never reported as MEASURED. Every simulated result carries one of two marks, defined in §11.1: **pool-supported** (the budget does not exceed the pool size) or **extrapolated** (it does) |
| PROJECTED | arithmetic on measured values or on a proved formula, including anything read off beyond the registered grid |
| INFERENCE | proofs and interpretation |
| EXTERNAL | model cards, published scores, literature. Not reproduced here |
| CODE | read from the source or git history at the commit above |

---

## 3. Motivation

What Q1–Q10 established, and what they did not.

1. **One configuration does not suit every encoder.** MEASURED (Q9, R8, R12): dimension
   reduction, plain int4 and centred binary are encoder-specific, and the amount of
   certifiable compression is dataset-specific.
2. **A valid selection rule exists, and it is conservative.** Method C (frozen in
   `E1/E1_2_FREEZE.md`) kept its error level in every check (SIMULATED on MEASURED
   per-query results, E1). With 225 labelled queries it deployed at a 0.95 target on all
   seven stores, at 0.97 reliably only on two InfoVQA stores, and almost never at 0.99.
3. **No distribution-free rule can do better at 0.99 with that many queries.**
   INFERENCE (proof, `E1/DESIGN.md`): at Method C's per-family level a lossless candidate
   needs 460–723 labelled queries on the measured pools.
4. **No encoder has been held out from the search space under a preregistered protocol.**
   CODE (git history): the default search space was fixed on 2026-09-27 (`cc2800f`,
   `cc4f3f6`). Its own comments justify it by results on ColPali-v1.3 and the text
   ColBERT; ColSmol-256M had also been measured by then. ColQwen2, ColQwen2.5 and
   ColEmbed 4B were measured later with the space unchanged, but with no prior
   commitment to leave it unchanged. E1.2 held back ColQwen2.5 and SciFact for the
   *decision rule's* validity, with compression "descriptive, not part of the
   criterion".

The next phase asks whether the decision procedure, not any configuration, carries over
to encoders that did not shape it. This document fixes the questions, the procedure and
the criteria before any further analysis.

The architecture proposal of 2026-10-04 (outside the repository) suggested experiments
X1–X3. It was checked against the repository. Appendix A lists what is adopted, changed
and rejected.

---

## 4. Research question

**Central question (long term).** For late-interaction retrievers, how much compression
can a single frozen procedure certify at a stated retention target and error level, on
encoders that played no part in its design?

**Decomposition.**

| part | status assumed here | tested by |
|---|---|---|
| A. Compression capability | encoder-dependent. Not assumed to transfer | not a hypothesis of this phase |
| B. Selection | possibly model-blind | XH |
| C. Certification | independent of the encoder under the assumptions in §16.2 | X1, X2 (consistency checks), XH |

**Sub-questions of this phase.**

| id | question | experiment | data |
|---|---|---|---|
| SQ1 | Can selection and certification be run against agreement with the uncompressed ranking when labels are unavailable, and how does that quantity relate to labelled retention? | X1 | existing stores |
| SQ2 | How does certification strength change with the number of queries? Is the abstention seen at strict targets a query-budget limitation? | X2 | existing stores |
| SQ3 | Do the conclusions depend on the encoder's arbitrary coordinate basis or vector order? | X3 | existing vectors |
| SQ4 | Does the frozen procedure certify useful compression on held-out encoders, and what does model-blindness cost against a tuned procedure and an oracle? | XH (designed here, not run) | future encodings |

**X1–X3 use development encoders only. They cannot answer the central question.** They
can remove or confirm obstacles to it. Only XH bears on it.

The result may be negative. §19 fixes in advance what counts as a negative finding.

---

## 5. Scope

**In scope.**

- Late-interaction retrieval scored by MaxSim: for a query with vectors q_t and a
  document with vectors d_v, the score is the sum over t of the maximum over v of the
  dot product q_t · d_v.
- Encoders whose token or patch vectors are L2-normalised, as all measured encoders are.
- Training-free compression of a fixed encoder's document vectors. Queries stay in full
  precision.
- Exact exhaustive search over a fixed corpus.
- The operators that exist in v0.3.0. No new operator is introduced in this phase.

**Out of scope.**

- Any scoring function other than MaxSim.
- Approximate search, engines, latency.
- Corpus growth after certification.
- Trained compressors.
- Corpus sizes beyond those measured (500 pages; 5,183 abstracts).
- The ColQwen2 discrepancy with SAP. It stays unresolved (E7c) and is not used as
  evidence for or against any hypothesis here.

---

## 6. Definitions

**Data.**

| term | definition |
|---|---|
| Store | One (encoder, corpus, query pool) with per-query metric values for the uncompressed index and for each candidate |
| Query pool | The fixed set of N queries of a store. Three pools exist: DocVQA (451), InfoVQA (494), SciFact (300) |
| Candidate | One compression pipeline of the frozen search space S0 (§14), indexed 0–39 |
| Chain | A family of S0: candidates in a fixed order from least to most aggressive. S0 has 7 chains |
| Distinct candidates | 37. Three codec-only pipelines each open two chains and are counted once in candidate-level tallies |

**Two constructs. They are never merged.**

| | labelled retention (L) | reference agreement (A) |
|---|---|---|
| Relevant set for a query | the dataset's labelled documents | the single top document of the uncompressed index (ties by document index) |
| Per-query value of the uncompressed index | b_L = nDCG@k against the labels | b_A = 1 by construction |
| Per-query value of a candidate | c_L = nDCG@k against the labels | a = 1/log2(1 + r) if r ≤ k, else 0, where r is the rank that candidate gives to the reference's top document |
| Cut-off k | 5 on the ViDoRe stores, 10 on SciFact, as stored | the same |
| Pool-level quantity | R_L = Σ c_L / Σ b_L | R_A = mean of a |
| What it measures | whether the compressed index is still as *right* | whether the compressed index still *agrees* with the uncompressed one, including agreement with its mistakes |
| Threshold symbol | T | T_A |

R_L can exceed 1. R_A cannot. A statement about one is never reported as a statement
about the other.

**Derived agreement measures (available from the stores).** Top-1 agreement is the
share of queries with a = 1. Containment@5 is the share with the reference's top
document inside the candidate's top 5 (on SciFact also @10).

**Procedure and outcomes.**

| term | definition |
|---|---|
| Budget n | number of calibration queries given to the procedure |
| Pool-supported budget | a registered budget that does not exceed the pool size N. The pool holds at least that many distinct, independent queries |
| Extrapolated budget | a registered budget larger than N. A sample of that size can only be made by drawing the same N queries repeatedly. It is a simulation device, not a number of independent queries (§11.1) |
| Pool-capped budget n_cap | the largest pool-supported registered budget: 450 where N ≥ 450, otherwise N. It is 450 on the DocVQA and InfoVQA pools and 300 on SciFact |
| Error level δ | 0.05 primary, 0.01 secondary |
| Compression of a candidate | the float32 size of the encoder's original document vectors (number of vectors × dimension × 4 bytes) divided by the candidate's stored bytes. Stored bytes are the codes plus every array needed to decode the documents or to transform queries: codec means and scales, and projection matrices. Offsets and identifiers are left out for every candidate alike. For every S0 candidate this equals `compression_vs_float32` in the store metadata |
| Deploy / abstain | the procedure returns a compressed candidate / returns the uncompressed index. Compression is 1.0 when it abstains |
| Non-trivial deployment | a deployed candidate other than `float16`. Every other candidate of S0 is at least as small as per-vector int8 (3.9x) |
| Substantial deployment | a deployed candidate with compression of at least 10x |
| Pool truth | R_L or R_A of a candidate over all N pool queries. It is the retention under Q_pool, the empirical distribution of the pool, and nothing else |
| Miss | a candidate is deployed and its pool truth is below the target |
| Free oracle | the smallest candidate whose pool truth is at least the target. Chosen with knowledge of every pool query. An upper reference, never a method |
| Chain oracle | in each chain, take the longest initial run of candidates whose pool truth exceeds the target; the smallest candidate among those runs. It is what fixed-sequence testing converges to as n grows |
| n* | the smallest registered budget at which the non-trivial deployment rate is at least 0.90, at that budget and at every larger registered budget; "none" if no registered budget qualifies. It is always reported with the class of the budget that decides it |

**Frozen procedure P0.** Search space S0 with Method C exactly as frozen in
`E1/E1_2_FREEZE.md`: exact betting p-values, fixed-sequence testing inside each chain at
δ/7, deploy the smallest rejected candidate (§14, §16.2).

---

## 7. Hypotheses

Five hypotheses. Each has a fixed metric, unit and decision rule. Thresholds are
conventions chosen before the analyses; §9.4 lists what was already known when they were
chosen.

### H1. Usefulness on held-out encoders (tested by XH only)

| | |
|---|---|
| Statement | The frozen procedure P0 deploys non-trivial compression while keeping its error level on encoders that took no part in its design |
| Unit | a held-out cell: one held-out encoder on one query pool, at the primary setting (labels, T = 0.95, δ = 0.05, n = n_cap, the pool-capped budget of §6) |
| Valid cell | its encoder is in the analysed set of §13.1, its baseline gate passed or is marked unverifiable, and its run is valid (§20) |
| Metric | non-trivial deployment rate over 2,000 resamples; miss rate with its 95% interval. SIMULATED, pool-supported |
| Cell succeeds if | non-trivial deployment rate ≥ 0.90 and the miss rate is consistent with δ (§16.3) |
| Supported if | succeeding cells ÷ valid cells ≥ 0.80 |
| Not supported if | that ratio is below 0.80 |
| Refuted for the tested set if | that ratio is below 0.50 |
| Not assessable if | fewer than 6 valid cells, or fewer than 4 held-out encoders, or fewer than 2 encoders outside the development families |

### H2. Price of model-blindness (tested by XH only)

| | |
|---|---|
| Statement | When both certify on the same queries, an encoder-specific tuned procedure certifies little more compression than the model-blind procedure P0 |
| Quantity | **PoMB**, the price of model-blindness. One definition, given in full in §13.6. For each resample, ρ = (compression deployed by the tuned procedure) ÷ (compression deployed by P0), both certifying on the same queries. PoMB of a cell is the median over the 2,000 resamples of log2 ρ |
| Direction | tuned in the numerator, model-blind in the denominator. Always |
| Reading, fixed here | PoMB = x means the tuned procedure typically certifies 2^x times the compression of P0. x > 0: model-blindness has a price. x = 0: none. x < 0: P0 certified more. Larger is worse for P0. A negative value is reported as negative and is not renamed or re-signed |
| Unit | a held-out encoder: the median of PoMB over its valid cells. Then the set of held-out encoders: the median and the maximum of those values |
| H2a, measurable | PoMB is defined in at least 80% of valid held-out cells. It is undefined where the tuned procedure could not be computed, or where both procedures abstain together in more than half of the resamples |
| H2a fails if | PoMB is undefined in more than 20% of valid cells. Conclusion: the price cannot be estimated at this budget |
| H2b, small | over held-out encoders, the median PoMB is at most 1 (a factor of 2) and the maximum is at most 2 (a factor of 4) |
| H2b fails if | the median exceeds 1, or the maximum exceeds 2. Conclusion: model-blindness was costly, with the values and the encoders named |
| H2b is not assessable if | H2a fails. It is then reported as not assessable, not as supported |
| Not part of the verdict | the ratio at equal total query budget (§13.6) is reported beside PoMB. It cannot change the verdict of H2a or H2b |

### H3. Query budget (tested by X2)

| | |
|---|---|
| Statement | With every query its pool can supply, P0 deploys non-trivial compression at T = 0.97 on all seven stores |
| Unit | a store, at its pool-capped budget n_cap: 450 on the six ViDoRe stores, 300 on SciFact |
| Metric | the non-trivial deployment rate of P0 under labels at T = 0.97, δ = 0.05, n = n_cap. SIMULATED, pool-supported |
| Supported if | the rate is at least 0.90 on all seven stores |
| Fails if | the rate is below 0.90 on any store. Conclusion: with every query the pool can supply, certified selection at 0.97 remained unavailable on that store. Nothing is concluded about larger budgets |
| Extension beyond the pools (secondary; simulation only) | n* over the whole registered grid, per store. It **holds in simulation** if n* exists on every store. It **fails in simulation** if on some store no registered budget up to 3,600 qualifies |
| How the extension is reported | Only as a statement about samples drawn repeatedly from the pool's own N queries (§11.1). Never as evidence of what that many independent queries would give |
| No presumption | No outcome asserts that a larger budget would succeed. If the extension fails, the report says that within the registered range more queries did not produce non-trivial deployment, and stops there |
| Secondary | the same at T = 0.99, and with "substantial" in place of "non-trivial" |

### H4. Reference agreement (tested by X1 and the agreement arm of X2)

| | |
|---|---|
| Statement | Agreement with the uncompressed ranking gives useful evidence about compression behaviour, and it is not equivalent to labelled retention |
| Unit | a (store, threshold) pair for the pool-level conditions; a cell for the resampling conditions |
| H4a, conservative proxy | CP1: on every store and for every threshold in {0.99, 0.97, 0.95}, no distinct candidate has R_A at or above the threshold while its R_L is more than 0.01 below it. CP2: in every cell with δ = 0.05 and a threshold in {0.99, 0.97, 0.95}, at every registered budget, when P0 certifies agreement at the threshold, the rate at which the deployed candidate's R_L is below the same threshold is consistent with 0.05 (§16.3) |
| H4a fails if | CP1 is violated for any (store, threshold), or CP2 is violated in any cell after the replication rule of §16.3. Conclusion: agreement certification does not protect labelled retention. For each violating cell the report states whether its budget is pool-supported or extrapolated |
| H4b, not equivalent | for at least one (store, threshold) the free oracle under agreement and under labels differ by a factor of 2 or more, or CP1 is violated |
| H4b fails if | on every store and threshold the two oracles are within a factor of 2 and CP1 holds. Conclusion: the two constructs led to the same decisions on these stores. That would not establish equivalence elsewhere |
| H4c, usable with the pool | under agreement at T_A = 0.95 and δ = 0.05, the non-trivial deployment rate at the pool-capped budget is at least 0.90 on all seven stores. It fails if the rate is below 0.90 on any store |

### H5. Coordinate symmetry (tested by X3)

| | |
|---|---|
| Statement | Each of the four registered transformations of §12.3, applied to each primary dataset, leaves the labelled retention of every S0 candidate and the outcome of P0 within the registered tolerances of §12.6 |
| What is tested, and what is not | The transformations are four fixed maps: one signed coordinate permutation, two rotations and one reordering of vectors, each with a registered seed. Each leaves every MaxSim score unchanged and keeps the vectors unit-norm. H5 is about these four maps on these datasets. It is not a claim that results are independent of the coordinate basis in general |
| Unit | a (primary dataset, registered transformation) pair: 4 × 4 = 16 pairs |
| Metric | per candidate, Δ = R_L after the transformation minus R_L before, with its class (§12.6); for the procedure, the ratio of median certified compression and the difference in non-trivial deployment rate |
| A pair holds if | no candidate is material and the procedure outcome is not material |
| Supported if | all 16 pairs hold |
| Fails if | any pair does not hold. The failing pairs, candidates and sizes are reported |
| Not assessable if | a primary dataset cannot pass the gates of §12.5 |

**Which experiment can test what.**

| | X1 | X2 | X3 | XH |
|---|---|---|---|---|
| H1 | no | no | no | yes |
| H2 | no | no | no | yes |
| H3 | — | yes | — | repeats on new pools |
| H4 | yes | agreement arm | — | repeats |
| H5 | — | — | yes | — |

---

## 8. Experimental units

| unit | what it is | count in X1–X3 | independence |
|---|---|---|---|
| Encoder family | ColPali (PaliGemma), ColQwen (Qwen2-VL and Qwen2.5-VL), text ColBERT | 3 | the only encoder-level replication |
| Query pool | DocVQA, InfoVQA, SciFact | 3 | independent of each other |
| Store | encoder × pool | 7 | **not independent**: three stores share each ViDoRe pool, and two share an encoder family |
| Cell | store × construct × threshold × δ × budget | fixed by the grids in §10–§11 | nested in stores |
| Resample | one seed: n queries drawn i.i.d. with replacement from the pool | 2,000 per cell | resamples of one pool, not independent datasets |
| Candidate | one pipeline | 40 (37 distinct) | strongly correlated |

**Unit of inference.**

- Statements about the procedure's error level and operating characteristics are per
  cell, over resamples, for that pool's empirical distribution.
- Statements across pools use the three pools as units, taking the worst store in each.
- Statements across encoders are descriptive counts. No inferential statistic is
  computed across encoders: there are too few, and they are not a sample from any
  defined population.

---

## 9. Existing data

### 9.1 Per-query stores (in git)

`reports/research/E1/store/`. Each `.npz` holds `base_labels` (N,), `cand_labels`
(40, N), `base_b1` (N,), `cand_b1` (40, N), float64. The sidecar `.json` holds the
candidate list with family, pipeline, bytes per document, vectors per document and
compression. All seven stores list the same 40 candidates in the same order (CODE,
checked on metadata).

| id | store | encoder | pool (N) | documents | dim | metric |
|---|---|---|---|---|---|---|
| D1 | `colpali_docvqa` | ColPali-v1.3 | DocVQA (451) | 500 | 128 | nDCG@5 |
| D2 | `colpali_infovqa` | ColPali-v1.3 | InfoVQA (494) | 500 | 128 | nDCG@5 |
| D3 | `colqwen2_docvqa` | ColQwen2-v1.0 | DocVQA (451) | 500 | 128 | nDCG@5 |
| D4 | `colqwen2_infovqa` | ColQwen2-v1.0 | InfoVQA (494) | 500 | 128 | nDCG@5 |
| D5 | `colqwen25_docvqa` | ColQwen2.5-v0.2 | DocVQA (451) | 500 | 128 | nDCG@5 |
| D6 | `colqwen25_infovqa` | ColQwen2.5-v0.2 | InfoVQA (494) | 500 | 128 | nDCG@5 |
| D7 | `scifact` | answerai-colbert-small-v1 | SciFact (300) | 5,183 | 96 | nDCG@10 |

The file hashes are listed in §22 and must match at run time.

- Every ViDoRe query has exactly one labelled page. SciFact has 277 queries with one
  relevant document and 23 with two to five (store metadata).
- D1–D4 were the development stores of E1.1. D5–D7 were the held-back confirmation
  stores of E1.2. For this phase all seven are development data.
- A provenance wart: the description strings inside `scifact.json` say "nDCG@5" while
  its `metric` field and the E1.2 freeze say nDCG@10. The `metric` field is
  authoritative.

### 9.2 Other existing artifacts

| artifact | location | use here |
|---|---|---|
| Per-seed selections of E1.1 and E1.2 | `reports/research/E1/real/`, `E1/confirm/` (`*_selections.npz`) | reproduction gate for X1 and X2 |
| Method C code | `scripts/research/e1_stats.py` (last changed `0ad40ba`), `e1_1_study.py` (`106ba51`) | imported unchanged |
| Per-query features (baseline rank, score margins) | `reports/research/Q3/features/`, six ViDoRe stores | one exploratory analysis in X1 |
| Per-query values for 76 configurations, labels only | `reports/research/Q4/store/`, six ViDoRe stores | rehearsal of the tuned comparator (§13.8) |
| Document and query vectors | outside git, under the data directory: ColPali (legacy caches), ColQwen2, ColQwen2.5, SciFact | X3 |

### 9.3 What does not exist

- **ColEmbed 4B (2,560-d): no per-query store and no vectors.** "The raw vectors were
  not kept" (R12). It cannot take part in X1, X2 or X3. Every result of this phase
  before XH is for 96-d and 128-d encoders.
- **No ranking lists.** The stores hold per-query metric values only. Agreement
  measures that need the compressed and reference top-k lists (overlap@k, rank
  correlation) cannot be computed from them.
- **No query pool larger than 494.**

### 9.4 What is already known about these data

This is not a blind preregistration. The following outcomes were known when the
thresholds were set.

- Under labels: pool retention of every candidate, Method C's and rule A's behaviour at
  n = 50, 100 and 225, and the free oracle (E1); the axis ablation (Q4); the margin
  analysis (Q3).
- Under the agreement construct: R7 reports, for an earlier search space on ColPali and
  ColSmol, that label-free selection "is conservative in the right direction" and
  chooses less compression. Outcomes of the bootstrap and point rules under the
  baseline@1 reference on these seven stores exist in
  `reports/universal/selection_rules/` and were replayed in E1.0.
- **Not known:** the agreement arrays have never been analysed under Method C in any
  committed report, and no budget above 225 has been simulated.

### 9.5 Boundary of the existing evidence

Each row is what the existing reports support. The last column is what it must not be
turned into, in this phase or in any report of it.

| existing statement | label and source | must not become |
|---|---|---|
| Storage at a million pages, and scan time at that size | PROJECTED (Q8) | measured scalability |
| Retention and compression on five encoders, four of them with per-query stores | MEASURED (Q4, R11, R12) | evidence about encoders that were not tested |
| Method C keeps its error level | INFERENCE (proof) under the assumptions of §16.2; SIMULATED checks (E1) | validity outside those assumptions |
| Compression factors at stated retention | MEASURED for a named encoder, corpus and query pool | a compression guarantee for anything else |
| Comparison with published methods: one comparable setup | EXTERNAL (Q7) | a claim of novelty or superiority |
| The ColQwen2 gap to SAP | unresolved (E7a, E7c) | evidence in either direction |
| The search space's behaviour on ColQwen2, ColQwen2.5 and ColEmbed 4B | MEASURED after the space was fixed, with no commitment not to change it | a held-out test |

---

## 10. X1 protocol: label-free reference agreement

### 10.1 What X1 is, and is not

- **Primary: a deployment-oriented label-free certification experiment.** It runs the
  frozen procedure with reference agreement in place of labelled retention and records
  what it deploys. What is certified is agreement with the uncompressed index.
- **Secondary: a surrogate-relationship analysis.** It examines how agreement relates to
  labelled retention on stores where both exist.
- **It is not a validation of retrieval quality.** Agreement and labelled quality are
  different constructs (§6). X1 does not assume that one implies the other, and no X1
  result may be reported as a statement about relevance.

### 10.2 Inputs

The seven stores of §9.1: `base_labels`, `cand_labels`, `base_b1`, `cand_b1`.
No vectors are needed. The seven stores, the 40 candidates and both constructs are all
used. None may be dropped.

### 10.3 Gates

Run first, in this order. A failure stops X1 until it is explained in the report.

| gate | check |
|---|---|
| G1 | file hashes equal those in §22 |
| G2 | every value is finite and in [0, 1]; `base_b1` equals 1 for every query |
| G3 | on the six ViDoRe stores, for every query with `base_labels` = 1 and every candidate, `cand_labels` equals `cand_b1` to within 1e-12. INFERENCE (proof): with one labelled page per query, a query the uncompressed index answers at rank 1 has that page as its reference document, so both constructs measure the same rank |
| G4 | Method C under labels at n = 50, 100, 225 reproduces the per-seed selections stored by E1.1 and E1.2 exactly |

### 10.4 Definitions specific to X1

| item | definition |
|---|---|
| Reference ranking | exact MaxSim over the uncompressed float32 vectors of the same encoder, ties by document index, as computed when the store was built. Only its top document per query is used |
| Compressed ranking | exact MaxSim over the candidate's decoded vectors |
| Agreement metric | a, as defined in §6 |
| Top-k values | k = 1 (top-1 agreement) and k = 5 (containment@5) on every store; additionally k = 10 on SciFact. These are the only depths the stores support |
| Query-level aggregation | none: a is already one value per query and candidate |
| Pool-level aggregation | R_A = mean of a over the pool; top-1 agreement rate; containment@5 rate |
| Thresholds | T_A ∈ {0.99, 0.97, 0.95} primary, 0.90 secondary |
| Groups | G1: queries with b_L = 1 (the reference document is the labelled page). G0: all other queries. Defined on the ViDoRe stores |

**On the thresholds.** No evidence says which level of agreement is acceptable. The
grid repeats the labelled targets so that the two arms can be compared cell by cell.
That is a design convenience. It is not a claim that T_A = 0.95 corresponds to 95% of
retrieval quality. The value 0.90 is added because agreement is expected to be the
stricter quantity (R7).

### 10.5 Analyses

**A1. Pool-level table (MEASURED).** For each store and each of the 37 distinct
candidates: R_L, R_A, top-1 agreement, containment@5, and R_A separately on G1 and G0.

**A2. Condition CP1 and the oracles (MEASURED).** For each store and threshold: the set
of candidates passing under each construct, the free oracle and the chain oracle under
each construct, and whether CP1 holds.

**A3. Certification under agreement (SIMULATED).** P0 applied to (b_A, a) with the
resampling protocol and the registered budgets of §11. Per cell: deployment rate,
non-trivial deployment rate, median and 10th/90th percentile compression, miss rate
against R_A with its interval. Every cell carries its budget class, pool-supported or
extrapolated (§11.1).

**A4. Cross-construct outcome (SIMULATED).** For every resample in which the agreement
arm deploys a candidate, record that candidate's R_L. The cross-construct miss rate is
the share of resamples in which a candidate is deployed and its R_L is below the same
numeric threshold. This is condition CP2. No statistical statement covers it; it is an
empirical test.

**A5. Paired comparison of the two arms (SIMULATED).** The same resample indices are
used under both constructs. Per cell: the median over resamples of
log2(compression under agreement ÷ compression under labels), and the shares of
resamples in which the agreement arm deploys less, the same, or more.

**A6. Exploratory.** Agreement loss by quartile of the baseline score margin, using
`Q3/features` on the ViDoRe stores. Labelled exploratory in the report; no hypothesis
depends on it.

### 10.6 What the existing data cannot support

- **Only one family of agreement measure is available**: where the reference's single
  top document lands, down to rank 5 (rank 10 on SciFact). Overlap between top-k lists,
  rank correlation, or agreement at depth cannot be computed from the stores.
- **Extension, secondary, limited.** X3 re-scores a subset of datasets. From its
  identity run, the top-10 lists of the reference and of each candidate are saved, and
  overlap@5 and overlap@10 (size of the intersection of the two top-k lists, divided by
  k, averaged over queries) are reported for those datasets only. No hypothesis depends
  on them.
- **Three pools, development encoders.** Nothing in X1 can establish agreement as a
  surrogate for quality in general.

---

## 11. X2 protocol: query budget and certification

### 11.1 The registered budgets, and two classes that are never mixed

**Common grid: n ∈ {50, 100, 225, 450, 900, 1800, 3600}. On a pool with fewer than 450
queries, the pool size N is added.** That adds 300 on SciFact. No other budget may be
added.

**No pool contains 900, 1,800 or 3,600 queries.** The largest has 494. The registered
budgets therefore fall into two classes, decided for each store by its pool size N.

| class | budgets | what a result at such a budget is |
|---|---|---|
| **Pool-supported** (n ≤ N: within the observed range) | 50, 100, 225, 450 on the six ViDoRe stores; 50, 100, 225, 300 on SciFact | The pool holds at least n distinct, independent queries, so a real calibration sample of that size exists. The rate is still estimated by resampling, with a sample no larger than the data, which is the device E1 used. Mark: *SIMULATED, pool-supported* |
| **Extrapolated** (n > N: beyond anything observed) | 900, 1800, 3600 on every store; also 450 on SciFact | A sample of that size contains at most N distinct queries, each drawn several times. The result describes P0 if the query population consisted of exactly these N queries with equal weight. Mark: *SIMULATED, extrapolated* |

**What an extrapolated result may and may not be used for.**

- It may be reported in this form only: "if queries were drawn from this pool's N
  queries, P0 would …".
- It is not an observation, and it is not evidence of what P0 does with n independent
  queries. Above the pool size the simulation resolves differences smaller than the
  pool's own uncertainty about the real query population, so a deployment it shows can
  rest on a margin the data do not support.
- It decides no primary hypothesis.

**The only observed budget is the pool itself.** One MEASURED outcome exists per store,
construct and threshold: the decision of P0 given the entire pool once, each real query
used once, in its stored order. It has no truth to be checked against and is reported as
a single realisation.

| budget | why it is registered |
|---|---|
| 50, 100, 225 | the E1 grid. Results at these budgets under labels must reproduce E1 exactly (gate G4) |
| 450, or N where N < 450 | the pool-capped budget n_cap: what Method C can use when the whole pool is available, since it needs no hold-out half |
| 900, 1800, 3600 | three doublings beyond the pools. 900 is the first doubling above every lower bound in `E1/DESIGN.md` and `E1_2_FREEZE.md` for a 0.99 target at Method C's level (460–723) |

### 11.2 Sampling

- Seeds 0–1999. For seed s the sample is the first n entries of
  `np.random.default_rng(s).integers(0, N, size=3600)`: i.i.d. with replacement from
  the pool.
- The run first verifies that the prefixes of length 50, 100 and 225 equal the E1 draws
  `integers(0, N, size=n)`. If they do not, every budget is drawn separately as in E1,
  and nesting across budgets is not assumed.
- The order of the queries inside a sample is the order drawn. It is never rearranged.

### 11.3 Arms, targets and procedures

| factor | levels |
|---|---|
| Construct | labels; agreement (the X1 arm) |
| Threshold | 0.99, 0.97, 0.95 for both; 0.90 for agreement only |
| δ | 0.05 primary; 0.01 secondary |
| Stores | all seven |

| procedure | role | budgets | seeds |
|---|---|---|---|
| **P0** (Method C over S0) | primary | all registered | 0–1999 |
| I8: test only `int8(per_vector)` at level δ | baseline: the simplest certified claim, with no multiplicity | all registered | 0–1999 |
| D (Bonferroni over 40), C-split | comparators from E1 | all registered | 0–1999 |
| A (shipped rule), A-bonf, B | comparators from E1 | 50, 100, 225 exactly as in E1 | 0–1999 |
| A, B | comparators | the pool-capped budget, and 900 | 0–499 (compute) |

A and B are not run above 900. Their bootstrap makes them the only costly part, and no
hypothesis depends on them. For every procedure, n is the total number of queries it
receives, as in E1: rule A's hold-out half is counted.

### 11.4 Outcomes per cell

For each store × construct × threshold × δ × budget × procedure (all SIMULATED):

- deployment rate and abstention rate;
- non-trivial deployment rate; substantial deployment rate;
- median, 10th and 90th percentile of compression (1.0 when abstaining), and the mean of
  log2 compression;
- miss rate against the pool truth of the same construct, with a 95% Clopper–Pearson
  interval;
- median compression divided by the chain oracle and by the free oracle.

The threshold, δ, the budget and the budget class are reported with every row. The
MEASURED full-pool decision of §11.1 is listed beside the simulated rows of its store.

### 11.5 Aggregate analysis, fixed here

1. Per cell, the curve of non-trivial deployment rate against budget. Pool-supported and
   extrapolated budgets are tabulated in separate columns and drawn with different
   markers. They are never joined into one unmarked curve.
2. **H3, primary.** Per store, the non-trivial deployment rate at the pool-capped budget
   under labels at T = 0.97, δ = 0.05, and the verdict of §7.
3. **Extension beyond the pools.** n* per (store, construct, threshold, δ), by the
   definition in §6, with the class of the budget that decides it. The "at every larger
   budget" clause of that definition is what handles a curve that is not monotone.
4. The same two steps at T = 0.99 and for "substantial" deployment, as secondary.

**Borderline rule.** If the Clopper–Pearson interval of a rate that decides a verdict or
an n* contains 0.90, the cell is re-run with seeds 2000–3999 and the pooled 4,000-seed
rate decides. The report states which cells used this rule.

### 11.6 Checks

- **Reproduction (G4).** As in §10.3.
- **Monotonicity.** INFERENCE (proof): the betting p-value is one over the running
  maximum of the wealth, so on nested samples it cannot increase with n. The set of
  rejected candidates in each chain can therefore only grow, and the compression P0
  deploys cannot decrease. If the samples are nested, a single violation for P0 is an
  implementation error and stops the run. No such property is assumed for A or B, or
  for rates across cells.
- **Validity.** For P0, I8, D and C-split, in every cell: the miss rate is consistent
  with δ (§16.3).

### 11.7 Predictions recorded before the run

PROJECTED from the E1 lower bound; the values for labels are those already tabulated in
`E1/DESIGN.md` and `E1_2_FREEZE.md`.

- Under labels at δ/7, P0 does not reach a non-trivial deployment rate of one half
  below 460–723 queries at T = 0.99, 152–238 at 0.97, and 90–141 at 0.95, depending on
  the store's baseline mean.
- Under agreement the baseline mean is 1, which is the most favourable case of the
  bound. The architecture proposal gives 423, 140 and 83 for the three thresholds.
- These bounds are for a lossless candidate. A candidate whose retention exceeds 1 is
  not covered by the argument, so they are predictions, not limits that must hold.
- At T = 0.97 the pool-capped budget exceeds the bound on every store, by a factor of
  about 1.6 on SciFact and 1.9 to 3 on the ViDoRe stores. H3 can therefore hold or
  fail; which one depends on how much the first candidates of each chain lose.
- At T = 0.99 the pool-capped budget is below the bound on every store. No non-trivial
  deployment is expected there at any pool-supported budget.

### 11.8 What X2 can and cannot show

- **At pool-supported budgets** it estimates how P0 behaves with as many queries as the
  pool really holds. This is the part that bears on H3.
- **At extrapolated budgets** it shows how P0 would behave if the query population were
  the pool's own N queries. It cannot show what 900, 1,800 or 3,600 independent queries
  would give, and it cannot be confirmed with these pools.
- **It does not presume that more queries help.** A curve that stays below 0.90 up to
  3,600 is a registered outcome (§7, §19).
- Pool truths under labels are already known (E1), so the behaviour of P0 under labels
  is partly predictable. The agreement arm is the part with unknown outcomes.
- With three pools it cannot separate "the budget" from "the composition of this pool".

---

## 12. X3 protocol: coordinate and symmetry test

### 12.1 Question

Are the measured retentions, and the decision of P0, properties of the geometry MaxSim
uses, or do they depend on the coordinate basis and vector order the encoder happened
to emit?

### 12.2 Which transformations are valid

Two requirements, both needed.

1. **It leaves every MaxSim score unchanged**, so the uncompressed system is the same
   retrieval system.
2. **It keeps document and query vectors L2-normalised**, because every measured encoder
   emits unit-norm vectors and the library states that its stages assume them
   (`encoders/base.py`). A transformation that breaks this does not produce another
   valid representation of the same encoder. It produces an input outside the tested
   scope.

| transformation | scores unchanged | unit norm kept | decision |
|---|---|---|---|
| Orthogonal map U applied to all document and query vectors | yes: (Uq)·(Ud) = q·d, up to float32 rounding | yes | **registered** |
| Signed permutation of coordinates (a special orthogonal map) | yes | yes | **registered as the control** |
| Reordering the vectors inside a document | yes: a maximum does not depend on order | yes | **registered** |
| Adding one constant vector to every document vector | rankings unchanged, scores shifted | **no** | rejected: outside the unit-norm scope. Its effect on sign codes is already measured (R8, R11) |
| Multiplying every document vector by one positive constant | rankings unchanged | **no** | rejected: outside the unit-norm scope |
| Invertible linear map on documents with its inverse transpose on queries | yes | **no** | rejected: outside the unit-norm scope |
| Duplicating a vector inside a document | yes | yes | rejected: it changes the uncompressed size, so compression ratios stop being comparable |
| Adding a constant vector to every query vector | **no**: it changes which document vector is the maximum | — | rejected: not a symmetry |
| Rescaling individual vectors, documents or query tokens | **no** | — | rejected: not a symmetry |
| Reordering documents in the corpus | yes | yes | not tested in this phase: low expected information for one more full pass |

### 12.3 Registered transformations and seeds

| id | transformation | applied to | seed |
|---|---|---|---|
| T0 | identity | — | — |
| T1 | signed coordinate permutation | documents and queries | 0 |
| T2a | dense random rotation | documents and queries | 1 |
| T2b | dense random rotation | documents and queries | 2 |
| T3 | random reordering of the rows of each document, with its row annotations | documents | 3 |

- A rotation is the orthogonal factor of a QR decomposition of a d × d standard normal
  matrix from `np.random.default_rng(seed)`, with each column multiplied by the sign of
  the corresponding diagonal entry of R. It is applied in float64 and stored as
  float32. Vectors are not re-normalised afterwards.
- No other transformation, seed or strength may be added.

### 12.4 Datasets, fixed here by a rule

One store per encoder, alternating the two ViDoRe pools in the order the encoders were
first measured:

- **Primary set:** D1 (ColPali, DocVQA), D4 (ColQwen2, InfoVQA), D5 (ColQwen2.5,
  DocVQA), D7 (text ColBERT, SciFact).
- **Extension set:** D2, D3, D6. Run only after the whole primary set is complete.
  Reported separately. Never substituted for a primary dataset.

Every dataset is run with all five transformations and all 40 candidates.

### 12.5 Procedure

1. For each dataset and transformation, apply the transformation to the stored vectors,
   then rebuild the per-query store exactly as E1.0 did: every candidate of S0 is fitted
   and applied to the transformed corpus, queries pass through the candidate's query
   transform, and exact MaxSim gives per-query values under both constructs.
2. **Gate, T0.** The T0 store must equal the committed E1 store. E1.0 obtained a maximum
   difference of 0.0. If the environment has changed, the gate passes when all 40 pool
   retentions agree within 0.001 and at least 99.5% of per-query entries are identical.
   Any difference is reported. If the gate fails, X3 stops.
3. **Symmetry check per transformed dataset, on the uncompressed vectors.** The largest
   absolute difference between the transformed and original score matrices, divided by
   the largest absolute original score, is at most 1e-4; and the uncompressed per-query
   metric is identical for at least 99% of queries with the pool mean within 0.001. A
   failure means the transformation was implemented incorrectly. The code is fixed and
   the run repeated; the failed attempt is logged.
4. P0 is run on each transformed store with the X2 resampling protocol at n = 225 and
   at the pool-capped budget (450, or 300 on SciFact), under labels,
   T ∈ {0.99, 0.97, 0.95}, δ = 0.05. Both budgets are pool-supported on every primary
   dataset.

PROJECTED from the build times recorded in the store metadata (8 to 54 minutes per
store): one pass over a dataset takes tens of minutes on the laptop, so the primary set
with five transformations is of the order of several hours.

### 12.6 Metrics, tolerance and decision

**Per candidate.** Δ = R_L(transformed) − R_L(T0), with a 95% interval from a paired
bootstrap over queries (2,000 resamples, seed 0, percentile). The same for R_A, as
secondary.

| class | rule |
|---|---|
| Invariant | absolute Δ ≤ 0.005 |
| Material | absolute Δ ≥ 0.02 and the interval excludes 0 |
| Small | anything else |

0.02 is the spacing of the registered targets, so a material change can move a candidate
across a target. 0.005 is below the resolution at which any decision changes.

**Per procedure outcome.** Against T0, with the same seeds, in each of the six settings
of step 4 (two budgets × three thresholds): the ratio of median certified compression,
and the difference in non-trivial deployment rate. The procedure outcome is material if,
in any one of the six settings, the ratio is at least 2 or at most one half, or the rate
differs by at least 0.25.

**Decision rules.**

- **A (dataset, transformation) pair holds** if no candidate is material and the
  procedure outcome is not material. Otherwise it does not hold.
- **H5 is supported only if all 16 primary pairs hold.** It fails if any pair does not.
  There is no "undecided" outcome: a material change under one registered
  transformation is a failure of invariance under that transformation.
- **How a failure is described.** Under T1 (signed permutation) or T3 (reordering): the
  implementation's result depends on coordinate or vector order, for example through
  tie handling. Under T2a or T2b: the result for that candidate on that dataset is not
  invariant under that rotation.
- **The two rotations are compared** as a secondary outcome. Their agreement is not
  required for a verdict.
- **What support would mean.** Invariance under these four maps, on these datasets,
  within these tolerances. Two rotations cannot establish invariance under every
  orthogonal map, and the report must not say they do.

### 12.7 Predictions recorded before the run

INFERENCE; none is supported by a measurement yet.

- T1: every candidate invariant.
- T3: codec-only candidates exactly unchanged; merged candidates invariant.
- T2: `float16` and the per-vector int8 chains invariant; centred int4 invariant;
  binary: no directional prediction.
- Merging uses cosines between vectors, which a rotation preserves, so any change in a
  merged candidate under T2 should come from its codec.

### 12.8 What X3 cannot show

- It does not explain the plain-int4 collapse at 2,560-d: those vectors were not kept,
  and plain int4 is not in S0.
- It does not probe the centred-binary hazard: that needs a shared direction in the
  queries, which no valid symmetry creates, and centred binary is not in S0.
- It says nothing about differences between real encoders that are not coordinate
  changes, such as how strongly their vectors share a common direction.
- Supporting H5 does not show that results are independent of the coordinate basis in
  general. It shows invariance under four registered maps.
- Passing X3 does not make a procedure model-blind, and failing it does not make it
  model-aware. Model-blindness is defined in §15 by what the procedure is allowed to
  know.

---

## 13. Future held-out encoder protocol (XH): designed here, not run

### 13.1 Three sets, and the order of events

**This document does not register XH. It registers the rules by which XH will be
registered.** Three sets must not be confused.

| set | what it is | fixed by | fixed when |
|---|---|---|---|
| **Candidate pool** | the nine encoders of §13.4 | this document | at its commit. Nothing is ever added to it in this phase |
| **Registered subset** | the members of the candidate pool that will be evaluated | the registration addendum | at the `heldout-freeze` tag, before any of their vectors exist |
| **Analysed set** | the registered encoders whose encoding succeeded | the encoding record | before any compression outcome exists for any held-out encoder |

| step | event | record |
|---|---|---|
| 1 | this document committed | commit hash in the registry |
| 2 | X1, X2, X3 run and reported | reports under `reports/research/X1`, `X2`, `X3` |
| 3 | **Registration.** An addendum lists all nine candidates, each with a disposition: registered, or not registered for one of the reasons in §13.3. It fixes the model revisions, the corpora, the query pools (by the rules of §13.5) and a compute plan | the addendum, committed and tagged `heldout-freeze` |
| 4 | **Encoding only.** Every registered encoder is encoded on every registered dataset, and the baseline gate is applied. No candidate of any search space is applied to these vectors | an encoding record with the hash of every vector file, committed before step 5 begins |
| 5 | **Analysis.** Stores are built and every procedure is run, exactly as §13.6–§13.9 | a report of every registered cell |

- **After step 3** no encoder is added, removed or substituted.
- **After step 4** the set of encoders cannot change at all.
- Steps 4 and 5 are separate so that nobody who has seen a compression outcome can still
  influence which encoders are analysed.

### 13.2 How an encoder becomes "held out"

All four must hold, and are stated in the registration addendum for each encoder.

1. No vector produced by that checkpoint has been compressed or scored by this project,
   on any branch or machine, before the tag.
2. It is not one of the development encoders: ColPali-v1.3, ColSmol-256M,
   answerai-colbert-small-v1, ColQwen2-v1.0, ColQwen2.5-v0.2, ColEmbed 4B v2.
3. It is a multi-vector encoder whose own documentation scores it with MaxSim. A
   per-query constant, such as dividing by the number of query vectors, does not change
   rankings and still counts as MaxSim.
4. Its family is recorded before encoding: backbone lineage, modality, and whether the
   vectors come from a learned low-dimensional projection or from raw hidden states. It
   is labelled "development family" if it shares the backbone lineage of a development
   encoder, otherwise "new family". The addendum writes down the backbone lineage of
   each development encoder, from its model card, so that this comparison is on record
   before any encoding.

### 13.3 Registration rules, and what may be known when

**Every member of the candidate pool is registered unless one of these four reasons
holds.** The addendum states the reason for each encoder it does not register.

1. It fails an eligibility condition of §13.2.
2. Its licence forbids the use.
3. A load test shows it cannot be loaded with every weight taken from the checkpoint.
4. The compute plan is exhausted. Candidates are taken in the order of §13.4 until the
   plan is used up, so this reason can only remove a tail of the list.

There is no other ground. An encoder is never left out because of what is known or
expected about how well it compresses.

| | before the tag, to decide dispositions | after the tag |
|---|---|---|
| **Allowed** | the model card: dimension, modality, licence, intended scoring, declared Matryoshka dimensions; a published uncompressed score; parameter count and expected encoding cost; a load test that reads only the shape of the output | encoding and the baseline gate (step 4); then the registered analysis (step 5) |
| **Forbidden** | any statistic of the encoder's vectors; applying any candidate to them; using any compression, selection or certification result for the checkpoint, published or our own, as a ground for a disposition | adding, removing or substituting an encoder, dataset, arm, budget or threshold; changing a frozen procedure; computing any compression outcome before the encoding record is committed; re-encoding with different settings after any outcome is seen; excluding an encoder or a cell for a reason not listed in §20 |

**How a registered encoder can still leave the analysed set.** Only through a technical
failure during step 4, written into the encoding record: it cannot be loaded or encoded,
or its encoding exceeds the compute plan. It is listed as failed and is not replaced.

**A failed baseline gate does not remove an encoder.** Its results are reported in a
separate table and left out of the hypothesis counts. That rule is fixed here, before
any outcome.

### 13.4 Candidate pool and order

A closed list of nine. Attributes marked EXTERNAL come from the architecture proposal
and were not verified; they are descriptive and are confirmed from the model cards at
registration. Which of the nine are evaluated is decided at registration by §13.3 and by
nothing else.

| order | encoder | modality | source of the entry |
|---|---|---|---|
| 1 | `colbert-ir/colbertv2.0` | text | repository registry ("unverified") |
| 2 | `lightonai/GTE-ModernColBERT-v1` | text | repository registry ("unverified") |
| 3 | `mixedbread-ai/mxbai-edge-colbert-v0-17m` | text | EXTERNAL |
| 4 | `BAAI/bge-m3`, multi-vector output | text | EXTERNAL |
| 5 | `jinaai/jina-colbert-v2` | text | EXTERNAL |
| 6 | `vidore/colSmol-500M` | page images | repository registry ("untested"); development family |
| 7 | `ModernVBERT/colmodernvbert` | page images | EXTERNAL |
| 8 | `nomic-ai/colnomic-embed-multimodal-7b` | page images | repository registry ("unverified"), listed there under the ColQwen family |
| 9 | `nvidia/nemotron-colembed-vl-8b-v2` | page images | repository registry ("untested"); development family |

The order matters only for reason 4 of §13.3. The registration addendum gives a
disposition for each of the nine, and the report covers every registered encoder,
including those that failed.

`google/xtr-base-en`, which the architecture proposal suggested, is left out. Its
retrieval-time scoring imputes missing similarities (EXTERNAL), so whether exact MaxSim
is its intended scorer would have to be decided later, and a decision left for later is
what a preregistration is meant to avoid.

### 13.5 Pools, tiers and the baseline gate

| tier | definition | claim it can support |
|---|---|---|
| Tier 1 | a held-out encoder on a development pool: SciFact for text; ViDoRe V1 DocVQA and InfoVQA (the same 500-page subsets) for page images | unseen encoder, seen pool |
| Tier 2 | a held-out encoder on a pool this project has never used | unseen encoder, unseen pool |

Tier 1 is weaker: the default search space and Method C's settings were chosen with
those pools in view.

**Tier 2 pools are chosen by rule at registration, not by preference.**

- Text: BEIR datasets other than SciFact whose full corpus has at most 30,000 documents
  and whose test pool has at least 300 queries, in alphabetical order, up to two.
- Page images: ViDoRe V1 subsets other than the four already used (DocVQA, InfoVQA,
  energy, TabFQuAD) with at least 300 queries, in alphabetical order, up to two.
- If none qualifies, Tier 2 is empty and the report says so.

**Baseline gate.** Before any compression result is computed for an encoder on a
dataset, its uncompressed score is compared with a published score for the same
dataset and metric, where one exists. The gate passes within 2.0 points on a 0–100
scale. The largest gap accepted so far is 1.8 points (ColQwen2.5 on DocVQA, R11). If no
published score exists, the cell is marked "baseline unverifiable" and kept.

### 13.6 Procedures compared, and the price of model-blindness

All procedures see the same resampled calibration sample of n queries per seed (seeds
0–1999, protocol of §11.2), with n = n_cap, the pool-capped budget, at the primary
setting. "First half" (S1) and "second half" (S2) are the two index sets of
`split_queries(n, 0.5, 0)`, as in E1. Every fitted stage, including PCA, is fitted on
documents only.

| procedure | definition | statement |
|---|---|---|
| **Frozen: P0** | Method C over S0 on all n queries | valid under §16.2 |
| Frozen, half | P0 on the second half only | valid |
| Frozen, wide arm (secondary) | Method C over `wide_search_space(d)` of v0.3.0 at δ divided by its number of chains | valid |
| **Tuned** | On the first half, compute the point retention of every candidate in the superset space S+ (§14.4). Keep the candidates whose first-half retention is at least T and that no other candidate beats on both size and first-half retention. Order them from largest to smallest stored size. Run fixed-sequence testing on the second half at level δ. Deploy the smallest rejected | valid: the order depends on the first half only (the C-split argument of `E1/DESIGN.md`) |
| Oracle | the free oracle over S+ on the whole pool | none: an upper reference |
| Baselines | rule A as shipped; I8 | A: none. I8: valid |

**The price of model-blindness: one definition.** For a held-out cell c (one encoder on
one query pool) and a seed s:

| symbol | definition |
|---|---|
| C(k) | the compression of candidate k, as defined in §6. For a candidate with a PCA stage the library's own figure leaves out the projection matrix (CODE: `CompressedCorpus.nbytes`), so input dimension × output dimension × 4 bytes is added to its stored bytes. A truncation needs no matrix |
| C_P(c, s) | C of the candidate that procedure P deploys on seed s; 1 if P abstains |
| **Model-blind value** C_blind(c, s) | C_P for P = "P0 applied to S2 only" |
| **Tuned value** C_tuned(c, s) | C_P for P = "tuned": tuning on S1 over S+, then certification on the same S2 |
| **Ratio** ρ(c, s) | C_tuned(c, s) ÷ C_blind(c, s). Tuned over model-blind, always |
| **PoMB(c)** | the median over s = 0, …, 1999 of log2 ρ(c, s), as `numpy.median` computes it |
| PoMB(e) | for encoder e, the median, again by `numpy.median`, of PoMB(c) over its valid cells in which PoMB is defined |
| Summary | the median and the maximum of PoMB(e) over the held-out encoders that have at least one such cell |

- **Why both values use S2.** Both procedures then certify on exactly the same queries.
  The only difference between them is that the tuned one was allowed to learn about the
  encoder from S1. That isolates the value of encoder-specific knowledge.
- **Reading, fixed before any result.** PoMB = x: the tuned procedure typically
  certifies 2^x times the compression of P0. x > 0 is a price paid for model-blindness.
  x = 0 is no price. x < 0 means P0 certified more. Larger is worse for P0. The number is
  reported with its sign and is not renamed.
- **Undefined.** PoMB(c) is undefined if the tuned procedure could not be computed, or
  if both procedures abstain together in more than half of the seeds. Such cells are
  counted and listed.
- **No interval is computed across encoders.**

**Secondary, with no role in any verdict.**

- *Ratio at equal total budget:* the median over seeds of
  log2( C_tuned(c, s) ÷ C_P0-on-all-n(c, s) ), where P0 uses all n queries and the tuned
  procedure still splits them. E1 found that splitting loses ("C-split loses to C"), so
  this ratio tilts towards P0. It is reported beside PoMB and decides nothing.
- *Price of certification:* log2( oracle compression ÷ median over seeds of C_tuned ).

**What limits this definition.** PoMB is relative to this tuned procedure and this
superset space. A cleverer tuner could gain more, so it is a lower bound on what
unrestricted tuning could gain. The report states that wherever it gives the number.

### 13.7 Primary setting and grid

- **Primary:** labels, T = 0.95, δ = 0.05, n = n_cap. That is 450 where the pool has at
  least 450 queries, otherwise the pool size. The primary budget is therefore
  pool-supported on every pool, and a registered pool has at least 300 queries (§13.5).
- **Secondary:** T = 0.97; n = 225; the agreement arm at T_A = 0.95 with n = n_cap.
- **Extrapolated budgets** (900, 1800, 3600) may be reported, marked as extrapolated
  under §11.1. They decide nothing in XH.
- The metric is nDCG@5 for ViDoRe datasets and nDCG@10 for BEIR datasets, as in all
  earlier work.

### 13.8 Shapes, operator availability and failures

- **May operator availability depend on the representation?** Yes, but only through a
  rule frozen before the tag that takes the dimension, the number of vectors per item
  or a label-free statistic of the vectors as input, and is the same for every encoder.
  P0 uses no such rule: S0 is the same 40 candidates for every encoder. The wide arm
  uses one: `projection_targets(d)`.
- **Unseen dimensions and vector counts.** S0 is defined by ratios and by per-vector
  codecs, so it is defined for any dimension and any number of vectors per item. How it
  behaves at the extremes is part of what is measured.
- **A candidate that raises an error or exceeds memory** is recorded as failed. It
  counts as not rejected, so its chain stops there. It is never replaced by another
  candidate.
- **Rehearsal.** Before step 3, the analysis code for the tuned comparator is exercised
  on the development stores, using the 76 configurations of `Q4/store` in place of S+.
  Its output is labelled a development rehearsal. It is not evidence for H1 or H2.

### 13.9 Reporting

For every registered encoder and dataset: its disposition and eligibility record, the
encoding record, baseline gate, tier, family, and every cell's deployment rate,
non-trivial deployment rate, compression quantiles, miss rate with its interval, PoMB,
the ratio at equal total budget and the price of certification. Failed gates, failed
encodings, failed candidates and missing cells are listed with reasons. Tier 1 and Tier
2 are reported separately and never pooled into one headline number.

---

## 14. Frozen search space

### 14.1 S0, the search space of P0

`optivision.calibration.recommended_search_space` as released in v0.3.0, with SciPy
installed. CODE: the function body has not changed since 2026-09-27.

**S0 is identified by content, not by name.** The authoritative definition is the list
of 40 candidate specifications stored in the sidecar files of §9.1, with the hash given
in §22.1. Checked during the pre-commit audit, on configuration metadata only:

- all seven sidecars carry the same 40 specifications in the same order;
- `recommended_search_space(d)` at the frozen code generates exactly that list, for
  d = 128 and d = 96.

A run whose generated candidate list does not hash to the value in §22.1 is invalid.
In particular, a run without SciPy generates a different list and is invalid.

| chain, in order | candidates, least to most aggressive |
|---|---|
| 1 `float16` | float16 |
| 2 `ward+int8` | int8; then Ward merging to 0.5, 0.33, 0.25, 0.15, 0.1 of the vectors, each followed by int8 |
| 3 `adaptive_centred+int8` | int8; then centred adaptive merging at radius 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, each followed by int8 |
| 4 `ward+int4` | as chain 2 with centred int4 |
| 5 `adaptive_centred+int4` | as chain 3 with centred int4 |
| 6 `ward+binary` | as chain 2 with binary |
| 7 `adaptive_centred+binary` | as chain 3 with binary |

Stage settings (CODE):

| stage | setting |
|---|---|
| Ward merging | `HierarchicalMerge(ratio=r)`, `max_distance=None`, `min_vectors=1`. Per document the budget is the ratio times the number of reducible vectors, rounded up, at least one. Protected rows are not merged |
| Adaptive merging | `AdaptiveMerge(radius=r, center="mean")`, all other arguments at their defaults |
| int8 | `Int8Quantizer("per_vector")` |
| int4 | `Int4Quantizer(center="mean")` |
| binary | `BinaryQuantizer()`, not centred |
| float16 | `Float16Quantizer()` |

There is no dimension reduction in S0. The same 40 candidates are used for every
encoder, whatever its dimension or number of vectors.

### 14.2 Decision and certification settings of P0

Frozen in `E1/E1_2_FREEZE.md`; code in `scripts/research/e1_stats.py` and
`e1_1_study.py`. The commits and content hashes are in §22.1. The code at the current
branch head is byte-identical to the code at the freeze commit.

| setting | value |
|---|---|
| Hypothesis per candidate | H0: the expected value of c − T·b is at most 0 |
| Test statistic | Y = (c − T·b + T) / (1 + T), tested against m0 = T / (1 + T) |
| p-value | exact betting p-value, aGRAPA bets, bet cap 0.9 |
| Multiplicity | fixed-sequence testing inside each chain, in the chain's own order, at level δ/7 |
| Choice | the smallest stored size among the rejected candidates; ties by fewer vectors, then by earlier position. Otherwise abstain |
| δ | 0.05 primary; 0.01 secondary |
| Queries used | all n; no hold-out split |

### 14.3 What is frozen before the held-out evaluation

Everything in this list is fixed by this document. None of it may change between the
commit of this document and the end of XH.

- the operators and their settings (§14.1), the candidate list, the chains and their
  order;
- the decision and certification settings (§14.2);
- the metric and cut-off per dataset type, both constructs, the thresholds and budgets;
- the resampling protocol and seeds;
- the tuned comparator, its superset space and the oracle definitions;
- the definition of the price of model-blindness, its direction and its reading;
- the registration rules for held-out encoders (§13.1–§13.5);
- the gates, exclusion rules, stopping rules and decision rules.

### 14.4 The superset space S+ of the tuned comparator

For an encoder of dimension d, S+ is:

- every candidate of S0;
- every pipeline "Ward ratio r or no merging, then PCA to w or no projection, then a
  codec", with r ∈ {0.5, 0.33, 0.25, 0.15, 0.1}, w ∈ `projection_targets(d)` of v0.3.0,
  and the codec in {float16, per-vector int8, centred int4, binary};
- if the model card declares Matryoshka output dimensions, the same grid with
  truncation to each declared dimension in place of PCA.

PCA here is `DimensionProjector(w)` with its v0.3.0 defaults, fitted on documents only,
exactly as `projection_search_space` of v0.3.0 builds it. Truncation is
`TruncateProjector`. Only operators that exist in v0.3.0 appear. S+ is allowed to use
the model card. That is what makes it the encoder-specific arm.

### 14.5 Rules

1. **No new operator** is introduced to make a held-out encoder perform better. An
   operator addition is a new research phase with its own preregistration.
2. **P0 is not revised.** Results of X1–X3 may suggest a better procedure. It may be
   added as a further, secondary arm by an amendment committed before the
   `heldout-freeze` tag. It never replaces P0, and P0 is always reported first.
3. **No arm is added after the tag.**

---

## 15. Model-blindness definition

### 15.1 Definition

A procedure is **model-blind** if, when it is applied to an encoder, everything it does
is determined by (a) what was frozen before any output of that encoder was seen, and
(b) the vectors and queries it is given. It receives no identifier of the encoder, and
no person chooses anything on its behalf after seeing which encoder it is.

### 15.2 Four kinds of knowledge

| kind | example | allowed in a model-blind procedure? |
|---|---|---|
| A. Generic interface knowledge | the input is a set of float vectors per document; scoring is MaxSim; the metric is nDCG@k | yes |
| B. Representation-shape knowledge | the dimension, the number of vectors per document, the data type, or a label-free statistic of the vectors computed by a frozen function | yes, if the rule that uses it was frozen beforehand and is the same for every encoder |
| C. Compression-operator knowledge | which operators exist, and what earlier encoders taught about them, as built into the frozen procedure | yes, as of the freeze. It must be declared (§15.4) |
| D. Encoder-specific tuning | thresholds, search spaces or defaults chosen for one encoder; tuning on that encoder's results; knowing which model is being evaluated | **no** |

### 15.3 Tests, each checkable

1. **No identity.** The procedure's inputs are arrays. Removing every name and
   provenance field leaves its output unchanged.
2. **No human routing.** One entry point and one frozen candidate rule. Nobody picks a
   search space by model type. In this phase the wide search space is a separately
   registered arm applied to every held-out encoder, not something chosen per encoder.
3. **Frozen first.** The procedure is identified by a commit that predates the first
   encoding of the encoder.
4. **No retuning.** Nothing is changed after any result for that encoder is seen.

A procedure that passes these four tests is model-blind *with respect to that encoder*.
The property is relative to an encoder and a date.

### 15.4 Where encoder knowledge already sits in P0

S0 was built from results on development encoders. It contains centred int4 and not
plain int4, and plain binary and not centred binary, because of what was measured on
ColPali and on the text ColBERT (CODE: comments in `recommended_search_space`; R3b, R8).
That is knowledge of kind C. It is allowed, and it is the reason for the next section.

### 15.5 The limitation this creates, stated plainly

- **P0 is not a held-out test on any of the six development encoders.** Its search
  space was fixed when ColPali-v1.3, ColSmol-256M and the text ColBERT had been
  measured, and its comments cite ColPali and the text ColBERT. It was then left
  unchanged after ColQwen2, ColQwen2.5 and ColEmbed 4B were measured, with no prior
  commitment to leave it unchanged. Its decision rule was developed on the ColPali and
  ColQwen2 stores.
- **Applying P0 to those encoders again says nothing about model-blind transfer.** X1,
  X2 and X3 do exactly that. They are development analyses.
- **Only encoders that satisfy §13.2 can test H1 and H2.**

### 15.6 What model-blindness is not

It is not invariance to symmetries of the representation. A procedure can be perfectly
model-blind and still give different results for two encoders that differ by a
rotation. X3 tests that separate property.

---

## 16. Statistical analysis

### 16.1 Metrics

| experiment | primary metric | secondary metrics |
|---|---|---|
| X1 | CP1 (pool level) and CP2 (cross-construct miss rate) | agreement-arm deployment and compression; paired log-ratio against the labels arm; the two oracles; group G1/G0 table |
| X2 | the non-trivial deployment rate at the pool-capped budget under labels at T = 0.97, δ = 0.05, per store (pool-supported) | n* over the whole grid (extension, with the extrapolated part marked); the same at 0.99 and for substantial deployment; deployment and compression curves; ratio to the oracles; all comparators; the MEASURED full-pool decision |
| X3 | for each of the 16 (primary dataset, registered transformation) pairs: the class of every candidate's Δ in R_L, and whether the P0 outcome is material | the same for R_A; the difference between the two rotations; the extension set |
| XH | share of valid held-out cells that succeed (H1); PoMB (H2) | ratio at equal total budget; price of certification; wide arm; baselines; agreement arm |

### 16.2 The statement Method C makes, and its assumptions

Method C is an application of known tools: fixed-sequence testing with a union bound
over chains, and betting p-values (Waudby-Smith and Ramdas, 2023). It is not presented
as a new statistical method. Its statement is

> P( a compressed candidate is deployed and its retention under Q is below T ) ≤ δ,

where the probability is over the draw of the calibration queries. It holds for every
sample size, with no large-sample approximation, and it is conditional on all of the
following.

| assumption | how it is met in X1–X3 | if it is not met or cannot be shown |
|---|---|---|
| A1. The calibration queries are i.i.d. draws from one distribution Q | by construction: resampling with replacement makes Q the pool's empirical distribution | the result is reported as a decision with no checked error level |
| A2. The corpus is fixed | the stores are fixed | any result on a changed corpus is outside the statement |
| A3. The candidate list, the chains and their order are fixed before the queries are seen, and every fitted stage is fitted on documents only | S0 fixed since 2026-09-27; stages are fitted by `Pipeline.compress(corpus)` (CODE) | the candidate is removed from the certified analysis and reported |
| A4. The per-query metric lies in [0, 1] and the baseline mean is positive | gate G2 | the store is invalid |
| A5. The order in which queries enter the sequential test is fixed before their outcomes are seen | resamples: the order drawn. Full-pool run: the dataset's stored order | the result is reported without an error level |
| A6. The reference is taken as given: the labels under L, the uncompressed index's top document under A | stated | not testable here. Label errors are not modelled |

Further limits of the statement, from `E1/RESULTS.md` §8: it is about Q, not about a
finite batch of future queries; it is marginal over calibration samples, not conditional
on the one at hand; it says nothing per query.

In X1–X3, Q is the pool's empirical distribution. A statement checked there is a
statement about that pool.

### 16.3 When a miss rate is "consistent with δ"

As in the E1.2 freeze: the lower end of the 95% Clopper–Pearson interval of the miss
rate is at most δ.

**Replication rule.** A cell that fails this check is re-run with seeds 2000–3999. It
counts as a failure only if the interval of the pooled 4,000 seeds still lies above δ.
Because validity is proved, a confirmed failure means an implementation error or a
violated assumption. It stops the experiment and is never read as a property of the
data.

The same rule, with 0.05, applies to the cross-construct miss rate of CP2. There a
confirmed failure is a finding (H4a fails), since no proof covers it.

### 16.4 Aggregation

- **Across queries:** ratio of sums for R_L, mean for R_A. No other weighting.
- **Across resamples:** rates and quantiles, with Clopper–Pearson intervals for rates.
- **Across stores:** none. Each store is reported. Pools are summarised by their worst
  store.
- **Across encoders:** counts of cells and medians, described as such. No inferential
  statistic.
- Queries from different stores are never pooled.

### 16.5 Multiple comparisons

- The hypotheses are few, each with one fixed decision rule. No p-value is computed
  across hypotheses, so none is adjusted.
- Many cells are checked for validity. The check is deliberately unadjusted, which makes
  it more likely to flag a cell, and the replication rule settles any flag.
- Secondary and exploratory outcomes carry no decision.

### 16.6 Pairing, bootstrap, permutation

- **Paired by seed:** every comparison between constructs, procedures or budgets within
  a store uses the same resample indices.
- **Paired by query:** X3 compares each query's value before and after a transformation.
- **Bootstrap:** only for the X3 interval of Δ (paired over queries, 2,000 resamples,
  seed 0, percentile). It is the standard interval for a difference of two ratios on the
  same queries.
- **Permutation tests:** none. No analysis needs one.

### 16.7 Ties, failures, abstention, invalid runs, missing data

| case | treatment |
|---|---|
| Ties in a ranking | broken by document index, as in the library |
| Ties in stored size | fewer vectors, then earlier position |
| The three duplicated pipelines | kept in both chains for the procedure; counted once in candidate-level tallies |
| A candidate that fails to run | recorded; not rejected; its chain stops there; never replaced |
| Abstention | compression 1.0; included in every median and rate |
| An invalid run (§20) | logged, repeated; its output is never reported as a result |
| A cell that cannot be computed | reported as missing with the reason; never imputed; every denominator is stated |
| Queries whose baseline value is 0 | kept. They enter the sums, as in E1 |
| A non-finite per-query value | the whole store is invalid |

### 16.8 Exploratory analyses

Only these, each labelled exploratory in the report: the margin analysis of §10.5 A6;
overlap@k from the X3 identity run; the comparison of the two rotations beyond the rule
in §12.6. Anything else done after seeing results is a deviation and is labelled so.

---

## 17. Primary outcomes

One per hypothesis.

| hypothesis | primary outcome | where decided |
|---|---|---|
| H1 | the share of valid held-out cells with non-trivial deployment rate ≥ 0.90 and a miss rate consistent with δ, at labels, T = 0.95, δ = 0.05, n = n_cap | §7, §13 |
| H2 | PoMB per held-out encoder, its median and maximum over encoders | §7, §13.6 |
| H3 | per store, the non-trivial deployment rate at the pool-capped budget under labels at T = 0.97, δ = 0.05 | §7, §11.5 |
| H4 | CP1 per (store, threshold) and CP2 per cell | §7, §10.5 |
| H5 | for each of the 16 (primary dataset, registered transformation) pairs, whether it holds | §7, §12.6 |

Every primary outcome of X1–X3 is either MEASURED or SIMULATED at a pool-supported
budget. None rests on an extrapolated budget.

## 18. Secondary outcomes

- **X1:** agreement-arm deployment rate, compression and miss rate per cell; the paired
  log-ratio against the labels arm; free and chain oracles under both constructs;
  R_A on G1 and G0; H4b and H4c.
- **X2:** the extension beyond the pools (n* over the whole grid, extrapolated part
  marked); T = 0.99; substantial deployment; every curve; the ratio to both oracles; I8,
  D, C-split, A and B; δ = 0.01; the MEASURED full-pool decision per store.
- **X3:** Δ in R_A; the difference between the two rotations; the extension set;
  overlap@k.
- **XH:** the ratio at equal total budget per cell; the price of certification; the wide
  arm; rule A and I8; T = 0.97; n = 225; the agreement arm; results by tier and by
  family; extrapolated budgets, marked.

---

## 19. Negative-result criteria

Each row is fixed now, with the sentence that will be reported. An explanation offered
after the fact is labelled INFERENCE and does not change the verdict. No threshold is
revisited.

| id | outcome | what will be reported |
|---|---|---|
| N1 | H1 not supported, or refuted | "P0 certified non-trivial compression on only x of y held-out cells. Model-blind selection did not transfer to the encoders tested." |
| N2 | on more than half of the held-out cells the non-trivial deployment rate is below 0.5 | "The certification abstained on most held-out encoders at this budget." |
| N3 | H2a fails | "The price of model-blindness could not be estimated at this budget." |
| N4 | H2b fails | "With both procedures certifying on the same queries, encoder-specific tuning certified more than twice the compression in the median", or "more than four times on encoder E", with the values and the encoders named |
| N5 | on a held-out encoder nothing in S0 is deployable while the tuned procedure certifies 4x or more | "The frozen search space did not transfer to encoder E." |
| N6 | H3 fails | "With every query the pool can supply (n queries), P0 did not reach non-trivial deployment at T = 0.97 on store D." Nothing is added about larger budgets |
| N7 | the H3 extension fails in simulation | "Even 3,600 queries drawn from the pool's own N queries did not produce non-trivial deployment at T = 0.97 on store D. Within the registered range, more queries did not remove the abstention." |
| N8 | H4a fails | "Certifying agreement did not protect labelled retention on store D. Label-free certification certifies agreement only." |
| N9 | H4c fails | "Under agreement at 0.95, with every query the pool can supply, P0 did not reach non-trivial deployment on store D." |
| N10 | under agreement the median certified compression is at most half that under labels at the same budget and threshold, in more than half of the pool-supported cells where the labels arm deploys non-trivially | "Agreement is materially stricter than labelled retention. Using it cost at least a factor of 2 in compression." |
| N11 | H5 fails under a rotation (T2a or T2b) | "The result for candidate k on dataset D is not invariant under the registered rotation." |
| N12 | H5 fails under the signed permutation (T1) or the reordering (T3) | "The implementation's result for candidate k on dataset D changes with the order of coordinates or of vectors." |
| N13 | a confirmed validity failure (§16.3) | "An error level was not kept in cell c. The cause is an implementation error or a violated assumption. No usefulness claim is made from that cell." |

A negative result on development encoders (N6–N12) is not softened by pointing to XH,
and a positive one is not presented as evidence of transfer. Every sentence above that
applies appears in the summary of the report, not only in its body.

---

## 20. Exclusion rules

**Data excluded from the start.**

- The generated corpora and the ColSmol-256M caches: 60 pages and 72 queries, "a sanity
  check, not evidence" (README).
- ColEmbed 4B: no per-query data and no vectors (§9.3).

**Queries.** None is excluded. A store with any non-finite value is invalid as a whole.

**Candidates.** None is excluded.

**Runs.** A run is invalid, and is repeated, if any of these holds:

- a gate fails;
- an input hash differs from §22;
- the code that ran is not the code at the recorded commit;
- the process was killed, or the machine slept during a timed or long run. Both have
  happened in this project (Q4, Q8).

**Held-out encoders.** There are exactly two points at which an encoder can drop out,
and both come before any compression outcome exists for any held-out encoder.

- **At registration** (§13.3): it is not registered for one of four reasons: ineligible
  under §13.2; licence; fails the load test; compute plan exhausted.
- **During encoding** (§13.1, step 4): a registered encoder cannot be loaded or encoded,
  or its encoding exceeds the compute plan. The encoding record says so.

After the encoding record is committed, no encoder is added, removed or substituted for
any reason.

A failed baseline gate is not an exclusion. That encoder's results are reported in a
separate table and left out of the hypothesis counts.

**No exclusion is made after results for that unit have been seen**, other than an
invalid run by the objective criteria above.

---

## 21. Stopping rules

1. **The design is fixed.** Every registered cell is run with 2,000 seeds. Nothing is
   stopped or extended because of a result.
2. **Extra seeds are used only when a pre-stated rule calls for them**: the borderline
   rule of §11.5 and the replication rule of §16.3.
3. **A failed gate stops that experiment.** Only implementation errors may be fixed.
   The protocol is not changed.
4. **X3:** each (dataset, transformation) gets at most three attempts. After that it is
   reported as incomplete. A primary dataset is never replaced.
5. **XH:** after the tag, no encoder, dataset, arm, budget or threshold is added,
   removed or substituted. No compression outcome is computed before the encoding
   record is committed. XH stops when every registered cell is done, failed, or
   excluded by rule.
6. **No library change.** X1–X3 need research scripts only. If a change to
   `src/optivision` appears necessary, work stops and the question goes to the project
   owner.
7. **The phase ends with its reports.** Further analysis is a new phase, or is labelled
   exploratory.

---

## 22. Provenance requirements

### 22.1 Frozen inputs

Every identifier below is a content hash. Tags and branch names can be moved; these
cannot.

**Commits.**

| item | commit |
|---|---|
| This document | the commit that adds it, to be recorded in the registry |
| Research branch head when this was drafted | `7d87e331912009416caebef55c9a271c1af74d69` |
| The v0.3.0 release | `4d520603469e179135ffd9df922b02d4504b4add`. The tag `v0.3.0` is a lightweight tag, so this hash is the authority, not the tag name |
| E1 preregistration, where the Method C code was first committed | `0ad40bad2ea3604b3570c6c203f54cbd12f2e1ef` |
| E1.2 freeze of Method C | `106ba510538cf1fb8cd8f1be39e665ed59b0dca9` |
| The four development stores | `4086e8cb242f7c127ebaaf1a9c1cf2770bf895d0` |
| The three confirmation stores | `b18ae19cae2b7daabb2ccf7b3c5fff4f2f47e209` |

**The candidate space S0.**

| item | identifier |
|---|---|
| Specification of the 40 candidates | sha256 `af29b01146a90a048ffea9cfbd66f0fedf988134829cd4dc56861f1207893520` |
| What is hashed | the JSON array of 40 objects with the keys `position`, `family` and `pipeline` (the stored stage specification), serialised in UTF-8 with sorted keys and the separators `,` and `:` |
| Where it comes from | identical in all seven sidecar files, and identical to what `recommended_search_space(d)` generates at the frozen code for d = 128 and d = 96, with SciPy installed |
| Library source tree at v0.3.0 | git tree `b38891f0bf6de09739d13c785e8debc48558fa54` for `src/optivision`. The research branch differs from it in one file, `storage.py`, which neither S0 nor Method C uses |

**Code files.** Git blob hashes. Each is the same at the v0.3.0 commit (library files)
or at the E1.2 freeze commit (research scripts) as at the branch head.

| file | git blob |
|---|---|
| `src/optivision/calibration.py` | `58f1727ceeb200f9d84dcd28452821b19d324904` |
| `src/optivision/compose.py` | `85b20d449706b5a5980fa90b3e9be2bd572b9ba8` |
| `src/optivision/stages/base.py` | `1516638bc68f72303a1cee7f234b48be4b2b3346` |
| `src/optivision/stages/merge.py` | `3f17b093c4adcd18fcf29f50a8e2a6d4f55c5a0f` |
| `src/optivision/stages/quantize.py` | `68d8816099a0fe4f32e87fdf04226172380c3d4f` |
| `src/optivision/stages/project.py` | `5d32185f5cfab195de50cd2b5180e401d9d192f6` |
| `src/optivision/compression/__init__.py` | `49b025086b2fe87edb39ce1235c733159b401bc1` |
| `src/optivision/compression/binary.py` | `cc1d86825d2cf1b77de3c283840a6250da7a8b63` |
| `src/optivision/scoring.py` | `3e1677327445077a455afe580320d583d547aee8` |
| `src/optivision/evaluation.py` | `192d1b2717c34cbafbd157ac373e67b695719bf6` |
| `src/optivision/representation.py` | `20adcd1dde815e1cf1f42760196ab4b45c20ffe8` |
| `scripts/research/e1_stats.py` (Method C; holds `BET_CAP = 0.9`) | `13494307a4a42cd2c1da63f34aa0952a92b33130` |
| `scripts/research/e1_1_study.py` (level δ/F, choice of the smallest rejected) | `5896e763b49d4a0f3f0cc988dc2b5ec5e6b2b9aa` |
| `scripts/research/e1_common.py` | `3bef181683307eb98a9c5e2d61ab02d87801bbfe` |

**Store files.** sha256 of the file contents.

| store | `.npz` (per-query values) | `.json` (candidate metadata) |
|---|---|---|
| `colpali_docvqa` | `bb301073e96b170fd8a35db6dbfa937c29cc6211659a88fd9a9b115fb0b0310b` | `d419cdb895316af796a743c5af6f2c8dc784ea4a7888abad02c6ddbc66215283` |
| `colpali_infovqa` | `cd88c233670801101728e21e30d439f1d8644703683ceff3ee9ee7e85295487f` | `62cfd92be80687e0eb5aa82e41b4de501ae8f1313b49375584d7f433b9c02406` |
| `colqwen2_docvqa` | `511fb87d33a3485f3259d0d239658d30a9655513b18b03412731ed80dd83cb6f` | `a68ff3a539a4454eb61ae7607a13a1910c0a93a3c86713a91dbe81dd612d6961` |
| `colqwen2_infovqa` | `3faadf535897b5bb78b0e4adb08d7d60916912522fbd3da1c17d634dac7806f8` | `b11bbb679f4eba4f63552b097ad3daef8f421c36071accfe63b7dfb5101147d4` |
| `colqwen25_docvqa` | `df50ae1b9adaab754962de2f1d2154e691b7a8919413f946958b67b0449e42ca` | `00875fd852234007b27eef7d57b44c1921dcfb942f21674c2badffe916b29107` |
| `colqwen25_infovqa` | `1aed74af53fbb24ac10eb123f463c5de4a2f77e9ece28b04ecfe99caa1a4dc50` | `f31b69e0193efad6e186369cdce17cf39c658519b4896a1ec500014c04d4bd75` |
| `scifact` | `e4ac7f15e8ccf652c0e928ed504f04bc65a9f94b7e11e2b68a3ea1638df90b98` | `d75d093a4409cf1611b833848bcabb72a06e1ccc1f27ca24aba4e5a7112775e3` |

### 22.2 What every run records

- the commit of the code and whether the working tree was clean. Each new analysis
  script is committed before its first run, and the run records that commit;
- the hash of the generated candidate specification, which must equal the value in
  §22.1;
- Python, numpy and SciPy versions; the machine;
- the hash of every input file and every output file;
- seeds, and for X3 the hash of each transformation matrix and of each vector file;
- start and end times, and for long runs a check that the machine did not sleep;
- the gates and their results, including failed attempts.

### 22.3 Where things go

- New scripts: `scripts/research/x1_*.py`, `x2_*.py`, `x3_*.py`, `xh_*.py`. Existing
  scripts are not edited. Method C is imported from `e1_stats.py` unchanged.
- Outputs: `reports/research/X1/`, `X2/`, `X3/`, `XH/`, each with a report, the JSON
  behind every table and the per-seed selections.
- Vectors and other large files stay outside git, with their hashes in the report.
- The registry `reports/research/README.md` gets one row per experiment.

### 22.4 How evidence is labelled in the reports

| result | label |
|---|---|
| per-query values; pool-level R_L and R_A; the single full-pool decision | MEASURED |
| the same on transformed vectors (X3) | MEASURED, marked "transformed vectors" |
| any rate, quantile or curve over resamples at a budget that does not exceed the pool size | SIMULATED, pool-supported |
| the same at a budget larger than the pool, and any n* decided by such a budget | SIMULATED, extrapolated. Reported only in the conditional form of §11.1 |
| lower-bound query counts; compute estimates; anything read beyond the grid | PROJECTED |
| proofs (validity, monotonicity, the G3 identity) and all interpretation | INFERENCE |
| model-card attributes, published scores, cited papers | EXTERNAL |

No table mixes labels without marking each column.

---

## 23. Known limitations

1. **X1–X3 use development encoders.** Three families, three pools, 96-d and 128-d
   only. They cannot test model-blind transfer.
2. **The stores are not independent.** Three stores share each ViDoRe pool, and
   ColQwen2 and ColQwen2.5 are one family.
3. **One pool per corpus, resampled with replacement.** Every SIMULATED result is about
   the pool's empirical distribution. Even at a pool-supported budget the rate is an
   estimate from resampling, not a count over independent samples. The only observed
   decision is the single full-pool run.
4. **The pools are small.** At most 494 queries. No sample of 900, 1,800 or 3,600
   independent queries exists. Results at those budgets are extrapolated simulations
   (§11.1) and decide no primary hypothesis.
5. **Agreement is measured narrowly.** Only the placement of the reference's single top
   document, to depth 5 or 10.
6. **Agreement rewards fidelity to the reference's errors.** On the DocVQA stores the
   uncompressed nDCG@5 is 0.58–0.62 (MEASURED, E1), so for many queries the reference's
   top document is not the labelled page.
7. **Thresholds are conventions**, set by someone who knew the labelled outcomes of
   §9.4. This applies to 0.90 and 0.80 for rates, the factors 2 and 4, and 0.005 and
   0.02 for X3.
8. **H3 is partly predictable** from the known pool truths and the lower bound. It can
   still fail: the pool-capped budget is at most twice the bound on four of the seven
   stores.
9. **PoMB is relative to one tuned comparator and one superset space**, and it compares
   the procedures at half the pool-capped budget, because both must certify on the same
   half.
10. **Tier 1 pools are not unseen.** The search space and the bet cap were chosen with
    those pools in view.
11. **The candidate encoder pool is not verified.** Four of its nine entries come from
    an external search that was not checked. The other five come from the repository's
    registry, where their loaders are marked untested or unverified.
12. **The corpus is fixed and small.** Retention falls as a corpus grows (R10). Nothing
    here addresses that.
13. **The ColQwen2 discrepancy with SAP is unresolved** (E7c). If it came from an
    encoding fault, D3 and D4 would describe a mis-encoded model. They are used as they
    are, and the discrepancy is not evidence for or against anything here.
14. **The bet cap of the betting test** was chosen on synthetic data at the E1 design
    stage.
15. **One analyst, no independent replication.**
16. **Until it is committed, this document constrains nothing.**
17. **XH is registered only by a later addendum.** The registered subset, the corpora
    and pools, the model revisions and the size of the compute plan are fixed there, not
    here. The size of the compute plan is the one discretionary choice left. It can only
    remove a tail of the fixed order of §13.4.
18. **The analysis scripts for X1–X3 do not exist yet.** The protocol is prose. Each
    script is committed before its first run, and the reproduction gate checks the
    shared machinery against E1.

---

## 24. What would constitute evidence for or against the central claim

The central claim: *a model-blind procedure can select and certify useful compression
for late-interaction retrieval on encoders that took no part in designing it.*

**Evidence for it requires all of these.**

- XH run as registered, with at least the minimum design of H1.
- H1 supported, with its Tier 2 cells reported separately and also succeeding.
- Every valid cell consistent with δ.
- H2a holds, and PoMB is reported whatever its size.

X2 and X3 results bear on how useful and how robust the procedure is. They are not part
of the evidence for transfer.

**Evidence against it is any of these.**

- N1, N2 or N5: the procedure did not transfer.
- N4: it transferred at a high price.
- N13: an error level was not kept and the cause is not an implementation error.

**What each kind of result can support.**

| result | strongest statement it supports |
|---|---|
| X1–X3 on development stores | "On three pools and three encoder families, under the pool's distribution, P0 behaved as follows." |
| XH, Tier 1 | "On N unseen encoders and seen query pools, P0 …" |
| XH, Tier 2 | "On N unseen encoders and unseen query pools, P0 …" |
| none of the above | anything about encoders not tested, corpora not tested, other scoring functions, or larger corpora |

A positive XH result would support the claim for the encoders and pools tested. It
would not make the procedure valid or useful for every encoder.

---

## 25. Explicit non-claims

This phase does not claim, and its reports must not imply:

1. **Universal compression.** No configuration, operator or procedure is claimed to
   work for all models.
2. **Any result for an encoder, corpus, corpus size, modality or scoring function that
   was not tested.** The scope is MaxSim over L2-normalised vectors.
3. **That Method C is novel.** It applies known post-selection testing.
4. **That Method C is valid outside its assumptions** (§16.2), or that its statement
   covers a growing corpus, a different query distribution, a finite batch of future
   queries or individual queries.
5. **Any guarantee for the shipped `optimize()` rule.**
6. **That agreement with the uncompressed ranking equals retrieval quality.**
7. **That any result at 900, 1,800 or 3,600 queries is an observation, or shows what
   that many independent queries would give.** No pool holds that many. Those results
   are simulations that redraw the pool's own queries. Nor is it claimed that more
   queries would remove an abstention that the registered budgets did not remove.
8. **That X1–X3 show model-blind transfer.**
9. **Novelty or priority of any finding.** No literature survey of statistical selection
   for index compression has been done (Q10). The spot check in the architecture
   proposal is a list of leads.
10. **Scalability.** The million-page figures of Q8 are projections and stay
    projections.
11. **Anything about SAP.** The ColQwen2 discrepancy stays unresolved.
12. **That results are independent of the coordinate basis.** X3 can show invariance
    under four registered transformations, and nothing wider. Passing it does not make
    a result model-independent.

---

## Appendix A. Evaluation of the architecture proposal

The proposal of 2026-10-04 was checked against the repository and the reports.

| item in the proposal | verdict | reason |
|---|---|---|
| Scope the claims to MaxSim | adopted | all evidence is MaxSim |
| Method C is an application of known methods | adopted | Q10 |
| Count families and query pools, not checkpoints | adopted | §8 |
| Freeze the procedure, then test on unseen encoders | adopted | §13 |
| A baseline gate for every new encoder | adopted | §13.5 |
| A price of model-blindness | adopted, redefined | the proposal said "tuned ÷ frozen, same rule, same n" without saying where the tuned procedure gets its information. §13.6 gives one definition: both procedures certify on the same queries, the ratio is tuned over model-blind, and its reading is fixed in advance |
| X1: Method C under the label-free reference | adopted, narrowed | the stores support only one agreement measure. The proposal leaned towards making agreement the default certified quantity; here it is a separate construct with no default status |
| X2: query-budget curves on new large pools | changed | no new data in this phase. X2 uses existing pools. Its hypothesis is decided at the pool-capped budget. Budgets above the pool size are extrapolated simulations, marked as such, and decide nothing. New pools belong to XH |
| X3: rotation, common shift, common rescaling, reordering, duplication | partly rejected | shift and rescaling leave the unit-norm scope; duplication changes the uncompressed size (§12.2). Rotation, signed permutation and reordering are kept |
| Symmetry as part of the definition of model-blind | rejected | they are different properties (§15.6) |
| "No encoder has ever been held out from the design of the defaults" | corrected | the space was fixed before ColQwen2, ColQwen2.5 and ColEmbed 4B were measured and was not changed afterwards. That is weak, unregistered evidence of transfer, not none (§3) |
| Hypothesis "variation across budgets exceeds variation across encoders" | rejected | not well defined. Replaced by H3, decided at the pool-capped budget |
| Design a second candidate generator after X1 and X3, then freeze | changed | P0 is fixed now and is never replaced. A later procedure can only be an added arm (§14.5) |
| Projected lower-bound counts for the agreement construct | kept as predictions | PROJECTED; cited, not recomputed (§11.7) |
| Candidate held-out encoders | kept as a closed pool | attributes unverified; XTR removed (§13.4) |
| Package architecture, migration, interfaces | not evaluated | outside a research preregistration |

---

## Appendix B. Skeptical review of this document

| question | safeguard | residual weakness |
|---|---|---|
| 1. Could the model selection change after results? | a closed candidate pool of nine; every candidate registered unless one of four stated reasons holds; registration committed and tagged before any encoding; an encoding-only step whose record is committed before any compression outcome exists; no addition, removal or substitution afterwards (§13.1, §13.3, §20) | the registration and the encoding record rely on discipline. Four candidates come from an unchecked external search |
| 2. Could the query budget change after results? | one registered grid; each hypothesis decided at the pool-capped budget; extrapolated budgets marked and decide nothing; "no other budget may be added" | the grid was chosen knowing E1 outcomes at 50–225 |
| 3. Could the search space change after results? | S0 fixed by hash and never revised; added arms only before the tag and never as replacements | none within this phase |
| 4. Could "model-blind" be redefined? | definition and four tests in §15 | whether kind-B rules are "the same for every encoder" needs judgement if such a rule is ever added |
| 5. Could a negative result be read as positive? | every hypothesis has a stated failure outcome (§7); §19 fixes the sentence for each and requires it in the report's summary; the price of model-blindness has one definition and one reading; "not assessable" is defined for H1 | a reader can still choose what to emphasise |
| 6. Does the primary metric measure the research question? | for XH, yes: certified deployment on held-out encoders. For X1–X3, no, and §4 and §24 say so | X1–X3 could be mistaken for the main result. They are not |
| 7. Are labelled quality and agreement conflated? | separate symbols, separate thresholds, separate claims; CP1 and CP2 test the relation instead of assuming it | the same numeric grid is used for both, which invites the comparison. §10.4 states why |
| 8. Are the statistical assumptions explicit? | §16.2, with the consequence of each violation | A1 holds by construction only in simulation. No real deployment is checked |
| 9. Is the held-out protocol genuinely held out? | eligibility, information allowed and forbidden before and after the tag, freeze before encoding, encoding separated from analysis, tiers | Tier 1 pools are seen. At least three of the nine candidates are in development families. XH itself is registered only by the later addendum |
| 10. Is measured evidence separated from inference? | §22.4: MEASURED, SIMULATED pool-supported and SIMULATED extrapolated are three separate marks | — |
| 11. Is a universal guarantee implied? | §25 items 1, 2, 4 | — |
| 12. Is novelty implied? | §25 items 3, 9 | — |
| 13. Is the scope restricted to late interaction and MaxSim? | §5; X3 rejects transformations outside the unit-norm scope | — |

Weaknesses found while drafting and reviewing, and how the text handles them:

- **An encoder whose intended scorer was unclear** was in the candidate pool. It was
  removed (§13.4).
- **A common shift of the documents** was first considered as a symmetry test, as the
  proposal suggested. It is rejected as outside the unit-norm scope (§12.2).

Corrections made in the pre-commit audit of 2026-10-04:

- **Budgets of 900, 1,800 and 3,600 exceed every pool.** An earlier draft decided H3 at
  900. Budgets are now split into pool-supported and extrapolated, every primary outcome
  is decided at a pool-supported budget, and extrapolated results may only be stated
  conditionally (§11.1).
- **H3 presumed that more queries would eventually help.** Its third outcome was "not
  resolved". It is now a claim about the pool-capped budget that can simply fail, with a
  simulation-only extension that can also fail (§7).
- **H5 was worded as invariance under a whole class of transformations.** It is now a
  claim about four registered maps, with no "undecided" outcome (§7, §12.6).
- **The price of model-blindness had two versions and a three-level verdict.** That left
  room to choose a reading afterwards. There is now one definition, one direction and
  one reading. The equal-budget ratio is reported and decides nothing (§7, §13.6).
- **The compression used in that ratio was not fully defined.** The library's figure
  leaves out a PCA matrix, which would have favoured the tuned procedure on small
  corpora. Stored bytes now include every array needed at query time (§6, §13.6).
- **A held-out encoder could have been dropped for compute after other encoders'
  outcomes were known.** Encoding and analysis are now separate steps, and the set of
  encoders is closed by a committed encoding record before any outcome exists (§13.1,
  §13.3, §20).
- **The search space and Method C were cited by abbreviated hashes and a movable tag.**
  §22.1 now gives full commit hashes, file content hashes, and a hash of the 40
  candidate specifications that a run must reproduce.
- **Two decision rules left a choice open**: which cells condition CP2 covers, and which
  settings decide a material procedure outcome in X3. Both are now stated (§7, §12.6).
- **Two statements about the origin of the search space were too strong.** It is now
  described as its own comments describe it (§3, §15.5).

---

## Appendix C. Amendments

### Amendment 1, 2026-10-04. Operational decisions before the first run of X1–X3

**Status.**

- **The registered text is unchanged.** Everything above the heading of this appendix is
  the text committed in `73085ab637679f30fffece6632eb7b9e92adff79` (blob
  `fa9deb73fdf380f06141d89cfaf145444f883506`), byte for byte. This amendment adds text
  under that heading and edits nothing else.
- **Operational clarifications, made before any computation.** It was written on
  2026-10-04, after the harness commit `95b1cc3ed388903976c67c7ab97751a8007cef93` and
  before any X1, X2 or X3 computation. No X1, X2 or X3 result existed. It is committed
  before the analyses it affects, so under §2 it is an amendment and not a deviation.
- **No outcome was inspected.** No per-query outcome value was read to make these
  decisions. What had been read: file hashes, array names, shapes and types, candidate
  metadata, and synthetic test data.
- **What it does not alter.** The scientific hypotheses (§7), the datasets and stores
  (§9), the candidate space S0 (§14.1), every threshold, the Method C settings (§14.2),
  the query-budget definitions and their two classes (§6, §11.1), the seeds, and the
  primary decision criteria (§7, §17). No store or candidate hash of §22.1 is changed.
- **Decided by** the project owner. The scripts are brought into line with this amendment
  in the commit that adds it.

**Four kinds of item.** Every item below carries one of these marks.

| mark | meaning |
|---|---|
| **clarification** | protocol clarification: it settles a point the registered text left open, or says which reading of a sentence applies |
| **implementation** | implementation requirement: how the scripts must behave for a run to be valid and traceable. It has no scientific content |
| **diagnostic** | diagnostic-only output: a number that is reported and decides nothing |
| **amendment** | an explicit pre-run change to a registered operational rule, made before any affected computation and without using experimental outcomes |

Three items state a rule that differs from a sentence of the registered text. Each says so
and governs X1–X3: F1 (the row of §16.7 on a candidate that fails to run, for X3 only), O1
(the drawn curves of §11.5) and O2 (the list of scripts in §22.3).

#### 1. The seven open points

The registered text left seven points technically open. Each is accepted exactly as the
harness at `95b1cc3` implements it. All seven are **clarifications**.

| id | where | decision |
|---|---|---|
| OP1 | §12.3, T1 | `rng = numpy.random.default_rng(0)`; `perm = rng.permutation(d)`; then independent signs `rng.integers(0, 2, size=d) * 2 - 1`. Coordinate j of the transformed vector is `signs[j] * v[perm[j]]`. Applied to documents and queries |
| OP2 | §12.3, T3 | One generator `numpy.random.default_rng(3)`. Documents are processed in corpus order, with an independent `rng.permutation(n_i)` for a document of n_i rows. The row annotations (`protected`, `positions`, `weights`, `importance`) move with their rows. Queries are not changed |
| OP3 | §12.3, T2a and T2b | The rotation convention is U·v: every document and query vector v becomes U v, as §12.2 writes it. For vectors stored as rows X this is `X @ U.T`, computed in float64 in blocks of 65,536 rows and stored as float32 |
| OP4 | §12.5 step 2, the T0 gate | The stricter gate. A T0 store that is not identical to the committed store passes only if both registered constructs and all four registered arrays satisfy it: the pool retentions of all 40 candidates agree within 0.001 under labels and under agreement, and at least 99.5% of the entries are identical in each of `base_labels`, `base_b1`, `cand_labels` and `cand_b1` |
| OP5 | §11.5, the borderline rule | The extra registered seeds (2000–3999) go to these cells and no others: the H3 cells, primary and secondary; the H4c cells; and, for each n*, the cells met when the registered budgets are scanned downward from the largest to the first one that fails, that one included. A cell below the first failure cannot change n* and gets no extra seeds |
| OP6 | §10.5, analysis A6 | Both registered Q3 margins, each reported separately: the relevance margin (`base_margin_rel`) and the top-2 margin (`base_margin_12`). Quartiles of the margin over the store's pool (`numpy.quantile` at 0.25, 0.5 and 0.75; a query on an edge goes to the upper quartile). Per distinct candidate, 1 − mean agreement in each quartile |
| OP7 | §19, N10 | A cell counts as one "where the labels arm deploys non-trivially" when its non-trivial deployment rate under labels is at least 0.90. Cells are (store, threshold, δ, pool-supported budget), and both error levels are counted together |

**D1, diagnostic.** The N10 count is also reported under the alternative reading "more
than half of the resamples deploy non-trivially", in place of "at least 0.90". The
alternative N10 count is always reported alongside the primary N10 count. It is
diagnostic only and does not affect any verdict, threshold, candidate selection, n*, the
H3 or H4c decisions, or any registered hypothesis outcome.

#### 2. Failure handling

A *unit* is one store in X1 and X2, and one (dataset, transformation) in X3, the identity
T0 included.

| id | mark | rule |
|---|---|---|
| F1 | amendment | **A candidate that fails to run in X3.** The error ends that unit. The error counts as one of the unit's three attempts (§21, rule 4). After three attempts the (dataset, transformation) pair is incomplete. F1 replaces the candidate-failure treatment of §16.7 ("recorded; not rejected; its chain stops there") for X3 only, where a store holds one row of values per candidate and has no way to hold a failed one. F1 does not apply to XH, where that row of §16.7 stands as registered |
| F2 | clarification | **H5 with incomplete pairs.** An incomplete pair prevents H5 from being supported. H5 fails if any completed pair fails. H5 is "not assessable" only when no completed pair fails and at least one pair is incomplete. A pair is incomplete when its own unit, or the T0 unit of its dataset, has no valid attempt after three |
| F3 | implementation | **Sleep.** A unit is invalid if the machine slept for more than 5 seconds during that unit. Sleep is measured as the wall-clock time of the unit minus its time on a clock that stops while the machine sleeps. The results of an invalid unit are not written. The attempt is logged and the unit is repeated (§20) |
| F4 | implementation | **Stop conditions.** The stop conditions of the harness at `95b1cc3` remain in force. They are listed below |
| F5 | implementation | **Same code.** Units may be combined into a summary only when the same code produced them. The rule is given below |

**F4, the stop conditions.** A run is refused, or stops and writes no result for the unit
concerned, in each of these cases.

- An input does not have its registered identity (§22.1, and part 4 below); or the
  candidate list the code generates does not hash to the registered value; or a frozen
  code file differs from its registered blob.
- The code is not committed; or the text of this document at the commit that runs is not
  the text the scripts implement.
- A gate fails: G1–G4 in X1; G1, G2 and G4 in X2; the T0 gate or the symmetry check in X3.
- In X2, the selections of A, A-bonf, B, C-split or D at n = 50, 100 or 225 under labels
  differ from those E1 stored. §11.3 runs them "exactly as in E1".
- P0 deploys less compression at a larger budget on nested samples (§11.6).
- A miss rate fails the check of §16.3 after its replication: P0, I8, D and C-split in X2;
  P0 in X1.
- In X1, an agreement value is neither 0 nor 1/log2(1 + r) for a rank r within the cut-off.
- In X3, a rebuilt store holds a value that is not finite or not in [0, 1], or a
  `base_b1` value other than 1; or the candidate list that was scored does not hash to the
  registered value; or, where the T0 store equals the committed store exactly, P0 at
  n = 225 and δ = 0.05 does not reproduce the selections E1 stored.
- In X3, an extension dataset is requested before the primary set is complete; or scoring
  is directed to a device other than the numpy path the stores were built on.

**F5, the same-code rule.**

- Each unit records the identity of the code that produced it. The identity is the git
  hashes, at the commit that ran, of `scripts/research/x_common.py`; of the experiment's
  script (`x1_agreement.py`, `x2_budget.py` or `x3_symmetry.py`); of the frozen scripts
  `e1_stats.py`, `e1_1_study.py` and `e1_common.py`; of the library tree `src/optivision`;
  in X3 also of the dataset loader `scripts/universal_study/s7b_selection_rules.py`; and
  the versions of Python, numpy and SciPy.
- A summary, a report and a verdict are built only from units whose recorded identity is
  the identity of the code that builds the summary.
- A unit with another identity is not used. It is repeated, and in X3 the repeat counts
  as one of the unit's three attempts.
- A commit that changes none of these files, such as one that adds results, leaves the
  identity unchanged.

#### 3. Outputs and reporting

| id | mark | rule |
|---|---|---|
| O1 | clarification | **No figure is required.** §11.5 item 1 speaks of curves "drawn with different markers". The tables that give the pool-supported budgets and the extrapolated budgets in separate columns satisfy §11.5. Both kinds of column hold SIMULATED rates; the only observed outcome remains the full-pool decision (§11.1) |
| O2 | clarification | **Scripts.** `scripts/research/x_common.py`, the module the three experiment scripts share, is part of the registered script set of §22.3 |
| O3 | implementation | **Registry.** The X1, X2 and X3 rows of `reports/research/README.md` are updated by hand after each actual run (§22.3). The scripts do not edit the registry |
| O4 | clarification | **Reporting conventions.** R1–R13 below are frozen. No other reporting convention is introduced |

| id | point | convention |
|---|---|---|
| R1 | full-pool decision: "per store, construct and threshold" (§11.1) names no δ | computed at both registered levels; 0.05 is shown in the report |
| R2 | thresholds of A2 and H4b | CP1 and the oracle comparison at 0.99, 0.97, 0.95; at 0.90 the agreement side only, since 0.90 is registered for agreement alone |
| R3 | H3 secondary: "the same at T = 0.99, and with 'substantial'" | three cells: (0.99, non-trivial), (0.97, substantial), (0.99, substantial) |
| R4 | "the class of the budget that decides" n* | the class of n* itself; when there is none, of the largest budget |
| R5 | an oracle that does not exist | compression 1.0, as E1 does; ratios to an oracle use the median over seeds of compression ÷ oracle compression |
| R6 | I8 | the first of the two `int8(per_vector)` positions (position 1) |
| R7 | validity check (§11.6, §16.3) | applied to P0 in X1 as well; a confirmed failure stops the run |
| R8 | replication and borderline re-runs | only the cell that triggered the rule is re-run; the pooled 4,000 seeds decide |
| R9 | X3 bootstrap | `default_rng(0).integers(0, N, size=(2000, N))`, the library's own convention; 2.5% and 97.5% quantiles; one index set for all candidates |
| R10 | X3 symmetry check | the "uncompressed per-query metric" is the labelled one; entries that are −inf in both score matrices are left out of the difference |
| R11 | X3 materiality | judged on the 37 distinct candidates; a pipeline that opens two chains has the same values in both |
| R12 | H5 when a pair fails and another is incomplete | "fails": a failure under one registered map is a failure; "not assessable" only when no pair fails and one is incomplete |
| R13 | N11 and N12 when only the P0 outcome is material | the registered sentence with "the outcome of P0" in place of "candidate k" |

#### 4. Auxiliary input hashes

**Implementation.** These sha256 hashes are provenance and input-integrity checks only.
They do not change the scientific protocol, and no store or candidate hash of §22.1 is
changed. The scripts stop on a mismatch, before the file is used.

E1 selection files, read by gate G4. Paths are under `reports/research/E1/`.

| file | sha256 |
|---|---|
| `real/colpali_docvqa_selections.npz` | `5118210e00d8a41d2f43d6969347400e709a73f6f29b1fba7b375aaae1efe0bd` |
| `real/colpali_infovqa_selections.npz` | `6f79293b9e41299a52765aa0bbedf1a3973ae55cbd735cde25af89307e17f7f8` |
| `real/colqwen2_docvqa_selections.npz` | `a326c212618ce49980ab8ea2dd7499aeb740ecc0e5b70bbbf1b8ce37167af26f` |
| `real/colqwen2_infovqa_selections.npz` | `5ae5ab0baa51a48d2c34e99e52093f36489707d90e04784cf4763e2661f1ca1e` |
| `confirm/colqwen25_docvqa_selections.npz` | `ef3dff3f9173a594f74a799a3913501d82722d5cda8cdb5d0a5f202da302578d` |
| `confirm/colqwen25_infovqa_selections.npz` | `81544c164fb14f74f80d86cc723dc3bf1dd524fc4f3df259339692e610ff95ab` |
| `confirm/scifact_selections.npz` | `381c4a3e09daefc89e0575135d0cd4d257142e271e1d326e6fb9386450a78935` |

Q3 feature files, read by analysis A6. Paths are under `reports/research/Q3/features/`.

| file | sha256 |
|---|---|
| `colpali_docvqa.npz` | `6415c7ddbc9600ea21d887ab619b7783f6678966dfb517504adf13da8116588b` |
| `colpali_infovqa.npz` | `5babb5469ef1c9ae3fcad67ac4914f093fffa4d1449d5b19fa33a1c9a1cad0c4` |
| `colqwen2_docvqa.npz` | `03ba315d8ee267628ab5265dbb3d403f38c3c6664d7837f7de515a64239e3506` |
| `colqwen2_infovqa.npz` | `1d9a3e6a2ac720640826286f2dd1c498446eb48cc627fadb8919a20bf7fbdde3` |
| `colqwen25_docvqa.npz` | `a5f9d2427f7cf1f032c8dff108503dd1455dbbb77500327521a1d1102f13bdf2` |
| `colqwen25_infovqa.npz` | `b039e5f0aaf0e3ed6cf7874cf8e0961271c764c5d39f127092ea6600e9c3462e` |

Vector files, read by X3. They lie outside git; paths are under the data directory.

| dataset | file | sha256 |
|---|---|---|
| `colpali_docvqa` | `cache/colpali_docvqa_test_subsampled.npz` | `2ff460f4283916a1c50a16bd510b181eba3ebfc2b10ca20611cb2ad9f3dfd5f3` |
| | `cache/colpali_docvqa_test_subsampled.queries.npz` | `722a67fc5017be9a3773e09c0a1af0f04f2aea2c59f8999eba2de17df7c6fbf6` |
| | `vidore_docvqa_test_subsampled/queries.json` | `4734e6e46c44af9e98ce26c66a7b19bd16ac8a82f526d5fbd56f2ca05d382708` |
| `colpali_infovqa` | `cache/colpali_infovqa_test_subsampled.npz` | `51453e93172cf14dc60994086f05af222e9fd0bd32a5b888060231e3a482ba95` |
| | `cache/colpali_infovqa_test_subsampled.queries.npz` | `b99796531e7eb79d1eaa9806e0448ad65703ce39ca903c3e95613aa5ccce1e46` |
| | `vidore_infovqa/queries.json` | `f5a64eacf890c407fc4632319f4317704454ad971c2637bf2b88b22870fa68af` |
| `colqwen2_docvqa` | `vectors/colqwen2-v1.0-merged_docvqa_test_subsampled_docs.npz` | `01412095103c4029487652887ab6668a15c0cc29a811c3e108dd18fb566b4da0` |
| | `vectors/colqwen2-v1.0-merged_docvqa_test_subsampled_queries.npz` | `e4e9e5798658d6116ea27b9b50fd872ef74376e9914abf0ac6da88c760b772b7` |
| | `vectors/colqwen2-v1.0-merged_docvqa_test_subsampled_qrels.json` | `5dcd8035448318a9e15716f08c10d958c1332853f57926585690cd8ee1ee7ab6` |
| `colqwen2_infovqa` | `vectors/colqwen2-v1.0-merged_infovqa_test_subsampled_docs.npz` | `c4182f57404173341dafe78eef02d8ed3ebc967f1471880ba212b7676d9b679f` |
| | `vectors/colqwen2-v1.0-merged_infovqa_test_subsampled_queries.npz` | `53504bb742a959546fd9fb35b53ed917eb092a43a00f914abc0e22e9750aa973` |
| | `vectors/colqwen2-v1.0-merged_infovqa_test_subsampled_qrels.json` | `9be499c909f3f618cb5fac43202bf0a10b734d3659313d96a3b4bdee0a714229` |
| `colqwen25_docvqa` | `vectors/colqwen2.5-v0.2_docvqa_test_subsampled_docs.npz` | `8b34b9eeabdfa2e5445e5d75b6ff03f0b0a0e0a61247c2f3327e6f4e44e29eed` |
| | `vectors/colqwen2.5-v0.2_docvqa_test_subsampled_queries.npz` | `1d46db6d1c96a06a90f846cf3ac06079492c94cebe5b4755313f95d7224e4d0f` |
| | `vectors/colqwen2.5-v0.2_docvqa_test_subsampled_qrels.json` | `5dcd8035448318a9e15716f08c10d958c1332853f57926585690cd8ee1ee7ab6` |
| `colqwen25_infovqa` | `vectors/colqwen2.5-v0.2_infovqa_test_subsampled_docs.npz` | `d4fe391654371fddb26c574d0862f364559d5e8fe1fa32afaf4e49ce192b1324` |
| | `vectors/colqwen2.5-v0.2_infovqa_test_subsampled_queries.npz` | `15e05d4b043a0c14744eeaa7fee97c2bf901a937a0eb60d083ad793ff93729da` |
| | `vectors/colqwen2.5-v0.2_infovqa_test_subsampled_qrels.json` | `9be499c909f3f618cb5fac43202bf0a10b734d3659313d96a3b4bdee0a714229` |
| `scifact` | `vectors/scifact_answerai-colbert-small-v1_docs.npz` | `e29c127bbc5ed500f7b6988b4f13b4b96e45e35915ce438d509a384c340c5864` |
| | `vectors/scifact_answerai-colbert-small-v1_queries.npz` | `ede243a3d79c9df49948357b6ad20b9693edfeef5bac07a398c68e828b00c53d` |
| | `vectors/scifact_qrels_test.json` | `7b35922b4c7fe32bfb0c0c4d0e9ac81ae60712a9a4caa43af2b100503c827a4a` |
