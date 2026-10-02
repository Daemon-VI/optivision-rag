# Research investigation after v0.3.0

Branch `research-investigation`, from `main` at `4d52060` (v0.3.0). Nothing in this
investigation changes `main`, the package defaults or any release.

Labels used in every table:
- **MEASURED**: produced in this repository; the artifact is named.
- **EXTERNAL**: reported elsewhere, not reproduced here.
- **PROJECTED**: an estimate (compute cost, scale). It is never a result.
- **INFERENCE**: our reading of measured evidence, not itself measured.

Status: Phase 0 done. **E1.0 + E1.1 done and awaiting review** (2026-10-02). Nothing
else has been started.

### E1 summary (details: `reports/research/E1/RESULTS.md`; design: `E1/DESIGN.md`)

- **E1.0 (MEASURED).** The per-query store reproduces all 3,360 committed selection
  outcomes exactly.
- **E1.1 (SIMULATED resampling of MEASURED per-query results).** Family-ordered
  fixed-sequence testing with exact betting p-values (method C) gives a valid,
  finite-sample statement for the *selected* configuration: P(deploy ∧ R_Q < T) ≤ δ
  under i.i.d. queries and a fixed corpus. It held in every check.
- With ≤ 225 calibration queries, C deploys only at T = 0.95, and at 0.97 on
  InfoVQA. There it matches the current rule's compression on InfoVQA (72.7x,
  63.0x).
- **No valid method can certify T = 0.99 with ≤ 225 queries** (a lower bound that
  applies to every distribution-free test).
- The current rule (A) has no valid post-selection statement. It misses the
  finite-pool target in up to 26% of resamples at n = 50 (real data), and up to 88%
  in synthetic near-miss configurations.


---

## 1. What exists (MEASURED unless marked)

| asset | where | notes |
|---|---|---|
| ColPali-v1.3 vectors, ViDoRe V1 DocVQA / InfoVQA (500 pages, 451 / 494 queries) | `optivision-rag-v2/data/cache/colpali_*` (outside git) | 128-d, CPU-usable |
| ColQwen2-v1.0 and ColQwen2.5-v0.2 vectors, same splits and labels | `optivision-rag-v2/data/vectors/colqwen*` (outside git) | 128-d, about 190 MB per split |
| answerai-colbert-small-v1 on BEIR SciFact (5,183 docs, 300 queries) | `optivision-rag-v2/data/vectors/scifact_*` | 96-d, 1.22 M vectors |
| Nemotron ColEmbed 4B vectors (2,560-d) | **not kept** | only aggregates exist (R12); re-encoding needs about 26 min per split on 2 T4s, plus re-analysis |
| out-of-sample selection results, 20 half/half splits, 6 rules x 3 targets | `reports/universal/selection_rules*/` (R7b, R11, R12) | stores each split's choice and held-out retention, **not** the per-query x per-candidate matrix |
| single-configuration coverage of the bootstrap bound; K-candidate worst case | `reports/universal/audit/bound_audit.json` (R10) | resampling from the 451 DocVQA queries; synthetic K = 1/10/40 |
| fixed-configuration grids and in-sample frontiers | `reports/universal/frontier/`, `wide/`, `merge/`, `quantize_project/` | token x codec for ColPali / ColQwen / SciFact; full A–G factorial only for the 2,560-d model |
| corpus size 100 → 500 pages (random distractors, fixed queries) | `reports/universal/audit/corpus_size*`, `colqwen_corpus_size*` | lossier configurations lose 0–3 points |
| geometry (nearest-neighbour cosine, corpus-mean norm, query–mean cosine) | `reports/universal/geometry/` | ColPali, ColQwen, SciFact; no Nemotron |
| latency | R9 (laptop CPU, i7-1165G7), R12 (T4 GPU) | exact scan only; binary is decoded to ±1 floats before the product |

Hardware available: the laptop above (4 cores, CPU only) and Kaggle 2 x T4 sessions
on the user's accounts. Downloads from Kaggle are slow (as low as about 10 kB/s was
observed), so large vectors must be analysed where they are produced.

Two facts from the code that shape the plan:
- `BinaryQuantizer.decode` returns ±1 signs. For a float query q and a sign code s,
  `q·s = 2·Σ_{i: s_i=+1} q_i − Σ_i q_i`. Asymmetric binary scoring can therefore be
  computed **exactly** from packed bits (a byte lookup table per query token)
  without decoding. Symmetric Hamming scoring (query also binarised) is a
  different score, and its quality must be measured.
- Retention is a ratio, `R = Σ_i c_i / Σ_i b_i` (compressed and float nDCG@5 per
  query). For a fixed target T, `R ≥ T` holds exactly when `E[c_i − T·b_i] ≥ 0`, and
  `c_i − T·b_i` lies in `[−T, 1]`. The target condition is thus a statement about
  the mean of a **bounded** per-query variable, which is what finite-sample,
  distribution-free bounds need (Hoeffding, Hoeffding–Bentkus, betting-style
  bounds). This is INFERENCE from the definitions; Phase 1 must verify it in code.

An honesty point for Q7 and Q10: Ward token pooling is Clavie et al. (2024).
OptiVision's merge stage *is* that published method. Our contribution, if any,
lies in the composition, the calibration and the evaluation, not in the merge.

---

## 2. Research matrix (Q1–Q10)

Compute is **PROJECTED** throughout.

### Q1 · A less conservative optimizer that still meets the target

| | |
|---|---|
| Existing evidence | R7, R7b, R11, R12: the 0.999 bootstrap bound met targets in 355/360 (128-d) and 119/120 (2,560-d) held-out choices. The point rule met 10–100% per cell; 0.975/0.99 bounds sit in between. On DocVQA the default chose 2–32x where the in-sample frontier shows 16–63x at ≥98%. |
| Answered | Point estimates are unreliable. The 0.999 bound is empirically safe on these pools. Bonferroni at 0.975/K gives almost the same choices as 0.999. |
| Not answered | How much compression is lost to the bound's width versus to true quality loss (no decomposition exists). Whether a valid method can choose more. "Met x/20" mixes a noisy held-out point estimate (about 225 queries) with the truth, so it cannot measure coverage near 95%. |
| Experiment | Same harness as Q2. Compare the compression each **valid** method selects at equal nominal risk. Decompose the gap to the oracle (the configuration that is truly feasible on the full pool) into width versus true loss. |
| Data / models | Per-query x candidate matrices: ColPali, ColQwen2, ColQwen2.5 (DocVQA, InfoVQA), SciFact |
| Metrics | median selected compression, oracle gap, coverage (Q2), share of "no feasible configuration" |
| Compute | shared with Q2 |
| Confounders | the pool is the population (finite, one corpus); candidate correlation; a monotone-walk stop rule |
| Value | High jointly with Q2; on its own it is a tuning question |

### Q2 · A valid statement for the selected configuration

| | |
|---|---|
| Existing evidence | R10: the bound covers one configuration near nominal from about 100 queries. A synthetic worst case shows the selected configuration can fail with up to 34–55% probability at K = 40, and Bonferroni does not fix it, because the bootstrap under-covers when losses are rare and large. |
| Answered | The current statement is per configuration and approximate (bootstrap/CLT). It is not a post-selection guarantee. |
| Not answered | Whether any method gives a valid coverage statement for the **selected** configuration under stated assumptions, and at what cost in compression and queries. |
| Experiment | (a) **Population-resampling coverage study** on real matrices: truth = full-pool retention, draw n ∈ {50, 100, 225} calibration queries, run each method, record whether the selected configuration's true retention is ≥ T. (b) Synthetic stress tests (rare large losses, many near-threshold candidates, K up to 82). Methods: **A** the current rule; **B** select on S1, certify the one chosen on an independent S2; **C** learn-then-test style fixed-sequence testing along the compression order, with finite-sample p-values for `E[c − T·b] ≥ 0` (Hoeffding–Bentkus or betting); **D** Bonferroni with the same exact bounds, as a reference. |
| Data / models | development: ColPali + ColQwen2 (DocVQA, InfoVQA); **confirmation, untouched until the method is frozen**: ColQwen2.5 + SciFact |
| Metrics | coverage P(true R_sel ≥ T) with Monte-Carlo interval, failure rate, selected compression, abstention rate, queries needed (n_select, n_certify), sensitivity to K ∈ {10, 40, 82} |
| Compute | CPU: about 5–6 h once to build matrices (the s7b scoring cost), then minutes to hours of resampling. No GPU. |
| Confounders | i.i.d. queries are an assumption (one pool, near-duplicate queries); the corpus is fixed (R10 shows corpus size changes retention, so the guarantee is per corpus); fitted stages (PCA, centring) use corpus vectors only, not calibration queries, which must be checked |
| Value | **Highest.** A correct, finite-sample statement about the selected compression of a retrieval index is a real contribution and fixes the main caveat in v0.3.0 |

### Q3 · Why DocVQA and InfoVQA behave differently

| | |
|---|---|
| Existing evidence | Float nDCG@5 DocVQA 0.58–0.66 against InfoVQA 0.85–0.93 (all four image encoders). Geometry is similar (nearest-neighbour cosine 0.93 / 0.93 on ColPali, 0.89 / 0.87 on ColQwen2). At 0.95 the default chose 12–32x (DocVQA) against 73–355x (InfoVQA). Fixed lossy pipelines lose more on DocVQA (ColPali Ward¼ + binary 95.4 against 97.5). |
| Answered | That the behaviour differs, consistently across encoders. That vector geometry alone does not separate the two (INFERENCE from similar NN cosines). |
| Not answered | Whether DocVQA is **less compressible** (lower true retention), **harder to certify** (higher per-query variance, so wider bounds), or both, and which page or query properties drive it. |
| Experiment | Pre-registered hypotheses: H3a baseline difficulty (rank of the relevant page), H3b score margin (relevant page against best non-relevant), H3c within-corpus near-duplicates, H3d per-query variance of `c − T·b`, H3e token redundancy and vectors per page, H3f query length. Measure each per query; decompose the selection gap into true loss and certification width; **intervention**: difficulty-matched subsamples (match the baseline-rank or margin distribution across datasets) and re-run selection. If matching removes the gap, the factor is supported; if two factors are collinear, report that they cannot be separated. |
| Data / models | ColPali, ColQwen2, ColQwen2.5 vectors (no Nemotron vectors) |
| Metrics | per-query rank, margin, retention distribution, bound width, selected compression after matching |
| Compute | CPU, about 2–4 h after the Q2 matrices exist |
| Confounders | matching shrinks n (wider bounds); DocVQA questions can be ambiguous across pages (label noise) |
| Value | Medium-high: it turns "it depends on the data" into a measurable predictor and explains Q1's conservatism |

### Q4 · Which compression axis drives the frontier

| | |
|---|---|
| Existing evidence | 2,560-d: the full A–G factorial with scan times (R12, in-sample frontier). 128-d: token x codec frontiers (R5b, ColPali/ColQwen/SciFact), dimension at 128-d (R4, ColPali). |
| Answered | At 2,560-d, width (PCA) is the most storage-efficient axis up to 8–63x and token count drives scan time. At 128-d, merging + codec dominates (R5b). |
| Not answered | The A–G factorial at 128-d (where projection has little headroom), **held-out** frontiers instead of in-sample ones, and separate RAM, disk, latency and compression-time objectives on one consistent latency protocol. |
| Experiment | A–G grid on ColQwen2.5 and ColPali (128-d, CPU); frontiers selected on one query half and reported on the other; one frontier per objective (disk, RAM tier, scan time, compression time). R12 reused as is for 2,560-d. Encoder time is independent of the compression choice, so the "encoding cost" objective is compression time. |
| Compute | CPU, about 4–6 h |
| Confounders | CPU latency is noisy (laptop); R12 latency was GPU, so not comparable across width |
| Value | Medium: it largely consolidates; the held-out frontier is the new part |

### Q5 + Q6 · Search latency and index-friendly codes

| | |
|---|---|
| Existing evidence | Binary is not faster (decoded to floats). Merging cuts scan time roughly in proportion to vectors. The two-tier rescoring is an unoptimised Python loop (about 350 ms per query at 2,560-d). |
| Answered | Scan time follows vector count. Decoding codes removes any codec speed benefit. |
| Not answered | End-to-end latency with codes scored natively, and with candidate generation plus exact rescoring. |
| Experiment | Step 1 is maths only: classify each approach as **exact** (same scores: asymmetric binary via LUT, integer-domain int8 with float queries if accumulation is exact) or **approximate** (symmetric Hamming, quantised queries, ANN or centroid candidate generation, MUVERA-style fixed-dimensional encodings). Step 2: prototype at most three (asymmetric packed-binary MaxSim; token-level ANN candidates + exact rerank; one two-stage binary → int8 path) and measure end-to-end latency, throughput, RAM, disk, candidate count, rerank cost and nDCG@5. |
| Compute | CPU, engineering-heavy (days); measurement hours. `faiss-cpu` would be a research-only dependency. |
| Confounders | Python overhead masks kernels; numpy 2.x `bitwise_count` versus compiled code; laptop thermal throttling (repeat with medians) |
| Value | Medium for research (exact native scoring is known in principle); high for usefulness. Mostly engineering. |

### Q7 · Comparison with strong existing methods

| | |
|---|---|
| Existing evidence | EXTERNAL list in `docs/AUDIT-2026-09-27.md` §4; none reproduced. Ward pooling is itself Clavie et al. 2024. |
| Answered | Nothing, under a controlled protocol. |
| Not answered | Everything. |
| Experiment | Candidates to vet (exact method, source, training, licence, compatibility): ColBERTv2/PLAID residual compression (k-means centroids + 1–2-bit residuals; training-free given the encoder); product quantisation (FAISS PQ) per token; MUVERA fixed-dimensional encodings + rerank; colpali-engine's hierarchical token pooler (should match our Ward exactly, which is an equivalence check, not a competitor); binary + rerank as in Vespa/Qdrant (equals our two-tier). Light-ColPali fine-tuned, MarginMerge, AnchorFold and anything else needing training stays **EXTERNAL**. Each baseline uses the same vectors, queries, labels, metric, byte accounting and latency harness, labelled "our reimplementation" where it is one. |
| Compute | CPU; implementation days; runs hours |
| Confounders | reimplementation fidelity; tuning effort must be equal (same budget for baselines as for ours) |
| Value | High for publication credibility; the fairness risk is high |

### Q8 · Corpus scale beyond 500 pages

| | |
|---|---|
| Existing evidence | 100 → 500 pages with random distractors: lossy configurations lose 0–3 points (R10, R11). SciFact is 5,183 docs but text. |
| Answered | Retention falls with corpus size at 100–500 pages for lossy configurations. |
| Not answered | 1k–10k pages for images; whether a configuration selected at 500 pages still meets the target at 5k–10k; latency, memory and selection stability with growth. |
| Experiment | Staged distractor scaling with the **fixed** 451 DocVQA queries and labels: 500 → 1,000 → 2,500 (then decide on 5,000 / 10,000). Distractors are in-domain DocVQA pages not in the subsample. Freeze the configuration chosen at 500, then measure at each scale; separately re-run selection at each scale. Report A quality, B calibration failure, C latency growth and D memory/storage growth separately. One encoder (ColQwen2.5 or ColQwen2), analysis on Kaggle. |
| Compute | GPU (T4): about 1–2.4 s/page to encode (MEASURED in smoke runs; batching may be faster), so about 0.6–1.3 h for +2,000 pages and 3–6.5 h for +9,500. Analysis on GPU: 1–3 h per scale. |
| Confounders | **licence of DocVQA pages (stop condition: check first)**; unlabelled relevant distractors (a generic question may match another page, which deflates the measured retention for both float and compressed); out-of-domain distractors would make the task artificially easy, so they are not used |
| Value | High: it is the biggest gap between the claims and deployment |

### Q9 · What generalises

| | |
|---|---|
| Existing evidence | Five encoders x up to two corpora, with fixed-configuration, frontier, geometry and selection reports for all but Nemotron's per-query data. |
| Answered | Partially, in prose across R8, R11 and R12. |
| Not answered | A single consolidated, labelled matrix with confidence per finding. |
| Experiment | Re-analysis only, no new runs. Findings x {ColPali, ColQwen2, ColQwen2.5, ColEmbed 4B, text ColBERT}, each cell MEASURED / EXTERNAL / INFERENCE / not measured, with interval-based agreement. |
| Compute | under 1 CPU-hour |
| Confounders | only two image corpora (dataset and model effects are partly confounded); one text model |
| Value | Medium; cheap; it frames every later phase |

### Q10 · Which questions are research and which are engineering

INFERENCE, to revisit after Phases 1–2:
- **Research contributions:** Q2 (valid post-selection certification for index
  compression), Q3 (predictors of compressibility versus certifiability), Q8
  (selection stability under corpus growth), and Q7 if controlled.
- **Mostly engineering:** Q5/Q6 (native scoring and ANN are known techniques; the
  value lies in measuring them honestly end to end), Q4 (consolidation).
- **Q1** is research only together with Q2; on its own it is tuning a confidence level.

---

## 3. Ranking

| rank | question | importance | rigour possible | compute | new data/models | story |
|---|---|---|---|---|---|---|
| 1 | Q2 (+Q1) | high | high (maths + resampling with known truth) | CPU | none | central caveat of v0.3.0 |
| 2 | Q3 | medium-high | medium (observational + matching) | CPU | none | explains Q1, informs Q8 |
| 3 | Q9 | medium | high (re-analysis) | trivial | none | frames everything |
| 4 | Q8 | high | medium (label noise) | GPU, hours | new pages (licence check) | deployment gap |
| 5 | Q4 | medium | high | CPU | none | consolidation + held-out frontier |
| 6 | Q5/Q6 | medium | high for exactness, medium for latency | CPU, engineering | optional faiss | usefulness |
| 7 | Q7 | high | medium-low (fidelity risk) | CPU, engineering | none | credibility |

---

## 4. Questions answered together

- **Q1 + Q2**: one coverage harness, one set of per-query x candidate matrices.
- **Q3** reuses the Q2 matrices and its oracle decomposition.
- **Q4 + Q5/Q6**: one latency protocol and one resource accounting.
- **Q8** reuses the frozen selection method from Q2 and the latency harness from Q5.
- **Q9** reads all of the above. Its first version comes early, from existing reports.

---

## 5. Proposed order and compute (PROJECTED)

| step | phase | compute | GPU | gate before the next step |
|---|---|---|---|---|
| 1 | **E1.0** build per-query x candidate matrices (default 40-candidate space) for the 7 cached datasets | CPU, about 5–6 h | no | matrices reproduce the committed s7b split results exactly (same seeds) |
| 2 | **E1.1** coverage study, methods A/B/C/D, development datasets | CPU, hours | no | a method is frozen (written decision) before confirmation |
| 3 | E1.2 confirmation on ColQwen2.5 + SciFact | CPU, minutes | no | report the guarantee statement with its assumptions |
| 4 | E9.0 consolidated generalisation matrix (first pass) | CPU, under 1 h | no | — |
| 5 | E3.x DocVQA versus InfoVQA | CPU, 2–4 h | no | each hypothesis supported, rejected or "not separable" |
| 6 | E4.x A–G factorial at 128-d, held-out frontiers | CPU, 4–6 h | no | — |
| 7 | E5.0 compatibility analysis (maths), then up to three prototypes | CPU, days of engineering | no | exactness tests pass before any latency claim |
| 8 | E8.x scale: licence check, then 1,000 and 2,500 pages | Kaggle T4, about 2–4 h | yes | cost re-estimated before 5,000 / 10,000 |
| 9 | E7.x baselines: vetting table, then fair reimplementations | CPU, engineering | no | fidelity check per baseline or it stays EXTERNAL |

Re-encoding Nemotron (about 1 h of 2 x T4 per split, plus analysis) is deferred.
It runs only if Phases 1–3 produce a hypothesis that depends on width.

---

## 6. Risks and confounders across the programme

- **One query pool per corpus.** Every held-out split comes from the same 451 /
  494 queries; "met x/20" counts are dependent. Phase 1 replaces them with
  resampling against a known truth (the pool), whose limitation is that the truth
  is this pool's distribution, not a deployment's.
- **Fixed corpus.** Any guarantee is per corpus (R10 shows retention moves with
  size), so Q8 must test whether it transfers.
- **Test-data discipline.** ColQwen2.5 and SciFact are confirmation sets for Phase
  1, and must not be looked at while methods are developed.
- **Label noise** (one labelled page per query; DocVQA ambiguity) affects Q3 and Q8.
- **Latency comparability.** A laptop CPU, a T4 GPU and Python overhead; only
  within-protocol comparisons are reported.
- **Reimplementation fidelity** for Q7 baselines.
- **Library behaviour.** Research code lives in `scripts/research/`. Any change to
  `src/optivision` is opt-in and new, never a changed default. A change that
  could alter v0.3.0 behaviour is a stop condition.

---

## 7. Branch and artifact layout

```
research-investigation            (from main 4d52060; never merged without review)
  docs/RESEARCH_INVESTIGATION.md  plan, decisions, results, limitations
  reports/research/README.md      experiment registry (one row per experiment)
  reports/research/e1_selection/  small JSON results, logs, environment records
  reports/research/e3_docvqa_infovqa/ ...
  scripts/research/               experiment scripts (no library defaults changed)
outside git: per-query matrices > 1 MB, vectors, scratch outputs
  (OPTIVISION_OUT, e.g. optivision-rag-v2/data/research/)
```

One commit per experiment stage. Every result records its seeds, the git commit,
numpy/scipy/torch versions, hardware, peak memory and output bytes.

---

## 8. The first experiment after Phase 0: E1.0 + E1.1

**Hypothesis (stated before execution).**
- H1a: on real candidate matrices, the current rule (A) reaches the nominal
  coverage of the target at n = 225 calibration queries but falls below it at
  n = 50.
- H1b: in the synthetic stress test, A under-covers as K grows (as R10 found),
  while B and C hold their nominal level 1 − δ.
- H1c: C selects at least as much compression as B at the same δ, because it uses
  all queries for both selection and certification without splitting.
- Any of these may be false; the result is reported either way.

**Protocol.**
1. E1.0: for ColPali and ColQwen2 (DocVQA, InfoVQA), score the 40 default
   candidates on all queries and store per-query nDCG@5 for the float baseline
   and each candidate (`scripts/research/e1_matrices.py`, outside git). Check that
   replaying `calibrate()` on the committed seeds reproduces
   `reports/universal/selection_rules/*.json` exactly.
2. E1.1: truth = full-pool retention per candidate. For n ∈ {50, 100, 225},
   T ∈ {0.99, 0.97, 0.95}, δ = 0.05 (also the current 0.001 level for method A),
   and 2,000 resamples (seeds 0–1999), run methods A–D and record the selected
   candidate, its true retention, its compression, and abstention.
3. Synthetic stress: K ∈ {10, 40, 82}, candidates 0.5–1 point below T with rare,
   large per-query losses (the R10 generator), same methods.

**Validity argument to check before calling anything a guarantee (method C).**
With `X_i = c_i − T·b_i ∈ [−T, 1]` i.i.d. over queries, a valid one-sided
p-value for H0: `E[X] < 0`, plus fixed-sequence testing in a pre-specified order
(most to least conservative), controls the family-wise error, so
P(selected configuration has true R < T) ≤ δ. The guarantee is over random
calibration sets, for a fixed corpus and fitted pipeline, under i.i.d. queries.
If the code or the simulation contradicts this, the word "guarantee" is not used.

**Outputs.** `reports/research/e1_selection/*.json` (coverage tables, small), the
registry row, and a write-up in this document.
**Cost.** CPU only, about 6–8 h in total (PROJECTED). No GPU.

---

## 9. Experiments not to run (little information for the cost)

- Re-running R7b / R11 / R12 (s7b, s13) to tidy tables: the artifacts already
  answer what they answer.
- More half/half splits on the same pools ("met x/100"): the splits are
  dependent, and the resampling design answers coverage properly.
- The 4,096-d 8B model, ColQwen3, ColNomic or Jina before E9.0 shows a question
  that needs a new model.
- ViDoRe V2/V3 before Q3 and Q8 are understood, and before a licence review.
- Point-estimate rule variants: unreliable in all seven measured datasets.
- Pseudo-query (document-fragment) calibration: settled in R7.
- Speeding up the Python rescoring loop and calling it a latency result.
- Comparing our numbers with paper numbers on other models or datasets as if
  controlled.
- Million-page runs or extrapolations.
- Training-based baselines (fine-tuned Light-ColPali, MarginMerge, AnchorFold):
  they stay EXTERNAL.
- Re-encoding Nemotron for Phase 1 alone.
- GPU latency sweeps of decode-to-float binary (already known not to be faster).
