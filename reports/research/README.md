# Research experiment registry

Plan and write-ups: `docs/RESEARCH_INVESTIGATION.md`. Values are labelled
MEASURED / SIMULATED / EXTERNAL / PROJECTED / INFERENCE. Large artifacts stay outside git.

| id | question | hypothesis (stated before running) | status | data | seeds | result | limitations |
|---|---|---|---|---|---|---|---|
| E0 | all | — (audit and plan) | done 2026-10-02 | existing reports | — | research matrix, plan | no experiment run |
| E1.0 | Q1, Q2 | the per-query store reproduces every committed s7b selection exactly | done 2026-10-02 | ColPali, ColQwen2 (DocVQA, InfoVQA) | s7b split seeds 0–19 | MEASURED: 3,360/3,360 outcomes identical, max diff 0.0 (`E1/e1_0_*.json`) | development datasets only |
| E1.1 | Q1, Q2 | H1a, H1b, H1c (verbatim in `E1/DESIGN.md`) | done 2026-10-02 (all 30 synthetic cells) | E1.0 store (development datasets) + synthetic | resamples 0–1999; synthetic generator 2026+K | `E1/RESULTS.md`: valid ordered testing (C) holds its δ, deploys only at T = 0.95 (and 0.97 on InfoVQA) with ≤ 225 queries; current rule misses up to 26% at n = 50; H1a causal part rejected, H1b and H1c supported | one pool per dataset; i.i.d. with replacement |
| E1.2 | Q2 | the method frozen after E1.1 (C, δ = 0.05) keeps its stated error level on unseen datasets | done 2026-10-02 (frozen first: `E1/E1_2_FREEZE.md`) | ColQwen2.5, SciFact (held back) | resamples 0–1999 | **confirmed**: all 27 cells consistent with δ = 0.05 (max miss 0.05%); stores reproduce 2,520/2,520 committed outcomes; C deploys at T = 0.95 on all three (7.8x, 76.0x, 11.8x) and at 0.97 only on InfoVQA (`E1/RESULTS.md`) | one pool per dataset; i.i.d. with replacement; fixed corpus |
| Q3 | Q3 | H3a–H3d (`Q3/PLAN.md`, `dc9cc90`); intervention pre-stated (`Q3/INTERVENTION.md`) | closed 2026-10-03 (supported, residual uncertainty) | E1 store `b18ae19` (read only) + features from existing vectors (checked equal to the store) | bootstrap seed 0; resamples 0–1999 | `Q3/REPORT.md`: same-pipeline retention differs little (0–12/40 beyond CI); certification SE 2.1–2.6x larger on DocVQA; margin matching closes 61–81% of the SE gap and 61–74% of A's selection gap; pre-stated reading: partial, both contribute (H3d); H3a not supported as main explanation | two corpora; one labelled page; observational matching; rank-matching robustness diverged |
| Q3H | Q3 | within-DocVQA split at DocVQA's median margin (`Q3/SPLIT.md`, `5734903`) | done 2026-10-03 | E1 store `b18ae19` + Q3 features (read only) | bootstrap seed 0; resamples 0–1999 | `Q3/REPORT.md` §18: high half (225 q, top-1 ≈ 1) closes 84–120% of the log SE gap; A/C selection gap to InfoVQA 1.3–4.7x (from 5.4–18.6x); pre-stated reading ambiguous (C: one model A, two C); Q3 closed with residual uncertainty (§19) | high half is a ceiling subset (no gains); observational split of one pool; ColPali residual |
| Q4 | Q4 | H4a–H4e (`Q4/PLAN.md`, `709f421`; clarifications `1153f17`) | done 2026-10-03, awaiting review | 6 × 128-d datasets re-scored from existing vectors (74 configurations each, checked equal to the E1 store and R5b/R11); ColEmbed 4B = R12 rows reused | bootstrap seed 0; split_queries seed 0; latency 5 rounds | `Q4/REPORT.md`: count drives latency (H4a supported in substance, formally mixed); H4b not supported; H4c, H4d partial; H4e not supported; A+C at 128-d against B+C at 2,560-d | laptop CPU timings; ColEmbed without per-query data or CPU latency; two corpora; library cannot persist int4c, 2-bit or the PCA basis |
| Q5/Q6 | Q5/Q6 | H5a–H5d (`Q5Q6/PLAN.md`, `a8db31b`; reviewed amendment `3a1e3dc`) | closed 2026-10-03, awaiting review | ColQwen2 DocVQA, existing vectors; Ward 1/4 > binary hot, int8 cold, 50 candidates | bootstrap seed 0; 1 warm-up + 5 timed rounds | `Q5Q6/REPORT.md`: strict exact criterion failed (float32 near-ties; `CORRECTNESS.md`), amended criterion passed; direct int8 24–34x slower; LUT binary 28–30x slower; native two-tier 9.7–11.7x slower; H5a–c rejected, H5d supported; batched decode-first rescoring 5.3x faster and exact | numpy/torch only (no compiled kernels); machine-wide drift in rounds 1–4; batch timing only |
| ENG-1 | Q5/Q6 follow-up (engineering, not research) | batched `ExactIndex.rescore` (`reports/engineering/batched_rescore/PLAN.md`, `12628a7`) | done 2026-10-03; keep (rule A) | six Q4 datasets, existing vectors | 1 warm-up + 5 timed rounds | `reports/engineering/batched_rescore/RESULTS.md`: 60/60 cells ranking-identical, scores bitwise-equal; rescore 5.5–9.3x, end-to-end 2.9–3.7x faster | one laptop CPU (two speed states, ratios stable); batch of all queries; bit-equality requires products with enough rows |
| Q7 | Q7 | comparability tiers fixed before reading (`Q7/PLAN.md`, `43123aa`) | done 2026-10-03 (literature only), awaiting review | primary papers, repositories and one labelled blog; no experiments | – | `Q7/REPORT.md`: one Tier A source (SAP, ViDoRe V1, same checkpoints); OptiVision's Ward 1/10 retains 4–10 points more than SAP's K-means baseline (unexplained; E7a proposed); the rest Tier B or C; HPC-ColPali excluded (self-declared estimates); no leaderboard | external tables partly single-extraction; SAP query counts and toolkit unknown |
| E7a | Q7 follow-up | frozen attribution plan (`Q7/E7a/PLAN.md`, `47e98fe`) | done 2026-10-03, awaiting review | six Q4 datasets, existing vectors, CPU | K-means and random seeds {0,1,2}; bootstrap seed 0 | `Q7/E7a/REPORT.md`: clustering (Ward over K-means, +0.7 to +3.7) and rescaling (+0.1 to +4.6) both matter; merging beats random (51/54); K-means/plain closes about 30–55% of the SAP gap at γ = 0.10; random control matches SAP on ColPali (≤1.6) but not ColQwen2 (up to 13.7); recommendation E7c | not a SAP reproduction; SAP values HTML-transcribed; seed spread up to 5–6 points |
| E7c | Q7 follow-up | frozen alignment plan (`Q7/E7c/PLAN.md`, `308490e`) | done 2026-10-03, awaiting review | ColPali and ColQwen2 × DocVQA and InfoVQA; official raw, MTEB and BEIR qrels; pytrec_eval | E7a seeds {0,1,2} | `Q7/E7c/REPORT.md`: only difference is the repeated-question label rule; metric engine identical; alignment ≤ 0.5 points on baselines, ≤ 0.65 on retention; ColQwen2 random-control gap to SAP unchanged (median 4.8 → 5.0, maximum 13.7 → 14.2): unresolved, upstream of evaluation; E7b not justified; recommend stop Q7 | encoding side not alignable without GPU; SAP evaluation details unpublished; SAP values HTML-transcribed |
| Q8 | Q8 | frozen scale model (`Q8/PLAN.md`, `3a39ba3`; validity amendment `ae67a46`) | done 2026-10-04, awaiting review | Q4/Q5Q6/R12 primitives + validation timing on ColQwen2 DocVQA tiled to 4,000 duplicated pages (timing only) | 1 warm-up + 5 rounds; first run invalidated (standby), rerun with keep-awake, no power events | `Q8/REPORT.md`: storage and RAM linear (exact projections); scan linear 500–4,000 pages (R² ≥ 0.999); old rescoring flat, batched gain falls 7.4x → 2.8x with overlap; two-tier hot scan dominates beyond about 1,300 pages; 1M pages = storage arithmetic only, about 8.6 s/query exhaustive (PROJECTED) | linearity only to 4,000 duplicated pages; batch-amortized timing; no paging model; no quality at scale; ColEmbed storage only |
| Q9 | Q9 | re-analysis of frozen Q1–Q8 / R-series evidence; experiment criteria in the brief | done 2026-10-04, awaiting review | no new data or experiment | – | `Q9/REPORT.md`: supported with scope limitations; core findings replicate across 4 visual + 1 text encoder and 3 corpora; dimension reduction and low-bit codecs model-specific; certifiable compression dataset-specific; codec speedups, untested models or corpora, quality at scale unsupported | one pool per corpus; text model/data confounded; ColEmbed aggregates only; coarse operating-point grids |
| Q10 | Q10 | synthesis of Q1–Q9, ENG-1 and R-series; classification framework in the report | done 2026-10-04 | no new data or experiment | – | `Q10/REPORT.md`: mixed research + engineering; findings established, novelty not established; method C a validated application (conditional guarantee); framework and batched rescoring are engineering; codec speed, native kernels and persistence are implementation-specific; do-not-claim list | Q7 literature limited to compression/engines; one laptop; tested models and corpora only |
| X1 | next phase: label-free reference agreement (SQ1) | H4a–H4c (`NEXT_PHASE_PREREGISTRATION.md` §7, §10; commit `73085ab637679f30fffece6632eb7b9e92adff79`) | pre-registered 2026-10-04; not started | the seven E1 stores (read only): labels and baseline@1 arrays | resamples 0–1999; block 2000–3999 only by the registered rules | – (not run) | development encoders only; one agreement measure (placement of the reference's top document); agreement is not retrieval quality; thresholds set knowing the labelled outcomes |
| X2 | next phase: query budget and certification (SQ2) | H3, and the agreement arm of H4 (`NEXT_PHASE_PREREGISTRATION.md` §7, §11; `73085ab`) | pre-registered 2026-10-04; not started | the seven E1 stores (read only); registered budgets 50–3600 | resamples 0–1999; block 2000–3999 only by the registered rules | – (not run) | decided at the pool-capped budget (450; 300 on SciFact); 900, 1800 and 3600 exceed every pool, are extrapolated simulations and decide no primary hypothesis |
| X3 | next phase: coordinate and symmetry test (SQ3) | H5 (`NEXT_PHASE_PREREGISTRATION.md` §7, §12; `73085ab`) | pre-registered 2026-10-04; not started | existing vectors: ColPali DocVQA, ColQwen2 InfoVQA, ColQwen2.5 DocVQA, SciFact (extension: the other three stores) | transformation seeds 0, 1, 2, 3; bootstrap seed 0; resamples 0–1999 | – (not run) | four registered maps only; 96-d and 128-d encoders; says nothing about model-blindness |
| XH | next phase: held-out encoders (SQ4) | H1, H2: definitions and decision rules only (`NEXT_PHASE_PREREGISTRATION.md` §7, §13; `73085ab`) | rules pre-registered 2026-10-04; experiment not yet registered (needs the `heldout-freeze` addendum); not started | closed pool of nine candidate encoders; none encoded | resamples 0–1999 | – (not run) | registered subset, corpora, pools, revisions and compute plan are fixed only by the addendum; Tier 1 pools are not unseen |

Hypotheses for E1.1, recorded before execution:
- **H1a**: the current point/bound-based selection procedure loses substantial
  compression because selection among many candidates is not explicitly accounted for.
- **H1b**: independent certification can provide a defensible post-selection
  statement but will reduce usable compression because of the split in queries.
- **H1c**: a properly constructed simultaneous/ordered-testing method can recover
  some compression relative to naive independent certification while controlling
  the intended error quantity.

Next-phase preregistration (X1, X2, X3; rules for XH), recorded before execution:
- **Document**: `NEXT_PHASE_PREREGISTRATION.md`, added by commit
  `73085ab637679f30fffece6632eb7b9e92adff79` (2026-10-04, parent `7d87e33`). That commit
  contains only that document (blob `fa9deb73fdf380f06141d89cfaf145444f883506`).
- **Before execution**: no X1, X2 or X3 computation was run before that commit, and none
  has been run before this registry entry.
- **Status**: with this entry the three conditions in §2 of the document hold, so it is
  PREREGISTERED. Its text is frozen and was not edited: the header keeps the wording it
  was committed with, and any change goes to its Appendix C as a dated amendment.
- **Scope**: it registers X1, X2 and X3, and the rules for XH. XH itself is registered
  only by the later `heldout-freeze` addendum.

Amendment 1 to the next-phase preregistration (X1, X2, X3), recorded before execution:
- **Document**: "Amendment 1, 2026-10-04" in Appendix C of
  `NEXT_PHASE_PREREGISTRATION.md`, added by commit
  `130286dd06c763f2bd11a66d706d98dbccf053e6` (2026-10-04, parent `95b1cc3`). The amended
  document is blob `445d94cdb1dda8671a7d4c4ba3bbb7f895f15700`, the text the X1–X3 scripts
  require at the commit that runs. The text above Appendix C is byte for byte the text
  registered by commit `73085ab637679f30fffece6632eb7b9e92adff79` (blob
  `fa9deb73fdf380f06141d89cfaf145444f883506`): the amendment replaces the line `None.`
  under that heading and changes nothing above it. The same commit brings the harness into
  line with it (`x_common.py`, `x1_agreement.py`, `x3_symmetry.py`,
  `tests/test_x_harness.py`, `X_HARNESS.md`).
- **Before execution**: the amendment was committed before any X1, X2 or X3 execution. No
  X1, X2 or X3 computation was run before that commit, and none has been run before this
  registry entry. No per-query outcome value has been read in this phase, for the
  preregistration, the harness or the amendment: only file hashes, array names, shapes and
  types, candidate metadata and synthetic test data.
- **Scope**: it records the operational clarifications OP1–OP7, accepted by the project
  owner as implemented; the alternative N10 count (D1), always reported and diagnostic
  only; the failure handling F1–F5, where F1 is marked as an amendment and replaces the
  candidate-failure treatment of §16.7 for X3 only; the operational requirements O1–O4 on
  outputs and reporting; the reporting conventions R1–R13; and the sha256 hashes of 34
  auxiliary input files (7 E1 selection files, 6 Q3 feature files, 21 vector files), as
  integrity checks only. It alters no hypothesis, dataset, candidate space, threshold,
  Method C setting, query-budget definition, seed or primary decision criterion.
- **Status**: under §2 of the document it is an amendment and not a deviation, because it
  was committed before the analyses it affects. X1, X2 and X3 are not started, and this
  entry reports no result.
- **Repository state at this entry**: nothing of this phase has been pushed
  (`origin/research-investigation` is at `7d87e33`). `main` and `origin/main` are at
  `4d520603469e179135ffd9df922b02d4504b4add`, untouched.
