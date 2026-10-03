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

Hypotheses for E1.1, recorded before execution:
- **H1a**: the current point/bound-based selection procedure loses substantial
  compression because selection among many candidates is not explicitly accounted for.
- **H1b**: independent certification can provide a defensible post-selection
  statement but will reduce usable compression because of the split in queries.
- **H1c**: a properly constructed simultaneous/ordered-testing method can recover
  some compression relative to naive independent certification while controlling
  the intended error quantity.
