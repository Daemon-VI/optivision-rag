# Research experiment registry

Plan and write-ups: `docs/RESEARCH_INVESTIGATION.md`. Values are labelled
MEASURED / SIMULATED / EXTERNAL / PROJECTED / INFERENCE. Large artifacts stay outside git.

| id | question | hypothesis (stated before running) | status | data | seeds | result | limitations |
|---|---|---|---|---|---|---|---|
| E0 | all | — (audit and plan) | done 2026-10-02 | existing reports | — | research matrix, plan | no experiment run |
| E1.0 | Q1, Q2 | the per-query store reproduces every committed s7b selection exactly | done 2026-10-02 | ColPali, ColQwen2 (DocVQA, InfoVQA) | s7b split seeds 0–19 | MEASURED: 3,360/3,360 outcomes identical, max diff 0.0 (`E1/e1_0_*.json`) | development datasets only |
| E1.1 | Q1, Q2 | H1a, H1b, H1c (verbatim in `E1/DESIGN.md`) | done 2026-10-02 (all 30 synthetic cells) | E1.0 store (development datasets) + synthetic | resamples 0–1999; synthetic generator 2026+K | `E1/RESULTS.md`: valid ordered testing (C) holds its δ, deploys only at T = 0.95 (and 0.97 on InfoVQA) with ≤ 225 queries; current rule misses up to 26% at n = 50; H1a causal part rejected, H1b and H1c supported | one pool per dataset; i.i.d. with replacement |
| E1.2 | Q2 | the method frozen after E1.1 (C, δ = 0.05) keeps its stated error level on unseen datasets | frozen 2026-10-02 (`E1/E1_2_FREEZE.md`), running | ColQwen2.5, SciFact (held back) | resamples 0–1999 | — | — |

Hypotheses for E1.1, recorded before execution:
- **H1a**: the current point/bound-based selection procedure loses substantial
  compression because selection among many candidates is not explicitly accounted for.
- **H1b**: independent certification can provide a defensible post-selection
  statement but will reduce usable compression because of the split in queries.
- **H1c**: a properly constructed simultaneous/ordered-testing method can recover
  some compression relative to naive independent certification while controlling
  the intended error quantity.
