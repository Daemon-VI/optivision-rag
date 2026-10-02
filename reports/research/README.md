# Research experiment registry

Plan and write-ups: `docs/RESEARCH_INVESTIGATION.md`. Values are labelled
MEASURED / SIMULATED / EXTERNAL / PROJECTED / INFERENCE. Large artifacts stay outside git.

| id | question | hypothesis (stated before running) | status | data | seeds | result | limitations |
|---|---|---|---|---|---|---|---|
| E0 | all | — (audit and plan) | done 2026-10-02 | existing reports | — | research matrix, plan | no experiment run |
| E1.0 | Q1, Q2 | the per-query store reproduces every committed s7b selection exactly | running | ColPali, ColQwen2 (DocVQA, InfoVQA) | s7b split seeds 0–19 | — | — |
| E1.1 | Q1, Q2 | H1a, H1b, H1c (verbatim in `E1/DESIGN.md`) | pre-registered 2026-10-02 | E1.0 store (development datasets) + synthetic | resamples 0–1999; synthetic generator 2026+K | — | — |
| E1.2 | Q2 | the method frozen after E1.1 keeps its stated error level on unseen datasets | not started; needs review of E1.1 | ColQwen2.5, SciFact (held back) | frozen after E1.1 | — | — |

Hypotheses for E1.1, recorded before execution:
- **H1a**: the current point/bound-based selection procedure loses substantial
  compression because selection among many candidates is not explicitly accounted for.
- **H1b**: independent certification can provide a defensible post-selection
  statement but will reduce usable compression because of the split in queries.
- **H1c**: a properly constructed simultaneous/ordered-testing method can recover
  some compression relative to naive independent certification while controlling
  the intended error quantity.
