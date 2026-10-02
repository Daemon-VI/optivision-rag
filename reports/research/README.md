# Research experiment registry

Plan and write-ups: `docs/RESEARCH_INVESTIGATION.md`. Values are labelled
MEASURED / EXTERNAL / PROJECTED / INFERENCE. Large artifacts stay outside git.

| id | question | hypothesis (stated before running) | status | data | seeds | commit | result | limitations |
|---|---|---|---|---|---|---|---|---|
| E0 | all | — (audit and plan) | done 2026-10-02 | existing reports | — | this branch | research matrix, plan | no experiment run |
| E1.0 | Q1, Q2 | per-query matrices reproduce the committed s7b choices exactly | planned | ColPali, ColQwen2 (DocVQA, InfoVQA) | s7b seeds | — | — | — |
| E1.1 | Q1, Q2 | H1a–H1c (docs §8) | planned | E1.0 matrices + synthetic | 0–1999 | — | — | — |
| E1.2 | Q2 | the frozen method keeps its level on unseen datasets | planned (confirmation) | ColQwen2.5, SciFact | frozen in E1.1 | — | — | — |
