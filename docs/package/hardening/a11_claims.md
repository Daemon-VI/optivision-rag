# A11 research-claim audit

## 1. Scope and what was executed

Worktree `ov-harden/a11`, branch `harden/a11`, commit 19ece53. Scope as in the task (89 files). I grepped
(case-insensitive, word variants) for the full term list: about 1,100 raw hits, each read in context. Hits that
are identifiers or ordinary words (`certify()` as a function name, "first stage", "validated" meaning "input
checked", `allow_device_transfer`, "best-ranked", "tampering" in the tamper-test sense) are grouped in the claim
table rather than listed one by one. Every hit that is a claim, a research number or a performance word is
covered by a row.

Numbers were checked against `reports/research/X_RESULTS.md`, `X1|X2|X3/*_summary.json`, `X3/x3_scifact.json`,
`X3/x3_colpali_infovqa.json`, `Q5Q6/REPORT.md`, `reports/engineering/batched_rescore/RESULTS.md`,
`E1/RESULTS.md`, and `docs/UNIVERSAL.md` (R11/R12 counts only). No LFS pointer blocked a needed file.
Nothing frozen was edited.

Executed: grep passes; a `python -I` script reading the X2/X3 JSON; pytest on
`tests/test_docs_examples.py tests/test_api_certify.py tests/test_core_selection_rules.py` (186 passed);
`ruff check` on the touched Python files (clean).

## 2. Findings

| ID | Sev | File:line | Problem | Evidence | Fix | Status |
|---|---|---|---|---|---|---|
| A11-01 | MEDIUM | `README.md:17`, `PACKAGE.md:3` | "model-agnostic interface" is the first thing a reader (and the PyPI page) sees; it is defined 110+ lines later. | Definition at README:157, PACKAGE:77. | Parenthetical after the tagline: one interface for any encoder's `[n, d]` vectors; how well a pipeline compresses depends on the encoder and is measured each time. | FIXED in 19ece53 |
| A11-02 | LOW | `README.md:204` | "the universal layer" (v0.3.0 name for the stages/`optimize` layer) used without saying what it names. | n/a | One parenthetical: the word names the encoder-independent interface, not a compression result for every encoder. | FIXED |
| A11-03 | MEDIUM | `src/optivision/api.py:89` (user-visible abstain note), `core/selection/engine.py:15`, `API_DESIGN.md:145`, `guide/QUICKSTART.md:151,160`, `guide/SELECTION_AND_CERTIFICATION.md:293` | "such looks multiplied the false-selection probability by 3 to 14". STATISTICAL_REVIEW 7.7 reports factors from 1.4 (2 seeds) and 2.4 (5 seeds) up to 14 (100 seeds); list editing 3.5 to 9; target scan about 5. "3 to 14" reads as a general range with a floor of 3. | `STATISTICAL_REVIEW.md` 7.7 table: 0.0120 to 0.0170 / 0.0285 / 0.0760 / 0.1690. | "by up to 14". `tests/test_api_certify.py:156` asserted the old string; updated to `"up to 14"` (same assertion, new wording). | FIXED |
| A11-04 | MEDIUM | `METHOD_C_SPEC.md:334`, `:267` | "No valid method can certify T = 0.99 with <= 225 queries" is literally false (SR-7 / W12: a valid test can reject with probability up to 0.48 at n = 225). The review's replacement wording W12 had not been applied here. | `STATISTICAL_REVIEW.md:292,333`. | Replaced with the W12 form: no valid test can reject a lossless candidate with probability >= 1/2 at T = 0.99, delta = 0.05 with fewer than 229 queries; line 267 softened to "none can be expected to certify". | FIXED |
| A11-05 | LOW | `METHOD_C_SPEC.md:345-346` | "Consistent with the proof ... against the proof": the derivation is the project's own and not externally reviewed (the same file says so at 311 and 681). | `STATISTICAL_REVIEW.md:342`. | "project's derivation (not externally reviewed)". | FIXED |
| A11-06 | LOW | `METHOD_C_SPEC.md:296` | Heading "6. The statistical guarantee" is a bare "guarantee" in the table of contents; the text below is conditional. | No anchors link to it. | Heading now "The statistical statement (a conditional, finite-sample guarantee, not externally reviewed)". Section number unchanged, so "section 6.x" references stay valid. | FIXED |
| A11-07 | LOW | `pyproject.toml:8`, `npm/package.json:4` | "Extreme Token Compression", carried from v0.3.0. Measured ratios range from 3.9x to about 120x by encoder and corpus; "extreme" is an unmeasured adjective. | README tables. | Not edited: released metadata, and renaming is a project decision. | PROPOSED |
| A11-08 | LOW | `METHOD_C_SPEC.md:345` | "27 of 27 held-back cells": "held-back" is not in the source; E1/RESULTS says "all 27 cells ... consistent with the claim". | `E1/RESULTS.md:251-255`. | Left; it is a citation in a spec and the numbers match. | NOT A DEFECT (noted) |
| A11-09 | NO ACTION | FINAL_RESEARCH_BOUNDARY, BASIS_DEPENDENCE, LIMITATIONS, SELECTION_AND_CERTIFICATION, CONCEPTS, README/PACKAGE research summary | Every research number matches the stored files (section 3). No XH or held-out-encoder transfer is implied; evidence labels are not mixed. | section 3 | none | NOT A DEFECT |

No CRITICAL or HIGH findings.

### Claim rows by term

Verdicts: S = SUPPORTED, Q = NEEDS QUALIFIER, U = UNSUPPORTED, N = NEGATED-OK.

| file:line | sentence (shortened) | verdict | basis |
|---|---|---|---|
| README:17, PACKAGE:3 | "...behind a model-agnostic interface." | Q, fixed (A11-01) | interface sense only |
| README:130,157; PACKAGE:50,77; guide/README:8,29; CONCEPTS:46,248; BASIS_DEPENDENCE:8; ADAPTER_ARCHITECTURE:154; core/__init__:6; stages/__init__:12 | "A model-agnostic interface is not model-independent compression quality" | N | explicit denial; matches boundary doc sec. 2-3 |
| README:535; guide/ARCHITECTURE:16; UNIVERSAL_PACKAGE_DESIGN:1,101; core/__init__:1; retrieve/__init__:1; codeid:12; traits:553; CHANGELOG "a model-agnostic layer" | "the model-agnostic layer/core" | S | code that branches on no encoder name (api.py:31 says so, relative to an encoder and a date) |
| README:205,485,521,547,548 | "universal layer / universal study" | Q, fixed once at 204 (A11-02) | historical name; directory names kept |
| UNIVERSAL_PACKAGE_DESIGN:11-14; FINAL_RESEARCH_BOUNDARY:35 | "What 'universal' means" | S | defined where first used there; strong form denied |
| UNIVERSAL_PACKAGE_DESIGN:32-33,54 | "universal interface / neither universal nor chosen per model" | S / N | |
| BOUNDARY:51-52,125,145; LIMITATIONS:46-48; CONCEPTS:68; METHOD_C_SPEC:286,338; api.py:31 | "model-blind in what the procedure inspects, not neutral" / "price of model-blindness" | S | preregistration section 15 sense; each says it is not evidence of transfer |
| BASIS_DEPENDENCE:9; BOUNDARY:125 | "model-blind ... while the compression depends on the basis. Both are true" | S | X3 (H5 fails) |
| README:120 | "No distribution-free or model-independent result" | N | |
| README:62,116; PACKAGE:43 | "no guarantee that the target is met" | N | |
| BOUNDARY:5,154 (heading); METHOD_C_SPEC:329,658,659; STATISTICAL_REVIEW | "does not guarantee / not guaranteed / never 'guaranteed retention'" | N | |
| METHOD_C_SPEC:296,313-314,334 | "statistical guarantee ... conditional, finite-sample guarantee" | Q, fixed (A11-06) | 313-314 already carry the no-external-review caveat |
| ARCHITECTURE:95 | "index-size guarantees" (budget mode keeps exactly the top-k fraction) | S | deterministic property of the legacy mask |
| ARCHITECTURE_AUDIT:272 | "guaranteed today only because stages/__init__ imports..." | S | import-order fact |
| traits.py:118,143,713; ARCHITECTURE_AUDIT:292; UNIVERSAL_PACKAGE_DESIGN:225; README:584; BACKEND_ARCHITECTURE:113; RETRIEVAL_AUDIT:279 | `lossless` trait (float32 stored exactly); "not lossless"; "never lossless" for decoded codes | S / N | defined as exactness of the stored representation; no compression claim |
| api.py/rules.py:235, statistics.py:57, METHOD_C_SPEC:6,341,658, UNIVERSAL_PACKAGE_DESIGN:79,385 | "not a new method", "not a statement of novelty" | N | no positive use of "novel", "first to", "state of the art" or "SOTA" in new material |
| README:302 | "best quality per byte - prune + int8: 14.2x at 96.0%" | S | v0.3.0 text, scoped to E1 and to two named operating points |
| README:341 | "E3 is also the best operating point in the project (... 60 pages and 72 queries ...)" | S | scoped and immediately called least representative |
| other "best" (ranked ids "best first", "best of 5" timing, "best / median" columns, "best of 20/100 seeds") | | S | ordinary or measurement label |
| API_DESIGN:214 | "No 'best' configuration" | N | |
| "optimal" | none in scope | n/a | |
| ARCHITECTURE:127,148; BACKEND_ARCHITECTURE:177,231-233; LIMITATIONS:139-150; RETRIEVAL_AUDIT:258-276; CHANGELOG ("5.5-9.3x faster ... 2.9-3.7x faster, on one laptop CPU"); BOUNDARY:7 | "faster / fast / slower / speedup" | S | each carries measurement and scope; numbers match Q5Q6/REPORT.md (23.8-34.1x, 28-30x, 4.4-4.7x, 9.7-11.7x slower) and batched_rescore/RESULTS.md (5.5-9.3x, 2.9-3.7x); RETRIEVAL_AUDIT W1 forbids "faster search from compression" |
| "efficient" | none in new material (README:591 is a paper title) | S | |
| core/retrieve/exact.py:24, types.py:236; BACKEND_ARCHITECTURE:131; METHOD_C_SPEC:341 | "not a scalability claim" | N | |
| ARCHITECTURE:83 | "unit() scales by the 99th percentile" | S | arithmetic |
| README:388; SELECTION A1; METHOD_C_SPEC:675 | "production": "not a production RAG service", "sample of a production log" | N / S | no "production-ready" anywhere |
| ENGINEERING_BENCHMARKS:7 | "the robust quantities are ratios inside one run" | S | scoped to that table |
| CERTIFICATE_SPEC:25; certificate.py:19-24; LIMITATIONS sec. 11; CONCEPTS:253; BOUNDARY:173 | secure / cryptographic / tamper | N | all deny: digest is integrity only; HMAC symmetric, no non-repudiation; forged evidence plus objects undetected without a key |
| CONCEPTS:253; CERTIFICATE_SPEC:378 ("supported, not proven"); state.py:17 ("not an integrity proof"); METHOD_C_SPEC:311,681 | "proof / proven" | N | |
| METHOD_C_SPEC:345-346 | "Consistent with the proof" | Q, fixed (A11-05) | |
| BUILD_AND_PORTABILITY_REPORT:26 | "sdist is proven buildable" | S | means build and `twine check` passed here (ENGINEERING); left |
| README:130-177; guide/*; CONCEPTS:198-211; QUICKSTART; SELECTION; cli_certify.py:104,124-131; api.py; decision.py | "certified / certifies / certificate" | S | every definition says: the rule rejected "retention <= T" at the stated level, claimed only under A1-A6 and C1-C4, only for the library's own rules. README:157-163 and certificate.py:10-14 deny that it says the compressed index keeps retrieval quality or transfers. The CLI headline spells the statement out and adds "agreement ..., not retrieval quality" for the agreement construct |
| BUILD_REPORT:166; collect.py ("must reproduce the fit"); METHOD_C_SPEC:207,683 | "reproducible / reproduces" | N / S | "builds are not reproducible across dates" |
| input-validation uses; collect.py:145, METHOD_C_SPEC:73,682 ("only nDCG@k is validated"); README:116, PACKAGE:41; cli_certify.py:104 | "validated" | S / N | statistical sense carries an INFERENCE tag; others deny validation |
| README:42-56, PACKAGE:20-29; API_DESIGN; MIGRATION:38,55; SELECTION:327 | "held-out" | S | always held-out queries / half (v0.3.0 R7b, optimize); never held-out encoders. README:193-195 and BOUNDARY:140-150 say held-out-encoder transfer (XH) is not run |
| spaces.py:17; LIMITATIONS:35-36; README:193; ADAPTER_ARCHITECTURE:155; CERTIFICATE_SPEC:27; all `allow_device_transfer` | "transfer" | N | |
| "generalis/generaliz", "significant" | none making a claim in new material | n/a | |
| "first" | ordinary ("first stage", "first access", "first look", "first release") | S | |

## 3. Research numbers checked

| Quoted in | Number | Stored value | Label |
|---|---|---|---|
| BOUNDARY:67, README, PACKAGE | H4a: CP1 21/21 pairs, CP2 150 cells | X_RESULTS sec. 3 | MEASURED / SIMULATED kept separate |
| BOUNDARY:69 | free oracles differ by >= 2x on 21 of 21 | same | MEASURED |
| BOUNDARY:71 | H4c 1.000 (six visual), 0.999 (SciFact), T_A = 0.95 | x1_summary | SIMULATED, pool-supported |
| BOUNDARY:74, CONCEPTS:179-181 | N10: 26 of 37 cells, at most half | X_RESULTS sec. 3, 5 (the 26 of 40 diagnostic is not quoted) | SIMULATED, pool-supported |
| BOUNDARY:87, LIMITATIONS:77, SELECTION:267 | H3 primary >= 0.90 on all seven; lowest 0.917 [0.905, 0.929], SciFact n = 300 | x2_summary SciFact 0.9175 | SIMULATED, pool-supported |
| SELECTION:263-264 | T = 0.99 non-trivial at most 0.010 on any store | x2 secondary: max 0.01 (colpali_infovqa), others <= 0.006 | yes |
| BOUNDARY:89-92 | the three secondary cells fail | x2 secondary: T=0.97 substantial rates 0.1145, 0.3345, 0.403, 0.2055 on DocVQA x3 and SciFact; T=0.99 rates <= 0.01 | yes |
| BOUNDARY:96, SELECTION:349, METHOD_C:345 | 2,800 cells, no confirmed failure | X_RESULTS sec. 2 | stated as resamples |
| BOUNDARY:101, README:190, PACKAGE:110, BASIS:74 | H5: 13 of 16 hold, 3 do not | X_RESULTS sec. 3 | |
| BOUNDARY:105, BASIS:76 | largest labelled-retention change 0.043 | x3_scifact T2b labels 0.04303 | MEASURED |
| BOUNDARY:107-109, BASIS:79-82 | SciFact n = 300, T = 0.97: 0.9175; 0.0995 and 0.1415; 6.6x vs 2.0x | `x3_scifact.json` T2a/T2b procedure_outcome | SIMULATED, pool-supported |
| BASIS:87-91 | ColPali InfoVQA (extension) n = 225: 0.4975; 0.698 (T2a); 0.2275 (T2b) | `x3_colpali_infovqa.json` | SIMULATED, pool-supported, labelled extension |
| BOUNDARY:113, BASIS:83 | extension: 8 of 12 hold, four rotations fail | X_RESULTS sec. 4 | reported separately |
| BASIS table | holds / does not hold, candidate counts 2, 1, 5 | X_RESULTS sec. 3 table | |
| LIMITATIONS:144-150; ARCHITECTURE:127 | Q5/Q6 slowdowns; 451 queries, 500 pages, 11th-gen laptop | `Q5Q6/REPORT.md` | MEASURED, scoped |
| CHANGELOG | rescoring 5.5-9.3x, two-tier 2.9-3.7x | `batched_rescore/RESULTS.md` | ENGINEERING, scoped |
| README:92-99 | 236 of 240; 119 of 120 | `docs/UNIVERSAL.md:802,1050` | v0.3.0 numbers |
| METHOD_C_SPEC:345 | 288 real + 720 synthetic; 27 cells, max miss 0.05%, CI [0.0, 0.28]% | `E1/RESULTS.md:156-157,251-255` | matches (see A11-08) |

No in-scope text states or implies that XH or held-out-encoder transfer was shown; XH is "not registered, not
run" in README, PACKAGE, BOUNDARY sec. 8, LIMITATIONS, CONCEPTS, UNIVERSAL_PACKAGE_DESIGN, METHOD_C_SPEC.

## 4. Checked and found correct

The README/PACKAGE research paragraph; FINAL_RESEARCH_BOUNDARY sections 2-9; the CONCEPTS "pairs easy to
confuse" table; CLI help and headline strings; abstain, fallback and certificate texts in `api.py` and
`certificate.py`; CHANGELOG Unreleased performance lines; the example scripts (no claim words).

## 5. Could not check

README text derived from `docs/UNIVERSAL.md` beyond the R11/R12 counts (frozen; not re-derived). The three
test files above were the only ones run (the lead runs the full suite).

## Appendix: overstated sentences in FROZEN files (historical, not edited)

- `docs/UNIVERSAL.md:1001-1003` "A bit-level Hamming search would be [faster]"; `:413-414, 1007-1008` "an
  unoptimised Python loop" (already listed in BACKEND_ARCHITECTURE sec. 11 and RETRIEVAL_BACKEND_AUDIT W9).
- `docs/UNIVERSAL.md` title and "universal" throughout (Q10 section 14 asks that the word not be used for compression).
- `reports/research/E1/DESIGN.md:135`, `E1/RESULTS.md:171`: "no valid method can certify T = 0.99 with <= 225
  queries" (literally false per STATISTICAL_REVIEW SR-7).
- `reports/research/E1/RESULTS.md:150-157`: "proof in DESIGN.md" (not externally reviewed per Q10).
- `docs/RESULTS.md`, `docs/IMPROVEMENTS.md`: not read in full.
