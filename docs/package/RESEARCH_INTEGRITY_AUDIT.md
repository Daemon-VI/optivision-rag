# Research integrity audit of the package layer

Status: 2026-10-06, branch `package-universal`. Independent audit, read-only except for this
file. The auditor did not write the code or the documents under audit. This is an internal
audit, not an external review, and it does not widen any claim of the package or of the
research record. It was asked to find where new code or text overstates, silently changes or
contaminates the completed research record (X1-X3, E1, Q10, preregistration). It did not redo
the mathematics: `STATISTICAL_REVIEW.md` is the internal statistical review and is assessed
in section 1A only.

## 0. Scope, method, what was run

**Scope.** Everything in `git diff 91564fd HEAD` on `package-universal`: 12 commits,
`1757bc9 .. 74477ba` (HEAD at the end of the audit is `74477ba`; the audit began at `988a086`,
continued at `7e5f54a`). The selection code (`core/selection/**`), the reference and evidence
types and the parity test did not change after `988a086` (`git diff 988a086 HEAD` is empty for
them), so the parity runs below apply to HEAD. Untracked at the end: `docs/package/ENGINEERING_BENCHMARKS.md`,
`reports/engineering/package_benchmarks/`, `scripts/engineering/package_bench.py`; read for
wording and data source only.

**Read in full:** `core/selection/{statistics,rules,engine,decision,spaces,__init__}.py`,
`core/evaluate/{reference,evidence,metrics,collect}.py`, `core/artifact/{record,report,__init__}.py`,
`core/artifact/certificate.py`, `api.py`, `cli_certify.py`, `core/stages/{traits,accounting,rotation}.py`,
`core/identity.py`, `core/vectors.py`, `core/similarity.py`, `core/pipeline.py`,
`adapters/nemotron.py`, `docs/package/{CERTIFICATE_SPEC,FINAL_RESEARCH_BOUNDARY,METHOD_C_SPEC,
UNIVERSAL_PACKAGE_DESIGN,STATISTICAL_REVIEW}.md`, `docs/package/guide/{QUICKSTART,
SELECTION_AND_CERTIFICATION,LIMITATIONS}.md`, the README/PACKAGE/CHANGELOG/npm/ARCHITECTURE
diffs, the docstring diffs of `index/{numpy,qdrant}_index.py` and `__init__.py`.
**Read in part (targeted):** `core/artifact/verification.py` (every check that touches rules,
levels, outcome and replay), `core/retrieve/{exact,types}.py`, `backends/*` (declarations),
`adapters/__init__.py` (diff), `testing/{synthetic,probes}.py`, `tests/test_core_certificate*.py`
(structure, tamper table, documented limits), `tests/test_core_stage_traits.py` (tolerances,
PCA witnesses), `tests/test_core_statistics_parity.py`, `docs/package/guide/{CONCEPTS,
BASIS_DEPENDENCE,README,ARCHITECTURE,ADAPTERS_GUIDE}.md`, `API_DESIGN.md`, `TEST_REPORT.md`,
`ENGINEERING_BENCHMARKS.md`. **Not read:** `core/artifact/{bundle,codeid}.py`, `core/stages/state.py`,
`core/retrieve/{tiered,checks}.py`, `backends/*` bodies, `testing/{conformance,transforms}.py`,
`ext/`, `legacy/`, most test bodies (see section 5).

**Research record read first:** `X_RESULTS.md`; preregistration sections 6, 7, 12, 13.1-13.4,
14, 15, 16.2, 23, 25 and Appendix C; `E1/DESIGN.md`, `E1/RESULTS.md`; `e1_stats.py`,
`e1_1_study.py`, `x_common.py` (method C, `CODE_BLOBS`, S0 hash); `Q10/REPORT.md`; `X3/X3_REPORT.md`,
`X3/x3_scifact.json`, `X3/x3_colpali_infovqa.json`, `X2/x2_<store>.json` (H3 cells).

**Run** (Python 3.11.9, NumPy 2.4.6, Windows; `PYTHONPATH=src`; no research script was
executed; the frozen modules `e1_stats`, `e1_1_study`, `x_common` were imported for their
functions only; no file was written outside the scratchpad):

| what | result |
|---|---|
| `git diff --stat 91564fd HEAD -- reports scripts paper docs/UNIVERSAL.md docs/RESULTS.md docs/RESEARCH_INVESTIGATION.md` | empty at `988a086`, `7e5f54a` and `74477ba` |
| `git diff --name-status 91564fd HEAD`, filter D/R | one rename: `src/optivision/adapters.py -> adapters/__init__.py` (R057); no file under `reports/` or `scripts/` created, deleted or renamed in a commit |
| 14 pinned blobs (`x_common.CODE_BLOBS`: 11 library files + `e1_stats`, `e1_1_study`, `e1_common`) | HEAD = base = registered for all 14; `git diff 91564fd HEAD` empty for each; CRLF-normalised working-tree hashes also equal; hashes of `x_common`, `x1_agreement`, `x2_budget`, `x3_symmetry`, `s7b_selection_rules` equal X_RESULTS section 1 |
| `pytest tests/test_x_harness.py` | 62 passed, 0 skipped (86 s) |
| `pytest tests/test_x_harness.py tests/test_core_layering.py tests/test_core_statistics_parity.py` (once, as instructed) | 95 passed, 0 skipped (145 s) |
| `pytest tests/test_core_statistics_parity.py`, default | 30 passed, 0 skipped (15 s) |
| same with `OPTIVISION_FULL_PARITY=1` | 30 passed, 0 skipped (60 s) |
| own differential test `diff_methodc.py` (scratchpad `audit/`) | 384 random problems, 176 boundary cases, 0 differences, see section 3 item 1 |
| own probes `probe_cert.py`, `probe_forge.py`, `probe_traits.py`, `probe_traits2.py`, `probe_space.py` | see findings; SYNTHETIC vectors only (`optivision.testing`) |
| numbers quoted in FINAL_RESEARCH_BOUNDARY, README, PACKAGE, guide | checked against X_RESULTS.md and the per-dataset JSON files, section 1 area 9 and 12 |

## 1. Verdicts

| area | verdict | one line |
|---|---|---|
| 1 Research record | PASS | Untouched at the byte level (diff empty, 14 blobs equal, harness 62/62, `src/` imports no research script). One NOTE (RI-19). |
| 2 Method C unchanged | PASS WITH FINDINGS | The arithmetic is identical (bitwise, about 2,400 independent comparisons); the findings are about what is added around it: RI-01, RI-05, RI-06. |
| 3 Candidate semantics | PASS | `default(dim)` hashes to `af29b011...` at 15 dimensions from 1 to 4096; positions, families, pipelines equal the research sidecar; 3 duplicates kept (37 distinct of 40); `wide` is separate. RI-17 on the word "registered". |
| 4 Hidden claims | PASS WITH FINDINGS | No `guarantee`, `novel`, `state of the art`, `faster`, `lossless` claim in `src/`; no claim of transfer. Wording findings RI-03, RI-11, RI-12, RI-15, RI-22. |
| 5 Basis traits | PASS WITH FINDINGS | Traits are used only descriptively (plus one A3 guard). Three declared invariances fail on constructed data and the pipeline-level fields do not say so: RI-07; related: RI-08 (rotation convention), RI-16 (third-party declarations). |
| 6 Leakage and contamination | PASS | Research files are read only by equality tests; no tuning; threshold revisions legitimate; no model output encoded or scored. |
| 7 Silent model assumptions | PASS WITH FINDINGS | Unit norm neither checked nor recorded (RI-13); graded qrels accepted silently (RI-14); duplicates accepted (RI-05). |
| 8 Decomposition claim | PASS WITH FINDINGS | Layers C, D, E are accurate; rows A and B and the title overstate: RI-15. |
| 9 Boundary and spec documents | PASS WITH FINDINGS | Every number checked is right; two omissions (RI-09, RI-10) and the spec/certificate mismatch RI-03. |
| 10 Certificate honesty | REJECT as built | `certified: true` is reachable for a rule that rejected nothing (RI-01), and verification can be defeated by renaming the rule (RI-02). Accepted when RI-01 and RI-02 are fixed with the tests listed there. |
| 11 `certify()` and CLI | PASS WITH FINDINGS | No data path into list, order or target; ABSTAIN is presented as normal; agreement is labelled everywhere. RI-06, RI-11, RI-12. |
| 12 README / PACKAGE / boundary numbers | PASS WITH FINDINGS | All numbers and verdicts match; the corrected X3 sentences are exact in FINAL_RESEARCH_BOUNDARY and UNIVERSAL_PACKAGE_DESIGN, one-sided in BASIS_DEPENDENCE (RI-09). |
| 13 XH eligibility | PASS | No model output encoded, loaded, compressed or scored; no per-model default; no reference to the research worktree's ColEmbed 8B results. |

**BLOCKER: none.** The closest are RI-01 and RI-02; I would not allow any external use of the
word "certificate" or the `verify` claim in the README until both are fixed.

### 1A. The MAJOR findings of STATISTICAL_REVIEW.md

I did not redo the mathematics. I checked each finding against the code and ran my own probes
where the finding is about a code path.

| SR | agree? | is the required change sufficient? |
|---|---|---|
| SR-1 duplicated or clustered queries | Agree. Reproduced (probe D): 80 queries plus their 80 exact copies under new ids were accepted, outcome SELECT, nothing in `report()` or the certificate. | No. W7 is a documentation change. Add detection of exact duplicate query vectors (digest of each query's rows) in `collect_evidence`, record the count in the evidence and the certificate, and print it in `report()` and the CLI (RI-05). Near-duplicates stay a documentation matter. |
| SR-2 order | Agree. Checked: `engine.py:54-77` default order "given"; the shopping risk is stated in the guide (section 6 table) and METHOD_C_SPEC 1.4 and 10.7, but not in `certify()`, `select()`, the CLI help or the certificate. | Partly. W4 and W8 are right for the documents. Also needed: the same sentence with the numbers in the docstring of `order_seed` and in the `--order-seed` help, and a caveat recorded in every certificate whose order kind is `given` (RI-06). Defaulting to a seeded permutation derived from query ids is acceptable but changes the meaning of "given"; either choice must be recorded. |
| SR-3 STATEMENT wording | Agree. CERTIFICATE_SPEC.md:272-274 says the certificate carries the "not a conditional probability, not the sample retention, not a future batch" sentences; the recorded `STATEMENT` (`rules.py:40-48`) does not. | Yes for the text (W1). Also change in the same commit: CERTIFICATE_SPEC section 5, the quoted statement in `guide/SELECTION_AND_CERTIFICATION.md` section 4 and `QUICKSTART.md`, `A1-A6` in `rules.py:39,45,138` and `certificate.py:12`, and every test that asserts on the text (RI-03). |
| SR-4 assumptions incomplete | Agree. | No. W2-W6 cover items 1, 3, 4 and 5 of the review's section 8 list. Item 2 (the per-query metric is a function of the single query, the same queries for baseline and candidates, no batch statistic) and item 6 (the sample size is not chosen by outcomes except through a prefix) appear in no W. Also record the retrieval procedure that was evaluated (score kind, tie-break, exact decode-first) so that "deployed = evaluated" is checkable (RI-04). |
| SR-5 custom rule (rated MINOR) | I rate it MAJOR: RI-01. | W16 alone is insufficient: `Decision.certified`, `certificate.result.certified` and the verifier also have to change. |

I agree with the MINOR and NOTE findings SR-6 to SR-14 and have nothing to add except that
SR-9 (numbers where the warning is) should include the abstain note (RI-11) and the CLI
(RI-12).

## 2. Findings

Severity: BLOCKER = must be fixed before this work can be called scientifically honest;
MAJOR = fix before any release or external use; MINOR; NOTE.

| id | sev | location | one line |
|---|---|---|---|
| RI-01 | MAJOR | `engine.py:149,163`; `certificate.py:502,542,556`; `record.py:76-78`; `CERTIFICATE_SPEC.md:365-367` | A custom rule that sets `certifying = True` and tests nothing gets `certified: true`, the library's STATEMENT, and `verify().ok`. |
| RI-02 | MAJOR | `verification.py:396,447,536,671`; `CERTIFICATE_SPEC.md:203-233`; `README.md:178` | Renaming `rule.name` turns the level, decision, text and replay checks into `skipped`; an edited target or p-values verify `ok` with all live objects supplied. |
| RI-03 | MAJOR | `rules.py:40-48`; `CERTIFICATE_SPEC.md:272-274`; `record.py:76-78` | The recorded statement says "only under A1-A6" and lacks the disclaimers the spec says it carries (SR-3). |
| RI-04 | MAJOR | `rules.py:50-73`; `certificate.py` (no retrieval-procedure field) | The assumption list is not the list the proof needs; the evaluated retrieval procedure is not recorded (SR-4). |
| RI-05 | MAJOR | `collect.py:157-229`; `evidence.py:193-220`; `guide/SELECTION_AND_CERTIFICATION.md:163` | Exact duplicate queries are accepted silently (SR-1). |
| RI-06 | MAJOR | `engine.py:54-77`; `api.py:441-442,457-462`; `cli_certify.py:148-149` | Default order and order-seed shopping are not warned about where the user reads (SR-2). |
| RI-07 | MINOR | `traits.py:229-241,291-342,377-415,434-452,623-634`; `guide/BASIS_DEPENDENCE.md:28,37` | Three declared invariances fail on constructed data and the pipeline-level fields carry no condition. |
| RI-08 | MINOR | `rotation.py:11-14` | "The same convention as X3" is true for the matrix and false for its application: the package applies the inverse rotation. |
| RI-09 | MINOR | `guide/BASIS_DEPENDENCE.md:81-84` | The ColPali InfoVQA sentence gives only the upward cell. |
| RI-10 | MINOR | `FINAL_RESEARCH_BOUNDARY.md:72-74`; `guide/CONCEPTS.md:178-181` | "26 of 37 pool-supported cells" drops the condition that decides which 37 cells. |
| RI-11 | MINOR | `api.py:80-86`; `guide/QUICKSTART.md:145` | The abstain note names "a lower target" next to "more queries". |
| RI-12 | MINOR | `cli_certify.py:103,125`; `api.py:24,110,201,414`; `certificate.py:10-17`; `CERTIFICATE_SPEC.md:327-329`; `guide/CONCEPTS.md:206-208`; `guide/QUICKSTART.md:69` | Wording stronger than the rule's statement: "select (certified)", "the index it was certified for", "covers this choice", "root of trust". |
| RI-13 | MINOR | `vectors.py:151-265`; `certificate.py:338-340`; `FINAL_RESEARCH_BOUNDARY.md:15-16`; `README.md:136-138` | Unit norm, the only regime studied, is neither checked nor recorded. |
| RI-14 | MINOR | `reference.py:54-67`; `cli_certify.py:56-67` | Graded qrels given as `{doc: grade}` count grade 0 as relevant, silently. |
| RI-15 | MINOR | `UNIVERSAL_PACKAGE_DESIGN.md:1,37,38,50-51` | Title and rows A and B overstate ("universal by construction", "proven from code"). |
| RI-16 | MINOR | `traits.py:183-223`; `certificate.py:457` | A third-party stage that declares itself ORTHOGONAL gives `basis_sensitive: false` with no marker. |
| RI-17 | MINOR | `spaces.py:268-270`; `certificate.py:536`; `cli_certify.py:94` | "registered default list" has no scope. |
| RI-18 | MINOR | see text | SR-7, SR-8, SR-9 wording items (agreed). |
| RI-19 | NOTE | untracked `reports/engineering/`, `scripts/engineering/` | New top-level directories next to the research record. |
| RI-20 | NOTE | `src/optivision/__init__.py` | `import optivision` now registers the `rotation` stage name. |
| RI-21 | NOTE | `statistics.py:11-18` | Citations (Westfall and Krishen; Angelopoulos et al.) are stated as fact, not in the research record, unchecked. |
| RI-22 | NOTE | `api.py:30` | "Model-blind by construction" omits "relative to an encoder and a date". |
| RI-23 | NOTE | docs | Research numbers are copied by hand; no test compares them with the JSON. |
| RI-24 | NOTE | `FINAL_RESEARCH_BOUNDARY.md:104-106` | The SciFact rates are given without `error = 0.05`. |
| RI-25 | NOTE | `statistics.py:279-281` | A zero-baseline prefix (`query_budget`) raises instead of recording an abstention. |

### RI-01 (MAJOR) A rule that declares `certifying = True` is certified, with the library's statement

What is wrong. `engine.py:149` sets `certified = bool(rule.certifying)` for any object that has
the attributes of the `SelectionRule` protocol; `certificate.py:502,542` then stores the
library's `STATEMENT` for it, and `verification.py` skips every decision-level check for a rule
whose name it does not know (lines 396, 447, 536, 671), so the report is `ok`. CERTIFICATE_SPEC
section 3 says "`certified` is true only for `select` under a certifying rule, and that is
enforced both when a certificate is created and ... when it is verified"; the enforcement is of
a flag the rule sets about itself. Section 8 of the spec admits it and answers with an
instruction to users ("must not be issued a certificate with this function", lines 365-367).
`rules.py:15` defines "certified" as "its hypothesis is rejected"; here nothing was rejected.

Evidence (`probe_cert.py`, probe A, synthetic vectors). A class `AlwaysSmallest` with
`certifying = True` whose `decide` returns the smallest candidate with p-values all 1.0:
`certify(..., rule=AlwaysSmallest())` returned `certified=True`; the certificate's `statement`
starts "P(a compressed candidate is selected and its retention under Q is below T) <= delta";
`assumptions` is the rule's own `("none: trust me",)`; `caveats` is empty;
`verify(..., evidence, corpus, queries, reference, pipeline, compressed).ok` is `True` at level
`binding` with `consistency.level`, `consistency.decision`, `rule.text` skipped. The one test
that covers a custom rule (`tests/test_core_certificate.py:405-437`) uses `certifying = False`;
nothing covers `True`. A subclass of `FixedSequenceBetting` that overrides `decide` and keeps the
name (probe A2) also gets `certified=True` at creation; only the level and decision checks
(and the replay with evidence) catch it afterwards.

Is it acceptable to record the library's STATEMENT for a rule the library knows nothing about?
No. The record is meant for third parties; `certified: true` plus the Method C statement
attributes to unexamined code a property the library only derived for two of its own classes.

Required change.
1. `select()`: `Decision.certified` is true only if `type(rule)` is exactly one of the built-in
   certifying classes (exact type, not `isinstance`, not the `name` string). For any other rule
   `certified` is false whatever the rule declares; `Decision.rule` records
   `{"name", "class": "<module>.<qualname>", "library_rule": false, "declared_certifying": <flag>,
   "config"}`.
2. `statement_for`: for a non-library rule record a fixed text: "This record was produced by a
   rule that is not part of this library (class X). The library has not examined it and makes no
   statement about it. Any statement it makes is its author's." Record the rule's own
   `assumptions()` under a label "declared by the rule's author, not examined by this library",
   and add a caveat sentence. Never record `STATEMENT` for it.
3. `verification.py`: when `result.certified` is true and the rule name is not built in, the
   checks `rule.text`, `consistency.decision` and `replay.decision` are `fail`, not `skipped`
   (a skip stays correct for `certified: false` records, which the existing test pins).
4. Tests: a custom rule with `certifying = True` gives `certified is False`; a subclass with a
   built-in name is recognised as non-library; both rows go into the tamper table.
5. Correct CERTIFICATE_SPEC sections 3 and 8, `UNIVERSAL_PACKAGE_DESIGN.md` D5 and D12, and the
   docstrings of `select()` and `SelectionRule.certifying`.

### RI-02 (MAJOR) Renaming the rule hides every edit of the decision

What is wrong. The checks that recompute the level, the decision step and the replay, and the
text check, are skipped for an unknown rule name. `report.ok` is "no check failed"
(`report.py:75-78`), so a forger needs only one extra field edit. CERTIFICATE_SPEC section 4.1
lists, for "target", "error level", "error level and level together" and "one p-value", the
checks `replay.decision` and `consistency.decision` as the ones that catch them "with
everything"; with the rule renamed none of them runs. The tamper tests use exactly this
adversary ("with the digest recomputed, as a careless or dishonest editor would") but no row
edits `rule.name`.

Evidence (`probe_forge.py`, honest certificate of a `FixedSequenceBetting` decision,
all live objects and the honest evidence supplied, digest recomputed after the edit):

| edit | `ok` | failed | skipped |
|---|---|---|---|
| honest | True | - | - |
| target 0.9 -> 0.999 | False | `replay.decision` | `signature` |
| target 0.9 -> 0.999 and `rule.name` -> `fixed_sequence_betting_v2` | **True** | none | `signature`, `consistency.level`, `consistency.decision`, `rule.text`, `replay.decision` |
| error 0.05 -> 0.5, level 0.5, every p-value 1e-9, every position certified, `rule.name` -> `my_rule` | **True** | none | the same |

The report's `level` is `binding`. `README.md:178` and `PACKAGE.md:98` annotate
`verify("artifact/").ok` with "integrity, replay from stored evidence, binding to stored codes":
true only for a built-in rule.

Required change. The same item 3 of RI-01 (a certified record with an unknown rule fails), plus:
(a) add rows "rule name" (renamed, with and without another edit) to the tamper table and to
CERTIFICATE_SPEC section 4.1; (b) the summary line of a report says how many decision-level
checks were skipped and why when `result.certified` is true; (c) the two README annotations say
"replay (built-in rules)". If a user-supplied rule is to be verifiable, `verify` takes
`rules={name: rule_object}` and a record that names a rule not in that mapping and not built in
fails.

### RI-03 (MAJOR) The recorded statement is not the statement the spec says it is (SR-3)

`rules.py:40-48`: "...and only under assumptions A1-A6" is logically wrong (nothing is claimed
if an assumption fails; the statement is not false). CERTIFICATE_SPEC.md:259-262 and 272-274
say the certificate carries "not a conditional probability given that something was selected,
not a statement about the retention measured on the sample, not a statement about a finite batch
of future queries" and that its wording "paraphrases the text the certificate records ... with
the same conditions". The recorded `STATEMENT` has none of those three sentences. A reader of a
certificate alone sees "certified" and a joint probability and nothing that blocks the reading
"95% of selections are right"; `STATISTICAL_REVIEW.md` section 6 shows 17% of selections below
target in a plausible mixture while the statement holds in every deployment.
Required: W1 of the review, verbatim or equivalent, and the same-commit changes listed in
section 1A (SR-3 row).

### RI-04 (MAJOR) The assumption list is incomplete and the evaluated retrieval is not recorded (SR-4)

`rules.py:50-73` lists A1-A6. The proof also needs: one look at one (T, error, rule, metric,
reference, list, order) per sample; per-query evidence that is a function of the single query
and the same queries for baseline and candidates; a reference fixed independently of the
candidates; the deployed index equal to the evaluated one (same corpus, same fitted pipeline,
same retrieval procedure); a sample size not chosen by outcomes except as a prefix.
Required: W2-W6 plus the two items not in any W (per-query evidence; sample size), renumber to
A1-A8 if A7 is split (one look; deployed = evaluated), and record in the certificate the
retrieval procedure evaluated (`score_kind`, `tie_break`, "exact decode-first MaxSim") next to
the pipeline, because `core/retrieve/types.py` already names them and a Qdrant backend serves
another procedure (`backends/qdrant_backend.py:9-13`).

### RI-05 (MAJOR) Exact duplicate queries are accepted silently (SR-1)

Evidence (probe D, synthetic): 80 queries plus the same 80 query vectors under new ids,
`certify(... target=0.5)` returned SELECT on 160 queries; `report()` has no field about
duplicates; `validate_corpus` checks only unique ids. `STATISTICAL_REVIEW.md` measured 0.211
against 0.05 when every query is counted twice. Required: in `collect_evidence` digest each
query's rows, count exact duplicates, store the count in `Evidence` and the certificate, show it
in `report()` and the CLI, and record a caveat when it is above zero (`certify` refuses unless
the caller passes `allow_duplicate_queries=True`). The documents say "near-duplicates" and must
say what W7 says.

### RI-06 (MAJOR) The default order and seed shopping are warned about only in the guide (SR-2)

What is wrong. The default order is the order the caller gave; A5 says "fixed before outcomes
are seen", which an order sorted by difficulty, date or source satisfies in letter and breaks in
effect (0.279 at a stated 0.05 in the review). `order_seed` is documented as "independent of the
data" (`api.py:441-442`, `engine.py:99-103`, `cli_certify.py:148-149`) with no word on trying
several seeds, which the review measured as 0.076 (best of 20) and 0.169 (best of 100). The
warning exists in `guide/SELECTION_AND_CERTIFICATION.md` section 6 and METHOD_C_SPEC 1.4 and
10.7 only. The decision records the seed that was used and nothing about the others, so a
shopped seed is undetectable. My own checks: a seeded permutation is exactly the frozen function
on the permuted arrays (60 cases, 0 differences), and prefix budgets are nested (282 comparisons,
0 violations), so `query_budget` is safe and `order_seed` shopping is not.
Required: the sentence of W8 with its two numbers in the `order_seed` docstrings and the
`--order-seed` help; a caveat in every certificate with order kind `given`: "queries were tested
in the order supplied; the statement requires that order to be independent of the outcomes".

### RI-07 (MINOR) Declared basis invariances that fail on constructed data

The module docstring (`traits.py:1-12`) says the traits are "in exact arithmetic ... checked on
synthetic data". The notes of each stage state conditions; the machine-readable value does not,
and `pipeline_traits` adds no note for an ORTHOGONAL stage (`traits.py:306-342`). Probes
(`probe_traits.py`, `probe_traits2.py`, synthetic):

- `pca` with a tied eigenvalue at the cut (singular values 4,3,2,2,1,0.5; `dim=3`): scores
  moved by up to 0.53 of the largest absolute score under random rotations, while
  `pipeline_traits(Pipeline([PCAProjector(dim=3)]))` is `orthogonal`, `basis_sensitive=False`,
  `conditionally_invariant=False`, `notes=[]`. (The test suite has this witness at
  `tests/test_core_stage_traits.py:722-760`; the certificate does not carry it.)
- `hierarchical_merge(ratio=0.5)` on documents of three orthonormal vectors (exactly equidistant):
  30 of 30 random rotations changed the scores, largest change 2.1 of the largest absolute
  score. The note says
  "exact ties ... are resolved by row order", which holds in exact arithmetic and not after
  float32 rotation. No witness test covers ties for the merge stages. (Duplicate vectors inside
  a document, and the cosine-tied variants I built, did not change.)
- `binary` with one coordinate that is exactly zero, d = 13, under a signed permutation: scores
  moved by 0.188 of the largest absolute score; without exact zeros 1.2e-7. The note says so
  (`traits.py:627-630`); `guide/BASIS_DEPENDENCE.md:28-30` does not, and the enum value is
  `signed_permutation` unconditionally.

Every default candidate is basis sensitive (`LIMITATIONS.md` section 3), so no default
certificate carries the unconditional `basis_sensitive: false`; custom and `wide` lists can.
Required: a field `holds_for` ("all data" or "generic data: no ties, distinct eigenvalues at
the cut, no exact zeros, exact arithmetic") on `StageTraits`, propagated to `PipelineTraits`
with a note and into the certificate; a tie witness for the merge stages; the sign-code
exception in BASIS_DEPENDENCE section 2; and "can be read from code and checked on synthetic
data" in place of "proven from code" (RI-15).

### RI-08 (MINOR) `RotationProjector` is not X3's rotation

`rotation.py:11-14`: "The same convention is used by the registered symmetry experiment for its
rotation maps." The matrix is the same (`seeded_rotation(128, s)` equals the X3 `rotation(128, s)`
for s = 1, 2). X3 applies `x @ U.T` (`x3_symmetry.py:83-88`, OP3: "every vector v becomes U v");
the stage applies `x @ U`. Checked numerically: the stage's output for seed s equals X3's inverse
rotation, not T2a or T2b. Required: say "same matrix, applied as the inverse; `RotationProjector`
does not reproduce T2a or T2b".

### RI-09 (MINOR) One-sided direction sentence

`guide/BASIS_DEPENDENCE.md:81-84`: "On ColPali InfoVQA (extension set) the outcome at T = 0.97
with 225 queries was higher under one rotation than in the native basis." From
`x3_colpali_infovqa.json`: native 0.4975; T2a 0.698 (median compression 1.0 -> 7.76); T2b 0.2275
(lower). Required: give both ("0.4975 native, 0.698 under T2a, 0.2275 under T2b"). The
corrected sentences elsewhere are exact: FINAL_RESEARCH_BOUNDARY 6 ("material under both
rotations, upward in some cells and downward in others": T2a n=225 T=0.97 up; T2b n=225 T=0.97
down and n=450 T=0.95 up with ratio 2.46) and 7 ("downward on SciFact and in both directions on
one extension dataset"); UNIVERSAL_PACKAGE_DESIGN.md:403-405; `rotation.py:51-52`.

### RI-10 (MINOR) N10 paraphrase

`X_RESULTS.md` section 5 (D1): the 26 of 37 are the pool-supported cells "where the labels arm
deploys non-trivially at a rate of at least 0.90". FINAL_RESEARCH_BOUNDARY.md:72-74 and
CONCEPTS.md:178-181 say "26 of 37 pool-supported cells", which reads as all such cells.
Required: add the condition.

### RI-11 (MINOR) The abstain note suggests a lower target (SR-9)

`api.py:80-86` and the printed output in `QUICKSTART.md:145`: "with more queries, or a lower
target, a later run may certify one. Decide on that before looking again". The next sentences
of the guide are right ("collect a new, independent sample, and choose the target and the list
beforehand"). Required: W15 wording in the note, with the 3-to-14-fold inflation the review
measured for further looks.

### RI-12 (MINOR) Wording stronger than "the rule rejected the hypothesis"

- `cli_certify.py:103`: the headline "select (certified)" in bold green; under the agreement
  construct the headline has no construct. Required: "select (the rule rejected 'retention <=
  target' for this candidate at level L; statement below)", and for agreement add "agreement with
  the uncompressed ranking, not retrieval quality" in the headline.
- `cli_certify.py:125`, `api.py:24,110,201,414`, FINAL_RESEARCH_BOUNDARY.md:159: "conditional
  on" next to a probability (SR-10, W10, W11).
- `certificate.py:10-17` and CERTIFICATE_SPEC.md:16: "is this the index it was certified for?"
  makes the index the object of certification. The binding checks show that the codes equal the
  recorded digests; the statement concerns the procedure. Required: "does this record match the
  index it was issued for".
- `CERTIFICATE_SPEC.md:327-329`: the manifest "is not the root of trust: the certificate is (its
  digest, the codes and fitted-state digests it records, an optional HMAC)". Without a key the
  digest is not a root of trust (the same document says so in sections 1 and 6). Required:
  "without an HMAC key nothing in the bundle is a root of trust".
- `guide/CONCEPTS.md:206-208`, `guide/QUICKSTART.md:69`: "covers this selection / this choice".
  The statement is about how often the procedure selects wrongly, not about this selection
  (`STATISTICAL_REVIEW.md` section 6). Required: "the rule rejected 'retention <= T' for this
  candidate at the level stated; the statement bounds how often the rule does so wrongly".

### RI-13 (MINOR) Unit norm is neither enforced nor recorded

The research scope is MaxSim over L2-normalised vectors (preregistration section 5, non-claim
2). `validate_corpus` has no norm check, `EncoderIdentity.normalized` is metadata the caller
declares, and the certificate records no norm. Probe C: a corpus with row norms 0.61 to 1.49
gave a SELECT, no warning, and "norm" appears nowhere in the certificate. The guides state the
scope (`LIMITATIONS.md` section 1, `ADAPTERS_GUIDE.md:37`, `guide/README.md` item 10); the README
section and FINAL_RESEARCH_BOUNDARY.md:15-16 say "for any late-interaction encoder ... scored by
MaxSim". Required: record `row_norm_min` and `row_norm_max` in `corpus` (descriptive), warn when
they are not within 1e-3 of 1 unless `encoder.normalized` is False, and add "L2-normalised
vectors are the only regime studied" to those two sentences.

### RI-14 (MINOR) Graded qrels are accepted silently

`LabelReference` takes `tuple(docs)` (`reference.py:63`); for `{doc: grade}` that is the keys,
so grade 0 counts as relevant (probe B: `['d000000', 'd000050']`, the second with grade 0). The
CLI `_read_qrels` does `set(v)` (`cli_certify.py:65`). `LIMITATIONS.md` section 9 explains it
with an example, the `LabelReference` docstring says it, the certificate spec says its identity
ignores grades; the `certify()` docstring and the CLI help do not. Research data are binary, so
no research number is affected. Required: refuse a mapping value unless `min_grade=` is given,
or at least warn once; mention it in the `qrels` help.

### RI-15 (MINOR) Design document overstates layers A and B and carries "universal" in the title

`UNIVERSAL_PACKAGE_DESIGN.md:1` "a universal interface and decision framework" (Q10 section 14
forbids "universal"; the section 1 definition limits it, the title does not); line 37 "universal
by construction; enforced by validation and tests" (unit norm is not enforced, RI-13); line 38
"universal per operator" and lines 50-51 "can be proven from code" (RI-07: generic data, exact
arithmetic, checked on synthetic data, not proven); line 40 omits that the derivation is not
externally reviewed and that the assumption list is under review (RI-04).
Proposed replacements. Title: "OptiVision package design: a model-agnostic interface and
decision framework". Row A: "the same contract for every encoder; enforced for arrays that pass
validation (finite, unique ids, consistent shapes; unit norm is not checked)". Row B: "declared
per operator for generic data in exact arithmetic (conditions in each operator's notes); read
from the code and checked on synthetic data with witnesses; says nothing about quality". Row D:
"does not depend on the encoder, given its assumptions; the derivation is the project's own and
has not been externally reviewed; the assumption list is under internal review
(STATISTICAL_REVIEW.md SR-4)". The remaining rows (C, E) and the corrections beneath them are
accurate and no stronger than preregistration section 15; "X2: at a 0.99 target the registered
deployment criterion was not met on any store at pool-sized budgets" is exact (the non-trivial
deployment rate was at most 0.010 on every store, checked in the seven `x2_<store>.json` files;
"nothing is certified" would be stronger than that and is not used).

### RI-16 (MINOR) A third-party declaration is recorded as a property of the code

`traits.py:183-223`: a stage's own `traits` attribute wins over the registry. Probe: a subclass
of `BinaryQuantizer` that declares `ORTHOGONAL` gives `pipeline_traits` `orthogonal`,
`basis_sensitive=False`, `notes=[]`. `certificate.py:457` stores that as `candidate.traits`.
Required: record per stage `source: "registry" | "declared by the stage class"` and a note when
the declaration is not from the registry; `testing.check_stage` result may be linked.

### RI-17 (MINOR) "Registered default" has no scope

`spaces.py:268-270` `is_registered_default`, the certificate field `space.registered_default`
and the CLI line "the registered default list" are true of the list's hash. The registration
(preregistration section 14.1, verified at d = 128 and 96) concerns P0 on the development stores.
Required: "the list registered in the preregistration for the development stores; not validated
for other encoders" in the CLI line and the field's documentation.

### RI-18 (MINOR) Agreed items of the statistical review

SR-7 (floors are for any valid test at 50% power; the implemented bet needs 1.3 to 1.7 times
more; `api.py:421` "at least one"), SR-8 (`SELECTION_AND_CERTIFICATION.md:166` "can the software
check A4: yes" checks the sample, not the population), SR-9 (numbers where the several-looks
warning is). Required as in the review (W9, W12, W13, W14, W15).

### NOTES

- RI-19. `reports/engineering/package_benchmarks/` and `scripts/engineering/` are untracked new
  directories beside `reports/research/`. They are not in the research record and
  `ENGINEERING_BENCHMARKS.md` labels every number "NEW ENGINEERING MEASUREMENT on SYNTHETIC data
  on ONE laptop ... not a research result" (checked: it names no encoder and no research store;
  the script has no model import). The rule "nothing under `reports/` changes" in
  `UNIVERSAL_PACKAGE_DESIGN.md` section 2 item 4 is written for `reports/research/**` and
  `reports/universal/**`, so this is consistent; keep it that way in the final commit and in the
  numbers (the document says the lead will re-run on a quiet machine).
- RI-20. `src/optivision/__init__.py` imports `core.stages`, which registers the `rotation`
  stage name in the pinned registry at `import optivision`. S0, its hash and the pinned blobs are
  unaffected; CHANGELOG states it.
- RI-21. `statistics.py:11-18` and `METHOD_C_SPEC.md` cite Westfall and Krishen (2001) and
  Angelopoulos et al. (2021). The research record cites Waudby-Smith and Ramdas and "learn-then-
  test-style" (Q10). `STATISTICAL_REVIEW.md` section 12 says its citations are from memory and
  unchecked. Keep that caveat next to the citations.
- RI-22. `api.py:30` "Model-blind by construction". Preregistration section 15.3: the property is
  relative to an encoder and a date, and tests 3 and 4 are about process. Add "in the sense of
  preregistration section 15".
- RI-23. The research numbers in the docs are copied by hand. I checked them (section 3); a test
  that reads `x3_scifact.json` and `x2_*.json` and asserts the quoted values would stop drift.
  It must not print or compute anything beyond equality.
- RI-24. `FINAL_RESEARCH_BOUNDARY.md:104-106` gives the SciFact rates 0.9175, 0.0995, 0.1415 at
  T = 0.97, n = 300 without the error level (0.05); BASIS_DEPENDENCE has it.
- RI-25. A prefix of the evidence whose baseline is zero everywhere raises `ZeroBaselineError`
  (`statistics.py:279-281`). A raise is an abstention for the statement (`STATISTICAL_REVIEW.md`
  7.6) but `certify(query_budget=m)` loses the evidence it has collected. Record an abstention
  with the reason instead.

## 3. What I tried to break and could not

Numbers are comparisons or cases; "0 differences" is bitwise unless stated.

1. **Method C is the frozen procedure.** Line-by-line comparison of `statistics.py:40-139` with
   `e1_stats.py:43-146` and `x_common.py:940-948`: `to_y`, the wealth recursion (priors, `t+1`,
   `t+2`, clip to `[0, 0.9/m0]`, `log1p`, running maximum including 1), `<=` inclusion, the
   per-family stop, `level = error / len(dict.fromkeys(families))`, the key
   `(bytes, vectors, position)`. Identical operation order. Independent differential test
   (`diff_methodc.py`) against `x_common.method_c` and `e1_stats.*` through the frozen
   `e1_1_study.Problem`: 384 random problems (K in 1..82, n in 1..225, single family, all
   distinct, interleaved, blocks, exact size ties, metric values 0 and 1, lossless and
   duplicated candidates, targets 0.05..1.0, errors 0.001..0.5; 77 selections, 307 abstentions,
   49 with several certified candidates): p-value bytes, certified sets and selected index equal,
   for `FixedSequenceBetting`, `BonferroniBetting` (method D, `e1_1_study.py:132-134`) and for
   `select()`; 176 cases with the level placed on a p-value and one ulp either side: equal; 720
   comparisons of the stacked `(S,K,n)` research use against the package `(K,n)` and `(n,)` use:
   equal. `float32` and `float16` input is widened before arithmetic; the frozen function on the
   narrow input differs, which is a documented, conservative change (`statistics.py:40-57`).
2. **Additions around Method C.** Seeded permutation equals the frozen function on the permuted
   arrays (60 cases); prefix budgets give nested certified sets and non-increasing p-values (282
   comparisons); input validation only raises (NaN, inf, outside [0,1], zero baseline mean,
   n = 0, K = 0, wrong dtype or shape, bad target or error: 19 refusals checked; `np.longdouble`
   is not refused on Windows, where it is 64-bit) and never changes a selection; the fallback is
   never certified and is ignored when something is selected (checked by reading
   `engine.py:148-153` and `decision.py:128-134`; the statistical review ran it);
   `BonferroniBetting` is method D, K counted with duplicates, same choice key.
3. **The parity tests are not vacuous.** They import the frozen modules by path, assert the
   `e1_stats.py` blob first, and fail rather than skip (0 skips in every run). Compared by
   default: 100,800 stored Method C and D selections (every fifth seed), 504,000 with
   `OPTIVISION_FULL_PARITY=1` (7 stores x 3 budgets x 3 targets x 2 error levels x 2 methods x
   2,000 seeds); 45,609 of the 504,000 (9.0%) are selections, 50 of 252 cells mix abstentions
   and selections, so a never-select or always-select implementation fails. 352 random problems
   compare p-value bytes. Weakness, not a finding: the level and `F` are recomputed in the test
   (`test_core_statistics_parity.py:104,132`) for the random problems; the frozen `Problem.F`
   and `run_cell` are used in test (b) with 8 seeds.
4. **Candidate space.** `CandidateSpace.default(d).identity == af29b01146...3520` for
   d in {1, 2, 7, 8, 13, 64, 96, 100, 128, 129, 256, 512, 1024, 2560, 4096}; 40 candidates, 7
   families, `is_registered_default`; every `(position, family, pipeline)` equals the research
   sidecar `colpali_docvqa.json`; 37 distinct pipelines of 40 (duplicates kept, nothing
   reordered or filtered); `wide(128)` has 54 candidates, 14 families, another identity.
5. **No use of traits to skip or assume.** Outside `traits.py` they are read by `api.report`,
   `cli_certify`, `certificate.py:238,457`, `verification.py:563` (descriptive) and by
   `collect.py:79-93` (refuses a stage that declares `fit_uses_queries`, or declares nothing,
   which protects A3). `conditionally_invariant` is never used as invariance
   (`PipelineTraits` keeps `basis_sensitive=True` for such a pipeline).
6. **Rotation and PCA are not presented as a fix.** `grep` for improve, fix, helps, whiten,
   decorrelate in `rotation.py`, `traits.py`, `probes.py`, the design document and
   BASIS_DEPENDENCE: no such claim; "a rotation stage is not presented as an improvement" occurs
   three times. The PCA claim is stated with its conditions in the traits note, the design
   document 5.3, BASIS_DEPENDENCE section 2, and the tests have witnesses for it.
7. **Leakage.** Tests that touch the research record: `test_core_statistics_parity.py` (per-query
   arrays and stored selections, equality only, no print), `test_core_statistics.py:41` and
   `test_core_spaces.py:30` (sidecar metadata), `test_core_rotation.py:98` (a docstring). None
   computes, selects or tunes from per-query values. The thresholds of `test_core_stage_traits.py`
   (`EQUAL = 1e-5`, `EQUAL_EIGEN = 5e-5`, `MIN_GAP = 0.01`, `WITNESS = 10 * EQUAL`) are the
   revision the stages engineer disclosed: the relaxed tolerance applies only to pipelines with
   an eigendecomposition, is derived in a comment (float32 noise 1e-7 divided by an eigenvalue
   gap of at least 1%), the gap is asserted on the test data
   (`test_the_test_data_have_well_separated_eigenvalues`), and an exactly isotropic witness
   shows a change above 0.1 where the condition fails. I judge the revision legitimate. (I did
   not see the engineer's own note of the two revisions; this reading is from the code and its
   comments.)
8. **Hidden claims.** `grep` of new `src/` and docs for universal, guarantee, certif, lossless,
   state of the art, novel, first, better than, faster, any/every/all model, invariant, robust,
   safe, scales: in `src/` only "safe outcome" (abstention) and score scales; the docs use the
   other words in prohibitions and in correct contexts. "Certified" is used only for members of
   the rejected set, apart from RI-01 and RI-12. No sentence presents X1-X3 as evidence of
   transfer; the speed statements restate Q5/Q6 (4.4-4.7x, 24-34x, 28-30x slower; checked in
   `Q5Q6/REPORT.md:93,114,122`) as "this stack, one laptop CPU".
9. **Research numbers in the documents.** Checked against X_RESULTS.md and the JSON files: H4a
   150/21 cells, H4b 21 of 21, H4c 1.000 and 0.999, N10, H3 primary lowest 0.917 [0.905, 0.929]
   (`x2_scifact.json`, n = 300), the three failing secondary cells, "at most 0.010" at T = 0.99
   (`x2_<store>.json`: 0.0015, 0.010, 0.0015, 0.0035, 0.006, 0.005, 0.0), 2,800 cells, H5 13 of
   16 and which three, 0.043, extension 8 of 12 and which four, SciFact 0.9175, 0.0995, 0.1415
   and 6.6x against 2.0x (`x3_scifact.json`), ColPali InfoVQA direction (RI-09), the list of
   materially changed candidate types, the registered tree hash `0b9523bd...` (equal to
   `git rev-parse 91564fd:src/optivision`), the different tree hash at HEAD (`8fc8891c...`).
10. **XH eligibility.** No vector of any model was produced in the diff.
    `tests/test_adapters_nemotron.py` replaces the loader with a 64-row stub model;
    `nvidia/nemotron-colembed-vl-8b-v2` is a registry string (unchanged "untested") and a test
    parameter; golden fixtures are seeded
    Gaussian arrays; examples, the benchmark and `testing` use `make_corpus`/`make_queries`; the
    adapters diff changes imports, adds `identity()` and `dimension` and one `load_adapter`
    branch for the loader `nemotron` (a loader family, not a preset), and leaves `MODELS`
    untouched; `presets/*.yaml` unchanged; no "8B" result appears in the diff (the two "8B"
    mentions in README and CHANGELOG exist at `91564fd`).
11. **Tamper table.** 60+ rows, each run twice (digest left alone: `digest` fails in all; digest
    recomputed: the table's `alone` and `full` columns), plus HMAC and documented-limit tests.
    Every row of CERTIFICATE_SPEC 4.1 has a case; the only adversary move that is missing is the
    rule rename (RI-02). The spec states plainly that the digest is not security, that HMAC is
    symmetric, and that forged evidence plus matching objects verifies without a key
    (`certificate.py:17-24`, spec lines 240-242, `QUICKSTART.md:195`, `LIMITATIONS.md` section 11);
    the "forged evidence and objects" qualifier is too narrow after RI-02.
12. **`certify()`.** No path from query data into the list (`CandidateSpace.default` ignores
    `dim`, the only input), the order, the target or the error level; target 0.97 is the
    `optimize()` default and error 0.05 the research's primary level, both constants; ABSTAIN
    returns the original corpus, exits 0, repr says "nothing certified: use the uncompressed
    index"; agreement is labelled in the repr, `report()`,
    CLI, certificate caveats; `CertifiedResult` prints a size ("x vs float32"), never a retention;
    the quickstart uses a fresh sample for each call and says "one pair per sample"; the examples
    print synthetic figures under the sentence "Nothing printed here is a result about any
    encoder or about retrieval quality".

## 4. Open questions for the lead

1. Custom rules: is `certified: true` ever to be reachable for code outside the library? If
   yes, who supplies and vouches for the statement text (a registered statement object with the
   rule's own text, never `STATEMENT`)? My answer in RI-01 is no.
2. XH: preregistration section 15.3 test 3 identifies the procedure by a commit that predates the
   first encoding. P0 as registered is the frozen research scripts; the package rule is
   bit-identical on this platform, but its tree hash differs (`FINAL_RESEARCH_BOUNDARY.md`
   section 10). Which implementation will the heldout-freeze addendum name, and will it record
   the package's parity run?
3. Should `certify()` default to a seeded permutation (SR-2) or keep "given" with a recorded
   caveat? Either must be explicit in the certificate.
4. The headline word "certified" (CLI, repr, `report()["certified"]`) is the strongest word in
   the product. Would you rename the field in the report to something that names the event (for
   example `hypothesis_rejected`) and keep "certified" for prose?
5. Unit norm: warn, refuse, or record only (RI-13)?
6. `docs/UNIVERSAL.md` is frozen and still says "A bit-level Hamming search would be faster"
   (`BACKEND_ARCHITECTURE.md:226` lists it). Is an erratum note in `docs/package/` enough?
7. Will the engineering benchmark numbers be re-run on a quiet machine before the document is
   committed? It says so itself. I reviewed wording only.

## 5. What I could not check

- The correctness of the citations (Westfall and Krishen 2001, Angelopoulos et al. 2021,
  Waudby-Smith and Ramdas, Ville, Hoeffding); no external source was consulted.
- Cross-platform and GPU bitwise parity: one machine, NumPy 2.4.6, Windows; the spec itself says
  bitwise equality across platforms is not promised, and the inclusive `<=` at the level can flip
  on another platform in a knife-edge case.
- That `collect_evidence` reproduces the research store builder on real vectors. It calls the same
  library functions (`score_candidate`, `per_query_metrics`); a comparison needs real encoder
  output, which this audit did not read and the package work must not produce.
- Whole-suite results (not run, as instructed); bodies of most tests; `bundle.py`, `codeid.py`,
  `state.py`, `tiered.py`, the Qdrant tests with a server.
- The mathematics of the derivation (see `STATISTICAL_REVIEW.md`), and the engineering benchmark
  numbers.
- Whether any text outside the diff (for example the `paper/` sources) repeats a sentence this
  audit asks to change.

Scratch scripts and raw output are in
`C:\Users\Lenovo\AppData\Local\Temp\claude\C--Users-Lenovo\1a357e2a-547c-499d-8080-bbcf9187bf4e\scratchpad\audit\`
and are not part of the repository.
