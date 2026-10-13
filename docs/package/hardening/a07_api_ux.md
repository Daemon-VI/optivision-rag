# a07 API/UX audit

## 1. Scope and what was executed
Read README.md, PACKAGE.md, QUICKSTART.md first, then used the package from scratch scripts
(`scratch/a07/t1..t7.py`, `cli/`): about 110 input cases through `certify` (shapes, dtypes, float64/16,
non-normalised, zero vectors, empty docs, ragged dims, 3-D array, tuples, lists of lists, generators, torch
fp32/bf16, NaN/inf, bad qrels, 22 bad scalar arguments), result introspection for select/abstain/fallback,
`save`, `verify`, `load_artifact`, retrieval from the result and from the loaded artifact (rankings equal),
and `optivision certify` / `verify` with about 20 CLI cases. Moments I had to read source: the CLI `.npz`
format (not in README/PACKAGE/QUICKSTART; only in API_DESIGN.md lines 51-59: `MultiVectorCorpus.save`).
Tests on the branch: test_api_certify, test_cli_certify, test_core_artifact_bundle, test_core_certificate:
367 passed after the first fix; api + cli 158 passed after the second. ruff clean.

## 2. Findings
| ID | Sev | Where | Problem / evidence | Fix | Status |
|---|---|---|---|---|---|
| A07-01 | HIGH | core/artifact/verification.py `_notes` | `verify()` on an ABSTAIN or FALLBACK record prints "Certificate verification: OK", `.ok` True, CLI "OK: no check failed", exit 0, with no hint that nothing (abstain) or the compressed index (fallback) is not certified. `load_artifact(fallback_dir)` returns a compressed index with `verification.ok` True. README advertises `verify("artifact/").ok` as the gate, so a user gating on `ok` would accept a not-certified index. | A report note (shown by `summary()`, the CLI and `to_dict()["notes"]`): ok means the record is intact, not that an index was certified. Select records unchanged (notes empty). | FIXED e23265c (ruff tidy in d41ac2f) |
| A07-02 | LOW | api.py `_warn_if_graded` | `certify(..., qrels=[...])` raised `AttributeError: 'list' object has no attribute 'items'` from inside the library. | Skip the graded check for non-mappings so `LabelReference` raises its TypeError naming `qrels`; test over 3 wrong types. | FIXED d41ac2f |
| A07-03 | MEDIUM | api.py / docs | Re-running `certify` with several targets on the same queries gives NO warning (three calls, 0.8/0.9/0.95: zero warnings, outcomes select/abstain/abstain). The docs warn (QUICKSTART s4, docstring), the code does not. | Proposal: process-local check on corpus + query identity and a UserWarning on a second `certify` with a different (target, error, space, seed). Needs a design decision. | PROPOSED |
| A07-04 | MEDIUM | docs | CLI input formats are not in README, PACKAGE or QUICKSTART. `--help` says "corpus .npz" but not how to make one. Only API_DESIGN.md shows `docs.save("docs.npz")`. QUICKSTART s6 points to `--help` only. | Add 3 lines to QUICKSTART s6 (`docs.save`, `queries.save`, qrels JSON). | PROPOSED |
| A07-05 | MEDIUM | CLI | `verify` exit codes are inconsistent and undocumented: missing path = 2 (usage error on stderr, empty stdout even with `--json`); tampered, unreadable or garbage = 1 (`--json` gives `{"ok": false, "error": ...}` for unreadable). | Document 0/1/2, or emit the JSON error object with exit 1 for a missing path. | PROPOSED |
| A07-06 | MEDIUM | verify output | "level reached: binding" is printed when only binding.pipeline/codes/bytes ran and binding.corpus/queries/reference were skipped, i.e. the certificate was never bound to the user's data. Reads as the strongest level. | Report `binding` only when corpus/queries/reference were supplied. | PROPOSED |
| A07-07 | MEDIUM | api / CLI | Mixed-up inputs run silently: `certify(docs, docs)`, CLI `certify queries.npz docs.npz` (swapped) and `certify docs.npz docs.npz` exit 0. Mismatched qrels ids (ints, wrong ids) give only "no query has a relevant document under this reference: there is nothing to test", with no hint about id types (documented ids are strings "0","1",... for lists). A partial id mismatch silently tests only the matched queries (5 of 80 in my test) with no warning; the count is only in `report()["n_queries_without_relevant_document"]`. | Warn when queries are dropped or corpus ids equal query ids; name the id types in the message. | PROPOSED |
| A07-08 | LOW | CLI | `certify docs.npz bad.json` (text file) reports "This file contains pickled (object) data ... allow_pickle": wrong diagnosis. | Say "not an .npz written by MultiVectorCorpus.save". | PROPOSED |
| A07-09 | LOW | reprs | `Artifact` repr dumps the whole VerificationReport (~4 KB); `Decision` repr prints all 40 p-values and the full pipeline mapping; `result.retriever()` has no repr. | Short reprs. | PROPOSED |
| A07-10 | LOW | discoverability | WHY it abstained: `result.report()["not_certified_note"]` and `decision.p_values` vs `decision.level`; the repr only says "nothing certified: use the uncompressed index". Next steps only in QUICKSTART s4. `result.pipeline` is None on ABSTAIN, so the README optimize() idiom `result.pipeline.label()` raises AttributeError. | Put the next step in the repr/report; add a smallest-p-value field. | PROPOSED |
| A07-11 | LOW | messages | "vectors must be floating point, got int32" / "must be 2-D ... got shape ()" name neither argument nor document index. A 3-D or single 2-D ndarray gets "expected a MultiVectorCorpus or a list of [n_i, d] arrays, got ndarray" without suggesting `list(array)`. `retriever.search([])` says "no documents" (means no queries). | Name argument/index; add hint. | PROPOSED |
| A07-12 | LOW | validation | Non-L2-normalised, all-zero and float16 vectors are accepted silently (a zero-vector corpus gives SELECT, certified True under agreement). PACKAGE.md says the research covers L2-normalised only; `certify` does not warn. | Optional warning on norms not 1 or 0. | PROPOSED |
| A07-13 | LOW | naming | `n_queries` vs `n_queries_given`; `store`/`compressed`/`corpus` on the result; CLI `--budget` vs API `query_budget`; `--given-order` vs `order_seed=None`; `verify` API `corpus=`/`reference=` vs CLI `--vectors`/`--qrels`. | none | NO ACTION (naming) |
| A07-14 | LOW | docs | Public parameters absent from the guide: `progress`, `created`, `overwrite`, `include_evidence`, `strict_code`. | docs | PROPOSED |

## 3. Checked and correct
Scalar arguments (target, error, metric, reference, order_seed, query_budget, fallback, encoder, created, rule,
space): clear, argument-named errors in all 22 cases. NaN/inf names the vector and document. Ragged and
mismatched dimensions refused. torch fp32/bf16, float64, mixed dtypes, tuples, lists of lists accepted.
Graded qrels, str values, reference/qrels conflicts, empty qrels refused clearly. Duplicate queries warn.
Fallback repr says "NOT certified"; abstain `retriever()` is exact over the original corpus. `save` refuses a
non-empty directory. Saved and loaded artifact give equal rankings. CLI: edited certificate.json exits 1 (fail
on artifact files, digest, replay.decision); flipped byte in codes.npz exits 1; `--json` parses; wrong queries
file fails binding.queries with exit 1. Empty documents score -inf and rank last.

## 4. Not checked
Signed certificates and `--key-file`; the `wide` space; legacy `--trust-pickle` caches; backends; the other guide
examples; any platform other than Windows/Python 3.11; behaviour with a real encoder.
