# a14: final research boundary review

## 1. Scope and what was executed

Reviewed `docs/package/FINAL_RESEARCH_BOUNDARY.md` (worktree `ov-harden/a14`, branch `harden/a14`) line by line against `reports/research/X_RESULTS.md`, `X1/x1_summary.json`, `X2/x2_summary.json`, `X3/x3_*.json` (python reads of the real JSON, not pointers), the preregistration sections 7, 12.6, 13, the Q10 evidence matrix, and the code and tests named below. Ran `tests/test_docs_examples.py` (18 passed). Compared the 11 byte-pinned files with base 91564fd by git blob id (all identical). No X1-X3 rerun, no XH, no encoding.

## 2. Number and verdict audit (document line numbers refer to the pre-edit file)

| doc line | claimed | source | source value | match |
|---|---|---|---|---|
| 66-68 | H4a CP1 21 pairs, CP2 150 cells | x1_summary.json H4a | CP1_holds, CP2_cells 150, 0 violations | yes |
| 69 | H4b 2x on 21 of 21 pairs | x1_summary H4b | 21 entries, min factor 2.29 | yes |
| 70-72 | H4c 1.000 six visual, 0.999 SciFact | x1_summary H4c by_pool | 1.0, 1.0, 0.999 | yes |
| 73-77 | N10: 26 of 37 | x1_summary N10 | 26 / 37 (diagnostic 26/40) | yes |
| 85-88 | H3 >= 0.90 all seven; lowest 0.917 [0.905,0.929] SciFact n=300 | x2_summary H3 primary | rate 0.9175 (1835/2000), ci [0.90457, 0.92919] | yes, but see ANN-01 |
| 89-92 | T=0.99 non-trivial fails 7; T=0.97 substantial fails DocVQA x3 + SciFact; T=0.99 substantial fails 7 | x2_summary secondary | rates 0.0-0.01; 0.1145/0.3345/0.403/0.2055 (fail); InfoVQA 0.9975/1.0/0.999 (pass); 0.99 substantial max 0.006 | yes |
| 96 | 2,800 cells, no confirmed failure | x2_summary | cells_checked 2800, confirmed_failures 0 | yes |
| 101-103 | H5 13 of 16; 3 rotations (ColPali DocVQA one, SciFact both) | X_RESULTS s3; x3_*.json pairs | T2a ColPali DocVQA holds=false, T2b true; SciFact both false | yes |
| 105 | largest labelled change 0.043 | x3_scifact T2b labels | 0.04303 | yes |
| 107-109 | SciFact 0.9175 -> 0.0995 / 0.1415; 6.6x vs 2.0x | x3_scifact procedure_outcome T=0.97 n=300 | 0.9175, 0.0995, 0.1415; 6.6149 -> 2.0 | yes |
| 110-112 | ColPali InfoVQA material both rotations, up and down | x3_colpali_infovqa | T2a 0.4975 -> 0.698; T2b 0.4975 -> 0.2275 | yes |
| 113-114 | extension 8 of 12; four failing are rotations | X_RESULTS s4; x3 jsons | ColPali InfoVQA T2a/T2b, ColQwen2 DocVQA T2a/T2b | yes |
| 140-146 | H1, H2 wording | prereg s7 | statement texts match | yes (criteria were missing, added) |
| 185 | eleven pinned files byte-identical | git rev-parse of blobs, base vs HEAD | 11 of 11 same | yes |

Labels (MEASURED / SIMULATED pool-supported) match X_RESULTS section 3 in every case.

## 3. Findings

| ID | Sev | Where | Problem | Status |
|---|---|---|---|---|
| ANN-01 | LOW | boundary s5 (old l.87-88) vs s6 (l.107) | SciFact rate printed as 0.917 in one place and 0.9175 in another. 0.9175 is exact (1835/2000); the registered reports print 0.917. No rounding error, but inconsistent presentation. | FIXED d236daa (both given, with the count) |
| ANN-02 | MEDIUM | boundary whole | Missing sections: consolidated basis-dependent list, plain "what certification means / does not mean", "what has not been tested", H1/H2 criteria, earlier research (Q/R series), 8B status. | FIXED d236daa (new sections 11-15, expanded 8, index table at top) |
| ANN-03 | LOW | old s10 | "enforced by a test" had no name; "binds it to a digest" named no function; "only library's own rules certified" named no check. | FIXED: tests `test_core_layering.py::test_package_does_not_import_research_scripts`, `test_x_harness.py::test_frozen_code_is_the_registered_code`, `corpus_identity` with tamper tests, `is_builtin_rule`/`statement_for` with three tests. No test asserts byte identity of the eleven files other than the harness hash check (which covers them). |
| ANN-04 | LOW | old s7 | "binds it to a digest of the corpus for that reason" overstates: the corpus check in `verify` is skipped unless the live corpus is passed; "certificates carry `basis_sensitive`" is None for an abstention. | FIXED (wording) |
| ANN-05 | LOW | old s9 item 3 | Certifying statement agreed with `STATEMENT` (rules.py, written into the certificate by `statement_for`) but omitted A1-A6/C1-C4 names and the "not externally reviewed" caveat was present only in the tail. Joint-probability, not-conditional, sample, finite batch, power: all present and matching. | FIXED (names added) |
| ANN-06 | LOW | PACKAGE.md "Install: Python 3.10 or newer", pyproject classifiers 3.10-3.13 | Verified platform is Python 3.11 only (brief rule 9; BUILD report: 3.10/3.12/3.13 not run). The boundary now says so; PACKAGE.md/README do not say what was verified. | PROPOSED (another agent edits README/PACKAGE.md): add "verified on Python 3.11, Windows" beside the install requirement |
| ANN-07 | NO ACTION | LIMITATIONS, BASIS_DEPENDENCE, CONCEPTS, SELECTION_AND_CERTIFICATION | Checked for contradictions with the boundary: 0.917, 0.010, 13/16, 8/12, 0.043, 0.9175/0.0995/0.1415, 26 of 37 all agree. "lowest 0.917" is the registered print. No edits needed. | NOT A DEFECT |

## 4. Not claimed because the documents do not say

The 8B encoder: the documents say only that a loader and pinned revision exist, it is XH candidate 9 in the preregistration (development family, "untested"), and it was not benchmarked. The boundary says exactly that. Nothing else was invented.

## 5. Could not check

Q1-Q9 raw numbers beyond the Q10 matrix rows cited (section 11 of the boundary cites the matrix, not the underlying files). External review of the derivation does not exist.
