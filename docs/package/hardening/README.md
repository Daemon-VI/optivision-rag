# Hardening audit reports (2026-10-08)

The fourteen files in this directory are the reports of the release-hardening audit of the
package, as their authors wrote them. Each auditor worked on its own branch cut from
`7a6769a`.

How to read them:

- The "Status" column of a report is the state when that report was written. An item marked
  "PROPOSED" may have been implemented since, and some were decided differently.
- The final disposition of every finding (fixed and in which commit, documented, open, no
  action), together with the release-candidate gate and the steps of a release, is in
  [`../RELEASE_READINESS_REPORT.md`](../RELEASE_READINESS_REPORT.md), section 16.
- Commit hashes inside the reports are those of the audit branches. The readiness report gives
  the hashes of the same changes on `package-universal`.
- Paths under `ov-harden\scratch` point to scratch directories of the audit machine. They are
  not part of the repository.
- In `a14_research_boundary.md` the finding IDs read `ANN-0x`; the readiness report calls them
  `A14-0x`.

| file | subject |
|---|---|
| `a01_release_audit.md` | metadata, versions, wheel and sdist contents, CLI, CI and publish workflows, release steps |
| `a02_compat.md` | compatibility with v0.3.0: names, signatures, behaviour, file formats |
| `a03_certificate_attack.md` | edits of certificates and bundles at every verification level |
| `a04_statistical_safety.md` | the selection rule against its specification; attempts to obtain a wrong `certified: true` |
| `a05_ci_storage.md` | the eight tests that fail on CI: cause, proof, derived tolerance |
| `a06_imports.md` | what `import optivision` loads and whether it matters |
| `a07_api_ux.md` | the package used as a new user would |
| `a08_clean_install.md` | build, clean install, documented flows against the installed wheel |
| `a09_docs_execution.md` | every runnable example of the new documents, executed |
| `a10_performance.md` | where time and memory go; the benchmark document against its data |
| `a11_claims.md` | claim words and research numbers in the new documents |
| `a12_architecture.md` | import graph, layering, duplicated abstractions, interfaces |
| `a13_portability.md` | Python and NumPy versions, serialization, optional dependencies |
| `a14_research_boundary.md` | `FINAL_RESEARCH_BOUNDARY.md` against the frozen record and the code |

These are engineering audits of the package on synthetic data. They are not research results,
and no registered experiment was run or re-run for them.
