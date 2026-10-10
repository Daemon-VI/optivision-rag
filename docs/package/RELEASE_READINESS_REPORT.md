# Release readiness report

Status: 2026-10-10, branch `package-universal`. The gate of section 2 ran on commit
`d44ee19420cda82ebd2593ca3e4447c366e8e90f`; this report was added in the commit after it and
changes no code. Base of the package work: `research-investigation` at
`91564fdb5bd941b439c9af9520a3fea589122320`. Start of the hardening pass:
`7a6769a6ac077e48ec9294761c024a8c957166e1`. Published release: v0.3.0 (`main` at
`4d520603469e179135ffd9df922b02d4504b4add`).

Nothing was pushed, tagged, published or released. `__version__` is still `0.3.0`, on purpose
(section 13). No registered experiment was run or re-run, XH was not run, and no file of the
research record was modified (section 11).

Every figure below is an ENGINEERING figure: a property of the code checked on synthetic data, on
one Windows laptop, or by comparison with the frozen research code. None is a research result.

## 1. Verdict

The code passes the release-candidate gate on the gate commit: no test fails, four tests fail as
expected (section 4), the wheel builds, installs into a clean environment and works from outside
the repository.

The branch is not ready to be tagged as it stands. What remains is not a code defect found by this
pass; it is the following.

1. **Three release steps were deliberately not taken** (section 13): the version is `0.3.0`, the
   same as the published package; `README.md` and `PACKAGE.md` (the PyPI long description) call
   the new layer "unreleased" and tell the reader to install a checkout, which is true today and
   false the moment a release exists; the changelog section is undated. They are mechanical and
   belong in one release commit.
2. **The branch has never run on the project's CI, on Linux or on macOS.** The publish workflow
   runs the test suite on Linux before it uploads. Eleven tests that would have failed there were
   found and fixed in this pass by forcing other BLAS kernels locally (sections 3 and 9), which
   lowers that risk and does not replace a Linux run.
3. **Owner decisions** (section 13): the version number; that a release from this branch merges
   the research branch into `main` (101 commits and the research record become public there); two
   pieces of wording; the default of the npm launcher.

Findings of the pass: 107 in all, none CRITICAL. Seven are HIGH: two code defects, both fixed (a
built-in rule instance could be given another `decide` and still be certified; `verify` said "OK"
for an abstention with no note), and five rows that are the two release steps of point 1 (the
version, and the "unreleased" wording), reported by two auditors. Section 16 lists every finding
with its disposition.

## 2. Release-candidate gate

Machine: 11th Gen Intel Core i7-1165G7, Windows 11 (10.0.26200). Interpreter of the full suite:
Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1 (the development environment of the repository).

| # | gate | what ran | result |
|---|---|---|---|
| 1 | full test suite | `OPTIVISION_FULL_PARITY=1 pytest -q` in the package worktree (default OpenBLAS kernel, the research data present) | 5,460 passed, 1 skipped, 4 xfailed, 0 failed (658 s) |
|   |   | the suite with `OPENBLAS_CORETYPE=Haswell`, in a detached checkout of the gate commit (research data as LFS pointers) | 5,446 passed, 15 skipped, 4 xfailed, 0 failed |
|   |   | the suite with `OPENBLAS_CORETYPE=Zen`, same checkout kind | 5,446 passed, 15 skipped, 4 xfailed, 0 failed |
| 2 | package build | `python -m build` on a `git archive` export of the gate commit; `twine check --strict`; a second wheel built from the sdist | wheel 346,201 bytes (sha256 `90e7134b19d9...`), sdist 302,378 bytes (sha256 `9afb7b2fa52f...`); `twine` PASSED for both; the wheel built from the sdist has the same file list |
| 3 | clean install | two fresh Python 3.11.9 environments: the wheel with the `merge` extra (SciPy), and the wheel alone | both installed |
| 4 | import smoke test | from a directory outside any checkout, `PYTHONPATH` unset | `optivision` is imported from `site-packages`; version `0.3.0`; the 51 names of `__all__` all resolve |
| 5 | CLI smoke test | `--version`; `--help` of all 15 subcommands; `certify` and `verify` end to end on files; the console script | pass. `verify` exits 0 on a verified artifact, 1 on an edited certificate, 2 on a missing path. Without SciPy, `certify` fails with a message that names `--allow-reduced`, and runs with that flag |
| 6 | documentation examples | `tests/test_docs_examples.py` (23 tests, inside gate 1: the guide pages, `API_DESIGN.md`, `MIGRATION.md`, the README and PACKAGE snippet, the documented command lines); the six scripts of `guide/examples/`, copied out of the repository and run against the installed wheel | pass; six of six scripts exit 0 |
| 7 | certificate tamper tests | `test_core_certificate_tamper.py` (431), `test_core_artifact_malformed.py` (21), `test_core_certificate.py` (241), `test_core_artifact_bundle.py` (99), inside gate 1; in the installed package, an edited target and an edited error level | pass; the edited records fail verification, `load_artifact` refuses the edited artifact, and a report on an edited record reads `level reached: none` |
| 8 | compatibility tests | `test_public_surface.py` (306), `test_golden_formats.py` (14), `test_api_compat.py` (33), inside gate 1; `optimize()` in the installed package | pass |
| 9 | lint used by CI | `ruff check src tests` (the command of `.github/workflows/ci.yml`) | all checks passed |
| 10 | package contents | the wheel and the sdist, by script | wheel: 108 entries, exactly the 102 tracked files of `src/optivision` plus the metadata; the four preset YAML files are the only data files; no tests, reports, scripts, documents, arrays or bytecode. sdist: 115 entries (`src`, `LICENSE`, `MANIFEST.in`, `PACKAGE.md`, `README.md`, `pyproject.toml`, `setup.cfg`, `PKG-INFO`) |
| + | other interpreters | the gate wheel reinstalled (no dependency change) into the scratch environments of the portability audit; the tests of the new layer (42 files) run from a copy of `tests/` outside the repository | Python 3.10.21, NumPy 1.26.4, SciPy 1.13.1: 4,471 passed, 51 skipped, 4 xfailed. Python 3.12.14, NumPy 2.0.2, SciPy 1.18.1: the same. Python 3.13.15, NumPy 2.5.3, SciPy 1.18.1: the same. Python 3.14.7 (5 files): 500 passed, 3 skipped. SciPy 1.10.1, the lowest allowed (5 files): 322 passed, 10 skipped |

The skips in the copies are tests that need files of the repository (the research data, the
documents); the run in the package worktree has them.

The gate was run twice. The first pass, on `eb059a9`, passed on the default kernel (5,453 passed,
1 skipped, 4 xfailed) and failed 3 cases under the Haswell kernel: one more test that asserted
bitwise equality of two scoring paths (L-03). The lead's review of the merged fixes, made while
that pass ran, found two defects in them (L-01, L-02). All three were fixed and the whole gate
was run again on `d44ee19`; the table shows the second run.

## 3. Test counts

| when | commit | passed | skipped | expected failures | failed |
|---|---|---|---|---|---|
| before the hardening pass (2026-10-07) | `7a6769a` | 5,407 | 1 | 4 | 0 |
| gate, first pass | `eb059a9` | 5,453 | 1 | 4 | 0 on the default kernel; 3 under Haswell |
| gate, final | `d44ee19` | 5,460 | 1 | 4 | 0 on the default, Haswell and Zen kernels |

The 53 added tests are regression tests of the fixes of this pass (malformed bundles 21, the
verification level, duplicate queries, the repeated-look and untested-query warnings, the
built-in rule check, the command-line flags and exit codes, the documented snippets, the storage
import, the array layout). The tests of `tests/test_storage.py` and one test of
`tests/test_core_retrieve.py` that asserted bitwise equality now assert a bound derived from the
operation count and the same ranking except between near-ties; a test checks that this bound
still rejects wrong columns, wrong queries and offsets.

The one skip is `tests/test_core_statistics.py:561`: it needs an extended-precision `long double`,
which this platform does not have.

Eleven tests in all were dependent on the BLAS kernel: the eight of `test_storage.py` that are red
on the repository's CI for the research branch, and three cases of one retrieval test that this
pass found by running the whole suite under an AVX2-class kernel. They pass on the development
machine (AVX-512 kernel) and fail with `OPENBLAS_CORETYPE=Haswell`. Section 9 has the cause.

## 4. Known expected failures

The suite does not "all pass": four tests are strict expected failures, and a claim about this
branch has to say so. They are in `tests/test_integration_layers.py::TestBadVectorInputs`:

- `test_queries[Pipeline.transform_queries|other_dimension]`: queries of another width fail inside
  NumPy's matrix product with a message that does not name the queries;
- `test_queries[Pipeline.transform_queries|raw_array]`, `test_documents[Pipeline.fit|raw_array]`,
  `test_documents[Pipeline.compress|raw_array]`: a raw array instead of a corpus raises
  `AttributeError`, not `TypeError` or `ValueError`.

They are intended, for one reason. All four are in `src/optivision/compose.py`, one of the eleven
files whose content hash the research harness pins (`CODE_BLOBS` in
`scripts/research/x_common.py`; `tests/test_x_harness.py::test_frozen_code_is_the_registered_code`
checks it, and the file's blob hash is the pinned `85b20d44...`). Editing the file would break the
equality between the code that ran the registered experiments and the code in the package. The
defects are the type and wording of an error on invalid input given directly to a v0.3.0 method;
valid input is not affected, and the new layer validates its input before it reaches those
methods. `xfail(strict=True)` means that a fix would make the suite fail, so the entries cannot
go stale silently. `docs/package/TEST_REPORT.md`, section 6, has the history.

## 5. Compatibility with v0.3.0

Audit A02 exported v0.3.0 and compared the two trees in separate processes: every public and
private name of the 45 modules of v0.3.0 (signatures, class hierarchies, dataclass fields,
constants), the stage and model registries, the help text of the 13 subcommands of v0.3.0, the
behaviour of 14 stage combinations, `calibrate` and `optimize` on seeded data, and artifacts
written by one tree and read by the other, in both directions.

- One break was found and fixed: `from optivision.storage import segment_reduce` raised
  `ImportError` (`e40ee4d`). It came from the research branch, not from the package work.
- Observable differences that are kept and documented in `MIGRATION.md`: `PageEncoderAdapter`
  raises on an `ids` list of the wrong length (v0.3.0 built an inconsistent corpus);
  `load_adapter` returns `NemotronAdapter` for the Nemotron ids; the stage registry gains
  `rotation`; the command line gains `certify` and `verify`; `optivision.__all__` has 51 names,
  the 40 of v0.3.0 and 11 new ones that load on first use; `python -m optivision.adapters`, a
  silent no-op before, is an error.
- Batched rescoring gives the same rankings as v0.3.0 except between scores closer than float32
  rounding, and scores equal to rounding; it is not bit-identical in general, and the documents
  no longer say so.
- `optimize()`, `calibrate()`, the old command lines and the v1 file formats are unchanged; the
  golden-file tests read files written by v0.3.0.

Not checked: real encoders, the Qdrant server path.

## 6. Integrity of certificates and artifacts

This is integrity checking, not security: without an HMAC key, anyone who can edit a certificate
can recompute its digest, and `CERTIFICATE_SPEC.md` says so. No cryptographic claim is made.

Audit A03 edited real certificates and bundles at every verification level, with and without a
key. Every edit of the specification's table was caught at the level the table names, including
every edit of the rule, the outcome and the `certified` flag with the digest recomputed; with a
key, every content edit and every manipulation of the signature failed; nothing on the load path
unpickles; manifest keys are never used as paths; 34 substitutions of live objects at the binding
level were caught, except corpus `metadata` and `attrs` and the order of labels, which the
identity does not cover by definition.

Fixed in this pass:

- `dim` and `stats` in the header of `codes.npz` were covered by no digest, and a changed `dim`
  loaded and verified; `load_artifact(verify=True)` checks them against the certificate and the
  pipeline (`7fc9a98`).
- Malformed bundle files raised raw exceptions instead of `ArtifactError` or
  `CertificateFormatError` (`7fc9a98`; 19 of the 21 new cases fail on the old code).
- `verify` on an ABSTAIN or FALLBACK record said "OK" with no note; the report now says that `ok`
  means the record is intact, not that an index was certified (`468416c`).
- `verify` reported `level reached: binding` when the checks that tie the certificate to the
  corpus, the queries and the reference had been skipped (`558aced`), and, after that fix, could
  still name a level above a failed one; now no level at or above a failed check is reported,
  and the value is `none` when an integrity check fails (`8984c9f`). The specification was
  brought in line (`8df90a5`).
- A Fortran-ordered input array gave the same array identity and a different certificate digest;
  the adapters make their items C-contiguous (`5a97aae`).

Limits that stay, all stated in the specification: a certificate issued honestly from fabricated
evidence, or made from a hand-built decision, verifies (the replay catches an inconsistent one);
unknown keys inside nested objects are not rejected; the digest covers the recorded Python, NumPy
and SciPy versions, so two environments give different digests for the same decision; a corpus
built directly from Fortran-ordered arrays, not through an adapter, can still give a different
digest (the constructor is in a pinned file).

## 7. Statistical safety

Audit A04 checked the implementation against `METHOD_C_SPEC.md`, `STATISTICAL_REVIEW.md` and
`RESEARCH_INTEGRITY_AUDIT.md` with its own code. The mathematics of the rule was not changed.

- p-values: equal bit for bit to the frozen research function and to an independent
  re-derivation on 4,000 random problems, edge cases included; the full rule matched independent
  decision code on 600 problems for each of the two rules. The gate reproduces the stored
  selections of the research record (`OPTIVISION_FULL_PARITY=1`).
- False selections: in 13 Monte Carlo configurations (20,000 or 100,000 replicates each) the
  highest rate was 0.0337 at an error level of 0.05 and 0.1063 at 0.20, and every upper
  confidence bound was below the level. This shows no violation in the configurations tried; it
  does not show that the bound is tight, and it does not replace the derivation, which is the
  project's own and has not been externally reviewed.
- The order assumption is load-bearing, as documented: queries sorted by outcome gave a
  false-selection rate of 100%, and reporting the best of 10 seeds gave 0.123 at a level of 0.05.
  `certify` and the command line shuffle with a recorded seed by default.

The property that the package never reports `certified: true` for a rule it cannot justify:

- One hole was found and fixed: a built-in rule instance carrying an instance attribute (a
  replaced `decide`, a shadowed `config`, a reassigned class) was still treated as built in
  (`026f1bb`). The audit's table of attempts has 19 numbered entries with sub-cases; the three
  that were wrongly certified were all of this kind, and the others (subclasses, same-name
  classes, a custom rule declaring itself certifying, edited or forged records, the fallback,
  bad inputs) were not.
  Patching the library's own classes in the running process is outside what the check can defend.
- Exact duplicate queries: `certify` used to return `certified: true` together with a caveat
  saying that the error level "does not apply". It now refuses them unless the caller passes
  `allow_duplicate_queries=True`, which asserts that the repeats are independent draws; the
  certificate records the assertion and the caveat is conditional (`371f248`).
- A FALLBACK certificate's statement now begins by saying that it was not established for the
  recorded pipeline (`d852000`).
- Looks: `certify` warns when the same documents and queries are used again in one process with
  another target, error level, order, candidate list, metric, reference or rule (`cb40593`,
  `1ea9fa4`). It cannot see other processes, and the obligation of one look stays with the user.
- `certify` warns with the counts when labels match only part of the queries, so a sample does
  not shrink without a word (`218886b`).

Not covered, as before: sampling without replacement, dependent queries, near-duplicates.

## 8. Performance

Audit A10 found no performance defect in the library: evidence collection is 95 to 100% of a
`certify()` call, the uncompressed reference ranking is computed once, each candidate is scored
once, nothing is quadratic by accident and no full-corpus float64 copy is made.

It found one unsupported comparison in the benchmark document (the memory of the NumPy backend
against the legacy loop was an artefact of measurement order). The script now measures both from
a cold cache, the benchmark was run again on 2026-10-10 with nothing else running, and
`ENGINEERING_BENCHMARKS.md` was regenerated from it (`22ec6e1`, `562a3ad`, `eb059a9`). From that
run, on synthetic data on one laptop:

- the two backend paths peak at 14.4 MB and 14.6 MB (d=128) and at 45.9 MB and 46 MB (d=1024);
- the retrieval facades took 1.01 to 1.03 times the direct calls at d=128 and 1 to 1.04 times at
  d=1024; the seam's own parts, measured alone, were 0.4% and 0.3% of the direct call. The
  processor changed speed state during the run (the probe ranged 3.27x), so absolute times of
  this run are not comparable with the earlier one, and the document says so;
- `certify()` took 4.6 s for 40 candidates, 200 documents and 200 queries, evidence collection
  99.6% of it; its traced memory peak was 86.2 MB, 18.8 times the float32 corpus. That multiple
  belongs to those sizes. `guide/LIMITATIONS.md` gives it beside the audit's 11.4 and 11.1 times
  for smaller corpora and says what the peak is made of.

The compute paths that existed before the package layer were not re-measured: the listed files
are byte-identical to the base commit except `storage.py`, which differs in an import and a
docstring only (checked by comparing the two syntax trees when the document is generated).

Nothing was optimised. Not measured: large corpora, real encoder output, a GPU.

## 9. Portability

What ran, all on one Windows machine: the full suite on Python 3.11.9; the tests of the new layer
against the installed gate wheel on Python 3.10.21 (NumPy 1.26.4, the lowest allowed), 3.12.14
(NumPy 2.0.2) and 3.13.15 (NumPy 2.5.3); smoke subsets on Python 3.14.7 and on SciPy 1.10.1, the
lowest allowed. Audit A13 found the seeded permutation, the array digests and the canonical JSON
bytes identical across NumPy 1.26.4, 2.0.2, 2.4.6 and 2.5.3, and a bundle saved by each of three
interpreters verified by all three.

Not tested: Linux, macOS, big-endian machines, GPUs, a Qdrant server, real encoder output through
the new layer. `README.md` and `PACKAGE.md` carry this statement beside the install requirement.

BLAS kernels. Two scoring paths that form matrix products of different shapes (batched against
per query; one query against a batch) get different kernels from BLAS, which order the additions
inside a dot product differently, so their scores differ in the last bits on some processors.
Audit A05 reproduced the eight failures of the repository's CI exactly by forcing
`OPENBLAS_CORETYPE=Haswell`, derived the bound `2 (gamma_d + gamma_(T-1)) B sum|q_t|` from the
operation count, and replaced the bitwise assertions with it and with a ranking check that
allows swaps only between scores inside twice the bound (`e882d82`). The production code is
correct and unchanged; making it bitwise independent of batching would mean giving up BLAS. The
gate ran the whole suite under the Haswell and Zen kernels as well as the default one.

`qdrant-client`, `pillow` and `pypdfium2` are declared dependencies, and `import optivision` loads
PIL and yaml (audit A06: both are hard dependencies, so this cannot break an install; removing
them from the import needs the lazy top-level package that the design deferred). What is optional
is using Qdrant: nothing imports it until a Qdrant index or backend is constructed.

## 10. Documentation and claims

- Audit A09 ran 79 fenced blocks: the six example scripts and the 29 Python blocks of the guide
  and design pages ran with no failure, the shown outputs matched, and the command lines were run
  for real. It corrected two statements and added four tests so that the design pages and the
  README snippet run under the suite (`ba2eb19`).
- Audit A11 checked every occurrence of the claim words (universal, model-agnostic, guaranteed,
  lossless, faster, first, novel, scalable, production-ready and others) in the new documents and
  user-visible strings, and every research number against the stored result files. No number was
  wrong and no text implies that transfer to held-out encoders was shown. Three wordings were
  corrected (`3757c5b`): "by 3 to 14" is "by up to 14"; a sentence saying that no valid method
  can certify a target of 0.99 with 225 queries was false and was replaced by the reviewed
  wording; "model-agnostic interface" is defined where it first appears.
- Audit A14 checked every number, count, verdict and evidence label of
  `FINAL_RESEARCH_BOUNDARY.md` against the frozen record (all match) and added the sections that
  were missing (`842c8e2`).
- Still true until a release and false after it: the "unreleased" wording (section 13).

## 11. Research boundary

- `git diff 91564fd d44ee19` is empty for `reports/research`, `reports/universal`,
  `scripts/research`, `scripts/universal_study`, `paper`, `docs/UNIVERSAL.md`, `docs/RESULTS.md`
  and `docs/IMPROVEMENTS.md`, and for the eleven pinned library files.
- `main` is at `4d52060` and equals `origin/main`; `research-investigation` is at `91564fd`.
- X1, X2 and X3 were not re-run. XH was not registered and not run; transfer to held-out
  encoders is not demonstrated. H5 failed: the compression that is certified depends on the
  coordinate basis. The interface is the same for every encoder; compression quality is not.
- No model was encoded and no real encoder output was compressed or scored, so no encoder lost
  eligibility for XH.
- The registered runs recorded the library tree `0b9523bd...`. This branch has another tree. A
  registered experiment on this branch would need a new dated amendment.
- `FINAL_RESEARCH_BOUNDARY.md` states what the package provides, what the research established,
  what depends on the model and on the basis, what certification means and does not mean, what
  XH would test and what has not been tested.

## 12. Unresolved issues

None is a release blocker. IDs refer to section 16 and to the reports in `hardening/`.

Usability of the new API and command line:
- swapped or identical document and query inputs are accepted (A07-07, in part);
- a text file given where an `.npz` is expected gets a misleading diagnosis (A07-08); some error
  messages do not name the argument (A07-11); the representations of `Artifact` and `Decision`
  are long, and the reason for an abstention is not in the representation (A07-09, A07-10);
- vectors that are not L2-normalised are accepted without a runtime warning; the certificate
  records the norms and a caveat (A07-12);
- five public parameters are not in the guide (A07-14);
- the refusal of duplicate queries is raised after evidence collection, so a large run pays for
  the collection before it fails (L-06).

Behaviour of v0.3.0, unchanged:
- without the `vlm` or `corpus` extras, the commands that need torch, reportlab or datasets end
  in a raw `ModuleNotFoundError` (A08-04, A08-05); the `app` extra installs Streamlit, which
  nothing in the wheel imports (A08-06); help text prints a dash that the Windows console shows
  as a replacement character (A08-10);
- the four expected failures of section 4.

Design decisions that were deferred, not defects:
- `import optivision` loads PIL and yaml; the dependency split (A06, A13-02);
- `certify` needs SciPy for the default candidate list, so `pip install optivision-rag` alone
  does not run the quickstart; the documents give the install line, and the reduced list is one
  argument or one flag away (A08-03, A06-04);
- two exception types for the same invalid-vector failure (A12-03); `optivision.adapters` has no
  `__all__` (A12-09); a 16 MiB read buffer for file hashes (A10-02); messages that cite
  repository paths the wheel does not contain (A08-07); the message for big-endian input
  (A13-05).

Research, outside the scope of a software release: XH; external review of the derivation;
sampling without replacement; anything outside L2-normalised MaxSim.

## 13. Release blockers, and the steps of a release

Blockers, in the order they would be cleared:

1. Owner decisions.
   - The version (section 14).
   - The merge. At the gate commit `package-universal` is 101 commits ahead of `main`: 52 of the
     research branch and 49 of the package work. A release from it publishes the research record on `main`. The LFS
     payload is small (127 files, 8.1 MB).
   - The summary line of the package on PyPI and npm, "Extreme Token Compression for
     Vision-Language Models", which describes the original recipe (A01-07, A11-07).
   - Whether the npm launcher should install the `merge` extra by default (A01-04).
2. One release commit: `__version__` in `src/optivision/__init__.py` and `version` in
   `npm/package.json`, identical; the two tests that hold the literal
   (`tests/test_api_compat.py`, `tests/test_core_certificate.py`); the changelog heading with its
   date, without the branch names and without "nothing below is released"; in `README.md` and
   `PACKAGE.md`, the "unreleased" and "install a checkout" wording replaced by
   `pip install "optivision-rag[merge]"`, the document paths of `PACKAGE.md` turned into absolute
   links (relative paths do not resolve on PyPI), "the latest release, 0.3.0" corrected; the
   status lines of the documents that say `__version__` stays `0.3.0`; the merge step added to
   `docs/RELEASING.md`.
3. The gate of section 2 again on that commit.
4. Push, and a green CI run on Linux (Python 3.10 and 3.13) and on the two launcher jobs. This
   is the first run of this code on Linux. `publish.yml` does not run ruff or the launcher test
   itself, so release only from a commit whose CI run is green (A01-06).
5. The trusted publishers of PyPI and npm and the `pypi` environment exist (not verifiable from
   here).
6. A GitHub release with a new tag on the merge commit; nothing publishes on a plain tag push.
7. From a clean environment: `pip install "optivision-rag[merge]"`, `optivision --version`, a
   `certify` and `verify` run, `npx optivision-rag@<version> --version`.

Until step 2 a build of this branch is called `0.3.0` and differs from the published `0.3.0`:
`pip install` of it over the published package does nothing, and PyPI would refuse the upload.

## 14. Recommended version

`0.4.0`.

- The change is additive: 11 new top-level names, two commands, new subpackages; the 40 names,
  the import paths, the commands and the file formats of v0.3.0 are kept, and the compatibility
  audit found one break, which is fixed.
- It is more than a patch: a second entry point (`certify`), a certificate format and an
  artifact format are new public surface, and three observable behaviours changed (section 5).
- The project is below 1.0 and the classifier is Beta, so a minor version is where new surface
  and documented behaviour changes go. `1.0.0` would promise a stability of the new interfaces
  that one hardening pass, on one machine and with no outside user, does not support.
- A release candidate cannot go through `publish.yml` (it compares the tag with the literal
  version, and the npm and Python version syntaxes of a pre-release differ). Section 13, step 4,
  is the rehearsal instead.

The version was not bumped in this pass, because the blockers of section 13 are not resolved.

## 15. Commits of the hardening pass

`7a6769a..d44ee19`, 31 commits, oldest first (62 files changed, 4,001 insertions, 1,616
deletions; most of the deletions are the regenerated benchmark record):

```
842c8e2 docs: complete the research boundary (basis dependence, meaning of certification, untested list, XH criteria, enforcing tests)
3757c5b Wording: define model-agnostic at first use, correct looks-inflation range and power statement, soften proof wording
ba2eb19 Docs: say 51 names in __all__ (40 from v0.3.0 plus 11), int4(mean) label; test the design pages, the documented CLI and the README certify snippets
e882d82 Rescore tests: compare batched and per-query scores within a derived float32 bound, not bitwise
e40ee4d Keep segment_reduce importable from optivision.storage as in v0.3.0
60ee426 Format the storage import for ruff
7fc9a98 Artifact loader: clean errors for malformed bundle files, deep JSON; check codes.npz dim and stats against the certificate
026f1bb fix(core): a built-in rule instance with instance attributes is not built in
468416c verify: note that ok does not mean a certified index for ABSTAIN and FALLBACK records
1aed6f6 certify: refuse qrels of the wrong type by name; parenthesise note strings
5a97aae adapters: make array items C-contiguous so layout does not change the certificate digest
371f248 certify: refuse exact duplicate queries unless allow_duplicate_queries=True; record the caller's assertion and make the caveat conditional
d852000 certificate: a fallback record's statement begins by saying it was not established for the recorded pipeline
558aced verify: report the highest level whose checks ran and passed; binding needs corpus, queries and reference; list skipped checks
2a45929 cli verify: exit codes 0/1/2 stated in --help; unreadable supplied files exit 1 with a JSON object
b2ca721 cli certify: --allow-reduced selects the reduced candidate list; the ImportError text names the flag
74ecfeb TwoTierRetriever: refuse a raw Backend as the first stage, naming BackendRetriever
22ec6e1 package_bench: measure numpy backend peak memory from a cold cache for both paths; add the warm figure
cb40593 certify: warn when the same documents and queries are looked at again with a different target, error, order or list
76956dc Docs: scope the bit-identical rescoring claim, state storage.py differences, certificate spec corrections, MIGRATION differences, pinned xfails, docstring accuracy
9f4f6d3 Docs: tested-platform statement, Qdrant dependency wording, SciPy for certify, CLI input files and exit codes
142c5e4 Docs: duplicate-query refusal, repeated-look warning, verify levels, memory and layout limits; merge the Unreleased changelog sections
8bf9751 Add the release-hardening audit reports (14 audits)
562a3ad package_bench: generate the unchanged-paths paragraph from a per-file comparison; state the certify() memory peak from the measured record
218886b certify: warn with the counts when queries without a relevant document are left untested; name the likely cause when labels match no query
eb059a9 Refresh the engineering benchmarks (cold-cache backend memory, certify memory peak); align the memory note in the limitations guide
8984c9f verify: no level at or above a failed check is reported; a skipped level does not stop the count
1ea9fa4 certify: the repeated-look warning also covers a changed metric, reference or rule
8df90a5 Docs: the verification level rule in the certificate spec and the guide; what the repeated-look warning covers
0137ba6 Retrieval test: a query alone or in a batch agrees within the derived float32 bound, not bitwise (kernel dependent)
d44ee19 Changelog: the retrieval equivalence test uses the derived bound
```

The package work as a whole is `91564fd..d44ee19`: 49 commits, 174 files changed.

## 16. Triage of every finding

Fourteen auditors worked on separate branches from `7a6769a`; their reports, as written, are in
`hardening/`. The lead merged the fixes, reviewed them (rows `L-`), decided the open ones and
ran the gate. "FIXED" names the commit on `package-universal`. "OPEN (release step)" is cleared
by section 13, step 2. Evidence, the proposed fix and the regression risk of each row are in
its report.

Findings: CRITICAL 0, HIGH 7, MEDIUM 28, LOW 55, NO ACTION 17 (107 in all).

| ID | severity | source | where | problem | disposition | note |
|---|---|---|---|---|---|---|
| A01-01 | HIGH | A01 | PACKAGE.md:50-53 and the "Documentation, in the repository checkout ... | PACKAGE.md is the PyPI long description (`readme = "PACKAGE.md"`). At release it would say the certify/core feature is "unreleased", "None of this is in a release", `__version__` is "still ... | OPEN (release step) | true until a release exists; rewrite in the release commit |
| A01-02 | HIGH | A01 | README.md:130-133, 165, 535, 545 | Same false statements in the GitHub README ("(unreleased)", "None of this is in a release", "install a clone of this branch", tree comments "unreleased core"). | OPEN (release step) | as A01-01 |
| A01-03 | HIGH | A01 | `src/optivision/__init__.py:54`, `npm/package.json:3` | Version is 0.3.0 everywhere. A branch build is labelled 0.3.0 with different contents; PyPI refuses a second 0.3.0 and pip will not replace an installed 0.3.0 (known: D18 ... | OPEN (release step) | bump `__init__.py` and `npm/package.json` together, update the two tests |
| A04-01 | HIGH | A04 | `core/selection/rules.py:306-314` (`is_builtin_rule`) | The "built in" test compared only `type(rule)`. A built-in instance with an instance attribute shadows the class method, because functions are non-data descriptors: `r = ... | FIXED 026f1bb |  |
| A07-01 | HIGH | A07 | core/artifact/verification.py `_notes` | `verify()` on an ABSTAIN or FALLBACK record prints "Certificate verification: OK", `.ok` True, CLI "OK: no check failed", exit 0, with no hint that nothing (abstain) or the compressed index ... | FIXED 468416c |  |
| A08-01 | HIGH | A08 | `PACKAGE.md` (wheel long description, `readme = "PACKAGE.md"`), lines ... | The text that PyPI will show for the built wheel says the certified-selection layer is "unreleased", "None of this is in a release: the latest release, 0.3.0 ... does not contain it, and ... | OPEN (release step) | = A01-01 |
| A08-02 | HIGH | A08 | `src/optivision/__init__.py` `__version__ = "0.3.0"` vs PyPI | The built wheel is called 0.3.0, the same version as the published one, but is a different file (`cmp` differs) with 51 vs 40 exported names. `pip install` of it over published 0.3.0 ... | OPEN (release step) | = A01-03 |
| A01-04 | MEDIUM | A01 | `npm/bin/optivision.js:94` (default `OPTIVISION_EXTRAS=corpus`) ... | `npx optivision-rag certify ...` installs without SciPy, so the default candidate list raises ImportError (clear message, but it says `pip install 'optivision-rag[merge]'`, which a user of ... | DOCUMENTED 9f4f6d3 | npm/README states `OPTIVISION_EXTRAS=corpus,merge`; launcher default unchanged (owner decision) |
| A01-05 | MEDIUM | A01 | CHANGELOG.md:3 and :84 | Two sections both titled "Unreleased" (`package-universal` and `research-investigation`), both naming branches. A 0.4.0 entry must merge them under one dated heading and drop "`__version__` ... | FIXED 142c5e4 | one Unreleased section; dating it is a release step |
| A01-06 | MEDIUM | A01 | `.github/workflows/publish.yml` | Publish-on-release does not run ruff or the npm launcher smoke test (only CI does, on push). It runs the full `pytest -q`, builds, and uploads, so publishing is not blocked by lint or the ... | OPEN (owner) | release only from a commit with green CI; optionally add ruff to publish.yml |
| A03-01 | MEDIUM | A03 | `bundle.py` `_states_from_file`, `_load_deployed`, `_load_evidence` ... | Edited files that the spec says raise `ArtifactError` (section 7: "raises `ArtifactError` for every unreadable file") raised raw `AttributeError`, `KeyError`, `TypeError` or ... | FIXED 7fc9a98 |  |
| A03-02 | MEDIUM | A03 | `bundle.py` `_load_deployed` | `dim` and `stats` in the `codes.npz` header are covered by no digest. With the manifest refreshed, `dim` 12 -> 16 or 99 loaded, and `load_artifact` and `verify` reported OK, although `dim` ... | FIXED 7fc9a98 |  |
| A04-03 | MEDIUM | A04 | `core/artifact/certificate.py:378-406` (`create_certificate`) ... | A `Decision` is not unforgeable and `create_certificate` does not replay `select` on the evidence it is given. A decision rebuilt with `Decision.from_dict` (built-in rule record pasted ... | NO CHANGE (documented limit) | CERTIFICATE_SPEC 4.1: a record made from a forged decision or evidence verifies at level 1; the replay catches an inconsistent one |
| A04-04 | MEDIUM | A04 | `Evidence` constructor, `select`, `create_certificate` | Evidence built by hand (not collected) on a real corpus and queries gives `certified=True`, and `verify_certificate(cert, evidence=fake, corpus, queries, reference, pipeline, compressed)` ... | NO CHANGE (documented limit) | CERTIFICATE_SPEC 4.1: a certificate issued honestly from forged evidence verifies |
| A04-05 | MEDIUM | A04 | `api.certify`, `record.caveats_for` | With 120 exact duplicate queries added (240 tested), `certify` still returns `certified=True`. The library knows A1 is violated (it counts duplicates), warns and adds a caveat saying the ... | FIXED 371f248 | `certify` refuses exact duplicate queries unless `allow_duplicate_queries=True` |
| A05-01 | MEDIUM | A05 | tests/test_storage.py:120-226 | Eight tests asserted bitwise equality of batched and per-query rescoring; they fail on CI (AVX2-class BLAS kernel) and pass on the AVX-512 development machine | FIXED e882d82 | derived float32 bound; the same fix for one more test in 0137ba6 |
| A07-03 | MEDIUM | A07 | api.py / docs | Re-running `certify` with several targets on the same queries gives NO warning (three calls, 0.8/0.9/0.95: zero warnings, outcomes select/abstain/abstain). The docs warn (QUICKSTART s4 ... | FIXED cb40593, 1ea9fa4 | UserWarning; also for a changed metric, reference or rule |
| A07-04 | MEDIUM | A07 | docs | CLI input formats are not in README, PACKAGE or QUICKSTART. `--help` says "corpus .npz" but not how to make one. Only API_DESIGN.md shows `docs.save("docs.npz")`. QUICKSTART s6 points to ... | FIXED 9f4f6d3 |  |
| A07-05 | MEDIUM | A07 | CLI | `verify` exit codes are inconsistent and undocumented: missing path = 2 (usage error on stderr, empty stdout even with `--json`); tampered, unreadable or garbage = 1 (`--json` gives `{"ok" ... | FIXED 2a45929 |  |
| A07-06 | MEDIUM | A07 | verify output | "level reached: binding" is printed when only binding.pipeline/codes/bytes ran and binding.corpus/queries/reference were skipped, i.e. the certificate was never bound to the user's data. ... | FIXED 558aced, 8984c9f | and no level at or above a failed check |
| A07-07 | MEDIUM | A07 | api / CLI | Mixed-up inputs run silently: `certify(docs, docs)`, CLI `certify queries.npz docs.npz` (swapped) and `certify docs.npz docs.npz` exit 0. Mismatched qrels ids (ints, wrong ids) give only ... | PARTLY FIXED 218886b | warning when labelled ids match only part of the queries; swapped or identical inputs are still accepted |
| A08-03 | MEDIUM | A08 | `docs/package/guide/QUICKSTART.md`, `README.md` certify block ... | Headline workflow `certify` does not run on the plain `pip install optivision-rag`; it needs `[merge]` (SciPy). The error is clear and actionable and the docs say so, but: the CLI ... | FIXED b2ca721, 9f4f6d3 | CLI `--allow-reduced`; the default list still needs the `merge` extra |
| A08-04 | MEDIUM | A08 | `src/optivision/encoders/colvlm.py:55,348` | Without the `vlm` extra, `optivision index ... -c colsmol` shows a raw traceback ending `ModuleNotFoundError: No module named 'torch'`, with no hint to install `optivision-rag[vlm]`. ... | OPEN | behaviour of v0.3.0, unchanged |
| A09-01 | MEDIUM | A09 | API_DESIGN.md:234, MIGRATION.md:6 | Both say the "40 names of `optivision.__all__`" are unchanged; `len(optivision.__all__)` is 51 (40 v0.3.0 + 11 new). MIGRATION section 1 itself says 51 in all, so the page contradicted ... | FIXED ba2eb19 |  |
| A09-03 | MEDIUM | A09 | API_DESIGN.md:7 "Every Python block below is run as written" ... | `tests/test_docs_examples.py` only ran the guide pages and examples; API_DESIGN and MIGRATION blocks, the documented CLI commands and the README/PACKAGE `certify` snippet were not run by ... | FIXED ba2eb19 |  |
| A10-01 | MEDIUM | A10 | `scripts/engineering/package_bench.py:296-313` (`backend_costs`); doc ... | The peak-memory column compares unequal states: legacy `NumpyIndex.search` 14.4 MB (d=128) / 45.9 MB (d=1024) versus `NumpyBackend.search_many` 1 MB / 0.64 MB. | FIXED 22ec6e1, eb059a9 | cold-cache measurement; document regenerated |
| A11-01 | MEDIUM | A11 | `README.md:17`, `PACKAGE.md:3` | "model-agnostic interface" is the first thing a reader (and the PyPI page) sees; it is defined 110+ lines later. | FIXED 3757c5b |  |
| A11-03 | MEDIUM | A11 | `src/optivision/api.py:89` (user-visible abstain note) ... | "such looks multiplied the false-selection probability by 3 to 14". STATISTICAL_REVIEW 7.7 reports factors from 1.4 (2 seeds) and 2.4 (5 seeds) up to 14 (100 seeds); list editing 3.5 to 9 ... | FIXED 3757c5b |  |
| A11-04 | MEDIUM | A11 | `METHOD_C_SPEC.md:334`, `:267` | "No valid method can certify T = 0.99 with <= 225 queries" is literally false (SR-7 / W12: a valid test can reject with probability up to 0.48 at n = 225). The review's replacement wording ... | FIXED 3757c5b |  |
| A13-01 | MEDIUM | A13 | `adapters/arrays.py` (`host.append`) | A Fortran-ordered item yields an F-contiguous corpus (`np.concatenate` keeps K-order). `array_digest` treats it as the same data, but float reductions differ in the last bit ... | FIXED 5a97aae | a corpus built directly from Fortran-ordered arrays (pinned `representation.py`) is a documented limit |
| A13-02 | MEDIUM | A13 | `pyproject.toml` dependencies | `qdrant-client`, `pillow`, `pypdfium2`, `pyyaml`, `typer`, `rich` are hard dependencies, and `import optivision` imports PIL. "Qdrant optional" holds only at import level: with ... | DOCUMENTED 9f4f6d3 | wording corrected; the dependency split is an owner decision |
| A14-02 | MEDIUM | A14 | boundary whole | Missing sections: consolidated basis-dependent list, plain "what certification means / does not mean", "what has not been tested", H1/H2 criteria, earlier research (Q/R series), 8B status. | FIXED 842c8e2 |  |
| L-01 | MEDIUM | LEAD | core/artifact/report.py `VerificationReport.level` | After 558aced the reported level could name a level above a failed one (integrity failed, replay passed: `replay`) | FIXED 8984c9f | found in the lead's review of the merged fixes |
| L-02 | MEDIUM | LEAD | api.py `_warn_if_repeated_look` | The repeated-look warning did not see a changed metric, reference or rule on the same queries | FIXED 1ea9fa4 |  |
| L-03 | MEDIUM | LEAD | tests/test_core_retrieve.py (T9) | One more bitwise assertion that depends on the BLAS kernel: 3 cases fail with OPENBLAS_CORETYPE=Haswell and would fail on CI | FIXED 0137ba6 | found by running the whole suite under the AVX2 kernel |
| A01-07 | LOW | A01 | pyproject.toml `description`, npm/package.json `description` | "Extreme Token Compression for Vision-Language Models" describes the original recipe; PACKAGE.md now leads with "Quality-constrained multi-vector compression behind a model-agnostic ... | OPEN (owner) | wording of the PyPI/npm summary line |
| A01-08 | LOW | A01 | docs/RELEASING.md step 3 | "wait for CI to pass on `main`": this work is on `package-universal`, not merged into `main`; the merge step is not stated. | OPEN (release step) | RELEASING.md needs the merge-to-main step |
| A02-01 | LOW | A02 | `src/optivision/storage.py:33` | `from optivision.storage import segment_reduce` worked in v0.3.0 (incidental re-export, no `__all__`) and raised ImportError on the branch: the batched `rescore` dropped the import. The ... | FIXED e40ee4d, 60ee426 |  |
| A02-02 | LOW | A02 | `src/optivision/storage.py` `ExactIndex.rescore` (unreleased batched ... | Not bit-identical to v0.3.0 in general. Docs claim "bit-identical" (CHANGELOG:92-93, test name "bitwise"), true on the ViDoRe data and in the test, but on my synthetic data 39 of 114 arrays ... | FIXED 76956dc | wording: same rankings, scores equal to float32 rounding |
| A02-03 | LOW | A02 | `adapters/__init__.py:222-227` (`PageEncoderAdapter.encode_documents`) | Now raises `ValueError` when `len(ids) != number of pages`. v0.3.0 silently produced a corpus with `len(corpus)=2` and `ids` of length 1, 3 or 0 (inconsistent). Code that passed a ... | DOCUMENTED 76956dc | MIGRATION.md |
| A02-04 | LOW | A02 | `load_adapter` | For `nvidia/nemotron-colembed-vl-*` (registry loader `"nemotron"`) v0.3.0 fell through to `SentenceTransformersAdapter`; the branch now returns ... | DOCUMENTED 76956dc | MIGRATION.md |
| A03-03 | LOW | A03 | `verification.py` schema check; spec section 2 | Only top-level fields are closed. An unknown key added inside `result`, `queries`, `space`, `code`, `package`, `evidence`, `reference`, `sizes`, `norms`, `candidate`, `sample`, `signature` ... | DOCUMENTED 76956dc | spec note: the schema check closes top-level fields only |
| A03-04 | LOW | A03 | `api.py` `verify` | For a directory whose files are malformed in a way `load_artifact(verify=False)` cannot read (for example `candidate.pipeline` not rebuildable), `verify` raises `ArtifactError` instead of ... | DOCUMENTED 76956dc | spec wording corrected |
| A04-02 | LOW | A04 | `api.py` docstrings of `certify` (qrels, warnings) | Said a `{doc: grade}` mapping with grade 0 is "read as its keys, grade 0 included, and warns once". The code warns and then `LabelReference` refuses with `ValueError` (the existing test ... | FIXED 026f1bb |  |
| A04-06 | LOW | A04 | `FixedSequenceBetting` on a `FALLBACK` decision | A fallback decision records `rule.certifying=true` and carries the full STATEMENT; only the caveat "not certified ... the statement above does not cover it" and `result.certified=false` say ... | FIXED d852000 |  |
| A05-03 | LOW | A05 | BACKEND_ARCHITECTURE.md, RETRIEVAL_BACKEND_AUDIT.md | The documents say "bit-identical" for batched rescoring; true on one stack | FIXED 76956dc |  |
| A07-02 | LOW | A07 | api.py `_warn_if_graded` | `certify(..., qrels=[...])` raised `AttributeError: 'list' object has no attribute 'items'` from inside the library. | FIXED 1aed6f6 |  |
| A07-08 | LOW | A07 | CLI | `certify docs.npz bad.json` (text file) reports "This file contains pickled (object) data ... allow_pickle": wrong diagnosis. | OPEN |  |
| A07-09 | LOW | A07 | reprs | `Artifact` repr dumps the whole VerificationReport (~4 KB); `Decision` repr prints all 40 p-values and the full pipeline mapping; `result.retriever()` has no repr. | OPEN |  |
| A07-10 | LOW | A07 | discoverability | WHY it abstained: `result.report()["not_certified_note"]` and `decision.p_values` vs `decision.level`; the repr only says "nothing certified: use the uncompressed index". Next steps only in ... | OPEN |  |
| A07-11 | LOW | A07 | messages | "vectors must be floating point, got int32" / "must be 2-D ... got shape ()" name neither argument nor document index. A 3-D or single 2-D ndarray gets "expected a MultiVectorCorpus or a ... | OPEN |  |
| A07-12 | LOW | A07 | validation | Non-L2-normalised, all-zero and float16 vectors are accepted silently (a zero-vector corpus gives SELECT, certified True under agreement). PACKAGE.md says the research covers L2-normalised ... | OPEN | the certificate records the norms and a caveat; no runtime warning |
| A07-13 | LOW | A07 | naming | `n_queries` vs `n_queries_given`; `store`/`compressed`/`corpus` on the result; CLI `--budget` vs API `query_budget`; `--given-order` vs `order_seed=None`; `verify` API ... | NO ACTION |  |
| A07-14 | LOW | A07 | docs | Public parameters absent from the guide: `progress`, `created`, `overwrite`, `include_evidence`, `strict_code`. | OPEN |  |
| A08-05 | LOW | A08 | `src/optivision/corpus.py:85`, `corpus.py:327` | Same pattern for `make-corpus` without reportlab (`ModuleNotFoundError: No module named 'reportlab'` from inside a rich traceback) and `fetch-vidore` without `datasets` (not run to ... | OPEN | behaviour of v0.3.0, unchanged |
| A08-06 | LOW | A08 | `pyproject.toml` `[project.optional-dependencies] app` | `app = ["streamlit>=1.34"]` installs Streamlit (venv2 pulled streamlit 1.65.0, pandas, pyarrow, altair, ...) but nothing in the wheel imports it: `grep streamlit src/` finds no import. ... | OPEN |  |
| A08-07 | LOW | A08 | `src/optivision/calibration.py:474,481`, `api.py:90,460,490` ... | Runtime messages and the certificate caveat text cite repository paths (`docs/UNIVERSAL.md`, `docs/package/STATISTICAL_REVIEW.md`, `reports/research/...`) that are not in the wheel or ... | OPEN |  |
| A08-08 | LOW | A08 | `optivision.__main__` | `python -m optivision` is not supported: `No module named optivision.__main__; 'optivision' is a package and cannot be directly executed`. `python -m optivision.cli` is the documented form ... | NO ACTION |  |
| A08-09 | LOW | A08 | CLI defaults `cli.py:302,333`, `cli_universal.py:305` | Relative defaults `reports/figures`, `configs/my.yaml`, `reports/universal` are written relative to cwd. They work installed (the parent directories are created; `optivision init-config` ... | NO ACTION |  |
| A08-10 | LOW | A08 | Windows console | In help and status text the em dash prints as `?` under Git Bash/cp1252 (`OptiVision RAG ? extreme token compression...`, `no index at data/index/synthetic ? run ...`). Cosmetic, likely the ... | OPEN | cosmetic |
| A09-02 | LOW | A09 | guide/LIMITATIONS.md:49 | `int4(centred)` is not a spelling the code uses; the label of the default candidates is `int4(mean)` (QUICKSTART prints it). 13+13+13 slots end in `int8(per_vector)`, `int4(mean)` ... | FIXED ba2eb19 |  |
| A09-04 | LOW | A09 | docs/package/UNIVERSAL_PACKAGE_DESIGN.md:67 | Same "40 names of `optivision.__all__`" wording (out of my scope list, `ARCHITECTURE_AUDIT.md:17,377` use `int4(centred)` too, as an audit of v0.3.0 where it was a description). | FIXED 76956dc |  |
| A10-02 | LOW | A10 | `core/artifact/bundle.py:73,146` `_HASH_BLOCK = 1<<24` | Every file hash allocates a 16 MiB read buffer even for a 150 KB file: save and load of a 155 KB artifact peak at 17 MB. | OPEN | not worth doing before a release |
| A10-03 | LOW | A10 | doc "What these numbers do not show" | Memory scaling of `certify()` is not stated anywhere: peak is about 11x the float32 corpus (measured), 4 bytes... see section 4. | FIXED 562a3ad, eb059a9 | the benchmark states the measured peak; LIMITATIONS.md gives both measurements |
| A11-02 | LOW | A11 | `README.md:204` | "the universal layer" (v0.3.0 name for the stages/`optimize` layer) used without saying what it names. | FIXED 3757c5b |  |
| A11-05 | LOW | A11 | `METHOD_C_SPEC.md:345-346` | "Consistent with the proof ... against the proof": the derivation is the project's own and not externally reviewed (the same file says so at 311 and 681). | FIXED 3757c5b |  |
| A11-06 | LOW | A11 | `METHOD_C_SPEC.md:296` | Heading "6. The statistical guarantee" is a bare "guarantee" in the table of contents; the text below is conditional. | FIXED 3757c5b |  |
| A11-07 | LOW | A11 | `pyproject.toml:8`, `npm/package.json:4` | "Extreme Token Compression", carried from v0.3.0. Measured ratios range from 3.9x to about 120x by encoder and corpus; "extreme" is an unmeasured adjective. | OPEN (owner) | = A01-07 |
| A11-08 | LOW | A11 | `METHOD_C_SPEC.md:345` | "27 of 27 held-back cells": "held-back" is not in the source; E1/RESULTS says "all 27 cells ... consistent with the claim". | NO ACTION |  |
| A12-01 | LOW | A12 | `adapters/__init__.py:352-353`, `adapters/arrays.py:46` ... | The only import cycle in the package: `adapters` <-> `adapters.arrays` <-> `adapters.nemotron` (size-3 SCC, identical at module level and with deferred imports). Controlled: bottom-of-file ... | NO ACTION |  |
| A12-02 | LOW | A12 | `core/retrieve/tiered.py`, `TwoTierRetriever.__init__` | Trap: a `Backend` (for example `NumpyBackend`) passed as `first` passes the `callable(first.search)` check, then fails at search time with an opaque `TypeError: float() argument must be ... ... | FIXED 74ecfeb |  |
| A12-03 | LOW | A12 | `core/retrieve/checks.py:54,81,95`; `core/vectors.py:135,151` | Two independent scans for non-finite vectors and two exception types for the same failure: `validate_corpus` raises `ContractError`, `as_queries`/`check_finite_store` raise plain ... | OPEN |  |
| A12-04 | LOW | A12 | `core/stages/state.py:424`, `core/stages/accounting.py:92` | Two array-byte counters: `fitted_state_nbytes` (public, re-exported from `core.stages`, referenced by 2 test files, 0 docs, no `src` caller) and `_state_bytes` (what `byte_accounting` uses ... | NO ACTION |  |
| A12-05 | LOW | A12 | `api.py:284-303` vs `core/selection/engine.py:57-62` | `_check_query_order` re-implements `engine._int_argument`; `_check_target_and_error` calls the rule validator on a dummy. Intentional and documented (fail before the expensive step, same ... | NO ACTION |  |
| A12-06 | LOW | A12 | `core/artifact/certificate.py:452-453`, `verification.py:281-282` ... | `_canonical_pipeline(spec)` and `_canon(x)` are one-line aliases of `identity.canonical_json` (5 and 13 uses). `_check_evidence_queries(ctx, ...)` never reads `ctx`; the call at line 1020 ... | NO ACTION |  |
| A12-07 | LOW | A12 | cross-layer vocabulary | `certify`/`select`/`collect_evidence` agree on `target`, `error`, `query_budget`, `order_seed`, `metric`, `space`, `progress`. Differences all mirror unchanged v0.3.0 names: `optimize()` ... | NO ACTION |  |
| A12-08 | LOW | A12 | exceptions | All new errors derive from `ValueError` except `CandidateEvaluationError(RuntimeError)` and `DtypeError(EvidenceError, TypeError)`. No common `OptiVisionError` root. "Not an integer" is ... | NO ACTION |  |
| A12-09 | LOW | A12 | `adapters/__init__.py` | No `__all__` (every other new package defines one; `len(optivision.adapters.__all__)` raises AttributeError / is absent). `import *` would export module internals. | OPEN |  |
| A12-10 | LOW | A12 | `ext/vision/__init__.py` docstring | "nothing in `optivision.core` depends on it" is true of source imports, not of run time: `core.stages.reducers` imports `optivision.stages` (whose `__init__` loads `stages/prune.py`, the ... | FIXED 76956dc |  |
| A13-03 | LOW | A13 | `core/artifact/bundle.py` docstring | Claims identical `.npz` bytes on every platform "for one NumPy version". Observed: `codes.npz` bytes differ between Python 3.10 and 3.12/3.13 for equal arrays (zip create/extract version 20 ... | FIXED 76956dc |  |
| A13-04 | LOW | A13 | certificate `code` block | The digest embeds `code.python`, `code.numpy`, `code.scipy`. Same inputs on 3.10/1.26.4, 3.12/2.0.2, 3.13/2.5.3 give identical content (decision, evidence, sample, candidate) except those ... | DOCUMENTED 76956dc |  |
| A13-05 | LOW | A13 | `validate_corpus` message | Big-endian input items are refused with a confusing message ("expected dtype >f4, got float32"). Safe refusal, no wrong certificate. | OPEN |  |
| A14-01 | LOW | A14 | boundary s5 (old l.87-88) vs s6 (l.107) | SciFact rate printed as 0.917 in one place and 0.9175 in another. 0.9175 is exact (1835/2000); the registered reports print 0.917. No rounding error, but inconsistent presentation. | FIXED 842c8e2 |  |
| A14-03 | LOW | A14 | old s10 | "enforced by a test" had no name; "binds it to a digest" named no function; "only library's own rules certified" named no check. | FIXED 842c8e2 |  |
| A14-04 | LOW | A14 | old s7 | "binds it to a digest of the corpus for that reason" overstates: the corpus check in `verify` is skipped unless the live corpus is passed; "certificates carry `basis_sensitive`" is None for ... | FIXED 842c8e2 |  |
| A14-05 | LOW | A14 | old s9 item 3 | Certifying statement agreed with `STATEMENT` (rules.py, written into the certificate by `statement_for`) but omitted A1-A6/C1-C4 names and the "not externally reviewed" caveat was present ... | FIXED 842c8e2 |  |
| A14-06 | LOW | A14 | PACKAGE.md "Install: Python 3.10 or newer", pyproject classifiers ... | Verified platform is Python 3.11 only (brief rule 9; BUILD report: 3.10/3.12/3.13 not run). The boundary now says so; PACKAGE.md/README do not say what was verified. | FIXED 9f4f6d3 | tested-platform statement beside the install requirement |
| L-04 | LOW | LEAD | CERTIFICATE_SPEC.md section 4, guide | The specification still defined `level` as the highest level in which a check ran | FIXED 8df90a5 |  |
| L-05 | LOW | LEAD | scripts/engineering/package_bench.py | Regenerating the benchmark document would have erased two hand-written sentences | FIXED 562a3ad | both are now generated from measured or checked data |
| L-06 | LOW | LEAD | api.py `certify` | The refusal of duplicate queries is raised after evidence collection, so a run on a large corpus pays for the collection before it fails | OPEN | the count needs the dropped queries and the order, known only after collection |
| A01-09 | NO ACTION | A01 | sdist | `prune tests`, no CHANGELOG/docs in sdist: tests cannot run from the sdist. Deliberate (MANIFEST.in). sdist has 137 entries: src (121), egg-info, LICENSE, MANIFEST.in, PACKAGE.md ... | NO ACTION |  |
| A02-05 | NO ACTION | A02 | `python -m optivision.adapters` | v0.3.0: module executed silently (rc 0, no-op). Now a package without `__main__`: "No module named optivision.adapters.__main__", rc 1. | NO ACTION |  |
| A02-06 | NO ACTION | A02 | `optivision.__init__` | Additive: 11 new names in `__all__` (resolved lazily via PEP 562 `__getattr__`), `dir()` gains `core, ext, backends, legacy, testing, adapters, storage` submodule attributes plus `_LAZY` ... | NO ACTION |  |
| A02-07 | NO ACTION | A02 | stage registry / CLI / adapters surface | `STAGES` gains `rotation` (an old process reading a spec that names `rotation` would reject it; v0.3.0 specs never contain it). CLI gains `certify` and `verify`; all 13 old subcommands' ... | NO ACTION |  |
| A03-05 | NO ACTION | A03 | see section 4 | Edits that pass and match the spec | NO ACTION |  |
| A04-07 | NO ACTION | A04 | `rules.py` class attributes | Class-level tampering (`FixedSequenceBetting.decide = f`, flipping `_BUILTIN_TYPES`) yields certified output or a refusal respectively: `FixedSequenceBetting.decide` patched gives ... | NO ACTION |  |
| A05-02 | NO ACTION | A05 | src/optivision/storage.py `ExactIndex.rescore` | Bitwise equality is not a property the batched code has or should have: BLAS orders the additions of a dot product by product shape and kernel | NO ACTION | production code is correct |
| A06-01 | NO ACTION | A06 | `src/optivision/__init__.py:15-52` | `import optivision` and `import optivision.core` both load PIL, yaml, numpy and nothing else heavy (no torch, scipy, sklearn, transformers, qdrant_client, typer, rich, pypdfium2). All 11 ... | NO ACTION |  |
| A06-02 | NO ACTION | A06 | design D2, `tests/test_core_layering.py` docstring ... | Documented, accepted deferral: "eager for the 40 old names ... numpy-only closure deferred; needs extras split". The ARCHITECTURE.md snippet prints `['PIL','numpy','yaml']` and matches what ... | NO ACTION |  |
| A06-03 | NO ACTION | A06 | cost | PIL+yaml import ~270-320 ms in isolation (cold, laptop loaded by 13 other agents); `import optivision` total 684-822 ms; numpy alone 246-395 ms. In the importtime run: PIL.Image 62 ms cum ... | NO ACTION |  |
| A06-04 | NO ACTION | A06 | `api.py:365` | In the core-only install, `certify(vecs, queries)` with the DEFAULT space raises a clear ImportError (needs SciPy, extra `[merge]`; suggests `allow_reduced=True`). This is intentional ... | PARTLY ADDRESSED b2ca721, 9f4f6d3 | CLI `--allow-reduced`; install line shown where `certify` is first used |
| A09-05 | NO ACTION | A09 | `optivision certify` usage line | Typer prints `Usage: -c certify` when invoked through `python -c`; with the installed `optivision` script it prints the real name. Not a defect. | NO ACTION |  |
| A10-04 | NO ACTION | A10 | `core/artifact/record.py: exact_duplicates` | O(n_queries) Python loop of one `array_digest` per query, run at creation and again at binding verification. | NO ACTION |  |
| A10-05 | NO ACTION | A10 | `optivision/__init__.py` (eager `from .core import stages`) ... | `import optivision` pays about 25 ms for the new core stages; the first `certify()` pays 0.4 to 0.6 s for importing scipy.cluster/spatial, which section 8 does not show. | NO ACTION |  |
| A11-09 | NO ACTION | A11 | FINAL_RESEARCH_BOUNDARY, BASIS_DEPENDENCE, LIMITATIONS ... | Every research number matches the stored files (section 3). No XH or held-out-encoder transfer is implied; evidence labels are not mixed. | NO ACTION |  |
| A13-06 | NO ACTION | A13 | `bundle.py`, `evidence.py` | No `os.replace`, mmap or locking in new code, so no Windows lock path. Writes are not atomic, mitigated by manifest unlinked first and written last. | NO ACTION |  |
| A14-07 | NO ACTION | A14 | LIMITATIONS, BASIS_DEPENDENCE, CONCEPTS, SELECTION_AND_CERTIFICATION | Checked for contradictions with the boundary: 0.917, 0.010, 13/16, 8/12, 0.043, 0.9175/0.0995/0.1415, 26 of 37 all agree. "lowest 0.917" is the registered print. No edits needed. | NO ACTION |  |
