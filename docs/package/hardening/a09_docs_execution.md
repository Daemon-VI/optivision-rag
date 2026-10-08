# A09 documentation execution report

Worktree `ov-harden/a09`, branch `harden/a09`, base 7a6769a, fix commit 86af8b7. Scratch: `scratch/a09`
(extractor `extract.py`, runner `run.py`, static checks `static.py`, `names.py`, `paths.py`).

## 1. Scope and what was executed

- Extractor pulled 79 fenced blocks (file, line, language) from the 9 guide pages, API_DESIGN,
  MIGRATION, CERTIFICATE_SPEC, BACKEND_ARCHITECTURE, ADAPTER_ARCHITECTURE, README, PACKAGE, npm/README.
- 6 example scripts run each in an empty temp dir with `PYTHONPATH=a09/src`: all exit 0, output as the
  docs describe.
- 29 python blocks of the guide pages + API_DESIGN + MIGRATION run per page, in order, in one session
  (per-block markers, per-block try/except): 0 failures. Every "text" output block that follows a
  python block was compared with the real output: all identical (QUICKSTART, SELECTION_AND_CERTIFICATION,
  BASIS_DEPENDENCE, BACKENDS_GUIDE, ADAPTERS_GUIDE, CONCEPTS, LIMITATIONS, ARCHITECTURE import list).
- CLI run for real on files I created: `optivision certify` / `verify` (API_DESIGN block), plus
  `--json`, `--budget`, abstain (exit 0), bad target (exit 2), existing `--out` (exit 2), edited
  certificate (verify exit 1), `verify certificate.json` alone. `make-corpus`, `index`, `search`,
  `stats`, `explain` with `-c synthetic`, and `bench --sweep --cache` with the synthetic config.
- Static: every `optivision.x.y` dotted name resolves; all CLI flags in the in-scope docs exist in
  `--help`; all file names in backticks exist (or are generated artifacts); signatures of certify,
  verify, verify_certificate, load_artifact, save_artifact, sign_certificate, select,
  collect_evidence, compress_selected, create_certificate, register_traits, traits_of, make_backend,
  result.save, CandidateSpace.default; the 25 certificate fields and every nested key listed in
  CERTIFICATE_SPEC section 2; the 10 assumptions and the statement verbatim; the 25 case ids of the
  tamper table (all but the glob `list-*-consistently` found in the test file); bundle file names,
  `codes.npz` / `fitted.npz` / `MANIFEST.json` header formats and versions; exception names, bases
  and where they live; `report()` keys; default values (`target 0.97`, `error 0.05`, `metric ndcg@5`,
  `order_seed 0`, `max_block_bytes` 256 MiB, `allow_reduced=False`); `STAGES` = 16; 40 candidates in 7
  families; no scipy/torch/qdrant_client loaded by `import optivision`; 11 lazy names, `__all__` size.

## 2. Table of blocks

Result key: PASS = executed, no error, printed output matches the doc.

| file:line | kind | result |
|---|---|---|
| guide/QUICKSTART.md 19, 53, 78, 94, 118, 139, 175, 206, 220 | python (9, one session) | PASS, all 9 output blocks identical (7.05x, 13 of 40, p 0.7295, etc.) |
| guide/SELECTION_AND_CERTIFICATION.md 51, 232 | python | PASS, outputs identical |
| guide/BASIS_DEPENDENCE.md 153, 180, 205, 270 | python | PASS; 195/216/283 outputs identical (block 153 table has no doc copy) |
| guide/BACKENDS_GUIDE.md 59, 95 | python | PASS, outputs identical |
| guide/ADAPTERS_GUIDE.md 45, 80 | python | PASS, outputs identical |
| guide/ARCHITECTURE.md 78 | python | PASS, `['PIL','numpy','yaml']` |
| guide/CONCEPTS.md 120 | python | PASS, 475->258 / 121 vectors identical |
| guide/LIMITATIONS.md 117 | python | PASS, identical |
| guide/ARCHITECTURE.md 12, 46; CONCEPTS.md 9 | text diagrams / trees | NOT RUNNABLE: illustration (tree checked against the file list, see 3) |
| guide/examples/*.py (6) | scripts | PASS (exit 0, prints) |
| API_DESIGN.md 28, 132, 154, 199 | python (4, one session) | PASS; digest equality True; error block prints TargetError / ErrorLevelError / ValueError |
| API_DESIGN.md 58 | bash (certify, verify) | PASS, exit 0 and 0; "no check failed" |
| MIGRATION.md 85 | python | PASS (one UserWarning from optimize on 75 queries, expected) |
| README.md 33, PACKAGE.md 12 | python | NOT RUNNABLE as written (`doc_vectors`, `labels_or_None` are placeholders); ran with synthetic stand-ins: PASS |
| README.md 168, PACKAGE.md 88 | python | PASS (SELECT, verify ok True), now covered by a test |
| README.md 217, 232, 499; PACKAGE.md 136; npm 13 | text diagrams | NOT RUNNABLE: illustration |
| README.md 393 | bash (pip install gradio, app.py) | NOT RUNNABLE: installs packages and starts a UI (app.py exists) |
| README.md 413, 435; PACKAGE.md 163; npm 55 | bash pip install | NOT RUNNABLE: would install; the extras named (`corpus`, `vlm`, `merge`, `bench`, `app`, `dev`) all exist in pyproject |
| README.md 446 (steps 1, 3, 4, 5) | bash | steps 1-5 run with `-c synthetic` (make-corpus, index, search, explain, bench --sweep --cache): PASS; with `colsmol.yaml`: NOT RUNNABLE, needs a ~0.5 GB model download; step 6 (streamlit) NOT RUNNABLE |
| PACKAGE.md 183, npm 74 | bash | PASS (make-corpus, index, search, stats with synthetic) |
| PACKAGE.md 195, npm 86; PACKAGE.md 207, npm 98 | bash / python | NOT RUNNABLE: need colsmol model and PDFs |
| PACKAGE.md 174, npm 65 | `npx optivision-rag --help` | NOT RUNNABLE safely: the launcher pip-installs `optivision-rag[corpus]==0.3.0` from PyPI on first run (see 5) |
| README.md 553 | bash pytest, ruff | not run (whole suite forbidden); ruff ran on the changed test file |
| CERTIFICATE_SPEC, BACKEND_ARCHITECTURE, ADAPTER_ARCHITECTURE | no fenced runnable blocks | n/a (prose and tables checked statically) |

## 3. Findings

| ID | Sev | file:line | problem / evidence | status |
|---|---|---|---|---|
| A09-01 | MEDIUM | API_DESIGN.md:234, MIGRATION.md:6 | Both say the "40 names of `optivision.__all__`" are unchanged; `len(optivision.__all__)` is 51 (40 v0.3.0 + 11 new). MIGRATION section 1 itself says 51 in all, so the page contradicted itself. | FIXED 86af8b7 (wording only: "40 v0.3.0 names") |
| A09-02 | LOW | guide/LIMITATIONS.md:49 | `int4(centred)` is not a spelling the code uses; the label of the default candidates is `int4(mean)` (QUICKSTART prints it). 13+13+13 slots end in `int8(per_vector)`, `int4(mean)`, `binary`, plus float16: the count 39 of 40 is right. | FIXED 86af8b7 |
| A09-03 | MEDIUM | API_DESIGN.md:7 "Every Python block below is run as written", guide/README.md:54 | `tests/test_docs_examples.py` only ran the guide pages and examples; API_DESIGN and MIGRATION blocks, the documented CLI commands and the README/PACKAGE `certify` snippet were not run by any test (the claim was true only because I ran them). | FIXED 86af8b7: 4 new tests (API_DESIGN, MIGRATION python in order; the two CLI commands against files the block writes, verify exit 0; README and PACKAGE certify snippet). 9 selected tests pass, ruff clean. |
| A09-04 | LOW | docs/package/UNIVERSAL_PACKAGE_DESIGN.md:67 | Same "40 names of `optivision.__all__`" wording (out of my scope list, `ARCHITECTURE_AUDIT.md:17,377` use `int4(centred)` too, as an audit of v0.3.0 where it was a description). | PROPOSED: same one-word edit if the owner wants it |
| A09-05 | NO ACTION | `optivision certify` usage line | Typer prints `Usage: -c certify` when invoked through `python -c`; with the installed `optivision` script it prints the real name. Not a defect. | NOT A DEFECT |

No block failed, no printed output differed, no flag, parameter, default, certificate field, exception
or file name was wrong apart from A09-01/02.

## 4. What `tests/test_docs_examples.py` covered, and did not

Covered (before my change): each `examples/*.py` (exit 0, at most 60 lines, says "synthetic", prints
something, leaves cwd empty); the python blocks of each guide page, concatenated in order (exit 0,
cwd empty); ASCII-only guide files; banned wording; relative links in the guide.
Not covered: printed outputs (no comparison of any "text" block with real output, so a changed 7.05x,
13 of 40 or a reworded caveat would not fail); API_DESIGN, MIGRATION, CERTIFICATE_SPEC,
BACKEND_ARCHITECTURE, ADAPTER_ARCHITECTURE, README, PACKAGE, npm/README (until this commit: API_DESIGN,
MIGRATION, README/PACKAGE certify now run); every bash block and the CLI (flags, exit codes);
names, signatures, defaults, certificate field names and file names mentioned in prose and tables
(for example "40 names", `int4(centred)`); blocks that need SciPy are skipped silently where SciPy is
missing (the default list needs it); blocks run only in a concatenated session, so one block failing
hides which; ASCII/wording rules are not applied to API_DESIGN/MIGRATION/CERTIFICATE_SPEC. I did not add
an output-matching test: outputs legitimately differ in last digits and by platform, which the docs say.

## 5. Could not check / side effect

- Real encoders, Qdrant server, colsmol/colpali configs, streamlit, pip installs, `npx` (see table).
- Side effect to report: `node npm/bin/optivision.js --help` (my attempt to run the npx block) began
  installing into `%LOCALAPPDATA%\optivision-rag\env-0.3.0-optivision-rag_corpus_==0.3.0` (a partial
  venv, nothing installed into it, no process left running). I could not delete it (the delete was
  denied by the permission layer), so the user may remove that one directory; the 0.2.0 sibling
  predates this run. Nothing else outside the worktree and scratch was touched.
- Platform: Python 3.11 on this Windows laptop only.
