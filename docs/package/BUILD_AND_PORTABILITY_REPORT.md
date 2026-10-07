# Build and portability report (release-engineering rehearsal)

THIS IS A REHEARSAL. Nothing was released, pushed or uploaded to any index. No git state was
changed. All artefacts (wheels, sdists, virtual environments, tree copies, logs) are in a scratch
directory outside the repository and are not part of it. The only file added to the repository by
this review is this report.

## 1. What was checked, on which tree

| item | value |
|---|---|
| tree used for every build and test | `git -c core.autocrlf=false archive 2247d54` (branch `package-universal`), i.e. the LF content a Linux CI checkout holds. Uncommitted edits in the worktree were ignored. HEAD moved to `77ce904` while the review ran; those later commits are NOT covered. |
| first (superseded) build | `988a086`, used only for the D18 demonstration and the early wheel-content check |
| host | Windows 11, Python 3.11.9, `build` 1.6.1 with build isolation (network to PyPI works), twine 7.0.0, ruff 0.16.3, setuptools 84.0.0 fetched by the isolated build |
| fresh venvs | `v1`: wheel + deps from PyPI (numpy 2.4.6, qdrant-client 1.19.1, typer 0.27.3, pillow 12.3.0, PyYAML 6.0.3), SciPy added later; `vn`: numpy 1.26.4 + scipy 1.13.1 + wheel |
| not re-run | task 3 was limited, at the coordinator's request, to the new-layer tests (`test_core_*`, `test_api_*`, `test_backends_*`, `test_adapters_*`, `test_integration_end_to_end.py`, `test_docs_examples.py`), plus a few extra files noted below |

Note on the host: `core.autocrlf=true` is set system-wide, so a plain `git archive HEAD` and the
worktree hold CRLF, while the index and any Linux checkout hold LF. 657 tracked worktree files are CRLF
and 16 are LF (files written by tools). That mix does not matter for the build (see F10).

## 2. Build (task 1)

Artefacts (from `2247d54`): `optivision_rag-0.3.0-py3-none-any.whl` 334,037 B (108 files, 947,504 B
unpacked) and `optivision_rag-0.3.0.tar.gz` 593,933 B (182 files). `python -m build` builds the wheel from
the sdist, so the sdist is proven buildable. `twine check --strict`: PASSED for both.

Wheel contents:
- Every package directory has an `__init__.py`: `optivision`, `adapters`, `backends`, `compression`,
  `core`, `core/artifact`, `core/evaluate`, `core/retrieve`, `core/selection`, `core/stages`,
  `encoders`, `ext`, `ext/vision`, `index`, `legacy`, `pruning`, `stages`, `testing`. A script compared
  every source directory holding `.py` files with the wheel: none missing, none without `__init__.py`.
- `api.py`, `cli_certify.py`, `core/artifact/*` (7 files), `testing/*` (5 files) are in.
- `presets/{colpali,colsmol,qdrant,synthetic}.yaml` are in.
- Nothing outside `optivision/` and the dist-info: no tests, no `tests/golden`, no `reports`, `scripts`,
  `docs`, `paper`, `npm`, no `.npz/.npy`, no `__pycache__`.

Metadata: Name `optivision-rag`, Version `0.3.0`, Metadata-Version 2.4, Requires-Python `>=3.10`,
License-Expression MIT, dependencies numpy `<3,>=1.26`, pillow, pypdfium2, qdrant-client, pyyaml,
typer, rich (unchanged from v0.3.0); extras vlm, corpus, bench, app, merge, dev; entry point
`optivision = optivision.cli:app`; long description is `PACKAGE.md` (10,435 chars, markdown). `pyproject.toml`
and both workflows are unchanged on the branch (`git diff 91564fd HEAD`).

Sdist contents: `src/optivision` (102), `tests/test*.py` (68), root `README.md`, `PACKAGE.md`, `LICENSE`,
`pyproject.toml`, `setup.cfg`, `PKG-INFO`, egg-info. No `reports/`, `scripts/`, `docs/`, `npm/`, `paper/`,
no large files, no LFS pointers.

### D18: the version stays 0.3.0

Anyone who builds this branch gets a wheel named `optivision_rag-0.3.0-...whl` whose contents differ
from the published 0.3.0 (it adds `core`, `api`, `certify`, ...). Demonstrated: in a venv that already
holds the older 0.3.0-labelled wheel, `pip install <new wheel>` and `pip install -U <new wheel>` both
answer "already installed with the same version ... Use --force-reinstall", and the new modules do not
appear. Consequences: (a) a user or CI cache cannot tell the two apart by version; (b) the wheel's
long description (`PACKAGE.md`) describes the unreleased core and `certify` under version 0.3.0;
(c) `publish.yml` accepts a `v0.3.0` tag (the literal check passes) but PyPI refuses a second
0.3.0 (RELEASING.md: "Neither registry ever accepts the same version twice, even after deletion"),
so a mistaken release fails late and noisily rather than overwriting; (d) certificates record
`0.3.0`; only the code digest tells the builds apart.

Recommendation, before any release and decided by the owner: bump to a new version in both
`src/optivision/__init__.py` and `npm/package.json` (the version test requires equality), and update
`CHANGELOG.md`, the "unreleased" section of `PACKAGE.md`, and the two tests that pin the literal
`0.3.0` (`tests/test_api_compat.py::test_version_is_still_0_3_0_and_a_literal`,
`tests/test_public_surface.py::test_version_is_a_string_literal_in_init`). If a pre-release is chosen,
PEP 440 (`0.4.0rc1`) and npm semver (`0.4.0-rc.1`) spell it differently, and
`tests/test_release_versions.py` compares the two strings, so it would fail; use a final version
(e.g. `0.4.0`) or change that test to compare normalised versions (`packaging.version.Version`
accepts `0.4.0-rc.1` and normalises it to `0.4.0rc1`).

## 3. Clean install and smoke test (task 2)

Commands (cwd = a scratch dir that is not the worktree, `PYTHONPATH` unset):

```
python -m venv v1 ; v1/Scripts/python -m pip install optivision_rag-0.3.0-py3-none-any.whl
python -c "import optivision; print(optivision.__version__, optivision.__file__)"
```

Results:
- `0.3.0`, file inside `v1\Lib\site-packages\optivision\__init__.py` (not the source tree).
- Imported OK: `optivision.core`, `.core.identity`, `.core.vectors`, `.core.stages`, `.core.selection`,
  `.core.evaluate`, `.core.retrieve`, `.core.artifact`, `.core.pipeline`, `.core.similarity`, `.backends`,
  `.adapters`, `.ext`, `.ext.vision`, `.legacy`, `.testing`, `.api`, `.cli`, `.cli_certify`.
- `optivision --version` prints `optivision-rag 0.3.0`; `python -m optivision.cli --help` lists
  `certify` and `verify` besides the v0.3.0 commands. `available_presets()` returns
  `['colpali','colsmol','qdrant','synthetic']`.
- v0.3.0 paths on seeded synthetic data (`optivision.testing.make_corpus/make_queries`): `Pipeline([Int8Quantizer()]).compress`,
  `MultiVectorCorpus.save/load` (round trip equal), `optimize()` (returns), `calibrate()` (returns).
- New paths: `CandidateSpace.default(allow_reduced=True)`, `collect_evidence`, `select`, `ExactRetriever`,
  `certify`, `result.save` (writes `MANIFEST.json`, `certificate.json`, `evidence.npz`), `verify`
  (report ok), `code_identity()`.
- Without SciPy (a fresh venv has none; it is an optional extra): `CandidateSpace.default(32)` raises
  `ImportError: the default candidate space needs SciPy ... pass allow_reduced=True`; `certify()` with the
  default list raises an ImportError naming the Ward families; `optimize()` and `calibrate()` warn
  "SciPy is not installed, so the default search space has no Ward merging families". With the reduced
  option the space has 22 candidates and `is_registered_default` is False. After `pip install scipy`
  (1.17.1): the default space builds with 40 candidates, identity starts `af29b01146a9`, and
  `is_registered_default` is True; the warnings are gone.
- `code_identity()["digest"]` of the installed wheel equals that of the LF source tree and of a CRLF copy
  of the same tree (`e0ca6e23...`); the line-ending claim in `codeid.py` holds (also covered by
  `test_core_codeid.py`).

## 4. New-layer tests against the installed wheel (task 3)

Run from a scratch copy of `tests/` (no `src/`, `docs/`, `pyproject.toml`, `scripts/`, `reports/`),
`PYTHONPATH` unset, installed wheel in `v1` with SciPy: **2889 passed, 38 skipped, 5 failed** in 78 s.
Slowest tests are all under 6 s (setup of `test_integration_end_to_end` 5.7 s). Each file also passes when run alone (loop over 37 files; only the
failures below recur), so the `from .test_x import helper` pattern works in isolation.

Skips are clean: torch absent, `scripts/research/e1_stats.py` absent, sidecars absent (7), no `docs/` examples, 64-bit longdouble.

The 5 failures all come from reading the repository layout and fail instead of skipping (F4).

## 5. CI portability (task 4)

CI (`ci.yml`): `ubuntu-latest`; Python 3.10 and 3.13; `actions/checkout@v7` without `lfs: true`;
`pip install -e ".[dev,corpus]"`; `pytest -q`; on 3.13 `ruff check src tests` and `python -m build`.
A second job builds the wheel on ubuntu and windows and runs the npm launcher against it.

- Lint: `ruff check --no-cache src tests` (0.16.3): all checks passed; `--target-version py310 --select E9,F,NPY201,UP036`: passed.
- Python 3.10: `ast.parse(..., feature_version=(3,10))` on 98 source and 71 test files: no failures.
  `vermin`: minimum required version 3.10 for `src` and `tests` (no 3.11+ stdlib or syntax). Grep for
  `StrEnum`, `Self`, `tomllib` (only inside `pytest.importorskip` in `test_public_surface.py`), `ExceptionGroup`,
  `except*`, `datetime.UTC`, `fromisoformat`: nothing that breaks on 3.10. Python 3.10 itself was not run.
- NumPy 1.26: the grep found no NumPy-2-only API (`dtype.newbyteorder` is valid in both). Real run in
  `vn` (numpy 1.26.4, scipy 1.13.1): 4494 passed, 36 skipped, 11 xfailed, 11 failed. Eight failures are
  the repository-layout ones (F4); three root causes are new (F5).
- Git LFS: `.gitattributes` routes `*.npz`/`*.npy` to LFS; `tests/golden/.gitattributes` unsets the filter
  (`git check-attr filter tests/golden/queries_v030.npz` = `unset`), and the committed golden blobs start
  with `PK` or `\x93NUMPY`, not with a pointer, and no tracked file under `tests/` is an LFS object.
  Only `tests/test_core_statistics_parity.py` reads LFS data (`reports/research/E1/store/*.npz`,
  `E1/{real,confirm}/*_selections.npz`); `test_core_spaces.py` and `test_core_statistics.py` read the
  `.json` sidecars (not LFS). With every `.npz` under `reports/` replaced by a pointer file carrying the
  real oid and size: `test_x_harness.py`, `test_core_statistics_parity.py`, `test_core_spaces.py` give
  129 passed, 14 skipped, 0 failed; the parity test skips with "is a Git LFS pointer, not the data". So
  no new LFS problem. (A first run with invented oids made three `test_x_harness` tests fail; that was
  the simulation, not the code: the harness compares the pointer's oid with the registered hash.)
- Subprocess and tools: every subprocess call uses `sys.executable`; no test calls `git`; no
  `os.replace`/`rename`/locking in new code; memory-mapped loads exist in tests (`mmap`) but are only read.
  Windows file locking was not hit.
- Bitwise float equality: `test_core_retrieve*.py` (84 exact comparisons against 18 tolerance
  ones), `test_core_artifact_bundle.py` (18 exact), `test_core_stage_state.py` (10 exact), `test_core_rotation.py` (13 exact) assert exact
  equality between two computations on the same machine. `test_integration_determinism.py` compares digests
  across processes on one machine and says so. The pre-existing `tests/test_storage.py::test_batched_rescore*` asserts bitwise
  equality of batched versus per-query scores and is currently RED on the repository's own CI (latest
  `ci` run on `research-investigation`: 8 failed on Python 3.10, all in that file). This is the
  BLAS-dependence the brief asked about, already visible, not caused by this branch (F1).
- Optional dependencies: with `scipy`, `qdrant_client`, `torch`, `transformers`, `colpali_engine`
  blocked by a `sitecustomize` import hook: 2996 passed, 147 skipped, 6 failed; four are the layout ones, two are new (F6, F7).

## 6. Findings

| id | sev | location | what breaks where | evidence | fix |
|---|---|---|---|---|---|
| F1 | MAJOR (pre-existing, blocks green CI) | `tests/test_storage.py::test_batched_rescore_*` (8 tests) | exact float equality between batched and per-query scoring fails on CI hardware; CI is red on `research-investigation` before this branch | `gh run list`: 5 of 6 recent runs failed; failing log shows these 8 tests | compare with `assert_allclose(rtol=1e-6, atol=0)` or make the batched path use the same reduction order; the branch cannot be CI-green until this is decided |
| F2 | MAJOR (owner decision) | `src/optivision/__init__.py` (`0.3.0`), D18 | a branch wheel carries the published version number with different contents; pip will not replace an installed 0.3.0; `PACKAGE.md` documents unreleased features under 0.3.0 | section 2 | bump before any release, see section 2 for the exact list of places and the pre-release caveat |
| F3 | MINOR | sdist (default setuptools rules) | the sdist ships 68 `tests/test*.py` but not `tests/conftest.py`, `tests/__init__.py` or `tests/golden/`; they cannot run from the sdist (fixtures `rng`, `page_image` and the relative imports are missing) | `inspect_sdist`: 68 test files, no conftest | add `MANIFEST.in` with `prune tests` (or include `tests/` completely, golden fixtures are small) |
| F4 | MINOR | `test_core_layering.py::test_core_exists`; `test_core_retrieve.py::TestSourceProperties` (2); `test_api_compat.py::test_version_is_still_0_3_0_and_a_literal`; `test_docs_examples.py::test_the_guide_has_the_documented_files`; `test_public_surface.py` (2: version literal, console script); `test_release_versions.py` | fail or error (not skip) when `src/`, `docs/` or `pyproject.toml` are absent, i.e. when tests run against an installed wheel or sdist. The other layering tests then pass vacuously over an empty file list. `test_x_harness.py` imports `scripts/research` at module level (pre-existing; not run without the repository layout) | section 4 | `pytest.skip(..., allow_module_level=True)` when the path does not exist, as `test_core_statistics_parity.py` already does |
| F5 | MINOR | `tests/test_core_vectors.py::test_each_violation_raises_a_contract_error...[offsets-*]` (3); `tests/test_core_selection_engine.py` seeded-order tests (2) | fail on Windows with NumPy < 2 only: the default integer there is int32, so the test's offsets array and `default_rng(11).permutation(300)` are int32 and the expected message/digest differ. The engine itself casts to int64 and is correct. Linux with NumPy 1.26 should not fail (not run) | `vn` run | give the arrays `dtype=np.int64` / `.astype(np.int64)` in those tests |
| F6 | MINOR | `tests/test_testing_helpers.py::test_the_built_in_stages_conform[HierarchicalMerge...]` | fails (ImportError "HierarchicalMerge needs SciPy") instead of skipping when SciPy is absent; the `dev` extra installs SciPy so CI is unaffected | blocked run | `pytest.importorskip("scipy")` for the Ward cases |
| F7 | MINOR | `tests/test_cli_certify.py::test_values_the_library_refuses_exit_with_status_two_and_say_why[args6-metric]` (and the CLI order of checks) | without SciPy the CLI hits the default-list ImportError (exit 1) before it validates `--metric` (expected exit 2) | blocked run | validate arguments before building the default space, or skip without SciPy |
| F8 | NOTE | `.github/workflows/ci.yml` | CI installs `-e` (editable): a package directory without `__init__.py` would import fine there and vanish from the wheel. Today none is missing, but nothing in CI would catch a future one | wheel inspection | add a CI step that installs the built wheel in a clean venv and runs a short import/`certify` smoke from outside the checkout |
| F9 | NOTE | `docs/package/UNIVERSAL_PACKAGE_DESIGN.md` 5.9 vs `api.py` | design says `compressed` on ABSTAIN is the uncompressed corpus; the implementation sets `pipeline` and `compressed` to `None` (documented in the `CertifiedResult` docstring) | `api.py:102-120` | update the design text |
| F10 | NOTE | build host | wheels built on Windows have CRLF in `METADATA` (186 of 186 lines); not byte-identical to a Linux build. `publish.yml` builds on ubuntu, so releases are unaffected. Line endings of `.py` files do not change `code_identity()` (verified) | `show_meta` | build releases only through the workflow |
| F11 | NOTE | `pyproject.toml` | `requires = ["setuptools>=77"]` is unpinned (built here with 84.0.0); fine for a pure-Python package, but builds are not reproducible across dates | wheel `Generator` | optional: pin an upper bound |

## 7. Not checked on this machine

Python 3.10, 3.12, 3.13 (only static checks and Python 3.11); Linux and macOS (no run; the CI
Linux cell is the check, and its current red state is F1); NumPy 1.26 on Linux and NumPy 2.0-2.3 (only
1.26.4 and 2.4.6 on Windows); other BLAS builds; a real Qdrant server; GPUs and the `vlm` extra (torch absent);
`ruff>=0.5` older than 0.16.3; the npm launcher test; the publish workflow itself;
commits after `2247d54` (HEAD was `77ce904` at the end); the legacy tests (`test_pipeline`, `test_storage` apart from CI log evidence, etc.) against the wheel, by request.

## 8. Commands for the final build and clean-install check (after everything is committed)

```
cd /c/Users/Lenovo/projects/optivision-package && git status --short        # must be empty
S=<scratch>; rm -rf $S/src && mkdir -p $S/src
git -c core.autocrlf=false archive --format=tar HEAD | tar -x -C $S/src
python -m build --outdir $S/dist $S/src                                       # sdist, then wheel from it
python -m twine check --strict $S/dist/*
python tools/inspect_wheel.py $S/dist/*.whl $S/src                            # package dirs, __init__.py, leaks
python -m venv $S/venv && $S/venv/Scripts/python -m pip install $S/dist/*.whl pytest scipy
cd $S && unset PYTHONPATH && $S/venv/Scripts/python -c "import optivision as o; print(o.__version__, o.__file__)"
$S/venv/Scripts/optivision --version && $S/venv/Scripts/python -m optivision.cli --help
$S/venv/Scripts/python <smoke script of section 3>
cp -r $S/src/tests $S/tr/tests && cd $S/tr && $S/venv/Scripts/python -m pytest -p no:cacheprovider -q tests/test_core_*.py tests/test_api_*.py tests/test_backends_*.py tests/test_adapters_*.py tests/test_integration_*.py tests/test_docs_examples.py
```

The helper scripts used here (`inspect_wheel.py`, `inspect_sdist.py`, `smoke.py`, `smoke2.py`) are in
the scratch `tools/` directory, not in the repository.
