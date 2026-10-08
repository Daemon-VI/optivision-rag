# a13 PORTABILITY audit

Branch `harden/a13`, one commit: `d7ae011`. Worktree base `7a6769a`.

## 1. Scope and what was executed

Wheel built from `git archive HEAD` (LF) with `uv build --wheel` in scratch; installed into fresh uv-managed venvs (interpreters downloaded by uv, nothing installed system-wide). Tests were run from a scratch copy of `tests/` with `PYTHONPATH` unset, against the installed wheel. Test selection (42 files): `test_core_* test_api_* test_backends_* test_adapters_* test_testing_* test_cli_certify test_integration_* test_docs_examples`.

| venv | Python | NumPy | SciPy | what ran | result |
|---|---|---|---|---|---|
| v3.10 | 3.10.21 | 1.26.4 (lowest allowed) | 1.13.1 | 42 files | 4425 passed, 46 skipped, 4 xfailed, 0 failed (681 s) |
| v3.12 | 3.12.14 | 2.0.2 | 1.18.1 | 42 files | first run died with Windows fatal exception 0x8007000e (E_OUTOFMEMORY in `platform._wmi_query`, machine shared by 14 agents); rerun: 4425 passed, 46 skipped, 4 xfailed (202 s) |
| v3.13 | 3.13.15 | 2.5.3 | 1.18.1 | 42 files | 4425 passed, 46 skipped, 4 xfailed, 0 failed (641 s) |
| v3.14 | 3.14.7 | 2.5.3 | latest | subset: identity, artifact_bundle, selection_engine, api_* | 492 passed, 3 skipped (smoke only) |
| vlow | 3.10.21 | 1.26.4 | 1.10.1 (lowest allowed) | rotation, selection_engine, spaces, merge, stage_state | 321 passed, 10 skipped |
| main venv | 3.11.9 | 2.4.6 | - | probe script only | - |

Also: `compileall -f src` under 3.10 and 3.13 (3.13 with `-W error`): clean. A probe compared across Python 3.10/NumPy 1.26.4, 3.11/2.4.6, 3.12/2.0.2, 3.13/2.5.3: `default_rng(seed).permutation(n)` (n=1,7,300,100000), `standard_normal`, `random(float32)`, `array_digest` variants, `canonical_json` bytes. An end-to-end `certify(created="2026-01-01T00:00:00Z")` on the synthetic example was saved under 3.10, 3.12 and 3.13 and each bundle verified by each of the three interpreters (9 of 9 ok, level `binding`).

The earlier report's F5 (Windows + NumPy 1.x int32 test failures) no longer reproduces: 0 failures on 3.10/NumPy 1.26.4 on Windows.

## 2. Findings

| ID | Sev | Where | Problem | Evidence | Fix | Status |
|---|---|---|---|---|---|---|
| A13-01 | MEDIUM | `adapters/arrays.py` (`host.append`) | A Fortran-ordered item yields an F-contiguous corpus (`np.concatenate` keeps K-order). `array_digest` treats it as the same data, but float reductions differ in the last bit: `norms.*.norm_min` 0.9999999780003288 vs ...287, so `evidence.digest` and the certificate `digest` differ for identical values. | C vs F input on 3.13 differ in exactly 4 fields (two norm_min, evidence digest, digest); `corpus.vectors.flags` C False / F True | `np.ascontiguousarray(array)` per item | FIXED in d7ae011 with regression test `test_fortran_ordered_items_give_the_same_memory_layout_as_c_ordered_ones` (fails before, passes after; 40 adapter tests pass, ruff clean). A `MultiVectorCorpus` built directly from F arrays (byte-pinned `representation.py`) and passed to `certify` still has the issue. |
| A13-02 | MEDIUM | `pyproject.toml` dependencies | `qdrant-client`, `pillow`, `pypdfium2`, `pyyaml`, `typer`, `rich` are hard dependencies, and `import optivision` imports PIL. "Qdrant optional" holds only at import level: with qdrant_client blocked, `optivision.backends`, `core`, `api`, `adapters`, `testing` import fine, but `QdrantIndex(...)` raises a bare `ModuleNotFoundError: No module named 'qdrant_client'` with no install hint (`index/qdrant_index.py:68`, v0.3.0 file). `make_backend("qdrant")` with a string raises `TypeError: no backend wrapper for str`. | import-blocker scripts | Move Qdrant (and if feasible PIL/pypdfium2) to extras, or document; add an install hint | PROPOSED (owner decision) |
| A13-03 | LOW | `core/artifact/bundle.py` docstring | Claims identical `.npz` bytes on every platform "for one NumPy version". Observed: `codes.npz` bytes differ between Python 3.10 and 3.12/3.13 for equal arrays (zip create/extract version 20 vs 45 and size fields); 3.12 and 3.13 agree. Verification still passed everywhere because the manifest records actual file hashes. | zip infolist comparison | Reword: stable per (Python minor, NumPy version) | PROPOSED (doc) |
| A13-04 | LOW (by design) | certificate `code` block | The digest embeds `code.python`, `code.numpy`, `code.scipy`. Same inputs on 3.10/1.26.4, 3.12/2.0.2, 3.13/2.5.3 give identical content (decision, evidence, sample, candidate) except those fields and so the digest. | field diff of three certificates | State in CERTIFICATE_SPEC/README that the digest is environment-bound | PROPOSED (doc) |
| A13-05 | LOW | `validate_corpus` message | Big-endian input items are refused with a confusing message ("expected dtype >f4, got float32"). Safe refusal, no wrong certificate. | big-endian run | clearer message or convert in the adapter | PROPOSED |
| A13-06 | NO ACTION | `bundle.py`, `evidence.py` | No `os.replace`, mmap or locking in new code, so no Windows lock path. Writes are not atomic, mitigated by manifest unlinked first and written last. | grep | - | NOT A DEFECT |

## 3. Checked and found correct

- Syntax/stdlib: nothing above 3.10 in `src` (`kw_only` and `zip(strict=True)` are 3.10 features; no `match`, `tomllib`, `StrEnum`, `batched`, `datetime.UTC`). Run on 3.10, 3.12, 3.13 (full new-layer suite) and 3.14 (subset).
- Canonical JSON (`core/identity.py:58`): `sort_keys`, compact separators, `ensure_ascii=True`, `allow_nan=False`, ASCII bytes, NaN/inf refused, numpy scalars converted via `int()`/`float()`; float text is Python `repr`. Probe output identical on 4 interpreters. `-0.0` and `0.0` are distinct.
- `array_digest`: dtype.str + shape header, bytes forced little-endian and C-contiguous; `>f4`, Fortran and non-contiguous variants of equal values give one digest (5 variants, 1 unique). Object/structured dtypes refused.
- Integer widths: `np.int_` search over `core/ backends/ api.py` found only transient index arrays; saved/digested arrays have explicit dtypes; permutations are cast to int64 before digesting (`engine.py:72`, `record.py:184`).
- `default_rng(seed).permutation(n)` and `standard_normal` were bit-identical on NumPy 1.26.4, 2.0.2, 2.4.6, 2.5.3. NumPy does not promise this across versions for Generator; the certificate recomputes the order and compares `indices_sha256` (`record.py`), so a future change fails loudly.
- Text I/O: new-code reads/writes use bytes plus explicit UTF-8 or `read_text(encoding="utf-8")`; new source is ASCII; JSON hashed after CRLF to LF; `codeid` normalises CRLF and uses `as_posix()`. Timestamps: UTC, numeric `strftime` (locale-independent), clock read only when `created` is absent, naive datetimes refused.
- SciPy: single lazy import (`core/selection/spaces.py:136`), clear ImportError; works on 1.10.1, 1.13.1, 1.18.1 with the same selection results in the example.
- Torch/GPU: no `.cuda()` or device strings in new code; `as_host_array` refuses non-CPU tensors unless `allow_device_transfer=True`.
- NumPy 1.26 vs 2.x: no failures either side.

## 4. Proposed honest support statement (README/pyproject)

- Tested (new-layer suite, 4425 tests, Windows 11 x86-64): Python 3.10 with NumPy 1.26.4 and SciPy 1.13.1; Python 3.12 with NumPy 2.0.2; Python 3.13 with NumPy 2.5.3. Python 3.11 with NumPy 2.4.6 is the project's main verified setup.
- Smoke only: Python 3.14.7 with NumPy 2.5.3 (492 tests; not in the classifiers); SciPy 1.10.1 on Python 3.10 (5 files).
- Statically checked only: Linux and macOS, other BLAS builds, CI cells not run by this audit.
- Unknown: big-endian hosts, GPU/torch paths, a real Qdrant server, NumPy 2.1-2.3 and 2.6+, SciPy 1.11/1.12, ARM, long paths, other locales.
- Do not claim: byte-identical bundles across machines or interpreters; a machine-independent certificate digest; that Qdrant is an optional install (it is a hard dependency).

## 5. Could not check

Linux/macOS execution, a big-endian host, a GPU, a Qdrant server, the `vlm` extra, NumPy 2.1-2.3 (only probes on 2.4.6), legacy (non-new-layer) tests on other interpreters, long-path or non-ASCII-directory saves on Windows.
