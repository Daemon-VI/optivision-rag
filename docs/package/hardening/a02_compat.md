# A02 Backward compatibility audit (branch package-universal HEAD 7a6769a vs v0.3.0 4d52060)

## 1. Scope and what was executed

v0.3.0 was exported with `git archive main` into `scratch/a02/v030`. Every comparison ran in separate
Python processes with `PYTHONPATH` pointing at each tree's `src` (venv Python 3.11).

- Surface dump (`surf.py`): all 45 modules of v0.3.0 walked and imported in both trees (98 in the new
  one; all import fresh in a clean process). For every public name: kind, `__module__`,
  `inspect.signature`, MRO, dataclass fields/defaults/params, class member signatures, constant values.
  Private (`_x`) names also compared. `optivision.__all__` and `dir(optivision)` compared.
- Registries: `STAGES`, `MODELS` (full `ModelInfo` repr), adapter module names.
- CLI: `--help` of the top level and all 13 v0.3.0 subcommands diffed (text). Plus an end-to-end CLI run
  in both trees (`init-config --backend synthetic`, `make-corpus`, `index`, `search`, `stats`): outputs
  identical apart from timings; config YAML and the saved index files byte-identical.
- Behaviour (`beh.py`, fixed seed, 60 docs x dim 32, 16 queries): 14 stage combinations
  (none, f16, int8 fixed/per_vector, int4, binary, PCA+int8, AdaptiveMerge+int8, RandomProjection+binary,
  Truncate+int4, lloyd2, Redundancy+int8, RandomPrune+f16, HierarchicalMerge+int8): `Pipeline.compress`,
  `to_dict`, `report`, decoded rows, `ExactIndex.score/search/rescore`, `TieredIndex.score`, save/load
  (`save_compressed`, `NpzStorage`), `to_legacy_pages` outcome; `calibrate()` with 3 search spaces x 2
  targets x 3 settings (lower_ci, point, baseline reference); `optimize()` twice (target met and not met);
  `pseudo_queries`; `NumpyIndex` add/save/load/score_all/search for binary, int8, none.
- Cross-loading: artifacts written by v0.3.0 read by the new tree and the reverse.
- Existing tests: `tests/test_public_surface.py tests/test_golden_formats.py tests/test_api_compat.py
  tests/test_adapters_page_ids.py tests/test_storage.py tests/test_adapters.py`: 384 passed.
  After my commit: `test_storage.py` + `test_public_surface.py` 328 passed; `ruff check src tests` clean.
- `adapters.py` -> `adapters/__init__.py`: `__module__` of all 6 public classes/functions, star-import,
  private names, subclassing patterns, pickle in both directions, `python -m`.

## 2. Findings

| ID | Severity | Location | Problem | Evidence | Fix | Risk | Status |
|---|---|---|---|---|---|---|---|
| A02-01 | LOW | `src/optivision/storage.py:33` | `from optivision.storage import segment_reduce` worked in v0.3.0 (incidental re-export, no `__all__`) and raised ImportError on the branch: the batched `rescore` dropped the import. The only missing name out of every public and private name in 45 modules. | `cmp.py`: `MISSING optivision.storage segment_reduce`. Private-name diff: nothing removed. | Restored the import with `noqa: F401`; regression test `tests/test_storage.py::test_names_v030_exposed_on_the_storage_module_still_import` (fails before, passes after). | None functionally. NOTE: it makes `storage.py` differ from base 91564fd by 6 lines, which contradicts `docs/package/ENGINEERING_BENCHMARKS.md:235` ("storage.py ... byte-identical to the base commit": that sentence was already about base vs HEAD, now false) and `ARCHITECTURE_AUDIT.md` line 5 phrasing. Lead: either keep the fix and amend that sentence, or drop the two commits. `storage.py` is not one of the 11 byte-pinned files. | FIXED in 1730fe5 + 518aa1b (format-only follow-up) |
| A02-02 | LOW | `src/optivision/storage.py` `ExactIndex.rescore` (unreleased batched version) | Not bit-identical to v0.3.0 in general. Docs claim "bit-identical" (CHANGELOG:92-93, test name "bitwise"), true on the ViDoRe data and in the test, but on my synthetic data 39 of 114 arrays differ by 1 ulp-scale amounts: `rescore` max abs diff 3.8e-6 (binary), 1e-6 elsewhere; `TieredIndex.score` max 1.9e-6. Cause is different matmul block shapes (BLAS rounding). Rankings of all 14 `TieredIndex.score` matrices unchanged; empty documents, empty queries, duplicate candidates and zero-candidate columns give identical results (-inf / 0 semantics preserved). `ExactIndex.score/search`, `calibrate`, `optimize`, `NumpyIndex` are bitwise identical. | `cmpb.py out_old out_new`; `rs.py` (one element differs 6.115678 vs 6.115677). `ARCHITECTURE_AUDIT.md` already admits "bitwise for most shapes, 1e-6 for very short queries". | Reword "bit-identical" to "identical rankings; scores equal to float32 rounding (<= 4e-6 absolute in a synthetic test)" in CHANGELOG/MIGRATION. No code change. | Code: none. A user comparing `TieredIndex` scores with `==` against stored v0.3.0 outputs would see 1e-6 differences. | PROPOSED (doc wording) |
| A02-03 | LOW | `adapters/__init__.py:222-227` (`PageEncoderAdapter.encode_documents`) | Now raises `ValueError` when `len(ids) != number of pages`. v0.3.0 silently produced a corpus with `len(corpus)=2` and `ids` of length 1, 3 or 0 (inconsistent). Code that passed a wrong-length list and never touched the ids (or indexed only into a longer list) used to run. | `pg.py`: old `['x']`, `['x','y','z']`, `[]` all returned; new raises `got N ids for 2 encoded pages`. | None; intentional and covered by `test_adapters_page_ids.py`. Mention in MIGRATION. | Breaks only code relying on an invalid corpus. | NOT A DEFECT (document) |
| A02-04 | LOW | `load_adapter` | For `nvidia/nemotron-colembed-vl-*` (registry loader `"nemotron"`) v0.3.0 fell through to `SentenceTransformersAdapter`; the branch now returns `NemotronAdapter(NemotronColEmbedEncoder(...))` (and sets `expected_dim`). Different class and loading path for those two ids. Not exercised (needs torch + weights). | Code reading; `MODELS` contents identical to v0.3.0. | None; check MIGRATION mentions it. | Only for users who loaded those ids via `load_adapter`; the v0.3.0 path could not have produced a working Nemotron adapter anyway (registry says loader "nemotron"). | NOT A DEFECT (document) |
| A02-05 | NO ACTION | `python -m optivision.adapters` | v0.3.0: module executed silently (rc 0, no-op). Now a package without `__main__`: "No module named optivision.adapters.__main__", rc 1. | run | None | Nobody runs this. | NOT A DEFECT |
| A02-06 | NO ACTION | `optivision.__init__` | Additive: 11 new names in `__all__` (resolved lazily via PEP 562 `__getattr__`), `dir()` gains `core, ext, backends, legacy, testing, adapters, storage` submodule attributes plus `_LAZY`, `_typing`, `_importlib`, `_core_stages`. `from optivision import *` now yields the 11 extra names (any user symbol of the same name, e.g. their own `verify`/`certify`, would be shadowed by a star import placed after theirs; unlikely). `import optivision` loads +10 modules (278 -> 288), time within noise (0.39-0.53 s both trees), no new heavy third-party imports. | `imp.py` x3 | None | Low. | NOT A DEFECT |
| A02-07 | NO ACTION | stage registry / CLI / adapters surface | `STAGES` gains `rotation` (an old process reading a spec that names `rotation` would reject it; v0.3.0 specs never contain it). CLI gains `certify` and `verify`; all 13 old subcommands' `--help` text is byte-identical. `EncoderAdapter` gains concrete `identity()` and `dimension` (property with setter); v0.3.0-style subclasses with only the two abstract methods, with `self.dimension = n` in `__init__`, with a class attribute `dimension = 7`, or with an instance attribute named `identity`, all keep working (tested). Star import of `optivision.adapters` adds `ArrayAdapter, CallableAdapter, EncoderIdentity, MODALITIES, NemotronAdapter, corpus_from_arrays, arrays, nemotron`. | `reg.py`, `ad.py`, `exc.py`, help diff | None | A subclass using `__slots__` would fail on the `dimension` setter (`_declared_dimension`); not a pattern in this codebase. | NOT A DEFECT |

## 3. Checked and found correct

- Public surface: no public or private name removed anywhere except A02-01; `__all__` of every module
  that had one is equal (top level is a strict superset: 40 old names preserved, same order of the old
  entries); every `inspect.signature` of every public callable, every dataclass field/default/order,
  every class MRO and `__module__`, constants (`STAGES` aside), and config defaults identical
  (the only text differences in the dump were memory addresses of `typer`/`dataclasses` sentinels).
- Exceptions: same type and message for the misuse cases tried (missing queries, unknown stage, unfit
  PCA, bad ratio, empty corpus, NaN input, unknown adapter id) except the intended `rotation` and A02-03.
- Behaviour: `Pipeline` for 14 combinations, `calibrate` (18 runs, all fields except wall-clock
  `query_ms`/`compress_seconds`), `optimize` (target met and not met), `pseudo_queries`, `NumpyIndex`
  (3 methods, save/load/search), `ExactIndex.score/search` are bitwise identical in 81 of 81 non-rescore arrays
  and all JSON.
- Files: every `.npz`, `store/*.npz` and `NumpyIndex` directory file written by the two trees is byte-identical
  (`cmp`); v0.3.0 artifacts load in the new tree and new-tree artifacts load in v0.3.0, with identical decoded
  rows, offsets and scores (54 of 54 arrays bitwise).
- `adapters` package: `__module__` of all public classes is `optivision.adapters` (unchanged), pickles
  made by v0.3.0 (adapter instance, `ModelInfo`, the class reference) load in the new tree and vice versa,
  `from optivision.adapters import X` works for every old name, relative imports in the new `__init__` are right.
- Golden/surface/compat/storage/adapter-id tests: 384 passed.

## 4. Could not check

- Real encoders (colpali-engine, sentence-transformers, Nemotron) and the Qdrant index (`qdrant_index.py` has
  docstring-only changes; code paths not run).
- Python versions other than 3.11, other platforms, a wheel install (the venv has an editable install of a
  different project that `PYTHONPATH` overrides; verified by printing module paths in the dump).
- `scripts/` and `app/` entry points outside the package; the VS Code extension and npm wrapper.
- Whole test suite (lead runs it).
