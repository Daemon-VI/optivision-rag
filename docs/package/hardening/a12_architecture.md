# a12: architecture review (HEAD 7a6769a)

## 1. Scope and what was executed

Read-only review. No repository change, no commit. Scripts live in `scratch/a12` (graph.py, dead.py, params.py, sigs.py, reg3.py, ns.py, iso.py, trap.py).

- `ast` import graph of all 98 modules under `src/optivision`: 341 module-to-module edges (300 module-level statements, 43 function-level, 17 under `TYPE_CHECKING`). `from . import x` and `from pkg import submodule` were resolved to the submodule. Strongly connected components were computed twice: module level only, and including deferred imports.
- Fresh-interpreter `python -c "import <module>"` for each of the 98 modules: all import standalone.
- Dead-code scan of every top-level def/class/constant in the new code (`core`, `backends`, `adapters`, `testing`, `ext`, `legacy`, `api.py`, `cli_certify.py`) against `src`, `tests`, `docs`, `scripts`, README (word-boundary match). Unused-parameter scan by `ast`. `ruff --select F,ARG,B,SIM,RET,ERA` on `src`: nothing reported in the new code.
- `inspect.signature` of every Protocol/ABC against every implementer (`Retriever`, `DenseScorer`, `Reranker`, `Backend`, `BaseIndex`, `SelectionRule`, `Reference`, `EncoderAdapter`).
- `__all__` resolution of `optivision`, `core`, the `core.*` subpackages, `backends`, `testing`, `legacy`, `ext.vision`; stage registry vs structural-trait declarations; one behavioural reproducer (A12-02).

## 2. Findings

| ID | Sev | Where | Problem | Status |
|---|---|---|---|---|
| A12-01 | LOW | `adapters/__init__.py:352-353`, `adapters/arrays.py:46`, `adapters/nemotron.py:31-32` | The only import cycle in the package: `adapters` <-> `adapters.arrays` <-> `adapters.nemotron` (size-3 SCC, identical at module level and with deferred imports). Controlled: bottom-of-file import after `EncoderAdapter` exists, with a comment and `noqa: F401`. Every module imports standalone. Fragile to reordering. | NO ACTION |
| A12-02 | LOW | `core/retrieve/tiered.py`, `TwoTierRetriever.__init__` | Trap: a `Backend` (for example `NumpyBackend`) passed as `first` passes the `callable(first.search)` check, then fails at search time with an opaque `TypeError: float() argument must be ... 'MultiVectorRepresentation'`, because `Backend.search(query, top_k)` is the legacy single-query method, not `Retriever.search(queries, k)`. Reproduced with `scratch/a12/trap.py`. `isinstance(backend, Retriever)` is False, so a clear check is cheap. `BackendRetriever` is the correct wrapper. | PROPOSED |
| A12-03 | LOW | `core/retrieve/checks.py:54,81,95`; `core/vectors.py:135,151` | Two independent scans for non-finite vectors and two exception types for the same failure: `validate_corpus` raises `ContractError`, `as_queries`/`check_finite_store` raise plain `ValueError`. `except ContractError` around `certify()` catches one path but not the other. Both are `ValueError`. Accidental (retrieve avoids importing `core.vectors`, though both are core). | PROPOSED (document, or raise `ContractError`) |
| A12-04 | LOW | `core/stages/state.py:424`, `core/stages/accounting.py:92` | Two array-byte counters: `fitted_state_nbytes` (public, re-exported from `core.stages`, referenced by 2 test files, 0 docs, no `src` caller) and `_state_bytes` (what `byte_accounting` uses, with the serving/append split). The public one is unused by the library. | NOT A DEFECT; optional |
| A12-05 | LOW | `api.py:284-303` vs `core/selection/engine.py:57-62` | `_check_query_order` re-implements `engine._int_argument`; `_check_target_and_error` calls the rule validator on a dummy. Intentional and documented (fail before the expensive step, same exception types). `api` accepts `numbers.Integral`, `engine` accepts `(int, np.integer)`: equal for NumPy ints. Further copies: `statistics._real_scalar`, `verification._int/_real`, `retrieve.checks.as_int`. | NOT A DEFECT |
| A12-06 | LOW | `core/artifact/certificate.py:452-453`, `verification.py:281-282`, `verification.py:937` | `_canonical_pipeline(spec)` and `_canon(x)` are one-line aliases of `identity.canonical_json` (5 and 13 uses). `_check_evidence_queries(ctx, ...)` never reads `ctx`; the call at line 1020 passes it for a uniform signature. | NO ACTION |
| A12-07 | LOW | cross-layer vocabulary | `certify`/`select`/`collect_evidence` agree on `target`, `error`, `query_budget`, `order_seed`, `metric`, `space`, `progress`. Differences all mirror unchanged v0.3.0 names: `optimize()` uses `quality_target`/`search_space`; CLI `--budget` vs `query_budget`; `certify(vectors, queries)` vs `collect_evidence(corpus, queries)`; `Backend.search(query, top_k)` next to `Backend.search_many(queries, k)` in one Protocol (mirrors `BaseIndex.search`). No `delta`, `alpha` or `query_vectors` anywhere in the new code. | NO ACTION |
| A12-08 | LOW | exceptions | All new errors derive from `ValueError` except `CandidateEvaluationError(RuntimeError)` and `DtypeError(EvidenceError, TypeError)`. No common `OptiVisionError` root. "Not an integer" is `TypeError` in `retrieve.as_int` but `QueryOrderError(ValueError)` in `engine._int_argument`. | NO ACTION |
| A12-09 | LOW | `adapters/__init__.py` | No `__all__` (every other new package defines one; `len(optivision.adapters.__all__)` raises AttributeError / is absent). `import *` would export module internals. | PROPOSED |
| A12-10 | LOW | `ext/vision/__init__.py` docstring | "nothing in `optivision.core` depends on it" is true of source imports, not of run time: `core.stages.reducers` imports `optivision.stages` (whose `__init__` loads `stages/prune.py`, the page code), `core.retrieve` imports `storage` which imports `types`, and `optivision/__init__` imports `core.stages` eagerly. `guide/ARCHITECTURE.md` already shows PIL/yaml loaded after `import optivision.core` and says so; only the ext docstring is stronger than the facts. | PROPOSED (wording) |

No finding above MEDIUM. The known example `_warn_if_graded` is NOT dead: defined `api.py:312`, called `api.py:538` inside `certify`, with `stacklevel=3` pointing at the caller of `certify`. Not confirmed as dead code.

## 3. Import graph evidence

**Cycles.** Module level: exactly one SCC larger than one, `{adapters, adapters.arrays, adapters.nemotron}` (A12-01). Including deferred and `TYPE_CHECKING` imports: the same single SCC. No cycle inside `core`, between `core` and any outer layer, or between a new layer and a v0.3.0 module. An apparent `core.vectors -> core` edge is a resolver artifact of `from . import identity`.

**Edges against the documented order** (`api, adapters, backends, ext, legacy, testing, cli_certify -> core -> pinned/agnostic`):
- core to outside: none beyond the allow-list. Core imports only `representation`, `compose`, `scoring`, `evaluation`, `calibration`, `stages` (and `stages.base/merge/project/quantize`), `storage`.
- v0.3.0 to new layers: one source edge, `cli.py:352 -> cli_certify` (designed). The root `optivision/__init__` imports `core.stages` eagerly (D2) and the rest by `_LAZY` string.
- Peer imports: `cli_certify -> cli_universal`, `backends -> index, types`, `adapters -> encoders.*` (lazy), `testing -> compose, scoring, calibration, stages.base`. Nothing runs against the order.
- `tests`/`scripts` imports: none. Environment reads in core/backends/adapters/api/testing: none. Third-party imports in new code: `rich`, `typer` (`cli_certify`), `sentence_transformers` (lazy, `adapters`), `scipy` (lazy, `core.selection.spaces`). Model-family names (ColPali, ColQwen, Nemotron) in `core`: none. They occur only in `adapters/`, `ext/` and v0.3.0 modules.
- Facade bypass: core does not route through its own facades. `compose` is imported directly by 7 core modules, `representation` by 8, `scoring` by 3 (plus `api.py`, `testing.conformance`), rather than via `core.pipeline`/`core.vectors`/`core.similarity`. They are the same objects (`core.pipeline.Pipeline is compose.Pipeline`), so no defect; "facade" means "adds on top", not "single entry point".

**Edges `tests/test_core_layering.py` does not catch.** It inspects `core/` only (plus a research-script check on the whole package):
1. Anything imported by `api`, `adapters`, `backends`, `ext`, `legacy`, `testing`, `cli_certify` (so the adapters cycle and `backends -> types` are untested; `TestSourceProperties` in `test_core_retrieve.py` covers only `core/retrieve` and `backends`).
2. Cycles inside core or between layers (none exist today).
3. Attribute access after a plain `import optivision.stages` (names are checked only for `from ... import`), `from optivision.stages import *`, and names taken via `import optivision.storage`.
4. String imports (`importlib.import_module` in `core/__init__`, `core/evaluate/__init__`, root `__init__`).
5. The run-time import closure (stated in ARCHITECTURE.md).
`TYPE_CHECKING` imports are treated as ordinary imports (stricter, fine).

## 4. Duplicated abstractions

| Pair | Verdict |
|---|---|
| `Pipeline.to_dict()`, `pipeline_spec`, `CandidateSpec`, `_pipeline_from`, `_canonical_pipeline` | One description (`to_dict`); the rest convert or hash it. Intentional. |
| `identity.digest/sha256_hex/array_digest`, `CorpusIdentity.digest`, `Evidence.digest`, `Certificate.digest` | One primitive, consumers call it. Direct `hashlib` in `bundle._file_record` (streamed file hash), `codeid._file_digest` (CRLF-normalised), `record.signature_value` (HMAC) differ on purpose. |
| Byte accounting: `CompressedCorpus.report()["bytes_per_doc"]`, `ByteAccounting.selection_bytes_per_doc`, `fitted_state_nbytes`, `_state_bytes` | Selection key re-derived on purpose and tested equal (`test_core_accounting.py:105,238`). `fitted_state_nbytes` redundant (A12-04). |
| `Ranking` vs legacy `SearchHit`/`SearchResult` | Distinct by design (D14/D15): batch, typed, carries score kind and tie-break; `BackendRetriever` bridges. |
| Vector validation: `validate_corpus`, `retrieve.checks.*`, `as_host_array` | Overlap only in the non-finite scan (A12-03); otherwise different jobs. |
| Integer/real checks (4 to 5 copies) | A12-05, A12-08. |
| Registries: `stages.base.STAGES`, `traits._DECLARED_BY`, `state._BUILDERS/_FACADES`, `rules._BUILTIN_RULES`, adapters model registry | Distinct concerns. Checked: all 16 registered stages have a library trait declaration; an undeclared stage is treated as basis sensitive (D11), so the failure mode is safe. |
| `ExactRetriever` / `NumpyBackend` / `NumpyIndex` / `ExactIndex` | `ExactRetriever`: in-memory exact MaxSim (delegates to `storage.ExactIndex`). `NumpyIndex`: legacy on-disk page index, unchanged. `NumpyBackend`: typed wrapper over `NumpyIndex`. Documented in BACKEND_ARCHITECTURE.md. Intentionally distinct; cost is two code paths that decode and score the same codes. |

## 5. Dead code, unused parameters, imports

- No top-level name in the new files is unreferenced everywhere. Names used only inside their own file (about 20 module constants such as `bundle.CERTIFICATE_FILE`, `record.CAVEAT_*`, `verification.RTOL`, `traits.LIBRARY`, `cli_certify.REFERENCES`) all have a real use. The only public name the library itself never calls is `fitted_state_nbytes` (A12-04).
- Unused parameters (AST): `__exit__(exc...)` x2 (protocol), `Evidence.__deepcopy__(memo)`, `LabelReference.relevant(base_scores)` (Protocol-mandated), `RotationProjector.fit(queries)` (stage interface), state-handler callables with a uniform signature, and `verification._check_evidence_queries(ctx)` (A12-06). No safely removable dead lines found.
- Unused imports: ruff F401 clean. No unreachable branch found (ruff B, SIM, RET, ERA clean).

## 6. Protocol conformance (inspect.signature)

`Retriever`, `DenseScorer`, `Reranker`: all implementers match names and defaults (`TieredRetriever`, `TwoTierRetriever`, `BackendRetriever` widen `queries` to `Any`, compatible). `DenseScorer` is implemented by Exact/Compressed/Tiered only and `Reranker` by Exact/Compressed only, as documented. `Backend` vs `NumpyBackend`/`QdrantBackend`: six methods identical. `BaseIndex` vs `NumpyIndex`: identical. `SelectionRule.decide` vs both built-in rules: identical. `Reference`: `LabelReference.construct` is an attribute, `RankingReference.construct` a property; both satisfy the runtime Protocol. `EncoderAdapter` abstract methods implemented by all adapters. No mismatch.

## 7. `legacy` and `ext.vision`

Both are PEP 562 re-export tables of 22 names. All resolve; neither imports torch or transformers; `legacy.Config is config.Config`; `ext.vision.PageRef is types.PageRef`. The docs describe them honestly ("no code moved", "re-exports", README "no code moved"), so they are labels, not boundaries: page code stays in place, `optivision.stages` still exports `SpatialPruner` beside the agnostic stages, and the top level keeps 16 of the same names (`Config`, `OptiVisionRAG`, `PageRef`, `SpatialPruner`, ...). Nothing enforces the namespaces except the core allow-list, which excludes both. The only overstatement is A12-10. Not oversold in the docs, and not empty.

## 8. Checked and found correct

Layer direction of all 341 edges; no core dependence on CLI, adapters, backends, vision, model names, `tests` or `scripts`; no environment reads; standalone import of all 98 modules; `__all__` of the new packages resolves (core 68, core.stages 40, core.selection 35, retrieve 21, artifact 13, evaluate 10, backends 15, testing 9, optivision 51); stage traits cover every registered stage; Protocols match implementers; digest and canonical-JSON logic is centralised; facade modules expose the same objects as the pinned ones.

## 9. Not checked

Behavioural equivalence of the duplicated integer validators on exotic inputs (read only); the wheel-only layout of `codeid` root detection; the full test suite (lead runs it); completeness of `ext.vision` against every page-specific name in the v0.3.0 tree.

## Proposed small patches

A12-02, in `core/retrieve/tiered.py` `TwoTierRetriever.__init__` after the `search` check:

    if hasattr(first, "search_many") and not hasattr(first, "n_docs"):
        raise TypeError(f"{type(first).__name__} looks like a Backend; wrap it in "
                        "optivision.backends.BackendRetriever(first)")

A12-09, `adapters/__init__.py`: add an `__all__` with `EncoderAdapter, PageEncoderAdapter, SentenceTransformersAdapter, ModelInfo, supported_models, load_adapter, ArrayAdapter, CallableAdapter, corpus_from_arrays, NemotronAdapter` and the existing public registry names.

A12-10, `ext/vision/__init__.py` docstring: change "and nothing in ``optivision.core`` depends on it" to "and no source file in ``optivision.core`` imports it".
