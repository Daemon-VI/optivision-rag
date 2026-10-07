# Package architecture, as built

Status: 2026-10-06, branch `package-universal`. This page describes the tree as it is, verified
against `find src/optivision -name "*.py"` and the `__init__` files. The design contract is
`../UNIVERSAL_PACKAGE_DESIGN.md`. The original page recipe is described in `../../ARCHITECTURE.md`
and is not repeated here. Modules that were still being finished when this was written
(`api.py`, `core/artifact/`, `core/evaluate/collect.py`, `testing/`) are described from the
design contract and the code as read; check the code before relying on a detail.

## 1. Layout

```text
src/optivision/
  __init__.py          the 40 v0.3.0 names, imported eagerly; certify, verify and a few more load on first use
  api.py               certify(), verify(), CertifiedResult                         (new)
  core/                the model-agnostic layer
    identity.py        canonical JSON and SHA-256 digests                           (new)
    vectors.py         facade over representation.py + contract checks, identities  (facade + new)
    pipeline.py        facade over compose.py + pipeline identity                   (facade + new)
    similarity.py      facade over scoring.py                                       (facade)
    stages/            traits, fitted state, byte accounting, RotationProjector     (new)
                       reducers.py codecs.py projections.py: the library's stage classes by kind
    evaluate/          evidence.py reference.py collect.py                          (new)
                       metrics.py: facade over evaluation.py + metrics_from_ranking
    selection/         spaces.py statistics.py rules.py decision.py engine.py       (new)
    retrieve/          types.py checks.py exact.py tiered.py                        (new; delegate to storage.py)
    artifact/          certificate.py record.py verification.py report.py bundle.py codeid.py   (new)
  adapters/            EncoderAdapter and the model registry (moved from adapters.py), arrays.py, nemotron.py
  backends/            base.py numpy_backend.py qdrant_backend.py factory.py         (wrappers; index/ is unchanged)
  ext/vision/          namespace: page types, pixel and grid pruning, page encoders  (re-exports)
  legacy/              namespace: Config, OptiVisionRAG, Compressor, TokenPruner     (re-exports)
  testing/             synthetic data, basis transforms, basis_probe, conformance checks   (new)
  cli_certify.py       the commands `optivision certify` and `optivision verify`     (new)
  ...                  every other v0.3.0 module, where it was: stages/, compression/, index/, pruning/,
                       encoders/, storage.py, optimize.py, calibration.py, types.py, config.py, cli.py ...
```

Existing files edited, against the research tip `91564fd`: `__init__.py` (additive: new names load
on first use; `__version__` stays `"0.3.0"`), `cli.py` (registers the two new commands),
`adapters.py` (became the package `adapters/`, registry unchanged) and the docstrings and comments
of `index/numpy_index.py` and `index/qdrant_index.py` (text only). The eleven pinned files are
byte-identical to the registered ones (`git hash-object`, 2026-10-06).

## 2. Dependency direction, and the test that enforces it

```text
api, adapters, backends, ext, legacy, testing, cli_certify  ->  core  ->  pinned and agnostic modules, numpy, stdlib
```

`tests/test_core_layering.py` parses every `.py` file under `src/optivision/core/` with `ast`.

It checks:

- Every import statement, at module level or inside a function, names the standard library,
  NumPy, `optivision.core.*`, or one of these modules: `representation`, `compose`, `scoring`,
  `evaluation`, `calibration`, `stages`, `stages.base`, `stages.merge`, `stages.project`,
  `stages.quantize`, `storage`. Relative imports are resolved first.
- From `optivision.storage` only `ExactIndex` and `TieredIndex` may be taken. From
  `optivision.stages` the name `SpatialPruner` may not.
- SciPy and torch may be imported only inside a function.
- Any other third-party module (PIL, yaml, typer, rich, qdrant_client, transformers, ...) fails.
- No file under `src/optivision/` (the whole package, not only `core`) imports a research script
  (`x_common`, `e1_stats`, `e1_common`, `e1_1_study`, `x1_agreement`, `x2_budget`, `x3_symmetry`,
  or `scripts`).

It does not check:

- What `import optivision.core` executes at run time. Importing any submodule runs
  `optivision/__init__.py`, which imports the original page recipe (see below).
- Imports made by string: `importlib.import_module`, `__import__`, `exec`.
- The other direction. Nothing in this test limits what `api`, `adapters`, `backends`, `ext`,
  `legacy` or `testing` import. `tests/test_core_retrieve.py::TestSourceProperties` checks the
  imports of `core/retrieve` and `backends` and that they read no environment variable.
- Whether the allow-listed modules are themselves pure. `optivision.stages` imports page code
  through `stages/prune.py`, and `optivision.storage` imports `optivision.types`.
- Cycles inside `core`, or which names are used from an allowed module (except the two above).

```python
import sys

import optivision.core  # noqa: F401

print(sorted({m.split(".")[0] for m in sys.modules} & {"PIL", "yaml", "numpy", "scipy", "torch"}))
```

```text
['PIL', 'numpy', 'yaml']
```

The layering is a source-level property today (design decision D2).

## 3. Facades over byte-pinned files, and why they are pinned

Eleven library files are pinned by git blob hash by the research harness: `scripts/research/
x_common.py` (`CODE_BLOBS`, `check_code`, read-only) and
`tests/test_x_harness.py::test_frozen_code_is_the_registered_code`. The registered runs of X1 to
X3 recorded those hashes and the hash of the `src/optivision` tree. A byte change, including
turning a file into a re-export shim, fails the check and the gate of a registered run. The hash
of the default candidate list pins, in the same way, the stage names, parameter names and
defaults that enter `Pipeline.to_dict()`. So the pinned files stay where they are and the core
wraps them.

| pinned file | core module | what the core adds |
|---|---|---|
| `representation.py` | `core/vectors.py` | `validate_corpus`, `corpus_identity`, `as_host_array`, `EncoderIdentity` |
| `compose.py` | `core/pipeline.py` | `pipeline_spec`, `pipeline_identity` |
| `scoring.py` | `core/similarity.py` | nothing: the same functions, with their semantics written down |
| `evaluation.py` | `core/evaluate/metrics.py`, `reference.py` | `metrics_from_ranking`; `LabelReference` and `RankingReference` call `relevant_from_qrels` and `relevant_from_baseline` |
| `calibration.py` | `core/selection/spaces.py`, `core/evaluate/collect.py` | `CandidateSpace` from the library's search spaces; evidence through `score_candidate` |
| `stages/base.py`, `merge.py`, `project.py`, `quantize.py` | `core/stages/` | traits, fitted state, byte accounting, `RotationProjector` |
| `compression/__init__.py`, `compression/binary.py` | none directly | reached through `stages/quantize.py` |

## 4. What is new code

- `core/identity.py`: canonical JSON (the same bytes as `json.dumps(obj, sort_keys=True,
  separators=(",", ":"))` for JSON-native input) and digests.
- `core/stages/traits.py`, `state.py`, `accounting.py`, `rotation.py`: declared basis traits (with
  the data conditions they hold under and who declared them) and the pipeline rule; fitted state as arrays with digests and a checked restore; the size key beside
  what it leaves out; a seeded rotation stage.
- `core/selection/`: the candidate list, the arithmetic of the rule (a port of the research's
  Method C, tested against the frozen functions), the two rules, the decision record, `select`.
- `core/evaluate/`: `Evidence`, the two references, `collect_evidence`, `compress_selected`.
- `core/retrieve/`: `Ranking`, the protocols, the retrievers.
- `core/artifact/`: the certificate, its verification, the artifact bundle, the code identity.
- `backends/`: wrappers with `BackendInfo`; `adapters/arrays.py` and `nemotron.py`; `testing/`;
  `api.py`; `cli_certify.py`.

No new native scoring kernel exists. Scoring is the library's decode-first `maxsim_matrix`.

## 5. Where vision-specific code lives

Page and patch-grid code stays at its old paths (`types.py`, `pruning/`, `encoders/`, `ingest.py`,
`viz.py`, `stages/prune.py`) and is listed in the namespace `optivision.ext.vision`, which
re-exports the same objects and resolves them on first access. Layout reaches the core only as
arrays: `positions` and `protected` in the corpus, per-document `metadata`, optional `images`. The
core never sees a `PatchGrid` or a `PageEncoding`. `SpatialPruner` needs page images and is not a
core stage. `RedundancyPruner` works on vectors alone and is taken by `core/stages/reducers.py`
from `optivision.stages`. `optivision.legacy` lists the original recipe in the same way.

## 6. From vectors to a searchable index

| step | call |
|---|---|
| 1. vectors become a checked corpus | `ArrayAdapter(identity).encode_documents(arrays, ids=...)` or `MultiVectorCorpus.from_arrays`; `validate_corpus`, `corpus_identity` |
| 2. the candidate list is fixed | `CandidateSpace.default(dim)`, `.wide(dim)` or `.from_families({...})` |
| 3. the reference is chosen | `LabelReference(qrels)` or `RankingReference()` |
| 4. every candidate is run on every query | `collect_evidence(corpus, queries, space, reference, metric=...)` returns `Evidence`; inside, `calibration.score_candidate` fits on documents, compresses and scores, and `evaluation.per_query_metrics` gives the metric |
| 5. the rule decides | `select(space, evidence, target=, error=, rule=, fallback=, query_budget=, order_seed=)` returns `Decision`; the arithmetic is `to_y`, `betting_pvalues`, `fixed_sequence_by_family`, `smallest` |
| 6. the chosen pipeline is fitted and applied | `compress_selected(corpus, decision.selected, expected_fitted=...)` returns `(Pipeline, CompressedCorpus)` and raises if the fit differs from the one evaluated |
| 7. the decision is recorded | `create_certificate(decision, evidence=, space=, corpus=, queries=, reference=, pipeline=, compressed=)`; optionally `sign_certificate` |
| 8. stored | `save_artifact(path, certificate=, compressed=, pipeline=, evidence=)`, `load_artifact(path)` |
| 9. checked | `verify_certificate(cert, evidence=, corpus=, queries=, reference=, pipeline=, compressed=, key=)` returns `VerificationReport` |
| 10. searched | `CompressedRetriever(compressed, pipeline).search(queries, k)`, `ExactRetriever`, `TieredRetriever`; a backend through `make_backend(index)` and `search_many` |

`optivision.certify()` runs steps 2 to 7 in that order (with the default list unless you give one) and adds no selection logic of its own;
`CertifiedResult.save` is step 8 and `optivision.verify` is step 9. Its design is in
[API_DESIGN.md](../API_DESIGN.md).

## 7. Extension points

**A new stage with declared traits**

1. Subclass `TokenReducer`, `DimensionReducer` or `Quantizer`, give it a `name`, implement
   `params()`, `fit` and its transform, and register it with `@register`.
2. Declare its traits with `register_traits(name, StageTraits(...))`, or as a function of the
   instance when they depend on parameters. State `fit_uses_queries` truthfully: an undeclared
   stage is `unknown`, and the evidence collector refuses a stage that reads queries when fitted.
   List in `conditions` the data your declared invariance holds for. The certificate records your
   declaration as `declared_by = "third_party"`: declared, not examined by the library.
3. If it holds fitted state, implement `get_state()` and `set_state(state)` with arrays and
   JSON-native scalars, so that artifacts can store it. `fitted` stays false until `set_state` ran.
4. Run `optivision.testing.check_stage(stage, corpus, queries)`: description round trip,
   determinism, input unchanged, traits declared.
5. Add a test of the declared invariance (apply the group element to documents and queries, refit,
   compare score matrices), below `orthogonal` a witness that a rotation changes the scores, and
   for each condition a witness: data that break it and move the scores.

**A new selection rule**

1. Provide `name`, `certifying`, `config()`, `assumptions()` and `decide(...)` returning a
   `RuleResult`, as `FixedSequenceBetting` does. Call `validate_evidence_arrays` first; never
   coerce or drop input.
2. A rule that is not one of the library's own classes (exact type; a subclass is not) is
   never certified, whatever `certifying` says: `Decision.certified` is false, and the
   certificate records the class and the declaration as the author's. Declare `certifying`
   truthfully for your own records.
3. Put your statement and every assumption in `assumptions()`; the certificate records them as
   declared by the rule's author and not examined by the library.
4. `rule_from_name` knows only the two built-in rules, so for any other rule the replay and
   rule-text checks of `verify_certificate` are reported as skipped; the record is not certified.
5. Test it against a reference implementation, and say what has and has not been reviewed.

**A new reference**

1. Provide `construct` (a string that says what is measured), `relevant(corpus_ids, query_ids,
   base_scores)` returning one integer index array per query, and `identity()` (JSON-native).
2. Return an empty array for a query with no relevant document; the collector removes it for
   every candidate together.
3. Never reuse `labels` or `agreement@1` for another quantity, and never merge constructs.
4. The certificate adds its agreement caveat only for constructs named `agreement@...`; if yours
   is not retrieval quality, say so in the name and in your report.
5. Add tests for the identity and for the empty case.

**A new backend** (details in `../BACKEND_ARCHITECTURE.md`, section 13)

1. Wrap an index that implements `BaseIndex` and do not edit it; hold it as `.index`.
2. State `BackendInfo` truthfully: `exact`, `score_kind`, `tie_break`, `dense_scores`, `upsert`.
3. `search_many` validates with `check_k` and `as_queries`, refuses what the engine mishandles
   before sending anything, and returns a `Ranking`.
4. Import the engine inside functions and register the wrapper in `make_backend`.
5. Add it to `BACKEND_FACTORIES` in `tests/test_backends_protocol.py` and write its own test file.

**A new adapter** (details in `../ADAPTER_ARCHITECTURE.md`, section 7)

1. Use `CallableAdapter` for your own model, or subclass `EncoderAdapter`.
2. Route arrays through `corpus_from_arrays`, drop padding rows with the attention mask, and say
   which dtype comes out.
3. Implement `identity()` with the facts you have; leave the rest `None`.
4. Run `optivision.testing.check_adapter(adapter, documents, queries)`.
5. A registry entry starts `unverified` or `untested`; raise it only with a committed
   measurement.

## 8. What was deliberately not done

- No physical move of a pinned file, and no edit of one (section 3).
- No lazy top-level import of the page recipe: `import optivision` still imports it eagerly
  (design decision D2). A NumPy-only import closure needs the extras split, and both are release
  decisions.
- No new native scoring kernel. Q5 and Q6 measured native compressed scoring as slower on one
  stack, and decode-first scoring stays the reference.
- No version bump: `__version__` stays `"0.3.0"`, and the certificate records the version and a
  digest of the decision-relevant source files.
- No change to `optimize()`, `calibrate()`, the file formats, the registry names, the model
  registry statuses, `NumpyIndex` and `QdrantIndex` (docstrings aside), or the research record.
