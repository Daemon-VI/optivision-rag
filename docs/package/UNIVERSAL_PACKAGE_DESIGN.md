# OptiVision package design: a universal interface and decision framework

Status: design contract, 2026-10-06. Branch `package-universal`, based on `research-investigation`
at `91564fd`. This document is the contract the implementation is built against. It rests on
three read-only audits in this directory: `ARCHITECTURE_AUDIT.md`, `METHOD_C_SPEC.md`,
`RETRIEVAL_BACKEND_AUDIT.md`.

Nothing here is a research result. Every statement about what an operator does to retrieval
quality is out of scope of this document; see `FINAL_RESEARCH_BOUNDARY.md`.

## 1. What "universal" means here

It does **not** mean that one compression configuration works on every encoder. The research
record does not support that (X3: the registered basis-invariance hypothesis H5 failed), and the
package must not imply it.

It means that any late-interaction encoder that can hand over `[n_vectors, dim]` float arrays per
document and per query can use the same:

1. interface (vector contract),
2. pipeline composition (reducers, projections, codecs),
3. candidate generation (a frozen, hashed candidate space),
4. evaluation (per-query evidence against a stated reference),
5. selection (a rule that may select a candidate or abstain),
6. certification (a record of what was decided, under which stated assumptions),
7. artifact representation (codes, fitted state, certificate),
8. backend abstraction (queries in, ranked ids out).

### 1.1 The decomposition, and a correction to it

The brief proposed four categories: universal interface, model-dependent operators,
model-dependent performance, model-independent statistical procedure. The repository evidence
supports that split only after two corrections, so the package uses five layers:

| layer | what it is | status |
|---|---|---|
| A. Interface contracts | shapes, dtypes, identities, serialisation | universal by construction; enforced by validation and tests |
| B. Structural operator properties | what an operator reads and which transformations leave its scores unchanged (derived from code) | universal per operator; checked by property tests; says nothing about quality |
| C. Development-informed defaults | the default candidate space, the chain order, the bet cap | frozen and declared, but **not neutral**: they were shaped by the development encoders |
| D. Decision procedure: validity | the error statement of the selection rule | does not depend on the encoder, **given its assumptions**; conditional, not unconditional |
| E. Empirical outcome | whether anything is selected, how much compression, what retention | depends on encoder, coordinate basis, corpus, query distribution, query budget, target and reference |

Corrections to the four-way split:

- "Model-independent statistical procedure" conflates validity (D) with yield (E). The rule looks
  only at arrays, so its error statement does not depend on the encoder; whether it selects
  anything, and what, does (X2: nothing is certified at 0.99 with pool-sized budgets; X3: the
  certified outcome on one store changed materially under rotation).
- "Model-dependent operators" conflates code with effect. The operators run on any float array
  (A) and have structural properties that can be proven from code (B); their effect on quality is
  empirical (E).
- The default candidate space is neither universal nor chosen per model: it is a frozen list that
  earlier development results shaped (C). "Model-blind" describes what the procedure may inspect.
  It is not evidence of transfer to an unseen encoder; that has not been tested (XH not run).

## 2. Constraints from the repository

1. **Eleven library files are byte-pinned by the frozen research harness**
   (`scripts/research/x_common.py:68-83`, `tests/test_x_harness.py`): `calibration.py`,
   `compose.py`, `scoring.py`, `evaluation.py`, `representation.py`,
   `stages/{base,merge,project,quantize}.py`, `compression/{__init__,binary}.py`. They are not
   edited, moved or reformatted. New layers are facades over them plus new files.
2. The hash of the default candidate space (`S0_SPEC_SHA256`) is pinned; stage names, parameter
   names and defaults that enter `Pipeline.to_dict()` are not changed.
3. v0.3.0 compatibility surface: the 40 names of `optivision.__all__`, 201 `(module, name)` import
   pairs used by tests, scripts and apps, the CLI entry points, file formats, registry names.
4. `reports/research/**`, `scripts/research/**`, `paper/**`, `reports/universal/**` and the
   registered inputs are never written by package work.

## 3. Decisions

| id | decision | reason |
|---|---|---|
| D1 | Additive layering. Pinned files stay where they are. No physical moves except `adapters.py` -> `adapters/__init__.py` (`git mv`). | Constraint 1; alias moves of unpinned modules buy little and risk monkeypatched names. |
| D2 | `import optivision` stays eager. Core purity is enforced at source level by an `ast` test with an explicit allow-list. A numpy-only import closure is deferred. | A lazy `__init__` changes import side effects and the `optivision.optimize` function/module binding; it also needs the dependency extras split. Both are release decisions. |
| D3 | The selection package is `optivision.core.selection` (not `select`). | `calibration.select` already exists; avoids a stdlib-looking name. |
| D4 | The certifying rule is a line-for-line port of the research's Method C, named for what it is: `FixedSequenceBetting`. Parity with the frozen functions is tested. It is described as an application of published tools, never as a new method. | `METHOD_C_SPEC.md`. |
| D5 | Outcomes are `SELECT`, `ABSTAIN`, `FALLBACK`. Nothing certified means `ABSTAIN` and the uncompressed index. `FALLBACK` exists only when the caller passes an explicit fallback pipeline; it is recorded as uncertified. There is no silent float16 fallback. | Spec section 4: the research has no float16 fallback; float16 must itself be certified. |
| D6 | The rule raises on non-finite or out-of-range input, on a zero-mean baseline, on `n = 0`, on `T` outside (0, 1], on `delta` outside (0, 1). | The frozen functions return NaN or plausible values silently; the harness stops the run. |
| D7 | Queries enter the test in the order given. An optional seeded permutation (independent of the data) is recorded. A query budget takes a prefix of that order and raises if it exceeds the pool. | Assumption A5; spec section 1.4. |
| D8 | One `(target, error)` per call. | Several looks at one sample are outside the statement. |
| D9 | The size key is the library's `bytes_per_doc` (codes plus codec overhead), as in the research. Fitted-state bytes are reported separately and are not in the key. | Parity; spec ambiguity 13. |
| D10 | A candidate space has an identity: sha256 of the canonical JSON of its ordered `{position, family, pipeline}` list, the same canonical form as the research, so the default space at d = 96 or 128 with SciPy hashes to the registered value. | Cross-check without importing research code. |
| D11 | Every stage has declared structural traits, including the group of basis changes that leaves its scores unchanged. A pipeline is `basis_sensitive` unless every stage is invariant under any orthogonal change of basis. The certificate records it. Unknown stages are treated as sensitive. | X3; no invariance is assumed. |
| D12 | A certificate is canonical JSON with a SHA-256 content digest and identity digests of everything the decision depended on. Verification has three levels: integrity, replay from evidence, binding to live objects. An optional HMAC is available; without a secret key the format gives integrity checking, not security. | Brief; no cryptographic claim beyond what is implemented. |
| D13 | A new artifact bundle stores fitted stage state, which `save_compressed` (pinned) cannot. The v1 formats stay readable. | Audit finding 9. |
| D14 | Retrieval seam: `Ranking`, `Retriever`, `DenseScorer`, `Reranker`; retrievers delegate to `ExactIndex` / `TieredIndex`. Metrics from a ranking are a new function with parity tests; pinned `evaluation.py` is not refactored. | Retrieval audit section 6. |
| D15 | `optivision.backends`: `Backend` protocol, `BackendInfo`, wrappers over `NumpyIndex` and `QdrantIndex`, which are not changed. `BackendInfo` records score kind and tie-break so scores from different backends are never compared. | Retrieval audit findings 3-5. |
| D16 | `optivision.adapters` becomes a package. The model registry and its evidence statuses are not changed. New: `ArrayAdapter` (bring your own vectors), `EncoderIdentity`. | Audit 3.4, 9.3. |
| D17 | Public API is additive: `optivision.certify()`, `verify_certificate()`, and the types they return. `optimize()` and `calibrate()` are unchanged. | Compatibility. |
| D18 | `__version__` stays `"0.3.0"` on this branch. No release. Certificates record the version and a digest of the decision-relevant source files. | No release without the owner's decision. |
| D19 | New tests are flat files `tests/test_core_*.py`, `tests/test_backends_*.py`, `tests/test_adapters_*.py`, `tests/test_api_*.py`. | `tests/` is a package without sub-packages. |
| D20 | Design and reference documents live in `docs/package/`. | Separate from `docs/UNIVERSAL.md`, which is a results record. |

## 4. Layout

```
src/optivision/
  core/                     model-agnostic layer
    identity.py             canonical JSON, sha256, array digests
    vectors.py              facade over representation.py + contract validation + identities
    pipeline.py             facade over compose.py + pipeline identity
    similarity.py           facade over scoring.py
    stages/
      __init__.py           contracts and agnostic stages (re-exports)
      reducers.py projections.py codecs.py     re-exports by kind
      traits.py             StageTraits, BasisInvariance, pipeline_traits
      state.py              fitted_state / restore_fitted_state
      rotation.py           RotationProjector (engineering capability, experimental)
      accounting.py         ByteAccounting
    retrieve/               Ranking, Retriever, ExactRetriever, CompressedRetriever, tiered
    evaluate/
      metrics.py            facade + metrics_from_ranking
      reference.py          LabelReference, RankingReference (agreement)
      evidence.py           Evidence, collect_evidence
    selection/
      spaces.py             CandidateSpec, CandidateSpace
      statistics.py         to_y, betting_pvalues, fixed_sequence_by_family, smallest
      rules.py              SelectionRule, FixedSequenceBetting, BonferroniBetting
      decision.py           Outcome, Decision
      engine.py             select()
    artifact/
      certificate.py        Certificate, create_certificate, verify_certificate
      bundle.py             save_artifact, load_artifact
      codeid.py             code_identity()
  adapters/                 EncoderAdapter, registry, ArrayAdapter, SentenceTransformersAdapter
  backends/                 Backend, BackendInfo, NumpyBackend, QdrantBackend
  ext/vision/               namespace: page types, pixel/grid pruning, page encoders
  legacy/                   namespace: Config, OptiVisionRAG, Compressor, TokenPruner
  testing/                  synthetic data, basis transforms, conformance checks
  api.py                    certify(), CertifiedResult
```

Dependency direction (enforced by `tests/test_core_layering.py`):

```
api, adapters, backends, ext, legacy, testing  ->  core  ->  (pinned agnostic modules, numpy, stdlib)
```

A module under `core/` may import, at source level, only: the standard library, numpy,
other `optivision.core` modules, and this allow-list of pinned or agnostic modules:
`optivision.representation`, `optivision.compose`, `optivision.scoring`, `optivision.evaluation`,
`optivision.calibration`, `optivision.stages` (agnostic names only, never `SpatialPruner`),
`optivision.stages.{base,merge,project,quantize}`, `optivision.storage` (`ExactIndex`,
`TieredIndex` only). SciPy and torch may be imported lazily inside functions. Never: `PIL`,
`yaml`, `typer`, `rich`, `qdrant_client`, `transformers`, `colpali_engine`, `optivision.types`,
`optivision.config`, `optivision.encoders`, `optivision.index`, `optivision.pruning`,
`optivision.pipeline`, `optivision.adapters`, `optivision.backends`, `optivision.ext`,
`optivision.legacy`, `optivision.api`.

Known limit (technical debt): importing any `optivision.*` module still executes
`optivision/__init__.py`, which imports the page recipe and therefore PIL and yaml. The layering
is a source-level property today.

## 5. Contracts

Python 3.10 syntax, `from __future__ import annotations`, ruff line length 100. Frozen dataclasses
for value objects. No global mutable state other than the existing stage registry and the trait
registry beside it. No environment variables read by new code. No implicit dtype or device
conversion: a function either keeps the dtype it was given or names the conversion in its
signature and docstring.

### 5.1 `core.identity`

```python
def canonical_json(obj) -> bytes      # sorted keys, separators (",", ":"), ASCII, no NaN/inf; numpy scalars and
                                      # arrays become Python values; tuples become lists; unknown types raise TypeError
def sha256_hex(data: bytes) -> str
def digest(obj) -> str                # sha256_hex(canonical_json(obj))
def array_digest(a: np.ndarray) -> str   # dtype (little-endian), shape and C-order bytes, hashed in blocks
def strings_digest(items: Sequence[str]) -> str
```

`canonical_json` is byte-identical to `json.dumps(obj, sort_keys=True, separators=(",", ":"))`
for JSON-native input, which is the form the research used for the candidate-space hash.

### 5.2 `core.vectors`

Re-exports `MultiVectorCorpus`, `MultiVectorRepresentation`, `as_corpus`. Adds:

```python
class ContractError(ValueError)
@dataclass(frozen=True)
class CorpusIdentity: n_docs: int; n_vectors: int; dimension: int; dtype: str
                      vectors_sha256: str; offsets_sha256: str; ids_sha256: str
                      def digest(self) -> str
def corpus_identity(corpus) -> CorpusIdentity
def validate_corpus(corpus, *, name="corpus", expect_dim=None, expect_dtype=None, require_finite=True,
                    allow_empty_docs=True, require_unique_ids=True) -> None      # raises ContractError
@dataclass(frozen=True)
class EncoderIdentity: model_id: str; revision: str | None; dimension: int; modality: str  # "text"|"image"|"other"
                       normalized: bool | None; inference_dtype: str | None; extra: Mapping[str, str]
                       def to_dict(self) -> dict;  @classmethod from_dict
```

`EncoderIdentity` is descriptive metadata for certificates and adapters. No core function may
branch on it.

### 5.3 `core.stages`

Re-exports the contracts (`Stage`, `TokenReducer`, `DimensionReducer`, `Quantizer`, `register`,
`stage_from_dict`, `DocView`, `Reduction`) and the agnostic stage classes.

```python
class BasisInvariance(str, Enum):
    ORTHOGONAL = "orthogonal"                  # scores unchanged under any orthogonal change of basis (after refit)
    SIGNED_PERMUTATION = "signed_permutation"  # unchanged only when axes are permuted and/or sign-flipped
    NONE = "none"                              # not even that (e.g. keeps the first k coordinates)
    UNKNOWN = "unknown"                        # undeclared: treated as NONE

@dataclass(frozen=True)
class StageTraits:
    name: str
    kind: str                          # "tokens" | "dimensions" | "bits"
    basis_invariance: BasisInvariance
    canonicalises_basis: bool = False  # output coordinates are determined by the data, not by the input basis (PCA)
    randomised_basis: bool = False     # applies an internal seeded random orthogonal map (realisation depends on basis)
    reads_vector_values: bool = True
    has_fitted_state: bool = False     # may depend on params: see traits_of(stage)
    fit_uses_queries: bool = False     # must be False for every built-in stage
    stochastic: bool = False           # uses a seeded RNG
    lossless: bool = False
    notes: str = ""

def register_traits(name: str, traits: StageTraits | Callable[[Stage], StageTraits]) -> None
def traits_of(stage: Stage | str) -> StageTraits          # unknown name -> UNKNOWN traits, never an error
@dataclass(frozen=True)
class PipelineTraits: stages: tuple[StageTraits, ...]
                      basis_invariance: BasisInvariance   # unconditional; canonicalisation is NOT credited here
                      conditionally_invariant: bool       # every non-ORTHOGONAL stage follows a canonicalising stage
                      basis_sensitive: bool                # basis_invariance != ORTHOGONAL (conservative)
                      has_fitted_state: bool; stochastic: bool; notes: tuple[str, ...]
def pipeline_traits(pipeline: Pipeline) -> PipelineTraits
```

Pipeline rule. Walk the stages in order with a state in {input, mixed}; start in `input`.
For each stage take its declared invariance; in state `mixed` a `SIGNED_PERMUTATION` stage
counts as `NONE` (a fixed, data-independent orthogonal map was applied before it, so even a
permutation of the input axes changes what the codec sees). The pipeline's invariance is the
weakest value met (`ORTHOGONAL` > `SIGNED_PERMUTATION` > `NONE`; `UNKNOWN` counts as `NONE`
and is reported as `UNKNOWN`). A stage with `randomised_basis=True` switches the state to
`mixed`. `basis_sensitive = (basis_invariance != ORTHOGONAL)`.

Canonicalisation is recorded but not credited. If every stage weaker than `ORTHOGONAL` comes
after a stage with `canonicalises_basis=True` (PCA), `conditionally_invariant` is True: the
codec then sees coordinates fixed by the data rather than by the input basis, provided the
retained principal directions are well separated (equal eigenvalues leave the basis
arbitrary). `basis_sensitive` stays True for such a pipeline, because the condition is on the
data.

These are statements about code, verified on synthetic data by property tests (apply a group
element to documents and queries, refit, compare score matrices; and, for every stage declared
weaker than `ORTHOGONAL`, a witness case in which a rotation does change the scores). They are
not statements about retrieval quality. The conditional invariance of a `pca -> codec` chain is
an engineering property that no registered experiment evaluated.

```python
def fitted_state(stage: Stage) -> dict[str, np.ndarray | float | int | str | None]   # {} if stateless
def restore_fitted_state(stage: Stage, state: Mapping) -> Stage                       # same class, unfitted instance in
def fitted_state_digest(stage: Stage) -> str | None
class RotationProjector(DimensionReducer)   # registry name "rotation"; seeded orthogonal matrix, output_dim == input_dim
@dataclass(frozen=True)
class ByteAccounting: code_bytes: int; codec_overhead_bytes: int; fitted_state_bytes: int; offsets_bytes: int
                      float32_bytes_before: int; n_docs: int
                      @property selection_bytes_per_doc  # (code_bytes + codec_overhead_bytes) / n_docs: the size key
                      @property total_bytes_per_doc
def byte_accounting(compressed: CompressedCorpus, pipeline: Pipeline) -> ByteAccounting
```

Built-in fitted state is read through a table of attribute names per stage class (the classes
are pinned). A new stage may instead implement `get_state()` / `set_state(state)`.

### 5.4 `core.pipeline`, `core.similarity`

Re-exports (`Pipeline`, `CompressedCorpus`, `save_compressed`, `load_compressed`;
`maxsim_matrix`, `maxsim_pair`, `rank`, `segment_reduce`). Adds
`pipeline_identity(pipeline) -> str` (digest of `to_dict()`) and
`fitted_identity(pipeline) -> dict[str, str | None]`.

### 5.5 `core.retrieve`

As in `RETRIEVAL_BACKEND_AUDIT.md` section 6.1: `Ranking`, protocols `Retriever`, `DenseScorer`,
`Reranker`; `ExactRetriever`, `CompressedRetriever`, `TieredRetriever` (delegates to
`TieredIndex`, checks that both tiers hold the same ids in the same order), `TwoTierRetriever`
(any first stage plus any reranker). Invariants I1-I9 of that section hold. Scoring is
decode-first; no native compressed kernel is added.

### 5.6 `core.evaluate`

```python
class LabelReference:      construct = "labels"        # relevance labels (qrels)
class RankingReference:    construct = "agreement@1"   # the uncompressed index's own top document
    # both: relevant(corpus_ids, query_ids, base_scores) -> list[np.ndarray];  identity() -> dict
@dataclass(frozen=True)
class Evidence:
    metric: str; construct: str
    baseline: np.ndarray              # float64 [n]
    candidates: np.ndarray            # float64 [K, n]
    query_ids: tuple[str, ...]        # in test order
    bytes_per_doc: np.ndarray; vectors_per_doc: np.ndarray       # float64 [K]
    space_identity: str
    dropped_queries: tuple[str, ...]  # removed by the label-only rule (no relevant document), for all candidates
    fitted: tuple[dict[str, str | None], ...]   # per candidate: fitted-state digests
    def digest(self) -> str
def collect_evidence(corpus, queries, space, reference, *, metric="ndcg@5", progress=None) -> Evidence
def metrics_from_ranking(indices, relevant, metrics) -> dict[str, np.ndarray]
```

`collect_evidence` scores every candidate of the space on every query with the library's own
functions (`calibration.score_candidate`, `evaluation.per_query_metrics`). Stages are fitted on
documents only. Queries with no relevant document are removed for the baseline and all
candidates together; never per candidate. `RankingReference` measures agreement with the
uncompressed ranking, which is not retrieval quality; its construct name appears in every
result that used it.

### 5.7 `core.selection`

```python
@dataclass(frozen=True)
class CandidateSpec: position: int; family: str; pipeline: dict     # pipeline = Pipeline.to_dict()
class CandidateSpace:                                               # immutable, ordered
    name: str; candidates: tuple[CandidateSpec, ...]
    @classmethod from_families(families: Mapping[str, Sequence[Pipeline]], name="custom")
    @classmethod default(dim) ; wide(dim)        # default raises without SciPy unless allow_reduced=True
    families -> tuple[str, ...] ; n_families ; identity -> str ; pipelines() -> list[Pipeline]
    is_registered_default -> bool                # identity == REGISTERED_DEFAULT_SPACE_SHA256

class Outcome(str, Enum): SELECT = "select"; ABSTAIN = "abstain"; FALLBACK = "fallback"

class SelectionRule(Protocol):
    name: str
    certifying: bool
    def config(self) -> dict
    def assumptions(self) -> tuple[str, ...]
    def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc, target, error) -> RuleResult

@dataclass(frozen=True)
class RuleResult: selected: int | None; certified: tuple[int, ...]; p_values: tuple[float, ...]; level: float

@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    selected: CandidateSpec | None      # None on ABSTAIN; the fallback spec on FALLBACK
    certified: bool                     # True only for SELECT under a certifying rule
    certified_positions: tuple[int, ...]
    p_values: tuple[float, ...]; level: float
    target: float; error: float
    rule: dict; assumptions: tuple[str, ...]
    n_queries: int; metric: str; construct: str
    space_identity: str; evidence_digest: str
    query_order: dict                   # {"kind": "given"} or {"kind": "seeded_permutation", "seed": s}; budget
    def to_dict(self) -> dict

def select(space, evidence, *, target, error, rule=None, fallback=None, query_budget=None, order_seed=None) -> Decision
```

`FixedSequenceBetting` is the port of Method C: level `error / n_families`, fixed-sequence
testing inside each family in list order, choice by `(bytes, vectors, position)`.
`BonferroniBetting` is method D of E1 (level `error / K`, no ordering needed), for candidate lists
without a meaningful order. Both use the same betting p-values. Neither is a new method.

The statement a certifying rule makes is the one in `METHOD_C_SPEC.md` section 6, with its six
assumptions, quoted in `Decision.assumptions`. The package never rephrases it as a conditional
probability, as guaranteed retention, or without its conditions.

### 5.8 `core.artifact`

See `CERTIFICATE_SPEC.md`. `create_certificate(...)`, `verify_certificate(...) ->
VerificationReport`, `save_artifact`, `load_artifact`, `code_identity()`.

### 5.9 `optivision.api`

```python
def certify(vectors, queries, qrels=None, *, target=0.97, error=0.05, metric="ndcg@5", reference="auto",
            space=None, rule=None, query_budget=None, order_seed=None, fallback=None, encoder=None,
            progress=None) -> CertifiedResult
```

`CertifiedResult`: `decision`, `outcome`, `selected`, `pipeline` (fitted), `compressed`,
`certificate`, `evidence`, `save(path)`, `report()`. On `ABSTAIN`, `compressed` is the corpus
under the empty pipeline (float32), as `optimize()` returns today when nothing is feasible.

## 6. Basis dependence

- No code assumes `codec(x)` and `codec(xR)` behave alike for an orthogonal `R`.
- Stage traits make the dependence explicit; pipelines and certificates carry
  `basis_sensitive`.
- A certificate binds the decision to the vectors as given (corpus digest). Rotating the
  vectors yields a different corpus identity, so a certificate never transfers across bases.
- `RotationProjector` and `optivision.testing.basis_probe` are diagnostics. X3 measured that
  rotation can reduce certified compression; a rotation stage is not presented as an improvement.
- Candidate families that are less basis-sensitive by construction (a canonicalising projection
  before the codec, learned transforms, basis search) are design possibilities, not results.

## 7. What is not changed

The pinned files; `optimize()` and `calibrate()` behaviour; file formats; registry names; the
model registry and its evidence statuses; `NumpyIndex` and `QdrantIndex`; the CLI commands;
`reports/**`, `scripts/research/**`, `paper/**`.

## 8. Phases

1. Audit (done). 2. Core data model. 3. Stages and traits. 4. Retrieval. 5. Selection and
evidence. 6. Certificate and artifact. 7. Backends and adapters. 8. Public API. 9. Tests.
10. Engineering benchmarks. 11. Documentation. 12. Integration, build, clean-install check,
independent research-integrity audit.
