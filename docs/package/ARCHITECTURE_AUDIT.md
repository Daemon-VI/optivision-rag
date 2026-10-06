# OptiVision `src/optivision` architecture audit

Read-only audit, 2026-10-06. No source file was modified; no experiment, benchmark or test suite was run. Nothing here is a statement about retrieval quality or research results.

- Baseline: worktree `optivision-package`, branch `package-universal`, HEAD `91564fd`. `src/` differs from tag `v0.3.0` (`4d52060`) only in `storage.py` (batched `ExactIndex.rescore`, +39/-17); `tests/` adds batched-rescore tests and `test_x_harness.py`.
- Size: 45 Python modules, 9,662 lines (+4 YAML presets, 95 lines); 351 test functions in 26 files. Python 3.11.9, numpy 2.4.6.
- Paths are relative to `src/optivision/` unless prefixed (`tests/`, `scripts/`, `docs/`). `P` marks a byte-pinned file (finding 2).
- Method: every module read; import graph by `ast` (relative imports resolved; "lazy" = inside a function); consumer surface by `ast` over `tests/`, `scripts/` (research and other), `app.py`, `app/`; several small read-only probes (Appendix A). The probe scripts are not committed.

## 0. Findings that drive the restructure

1. **Two generations, confirmed by first-add dates** (`git log --diff-filter=A`). Gen-1, the original recipe (in v0.1.1, 2026-08-17..30; 27 modules, 5,293 lines): `types config pipeline ingest corpus metrics bench viz demo cli`, packages `pruning/ compression/ encoders/ index/`, `presets/`. Gen-2 (v0.2.0, 2026-09-27; 17 modules, 4,279 lines): `representation stages/ compose scoring evaluation calibration optimize pareto adapters storage benchmark cli_universal`. `encoders/nemotron.py` (v0.3.0) sits in a gen-1 package but is a gen-2 consumer. `__init__.py` (90 lines) imports both generations eagerly (`__init__.py:13-45`).
2. **The model-agnostic core is byte-pinned.** Eleven library files are pinned by git blob SHA-1 in frozen research code (`scripts/research/x_common.py:68-83`, checked by `check_code`, `:649-657`) and by `tests/test_x_harness.py:249-253`: `calibration.py compose.py scoring.py evaluation.py representation.py stages/{base,merge,project,quantize}.py compression/{__init__,binary}.py`. I ran `check_code()`: all 11 (and the 3 frozen scripts) match today. They are 3,273 lines, 34 % of the Python, and they are the core. Any byte change (including turning one into a re-export shim) fails that test and the registered-run gate. Also pinned: the sha256 of `Pipeline.to_dict()` of the 40 default candidates (`x_common.py:60,453-461,678-686`), and every commit under `src/optivision` changes the recorded library-tree identity (`:223,625-643`).
3. **No import cycle between modules, one eager hub.** Explicit edges are acyclic (top-level and lazy). But `optivision/__init__.py` imports both generations, so every submodule import has an implicit edge to it: one strongly connected component of 31 modules (36 with lazy edges). Measured: `import optivision.scoring` loads numpy, PIL and yaml, and fails if either PIL or yaml is absent.
4. **Layering violations are few and sit mostly in unpinned files.** Gen-2 -> gen-1 explicit edges: `stages/quantize.py:18-29` -> `compression`, `stages/prune.py:16-18` -> `pruning.*`, `storage.py:34` -> `types`, `representation.py:402` -> `bench` (lazy), `adapters.py:201` -> `encoders.colvlm` (lazy). One reverse edge: `index/numpy_index.py:24` -> `scoring`. The pinned files' own impurity (`compression/__init__.py:7-8` imports `config`, which imports yaml) is reachable only through unpinned files, so a numpy-only core import closure needs no pinned edit (section 9.2).
5. **Compatibility surface: 201 distinct `(module, name)` pairs over 34 modules** imported by tests (140 pairs), non-research scripts (73), frozen research scripts (31) and apps (10), plus `__all__` (40). 11 are private names; 5 of those are used by frozen research code. CLI, `python -m optivision.cli`, bundled presets and file formats add non-import contracts (section 3).
6. **Most codecs are coordinate-dependent.** `binary`, `int4`, `int8` (all scale modes) and `truncate` depend on the coordinate basis; token reducers and PCA do not; `random_projection` and `lloyd2` do in realisation but not in distribution; `float16` is near-invariant (section 5.2-5.3). 39 of the 40 default candidate slots (36 of 37 distinct pipelines) end in `int8(per_vector)`, `int4(centred)` or `binary`.
7. **Selection exists in one form only.** `calibration.py` implements the bootstrap lower-bound rule with a family walk. Method C (betting p-values, fixed-sequence testing) exists only in frozen `scripts/research/e1_stats.py`; the data it needs (per-candidate per-query metrics, families, bytes, vectors) is already produced by `score_space` (section 7.5).
8. **MaxSim is implemented 3 times in production paths** (`maxsim_matrix`, `ExactIndex.rescore`, `NumpyIndex`), plus one reference (`maxsim_pair`) and 6 legacy helpers in `compression/`, with inconsistent empty-document semantics (-inf in the new scorers, 0.0 in the legacy helpers; both pinned by tests).
9. **Artifacts carry codes and a pipeline description, not fitted state or provenance.** `save_compressed` refuses fitted codecs; PCA components and merge means are never stored; `attrs` (model, encoder) are dropped on compress; byte accounting counts only quantizer state (section 4.3).
10. **Hidden configuration:** two env vars in `src` (`OPTIVISION_SCORE_DEVICE`, `OPTIVISION_CACHE_BYTES`), SciPy presence changes the default search space (40 vs 22 candidates), two import-time global side effects (PIL pixel limit; torch TF32 flag on first GPU score), a module-level stage registry, and two fit samplers whose seed is a hard-coded `0` that no configuration reaches (`stages/merge.py:146`, `stages/quantize.py:81`).

## 1. Module inventory

Gen: 1 = original recipe, 2 = composable layer. Lines are `wc -l`.

| module | lines | gen | P | purpose |
|---|---|---|---|---|
| `__init__.py` | 90 | both | | facade: 40-name `__all__`, `__version__ = "0.3.0"` literal |
| `representation.py` | 562 | 2 | P | `MultiVectorRepresentation`, `MultiVectorCorpus` (contiguous `[N,d]` + offsets, optional aligned arrays), `.npz` I/O, adapters from gen-1 `PageEncoding` and legacy caches |
| `stages/base.py` | 287 | 2 | P | stage contracts, `STAGES` registry, `DocView`, `Reduction`, fit-sample caps |
| `stages/merge.py` | 314 | 2 | P | `AdaptiveMerge`, `HierarchicalMerge` (SciPy), `RandomPruner` |
| `stages/project.py` | 207 | 2 | P | `PCAProjector`, `RandomProjector`, `TruncateProjector`, `DimensionProjector` facade |
| `stages/quantize.py` | 342 | 2 | P | `Float32/Float16/Binary/Int8/Int4/Lloyd2Quantizer` (wrap gen-1 codec primitives) |
| `stages/prune.py` | 131 | 2 | | `SpatialPruner`, `RedundancyPruner` (wrap gen-1 `pruning.*`) |
| `stages/__init__.py` | 53 | 2 | | re-exports; importing it registers every stage |
| `compose.py` | 297 | 2 | P | `Pipeline`, `CompressedCorpus`, `save_compressed`, `load_compressed` |
| `scoring.py` | 197 | 2 | P | `maxsim_matrix`, `segment_reduce`, `rank`, `maxsim_pair`, `VectorStore` protocol |
| `evaluation.py` | 216 | 2 | P | per-query metrics, relevance from qrels/baseline, bootstrap retention and lower bound, `split_queries`, `measure` |
| `calibration.py` | 565 | 2 | P | candidate spaces, `score_space`, `select`, `calibrate`, `pseudo_queries` |
| `optimize.py` | 91 | 2 | | `optimize()` |
| `pareto.py` | 52 | 2 | | `pareto_front`, `choose` |
| `adapters.py` | 205 | 2 | | model registry with status, `EncoderAdapter`, `PageEncoderAdapter`, `SentenceTransformersAdapter`, `load_adapter` |
| `storage.py` | 205 | 2 | | `ExactIndex`, `TieredIndex`, `StorageBackend`/`NpzStorage`, `to_legacy_pages` |
| `benchmark.py` | 212 | 2 | | `Dataset`, vector/legacy dataset loaders, `legacy_variants`, `run_matrix`, result files |
| `cli_universal.py` | 343 | 2 | | Typer commands `inspect compress calibrate benchmark compare`, pipeline shorthand parser |
| `types.py` | 146 | 1 | | `PageRef PatchGrid PageEncoding PrunedPage CompressedPage SearchHit SearchResult` |
| `config.py` | 200 | 1 | | YAML `Config` and six sections, bundled presets |
| `pipeline.py` | 286 | 1 | | `OptiVisionRAG`, `IndexReport`, float-vector cache |
| `ingest.py` | 85 | 1 | | PDF/image folder -> `(PageRef, PIL.Image)` |
| `corpus.py` | 359 | 1 | | synthetic PDF corpus (reportlab), ViDoRe subset, `load_queries` |
| `metrics.py` | 172 | 1 | | id-based IR metrics, Kendall tau variants, `storage_summary` |
| `bench.py` | 587 | 1 | | ablation harness, `EncodedCorpus` encode cache |
| `viz.py` | 100 | 1 | | keep-mask and saliency overlays |
| `demo.py` | 369 | 1 | | single-page demo adapter, optional Qdrant store |
| `cli.py` | 353 | 1 | | Typer app (`make-corpus fetch-vidore index search stats bench explain init-config`); registers gen-2 commands (`:347-349`) |
| `pruning/__init__.py` | 130 | 1 | | `TokenPruner` |
| `pruning/codebook.py` | 127 | 1 | | probe-direction saliency (`fit_codebook`, `codebook_saliency`) |
| `pruning/redundancy.py` | 78 | 1 | | `prune_redundant` greedy cosine clustering |
| `pruning/saliency.py` | 86 | 1 | | pixel ink/edge saliency (PIL) |
| `pruning/spatial.py` | 78 | 1 | | `build_keep_mask`, `dilate_mask` |
| `compression/__init__.py` | 170 | 1 | P | `Compressor`, `INT8_SCALE`, `decode`, re-exports of both codecs |
| `compression/binary.py` | 116 | 1 | P | sign-bit pack/unpack, legacy MaxSim helpers |
| `compression/lloyd2.py` | 229 | 1 | | rotated 2-bit Lloyd-Max codec, legacy MaxSim helpers |
| `encoders/__init__.py` | 33 | 1 | | `get_encoder` registry |
| `encoders/base.py` | 54 | 1 | | `BaseEncoder`, `l2_normalise` |
| `encoders/colvlm.py` | 616 | 1 | | ColPali/ColQwen/ColSmol loader, patch-grid recovery, multi-GPU placement |
| `encoders/synthetic.py` | 170 | 1 | | deterministic offline stand-in encoder |
| `encoders/nemotron.py` | 129 | 1* | | NVIDIA ColEmbed 2,560/4,096-d encoder; not a `BaseEncoder`; no adapter |
| `index/__init__.py` | 51 | 1 | | `get_index`, `open_index` |
| `index/base.py` | 39 | 1 | | `BaseIndex` |
| `index/numpy_index.py` | 269 | 1 | | exact page-level MaxSim index over packed codes |
| `index/qdrant_index.py` | 261 | 1 | | Qdrant multivector backend |
| `presets/*.yaml` | 95 | 1 | | `synthetic colsmol colpali qdrant` |

`*` added in v0.3.0 (2026-10-02).

## 2. Import graph

### 2.1 Explicit internal edges (T = top-level, L = lazy)

```
__init__             T: calibration compose config optimize pipeline representation scoring stages types
adapters             T: representation                                  L: encoders.colvlm
bench                T: compression config corpus encoders index.numpy_index ingest metrics pruning types
benchmark            T: compose evaluation representation stages
calibration          T: compose evaluation representation scoring stages
cli                  T: cli_universal config            L: __init__ bench corpus ingest pipeline viz
cli_universal        T: -                               L: benchmark calibration compose representation stages
compose              T: representation stages.base stages.quantize
compression/__init__ T: compression.binary compression.lloyd2 config types
demo                 T: compression config encoders ingest pruning types viz   L: index.qdrant_index pruning
encoders/__init__    T: config encoders.base encoders.synthetic        L: encoders.colvlm
encoders.base        T: types
encoders.colvlm      T: encoders.base types
encoders.nemotron    T: encoders.colvlm
encoders.synthetic   T: encoders.base types
evaluation           T: scoring
index/__init__       T: compression config index.base index.numpy_index   L: index.qdrant_index
index.base           T: types
index.numpy_index    T: compression index.base scoring types
index.qdrant_index   T: compression index.base types
ingest               T: config types
optimize             T: calibration compose representation
pipeline             T: compression config encoders index ingest pruning types   L: compression
pruning/__init__     T: config pruning.{codebook,redundancy,saliency,spatial} types
representation       T: -                                               L: bench
stages/__init__      T: stages.{base,merge,project,prune,quantize}
stages.base          T: representation
stages.merge         T: stages.base
stages.project       T: representation stages.base
stages.prune         T: pruning.{redundancy,saliency,spatial} stages.base
stages.quantize      T: compression representation stages.base
storage              T: compose representation scoring types
viz                  T: types
no internal imports: config corpus metrics pareto scoring types compression.{binary,lloyd2} pruning.{codebook,redundancy,saliency,spatial}
```

Fan-in (importing modules): `types` 15, `representation` 11, `config` 10, `compression` 8, `compose` 6, `stages.base` 6. `config` is the real gen-1 hub (10 importers, including `compression`, `index`, `encoders`, `ingest`, `pruning`).

### 2.2 Cycles and cross-generation edges

- Explicit edges: no cycle, with or without lazy edges (SCC over the graph above).
- Implicit parent-package edges: `optivision/__init__` imports `.pipeline` and everything else, and each submodule import first executes `optivision/__init__`. Result: a 31-module SCC (36 with lazy edges: adds `bench corpus encoders.colvlm index.qdrant_index metrics`). It does not fail today because no submodule reads a name from the package namespace at import time (the only `from . import __version__` is inside a function, `cli.py:29`).
- `stages/__init__.py:16-26` imports `merge project prune quantize` eagerly, so importing any `optivision.stages.*` (including `stages.base`) also imports `stages.prune` -> `pruning/__init__` -> PIL, `config` -> yaml.
- Gen-2 -> gen-1: see finding 4; also `cli.py:347` (gen-1 CLI hosts gen-2 commands) and `benchmark.py` via `MultiVectorCorpus.from_legacy_cache` -> `bench`. Gen-1 -> gen-2: `index/numpy_index.py:24` only.

### 2.3 Third-party imports

| library | top-level in | lazy in | pyproject |
|---|---|---|---|
| numpy | 34 modules (all except `__init__ cli config corpus demo encoders/__init__ index/__init__ ingest optimize pareto stages/__init__`) | - | core |
| PIL | `bench demo encoders/{base,colvlm,synthetic} ingest pipeline pruning/__init__ pruning/saliency viz` (10) | `viz.py:41,71` | core (`pillow`) |
| yaml | `config.py:14` | - | core |
| typer, rich | `cli.py:16-19`, `cli_universal.py:26-28` | - | core |
| torch | - | `scoring.py:83,93`; `demo.py:142`; `encoders/colvlm.py` (`:55,67,251,317,347`); `encoders/nemotron.py:48` | `[vlm]` |
| scipy | - | `stages/merge.py:257,285`; `calibration.py:326` | `[merge]`, `[dev]` |
| qdrant_client | - | `index/qdrant_index.py:58,82,160` | core |
| pypdfium2 | - | `ingest.py:28,78`; `demo.py:51,72` | core |
| reportlab, datasets | - | `corpus.py:85,148,219`, `:327` | `[corpus]`, `[bench]` |
| transformers, colpali_engine | - | `nemotron.py:49`; `colvlm.py:348` | `[vlm]` |
| sentence_transformers | - | `adapters.py:171` | not declared |
| peft, accelerate, huggingface_hub | - | `colvlm.py:173,321,98` | not declared by name |

Import-time footprint (measured, subprocess `sys.modules`): `import optivision` and `import optivision.{scoring,stages,representation,encoders.colvlm,encoders.nemotron,index.qdrant_index,benchmark,demo}` each load exactly numpy, PIL, yaml (no torch, scipy, qdrant_client); `import optivision.cli` adds typer and rich. `import optivision` costs about 0.26 s cumulative, 0.11 s of it numpy.

Vision- or model-specific importers: PIL (10 modules above); torch/colpali/transformers (`encoders/colvlm`, `encoders/nemotron`, `demo`); in gen-2: `stages/prune.py`, `adapters.py` (`PageEncoderAdapter`), `benchmark.py` (`SpatialPruner` in `legacy_variants`), `storage.py` (`doc::pN` id convention, `:196-204`), `representation.py` (`from_page_encodings`, `images`), `cli_universal.py` (`s.name == "spatial"` at `:177,:334`).

## 3. Compatibility surface

### 3.1 `optivision.__all__` (40, `__init__.py:49-90`)

- config: `Config CompressionConfig EncoderConfig IndexConfig IngestConfig PruningConfig SearchConfig`
- pipeline: `OptiVisionRAG IndexReport`; types: `PageRef PageEncoding PrunedPage CompressedPage SearchHit SearchResult`
- representation: `MultiVectorCorpus MultiVectorRepresentation`; compose: `Pipeline CompressedCorpus`; scoring: `maxsim_matrix`
- stages (15): `AdaptiveMerge HierarchicalMerge RandomPruner RedundancyPruner SpatialPruner PCAProjector RandomProjector TruncateProjector DimensionProjector Float32Quantizer Float16Quantizer Int8Quantizer Int4Quantizer Lloyd2Quantizer BinaryQuantizer`
- calibration: `calibrate CalibrationResult`; optimize: `optimize OptimizationResult`; `__version__`

Every submodule loaded by `import optivision` is also an attribute of the package: `calibration compose compression config encoders evaluation index ingest pipeline pruning representation scoring stages types` (checked with `dir()`; `tests/test_wide_vectors.py:28` uses `from optivision import calibration`). `optimize` is the function, not the module. `adapters bench benchmark storage demo viz corpus metrics cli*` are not loaded by `import optivision`.

### 3.2 Submodule paths imported by consumers (exact)

Union of tests, non-research scripts, frozen research scripts, apps (`ast`; per-group pairs/modules/private: tests 140/30/6, scripts 73/21/2, frozen research 31/9/5, apps 10/5/0).

```
adapters         MEASURED UNVERIFIED MODELS supported_models load_adapter PageEncoderAdapter SentenceTransformersAdapter
bench            EncodedCorpus Variant default_variants keep_ratio_sweep run_variant run_benchmark  [_encode_queries_cached]
benchmark        Dataset legacy_variants load_legacy_dataset load_vector_dataset run_matrix save_result
calibration      Candidate calibrate score_space score_candidate select pseudo_queries basic_/recommended_/default_/wide_search_space
                 projection_targets  [_compress _judge _per_candidate_confidence _rank_key]  attr TRANSFORM_CACHE_BYTES (monkeypatched) _cache_put
cli              app
cli_universal    parse_pipeline
compose          Pipeline CompressedCorpus save_compressed load_compressed
compression      Compressor INT8_SCALE decode code_nbytes lloyd2_code_nbytes pack_bits unpack_signs pack2 unpack2 rotation_matrix
                 fit_lloyd2 encode_lloyd2 decode_lloyd2 maxsim_{float,asymmetric,asymmetric_batch,asymmetric_lloyd2,asymmetric_lloyd2_batch}  [_int8_codes]
compression.binary pack_bits unpack_signs
config           Config CompressionConfig PruningConfig available_presets
corpus           CorpusSpec generate_synthetic_corpus load_queries
demo             CompressionResult DemoError DemoPipeline load_page looks_blank on_device pdf_page_count pick_device store_in_qdrant
encoders         SyntheticEncoder get_encoder, submodule nemotron
encoders.colvlm  ColVLMEncoder BACKENDS layer_device_map  [_adapter_base _from_pretrained]
encoders.nemotron NemotronColEmbedEncoder KNOWN   attr _from_pretrained (monkeypatched)
encoders.synthetic SyntheticEncoder
evaluation       measure mean_metrics parse_metric per_query_metrics relevant_from_{baseline,qrels} retention retention_lower_bound split_queries
index.numpy_index NumpyIndex
index.qdrant_index QdrantIndex
ingest           iter_pages
metrics          evaluate ndcg_at_k mrr_at_k recall_at_k rank_correlation rank_correlation_shared
pareto           choose dominates pareto_front
pipeline         OptiVisionRAG
pruning          TokenPruner build_keep_mask codebook_saliency fit_codebook patch_saliency prune_redundant
pruning.saliency patch_saliency
representation   MultiVectorCorpus MultiVectorRepresentation as_corpus
scoring          maxsim_matrix maxsim_pair rank segment_reduce
stages           the 15 classes above + stage_from_dict
stages.base      DocView FIT_SAMPLE_ROWS fit_sample_rows
stages.merge     WIDE_DIM  [_pool _unit _ward_input]
storage          ExactIndex TieredIndex NpzStorage to_legacy_pages
types            PageRef PatchGrid PageEncoding PrunedPage CompressedPage
viz              overlay_mask overlay_saliency
```

Square brackets = private names. Used by **frozen** research code: `calibration._compress`, `calibration._per_candidate_confidence`, `stages.merge._pool`, `_unit`, `_ward_input`; frozen scripts also use `storage.{ExactIndex,TieredIndex}`, `scoring.{maxsim_matrix,rank,segment_reduce}`, `evaluation.*`, `calibration.{recommended_search_space,score_space,score_candidate}`, `compose.{Pipeline,CompressedCorpus,save_compressed}`, the 8 stage classes, and the aligned-array attribute names `protected positions weights importance`.

Documented examples: `PACKAGE.md:13,126`, `README.md:33`, `npm/README.md:94`, `docs/UNIVERSAL.md:68,110` import `MultiVectorCorpus optimize Config OptiVisionRAG Pipeline AdaptiveMerge Int8Quantizer` and call `optivision.adapters.supported_models()`.

### 3.3 Non-import contracts

- Entry points: `optivision = optivision.cli:app` (`pyproject.toml`); `python -m optivision.cli` (VS Code extension, `vscode-extension/src/extension.ts:72-73`); the npm launcher (`npm/bin/optivision.js`) installs the PyPI version equal to `npm/package.json` (`tests/test_release_versions.py`).
- CLI surface used by the extension: `init-config make-corpus index [--append] search --top-k stats explain --out --limit bench --out fetch-vidore --dataset --out --limit`, global `-c/--config`; plus `--version/-V`, `--trust-pickle` (`cli_universal.py:47`).
- Package data `optivision = ["presets/*.yaml"]` and `config.py:189` (`Path(__file__).with_name("presets")`); preset names `synthetic colsmol colpali qdrant`.
- Env: `OPTIVISION_SCORE_DEVICE`, `OPTIVISION_CACHE_BYTES` (library); `OPTIVISION_CONFIG` (`app.py:43`); `OPTIVISION_{PYTHON,EXTRAS,PIP_SPEC,HOME,REINSTALL}` (npm launcher).
- Error and warning text matched by tests: `"non-finite.*document 1"`, `"queries contain non-finite"`, `"no vectors"`, `"page images"`, `"clips components"`, `"calibrating on 20 queries"`, `"sample queries"`, `"bundled presets"`, `"pin a revision"`.

### 3.4 Name shadowing present today

- `__init__.py:24` rebinds `optivision.optimize` from the submodule to the function: `import optivision.optimize as m` yields the function (verified); `from optivision.optimize import OptimizationResult` still works.
- `adapters.py` is a module. Adding `adapters/__init__.py` next to it silently shadows it (package wins; verified, App. A.4): the move must be `git mv adapters.py adapters/__init__.py`.

## 4. Core data types

### 4.1 Gen-1 (`types.py`)

| type | fields (dtype, shape) |
|---|---|
| `PageRef` `:18-29` | `doc_id:str`, `page_no:int` (1-based), `source_path:str` or None, `image_path:str` or None; `page_id = f"{doc_id}::p{page_no}"` |
| `PatchGrid` `:32-53` | `rows`, `cols`, `token_index:int32[rows,cols]` -> row in the embedding matrix; `contiguous()` |
| `PageEncoding` `:56-79` | `ref`, `embeddings:float32[n_tokens,dim]` (L2-normalised by contract, not enforced; `encoders/base.py:51`), `grid`, `image_size=(w,h)`, `text_token_index:int32[k]` (default empty), `meta:dict` |
| `PrunedPage` `:82-101` | `embeddings:float32[n_kept,dim]`, `kept_token_index:int32[n_kept]`, `keep_mask:bool[rows,cols]`, `grid`, `n_tokens_before`, `saliency:float32[rows,cols]` or None, `stats` |
| `CompressedPage` `:104-129` | `codes:uint8[n_vectors, code_bytes]` (docstring says sign bits; holds every method: binary `ceil(d/8)`, int8 `d`, lloyd2 `ceil(d/4)`, none `4d`), `dim`, `n_tokens_before`, `n_tokens_after`, `scale:float32[n]` or None, `stats`; `nbytes = codes + scale`; `raw_nbytes = n_tokens_before*dim*4` |

### 4.2 Gen-2

| type | contract |
|---|---|
| `MultiVectorRepresentation` `representation.py:106-150` | `vectors:[n,d]` any floating dtype (float16/32/64 kept as is; torch bfloat16 widened to float32; integers raise `TypeError`, `:51-69`); `doc_id`, `metadata`; aligned arrays `protected:bool[n]`, `positions:float32[n,2]` (row,col in a 0..1 page frame, NaN off-grid), `weights:float32[n]` (rows merged into each), `importance:float32[n]` (`:45,72-100`); rows need not be unit norm (`norm_stats`) |
| `MultiVectorCorpus` `:170-` | `vectors:[N,d]`, `offsets:int64[n_docs+1]` (starts 0, non-decreasing, ends N; empty documents allowed, `:200-209`), `ids:list[str]`, `metadata:list[dict]`, the four aligned arrays over N rows, `images:list` or None (never serialised), `attrs:dict`. `__getitem__` returns views; slices are rejected; `select` copies; `with_vectors` keeps row count; `decode_rows(lo,hi)` -> float32 (the `VectorStore` read interface) |
| `CompressedCorpus` `compose.py:48-118` | `codes:uint8[N', code_bytes(d')]`, `offsets:int64`, `dim` (after projection), `quantizer` (stateful object), `ids`, `stats` (`vectors_before dim_before float32_bytes_before native_bytes_before native_dtype_before`, `:219-225`), `pipeline:dict` or None. Drops `metadata attrs protected positions weights importance` |
| `DocView`, `Reduction` `stages/base.py:88-118` | what a token reducer sees (non-protected rows as float32, `positions importance weights metadata image`) and returns (`vectors`, `labels:int64[n_free]` with -1 = dropped, optional `importance`) |

### 4.3 Serialisation

| artifact | code | content | notes |
|---|---|---|---|
| corpus `.npz` | `representation.py:496-539` | `vectors offsets header(uint8 JSON)` + optional aligned arrays; header `{format:"optivision-multivector", version:1, ids, metadata, attrs}` | `allow_pickle=False`; version > 1 rejected; `images` dropped; `load(mmap=True)` is a no-op for `.npz` (verified: result is a plain `ndarray`) |
| compressed `.npz` | `compose.py:257-297` | `codes offsets header`; header `{format:"optivision-compressed", version:1, dim, ids, stats, pipeline, quantizer}` | `version` not checked on load; quantizers with `overhead_bytes() > 0` raise `NotImplementedError` (`:264`); only stage names + constructor args stored, so `Pipeline.from_dict` yields unfitted stages and PCA components, merge means, per-dimension ranges, Lloyd-2 mu/sigma are lost |
| numpy index dir | `index/numpy_index.py:61-108` | `codes.npy offsets.npy meta.json{backend,dim,method,n_pages,refs[PageRef.__dict__],page_stats,codec{mu,sigma,seed}}` | no version field; Lloyd-2 rotation regenerated from `(dim, seed)` via `np.linalg.qr` |
| encode cache | `bench.py:225-271,349-378` | `emb_i grid_i text_i img_i` (512x512 uint8 gray), `meta` (JSON str in a 0-d object array); query cache `q_i texts encode_ms` | needs `allow_pickle=True` (`:255,:361`); `cli_universal` refuses it without `--trust-pickle` |
| float cache | `pipeline.py:250-268` | `vectors offsets page_ids(object)` | `allow_pickle=True` |
| manifest | `pipeline.py:201-206` | `{config: Config.to_dict(), index: stats}` read by `optivision stats` | |
| Qdrant | `qdrant_index.py:39,168-181` | point id `uuid5(_NAMESPACE, page_id)`; payload `page_id doc_id page_no source_path image_path n_tokens_before n_tokens_after nbytes raw_nbytes` | namespace UUID is persistent identity |

No class path or module path is stored anywhere (only registry names and format strings), and no custom class is pickled: moving modules cannot break an existing file by itself.

Byte accounting: `CompressedCorpus.nbytes = codes.nbytes + quantizer.overhead_bytes()` (`compose.py:93-96`). Not counted: offsets, ids, PCA components (`d_in*k*4` bytes), `AdaptiveMerge.mean`. The selection key is this number (`calibration._rank_key`, `:213-218`).

### 4.4 Where vision-specific assumptions live

- `types.py` entire (page grid, images, saliency); `PageRef.page_no/image_path`.
- `representation.py`: `images` (`:196,221-223`), `from_page_encodings` (`:345-393`: text/thumbnail tokens -> `protected`, grid cell centres -> `positions`, `grid_rows grid_cols image_size page_ref` into metadata), `from_legacy_cache` (`:395-412`).
- `stages/base.py`: `DocView.image` (`:103`), position and importance bookkeeping in `TokenReducer.transform` (`:121-214`); protected rows are moved last in each document (`:176-179`), so row order is not preserved.
- `stages/prune.py:65-100`: `SpatialPruner` needs `doc.image`, `metadata["grid_rows"/"grid_cols"]`, `positions`; raises otherwise.
- `pruning/*`, `encoders/*`, `ingest.py`, `viz.py`, `demo.py`, `corpus.py`, `adapters.PageEncoderAdapter`: vision/page specific by construction.
- Generic but vision-shaped: `importance` seeding in `AdaptiveMerge` (`merge.py:163-167`), `positions` mean pooling (`base.py:217-230`).

## 5. Stage system

### 5.1 Contracts (`stages/base.py`)

- `Stage`: `fit(corpus, queries=None) -> Stage` (default no-op), `fitted` (default True), abstract `params()`, `to_dict() = {"stage": name, **params()}` (`:49-68`). `kind`/`name` are class variables; `Pipeline` dispatches by `isinstance`, not `kind`.
- `TokenReducer.reduce(DocView) -> Reduction` + concrete `transform` (`:121-214`); `DimensionReducer.project/output_dim` + `transform`, `transform_queries` (`:243-260`); `Quantizer.encode/decode/code_bytes/overhead_bytes/bits_per_dim` (`:266-287`). Queries pass only through `DimensionReducer`s (`compose.py:177-183`); token reducers and the quantizer are document-side.
- Registry: global `STAGES` (`:30`), `@register` raises on a duplicate name unless it is the same class (`:33-38`), `stage_from_dict` = `STAGES[name](**params)` (`:41-46`). Registration happens at import of the stage module, guaranteed today only because `stages/__init__.py` imports all four.
- `Pipeline` (`compose.py:121-254`): at most one quantizer and it is last (`:126-130`); `fit` fits each stage on the output of the previous ones and then the quantizer (`:154-168`); `compress(fit=None)` fits only if unfitted, `fit=True` refits (`:185-194`); `encode` quantizes in 65,536-row blocks with a finiteness check (`:196-226`). `fit` discards the transformed corpus it computes, so `compress` and `calibration._compress` (`:128-129`) run every token/dimension stage twice on first use.
- Decode: `CompressedCorpus.decode_rows` = `quantizer.decode(codes[lo:hi], dim)` -> float32 in the post-projection space (`compose.py:86-87`); legacy decode by method name is `compression.decode` (`compression/__init__.py:151-170`).
- Fit sampling: `fit_sample_rows(dim, itemsize)` caps at 200,000 rows and 256 MiB (`base.py:78-85`); unchanged for d <= 160 (`tests/test_wide_vectors.py:20-25`). Four independent sampling implementations exist: `quantize._fit_sample` (rng 0), `project._sample_rows` (rng `seed`), `AdaptiveMerge.fit` (rng 0, `merge.py:146`), `fit_lloyd2` (`lloyd2.py:141-146`).

### 5.2 Concrete stages

"Basis" = behaviour if every document and query vector is multiplied by one orthogonal matrix Q. "Realisation" = the specific output; "distribution" = its quality statistics over the draw of any internal random matrix.

| registry name | kind | fitted state | RNG / seed | basis under Q (justification) |
|---|---|---|---|---|
| `spatial` | tokens | none | none | independent of vector values: rows chosen from pixels and positions (`prune.py:65-100`); vision only |
| `redundancy` | tokens | none | none; visit order `np.argsort(-importance)` is an unstable sort (`prune.py:126`) | invariant: only Gram products `emb @ emb.T` (`pruning/redundancy.py:57-64`) and mean pooling (equivariant) |
| `adaptive_merge` | tokens | `mean f32[d]` iff `center="mean"` (`merge.py:135-148`) | `default_rng(0)` subsample when rows exceed the cap (`:146`); argmax/argmin ties go to the first index | invariant: cosines of `_unit(v)` or `_unit(v - mean)` (`:174-179,183,193`); `_pool` (`:48-60`) and the mean are equivariant |
| `hierarchical_merge` | tokens | none | none (SciPy Ward) | invariant: Ward on `_unit(v)`; for d >= 512 Gram-derived distances (`:277-289`) |
| `random_prune` | tokens | none | `default_rng((seed, doc.index))` (`:310`): stream keyed by document position, so reordering or subsetting documents changes the draw | independent of values |
| `pca` | dim | `components f32[d_in,k]`, `mean`, `explained` | row subsample `default_rng(seed)` (`project.py:35-39`); `eigh` | invariant up to per-component sign: eigenvectors of `Q^T M Q` are `Q^T V`, so `(xQ)(Q^T V) = xV` (`:80-97`); signs cancel in inner products |
| `random_projection` | dim | `matrix f32[d,k]` | `default_rng(seed).standard_normal((d,k))` + QR (`:139-141`) | realisation changes (matrix is drawn in input coordinates); distribution does not |
| `truncate` | dim | none | none | basis-dependent by construction: `[:, :k]` (`:207`) |
| `project` | dim | inner projector | forwards | as the chosen method |
| `float32` | quant | none | - | lossless |
| `float16` | quant | none | - | per-coordinate IEEE rounding: relative error bound holds in any basis, bit patterns change |
| `int8` | quant | `ranges f32[d]` (`per_dimension`), `mean` (centre) | `_fit_sample` rng 0 | `fixed`: absolute step 0.5/127 and clipping at +-0.5 (`compression/__init__.py:139`, warning `quantize.py:200-206`); `per_vector`: step = max_j abs(v_j)/127 (`:212`), an infinity-norm functional; `per_dimension`: per-axis 99.99th percentile (`:189-190`). All basis-dependent |
| `int4` | quant | `mean` iff centre | rng 0 | per-vector max-abs scale, 15 levels (`:281-286`): basis-dependent |
| `binary` | quant | `mean` iff centre | rng 0 | `v > 0` (`compression/binary.py:36`) keeps the orthant in the stored basis; decode is +-1 (`:54`): strongly basis-dependent |
| `lloyd2` | quant | `Lloyd2Codec(mu, sigma, seed, dim)`; rotation not stored | `fit_lloyd2` rng(seed) 50,000-row sample; `R = qr(default_rng(seed).standard_normal((d,d)))`, `lru_cache(8)` (`lloyd2.py:110-124`) | explicit random rotation, then fixed Gaussian thresholds times one global sigma (`:62-63,147,189-190`): realisation changes, quality distribution is basis-free; `R` depends on the LAPACK QR |

All deterministic given data, seeds and the numeric libraries; none uses a global RNG (`np.random.seed` is never called).

### 5.3 Probe (synthetic; Appendix A.2)

Anisotropic unit vectors (d=64, lognormal per-axis scales), one orthogonal Q. This checks code behaviour only; it says nothing about retrieval quality.

| stage | observation |
|---|---|
| `adaptive_merge` (plain, centred), `hierarchical_merge`, `redundancy`, `random_prune` | identical row counts; `max abs(out(CQ) - out(C)Q)` between 1.5e-8 and 5.2e-8 |
| `pca` / `random_projection` / `truncate` | max score change over the span of float scores: 4.1e-7 / 1.2 / 0.28 |
| mean `1-cos(decode(encode(x)), x)`, original vs rotated basis | float16 1.8e-8 / 2.1e-8; int8 fixed 2.8e-3 / 4.1e-5; int8 per_vector 4.8e-5 / 1.8e-5; int8 per_dimension 2.2e-5 / 2.2e-5; int4 1.4e-2 / 5.7e-3; int4 centred 1.1e-2 / 4.5e-3; binary 0.405 / 0.200; binary centred 0.468 / 0.277; lloyd2 4.80e-2 / 4.79e-2 |

`int8 per_dimension` shows no change in error here because per-axis ranges absorb the anisotropy; its dependence is through axis-aligned tails and rounding, not captured by this probe.

Consequence for the design: for the token and PCA stages, basis independence is structural (they use inner products only). For the codecs it is an assumption about the encoder's coordinate basis, which this audit did not test on real encoders. A rotation stage (random orthogonal or Hadamard) ahead of the codec would be the basis-removing tool; none exists as a stage (`lloyd2` embeds one privately).

## 6. Scoring and retrieval

### 6.1 The seam today

- Read interface: `VectorStore` protocol (`scoring.py:26-31`): `offsets`, `__len__`, `decode_rows(lo,hi) -> float32`. Implemented by `MultiVectorCorpus` (`representation.py:284`) and `CompressedCorpus` (`compose.py:86`). `NumpyIndex` is not a `VectorStore`.
- Score interface: `maxsim_matrix(queries, docs, max_block_bytes=256 MiB, query_transform=None) -> float32[n_q,n_d]` (`scoring.py:99-168`); exact, page-aligned blocks, empty document -> -inf, empty query -> 0. `rank` is a stable argsort (`:171-186`).
- Index interface (gen-2): `ExactIndex.score/search/rescore` (`storage.py:37-102`; `search` returns row indices and scores), `TieredIndex` (hot shortlist + cold rescore; non-shortlisted documents are shifted below the shortlist floor, `:116-137`).
- Index interface (gen-1): `BaseIndex.add(CompressedPage[])/search(query ndarray, top_k) -> SearchHit[]` one query at a time (`index/base.py`). Bridge: `to_legacy_pages` accepts only float32, binary without centring, and int8 `fixed`; others raise (`storage.py:177-205`); it approximates `n_tokens_before` by the corpus mean (`:193-194,202`). No reverse bridge.

### 6.2 MaxSim implementations

| where | scope | empty document | device |
|---|---|---|---|
| `scoring.maxsim_matrix` `:99` | all queries x all docs, blocked | -inf | optional torch |
| `scoring.maxsim_pair` `:189` | one pair (reference) | -inf | none |
| `storage.ExactIndex.rescore` `:58-102` (inline `(qv[r] @ block.T).max(axis=1)`, `:99`) | per-query candidate sets, document-batched | -inf (empty query 0) | none (ignores the env var) |
| `index/numpy_index._segment_maxsim` `:157-173` and its own block loop `:196-221` | one query x all pages | -inf | none |
| `compression/binary.py`: `maxsim_float :97`, `maxsim_asymmetric :61`, `_batch :73`, `maxsim_hamming :105` | per page / batch | 0.0 | none |
| `compression/lloyd2.py`: `maxsim_asymmetric_lloyd2 :203`, `_batch :214` | per page / batch | 0.0 | none |
| Qdrant `MAX_SIM` (`qdrant_index.py:103-105`) | server | n/a | server; `Distance.COSINE` (`:101`) |

`pipeline._rerank` uses `maxsim_float` (`:282`). Raw inner product (`maxsim_matrix`) and cosine (Qdrant) agree only for unit-norm or constant-norm vectors. `NumpyIndex.search` breaks ties with `argpartition`+`argsort` (`:229-230`), not by index.

### 6.3 Device handling

- Only mechanism: `os.environ["OPTIVISION_SCORE_DEVICE"]`, read on every `maxsim_matrix` call (`scoring.py:124`). When set, `_to_device` sets `torch.backends.cuda.matmul.allow_tf32 = False` and `cudnn.allow_tf32 = False` process-wide and never restores them (`:85-86`), uploads each query chunk once (`:87`), uploads every document block (`:96`) and downloads the full similarity block to numpy (`.cpu().numpy()`, `:96`); max/sum stay on the CPU. No fallback if torch or the device is missing (ImportError/RuntimeError at first chunk). `ExactIndex.rescore` and `NumpyIndex` never use it.
- Encoders move tensors explicitly (`colvlm.py:541-544`, `nemotron.py:99`); `demo.on_device` moves a model and restores it (`demo.py:117-137`).

### 6.4 Dtype conversions

| where | conversion |
|---|---|
| `representation.py:59-63` | torch tensor -> `.detach().cpu()`; bfloat16 -> float32; other floating dtypes kept |
| `representation.py:290,308-311` | `decode_rows` casts to float32 (copy for float16/64); `from_documents` promotes to the common dtype, copying every document |
| `stages/base.py:152,161,176` | token reducers compute in float32 and cast back to the corpus dtype (merged float16 vectors are re-rounded) |
| `stages/base.py:257` | `DimensionReducer.transform` casts the whole corpus to float32 at once (no blocks) and returns float32 even for float16 input |
| `compose.py:215` | quantizer input blocks cast to float32 |
| `stages/quantize.py:324` | `Lloyd2Quantizer.fit` casts the whole corpus to float32 before sampling (the byte cap in `fit_sample_rows` is not applied) |
| `project.py:82,102` | PCA fits in float64 on a float32 sample, projects in float32 |
| `scoring.py:146-158` | queries and blocks forced to float32; the GPU path computes in float32 with TF32 off |
| `qdrant_index.py:157,197` | `.tolist()` of decoded float32 (Python float lists) |

## 7. Selection

### 7.1 Algorithm (`calibration.py:413-531`)

1. Validate: target in (0,1], `safety` in {point, lower_ci}, `reference` auto -> labels if qrels else baseline; at least 4 queries.
2. `cal_idx, hold_idx = split_queries(n, 0.5, seed)` (`evaluation.py:138-145`: `default_rng(seed).permutation`, `cut = round(n*fraction)`).
3. `base_scores = maxsim_matrix(queries, corpus)` over all queries; relevance from qrels (ids not found are silently dropped, `evaluation.py:45`) or from the baseline's own top-`depth` documents (`:49-52`); per-query metric, NaN when a query has no relevant document.
4. Warn if fewer than 100 calibration queries have a nonzero baseline metric (`:66-69,468-476`); warn if SciPy is missing and the default space is used (`:478-484`).
5. For each family in order, each step: `score_candidate` compresses the corpus (token/dimension prefix fitted once per distinct prefix and cached, LRU by bytes, `:96-132`) and scores all queries (`:135-143`); `_judge` computes retention on calibration queries (`evaluation.retention`: ratio of means, paired bootstrap, `rng = default_rng(seed)`, `idx = rng.integers(0, n, (n_boot, n))`) and, for `lower_ci`, `retention_lower_bound` (`evaluation.py:109-135`): percentile `1-confidence` when confidence <= 0.99, else `point - z*std(ratios)` with `z = NormalDist().inv_cdf(confidence)`.
6. Feasible = bound >= target + margin and (no byte cap or bytes <= cap) (`:184-210`). With `assume_monotone` the first infeasible step ends its family (`:499-502`).
7. Winner = `min` over feasible of `(bytes_per_doc, vectors_per_doc, position)` (`_rank_key`, `:213-218`); wall time is deliberately not a key.
8. Only the winner is evaluated on the held-out queries (`_holdout`, `:247-260`). `select()` repeats the judging, choice and hold-out steps (5 without the scoring, 6-8) from stored `ScoredCandidate`s.

Defaults differ between entry points: `calibrate` `safety="lower_ci"`, `confidence=0.999`; `select` `safety="point"`, `confidence=0.975` (`:231-233` vs `:424-431`); `select` has no early stop; `multiplicity="bonferroni"` divides `1-confidence` by the candidate count (`:221-226`).

### 7.2 Candidate spaces (`calibration.py:305-401`)

| function | size | definition |
|---|---|---|
| `basic_search_space` | 30 | 3 codecs (`float16 int8(fixed) binary`) x 10 `AdaptiveMerge` radii (`None .95 .9 .85 .8 .75 .7 .65 .6 .5`) |
| `recommended_search_space` = `default_search_space` | **40** with SciPy, 22 without | `float16` (1) + for each of `int8(per_vector)`, `int4(centred)`, `binary`: `ward+X` (6: `HierarchicalMerge` ratio `None .5 .33 .25 .15 .1`) + `adaptive_centred+X` (7: radius `None .9 .8 .7 .6 .5 .4`, `center="mean"`) = 1 + 3x13; 37 distinct (the `None` step of both chains is the bare codec); 7 chains |
| `projection_search_space` | 7 (d=96), 14 (128), 35 (1,024), **42** (2,560 or 4,096) | widths `{d/2,d/4,d/8,d/16,128}` kept if `64 <= w < d` (`projection_targets`); 3 `pca+X` families x (1 + #widths) + 4 `ward{.33,.25}+pca+{int8,int4}` families x (1 + #widths) |
| `wide_search_space` | 47 (96), 54 (128), 75 (1,024), **82** (2,560 or 4,096) | default + projection; opt-in, labelled experimental |

Counted by calling the functions (SciPy present). Families are ordered least to most aggressive; the space depends on `dim` and on whether SciPy imports (`_scipy_available`, `:324-329`).

### 7.3 Randomness

One `seed` (default 0) drives the query split and the bootstrap; the same bootstrap indices are reused for every candidate (common random numbers) and for `retention` and `retention_lower_bound` (same seed and shape). Candidate generation has no randomness. Stage RNGs are fixed constants (section 5.2). `pseudo_queries` (`:537-565`, experimental) uses its own `seed`.

### 7.4 Abstention

- No feasible candidate on the calibration queries: `selected=None`, `holdout=None`, `met_target_on_holdout=None`; `optimize()` returns the corpus compressed with an empty `Pipeline` (float32, ratio 1.0) and `target_met=None` (`optimize.py:89-91`); the CLI prints "nothing met the target".
- A selected candidate that misses the target on the held-out queries is still applied; `target_met=False` is the only signal.
- Candidates skipped by the early stop do not appear in `CalibrationResult.candidates`.

### 7.5 Gap to Method C and the integration seam

Frozen primitives (`scripts/research/e1_stats.py`): `to_y` `:43` (`Y = (c - T b + T)/(1+T)`, `m0 = T/(1+T)`), `betting_pvalues` `:47-68` (aGRAPA bet, cap `BET_CAP = 0.9`, observation-ordered, float64), `fixed_sequence_by_family` `:118-129`, `smallest` `:142-146` with key `(bytes, vectors, position)` (`:98`); orchestration `x_common.method_c` `:940-942` at level `delta / n_chains`; it uses all n queries with no hold-out (`x_common.py:668`). Inputs: `pq[K,n]`, `base[n]`, `families[K]`, `bytes_per_doc[K]`, `vectors_per_doc[K]` - all obtainable from `score_space` (`ScoredCandidate.scores`, `.report`) plus `per_query_metrics`. Missing in `src`: the two testing functions, the all-queries (no-split) mode and its observation-order contract, a result type, and certificate emission. `e1_stats.lower_bound_all` is a vectorised copy of `evaluation.retention_lower_bound` (its docstring, `:30-31`); a core port should add parity tests against the frozen originals and leave both untouched.

## 8. Duplicated logic, global state, hidden configuration, legacy paths

### 8.1 Duplication

| what | copies |
|---|---|
| MaxSim | section 6.2 (3 production paths, 1 reference, 6 legacy helpers, 1 test-only `_rescore_per_query`) |
| IR metrics | `metrics.py` (id lists, empty-relevance -> 0.0) vs `evaluation.per_query_metrics` (score matrices, -> NaN); same formulas |
| bootstrap | `evaluation.retention` and `retention_lower_bound` repeat the resampling (`:89-135`); `e1_stats.lower_bound_stats` is a third, frozen copy |
| codecs | `Compressor`/`compression.decode` (method string) vs `Quantizer` classes; shared primitives `pack_bits unpack_signs _int8_codes decode_int8 encode_lloyd2 decode_lloyd2` |
| pruners | `TokenPruner` vs `SpatialPruner`+`RedundancyPruner` (row-for-row equal by `tests/test_stages.py:46-66`); `TokenPruner` also has codebook saliency, `keep_text_tokens`, `enabled`, which have no stage |
| persistence | five formats (section 4.3), three needing pickle or JSON-in-object-array |
| unit-normalise | `merge._unit`, `pruning/codebook._normalise`, `pruning/redundancy._normalise`, `encoders/base.l2_normalise` |
| row sampling | four implementations (section 5.1) |
| page blocking | `scoring._page_blocks` and `numpy_index.py:204-221` |
| top-k | `scoring.rank` (stable) vs `argpartition` in `NumpyIndex.search` and `bench.run_variant:325-326` |

### 8.2 Global state and side effects

- `STAGES` registry (`base.py:30`); `adapters.MODELS` (`:86`), `colvlm.BACKENDS` (`:45`), `nemotron.KNOWN` (`:34`); `lru_cache(8)` rotation matrices, shared and returned un-copied (`lloyd2.py:110-124`).
- `calibration.TRANSFORM_CACHE_BYTES` read once at import from `OPTIVISION_CACHE_BYTES` (`:98`); tests patch the module attribute.
- `ingest.py:13` sets `Image.MAX_IMAGE_PIXELS = 300_000_000` at import, so `import optivision` changes PIL's decompression-bomb limit process-wide.
- `scoring._to_device` flips TF32 flags (section 6.3).
- `calibration._compress` rewrites the caller's `pipe.stages` in place with cached fitted stage objects shared by every pipeline with the same prefix (`:131`).
- `Int8Quantizer(scale="fixed")` warns on every block that clips (`quantize.py:200-206`, `stacklevel=3`).

### 8.3 Silent or surprising behaviour

- `OptiVisionRAG` cannot use `method="lloyd2"` (`pipeline.py:84` builds `Compressor` without a codec; `compression/__init__.py:108-113` raises) and silently falls back to pixel saliency when `pruning.saliency="codebook"` (no codebook is ever fitted; `pruning/__init__.py:55-69`).
- Dead options: `Config.seed` (`config.py:112`) is never read; `CompressionConfig.asymmetric` (`config.py:71`) is never read; `keep_norm` (`config.py:70`) adds `scale` to `CompressedPage.nbytes` but `NumpyIndex.add` neither stores nor uses it.
- `load_adapter` has no branch for `loader == "nemotron"` and falls through to `SentenceTransformersAdapter` (`adapters.py:200-205`); `scripts/encode_vectors.py:109-150` bypasses it.
- `SyntheticEncoder` page vectors depend on encode order (stateful `self._rng`, `synthetic.py:128`).
- Stale text: `stages/project.py:21-22` says only 128-d models were measured (pinned file, cannot be corrected); `types.CompressedPage.codes` docstring says sign bits only; `docs/ARCHITECTURE.md` describes gen-1 only.

### 8.4 Legacy paths

`OptiVisionRAG` (`pipeline.py`), `bench`, `demo`, `NumpyIndex`, `QdrantIndex`, `Config`/presets, the 8 gen-1 CLI commands, `corpus`, `viz`, `ingest`, `metrics`. They are public (`__all__`, CLI, extension, Streamlit/Gradio apps) and tested: 9 test files that are gen-1 only (`compression demo index pipeline pruning qdrant presets colvlm_loading nemotron_encoder`) hold 143 of the 351 test functions, and `test_stages`, `test_representation`, `test_storage` and `test_scoring` also exercise gen-1 equivalence.

## 9. Recommendation

### 9.1 Layer rules (enforce with an `ast` test; Appendix A.1 is the starting point)

```
core        numpy only; lazy scipy, torch.  Never PIL, yaml, typer, rich, qdrant_client, adapters, backends, ext, legacy
adapters    core; lazy sentence_transformers
backends    core; lazy qdrant_client
ext.vision  core, adapters; PIL, pypdfium2; lazy torch, transformers, colpali_engine, peft
legacy      anything below; yaml, typer, rich
optivision/__init__   PEP 562 lazy re-exports; __version__ stays a string literal
```

Target layout (P = pinned file; where it stays physically, `core.*` is an alias or facade of it):

```
optivision/
  __init__.py          lazy facade, 40-name __all__, __version__ literal
  core/                numpy-only import closure
    vectors.py         = representation.py (P)
    stages/            = stages/{base,merge,project,quantize}.py (P) + RedundancyPruner + new stages (e.g. rotation)
    pipeline.py        = compose.py (P)
    similarity.py      = scoring.py (P)
    retrieve.py        ExactIndex, TieredIndex (from storage.py)
    evaluate.py        = evaluation.py (P) + id-based metrics + Dataset/run_matrix (from benchmark.py)
    selection/         = calibration.py (P, bootstrap rule) + method_c.py (new) + pareto + optimize
    artifact.py        NpzStorage/StorageBackend + new fitted-state format + certificate (new)
  adapters/            EncoderAdapter, MODELS registry, SentenceTransformersAdapter, Nemotron adapter (new)
  backends/            base, numpy (NumpyIndex), qdrant
  ext/vision/          page types, pruning/*, SpatialPruner, encoders/*, PageEncoderAdapter, ingest, viz, demo, corpus
  legacy/              Config + presets, OptiVisionRAG, bench, Compressor, TokenPruner, CompressedPage, cli
```

Every old import path stays resolvable (section 3.2); moved modules become `sys.modules` aliases of the new module object.

### 9.2 What the pin allows

- Zero-edit: keep the 11 pinned files where they are; build `core.*` as `sys.modules` aliases or thin facades over them, and add new core code (Method C, artifact/certificate with fitted state, rotation stage) in new files that import them. New stages register through the pinned `stages.base.register`.
- Edits to unpinned files that make the pinned set's import closure numpy-only: (a) import yaml inside `Config.load/dump` (`config.py:14`); (b) lazy vision imports in `stages/__init__.py` and `stages/prune.py` (give `RedundancyPruner` a pure-numpy `prune_redundant` import); (c) PEP 562 in `optivision/__init__.py`. Expected result: `import optivision.scoring` loads numpy only (verify with Appendix A.3).
- Needs a decision outside engineering: physically moving the 11 files into `core/`. Any byte change breaks `tests/test_x_harness.py:249-253` and `x_common.validate_inputs`; re-registering the blobs means amending frozen research code.

### 9.3 Per-module plan (shim = old path keeps resolving to the same module object)

| module | target | mechanism | main risk |
|---|---|---|---|
| `representation.py` P | core.vectors | stays; `core/vectors.py` alias | vision/legacy hooks (`from_page_encodings`, `images`, `from_legacy_cache`) are inside a pinned class: leave them, add `ext.vision` constructors as the supported path |
| `stages/{base,merge,project}.py` P | core.stages | stays | `_pool _unit _ward_input` used by frozen code; `STAGES` global |
| `stages/quantize.py`, `compression/{__init__,binary}.py` P | core.stages / core.codecs | stay | `compression/__init__.py` mixes codec primitives with legacy `Compressor` and imports `config`, `types`: allow-list in the layer test |
| `compose.py` P | core.pipeline | stays | `save_compressed` cannot hold fitted state: write a new artifact format beside it, keep `optivision-compressed v1` readable |
| `scoring.py`, `evaluation.py` P | core.similarity, core.evaluate | stay | env var and TF32 flag; empty-document semantics |
| `calibration.py` P | core.select (rule A) | stays; Method C in a new sibling module | private names; `calibration.select` vs a package named `select` (also a stdlib name): prefer `selection` |
| `optimize.py`, `pareto.py` | core.select | alias move | `optivision.optimize` is the function (keep) |
| `storage.py` | split: core.retrieve (`ExactIndex TieredIndex`), core.artifact (`StorageBackend NpzStorage`), legacy bridge (`to_legacy_pages`) | `storage.py` stays as alias facade of all 5 names | unreleased batched `rescore`; id convention `doc::pN` |
| `benchmark.py` | core.evaluate (`Dataset run_matrix save_result load_vector_dataset`) + legacy (`load_legacy_dataset legacy_variants`) | split + facade | pulls the `stages` package (PIL) until 9.2(b) |
| `metrics.py` | core.evaluate (id-based) | alias move | do not merge with `evaluation` (0.0 vs NaN) |
| `adapters.py` | `adapters/` package | `git mv` to `adapters/__init__.py`, never both | relative import `.encoders.colvlm`; `MODELS` pinned by a test; `PageEncoderAdapter` must stay importable from `optivision.adapters` (lazy attribute) |
| `encoders/*` | adapters (generic) + ext.vision (page encoders); add a Nemotron `EncoderAdapter` and a `load_adapter` branch | old paths as `sys.modules` aliases | `tests/test_nemotron_encoder.py` patches `nemotron._from_pretrained`: a `from x import *` shim would not intercept |
| `types.py` | ext.vision (`PageRef PatchGrid PageEncoding PrunedPage`), legacy (`CompressedPage SearchHit SearchResult`) | facade re-exporting the same class objects | `PageRef.__dict__` keys are on-disk (`numpy_index.py:99`, `bench.py:243`); `compression/__init__.py:8` imports it |
| `pruning/*` | ext.vision; `redundancy.prune_redundant` to core | alias moves | pinned import chain through `pruning.*`; codebook fallback (8.3) |
| `index/*` | backends | alias moves; keep `optivision.index.numpy_index` and `.qdrant_index` | Qdrant COSINE vs raw inner product; point-id namespace |
| `config.py`, `presets/` | legacy | stay | `config.py:189` and package-data glob; unknown-key rejection means YAML compatibility is by key name |
| `pipeline bench demo viz ingest corpus cli` | legacy / ext.vision | stay | `ingest.py:13` global; `optivision.cli:app` and `-m optivision.cli` fixed |
| `cli_universal.py` | application layer (needs typer, rich) | stays | `"spatial"` name checks (`:177,:334`) |
| `__init__.py` | facade | lazy `__getattr__` + `__all__` | `__version__` literal (setuptools `attr`, `tests/test_release_versions.py`); registration side effects; `Image.MAX_IMAGE_PIXELS` assignment moves with `ingest` |

### 9.4 Sequence

1. Guardrails without source changes: a public-surface test over the 201 pairs and 11 private names; golden files written by v0.3.0 (corpus `.npz`, compressed `.npz`, numpy index dir, encode cache) read by new code; the layer test with its allow-list; keep `test_x_harness.py` green.
2. Additive core modules (Method C, artifact/certificate, rotation stage); no pinned edit.
3. The three unpinned edits of 9.2.
4. Alias moves of unpinned modules (storage split, types split, `adapters/`, `index` -> `backends`, `pruning`/`encoders` -> `ext.vision`).
5. Pin decision.

### 9.5 Risk register

| risk | where |
|---|---|
| pinned bytes and S0 spec hash | section 0.2; `tests/test_x_harness.py:249-253` |
| `from x import *` shims drop private names and break monkeypatching (`TRANSFORM_CACHE_BYTES`, `nemotron._from_pretrained`, `OPTIVISION_SCORE_DEVICE`); use module aliases | 3.3 |
| duplicate registration if one file is imported under two names (`register` raises) | `base.py:35` |
| lazy stage registry: `stage_from_dict({"stage":"spatial"})` fails in a fresh process unless the vision stage module was imported | `base.py:44` |
| `adapters` file -> package shadowing | 3.4 |
| class identity: `__all__` and `isinstance` need the same objects, not copies | types, stages |
| file-format and key stability | 4.3 |
| SciPy presence changes the default space (40 vs 22) and thus results | 7.2 |
| platform numerics: BLAS thread/kernel (`tests/test_storage.py:161-166` accepts 1e-6), unstable `argsort` ties (`prune.py:126`, `pruning/__init__.py:93`), LAPACK QR/`eigh` signs | 5.2 |
| packaging: new subpackages need `__init__.py`; presets path; entry points; npm version | 3.3 |
| memory: `DimensionReducer.transform` and `Lloyd2Quantizer.fit` copy whole corpora; `QdrantIndex` builds Python lists | 6.4 |

## 10. Must-not-break invariants

- 11 pinned blobs (section 0.2) and `S0_SPEC_SHA256`: `Pipeline.to_dict()` of the 40 default candidates for d=128 and d=96, including family names, order, parameter names and defaults of `AdaptiveMerge HierarchicalMerge Int8Quantizer Int4Quantizer BinaryQuantizer Float16Quantizer`.
- Registry names: `adaptive_merge binary float16 float32 hierarchical_merge int4 int8 lloyd2 pca project random_projection random_prune redundancy spatial truncate`; `Pipeline.to_dict`/`from_dict` round-trip (`tests/test_stages.py:157-162`); CLI shorthand parser.
- Formats: `optivision-multivector` v1, `optivision-compressed` v1, numpy-index directory, encode/query/float caches, `manifest.json`, Qdrant payload and point ids, `Config` YAML keys and presets.
- Numerics under test: new stages reproduce the legacy pruners row for row and codecs code for code (`tests/test_stages.py:46-88`); `maxsim_matrix` equals brute force and is block-budget invariant to 1e-6 (`tests/test_scoring.py:29-50`); ties break by index; `segment_reduce` empty-segment handling; `ExactIndex.score == maxsim_matrix`, batched `rescore` equals the per-query reference (bitwise for most shapes, 1e-6 for very short queries), `TieredIndex` shift semantics (`tests/test_storage.py`); float16 corpora scored in float32; PCA at full rank preserves scores; centring does not change rankings; Ward fast path equals SciPy for d >= 512; torch scoring within 1e-4 of numpy; no `RuntimeWarning` on degenerate inputs; zero vectors encode to exact zeros; NaN/inf refused with `document N` in the message.
- Selection: `_rank_key` tie-break `(bytes, vectors, position)`; `calibrate` defaults (`lower_ci`, 0.999, 1,000 bootstrap resamples, split 0.5, seed 0, `assume_monotone=True`, minimum 100 informative queries); lower bound switches from percentile to normal approximation above 0.99; `split_queries` and the bootstrap draw order (relied on by frozen code); stricter rules never choose larger savings; cached transforms give identical scores.
- Legacy semantics: empty document scores 0.0 in `compression.maxsim_*` but -inf in the new scorers; `rank_correlation` vs `rank_correlation_shared`; `Compressor` byte counts (binary exactly 32x, int8 4x, lloyd2 16x).
- Registry honesty: exactly seven measured model ids in `adapters.MODELS` (`tests/test_adapters.py:20-26`); every entry carries evidence text.
- Public surface: the 40-name `__all__`; `optivision.optimize` is a function; `__version__ == npm/package.json` version.

### Open questions for the architect

1. Is a pin amendment acceptable, or must `core/` stay a facade over the original file locations indefinitely?
2. Should `import optivision` stay eager (side effects: PIL limit, stage registration), or become lazy?
3. Should Method C replace or sit beside the lower-bound rule in `optimize()`, and what is the hold-out policy (Method C uses all queries; `calibrate` splits)?
4. Which artifact fields does a certificate need beyond `pipeline`, `stats`, `attrs`: fitted state, library tree hash, numpy/BLAS versions, query count, observation order, seeds?

## Appendix A. Probes (all read-only; run from the worktree root)

A.1 Import graph: `ast` walk of `src/optivision/**/*.py` collecting `Import`/`ImportFrom`, resolving relative levels, tagging a statement lazy when it sits inside a function; SCCs by Tarjan, with and without implicit parent-package edges.

A.2 Basis probe (output quoted in section 5.3):

```python
import sys, warnings
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np
from optivision.representation import MultiVectorCorpus
from optivision.compose import Pipeline
from optivision.scoring import maxsim_matrix
from optivision.stages import (AdaptiveMerge, HierarchicalMerge, RedundancyPruner, RandomPruner, PCAProjector,
    RandomProjector, TruncateProjector, Float16Quantizer, Int8Quantizer, Int4Quantizer, BinaryQuantizer, Lloyd2Quantizer)
rng = np.random.default_rng(0); d = 64
scale = np.exp(rng.normal(0, 1.0, d))                      # anisotropic coordinate scales
unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
mk = lambda n, lo, hi: [unit(rng.standard_normal((rng.integers(lo, hi), d)) * scale + 0.5 * scale).astype(np.float32) for _ in range(n)]
docs, qs = mk(30, 20, 40), mk(8, 6, 12)
Q = np.linalg.qr(rng.standard_normal((d, d)))[0]            # one orthogonal matrix
rot = lambda arrs: [(a.astype(np.float64) @ Q).astype(np.float32) for a in arrs]
C, CQ = MultiVectorCorpus.from_arrays(docs), MultiVectorCorpus.from_arrays(rot(docs))
QC, QCQ = MultiVectorCorpus.from_arrays(qs), MultiVectorCorpus.from_arrays(rot(qs))
span = float(np.ptp(maxsim_matrix(QC, C)))
name = lambda st: st.name + (" centred" if st.params().get("center") else "")
for st in (AdaptiveMerge(radius=0.8), AdaptiveMerge(radius=0.8, center="mean"), HierarchicalMerge(ratio=0.5),
           RedundancyPruner(0.9), RandomPruner(0.5, 0)):
    a = Pipeline([st]).fit(C).transform(C); b = Pipeline([type(st)(**st.params())]).fit(CQ).transform(CQ)
    print(name(st), np.array_equal(a.counts, b.counts), np.abs(a.vectors.astype(np.float64) @ Q - b.vectors).max())
for st in (PCAProjector(16), RandomProjector(16, 0), TruncateProjector(16)):
    a = Pipeline([st]).fit(C); b = Pipeline([type(st)(**st.params())]).fit(CQ)
    s1 = maxsim_matrix(a.transform_queries(QC), a.transform(C)); s2 = maxsim_matrix(b.transform_queries(QCQ), b.transform(CQ))
    print(name(st), np.abs(s1 - s2).max() / span)
def err(c, make):                                            # mean(1 - cos(decode(encode(x)), x))
    comp = Pipeline([make()]).compress(c); y = comp.decode_rows(0, comp.num_vectors).astype(np.float64)
    x = np.asarray(c.vectors, np.float64)
    return float(np.mean(1 - (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))))
for label, make in {"float16": Float16Quantizer, "int8 fixed": lambda: Int8Quantizer("fixed"),
                    "int8 per_vector": lambda: Int8Quantizer("per_vector"), "int8 per_dimension": lambda: Int8Quantizer("per_dimension"),
                    "int4": Int4Quantizer, "int4 centred": lambda: Int4Quantizer(center="mean"), "binary": BinaryQuantizer,
                    "binary centred": lambda: BinaryQuantizer(center="mean"), "lloyd2": Lloyd2Quantizer}.items():
    print(label, err(C, make), err(CQ, make))
```

A.3 Import-closure and pin checks: `sys.modules["PIL"] = None` (or `"yaml"`) then `import optivision.scoring` -> `ModuleNotFoundError`; `subprocess` listing `sys.modules` third-party names after each import; `sys.path.insert(0, "scripts/research"); import x_common; x_common.check_code()` -> `ok: True`, 14 of 14 files match.

A.4 Shadowing: a temporary package with both `adapters.py` and `adapters/__init__.py` resolves to the package; `import optivision.optimize as m` resolves to a function; `MultiVectorCorpus.load(p, mmap=True).vectors` is a plain `ndarray` (not a `memmap`); candidate counts from calling the search-space functions for d in {96, 128, 1024, 2560, 4096}.
