# optivision.easy: internal contract

Design source: `C:/Users/Lenovo/projects/optivision-easy-design/FINAL_DESIGN.md` (sections cited as
FD 5.1 etc.; section 14 = one release, everything in 0.6.0). This file fixes the module boundaries so
that modules can be built in parallel. Public names are in `src/optivision/easy/__init__.py`; shared
value types and errors are in `_types.py`; the number formatter is `_claim.py`; feature switches are
`_ledger.py` + `presets/easy_ledger.json`. Those four files are owned by the integrator: do not edit
them; if you need a change, say so in your report.

## Hard rules (all modules)

- **Additive only.** Never edit the 11 byte-pinned files: `calibration.py`, `compose.py`,
  `stages/base.py`, `stages/merge.py`, `stages/quantize.py`, `stages/project.py`,
  `compression/__init__.py`, `compression/binary.py`, `scoring.py`, `evaluation.py`,
  `representation.py`. Never edit `scripts/research/`, `reports/universal/`, `paper/`, or any
  existing test's expectations. Existing public signatures do not change.
- **Light import.** `import optivision.easy` and the modules `_types`, `_claim`, `_ledger`,
  `_levels`, `_resolve`, `_options`, `_chart`, `_glossary`, `_doctor` must not import torch, scipy,
  transformers, sentence_transformers, colpali_engine, pypdfium2 or PIL at module level. Import
  heavy things inside functions.
- **Every number shown to a user goes through `_claim`** (`claim()`, `Line`, `Block`); templates
  are ASCII, end with a `Next:` line, never mix evidence kinds on one line, never use the words
  in `tests/test_docs_examples.py` `BANNED_PHRASES` / `BANNED_WORDS`, nor "lossless", "universal",
  "guaranteed", "first", "SOTA", "state of the art", "veto".
- **Gated features** call `_ledger.require("<feature>")` (raises `FeatureOff` with the reason) or
  check `_ledger.is_on`. Never switch a gated feature on in code.
- Style: match the surrounding repo (docstrings with reasons, `from __future__ import annotations`,
  dataclasses, no type-ignore spam). Line length up to 120. Tests are offline (no network, no GPU);
  use `optivision.adapters.CallableAdapter` / `ArrayAdapter` and `optivision.testing.synthetic`.
- Tests live in `tests/test_easy_<area>.py`. Run with
  `PYTHONPATH=src C:/Users/Lenovo/projects/optivision-rag-v2/.venv/Scripts/python.exe -m pytest -q tests/test_easy_<area>.py`.
- Do not `git commit` unless your task says so; never add any Claude / AI attribution anywhere.

## Reused core APIs (unchanged)

- `optivision.compose.Pipeline` (`from_dict`, `to_dict`, `fit`, `transform`, `transform_queries`,
  `compress(corpus, fit=False)`, `encode`), `CompressedCorpus(codes, offsets, dim, quantizer, ids)`.
- `optivision.core.stages.state.pipeline_fitted_state(pipeline)` / `restore_pipeline_state(...)`.
- `optivision.representation.MultiVectorCorpus` (`vectors`, `offsets`, `ids`, `from_arrays`, `select`,
  `save`/`load(mmap=True)`).
- `optivision.core.retrieve.exact.CompressedRetriever(store, pipeline, max_block_bytes)` with
  `.search(queries, k) -> Ranking` and `.rescore(queries, candidates)`; `ExactRetriever`;
  `optivision.core.retrieve.tiered.TwoTierRetriever(first, reranker, candidates)`.
- `optivision.api.certify(...)`, `optivision.core.evaluate.reference.RankingReference/LabelReference`,
  `optivision.core.selection.spaces.CandidateSpace.from_families`,
  `optivision.core.selection.statistics.to_y/betting_pvalues`.
- `optivision.adapters`: `MODELS` (id -> `ModelInfo`), `load_adapter(model_id, loader=None, **kw)`,
  `EncoderAdapter` (`encode_documents(docs, ids)`, `encode_queries(queries, ids)`),
  `CallableAdapter`, `ArrayAdapter`, `check_retriever(adapter, documents, queries, allowed_misses=2)`,
  `measured.measured_results(model_id)`, `measured._measured_via(model_id)`.
- `optivision.types.PageRef(doc_id, page_no).page_id`, `optivision.ingest.downscale`,
  `optivision.demo.looks_blank`, `optivision.core.identity.canonical_json`,
  `optivision.core.artifact.codeid._file_digest`.

## Ids and sources

- A folder source `.../my_pdfs/` gives `doc_id = "my_pdfs/<relpath posix>"`; a single file
  `.../x/report.pdf` gives `doc_id = "x/report.pdf"`. Page ids are `f"{doc_id}::p{page}"`, page 1-based;
  an image file is one page. Two source folders with the same name are told apart by `"name~2"`.
- Built from vectors: ids are the given `ids` or `"0".."n-1"`; `Hit.file` is None.

## Modules

### `_resolve.py` (rule agent)
- `resolve(model: str | None) -> ModelInfo | None` exact id, case-insensitive id, or unique short name.
  Raises `UnknownModel` (3 closest ids) / `AmbiguousModel` (matches with loader and licence).
- `rows_model(model_id) -> tuple[str | None, str]`: the catalogue id whose rows are used and the basis
  string (`"own rows"`, `"measured via <id>"`, `"unmeasured"`).

### `_levels.py` (rule agent) + `scripts/make_easy_levels.py` + `presets/easy_levels.json`
- `easy_levels.json` is generated (with source sha256s) from `presets/measured.json` and the follow-up
  reports (FD 5.1.2 step 1), excluding legacy rows of the three B0 models until B0 rows exist. It holds,
  per model: modality, datasets (name, pages, queries), and for each ladder rung and dataset:
  `bytes_per_page`, `ratio_f32`, `low`, `point`, `vectors_per_page`; plus pooled minima for unmeasured
  models (FD 5.1.4) and the margin constants.
- `LADDER: dict[modality, list[RungDef]]` with `RungDef(name, plain, chain, spec, one_bit, merged,
  size_class)`; float16 and float32 rows first.
- `s_of_n(n) -> float`; `need(target, rung, n_pages) -> float` (margins from the ledger `margins` entry).
- `select(model: str | None, target: float, n_pages: int | None = None, *, checked: bool = False,
  modality: str | None = None, vectors_per_page: float | None = None, layout: str | None = None)
  -> Selection` (`_types.Selection`). `layout=` is a hand pick ("centred 4-bit" or a row label):
  it returns a Selection with `hand_picked=True` and that row as pick, statuses unchanged.
  `vectors_per_page` drives the shape gate (G-shape, FD 5.5).
- `row(model, name) -> LayoutRow`; `golden fixture` in `tests/data/easy_rule_golden.json`.

### `_options.py` (rule agent) — public `options()` and `Options` (FD 4.1)
- `Options.rows`, `.pick`, `.selection`, `.as_dict()`, `.save(path)` (.html/.svg via `_chart`),
  `_repr_html_`, `__str__` (a `_claim.Block`).
- Page counting for `pages="folder/"` uses `_docs.count_pages(source)`; the time estimate uses
  `_encode.estimate(model_id, n_pages)`; disk uses `shutil.disk_usage`. Until those modules exist,
  import them lazily inside functions.

### `_chart.py`, `_glossary.py`, `_report.py`, honesty tests (presentation agent)
- `_chart.svg(selection: Selection, *, n_pages=None) -> str` (static SVG, x = MB per 1,000 pages log,
  y = lowest kept with a bar to worst point, per-rung needed tick, hollow refused marks with reason,
  pick ringed); `_chart.html(selection) -> str` self-contained page with the vanilla-JS target slider
  and page-count control (no CDN), whose rule is the same as `_levels.select` (a node golden test).
- `_glossary.glossary(term: str | None = None)`, `TERMS` (FD 6.5).
- `_report.Report`: built from an index's manifest; every number with its tag; `.save(path)` (.html,
  .md, .json); `.lines`; an A4-printable HTML section.

### `_docs.py` (documents agent) — FD 5.2
- `FileRecord(path, relpath, doc_id, size, mtime, kind)`; `list_source(source) -> (files, skipped)`
  sorted, recursive, `.pdf` + common images; `skipped` is `[(path, reason)]`.
- `count_pages(source, progress=None) -> PageCount(n_pages, n_files, skipped)` without rendering.
- `iter_pages(file: FileRecord, *, dpi=150, max_side=1536) -> Iterator[(PageRecord, PIL.Image)]`,
  closes files in `finally`; `PageRecord(id, doc_id, file, relpath, page, text, n_chars, blank, sha256,
  landscape)`; sha256 of the rendered RGB bytes.
- `render_page(path, page, dpi=150) -> PIL.Image`.
- `profile(pages: Sequence[PageRecord], vectors_per_page: Sequence[int] | None = None) -> FolderProfile`
  (n_pages, n_text_pages, n_blank, aliases {id: canonical id}, boilerplate lines, landscape_share,
  text_share, checkable: bool, reason).

### `_probes.py`, `_veto.py` (documents agent) — FD 5.3.1, 5.5.1
- `make_probes(pages: Sequence[PageRecord], *, seed=0, max_pages=600, per_page=3) -> list[Probe]`,
  `Probe(id, text, family, page_id, relevant: frozenset[str])`.
- `hard_set(probes, ref_rankings: dict[probe_id, list[(doc_id, score)]], *, k_keep=10) -> HardSet`
  (keep rule, hardness g, hardest third, min 60, cap 300, frozen sha256).
- `ndcg_at(ranking_ids, relevant, k=5) -> float`.
- `practice_check(cand: dict[probe_id, list[str]], ref: dict[probe_id, list[str]], relevant,
  clusters: dict[probe_id, str], target, *, B=2000, seed=0) -> CheckStat(R, U, n, passed)`.
- `fixed_sequence(sequence: Sequence[LayoutRow], rank_fn: Callable[[LayoutRow], dict[probe_id, list[str]]],
  ref, relevant, clusters, target) -> SequenceResult(chosen: LayoutRow, steps: list[(row, CheckStat)])`.
- `presets/veto_strength.json` + `scripts/make_veto_strength.py` (simulation, FD 5.5.1 table).
- Opt-in tiers in `_probes_vlm.py` (local VLM, extra `easy-gen`), `_probes_api.py` (API LLM, cost first,
  `allow_upload=True`), `_ocr.py` (extra `easy-ocr`): probes-tier agent.

### `_store.py`, `_index.py` (storage agent) — FD 9 `_store`, `_index`
- Index folder `<name>.ovx/`: `manifest.json` (written last, `os.replace`), `segments/NNN/codes.npy`,
  `offsets.npy`, `ids.json`; `state/pipeline.json` (+ `state/fitted.npz`), `text/NNN.zlib.json`,
  `aliases.json`, `reference/` (probe texts, query vectors, relevant sets, float16 reference scores),
  optional `float16/NNN/{vectors.npy,offsets.npy}`, `build.json` (build record), `log/` (opt-in query log).
- `Store.create(path, *, pipeline: fitted Pipeline, info: dict) -> Store`; `Store.open(path, mode="r"|"a")`;
  `store.append(compressed: CompressedCorpus, texts: dict[id,str] | None, float16: MultiVectorCorpus | None)`;
  `store.remove(ids)`; `store.commit(**manifest_updates)`; `store.manifest`; `store.ids`; `store.n_pages`;
  `store.pipeline() -> fitted Pipeline`; `store.segments() -> list[CompressedCorpus]` (memory-mapped codes);
  `store.float16_corpus() -> MultiVectorCorpus | None`; `store.text(id)`; `store.reference` get/set;
  `store.sizes() -> dict` (vectors, text, record, reference, float16, total bytes);
  `store.lock()` context manager (`os.open(O_CREAT|O_EXCL)`; `IndexLocked`); `store.compact()`
  (above 8 segments); `store.close()`. Round trip must be bit-exact.
- `_index.Searcher(store)`: `.search(queries: MultiVectorCorpus, k) -> list[list[(id, score)]]` exact
  top-k merged over segments with `CompressedRetriever(..., max_block_bytes=32 MiB)`, ties by id order;
  `.rank_all(queries, k)` for the practice check; `rank_float(corpus, queries, k, pipeline=None)` helper
  that compresses a float16 corpus with a candidate pipeline and ranks (used by build's practice check).
- `_tiered.py` (integrations agent): small-memory layout (`TwoTierRetriever` over a hot binary copy and
  memory-mapped int8), gated by `small_memory_layout`.

### `_encode.py`, `_doctor.py` (encoder agent) — FD 4.2, 4.4, 5.5 G-wire
- `load_encoder(model: str | EncoderAdapter, *, device="auto", dtype="recorded", trust_remote_code=False)
  -> Encoder` with `.adapter`, `.info`, `.record` (build-record dict: model, loader, revision/resolved sha,
  dtype, attention impl, max_side, batch size, library versions, device), `.encode_pages(images, ids,
  progress=None) -> MultiVectorCorpus` (CUDA OOM halves the chunk down to 1, records the final size),
  `.encode_queries(texts) -> MultiVectorCorpus`. Raises `RemoteCodeRefused`, `MissingExtra`.
- `wiring_check(encoder, pages: list[(PageRecord, image)]) -> Wiring(ok, hits, n, used_sample, message)`
  (12 text pages from different files, a sentence unique among them, `allowed_misses=2`; on failure rerun
  on `easy/data/sample.pdf`; raise `WiringError` only if both fail).
- `estimate(model_id, n_pages) -> Estimate(lo_s, hi_s, basis, gpu)` from `presets/encode_rates.json`
  (generated by `scripts/make_encode_rates.py` from campaign `run.json` files).
- `install_line(model_id) -> str`, `recorded_dtype(model_id) -> str | None`.
- `easy/data/sample.pdf`: 20 pages we generate (reportlab), distinct sentence per page, MIT.
- `_doctor.doctor(model=None) -> DoctorReport` (FD 4.4) + `presets/known_issues.json`.

### `_api.py`, `_build.py` (build agent) — FD 4.2, 4.3, 4.4, 5.6, 5.7
- `build(source, model=None, target=None, *, path=None, ids=None, layout=None, keep_float16=False,
  keep_text=True, work_dir=None, prune=False, rebuild=False, trust_remote_code=False, device="auto",
  dtype="recorded", progress=True, accuracy=None, sample=None, probes="text", queries=None, answers=None,
  max_disk=None, allow_upload=False) -> Index`.
- Order: resolve -> plan (count, disk check) -> lock -> load encoder -> wiring -> list + diff sources
  (path,size,mtime; sha256 on change) -> encode to float16 shards in `work_dir` (500-page segments, atomic
  shard manifest, resume) -> profile -> fit on a seeded sample (cap 200,000 rows, seed 0) -> selection
  (`_levels.select`) -> shape gate -> practice check (`_probes` + `_veto.fixed_sequence`) -> real-query gate
  -> settle segments `compress(fit=False)` into the store -> text, aliases, reference -> time 20 sample
  queries -> manifest -> printed summary (FD 4.2) -> delete shards unless `keep_float16`.
- `load(path, *, device="auto", log_queries=False) -> Index`.
- `Index`: `search(query | list, k=5)`, `search_vectors(q, k=5)`, `add(source, ids=None) -> SyncReport`,
  `check(...)` (delegates to `_check.run_check(index, ...)`), `upgrade(...)` (delegates to
  `_upgrade.run_upgrade(index, ...)`), `report()` (`_report.Report.from_index(index)`), `explain()`,
  `relink(folder)`, `demo()` (`integrations.demo.launch(index)`), `export_qdrant(...)`, `close()`,
  context manager; attributes `target, layout (LayoutRow), size, n_pages, examples, evidence, pipeline,
  retriever, path, selection, record`. Internals other modules may use: `index._store`, `index._searcher`,
  `index._encoder()` (lazy load at the recorded revision/dtype), `index._manifest`.

### `_check.py`, `_upgrade.py`, `_querylog.py`, `_spotcheck.py` (checks agent) — FD 5.5.3, 5.7
- `run_check(index, queries=None, *, answers=None, drift="auto") -> CheckReport` (`.ok`, `.lines`,
  `.as_dict()`, `.apply()` prints its time then re-encodes; never runs on its own).
- `run_upgrade(index, queries, *, answers=None, apply=False) -> UpgradeReport` through the unchanged
  `certify()` with error `0.05 * 2**-k` (k = attempt count stored in the manifest), never-used de-duplicated
  queries, `RankingReference(depth=1)` or `LabelReference`, `CandidateSpace.from_families` of larger rungs
  qualifying at N, fallback = current pipeline, `order_seed=0`; needs `keep_float16=True` or re-encode.
- Opt-in query log (local, privacy notice; never on by default), spot check (re-encode each question's
  stored top 10; can only flag), `max_disk=` helper (highest target in 0.01 steps that fits).

### `cli_easy.py` (CLI agent) — FD 4.5, 6.2
- `register(app)`: commands `options`, `build`, `find`, `check`, `doctor`, `quickstart` in
  `rich_help_panel="Easy mode"`; `sys.stdout.reconfigure(errors="replace")`; the 16 existing commands stay.
  One line in `cli.py` registers it. npm launcher routing in `npm/`.

### `integrations/` (integrations agent) — FD 14
- `integrations/langchain.py` (`OptiVisionRetriever(BaseRetriever)`), `integrations/llamaindex.py`
  (`OptiVisionRetriever(BaseRetriever)`), both carrying the evidence tags in document metadata;
  `integrations/qdrant.py` export (gated `qdrant_export`), `integrations/byaldi.py` import (gated
  `byaldi_import`), `integrations/demo.py` Gradio search box (extra `easy-ui`, never logs searches).

## Extras (pyproject, integrator)

`easy` (scipy), `easy-st`, `easy-colpali`, `easy-ocr`, `easy-gen`, `easy-ui`, `langchain`, `llamaindex`,
`byaldi`. Agents state what they need; the integrator edits `pyproject.toml`.
