# Changelog

## Unreleased (branch `package-universal`, based on `research-investigation` at `91564fd`)

`__version__` stays `0.3.0` and nothing below is released. `optimize()`, `calibrate()`, the 40
names of `optivision.__all__`, the CLI commands that existed, every old import path and the v1
file formats are unchanged; `docs/package/MIGRATION.md` lists the observable differences. The
additions are engineering capability, tested on synthetic data and against the research code for
numerical equality; no registered experiment evaluated them as a package, and none of the
research results (`reports/research/`) was changed or re-run.

- **`optivision.certify()` and `CertifiedResult`** (`optivision/api.py`): a second entry point
  beside `optimize()`. It scores a frozen candidate list on all the queries, lets a selection
  rule choose the smallest certified candidate or abstain, and returns the decision, a
  certificate, the evidence, and (unless it abstained) the fitted pipeline and the compressed
  corpus. Parameters: `target`, `error`, `metric`, `reference` (`auto`, `labels`, `agreement`
  or a reference object), `space`, `rule`, `query_budget`, `order_seed`, `fallback`, `encoder`,
  `progress`, `created`. `result.retriever()`, `result.save()` and `result.report()`;
  `optivision.verify()` checks a certificate or an artifact directory. Nothing certified gives
  `Outcome.ABSTAIN` and the uncompressed index; there is no default fallback. Design:
  `docs/package/API_DESIGN.md`.
- **New top-level names**, loaded on demand: `certify`, `verify`, `verify_certificate`,
  `load_artifact`, `save_artifact`, `CertifiedResult`, `Certificate`, `CandidateSpace`,
  `Decision`, `Outcome`, `EncoderIdentity`. `import optivision` also registers the `rotation`
  stage.
- **CLI**: `optivision certify VECTORS QUERIES` and `optivision verify PATH`, with `--json`.
  `certify` exits 0 for every outcome and 2 for invalid input; `verify` exits 1 when a check
  fails.
- **`optivision.core`**, a model-agnostic layer over the existing modules (which are not moved
  or edited): content identities and the vector contract (`identity`, `vectors`); stage traits
  that state which changes of coordinate basis leave a stage's scores unchanged, fitted-state
  save and restore, byte accounting and a seeded `RotationProjector` (`stages`); a retrieval
  seam with exact, compressed, tiered and two-tier retrievers (`retrieve`); references
  (`LabelReference`, `RankingReference`), `Evidence` and `collect_evidence` (`evaluate`);
  `CandidateSpace`, the rule (`FixedSequenceBetting`, `BonferroniBetting`), `Decision` and
  `select` (`selection`); `Certificate`, `create_certificate`, `verify_certificate`,
  `sign_certificate` (optional HMAC), `save_artifact`, `load_artifact` and `code_identity`
  (`artifact`). Specifications: `docs/package/CERTIFICATE_SPEC.md`, `METHOD_C_SPEC.md`.
- **`optivision.backends`**: a `Backend` protocol and wrappers over `NumpyIndex` and
  `QdrantIndex`, which are not changed, and the canonical import place for the indexes.
- **Adapters** (`optivision.adapters` is now a package; same imports): `ArrayAdapter`,
  `CallableAdapter`, `corpus_from_arrays` for bringing your own vectors; `NemotronAdapter`;
  `EncoderAdapter.identity()` and `.dimension`. `load_adapter` now returns a `NemotronAdapter`
  for the NVIDIA ColEmbed ids (it used to fall through to the sentence-transformers adapter).
- **Namespaces and helpers**: `optivision.ext.vision` and `optivision.legacy` name the page
  layer and the original recipe without moving code; `optivision.testing` holds `make_corpus`,
  `make_queries`, `random_orthogonal`, `signed_permutation`, `apply_basis`, `reorder_rows`,
  `basis_probe`, `check_adapter` and `check_stage`.
- **Tests and documents**: `tests/test_core_*`, `test_api_*`, `test_backends_*`,
  `test_adapters_*`, `test_cli_certify.py`, `test_integration_*`, `test_public_surface.py` and
  `test_golden_formats.py` (files written by v0.3.0 keep reading back); the user guide in
  `docs/package/guide/` with examples that the tests run; the design and audit documents in
  `docs/package/`.

## Unreleased (branch `research-investigation`)

- **`ExactIndex.rescore` is batched.**
  - Each distinct candidate document is decoded once and scored against every query
    that shortlisted it, instead of re-decoding every query's candidates. The
    arithmetic (decode, float32 products, max, per-query sum), the column order and
    the API are unchanged.
  - On six ViDoRe datasets (500 pages, 50 candidates, all queries scored in one
    batch), rankings and per-query nDCG@5 were identical and the scores
    bit-identical.
  - Rescoring was 5.5–9.3x faster and the two-tier query 2.9–3.7x faster, on one
    laptop CPU. Evidence: `reports/engineering/batched_rescore/RESULTS.md`.

## 0.3.0 (2026-10-02)

The measured model set grows from three to six encoders, one of them
2,560 dimensions wide. Evidence and caveats are in `docs/UNIVERSAL.md` (R11, R12).

### Measured

- **ColQwen2-v1.0 and ColQwen2.5-v0.2 (R11)** on ViDoRe V1 DocVQA and InfoVQA
  (500 pages, 451 / 494 queries each). Their float baselines are within 0.5–1.8
  points of the published scores. Out of sample, `calibrate()` met its target in
  236 of 240 choices.
- **NVIDIA Nemotron ColEmbed 4B, 2,560-d (R12)** on the same two splits, float32
  on two Kaggle T4s, pinned revision `0ed152d9`. The model is licensed
  CC-BY-NC-4.0 and was evaluated for non-commercial research only.
  - Baseline nDCG@5 0.6645 / 0.9344 (published: 67.39 / 93.31).
  - Fixed configurations: Ward merging to a quarter keeps 98.8% / 99.5%; PCA to 320
    dims 99.1% / 99.4%; int8, centred int4 and binary 98.9–100.2%. Plain
    (uncentred) int4 collapses (0.4–0.9%); the cause is unresolved, and the default
    search space never uses it.
  - Out of sample over 20 splits per cell, `calibrate()` met its target in 119 of
    120 choices with the default space and 118 of 120 with the opt-in PCA families.
    These are empirical counts, not a guarantee or a success probability.
  - The best compression ratios R12 reports (for example 16x, 63x and 387x) are
    read off an in-sample Pareto frontier, so they are optimistic.

OptiVision has now been evaluated empirically on a 2,560-dimensional
multi-vector retrieval model, and its model-agnostic compression and optimization
machinery kept working at that width. Not measured: any 4,096-d model (the
`nemotron-colembed-vl-8b-v2` loader exists but was not benchmarked), ViDoRe
V2/V3, and corpora beyond 500 pages.

### Added

- `optivision.encoders.nemotron.NemotronColEmbedEncoder` for NVIDIA ColEmbed 4B /
  8B v2. It requires a pinned revision of the remote code and checks the
  architecture, pooling and width.
- `calibration.projection_search_space` and `wide_search_space`: opt-in PCA
  families. The default search space is unchanged.
- `OPTIVISION_SCORE_DEVICE=cuda` runs the query x document products on a GPU.
- `scripts/wide_smoke.py`, `scripts/colqwen_smoke.py`,
  `scripts/encode_vectors.py --backend nemotron`,
  `scripts/universal_study/s13_wide_study.py`, `s7b_selection_rules.py --space`.
- Kaggle notebooks `notebooks/kaggle_colqwen_encode.ipynb` and
  `notebooks/kaggle_wide_model.ipynb`, and instructions in `docs/GPU_RUN.md`.

### Fixed

- Checkpoint loads that would leave weights randomly initialised are now
  refused. The ColQwen key mapping was corrected.
- Multi-GPU placement splits only between whole decoder layers and handles
  nested decoder stacks.
- Fitting samples (PCA, centring, per-dimension int8) are capped in bytes as well
  as rows. At 2,560–4,096-d this cuts them from 4–7 GB to about 256 MB; they are
  unchanged up to 160-d.
- The calibration transform cache has a byte budget with LRU eviction
  (`OPTIVISION_CACHE_BYTES`).
- Ward merging at 512-d and wider uses a matrix product for distances: about 17x
  faster at 2,560-d, with identical clusterings.
- Configurations with identical stored size are tie-broken deterministically, not
  by measured latency.
- `Pipeline.label()` names non-PCA projections, for example
  `project(640, random)`.

## 0.2.0 (2026-10-01)

The universal multi-vector compression core: composable token / dimension /
codec stages, `calibrate()` and `optimize()`, and measurements on ColPali,
ColSmol and a text ColBERT. See the GitHub release notes and
`docs/RELEASE-AUDIT-2026-09-27.md`.
