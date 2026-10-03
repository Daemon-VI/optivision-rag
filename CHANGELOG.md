# Changelog

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
