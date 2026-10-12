# NVIDIA Nemotron ColEmbed 8B (4,096-d): encoder check

**Status, 2026-10-06.** A run of `nvidia/nemotron-colembed-vl-8b-v2` on the two ViDoRe V1
splits used for every other visual encoder here, with the same scripts as the 4B run
(R12 in `docs/UNIVERSAL.md`). All numbers below are MEASURED and are copied from the JSON
files named in §5. This file reports them; it changes no earlier report.

**What this run is, and is not.**

- It is a development-style measurement of one more encoder, on the pattern of R12.
- It is **not** held-out or transfer evidence. The encoder is candidate 9 of the closed
  pool of `reports/research/NEXT_PHASE_PREREGISTRATION.md` §13.4. Its vectors were
  compressed and scored here before any `heldout-freeze` tag exists, so it no longer
  meets condition 1 of §13.2 and cannot be registered for XH. The registry (`reports/research/README.md`) does not
  say so yet on this branch: that note is still to be added to the research record, and
  until it is, this file is where the ineligibility is written down.
- It is not part of X1, X2 or X3, and it was not run under Method C. The selection
  results in §4 are those of the shipped rule and of the comparison rules of
  `s7b_selection_rules.py`, over 20 random half/half splits of one query pool. They are
  empirical summaries with no post-selection guarantee.
- Licence of the model: CC-BY-NC-4.0, non-commercial. Weights were downloaded only inside
  the Kaggle sessions; only metrics and logs are committed.

## 1. How it was run

| | |
|---|---|
| model, revision | `nvidia/nemotron-colembed-vl-8b-v2`, `34b640612f311ed05a6c7c62c6564847ed555f5f` |
| code | `main` at `4d520603469e179135ffd9df922b02d4504b4add` (v0.3.0) |
| notebooks | `notebooks/kaggle_nemotron8b_docvqa.ipynb`, `notebooks/kaggle_nemotron8b_infovqa.ipynb` |
| hardware | two Kaggle Tesla T4s; 36 decoder layers split 18 / 18; `sdpa` attention |
| inference dtype | **float16** (the 4B run used float32); vectors stored as float32 |
| software | Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, accelerate 1.14.0, numpy 2.1.3, scipy 1.16.3 |
| data | `vidore/docvqa_test_subsampled` (500 pages, 451 queries), `vidore/infovqa_test_subsampled` (500 pages, 494 queries) |
| scoring | exact MaxSim on the GPU |
| duration | 4.42 h (DocVQA), 4.29 h (InfoVQA), encoding included |

Smoke test on 6 DocVQA pages (`smoke_nemotron-colembed-vl-8b-v2.json`): empty loading
report, width 4,096, finite vectors, unit-norm error 5.4e-7, each query retrieved its own
page (6 of 6), and exact MaxSim matched the model's own `colbert_score` to 3.8e-6.

**Caveat on float16.** The baseline of this run is the float16-inference encoding. No
float32 encoding of the 8B exists, so the float16-against-float32 check that R12 said an
8B run would need was not made. On the 4B that check gave a median vector cosine of
0.999998 and a minimum of 0.923. Every retention below is relative to the float16-inference
baseline, and "float32 bytes" is the size of those vectors stored at 32 bits.

## 2. Baseline

| | DocVQA | InfoVQA |
|---|---|---|
| nDCG@5 | 0.6731 | 0.9465 |
| nDCG@10 | 0.6933 | 0.9499 |
| Recall@1 | 0.5942 | 0.9170 |
| Recall@5 | 0.7406 | 0.9696 |
| vectors per page | 758 | 738 |
| bytes per page at float32 | 12,424,151 | 12,090,868 |
| bytes per page at float16 | 6,212,076 | 6,045,434 |
| exact scan, ms per query (T4) | 63 | 72 |
| encode time for 500 pages, s | 503 | 606 |

For comparison, the 4B baselines in this repository are 0.6645 and 0.9344. No published
score for the 8B was checked for this file.

## 3. Fixed configurations

Retention of nDCG@5 against this encoder's own baseline, with the 95% paired bootstrap
interval. A configuration read off this table was not chosen in advance; the table is not
a selection result.

| configuration | factor | DocVQA | InfoVQA |
|---|---|---|---|
| float16 | 2.0x | 0.999 [0.996, 1.001] | 1.000 [1.000, 1.000] |
| int8 per vector | 4.0x | 0.989 [0.978, 0.999] | 0.999 [0.995, 1.003] |
| centred int4 | 8.0x | 0.996 [0.987, 1.007] | 0.999 [0.995, 1.003] |
| plain int4 | 8.0x | **0.011** [0.003, 0.021] | **0.007** [0.001, 0.014] |
| 2-bit Lloyd | 16.0x | 0.994 [0.979, 1.008] | 0.996 [0.992, 0.999] |
| binary | 32.0x | 0.990 [0.978, 1.002] | 0.998 [0.994, 1.001] |
| merge to 1/4 | 4.0x | 0.992 [0.980, 1.005] | 0.994 [0.988, 1.000] |
| merge to 1/10 | 9.9x | 0.972 [0.952, 0.993] | 0.988 [0.980, 0.996] |
| merge to 1/4 + binary | 127.7x | 0.982 [0.968, 0.996] | 0.996 [0.990, 1.002] |
| PCA to 2,048 | 2.0x | 1.001 [0.996, 1.006] | 0.999 [0.997, 1.002] |
| PCA to 1,024 | 4.0x | 0.996 [0.985, 1.005] | 0.998 [0.992, 1.003] |
| PCA to 512 | 8.0x | 0.989 [0.973, 1.002] | 0.994 [0.986, 1.001] |
| PCA to 128 | 32.0x | 0.948 [0.922, 0.976] | 0.968 [0.952, 0.982] |
| PCA to 512 + binary | 256x | 0.972 [0.953, 0.992] | 0.994 [0.988, 1.000] |
| PCA to 128 + binary | 1,024x | 0.955 [0.928, 0.983] | 0.985 [0.974, 0.996] |
| two-tier: binary in RAM, int8 rescoring of 50 | RAM 32x | 0.989 [0.978, 0.999] | 0.999 [0.995, 1.003] |
| two-tier: merged binary in RAM, int8 rescoring of 50 | RAM 128x | 0.989 [0.978, 0.999] | 0.999 [0.995, 1.003] |

What this agrees with in the 4B run: plain int4 collapses again, to about 1% of the
baseline; centred int4, binary and merging to 1/4 stay within about one point; projection
to one eighth of the width (512) costs about one point. What differs: per-vector int8
reads 0.989 on DocVQA here against 0.997 on the 4B, with overlapping intervals.

All sign-code and 4-bit rows are in the encoder's native coordinate basis. X3 found such
results basis-dependent on the 128-d and 96-d development encoders; no rotation was applied
to these vectors.

## 4. Selection rules (20 splits; empirical, no guarantee)

The shipped rule (`calibrate()` default), labelled reference. "Met" is the share of splits
in which the held-out half reached the target; "worst" is the lowest held-out retention.

| space | target | DocVQA: met | worst | median factor | InfoVQA: met | worst | median factor |
|---|---|---|---|---|---|---|---|
| default (40) | 0.99 | 15 of 15 selected (5 abstained) | 0.996 | 2.0x | 18 of 20 | 0.987 | 14.0x |
| default (40) | 0.97 | 20 of 20 | 0.973 | 8.0x | 20 of 20 | 0.970 | 298.7x |
| default (40) | 0.95 | 20 of 20 | 0.966 | 123.2x | 20 of 20 | 0.970 | 436.1x |
| wide (82) | 0.99 | 15 of 15 selected (5 abstained) | 0.996 | 2.0x | 19 of 20 | 0.987 | 16.0x |
| wide (82) | 0.97 | 18 of 20 | 0.959 | 8.0x | 20 of 20 | 0.977 | 512.0x |
| wide (82) | 0.95 | 20 of 20 | 0.952 | 127.7x | 20 of 20 | 0.974 | 1,024x |

- The shipped rule missed its target in 5 of the 230 selections above: twice on InfoVQA at
  0.99 in the default space, once there in the wide space, and twice on DocVQA at 0.97 in
  the wide space.
- Choosing by the point estimate met the target in 40–100% of splits per cell, which is
  the pattern of every earlier encoder.
- With the uncompressed ranking as the reference instead of labels, the rule's choices met
  the labelled target in every split, and chose less compression (median 2.0x to 8.0x on
  DocVQA). This is the direction X1 measured on the development stores. Agreement with the
  reference ranking is not retrieval quality.

## 5. Files

| what | where |
|---|---|
| fixed configurations, two-tier rows, in-sample frontier | `reports/universal/wide/nemotron-colembed-vl-8b-v2_{docvqa,infovqa}_test_subsampled.json` |
| selection rules, default space | `reports/universal/selection_rules/nemotron-colembed-vl-8b-v2_*.json` |
| selection rules, wide space | `reports/universal/selection_rules_wide/nemotron-colembed-vl-8b-v2_*.json` |
| smoke test | `reports/universal/wide/smoke_nemotron-colembed-vl-8b-v2.json` |
| logs, GPU, commit, `pip freeze`, run summary, relevance labels | `reports/universal/wide/logs/nemotron-colembed-vl-8b-v2/` |

The result files are byte for byte those exported from the two Kaggle sessions. The
archives they came from have sha256 `112a84e329be8963577233623b256c2e64c8ae374fbbc9965425fedb71dcf9f9`
(DocVQA) and `63751b77279ed54fe074b14256b7c5d39c9ae2b277f0b6e6cc01acf996fdd88b` (InfoVQA).
The encoded vectors were not exported and do not exist outside those sessions, so nothing
that needs the vectors (a rotation test, Method C on per-query stores built here, a
float32 comparison) can be run without encoding again.

## 6. Not updated by this file

- `src/optivision/adapters.py` still lists the 8B as untested ("not encoded here"), and
  `docs/UNIVERSAL.md` (R12) still says it was not benchmarked. Both belong to the released
  library on `main` and were left unchanged on this branch.
- `reports/universal/TABLES.md` was not regenerated.
- `paper/optivision_research.tex` lists 4,096-d models under "not covered".
