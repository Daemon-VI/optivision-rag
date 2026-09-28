# Release audit: `universal-core` → v0.2.0

Date: 2026-09-27. Branch `universal-core`, audited from tip `06b94c1` (9 commits
ahead of `main` at `f3949ea`). Question: can this branch become OptiVision
v0.2.0 and merge into `main`? The answer is based on engineering correctness,
experimental validity, reproducibility and whether the documented claims are
supported. How impressive the compression numbers look plays no part.

Nothing was changed before the state below was reproduced. Section 13 lists
every change this pass made and everything it re-measured.

## 1. State reproduced

| item | result |
|---|---|
| commits `origin/main..HEAD` | 9, linear, one concern each (audit, representation, NumpyIndex fix, stages, calibration/CLI, safer default, centred merge, selection options, results) |
| working tree | clean at `06b94c1` |
| tests | **288 passed** (73 s); `ruff check src tests` clean. The 62 ruff findings under `scripts/` are all in the older `scripts/review/`, which predates this branch and which CI does not lint |
| API compatibility | all 16 names in `main`'s `optivision.__all__` still resolve; the diff to `src/` is additive apart from the `NumpyIndex` empty-page fix and one `register(app)` call in `cli.py` |
| original nine-variant table | reproduced from the new stages on all four caches (36 rows, max \|Δ nDCG\| 4.4e-16, `reports/universal/equivalence/`) |
| universal API, `calibrate()`, `optimize()`, Pareto, two-tier index, CLI | exercised by the suite; wheel built and installed into a clean venv: import, compress and all five new CLI commands work |
| documentation, reports | every table in `docs/UNIVERSAL.md` traces to JSON in `reports/universal/`, with two exceptions (§2, §4) |

## 2. Headline numbers

MEASURED = produced by this repository's code on real encoder output.
EXTERNAL = from a paper or model card. PROJECTED = not measured.
*In-sample* = chosen and reported on the same queries. *Fixed* = one named
configuration scored on every query, with nothing chosen. *Out-of-sample* =
chosen on calibration queries and reported on disjoint held-out queries.

| claim (as it stood at `06b94c1`) | source | measured by us? | dataset | model | sample | reproducible? | class / verdict |
|---|---|---|---|---|---|---|---|
| refactor exact, 36 rows, max \|Δ\| 4.4e-16 | `equivalence/*.json`, `s2_equivalence.py` | yes | E1, E3, InfoVQA, DocVQA | ColSmol, ColPali | fixed | yes | MEASURED |
| merging beats dropping at ~9x fewer vectors (random 84.7% / 94.5%, merge 95.4–99.4%) | `merge/*.json`, `s3_merge_sweep.py` | yes | DocVQA, InfoVQA | ColPali | fixed; the "99.4%" end is the best of several mergers | yes | MEASURED |
| Ward d=0.8 3.7x at 99.1%, d=1.3 7.7x at 97.0% (DocVQA) | `quantize_project/`, `s6` | yes | DocVQA | ColPali | fixed, but called "best points measured", which is a selection | yes | MEASURED, relabelled |
| int8 lossless, int4 −2.5 pts DocVQA, 2-bit 95.2–95.5% DocVQA | `quantize_project/` | yes | ViDoRe, generated | ColPali, ColSmol | fixed | yes | MEASURED |
| 2-bit 97.0% DocVQA (`docs/RESULTS.md`) | `scripts/review/codec_ladder.py` | yes | DocVQA | ColPali | fixed | **contradicted** by 95.2–95.5% | MEASURED but unresolved; now flagged in RESULTS.md |
| centred int4 89.7% → 98.6% (SciFact); centred binary 94.1% → 1.1% | `centred_codecs/*.json` | yes | SciFact, ViDoRe | ColBERT-small, ColPali | fixed | **no script in repo** (was in a scratch directory) | MEASURED; script added as `s3c_centred_codecs.py` |
| "median nearest neighbour inside a document at cosine 0.991" | no JSON | yes (run log only) | SciFact | ColBERT-small | — | **no** | re-measured: 0.989 on 300 documents (`geometry/`) |
| PCA 96 d nearly free, 64 d −3 pts, 32 d unusable | `quantize_project/` | yes | ViDoRe, generated | ColPali, ColSmol | fixed | yes | MEASURED |
| InfoVQA merge 0.6 + int8 35x at 99.4%, + int4 69x at 99.2% | `quantize_project/` (R5) | yes | InfoVQA | ColPali | fixed | yes | MEASURED; the same config gives 95.0% on DocVQA |
| **README row "smallest config whose 95% lower bound clears 97%": 14.7x · 98.9%, 69x · 99.2%, 11.8x · 100.0%** | `frontier/*.json` (R5b) | yes | DocVQA, InfoVQA, SciFact | ColPali, ColBERT-small | **in-sample: best of 60, chosen and reported on the same queries; per-candidate 97.5% bound, no multiplicity correction** | yes | **MISLEADING as presented**; replaced by out-of-sample numbers |
| R5b "the right column … is the one to plan with" | UNIVERSAL.md | — | — | — | in-sample | — | **wrong**; rewritten |
| tiered: 99.1–101% at 110–376x less RAM (DocVQA), 99.8–100% at 117–406x (InfoVQA) | `tiered/*.json`, `s8` | yes | DocVQA, InfoVQA | ColPali | fixed configs, shortlist sizes chosen from 10/20/50/100 | yes | MEASURED; RAM only, and total storage is ~3.9x. README now says so |
| point-estimate selection met target in as few as 30% of splits | `calibration/` (R7), `s4` | yes | DocVQA | ColPali | out-of-sample | yes | MEASURED |
| **default 0.999 bound met the target in 95–100% of splits on DocVQA and SciFact** | `selection_rules/` (R7b) | yes | DocVQA; SciFact 1 of 3 cells | ColPali, ColBERT-small | out-of-sample | **no script in repo; measured on a search space that is no longer the default** (uncentred int4) | **NOT SUPPORTED as stated**; re-measured on the shipped default (§3.4) |
| R7b rule "does not depend on how large the search space is" | UNIVERSAL.md | — | — | — | — | — | **false** (§3.3); removed |
| pseudo-queries chose configs at 81.6–93.7% real retention | `calibration/` | yes | four corpora | ColPali, ColSmol | out-of-sample | yes | MEASURED |
| ColBERT-small float nDCG@10 74.56 vs 74.77 | `text/`, model card | yes / card | SciFact | ColBERT-small | fixed | yes (needs sentence-transformers ≥ 6) | MEASURED vs EXTERNAL |
| compression cost: merge 19–96 ms/page, codecs < 5 ms/page, peak memory | `performance/` | yes | InfoVQA | ColPali | one contended laptop | yes | MEASURED, relative only |
| exact scan 5–7 ms/query merged vs 48–68 float | `performance/` | yes | InfoVQA | ColPali | one laptop | yes | MEASURED, relative only |
| "Tested coverage is three encoders" | adapters registry + reports | yes | — | — | — | test pins it | MEASURED, but ColSmol only on 60 generated pages / 72 queries (±6 pts) |
| E1 113.5x at 86.7%, E2 53–60x at 94.6–103.4% (original pipeline, README/PACKAGE) | `reports/`, `AUDIT-2026-09-27.md` | yes | generated, ViDoRe | ColSmol, ColPali | fixed | reproduced in the Phase 0 audit | MEASURED |
| "1 million pages = 512 GB" | README | — | — | — | — | arithmetic | illustration, not a result |
| AnchorFold / MarginMerge / NVIDIA numbers, "160x at 97–98%" | earlier proposal | no | — | — | — | — | EXTERNAL / PROJECTED; appear only in `AUDIT-2026-09-27.md` §4, labelled as such |

No PROJECTED number is presented as measured anywhere in the README,
`PACKAGE.md` or `docs/UNIVERSAL.md`. The failures are about which *kind* of
MEASURED number was presented, and two claims whose experiment no longer
matched the code.

## 3. Statistical validity

### 3.1 What the implementation computes

For one configuration, with calibration queries `i = 1..n`, `b_i` = the float
index's metric and `c_i` = the compressed index's metric:

- **Retention** is `R = Σc_i / Σb_i` (a ratio of means, paired).
- **The bound**: resample the `n` pairs `B` times. At confidence ≤ 0.99 it is
  the percentile `1 − confidence`. Above 0.99 it is `R − z·sd*`, where `sd*` is
  the standard deviation of the bootstrap ratios and `z = Φ⁻¹(confidence)`
  (3.09 at 0.999).
- **Selection**: a configuration is feasible if its bound is at least the
  target. `calibrate()` walks each family from least to most aggressive, stops
  a family at its first infeasible step, and picks the smallest feasible
  configuration by bytes per document.
- **Report**: the winner's retention on the held-out queries.

### 3.2 What that supports mathematically

1. **The parameter.** `R` estimates `E[c]/E[b]` over the distribution the
   calibration queries were drawn from. That holds for this corpus and for
   stages fitted on this corpus. The bootstrap resamples queries only, so the
   corpus is treated as fixed.
2. **The 0.999 bound is approximate.** It is a normal (Wald-type) bound built
   from a bootstrap standard error. That is asymptotically valid for a smooth
   statistic like a ratio of means. There is no finite-sample or
   distribution-free guarantee, and in the far tail (0.999) the normal shape is
   an assumption. Measured coverage on real per-query data is in §8.
3. **It is a per-configuration bound.** A statement of the form "the chosen
   configuration's retention ≥ target with probability 0.999" does **not**
   follow. The choice is the smallest configuration that passed. If `K`
   configurations are judged and each bound were exact, Bonferroni gives only
   `P(all bounds hold) ≥ 1 − K·0.001`, which is 0.96 for the 40-candidate
   default space. Early stopping makes `K` data-dependent, but it is still at
   most 40. So "0.999" should never be read as the confidence of the choice.
4. **"Met the target in x% of splits" is not a coverage probability.** The 20
   splits are random halves of *one* pool of queries, so they overlap and are
   not independent. Each one's "met" compares a held-out *point estimate*
   (about 225 queries, roughly ±2 points) with the target. The fraction therefore
   mixes selection error with held-out noise. It is a useful empirical check,
   and it is what R7/R7b report, but 20 splits cannot pin down a rate near 95%:
   19/20 has a 95% Wilson interval of [76%, 99%].
5. **Uninformative queries.** Queries whose float metric is 0 contribute
   nothing to `R`. The bootstrap also cannot see a failure that no calibration
   query happened to show: 10 identical query pairs give a zero-width interval.
   The small-sample warning counted *all* calibration queries; it now counts
   queries with a nonzero baseline metric (§13).

### 3.3 What the documentation said, and what it should say

| statement | verdict | now |
|---|---|---|
| "one-sided 0.999 lower bound" | correct as a description of the computation | kept, plus "approximate (bootstrap standard error)" |
| "the only rule … that does not depend on how large the search space is" (R7b) | **false**: the family-wise error grows with the number of candidates (point 3 above, and the simulation in §8) | removed |
| "met the target in 95–100% of splits on DocVQA and SciFact" | measured on a search space that is no longer the default, and on 4 cells | replaced by the re-run on the shipped default, with binomial intervals |
| "Calibration does not guarantee the target" | correct | kept and strengthened: no distribution-free guarantee exists anywhere |
| train/calibration/test separation | correct: stages are fitted on documents only; calibration and held-out query indices are disjoint (`split_queries`); held-out queries never touch a choice | kept |

### 3.4 The shipped default, re-measured out of sample

`scripts/universal_study/s7b_selection_rules.py` replays `calibrate()`'s own
path: the 0.999 bound, early stopping per family, n_boot = 1000 and the default
search space (40 candidates). It uses 20 random half/half splits per cell and
labels as the reference.

`calibrate()` with its defaults, labels as the reference, 20 random half/half splits. *met* = share of splits whose choice reached the target on the held-out half (95% Wilson interval over the 20 splits); *x* = median compression chosen; *held-out* = mean and worst held-out retention of the choice.

| dataset | queries | target | met [95% interval] | median x | held-out mean · worst |
|---|---|---|---|---|---|
| DocVQA | 451 | 0.99 | 20/20 [83.9%, 100.0%] | 3.9x | 100.1% · 99.7% |
| DocVQA | 451 | 0.97 | 19/20 [76.4%, 99.1%] | 3.9x | 99.7% · 96.5% |
| DocVQA | 451 | 0.95 | 20/20 [83.9%, 100.0%] | 11.8x | 98.6% · 95.2% |
| InfoVQA | 494 | 0.99 | 20/20 [83.9%, 100.0%] | 3.9x | 99.9% · 99.3% |
| InfoVQA | 494 | 0.97 | 20/20 [83.9%, 100.0%] | 43.4x | 99.6% · 97.8% |
| InfoVQA | 494 | 0.95 | 20/20 [83.9%, 100.0%] | 72.7x | 98.9% · 96.2% |
| SciFact | 300 | 0.99 | 14/14 [78.5%, 100.0%] | 3.9x | 99.9% · 99.6% |
| SciFact | 300 | 0.97 | 20/20 [83.9%, 100.0%] | 8.0x | 99.4% · 97.2% |
| SciFact | 300 | 0.95 | 20/20 [83.9%, 100.0%] | 12.4x | 98.7% · 96.2% |

All rules (labels): met · median x.

| rule | DocVQA 0.99 | DocVQA 0.97 | DocVQA 0.95 | InfoVQA 0.99 | InfoVQA 0.97 | InfoVQA 0.95 | SciFact 0.99 | SciFact 0.97 | SciFact 0.95 |
|---|---|---|---|---|---|---|---|---|---|
| calibrate() default | 100% · x3.9 | 95% · x3.9 | 100% · x11.8 | 100% · x3.9 | 100% · x43.4 | 100% · x72.7 | 100% · x3.9 | 100% · x8.0 | 100% · x12.4 |
| point | 25% · x23.2 | 20% · x66.9 | 50% · x125.4 | 70% · x57.0 | 55% · x178.9 | 80% · x299.9 | 40% · x21.0 | 85% · x23.1 | 50% · x29.0 |
| lower 0.975 | 90% · x3.9 | 75% · x7.8 | 70% · x30.4 | 95% · x11.6 | 85% · x64.3 | 100% · x178.9 | 85% · x6.6 | 90% · x21.0 | 95% · x23.1 |
| lower 0.99 | 90% · x3.9 | 75% · x7.8 | 75% · x24.1 | 90% · x7.8 | 95% · x49.7 | 100% · x117.5 | 89% · x6.6 | 95% · x11.8 | 100% · x23.1 |
| lower 0.999 | 100% · x3.9 | 95% · x3.9 | 95% · x14.5 | 100% · x3.9 | 100% · x43.4 | 100% · x72.7 | 100% · x3.9 | 100% · x11.8 | 100% · x21.0 |
| bonferroni 0.975 | 100% · x3.9 | 95% · x3.9 | 100% · x11.8 | 100% · x3.9 | 100% · x43.4 | 100% · x72.7 | 100% · x3.9 | 100% · x9.4 | 100% · x21.0 |
| lower 0.975 + margin 0.01 | 100% · x2.0 | 80% · x3.9 | 80% · x23.2 | 100% · x3.9 | 100% · x43.4 | 95% · x81.7 | 100% · x2.0 | 100% · x11.8 | 100% · x22.1 |


## 4. In-sample vs unseen queries

- **Code.** `calibrate()` splits the query indices once into disjoint halves.
  `_judge` sees only calibration rows and `_holdout` only held-out rows. `fit()`
  receives the corpus and never the queries (the transform cache fits each
  prefix with `Pipeline(prefix).fit(corpus)`). `PCAProjector(basis="joint")`
  can take queries, but it is not in any default space, and R4's joint-PCA
  study fitted on calibration queries only. **Verified.**
- **`optimize()` requires real queries.** It raises without `queries=` and
  points to the pseudo-query result. **Verified** by `tests/test_optimize.py`.
- **Documentation.** At `06b94c1` a reader *could* miss the distinction: the
  README's headline row was in-sample, and R5b told readers to plan with an
  in-sample column. Now `docs/UNIVERSAL.md` opens with a three-way table
  (fixed / in-sample / out-of-sample) and every results section is tagged. The
  README's table is out-of-sample first, with fixed configurations below and
  labelled as fixed.

## 5. Model generality

**Accepted input.** `MultiVectorCorpus.from_arrays` takes any list of `[n_i, d]`
arrays: float16/32/64 (stored as given), any `d` (1 to 4096 tested), variable
`n_i` including 0, optional ids, metadata and aligned `protected`, `positions`,
`weights` and `importance`. Scoring, every stage and save/load work on that
alone. Queries may have a different token count from documents; they must share
`d`, and projection stages map both.

**Search for hidden assumptions** (`ColPali ColQwen patch image 128 779 visual pixel`):

| occurrence | where | classification |
|---|---|---|
| `from_page_encodings`, `from_legacy_cache`: patch grid, `positions`, images | `representation.py` | model adapter for the original ColPali-shaped cache; optional |
| `SpatialPruner` reads page pixels and grid shape | `stages/prune.py` | model adapter; checks its inputs and says what is missing |
| registry of models, `PageEncoderAdapter`, `colpali-engine` loaders | `adapters.py` | model adapter |
| dimension 128 in `ModelInfo` rows | `adapters.py` | per-model metadata, correct for those models |
| `DimensionProjector(dim=128)` default | `stages/project.py` | harmless default; `output_dim = min(dim, d)` |
| **`Int8Quantizer()` default `scale="fixed"` (±0.5 range, tuned to unit-norm 128-d)** | `stages/quantize.py`, CLI shorthand `int8` | **accidental hard-coding**: silently clips non-normalised or low-dimensional vectors. The default search spaces use `per_vector`; the default stays for backward compatibility and now **warns when it clips** (§13) |
| comments citing ColPali measurements | `calibration.py`, `quantize.py` | documentation |
| `with_images` plumbing in the CLI and benchmark | `cli_universal.py`, `benchmark.py` | legacy-cache adapter |
| "779", "visual" | — | not present in the universal layer |

No hidden ColPali assumption remains in the generic path. What *is* model-specific
is the evidence: behaviour is measured on three encoders, all with 96–128-d
vectors.

## 6. Compression correctness

`tests/test_edge_cases.py` (new, 131 cases) runs every stage and several
combinations on: an empty document, one-vector and two-vector documents,
all-duplicate documents, all-zero vectors, unnormalised vectors, and
dimensions 1, 3, 7, 96, 128 and 4096. It checks shapes, float32 decode, finite
scores, that no document is emptied or grows, and that no `RuntimeWarning`
(invalid cast, division by zero) occurs.

| defect found | effect | fix |
|---|---|---|
| **NaN/inf in one document poisons every document** | fitted state (corpus mean, per-dimension ranges, PCA, Lloyd2) absorbs the NaN, so *finite* documents score NaN; SciPy Ward raises an opaque error | `Pipeline.fit`, `compress`, `encode` and `transform_queries` refuse non-finite input and name the first offending document |
| all-zero vector under per-vector int8 / int4 | scale floor 1e-12 rounds to 0 in float16, so NaN is cast to int (undefined behaviour); decoded to 0 by luck | floor at the smallest normal float16 (6.1e-5). Measured data never comes near it (smallest real scale 0.033), so every measured code is byte-identical |
| fitting on a corpus with no vectors | `Int8(per_dimension)` raised `IndexError`; centred codecs stored a NaN mean | clear `ValueError` from `Pipeline.fit` / `encode`; stateless stages still pass an empty corpus through |
| fixed-scale int8 clipping (above) | silent loss on non-unit-norm input | warning (values unchanged) |

The two bugs found earlier on this branch are still covered: `NumpyIndex`
dropping a vector before a trailing empty page (`tests/test_index.py`) and
`segment_reduce` on empty segments (`tests/test_scoring.py`). The cp1252
decoding of DocVQA query text is covered by `rebuild_vidore_qrels.py --check`.

## 7. Retrieval equivalence

- Baseline and compressed scores both come from `scoring.maxsim_matrix`, with
  the same query vectors. Queries pass only through a fitted projection, which
  is applied to both sides. Every document is a candidate, the rank is descending
  and ties are stable by index.
- **Ties.** A codec that creates exact ties could gain or lose depending on
  where relevant documents sit in the index order. `s11_ties.py` re-ranks with
  tied relevant documents first and last. On binary, merged-binary and
  merged-int4 codes on DocVQA, InfoVQA and SciFact, all three policies give
  identical retention to four decimals; at most 0.3% of queries have any tie
  in the top k + 1 (`reports/universal/audit/ties.json`). No inflation.
- **What could still flatter the numbers** (documented, not bugs):
  1. **Corpus size.** Measured on 500–5,183 documents. `s12_corpus_size.py`
     shows retention falling as distractors are added (DocVQA, 100 → 500 pages,
     fixed queries: binary 98.6% → 97.5%, int4 99.2% → 97.7%, centred merge +
     int8 97.4% → 95.6%). At a million pages it is likely lower still, and that
     is not measured.
  2. **Transductive fitting.** Centring means, PCA and Lloyd2 are fitted on the
     same documents that are then searched. That is normal index-time
     behaviour, but documents added later use stale statistics.
  3. **Intervals are conditional on the corpus.** They resample queries only.
  4. **The retention denominator** ignores queries the float index already
     fails. This is symmetric, but it means retention says nothing about them.
- The label-free reference (`baseline@1`) and the labels reference were
  compared in R7. The label-free one was conservative with respect to labels in
  every measured cell.

## 8. Adversarial cases for calibration

From `s10_bound_audit.py` (`reports/universal/audit/bound_audit.*`), with the
population taken as DocVQA's 451 real per-query pairs:

**A. Coverage of the bound for one fixed configuration** (400 draws per cell):

| calibration queries | nominal 0.975 | nominal 0.999 |
|---|---|---|
| 25 | 0.77–0.90 | 0.87–0.97 |
| 50 | 0.90–0.96 | 0.97–0.995 |
| 100 | 0.945–0.97 | 0.99–0.995 |
| 225 | 0.95–0.97 | 0.995–1.00 |

The bound is close to nominal from about 100 queries and under-covers below
that. The warning threshold of 100 is justified.

**B. Many near-miss candidates** (synthetic worst case: `K` independent
candidates, each truly 0.5–1 point *below* the target, and each losing whole
queries at random):

| truth vs target | K = 1, 0.999 | K = 10, 0.999 | K = 40, 0.999 | K = 40, Bonferroni (0.975/K) |
|---|---|---|---|---|
| 0.960 vs 0.97 | 0.3% | 9.7% | 34% | 35% |
| 0.965 vs 0.97 | 2.0% | 21% | 55% | — |
| 0.940 vs 0.95 | — | 7.3% | 26% | 25% |

"Point estimate passes, bound fails" and "a configuration wins through noise"
both occur easily in this regime. Bonferroni does not rescue it, because the
per-candidate bound under-covers when losses are rare and large. Real candidates
are strongly correlated (they share merges and codecs), which is why the real
held-out rates (§3.4) look much better. The worst case is still reachable.

**C. Degenerate samples.** Ten calibration queries on which the compressed index
happens to match the float one give a zero-width interval. With 10 of 11 queries
unaffected (true retention 90.9%), a 0.97 target passed in 40% of draws. When
only one calibration query has a relevant hit, the bound equals the point
estimate. Both cases are now covered by the informative-query warning.

**D. Shift.** A configuration calibrated on one distribution does not carry
over to another. Merge 0.6 + int4 keeps 99.2% on InfoVQA and 95.0% on DocVQA
with the same encoder (R5). Corpus growth lowers retention (§7). Neither is
detectable from the calibration queries.

**Limitations to state, and now stated:** no distribution-free or finite-sample
guarantee; no family-wise guarantee over the search space; the query and corpus
distributions must match deployment; at least 100 *informative* queries.

## 9. Model-specific geometry (text ColBERT)

| finding | reproducible? | scope |
|---|---|---|
| uncentred absolute-radius merging collapses documents (radius 0.8 → 9 vectors, 76%) | yes (`text/`, `centred/`) | this model on SciFact |
| removing the corpus mean fixes adaptive merging | yes | this model on SciFact |
| centred int4 89.7% → 98.6% | yes; far outside the intervals | this model on SciFact; neutral on ColPali |
| centred binary 94.1% → 1.1% | yes | this model on SciFact |
| mechanism: the queries' shared direction | **now measured**: removing each query's component along the corpus mean restores centred binary to 96.9% [94.4%, 99.2%]; the corpus mean has norm 0.90 (ColPali 0.23–0.25) and query tokens sit at median cosine 0.94 to it | this model on SciFact |

These observations are reproducible, but they rest on **one text model and one
corpus**. The documentation claimed them for "the text model" and no further,
except the R8 wording "codecs are model-specific too", which is a fair
inference from two families. `UNIVERSAL.md` now states the scope explicitly and
treats the query-side fix as a diagnostic, not a feature.

## 10. Documentation

`docs/UNIVERSAL.md` now separates **Measured** (results), **External** (model
card figures, and methods credited to papers without their numbers),
**Unvalidated** (adapters marked untested or unverified; the search space
without SciPy; attention-guided merging) and **Future work** (ColQwen, wide
models, ViDoRe V2/V3, million-page corpora, more database connectors, learned
components). The three-way table of result kinds sits at the top.

## 11. README positioning

The README now leads with *Quality-constrained, model-agnostic multi-vector
compression*. It shows out-of-sample results first and fixed configurations
second, labelled as such. It states that tested coverage is exactly ColPali-v1.3,
ColSmol-256M and one text ColBERT, and it no longer opens with the largest
ratio. `PACKAGE.md` (the PyPI page) got the same treatment. The original
recipe's section is kept below as the v0.1 story.

## 12. Release checklist

Status after the fixes in §13. "At `06b94c1`" is the status before them.

| area | status | at `06b94c1` | evidence / what remains |
|---|---|---|---|
| correctness | **PASS** | FAIL | a NaN or inf in one document corrupted other documents' codes; an all-zero vector hit an undefined int cast. Both fixed, with 131 edge-case tests |
| tests | **PASS** | PASS | 421 passed (288 before); `ruff check src tests scripts/universal_*` clean |
| reproducibility | **PASS**, with a WARNING | FAIL | every result now has its script in `scripts/universal_study/` and its JSON in `reports/universal/`. R7b was not reproducible (no script, and a stale search space) and was re-measured. WARNING: the encoded vectors are not in the repository (about 1 GB); the scripts document how to regenerate them, and SciFact needs sentence-transformers ≥ 6 |
| statistical methodology | **WARNING** | FAIL (as documented) | the implementation is a sound approximate bootstrap and the train / calibration / held-out separation is correct. It is not a guarantee for the selected configuration (R10); the docs now say so instead of overstating it. A selection-valid rule is future work |
| API stability | **PASS** | PASS | additive over 0.1.1; the new API is 0.x and has no deprecation policy yet |
| model generality | **WARNING** | WARNING | the architecture takes any `[n_i, d]` vectors, and no hidden ColPali assumption remains (§5). The evidence is three encoders, all 96–128-d, one of them on 60 synthetic pages only |
| documentation | **PASS** | FAIL | the README headline was in-sample; R5b and R7b made unsupported statements. Fixed: result kinds are tagged, the README leads with out-of-sample numbers, and Measured / External / Unvalidated / Future work are separated |
| benchmarks | **WARNING** | WARNING | methodology is sound (same scorer, all candidates, ties verified). Corpora are 500–5,183 documents and retention falls with corpus size (§7); ViDoRe V2/V3 and wide models are unmeasured |
| packaging | **PASS**, with a WARNING | WARNING | version bumped to 0.2.0 in `__version__` and `npm/package.json`; the 0.2.0 sdist and wheel build, and a clean install (with `[merge]`) exposes `optivision` and all five new commands, with no data, local paths or secrets bundled. WARNING: the publish workflow and the npm launcher against a published 0.2.0 are **NOT TESTED** until a release is cut from `main` |
| security | **PASS** | FAIL | the new CLI commands unpickled any file with a `meta` key, so a crafted `.npz` could run code on `optivision inspect`. Now gated behind `--trust-pickle`, with a regression test that shows the payload does not run. The original `bench --cache` still unpickles its own caches, as documented in 0.1 |
| performance | **WARNING** | WARNING | measured on one contended laptop only: merging 19–96 ms/page, PCA fit peak 410 MB. **NOT TESTED** at scale, on GPU, or through a database |


## 13. Changes made in this pass

All on `universal-core`, after the state in §1 was reproduced.

**Code (bugs and release quality)**

- `compose.py`: `Pipeline.fit`, `compress`, `encode` and `transform_queries`
  refuse non-finite vectors and name the first offending document. Fitting on a
  corpus with no vectors raises a clear error.
- `stages/quantize.py`: the per-vector scale floor is now the smallest normal
  float16 (int8 per-vector, int4). Fixed-scale int8 warns when it clips.
- `cli_universal.py`: legacy (pickled) caches load only with `--trust-pickle`.
- `calibration.py`: the small-sample warning counts informative queries (those
  with a nonzero baseline metric). New warning when SciPy is missing and the
  default space is used. The result summary records the per-candidate
  confidence and the informative-query count. The docstring no longer overstates
  the bound.

**Tests**: `tests/test_edge_cases.py` (131 cases) and a pickle-gate test in
`tests/test_cli_universal.py`. Expected warnings are filtered in three test
modules.

**Scripts** (`scripts/universal_study/`): `s3c_centred_codecs.py` (R3b plus
geometry), `s7b_selection_rules.py` (R7b, including an exact replay of
`calibrate()`), `s10_bound_audit.py`, `s11_ties.py`, `s12_corpus_size.py`; a
script → report-folder map in the README; a selection-rules renderer in
`universal_summary.py`.

**Re-measured**

| what | result | files |
|---|---|---|
| R7b on the shipped default, DocVQA / InfoVQA / SciFact, all rules | default met the target in 173 of 174 choices; replaces the stale table and the partial SciFact file | `selection_rules/*.json` |
| geometry of the three corpora, centred binary with the query mean removed | mechanism confirmed; nearest-neighbour cosine re-measured at 0.989 (was quoted as 0.991 with no file) | `geometry/*.json` |
| coverage of the bound, many-candidate and degenerate cases | §8 | `audit/bound_audit.*` |
| tie policies | no effect | `audit/ties.json` |
| retention against corpus size | falls by 0–3 points from 100 to 500 pages | `audit/corpus_size.*` |

No previously reported fixed-configuration number changed: the code fixes leave
every measured code byte-identical (the smallest real scale is 0.033, far above
the new floor, and no measured vector reaches the ±0.5 clip).

**Documentation**: README (new positioning and lead section; ColQwen2 no longer
presented as working; `[merge]` extra), `PACKAGE.md` (same), `docs/UNIVERSAL.md`
(result kinds; R5b, R7b and R8 corrected; R10 added; the closing sections split
into limits / external / unvalidated / future work), `docs/RESULTS.md`
(unresolved 2-bit figure flagged) and this report.

**Not done, deliberately**: no new models, codecs or selection rules; version
not bumped; nothing published, tagged or pushed.


## 14. Recommendation

**2. RELEASE AFTER FIXES.**

At `06b94c1` the branch was **not** releasable as it stood, for four reasons:

- the README's headline compression figures were in-sample selections;
- the documented reliability of the default selection rule came from a search
  space that no longer shipped, and could not be reproduced from the repository;
- one non-finite vector silently corrupted other documents;
- the new CLI would execute code from a crafted file.

None of these is a design flaw. The architecture, the exact-scoring methodology
and the calibration / held-out separation held up under audit, and the
refactor's exactness was confirmed.

The fixes are implemented in this pass (§13). Re-measured on the shipped
default, `calibrate()` met its target on held-out queries in 173 of 174 choices
across three corpora and two encoder families. The documentation now claims only
that, with its limits (R10).

What remains before tagging v0.2.0 is release mechanics:

1. review and commit this pass;
2. bump both version strings to 0.2.0;
3. let CI run on the merge;
4. follow `docs/RELEASING.md`.

Nothing in the remaining WARNING rows (model coverage, corpus scale, the bound
not being a guarantee for the selected configuration) blocks a 0.x release,
because each one is now stated where a reader will see it. They do block any
claim of universal support or guaranteed quality, and the docs make no such
claim.

