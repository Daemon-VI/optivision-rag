# Test report: integration, reliability and strength checks of the model-agnostic layer

Status: 2026-10-06, branch `package-universal`. Author: the testing and reliability engineer of
the package restructuring. This file records what the tests cover, what they found and what they
do not cover. It is not a research document.

**What these tests are.** They check the implementation on synthetic data and against the frozen
research functions. They are not research experiments: no registered experiment was re-run, no
experiment script under `scripts/research/` was executed, no real encoder output and no real
benchmark data were read. (The golden-file parity test of section 4 reads stored per-query arrays
that already exist in `reports/research/E1/`, read-only, as an equality check.) A number computed
on synthetic vectors says nothing about any encoder, pipeline or retrieval quality. Agreement with
the uncompressed ranking is not retrieval quality, and no test here merges the two.

## 1. Summary

* Integration tests added: 1370 tests in five files (`tests/test_integration_*.py`),
  1359 passing and 11 expected failures (`xfail(strict=True)`, each one a
  defect of section 6) when they were written. D1, D2 (the retriever cases) and D3 were fixed
  afterwards (section 6): the 7 tests that expected them now pass, so
  `test_integration_layers.py` gives 548 passed and 4 xfailed, and the 4 left are the pinned
  `compose.py` cases (D4 and the `Pipeline.transform_queries` width). The five files together took
  about 130 s on this shared machine.
* Existing tests of the new layers (written by the engineers of those layers, not changed here):
  2926 tests in 35 files.
* Strength check: 62 one-line mutations of a scratch copy of `src/`, each run against the
  existing tests of its layer: 59 caught, 3 survived (M01b, M24, M43). Each
  survivor is also missed by every other existing new-layer test file (section 5), and each is
  killed by a test of this work.
* Defects: 3 in the new layers and 1 in a pinned file (section 6); none touches the selection
  arithmetic. The first (`ExactRetriever` does not validate documents) is the one that can give a
  wrong ranking without a message. The three in the new layers were fixed after this report was
  first written; the one in the pinned file stays.
* Parity of the selection arithmetic with the frozen research functions: bitwise on 352 random
  problems and 3 stacked shapes, exact on more than 300 boundary trials, exact on 100,800
  selections of the stored E1 results by default and on 504,000 with `OPTIVISION_FULL_PARITY=1`
  (section 4).

## 2. What was added, by file

| file | tests |
|---|---|
| `tests/test_integration_determinism.py` | 8 |
| `tests/test_integration_end_to_end.py` | 64 |
| `tests/test_integration_layers.py` | 552 |
| `tests/test_integration_properties.py` | 658 |
| `tests/test_integration_serialization.py` | 88 |
| total (5 files) | 1370 |

| file | what it covers |
|---|---|
| `tests/test_integration_layers.py` | pipeline composition and identity; basis-sensitivity of random pipelines; ranking equivalence across the retrievers and the NumPy backend; reference, metric and evidence stability; candidate-space identity; the bad-input tables (vectors, evidence arrays, dtypes). Also the shared helpers the other files import (`build_corpus`, `build_queries`, the pipeline table, `maxsim64`, the rounding-aware comparison). |
| `tests/test_integration_properties.py` | selection invariants over many seeds on a small list and on the 40-candidate default list; the exact level boundary; the statistical sanity of the whole decision path. |
| `tests/test_integration_determinism.py` | the same input and configuration give the same digests, twice in one process and once in a fresh interpreter with another `PYTHONHASHSEED`; identities do not depend on how bytes are held. |
| `tests/test_integration_serialization.py` | save and load preserve meaning for corpora, the v1 compressed file, evidence and fitted state. |
| `tests/test_integration_end_to_end.py` | `certify` -> `save` -> `load_artifact` -> `verify_certificate` at every level -> `retriever().search`, on text-like, vision-like, float16, agreement, unlabelled-query, edge-case, abstaining and fallback scenarios; tampering. |

## 3. Coverage

### 3.1 By layer

Existing tests (unchanged by this work):

| file | layer | tests |
|---|---|---|
| `tests/test_adapters_arrays.py` | adapters | 39 |
| `tests/test_adapters_identity.py` | adapters | 16 |
| `tests/test_adapters_nemotron.py` | adapters | 18 |
| `tests/test_adapters_page_ids.py` | adapters | 4 |
| `tests/test_api_certify.py` | public API | 110 |
| `tests/test_api_compat.py` | public API (compatibility) | 33 |
| `tests/test_backends_numpy.py` | backends | 17 |
| `tests/test_backends_protocol.py` | backends | 53 |
| `tests/test_backends_qdrant.py` | backends | 18 |
| `tests/test_core_accounting.py` | stages (byte accounting) | 66 |
| `tests/test_core_artifact_bundle.py` | artifact (bundle) | 96 |
| `tests/test_core_certificate.py` | artifact (certificate) | 206 |
| `tests/test_core_certificate_tamper.py` | artifact (tampering) | 325 |
| `tests/test_core_codeid.py` | artifact (code identity) | 38 |
| `tests/test_core_collect_evidence.py` | evaluate (collect, deploy) | 50 |
| `tests/test_core_evidence_types.py` | evaluate (evidence) | 72 |
| `tests/test_core_facades.py` | facades (pipeline, similarity) | 40 |
| `tests/test_core_identity.py` | identity | 164 |
| `tests/test_core_layering.py` | dependency direction | 3 |
| `tests/test_core_metrics_from_ranking.py` | evaluate (metrics) | 47 |
| `tests/test_core_reference.py` | evaluate (references) | 27 |
| `tests/test_core_retrieve.py` | retrieve | 113 |
| `tests/test_core_retrieve_tiered.py` | retrieve (two tiers) | 66 |
| `tests/test_core_rotation.py` | stages (rotation) | 44 |
| `tests/test_core_selection_engine.py` | selection (engine) | 79 |
| `tests/test_core_selection_rules.py` | selection (rules) | 46 |
| `tests/test_core_spaces.py` | selection (candidate spaces) | 51 |
| `tests/test_core_stage_contracts.py` | stages | 350 |
| `tests/test_core_stage_state.py` | stages (state) | 129 |
| `tests/test_core_stage_traits.py` | stages (traits) | 149 |
| `tests/test_core_statistics.py` | selection (arithmetic) | 130 |
| `tests/test_core_statistics_parity.py` | selection (parity with research) | 30 |
| `tests/test_core_vectors.py` | vector contract | 113 |
| `tests/test_ext_namespaces.py` | ext and legacy namespaces | 59 |
| `tests/test_testing_helpers.py` | testing helpers | 125 |
| total (35 files) | | 2926 |

The integration tests touch every layer: `core.identity`, `core.vectors`, `core.pipeline`,
`core.similarity`, `core.stages` (traits, state, accounting, rotation), `core.retrieve`,
`core.evaluate` (metrics, references, evidence, `collect_evidence`, `compress_selected`),
`core.selection` (spaces, statistics, rules, engine, decisions), `core.artifact` (certificate,
bundle, verification), `backends` (the NumPy backend; the Qdrant wrapper is not used), `testing`
(`make_corpus`, `make_queries`, `basis_probe`) and the public entry points `certify`, `verify`,
`verify_certificate` and `load_artifact`.

### 3.2 By property

| property | integration tests | existing tests that also cover it |
|---|---|---|
| A pipeline rebuilt from `to_dict()` (and from its JSON) is identical in behaviour: codes, decoded vectors, fitted-state digests, query projection and scores are bitwise equal after fitting on the same corpus (15 pipelines x float32 and float16 x 3 seeds with dimensions 13, 21, 37; an empty and a single-vector document in every corpus) | `layers::TestPipelineComposition` | `test_core_stage_state.py`, `test_core_stage_contracts.py` |
| The pipeline identity is equal for equal specs (key order and fitting do not matter) and changes with every parameter of every registered stage and with stage order, membership and count | `layers::TestPipelineComposition` | `test_core_facades.py` |
| A pipeline with a coordinate codec (binary, int4, int8, float16, truncate, lloyd2) is `basis_sensitive` in any position and after any stage (500 random valid orders, 500 plain stage lists in any order); sensitivity is never lost by appending stages; only orthogonally invariant stages give a pipeline that is not sensitive | `layers::TestBasisSensitivityOfPipelines` | `test_core_stage_traits.py` |
| `ExactRetriever` equals `maxsim_matrix` + `rank` (bitwise); `CompressedRetriever` equals scoring the projected queries + `rank` (bitwise) for every pipeline; the float32 pipeline ranks exactly like the uncompressed corpus | `layers::TestRankingEquivalence` | `test_core_retrieve.py` |
| A two-tier or tiered search whose shortlist is the whole corpus equals the exact ranking; the NumPy backend agrees with the core retriever; both under the rounding-aware criterion (section 3.3) | `layers::TestRankingEquivalence` | `test_core_retrieve_tiered.py`, `test_backends_numpy.py` |
| Reference relevant sets are stable across calls and after a corpus save and load; `metrics_from_ranking` equals `per_query_metrics` for ndcg, recall, mrr and hit at several cut-offs, with ties, unreachable documents and padding | `layers::TestReferenceAndEvidenceStability` | `test_core_reference.py`, `test_core_metrics_from_ranking.py` |
| An `Evidence` built twice has one digest, survives `to_npz` and `from_npz`, and its digest changes with every field (one ulp is enough) | `layers::TestReferenceAndEvidenceStability`, `serialization::TestEvidenceFiles` | `test_core_evidence_types.py` |
| Every candidate of a space can be deployed as it was evaluated: same fitted-state digests, same per-query metrics, same sizes as `byte_accounting` | `layers::TestReferenceAndEvidenceStability` | `test_core_collect_evidence.py`, `test_core_accounting.py` |
| The default space has the registered identity at any dimension; insertion order and family order are part of an identity; `from_dict(to_dict())` is the identity; `pipelines()` returns fresh objects | `layers::TestCandidateIdentity` | `test_core_spaces.py` |
| Bad vectors (NaN, infinity, another dimension, no data, a raw array), bad evidence arrays (32 kinds) and wrong dtypes are refused with a `ValueError` or `TypeError` that names the problem; documented exceptions and defects are listed in the tables | `layers::TestBadVectorInputs`, `TestBadEvidenceInputs`, `TestDtypesAreNeverChangedSilently` | per layer |
| Selection invariants: the selected candidate is a certified member with the smallest key; nothing is selected when nothing is certified; a lower error level never enlarges the certified set; on nested prefixes the p-values and the selected size never increase (METHOD_C_SPEC 2.3, TV11); interleaving chains, relabelling families and changing query ids do not change the decision | `properties::TestTheDecisionIsConsistent` | `test_core_selection_engine.py`, `test_core_selection_rules.py` |
| The level boundary: `p <= level` rejects, the level is `error / F` (chain rule) and `error / K` (union rule), one ulp below rejects nothing | `properties::TestTheLevelIsInclusiveAndExactlyTheErrorOverTheChains` | `test_core_statistics.py` (TV7), `test_core_selection_rules.py` |
| Statistical sanity of the whole decision path at the null boundary, 2,000 samples per configuration (section 3.4) | `properties::TestStatisticalSanityOfTheDecisionPath` | `test_core_statistics.py` (20,000-replicate check of the arithmetic) |
| Determinism: 194 digests (corpus identity, fitted state, codes, query projections, scores, rankings, byte accounting, traits, space identity, evidence, decisions, certificate, bundle files) are equal in two runs and in a fresh interpreter with another hash seed | `determinism` | - |
| Serialization: corpora (all aligned arrays, float64 included, NaN bit patterns of `positions`), the v1 compressed file (11 pipelines; the codecs with fitted state are refused), evidence, fitted state of 15 pipelines | `serialization` | `test_core_vectors.py`, `test_golden_formats.py`, `test_core_artifact_bundle.py` |
| The certified index is the shipped index: the loaded artifact reproduces the recorded per-query metric values exactly, through the stored codes and the restored fitted state, and through the ranking of the deployed retriever | `end_to_end::TestTheCertifiedIndexIsTheShippedIndex` | `test_api_certify.py`, `test_core_certificate.py` |
| Verification at all three levels with live objects; tampering at each level is reported by the check that is meant to report it | `end_to_end::TestTamperingIsNoticed`, `TestTheCertifiedIndexIsTheShippedIndex` | `test_core_certificate_tamper.py` |
| Abstention, fallback, unlabelled queries, query order and budget, the other rule, other metrics, float16 input, the agreement construct, the declared encoder never changing a decision | `end_to_end` | `test_api_certify.py` |

### 3.3 The rounding-aware criterion used for the backend and two-tier comparisons

Two float32 scoring paths are equal to rounding, never bitwise (`docs/package/BACKEND_ARCHITECTURE.md`
sections 5 and 10), so the comparisons use the criterion of that document, made explicit:

* the float64 MaxSim `E` is computed from the decoded float32 values;
* the float32 error bound per query is `B = 2 * (dim + n_tokens + 2) * 2^-23 * sum |q_i| * max |d_j|`;
* each returned score is within `B` of `E`; no document is ranked above one whose `E` exceeds its
  own by more than `2B`; for every depth `k` in (1, 3, 5) at which `E` separates the k-th from the
  (k+1)-th document by more than `2B`, the top-k set equals the float64 top-k set; the scores at
  one rank of two paths differ by at most `2B`.

Identical order is required wherever the criterion forces it, and near-ties may swap. The
criterion fails a path that ranks wrongly by more than rounding; it says nothing about retrieval
quality.

### 3.4 Statistical sanity of the decision path

`properties::TestStatisticalSanityOfTheDecisionPath` runs `select()` on 2,000 seeded synthetic
samples per configuration (225 queries, the "rare total loss" construction of METHOD_C_SPEC 9.2: a
candidate loses the whole per-query value with probability `1 - R`, so its true retention is
exactly `R`). The bound is `error + 4 * sqrt(error (1 - error) / 2000)`, which is 0.0695 at error
0.05. An implementation check, not a power study and not a research result.

| configuration | what is counted | observed | bound |
|---|---|---|---|
| 4 chains x 2 candidates, target 0.95; chains a and b start just below the target (R = 0.948), chains c and d exactly at it (R = 0.95), all second candidates at R = 0.90; every candidate is a true null | a candidate with R below the target is selected | 0.004 (8 of 2,000) | 0.0695 |
| same | any candidate is certified (a false rejection of a true null: the family-wise error) | 0.0095 | 0.0695 |
| same samples, arithmetic only, the level `error` instead of `error / 4` (negative control) | any candidate is certified | 0.090 | must exceed 0.0695 (it does) |
| 5 candidates in 3 chains, one equal to the baseline on every query (R = 1) and four bad (R = 0.3 to 0.8), target 0.9 | a bad candidate is selected or certified | 0 | 0.0695 |
| same | the candidate with R = 1 is selected | 1.0 | at least 0.8 (a rule that always abstains would pass the line above) |

The observed rates are well below the bound because the rule is conservative at these sample
sizes (METHOD_C_SPEC 9.2 reports the same for the arithmetic). The test would notice a missing
split of the level over the chains (the negative control), a reversed test direction or a gross
error in `to_y`; it would not notice a small change of the level or of the bet, which the parity
tests and the boundary tests catch. Runtime: about 11 s per configuration (the engine is not
vectorised across samples).

## 4. Parity evidence for the selection rule against the frozen research functions

Source: `tests/test_core_statistics_parity.py` (existing, written with the layer; run here, not
changed). Compared with the frozen `scripts/research/e1_stats.py` (imported by path inside the
test; its git blob is checked against the registered `13494307...`) and with `e1_1_study.run_cell`:

| what is compared | cases | comparison |
|---|---|---|
| the p-values of `betting_pvalues(to_y(...))`, `FixedSequenceBetting.decide` and `BonferroniBetting.decide` against the frozen functions; rejected sets and selections | 352 random problems (8 sample sizes 1, 2, 3, 7, 25, 100, 225, 450 x 44 candidate counts; arbitrary family interleavings; exact size ties; targets drawn from 7 values, error levels from 3) | p-values bitwise (`tobytes`); sets, selections and levels exactly |
| `fixed_sequence_by_family` and `smallest` with the level placed at a p-value, one ulp below and one ulp above | at least 300 trials | exact |
| stacked inputs of `betting_pvalues` | 3 shapes (2-D, 3-D, 4-D) | bitwise |
| both rules against the frozen orchestration `run_cell` on synthetic data | n in {50, 225}, 8 seeds, every target and error level of the study, both rules | selections exactly |
| the stored E1 selections of Method C and of method D regenerated from the stored per-query arrays seed by seed (METHOD_C_SPEC 9.3) | 7 stores x 3 sample sizes x 3 targets x 2 error levels x 2 methods x 400 seeds (every fifth of 2,000) = 100,800 selections by default; x 2,000 seeds = 504,000 with `OPTIVISION_FULL_PARITY=1` | exact (int16 arrays equal) |
| the same stored selections through `decide` and `select()` on seeds 0 and 1999 | 7 stores x 144 decisions = 1,008 | exact |

Measured on this machine (Windows 11, Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, shared with other
jobs): default mode `30 passed in 13.05 s` (wall 14.0 s); `OPTIVISION_FULL_PARITY=1`:
`30 passed in 41.40 s`. All 30 tests ran (none skipped): the stored E1 stores are present in this
worktree and are real files, not Git LFS pointers.

This is an equality check of an implementation against its source. It does not show that the
procedure is valid; that is the statement of METHOD_C_SPEC section 6, conditional on A1-A6.

## 5. Independent strength check of the existing tests

**Method.** `src/` was copied to a scratch directory (never the worktree); for each mutation one
line of one source file was changed in the copy, the existing test files of the layer were run
against it with `PYTHONPATH` pointing at the copy and `-x` (stop at the first failure), and the
copy was restored. One mutation at a time. A mutation is *caught* when at least one existing test
fails. The behaviours are those the design documents say matter: the inclusive level comparison,
the family stop, the tie-break order, `level = error / number of families`, `basis_sensitive` for
a codec after a rotation, the `Ranking` tie order, the corpus digest covering the aligned arrays,
bit parity of the arithmetic, `validate_corpus` finiteness, the fitted-state digest covering every
array, the label-only query drop, canonical JSON key sorting, and smaller ones (input checks,
digest coverage, restore checks). 62 mutations in all (34 chosen first, then 28 of smaller
behaviours).

**Result.** 59 of 62 were caught by the existing tests of their layer.
Survivors: M01b, M24, M43. Each survivor was then run against every other existing new-layer test
file (between 2,634 and 2,730 tests besides those of its layer; the count grew as other tests
were added during the work) and survived those too, so the gap is real and not a matter of which
files were run. The last column shows the first failure under `-x`; other
tests may fail as well.

| id | source file | one-line change | behaviour | first failing existing test |
|---|---|---|---|---|
| M01a | `core/selection/statistics.py` | `if pk <= level:` -> `if pk < level:` | inclusive level comparison in fixed_sequence_by_family (p <= level) | test_core_statistics.py :: test_tv7_decisions_from_hand_made_p_vectors |
| M01b | `core/selection/rules.py` | `np.flatnonzero(p <= level)` -> `np.flatnonzero(p < level)` | inclusive level comparison in the Bonferroni rule | NOT CAUGHT by the layer's files nor by the other existing files (2634 passed in 105.05s (0:01:45)); KILLED by test_integration_properties.py :: TestTheLevelIsInclusiveAndExactlyTheErrorOverTheChains :: test_the_union_bound_rule_rejects_exactly_at_error_over_the_number_of_candidates |
| M02 | `core/selection/statistics.py` | `stopped.add(f)` -> `pass` | the family stops at its first non-rejection | test_core_statistics.py :: test_tv7_decisions_from_hand_made_p_vectors |
| M03a | `core/selection/statistics.py` | `(bytes_[k], vectors[k], k)` -> `(bytes_[k], vectors[k], -k)` | tie-break: earlier position wins | test_core_statistics.py :: test_tv7_decisions_from_hand_made_p_vectors |
| M03b | `core/selection/statistics.py` | `(bytes_[k], vectors[k], k)` -> `(bytes_[k], k)` | tie-break: fewer vectors before position | test_core_statistics.py :: test_smallest_uses_exact_size_then_vectors_then_position |
| M03c | `core/selection/statistics.py` | `(bytes_[k], vectors[k], k)` -> `(vectors[k], bytes_[k], k)` | size key order: bytes before vectors | test_core_statistics.py :: test_tv7_decisions_from_hand_made_p_vectors |
| M04a | `core/selection/rules.py` | `n_families = len(dict.fromkeys(v.families))` -> `n_families = len(v.families)` | level = error / number of DISTINCT families | test_core_selection_rules.py :: test_the_level_counts_the_distinct_labels_of_the_list_passed |
| M04b | `core/selection/rules.py` | `level = v.error / n_families` -> `level = v.error` | level = error / number of families (no division) | test_core_selection_rules.py :: test_the_level_counts_the_distinct_labels_of_the_list_passed |
| M05a | `core/stages/traits.py` | `elif t.randomised_basis: canonical = False` -> `elif False: canonical = False` | a seeded random map ends canonical coordinates (no conditional invariance after it) | test_core_stage_traits.py :: TestPipelineRule :: test_rule_and_data_agree |
| M05b | `core/stages/traits.py` | `basis_sensitive=overall is not _O,` -> `basis_sensitive=overall in (_N, _U),` | basis_sensitive unless every stage is orthogonally invariant | test_core_stage_traits.py :: TestSignedPermutation :: test_equal_under_signed_permutations_and_not_under_rotations |
| M05c | `core/stages/traits.py` | `elif mixed and inv is _S:` -> `elif False and inv is _S:` | a signed-permutation codec after a rotation counts as NONE | test_core_stage_traits.py :: TestNone :: test_a_plain_permutation_of_the_axes_changes_the_scores |
| M05d | `core/stages/traits.py` | `if not canonical or t.basis_invariance is not _S:` -> `if not canonical:` | only SIGNED_PERMUTATION stages are credited after PCA | test_core_stage_traits.py :: TestPipelineRule :: test_rule_and_data_agree |
| M06a | `core/retrieve/types.py` | `if (tied & ~(indices[:, 1:] > indices[:, :-1])).any():` -> `if (tied & ~(indices[:, 1:] < indices[:, :-1])).any():` | Ranking tie order: equal scores by ascending position | test_core_retrieve.py :: TestRanking :: test_ties_must_ascend_by_position_when_the_convention_is_index |
| M06b | `core/retrieve/tiered.py` | `order = np.argsort(keyed, axis=1, kind="stable")` -> `order = np.tile(np.arange(keyed.shape[1]), (keyed.shape[0]` | TwoTierRetriever sorts the shortlist by store position before ranking | test_core_retrieve_tiered.py :: TestTwoTierRetriever :: test_t11_exact_cold_ties_break_by_store_position_not_by_shortlist_order |
| M07a | `core/vectors.py` | `_ALIGNED = ("protected", "positions", "weights", "importan` -> `_ALIGNED = ("positions", "weights", "importance")` | the corpus identity covers the protected array | test_core_vectors.py :: test_identity_can_be_recomputed_from_the_arrays_alone |
| M07b | `core/vectors.py` | `_ALIGNED = ("protected", "positions", "weights", "importan` -> `_ALIGNED = ("protected", "positions", "weights")` | the corpus identity covers the importance array | test_core_vectors.py :: test_identity_changes_with_exactly_the_array_that_changed |
| M08a | `core/selection/statistics.py` | `return (c - target * b + target) / (1.0 + target)` -> `return (c + target * (1.0 - b)) / (1.0 + target)` | to_y operation order (bit parity with the frozen function) | test_core_statistics_parity.py :: test_random_problems_match_the_frozen_functions_bit_for_bit |
| M08b | `core/selection/statistics.py` | `BET_CAP = 0.9` -> `BET_CAP = 0.95` | bet cap constant | test_core_statistics.py :: test_bet_cap_is_the_design_constant |
| M08c | `core/selection/statistics.py` | `mu_t = (0.5 + s1) / (t + 2)` -> `mu_t = (0.5 + s1) / (t + 1)` | variance update denominator | test_core_statistics.py :: test_tv1_all_ones_anchor_values |
| M09 | `core/vectors.py` | `bad = _first_bad(vectors, nan_ok=False)` -> `bad = _first_bad(vectors, nan_ok=True)` | validate_corpus finiteness (NaN) | test_core_vectors.py :: test_a_failing_validation_leaves_the_corpus_alone_too |
| M10a | `core/stages/state.py` | `{k: array_digest(v) if isinstance(v, np.ndarray) else v fo` -> `{k: array_digest(v) if isinstance(v, np.ndarray) else v fo` | the fitted-state digest covers every array (mean) | test_core_stage_state.py :: TestDigest :: test_digest_is_stable_across_fits_and_follows_the_data_where_declared |
| M10b | `core/stages/state.py` | `{k: array_digest(v) if isinstance(v, np.ndarray) else v fo` -> `{k: array_digest(v) if isinstance(v, np.ndarray) else v fo` | the fitted-state digest covers every array (components) | test_core_stage_state.py :: TestDigest :: test_digest_is_the_identity_digest_of_the_entries |
| M11 | `core/evaluate/collect.py` | `dropped_queries=tuple(ids[i] for i in np.flatnonzero(~keep` -> `dropped_queries=tuple(ids[i] for i in np.flatnonzero(keep)` | label-only query drop: the queries listed as dropped are those without a relevant document | test_core_collect_evidence.py :: TestUnlabelledQueries :: test_dropped_for_the_baseline_and_all_candidates_together |
| M12 | `core/identity.py` | `json.dumps(_plain(obj), sort_keys=True,` -> `json.dumps(_plain(obj), sort_keys=False,` | canonical JSON sorts keys | test_core_identity.py :: test_canonical_json_equals_json_dumps_for_json_native_objects |
| M13 | `core/selection/spaces.py` | `return digest([spec.to_dict() for spec in self.candidates]` -> `return digest([{"position": s.position, "pipeline": thaw_j` | the space identity includes the family labels | test_core_spaces.py :: test_the_default_space_hashes_to_the_registered_value |
| M14 | `core/selection/engine.py` | `order = order[:budget]` -> `order = order[-budget:]` | a query budget takes a prefix of the order | test_core_selection_engine.py :: test_a_budget_takes_a_prefix_of_the_given_order |
| M15 | `core/evaluate/evidence.py` | `"query_ids": strings_digest(self.query_ids),` -> `"query_ids": strings_digest(()),` | the evidence digest covers the query ids | test_core_evidence_types.py :: test_the_digest_is_sensitive_to_every_field_and_to_one_ulp |
| M16 | `core/identity.py` | `h.update(f"{dtype.str}\|{','.join(str(n) for n in a.shape)}` -> `h.update(f"\|{','.join(str(n) for n in a.shape)}\|".encode("` | array_digest covers the dtype | test_core_identity.py :: test_array_digest_scheme_is_dtype_shape_then_little_endian_bytes |
| M17 | `core/stages/accounting.py` | `return (self.code_bytes + self.codec_overhead_bytes) / max` -> `return (self.code_bytes + self.codec_overhead_bytes + self` | the size key excludes fitted-state bytes (D9) | test_core_accounting.py :: TestSizeKey :: test_selection_bytes_are_the_libraries_size_key |
| M18 | `core/selection/engine.py` | `if evidence.space_identity != space.identity:` -> `if False and evidence.space_identity != space.identity:` | evidence collected on another space is refused | test_core_selection_engine.py :: test_evidence_from_another_space_is_refused |
| M19 | `core/evaluate/metrics.py` | `ideal = discounts[: min(rel.size, k)].sum()` -> `ideal = discounts[:k].sum()` | nDCG ideal DCG over min(#relevant, k) | test_core_metrics_from_ranking.py :: test_parity_with_per_query_metrics_on_random_score_matrices |
| M20 | `core/selection/rules.py` | `selected = smallest(rejected, v.bytes_per_doc, v.vectors_p` -> `selected = rejected[0] if rejected else None` | the choice among certified candidates is the smallest, not the first | test_core_selection_rules.py :: test_selection_is_the_smallest_stored_size_among_the_certified |
| M21 | `core/stages/rotation.py` | `signs = np.where(np.diag(r) < 0, -1.0, 1.0)` -> `signs = np.ones(r.shape[0])` | the rotation fixes the LAPACK sign convention | test_core_rotation.py :: TestMatrix :: test_the_sign_convention_is_the_positive_diagonal_one |
| M22 | `core/evaluate/reference.py` | `canonical = {str(q): sorted(str(d) for d in docs) for q, d` -> `canonical = {str(q): [str(d) for d in docs] for q, docs in` | the label reference identity ignores the order of the listed documents | test_core_reference.py :: test_the_label_identity_does_not_depend_on_dictionary_or_collection_ordering |
| M24 | `core/retrieve/types.py` | `if np.isnan(scores[inside]).any():` -> `if False and np.isnan(scores[inside]).any():` | a Ranking refuses NaN scores | NOT CAUGHT by the layer's files nor by the other existing files (2669 passed, 1 skipped in 153.45s (0:02:33)); KILLED by test_integration_layers.py :: TestDtypesAreNeverChangedSilently :: test_a_ranking_refuses_nan_scores_by_name_at_any_width |
| M25 | `core/retrieve/checks.py` | `if k < 1:` -> `if k < 0:` | a result depth k >= 1 | test_core_retrieve.py :: TestRanking :: test_top_takes_a_prefix_and_never_extends |
| M26 | `core/retrieve/checks.py` | `if not finite_rows.all():` -> `if False:` | queries with NaN or infinity are refused | test_core_retrieve.py :: TestContract :: test_queries_with_nan_or_inf_are_refused_not_ranked_in_index_order |
| M27 | `core/retrieve/checks.py` | `if lo < 0 or hi >= n_docs:` -> `if lo < -1 or hi >= n_docs:` | -1 is not a candidate position | test_core_retrieve.py :: TestContract :: test_t20_rescore_rejects_bad_candidates |
| M28 | `core/retrieve/tiered.py` | `if m < n_docs and k > m:` -> `if k > m:` | k above the shortlist is clamped when the shortlist covers the corpus | test_core_retrieve_tiered.py :: TestTwoTierRetriever :: test_a_shortlist_of_the_whole_corpus_is_the_rerankers_exact_ranking |
| M29 | `core/selection/statistics.py` | `if float(b.mean()) == 0.0:` -> `if float(b.mean()) < 0.0:` | a zero mean baseline is refused | test_core_statistics.py :: test_zero_mean_baseline_raises_and_a_single_zero_does_not |
| M30 | `core/selection/statistics.py` | `bad = (arr < 0.0) \| (arr > 1.0)` -> `bad = (arr < 0.0) \| (arr > 2.0)` | metric values must lie in [0, 1] | test_core_statistics.py :: test_metric_values_outside_zero_one_raise_and_are_never_clipped |
| M31 | `core/selection/statistics.py` | `if isinstance(value, (bool, np.bool_)) or not isinstance(` -> `if False or not isinstance(` | a boolean is not a target or an error level | test_core_statistics.py :: test_target_outside_zero_one_or_not_one_real_number_raises |
| M32 | `core/selection/statistics.py` | `if n == 0:` -> `if n == -1:` | no queries is an error, not p = 1 | test_core_statistics.py :: test_zero_queries_raises_instead_of_returning_p_equal_one |
| M33 | `core/selection/rules.py` | `level = v.error / v.candidates.shape[0] # scripts/research` -> `level = v.error / len(dict.fromkeys(v.families))` | the union-bound level is error / K (candidates), not error / F | test_core_selection_rules.py :: test_bonferroni_tests_every_candidate_at_delta_over_k_with_no_chain_stop |
| M34 | `core/selection/decision.py` | `if self.certified != self.rule["certifying"]:` -> `if False:` | certified must equal the rule's certifying flag on SELECT | test_core_selection_engine.py :: test_an_inconsistent_decision_cannot_be_built |
| M35 | `core/evaluate/evidence.py` | `if len(ids) != n:` -> `if False:` | one query id per query | test_core_evidence_types.py :: test_structure_errors_raise_a_value_error |
| M36 | `core/evaluate/evidence.py` | `if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= s` -> `if idx.size and (int(idx.min()) < -10**9 or int(idx.max())` | Evidence.subset refuses negative positions | test_core_evidence_types.py :: test_subset_refuses_bad_indices_instead_of_wrapping |
| M37 | `core/evaluate/collect.py` | `if not traits.fit_uses_queries:` -> `if True:` | a stage fitted on queries is refused (assumption A3) | test_core_collect_evidence.py :: TestStagesAreFittedOnDocumentsOnly :: test_a_stage_that_reads_queries_is_refused_before_any_work |
| M38 | `core/evaluate/collect.py` | `if bad:` -> `if False:` | compress_selected raises when the fitted state differs from the evaluated one | test_core_collect_evidence.py :: TestCompressSelected :: test_a_wrong_expected_digest_raises |
| M39 | `core/stages/state.py` | `if expected is not None and actual != expected:` -> `if False:` | restore with expected_digest checks it | test_core_stage_state.py :: TestStateless :: test_no_state_means_empty_dict_none_and_zero_bytes |
| M40 | `core/identity.py` | `if not math.isfinite(value):` -> `if False:` | canonical JSON refuses NaN and infinity | test_core_identity.py :: test_non_finite_numbers_are_refused |
| M41 | `core/identity.py` | `if isinstance(items, (str, bytes)):` -> `if isinstance(items, bytes):` | strings_digest refuses a bare string | test_core_identity.py :: test_a_bare_string_is_not_a_sequence_of_strings |
| M42 | `core/stages/traits.py` | `if entry is not None and registered in (None, type(stage))` -> `if entry is not None:` | traits are not inherited by a subclass of a registered stage | test_core_stage_traits.py :: TestUnknownAndThirdParty :: test_a_subclass_does_not_inherit_the_traits_of_its_parent |
| M43 | `core/stages/state.py` | `if isinstance(value, (bool, np.bool_)) or not isinstance(` -> `if False or not isinstance(` | a boolean is not a number in a restored state | NOT CAUGHT by the layer's files nor by the other existing files (2730 passed, 1 skipped in 152.31s (0:02:32)); KILLED by test_integration_serialization.py :: TestFittedStateRestores :: test_a_boolean_is_not_a_number_in_a_restored_state |
| M44 | `core/selection/spaces.py` | `if not math.isfinite(value):` -> `if False:` | a candidate pipeline description cannot hold NaN | test_core_spaces.py :: test_candidate_spec_validates_its_fields |
| M45 | `core/selection/engine.py` | `if isinstance(value, (bool, np.bool_)) or not isinstance(v` -> `if not isinstance(value, (int, np.integer)):` | a boolean is not a query budget or a seed | test_core_selection_engine.py :: test_a_budget_that_cannot_be_used_raises |
| M46 | `core/retrieve/exact.py` | `if max_block_bytes < 1:` -> `if max_block_bytes < -1:` | max_block_bytes >= 1 | test_core_retrieve.py :: TestContract :: test_construction_is_validated |
| M47 | `core/retrieve/exact.py` | `if not all(isinstance(i, str) for i in self.ids):` -> `if False:` | document ids must be strings | test_core_retrieve.py :: TestContract :: test_ids_must_be_strings_and_match_the_documents |
| M48 | `backends/numpy_backend.py` | `if n_pages and self.index.decoded_nbytes == 0:` -> `if False:` | an index with no vectors scores every page -inf | test_backends_numpy.py :: TestTiesAndEmpties :: test_t17_an_all_empty_index_returns_minus_infinity_not_zero |
| M49 | `core/stages/accounting.py` | `_APPEND_ONLY = {"pca": frozenset({"mean"}), "adaptive_merg` -> `_APPEND_ONLY = {"pca": frozenset({"mean"}), "adaptive_merg` | the AdaptiveMerge mean is append-only state | test_core_accounting.py :: TestSizeKey :: test_the_parts_add_up |
| M50 | `core/vectors.py` | `if not allow_empty_docs and (steps == 0).any():` -> `if False and (steps == 0).any():` | allow_empty_docs=False refuses a document with no vectors | test_core_vectors.py :: test_corpora_without_vectors_pass_and_have_an_identity |
| M51 | `core/vectors.py` | `(("weights", False), ("importance", False), ("positions", ` -> `(("weights", True), ("importance", False), ("positions", T` | weights must be finite | test_core_vectors.py :: test_each_violation_raises_a_contract_error_that_names_the_offender |

**The survivors and the tests that kill them.**

* **M01b** (`core/selection/rules.py`, `BonferroniBetting.decide`: `np.flatnonzero(p <= level)`
  changed to `p < level`). No existing test places a p-value exactly on `error / K` for the
  union-bound rule; the boundary of the chain rule is covered (TV7 in `test_core_statistics.py`),
  the union rule's is not. Killed by
  `properties::TestTheLevelIsInclusiveAndExactlyTheErrorOverTheChains::test_the_union_bound_rule_rejects_exactly_at_error_over_the_number_of_candidates`
  (and `test_select_applies_the_same_comparison_and_records_the_level[union]`): with 8 candidates
  `error = 8 p` gives `error / 8 == p` exactly in floating point, the candidates must be
  certified, and one ulp below `error` must certify none. The same tests also kill M01a, M04a,
  M04b and M33 independently of the existing tests.
* **M24** (`core/retrieve/types.py`, `_check_order`: the NaN check disabled). No existing test
  constructs a `Ranking` with a NaN score. The mutant is partly equivalent (with two or more
  columns the descending-order check also raises, with another message), but a one-column
  `Ranking(ids=[["a"]], scores=[[nan]])` is accepted by it, and with it an `ExactRetriever` whose
  every score is NaN would return a NaN "ranking" at depth one. Killed by
  `layers::TestDtypesAreNeverChangedSilently::test_a_ranking_refuses_nan_scores_by_name_at_any_width`
  (four shapes, the message must name NaN).
* **M43** (`core/stages/state.py`, `_coerce`: a boolean accepted as a number in a restored fitted
  state, so `seed=True` is read as 1). No existing test restores a state holding a boolean. Killed
  by `serialization::TestFittedStateRestores::test_a_boolean_is_not_a_number_in_a_restored_state`.

**Limits of this check.** One mutation per run; the list is mine, so the result is only as broad
as the list; equivalent mutants are possible (M24 is partly one: with two or more columns the
descending-order check of `Ranking` also raises, with another message). The scratch copy of `src/`
was taken while work on `src/` continued: at the end three files differed from the tree only in
line endings (written by the harness) and one comment, so the 62 layer-phase results hold for the
final tree, and each result names a failing test that belongs to the mutated behaviour. The first
run of M43 against the other existing files failed in `test_api_certify.py` for a reason unrelated
to the mutation (that file had been extended after the copy was taken); the copy was refreshed
from `src/` (no difference), the unmutated copy was checked to pass that file (110 tests), and the
run was repeated: it survived. The runs of the survivors against the five integration files were
made on the first copy; every test that ran before the killing test passed on it.

## 6. Defects found

Each was kept out of the suite with `xfail(strict=True)`; the reason string starts with `DEFECT`.
When one is fixed its test turns into an unexpected pass, which fails the suite and says so.

**Status after the fixes.** D1, D2 (the `CompressedRetriever` and `TwoTierRetriever` cases) and D3
were fixed in `core/retrieve` and `core/evaluate/reference.py`, their `xfail` entries were removed
from `KNOWN_DEFECTS` and the tests pass. D4 and the `Pipeline.transform_queries` wrong-width case
are in the byte-pinned `compose.py`: they are not fixed, and their four entries stay in
`KNOWN_DEFECTS` as expected failures.

**The four strict expected failures are known defects that are preserved on purpose.** `compose.py`
is one of the files whose content hash the research harness pins (`scripts/research/x_common.py`);
editing it to fix them would break that pin. A normal run of the suite therefore reports four
`xfailed` and no failure; a statement that "all tests pass" without that count is incomplete.
Because they are `xfail(strict=True)`, fixing one makes it an unexpected pass, which fails the suite.
The four, all in `tests/test_integration_layers.py::TestBadVectorInputs`:

* `test_queries[Pipeline.transform_queries|other_dimension]` (a projector fails inside NumPy's matmul
  with a message that does not name the queries);
* `test_queries[Pipeline.transform_queries|raw_array]`, `test_documents[Pipeline.fit|raw_array]` and
  `test_documents[Pipeline.compress|raw_array]` (a raw array raises `AttributeError`, not
  `TypeError` or `ValueError`).

| id | status | how |
|---|---|---|
| D1 | fixed | `ExactRetriever` and `CompressedRetriever` scan a float store once, at construction, in blocks and in the stored dtype, and refuse NaN or infinity with a message naming the first document; integer code stores are not scanned. Tests: `test_core_retrieve.py::TestDocumentsAreScannedOnce` and the three `ExactRetriever(corpus)` rows of the bad-input table |
| D2 | fixed for `CompressedRetriever` and `TwoTierRetriever`; not fixable for `Pipeline.transform_queries` (pinned) | the query dimension is compared with the width the pipeline's first dimension stage was fitted on when its fitted state shows it, else one projected query vector is compared with the store; the `ValueError` names the queries and both dimensions. Tests: `test_core_retrieve.py::TestQueryWidthThroughAPipeline` and the three rows of the bad-input table |
| D3 | fixed | `RankingReference.relevant` refuses NaN and a row without a finite score, naming the first query. Tests: `test_core_reference.py`, `test_integration_layers.py::TestDtypesAreNeverChangedSilently::test_the_agreement_reference_refuses_nan_scores` |
| D4 | not fixed (pinned `compose.py`) | reported only |

The table below is the original finding.

| id | where | what | severity | tests |
|---|---|---|---|---|
| D1 | `core/retrieve/exact.py` (`ExactRetriever._setup`), `checks.py` | `ExactRetriever` validates queries but not documents. A NaN in a document gives it a NaN score: `search(k < n_docs)` never returns it and says nothing (NaN sorts last; `k >= n_docs` raises "scores contain NaN"), `score()` returns the NaN. An infinite component gives the document an infinite score and it ranks first (`inf` observed at rank 1). `certify` and `collect_evidence` validate their inputs, so the exposure is a retriever built directly over a corpus that was never validated. | moderate: a wrong ranking without a message | `layers::TestBadVectorInputs::test_documents[ExactRetriever(corpus)\|nan]`, `[...\|inf]`, `[...\|negative_inf]` |
| D2 | `core/retrieve/exact.py` (`_query_dim = None` whenever a pipeline is given) | `CompressedRetriever(store, pipeline).search(queries)` and a `TwoTierRetriever` with such a first stage do not compare the query dimension with the store; wrong-width queries fail inside NumPy's matmul ("matmul: Input operand 1 has a mismatch in its core dimension 0 ...") with a message that does not name the queries, although `checks.py` promises a message instead of a failure inside a matrix product. The pinned `Pipeline.transform_queries` does the same for a projector. | minor | `layers::TestBadVectorInputs::test_queries[CompressedRetriever.search\|other_dimension]`, `[CompressedRetriever(projected).search\|...]`, `[TwoTierRetriever.search\|...]`, `[Pipeline.transform_queries\|other_dimension]` |
| D3 | `core/evaluate/reference.py` (`RankingReference.relevant`) | the baseline score matrix is not checked for NaN; a NaN matrix gives an index-order "reference" (the first documents) without a message. Reachable only by calling the reference directly with a hand-made matrix. | minor | `layers::TestDtypesAreNeverChangedSilently::test_the_agreement_reference_refuses_nan_scores` |
| D4 | pinned `compose.py` (reached through the `core.pipeline` facade) | `Pipeline.fit`, `compress` and `transform_queries` given a raw array raise `AttributeError: 'numpy.ndarray' object has no attribute 'vectors'`, not a `TypeError`. Not fixable without changing a byte-pinned file; reported only. | minor, pinned | `layers::TestBadVectorInputs::test_documents[Pipeline.fit\|raw_array]`, `[Pipeline.compress\|raw_array]`, `test_queries[Pipeline.transform_queries\|raw_array]` |

No defect was found in the selection arithmetic, the decision record, the certificate, the bundle
or the verification: in every scenario the per-query metrics of the deployed artifact equal the
recorded ones exactly, and every tampering of `TestTamperingIsNoticed` is reported by the check
that is meant to report it.

### Observations that are not defects (documented behaviour, recorded so nobody is surprised)

* `validate_corpus` accepts a corpus with no documents (and `allow_empty_docs=False` cannot flag it:
  it only looks at documents with no vectors); `corpus_identity` hashes NaN bytes as stored.
  `certify` and `collect_evidence` do refuse an empty corpus and an empty query set.
* A retriever given a query set with no queries returns an empty `Ranking` (documented).
* `validate_evidence_arrays` widens integer, float16 and float32 input to float64 exactly
  (documented), so "an integer array is refused" is true of corpora, `Evidence` and `Ranking`, not
  of the rule's own arrays. `Evidence` refuses anything but float64.
* A float64 corpus is narrowed to float32 by the scorers (documented in `exact.py`) and by the
  pinned `Pipeline.compress`. `certify` records the dtype it was given (`corpus.dtype` in the
  certificate and `native_dtype_before` in the compressed corpus statistics) and does not refuse it.
* `strings_digest([1, 2])` equals `strings_digest(["1", "2"])` (items go through `str`); corpus ids
  are always strings, so no corpus is affected.
* `metrics_from_ranking` accepts relevant arrays of floats (they compare equal to integers);
  `collect_evidence` refuses them.
* `Pipeline.compress` of a corpus with no vectors succeeds for a stateless pipeline (pinned).

## 7. What is NOT tested

* **Real encoder output.** All vectors are seeded Gaussian draws (unit norm, with a controllable
  anisotropy), queries are noisy cuts of documents. No ColPali, ColQwen, ColBERT or Nemotron
  vectors, no real qrels and no real query distribution. None of the assumptions and conditions
  of the rule's statement can be checked on synthetic data, and no coverage or power claim follows from the simulations.
* **Cross-platform bit-reproducibility.** Determinism is shown on this machine, in two processes,
  with one NumPy, one BLAS and one thread configuration; digests are compared bitwise. On another
  CPU, BLAS or NumPy the fitted-state digests (PCA, Lloyd-2) may differ and `compress_selected`
  would then refuse to deploy (by design).
* **Big-endian hosts.** `array_digest` canonicalises to little-endian; it is tested only by
  converting arrays to big-endian dtypes on a little-endian host.
* **Python versions other than 3.11.9, NumPy other than 2.4.6** (the declared minimum 1.26 is not
  exercised), SciPy other than 1.17.1, and the no-SciPy path of the default candidate space (not
  exercised by the integration tests).
* **Qdrant** in any mode (the Qdrant tests are the existing `test_backends_qdrant.py`), a Qdrant
  server, and approximate search.
* **GPUs and `OPTIVISION_SCORE_DEVICE`** (removed from the environment of the determinism test).
* **Corpora beyond a few thousand vectors**: the largest here has about 2,500 vectors of 96 or
  128 dimensions; nothing about memory, blocking (`max_block_bytes`), large `k` or timing at scale.
* **Concurrency**: no test runs two calls at once (`TieredRetriever.last_timing` is documented as
  instance state), and no test runs on a read-only or network file system. All tests ran on Windows.
* **The CLI** (`cli_certify.py`), the legacy `optimize()` path, the adapters and the `ext` and
  `legacy` namespaces are not exercised by these files (they have their own tests).
* **Power.** Nothing here measures how often the rule certifies; abstention is tested as a correct
  outcome, not as a rate.
* **Signed certificates** beyond sign, verify, wrong key and no key; `strict_code`.
* **Several `(target, error)` looks at one sample.** `select` takes one pair per call; the tests
  check that nesting properties hold when it is called repeatedly, not that doing so is valid (it
  is outside the statement of the rule).

## 8. Exact commands and counts

Run from the worktree root with the Python of the research checkout's venv (Python 3.11.9,
numpy 2.4.6, scipy 1.17.1, pytest 9.1.1); `PYTHONPATH=src` is required.

```
PYTHONPATH=src ../optivision-rag-v2/.venv/Scripts/python -m pytest \
    tests/test_integration_layers.py tests/test_integration_properties.py \
    tests/test_integration_determinism.py tests/test_integration_serialization.py \
    tests/test_integration_end_to_end.py -q
```

Result when written: `1359 passed, 11 xfailed in 128.15s (0:02:08)`. After D1 to D3 were fixed,
`test_integration_layers.py` alone gives `548 passed, 4 xfailed`; the other four files were not
re-run for this note.

```
PYTHONPATH=src ../optivision-rag-v2/.venv/Scripts/python -m pytest tests/test_core_layering.py tests/test_x_harness.py -q
```

Result: `65 passed in 121.35s (0:02:01)`

```
../optivision-rag-v2/.venv/Scripts/python -m ruff check tests/test_integration_layers.py tests/test_integration_properties.py tests/test_integration_determinism.py tests/test_integration_serialization.py tests/test_integration_end_to_end.py
```

Result: `All checks passed!`

Parity (section 4):

```
PYTHONPATH=src ../optivision-rag-v2/.venv/Scripts/python -m pytest tests/test_core_statistics_parity.py -q
OPTIVISION_FULL_PARITY=1 PYTHONPATH=src ../optivision-rag-v2/.venv/Scripts/python -m pytest tests/test_core_statistics_parity.py -q
```

Mutation check (section 5): copy `src/` to a scratch directory, apply one change of the table to
the copy, then run, for example,
`PYTHONPATH=<scratch>/src ../optivision-rag-v2/.venv/Scripts/python -m pytest tests/test_core_statistics.py tests/test_core_selection_rules.py tests/test_core_selection_engine.py tests/test_core_statistics_parity.py -q -x`
and read the first failing test. Restore the file before the next change. The files run for each
mutation were those of its layer; the survivors were then run against all the other existing
new-layer files and against the five integration files.

The full suite was not run by this work (the lead runs it). The tests added here edit no existing
file and write nothing outside `tmp_path`.
