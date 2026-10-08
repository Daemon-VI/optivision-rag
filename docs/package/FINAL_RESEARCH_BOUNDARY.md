# The boundary between the package and the research

Status: 2026-10-08, branch `package-universal`. This document states what the OptiVision
package supports, what its research record does and does not show, and what the package does
not guarantee. Where it repeats a research result it copies it from
`reports/research/X_RESULTS.md` and names the experiment; it adds no analysis.

Evidence labels, as in the research reports: **MEASURED** (computed once on a whole pool or
corpus), **SIMULATED** (resamples of one measured pool; *pool-supported* when the budget does
not exceed the pool), **PROJECTED** (arithmetic on measured quantities, Q8), **ENGINEERING** (a
property of the code checked by tests on synthetic data; not a research result).

Where to find each question:

| question | section |
|---|---|
| What the package provides | 1 (and 2, what "universal" means) |
| What the research established | 4, 5, 6 (X1, X2, X3) and 11 (earlier work) |
| What is model dependent | 3 |
| What is basis dependent | 7 (the failure of H5) and 12 (the consolidated list) |
| What certification means | 13 |
| What certification does not mean | 9 and 14 |
| What XH would test | 8 |
| What has not been tested | 15 |
| How the package and the research record are kept apart | 10 |

## 1. What the package provides (what OptiVision currently supports)

For any late-interaction encoder whose output is one `[n_vectors, dim]` float array per
document and per query, scored by MaxSim (L2-normalised vectors are the only regime the research
record studied):

- a vector contract with validation and content identities;
- composable compression stages (token reducers, projections, codecs), each with declared
  structural traits, including its dependence on the coordinate basis;
- a frozen, hashed candidate list;
- per-query evidence against a stated reference (relevance labels, or agreement with the
  uncompressed ranking);
- a selection rule that either selects a candidate or abstains;
- a certificate that records the decision and can be verified, and an artifact bundle that
  stores codes and fitted state;
- a retrieval seam (queries in, ranked ids out) and a backend interface over the NumPy and
  Qdrant indexes;
- the v0.3.0 API unchanged (`optimize`, `calibrate`, `Pipeline`, the stages, `OptiVisionRAG`).

All of the above is ENGINEERING. It is tested on synthetic data and against the frozen research
code for numerical equality. None of it was evaluated by a registered experiment as a package,
and no real encoder output was passed through the new package layer (section 15).

## 2. What "universal" means

The interface, the composition of pipelines, the generation of candidates, the evaluation, the
decision procedure, the certificate, the artifact and the backend seam are the same for every
encoder. That is all it means.

It does **not** mean that one compression configuration, one operator or one candidate list
works on every encoder. No such claim is made, and the research record contradicts the strong
form of it (section 6).

## 3. What is model dependent

- The retention any pipeline achieves. It depends on the encoder, on the coordinate basis its
  vectors happen to be in, on the corpus and on the queries.
- Whether the selection rule selects anything, and how much compression it selects. It depends
  on all of the above and on the query budget, the target and the reference.
- The default candidate list. It is frozen and the same for every encoder (it does not depend on
  the vector dimension), but it was shaped by results on the development encoders. It is
  model-blind in what the procedure inspects, not neutral in how it was built.
- Encoding itself: adapters, patch grids, page images and pixel-space pruning are specific to
  model families and live outside the core.

What does not depend on the encoder: the selection rule's error statement, given its
assumptions (section 9), because the rule looks only at per-query arrays.

## 4. What the research established (1): X1

Scope: the seven development stores (three visual encoders on DocVQA and InfoVQA, one text
encoder on SciFact), one query pool per corpus, the registered list of 40 pipelines.

- **H4a supported.** Agreement with the uncompressed ranking behaved as a conservative proxy
  for labelled retention: CP1 held on all 21 (store, threshold) pairs (MEASURED); CP2 held on
  150 cells (SIMULATED).
- **H4b supported.** The two constructs are not equivalent: the free oracles differ by a factor
  of 2 or more on 21 of 21 pairs (MEASURED).
- **H4c supported.** Agreement was usable with the pool: the non-trivial deployment rate at
  T_A = 0.95, delta = 0.05 at the pool-capped budget was 1.000 on the six visual stores and
  0.999 on SciFact (SIMULATED, pool-supported).
- The registered negative-result sentence N10 applies: agreement is materially stricter than
  labelled retention; using it cost at least a factor of 2 in compression (in 26 of 37
  pool-supported cells where the labels arm deploys non-trivially at a rate of at least 0.90,
  the median certified compression under agreement was at most half that under labels;
  `reports/research/X_RESULTS.md`, sections 3 and 5).

What X1 did not show: that agreement is retrieval quality. It is not, and there is no
statistical statement connecting the two constructs. The CP1/CP2 conditions are empirical
observations on these stores.

## 5. What the research established (2): X2

- **H3, primary, supported within pool-supported scope.** With every query its pool can supply,
  the procedure deployed non-trivial compression at T = 0.97, delta = 0.05 on all seven stores
  at a rate of at least 0.90; the lowest was 0.9175 (1,835 of 2,000 resamples; 0.917 [0.905,
  0.929] in the registered reports) on SciFact at n = 300 (SIMULATED, pool-supported). That
  value is close to the 0.90 threshold. By pool the lowest rates were 0.998 (DocVQA), 1.000
  (InfoVQA) and 0.9175 (SciFact).
- **The three stricter secondary cells fail.** T = 0.99 non-trivial fails on all seven stores;
  T = 0.97 substantial fails on the three DocVQA stores and SciFact; T = 0.99 substantial fails
  on all seven. "Fails" means the registered criterion, a deployment rate of at least 0.90,
  was not met; at T = 0.99 it was not met on any store at pool-sized budgets (at most 0.010 for
  non-trivial deployment, SIMULATED, pool-supported). The internal statistical review
  (`STATISTICAL_REVIEW.md`) notes that this follows from the arithmetic of the sample size and
  is not a finding about the encoders.
- The extension beyond the pools holds in simulation only; a result at 900, 1,800 or 3,600
  queries is a statement about redrawing the pool's own queries, not about that many
  independent queries. At T = 0.97 the registered extension verdict ("holds in simulation")
  rests on pool-supported budgets only (225, 300 or 450).
- The validity check covered 2,800 cells with no confirmed failure. These are resamples of one
  pool per corpus: they check the implementation against the derivation; they do not replace it.

## 6. What the research established (3): X3

- **H5 fails.** 13 of 16 primary (dataset, transformation) pairs hold; 3 do not.
- The signed coordinate permutation and the reordering of vectors changed nothing material on
  any dataset. The three failing pairs are rotations: ColPali DocVQA under one rotation, and
  SciFact under both.
- The largest change in labelled retention on a primary pair is 0.043 (MEASURED on the
  transformed vectors).
- On SciFact at T = 0.97, delta = 0.05, n = 300, the non-trivial deployment rate of the procedure was 0.9175
  in the native basis and 0.0995 and 0.1415 under the two rotations, with median certified
  compression 6.6x against 2.0x (SIMULATED, pool-supported; `reports/research/X3/x3_scifact.json`).
- The direction is not uniform. On the extension dataset ColPali InfoVQA the procedure
  outcome was material under both rotations, upward in some cells and downward in others
  (`reports/research/X3/x3_colpali_infovqa.json`).
- In the extension set, reported separately, 8 of 12 pairs hold; the four that do not are
  rotations.

## 7. What the failure of H5 means

H5, as registered: each of four fixed maps (one signed coordinate permutation, two rotations,
one reordering of vectors), applied to each primary dataset, leaves the labelled retention of
every candidate and the outcome of the default procedure within registered tolerances. It was
supported only if all 16 pairs held, and it is not a claim about the coordinate basis in
general.

Within the registration: on these datasets, the measured result of some sign-code (`binary`)
and 4-bit (`int4`) pipelines depends on the coordinate basis under the two registered
rotations, beyond the registered tolerance. It is not a statement about other maps, other
encoders or other candidates.

For the package:

- The selection procedure can be model-blind in what it is allowed to inspect while the
  compression it yields depends on the coordinate basis. Both are true at once.
- A certificate is about the vectors as given. It does not carry over to a rotated copy of the
  same index. The certificate records a digest of the corpus (`corpus_identity`), and `verify`
  compares it when the live corpus is passed (section 10).
- No stage is assumed to behave the same in another basis. Each stage declares the basis
  changes that leave its scores unchanged, pipelines carry `basis_sensitive`, and a certificate
  that selected a candidate records it for the deployed pipeline (section 12).
- The rule's statement is unaffected: whatever basis the vectors are in, the rule tests the
  pipeline on those vectors. What changes with the basis is the yield, not the validity.
- A rotation stage exists as a diagnostic. X3 found that rotation changes what is certified,
  downward on SciFact and in both directions on one extension dataset. Nothing establishes
  that a rotation helps or harms in general; it has to be measured like any other candidate.

## 8. What XH would test

XH is the held-out encoder protocol designed in section 13 of the preregistration
(`reports/research/NEXT_PHASE_PREREGISTRATION.md`). The preregistration registers the rules by
which XH will be registered, not XH itself. It has not been registered and has not been run.

- **H1**: whether the frozen procedure deploys non-trivial compression while keeping its error
  level on encoders that took no part in its design. Unit: a held-out cell, at the primary
  setting (labels, T = 0.95, delta = 0.05, the pool-capped budget). A cell succeeds if the
  non-trivial deployment rate is at least 0.90 and the miss rate is consistent with delta. H1 is
  supported if at least 0.80 of the valid cells succeed, not supported below 0.80, refuted for
  the tested set below 0.50, and not assessable with fewer than 6 valid cells, 4 held-out
  encoders, or 2 encoders outside the development families.
- **H2**: the price of model-blindness, the compression an encoder-specific tuned procedure
  certifies relative to the model-blind one on the same queries. Its measure is PoMB, the median
  over resamples of log2 of that ratio. H2a asks whether PoMB is defined in at least 80% of valid
  cells. H2b asks whether, over held-out encoders, the median PoMB is at most 1 (a factor of 2)
  and the maximum at most 2 (a factor of 4).

Until XH is registered and run under its own rules, transfer to held-out encoders has **not**
been demonstrated. X1-X3 are development analyses: they reuse the encoders that shaped the
candidate list and say nothing about transfer. An encoder becomes ineligible for XH once any
of its vectors has been compressed or scored by this project before the registration tag; this
package work encoded no model and scored no real encoder output.

## 9. What the package does not guarantee

1. That any pipeline, codec or candidate list keeps retrieval quality on an encoder, a corpus,
   a corpus size, a modality or a coordinate basis that was not measured.
2. Anything for the shipped `optimize()` / `calibrate()` rule. It is an empirical rule with a
   held-out check and no post-selection statement.
3. Anything beyond the certifying rule's own statement, which is: the probability that a
   compressed candidate is selected and its retention under the query distribution is below the
   target is at most the error level, over the draw of the calibration queries. The derivation
   holds only if the assumptions and conditions that the statement lists hold (A1-A6 and
   C1-C4); they are recorded in every certificate and set out in
   `guide/SELECTION_AND_CERTIFICATION.md`, and nothing is claimed when one fails. The statement
   is a joint probability. It is not a probability conditional on a selection, not a statement
   about the retention measured on the sample, not a statement about a finite batch of future
   queries, and not a statement about power. The rule applies published tools; the derivation
   that combines them is the project's own and has not been externally reviewed. Only the
   library's own rules are ever certified: a rule counts as the library's own only if it is
   exactly one of the two built-in classes (`is_builtin_rule` compares the exact type), and a
   rule written by anyone else is recorded as not certified, with a statement saying the library
   has not examined it.
4. That the rule selects anything. It may abstain, and at strict targets with the query
   budgets studied it did.
5. That agreement with the uncompressed ranking is retrieval quality.
6. That a certificate is secure against forgery without a signature key (see
   `CERTIFICATE_SPEC.md`).
7. Speed. Scoring decodes codes to float32; native compressed scoring was slower in NumPy and
   torch on one laptop CPU (Q5/Q6), compiled kernels were not tested, and nothing general is
   claimed either way.
8. Scale. The million-page figures of Q8 are projections (PROJECTED), and quality at scale is
   unmeasured.

## 10. How the package and the research record are kept apart

- The research record (`reports/research/**`, `scripts/research/**`, the registered inputs,
  the preregistration and its amendment, `paper/**`, `reports/universal/**`) is not modified
  by package work. The package work lives on its own branch; the research branch is unchanged.
- Eleven library files that the research harness pins by content hash are byte-identical to
  the registered ones (`tests/test_x_harness.py::test_frozen_code_is_the_registered_code`
  checks the pinned hashes in `scripts/research/x_common.py`). New package code is in new files.
- The package does not import research scripts
  (`tests/test_core_layering.py::test_package_does_not_import_research_scripts`). Tests may read
  the frozen research functions and stored selections, read-only, to check that the ported
  selection rule is numerically the same procedure (`tests/test_core_statistics_parity.py`).
- A certificate binds a decision to the corpus by a digest of the vectors, their dtype, ids and
  order (`corpus_identity` in `core/vectors.py`, recorded when the certificate is made and
  compared by `verify` as `binding.corpus` when the live corpus is passed; tests:
  `test_a_different_corpus_fails_the_corpus_binding` and
  `test_a_corpus_in_another_dtype_fails_the_corpus_binding` in
  `tests/test_core_certificate_tamper.py`). Without the live corpus, that check is skipped.
- Only the library's own rules are certified: `is_builtin_rule` and `rule_record`
  (`core/selection/rules.py`) decide it, `statement_for` (`core/artifact/record.py`) writes the
  statement the certificate carries (the rule's `STATEMENT` for a built-in rule, a fixed
  "not examined" text otherwise). Tests: `test_only_the_exact_types_of_the_two_library_rules_are_built_in`,
  `test_a_rule_that_declares_certifying_is_not_certified_by_the_library` and
  `test_a_rule_of_someone_elses_is_not_certified_through_certify_either`.
- The registered runs recorded the library tree `0b9523bd10f43941a9704cbf6fae1db1804880ee`. The
  package branch has a different tree, because it adds files. X1-X3 belong to the commits
  they were run at. A future registered experiment must record the tree it runs on; if it is to
  run on the package branch, that needs a new dated amendment, not a silent re-pin.
- X1, X2 and X3 were not re-run. XH was not run. No hypothesis, protocol or historical result
  was changed.

## 11. What the research established (4): earlier work

Before X1-X3 the project ran the R-series (`docs/UNIVERSAL.md`) and the Q1-Q10 investigation
(`reports/research/`; the evidence matrix is in `Q10/REPORT.md`, section 5). Each finding is
bounded by the models, corpora and hardware it was measured on: four visual encoders and one text
encoder, three corpora, one pool of queries each, one laptop for timing. What the record supports,
with its scope and not more:

- The shipped `optimize()` met its target in 355 of 360 (128-d) and 119 of 120 (2,560-d) held-out
  choices (Q10, row 1; empirical, development pools, 20 splits). This concerns the empirical
  rule, not the certifying rule (section 9, item 2).
- The certifying rule's error level held in every check made: E1.1 on synthetic and development
  data, and E1.2 on held-back datasets, where all 27 cells were consistent with delta = 0.05
  (`reports/research/E1/RESULTS.md`; SIMULATED, one pool per dataset, queries drawn with
  replacement, fixed corpus).
- Merging with relative per-page budgets beat pruning; dimension reduction was
  model-specific (harmful at 128-d, nearly free to d/8 at 2,560-d); per-vector int8 was
  near-lossless on every encoder measured; plain int4 and binary codes were model-specific;
  compression axes did not multiply (Q10, rows 5-10).
- DocVQA got less compression mainly because its quality was harder to certify, not because it
  compressed worse (Q10, row 3; supported with residual uncertainty).
- Scan cost followed the number of vectors; native compressed-code scoring was slower on that
  stack (Q10, rows 4 and 12).
- Retention fell when distractor pages were added (`docs/UNIVERSAL.md`, R10; MEASURED, 5 draws).

These are findings about those models and corpora. They are not evidence for transfer to other
encoders (section 8), and Q10 states that they do not establish priority or novelty.

## 12. What is basis dependent, and what is not

A consolidated list. Its source is the stage traits (`optivision.core.stages.traits`,
`guide/BASIS_DEPENDENCE.md`, section 4) and X3. A declared invariance is ENGINEERING: a statement
about code, in exact arithmetic, for data that meet the stage's listed conditions (for example no
exact ties at a PCA cut, no coordinate exactly zero for a sign code); the tests are in
`tests/test_core_stage_traits.py`, and nothing checks the conditions against your data. It says
nothing about retrieval quality.

Declared invariance of each stage:

| declared level | stages |
|---|---|
| unchanged under any orthogonal change of basis (`orthogonal`) | `adaptive_merge`, `hierarchical_merge`, `redundancy`, `random_prune`, `spatial`, `pca` (canonical up to sign), `float32`, `rotation` |
| unchanged only under a signed permutation of axes (`signed_permutation`) | `binary`, `int8`, `int4`, `float16` |
| not even that (`none`) | `truncate`, `random_projection`, `lloyd2`, `project` (it takes the traits of its method) |

A pipeline is `basis_sensitive` unless every stage is declared `orthogonal`. A seeded random map
earlier in the pipeline puts the later `signed_permutation` stages at `none`. PCA is recorded and is
not credited as invariance. All 40 candidates of the default list are `basis_sensitive` (39 end in
`int8`, `int4` or `binary`; the other is `float16`).

What X3 measured about it (sections 6 and 7): the signed permutation and the reordering of vectors
changed nothing material on any dataset; the two rotations changed the measured result of some
`binary` and `int4` pipelines and the procedure's outcome, in the primary set on ColPali DocVQA and
SciFact, and in the extension set on ColPali InfoVQA and ColQwen2 DocVQA. Which stages are
invariant in code and which pipelines changed in X3 are different questions; the second is
measured only for the seven datasets and four maps.

What a certificate is bound to, and so does not carry over from: the corpus (vectors, dtype, ids,
order), the queries, the reference, the candidate list, the rule, the deployed pipeline with the
digests of its fitted state and its codes, and the code that produced it. A re-encoded, rotated,
whitened, rescaled, extended or truncated copy is another corpus, and the certificate says nothing
about it. A certificate for a selection records `basis_sensitive` and the invariance level of the
deployed pipeline; `basis_sensitive=False` is conditional on the listed conditions and is not
"quality is kept".

What is not basis dependent: the interface and its validation; the rule's error statement given
its assumptions (it sees only per-query arrays); the exact float MaxSim scores (unchanged by an
orthogonal map, up to rounding).

## 13. What certification means

In plain terms:

- The rule tested, for each candidate, the claim "this compression keeps at most the target share
  (say 97 percent) of the uncompressed index's quality, averaged over the kind of queries you
  sampled". A candidate is certified when that claim was rejected at the stated error level.
- If you ran the procedure again and again on fresh, independent samples of queries from the same
  source, in at most the share `error` of those runs (5 percent for error 0.05) would it both
  select a compressed index and select one whose true retention is below the target. Runs that
  abstain are not counted as wrong.
- This holds for this corpus, this candidate list in its fixed order, this one target and error
  level, used once on one sample. It holds if the assumptions the certificate lists hold: queries
  that are independent draws from the distribution you will serve, a corpus that does not change, a
  list fixed before the queries are seen, a metric in [0, 1] with a positive baseline, a query
  order unrelated to the outcomes, a reference taken as given, one look per sample, and the
  deployed index being the evaluated one. The software checks some of these and cannot check
  others (`guide/SELECTION_AND_CERTIFICATION.md`, section 6).
- Abstaining is a correct outcome. The statement also holds if the rule never selects anything.
- Under the agreement construct "retention" means agreement with the uncompressed ranking.

Verification has three levels: integrity (the certificate alone), replay (the decision recomputed
from stored evidence, for the library's own rules) and binding (the live objects match the recorded
identities). `ok` means no check failed; it does not make the certificate true.

## 14. What certification does not mean

- It does not mean the chosen index keeps retrieval quality. Under labels it is a statement about
  an average over a query distribution; under agreement it is about fidelity to the uncompressed
  ranking, mistakes included, and nothing connects that to relevance.
- It does not mean this particular selection is right. The statement is about how often the
  procedure goes wrong; it is a joint probability, not "given a selection, the chance it is below
  target". A rule that rarely selects can satisfy it while most of its selections are below the
  target.
- It does not describe the retention measured on your sample (optimistically biased by the
  selection; recorded as not certified), the next batch of queries, or power.
- It does not carry over to another encoder, corpus, corpus size, modality, coordinate basis or
  re-encoding, and not to a query distribution unlike your sample.
- It is not a guarantee if an assumption fails; it does not cover trying several targets, lists,
  orders or seeds on one sample.
- It is not a proof of correctness of the method: the derivation is the project's own, an
  application of published tools, and has not been externally reviewed. The internal second
  reading (`STATISTICAL_REVIEW.md`) is not external review.
- It is not a security token. Without a signing key the digest detects accidental and naive edits
  only; an HMAC key closes that gap and gives no non-repudiation (`CERTIFICATE_SPEC.md`).
- A rule written by anyone else is never certified by the library, whatever it declares.
- It does not mean the compressed index is faster. The package makes no speed claim.

## 15. What has not been tested

Research record (nothing below was registered or run):

- Held-out encoders (XH, section 8). The X1-X3 encoders shaped the candidate list.
- Any similarity other than MaxSim over L2-normalised vectors. Unnormalised vectors are accepted
  by the contract, recorded in the certificate with a caveat, and not studied.
- Single-vector (dense) encoders. The scope is late-interaction retrieval.
- Corpus sizes beyond 500 pages (visual) and 5,183 abstracts (text). Quality at scale is
  unmeasured; the million-page figures are PROJECTED.
- Other modalities than page images and text.
- Encoders other than those listed in section 4. The 4,096-d `nvidia/nemotron-colembed-vl-8b-v2`
  has a loader and a pinned revision and was not benchmarked or loaded by this work
  (`docs/UNIVERSAL.md`, R12). The 2,560-d 4B encoder was measured only with the empirical rule of
  `calibrate()`, not the selection rule.
- Other maps than the four registered ones, and bases other than the two registered rotations.
- Learned or trained compressors, approximate search, corpus growth after certification.

The package layer:

- Real encoder output through the new package layer. It was tested on synthetic data and, for the
  selection arithmetic, for numerical equality with the frozen research code (including the stored
  selections of the E1 runs). `certify`, the certificate and the artifact were not run on real
  encoder vectors.
- Platforms. The full test suite ran on Python 3.11 on one Windows laptop. The tests of the new
  layer ran against the installed wheel on Python 3.10 (NumPy 1.26.4, SciPy 1.13.1), 3.12
  (NumPy 2.0.2) and 3.13 (NumPy 2.5.3), all on the same Windows machine; 3.14 had a smoke test
  only. Not tested: Linux, macOS, big-endian machines, GPUs, a Qdrant server, other BLAS builds on
  the full suite, the `vlm` extra, and real encoder output through the new layer
  (`BUILD_AND_PORTABILITY_REPORT.md`, section 7). Bitwise equality across platforms is not
  promised.
- A Qdrant server (its approximate graph, its own quantisation and rescoring), GPUs, and compiled
  scoring kernels.
- Samples drawn without replacement from a small finite pool, and dependent queries (sessions,
  authors, near-duplicates): no proof covers them (`STATISTICAL_REVIEW.md`; assumption A1).
- Several looks at one sample. Simulations multiplied the false-selection probability by 3 to 14
  (`STATISTICAL_REVIEW.md`, section 7.7).
- External review of the derivation. None has been done.
