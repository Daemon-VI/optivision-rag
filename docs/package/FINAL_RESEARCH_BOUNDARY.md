# The boundary between the package and the research

Status: 2026-10-06, branch `package-universal`. This document states what the OptiVision
package supports, what its research record does and does not show, and what the package does
not guarantee. Where it repeats a research result it copies it from
`reports/research/X_RESULTS.md` and names the experiment; it adds no analysis.

Evidence labels, as in the research reports: **MEASURED** (computed once on a whole pool or
corpus), **SIMULATED** (resamples of one measured pool; *pool-supported* when the budget does
not exceed the pool), **ENGINEERING** (a property of the code checked by tests on synthetic
data; not a research result).

## 1. What OptiVision currently supports

For any late-interaction encoder whose output is one `[n_vectors, dim]` float array per
document and per query, scored by MaxSim:

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
code for numerical equality. None of it was evaluated by a registered experiment as a package.

## 2. What "universal" means

The interface, the composition of pipelines, the generation of candidates, the evaluation, the
decision procedure, the certificate, the artifact and the backend seam are the same for every
encoder. That is all it means.

It does **not** mean that one compression configuration, one operator or one candidate list
works on every encoder. No such claim is made, and the research record contradicts the strong
form of it (section 6).

## 3. What remains model-dependent

- The retention any pipeline achieves. It depends on the encoder, on the coordinate basis its
  vectors happen to be in, on the corpus and on the queries.
- Whether the selection rule selects anything, and how much compression it selects. It depends
  on all of the above and on the query budget, the target and the reference.
- The default candidate list. It is frozen and the same for every encoder, but it was shaped by
  results on the development encoders. It is model-blind in what the procedure inspects, not
  neutral in how it was built.
- Encoding itself: adapters, patch grids, page images and pixel-space pruning are specific to
  model families and live outside the core.

What does not depend on the encoder: the selection rule's error statement, given its
assumptions (section 9), because the rule looks only at per-query arrays.

## 4. What X1 demonstrated

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
  labelled retention; using it cost at least a factor of 2 in compression (26 of 37
  pool-supported cells).

What X1 did not show: that agreement is retrieval quality. It is not, and there is no
statistical statement connecting the two constructs. The CP1/CP2 conditions are empirical
observations on these stores.

## 5. What X2 demonstrated

- **H3, primary, supported within pool-supported scope.** With every query its pool can supply,
  the procedure deployed non-trivial compression at T = 0.97, delta = 0.05 on all seven stores
  at a rate of at least 0.90; the lowest was 0.917 [0.905, 0.929] on SciFact at n = 300
  (SIMULATED, pool-supported). That value is close to the 0.90 threshold.
- **The three stricter secondary cells fail.** T = 0.99 non-trivial fails on all seven stores;
  T = 0.97 substantial fails on the three DocVQA stores and SciFact; T = 0.99 substantial fails
  on all seven. "Fails" means the registered criterion, a deployment rate of at least 0.90,
  was not met; at T = 0.99 it was not met on any store at pool-sized budgets.
- The extension beyond the pools holds in simulation only; a result at 900, 1,800 or 3,600
  queries is a statement about redrawing the pool's own queries, not about that many
  independent queries.
- The validity check covered 2,800 cells with no confirmed failure. These are resamples of one
  pool per corpus: they check the implementation against the derivation; they do not replace it.

## 6. What X3 demonstrated

- **H5 fails.** 13 of 16 primary (dataset, transformation) pairs hold; 3 do not.
- The signed coordinate permutation and the reordering of vectors changed nothing material on
  any dataset. The three failing pairs are rotations: ColPali DocVQA under one rotation, and
  SciFact under both.
- The largest change in labelled retention on a primary pair is 0.043 (MEASURED on the
  transformed vectors).
- On SciFact at T = 0.97, n = 300, the non-trivial deployment rate of the procedure was 0.9175
  in the native basis and 0.0995 and 0.1415 under the two rotations, with median certified
  compression 6.6x against 2.0x (SIMULATED, pool-supported; `reports/research/X3/x3_scifact.json`).
- The direction is not uniform. On the extension dataset ColPali InfoVQA the procedure
  outcome was material under both rotations, upward in some cells and downward in others
  (`reports/research/X3/x3_colpali_infovqa.json`).
- In the extension set, reported separately, 8 of 12 pairs hold; the four that do not are
  rotations.

## 7. What the failure of H5 means

Within the registration: on these datasets, the measured result of some sign-code (`binary`)
and 4-bit (`int4`) pipelines depends on the coordinate basis under the two registered
rotations, beyond the registered tolerance. It is not a statement about other maps, other
encoders or other candidates.

For the package:

- The selection procedure can be model-blind in what it is allowed to inspect while the
  compression it yields depends on the coordinate basis. Both are true at once.
- A certificate is about the vectors as given. It does not carry over to a rotated copy of the
  same index, and the package binds it to a digest of the corpus for that reason.
- No stage is assumed to behave the same in another basis. Each stage declares the basis
  changes that leave its scores unchanged, and pipelines and certificates carry
  `basis_sensitive`.
- The rule's statement is unaffected: whatever basis the vectors are in, the rule tests the
  pipeline on those vectors. What changes with the basis is the yield, not the validity.
- A rotation stage exists as a diagnostic. X3 found that rotation changes what is certified,
  downward on SciFact and in both directions on one extension dataset. Nothing establishes
  that a rotation helps or harms in general; it has to be measured like any other candidate.

## 8. What XH would test

XH is the held-out encoder protocol designed in section 13 of the preregistration. It has not
been registered and has not been run.

- **H1**: whether the frozen procedure deploys non-trivial compression while keeping its error
  level on encoders that took no part in its design.
- **H2**: the price of model-blindness, the compression an encoder-specific tuned procedure
  certifies relative to the model-blind one on the same queries.

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
3. More than the certifying rule's own statement: the probability that a compressed candidate
   is selected and its retention under the query distribution is below the target is at most
   the error level, over the draw of the calibration queries, **conditional on** independent
   queries from the deployment distribution, a fixed corpus, a candidate list and order fixed
   before the queries were seen with stages fitted on documents only, a bounded metric with a
   positive baseline mean, a query order fixed in advance, and the reference taken as given.
   It is not a probability conditional on a selection, not a statement about a finite batch of
   future queries, and not a statement about power. The rule applies published tools; the
   derivation that combines them is the project's own and has not been externally reviewed.
4. That the rule selects anything. It may abstain, and at strict targets with the query
   budgets studied it did.
5. That agreement with the uncompressed ranking is retrieval quality.
6. That a certificate is secure against forgery without a signature key (see
   `CERTIFICATE_SPEC.md`).
7. Speed. Scoring decodes codes to float32; native compressed scoring was slower in NumPy and
   torch on one laptop CPU (Q5/Q6), compiled kernels were not tested, and nothing general is
   claimed either way.
8. Scale. The million-page figures of Q8 are projections.

## 10. How the package and the research record are kept apart

- The research record (`reports/research/**`, `scripts/research/**`, the registered inputs,
  the preregistration and its amendment, `paper/**`, `reports/universal/**`) is not modified
  by package work. The package work lives on its own branch; the research branch is unchanged.
- Eleven library files that the research harness pins by content hash are byte-identical to
  the registered ones. New package code is in new files.
- The package does not import research scripts (enforced by a test). Tests may read the
  frozen research functions and stored selections, read-only, to check that the ported
  selection rule is numerically the same procedure.
- The registered runs recorded the library tree `0b9523bd10f43941a9704cbf6fae1db1804880ee`. The
  package branch has a different tree, because it adds files. X1-X3 belong to the commits
  they were run at. A future registered experiment must record the tree it runs on; if it is to
  run on the package branch, that needs a new dated amendment, not a silent re-pin.
- X1, X2 and X3 were not re-run. XH was not run. No hypothesis, protocol or historical result
  was changed.
