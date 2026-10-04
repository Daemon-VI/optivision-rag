# X1–X3 harness: implementation notes

**Status, 2026-10-04.** The code that will run X1, X2 and X3 is written, tested and
dry-run validated. **No X1, X2 or X3 computation has been run. No per-query value was
read while writing it. No result exists.**

- Specification: `NEXT_PHASE_PREREGISTRATION.md`, commit
  `73085ab637679f30fffece6632eb7b9e92adff79` (blob `fa9deb73fdf380f06141d89cfaf145444f883506`),
  recorded in the registry by `d408b96db87698f76f5f320b8ee8591a3d75ecd9`. Section numbers
  below refer to it. That document was not edited.
- XH is not built. Nothing in `src/optivision` was changed, and no existing script, report
  or store was edited.
- **A registered run is refused until the open points of §6 are accepted** by the project
  owner. The code carries one reading for each. None of them is decided here.

## 1. Files

| file | role |
|---|---|
| `scripts/research/x_common.py` | registered constants and hashes; input validation; sampling; the resampling engine; decision rules; output writers; the run log |
| `scripts/research/x1_agreement.py` | X1, label-free reference agreement (§10) |
| `scripts/research/x2_budget.py` | X2, query budget and certification (§11) |
| `scripts/research/x3_symmetry.py` | X3, coordinate and symmetry test (§12) |
| `tests/test_x_harness.py` | 52 tests on synthetic fixtures |
| this file | how to run, what is checked, what is open |

Method C is not re-implemented. `betting_pvalues`, `to_y`, `fixed_sequence_by_family`,
`fixed_sequence`, `walk_select` and `smallest` are imported from `e1_stats.py`, and
`a_bounds`, `clopper_pearson` and `Problem` from `e1_1_study.py`. The harness adds only the
loop over budgets, seeds and stores. That loop follows `e1_1_study.run_cell`, and gate G4
compares its output with the selections E1 stored, seed by seed.

## 2. Running

```
export OPTIVISION_DATA=../optivision-rag-v2/data      # X3 only: where the vectors are
export OPTIVISION_CACHE_BYTES=750000000               # X3 only: bound on the transform cache
export PYTHONPATH="src;scripts/research"

python scripts/research/x1_agreement.py               # dry run (the default)
python scripts/research/x1_agreement.py --run         # the registered experiment
python scripts/research/x2_budget.py --run
python scripts/research/x3_symmetry.py --run --dataset colpali_docvqa   # one dataset at a time
```

- **A dry run** reads file hashes, array headers and candidate metadata. It reads no
  per-query value, computes no result and writes nothing. `--json` prints the full report.
- **A run** first checks every input (§3), then every gate for every selected store, then
  analyses store by store. One store is one unit of work. Its files are written only if all
  its checks passed and the machine did not sleep. Every attempt is logged in `runs.json`
  (`attempts.json` for X3), including failed gates and interrupted runs.
- `--store NAME` (X1, X2) repeats one unit. The summary and the report are written when all
  seven stores have a valid unit from the same commit.
- **X3** has one unit per (dataset, transformation), at most three attempts each (§21). The
  extension set (`--extension`) is refused until the whole primary set is complete.
- Run one job at a time. X3 holds a corpus and its transformed copy in memory.
- A run asks Windows not to sleep while it works. Closing the lid can still suspend the
  machine; a unit during which it slept is marked invalid and is repeated (§20).

PROJECTED durations on the laptop. X1 and X2 are from timing the engine on a synthetic
store of 451 queries; X3 is five store builds per dataset at the build times recorded in
the E1 sidecars.

| experiment | duration | peak memory |
|---|---|---|
| X1 | about 25 minutes (30 s per store and threshold) | about 0.6 GB |
| X2 | about 2 hours (18 minutes per store; rule A's bootstrap is about 13 of them) | about 0.6 GB |
| X3, primary set | about 7 hours (2.5 h, 2.0 h, 0.7 h, 1.9 h for D1, D4, D5, D7) | not measured |
| X3, extension set | about 7 hours | not measured |

## 3. What is checked before anything is read

A mismatch stops the run before any array is loaded. Nothing is substituted.

| input | registered identity (§22.1) | check |
|---|---|---|
| seven store `.npz` files | sha256 | the file's bytes |
| seven sidecar `.json` files | sha256 | the file's bytes (see L2) |
| array names, shapes, types | §9.1 | `.npy` headers only |
| pool size, documents, dimension, metric | §9.1 | sidecar fields |
| candidate list in every sidecar | sha256 `af29b011…3520` | recomputed from the sidecar |
| candidate list the code generates | the same hash, for d = 128 and d = 96 | `recommended_search_space` |
| 14 code files | git blob hashes | recomputed from the working files |
| Method C constants | bet cap 0.9; levels 0.05 and 0.01; δ/7; E1 grid and seeds | read from the imported modules |
| E1 selection files (gate G4) | not in §22.1 | present, real content, expected keys; sha256 recorded |
| code state | §22.2 | clean working tree; preregistration blob at HEAD is one the harness implements |

Gates that read values run next, in the registered order: G2 (finite, in [0, 1],
`base_b1` = 1), G3 (ViDoRe stores: both constructs agree where `base_labels` = 1), G4
(Method C under labels at n = 50, 100, 225 equals the stored E1 selections). X2 also
compares A, A-bonf, B, C-split and D with the stored E1 selections, because §11.3 runs them
"exactly as in E1". X3 has its own gates: the T0 store against the committed store, and the
symmetry check on the uncompressed vectors.

## 4. What each run computes

**X1 (§10).** Per store: A1 pool table for the 37 distinct candidates; A2 passing sets,
free and chain oracles, CP1; P0 under agreement at every registered budget, thresholds
0.99, 0.97, 0.95, 0.90, both error levels (A3); the labelled retention of what the
agreement arm deploys (A4, CP2); the paired comparison with P0 under labels on the same
resamples (A5); A6. Across stores: H4a, H4b, H4c, and the sentences N8, N9, N10.

**X2 (§11).** Per store and construct: P0, I8, D and C-split at every registered budget
with seeds 0–1999; A, A-bonf and B at 50, 100, 225 with seeds 0–1999; A and B at the
pool-capped budget and at 900 with seeds 0–499. Per cell the outcomes of §11.4. The
MEASURED full-pool decision. The monotonicity and validity checks of §11.6. H3 at the
pool-capped budget, n* per (construct, threshold, δ) with the class of the budget that
decides it, and the sentences N6, N7.

**X3 (§12).** Per dataset and transformation: the map, the symmetry check, the store
rebuilt with the calls of `e1_0_rebuild.build`, P0 at n = 225 and at the pool-capped budget
(labels, three thresholds, δ = 0.05). From the stores: per-candidate change in R_L and R_A
with the paired bootstrap interval and its class, the procedure outcome against T0, whether
each pair holds, H5, and the sentences N11, N12. The identity run also saves the top-10
lists behind overlap@5 and overlap@10.

Budgets, per store, as the code derives them:

| pool | pool-supported | extrapolated | pool-capped budget |
|---|---|---|---|
| DocVQA (451), InfoVQA (494) | 50, 100, 225, 450 | 900, 1800, 3600 | 450 |
| SciFact (300) | 50, 100, 225, 300 | 450, 900, 1800, 3600 | 300 |

Every simulated row carries `budget_class` and `evidence`
(`SIMULATED, pool-supported` or `SIMULATED, extrapolated`). Only the full-pool decision and
the pool-level tables are `MEASURED`. The reports tabulate the two classes in separate
columns.

## 5. Outputs

Written only by a run, under `reports/research/X1/`, `X2/`, `X3/`. Paths outside those
directories are refused.

| file | content |
|---|---|
| `x1_<store>.json`, `x2_<store>.json` | every table of that store, the gates, and a header with the preregistration commit, the code commit, the Method C configuration, the seeds and the input hashes |
| `x?_<store>_selections.npz` | per-seed selections (−1 = abstain), keys as in the E1 files, prefixed `L_` or `A_` for the construct and `rep_` for the replication block |
| `x?_summary.json`, `X?_REPORT.md` | verdicts, the registered negative-result sentences that apply, short tables |
| `x?_provenance.json` | environment, git state, input validation, sha256 of every output |
| `runs.json`, `attempts.json` | every attempt: start, end, time not awake, status, reason |
| X3: `store/`, `selections/`, `lists/`, `units/`, `x3_<dataset>.json` | the rebuilt stores, P0 selections, top-10 lists, one record per unit, the per-dataset analysis |

Result files hold no timestamp, so the same inputs at the same commit give the same bytes.
Text outputs are written with LF line endings on every platform, and the recorded hashes
are of those bytes, which is also what git stores.

## 6. Open points: decisions needed before a run

The preregistration leaves these technically open. Each row gives the reading the code
carries. A registered run of an affected experiment is refused until the point is listed
in `ACCEPTED_OPEN_POINTS`.

| id | affects | what the text leaves open | implemented reading | what the choice can change |
|---|---|---|---|---|
| OP1 | X3 | T1 has seed 0, but §12.3 does not say how the permutation and the signs are drawn | `rng = default_rng(0)`; `perm = rng.permutation(d)`; `signs = rng.integers(0, 2, size=d) * 2 - 1`; coordinate j becomes `signs[j] * v[perm[j]]` | which signed permutation is tested |
| OP2 | X3 | T3 has seed 3, but §12.3 does not say how the per-document orders are drawn | one `default_rng(3)`; documents in corpus order; `rng.permutation(n_i)` each; `protected`, `positions`, `weights`, `importance` move with their rows | which reordering is tested |
| OP3 | X3 | §12.3 defines U but not how it is applied to a stored row; §12.2 writes (Uq)·(Ud) | every vector v becomes U v (rows `X @ U.T`), in float64, in blocks of 65,536 rows, stored as float32 | U or its transpose: two different rotations |
| OP4 | X3 | T0 gate fallback: "all 40 pool retentions" and "per-query entries" name no construct or array | the stricter reading: both constructs within 0.001; at least 99.5% identical entries in each of the four arrays | only whether a T0 store that is not exactly equal passes |
| OP5 | X1, X2 | borderline rule: which rates "decide a verdict or an n*" | the H3 cells (primary and secondary), the H4c cells, and for each n* the cells met scanning budgets from the largest down to the first failure | which cells get seeds 2000–3999. No verdict or n* differs from a wider reading |
| OP6 | X1 | A6 names neither the margin, nor the candidates, nor the aggregation | both margins in `Q3/features` (relevance margin, top-2 margin), separately; pool quartiles; per distinct candidate, 1 − mean of a in the quartile | an exploratory table only |
| OP7 | X1 | N10: when a cell counts as "the labels arm deploys non-trivially" | its non-trivial deployment rate is at least 0.90; cells are (store, threshold, δ, pool-supported budget), both error levels together | whether the N10 sentence is reported |

To accept them:

1. Decide each point: accept the implemented reading or state another.
2. Record the decisions as a dated amendment in Appendix C of the preregistration and
   commit it before the run it affects (§2 of that document).
3. In the same commit, add the amended file's blob hash to `PREREG_BLOBS_IMPLEMENTED` and
   the accepted identifiers to `ACCEPTED_OPEN_POINTS` in `x_common.py`. If a decision
   differs from the implemented reading, change the code in that commit too.
4. Run the dry run again. It lists what still blocks a run.

## 7. Readings taken where the text has a basis

Not blocking, listed so they can be challenged before any result exists.

| id | point | reading |
|---|---|---|
| R1 | full-pool decision: "per store, construct and threshold" (§11.1) names no δ | computed at both registered levels; 0.05 is shown in the report |
| R2 | thresholds of A2 and H4b | CP1 and the oracle comparison at 0.99, 0.97, 0.95; at 0.90 the agreement side only, since 0.90 is registered for agreement alone |
| R3 | H3 secondary: "the same at T = 0.99, and with 'substantial'" | three cells: (0.99, non-trivial), (0.97, substantial), (0.99, substantial) |
| R4 | "the class of the budget that decides" n* | the class of n* itself; when there is none, of the largest budget |
| R5 | an oracle that does not exist | compression 1.0, as E1 does; ratios to an oracle use the median over seeds of compression ÷ oracle compression |
| R6 | I8 | the first of the two `int8(per_vector)` positions (position 1) |
| R7 | validity check (§11.6, §16.3) | applied to P0 in X1 as well; a confirmed failure stops the run |
| R8 | replication and borderline re-runs | only the cell that triggered the rule is re-run; the pooled 4,000 seeds decide |
| R9 | X3 bootstrap | `default_rng(0).integers(0, N, size=(2000, N))`, the library's own convention; 2.5% and 97.5% quantiles; one index set for all candidates |
| R10 | X3 symmetry check | the "uncompressed per-query metric" is the labelled one; entries that are −inf in both score matrices are left out of the difference |
| R11 | X3 materiality | judged on the 37 distinct candidates; a pipeline that opens two chains has the same values in both |
| R12 | H5 when a pair fails and another is incomplete | "fails": a failure under one registered map is a failure; "not assessable" only when no pair fails and one is incomplete |
| R13 | N11 and N12 when only the P0 outcome is material | the registered sentence with "the outcome of P0" in place of "candidate k" |
| R14 | sleep check (§20, §22.2) | wall-clock time minus time on a clock that stops during sleep; more than 5 s marks the unit invalid |
| R15 | units at different commits | X1 and X2: the summary needs all seven stores from one commit. X3: each unit records its commit |

## 8. Limitations found

- **L1. The stores are Git LFS objects.** A checkout without LFS holds pointer files. The
  harness recognises a pointer whose object id is the registered hash, reports it, and
  refuses to run. The CI checkout is of this kind, so the tests there verify identity
  through the pointer and never open a store.
- **L2. The registered sidecar hashes are those of a checkout with CRLF line endings.** Git
  stores those files with LF. On the registration machine (`core.autocrlf=true`) the bytes
  match and gate G1 passes as registered. On an LF checkout the raw hash differs: the
  harness reports that the CRLF form matches and refuses to run. The hash of the candidate
  specification does not depend on line endings.
- **L3. The E1 selection files and the Q3 feature files have no registered hash.** A run
  records their sha256. Their content is checked by gate G4 (selections) and by an
  alignment check against `base_labels` (features).
- **L4. The rotations depend on the linear-algebra library.** QR and the matrix product can
  differ in the last bits between machines. Each matrix's sha256 is recorded with the run.
  On this machine the dry run gives, for d = 128: T1 `fe558cab6a3e…`, T2a `479c843b76a8…`,
  T2b `2b9074e732df…`; for d = 96: T1 `9bbb554a5c7a…`, T2a `61d2225c05ee…`, T2b
  `5d8a7c858243…`.
- **L5. The shared module is named `x_common.py`.** §22.3 lists `x1_*.py`, `x2_*.py`,
  `x3_*.py`. The name follows `e1_common.py` and `q56_common.py`.
- **L6. Only Python 3.11 was available locally.** CI runs 3.10 and 3.13.
- **L7. X3 memory was not measured.** It was exercised on small synthetic vectors only.
- **L8. No figure is drawn.** §11.5 asks for curves with different markers for the two
  budget classes. The harness writes the tables, in separate columns, and the JSON a
  figure would be drawn from. A figure made later must keep the two classes apart.
- **L9. X3 does not handle a candidate that fails to run.** §16.7 says to record it and
  treat it as not rejected. Here an error ends the unit, which is logged and counts as one
  of its three attempts. A store holds one row of values per candidate, and a failed
  candidate has none. None of the 40 candidates failed when the E1 stores were built.
- **L10. The registry rows are not updated by the scripts.** After a run, the X1, X2 and X3
  rows of `README.md` are edited by hand (§22.3).

## 9. Tests

`tests/test_x_harness.py`, synthetic fixtures only: two stores in the E1 format with the
registered 40 candidates and made-up values, and a small set of unit-norm vectors.

| area | what is tested |
|---|---|
| candidate space | generated list hashes to the registered value at d = 128 and 96; 40 candidates, 7 chains, 37 distinct; every registered sidecar carries it |
| input hashes | a changed array, a changed candidate list, an LFS pointer, other line endings: each stops a run before anything is written |
| budgets | the registered grid per pool size; classes; the pool-capped budget; rule A and B budgets and seeds |
| labels | every simulated row carries its class; nothing above the pool is called pool-supported; only the full-pool rows are MEASURED |
| seeds | samples are prefixes of the registered draw and equal E1's draws; a run is reproduced byte for byte; chunked equals unchunked |
| transformations | seeds; orthogonality; the registered QR definition; determinism; scores and norms kept; T1 exact; T3 keeps rows in their document and moves the annotations |
| Method C | bet cap 0.9; level δ/7; smallest rejected; the tie-break; the wrapper equals the frozen calls; the engine reproduces `e1_1_study.run_cell` for all ten E1 procedures |
| decision rules | oracles; borderline rule; n*; replication rule; monotonicity; gates G2, G3, G4; X3 classes, bootstrap interval, T0 gate, attempt limit, extension order |
| outputs | schema of every file; provenance header; output hashes; paths confined to their directory |
| no mutation | inputs unchanged after a run; dry runs and refused runs change no file under `reports/research` |

The frozen `run_cell` is used to produce the "stored E1 selections" of the synthetic
stores, so gate G4 is tested against the frozen code itself.

```
PYTHONPATH=src python -m pytest tests/test_x_harness.py -q
```
