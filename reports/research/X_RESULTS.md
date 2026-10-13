# X1–X3: execution record and results

**Status, 2026-10-05.** X1, X2 and X3 were run as registered, on 2026-10-04 and
2026-10-05. This file is the index to what was run and what it gave. It adds no analysis:
every number below is copied from the report or the JSON files the registered scripts
wrote, and each section names its source.

Evidence labels used below: **MEASURED** (computed once on a whole pool or a whole corpus),
**SIMULATED** (resamples of one measured pool; *pool-supported* when the budget does not
exceed the pool, *extrapolated* when it does). Nothing here is EXTERNAL or PROJECTED. An
extrapolated budget says what the procedure would do if queries were redrawn from the
pool's own N queries. It is not evidence of what that many independent queries would give.

Scope of every result: the seven development stores (three visual encoders on DocVQA and
InfoVQA, one text encoder on SciFact), one pool per corpus, the registered candidate space
of 40 pipelines. Reference agreement is where a candidate places the document the
uncompressed index ranks first. It is not retrieval quality, and no result here is a
statement about relevance, about other encoders or about other corpora.

## 1. What was executed

| item | identity |
|---|---|
| preregistration | `NEXT_PHASE_PREREGISTRATION.md`, commit `73085ab637679f30fffece6632eb7b9e92adff79`, blob `fa9deb73fdf380f06141d89cfaf145444f883506` |
| amendment in force | Amendment 1 (2026-10-04), commit `130286dd06c763f2bd11a66d706d98dbccf053e6`; amended blob `445d94cdb1dda8671a7d4c4ba3bbb7f895f15700`, the text every run required at HEAD |
| registry entry of the amendment | commit `d125232be989671af35eb06bf46152fd5e42db8d`, before any execution |
| X1 run at | `d125232be989671af35eb06bf46152fd5e42db8d` |
| X2 run at | `e4d48653967b470945a9a70da8c9d51bd0f78da0` (the X1 results commit; no code change) |
| X3 run at | `59d08f24bd8320486f7be1cd685a7b64cc5e619e` (the X2 results commit; no code change) |

Code, identical at all three commits (git hashes; recorded with every unit):

| file | hash |
|---|---|
| `scripts/research/x_common.py` | `0b8031d46459e32df4565c6c9668174eadfc3794` |
| `scripts/research/x1_agreement.py` | `47c0ae575d548421116c126cd24d7fefdc993980` |
| `scripts/research/x2_budget.py` | `0999a161186687474a32a6be8b6af7432c3b7cd8` |
| `scripts/research/x3_symmetry.py` | `8cd10c0b04ea43e491106fc2d3cd06ec154c8195` |
| `scripts/research/e1_stats.py` | `13494307a4a42cd2c1da63f34aa0952a92b33130` |
| `scripts/research/e1_1_study.py` | `5896e763b49d4a0f3f0cc988dc2b5ec5e6b2b9aa` |
| `scripts/research/e1_common.py` | `3bef181683307eb98a9c5e2d61ab02d87801bbfe` |
| `scripts/universal_study/s7b_selection_rules.py` (X3 loader) | `1b02292cea8c5d732af9d88d256aece7799b469e` |
| `src/optivision` (tree) | `0b9523bd10f43941a9704cbf6fae1db1804880ee` |

The scripts are their own analysis: each writes its tables, its summary, its report and its
provenance file. No other analysis script was written or run.

Environment: Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, OpenBLAS 0.3.31 (scipy-openblas);
Windows 11 (10.0.26200); Intel Core i7-1165G7, 4 cores, 8 logical; 15.8 GB RAM. Launch
environment as in `X_HARNESS.md` §2: `PYTHONPATH="src;scripts/research"`; for X3 also
`OPTIVISION_DATA=../optivision-rag-v2/data` and `OPTIVISION_CACHE_BYTES=750000000`;
`OPTIVISION_SCORE_DEVICE` unset. The provenance files record `dirty_tree: true` because
the output directory of the run itself was untracked when they were written; they also
record no tracked change and no uncommitted code.

Inputs, all read only: the seven E1 stores and sidecars (sha256 in §22.1 of the
preregistration) and the 34 auxiliary files of Amendment 1, part 4 (7 E1 selection files,
6 Q3 feature files, 21 vector files). All 48 hashes were checked before X1 and again after
each of X1, X2 and X3: no mismatch at any point.

Configuration, unchanged from the registered text and recorded in every output header:
resample seeds 0–1999 and the block 2000–3999 under the registered rules only; samples are
the first n entries of `default_rng(seed).integers(0, N, size=3600)`; budgets 50, 100, 225,
450, 900, 1800, 3600 and the pool size where it is below 450; pool-capped budget 450 (300
on SciFact); δ = 0.05 and 0.01; Method C with bet cap 0.9, seven chains at δ/7, imported
unchanged from the frozen E1 scripts; candidate specification sha256
`af29b01146a90a048ffea9cfbd66f0fedf988134829cd4dc56861f1207893520` (40 candidates, 7
chains, 37 distinct); X3 transformation seeds 0, 1, 2, 3 and bootstrap seed 0 with 2,000
resamples.

## 2. Execution ledger

| | X1 | X2 | X3 |
|---|---|---|---|
| units | 7 stores | 7 stores | 35 = 7 datasets × (T0, T1, T2a, T2b, T3) |
| attempts logged | 7 | 15 | 35 |
| valid | 7 | 8 | 35 |
| invalid | 0 | 7 | 0 |
| computation in valid units | 26 min | 94 min | 7.3 h |
| attempt log | `X1/runs.json` | `X2/runs.json` | `X3/attempts.json` |
| results commit | `e4d4865` | `59d08f2` | `bbad6bc` |

**X2, the invalid attempts.** All seven are operational and are recorded in
`X2/runs.json`; none involves a result.

- First pass, 2026-10-04: the process was stopped by the host for low system memory after
  one store. That store's unit was valid; the other six are logged as "did not finish: the
  process was killed or interrupted". Six invalid attempts.
- Second pass, 2026-10-05, the same command and code: six stores valid. For
  `colqwen25_docvqa` the machine slept for about 18 minutes during the unit, so the unit is
  invalid under rule F3 and no output was written for it. One invalid attempt.
- Third invocation, `--store colqwen25_docvqa`: valid.

`colpali_docvqa` therefore has two valid attempts. Their output hashes in `runs.json` are
identical, byte for byte. X1 and X2 units carry no attempt limit.

**X3.** Every unit was valid on its first attempt, none slept, and all 35 carry one code
identity. No candidate failed, so rule F1 was never applied; no unit used a second attempt;
no pair is incomplete. The primary set (16 pairs) was complete before the first extension
dataset was started.

Borderline and replication rules: X1, no cell triggered either. X2, seven cells used the
borderline rule (listed in `X2/X2_REPORT.md`); none is a primary H3 cell; the validity
check covered 2,800 cells with no re-run and no confirmed failure. No monotonicity
violation in X1, X2 or X3.

## 3. Primary registered results

| hypothesis | verdict | evidence | source |
|---|---|---|---|
| H4a, agreement is a conservative proxy | supported | CP1 MEASURED: holds on all 21 (store, threshold) pairs. CP2 SIMULATED: holds on 150 cells, none re-run | `X1/x1_summary.json` |
| H4b, the two constructs are not equivalent | supported | MEASURED: free oracles differ by a factor of 2 or more on 21 of 21 pairs | `X1/x1_summary.json` |
| H4c, agreement is usable with the pool | supported | SIMULATED, pool-supported: non-trivial deployment rate at T_A = 0.95, δ = 0.05 at the pool-capped budget is 1.000 on the six visual stores and 0.999 on SciFact | `X1/x1_summary.json` |
| H3, primary | supported | SIMULATED, pool-supported: non-trivial deployment rate of P0 under labels at T = 0.97, δ = 0.05 at the pool-capped budget is at least 0.90 on all seven stores; by pool, the lowest are 0.998 (DocVQA), 1.000 (InfoVQA) and 0.917 [0.905, 0.929] (SciFact, n = 300) | `X2/x2_summary.json` |
| H5, invariance under the four registered maps | **fails** | 13 of 16 primary pairs hold; 3 do not; none incomplete | `X3/x3_summary.json` |

H5 in detail (per-candidate change in labelled retention MEASURED on the transformed
vectors; the P0 outcome SIMULATED, pool-supported):

| dataset | T1 signed permutation | T2a rotation | T2b rotation | T3 row reordering |
|---|---|---|---|---|
| ColPali DocVQA | holds | does not hold (2 candidates material) | holds | holds |
| ColQwen2 InfoVQA | holds | holds | holds | holds |
| ColQwen2.5 DocVQA | holds | holds | holds | holds |
| SciFact | holds | does not hold (1 candidate; P0 outcome) | does not hold (5 candidates; P0 outcome) | holds |

The largest change in labelled retention on a primary pair is 0.043. Every T0 store equals
the committed E1 store, and the symmetry check on the uncompressed vectors passed for every
pair (per-query metric identical; score difference at most 3.6e-07 of the largest score).

Registered negative-result sentences that apply, verbatim from the reports:

- N10 (X1): "Agreement is materially stricter than labelled retention. Using it cost at
  least a factor of 2 in compression."
- N11 (X3), ten sentences of the form "The result for candidate k on dataset D is not
  invariant under the registered rotation.": on ColPali DocVQA for `binary` and
  `hierarchical_merge(0.25) > binary`; on SciFact for `adaptive_merge(0.6) > int4(mean)`,
  `adaptive_merge(0.7) > int4(mean)`, `hierarchical_merge(0.33) > binary`,
  `adaptive_merge(0.8) > binary`, `adaptive_merge(0.7) > binary`,
  `adaptive_merge(0.6) > binary`, and twice for the outcome of P0 (once per rotation).
- N6, N7 (X2), N8, N9 (X1) and N12 (X3) do not apply.

What H5 failing means, within the registration: on these datasets, the measured result of
some `binary` and `int4(mean)` pipelines depends on the coordinate basis under the two
registered rotations, beyond the registered tolerance. It is not a statement about other
maps, other encoders or other candidates. The signed permutation and the row reordering
changed nothing material on any dataset.

## 4. Secondary registered results

- **H3, the three secondary cells (X2; SIMULATED, pool-supported): all fail.** T = 0.99
  non-trivial fails on all seven stores; T = 0.97 substantial fails on the three DocVQA
  stores and SciFact; T = 0.99 substantial fails on all seven.
- **H3, extension beyond the pools (X2; simulation only): holds in simulation.** At
  T = 0.97, δ = 0.05, n* exists on every store and is pool-supported on every store (225 on
  two InfoVQA stores, 300 on SciFact, 450 on the others), so this entry rests on no
  extrapolated budget. Other n* entries in `X2/X2_REPORT.md` are decided at 900, 1800 or
  3600 and are marked extrapolated there; they are statements about redrawing the pool's
  own queries only.
- **The full-pool decision (X2; MEASURED)**, one realisation per (store, construct,
  threshold), is tabulated in `X2/X2_REPORT.md`. It is the only measured outcome of X2.
- **X3, agreement retention:** material changes under the rotations on ColPali DocVQA (T2a,
  T2b), ColQwen2.5 DocVQA (T2b) and SciFact (T2a, T2b), largest 0.079; none under T1 or T3.
- **X3, extension set, reported separately and never substituted for a primary dataset:**
  8 of 12 pairs hold. The four that do not are rotations: ColPali InfoVQA T2a and T2b (the
  P0 outcome is material; no candidate is) and ColQwen2 DocVQA T2a and T2b (2 and 1
  candidates material). ColQwen2.5 InfoVQA holds under all four maps.

## 5. Diagnostics and exploratory outputs

These decide nothing.

- **D1, the alternative N10 count (X1).** Primary count, which decides: in 26 of 37
  pool-supported cells where the labels arm deploys non-trivially at a rate of at least
  0.90, the median certified compression under agreement is at most half that under labels.
  Diagnostic count under the other reading ("more than half of the resamples deploy
  non-trivially"): 26 of 40.
- **A6 (X1; exploratory, MEASURED):** agreement loss by quartile of the baseline score
  margin, in each `X1/x1_<store>.json`.
- **X3 (exploratory):** the two rotations compared, and overlap@5 and overlap@10 from the
  identity run, in each `X3/x3_<dataset>.json`; the top-10 lists in `X3/lists/`.

## 6. Where everything is

| what | where |
|---|---|
| reports written by the scripts | `X1/X1_REPORT.md`, `X2/X2_REPORT.md`, `X3/X3_REPORT.md` |
| verdicts and sentences | `X?/x?_summary.json` |
| every table, per store or dataset | `X1/x1_<store>.json`, `X2/x2_<store>.json`, `X3/x3_<dataset>.json` |
| per-seed selections | `X1/` and `X2/` `*_selections.npz`; `X3/selections/` |
| X3 rebuilt stores, top-10 lists, unit records | `X3/store/`, `X3/lists/`, `X3/units/` |
| every attempt, valid or not | `X1/runs.json`, `X2/runs.json`, `X3/attempts.json` |
| environment, git state, input validation, sha256 of every output | `X?/x?_provenance.json` |
| registry rows | `README.md`, rows X1, X2, X3 |
| how the harness works and how to run it | `X_HARNESS.md` (its status line describes the state before execution) |

To verify a run: the provenance file of each experiment lists the sha256 of every output
beside it (17 for X1, 17 for X2, 157 for X3). They were recomputed after each run with no
mismatch. The four files below anchor each directory:

| file | sha256 |
|---|---|
| `X1/x1_provenance.json` | `7010a7699996f7721cb813faaf331a65396bb6b07b074997196099602f01ddf0` |
| `X1/x1_summary.json` | `961ba2efd65787625b3eda2996dd5c86305e03a975ba92c09b6629c09de0c226` |
| `X1/X1_REPORT.md` | `3b16fceb611ee5c2da10759f9c9a6f41c9fd39f190a631490ae87faa24d6de8c` |
| `X1/runs.json` | `52656e9bc00ac24af7a7db7015c05a6d36b55fb7fa3927b55ba41252f6c3a910` |
| `X2/x2_provenance.json` | `0241b532dbba3dc18f6d5efb5af0d51d7ff1af28550cd557034d47c72a0ca3b1` |
| `X2/x2_summary.json` | `c7e99a8a8ce19746402c25be2ff817b9e2a5df06d8631b450d51408932577906` |
| `X2/X2_REPORT.md` | `7895e7412422b0d1ca7a12710f5746ea0174c1b18e271d8770d0c76e85ee4bfd` |
| `X2/runs.json` | `83bd56a4b536ff02f279dd2611dd784c341e1d2049791ecba015a27a285d70c6` |
| `X3/x3_provenance.json` | `0e3f4522d98905c891bcfa2e9fb546a8251b3547d3cc938adfe01043c4431d41` |
| `X3/x3_summary.json` | `68cef184bbe88a525e2b35689eb1c6902390a9d1160c5e25a0673b3cf61f1e0e` |
| `X3/X3_REPORT.md` | `d4e24f74ac22b2f8095b1dff7e8dd900336c80aae2f7d0ff9b570570d076e6a2` |
| `X3/attempts.json` | `dc0cfe834b87443da850330a8896bf5d50a22b9aa8025c5188b93656e92f08ba` |

The hashes are of the files with LF line endings, which is how the scripts write them and
how git stores them.

## 7. Deviations and things a reader should know

- **No deviation from the registered protocol.** No hypothesis, dataset, candidate,
  threshold, seed, budget, Method C setting or decision rule was changed, and no code file
  was edited between the amendment and the end of X3.
- **Operational events**, all handled by the registered rules and all in the attempt logs:
  the X2 process stopped for low memory, and one X2 unit invalidated by sleep (§2).
- **Power.** Mains power dropped several times during the two days. Parts of X3 ran on
  battery. The sleep check of rule F3 passed for every X3 unit; timing is not an outcome of
  X1–X3.
- **Machine settings changed during execution**, outside the repository: sleep and
  hibernate on battery were set to never before X3, so that a power cut would not suspend a
  unit. X3 was launched as a process detached from the session that started it. Neither
  changes what the scripts compute.
- **`X_HARNESS.md` was not edited.** Its opening status ("no X1, X2 or X3 computation has
  been run") is the state on 2026-10-04, before execution.
- **XH was not run** and is not part of this record.
