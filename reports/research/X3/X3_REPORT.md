# X3: coordinate and symmetry test

Specification: `reports/research/NEXT_PHASE_PREREGISTRATION.md` section 12, commit `73085ab637679f30fffece6632eb7b9e92adff79`. Code commit `59d08f24bd8320486f7be1cd685a7b64cc5e619e`. Written by `scripts/research/x3_symmetry.py`; every number is in the JSON files beside this report.

**What is tested.** Four fixed maps with registered seeds: a signed coordinate permutation (T1), two rotations (T2a, T2b) and a reordering of the rows of each document (T3). Per-candidate retentions are MEASURED on transformed vectors; the outcome of P0 is SIMULATED at pool-supported budgets. Support would mean invariance under these four maps on these datasets within the registered tolerances. It would not show that results are independent of the coordinate basis in general.

## Summary

- **H5: fails** (13 of 16 primary pairs hold).
- The result for candidate `binary` on dataset colpali_docvqa is not invariant under the registered rotation.
- The result for candidate `hierarchical_merge(0.25) > binary` on dataset colpali_docvqa is not invariant under the registered rotation.
- The result for candidate `adaptive_merge(0.6) > int4(mean)` on dataset scifact is not invariant under the registered rotation.
- The result for the outcome of P0 on dataset scifact is not invariant under the registered rotation.
- The result for candidate `adaptive_merge(0.7) > int4(mean)` on dataset scifact is not invariant under the registered rotation.
- The result for candidate `hierarchical_merge(0.33) > binary` on dataset scifact is not invariant under the registered rotation.
- The result for candidate `adaptive_merge(0.8) > binary` on dataset scifact is not invariant under the registered rotation.
- The result for candidate `adaptive_merge(0.7) > binary` on dataset scifact is not invariant under the registered rotation.
- The result for candidate `adaptive_merge(0.6) > binary` on dataset scifact is not invariant under the registered rotation.
- The result for the outcome of P0 on dataset scifact is not invariant under the registered rotation.

## Pairs

Classes of the change in R_L over the 37 distinct candidates: invariant (at most 0.005), material (at least 0.02 with an interval excluding 0), small (anything else).

| dataset | set | map | invariant | small | material | largest change in R_L | P0 outcome material | holds |
|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | primary | T1 | 37 | 0 | 0 | 0.000 | no | yes |
| colpali_docvqa | primary | T2a | 19 | 16 | 2 | 0.040 | no | no |
| colpali_docvqa | primary | T2b | 22 | 15 | 0 | 0.030 | no | yes |
| colpali_docvqa | primary | T3 | 31 | 6 | 0 | 0.010 | no | yes |
| colqwen2_infovqa | primary | T1 | 37 | 0 | 0 | 0.000 | no | yes |
| colqwen2_infovqa | primary | T2a | 29 | 8 | 0 | 0.011 | no | yes |
| colqwen2_infovqa | primary | T2b | 30 | 7 | 0 | 0.008 | no | yes |
| colqwen2_infovqa | primary | T3 | 37 | 0 | 0 | 0.002 | no | yes |
| colqwen25_docvqa | primary | T1 | 37 | 0 | 0 | 0.000 | no | yes |
| colqwen25_docvqa | primary | T2a | 19 | 18 | 0 | 0.031 | no | yes |
| colqwen25_docvqa | primary | T2b | 25 | 12 | 0 | 0.016 | no | yes |
| colqwen25_docvqa | primary | T3 | 35 | 2 | 0 | 0.006 | no | yes |
| scifact | primary | T1 | 37 | 0 | 0 | 0.000 | no | yes |
| scifact | primary | T2a | 18 | 18 | 1 | 0.042 | yes | no |
| scifact | primary | T2b | 17 | 15 | 5 | 0.043 | yes | no |
| scifact | primary | T3 | 33 | 4 | 0 | 0.009 | no | yes |
| colpali_infovqa | extension | T1 | 37 | 0 | 0 | 0.000 | no | yes |
| colpali_infovqa | extension | T2a | 31 | 6 | 0 | 0.011 | yes | no |
| colpali_infovqa | extension | T2b | 24 | 13 | 0 | 0.019 | yes | no |
| colpali_infovqa | extension | T3 | 37 | 0 | 0 | 0.004 | no | yes |
| colqwen2_docvqa | extension | T1 | 37 | 0 | 0 | 0.000 | no | yes |
| colqwen2_docvqa | extension | T2a | 21 | 14 | 2 | 0.036 | no | no |
| colqwen2_docvqa | extension | T2b | 20 | 16 | 1 | 0.024 | no | no |
| colqwen2_docvqa | extension | T3 | 37 | 0 | 0 | 0.005 | no | yes |
| colqwen25_infovqa | extension | T1 | 37 | 0 | 0 | 0.000 | no | yes |
| colqwen25_infovqa | extension | T2a | 30 | 7 | 0 | 0.013 | no | yes |
| colqwen25_infovqa | extension | T2b | 32 | 5 | 0 | 0.011 | no | yes |
| colqwen25_infovqa | extension | T3 | 37 | 0 | 0 | 0.002 | no | yes |

## Gates

| dataset | map | matrix or row-order sha256 | T0 equals the committed store | symmetry check: score difference / largest score | identical per-query metric |
|---|---|---|---|---|---|
| colpali_docvqa | T0 | – | yes | – | – |
| colpali_docvqa | T1 | fe558cab6a3edd9f | – | 2.6e-07 | 1.000 |
| colpali_docvqa | T2a | 479c843b76a8bd17 | – | 2.6e-07 | 1.000 |
| colpali_docvqa | T2b | 2b9074e732df70aa | – | 2.6e-07 | 1.000 |
| colpali_docvqa | T3 | d351fc25bb2c3ce5 | – | 0.0e+00 | 1.000 |
| colqwen2_infovqa | T0 | – | yes | – | – |
| colqwen2_infovqa | T1 | fe558cab6a3edd9f | – | 1.6e-07 | 1.000 |
| colqwen2_infovqa | T2a | 479c843b76a8bd17 | – | 1.6e-07 | 1.000 |
| colqwen2_infovqa | T2b | 2b9074e732df70aa | – | 1.6e-07 | 1.000 |
| colqwen2_infovqa | T3 | 7a66013c9d7e4048 | – | 0.0e+00 | 1.000 |
| colqwen25_docvqa | T0 | – | yes | – | – |
| colqwen25_docvqa | T1 | fe558cab6a3edd9f | – | 1.2e-07 | 1.000 |
| colqwen25_docvqa | T2a | 479c843b76a8bd17 | – | 1.8e-07 | 1.000 |
| colqwen25_docvqa | T2b | 2b9074e732df70aa | – | 2.4e-07 | 1.000 |
| colqwen25_docvqa | T3 | 4f9af373c1e1d2c2 | – | 0.0e+00 | 1.000 |
| scifact | T0 | – | yes | – | – |
| scifact | T1 | 9bbb554a5c7a5a01 | – | 3.6e-07 | 1.000 |
| scifact | T2a | 61d2225c05ee16cd | – | 3.6e-07 | 1.000 |
| scifact | T2b | 5d8a7c858243a66e | – | 3.6e-07 | 1.000 |
| scifact | T3 | d945bf3c73b3bcbf | – | 0.0e+00 | 1.000 |
| colpali_infovqa | T0 | – | yes | – | – |
| colpali_infovqa | T1 | fe558cab6a3edd9f | – | 1.7e-07 | 1.000 |
| colpali_infovqa | T2a | 479c843b76a8bd17 | – | 2.2e-07 | 1.000 |
| colpali_infovqa | T2b | 2b9074e732df70aa | – | 1.7e-07 | 1.000 |
| colpali_infovqa | T3 | d351fc25bb2c3ce5 | – | 0.0e+00 | 1.000 |
| colqwen2_docvqa | T0 | – | yes | – | – |
| colqwen2_docvqa | T1 | fe558cab6a3edd9f | – | 1.7e-07 | 1.000 |
| colqwen2_docvqa | T2a | 479c843b76a8bd17 | – | 2.2e-07 | 1.000 |
| colqwen2_docvqa | T2b | 2b9074e732df70aa | – | 2.2e-07 | 1.000 |
| colqwen2_docvqa | T3 | 4f9af373c1e1d2c2 | – | 0.0e+00 | 1.000 |
| colqwen25_infovqa | T0 | – | yes | – | – |
| colqwen25_infovqa | T1 | fe558cab6a3edd9f | – | 1.6e-07 | 1.000 |
| colqwen25_infovqa | T2a | 479c843b76a8bd17 | – | 1.6e-07 | 1.000 |
| colqwen25_infovqa | T2b | 2b9074e732df70aa | – | 1.6e-07 | 1.000 |
| colqwen25_infovqa | T3 | 7a66013c9d7e4048 | – | 0.0e+00 | 1.000 |

## Secondary

| dataset | map | R_A invariant | R_A small | R_A material | largest change in R_A |
|---|---|---|---|---|---|
| colpali_docvqa | T1 | 37 | 0 | 0 | 0.000 |
| colpali_docvqa | T2a | 18 | 14 | 5 | 0.079 |
| colpali_docvqa | T2b | 15 | 16 | 6 | 0.072 |
| colpali_docvqa | T3 | 34 | 3 | 0 | 0.012 |
| colqwen2_infovqa | T1 | 37 | 0 | 0 | 0.000 |
| colqwen2_infovqa | T2a | 32 | 5 | 0 | 0.010 |
| colqwen2_infovqa | T2b | 35 | 2 | 0 | 0.009 |
| colqwen2_infovqa | T3 | 37 | 0 | 0 | 0.002 |
| colqwen25_docvqa | T1 | 37 | 0 | 0 | 0.001 |
| colqwen25_docvqa | T2a | 19 | 18 | 0 | 0.019 |
| colqwen25_docvqa | T2b | 15 | 18 | 4 | 0.040 |
| colqwen25_docvqa | T3 | 37 | 0 | 0 | 0.005 |
| scifact | T1 | 37 | 0 | 0 | 0.000 |
| scifact | T2a | 14 | 22 | 1 | 0.043 |
| scifact | T2b | 13 | 18 | 6 | 0.058 |
| scifact | T3 | 35 | 2 | 0 | 0.012 |
| colpali_infovqa | T1 | 37 | 0 | 0 | 0.000 |
| colpali_infovqa | T2a | 30 | 7 | 0 | 0.017 |
| colpali_infovqa | T2b | 31 | 6 | 0 | 0.011 |
| colpali_infovqa | T3 | 37 | 0 | 0 | 0.004 |
| colqwen2_docvqa | T1 | 37 | 0 | 0 | 0.000 |
| colqwen2_docvqa | T2a | 20 | 14 | 3 | 0.035 |
| colqwen2_docvqa | T2b | 18 | 16 | 3 | 0.031 |
| colqwen2_docvqa | T3 | 35 | 2 | 0 | 0.008 |
| colqwen25_infovqa | T1 | 37 | 0 | 0 | 0.000 |
| colqwen25_infovqa | T2a | 30 | 7 | 0 | 0.012 |
| colqwen25_infovqa | T2b | 25 | 12 | 0 | 0.013 |
| colqwen25_infovqa | T3 | 37 | 0 | 0 | 0.002 |

The two rotations compared, and overlap@5 and overlap@10 from the identity run (exploratory), are in each dataset's JSON file.
