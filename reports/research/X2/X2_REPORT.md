# X2: query budget and certification

Specification: `reports/research/NEXT_PHASE_PREREGISTRATION.md` section 11, commit `73085ab637679f30fffece6632eb7b9e92adff79`. Code commit `e4d48653967b470945a9a70da8c9d51bd0f78da0`. Written by `scripts/research/x2_budget.py`; every number is in the JSON files beside this report.

**How to read the budgets.** Every rate is SIMULATED: resamples of one MEASURED pool. A pool-supported budget does not exceed the pool size. An extrapolated budget does: it says what P0 would do if queries were drawn from this pool's own N queries, and it is not evidence of what that many independent queries would give. The only MEASURED outcome is the full-pool decision.

## Summary

- **H3 (primary; SIMULATED, pool-supported): supported.** Non-trivial deployment rate of P0 under labels at T = 0.97, delta = 0.05, at the pool-capped budget, required to be at least 0.90 on all seven stores.
- **Extension beyond the pools (secondary; simulation only): holds in simulation.**
- Validity: 2800 cells checked against their error level, 0 re-run with seeds 2000-3999, 0 confirmed failures.

## Gates and checks

| store | G1 hashes | G2 values | G4 reproduces E1 (Method C) | E1 comparators reproduced | samples nested | monotonicity violations (labels / agreement) |
|---|---|---|---|---|---|---|
| colpali_docvqa | yes | yes | yes | yes | yes | 0 / 0 |
| colpali_infovqa | yes | yes | yes | yes | yes | 0 / 0 |
| colqwen2_docvqa | yes | yes | yes | yes | yes | 0 / 0 |
| colqwen2_infovqa | yes | yes | yes | yes | yes | 0 / 0 |
| colqwen25_docvqa | yes | yes | yes | yes | yes | 0 / 0 |
| colqwen25_infovqa | yes | yes | yes | yes | yes | 0 / 0 |
| scifact | yes | yes | yes | yes | yes | 0 / 0 |

## H3. Non-trivial deployment at the pool-capped budget (SIMULATED, pool-supported)

P0 under labels, T = 0.97, delta = 0.05.

| store | pool | n | rate [95% interval] | seeds deciding | borderline rule used | at least 0.90 |
|---|---|---|---|---|---|---|
| colpali_docvqa | 451 | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| colpali_infovqa | 494 | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| colqwen2_docvqa | 451 | 450 | 0.998 [0.995, 0.999] | 2000 | no | yes |
| colqwen2_infovqa | 494 | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| colqwen25_docvqa | 451 | 450 | 1.000 [0.997, 1.000] | 2000 | no | yes |
| colqwen25_infovqa | 494 | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| scifact | 300 | 300 | 0.917 [0.905, 0.929] | 2000 | no | yes |

Each pool by its worst store: DocVQA: colqwen2_docvqa (0.998); InfoVQA: colpali_infovqa (1.000); SciFact: scifact (0.917).

## n*, the smallest registered budget from which the rate stays at or above 0.90

P0, delta = 0.05, non-trivial deployment. Each entry carries the class of the budget that decides it; an extrapolated entry is a statement about redrawing the pool's own queries.

| store | construct | threshold | n* | decided at | class |
|---|---|---|---|---|---|
| colpali_docvqa | labels | 0.990 | 1800 | 1800 | extrapolated |
| colpali_docvqa | labels | 0.970 | 450 | 450 | pool-supported |
| colpali_docvqa | labels | 0.950 | 225 | 225 | pool-supported |
| colpali_docvqa | agreement | 0.990 | none | 3600 | extrapolated |
| colpali_docvqa | agreement | 0.970 | 900 | 900 | extrapolated |
| colpali_docvqa | agreement | 0.950 | 225 | 225 | pool-supported |
| colpali_docvqa | agreement | 0.900 | 100 | 100 | pool-supported |
| colpali_infovqa | labels | 0.990 | 900 | 900 | extrapolated |
| colpali_infovqa | labels | 0.970 | 450 | 450 | pool-supported |
| colpali_infovqa | labels | 0.950 | 225 | 225 | pool-supported |
| colpali_infovqa | agreement | 0.990 | 900 | 900 | extrapolated |
| colpali_infovqa | agreement | 0.970 | 225 | 225 | pool-supported |
| colpali_infovqa | agreement | 0.950 | 225 | 225 | pool-supported |
| colpali_infovqa | agreement | 0.900 | 100 | 100 | pool-supported |
| colqwen2_docvqa | labels | 0.990 | 1800 | 1800 | extrapolated |
| colqwen2_docvqa | labels | 0.970 | 450 | 450 | pool-supported |
| colqwen2_docvqa | labels | 0.950 | 225 | 225 | pool-supported |
| colqwen2_docvqa | agreement | 0.990 | none | 3600 | extrapolated |
| colqwen2_docvqa | agreement | 0.970 | 450 | 450 | pool-supported |
| colqwen2_docvqa | agreement | 0.950 | 225 | 225 | pool-supported |
| colqwen2_docvqa | agreement | 0.900 | 100 | 100 | pool-supported |
| colqwen2_infovqa | labels | 0.990 | 900 | 900 | extrapolated |
| colqwen2_infovqa | labels | 0.970 | 225 | 225 | pool-supported |
| colqwen2_infovqa | labels | 0.950 | 225 | 225 | pool-supported |
| colqwen2_infovqa | agreement | 0.990 | 900 | 900 | extrapolated |
| colqwen2_infovqa | agreement | 0.970 | 225 | 225 | pool-supported |
| colqwen2_infovqa | agreement | 0.950 | 225 | 225 | pool-supported |
| colqwen2_infovqa | agreement | 0.900 | 100 | 100 | pool-supported |
| colqwen25_docvqa | labels | 0.990 | 1800 | 1800 | extrapolated |
| colqwen25_docvqa | labels | 0.970 | 450 | 450 | pool-supported |
| colqwen25_docvqa | labels | 0.950 | 225 | 225 | pool-supported |
| colqwen25_docvqa | agreement | 0.990 | 1800 | 1800 | extrapolated |
| colqwen25_docvqa | agreement | 0.970 | 450 | 450 | pool-supported |
| colqwen25_docvqa | agreement | 0.950 | 225 | 225 | pool-supported |
| colqwen25_docvqa | agreement | 0.900 | 100 | 100 | pool-supported |
| colqwen25_infovqa | labels | 0.990 | 900 | 900 | extrapolated |
| colqwen25_infovqa | labels | 0.970 | 225 | 225 | pool-supported |
| colqwen25_infovqa | labels | 0.950 | 225 | 225 | pool-supported |
| colqwen25_infovqa | agreement | 0.990 | 900 | 900 | extrapolated |
| colqwen25_infovqa | agreement | 0.970 | 225 | 225 | pool-supported |
| colqwen25_infovqa | agreement | 0.950 | 225 | 225 | pool-supported |
| colqwen25_infovqa | agreement | 0.900 | 100 | 100 | pool-supported |
| scifact | labels | 0.990 | 1800 | 1800 | extrapolated |
| scifact | labels | 0.970 | 300 | 300 | pool-supported |
| scifact | labels | 0.950 | 225 | 225 | pool-supported |
| scifact | agreement | 0.990 | none | 3600 | extrapolated |
| scifact | agreement | 0.970 | 900 | 900 | extrapolated |
| scifact | agreement | 0.950 | 225 | 225 | pool-supported |
| scifact | agreement | 0.900 | 100 | 100 | pool-supported |

## Cells that used the borderline rule

A rate whose 95% interval contained 0.90 was re-run with seeds 2000-3999; the pooled rate decides.

| store | construct | threshold | delta | deployment | n | rate, 2,000 seeds | rate, 4,000 seeds | at least 0.90 |
|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | agreement | 0.900 | 0.010 | nontrivial | 100 | 0.901 | 0.890 | no |
| colpali_infovqa | agreement | 0.950 | 0.050 | substantial | 225 | 0.891 | 0.897 | no |
| colpali_infovqa | agreement | 0.900 | 0.010 | substantial | 100 | 0.899 | 0.901 | yes |
| colqwen2_docvqa | labels | 0.950 | 0.010 | substantial | 450 | 0.905 | 0.902 | yes |
| colqwen2_infovqa | labels | 0.990 | 0.050 | substantial | 1800 | 0.890 | 0.887 | no |
| colqwen2_infovqa | agreement | 0.950 | 0.050 | substantial | 225 | 0.907 | 0.903 | yes |
| scifact | labels | 0.950 | 0.050 | substantial | 225 | 0.887 | 0.886 | no |

## Non-trivial deployment rate of P0 by budget: labels, threshold 0.99, delta = 0.05 (SIMULATED)

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.000 | 0.000 | – | 0.002 | – | 0.153 | 1.000 | 1.000 |
| colpali_infovqa | 0.000 | 0.000 | 0.000 | – | 0.010 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 0.000 | 0.000 | – | 0.002 | – | 0.360 | 1.000 | 1.000 |
| colqwen2_infovqa | 0.000 | 0.000 | 0.000 | – | 0.004 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 0.000 | 0.003 | – | 0.006 | – | 0.057 | 0.999 | 1.000 |
| colqwen25_infovqa | 0.000 | 0.000 | 0.000 | – | 0.005 | – | 1.000 | 1.000 | 1.000 |
| scifact | 0.000 | 0.000 | 0.000 | 0.000 | – | 0.001 | 0.779 | 1.000 | 1.000 |

## Non-trivial deployment rate of P0 by budget: labels, threshold 0.97, delta = 0.05 (SIMULATED)

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.000 | 0.007 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colpali_infovqa | 0.000 | 0.000 | 0.497 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 0.000 | 0.007 | – | 0.998 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_infovqa | 0.000 | 0.000 | 0.999 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 0.001 | 0.025 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| scifact | 0.000 | 0.000 | 0.059 | 0.917 | – | 1.000 | 1.000 | 1.000 | 1.000 |

## Non-trivial deployment rate of P0 by budget: labels, threshold 0.95, delta = 0.05 (SIMULATED)

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.001 | 0.978 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colpali_infovqa | 0.000 | 0.001 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 0.001 | 0.969 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_infovqa | 0.000 | 0.001 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 0.009 | 0.986 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 0.002 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| scifact | 0.000 | 0.000 | 1.000 | 1.000 | – | 1.000 | 1.000 | 1.000 | 1.000 |

## Non-trivial deployment rate of P0 by budget: agreement, threshold 0.99, delta = 0.05 (SIMULATED)

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.000 | 0.000 | 0.000 |
| colpali_infovqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.011 | 0.181 | 0.548 |
| colqwen2_infovqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.684 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.998 | 1.000 | 1.000 |
| scifact | 0.000 | 0.000 | 0.000 | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.000 |

## Non-trivial deployment rate of P0 by budget: agreement, threshold 0.97, delta = 0.05 (SIMULATED)

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.000 | 0.019 | – | 0.641 | – | 0.992 | 1.000 | 1.000 |
| colpali_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 0.000 | 0.211 | – | 0.991 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 0.000 | 0.759 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 0.000 | 0.973 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| scifact | 0.000 | 0.000 | 0.042 | 0.333 | – | 0.844 | 1.000 | 1.000 | 1.000 |

## Non-trivial deployment rate of P0 by budget: agreement, threshold 0.95, delta = 0.05 (SIMULATED)

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.000 | 0.917 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colpali_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 0.000 | 0.998 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| scifact | 0.000 | 0.000 | 0.967 | 0.999 | – | 1.000 | 1.000 | 1.000 | 1.000 |

## Non-trivial deployment rate of P0 by budget: agreement, threshold 0.9, delta = 0.05 (SIMULATED)

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.995 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colpali_infovqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_infovqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| scifact | 0.000 | 0.999 | 1.000 | 1.000 | – | 1.000 | 1.000 | 1.000 | 1.000 |

## Comparators at the pool-capped budget, labels, delta = 0.05 (SIMULATED, pool-supported)

Each cell: non-trivial deployment rate / median compression / miss rate. Rule A carries no error level; A and B use seeds 0-499 at this budget where it is not an E1 budget.

| store | threshold | P0 | I8 | D | C-split | B | A |
|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.990 | 0.00 / 1.0 / 0.002 | 0.01 / 1.0 / 0.000 | 0.00 / 1.0 / 0.000 | 0.00 / 1.0 / 0.002 | 0.00 / 1.0 / 0.000 | 0.96 / 3.9 / 0.008 |
| colpali_docvqa | 0.970 | 1.00 / 3.9 / 0.001 | 1.00 / 3.9 / 0.000 | 0.64 / 3.9 / 0.001 | 0.34 / 1.0 / 0.001 | 0.50 / 3.9 / 0.000 | 1.00 / 3.9 / 0.004 |
| colpali_docvqa | 0.950 | 1.00 / 15.4 / 0.000 | 1.00 / 3.9 / 0.000 | 1.00 / 12.7 / 0.000 | 0.68 / 7.8 / 0.001 | 0.56 / 3.9 / 0.000 | 1.00 / 12.7 / 0.000 |
| colpali_infovqa | 0.990 | 0.01 / 1.0 / 0.000 | 0.76 / 3.9 / 0.000 | 0.01 / 1.0 / 0.000 | 0.01 / 1.0 / 0.000 | 0.00 / 1.0 / 0.000 | 1.00 / 7.8 / 0.000 |
| colpali_infovqa | 0.970 | 1.00 / 49.7 / 0.001 | 1.00 / 3.9 / 0.000 | 1.00 / 43.4 / 0.000 | 0.78 / 30.4 / 0.001 | 0.57 / 22.0 / 0.000 | 1.00 / 43.4 / 0.002 |
| colpali_infovqa | 0.950 | 1.00 / 72.7 / 0.001 | 1.00 / 3.9 / 0.000 | 1.00 / 72.7 / 0.000 | 0.98 / 72.7 / 0.001 | 0.74 / 72.7 / 0.000 | 1.00 / 72.7 / 0.004 |
| colqwen2_docvqa | 0.990 | 0.00 / 1.0 / 0.000 | 0.06 / 1.0 / 0.000 | 0.02 / 1.0 / 0.000 | 0.02 / 1.0 / 0.000 | 0.00 / 1.0 / 0.000 | 0.58 / 3.9 / 0.000 |
| colqwen2_docvqa | 0.970 | 1.00 / 7.8 / 0.000 | 1.00 / 3.9 / 0.000 | 0.90 / 15.1 / 0.000 | 0.28 / 1.0 / 0.001 | 0.52 / 3.9 / 0.000 | 1.00 / 7.8 / 0.000 |
| colqwen2_docvqa | 0.950 | 1.00 / 47.6 / 0.000 | 1.00 / 3.9 / 0.000 | 1.00 / 47.6 / 0.000 | 0.69 / 15.3 / 0.000 | 0.67 / 15.1 / 0.000 | 1.00 / 20.7 / 0.000 |
| colqwen2_infovqa | 0.990 | 0.00 / 1.0 / 0.000 | 1.00 / 3.9 / 0.000 | 0.00 / 1.0 / 0.000 | 0.01 / 1.0 / 0.000 | 0.01 / 1.0 / 0.000 | 1.00 / 7.8 / 0.006 |
| colqwen2_infovqa | 0.970 | 1.00 / 39.1 / 0.001 | 1.00 / 3.9 / 0.000 | 1.00 / 29.7 / 0.000 | 0.85 / 15.3 / 0.001 | 0.42 / 1.0 / 0.000 | 1.00 / 32.0 / 0.004 |
| colqwen2_infovqa | 0.950 | 1.00 / 122.3 / 0.000 | 1.00 / 3.9 / 0.000 | 1.00 / 114.1 / 0.000 | 0.98 / 63.0 / 0.000 | 0.50 / 28.7 / 0.000 | 1.00 / 114.1 / 0.000 |
| colqwen25_docvqa | 0.990 | 0.01 / 1.0 / 0.000 | 0.00 / 1.0 / 0.000 | 0.00 / 1.0 / 0.001 | 0.01 / 1.0 / 0.002 | 0.00 / 1.0 / 0.000 | 0.67 / 3.9 / 0.002 |
| colqwen25_docvqa | 0.970 | 1.00 / 7.8 / 0.001 | 1.00 / 3.9 / 0.000 | 0.77 / 5.4 / 0.001 | 0.20 / 1.0 / 0.001 | 0.33 / 1.0 / 0.000 | 1.00 / 7.8 / 0.002 |
| colqwen25_docvqa | 0.950 | 1.00 / 29.7 / 0.000 | 1.00 / 3.9 / 0.000 | 1.00 / 22.8 / 0.000 | 0.55 / 7.8 / 0.000 | 0.51 / 7.8 / 0.000 | 1.00 / 16.3 / 0.000 |
| colqwen25_infovqa | 0.990 | 0.01 / 1.0 / 0.000 | 1.00 / 3.9 / 0.000 | 0.00 / 1.0 / 0.000 | 0.01 / 1.0 / 0.000 | 0.00 / 1.0 / 0.000 | 1.00 / 9.6 / 0.002 |
| colqwen25_infovqa | 0.970 | 1.00 / 63.0 / 0.000 | 1.00 / 3.9 / 0.000 | 1.00 / 47.5 / 0.001 | 0.86 / 22.8 / 0.001 | 0.50 / 1.0 / 0.000 | 1.00 / 47.5 / 0.000 |
| colqwen25_infovqa | 0.950 | 1.00 / 122.3 / 0.000 | 1.00 / 3.9 / 0.000 | 1.00 / 122.3 / 0.000 | 0.98 / 76.0 / 0.000 | 0.63 / 76.0 / 0.000 | 1.00 / 122.3 / 0.000 |
| scifact | 0.990 | 0.00 / 1.0 / 0.000 | 0.01 / 1.0 / 0.000 | 0.00 / 1.0 / 0.000 | 0.00 / 1.0 / 0.000 | 0.00 / 1.0 / 0.000 | 0.56 / 3.9 / 0.002 |
| scifact | 0.970 | 0.92 / 6.6 / 0.000 | 1.00 / 3.9 / 0.000 | 0.30 / 1.0 / 0.000 | 0.19 / 1.0 / 0.000 | 0.12 / 1.0 / 0.000 | 1.00 / 9.2 / 0.002 |
| scifact | 0.950 | 1.00 / 13.3 / 0.001 | 1.00 / 3.9 / 0.000 | 1.00 / 21.0 / 0.000 | 0.83 / 13.0 / 0.000 | 0.48 / 1.0 / 0.000 | 1.00 / 13.3 / 0.000 |

## The full-pool decision (MEASURED)

P0 given the whole pool once, in stored order, delta = 0.05. One realisation per row; it has no truth to be checked against.

| store | construct | threshold | queries | deployed candidate | compression |
|---|---|---|---|---|---|
| colpali_docvqa | labels | 0.990 | 451 | none (abstains) | 1.000 |
| colpali_docvqa | labels | 0.970 | 451 | int8(per_vector) | 3.938 |
| colpali_docvqa | labels | 0.950 | 451 | hierarchical_merge(0.33) > int4(mean) | 23.182 |
| colpali_docvqa | agreement | 0.990 | 451 | none (abstains) | 1.000 |
| colpali_docvqa | agreement | 0.970 | 451 | int8(per_vector) | 3.938 |
| colpali_docvqa | agreement | 0.950 | 451 | int8(per_vector) | 3.938 |
| colpali_docvqa | agreement | 0.900 | 451 | int8(per_vector) | 3.938 |
| colpali_infovqa | labels | 0.990 | 494 | none (abstains) | 1.000 |
| colpali_infovqa | labels | 0.970 | 494 | hierarchical_merge(0.15) > int4(mean) | 49.673 |
| colpali_infovqa | labels | 0.950 | 494 | hierarchical_merge(0.1) > int4(mean) | 72.699 |
| colpali_infovqa | agreement | 0.990 | 494 | none (abstains) | 1.000 |
| colpali_infovqa | agreement | 0.970 | 494 | int4(mean) | 7.757 |
| colpali_infovqa | agreement | 0.950 | 494 | hierarchical_merge(0.33) > int4(mean) | 23.182 |
| colpali_infovqa | agreement | 0.900 | 494 | adaptive_merge(0.8) > binary | 109.550 |
| colqwen2_docvqa | labels | 0.990 | 451 | none (abstains) | 1.000 |
| colqwen2_docvqa | labels | 0.970 | 451 | hierarchical_merge(0.25) > int8(per_vector) | 15.085 |
| colqwen2_docvqa | labels | 0.950 | 451 | hierarchical_merge(0.15) > int4(mean) | 47.567 |
| colqwen2_docvqa | agreement | 0.990 | 451 | none (abstains) | 1.000 |
| colqwen2_docvqa | agreement | 0.970 | 451 | int8(per_vector) | 3.938 |
| colqwen2_docvqa | agreement | 0.950 | 451 | int8(per_vector) | 3.938 |
| colqwen2_docvqa | agreement | 0.900 | 451 | hierarchical_merge(0.5) > int8(per_vector) | 7.762 |
| colqwen2_infovqa | labels | 0.990 | 494 | none (abstains) | 1.000 |
| colqwen2_infovqa | labels | 0.970 | 494 | hierarchical_merge(0.15) > int4(mean) | 47.464 |
| colqwen2_infovqa | labels | 0.950 | 494 | adaptive_merge(0.6) > binary | 114.052 |
| colqwen2_infovqa | agreement | 0.990 | 494 | none (abstains) | 1.000 |
| colqwen2_infovqa | agreement | 0.970 | 494 | hierarchical_merge(0.5) > int4(mean) | 15.280 |
| colqwen2_infovqa | agreement | 0.950 | 494 | binary | 32.000 |
| colqwen2_infovqa | agreement | 0.900 | 494 | hierarchical_merge(0.1) > binary | 280.674 |
| colqwen25_docvqa | labels | 0.990 | 451 | none (abstains) | 1.000 |
| colqwen25_docvqa | labels | 0.970 | 451 | adaptive_merge(0.8) > int8(per_vector) | 7.958 |
| colqwen25_docvqa | labels | 0.950 | 451 | hierarchical_merge(0.25) > int4(mean) | 29.711 |
| colqwen25_docvqa | agreement | 0.990 | 451 | none (abstains) | 1.000 |
| colqwen25_docvqa | agreement | 0.970 | 451 | int8(per_vector) | 3.938 |
| colqwen25_docvqa | agreement | 0.950 | 451 | int8(per_vector) | 3.938 |
| colqwen25_docvqa | agreement | 0.900 | 451 | int4(mean) | 7.757 |
| colqwen25_infovqa | labels | 0.990 | 494 | none (abstains) | 1.000 |
| colqwen25_infovqa | labels | 0.970 | 494 | hierarchical_merge(0.1) > int4(mean) | 68.030 |
| colqwen25_infovqa | labels | 0.950 | 494 | hierarchical_merge(0.25) > binary | 122.348 |
| colqwen25_infovqa | agreement | 0.990 | 494 | none (abstains) | 1.000 |
| colqwen25_infovqa | agreement | 0.970 | 494 | adaptive_merge(0.9) > int4(mean) | 9.553 |
| colqwen25_infovqa | agreement | 0.950 | 494 | hierarchical_merge(0.33) > int4(mean) | 22.771 |
| colqwen25_infovqa | agreement | 0.900 | 494 | hierarchical_merge(0.25) > binary | 122.348 |
| scifact | labels | 0.990 | 300 | none (abstains) | 1.000 |
| scifact | labels | 0.970 | 300 | adaptive_merge(0.9) > int8(per_vector) | 6.615 |
| scifact | labels | 0.950 | 300 | adaptive_merge(0.9) > int4(mean) | 12.965 |
| scifact | agreement | 0.990 | 300 | none (abstains) | 1.000 |
| scifact | agreement | 0.970 | 300 | int8(per_vector) | 3.918 |
| scifact | agreement | 0.950 | 300 | adaptive_merge(0.8) > int8(per_vector) | 7.978 |
| scifact | agreement | 0.900 | 300 | hierarchical_merge(0.5) > int4(mean) | 15.328 |
