# X1: label-free reference agreement

Specification: `reports/research/NEXT_PHASE_PREREGISTRATION.md` section 10, commit `73085ab637679f30fffece6632eb7b9e92adff79`. Code commit `d125232be989671af35eb06bf46152fd5e42db8d`. Written by `scripts/research/x1_agreement.py`; every number is in the JSON files beside this report.

**What is measured.** Reference agreement R_A: where a candidate places the document the uncompressed index ranks first. It is not retrieval quality, and nothing below is a statement about relevance. Labelled retention R_L appears only in the registered cross-checks. All seven stores are development data; nothing here bears on transfer to other encoders.

## Summary

- **H4a (conservative proxy): supported.** CP1 (MEASURED) holds; CP2 (SIMULATED) holds over 150 cells, 0 of them re-run with seeds 2000-3999.
- **H4b (not equivalent): supported.** 21 (store, threshold) pairs have free oracles a factor of 2 or more apart, or CP1 is violated.
- **H4c (usable with the pool; SIMULATED, pool-supported): supported.**
- Agreement is materially stricter than labelled retention. Using it cost at least a factor of 2 in compression.

## Gates

| store | G1 hashes | G2 values | G3 identity | G4 reproduces E1 | samples nested |
|---|---|---|---|---|---|
| colpali_docvqa | yes | yes | yes | yes | yes |
| colpali_infovqa | yes | yes | yes | yes | yes |
| colqwen2_docvqa | yes | yes | yes | yes | yes |
| colqwen2_infovqa | yes | yes | yes | yes | yes |
| colqwen25_docvqa | yes | yes | yes | yes | yes |
| colqwen25_infovqa | yes | yes | yes | yes | yes |
| scifact | yes | yes | not applicable | yes | yes |

## A2. Candidates passing, oracles and CP1 (MEASURED)

Compression of the free oracle and of the chain oracle under each construct; 1.0 means none.

| store | threshold | pass R_L | pass R_A | free oracle R_L | free oracle R_A | chain oracle R_L | chain oracle R_A | CP1 holds |
|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.990 | 4 | 1 | 11.770 | 2.000 | 11.770 | 2.000 | yes |
| colpali_docvqa | 0.970 | 14 | 2 | 49.673 | 3.938 | 49.673 | 3.938 | yes |
| colpali_docvqa | 0.950 | 26 | 2 | 125.445 | 3.938 | 125.445 | 3.938 | yes |
| colpali_docvqa | 0.900 | – | 3 | – | 7.824 | – | 7.824 | – |
| colpali_infovqa | 0.990 | 20 | 4 | 72.699 | 7.757 | 72.699 | 7.757 | yes |
| colpali_infovqa | 0.970 | 29 | 13 | 178.939 | 43.375 | 178.939 | 23.182 | yes |
| colpali_infovqa | 0.950 | 35 | 18 | 299.927 | 64.331 | 299.927 | 64.331 | yes |
| colpali_infovqa | 0.900 | – | 35 | – | 299.927 | – | 299.927 | – |
| colqwen2_docvqa | 0.990 | 18 | 2 | 58.929 | 3.938 | 47.567 | 3.938 | yes |
| colqwen2_docvqa | 0.970 | 27 | 2 | 85.267 | 3.938 | 68.088 | 3.938 | yes |
| colqwen2_docvqa | 0.950 | 36 | 3 | 280.916 | 5.024 | 280.916 | 5.024 | yes |
| colqwen2_docvqa | 0.900 | – | 7 | – | 15.085 | – | 15.085 | – |
| colqwen2_infovqa | 0.990 | 20 | 4 | 47.464 | 7.757 | 47.464 | 7.757 | yes |
| colqwen2_infovqa | 0.970 | 34 | 15 | 161.207 | 29.658 | 161.207 | 29.658 | yes |
| colqwen2_infovqa | 0.950 | 37 | 30 | 280.674 | 122.348 | 280.674 | 122.348 | yes |
| colqwen2_infovqa | 0.900 | – | 37 | – | 280.674 | – | 280.674 | – |
| colqwen25_docvqa | 0.990 | 13 | 2 | 43.557 | 3.938 | 43.557 | 3.938 | yes |
| colqwen25_docvqa | 0.970 | 25 | 2 | 122.569 | 3.938 | 64.656 | 3.938 | yes |
| colqwen25_docvqa | 0.950 | 34 | 3 | 280.916 | 7.757 | 280.916 | 7.757 | yes |
| colqwen25_docvqa | 0.900 | – | 5 | – | 11.569 | – | 7.757 | – |
| colqwen25_infovqa | 0.990 | 22 | 5 | 75.956 | 9.553 | 68.030 | 9.553 | yes |
| colqwen25_infovqa | 0.970 | 36 | 14 | 280.674 | 22.771 | 280.674 | 22.771 | yes |
| colqwen25_infovqa | 0.950 | 37 | 29 | 280.674 | 122.348 | 280.674 | 122.348 | yes |
| colqwen25_infovqa | 0.900 | – | 37 | – | 280.674 | – | 280.674 | – |
| scifact | 0.990 | 9 | 1 | 21.016 | 2.000 | 11.796 | 2.000 | yes |
| scifact | 0.970 | 15 | 6 | 23.120 | 9.169 | 23.120 | 9.169 | yes |
| scifact | 0.950 | 18 | 8 | 26.095 | 10.722 | 26.095 | 10.722 | yes |
| scifact | 0.900 | – | 17 | – | 26.095 | – | 26.095 | – |

## H4c. Non-trivial deployment under agreement at the pool-capped budget (SIMULATED, pool-supported)

T_A = 0.95, delta = 0.05.

| store | n | rate [95% interval] | seeds deciding | borderline rule used | at least 0.90 |
|---|---|---|---|---|---|
| colpali_docvqa | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| colpali_infovqa | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| colqwen2_docvqa | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| colqwen2_infovqa | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| colqwen25_docvqa | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| colqwen25_infovqa | 450 | 1.000 [0.998, 1.000] | 2000 | no | yes |
| scifact | 300 | 0.999 [0.996, 1.000] | 2000 | no | yes |

Each pool by its worst store: DocVQA: colpali_docvqa (1.000); InfoVQA: colpali_infovqa (1.000); SciFact: scifact (0.999).

## A3. Non-trivial deployment rate of P0 under agreement, T_A = 0.99, delta = 0.05 (SIMULATED)

An extrapolated column says what P0 would do if queries were drawn from the pool's own N queries. It is not what that many independent queries would give.

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.000 | 0.000 | 0.000 |
| colpali_infovqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.011 | 0.181 | 0.548 |
| colqwen2_infovqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.684 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.998 | 1.000 | 1.000 |
| scifact | 0.000 | 0.000 | 0.000 | 0.000 | – | 0.000 | 0.000 | 0.000 | 0.000 |

## A3. Non-trivial deployment rate of P0 under agreement, T_A = 0.97, delta = 0.05 (SIMULATED)

An extrapolated column says what P0 would do if queries were drawn from the pool's own N queries. It is not what that many independent queries would give.

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.000 | 0.019 | – | 0.641 | – | 0.992 | 1.000 | 1.000 |
| colpali_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 0.000 | 0.211 | – | 0.991 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 0.000 | 0.759 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 0.000 | 0.973 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| scifact | 0.000 | 0.000 | 0.042 | 0.333 | – | 0.844 | 1.000 | 1.000 | 1.000 |

## A3. Non-trivial deployment rate of P0 under agreement, T_A = 0.95, delta = 0.05 (SIMULATED)

An extrapolated column says what P0 would do if queries were drawn from the pool's own N queries. It is not what that many independent queries would give.

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.000 | 0.917 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colpali_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 0.000 | 0.998 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 0.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| scifact | 0.000 | 0.000 | 0.967 | 0.999 | – | 1.000 | 1.000 | 1.000 | 1.000 |

## A3. Non-trivial deployment rate of P0 under agreement, T_A = 0.9, delta = 0.05 (SIMULATED)

An extrapolated column says what P0 would do if queries were drawn from the pool's own N queries. It is not what that many independent queries would give.

| store | n = 50 (pool-supported) | n = 100 (pool-supported) | n = 225 (pool-supported) | n = 300 (pool-supported) | n = 450 (pool-supported) | n = 450 (extrapolated) | n = 900 (extrapolated) | n = 1800 (extrapolated) | n = 3600 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| colpali_docvqa | 0.000 | 0.995 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colpali_infovqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_docvqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen2_infovqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_docvqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| colqwen25_infovqa | 0.000 | 1.000 | 1.000 | – | 1.000 | – | 1.000 | 1.000 | 1.000 |
| scifact | 0.000 | 0.999 | 1.000 | 1.000 | – | 1.000 | 1.000 | 1.000 | 1.000 |

## A3 to A5 at the pool-capped budget, delta = 0.05 (SIMULATED, pool-supported)

Miss rate against R_A; cross-construct miss rate against R_L (condition CP2, an empirical test with no statistical statement behind it).

| store | threshold | median compression, agreement | median compression, labels | median log2 ratio | miss rate vs R_A | cross-construct miss rate vs R_L |
|---|---|---|---|---|---|---|
| colpali_docvqa | 0.990 | 1.000 | 1.000 | 0.000 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colpali_docvqa | 0.970 | 3.938 | 3.938 | -0.978 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colpali_docvqa | 0.950 | 3.938 | 15.410 | -1.968 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colpali_infovqa | 0.990 | 1.000 | 1.000 | 0.000 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colpali_infovqa | 0.970 | 14.029 | 49.673 | -2.238 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colpali_infovqa | 0.950 | 23.182 | 72.699 | -1.649 | 0.003 [0.001, 0.007] | 0.000 [0.000, 0.002] |
| colqwen2_docvqa | 0.990 | 1.000 | 1.000 | 0.000 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colqwen2_docvqa | 0.970 | 3.938 | 7.762 | -0.979 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colqwen2_docvqa | 0.950 | 3.938 | 47.567 | -3.594 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colqwen2_infovqa | 0.990 | 1.000 | 1.000 | 0.000 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colqwen2_infovqa | 0.970 | 13.841 | 39.076 | -1.699 | 0.002 [0.001, 0.005] | 0.000 [0.000, 0.002] |
| colqwen2_infovqa | 0.950 | 29.658 | 122.348 | -1.988 | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.002] |
| colqwen25_docvqa | 0.990 | 1.000 | 1.000 | 0.000 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colqwen25_docvqa | 0.970 | 3.938 | 7.757 | -0.978 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colqwen25_docvqa | 0.950 | 3.938 | 29.711 | -2.915 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colqwen25_infovqa | 0.990 | 1.000 | 1.000 | 0.000 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| colqwen25_infovqa | 0.970 | 9.553 | 63.031 | -2.375 | 0.001 [0.000, 0.004] | 0.000 [0.000, 0.002] |
| colqwen25_infovqa | 0.950 | 22.771 | 122.348 | -2.556 | 0.002 [0.000, 0.004] | 0.000 [0.000, 0.002] |
| scifact | 0.990 | 1.000 | 1.000 | 0.000 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| scifact | 0.970 | 2.000 | 6.615 | -1.726 | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.002] |
| scifact | 0.950 | 7.978 | 13.314 | -0.739 | 0.001 [0.000, 0.004] | 0.000 [0.000, 0.002] |

## N10 and A6

N10: in 26 of 37 pool-supported cells where the labels arm deploys non-trivially, the median certified compression under agreement is at most half that under labels.

Diagnostic only, deciding nothing: with "more than half of the resamples deploy non-trivially" in place of a rate of at least 0.90, the count is 26 of 40.

A6 (exploratory, MEASURED): agreement loss by quartile of the baseline score margin is in each store's JSON file under `A6`. No hypothesis depends on it.
