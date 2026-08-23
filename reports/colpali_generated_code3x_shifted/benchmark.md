**Corpus**: 60 pages, 72 queries  
**Encoder**: colpali (dim 128)  
**Tau**: rank agreement over a pool of 60 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 76.3 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 527.87 | 1.0x | 0.7194 | 0.4306 | 1.0000 | 100.0% | 1.000 | 1.000 | 5.85 |
| binary-only | 1031.0 | 16.50 | 32.0x | 0.6899 | 0.3611 | 1.0000 | 95.9% | 0.706 | 0.735 | 6.44 |
| int8-only | 1031.0 | 131.97 | 4.0x | 0.7208 | 0.4444 | 1.0000 | 100.2% | 0.993 | 0.986 | 5.86 |
| spatial-only | 387.1 | 198.19 | 2.7x | 0.7063 | 0.4306 | 1.0000 | 98.2% | 0.872 | 0.854 | 2.28 |
| spatial+redundancy | 282.9 | 144.86 | 3.6x | 0.6976 | 0.3889 | 1.0000 | 97.0% | 0.856 | 0.831 | 1.67 |
| prune+int8 | 282.9 | 36.22 | 14.6x | 0.6984 | 0.3889 | 1.0000 | 97.1% | 0.855 | 0.837 | 1.70 |
| optivision | 282.9 | 4.53 | 116.6x | 0.6823 | 0.3333 | 1.0000 | 94.8% | 0.670 | 0.718 | 1.72 |
| optivision-aggressive | 142.2 | 2.27 | 232.0x | 0.7033 | 0.4306 | 1.0000 | 97.8% | 0.647 | 0.623 | 0.97 |
| keep-50pct | 363.1 | 5.81 | 90.9x | 0.6892 | 0.3750 | 1.0000 | 95.8% | 0.672 | 0.720 | 2.10 |
| keep-40pct | 303.0 | 4.85 | 108.9x | 0.6913 | 0.3889 | 1.0000 | 96.1% | 0.668 | 0.727 | 1.78 |
| keep-30pct | 236.2 | 3.78 | 139.7x | 0.7057 | 0.4167 | 1.0000 | 98.1% | 0.656 | 0.728 | 1.46 |
| keep-20pct | 162.7 | 2.60 | 202.8x | 0.7058 | 0.4028 | 1.0000 | 98.1% | 0.629 | 0.667 | 1.12 |
| keep-10pct | 85.5 | 1.37 | 385.9x | 0.4612 | 0.2778 | 0.7361 | 64.1% | 0.572 | 0.372 | 0.72 |

- `baseline-float32` — ColPali as published: every patch, full precision
- `binary-only` — quantization alone (32x)
- `int8-only` — scalar quantization alone (4x)
- `spatial-only` — blank-patch pruning alone
- `spatial+redundancy` — both pruning stages, full precision
- `prune+int8` — pruning with the cheaper quantizer — the quality-first option
- `optivision` — full pipeline: prune + binary
- `optivision-aggressive` — fixed 25% token budget
- `keep-50pct` — top 50% most salient patches, binary
- `keep-40pct` — top 40% most salient patches, binary
- `keep-30pct` — top 30% most salient patches, binary
- `keep-20pct` — top 20% most salient patches, binary
- `keep-10pct` — top 10% most salient patches, binary
