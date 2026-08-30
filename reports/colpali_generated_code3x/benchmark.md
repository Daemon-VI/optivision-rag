**Corpus**: 60 pages, 72 queries  
**Encoder**: colpali (dim 128)  
**Tau**: rank agreement over a pool of 60 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 71.0 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 527.87 | 1.0x | 0.6880 | 0.3889 | 1.0000 | 100.0% | 1.000 | 1.000 | 5.38 |
| binary-only | 1031.0 | 16.50 | 32.0x | 0.6992 | 0.4167 | 1.0000 | 101.6% | 0.706 | 0.741 | 5.11 |
| int8-only | 1031.0 | 131.97 | 4.0x | 0.6886 | 0.3889 | 1.0000 | 100.1% | 0.993 | 0.994 | 5.25 |
| lloyd2-only | 1031.0 | 33.00 | 16.0x | 0.6963 | 0.4028 | 1.0000 | 101.2% | 0.837 | 0.822 | 5.36 |
| spatial-only | 364.1 | 186.39 | 2.8x | 0.6940 | 0.3889 | 1.0000 | 100.9% | 0.870 | 0.882 | 1.99 |
| spatial+redundancy | 272.3 | 139.40 | 3.8x | 0.6858 | 0.3611 | 1.0000 | 99.7% | 0.860 | 0.866 | 1.56 |
| prune+int8 | 272.3 | 34.85 | 15.1x | 0.6813 | 0.3472 | 1.0000 | 99.0% | 0.860 | 0.866 | 1.58 |
| optivision | 272.3 | 4.36 | 121.2x | 0.6904 | 0.3750 | 1.0000 | 100.3% | 0.675 | 0.726 | 1.56 |
| optivision-aggressive | 146.3 | 2.34 | 225.4x | 0.6858 | 0.3750 | 1.0000 | 99.7% | 0.654 | 0.698 | 0.92 |
| keep-50pct | 366.8 | 5.87 | 89.9x | 0.6874 | 0.3750 | 1.0000 | 99.9% | 0.677 | 0.715 | 1.98 |
| keep-40pct | 304.6 | 4.87 | 108.3x | 0.7039 | 0.3889 | 1.0000 | 102.3% | 0.683 | 0.698 | 1.63 |
| keep-30pct | 240.3 | 3.84 | 137.3x | 0.6998 | 0.3889 | 1.0000 | 101.7% | 0.676 | 0.716 | 1.37 |
| keep-20pct | 167.0 | 2.67 | 197.6x | 0.6738 | 0.3472 | 1.0000 | 97.9% | 0.660 | 0.714 | 1.02 |
| keep-10pct | 91.3 | 1.46 | 361.5x | 0.4896 | 0.2917 | 0.7917 | 71.2% | 0.596 | 0.379 | 0.69 |

- `baseline-float32` — ColPali as published: every patch, full precision
- `binary-only` — quantization alone (32x)
- `int8-only` — scalar quantization alone (4x)
- `lloyd2-only` — rotated 2-bit Lloyd-Max quantization alone (16x)
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
