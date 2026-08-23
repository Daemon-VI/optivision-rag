**Corpus**: 60 pages, 72 queries  
**Encoder**: colsmol (dim 128)  
**Tau**: rank agreement over a pool of 60 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 53.4 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7525 | 0.5000 | 0.9861 | 100.0% | 1.000 | 1.000 | 3.81 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6987 | 0.4306 | 0.9583 | 92.9% | 0.617 | 0.637 | 3.86 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.7576 | 0.5139 | 0.9861 | 100.7% | 0.991 | 0.988 | 3.88 |
| spatial-only | 407.1 | 208.42 | 2.1x | 0.7525 | 0.5000 | 0.9861 | 100.0% | 0.972 | 0.998 | 2.09 |
| spatial+redundancy | 272.6 | 139.56 | 3.2x | 0.7455 | 0.4722 | 0.9861 | 99.1% | 0.889 | 0.895 | 1.33 |
| prune+int8 | 272.6 | 34.89 | 12.8x | 0.7499 | 0.4722 | 1.0000 | 99.7% | 0.887 | 0.889 | 1.26 |
| optivision | 272.6 | 4.36 | 102.7x | 0.6798 | 0.4028 | 0.9583 | 90.3% | 0.602 | 0.573 | 1.31 |
| optivision-aggressive | 185.0 | 2.96 | 151.3x | 0.6734 | 0.3889 | 0.9444 | 89.5% | 0.587 | 0.612 | 1.06 |
| keep-50pct | 299.6 | 4.79 | 93.5x | 0.6764 | 0.4167 | 0.9444 | 89.9% | 0.608 | 0.562 | 1.34 |
| keep-40pct | 275.3 | 4.41 | 101.7x | 0.6784 | 0.3889 | 0.9583 | 90.2% | 0.604 | 0.559 | 1.51 |
| keep-30pct | 242.8 | 3.88 | 115.3x | 0.6942 | 0.4444 | 0.9583 | 92.3% | 0.598 | 0.561 | 1.16 |
| keep-20pct | 205.2 | 3.28 | 136.4x | 0.7023 | 0.4583 | 0.9444 | 93.3% | 0.585 | 0.572 | 1.07 |
| keep-10pct | 157.8 | 2.52 | 177.4x | 0.6843 | 0.4306 | 0.9444 | 90.9% | 0.580 | 0.615 | 0.94 |

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
