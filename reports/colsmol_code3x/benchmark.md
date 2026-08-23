**Corpus**: 60 pages, 72 queries  
**Encoder**: colsmol (dim 128)  
**Tau**: rank agreement over a pool of 60 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 59.6 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7181 | 0.4444 | 0.9861 | 100.0% | 1.000 | 1.000 | 4.79 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6532 | 0.3889 | 0.9444 | 91.0% | 0.680 | 0.643 | 4.39 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.7189 | 0.4306 | 0.9861 | 100.1% | 0.990 | 0.977 | 4.17 |
| spatial-only | 390.4 | 199.89 | 2.2x | 0.6981 | 0.4306 | 0.9722 | 97.2% | 0.967 | 0.957 | 2.10 |
| spatial+redundancy | 273.9 | 140.24 | 3.2x | 0.7067 | 0.4444 | 0.9722 | 98.4% | 0.900 | 0.824 | 1.63 |
| prune+int8 | 273.9 | 35.06 | 12.8x | 0.7004 | 0.4306 | 0.9722 | 97.5% | 0.899 | 0.824 | 1.80 |
| optivision | 273.9 | 4.38 | 102.2x | 0.6428 | 0.3472 | 0.9306 | 89.5% | 0.670 | 0.662 | 1.45 |
| optivision-aggressive | 186.2 | 2.98 | 150.3x | 0.6338 | 0.3333 | 0.9444 | 88.3% | 0.647 | 0.576 | 1.11 |
| keep-50pct | 307.7 | 4.92 | 91.0x | 0.6397 | 0.3611 | 0.9306 | 89.1% | 0.671 | 0.647 | 1.39 |
| keep-40pct | 278.4 | 4.45 | 100.6x | 0.6445 | 0.3611 | 0.9306 | 89.8% | 0.665 | 0.653 | 1.54 |
| keep-30pct | 246.1 | 3.94 | 113.8x | 0.6686 | 0.3889 | 0.9583 | 93.1% | 0.670 | 0.635 | 1.10 |
| keep-20pct | 206.7 | 3.31 | 135.4x | 0.6409 | 0.3611 | 0.9306 | 89.2% | 0.652 | 0.615 | 1.40 |
| keep-10pct | 162.1 | 2.59 | 172.8x | 0.6315 | 0.3889 | 0.9028 | 87.9% | 0.617 | 0.581 | 1.24 |

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
