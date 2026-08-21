**Corpus**: 60 pages, 72 queries  
**Encoder**: colpali (dim 128)  
**Query encode**: 75.1 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | q ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 527.87 | 1.0x | 0.6954 | 0.3750 | 1.0000 | 100.0% | 1.000 | 5.54 |
| binary-only | 1031.0 | 16.50 | 32.0x | 0.6845 | 0.3750 | 1.0000 | 98.4% | 0.762 | 5.41 |
| int8-only | 1031.0 | 131.97 | 4.0x | 0.6954 | 0.3750 | 1.0000 | 100.0% | 0.988 | 5.43 |
| spatial-only | 319.7 | 163.69 | 3.2x | 0.7146 | 0.4444 | 1.0000 | 102.8% | 0.737 | 1.81 |
| spatial+redundancy | 245.3 | 125.58 | 4.2x | 0.7107 | 0.4306 | 1.0000 | 102.2% | 0.742 | 1.44 |
| prune+int8 | 245.3 | 31.39 | 16.8x | 0.7115 | 0.4306 | 1.0000 | 102.3% | 0.735 | 1.51 |
| optivision | 245.3 | 3.92 | 134.5x | 0.6863 | 0.3611 | 1.0000 | 98.7% | 0.614 | 1.48 |
| optivision-aggressive | 150.9 | 2.41 | 218.6x | 0.6734 | 0.3472 | 1.0000 | 96.8% | 0.613 | 1.00 |
| keep-50pct | 366.6 | 5.86 | 90.0x | 0.6870 | 0.3750 | 1.0000 | 98.8% | 0.663 | 1.99 |
| keep-40pct | 304.9 | 4.88 | 108.2x | 0.6984 | 0.4167 | 1.0000 | 100.4% | 0.652 | 1.71 |
| keep-30pct | 245.4 | 3.93 | 134.4x | 0.6931 | 0.4028 | 1.0000 | 99.7% | 0.616 | 1.48 |
| keep-20pct | 171.3 | 2.74 | 192.5x | 0.6789 | 0.3472 | 1.0000 | 97.6% | 0.608 | 1.10 |
| keep-10pct | 94.2 | 1.51 | 350.3x | 0.5437 | 0.3194 | 0.8056 | 78.2% | 0.373 | 0.73 |

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
