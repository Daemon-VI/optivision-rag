**Corpus**: 500 pages, 100 queries  
**Encoder**: colpali (dim 128)  
**Query encode**: 89.2 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | q ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 527.87 | 1.0x | 0.8403 | 0.6500 | 0.9700 | 100.0% | 1.000 | 66.52 |
| binary-only | 1031.0 | 16.50 | 32.0x | 0.8453 | 0.6600 | 0.9700 | 100.6% | 0.736 | 65.79 |
| int8-only | 1031.0 | 131.97 | 4.0x | 0.8540 | 0.6800 | 0.9700 | 101.6% | 0.992 | 60.16 |
| spatial-only | 727.5 | 372.47 | 1.4x | 0.8494 | 0.6700 | 0.9700 | 101.1% | 0.890 | 45.98 |
| spatial+redundancy | 550.6 | 281.92 | 1.9x | 0.8551 | 0.6800 | 0.9700 | 101.8% | 0.868 | 30.24 |
| prune+int8 | 550.6 | 70.48 | 7.5x | 0.8538 | 0.6800 | 0.9700 | 101.6% | 0.865 | 29.69 |
| optivision | 550.6 | 8.81 | 59.9x | 0.8686 | 0.7200 | 0.9700 | 103.4% | 0.696 | 30.08 |
| optivision-aggressive | 173.1 | 2.77 | 190.6x | 0.8158 | 0.6100 | 0.9600 | 97.1% | 0.650 | 9.13 |
| keep-50pct | 408.9 | 6.54 | 80.7x | 0.8556 | 0.7000 | 0.9700 | 101.8% | 0.691 | 20.61 |
| keep-40pct | 336.2 | 5.38 | 98.1x | 0.8424 | 0.6700 | 0.9600 | 100.3% | 0.682 | 17.18 |
| keep-30pct | 259.9 | 4.16 | 126.9x | 0.8299 | 0.6400 | 0.9700 | 98.8% | 0.639 | 13.68 |
| keep-20pct | 180.4 | 2.89 | 182.8x | 0.8121 | 0.6300 | 0.9500 | 96.6% | 0.642 | 10.06 |
| keep-10pct | 96.4 | 1.54 | 342.4x | 0.7545 | 0.5700 | 0.8900 | 89.8% | 0.559 | 6.03 |

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
