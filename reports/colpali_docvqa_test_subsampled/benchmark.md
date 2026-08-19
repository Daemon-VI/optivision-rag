**Corpus**: 500 pages, 451 queries  
**Encoder**: colpali (dim 128)  
**Query encode**: 72.8 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | q ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 527.87 | 1.0x | 0.5841 | 0.4945 | 0.6608 | 100.0% | 1.000 | 55.86 |
| binary-only | 1031.0 | 16.50 | 32.0x | 0.5625 | 0.4789 | 0.6364 | 96.3% | 0.527 | 56.43 |
| int8-only | 1031.0 | 131.97 | 4.0x | 0.5838 | 0.4989 | 0.6563 | 99.9% | 0.919 | 54.97 |
| spatial-only | 846.3 | 433.30 | 1.2x | 0.5794 | 0.4945 | 0.6519 | 99.2% | 0.873 | 44.96 |
| spatial+redundancy | 557.5 | 285.43 | 1.8x | 0.5691 | 0.4856 | 0.6386 | 97.4% | 0.715 | 24.36 |
| prune+int8 | 557.5 | 71.36 | 7.4x | 0.5690 | 0.4878 | 0.6386 | 97.4% | 0.712 | 24.00 |
| optivision | 557.5 | 8.92 | 59.2x | 0.5523 | 0.4545 | 0.6341 | 94.6% | 0.510 | 23.49 |
| optivision-aggressive | 156.1 | 2.50 | 211.3x | 0.4874 | 0.4080 | 0.5610 | 83.4% | 0.409 | 6.79 |
| keep-50pct | 373.9 | 5.98 | 88.2x | 0.5404 | 0.4612 | 0.6142 | 92.5% | 0.486 | 15.46 |
| keep-40pct | 309.0 | 4.94 | 106.8x | 0.5276 | 0.4501 | 0.5987 | 90.3% | 0.459 | 13.32 |
| keep-30pct | 239.8 | 3.84 | 137.6x | 0.5076 | 0.4146 | 0.5942 | 86.9% | 0.455 | 10.24 |
| keep-20pct | 165.3 | 2.64 | 199.6x | 0.4778 | 0.3925 | 0.5543 | 81.8% | 0.398 | 7.02 |
| keep-10pct | 86.0 | 1.38 | 383.6x | 0.3899 | 0.3126 | 0.4634 | 66.7% | 0.393 | 4.45 |

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
