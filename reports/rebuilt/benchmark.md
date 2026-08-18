**Corpus**: 60 pages, 72 queries  
**Encoder**: colsmol (dim 128)  
**Query encode**: 37.9 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | q ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7823 | 0.5694 | 0.9722 | 100.0% | 1.000 | 3.58 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6875 | 0.4167 | 0.9444 | 87.9% | 0.585 | 4.43 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.7877 | 0.5694 | 0.9861 | 100.7% | 0.972 | 4.29 |
| spatial-only | 356.1 | 182.32 | 2.5x | 0.7602 | 0.5694 | 0.9444 | 97.2% | 0.935 | 2.50 |
| spatial+redundancy | 246.8 | 126.34 | 3.5x | 0.7519 | 0.5139 | 0.9722 | 96.1% | 0.866 | 1.52 |
| prune+int8 | 246.8 | 31.59 | 14.2x | 0.7511 | 0.5139 | 0.9722 | 96.0% | 0.864 | 1.81 |
| optivision | 246.8 | 3.95 | 113.5x | 0.6782 | 0.4028 | 0.9583 | 86.7% | 0.606 | 1.79 |
| optivision-aggressive | 186.3 | 2.98 | 150.3x | 0.6680 | 0.3611 | 0.9722 | 85.4% | 0.602 | 1.29 |
| keep-50pct | 288.1 | 4.61 | 97.2x | 0.6704 | 0.4167 | 0.9444 | 85.7% | 0.548 | 2.04 |
| keep-40pct | 263.6 | 4.22 | 106.2x | 0.6721 | 0.4167 | 0.9583 | 85.9% | 0.561 | 1.88 |
| keep-30pct | 241.6 | 3.87 | 115.9x | 0.6721 | 0.4028 | 0.9583 | 85.9% | 0.588 | 1.78 |
| keep-20pct | 206.1 | 3.30 | 135.9x | 0.6676 | 0.3750 | 0.9722 | 85.3% | 0.586 | 1.27 |
| keep-10pct | 162.9 | 2.61 | 171.8x | 0.6700 | 0.4167 | 0.9306 | 85.6% | 0.551 | 0.83 |

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
