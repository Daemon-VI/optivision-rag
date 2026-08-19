**Corpus**: 500 pages, 494 queries  
**Encoder**: colpali (dim 128)  
**Query encode**: 84.3 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | q ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 527.87 | 1.0x | 0.8458 | 0.7753 | 0.9028 | 100.0% | 1.000 | 61.03 |
| binary-only | 1031.0 | 16.50 | 32.0x | 0.8243 | 0.7530 | 0.8846 | 97.4% | 0.643 | 62.92 |
| int8-only | 1031.0 | 131.97 | 4.0x | 0.8467 | 0.7773 | 0.9028 | 100.1% | 0.987 | 54.51 |
| spatial-only | 1000.2 | 512.11 | 1.0x | 0.8444 | 0.7753 | 0.9008 | 99.8% | 0.986 | 60.18 |
| spatial+redundancy | 618.1 | 316.45 | 1.7x | 0.8434 | 0.7733 | 0.9028 | 99.7% | 0.899 | 31.24 |
| prune+int8 | 618.1 | 79.11 | 6.7x | 0.8418 | 0.7713 | 0.9008 | 99.5% | 0.898 | 31.24 |
| optivision | 618.1 | 9.89 | 53.4x | 0.8225 | 0.7510 | 0.8826 | 97.2% | 0.655 | 30.71 |
| optivision-aggressive | 137.7 | 2.20 | 239.6x | 0.7676 | 0.6761 | 0.8482 | 90.8% | 0.520 | 6.79 |
| keep-50pct | 360.9 | 5.77 | 91.4x | 0.8014 | 0.7227 | 0.8664 | 94.7% | 0.605 | 17.09 |
| keep-40pct | 296.3 | 4.74 | 111.4x | 0.7948 | 0.7065 | 0.8684 | 94.0% | 0.582 | 14.03 |
| keep-30pct | 227.8 | 3.64 | 144.8x | 0.7809 | 0.7004 | 0.8502 | 92.3% | 0.558 | 10.91 |
| keep-20pct | 158.1 | 2.53 | 208.7x | 0.7563 | 0.6640 | 0.8320 | 89.4% | 0.520 | 7.83 |
| keep-10pct | 85.2 | 1.36 | 387.4x | 0.7223 | 0.6275 | 0.8016 | 85.4% | 0.504 | 4.66 |

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
