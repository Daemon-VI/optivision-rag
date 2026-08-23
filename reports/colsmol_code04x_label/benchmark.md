**Corpus**: 60 pages, 72 queries  
**Encoder**: colsmol (dim 128)  
**Tau**: rank agreement over a pool of 60 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 78.7 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7489 | 0.4861 | 0.9861 | 100.0% | 1.000 | 1.000 | 8.18 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6736 | 0.3750 | 0.9583 | 89.9% | 0.627 | 0.600 | 8.12 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.7441 | 0.4722 | 0.9861 | 99.4% | 0.990 | 0.979 | 5.85 |
| spatial-only | 350.2 | 179.30 | 2.5x | 0.7243 | 0.4722 | 0.9583 | 96.7% | 0.943 | 0.918 | 2.56 |
| spatial+redundancy | 247.7 | 126.81 | 3.5x | 0.7287 | 0.5000 | 0.9583 | 97.3% | 0.900 | 0.865 | 1.70 |
| prune+int8 | 247.7 | 31.70 | 14.1x | 0.7350 | 0.5139 | 0.9583 | 98.1% | 0.899 | 0.866 | 1.94 |
| optivision | 247.7 | 3.96 | 113.0x | 0.6608 | 0.3611 | 0.9444 | 88.2% | 0.622 | 0.597 | 1.68 |
| optivision-aggressive | 185.9 | 2.98 | 150.6x | 0.6603 | 0.3889 | 0.9444 | 88.2% | 0.583 | 0.605 | 1.49 |
| keep-50pct | 288.1 | 4.61 | 97.2x | 0.6677 | 0.3889 | 0.9444 | 89.2% | 0.620 | 0.589 | 1.92 |
| keep-40pct | 264.0 | 4.22 | 106.1x | 0.6774 | 0.4028 | 0.9444 | 90.5% | 0.615 | 0.602 | 1.65 |
| keep-30pct | 242.2 | 3.87 | 115.6x | 0.6686 | 0.3611 | 0.9583 | 89.3% | 0.610 | 0.622 | 1.13 |
| keep-20pct | 206.4 | 3.30 | 135.6x | 0.6668 | 0.3889 | 0.9444 | 89.0% | 0.588 | 0.638 | 1.58 |
| keep-10pct | 162.6 | 2.60 | 172.2x | 0.5481 | 0.2917 | 0.8194 | 73.2% | 0.421 | 0.407 | 1.02 |

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
