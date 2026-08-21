**Corpus**: 60 pages, 72 queries  
**Encoder**: colsmol (dim 128)  
**Tau**: rank agreement over a pool of 60 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 65.7 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7823 | 0.5694 | 0.9722 | 100.0% | 1.000 | 1.000 | 6.98 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6875 | 0.4167 | 0.9444 | 87.9% | 0.643 | 0.585 | 7.88 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.7877 | 0.5694 | 0.9861 | 100.7% | 0.990 | 0.972 | 8.29 |
| spatial-only | 356.1 | 182.32 | 2.5x | 0.7602 | 0.5694 | 0.9444 | 97.2% | 0.951 | 0.935 | 2.26 |
| spatial+redundancy | 246.8 | 126.34 | 3.5x | 0.7519 | 0.5139 | 0.9722 | 96.1% | 0.887 | 0.866 | 2.83 |
| prune+int8 | 246.8 | 31.59 | 14.2x | 0.7511 | 0.5139 | 0.9722 | 96.0% | 0.886 | 0.864 | 2.45 |
| optivision | 246.8 | 3.95 | 113.5x | 0.6782 | 0.4028 | 0.9583 | 86.7% | 0.624 | 0.606 | 2.82 |
| optivision-aggressive | 186.3 | 2.98 | 150.3x | 0.6680 | 0.3611 | 0.9722 | 85.4% | 0.599 | 0.602 | 2.62 |
| keep-50pct | 288.1 | 4.61 | 97.2x | 0.6704 | 0.4167 | 0.9444 | 85.7% | 0.630 | 0.548 | 2.65 |
| keep-40pct | 263.6 | 4.22 | 106.2x | 0.6721 | 0.4167 | 0.9583 | 85.9% | 0.626 | 0.561 | 2.54 |
| keep-30pct | 241.6 | 3.87 | 115.9x | 0.6721 | 0.4028 | 0.9583 | 85.9% | 0.619 | 0.588 | 3.25 |
| keep-20pct | 206.1 | 3.30 | 135.9x | 0.6676 | 0.3750 | 0.9722 | 85.3% | 0.600 | 0.586 | 2.11 |
| keep-10pct | 162.9 | 2.61 | 171.8x | 0.6700 | 0.4167 | 0.9306 | 85.6% | 0.578 | 0.551 | 1.62 |

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
