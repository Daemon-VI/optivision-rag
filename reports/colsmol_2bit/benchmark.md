**Corpus**: 60 pages, 72 queries  
**Encoder**: colsmol (dim 128)  
**Tau**: rank agreement over a pool of 60 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 37.9 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7823 | 0.5694 | 0.9722 | 100.0% | 1.000 | 1.000 | 11.50 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6875 | 0.4167 | 0.9444 | 87.9% | 0.643 | 0.585 | 12.47 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.7877 | 0.5694 | 0.9861 | 100.7% | 0.990 | 0.972 | 13.07 |
| lloyd2-only | 875.0 | 28.01 | 16.0x | 0.7497 | 0.5000 | 0.9861 | 95.8% | 0.787 | 0.737 | 10.82 |
| spatial-only | 356.1 | 182.32 | 2.5x | 0.7602 | 0.5694 | 0.9444 | 97.2% | 0.951 | 0.935 | 5.65 |
| spatial+redundancy | 246.8 | 126.34 | 3.5x | 0.7519 | 0.5139 | 0.9722 | 96.1% | 0.887 | 0.866 | 3.55 |
| prune+int8 | 246.8 | 31.59 | 14.2x | 0.7511 | 0.5139 | 0.9722 | 96.0% | 0.886 | 0.864 | 3.60 |
| optivision | 246.8 | 3.95 | 113.5x | 0.6782 | 0.4028 | 0.9583 | 86.7% | 0.624 | 0.606 | 3.95 |
| optivision-aggressive | 186.3 | 2.98 | 150.3x | 0.6680 | 0.3611 | 0.9722 | 85.4% | 0.599 | 0.602 | 2.61 |

- `baseline-float32` — ColPali as published: every patch, full precision
- `binary-only` — quantization alone (32x)
- `int8-only` — scalar quantization alone (4x)
- `lloyd2-only` — rotated 2-bit Lloyd-Max quantization alone (16x)
- `spatial-only` — blank-patch pruning alone
- `spatial+redundancy` — both pruning stages, full precision
- `prune+int8` — pruning with the cheaper quantizer — the quality-first option
- `optivision` — full pipeline: prune + binary
- `optivision-aggressive` — fixed 25% token budget
