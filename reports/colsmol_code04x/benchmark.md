**Corpus**: 60 pages, 72 queries  
**Encoder**: colsmol (dim 128)  
**Tau**: rank agreement over a pool of 60 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 53.2 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7006 | 0.4028 | 0.9722 | 100.0% | 1.000 | 1.000 | 4.07 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6335 | 0.3472 | 0.9306 | 90.4% | 0.643 | 0.684 | 3.97 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.6973 | 0.3889 | 0.9722 | 99.5% | 0.990 | 0.978 | 3.83 |
| spatial-only | 350.2 | 179.30 | 2.5x | 0.6887 | 0.4028 | 0.9583 | 98.3% | 0.940 | 0.950 | 1.74 |
| spatial+redundancy | 246.2 | 126.07 | 3.6x | 0.6867 | 0.4167 | 0.9583 | 98.0% | 0.895 | 0.899 | 1.19 |
| prune+int8 | 246.2 | 31.52 | 14.2x | 0.6848 | 0.4167 | 0.9583 | 97.7% | 0.894 | 0.895 | 1.15 |
| optivision | 246.2 | 3.94 | 113.7x | 0.6236 | 0.3056 | 0.9583 | 89.0% | 0.632 | 0.683 | 1.43 |
| optivision-aggressive | 187.2 | 2.99 | 149.6x | 0.6218 | 0.3194 | 0.9306 | 88.8% | 0.613 | 0.640 | 1.07 |
| keep-50pct | 288.5 | 4.62 | 97.1x | 0.6402 | 0.3472 | 0.9444 | 91.4% | 0.631 | 0.683 | 1.34 |
| keep-40pct | 264.1 | 4.23 | 106.0x | 0.6320 | 0.3333 | 0.9444 | 90.2% | 0.624 | 0.682 | 1.23 |
| keep-30pct | 242.1 | 3.87 | 115.7x | 0.6387 | 0.3472 | 0.9444 | 91.2% | 0.623 | 0.677 | 1.17 |
| keep-20pct | 206.4 | 3.30 | 135.7x | 0.6136 | 0.3333 | 0.9167 | 87.6% | 0.577 | 0.651 | 1.10 |
| keep-10pct | 162.8 | 2.60 | 172.0x | 0.5699 | 0.3056 | 0.8611 | 81.3% | 0.450 | 0.373 | 1.00 |

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
