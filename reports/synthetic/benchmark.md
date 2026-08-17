**Corpus**: 60 pages, 72 queries  
**Encoder**: synthetic (dim 128)  
**Query encode**: 0.0 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | q ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1028.0 | 526.34 | 1.0x | 1.0000 | 1.0000 | 1.0000 | 100.0% | 1.000 | 2.40 |
| binary-only | 1028.0 | 16.45 | 32.0x | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.672 | 2.36 |
| int8-only | 1028.0 | 131.58 | 4.0x | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.934 | 2.29 |
| spatial-only | 316.7 | 162.14 | 3.2x | 1.0000 | 1.0000 | 1.0000 | 100.0% | 1.000 | 0.90 |
| spatial+redundancy | 176.0 | 90.12 | 5.8x | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.984 | 0.55 |
| prune+int8 | 176.0 | 22.53 | 23.4x | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.925 | 0.57 |
| optivision | 176.0 | 2.82 | 186.9x | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.670 | 0.56 |
| optivision-aggressive | 159.1 | 2.55 | 206.7x | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.658 | 0.52 |

- `baseline-float32` — ColPali as published: every patch, full precision
- `binary-only` — quantization alone (32x)
- `int8-only` — scalar quantization alone (4x)
- `spatial-only` — blank-patch pruning alone
- `spatial+redundancy` — both pruning stages, full precision
- `prune+int8` — pruning with the cheaper quantizer — the quality-first option
- `optivision` — full pipeline: prune + binary
- `optivision-aggressive` — fixed 25% token budget
