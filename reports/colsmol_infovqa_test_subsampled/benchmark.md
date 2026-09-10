**Corpus**: 500 pages, 494 queries  
**Encoder**: colsmol (dim 128)  
**Tau**: rank agreement over a pool of 500 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 77.6 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 713.3 | 365.22 | 1.0x | 0.8257 | 0.7672 | 0.8725 | 100.0% | 1.000 | 1.000 | 36.54 |
| binary-only | 713.3 | 11.41 | 32.0x | 0.8091 | 0.7429 | 0.8623 | 98.0% | 0.745 | 0.582 | 34.25 |
| int8-only | 713.3 | 91.31 | 4.0x | 0.8272 | 0.7692 | 0.8745 | 100.2% | 0.994 | 0.985 | 34.09 |
| spatial-only | 695.7 | 356.22 | 1.0x | 0.8259 | 0.7672 | 0.8725 | 100.0% | 0.996 | 0.993 | 33.50 |
| spatial+redundancy | 529.5 | 271.09 | 1.3x | 0.8194 | 0.7530 | 0.8725 | 99.2% | 0.953 | 0.891 | 25.74 |
| prune+int8 | 529.5 | 67.77 | 5.4x | 0.8211 | 0.7551 | 0.8745 | 99.4% | 0.952 | 0.887 | 25.75 |
| optivision | 529.5 | 8.47 | 43.1x | 0.8094 | 0.7368 | 0.8664 | 98.0% | 0.742 | 0.572 | 25.56 |
| optivision-aggressive | 193.3 | 3.09 | 118.1x | 0.7218 | 0.6417 | 0.7915 | 87.4% | 0.634 | 0.468 | 9.66 |
| keep-50pct | 330.3 | 5.28 | 69.1x | 0.7715 | 0.6862 | 0.8441 | 93.4% | 0.688 | 0.536 | 16.15 |
| keep-40pct | 286.8 | 4.59 | 79.6x | 0.7527 | 0.6700 | 0.8239 | 91.2% | 0.670 | 0.512 | 15.63 |
| keep-30pct | 242.4 | 3.88 | 94.2x | 0.7256 | 0.6457 | 0.7915 | 87.9% | 0.649 | 0.495 | 12.11 |
| keep-20pct | 197.0 | 3.15 | 115.9x | 0.6954 | 0.6093 | 0.7652 | 84.2% | 0.621 | 0.468 | 9.89 |
| keep-10pct | 150.7 | 2.41 | 151.5x | 0.6885 | 0.5951 | 0.7652 | 83.4% | 0.585 | 0.464 | 6.04 |
| cb-keep-50pct | 345.5 | 5.53 | 66.1x | 0.7926 | 0.7166 | 0.8583 | 96.0% | 0.718 | 0.554 | 16.94 |
| cb-random-50pct | 344.7 | 5.52 | 66.2x | 0.7874 | 0.7146 | 0.8462 | 95.4% | 0.714 | 0.541 | 16.94 |
| cb-kmeans-50pct | 338.4 | 5.41 | 67.5x | 0.7654 | 0.6903 | 0.8320 | 92.7% | 0.700 | 0.527 | 17.03 |
| cb-keep-30pct | 261.9 | 4.19 | 87.2x | 0.7731 | 0.6883 | 0.8462 | 93.6% | 0.698 | 0.526 | 13.08 |
| cb-random-30pct | 261.2 | 4.18 | 87.4x | 0.7709 | 0.6923 | 0.8360 | 93.4% | 0.692 | 0.523 | 13.00 |
| cb-kmeans-30pct | 255.7 | 4.09 | 89.3x | 0.7471 | 0.6579 | 0.8279 | 90.5% | 0.679 | 0.514 | 12.67 |
| cb-keep-10pct | 160.5 | 2.57 | 142.2x | 0.7128 | 0.6032 | 0.8016 | 86.3% | 0.630 | 0.453 | 6.57 |
| cb-random-10pct | 160.0 | 2.56 | 142.7x | 0.7094 | 0.6134 | 0.7874 | 85.9% | 0.624 | 0.466 | 6.49 |
| cb-kmeans-10pct | 159.5 | 2.55 | 143.1x | 0.6798 | 0.5668 | 0.7794 | 82.3% | 0.610 | 0.422 | 6.39 |

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
- `cb-keep-50pct` — top 50% by codebook wins, binary — matches keep-50pct
- `cb-random-50pct` — control: probes drawn at random, top 50%, binary
- `cb-kmeans-50pct` — control: probes fitted to patch density, top 50%, binary
- `cb-keep-30pct` — top 30% by codebook wins, binary — matches keep-30pct
- `cb-random-30pct` — control: probes drawn at random, top 30%, binary
- `cb-kmeans-30pct` — control: probes fitted to patch density, top 30%, binary
- `cb-keep-10pct` — top 10% by codebook wins, binary — matches keep-10pct
- `cb-random-10pct` — control: probes drawn at random, top 10%, binary
- `cb-kmeans-10pct` — control: probes fitted to patch density, top 10%, binary
