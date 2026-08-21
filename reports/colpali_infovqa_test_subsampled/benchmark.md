**Corpus**: 500 pages, 494 queries  
**Encoder**: colpali (dim 128)  
**Tau**: rank agreement over a pool of 500 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 97.0 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 527.87 | 1.0x | 0.8458 | 0.7753 | 0.9028 | 100.0% | 1.000 | 1.000 | 59.59 |
| binary-only | 1031.0 | 16.50 | 32.0x | 0.8243 | 0.7530 | 0.8846 | 97.4% | 0.734 | 0.643 | 59.84 |
| int8-only | 1031.0 | 131.97 | 4.0x | 0.8467 | 0.7773 | 0.9028 | 100.1% | 0.994 | 0.987 | 57.19 |
| spatial-only | 1000.2 | 512.11 | 1.0x | 0.8444 | 0.7753 | 0.9008 | 99.8% | 0.992 | 0.986 | 63.37 |
| spatial+redundancy | 618.1 | 316.45 | 1.7x | 0.8434 | 0.7733 | 0.9028 | 99.7% | 0.950 | 0.899 | 28.89 |
| prune+int8 | 618.1 | 79.11 | 6.7x | 0.8418 | 0.7713 | 0.9008 | 99.5% | 0.950 | 0.898 | 30.81 |
| optivision | 618.1 | 9.89 | 53.4x | 0.8225 | 0.7510 | 0.8826 | 97.2% | 0.734 | 0.655 | 29.16 |
| optivision-aggressive | 137.7 | 2.20 | 239.6x | 0.7676 | 0.6761 | 0.8482 | 90.8% | 0.614 | 0.520 | 6.93 |
| keep-50pct | 360.9 | 5.77 | 91.4x | 0.8014 | 0.7227 | 0.8664 | 94.7% | 0.686 | 0.605 | 17.12 |
| keep-40pct | 296.3 | 4.74 | 111.4x | 0.7948 | 0.7065 | 0.8684 | 94.0% | 0.665 | 0.582 | 14.43 |
| keep-30pct | 227.8 | 3.64 | 144.8x | 0.7809 | 0.7004 | 0.8502 | 92.3% | 0.636 | 0.558 | 11.13 |
| keep-20pct | 158.1 | 2.53 | 208.7x | 0.7563 | 0.6640 | 0.8320 | 89.4% | 0.599 | 0.520 | 8.11 |
| keep-10pct | 85.2 | 1.36 | 387.4x | 0.7223 | 0.6275 | 0.8016 | 85.4% | 0.540 | 0.504 | 4.99 |
| cb-keep-50pct | 354.8 | 5.68 | 93.0x | 0.8134 | 0.7368 | 0.8765 | 96.2% | 0.705 | 0.603 | 17.67 |
| cb-random-50pct | 363.6 | 5.82 | 90.7x | 0.8178 | 0.7368 | 0.8866 | 96.7% | 0.704 | 0.601 | 17.38 |
| cb-kmeans-50pct | 309.2 | 4.95 | 106.7x | 0.8052 | 0.7287 | 0.8704 | 95.2% | 0.690 | 0.592 | 15.43 |
| cb-keep-30pct | 246.7 | 3.95 | 133.7x | 0.8113 | 0.7267 | 0.8826 | 95.9% | 0.691 | 0.574 | 11.82 |
| cb-random-30pct | 250.5 | 4.01 | 131.7x | 0.8098 | 0.7105 | 0.8887 | 95.7% | 0.687 | 0.586 | 12.21 |
| cb-kmeans-30pct | 210.1 | 3.36 | 157.0x | 0.8001 | 0.7227 | 0.8623 | 94.6% | 0.679 | 0.580 | 10.29 |
| cb-keep-10pct | 102.8 | 1.64 | 321.0x | 0.7575 | 0.6640 | 0.8320 | 89.6% | 0.612 | 0.502 | 5.40 |
| cb-random-10pct | 102.9 | 1.65 | 320.7x | 0.7669 | 0.6781 | 0.8401 | 90.7% | 0.605 | 0.518 | 5.56 |
| cb-kmeans-10pct | 96.2 | 1.54 | 342.8x | 0.7500 | 0.6457 | 0.8360 | 88.7% | 0.608 | 0.478 | 5.36 |

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
