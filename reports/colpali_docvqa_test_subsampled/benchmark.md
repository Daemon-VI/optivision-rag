**Corpus**: 500 pages, 451 queries  
**Encoder**: colpali (dim 128)  
**Tau**: rank agreement over a pool of 500 candidates, comparable across runs only at a comparable pool. `Tau(k)` is the superseded top-10 shared-ids statistic, kept because the paper's tables quote it  
**Query encode**: 80.3 ms/query

| Variant | Tok/pg | KB/pg | Compr. | nDCG@5 | R@1 | Hit@5 | Retain | Tau | Tau(k) | q ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 527.87 | 1.0x | 0.5841 | 0.4945 | 0.6608 | 100.0% | 1.000 | 1.000 | 58.82 |
| binary-only | 1031.0 | 16.50 | 32.0x | 0.5625 | 0.4789 | 0.6364 | 96.3% | 0.587 | 0.527 | 58.56 |
| int8-only | 1031.0 | 131.97 | 4.0x | 0.5838 | 0.4989 | 0.6563 | 99.9% | 0.950 | 0.919 | 52.78 |
| spatial-only | 846.3 | 433.30 | 1.2x | 0.5794 | 0.4945 | 0.6519 | 99.2% | 0.925 | 0.873 | 45.79 |
| spatial+redundancy | 557.5 | 285.43 | 1.8x | 0.5691 | 0.4856 | 0.6386 | 97.4% | 0.823 | 0.715 | 25.82 |
| prune+int8 | 557.5 | 71.36 | 7.4x | 0.5690 | 0.4878 | 0.6386 | 97.4% | 0.815 | 0.712 | 26.63 |
| optivision | 557.5 | 8.92 | 59.2x | 0.5523 | 0.4545 | 0.6341 | 94.6% | 0.555 | 0.510 | 25.94 |
| optivision-aggressive | 156.1 | 2.50 | 211.3x | 0.4874 | 0.4080 | 0.5610 | 83.4% | 0.458 | 0.409 | 7.88 |
| keep-50pct | 373.9 | 5.98 | 88.2x | 0.5404 | 0.4612 | 0.6142 | 92.5% | 0.534 | 0.486 | 17.88 |
| keep-40pct | 309.0 | 4.94 | 106.8x | 0.5276 | 0.4501 | 0.5987 | 90.3% | 0.513 | 0.459 | 14.70 |
| keep-30pct | 239.8 | 3.84 | 137.6x | 0.5076 | 0.4146 | 0.5942 | 86.9% | 0.479 | 0.455 | 11.86 |
| keep-20pct | 165.3 | 2.64 | 199.6x | 0.4778 | 0.3925 | 0.5543 | 81.8% | 0.436 | 0.398 | 8.14 |
| keep-10pct | 86.0 | 1.38 | 383.6x | 0.3899 | 0.3126 | 0.4634 | 66.7% | 0.368 | 0.393 | 4.76 |
| cb-keep-50pct | 372.8 | 5.96 | 88.5x | 0.5338 | 0.4523 | 0.6075 | 91.4% | 0.529 | 0.520 | 17.84 |
| cb-random-50pct | 381.9 | 6.11 | 86.4x | 0.5392 | 0.4523 | 0.6186 | 92.3% | 0.526 | 0.470 | 18.47 |
| cb-kmeans-50pct | 322.8 | 5.17 | 102.2x | 0.5071 | 0.4124 | 0.5898 | 86.8% | 0.521 | 0.471 | 15.10 |
| cb-keep-30pct | 254.6 | 4.07 | 129.6x | 0.5150 | 0.4257 | 0.5942 | 88.2% | 0.504 | 0.465 | 12.32 |
| cb-random-30pct | 256.5 | 4.10 | 128.6x | 0.5289 | 0.4523 | 0.6009 | 90.5% | 0.495 | 0.487 | 12.54 |
| cb-kmeans-30pct | 201.6 | 3.23 | 163.7x | 0.4849 | 0.3947 | 0.5654 | 83.0% | 0.500 | 0.469 | 10.29 |
| cb-keep-10pct | 102.5 | 1.64 | 322.0x | 0.4551 | 0.3636 | 0.5366 | 77.9% | 0.428 | 0.432 | 5.39 |
| cb-random-10pct | 102.2 | 1.64 | 322.7x | 0.4493 | 0.3659 | 0.5255 | 76.9% | 0.428 | 0.437 | 5.47 |
| cb-kmeans-10pct | 90.2 | 1.44 | 365.8x | 0.3794 | 0.2971 | 0.4501 | 64.9% | 0.417 | 0.388 | 4.88 |

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
