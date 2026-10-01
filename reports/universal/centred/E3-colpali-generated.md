**E3-colpali-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vec/doc | dim | bits | KB/doc | x float32 | ndcg@5 | retention [95% CI] | q ms |
|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.6954 | 100.0% [100.0%, 100.0%] | 5.1 |
| adaptive radius=0.9 centred | 615.9 | 128 | 32 | 315.34 | 1.7x | 0.6923 | 99.6% [95.1%, 104.6%] | 2.7 |
| adaptive radius=0.8 centred | 314.2 | 128 | 32 | 160.87 | 3.3x | 0.6883 | 99.0% [92.9%, 105.3%] | 2.2 |
| adaptive radius=0.7 centred | 178.2 | 128 | 32 | 91.21 | 5.8x | 0.7055 | 101.5% [95.5%, 107.7%] | 1.1 |
| adaptive radius=0.6 centred | 113.2 | 128 | 32 | 57.93 | 9.1x | 0.6976 | 100.3% [93.1%, 108.1%] | 0.7 |
| adaptive radius=0.5 centred | 78.3 | 128 | 32 | 40.10 | 13.2x | 0.7008 | 100.8% [94.2%, 108.3%] | 0.7 |
| adaptive radius=0.4 centred | 57.6 | 128 | 32 | 29.48 | 17.9x | 0.7125 | 102.5% [94.3%, 111.9%] | 0.3 |
| adaptive radius=0.3 centred | 41.9 | 128 | 32 | 21.43 | 24.6x | 0.6700 | 96.4% [88.3%, 105.4%] | 0.5 |
| adaptive ratio=0.5 centred | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.6926 | 99.6% [95.1%, 104.6%] | 3.0 |
| adaptive ratio=0.25 centred | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.7046 | 101.3% [96.2%, 107.6%] | 1.2 |
| adaptive ratio=0.1 centred | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.7058 | 101.5% [93.9%, 109.9%] | 0.6 |
