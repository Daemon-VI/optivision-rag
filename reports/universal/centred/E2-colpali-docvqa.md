**E2-colpali-docvqa** — 500 docs, 451 queries, reference: labels

| pipeline | vec/doc | dim | bits | KB/doc | x float32 | ndcg@5 | retention [95% CI] | q ms |
|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.5841 | 100.0% [100.0%, 100.0%] | 26.6 |
| adaptive radius=0.9 centred | 590.8 | 128 | 32 | 302.47 | 1.7x | 0.5780 | 99.0% [97.4%, 100.5%] | 16.1 |
| adaptive radius=0.8 centred | 319.6 | 128 | 32 | 163.62 | 3.2x | 0.5740 | 98.3% [96.5%, 100.1%] | 8.9 |
| adaptive radius=0.7 centred | 194.6 | 128 | 32 | 99.62 | 5.3x | 0.5611 | 96.1% [93.6%, 98.3%] | 6.3 |
| adaptive radius=0.6 centred | 130.8 | 128 | 32 | 66.99 | 7.9x | 0.5627 | 96.3% [93.8%, 98.9%] | 4.9 |
| adaptive radius=0.5 centred | 93.3 | 128 | 32 | 47.75 | 11.1x | 0.5527 | 94.6% [91.5%, 97.7%] | 3.4 |
| adaptive radius=0.4 centred | 67.6 | 128 | 32 | 34.61 | 15.3x | 0.5446 | 93.2% [90.0%, 96.5%] | 2.9 |
| adaptive radius=0.3 centred | 45.3 | 128 | 32 | 23.21 | 22.7x | 0.5245 | 89.8% [86.0%, 93.8%] | 1.9 |
| adaptive ratio=0.5 centred | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.5810 | 99.5% [98.0%, 100.9%] | 13.6 |
| adaptive ratio=0.25 centred | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.5668 | 97.0% [94.8%, 99.2%] | 6.9 |
| adaptive ratio=0.1 centred | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.5502 | 94.2% [91.2%, 97.0%] | 3.3 |
