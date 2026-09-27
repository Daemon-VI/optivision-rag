**E2-colpali-infovqa** — 500 docs, 494 queries, reference: labels

| pipeline | vec/doc | dim | bits | KB/doc | x float32 | ndcg@5 | retention [95% CI] | q ms |
|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.8458 | 100.0% [100.0%, 100.0%] | 34.4 |
| adaptive radius=0.9 centred | 570.1 | 128 | 32 | 291.89 | 1.8x | 0.8439 | 99.8% [99.1%, 100.4%] | 25.0 |
| adaptive radius=0.8 centred | 301.2 | 128 | 32 | 154.19 | 3.4x | 0.8443 | 99.8% [99.1%, 100.7%] | 13.7 |
| adaptive radius=0.7 centred | 184.4 | 128 | 32 | 94.40 | 5.6x | 0.8460 | 100.0% [99.1%, 101.0%] | 8.9 |
| adaptive radius=0.6 centred | 124.3 | 128 | 32 | 63.65 | 8.3x | 0.8377 | 99.0% [97.7%, 100.4%] | 4.8 |
| adaptive radius=0.5 centred | 88.1 | 128 | 32 | 45.12 | 11.7x | 0.8238 | 97.4% [95.7%, 98.9%] | 3.4 |
| adaptive radius=0.4 centred | 61.5 | 128 | 32 | 31.49 | 16.8x | 0.8046 | 95.1% [93.1%, 97.1%] | 3.3 |
| adaptive radius=0.3 centred | 39.3 | 128 | 32 | 20.14 | 26.2x | 0.7832 | 92.6% [90.5%, 94.7%] | 2.3 |
| adaptive ratio=0.5 centred | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.8437 | 99.7% [99.1%, 100.3%] | 19.8 |
| adaptive ratio=0.25 centred | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.8455 | 100.0% [99.1%, 100.9%] | 8.9 |
| adaptive ratio=0.1 centred | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.8315 | 98.3% [96.7%, 100.0%] | 4.0 |
