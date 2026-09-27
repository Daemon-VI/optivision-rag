**E3-colpali-generated** — 60 docs, 36 queries, reference: labels

| pipeline | vec/doc | dim | bits | KB/doc | x float32 | ndcg@5 | retention [95% CI] | q ms |
|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.6841 | 100.0% [100.0%, 100.0%] | 4.0 |
| pca-joint 64 (fit on cal queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.6805 | 99.5% [95.1%, 104.1%] | 3.1 |
| pca-docs 64 (same held-out queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.6595 | 96.4% [90.3%, 102.2%] | 3.2 |
| pca-joint 32 (fit on cal queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.6438 | 94.1% [86.3%, 101.6%] | 2.7 |
| pca-docs 32 (same held-out queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.6517 | 95.3% [86.2%, 104.8%] | 3.0 |
