**E2-colpali-infovqa** — 500 docs, 247 queries, reference: labels

| pipeline | vec/doc | dim | bits | KB/doc | x float32 | ndcg@5 | retention [95% CI] | q ms |
|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.8581 | 100.0% [100.0%, 100.0%] | 37.9 |
| pca-joint 64 (fit on cal queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.8363 | 97.5% [95.5%, 99.2%] | 30.3 |
| pca-docs 64 (same held-out queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.8311 | 96.9% [94.8%, 98.7%] | 29.2 |
| pca-joint 32 (fit on cal queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.7454 | 86.9% [82.9%, 90.6%] | 26.3 |
| pca-docs 32 (same held-out queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.7742 | 90.2% [86.5%, 93.7%] | 27.4 |
