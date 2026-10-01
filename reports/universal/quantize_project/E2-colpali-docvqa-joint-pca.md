**E2-colpali-docvqa** — 500 docs, 225 queries, reference: labels

| pipeline | vec/doc | dim | bits | KB/doc | x float32 | ndcg@5 | retention [95% CI] | q ms |
|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.5400 | 100.0% [100.0%, 100.0%] | 34.1 |
| pca-joint 64 (fit on cal queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.5289 | 97.9% [95.5%, 100.4%] | 26.0 |
| pca-docs 64 (same held-out queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.5320 | 98.5% [95.9%, 101.1%] | 27.2 |
| pca-joint 32 (fit on cal queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.4459 | 82.6% [77.1%, 87.5%] | 29.2 |
| pca-docs 32 (same held-out queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.4563 | 84.5% [78.8%, 89.8%] | 30.6 |
