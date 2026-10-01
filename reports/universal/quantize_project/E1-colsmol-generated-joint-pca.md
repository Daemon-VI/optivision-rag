**E1-colsmol-generated** — 60 docs, 36 queries, reference: labels

| pipeline | vec/doc | dim | bits | KB/doc | x float32 | ndcg@5 | retention [95% CI] | q ms |
|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 128 | 32 | 448.00 | 1.0x | 0.7751 | 100.0% [100.0%, 100.0%] | 4.4 |
| pca-joint 64 (fit on cal queries) | 875.0 | 64 | 32 | 224.00 | 2.0x | 0.7625 | 98.4% [93.6%, 103.1%] | 3.4 |
| pca-docs 64 (same held-out queries) | 875.0 | 64 | 32 | 224.00 | 2.0x | 0.7486 | 96.6% [89.1%, 103.4%] | 3.8 |
| pca-joint 32 (fit on cal queries) | 875.0 | 32 | 32 | 112.00 | 4.0x | 0.6225 | 80.3% [68.3%, 91.2%] | 3.0 |
| pca-docs 32 (same held-out queries) | 875.0 | 32 | 32 | 112.00 | 4.0x | 0.7098 | 91.6% [80.0%, 101.4%] | 3.3 |
