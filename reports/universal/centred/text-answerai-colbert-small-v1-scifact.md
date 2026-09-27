**text-answerai-colbert-small-v1-scifact** — 5183 docs, 300 queries, reference: labels

| pipeline | vec/doc | dim | bits | KB/doc | x float32 | ndcg@5 | retention [95% CI] | q ms |
|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 235.7 | 96 | 32 | 90.49 | 1.0x | 0.7308 | 100.0% [100.0%, 100.0%] | 133.2 |
| adaptive radius=0.9 centred | 139.6 | 96 | 32 | 53.60 | 1.7x | 0.7300 | 99.9% [99.0%, 100.8%] | 88.4 |
| adaptive radius=0.8 centred | 115.7 | 96 | 32 | 44.44 | 2.0x | 0.7247 | 99.2% [98.2%, 100.0%] | 74.4 |
| adaptive radius=0.7 centred | 100.7 | 96 | 32 | 38.67 | 2.3x | 0.7308 | 100.0% [99.0%, 101.0%] | 55.6 |
| adaptive radius=0.6 centred | 86.1 | 96 | 32 | 33.07 | 2.7x | 0.7263 | 99.4% [97.7%, 100.9%] | 46.5 |
| adaptive radius=0.5 centred | 69.4 | 96 | 32 | 26.63 | 3.4x | 0.7054 | 96.5% [94.3%, 98.6%] | 32.1 |
| adaptive radius=0.4 centred | 49.3 | 96 | 32 | 18.92 | 4.8x | 0.6966 | 95.3% [92.6%, 97.9%] | 26.5 |
| adaptive radius=0.3 centred | 29.3 | 96 | 32 | 11.23 | 8.1x | 0.6294 | 86.1% [81.3%, 90.4%] | 26.7 |
| adaptive ratio=0.5 centred | 118.1 | 96 | 32 | 45.34 | 2.0x | 0.7280 | 99.6% [98.6%, 100.6%] | 46.7 |
| adaptive ratio=0.25 centred | 59.3 | 96 | 32 | 22.77 | 4.0x | 0.7011 | 95.9% [93.5%, 98.3%] | 28.8 |
| adaptive ratio=0.1 centred | 24.0 | 96 | 32 | 9.22 | 9.8x | 0.6158 | 84.3% [80.1%, 88.8%] | 15.5 |
