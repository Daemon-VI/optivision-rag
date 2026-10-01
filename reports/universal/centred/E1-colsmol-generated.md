**E1-colsmol-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vec/doc | dim | bits | KB/doc | x float32 | ndcg@5 | retention [95% CI] | q ms |
|---|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 128 | 32 | 448.00 | 1.0x | 0.7823 | 100.0% [100.0%, 100.0%] | 7.7 |
| adaptive radius=0.9 centred | 357.9 | 128 | 32 | 183.24 | 2.4x | 0.7556 | 96.6% [90.6%, 102.4%] | 3.6 |
| adaptive radius=0.8 centred | 235.2 | 128 | 32 | 120.40 | 3.7x | 0.7314 | 93.5% [87.9%, 98.7%] | 1.8 |
| adaptive radius=0.7 centred | 187.1 | 128 | 32 | 95.79 | 4.7x | 0.6914 | 88.4% [80.9%, 96.4%] | 1.4 |
| adaptive radius=0.6 centred | 162.2 | 128 | 32 | 83.02 | 5.4x | 0.6658 | 85.1% [77.7%, 92.3%] | 1.2 |
| adaptive radius=0.5 centred | 146.8 | 128 | 32 | 75.19 | 6.0x | 0.6594 | 84.3% [76.7%, 92.0%] | 1.1 |
| adaptive radius=0.4 centred | 135.5 | 128 | 32 | 69.38 | 6.5x | 0.6633 | 84.8% [77.1%, 92.9%] | 1.1 |
| adaptive radius=0.3 centred | 125.7 | 128 | 32 | 64.38 | 7.0x | 0.6264 | 80.1% [72.3%, 88.5%] | 0.9 |
| adaptive ratio=0.5 centred | 491.0 | 128 | 32 | 251.39 | 1.8x | 0.7701 | 98.4% [94.7%, 102.0%] | 3.1 |
| adaptive ratio=0.25 centred | 299.0 | 128 | 32 | 153.09 | 2.9x | 0.7301 | 93.3% [87.1%, 99.7%] | 2.5 |
| adaptive ratio=0.1 centred | 184.0 | 128 | 32 | 94.21 | 4.8x | 0.7007 | 89.6% [81.9%, 97.3%] | 1.2 |
