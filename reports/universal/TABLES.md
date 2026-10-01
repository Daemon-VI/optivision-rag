## calibration

**E1-colsmol-generated** — 60 docs, 72 queries, 20 random half/half splits per row

| reference | rule | target | held-out met target | held-out mean | held-out worst | met (labels) | median compression |
|---|---|---|---|---|---|---|---|
| labels | point | 0.99 | 35.0% (20/20 selected) | 95.2% | 77.4% | 35.0% | 8.5x |
| labels | point | 0.97 | 55.0% (20/20 selected) | 95.3% | 77.4% | 55.0% | 12.3x |
| labels | point | 0.95 | 45.0% (20/20 selected) | 92.3% | 76.9% | 45.0% | 17.9x |
| labels | lower_ci | 0.99 | 95.0% (20/20 selected) | 100.6% | 95.5% | 95.0% | 4.0x |
| labels | lower_ci | 0.97 | 80.0% (20/20 selected) | 99.9% | 95.0% | 80.0% | 4.0x |
| labels | lower_ci | 0.95 | 100.0% (20/20 selected) | 99.4% | 95.0% | 100.0% | 4.0x |
| baseline@1 | point | 0.99 | 85.0% (20/20 selected) | 99.7% | 97.9% | 100.0% | 2.0x |
| baseline@1 | point | 0.97 | 100.0% (20/20 selected) | 99.2% | 97.9% | 100.0% | 4.0x |
| baseline@1 | point | 0.95 | 100.0% (20/20 selected) | 99.2% | 97.9% | 100.0% | 4.0x |
| baseline@1 | lower_ci | 0.99 | 85.0% (20/20 selected) | 99.7% | 97.9% | 100.0% | 2.0x |
| baseline@1 | lower_ci | 0.97 | 100.0% (20/20 selected) | 99.7% | 97.9% | 100.0% | 2.0x |
| baseline@1 | lower_ci | 0.95 | 100.0% (20/20 selected) | 99.2% | 97.9% | 100.0% | 4.0x |

Pseudo-query proxy (choose on document fragments, judge on real labelled queries):

| target | chosen on fragments | fragment retention | real-query retention |
|---|---|---|---|
| 0.99 | adaptive_merge(0.6) > binary | 100.0% | 81.6% |
| 0.97 | adaptive_merge(0.5) > binary | 97.1% | 83.2% |
| 0.95 | adaptive_merge(0.5) > binary | 97.1% | 83.2% |

Fixed configurations that exist today (all queries):

| configuration | KB/doc | x float32 | retention (labels) |
|---|---|---|---|
| int8-only | 112.00 | 4.0x | 100.7% |
| binary-only | 14.00 | 32.0x | 87.9% |
| prune+int8 | 31.59 | 14.2x | 96.0% |
| optivision (prune+binary) | 3.95 | 113.5x | 86.7% |

**E2-colpali-docvqa** — 500 docs, 451 queries, 20 random half/half splits per row

| reference | rule | target | held-out met target | held-out mean | held-out worst | met (labels) | median compression |
|---|---|---|---|---|---|---|---|
| labels | point | 0.99 | 80.0% (20/20 selected) | 98.6% | 88.9% | 80.0% | 4.0x |
| labels | point | 0.97 | 30.0% (20/20 selected) | 95.5% | 88.9% | 30.0% | 27.4x |
| labels | point | 0.95 | 45.0% (20/20 selected) | 94.1% | 89.4% | 45.0% | 41.1x |
| labels | lower_ci | 0.99 | 100.0% (20/20 selected) | 99.8% | 99.4% | 100.0% | 2.0x |
| labels | lower_ci | 0.97 | 90.0% (20/20 selected) | 98.9% | 92.8% | 90.0% | 4.0x |
| labels | lower_ci | 0.95 | 80.0% (20/20 selected) | 96.4% | 88.9% | 80.0% | 10.1x |
| baseline@1 | point | 0.99 | 100.0% (20/20 selected) | 99.8% | 99.6% | 100.0% | 2.0x |
| baseline@1 | point | 0.97 | 80.0% (20/20 selected) | 97.7% | 96.3% | 100.0% | 4.0x |
| baseline@1 | point | 0.95 | 100.0% (20/20 selected) | 97.3% | 96.3% | 100.0% | 4.0x |
| baseline@1 | lower_ci | 0.99 | 100.0% (17/20 selected) | 99.8% | 99.6% | 100.0% | 2.0x |
| baseline@1 | lower_ci | 0.97 | 90.0% (20/20 selected) | 99.5% | 96.3% | 100.0% | 2.0x |
| baseline@1 | lower_ci | 0.95 | 100.0% (20/20 selected) | 97.4% | 96.3% | 100.0% | 4.0x |

Pseudo-query proxy (choose on document fragments, judge on real labelled queries):

| target | chosen on fragments | fragment retention | real-query retention |
|---|---|---|---|
| 0.99 | adaptive_merge(0.5) > binary | 100.0% | 87.8% |
| 0.97 | adaptive_merge(0.5) > binary | 100.0% | 87.8% |
| 0.95 | adaptive_merge(0.5) > binary | 100.0% | 87.8% |

Fixed configurations that exist today (all queries):

| configuration | KB/doc | x float32 | retention (labels) |
|---|---|---|---|
| int8-only | 131.97 | 4.0x | 99.9% |
| binary-only | 16.50 | 32.0x | 96.3% |
| prune+int8 | 71.36 | 7.4x | 97.4% |
| optivision (prune+binary) | 8.92 | 59.2x | 94.6% |

**E2-colpali-infovqa** — 500 docs, 494 queries, 20 random half/half splits per row

| reference | rule | target | held-out met target | held-out mean | held-out worst | met (labels) | median compression |
|---|---|---|---|---|---|---|---|
| labels | point | 0.99 | 80.0% (20/20 selected) | 99.4% | 95.9% | 80.0% | 32.6x |
| labels | point | 0.97 | 75.0% (20/20 selected) | 97.4% | 95.7% | 75.0% | 191.8x |
| labels | point | 0.95 | 100.0% (20/20 selected) | 97.1% | 95.7% | 100.0% | 284.8x |
| labels | lower_ci | 0.99 | 95.0% (20/20 selected) | 99.8% | 98.2% | 95.0% | 10.8x |
| labels | lower_ci | 0.97 | 85.0% (20/20 selected) | 99.1% | 95.9% | 85.0% | 35.6x |
| labels | lower_ci | 0.95 | 100.0% (20/20 selected) | 97.4% | 95.7% | 100.0% | 171.9x |
| baseline@1 | point | 0.99 | 60.0% (20/20 selected) | 99.2% | 97.2% | 100.0% | 5.1x |
| baseline@1 | point | 0.97 | 50.0% (20/20 selected) | 97.2% | 93.7% | 100.0% | 19.0x |
| baseline@1 | point | 0.95 | 50.0% (20/20 selected) | 95.0% | 94.1% | 100.0% | 86.0x |
| baseline@1 | lower_ci | 0.99 | 95.0% (20/20 selected) | 99.6% | 98.1% | 100.0% | 5.1x |
| baseline@1 | lower_ci | 0.97 | 100.0% (20/20 selected) | 98.4% | 97.3% | 100.0% | 10.8x |
| baseline@1 | lower_ci | 0.95 | 85.0% (20/20 selected) | 96.8% | 93.7% | 100.0% | 21.5x |

Pseudo-query proxy (choose on document fragments, judge on real labelled queries):

| target | chosen on fragments | fragment retention | real-query retention |
|---|---|---|---|
| 0.99 | adaptive_merge(0.5) > binary | 100.0% | 93.7% |
| 0.97 | adaptive_merge(0.5) > binary | 100.0% | 93.7% |
| 0.95 | adaptive_merge(0.5) > binary | 100.0% | 93.7% |

Fixed configurations that exist today (all queries):

| configuration | KB/doc | x float32 | retention (labels) |
|---|---|---|---|
| int8-only | 131.97 | 4.0x | 100.1% |
| binary-only | 16.50 | 32.0x | 97.4% |
| prune+int8 | 79.12 | 6.7x | 99.5% |
| optivision (prune+binary) | 9.89 | 53.4x | 97.2% |

**E3-colpali-generated** — 60 docs, 72 queries, 20 random half/half splits per row

| reference | rule | target | held-out met target | held-out mean | held-out worst | met (labels) | median compression |
|---|---|---|---|---|---|---|---|
| labels | point | 0.99 | 60.0% (20/20 selected) | 99.3% | 93.4% | 60.0% | 551.2x |
| labels | point | 0.97 | 80.0% (20/20 selected) | 99.2% | 93.4% | 80.0% | 551.2x |
| labels | point | 0.95 | 90.0% (20/20 selected) | 99.5% | 93.4% | 90.0% | 551.2x |
| labels | lower_ci | 0.99 | 66.7% (12/20 selected) | 98.7% | 93.1% | 66.7% | 5.2x |
| labels | lower_ci | 0.97 | 70.0% (20/20 selected) | 98.2% | 92.4% | 70.0% | 25.1x |
| labels | lower_ci | 0.95 | 70.0% (20/20 selected) | 97.4% | 92.4% | 70.0% | 151.1x |
| baseline@1 | point | 0.99 | 90.0% (20/20 selected) | 99.6% | 95.5% | 100.0% | 2.0x |
| baseline@1 | point | 0.97 | 80.0% (20/20 selected) | 97.7% | 95.5% | 100.0% | 5.2x |
| baseline@1 | point | 0.95 | 100.0% (20/20 selected) | 97.8% | 95.5% | 100.0% | 5.2x |
| baseline@1 | lower_ci | 0.99 | 90.0% (20/20 selected) | 99.6% | 95.5% | 100.0% | 2.0x |
| baseline@1 | lower_ci | 0.97 | 90.0% (20/20 selected) | 99.6% | 95.5% | 100.0% | 2.0x |
| baseline@1 | lower_ci | 0.95 | 100.0% (20/20 selected) | 98.6% | 95.5% | 100.0% | 2.0x |

Pseudo-query proxy (choose on document fragments, judge on real labelled queries):

| target | chosen on fragments | fragment retention | real-query retention |
|---|---|---|---|
| 0.99 | adaptive_merge(0.5) > binary | 100.0% | 100.1% |
| 0.97 | adaptive_merge(0.5) > binary | 100.0% | 100.1% |
| 0.95 | adaptive_merge(0.5) > binary | 100.0% | 100.1% |

Fixed configurations that exist today (all queries):

| configuration | KB/doc | x float32 | retention (labels) |
|---|---|---|---|
| int8-only | 131.97 | 4.0x | 100.0% |
| binary-only | 16.50 | 32.0x | 98.4% |
| prune+int8 | 31.41 | 16.8x | 102.3% |
| optivision (prune+binary) | 3.93 | 134.4x | 98.7% |

## centred

**E1-colsmol-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 128 | 32 | 448.00 | 1.0x | 0.7823 | 100.0% [100.0%, 100.0%] |
| adaptive radius=0.9 centred | 357.9 | 128 | 32 | 183.24 | 2.4x | 0.7556 | 96.6% [90.6%, 102.4%] |
| adaptive radius=0.8 centred | 235.2 | 128 | 32 | 120.40 | 3.7x | 0.7314 | 93.5% [87.9%, 98.7%] |
| adaptive radius=0.7 centred | 187.1 | 128 | 32 | 95.79 | 4.7x | 0.6914 | 88.4% [80.9%, 96.4%] |
| adaptive radius=0.6 centred | 162.2 | 128 | 32 | 83.02 | 5.4x | 0.6658 | 85.1% [77.7%, 92.3%] |
| adaptive radius=0.5 centred | 146.8 | 128 | 32 | 75.19 | 6.0x | 0.6594 | 84.3% [76.7%, 92.0%] |
| adaptive radius=0.4 centred | 135.5 | 128 | 32 | 69.38 | 6.5x | 0.6633 | 84.8% [77.1%, 92.9%] |
| adaptive radius=0.3 centred | 125.7 | 128 | 32 | 64.38 | 7.0x | 0.6264 | 80.1% [72.3%, 88.5%] |
| adaptive ratio=0.5 centred | 491.0 | 128 | 32 | 251.39 | 1.8x | 0.7701 | 98.4% [94.7%, 102.0%] |
| adaptive ratio=0.25 centred | 299.0 | 128 | 32 | 153.09 | 2.9x | 0.7301 | 93.3% [87.1%, 99.7%] |
| adaptive ratio=0.1 centred | 184.0 | 128 | 32 | 94.21 | 4.8x | 0.7007 | 89.6% [81.9%, 97.3%] |

**E2-colpali-docvqa** — 500 docs, 451 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.5841 | 100.0% [100.0%, 100.0%] |
| adaptive radius=0.9 centred | 590.8 | 128 | 32 | 302.47 | 1.7x | 0.5780 | 99.0% [97.4%, 100.5%] |
| adaptive radius=0.8 centred | 319.6 | 128 | 32 | 163.62 | 3.2x | 0.5740 | 98.3% [96.5%, 100.1%] |
| adaptive radius=0.7 centred | 194.6 | 128 | 32 | 99.62 | 5.3x | 0.5611 | 96.1% [93.6%, 98.3%] |
| adaptive radius=0.6 centred | 130.8 | 128 | 32 | 66.99 | 7.9x | 0.5627 | 96.3% [93.8%, 98.9%] |
| adaptive radius=0.5 centred | 93.3 | 128 | 32 | 47.75 | 11.1x | 0.5527 | 94.6% [91.5%, 97.7%] |
| adaptive radius=0.4 centred | 67.6 | 128 | 32 | 34.61 | 15.3x | 0.5446 | 93.2% [90.0%, 96.5%] |
| adaptive radius=0.3 centred | 45.3 | 128 | 32 | 23.21 | 22.7x | 0.5245 | 89.8% [86.0%, 93.8%] |
| adaptive ratio=0.5 centred | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.5810 | 99.5% [98.0%, 100.9%] |
| adaptive ratio=0.25 centred | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.5668 | 97.0% [94.8%, 99.2%] |
| adaptive ratio=0.1 centred | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.5502 | 94.2% [91.2%, 97.0%] |

**E2-colpali-infovqa** — 500 docs, 494 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.8458 | 100.0% [100.0%, 100.0%] |
| adaptive radius=0.9 centred | 570.1 | 128 | 32 | 291.89 | 1.8x | 0.8439 | 99.8% [99.1%, 100.4%] |
| adaptive radius=0.8 centred | 301.2 | 128 | 32 | 154.19 | 3.4x | 0.8443 | 99.8% [99.1%, 100.7%] |
| adaptive radius=0.7 centred | 184.4 | 128 | 32 | 94.40 | 5.6x | 0.8460 | 100.0% [99.1%, 101.0%] |
| adaptive radius=0.6 centred | 124.3 | 128 | 32 | 63.65 | 8.3x | 0.8377 | 99.0% [97.7%, 100.4%] |
| adaptive radius=0.5 centred | 88.1 | 128 | 32 | 45.12 | 11.7x | 0.8238 | 97.4% [95.7%, 98.9%] |
| adaptive radius=0.4 centred | 61.5 | 128 | 32 | 31.49 | 16.8x | 0.8046 | 95.1% [93.1%, 97.1%] |
| adaptive radius=0.3 centred | 39.3 | 128 | 32 | 20.14 | 26.2x | 0.7832 | 92.6% [90.5%, 94.7%] |
| adaptive ratio=0.5 centred | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.8437 | 99.7% [99.1%, 100.3%] |
| adaptive ratio=0.25 centred | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.8455 | 100.0% [99.1%, 100.9%] |
| adaptive ratio=0.1 centred | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.8315 | 98.3% [96.7%, 100.0%] |

**E3-colpali-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.6954 | 100.0% [100.0%, 100.0%] |
| adaptive radius=0.9 centred | 615.9 | 128 | 32 | 315.34 | 1.7x | 0.6923 | 99.6% [95.1%, 104.6%] |
| adaptive radius=0.8 centred | 314.2 | 128 | 32 | 160.87 | 3.3x | 0.6883 | 99.0% [92.9%, 105.3%] |
| adaptive radius=0.7 centred | 178.2 | 128 | 32 | 91.21 | 5.8x | 0.7055 | 101.5% [95.5%, 107.7%] |
| adaptive radius=0.6 centred | 113.2 | 128 | 32 | 57.93 | 9.1x | 0.6976 | 100.3% [93.1%, 108.1%] |
| adaptive radius=0.5 centred | 78.3 | 128 | 32 | 40.10 | 13.2x | 0.7008 | 100.8% [94.2%, 108.3%] |
| adaptive radius=0.4 centred | 57.6 | 128 | 32 | 29.48 | 17.9x | 0.7125 | 102.5% [94.3%, 111.9%] |
| adaptive radius=0.3 centred | 41.9 | 128 | 32 | 21.43 | 24.6x | 0.6700 | 96.4% [88.3%, 105.4%] |
| adaptive ratio=0.5 centred | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.6926 | 99.6% [95.1%, 104.6%] |
| adaptive ratio=0.25 centred | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.7046 | 101.3% [96.2%, 107.6%] |
| adaptive ratio=0.1 centred | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.7058 | 101.5% [93.9%, 109.9%] |

**text-answerai-colbert-small-v1-scifact** — 5183 docs, 300 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 235.7 | 96 | 32 | 90.49 | 1.0x | 0.7308 | 100.0% [100.0%, 100.0%] |
| adaptive radius=0.9 centred | 139.6 | 96 | 32 | 53.60 | 1.7x | 0.7300 | 99.9% [99.0%, 100.8%] |
| adaptive radius=0.8 centred | 115.7 | 96 | 32 | 44.44 | 2.0x | 0.7247 | 99.2% [98.2%, 100.0%] |
| adaptive radius=0.7 centred | 100.7 | 96 | 32 | 38.67 | 2.3x | 0.7308 | 100.0% [99.0%, 101.0%] |
| adaptive radius=0.6 centred | 86.1 | 96 | 32 | 33.07 | 2.7x | 0.7263 | 99.4% [97.7%, 100.9%] |
| adaptive radius=0.5 centred | 69.4 | 96 | 32 | 26.63 | 3.4x | 0.7054 | 96.5% [94.3%, 98.6%] |
| adaptive radius=0.4 centred | 49.3 | 96 | 32 | 18.92 | 4.8x | 0.6966 | 95.3% [92.6%, 97.9%] |
| adaptive radius=0.3 centred | 29.3 | 96 | 32 | 11.23 | 8.1x | 0.6294 | 86.1% [81.3%, 90.4%] |
| adaptive ratio=0.5 centred | 118.1 | 96 | 32 | 45.34 | 2.0x | 0.7280 | 99.6% [98.6%, 100.6%] |
| adaptive ratio=0.25 centred | 59.3 | 96 | 32 | 22.77 | 4.0x | 0.7011 | 95.9% [93.5%, 98.3%] |
| adaptive ratio=0.1 centred | 24.0 | 96 | 32 | 9.22 | 9.8x | 0.6158 | 84.3% [80.1%, 88.8%] |

## equivalence

**E1-colsmol-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 128 | 32 | 448.00 | 1.0x | 0.7823 | 100.0% [100.0%, 100.0%] |
| binary-only | 875.0 | 128 | 1 | 14.00 | 32.0x | 0.6875 | 87.9% [87.9%, 87.9%] |
| int8-only | 875.0 | 128 | 8 | 112.00 | 4.0x | 0.7877 | 100.7% [100.7%, 100.7%] |
| lloyd2-only | 875.0 | 128 | 2 | 28.01 | 16.0x | 0.7497 | 95.8% [95.8%, 95.8%] |
| spatial-only | 356.1 | 128 | 32 | 182.32 | 2.5x | 0.7602 | 97.2% [97.2%, 97.2%] |
| spatial+redundancy | 246.8 | 128 | 32 | 126.34 | 3.5x | 0.7519 | 96.1% [96.1%, 96.1%] |
| prune+int8 | 246.8 | 128 | 8 | 31.59 | 14.2x | 0.7511 | 96.0% [96.0%, 96.0%] |
| optivision | 246.8 | 128 | 1 | 3.95 | 113.5x | 0.6782 | 86.7% [86.7%, 86.7%] |
| optivision-aggressive | 186.3 | 128 | 1 | 2.98 | 150.3x | 0.6680 | 85.4% [85.4%, 85.4%] |

**E2-colpali-docvqa** — 500 docs, 451 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.5841 | 100.0% [100.0%, 100.0%] |
| binary-only | 1031.0 | 128 | 1 | 16.50 | 32.0x | 0.5625 | 96.3% [96.3%, 96.3%] |
| int8-only | 1031.0 | 128 | 8 | 131.97 | 4.0x | 0.5838 | 99.9% [99.9%, 99.9%] |
| lloyd2-only | 1031.0 | 128 | 2 | 32.99 | 16.0x | 0.5562 | 95.2% [95.2%, 95.2%] |
| spatial-only | 846.3 | 128 | 32 | 433.30 | 1.2x | 0.5794 | 99.2% [99.2%, 99.2%] |
| spatial+redundancy | 557.5 | 128 | 32 | 285.43 | 1.8x | 0.5691 | 97.4% [97.4%, 97.4%] |
| prune+int8 | 557.5 | 128 | 8 | 71.36 | 7.4x | 0.5689 | 97.4% [97.4%, 97.4%] |
| optivision | 557.5 | 128 | 1 | 8.92 | 59.2x | 0.5525 | 94.6% [94.6%, 94.6%] |
| optivision-aggressive | 156.2 | 128 | 1 | 2.50 | 211.2x | 0.4874 | 83.4% [83.4%, 83.4%] |

**E2-colpali-infovqa** — 500 docs, 494 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.8458 | 100.0% [100.0%, 100.0%] |
| binary-only | 1031.0 | 128 | 1 | 16.50 | 32.0x | 0.8243 | 97.4% [97.4%, 97.4%] |
| int8-only | 1031.0 | 128 | 8 | 131.97 | 4.0x | 0.8467 | 100.1% [100.1%, 100.1%] |
| lloyd2-only | 1031.0 | 128 | 2 | 32.99 | 16.0x | 0.8412 | 99.4% [99.4%, 99.4%] |
| spatial-only | 1000.2 | 128 | 32 | 512.11 | 1.0x | 0.8444 | 99.8% [99.8%, 99.8%] |
| spatial+redundancy | 618.1 | 128 | 32 | 316.46 | 1.7x | 0.8434 | 99.7% [99.7%, 99.7%] |
| prune+int8 | 618.1 | 128 | 8 | 79.12 | 6.7x | 0.8418 | 99.5% [99.5%, 99.5%] |
| optivision | 618.1 | 128 | 1 | 9.89 | 53.4x | 0.8225 | 97.2% [97.2%, 97.2%] |
| optivision-aggressive | 137.7 | 128 | 1 | 2.20 | 239.6x | 0.7676 | 90.7% [90.7%, 90.7%] |

**E3-colpali-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.6954 | 100.0% [100.0%, 100.0%] |
| binary-only | 1031.0 | 128 | 1 | 16.50 | 32.0x | 0.6845 | 98.4% [98.4%, 98.4%] |
| int8-only | 1031.0 | 128 | 8 | 131.97 | 4.0x | 0.6954 | 100.0% [100.0%, 100.0%] |
| lloyd2-only | 1031.0 | 128 | 2 | 33.00 | 16.0x | 0.6847 | 98.5% [98.5%, 98.5%] |
| spatial-only | 319.7 | 128 | 32 | 163.69 | 3.2x | 0.7146 | 102.8% [102.8%, 102.8%] |
| spatial+redundancy | 245.4 | 128 | 32 | 125.64 | 4.2x | 0.7107 | 102.2% [102.2%, 102.2%] |
| prune+int8 | 245.4 | 128 | 8 | 31.41 | 16.8x | 0.7115 | 102.3% [102.3%, 102.3%] |
| optivision | 245.4 | 128 | 1 | 3.93 | 134.4x | 0.6863 | 98.7% [98.7%, 98.7%] |
| optivision-aggressive | 150.9 | 128 | 1 | 2.41 | 218.6x | 0.6870 | 98.8% [98.8%, 98.8%] |

## merge

**E1-colsmol-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 128 | 32 | 448.00 | 1.0x | 0.7823 | 100.0% [100.0%, 100.0%] |
| spatial+redundancy (legacy) | 246.8 | 128 | 32 | 126.34 | 3.5x | 0.7519 | 96.1% [90.1%, 101.5%] |
| redundancy t=0.95 | 414.6 | 128 | 32 | 212.30 | 2.1x | 0.7771 | 99.3% [94.7%, 103.9%] |
| redundancy t=0.92 | 327.1 | 128 | 32 | 167.48 | 2.7x | 0.7614 | 97.3% [92.9%, 101.9%] |
| redundancy t=0.88 | 263.7 | 128 | 32 | 135.03 | 3.3x | 0.7555 | 96.6% [90.9%, 102.2%] |
| redundancy t=0.85 | 234.9 | 128 | 32 | 120.29 | 3.7x | 0.7425 | 94.9% [88.8%, 100.9%] |
| redundancy t=0.8 | 200.5 | 128 | 32 | 102.66 | 4.4x | 0.7209 | 92.1% [84.7%, 99.9%] |
| redundancy t=0.75 | 182.0 | 128 | 32 | 93.18 | 4.8x | 0.7258 | 92.8% [85.2%, 100.1%] |
| adaptive radius=0.9 | 285.1 | 128 | 32 | 145.95 | 3.1x | 0.7647 | 97.7% [91.7%, 104.2%] |
| adaptive radius=0.85 | 227.3 | 128 | 32 | 116.39 | 3.8x | 0.7542 | 96.4% [90.1%, 102.8%] |
| adaptive radius=0.8 | 195.8 | 128 | 32 | 100.25 | 4.5x | 0.7479 | 95.6% [88.8%, 102.7%] |
| adaptive radius=0.75 | 176.9 | 128 | 32 | 90.58 | 4.9x | 0.6915 | 88.4% [81.5%, 95.0%] |
| adaptive radius=0.7 | 163.9 | 128 | 32 | 83.90 | 5.3x | 0.6933 | 88.6% [81.3%, 95.7%] |
| adaptive radius=0.65 | 154.9 | 128 | 32 | 79.29 | 5.7x | 0.6879 | 87.9% [80.6%, 96.4%] |
| adaptive radius=0.6 | 147.5 | 128 | 32 | 75.53 | 5.9x | 0.6766 | 86.5% [78.7%, 94.2%] |
| adaptive radius=0.5 | 135.9 | 128 | 32 | 69.56 | 6.4x | 0.6598 | 84.3% [77.3%, 91.7%] |
| adaptive ratio=0.5 | 491.0 | 128 | 32 | 251.39 | 1.8x | 0.7701 | 98.4% [95.0%, 101.7%] |
| hierarchical ratio=0.5 | 491.0 | 128 | 32 | 251.39 | 1.8x | 0.7678 | 98.1% [94.5%, 101.5%] |
| adaptive ratio=0.33 | 361.0 | 128 | 32 | 184.83 | 2.4x | 0.7823 | 100.0% [95.6%, 104.5%] |
| hierarchical ratio=0.33 | 361.0 | 128 | 32 | 184.83 | 2.4x | 0.7649 | 97.8% [93.2%, 102.5%] |
| adaptive ratio=0.25 | 299.0 | 128 | 32 | 153.09 | 2.9x | 0.7649 | 97.8% [92.6%, 103.5%] |
| hierarchical ratio=0.25 | 299.0 | 128 | 32 | 153.09 | 2.9x | 0.7795 | 99.6% [94.9%, 104.9%] |
| adaptive ratio=0.2 | 261.0 | 128 | 32 | 133.63 | 3.4x | 0.7532 | 96.3% [90.3%, 102.5%] |
| hierarchical ratio=0.2 | 261.0 | 128 | 32 | 133.63 | 3.4x | 0.7336 | 93.8% [88.1%, 100.0%] |
| adaptive ratio=0.1 | 184.0 | 128 | 32 | 94.21 | 4.8x | 0.7115 | 90.9% [84.8%, 97.5%] |
| hierarchical ratio=0.1 | 184.0 | 128 | 32 | 94.21 | 4.8x | 0.7168 | 91.6% [85.0%, 98.5%] |
| adaptive ratio=0.05 | 146.0 | 128 | 32 | 74.75 | 6.0x | 0.6792 | 86.8% [78.7%, 94.9%] |
| hierarchical ratio=0.05 | 146.0 | 128 | 32 | 74.75 | 6.0x | 0.6819 | 87.2% [79.9%, 94.0%] |
| random ratio=0.5 | 491.0 | 128 | 32 | 251.39 | 1.8x | 0.7326 | 93.6% [87.0%, 101.1%] |
| random ratio=0.25 | 299.0 | 128 | 32 | 153.09 | 2.9x | 0.6894 | 88.1% [80.5%, 96.6%] |
| random ratio=0.1 | 184.0 | 128 | 32 | 94.21 | 4.8x | 0.6474 | 82.7% [75.3%, 91.5%] |

**E2-colpali-docvqa** — 500 docs, 451 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.5841 | 100.0% [100.0%, 100.0%] |
| spatial+redundancy (legacy) | 557.5 | 128 | 32 | 285.43 | 1.8x | 0.5691 | 97.4% [95.6%, 99.1%] |
| redundancy t=0.95 | 800.5 | 128 | 32 | 409.84 | 1.3x | 0.5753 | 98.5% [97.2%, 99.5%] |
| redundancy t=0.92 | 650.4 | 128 | 32 | 333.02 | 1.6x | 0.5696 | 97.5% [95.8%, 99.0%] |
| redundancy t=0.88 | 489.9 | 128 | 32 | 250.84 | 2.1x | 0.5730 | 98.1% [96.3%, 99.8%] |
| redundancy t=0.85 | 399.3 | 128 | 32 | 204.43 | 2.6x | 0.5702 | 97.6% [95.8%, 99.4%] |
| redundancy t=0.8 | 293.2 | 128 | 32 | 150.10 | 3.5x | 0.5671 | 97.1% [94.9%, 99.4%] |
| redundancy t=0.75 | 223.1 | 128 | 32 | 114.23 | 4.6x | 0.5598 | 95.8% [93.4%, 98.1%] |
| adaptive radius=0.9 | 569.7 | 128 | 32 | 291.70 | 1.8x | 0.5701 | 97.6% [95.8%, 99.2%] |
| adaptive radius=0.85 | 406.3 | 128 | 32 | 208.05 | 2.5x | 0.5703 | 97.6% [95.7%, 99.3%] |
| adaptive radius=0.8 | 300.0 | 128 | 32 | 153.62 | 3.4x | 0.5666 | 97.0% [95.1%, 98.6%] |
| adaptive radius=0.75 | 229.5 | 128 | 32 | 117.49 | 4.5x | 0.5615 | 96.1% [93.9%, 98.4%] |
| adaptive radius=0.7 | 181.5 | 128 | 32 | 92.93 | 5.7x | 0.5649 | 96.7% [94.3%, 98.9%] |
| adaptive radius=0.65 | 147.2 | 128 | 32 | 75.35 | 7.0x | 0.5617 | 96.1% [93.8%, 98.6%] |
| adaptive radius=0.6 | 122.2 | 128 | 32 | 62.56 | 8.4x | 0.5618 | 96.2% [93.6%, 99.0%] |
| adaptive radius=0.5 | 87.7 | 128 | 32 | 44.92 | 11.8x | 0.5497 | 94.1% [91.1%, 97.3%] |
| adaptive ratio=0.5 | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.5758 | 98.6% [97.0%, 100.1%] |
| hierarchical ratio=0.5 | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.5833 | 99.9% [98.6%, 101.2%] |
| adaptive ratio=0.33 | 345.0 | 128 | 32 | 176.64 | 3.0x | 0.5717 | 97.9% [95.9%, 99.9%] |
| hierarchical ratio=0.33 | 345.0 | 128 | 32 | 176.64 | 3.0x | 0.5794 | 99.2% [97.4%, 101.0%] |
| adaptive ratio=0.25 | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.5703 | 97.6% [95.6%, 99.7%] |
| hierarchical ratio=0.25 | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.5775 | 98.9% [96.9%, 100.8%] |
| adaptive ratio=0.2 | 212.0 | 128 | 32 | 108.54 | 4.9x | 0.5628 | 96.3% [94.0%, 98.5%] |
| hierarchical ratio=0.2 | 212.0 | 128 | 32 | 108.54 | 4.9x | 0.5700 | 97.6% [95.1%, 100.0%] |
| adaptive ratio=0.1 | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.5575 | 95.4% [92.8%, 98.2%] |
| hierarchical ratio=0.1 | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.5575 | 95.4% [92.8%, 98.2%] |
| adaptive ratio=0.05 | 59.0 | 128 | 32 | 30.21 | 17.5x | 0.5463 | 93.5% [90.0%, 97.2%] |
| hierarchical ratio=0.05 | 59.0 | 128 | 32 | 30.21 | 17.5x | 0.5238 | 89.7% [85.9%, 93.2%] |
| random ratio=0.5 | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.5719 | 97.9% [95.9%, 100.1%] |
| random ratio=0.25 | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.5496 | 94.1% [90.8%, 97.2%] |
| random ratio=0.1 | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.4950 | 84.7% [80.5%, 89.1%] |

**E2-colpali-infovqa** — 500 docs, 494 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.8458 | 100.0% [100.0%, 100.0%] |
| spatial+redundancy (legacy) | 618.1 | 128 | 32 | 316.46 | 1.7x | 0.8434 | 99.7% [99.1%, 100.3%] |
| redundancy t=0.95 | 805.1 | 128 | 32 | 412.23 | 1.3x | 0.8458 | 100.0% [99.8%, 100.2%] |
| redundancy t=0.92 | 633.0 | 128 | 32 | 324.12 | 1.6x | 0.8462 | 100.0% [99.6%, 100.5%] |
| redundancy t=0.88 | 466.0 | 128 | 32 | 238.58 | 2.2x | 0.8483 | 100.3% [99.7%, 100.9%] |
| redundancy t=0.85 | 377.8 | 128 | 32 | 193.44 | 2.7x | 0.8446 | 99.9% [99.2%, 100.6%] |
| redundancy t=0.8 | 276.6 | 128 | 32 | 141.63 | 3.7x | 0.8466 | 100.1% [99.1%, 101.1%] |
| redundancy t=0.75 | 211.5 | 128 | 32 | 108.27 | 4.9x | 0.8408 | 99.4% [98.3%, 100.6%] |
| adaptive radius=0.9 | 546.3 | 128 | 32 | 279.71 | 1.9x | 0.8449 | 99.9% [99.4%, 100.5%] |
| adaptive radius=0.85 | 383.4 | 128 | 32 | 196.32 | 2.7x | 0.8464 | 100.1% [99.3%, 100.8%] |
| adaptive radius=0.8 | 282.6 | 128 | 32 | 144.68 | 3.6x | 0.8446 | 99.8% [99.0%, 100.8%] |
| adaptive radius=0.75 | 217.0 | 128 | 32 | 111.11 | 4.8x | 0.8443 | 99.8% [98.8%, 100.9%] |
| adaptive radius=0.7 | 172.0 | 128 | 32 | 88.08 | 6.0x | 0.8435 | 99.7% [98.6%, 100.8%] |
| adaptive radius=0.65 | 139.7 | 128 | 32 | 71.52 | 7.4x | 0.8427 | 99.6% [98.4%, 100.9%] |
| adaptive radius=0.6 | 115.8 | 128 | 32 | 59.31 | 8.9x | 0.8407 | 99.4% [98.1%, 100.6%] |
| adaptive radius=0.5 | 81.6 | 128 | 32 | 41.76 | 12.6x | 0.8251 | 97.5% [95.7%, 99.4%] |
| adaptive ratio=0.5 | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.8477 | 100.2% [99.7%, 100.8%] |
| hierarchical ratio=0.5 | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.8446 | 99.9% [99.2%, 100.4%] |
| adaptive ratio=0.33 | 345.0 | 128 | 32 | 176.64 | 3.0x | 0.8469 | 100.1% [99.4%, 100.9%] |
| hierarchical ratio=0.33 | 345.0 | 128 | 32 | 176.64 | 3.0x | 0.8409 | 99.4% [98.6%, 100.2%] |
| adaptive ratio=0.25 | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.8454 | 99.9% [99.1%, 100.9%] |
| hierarchical ratio=0.25 | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.8449 | 99.9% [99.0%, 100.8%] |
| adaptive ratio=0.2 | 212.0 | 128 | 32 | 108.54 | 4.9x | 0.8431 | 99.7% [98.6%, 100.7%] |
| hierarchical ratio=0.2 | 212.0 | 128 | 32 | 108.54 | 4.9x | 0.8443 | 99.8% [98.9%, 100.8%] |
| adaptive ratio=0.1 | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.8331 | 98.5% [97.0%, 99.9%] |
| hierarchical ratio=0.1 | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.8376 | 99.0% [97.7%, 100.2%] |
| adaptive ratio=0.05 | 59.0 | 128 | 32 | 30.21 | 17.5x | 0.8000 | 94.6% [92.5%, 96.4%] |
| hierarchical ratio=0.05 | 59.0 | 128 | 32 | 30.21 | 17.5x | 0.8227 | 97.3% [95.7%, 98.9%] |
| random ratio=0.5 | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.8341 | 98.6% [97.4%, 99.8%] |
| random ratio=0.25 | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.8274 | 97.8% [96.2%, 99.3%] |
| random ratio=0.1 | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.7992 | 94.5% [92.2%, 96.5%] |

**E3-colpali-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.6954 | 100.0% [100.0%, 100.0%] |
| spatial+redundancy (legacy) | 245.4 | 128 | 32 | 125.64 | 4.2x | 0.7107 | 102.2% [94.6%, 110.4%] |
| redundancy t=0.95 | 795.0 | 128 | 32 | 407.04 | 1.3x | 0.7052 | 101.4% [98.9%, 104.3%] |
| redundancy t=0.92 | 605.9 | 128 | 32 | 310.20 | 1.7x | 0.6905 | 99.3% [95.5%, 103.3%] |
| redundancy t=0.88 | 414.1 | 128 | 32 | 211.99 | 2.5x | 0.6935 | 99.7% [95.2%, 104.7%] |
| redundancy t=0.85 | 317.7 | 128 | 32 | 162.66 | 3.2x | 0.7054 | 101.4% [95.3%, 108.1%] |
| redundancy t=0.8 | 214.9 | 128 | 32 | 110.04 | 4.8x | 0.6959 | 100.1% [92.8%, 107.7%] |
| redundancy t=0.75 | 156.2 | 128 | 32 | 79.97 | 6.6x | 0.6976 | 100.3% [93.0%, 107.9%] |
| adaptive radius=0.9 | 507.4 | 128 | 32 | 259.76 | 2.0x | 0.6981 | 100.4% [96.1%, 105.1%] |
| adaptive radius=0.85 | 327.4 | 128 | 32 | 167.60 | 3.1x | 0.6836 | 98.3% [93.4%, 103.5%] |
| adaptive radius=0.8 | 224.1 | 128 | 32 | 114.73 | 4.6x | 0.6886 | 99.0% [92.9%, 105.7%] |
| adaptive radius=0.75 | 163.7 | 128 | 32 | 83.83 | 6.3x | 0.6976 | 100.3% [93.9%, 106.8%] |
| adaptive radius=0.7 | 125.6 | 128 | 32 | 64.31 | 8.2x | 0.6941 | 99.8% [93.0%, 107.6%] |
| adaptive radius=0.65 | 100.0 | 128 | 32 | 51.23 | 10.3x | 0.6964 | 100.2% [93.0%, 107.7%] |
| adaptive radius=0.6 | 82.3 | 128 | 32 | 42.12 | 12.5x | 0.7073 | 101.7% [93.6%, 111.0%] |
| adaptive radius=0.5 | 59.9 | 128 | 32 | 30.64 | 17.2x | 0.7059 | 101.5% [94.5%, 109.8%] |
| adaptive ratio=0.5 | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.6963 | 100.1% [96.3%, 104.0%] |
| hierarchical ratio=0.5 | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.6935 | 99.7% [94.8%, 105.0%] |
| adaptive ratio=0.33 | 345.0 | 128 | 32 | 176.64 | 3.0x | 0.6947 | 99.9% [95.1%, 105.2%] |
| hierarchical ratio=0.33 | 345.0 | 128 | 32 | 176.64 | 3.0x | 0.6901 | 99.2% [94.5%, 104.0%] |
| adaptive ratio=0.25 | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.6951 | 100.0% [95.4%, 104.8%] |
| hierarchical ratio=0.25 | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.6911 | 99.4% [94.5%, 104.1%] |
| adaptive ratio=0.2 | 212.0 | 128 | 32 | 108.54 | 4.9x | 0.7011 | 100.8% [94.9%, 107.2%] |
| hierarchical ratio=0.2 | 212.0 | 128 | 32 | 108.54 | 4.9x | 0.6909 | 99.4% [95.1%, 103.7%] |
| adaptive ratio=0.1 | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.7111 | 102.3% [95.2%, 110.6%] |
| hierarchical ratio=0.1 | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.6787 | 97.6% [91.6%, 104.0%] |
| adaptive ratio=0.05 | 59.0 | 128 | 32 | 30.21 | 17.5x | 0.7104 | 102.2% [94.7%, 110.3%] |
| hierarchical ratio=0.05 | 59.0 | 128 | 32 | 30.21 | 17.5x | 0.6900 | 99.2% [93.1%, 106.2%] |
| random ratio=0.5 | 519.0 | 128 | 32 | 265.73 | 2.0x | 0.6754 | 97.1% [92.4%, 102.2%] |
| random ratio=0.25 | 263.0 | 128 | 32 | 134.66 | 3.9x | 0.6879 | 98.9% [91.7%, 106.7%] |
| random ratio=0.1 | 110.0 | 128 | 32 | 56.32 | 9.4x | 0.6388 | 91.9% [83.4%, 101.6%] |

## quantize_project

**E1-colsmol-generated** — 60 docs, 36 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 128 | 32 | 448.00 | 1.0x | 0.7751 | 100.0% [100.0%, 100.0%] |
| pca-joint 64 (fit on cal queries) | 875.0 | 64 | 32 | 224.00 | 2.0x | 0.7625 | 98.4% [93.6%, 103.1%] |
| pca-docs 64 (same held-out queries) | 875.0 | 64 | 32 | 224.00 | 2.0x | 0.7486 | 96.6% [89.1%, 103.4%] |
| pca-joint 32 (fit on cal queries) | 875.0 | 32 | 32 | 112.00 | 4.0x | 0.6225 | 80.3% [68.3%, 91.2%] |
| pca-docs 32 (same held-out queries) | 875.0 | 32 | 32 | 112.00 | 4.0x | 0.7098 | 91.6% [80.0%, 101.4%] |

**E1-colsmol-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 128 | 32 | 448.00 | 1.0x | 0.7823 | 100.0% [100.0%, 100.0%] |
| float16 | 875.0 | 128 | 16 | 224.00 | 2.0x | 0.7823 | 100.0% [100.0%, 100.0%] |
| int8 fixed (legacy) | 875.0 | 128 | 8 | 112.00 | 4.0x | 0.7877 | 100.7% [99.8%, 102.3%] |
| int8 per-vector | 875.0 | 128 | 8.12 | 113.75 | 3.9x | 0.7772 | 99.3% [98.0%, 100.0%] |
| int8 per-dimension | 875.0 | 128 | 8 | 112.01 | 4.0x | 0.7772 | 99.3% [98.0%, 100.0%] |
| int4 per-vector | 875.0 | 128 | 4.12 | 57.75 | 7.8x | 0.7502 | 95.9% [92.2%, 99.8%] |
| lloyd2 (2-bit) | 875.0 | 128 | 2 | 28.01 | 16.0x | 0.7497 | 95.8% [89.8%, 102.2%] |
| lloyd2 (2-bit, seed 1) | 875.0 | 128 | 2 | 28.01 | 16.0x | 0.7397 | 94.6% [88.4%, 101.1%] |
| lloyd2 (2-bit, seed 2) | 875.0 | 128 | 2 | 28.01 | 16.0x | 0.7337 | 93.8% [88.8%, 98.7%] |
| binary | 875.0 | 128 | 1 | 14.00 | 32.0x | 0.6875 | 87.9% [80.9%, 95.7%] |
| binary centred | 875.0 | 128 | 1 | 14.01 | 32.0x | 0.7128 | 91.1% [83.8%, 99.5%] |
| pca-docs 96 | 875.0 | 96 | 32 | 336.00 | 1.3x | 0.7861 | 100.5% [98.2%, 103.0%] |
| pca-docs 64 | 875.0 | 64 | 32 | 224.00 | 2.0x | 0.7839 | 100.2% [96.1%, 104.2%] |
| pca-docs 48 | 875.0 | 48 | 32 | 168.00 | 2.7x | 0.7514 | 96.0% [89.6%, 101.6%] |
| pca-docs 32 | 875.0 | 32 | 32 | 112.00 | 4.0x | 0.7103 | 90.8% [83.3%, 98.5%] |
| random-proj 96 | 875.0 | 96 | 32 | 336.00 | 1.3x | 0.7529 | 96.2% [90.4%, 102.1%] |
| truncate 96 | 875.0 | 96 | 32 | 336.00 | 1.3x | 0.7702 | 98.5% [94.3%, 103.0%] |
| random-proj 64 | 875.0 | 64 | 32 | 224.00 | 2.0x | 0.7028 | 89.8% [83.2%, 96.8%] |
| truncate 64 | 875.0 | 64 | 32 | 224.00 | 2.0x | 0.7267 | 92.9% [87.2%, 99.2%] |
| random-proj 32 | 875.0 | 32 | 32 | 112.00 | 4.0x | 0.6344 | 81.1% [72.9%, 89.2%] |
| truncate 32 | 875.0 | 32 | 32 | 112.00 | 4.0x | 0.6024 | 77.0% [68.1%, 86.6%] |
| ward distance=0.6 (per-doc count) | 276.1 | 128 | 32 | 141.35 | 3.2x | 0.7617 | 97.4% [92.7%, 102.3%] |
| ward distance=0.8 (per-doc count) | 220.3 | 128 | 32 | 112.78 | 4.0x | 0.7361 | 94.1% [88.3%, 99.8%] |
| ward distance=1.0 (per-doc count) | 190.1 | 128 | 32 | 97.31 | 4.6x | 0.7206 | 92.1% [84.6%, 99.5%] |
| ward distance=1.3 (per-doc count) | 166.8 | 128 | 32 | 85.41 | 5.2x | 0.6785 | 86.7% [80.1%, 93.3%] |
| ward distance=1.6 (per-doc count) | 152.3 | 128 | 32 | 77.98 | 5.7x | 0.6753 | 86.3% [78.9%, 93.8%] |
| adaptive radius=0.8 refine=3 | 195.8 | 128 | 32 | 100.25 | 4.5x | 0.7387 | 94.4% [87.5%, 101.8%] |
| adaptive radius=0.7 refine=3 | 163.9 | 128 | 32 | 83.90 | 5.3x | 0.6879 | 87.9% [80.3%, 95.7%] |
| adaptive radius=0.6 refine=3 | 147.5 | 128 | 32 | 75.53 | 5.9x | 0.6758 | 86.4% [78.2%, 94.8%] |
| merge 0.6 > int8 per-vector | 147.5 | 128 | 8.12 | 19.18 | 23.4x | 0.6772 | 86.6% [78.7%, 94.4%] |
| merge 0.6 > int4 | 147.5 | 128 | 4.12 | 9.74 | 46.0x | 0.6691 | 85.5% [77.7%, 93.3%] |
| merge 0.6 > lloyd2 | 147.5 | 128 | 2 | 4.73 | 94.7x | 0.6697 | 85.6% [77.5%, 94.0%] |
| merge 0.6 > binary | 147.5 | 128 | 1 | 2.36 | 189.8x | 0.6385 | 81.6% [74.0%, 90.7%] |
| merge 0.6 > binary centred | 147.5 | 128 | 1 | 2.37 | 189.1x | 0.6603 | 84.4% [76.6%, 92.5%] |
| merge 0.6 > pca-docs 64 > int8 per-vector | 147.5 | 64 | 8.25 | 9.74 | 46.0x | 0.6625 | 84.7% [77.1%, 92.1%] |
| merge 0.6 > pca-docs 64 > binary | 147.5 | 64 | 1 | 1.18 | 379.6x | 0.6511 | 83.2% [74.8%, 91.9%] |

**E2-colpali-docvqa** — 500 docs, 225 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.5400 | 100.0% [100.0%, 100.0%] |
| pca-joint 64 (fit on cal queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.5289 | 97.9% [95.5%, 100.4%] |
| pca-docs 64 (same held-out queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.5320 | 98.5% [95.9%, 101.1%] |
| pca-joint 32 (fit on cal queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.4459 | 82.6% [77.1%, 87.5%] |
| pca-docs 32 (same held-out queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.4563 | 84.5% [78.8%, 89.8%] |

**E2-colpali-docvqa** — 500 docs, 451 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.5841 | 100.0% [100.0%, 100.0%] |
| float16 | 1031.0 | 128 | 16 | 263.94 | 2.0x | 0.5839 | 100.0% [99.8%, 100.0%] |
| int8 fixed (legacy) | 1031.0 | 128 | 8 | 131.97 | 4.0x | 0.5838 | 99.9% [99.1%, 100.7%] |
| int8 per-vector | 1031.0 | 128 | 8.12 | 134.03 | 3.9x | 0.5844 | 100.0% [99.8%, 100.4%] |
| int8 per-dimension | 1031.0 | 128 | 8 | 131.97 | 4.0x | 0.5841 | 100.0% [99.6%, 100.4%] |
| int4 per-vector | 1031.0 | 128 | 4.12 | 68.05 | 7.8x | 0.5695 | 97.5% [95.4%, 99.4%] |
| lloyd2 (2-bit) | 1031.0 | 128 | 2 | 32.99 | 16.0x | 0.5562 | 95.2% [92.5%, 97.7%] |
| lloyd2 (2-bit, seed 1) | 1031.0 | 128 | 2 | 32.99 | 16.0x | 0.5576 | 95.5% [92.6%, 98.1%] |
| lloyd2 (2-bit, seed 2) | 1031.0 | 128 | 2 | 32.99 | 16.0x | 0.5563 | 95.2% [92.5%, 97.7%] |
| binary | 1031.0 | 128 | 1 | 16.50 | 32.0x | 0.5625 | 96.3% [93.7%, 98.8%] |
| binary centred | 1031.0 | 128 | 1 | 16.50 | 32.0x | 0.5558 | 95.2% [92.2%, 98.0%] |
| pca-docs 96 | 1031.0 | 96 | 32 | 395.90 | 1.3x | 0.5805 | 99.4% [98.2%, 100.6%] |
| pca-docs 64 | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.5688 | 97.4% [94.9%, 99.7%] |
| pca-docs 48 | 1031.0 | 48 | 32 | 197.95 | 2.7x | 0.5419 | 92.8% [89.7%, 95.6%] |
| pca-docs 32 | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.4692 | 80.3% [76.2%, 84.4%] |
| random-proj 96 | 1031.0 | 96 | 32 | 395.90 | 1.3x | 0.5729 | 98.1% [95.9%, 100.4%] |
| truncate 96 | 1031.0 | 96 | 32 | 395.90 | 1.3x | 0.5708 | 97.7% [95.4%, 99.9%] |
| random-proj 64 | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.5388 | 92.2% [89.0%, 95.2%] |
| truncate 64 | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.5249 | 89.8% [86.2%, 93.3%] |
| random-proj 32 | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.3777 | 64.7% [59.4%, 69.5%] |
| truncate 32 | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.3600 | 61.6% [56.4%, 67.0%] |
| ward distance=0.6 (per-doc count) | 425.4 | 128 | 32 | 217.80 | 2.4x | 0.5778 | 98.9% [97.5%, 100.3%] |
| ward distance=0.8 (per-doc count) | 275.5 | 128 | 32 | 141.04 | 3.7x | 0.5789 | 99.1% [97.3%, 100.9%] |
| ward distance=1.0 (per-doc count) | 195.8 | 128 | 32 | 100.24 | 5.3x | 0.5695 | 97.5% [95.2%, 99.7%] |
| ward distance=1.3 (per-doc count) | 134.0 | 128 | 32 | 68.62 | 7.7x | 0.5663 | 97.0% [94.4%, 99.6%] |
| ward distance=1.6 (per-doc count) | 99.9 | 128 | 32 | 51.14 | 10.3x | 0.5598 | 95.8% [93.3%, 98.4%] |
| adaptive radius=0.8 refine=3 | 300.0 | 128 | 32 | 153.62 | 3.4x | 0.5677 | 97.2% [95.4%, 98.8%] |
| adaptive radius=0.7 refine=3 | 181.5 | 128 | 32 | 92.93 | 5.7x | 0.5622 | 96.3% [93.6%, 98.7%] |
| adaptive radius=0.6 refine=3 | 122.2 | 128 | 32 | 62.56 | 8.4x | 0.5618 | 96.2% [93.7%, 98.8%] |
| merge 0.6 > int8 per-vector | 122.2 | 128 | 8.12 | 15.89 | 33.2x | 0.5601 | 95.9% [93.2%, 98.6%] |
| merge 0.6 > int4 | 122.2 | 128 | 4.12 | 8.06 | 65.5x | 0.5547 | 95.0% [92.1%, 98.0%] |
| merge 0.6 > lloyd2 | 122.2 | 128 | 2 | 3.91 | 135.0x | 0.5522 | 94.5% [91.5%, 97.7%] |
| merge 0.6 > binary | 122.2 | 128 | 1 | 1.96 | 270.0x | 0.5215 | 89.3% [86.0%, 92.6%] |
| merge 0.6 > binary centred | 122.2 | 128 | 1 | 1.96 | 269.9x | 0.5287 | 90.5% [87.0%, 93.9%] |
| merge 0.6 > pca-docs 64 > int8 per-vector | 122.2 | 64 | 8.25 | 8.06 | 65.5x | 0.5477 | 93.8% [90.8%, 96.7%] |
| merge 0.6 > pca-docs 64 > binary | 122.2 | 64 | 1 | 0.98 | 540.0x | 0.5004 | 85.7% [81.3%, 89.8%] |

**E2-colpali-infovqa** — 500 docs, 247 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.8581 | 100.0% [100.0%, 100.0%] |
| pca-joint 64 (fit on cal queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.8363 | 97.5% [95.5%, 99.2%] |
| pca-docs 64 (same held-out queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.8311 | 96.9% [94.8%, 98.7%] |
| pca-joint 32 (fit on cal queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.7454 | 86.9% [82.9%, 90.6%] |
| pca-docs 32 (same held-out queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.7742 | 90.2% [86.5%, 93.7%] |

**E2-colpali-infovqa** — 500 docs, 494 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.8458 | 100.0% [100.0%, 100.0%] |
| float16 | 1031.0 | 128 | 16 | 263.94 | 2.0x | 0.8458 | 100.0% [100.0%, 100.0%] |
| int8 fixed (legacy) | 1031.0 | 128 | 8 | 131.97 | 4.0x | 0.8467 | 100.1% [100.0%, 100.3%] |
| int8 per-vector | 1031.0 | 128 | 8.12 | 134.03 | 3.9x | 0.8458 | 100.0% [100.0%, 100.0%] |
| int8 per-dimension | 1031.0 | 128 | 8 | 131.97 | 4.0x | 0.8456 | 100.0% [99.9%, 100.0%] |
| int4 per-vector | 1031.0 | 128 | 4.12 | 68.05 | 7.8x | 0.8448 | 99.9% [99.4%, 100.4%] |
| lloyd2 (2-bit) | 1031.0 | 128 | 2 | 32.99 | 16.0x | 0.8412 | 99.4% [98.5%, 100.4%] |
| lloyd2 (2-bit, seed 1) | 1031.0 | 128 | 2 | 32.99 | 16.0x | 0.8425 | 99.6% [98.4%, 100.7%] |
| lloyd2 (2-bit, seed 2) | 1031.0 | 128 | 2 | 32.99 | 16.0x | 0.8401 | 99.3% [98.3%, 100.3%] |
| binary | 1031.0 | 128 | 1 | 16.50 | 32.0x | 0.8243 | 97.4% [96.0%, 98.7%] |
| binary centred | 1031.0 | 128 | 1 | 16.50 | 32.0x | 0.8328 | 98.5% [97.1%, 99.7%] |
| pca-docs 96 | 1031.0 | 96 | 32 | 395.90 | 1.3x | 0.8441 | 99.8% [99.2%, 100.3%] |
| pca-docs 64 | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.8224 | 97.2% [95.9%, 98.5%] |
| pca-docs 48 | 1031.0 | 48 | 32 | 197.95 | 2.7x | 0.8075 | 95.5% [93.6%, 97.2%] |
| pca-docs 32 | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.7366 | 87.1% [84.3%, 89.8%] |
| random-proj 96 | 1031.0 | 96 | 32 | 395.90 | 1.3x | 0.8315 | 98.3% [97.2%, 99.3%] |
| truncate 96 | 1031.0 | 96 | 32 | 395.90 | 1.3x | 0.8357 | 98.8% [97.5%, 100.0%] |
| random-proj 64 | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.8056 | 95.2% [93.1%, 97.4%] |
| truncate 64 | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.8151 | 96.4% [94.3%, 98.1%] |
| random-proj 32 | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.6835 | 80.8% [77.7%, 84.0%] |
| truncate 32 | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.6967 | 82.4% [78.8%, 85.6%] |
| ward distance=0.6 (per-doc count) | 417.9 | 128 | 32 | 213.99 | 2.5x | 0.8441 | 99.8% [99.1%, 100.4%] |
| ward distance=0.8 (per-doc count) | 270.3 | 128 | 32 | 138.41 | 3.8x | 0.8470 | 100.1% [99.3%, 101.1%] |
| ward distance=1.0 (per-doc count) | 191.9 | 128 | 32 | 98.25 | 5.4x | 0.8419 | 99.5% [98.5%, 100.5%] |
| ward distance=1.3 (per-doc count) | 127.7 | 128 | 32 | 65.40 | 8.1x | 0.8363 | 98.9% [97.7%, 99.9%] |
| ward distance=1.6 (per-doc count) | 93.0 | 128 | 32 | 47.63 | 11.1x | 0.8343 | 98.6% [97.3%, 99.8%] |
| adaptive radius=0.8 refine=3 | 282.6 | 128 | 32 | 144.68 | 3.6x | 0.8436 | 99.7% [98.9%, 100.7%] |
| adaptive radius=0.7 refine=3 | 172.0 | 128 | 32 | 88.08 | 6.0x | 0.8413 | 99.5% [98.4%, 100.5%] |
| adaptive radius=0.6 refine=3 | 115.8 | 128 | 32 | 59.31 | 8.9x | 0.8467 | 100.1% [99.0%, 101.3%] |
| merge 0.6 > int8 per-vector | 115.8 | 128 | 8.12 | 15.06 | 35.1x | 0.8407 | 99.4% [98.1%, 100.6%] |
| merge 0.6 > int4 | 115.8 | 128 | 4.12 | 7.65 | 69.0x | 0.8391 | 99.2% [97.9%, 100.5%] |
| merge 0.6 > lloyd2 | 115.8 | 128 | 2 | 3.71 | 142.4x | 0.8363 | 98.9% [97.3%, 100.4%] |
| merge 0.6 > binary | 115.8 | 128 | 1 | 1.85 | 284.8x | 0.8173 | 96.6% [95.1%, 98.2%] |
| merge 0.6 > binary centred | 115.8 | 128 | 1 | 1.85 | 284.7x | 0.8201 | 97.0% [95.4%, 98.4%] |
| merge 0.6 > pca-docs 64 > int8 per-vector | 115.8 | 64 | 8.25 | 7.65 | 69.0x | 0.8161 | 96.5% [94.8%, 98.2%] |
| merge 0.6 > pca-docs 64 > binary | 115.8 | 64 | 1 | 0.93 | 569.6x | 0.7734 | 91.4% [89.0%, 93.8%] |

**E3-colpali-generated** — 60 docs, 36 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.6841 | 100.0% [100.0%, 100.0%] |
| pca-joint 64 (fit on cal queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.6805 | 99.5% [95.1%, 104.1%] |
| pca-docs 64 (same held-out queries) | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.6595 | 96.4% [90.3%, 102.2%] |
| pca-joint 32 (fit on cal queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.6438 | 94.1% [86.3%, 101.6%] |
| pca-docs 32 (same held-out queries) | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.6517 | 95.3% [86.2%, 104.8%] |

**E3-colpali-generated** — 60 docs, 72 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 1031.0 | 128 | 32 | 527.87 | 1.0x | 0.6954 | 100.0% [100.0%, 100.0%] |
| float16 | 1031.0 | 128 | 16 | 263.94 | 2.0x | 0.6935 | 99.7% [99.2%, 100.0%] |
| int8 fixed (legacy) | 1031.0 | 128 | 8 | 131.97 | 4.0x | 0.6954 | 100.0% [97.9%, 102.3%] |
| int8 per-vector | 1031.0 | 128 | 8.12 | 134.03 | 3.9x | 0.6876 | 98.9% [97.1%, 100.1%] |
| int8 per-dimension | 1031.0 | 128 | 8 | 131.98 | 4.0x | 0.6935 | 99.7% [97.7%, 101.8%] |
| int4 per-vector | 1031.0 | 128 | 4.12 | 68.05 | 7.8x | 0.7086 | 101.9% [98.2%, 106.0%] |
| lloyd2 (2-bit) | 1031.0 | 128 | 2 | 33.00 | 16.0x | 0.6847 | 98.5% [93.1%, 105.0%] |
| lloyd2 (2-bit, seed 1) | 1031.0 | 128 | 2 | 33.00 | 16.0x | 0.7056 | 101.5% [96.9%, 106.3%] |
| lloyd2 (2-bit, seed 2) | 1031.0 | 128 | 2 | 33.00 | 16.0x | 0.7029 | 101.1% [95.1%, 107.4%] |
| binary | 1031.0 | 128 | 1 | 16.50 | 32.0x | 0.6845 | 98.4% [91.9%, 105.4%] |
| binary centred | 1031.0 | 128 | 1 | 16.50 | 32.0x | 0.6891 | 99.1% [93.4%, 105.1%] |
| pca-docs 96 | 1031.0 | 96 | 32 | 395.90 | 1.3x | 0.6941 | 99.8% [96.6%, 103.2%] |
| pca-docs 64 | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.7014 | 100.9% [96.2%, 106.2%] |
| pca-docs 48 | 1031.0 | 48 | 32 | 197.95 | 2.7x | 0.6946 | 99.9% [94.6%, 105.8%] |
| pca-docs 32 | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.6844 | 98.4% [91.2%, 105.6%] |
| random-proj 96 | 1031.0 | 96 | 32 | 395.90 | 1.3x | 0.6898 | 99.2% [94.5%, 103.9%] |
| truncate 96 | 1031.0 | 96 | 32 | 395.90 | 1.3x | 0.6862 | 98.7% [94.4%, 102.9%] |
| random-proj 64 | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.6850 | 98.5% [93.1%, 104.1%] |
| truncate 64 | 1031.0 | 64 | 32 | 263.94 | 2.0x | 0.7003 | 100.7% [95.1%, 106.8%] |
| random-proj 32 | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.7204 | 103.6% [96.0%, 112.1%] |
| truncate 32 | 1031.0 | 32 | 32 | 131.97 | 4.0x | 0.6047 | 87.0% [77.7%, 95.8%] |
| ward distance=0.6 (per-doc count) | 371.7 | 128 | 32 | 190.30 | 2.8x | 0.7040 | 101.2% [97.3%, 105.6%] |
| ward distance=0.8 (per-doc count) | 229.9 | 128 | 32 | 117.69 | 4.5x | 0.6838 | 98.3% [93.5%, 103.3%] |
| ward distance=1.0 (per-doc count) | 161.5 | 128 | 32 | 82.68 | 6.4x | 0.6880 | 98.9% [92.1%, 105.6%] |
| ward distance=1.3 (per-doc count) | 109.9 | 128 | 32 | 56.26 | 9.4x | 0.6863 | 98.7% [92.6%, 104.7%] |
| ward distance=1.6 (per-doc count) | 83.8 | 128 | 32 | 42.88 | 12.3x | 0.7082 | 101.8% [93.9%, 110.7%] |
| adaptive radius=0.8 refine=3 | 224.1 | 128 | 32 | 114.73 | 4.6x | 0.6853 | 98.6% [92.9%, 104.7%] |
| adaptive radius=0.7 refine=3 | 125.6 | 128 | 32 | 64.31 | 8.2x | 0.7088 | 101.9% [94.7%, 109.8%] |
| adaptive radius=0.6 refine=3 | 82.3 | 128 | 32 | 42.12 | 12.5x | 0.7198 | 103.5% [96.5%, 111.2%] |
| merge 0.6 > int8 per-vector | 82.3 | 128 | 8.12 | 10.69 | 49.4x | 0.7067 | 101.6% [93.6%, 110.9%] |
| merge 0.6 > int4 | 82.3 | 128 | 4.12 | 5.43 | 97.2x | 0.6984 | 100.4% [92.5%, 109.5%] |
| merge 0.6 > lloyd2 | 82.3 | 128 | 2 | 2.64 | 199.9x | 0.7085 | 101.9% [93.5%, 111.4%] |
| merge 0.6 > binary | 82.3 | 128 | 1 | 1.32 | 401.0x | 0.6986 | 100.5% [92.6%, 109.2%] |
| merge 0.6 > binary centred | 82.3 | 128 | 1 | 1.32 | 398.5x | 0.7031 | 101.1% [91.5%, 111.2%] |
| merge 0.6 > pca-docs 64 > int8 per-vector | 82.3 | 64 | 8.25 | 5.43 | 97.2x | 0.6868 | 98.8% [91.4%, 107.4%] |
| merge 0.6 > pca-docs 64 > binary | 82.3 | 64 | 1 | 0.66 | 802.1x | 0.6696 | 96.3% [88.3%, 106.2%] |

## text

**text-answerai-colbert-small-v1-scifact** — 5183 docs, 300 queries, 20 random half/half splits per row

| reference | rule | target | held-out met target | held-out mean | held-out worst | met (labels) | median compression |
|---|---|---|---|---|---|---|---|
| labels | point | 0.99 | 75.0% (20/20 selected) | 99.0% | 95.0% | 75.0% | 4.0x |
| labels | point | 0.97 | 60.0% (20/20 selected) | 97.8% | 95.0% | 60.0% | 9.3x |
| labels | point | 0.95 | 70.0% (20/20 selected) | 96.0% | 91.7% | 70.0% | 18.5x |
| labels | lower_ci | 0.99 | 95.0% (20/20 selected) | 99.7% | 98.6% | 95.0% | 2.0x |
| labels | lower_ci | 0.97 | 100.0% (20/20 selected) | 99.6% | 98.6% | 100.0% | 4.0x |
| labels | lower_ci | 0.95 | 100.0% (20/20 selected) | 98.3% | 95.0% | 100.0% | 9.3x |
| baseline@1 | point | 0.99 | 60.0% (20/20 selected) | 99.3% | 98.0% | 100.0% | 2.0x |
| baseline@1 | point | 0.97 | 100.0% (20/20 selected) | 98.8% | 98.0% | 100.0% | 4.0x |
| baseline@1 | point | 0.95 | 100.0% (20/20 selected) | 98.8% | 98.0% | 100.0% | 4.0x |
| baseline@1 | lower_ci | 0.99 | 100.0% (20/20 selected) | 99.9% | 99.8% | 100.0% | 2.0x |
| baseline@1 | lower_ci | 0.97 | 100.0% (20/20 selected) | 98.9% | 98.0% | 100.0% | 4.0x |
| baseline@1 | lower_ci | 0.95 | 100.0% (20/20 selected) | 98.8% | 98.0% | 100.0% | 4.0x |

Pseudo-query proxy (choose on document fragments, judge on real labelled queries):

| target | chosen on fragments | fragment retention | real-query retention |
|---|---|---|---|

Fixed configurations that exist today (all queries):

| configuration | KB/doc | x float32 | retention (labels) |
|---|---|---|---|

**text-answerai-colbert-small-v1-scifact** — 5183 docs, 300 queries, reference: labels

| pipeline | vectors/doc | dim | bits/dim | KB/doc | x float32 | nDCG@5 | retention [95% CI] |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 235.7 | 96 | 32 | 90.49 | 1.0x | 0.7308 | 100.0% [100.0%, 100.0%] |
| adaptive radius=0.95 | 85.5 | 96 | 32 | 32.83 | 2.8x | 0.6614 | 90.5% [86.9%, 93.8%] |
| adaptive radius=0.9 | 50.9 | 96 | 32 | 19.53 | 4.6x | 0.7131 | 97.6% [95.2%, 99.9%] |
| adaptive radius=0.85 | 23.9 | 96 | 32 | 9.16 | 9.9x | 0.6431 | 88.0% [84.4%, 91.7%] |
| adaptive radius=0.8 | 9.4 | 96 | 32 | 3.63 | 24.9x | 0.5400 | 73.9% [68.6%, 78.5%] |
| adaptive radius=0.7 | 1.8 | 96 | 32 | 0.69 | 130.4x | 0.4264 | 58.4% [52.6%, 63.8%] |
| adaptive radius=0.6 | 1.1 | 96 | 32 | 0.40 | 224.1x | 0.4274 | 58.5% [53.1%, 64.4%] |
| hierarchical ratio=0.5 | 118.1 | 96 | 32 | 45.34 | 2.0x | 0.7306 | 100.0% [98.8%, 101.2%] |
| random ratio=0.5 | 118.1 | 96 | 32 | 45.34 | 2.0x | 0.6611 | 90.5% [86.7%, 94.0%] |
| hierarchical ratio=0.33 | 78.3 | 96 | 32 | 30.06 | 3.0x | 0.7275 | 99.5% [97.9%, 101.3%] |
| random ratio=0.33 | 78.3 | 96 | 32 | 30.06 | 3.0x | 0.5888 | 80.6% [75.8%, 85.2%] |
| hierarchical ratio=0.25 | 59.3 | 96 | 32 | 22.77 | 4.0x | 0.6854 | 93.8% [90.5%, 97.2%] |
| random ratio=0.25 | 59.3 | 96 | 32 | 22.77 | 4.0x | 0.5656 | 77.4% [72.1%, 82.7%] |
| redundancy t=0.92 | 68.8 | 96 | 32 | 26.41 | 3.4x | 0.7070 | 96.7% [94.4%, 99.0%] |
| redundancy t=0.85 | 31.1 | 96 | 32 | 11.94 | 7.6x | 0.6783 | 92.8% [89.9%, 95.8%] |
| float16 | 235.7 | 96 | 16 | 45.25 | 2.0x | 0.7308 | 100.0% [100.0%, 100.0%] |
| int8 fixed (legacy) | 235.7 | 96 | 8 | 22.62 | 4.0x | 0.7274 | 99.5% [98.6%, 100.4%] |
| int8 per-vector | 235.7 | 96 | 8.17 | 23.09 | 3.9x | 0.7297 | 99.8% [99.2%, 100.4%] |
| int4 per-vector | 235.7 | 96 | 4.17 | 11.78 | 7.7x | 0.6571 | 89.9% [85.8%, 93.8%] |
| lloyd2 (2-bit) | 235.7 | 96 | 2 | 5.66 | 16.0x | 0.6123 | 83.8% [79.0%, 88.2%] |
| binary | 235.7 | 96 | 1 | 2.83 | 32.0x | 0.6803 | 93.1% [89.8%, 96.0%] |
| adaptive 0.8 > int8 per-vector | 9.4 | 96 | 8.17 | 0.93 | 97.7x | 0.5359 | 73.3% [68.0%, 78.3%] |
| adaptive 0.8 > binary | 9.4 | 96 | 1 | 0.11 | 798.0x | 0.2749 | 37.6% [31.9%, 43.4%] |

## tiered

**E2-colpali-docvqa** — 500 docs, 451 queries

| hot (RAM) | cold (disk) | shortlist | hot KB/doc | cold KB/doc | retention [95% CI] | rescore ms/q |
|---|---|---|---|---|---|---|
| hot: binary | - | - | 16.50 | 0.00 | 96.3% [93.7%, 98.8%] | 0.0 |
| hot: binary | cold int8/vec (all vectors) | 10 | 16.50 | 134.03 | 99.6% [98.1%, 100.8%] | 8.2 |
| hot: binary | cold int8/vec (all vectors) | 20 | 16.50 | 134.03 | 99.4% [98.1%, 100.7%] | 19.2 |
| hot: binary | cold int8/vec (all vectors) | 50 | 16.50 | 134.03 | 100.1% [99.5%, 100.9%] | 42.0 |
| hot: binary | cold int8/vec (all vectors) | 100 | 16.50 | 134.03 | 100.0% [99.4%, 100.5%] | 70.1 |
| hot: binary | cold float16 (all vectors) | 10 | 16.50 | 263.94 | 99.3% [97.9%, 100.5%] | 9.2 |
| hot: binary | cold float16 (all vectors) | 20 | 16.50 | 263.94 | 99.3% [98.0%, 100.4%] | 17.7 |
| hot: binary | cold float16 (all vectors) | 50 | 16.50 | 263.94 | 100.0% [99.4%, 100.5%] | 49.0 |
| hot: binary | cold float16 (all vectors) | 100 | 16.50 | 263.94 | 99.8% [99.3%, 100.0%] | 82.9 |
| hot: binary | cold int8/vec (same merge None) | 10 | 16.50 | 134.03 | 99.6% [98.1%, 100.8%] | 7.6 |
| hot: binary | cold int8/vec (same merge None) | 20 | 16.50 | 134.03 | 99.4% [98.1%, 100.7%] | 11.7 |
| hot: binary | cold int8/vec (same merge None) | 50 | 16.50 | 134.03 | 100.1% [99.5%, 100.9%] | 35.9 |
| hot: binary | cold int8/vec (same merge None) | 100 | 16.50 | 134.03 | 100.0% [99.4%, 100.5%] | 75.2 |
| hot: merge 0.8 > binary | - | - | 4.80 | 0.00 | 91.7% [88.5%, 94.8%] | 0.0 |
| hot: merge 0.8 > binary | cold int8/vec (all vectors) | 10 | 4.80 | 134.03 | 98.3% [96.5%, 100.0%] | 8.2 |
| hot: merge 0.8 > binary | cold int8/vec (all vectors) | 20 | 4.80 | 134.03 | 99.8% [98.5%, 101.1%] | 16.6 |
| hot: merge 0.8 > binary | cold int8/vec (all vectors) | 50 | 4.80 | 134.03 | 101.0% [100.1%, 102.0%] | 39.7 |
| hot: merge 0.8 > binary | cold int8/vec (all vectors) | 100 | 4.80 | 134.03 | 100.5% [99.7%, 101.2%] | 79.3 |
| hot: merge 0.8 > binary | cold float16 (all vectors) | 10 | 4.80 | 263.94 | 98.2% [96.4%, 99.8%] | 9.0 |
| hot: merge 0.8 > binary | cold float16 (all vectors) | 20 | 4.80 | 263.94 | 99.8% [98.4%, 101.2%] | 13.2 |
| hot: merge 0.8 > binary | cold float16 (all vectors) | 50 | 4.80 | 263.94 | 101.0% [100.0%, 102.0%] | 33.2 |
| hot: merge 0.8 > binary | cold float16 (all vectors) | 100 | 4.80 | 263.94 | 100.4% [99.7%, 101.2%] | 81.8 |
| hot: merge 0.8 > binary | cold int8/vec (same merge 0.8) | 10 | 4.80 | 39.00 | 97.0% [94.5%, 98.9%] | 2.3 |
| hot: merge 0.8 > binary | cold int8/vec (same merge 0.8) | 20 | 4.80 | 39.00 | 96.9% [94.9%, 98.6%] | 3.8 |
| hot: merge 0.8 > binary | cold int8/vec (same merge 0.8) | 50 | 4.80 | 39.00 | 97.3% [95.5%, 98.9%] | 9.1 |
| hot: merge 0.8 > binary | cold int8/vec (same merge 0.8) | 100 | 4.80 | 39.00 | 97.2% [95.3%, 98.7%] | 25.8 |
| hot: merge 0.6 > binary | - | - | 1.96 | 0.00 | 89.3% [86.0%, 92.6%] | 0.0 |
| hot: merge 0.6 > binary | cold int8/vec (all vectors) | 10 | 1.96 | 134.03 | 95.6% [92.8%, 98.2%] | 6.6 |
| hot: merge 0.6 > binary | cold int8/vec (all vectors) | 20 | 1.96 | 134.03 | 98.2% [96.1%, 100.2%] | 11.3 |
| hot: merge 0.6 > binary | cold int8/vec (all vectors) | 50 | 1.96 | 134.03 | 99.1% [97.5%, 100.5%] | 25.4 |
| hot: merge 0.6 > binary | cold int8/vec (all vectors) | 100 | 1.96 | 134.03 | 99.3% [98.0%, 100.4%] | 60.7 |
| hot: merge 0.6 > binary | cold float16 (all vectors) | 10 | 1.96 | 263.94 | 95.6% [92.8%, 98.2%] | 6.4 |
| hot: merge 0.6 > binary | cold float16 (all vectors) | 20 | 1.96 | 263.94 | 98.0% [96.0%, 100.0%] | 13.4 |
| hot: merge 0.6 > binary | cold float16 (all vectors) | 50 | 1.96 | 263.94 | 98.8% [97.2%, 100.1%] | 38.7 |
| hot: merge 0.6 > binary | cold float16 (all vectors) | 100 | 1.96 | 263.94 | 99.2% [97.8%, 100.2%] | 78.9 |
| hot: merge 0.6 > binary | cold int8/vec (same merge 0.6) | 10 | 1.96 | 15.89 | 94.4% [91.4%, 97.5%] | 1.0 |
| hot: merge 0.6 > binary | cold int8/vec (same merge 0.6) | 20 | 1.96 | 15.89 | 95.8% [93.0%, 98.6%] | 2.7 |
| hot: merge 0.6 > binary | cold int8/vec (same merge 0.6) | 50 | 1.96 | 15.89 | 95.9% [93.3%, 98.6%] | 5.2 |
| hot: merge 0.6 > binary | cold int8/vec (same merge 0.6) | 100 | 1.96 | 15.89 | 95.9% [93.2%, 98.6%] | 8.7 |
| hot: merge 0.5 > binary | - | - | 1.40 | 0.00 | 87.8% [84.1%, 91.3%] | 0.0 |
| hot: merge 0.5 > binary | cold int8/vec (all vectors) | 10 | 1.40 | 134.03 | 95.5% [92.8%, 97.9%] | 7.8 |
| hot: merge 0.5 > binary | cold int8/vec (all vectors) | 20 | 1.40 | 134.03 | 96.0% [93.5%, 98.1%] | 13.4 |
| hot: merge 0.5 > binary | cold int8/vec (all vectors) | 50 | 1.40 | 134.03 | 99.5% [97.8%, 101.0%] | 33.0 |
| hot: merge 0.5 > binary | cold int8/vec (all vectors) | 100 | 1.40 | 134.03 | 99.8% [98.4%, 100.8%] | 60.3 |
| hot: merge 0.5 > binary | cold float16 (all vectors) | 10 | 1.40 | 263.94 | 95.4% [92.7%, 97.7%] | 5.4 |
| hot: merge 0.5 > binary | cold float16 (all vectors) | 20 | 1.40 | 263.94 | 96.0% [93.5%, 98.1%] | 12.2 |
| hot: merge 0.5 > binary | cold float16 (all vectors) | 50 | 1.40 | 263.94 | 99.3% [97.7%, 100.8%] | 35.8 |
| hot: merge 0.5 > binary | cold float16 (all vectors) | 100 | 1.40 | 263.94 | 99.6% [98.3%, 100.7%] | 56.0 |
| hot: merge 0.5 > binary | cold int8/vec (same merge 0.5) | 10 | 1.40 | 11.41 | 93.4% [90.1%, 96.4%] | 0.8 |
| hot: merge 0.5 > binary | cold int8/vec (same merge 0.5) | 20 | 1.40 | 11.41 | 93.7% [90.7%, 96.9%] | 1.6 |
| hot: merge 0.5 > binary | cold int8/vec (same merge 0.5) | 50 | 1.40 | 11.41 | 94.5% [91.6%, 97.5%] | 3.8 |
| hot: merge 0.5 > binary | cold int8/vec (same merge 0.5) | 100 | 1.40 | 11.41 | 94.3% [91.4%, 97.4%] | 6.9 |

**E2-colpali-infovqa** — 500 docs, 494 queries

| hot (RAM) | cold (disk) | shortlist | hot KB/doc | cold KB/doc | retention [95% CI] | rescore ms/q |
|---|---|---|---|---|---|---|
| hot: binary | - | - | 16.50 | 0.00 | 97.4% [96.0%, 98.7%] | 0.0 |
| hot: binary | cold int8/vec (all vectors) | 10 | 16.50 | 134.03 | 99.9% [99.7%, 100.1%] | 6.2 |
| hot: binary | cold int8/vec (all vectors) | 20 | 16.50 | 134.03 | 100.0% [100.0%, 100.0%] | 12.1 |
| hot: binary | cold int8/vec (all vectors) | 50 | 16.50 | 134.03 | 100.0% [100.0%, 100.0%] | 36.7 |
| hot: binary | cold int8/vec (all vectors) | 100 | 16.50 | 134.03 | 100.0% [100.0%, 100.0%] | 87.8 |
| hot: binary | cold float16 (all vectors) | 10 | 16.50 | 263.94 | 99.9% [99.7%, 100.1%] | 12.7 |
| hot: binary | cold float16 (all vectors) | 20 | 16.50 | 263.94 | 100.0% [100.0%, 100.0%] | 18.8 |
| hot: binary | cold float16 (all vectors) | 50 | 16.50 | 263.94 | 100.0% [100.0%, 100.0%] | 43.8 |
| hot: binary | cold float16 (all vectors) | 100 | 16.50 | 263.94 | 100.0% [100.0%, 100.0%] | 96.2 |
| hot: binary | cold int8/vec (same merge None) | 10 | 16.50 | 134.03 | 99.9% [99.7%, 100.1%] | 7.9 |
| hot: binary | cold int8/vec (same merge None) | 20 | 16.50 | 134.03 | 100.0% [100.0%, 100.0%] | 14.5 |
| hot: binary | cold int8/vec (same merge None) | 50 | 16.50 | 134.03 | 100.0% [100.0%, 100.0%] | 34.1 |
| hot: binary | cold int8/vec (same merge None) | 100 | 16.50 | 134.03 | 100.0% [100.0%, 100.0%] | 73.3 |
| hot: merge 0.8 > binary | - | - | 4.52 | 0.00 | 97.1% [95.7%, 98.4%] | 0.0 |
| hot: merge 0.8 > binary | cold int8/vec (all vectors) | 10 | 4.52 | 134.03 | 99.6% [98.8%, 100.3%] | 7.7 |
| hot: merge 0.8 > binary | cold int8/vec (all vectors) | 20 | 4.52 | 134.03 | 99.8% [99.2%, 100.2%] | 14.5 |
| hot: merge 0.8 > binary | cold int8/vec (all vectors) | 50 | 4.52 | 134.03 | 100.0% [100.0%, 100.1%] | 33.4 |
| hot: merge 0.8 > binary | cold int8/vec (all vectors) | 100 | 4.52 | 134.03 | 100.0% [100.0%, 100.0%] | 72.9 |
| hot: merge 0.8 > binary | cold float16 (all vectors) | 10 | 4.52 | 263.94 | 99.6% [98.8%, 100.3%] | 10.8 |
| hot: merge 0.8 > binary | cold float16 (all vectors) | 20 | 4.52 | 263.94 | 99.8% [99.2%, 100.2%] | 24.9 |
| hot: merge 0.8 > binary | cold float16 (all vectors) | 50 | 4.52 | 263.94 | 100.0% [100.0%, 100.1%] | 48.5 |
| hot: merge 0.8 > binary | cold float16 (all vectors) | 100 | 4.52 | 263.94 | 100.0% [100.0%, 100.0%] | 104.8 |
| hot: merge 0.8 > binary | cold int8/vec (same merge 0.8) | 10 | 4.52 | 36.73 | 99.6% [98.6%, 100.6%] | 2.8 |
| hot: merge 0.8 > binary | cold int8/vec (same merge 0.8) | 20 | 4.52 | 36.73 | 99.7% [98.7%, 100.7%] | 4.6 |
| hot: merge 0.8 > binary | cold int8/vec (same merge 0.8) | 50 | 4.52 | 36.73 | 99.9% [99.1%, 100.9%] | 11.8 |
| hot: merge 0.8 > binary | cold int8/vec (same merge 0.8) | 100 | 4.52 | 36.73 | 99.9% [99.1%, 100.9%] | 22.2 |
| hot: merge 0.6 > binary | - | - | 1.85 | 0.00 | 96.6% [95.1%, 98.2%] | 0.0 |
| hot: merge 0.6 > binary | cold int8/vec (all vectors) | 10 | 1.85 | 134.03 | 99.8% [99.1%, 100.5%] | 8.1 |
| hot: merge 0.6 > binary | cold int8/vec (all vectors) | 20 | 1.85 | 134.03 | 99.8% [99.4%, 100.3%] | 16.2 |
| hot: merge 0.6 > binary | cold int8/vec (all vectors) | 50 | 1.85 | 134.03 | 100.0% [100.0%, 100.1%] | 33.3 |
| hot: merge 0.6 > binary | cold int8/vec (all vectors) | 100 | 1.85 | 134.03 | 100.0% [100.0%, 100.1%] | 94.6 |
| hot: merge 0.6 > binary | cold float16 (all vectors) | 10 | 1.85 | 263.94 | 99.8% [99.1%, 100.5%] | 11.0 |
| hot: merge 0.6 > binary | cold float16 (all vectors) | 20 | 1.85 | 263.94 | 99.8% [99.4%, 100.3%] | 22.2 |
| hot: merge 0.6 > binary | cold float16 (all vectors) | 50 | 1.85 | 263.94 | 100.0% [100.0%, 100.1%] | 46.8 |
| hot: merge 0.6 > binary | cold float16 (all vectors) | 100 | 1.85 | 263.94 | 100.0% [100.0%, 100.1%] | 119.8 |
| hot: merge 0.6 > binary | cold int8/vec (same merge 0.6) | 10 | 1.85 | 15.06 | 99.5% [98.2%, 100.7%] | 1.4 |
| hot: merge 0.6 > binary | cold int8/vec (same merge 0.6) | 20 | 1.85 | 15.06 | 99.3% [98.0%, 100.5%] | 2.6 |
| hot: merge 0.6 > binary | cold int8/vec (same merge 0.6) | 50 | 1.85 | 15.06 | 99.4% [98.1%, 100.6%] | 6.4 |
| hot: merge 0.6 > binary | cold int8/vec (same merge 0.6) | 100 | 1.85 | 15.06 | 99.4% [98.1%, 100.6%] | 16.6 |
| hot: merge 0.5 > binary | - | - | 1.30 | 0.00 | 93.7% [91.6%, 95.7%] | 0.0 |
| hot: merge 0.5 > binary | cold int8/vec (all vectors) | 10 | 1.30 | 134.03 | 98.4% [97.3%, 99.4%] | 10.1 |
| hot: merge 0.5 > binary | cold int8/vec (all vectors) | 20 | 1.30 | 134.03 | 99.3% [98.5%, 100.0%] | 22.6 |
| hot: merge 0.5 > binary | cold int8/vec (all vectors) | 50 | 1.30 | 134.03 | 99.8% [99.4%, 100.1%] | 45.9 |
| hot: merge 0.5 > binary | cold int8/vec (all vectors) | 100 | 1.30 | 134.03 | 100.0% [100.0%, 100.1%] | 97.9 |
| hot: merge 0.5 > binary | cold float16 (all vectors) | 10 | 1.30 | 263.94 | 98.4% [97.3%, 99.4%] | 10.8 |
| hot: merge 0.5 > binary | cold float16 (all vectors) | 20 | 1.30 | 263.94 | 99.3% [98.5%, 100.0%] | 23.2 |
| hot: merge 0.5 > binary | cold float16 (all vectors) | 50 | 1.30 | 263.94 | 99.8% [99.4%, 100.1%] | 56.7 |
| hot: merge 0.5 > binary | cold float16 (all vectors) | 100 | 1.30 | 263.94 | 100.0% [100.0%, 100.1%] | 109.7 |
| hot: merge 0.5 > binary | cold int8/vec (same merge 0.5) | 10 | 1.30 | 10.60 | 97.3% [95.5%, 99.2%] | 1.3 |
| hot: merge 0.5 > binary | cold int8/vec (same merge 0.5) | 20 | 1.30 | 10.60 | 97.4% [95.5%, 99.3%] | 2.1 |
| hot: merge 0.5 > binary | cold int8/vec (same merge 0.5) | 50 | 1.30 | 10.60 | 97.6% [95.7%, 99.4%] | 5.5 |
| hot: merge 0.5 > binary | cold int8/vec (same merge 0.5) | 100 | 1.30 | 10.60 | 97.6% [95.7%, 99.4%] | 9.5 |

