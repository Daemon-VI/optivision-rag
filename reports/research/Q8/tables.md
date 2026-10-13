#### colpali_docvqa (float32 527.9 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 1031.0 × 128 | 527.88 KB | 0.0 KB | 1.0x | 263.9 MB | 5.28 GB | 52.8 GB | 527.9 GB | 16,272 |
| float16 | 1031.0 × 128 | 263.94 KB | 0.0 KB | 2.0x | 132.0 MB | 2.64 GB | 26.4 GB | 263.9 GB | 32,544 |
| int8(per_vector) | 1031.0 × 128 | 134.04 KB | 0.0 KB | 3.9x | 67.0 MB | 1.34 GB | 13.4 GB | 134.0 GB | 64,085 |
| int4(mean) | 1031.0 × 128 | 68.05 KB | 0.5 KB | 7.8x | 34.0 MB | 0.68 GB | 6.8 GB | 68.1 GB | 126,222 |
| binary | 1031.0 × 128 | 16.50 KB | 0.0 KB | 32.0x | 8.3 MB | 0.17 GB | 1.7 GB | 16.5 GB | 520,475 |
| hierarchical_merge(0.25) | 263.0 × 128 | 134.66 KB | 0.0 KB | 3.9x | 67.3 MB | 1.35 GB | 13.5 GB | 134.7 GB | 63,787 |
| hierarchical_merge(0.25) > int8(per_vector) | 263.0 × 128 | 34.20 KB | 0.0 KB | 15.4x | 17.1 MB | 0.34 GB | 3.4 GB | 34.2 GB | 251,182 |
| hierarchical_merge(0.25) > binary | 263.0 × 128 | 4.22 KB | 0.0 KB | 125.4x | 2.1 MB | 0.04 GB | 0.4 GB | 4.2 GB | 2,037,453 |
| project(64) > int8(per_vector) | 1031.0 × 64 | 68.05 KB | 32.8 KB | 7.8x | 34.1 MB | 0.68 GB | 6.8 GB | 68.1 GB | 126,221 |
| hierarchical_merge(0.25) > project(64) > binary | 263.0 × 64 | 2.11 KB | 32.8 KB | 250.9x | 1.1 MB | 0.02 GB | 0.2 GB | 2.1 GB | 4,067,157 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 4.2 KB / disk 138.3 KB | 0 | – | RAM 2.1 MB / disk 69 MB | 0.042 / 1.38 GB | 0.42 / 13.8 GB | 4.2 / 138 GB | 2,037,453 |

#### colpali_infovqa (float32 527.9 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 1031.0 × 128 | 527.88 KB | 0.0 KB | 1.0x | 263.9 MB | 5.28 GB | 52.8 GB | 527.9 GB | 16,272 |
| float16 | 1031.0 × 128 | 263.94 KB | 0.0 KB | 2.0x | 132.0 MB | 2.64 GB | 26.4 GB | 263.9 GB | 32,544 |
| int8(per_vector) | 1031.0 × 128 | 134.04 KB | 0.0 KB | 3.9x | 67.0 MB | 1.34 GB | 13.4 GB | 134.0 GB | 64,085 |
| int4(mean) | 1031.0 × 128 | 68.05 KB | 0.5 KB | 7.8x | 34.0 MB | 0.68 GB | 6.8 GB | 68.1 GB | 126,222 |
| binary | 1031.0 × 128 | 16.50 KB | 0.0 KB | 32.0x | 8.3 MB | 0.17 GB | 1.7 GB | 16.5 GB | 520,475 |
| hierarchical_merge(0.25) | 263.0 × 128 | 134.66 KB | 0.0 KB | 3.9x | 67.3 MB | 1.35 GB | 13.5 GB | 134.7 GB | 63,787 |
| hierarchical_merge(0.25) > int8(per_vector) | 263.0 × 128 | 34.20 KB | 0.0 KB | 15.4x | 17.1 MB | 0.34 GB | 3.4 GB | 34.2 GB | 251,182 |
| hierarchical_merge(0.25) > binary | 263.0 × 128 | 4.22 KB | 0.0 KB | 125.4x | 2.1 MB | 0.04 GB | 0.4 GB | 4.2 GB | 2,037,453 |
| project(64) > int8(per_vector) | 1031.0 × 64 | 68.05 KB | 32.8 KB | 7.8x | 34.1 MB | 0.68 GB | 6.8 GB | 68.1 GB | 126,221 |
| hierarchical_merge(0.25) > project(64) > binary | 263.0 × 64 | 2.11 KB | 32.8 KB | 250.9x | 1.1 MB | 0.02 GB | 0.2 GB | 2.1 GB | 4,067,157 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 4.2 KB / disk 138.3 KB | 0 | – | RAM 2.1 MB / disk 69 MB | 0.042 / 1.38 GB | 0.42 / 13.8 GB | 4.2 / 138 GB | 2,037,453 |

#### colqwen2_docvqa (float32 383.1 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 748.3 × 128 | 383.14 KB | 0.0 KB | 1.0x | 191.6 MB | 3.83 GB | 38.3 GB | 383.1 GB | 22,419 |
| float16 | 748.3 × 128 | 191.57 KB | 0.0 KB | 2.0x | 95.8 MB | 1.92 GB | 19.2 GB | 191.6 GB | 44,838 |
| int8(per_vector) | 748.3 × 128 | 97.29 KB | 0.0 KB | 3.9x | 48.6 MB | 0.97 GB | 9.7 GB | 97.3 GB | 88,293 |
| int4(mean) | 748.3 × 128 | 49.40 KB | 0.5 KB | 7.8x | 24.7 MB | 0.49 GB | 4.9 GB | 49.4 GB | 173,898 |
| binary | 748.3 × 128 | 11.98 KB | 0.0 KB | 32.0x | 6.0 MB | 0.12 GB | 1.2 GB | 12.0 GB | 716,966 |
| hierarchical_merge(0.25) | 195.4 × 128 | 100.04 KB | 0.0 KB | 3.8x | 50.0 MB | 1.00 GB | 10.0 GB | 100.0 GB | 85,868 |
| hierarchical_merge(0.25) > int8(per_vector) | 195.4 × 128 | 25.41 KB | 0.0 KB | 15.1x | 12.7 MB | 0.25 GB | 2.5 GB | 25.4 GB | 338,111 |
| hierarchical_merge(0.25) > binary | 195.4 × 128 | 3.13 KB | 0.0 KB | 122.6x | 1.6 MB | 0.03 GB | 0.3 GB | 3.1 GB | 2,740,997 |
| project(64) > int8(per_vector) | 748.3 × 64 | 49.40 KB | 32.8 KB | 7.8x | 24.7 MB | 0.49 GB | 4.9 GB | 49.4 GB | 173,897 |
| hierarchical_merge(0.25) > project(64) > binary | 195.4 × 64 | 1.57 KB | 32.8 KB | 245.1x | 0.8 MB | 0.02 GB | 0.2 GB | 1.6 GB | 5,467,987 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 3.1 KB / disk 100.4 KB | 0 | – | RAM 1.6 MB / disk 50 MB | 0.031 / 1.00 GB | 0.31 / 10.0 GB | 3.1 / 100 GB | 2,740,997 |

#### colqwen2_infovqa (float32 375.0 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 732.5 × 128 | 375.03 KB | 0.0 KB | 1.0x | 187.5 MB | 3.75 GB | 37.5 GB | 375.0 GB | 22,904 |
| float16 | 732.5 × 128 | 187.52 KB | 0.0 KB | 2.0x | 93.8 MB | 1.88 GB | 18.8 GB | 187.5 GB | 45,807 |
| int8(per_vector) | 732.5 × 128 | 95.23 KB | 0.0 KB | 3.9x | 47.6 MB | 0.95 GB | 9.5 GB | 95.2 GB | 90,202 |
| int4(mean) | 732.5 × 128 | 48.35 KB | 0.5 KB | 7.8x | 24.2 MB | 0.48 GB | 4.8 GB | 48.4 GB | 177,657 |
| binary | 732.5 × 128 | 11.73 KB | 0.0 KB | 32.0x | 5.9 MB | 0.12 GB | 1.2 GB | 11.7 GB | 732,456 |
| hierarchical_merge(0.25) | 191.6 × 128 | 98.10 KB | 0.0 KB | 3.8x | 49.0 MB | 0.98 GB | 9.8 GB | 98.1 GB | 87,566 |
| hierarchical_merge(0.25) > int8(per_vector) | 191.6 × 128 | 24.91 KB | 0.0 KB | 15.1x | 12.5 MB | 0.25 GB | 2.5 GB | 24.9 GB | 344,795 |
| hierarchical_merge(0.25) > binary | 191.6 × 128 | 3.07 KB | 0.0 KB | 122.3x | 1.5 MB | 0.03 GB | 0.3 GB | 3.1 GB | 2,795,052 |
| project(64) > int8(per_vector) | 732.5 × 64 | 48.35 KB | 32.8 KB | 7.8x | 24.2 MB | 0.48 GB | 4.8 GB | 48.4 GB | 177,656 |
| hierarchical_merge(0.25) > project(64) > binary | 191.6 × 64 | 1.54 KB | 32.8 KB | 244.7x | 0.8 MB | 0.02 GB | 0.2 GB | 1.5 GB | 5,575,541 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 3.1 KB / disk 98.3 KB | 0 | – | RAM 1.5 MB / disk 49 MB | 0.031 / 0.98 GB | 0.31 / 9.8 GB | 3.1 / 98 GB | 2,795,052 |

#### colqwen25_docvqa (float32 383.1 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 748.3 × 128 | 383.14 KB | 0.0 KB | 1.0x | 191.6 MB | 3.83 GB | 38.3 GB | 383.1 GB | 22,419 |
| float16 | 748.3 × 128 | 191.57 KB | 0.0 KB | 2.0x | 95.8 MB | 1.92 GB | 19.2 GB | 191.6 GB | 44,838 |
| int8(per_vector) | 748.3 × 128 | 97.29 KB | 0.0 KB | 3.9x | 48.6 MB | 0.97 GB | 9.7 GB | 97.3 GB | 88,293 |
| int4(mean) | 748.3 × 128 | 49.40 KB | 0.5 KB | 7.8x | 24.7 MB | 0.49 GB | 4.9 GB | 49.4 GB | 173,898 |
| binary | 748.3 × 128 | 11.98 KB | 0.0 KB | 32.0x | 6.0 MB | 0.12 GB | 1.2 GB | 12.0 GB | 716,966 |
| hierarchical_merge(0.25) | 195.4 × 128 | 100.04 KB | 0.0 KB | 3.8x | 50.0 MB | 1.00 GB | 10.0 GB | 100.0 GB | 85,868 |
| hierarchical_merge(0.25) > int8(per_vector) | 195.4 × 128 | 25.41 KB | 0.0 KB | 15.1x | 12.7 MB | 0.25 GB | 2.5 GB | 25.4 GB | 338,111 |
| hierarchical_merge(0.25) > binary | 195.4 × 128 | 3.13 KB | 0.0 KB | 122.6x | 1.6 MB | 0.03 GB | 0.3 GB | 3.1 GB | 2,740,997 |
| project(64) > int8(per_vector) | 748.3 × 64 | 49.40 KB | 32.8 KB | 7.8x | 24.7 MB | 0.49 GB | 4.9 GB | 49.4 GB | 173,897 |
| hierarchical_merge(0.25) > project(64) > binary | 195.4 × 64 | 1.57 KB | 32.8 KB | 245.1x | 0.8 MB | 0.02 GB | 0.2 GB | 1.6 GB | 5,467,987 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 3.1 KB / disk 100.4 KB | 0 | – | RAM 1.6 MB / disk 50 MB | 0.031 / 1.00 GB | 0.31 / 10.0 GB | 3.1 / 100 GB | 2,740,997 |

#### colqwen25_infovqa (float32 375.0 KB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor | 500 (MEASURED) | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM (PROJECTED) |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 732.5 × 128 | 375.03 KB | 0.0 KB | 1.0x | 187.5 MB | 3.75 GB | 37.5 GB | 375.0 GB | 22,904 |
| float16 | 732.5 × 128 | 187.52 KB | 0.0 KB | 2.0x | 93.8 MB | 1.88 GB | 18.8 GB | 187.5 GB | 45,807 |
| int8(per_vector) | 732.5 × 128 | 95.23 KB | 0.0 KB | 3.9x | 47.6 MB | 0.95 GB | 9.5 GB | 95.2 GB | 90,202 |
| int4(mean) | 732.5 × 128 | 48.35 KB | 0.5 KB | 7.8x | 24.2 MB | 0.48 GB | 4.8 GB | 48.4 GB | 177,657 |
| binary | 732.5 × 128 | 11.73 KB | 0.0 KB | 32.0x | 5.9 MB | 0.12 GB | 1.2 GB | 11.7 GB | 732,456 |
| hierarchical_merge(0.25) | 191.6 × 128 | 98.10 KB | 0.0 KB | 3.8x | 49.0 MB | 0.98 GB | 9.8 GB | 98.1 GB | 87,566 |
| hierarchical_merge(0.25) > int8(per_vector) | 191.6 × 128 | 24.91 KB | 0.0 KB | 15.1x | 12.5 MB | 0.25 GB | 2.5 GB | 24.9 GB | 344,795 |
| hierarchical_merge(0.25) > binary | 191.6 × 128 | 3.07 KB | 0.0 KB | 122.3x | 1.5 MB | 0.03 GB | 0.3 GB | 3.1 GB | 2,795,052 |
| project(64) > int8(per_vector) | 732.5 × 64 | 48.35 KB | 32.8 KB | 7.8x | 24.2 MB | 0.48 GB | 4.8 GB | 48.4 GB | 177,656 |
| hierarchical_merge(0.25) > project(64) > binary | 191.6 × 64 | 1.54 KB | 32.8 KB | 244.7x | 0.8 MB | 0.02 GB | 0.2 GB | 1.5 GB | 5,575,541 |
| two-tier (Ward 1/4 > binary hot, int8 cold) | – | RAM 3.1 KB / disk 98.3 KB | 0 | – | RAM 1.5 MB / disk 49 MB | 0.031 / 0.98 GB | 0.31 / 9.8 GB | 3.1 / 98 GB | 2,795,052 |

#### ColEmbed 4B docvqa (float32 7.77 MB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor (codes) | 500 | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 758.3 × 2560 | 7765.1 KB | 0.00 MB | 1.0x | 3.883 GB | 77.65 GB | 776.5 GB | 7765 GB | 1,106 |
| int8(per_vector) | 758.3 × 2560 | 1942.8 KB | 0.00 MB | 4.0x | 0.971 GB | 19.43 GB | 194.3 GB | 1943 GB | 4,421 |
| binary | 758.3 × 2560 | 242.7 KB | 0.00 MB | 32.0x | 0.121 GB | 2.43 GB | 24.3 GB | 243 GB | 35,398 |
| hierarchical_merge(0.25) > int8(per_vector) | 190.1 × 2560 | 487.0 KB | 0.00 MB | 15.9x | 0.243 GB | 4.87 GB | 48.7 GB | 487 GB | 17,639 |
| project(640) > int8(per_vector) | 758.3 × 640 | 486.8 KB | 6.55 MB | 16.0x | 0.250 GB | 4.87 GB | 48.7 GB | 487 GB | 17,630 |
| hierarchical_merge(0.25) > project(640) > binary | 190.1 × 640 | 15.2 KB | 6.55 MB | 510.7x | 0.014 GB | 0.16 GB | 1.5 GB | 15 GB | 564,191 |
| two-tier (binary hot, int8 cold) | – | RAM 242.7 KB / disk 2.19 MB | 0 | – | RAM 121 MB / disk 1.09 GB | 2.43 / 21.9 GB | 24.3 / 219 GB | 243 / 2185 GB | 35,399 |
| two-tier (ward 1/4 > binary hot, int8 cold) | – | RAM 60.8 KB / disk 2.00 MB | 0 | – | RAM 30 MB / disk 1.00 GB | 0.61 / 20.0 GB | 6.1 / 200 GB | 61 / 2004 GB | 141,229 |

#### ColEmbed 4B infovqa (float32 7.56 MB per page)

| configuration | vectors/page × dim | bytes/page | shared (fixed) | factor (codes) | 500 | 10k | 100k | 1M (PROJECTED) | pages within 8 GB RAM |
|---|---|---|---|---|---|---|---|---|---|
| float32 | 738.0 × 2560 | 7556.8 KB | 0.00 MB | 1.0x | 3.778 GB | 75.57 GB | 755.7 GB | 7557 GB | 1,136 |
| int8(per_vector) | 738.0 × 2560 | 1890.7 KB | 0.00 MB | 4.0x | 0.945 GB | 18.91 GB | 189.1 GB | 1891 GB | 4,543 |
| binary | 738.0 × 2560 | 236.2 KB | 0.00 MB | 32.0x | 0.118 GB | 2.36 GB | 23.6 GB | 236 GB | 36,373 |
| hierarchical_merge(0.25) > int8(per_vector) | 184.9 × 2560 | 473.8 KB | 0.00 MB | 16.0x | 0.237 GB | 4.74 GB | 47.4 GB | 474 GB | 18,131 |
| project(640) > int8(per_vector) | 738.0 × 640 | 473.8 KB | 6.55 MB | 16.0x | 0.243 GB | 4.74 GB | 47.4 GB | 474 GB | 18,116 |
| hierarchical_merge(0.25) > project(640) > binary | 184.9 × 640 | 14.8 KB | 6.55 MB | 510.8x | 0.014 GB | 0.15 GB | 1.5 GB | 15 GB | 579,901 |
| two-tier (binary hot, int8 cold) | – | RAM 236.1 KB / disk 2.13 MB | 0 | – | RAM 118 MB / disk 1.06 GB | 2.36 / 21.3 GB | 23.6 / 213 GB | 236 / 2127 GB | 36,374 |
| two-tier (ward 1/4 > binary hot, int8 cold) | – | RAM 59.2 KB / disk 1.95 MB | 0 | – | RAM 30 MB / disk 0.97 GB | 0.59 / 19.5 GB | 5.9 / 195 GB | 59 / 1950 GB | 145,164 |

#### Scan latency, laptop CPU (MEASURED at 500 pages; PROJECTED beyond under the V1 linearity assumption)

| model | configuration | 500 MEASURED ms/query | 10k PROJECTED | 100k PROJECTED | 1M PROJECTED |
|---|---|---|---|---|---|
| ColPali | float32 | 22.23 | 516 ms | not projected (index 53 GB > 8 GB) | not projected (index 528 GB > 8 GB) |
| ColPali | int8(per_vector) | 22.75 | 445 ms | not projected (index 13 GB > 8 GB) | not projected (index 134 GB > 8 GB) |
| ColPali | binary | 22.66 | 449 ms | 4.49 s | not projected (index 17 GB > 8 GB) |
| ColPali | hierarchical_merge(0.25) | 6.22 | 124 ms | not projected (index 13 GB > 8 GB) | not projected (index 135 GB > 8 GB) |
| ColPali | hierarchical_merge(0.25) > binary | 6.18 | 123 ms | 1.23 s | 12.25 s |
| ColPali | project(64) | 17.84 | 361 ms | not projected (index 26 GB > 8 GB) | not projected (index 264 GB > 8 GB) |
| ColQwen2 | float32 | 15.69 | 364 ms | not projected (index 38 GB > 8 GB) | not projected (index 383 GB > 8 GB) |
| ColQwen2 | int8(per_vector) | 15.91 | 311 ms | not projected (index 10 GB > 8 GB) | not projected (index 97 GB > 8 GB) |
| ColQwen2 | binary | 15.99 | 317 ms | 3.17 s | not projected (index 12 GB > 8 GB) |
| ColQwen2 | hierarchical_merge(0.25) | 4.27 | 85 ms | not projected (index 10 GB > 8 GB) | not projected (index 100 GB > 8 GB) |
| ColQwen2 | hierarchical_merge(0.25) > binary | 4.35 | 86 ms | 863 ms | 8.62 s |
| ColQwen2 | project(64) | 12.40 | 251 ms | not projected (index 19 GB > 8 GB) | not projected (index 192 GB > 8 GB) |
| ColQwen2.5 | float32 | 15.79 | 366 ms | not projected (index 38 GB > 8 GB) | not projected (index 383 GB > 8 GB) |
| ColQwen2.5 | int8(per_vector) | 16.06 | 314 ms | not projected (index 10 GB > 8 GB) | not projected (index 97 GB > 8 GB) |
| ColQwen2.5 | binary | 16.00 | 317 ms | 3.17 s | not projected (index 12 GB > 8 GB) |
| ColQwen2.5 | hierarchical_merge(0.25) | 4.22 | 84 ms | not projected (index 10 GB > 8 GB) | not projected (index 100 GB > 8 GB) |
| ColQwen2.5 | hierarchical_merge(0.25) > binary | 4.39 | 87 ms | 870 ms | 8.70 s |
| ColQwen2.5 | project(64) | 12.39 | 250 ms | not projected (index 19 GB > 8 GB) | not projected (index 192 GB > 8 GB) |

#### Two-tier, ColQwen2 DocVQA (hot Ward 1/4 > binary PROJECTED linear; rescoring MEASURED)

| pages | hot stage ms/query (PROJECTED) | old rescoring ms/query (MEASURED, flat 500-4,000) | total with old rescoring | hot share |
|---|---|---|---|---|
| 500 | 4.4 | 11.2 | 15.6 | 28% |
| 1,000 | 8.7 | 11.2 | 19.9 | 44% |
| 10,000 | 86.3 | 11.2 | 97.5 | 88% |
| 100,000 | 862.5 | 11.2 | 873.8 | 99% |
| 1,000,000 | 8625.0 | 11.2 | 8636.2 | 100% |
