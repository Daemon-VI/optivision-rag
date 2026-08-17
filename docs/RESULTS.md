# Results

## How to reproduce

```bash
optivision make-corpus data/corpus --docs 30 --pages 2
./run_bench.sh reports/colsmol          # or: make bench
```

Outputs land in `reports/colsmol/`: `benchmark.md` (the table), `benchmark.json`
(every field, including the ones the table omits) and `run.log`.

## Experimental setup

| | |
|---|---|
| Encoder | `vidore/colSmol-256M` (ColIdefics3), float32, CPU |
| Hardware | 8 GB DDR4 single-channel laptop, no GPU |
| Corpus | 60 generated pages (30 documents x 2 pages), A4 at 150 dpi |
| Queries | 72 — 60 *precise* (subject + unique code, one relevant page) and 12 *topical* (subject only, several relevant pages) |
| Index | exact brute-force MaxSim (`NumpyIndex`), not ANN |
| Vector dim | 128 |
| Tokens/page before compression | 875 (768 page-grid patches + 64 thumbnail + 43 instruction) |

**Why an exact index.** Every quality number comes from exhaustive scoring, so a
change in the metric can only be caused by pruning or quantization. An ANN index
would add its own recall loss and confound the measurement. The Qdrant backend is
tested separately (`tests/test_qdrant.py`) and is the deployment path, not the
measurement path.

**Why one encoding pass.** `EncodedCorpus` encodes the corpus once and every ablation
row replays over the cached vectors. Nothing but the compression settings differ
between rows.

**Two query families.** Precise queries contain a rare identifier that dominates the
match — they check that compression did not break retrieval outright. Topical queries
name only a subject shared by several pages, so the metric depends on *ordering* a
group of near-identical candidates; that is where compression damage shows.

## What the numbers say

The generated table is [`reports/colsmol/benchmark.md`](../reports/colsmol/benchmark.md).
Measured run, 60 pages / 72 queries, ColSmol-256M on CPU:

| variant | tok/pg | KB/pg | compr. | nDCG@5 | R@1 | retain | tau |
|---|---|---|---|---|---|---|---|
| baseline-float32 | 875.0 | 448.00 | 1.0x | 0.7823 | 0.5694 | 100.0% | 1.000 |
| spatial-only | 356.1 | 182.32 | 2.5x | 0.7602 | 0.5694 | 97.2% | 0.935 |
| spatial+redundancy | 246.8 | 126.34 | 3.5x | 0.7519 | 0.5139 | 96.1% | 0.866 |
| int8-only | 875.0 | 112.00 | 4.0x | 0.7787 | 0.5556 | 99.5% | 0.973 |
| prune+int8 | 246.8 | 31.59 | 14.2x | 0.7596 | 0.5278 | 97.1% | 0.866 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6875 | 0.4167 | 87.9% | 0.585 |
| optivision | 246.8 | 3.95 | 113.5x | 0.6782 | 0.4028 | 86.7% | 0.606 |
| optivision-aggressive | 186.3 | 2.98 | 150.3x | 0.6680 | 0.3611 | 85.4% | 0.602 |
| keep-10pct | 162.9 | 2.61 | 171.8x | 0.6700 | 0.4167 | 85.6% | 0.551 |

Re-running the benchmark reproduces these figures exactly — the pipeline is
deterministic given a fixed corpus and encoder.

Three findings, and the third one is the interesting one.

**1. Pruning is close to free.** Removing blank patches costs 2.8% of nDCG@5 for a
2.5x token reduction; adding redundancy collapsing brings it to 3.5x for 3.9%. Rank
agreement with the uncompressed baseline stays at tau = 0.87-0.94. This is the central
claim of the abstract and it holds.

**2. Pruning harder is nearly free too.** The `keep-N pct` sweep is almost flat:
keeping the top 10% of patches (163 vectors/page) scores 0.6700, and keeping the top
50% (288 vectors/page) scores 0.6704. Once quantization is in play, the token budget
barely moves quality — so the aggressive setting is not a trade-off, it is close to a
free 1.8x.

**3. Binary quantization, not pruning, is where the quality goes.** `binary-only`
loses 12.1% of nDCG@5 *without dropping a single token*, and its tau of 0.585 says it
substantially reorders results. Meanwhile `int8-only` gives 4x for 0.5%. Every
configuration that includes binary quantization lands in a narrow 85-88% band
regardless of how much pruning is applied — the binary floor dominates.

That is a genuinely useful result for the project rather than a flattering one: the
two halves of the proposal do not contribute symmetrically. Spatial and redundancy
pruning are the safe, high-confidence part. Binary quantization buys the dramatic 32x
but is responsible for essentially all of the quality loss.

**The practical consequence** is that the pipeline has two defensible operating
points, and which one is right depends on the deployment:

| goal | configuration | compression | nDCG@5 retained | tau |
|---|---|---|---|---|
| smallest index | prune + binary | 113.5x | 86.7% | 0.606 |
| best quality per byte | prune + int8 | 14.2x | 97.1% | 0.866 |

`prune+int8` is in the variant list for exactly this reason. If an examiner asks "can
you get most of the saving for less loss?", that row is the answer — and note its tau
of 0.866 is *identical* to `spatial+redundancy`, meaning int8 adds no measurable
ranking distortion on top of the pruning. All of the reordering in the full pipeline
comes from the one-bit codec.

One honest note on that row: `prune+int8` scores 0.7596 against `spatial+redundancy`'s
0.7519, i.e. quantizing appears to *improve* nDCG@5 slightly. That is noise at 72
queries, not a real effect — read it as "int8 quantization is free here", not as
"int8 helps".

**Caveat on magnitudes.** ColSmol-256M is a 256M-parameter model; its embeddings are
lower-dimensional in effect than ColPali-3B's and are plausibly less robust to
one-bit quantization. The binary penalty measured here should be treated as an upper
bound until the same table is produced on ColPali with a GPU — which is item 1 on the
[roadmap](ROADMAP.md).

## Reading the table

| column | meaning |
|---|---|
| `Tok/pg` | vectors actually stored per page (after both pruning stages) |
| `KB/pg` | index bytes per page |
| `Compr.` | index bytes vs. the uncompressed float32 index |
| `nDCG@5`, `R@1`, `Hit@5` | absolute retrieval quality against ground truth |
| `Retain` | nDCG@5 as a fraction of the `baseline-float32` row |
| `Tau` | Kendall tau-b against the **baseline's own ranking** |
| `q ms` | median query latency over the whole corpus |

`Tau` is the load-bearing column. Absolute metrics answer "is this system any good?",
which depends mostly on the model. Tau answers "did compression change the model's
mind?", which is the only question this project controls.

## Variants

| variant | pruning | quantization | what it isolates |
|---|---|---|---|
| `baseline-float32` | off | none | ColPali as published — the reference |
| `binary-only` | off | binary | quantization damage alone |
| `int8-only` | off | int8 | the 4x middle ground |
| `spatial-only` | blank-patch | none | pruning damage alone |
| `spatial+redundancy` | both stages | none | cost of collapsing duplicates |
| `optivision` | both stages | binary | the full proposal |
| `optivision-aggressive` | 25% budget | binary | a hard token budget |
| `keep-N pct` (`--sweep`) | top N% | binary | the quality/size curve |

## Storage arithmetic

The headline number the project is aimed at:

```
baseline   875 vectors x 128 dims x 4 bytes  =  448 KB / page
                                                448 GB / million pages

optivision  ~N vectors x 128 bits            =  see benchmark.md
```

`bytes_per_page` and `gb_per_million_pages` are both in `benchmark.json`, computed
from the pipeline's own byte accounting rather than from files on disk, so the figure
is not inflated by container overhead or deflated by filesystem compression.

## Caveats we are not hiding

1. **The corpus is generated, not scanned.** Its whitespace profile is realistic and
   its ground truth is exact, but real scans add noise, skew and bleed-through.
   `optivision fetch-vidore` pulls the actual ViDoRe benchmark to test that; the
   harness treats both identically.
2. **ColSmol-256M is not ColPali-3B.** Absolute quality is lower than the published
   ColPali numbers. The compression behaviour is what transfers — the pruning and
   quantization stages never touch the model.
3. **Synthetic-encoder numbers are not results.** `reports/synthetic/benchmark.md`
   exists to prove the plumbing; its hashed word vectors are near-orthogonal, MaxSim
   behaves like exact matching and every quality column saturates at 1.0. Only `Tau`
   is meaningful there. Do not put that table in the report.
4. **60 pages is a small corpus.** Latency at this scale is dominated by fixed
   overhead, and nDCG has coarse resolution. The storage ratios are scale-independent;
   the quality figures would tighten with more pages.
