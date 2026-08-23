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
| int8-only | 875.0 | 112.00 | 4.0x | 0.7877 | 0.5694 | 100.7% | 0.972 |
| prune+int8 | 246.8 | 31.59 | 14.2x | 0.7511 | 0.5139 | 96.0% | 0.864 |
| binary-only | 875.0 | 14.00 | 32.0x | 0.6875 | 0.4167 | 87.9% | 0.585 |
| optivision | 246.8 | 3.95 | 113.5x | 0.6782 | 0.4028 | 86.7% | 0.606 |
| optivision-aggressive | 186.3 | 2.98 | 150.3x | 0.6680 | 0.3611 | 85.4% | 0.602 |
| keep-10pct | 162.9 | 2.61 | 171.8x | 0.6700 | 0.4167 | 85.6% | 0.551 |

Re-running the benchmark reproduces these figures exactly — the pipeline is
deterministic given a fixed corpus and encoder.

Three findings, and the third one is the interesting one.

> **Read this section as "E1: what ColSmol-256M says on generated pages", not as the
> final answer.** Every finding below was re-run under ColPali-v1.3 on real ViDoRe
> pages; two did not transfer and the third reversed. The corrections are in
> [*The same table on ColPali-3B and real ViDoRe pages*](#the-same-table-on-colpali-3b-and-real-vidore-pages)
> further down, finding by finding. Both are kept because the disagreement between them
> is the result.

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
| best quality per byte | prune + int8 | 14.2x | 96.0% | 0.864 |

`prune+int8` is in the variant list for exactly this reason. If an examiner asks "can
you get most of the saving for less loss?", that row is the answer — and note its tau
of 0.866 is *identical* to `spatial+redundancy`, meaning int8 adds no measurable
ranking distortion on top of the pruning. All of the reordering in the full pipeline
comes from the one-bit codec.

One honest note on that row: `prune+int8` scores 0.7511 against `spatial+redundancy`'s
0.7519, a gap of 0.0008 at 72 queries. Read it as "int8 quantization is free on top of
pruning", not as a measurable cost. (An earlier version of this table had the two the
other way round, with int8 apparently *improving* nDCG@5; that came from the
pre-rescaling int8 codec and the inversion went away with it.)

**Caveat on magnitudes — and it turned out to be the whole story.** ColSmol-256M is a
256M-parameter model; its embeddings are lower-dimensional in effect than ColPali-3B's
and are plausibly less robust to one-bit quantization. The binary penalty measured here
should be treated as an upper bound until the same table is produced on ColPali with a
GPU.

That run has since happened, and it confirmed this caveat rather than the finding it
qualifies: the 12.1% binary penalty collapses to at most 3.7% on ColPali. We are
leaving the caveat in its original wording, written *before* the run, because the
prediction being on record is worth more than a tidy edit.

## The same table on ColPali-3B and real ViDoRe pages

The caveat above turned out to be the important one. Repeating the ablation with
the reference encoder on the benchmark everyone else reports changes two of the
three findings above, and reverses the third.

| | |
|---|---|
| Encoder | `vidore/colpali-v1.3-merged` (ColPali), bfloat16, Tesla T4 |
| Corpus | four ViDoRe test splits, 500 pages each except tabfquad (280) |
| Index | exact brute-force MaxSim (`NumpyIndex`), as before |
| Tokens/page before compression | 1031 |
| Stack | torch 2.10.0+cu128, transformers 5.15.1, peft 0.19.1, colpali-engine 0.3.17, datasets 5.0.0 |
| Reproduce | `bash scripts/run_bench_gpu.sh` - see [GPU_RUN.md](GPU_RUN.md) |

Per split, the full pipeline against its own uncompressed baseline:

| split | character | pages | queries | baseline nDCG@5 | optivision | retained | tau |
|---|---|---|---|---|---|---|---|
| `docvqa_test_subsampled` | scanned industry documents | 500 | 451 | 0.5841 | 59.2x | 94.6% | 0.510 |
| `syntheticDocQA_energy_test` | clean synthetic reports | 500 | 100 | 0.8403 | 59.9x | 103.4% | 0.696 |
| `infovqa_test_subsampled` | infographics | 500 | 494 | 0.8458 | 53.4x | 97.2% | 0.655 |
| `tabfquad_test_subsampled` | tables | 280 | 280 | 0.5246 | 57.8x | 95.6% | 0.674 |

nDCG@5 retained, by variant and split:

| variant | docvqa | energy | infovqa | tabfquad | compression |
|---|---|---|---|---|---|
| `spatial-only` | 99.2% | 101.1% | 99.8% | 99.8% | 1.0-1.4x |
| `spatial+redundancy` | 97.4% | 101.8% | 99.7% | 99.2% | 1.7-1.9x |
| `int8-only` | 99.9% | 101.6% | 100.1% | 99.9% | 4.0x |
| `binary-only` | 96.3% | 100.6% | 97.4% | 100.3% | 32.0x |
| `prune+int8` | 97.4% | 101.6% | 99.5% | 99.8% | 6.7-7.5x |
| `optivision` | 94.6% | 103.4% | 97.2% | 95.6% | 53.4-59.9x |
| `keep-50pct` | 92.5% | 101.8% | 94.7% | 97.4% | 80.7-91.4x |
| `keep-10pct` | 66.7% | 89.8% | 85.4% | 81.6% | 342.4-387.4x |

**1. Pruning is still close to free - and buys much less.** Both stages together
retain 97.4-101.8% of nDCG@5, better than on the generated corpus. But the token
reduction collapses from 3.5x to 1.7-1.9x. Real benchmark pages are far denser than
generated ones: on `infovqa`, spatial pruning keeps
1000 of 1031 patches and saves essentially nothing. The premise that a page
is mostly blank paper holds for documents we generated and largely fails on the
benchmark. **The claim survives; the number behind it does not.**

**2. Pruning harder is not free. This finding does not survive.** On the generated
corpus the keep-ratio sweep was flat, and we concluded the token budget barely
mattered. On ViDoRe it falls off a cliff:

| split | keep-50% | keep-10% |
|---|---|---|
| docvqa | 92.5% | 66.7% |
| energy | 101.8% | 89.8% |
| infovqa | 94.7% | 85.4% |
| tabfquad | 97.4% | 81.6% |

The flat sweep was an artifact of a corpus whose salient content fit inside the top
10% of patches. Dense real pages do not have that property, so the aggressive
setting is a genuine trade-off rather than a free saving.

**3. Binary quantization is not where the quality goes. This reverses.** On
ColSmol, `binary-only` cost 12.1% of nDCG@5 without dropping a token, and the
discussion was built around that asymmetry. On ColPali it costs almost nothing:

| split | binary-only retained | tau |
|---|---|---|
| docvqa | 96.3% | 0.527 |
| energy | 100.6% | 0.736 |
| infovqa | 97.4% | 0.643 |
| tabfquad | 100.3% | 0.686 |

The 256M model's embeddings were the fragile part, not the one-bit codec. A 3B
model's 128-d projections survive sign-thresholding nearly intact - which is what
the caveat in the previous section predicted, stated before the run rather than
after it.

Read the tau column alongside it: binary quantization still reorders results
substantially (0.53-0.74 tau). It moves the ranking; it just keeps the relevant
page inside the top 5. Tau is measuring real distortion that nDCG@5 is too coarse
to charge for.

> **Superseded by E3.** The paragraph above went on to explain the difference by
> *retrieval margin* - a stronger encoder separating the gold page from its
> neighbours, so the same perturbation moves fewer pages across the top-5 cut.
> E3 refutes that; see the next section. The observation that tau falls while
> nDCG@5 does not still holds. The explanation for it does not.

**The operating points invert.** On ColSmol, `prune+int8` was the quality-first
recommendation and binary was the compromise. On ColPali the full pipeline
dominates it - roughly 8x the compression at the same retention:

| goal | configuration | compression | nDCG@5 retained |
|---|---|---|---|
| smallest index | prune + binary | 53-60x | 94.6-103.4% |
| best quality per byte | prune + int8 | 6.7-7.5x | 97.4-101.6% |

**Two things not to over-read.** The `energy` split reports retention above 100%
for most variants, peaking at 103.4% - compression appearing to improve retrieval.
That split has only 100 queries, so those differences are noise. Read them as
"compression is free here", not as an improvement. Lead with `docvqa` (451
queries) and `infovqa` (494), which are the statistically meaningful splits.

And these are 500-page corpora. Absolute nDCG is not comparable to published ViDoRe
leaderboard numbers, which score the full splits; only the *ratios* between rows are
the measurement here, and those ratios are what the paper claims.
## E3 - ColPali-3B on the generated corpus

E1 and E2 swap the encoder and the corpus at once, so neither can say which one
moved the attribution. E3 is the missing cell: the reference encoder over E1's own
60 pages and 72 queries, regenerated at seed 7 so the pages are the same ones, not
merely similar ones. `reports/colpali_generated/`.

| | corpus fixed, encoder swapped | encoder fixed, corpus swapped |
|---|---|---|
| **one-bit codec costs** | E1 -> E3: **12.1 -> 1.6 points** | E3 -> E2: 1.6 -> 3.7 points |
| **pruning buys** | E1 -> E3: 3.55x -> 4.20x | E3 -> E2: **4.20x -> 1.85x** |

**The corpus sets what pruning buys.** That half is solid, and
`python scripts/compare_regimes.py` prints the table from the benchmark files.

**The codec half needed a per-query look, and the per-query look reversed the
reading** (2026-08-23; `scripts/review/q9_margin.py`, `reports/margin_summary.txt`;
the full argument is [REVIEW-2026-08-21.md](REVIEW-2026-08-21.md) sections 2 and 8).
An earlier version of this section said the encoder sets what the codec costs, and
that E3 refutes the margin explanation because ColPali is the weaker retriever on
these pages yet loses nothing. Both statements rest on aggregates. Per query, with
*m* = (gold score - best other score) / gold score under float and *sigma* = the
score noise a codec adds (residual around a linear fit to the float scores):

| | ColSmol-256M (E1) | ColPali-3B (E3) |
|---|---|---|
| sign-codec noise sigma | 2.8% of score | 2.5% |
| median *m*, 60 precise queries | -0.05% | -1.4% |
| median *m*, 12 topical queries | +31% | +47% |
| queries flipped by sign, \|m\|/sigma < 0.5 | 32% | 43% |
| queries flipped by sign, \|m\|/sigma > 2 | **0%** | **0%** |
| precise R@1, float -> sign | 0.483 -> 0.317 | 0.250 -> 0.250 (7 won->lost, 7 lost->won) |

The codec adds the same noise under both encoders. E1's precise queries are decided
by three digits inside a nine-character code at a 0.05% margin, so a 3% perturbation
flips a third of them. ColPali cannot read an 11 pt code at 448 px: it wins 15 of
the 60 precise queries in float (the subject-only floor is 12), its margin on them is
negative, and what the sign code does is reshuffle queries it had already lost.
"Loses nothing" is a floor effect, not robustness. Render only the code's glyphs 3x
larger (`make-corpus --code-scale 3`, `reports/colpali_generated_code3x_shifted/`)
and ColPali's precise R@1 rises to 0.317 and its one-bit cost from 1.6 to 4.1 points
(72 queries, inside the CI; the per-query structure is the evidence, not the
aggregate).

The rule that holds on every cache we have - six ColSmol, two ColPali: **a codec
costs the queries whose float margin is smaller than its score noise.** It is
verified on the generated corpus and is a prediction, not a measurement, on ViDoRe.
rho = cos(d, sign d) is 0.800 on every cache against a random reference of 0.798 and
says nothing; drop it.

**Caveats, both real.** E3's `hit@5` is 1.000 on nearly every row, so nDCG@5 can
only move by reordering inside the top 5. E1's is 0.972 on the same corpus, so
E1 vs E3 remains a fair comparison; it is E2, at 0.661, that sits on a different
scale - which is why R@1 is quoted above. And a 3B model scoring *below* a 256M
model on these pages is a fact about the generator, not about the encoders. It
belongs in the limitations.

## E4 - the font-scale control and the codec ladder (ColSmol, CPU)

`make-corpus --code-scale` re-renders E1's 60 pages with only the unique code's
glyphs at 0.4x or 3x; every other glyph, the RNG sequence and `queries.json` are
byte-identical (pixel diff 0.15% / 0.93%), and the tiled ColSmol re-encodes them
(~27 min CPU each, `reports/colsmol_code04x/`, `reports/colsmol_code3x/`,
`scripts/review/q8_fontscale.py`). Every comparison is paired per query.

| tiled ColSmol, 875 tok | float nDCG@5 / R@1 | precise float R@1 | sign retain | one-bit loss [95% CI] | paired vs 1x |
|---|---|---|---|---|---|
| code 0.4x (4.4 pt) | 0.701 / 0.403 | 0.283 | 90.4% | -0.067 [-0.125, -0.009] | +0.028 [-0.029, +0.086] |
| code 1x (11 pt, E1) | 0.782 / 0.569 | 0.483 | 87.9% | -0.095 [-0.159, -0.033] | - |
| code 3x (33 pt) | 0.718 / 0.444 | 0.333 | 91.0% | -0.065 [-0.126, -0.008] | +0.030 [-0.043, +0.101] |

Glyph size moves the *float* retriever in both directions (a 33 pt code spans three
patches, so no single document vector holds the string a query vector must match)
and does not move the one-bit cost. The codec's cost is not about legibility or
evidence concentration either; it is the margin rule above. ColSmol's tokens are
local - rows the manipulation did not touch have cos(1x, variant) = 1.00 - and a
58 pt layout shift re-encodes half the page (cos 0.35-0.55 on the moved rows), which
is why the first-round `*_shifted` / `*_label` runs are kept separately.

Codec ladder, retention of float nDCG@5 (`scripts/review/codec_ladder.py`,
`reports/ladder_*.txt`; CIs are paired bootstrap vs the sign code):

| codec | bytes/vec | E1 | infovqa (494 q) | docvqa (451 q) | E3 |
|---|---|---|---|---|---|
| int8 | 128 | 100.7% | 100.1% | 99.9% | 100.0% |
| 2-bit Lloyd-Max, rotated | 32 | 94.5% | 99.4% | 97.0% | 99.4% |
| sign(d) - the pipeline | 16 | 87.9% | 97.4% | 96.3% | 98.4% |
| sign(d - mu) | 16 | 91.1% | 98.3% (+0.7 [-0.2, +1.8]) | 94.9% (-0.8 [-2.3, +0.7]) | 99.1% |
| sign(ITQ(d - mu)) | 16 | 93.6-94.3% | 98.3% (+0.7 [-0.6, +2.0]) | 95.9% (-0.2 [-2.0, +1.6]) | 99.3% |
| centroid K=4096 + 1-bit residual | 17.5 | 92-100% (seed-dependent) | 98.8% (+1.2 [-0.1, +2.5]) | **89.1% (-4.2 [-6.5, -2.1])** | 97.7% |

What holds everywhere: int8 is lossless at 4x; 2-bit recovers about half the sign
loss at 16x; the sign code costs 2.6-3.7 points on ViDoRe and changes the top-1 page
of 11% of infovqa queries (top-1 agreement 0.89, top-5 overlap 0.77). What does not
transfer: the zero-byte refinements that gain 3-10 points on E1 are inside +/-1 point
on ViDoRe, and the residual code is significantly worse on docvqa (the K cap, 126
vectors per centroid, not PLAID's codec). E1's gains come from blank-page anisotropy
(||mean|| 0.57 on E1 against 0.23 on infovqa; blank-patch distractor promotion
34.7% -> 46.8% on E1, 7.9% -> 8.2% on infovqa), which is a corpus property.

## Token selection on dense pages - the Stage-II controls

`CODEBOOK=1` on infovqa and docvqa (`reports/colpali_{infovqa,docvqa}_test_subsampled/`).
All rows one-bit; budgets within 3% of each other except `cb-kmeans` (15-20% fewer
tokens, not a fair row). `cb-keep` = 256 greedy farthest-point probes from a 20k-patch
sample (the designed selector); `cb-random` = 256 random unit vectors that have never
seen the corpus.

| budget | pixel keep-N | cb-keep (designed) | cb-random | cb-kmeans |
|---|---|---|---|---|
| infovqa 50% | 94.7% | 96.2% | **96.7%** | 95.2% |
| infovqa 30% | 92.3% | 95.9% | 95.7% | 94.6% |
| infovqa 10% | 85.4% | 89.6% | **90.7%** | 88.7% |
| docvqa 50% | 92.5% | 91.4% | **92.3%** | 86.8% |
| docvqa 30% | 86.9% | 88.2% | **90.5%** | 83.0% |
| docvqa 10% | 66.7% | **77.9%** | 76.9% | 64.9% |

Random probes are within +/-1 point of the designed probes at every budget on
infovqa and ahead by 1-2 points at 50% and 30% on docvqa. The probe construction has
no measurable effect; what beats pixel saliency below 50% (by 4-11 points at 10%) is
selecting by embedding coverage instead of ink, which is Ma et al. (ACL Findings
2025, arXiv:2506.04997) "random pruning beats saliency pruning" reproduced. The
winner sets say there is nothing for a selector to find on dense pages: 74% / 66% of
patches win at least one MaxSim, and the winner-only index fitted on half the queries
is 64% / 52% of the index (1.6x / 1.9x) at 100% / 99.7% retention on the other half.
On E1 the same "100% held-out" figure was subject leakage (85-96% on subject-disjoint
splits). Any learned selector for this regime has to beat random dropping at a
matched budget.

## A caution about the tau column

`scripts/tau_audit.py` reproduces E1's published 0.585 exactly and shows what it
is: tau-b over the intersection of two **top-10** hit lists, because that is what
`bench` hands to `rank_correlation`. It is not rank agreement over the corpus,
which the same run puts at 0.643.

```
top-5     tau 0.402    4 queries fell back to tau=1.0
top-10    tau 0.585    <- the published number
top-20    tau 0.691
top-60    tau 0.643    (whole corpus)
```

Three things follow. The cutoff moves the number, so quote tau with its k.
`rank_correlation` returns 1.0 when fewer than two ids are common, so two rankings
sharing nothing score as perfect agreement - harmless on E1 at k=10, not obviously
harmless on a 500-page split with `hit@5` of 0.661. And top-10 covers 17% of a
60-page corpus against 2% of a 500-page one, so **E1's 0.585 and E2's 0.527 are
not the same measurement** and should not be compared as though they were. E1 vs
E3 is unaffected: same pages, same queries, same cutoff.

Where one number is needed, report top-1 agreement and top-5 overlap with the float
ranking instead (or alongside): they have no cutoff and say what a user would see.
`scripts/review/codec_ladder.py` prints both, with tau_AP for a correlation.

## Reading the table

| column | meaning |
|---|---|
| `Tok/pg` | vectors actually stored per page (after both pruning stages) |
| `KB/pg` | index bytes per page |
| `Compr.` | index bytes vs. the uncompressed float32 index |
| `nDCG@5`, `R@1`, `Hit@5` | absolute retrieval quality against ground truth |
| `Retain` | nDCG@5 as a fraction of the `baseline-float32` row |
| `Tau` | Kendall tau-b against the baseline's ranking, over the shared ids of the two **top-10** lists - not the whole corpus; see the caution above |
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
