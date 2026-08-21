# Roadmap — where Stage-I stops and Stage-II starts

## What Stage-I delivers (done)

- The full pipeline: encode → spatial prune → redundancy prune → binary quantize →
  index → MaxSim search, with the model left unmodified.
- Three encoder backends (ColSmol / ColPali / ColQwen2) behind one interface, plus an
  offline stand-in so the pipeline is testable with no download.
- Two index backends: an exact brute-force reference and Qdrant multivector MaxSim.
- An ablation harness that encodes once and replays every compression setting, with
  both absolute quality and rank-agreement against the uncompressed baseline.
- A corpus generator with exact ground truth, and a ViDoRe loader for the real benchmark.
- Figures showing which patches were dropped, a CLI and a demo UI.
- 75 tests and a clean lint.

## Stage-II candidates, roughly in order of value

**1. Run the real ViDoRe benchmark on a GPU. — DONE, and it changed the ordering
below.**
Four splits under ColPali-v1.3 on a free T4: 1,780 pages, 1,325 queries, archived in
`reports/colpali_*/` and reproducible with `bash scripts/run_bench_gpu.sh`. Two of the
three Stage-I findings did not transfer and one reversed. Read
[RESULTS.md](RESULTS.md#the-same-table-on-colpali-3b-and-real-vidore-pages) before
picking up anything below, because the reference encoder moved which lever matters.

**1b. Separate encoder scale from corpus provenance. — DONE, and it refuted an explanation.**
Ran as `MODE=generated bash scripts/run_bench_gpu.sh`; results in
`reports/colpali_generated/`, printed by `python scripts/compare_regimes.py`. The answer
was neither of the two we expected — it was both, on different axes:

```
codec cost, corpus fixed (E1 -> E3):  12.1 -> 1.6 points
codec cost, encoder fixed (E3 -> E2):  1.6 -> 3.7 points
prune gain, corpus fixed (E1 -> E3):  3.55x -> 4.20x
prune gain, encoder fixed (E3 -> E2): 4.20x -> 1.85x
```

The encoder sets what the codec costs; the corpus sets what pruning buys. It also
refutes the retrieval-margin explanation the paper offered, because ColPali is the
*weaker* retriever on those pages (nDCG@5 0.6954 against 0.7823) and still loses nothing
to one-bit codes. See [RESULTS.md](RESULTS.md#e3---colpali-3b-on-the-generated-corpus).

Two follow-ups this opened, both cheap:

- *Fix what tau reports.* `scripts/tau_audit.py` shows the published value is tau-b over
  the intersection of two top-10 lists, not over the corpus, and that it moves with the
  cutoff. `metrics.rank_correlation` should take a candidate pool and stop returning 1.0
  for disjoint lists. This changes published numbers, so land it with a note.
- *Measure the geometry.* If ColPali loses less to sign-thresholding than ColSmol does,
  that should be visible in the patch embeddings themselves — how much of each vector's
  norm sits in near-zero components. That is the mechanism the margin story was standing
  in for, and it needs the encode caches, which the runner does not currently archive.

**1c. Rate allocation in retrieval space, not pixel space. — the Stage-II thesis.**
Every compression stage in Stage-I asks a pixel-space question: does this patch have
ink, does it duplicate a neighbour. But a late-interaction score is a *sum over query
tokens of a max over patches*, so a patch contributes exactly nothing unless it is the
argmax for some query token. Its quantization error is multiplied by zero. Uniform
allocation therefore provably spends bits it cannot recover, and the waste grows with
the patch count.

How large is the dead set? `scripts/winner_stats.py` measures it on cached embeddings
and needs no relevance labels, so it runs on any corpus. On E1's 60 pages under
ColSmol:

```
patches that ever win a MaxSim, over all 72 queries: 4,424 / 52,500 = 8.4%
  per page: median 8.3%  min 7.1%  max 10.3%
```

91.6% of the index is encoded, quantized, stored and never read. And the set is a
property of the page rather than the query: fitted on 36 queries it is 8.1% of
patches, and retrieving the *other* 36 queries against a winner-only index gives
nDCG@5 0.7865 against 0.7865 for the full index — 12.4x smaller at no measured cost,
where the hand-tuned spatial detector gets 2.46x and pays 2.8 points.

The stacking result is the one that matters, because it attacks the stage the paper
blames. Binarization collides sign vectors and promotes patches that were never
competitive, so removing the dead set removes the distractors it invents:

```
binary, full index   0.6749
binary, winners only 0.7003     <- pruning *recovers* quality the codec lost
```

That is a different claim from "prune, then quantize, and hope the losses do not
compound." It says the two stages interact, with a sign the current framing cannot
express. What to build, in order:

- *A label-free win-rate estimator.* The oracle above uses the evaluation queries,
  which a deployment does not have. Replace them with a query codebook — k-means
  centroids over query-token embeddings from any corpus, or the encoder's own text
  token embeddings — and measure what fraction of the oracle winner set it recovers.
  This is the step that decides whether any of this is deployable.
- *Distil it into a head.* A linear probe or two-layer MLP on the patch embedding
  predicting win rate, supervised by MaxSim outcomes rather than by human saliency.
  One forward pass at index time, no queries needed. This is item 3, given a
  supervision signal it did not previously have.
- *Non-uniform bit allocation.* Drop / 1-bit / int8 / fp16 tiers assigned by predicted
  win rate under a Lagrangian size budget, instead of one precision for everything.
  The ~8% that decide rankings keep precision while the mean rate stays near one bit.
  The current pipeline is the budget-matched baseline.
- *The unification.* Winner fraction is a per-(encoder, corpus) scalar computable
  without labels. If codec cost tracks it, E1 and E2 stop being two contradictory
  points and become two samples of one curve — which is the explanation the paper
  currently reports as an open question. Run `winner_stats.py` on the ViDoRe caches to
  find out; the run needs the caches archived, which `run_bench_gpu.sh` does not
  currently do.

Caveat worth stating before anyone gets excited: E1's generated pages are sparse, and
60 pages is an easy retrieval problem. The 8.4% will rise on dense ViDoRe pages and
the held-out retention will fall below 100%. The size of that gap is the experiment.

**1d. Stage-II, started: probe coverage beats pixel saliency at a tight budget.**
`scripts/probe_eval.py` scores a patch selector against the oracle winner set on cached
embeddings in seconds, which is the loop this work needed — the full 13-variant
benchmark answers a question about the pipeline when the open question is about one
step. On E1, selection in isolation:

```
                       keep 30%            keep 10%
                  nDCG@5  oracle rec   nDCG@5  oracle rec
oracle (ceiling)  0.8016      100%     0.8016      100%
pixel             0.7980     69.5%     0.6372     28.7%
probe:kmeans      0.6571     30.6%     0.3602      9.6%
probe:random      0.7738     65.7%     0.6738     37.9%
probe:farthest    0.7672     67.6%     0.7164     35.9%
```

**Coverage is the property that matters, not fit.** k-means puts probes where patches
are dense, which is where the *redundant* patches are; a query is a rare, specific
direction. Greedy farthest-point sampling maximises the minimum angle between probes
(max pairwise cosine 0.19 against k-means' 0.36) and is now the default.

End to end on E1, with the redundancy stage and the one-bit codec on top:

| budget | pixel | farthest | random | kmeans |
|---|---|---|---|---|
| keep 50% | 85.7% | 86.0% | 83.3% | 84.2% |
| keep 30% | **85.9%** | 85.1% | 84.9% | 82.7% |
| keep 10% | 85.6% | **88.2%** | 81.0% | 78.6% |

At keep-10% that is **+2.5 points over pixel saliency and the best retention of any
one-bit configuration on E1** — above `binary-only` at 87.9%, which is the
winner-set effect from 1c showing up in the real pipeline: dropping patches that never
win removes the distractors binarisation invents.

Read it with the caveat it deserves. 2.5 points over 72 queries is under two queries,
and E1 is the incumbent's best case. What makes it worth continuing is the *shape* —
the advantage appears exactly where the budget is tight and pixel statistics run out,
which is what the argument predicts.

Next, in order:

- *The dense-page test.* `--codebook` on `infovqa`, where the pixel detector returns
  1.03x and has nothing to remove. 494 queries, so the power problem goes away too.
  This is the run that decides whether Stage-II is a paper.
- *Text-derived probes.* Queries are text and `encode_queries` already exists, so the
  most query-like probes are one encode away. Needs the probes cached beside the encode
  cache, or a warm-cache replay has no encoder to build them with.
- *Then bit allocation.* Selection is the first half of 1c; spending bits by predicted
  win rate rather than uniformly is the second, and it has not been tried.

**1e. The dense-page test ran, and Stage-II holds where it was supposed to.**
`CODEBOOK=1 SPLITS="vidore/infovqa_test_subsampled" bash scripts/run_bench_gpu.sh`.
500 pages, 494 queries, the split where pixel saliency returns 1.03x and has nothing
left to remove. Full pipeline, retention of baseline nDCG@5:

| budget | pixel | farthest | random | kmeans |
|---|---|---|---|---|
| keep 50% | 94.7% | 96.2% | **96.7%** | 95.2% |
| keep 30% | 92.3% | **95.9%** | 95.7% | 94.6% |
| keep 10% | 85.4% | 89.6% | **90.7%** | 88.7% |

**Every probe variant beats pixel saliency at every budget**, by +1.4, +3.6 and +4.2
points for the farthest-point default, and the gap widens as the budget tightens —
which is the shape the argument predicts. Compression is not exactly matched (134x
against 145x at keep-30%); interpolating pixel to 134x still leaves about +3 points.

Two corrections come with it, and both matter more than the win.

*The dead set is corpus-specific, and much smaller on real pages.* On E1, 8.4% of
patches ever win a MaxSim and the headroom is 11.9x. On `infovqa` it is **74.2%**, and
the headroom is **1.3x**. The oracle at 48% of patches scores 0.8390 against the full
index's 0.8410 — lossless, but at 2x, not 12x. "Most of the index is never read" is
true of sparse generated pages and false of dense real ones, so item 1c's framing needs
rewriting before it goes anywhere near the paper.

*The patch-geometry prediction in Section VI-A looks wrong.* It says the reference
encoder loses less to one-bit codes because more of each patch vector survives the
sign. Measured, rho is 0.7991 for ColPali on `infovqa` against 0.8001 for ColSmol on
E1 — the same number, both sitting on the random-unit reference of 0.7979 — and
ColPali flips *more* arg maxes, 76.6% against 56.1%. That is evidence against the
mechanism the paper now offers. It is not yet decisive, because those two runs differ
in corpus as well as encoder; the controlled comparison is geometry on E3, which is a
`MODE=generated` re-run away and cheap.

Next:

- *Geometry on E3*, to settle Section VI-A against a proper control rather than a
  cross-corpus comparison.
- *`docvqa` with `CODEBOOK=1`*, to see whether the win replicates on a second dense
  split before it is written up.
- *Bit allocation*, still untried, and now the more promising half: if 74% of patches
  win on real pages, dropping them is not where the gain is — spending fewer bits on
  the ones that win rarely might be.

**2. Close the binary-quantization gap — a big lever under a *small* encoder only.**
Under ColSmol the measurement located the loss precisely: pruning costs ~4% of nDCG@5,
binary quantization costs ~12%, and pruning harder costs almost nothing on top. Under
ColPali-3B that gap is largely gone — `binary-only` retains 96.3-100.6% — so this work
is worth doing for edge/small-model deployments and is *not* the top lever if you are
running a reference-scale encoder. Concretely:

- *Two-bit or product quantization on the document side.* Sits between binary (32x)
  and int8 (4x). If it recovers most of the 12% at ~16x it dominates both current
  operating points. `compression/` is where the codec goes; the ablation table already
  has a slot for the row.
- *Rescoring the shortlist.* Fetch top-k with binary codes, rescore with int8 or float
  vectors for those pages only. `SearchConfig.rerank` and `prefilter_k` already exist
  (see item 4) — this is the cheapest way to test whether the loss is a ranking
  problem or a recall problem.
- *Diagnose before optimising.* Is the 12% concentrated in the topical queries (where
  candidates are near-identical) or spread evenly? `benchmark.json` holds the per-query
  runs; splitting the metric by query type answers this in a few lines and would say
  whether a shortlist rescore can help at all.

**3. Learned saliency instead of hand-tuned weights. — promoted: this is the binding
limit under a reference-scale encoder.**
ViDoRe made the case. Spatial pruning returns 1.03x on `infovqa` against 1.42x on the
sparse `energy` split: pixel statistics find almost nothing to discard on an
infographic, and pushing the budget down by hand costs 25.8 points of retention on
`docvqa` between keep-50% and keep-10%. A selector that knows which patches ever win a
MaxSim max is exactly what that regime is short of.

Ink density and edge energy with weights 0.6/0.4 is a deliberate first cut — cheap,
interpretable, no extra forward pass. A small model predicting "will this patch ever
win a MaxSim max?" from the patch embedding could prune harder at the same quality.
The honest comparison is against the current detector at matched token budget, which
the `keep_ratio` sweep already provides.

**4. Two-tier index: binary prefilter + float rerank.**
The scaffolding is in (`SearchConfig.rerank`, `rerank_cache`, `prefilter_k`,
`pipeline._rerank`) but it is not the default because keeping float vectors defeats
the compression. The interesting version keeps floats for the top ~1% of pages by
access frequency, or re-encodes candidates on demand. Measure the quality ceiling
first with the existing cache to see whether the gap is worth engineering.

**5. Per-page adaptive budgets.**
Currently either a global threshold or a global `keep_ratio`. A dense table page
genuinely needs more vectors than a title page, and the saliency histogram already
says which is which. Allocate a corpus-level token budget across pages by information
content rather than uniformly.

**6. Real scanned documents.**
Everything so far assumes clean renders. Skew correction, bleed-through and JPEG
artefacts all change the ink statistics that spatial pruning depends on. Worth testing
the saliency estimator on genuinely degraded scans before claiming the applications
listed in the abstract (land records, old patient reports).

## Things deliberately not on the list

- **Fine-tuning or distilling the encoder.** It would break the central claim that
  this is a drop-in change for an existing ColPali deployment.
- **Answer generation on top of retrieval.** Downstream of this project; it would
  dilute a Stage-I scope that is already complete.
- **A custom ANN index.** Qdrant does that job, and the measurement path deliberately
  avoids ANN so that quality numbers isolate the compression.
