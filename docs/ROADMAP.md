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

**1. Run the real ViDoRe benchmark on a GPU.**
The single highest-value next step and mostly a compute problem, not a code problem.
`optivision fetch-vidore` and `configs/colpali.yaml` already exist; what is missing is
a machine that can encode a few thousand pages. Report ColPali-v1.3 numbers on
`vidore/docvqa_test_subsampled` and friends, against the published baselines.

**2. Close the binary-quantization gap — now the single biggest lever.**
The Stage-I measurement located the loss precisely: pruning costs ~4% of nDCG@5,
binary quantization costs ~12%, and pruning harder costs almost nothing on top. So the
highest-value work is no longer "prune better", it is "quantize better". Concretely:

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

**3. Learned saliency instead of hand-tuned weights.**
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
