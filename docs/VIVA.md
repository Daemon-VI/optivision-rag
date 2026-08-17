# Viva notes

Questions an examiner is likely to ask, and the answer this codebase supports.
Every claim here is checkable — the file to open is named.

---

**Q. What exactly is your contribution? ColPali already exists.**

The model is stock and the query path is untouched. The contribution sits entirely
between "model emitted vectors" and "vectors went into the index": drop the patch
vectors that sit on blank paper, collapse the ones that duplicate each other, and
store the survivors as sign bits. A team already running ColPali can adopt it without
retraining or re-encoding their pipeline logic. See `src/optivision/pipeline.py` —
the whole contribution is two calls between the encoder and the index.

---

**Q. Why not just use OCR?**

OCR is what fails on the documents this is aimed at: old carbon copies, seals and
stamps over text, handwritten annotations, degraded microfilm, forms where layout
carries meaning. Vision retrieval reads the page as an image, which is why it works
there. We keep that property and attack only its cost.

---

**Q. Why not a single vector per page? That would be 1000× smaller.**

Because it loses the thing that makes patch retrieval accurate. A page vector has to
summarise an invoice's line items, its GSTIN and its subject line in 128 numbers.
Late interaction keeps a vector per region and lets each query term find its own best
region — `max_j q_i · d_j`. Our position is that patch retrieval is worth keeping and
only its *storage* needs fixing. `spatial-only` and `binary-only` rows in the ablation
show the two halves separately.

---

**Q. Doesn't dropping patches lose information?**

Only where there is none. The keep-mask is computed from ink density and edge energy
per patch — run `optivision explain` and look at the figure: green cells sit on the
text block and the footer, red cells on the margins and the empty lower half of the
page. The ablation quantifies what it costs: compare `spatial-only` to
`baseline-float32`.

---

**Q. How do you know a patch is blank? What about a grey photocopy?**

We do not assume the paper is white. The background level is estimated per page as the
95th percentile of intensity, and ink is measured relative to *that*. There is a test
for exactly this case — `test_grey_scan_is_not_all_ink` in `tests/test_pruning.py`
feeds a uniformly grey page and asserts the saliency stays near zero. Normalisation
also uses the 99th percentile rather than the max, so a single dust speck cannot
squash the rest of the page toward zero.

---

**Q. Which of your two ideas actually does the work?**

Pruning, and the measurement says so plainly. Blank-patch and redundancy pruning give
3.5x fewer vectors for 3.9% of nDCG@5, holding tau at 0.87. Binary quantization gives
32x but loses 12.1% *without dropping a single token* — every configuration containing
it lands in a narrow 85-88% band no matter how much pruning is applied.

The cleanest way to see it: `prune+int8` has a Kendall tau of 0.866 against the
baseline ranking — *exactly* the same as pruning with no quantization at all. So int8
adds no measurable reordering, and all of the reordering in the full pipeline comes
from the one-bit codec.

We report that rather than hiding it, because it is the finding that tells a deployer
what to do: prune + int8 (14.2x at 97.1% of baseline nDCG@5) when quality matters,
prune + binary (113.5x at 86.7%) when index size is the binding constraint. Both rows
are in the table.

---

**Q. Binary quantization throws away 31 of every 32 bits. Why does retrieval survive?**

Two reasons, and they are specific to this setting:

1. The vectors are L2-normalised and roughly zero-centred per dimension, so the sign
   keeps the orthant and discards only the position within it.
2. The distortion is nearly the *same for every document vector*, so it shifts all
   scores together. Retrieval only cares about order.

And we do not binarise both sides. The query stays float32 and is scored against ±1
document codes (asymmetric scoring). A corpus has millions of vectors; a query has
twenty — query precision is the cheapest thing to keep. `maxsim_hamming` in
`compression/binary.py` implements the fully-symmetric alternative so the ablation can
price the difference.

---

**Q. How do you prove compression did not hurt the ranking?**

Two separate measurements, because they answer different questions.

- **Absolute** — nDCG@5 and Recall@1 against ground truth: is the system any good?
- **Kendall tau against the uncompressed baseline's own ranking**: did compression
  change the model's mind? This is the honest measure of compression damage, and it
  stays informative even when the absolute metric saturates at 1.0.

Both are in the ablation table. `metrics.rank_correlation` implements tau-b.

---

**Q. Why is redundancy pruning safe? Averaging vectors usually destroys information.**

It usually does — for a single-vector retriever. Under MaxSim the score is a *max*, so
if two document vectors are within cosine 0.92, the max over the pair and the
similarity to their normalised mean differ by an amount bounded by their angular
spread. Collapsing a tight cluster barely moves the score. The stage is reported
separately in the ablation precisely so this claim is falsifiable rather than assumed.

---

**Q. Your benchmark uses generated documents. Isn't that cheating?**

The generated corpus gives exact ground truth and runs anywhere, and its whitespace
profile is the realistic part. But we say plainly what it cannot do: real scans have
noise, skew and bleed-through. `optivision fetch-vidore` pulls the actual ViDoRe
benchmark used by the ColPali paper for that reason, and the harness treats both
identically — a folder of pages plus a queries file.

The *synthetic encoder* comes with a stronger warning: its hashed word vectors are
near-orthogonal, so MaxSim behaves like exact matching and quality metrics saturate at
1.0. It is a correctness harness for the plumbing, not a source of results. This is
stated in the README, in the module docstring, and in the results doc.

---

**Q. What was the hardest bug?**

The patch-grid mapping. ColSmol does not emit one flat grid of image tokens — it cuts
an A4 page into 12 tiles of 8×8 patches plus a whole-page thumbnail, giving 13 runs of
64 tokens. The obvious `isqrt(832)` guess yields a 32×26 rectangle, which maps every
saliency score onto the wrong patch. It fails *silently*: the index builds, search
returns pages, quality is just quietly worse. The fix detects the run structure and
stitches the tiles into a true 32×24 grid
(`ColVLMEncoder._grid_for`). `optivision explain` exists largely so this is checkable
by eye.

---

**Q. What are the limits of what you built?**

- Encoding is the bottleneck: ~30 s/page for ColSmol-256M on a CPU laptop. Pruning
  plus quantization is ~15 ms/page, so the compression is effectively free — but
  indexing a large corpus needs a GPU.
- Redundancy merging changes vectors, not just their count. Nearly free under MaxSim,
  but not lossless.
- Quality numbers come from an exact brute-force index, deliberately, so they are not
  confounded by ANN recall. A production Qdrant deployment adds its own recall
  question on top.
- Stage-I scope is the storage side only. Reranking, multi-page documents and
  answer generation are downstream of this.
