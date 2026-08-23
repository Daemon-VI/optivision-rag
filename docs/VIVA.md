# Viva notes

Questions an examiner is likely to ask, and the answer this codebase supports.
Every claim here is checkable — the file to open is named.

---

**Q. What exactly is your contribution? ColPali already exists.**

Two halves, and the second is the one to lead with.

**The artifact.** The model is stock and the query path is untouched. The contribution
sits entirely between "model emitted vectors" and "vectors went into the index": drop
the patch vectors that sit on blank paper, collapse the ones that duplicate each other,
and store the survivors as sign bits. A team already running ColPali can adopt it
without retraining or re-encoding their pipeline logic. See `src/optivision/pipeline.py`
— the whole contribution is two calls between the encoder and the index.

**The measurement.** Everyone who prunes *and* quantizes reports the product of the two
savings. Nobody reports the ratio of the two costs. We measured it twice — ColSmol-256M
on generated pages, ColPali-3B on four ViDoRe splits — and got **opposite** answers from
the same code. Under the small encoder the one-bit codec costs 12.1% and pruning is
nearly free; under the reference encoder the codec costs at most 3.7% and what fails is
the assumption that a page is mostly blank paper. So the attribution is not a property
of the compression layer at all — it is a property of the queries and the corpus: the
corpus decides what pruning can remove, and the codec costs exactly the queries decided
by a margin smaller than its score noise, which is the same ~3% under both encoders.
That is why a compression ratio published without naming the setting is not enough
information to act on. That is the paper's claim, and it is stronger than the one we
set out to make.

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

*This is the question the project turns on, and the answer changed twice — once after
the real encoder, once after the per-query review. Give the current one.*

**It depends on the setting, and demonstrating that — and saying which part of the
setting — is the result.** Pruning's yield is set by the corpus; the codec's cost is set
by how many queries are decided by less than the codec's score noise.

On **ColSmol-256M with generated pages**, pruning does the work. Blank-patch and
redundancy pruning give 3.5x fewer vectors for 3.9% of nDCG@5 at tau = 0.87; binary
quantization gives 32x but loses 12.1% *without dropping a single token*, and every
configuration containing it lands in a narrow 85-88% band no matter how much pruning is
applied.

On **ColPali-3B with real ViDoRe pages**, that inverts. `binary-only` costs at most
3.7% (96.3% retained on `docvqa`, ~100% on two splits) — the codec is nearly free. What
stops working is pruning: spatial pruning returns **1.03x on infographics** against
1.42x on the sparse `energy` split, because there is simply nothing on the page to
discard. And the token-budget sweep that was flat on ColSmol becomes a **25.8-point**
drop from keep-50% to keep-10% on `docvqa`.

**The run that explains the reversal — and the answer has changed twice, so get the
current one right.** The first draft said the codec distorts both encoders equally and
only the stronger encoder has the *margin* to absorb it. The second draft (after E3) said
margin cannot be the mechanism because ColPali is the weaker retriever on those pages and
still loses nothing, so the encoder's patch geometry must decide the cost. **Both are
wrong, and the second is wrong in a way an examiner can see from the E3 table.** The
current answer, verified per query on eight encode caches (`scripts/review/q9_margin.py`,
`docs/REVIEW-2026-08-21.md` sections 2 and 8):

E3 runs ColPali-3B over E1's exact 60 pages and 72 queries, so the corpus is held fixed
and only the encoder changes:

| | corpus fixed, encoder swapped | encoder fixed, corpus swapped |
|---|---|---|
| one-bit codec costs | E1 → E3: **12.1 → 1.6 points** | E3 → E2: 1.6 → 3.7 points |
| pruning buys | E1 → E3: 3.55x → 4.20x | E3 → E2: **4.20x → 1.85x** |

**The corpus sets what pruning buys — that half is solid. The codec half is about the
queries, not the encoder.** A sign code adds the same noise to a page's score under both
encoders: about 2.8% of the score for ColSmol, 2.5% for ColPali (2-bit: half that; int8:
0.06%). What differs is how close each query is to a tie. E1's 60 precise queries are
decided by three digits inside a nine-character code, and the float retriever wins them
by a median margin of **0.05%** of the score — a 3% perturbation is a coin flip on those.
On every cache, for every codec, **no query whose margin exceeds twice the codec's noise
changes its top result**, and 32–43% of those below half the noise do.

Why ColPali "loses nothing": it cannot read an 11 pt code at 448 px. It wins 15 of the 60
precise queries in float, where a subject-only ranker gets 12 by chance, and its margin
on them is *negative* (−1.4%). Under the sign code 7 flip from won to lost and 7 from
lost to won, which nets to the "unchanged" R@1 of 0.375. It is a floor, not robustness.
Render only the code's glyphs 3x larger and ColPali's precise R@1 rises to 0.317 and its
one-bit cost to 4.1 points — the same mechanism on the encoder that looked immune (at
72 queries that aggregate is inside its CI; the per-query structure is the evidence).

The one-sentence answer: **a codec costs the queries whose float margin over the best
competitor is smaller than the codec's score noise; E1 is the worst case by construction,
E3 is a floor, and E2's 2.6–3.7 points measure how many real queries are within ~3% of a
tie.** If asked what "the encoder's patch geometry" has to do with it: nothing we could
measure — rho = cos(d, sign d) is 0.800 on every cache against 0.798 for random vectors,
and the arg-max flip rate is 75–83% everywhere. Say plainly that an earlier version of
the paper offered that mechanism and the measurement removed it.

**Two follow-ups an examiner may raise.** (1) *Does the rule hold on ViDoRe?* It is
verified on the generated corpus and predicted on ViDoRe, where we did not retain the
caches; say so. (2) *What about the cheap fixes to the sign code?* Centring, an ITQ
rotation and a centroid-plus-residual code recover 3–10 points on E1 and are inside ±1
point on ViDoRe with CIs (the residual code is −4.2 on docvqa at our centroid cap). What
holds everywhere: int8 is lossless at 4x, and a rotated 2-bit code recovers about half
the sign loss at 16x.

If an examiner presses on tau itself, concede the point first: `scripts/tau_audit.py`
shows the published 0.585 is tau-b over the intersection of two **top-10** lists, not
over the corpus (0.643 for the same run), the value moves with the cutoff, and top-10 is
17% of a 60-page corpus against 2% of a 500-page one. So E1's 0.585 and E2's 0.527 were
never the same measurement. E1 vs E3 is clean — same pages, same queries, same cutoff —
which is exactly why the argument now rests on it.

**What it tells a deployer.** Under a small encoder the two operating points are a real
exchange — prune + int8 at 14.2x/96.0% against prune + binary at 113.5x/86.7%. Under the
reference encoder prune + binary gives roughly 8x the compression at the *same* nDCG@5,
so it simply dominates, and only tau separates them: take binary if the output feeds a
downstream reader, int8 if a human sees the ranked list. Both rows are in both tables.

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

It did, and then it did not. We ran the whole ablation again on the **real ViDoRe
benchmark** — `docvqa`, `infovqa`, `tabfquad` and `syntheticDocQA_energy`, 1,780 pages
and 1,325 queries under ColPali-v1.3 on a T4. The archived runs are in
`reports/colpali_*/` and `scripts/run_bench_gpu.sh` reproduces them; the harness treats
both corpora identically, as a folder of pages plus a queries file.

Answer the follow-up before it is asked: **the real benchmark contradicted us.** Two of
the three findings did not transfer and one reversed outright. We report both
experiments rather than quietly replacing the first, because the disagreement between
them is what the paper is about — see the contribution answer above.

The generated corpus stays in the paper for what it is good at: exact ground truth, and
a whitespace profile that is realistic for scanned forms even though the noise, skew and
bleed-through of a real scan are absent. Comparing the two is what isolated page density
as the variable that matters for pruning.

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

- **Two encoders is two points, not a curve.** We can say the attribution reverses
  between 256M and 3B; we cannot say where in between it crosses, nor whether the
  driver is parameter count, embedding quality, or the corpus change that comes with
  it in our design. Running ColPali on the generated corpus separates those and is the
  obvious next experiment — it is one command, and it is in the roadmap.
- **The ViDoRe runs use 500-page subsets** of each split, so absolute nDCG@5 is not
  comparable to published leaderboard figures. Only the ratios between rows are the
  measurement, and those are all we claim.
- **Statistical power on the small splits.** 72 queries in E1 and 100 on the `energy`
  split, where several rows read above 100% retention. That is noise, and we decline to
  read it as compression improving retrieval. `docvqa` (451) and `infovqa` (494) carry
  the conclusions.
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
