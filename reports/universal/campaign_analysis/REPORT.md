# Campaign analysis: patterns across 55 measured models

Exploratory, post-hoc analysis of the Kaggle measurement campaign (`reports/universal/campaign`,
`src/optivision/presets/measured.json`), run on 2026-10-17. Nothing here was preregistered. Each
analyst ran about 35 to 195 tests, about 1,100 in all, with no multiplicity correction. Treat every
result as hypothesis-generating unless it is marked as resting on many independent models. No
repository file other than this folder was written.

## How it was done

- **Analysts.** Eleven analyst agents each took one question, wrote and ran their own code on the raw
  study files, and returned up to six findings with the numbers and the script that produced them.
- **Literature.** One literature agent mapped published work.
- **Skeptics.** Each of the 66 findings went to an independent skeptic agent, which rewrote the
  computation from the raw files and tried to refute it. 65 survived with caveats and 1 was refuted.
  The skeptics' corrected wording is what this report uses.
- **Positioning and gaps.** A positioning agent compared the surviving findings with the literature,
  and a critic agent listed the gaps.
- **Records.** Every finding, verdict and correction is in `findings.json`. The scripts are in
  `scripts/<question>/` and the skeptics' recomputations in `scripts/verify/`. The scripts use the
  absolute local paths they were run with.

**Data.** There are 95 model-dataset units from 55 models: 80 image units on ViDoRe V1 DocVQA and
InfoVQA (500 pages each) and 15 text units on BEIR SciFact. The selection study covers 32 models (59
files).

**Caveats that apply throughout:**

- Modality and dataset are the same variable, so no text-versus-image difference can be attributed
  to modality.
- Corpora are small (500 pages).
- Intervals are bootstraps over queries and ignore corpus sampling.
- Sibling checkpoints are correlated: roughly 13 to 15 independent organisations for images and
  about 10 families for text.
- Retention above 1 is noise.

## 1. Results that rest on many models and are directly usable

**1.1 int8 is the corpus-agnostic default.**
- `int8(per_vector)` (3.9x) has a retention lower bound of at least 0.95 on 88 of 88 units with the
  full grid. The worst is LateOn/SciFact at 0.966.
- It reaches at least 0.99 on 85 of 88 units.

**1.2 Use int4 with the mean subtracted (`int4(mean)`), never plain int4.**
- `int4(mean)` (7.8x) holds at a lower bound of at least 0.95 on 87 of 88 units. The exception is
  LateOn/SciFact, with a lower bound of 0.92 and a point estimate of 0.95.
- Plain int4 has a lower bound under 0.95 on 9 of 88 units.
- It collapses on three same-release pairs: nemotron-colembed-vl 4b/8b (retention 0.004 to 0.011 on
  both datasets), mxbai-edge-colbert 17m/32m (0.27 and 0.79) and LateOn/mLateOn (0.30 and 0.77).
- Centring restores all of them to 0.95 to 1.00.
- Other checkpoints sharing a backbone or organisation do not collapse: GTE-, Reason- and ColBERT-Zero
  ModernColBERT, mxbai-colbert-large, and EVIE at 2048 and 4096 dims. So no registry attribute
  predicts the collapse: not dimension (Spearman 0.09), not inference dtype, not family.
- The release line predicts it, but that rests on only 3 clusters.
- The cost of centring is one mean vector per corpus (10 to 16 KB for 2560- to 4096-d), not per
  document.
- The mechanism is untested. It could be a few outlier coordinates setting the per-vector scale, or a
  shared offset, because the vectors were not kept.

**1.3 Choosing by point estimate fails; nominal confidence is not the held-out miss rate.** In the
selection study, about 3,500 query splits per rule, the held-out miss rates were:

| Rule | Held-out miss rate | Notes |
|---|---|---|
| point estimate | 48% | met 0.66 / 0.54 / 0.37 at targets 0.95 / 0.97 / 0.99 |
| lower 0.975 | 12.5% | |
| lower 0.99 | 8.7% | |
| lower 0.999 | 2.9% | about 16% more compression than the default, about 1 point more misses |
| bonferroni 0.975 | 2.2% | |
| `calibrate()` default | 1.8% | worst model exactly 0.950; misses are small (mean shortfall 0.56 points, maximum 2.1) |

- Miss rates are much higher when the chosen pipeline sits within 0.5 points of the target (59%).
- The default and lower 0.999 disagree in 21.5% of splits. In those splits lower 0.999 picks about
  2x more compression and misses 7.1% against 2.1%; 18 models favour the default and 0 favour lower
  0.999.
- The early-stopping guard (`assume_monotone`) is the likely reason, but n_boot also differs, so
  this is not isolated.
- These rates come from re-splitting the same queries, not from new corpora.

**1.4 DocVQA is the stricter dataset; safe sets transfer one way only.**
- A pipeline safe on DocVQA (lower bound at least 0.95) is safe on InfoVQA about 99% of the time. The
  reverse holds only 74 to 78% of the time.
- The gap is concentrated in aggressive pipelines: binary is 0.015 lower on DocVQA in 33 of 37
  models, while int8 and int4 show no gap.
- It cannot be separated from DocVQA's lower baseline (0.61 against 0.92).

**1.5 recall@1 degrades first.** On average, retention ranks recall@1 < ndcg@5 < ndcg@10 < recall@5
on all three datasets. 13 to 50% of rows that are ndcg@5-safe at the lower bound fail another metric.
A top-1 use case needs a margin.

**1.6 Scan time follows the number of vectors, not bytes.**
- In this harness (dense MaxSim on decoded vectors), log relative scan time ≈ 0.91 × log relative
  vector count.
- Codecs shrink memory but not scan time; merging to 0.25 gives about 3.8x faster scans.
- This says nothing about real binary or int8 kernels.

**1.7 Per-corpus calibration is where the compression is.**
- The best fixed pipeline that holds on at least 95% of units is `int4(mean)` at 7.8x pooled. Per
  dataset it is 3.9x on SciFact, 15.7x on DocVQA and 63.8x on InfoVQA.
- `calibrate()`'s median choice is about 73x and meets 0.95 on held-out splits 98.3% of the time,
  about 6 to 9x the best fixed default on the same pairs.

## 2. Moderate results, useful for design

**2.1 Stage losses roughly multiply, which makes them a screen but not a certificate.**
- Predicting a combined pipeline's retention as the product of its single stages gives a mean error
  of about 0.009, with 91% within 0.02. That is no better than adding the losses.
- At T = 0.95 the screen has precision 0.97 and recall 0.98, but precision is lower on DocVQA (0.87)
  and SciFact (0.77) at T = 0.98.
- The prediction breaks when any stage alone loses more than 5% (mean error 0.022, and 0.11 below
  0.80).
- After merging, uncentred binary, int4 or lloyd2 lands 3.6 to 7.1 points below the product on
  SciFact models, driven by LFM2, mxbai-edge, LateOn and mLateOn.
- `int8` and `int4(mean)` compose cleanly.
- Use it to shortlist, then measure.

**2.2 Token merging bends between keep 0.25 and 0.1.**
- Loss accelerates as the keep fraction falls, with no sharp knee.
- Image models lose about 1% at keep 0.25 and 1.2 to 1.7 points more at 0.1. Text loses far more.
- The number of vectors kept per page tracks loss somewhat better than the fraction kept, which
  explains part of the text-versus-image gap: text documents start with fewer vectors.
- Express a merge budget as a minimum number of vectors kept (roughly 150 to 250 for under 1% loss on
  these datasets) rather than as a fraction.
- `hierarchical_merge(0.1)` actually keeps 21 to 27% of vectors on colSmol-500M and colsmolvlm.

**2.3 Projection depends on the absolute number of kept dimensions, not on its ratio to the
original.**
- The dimension needed for a given retention grows only about as d^0.2 to d^0.3.
- `project(128)` costs about 0.5 to 1.4% at 320-d, 1.7 to 2.4% at 640-d and about 4.2 to 4.4% at
  4096-d. 4096-d needs `project(256)` to reach about 98.5%.
- The PCA fitted on the corpus beats a single random projection by about 1.6 points at a quarter of
  the dimensions (mostly on DocVQA).
- Random projection to 32 dimensions collapses on the LightOn and LFM text checkpoints (under 0.05)
  while PCA does not. The PCA is fitted on the corpus it is scored on, so these figures are
  optimistic.

**2.4 Binary is hardest at 128-d.**
- Image models above 128-d lose about 0.7 points against 2.3 for 128-d, but this is a step, not a
  trend with dimension.
- It is confounded with newer families and float16 inference.
- The binary penalty is a repeatable model property across DocVQA and InfoVQA (rho about 0.58,
  falling to 0.45 with one model per organisation).
- Which codec ends up cheapest at a threshold does not transfer between datasets.

**2.5 Tiered binary-hot plus int8-cold saves RAM, not storage.**
- A binary first stage with int8 rescoring of 50 candidates holds a lower bound of at least 0.95 on
  84 of 88 units (74 of 74 image), against 62 of 88 for binary alone.
- Total storage is 3.4 to 3.8x, no better than int8 alone.
- On the 500-page sets the shortlist is 10% of the corpus. On SciFact it is 1%, and that is where
  tiering fails (10 of 14), so the image result may not hold at realistic corpus sizes.

**2.6 A `/recommend` lookup is defensible only as a ladder with stated hit rates.**
- In a leave-one-model-out test, the pipeline every other model passed almost never fails, but it is
  simply `int4(mean)` / `int8` / `float16`.
- Buckets by dimension and modality recover 30 to 48x for 128-d image models. They hit about 94 to
  96% on point retention but only 64 to 76% on the lower bound.
- Every other bucket (text, 320-d and above) has too few peers to recommend anything.
- Family matching and nearest-neighbour metadata add nothing beyond dimension and modality.

**2.7 A calibration that needs no relevance labels is possible.**
- Using the baseline's own top-1 page as the reference, the default made 0 misses against real
  labels across about 175 cells. The 95% upper bound is about 1.7% per cell.
- It costs about 0.42x the compression of label-based selection.

## 3. Weak or negative results (do not build on these)

- **Family does not beat the image-wide mean** as a predictor of a held-out model's profile: family
  explains about a fifth of profile variance, but predicts no better than the image-wide mean.
- **Size and dtype show no consistent effect** on merge and codec pipelines: parameter count and
  float16 inference are within about 0.5 points. Larger siblings lose about 1 point more on
  aggressive projection.
- **"More vectors per page means more redundancy" is a between-family contrast.** It is driven by
  ColNetraEmbed and vanishes within families.
- **Small text dimension (48 to 64-d) as an int4 risk** rests on 3 models, 2 of them siblings.
- **The geometry reports cannot test the int4 mechanism**, because none of the collapsed models has
  them.
- **Refuted: "text sub-multiplicativity persists at tiny losses."** It does not: the shortfall is
  -0.004 when both stages lose under 1%.

## 4. Catalogue and documentation issues found

1. **Metrics are mixed.**
   - Text catalogue rows are ndcg@10 (the builder's choice), image rows are ndcg@5, and the
     selection study uses ndcg@5 for text.
   - Switching the metric changes the pass/fail verdict for 27 of 126 SciFact rows at 0.99.
2. **Compression is always measured against float32 bytes.** That includes the 13 to 14 models whose
   baseline is float16 inference, where the saving over what they would really store is half the
   quoted ratio.
3. **The two colbertv2.0 entries behave differently.**
   - colbert-ir (157 vectors per document, ndcg@10 0.646) and lightonai (236 vectors, 0.692) are
     probably the same weights with a different document length.
   - colbert-ir is the most downloaded model in the listing (42% of downloads).
4. **Four entries passed `check_retriever` at exactly the 10-of-12 minimum**, including
   webAI-ColVec1.1-8b. For text models the check also tracks model quality (recall@1), so it is not a
   pure test of whether the model loaded correctly.
5. **Unrecorded run settings.**
   - The random projection seed is not recorded, so every model with the same dimension shared one
     draw.
   - The text inference dtype is not recorded (`auto`).
6. **Display.** Retention above 1 should be shown as tied, not as an improvement.

None of these were changed in this pass.

## 5. Relation to published work (positioning agent; absence of a source is not a novelty claim)

- **Consistent with published work:**
  - The codec tiers, and binary being lower on DocVQA (ColBERTv2 residual coding; Vespa's binary
    ColPali at about 94.5%).
  - The token-pooling curve (Clavié et al. 2024: about 97% at a pooling factor of 4; Light-ColPali).
  - Scan time following vector count in a dense harness.
- **Extends published work** to many post-hoc checkpoints: binary loss falling with dimension
  (Qdrant notes 1-bit at 128-d is hard), and PCA beating random projection (single-vector studies;
  jina-colbert-v2's trained Matryoshka head).
- **No source found:**
  - The lineage-specific plain-int4 collapse and its repair by centring.
  - The multiplicative screening of combined pipelines.
  - The held-out behaviour of the lower-bound selection rules across many models.
  - The one-way DocVQA/InfoVQA transfer.
  - The recall@1-first ordering under compression.

Sources are in `literature_digest.md`.

## 6. What would most change these conclusions

From the critic, in order of value:

1. **Break the modality/dataset confound.** Run 6 to 8 text models on a second text corpus (NFCorpus,
   FiQA, TREC-COVID) and 4 to 6 image models on a non-ViDoRe-V1 page set, with the same pipeline
   grid.
2. **Test at a larger corpus size** (5,000 to 20,000 pages, or full SciFact subsampling), where the
   tiered, merge and selection results are most likely to break.
3. **Save a few hundred vectors** for the collapsed and intact models (nemotron, LateOn,
   mxbai-edge, EVIE). Compute offset norm, the maximum-to-RMS ratio per vector and the share of
   variance in the top coordinates, and predict int4 collapse before measuring it on new models.
4. **Use more seeds and replicates.** Run 5 or more random-projection seeds and repeat runs of 6 to 8
   models, to establish noise floors for comparing siblings.
5. **Preregister these patterns as predictions for the next batch.** For example: `int4(mean)` holds
   at a lower bound of at least 0.98 on new models; plain-int4 collapse occurs in 10% of models or
   fewer; `hierarchical_merge(0.25) > int8` passes 0.95 on at least 90% of new image models; the
   `calibrate()` default misses no more than 3%. This turns the post-hoc findings into tests.
6. **Real kernels and a fairer grid.** Time real binary and int8 kernels, and add trained compressors
   (PLAID residuals, trained pooling) as reference points.
