# Follow-up analysis

## What this is

- **Data.** The follow-up batches (`f1` to `f8`, `f3b`), read alongside the campaign's wide,
  selection and geometry files. `SUMMARY.md` scores the 14 predictions; this report looks for
  anything else in the same data.
- **Method.** Six analysts each took one angle. A skeptic then re-ran each finding's numbers with
  their own script and gave it a verdict: confirmed, weakened (true once corrected) or refuted.
  30 findings: 3 confirmed, 26 weakened, 1 refuted.
- **What is reported.** Only the corrected claims are reported here. For the most useful finding
  (section 1) the key numbers were re-run a third time while this report was written.
- **Evidence label: exploratory and post hoc.** Nothing here was predicted in advance. Unit counts
  include models repeated across datasets and the scale units. "Safe" means a lower 95% bootstrap
  bound ≥ 0.95 on the same data, not a held-out guarantee.

## 1. Plain int4's collapse on answerai is a norm error, and renormalising costs no bytes

- **Setup.** answerai-colbert-small-v1 was re-encoded on CPU (vectors per doc 235.49, the same as
  the Kaggle run; baseline ndcg@10 0.369 against 0.367). Retrieval was exact MaxSim, with
  ndcg@10 retention.

| | NFCorpus | SciFact |
|---|---|---|
| plain int4 | 0.414 [0.361, 0.471] | 0.888 [0.85, 0.929] |
| float32 directions scaled by int4's decoded norms | 0.427 | 0.893 |
| int4, decoded vector renormalised | 0.970 | 1.001 |
| int4(mean) | 0.924 | — |
| int8 | 0.988 | — |

- **Mechanism.** The model's vectors are unit norm. Plain int4's per-vector max-abs scale gives
  decoded norms of 1.006 ± 0.011, and that norm noise alone reproduces the collapse.
  - Shuffling the norms across vectors is just as bad (0.387 / 0.853), so what matters is how
    much the norms vary, not which vector gets which norm.
- **The fix.** Store `amax * ||v|| / ||decoded||` in the float16 slot `Int4Quantizer` already
  writes. The decode is unchanged, there are no extra bytes, and the decoded-norm sd falls to
  0.00025.
- **Why NFCorpus, not SciFact.** The error has a similar size on both, but the signal does not.
  - NFCorpus queries have 8 real tokens against 22.7 on SciFact. Their [MASK] expansion tokens
    are nearly parallel (pairwise cosine 0.92), and the spread of true document scores is smaller
    (sd 0.094 against 0.188).
  - The mean per-query ratio of int4 score error to true score spread is 3.0 on NFCorpus and 1.0
    on SciFact.
  - Within each dataset, shorter queries lose more (Spearman about −0.2). Length-matched NFCorpus
    queries still lose more, so query length is a contributor, not the whole cause.
- **Limits.**
  - One model, two datasets, and a re-encode that is not byte-identical to the campaign vectors.
    The absolute values differ from the Kaggle run by 0.01 to 0.06.
  - Not tested on LateOn, mLateOn or mxbai-edge-17m. LateOn's problem is a dominant coordinate
    (top-coordinate second-moment share 0.29), which renormalising may not fix.
  - Renormalising combined with int4(mean) was not tested.
- **Scripts:** `scripts/int4_answerai/normfix.py`, `ablate.py`, `perquery.py` and
  `scripts/verify-answerai-collapse/own.py`.

## 2. A ladder that was safe in-sample, by modality

Out of 103 units (49 models; 25 text, 58 image at 128-d, 20 image wider than 128-d):

| Rung | Text | Image 128-d | Image > 128-d |
|---|---|---|---|
| float16 (2x) | 25/25 | 58/58 | 20/20 |
| int8 per-vector (about 4x) | 25/25 | 58/58 | 20/20 |
| int4(mean) (about 8x) | 23/25 | 58/58 | 19/20 |
| plain int4 (about 8x) | 13/25 | 58/58 | 20/20 |
| binary (32x) | 4/25 | 39/58 | 20/20 |
| merge(0.33) > int4(mean) (about 23x) | 8/25 | 56/58 | 19/20 |
| merge(0.25) > int4(mean) (about 31x) | 4/25 | 56/58 | 20/20 |

- **int4(mean) misses:**
  - LateOn on SciFact (lower bound 0.917).
  - answerai on NFCorpus (0.924).
  - EVIE-4.5B on TabFQuAD (0.929).
- **For text, stop at int8 or int4(mean) unless calibrating.** Plain int4, lloyd2 (6/25) and
  binary are not safe for text without measuring.
- **The fixed ladder leaves compression unused.** A per-unit oracle over the same rungs reaches a
  median of 15.7x (text), 96x (128-d image) and 128x (wide image).
- **Script:** `scripts/defaults/ladder2.py`.

## 3. Transfer between datasets

**Text, SciFact to NFCorpus (8 models)**
- **What transferred:** float16, int8 and merge(0.5), within about 0.02 with no disjoint CIs.
  int4(mean) and project(64) moved about 0.04 on average, but answerai's int4(mean) fell
  below the bar.
- **Binary lost 4 to 17 points on every model** (mean −0.082; 5 of 8 with disjoint CIs).
- **Plain int4 and lloyd2(7) swing by up to ±0.4 per model, in either direction.**
  - Worse on NFCorpus: answerai and mLateOn.
  - Worse on SciFact: LateOn and mxbai-edge-17m.
  - So "NFCorpus is harder for int4" does not hold.
- **The order of pipelines transfers; the level does not.**
  - Within a model, the SciFact order predicts the NFCorpus order (Spearman 0.87 to 0.98).
  - The NFCorpus deficit comes from binary and merge pipelines (mean −0.024 at 2.5 to 8x,
    −0.048 at 8 to 16.5x). Other pipelines are near zero up to about 16x.
- **A margin is needed when calibrating on one dataset.**
  - 26 of 127 pipelines safe on SciFact were not safe on NFCorpus (18 of 115 without answerai).
  - At a SciFact lower bound ≥ 0.97 the failures fell to 6 of 77, all of them answerai; the
    other 7 models had 0 of 67.
  - Most misses were small: 12 of 26 had a point value below 0.95, and 2 a lower bound below 0.90.
- **answerai compounds stage losses.** merge(0.33) alone fell from 0.998 to 0.949, and
  int4(mean) alone from 0.986 to 0.948. Chained, they fell from 0.987 to 0.836.
- **Not determined.** The cause of NFCorpus's larger drop. Baseline level, relevant documents
  per query (1.1 against 38), corpus size and domain are all confounded across two datasets.

**Pages (4 models, DocVQA, InfoVQA, arXivQA and TabFQuAD)**
- **Safe on every set for every model:** int8, int4 and merge(0.25) > int4 (23x for colSmol,
  30 to 32x for the others). EVIE's merge(0.25) > int4 was borderline (0.954).
- **Binary and lloyd2(7) depend on the model.** They failed for colSmol (0.919) and colpali
  (0.940), and passed for Vultron and EVIE.
- **Aggressive projection lost 3 to 6 more points on TabFQuAD** for 2 of 4 models
  (colSmol project(64), EVIE project(128) and (256)). It did not for colpali or Vultron. Merge
  rungs stayed between 0.99 and 1.02 everywhere.
- **Transfers that broke at the point estimate were rare.**
  - Of 318 correlated transfers, 7 had a point value below 0.95: 5 from colSmol on TabFQuAD and
    2 from EVIE.
  - EVIE's 39 lower-bound misses on TabFQuAD are mostly CI width.
  - InfoVQA alone was a worse calibration source than DocVQA alone.
- **TabFQuAD values above 1 are noise.** 43 of 295 rows are above 1.01, and the CI excludes 1 in
  only 14. The set has 280 queries and baseline recall@1 of 0.17 to 0.23.

## 4. Corpus size

- **Aggressive projection degrades with corpus size.** Projection to 25% of dimension or less
  lost a mean of 0.065 to 0.159 at a 2.9x (text) or 5x (pages) larger corpus.
  - Codec-only, merge > codec and merge > binary pipelines changed by a mean of −0.018 to +0.015.
  - The projection rows were already lossy (0.24 to 0.90), so "projection" and "already lossy"
    cannot be separated.
  - Individual non-projection rows can still drop 0.05 to 0.09 (answerai lloyd2).
- **Rows that started at retention ≥ 0.9 changed by a mean of −0.003 to −0.008.** A 0.03 margin
  missed 1, 2, 3 and 1 rows for the four models; the worst change was −0.071.
- **Tiered rows barely moved.** Binary hot index with int8 rescoring of 50 candidates changed by
  0.000 to −0.024 across 8 rows.
- **The text enlargement was a weak test.** The 10,000 out-of-domain FiQA documents left the
  baselines unchanged, yet projection rows still degraded on them.
- **`calibrate()` default.** Re-calibrated on the larger corpus, it met its target in 0.95 to
  1.00 of splits; the one-split differences are noise. It never chose more compression than on
  the small corpus.
  - This tests re-calibrating at the larger size, not calibrating small and deploying large.

## 5. Geometry as an early warning

- **A threshold on one vector statistic separates every int4 collapse in-sample.** The statistic
  is the second-moment share of the top 4 coordinates (sm4), and it needs no codec run.
  - sm4 > 0.15 to 0.18 flags all 10 units where plain int4 fell below 0.90 (5 models), and no
    other unit (36 units, 18 models).
  - The margin is thin and rests on one model. The highest non-collapse value is 0.163, the
    lowest collapse value 0.197, and only answerai is near the boundary. Leave-one-model-out
    misses all 3 answerai units.
- **Cosine to the corpus mean works only as a one-sided gate.** Above 0.90 it misses no collapse
  but raises false alarms. A 0.95 cut misses 4 units.
- **The int4 score-correlation statistic (P12/P14) adds nothing over the free statistics.** It
  scores AUC 0.94 against 1.00 for sm4, needs a cutoff of about 0.99, and misses mxbai-edge-32m
  in leave-one-model-out.
- **Geometry flags risk but does not forecast retention.** For the 4 collapse models measured on
  two corpora, sm4 is nearly identical across them while plain-int4 retention differs by 0.34 to
  0.44.
- **Nothing geometric predicted binary retention.** No statistic beat a per-model mean baseline.
- **Always using int4(mean) instead of int4 helps text and slightly hurts pages.**
  - It lifts the minimum from 0.235 to 0.948 and the mean from 0.883 to 0.994.
  - All the gain is on small text models. On the 15 page units the mean change is −0.0035, and
    EVIE TabFQuAD falls by 0.041.

## 6. Selection rules

- **The `calibrate()` default never did worse than "lower 0.999".** Over 66 unit-target pairs it
  was equal in 46 and higher in 20.
  - Pooled met rate at targets 0.95 / 0.97 / 0.99: 0.989 / 0.989 / 0.973, against
    0.966 / 0.968 / 0.948.
  - The only difference between the two rules is the per-family early stop. That it acts as
    multiplicity control is a hypothesis; nothing ablates it.
- **Lower-confidence rules under-deliver.** Lower 0.975 and lower 0.99 met the target in 0.83 to
  0.91 of splits.
- **Strict targets on small query sets.** The default's two misses at 0.99 are colSmol and
  Vultron on TabFQuAD (met 0.85 and 0.80).
- **What the default picks.** At 0.99 it is almost always float16 or int8. Merge pipelines are
  picked in 83% of text splits at 0.95 and 44% at 0.97.
- **Refuted: "the default leaves little compression on the table for text".** An in-sample oracle
  (lower bound ≥ 0.97) beat the default's median compression by 1.4 to 4x in 9 of 10 text units
  at 0.97.
  - The oracle is optimistic, so this shows neither that the default is too conservative nor
    that it is not.

## 7. Where the package disagrees with the data

- **The colpali preset uses binary.** `src/optivision/presets/colpali.yaml` sets
  `compression.method: binary`. Binary missed the 0.95 lower bound in 17 of 27 DocVQA 128-d units
  and 19 of 58 128-d image units.
- **"Centred int4 is never worse beyond noise" (docs/UNIVERSAL.md).** It compares centred with
  plain int4 and holds overall: safe in 100 of 103 units against 91 of 103.
  - Counterexamples: EVIE TabFQuAD, where centring hurt, and LateOn SciFact and answerai NFCorpus,
    which stay below the bar either way.
- **The early stop goes unmentioned.** The docs call the default "conservative" because of the
  0.999 bound, but in this data the early stop is the difference that shows up.

## Not supported

- NFCorpus being intrinsically harder for int4.
- Baseline level or relevant documents per query as the cause of NFCorpus's larger drops.
- Document length or dimension as the cause of answerai's collapse.
- Uncentred or centred peak-to-RMS as collapse predictors (AUC 0.60 and 0.77): EVIE and Vultron
  have high values but are fine.
- Geometry predicting binary retention, or int4(mean)'s residual misses.
- Compression ratio itself driving the loss at larger corpus size.
- A baseline threshold (such as ndcg@5 > 0.85) for an easy calibration set.

## Suggested next steps (none of them done)

1. **Measure the renormalised int4 scale** on LateOn, mLateOn, mxbai-edge-17m/32m and a few
   controls, both alone and combined with int4(mean), before changing `Int4Quantizer`.
2. **Document the ladder in section 2** as a post hoc starting point, not a guarantee.
3. **Annotate the colpali preset** (binary needs per-corpus checking), or move it to int4(mean)
   or merge(0.25) > int4.
4. **Add an "at risk" warning** from sm4 or cosine to the mean, worded as risk, not as expected
   retention.
5. **Advise a margin when calibrating on one dataset** (a lower bound ≥ 0.97, or re-certify on
   target data), and a warning for strict targets on small query sets.
6. **Re-measure aggressive projection on a same-domain, larger corpus.**

## Reproducing

- **Layout.** Each angle's scripts are under `scripts/<angle>/`, and the skeptics' re-runs under
  `scripts/verify-<angle>/`. They are one-off analysis scripts with absolute paths for this
  machine. Run them with `python -I`, from the directory of the script.
- **`transfer/df.pkl`.** It is rebuilt by `transfer/a2.py` and was not committed.
- **The answerai re-encode** (`int4_answerai/encode.py`, `qenc.py`) writes about 800 MB of
  vectors to `int4_answerai/vec/`. They were not committed, so re-run the encode first.
