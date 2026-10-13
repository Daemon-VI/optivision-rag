# OptiVision research investigation: final summary for the paper

The text below is written to drop into a paper (abstract, contributions, results,
limitations). Every number is traceable to a closed report on branch
`research-investigation`; the source map is in §8.

**Labels:**
- **MEASURED:** measured in this repository.
- **PROJECTED:** arithmetic on measured values.
- **EXTERNAL:** third-party.

Nothing here claims novelty: Q10 found the empirical findings established but their
priority against the full literature not established. The claim boundaries in §7 are
binding.

---

## 1. Abstract (draft)

Multi-vector late-interaction retrievers store hundreds of vectors per page, and that
makes their indexes large. We study post-hoc, training-free compression of fixed page
embeddings along three axes:
- vector count (Ward merging with per-page budgets);
- dimension (PCA);
- quantization (int8, centred int4, binary).

**Setup.** Four visual encoders (ColPali v1.3, ColQwen2 v1.0, ColQwen2.5 v0.2, and the
2,560-d ColEmbed 4B) on ViDoRe V1 DocVQA and InfoVQA, plus a text ColBERT on SciFact.
All studies were pre-registered.

**Axis findings:**
- **Vector count is the only axis that reduces exhaustive search cost.** Search cost is
  linear in stored vectors.
- **Merging to a quarter of the vectors keeps nDCG@5 within 1.3 points of float32 on
  every 128-d dataset measured,** and merging beats random pruning.
- **int8 and centred int4 are near-lossless on every encoder measured.**
- **Dimension reduction is model-dependent:** harmful at 128-d, nearly free down to
  one-eighth of the width at 2,560-d.
- **The axes interact:** compression factors do not multiply, and PCA followed by coarse
  codecs is consistently antagonistic.

**Why datasets differ.** DocVQA admits far less certified compression than InfoVQA
mainly because its quality is harder to certify, not because it is less compressible.

**Certification.** For selecting a configuration with a statistical statement, we apply
fixed-sequence testing with exact betting p-values. The result:
- the probability of deploying a configuration below target is bounded by δ, under
  i.i.d. queries and a fixed corpus;
- this held in 27 of 27 frozen held-back cells;
- no distribution-free method can certify a 0.99 retention target with 225 or fewer
  calibration queries.

**Limits.** Native compressed-code scoring was slower than decode-then-BLAS on our
software stack. Retrieval quality beyond 500 pages is unmeasured.

## 2. Contributions (as defensibly stated)

1. **An empirical map of three compression axes and their interactions** for multi-vector
   visual retrievers, with pre-registered hypotheses, paired confidence intervals and an
   explicit statement of where each finding generalizes (4 visual + 1 text encoder,
   3 corpora).
2. **An explanation of dataset-dependent compression:** certification difficulty, driven
   by the relevance-margin composition of the query pool, rather than intrinsic
   compressibility.
3. **A validated application of known post-selection testing** (learn-then-test-style
   fixed-sequence testing with betting p-values; Waudby-Smith & Ramdas) to compression
   selection, together with its finite-sample limits. This is not a new statistical
   method.
4. **An open, model-agnostic compression and selection framework** (released as
   OptiVision v0.3.0): composable stages, exact byte accounting, two-tier indexing, and
   an exact batched rescoring path. This is an engineering contribution.

## 3. Main results

### 3.1 Compression axes (MEASURED; Q4, R12)

**Retention** is nDCG@5 against the same encoder's float32 index. **"128-d"** means
ColPali, ColQwen2 and ColQwen2.5 on DocVQA and InfoVQA. **"2,560-d"** means ColEmbed 4B.

| axis or configuration | 128-d retention | 2,560-d retention | scan time, fraction of float32 (CPU, 128-d) |
|---|---|---|---|
| merge to 1/4 of vectors (about 3.9x) | 0.989–1.006 | 0.988 / 0.995 | 0.27–0.28 |
| merge to 1/10 | 0.954–0.992 | 0.984 / 0.996 | 0.13 |
| PCA to d/2 | 0.972–0.991 | 1.002 / 0.999 | 0.79–0.80 |
| PCA to d/4 | 0.803–0.907 | 0.993 / 0.996 | 0.74 |
| int8 per vector (3.9x) | 0.998–1.001 | 0.997 / 1.002 | 1.01–1.02 |
| centred int4 (7.8x) | 0.990–1.001 | 0.997 / 1.002 | 1.03–1.04 |
| plain int4 (7.8x) | 0.975–1.004 | **0.009 / 0.004** | 1.04–1.05 |
| binary (32x) | 0.963–0.995 | 0.989 / 0.998 | 1.01–1.02 |
| merge 1/4 + binary (about 122–125x) | 0.954–0.982 | 0.974 (DocVQA) | 0.28 |
| two-tier: merged binary in RAM + int8 rescoring of 50 candidates | 0.999–1.005 | 0.997 / 1.003 | RAM 122–125x smaller |

- **Codecs do not speed up search in this implementation,** because codes are decoded to
  float32 before scoring.
- **Combined factors are 0.69–1.00 of the product of the single-axis factors** (128-d).
- **Interactions:** PCA combined with binary or centred int4 is the recurring negative
  interaction, reaching −18 points (Q4 §13).

### 3.2 Attribution of merging quality (MEASURED; E7a)

At equal vector counts on six datasets:
- **Clustering method:** Ward clustering retains +0.7 to +3.7 points more than K-means.
- **Rescaling:** rescaling merged vectors to their members' mean norm adds +0.1 to +4.6
  points.
- **Against random pruning:** all merging variants beat random pruning in 51 of 54
  comparisons, by up to 18 points at 1/20 of the vectors.

### 3.3 Why DocVQA compresses less (MEASURED; Q3)

**The gap is certification, not compressibility.**
- For the same pipeline, pool retention on DocVQA is within 0–2 points of InfoVQA.
- The n = 225 certification standard error is 2.1–2.6x larger on DocVQA. It splits about
  evenly into a lower mean baseline and a larger per-query spread from near-boundary
  queries.

**Most of it traces to relevance margins.**
- Reweighting InfoVQA to DocVQA's relevance-margin distribution closes 61–81% of the
  standard-error gap.
- Restricting DocVQA to its positive-margin half closes 84–120%.

### 3.4 Selection and certification (E1, R7b, R11, R12)

- **The shipped rule (0.999 bootstrap bound) is empirically safe but has no
  post-selection guarantee.**
  - It met the target in 355 of 360 out-of-sample choices on the 128-d models and 119 of
    120 on the 2,560-d model.
  - Choosing by the point estimate met the target in only 10–100% of splits.
- **Method C (MEASURED by resampling):**
  - P(deploy ∧ R < T) ≤ δ held in every check, including 27 of 27 frozen held-back cells
    (ColQwen2.5, SciFact), with a maximum miss of 0.05% at δ = 0.05.
  - With ≤ 225 queries it deploys at T = 0.95 everywhere and at 0.97 only on InfoVQA.
  - **No valid method certifies T = 0.99 at these sizes.**

### 3.5 Search cost and scale (MEASURED and PROJECTED; Q5/Q6, ENG-1, Q8)

- **Scan time is linear in stored vectors** from 500 to 4,000 pages: six configurations,
  R² ≥ 0.999.
- **Two-tier search:** the exhaustive first stage dominates beyond about 1,300 pages
  (PROJECTED), so two-tier reduces RAM but not the scaling class.
- **Native compressed-code scoring was slower on our stack** (int8 24–34x, lookup-table
  binary 28–30x), with rankings equal up to float32 near-ties.
- **Batched exact rescoring** gives median speedups of 6.7x (rescoring) and 3.25x (end to
  end), with bit-identical rankings. Its gain falls as candidate overlap falls.
- **Index size at 1M pages** (PROJECTED, ColQwen2): 383 GB float32, 12 GB binary, 3.1 GB
  merged binary.

### 3.6 Generalization (Q9)

**Verdict: supported with scope limitations.**

| status | findings |
|---|---|
| Replicated across all encoders and corpora measured | vector count drives cost; relative merging beats pruning; int8 and centred int4 are near-lossless; factors do not multiply; two-tier keeps quality; certification drives dataset differences; method C's conditional validity |
| Model-specific | dimension reduction; plain int4; centred binary (collapses on the text model); 2-bit and binary strength |
| Dataset-specific | the amount of certifiable compression |

## 4. Comparison with prior work (EXTERNAL; Q7)

**Only one published setup is directly comparable:** Structural Anchor Pruning (SAP,
arXiv 2601.20107v3). It uses the same ColPali v1.3 and ColQwen2 v1.0 checkpoints and the
same ViDoRe V1 subsets.

**At about 1/10 of the vectors, our measured Ward merging retains 4–10 points more than
SAP's reported K-means baseline.** We investigated this in two controlled follow-ups:
- **E7a:** our clustering and rescaling choices explain part of the gap.
- **E7c:** aligning with the official ViDoRe evaluation (legacy and MTEB) changes results
  by at most 0.65 points.

**What remains:**
- On ColPali, random-pruning controls agree with SAP within 1.6 points.
- On ColQwen2 they disagree by up to 14 points before any compression method is
  applied. That points to an upstream difference, plausibly in encoding settings.
- **The discrepancy is unresolved** and is reported as such. It is not evidence of
  superiority or inferiority.

Ward token pooling itself is prior work (Clavié et al., 2024). Shortlist-and-rescore and
binary multi-vector codes are established practice.

## 5. Limitations and threats to validity

**Scope:**
- **Encoders and corpora:** four visual encoders across two widths (three of them
  128-d), one text encoder, and three corpora. The visual corpora are 500-page ViDoRe V1
  subsets.
- **Query pools:** one per corpus, so split-based rates are summaries rather than
  population estimates.
- **ColEmbed 4B:** aggregate results only, with GPU-only timing.
- **Not covered:** 4,096-d or other model families, ViDoRe V2/V3, retrieval quality
  beyond 500 pages (R10 shows lossy configurations losing 0–3 points from 100 to 500
  pages), and timing beyond 4,000 duplicated pages.

**Implementation and hardware:**
- Timing is from one laptop CPU with two power states.
- Codec speed and native-kernel results are specific to this implementation and
  software stack.
- The library cannot yet persist centred int4 or 2-bit codecs, or the PCA basis.

**Statistics:**
- Method C assumes i.i.d. queries and a fixed corpus.
- The shipped selector's success rates are empirical.

**Literature:** the survey covered multi-vector compression and retrieval engines, not
the statistical-selection literature, so novelty and priority are not established.

## 6. Unresolved questions (future work)

- the ColQwen2-vs-SAP discrepancy (encoding settings);
- the residual certification gap on DocVQA;
- the cause of plain int4's collapse at 2,560-d;
- batched rescoring at very low candidate overlap;
- retrieval quality at large corpus sizes;
- whether compiled native kernels outperform decode-then-BLAS;
- sublinear candidate generation (ANN or MUVERA-style), needed to change the scaling
  class.

## 7. Claim boundaries (do not write)

**Novelty and comparison:**
- "first", "novel algorithm", "state of the art";
- "better (or worse) than SAP".

**Generality:**
- "universal", "any multi-vector model", anything about 4,096-d or untested models.

**Guarantees:**
- "guaranteed retention" without method C's conditions;
- any guarantee for the shipped `optimize()` rule.

**Scale:**
- "scales to / handles millions of pages", "production-ready at scale". The 1M-page
  figures are projected storage arithmetic.

**Speed:**
- "quantization or binary codes make search faster" (not in this implementation);
- "native compressed scoring is inherently slower" (true only on this stack).

**Numbers:**
- any compression factor without its model, dataset, retention and interval.

## 8. Source map (all on `research-investigation`)

| section | reports |
|---|---|
| Axes, interactions, latency | `reports/research/Q4/REPORT.md`; R12 in `docs/UNIVERSAL.md` |
| Merging attribution | `reports/research/Q7/E7a/REPORT.md` |
| Dataset differences | `reports/research/Q3/REPORT.md` (§12, §18–19) |
| Selection and certification | `reports/research/E1/RESULTS.md`, `E1/DESIGN.md`; R7b, R11, R12 |
| Native scoring | `reports/research/Q5Q6/REPORT.md`, `CORRECTNESS.md` |
| Batched rescoring | `reports/engineering/batched_rescore/RESULTS.md` |
| External comparison | `reports/research/Q7/REPORT.md`, `Q7/E7a/REPORT.md`, `Q7/E7c/REPORT.md` |
| Scale | `reports/research/Q8/REPORT.md` |
| Generalization | `reports/research/Q9/REPORT.md` |
| Contribution classification | `reports/research/Q10/REPORT.md` |
