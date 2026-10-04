# Q9 report: how far OptiVision's compression and quality findings generalize

Branch `research-investigation`, after Q8 (`54ce415`). **Q9 is a re-analysis of frozen
results.**
- No new experiment was run (§3 explains why none was needed).
- No library change. Q7 is not reopened, E7b is not run and Q10 is not started.

**Labels:**
- **MEASURED:** directly measured, in the cited report;
- **EXTERNAL:** third-party evidence (Q7);
- **PROJECTED:** arithmetic from measured quantities (Q8);
- **INFERENCE:** our reading.

## 1. Research question

How well do OptiVision's compression and quality relationships generalize across
models, datasets, compression settings and operating points, and which claims does the
existing evidence actually support?

## 2. Scope (the evidence base, and its edges)

**Models and data with measured results:**

| model | width | data | per-query results | CPU latency | source |
|---|---|---|---|---|---|
| ColPali v1.3 | 128 | ViDoRe V1 DocVQA, InfoVQA (500 pages; 451 / 494 queries) | yes | yes | R1–R7b, Q3, Q4, E1, E7a |
| ColQwen2 v1.0 | 128 | same | yes | yes | R11, Q3, Q4, Q5/Q6, E1, E7a, E7c, Q8 |
| ColQwen2.5 v0.2 | 128 | same | yes | yes | R11, Q3, Q4, E1.2, E7a |
| ColEmbed 4B (nemotron v2) | 2,560 | same | **no** (aggregates with intervals) | **no** (GPU only) | R12, reused in Q4 |
| answerai-colbert-small-v1 (text) | 96 | BEIR SciFact (5,183 docs, 300 queries; nDCG@10) | yes | no | R8, R7b, E1.2 |

**Not covered by any measurement:**
- 4,096-d or other widths;
- other visual retrievers (ColQwen3, ColNomic, ColSmol beyond early tests);
- ViDoRe V2/V3 and other corpora;
- corpora above 500 pages for visual retrieval, or above 5,183 documents for text;
- GPU timing for 128-d models;
- native compressed scoring kernels.

## 3. Existing evidence, and whether a new experiment is needed

| Q9 part | answerable from | open gap | does the gap change the conclusion? |
|---|---|---|---|
| A. models | Q4 (3 × 128-d, every axis, per-query), R12 (ColEmbed rows), R8 (text), R11, E7a | 4,096-d and other families | no: it bounds the claim, and filling it needs new models (excluded) |
| B. datasets | Q3, R7b, R11, R12, Q4, E1.2 | other corpora; text dataset and text model confounded | no: it bounds the claim, and filling it needs new data (excluded) |
| C. operating points | R7b, R11, R12 (3 targets × 20 splits), E1.1/E1.2, Q4 and E7a (vector ratios) | 0.99 with valid post-selection certification at n ≤ 225 (impossible, E1) | no |
| D. axes | Q4, E7a, Q5/Q6, batched rescore | native kernels | no (Q5/Q6 closed) |
| E. guarantee | E1.1, E1.2 | larger query pools; 2,560-d per-query data | no: the claim is stated narrowly |
| F. scale | Q8, R10 | quality at large N | no: it is reported as unestablished |

Under the brief's criteria, no specific Q9 claim needs a new experiment to be stated
correctly. The gaps limit the scope of claims rather than reversing any of them.
**No experiment was run.**

## 4. Model generalization (Q9A)

### 4.1 Consistent across the three 128-d ColPali-family models (MEASURED, Q4 unless marked)

| behaviour | evidence | status |
|---|---|---|
| Fewer vectors cut CPU scan time roughly in proportion | Q4: 0.52–0.53 / 0.27–0.28 / 0.16 of float32 at 1/2, 1/4, 1/8 on all three; Q8: linear 500–4,000 pages (ColQwen2) | consistent |
| int8 per-vector within about 0.3 points of float32 | Q4: 99.85–100.09%, six datasets | consistent |
| Centred int4 within about 1 point | Q4: 99.0–100.1% | consistent |
| Ward merging to 1/2–1/4 costs ≤ 1.3 points, and merging beats random pruning | Q4 (0.987–1.006 over 1/2, 1/3, 1/4); E7a: merging > random in 51/54 comparisons | consistent |
| Ward to 1/3 + int8 keeps ≥ 99.2% at about 12x | R11 table: 99.2–99.9% on ColPali, ColQwen2 and ColQwen2.5 | consistent |
| Dimension reduction collapses beyond d/2 (PCA 32: −9 to −20 points) | Q4, all three | consistent |
| PCA + coarse codecs interact negatively; merging + codecs is near-multiplicative | Q4 §13 | consistent in direction; size varies |
| Clustering choice and norm rescaling each add retention | E7a: Ward over K-means +0.7 to +3.7; rescaling +0.1 to +4.6 | consistent |
| Codecs give no scan speedup (decode to float32) | Q4: 101–107% of float32 on all three | consistent, **implementation-specific** |

### 4.2 Carries over to the 2,560-d ColEmbed model (MEASURED, R12; GPU timing)

| behaviour | ColEmbed evidence | status |
|---|---|---|
| Vector count drives scan time | R12 GPU: 0.44–0.47 / 0.25–0.26 / 0.11–0.12 at 1/2, 1/4, 1/10 (Q4 §11) | carries over (within-device ratios) |
| int8 and centred int4 near-lossless | 99.7–100.3% | carries over |
| Binary at 32x | 98.9% / 99.8% | carries over (better than ColPali) |
| Ward merging to 1/4 | 98.8% / 99.5% | carries over |
| Negative PCA + coarse-codec interactions | point estimates in the same direction (Q4 §13) | carries over in direction, without intervals |
| Factors do not multiply | 0.53–1.00 (index bytes, including basis) | carries over |

### 4.3 Clearly does not generalize (MEASURED)

| behaviour | 128-d | 2,560-d / other |
|---|---|---|
| **Dimension reduction** | PCA to d/4 loses 9–20 points | PCA to d/8 loses ≤ 0.9 points (R12) |
| **Best storage pipeline in the mid range** | A+C (merge + codec) owns 8–125x | B+C (PCA + codec) owns 8–63x / 105x |
| **Plain (uncentred) int4** | ColQwen2 / 2.5 99.9–100.4%; ColPali 97.5% / 99.9% | ColEmbed collapses (0.4–0.9%, R12); the text model loses 10 points (89.7%, R8) |
| **2-bit and binary strength** | weakest on ColPali DocVQA (95.2%, 96.3%); strong on ColQwen2.5 | text model: binary 94.1%, 2-bit 85.3% (R8) |
| **Centred binary** | ColPali about ±1 point; ColQwen 96.5–99.3% (R3b, R11) | text model collapses to 1.1% (R3b: queries share one dominant direction) |
| **Absolute merge thresholds** | the same cosine radius merges about half as much on ColQwen as on ColPali (R11) | the text model's narrow cone makes absolute radii collapse documents (R8) |

### 4.4 Model-family-specific conclusions (INFERENCE)

- **Which codec is "safe" beyond int8 and centred int4 depends on the model.** Plain int4,
  2-bit and binary vary by up to 100 points (int4 on ColEmbed), and by 2–10 points
  among the other models.
- **Dimension reduction is useful only for the wide model measured.** Nothing supports
  "useful at ≥ X dimensions" for widths not measured.
- **Relative budgets (merge to a fraction of each page's vectors) transfer across
  models; absolute thresholds do not.**
- **Unmeasured models:** no conclusion extends to 4,096-d, ColQwen3, ColNomic or any
  other family.

## 5. Dataset generalization (Q9B)

| observation | DocVQA | InfoVQA | SciFact (text) | status |
|---|---|---|---|---|
| Fixed-pipeline retention (same configuration) | close to InfoVQA (Q3: 0–2 points; 0, 3 and 12 of 40 candidates beyond intervals) | – | different model, so not comparable | consistent across DocVQA and InfoVQA (MEASURED) |
| Compression the optimizer chose (out of sample, 0.95) | 11.8x (ColPali), 20.7–22.8x (ColQwen), 32x (ColEmbed) | 72.7x, 114–122x, 355x | 12.4x | **dataset-dependent** (MEASURED, R7b/R11/R12) |
| Certification difficulty | SE 2.1–2.6x larger than InfoVQA (Q3) | – | – | the main driver (Q3, preserved) |
| Merging beats random or token dropping | yes (E7a; R2) | yes | yes (R8: 99.8% against 91.2%) | consistent (MEASURED) |
| int8 near-lossless | yes | yes | yes (R8: 100.1%) | consistent |
| Vector compression at 99.8% retention | about 9x (ColPali, R8 comparison) | – | about 3x (text) | model and dataset confounded |

**Q3's conclusion, preserved unchanged:** DocVQA gets less compression mainly because its
quality is harder to certify, not because it is intrinsically less compressible.
Relevance-margin composition explains most of that difficulty, with residual
uncertainty (Q3 §19).

**Survives a dataset change (MEASURED on DocVQA and InfoVQA):**
- the relative behaviour of every axis;
- int8 and centred int4 safety;
- merging over random pruning;
- the latency behaviour (pages are the same size).

**Stays dataset-specific:**
- the *amount* of compression a certified selection can choose (driven by
  certification difficulty);
- lossier pipelines' sensitivity to corpus growth (R10: DocVQA and InfoVQA lose
  different amounts);
- **SciFact cannot separate dataset from model** (one text model on one corpus), so no
  claim about "text corpora" is supported.

## 6. Operating-point generalization (Q9C)

| operating point | evidence | behaviour |
|---|---|---|
| **Retention targets 0.99 / 0.97 / 0.95** (chosen compression) | R7b, R11, R12: median compression **never decreases** as the target loosens, in all 9 dataset × encoder cells (ColPali and SciFact R7b; ColQwen2 and 2.5 R11; ColEmbed R12, e.g. InfoVQA 8.0 → 277.7 → 354.8x) | monotonic (MEASURED) |
| Targets: met rate of the shipped rule (A) | 355/360 (R7b + R11), 119/120 (R12); misses at all three targets, worst 97.4% against 0.99 | approximately consistent, **empirical, no guarantee** |
| Targets: valid certification (method C) | deploys at 0.95 everywhere, at 0.97 only on InfoVQA, **never at 0.99 with ≤ 225 queries** (a lower bound for any distribution-free test, E1) | strongly target-dependent: 0.95 results do **not** carry to 0.99 |
| Vector-retention ratio (Ward 1/2 → 1/10, fixed codec) | Q4: monotonic within noise (for example ColQwen2 DocVQA 1.003, 0.997, 1.006, 0.986, 0.977); E7a 1/5 → 1/10 → 1/20: strictly decreasing for every method on all six datasets | approximately monotonic; small non-monotonic wiggles within intervals at ≤ 1/4 |
| PCA width | Q4: monotonic decrease; steep at 128-d, flat to d/8 at 2,560-d | monotonic, **model-dependent slope** |
| Codec bits | Q4: float16 ≈ int8 ≈ int4c > 2-bit ≈ binary on 128-d; plain int4 non-monotonic on ColEmbed (collapses while 2-bit and binary hold) | **configuration-dependent**; not monotonic in bits |
| Combinations | Q4 §13: interactions concentrated in PCA + coarse codecs | configuration-dependent |

**Smooth scaling is not demonstrated.** The measured grids are coarse: 5 merge ratios,
4 widths, 6 codecs and 3 targets. Monotonic in the measured points is all that can be
said. A 0.95 pattern does not transfer to 0.99, because certification at 0.99 is
infeasible at these query counts.

## 7. Compression-axis generalization (Q9D; classifying Q4's findings)

| finding | class | basis |
|---|---|---|
| Vector merging (Ward, relative budgets) preserves quality to about 1/4; merging > pruning | **strongly replicated** | 4 visual models + text (Q4, R8, R11, R12, E7a) |
| Vector count drives scan time | **strongly replicated** | 3 models on CPU, ColEmbed on GPU, linear 500–4,000 pages (Q8) |
| Dimension reduction | **model-specific** | harmful at 128-d, nearly free to d/8 at 2,560-d |
| int8 per-vector near-lossless | **strongly replicated** | every model and dataset measured, including text |
| Centred int4 near-lossless | **strongly replicated** (≤ 1.4 points) | all visual models within about 1 point (Q4, R12); text model 98.6%, and 98.7% after Ward at 23x (R3b), where plain int4 lost 10 points |
| Plain int4 | **model-specific** | fine on ColQwen, −2.5 on ColPali DocVQA, −10 on text, collapse on ColEmbed |
| Binary | **replicated with scope limitations** | 94.1–99.8% across models; strength varies by model and dataset (ColPali DocVQA weakest among visual models) |
| Combinations and interactions | **replicated with scope limitations** | negative PCA + coarse-codec interactions in all models (ColEmbed points only); share above the threshold 14–34% (H4d partial) |
| **Compression factors do not simply multiply** | **strongly replicated** (preserved) | 0.69–1.00 (128-d), 0.53–1.00 (ColEmbed): per-vector scales and the shared basis |
| Two-tier search (binary shortlist + int8 rescoring) | **replicated with scope limitations** | quality about 100% on all 6 128-d datasets and ColEmbed; RAM benefit universal; latency depends on the implementation (Q5/Q6 batched rescoring) and on overlap and N (Q8) |
| Compressed or binary codes make search faster | **unsupported** in the current implementation (preserved) | quantized and binary paths decode to float32, so a smaller index does **not** mean faster search (Q4, Q5/Q6); native kernels were slower on this stack |

## 8. Optimizer and guarantee generalization (Q9E)

**What the post-selection guarantee supports** (method C, E1.1; MEASURED by simulation
on measured per-query data):
- P(deploy ∧ R_Q < T) ≤ δ for the *selected* configuration;
- under i.i.d. queries from one query pool, a fixed corpus and the fitted
  pipelines;
- on the datasets checked: ColPali, ColQwen2 (development) and ColQwen2.5 and SciFact
  (held-back E1.2, frozen before loading).
- It held in 27 of 27 held-back cells, with a maximum miss of 0.05% at δ = 0.05.

**What it does not support:**
- a different corpus, or a growing one (R10: retention moves with corpus size);
- queries not drawn like the calibration pool;
- **the 2,560-d model** (no per-query data, so C was never run on it);
- certification at T = 0.99 with ≤ 225 queries (impossible for any valid method);
- a universal or model-free guarantee;
- the shipped rule A. A has **no** post-selection guarantee; its met rates (355/360,
  119/120) are empirical summaries over splits of one pool per corpus.

**Does the held-back evidence strengthen generalization?** Narrowly, yes (INFERENCE):
- The validity of C did not depend on which of the tested encoders or corpora was
  used. It held on a held-back visual model and on a different modality (a text
  model with nDCG@10).
- That supports "**the method's validity held on every encoder and corpus it was
  tested on (four encoders, three corpora), under its stated assumptions**".
- It does **not** support validity for untested corpora or deployment conditions,
  where the i.i.d. and fixed-corpus assumptions may fail.

## 9. Interaction with Q8 (Q9F)

| quantity | how it scales with N | quality implication |
|---|---|---|
| storage, RAM | linear (exact; PROJECTED at 1M) | none |
| scan latency | linear in stored vectors, MEASURED to 4,000 duplicated pages; PROJECTED beyond | none |
| two-tier | hot stage dominates beyond about 1,300 pages (PROJECTED) | none |
| **retrieval quality** | **not measured above 500 pages** (visual) or 5,183 docs (text) | R10: lossy configurations lose 0–3 points from 100 to 500 pages (ColPali), 0–2 (ColQwen) |

- **Predictable resource scaling says nothing about quality scaling.** Every quality
  generalization claim in §4–8 is bound to 500-page corpora (text: 5,183 documents).
- INFERENCE from R10: retention of lossy pipelines more likely declines than holds as
  distractors grow. Q8's million-page figures remain **PROJECTED** storage arithmetic.

## 10. Limitations

- **Coverage:** four visual encoders across two widths (three of them 128-d), and one
  text encoder.
  - Visual data: two ViDoRe V1 subsets of 500 pages each.
  - Text data: one corpus.
  - Generalization is assessed within this set.
- **ColEmbed:** aggregates only (no per-query data, so no paired intervals for
  interactions), GPU-only timing.
- **One query pool per corpus.** Split-based met rates are summaries, not rates with
  population-level intervals.
- **Text confound:** SciFact confounds model and dataset (text model and text corpus
  change together).
- **Coarse operating-point grids,** so smoothness is not established.
- **The ColQwen2-vs-SAP external discrepancy is unresolved** (Q7/E7c). It sits upstream
  of evaluation and does not affect internal comparisons.

## 11. What is genuinely generalized (within the tested set)

- **Vector count is the latency lever** (all models; linear in N up to 4,000 pages).
- **Merging with relative per-page budgets beats pruning and preserves quality to
  about 1/4 of the vectors** (all models and both visual datasets; text up to 1/3).
- **int8 per-vector is near-lossless everywhere measured.** Centred int4 is too on all
  visual models.
- **Compression factors do not multiply, and the axes interact.** PCA with coarse
  codecs is the recurring antagonistic pairing.
- **Two-tier search preserves quality and cuts RAM.**
- **Certification difficulty, not compressibility, drives the dataset differences in
  chosen compression** (Q3, DocVQA against InfoVQA).
- **Method C's post-selection validity held on every encoder and corpus tested,** under
  i.i.d. and fixed-corpus assumptions.

## 12. What remains model- or dataset-specific

- **Model-specific:** dimension reduction; plain int4; centred binary (a hazard on the
  text model); the strength of 2-bit and binary; absolute merge thresholds; the best storage pipeline (A+C at 128-d against
  B+C at 2,560-d).
- **Dataset-specific:** how much compression can be certified (DocVQA ≪ InfoVQA);
  sensitivity of lossy pipelines to corpus growth.
- **Implementation-specific:** codec speed (decode to float32); the batched-rescoring
  gain (overlap-dependent).
- **Not established at all:**
  - any 4,096-d or untested model;
  - other corpora or ViDoRe versions;
  - quality at large N;
  - certification at 0.99 with small query sets;
  - guarantees for the 2,560-d model.

## 13. Final Q9 decision

**Supported with scope limitations.**
- The core findings replicate across every encoder and corpus measured: the vector
  count/latency relationship, relative merging, int8 safety, non-multiplicative
  factors and axis interactions, two-tier quality, and the conditional validity of
  method C.
- **Model-specific:** dimension reduction and the low-bit codecs.
- **Dataset-specific:** the amount of certifiable compression.
- **Unsupported:**
  - codec-driven speedups (current implementation);
  - any extension beyond the tested set, i.e. untested models or widths, other
    corpora, or larger corpora for quality;
  - a guarantee broader than method C's stated assumptions.

**Q9 is closed. No new experiment was required.** Q10 has not been started.
