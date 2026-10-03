# Q7 plan: comparison with existing multi-vector compression methods (stated before reading any external results)

Branch `research-investigation`, after `198f83a`. This is a literature and
comparability investigation:
- no experiments, no implementations of other methods;
- no new dependencies, no library change, no Q8.

## 1. The object OptiVision measures (what a comparison must match)

**Input.** Fixed multi-vector page embeddings from an existing late-interaction
encoder: ColPali v1.3 (`vidore/colpali-v1.3-merged`; first written as "v1.2", a
label error carried over from the Q4 plan), ColQwen2 v1.0, ColQwen2.5 v0.2 (128-d) and ColEmbed 4B
(2,560-d).

**Transformation.** Post hoc and training-free: no encoder retraining, and no labels
used for the transformation.
- Vector-count reduction (Ward merging within a page).
- Dimension reduction (PCA fitted on documents).
- Scalar or binary quantization.
- Combinations of these, and a two-tier index (a binary shortlist rescored
  exactly).

**Scoring.** Exact MaxSim over the full corpus, or a 50-candidate shortlist rescored
exactly.

**Measurement.**
- Benchmark: ViDoRe V1 DocVQA and InfoVQA test splits (500 pages; 451 / 494
  queries).
- Quality: nDCG@5, with retention against the same encoder's uncompressed float32
  index.
- Storage: actual bytes per page from arrays.
- Latency: on a laptop CPU, existing numpy implementation (MEASURED in Q4 and Q5/Q6),
  plus R12's GPU timings for ColEmbed.

## 2. Comparability tiers (fixed now; methods are assigned from their setup, never from their numbers)

A method's tier is decided from **what it reports and under what setup**, read from
the paper's or repository's method and experiment sections. Its result values are
not considered. A method can appear in several tiers if it reports several setups
(for example, text ColBERT results in Tier C and ViDoRe results in Tier A or B). Each
setup is classified separately.

**Tier A: directly comparable.** All of the following hold:
1. **Same encoder:** the same checkpoint family OptiVision measured, ColPali (any
   v1.x) or ColQwen2 / ColQwen2.5. A different version of the same family is allowed
   but flagged.
2. **Same benchmark and metric:** ViDoRe V1 DocVQA and/or InfoVQA, reporting nDCG@5,
   or retention computed from nDCG@5 against the same encoder's uncompressed
   baseline.
3. **Same kind of transformation:** post hoc compression of the stored page
   embeddings that keeps late-interaction (MaxSim-type) scoring.
4. **Storage stated precisely enough** to derive bytes per page, or a ratio against
   an explicit float baseline.

**Tier B: partially comparable.** At least two of the four Tier A conditions hold,
or conditions 1–3 hold but 4 does not. At least one major dimension differs, for
example:
- the model;
- the benchmark (other ViDoRe tasks or another document set);
- the metric;
- training-based rather than post hoc;
- approximate candidate generation;
- unstated storage accounting.

**Tier C: contextual only.** Fewer than two conditions hold. Examples:
- text ColBERT on MS MARCO or BEIR;
- retrieval-engine papers measuring latency on other corpora;
- methods that retrain the encoder to produce fewer or smaller vectors.

These are relevant background, but their numbers are not compared numerically with
OptiVision.

## 3. Comparison rules (fixed now)

- **Numbers are put side by side only for Tier A**, and only quantity by quantity
  (quality retention, bytes per page, vectors per page) when the baseline and
  accounting match. Mismatches are listed per method.
- **Tier B gets a qualitative, caveated comparison.** Its numbers are quoted as
  EXTERNAL with their own baselines, and nothing is normalized without the needed
  denominators.
- **Latency is never compared across hardware or protocols.** At most, within-paper
  ratios are quoted next to OptiVision's within-machine ratios.
- **No leaderboard, no winner, no overall ranking.**

**Labels:**
- **MEASURED:** OptiVision's own results, with artifact and commit;
- **EXTERNAL:** published or repository numbers, with citation, kept with their
  original wording and baseline;
- **PROJECTED:** estimates. None are planned; if a source's own number is an
  estimate, it is flagged;
- **INFERENCE:** our reading.

**Normalization** (bytes per page, ratio, vectors per page, dimensions, bits per
value, retention) happens only when the source states the needed baseline. Original
quantities are always kept next to normalized ones. A bare "Nx compression" with an
unknown denominator is reported as given and is not converted.

**Source priority:** peer-reviewed paper or arXiv preprint, then the official
repository, then official documentation. Blog posts are used only when no primary
source exists, and are labelled. Disagreements between sources are recorded with
the reason, where it can be found.

## 4. Search scope

**Included:**
- multi-vector and late-interaction retrieval compression: ColBERT-family residual
  or product quantization, token pruning, token pooling or merging, dimension
  reduction, binary and low-bit codes, compressed late-interaction engines;
- visual-document multi-vector compression (ColPali-family);
- fixed-dimensional or constant-space multi-vector methods, as context.

**Excluded:** generic ANN and vector-database literature, unless it directly targets
multi-vector or late-interaction compression.

## 5. Deliverables

`REPORT.md` containing:
- the inventory;
- the comparison table;
- the tier table;
- directly comparable, related and missing evidence;
- answers to Q7.1–Q7.8;
- proposed follow-up experiments, specified but not run;
- updates to the registry and the investigation status.
