# Q10 report: research and engineering — what this investigation established

Branch `research-investigation`, after Q9 (`4fa5c5b`). Q10 is a **synthesis**:
- no experiment was run (§3);
- no result or historical report was changed;
- Q1–Q9 are not reopened, and E7b was not run.

**Labels:**
- **MEASURED:** a direct measurement, in the cited report;
- **EXTERNAL:** third-party (Q7);
- **PROJECTED:** arithmetic on measured values (Q8);
- **INFERENCE:** our reading.

## 1. Research question

Which OptiVision findings are research contributions, which are engineering
contributions, and which are implementation-specific observations? And what can the
project defensibly claim as its overall contribution?

## 2. Scope

**Covered:** the full investigation record:

| stage | content |
|---|---|
| Phase 0 | `docs/RESEARCH_INVESTIGATION.md` |
| E1.0–E1.2 | Q1 and Q2 |
| Q3 | dataset differences |
| Q4 | axis ablation |
| Q5/Q6 | native scoring |
| ENG-1 | batched rescoring |
| Q7, E7a, E7c | external comparison and attribution |
| Q8 | scaling |
| Q9 | generalization |

Plus the release-era R-series evidence that these build on (`docs/UNIVERSAL.md`: R1–R12),
and the source code where needed to separate algorithm from implementation.

**Not covered:** any new measurement, and any literature beyond Q7's survey, which
looked at multi-vector compression and retrieval engines, **not** at statistical
selection or certification methods. That limits every novelty judgement below.

## 3. Evidence base and the decision not to experiment

Each finding below cites a closed report. No classification depends on a missing
measurement; the open questions (§12) concern scope, not category. Q10 therefore runs
**no** experiment.

## 4. Classification framework

| class | meaning |
|---|---|
| **A. Research finding** | an empirical or methodological insight beyond implementing known functionality |
| **B. Methodological contribution** | a procedure or analysis design whose value lies in how evidence is produced |
| **C. Engineering contribution** | software that makes known techniques usable, composable, measurable or faster |
| **D. Implementation-specific observation** | a result that depends on the current code, kernels, formats or hardware |
| **E. Scope-limited empirical claim** | real, but bounded by the models, data, corpus size, query pools, operating points or hardware tested |

**Novelty levels:**
1. **Established by this project:** the finding is measured.
2. **Potentially novel / research-worthy:** no prior example was found in Q7's (limited)
   survey.
3. **Novelty not established:** prior art exists, or was not searched.

## 5. Evidence matrix

| # | finding | evidence | class | scope | strength | novelty |
|---|---|---|---|---|---|---|
| 1 | **The shipped optimizer is conservative but empirically safe.** It met targets in 355/360 (128-d) and 119/120 (2,560-d) held-out choices. The point rule met 10–100% per cell. DocVQA conservatism comes from certification width, not quality loss. | R7b, R11, R12, E1.1, Q3 | A + E | 4 visual + 1 text encoder, 3 corpora, one pool each, 20 splits | strong (empirical) | 1 / 3: measured; "conservative bootstrap selection" is not new |
| 2 | **Method C (learn-then-test-style fixed-sequence testing with exact betting p-values) gives P(deploy ∧ R_Q < T) ≤ δ for the selected configuration,** under i.i.d. queries and a fixed corpus. It held in every check, including 27/27 frozen held-back cells. No valid method certifies T = 0.99 with ≤ 225 queries. | E1.1, E1.2, `E1/DESIGN.md` | B (validated application) + A (the 0.99 sample-size boundary) + E | 4 encoders, 3 corpora; not 2,560-d; not growing or shifted corpora | strong within assumptions | 3: its components are established methods (Waudby-Smith & Ramdas; learn-then-test); the application to compression retention was not searched for in the statistics literature |
| 3 | **DocVQA gets less compression mainly because its quality is harder to certify** (certification SE 2.1–2.6x), not because it compresses worse. Relevance-margin composition explains most of the gap; a residual remains. | Q3, Q3H | A + E | 2 corpora × 3 encoders | supported with residual uncertainty | 2 (none found in Q7's survey; not searched beyond it) |
| 4 | **Vector count drives scan cost about proportionally;** dimension and codecs barely move it in the current code. | Q4, Q8 (linear 500–4,000 pages) | A + E (+ D for the codec part) | 3 × 128-d CPU, ColEmbed GPU | strong | 3: the proportionality is expected from MaxSim's cost; the measurement is ours |
| 5 | **Merging with relative per-page budgets keeps quality to about 1/4 of the vectors and beats pruning.** Ward clustering and norm rescaling each add retention. | Q4, R2, R8, R11, R12, E7a | A + E | every encoder measured | strong | 3: Ward token pooling is Clavié et al. (2024); merge > prune is also reported by Light-ColPali and by Jha et al. 2026 (Q7). The E7a attribution of rescoring and clustering is ours. |
| 6 | **Dimension reduction is model-specific:** harmful at 128-d (d/4: −9 to −20 points), nearly free to d/8 at 2,560-d. | Q4, R4, R12 | A + E | 3 × 128-d, 1 × 2,560-d | strong within the tested set | 2/3: not found for visual retrievers in Q7; Matryoshka-style text results exist (Jina-ColBERT-v2) |
| 7 | **Per-vector int8 is near-lossless everywhere measured.** | Q4, R3, R8, R12 | A (confirmation) + E | all encoders | strong | 3: a standard expectation |
| 8 | **Centred int4 is near-lossless (≤ 1.4 points); plain int4 is model-specific** (collapses on ColEmbed, −10 points on the text model). | Q4, R3b, R8, R12 | A + E | as tested | strong | 2 for the specific collapse pattern; cause unresolved |
| 9 | **Binary: 94.1–99.8% depending on the model; centred binary collapses on the text model.** | Q4, R3b, R8, R11, R12 | A + E | as tested | strong | 3: binary multi-vector codes are common practice (Vespa, BLOG, Q7) |
| 10 | **Compression axes interact,** and factors do not multiply (0.53–1.00). PCA + coarse codecs is the recurring antagonistic pairing. | Q4 §13 | A + E | 3 × 128-d with intervals; ColEmbed points | supported (H4d partial) | 2: no visual-retrieval study of these interactions found in Q7 |
| 11 | **Two-tier (binary shortlist + exact int8 rescoring) keeps quality and cuts RAM about 30x;** it does not change the scaling class. | Q4, R6, Q5/Q6, Q8 | C (implementation) + A/E (measured behaviour) | 6 × 128-d datasets + ColEmbed | strong | 3: shortlist-and-rescore is established practice |
| 12 | **Native compressed-code scoring was slower** (direct int8 24–34x, LUT binary 28–30x) on this stack. Equivalence held under a rounding-aware criterion; the strict bit-exact criterion failed on float32 near-ties. | Q5/Q6 | D | numpy / torch CPU, one laptop | strong for this stack only | not a general claim |
| 13 | **Batched exact rescoring:** median rescoring speedup 6.7x, end to end 3.25x; bit-identical on 60/60 cells across 6 datasets. The gain shrinks as candidate overlap falls (7.4x → 2.8x). | ENG-1, Q8 V2 | C (+ D for the overlap dependence) | 6 datasets, one laptop, all-query batches | strong | not claimed as research |
| 14 | **SAP comparison:** the only Tier A source. Evaluation alignment changes nothing material; mechanism choices explain part of the gap. **The ColQwen2 disagreement is unresolved.** | Q7, E7a, E7c | E (unresolved) | 2 models × 2 subsets | unresolved | – (not evidence of superiority or inferiority) |
| 15 | **Corpus scaling:** storage and RAM are linear (exact); scan time is linear to 4,000 duplicated pages; 1M-page figures are PROJECTED; quality at scale is unmeasured. | Q8, R10 | A (scan linearity, rescoring vs overlap) + E | one model for timing; ≤ 4,000 duplicated pages | strong for resources; none for quality | 3: linear exhaustive cost is expected |
| 16 | **Generalization: supported with scope limitations.** | Q9 | E | 4 visual + 1 text encoder, 3 corpora | as stated | – |

## 6. Research findings (class A; all scope-limited, see §11)

The investigation establishes these empirical findings:
- **Certification, not compressibility, drives the dataset differences** (#3). This is
  the most explanatory finding: it turns "compression depends on the data" into a
  measured, partly attributed mechanism (relevance margin, certification SE).
- **The finite-sample boundary for certifying retention** (#2): no distribution-free
  test certifies T = 0.99 with ≤ 225 queries. Within the tested pools, the valid method
  deploys only at 0.95, and at 0.97 on InfoVQA.
- **Axis-level behaviour** (#4–#10):
  - count drives cost;
  - merging beats pruning, with clustering and rescaling attributed (E7a);
  - dimension reduction is model-specific;
  - codec behaviour is model-specific beyond int8 and centred int4;
  - factors don't multiply, and PCA + coarse codecs is antagonistic.
- **Empirical safety of the shipped optimizer** (#1) and the unreliability of
  point-estimate selection.

**Novelty.** The investigation establishes these findings empirically. It does not
establish priority or novelty relative to the complete literature. Q7's survey covered
multi-vector compression and retrieval engines; it found **no** prior study of the
certification-driven dataset differences (#3) or of axis interactions on visual
retrievers (#10). Those are the most plausible candidates for research-worthy novelty
(level 2), but only a fuller literature search could establish it.

## 7. Methodological contributions (class B)

- **Method C as a validated application** (#2).
  - It combines established statistical tools: exact betting p-values for a bounded
    mean (Waudby-Smith & Ramdas) and learn-then-test-style fixed-sequence testing.
  - It is used with the identity R ≥ T ⇔ E[c − T·b] ≥ 0, which makes retention a
    bounded-mean problem.
  - It was validated by population resampling and by frozen held-back confirmation.
  - It is properly described as **a validated methodology that provides a conditional
    finite-sample statistical guarantee** (i.i.d. queries, fixed corpus, fitted stages).
  - It is **not** a novel theoretical contribution: its proofs rest on known results.
- **The evaluation harness and pre-registration discipline** (frozen plans, held-back
  confirmation, per-query stores, paired bootstraps, exact-reproduction checks). It
  produced trustworthy, reproducible evidence and caught errors. It is a methodological
  strength of the project, not a contribution to the field.
- **Target-retention-driven selection and Pareto reporting** are useful framing, but not
  new methodology (Pareto frontiers and constrained selection are standard).

So: OptiVision **implements and validates** a useful methodology. It did **not**
introduce a novel one.

## 8. Engineering contributions (class C)

- **The model-agnostic compression framework:**
  - stage composition (merge, project, codec), relative per-page budgets and protected
    tokens;
  - the codecs (per-vector int8, centred int4, binary, 2-bit);
  - exact byte accounting.

  It was validated on 4 visual encoders and 1 text encoder, and released as v0.3.0
  (PyPI and npm). This is the project's largest contribution in volume. It is an
  engineering integration of known techniques, plus measured defaults.
- **calibrate() / optimize()** with the 0.999 bound and per-family early stopping: a
  practical, measured selector (its behaviour is research finding #1).
- **Storage and search abstractions:** `ExactIndex`, `TieredIndex` (two-tier), and
  blockwise decode with a bounded memory budget.
- **Batched exact rescoring** (ENG-1): an engineering optimization, **not** a research
  discovery. Median rescoring speedup about 6.7x and end-to-end about 3.25x,
  bit-identical on the six tested datasets. Its gain depends on candidate overlap (Q8).
- **Research tooling:** the scripts and stores that make every report reproducible.

## 9. Implementation-specific observations (class D; not properties of compression)

- **Quantized and binary codes are decoded to float32 before scoring,** so a smaller
  index does not mean faster search in the current code (Q4: 101–107% of float32 scan
  time).
- **Native compressed-code scoring is negative on this tested software stack, NOT
  inherently negative in general** (Q5/Q6). On this machine:
  - torch's `_weight_int8pack_mm` and numpy LUT gathers lost to decode + OpenBLAS;
  - no compiled kernels were available.
- **The two-tier latency penalty came from a per-query rescoring loop** (fixed in
  ENG-1). The batched gain depends on overlap.
- **Persistence:** `save_compressed` refuses centred int4 and 2-bit codecs, and does not
  store the PCA basis, so a saved projected index can't be queried after reload (Q4 §9).
- **Memory:** single-tier RAM equals the index size. The dense output score matrix grows
  with queries × pages (about 1.8 GB for 451 queries at 1M pages, PROJECTED).
- **Format effects:** per-vector float16 scales cap int8 at 3.9x on 128-d and shrink the
  gain after narrow PCA. The PCA basis is a fixed cost that dominates tiny indexes at
  2,560-d.
- **Timing environment:** one laptop with two CPU speed states (up to about 2.8x apart)
  and standby interruptions (Q8 rerun). Absolute milliseconds are hardware-specific.
- **Bit-exact equality is shape-dependent:** BLAS kernels differ for products with very
  few rows (ENG-1).

## 10. Generalized findings (within the tested set; Q9)

- vector count is the latency lever;
- relative merging beats pruning and preserves quality to about 1/4 of the vectors;
- int8 and centred int4 are near-lossless;
- factors do not multiply, and the axes interact;
- two-tier preserves quality and cuts RAM;
- certification difficulty drives the dataset differences;
- method C's conditional validity held.

**Tested set:** 4 visual encoders (three 128-d, one 2,560-d), 1 text encoder, 3 corpora.

## 11. Scope limitations

**Data and evaluation:**
- ViDoRe V1 DocVQA and InfoVQA, 500 pages each; one text corpus (SciFact).
- One query pool per corpus.
- ColEmbed with aggregates only, and GPU-only timing.
- Coarse operating-point grids.

**Hardware:** one laptop CPU.

**Not measured:**
- corpora above 500 pages for quality, or above 4,000 duplicated pages for timing;
- 4,096-d or other model families;
- ViDoRe V2/V3.

## 12. Unresolved questions

- **The ColQwen2 disagreement with SAP:** upstream of evaluation; possibly encoding
  settings (E7c).
- **The residual 19–39% of the certification gap** (Q3), and the ColPali residual (Q3H).
- **The cause of plain int4's collapse on ColEmbed** (R12).
- **Batched rescoring at very low candidate overlap** (Q8).
- **Retrieval quality at large corpus sizes,** where R10 suggests decline for lossy
  pipelines.
- **Whether compiled native kernels would beat decode + BLAS** (Q5/Q6 recommendation).
- **The novelty of findings #3 and #10** against the full literature.

## 13. Novelty and priority limitations

- **Ward token pooling** is Clavié et al. (2024). OptiVision's merge stage is that
  method, with norm rescaling and relative budgets.
- **Two-tier shortlist-and-rescore** and **binary multi-vector codes** are established
  practice.
- **Method C's statistical machinery is established;** its application here was not
  checked against the statistics or learn-then-test literature.
- **Q7 covered compression and engine literature only,** and found one Tier A comparison
  (SAP), whose disagreement is unresolved.
- **Conclusion:** the investigation establishes the empirical findings, but does not
  establish priority or novelty relative to the complete literature.

**Earlier wording, documented and not rewritten.**
- **Phase 0** (`docs/RESEARCH_INVESTIGATION.md`, Q2) said a valid post-selection statement
  "is a real contribution". That was a pre-investigation expectation. Q10 classifies the
  result as a validated application of known methods (§7).
- **Q4 / R12:** the "width is the most efficient axis" reading was qualified in Q4 (§15).
- **Other corrections already made:** the ColPali checkpoint label (Q4), the
  protected-token convention (Q7), and the E7c plan's quoted median (5.6 → 4.82). Each
  is documented where it occurred.

## 14. Claims that should NOT be made (README, paper, website)

**Novelty and comparison:**
- "first", "novel method", "new compression algorithm", "state of the art";
- "better than SAP" or "worse than SAP". The comparison is an unresolved disagreement,
  not evidence of superiority or inferiority.

**Generality:**
- "universal", "works with any multi-vector or multimodal model", or anything about
  4,096-d or untested models.

**Guarantees:**
- "guaranteed retention", without the full condition: method C only, i.i.d. queries,
  fixed corpus, tested pools, never at 0.99 with ≤ 225 queries.
- that the shipped `optimize()` rule has a guarantee. Its rates are empirical.

**Scale:**
- "scales to millions of pages", "handles 1M pages", "production-ready at scale". The
  1M figures are projected resource arithmetic, not demonstrated capability, and
  quality at scale is unmeasured.

**Speed:**
- "compressed or binary codes make search faster", or "N× faster search from
  quantization". That is not true of the current code.
- "native compressed scoring is slower" as a general statement. It holds only for this
  software stack.
- the batched-rescoring speedup as a research result, or as hardware-independent.

**Specific numbers:**
- that a specific compression factor (e.g. "125x") is safe without its model, dataset,
  retention and interval.

## 15. Strongest defensible overall contribution

OptiVision contributes:
1. **An engineered, model-agnostic multi-vector compression and selection framework,**
   validated on 5 encoders (4 visual, 1 text) and released as v0.3.0.
2. **A pre-registered empirical study that maps how the compression axes behave and
   interact,** and why datasets differ.
   - The key finding: certification difficulty, not compressibility.
   - The study states where each finding generalizes and where it does not.
3. **A validated, conditional finite-sample certification procedure** for the selected
   compression configuration (method C), with its sample-size limits.

Its research value lies in **established empirical findings and their measured scope**,
not in a new algorithm. Novelty relative to the full literature is not established.

## 16. Final Q10 verdict

**Mixed research + engineering contribution.**
- **Research:** the findings are established empirically (within the tested scope), but
  their novelty is not established.
- **Methodology:** method C is a validated application of known methods, not a new
  theory.
- **Engineering:** the framework, codecs, accounting, two-tier indexing and batched
  rescoring are substantial and validated.
- **Implementation-specific:** codec speed, native kernels and persistence gaps are
  observations about this implementation, not properties of compression.

Q10 is closed. No further research question has been started.
