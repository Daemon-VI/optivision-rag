# Q7 report: what can and cannot be compared between OptiVision and existing multi-vector compression methods

Branch `research-investigation`. Tier criteria and comparison rules are fixed in
`PLAN.md` (`43123aa`), before any external result was read. No experiment was run
and no other method was implemented. The library is unchanged. Q8 is not started.

**Labels.**
- **MEASURED:** OptiVision's own results, from the artifacts cited.
- **EXTERNAL:** numbers published by others, kept in their original wording and with
  their own baselines.
- **PROJECTED:** an estimate. None of ours appear here; one external source's numbers
  are flagged as the authors' own estimates (§5).
- **INFERENCE:** our reading.

**How the sources were checked.**
- External facts come from primary sources: arXiv abstract and HTML pages, the ACL
  Anthology and official repositories. One vendor blog is used and labelled BLOG.
- Most were collected by three literature-search passes, using a summarizing web
  reader that can misread tables.
- **Re-checked against the primary source by a second, independent fetch:**
  - ColPali §5.2 and Appendix B.3;
  - ColBERTv2 §5.3;
  - SAP Tables 7 and 8;
  - Light-ColPali's training setup and memory definition;
  - the tiers of AdaMerge, PtM and the Visual RAG Toolkit;
  - the training-free comparison by Jha et al.;
  - the Vespa blog.
- Every other external number is marked "single extraction" and should be read
  against the PDF before being quoted elsewhere.

**One correction made during Q7.** OptiVision's ColPali checkpoint is
`vidore/colpali-v1.3-merged`, not "v1.2" as the Q4 documents said. This was a label
error only (`a59ee31`).

## 1. OptiVision's side of the comparison (MEASURED)

**Setup.**
- **Encoders:** ColPali `vidore/colpali-v1.3-merged`, ColQwen2
  `vidore/colqwen2-v1.0-merged` and ColQwen2.5 `vidore/colqwen2.5-v0.2`, all 128-d;
  ColEmbed 4B, 2,560-d (R12; `docs/UNIVERSAL.md`).
- **Benchmark:** ViDoRe V1 DocVQA and InfoVQA test splits: 500 pages, with 451 / 494
  queries that have relevance labels.
- **Metric:** nDCG@5, with retention = mean(compressed) / mean(uncompressed float32)
  for the same encoder.
- **Transformations:** post hoc and training-free.
  - Ward merging within each page; the merged vector is the cluster mean, rescaled to
    the members' mean norm.
  - PCA fitted on the documents.
  - int8, centred int4, 2-bit, binary.
  - Combinations of these, and a two-tier index (binary shortlist, exact int8
    rescoring).
- **Scoring:** exact MaxSim.
- **Accounting:** storage as actual array bytes per page; latency on a laptop CPU with
  the existing numpy implementation (Q4, Q5/Q6), plus R12's GPU timings for ColEmbed.
- **Evidence:**
  - Q4: the axis ablation, 6 datasets × 74 configurations;
  - Q3 and E1: certification and selection;
  - Q5/Q6: native scoring;
  - the batched-rescore validation;
  - R11 and R12: the ColQwen and wide-model validations.

## 2. Comparability tiers (criteria from `PLAN.md` §2, applied to each source's setup)

The four Tier A conditions:
1. **Same encoder family** (ColPali or ColQwen2 / 2.5).
2. **ViDoRe V1 DocVQA and/or InfoVQA, with nDCG@5** against the same encoder's
   uncompressed baseline.
3. **Post hoc compression that keeps late interaction.**
4. **Storage derivable** as bytes per page, or as a ratio against an explicit float
   baseline.

- **Tier A:** all four hold.
- **Tier B:** at least two hold, with at least one major mismatch.
- **Tier C:** fewer than two hold.

| method / setup | condition 1 | 2 | 3 | 4 | tier |
|---|---|---|---|---|---|
| **SAP, Structural Anchor Pruning** (Liu, Hu, Zhang, Xiao; arXiv 2601.20107v3, 29 Aug 2026), with its in-table baselines *Cluster* (K-means) and *Random* | yes (`vidore/colpali-v1.3`, `vidore/colqwen2-v1.0`) | yes (per-subset DocVQA and InfoVQA, nDCG@5) | yes (training-free, index-time, MaxSim kept) | yes (γ = fraction of vectors kept; an index-size example is given) | **A** |
| ColPali paper, hierarchical token pooling (Faysse et al., arXiv 2407.01449v6, ICLR 2025, §5.2) | yes (ColPali; version not stated in the sentence) | partly (ViDoRe; the metric for "97.8%" is not named in the text; per-dataset only in Fig. 3) | yes | yes (vector count) | B |
| Light-ColPali / Light-ColQwen2 (Ma et al., arXiv 2506.04997v1, Jun 2025) | no (the authors **retrained** ColPali and ColQwen2 from PaliGemma and Qwen2-VL; not the released checkpoints) | yes for fine-tuned rows; training-free rows only as a 6-dataset average (Fig. 4) | headline rows need fine-tuning | yes (#Mem relative to single-vector DSE; float16) | B |
| DocPruner (Yan et al., arXiv 2509.23883v1, Sep 2025) | yes (ColQwen2.5; version not verified) | no (ViDoRe **V2** and JinaVDR) | yes (training-free, index-time pruning) | yes (vectors kept) | B |
| Prune-then-Merge (PtM) (Yan et al., arXiv 2602.19549v2, Apr 2026) | yes (ColQwen2.5; version not verified) | partly (ViDoRe V1 **aggregate only**, plotted; no per-subset table) | yes | yes (vectors kept) | B |
| AdaMerge (You and Ni, arXiv 2609.22562v1, 18 Sep 2026) | yes (ColQwen2.5-v0.2) | partly (ViDoRe V1 averages only) | yes | partly (vectors kept; precision not stated) | B |
| Visual RAG Toolkit (Yeroyan, arXiv 2602.12510v1, Feb 2026) | yes (ColPali; version not stated) | no (ViDoRe V2) | yes (pooling + exact MaxSim rerank) | no (qualitative "thousands to dozens") | B |
| Vespa binarization of ColPali (Bergum, Vespa blog, 20 Sep 2024) | partly (ColPali; checkpoint not stated) | yes (DocVQA, nDCG@5) | yes (binary, Hamming MaxSim, optional float rerank) | yes (32x against float32) | B (BLOG) |
| Training-free sequence-compression comparison (Jha, Zuo, Kriz, Van Durme, arXiv 2603.22434v1, Mar 2026) | no (ColBERT, text) | no (BEIR, CoIR) | yes | partly (vectors kept) | C |
| Token Pooling (Clavié, Chaffin, Adams, arXiv 2409.14683, 2024) | no (ColBERTv2, JaColBERTv2) | no | yes | partly | C |
| ColBERTv2 residual compression (Santhanam et al., NAACL 2022) | no | no | partly (needs a centroid codebook; trained model) | yes (16 / 25 GiB against 154 GiB) | C |
| PLAID, EMVB, WARP, XTR, CITADEL, SPLATE, DESSERT, MUVERA, ColBERT-serve, ColBERTSaR | no (text ColBERT, XTR or T5) | no (MS MARCO, BEIR, LoTTE) | varies; several change the retrieval algorithm or need training | varies | C |
| ColBERTer, token pruning (Lassance et al.), static pruning (Acquavia et al.), ConstBERT, Jina-ColBERT-v2 (Matryoshka), SDR, BTR | no | no | varies (ColBERTer, ConstBERT and SDR need training; BTR compresses an RAG reader, not a retriever) | varies | C |
| MetaEmbed, ColFlor, ModernVBERT, DistilVDR (new or smaller trained visual retrievers) | no (new models) | varies | no (trained representations) | varies | C |
| HPC-ColPali (Duong, arXiv 2506.21601v2) | – | – | – | – | **excluded**: the paper states its results are estimated (§5) |

**INFERENCE: only one source meets Tier A.** SAP, which reports per-subset ViDoRe V1
numbers for the exact checkpoints OptiVision measured.

## 3. Structured comparison table (EXTERNAL unless marked; original wording and baselines kept)

Mechanism key: **count** = changes vector count; **dim** = changes dimension;
**quant** = quantizes; **algo** = changes the retrieval algorithm; **train** =
needs training.

| method | mechanism (count / dim / quant / algo / train) | model-specific | benchmark and metric | reported quality | reported storage (baseline) | reported latency (hardware) | code | comparison status |
|---|---|---|---|---|---|---|---|---|
| **OptiVision** (MEASURED) | count, dim, quant, combinations; exact MaxSim; two-tier optional; no training | no (6 encoders, 128-d and 2,560-d) | ViDoRe V1 DocVQA and InfoVQA, nDCG@5 retention | Q4 grid; e.g. Ward to about 1/4 of the vectors: 0.989–1.006 on six 128-d datasets | actual bytes per page; codes + offsets + shared state (float32 in-memory baseline) | laptop CPU, numpy; scan time follows vector count (Q4) | this repo | – |
| SAP (2601.20107v3) | count (query-agnostic importance pruning from structural "anchors"); no training | the method uses the model's attention / layer structure | ViDoRe V1 (incl. DocVQA, InfoVQA) and V2, nDCG@5 retention | ColPali DocVQA at γ = 0.10: 88.71%; InfoVQA: 94.96% (Table 7) | γ = fraction of visual tokens kept; "γ = 0.10 shrinks index from 751 MB to 75 MB" (ViDoRe V2) | MaxSim 6.1 → 0.78 ms per query at γ = 0.10 (ViDoRe V2, V100 GPU) | not verified | **Tier A** (§4) |
| ColPali token pooling (2407.01449v6 §5.2) | count (hierarchical mean pooling); no training | no | ViDoRe | "With a pool factor of 3, the total number of vectors is reduced by 66.7% while 97.8% of the original performance is maintained." | 257.5 KB per page uncompressed (float16, App. B.3) | App. B.4 exists, not extracted | colpali-engine | B |
| Light-ColPali (2506.04997v1) | count (semantic clustering, post-projector), with fine-tuning; training-free variants also | built on the ColPali codebase | 9 datasets incl. ViDoRe InfoVQA and DocVQA, nDCG@5 | retrained ColQwen2-2B: InfoVQA / DocVQA 91.5 / 55.4 uncompressed; Light-ColQwen2 at merging factor 9: 90.4 / 56.1 (Table 2, single extraction) | #Mem relative to DSE single-vector = 1.0x; abstract: "98.2% … with only 11.8% of original memory" | offline only (training and embedding time, A100) | ColPali codebase | B |
| DocPruner (2509.23883v1) | count (adaptive attention pruning); no training | uses model attention | ViDoRe V2, JinaVDR, nDCG@5 | ColQwen2.5: −0.69% nDCG@5 at −51.55% storage (k = −0.25, single extraction) | vectors kept (unpruned baseline) | encoding 0.47 → 0.77 s per document (A100) | not verified | B |
| PtM (2602.19549v2) | count (prune, then hierarchical merge); no training | uses model attention | 29 VDR datasets; ViDoRe V1 aggregate | aggregate only ("54.60%" storage reduction across models) | vectors kept | not extracted | not verified | B |
| AdaMerge (2609.22562v1) | count (hierarchical clustering with an adaptive cut); no training | no | ViDoRe V1 and V2 averages, nDCG@5 | averages only | 1 − M/N vectors | about 9–10 ms per document to compress (one CPU thread) | not verified | B |
| Vespa ColPali binarization (BLOG) | quant (binary), Hamming MaxSim, optional float rerank; no training | no | DocVQA, nDCG@5 | float 52.4; binary–binary 49.5; binary + float rerank 51.6 | "save 32x on storage (memory) … instead of float" | Hamming about 3.5x faster than float dot product; "about 200M 128-bit hamming distances per second per CPU core" | Vespa | B |
| Jha et al. 2026 (2603.22434v1) | count (pruning and pooling variants incl. Ward); no training | no | BEIR (6), CoIR (3), nDCG@10, R@100 | hierarchical pooling at r = 0.20: BEIR nDCG@10 95.7% of baseline | token ratios only | complexity only | – | C |
| Token Pooling (2409.14683) | count (Ward clustering, mean pooling); no training | no | BEIR, LoTTE subsets | "50% reduction … virtually no retrieval performance degradation"; 66–75% fewer vectors with degradation "below 5% on most datasets" (abstract) | vector count | not reported | – | C |
| ColBERTv2 (NAACL 2022 §5.3) | quant (centroid + 1–2-bit residual); trained model | ColBERT | MS MARCO, BEIR, LoTTE | MRR@10 39.7 (single extraction) | "16 GiB or 25 GiB … compression ratios of 6–10×" against ColBERT's 154 GiB; includes a 4.5 GiB inverted list | 50–250 ms per query (Titan V + Xeon; single extraction) | stanford-futuredata/ColBERT | C |
| PLAID (CIKM 2022) | algo (centroid interaction) over ColBERTv2 codes | ColBERTv2 | MS MARCO etc. | "without impacting quality" | inherits ColBERTv2 | "up to 7× on a GPU and 45× on a CPU against vanilla ColBERTv2" | same | C |
| EMVB (ECIR 2024) | quant (PQ) + algo (bit-vector filtering) | ColBERTv2 / PLAID | MS MARCO, LoTTE | quality equal to PLAID | "reducing the memory footprint by 1.8×" against PLAID | "up to 2.8× faster" (single-thread Xeon Gold 5318Y) | CosimoRulli/emvb | C |
| MUVERA (NeurIPS 2024) | algo (fixed-dimensional encodings; approximate first stage) | no | BEIR | "10% improved recall with 90% lower latency" than prior heuristics | FDE + PQ (32x, per the Google blog) | not hardware-specific in the abstract | google/graph-mining | C |
| XTR, WARP, CITADEL, SPLATE, DESSERT, ConstBERT, ColBERTer, Jina-ColBERT-v2, token pruning, ColBERT-serve, ColBERTSaR, SDR, BTR | see the Tier C row of §2 | – | text benchmarks | see the literature-search notes (single extraction) | – | – | – | C |

## 4. A. Directly comparable evidence (Tier A: SAP)

**Source.** SAP Tables 7 and 8 (ColPali `vidore/colpali-v1.3` and ColQwen2
`vidore/colqwen2-v1.0` on ViDoRe V1; retention = nDCG@5 pruned / nDCG@5 full × 100).
- These were transcribed from the arXiv HTML (v3) by automated extraction and checked
  for internal consistency (NDCG@5 / upper bound ≈ %).
- They were not checked against the PDF.

**OptiVision's matching rows** are Ward merging at ratios 1/4, 1/8 and 1/10 (MEASURED,
`reports/research/Q4/store/`). Their actual vector fractions are 25.5–26.1%, 13.1–13.8%
and 10.7–11.4%.

**Uncompressed baselines (nDCG@5):**

| | OptiVision (MEASURED) | SAP (EXTERNAL) |
|---|---|---|
| ColPali DocVQA | 0.584 | 0.59 |
| ColPali InfoVQA | 0.846 | 0.85 |
| ColQwen2 DocVQA | 0.606 | 0.59 |
| ColQwen2 InfoVQA | 0.920 | 0.91 |

**Retention at about 1/10 of the vectors** (the only level both sides measured):

| | OptiVision Ward 1/10 (MEASURED; 10.7–11.4% kept) | SAP "Cluster", K-means (EXTERNAL; γ = 0.10) | SAP "Random" (EXTERNAL) | SAP (EXTERNAL) |
|---|---|---|---|---|
| ColPali DocVQA | 0.954 | 86.09% | 85.03% | 88.71% |
| ColPali InfoVQA | 0.990 | 94.64% | 93.16% | 94.96% |
| ColQwen2 DocVQA | 0.977 | 87.41% | 77.75% | 88.45% |
| ColQwen2 InfoVQA | 0.973 | 89.70% | 89.14% | 92.67% |

At the other levels the two sides don't share a γ. OptiVision measured 1/4 and 1/8;
SAP reports γ = 0.20 and 0.05. Those are listed separately and not paired:
- **OptiVision, Ward 1/4:** ColPali 0.989 / 0.999, ColQwen2 1.006 / 0.991.
- **SAP, γ = 0.20:** SAP 96.51 / 97.63 (ColPali) and 95.62 / 96.74 (ColQwen2); Cluster
  93.44 / 98.04 and 94.74 / 95.89.

**What this does and does not show.**
- **The quantities are comparable in form:** same checkpoints, same subsets, same
  metric and the same retention definition.
- **The values disagree in an unexplained way.** At a similar vector fraction,
  OptiVision's Ward merging retains more than SAP reports for its own clustering
  baseline: 9–10 points more on DocVQA and 4–8 points on InfoVQA. That is larger than
  OptiVision's sampling noise; its 95% intervals at Ward 1/10 span about ±1.3–2.7
  points. SAP reports no intervals.
- This is recorded as a **disagreement to investigate**, not as a ranking.
- **Identified methodological differences (INFERENCE; none tested):**
  1. **Merged-vector construction.** OptiVision rescales each cluster mean to the
     members' mean norm (`stages/merge.py::_pool`). SAP's "Cluster" baseline uses
     K-means centroids, which are plain means. A plain mean of dissimilar vectors is
     shorter, and MaxSim is a dot product, so merged vectors lose score.
  2. **Clustering algorithm:** Ward on unit-normalized vectors against K-means
     minimizing within-cluster sum of squares.
  3. **What γ counts.** SAP's γ counts *visual tokens*. OptiVision's ratio applies to
     all of a page's vectors (ColPali: 1,031 per page, including non-patch tokens).
  4. **Evaluation sets and tooling.** The ColQwen2 baselines differ by 1–1.6 points,
     so the query sets or evaluation code are not identical. SAP's query counts and
     toolkit version were not found.
- **Storage.** Both report a vector fraction. OptiVision's bytes per page also count
  offsets (and any shared state), which matters little at 128-d. SAP's index-size
  example is for ViDoRe V2. Bytes per page are therefore comparable only as "fraction
  of vectors × the same per-vector bytes".
- **Latency is not comparable.** SAP measured on a V100 GPU over ViDoRe V2;
  OptiVision on a laptop CPU over V1.

## 5. B. Related but not directly comparable evidence

- **ColPali's own pooling** (Tier B). The ColPali authors report "97.8% of the original
  performance" at pool factor 3 (66.7% fewer vectors).
  - The text doesn't name the metric, and per-dataset values are only plotted.
  - OptiVision's nearest row is Ward 1/3. On ColPali it retains 0.992 (DocVQA) and
    0.994 (InfoVQA), but against a different aggregate.
  - The two cannot be put on one scale.
  - The binarization sentence in the ColPali paper ("can reduce storage costs by two
    orders of magnitude with minimal performance hits") is not the authors' own
    measurement.
- **Light-ColPali** (Tier B). Its compressed and uncompressed rows come from models
  the authors *retrained*.
  - Their ColQwen2 DocVQA baseline is 55.4, against OptiVision's 60.6 with the
    released checkpoint.
  - The headline method needs fine-tuning (5 epochs on 8 A100s).
  - Its memory column is relative to a single-vector model.
  - Its training-free rows are only averages.
  - It does support one qualitative point OptiVision also measured: merging retains
    far more than pruning at equal reduction.
- **DocPruner, PtM, AdaMerge, the Visual RAG Toolkit** (Tier B). They are post hoc
  and use ColPali-family encoders, but report ViDoRe V2 or only aggregates. None
  reports the DocVQA or InfoVQA subsets OptiVision measured.
- **Vespa binarization** (Tier B, BLOG).
  - On DocVQA: float 52.4, binary query against binary documents 49.5, binary with
    float rerank 51.6.
  - The checkpoint and corpus size are not stated.
  - The float baseline (52.4) is 6 points below OptiVision's ColPali v1.3 DocVQA
    baseline (58.4), which suggests an earlier checkpoint.
  - The closest OptiVision measurements use different encoders or query modes, so
    none of them is the same setup:
    - binary documents with float queries on ColPali: 0.963 (Q4);
    - binarized query on ColQwen2: retention 0.948 for the hot stage alone, 0.998
      after int8 rescoring (Q5/Q6);
    - two-tier on ColPali DocVQA: 1.001.
- **Text late-interaction compression** (Tier C), as context only:
  - ColBERTv2's residual codes (6–10x against ColBERT's 154 GiB index);
  - EMVB's PQ (1.8x less memory than PLAID);
  - Token Pooling (50% fewer vectors, "virtually no" loss);
  - Jha et al. 2026 (merging beats pruning, training-free);
  - engines such as PLAID, WARP and MUVERA that change candidate generation.

  Their models, corpora and metrics (MRR@10, nDCG@10, Success@5) differ, so no number
  is compared.
- **Excluded source.** HPC-ColPali's results section states: "All numerical results
  presented herein are estimated based on the theoretical advantages of HPC-ColPali's
  design and typical performance gains observed in similar research." Its numbers are
  not used.

## 6. C. Missing evidence

- **No published per-subset ViDoRe V1 numbers for these setups:**
  - ColQwen2.5 compression (DocPruner, PtM and AdaMerge give V2 or averages only);
  - ColPali-family **dimension reduction**;
  - ColPali-family **scalar quantization** (int8, int4);
  - two-tier binary-plus-rescore designs.

  The only published ColPali binarization number is the Vespa blog's.
- **No external evaluation of compression for 2,560-d visual retrievers.** OptiVision's
  R12 ColEmbed 4B rows have no external counterpart.
- **No external work combines all three axes** (count + dimension + quantization) on
  ColPali-family embeddings, or measures their interactions. Light-ColPali explicitly
  lists dimension reduction and quantization as unexplored future work.
- **No external CPU latency for ColPali-family scoring** under exact MaxSim with
  stated hardware. The exception is Vespa's per-core Hamming throughput (BLOG).
- **Not obtained:** SAP's query counts, evaluation toolkit and code; ColPali B.4
  latency details; per-dataset ColPali pooling values (Fig. 3 only); several Tier C
  table values (single extraction).
- **Not verified beyond a search result:** PULSAR (arXiv 2608.28572), MURE
  (2603.13349), AdaptiveEmbed (2608.25412), EigenLI (2609.07561), "Towards Lossless
  Token Pruning in Late-Interaction Retrieval Models" (2504.12778) and ColBERTSaR's
  effectiveness table.

## 7. Answers

**Q7.1 Closest existing methods** (INFERENCE):
- **Same task, same checkpoints:** SAP and its training-free baselines (one axis: the
  vector count).
- **Same mechanism family:** ColPali's own hierarchical pooling, Light-ColPali's
  training-free merging, AdaMerge, and Token Pooling / Jha et al. on text, which use
  Ward clustering as OptiVision does.
- **Same quantization family, on visual retrieval:** the Vespa blog only.

**Q7.2 Combinations OptiVision covers that others don't directly** (INFERENCE from §6):
- the three-axis ablation on ColPali-family embeddings, with measured interactions
  (Q4);
- the same pipeline on a 2,560-d model;
- exact two-tier rescoring with its latency decomposed (Q5/Q6, batched rescore);
- finite-sample selection and certification statistics (E1, Q3).

None of the visual-retrieval sources found reports quantization combined with
merging.

**Q7.3 Model-specific versus model-agnostic:**
- **Model-agnostic** (operate on embeddings alone): OptiVision; ColPali pooling;
  AdaMerge; Token Pooling; the Jha et al. methods; MUVERA; Vespa binarization.
- **Use model internals:** SAP (layer and attention structure); DocPruner and PtM
  (attention); Light-ColPali (merge placement inside the model, with fine-tuning).
- **Tied to an index format or training recipe:** ColBERTv2, PLAID, EMVB, WARP, XTR,
  CITADEL, ConstBERT and ColBERTer.

**Q7.4 Training, calibration or adaptation needs:**
- **Fine-tuning:** Light-ColPali (headline), ColBERTer, ConstBERT, CITADEL, XTR and
  SPLATE (an adapter).
- **Fitted codebooks:** ColBERTv2, PLAID and EMVB (EMVB's JMPQ also uses training
  queries).
- **Training-free:** SAP, DocPruner, PtM, AdaMerge, the pooling methods, MUVERA,
  OptiVision's stages and Vespa's binarization.
- **Calibration:** OptiVision's optimizer additionally uses labelled calibration
  queries to *choose* a configuration. Its fixed pipelines need none.

**Q7.5 Quantities that are actually comparable:**
- **Today:** only nDCG@5 retention on ViDoRe V1 DocVQA and InfoVQA, against vector
  fraction, for ColPali v1.3 and ColQwen2 v1.0 (SAP against OptiVision).
- **Not comparable on current evidence:** storage in bytes (others report vector
  fractions or relative memory), latency (different hardware and protocols), and
  every number from Tier B and C.

**Q7.6 Where OptiVision's evidence is broader or different** (descriptive):
- **Scope:** three axes and their combinations on four encoders including one
  2,560-d model; actual byte accounting; CPU latency per axis; the native-scoring and
  batched-rescore results.
- **Statistics:** held-out frontiers; finite-sample selection guarantees for the
  chosen configuration (E1); certification analysis (Q3).
- **Narrowness:** OptiVision evaluates only two ViDoRe V1 subsets.

**Q7.7 Where existing methods have evidence OptiVision lacks:**
- **Wider benchmarks:**
  - all ViDoRe V1 datasets, plus V2, JinaVDR, MMLongBench and 29-dataset suites (SAP,
    DocPruner, PtM, Light-ColPali);
  - multilingual evaluation (DocPruner).
- **Different approaches:**
  - query-agnostic importance pruning using model attention (SAP, DocPruner);
  - fine-tuned merging that recovers quality at extreme factors (Light-ColPali);
  - large-scale engine latency on text corpora up to 140M passages (PLAID);
  - learned candidate generation (SPLATE, XTR, CITADEL).
- **Practice:** production binarization (Vespa).

**Q7.8 What a true apples-to-apples comparison would need** (specified, not run):
- **E7a: reconcile OptiVision's Ward merging with SAP's "Cluster" baseline.** On the
  six Q4 stores (CPU, existing vectors, no encoding), apply at γ ∈ {0.20, 0.10, 0.05},
  counting all vectors and, separately, patch tokens only:
  1. K-means with plain-mean centroids, as in SAP;
  2. Ward with plain means;
  3. Ward with OptiVision's norm rescaling;
  4. random pruning.

  Report per-subset nDCG@5 retention with paired intervals. This attributes the
  4–10-point disagreement to norm handling, clustering algorithm, token set or
  evaluation set. It needs no external code. Estimated effort: about 1–2 hours of
  CPU (PROJECTED).
- **E7b: run SAP itself.** This needs per-page attention and layer statistics from the
  encoder, which OptiVision's stored vectors do not include. It would mean GPU
  re-encoding with SAP's released code, if available, and is therefore a separate
  GPU experiment needing approval.
- **E7c: align the evaluation set** with the official ViDoRe toolkit (query counts and
  qrels), to explain the 1–1.6-point ColQwen2 baseline gap before any further
  numerical comparison.
- **No other method is a candidate for a numerical comparison** without new encoding
  or reimplementation. Tier B and C methods would need their own evaluation on
  OptiVision's datasets, a reimplementation decision outside Q7.

## 8. Conclusion (the evidence landscape)

- **Only one published setup is directly comparable:** SAP's ViDoRe V1 tables for
  ColPali v1.3 and ColQwen2 v1.0.
- **It disagrees by 4–10 points.** At about 1/10 of the vectors, OptiVision's measured
  Ward merging retains more nDCG@5 than SAP reports for its own training-free
  clustering baseline: 9–10 points more on DocVQA and 4–8 on InfoVQA. Several concrete methodological differences
  could explain this; until E7a attributes it, neither number should be read as
  showing one approach is better.
- **Everything else is context:**
  - visual-retrieval papers on other benchmarks, aggregates, retrained models or
    blog evidence;
  - text ColBERT compression and engines on MS MARCO, BEIR and LoTTE.

  Their numbers cannot be compared with OptiVision's without new experiments.
- **No leaderboard is possible on current evidence, and none is offered.**
