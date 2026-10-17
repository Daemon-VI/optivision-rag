Literature digest (also at C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/literature/digest.md; the file holds a shortened version of this text). Exploratory scouting by web search/fetch on 2026-10-17; one or two queries per topic. Items marked (snippet) were not verified in the paper body. Vendor posts are not peer reviewed.

1. Residual compression and PLAID. ColBERTv2 stores centroid id plus a 1- or 2-bit residual per dimension, about 20-36 bytes per vector vs 256 (the byte figure is a secondary summary). On MS MARCO with vanilla ColBERT: float 36.2 MRR@10 / 82.1 R@50; 2-bit 36.2 / 82.3; 1-bit 35.5 / 81.6 (snippet). PLAID reports up to 6.8-7x GPU and 45x CPU speedup at comparable quality. Placement: 2-bit residual coding is a near-neutral, trained-for-it baseline on a text model; OptiVision's post-hoc int4/binary on arbitrary checkpoints is a different setting (no per-centroid residual).

2. Token pooling. Clavie, Chaffin, Adams 2024 (arXiv 2409.14683) and the Answer.AI post: hierarchical clustering pooling beat k-means and sliding window. Unquantised BEIR small/medium sets, relative to unpooled: factor 2 (about 50% fewer vectors) about 100.6%; factor 3 about 99%; factor 4 about 97%; worse beyond 4. With 2-bit ColBERTv2: most sets lose <3% at factor 2, Touche rises up to 40%, fiqa degrades faster. Abstract: about 50% index reduction with virtually no loss, 66-75% vector reduction with <5% loss on most datasets. Placement: text-model pooling results come from a different dataset family than SciFact and ViDoRe; the fiqa outlier shows dataset-dependent heterogeneity.

3. Visual-token merging vs pruning. Light-ColPali (arXiv 2506.04997), ColQwen2/ColPali, ViDoRe, nDCG@5: semantic-clustering merging retains 97.5% (factor 9) and 92.6% (factor 25) training-free, avg of six datasets; fine-tuned Light-ColQwen2 98.2% at 11.8% memory (factor 9), 96.3% at 3.0%, 94.6% at 1.8-2.8%. Pruning is worse: at factor 9 ColQwen2+pruning 90.9% rel., ColPali+pruning 93.6%; on DocVQA/InfoVQA best pruning keeps 78.3%/88.5% at ratio 0.9; random pruning beat score-based by 3.9 and attention-based by 19.6 nDCG points on InfoVQA at 0.95. Merging location matters (post-projector 81.4 vs pre-encoder 59.1 avg nDCG@5). Placement: directly the same datasets (DocVQA, InfoVQA) as the image side of the campaign; compare hierarchical_merge/adaptive_merge retention at matched vector counts. Fine-tuning recovers 61-67% of the loss at factors 25/49.

4. DocPruner (arXiv 2509.23883). Per-document adaptive threshold tau = mean + k*sd of patch attention scores (k=-0.25 best). ViDoRe-V2 nDCG@5: ColQwen2.5 0.5508 to 0.5470 at 51.6% pruned; ColNomic 0.5946 to 0.5960 at 43.6%; JinaV4 0.5687 to 0.5608 at 54.1%. JinaVDR: ColNomic 61% pruned, 0.8151 to 0.8191. At about 75% compression ColNomic 0.5730 vs best merging 0.5426. On JinaV4 merging sometimes beats DocPruner (authors attribute to mean-pooling training). Encoding time +60-66%. Placement: model-specific; the method needs model attention and is not post-hoc on embeddings alone. Pruning-vs-merging ranking differs by model.

5. Other visual compression. GLIE (arXiv 2609.11808, abstract): k-means centroids renormalised to the unit sphere, vectors near a manifold of intrinsic dimension 5-6, 4 vectors per page keeps nearly 80% of nDCG@5 on ViDoRe v1 vs 70% for best prior post-hoc; normalising centroids gives up to +0.093 nDCG@5 over raw centroids. Relevant to why lloyd2/merge need re-normalisation. HPC-ColPali (arXiv 2506.21601): K-means codes 32x, pruning, ViDoRe nDCG@10 0.85 to 0.84 (K=256, p=60%); the paper itself says its numbers are estimates and has internal inconsistencies, so cite with caution. Static token pruning for text ColBERT (Lassance et al., arXiv 2112.06540): up to about 30% of tokens prunable on MS MARCO passages, harder on documents; later work says static heuristics degrade beyond 50% pruning.

6. Binary, int8, int4 quantisation. Vespa blog (scaling ColPali to billions), DocVQA test nDCG@5: float 52.4; binary hamming MaxSim 49.5 (about 94.5%, derived); binary + float rerank 51.6 (about 98.5%, derived); 32x storage, about 16KB per page, hamming MaxSim about 3.5-4x faster. Qdrant ColPali blog: binary + rescoring + oversampling roughly halves search time vs scalar quantisation at similar accuracy on a small UFO dataset. Qdrant multi-vector course: int8 4x memory, claims <1% loss with no measurement shown; 1-bit/1.5-bit/2-bit options; says 128-d binary 'presents unique challenges' and rescoring matters because small errors compound in MaxSim; centring not mentioned. Hugging Face embedding-quantisation blog (single-vector, 1024-d): binary + rescoring keeps about 96%, without rescoring about 92.5%; MongoDB/mixedbread posts about 95%. LFM2.5-ColBERT 350M model cards (third-party, small): 8-bit 0.741 vs bf16 0.740 NDCG@10, 4-bit 0.731 (98.7%), 50-100 queries per dataset, run-to-run noise up to 0.002. Placement: a 94.5% binary retention on DocVQA for ColPali (128-d) is a published reference point for the campaign's binary rows; single-vector retention at 1024-d is not comparable to 128-d multi-vector. I found no Jina/Weaviate/LightOn/Answer.AI quantisation benchmark for multi-vector within these searches.

7. MUVERA / fixed dimensional encodings (arXiv 2405.19504, NeurIPS 2024). ColBERTv2 (128-d) on six BEIR datasets: FDE dims 4,096-20,480 (10,240 used end to end); PQ-256-8 gives 32x with negligible loss; vs PLAID about 10% higher recall and about 90% lower latency (average over 6 datasets), within 0.4% of PLAID recall on MS MARCO. Theorem: with high probability the returned document's normalised Chamfer similarity is within epsilon of the best, with FDE dimension m^{O(1/eps)} log n. Weaviate implements MUVERA (v1.31) and its slides note worse recall unless rescored. Placement: this is a candidate-generation proxy, not a lossy storage format; the only formal guarantee found for multi-vector approximation.

8. Matryoshka and dimension reduction. Jina-ColBERT-v2 (arXiv 2408.16672): linear heads at 64/96/128 (and larger) trained with Matryoshka loss; 128 to 64 dims costs 0.01 nDCG@10 (1.59%), up to 50% storage saved; weight-tied truncation (MRL-E) did not preserve performance, authors hypothesise because the base ColBERT dimension is already low. Self-reported, no independent replication found. Placement: a trained-for-truncation model vs OptiVision project(k) / project(k, random) applied post hoc; the low-dimension caveat matches the Qdrant 128-d binary caveat.

9. Anisotropy, centring, whitening. Su et al. 2021 (arXiv 2103.15316): whitening improves isotropy and sentence-similarity results and can cut dimension and storage. A 2026 ANN-recall paper (arXiv 2609.00364, snippet) argues the common cone component cannot change exact rankings, so centring acts through approximate search or the encoder; reports that whitening rewrites 62% of true neighbours depending on frame. Practitioner post: a shared mean makes sign bits nearly constant, so subtract a corpus-sample mean (same for queries) before binarising (blog, not peer reviewed). RaBitQ (arXiv 2405.12497): vectors normalised relative to a reference centroid, then a random rotation, with a theoretical error bound for D-bit codes (secondary explainers; constants not checked). Jina arXiv 2602.00079 explicitly does no centring (lossless float32 coding, 1.5x). Placement: the evidence on centring before quantisation is mostly mechanistic or practitioner; I found no controlled multi-vector study isolating it, so a campaign finding about centring has little direct published comparison from this search.

10. Statistical guarantees. No paper found giving a distribution-free guarantee for retrieval quality after embedding compression. Closest: ConANN (VLDB 2025, conformal risk control on IVF, false-negative-rate target, deviation about 0.03 points reported, snippet); HNSW with CRC / Learn-then-Test (arXiv 2607.02338); ConRAD (arXiv 2605.03806); Intrator et al. FEVER 2024 conformal IR. These control ANN recall against exact search, not nDCG retention against the float baseline. A selection rule with lower-CI bounds over calibration queries and holdout splits, as in selection_rules, is thus an empirical rather than guaranteed analogue.

Limits of this scout: search-snippet depth for some items, no Google Scholar, English only, single pass, arXiv ids from 2026 are taken from the search tool and not independently confirmed. No novelty claim should be made from absence of hits.

[
 {
  "topic": "ColBERTv2 residual compression",
  "url": "https://arxiv.org/pdf/2112.01488",
  "claim": "MS MARCO vanilla ColBERT: float 36.2 MRR@10/82.1 R@50; 2-bit 36.2/82.3; 1-bit 35.5/81.6 (snippet); about 20-36 bytes per vector vs 256 (secondary)",
  "setting": "ColBERT/ColBERTv2, MS MARCO, text"
 },
 {
  "topic": "PLAID",
  "url": "https://arxiv.org/pdf/2205.09707",
  "claim": "Up to about 7x GPU and 45x CPU speedup over ColBERTv2 at comparable MRR@10/Recall@100",
  "setting": "ColBERTv2, MS MARCO v1"
 },
 {
  "topic": "Token pooling",
  "url": "https://arxiv.org/abs/2409.14683",
  "claim": "About 50% index reduction with virtually no loss; 66-75% vector reduction with <5% degradation on most datasets",
  "setting": "ColBERT-like text models, BEIR-style datasets"
 },
 {
  "topic": "Token pooling blog",
  "url": "https://www.answer.ai/posts/colbert-pooling.html",
  "claim": "Hierarchical best; unquantised factor 2 about 100.6%, 3 about 99%, 4 about 97%; 2-bit: <3% loss at factor 2 except fiqa; Touche up to +40%",
  "setting": "ColBERT, small/medium BEIR subsets, 16-bit and ColBERTv2 2-bit"
 },
 {
  "topic": "Light-ColPali visual token merging",
  "url": "https://arxiv.org/html/2506.04997v1",
  "claim": "Merging retains 97.5% (factor 9) / 92.6% (factor 25) training-free; fine-tuned 98.2% at 11.8% memory, 94.6% at 2.8%; pruning 90.9-93.6% at factor 9; random pruning beats attention pruning",
  "setting": "ColQwen2, ColPali, six ViDoRe datasets, nDCG@5, incl. DocVQA and InfoVQA"
 },
 {
  "topic": "DocPruner",
  "url": "https://arxiv.org/html/2509.23883v1",
  "claim": "44-54% patches pruned with -1.4% to +0.2% nDCG@5 change on ViDoRe-V2; beats merging at about 75% compression for ColNomic",
  "setting": "ColQwen2.5, ColNomic, JinaV4; ViDoRe-V2, JinaVDR"
 },
 {
  "topic": "GLIE late-interaction compression",
  "url": "https://arxiv.org/abs/2609.11808",
  "claim": "4 vectors per page keeps nearly 80% nDCG@5 vs 70% best prior post-hoc; renormalised centroids up to +0.093 nDCG@5; vectors near manifold of intrinsic dim 5-6",
  "setting": "Three visual encoders (unnamed on page), ViDoRe v1/v2"
 },
 {
  "topic": "HPC-ColPali",
  "url": "https://arxiv.org/html/2506.21601v2",
  "claim": "Claims 32x K-means code compression, nDCG@10 0.85 to 0.84; paper states numbers are estimated and has internal inconsistencies",
  "setting": "ColQwen2.5, ViDoRe, SEC-Filings"
 },
 {
  "topic": "Binary ColPali (Vespa)",
  "url": "https://blog.vespa.ai/scaling-colpali-to-billions/",
  "claim": "DocVQA nDCG@5 float 52.4, binary hamming 49.5, binary+float rerank 51.6; 32x storage, hamming MaxSim about 3.5-4x faster",
  "setting": "ColPali 128-d, 1030 patches per page, DocVQA test"
 },
 {
  "topic": "Qdrant quantisation guidance",
  "url": "https://qdrant.tech/course/multi-vector-search/module-3/quantization-techniques/",
  "claim": "int8 4x memory, claims <1% loss unmeasured; 128-d binary is challenging; suggests 1.5/2-bit and float rescoring; no centring mentioned",
  "setting": "ColModernVBERT 128-d, course material"
 },
 {
  "topic": "Qdrant ColPali binary quantisation",
  "url": "https://qdrant.tech/blog/qdrant-colpali/",
  "claim": "BQ with rescoring and oversampling cuts search time nearly half vs scalar quantisation with similar accuracy",
  "setting": "ColPali, small UFO dataset"
 },
 {
  "topic": "Single-vector binary/int8 quantisation",
  "url": "https://huggingface.co/blog/embedding-quantization",
  "claim": "Binary plus rescoring keeps about 96%, without rescoring about 92.5%, 32x smaller",
  "setting": "Single-vector 1024-d text embeddings, not multi-vector"
 },
 {
  "topic": "Small ColBERT 4/8-bit model cards",
  "url": "https://huggingface.co/mlx-community/LFM2.5-ColBERT-350M-4bit",
  "claim": "bf16 0.740, 8-bit 0.741, 4-bit 0.731 NDCG@10 (98.7%); run-to-run noise up to 0.002",
  "setting": "LFM2.5-ColBERT 350M, NanoBEIR + MIRACL, 50-100 queries per dataset, third-party"
 },
 {
  "topic": "MUVERA",
  "url": "https://arxiv.org/html/2405.19504v1",
  "claim": "FDE 4,096-10,240-d, PQ 32x with negligible loss; about 10% higher recall, about 90% lower latency vs PLAID; epsilon-approximation theorem for Chamfer similarity",
  "setting": "ColBERTv2 128-d, six BEIR datasets"
 },
 {
  "topic": "Weaviate MUVERA",
  "url": "https://weaviate.io/blog/muvera",
  "claim": "MUVERA encoding implemented in Weaviate 1.31; lower memory, worse recall unless rescored (slides)",
  "setting": "Vendor engineering post; benchmark section not read"
 },
 {
  "topic": "Matryoshka multi-vector",
  "url": "https://arxiv.org/pdf/2408.16672",
  "claim": "Dimension 128 to 64 costs 0.01 nDCG@10 (1.59%), up to 50% storage saved; weight-tied MRL-E truncation did not preserve performance",
  "setting": "Jina-ColBERT-v2, multilingual retrieval, self-reported"
 },
 {
  "topic": "Whitening",
  "url": "https://arxiv.org/abs/2103.15316",
  "claim": "Whitening improves isotropy and sentence-similarity results, reduces dimension and storage",
  "setting": "Single-vector sentence embeddings (BERT-like)"
 },
 {
  "topic": "Embedding statistics and ANN recall",
  "url": "https://arxiv.org/pdf/2609.00364",
  "claim": "Argues the common cone component cannot change exact rankings; centring/whitening effects depend on frame (whitening rewrites 62% of true neighbours); snippet only",
  "setting": "Single-vector embeddings, ANN recall"
 },
 {
  "topic": "Centring before binarisation (practitioner)",
  "url": "https://dev.to/ji_ai/binary-quantized-embeddings-32x-smaller-if-you-center-first-4bk1",
  "claim": "Shared mean makes bits near-constant; subtract corpus-sample mean before sign, same centre for queries; blog, not peer reviewed",
  "setting": "Single-vector embeddings"
 },
 {
  "topic": "RaBitQ",
  "url": "https://arxiv.org/abs/2405.12497v1",
  "claim": "Centroid-relative normalisation plus random rotation gives D-bit codes with a theoretical error bound (constants not checked)",
  "setting": "Single-vector ANN, SIGMOD 2024"
 },
 {
  "topic": "Lossless float32 embedding compression",
  "url": "https://arxiv.org/html/2602.00079v4",
  "claim": "Spherical-angle coding gives about 1.5x, no centring or rotation, BEIR SciFact nDCG@10 unchanged to six decimals",
  "setting": "20 text models, ColBERT-style jina-colbert-v2 and BGE-M3 included"
 },
 {
  "topic": "Static token pruning for ColBERT",
  "url": "https://arxiv.org/abs/2112.06540",
  "claim": "Up to about 30% of index tokens prunable on MS MARCO passages without significant drop; documents harder",
  "setting": "ColBERT, MS MARCO"
 },
 {
  "topic": "Conformal guarantees for ANN",
  "url": "https://www.vldb.org/pvldb/vol19/p29-horchidan.pdf",
  "claim": "Conformal risk control for IVF ANN false-negative rate (ConANN, per search summary); no conformal guarantee for quantised or multi-vector retrieval found",
  "setting": "Single-vector IVF/HNSW indexes, recall vs exact search"
 },
 {
  "topic": "Conformal risk control with HNSW",
  "url": "https://arxiv.org/pdf/2607.02338",
  "claim": "CRC and Learn-then-Test certify per-query recall of approximate HNSW results against exact search",
  "setting": "HNSW single-vector retrieval"
 }
]