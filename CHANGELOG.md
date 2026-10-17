# Changelog

## Unreleased

### Kaggle measurement campaign, batches 01-04

- **14 more models measured** on ViDoRe V1 DocVQA + InfoVQA (500 pages each) over two Kaggle
  T4s, at the revisions pinned in the registry: `vidore/colpali`, `colpali-v1.1`, `-v1.2`,
  `-v1.2-merged`, `-v1.3`, `vidore/colsmolvlm-v0.1`, `vidore/colSmol-500M`,
  `vidore/colqwen2-v0.1`, `vidore/colqwen-omni-v0.1`, `nomic-ai/colnomic-embed-multimodal-3b`
  and `-7b` (float16 inference), `tsystems/colqwen2.5-3b-multilingual-v1.0` and its `-merged`
  copy, `ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1` (float16 inference). Their registry
  status is `measured` and the catalogue (`presets/measured.json`) holds their fixed-pipeline
  rows: 22 models, 2484 rows. 24 of the 65 registry models are measured. Records:
  `reports/universal/campaign/`, one folder per model plus `outcomes.json`.
- **Four failures are recorded, not hidden.** `vidore/colqwen2.5-v0.1` and
  `Metric-AI/ColQwen2.5-3b-multilingual-v1.0` ran a T4 out of memory on one page at their
  checkpoints' own page-size limit (16384 visual tokens). `Cognitive-Lab/ColNetraEmbed`'s load
  was refused because its language-model keys are nested one level deeper than ColGemma3
  expects. `Metric-AI/ColQwen2.5-7b-multilingual-v1.0` cannot be loaded at all: its adapter
  names a base model that is not on the Hugging Face Hub (now `unverified`).
- **Page-resolution cap.** `ColVLMEncoder` and `ColEngineTokenEncoder` take
  `max_num_visual_tokens`. Registry entries can carry `load_options`, which `load_adapter`
  passes unless the caller overrides them. The two ColQwen2.5 checkpoints above get the
  768-token cap their model cards state. A cap set on an encoder is part of the encoder
  identity.
- **Nested checkpoint keys.** When the weights a load leaves missing are exactly the
  checkpoint's unmatched tensors one nesting level off, the loader reloads once with those
  names mapped and records the rules in `loading_report["key_remap"]`. Any other mismatch is
  still an error.
- `scripts/ingest_campaign.py` copies a batch's `results_<batch>.zip` into the repository.
  `scripts/build_measured_catalog.py` reads `campaign/outcomes.json`. A retry notebook,
  `notebooks/kaggle_batches/batch_12_retry.ipynb`, re-runs the three models the fixes cover.

### Kaggle measurement campaign, batches 05 and 08-10

- **5 more models measured**, at their pinned revisions (each equal to the Hub's main):
  `vidore/colpali-v1.2-hf`, `vidore/colpali-v1.3-hf`, `vidore/colqwen2-v1.0-hf` (float16
  inference through sentence-transformers 6.1.0), `ModernVBERT/colmodernvbert` (float32,
  sentence-transformers) and `TomoroAI/tomoro-colqwen3-embed-8b` (float16 over two T4s). That
  is the first time the sentence-transformers image path has run here. 29 of the 65 registry
  models are now measured; the catalogue holds 27 models and 3044 rows.
- **11 failures are recorded in the registry evidence**:
  - The two `Hcompany/NeoMME` models need transformers 5.17; the session had 5.16.1.
  - `tencent/EVIE-8B` and `webAI-Official/webAI-ColVec1.1-8b` were split over two GPUs, but
    sentence-transformers moved them back onto one, which ran out of memory.
  - `webAI-Official/webAI-ColVec1-4b`'s own `from_pretrained` returns no loading info.
  - `nvidia/llama-nemotron-colembed-vl-3b-v2` passed `check_retriever` but was killed for host
    memory: 500 pages of ~1800 × 3072 float32 vectors.
  - `jinaai/jina-embeddings-v4`, `ibm-granite/granite-vision-3.3-2b-embedding` and
    `5CD-AI/Vintern-Embedding-1B` ship remote code that fails under transformers 5.
  - The two `nvidia/llama-nemoretriever-colembed` v1 models import flash_attn, which does not
    run on a T4.
- **The sentence-transformers adapter keeps a `device_map` placement.** sentence-transformers'
  `encode` calls `model.to(device)` on every call; the adapter now leaves the split in place.
- **A `from_pretrained` that returns the model alone is still checked.** The loader compares
  the model's state-dict keys with the tensor names in the checkpoint's safetensors headers
  and refuses the load if any weight would be random, as before.
- `notebooks/kaggle_batches/batch_13_retry.ipynb` re-runs the five models these fixes cover,
  with transformers>=5.17.

### Kaggle measurement campaign, batch 07

- **3 more models measured** through colpali-engine (ColQwen3_5), at their pinned revisions:
  `vultr/VultronRetrieverFlash-Qwen3.5-0.8B` and `vultr/VultronRetrieverCore-Qwen3.5-4.5B`
  (float32), `vultr/VultronRetrieverPrime-Qwen3.5-8B` (float16 inference over two T4s; the
  session had no time left for its selection study). 32 of the 65 registry models are now
  measured; the catalogue holds 30 models and 3428 rows.

### Kaggle measurement campaign, batch 06

- **3 more models measured** through sentence-transformers 6.1.0 (float16 inference, one T4), at
  their pinned revisions: `VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1`, `-4b-v0.1` and
  `tencent/EVIE-4.5B` (2048 dimensions). 35 of the 65 registry models are now measured; the
  catalogue holds 33 models and 3782 rows.
- `webAI-Official/webAI-ColVec1.1-4b` loaded but failed `check_retriever`: 9 of 12 queries
  ranked their own page first, and 10 are required. Its card states bfloat16 weights and
  evaluation, which a T4 cannot run natively. `batch_14c_colvec_retry` runs it in float32 over two T4s
  (it was added to `batch_13_retry` after that batch had already been imported).

### Kaggle measurement campaign, batch 12_retry

- **The three re-runs after the batch 01-04 fixes are measured** (float32 over two T4s, at
  their pinned revisions). `vidore/colqwen2.5-v0.1` and
  `Metric-AI/ColQwen2.5-3b-multilingual-v1.0` ran with pages capped at 768 visual tokens (about
  740 vectors per page). `Cognitive-Lab/ColNetraEmbed` loaded with the released colpali-engine
  0.3.18 once its nested `language_model.model.*` keys were mapped; the mapping is recorded in
  `loading_report["key_remap"]`. 38 of the 65 registry models are now measured; the catalogue
  holds 36 models and 4106 rows.

### Kaggle measurement campaign, batches 11_text and 13_retry

- **13 more models measured.** 11 text models were measured on BEIR SciFact through
  sentence-transformers 6.1.0: `NeuML/colbert-bert-tiny`, the two `mixedbread-ai/mxbai-edge-colbert-v0`
  models (17m, 32m), `colbert-ir/colbertv2.0`, `lightonai/GTE-ModernColBERT-v1`,
  `Reason-ModernColBERT`, `ColBERT-Zero`, `LateOn`, `mLateOn`, `mixedbread-ai/mxbai-colbert-large-v1`
  and `LiquidAI/LFM2-ColBERT-350M`. Two image models were also measured:
  `Hcompany/NeoMME-260M-Retriever-ST-late` (with transformers 5.19.0) and
  `webAI-Official/webAI-ColVec1-4b` (its weights checked against the checkpoint's tensor names).
  51 of the 65 registry models are now measured; the catalogue holds 49 models and 4936 rows.
  `colbert-ir/colbertv2.0` and `GTE-ModernColBERT-v1` are pinned to the Hub commit that was
  measured.
- **lightonai/colbertv2.0's results were lost.** It shares the short name `colbertv2.0` with
  `colbert-ir/colbertv2.0` in the same batch, and the later run overwrote its folder. Only its
  selection study survived, and the ingest set it aside in `campaign/batches/11_text/orphans/`.
  The ingest now marks a model whose results are missing as incomplete, and keeps a
  selection-study file out of a folder whose run never made one. `measure_models.py` and
  `make_kaggle_batches.py` now refuse two models with the same short name in one batch.
- **Text runs now measure the registry's pin.** `encode_vectors.py` and `encode_text_vectors.py`
  pass the registry's revision to sentence-transformers. Batches up to now loaded the Hub's main
  branch, which matched the pin for every model measured. The loaded revision is recorded in the
  run info.
- **Split sentence-transformers models leave GPU 0 more room.** GPU 0's share of the weights is
  capped at 45% of its memory. `tencent/EVIE-8B` was split over the two T4s this time, but still
  ran GPU 0 out of memory in its forward pass.
- **Still unmeasured:**
  - `Hcompany/NeoMME-800M-Retriever-ST-late` ran out of GPU memory in attention at 4 pages per
    batch.
  - `LiquidAI/LFM2.5-ColBERT-350M` ran out of session time in its study.
  - `jinaai/jina-colbert-v2` was not reached.
  - `webAI-Official/webAI-ColVec1.1-8b` cannot run on T4s: in float16 it does not fit one, and
    split over two, its own forward multiplies tensors that sit on different GPUs.
- Four notebooks re-run the rest, side by side, because one would need more than an 11-hour
  session:
  - `batch_14a_text_retry` runs the three text models.
  - `batch_14b_neomme_retry` runs NeoMME-800M one page at a time.
  - `batch_14c_colvec_retry` runs `webAI-ColVec1.1-4b` in float32 over two T4s.
  - `batch_14d_evie_retry` runs EVIE-8B.

  `batch_13_retry.ipynb` is back to the list that ran.

### Kaggle measurement campaign, batch 14b_neomme_retry

- **`Hcompany/NeoMME-800M-Retriever-ST-late` measured.** It ran in float32 on one T4, one page per
  batch, with transformers 5.20.0, and the selection study completed. 52 of the 65 registry models
  are now measured; the catalogue holds 50 models and 5044 rows.

### Kaggle measurement campaign, batch 14a_text_retry

- **Three more text models measured on BEIR SciFact, each with its selection study:**
  `lightonai/colbertv2.0`, `LiquidAI/LFM2.5-ColBERT-350M` and `jinaai/jina-colbert-v2`. Each ran
  at its registry pin. 55 of the 65 registry models are now measured; the catalogue holds 53
  models and 5206 rows.
- `lightonai/colbertv2.0`'s float baseline (nDCG@5 0.692) differs from `colbert-ir/colbertv2.0`'s
  (0.646), although it is described as a conversion of the same model. Both ran through
  sentence-transformers, but a document gives about 236 vectors against about 157, so the two
  configurations truncate or tokenize documents differently. That was not investigated further.

### Kaggle measurement campaign, batch 14c_colvec_retry

- **`webAI-Official/webAI-ColVec1.1-4b` failed again, over two T4s in float32.** Its own forward
  (`modeling_colqwen35_bidirection.py`) multiplies the projection head's output, which
  `device_map="auto"` put on the last GPU, by the attention mask, which is on the first. The 8B
  model failed the same way in batch 13_retry.
- **The sentence-transformers adapter moves a split model's small heads to the input device.**
  Top-level modules holding at most 2% of the parameters go to the device the inputs are on,
  and accelerate's hook copies the hidden states across. `placement["heads_on_input_device"]`
  names the modules moved.
- Two notebooks re-run the ColVec1.1 models with the fix: `batch_15a_colvec4b_retry` (4B,
  float32) and `batch_15b_colvec8b_retry` (8B, float16). The 8B is no longer listed as
  unloadable.

### Kaggle measurement campaign, batches 15b_colvec8b_retry and 16_evie_retry

- **`webAI-Official/webAI-ColVec1.1-8b` measured** in float16 over two T4s, with its projection
  head on the input GPU. `check_retriever` passed with 10 of 12, the minimum.
- **`tencent/EVIE-8B` measured** in float16 over two T4s, at the 1,024-token page budget its card
  gives for its evaluation protocol (`processor_kwargs`, recorded in the registry notes).
- Both selection studies completed. 57 of the 65 registry models are now measured; the catalogue
  holds 55 models and 5492 rows.
- Plain int4 held on EVIE-8B's 4096-d vectors (retention 1.001 and 1.000), but collapsed on
  `nvidia/nemotron-colembed-vl-8b-v2`, which is also 4096-d (0.011 and 0.007). The collapse is not
  a matter of width alone.

### Kaggle measurement campaign, batch 15a_colvec4b_retry

- **Moving the projection head to the input GPU worked.** `webAI-Official/webAI-ColVec1.1-4b` ran
  over two T4s in float32 with `lm_head` and `embedding_proj_layer` on GPU 0.
- **It still failed `check_retriever`, with 9 of 12 queries, the same as in float16.** It was
  loaded as its card shows, so precision is ruled out as the cause. It stays unverified and is
  not re-run. Its card reports ViDoRe V3 only.

### Kaggle measurement campaign, batch 14d_evie_retry

- **`tencent/EVIE-8B` passed `check_retriever` (11 of 12) but ran out of memory in the encode.**
  Its processor allows 16,384 visual tokens a page, against about 1,000 for `EVIE-4.5B`. One
  DocVQA page of about 8500 tokens needed 4.4 GiB for a single attention call on GPU 0.
- **A batch can pass `processor_kwargs` to a sentence-transformers page model.**
  `measure_models.py` forwards it as `encode_vectors.py --processor-kwargs`, and the run info
  records it. `batch_16_evie_retry` uses the 1,024-token page budget that EVIE-8B's card gives
  for its evaluation protocol.

## 0.4.0 (2026-10-12)

Built on the `research-investigation` work (`91564fd`); this section also holds the batched
`ExactIndex.rescore` that the base already carried.

`optimize()`, `calibrate()`, the 40 names of v0.3.0's `optivision.__all__` (it has 52 now, 12
new), the CLI commands that existed, every old import path and the v1 file formats are
unchanged; `docs/package/MIGRATION.md` lists the observable differences. The additions are
engineering capability, tested on synthetic data and against the research code for numerical
equality; no registered experiment evaluated them as a package, and none of the research
results (`reports/research/`) was changed or re-run. The model coverage entries at the end of
this section add loading paths and three measured models; a loading path is not evidence that
compression works for that model.

### Certified selection and the model-agnostic core

- **`optivision.certify()` and `CertifiedResult`** (`optivision/api.py`): a second entry point
  beside `optimize()`. It scores a frozen candidate list on all the queries, lets a selection
  rule choose the smallest certified candidate or abstain, and returns the decision, a
  certificate, the evidence, and (unless it abstained) the fitted pipeline and the compressed
  corpus. Parameters: `target`, `error`, `metric`, `reference` (`auto`, `labels`, `agreement`
  or a reference object), `space`, `rule`, `query_budget`, `order_seed`, `fallback`, `encoder`,
  `progress`, `created`. `result.retriever()`, `result.save()` and `result.report()`;
  `optivision.verify()` checks a certificate or an artifact directory. Nothing certified gives
  `Outcome.ABSTAIN` and the uncompressed index; there is no default fallback. Design:
  `docs/package/API_DESIGN.md`.
- **New top-level names**, loaded on demand: `certify`, `verify`, `verify_certificate`,
  `load_artifact`, `save_artifact`, `CertifiedResult`, `Certificate`, `CandidateSpace`,
  `Decision`, `Outcome`, `EncoderIdentity`. `import optivision` also registers the `rotation`
  stage.
- **CLI**: `optivision certify VECTORS QUERIES` and `optivision verify PATH`, with `--json`.
  `certify` exits 0 for every outcome and 2 for invalid input; `verify` exits 0 when every check
  that ran passed, 1 when a check failed or the file is unreadable or malformed, and 2 for a usage
  error, a missing path included. `--allow-reduced` (the smaller candidate list, no SciPy) and
  `--allow-duplicate-queries`.
- **`optivision.core`**, a model-agnostic layer over the existing modules (which are not moved
  or edited): content identities and the vector contract (`identity`, `vectors`); stage traits
  that state which changes of coordinate basis leave a stage's scores unchanged, fitted-state
  save and restore, byte accounting and a seeded `RotationProjector` (`stages`); a retrieval
  seam with exact, compressed, tiered and two-tier retrievers (`retrieve`); references
  (`LabelReference`, `RankingReference`), `Evidence` and `collect_evidence` (`evaluate`);
  `CandidateSpace`, the rule (`FixedSequenceBetting`, `BonferroniBetting`), `Decision` and
  `select` (`selection`); `Certificate`, `create_certificate`, `verify_certificate`,
  `sign_certificate` (optional HMAC), `save_artifact`, `load_artifact` and `code_identity`
  (`artifact`). Specifications: `docs/package/CERTIFICATE_SPEC.md`, `METHOD_C_SPEC.md`.
- **`optivision.backends`**: a `Backend` protocol and wrappers over `NumpyIndex` and
  `QdrantIndex`, which are not changed, and the canonical import place for the indexes.
- **Adapters** (`optivision.adapters` is now a package; same imports): `ArrayAdapter`,
  `CallableAdapter`, `corpus_from_arrays` for bringing your own vectors; `NemotronAdapter`;
  `EncoderAdapter.identity()` and `.dimension`. `load_adapter` now returns a `NemotronAdapter`
  for the NVIDIA ColEmbed ids (it used to fall through to the sentence-transformers adapter).
- **Namespaces and helpers**: `optivision.ext.vision` and `optivision.legacy` name the page
  layer and the original recipe without moving code; `optivision.testing` holds `make_corpus`,
  `make_queries`, `random_orthogonal`, `signed_permutation`, `apply_basis`, `reorder_rows`,
  `basis_probe`, `check_adapter` and `check_stage`.
- **Tests and documents**: `tests/test_core_*`, `test_api_*`, `test_backends_*`,
  `test_adapters_*`, `test_cli_certify.py`, `test_integration_*`, `test_public_surface.py` and
  `test_golden_formats.py` (files written by v0.3.0 keep reading back); the user guide in
  `docs/package/guide/` with examples that the tests run; the design and audit documents in
  `docs/package/`.
- **`ExactIndex.rescore` is batched** (from `research-investigation`).
  - Each distinct candidate document is decoded once and scored against every query
    that shortlisted it, instead of re-decoding every query's candidates. The
    arithmetic (decode, float32 products, max, per-query sum), the column order and
    the API are unchanged.
  - On six ViDoRe datasets (500 pages, 50 candidates, all queries scored in one
    batch), rankings and per-query nDCG@5 were identical and the scores were equal on the
    stack used. In general batched and per-query rescoring give the same rankings except
    between scores closer than the float32 rounding of the dot products, and scores equal to
    within a few float32 ulps; bitwise equality holds on some BLAS kernels only (measured equal
    on an AVX-512 kernel, about one ulp different on an AVX2-class kernel). The same holds for a
    comparison with v0.3.0 scores.
  - Rescoring was 5.5-9.3x faster and the two-tier query 2.9-3.7x faster, on one
    laptop CPU. Evidence: `reports/engineering/batched_rescore/RESULTS.md`.
- **After the hardening pass** (an internal audit of the package before a possible release):
  - `certify(..., allow_duplicate_queries=False)`: exact duplicate queries among the tested
    queries now raise an error unless the caller passes `allow_duplicate_queries=True` (CLI
    `--allow-duplicate-queries`), which asserts that they are independent draws; the certificate
    records the assertion.
  - `certify` gives a `UserWarning` when it is called again in the same process on the same
    queries and corpus with a different target, error level,
    query order (seed or budget), candidate space, metric, reference or rule.
  - `certify` gives a `UserWarning` with the counts when some of the queries given have no
    relevant document under the reference and are therefore not tested (labels whose ids match
    only part of the queries used to shrink the sample without a word; the count was, and is, in
    the certificate). The error for labels that match no query now names the likely cause.
  - `verify` reports the highest level whose checks ran without a failure at that level or below
    (`binding` only with the corpus, the queries and the reference supplied; `none` when an
    integrity check failed) and lists the checks it skipped.
    `verify().ok` on an ABSTAIN or FALLBACK record means that the record is intact, not that an
    index was certified.
  - Loading an artifact: a malformed bundle file gives `ArtifactError` or
    `CertificateFormatError` instead of a raw exception, deeply nested JSON is refused cleanly, and
    `load_artifact(verify=True)` checks `dim` and `stats` of the codes header against the
    certificate and the pipeline.
  - Adapters make their array items C-contiguous, so the memory layout of the input does not change
    the certificate digest. (A corpus built directly from Fortran-ordered arrays can still give
    last-bit different norms and a different digest.)
  - `certify` refuses `qrels` of the wrong type and names the type; a built-in rule instance that
    carries instance attributes is not treated as built in; note strings were parenthesised.
  - `from optivision.storage import segment_reduce` works again, as in v0.3.0. The rescoring tests
    compare batched and per-query scores within a derived float32 bound instead of bitwise.
    The retrieval test of one query alone against the same query in a batch uses the same bound
    (the two were equal bit for bit on one BLAS kernel only).
  - Documentation: 51 names in `__all__` (the 40 of v0.3.0 plus 11), the `int4(mean)` label, the
    scope of the platforms that were tested, the `certify` dependency on SciPy, how to write the
    files the command line reads, and the four expected failures in the pinned `compose.py`
    (`docs/package/TEST_REPORT.md`).
  - `docs/package/RELEASE_READINESS_REPORT.md`: the release-candidate gate, the disposition of
    every finding and the steps a release still needs; the audit reports are in
    `docs/package/hardening/`.
- **After the internal reviews** (`docs/package/RESEARCH_INTEGRITY_AUDIT.md`,
  `docs/package/STATISTICAL_REVIEW.md`; internal, not external):
  - *Certified only for the library's own rules.* A decision and a certificate record the rule's
    class and whether it is one of the library's own (the exact types `FixedSequenceBetting` and
    `BonferroniBetting`). A rule written by anyone else is recorded with what it declared, its
    assumptions are labelled as its author's, its selection is never `certified`, and the
    certificate carries a fixed text saying the library has not examined it instead of the
    library's statement. `verify` fails a record that says `certified: true` for a rule it does
    not know, instead of skipping the level, decision, text and replay checks, and the report
    has `notes`.
  - *The statement and the assumptions were reworded*: the stored statement says that it is a
    joint probability and not the probability that a selection is below the target, not about the
    retention on the sample, not about a finite batch of future queries, not about power, and that
    A1-A6 and C1-C4 are sufficient conditions. A1, A3 and A5 are stricter, and four conditions of
    use were added (one look, same queries and metric, a reference fixed independently of the
    candidates, deployed equals evaluated): ten sentences in all.
  - *`certify()` and `optivision certify` test the queries in a seeded random order by default*
    (`order_seed=0`; `--order-seed`, and `--given-order` for the order of the file).
    `select()` keeps the order given. The certificate says what a given order requires and when
    the seed was chosen by the caller.
  - *Exact duplicate queries are counted* (`queries.exact_duplicates`,
    `report()["n_exact_duplicate_queries"]`, a caveat; see the hardening entry below for
    `allow_duplicate_queries`). *Vector norms are recorded* (new top-level field `norms`; a caveat when the
    vectors are not unit-norm). The certificate has 25 fields; `verify` recounts both from the
    live queries and vectors.
  - Wording: the headline of the command line no longer says "certified" on its own, the abstain
    note no longer suggests a lower target on the same queries, "conditional on" no longer sits
    beside a probability, and the guide and the specification say what each assumption asks of
    you.

### Model coverage

- **The model registry grew from 12 to 65 late-interaction models** (`optivision.adapters.MODELS`,
  `supported_models()`): 10 `measured`, 23 `untested`, 32 `unverified`; 50 take page images and 15
  take text. `measured` means retrieval quality with and without compression was measured by this
  repository on the model's real output; `untested` means the loading code is the one a measured
  sibling uses and the checkpoint was never encoded here; `unverified` means the loader is written
  against the model card and was never run. A status describes the loading path, not how well a
  pipeline compresses that model's vectors. `ModelInfo` has three optional fields, `license`,
  `revision` and `notes`. Page: `docs/package/guide/MODELS.md` (generated by
  `scripts/build_models_doc.py`).
- **Three models became `measured`** from Kaggle runs committed under `reports/universal/wide/`:
  `nvidia/nemotron-colembed-vl-8b-v2` (float16 inference), `athrael-soju/colqwen3.5-4.5B-v3` and
  `TomoroAI/tomoro-colqwen3-embed-4b`. These are development-style, in-sample measurements on
  ViDoRe V1 DocVQA and InfoVQA (500 pages each); they were not run under the selection rule of
  `certify` and are not a statement about another corpus.
- **`nomic-ai/colnomic-embed-multimodal-7b` changed loader**: from `sentence-transformers`
  (`unverified`) to `colpali-engine:colqwen2.5` (`untested`).
- **`load_adapter(model_id, loader=None, **kwargs)`**: the new `loader` argument builds an adapter
  for a checkpoint that is not in the registry. New loader families: `colpali-engine:<family>`
  (a token-vector path for ColQwen3, ColQwen3.5, ColModernVBert, ColGemma3 and ColQwen2.5-Omni),
  `remote-code:<recipe>` (needs `trust_remote_code=True` from the caller; the revision is pinned)
  and `pylate`. `supported_loaders()` lists them.
- **New modules**: `optivision.encoders.late` (`ColEngineTokenEncoder`, `RemoteCodeEncoder`,
  `Recipe`, `RECIPES`, `PyLateEncoder`); `optivision.adapters.tokens` (`TokenEncoderAdapter`,
  `check_retriever`, which encodes matched pairs and reports whether an adapter behaves like a
  retriever; it shows the model was driven correctly, not that compression works);
  `optivision.adapters.measured` (`measured_models`, `measured_results`, `measured_shortlist`)
  over the shipped catalogue `optivision/presets/measured.json`, built by
  `scripts/build_measured_catalog.py` from the committed reports.
- **`optivision.encode_and_certify(model, documents, queries, qrels, *, document_ids, query_ids,
  adapter_options, **certify_options)`**: encodes with a model, then calls `certify`. It is the
  twelfth new name in `optivision.__all__` (52 in all); `check_retriever`, `TokenEncoderAdapter` and
  the `measured_*` functions are imported from `optivision.adapters`.
- **CLI**: `optivision models` (`optivision/cli_models.py`) lists the registry and the measured
  catalogue. The CLI has 16 commands (the 13 of v0.3.0, plus `certify`, `verify` and `models`).
- **`scripts/encode_vectors.py`** gained `--loader`, `--trust-remote-code`, `--revision` and
  `--check-pairs`.
- The certificate records package version 0.4.0.

## 0.3.0 (2026-10-02)

The measured model set grows from three to six encoders, one of them
2,560 dimensions wide. Evidence and caveats are in `docs/UNIVERSAL.md` (R11, R12).

### Measured

- **ColQwen2-v1.0 and ColQwen2.5-v0.2 (R11)** on ViDoRe V1 DocVQA and InfoVQA
  (500 pages, 451 / 494 queries each). Their float baselines are within 0.5–1.8
  points of the published scores. Out of sample, `calibrate()` met its target in
  236 of 240 choices.
- **NVIDIA Nemotron ColEmbed 4B, 2,560-d (R12)** on the same two splits, float32
  on two Kaggle T4s, pinned revision `0ed152d9`. The model is licensed
  CC-BY-NC-4.0 and was evaluated for non-commercial research only.
  - Baseline nDCG@5 0.6645 / 0.9344 (published: 67.39 / 93.31).
  - Fixed configurations: Ward merging to a quarter keeps 98.8% / 99.5%; PCA to 320
    dims 99.1% / 99.4%; int8, centred int4 and binary 98.9–100.2%. Plain
    (uncentred) int4 collapses (0.4–0.9%); the cause is unresolved, and the default
    search space never uses it.
  - Out of sample over 20 splits per cell, `calibrate()` met its target in 119 of
    120 choices with the default space and 118 of 120 with the opt-in PCA families.
    These are empirical counts, not a guarantee or a success probability.
  - The best compression ratios R12 reports (for example 16x, 63x and 387x) are
    read off an in-sample Pareto frontier, so they are optimistic.

OptiVision has now been evaluated empirically on a 2,560-dimensional
multi-vector retrieval model, and its model-agnostic compression and optimization
machinery kept working at that width. Not measured: any 4,096-d model (the
`nemotron-colembed-vl-8b-v2` loader exists but was not benchmarked), ViDoRe
V2/V3, and corpora beyond 500 pages.

### Added

- `optivision.encoders.nemotron.NemotronColEmbedEncoder` for NVIDIA ColEmbed 4B /
  8B v2. It requires a pinned revision of the remote code and checks the
  architecture, pooling and width.
- `calibration.projection_search_space` and `wide_search_space`: opt-in PCA
  families. The default search space is unchanged.
- `OPTIVISION_SCORE_DEVICE=cuda` runs the query x document products on a GPU.
- `scripts/wide_smoke.py`, `scripts/colqwen_smoke.py`,
  `scripts/encode_vectors.py --backend nemotron`,
  `scripts/universal_study/s13_wide_study.py`, `s7b_selection_rules.py --space`.
- Kaggle notebooks `notebooks/kaggle_colqwen_encode.ipynb` and
  `notebooks/kaggle_wide_model.ipynb`, and instructions in `docs/GPU_RUN.md`.

### Fixed

- Checkpoint loads that would leave weights randomly initialised are now
  refused. The ColQwen key mapping was corrected.
- Multi-GPU placement splits only between whole decoder layers and handles
  nested decoder stacks.
- Fitting samples (PCA, centring, per-dimension int8) are capped in bytes as well
  as rows. At 2,560–4,096-d this cuts them from 4–7 GB to about 256 MB; they are
  unchanged up to 160-d.
- The calibration transform cache has a byte budget with LRU eviction
  (`OPTIVISION_CACHE_BYTES`).
- Ward merging at 512-d and wider uses a matrix product for distances: about 17x
  faster at 2,560-d, with identical clusterings.
- Configurations with identical stored size are tie-broken deterministically, not
  by measured latency.
- `Pipeline.label()` names non-PCA projections, for example
  `project(640, random)`.

## 0.2.0 (2026-10-01)

The universal multi-vector compression core: composable token / dimension /
codec stages, `calibrate()` and `optimize()`, and measurements on ColPali,
ColSmol and a text ColBERT. See the GitHub release notes and
`docs/RELEASE-AUDIT-2026-09-27.md`.
