# Retrieval and backend audit

Read-only audit of `src/optivision` ahead of the retrieval seam (queries -> candidate index -> ranked ids) and the backend interface.
Worktree `optivision-package`, branch `package-universal`, HEAD `91564fd`, package 0.3.0, 2026-10-06. No existing file was modified; this is the only file added.

Labels: **READ** (from code, cited `file:line`), **MEASURED-HERE** (throwaway snippets on synthetic arrays, Appendix A; Python 3.11.9, numpy 2.4.6,
qdrant-client 1.19.0, torch 2.11.0+cpu, Windows 11), **REPORT** (a committed repository report, named), **INFERENCE** (our reading). Paths are relative to the worktree root.

## 0. Findings that shape the design

1. **`ExactIndex` and `TieredIndex` already are the seam.** `storage.py` offers `score(queries)` (dense float32), `search(queries, k) -> (indices, scores)` and, on
   `ExactIndex`, `rescore(queries, candidates)` (`storage.py:50-102,116-142`). Missing: a typed result with ids, protocol names, `k` validation, a tier-alignment check.
2. **Two-tier retrieval is in the library, but nothing in the library calls it.** `TieredIndex` is imported only by `scripts/` and `tests/`; `calibrate`, `optimize`,
   `benchmark.run_matrix`, the CLI and `OptiVisionRAG` are single-tier (section 4).
3. **Ties are broken three ways.** `rank()` is stable by position (`scoring.py:183`); `NumpyIndex.search` uses `argpartition` plus a default `argsort`
   (`numpy_index.py:229-230`); Qdrant local mode reverses a default `argsort` (qdrant-client `local_collection.py:711`). MEASURED-HERE: with 200 to 5,000 exactly tied
   pages `NumpyIndex.search` returned different members and order than `rank()` in 4 of 4 cases.
4. **The backends do not compute the same score.** NumPy: inner-product MaxSim (`scoring.py:3`). Qdrant: `Distance.COSINE` MaxSim (`index/qdrant_index.py:101`).
   MEASURED-HERE (30 queries, 40 pages, d=32): identical full rankings for unit-norm query tokens with `method` `"binary"` or `"none"` (Qdrant score x sqrt(d) equals the
   NumPy score for binary); 30 of 30 rankings differ for mixed-norm query tokens; with `"int8"` (separate 60-query run) top-1 agreed 60/60 but the top-5 set only 55/60
   (scores differ by up to 1.2%).
5. **Qdrant receives decoded codes, not float originals, unless `method="none"`** (`qdrant_index.py:155-157`); the module docstring (`:10-14`) says otherwise. Server-side
   rescoring is whatever Qdrant decides: no `search_params` are passed.
6. **Empty documents have three conventions:** `-inf` (`maxsim_matrix`, `maxsim_pair`, `rescore`, mixed `NumpyIndex`), `0.0` (legacy `compression.maxsim_*`, test-pinned; and
   `NumpyIndex` when every page is empty), and a crash (`QdrantIndex.add` raised `IndexError`, local mode).
7. **Equality is rounding-level, never bitwise.** MEASURED-HERE: `max_block_bytes` changes bits; `NumpyIndex.score_all` vs `maxsim_matrix` differ by up to 1.9e-6 and 1 of
   40 queries had a different full ranking (one swap at rank 111, gap 9.5e-7). REPORT: Q5/Q6 found 45 float32 near-tie swaps.
8. **`OPTIVISION_SCORE_DEVICE` reaches only `maxsim_matrix`.** `ExactIndex.rescore`, `NumpyIndex` and the legacy helpers are CPU numpy.
9. **The block budget bounds the decode block and similarity block, not the query chunk, the dense output matrix, or `rescore`'s query materialisation** (section 8).
10. **Q5/Q6 is a statement about one software stack, and several repository texts already contradict or over-extend it** (section 5.3). The seam must keep decode-first
    scoring as the reference semantics and carry no speed claim about compressed codes.

## 1. MaxSim implementations and exact semantics

### 1.1 Inventory

| # | implementation | role and callers |
|---|---|---|
| M1 | `scoring.maxsim_matrix` `scoring.py:99-168` | the reference; batch queries x documents -> float32 `[n_q, n_d]`. Callers: `evaluation.measure` (`evaluation.py:205`), `calibration.score_candidate` (`calibration.py:140`), `calibrate` baseline (`calibration.py:461`), `ExactIndex.score` (`storage.py:51`) |
| M2 | `scoring.maxsim_pair` `scoring.py:189-197` | textbook single pair; the oracle in `tests/test_scoring.py:17-18` |
| M3 | `ExactIndex.rescore` `storage.py:58-102` | exact MaxSim of each query against its own candidate documents only; CPU numpy |
| M4 | `NumpyIndex.score_all` / `_segment_maxsim` `index/numpy_index.py:156-222` | legacy one-query path -> float32 `[n_pages]`; callers `NumpyIndex.search`, `bench.run_variant` (`bench.py:323`) |
| M5 | `compression/binary.py:61-116`, `compression/lloyd2.py:203-229` | per-page helpers `maxsim_asymmetric(_batch)`, `maxsim_float`, `maxsim_hamming`, `maxsim_asymmetric_lloyd2(_batch)`. In `src` only `maxsim_float` is used (`pipeline.py:273,282`); `maxsim_hamming` is exported, untested, unused |
| M6 | torch branch of M1 `scoring.py:80-96,150,158` | query x document products on `OPTIVISION_SCORE_DEVICE` |
| M7 | Qdrant `MAX_SIM` `index/qdrant_index.py:97-108,196-219` | engine side, cosine |
| M8 | `pipeline._rerank` `pipeline.py:271-286` | float-cache rescoring of prefilter hits, one query at a time |
| M9 | scripts only: `scripts/research/q56_common.py:45-134` (`Int8Direct`, `BinaryLUT`, `BinaryHamming`, `maxsim`), `scripts/review/q11_margin_rescoring.py:55-56` | experiments, not library |

### 1.2 Reference semantics (M1; READ unless marked)

- **Score:** `sum_i max_j <q_i, d_j>` over the rows of one query and one document: raw float32 inner product; no cosine, no normalisation, no mask, no weights (`scoring.py:3-4,161-164`).
- **Order of work per block:** `sims = qv @ block.T` (`:92`; torch `:96`) -> `segment_reduce(sims, local_d, np.maximum, axis=1, fill=-inf)` per document (`:161`) ->
  `segment_reduce(per_token, local_q, np.add, axis=0, fill=0.0)` per query (`:164`). `segment_reduce` reduces non-empty segments only, so an empty segment cannot truncate its
  neighbour (`:56-77`; regressions `tests/test_scoring.py:83-92`, `tests/test_index.py:209-223`). The per-query sum is a sequential float32 `reduceat`.
- **Empties:** empty document -> `-inf` for every query (`:155,167`; docstring `:115-116`); empty query -> `0` against non-empty documents (`:135-137,164`); an empty
  document stays `-inf` even for an empty query (`:167`). Pinned by `tests/test_scoring.py:30-41`.
- **Output and NaN:** `np.zeros((n_q, n_d), float32)` (`:126`). `maxsim_matrix` never checks for NaN: one NaN query token gives an all-NaN row, and `rank` then returns `[0, 1, 2, ...]`, a silent
  index-order "result" (MEASURED-HERE). Non-finite input is rejected in `Pipeline.transform_queries` (`compose.py:36-45,177-183`), not in the scorer.
- **Store contract:** any object with `offsets`, `__len__` and `decode_rows(lo, hi) -> float32` (`scoring.py:26-31`). Queries are a store too (`:107-108`).
- **Asymmetric and decode-first:** documents are decoded block by block to float32 and multiplied by float32 queries (`stages/quantize.py:3-5`). Decodes: float32 view (`:47-48`);
  float16 widened (`:67-68`); binary bit=1 iff value > 0 -> +1 else -1, so an exact zero maps to -1 (`compression/binary.py:31-54`, `quantize.py:128-137`); fixed int8
  `code * 0.5 / 127` (`compression/__init__.py:73,143-148`); per-vector int8 `code * fp16_scale / 127` (`quantize.py:217-230`); int4 (`:288-300`); lloyd2
  `levels[idx] @ R.T + mu` (`lloyd2.py:194-200`). The float32 rounding of the decode multiplier is part of the score (`reports/research/Q5Q6/CORRECTNESS.md:54-72`).

### 1.3 Variants and their conventions

| variant | formula | empty document | empty query | notes |
|---|---|---|---|---|
| M1 / M3 | `sum_i max_j q_i.d_j`, float32 | `-inf` | 0 | `rescore` keeps column order and duplicates (`tests/test_storage.py:131-138`) |
| M4 | same; `per_token_max.sum(axis=0)` (`numpy_index.py:170-171`) | `-inf` (`:172`); `0` for all pages if the index holds no vectors (`:188-189`; MEASURED-HERE `[0, 0]`) | 0 | not bitwise equal to M1 |
| M5 `maxsim_asymmetric` | same against +/-1 codes | `0.0` (`binary.py:67-68`) | 0.0 | `_batch` leaves empty pages at 0.0 (`:89-93`) |
| M5 `maxsim_float` | same, float docs | `0.0` (`:99-100`; pinned `tests/test_compression.py:68`) | 0.0 | used by `_rerank` |
| M5 `maxsim_hamming` | `sum_i max_j (s_qi . s_dj)/dim`, both sides unpacked to +/-1 (`:111-116`) | `0.0` | 0.0 | docstring says popcount; the code is a float GEMM |
| M7 Qdrant | `sum_i max_j cos(q_i, d_j)` | `add` crashes (MEASURED-HERE) | untested | both sides L2-normalised by the engine (local: `local/distances.py:121-135`, upsert `local_collection.py:2487-2492`) |
| M8 `_rerank` | `maxsim_float(query, cached_doc)` | `0.0` | - | uncached hits keep compressed-scale scores and are sorted with float-scale scores (`pipeline.py:276-286`) |

### 1.4 What the scorer does not have

- **Masking and padding:** none. Encoders drop padded tokens before returning vectors (`encoders/nemotron.py:93-105`, `encoders/colvlm.py:543-549,601-606`); lengths live in
  `offsets`. MEASURED-HERE: a zero row appended to a query is neutral (adds 0); a zero row appended to a document is not (it lifts a negative best match to 0 and flipped a ranking).
- **Query-token weights:** none anywhere in `src`.
- **Merged-token weights:** `MultiVectorCorpus.weights` ("how many original vectors a row stands for", `representation.py:11-12`) is read only by token reducers and pooling
  (`stages/base.py:149-183`, `stages/merge.py:48-60,200,215,270`); it never reaches `maxsim_matrix`, a codec or an index, and `save_compressed` does not store it
  (`compose.py:268-282`). A merged row counts once; with a max over document rows, multiplicity is irrelevant (MEASURED-HERE: 1 vs 3 copies, identical scores).
- **Normalisation:** the scorer assumes none. The pipeline assumes L2-normalised encoder output (`encoders/base.py:17-18`); `Int8Quantizer(scale="fixed")` assumes about
  unit-norm 128-d and warns on clipping (`quantize.py:199-206`); merges rescale pooled rows to the members' weighted mean norm (`merge.py:29-31,48-60`); binary decode has norm
  sqrt(d), so absolute scores compare only within one codec; PCA is uncentred by default and queries are never centred (`project.py:107-112`).

### 1.5 dtype, device, environment and size limits

| path | behaviour |
|---|---|
| M1 CPU | query rows `np.ascontiguousarray(..., float32)` (`scoring.py:146`): float64 queries are silently narrowed (MEASURED-HERE, error vs float64 reference 4.7e-8). Documents `decode_rows` -> float32 (`representation.py:290`, `compose.py:86-87`); float16 widened exactly; float32 BLAS, float32 output |
| M1 torch | `OPTIVISION_SCORE_DEVICE` read on every call (`:124`; empty = unset). Query chunk `.to(device)` once (`:87,150`); each document block `.to(device)` and each similarity block `.cpu().numpy()` (`:96`); max and sum stay in numpy. Sets `allow_tf32=False` for CUDA matmul and cudnn globally (`:85-86`). Docstring promises ~1e-6 (`:119-122`); the test allows `atol=1e-4` and only runs device `"cpu"` (`tests/test_wide_vectors.py:52-64`) |
| M4 | `np.ascontiguousarray(query, float32)` accepts any dtype including integers (`numpy_index.py:191`), whereas `MultiVectorCorpus` rejects integers (`representation.py:66-68`); CPU only, no env var |
| M3 | CPU numpy; ignores `OPTIVISION_SCORE_DEVICE`; candidates cast to int64 (`storage.py:70`), so float ids pass silently; all query tokens held as float32 (`:78`) |
| M7 | query sent as Python lists (`qdrant_index.py:197`) |

Limits: `DEFAULT_MAX_BLOCK_BYTES` 256 MiB (`scoring.py:23`; also `ExactIndex` `storage.py:42`, `measure` `evaluation.py:194`). Query chunk `max(1, min(T, max_block_bytes // 16384))`
tokens (`scoring.py:141`, 16,384 at the default); document block `max(1, max_block_bytes // (4*T_chunk + 4*D + 1))` rows, page-aligned, never splitting a page (`:151`, `:34-49`);
rescoring step `max_block_bytes // (4*block_rows)` tokens per product (`storage.py:96`). `NumpyIndex.DEFAULT_MAX_DECODED_BYTES` 256 MiB: cache below, stream above, with
`bytes_per_vector = 5*dim + 4*n_q` (`numpy_index.py:32,193-204`). `OPTIVISION_CACHE_BYTES` 4 GiB LRU of transformed corpora for calibration (`calibration.py:98-110`; not retrieval).
`ENCODE_BLOCK_ROWS` 65,536 (`compose.py:33`); fit samples 200,000 rows or 256 MiB (`stages/base.py:78-79`); `lru_cache(maxsize=8)` on lloyd2 rotations (`lloyd2.py:110`).

### 1.6 Tie-breaking

| path | rule |
|---|---|
| `rank()` `scoring.py:171-186` | `np.argsort(-scores, kind="stable")`: descending, ties by ascending position; NaN last (MEASURED-HERE) |
| `ExactIndex.search`, `TieredIndex.search`, `per_query_metrics`, `relevant_from_baseline`, `measure` | through `rank()`: stable by position |
| `TieredIndex` shortlist and final | shortlist `rank(hot, m)` (`storage.py:127`); final ties among shortlisted documents break by document position, not shortlist order (MEASURED-HERE, I5) |
| `NumpyIndex.search` `numpy_index.py:224-234` | `argpartition(-s, k-1)[:k]` then `argsort(-s[top])`: exact top-k set up to ties; tie members and order unspecified (MEASURED-HERE) |
| `QdrantIndex.search` | engine order; local mode `np.argsort(scores)[::-1]` (`local_collection.py:711`); MEASURED-HERE 10 tied pages came back `d8, d9, d6, d7, d3, ...` |
| `pipeline._rerank` `pipeline.py:285` | Python stable sort on `-score`: ties keep input order |
| `bench.run_variant` `bench.py:325-330` | top-k by `argpartition`; `deep` ranking by default `np.argsort(-scores)` (unstable) |
| `scripts/review/q11_margin_rescoring.py:69,77-80` | default `np.argsort`; `np.sign(0) = 0`, unlike the codec's `-1` |

## 2. Index interface and the two backends

### 2.1 `BaseIndex` and the factory

`index/base.py:13-39`: `add(pages: Sequence[CompressedPage])`, `search(query: ndarray [n_tokens, dim], top_k=5) -> list[SearchHit]`, `save()`, `stats() -> dict`, `n_pages`, `close()`,
a context manager, class attribute `backend`. One query per call; `SearchHit(ref, score, rank, stage)` (`types.py:132-137`). No batch search, no dense scores, no candidate rescoring,
no positional ids. `NumpyIndex` adds `score_all`, `refs`, `load`, `decoded_nbytes` outside the ABC and `bench.py:317-323` uses them, so core code already knows its backend.
`index/__init__.py:15-46`: `numpy` -> `NumpyIndex.load(path)` when `meta.json` exists and `recreate` is false, ignoring the `dim`, `method`, `codec` arguments (`:29-30`);
`qdrant` -> `QdrantIndex(...)` without `mode`, so `server_quantized` always applies (`:35-45`).

### 2.2 `NumpyIndex` (`index/numpy_index.py`)

- **Stores** `codes` uint8 `[total_vectors, nbytes]`, `offsets` int64 `[n_pages+1]`, `refs`, `page_stats`, optional `Lloyd2Codec` (`:52-57`). `method` selects `compression.decode`
  (`compression/__init__.py:151-170`): `none` float32 bytes (`4*dim`), `binary` sign bytes (`ceil(dim/8)`), `int8` fixed-scale, `lloyd2` 2-bit. Not storable: float16, int4,
  per-vector or per-dimension int8, centred codecs (`to_legacy_pages` raises, `storage.py:189-192`).
- **Exact:** brute force over decoded codes; the only approximation is quantisation. Cached path when `decoded_nbytes <= max_decoded_bytes` (`:193-194`), streaming blocks
  otherwise (`:196-222`; `tests/test_index.py:121-167`).
- **`add`** appends without dedupe (MEASURED-HERE: same page twice -> `n_pages == 2`) and re-concatenates the whole code array every call (`:115-118`): ingest in batches is
  quadratic. A width mismatch raises `ValueError` before any state changes (MEASURED-HERE).
- **Persistence** (`:89-108`): `<path>/codes.npy` (uint8), `offsets.npy` (int64), `meta.json` `{backend, dim, method, n_pages, refs[], page_stats[], codec{mu, sigma, seed} or null}`;
  `load` reads full arrays, no mmap (`:83-85`). `OptiVisionRAG` adds `manifest.json` (`pipeline.py:201-206`). `stats` (`:246-269`) reports `codes.nbytes` (+ codec overhead).

### 2.3 `QdrantIndex` (`index/qdrant_index.py`)

- **Layout:** one point per page, id `uuid5(namespace, page_id)` so re-adding overwrites (`:39,168`; `tests/test_qdrant.py:54-59`); payload `page_id, doc_id, page_no, source_path,
  image_path`, token counts, `nbytes`, `raw_nbytes` (`:170-181`).
- **Collection** (`:97-108`): `VectorParams(size=dim, distance=COSINE, on_disk=on_disk, multivector_config=MAX_SIM, quantization_config=BinaryQuantization(always_ram=True))` when
  `mode == "server_quantized"`; `hnsw_config`, `optimizers_config` and search params are never set. `always_ram` is deprecated in qdrant-client 1.19.0 (`memory` replaces it,
  `http/models/models.py:99`) while `pyproject.toml:35` allows `qdrant-client>=1.10`. `client_binary` is unreachable through `get_index`.
- **What is sent** (`:155-157`): `decode(page.codes, dim, method, codec)` -> float32 -> `.tolist()`: +/-1 floats for binary, decoded floats for int8, original values only for
  `none`. Local mode also L2-normalises stored vectors on upsert (MEASURED-HERE: float32, unit-norm rows).
- **Search** (`:196-219`): `query_points(query=list-of-lists, limit=top_k, with_payload=True)`; `quantization.rescore` stays `None` ("qdrant decides automatically", client docs
  `models.py:2424-2427`); no `oversampling`, `hnsw_ef` or `exact`. Score = `point.score`, the sum of per-token max cosines.
- **Exact vs approximate:** server = HNSW + binary quantisation + engine-decided rescoring: approximate and never measured here (`docs/RESULTS.md:25-29`, `README.md:495-496`).
  Local path mode (what the tests use) = brute force; `search_params` ignored with a warning (`qdrant_local.py:410-414`); the quantisation config is stored but inert; more than
  20,000 points warns (`qdrant_local.py:69,145-153`).
- **Persistence:** server engine-managed. Local: `<path>/.lock`, `meta.json`, `collection/<name>/storage.sqlite` (MEASURED-HERE). `save()` is a no-op (`:223-225`); `stats()` is
  client-side pipeline accounting re-hydrated from payloads on reopen (`:110-151`), not the engine's footprint (`docs/AUDIT-2026-09-27.md:63-65`).
- **Other:** `n_pages` is `count(exact=True)` (`:235-237`), called once per query by `OptiVisionRAG.search_many` (`pipeline.py:235`); `recreate=True` deletes the collection at
  construction (`:84-87`) and `OptiVisionRAG.build` defaults to it (`pipeline.py:118`); `add` with a zero-vector page raised `IndexError` (MEASURED-HERE).

### 2.4 Side by side

| property | `NumpyIndex` | `QdrantIndex` local | `QdrantIndex` server |
|---|---|---|---|
| score | inner-product MaxSim | cosine MaxSim, float32 stored | cosine MaxSim in engine |
| exact | yes, over decoded codes | yes (brute force) | no (HNSW, quantisation) |
| ties | unspecified | unspecified | engine |
| stored form | uint8 codes (`none`, `binary`, `int8`, `lloyd2`) | decoded float32, unit-normalised | decoded floats + binary quantisation in RAM |
| dense scores | `score_all` | no | no |
| re-add same page | appends a duplicate | overwrites | overwrites |
| empty page | `-inf` in a mixed index | `IndexError` | untested |
| rescoring | none (caller reranks) | none | engine decides |
| `recreate` | overwrites at `save()` | deletes immediately | deletes immediately |
| beyond 256 MiB decoded | re-decodes the whole index per query | brute force | engine |

### 2.5 Other storage surfaces

- `CompressedCorpus` `.npz` (`compose.py:257-297`): `codes` uint8, `offsets` int64, `header` JSON bytes `{format "optivision-compressed", version 1, dim, ids, stats, pipeline,
  quantizer}`, `allow_pickle=False`. Refuses quantizers with shared state (centred codecs, per-dimension int8, lloyd2: `:264-265`). The `pipeline` entry holds stage parameters only
  (`compose.py:230-231`), so fitted PCA bases are lost and a reloaded projected store cannot be queried without refitting (`reports/research/Q4/REPORT.md` section 9, Q10 section 9).
- `MultiVectorCorpus` `.npz` (`representation.py:496-539`); `load(mmap=True)` does not memory-map: npz members are read in full (MEASURED-HERE: `type(vectors)` is `ndarray`).
- `StorageBackend` / `NpzStorage` (`storage.py:148-174`) write, read and list `CompressedCorpus` files.
- `to_legacy_pages` (`storage.py:177-205`) is the only bridge to the legacy backends (float32, plain binary, fixed int8). Ids are split on `"::p"` and rebuilt as
  `PageRef(doc, page_no)` (`:197-199`): `"abc"` returns as `"abc::p1"`, `"a::pb"` as `"a::p1"`, so id-based backend results need a normalised id convention or position mapping.

## 3. Evaluation path

1. **Scores:** `maxsim_matrix(pipeline.transform_queries(queries), compressed, max_block_bytes)` in `measure` (`evaluation.py:205`) and `score_candidate` (`calibration.py:140`).
   The legacy bench scores one query at a time with `NumpyIndex.score_all` (`bench.py:321-327`).
2. **Ranking:** `rank(scores, k)`; `per_query_metrics` ranks only the top `kmax`, the largest `k` among the requested metrics (`evaluation.py:60-61`; the defaults give 10, `:29`).
3. **Per-query metrics** (`evaluation.py:55-82`, binary gain): `hits = isin(top[q], relevant)`; nDCG@k = `sum_{i<=k} hit_i / log2(i+1)` over the ideal sum for `min(|rel|, k)` ranks;
   recall@k = `hits@k / min(|rel|, k)` (not hits over `|rel|`); hit@k; mrr@k. A query with no relevant document is NaN, skipped by `mean_metrics` (`:85-86`) and by `retention`'s
   finite mask (`:93`). The id-based twin `metrics.ndcg_at_k`, `recall_at_k`, `evaluate` (`metrics.py:15-39,130-157`) is asserted equal on random data
   (`tests/test_evaluation.py:28-39`); it differs only for queries with no relevant document (0 vs NaN).
4. **Relevance:** labels via `relevant_from_qrels`, which maps ids to corpus positions and silently drops ids not in the corpus (`:39-46`); label-free via
   `relevant_from_baseline(baseline_scores, depth)`, the uncompressed ranking's own top `depth` positions (`:49-52`; ties by index).
5. **Uncompressed reference:** exact float32 MaxSim over the original, untransformed corpus: `maxsim_matrix(queries, corpus)` (`calibration.py:461`) or `measure(Pipeline(), ...)`
   (`benchmark.py:145`). An empty `Pipeline` stores float32 bytes (`compose.py:144-146`) that decode to the same values, so both agree. Computed once per run; it is both the
   retention denominator and the label-free reference.
6. **Retention:** `retention(c, b)` = mean(c) / mean(b) over queries finite in both, with a paired percentile bootstrap, 1,000 resamples, seed 0 (`:89-106`); `retention_lower_bound`
   (`:109-135`); `split_queries` (`:138-145`).
7. **What the selection layer consumes:** `ScoredCandidate.scores`, float32 `[n_queries, n_docs]` per candidate (`calibration.py:76-93`) -> `_judge` -> `per_query_metrics(sc.scores[idx], ...)`
   (`:193`); `select`, `_holdout` (`:235,250-251`); `Measurement.per_query` (`evaluation.py:148-159`). `query_ms` is scoring time over the queries of one batch
   (`evaluation.py:170-172`): batch-amortised, not a per-query latency. Only the top-`kmax` of each row is ever read, so the seam needs `search(queries, kmax)` for metrics and a
   dense `score` only where the matrix is kept for re-selection. Full rankings are also read by `metrics.rank_correlation` (`metrics.py:42-94`) and the bench's `deep` lists (`bench.py:330`).

## 4. Two-tier retrieval today

**Library, framework path:** `src/optivision/storage.py`. `ExactIndex` (`:37-102`) and `TieredIndex` (`:105-142`) are imported by `scripts/` and `tests/` (grep) and by no library entry
point; the default search spaces are single-tier (`calibration.py:332-361`). The only other two-stage code is the legacy float-cache rerank in `pipeline.py` (below).

**`TieredIndex(hot, cold, candidates=50)`** (`storage.py:116-142`):
1. `hot = hot_index.score(queries)`: dense `[n_q, n_docs]`, an exhaustive scan of the hot store (a RAM saving, not fewer documents scanned).
2. `m = min(candidates, n_docs)`; `shortlist = rank(hot, m)` (stable).
3. `cold = cold_index.rescore(queries, shortlist)`: exact decode-first MaxSim of each query against its own `m` documents, batched per distinct document (`:58-102`, commit `198f83a`;
   equal to the earlier per-query body on that stack; in general equal to within a few float32 ulps, with the same ranking except between scores closer than float32 rounding of the dot products, and bitwise equal on some BLAS kernels only (equal on an AVX-512 kernel, about 1 ulp different on an AVX2-class kernel in a later check), `reports/engineering/batched_rescore/RESULTS.md`).
4. Dense output: `out = hot.copy()`; `floor` = row minimum of finite `cold`; `top_hot` = row maximum of finite `hot`; `shift = top_hot - floor + 1` (0 when `floor` is not finite);
   `out -= shift`; `out[shortlist] = cold` (`:130-135`): shortlisted documents keep cold scores, the rest sit at least 1.0 below the lowest finite cold score.
5. `search(queries, k)` = `rank(score(queries), k)` plus `take_along_axis` (`:139-142`). MEASURED-HERE: the final ranking is [shortlist sorted by cold score, ties by document position]
   + [all other documents in hot order]; `candidates > n_docs` clamps and `candidates >= n_docs` reproduces the cold ranking (`tests/test_storage.py:51-55`).

**Parameters:** `candidates=50` is a fixed policy in Q4, R12, Q5/Q6 and ENG-1, not tuned (`reports/research/Q5Q6/PLAN.md:12-17`); R6 also used 10, 20, 100
(`scripts/universal_study/s8_tiered.py:60-63`), ENG-1 used 1, 10, 50, 200, 500. Measured configuration: hot `Pipeline([HierarchicalMerge(ratio=0.25), BinaryQuantizer()])` or
`Pipeline([BinaryQuantizer()])`; cold `Pipeline([Int8Quantizer("per_vector")])` over all unmerged vectors (`docs/UNIVERSAL.md:407-409`: a merged cold tier recovers only 95.9-97.3%).
Each `ExactIndex` takes raw queries and applies its own `query_transform` (`storage.py:47-48`). Nothing checks that hot and cold hold the same documents in the same order.
`last_timing` is shared mutable state (`:114,136`). Script accounting: RAM = hot bytes per doc, disk = hot + cold (`s13_wide_study.py:157-160`).

**Legacy two-stage path:** `OptiVisionRAG.search_many` fetches `max(k, prefilter_k)` hits from a `BaseIndex` and, when `rerank` and `rerank_cache` are set, rescoring them with float
vectors (`pipeline.py:223-225,271-286`). It reloads the whole float cache with `np.load(..., allow_pickle=True)` on every query (`:264,275`), mixes float-scale and compressed-scale
scores for uncached hits, and reports `reranked=len(hits)` even when no cache was used (`:236`).

**Scripts that compute the "two-tier" rows** (all via `TieredIndex.score` + `per_query_metrics` unless noted):

| script | rows |
|---|---|
| `scripts/universal_study/s8_tiered.py:30-77` | R6: hot binary or adaptive-merge > binary; cold int8 all vectors, float16, or int8 of the same merge; m in 10, 20, 50, 100 (`docs/UNIVERSAL.md:381-414`) |
| `scripts/universal_study/s13_wide_study.py:144-164` | R12 (2,560-d): hot binary and Ward 1/4 > binary; cold int8 per-vector; m=50; RAM and disk bytes; `hot_ms`, `rescore_ms` |
| `scripts/research/q4_ablation.py:154-171`, `q4_latency.py:63-93` | Q4: the two R12 rows on six 128-d datasets; component timings |
| `scripts/research/q56_common.py:168-189`, `q56_correctness.py`, `q56_latency.py:66-81` | Q5/Q6: `merge_tiers` is a verbatim copy of `TieredIndex.score`'s construction; `experimental_two_tier` swaps in native scorers |
| `scripts/engineering/batched_rescore.py`, `scripts/research/q8_scaling.py`, `q8_projections.py` | ENG-1 validation (m 1..500, six datasets); Q8 rescoring vs candidate overlap, projections |
| `scripts/review/q11_margin_rescoring.py` | early prototype: `np.sign` binary shortlist K=10, fixed int8, margin-triggered; different tie rules |

## 5. What Q5/Q6 measured, and the wording it imposes

### 5.1 Conditions (`reports/research/Q5Q6/REPORT.md:15-23`, `PLAN.md`, `latency.json:411-424`)

- **Data:** ColQwen2 v1.0 on ViDoRe V1 DocVQA: 500 pages, 374,154 vectors (128-d), 451 queries (9,111 tokens). Two-tier: hot Ward 1/4 > binary (97,683 vectors), cold per-vector int8 of
  all vectors, 50 candidates.
- **Hardware and stack:** the Q4 laptop (Intel 11th-gen mobile, 8 logical CPUs; i7-1165G7 per `reports/research/Q8/REPORT.md`), Windows, Python 3.11.9, numpy 2.4.6 (OpenBLAS 0.3.31),
  torch 2.11.0+cpu with 4 threads. No numba, no compiler, no SIMD intrinsics. Commit `3a1e3dc`.
- **Protocol:** one timed call scores all 451 queries (batch); ms/query = wall / 451; one untimed warm-up, 5 timed rounds, round-robin with rotating start. Rounds 1-4 ran about 1.35x
  slower than round 5 for every configuration, so conclusions rest on within-round ratios. No per-query p95.
- **"Current"** meant the library at `3a1e3dc`, whose rescoring was a per-query loop; `198f83a` replaced it with the batched body that Q5/Q6 had used as its control.

### 5.2 Results (MEASURED, this stack only)

| question | result |
|---|---|
| Is direct int8 scoring (`torch.ops.aten._weight_int8pack_mm`, float query x int8 codes, per-vector multiplier, no decoded matrix) equivalent? | Strict criterion (identical full ranking, every query) failed: 407/451 (scan), 450/451 (rescoring); all 45 discordances were float32 near-ties (float64 gap <= 2.7e-6 vs error bound >= 1.7e-4, rank >= 35). The reviewed rounding-aware criterion (identical top-50 set, per-query nDCG@5, discordances within the bound) passed. nDCG@5 0.6063 vs float32 0.6065; retention 0.9997 [0.9937, 1.0056] (`REPORT.md:36-68`, `CORRECTNESS.md:23-72`, `PLAN.md:194-214`) |
| Faster? | Full scan 604 ms/query vs 22.5 for decode + BLAS: 23.8-34.1x slower. Rescoring about 3.5x slower (`REPORT.md:89-96`) |
| Exact binary first stage by per-byte lookup tables in numpy | shortlist identical for 451/451; 172 ms vs 5.8: 28-30x slower (`:107-116`) |
| Approximate binarised query (XOR + popcount, numpy >= 2) | 26.0 ms vs 5.8: 4.4-4.7x slower; labelled page in shortlist 81.2% vs 82.9%; final retention 0.9976 vs 0.9997 (`:118-130`) |
| End-to-end two-tier | library at the time 23.5 ms; native exact 250 (9.7-11.7x slower); control, batched decode-first, 8.6 (2.2-3.2x faster; 451/451 identical rankings) (`:132-148`) |
| Memory | transient peak 266-272 MB at the 256 MiB cap; lookup-table path 403 MB; XOR path 567 MB; resident float32 191.6 MB, binary hot 1.57 MB, int8 cold 48.6 MB (`:150-177`) |

Follow-up (`reports/engineering/batched_rescore/RESULTS.md`): batched `ExactIndex.rescore` kept; six 128-d datasets, rescoring 5.5-9.3x faster (median 6.7x), end to end 2.9-3.7x
(median 3.25x), scores bit-identical on that stack (equal to within a few float32 ulps in general, see above); the gain falls with candidate overlap (7.4x to 2.8x, `reports/research/Q10/REPORT.md:81`).

### 5.3 Not shown, and constraints for package text

Not shown: other hardware, models, datasets or widths (no batched-rescore timing at 2,560-d is recorded in the reports read); corpora beyond 4,000 duplicated pages (Q8); single-query
serving ("most of the batching gain would disappear", RESULTS section 4); a disk-backed cold tier; GPU; compiled int8 or lookup kernels ("INFERENCE: compiled int8 or LUT kernels are known to
beat decode + GEMM in other systems; that was not testable here", `REPORT.md:213-227`); Qdrant at any size. The do-not-claim list is `reports/research/Q10/REPORT.md:236-266`.

- W1. Never write that compressed or binary codes make search faster, or "Nx faster search". Say codes are decoded to float32 before scoring and scan time follows vector count
  (Q4 section 11: 0.53, 0.28, 0.16 of the float32 scan at 1/2, 1/4, 1/8 of the vectors; codecs 1.01-1.04x on the three 128-d models).
- W2. Say "native compressed scoring was slower in numpy and torch on one laptop CPU"; never the general form; say compiled kernels were not tested.
- W3. The batched-rescoring speedup is an engineering result (six 128-d datasets, one laptop, all-query batches): not hardware-independent, not a single-query claim, not research.
- W4. Two-tier buys resident RAM (hot tier 32x to about 125x smaller than float32 on the tested sets, retention 0.999-1.005 at 50 candidates; Q4 section 10), not total storage (disk
  about 3.5-3.9x) and not, as a general claim, speed.
- W5. MaxSim over decoded codes is "exact over the stored representation", never "lossless" or "exact retrieval of the float index".
- W6. No "scales to millions" or "handles 1M pages"; Q8's 1M figures are PROJECTED; the dense `[n_queries, n_docs]` output grows with the corpus.
- W7. Qdrant: "tested separately, the deployment path, not the measurement path"; local mode "is not a performance claim" (`docs/RESULTS.md:25-29`, `README.md:495-496`). No recall,
  latency or float-quality claim for server mode, and none that Qdrant rescoring recovers float32 quality (originals are not sent unless `method="none"`).
- W8. Any alternative scoring path (native kernel, another backend) is judged by the amended criterion (identical top-50 set, identical per-query nDCG@5, every discordance inside the
  float32 error bound), not bitwise equality (`PLAN.md:194-214`).
- W9. Reword before shipping: `docs/ARCHITECTURE.md:117` ("is faster"); `docs/UNIVERSAL.md:1001-1003` ("A bit-level Hamming search would be [faster]"; the numpy variants measured
  4-30x slower); `docs/UNIVERSAL.md:413-414,1007-1008` ("unoptimised Python loop", now batched); `README.md:281-288` ("independent of corpus size ... or a million"; measured on a
  4,000-page single-query `NumpyIndex`, `docs/IMPROVEMENTS.md:29-38`, while resident codes and the dense score matrix do grow); `index/numpy_index.py:3-6` (the universal layer's numbers
  come from `maxsim_matrix`); `index/qdrant_index.py:10-14`; `compression/binary.py:106` (popcount is not what runs).

## 6. Proposed minimal seam

New modules only (`optivision/retrieval.py`, `optivision/backends.py`); no existing file changes. The module names are placeholders: placement should follow whatever package layout is
chosen elsewhere (the protocols and `Ranking` are numpy-only; only the Qdrant wrapper touches `qdrant_client`, lazily, as `index/__init__.py:33` does). Python 3.10 syntax
(`requires-python >= 3.10`), `from __future__ import annotations`, numpy >= 1.26 (no `np.bitwise_count`). A prototype of every class below ran in the scratchpad against the
unchanged library: 16 of 16 equivalence checks passed (Appendix A).

### 6.1 Types and signatures

```python
@dataclass(frozen=True)
class Ranking:
    ids: list[list[str]]                # ranked document ids per query, best first
    scores: np.ndarray                  # float32 [n_q, width]; -inf pads short rows
    indices: np.ndarray | None = None   # int64 [n_q, width] store positions, -1 pads; None for id-only backends
    score_kind: str = "maxsim_ip"       # "maxsim_ip" | "maxsim_cosine" | "tiered"; compare scores only within a kind
    tie_break: str = "index"            # "index" (ascending store position) | "unspecified"

@runtime_checkable
class Retriever(Protocol):              # queries in -> ranked ids out
    @property
    def n_docs(self) -> int: ...
    def search(self, queries: MultiVectorCorpus, k: int = 10) -> Ranking: ...          # raw queries; k >= 1

@runtime_checkable
class DenseScorer(Protocol):            # optional capability: what calibration keeps
    def score(self, queries: MultiVectorCorpus) -> np.ndarray: ...                     # float32 [n_q, n_docs]; -inf = never retrievable

@runtime_checkable
class Reranker(Protocol):
    def rescore(self, queries: MultiVectorCorpus, candidates: np.ndarray) -> np.ndarray: ...   # int64 [n_q, m] -> float32 [n_q, m]

class ExactRetriever:                   # float corpus; Retriever + DenseScorer + Reranker
    ids: list[str]
    def __init__(self, corpus: MultiVectorCorpus, max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES) -> None: ...
class CompressedRetriever(ExactRetriever):   # same methods over a CompressedCorpus, with the pipeline's query projection
    def __init__(self, store: CompressedCorpus, pipeline: Pipeline | None = None, max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES) -> None: ...

class TieredRetriever:                  # in-process tiers; score() and search() are TieredIndex's, unchanged
    def __init__(self, hot: ExactRetriever, cold: ExactRetriever, candidates: int = 50) -> None: ...   # ValueError unless hot.ids == cold.ids
    last_timing: dict[str, float]
class TwoTierRetriever:                 # any shortlist source + any reranker (e.g. a backend, then a cold store)
    def __init__(self, first: Retriever, reranker: Reranker, candidates: int = 50) -> None: ...
    def search(self, queries: MultiVectorCorpus, k: int = 10) -> Ranking: ...          # 1 <= k <= min(candidates, n_docs)

@dataclass(frozen=True)
class BackendInfo:
    name: str; exact: bool; score_kind: str; tie_break: str    # "numpy"|"qdrant"; brute force vs ANN/quantised; "index"|"unspecified"
    dense_scores: bool; upsert: bool                           # score_all available; add() overwrites a page_id instead of appending

class Backend(Protocol):                # BaseIndex + info + search_many
    info: BackendInfo
    @property
    def n_pages(self) -> int: ...
    def add(self, pages: Sequence[CompressedPage]) -> None: ...
    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]: ...            # BaseIndex.search, unchanged
    def search_many(self, queries: MultiVectorCorpus, k: int = 10) -> Ranking: ...         # new
    def save(self) -> None: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...
def make_backend(index: BaseIndex) -> Backend: ...      # NumpyBackend / QdrantBackend; BackendInfo keyed on index.backend
```

`TwoTierRetriever.search`: `sl = first.search(queries, m)`; sort the shortlist by store position; `cold = reranker.rescore(queries, positions)` with `-1` pads masked (`rescore` raises on `-1`);
`rank(cold, k)`. When `first` is id-only (`indices is None`, a backend), positions come from a one-time `{id: position}` map of the reranker's store ids; an unknown id raises `KeyError`.

### 6.2 Delegation (no behaviour change)

| new | delegates to, unchanged |
|---|---|
| `ExactRetriever` | `ExactIndex.score/search/rescore` (`storage.py:50-102`) -> `maxsim_matrix`, `rank`; store `MultiVectorCorpus.decode_rows` (`representation.py:284-290`); ids `corpus.ids` |
| `CompressedRetriever` | the same over `CompressedCorpus.decode_rows` -> `Quantizer.decode` (`compose.py:86-87`); projection `Pipeline.transform_queries` through `ExactIndex._q` (`storage.py:47-48`) |
| `TieredRetriever` | `TieredIndex.score/search/last_timing` (`storage.py:116-142`) |
| `TwoTierRetriever` | `first.search`, `Reranker.rescore`, `scoring.rank`; about 15 new lines, tested against `TieredIndex.search` |
| `Reranker` | `ExactIndex.rescore`; optional `FloatCacheReranker` over `pipeline._rerank` / `maxsim_float` (`pipeline.py:271-286`) for the legacy float cache |
| `NumpyBackend` | `NumpyIndex.add/save/load/stats/refs/score_all` and `rank`; `search_many` = `score_all` + `rank` (deterministic ties); `NumpyIndex.search` stays reachable as `search` |
| `QdrantBackend` | `QdrantIndex.add/search/n_pages/stats/close`; `search_many` loops `search`, ids from `hit.ref.page_id`; `info = (exact=local, "maxsim_cosine", "unspecified")` |
| metrics hook | split `per_query_metrics` at `evaluation.py:61` into `_metrics_from_top(top, relevant, parsed)` (pure refactor): `Ranking.indices` then feeds the selection layer with no dense matrix |

### 6.3 Invariants the wrappers must keep

- I1. Scores float32; in-process retrievers rank with `rank()`: descending, ties by ascending store position.
- I2. Empty document `-inf`, empty query `0`; wrappers map the `NumpyIndex` all-empty `0` to `-inf`.
- I3. Retrievers take raw queries and own their `query_transform` (`ExactIndex._q`); tiers may project differently.
- I4. Decode-first asymmetric scoring is the reference; no native kernel without the W8 criterion.
- I5. Tiered semantics: shortlist = `rank(hot, min(m, n_docs))`; final = shortlist sorted by cold score with ties by store position, then the rest in hot order; dense `score` keeps the +1
  shift. MEASURED-HERE: a generic version that ranks the shortlist in first-stage order breaks cold ties differently (`[2, 0, 1]` vs `TieredIndex`'s `[0, 2, 1]`).
- I6. `candidates` clamps to `n_docs`; `k >= 1` is validated (`rank(scores, -1)` silently drops the last column); `rescore` ids outside `[0, n_docs)` get a clear error instead of
  `ZeroDivisionError` (`-1`, `storage.py:96`) or `IndexError`.
- I7. Hot and cold are aligned (`ids` equal) or construction fails; `TieredIndex` does not check.
- I8. `NumpyIndex`, `QdrantIndex`, `get_index`, `mode` and legacy `search` are untouched; `BackendInfo` records `score_kind` and `tie_break` so NumPy and Qdrant scores are never compared.
- I9. `runtime_checkable` Protocols check attribute presence only; behaviour is pinned by the tests below.

Order of work: (1) `retrieval.py` with `Ranking`, protocols, `ExactRetriever`, `CompressedRetriever` + T1-T9; (2) tiered classes + T10-T13; (3) `backends.py` + T14-T18; (4)
`_metrics_from_top`, then let `measure` and `score_candidate` accept a retriever while keeping the dense path (T19); (5) reword docs per W1-W9.

## 7. Known-ranking tests

All run on CPU numpy in milliseconds; hand values below were checked against the current code.

**Hand-computable**
- T1. `q = [[1,0],[0,1]]`; `d0 = [[1,0],[0,1]]`, `d1 = [[.5,.5]]`, `d2 = [[1,0]]`, `d3 = []`, `d4 = [[-1,0],[0,-1]]`, `d5 = [[-1,-1]]`. Scores `[2, 1, 1, -inf, 0, -2]`, ranking
  `[0,1,2,4,5,3]` (tie `d1`/`d2` by position, negative score kept, empty last). `maxsim_pair` agrees.
- T2. One vs three copies of a document vector score identically (weights are irrelevant).
- T3. Inner product vs cosine: `q = [[10,0],[0,.1]]`; `A = [[1,0]]`, `B = [[0,1]]`, `C = [[.7071,.7071]]`. Inner product `[10, 0.1, 7.1418]` -> `A, C, B`; cosine `[1, 1, 1.4142]`
  -> `C, A, B`. Pins `score_kind` and the unit-norm precondition for Qdrant parity.
- T4. Padding: appending `[0,0]` to a query changes nothing; appending it to `d0 = [[-1,0]]` (query `[[1,0]]`, other document `[[-.5,0]]`) flips `[1,0]` to `[0,1]`.
- T5. Empty query row: 0 against non-empty documents, `-inf` against empty ones; all-empty query set.

**Equivalence**
- T6. `ExactRetriever.search(Q, k).indices == rank(maxsim_matrix(Q, C), k)`, scores equal `take_along_axis`, ids equal `corpus.ids[indices]` (extends `tests/test_storage.py:26-31`).
- T7. `CompressedRetriever` over float32 `Pipeline()` equals `ExactRetriever` bitwise; over binary and per-vector int8 equals `rank(maxsim_matrix(Q, compressed), k)`.
- T8. Budget invariance: `max_block_bytes` of 64, 4096 and default give equal rankings up to float32 near-ties (`assert_close_modulo_ties`: allclose scores, order equal outside gaps
  below ~1e-5 relative), not `array_equal` (`tests/test_scoring.py:43-49` uses `rtol=1e-6`).
- T9. One query vs the same query inside a batch: bitwise equal on the AVX-512 kernel used here when queries have enough tokens (14-30 here), and only equal to within a few float32 ulps on some other BLAS kernels; allclose for 1-3 tokens (`batched_rescore/RESULTS.md` section 1;
  `tests/test_storage.py:156-165`).
- T10. Two-tier with `candidates >= n_docs` equals the cold ranking; with `candidates = 1` column 0 is the hot top-1 and the rest is the hot ranking; in general shortlist-by-cold + rest-by-hot.
- T11. Generic `TwoTierRetriever` equals `TieredRetriever.search` for `k <= m`, including exact cold ties among shortlisted documents (hot ranks `d2` above `d0`, cold ties them: `[0, 2, 1]`).
- T12. Hot and cold the same store: tiered ranking equals the single-tier ranking up to near-ties. T13. Misaligned tiers (reversed document order) raise at construction.
- T14. NumPy vs Qdrant on identical pages from `to_legacy_pages`, `method` binary or none, unit-norm queries: equal ids outside tie groups; for binary `qdrant_score * sqrt(d)` equals the NumPy
  score. Build Qdrant with `QdrantIndex(path=tmp_path)` as `tests/test_qdrant.py:29-34` does (there is no `:memory:` option). For int8 assert top-1 and `|ratio - 1| < 2e-2` only; for
  mixed-norm queries assert the documented difference (T3), not equality.
- T15. `NumpyBackend.search_many` vs `ExactRetriever` on the same binary store: allclose 1e-4, top-3 ids equal; `save`/`load` preserves the ranking (`tests/test_index.py:89-99`).

**Ties, determinism, contract**
- T16. 50 identical documents: in-process retrievers and `NumpyBackend` return `[0..k-1]`; legacy `NumpyIndex.search` does not (test the wrapper); Qdrant: compare the multiset of scores and
  the id set per tie group, never positions.
- T17. Empty documents: `-inf` through every retriever and last in the ranking; `NumpyBackend` on an all-empty index returns `-inf`; `QdrantBackend.add` rejects an empty page with a clear error.
- T18. Repeating a call in one process is bitwise equal; permuting document order permutes the ranking (no ties). Metamorphic, as in `scripts/research/x3_symmetry.py`: an orthogonal
  rotation of documents and queries, permuting rows inside documents or query tokens, leaves scores equal within 1e-4 relative; scaling all queries by `c > 0` keeps the ranking.
- T19. `per_query_metrics(scores, rel)` equals `_metrics_from_top(rank(scores, kmax), rel)` bitwise; a query with no relevant document is NaN in both.
- T20. Contract: `rescore` keeps column order and duplicates (`tests/test_storage.py:131-138`); `-1` and out-of-range ids, float candidates and `k < 1` are rejected by the wrappers.

## 8. Risks

**Silent dtype conversion**
- float64 queries and documents are narrowed to float32 (`scoring.py:146`, `representation.py:290`); `NumpyIndex.search` casts any dtype, integers included (`numpy_index.py:191`), while
  `MultiVectorCorpus` rejects integers; `rescore` casts candidate ids to int64 (`storage.py:70`). Convert through `MultiVectorCorpus` at the seam.
- Qdrant receives decoded codes, so "int8 in Qdrant" is decoded int8 floats plus engine-side binary quantisation by default, not int8 storage.
- Scores of different `score_kind`s are not comparable; `TieredIndex`'s dense matrix mixes cold and shifted hot scales by design. `to_legacy_pages` re-splits ids (section 2.5).

**Silent device transfers**
- `OPTIVISION_SCORE_DEVICE` is process-wide, read per call, honoured only by `maxsim_matrix`. On CUDA every document block goes host to device and every `[T, rows]` similarity block
  back (`scoring.py:96`), up to 256 MiB per block; max and sum run on the CPU; TF32 is disabled for the whole process (`:85-86`). `device="cpu"` still routes through torch, not numpy's BLAS.
- A GPU hot tier with the CPU `rescore` is the default combination; wrappers should report the device each tier used.

**Memory at 2,560-4,096 dimensions** (MEASURED-HERE unless marked)
- `maxsim_matrix` peak equals the budget for int8 and binary stores at D=2,560 (tracemalloc 16.0 MiB at a 16 MiB budget): document block `rows*D*4` plus similarity block `T*rows*4`. The
  formula omits the query chunk `T_chunk*D*4`: a zero-copy view for contiguous float32 queries, a real copy after `transform_queries` or for float16/float64 queries. DERIVED at the 256 MiB
  default: 9,111 query tokens add 89 MiB at 2,560-d and 142 MiB at 4,096-d; 16,384 tokens at 4,096-d add 256 MiB. For float32 stores the formula reserves a decode block that is never
  allocated (peak 4.7 MiB at the 16 MiB budget).
- Decode temporaries beyond the formula: binary holds a `uint8 [rows, D]` unpack plus the float32 block; lloyd2 up to three `[rows, D]` float32 arrays and a `D x D` rotation (64 MiB at
  4,096-d, up to 8 cached, `lloyd2.py:110-124`).
- `rescore` materialises every query token as float32 (`storage.py:78`) and gathers `qv[r]` per page outside the budget (`:99`); `best` is `tokens*m*4`. Large query batches at wide D are unbounded.
- Dense outputs: `maxsim_matrix` and `TieredIndex.score` return `[n_q, n_docs]` float32 (1.8 GB for 451 queries at 1M pages, PROJECTED, Q8); `TieredIndex.score` also holds `hot.copy()`
  and a masked temporary (`storage.py:130-133`), a multiple of that. A `search`-only retriever should avoid the dense matrix.
- `NumpyIndex`'s decoded cache holds only about 26,000 vectors at 2,560-d (DERIVED: 256 MiB / (2,560 x 4 B)); beyond that every query re-decodes the index (`numpy_index.py:196-222`); `add` is quadratic (`:115-118`).
- `QdrantIndex.add` builds Python floats for every element (about 32 bytes each, roughly 60 MB for one 750-vector 2,560-d page, INFERENCE from CPython float size) for the whole batch
  before `upsert` (`qdrant_index.py:155-192`). `MultiVectorCorpus.load(mmap=True)` does not memory-map; `NumpyIndex.load` has no mmap option.

**Non-determinism**
- Last-bit effects: block budget; BLAS kernel choice for products with 1-3 rows; different blocking in the single-query product of `NumpyIndex` versus the batched product of
  `maxsim_matrix` (MEASURED-HERE, `batched_rescore/RESULTS.md` section 1). INFERENCE, not measured: BLAS thread count and CPU kernel selection can change bits across machines.
  Near-ties then flip positions (Q5/Q6).
- `NumpyIndex.search` tie membership; Qdrant local tie order; server HNSW and quantisation are approximate and unmeasured here; timing carries the CPU speed states seen in Q8 and
  ENG-1 (about 2.8x apart). `TieredIndex.last_timing` is shared mutable state.

**Semantic drift**
- Backends differ in re-add (append vs overwrite), `recreate` timing, empty pages and score scale (section 2.4); `get_index` silently prefers on-disk NumPy metadata over its arguments.
- `relevant_from_qrels` drops unknown ids silently, so an id-convention mismatch yields NaN queries rather than an error. The legacy rerank cache is loaded with `allow_pickle=True` on every query.

## Appendix A. Checks run for this audit

Throwaway scripts in the session scratchpad on synthetic arrays; nothing was written under the worktree except this file.

| id | check | outcome |
|---|---|---|
| A1 | T1 through `maxsim_matrix`, `maxsim_pair`, `rank`; NaN handling | `[2, 1, 1, -inf, 0, -2]`, `[0, 1, 2, 4, 5, 3]`, agrees; NaN scores rank last; one NaN query token makes the whole row NaN and `rank` returns `[0, 1, 2, ...]` |
| A2 | ties: 10 identical + 2 other pages; then 200, 2,000, 2,000, 5,000 tied pages | `rank` and `ExactIndex` give `[0..k-1]`; `NumpyIndex.search` matched at 12 pages, differed in the four larger cases; Qdrant local scrambled |
| A3 | NumPy vs Qdrant local, 40 pages, d=32, 30 queries per method | full rankings that differ, unit-norm / mixed-norm queries: binary 0/30 and 30/30 (score ratio 1.000000), none 0/30 and 30/30, int8 28/30 and 30/30; separate 60-query int8 run: top-1 agree 60/60, top-5 set 55/60, top-10 set 56/60, max score deviation 1.2% |
| A4 | empty pages | `NumpyIndex` mixed `[0, -inf, 1]`, all-empty `[0, 0]`; `maxsim_pair` `-inf`; `maxsim_float` and `maxsim_asymmetric_batch` 0.0; `QdrantIndex.add` `IndexError` |
| A5 | determinism | budgets 64/4096/default: not bitwise (max 9.2e-5 at scores of order 1e2), equal ranks; `score_all` vs `maxsim_matrix` max 1.9e-6, 1 of 40 queries re-ranked (rank 111, gap 9.5e-7); one query vs batch bitwise equal (14-30 tokens) |
| A6 | memory at D=2,560, 16 MiB budget; `rescore` and `TieredIndex` contracts | peak 4.7 MiB (float32), 16.0 MiB (int8, binary); `-1` -> `ZeroDivisionError`, id 30 of 30 -> `IndexError`, float ids cast; tiered ranking identity for m = 1, 5, all |
| A7 | other facts | `load(mmap=True)` is an `ndarray`; same page added twice -> 2 pages; width mismatch raises before mutating; Qdrant local layout and float32 unit-norm storage |
| A8 | prototype of section 6.1 against the unchanged library | 16 of 16 equivalence checks pass (exact == `rank(maxsim_matrix)`, compressed binary/int8, generic two-tier == `TieredIndex.search` for m = 1, 5, 12, 40, m = n_docs == cold ranking, misaligned tiers rejected, tie order, NumPy and Qdrant wrappers); a variant without the by-position shortlist sort fails the tie case |
