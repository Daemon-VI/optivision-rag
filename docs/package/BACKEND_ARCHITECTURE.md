# Retrieval seam and backend interface, as built

Status: 2026-10-06, branch `package-universal`. This describes what exists in
`src/optivision/core/retrieve/` and `src/optivision/backends/`. Contract: `UNIVERSAL_PACKAGE_DESIGN.md`
D14, D15, section 5.5. Specification: `RETRIEVAL_BACKEND_AUDIT.md` sections 5 to 8.

Nothing here is a research result and no measurement was made for it. Numbers are quoted from the
committed report named beside them. Labels: **MEASURED** (committed report), **OBSERVED** (a
throwaway check made while building this code, not a committed report), **ENGINEERING** (pinned by a
test in this repository, synthetic data), **NOT MEASURED**.

## 1. What was built

Nothing existing was edited: the pinned files, `storage.py`, `index/*`, `pipeline.py` and every
existing test are as they were. Every new retriever adds validation and a typed result to an existing
function; none adds arithmetic.

| new name | file | delegates to, unchanged |
|---|---|---|
| `Ranking`, `Retriever`, `DenseScorer`, `Reranker`, `comparable` | `core/retrieve/types.py` | (types only) |
| `ExactRetriever`, `CompressedRetriever` | `core/retrieve/exact.py` | `ExactIndex.score/search/rescore` -> `maxsim_matrix`, `rank`; queries through `Pipeline.transform_queries` |
| `TieredRetriever` | `core/retrieve/tiered.py` | `TieredIndex.score/search/last_timing` |
| `TwoTierRetriever` | `core/retrieve/tiered.py` | any `Retriever` + any `Reranker`, `scoring.rank` |
| `ranking_from_scores`, `ranking_from_positions` | `core/retrieve/exact.py` | `scoring.rank`, `take_along_axis` |
| `check_k`, `as_queries`, `check_positions`, `check_candidates` | `core/retrieve/checks.py` | (validation only) |
| `BackendInfo`, `Backend`, `BackendRetriever`, `legacy_page_id`, `legacy_id_mismatches` | `backends/base.py` | |
| `NumpyBackend` | `backends/numpy_backend.py` | `NumpyIndex.add/search/save/stats/n_pages/close/score_all` |
| `QdrantBackend` | `backends/qdrant_backend.py` | `QdrantIndex.add/search/save/stats/n_pages/close` |
| `make_backend` | `backends/factory.py` | |

`optivision.backends` also re-exports `BaseIndex`, `NumpyIndex`, `get_index`, `open_index` and
(imported on first access) `QdrantIndex` as the canonical import place. They are the same objects as
in `optivision.index`, which keeps working. `core/retrieve` imports only NumPy, the standard library,
`optivision.core.*`, `scoring`, `compose`, `representation` and `ExactIndex`/`TieredIndex`;
`backends` imports `core.retrieve`, `index.*` and `types`. Both are checked by tests.

## 2. The types

`Ranking` is a frozen dataclass: `ids` (`list[list[str]]`), `scores` (float32 `[n_q, width]`),
`indices` (int64, or `None` for id-only producers), `score_kind`, `tie_break`. Rows shorter than
`width` are padded (`-inf`, index `-1`); padding is told apart from a real empty document (also
`-inf`) by `len(ids[q])` and `indices == -1`. Construction validates shapes and dtypes, descending
order, the padding convention, rejects NaN, and for `tie_break="index"` checks that equal scores have
ascending positions. It keeps read-only private copies. `==` is exact; `top(k)`, `to_lists()`,
`lengths()` are helpers.

`Retriever` (`n_docs`, `search(queries, k)`), `DenseScorer` (`score`), `Reranker` (`rescore`) are
runtime-checkable. `isinstance` checks attribute presence only; before Python 3.12 it also evaluates
properties, so `isinstance(backend, Backend)` calls `n_pages` once (OBSERVED on 3.11.9), which asks an
engine to count its points.

## 3. Invariants

| | as built | pinned by |
|---|---|---|
| I1 | scores float32; in-process retrievers rank with `rank()`: descending, ties by ascending store position | `test_core_retrieve.py`, `Ranking` validation |
| I2 | empty document `-inf`, empty query 0; `NumpyBackend` maps `NumpyIndex`'s all-empty 0 to `-inf`; `QdrantBackend` refuses empty pages and queries | `test_core_retrieve.py` T5, T17; backend tests |
| I3 | retrievers take raw queries and own their projection; the two tiers of a `TieredRetriever` project separately | `test_core_retrieve_tiered.py` |
| I4 | decode-first asymmetric scoring is the reference; no native kernel | nothing to pin: no kernel exists; an alternative must pass the W8 criterion (section 10) |
| I5 | shortlist = `rank(hot, min(m, n))`; final = shortlist by cold score, ties by store position, then the rest in hot order; `TwoTierRetriever` sorts the shortlist by position first | tie-heavy and hand-built tests (T11) |
| I6 | `candidates` clamps to `n_docs`; `k >= 1` and integer; candidate positions must be integers in `[0, n_docs)` (`IndexError`, `TypeError`; no `ZeroDivisionError`) | T20 |
| I7 | `TieredRetriever` raises unless hot and cold have the same ids in the same order (T13); `TwoTierRetriever` checks it when both expose `ids` | T13 |
| I8 | `NumpyIndex`, `QdrantIndex`, `get_index`, legacy `search` untouched; `BackendInfo` records kind and tie-break | `test_backends_protocol.py` |
| I9 | protocols check presence only; behaviour is tested | `TestProtocols` |

Added checks (not in the audit): queries must be finite and of the store's dimension (the scorer returns
an all-NaN row for a NaN token, and `rank` turns it into index order); the two-tier path rejects a
document proposed twice, a position past the store, an id the store does not hold (`KeyError`), and
proposed ids that disagree with the reranker's ids.

## 4. Score kinds, and why scores are not compared across backends

`score_kind` is `maxsim_ip` (inner-product MaxSim, `scoring.py`; `NumpyBackend`, all in-process
retrievers), `maxsim_cosine` (`QdrantBackend`) or `tiered`. NumPy's score is the raw inner product.
Qdrant normalises stored vectors and queries, so its score is a sum of cosines. They coincide only for
unit-norm query tokens against unit-norm stored vectors, and then only up to the norm of the decoded
document vectors: a binary code decodes to norm `sqrt(d)`, so Qdrant's score times `sqrt(d)` equals the
NumPy score there. With mixed-norm query tokens the rankings differ (audit finding 4 and Appendix A3, a throwaway
check rather than a committed report; re-checked on synthetic queries in `test_backends_qdrant.py`, ENGINEERING). A tiered
`Ranking` mixes cold scores (shortlist) with hot scores shifted below them (the rest).

`comparable(a, b)` (on rankings or `BackendInfo`) is true when `score_kind` and `tie_break` agree. It is
necessary, not sufficient: two codecs in one kind still differ in scale. Ranks may be compared across
kinds; scores may not.

## 5. Ties, empties, determinism

| path | ties | one process, same call twice |
|---|---|---|
| in-process retrievers, `NumpyBackend.search_many` | ascending position (stable `rank`) | bitwise equal (ENGINEERING) |
| legacy `NumpyIndex.search` (left unchanged) | unspecified: `argpartition` then `argsort` | not tested |
| Qdrant local | unspecified; 10 identical pages came back in reverse order (OBSERVED) | same ids; scores differed by one float32 ulp in some calls (OBSERVED) |

Equality between two scoring paths is rounding-level, never bitwise (audit finding 7; `max_block_bytes`,
batch shape and BLAS kernel choice change the last bits), so near-ties can swap. Tests compare scores
with tolerances and orders outside near-tie groups. Empty document: `-inf`, ranked last. Empty query: 0
against every non-empty document. One pair is inconsistent in the existing code: for a query without tokens
against a document without vectors, `maxsim_matrix` gives `-inf` and `ExactIndex.rescore` gives 0.0 (OBSERVED);
`rescore` is delegated unchanged, so the tiered retrievers inherit it. The legacy `compression.maxsim_*`
helpers return 0.0 for an empty page (test-pinned) and `QdrantIndex.add` fails on one; those are not reached
through this seam.

## 6. What is exact over what

`ExactRetriever` is exact MaxSim over the float32 view of the corpus. `CompressedRetriever`,
`NumpyBackend` and local Qdrant are exact over the *stored representation* (the decoded codes), which
is not the float vectors the codes came from: never "lossless", never "exact retrieval of the float
index" (W5). Two-tier search is exact over the hot store for the shortlist and exact over the cold store
for the shortlisted documents only; a document outside the shortlist cannot be recovered. A Qdrant
server searches an HNSW graph with binary quantisation and engine-decided rescoring: approximate, NOT
MEASURED here. `QdrantIndex` sends decoded codes (`binary`: +/-1 floats; `int8`: dequantised floats; the
originals only for `method="none"`), so engine-side rescoring cannot restore float32 quality.

## 7. Memory: what is materialised

| call | held in memory (DERIVED from shapes and the code) |
|---|---|
| `ExactRetriever.score` / `search` | dense float32 `[n_queries, n_docs]`; `max_block_bytes` (256 MiB default) bounds the decoded document block and the similarity block only, not the query chunk and not the matrix |
| `rescore` | every query vector as float32, plus float32 `[n_query_vectors, m]`; per-page gathers are outside the budget |
| `TieredRetriever.score` / `search` | the dense hot matrix, a copy of it and masked temporaries: a small multiple of `[n_queries, n_docs]` |
| `TwoTierRetriever.search` | the first stage's `[n_queries, m]` ranking and the rescoring tables; no dense matrix of its own |
| `NumpyBackend.score` | dense `[n_queries, n_pages]`; `NumpyIndex` caches the decoded float32 index up to 256 MiB, then decodes block by block for every query |
| `QdrantBackend` | queries cross as Python lists; `QdrantIndex.add` builds Python floats for a whole batch |

The dense matrix grows with the corpus, so none of this is a scalability claim (W6): Q8's figures for 1M
pages (for example 1.8 GB of scores for 451 queries) are PROJECTED arithmetic, `reports/research/Q8/REPORT.md`.

## 8. Device handling, as it is

New code selects no device, transfers nothing and reads no environment variable (a test checks the
sources). The only control in the library is `OPTIVISION_SCORE_DEVICE`, read on every call inside
`maxsim_matrix`: it applies to `score`, `search` and `TieredRetriever.score`, not to `rescore`,
`NumpyBackend` or Qdrant, which are CPU NumPy. On a GPU only the products move; the maximum and the sum
stay on the CPU and TF32 is switched off process-wide (audit section 8). Wrappers do not report a device.

## 9. Two tiers, and what they buy

A hot store (for example merged vectors as binary codes) shortlists `candidates` documents; a cold store
(int8 or float) rescores only those. What it buys is **resident memory**: on the tested 128-d sets the hot
tier was 32x (binary) to about 125x (Ward 1/4 then binary) smaller than float32, at retention 0.999 to
1.005 with 50 candidates (MEASURED, `reports/research/Q4/REPORT.md` section 10). It does not buy total
storage (disk about 3.5 to 3.8x smaller than float32 there, same section) and, as a general claim,
not speed (W4): the hot scan still reads every hot document, and the dense matrix is still built. Quality is
bounded by whether the right document reaches the shortlist and by the cold scores, both to be measured
per use. `TwoTierRetriever` equals `TieredIndex.search` for `k <= candidates` while the shortlist has at
least `k` documents with a finite cold score; if a shortlisted document is empty in the cold store,
`TieredIndex` ranks finite documents outside the shortlist above it and `TwoTierRetriever`, having no such
documents, ranks it last within the shortlist (ENGINEERING, documented difference).

## 10. What Q5/Q6 measured, and the wording it imposes

Conditions (MEASURED, `reports/research/Q5Q6/REPORT.md`): ColQwen2 v1.0 on ViDoRe V1 DocVQA, 500 pages
(374,154 vectors, 128-d), 451 queries scored as one batch; two-tier = Ward 1/4 then binary hot, per-vector
int8 cold, 50 candidates; one 11th-generation Intel laptop CPU (8 logical CPUs), Windows, Python 3.11, numpy 2.4.6
(OpenBLAS), torch 2.11.0+cpu with 4 threads; no numba, no compiler, no SIMD intrinsics. Timed rounds
drifted (rounds 1-4 about 1.35x slower than round 5), so conclusions rest on within-round ratios.

Results on that stack: scoring int8 codes directly with torch's mixed float-by-int8 kernel was 23.8 to
34.1x **slower** than decoding and using BLAS (604 against 22.5 ms/query, full scan) and about 3.5x slower for
rescoring; an exact binary first stage by lookup tables in numpy was 28 to 30x slower (172 against 5.8); a
binarised-query XOR/popcount first stage was 4.4 to 4.7x slower and approximate; the complete native two-tier
was 9.7 to 11.7x slower end to end. Equivalence: the strict criterion (identical full ranking) failed, 407 and
450 of 451 queries, and all 45 discordances were float32 near-ties; the reviewed rounding-aware criterion
(identical top-50 set, identical per-query nDCG@5, every discordance inside the float32 bound) passed.

So, and only so: **native compressed scoring was slower in numpy and torch on one laptop CPU; compiled kernels
were not tested** (the report's own inference that compiled kernels beat decode plus GEMM elsewhere "was not
testable here"). Nothing general is claimed (W2). Decode-first scoring is therefore the reference semantics (I4)
and no kernel is shipped. Further constraints on package text: W1: codes are decoded to float32 before
scoring, scan time follows the vector count (0.53, 0.28, 0.16 of the float32 scan at 1/2, 1/4, 1/8 of the
vectors; codecs 1.01 to 1.04x, `Q4/REPORT.md` section 11), so no "faster search from compression". W3: the batched
`rescore` (5.5 to 9.3x on rescoring, 2.9 to 3.7x end to end, bit-identical rankings) is an engineering result on six
128-d datasets, one laptop, all-query batches of 451 to 494 queries
(`reports/engineering/batched_rescore/RESULTS.md`); it is not hardware-independent, not a single-query claim, not
research. W7: Qdrant is "tested separately ... the deployment path, not the measurement path" (`docs/RESULTS.md`),
local mode is "convenient but not a performance claim" (`README.md`); no recall, latency or float-quality claim for the server.
W8: any alternative scoring path is judged by the rounding-aware criterion above, not bitwise equality.

## 11. Page ids

A backend names a page `"<doc_id>::p<page_no>"`. `to_legacy_pages` rebuilds each `PageRef` from a corpus id
by splitting at the first `"::p"`, so `"abc"` returns as `"abc::p1"`, `"a::pb"` as `"a::p1"`, `"x::p01"` as
`"x::p1"` (and `"z::p²"` makes it raise). Ids already of the form `"<doc>::p<N>"` survive. Nothing rewrites ids:
`TwoTierRetriever` raises `KeyError` for an id its reranker's store does not hold, `legacy_id_mismatches(ids)`
lists the corpus ids that would not survive, and `BackendRetriever` drops backend positions (a page's position
in an index is insertion order, not a contract) so that ids are always what is matched.

## 12. Defects left unchanged (reported, not fixed)

From the audit (section 8, findings 3 to 6): three tie conventions; NumPy and Qdrant score differently; Qdrant
receives decoded codes (the `index/qdrant_index.py` docstring said otherwise and was corrected); three empty-document conventions;
float64 narrowing in `scoring.py` and `NumpyIndex.search`; `OPTIVISION_SCORE_DEVICE` global and partial;
`NumpyIndex.add` quadratic and not idempotent; `get_index` prefers on-disk metadata over its arguments;
`OptiVisionRAG` rerank reloads a pickled cache per query and mixes score scales; `relevant_from_qrels` drops
unknown ids silently; `TieredIndex.last_timing` is shared state and `TieredIndex` fails on an empty corpus (and gives NaN scores below the shortlist if every hot document is empty for a query).
Found while building this (OBSERVED, throwaway snippets): `QdrantIndex.add` with an empty page raises
`IndexError` after the point is registered, so in local mode `n_pages` rose 6 to 7 and every later `search` on
that open index failed until it was reopened (the wrapper refuses empty pages first); Qdrant local rejects an
empty query with an opaque `ValueError` and a NaN query with an `AssertionError` (the wrapper validates);
`to_legacy_pages` crashes on an id whose suffix is a superscript digit.

## 13. Adding a backend

1. Wrap an index that implements `BaseIndex`; do not edit it. Hold it as `.index`.
2. `info = BackendInfo(...)` with true values for `exact`, `score_kind`, `tie_break`, `dense_scores`, `upsert`.
3. `search_many` validates with `check_k` and `as_queries`, refuses inputs the engine mishandles before sending
   anything, and returns a `Ranking` (`ranking_from_scores` for a dense matrix; id-only rankings carry
   `indices=None`). Name any float conversion in the docstring.
4. Import the engine inside functions; register the wrapper in `make_backend`; export it.
5. Add a factory to `BACKEND_FACTORIES` in `tests/test_backends_protocol.py` and run
   `PYTHONPATH=src python -m pytest tests/test_backends_protocol.py -k <name>`; add `tests/test_backends_<name>.py`
   for its score relation to the NumPy backend, ties, empties and persistence.
6. Run `tests/test_core_layering.py` and `tests/test_core_retrieve.py::TestSourceProperties`.

## 14. Existing texts the audit flagged as overstated (W9)

Status after the later documentation pass on this branch: the `docs/ARCHITECTURE.md`,
`README.md`, `index/numpy_index.py` and `index/qdrant_index.py` texts below were reworded
(docstrings and prose only; no code change). The `docs/UNIVERSAL.md` texts belong to a results
record that is not edited, and `compression/binary.py` is byte-pinned by the research
harness; those three rows still stand.

| where | says | audit reason |
|---|---|---|
| `docs/ARCHITECTURE.md:117` | the symmetric popcount alternative "is faster" | the numpy variants measured were slower (section 10) |
| `docs/UNIVERSAL.md:1001-1003` | "A bit-level Hamming search would be [faster]" | same |
| `docs/UNIVERSAL.md:413-414, 1007-1008` | rescoring is "an unoptimised Python loop" | the library now batches it |
| `README.md:281-288` | peak RAM independent of corpus size, "or a million" | measured on 4,000 pages, one query at a time; resident codes and the dense matrix grow |
| `src/optivision/index/numpy_index.py:3-6` | every quality number is produced there | the universal layer's numbers come from `maxsim_matrix` |
| `src/optivision/index/qdrant_index.py:10-14` | the engine keeps originals for rescoring | decoded codes are sent unless `method="none"` |
| `src/optivision/compression/binary.py:106` | `maxsim_hamming` "scored by popcount" | the code is a float matrix product |
