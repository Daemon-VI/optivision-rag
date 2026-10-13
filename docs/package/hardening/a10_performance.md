# A10 performance review: ENGINEERING_BENCHMARKS.md, package_bench.py, stored JSON

Worktree `C:\Users\Lenovo\projects\ov-harden\a10` (HEAD 7a6769a), scratch `...\scratch\a10`. No library code changed, no commits, bench JSON and doc untouched, no registered experiment touched or rerun.

NOISE CAVEAT (applies to every timing below): the laptop was shared with about 13 other agents. Absolute times and wall-clock ratios in this report are noisy (one grid run gave a "non-collect share" of -155 %, i.e. pure noise). I rely on call counts, memory, byte counts and cProfile shares; timings are given only as context and marked as such.

## 1. What I executed

- Read the doc, `package_bench.py` (sections and helpers), `bench.json` (163 records), `environment.json`; read `api.certify`, `collect_evidence`, `create_certificate`, `verify_certificate/run_checks`, `bundle.save/load`, `identity.array_digest`, `corpus_identity`, `NumpyIndex/NumpyBackend`, `scoring.maxsim_matrix`, `calibration._compress`.
- Re-rendered the doc from the stored JSON (`package_bench.py --no-run --write-doc <scratch>`): identical to the committed doc except CRLF (the checkout has CRLF, the generator writes LF; an autocrlf artifact, not a defect). So every doc number is a deterministic function of the JSON.
- Recomputed independently from the `all` arrays: all 22 facade ratios (median of within-round ratios) match the stored ones exactly; corpus_identity/hashlib 1.361 (d=128) and 1.346 (d=1024) match "1.35 to 1.36"; section 5 shares sum to 100.0 %; section 6 sizes and 119.4 % match; section 7 98.3 % matches (0.98266).
- `git diff b06780b HEAD -- src scripts/engineering` is empty, so the measured code is HEAD's code; the byte-identity claim for pre-existing paths re-checks (`git diff 91564fd HEAD --stat` on the listed paths prints nothing).
- cProfile of `certify()` (default 40-candidate space; 40 to 160 docs, 40 to 160 queries, d=32..128, 7 configurations), tracemalloc of certify / save / load / verify, interleaved A/B loops (25 rounds, alternating order) for the retrieval seam and the NumPy backend, fresh-interpreter import loop (9 rounds, rotating order, base 91564fd exported with `git archive` into scratch vs HEAD), certificate size vs queries and candidates. Scripts are in the scratch directory (`prof.py grid.py mem.py art.py art2.py seam.py imp.py csize.py pqm.py scale.py`).

## 2. Findings

| ID | Sev | Where | Problem | Status |
|---|---|---|---|---|
| A10-01 | MEDIUM | `scripts/engineering/package_bench.py:296-313` (`backend_costs`); doc section 2 table, rows "numpy backend" | The peak-memory column compares unequal states: legacy `NumpyIndex.search` 14.4 MB (d=128) / 45.9 MB (d=1024) versus `NumpyBackend.search_many` 1 MB / 0.64 MB. | PROPOSED |
| A10-02 | LOW | `core/artifact/bundle.py:73,146` `_HASH_BLOCK = 1<<24` | Every file hash allocates a 16 MiB read buffer even for a 150 KB file: save and load of a 155 KB artifact peak at 17 MB. | NO ACTION |
| A10-03 | LOW | doc "What these numbers do not show" | Memory scaling of `certify()` is not stated anywhere: peak is about 11x the float32 corpus (measured), 4 bytes... see section 4. | PROPOSED (one line of doc) |
| A10-04 | NO ACTION | `core/artifact/record.py: exact_duplicates` | O(n_queries) Python loop of one `array_digest` per query, run at creation and again at binding verification. | NOT A DEFECT |
| A10-05 | NO ACTION | `optivision/__init__.py` (eager `from .core import stages`), `spaces._have_scipy` | `import optivision` pays about 25 ms for the new core stages; the first `certify()` pays 0.4 to 0.6 s for importing scipy.cluster/spatial, which section 8 does not show. | NOT A DEFECT |

### A10-01 (MEDIUM): numpy backend memory rows are an artefact of measurement order
`backend_costs` builds one `NumpyIndex`, wraps it in a `NumpyBackend` that holds the same index object, and measures peak memory with tracemalloc "during the warm-up call". `legacy` runs first. `NumpyIndex.score_all` lazily builds and caches a full float32 expansion of the corpus (`_decoded_matrix`, `numpy_index.py:145`; 36.6 MB at d=1024, which equals the 45.9 MB legacy peak). The backend then runs on a warm cache. Evidence (`seam.py`, d=128, 60 queries, same index): cold cache peak 4.42 MB (backend first) vs 4.38 MB (legacy), warm cache 0.28 MB vs 0.31 MB. The two paths allocate the same; the table's 14x to 70x difference is only the one-time decode. The doc sentence about "peak traced memory of each facade within 0.0%" does not cover these rows, but the table sits next to it and invites the reading "the backend needs 14 to 70 times less memory". Fix (cheap, benchmark script only): clear `index._decoded` before each peak measurement, or measure both variants with a pre-warmed cache, or drop the peak column for that case and add a footnote. Gain: removes a wrong-looking comparison; risk: none to the library. Worth doing before release (doc accuracy; the fix is regenerate-only). Side fact, not a defect: the legacy index caches a float32 copy of the corpus (32x its binary codes) below `max_decoded_bytes`; that is existing behaviour.

### A10-02 (LOW): 16 MiB hashing block
`tracemalloc` peak by piece (scratch `art2.py`): `_check_manifest` 16.95 MB, `load_artifact(verify=False)` 0.51 MB, `verify_certificate` replay 0.07 MB. `handle.read(16 MiB)` on a small file allocates the full request. Constant, not proportional to data, so it only matters for tiny artifacts on tiny machines. Fix: `_HASH_BLOCK = 1 << 20` (digest unchanged: block size does not affect SHA-256). Expected gain about 16 MB peak on load/save/verify of an artifact. Risk: very low. Before release: no.

### A10-04 (NO ACTION): exact_duplicates
In the binding check `_check_binding_queries` (52 ms of a 204 ms binding verify at 400 queries, loaded machine) 49 ms is `exact_duplicates` = 411 `array_digest` calls (about 120 us each); cost is linear in queries, and the count grew 81 -> 121 -> 201 for 40/80/160 queries in `certify`. Linear, tiny at normal sizes (tens of thousands of queries would cost seconds). Fix would be hashing rows in a vectorised way (changes digest semantics; do not).

### A10-05 (NO ACTION): import
`python -X importtime -c "import optivision"`: `optivision.core.stages` cumulative 25.7 ms of 524 ms (eager so that the `rotation` stage and traits register, `__init__.py:25`). Fresh-interpreter loop (9 rounds, rotating, loaded machine): `import optivision` base 91564fd min 572 ms, HEAD min 606 ms (+34 ms, about 6 %; medians 716 vs 846 ms are noise-dominated). First `optivision.certify` attribute access about 30 ms (lazy `__getattr__`); `import scipy.cluster.vq, scipy.spatial` after `import optivision` 390 to 580 ms (warm cache), which the first `certify()` pays through `CandidateSpace.default -> _have_scipy`. Section 8 reports 490 ms for `import optivision` and 143 ms for the layer and "third-party packages: none"; that is consistent but silent about scipy at first use. The doc could add one sentence; not required.

## 3. certify() overhead: where the time goes

Counts from cProfile (40-candidate default space; identical in every configuration, independent of docs/queries/dim):

| quantity | count | meaning |
|---|---|---|
| `maxsim_matrix` | 41 | 1 baseline + 1 per candidate: each pipeline scored once; uncompressed reference ranking computed once |
| `score_candidate` | 40 | one per candidate |
| `Pipeline.fit` (`compose.py`) | 12 | 12 distinct stage prefixes; the shared transform cache means merge/projection stages are fitted once per prefix, not per candidate |
| `per_query_metrics` | 41 | 1 baseline + 40 |
| `corpus_identity` | 2 | corpus once, queries once (create_certificate) |
| `validate_corpus` | 6 (ABSTAIN) | corpus 3x and queries 3x (`_as_corpus`, `collect_evidence`, `create_certificate`; a SELECT adds a 4th corpus pass in `compress_selected`) |
| `array_digest` | 81 / 121 / 201 for 40 / 80 / 160 queries | 1 per query (exact_duplicates) plus about 40 constant |
| `pipeline_fitted_identity` | 40 | one per candidate, 0.3 % |

Shares of the profiled time (cProfile, d=32..128, noisy but stable in order): `collect_evidence` 95 to 100 %; `score_candidate` 79 to 94 %; `maxsim_matrix` 10 to 49 %; `fit` 18 to 37 %; `per_query_metrics` 4 to 18 %; `create_certificate` 0.2 to 3 % (lowest when compression dominates); fitted-state digests 0.3 %; `select` 0.1 %. The stored bench agrees (compression 66 %, scoring 25 %, metrics 8 %, digests 0.1 %; certify minus collect = 0.2 s of 14.4 s). Validation passes cost about 7 ms per 11 MB at 1.5 GB/s (bench section 1), so 6 passes are under 0.1 % at bench sizes. Nothing is encoded, fitted or scored twice except the deployed candidate, which `compress_selected` re-fits on purpose to check the fitted-state digest (at most one candidate's compression, about 1.4 % in the bench: certify-collect = 0.2 s).

Scaling: linear in the number of candidates (maxsim calls = K+1; per-candidate cost depends on the stage chain, ward merge being the heavy one) and in queries and documents through `maxsim_matrix`; certificate creation is not quadratic anywhere I could find. Wall-clock grid (`scale.py`, min of 3, loaded machine, shown only to document that it is not usable): 19 candidates 0.83 to 6.4 s over docs 40/80/160, d=32..128, with "non-collect" shares from -154 % to +35 %. Do not quote these. `optimize()` for context: bench says 9.31 s vs 14.4 s on the same data, not comparable (different procedure), as the doc states.

## 4. Memory

- certify peak (tracemalloc, d=128, default 40 candidates, ABSTAIN path): 16.3 MB for a 1.42 MB corpus (11.4x) and 31.0 MB for 2.78 MB (11.1x): linear. 8.4x of the corpus is the transform cache (12 distinct prefixes, `calibration._cache_put`); the rest is the score matrix, per-candidate codes and the query side. A float16 corpus gave 16.5x of its own (smaller) size with the cache at the same 8.4x: widening to float32 for scoring blocks is bounded by `max_block_bytes`.
- The cache is capped by `OPTIVISION_CACHE_BYTES`, default 4 GiB (`calibration.py:98`, pinned file, pre-existing); beyond that least-recently-used prefixes are evicted and recomputed, so a large corpus degrades speed, not correctness; but a corpus above about 0.4 GB (float32 vectors) will use up to 4 GiB of cache on top of itself.
- `per_query_metrics` (pinned `evaluation.py`) peaks at exactly 3.00x the score matrix on top of it (10.8 MB for a 3.6 MB matrix; 43.2 MB for 14.4 MB), so the dense `[n_queries, n_docs]` step costs about 16 bytes per query-document pair. For 1000 queries x 100k documents that is 1.6 GB per candidate, transiently. Linear in the product, not accidental; the doc already says "the dense score matrix grows with the corpus" but gives no factor.
- The similarity block in `maxsim_matrix` is bounded (`DEFAULT_MAX_BLOCK_BYTES` 256 MiB); the doc's 243 MB figure for d=128 is that cap being nearly reached (2712 x 22407 x 4 B = 243 MB), not a growth law.
- float64 promotion: only where intended (evidence arrays, statistics, a norm check in 16 MB float64 blocks). No full-corpus float64 copy found.
- Copies: `as_corpus` does not copy a `MultiVectorCorpus`; the original corpus, the transformed corpora (cache) and the candidate codes are the only corpus-sized objects. Not "N copies" in the sense of a defect.
- Save/load: peaks 17.0 MB / 16.9 MB for a 155 KB artifact (A10-02).

## 5. Serialization and verification (SELECT outcome; 100 docs, 400 queries, d=64, 4-candidate space, target 0.9; times are loaded-machine context only)

- On disk: certificate.json 16.1 KB, codes.npz 157,147 B vs raw codes+offsets 155,134 B (101.3 %; `.npz` is stored uncompressed, no gain and almost no overhead), evidence.npz 21.3 KB, fitted.npz 878 B, MANIFEST 610 B; total 196 KB (126 % of raw codes). Bench (40 candidates): 705 KB, 119 %.
- Certificate size is constant in the number of queries (16,122 / 16,119 / 16,115 B for 200 / 800 / 3200 queries; the order is stored as a digest) and grows with candidates (about +1.3 KB per three added candidates; 35.9 KB at 40). Evidence arrays grow with K x n (8 B per cell).
- Verification cost by level (this run, context only): integrity 17 ms, replay 78 ms (replays `select` over the evidence), binding 187 ms for everything live; bench: 11.5 / 23.3 / 65.8 ms. Binding hashes the live corpus once (1.16 MB hashed for a 1.16 MB corpus: `corpus_identity` once) and the queries once (0.62 MB), plus one `norm_record` pass each and the per-query exact_duplicates loop; it does not re-hash the corpus several times. Codes and offsets are digested once each.
- `save_artifact` runs a full `verify_certificate` (including the replay) and hashes every written file (manifest): in a profile `_write_manifest` and the replay were most of the time; `load_artifact(verify=True)` repeats the manifest hash and the replay. Both are by design; load verified/unverified was 116 vs 12.8 ms in the bench.

## 6. Retrieval seam and backend

- Call counts for one `ExactRetriever.search`: `maxsim_matrix` 1, `_prepare` 1, `as_queries` 1, `ranking_from_positions` 1; no double scoring. The seam adds only the query check and the Ranking construction, which the bench measured alone (0.1 to 1.1 % of the direct call at k=10, 1.6 to 1.7 % at k=100).
- Interleaved A/B (150 docs, 150 queries, d=128, 25 rounds alternating order, loaded machine): median of within-round ratios 0.84 (float) and 0.89 (binary) with q10-q90 of about 0.5 to 1.65: the machine noise is 100 times the seam, so this loop confirms the doc's own statement that wall-clock ratios cannot resolve the seam; it neither supports nor contradicts "1.01 to 1.03x". The structural argument (one scoring pass + 2 small parts) supports the stated conclusion; the stored ratios are within the doc's own 4.8 % scatter bound. `NumpyBackend.search_many` vs legacy loop: ratio 1.025 here, bench 0.98 to 1.00: equal within noise (`search_many` = `score` loop of `score_all` + `ranking_from_scores`).

## 7. ENGINEERING_BENCHMARKS.md: claims vs data

Every number is supported by `bench.json` (see section 1). The doc contains none of the words faster / slower / negligible / scalable / cheap; every ratio sits next to its measurement and the scope sentence (synthetic, one laptop, not a research result) is in the first paragraph and in the closing list. Sentences to tighten (none changes a number):
1. Section 2 table, numpy rows: A10-01.
2. "quiet re-run: no other job started by this session was running" next to a speed probe range of 1.48x and 26 % system CPU at start: "quiet" is an assertion the data only partly supports; say "no other job started by this session".
3. Section 3 gives shares such as 0.00325 % and 0.000937 % (3 significant digits) from timings of calls that are 7 us against 12 ms measured in separate rounds; the digits are not meaningful. Statement of the Method ("carry the drift of the run") covers it; consider two significant digits.
4. Section 7 "dominant cost of a certify() call at these sizes" is supported (98.3 %) and scoped.
5. "In this run the median of a repeated direct call exceeded its best by up to 4.8 %": computed over the ExactIndex/TieredIndex rows only; the legacy NumpyIndex row is 5.1 % and the facade rows reach 10.1 %. True as scoped, but "a repeated direct call" reads wider than the two index types.
6. The doc does not state that certify peak memory is about 11x the corpus or that the first call imports scipy (A10-03, A10-05). Limits are otherwise listed honestly (no real data, no GPU, no single-query latency, no scale).

## 8. Bottlenecks table

| Where | Measured share | Cheap fix | Gain / risk | Before release? |
|---|---|---|---|---|
| `collect_evidence -> score_candidate` (compress + maxsim, pinned code) | 79 to 100 % of certify; bench: compression 66 %, scoring 25 % | none that is cheap; the cost is the number of candidates x corpus x queries | n/a | no (not a defect) |
| `Pipeline.fit` of ward merge (pinned) | 18 to 37 % of certify in profile | already cached per prefix (12 fits for 40 candidates) | none left | no |
| `per_query_metrics` | 4 to 18 % | none (pinned) | n/a | no |
| `create_certificate` | 0.2 to 3 % (51 ms in bench) | none needed | n/a | no |
| `exact_duplicates` per-query digest | about 25 % of a binding verify at 400 queries | none without changing digest semantics | n/a | no |
| `_HASH_BLOCK` 16 MiB read | 16 MB constant peak | 1 MiB | -16 MB peak, no risk | no |
| numpy backend memory rows in the doc | misleading, not a runtime cost | clear the decode cache or drop the column, regenerate | removes a wrong comparison; script-only | yes (A10-01) |

## 9. What I checked and found correct
Candidate encoded/fitted/scored once; baseline computed once; no repeated corpus hashing in binding; no float64 corpus copy; no quadratic behaviour in certify, save, load or verify; bounded similarity block; certificate size independent of the number of queries; doc regenerates from the JSON and all recomputed ratios match; measured code equals HEAD.

## 10. What I could not check
Absolute timings and any wall-clock ratio finer than about 50 % (shared machine); memory outside tracemalloc (BLAS buffers); large-corpus behaviour (more than a few hundred documents; the 4 GiB cache cap and the 16 B per query-document pair were derived from code and small runs, not run at scale); real encoder output (forbidden); the exact moment the cache starts evicting; behaviour on another machine or Python.
