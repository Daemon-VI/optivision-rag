# A05: CI failures in tests/test_storage.py::test_batched_rescore_*

## 1. Scope and what was executed
- `gh run list/view --log-failed` (read-only) on Daemon-VI/optivision-rag: run 37130562928 (commit 198f83a) and later runs.
- `git diff main HEAD` on storage.py, test_storage.py, scoring.py; `git log`.
- Local reproduction by forcing the OpenBLAS kernel: `OPENBLAS_CORETYPE={default,Haswell,Zen,SkylakeX,Sandybridge,Nehalem,Prescott}` on the dev venv (numpy 2.4.6, scipy-openblas 0.3.31, Windows). Scripts in `scratch/a05` (ulp.py, flips.py). A numpy 2.2.6 scratch venv (CI's version) gave the same tile-shape behaviour as ulp.py.
- Mutation experiments on storage.py (5 mutants); test_storage.py run on 5 kernels. The whole suite was NOT run.

## 2. Findings

### A05-01 (MEDIUM, tests/test_storage.py:120-226) Bitwise assertions are stale; the red CI is a real, reproduced BLAS dependence. FIXED in 9abd097
**Exact failure (CI log):** job `test (Python 3.10)`, ubuntu-24.04, Python 3.10.21, numpy 2.2.6 (manylinux OpenBLAS): 8 failed, 434 passed, 7 skipped. Failing: `test_batched_rescore_is_bitwise_the_per_query_rescore[{6,30}-{float,int8,binary}]`, `test_batched_rescore_keeps_column_order_and_duplicates`, `test_batched_rescore_with_projection_and_reload`. All fail at `np.testing.assert_array_equal(batched, per_query)`. Examples: `[6-float]` 11/42 mismatched, max abs 2.38e-07 (rel 2.25e-07); `[6-binary]` 5/42, 4.77e-07; `[30-binary]` 77/210, max abs 1.43e-06 (rel 2.0e-07); duplicates 8/28; projection 13/42. Differences are about 1 float32 ulp relative. The Python 3.13 job (numpy 2.5.3) PASSED on the same code, so it depends on the numpy/OpenBLAS build or runner CPU, not on the Python version. `m=1` passes (same product shape).

**Reproduction:** with `OPENBLAS_CORETYPE=Haswell` the local run gives exactly 8 failures with the same mismatch counts and magnitudes as CI (11/42, 2.3841858e-07, ..., 1.4305115e-06). Default kernel (AVX-512 here): 0 failures. Zen and Prescott: 8 failures; Nehalem: 4; Sandybridge and SkylakeX: 0. So the CI runner dispatches an AVX2-class kernel. This is inferred from the exact numeric match; the runner CPU is not printed in the log.

**Pre-dates package-universal?** Yes. The batched `rescore` and these tests came in with commit 198f83a (research-investigation, 2026-10-03), whose CI run 37130562928 is the first red one; later research-investigation runs (e.g. 37178276369) fail the same 8. They are not in v0.3.0: `git diff main HEAD` shows storage.py (+56/-18) and test_storage.py differing only by 198f83a; `git diff 91564fd HEAD -- storage.py test_storage.py scoring.py` was empty (package-universal changed none of them before my commit). Main at 4d52060 is green (run 36985326463), but only because it lacks the new code and tests.

### A05-02 Bitwise equality is not an appropriate contract (NOT a production defect)
Both paths do the same logical work: a d-term float32 dot product per (query token, document vector), the max over the document's vectors (exact), then a float32 sum over the query's T tokens. What differs is the shape of the GEMM: per-query computes `[T x d] @ [d x sum(len)]` once per query; batched computes `[rows x d] @ [d x len_doc]` per distinct document, chunked by `max_block_bytes`. OpenBLAS picks micro-kernels, M/N edge handling, k-unrolling and FMA use by shape and CPU, which changes the order and grouping of the d additions inside each dot product. Float addition is not associative, so the last bits differ.

Primitive-level demonstration (ulp.py, 64x16 unit vectors, one 64x64 product vs the same elements computed in tiles):

| tile (M x N) | default (AVX-512): entries differing | Haswell: entries differing | max diff (ulp of 1.0) |
|---|---|---|---|
| 1x64 (gemv path) | 70.1% | 69.4% | 1.00 |
| 4x64 | 0.0% | 55.2% | 0 / 1.00 |
| 4x7, 4x3, 7x5 | 1.0-1.2% | 12-13% | 0.5-1.00 |
| 17x33 | 0.0% | 28.4% | 0 / 1.00 |

Every variant is within 0.93 ulp(1.0) of the float64 value, so none is "more correct". Even on the AVX-512 laptop, one-row products differ from 64x64 ones in 70% of entries; the bitwise contract held there only because the test shapes happened to hit equivalent kernels (the existing `chunks_within_the_block_budget` test already conceded this). Thread count was not varied (products are tiny); the cause is kernel selection, shown by `OPENBLAS_CORETYPE`.

**Derived tolerance.** For any summation order, |fl(a.b) - a.b| <= gamma_d * sum|a_i b_i| <= gamma_d |a||b|, with gamma_n = n u / (1 - n u), u = 2^-24. Two orderings differ by at most 2 gamma_d |a||b|. Max is 1-Lipschitz in the sup norm (this also covers a different argmax when two vectors tie within the error), so each token's best score moves by at most 2 gamma_d |q_t| B, B = largest document-vector norm. The float32 sum over T tokens adds at most 2 gamma_{T-1} sum_t|x_t|. Per query: **tol = 2 (gamma_d + gamma_{T-1}) B sum_t |q_t|**. For unit vectors with T=4 this is 9.5e-6 at d=16 and 6.3e-5 at d=128. Observed maxima were 1.9e-6 (d=16, binary, Haswell) and 3.8e-6 (d=128, binary), so the bound is rigorous but 3-20x pessimistic, as worst-case bounds are.

**Rankings can change.** flips.py: 1500 queries x 200 candidates from 400 docs, Haswell kernel, d=16: 2/1500 (float), 1/1500 (int8) and 3/1500 (binary) queries have a different candidate order; none within the top 10; at d=128 none. Every disagreement was between candidates whose scores differ by less than 2*tol (0 violations). Default kernel: 0 differences. Binary-quantised data has many true ties, which a 1-ulp perturbation reorders.

### A05-03 Making production bitwise batch-invariant: NOT RECOMMENDED
It would need a fixed operation sequence independent of shape (bypass BLAS: einsum or sequential FMA over d, materialising rows x len x d). That removes the 5.5-9.3x rescoring speedup the change was made for, scales badly at d=128-2560, and still would not match v0.3.0's per-query result (itself dependent on the total length of the candidate set) or the pinned `maxsim_matrix`. The correct contract is agreement within the derived tolerance and the same ranking except between scores closer than 2*tol. No production logic change; only the docstring in storage.py now says scores may differ in the last bits.

### A05-04 (LOW, PROPOSED) Docs claim "bit-identical" for batched rescoring
`docs/package/BACKEND_ARCHITECTURE.md:178` ("bit-identical rankings"), `RETRIEVAL_BACKEND_AUDIT.md:214,404` and the 198f83a message hold on this stack only. Wording should become "identical on this stack; equal within float32 rounding in general". Not edited (outside scope). `BUILD_AND_PORTABILITY_REPORT.md:145` already records the CI failure.

## 3. Fix (commit 9abd097 on harden/a05)
- tests/test_storage.py: the 8 failing tests (the parametrised one is renamed `..._agrees_with_the_per_query_rescore`) call `_assert_scores_agree(got, want, _rescore_tolerance(idx, queries))`: per-query derived bound, -inf positions must match exactly, ranking equal except within 2*tol. `chunks_within_the_block_budget` (rtol=1e-6 with no atol, brittle near zero) uses the same helper. The reload-vs-fresh bitwise assertion is kept (same code and shapes on both sides, genuinely deterministic).
- New `test_agreement_check_rejects_real_batching_errors` (rolled columns, rolled queries, constant offset, 10x tol are rejected; 0.5x tol is accepted).
- Verification: 22/22 pass under default, Haswell, Zen, Nehalem and Prescott kernels; ruff clean. Mutation check on storage.py, each run under default and Haswell kernels, all CAUGHT: token-slice off-by-one in the sum (17 failed), reversed column index within a chunk (11), column index shifted (14), max replaced by min (17), chunk loop starting at 1 (18).
- Regression risk: tests only; the bound is about 1e-5 against logic errors that are typically 1e-2 or larger.

## 4. What I could not check
- The CI runner's CPU model (AVX2 inferred from exact numeric reproduction).
- Linux itself, a thread-count sweep, real encoder data; the rank-flip experiment is synthetic.
- Other bitwise-equality tests (`tests/test_core_retrieve.py:420` T9 "bitwise on this stack" and the exact comparisons listed in BUILD_AND_PORTABILITY_REPORT.md:142-146) may be similarly kernel-dependent. The latest CI run has only these 8 red, so they apparently pass there, but I did not test them under other kernels.
- The whole test suite was not run.
