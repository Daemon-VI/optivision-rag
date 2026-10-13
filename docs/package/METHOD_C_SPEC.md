# Method C: implementation specification (frozen semantics)

Status: draft for implementation review, 2026-10-06. Worktree `optivision-package`, branch
`package-universal`, HEAD `91564fd`. Purpose: re-implement Method C as a clean pure-NumPy module
inside `src/optivision/` with **no change of semantics**, and prove it equal to the frozen
functions. This file is a specification, not code, and not a statement of novelty.

**Positioning (mandatory wording rule).** Method C is an assembly of published tools: exact
p-values by *testing by betting* (Waudby-Smith and Ramdas, JRSSB 86(1), 2024; the frozen code
says "2023", the online-first year, `R/e1_stats.py:16`), *fixed-sequence* multiple testing in
pre-specified order (Westfall and Krishen 2001; learn-then-test style, Angelopoulos et al. 2021),
and a union bound over chains. It must never be described as a new statistical method
(`PRE:1161-1164`, `PRE:1562`, `paper/optivision_research.tex:389-392`,
`reports/research/PAPER_SUMMARY.md:65-68`, `reports/research/Q10/REPORT.md:113-121`).

## 0. Conventions, sources, identity

**Citation prefixes.** `R/` = `scripts/research/`; `PRE` = `reports/research/NEXT_PHASE_PREREGISTRATION.md`
(amended text, blob `445d94cd`); `DES` = `reports/research/E1/DESIGN.md`; `RES` =
`reports/research/E1/RESULTS.md`; `FRZ` = `reports/research/E1/E1_2_FREEZE.md`; `HAR` =
`reports/research/X_HARNESS.md`; `XRES` = `reports/research/X_RESULTS.md`; `LIB` = `src/optivision/`.
Citations are `file:line` (inclusive ranges).

**Evidence tags.** `[CODE]` read in code. `[DOC]` stated in a research document. `[VERIFIED]` the spec
author ran the frozen pure functions on synthetic arrays, or read candidate metadata (sidecar JSON);
no per-query store value was read and no experiment script was run. `[INFERENCE]` spec-author
reasoning, not stated in the research documents. `[AMBIGUOUS]` code or documents are silent or
inconsistent: a decision is needed (collected in section 11).

**Source priority.** Semantics: the frozen code (`R/e1_stats.py`, `R/e1_1_study.py`, wrapper
`R/x_common.py:940-948`), then `FRZ`, then `PRE` section 14.2. Assumptions and claims: `PRE` section 16.2,
`DES`, `RES`. `FRZ:18`: "No parameter, ordering, bet, level or tie-break may change after this point."

**Identity of the frozen code** `[VERIFIED]` (`git hash-object` of the worktree files equals the
registered blobs, `PRE:1415-1417`, `R/x_common.py:80-82`): `R/e1_stats.py` `13494307a4a42cd2c1da63f34aa0952a92b33130`;
`R/e1_1_study.py` `5896e763b49d4a0f3f0cc988dc2b5ec5e6b2b9aa`; `R/e1_common.py`
`3bef181683307eb98a9c5e2d61ab02d87801bbfe`. Freeze commit `106ba510538cf1fb8cd8f1be39e665ed59b0dca9` (`PRE:1386`).

### 0.1 The procedure on one screen

```
inputs : c[K,n], b[n] in [0,1] float64 ; families[K] ; bytes[K], vectors[K] ; T ; delta
Y[k,i] = (c[k,i] - T*b[i] + T) / (1 + T) ;  m0 = T / (1 + T)
p[k]   = min(1, 1 / max_t K_t(Y[k,:]; m0))     # betting wealth, aGRAPA bets, cap 0.9, queries in given order
level  = delta / F ,  F = number of distinct family labels in the list
rejected = []                                   # "certified" candidates
for k in list order:  f = families[k]
    if f already stopped: skip
    if p[k] <= level: rejected.append(k)  else: stop family f   # fixed-sequence, per chain
k_hat = argmin over rejected of (bytes[k], vectors[k], k)      # none rejected -> abstain (uncompressed index)
```

### 0.2 What is NOT Method C (do not import, reuse or mix)

Rule A / `calibrate()` (`R/e1_stats.py:71-115`, `LIB/calibration.py:413-531`: bootstrap lower bound,
`split_queries`, early stop on the bound, `walk_select`), A-bonf, **B** (A plus one test on a hold-out
half, `R/e1_1_study.py:21-23`), **C-split** (order by half-sample point retention, `R/e1_stats.py:132-139`,
`R/e1_1_study.py:128-131`), **D** (Bonferroni over K, `R/e1_1_study.py:132-134`), **I8**
(single test of position 1, `R/x_common.py:1003-1004`). `fixed_sequence` (`R/e1_stats.py:132`) is C-split's, not C's.
Method C contains no bootstrap, no seed, no random number, no split.

## 1. Inputs

### 1.1 Per-query arrays

| name | shape | dtype | meaning | source |
|---|---|---|---|---|
| `c` | (K, n) | float64 | `c[k,i]`: per-query metric of candidate k on query i | `R/e1_common.py:78`, `R/e1_stats.py:3-5` |
| `b` | (n,) | float64 | `b[i]`: the same metric for the uncompressed (float32) index on the same query | `R/e1_common.py:77` |

- **Metric.** Research: nDCG@5 (ViDoRe stores), nDCG@10 (SciFact), binary relevance, "as stored"
  (`PRE:154`; `LIB/evaluation.py:55-82`; `R/e1_0_rebuild.py:49-54`). Statistically any per-query score
  in [0,1] is admissible; only nDCG@k is *validated* [INFERENCE]. `c` and `b` must use the same metric, cut-off,
  reference and query set. Two constructs define `(b, c)` differently (section 8).
- **Range.** float64 in [0, 1], inclusive; `b[i] = 0` allowed; `c[k,i] > b[i]` allowed (retention can exceed 1,
  `PRE:159`); mean of `b` must be positive (assumption A4, `PRE:1176`, `DES:95`). Then `Y` lies in [0,1]
  because `c - T*b in [-T, 1]` (`R/e1_stats.py:7-9`).
- **Finite.** Frozen functions do **not** check finiteness or range; the harness does (`R/x_common.py:886-897`,
  gate G2; `R/e1_1_study.py:199-200`; `PRE:1241`: a non-finite value invalidates the whole store).
- **Dtype.** Stores are float64 and the harness requires it (`R/x_common.py:535-540`). `to_y` does not cast
  (section 2.4). Cast to float64 **before** any arithmetic.

### 1.2 Candidate list and attributes

| attribute | shape | meaning and use | source |
|---|---|---|---|
| position `k` | implicit 0..K-1 | index in the flattened list (`position` = running count over chains) | `R/x_common.py:453-461`, `LIB/calibration.py:490-492` |
| `families[k]` | K labels (str, hashable) | chain id; equality only | `R/e1_common.py:81`, `R/e1_1_study.py:63-70` |
| `bytes_per_doc[k]` | K float64 | stored bytes per document = `CompressedCorpus.nbytes / n_docs`; used only to choose among certified candidates | `LIB/compose.py:93-96, 108`; `R/e1_common.py:82` |
| `vectors_per_doc[k]` | K float64 | vectors kept per document; second tie-break | `LIB/compose.py:104`; `R/e1_common.py:83` |

Not used by the decision: `compression_vs_float32` (`LIB/compose.py:109`; reporting, `R/x_common.py:1070-1072`),
`label`, `pipeline` (identity hash only), `dim`, `bits_per_dim`.

- **Frozen before the calibration queries are seen**: the list, its order, its family labels, and every fitted
  stage (fitted on document vectors only) (`PRE:1175`, `DES:92-94`, `RES:146-147`).
- **S0 (the default list of the research)**: 40 candidates, 7 chains, 37 distinct pipelines; sha256 of its
  specification `af29b01146a90a048ffea9cfbd66f0fedf988134829cd4dc56861f1207893520`
  (`R/x_common.py:60-63`, `PRE:1390-1397`). It exists only with SciPy installed; without SciPy the list has 4
  chains and is **not** S0 (`PRE:1004-1005`, `LIB/calibration.py:346-361`). Layout `[CODE]` `[VERIFIED]` on all seven sidecars:

| chain | family label | positions | members, least to most aggressive |
|---|---|---|---|
| 1 | `float16` | 0 | float16 |
| 2 | `ward+int8` | 1-6 | int8(per_vector); Ward ratio 0.5, 0.33, 0.25, 0.15, 0.1 each then int8 |
| 3 | `adaptive_centred+int8` | 7-13 | int8(per_vector); centred adaptive merge radius 0.9 .. 0.4 then int8 |
| 4 | `ward+int4` | 14-19 | int4(mean); Ward ratios then int4 |
| 5 | `adaptive_centred+int4` | 20-26 | int4(mean); radii then int4 |
| 6 | `ward+binary` | 27-32 | binary; Ward ratios then binary |
| 7 | `adaptive_centred+binary` | 33-39 | binary; radii then binary |

  Duplicated pipelines (identical stored size and identical `c`): positions (1,7), (14,20), (27,33)
  (`PRE:145`). In all seven sidecars bytes and vectors are non-increasing along every chain, chains are
  contiguous, and the only exact `(bytes, vectors)` ties are those three pairs `[VERIFIED]`.

### 1.3 Parameters

| parameter | domain | research grid | notes |
|---|---|---|---|
| `T` target | (0, 1] `[AMBIGUOUS]`: frozen code has no check; `calibrate()` requires (0,1] (`LIB/calibration.py:449-450`) | labels 0.99, 0.97, 0.95; agreement also 0.90 (`R/x_common.py:235-236`) | Python float / float64 only |
| `delta` | (0, 1) `[AMBIGUOUS]` | 0.05 primary, 0.01 (`R/x_common.py:237-238`) | one fixed (T, delta) per calibration sample, see 6.3 |
| `BET_CAP` | constant 0.9 | 0.9 | not a parameter; frozen at design from synthetic power checks (`DES:164-167`, `R/e1_stats.py:25-28`) |
| `n` | number of columns | 50, 100, 225 (E1); 450..3600 (X2); full pool | no minimum in the frozen code; p = 1 for n = 0 |
| seed | none | - | Method C is deterministic |

### 1.4 Query order and exchangeability requirement

- **Order used.** The order of the last axis, as given (`R/e1_stats.py:48-51`). The harness keeps the drawn
  order of a resample and never rearranges it (`PRE:522`, `R/e1_1_study.py:215-217`); the full-pool run uses
  the stored order (`PRE:1177`, `R/x_common.py:945-948`). The module must neither sort, shuffle nor de-duplicate.
- **Requirement.** (A1) calibration queries are i.i.d. draws from one distribution Q; (A5) the order in which
  queries enter the sequential test is fixed before their outcomes are seen (`PRE:1173, 1177`).
  [INFERENCE] Under i.i.d. sampling any outcome-independent order is valid (arrival order, or a permutation drawn from a seed
  independent of the data and recorded). An order that depends on `b`, `c`, scores, or on trying several orders and keeping the best p-value, is not.
- **The p-value is order-sensitive** (TV4: the same multiset gives 4.97e-6 or 1.0). Two analysts with different
  fixed orders get different, individually valid, decisions.
- **i.i.d. is stronger than exchangeable.** The research used sampling *with replacement* from a finite pool so that Q is
  exactly the pool (`R/e1_1_study.py:5-13`); it states that a real calibration set is a sample without replacement and that near-duplicate
  queries would weaken i.i.d. (`RES:191-193`). Sampling without replacement from a finite population, sessions
  and time-ordered streams with drift are **not covered** by any document `[AMBIGUOUS]`.

### 1.5 Input validation the module must add (the frozen functions have none)

Raise (never coerce, never drop) on: wrong shapes (`c` (K,n), `b` (n,), lengths of `families`, `bytes`,
`vectors` equal K), K < 1, any NaN/inf in `c`, `b`, `bytes`, `vectors`; any value outside [0,1]; `mean(b) == 0`;
`T` not in (0,1]; `delta` not in (0,1). Queries without a relevant document (metric NaN, `LIB/evaluation.py:55-58`)
are removed **by the caller, for all candidates and the baseline together, by a label-only rule**; per-candidate
dropping (as `LIB/evaluation.py:93-94` does) would give candidates different query sets and is forbidden.
`n = 0`: frozen returns p = 1.0 (abstain); either abstain or raise, but pick one and test it `[AMBIGUOUS]`.

## 2. Transformation to Y and the betting p-value

### 2.1 Formulae

- **Identity** (`R/e1_stats.py:7-13`, `DES:51-54`): with `R_Q(k) = E_Q[c_k] / E_Q[b]` and `E_Q[b] > 0`,
  `R_Q(k) >= T  <=>  E_Q[c_k - T*b] >= 0`.
- **Statistic** (`R/e1_stats.py:43-44`): `Y = (c - target*b + target) / (1.0 + target)`; evaluate in exactly this
  operation order (section 10, item 2). **Null level** (`R/e1_1_study.py:114`, `R/x_common.py:947`): `m0 = T / (1 + T)`.
- **Hypothesis**: `H0_k: E_Q[Y_k] <= m0` (equivalently `R_Q(k) <= T`); rejecting H0_k is the claim `R_Q(k) > T` (`R/e1_stats.py:11-13`).
- Corner values at T = 0.95 `[VERIFIED]`: (c,b) = (1,1) gives Y = 0.5128205128205129 (> m0); (0,1) gives 0.0; (1,0)
  gives 1.0; (0,0) gives exactly `m0` = 0.48717948717948717 (no evidence either way).

### 2.2 Algorithm (`R/e1_stats.py:47-68`; time index t is 0-based, state is *before* observation t)

```
s1 = sum_{s<t} Y_s          s2 = sum_{s<t} (Y_s - mu~_s)^2 ,  mu~_s = (0.5 + sum_{r<=s} Y_r) / (s + 2)
mu_t   = (0.5 + s1) / (t + 1)          # prior mean 1/2 counts as one pseudo-observation
var_t  = (0.25 + s2) / (t + 1)         # prior variance 1/4
d      = mu_t - m0
lam_t  = clip( d / (var_t + d*d), 0, BET_CAP / m0 )          # aGRAPA bet, predictable
log K_t = log K_{t-1} + log1p( lam_t * (Y_t - m0) )          # log K_0 = 0
p      = min(1, exp( - max(0, max_t log K_t) ))               # running max includes the starting wealth 1
```

- The bet at time t uses only `Y_0..Y_{t-1}` (predictable). The variance update uses the mean **including**
  the current observation, denominator `t+2` (`R/e1_stats.py:65-67`). `log_k` is updated before the running
  maximum (`R/e1_stats.py:63-64`).
- Validity logic (`R/e1_stats.py:15-28`): `E[1 + lam_t (Y_t - m0) | past] <= 1` under H0 and i.i.d. sampling, so
  `K_t` is a non-negative supermartingale; Ville's inequality gives `P(max_t K_t >= 1/delta) <= delta`. The factor
  `1 + lam (Y - m0) >= 1 - 0.9 = 0.1 > 0` for `Y in [0,1]` because `lam * m0 <= 0.9`. Validity holds for any cap in (0,1); 0.9 was
  chosen from synthetic power checks (c = 0.5 cannot certify even a lossless candidate at T = 0.97 with 225 queries, `DES:157-167`).
- It is the **running maximum** over t (anytime-valid), not the final wealth.
- Shapes: leading axes broadcast, time is the last axis; 1-D input returns a scalar `np.float64`.
- Rejection compares `p <= level` (inclusive, `R/e1_stats.py:125`).

### 2.3 Properties used as tests (all `[VERIFIED]`, section 9)

`p` in (0, 1]; `p = 1` if wealth never exceeds 1 (e.g. `Y` identically 0, or identically `m0`); `p` is
non-increasing when observations are appended (prefix property, `PRE:579-584`, `R/x_common.py:1192-1214`); order-sensitive;
no interaction between candidates (`p(c[perm]) = p(c)[perm]`); no randomness.

### 2.4 Edge cases, with frozen behaviour (`[VERIFIED]`) and the required behaviour of the module

| case | frozen behaviour | module |
|---|---|---|
| ties between Y values or p-values | no special treatment; `p == level` is a rejection | same |
| `b[i] = 0` | `Y = (c+T)/(1+T) >= m0`: a non-negative contribution; query kept (`PRE:1240`) | same |
| all `b = 0` (mean 0) | no error; R undefined (0/0) | raise (A4) |
| NaN in `y` | `p = NaN` silently, no warning; `NaN <= level` is False, so the chain **stops silently** | raise before computing |
| inf in `y` | `p = NaN` with RuntimeWarning | raise |
| `y < 0` after a winning streak | `log1p` argument <= -1: `p = NaN` with RuntimeWarning | raise (range check) |
| `y > 1` (e.g. constant 2.0) | returns a plausible small p (no error); validity argument lost | raise (range check) |
| `n = 0` | `p = 1.0` | see 1.5 |
| `float32` inputs | `to_y` stays float32 (NumPy keeps the Python-float scalar weak); in a probe the p-value changed by 1.7e-6 relative; `betting_pvalues` casts only afterwards | cast to float64 first; test `to_y(...).dtype == float64` |
| Python list inputs | `to_y` raises `TypeError` | accept via `np.asarray` |
| `T` as `np.float32` | the float32 rounding of T changes m0 and Y | convert to Python float / float64 |
| bit reproducibility | on numpy 2.4.6, Windows, batch shapes 1..9 rows and 3-D stacks gave bitwise-identical p; the harness still treats vectorised arithmetic paths as platform-dependent (`R/x_common.py:267-270`) | compare p at `rtol=1e-12`; decisions exactly |

## 3. Families, chains, error split, fixed-sequence rule

- **Family = chain.** Candidates sharing a label, in **list order**. The within-chain order is the order of appearance
  in the list; it is the pre-specified order of the fixed-sequence test and it is *not* sorted by bytes, by `c` or by anything
  data-dependent. S0 builds it "from least to most aggressive" (`LIB/calibration.py:26-29`, `PRE:144`); validity does not
  depend on that order being monotone, power does `[INFERENCE]` (`DES:81-85`: a false rejection needs the first true null of the order rejected).
- **Origin of the grouping and of the order** `[CODE]`: the dict keys and list order of `recommended_search_space(dim)` (`LIB/calibration.py:352-360`; Python dicts keep insertion order) are flattened chain by chain
  (`R/x_common.py:453-461`), stored as `family` and `position` in the sidecar (`R/e1_0_rebuild.py:60`) and read back as `Problem.families` (`R/e1_common.py:81`). There is no other ordering key: chain order = list order.
- **A candidate that cannot be evaluated** (error, out of memory) is "recorded; not rejected; its chain stops there; never replaced" (`PRE:1236`; the X3-only amendment F1, `PRE:1731`, does not apply): treat it as `p = 1`.
  Dropping or substituting a candidate *after* seeing the calibration queries is forbidden; a list fixed in advance and passed whole is what the statement covers (A3). A caller-side subset of S0 changes `F` and the list hash.
- **Interleaving is harmless.** The per-family `stopped` flag makes chains independent, so a list whose chains are interleaved
  (within-chain order preserved) gives the same rejected set `[VERIFIED]`; S0 is contiguous.
- **Error split.** Equal split: `level = delta / F`, `F = len(dict.fromkeys(families))`, the number of **distinct family labels in the
  list passed** (`R/e1_1_study.py:70`, used at `R/e1_1_study.py:125` and `R/x_common.py:942`; true division). Not `delta/K`
  (that is D), no weights, no recycling of unused level, no re-count of "active" chains. The `float16` chain counts: S0 has F = 7 although
  chain 1 has one member; dropping it would change the level to `delta/6` (changes semantics).
- **Fixed-sequence rule, exactly** (`R/e1_stats.py:118-129`):

```
for k in 0..K-1 (list order):  f = families[k]
    if f in stopped: continue                  # p[k] is ignored, however small
    if p[k] <= level: rejected.append(k)
    else: stopped.add(f)                       # first non-rejection ends this chain only
```

  Consequences `[VERIFIED]`: stop at the first non-rejection **of that chain**; later members are never rejected even with `p = 0`;
  chains are independent; a NaN p stops the chain; the same `level` applies at every step (no alpha propagation);
  position 2 with `p = 0` but position 1 with `p = 1` is unreachable.
- **Certified = rejected.** The certified set is the union of the per-chain rejected prefixes. [INFERENCE] Because the FWER
  over all K nulls is controlled (`DES:81-85`), with probability >= 1 - delta *every* certified candidate has `R_Q > T` simultaneously; the module may therefore
  expose the whole certified set, and any rule that picks a member of it keeps the statement.
- **Duplicates** across chains are both tested and not merged (they have identical p). Candidates are never de-duplicated.

## 4. Choice among certified candidates; deployment, fallback, abstention

- **Choice** (`R/e1_stats.py:142-146`, `98-99`; `FRZ:16`; `PRE:1043, 1234`): `k_hat = argmin_{k in rejected} (bytes[k], vectors[k], k)`:
  smallest stored size; ties by fewer vectors; then earlier position (= smaller list index). Comparisons are **exact float equality**,
  no tolerance, no rounding. Chain position and compression ratio play no role. Key unique, so no further ties.
  (Rule A's `rank_key` uses the position among the *judged*, `R/e1_common.py:92-94`; the order of positions is the same, but Method C uses `k` itself.)
- **Nothing certified**: `rejected == []`, `smallest` returns `None`, stored as `-1` in the selection arrays
  (`R/e1_1_study.py:127`, `R/x_common.py:958, 998`). The procedure returns the **uncompressed index**; compression is 1.0.

| term | in code and documents | meaning |
|---|---|---|
| deploy | `k >= 0` (`R/e1_1_study.py:158`, `R/x_common.py:1077`); `DES:63-64` | a compressed candidate is returned |
| abstain | `-1` / `None`; `PRE:176` "returns the uncompressed index. Compression is 1.0 when it abstains"; `PRE:1043` "Otherwise abstain" | the uncompressed index is returned |
| fallback | only in `R/e1_1_study.py:21-27`: B "else fall back to float32 (abstain)"; C "else float32"; `FRZ:16` "otherwise float32" | **a synonym of abstain**; it is the uncompressed float32 index |
| certified | member of `rejected` | `p` reached through the chain with `p <= delta/F` |
| non-trivial deployment | `dep & (k != float16_index)` (`R/x_common.py:1080`); `PRE:177` | deployed candidate other than `float16` |
| miss | `dep & (truth < target)` (`R/x_common.py:1081`, `R/e1_1_study.py:162`); `PRE:180` | compressed candidate deployed and its (pool) retention below T |

- **There is no float16 fallback** in the code or in any research document. `float16` is candidate 0 / chain 1 and must itself be
  certified (`p_0 <= delta/F`) to be deployed; "non-trivial deployment" is defined as "a deployed candidate other than `float16`" (`PRE:177`), i.e. float16 (2x) is the smallest possible deployment, not a safety net. On abstention the
  procedure returns the uncompressed index. `[AMBIGUOUS]` if the package wants an uncertified float16 default when nothing is certified, that
  is **new semantics**: it must be reported as a separate status (e.g. `fallback_uncertified`), never counted in `certified`, never covered by the
  guarantee text. Float16 measured 99.7-100% retention on the development corpora (`docs/UNIVERSAL.md:254, 557`) but that is an empirical figure, not a certificate, and on one model the per-vector cosine minimum was 0.923 (`docs/UNIVERSAL.md:870-872, 1081`: "float16 encoding is not free").
- **Is the fallback certified?** The uncompressed index is the reference itself (R = 1 under labels, R_A = 1 under agreement); no hypothesis is tested
  for it and a miss requires a *compressed* deployment, so abstention can never be a miss. Abstention is the safe outcome, not an error: the research reports
  frequent abstention (at n = 50 and 100 no valid method deploys in more than 0.25% of resamples; none can be expected to certify T = 0.99 with <= 225 queries, `RES:70-71, 171-173`; see section 6.3).

## 5. Data splitting, calibration, what may be looked at

- **No split, no screening, no calibration step.** All `n` queries are used for every candidate; "all n; no hold-out split"
  (`R/x_common.py:668`, `PRE:1045`). Nothing is fitted on the calibration queries: the only adaptive element is the bet `lam_t`,
  computed from `Y_0..Y_{t-1}` of the same candidate's sequence (predictable; part of the test statistic, not a selection step). `split_queries`
  appears in `run_cell`/`simulate` only for A, B and C-split (`R/e1_1_study.py:82`, `R/x_common.py:969`).
- **Fixed before the calibration queries are seen** (fixed-sequence validity needs the order independent of the data it is tested on):
  the candidate list, the family labels and chains, the within-chain order, `F`, the fitted stages (document vectors only, `Pipeline.compress(corpus)`),
  `T`, `delta`, the bet cap, the query order. `PRE:1175, 1177`; `DES:92-94`; `RES:146-147`; `paper/optivision_research.tex:382-383` ("stages fitted without calibration queries").
- **May be looked at before** (kind A/B/C knowledge, `PRE:1100-1105, 1121-1126`): the corpus and label-free statistics of document vectors computed by a frozen function;
  dimension, vectors per document, stored sizes; results of earlier experiments on *other* data and encoders (S0 itself was built from development-encoder results and is declared as such, `PRE:1123-1126, 1128-1138`);
  model cards. A single entry point and one frozen list, no human routing by encoder (`PRE:1107-1118`).
- **Must not be looked at / used** `[INFERENCE]` consistent with `PRE:1175`: anything computed from the calibration queries or their results, including scores, rank lists,
  per-query metrics of baseline or candidates, "pilot" runs on the same queries, and which candidates "look good". Do not choose `T`, `delta`, the list, chain order or query order,
  and do not re-run with an amended list, on the same sample after seeing outcomes.
- **Seeds**: none in the procedure. A seed appears only in the harness (resampling indices `default_rng(seed).integers(0, N, size=n)`, `R/e1_1_study.py:215-217`) and in the test vectors.

### 5.1 Model-blindness (`PRE:1089-1146`)

- **Definition** (`PRE:1093-1096`): "everything it does is determined by (a) what was frozen before any output of that encoder was seen, and (b) the vectors and queries it is given. It receives no
  identifier of the encoder, and no person chooses anything on its behalf after seeing which encoder it is." Four checkable tests (`PRE:1107-1118`): **no identity** (inputs are arrays; removing every name and provenance field
  leaves the output unchanged), **no human routing** (one entry point, one frozen candidate rule), **frozen first** (the procedure's commit predates the first encoding of the encoder), **no retuning**. The property is "relative to an encoder and a date" (`PRE:1118-1119`).
- **For the module**: its output is a function of `(c, b, families, bytes, vectors, T, delta)` only; family labels are grouping keys, compared for equality (relabelling them bijectively must not change the result, TV10); the module must not branch on encoder
  name, dimension, modality or dataset, and must not carry per-encoder defaults for `T`, `delta`, the cap or the list. Which list is passed is the caller's choice; the *wide* space is a separately registered arm, never a per-encoder switch (`PRE:1111-1113`).
- **Encoder knowledge already inside S0** is "kind C" and declared (`PRE:1121-1126`): S0 contains centred int4 and plain binary because of results on ColPali and the text ColBERT. **P0 is not a held-out test on any development encoder**; applying it to those encoders again "says nothing about model-blind transfer" (X1-X3 do exactly that: they are development analyses, `PRE:1130-1137`). XH (held-out encoders) was not run (`XRES:235`).
  Model-blindness is **not** invariance to symmetries of the representation (`PRE:1140-1144`; X3).

## 6. The statistical statement (a conditional, finite-sample guarantee, not externally reviewed)

### 6.1 Statement, verbatim

`PRE:1161-1169`:

> Method C is an application of known tools: fixed-sequence testing with a union bound over chains, and betting p-values (Waudby-Smith and Ramdas, 2023). It is not presented as a new statistical method. Its statement is
> P( a compressed candidate is deployed and its retention under Q is below T ) ≤ δ,
> where the probability is over the draw of the calibration queries. It holds for every sample size, with no large-sample approximation, and it is conditional on all of the following.

`RES:150-157`: "Under these, the procedure satisfies, for every sample size n, `P( a compressed candidate k̂ is deployed  and  E_Q[c_k̂] / E_Q[b] < T ) ≤ δ.` Here the probability is over the draw of the calibration queries. This is a finite-sample, distribution-free statement about the selected configuration (INFERENCE, proof in `DESIGN.md`)."

- **Controlled quantity**: the probability of the **joint** event {a compressed candidate is deployed} and {its retention `R_Q = E_Q[c]/E_Q[b]` is below T}; level `delta`; over the draw of the i.i.d.
  calibration queries only. The corpus, list, order and fitted stages are fixed, not random. Per fixed `(T, delta)`. [INFERENCE] Equivalent wording: with probability >= 1 - delta the
  procedure abstains or deploys a candidate with `R_Q >= T`. It is **not** `P(R_Q < T | deployed) <= delta` and not a statement about the sample retention.
- **Proof status**: the argument is the project's own (`DES:77-88`, labelled INFERENCE/proof): validity of each p-value (Ville), fixed-sequence FWER control inside a chain, union bound over F
  chains at `delta/F`, and "If a deployed k̂ has R(k̂) < T, then H0_k̂ is true and was rejected, so the event lies inside the FWER event" (`DES:84-85`).
  "Its application here was not checked against the statistics or learn-then-test literature" (`reports/research/Q10/REPORT.md:220-221`): obtain an external statistical review before any release wording says "guarantee".
- **Approved wording** (`reports/research/Q10/REPORT.md:119-120`, `paper/optivision_research.tex:389-392`): "a validated application of known methods, providing a conditional, finite-sample guarantee"; never "guaranteed retention" without its conditions (`Q10/REPORT.md:247-250`).

### 6.2 Assumptions (all required)

| id | assumption | source |
|---|---|---|
| A1 | calibration queries are i.i.d. draws from one distribution Q (the deployment distribution) | `PRE:1173`, `DES:91` |
| A2 | the corpus is fixed (retention moves with corpus size, R10) | `PRE:1174`, `DES:99-100` |
| A3 | candidate list, chains and order fixed before the queries are seen; every fitted stage fitted on documents only | `PRE:1175`, `DES:92-94` |
| A4 | per-query metric in [0,1]; `E_Q[b] > 0` | `PRE:1176`, `DES:95` |
| A5 | the order in which queries enter the sequential test is fixed before their outcomes are seen | `PRE:1177` |
| A6 | the reference is taken as given: labels under L, the uncompressed index's top document under A; label errors are not modelled | `PRE:1178`, `DES:96` |

Documented consequence of a failure (`PRE:1173-1178`, third column): A1 "reported as a decision with no checked error level"; A2 "outside the statement"; A3 "the candidate is removed from the certified analysis and reported"; A4 "the store is invalid"; A5 "reported without an error level"; A6 not testable.

### 6.3 What is NOT guaranteed

- Anything about a distribution other than Q, a different or growing corpus, a different corpus size (`DES:99-100`, `PRE:1563-1565`, `RES:177-179`).
- The realised retention on a finite batch of future queries (it fluctuates around R_Q); anything per query; anything conditional on the sample at hand: the statement is marginal over calibration samples (`DES:101-105`, `PRE:1180-1182`).
- Any property of A / `calibrate()` / `optimize()` (`PRE:1566`, `RES:168-170`).
- **Power.** The guarantee holds even if the procedure always abstains. No valid test can reject a lossless candidate with probability >= 1/2 at T = 0.99, delta = 0.05 with fewer than 229 queries (`DES:109-117`, `RES:171-173`; `STATISTICAL_REVIEW.md`, W12); a lossless candidate needs
  `n >= log(1/(2 delta)) / (-log(1-eps))`, `eps = beta (1-T) / (T + beta (1-T))`, `beta = E_Q[b]` (`DES:112-117`). Deterministic best case (b = c = 1) for the frozen test: TV3.
- **Agreement is not retrieval quality**: certifying `R_A >= T_A` says nothing statistical about `R_L` (the cross-construct miss rate "is an empirical test", `PRE:447-451`; section 8).
- The coordinate basis: H5 failed under two registered rotations (`XRES:115-129, 143-147`): what is certified is the retention of the actual pipeline on the actual stored vectors.
- **Transfer to unseen encoders**: untested. S0 was shaped by development encoders (`PRE:1121-1138`); X1-X3 are development analyses; XH was not run (`XRES:235`). "Model-blind" describes the procedure's inputs (5.1), not an empirical transfer result.
- **Several (T, delta) from one sample** and re-analysis of the same sample with a changed list are not addressed by any document `[AMBIGUOUS]`; each extra look consumes error level `[INFERENCE]`.
- Winner's curse: the sample retention of the selected candidate is optimistically biased; only the p-values and the statement above are valid `[INFERENCE]`.
- Not a statement about novelty, scalability beyond tested corpora, or other scoring functions than MaxSim (`PRE:1554-1581`, scope `PRE:110-131`).

### 6.4 Evidence in the research (does not widen the claim, `RES:285-287`)

Consistent with the project's derivation (not externally reviewed) in 288 real and 720 synthetic checks (`RES:156-157`); confirmation: 27 of 27 held-back cells, maximum miss rate 0.05%, 95% CI [0.0%, 0.28%] (`RES:251-255`); X1-X3 validity check over 2,800 cells, no confirmed failure (`XRES:102-105`).
All are resamples of one pool per corpus (Q = Q_pool): they check the implementation against the derivation, they do not replace it.

## 7. `method_c_config()` and identification

`R/x_common.py:660-668`, printed verbatim by import `[VERIFIED]` (importing `x_common` runs no gate and wrote no file):

```json
{"source": "scripts/research/e1_stats.py and e1_1_study.py, imported unchanged",
 "hypothesis": "H0_k: E[c_k - T b] <= 0", "statistic": "Y = (c - T b + T) / (1 + T); m0 = T / (1 + T)",
 "p_value": "exact betting p-value, aGRAPA bets", "bet_cap": 0.9,
 "multiplicity": "fixed-sequence testing inside each chain, in the chain's own order",
 "level": "delta / number of chains", "chains": 7,
 "choice": "smallest stored size among the rejected; ties by fewer vectors, then earlier position",
 "deltas": [0.05, 0.01], "queries_used": "all n; no hold-out split"}
```

- **It is not hashed anywhere in the research code.** It is embedded verbatim in every result header (`R/x_common.py:1252`) and echoed by the dry run (`R/x_common.py:1605-1606`).
  Only `bet_cap` is read from the imported module (`S.BET_CAP`); `chains: 7` and `deltas` are literals/constants of the harness.
- `check_method_c()` (`R/x_common.py:671-675`) sets `ok` from `BET_CAP == 0.9`, the E1 `DELTAS`, `TARGETS`, `NS`, `SEEDS`, `A_CONF`; `require_run_allowed` refuses a run if it is false
  ("Method C constants differ from the E1.2 freeze", `R/x_common.py:743-744`). It does not compare `chains` with the actual list.
- **What identifies Method C in the research**: (i) git blobs of `e1_stats.py`, `e1_1_study.py`, `e1_common.py` (`R/x_common.py:80-82`, `check_code` `649-657`; per-run `code_identity` `625-643`);
  (ii) the S0 specification hash (`R/x_common.py:60`, `check_generated_space` `678-686`; `spec_sha256` = sha256 of `json.dumps(spec, sort_keys=True, separators=(",", ":"))` UTF-8, `445-450`; `[VERIFIED]` reproduced from all seven sidecars);
  (iii) the preregistration blob (`R/x_common.py:52`); (iv) `FRZ`.
- Frozen by the code but **absent from the config dict**: prior mean 1/2 and prior variance 1/4, bet `lam >= 0` clipping, running maximum including the initial wealth 1, inclusive `<=`, `level = delta / F` with `F` counted from the list.
- Reference only (spec-author computation, **not a research artefact**): sha256 of the canonical JSON of the dict above (sorted keys, separators `,` `:`)
  `6003720e9323aaedb90ed309de44c1498be298bacc3944dcb0877adc42727b8e`. If the module defines its own config, extend it with the absent fields, hash it with the same canonicalisation, record it with the list hash, numpy version and code version in every result, and test `bet_cap`, `chains`, `deltas` against the values above.

## 8. The two constructs

| | labelled retention (R_L) | agreement with the uncompressed reference (R_A, label-free) |
|---|---|---|
| relevant set | the dataset's labelled documents | the single top document of the uncompressed index, ties by document index (`LIB/evaluation.py:49-52`, `PRE:151`) |
| `b[i]` | nDCG@k of the uncompressed index against the labels (`base_labels`) | exactly 1 by construction (`base_b1`; gate `PRE:409`, `R/x_common.py:895`) |
| `c[k,i]` | nDCG@k of candidate k against the labels (`cand_labels`) | `a = 1/log2(1+r)` if the candidate ranks the reference's top document at rank `r <= k`, else 0 (`PRE:153`; same `per_query_metrics` call with the baseline's top-1 as the only relevant document, `R/e1_0_rebuild.py:44, 51, 53`) |
| cut-off k | 5 on ViDoRe, 10 on SciFact | the same |
| pool quantity | `R_L = sum c / sum b` (ratio of means; can exceed 1) | `R_A = mean(a)` (cannot exceed 1) |
| threshold | `T` in {0.99, 0.97, 0.95} | `T_A` in {0.99, 0.97, 0.95, 0.90} |
| code path | `E.Store(d, "labels")` | `E.Store(d, "baseline@1")` -> keys `base_b1`, `cand_b1` (`R/e1_common.py:76-78`, `R/x_common.py:257`) |

- Method C is the same function in both: with `b = 1`, `Y = (a - T*1 + T)/(1+T)` (`= a/(1+T)` mathematically, **not** bitwise: do not simplify, item 2 of section 10).
  The H0 is `E[a] <= T_A`. `PRE:147, 159-160`: "They are never merged. ... A statement about one is never reported as a statement about the other."
- **Documented caveat: agreement is not retrieval quality.** It measures whether the compressed index still *agrees* with the uncompressed one, "including agreement with its mistakes" (`PRE:156`); "It is not a validation of retrieval quality ... no X1 result may be reported as a statement about relevance" (`PRE:392-394`);
  agreement "rewards fidelity to the reference's errors" (on DocVQA the uncompressed nDCG@5 is 0.58-0.62, `PRE:1483-1485`); non-claim 6 (`PRE:1567`); `LIB/evaluation.py:8-11` ("stricter: it also counts changes of mind that happen to land on another correct page"). Measured on the seven development stores: CP1 held on 21/21 store-threshold pairs and CP2 on 150 cells, but there is no statistical statement behind CP1/CP2 (`PRE:450-451`), and the agreement arm certified at most half the labelled compression in 26 of the 37 pool-supported cells where the labels arm deploys non-trivially (N10 applies, `XRES:111-113, 133-135, 173-177`).
- Only the top-1 reference was studied; no other agreement measure (overlap@k, rank correlation, depth > 1) exists in the stores or the validation (`PRE:464-466`) `[AMBIGUOUS]` for any other definition.
- Gate for the arrays: values on the grid `{0} U {1/log2(1+r)}` (`R/x1_agreement.py:45-59`), `b == 1` everywhere (`R/x_common.py:895`).

## 9. Reference implementation and test vectors

### 9.1 Reference (pure NumPy; every function equals the frozen one line for line)

`[VERIFIED]` bitwise equal p-values and identical decisions to `e1_stats` and `x_common.method_c` in 400 random problems (K 1..44, n in {0,1,2,3,7,25,100,225,450}, arbitrary family interleavings, many exact size ties, T in {0.5..1.0}, delta in {0.2,0.05,0.01}) plus 300 boundary trials around the level.

```python
# spec-test: ref
from __future__ import annotations
import numpy as np

BET_CAP = 0.9                                   # R/e1_stats.py:40

def to_y(c, b, target):                         # R/e1_stats.py:43-44 (same operation order, float64 first)
    c = np.asarray(c, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (c - target * b + target) / (1.0 + target)

def betting_pvalues(y, m0):                     # R/e1_stats.py:47-68; time on the LAST axis
    y = np.asarray(y, dtype=np.float64)
    n, lead = y.shape[-1], y.shape[:-1]
    log_k, log_kmax = np.zeros(lead), np.zeros(lead)
    s1, s2 = np.zeros(lead), np.zeros(lead)
    cap = BET_CAP / m0
    for t in range(n):
        mu = (0.5 + s1) / (t + 1)
        var = (0.25 + s2) / (t + 1)
        d = mu - m0
        lam = np.clip(d / (var + d * d), 0.0, cap)
        yt = y[..., t]
        log_k = log_k + np.log1p(lam * (yt - m0))
        log_kmax = np.maximum(log_kmax, log_k)
        s1 = s1 + yt
        mu_t = (0.5 + s1) / (t + 2)
        s2 = s2 + (yt - mu_t) ** 2
    return np.minimum(1.0, np.exp(-log_kmax))

def fixed_sequence_by_family(p, families, level):   # R/e1_stats.py:118-129
    rejected, stopped = [], set()
    for k, pk in enumerate(p):
        f = families[k]
        if f in stopped:
            continue
        if pk <= level:
            rejected.append(k)
        else:
            stopped.add(f)
    return rejected

def smallest(cands, bytes_, vectors):           # R/e1_stats.py:142-146 (position == index k)
    return min(cands, key=lambda k: (bytes_[k], vectors[k], k)) if cands else None

def method_c(c, b, families, bytes_, vectors, target, delta):
    """c (K,n), b (n,) -> (selected index or None, p (K,), rejected list). Section 1.5 validation omitted."""
    c = np.asarray(c, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    F = len(dict.fromkeys(families))                        # R/e1_1_study.py:70
    m0 = target / (1.0 + target)                            # R/x_common.py:947
    p = betting_pvalues(to_y(c, b[None, :], target), m0)    # R/e1_1_study.py:116
    rejected = fixed_sequence_by_family(p, families, delta / F)   # R/x_common.py:942
    return smallest(rejected, bytes_, vectors), p, rejected
```

### 9.2 Numerical vectors and properties (>= 8; `[VERIFIED]` against the frozen functions; run with the block above in scope)

Tolerance: p-values `rtol=1e-12, atol=0` (bitwise equality was observed on one platform, not promised); rejected lists, selections and booleans exactly.
Executed on 2026-10-06 by extracting the fenced blocks tagged `spec-test` from this file and running `ref`, then `vectors`, then `slow` against `e1_stats` / `x_common` (Python 3.11.9, numpy 2.4.6, Windows): all assertions pass, TV9 equals the frozen functions, `FP_MATCH` all True.
Run from the repository root (the blocks read `reports/research/E1/store/colpali_docvqa.json`, metadata only; `# spec-test: vectors` expects the names of `ref` in scope).

```python
# spec-test: vectors
import json
M0 = lambda T: T / (1.0 + T)
close = lambda a, b: bool(np.isclose(a, b, rtol=1e-12, atol=0.0))

# TV0  m0 values
assert (M0(0.99), M0(0.97), M0(0.95), M0(0.90)) == (0.49748743718592964, 0.49238578680203043,
                                                    0.48717948717948717, 0.4736842105263158)

# TV1  all-ones Y (c=1, b=0 on every query): p by (T, n)
TV1 = {(0.95, 1): 0.9743918053777209, (0.95, 2): 0.6096967495622448, (0.95, 5): 0.1011749929010796,
       (0.95, 10): 0.0038903901009042755, (0.95, 50): 1.0298567486376658e-14, (0.95, 225): 2.287469300795809e-65,
       (0.99, 1): 0.9949751268780463, (0.99, 10): 0.004276708556589193, (0.99, 225): 1.792208764226269e-63,
       (0.97, 5): 0.10395635389076212, (0.90, 5): 0.09445940121038329}
for (T, n), p in TV1.items():
    assert close(betting_pvalues(np.ones(n), M0(T)), p), (T, n)

# TV2  no evidence -> p == 1.0 exactly ; empty sample -> 1.0
assert betting_pvalues(np.zeros(50), M0(0.95)) == 1.0
assert betting_pvalues(np.full(50, M0(0.95)), M0(0.95)) == 1.0
assert betting_pvalues(np.ones(0), M0(0.95)) == 1.0

# TV3  perfect agreement (b=c=1, Y = 1/(1+T)): p at n=225 and the smallest n that rejects at a level
TV3_P225 = {0.99: 0.19649653893886854, 0.97: 0.0031521656162632767, 0.95: 4.601695771539229e-05, 0.90: 7.635302410969988e-10}
for T, p in TV3_P225.items():
    assert close(betting_pvalues(to_y(np.ones(225), np.ones(225), T), M0(T)), p), T
TV3_MIN_N = {  # T: (0.05, 0.05/7, 0.01, 0.01/7)
    0.99: (377, 592, 555, 770), 0.97: (125, 196, 183, 254), 0.95: (74, 116, 109, 151), 0.90: (37, 57, 54, 74)}
for T, ns in TV3_MIN_N.items():
    y = to_y(np.ones(800), np.ones(800), T)
    for level, nmin in zip((0.05, 0.05 / 7, 0.01, 0.01 / 7), ns):
        assert betting_pvalues(y[:nmin], M0(T)) <= level < betting_pvalues(y[:nmin - 1], M0(T)), (T, level)

# TV4  order sensitivity (T = 0.95): same multiset, different p
m0 = M0(0.95)
assert close(betting_pvalues(np.r_[np.ones(20), np.zeros(20)], m0), 4.968445298450026e-06)
assert betting_pvalues(np.r_[np.zeros(20), np.ones(20)], m0) == 1.0
assert close(betting_pvalues(np.tile([1.0, 0.0], 20), m0), 0.9743918053777209)   # == TV1 (0.95, 1)
assert betting_pvalues(np.tile([0.0, 1.0], 20), m0) == 1.0

# TV5  prefix monotonicity: appending queries never increases p (data-independent property)
y = np.clip(np.random.default_rng(7).normal(0.62, 0.2, 225), 0, 1)
pref = np.array([betting_pvalues(y[:m], m0) for m in range(1, 226)])
assert (np.diff(pref) <= 0).all() and pref[-1] < pref[0]

# TV6  to_y corner values (T = 0.95) and dtype
assert tuple(float(to_y(c_, b_, 0.95)) for c_, b_ in ((1, 1), (0, 1), (1, 0), (0, 0))) == (
    0.5128205128205129, 0.0, 1.0, 0.48717948717948717) and float(to_y(0, 0, 0.95)) == m0
assert to_y(np.float32([1, .5]), np.float32([1, 1]), 0.95).dtype == np.float64   # the module must cast; the frozen one returns float32

# ---- S0 metadata (sidecar; metadata only) ---------------------------------------------------
META = json.load(open("reports/research/E1/store/colpali_docvqa.json", encoding="utf-8"))["candidates"]
FAM = [c["family"] for c in META]
BYT = np.array([c["bytes_per_doc"] for c in META]); VEC = np.array([c["vectors_per_doc"] for c in META])
K, F = len(META), len(dict.fromkeys(FAM))
assert (K, F) == (40, 7)

# TV7  decisions from hand-made p vectors on S0 (delta = 0.05, level = 0.05/7 = 0.0071428571428571435)
lvl = 0.05 / F
def decide(over, default, delta=0.05):
    p = np.full(K, default)
    for k, v in over.items(): p[k] = v
    rej = fixed_sequence_by_family(p, FAM, delta / F)
    return smallest(rej, BYT, VEC), rej
assert decide({}, 0.0)[0] == 39 and len(decide({}, 0.0)[1]) == 40            # all rejected: smallest bytes (1081.408)
assert decide({38: 1.0}, 0.0)[0] == 32 and len(decide({38: 1.0}, 0.0)[1]) == 38   # chain 7 stops at 38; 39 unreachable
assert decide({1: 0.0, 7: 0.0}, 1.0) == (1, [1, 7])      # exact tie (bytes, vectors): earlier position
assert decide({14: 0.0, 20: 0.0}, 1.0) == (14, [14, 20]) and decide({27: 0.0, 33: 0.0}, 1.0) == (27, [27, 33])
assert decide({0: 0.0}, 1.0) == (0, [0])                  # float16 alone: deployed, "trivial"
assert decide({2: 0.0}, 1.0) == (None, [])                # position 1 has p=1: position 2 unreachable
assert decide({}, 1.0) == (None, [])                      # abstain
assert decide({0: lvl}, 1.0) == (0, [0])                  # p == level rejects
assert decide({0: float(np.nextafter(lvl, 1.0))}, 1.0) == (None, [])
assert decide({27: 0.0, 28: 0.0, 33: 0.0, 34: 0.0, 35: 0.0}, 1.0) == (35, [27, 28, 33, 34, 35])   # cross-chain: smallest bytes wins

# TV8  F counts distinct labels in the list, not "active" chains
p8 = np.ones(8); p8[0] = 0.05 / 7
assert fixed_sequence_by_family(p8, [f"f{j}" for j in range(8)], 0.05 / 8) == []
assert fixed_sequence_by_family(p8[:7], [f"f{j}" for j in range(7)], 0.05 / 7) == [0]

# TV9  deterministic end-to-end problem on S0 metadata (no RNG), n = 450
depth, seen = [], {}
for f in FAM: depth.append(seen.get(f, 0)); seen[f] = seen.get(f, 0) + 1
chain_ix = [list(dict.fromkeys(FAM)).index(f) for f in FAM]
VALS = np.concatenate([1.0 / np.log2(np.arange(2, 7)), [0.0]])      # nDCG@5 values: ranks 1..5, miss
CYC = np.array([0, 0, 0, 1, 2, 5, 0, 3, 5, 4])
n = 450; i = np.arange(n); kk = np.arange(K)[:, None]
b9 = VALS[CYC[i % 10]]
h = ((i[None, :] * 7919 + kk * 104729 + 13) % 1009) / 1009.0
loss = (0.002 * np.maximum(np.array(depth) - 1, 0) * (1 + 0.5 * np.array(chain_ix)))[:, None]
c9 = np.where(h < loss, 0.0, b9[None, :])
EXPECT = {  # (T, delta): (selected, rejected)
    (0.99, 0.05): (None, []), (0.99, 0.01): (None, []),
    (0.97, 0.05): (28, [0, 1, 2, 3, 7, 8, 14, 15, 20, 21, 27, 28, 33, 34]),
    (0.97, 0.01): (28, [0, 1, 2, 7, 8, 14, 15, 20, 21, 27, 28, 33, 34]),
    (0.95, 0.05): (30, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 20, 21, 22, 27, 28, 29, 30, 33, 34, 35]),
    (0.95, 0.01): (35, [0, 1, 2, 3, 4, 7, 8, 9, 10, 14, 15, 16, 20, 21, 27, 28, 29, 33, 34, 35])}
PCHK = {0.95: {0: 6.069652166871767e-06, 3: 2.5280520281851414e-05, 4: 0.00019715526301319086, 5: 0.003097429218431128,
               6: 0.0031039904927338195, 28: 6.069652166871767e-06, 30: 0.005487977456782755,
               35: 0.0006658290732972056, 39: 0.9719748201470284},
        0.97: {0: 0.000958527981069801, 3: 0.0037845824328897413, 4: 0.026585268871457415, 9: 0.038583562515132716,
               10: 0.0031892869322677647, 30: 0.214876440200214, 39: 1.0},   # p[10] <= level but unreachable (9 stops chain 3)
        0.99: {0: 0.13320105693579673, 3: 0.5451647971928198}}
for (T, delta), (sel_e, rej_e) in EXPECT.items():
    sel, p, rej = method_c(c9, b9, FAM, BYT, VEC, T, delta)
    assert (sel, rej) == (sel_e, rej_e), (T, delta)
    for k, v in PCHK[T].items(): assert close(p[k], v), (T, k)
assert 10 not in EXPECT[(0.97, 0.05)][1]
# monotone in delta: rejected(0.01) is a subset of rejected(0.05)
assert all(set(EXPECT[(T, 0.01)][1]) <= set(EXPECT[(T, 0.05)][1]) for T in (0.99, 0.97, 0.95))
# same-platform fingerprints of the p vectors (little-endian float64 bytes): optional, bitwise
import hashlib
FP = {0.99: "59f3788406a129bbff850ce2ea3f40342958913f9377b000cf30d7f6590952ae",
      0.97: "676cced61cbb56ce2f40f2fe47ad021e37eb719254f2c72ae63c816378e410ca",
      0.95: "4f4b03e005c94dd94ebff0537fa85086a970ba6bee3ce574c71e2325c8da6674"}
FP_MATCH = {T: hashlib.sha256(method_c(c9, b9, FAM, BYT, VEC, T, 0.05)[1].astype("<f8").tobytes()).hexdigest() == fp
            for T, fp in FP.items()}   # informational: expected all True on numpy 2.4.6; do not gate CI on other platforms

# TV10 structure properties (T = 0.95 problem above)
sel0, p0, rej0 = method_c(c9, b9, FAM, BYT, VEC, 0.95, 0.05)
perm = np.random.default_rng(5).permutation(K)                                # candidate-wise independence, bitwise
assert np.array_equal(betting_pvalues(to_y(c9[perm], b9[None, :], 0.95), M0(0.95)), p0[perm])
assert all(p0[a] == p0[bb] for a, bb in ((1, 7), (14, 20), (27, 33)))         # duplicated pipelines: identical p
order = []                                                                     # interleave chains, keep within-chain order
queues = {f: [k for k in range(K) if FAM[k] == f] for f in dict.fromkeys(FAM)}
rr = np.random.default_rng(3)
while any(queues.values()):
    f = rr.choice([f for f, q in queues.items() if q]); order.append(queues[f].pop(0))
order = np.array(order)
rej_i = fixed_sequence_by_family(p0[order], [FAM[j] for j in order], 0.05 / F)
assert sorted(order[rej_i]) == rej0                                            # same certified set
rel = {f: f"zz{j}" for j, f in enumerate(dict.fromkeys(FAM))}                   # bijective relabelling of families
assert method_c(c9, b9, [rel[f] for f in FAM], BYT, VEC, 0.95, 0.05)[0] == sel0
rev = np.array([k for f in dict.fromkeys(FAM) for k in reversed([j for j in range(K) if FAM[j] == f])])
rej_r = fixed_sequence_by_family(p0[rev], [FAM[j] for j in rev], 0.05 / F)       # each chain reversed (most aggressive first)
assert sorted(rev[rej_r]) != rej0                                              # chain order matters (power, not validity)

# TV11 nested samples: selected bytes never increase with n (R/x_common.py:1192-1214)
last = np.inf
for nn in (50, 100, 150, 225, 300, 450):
    s_, _, _ = method_c(c9[:, :nn], b9[:nn], FAM, BYT, VEC, 0.95, 0.05)
    cur = np.inf if s_ is None else BYT[s_]
    assert cur <= last; last = cur
```

```python
# spec-test: slow  (type-I error at the null boundary, 20,000 replicates; needs only the `ref` block)
M0 = lambda T: T / (1.0 + T)
reps, lv = 20000, (0.05, 0.05 / 7, 0.01)
T = 0.95; m0 = M0(T)
for n in (50, 225):                                          # two-point null, P(Y=1) = m0
    y = (np.random.default_rng(2026).random((reps, n)) < m0).astype(np.float64)
    rate = [float((betting_pvalues(y, m0) <= l).mean()) for l in lv]
    assert all(r <= l + 4 * (l * (1 - l) / reps) ** 0.5 for r, l in zip(rate, lv))
    # observed: n=50 [0.0159, 0.00195, 0.00335]; n=225 [0.0201, 0.0032, 0.00445]
vals = np.r_[1.0, 1 / np.log2([3, 4, 5, 6]), 0.0]; pr = np.array([.50, .08, .05, .03, .02, .32])
for T, n in ((0.95, 50), (0.95, 225), (0.97, 50), (0.97, 225)):   # rare total losses, R = T exactly (e1_1_study.py:240-285 "total" mode)
    g = np.random.default_rng(99); bq = vals[g.choice(6, size=(reps, n), p=pr)]
    cq = np.where(g.random((reps, n)) < 1 - T, 0.0, bq)
    rate = [float((betting_pvalues(to_y(cq, bq, T), M0(T)) <= l).mean()) for l in lv]
    assert all(r <= l + 4 * (l * (1 - l) / reps) ** 0.5 for r, l in zip(rate, lv))
    # observed (rates at levels 0.05, 0.05/7, 0.01): T=0.95 n=50 [0, 0, 0]; n=225 [0.0259, 0.0022, 0.00355]; T=0.97 n=50 [0, 0, 0]; n=225 [0.01165, 0, 0]
```

### 9.3 Golden-file test against the stored E1 selections (strongest; **not executed by the spec author**)

For each of the seven stores (`reports/research/E1/store/<store>.npz` + `.json`; real files, not LFS pointers in this worktree; sha256 in `PRE:1423-1429`), `E.Store(d, "labels")`:
`idx = np.random.default_rng(seed).integers(0, N, size=n)`, `c = st.pq[:, idx]`, `b = st.base[idx]` (`R/e1_1_study.py:215-217`), seeds 0..1999, `n` in {50,100,225}, `T` in {0.99,0.97,0.95}, `delta` in {0.05,0.01};
the new module's selection (`-1` for abstain) must equal `np.load("reports/research/E1/{real|confirm}/<store>_selections.npz")[f"n{n}_T{T}_C@{delta}"]` (int16) seed by seed, exactly (`R/x_common.py:909-927` gate G4, key `R/x_common.py:951-952`;
development stores in `real/`, the three held-back stores in `confirm/`, `R/x_common.py:135-138`). Keep whole seeds together when chunking (`R/x_common.py:267-270`). Reading these stores reads per-query values: do it as an equivalence test, not as new analysis.
The agreement arm has no stored selections from E1; the X1 results (`reports/research/X1/x1_<store>_selections.npz`, keys prefixed `L_` or `A_`, e.g. `A_n225_T0.95_C@0.05`; threshold 0.90 is written `T0.9`) can serve (`HAR:152`, `R/x1_agreement.py:266-267`); they were produced by the harness wrapper (`R/x_common.simulate`), which gate G4 ties to E1 under labels only.

### 9.4 Property list for the test suite (maps to the blocks above)

TV1 all-ones anchors (p by T and n); TV2 no-evidence p = 1; TV3 best-case sample sizes (n >= 116 at T = 0.95, delta/7); TV4 order sensitivity; TV5/TV11 prefix and nested monotonicity (`PRE:579-584`: any violation on nested samples is an implementation error);
TV6 `to_y` corners and dtype; TV7 chain-stop, tie-break, level boundary, abstention, cross-chain choice; TV8 `F` semantics; TV9 end-to-end golden; TV10 candidate independence, duplicate identity, interleaving invariance, family-relabel invariance (`PRE:1109-1110`: output unchanged when names are removed), chain-order sensitivity; delta-monotonicity (TV9);
slow: type-I error; 9.3 golden file; `method_c_config` values (section 7); S0 hash from sidecar equals `af29b011...` (section 1.2, `[VERIFIED]`).

## 10. What an implementation must not do (silent weakening or change of semantics)

Statistical validity
1. Do not continue past the first non-rejection of a chain, "test everything and keep the rejected" (that is uncorrected multiplicity), or score candidates after the stop and use them.
2. Do not rearrange `to_y` (`(c - T*b + T)/(1+T)` is not bit-identical to `a/(1+T)`, `(c + T*(1-b))/(1+T)`, ...; differences ~1e-17 `[VERIFIED]`), do not special-case `b = 1`, do not compute in float32, do not change `m0 = T/(1+T)`.
3. Do not alter the bet: prior 1/2 and 1/4, denominators `t+1` / `t+2`, clip to `[0, 0.9/m0]`, `log1p`, running maximum including the start; do not use the final wealth, `1/K_n`, a two-sided or normal-approximation p-value.
4. Do not use `delta/K`, an unequal split, alpha recycling, or an `F` other than the number of distinct family labels of the list actually passed; do not drop the `float16` chain or de-duplicate the three duplicated pipelines.
5. Do not reorder chains or candidates by bytes, by `c`, by point retention or by anything computed from the sample (that is C-split and needs a split); do not sort by family.
6. Do not split, thin, shuffle, sort or de-duplicate the queries; do not filter queries per candidate; do not drop NaN queries silently; do not stop early and report a different p.
7. Do not fit any stage, threshold, scale, centre or clustering on calibration queries; do not let queries influence the candidate list; do not re-run with an altered list or order on the same sample; do not pick the best of several `(T, delta)` or several query orders.
8. Do not mix Method C with the bootstrap bound, `calibrate()`, `retention()`/`retention_lower_bound()`, `split_queries`, `walk_select`; do not report the sample retention of the winner as certified.
9. Do not accept values outside [0,1], non-finite values, `mean(b) = 0`, or `T`/`delta` outside their domains; the frozen functions return NaN/plausible values silently there (section 2.4).
10. Do not apply different arithmetic to the two constructs, and never merge them: never report `R_A >= T_A` as retention or relevance (`PRE:147, 159-160`).

Semantics of the outcome
11. Do not change the choice key `(bytes, vectors, position)`; no tolerance on ties; do not break ties by compression ratio, label or chain; `position` is the list index.
12. Do not use `<` for `<=` at the level; do not compute the level as `delta*(1/F)` or in float32.
13. Do not return any candidate when nothing is certified and call it certified; do not substitute float16 or any other candidate as a silent fallback; abstain returns the uncompressed index and says so.
14. Do not phrase the result as `P(R >= T | deployed) >= 1-delta`, as "95% confident the compressed index retains T", or as "guaranteed retention" without its conditions (section 6); do not call Method C new.
15. Do not state the guarantee for other corpora, sizes, query distributions, finite future batches, coordinate-rotated vectors or other scoring functions; do not describe validation on pool resamples as independent evidence.
16. Do not derive `F`, `bet_cap` or `level` from a config dict that disagrees with the code, or hard-code `7` where the list defines `F`; do not accept an S0-labelled list whose spec hash differs from `af29b011...`, or one built without SciPy.

Model-blindness (`PRE:1089-1118`)
17. Inputs are arrays; no encoder name or provenance may change the output; no per-encoder choice of list, `T`, `delta` or cap; the wide space is a separately registered arm, not a per-encoder switch.

Numerics
18. Do not vectorise across queries (the recursion is sequential in t); vectorising across candidates is fine. Do not use `math`/`float128`/other log1p implementations if bit-closeness to the frozen function is wanted; expect <= 1 ulp differences across numpy builds and compare at `rtol=1e-12`.
19. Do not draw any random number inside Method C; a seed would be spurious. If an optional query permutation is offered, default identity, seed independent of data, recorded in the result.
20. Do not reuse `Y` or p-values across targets: both `Y` and `m0` depend on `T`. p-values do not depend on `delta` and may be shared across `delta` (the harness does, `R/e1_1_study.py:116, 121-127`).

## 11. Ambiguities and open decisions (code or documents silent or inconsistent)

1. **Domain of `T` and `delta`**: frozen code unchecked; research grid only {0.99, 0.97, 0.95, (0.90)} and {0.05, 0.01}. Proposed: `T` in (0,1], `delta` in (0,1); warn outside the grid.
2. **float16 as an uncertified fallback**: not defined anywhere (section 4). Needs an explicit product decision and a separate status.
3. **Non-finite, out-of-range, `mean(b)=0`, `n=0`**: frozen behaviour is silent NaN or p = 1 (section 2.4); the harness stops the run. Proposed: raise.
4. **Query-order source in production**: A5 requires an outcome-independent fixed order; no document says whether a seeded permutation or arrival order is preferred.
5. **Sampling without replacement, sessions, drift, near-duplicate queries**: not covered by A1; documents only warn (`RES:191-193`).
6. **Several `T`/`delta` from one sample, repeated analyses of the same sample**: not addressed (6.3).
7. **`chains: 7` in `method_c_config()` is a literal**; the code uses `Problem.F` (list-derived). For S0 they agree.
8. **Citation year** of Waudby-Smith and Ramdas: 2023 in `R/e1_stats.py:16`, `PRE:1162`, `DES:56`; 2024 (vol. 86) in the paper bibliography (`paper/optivision_research.tex:738-740`). Same paper.
9. **Assumption wording differs slightly**: `RES:146-148` and `DES:92-96` (list "fixed before the queries are seen") vs `PRE:1175` (adds "every fitted stage is fitted on documents only") vs the paper ("stages fitted without calibration queries"). Use the union (section 6.2).
10. **Proof not externally reviewed** (`Q10/REPORT.md:220-221`); evidence is resampling on pools.
11. **Other metrics (recall@k, MRR, hit@k) and agreement depths > 1** are statistically admissible but unvalidated; keep nDCG@k as the validated default or label others experimental.
12. **Bit reproducibility across platforms** is not promised by the frozen code (`R/x_common.py:267-270`); define equivalence as in 9.2.
13. **`bytes_per_doc` definition**: `PRE:175` defines stored bytes (codes plus every array needed at query time, PCA matrices included) but `LIB/compose.py:93-96` counts codes plus codec overhead and the preregistration notes the library figure omits a PCA matrix (`PRE:917`, `PRE:1649-1651`). Immaterial for S0 (no PCA); for a list with PCA the size key would change which candidate wins, not validity.
14. **Terminology**: the research says "certify" for "reject"; use `certified` only for members of `rejected`.
