# A04 statistical safety audit

Branch `harden/a04`, base HEAD 7a6769a. One fix commit: `8a94c40`. Scratch scripts: `C:\Users\Lenovo\projects\ov-harden\scratch\a04\` (`p_check.py`, `rule_check.py`, `order_check.py`, `mc.py`, `mc2.py`, `mc3.py`, `attacks.py`, `e2e.py`, `misc.py`, `mc*.out`).

## 1. Scope and what was executed

Read in full: `METHOD_C_SPEC.md` (lines 1-543), `core/selection/{statistics,rules,decision,engine,spaces}.py`, `core/evaluate/{evidence,reference,metrics,collect}.py`, `core/artifact/{certificate,record}.py`, parts of `verification.py` (decision, outcome, level, rule-text, replay checks), `api.py`, `cli_certify.py`, frozen `scripts/research/e1_stats.py`. I skimmed `STATISTICAL_REVIEW.md` and `RESEARCH_INTEGRITY_AUDIT.md` and used them only for cross-reference.

Executed, all with my own code in scratch (no package helper used to check itself):

| check | size | result |
|---|---|---|
| p-value: scalar pure-Python re-derivation from the spec (`math.log1p`, no numpy) vs package `betting_pvalues` vs frozen `e1_stats.betting_pvalues` | 4,000 random problems: n in {1,2,3,5,10,50,225}; T from 1e-6 to 1.0; all-ties, all-zero, all-one, binary, nDCG-like values; c == b; c = T*b; baseline all zero; `warnings` set to errors | 0 bitwise differences package vs frozen; 0 vs my scalar; no NaN/warning; n=1 and TV1 values match |
| retention exactly at target (c = T*b, n=300, T in .5..99) | 5 cases | p == 1.0 exactly in every case |
| full rule vs my own independent decision code (per-family chains, Bonferroni, `(bytes, vectors, index)` key) | 600 random problems x 2 rules, K 1..24, interleaved families, many size ties | 0 mismatches in selected, certified set, level and p-vector (258 non-empty selections) |
| ordering and tie behaviour with hand-built evidence | 9 scenarios | all as the spec says (see 3.2) |
| Monte Carlo under the null, own vectorised p-values and own decision code | 11 configurations at 20,000 and 100,000 replicates (section 2) | rate within the level in all |
| 40 attack attempts on `certified=True` | section 4 | 1 real gap found and fixed |
| end-to-end with `certify()` on `optivision.testing` data | forgery, hand evidence, 40 seeds, budget, duplicates, fallback, graded labels, NaN vectors | section 4 and 5 |
| tests | `test_core_selection_rules/engine`, `test_core_statistics`, `test_core_certificate`, `test_core_certificate_tamper`, `test_api_certify` | 914 passed after the fix; `ruff check src tests` clean |

## 2. Error accounting (own Monte Carlo)

"False selection" = a compressed candidate is selected whose retention is at or below T. In the all-null configurations every candidate is null, so any rejection counts. Candidates ordered so that later = smaller; Wilson 95% interval. Level is `error` (delta).

| config | K / F | T, delta, n | R | rule C (fixed-sequence) | rule D (Bonferroni) |
|---|---|---|---|---|---|
| S0-shaped (40 cand, 7 chains) all at boundary 0.95, c=b*Bern(.95), b~Bern(.7) | 40/7 | .95/.05/225 | 100,000 | 0.01976 [0.0189, 0.0206] | 0.01051 [0.0099, 0.0112] |
| same, n=100 | 40/7 | .95/.05/100 | 20,000 | 0 (upper 1.9e-4) | 0 |
| c independent of b, E[c]=T E[b] (c may exceed b) | 40/7 | .95/.05/225 | 20,000 | 0.01775 [0.0160, 0.0197] | 0.01535 [0.0137, 0.0171] |
| 12 cand, retention 0.90 | 12/3 | .95/.05/225 | 20,000 | 5e-5 | 0 |
| 12 cand boundary, one common noise draw across candidates | 12/3 | .95/.05/225 | 20,000 | 0.0091 [0.0079, 0.0105] | 0.00135 |
| 12 cand boundary, Beta b, c = b*U | 12/3 | .95/.05/225 and .95/.20/30 | 20,000 each | 0 | 0 |
| K=1, b=1, c~Bern(T), extreme variance | 1/1 | .5/.05/20 | 100,000 | 0.01536 [0.0146, 0.0161] | same |
| | 1/1 | .9/.05/60 | 100,000 | 0.02338 [0.0225, 0.0243] | same |
| | 1/1 | .97/.05/400 | 100,000 | 0.03366 [0.0326, 0.0348] | same |
| | 1/1 | .5/.20/200 | 100,000 | 0.10626 [0.1044, 0.1082] | same |
| 10 candidates, each its own chain | 10/10 | .9/.05/100 | 20,000 | 0.00755 | 0.00755 |
| mixed truth, 3 true (0.99+) and 3 false (0.90) per chain, miss = deployed with truth below T | 18/3 | .95/.05/225 | 100,000 | 3e-5 | 0 |
| mixed truth, first false at 0.949 right after 0.96, n=450 | 18/3 | .95/.05/450 | 100,000 | 1.0e-4 | 0.00162 |

The highest rate observed anywhere was 0.1063 at delta 0.20 and 0.0337 at delta 0.05; every upper confidence bound is below the level. The rule is conservative at the boundary (Ville), so this MC confirms "no violation" and cannot confirm tightness. No evidence against the statement in the tested configurations.

Order assumption A5 (measured, K=1, T=.9, n=60, delta .05):

| order | false-selection |
|---|---|
| independent (seeded permutation, one seed) | 0.0228 |
| sorted losses-first | 0.0143 |
| sorted wins-first (order depends on outcomes) | 1.0000 (20,000 of 20,000) |
| best of 10 seeds reported | 0.123 |
| best of 100 seeds | 0.366 |

So A5 and "one look" are load-bearing, not formalities. The code defends them as far as it can: `certify()` and the CLI shuffle with recorded seed 0 by default, `select()` uses the given order and the certificate adds a caveat for `given` and for a non-default seed. Nothing can detect repeated looks (the functions are stateless).

## 3. Findings

| ID | Sev | Location | Problem | Evidence | Fix | Status |
|---|---|---|---|---|---|---|
| A04-01 | HIGH | `core/selection/rules.py:306-314` (`is_builtin_rule`) | The "built in" test compared only `type(rule)`. A built-in instance with an instance attribute shadows the class method, because functions are non-data descriptors: `r = FixedSequenceBetting(); object.__setattr__(r, "decide", fake)`. `select(..., rule=r)` then ran `fake` and returned `certified=True`, `rule.builtin=True`, recorded the built-in's name and statement. Same for a custom object whose `__class__` is reassigned to the built-in, and for a shadowed `config()` (recorded `bet_cap` 0.99 in a certified decision). | `attacks.py` cases 5, 6, 8e: before the fix `outcome=select certified=True builtin=True` with weak evidence (all candidates metric 0 vs baseline 1); after: `certified=False builtin=False` | `and not getattr(rule, "__dict__", None)` (the built-in rules have no fields, so any instance attribute makes it non-built-in). Failing-first regression tests in `tests/test_core_selection_rules.py` and `tests/test_core_selection_engine.py` | FIXED in 8a94c40 |
| A04-02 | LOW | `api.py` docstrings of `certify` (qrels, warnings) | Said a `{doc: grade}` mapping with grade 0 is "read as its keys, grade 0 included, and warns once". The code warns and then `LabelReference` refuses with `ValueError` (the existing test says so). The docs were stale, the behaviour is the safe one. | `misc.py`: `ValueError ... grade 0` after a `UserWarning` | Docstring corrected | FIXED in 8a94c40 |
| A04-03 | MEDIUM | `core/artifact/certificate.py:378-406` (`create_certificate`), `selection/decision.py` (`from_dict`) | A `Decision` is not unforgeable and `create_certificate` does not replay `select` on the evidence it is given. A decision rebuilt with `Decision.from_dict` (built-in rule record pasted, consistent invented p-values, `certified=True`) is accepted; the resulting certificate is `certified=True` and passes level-1 verification. Level-2 replay (`verify_certificate(cert, evidence=...)`) fails it (`replay.decision`). Editing a saved certificate and recomputing the digest is caught at level 1 when the edit is inconsistent (`consistency.outcome`, `consistency.decision`) and at level 2 always. | `e2e.py` A and B | For a built-in rule, `create_certificate` re-runs `select` on the evidence with the recorded settings and refuses on mismatch (about 20 lines; needs the fallback and budget from the record). I did not implement it: it changes what `create_certificate` accepts, so it needs a decision | PROPOSED. Not reachable through `certify()`, the CLI or `select()` |
| A04-04 | MEDIUM (by design) | `Evidence` constructor, `select`, `create_certificate` | Evidence built by hand (not collected) on a real corpus and queries gives `certified=True`, and `verify_certificate(cert, evidence=fake, corpus, queries, reference, pipeline, compressed)` passes. Only re-collecting evidence from the vectors (`evidence.digest`, `evidence.sample`, `replay.decision` fail) exposes it. The module docstring says forged matching evidence is not detected, and condition C2/C4 make the evidence the caller's responsibility; but no field of the certificate says whether the evidence was produced by `collect_evidence` in the same process. | `e2e.py` C | Either record a provenance flag in the evidence (not tamper-proof), or document explicitly in the certificate guide that the verifier does not re-score. No code change made | PROPOSED, documentation level |
| A04-05 | MEDIUM (by design) | `api.certify`, `record.caveats_for` | With 120 exact duplicate queries added (240 tested), `certify` still returns `certified=True`. The library knows A1 is violated (it counts duplicates), warns and adds a caveat saying the error level "does not apply", but the boolean remains True. | `misc.py` dup queries: `select True`, `dup count 120`, warning and caveat present | Consider refusing or marking `certified=False` when `exact_duplicates > 0`. This is a policy decision (D-level); not implemented | PROPOSED |
| A04-06 | LOW | `FixedSequenceBetting` on a `FALLBACK` decision | A fallback decision records `rule.certifying=true` and carries the full STATEMENT; only the caveat "not certified ... the statement above does not cover it" and `result.certified=false` say otherwise. Not an overclaim in the flags; a reader of the statement text alone could be misled. | `misc.py` fallback | Prefix the statement or use `STATEMENT_UNEXAMINED`-style text for fallback outcomes | PROPOSED |
| A04-07 | NO ACTION | `rules.py` class attributes | Class-level tampering (`FixedSequenceBetting.decide = f`, flipping `_BUILTIN_TYPES`) yields certified output or a refusal respectively: `FixedSequenceBetting.decide` patched gives `certified=True`. Modifying the library's own classes in the running process is outside what an in-process check can defend; the certificate's `code` identity is computed from files, not from live objects. | `attacks.py` case 7 | None | NOT A DEFECT |

No defect found in the mathematics. Nothing in `statistics.py` or the rules was changed.

### 3.1 Statement and semantics (certificate)

Checked against `METHOD_C_SPEC.md` section 6: `STATEMENT` is the joint probability, states explicitly that it is not the conditional probability given selection, not about observed retention, not about future batches, not about power, over the draw of the calibration queries with corpus/list/order/T/delta fixed, "sufficient conditions", "derivation is this project's own and has not been externally reviewed". A1-A6 and C1-C4 are quoted and match the spec table (A2 corpus fixed, A3 list/order fixed and documents-only fitting, A4 bounded and positive baseline, A5 order, A6 reference as given). The Bonferroni variant drops the order requirement and says so, and its level is delta/K (verified: K=40, 7 families, min n for a perfect candidate at T=.95 is 154 vs 116 for the chain rule, level 0.00125 vs 0.00714; 116 equals spec TV3). Caveats exist for agreement, fallback, given order, non-default seed, duplicates, non-unit norms and dropped queries. `result.certified`, `Decision.certified`, `Certificate.certified`, `report()['certified']` and the CLI header all derive from the same flag. No "guaranteed", "first" or "novel" wording found in these files.

### 3.2 Ordering, stopping, family handling (own scenarios)

First non-rejection stops its chain only (a perfect later member is ignored: certified (0,)); interleaved families give the chain-wise result; F counts every distinct label including single-member chains (level 0.05/3); ties go to fewer vectors then lower index; the smallest by bytes is chosen even when an earlier chain member is certified (not the most aggressive position); relabelling families bijectively leaves the result unchanged; Bonferroni has no stopping and level delta/K; p exactly equal to the level rejects (spec TV7, the code uses `<=`).

## 4. The CRITICAL property: attempts to obtain a wrong `certified=True`

| # | attempt | result |
|---|---|---|
| 1 | subclass of `FixedSequenceBetting` overriding `decide`, same name | certified=False, builtin=False |
| 2 | subclass without override | certified=False (exact type only) |
| 3 | unrelated class with `name="fixed_sequence_betting"`, `certifying=True` | certified=False |
| 3b | class named `FixedSequenceBetting` with the built-in's `__module__` | certified=False (identity is `type()` membership, not a string) |
| 4 | custom rule declaring `certifying=True` | certified=False, declaration recorded as `declared_certifying`, assumptions prefixed as the author's |
| 5 | built-in instance, instance-dict `decide` | WRONGLY certified=True before the fix; certified=False after (A04-01) |
| 5b | same plus renamed `name` | `ValueError` (Decision validation) |
| 5c | ordinary `rule.decide = f` | `FrozenInstanceError` |
| 6 | built-in instance, shadowed `config()` | wrongly certified with altered recorded config before the fix; certified=False after |
| 7 | patch `FixedSequenceBetting.decide` on the class | certified=True (library tampering, A04-07) |
| 7c | add a subclass to private `_BUILTIN_TYPES` | `ValueError` from Decision (class name mismatch) |
| 8 | `copy.copy`, pickle round trip of a built-in | certified=True, correct (same type, no instance state) |
| 8c | pickled custom/subclass | stays uncertified |
| 8e | custom object with shadow `decide`, `__class__ := FixedSequenceBetting` | wrongly certified before the fix; fixed |
| 9 | `dataclasses.replace(abstain, certified=True)` / with `outcome=SELECT` | `ValueError` both |
| 9c | custom decision dict with builtin/certifying flipped | `ValueError` (class name mismatch) |
| 9d, 9e | `Decision.from_dict` with a built-in rule record pasted and forged select | accepted (A04-03); caught by replay |
| 10 | `fallback=` with nothing certified | outcome FALLBACK, certified=False; with a certified candidate the fallback is ignored; with a custom rule uncertified |
| 11 | evidence from another space / right identity wrong K | `SpaceMismatchError` both |
| 12 | NaN baseline, NaN candidate, metric 1.5, -0.1, zero baseline, n=0, inf bytes | all refused with specific `EvidenceError` subclasses |
| 13 | target 0, 1.0001, ndarray, True; error 1, list; budget 0, > n; seed -1, 1.5 | all refused |
| 14 | hand-made evidence | accepted, A04-04 |
| 15 | `query_budget` | prefix of the recorded order, budget > n refused, recorded in `query_order`; a budget of 5 on strong evidence abstains (reachable) |
| 16 | repeated `certify()` over 40 seeds | each call certifies independently; nothing tracks looks (seeds gave selections at positions 2, 4, 5). Documented, not defendable |
| 17 | CLI | no `--rule` or `--fallback` option exists; `--given-order` and `--order-seed` are exclusive; `--reference agreement` with `--qrels` refused; graded qrels object refused before scoring |
| 18 | edited saved certificate with recomputed digest | inconsistent edit fails level 1; consistent edit fails level 2 with evidence; only a signature (key) protects without evidence |
| 19 | `certify(rule=Subclass)` end to end | certified=False, statement is the "not examined" text |

Abstention: reachable (many cases), never replaced silently (the only path to FALLBACK is an explicit `fallback=` argument; default space without SciPy raises `ImportError` rather than shrinking the list); FALLBACK is `certified=False` with no certified positions and `selected.position=-1`, enforced in `Decision._check_consistency` and re-checked by the verifier.

## 5. Checked and correct

- p-values: exact equality with the frozen function and an independent scalar derivation on 4,000 inputs including edge cases (section 1). `to_y` casts to float64 first (spec 2.4). `validate_evidence_arrays` refuses NaN/inf, out-of-range, empty, zero-mean baseline, bad target/error, non-string family labels (all `EvidenceError`); nothing is clipped or dropped.
- Level: delta/F with F from the list passed (not delta/K); verifier recomputes the same.
- Queries without a relevant document are removed jointly for all candidates before the rule (`collect_evidence`, label-only), recorded in `dropped_queries` and a caveat; a NaN on a kept query raises `CandidateEvaluationError`.
- Fitted-on-documents-only (A3) is enforced in `collect_evidence` and `compress_selected` via traits, and a stage with unknown traits is refused.
- Seeded shuffle default in `certify` and CLI; order recorded with an index digest and checked by `tested_indices`.

## 6. Could not check

- Tightness of the error rate (the rule is conservative at the boundary; I did not construct a case near the level).
- External validity of the combined derivation (not externally reviewed, as the package says).
- `STATISTICAL_REVIEW.md` and `RESEARCH_INTEGRITY_AUDIT.md` numerical claims beyond the ones reproduced here (3-14x multiple-look inflation was not reproduced in that exact configuration; my best-of-10 and best-of-100 seed experiment gave 5.4x and 16x relative to a single seed in a different configuration).
- Real encoder data (not allowed), `verification.py` beyond the decision, outcome, level, rule-text and replay checks, `bundle.py`.
