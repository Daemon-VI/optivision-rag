# Easy-mode validation: results

Written by `scripts/easy_validation/run_all.py` on 2026-11-13 with backend `optivision.easy` from 12 batch folder(s). Thresholds and decision rules: reports/easy/PREREG_EASY.md (part 1), PREREG_EASY_PART2.md and PREREG_EASY_PART3.md. Rows marked in-sample are fitted on the units they are reported on.

| id | prediction | pass if | result | detail |
|---|---|---|---|---|
| V1a | int8 is safe | point >= 0.97 on every page unit, held-out included | not scored | not scored: 2 unit(s) missing; int8 point >= 0.97 on 20 of 20 page units; lowest 0.9982 |
| V1b | rule + practice check, 0.97 | point >= 0.97 on >= 90% of units, >= 0.94 on all | not scored | not scored: 1 unit(s) missing; point >= 0.97 on 100% of 11 units (need 90%); lowest 0.9782 (need >= 0.94) |
| V1c | at 0.95 | point >= 0.95 on >= 90%; low >= 0.93 on >= 90% | not scored | not scored: 1 unit(s) missing; point >= 0.95 on 100%, low >= 0.93 on 100% of 11 units (need 90% each) |
| V1d | at 0.98, 0.90 | point >= T on >= 90%, each | not scored | not scored: 1 unit(s) missing; point >= 0.98 on 100% of 11 units (need 90%); point >= 0.9 on 100% of 11 units (need 90%) |
| V1-unchecked | unchecked rule on TechSlides and on V1 corpora with text hidden | V1b thresholds at 0.97 and V1c thresholds at 0.95 | not scored | not scored: 1 unit(s) missing; point >= 0.97 on 100% of 15 units (need 90%); lowest 0.9782 (need >= 0.94); point >= 0.95 on 100%, low >= 0.93 on 100% of 15 units (need 90% each) |
| V2 | size rule at 5,244 pages (colSmol-500M, colqwen2-v1.0-hf) | point >= T in 4/4 (0.95, 0.97); int8 >= 0.97 | not scored | not scored: 1 unit(s) missing; point >= T in 2/2; int8 point >= 0.97 on 1/1 |
| V1-arms | B (no 1-bit margin), C (s(N) = 0), D (flat +0.01) | reported with misses and severe misses (< T - 0.02) | not scored | reported, not pass/fail |
| V1t | text rule | point >= T on >= 90% of text units at 0.97, 0.95 | fail | point >= 0.97 on 100% of 8 units (need 90%); point >= 0.95 on 88% of 8 units (need 90%) |
| V1u | unmeasured rule, held-out encoders | picks keep point >= T on every unit; int8 and int4(mean) lows reported | not scored | not scored: 2 unit(s) missing; point >= T on 8 of 8 units |
| S | size study: V1/V2 vectors at N in {250, 500, 1,000, full} | every rung scored | not scored (in-sample) | reported; arm S refits the size coefficients from the V1 units |
| arm S | refit size coefficients, checked on V2 (kept out) | V2 passes with the refit coefficients | not scored (in-sample) | not scored: 1 unit(s) missing; point >= T in 2/2; int8 point >= 0.97 on 1/1 |
| V3 | refit and growth | |delta point| <= 0.005 on >= 95% of pairs | not scored | not scored: 1 unit(s) missing; |delta point| <= 0.005 on 82.4% of 102 pairs (need 95%) |
| V4 | probes are optimistic | probe minus real retention > 0 in >= 70% of units | fail | probe - real > 0 in 33% of 54 units (rule picks other than float) |
| V5a | planted faults (random scores; project(16) > binary; adaptive_merge(0.6) > int4) | fires on >= 95% of (checked unit, fault) pairs with real retention < 0.80 | pass | fires on 100% of 112 pairs |
| V5b | stays quiet | fires on <= 10% of forced pairs with real low >= T | fail | fires on 11% of 325 pairs |
| V5c | catches misses | fires on >= 50% of pairs with real point < T - 0.03 (>= 6 pairs) | pass | fires on 60% of 30 pairs |
| V5d | costs little | units meeting T with check >= without; median ratio >= 0.8x | pass | meeting T: 56 with the check, 56 without; median ratio 1.00x |
| V4 (ocr) | probes are optimistic | probe minus real retention > 0 in >= 70% of units | fail | probe - real > 0 in 35% of 54 units (rule picks other than float) |
| V5a (ocr) | planted faults (random scores; project(16) > binary; adaptive_merge(0.6) > int4) | fires on >= 95% of (checked unit, fault) pairs with real retention < 0.80 | pass | fires on 100% of 112 pairs |
| V5b (ocr) | stays quiet | fires on <= 10% of forced pairs with real low >= T | pass | fires on 8% of 325 pairs |
| V5c (ocr) | catches misses | fires on >= 50% of pairs with real point < T - 0.03 (>= 6 pairs) | fail | fires on 43% of 30 pairs |
| V5d (ocr) | costs little | units meeting T with check >= without; median ratio >= 0.8x | pass | meeting T: 56 with the check, 56 without; median ratio 1.00x |
| V4 (vlm) | probes are optimistic | probe minus real retention > 0 in >= 70% of units | fail | probe - real > 0 in 45% of 31 units (rule picks other than float) |
| V5a (vlm) | planted faults (random scores; project(16) > binary; adaptive_merge(0.6) > int4) | fires on >= 95% of (checked unit, fault) pairs with real retention < 0.80 | pass | fires on 100% of 64 pairs |
| V5b (vlm) | stays quiet | fires on <= 10% of forced pairs with real low >= T | pass | fires on 3% of 200 pairs |
| V5c (vlm) | catches misses | fires on >= 50% of pairs with real point < T - 0.03 (>= 6 pairs) | fail | fires on 0% of 7 pairs |
| V5d (vlm) | costs little | units meeting T with check >= without; median ratio >= 0.8x | pass | meeting T: 32 with the check, 32 without; median ratio 1.00x |
| V6 | wiring | passes on every loader that loads on Kaggle; fails on a re-initialised norm and a dtype mismatch | fail | 6 of 7 loaded loaders passed; 0 of 0 planted faults failed the check |
| DT | document-type presets (slides, reports, academic, tables) | >= 2 scored datasets, >= 4 models, low >= T on every unit of the type | pass | 1/4 types passed; slides: fail (2 scored dataset(s) (need >= 2), 3/4 models complete, low >= T on 28/28 units (worst low - T: +0.0053)); reports: fail (1 scored dataset(s) (need >= 2), 4/4 models complete, low >= T on 13/16 units (worst low - T: -0.0222)); academic: pass (2 scored dataset(s) (need >= 2), 4/4 models complete, low >= T on 32/32 units (worst low - T: +0.0103)); tables: fail (2 scored dataset(s) (need >= 2), 4/4 models complete, low >= T on 26/32 units (worst low - T: -0.0307)) |
| probe_allowance | practice check raising layout one step (T in 0.95, 0.97) | >= 20 probe-passing cases, real point >= T on >= 90%, >= T - 0.02 on all | pass (in-sample) | 24 probe-passing step-up cases (of 28 tested): point >= T on 100% (need >= 90%), worst point - T = +0.0052 (need >= -0.02) |
| probe_allowance (ocr) | practice check raising layout one step (T in 0.95, 0.97) | >= 20 probe-passing cases, real point >= T on >= 90%, >= T - 0.02 on all | pass (in-sample) | 26 probe-passing step-up cases (of 28 tested): point >= T on 92% (need >= 90%), worst point - T = -0.0041 (need >= -0.02) |
| probe_allowance (vlm) | practice check raising layout one step (T in 0.95, 0.97) | >= 20 probe-passing cases, real point >= T on >= 90%, >= T - 0.02 on all | fail (in-sample) | only 15 probe-passing step-up case(s) of 16 tested (need >= 20); point >= T on 100%, worst point - T = +0.0165 |

## Decision rules that fired (proposed, not applied)

- Arm B passes V1b, V1c, no severe 1-bit miss: remove the 1-bit margin `{"features.margins.one_bit_margin": 0.0}`
- V4 fails: practice questions described as "not reliably easier or harder than real questions" `{"features.practice_check.wording": "not reliably easier or harder than real questions"}`
- V5b or V5c fails: the practice check becomes a warning `{"features.practice_check.mode": "warning"}`
- DT campaign type rule passes: enable type presets for ['academic'] `{"features.type_presets.on": true, "features.type_presets.enabled_types": ["academic"]}` PREREG_EASY part 3
- probe_allowance passes (>= 20 cases, >= 90% >= T, all >= T - 0.02): enable 1-step upward probe allowance for text_layer `{"features.probe_allowance.on": true, "features.probe_allowance.text_layer": true}` adjusted on our validation sets (in-sample)
- probe_allowance (ocr) passes (>= 20 cases, >= 90% >= T, all >= T - 0.02): enable 1-step upward probe allowance for ocr `{"features.probe_allowance.on": true, "features.probe_allowance.ocr": true}` adjusted on our validation sets (in-sample)

Status line this would give: "adjusted on our validation sets (in-sample)"
