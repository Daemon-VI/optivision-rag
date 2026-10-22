# PREREG_EASY part 1 (rule, units, thresholds)

Source: FINAL_DESIGN section 8. Frozen by `python scripts/easy_validation/freeze.py --part 1` before any test unit
(B1-B3) is encoded and before 0.6.0 is published; the freeze records SHA-256 (CRLF turned into LF) of the files in
`scripts/easy_validation/freeze.py` PARTS["1"]. B0 (follow-ups f14a/f14b) re-measured design catalogue rows of three
models; it is design data, ran before this freeze, and is not a test unit. Its rows are in `easy_levels.json` as frozen.

## Units (English-original query counts; units under 150 queries are reported, not scored)

| key | Hugging Face id | revision | pages | queries | licence | roles |
|---|---|---|---|---|---|---|
| v3_hr | vidore/vidore_v3_hr | 0cdf0979f2 | 1110 | 318 | cc-by-4.0 | V1, V5, V1u, V4 |
| v3_computer_science | vidore/vidore_v3_computer_science | d5cc75883d | 1360 | 215 | cc-by-4.0 | V1, V5, V1u, V4 |
| v3_pharmaceuticals | vidore/vidore_v3_pharmaceuticals | 3abd4aa8a9 | 2313 | 364 | cc-by-4.0 | V1, V5, V4 |
| v3_industrial | vidore/vidore_v3_industrial | e26c864724 | 5244 | 283 | cc-by-4.0 | V2, S |
| rmm_techslides | ibm-research/REAL-MM-RAG_TechSlides_BEIR | 614ad5cac2 | 1963 | 1354 | cdla-permissive-2.0 | V1-unchecked |
| beir_fiqa10k | BeIR/fiqa | 979c07a7cb | 10000 | 648 | cc-by-sa-4.0 | V1t |
| beir_arguana | BeIR/arguana | 9bcf8fc032 | 8674 | 1406 | cc-by-sa-4.0 | V1t |

Units are model x corpus x T, T in [0.9, 0.95, 0.97, 0.98]. Page metric ndcg@5, text metric
ndcg@10. Bootstrap 1000 resamples, seed 0. Size study rungs [250, 500, 1000, 'full'] (seed 0).
Fit sample 200000 rows (seed 0); growth fraction 0.5.

## Models

Design: vidore/colpali-v1.3, vidore/colqwen2-v1.0-hf, vidore/colqwen2.5-v0.2, vidore/colSmol-500M. V2 models: vidore/colSmol-500M, vidore/colqwen2-v1.0-hf. Text: colbert-ir/colbertv2.0, answerdotai/answerai-colbert-small-v1, lightonai/LateOn, lightonai/GTE-ModernColBERT-v1.
Held-out (V1u): the 2 most-downloaded late-interaction page retrievers not in the registry at freeze that pass
`check_retriever` and fit on 2xT4 (candidate list and exclusions: `units.HELDOUT_CANDIDATES`, `HELDOUT_EXCLUDED`).

## Thresholds (as the analyses use them)

```json
{
 "V1a": {
  "rung": "int8(per_vector)",
  "point_min": 0.97,
  "share": 1.0
 },
 "V1b": {
  "target": 0.97,
  "point_min": 0.97,
  "share": 0.9,
  "all_point_min": 0.94
 },
 "V1c": {
  "target": 0.95,
  "point_min": 0.95,
  "share": 0.9,
  "low_min": 0.93,
  "low_share": 0.9
 },
 "V1d": {
  "targets": [
   0.98,
   0.9
  ],
  "share": 0.9
 },
 "V1-unchecked": {
  "as": [
   "V1b",
   "V1c"
  ]
 },
 "V1-arms": {
  "severe_below": 0.02
 },
 "V1t": {
  "targets": [
   0.97,
   0.95
  ],
  "share": 0.9
 },
 "V1u": {
  "share": 1.0
 },
 "V2": {
  "targets": [
   0.95,
   0.97
  ],
  "n_pages": 5244,
  "int8_point_min": 0.97
 },
 "V3": {
  "abs_delta_max": 0.005,
  "share": 0.95
 },
 "V4": {
  "share": 0.7
 },
 "V5a": {
  "real_below": 0.8,
  "fire_share": 0.95
 },
 "V5b": {
  "fire_share_max": 0.1
 },
 "V5c": {
  "miss_below": 0.03,
  "fire_share": 0.5,
  "min_pairs": 6
 },
 "V5d": {
  "median_ratio_min": 0.8
 }
}
```

Size coefficients {"codec": 0.005, "merge>codec": 0.0175, "binary": 0.0175, "merge>binary": 0.04}; one-bit margin 0.025; rounding step 0.0025.

## Decision rules (fixed now; rules propose, `decide.py` never applies them)

Any margin changed by these rules is labelled "adjusted on our validation sets (in-sample)" until a fresh, separately
preregistered unit set confirms it. Outcomes go to `presets/easy_ledger.json`; status labels change only from recorded
outcomes.

| Outcome | Action |
|---|---|
| V1b or V1c fails | +0.01 on compressed rungs at that target; if V1 still fails, cap at codec chains |
| V1d fails at 0.90 | Lowest target becomes 0.95 |
| V1-unchecked fails | +0.01 merge penalty for unchecked folders |
| Arm B passes V1b, V1c, no severe 1-bit miss | Remove the 1-bit margin |
| Arm C passes V1, V2 | Halve the size coefficients |
| Arm D passes, A fails | Flat +0.01 |
| Arm S | Coefficient = largest low drop per unit of s(N), rounded up to 0.0025; adopted only if V2 (kept out) passes; a negative s(N) only if measured |
| V1u int8 low >= 0.98 on all held-out page units | Unmeasured pages: int8 at 0.98 |
| V1u int4(mean) low >= 0.95 on all held-out page units | Unmeasured pages: int4(mean) at 0.95 |
| V1u picks miss | Unmeasured: float16 above 0.95 |
| V2 fails | Codecs only above 2,500 pages |
| V3 fails | Re-encode advised past 1.5x growth |
| V4 fails | "not reliably easier or harder than real questions" |
| V5a fails | Practice questions only as examples |
| V5b or V5c fails | The practice check becomes a warning |
| V8 fails | Docs say "upgrades are rare" |
