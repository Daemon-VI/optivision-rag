# Bytes against retention, and merging

The analyst's findings, then the verifier's corrections, which take precedence.

Post hoc, exploratory analysis. Script: `C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/frontier/frontier.py`; full output: `.../frontier/out.txt`.

Units are the model x dataset JSONs of the s17 `b` batches: text f9b = 24, pages f10b+f11b = 14 (6+8), SciDocs f12b = 8. f13b has only 2 units and is reported in the output file but not summarised here. Queries per unit are 300-323 for text, 280-451 for pages and 1000 for SciDocs. Retention is ndcg@5 over the same unit's float32 baseline. "Point" means retention >= 0.95; "LB" means the lower end of the 95% query-bootstrap CI >= 0.95. bytes/f32 is the mean over units. "A>B" counts units where A's point retention is higher than B's (not a significance test). "CI-sep" counts units where A's CI lower bound is above B's CI upper bound.

**Frontier definition.** A configuration is on a unit's frontier if no other row has bytes <= and retention >= with at least one strict. The comparison uses point retention only, so near-ties of 0.001 count as dominance. Frontier counts are therefore noisy for rows within bootstrap noise of each other (int8, float16, int4norm(mean)). Read them as rough indicators.

**Per-family counts.** Each cell is point / LB. "Mean" is the mean retention over units. Frontier is the number of units on whose frontier the config sits.

| config (bytes/f32) | text n=24: pt / LB / mean / frontier | pages n=14: pt / LB / mean / frontier | SciDocs n=8: pt / LB / mean / frontier |
|---|---|---|---|
| float16 (0.50) | 24 / 24 / 1.000 / 10 | 14 / 14 / 1.000 / 1 | 8 / 8 / 1.000 / 5 |
| int8 per-vec (0.255) | 24 / 24 / 0.999 / 7 | 14 / 14 / 1.001 / 0 | 8 / 8 / 0.999 / 4 |
| int4 (0.13) | 13 / 12 / 0.820 / 2 | 10 / 10 / 0.722 / 4 | 2 / 2 / 0.657 / 0 |
| int4norm (0.13) | 20 / 19 / 0.979 / 1 | 10 / 10 / 0.815 / 1 | 6 / 5 / 0.960 / 1 |
| int4(mean) (0.13) | 23 / 22 / 0.993 / 4 | 14 / 13 / 0.997 / 1 | 7 / 6 / 0.979 / 1 |
| **int4norm(mean) (0.13)** | **24 / 24 / 0.998 / 11** | **14 / 13 / 1.000 / 4** | **8 / 8 / 0.995 / 6** |
| lloyd2(7) (0.0625) | 9 / 5 / 0.839 / 4 | 14 / 12 / 0.995 / 4 | 1 / 1 / 0.656 / 1 |
| binary (0.031) | 6 / 2 / 0.815 / 22 | 14 / 12 / 0.990 / 11 | 1 / 0 / 0.680 / 8 |
| merge.33>int8 (0.085) | 19 / 14 / 0.969 / 3 | 14 / 13 / 0.997 / 1 | 6 / 2 / 0.959 / 4 |
| merge.33>int4 (0.043) | 8 / 6 / 0.719 / 0 | 10 / 9 / 0.718 / 1 | 2 / 0 / 0.546 / 0 |
| merge.33>int4norm (0.043) | 14 / 11 / 0.946 / 2 | 10 / 10 / 0.807 / 2 | 2 / 0 / 0.901 / 1 |
| merge.33>int4(mean) (0.043) | 17 / 9 / 0.956 / 0 | 14 / 13 / 0.994 / 1 | 3 / 0 / 0.928 / 0 |
| merge.33>int4norm(mean) (0.043) | 20 / 14 / 0.970 / 9 | 14 / 13 / 0.999 / 4 | 5 / 2 / 0.954 / 4 |
| merge.33>binary (0.010) | 5 / 0 / 0.765 / 17 | 13 / 10 / 0.983 / 7 | 1 / 0 / 0.622 / 7 |
| merge.25>int8 (0.064) | 18 / 12 / 0.937 / 2 | 14 / 13 / 0.995 / 0 | 5 / 0 / 0.936 / 1 |
| merge.25>int4(mean) (0.033) | 19 / 6 / 0.922 / 1 | 14 / 14 / 0.995 / 1 | 2 / 0 / 0.913 / 1 |
| merge.25>int4norm(mean) (0.033) | 19 / 10 / 0.937 / 11 | 14 / 14 / 0.998 / 3 | 4 / 0 / 0.935 / 5 |
| merge.25>binary (0.008) | 3 / 0 / 0.728 / 24 | 12 / 11 / 0.982 / 14 | 0 / 0 / 0.596 / 8 |

float32 trivially meets 1.0 everywhere. The per-codec merge(x)>int4 and merge(x)>int4norm rows not shown for 0.25 are in out.txt.

**1. int4norm(mean) against int8 per-vector (about 2x the bytes).**
- Mean retention differences, int4norm(mean) minus int8:

| family | mean diff | units where int4norm(mean) is higher | units where int8 is higher | CI-separated |
|---|---|---|---|---|
| text | -0.0009 | 9 | 15 | 0 |
| pages | -0.0014 | 9 | 4 | 1 in int8's favour |
| SciDocs | -0.0041 | 4 | 4 | 0 |

- Both meet the point threshold in every unit, and the LB threshold in every unit except 1 page unit for int4norm(mean). On this data int4norm(mean) is indistinguishable from int8 at about half the bytes, apart from a small mean deficit.
- Whether this holds beyond these 46 units, or for other corpora, is not shown.

**2. int4norm(mean) against int4(mean).**
- Same bytes. int4norm(mean) is ahead in 17/24 text, 10/13 pages and 7/8 SciDocs units, by mean +0.0054, +0.0028 and +0.0165.
- Almost never CI-separated (text 0, pages 0, SciDocs 1). Norm preservation on top of mean-centring is a small, consistent-in-sign but mostly sub-noise gain.
- Without mean-centring the effect is large. int4norm minus int4 is +0.2275 on merge.33 in text, and on SciDocs +0.35 with 6/8 CI-separated.
- int4(mean) misses LB in 2 text units, 1 page unit and 2 SciDocs units, where int4norm(mean) misses LB in 0 text, 1 page and 0 SciDocs units.

**3. Merging x norm-preservation.**
- Mean retention difference between int4norm(mean) and int4(mean) as the post-merge codec (units where int4norm(mean) is higher; CI-separated units):

| merge | text | pages | SciDocs |
|---|---|---|---|
| 0.33 | +0.0138 (17/24; 2 CI-sep) | +0.0050 (9/13; 0) | +0.0263 (7/8; 1) |
| 0.25 | +0.0154 (22/24; 0) | +0.0029 (10/14; 0) | +0.0220 (5/8; 1) |

- The gain from norm-preservation after merging is a bit larger than the gain at the unmerged level for text (+0.0138/+0.0154 vs +0.0054) and for SciDocs. The suspicion is that merging changes vector norms, which the norm-preserving codec protects against. This was not tested directly.
- merge.33>int4norm(mean) minus merge.33>int8 (int4norm(mean) post-merge at 0.51x the bytes of the int8 version):

| family | mean diff | units where int4norm(mean) is higher | units where int8 is higher | CI-separated |
|---|---|---|---|---|
| text | +0.0009 | 14 | 10 | 0 |
| pages | +0.0020 | 7 | 6 | 0 |
| SciDocs | -0.0045 | 2 | 6 | 0 |

- merge.25 gives the same picture (text -0.0002, pages +0.0027, SciDocs -0.0011).
- So, in this data, post-merge int4norm(mean) is not distinguishable from post-merge int8 at half the bytes, whereas post-merge int4(mean) is worse than int8:
  - text -0.0129 at 0.33 and -0.0157 at 0.25;
  - SciDocs -0.0308 and -0.0231;
  - pages -0.003 and -0.0002.
  - It is CI-separated in only 1 unit per family at most.
- In the unmerged case, int4(mean) is about as good as int8 on pages and a little worse in text and SciDocs. See the per-family table (0.993 against 0.999 in text, 0.979 against 0.999 in SciDocs).
- Merge retention by family: merging costs about 0.03 mean retention at 0.33 and about 0.06 at 0.25 for text, 0.04-0.06 for SciDocs and about 0.001-0.003 for pages. The merge cost relative to unmerged int4norm(mean) is mean -0.0278 in text (22/24 lower, 6 CI-separated), -0.0407 in SciDocs (8/8 lower, 4 CI-separated) and -0.0011 in pages (8/14 lower, none CI-separated).
- merge.25>int8 has a large collapse in one text unit (min 0.509, LFM2-ColBERT-350M, one of the two LFM2 text units; its merge.25>int4norm(mean) is 0.516). merge.33 does not show it (0.915). merge.25 has fewer bytes than merge.33 (about 0.25x vs 0.33x), so some of its cost is a bytes trade, but the LFM2 drop is a model-specific anomaly in these data.

**4. Merge rows against int4norm(mean) in size.**
- Mean bytes/f32: merge.33>int8 0.085, int4norm(mean) 0.130, merge.33>int4norm(mean) 0.043, merge.25>int4norm(mean) 0.033.
- merge.33>int8 uses 0.65x the bytes of int4norm(mean) but is lower in 23/24 text units (mean -0.0287; 7 CI-separated), 8/8 SciDocs units (-0.0362; 4 CI-separated) and 9/14 page units (-0.0030; none).
- Smaller sizes via merging buy about 0.03 in text, 0.04 in SciDocs and about nothing in pages (page-family caveat: fewer than half the units are on any one model family).
- merge.33>int4norm(mean) is on the frontier in 9/24 text, 4/14 pages and 4/8 SciDocs units. merge.25>int4norm(mean) is on the frontier in 11/24, 3/14 and 5/8, mostly because it is cheap rather than accurate: only 10 text units and 0 SciDocs units have LB >= 0.95 there.
- Counts to reach the threshold: merge.33>int4norm(mean) is point >= 0.95 in 20/24 text, 14/14 pages and 5/8 SciDocs units; LB in 14/24, 13/14 and 2/8.
- Treat the page results as a family where merging is cheap, since that holds for all page codecs (merge.33>int8 pages 0.997, merge.33>binary 0.983).

**5. lloyd2(7) and binary against int4norm(mean).**
- lloyd2(7) is 0.0625 of f32 (about half the bytes of int4norm(mean)). Mean retention difference, int4norm(mean) minus lloyd2:

| family | mean diff | units where int4norm(mean) is higher | units where lloyd2 is higher | CI-separated |
|---|---|---|---|---|
| text | +0.1587 | 22 | 2 | 17 |
| pages | +0.0048 | 8 | 6 | 2 |
| SciDocs | +0.3392 | 8 | 0 | 7 |

- Binary (0.031, 0.24x int4norm(mean)) minus int4norm(mean): -0.1824 in text (19 CI-separated), -0.0102 in pages (1) and -0.3151 in SciDocs (7).
- Binary is on the frontier by size in text 22/24, pages 11/14 and SciDocs 8/8, because nothing is smaller (no merge variant is cheaper except merge>binary). It meets the point threshold in 6/24 text, 14/14 pages and 1/8 SciDocs units, and LB in 2/24, 12/14 and 0/8.
- lloyd2(7) compared with merge.33>int4norm(mean): lloyd2 is lower in 17/24 text units (mean -0.1309; 13 CI-separated; 1.45x the bytes) and 7/8 SciDocs units (-0.2985; 6 CI-separated). In pages it is level (-0.0037, no CI-separated unit).
- The cheaper-and-better options in text and SciDocs are therefore merge.33>int4norm(mean) over lloyd2(7); in pages lloyd2(7) and binary are competitive.
- The large gaps in text and SciDocs are model-dependent. The low lloyd2 values are concentrated in a few models (answerai 0.479, mxbai-edge-colbert 0.268-0.735, GTE-ModernColBERT 0.75-0.83, SciDocs mxbai-edge 0.175).

**6. What the data shows versus what I suspect.**
- Shown: int4norm(mean) is at or near int8 retention at about half the bytes in all three families here, with every unit at point >= 0.95. Merge-then-int4norm(mean) matches merge-then-int8 at about half the bytes, whereas merge-then-int4(mean) is lower in text and SciDocs. Neither lloyd2(7) nor binary reaches int4norm(mean) in text or SciDocs.
- Suspected, not tested: that merging perturbs norms and norm-preservation compensates; that the family differences (pages are robust, text and SciDocs less so) reflect dimensionality or corpus properties rather than model identity.

**Caveats (analyst).**
- Exploratory, post hoc; no multiple-comparison control; units are not independent (same models appear across datasets and families, 2 datasets per model in text).
- Frontier counts use point estimates only; near-ties within bootstrap noise count as domination, so frontier counts for int8/float16/int4norm(mean) are unstable.
- CI-separated counts are conservative (non-overlap of 95% per-unit CIs), not a paired test; paired per-query bootstrap was not run.
- Bytes ratios are corpus-mean bytes/doc; merge rows' bytes differ per unit (pages merge.25 ~0.25-0.35x int4norm(mean)).
- Page family mixes f10b (128d, 6 units) and f11b (wide, 8 units) and contains few distinct models; pages are near ceiling for most codecs, which limits discrimination.
- f13b (2 units) excluded from family summaries; its numbers are in out.txt only.
- Mechanistic claim (merging perturbs norms) is a suspicion, not tested.

## Verifier's corrections

Post hoc, exploratory verification. Units: s17 b-batches. Text f9b has 24 units (300 or 323 queries). Pages f10b+f11b has 14 units (280 or 451 queries). SciDocs f12b has 8 units (1000 queries). Retention is ndcg@5 over the unit's float32 baseline. Verification script: C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/frontier_verify/v.py (plus v2.py for the LFM2 check).

Reproduced exactly:
- Point/LB threshold counts for int4norm(mean), int8, int4(mean), lloyd2(7), binary, merge.33>int4norm(mean) and merge.25>int4norm(mean).
- All pairwise mean differences, higher/lower unit counts and CI-separated counts, and bytes ratios in sections 1, 2, 3, 4 and 5 of the analyst's findings.

Wording corrections:
- 'Indistinguishable from int8' is an informal reading of non-overlapping per-unit CIs, not a paired test. The mean deficit is consistently negative: -0.0009 text, -0.0014 pages, -0.0041 SciDocs. One page unit is CI-separated in int8's favour.
- The merge.25 weakness is not one unit. merge.25>int8 in text is 0.509 for one LFM2 unit and 0.769 for the other LFM2 unit. It is also about 0.79 for one answerai unit. merge.33>int8 is 0.915 and 0.866 for the two LFM2 units.
- Frontier (Pareto) counts were not independently recomputed. Treat them as unverified and noisy, as the analyst's own caveat says.
- Causal language (norm-preservation compensating for merge-induced norm changes) remains untested. Units are not independent, since the same models recur across datasets.
