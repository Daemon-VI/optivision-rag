# PREREG_EASY part 3 (DT campaign and practice-question allowance)

Frozen before the DT and Tier B runs (`freeze.py --part 3`).

## DT campaign (FINAL_DESIGN 5.8)

A type preset ships only if the type has 2 or more scored datasets (150 queries or more), measured with 4 or more models, and low >= target on every
unit (T in [0.9, 0.95, 0.97, 0.98]). Datasets under 150 queries are reported only. Mixed folders use the
most sensitive type covering at least 5% of pages.

| type | datasets |
|---|---|
| slides | v3_pharmaceuticals, rmm_techslides |
| reports | v3_hr, v1_syn_government |
| academic | v3_computer_science, v1_arxivqa, v1_syn_ai |
| tables | v1_tabfquad, v1_tatdqa |

Models: vidore/colpali-v1.3, vidore/colqwen2-v1.0-hf, vidore/colqwen2.5-v0.2, vidore/colSmol-500M. New corpora (V1 format, revisions pinned):

| key | id | revision | pages | queries | licence |
|---|---|---|---|---|---|
| v1_arxivqa | vidore/arxivqa_test_subsampled | b8a106812c | 500 | 500 | cc-by-sa-4.0 |
| v1_tabfquad | vidore/tabfquad_test_subsampled | 16c8e63361 | 280 | 280 | mit |
| v1_tatdqa | vidore/tatdqa_test | b46fea4369 | 1663 | 1663 | cc-by-4.0 |
| v1_syn_government | vidore/syntheticDocQA_government_reports_test | 91cf66572d | 1000 | 100 | mit |
| v1_syn_ai | vidore/syntheticDocQA_artificial_intelligence_test | 5694fec64c | 1000 | 100 | mit |

Reported only: v1_syn_ai, v1_syn_government. Scans (DocVQA) and infographics (InfoVQA) have one V1 set each and
stay under the worst-dataset rule. GPU budget 14.0 h (planned 10.14-12.68 h).

## Practice-question allowance (Tier B, FD 5.3.2)

Generator Qwen/Qwen2-VL-2B-Instruct (apache-2.0) on v3_hr, v3_computer_science, v3_pharmaceuticals; the question sets are
encoded by the design and held-out models and scored apart (V4/V5). Practice questions may raise one step only if
this test passes; until then they can only step down.

**Allowance test.** On every scored V3 unit (design and held-out models; T in [0.95, 0.97]), take each case where the
practice check (Tier A, and separately Tier B) passes on the rung one step more compressed than the catalogue pick.
The allowance is switched on for that tier only if there are 20 or more such cases, the real point retention is >= T
in at least 90% of them, and >= T - 0.02 in all of them. Otherwise `probe_allowance` stays off. A pass is labelled
"adjusted on our validation sets (in-sample)" until a fresh unit set confirms it.
