# Deviations from the frozen preregistration

Changes made after PREREG_EASY parts 1-3 were frozen. None of them touches a frozen file, a
threshold, a decision rule or an analysis.

## D1 (2026-10-22): sentence-transformers 6 installed in six notebooks

**What happened:** the first e3a run (commit 96a8092) recorded `vidore/colqwen2-v1.0-hf` as
`error: ImportError: SentenceTransformersAdapter needs sentence-transformers>=6.0`.
`units.py` lists the extra pip packages per batch and left this one out for every batch that
runs that model: e1a, e2a, e2b, e3a, e4a and e4b. Kaggle ships 5.7.0.

**Change:** `scripts/make_easy_validation.py` (not frozen) now adds
`sentence-transformers[image]>=6.0.0` to a notebook's extra packages whenever an encode job
loads through sentence-transformers. The six notebooks were regenerated. e1b, e3b and e5
already installed it. The installed versions go to each run's `pip_freeze.txt`.

**Data:** that first e3a run produced no scored rows. Its other model,
`vidore/colSmol-500M`, failed `check_retriever` on v3_industrial: 8 of 12 queries ranked their
own page first, and 10 are needed. That is the frozen rule working as written, and it is
expected to happen again when e3a is rerun. The archive is kept outside the repository, and
only the rerun is ingested.

**Top-ups:** e1a, e2a and e2b had already run from the old notebooks, so each lacks
colqwen2-v1.0-hf, and all their other jobs completed. Instead of rerunning them, two top-up
notebooks in `notebooks/kaggle_easy_topup/` run only that model's original jobs, unchanged:

- `e6a_easy_topup_b1` covers the e1a job (v3_hr, v3_computer_science).
- `e6b_easy_topup_b2` covers the e2a and e2b jobs (v3_pharmaceuticals, rmm_techslides).

`data.load_unit` merges a unit's files across batch folders, so the top-ups sit beside the
first runs.

**Also recorded:** colSmol-500M failed `check_retriever` on v3_pharmaceuticals (9 of 12) in e2a,
under the frozen rule.

## D2 (2026-10-22): the V6 wiring test ran only in part

**What happened:** in e1b (commit 96a8092), `vidore/colqwen2.5-v0.2` ran out of GPU memory in
float32 on one T4. `wiring_v6.py` (frozen) puts that model on one GPU because it is a fault model.
After the error the adapter was still referenced, so its memory was not freed. The next six
loaders then failed to load with out-of-memory errors: colSmol-256M, colSmol-500M, ColNetraEmbed,
colqwen-omni-v0.1, VultronRetrieverFlash-Qwen3.5-0.8B and colqwen2-v1.0-hf.

**Result as recorded:** "6 of 7 loaded loaders passed; faults 0". Of the 13 representatives:

- 6 loaded and passed the wiring check: colpali-v1.3, colqwen2-v0.1, answerai-colbert-small-v1,
  tomoro-colqwen3-embed-4b, ColVec1-4b and nemotron-colembed-vl-4b-v2.
- 1 loaded and then errored: colqwen2.5-v0.2.
- 6 did not load, as listed above.

Neither of the two planted-fault models ran, so **the planted-fault half of V6 has no result**.
V6 gates no decision in PREREG_EASY parts 1-3, so no decision changes. It is reported as
incomplete, and no pass is claimed for the untested loaders.

**Not changed:** `wiring_v6.py` is frozen (part 2) and stays as it is. Any later re-test is a
new, separately labelled run.

## D3 (2026-10-22): held-out selection

`heldout_selected.json` recorded the frozen rule as written. OpenSearch-AI/Ops-Colqwen3-4B came
first by downloads and errored on load (`mm_token_type_ids` missing), so the next two were taken:
tencent/EVIE-Preview-4.5B (check 10/12) and webAI-Official/webAI-ColVec1-9b (check 11/12), both
complete on v3_hr. This is listed here for completeness and is not a deviation.
