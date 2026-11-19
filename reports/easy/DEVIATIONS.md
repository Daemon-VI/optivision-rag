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

**e4a and e4b** ran at commit 31f94d3 but from notebook copies made before the fix. Their
`pip_freeze.txt` shows sentence-transformers 5.7.0, so colqwen2-v1.0-hf is missing from both,
and all their other jobs completed. `e6c_easy_topup_dt` runs that model's two DT jobs,
unchanged. The e6a and e6b top-ups completed, with checks of 10, 10, 11 and 12 of 12.

## D4 (2026-10-23): Tier B rerun through an entry script

**What happened:** the first e5 run (commit 3c8c7e6) made no practice questions. `vlm_probes.py`
looks for `make_vlm_probes`, `make_probes_vlm` or `make_probes` on
`optivision.easy._probes_vlm`. That module names its generator `generate_vlm_probes`. Both
files are frozen (part 2). A CPU dry run of the frozen harness then found two more mismatches:

- the generator returns a `ProbeBatch`, and the harness iterates it and takes its length;
- the harness passes the page images by position, and the tier looks them up by page id.

**Change:** a new script, `scripts/easy_validation/tierb_entry.py` (not frozen), adds
`make_vlm_probes` to that module. It is the same generator, with the keywords the harness passes,
images looked up by page id, and the probes returned as a list. The script then runs
`encode_unit.main` with the same arguments. The rerun is `e5b_easy_tierb_vlm`, the same jobs
run through this script. No frozen file, prompt, generation setting, threshold or analysis
changes. A test runs `vlm_probes.generate` end to end with a fake VLM. The real Qwen2-VL
backend has not yet run on a GPU, and e5b is its first run.

**Product fix found on the way:** `optivision.easy._build` called `make_probes` on the VLM and
API tiers, which do not have it. It now calls `generate_vlm_probes` and `generate_api_probes`
and takes their `.probes`.

**Data:** the first e5 archive is kept outside the repository and is not ingested.

## D5 (2026-10-31): incremental ZIP64 archiving in `make_easy_validation.py` for `e3b`

**What happened:** `e3b_easy_b3_heldout_text` ran on Kaggle, but the final `zip -q -r -0` shell
command failed to create `results_e3b_easy_b3_heldout_text.zip` because `webAI-ColVec1-9b` on
`v3_computer_science` (1,360 pages, ~8.85 GB in float16) plus `tencent/EVIE-Preview-4.5B` and the
four text models on `beir_fiqa10k` and `beir_arguana` write ~12 GB to `/kaggle/working`, and
creating a second ~12 GB `.zip` file alongside the source folder exceeds Kaggle's 19.5 GB
`/kaggle/working` quota.

**Change:** `scripts/make_easy_validation.py` (not frozen) now removes the cloned repo before
archiving and writes `results_<batch>.zip` (`ZIP_STORED`, `allowZip64=True`) file-by-file in
Python, unlinking each source file immediately after adding it to the archive so peak usage in
`/kaggle/working` stays within 1x the output size plus one shard (~208 MB). If a resumed input
already has all job records marked `complete`, the notebook skips re-running `encode_unit.py` and
proceeds directly to archiving.

## D6 (2026-11-13): how the scoring was run on a 15.8 GB machine

**What happened:** `run_all.py` scores every unit on a CPU. The two `webAI-ColVec1-9b` units are 7.2 GB
(v3_hr) and 8.7 GB (v3_computer_science) of float16 vectors, 14.4 and 17.4 GB as float32. They did
not fit this machine's memory, and the first runs were stopped before the machine ran out of memory.

**Changes (all in `scripts/easy_validation/run_all.py`, which is not frozen; no frozen file changed,
and none changes any number):**

- `--min-free-gb` (default 1) stops the run cleanly when free memory runs low.
- `EASY_MAX_LIVE_GB` (default 3) caps the vectors held in memory at once; a smaller cap only evicts and reloads from disk.
- `--skip-models` reports the named models' units as not scored. It was used with `webAI-Official/webAI-ColVec1-9b`.

**What is not scored, and why:**

- `webAI-ColVec1-9b` on v3_hr and v3_computer_science: too large for this machine. V1u was therefore
  scored on `EVIE-Preview-4.5B` alone: point >= T on 8 of 8 units. V1u is reported as **not scored**,
  and no pass is claimed for the second held-out encoder. It can be scored on a machine with 32 GB or
  more of memory, or in a Kaggle session.
- `colSmol-500M` on v3_pharmaceuticals (9 of 12) and v3_industrial (8 of 12) failed `check_retriever`.
  They have no usable vectors, by the frozen rule. This is why V1a-V1d, V1-unchecked, V2, arm S and V3
  read "not scored" although every unit that exists passed or failed on its own numbers.

**`part3.py` was written after the data came back.** It implements the DT campaign and the
practice-question allowance test of the frozen PREREG_EASY_PART3.md (written before the data) and is
not itself frozen. The rules, thresholds and datasets come from the frozen text; the code is
available for review in the repository.

**V3 is reported as missed, not as "not scored".** The check needs |delta point| <= 0.005 on 95% of
pairs; 82.4% of 102 pairs met it. One unit lacks data, so the formal label is "not scored", but the
available pairs already fall short.

## D7 (2026-11-13): the ledger was applied in part

The decision table (`reports/easy/results/decisions.json`) was applied as written where the code reads
the setting:

- the 1-bit margin is removed (0.025 -> 0.0): arm B passed V1b and V1c with no severe 1-bit miss, on
  176 units (every arm had 0 misses);
- the practice check is now `warning`, because V5b failed on text-layer questions and V5c failed on
  OCR and VLM questions; it reports and does not change the layout;
- practice questions are described as "not reliably easier or harder than real questions" (V4 failed
  on all three kinds: 33%, 35% and 45% of units).

Two changes passed their tests but are **not switched on**:

- type presets: the DT test checks that the standard rule's picks keep `low >= target` on each document
  type's datasets. They did for **academic** (32 of 32 units; 3 datasets, 4 models; worst margin +0.010) and
  did not for slides (28 of 28 but only 3 of 4 models, since colSmol-500M failed its check on pharmaceuticals),
  reports (13 of 16) or tables (26 of 32). The test produces no type-specific numbers, so there is no preset
  that changes the rule. The academic result is documented as evidence only;
- the probe allowance: passed for text-layer (24 cases) and OCR (26 cases) questions, failed for VLM
  questions (15 cases, 20 needed). It has no effect while the practice check is a warning, so it stays off.

Both are in-sample results on our own validation sets.
V6 (the wiring test) failed, and its planted-fault half did not run (D2); it gates no decision.

## D8 (2026-11-13): Qdrant export and byaldi import switched on; the small-memory layout left off

- **Qdrant export: on.** Its gate test (`tests/test_easy_qdrant.py`) passed: top-10 equality 100% between Qdrant's MaxSim over the
  exported collection and the index's own exact search, on synthetic corpora in qdrant-client's local mode, float32 storage. The
  float16 option is outside the gate.
- **byaldi import: on.** Its gate test (`tests/test_easy_byaldi.py`) passed: an index folder written by the test's own writer, which
  follows byaldi 0.0.x's source, comes back bit for bit in byaldi's order, and a search over it ranks as MaxSim over byaldi's tensors.
  The test has not run against the byaldi package, which is not installed here.
- **Small-memory layout: off.** V9 ran at 20,000 pages on synthetic vectors (`reports/easy/v9/`). The layout's two-tier search
  matched the campaign's in-RAM two-tier path bit for bit (64 of 64 queries), and its private memory (318 MB and 140 MB) stayed within the
  bound (522 MB and 342 MB). Its working set (1.71 GB and 1.52 GB) was above the bound, because it counts the 8-bit file's pages that the
  operating system keeps cached. The design's "resident" is read here as the working set, so V9 is reported as not passed. Its top 10 matched
  plain 8-bit search on 15% to 30% of results (top 1 on all 64 queries); the layout is a shortlist by design, and no equivalence with
  plain 8-bit search is claimed.
