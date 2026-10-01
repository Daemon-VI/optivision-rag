# Universal-layer study scripts

The exact scripts that produced every number in `docs/UNIVERSAL.md` and
`reports/universal/`. Only their input and output paths were made configurable
when they were copied into the repository.

```bash
export OPTIVISION_DATA=data                  # encode caches, query files, SciFact vectors
export OPTIVISION_OUT=reports/universal/raw  # where each script writes its JSON
```

## Inputs

| input | how to get it |
|---|---|
| `data/cache/colsmol.npz`, `data/cache/colpali_generated.npz` (+ `.queries.npz`) | `optivision bench ... --cache` on the generated corpus (CPU for ColSmol, GPU for ColPali) |
| `data/cache/colpali_{infovqa,docvqa}_test_subsampled.npz` (+ `.queries.npz`) | `scripts/run_bench_gpu.sh` (Kaggle / any CUDA box) |
| `data/corpus/queries.json` | `optivision make-corpus data/corpus --docs 30 --pages 2` (seed 7) |
| `data/vidore_infovqa/queries.json`, `data/vidore_docvqa_test_subsampled/queries.json` | `scripts/rebuild_vidore_qrels.py <split> --out ... --check <cache>.queries.npz` |
| `data/scifact/{corpus,queries}.json`, `qrels_test.tsv` | BEIR SciFact from `BeIR/scifact` and `BeIR/scifact-qrels` on Hugging Face |
| `data/vectors/scifact_*` | `s9_encode_scifact.py` in an environment with `sentence-transformers>=6` |
| `data/vectors/colqwen2*_{docs,queries}.npz`, `_qrels.json` | `notebooks/kaggle_colqwen_encode.ipynb` (Kaggle T4, `docs/GPU_RUN.md`) or `scripts/encode_vectors.py` on any GPU |

Files written by `scripts/encode_vectors.py` are passed to `s3c`, `s5`, `s7b` and `s12` as `vec:<prefix>`, for example `vec:colqwen2.5-v0.2_docvqa_test_subsampled` (`optivision.benchmark.load_vector_dataset`).

## Order

| script | what it measures | this laptop (i7-1165G7, CPU) |
|---|---|---|
| `s2_equivalence.py` | the original nine-variant table rebuilt from stages, against the committed reports | ~10 min |
| `s3_merge_sweep.py` | token reduction methods at matched budgets | ~40 min |
| `s3b_centred_merge.py` | adaptive merging with the corpus mean removed | ~20 min |
| `s4_calibration_study.py` | 20 random calibration / held-out splits per target, reference and rule; pseudo-query proxy | ~70 min |
| `s5_frontier.py` | token stage x codec grid and its Pareto frontier | ~2 h |
| `s6_quantize_project.py` | codecs alone, projections alone, merge + codec combinations | ~40 min |
| `s8_tiered.py` | hot binary shortlist + cold rescoring | ~30 min |
| `s9_encode_scifact.py`, `s9_text_study.py` | the same measurements on a text ColBERT | ~15 min encode, ~60 min study |
| `s16_performance.py` | compression time, peak traced memory, scan time | ~10 min |
| `s3c_centred_codecs.py` | centred int8 / int4 / binary (R3b); `--geometry` for the anisotropy diagnostics (R8) | ~20 min |
| `s7b_selection_rules.py` | every selection rule, including `calibrate()`'s own default path, on the default search space (R7b) | ~35 min per ViDoRe split, longer for SciFact |
| `s10_bound_audit.py` | coverage of the lower bound, many-candidate and degenerate cases (R10, release audit) | ~30 min |
| `s11_ties.py` | whether index tie-breaking changes retention (release audit) | ~15 min |
| `s12_corpus_size.py` | retention as distractors are added (release audit) | ~10 min |

Timings were taken with several jobs sharing four cores; alone they run faster.

Each script writes under `$OPTIVISION_OUT` in the folder it was first run with;
copy the JSON to the matching folder of `reports/universal/`:

| script | writes | committed as |
|---|---|---|
| `s2` | `phase2/legacy_equiv/` | `equivalence/` |
| `s3` | `phase3/results/` | `merge/` |
| `s3b` | `centred/results/` | `centred/` |
| `s3c` | `centred_codecs/`, `geometry/` | same |
| `s4` | `phase4/results/` | `calibration/` |
| `s5` | `frontier/results/` | `frontier/` |
| `s6` | `phase67/results/` | `quantize_project/` |
| `s7b` | `selection_rules/` | same |
| `s8` | `phase8/results/` | `tiered/` |
| `s9` | `phase9/results/` | `text/` |
| `s10` | the path given as its argument | `audit/bound_audit.json` |
| `s11`, `s12` | `audit/` | same |
| `s16` | `perf/results.json` | `performance/` |

Then render the tables:

```bash
python scripts/universal_tables.py reports/universal > reports/universal/TABLES.md
python scripts/universal_summary.py reports/universal
```
