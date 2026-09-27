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

Timings were taken with several jobs sharing four cores; alone they run faster.

Then render the tables:

```bash
python scripts/universal_tables.py reports/universal > reports/universal/TABLES.md
python scripts/universal_summary.py reports/universal
```
