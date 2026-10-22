# DRAFT — NOT FROZEN: PREREG_EASY part 2 (probes, hard set, statistic, seeds)

Frozen before B1 (`freeze.py --part 2`). Not frozen. The integrator freezes it once the DRAFT mark is removed.

- Probe generator: `src/optivision/easy/_probes.py` (text layer first, then OCR, then VLM), page sources ['text_layer', 'ocr', 'vlm'].
- Probes per page 3, at most 600 pages per corpus, hard-set minimum 60, seed 0.
- Practice-question bootstrap B = 2000, seed 0.
- Planted faults (V5a): {"random scores": null, "project(16) > binary": {"stages": [{"dim": 16, "method": "pca", "stage": "project"}, {"stage": "binary"}]}, "adaptive_merge(0.6) > int4": {"stages": [{"center": "mean", "max_vectors": null, "min_vectors": 1, "n_importance_seeds": 0, "radius": 0.6, "ratio": null, "refine": 0, "representative": "mean", "seeding": "farthest", "stage": "adaptive_merge"}, {"stage": "int4"}]}}, seed 0.
- Veto: `src/optivision/easy/_veto.py` and `presets/veto_strength.json`.
- Wiring test V6: one representative per registry loader that loads on Kaggle (`units.V6_LOADERS`), plus the faults in
  `units.V6_FAULT_MODELS`.
- Held-out selection runs in e1b and is recorded in `heldout_selected.json` before e3b.
