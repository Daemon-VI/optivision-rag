"""The easy-mode validation programme (FINAL_DESIGN section 8; reports/easy/PREREG_EASY*.md).

``units.py`` lists the corpora, models, Kaggle batches and every threshold; ``encode_unit.py``
runs on a Kaggle GPU and saves float16 vectors, qrels, practice-question probes and run
records; everything else scores the predictions on a CPU from those saved files
(``run_all.py``). Nothing in ``src/optivision`` imports this folder.
"""
