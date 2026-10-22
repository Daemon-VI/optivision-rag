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
