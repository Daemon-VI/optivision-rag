# Change Log

## 0.1.0

First release.

- Control panel with a form per CLI command: init-config, make-corpus, index, search, stats,
  explain, plus bench and fetch-vidore under an Advanced section.
- Auto-detects the CLI (`optivision` on PATH, else a Python launcher running
  `optivision.cli`); one-click `pip install optivision-rag` when neither is found.
- Explain renders its output figures inline in the panel.
- Every CLI call runs without a shell (`spawn(..., { shell: false })`).
