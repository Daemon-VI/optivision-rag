# a06 import/dependency audit

## 1. Scope and what was executed
Worktree a06 (HEAD 7a6769a), no source changes, no commits.
- sys.modules diffs in fresh subprocesses: `import optivision`, `import optivision.core`, `import optivision.api`, and `optivision.<name>` for each of the 11 lazy names.
- `python -X importtime -c "import optivision"`; wall-clock timing 3x each of PIL+yaml, numpy, optivision.
- `git archive HEAD` -> `python -m build --wheel` (optivision_rag-0.3.0) -> fresh venv in scratch, `pip install` of the wheel only (no extras). Installed: numpy 2.4.6, pillow 12.3.0, pypdfium2, PyYAML, qdrant-client, typer, rich. Absent: scipy, torch, transformers, sklearn, colpali. Ran with `python -I` from outside the repo: import, `certify`, `verify`, `optivision --help`.

## 2. Findings
| ID | Sev | Where | Problem / evidence | Fix | Status |
|---|---|---|---|---|---|
| A06-01 | NO ACTION | `src/optivision/__init__.py:15-52` | `import optivision` and `import optivision.core` both load PIL, yaml, numpy and nothing else heavy (no torch, scipy, sklearn, transformers, qdrant_client, typer, rich, pypdfium2). All 11 lazy names give the same set; those touching `api`/`artifact` add `importlib.metadata`. PIL and yaml are HARD dependencies (`pyproject.toml:31-39`), so this cannot break an install. Chain: `__init__` -> `.calibration` (pinned) -> `.stages` -> `stages/prune.py` -> `pruning/__init__.py` (`from PIL import Image`) and `pruning.redundancy` -> `config.py` (`import yaml`); also `__init__` -> `.pipeline` (`from PIL import Image`), which is one of the 40 eager names (OptiVisionRAG). | none | NOT A DEFECT |
| A06-02 | NO ACTION | design D2, `tests/test_core_layering.py` docstring, `guide/ARCHITECTURE.md:75-90` | Documented, accepted deferral: "eager for the 40 old names ... numpy-only closure deferred; needs extras split". The ARCHITECTURE.md snippet prints `['PIL','numpy','yaml']` and matches what I measured. The layering test is explicitly source-level only and says so. | none | NOT A DEFECT |
| A06-03 | NO ACTION | cost | PIL+yaml import ~270-320 ms in isolation (cold, laptop loaded by 13 other agents); `import optivision` total 684-822 ms; numpy alone 246-395 ms. In the importtime run: PIL.Image 62 ms cum, yaml 42 ms cum, of 472 ms total. So PIL+yaml is roughly 20-35% of import time; numbers are noisy. | none | leave |
| A06-04 | NO ACTION (docs already cover it) | `api.py:365` | In the core-only install, `certify(vecs, queries)` with the DEFAULT space raises a clear ImportError (needs SciPy, extra `[merge]`; suggests `allow_reduced=True`). This is intentional ("never reduced silently") and is stated in README:165,176,417, QUICKSTART:11, CONCEPTS:148, LIMITATIONS:51,217. With `space=CandidateSpace.default(16, allow_reduced=True)` certify returns a CertifiedResult (outcome ABSTAIN on random data), `verify(cert, corpus=, queries=)` runs and prints its report, and no scipy/torch/etc. appears in sys.modules. Users must know the default path needs `[merge]`; the error message tells them. | none | NOT A DEFECT |

Core-only install results: `import optivision` OK; `optivision.certify` default space -> actionable ImportError (SciPy); reduced space OK; `verify` OK; `optivision --help` OK (typer/rich only loaded by CLI, not by import).

## 3. Why no fix
A "minimal deferred-import" fix does not achieve its goal. Moving PIL/yaml out of `pruning/__init__.py`, `config.py` would still leave `from PIL import Image` in `pipeline.py`, which `__init__` imports eagerly for `OptiVisionRAG`; deferring that changes the v0.3.0 module namespace (`optivision.pipeline.Image`) and annotations. The chain also goes through pinned `calibration.py` -> `stages`. Removing PIL/yaml from `import optivision.core` is exactly the deferred lazy-`__init__` / extras-split decision (D2). Both are hard dependencies, so users are unaffected apart from ~0.3 s.

## 4. Checked and correct
Lazy-name table complete (11 names, all resolve); no optional/heavy module leaks into plain import or reduced-space certify; CLI works in core-only install; docs state the SciPy requirement consistently.

## 5. Not checked
Python versions other than 3.11, non-Windows, a pre-built-wheel from PyPI (only a local build), import time on a quiet machine.
