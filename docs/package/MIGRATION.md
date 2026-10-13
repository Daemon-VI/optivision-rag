# Migrating from v0.3.0

Status: 2026-10-06, updated 2026-10-12 for version 0.4.0 (`__version__` is `"0.4.0"`). No
deprecation is made by this work.

**A v0.3.0 program needs no change.** `optimize()`, `calibrate()`, the 40 v0.3.0 names of
`optivision.__all__`, every old import path, the CLI commands and options, the registry names
and the v1 file formats are as they were. What follows lists every observable difference found
(by comparing the source tree with tag `v0.3.0`), how to move from `optimize()` to the new
`certify()` and what changes in meaning when you do, the file formats, and a note for people
who run the registered research scripts. The design of `certify` is in `API_DESIGN.md`.

## 1. Observable differences

| what | detail | matters if |
|---|---|---|
| `optivision.adapters` is a package | `adapters/__init__.py` (the old module, moved with `git mv`) plus `arrays.py` and `nemotron.py`. Same names, same objects; `__file__` and `__path__` differ | you read the module's file path, or ship a stub that shadows it |
| 12 new names in `optivision.__all__` | `CandidateSpace`, `Certificate`, `CertifiedResult`, `Decision`, `EncoderIdentity`, `Outcome`, `certify`, `encode_and_certify`, `load_artifact`, `save_artifact`, `verify`, `verify_certificate` (52 names in all). They load on demand (PEP 562) | you run `from optivision import *` (it resolves every name, so it loads the certificate layer) |
| new registered stage `rotation` | `RotationProjector`, from `optivision.core.stages` only (`optivision.stages` and `__all__` do not list it). `STAGES` has 16 entries, the message of `stage_from_dict` for an unknown name lists it, and the shorthand of `optivision compress` accepts `rotation(seed=3)`. No existing stage name, parameter or default changed; the hash of the registered default list is unchanged | you enumerate `STAGES` |
| `load_adapter` for the NVIDIA ColEmbed family | A model id whose registry loader is `"nemotron"` now returns a `NemotronAdapter` (needs torch and transformers). In v0.3.0 it fell through to `SentenceTransformersAdapter`, which does not know that checkpoint | you called `load_adapter` on such an id |
| `load_adapter` for `nomic-ai/colnomic-embed-multimodal-7b` | Builds a colpali-engine page adapter (loader `colpali-engine:colqwen2.5`, status `untested`) instead of the sentence-transformers adapter (`unverified`) | you called `load_adapter` on that id |
| `ModelInfo` has three more optional fields | `license`, `revision` and `notes`, defaulting to None, None and an empty string. Positional construction with the 7 earlier arguments still works | you construct `ModelInfo` or compare it field by field |
| the registry is larger | `MODELS` and `supported_models()` return 65 entries (12 in v0.3.0): 10 measured, 23 untested, 32 unverified | you iterate the registry and expect the earlier list |
| the status of `nvidia/nemotron-colembed-vl-8b-v2` | Now `measured` (float16 inference), from the committed runs under `reports/universal/wide/` | you read the status of that id |
| the certificate records version 0.4.0 | The version field and, through the environment record, the digest of a certificate differ from those of an earlier build for the same decision and evidence | you compare certificate digests across versions |
| `load_adapter` has a `loader` argument | `load_adapter(model_id, loader=None, **kwargs)`; without it the call behaves as before | nothing, unless you pass a third positional argument |
| adapters describe themselves | `EncoderAdapter.identity()` and `.dimension` (optional: a subclass without them keeps working); new `ArrayAdapter`, `CallableAdapter`, `corpus_from_arrays` for bringing your own vectors, and `NemotronAdapter`. Corpora they build carry `attrs["encoder_identity"]` (and `attrs["model"]`), which `.npz` files keep in their header | you compare `attrs` dicts exactly |
| two new CLI commands | `optivision certify` and `optivision verify`, registered by `optivision.cli` (so also under `python -m optivision.cli`). No existing command or option changed | you parse the output of `optivision --help` |
| new packages and modules | `optivision.core` (identity, vectors, pipeline, similarity, stages, retrieve, evaluate, selection, artifact), `optivision.backends`, `optivision.api`, `optivision.cli_certify`, `optivision.testing`, and the namespaces `optivision.ext.vision` and `optivision.legacy`. Nothing was moved or copied: `optivision.backends.NumpyIndex is optivision.index.numpy_index.NumpyIndex`, and so on | you build your own wheel list (new subpackages carry `__init__.py`) |
| import time | `import optivision` now also imports `optivision.core.stages`, which registers `rotation` and the stage traits. It still loads numpy, PIL and yaml and no other third-party package: not SciPy, torch or qdrant_client. Measured (below): no change beyond noise. Accessing one of the new names loads the certificate layer on demand | you are sensitive to start-up time |
| `ExactIndex.rescore` is batched | Present in the base of this branch, not in the tag: each candidate document is decoded once. Arithmetic, column order and API unchanged (see the `research-investigation` entry of `CHANGELOG.md`). The scores are not bitwise those of v0.3.0 in general: batched and per-query rescoring give the same rankings except between scores closer than the float32 rounding of the dot products, and scores equal to within a few float32 ulps (largest absolute difference seen in a synthetic comparison with v0.3.0: 3.8e-6, binary codes). Bitwise equality holds on some BLAS kernels only: measured equal on the AVX-512 kernel of the development laptop, about one ulp different on an AVX2-class kernel (the CI runner), so `TieredIndex.score` values can differ in the last bits as well | you compared timings, or compare scores with `==` or hash them |
| `PageEncoderAdapter` checks the ids | `encode_documents` raises `ValueError` when `len(ids)` differs from the number of encoded pages. v0.3.0 returned a corpus whose `ids` did not match its length without complaint | you passed an `ids` list of the wrong length |
| `python -m optivision.adapters` | A no-op in v0.3.0 (the module ran silently); it is a package without `__main__` now, so the command fails with "No module named optivision.adapters.__main__" | you ran it |
| documentation edits only | `index/numpy_index.py` and `index/qdrant_index.py` differ from the tag in docstrings and comments, not in code | never |

Import time, `python -c "import optivision"` in a fresh interpreter, median of 25 runs
interleaved between the tag-equivalent `__init__.py` and this one, on one laptop CPU, Python
3.11.9, quiet machine: 281 ms before (minimum 244), 278 ms after (minimum 248). With
`optivision.certify` accessed as well: 355 ms (minimum 288), so loading the certificate layer
costs about 77 ms, around half of it `importlib.metadata`. Differences under about 10 ms are
noise on a shared machine. Reproduce with `python -X importtime -c "import optivision"`.

## 2. From `optimize()` to `certify()`

Moving is a choice of what you want stated, not a spelling change. Keep `optimize()` when an
empirical choice with a held-out check is what you need.

| `optimize(...)` | `certify(...)` |
|---|---|
| `vectors`, `queries=`, `qrels=`, `metric=` | the same, with `queries` positional and required |
| `quality_target=0.97` | `target=0.97`, and a new `error=0.05` (the level of the statement) |
| `search_space={family: [Pipeline, ...]}` | `space=` takes that dict, or a `CandidateSpace`, or `None` for the default list |
| `reference="auto" / "labels" / "baseline"` (through `**calibrate_kwargs`) | `reference="auto" / "labels" / "agreement"` (`"baseline"` is spelled `"agreement"`), or a reference object |
| `calibration_fraction`, `seed`, `n_boot`, `safety`, `confidence`, `multiplicity`, `margin`, `max_bytes_per_doc`, `assume_monotone` | no equivalent: there is no split, no bootstrap, no byte cap and no early stop |
| (none) | `rule`, `query_budget`, `order_seed` (default 0), `fallback`, `encoder`, `progress`, `created` |

What changes in meaning:

- **The target.** `quality_target` is a threshold that, by default, a lower bound of retention measured on the
  calibration half has to reach. `target` is the retention to be exceeded, and what the result states is
  the rule's statement at `(target, error)`, claimed only under assumptions A1-A6 and conditions
  C1-C4. The same number does not mean the same thing, in either direction.
- **No held-out figure.** All queries are tested. There is no `quality_retention` and no
  `target_met`; `certified` says that the rule certified the choice, not that a held-out
  retention reached the target.
- **Nothing qualifies.** `optimize()` returns your corpus as a float32 copy with
  `target_met is None`. `certify()` returns `Outcome.ABSTAIN`: `compressed` and `pipeline` are
  `None`, nothing is copied, and `store` is your corpus with its own dtype. Code that reads
  `result.compressed` must handle `None`; `result.store` and `result.retriever()` work for every
  outcome.
- **Repeating.** Re-running `optimize()` with another target is an ordinary new empirical choice.
  Re-running `certify()` on the same queries with another target, error level or list is not
  covered by the statement; use fresh queries.
- **The order of the queries.** `optimize()` splits the queries into two halves with a seed.
  `certify()` tests all of them in a seeded random order (`order_seed=0`, recorded in the
  certificate); `order_seed=None` tests them in the order given, which is valid only if that
  order is unrelated to the outcomes. Trying several seeds and reporting one is outside the
  statement.
- **Warnings.** `calibrate` warns below 100 informative queries and, without SciPy, continues
  with a smaller default list. `certify` never warns about few queries (it abstains) and
  raises `ImportError` without SciPy instead of reducing the list. It refuses queries
  that repeat an earlier query exactly unless you pass `allow_duplicate_queries=True`, which
  asserts that they are independent draws and is recorded in the certificate (the count is in
  the certificate and in `report()`). It gives a `UserWarning` when it is called again in the
  same process on the same queries and corpus with another target, error level,
  query order (seed or budget), candidate space, metric, reference or rule.

| `OptimizationResult` | `CertifiedResult` |
|---|---|
| `.compressed`, `.pipeline` | the same names; `None` on ABSTAIN |
| `.compression_ratio` | the same definition; 1.0 when nothing was compressed |
| `.memory_saving` | `1 - 1 / result.compression_ratio` |
| `.target_met`, `.quality_retention` | none; see `.certified`, `.outcome`, `.report()` |
| `.calibration` | `.decision`, `.evidence`, `.certificate` |

```python
import optivision
from optivision import certify
from optivision.testing import make_corpus, make_queries

docs = make_corpus(60, 16, tokens=(8, 12), seed=0)  # synthetic stand-in for your vectors
queries, qrels = make_queries(docs, 150, noise=1.0, seed=1)

old = optivision.optimize(docs, queries=queries, qrels=qrels, quality_target=0.9)  # unchanged
print(old.pipeline.label(), old.compression_ratio, old.target_met)

new = certify(docs, queries, qrels, target=0.9, error=0.05)
print(new.outcome, new.certified, new.compression_ratio)
ranking = new.retriever().search(queries, k=5)  # the chosen index, or yours on ABSTAIN
print(ranking.ids[0])
```

## 3. File formats

Unchanged and still read: `optivision-multivector` v1 (`MultiVectorCorpus.save` / `load`),
`optivision-compressed` v1 (`save_compressed` / `load_compressed`), the NumPy index directory,
the encode, query and float caches, `manifest.json`, `Config` YAML and the Qdrant payload and
point ids. Fixtures written by v0.3.0 are read back in `tests/test_golden_formats.py`.

Additional, versioned, written only by the new layer and never pickled: the artifact directory
(`certificate.json`, `codes.npz`, `fitted.npz`, `evidence.npz`, `MANIFEST.json`;
`optivision-artifact` v1), the certificate (`optivision-certificate` v1), the evidence file
(`optivision-evidence` v1), the candidate space and the decision record (version 1). The
artifact exists because `save_compressed` cannot store fitted state (a PCA basis, a codec's
mean); that format is not changed and keeps refusing such codecs.

## 4. Deprecations

None. Nothing was deprecated, renamed or removed by this work, and no warning was added to any
existing call.

## 5. If you run the registered research scripts

The research harness pins 11 library files by content hash (`scripts/research/x_common.py`,
checked by `tests/test_x_harness.py`): `calibration.py`, `compose.py`, `scoring.py`,
`evaluation.py`, `representation.py`, four files under `stages/` and two under `compression/`.
They are byte-identical to the registered ones, as are `optimize.py` and the stage and codec
names and parameters that make up the registered default list. The package adds files, so the
library tree identity differs from the one the registered runs recorded
(`0b9523bd10f43941a9704cbf6fae1db1804880ee`). X1, X2 and X3 were not re-run and XH was not run.
A new registered experiment must record the tree it runs on; running one on this branch needs a
new dated amendment, not a silent re-pin. The package never imports research scripts (a test
enforces it). See `FINAL_RESEARCH_BOUNDARY.md`, section 10.
