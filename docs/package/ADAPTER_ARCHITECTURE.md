# Adapters and the model boundary, as built

Status: 2026-10-06, branch `package-universal`. This describes `src/optivision/adapters/`,
`src/optivision/ext/`, `src/optivision/legacy/` and `src/optivision/testing/`. Contract:
`UNIVERSAL_PACKAGE_DESIGN.md` D1, D2, D16 and section 4. Audit: `ARCHITECTURE_AUDIT.md`
sections 3, 4.4, 8 and 9.3.

Nothing here is a research result. No model was loaded, encoded or scored to write it, and no
statement below is about how well any model's vectors compress. Labels: **ENGINEERING**
(pinned by a test in this repository, on synthetic data or stubs), **NOT MEASURED**.

## 1. What was built

| name | file | note |
|---|---|---|
| `EncoderAdapter.identity()`, `.dimension` | `adapters/__init__.py` | concrete, defaults from `info`; old subclasses keep working |
| `PageEncoderAdapter.identity()`, `SentenceTransformersAdapter.identity()` | `adapters/__init__.py` | facts from the wrapped encoder or the registry; unknown stays `None` |
| `ArrayAdapter`, `CallableAdapter`, `corpus_from_arrays` | `adapters/arrays.py` | bring your own vectors; the one checked conversion |
| `NemotronAdapter`, the `load_adapter` branch for loader `"nemotron"` | `adapters/nemotron.py` | the branch was missing (audit 8.3) |
| `optivision.ext.vision`, `optivision.legacy` | `ext/vision/__init__.py`, `legacy/__init__.py` | namespaces: the same objects, resolved lazily |
| `optivision.testing` | `testing/*.py` | synthetic data, basis transforms, `basis_probe`, `check_adapter`, `check_stage` |

Not changed: `MODELS`, `_MODELS` and every status and evidence string; the `encode_*` behaviour of
`PageEncoderAdapter` and `SentenceTransformersAdapter`; `encoders/*`; the pinned files.

## 2. The contract

`EncoderAdapter` stays the abstract class it was: `encode_documents(documents, ids=None)` and
`encode_queries(queries, ids=None)`, each returning a `MultiVectorCorpus`.

| member | meaning |
|---|---|
| `encode_documents`, `encode_queries` | one corpus document per input, `ids=` honoured, padding rows already dropped |
| `info` | the registry `ModelInfo`, or `None` (class attribute, as before) |
| `identity()` | `EncoderIdentity` or `None`; default built from `info` when its dimension is known |
| `dimension` | from `identity()`, `None` when unknown; a subclass may assign it |

**An adapter is responsible for:** encoding; returning a `MultiVectorCorpus`; dropping padded rows
(with the model's attention mask) before handing vectors over; stating what dtype comes out and
doing every device or dtype conversion explicitly; a descriptive identity limited to facts it
has; carrying layout, if the model exposes it, through the corpus (section 6).

**An adapter must never:** choose or alter compression settings by model (nothing in the core
reads an identity, and no adapter picks a pipeline); hide a device transfer or a dtype
conversion; let `MultiVectorCorpus.from_arrays` promote mixed dtypes; guess a revision or a
normalisation flag; attach a registry status it was not given.

## 3. Three ways in

**Precomputed arrays.** `ArrayAdapter(identity=None, *, allow_device_transfer=False)` takes one
`[n_i, d]` NumPy array or CPU tensor per item. Every item goes through
`core.vectors.as_host_array`, so a tensor on an accelerator is refused unless
`allow_device_transfer=True`, bfloat16 is refused, and a list is refused. Checked: 2-D, floating,
one dimension (equal to `identity.dimension` if given), finite, unique ids, one dtype. The library's
`MultiVectorCorpus.from_arrays` promotes a mix of float dtypes to a common one (`numpy.result_type`:
float16 with float32 becomes float32); the adapter refuses a mix instead and keeps a single dtype
as given. A 3-D array is refused as a probable padded batch. The corpus owns its memory (the
conversion concatenates; peak is about twice the vectors).

**A callable.** `CallableAdapter(encode_documents, encode_queries, identity=None)` wraps two
functions that return one array per input. Same checks, plus the count must match the inputs.
Device, precision and batching are the functions' business.

**A loader for a known family.** `load_adapter(model_id, **kwargs)`: loaders `colpali-engine:*`
give `PageEncoderAdapter(ColVLMEncoder)`; loader `nemotron` gives `NemotronAdapter` (it used to fall
through to the sentence-transformers adapter), built with `expected_dim` from the registry unless
overridden; anything else gives `SentenceTransformersAdapter`.

**A custom adapter** subclasses `EncoderAdapter`, sends its arrays through `corpus_from_arrays`
and overrides `identity()`.

`NemotronAdapter` takes page images, or `(PageRef, image)` pairs as `iter_pages` yields them. It
wraps `NemotronColEmbedEncoder`, whose `_embed` masks, normalises and drops padded positions and
returns float32 whatever dtype the model ran in. It carries no `positions`, `protected` rows or
images (the encoder exposes no patch grid), so a pixel-space stage cannot run on its corpus and none
is faked.

## 4. Identity

`EncoderIdentity(model_id, revision, dimension, modality, normalized, inference_dtype, extra)` is
descriptive. A certificate may record it; no decision depends on it.

| adapter | `model_id`, `dimension` | `revision` | `normalized` | `inference_dtype` | `modality` |
|---|---|---|---|---|---|
| default (`info`) | registry | `None` | `None` | `None` | registry |
| `PageEncoderAdapter` | encoder `checkpoint`/`dim`, else registry | encoder `revision` if it has one | `None` | encoder `torch_dtype` if it has one | registry, else `image` |
| `SentenceTransformersAdapter` | registry only (its API is unverified here) | `None` | `None` | `None` | registry |
| `NemotronAdapter` | encoder | the pinned revision | `True` for the real class, which normalises each row; else `None` | encoder `torch_dtype` | `image` |
| `ArrayAdapter`, `CallableAdapter` | what the caller passed | the caller's | the caller's | the caller's | the caller's |

The registry status is not part of an identity, and the evidence strings (prose) are never parsed.
The corpora of `ArrayAdapter`, `CallableAdapter` and `NemotronAdapter` carry the identity as a plain
dict under `attrs["encoder_identity"]` (and `attrs["model"]`); `PageEncoderAdapter` and
`SentenceTransformersAdapter` write the `attrs` they always wrote, and the key `encoder` keeps the
string name `PageEncoderAdapter` gives it.

## 5. The model registry

`MODELS` is a table of `ModelInfo` entries with one status each:

| status | meaning |
|---|---|
| `measured` | retrieval with and without compression was measured by this repository on that model's real output; the numbers are in `reports/` |
| `untested` | a loading path exists, known to work for a sibling model; this checkpoint was never encoded or benchmarked here |
| `unverified` | written against a library's documented API; that library was never installed and run here |

A status changes only when a measurement is committed, in the same commit as the evidence string.
`tests/test_adapters.py` pins the set of `measured` ids and `tests/test_adapters_identity.py` pins
every entry's status. An adapter does not change a status, and writing one for a model is not a
measurement. `ArrayAdapter` never looks the registry up, so arrays labelled with a measured model's
name carry no status.

## 6. Where vision-specific logic lives

Layout reaches the core only as arrays and metadata. `PageEncoderAdapter` builds the corpus through
`MultiVectorCorpus.from_page_encodings`: `protected` (instruction and thumbnail tokens),
`positions` (patch centres in a 0..1 page frame, NaN off the grid), per-document `metadata`
(`grid_rows`, `grid_cols`, `image_size`, `page_ref`) and the optional `images` list. Core code never
sees a `PatchGrid`, a `PageEncoding` or a page image.

`optivision.ext.vision` lists the page surface (page types, `SpatialPruner` and the pixel/grid
pruning functions, `TokenPruner`, the page encoders, `PageEncoderAdapter`, `NemotronAdapter`,
`iter_pages`, the overlays). `optivision.legacy` lists the original recipe (`Config`,
`OptiVisionRAG`, `Compressor`, `CompressedPage`, `SearchHit`, `SearchResult`, the `bench`
harness); `TokenPruner` is in both. Both namespaces re-export the original objects, move no
code, keep every old import path, and resolve names on access, so importing them loads neither
torch nor transformers nor Qdrant (**ENGINEERING**: `tests/test_ext_namespaces.py`). The core must
not import them; `tests/test_core_layering.py` checks that.

## 7. Adding an encoder family

1. Pick the way in. If you only need to run your own model, a `CallableAdapter` is enough.
2. Write the two methods; route arrays through `corpus_from_arrays`; drop padding with the mask;
   say what dtype comes out; keep device moves explicit; import heavy libraries inside functions.
3. Implement `identity()` with the facts you have. Leave the rest `None`.
4. Test without weights, as `tests/test_adapters_nemotron.py` does: replace the loader and the
   processor with tiny stand-ins and run the real encoding code.
5. Run `optivision.testing.check_adapter(adapter, documents, queries)` (it calls each `encode_*`
   method twice). It returns a list of problems; an empty list means the output is well formed
   (unique ids, floating, finite, one dimension, `ids=` honoured, deterministic, no all-zero rows,
   identity consistent).
6. A new registry entry starts `unverified` or `untested`. Raise it only with a committed
   measurement and its evidence.
7. If the model has a patch grid, put it in `positions`/`protected`/`metadata`; do not invent one.
8. Measure on the real vectors, in the basis they are in, before saying anything about
   compression. `optivision.testing.basis_probe` shows how one pipeline moves under a change of
   basis on one data set (a diagnostic; engineering capability; see
   `reports/research/X_RESULTS.md` for the registered experiment on the question).

## 8. What an adapter's existence does not imply

An adapter makes a model usable with the interface; it says nothing about how well any pipeline
compresses that model's vectors: that has to be measured on that model's vectors, in the basis they
are in. A model-agnostic interface is not model-independent compression quality, and a status of
`measured` for one model transfers to no other. What any pipeline does to any model's vectors is
**NOT MEASURED** by this work: it built the interface, ran stand-ins and synthetic data, and loaded
no model.

## 9. Padding, masks and normalisation

**Padding.** A batch is padded to a common length; padded positions are not vectors. The core cannot
tell them apart, and a padded row scored as a vector changes MaxSim. The adapter drops them with
the attention mask before handing vectors over (`NemotronColEmbedEncoder._embed` does:
`emb[i][keep[i]]`). `ArrayAdapter` refuses a 3-D array; `check_adapter` reports all-zero rows as a
likely sign. Some late-interaction models pad queries with learned tokens on purpose; those rows are
vectors the model defines, and an adapter should keep what the model's documented procedure keeps
(record it in `extra` when it matters).

**Normalisation.** The research scope is MaxSim over L2-normalised vectors
(`reports/research/NEXT_PHASE_PREREGISTRATION.md` section 25, non-claim 2, and section 5). The corpus
type does not require unit norm and several codecs assume it. An adapter reports `normalized` only
when it knows; `check_adapter` verifies a `normalized=True` claim against the vectors. Vectors that
are not unit norm are outside what was tested, and nothing here says anything about them.

## 10. Known gaps, left unchanged

From `ARCHITECTURE_AUDIT.md` section 8.3, the items that concern encoders:

- `load_adapter` fell through for loader `"nemotron"`: **fixed here**, the only change to existing behaviour.
- `scripts/encode_vectors.py` still builds the Nemotron encoder directly (`--backend nemotron`); it
  now matches `load_adapter` but was not edited.
- `SyntheticEncoder` page vectors depend on encode order (stateful RNG). `check_adapter` reports
  `PageEncoderAdapter(SyntheticEncoder())` as not deterministic; with `noise=0.0` it conforms.

Also unchanged: `SentenceTransformersAdapter` converts model output with `.detach().cpu().float()`
(a silent device transfer and widening to float32), and `MultiVectorCorpus.from_arrays` itself moves
tensors to the CPU and widens bfloat16 (`as_float_matrix`) for any adapter that hands it tensors;
`ArrayAdapter` is the strict path. `ColVLMEncoder.dim` answers 128 when the model states none. The
registry holds no machine-readable revision. The sentence-transformers image path remains
unverified.

## 11. Tests

`test_adapters_arrays.py`, `test_adapters_identity.py`, `test_adapters_nemotron.py`,
`test_ext_namespaces.py`, `test_testing_helpers.py`; unchanged and still passing: `test_adapters.py`,
`test_nemotron_encoder.py`, `test_colvlm_loading.py`, `test_public_surface.py`.
