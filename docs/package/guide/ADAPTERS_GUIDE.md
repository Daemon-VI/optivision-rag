# Bringing your own encoder

Status: 2026-10-06, branch `package-universal`. A short how-to for getting vectors in. Reference
detail (the contract, identity, the model registry, padding, normalisation) is in
[ADAPTER_ARCHITECTURE.md](../ADAPTER_ARCHITECTURE.md). Terms are in [CONCEPTS.md](CONCEPTS.md).

The package never needs to call your model. It needs one `[n_i, d]` float array per document and
per query. An adapter is the thin, checked bridge from a model to those arrays. That an adapter
exists for a model says nothing about how well any pipeline compresses that model's vectors. That
has to be measured on those vectors, in the basis they are in, and a registry status of
`measured` for one model transfers to no other.

## 1. Pick the way in

| you have | use |
|---|---|
| arrays an encoder already produced | `ArrayAdapter(identity)`: `encode_documents(arrays, ids=...)`, `encode_queries(...)` |
| a function that encodes | `CallableAdapter(encode_documents, encode_queries, identity)` |
| a `MultiVectorCorpus` already | nothing: pass it to `certify` as it is |
| a model family the repository has a loader for (ColPali, ColQwen, ColSmol, Nemotron ColEmbed, sentence-transformers models) | `load_adapter(model_id)`; needs the model libraries; the registry status says how far that model was tested |
| something else | subclass `EncoderAdapter`; route arrays through `corpus_from_arrays`; override `identity()` |

## 2. What an adapter does and does not do

An adapter encodes, returns a `MultiVectorCorpus` with one document per input, drops padding rows
before handing vectors over, says which dtype comes out, and describes the model it wraps.

It never chooses or alters compression settings by model: nothing in the core reads an identity.
It never hides a device transfer or a dtype change. `ArrayAdapter` refuses a tensor on an
accelerator unless you pass `allow_device_transfer=True`, refuses bfloat16, refuses an item that is
a Python list, and refuses a mix of dtypes instead of promoting them.

- **Padding.** A batch is padded to a common length, and padded positions are not vectors. A
  padded row scored as a vector changes MaxSim. Drop padded rows with the attention mask. A 3-D
  array is refused as a probable padded batch.
- **Normalisation.** The registered experiments covered unit-length vectors. The corpus type does
  not require unit length and several codecs assume it. Vectors that are not unit length are
  outside what was tested.
- **Identity.** `EncoderIdentity(model_id, dimension, modality, revision=None, normalized=None,
  inference_dtype=None, extra={})` is descriptive. A certificate may record it. No decision depends
  on it, and it is never verified. State only the facts you have.

## 3. Arrays you already have

```python
import numpy as np

from optivision.adapters import ArrayAdapter
from optivision.core.vectors import ContractError

adapter = ArrayAdapter()  # an identity is optional
docs = adapter.encode_documents([np.ones((3, 8), dtype=np.float32), np.ones((2, 8), dtype=np.float32)])
print(len(docs), "documents,", docs.num_vectors, "vectors, ids", docs.ids)

mixed = [np.ones((3, 8), dtype=np.float32), np.ones((2, 8), dtype=np.float16)]
batch = np.ones((4, 5, 8), dtype=np.float32)  # a padded batch, as many models return it
for bad in (mixed, batch):
    try:
        adapter.encode_documents(bad)
    except ContractError as exc:
        print("refused:", str(exc)[:70])
```

```text
2 documents, 5 vectors, ids ['0', '1']
refused: documents[1] has dtype float16 but documents[0] has float32; mixed dty
refused: documents: expected a sequence of [n_i, d] arrays, got a ndarray of sh
```

## 4. A function that encodes, and the conformance check

`check_adapter` calls each `encode_*` method twice on a few real inputs and returns a list of
problems. An empty list means the output is well formed: one document per input, finite floating
vectors of one dimension, unique ids, `ids=` honoured, a repeat call identical, no all-zero rows,
and an identity that agrees with the vectors. It says nothing about quality.

The stand-in encoder below maps text to deterministic random unit vectors. It says nothing about
any model.

```python
import hashlib

from optivision.adapters import CallableAdapter
from optivision.core.vectors import EncoderIdentity
from optivision.testing import check_adapter

DIM = 16


def embed(text: str) -> np.ndarray:
    """Stand-in for a model call: one [n_i, DIM] float32 array, no padding rows."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal((3 + len(text) % 5, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


identity = EncoderIdentity(model_id="stand-in", dimension=DIM, modality="text",
                           normalized=True, inference_dtype="float32")
adapter = CallableAdapter(lambda items: [embed(t) for t in items],
                          lambda items: [embed(t) for t in items], identity)

documents = [f"document number {i}" for i in range(5)]
print("conformance problems:", check_adapter(adapter, documents, ["number 1", "number 2"]))
corpus = adapter.encode_documents(documents, ids=[f"d{i}" for i in range(5)])
print(len(corpus), "documents,", corpus.num_vectors, "vectors, dimension", corpus.dimension)
```

```text
conformance problems: []
5 documents, 25 vectors, dimension 16
```

From here, `certify(corpus, queries, qrels)` takes the corpora as they are ([QUICKSTART.md](QUICKSTART.md)).
The script is `examples/bring_your_own_encoder.py`.

## 5. Before you rely on the vectors

1. Run `check_adapter` on a few real inputs.
2. Measure on your vectors. `certify` does that, and may abstain.
3. If you re-encode (another model revision, precision or pooling), measure again: the certificate
   is bound to the vectors as given ([BASIS_DEPENDENCE.md](BASIS_DEPENDENCE.md)).
4. A new registry entry starts as `unverified` or `untested`. Raise it only with a committed
   measurement and its evidence.
