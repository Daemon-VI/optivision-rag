"""Bring your own encoder: wrap two functions, check the contract, hand the vectors on.

The 'encoder' is a stand-in that maps text to deterministic random unit vectors. It exercises
the code and says nothing about any model.

    PYTHONPATH=src python docs/package/guide/examples/bring_your_own_encoder.py
"""

import hashlib

import numpy as np

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
queries = ["number 1", "number 2"]
print("conformance problems:", check_adapter(adapter, documents, queries))

corpus = adapter.encode_documents(documents, ids=[f"d{i}" for i in range(5)])
print(len(corpus), "documents,", corpus.num_vectors, "vectors, dimension", corpus.dimension)

# A padded batch is refused: padding rows would be scored as if they were vectors.
padded = CallableAdapter(lambda items: np.zeros((len(items), 7, DIM), dtype=np.float32),
                         lambda items: np.zeros((len(items), 7, DIM), dtype=np.float32))
for problem in check_adapter(padded, documents, queries):
    print("-", problem[:96])
