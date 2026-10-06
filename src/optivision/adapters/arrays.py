"""Bring your own vectors: the adapters that need no model.

``ArrayAdapter`` takes the ``[n_i, d]`` arrays some model has already produced, somewhere
else, and returns them as a :class:`~optivision.representation.MultiVectorCorpus` after
checking them. ``CallableAdapter`` does the same for two functions that return such arrays.
``corpus_from_arrays`` is the checked conversion both use; a custom adapter can call it
too, so that every adapter hands the core vectors that passed the same checks.

Adapters produce vectors and descriptive identity only. Nothing here chooses or alters
compression settings by model, and the identity that is passed in is recorded, never used.

What is checked, and what is not done for you:

* Every item goes through :func:`optivision.core.vectors.as_host_array`. A torch tensor on
  an accelerator is refused unless ``allow_device_transfer=True``; a dtype NumPy cannot hold
  (bfloat16) is refused; a list is refused. Nothing is moved or widened silently.
* ``MultiVectorCorpus.from_arrays`` promotes a mix of float dtypes to one common dtype
  (``numpy.result_type``): float16 and float32 items would silently become float32. These
  adapters refuse a mix instead, so convert the items explicitly before handing them over.
  A single dtype is kept as it is.
* The dimension is the same in every item and, when an identity is given, equal to its
  ``dimension``; the values are floating point and finite; the ids are unique.
* Padding is not detected. A model that pads a batch to a common length must drop the
  padded rows before the arrays are handed over, using its attention mask: a padded row
  would be scored as if it were a vector. A 3-D array (the usual shape of a padded batch)
  is refused for that reason.

The corpus owns its memory (the items are concatenated), so the caller's arrays are never
modified or shared; the peak is about twice the size of the vectors while it is built.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..core.vectors import (
    ContractError,
    EncoderIdentity,
    MultiVectorCorpus,
    as_host_array,
    validate_corpus,
)
from . import EncoderAdapter

__all__ = ["ArrayAdapter", "CallableAdapter", "corpus_from_arrays"]


def _refuse_batch(what: str, obj: Any) -> ContractError:
    """The error for a whole array or tensor where a sequence of per-item arrays belongs."""
    shape = tuple(getattr(obj, "shape", ()))
    if len(shape) == 3:
        hint = (f"a 3-D array {shape} looks like a padded batch [n_items, max_tokens, dim], whose "
                "padding rows would be scored as vectors; pass a list of [n_i, d] arrays with "
                "the padding rows removed (list(batch) is enough only if no row is padding)")
    else:
        hint = "pass a list with one [n_i, d] array per item"
    kind = type(obj).__name__
    return ContractError(f"{what}: expected a sequence of [n_i, d] arrays, got a {kind}"
                         f"{' of shape ' + str(shape) if shape else ''}; {hint}")


def _check_identity(identity: Any) -> None:
    if identity is not None and not isinstance(identity, EncoderIdentity):
        raise TypeError(
            f"identity must be an EncoderIdentity or None, got {type(identity).__name__}"
        )


def _record(identity: EncoderIdentity | None, attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Corpus ``attrs``: the caller's, plus the identity as a plain dict when there is one."""
    out = dict(attrs or {})
    if identity is not None:
        out.setdefault("model", identity.model_id)
        out["encoder_identity"] = identity.to_dict()
    return out


def corpus_from_arrays(
    arrays: Sequence[Any],
    ids: Sequence[str] | None = None,
    *,
    identity: EncoderIdentity | None = None,
    allow_device_transfer: bool = False,
    name: str = "documents",
    metadata: Sequence[dict[str, Any]] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> MultiVectorCorpus:
    """Check ``arrays`` and return them as a corpus (see the module docstring).

    Args:
        arrays: one ``[n_i, d]`` NumPy array or CPU torch tensor per item. A 3-D array, a
            string and a mapping are refused.
        ids: one id per item; the library's default (``"0"``, ``"1"``, ...) if None.
        identity: if given, ``d`` must equal ``identity.dimension``; it is recorded in
            ``attrs["encoder_identity"]`` (and ``attrs["model"]`` unless ``attrs`` has one).
        allow_device_transfer: copy a tensor that lives on an accelerator to the host.
        name: prefix of every error message, e.g. ``"queries"``.
        metadata: one dict per item (optional).
        attrs: corpus-level provenance, copied into the corpus.

    Raises:
        ContractError: an item is not a 2-D floating array of the expected dimension, dtypes
            are mixed, a value is NaN or infinite, or ids repeat. ``TypeError`` for an item
            that is not an array (a list, for example).
    """
    if isinstance(arrays, (str, bytes, Mapping)) or hasattr(arrays, "shape"):
        raise _refuse_batch(name, arrays)
    items = list(arrays)
    if not items:
        raise ContractError(f"{name}: no items to convert")
    if ids is not None:
        if isinstance(ids, (str, bytes)):
            raise TypeError(f"{name}: ids must be a sequence of strings, not one string")
        ids = list(ids)
        if len(ids) != len(items):
            raise ContractError(f"{name}: {len(ids)} ids for {len(items)} items")
    if metadata is not None and len(metadata) != len(items):
        raise ContractError(f"{name}: {len(metadata)} metadata entries for {len(items)} items")

    host: list[np.ndarray] = []
    for i, item in enumerate(items):
        array = as_host_array(item, allow_device_transfer=allow_device_transfer,
                              name=f"{name}[{i}]")
        if array.ndim != 2:
            raise ContractError(
                f"{name}[{i}] must be 2-D [n_vectors, dim], got shape {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise ContractError(f"{name}[{i}] must be floating point, got {array.dtype}")
        host.append(array)

    dtype = host[0].dtype
    dim = host[0].shape[1] if identity is None else identity.dimension
    for i, array in enumerate(host):
        if array.shape[1] != dim:
            reference = (f"{name}[0] has {dim}" if identity is None else
                         f"the identity of {identity.model_id!r} says {dim}")
            raise ContractError(f"{name}[{i}] has dimension {array.shape[1]} but {reference}")
        if array.dtype != dtype:
            raise ContractError(
                f"{name}[{i}] has dtype {array.dtype} but {name}[0] has {dtype}; mixed dtypes "
                "would be promoted to one common dtype by MultiVectorCorpus.from_arrays, so "
                "convert them explicitly (for example with .astype(np.float32)) before passing them"
            )
    if dim < 1:
        raise ContractError(f"{name}: the dimension must be at least 1, got {dim}")

    corpus = MultiVectorCorpus.from_arrays(host, ids=ids, attrs=_record(identity, attrs))
    if metadata is not None:
        corpus.metadata = [dict(m) for m in metadata]
    validate_corpus(corpus, name=name, expect_dim=dim, expect_dtype=dtype)
    return corpus


class ArrayAdapter(EncoderAdapter):
    """Vectors some model already produced, as an adapter.

    ``encode_documents`` and ``encode_queries`` take one ``[n_i, d]`` NumPy array or CPU
    torch tensor per document or query and return a corpus, after the checks of
    :func:`corpus_from_arrays` (floating, finite, one dimension, one dtype, unique ids; no
    silent device transfer, dtype change or padding). Nothing is encoded here: the model
    ran elsewhere.

    ``identity`` is the caller's description of that model. It is recorded in the corpus
    ``attrs`` and checked against the dimension of the arrays; it is never looked up in
    the model registry, so this adapter never carries a ``model_status``: that the arrays
    came from a registered model is the caller's statement, not something verified here.
    """

    def __init__(self, identity: EncoderIdentity | None = None, *,
                 allow_device_transfer: bool = False) -> None:
        _check_identity(identity)
        self._identity = identity
        self.allow_device_transfer = bool(allow_device_transfer)

    def identity(self) -> EncoderIdentity | None:
        return self._identity

    def _convert(
        self, arrays: Sequence[Any], ids: Sequence[str] | None, name: str
    ) -> MultiVectorCorpus:
        return corpus_from_arrays(
            arrays, ids, identity=self._identity, allow_device_transfer=self.allow_device_transfer,
            name=name, attrs={"loader": "arrays"},
        )

    def encode_documents(
        self, documents: Sequence[Any], ids: Sequence[str] | None = None
    ) -> MultiVectorCorpus:
        return self._convert(documents, ids, "documents")

    def encode_queries(
        self, queries: Sequence[Any], ids: Sequence[str] | None = None
    ) -> MultiVectorCorpus:
        return self._convert(queries, ids, "queries")


class CallableAdapter(EncoderAdapter):
    """Two functions that produce vectors, as an adapter.

    ``encode_documents(items)`` and ``encode_queries(items)`` are the caller's functions:
    each takes the list of inputs and returns one ``[n_i, d]`` array (or CPU tensor) per
    input, in order, with padding rows already dropped. Their results go through the same
    checks as :class:`ArrayAdapter`, and a result with a different length than the input
    is refused. The functions are called with a list even if a generator was passed.

    The adapter has no say in how the functions encode: device, precision and batching are
    theirs. ``identity`` is the caller's description of the model, recorded and checked
    against the dimension, never looked up in the registry.
    """

    def __init__(
        self,
        encode_documents: Callable[[list[Any]], Sequence[Any]],
        encode_queries: Callable[[list[Any]], Sequence[Any]],
        identity: EncoderIdentity | None = None,
        *,
        allow_device_transfer: bool = False,
    ) -> None:
        for label, fn in (("encode_documents", encode_documents),
                          ("encode_queries", encode_queries)):
            if not callable(fn):
                raise TypeError(f"{label} must be callable, got {type(fn).__name__}")
        _check_identity(identity)
        self._document_function = encode_documents
        self._query_function = encode_queries
        self._identity = identity
        self.allow_device_transfer = bool(allow_device_transfer)

    def identity(self) -> EncoderIdentity | None:
        return self._identity

    def _run(self, function: Callable[[list[Any]], Sequence[Any]], inputs: Sequence[Any],
             ids: Sequence[str] | None, name: str) -> MultiVectorCorpus:
        items = list(inputs)
        result = function(items)
        if isinstance(result, MultiVectorCorpus):
            raise ContractError(
                f"{name}: the function returned a MultiVectorCorpus; return a list of "
                "[n_i, d] arrays instead (a corpus needs no adapter)"
            )
        if hasattr(result, "shape"):
            raise _refuse_batch(f"{name} (returned by the function)", result)
        result = list(result)
        if len(result) != len(items):
            raise ContractError(f"{name}: the function returned {len(result)} arrays for "
                                f"{len(items)} inputs")
        return corpus_from_arrays(
            result, ids, identity=self._identity, allow_device_transfer=self.allow_device_transfer,
            name=name, attrs={"loader": "callable"},
        )

    def encode_documents(
        self, documents: Sequence[Any], ids: Sequence[str] | None = None
    ) -> MultiVectorCorpus:
        return self._run(self._document_function, documents, ids, "documents")

    def encode_queries(
        self, queries: Sequence[Any], ids: Sequence[str] | None = None
    ) -> MultiVectorCorpus:
        return self._run(self._query_function, queries, ids, "queries")
