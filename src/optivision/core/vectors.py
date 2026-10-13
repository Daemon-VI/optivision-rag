"""The vector contract: what a multi-vector corpus is, how to name it, how to check it.

``MultiVectorCorpus``, ``MultiVectorRepresentation`` and ``as_corpus`` are the
library's own objects from :mod:`optivision.representation`, re-exported unchanged
(same classes, no copies, no subclasses). This module adds what a caller needs
around them:

``corpus_identity``
    SHA-256 digests of the arrays that define a corpus, so a result can say which
    corpus it was computed on.
``validate_corpus``
    The checks an adapter or a loader runs before handing vectors on. It looks and
    never changes anything.
``as_host_array``
    The explicit place where a framework tensor becomes a NumPy array: no silent
    device transfer and no silent dtype change.
``EncoderIdentity``
    Descriptive metadata about the model that produced the vectors.

Nothing here knows about pages, images or a particular encoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, NoReturn

import numpy as np

from ..representation import MultiVectorCorpus, MultiVectorRepresentation, as_corpus
from . import identity

__all__ = [
    "MODALITIES",
    "ContractError",
    "CorpusIdentity",
    "EncoderIdentity",
    "MultiVectorCorpus",
    "MultiVectorRepresentation",
    "as_corpus",
    "as_host_array",
    "corpus_identity",
    "validate_corpus",
]

#: Arrays aligned with the vectors, in the order they are hashed and checked.
_ALIGNED = ("protected", "positions", "weights", "importance")

#: Row-block size of the finiteness scan: a block is looked at in its own dtype, so a
#: float16 corpus is never widened and nothing as large as the corpus is allocated.
_BLOCK_BYTES = 8 * 1024 * 1024

MODALITIES = ("text", "image", "other")


class ContractError(ValueError):
    """An input broke the vector contract. The message names the first offender."""


# ------------------------------------------------------------------ identity


@dataclass(frozen=True)
class CorpusIdentity:
    """Content digests of a corpus.

    Hashed, each array in the dtype it is stored in (a float16 and a float32 copy of
    the same numbers are different corpora): ``vectors``, ``offsets``, ``ids``, and
    each aligned array that is present (``protected``, ``positions``, ``weights``,
    ``importance``). The aligned arrays are part of the identity because they change
    what token reducers do. An absent array and an all-default one are different
    identities, because stages test for presence.

    Not hashed: ``images``, ``metadata`` and ``attrs``. Two corpora that differ only
    there have the same identity.

    The digests are functions of the stored bytes (dtype, shape and values as stored,
    see ``identity.array_digest``), so anyone can recompute them from the arrays alone.
    Values that compare equal but are stored differently have different digests:
    ``0.0`` and ``-0.0``, and NaNs with different bit patterns. ``positions`` marks
    off-grid rows with NaN; the library writes it as ``np.nan`` cast to float32 and
    ``save``/``load`` keep the bits, so the digest is stable across both.

    An identity is an integrity check, not a signature: anyone can recompute it.
    """

    n_docs: int
    n_vectors: int
    dimension: int
    dtype: str
    vectors_sha256: str
    offsets_sha256: str
    ids_sha256: str
    protected_sha256: str | None = None
    positions_sha256: str | None = None
    weights_sha256: str | None = None
    importance_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-native fields; an absent aligned array is ``None``."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def digest(self) -> str:
        """One SHA-256 over :meth:`to_dict`, for binding a result to a corpus."""
        return identity.digest(self.to_dict())


def corpus_identity(corpus: MultiVectorCorpus) -> CorpusIdentity:
    """Digests of ``corpus``; the vectors are hashed in blocks, in their own dtype."""
    if not isinstance(corpus, MultiVectorCorpus):
        raise TypeError(
            f"corpus_identity needs a MultiVectorCorpus, got {type(corpus).__name__}; "
            "build one explicitly with as_corpus()"
        )
    aligned = {
        f"{key}_sha256": None if array is None else identity.array_digest(array)
        for key, array in ((key, getattr(corpus, key)) for key in _ALIGNED)
    }
    return CorpusIdentity(
        n_docs=len(corpus),
        n_vectors=corpus.num_vectors,
        dimension=corpus.dimension,
        dtype=str(corpus.vectors.dtype),
        vectors_sha256=identity.array_digest(corpus.vectors),
        offsets_sha256=identity.array_digest(corpus.offsets),
        ids_sha256=identity.strings_digest(corpus.ids),
        **aligned,
    )


# ---------------------------------------------------------------- validation


def _first_bad(a: np.ndarray, *, nan_ok: bool) -> tuple[tuple[int, ...], float] | None:
    """Index and value of the first NaN or infinity (only infinity if ``nan_ok``).

    ``a`` is scanned in blocks of rows, each in its own dtype.
    """
    row_bytes = max(1, a.itemsize * int(np.prod(a.shape[1:], dtype=np.int64)))
    step = max(1, _BLOCK_BYTES // row_bytes)
    for lo in range(0, a.shape[0], step):
        block = a[lo : lo + step]
        bad = np.isinf(block) if nan_ok else ~np.isfinite(block)
        if bad.any():
            index = np.unravel_index(int(np.argmax(bad)), bad.shape)
            return (lo + int(index[0]), *(int(i) for i in index[1:])), float(block[index])
    return None


def validate_corpus(
    corpus: MultiVectorCorpus,
    *,
    name: str = "corpus",
    expect_dim: int | None = None,
    expect_dtype: Any = None,
    require_finite: bool = True,
    allow_empty_docs: bool = True,
    require_unique_ids: bool = True,
) -> None:
    """Raise :class:`ContractError` if ``corpus`` breaks the vector contract.

    The constructor of ``MultiVectorCorpus`` already checks shapes, but its attributes
    can be reassigned and it never looks at the numbers; this function checks the
    object as it is now. It reads and never modifies or converts anything. The first
    violation found is reported, always in this order: the object itself; ``vectors``
    (2-D, floating, ``expect_dim``, ``expect_dtype``); ``offsets``; ``ids`` (count,
    type, uniqueness); ``metadata`` and ``images`` counts; the aligned arrays (shape,
    dtype); empty documents; finiteness. The message names the document or vector row.

    Args:
        name: prefix of every message, e.g. ``"queries"``.
        expect_dim, expect_dtype: if given, the vectors must have exactly this
            dimension and dtype.
        require_finite: no NaN or infinity in ``vectors``, ``weights`` or
            ``importance``, and no infinity in ``positions`` (NaN there is the
            off-grid marker). Scanned block by block in the stored dtype.
        allow_empty_docs: set False to refuse documents with no vectors.
        require_unique_ids: set False for stores whose ids repeat.
    """

    def fail(message: str) -> NoReturn:
        raise ContractError(f"{name}: {message}")

    if not isinstance(corpus, MultiVectorCorpus):
        fail(f"expected a MultiVectorCorpus, got {type(corpus).__name__}")
    vectors, offsets, ids = corpus.vectors, corpus.offsets, corpus.ids

    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        shape = getattr(vectors, "shape", type(vectors).__name__)
        fail(f"vectors must be a 2-D array [n_vectors, dim], got {shape}")
    if not np.issubdtype(vectors.dtype, np.floating):
        fail(f"vectors must be floating point, got {vectors.dtype}")
    if expect_dim is not None and vectors.shape[1] != expect_dim:
        fail(f"expected dimension {expect_dim}, got {vectors.shape[1]}")
    if expect_dtype is not None and vectors.dtype != np.dtype(expect_dtype):
        fail(f"expected dtype {np.dtype(expect_dtype)}, got {vectors.dtype}")
    n_vectors = vectors.shape[0]

    if not (isinstance(offsets, np.ndarray) and offsets.ndim == 1 and offsets.size >= 1
            and offsets.dtype == np.int64):
        detail = (f"{offsets.dtype} {offsets.shape}" if isinstance(offsets, np.ndarray)
                  else type(offsets).__name__)
        fail(f"offsets must be a 1-D int64 array with n_docs + 1 entries, got {detail}")
    if int(offsets[0]) != 0:
        fail(f"offsets must start at 0, got {int(offsets[0])}")
    steps = np.diff(offsets)
    if (steps < 0).any():
        i = int(np.argmax(steps < 0))
        fail(f"offsets decrease at document {i}: offsets[{i}]={int(offsets[i])} "
             f"> offsets[{i + 1}]={int(offsets[i + 1])}")
    if int(offsets[-1]) != n_vectors:
        fail(f"offsets end at {int(offsets[-1])} but there are {n_vectors} vectors")
    n_docs = int(offsets.size) - 1

    if len(ids) != n_docs:
        fail(f"{len(ids)} ids for {n_docs} documents")
    first_seen: dict[str, int] = {}
    for i, doc_id in enumerate(ids):
        if not isinstance(doc_id, str):
            fail(f"the id of document {i} is {type(doc_id).__name__}, not str")
        if require_unique_ids and first_seen.setdefault(doc_id, i) != i:
            fail(f"id {doc_id!r} is used by documents {first_seen[doc_id]} and {i}")
    if len(corpus.metadata) != n_docs:
        fail(f"{len(corpus.metadata)} metadata entries for {n_docs} documents")
    if corpus.images is not None and len(corpus.images) != n_docs:
        fail(f"{len(corpus.images)} images for {n_docs} documents")

    for key, shape, kind in (
        ("protected", (n_vectors,), "b"),
        ("positions", (n_vectors, 2), "f"),
        ("weights", (n_vectors,), "f"),
        ("importance", (n_vectors,), "f"),
    ):
        array = getattr(corpus, key)
        if array is None:
            continue
        if not isinstance(array, np.ndarray) or array.shape != shape:
            got = getattr(array, "shape", type(array).__name__)
            fail(f"{key} must have shape {shape}, got {got}")
        if array.dtype.kind != kind:
            fail(f"{key} must be {'bool' if kind == 'b' else 'floating point'}, got {array.dtype}")

    if not allow_empty_docs and (steps == 0).any():
        i = int(np.argmax(steps == 0))
        fail(f"document {i} (id {ids[i]!r}) has no vectors")

    if not require_finite:
        return

    def locate(row: int) -> str:
        doc = int(np.searchsorted(offsets, row, side="right")) - 1
        return f"document {doc}, id {ids[doc]!r}"

    bad = _first_bad(vectors, nan_ok=False)
    if bad is not None:
        (row, col), value = bad
        fail(f"vector {row} ({locate(row)}) has a non-finite value, {value}, at column {col}")
    for key, nan_ok in (("weights", False), ("importance", False), ("positions", True)):
        array = getattr(corpus, key)
        bad = None if array is None else _first_bad(array, nan_ok=nan_ok)
        if bad is not None:
            row = bad[0][0]
            what = "an infinite" if nan_ok else "a non-finite"
            fail(f"{key}[{row}] ({locate(row)}) is {what} value, {bad[1]}")


# ------------------------------------------------------------ host conversion


def as_host_array(
    x: Any,
    *,
    allow_device_transfer: bool = False,
    allow_dtype_widening: bool = False,
    name: str = "vectors",
) -> np.ndarray:
    """Return ``x`` as a NumPy array without any silent device transfer or dtype change.

    * A NumPy array is returned as it is: the same object, no copy.
    * A torch tensor (recognised by its module name; torch is never imported here) on
      the CPU is viewed through ``detach().numpy()`` and shares memory with the
      tensor. On any other device it raises :class:`ContractError` unless
      ``allow_device_transfer=True``, which copies it to the host.
    * A torch dtype that NumPy cannot represent (bfloat16, float8) raises
      :class:`ContractError` unless ``allow_dtype_widening=True``, which converts to
      float32 and so copies. Every other dtype is kept, float16 included.
    * Any other object exposing ``__array__`` goes through ``np.asarray``. Only torch
      tensors are checked for device placement; a framework whose ``__array__`` copies
      from an accelerator does so here without a flag.
    * Anything else, lists included, raises ``TypeError``: convert it explicitly with
      ``np.asarray(x, dtype=...)`` so the dtype is your decision.

    The ``representation.as_float_matrix`` that the corpus constructors use moves tensors
    to the CPU and widens bfloat16 without being asked; adapters should come through
    this function first. Shape and dtype kind are not checked here; ``validate_corpus``
    does that.
    """
    if isinstance(x, np.ndarray):
        return x
    if type(x).__module__.split(".")[0] == "torch" and hasattr(x, "detach"):
        return _tensor_to_host(x, allow_device_transfer, allow_dtype_widening, name)
    if hasattr(x, "__array__"):
        return np.asarray(x)
    raise TypeError(
        f"{name}: cannot take a {type(x).__name__} as a host array; pass a NumPy array, a torch "
        "tensor or an object with __array__ (convert lists with np.asarray(x, dtype=...))"
    )


def _tensor_to_host(
    t: Any, allow_device_transfer: bool, allow_dtype_widening: bool, name: str
) -> np.ndarray:
    device = getattr(t, "device", None)
    if getattr(device, "type", "cpu") != "cpu":
        if not allow_device_transfer:
            raise ContractError(
                f"{name} is a torch tensor on device '{device}'; pass allow_device_transfer=True "
                "to copy it to the host"
            )
        t = t.detach().cpu()
    else:
        t = t.detach()
    try:
        return t.numpy()
    except TypeError as exc:  # torch refuses dtypes NumPy lacks: "unsupported ScalarType BFloat16"
        if not allow_dtype_widening:
            raise ContractError(
                f"{name} has torch dtype {t.dtype}, which NumPy cannot represent; pass "
                "allow_dtype_widening=True to convert it to float32"
            ) from exc
        return t.float().numpy()


# ---------------------------------------------------------- encoder identity


@dataclass(frozen=True, kw_only=True)
class EncoderIdentity:
    """Which model produced the vectors, as a record for certificates and adapters.

    Descriptive metadata only: no core function branches on it, so naming a model
    here never changes what a stage, a scorer or a selection rule does. It is
    keyword-only so that fields cannot be passed in the wrong order.

    ``modality`` is one of ``MODALITIES``. ``normalized`` says whether the output rows
    are unit-norm and ``inference_dtype`` which dtype the model ran in (``None``:
    not stated). ``extra`` holds free-form string pairs such as a pooling mode or a
    prompt prefix.
    """

    model_id: str
    revision: str | None = None
    dimension: int
    modality: str
    normalized: bool | None = None
    inference_dtype: str | None = None
    extra: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ContractError(f"model_id must be a non-empty string, got {self.model_id!r}")
        if (isinstance(self.dimension, bool) or not isinstance(self.dimension, (int, np.integer))
                or self.dimension <= 0):
            raise ContractError(f"dimension must be a positive integer, got {self.dimension!r}")
        if self.modality not in MODALITIES:
            raise ContractError(f"modality must be one of {MODALITIES}, got {self.modality!r}")
        for key in ("revision", "inference_dtype"):
            value = getattr(self, key)
            if value is not None and not isinstance(value, str):
                raise ContractError(f"{key} must be a string or None, got {value!r}")
        if self.normalized is not None and not isinstance(self.normalized, (bool, np.bool_)):
            raise ContractError(f"normalized must be a bool or None, got {self.normalized!r}")
        if not isinstance(self.extra, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.extra.items()
        ):
            raise ContractError("extra must map strings to strings")
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(self, "extra", dict(self.extra))
        if self.normalized is not None:
            object.__setattr__(self, "normalized", bool(self.normalized))

    def to_dict(self) -> dict[str, Any]:
        """JSON-native fields; ``extra`` is sorted by key."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "modality": self.modality,
            "normalized": self.normalized,
            "inference_dtype": self.inference_dtype,
            "extra": {k: self.extra[k] for k in sorted(self.extra)},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EncoderIdentity:
        """Inverse of :meth:`to_dict`; unknown or missing fields raise ``ContractError``."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ContractError(f"unknown EncoderIdentity field(s): {unknown}")
        missing = sorted({"model_id", "dimension", "modality"} - set(data))
        if missing:
            raise ContractError(f"EncoderIdentity needs {missing}")
        return cls(**dict(data))
