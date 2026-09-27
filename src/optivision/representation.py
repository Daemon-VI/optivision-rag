"""A model-agnostic multi-vector data model.

Everything in the universal layer -- stages, pipelines, calibration, scoring --
speaks this format and nothing else, and it needs only vectors. Structure that
some models have and others do not rides along as optional arrays aligned with
the vectors:

    protected   bool [n]     rows token reduction must keep verbatim (instruction
                             tokens, a ColSmol thumbnail run, a [CLS] token)
    positions   float32 [n, 2]  (row, col) of a patch centre in a 0..1 page frame;
                             NaN for rows that are not image patches
    weights     float32 [n]  how many original vectors a row stands for (1 unless
                             a merge produced it)
    importance  float32 [n]  an external saliency signal, e.g. attention, when the
                             encoder exposes one

Stages that need one of these (the pixel-space pruner needs positions and page
images) check for it and say so, instead of assuming a ColPali-shaped input.

Two containers:

``MultiVectorRepresentation``
    One document or one query.

``MultiVectorCorpus``
    Many of them, stored as one contiguous ``[N, d]`` matrix plus ``offsets``
    (document ``i`` owns rows ``offsets[i]:offsets[i+1]``). A corpus of a million
    pages is one allocation, not a million small ones, and indexing a document
    returns views rather than copies.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "optivision-multivector"
FORMAT_VERSION = 1

_ALIGNED = ("protected", "positions", "weights", "importance")


# ------------------------------------------------------------------ helpers


def as_float_matrix(x: Any, name: str = "vectors") -> np.ndarray:
    """Coerce numpy arrays, nested lists or torch tensors to a 2-D float array.

    Floating dtypes are kept as they are (float16 stays float16, so a caller's
    memory budget is respected); bfloat16 tensors, which numpy cannot represent,
    are widened to float32. Integer input is rejected rather than silently cast:
    int8 codes passed where float vectors belong is a bug worth hearing about.
    """
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # a torch tensor, without importing torch
        t = x.detach().cpu()
        if str(getattr(t, "dtype", "")) == "torch.bfloat16":
            t = t.float()
        x = t.numpy()
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [n_vectors, dim], got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"{name} must be floating point, got {arr.dtype}")
    return arr


def _check_aligned(name: str, arr: np.ndarray | None, n: int, width: int | None = None) -> None:
    if arr is None:
        return
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows but there are {n} vectors")
    if width is not None and (arr.ndim != 2 or arr.shape[1] != width):
        raise ValueError(f"{name} must be [{n}, {width}], got {arr.shape}")
    if width is None and arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D [{n}], got {arr.shape}")


def _aligned_arrays(
    n: int,
    protected: Any = None,
    positions: Any = None,
    weights: Any = None,
    importance: Any = None,
) -> dict[str, np.ndarray | None]:
    out = {
        "protected": None if protected is None else np.asarray(protected, dtype=bool),
        "positions": None if positions is None else np.asarray(positions, dtype=np.float32),
        "weights": None if weights is None else np.asarray(weights, dtype=np.float32),
        "importance": None if importance is None else np.asarray(importance, dtype=np.float32),
    }
    _check_aligned("protected", out["protected"], n)
    _check_aligned("positions", out["positions"], n, width=2)
    _check_aligned("weights", out["weights"], n)
    _check_aligned("importance", out["importance"], n)
    return out


# ------------------------------------------------------------ one document


@dataclass
class MultiVectorRepresentation:
    """One document (or one query) as a set of vectors.

    ``vectors`` is ``[num_vectors, dimension]`` in any floating dtype. Nothing
    here requires L2-normalised rows -- MaxSim is an inner product and some
    models do not normalise -- but :meth:`norm_stats` reports it, because
    several codecs are only well-behaved on unit vectors.
    """

    vectors: np.ndarray
    doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    protected: np.ndarray | None = None
    positions: np.ndarray | None = None
    weights: np.ndarray | None = None
    importance: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vectors = as_float_matrix(self.vectors)
        aligned = _aligned_arrays(
            self.vectors.shape[0], self.protected, self.positions, self.weights, self.importance
        )
        for name, value in aligned.items():
            setattr(self, name, value)

    @property
    def num_vectors(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def dtype(self) -> np.dtype:
        return self.vectors.dtype

    @property
    def nbytes(self) -> int:
        return int(self.vectors.nbytes)

    def norm_stats(self) -> dict[str, float]:
        return _norm_stats(self.vectors)


def _norm_stats(vectors: np.ndarray, sample: int = 100_000) -> dict[str, float]:
    if vectors.shape[0] == 0:
        return {"norm_mean": 0.0, "norm_min": 0.0, "norm_max": 0.0, "unit_fraction": 0.0}
    v = vectors
    if v.shape[0] > sample:  # a strided sample is plenty for a summary
        v = v[:: max(1, v.shape[0] // sample)]
    norms = np.linalg.norm(v.astype(np.float32, copy=False), axis=1)
    return {
        "norm_mean": float(norms.mean()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "unit_fraction": float(np.mean(np.abs(norms - 1.0) < 1e-3)),
    }


# ------------------------------------------------------------------ corpus


class MultiVectorCorpus:
    """Many multi-vector documents as one contiguous matrix plus offsets.

    Args:
        vectors: ``[N, d]`` floating array, all documents concatenated.
        offsets: ``int64 [n_docs + 1]``; document ``i`` owns rows
            ``offsets[i]:offsets[i+1]``. Empty documents are allowed.
        ids: one string per document (defaults to ``"0"``, ``"1"``, ...).
        metadata: one dict per document (optional).
        protected, positions, weights, importance: optional arrays aligned
            with ``vectors``, see the module docstring.
        images: optional per-document page images, used only by stages that
            read pixels. Never serialised.
        attrs: corpus-level provenance, e.g. ``{"model": "vidore/colpali-v1.3"}``.
    """

    def __init__(
        self,
        vectors: Any,
        offsets: Any,
        ids: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        protected: Any = None,
        positions: Any = None,
        weights: Any = None,
        importance: Any = None,
        images: Sequence[Any] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.vectors = as_float_matrix(vectors)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        n_docs = int(self.offsets.size) - 1
        if n_docs < 0 or self.offsets[0] != 0:
            raise ValueError("offsets must start at 0 and have n_docs + 1 entries")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("offsets must be non-decreasing")
        if int(self.offsets[-1]) != self.vectors.shape[0]:
            raise ValueError(
                f"offsets end at {int(self.offsets[-1])} but there are {self.vectors.shape[0]} vectors"
            )
        self.ids = [str(i) for i in ids] if ids is not None else [str(i) for i in range(n_docs)]
        if len(self.ids) != n_docs:
            raise ValueError(f"{len(self.ids)} ids for {n_docs} documents")
        self.metadata = [dict(m) for m in metadata] if metadata is not None else [{} for _ in range(n_docs)]
        if len(self.metadata) != n_docs:
            raise ValueError(f"{len(self.metadata)} metadata entries for {n_docs} documents")
        aligned = _aligned_arrays(self.vectors.shape[0], protected, positions, weights, importance)
        self.protected = aligned["protected"]
        self.positions = aligned["positions"]
        self.weights = aligned["weights"]
        self.importance = aligned["importance"]
        if images is not None and len(images) != n_docs:
            raise ValueError(f"{len(images)} images for {n_docs} documents")
        self.images = list(images) if images is not None else None
        self.attrs = dict(attrs or {})

    # ------------------------------------------------------------ shape

    def __len__(self) -> int:
        return int(self.offsets.size) - 1

    @property
    def n_docs(self) -> int:
        return len(self)

    @property
    def num_vectors(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def dtype(self) -> np.dtype:
        return self.vectors.dtype

    @property
    def nbytes(self) -> int:
        """Bytes held by the vectors themselves (the thing compression shrinks)."""
        return int(self.vectors.nbytes)

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    def span(self, i: int) -> tuple[int, int]:
        return int(self.offsets[i]), int(self.offsets[i + 1])

    # ---------------------------------------------------------- access

    def __getitem__(self, i: int) -> MultiVectorRepresentation:
        if isinstance(i, slice):
            raise TypeError("use corpus.select(range(...)) for a sub-corpus")
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        lo, hi = self.span(i)
        return MultiVectorRepresentation(
            vectors=self.vectors[lo:hi],
            doc_id=self.ids[i],
            metadata=self.metadata[i],
            protected=None if self.protected is None else self.protected[lo:hi],
            positions=None if self.positions is None else self.positions[lo:hi],
            weights=None if self.weights is None else self.weights[lo:hi],
            importance=None if self.importance is None else self.importance[lo:hi],
        )

    def __iter__(self) -> Iterator[MultiVectorRepresentation]:
        for i in range(len(self)):
            yield self[i]

    def decode_rows(self, lo: int, hi: int) -> np.ndarray:
        """float32 rows ``lo:hi`` -- the interface every scorer reads through.

        A compressed corpus implements the same method by decoding its codes, so
        exact MaxSim code never needs to know which one it is scoring.
        """
        return np.asarray(self.vectors[lo:hi], dtype=np.float32)

    # ------------------------------------------------------ construction

    @classmethod
    def from_documents(
        cls,
        docs: Sequence[MultiVectorRepresentation | Any],
        ids: Sequence[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> MultiVectorCorpus:
        """Concatenate documents. Plain arrays are accepted as documents too."""
        reps = [d if isinstance(d, MultiVectorRepresentation) else MultiVectorRepresentation(d) for d in docs]
        if not reps:
            raise ValueError("no documents")
        dims = {r.dimension for r in reps}
        if len(dims) != 1:
            raise ValueError(f"documents disagree on dimension: {sorted(dims)}")
        dtype = np.result_type(*[r.dtype for r in reps])
        counts = np.array([r.num_vectors for r in reps], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        vectors = np.concatenate([np.asarray(r.vectors, dtype=dtype) for r in reps], axis=0)

        def gather(name: str, default: Any) -> np.ndarray | None:
            parts = [getattr(r, name) for r in reps]
            if all(p is None for p in parts):
                return None
            filled = [p if p is not None else default(r.num_vectors) for p, r in zip(parts, reps, strict=True)]
            return np.concatenate(filled, axis=0)

        doc_ids = list(ids) if ids is not None else [
            r.doc_id if r.doc_id is not None else str(i) for i, r in enumerate(reps)
        ]
        return cls(
            vectors,
            offsets,
            ids=doc_ids,
            metadata=[r.metadata for r in reps],
            protected=gather("protected", lambda n: np.zeros(n, dtype=bool)),
            positions=gather("positions", lambda n: np.full((n, 2), np.nan, dtype=np.float32)),
            weights=gather("weights", lambda n: np.ones(n, dtype=np.float32)),
            importance=gather("importance", lambda n: np.zeros(n, dtype=np.float32)),
            attrs=attrs,
        )

    @classmethod
    def from_arrays(
        cls,
        arrays: Sequence[Any],
        ids: Sequence[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> MultiVectorCorpus:
        """The precomputed-vectors path: a list of ``[n_i, d]`` arrays or tensors."""
        return cls.from_documents([MultiVectorRepresentation(a) for a in arrays], ids=ids, attrs=attrs)

    @classmethod
    def from_page_encodings(
        cls,
        encodings: Sequence[Any],
        images: Sequence[Any] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> MultiVectorCorpus:
        """Adapter from the original ColPali-shaped :class:`~optivision.types.PageEncoding`.

        Instruction and thumbnail tokens become ``protected``; patch tokens get
        grid-centre ``positions``; the grid shape goes into metadata so the
        pixel-space pruner can rebuild it. Row order is preserved exactly, so a
        document's vectors here are the same rows as ``enc.embeddings``.
        """
        docs = []
        for enc in encodings:
            n = enc.n_tokens
            protected = np.zeros(n, dtype=bool)
            protected[np.asarray(enc.text_token_index, dtype=np.int64)] = True
            positions = np.full((n, 2), np.nan, dtype=np.float32)
            rows, cols = enc.grid.rows, enc.grid.cols
            rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
            idx = np.asarray(enc.grid.token_index, dtype=np.int64).reshape(-1)
            positions[idx, 0] = (rr.reshape(-1) + 0.5) / rows
            positions[idx, 1] = (cc.reshape(-1) + 0.5) / cols
            meta = dict(enc.meta)
            meta.update(
                {
                    "grid_rows": int(rows),
                    "grid_cols": int(cols),
                    "image_size": [int(s) for s in enc.image_size],
                    "page_ref": dict(enc.ref.__dict__),
                }
            )
            docs.append(
                MultiVectorRepresentation(
                    enc.embeddings,
                    doc_id=enc.ref.page_id,
                    metadata=meta,
                    protected=protected,
                    positions=positions,
                )
            )
        corpus = cls.from_documents(docs, attrs=attrs)
        if images is not None:
            if len(images) != len(corpus):
                raise ValueError(f"{len(images)} images for {len(corpus)} pages")
            corpus.images = list(images)
        return corpus

    @classmethod
    def from_legacy_cache(cls, path: str | Path, with_images: bool = True) -> MultiVectorCorpus:
        """Load an encode cache written by ``optivision bench --cache``.

        That format stores its metadata as a pickled object array, so only load
        caches you produced yourself.
        """
        from .bench import EncodedCorpus  # local import: bench pulls in the encoder registry

        enc = EncodedCorpus.load(path)
        encoders = {e.meta.get("encoder") for e in enc.encodings if e.meta.get("encoder")}
        attrs = {"source": str(path), "format": "legacy-encode-cache"}
        if len(encoders) == 1:
            attrs["encoder"] = next(iter(encoders))
        checkpoints = {e.meta.get("checkpoint") for e in enc.encodings if e.meta.get("checkpoint")}
        if len(checkpoints) == 1:
            attrs["model"] = next(iter(checkpoints))
        return cls.from_page_encodings(enc.encodings, images=enc.images if with_images else None, attrs=attrs)

    # ---------------------------------------------------------- subsets

    def select(self, indices: Sequence[int] | np.ndarray) -> MultiVectorCorpus:
        """A new corpus holding the given documents, in the given order."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("document index out of range")
        spans = [np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]
        rows = np.concatenate(spans) if spans else np.zeros(0, dtype=np.int64)
        counts = self.counts[idx] if idx.size else np.zeros(0, dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)])

        def take(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else a[rows]

        return MultiVectorCorpus(
            self.vectors[rows],
            offsets,
            ids=[self.ids[i] for i in idx],
            metadata=[self.metadata[i] for i in idx],
            protected=take(self.protected),
            positions=take(self.positions),
            weights=take(self.weights),
            importance=take(self.importance),
            images=None if self.images is None else [self.images[i] for i in idx],
            attrs=self.attrs,
        )

    def with_vectors(self, vectors: Any, **changes: Any) -> MultiVectorCorpus:
        """Same documents and rows, new vectors (e.g. after a projection).

        The row count must not change; token reduction builds a new corpus.
        """
        vectors = as_float_matrix(vectors)
        if vectors.shape[0] != self.num_vectors:
            raise ValueError("with_vectors keeps every row; use a token-reduction stage to drop rows")
        kwargs: dict[str, Any] = {
            "ids": self.ids,
            "metadata": self.metadata,
            "protected": self.protected,
            "positions": self.positions,
            "weights": self.weights,
            "importance": self.importance,
            "images": self.images,
            "attrs": self.attrs,
        }
        kwargs.update(changes)
        return MultiVectorCorpus(vectors, self.offsets, **kwargs)

    def astype(self, dtype: Any) -> MultiVectorCorpus:
        dtype = np.dtype(dtype)
        if self.vectors.dtype == dtype:
            return self
        return self.with_vectors(self.vectors.astype(dtype))

    # ----------------------------------------------------------- summary

    def summary(self) -> dict[str, Any]:
        counts = self.counts
        out: dict[str, Any] = {
            "n_docs": len(self),
            "num_vectors": self.num_vectors,
            "dimension": self.dimension,
            "dtype": str(self.dtype),
            "vectors_per_doc_mean": float(counts.mean()) if counts.size else 0.0,
            "vectors_per_doc_min": int(counts.min()) if counts.size else 0,
            "vectors_per_doc_max": int(counts.max()) if counts.size else 0,
            "empty_docs": int(np.sum(counts == 0)),
            "bytes": self.nbytes,
            "bytes_per_doc": self.nbytes / max(1, len(self)),
            "float32_bytes_per_doc": float(counts.mean()) * self.dimension * 4 if counts.size else 0.0,
            "protected_vectors": int(self.protected.sum()) if self.protected is not None else 0,
            "has_positions": self.positions is not None,
            "has_images": self.images is not None,
            "has_importance": self.importance is not None,
        }
        out.update(_norm_stats(self.vectors))
        out.update({f"attr_{k}": v for k, v in self.attrs.items() if isinstance(v, (str, int, float))})
        return out

    # ------------------------------------------------------ serialisation

    def save(self, path: str | Path, compressed: bool = False) -> Path:
        """Write a pickle-free ``.npz`` (images are not stored)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "ids": self.ids,
            "metadata": self.metadata,
            "attrs": self.attrs,
        }
        arrays: dict[str, np.ndarray] = {
            "vectors": self.vectors,
            "offsets": self.offsets,
            "header": np.frombuffer(json.dumps(header, default=_json_default).encode("utf-8"), dtype=np.uint8),
        }
        for name in _ALIGNED:
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        (np.savez_compressed if compressed else np.savez)(path, **arrays)
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path, mmap: bool = False) -> MultiVectorCorpus:
        """Read a file written by :meth:`save`. Never unpickles anything."""
        data = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
        if "header" not in data.files:
            raise ValueError(f"{path} is not an {FORMAT} file (no header); legacy encode caches "
                             "load with MultiVectorCorpus.from_legacy_cache")
        header = json.loads(bytes(data["header"]).decode("utf-8"))
        if header.get("format") != FORMAT:
            raise ValueError(f"{path}: unexpected format {header.get('format')!r}")
        if int(header.get("version", 0)) > FORMAT_VERSION:
            raise ValueError(f"{path}: format version {header['version']} is newer than this package")
        kwargs = {name: data[name] for name in _ALIGNED if name in data.files}
        return cls(
            data["vectors"],
            data["offsets"],
            ids=header["ids"],
            metadata=header["metadata"],
            attrs=header.get("attrs", {}),
            **kwargs,
        )


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def as_corpus(x: Any) -> MultiVectorCorpus:
    """Accept a corpus, a list of documents/arrays, or one representation."""
    if isinstance(x, MultiVectorCorpus):
        return x
    if isinstance(x, MultiVectorRepresentation):
        return MultiVectorCorpus.from_documents([x])
    if isinstance(x, (list, tuple)):
        return MultiVectorCorpus.from_documents(list(x))
    raise TypeError(
        f"expected a MultiVectorCorpus or a list of [n_i, d] arrays, got {type(x).__name__}"
    )
