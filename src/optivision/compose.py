"""Composable compression pipelines.

    pipeline = Pipeline([AdaptiveMerge(...), PCAProjector(96), Int8Quantizer()])
    compressed = pipeline.fit(corpus).compress(corpus)
    scores = maxsim_matrix(pipeline.transform_queries(queries), compressed)

A pipeline is an ordered list of stages. Token reducers and dimension reducers
may appear in any order; at most one quantizer, and it comes last (without one
the result is stored as float32). Queries only ever pass through the dimension
reducers, because a projection changes the space both sides must share while
pruning, merging and quantization are document-side choices.

Byte accounting is against the float32 original (``n_vectors_before * dim_before
* 4``), which is the convention the rest of the repository reports. The
``native_bytes_before`` field keeps the input's own dtype, so a float16 corpus
can be reported honestly as "2x smaller than it looked".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .representation import MultiVectorCorpus
from .stages.base import DimensionReducer, Quantizer, Stage, TokenReducer, stage_from_dict
from .stages.quantize import Float32Quantizer

#: rows per quantizer call in Pipeline.encode (bounds the codec's float temporaries)
ENCODE_BLOCK_ROWS = 65_536


def _require_finite(corpus: MultiVectorCorpus, what: str) -> None:
    """Refuse NaN / inf vectors. A single one would otherwise leak into fitted
    state (corpus means, ranges, PCA) and corrupt every other document's codes."""
    v = corpus.vectors
    for lo in range(0, corpus.num_vectors, ENCODE_BLOCK_ROWS):
        block = np.asarray(v[lo : lo + ENCODE_BLOCK_ROWS])
        if not np.isfinite(block).all():
            row = lo + int(np.argmin(np.isfinite(block).all(axis=1)))
            doc = int(np.searchsorted(corpus.offsets, row, side="right")) - 1
            raise ValueError(f"{what} contain non-finite values (first at vector {row}, document {doc})")


class CompressedCorpus:
    """Quantized documents plus the accounting needed to judge them.

    Implements the ``offsets`` / ``decode_rows`` store interface, so
    :func:`optivision.scoring.maxsim_matrix` scores it exactly like a float corpus.
    """

    def __init__(
        self,
        codes: np.ndarray,
        offsets: np.ndarray,
        dim: int,
        quantizer: Quantizer,
        ids: Sequence[str],
        stats: dict[str, Any] | None = None,
        pipeline: dict[str, Any] | None = None,
    ) -> None:
        self.codes = np.ascontiguousarray(codes, dtype=np.uint8)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.dim = int(dim)
        self.quantizer = quantizer
        self.ids = list(ids)
        self.stats = dict(stats or {})
        self.pipeline = pipeline
        if self.codes.shape[0] != int(self.offsets[-1]):
            raise ValueError("codes and offsets disagree on the number of vectors")

    def __len__(self) -> int:
        return int(self.offsets.size) - 1

    @property
    def num_vectors(self) -> int:
        return int(self.codes.shape[0])

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    def decode_rows(self, lo: int, hi: int) -> np.ndarray:
        return self.quantizer.decode(self.codes[lo:hi], self.dim)

    def to_corpus(self) -> MultiVectorCorpus:
        """Decoded float32 vectors (for inspection; this is the size you avoided)."""
        return MultiVectorCorpus(self.decode_rows(0, self.num_vectors), self.offsets, ids=self.ids)

    @property
    def nbytes(self) -> int:
        """Stored bytes: the codes plus whatever shared state the codec keeps."""
        return int(self.codes.nbytes) + int(self.quantizer.overhead_bytes())

    def report(self) -> dict[str, Any]:
        n_docs = max(1, len(self))
        before = int(self.stats.get("float32_bytes_before", 0))
        out = {
            "n_docs": len(self),
            "vectors": self.num_vectors,
            "vectors_per_doc": self.num_vectors / n_docs,
            "dim": self.dim,
            "bits_per_dim": self.quantizer.bits_per_dim(self.dim),
            "bytes": self.nbytes,
            "bytes_per_doc": self.nbytes / n_docs,
            "compression_vs_float32": before / self.nbytes if self.nbytes else 0.0,
            "vector_reduction": (
                self.stats.get("vectors_before", 0) / self.num_vectors if self.num_vectors else 0.0
            ),
        }
        native = self.stats.get("native_bytes_before")
        if native:
            out["compression_vs_native"] = native / self.nbytes if self.nbytes else 0.0
        out.update({k: v for k, v in self.stats.items() if k not in out})
        return out


class Pipeline:
    """An ordered list of stages. See the module docstring."""

    def __init__(self, stages: Sequence[Stage] | None = None) -> None:
        stages = list(stages or [])
        quantizers = [i for i, s in enumerate(stages) if isinstance(s, Quantizer)]
        if len(quantizers) > 1:
            raise ValueError("a pipeline holds at most one quantizer")
        if quantizers and quantizers[0] != len(stages) - 1:
            raise ValueError("the quantizer must be the last stage")
        for s in stages:
            if not isinstance(s, (TokenReducer, DimensionReducer, Quantizer)):
                raise TypeError(f"{s!r} is not a stage")
        self.stages = stages

    # ------------------------------------------------------------ parts

    @property
    def quantizer(self) -> Quantizer:
        if self.stages and isinstance(self.stages[-1], Quantizer):
            return self.stages[-1]
        return Float32Quantizer()

    @property
    def vector_stages(self) -> list[Stage]:
        return [s for s in self.stages if not isinstance(s, Quantizer)]

    @property
    def fitted(self) -> bool:
        return all(s.fitted for s in self.stages)

    # ------------------------------------------------------------ run

    def fit(self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None) -> Pipeline:
        """Fit every stage on the output of the stages before it."""
        if corpus.num_vectors == 0:
            raise ValueError("cannot fit a pipeline on a corpus with no vectors")
        _require_finite(corpus, "documents")
        if queries is not None:
            _require_finite(queries, "queries")
        x, q = corpus, queries
        for stage in self.vector_stages:
            stage.fit(x, q)
            x = stage.transform(x)
            if isinstance(stage, DimensionReducer) and q is not None:
                q = stage.transform_queries(q)
        self.quantizer.fit(x, q)
        return self

    def transform(self, corpus: MultiVectorCorpus) -> MultiVectorCorpus:
        """Token and dimension stages only: the float vectors the quantizer will see."""
        x = corpus
        for stage in self.vector_stages:
            x = stage.transform(x)
        return x

    def transform_queries(self, queries: MultiVectorCorpus) -> MultiVectorCorpus:
        _require_finite(queries, "queries")
        q = queries
        for stage in self.vector_stages:
            if isinstance(stage, DimensionReducer):
                q = stage.transform_queries(q)
        return q

    def compress(self, corpus: MultiVectorCorpus, fit: bool | None = None) -> CompressedCorpus:
        """Run every stage. Stages with unfitted state are fit on ``corpus`` unless
        ``fit=False``, in which case an unfitted stage is an error."""
        if fit or (fit is None and not self.fitted):
            self.fit(corpus)  # checks the input
        elif not self.fitted:
            raise RuntimeError("pipeline has unfitted stages; call fit() first")
        else:
            _require_finite(corpus, "documents")
        return self.encode(self.transform(corpus), original=corpus)

    def encode(self, transformed: MultiVectorCorpus, original: MultiVectorCorpus | None = None) -> CompressedCorpus:
        """Quantize an already-transformed corpus (the output of :meth:`transform`).

        Lets callers that try many codecs on the same merged / projected vectors
        (calibration does) run the token and dimension stages once. Byte
        accounting is against ``original`` when given.
        """
        original = original if original is not None else transformed
        x = transformed
        q = self.quantizer
        if not q.fitted:
            if x.num_vectors == 0:
                raise ValueError(f"cannot fit {q.name} on a corpus with no vectors")
            q.fit(x)
        # Encode in row blocks into one preallocated array: a codec's float
        # temporaries then cost one block, not a second copy of the corpus.
        codes = np.empty((x.num_vectors, q.code_bytes(x.dimension)), dtype=np.uint8)
        for lo in range(0, x.num_vectors, ENCODE_BLOCK_ROWS):
            hi = min(lo + ENCODE_BLOCK_ROWS, x.num_vectors)
            block = np.asarray(x.vectors[lo:hi], dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError(f"documents contain non-finite values (vectors {lo}..{hi - 1} after transform)")
            codes[lo:hi] = q.encode(block)
        stats = {
            "vectors_before": original.num_vectors,
            "dim_before": original.dimension,
            "float32_bytes_before": original.num_vectors * original.dimension * 4,
            "native_bytes_before": original.nbytes,
            "native_dtype_before": str(original.dtype),
        }
        return CompressedCorpus(codes, x.offsets, x.dimension, q, x.ids, stats=stats, pipeline=self.to_dict())

    # ------------------------------------------------------ description

    def to_dict(self) -> dict[str, Any]:
        return {"stages": [s.to_dict() for s in self.stages]}

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> Pipeline:
        return cls([stage_from_dict(s) for s in spec.get("stages", [])])

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def label(self) -> str:
        """A short human-readable name, e.g. ``redundancy(0.92) > binary``."""
        parts = []
        for s in self.stages:
            p = s.params()
            key = next(iter(p.values()), None) if p else None
            parts.append(f"{s.name}({key})" if key is not None and not isinstance(key, bool) else s.name)
        return " > ".join(parts) if parts else "float32"

    def __repr__(self) -> str:
        return f"Pipeline({self.stages!r})"


def save_compressed(compressed: CompressedCorpus, path: str | Path) -> Path:
    """Write codes + offsets + the pipeline that produced them (pickle-free).

    Quantizers with fitted state (``lloyd2``, per-dimension int8, centred
    binary) are not serialisable yet; saving one raises rather than writing a
    file that cannot be decoded.
    """
    if compressed.quantizer.overhead_bytes():
        raise NotImplementedError(f"saving a fitted {compressed.quantizer.name} codec is not supported yet")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "format": "optivision-compressed",
        "version": 1,
        "dim": compressed.dim,
        "ids": compressed.ids,
        "stats": compressed.stats,
        "pipeline": compressed.pipeline,
        "quantizer": compressed.quantizer.to_dict(),
    }
    np.savez(
        path,
        codes=compressed.codes,
        offsets=compressed.offsets,
        header=np.frombuffer(json.dumps(header, default=str).encode("utf-8"), dtype=np.uint8),
    )
    return path


def load_compressed(path: str | Path) -> CompressedCorpus:
    data = np.load(path, allow_pickle=False)
    header = json.loads(bytes(data["header"]).decode("utf-8"))
    if header.get("format") != "optivision-compressed":
        raise ValueError(f"{path} is not a compressed optivision corpus")
    quantizer = stage_from_dict(header["quantizer"])
    if not isinstance(quantizer, Quantizer):
        raise TypeError(f"{path}: header names a non-quantizer stage")
    return CompressedCorpus(
        data["codes"], data["offsets"], header["dim"], quantizer, header["ids"],
        stats=header.get("stats"), pipeline=header.get("pipeline"),
    )
