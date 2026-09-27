"""Quantizers: the last stage of a pipeline.

Each one maps float document vectors to uint8 codes and back. Scoring is always
asymmetric -- the float32 query against the decoded document -- so the codec
decides only what the document side loses.

The existing codecs are wrapped, not rewritten: ``binary`` and ``int8`` call the
functions the original ``Compressor`` uses, so their codes are byte-identical.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..compression import (
    INT8_SCALE,
    Lloyd2Codec,
    decode_int8,
    decode_lloyd2,
    encode_lloyd2,
    fit_lloyd2,
    lloyd2_code_nbytes,
    pack_bits,
    unpack_signs,
)
from ..compression import _int8_codes as _legacy_int8_codes
from ..representation import MultiVectorCorpus
from .base import Quantizer, register


@register
class Float32Quantizer(Quantizer):
    """No quantization: the float32 reference, stored as raw bytes."""

    name: ClassVar[str] = "float32"

    def params(self) -> dict[str, Any]:
        return {}

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        return v.view(np.uint8).reshape(v.shape[0], v.shape[1] * 4)

    def decode(self, codes: np.ndarray, dim: int) -> np.ndarray:
        return np.ascontiguousarray(codes).view(np.float32).reshape(-1, dim)

    def code_bytes(self, dim: int) -> int:
        return 4 * dim


@register
class Float16Quantizer(Quantizer):
    """Half precision: 2x, and the format many models are already served in."""

    name: ClassVar[str] = "float16"

    def params(self) -> dict[str, Any]:
        return {}

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(vectors, dtype=np.float16)
        return v.view(np.uint8).reshape(v.shape[0], v.shape[1] * 2)

    def decode(self, codes: np.ndarray, dim: int) -> np.ndarray:
        return np.ascontiguousarray(codes).view(np.float16).reshape(-1, dim).astype(np.float32)

    def code_bytes(self, dim: int) -> int:
        return 2 * dim


def _fit_sample(vectors: np.ndarray, n: int = 200_000, seed: int = 0) -> np.ndarray:
    v = np.asarray(vectors)
    if v.shape[0] > n:
        v = v[np.sort(np.random.default_rng(seed).choice(v.shape[0], n, replace=False))]
    return np.asarray(v, dtype=np.float32)


@register
class BinaryQuantizer(Quantizer):
    """One sign bit per dimension (32x vs float32), decoded to +/-1.

    ``center="mean"`` takes the sign of ``v - mu`` for a fitted corpus mean
    ``mu`` (the review's ``sign(d - mu)``). Decoding stays +/-1: adding ``mu``
    back would shift every query token's score by the same constant across
    documents, so MaxSim rankings are identical either way. ``mu`` is counted
    as stored overhead because appending new documents needs it.

    **Measured hazard:** centring helps on ColPali InfoVQA (97.4% -> 98.5%) but
    collapses a text ColBERT whose queries share one dominant direction
    (SciFact: 94.1% -> 1.1%): the query's common component multiplies the
    sign code's large error on each small residual and drowns the relevance
    signal. Scalar codecs do not have this problem because they decode the
    residual accurately and add ``mu`` back. Measure before using it.
    """

    name: ClassVar[str] = "binary"

    def __init__(self, center: str | None = None) -> None:
        if center not in (None, "mean"):
            raise ValueError("center must be None or 'mean'")
        self.center = center
        self.mean: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {"center": self.center} if self.center else {}

    @property
    def fitted(self) -> bool:
        return self.center is None or self.mean is not None

    def fit(self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None) -> BinaryQuantizer:
        if self.center == "mean":
            self.mean = _fit_sample(corpus.vectors).mean(axis=0).astype(np.float32)
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        v = np.asarray(vectors, dtype=np.float32)
        if self.center == "mean":
            if self.mean is None:
                raise RuntimeError("BinaryQuantizer(center='mean') must be fit before use")
            v = v - self.mean
        return pack_bits(v)

    def decode(self, codes: np.ndarray, dim: int) -> np.ndarray:
        return unpack_signs(codes, dim)

    def code_bytes(self, dim: int) -> int:
        return (dim + 7) // 8

    def overhead_bytes(self) -> int:
        return int(self.mean.nbytes) if self.mean is not None else 0


@register
class Int8Quantizer(Quantizer):
    """8-bit symmetric scalar quantization (4x vs float32, a little less with scales).

    ``scale`` decides what 127 means:

    ``"fixed"``          one constant (0.5) for every component -- the original
                         codec, tuned to unit-norm 128-d vectors. On wide
                         embeddings (components ~1/sqrt(d)) it wastes most of the
                         int8 range.
    ``"per_vector"``     each vector's own max |component|, stored as float16
                         alongside it (+2 bytes/vector). No fitting, any dimension.
    ``"per_dimension"``  a fitted per-dimension range (99.99th percentile of
                         |component|), stored once per index; out-of-range values
                         clip.
    """

    name: ClassVar[str] = "int8"

    def __init__(self, scale: str = "fixed", center: str | None = None) -> None:
        if scale not in {"fixed", "per_vector", "per_dimension"}:
            raise ValueError(f"unknown int8 scale mode {scale!r}")
        if center not in (None, "mean"):
            raise ValueError("center must be None or 'mean'")
        self.scale = scale
        self.center = center
        self.ranges: np.ndarray | None = None
        self.mean: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {"scale": self.scale, **({"center": self.center} if self.center else {})}

    @property
    def fitted(self) -> bool:
        return (self.scale != "per_dimension" or self.ranges is not None) and (
            self.center is None or self.mean is not None
        )

    def fit(self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None) -> Int8Quantizer:
        sample = _fit_sample(corpus.vectors)
        if self.center == "mean":
            self.mean = sample.mean(axis=0).astype(np.float32)
            sample = sample - self.mean
        if self.scale == "per_dimension":
            self.ranges = np.maximum(np.percentile(np.abs(sample), 99.99, axis=0), 1e-8).astype(np.float32)
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        v = np.asarray(vectors, dtype=np.float32)
        if self.center == "mean":
            if self.mean is None:
                raise RuntimeError("Int8Quantizer(center='mean') must be fit before use")
            v = v - self.mean
        if self.scale == "fixed":
            return _legacy_int8_codes(v)
        if self.scale == "per_dimension":
            if self.ranges is None:
                raise RuntimeError("Int8Quantizer(scale='per_dimension') must be fit before use")
            q = np.clip(np.round(v / self.ranges * 127.0), -127, 127).astype(np.int8)
            return q.view(np.uint8)
        amax = np.maximum(np.abs(v).max(axis=1), 1e-12).astype(np.float16).astype(np.float32)
        q = np.clip(np.round(v / amax[:, None] * 127.0), -127, 127).astype(np.int8)
        scales = amax.astype(np.float16).view(np.uint8).reshape(-1, 2)
        return np.concatenate([q.view(np.uint8), scales], axis=1)

    def decode(self, codes: np.ndarray, dim: int) -> np.ndarray:
        c = np.ascontiguousarray(codes)
        if self.scale == "fixed":
            out = decode_int8(c, dim)
        elif self.scale == "per_dimension":
            out = c.view(np.int8).reshape(-1, dim).astype(np.float32)
            out *= self.ranges / 127.0
        else:
            out = np.ascontiguousarray(c[:, :dim]).view(np.int8).astype(np.float32)
            scale = np.ascontiguousarray(c[:, dim : dim + 2]).view(np.float16).astype(np.float32)
            out *= scale / 127.0
        if self.center == "mean":
            out += self.mean
        return out

    def code_bytes(self, dim: int) -> int:
        return dim + (2 if self.scale == "per_vector" else 0)

    def overhead_bytes(self) -> int:
        n = int(self.ranges.nbytes) if self.ranges is not None else 0
        return n + (int(self.mean.nbytes) if self.mean is not None else 0)

    @property
    def full_scale(self) -> float:
        return INT8_SCALE


@register
class Int4Quantizer(Quantizer):
    """4-bit symmetric scalar quantization with a per-vector float16 scale (~8x).

    Levels -7..7 times each vector's max |component| / 7; two codes per byte.
    """

    name: ClassVar[str] = "int4"

    def __init__(self, center: str | None = None) -> None:
        if center not in (None, "mean"):
            raise ValueError("center must be None or 'mean'")
        self.center = center
        self.mean: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {"center": self.center} if self.center else {}

    @property
    def fitted(self) -> bool:
        return self.center is None or self.mean is not None

    def fit(self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None) -> Int4Quantizer:
        if self.center == "mean":
            self.mean = _fit_sample(corpus.vectors).mean(axis=0).astype(np.float32)
        return self

    def overhead_bytes(self) -> int:
        return int(self.mean.nbytes) if self.mean is not None else 0

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        v = np.asarray(vectors, dtype=np.float32)
        if self.center == "mean":
            if self.mean is None:
                raise RuntimeError("Int4Quantizer(center='mean') must be fit before use")
            v = v - self.mean
        n, dim = v.shape
        amax = np.maximum(np.abs(v).max(axis=1), 1e-12).astype(np.float16).astype(np.float32)
        q = (np.clip(np.round(v / amax[:, None] * 7.0), -7, 7) + 8).astype(np.uint8)  # 1..15
        if dim % 2:
            q = np.concatenate([q, np.full((n, 1), 8, np.uint8)], axis=1)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        return np.concatenate([packed, amax.astype(np.float16).view(np.uint8).reshape(-1, 2)], axis=1)

    def decode(self, codes: np.ndarray, dim: int) -> np.ndarray:
        c = np.ascontiguousarray(codes)
        nb = (dim + 1) // 2
        packed = c[:, :nb]
        out = np.empty((c.shape[0], nb * 2), dtype=np.float32)
        out[:, 0::2] = (packed & 0x0F).astype(np.float32) - 8.0
        out[:, 1::2] = (packed >> 4).astype(np.float32) - 8.0
        scale = np.ascontiguousarray(c[:, nb : nb + 2]).view(np.float16).astype(np.float32)
        out = out[:, :dim]
        out *= scale / 7.0
        if self.center == "mean":
            out += self.mean
        return out

    def code_bytes(self, dim: int) -> int:
        return (dim + 1) // 2 + 2


@register
class Lloyd2Quantizer(Quantizer):
    """Random rotation + 2-bit Lloyd-Max levels (16x), fitted mean and scale."""

    name: ClassVar[str] = "lloyd2"

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.codec: Lloyd2Codec | None = None

    def params(self) -> dict[str, Any]:
        return {"seed": self.seed}

    @property
    def fitted(self) -> bool:
        return self.codec is not None

    def fit(self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None) -> Lloyd2Quantizer:
        self.codec = fit_lloyd2(np.asarray(corpus.vectors, dtype=np.float32), seed=self.seed)
        return self

    def _need(self) -> Lloyd2Codec:
        if self.codec is None:
            raise RuntimeError("Lloyd2Quantizer must be fit before use")
        return self.codec

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        return encode_lloyd2(np.asarray(vectors, dtype=np.float32), self._need())

    def decode(self, codes: np.ndarray, dim: int) -> np.ndarray:
        return decode_lloyd2(codes, dim, self._need()).astype(np.float32, copy=False)

    def code_bytes(self, dim: int) -> int:
        return lloyd2_code_nbytes(dim)

    def overhead_bytes(self) -> int:
        return self.codec.overhead_bytes if self.codec is not None else 0
