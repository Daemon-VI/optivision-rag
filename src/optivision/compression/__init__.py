"""Vector compression: PrunedPage -> CompressedPage."""

from __future__ import annotations

import numpy as np

from ..config import CompressionConfig
from ..types import CompressedPage, PrunedPage
from .binary import (
    code_nbytes,
    maxsim_asymmetric,
    maxsim_asymmetric_batch,
    maxsim_float,
    maxsim_hamming,
    pack_bits,
    unpack_signs,
)
from .lloyd2 import (
    Lloyd2Codec,
    decode_lloyd2,
    encode_lloyd2,
    fit_lloyd2,
    maxsim_asymmetric_lloyd2,
    maxsim_asymmetric_lloyd2_batch,
    pack2,
    rotation_matrix,
    unpack2,
)
from .lloyd2 import code_nbytes as lloyd2_code_nbytes

__all__ = [
    "Compressor",
    "Lloyd2Codec",
    "code_nbytes",
    "decode_lloyd2",
    "encode_lloyd2",
    "fit_lloyd2",
    "lloyd2_code_nbytes",
    "maxsim_asymmetric",
    "maxsim_asymmetric_batch",
    "maxsim_asymmetric_lloyd2",
    "maxsim_asymmetric_lloyd2_batch",
    "maxsim_float",
    "maxsim_hamming",
    "pack2",
    "pack_bits",
    "rotation_matrix",
    "unpack2",
    "unpack_signs",
]

# Full-scale value for int8 quantization.
#
# These vectors are L2-normalised, so a component is not free to reach 1.0: the
# mass is spread over ``dim`` dimensions and |c| concentrates near 1/sqrt(dim).
# On real ColSmol-256M output (dim 128) the mean |component| is 0.071 and the
# largest seen is 0.363. Mapping [-1, 1] onto the int8 range — as a naive
# ``round(v * 127)`` does — therefore sends the largest real component to level
# 46 and leaves ~two thirds of the 255 available levels permanently unused.
#
# Rescaling to the range the data actually occupies costs nothing: no extra byte
# is stored, so int8 stays exactly 4x. Measured on real vectors, 1 - cosine
# drops 4.0x, from 3.28e-4 to 8.20e-5.
#
# 0.5 rather than the observed maximum, because the cost is asymmetric: a scale
# slightly too large loses a little resolution, one slightly too small *clips*
# the tail and puts a hard error on exactly the largest, most informative
# components. At 0.5 (~5.7 sigma for unit-norm 128-d vectors) clipping is
# effectively impossible — worst-case reconstruction error over 5000 random unit
# vectors is 0.0020, against 0.0125 at scale 0.4 where the tail does clip.
# This follows from unit-norm geometry rather than from one checkpoint, so it
# transfers across the ColPali family.
INT8_SCALE = 0.50


class Compressor:
    """Turns pruned float vectors into the bytes that go into the index.

    ``codec`` carries corpus-fitted state that a stateless per-vector method
    (``binary``, ``int8``) does not need but ``lloyd2`` does -- the rotated
    2-bit quantizer's mean and scale are properties of the corpus, fit once via
    :func:`fit_lloyd2` and shared by every page, the same way
    ``TokenPruner(codebook=...)`` shares one fitted probe set.
    """

    def __init__(self, cfg: CompressionConfig, codec: Lloyd2Codec | None = None) -> None:
        self.cfg = cfg
        self.codec = codec

    @property
    def bytes_per_vector(self) -> int | None:
        """Index cost of one vector, or None when it depends on the dimension."""
        return None

    def compress(self, page: PrunedPage) -> CompressedPage:
        cfg = self.cfg
        vectors = np.ascontiguousarray(page.embeddings, dtype=np.float32)
        dim = int(vectors.shape[1])

        if not cfg.enabled or cfg.method == "none":
            # Store float32 verbatim, viewed as bytes, so the accounting in
            # CompressedPage stays uniform across methods.
            codes = vectors.view(np.uint8).reshape(vectors.shape[0], dim * 4)
        elif cfg.method == "binary":
            codes = pack_bits(vectors)
        elif cfg.method == "int8":
            codes = _int8_codes(vectors)
        elif cfg.method == "lloyd2":
            if self.codec is None:
                raise ValueError(
                    "compression.method='lloyd2' needs a fitted codec: "
                    "Compressor(cfg, codec=fit_lloyd2(corpus_vectors))"
                )
            codes = encode_lloyd2(vectors, self.codec)
        else:
            raise ValueError(f"unknown compression method {cfg.method!r}")

        scale = None
        if cfg.keep_norm:
            scale = np.linalg.norm(vectors, axis=1).astype(np.float32)

        return CompressedPage(
            ref=page.ref,
            codes=np.ascontiguousarray(codes, dtype=np.uint8),
            dim=dim,
            n_tokens_before=page.n_tokens_before,
            n_tokens_after=int(vectors.shape[0]),
            scale=scale,
            stats={
                "method": cfg.method if cfg.enabled else "none",
                "keep_ratio": page.keep_ratio,
                **page.stats,
            },
        )


def _int8_codes(vectors: np.ndarray) -> np.ndarray:
    """Scalar quantization to int8 — the 4x middle ground, for the ablation."""
    q = np.clip(np.round(vectors / INT8_SCALE * 127.0), -127, 127).astype(np.int8)
    return q.view(np.uint8)


def decode_int8(codes: np.ndarray, dim: int) -> np.ndarray:
    # Scaled in place: the chained form allocates the full array once per step,
    # which matters because this runs on whole index blocks.
    out = codes.view(np.int8).reshape(-1, dim).astype(np.float32)
    out *= INT8_SCALE / 127.0
    return out


def decode(
    codes: np.ndarray, dim: int, method: str, codec: Lloyd2Codec | None = None
) -> np.ndarray:
    """Inverse of :meth:`Compressor.compress` for one page's codes.

    ``codec`` is required when ``method == "lloyd2"`` -- the rotation and
    centring it carries are corpus-fitted state, not derivable from the codes
    alone. Every other method ignores it.
    """
    if method in ("none", None):
        return codes.view(np.float32).reshape(-1, dim)
    if method == "binary":
        return unpack_signs(codes, dim)
    if method == "int8":
        return decode_int8(codes, dim)
    if method == "lloyd2":
        if codec is None:
            raise ValueError("decoding method='lloyd2' needs the fitted codec")
        return decode_lloyd2(codes, dim, codec)
    raise ValueError(f"unknown compression method {method!r}")
