"""int4 with a norm-preserving scale: an exploratory codec for ``s17_int4_norm.py``.

It lives beside the study, not in the package: ``optivision.stages.quantize`` is byte-pinned
for the X1-X3 experiments (``scripts/research/x_common.py`` checks it), and the codec is not
registered as a stage (``optivision.stages.base.STAGES``), so no package spec can name it.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from optivision.stages.quantize import _MIN_SCALE, Int4Quantizer


class NormPreservingInt4Quantizer(Int4Quantizer):
    """int4 whose float16 scale gives each decoded vector the norm of the vector it codes.

    The levels, the code size and ``decode`` are :class:`Int4Quantizer`'s; only the stored
    scale differs. Under ``center="mean"`` the norm is that of the decoded vector with the
    mean added back. Labels: ``int4norm`` and ``int4norm(mean)``.
    """

    name: ClassVar[str] = "int4norm"

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        v = np.asarray(vectors, dtype=np.float32)
        target = np.linalg.norm(v, axis=1)
        if self.center == "mean":
            if self.mean is None:
                raise RuntimeError("NormPreservingInt4Quantizer(center='mean') must be fit before use")
            v = v - self.mean
        n, dim = v.shape
        amax = np.maximum(np.abs(v).max(axis=1), _MIN_SCALE).astype(np.float16).astype(np.float32)
        levels = np.clip(np.round(v / amax[:, None] * 7.0), -7, 7)
        q = (levels + 8).astype(np.uint8)  # 1..15, as Int4Quantizer
        scale = self._norm_preserving_scale(levels, amax, target)
        if dim % 2:
            q = np.concatenate([q, np.full((n, 1), 8, np.uint8)], axis=1)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        return np.concatenate([packed, scale.astype(np.float16).view(np.uint8).reshape(-1, 2)], axis=1)

    def _norm_preserving_scale(self, levels: np.ndarray, amax: np.ndarray, target: np.ndarray) -> np.ndarray:
        """The stored scale s with ||mean + levels * s / 7|| = target, keeping the levels.

        Without a mean it is a ratio of norms. With one it is the root of a quadratic
        in s, the root nearer the plain scale; a vector with no positive root keeps
        the plain scale.
        """
        a = np.einsum("ij,ij->i", levels, levels)
        m = self.mean if self.center == "mean" else None
        if m is None:
            step = target / np.sqrt(np.where(a > 0, a, 1.0))
        else:
            b = 2.0 * (levels @ m)
            c = float(m @ m) - target * target
            disc = np.sqrt(np.maximum(b * b - 4.0 * a * c, 0.0))
            safe = np.where(a > 0, 2.0 * a, 1.0)
            hi, lo = (-b + disc) / safe, (-b - disc) / safe
            plain = amax / 7.0
            step = np.where(np.abs(hi - plain) <= np.abs(lo - plain), hi, lo)
            step = np.where(b * b - 4.0 * a * c >= 0, step, plain)
        ok = (a > 0) & (step > 0) & np.isfinite(step)
        return np.where(ok, np.minimum(7.0 * step, float(np.finfo(np.float16).max)), amax).astype(np.float32)
