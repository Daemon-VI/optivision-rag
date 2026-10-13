"""A seeded orthogonal map as a dimension stage.

``RotationProjector`` multiplies every document and query vector by one orthogonal
matrix. Inner products, and therefore float MaxSim scores, are unchanged up to float32
rounding; what changes is the coordinate basis that a later coordinate codec (sign
bits, fixed-step integers) sees. That makes it a diagnostic for basis dependence: put it
in front of a codec and compare with the same pipeline without it.

The matrix is the orthogonal factor of a QR decomposition of a standard normal matrix
drawn from ``np.random.default_rng(seed)``, with each column multiplied by the sign of
the corresponding diagonal entry of ``R``. LAPACK is free to return ``R`` with either
sign on its diagonal; without that last step the matrix would depend on the library
build. The step also makes the factor Haar-distributed over seeds.

The matrix is built the same way as the rotation of the registered symmetry experiment
(``rotation`` in ``scripts/research/x3_symmetry.py``, read-only), but it is applied
differently. This stage multiplies row vectors as ``x @ R``. The experiment applied its
matrix ``U`` as ``v -> U v``, which for row vectors is ``x @ U.T``. For the same seed
``R == U``, so the two maps are inverses of each other: both are rotations, and results
for one seed do not transfer to the other. ``RotationProjector`` does not reproduce the
experiment's rotations.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from optivision.representation import MultiVectorCorpus
from optivision.stages.base import DimensionReducer, register

__all__ = ["RotationProjector", "seeded_rotation"]


def seeded_rotation(dim: int, seed: int) -> np.ndarray:
    """The orthogonal matrix ``R`` (float32, ``[dim, dim]``) used by ``RotationProjector``.

    Rows are vectors, so a batch ``x`` becomes ``x @ R``. The draw happens in float64 and
    needs about ``3 * 8 * dim**2`` bytes while it runs; the result is ``4 * dim**2``.
    """
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim < 1:
        raise ValueError(f"dim must be a positive integer, got {dim!r}")
    g = np.random.default_rng(seed).standard_normal((int(dim), int(dim)))
    q, r = np.linalg.qr(g)
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return np.ascontiguousarray(q * signs, dtype=np.float32)


@register
class RotationProjector(DimensionReducer):
    """Rotate documents and queries by one seeded orthogonal matrix (output dimension = input).

    Engineering capability; not evaluated by the registered experiments. It is a diagnostic
    for basis dependence and is not presented as an improvement: a registered experiment
    (X3; ``reports/research/X_RESULTS.md``, per-dataset outputs in ``reports/research/X3/``)
    found that the outcome of sign-code and 4-bit pipelines depends on the coordinate basis
    under rotation: rotating the vectors can reduce the compression the selection certifies
    (it did on SciFact) and on one extension dataset it moved the outcome in both directions.
    A rotation in front of a codec has to be measured like any other candidate, and the
    selection may abstain.

    The float scores of a pipeline made only of this stage are those of the input, up to
    float32 rounding. The coordinates a later codec sees are a rotation of the stored ones
    (the pipeline rule in ``core.stages.traits`` therefore treats later
    ``SIGNED_PERMUTATION`` stages as ``NONE``).

    The matrix does not depend on the data, only on ``seed`` and the corpus dimension. It
    is built at ``fit`` and is the fitted state, so a stored artifact does not depend on
    the QR routine of the machine that loads it.

    Memory: the stage holds a ``d x d`` float32 matrix (4 * d**2 bytes: about 64 MiB at
    d = 4096), and ``DimensionReducer.transform`` casts the whole corpus to float32 at once.
    """

    name: ClassVar[str] = "rotation"

    def __init__(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        self.seed = int(seed)
        self.matrix: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {"seed": self.seed}

    @property
    def fitted(self) -> bool:
        return self.matrix is not None

    def output_dim(self, input_dim: int) -> int:
        return input_dim

    def fit(
        self, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None
    ) -> RotationProjector:
        self.matrix = seeded_rotation(corpus.dimension, self.seed)
        return self

    def project(self, vectors: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("RotationProjector must be fit before use")
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != self.matrix.shape[0]:
            raise ValueError(
                f"RotationProjector was fitted on {self.matrix.shape[0]}-d vectors, "
                f"got shape {v.shape}"
            )
        return v @ self.matrix
