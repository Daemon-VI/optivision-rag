"""Dimension reducers: the pinned projectors plus ``RotationProjector``.

Four of the five are the classes of ``optivision.stages`` (same objects). Projections apply
to documents and to queries alike; none of them is evaluated here for retrieval quality.
"""

from __future__ import annotations

from optivision.stages.project import (
    DimensionProjector,
    PCAProjector,
    RandomProjector,
    TruncateProjector,
)

from .rotation import RotationProjector

__all__ = [
    "DimensionProjector",
    "PCAProjector",
    "RandomProjector",
    "RotationProjector",
    "TruncateProjector",
]
