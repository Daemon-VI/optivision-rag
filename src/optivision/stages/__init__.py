"""Composable compression stages. See :mod:`optivision.stages.base` for the contract."""

from __future__ import annotations

from .base import (
    STAGES,
    DimensionReducer,
    DocView,
    Quantizer,
    Reduction,
    Stage,
    TokenReducer,
    register,
    stage_from_dict,
)
from .merge import AdaptiveMerge, HierarchicalMerge, RandomPruner
from .project import DimensionProjector, PCAProjector, RandomProjector, TruncateProjector
from .prune import RedundancyPruner, SpatialPruner
from .quantize import (
    BinaryQuantizer,
    Float16Quantizer,
    Float32Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
)

__all__ = [
    "STAGES",
    "AdaptiveMerge",
    "BinaryQuantizer",
    "DimensionProjector",
    "DimensionReducer",
    "DocView",
    "Float16Quantizer",
    "Float32Quantizer",
    "HierarchicalMerge",
    "Int4Quantizer",
    "Int8Quantizer",
    "Lloyd2Quantizer",
    "PCAProjector",
    "Quantizer",
    "RandomProjector",
    "RandomPruner",
    "Reduction",
    "RedundancyPruner",
    "SpatialPruner",
    "Stage",
    "TokenReducer",
    "TruncateProjector",
    "register",
    "stage_from_dict",
]
