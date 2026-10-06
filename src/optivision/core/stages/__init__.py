"""Compression stages for any ``[n_vectors, dim]`` float vectors, with their basis traits.

The stage contracts and the model-agnostic stage classes are the objects of
``optivision.stages`` (nothing is copied); on top of them this package adds:

* ``traits``: for every stage, which changes of coordinate basis leave its scores unchanged
  (``BasisInvariance``), and the rule that combines them for a pipeline;
* ``state``: fitted state as arrays and scalars, with digests and a checked restore;
* ``accounting``: the library's size key beside the bytes it leaves out;
* ``rotation``: ``RotationProjector``, a seeded orthogonal map for diagnosing basis dependence.

A model-agnostic interface is not model-independent compression quality. Most codecs act on
the stored coordinates, so the same pipeline can keep quality in one basis of an encoder and
lose it in another; nothing here assumes ``codec(x)`` behaves like ``codec(xR)``.

Importing this package registers the ``"rotation"`` stage and the traits of every built-in
stage. Code that rebuilds pipelines from their dict form must import it before reading a
pipeline that contains a rotation.
"""

from __future__ import annotations

from optivision.stages.base import (
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

from .accounting import ByteAccounting, byte_accounting
from .codecs import (
    BinaryQuantizer,
    Float16Quantizer,
    Float32Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
)
from .projections import (
    DimensionProjector,
    PCAProjector,
    RandomProjector,
    RotationProjector,
    TruncateProjector,
)
from .reducers import AdaptiveMerge, HierarchicalMerge, RandomPruner, RedundancyPruner
from .rotation import seeded_rotation
from .state import (
    fitted_state,
    fitted_state_digest,
    fitted_state_nbytes,
    pipeline_fitted_identity,
    pipeline_fitted_state,
    restore_fitted_state,
    restore_pipeline_state,
)
from .traits import (
    BasisInvariance,
    PipelineTraits,
    StageTraits,
    pipeline_traits,
    register_traits,
    traits_of,
)

__all__ = [
    "STAGES",
    "AdaptiveMerge",
    "BasisInvariance",
    "BinaryQuantizer",
    "ByteAccounting",
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
    "PipelineTraits",
    "Quantizer",
    "RandomProjector",
    "RandomPruner",
    "Reduction",
    "RedundancyPruner",
    "RotationProjector",
    "Stage",
    "StageTraits",
    "TokenReducer",
    "TruncateProjector",
    "byte_accounting",
    "fitted_state",
    "fitted_state_digest",
    "fitted_state_nbytes",
    "pipeline_fitted_identity",
    "pipeline_fitted_state",
    "pipeline_traits",
    "register",
    "register_traits",
    "restore_fitted_state",
    "restore_pipeline_state",
    "seeded_rotation",
    "stage_from_dict",
    "traits_of",
]
