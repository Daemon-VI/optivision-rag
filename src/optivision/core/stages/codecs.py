"""Quantizers (the last stage of a pipeline): the same classes as ``optivision.stages``.

Every codec here other than ``Float32Quantizer`` acts on the coordinates of the vectors as
stored. How much a codec keeps therefore depends on the coordinate basis; see
``core.stages.traits`` for what is declared per codec.
"""

from __future__ import annotations

from optivision.stages.int4norm import NormPreservingInt4Quantizer
from optivision.stages.quantize import (
    BinaryQuantizer,
    Float16Quantizer,
    Float32Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
)

__all__ = [
    "BinaryQuantizer",
    "Float16Quantizer",
    "Float32Quantizer",
    "Int4Quantizer",
    "Int8Quantizer",
    "Lloyd2Quantizer",
    "NormPreservingInt4Quantizer",
]
