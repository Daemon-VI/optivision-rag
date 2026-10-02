"""OptiVision RAG — extreme token compression for vision-language document retrieval.

Quick start::

    from optivision import Config, OptiVisionRAG

    rag = OptiVisionRAG(Config.load("configs/colsmol.yaml"))
    report = rag.build("data/corpus/pdfs")
    print(report.compression_ratio)
    print(rag.search("office memorandum on fire safety audit").hits[0].ref.page_id)
"""

from .calibration import CalibrationResult, calibrate
from .compose import CompressedCorpus, Pipeline
from .config import (
    CompressionConfig,
    Config,
    EncoderConfig,
    IndexConfig,
    IngestConfig,
    PruningConfig,
    SearchConfig,
)
from .optimize import OptimizationResult, optimize
from .pipeline import IndexReport, OptiVisionRAG
from .representation import MultiVectorCorpus, MultiVectorRepresentation
from .scoring import maxsim_matrix
from .stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    DimensionProjector,
    Float16Quantizer,
    Float32Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RandomProjector,
    RandomPruner,
    RedundancyPruner,
    SpatialPruner,
    TruncateProjector,
)
from .types import CompressedPage, PageEncoding, PageRef, PrunedPage, SearchHit, SearchResult

__version__ = "0.3.0"

__all__ = [
    "AdaptiveMerge",
    "BinaryQuantizer",
    "CalibrationResult",
    "CompressedCorpus",
    "CompressedPage",
    "CompressionConfig",
    "Config",
    "DimensionProjector",
    "EncoderConfig",
    "Float16Quantizer",
    "Float32Quantizer",
    "HierarchicalMerge",
    "IndexConfig",
    "IndexReport",
    "IngestConfig",
    "Int4Quantizer",
    "Int8Quantizer",
    "Lloyd2Quantizer",
    "MultiVectorCorpus",
    "MultiVectorRepresentation",
    "OptiVisionRAG",
    "OptimizationResult",
    "PCAProjector",
    "PageEncoding",
    "PageRef",
    "Pipeline",
    "PrunedPage",
    "PruningConfig",
    "RandomProjector",
    "RandomPruner",
    "RedundancyPruner",
    "SearchConfig",
    "SearchHit",
    "SearchResult",
    "SpatialPruner",
    "TruncateProjector",
    "__version__",
    "calibrate",
    "maxsim_matrix",
    "optimize",
]
