"""OptiVision RAG — extreme token compression for vision-language document retrieval.

Quick start::

    from optivision import Config, OptiVisionRAG

    rag = OptiVisionRAG(Config.load("configs/colsmol.yaml"))
    report = rag.build("data/corpus/pdfs")
    print(report.compression_ratio)
    print(rag.search("office memorandum on fire safety audit").hits[0].ref.page_id)
"""

from .config import (
    CompressionConfig,
    Config,
    EncoderConfig,
    IndexConfig,
    IngestConfig,
    PruningConfig,
    SearchConfig,
)
from .pipeline import IndexReport, OptiVisionRAG
from .types import CompressedPage, PageEncoding, PageRef, PrunedPage, SearchHit, SearchResult

__version__ = "0.1.0"

__all__ = [
    "CompressedPage",
    "CompressionConfig",
    "Config",
    "EncoderConfig",
    "IndexConfig",
    "IndexReport",
    "IngestConfig",
    "OptiVisionRAG",
    "PageEncoding",
    "PageRef",
    "PrunedPage",
    "PruningConfig",
    "SearchConfig",
    "SearchHit",
    "SearchResult",
    "__version__",
]
