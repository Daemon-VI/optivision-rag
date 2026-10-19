"""OptiVision RAG — extreme token compression for vision-language document retrieval.

Quick start::

    from optivision import Config, OptiVisionRAG

    rag = OptiVisionRAG(Config.load("configs/colsmol.yaml"))
    report = rag.build("data/corpus/pdfs")
    print(report.compression_ratio)
    print(rag.search("office memorandum on fire safety audit").hits[0].ref.page_id)
"""

import importlib as _importlib
import typing as _typing

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

# Importing the core stage package registers the ``rotation`` stage and the structural traits of
# every built-in stage, so that ``Pipeline.from_dict`` can rebuild a specification that names it.
from .core import stages as _core_stages  # noqa: F401  (imported for its registrations)
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

__version__ = "0.5.0"

__all__ = [
    "AdaptiveMerge",
    "BinaryQuantizer",
    "CalibrationResult",
    "CandidateSpace",
    "Certificate",
    "CertifiedResult",
    "CompressedCorpus",
    "CompressedPage",
    "CompressionConfig",
    "Config",
    "Decision",
    "DimensionProjector",
    "EncoderConfig",
    "EncoderIdentity",
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
    "Outcome",
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
    "certify",
    "encode_and_certify",
    "load_artifact",
    "maxsim_matrix",
    "optimize",
    "save_artifact",
    "verify",
    "verify_certificate",
]

# --- certified selection (additive) ---------------------------------------------------------
# These names are resolved on demand (PEP 562) rather than imported here: loading the
# certificate and artifact modules reads ``importlib.metadata`` and costs more than importing the
# rest of the package did before, and nobody who only compresses needs them. ``from optivision
# import certify`` and ``optivision.certify`` work as for any other name; the 40 names above are
# imported eagerly, as they always were.
_LAZY = {
    "CandidateSpace": ".core.selection",
    "Certificate": ".core.artifact",
    "CertifiedResult": ".api",
    "Decision": ".core.selection",
    "EncoderIdentity": ".core.vectors",
    "Outcome": ".core.selection",
    "certify": ".api",
    "encode_and_certify": ".api",
    "load_artifact": ".core.artifact",
    "save_artifact": ".core.artifact",
    "verify": ".api",
    "verify_certificate": ".core.artifact",
}

if _typing.TYPE_CHECKING:  # for type checkers and IDEs; at run time the names above load on demand
    from .api import CertifiedResult, certify, encode_and_certify, verify
    from .core.artifact import Certificate, load_artifact, save_artifact, verify_certificate
    from .core.selection import CandidateSpace, Decision, Outcome
    from .core.vectors import EncoderIdentity


def __getattr__(name: str) -> _typing.Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(module, __name__), name)
    globals()[name] = value  # later lookups are plain attribute reads
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
