"""The model-agnostic layer.

Everything here works on ``[n_vectors, dim]`` float arrays and knows nothing about
pages, images or a particular encoder. See ``docs/package/UNIVERSAL_PACKAGE_DESIGN.md``.

A model-agnostic interface is not model-independent compression quality: the same
pipeline can keep retrieval quality on one encoder and lose it on another, or in
another coordinate basis of the same encoder. The selection layer measures; it may
abstain.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type checkers and IDEs; at run time the names below load on demand
    from .artifact import (
        Artifact,
        Certificate,
        VerificationReport,
        code_identity,
        create_certificate,
        load_artifact,
        save_artifact,
        sign_certificate,
        verify_certificate,
    )
    from .evaluate import (
        Evidence,
        LabelReference,
        RankingReference,
        Reference,
        metrics_from_ranking,
        per_query_metrics,
    )
    from .evaluate.collect import collect_evidence, compress_selected
    from .identity import array_digest, canonical_json, digest, sha256_hex, strings_digest
    from .pipeline import (
        CompressedCorpus,
        Pipeline,
        load_compressed,
        pipeline_identity,
        pipeline_spec,
        save_compressed,
    )
    from .retrieve import (
        CompressedRetriever,
        ExactRetriever,
        Ranking,
        Retriever,
        TieredRetriever,
        TwoTierRetriever,
    )
    from .selection import (
        STATEMENT,
        BonferroniBetting,
        CandidateSpace,
        CandidateSpec,
        Decision,
        FixedSequenceBetting,
        Outcome,
        SelectionRule,
        select,
    )
    from .similarity import VectorStore, maxsim_matrix, maxsim_pair, rank, segment_reduce
    from .stages import (
        BasisInvariance,
        ByteAccounting,
        PipelineTraits,
        RotationProjector,
        Stage,
        StageTraits,
        byte_accounting,
        fitted_state,
        pipeline_traits,
        restore_fitted_state,
        traits_of,
    )
    from .vectors import (
        ContractError,
        CorpusIdentity,
        EncoderIdentity,
        MultiVectorCorpus,
        MultiVectorRepresentation,
        as_corpus,
        as_host_array,
        corpus_identity,
        validate_corpus,
    )

#: name -> the core module that defines it. Names are resolved on access (PEP 562), so
#: ``import optivision.core`` costs nothing and no core module is imported by another one's
#: package initialiser (the certificate layer reads source files and loads ``importlib.metadata``).
_EXPORTS: dict[str, str] = {
    **dict.fromkeys(("canonical_json", "sha256_hex", "digest", "array_digest", "strings_digest"),
                    "identity"),
    **dict.fromkeys(("ContractError", "CorpusIdentity", "EncoderIdentity", "MultiVectorCorpus",
                     "MultiVectorRepresentation", "as_corpus", "as_host_array", "corpus_identity",
                     "validate_corpus"), "vectors"),
    **dict.fromkeys(("Pipeline", "CompressedCorpus", "load_compressed", "save_compressed",
                     "pipeline_identity", "pipeline_spec"), "pipeline"),
    **dict.fromkeys(("maxsim_matrix", "maxsim_pair", "rank", "segment_reduce", "VectorStore"),
                    "similarity"),
    **dict.fromkeys(("Stage", "BasisInvariance", "StageTraits", "PipelineTraits", "pipeline_traits",
                     "traits_of", "ByteAccounting", "byte_accounting", "fitted_state",
                     "restore_fitted_state", "RotationProjector"), "stages"),
    **dict.fromkeys(("Ranking", "Retriever", "ExactRetriever", "CompressedRetriever",
                     "TieredRetriever", "TwoTierRetriever"), "retrieve"),
    **dict.fromkeys(("Evidence", "Reference", "LabelReference", "RankingReference",
                     "collect_evidence", "compress_selected", "per_query_metrics",
                     "metrics_from_ranking"), "evaluate"),
    **dict.fromkeys(("CandidateSpace", "CandidateSpec", "Decision", "Outcome", "SelectionRule",
                     "FixedSequenceBetting", "BonferroniBetting", "select", "STATEMENT"),
                    "selection"),
    **dict.fromkeys(("Certificate", "Artifact", "VerificationReport", "create_certificate",
                     "sign_certificate", "verify_certificate", "save_artifact", "load_artifact",
                     "code_identity"), "artifact"),
}

# A literal list (it must equal ``sorted(_EXPORTS)``; a test checks that), so that tools can read it.
__all__ = [
    "STATEMENT",
    "Artifact",
    "BasisInvariance",
    "BonferroniBetting",
    "ByteAccounting",
    "CandidateSpace",
    "CandidateSpec",
    "Certificate",
    "CompressedCorpus",
    "CompressedRetriever",
    "ContractError",
    "CorpusIdentity",
    "Decision",
    "EncoderIdentity",
    "Evidence",
    "ExactRetriever",
    "FixedSequenceBetting",
    "LabelReference",
    "MultiVectorCorpus",
    "MultiVectorRepresentation",
    "Outcome",
    "Pipeline",
    "PipelineTraits",
    "Ranking",
    "RankingReference",
    "Reference",
    "Retriever",
    "RotationProjector",
    "SelectionRule",
    "Stage",
    "StageTraits",
    "TieredRetriever",
    "TwoTierRetriever",
    "VectorStore",
    "VerificationReport",
    "array_digest",
    "as_corpus",
    "as_host_array",
    "byte_accounting",
    "canonical_json",
    "code_identity",
    "collect_evidence",
    "compress_selected",
    "corpus_identity",
    "create_certificate",
    "digest",
    "fitted_state",
    "load_artifact",
    "load_compressed",
    "maxsim_matrix",
    "maxsim_pair",
    "metrics_from_ranking",
    "per_query_metrics",
    "pipeline_identity",
    "pipeline_spec",
    "pipeline_traits",
    "rank",
    "restore_fitted_state",
    "save_artifact",
    "save_compressed",
    "segment_reduce",
    "select",
    "sha256_hex",
    "sign_certificate",
    "strings_digest",
    "traits_of",
    "validate_corpus",
    "verify_certificate",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # later lookups are plain attribute reads
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
