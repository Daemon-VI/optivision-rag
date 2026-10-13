"""The retrieval seam of the model-agnostic layer: queries in, ranked ids out.

    from optivision.core.retrieve import ExactRetriever, CompressedRetriever, TieredRetriever

Every retriever delegates to the library's existing functions and adds validation and a typed
:class:`Ranking`; none adds arithmetic.

* ``ExactRetriever`` / ``CompressedRetriever``: ``ExactIndex`` over a float corpus / a compressed
  corpus with the pipeline's query projection.
* ``TieredRetriever``: ``TieredIndex`` (hot shortlist, cold rescoring), with an alignment check.
* ``TwoTierRetriever``: the same for any shortlist source (a backend, for instance) and any
  reranker.

Invariants (``docs/package/BACKEND_ARCHITECTURE.md``): scores are float32; in-process retrievers
rank with ``scoring.rank`` (descending, ties by ascending store position); an empty document
scores ``-inf`` and an empty query 0; scoring is decode-first, with no native compressed-domain
kernel; ``k >= 1`` and candidate positions are validated; ``candidates`` is clamped to the number
of documents; tiers must hold the same ids in the same order; ``isinstance`` against a protocol
checks attribute presence only.

Scores are comparable only within one ``score_kind`` (:func:`comparable`). The dense
``[n_queries, n_docs]`` matrix is materialised by scoring and by search, so the interface makes no
claim about large corpora. Nothing here selects a device or reads an environment variable.
"""

from __future__ import annotations

from .checks import as_queries, check_candidates, check_k, check_positions
from .exact import (
    CompressedRetriever,
    ExactRetriever,
    ranking_from_positions,
    ranking_from_scores,
)
from .tiered import TieredRetriever, TwoTierRetriever
from .types import (
    SCORE_MAXSIM_COSINE,
    SCORE_MAXSIM_IP,
    SCORE_TIERED,
    TIE_INDEX,
    TIE_UNSPECIFIED,
    DenseScorer,
    Ranking,
    Reranker,
    Retriever,
    comparable,
    require_comparable,
)

__all__ = [
    "SCORE_MAXSIM_COSINE",
    "SCORE_MAXSIM_IP",
    "SCORE_TIERED",
    "TIE_INDEX",
    "TIE_UNSPECIFIED",
    "CompressedRetriever",
    "DenseScorer",
    "ExactRetriever",
    "Ranking",
    "Reranker",
    "Retriever",
    "TieredRetriever",
    "TwoTierRetriever",
    "as_queries",
    "check_candidates",
    "check_k",
    "check_positions",
    "comparable",
    "ranking_from_positions",
    "ranking_from_scores",
    "require_comparable",
]
