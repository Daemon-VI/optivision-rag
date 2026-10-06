"""References: what "relevant" means when per-query metrics are computed.

Two references exist, and they measure different things. Results under the two are never
merged, and a statement about one is never reported as a statement about the other.

``LabelReference``    relevance labels (qrels). The metric of a compressed index against the
                      labels, relative to the metric of the uncompressed index against the
                      same labels: "is the compressed index still as right as the original?"

``RankingReference``  the uncompressed index's own top document(s). Agreement with the
                      uncompressed ranking measures fidelity to that ranking, *including its
                      mistakes*. It needs no labels, it is not retrieval quality, and only
                      depth 1 was studied.

Both delegate to the library's own functions (``evaluation.relevant_from_qrels`` and
``evaluation.relevant_from_baseline``), which ``calibration.reference_relevance`` also uses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from optivision.core.identity import digest
from optivision.evaluation import relevant_from_baseline, relevant_from_qrels


@runtime_checkable
class Reference(Protocol):
    """What ``collect_evidence`` needs from a reference."""

    construct: str

    def relevant(self, corpus_ids: Sequence[str], query_ids: Sequence[str],
                 base_scores: np.ndarray | None) -> list[np.ndarray]: ...

    def identity(self) -> dict[str, Any]: ...


class LabelReference:
    """Relevance labels: ``qrels[query_id]`` is an iterable of relevant document ids.

    Relevance is binary, as in ``relevant_from_qrels``: every document listed for a query
    is relevant whatever its grade, and a document id missing from the corpus is ignored.
    A query without a listed relevant document gets an empty relevant set, which the
    caller removes for every candidate and the baseline together. ``base_scores`` is not
    used. The mapping is copied at construction, so later changes to it have no effect.
    """

    construct = "labels"

    def __init__(self, qrels: Mapping[Any, Any]) -> None:
        if not isinstance(qrels, Mapping):
            raise TypeError(f"qrels must be a mapping {{query_id: documents}}, "
                            f"got {type(qrels).__name__}")
        snapshot = {}
        for query_id, docs in qrels.items():
            if isinstance(docs, (str, bytes)):
                raise TypeError(f"qrels[{query_id!r}] must be a collection of document ids, "
                                "not a single string")
            snapshot[query_id] = tuple(docs)
        self._qrels = snapshot
        # what the metric reads: which documents are listed per query, not their grades
        canonical = {str(q): sorted(str(d) for d in docs) for q, docs in snapshot.items()}
        self._digest = digest({"qrels": canonical})

    def relevant(self, corpus_ids: Sequence[str], query_ids: Sequence[str],
                 base_scores: np.ndarray | None = None) -> list[np.ndarray]:
        return relevant_from_qrels(self._qrels, query_ids, corpus_ids)

    def identity(self) -> dict[str, Any]:
        """``{"construct", "qrels_sha256"}``; the digest does not depend on dict ordering."""
        return {"construct": self.construct, "qrels_sha256": self._digest}

    def __repr__(self) -> str:
        return f"LabelReference(n_queries={len(self._qrels)})"


class RankingReference:
    """Agreement with the uncompressed index's own top ``depth`` documents.

    The relevant set of a query is the baseline's top ``depth`` documents (ties broken by
    document index), so the baseline scores 1 on every query by construction and a
    compressed index scores by how well it keeps those documents near the top. This is
    fidelity to the uncompressed ranking including its mistakes. It is not retrieval
    quality and must not be reported as relevance. Results under this construct are never
    merged with results under ``LabelReference``, and a statement about one is never
    reported as a statement about the other. Only ``depth = 1`` was studied; any other depth
    is an engineering option whose construct string differs (``f"agreement@{depth}"``) and
    whose statistics are not validated.
    """

    def __init__(self, depth: int = 1) -> None:
        if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth < 1:
            raise ValueError(f"depth must be a positive integer, got {depth!r}")
        self.depth = int(depth)

    @property
    def construct(self) -> str:
        return f"agreement@{self.depth}"

    def relevant(self, corpus_ids: Sequence[str], query_ids: Sequence[str],
                 base_scores: np.ndarray | None) -> list[np.ndarray]:
        if base_scores is None:
            raise ValueError("the ranking reference needs the baseline score matrix")
        scores = np.asarray(base_scores)
        if scores.ndim != 2:
            raise ValueError(f"base_scores must be [n_queries, n_docs], got shape {scores.shape}")
        if scores.shape[0] != len(query_ids) or scores.shape[1] != len(corpus_ids):
            raise ValueError(f"base_scores has shape {scores.shape} for {len(query_ids)} "
                             f"queries and {len(corpus_ids)} documents")
        if self.depth > scores.shape[1]:
            raise ValueError(f"depth {self.depth} exceeds the corpus size {scores.shape[1]}")
        return relevant_from_baseline(scores, depth=self.depth)

    def identity(self) -> dict[str, Any]:
        return {"construct": self.construct, "depth": self.depth}

    def __repr__(self) -> str:
        return f"RankingReference(depth={self.depth})"
