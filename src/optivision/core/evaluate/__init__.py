"""Evaluation: per-query metrics, references and the evidence a selection rule sees.

Two references are never merged: ``LabelReference`` (relevance labels) and
``RankingReference`` (agreement with the uncompressed ranking, including its mistakes;
not retrieval quality).

``collect_evidence`` lives in ``core/evaluate/collect.py`` and is exported lazily (it
depends on the candidate space, which in turn uses ``Evidence``).
"""

from __future__ import annotations

from typing import Any

from .evidence import Evidence, EvidenceFormatError
from .metrics import (
    DEFAULT_METRICS,
    mean_metrics,
    metrics_from_ranking,
    parse_metric,
    per_query_metrics,
)
from .reference import LabelReference, RankingReference, Reference

__all__ = [
    "DEFAULT_METRICS",
    "Evidence",
    "EvidenceFormatError",
    "LabelReference",
    "RankingReference",
    "Reference",
    "mean_metrics",
    "metrics_from_ranking",
    "parse_metric",
    "per_query_metrics",
]


def __getattr__(name: str) -> Any:
    if name == "collect_evidence":
        try:
            from .collect import collect_evidence
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.collect":
                raise
            raise AttributeError(f"module {__name__!r} has no attribute {name!r} "
                                 "(core/evaluate/collect.py does not exist yet)") from exc
        return collect_evidence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
