"""Evaluation: per-query metrics, references and the evidence a selection rule sees.

Two references are never merged: ``LabelReference`` (relevance labels) and
``RankingReference`` (agreement with the uncompressed ranking, including its mistakes;
not retrieval quality).

``collect_evidence`` and ``compress_selected`` live in ``core/evaluate/collect.py`` and are
exported lazily (they depend on the candidate space, which in turn uses ``Evidence``).
"""

from __future__ import annotations

import importlib
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


_LAZY = ("collect_evidence", "compress_selected")


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        try:
            collect = importlib.import_module(f"{__name__}.collect")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.collect":
                raise
            raise AttributeError(f"module {__name__!r} has no attribute {name!r} "
                                 "(core/evaluate/collect.py does not exist yet)") from exc
        return getattr(collect, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
