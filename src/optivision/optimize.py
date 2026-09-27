"""``optimize()``: the one-call entry point.

    from optivision import optimize

    result = optimize(doc_vectors, queries=query_vectors, quality_target=0.97, metric="ndcg@5")
    result.compressed           # the whole corpus, compressed with the chosen pipeline
    result.compression_ratio    # vs the float32 original
    result.quality_retention    # measured on held-out queries, not the ones used to choose
    result.memory_saving        # fraction of float32 bytes saved

Quality is a property of retrieval, so ``optimize`` needs sample queries: a few
hundred real queries is typical. Labels are optional -- without them the float
index's own ranking is the reference (see :mod:`optivision.calibration`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .calibration import CalibrationResult, calibrate
from .compose import CompressedCorpus, Pipeline
from .representation import as_corpus


@dataclass
class OptimizationResult:
    pipeline: Pipeline
    compressed: CompressedCorpus
    calibration: CalibrationResult

    @property
    def target_met(self) -> bool | None:
        """Whether the held-out retention reached the target (None if nothing was selected)."""
        return self.calibration.met_target_on_holdout

    @property
    def compression_ratio(self) -> float:
        return float(self.compressed.report()["compression_vs_float32"])

    @property
    def quality_retention(self) -> float | None:
        h = self.calibration.holdout
        return None if h is None else float(h["retention"])

    @property
    def memory_saving(self) -> float:
        before = float(self.compressed.stats.get("float32_bytes_before", 0))
        return 1.0 - self.compressed.nbytes / before if before else 0.0

    def report(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline.to_dict(),
            "label": self.pipeline.label(),
            "compression_ratio": self.compression_ratio,
            "memory_saving": self.memory_saving,
            "quality_retention_holdout": self.quality_retention,
            "target_met_on_holdout": self.target_met,
            "compressed": self.compressed.report(),
            "calibration": self.calibration.summary(),
        }


def optimize(
    vectors: Any,
    queries: Any = None,
    qrels: dict[str, Any] | None = None,
    quality_target: float = 0.97,
    metric: str = "ndcg@5",
    search_space: dict[str, Sequence[Pipeline]] | None = None,
    **calibrate_kwargs: Any,
) -> OptimizationResult:
    """Find the smallest pipeline meeting ``quality_target`` and compress with it.

    If no candidate meets the target on the calibration queries, the corpus is
    returned uncompressed (float32) and ``result.target_met`` is None -- a loud
    "could not", rather than a quietly worse index.
    """
    corpus = as_corpus(vectors)
    if queries is None:
        raise ValueError(
            "optimize() needs sample queries to measure retrieval quality; pass "
            "queries=[...] (a few hundred real ones is typical, labels optional)"
        )
    query_corpus = as_corpus(queries)
    cal = calibrate(corpus, query_corpus, qrels, quality_target=quality_target, metric=metric,
                    search_space=search_space, **calibrate_kwargs)
    pipeline = cal.pipeline or Pipeline()
    compressed = pipeline.compress(corpus, fit=True)
    return OptimizationResult(pipeline=pipeline, compressed=compressed, calibration=cal)
