"""Token reducers that need nothing but the vectors: the same classes as ``optivision.stages``.

``RedundancyPruner`` lives in ``optivision.stages.prune`` beside the page-specific
``SpatialPruner``; it is taken from the ``optivision.stages`` namespace so that core never
names the page-specific module or class.
"""

from __future__ import annotations

from optivision.stages import AdaptiveMerge, HierarchicalMerge, RandomPruner, RedundancyPruner

__all__ = ["AdaptiveMerge", "HierarchicalMerge", "RandomPruner", "RedundancyPruner"]
