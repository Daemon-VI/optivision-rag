"""Exact late-interaction (MaxSim) scoring.

The scorers are the library's own, from :mod:`optivision.scoring`, re-exported
unchanged: ``maxsim_matrix`` (every query against every document, blocked to a memory
budget), ``maxsim_pair`` (the textbook definition, one pair), ``rank``,
``segment_reduce`` and the ``VectorStore`` protocol (``offsets`` plus
``decode_rows``), which both ``MultiVectorCorpus`` and ``CompressedCorpus`` satisfy.

What a score means:

* Scoring is exact over the *stored* representation. For a compressed store the
  vectors are decoded block by block first, so the score is that of the decoded
  vectors. With a lossy codec that is not the score of the float index they came
  from, and a ranking from it need not be the ranking the uncompressed vectors give.
* Scores are float32, whatever the dtype of the stored vectors.
* ``rank`` is a stable sort: documents with equal scores rank by ascending store
  position.
* An empty document scores ``-inf``; an empty query scores 0 against every other
  document.

``maxsim_matrix`` reads the ``OPTIVISION_SCORE_DEVICE`` environment variable on every
call (existing behaviour of the pinned module, kept as it is).
"""

from __future__ import annotations

from ..scoring import (
    DEFAULT_MAX_BLOCK_BYTES,
    VectorStore,
    maxsim_matrix,
    maxsim_pair,
    rank,
    segment_reduce,
)

__all__ = [
    "DEFAULT_MAX_BLOCK_BYTES",
    "VectorStore",
    "maxsim_matrix",
    "maxsim_pair",
    "rank",
    "segment_reduce",
]
