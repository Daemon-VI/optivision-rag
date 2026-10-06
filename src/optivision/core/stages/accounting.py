"""Byte accounting of a compressed corpus, with the library's size key kept separate.

``CompressedCorpus.nbytes`` is the number the library ranks candidates by (codes plus the
quantizer's own shared state, divided by the number of documents in ``report()``). It does
not include everything a deployed index needs: the projection matrices that queries must go
through, the document offsets. This module reports that figure unchanged as
``selection_bytes_per_doc`` and the omitted parts beside it, so a reader can see both while
the selection rule keeps using the size key.

What is counted where, per stage (read from the stage code):

* ``codec_overhead_bytes``: ``quantizer.overhead_bytes()``. For ``binary``/``int4``/``int8``
  the fitted mean and per-dimension ranges; for ``lloyd2`` the mean and one scalar. The
  Lloyd-2 rotation is regenerated from ``(dim, seed)`` and is not stored, so it costs no
  stored bytes; decoding builds a ``d x d`` float32 matrix in memory.
* ``fitted_state_bytes``: arrays of the token and dimension stages that queries or decoding
  read after compression: PCA ``components`` (queries are projected with them), the
  random-projection matrix, the rotation matrix.
* ``append_state_bytes``: arrays read only when more documents are compressed with the
  fitted pipeline, never when the stored codes are scored: the PCA ``mean`` (queries are not
  centred; documents are) and the ``AdaptiveMerge`` mean. Not part of ``total_bytes_per_doc``.

Not counted anywhere: document ids, the stats header, the pipeline description, a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from optivision.stages.base import Quantizer, Stage

from .state import fitted_state

if TYPE_CHECKING:
    from optivision.compose import CompressedCorpus, Pipeline

__all__ = ["ByteAccounting", "byte_accounting"]

#: State arrays that matter only when further documents are compressed, per stage name.
_APPEND_ONLY = {"pca": frozenset({"mean"}), "adaptive_merge": frozenset({"mean"})}


@dataclass(frozen=True)
class ByteAccounting:
    """Stored bytes of one compressed corpus. Field meanings are in the module docstring.

    ``selection_bytes_per_doc`` is the size key of the library:
    ``(codes + codec overhead) / n_docs``, with ``n_docs`` floored at 1 as in
    ``CompressedCorpus.report``. ``total_bytes_per_doc`` adds the serving-time fitted state
    and the offsets. ``float32_bytes_before`` is the float32 size of the input corpus.
    """

    code_bytes: int
    codec_overhead_bytes: int
    fitted_state_bytes: int
    offsets_bytes: int
    float32_bytes_before: int
    n_docs: int
    append_state_bytes: int = 0

    @property
    def selection_bytes_per_doc(self) -> float:
        return (self.code_bytes + self.codec_overhead_bytes) / max(1, self.n_docs)

    @property
    def total_bytes_per_doc(self) -> float:
        total = (
            self.code_bytes
            + self.codec_overhead_bytes
            + self.fitted_state_bytes
            + self.offsets_bytes
        )
        return total / max(1, self.n_docs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_bytes": self.code_bytes,
            "codec_overhead_bytes": self.codec_overhead_bytes,
            "fitted_state_bytes": self.fitted_state_bytes,
            "append_state_bytes": self.append_state_bytes,
            "offsets_bytes": self.offsets_bytes,
            "float32_bytes_before": self.float32_bytes_before,
            "n_docs": self.n_docs,
            "selection_bytes_per_doc": self.selection_bytes_per_doc,
            "total_bytes_per_doc": self.total_bytes_per_doc,
        }


def _state_bytes(stage: Stage) -> tuple[int, int]:
    """``(serving, append-only)`` bytes of the arrays in a stage's fitted state."""
    inner = getattr(stage, "inner", None) if stage.name == "project" else None
    append_keys = _APPEND_ONLY.get(inner.name if inner is not None else stage.name, frozenset())
    serving = append = 0
    for key, value in fitted_state(stage).items():
        if isinstance(value, np.ndarray):
            if key in append_keys:
                append += int(value.nbytes)
            else:
                serving += int(value.nbytes)
    return serving, append


def byte_accounting(compressed: CompressedCorpus, pipeline: Pipeline) -> ByteAccounting:
    """Account for ``compressed``, which ``pipeline`` produced and which must still be fitted.

    Raises ``ValueError`` when the corpus records a different pipeline description than
    ``pipeline.to_dict()``, and ``RuntimeError`` when a stage that holds state is unfitted.
    Third-party stages count all of their array state as serving state.
    """
    recorded = compressed.pipeline
    if recorded is not None and recorded != pipeline.to_dict():
        raise ValueError(
            "the compressed corpus was not produced by this pipeline (descriptions differ)"
        )
    code_bytes = int(compressed.codes.nbytes)
    serving = append = 0
    for stage in pipeline.stages:
        if isinstance(stage, Quantizer):
            continue  # a quantizer's state is already inside codec_overhead_bytes
        s, a = _state_bytes(stage)
        serving, append = serving + s, append + a
    return ByteAccounting(
        code_bytes=code_bytes,
        codec_overhead_bytes=int(compressed.nbytes) - code_bytes,
        fitted_state_bytes=serving,
        offsets_bytes=int(compressed.offsets.nbytes),
        float32_bytes_before=int(compressed.stats.get("float32_bytes_before", 0)),
        n_docs=len(compressed),
        append_state_bytes=append,
    )
