"""Pipelines and their identity.

``Pipeline``, ``CompressedCorpus``, ``save_compressed`` and ``load_compressed`` are the
library's own objects from :mod:`optivision.compose`, re-exported unchanged. This
module adds the identity of a pipeline *specification*: the stage names and
parameters that ``Pipeline.to_dict()`` returns.

A pipeline identity says nothing about fitted state (a PCA basis, a codec's mean) and
nothing about how well the pipeline works on any encoder; two pipelines with the same
specification fitted on different corpora have the same identity.
"""

from __future__ import annotations

import json
from typing import Any

from ..compose import CompressedCorpus, Pipeline, load_compressed, save_compressed
from .identity import canonical_json, digest

__all__ = [
    "CompressedCorpus",
    "Pipeline",
    "load_compressed",
    "pipeline_identity",
    "pipeline_spec",
    "save_compressed",
]


def pipeline_spec(pipeline: Pipeline) -> dict[str, Any]:
    """A deep, JSON-native copy of ``pipeline.to_dict()``.

    NumPy scalars become Python numbers and tuples become lists; mutating the result
    never touches the pipeline. ``Pipeline.from_dict`` accepts it. A stage parameter
    with no JSON form (NaN, a set, bytes) raises ``ValueError`` or ``TypeError``.
    """
    return json.loads(canonical_json(pipeline.to_dict()))


def pipeline_identity(pipeline: Pipeline) -> str:
    """SHA-256 of the canonical JSON of ``pipeline.to_dict()``.

    Key order, whitespace and fitted state do not matter; stage order and every
    parameter do. The canonical form is the one the research used for its candidate
    list (sorted keys, compact separators), applied to one pipeline.
    """
    return digest(pipeline.to_dict())
