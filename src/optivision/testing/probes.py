"""A diagnostic for one pipeline's dependence on the coordinate basis, on one data set.

``basis_probe`` applies changes of basis that leave the float scores unchanged
(``signed_permutation`` and ``rotation`` from ``optivision.testing.transforms``) to the
documents and the queries, refits the pipeline from its description on each transformed
copy, and reports how much its compressed scores and its top-k move.

Engineering capability; not a research result. It describes what happened for the seeds and
the data it was given: a small change does not show that a pipeline is invariant, and a large
one does not show that it loses retrieval quality (agreement between two rankings of the same
pipeline is not retrieval quality). The registered experiment on this question, its
protocol and its results are in ``reports/research/X_RESULTS.md``; nothing here repeats or
extends them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..calibration import score_candidate
from ..core.pipeline import Pipeline, pipeline_spec
from ..core.similarity import maxsim_matrix, rank
from ..core.vectors import MultiVectorCorpus
from .transforms import apply_basis, random_orthogonal, signed_permutation

__all__ = ["basis_probe"]

_TRANSFORMS = {"signed_permutation": signed_permutation, "rotation": random_orthogonal}


def _scores(
    spec: dict[str, Any], corpus: MultiVectorCorpus, queries: MultiVectorCorpus
) -> np.ndarray:
    """Scores of a freshly built, freshly fitted pipeline, by the library's own path."""
    return score_candidate(Pipeline.from_dict(spec), corpus, queries).scores


def _largest_change(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute difference over entries that are finite in both (``inf`` if the
    pattern of ``-inf`` entries, the empty documents, differs)."""
    both = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    return float(np.max(np.abs(a[both].astype(np.float64) - b[both]), initial=0.0))


def _overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Mean over queries of the fraction of the top-``k`` documents the two rankings share."""
    top_a, top_b = rank(a, k), rank(b, k)
    shared = [len(set(x.tolist()) & set(y.tolist())) for x, y in zip(top_a, top_b, strict=True)]
    return float(np.mean(shared) / top_a.shape[1])


def basis_probe(
    pipeline: Pipeline,
    corpus: MultiVectorCorpus,
    queries: MultiVectorCorpus,
    *,
    transforms: tuple[str, ...] = ("signed_permutation", "rotation"),
    seeds: tuple[int, ...] = (0, 1, 2),
    k: int = 10,
    float_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """How much one pipeline's scores and top-``k`` move under changes of basis.

    For every transform and seed: the documents and the queries are multiplied by the same
    orthogonal matrix (``"signed_permutation"``: axes permuted and flipped; ``"rotation"``:
    a dense random orthogonal matrix); the pipeline is rebuilt from ``pipeline.to_dict()``,
    so it is fresh and refitted on the transformed documents; documents are compressed and
    scored with ``optivision.calibration.score_candidate``, the library's own path; and the
    result is compared with the same pipeline run on the vectors as given.

    Reported per run (and summarised per transform):

    ``max_relative_score_change``
        the largest absolute difference between the pipeline's scores in the two bases,
        divided by the span (maximum minus minimum) of its scores in the given basis;
    ``top_k_overlap``
        the mean over queries of the share of the top-``k`` documents that the two rankings
        have in common, in ``[0, 1]`` (``k`` is capped at the number of documents);
    ``float_baseline_max_relative_change`` and ``float_baseline_unchanged``
        the same comparison for the uncompressed float scores, divided by the largest
        absolute float score, against ``float_tolerance``. The float scores must not change
        beyond rounding: that is the check that the transform is a valid symmetry. When it
        fails, the other numbers describe a broken transform or very low precision data and
        should not be read (``"valid_symmetry"`` is then False).

    The pipeline is not modified. The documents and queries are used in the dtype they have.
    Cost: one compression and one scoring pass per transform and seed, plus one for the
    given basis. Returns plain Python values (JSON-serialisable).

    Engineering capability; not a research result. The registered experiment on this
    question, with its protocol and results, is in ``reports/research/X_RESULTS.md``.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"pipeline must be a Pipeline, got {type(pipeline).__name__}")
    for what, obj in (("corpus", corpus), ("queries", queries)):
        if not isinstance(obj, MultiVectorCorpus):
            raise TypeError(f"{what} must be a MultiVectorCorpus, got {type(obj).__name__}")
    if corpus.dimension != queries.dimension:
        raise ValueError(f"corpus has dimension {corpus.dimension}, queries {queries.dimension}")
    if len(corpus) == 0 or len(queries) == 0 or corpus.num_vectors == 0:
        raise ValueError("basis_probe needs documents with vectors and at least one query")
    unknown = [t for t in transforms if t not in _TRANSFORMS]
    if unknown or not transforms:
        raise ValueError(f"transforms must be a non-empty subset of {sorted(_TRANSFORMS)}, "
                         f"got {tuple(transforms)!r}")
    if not seeds or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0
                        for s in seeds):
        raise ValueError(f"seeds must be a non-empty tuple of non-negative ints, got {seeds!r}")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if not np.isfinite(float_tolerance) or float_tolerance < 0:
        raise ValueError("float_tolerance must be finite and not negative")

    k_eff = min(int(k), len(corpus))
    spec = pipeline.to_dict()
    description = pipeline_spec(pipeline)  # JSON-native for the report; fails before any work
    native = _scores(spec, corpus, queries)
    float_native = maxsim_matrix(queries, corpus)
    finite = native[np.isfinite(native)]
    span = float(finite.max() - finite.min()) if finite.size else 0.0
    if not span > 0.0:
        raise ValueError("the pipeline's scores have zero span on this data, so a relative "
                         "change is undefined; use more documents or queries")
    float_scale = float(np.max(np.abs(float_native[np.isfinite(float_native)])))
    if not float_scale > 0.0:
        raise ValueError("the float scores are all zero, so a relative change is undefined")

    runs: list[dict[str, Any]] = []
    for transform in transforms:
        for seed in seeds:
            matrix = _TRANSFORMS[transform](corpus.dimension, int(seed))
            moved_corpus = apply_basis(corpus, matrix)
            moved_queries = apply_basis(queries, matrix)
            moved = _scores(spec, moved_corpus, moved_queries)
            float_moved = maxsim_matrix(moved_queries, moved_corpus)
            relative = _largest_change(float_native, float_moved) / float_scale
            runs.append({
                "transform": transform,
                "seed": int(seed),
                "max_relative_score_change": _largest_change(native, moved) / span,
                "top_k_overlap": _overlap(native, moved, k_eff),
                "float_baseline_max_relative_change": relative,
                "float_baseline_unchanged": bool(relative <= float_tolerance),
            })

    summary = {}
    for transform in transforms:
        mine = [r for r in runs if r["transform"] == transform]
        summary[transform] = {
            "max_relative_score_change": max(r["max_relative_score_change"] for r in mine),
            "mean_top_k_overlap": float(np.mean([r["top_k_overlap"] for r in mine])),
            "min_top_k_overlap": min(r["top_k_overlap"] for r in mine),
            "float_baseline_unchanged": all(r["float_baseline_unchanged"] for r in mine),
        }
    valid = all(r["float_baseline_unchanged"] for r in runs)
    result: dict[str, Any] = {
        "pipeline": description,
        "label": pipeline.label(),
        "n_documents": len(corpus),
        "n_queries": len(queries),
        "dimension": corpus.dimension,
        "dtype": str(corpus.dtype),
        "k": k_eff,
        "score_span": span,
        "float_tolerance": float(float_tolerance),
        "runs": runs,
        "summary": summary,
        "valid_symmetry": valid,
        "declared_basis_invariance": _declared(pipeline),
    }
    if not valid:
        result["note"] = ("the float scores changed beyond the tolerance under at least one "
                          "transform; the pipeline numbers describe a broken transform or very "
                          "low precision data and should not be interpreted")
    return result


def _declared(pipeline: Pipeline) -> str | None:
    """The pipeline's declared basis invariance (``optivision.core.stages``), if available."""
    try:
        from ..core.stages import pipeline_traits
    except ImportError:  # the traits module is not part of this installation yet
        return None
    return str(pipeline_traits(pipeline).basis_invariance.value)
