"""Collect per-query evidence for a frozen candidate space, and deploy the one that was chosen.

``collect_evidence`` scores every candidate of a :class:`CandidateSpace`, in list order, on
every query, with the library's own functions: ``calibration.score_candidate`` (compress the
corpus, project the queries, exact MaxSim) with one shared transform cache, as ``calibrate``
does, and ``evaluation.per_query_metrics`` for the metric. It records what the selection rule
is allowed to see: the per-query metric of the uncompressed index and of every candidate,
the library's stored size and vectors per document of every candidate, and the digest of
every stage's fitted state, so that the index deployed later can be compared with the one that
was evaluated.

Rules that keep the rule's assumptions intact:

* Stages are fitted on documents only. ``score_candidate`` never passes queries to ``fit``,
  and a stage whose declared traits say ``fit_uses_queries`` (``PCAProjector(basis="joint")``,
  or a stage with no declared traits) is refused up front: assumption A3 of the rule's statement
  needs that no fitted stage uses the calibration queries.
* A query with no relevant document has a NaN baseline metric. It is removed for the baseline
  and for every candidate together (the rule looks only at the reference, never at a candidate)
  and listed in ``dropped_queries``. A NaN on a kept query for any candidate raises.
* A candidate that cannot be evaluated raises :class:`CandidateEvaluationError`. It is never
  skipped and never replaced: dropping or substituting a candidate after looking at the
  calibration queries is outside the rule's statement.

Nothing here draws a random number, reads the clock or reads an environment variable. The
pinned ``optivision.scoring.maxsim_matrix`` does read ``OPTIVISION_SCORE_DEVICE`` (a GPU product
agrees with the NumPy one only to float32 rounding); leave it unset when the evidence has to be
reproduced. The timings that ``score_candidate`` records are ignored.

Dtypes: the library returns float32 score matrices; only metric values leave this module, and
they are stored as float64 (``per_query_metrics`` already returns float64; the conversion is
made explicit). ``bytes_per_doc`` and ``vectors_per_doc`` are the Python floats of the
library's ``report()``, stored as float64.

With the ``agreement@1`` reference the metric measures agreement with the uncompressed
ranking, including its mistakes. It is not retrieval quality.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from optivision.calibration import score_candidate
from optivision.compose import CompressedCorpus, Pipeline
from optivision.core.evaluate.evidence import Evidence
from optivision.core.evaluate.metrics import parse_metric, per_query_metrics
from optivision.core.evaluate.reference import Reference
from optivision.core.selection.spaces import CandidateSpace, CandidateSpec, thaw_json
from optivision.core.stages.state import pipeline_fitted_identity
from optivision.core.stages.traits import BasisInvariance, traits_of
from optivision.core.vectors import ContractError, validate_corpus
from optivision.scoring import maxsim_matrix

__all__ = [
    "CandidateEvaluationError",
    "FittedStateMismatchError",
    "collect_evidence",
    "compress_selected",
]


class CandidateEvaluationError(RuntimeError):
    """A candidate of the space could not be evaluated. The original error is ``__cause__``.

    There is no skipping and no substitution: the rule's statement covers a list that was
    fixed in advance and evaluated whole.
    """


class FittedStateMismatchError(ValueError):
    """Fitting the chosen pipeline again did not reproduce the fitted state that was evaluated."""


def _check_fitted_on_documents_only(pipeline: Pipeline, where: str) -> None:
    for i, stage in enumerate(pipeline.stages):
        traits = traits_of(stage)
        if not traits.fit_uses_queries:
            continue
        if traits.basis_invariance is BasisInvariance.UNKNOWN:
            why = (f"no traits are declared for the stage {stage.name!r}, so it cannot be shown "
                   "to be fitted on documents only; if it is, declare that with "
                   "optivision.core.stages.register_traits(...) (fit_uses_queries=False)")
        else:
            why = (f"the stage {stage.name!r} declares fit_uses_queries=True (for example "
                   "PCAProjector(basis='joint') also reads the queries)")
        raise ValueError(
            f"{where}: stage {i} cannot be used here: {why}. The selection rule's statement "
            "assumes that no fitted stage uses the calibration queries, so that every stage is "
            "fitted on documents only (assumption A3), and the evidence collector never passes "
            "queries to a stage's fit."
        )


def _checked_relevance(relevant: Any, n_queries: int, n_docs: int) -> list[np.ndarray]:
    if len(relevant) != n_queries:
        raise ValueError(f"the reference returned {len(relevant)} relevant sets for "
                         f"{n_queries} queries")
    out = []
    for i, rel in enumerate(relevant):
        arr = np.asarray(rel)
        if arr.ndim != 1 or (arr.size and arr.dtype.kind not in "iu"):
            raise ValueError(f"the relevant set of query {i} must be a 1-D integer array of "
                             f"document indices, got dtype {arr.dtype} shape {arr.shape}")
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= n_docs):
            raise ValueError(f"the relevant set of query {i} holds a document index outside "
                             f"[0, {n_docs})")
        out.append(arr)
    return out


def _check_inputs(corpus: Any, queries: Any) -> None:
    validate_corpus(corpus, name="corpus")
    validate_corpus(queries, name="queries")
    if len(corpus) == 0 or corpus.num_vectors == 0:
        raise ContractError("corpus: there are no vectors to compress")
    if len(queries) == 0:
        raise ContractError("queries: there are no queries")
    if corpus.dimension != queries.dimension:
        raise ContractError(f"queries have dimension {queries.dimension}, the corpus has "
                            f"{corpus.dimension}")


def collect_evidence(
    corpus: Any,
    queries: Any,
    space: CandidateSpace,
    reference: Reference,
    *,
    metric: str = "ndcg@5",
    progress: Callable[[int, int, str], None] | None = None,
) -> Evidence:
    """Per-query evidence of every candidate of ``space`` against ``reference``.

    Args:
        corpus, queries: ``MultiVectorCorpus`` objects, validated with ``validate_corpus``
            (finite values, unique ids, equal dimension). Neither is modified.
        space: the frozen candidate list. Candidates are evaluated in list order, each from a
            fresh pipeline; the transform cache is shared between them.
        reference: ``LabelReference`` (qrels) or ``RankingReference`` (agreement with the
            uncompressed index's own top documents, which is not retrieval quality). It decides
            which documents are relevant to which query.
        metric: ``ndcg@k``, ``recall@k``, ``mrr@k`` or ``hit@k``. Only nDCG@k is validated by
            the registered experiments; the others are an engineering option. The name is stored
            in lower case.
        progress: optional ``progress(position, n_candidates, family)``, called after each
            candidate is evaluated; ``position`` counts from 1, so the last call has
            ``position == n_candidates``.

    The queries of the evidence are the queries that have a relevant document, in the order
    given; the others are listed in ``Evidence.dropped_queries``. Raises ``ValueError`` (a
    ``ContractError`` for invalid vectors) on invalid input, when no query is labelled, or when
    a candidate contains a stage fitted on queries; :class:`CandidateEvaluationError` when a
    candidate cannot be evaluated or has a NaN metric on a labelled query.
    """
    if not isinstance(space, CandidateSpace):
        raise TypeError(f"space must be a CandidateSpace, got {type(space).__name__}")
    if not isinstance(reference, Reference):
        raise TypeError("reference must provide construct, relevant() and identity() "
                        "(LabelReference or RankingReference)")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    kind, cutoff = parse_metric(metric)
    metric = f"{kind}@{cutoff}"
    _check_inputs(corpus, queries)

    pipelines = space.pipelines()
    for position, pipe in enumerate(pipelines):
        _check_fitted_on_documents_only(pipe, f"candidate {position} ({space[position].family})")

    base_scores = maxsim_matrix(queries, corpus)
    relevant = _checked_relevance(
        reference.relevant(corpus.ids, queries.ids, base_scores), len(queries), len(corpus))
    base_pq = np.asarray(per_query_metrics(base_scores, relevant, [metric])[metric],
                         dtype=np.float64)
    del base_scores
    keep = np.fromiter((rel.size > 0 for rel in relevant), dtype=bool, count=len(relevant))
    if not np.array_equal(np.isnan(base_pq), ~keep):
        raise RuntimeError("the baseline metric is NaN on queries that have a relevant "
                           "document, or finite on queries that have none")
    if not keep.any():
        raise ValueError("no query has a relevant document under this reference: there is "
                         "nothing to test (with labels, check that their query ids and document "
                         "ids match the ids of the queries and vectors; a plain list gets the "
                         "ids '0', '1', ...)")

    cache: dict = {}
    rows: list[np.ndarray] = []
    sizes: list[float] = []
    vectors: list[float] = []
    fitted: list[tuple[str | None, ...]] = []
    for position, (spec, pipe) in enumerate(zip(space.candidates, pipelines, strict=True)):
        label = pipe.label()
        try:
            scored = score_candidate(pipe, corpus, queries, family=spec.family, cache=cache)
            per_query = np.asarray(per_query_metrics(scored.scores, relevant, [metric])[metric],
                                   dtype=np.float64)
            digests = tuple(pipeline_fitted_identity(pipe))
            size = float(scored.bytes_per_doc)
            n_vectors = float(scored.report["vectors_per_doc"])
        except Exception as exc:
            raise CandidateEvaluationError(
                f"candidate {position} (family {spec.family!r}, {label}) could not be "
                f"evaluated: {type(exc).__name__}: {exc}. A candidate is never skipped or "
                "replaced; fix the candidate or the space and collect again."
            ) from exc
        if np.isnan(per_query[keep]).any():
            raise CandidateEvaluationError(
                f"candidate {position} (family {spec.family!r}, {label}) has a NaN metric on a "
                "query that has a relevant document")
        rows.append(per_query[keep])
        sizes.append(size)
        vectors.append(n_vectors)
        fitted.append(digests)
        if progress is not None:
            progress(position + 1, len(space), spec.family)

    ids = queries.ids
    return Evidence(
        metric=metric,
        construct=reference.construct,
        baseline=base_pq[keep],
        candidates=np.stack(rows),
        query_ids=tuple(ids[i] for i in np.flatnonzero(keep)),
        bytes_per_doc=np.asarray(sizes, dtype=np.float64),
        vectors_per_doc=np.asarray(vectors, dtype=np.float64),
        space_identity=space.identity,
        dropped_queries=tuple(ids[i] for i in np.flatnonzero(~keep)),
        fitted=tuple(fitted),
    )


def _pipeline_from(spec: CandidateSpec | Mapping[str, Any] | Pipeline) -> Pipeline:
    if isinstance(spec, CandidateSpec):
        return spec.to_pipeline()
    if isinstance(spec, Pipeline):
        return Pipeline.from_dict(spec.to_dict())  # a fresh, unfitted copy; spec is untouched
    if isinstance(spec, Mapping):
        return Pipeline.from_dict(thaw_json(spec))
    raise TypeError("spec must be a CandidateSpec, a Pipeline.to_dict() mapping or a Pipeline, "
                    f"got {type(spec).__name__}")


def compress_selected(
    corpus: Any,
    spec: CandidateSpec | Mapping[str, Any] | Pipeline,
    *,
    expected_fitted: tuple[str | None, ...] | list[str | None] | None = None,
) -> tuple[Pipeline, CompressedCorpus]:
    """Fit and apply the chosen pipeline to ``corpus``: ``(fitted pipeline, compressed corpus)``.

    ``spec`` is a ``CandidateSpec`` (for example ``Decision.selected``), a
    ``Pipeline.to_dict()`` mapping, or a ``Pipeline`` (rebuilt from its description, so the
    object passed is neither fitted nor changed). Stages are fitted on ``corpus`` only.

    ``expected_fitted`` is the tuple of per-stage fitted-state digests recorded in the evidence
    for this candidate (``Evidence.fitted[position]``). When given, the new fit must reproduce
    it exactly, otherwise :class:`FittedStateMismatchError` is raised: the index that is
    deployed has to be the one that was evaluated. A mismatch means a different corpus, a
    different pipeline, or a numerical library that does not reproduce the fit bit for bit.
    """
    validate_corpus(corpus, name="corpus")
    pipeline = _pipeline_from(spec)
    _check_fitted_on_documents_only(pipeline, "the selected pipeline")
    compressed = pipeline.compress(corpus, fit=True)
    if expected_fitted is not None:
        want = tuple(expected_fitted)
        got = tuple(pipeline_fitted_identity(pipeline))
        if len(want) != len(got):
            raise FittedStateMismatchError(
                f"the evidence recorded {len(want)} stage digest(s), the pipeline has "
                f"{len(got)} stage(s)")
        bad = [i for i, (a, b) in enumerate(zip(want, got, strict=True)) if a != b]
        if bad:
            detail = "; ".join(
                f"stage {i} ({pipeline.stages[i].name}): evaluated {_short(want[i])}, "
                f"deployed {_short(got[i])}" for i in bad)
            raise FittedStateMismatchError(
                "the fitted state of the pipeline fitted now differs from the one evaluated "
                f"({detail}). The index would not be the one that was evaluated: check that this "
                "is the same corpus and the same pipeline, and that the numerical libraries "
                "reproduce the fit")
    return pipeline, compressed


def _short(digest: str | None) -> str:
    return "no state" if digest is None else digest[:12] + "..."
