"""Conformance checks for the code people add: adapters and stages.

``check_adapter`` and ``check_stage`` return a list of problems found, in plain sentences.
An empty list means every check passed. A problem in the object under test is reported, not
raised (an exception it raises while being checked is reported too); only a call that cannot
be checked at all (no documents to encode) raises.

The checks look at the contract, not at quality. An adapter that passes returns well-formed
vectors; it is not thereby a good encoder. A stage that passes is deterministic, describes
itself, leaves its input alone and has declared traits; it is not thereby a good codec, and
how well any pipeline compresses a model's vectors has to be measured on those vectors.

Numpy-only; importable without pytest.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..compose import Pipeline
from ..core.identity import canonical_json
from ..core.vectors import (
    ContractError,
    EncoderIdentity,
    MultiVectorCorpus,
    corpus_identity,
    validate_corpus,
)
from ..scoring import maxsim_matrix
from ..stages.base import STAGES, DimensionReducer, Quantizer, Stage, TokenReducer, stage_from_dict

__all__ = ["check_adapter", "check_stage"]

#: aligned arrays compared by the repeat-call check
_ALIGNED = ("protected", "positions", "weights", "importance")


# ------------------------------------------------------------------ adapters


def _row_norms(vectors: np.ndarray) -> np.ndarray:
    """L2 norms of the rows, accumulated in float64, in blocks (no corpus-sized temporary)."""
    out = np.empty(vectors.shape[0])
    step = max(1, (1 << 22) // max(1, vectors.shape[1]))
    for lo in range(0, vectors.shape[0], step):
        block = vectors[lo : lo + step]
        out[lo : lo + step] = np.sqrt(np.einsum("ij,ij->i", block, block, dtype=np.float64))
    return out


def _equal(a: Any, b: Any) -> bool:
    """Same shape, same dtype, same values (NaN equals NaN)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f"
                else np.array_equal(a, b))


def _same_corpus(a: MultiVectorCorpus, b: MultiVectorCorpus, atol: float) -> str | None:
    """What differs between two corpora from the same call, or None."""
    if a.dtype != b.dtype:
        return f"dtype {a.dtype} then {b.dtype}"
    if a.vectors.shape != b.vectors.shape or not _equal(a.offsets, b.offsets):
        return "different numbers of vectors"
    if atol > 0:
        equal = bool(np.allclose(a.vectors, b.vectors, rtol=0.0, atol=atol, equal_nan=True))
    else:
        equal = _equal(a.vectors, b.vectors)
    if not equal:
        return "different vectors"
    for key in _ALIGNED:
        x, y = getattr(a, key), getattr(b, key)
        if (x is None) != (y is None) or (x is not None and not _equal(x, y)):
            return f"different {key}"
    return None


def _validate(corpus: MultiVectorCorpus, kind: str) -> list[str]:
    """``validate_corpus`` in three passes, so that ids and finiteness are reported on their
    own and a repeated id does not hide a NaN (or the reverse)."""
    try:
        validate_corpus(corpus, name=kind, require_finite=False, require_unique_ids=False)
    except ContractError as exc:
        return [str(exc)]
    problems = []
    for options in ({"require_finite": False}, {"require_unique_ids": False}):
        try:
            validate_corpus(corpus, name=kind, **options)
        except ContractError as exc:
            problems.append(str(exc))
    return problems


def _encode_checked(
    call: Callable[..., Any],
    method: str,
    kind: str,
    inputs: list[Any],
    atol: float,
    problems: list[str],
) -> MultiVectorCorpus | None:
    """Run one ``encode_*`` method twice and check what comes back; None if unusable."""
    try:
        corpus = call(inputs)
    except Exception as exc:  # noqa: BLE001 -- whatever the adapter raises is a finding
        problems.append(f"{method} raised {type(exc).__name__}: {exc}")
        return None
    if not isinstance(corpus, MultiVectorCorpus):
        problems.append(f"{method} returned a {type(corpus).__name__}, not a MultiVectorCorpus")
        return None
    if len(corpus) != len(inputs):
        problems.append(f"{method} returned {len(corpus)} documents for {len(inputs)} inputs")
    problems.extend(_validate(corpus, kind))
    v = corpus.vectors
    if isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype.kind == "f" and v.shape[0]:
        zero = int(np.sum(_row_norms(v) == 0))
        if zero:
            problems.append(
                f"{kind}: {zero} of {v.shape[0]} vectors are all zero; if they are padding "
                "rows, drop them (with the attention mask) before handing vectors over"
            )

    wanted = [f"{kind[0]}{i}" for i in range(len(inputs))]
    try:
        again = call(inputs, ids=wanted)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{method}(..., ids=...) raised {type(exc).__name__}: {exc}")
        return corpus
    if not isinstance(again, MultiVectorCorpus):
        problems.append(f"{method}(..., ids=...) returned a {type(again).__name__}")
        return corpus
    if list(again.ids) != wanted:
        problems.append(f"{method} does not honour ids=: asked for {wanted[:3]}, got "
                        f"{list(again.ids)[:3]}")
    difference = _same_corpus(corpus, again, atol)
    if difference is not None:
        problems.append(f"{method} is not deterministic: a repeat call gave {difference}")
    return corpus


def check_adapter(
    adapter: Any,
    documents: Sequence[Any],
    queries: Sequence[Any],
    *,
    atol: float = 0.0,
    norm_tolerance: float = 2e-3,
) -> list[str]:
    """Problems found in an encoder adapter, or ``[]`` if it conforms.

    ``documents`` and ``queries`` are a few real inputs of the kind the adapter takes
    (page images, strings, arrays); each ``encode_*`` method is called on them twice. The
    checks:

    * both methods exist and return a ``MultiVectorCorpus`` with one document per input;
    * ``validate_corpus`` passes: 2-D floating vectors, consistent offsets, unique ids,
      finite values (ids and finiteness are reported separately);
    * no vector is all zero (a likely padding row that should have been dropped);
    * ``ids=`` is honoured, and a repeat call gives identical vectors and aligned arrays
      (``atol`` allows a difference, for models that are not bit-reproducible);
    * documents and queries have the same dimension;
    * ``identity()`` is ``None`` or an ``EncoderIdentity`` whose ``dimension`` is that of the
      vectors and, if it says ``normalized=True``, whose vectors have unit norm to within
      ``norm_tolerance``; ``dimension``, if the adapter has it, agrees too.

    The inputs given to the adapter are not inspected. Passing says the output is well
    formed; it says nothing about how well any pipeline compresses these vectors.
    """
    docs_in, queries_in = list(documents), list(queries)
    if not docs_in or not queries_in:
        raise ValueError("check_adapter needs at least one document and one query to encode")
    problems: list[str] = []
    for method in ("encode_documents", "encode_queries"):
        if not callable(getattr(adapter, method, None)):
            problems.append(f"the adapter has no callable {method}()")
    if problems:
        return problems

    docs = _encode_checked(adapter.encode_documents, "encode_documents", "documents", docs_in,
                           atol, problems)
    qs = _encode_checked(adapter.encode_queries, "encode_queries", "queries", queries_in,
                         atol, problems)
    if docs is not None and qs is not None and docs.dimension != qs.dimension:
        problems.append(f"documents have dimension {docs.dimension} but queries have "
                        f"{qs.dimension}")

    try:
        identity = adapter.identity() if callable(getattr(adapter, "identity", None)) else None
    except Exception as exc:  # noqa: BLE001
        problems.append(f"identity() raised {type(exc).__name__}: {exc}")
        identity = None
    if identity is not None and not isinstance(identity, EncoderIdentity):
        problems.append(f"identity() returned a {type(identity).__name__}, not an "
                        "EncoderIdentity or None")
        identity = None
    declared = [("identity()", identity.dimension)] if identity is not None else []
    try:
        dimension = getattr(adapter, "dimension", None)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"dimension raised {type(exc).__name__}: {exc}")
        dimension = None
    if dimension is not None:
        declared.append(("dimension", dimension))
    for what, value in declared:
        for kind, corpus in (("documents", docs), ("queries", qs)):
            if corpus is not None and value != corpus.dimension:
                problems.append(f"{what} says {value} but the {kind} have dimension "
                                f"{corpus.dimension}")
    if identity is not None and identity.normalized is True:
        for kind, corpus in (("documents", docs), ("queries", qs)):
            if corpus is None or not corpus.num_vectors:
                continue
            worst = float(np.max(np.abs(_row_norms(corpus.vectors) - 1.0)))
            if worst > norm_tolerance:
                problems.append(f"identity() says normalized=True but the {kind} have a vector "
                                f"whose norm differs from 1 by {worst:.3g} (tolerance "
                                f"{norm_tolerance:g})")
    return problems


# -------------------------------------------------------------------- stages


def _outcome(
    stage: Stage,
    corpus: MultiVectorCorpus,
    fit_queries: MultiVectorCorpus | None,
    score_queries: MultiVectorCorpus | None,
) -> dict[str, Any]:
    """Everything observable about a stage fitted on ``corpus`` (and ``fit_queries``), as
    arrays: the codes, the decoded vectors and, if ``score_queries`` is given, how it
    transforms and scores them."""
    pipeline = Pipeline([stage])
    pipeline.fit(corpus, fit_queries)
    compressed = pipeline.compress(corpus, fit=False)
    out: dict[str, Any] = {
        "codes": compressed.codes,
        "offsets": compressed.offsets,
        "decoded vectors": compressed.decode_rows(0, compressed.num_vectors),
        "ids": np.asarray(compressed.ids, dtype=object),
    }
    if score_queries is not None:
        moved = pipeline.transform_queries(score_queries)
        out["query vectors"] = moved.vectors
        out["scores"] = maxsim_matrix(moved, compressed)
    return out


def _differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Names of the observables in which two outcomes differ."""
    if a.keys() != b.keys():
        return ["the set of observables"]
    return [key for key in a if not _equal(a[key], b[key])]


def _expected_kind(stage: Stage) -> str | None:
    if isinstance(stage, TokenReducer):
        return "tokens"
    if isinstance(stage, DimensionReducer):
        return "dimensions"
    if isinstance(stage, Quantizer):
        return "bits"
    return None


def _check_description(stage: Stage, problems: list[str]) -> tuple[Any, Stage | None]:
    """The JSON-native description of ``stage`` and a stage rebuilt from it, with the
    problems found on the way (either may be None, and then no rebuilt stage is compared)."""
    name = stage.name
    try:
        spec = stage.to_dict()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"to_dict() raised {type(exc).__name__}: {exc}")
        return None, None
    try:
        plain = json.loads(canonical_json(spec))
    except (TypeError, ValueError) as exc:
        problems.append(f"to_dict() is not JSON-native ({type(exc).__name__}: {exc})")
        return None, None
    registered = STAGES.get(name)
    if registered is None:
        problems.append(f"{name!r} is not registered; decorate the class with "
                        "optivision.stages.base.register")
        return plain, None
    if registered is not type(stage):
        problems.append(f"{name!r} is registered for {registered.__name__}, not for "
                        f"{type(stage).__name__}; give the class a name of its own and "
                        "register it")
        return plain, None
    try:
        rebuilt = stage_from_dict(plain)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"stage_from_dict(to_dict()) raised {type(exc).__name__}: {exc}")
        return plain, None
    try:
        again = rebuilt.to_dict()
        same = canonical_json(again) == canonical_json(plain)
    except Exception as exc:  # noqa: BLE001
        again, same = f"{type(exc).__name__}: {exc}", False
    if type(rebuilt) is not type(stage) or not same:
        problems.append(f"to_dict() does not survive stage_from_dict: {plain!r} became {again!r}")
    return plain, rebuilt


def check_stage(
    stage: Stage, corpus: MultiVectorCorpus, queries: MultiVectorCorpus | None = None
) -> list[str]:
    """Problems found in a compression stage, or ``[]`` if it conforms.

    The stage is exercised on ``corpus`` (and ``queries`` if given) through a one-stage
    ``Pipeline``, on deep copies, so the instance passed in is never fitted or changed. The
    checks:

    * ``to_dict()`` is JSON-native, the name is registered for this class, and
      ``stage_from_dict`` rebuilds a stage of the same class with the same description and
      the same behaviour as the original;
    * two fits of copies give identical codes, decoded vectors and, with queries,
      identical transformed queries and scores (determinism);
    * ``corpus`` and ``queries`` are not modified (compared by ``corpus_identity``: vectors,
      offsets, ids and aligned arrays; metadata is not compared);
    * traits are declared (``optivision.core.stages.traits_of`` is not ``UNKNOWN``) and name
      the right kind of stage;
    * with queries: if fitting with and without them gives different results, the traits
      declare ``fit_uses_queries``, because the selection rule assumes that stages are fitted
      on documents only.

    If ``optivision.core.stages`` cannot be imported the traits checks are skipped and a
    warning says so. A stage that needs more than vectors (``spatial`` needs page images)
    reports the exception it raises on data without them. Passing says nothing about
    retrieval quality.
    """
    if not isinstance(stage, Stage):
        raise TypeError(f"stage must be a Stage, got {type(stage).__name__}")
    for what, obj in (("corpus", corpus), ("queries", queries)):
        if obj is not None and not isinstance(obj, MultiVectorCorpus):
            raise TypeError(f"{what} must be a MultiVectorCorpus, got {type(obj).__name__}")

    def digests() -> tuple[str, str | None]:
        return (corpus_identity(corpus).digest(),
                None if queries is None else corpus_identity(queries).digest())

    problems: list[str] = []
    before = digests()
    plain, rebuilt = _check_description(stage, problems)

    # behaviour: two copies, the rebuilt stage and, with queries, a copy fitted without them
    runs: dict[str, dict[str, Any]] = {}

    def run(label: str, make: Callable[[], Stage], fit_queries: MultiVectorCorpus | None) -> str:
        """Fit and apply a fresh stage; the error text if it raised, else an empty string."""
        try:
            runs[label] = _outcome(make(), corpus, fit_queries, queries)
        except Exception as exc:  # noqa: BLE001 -- whatever the stage raises is a finding
            return f"{type(exc).__name__}: {exc}"
        return ""

    without_queries = ""
    error = run("first copy", lambda: copy.deepcopy(stage), queries)
    if error:
        problems.append(f"fitting and applying the stage raised {error}")
    else:
        follow_ups: list[tuple[str, Callable[[], Stage], MultiVectorCorpus | None]] = [
            ("second copy", lambda: copy.deepcopy(stage), queries),
        ]
        if rebuilt is not None:
            follow_ups.append(("rebuilt stage", lambda: stage_from_dict(plain), queries))
        for label, make, fit_queries in follow_ups:
            error = run(label, make, fit_queries)
            if error:
                problems.append(f"the {label} raised {error}")
        if queries is not None:  # judged below, against what the traits declare
            without_queries = run("copy fitted without queries", lambda: copy.deepcopy(stage), None)
    base = runs.get("first copy")
    if base is not None and "second copy" in runs:
        diff = _differences(base, runs["second copy"])
        if diff:
            problems.append(f"not deterministic: two fits of the same stage differ in {diff}")
    if base is not None and "rebuilt stage" in runs:
        diff = _differences(base, runs["rebuilt stage"])
        if diff:
            problems.append("the stage rebuilt by stage_from_dict behaves differently from the "
                            f"original in {diff}")

    after = digests()
    if after[0] != before[0]:
        problems.append("the stage modified the corpus it was given")
    if after[1] != before[1]:
        problems.append("the stage modified the queries it was given")

    try:
        from ..core.stages import traits_of
    except ImportError:
        warnings.warn("optivision.core.stages could not be imported: the declared-traits checks "
                      "were skipped", stacklevel=2)
        return problems
    try:
        traits = traits_of(stage)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"traits_of(stage) raised {type(exc).__name__}: {exc}")
        return problems
    if str(getattr(traits.basis_invariance, "value", traits.basis_invariance)) == "unknown":
        problems.append(f"no basis traits are declared for {stage.name!r} (traits_of is UNKNOWN, "
                        "so it is treated as dependent on the coordinate basis); declare them "
                        "with optivision.core.stages.register_traits")
    expected = _expected_kind(stage)
    if expected is not None and traits.kind not in (expected, "unknown"):
        problems.append(f"the traits say kind {traits.kind!r} but this is a {expected!r} stage")
    if traits.fit_uses_queries:
        return problems
    if without_queries:
        problems.append(f"fitting without queries raised {without_queries}, but the traits do "
                        "not declare fit_uses_queries")
    elif base is not None and "copy fitted without queries" in runs:
        diff = _differences(base, runs["copy fitted without queries"])
        if diff:
            problems.append(f"fitting with and without queries gives different results ({diff}) "
                            "but the traits do not declare fit_uses_queries")
    return problems
