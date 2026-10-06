"""Cross-layer integration tests of the model-agnostic layer, on small seeded synthetic data.

The unit tests of each layer (``tests/test_core_*.py``, ``tests/test_backends_*.py``) were written
by the engineer who built that layer. These tests cross the layers: a pipeline is described,
rebuilt, fitted and compressed; the corpus is searched through the retrievers and the backend;
references, evidence and candidate spaces are built from the same inputs more than once. Nothing
here is a research experiment and no number computed on synthetic vectors says anything about an
encoder: they check that the implementation is consistent with itself and with the frozen
library functions it delegates to.

Shared helpers (``build_corpus``, ``build_queries``, ``PIPELINES``, ``maxsim64``, ...) live in
this module and are imported by the other ``test_integration_*`` files.

Defects found while writing these tests are kept out of the suite with
``xfail(strict=True)`` and a reason that starts with ``DEFECT``; ``docs/package/TEST_REPORT.md``
lists them.
"""

from __future__ import annotations

import itertools
import json
import re
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from optivision.api import certify
from optivision.backends import BackendRetriever, NumpyBackend, NumpyIndex
from optivision.compose import Pipeline
from optivision.core.evaluate import (
    Evidence,
    EvidenceFormatError,
    LabelReference,
    RankingReference,
    metrics_from_ranking,
    per_query_metrics,
)
from optivision.core.evaluate.collect import collect_evidence, compress_selected
from optivision.core.identity import array_digest, canonical_json, strings_digest
from optivision.core.pipeline import pipeline_identity
from optivision.core.retrieve import (
    CompressedRetriever,
    ExactRetriever,
    Ranking,
    TieredRetriever,
    TwoTierRetriever,
)
from optivision.core.selection import (
    REGISTERED_DEFAULT_SPACE_SHA256,
    BonferroniBetting,
    CandidateSpace,
    FixedSequenceBetting,
    select,
    validate_evidence_arrays,
)
from optivision.core.similarity import maxsim_matrix, rank
from optivision.core.stages import (
    BasisInvariance,
    RotationProjector,
    byte_accounting,
    pipeline_fitted_identity,
    pipeline_traits,
)
from optivision.core.vectors import (
    ContractError,
    corpus_identity,
    validate_corpus,
)
from optivision.representation import MultiVectorCorpus
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    DimensionProjector,
    Float16Quantizer,
    Float32Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RandomProjector,
    RandomPruner,
    RedundancyPruner,
    TruncateProjector,
)
from optivision.storage import to_legacy_pages
from optivision.testing import basis_probe, make_corpus, make_queries

try:
    import scipy.cluster.hierarchy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - SciPy is a dev dependency
    HAVE_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy is not installed")

#: float16 and float32 corpora; dimensions that are not multiples of 8 (13, 21, 37)
DTYPES = (np.float32, np.float16)
DTYPE_IDS = ("float32", "float16")
CASES = ((0, 13), (1, 21), (2, 37))  # (seed, dimension)
CASE_IDS = tuple(f"seed{s}-d{d}" for s, d in CASES)


# ====================================================================== data helpers


def build_corpus(
    seed: int,
    dim: int = 13,
    dtype: object = np.float32,
    n_docs: int = 24,
    tokens: tuple[int, int] = (3, 9),
    anisotropy: float = 0.7,
) -> MultiVectorCorpus:
    """A seeded corpus that always holds one empty document and one single-vector document.

    Document ids are ``d000000::p1``, ... (the form every backend keeps), documents 1 and 2
    are the single-vector and the empty one. Built on ``optivision.testing.make_corpus``.
    """
    base = make_corpus(n_docs, dim, tokens=tokens, seed=seed, dtype=dtype, anisotropy=anisotropy)
    arrays = [np.asarray(base.vectors[lo:hi]) for lo, hi in (base.span(i) for i in range(n_docs))]
    arrays[1] = arrays[1][:1]
    arrays[2] = arrays[2][:0]
    return MultiVectorCorpus.from_arrays(
        arrays, ids=[f"{i}::p1" for i in base.ids], attrs={"synthetic": True}
    )


def build_queries(
    corpus: MultiVectorCorpus, seed: int, n: int = 8, noise: float = 0.3
) -> tuple[MultiVectorCorpus, dict[str, set[str]]]:
    """Queries cut from known documents (``make_queries``) and their qrels."""
    return make_queries(corpus, n, tokens=(3, 6), noise=noise, seed=seed + 100)


def corpus_with_value(corpus: MultiVectorCorpus, value: float, row: int = 2, col: int = 1):
    """A copy of ``corpus`` with one vector entry replaced (used for NaN and infinity)."""
    v = np.array(corpus.vectors, copy=True)
    v[row, col] = value
    return MultiVectorCorpus(v, corpus.offsets, ids=corpus.ids)


def no_documents(dim: int, dtype: object = np.float32) -> MultiVectorCorpus:
    """A corpus with no documents (and so no vectors)."""
    return MultiVectorCorpus(np.zeros((0, dim), dtype=dtype), [0])


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Same dtype, same shape, same bytes (NaN equals NaN)."""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ============================================================ rounding-aware comparison

_U = 2.0**-23  # twice the float32 unit roundoff: a deliberately loose constant


def maxsim64(queries: MultiVectorCorpus, store) -> np.ndarray:
    """MaxSim in float64 from the values a float32 scorer sees (decoded float32 widened).

    An empty document scores ``-inf``; the scorers do the same. Used only as the yardstick of
    the rounding-aware comparison below.
    """
    q_off, d_off = np.asarray(queries.offsets), np.asarray(store.offsets)
    qv = np.asarray(queries.decode_rows(0, int(q_off[-1])), dtype=np.float64)
    dv = np.asarray(store.decode_rows(0, int(d_off[-1])), dtype=np.float64)
    out = np.empty((len(queries), len(store)))
    for i in range(len(queries)):
        q = qv[q_off[i] : q_off[i + 1]]
        for j in range(len(store)):
            lo, hi = int(d_off[j]), int(d_off[j + 1])
            out[i, j] = -np.inf if hi == lo else (q @ dv[lo:hi].T).max(axis=1).sum()
    return out


def score_bound(queries: MultiVectorCorpus, store) -> np.ndarray:
    """A float32 error bound per query: how far a float32 MaxSim may be from the float64 one.

    ``2 * (dim + n_tokens + 2) * 2**-23 * sum_i |q_i| * max_j |d_j|`` (a standard bound for a
    float32 inner product and a float32 sum, doubled for safety). Any two float32 paths that
    both stay within it of the float64 score can disagree about the order of two documents
    only if their float64 scores differ by less than twice the bound.
    """
    q_off, d_off = np.asarray(queries.offsets), np.asarray(store.offsets)
    qv = np.asarray(queries.decode_rows(0, int(q_off[-1])), dtype=np.float64)
    dv = np.asarray(store.decode_rows(0, int(d_off[-1])), dtype=np.float64)
    dmax = float(np.linalg.norm(dv, axis=1).max()) if dv.size else 0.0
    norms = np.linalg.norm(qv, axis=1)
    bound = np.empty(len(queries))
    for i in range(len(queries)):
        seg = norms[q_off[i] : q_off[i + 1]]
        bound[i] = 2.0 * (qv.shape[1] + seg.size + 2) * _U * float(seg.sum()) * dmax
    return bound


def assert_agrees_with_exact(
    ranking: Ranking, store_ids: list[str], exact: np.ndarray, bound: np.ndarray, label: str = ""
) -> None:
    """The rounding-aware criterion (``docs/package/BACKEND_ARCHITECTURE.md`` sections 5, 10).

    For every query: each returned score is within ``bound`` of the float64 score; no document
    is ranked above one whose float64 score exceeds its own by more than ``2 * bound``; and
    for every depth ``k`` at which the float64 scores separate the k-th from the (k+1)-th by
    more than ``2 * bound``, the top-k set is the float64 top-k set.
    """
    position = {doc_id: i for i, doc_id in enumerate(store_ids)}
    for q in range(ranking.n_queries):
        got = [position[d] for d in ranking.ids[q]]
        scores = ranking.scores[q, : len(got)].astype(np.float64)
        truth = exact[q, got]
        finite = np.isfinite(truth)
        assert np.array_equal(finite, np.isfinite(scores)), (label, q, "-inf pattern")
        assert (np.abs(scores[finite] - truth[finite]) <= bound[q]).all(), (label, q, "score")
        with np.errstate(invalid="ignore"):
            later_is_clearly_better = (truth[None, :] - truth[:, None]) > 2.0 * bound[q]
        upper = np.triu(np.ones((len(got), len(got)), dtype=bool), k=1)
        assert not (later_is_clearly_better & upper).any(), (label, q, "order")
        order = np.argsort(-exact[q], kind="stable")
        sorted_scores = exact[q][order]
        for k in (1, 3, 5):
            if k < len(got) and np.isfinite(sorted_scores[k]) and (
                sorted_scores[k - 1] - sorted_scores[k] > 2.0 * bound[q]
            ):
                assert set(got[:k]) == set(order[:k].tolist()), (label, q, k, "top-k set")


# =============================================================== the pipeline table

#: pipelines spanning the three stage kinds and their combinations; every entry builds a fresh
#: pipeline. Dimensions up to 13 are enough for every width used here.
PIPELINES: dict[str, Callable[[], Pipeline]] = {
    "empty": lambda: Pipeline([]),
    "float16": lambda: Pipeline([Float16Quantizer()]),
    "float32-explicit": lambda: Pipeline([Float32Quantizer()]),
    "merge": lambda: Pipeline([AdaptiveMerge(radius=0.5)]),
    "merge+int8": lambda: Pipeline(
        [AdaptiveMerge(radius=0.5, center="mean"), Int8Quantizer("per_vector")]
    ),
    "prune+binary-centred": lambda: Pipeline(
        [RandomPruner(0.5, seed=2), BinaryQuantizer(center="mean")]
    ),
    "redundancy+int8": lambda: Pipeline([RedundancyPruner(0.7), Int8Quantizer("per_vector")]),
    "pca+binary": lambda: Pipeline([PCAProjector(8), BinaryQuantizer()]),
    "rotation+int4": lambda: Pipeline([RotationProjector(seed=3), Int4Quantizer()]),
    "rotation+pca+int8-per-dim": lambda: Pipeline(
        [RotationProjector(seed=1), PCAProjector(10, center=True), Int8Quantizer("per_dimension")]
    ),
    "truncate+int8": lambda: Pipeline([TruncateProjector(9), Int8Quantizer("per_vector")]),
    "randproj+float16": lambda: Pipeline([RandomProjector(11, seed=2), Float16Quantizer()]),
    "project-pca+int4-centred": lambda: Pipeline(
        [DimensionProjector(8, method="pca"), Int4Quantizer(center="mean")]
    ),
    "lloyd2": lambda: Pipeline([Lloyd2Quantizer(seed=7)]),
    "ward+int4": pytest.param(
        lambda: Pipeline([HierarchicalMerge(ratio=0.5), Int4Quantizer(center="mean")]),
        marks=needs_scipy,
    ),
}


def pipeline_params():
    return [pytest.param(name, id=name, marks=getattr(make, "marks", ())) for name, make
            in PIPELINES.items()]


def make_pipeline(name: str) -> Pipeline:
    entry = PIPELINES[name]
    return (entry.values[0] if hasattr(entry, "values") else entry)()


# ===================================================== 1. pipeline composition


class TestPipelineComposition:
    @pytest.mark.parametrize("name", pipeline_params())
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize(("seed", "dim"), CASES, ids=CASE_IDS)
    def test_a_rebuilt_pipeline_behaves_identically(self, name, dtype, seed, dim):
        """``Pipeline.from_dict(p.to_dict())`` (and its JSON form) gives bitwise-equal codes,
        decoded vectors, fitted-state digests, query transforms and scores after fitting on
        the same corpus (documents include an empty and a single-vector document)."""
        corpus = build_corpus(seed, dim, dtype)
        queries, _ = build_queries(corpus, seed)
        original = make_pipeline(name)
        spec = original.to_dict()
        rebuilt = Pipeline.from_dict(spec)
        via_json = Pipeline.from_dict(json.loads(canonical_json(spec)))
        assert rebuilt.to_dict() == spec == via_json.to_dict()
        assert pipeline_identity(original) == pipeline_identity(rebuilt)
        assert pipeline_identity(original) == pipeline_identity(via_json)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            first = original.compress(corpus)
            others = [rebuilt.compress(corpus), via_json.compress(corpus)]
        assert pipeline_identity(original) == pipeline_identity(rebuilt), "fitting changed it"
        for twin, other in zip((rebuilt, via_json), others, strict=True):
            assert bitwise_equal(first.codes, other.codes)
            assert bitwise_equal(first.offsets, other.offsets)
            assert first.dim == other.dim and first.ids == other.ids
            assert first.nbytes == other.nbytes and first.report() == other.report()
            assert bitwise_equal(
                first.decode_rows(0, first.num_vectors), other.decode_rows(0, other.num_vectors)
            )
            assert pipeline_fitted_identity(original) == pipeline_fitted_identity(twin)
            projected, projected_twin = (
                original.transform_queries(queries), twin.transform_queries(queries))
            assert bitwise_equal(projected.vectors, projected_twin.vectors)
            assert bitwise_equal(projected.offsets, projected_twin.offsets)
            assert bitwise_equal(maxsim_matrix(projected, first),
                                 maxsim_matrix(projected_twin, other))

    @pytest.mark.parametrize("name", pipeline_params())
    def test_the_identity_of_a_pipeline_is_the_same_for_equal_specs(self, name):
        a, b = make_pipeline(name), make_pipeline(name)
        assert a is not b and pipeline_identity(a) == pipeline_identity(b)
        reversed_keys = {"stages": [dict(reversed(list(s.items()))) for s in a.to_dict()["stages"]]}
        assert pipeline_identity(Pipeline.from_dict(reversed_keys)) == pipeline_identity(a)

    def test_different_pipelines_have_different_identities(self):
        names = [n for n in PIPELINES if n != "ward+int4" or HAVE_SCIPY]
        identities = {pipeline_identity(make_pipeline(n)) for n in names}
        assert len(identities) == len(names)

    #: every constructor parameter of every registered stage, changed from its default
    VARIANTS = (
        [("adaptive_merge", AdaptiveMerge, {k: v}) for k, v in (
            ("radius", 0.7), ("ratio", 0.5), ("max_vectors", 4), ("min_vectors", 2),
            ("refine", 2), ("representative", "center"), ("seeding", "importance"),
            ("n_importance_seeds", 2), ("center", "mean"))]
        + [("hierarchical_merge", HierarchicalMerge, {k: v}) for k, v in (
            ("ratio", 0.25), ("max_distance", 0.5), ("min_vectors", 2))]
        + [("random_prune", RandomPruner, {k: v}) for k, v in (
            ("ratio", 0.25), ("seed", 5), ("min_vectors", 2))]
        + [("redundancy", RedundancyPruner, {k: v}) for k, v in (
            ("threshold", 0.8), ("merge", False), ("max_pairwise", 100))]
        + [("pca", PCAProjector, {k: v}) for k, v in (
            ("dim", 9), ("center", True), ("basis", "joint"), ("query_weight", 0.25),
            ("sample", 1000), ("seed", 5))]
        + [("random_projection", RandomProjector, {k: v}) for k, v in (("dim", 9), ("seed", 5))]
        + [("truncate", TruncateProjector, {"dim": 9})]
        + [("rotation", RotationProjector, {"seed": 5})]
        + [("project", DimensionProjector, {k: v}) for k, v in (
            ("dim", 9), ("method", "random"), ("method", "truncate"), ("center", True))]
        + [("int8", Int8Quantizer, {k: v}) for k, v in (
            ("scale", "per_vector"), ("scale", "per_dimension"), ("center", "mean"))]
        + [("int4", Int4Quantizer, {"center": "mean"}), ("binary", BinaryQuantizer, {"center": "mean"}),
           ("lloyd2", Lloyd2Quantizer, {"seed": 8})]
    )

    @pytest.mark.parametrize(
        ("stage_name", "cls", "change"), VARIANTS,
        ids=[f"{n}:{next(iter(c))}={next(iter(c.values()))}" for n, _, c in VARIANTS])
    def test_the_identity_changes_when_any_parameter_changes(self, stage_name, cls, change):
        base = Pipeline([cls()] if cls is not DimensionProjector else [cls(8)])
        kwargs = dict(change)
        if cls is DimensionProjector and "dim" not in kwargs:
            kwargs["dim"] = 8
        changed = Pipeline([cls(**kwargs)])
        assert changed.stages[0].name == stage_name
        assert changed.to_dict() != base.to_dict()
        assert pipeline_identity(changed) != pipeline_identity(base)
        for key, value in change.items():  # the parameter is in the spec, with its value
            assert changed.to_dict()["stages"][0][key] == value

    def test_the_identity_changes_with_stage_order_membership_and_count(self):
        a, b = AdaptiveMerge(radius=0.5), PCAProjector(8)
        base = Pipeline([a, b, BinaryQuantizer()])
        swapped = Pipeline([PCAProjector(8), AdaptiveMerge(radius=0.5), BinaryQuantizer()])
        fewer = Pipeline([PCAProjector(8), BinaryQuantizer()])
        other_codec = Pipeline([AdaptiveMerge(radius=0.5), PCAProjector(8), Int4Quantizer()])
        no_codec = Pipeline([AdaptiveMerge(radius=0.5), PCAProjector(8)])
        twice = Pipeline([AdaptiveMerge(radius=0.5), AdaptiveMerge(radius=0.5), PCAProjector(8),
                          BinaryQuantizer()])
        identities = [pipeline_identity(p) for p in (base, swapped, fewer, other_codec, no_codec, twice)]
        assert len(set(identities)) == len(identities)


# ----------------------------------------------------------------- traits properties

_TOKEN_POOL = (
    lambda: AdaptiveMerge(radius=0.6),
    lambda: AdaptiveMerge(radius=0.6, center="mean"),
    lambda: HierarchicalMerge(ratio=0.5),
    lambda: RedundancyPruner(0.8),
    lambda: RandomPruner(0.5, seed=1),
)
_DIMENSION_POOL = (
    lambda: PCAProjector(8),
    lambda: PCAProjector(8, center=True),
    lambda: RandomProjector(8, seed=1),
    lambda: TruncateProjector(8),
    lambda: RotationProjector(seed=2),
    lambda: DimensionProjector(8, method="pca"),
    lambda: DimensionProjector(8, method="random"),
    lambda: DimensionProjector(8, method="truncate"),
)
_QUANTIZER_POOL = (
    lambda: None,
    lambda: Float32Quantizer(),
    lambda: Float16Quantizer(),
    lambda: Int8Quantizer("fixed"),
    lambda: Int8Quantizer("per_vector"),
    lambda: Int8Quantizer("per_dimension", center="mean"),
    lambda: Int4Quantizer(),
    lambda: Int4Quantizer(center="mean"),
    lambda: BinaryQuantizer(),
    lambda: BinaryQuantizer(center="mean"),
    lambda: Lloyd2Quantizer(seed=3),
)
#: orthogonally invariant stages (declared ORTHOGONAL), and the one quantizer that adds no error
_ORTHOGONAL_POOL = (
    lambda: AdaptiveMerge(radius=0.6),
    lambda: AdaptiveMerge(radius=0.6, center="mean"),
    lambda: HierarchicalMerge(ratio=0.5),
    lambda: RedundancyPruner(0.8),
    lambda: RandomPruner(0.5, seed=1),
    lambda: PCAProjector(8),
    lambda: PCAProjector(8, basis="joint"),
    lambda: DimensionProjector(8, method="pca"),
    lambda: RotationProjector(seed=2),
    lambda: Float32Quantizer(),
)


def is_coordinate_codec(stage) -> bool:
    """binary, int4, int8, float16, lloyd2 and the leading-coordinate cut (``truncate``)."""
    if isinstance(stage, DimensionProjector):
        return stage.method == "truncate"
    return isinstance(stage, (BinaryQuantizer, Int4Quantizer, Int8Quantizer, Float16Quantizer,
                              Lloyd2Quantizer, TruncateProjector))


def random_valid_pipeline(rng: np.random.Generator) -> Pipeline:
    """Zero to three token or dimension stages in any order, then at most one quantizer."""
    pool = _TOKEN_POOL + _DIMENSION_POOL
    stages = [pool[int(i)]() for i in rng.integers(0, len(pool), size=int(rng.integers(0, 4)))]
    quantizer = _QUANTIZER_POOL[int(rng.integers(0, len(_QUANTIZER_POOL)))]()
    return Pipeline(stages + ([quantizer] if quantizer is not None else []))


class TestBasisSensitivityOfPipelines:
    def test_a_pipeline_with_a_coordinate_codec_is_always_basis_sensitive(self):
        """Random orders valid for ``Pipeline``; whatever precedes the codec."""
        rng = np.random.default_rng(2026)
        seen = {"after_rotation": 0, "after_pca": 0, "after_random": 0, "after_merge": 0,
                "truncate_inside": 0, "codec_alone": 0, "no_codec": 0}
        for _ in range(500):
            p = random_valid_pipeline(rng)
            traits = pipeline_traits(p)
            names = [s.name for s in p.stages]
            if any(is_coordinate_codec(s) for s in p.stages):
                assert traits.basis_sensitive is True, p.label()
                assert traits.basis_invariance is not BasisInvariance.ORTHOGONAL, p.label()
                last = p.stages[-1]
                if is_coordinate_codec(last) and len(names) > 1:
                    before = names[:-1]
                    seen["after_rotation"] += "rotation" in before
                    seen["after_pca"] += "pca" in before
                    seen["after_random"] += "random_projection" in before
                    seen["after_merge"] += any(n in before for n in (
                        "adaptive_merge", "hierarchical_merge", "redundancy", "random_prune"))
                seen["truncate_inside"] += any(
                    is_coordinate_codec(s) for s in p.stages[:-1]) or isinstance(
                    last, TruncateProjector)
                seen["codec_alone"] += len(names) == 1
            else:
                seen["no_codec"] += 1
            assert traits.basis_sensitive == (traits.basis_invariance is not BasisInvariance.ORTHOGONAL)
        assert all(count >= 10 for count in seen.values()), seen  # the draw is not vacuous

    def test_a_coordinate_codec_in_any_position_of_a_stage_list_is_sensitive(self):
        """``pipeline_traits`` also takes a plain list of stages, in any order (a quantizer in
        the middle is not a valid ``Pipeline`` but the rule is an order-based walk)."""
        rng = np.random.default_rng(7)
        pool = _TOKEN_POOL + _DIMENSION_POOL + _QUANTIZER_POOL[1:]
        checked = 0
        for _ in range(500):
            stages = [pool[int(i)]() for i in rng.integers(0, len(pool), size=int(rng.integers(1, 7)))]
            if any(is_coordinate_codec(s) for s in stages):
                assert pipeline_traits(stages).basis_sensitive is True, [s.name for s in stages]
                checked += 1
        assert checked > 200

    def test_sensitivity_is_never_lost_by_appending_stages(self):
        rng = np.random.default_rng(11)
        pool = _TOKEN_POOL + _DIMENSION_POOL + _QUANTIZER_POOL[1:]
        for _ in range(300):
            stages = [pool[int(i)]() for i in rng.integers(0, len(pool), size=int(rng.integers(1, 6)))]
            flags = [pipeline_traits(stages[: i + 1]).basis_sensitive for i in range(len(stages))]
            for earlier, later in itertools.pairwise(flags):
                assert later or not earlier, [s.name for s in stages]

    def test_only_orthogonally_invariant_stages_give_a_pipeline_that_is_not_sensitive(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            stages = [_ORTHOGONAL_POOL[int(i)]() for i in
                      rng.integers(0, len(_ORTHOGONAL_POOL), size=int(rng.integers(1, 5)))]
            assert pipeline_traits(stages).basis_sensitive is False, [s.name for s in stages]
            for codec in (BinaryQuantizer(), Int4Quantizer(), Int8Quantizer("per_vector"),
                          Float16Quantizer(), Lloyd2Quantizer(), TruncateProjector(8)):
                assert pipeline_traits([*stages, codec]).basis_sensitive is True


# ======================================================== 2. ranking equivalence

K_VALUES = (1, 5, 24, 31)  # depth 24 is the whole corpus, 31 exceeds it (clamped)


class TestRankingEquivalence:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize(("seed", "dim"), CASES, ids=CASE_IDS)
    def test_exact_retriever_is_maxsim_matrix_plus_rank(self, dtype, seed, dim):
        corpus = build_corpus(seed, dim, dtype)
        queries, _ = build_queries(corpus, seed)
        scores = maxsim_matrix(queries, corpus)
        retriever = ExactRetriever(corpus)
        assert bitwise_equal(retriever.score(queries), scores)
        for k in K_VALUES:
            got = retriever.search(queries, k)
            top = rank(scores, k)
            assert got.width == min(k, len(corpus))
            assert bitwise_equal(got.indices, top.astype(np.int64))
            assert bitwise_equal(got.scores, np.take_along_axis(scores, top, axis=1))
            assert got.ids == [[corpus.ids[i] for i in row] for row in top]

    @pytest.mark.parametrize("name", pipeline_params())
    @pytest.mark.parametrize(("seed", "dim"), CASES[:2], ids=CASE_IDS[:2])
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_compressed_retriever_is_projected_scoring_plus_rank(self, name, dtype, seed, dim):
        corpus = build_corpus(seed, dim, dtype)
        queries, _ = build_queries(corpus, seed)
        pipeline = make_pipeline(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            compressed = pipeline.compress(corpus)
        scores = maxsim_matrix(pipeline.transform_queries(queries), compressed)
        retriever = CompressedRetriever(compressed, pipeline)
        assert bitwise_equal(retriever.score(queries), scores)
        for k in (3, 24, 31):
            got = retriever.search(queries, k)
            top = rank(scores, k)
            assert bitwise_equal(got.indices, top.astype(np.int64))
            assert bitwise_equal(got.scores, np.take_along_axis(scores, top, axis=1))
            assert got.ids == [[compressed.ids[i] for i in row] for row in top]

    @pytest.mark.parametrize("pipeline", [lambda: Pipeline([]), lambda: Pipeline([Float32Quantizer()])],
                             ids=["empty", "float32"])
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize(("seed", "dim"), CASES, ids=CASE_IDS)
    def test_the_float32_pipeline_ranks_exactly_like_the_uncompressed_corpus(
            self, pipeline, dtype, seed, dim):
        corpus = build_corpus(seed, dim, dtype)
        queries, _ = build_queries(corpus, seed)
        p = pipeline()
        compressed = p.compress(corpus)
        assert compressed.nbytes == corpus.num_vectors * dim * 4  # float32 codes, no overhead
        for k in K_VALUES:
            assert CompressedRetriever(compressed, p).search(queries, k) == \
                ExactRetriever(corpus).search(queries, k)

    @pytest.mark.parametrize("first_stage", ["retriever", "ids-only"])
    @pytest.mark.parametrize("codec", ["binary", "int8"])
    @pytest.mark.parametrize(("seed", "dim"), CASES, ids=CASE_IDS)
    def test_a_two_tier_search_with_the_whole_corpus_as_shortlist_is_the_exact_ranking(
            self, first_stage, codec, seed, dim):
        """The shortlist is every document, so the first stage only proposes and the reranker's
        exact scores decide: equal to ``ExactRetriever`` within the rounding-aware criterion
        (two scoring paths, ``rescore`` and ``maxsim_matrix``, are equal to float32 rounding,
        never bitwise)."""
        corpus = build_corpus(seed, dim)
        queries, _ = build_queries(corpus, seed)
        make = {"binary": lambda: Pipeline([BinaryQuantizer()]),
                "int8": lambda: Pipeline([Int8Quantizer("per_vector")])}[codec]
        hot_pipeline = make()
        hot = CompressedRetriever(hot_pipeline.compress(corpus), hot_pipeline)
        first = hot if first_stage == "retriever" else BackendRetriever(_IdOnly(hot))
        exact = ExactRetriever(corpus)
        n = len(corpus)
        truth, bound = maxsim64(queries, corpus), score_bound(queries, corpus)
        for k in (1, 5, n, n + 7):
            two = TwoTierRetriever(first, exact, candidates=n).search(queries, k)
            reference = exact.search(queries, k)
            assert two.width == reference.width == min(k, n)
            assert two.score_kind == reference.score_kind and two.tie_break == "index"
            assert_agrees_with_exact(two, corpus.ids, truth, bound, "TwoTierRetriever")
            assert_agrees_with_exact(reference, corpus.ids, truth, bound, "ExactRetriever")
            assert_scores_close(two, reference, bound)

    @pytest.mark.parametrize("codec", ["binary", "int8"])
    @pytest.mark.parametrize(("seed", "dim"), CASES, ids=CASE_IDS)
    def test_a_tiered_search_with_the_whole_corpus_as_shortlist_is_the_exact_ranking(
            self, codec, seed, dim):
        corpus = build_corpus(seed, dim)
        queries, _ = build_queries(corpus, seed)
        hot_pipeline = {"binary": lambda: Pipeline([BinaryQuantizer()]),
                        "int8": lambda: Pipeline([Int8Quantizer("per_vector")])}[codec]()
        hot = CompressedRetriever(hot_pipeline.compress(corpus), hot_pipeline)
        exact = ExactRetriever(corpus)
        n = len(corpus)
        truth, bound = maxsim64(queries, corpus), score_bound(queries, corpus)
        for k in (1, 5, n, n + 7):
            tiered = TieredRetriever(hot, exact, candidates=n).search(queries, k)
            assert tiered.score_kind == "tiered" and tiered.width == min(k, n)
            assert_agrees_with_exact(tiered, corpus.ids, truth, bound, "TieredRetriever")
            assert_scores_close(tiered, exact.search(queries, k), bound)

    @pytest.mark.filterwarnings("ignore:Int8Quantizer:UserWarning")
    @pytest.mark.parametrize("merge", [False, True], ids=["plain", "merged"])
    @pytest.mark.parametrize(("codec", "method"), [(None, "none"), ("binary", "binary"),
                                                    ("int8", "int8")])
    @pytest.mark.parametrize(("seed", "dim"), CASES, ids=CASE_IDS)
    def test_the_numpy_backend_agrees_with_the_core_retriever_within_rounding(
            self, tmp_path, codec, method, merge, seed, dim):
        """``search_many`` over the same codes (an empty document included, page ids kept):
        scores within the float32 bound of the float64 score, no pair of documents whose
        float64 scores differ by more than twice the bound in the wrong order, and identical
        top-k sets wherever the gap at the boundary exceeds it."""
        corpus = build_corpus(seed, dim)
        queries, _ = build_queries(corpus, seed)
        stages = [AdaptiveMerge(radius=0.5)] if merge else []
        stages += {None: [], "binary": [BinaryQuantizer()], "int8": [Int8Quantizer("fixed")]}[codec]
        pipeline = Pipeline(stages)
        store = pipeline.compress(corpus)
        backend = NumpyBackend(NumpyIndex(tmp_path, dim=dim, method=method))
        backend.add(to_legacy_pages(store))
        core = CompressedRetriever(store, pipeline)
        truth, bound = maxsim64(queries, store), score_bound(queries, store)
        for k in (1, 5, len(corpus)):
            via_backend = backend.search_many(queries, k)
            via_core = core.search(queries, k)
            assert via_backend.score_kind == via_core.score_kind
            assert via_backend.tie_break == via_core.tie_break == "index"
            assert_agrees_with_exact(via_backend, store.ids, truth, bound, "NumpyBackend")
            assert_agrees_with_exact(via_core, store.ids, truth, bound, "CompressedRetriever")
            assert_scores_close(via_backend, via_core, bound)


def assert_scores_close(a: Ranking, b: Ranking, bound: np.ndarray) -> None:
    """Scores at the same rank differ by at most twice the float32 bound (finite entries)."""
    both = np.isfinite(a.scores) & np.isfinite(b.scores)
    assert np.array_equal(np.isfinite(a.scores), np.isfinite(b.scores))
    limit = np.broadcast_to(2.0 * bound[:, None], a.scores.shape)
    with np.errstate(invalid="ignore"):  # -inf minus -inf, an empty document in both
        gap = np.abs(a.scores.astype(np.float64) - b.scores.astype(np.float64))
    assert (gap[both] <= limit[both]).all()


class _IdOnly:
    """A stand-in for a backend: ranks like a retriever but reports ids without positions."""

    def __init__(self, retriever):
        self._retriever = retriever
        self.n_pages = retriever.n_docs

    def search_many(self, queries, k=10):
        return self._retriever.search(queries, k)


# ================================================== 3. reference and evidence stability

SMALL_FAMILIES = {
    "float16": [lambda: Pipeline([Float16Quantizer()])],
    "merge+int8": [lambda: Pipeline([Int8Quantizer("per_vector")]),
                   lambda: Pipeline([AdaptiveMerge(radius=0.7, center="mean"),
                                     Int8Quantizer("per_vector")])],
    "pca+binary": [lambda: Pipeline([BinaryQuantizer()]),
                   lambda: Pipeline([PCAProjector(10), BinaryQuantizer()]),
                   lambda: Pipeline([PCAProjector(6), BinaryQuantizer()])],
    "rotation+int4": [lambda: Pipeline([RotationProjector(seed=1), Int4Quantizer()])],
}


def small_space(name: str = "small") -> CandidateSpace:
    return CandidateSpace.from_families(
        {family: [make() for make in makes] for family, makes in SMALL_FAMILIES.items()}, name=name)


class TestReferenceAndEvidenceStability:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("seed", (0, 1, 2))
    def test_references_give_the_same_relevant_sets_across_calls_and_a_save_load(
            self, tmp_path, dtype, seed):
        corpus = build_corpus(seed, 21, dtype)
        queries, qrels = build_queries(corpus, seed)
        labels = LabelReference(qrels)
        snapshot = {q: set(docs) for q, docs in qrels.items()}
        first = labels.relevant(corpus.ids, queries.ids)
        for again in (labels.relevant(corpus.ids, queries.ids),
                      LabelReference(dict(reversed(list(qrels.items())))).relevant(
                          corpus.ids, queries.ids)):
            assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
        qrels["q00000"] = {"no-such-document"}  # the reference copied the mapping
        assert all(np.array_equal(a, b) for a, b in zip(
            first, labels.relevant(corpus.ids, queries.ids), strict=True))
        assert labels.identity() == LabelReference(snapshot).identity()
        assert labels.identity() != LabelReference(qrels).identity()

        path = corpus.save(tmp_path / "corpus.npz")
        for loaded in (MultiVectorCorpus.load(path), MultiVectorCorpus.load(path, mmap=True),
                       MultiVectorCorpus.load(corpus.save(tmp_path / "c.npz", compressed=True))):
            assert loaded.ids == corpus.ids
            assert all(np.array_equal(a, b) for a, b in zip(
                first, labels.relevant(loaded.ids, queries.ids), strict=True))

        base = maxsim_matrix(queries, corpus)
        for depth in (1, 3):
            agreement = RankingReference(depth)
            ref = agreement.relevant(corpus.ids, queries.ids, base)
            again = agreement.relevant(corpus.ids, queries.ids, base.copy())
            assert all(np.array_equal(a, b) for a, b in zip(ref, again, strict=True))
            loaded = MultiVectorCorpus.load(path)
            after = agreement.relevant(loaded.ids, queries.ids, maxsim_matrix(queries, loaded))
            assert all(np.array_equal(a, b) for a, b in zip(ref, after, strict=True))
            assert all(r.size == depth and (np.diff(r) > 0).all() for r in ref)
        assert RankingReference(1).identity() == {"construct": "agreement@1", "depth": 1}

    @pytest.mark.parametrize("seed", range(6))
    def test_metrics_from_a_ranking_equal_the_metrics_from_the_scores(self, seed):
        """Every metric the library offers (ndcg, recall, mrr, hit) at several cut-offs, with
        ties, with queries that have no relevant document, with fewer documents than the
        cut-off (padded with -1)."""
        rng = np.random.default_rng(seed)
        n_q, n_docs = 17, int(rng.choice([4, 9, 30]))
        scores = (np.round(rng.random((n_q, n_docs)) * 4) / 4).astype(np.float32)  # exact ties
        scores[:, rng.random(n_docs) < 0.1] = -np.inf  # a document that can never be retrieved
        relevant = []
        for _ in range(n_q):
            size = int(rng.choice([0, 1, 1, 2, 5]))
            relevant.append(np.sort(rng.choice(n_docs, size=min(size, n_docs), replace=False))
                            .astype(np.int64))
        metrics = [f"{kind}@{k}" for kind in ("ndcg", "recall", "mrr", "hit") for k in (1, 3, 5, 10)]
        expected = per_query_metrics(scores, relevant, metrics)
        kmax = 10
        top = rank(scores, kmax)
        if top.shape[1] < kmax:
            top = np.pad(top, ((0, 0), (0, kmax - top.shape[1])), constant_values=-1)
        got = metrics_from_ranking(top, relevant, metrics)
        assert list(got) == list(expected) == metrics
        for name in metrics:
            assert np.array_equal(got[name], expected[name], equal_nan=True), name
            assert np.isnan(got[name]).sum() == sum(r.size == 0 for r in relevant)
        wider = rank(scores, n_docs)  # a longer ranking changes nothing
        again = metrics_from_ranking(wider if n_docs >= kmax else top, relevant, metrics)
        for name in metrics:
            assert np.array_equal(again[name], expected[name], equal_nan=True), name

    @pytest.mark.parametrize("reference", ["labels", "agreement"])
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_evidence_built_twice_has_one_digest_and_survives_a_file(
            self, tmp_path, reference, dtype):
        corpus = build_corpus(3, 21, dtype)
        queries, qrels = build_queries(corpus, 3, n=10)
        make_reference = (lambda: LabelReference(qrels)) if reference == "labels" else RankingReference
        first = collect_evidence(corpus, queries, small_space(), make_reference())
        second = collect_evidence(corpus, queries, small_space("renamed"), make_reference())
        assert first == second and first.digest() == second.digest()
        assert bitwise_equal(first.candidates, second.candidates)
        assert first.fitted == second.fitted and len(first.fitted) == len(small_space())
        path = first.to_npz(tmp_path / "evidence")
        loaded = Evidence.from_npz(path)
        assert loaded == first and loaded.digest() == first.digest()
        assert bitwise_equal(loaded.candidates, first.candidates)
        assert bitwise_equal(loaded.baseline, first.baseline)
        assert loaded.query_ids == first.query_ids and loaded.fitted == first.fitted
        space = small_space()
        for target in (0.5, 0.9):
            d1 = select(space, first, target=target, error=0.1)
            d2 = select(space, loaded, target=target, error=0.1)
            assert d1.to_dict() == d2.to_dict()

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_every_candidate_can_be_deployed_as_it_was_evaluated(self, dtype):
        """The index built later by ``compress_selected`` (a fresh fit, no shared transform
        cache) is the one ``collect_evidence`` scored, for every candidate of the space: the
        fitted-state digests match (checked by ``expected_fitted``), the per-query metrics are
        reproduced exactly, and the size the rule used is the size ``byte_accounting`` reports."""
        corpus = build_corpus(6, 21, dtype)
        queries, qrels = build_queries(corpus, 6, n=12)
        space, reference = small_space(), LabelReference(qrels)
        evidence = collect_evidence(corpus, queries, space, reference)
        relevant = reference.relevant(corpus.ids, queries.ids, None)
        keep = [i for i, rel in enumerate(relevant) if rel.size]
        assert len(keep) == evidence.n_queries
        for k, spec in enumerate(space):
            pipeline, compressed = compress_selected(corpus, spec, expected_fitted=evidence.fitted[k])
            assert list(evidence.fitted[k]) == pipeline_fitted_identity(pipeline)
            scores = maxsim_matrix(pipeline.transform_queries(queries), compressed)
            got = per_query_metrics(scores, relevant, [evidence.metric])[evidence.metric][keep]
            assert bitwise_equal(got, evidence.candidates[k]), spec.family
            accounting = byte_accounting(compressed, pipeline)
            assert accounting.selection_bytes_per_doc == evidence.bytes_per_doc[k]
            assert compressed.num_vectors / len(compressed) == evidence.vectors_per_doc[k]
            assert accounting.code_bytes + accounting.codec_overhead_bytes == compressed.nbytes
            assert accounting.total_bytes_per_doc >= accounting.selection_bytes_per_doc

    def test_every_field_of_the_evidence_enters_its_digest(self):
        corpus = build_corpus(4, 21)
        queries, qrels = build_queries(corpus, 4, n=10)
        ev = collect_evidence(corpus, queries, small_space(), LabelReference(qrels))
        base = {
            "metric": ev.metric, "construct": ev.construct, "baseline": ev.baseline,
            "candidates": ev.candidates, "query_ids": ev.query_ids,
            "bytes_per_doc": ev.bytes_per_doc, "vectors_per_doc": ev.vectors_per_doc,
            "space_identity": ev.space_identity, "dropped_queries": ev.dropped_queries,
            "fitted": ev.fitted,
        }
        assert Evidence(**base).digest() == ev.digest()

        def bumped(array, index):
            a = np.array(array, copy=True)
            a[index] = np.nextafter(a[index], np.inf)  # one ulp
            return a

        rows = [list(entry) for entry in ev.fitted]
        stateful = next(k for k, entry in enumerate(rows) if entry[0] is not None)
        rows[stateful][0] = "f" * 64  # the fitted-state digest of one stage of one candidate
        changed_fitted = tuple(tuple(entry) for entry in rows)

        changes = {
            "metric": "recall@5", "construct": "agreement@1",
            "baseline": bumped(ev.baseline, 0), "candidates": bumped(ev.candidates, (1, 2)),
            "query_ids": tuple(reversed(ev.query_ids)),
            "bytes_per_doc": bumped(ev.bytes_per_doc, 0),
            "vectors_per_doc": bumped(ev.vectors_per_doc, 0),
            "space_identity": "0" * 64, "dropped_queries": (*ev.dropped_queries, "extra"),
            "fitted": changed_fitted,
        }
        for field, value in changes.items():
            assert Evidence(**{**base, field: value}).digest() != ev.digest(), field
        assert len({Evidence(**{**base, f: v}).digest() for f, v in changes.items()}) == len(changes)
        with pytest.raises(EvidenceFormatError, match="float64"):
            Evidence(**{**base, "candidates": ev.candidates.astype(np.float32)})


# ===================================================== 4. candidate identity stability


class TestCandidateIdentity:
    @needs_scipy
    @pytest.mark.parametrize("dim", (64, 96, 128, 1024))
    def test_the_default_space_does_not_depend_on_the_dimension(self, dim):
        space = CandidateSpace.default(dim)
        assert space.identity == REGISTERED_DEFAULT_SPACE_SHA256
        assert space.is_registered_default
        assert space.identity == CandidateSpace.default(128).identity
        assert len(space) == 40 and space.n_families == 7

    def test_the_order_of_insertion_is_part_of_the_identity(self):
        a, b, c = (lambda: Pipeline([Float16Quantizer()]), lambda: Pipeline([BinaryQuantizer()]),
                   lambda: Pipeline([Int4Quantizer()]))
        fam_one, fam_two = [a(), b()], [c()]
        forward = CandidateSpace.from_families({"x": fam_one, "y": fam_two})
        backward = CandidateSpace.from_families({"y": fam_two, "x": fam_one})
        inner = CandidateSpace.from_families({"x": [b(), a()], "y": fam_two})
        relabelled = CandidateSpace.from_families({"u": fam_one, "v": fam_two})
        again = CandidateSpace.from_families({"x": [a(), b()], "y": [c()]}, name="other name")
        assert forward.identity == again.identity  # the name is not part of the identity
        assert len({forward.identity, backward.identity, inner.identity, relabelled.identity}) == 4
        assert forward.families == ("x", "y") and backward.families == ("y", "x")
        # the same pipelines in a different order are a different space with different positions
        assert [s.position for s in backward] == [0, 1, 2]
        assert backward[0].family == "y" and forward[0].family == "x"

    def test_from_dict_of_to_dict_is_the_identity(self):
        for space in (small_space(), CandidateSpace.from_families({"only": [Pipeline([])]})):
            for data in (space.to_dict(), json.loads(json.dumps(space.to_dict()))):
                again = CandidateSpace.from_dict(data)
                assert again.identity == space.identity and again.name == space.name
                assert again.to_dict() == space.to_dict()
                assert [s.to_dict() for s in again] == [s.to_dict() for s in space]
        data = small_space().to_dict()
        data["candidates"][0]["pipeline"]["stages"][0]["stage"] = "float32"
        with pytest.raises(ValueError, match="identity"):
            CandidateSpace.from_dict(data)

    def test_the_pipelines_of_a_space_are_fresh_objects_every_time(self):
        space = small_space()
        identity, frozen = space.identity, [s.to_dict() for s in space]
        first, second = space.pipelines(), space.pipelines()
        assert all(a is not b for a, b in zip(first, second, strict=True))
        assert all(a.stages[i] is not b.stages[i] for a, b in zip(first, second, strict=True)
                   for i in range(len(a.stages)))
        corpus = build_corpus(0, 21)
        fitted = first[4]  # a PCA stage then binary: stateful until it is fitted
        assert not fitted.fitted
        fitted.compress(corpus)
        assert fitted.fitted and not second[4].fitted and not space.pipelines()[4].fitted
        first[1].stages[0].center = "tampered"  # mutating a built stage
        first[0].stages.append(Float16Quantizer())
        again = space.pipelines()
        assert [p.to_dict() for p in again] == [Pipeline.from_dict(s["pipeline"]).to_dict()
                                                for s in frozen]
        assert space.identity == identity and [s.to_dict() for s in space] == frozen
        with pytest.raises(TypeError):
            space[0].pipeline["stages"] = []  # type: ignore[index]


# ================================================= 5. shapes, dtypes, validation table


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    """One clean corpus, queries, stores, retrievers, backend and space for the bad-input tables."""
    corpus = build_corpus(5, 8, tokens=(3, 6), n_docs=12)
    queries, qrels = build_queries(corpus, 5, n=6)
    pipeline = Pipeline([BinaryQuantizer()])
    store = pipeline.compress(corpus)
    projector = Pipeline([PCAProjector(4), BinaryQuantizer()])
    projected = projector.compress(corpus)
    root = tmp_path_factory.mktemp("bad-inputs")
    backend = NumpyBackend(NumpyIndex(root, dim=8, method="binary"))
    backend.add(to_legacy_pages(store))
    exact = ExactRetriever(corpus)
    compressed = CompressedRetriever(store, pipeline)
    space = CandidateSpace.from_families({"f16": [Pipeline([Float16Quantizer()])],
                                          "bin": [Pipeline([BinaryQuantizer()])]})
    return SimpleNamespace(
        corpus=corpus, queries=queries, qrels=qrels, pipeline=pipeline, store=store,
        projector=projector, projected=projected, backend=backend, exact=exact,
        compressed=compressed, projected_retriever=CompressedRetriever(projected, projector),
        tiered=TieredRetriever(compressed, exact, candidates=4),
        two_tier=TwoTierRetriever(compressed, exact, candidates=4),
        space=space, reference=LabelReference(qrels))


def _bad_inputs(dim: int) -> dict[str, Callable[[MultiVectorCorpus], object]]:
    """Bad vector inputs, made from a clean corpus of dimension ``dim``."""
    return {
        "nan": lambda c: corpus_with_value(c, np.nan),
        "inf": lambda c: corpus_with_value(c, np.inf),
        "negative_inf": lambda c: corpus_with_value(c, -np.inf),
        "other_dimension": lambda c: make_corpus(len(c), dim - 2, tokens=(3, 4), seed=9),
        "no_documents": lambda c: no_documents(dim),
        "raw_array": lambda c: np.asarray(c.vectors),
    }


#: the message must name the problem; each pattern needs every one of its lookaheads
NAMES = {
    "nan": r"(?is)(?=.*(non-finite|nan))",
    "inf": r"(?is)(?=.*(non-finite|infinit))",
    "negative_inf": r"(?is)(?=.*(non-finite|infinit))",
    "other_dimension": r"(?is)(?=.*dimension)(?=.*(quer|document|corpus|vectors))",
    "no_documents": r"(?is)(?=.*(no vectors|no queries|no documents|nothing|at least one|empty))",
    "raw_array": r"(?is)(?=.*(MultiVectorCorpus|corpus|list of|vector store|ndarray))",
}

#: entry point -> callable(fx, bad queries)
QUERY_SLOT: dict[str, Callable] = {
    "ExactRetriever.search": lambda fx, bad: fx.exact.search(bad, 3),
    "ExactRetriever.score": lambda fx, bad: fx.exact.score(bad),
    "ExactRetriever.rescore": lambda fx, bad: fx.exact.rescore(
        bad, np.zeros((len(bad) if hasattr(bad, "offsets") else 1, 2), np.int64)),
    "CompressedRetriever.search": lambda fx, bad: fx.compressed.search(bad, 3),
    "CompressedRetriever(projected).search": lambda fx, bad: fx.projected_retriever.search(bad, 3),
    "TieredRetriever.search": lambda fx, bad: fx.tiered.search(bad, 3),
    "TwoTierRetriever.search": lambda fx, bad: fx.two_tier.search(bad, 3),
    "NumpyBackend.search_many": lambda fx, bad: fx.backend.search_many(bad, 3),
    "NumpyBackend.score": lambda fx, bad: fx.backend.score(bad),
    "Pipeline.transform_queries": lambda fx, bad: fx.projector.transform_queries(bad),
    "collect_evidence(queries)": lambda fx, bad: collect_evidence(
        fx.corpus, bad, fx.space, fx.reference),
    "basis_probe(queries)": lambda fx, bad: basis_probe(fx.pipeline, fx.corpus, bad),
    "certify(queries)": lambda fx, bad: certify(fx.corpus, bad, fx.qrels, space=fx.space),
}

#: entry point -> callable(fx, bad documents)
CORPUS_SLOT: dict[str, Callable] = {
    "validate_corpus": lambda fx, bad: validate_corpus(bad),
    "corpus_identity": lambda fx, bad: corpus_identity(bad),
    "Pipeline.fit": lambda fx, bad: Pipeline([Float16Quantizer()]).fit(bad),
    "Pipeline.compress": lambda fx, bad: Pipeline([Float16Quantizer()]).compress(bad),
    "ExactRetriever(corpus)": lambda fx, bad: ExactRetriever(bad).search(fx.queries, 3),
    "collect_evidence(corpus)": lambda fx, bad: collect_evidence(
        bad, fx.queries, fx.space, fx.reference),
    "compress_selected": lambda fx, bad: compress_selected(bad, fx.space[0]),
    "basis_probe(corpus)": lambda fx, bad: basis_probe(fx.pipeline, bad, fx.queries),
    "certify(vectors)": lambda fx, bad: certify(bad, fx.queries, fx.qrels, space=fx.space),
}

#: a corpus of another width is a valid corpus on its own: only a function that compares the
#: documents with queries can refuse it
NOT_AN_INPUT = {
    ("validate_corpus", "other_dimension"), ("corpus_identity", "other_dimension"),
    ("Pipeline.fit", "other_dimension"), ("Pipeline.compress", "other_dimension"),
    ("compress_selected", "other_dimension"),
}

_NO_QUERIES = "a query set with no queries is accepted and answered with an empty ranking"

#: (entry point, kind) -> why the input is accepted: documented behaviour, not an error
ACCEPTED = {
    ("ExactRetriever.rescore", "no_documents"): _NO_QUERIES,
    ("ExactRetriever.search", "no_documents"): _NO_QUERIES,
    ("ExactRetriever.score", "no_documents"): _NO_QUERIES,
    ("CompressedRetriever.search", "no_documents"): _NO_QUERIES,
    ("CompressedRetriever(projected).search", "no_documents"): _NO_QUERIES,
    ("TieredRetriever.search", "no_documents"): _NO_QUERIES,
    ("TwoTierRetriever.search", "no_documents"): _NO_QUERIES,
    ("NumpyBackend.search_many", "no_documents"): _NO_QUERIES,
    ("NumpyBackend.score", "no_documents"): _NO_QUERIES,
    ("Pipeline.transform_queries", "no_documents"): "no vectors: nothing to project or reject",
    ("validate_corpus", "no_documents"): "a corpus with no documents satisfies the contract",
    ("corpus_identity", "nan"): "an identity hashes the stored bytes, NaN included",
    ("corpus_identity", "inf"): "an identity hashes the stored bytes, infinity included",
    ("corpus_identity", "negative_inf"): "an identity hashes the stored bytes, infinity included",
    ("corpus_identity", "no_documents"): "an empty corpus has an identity",
    ("Pipeline.compress", "no_documents"): "a stateless pipeline can compress zero vectors",
    ("ExactRetriever(corpus)", "no_documents"): "a store with no documents has an empty ranking",
}

_PROMISE = (
    "although checks.py promises a message that names the problem instead of a failure inside "
    "a matrix product")

#: (entry point, kind) -> the defect; docs/package/TEST_REPORT.md lists them all
KNOWN_DEFECTS = {
    ("CompressedRetriever.search", "other_dimension"): (
        "DEFECT: CompressedRetriever given a pipeline skips the query-dimension check "
        "(exact.py sets _query_dim=None whenever a pipeline is given), so wrong-width queries "
        "fail inside NumPy's matmul with a message that does not name the queries, " + _PROMISE),
    ("CompressedRetriever(projected).search", "other_dimension"): (
        "DEFECT: same root cause: the projected queries are never compared with the store's "
        "dimension, " + _PROMISE),
    ("TwoTierRetriever.search", "other_dimension"): (
        "DEFECT: same root cause: the first stage here is a CompressedRetriever with a "
        "pipeline, so wrong-width queries fail inside NumPy's matmul, " + _PROMISE),
    ("Pipeline.transform_queries", "other_dimension"): (
        "DEFECT (pinned compose.py, reached through the core.pipeline facade): a projector "
        "fails inside NumPy's matmul with a message that does not name the queries"),
    ("Pipeline.transform_queries", "raw_array"): (
        "DEFECT (pinned compose.py): a raw array raises AttributeError, not TypeError/ValueError"),
    ("Pipeline.fit", "raw_array"): (
        "DEFECT (pinned compose.py): a raw array raises AttributeError, not TypeError/ValueError"),
    ("Pipeline.compress", "raw_array"): (
        "DEFECT (pinned compose.py): a raw array raises AttributeError, not TypeError/ValueError"),
    ("ExactRetriever(corpus)", "nan"): (
        "DEFECT: ExactRetriever validates the queries but not the documents; a NaN document "
        "vector gives that document a NaN score, so search(k < n_docs) never returns it and says "
        "nothing (k >= n_docs raises 'scores contain NaN')"),
    ("ExactRetriever(corpus)", "inf"): (
        "DEFECT: ExactRetriever does not validate documents; an infinite document vector gives "
        "infinite or NaN scores that are returned or dropped without a message"),
    ("ExactRetriever(corpus)", "negative_inf"): (
        "DEFECT: ExactRetriever does not validate documents; see the NaN case"),
}


def table(functions: dict[str, Callable]) -> list:
    """One test case for every (entry point, kind of bad input) that is an input at all."""
    rows = []
    for name in functions:
        for kind in NAMES:
            if (name, kind) in NOT_AN_INPUT:
                continue
            marks = []
            if (name, kind) in KNOWN_DEFECTS:
                marks.append(pytest.mark.xfail(strict=True, reason=KNOWN_DEFECTS[(name, kind)]))
            rows.append(pytest.param(name, kind, id=f"{name}|{kind}", marks=marks))
    return rows


class TestBadVectorInputs:
    """Every public entry that takes vectors refuses NaN, infinity, a dimension that does not
    match, no data and a raw array, with a ``ValueError`` or ``TypeError`` whose message names
    the problem. Documented exceptions are listed in ``ACCEPTED``; defects in ``KNOWN_DEFECTS``."""

    @pytest.mark.parametrize(("name", "kind"), table(QUERY_SLOT))
    def test_queries(self, fx, name, kind):
        # the bad queries are made from the documents (any vector set will do) except for the
        # dimension case, which needs the queries' own size
        bad = _bad_inputs(fx.corpus.dimension)[kind](fx.corpus)
        self._check(fx, QUERY_SLOT[name], name, kind, bad)

    @pytest.mark.parametrize(("name", "kind"), table(CORPUS_SLOT))
    def test_documents(self, fx, name, kind):
        bad = _bad_inputs(fx.corpus.dimension)[kind](fx.corpus)
        self._check(fx, CORPUS_SLOT[name], name, kind, bad)

    @staticmethod
    def _check(fx, call, name, kind, bad):
        if (name, kind) in ACCEPTED:
            call(fx, bad)  # documented: accepted, no exception
            return
        with pytest.raises((ValueError, TypeError)) as raised:
            call(fx, bad)
        assert re.search(NAMES[kind], str(raised.value)), (
            f"{name} / {kind}: the message does not name the problem: {raised.value}")

    def test_the_accepted_inputs_are_documented_and_still_accepted(self, fx):
        """An accepted entry does not silently turn into an error (or the table is stale)."""
        assert all(isinstance(why, str) and why for why in ACCEPTED.values())
        ranking = fx.exact.search(no_documents(8), 3)
        assert ranking.n_queries == 0 and ranking.ids == []
        assert fx.backend.search_many(no_documents(8), 3).n_queries == 0
        assert validate_corpus(no_documents(8)) is None
        assert corpus_identity(corpus_with_value(fx.corpus, np.nan)).n_docs == len(fx.corpus)

    @pytest.mark.parametrize(
        ("vectors", "offsets", "error", "text"),
        [
            (np.ones((3, 4), dtype=np.int64), [0, 3], TypeError, "floating"),
            (np.ones((3, 4), dtype=np.int8), [0, 3], TypeError, "floating"),
            (np.ones((3, 4), dtype=bool), [0, 3], TypeError, "floating"),
            (np.ones((3, 4), dtype=complex), [0, 3], TypeError, "floating"),
            (np.ones(4, dtype=np.float32), [0, 4], ValueError, "2-D"),
            (np.ones((1, 3, 4), dtype=np.float32), [0, 3], ValueError, "2-D"),
            (np.ones((3, 4), dtype=np.float32), [0, 2], ValueError, "offsets"),
            (np.ones((3, 4), dtype=np.float32), [1, 3], ValueError, "offsets"),
            (np.ones((3, 4), dtype=np.float32), [0, 2, 1, 3], ValueError, "non-decreasing"),
        ],
        ids=["int64", "int8", "bool", "complex", "1-D", "3-D", "short-offsets", "offset-start",
             "decreasing-offsets"],
    )
    def test_the_corpus_constructor_refuses_wrong_dtypes_and_shapes(
            self, vectors, offsets, error, text):
        with pytest.raises(error, match=text):
            MultiVectorCorpus(vectors, offsets)

    @pytest.mark.parametrize("what", ["queries", "vectors"])
    @pytest.mark.parametrize("bad", [None, [], "text", 3.5, {"a": 1}], ids=repr)
    def test_certify_refuses_inputs_that_are_not_vectors(self, fx, what, bad):
        kwargs = {"queries": fx.queries, "vectors": fx.corpus}
        kwargs[what] = bad
        with pytest.raises((ValueError, TypeError)):
            certify(kwargs["vectors"], kwargs["queries"], fx.qrels, space=fx.space)

    def test_certify_refuses_integer_arrays_naming_the_dtype(self, fx):
        ints = [np.ones((3, 8), dtype=np.int32)]
        with pytest.raises(TypeError, match="floating point"):
            certify(ints, fx.queries, space=fx.space)
        with pytest.raises(TypeError, match="floating point"):
            certify(fx.corpus, ints, space=fx.space)


class TestBadEvidenceInputs:
    """The same for the arrays a selection rule and an ``Evidence`` receive."""

    @staticmethod
    def arrays(n=12, k=4, seed=0):
        rng = np.random.default_rng(seed)
        b = rng.random(n)
        c = np.clip(b[None, :] * rng.random((k, n)) ** 0.2, 0.0, 1.0)
        return {"candidates": c, "baseline": b, "families": ["a", "a", "b", "b"][:k],
                "bytes_per_doc": np.array([4.0, 3.0, 2.0, 1.0])[:k],
                "vectors_per_doc": np.ones(k), "target": 0.9, "error": 0.05}

    BAD: ClassVar[dict] = {
        "candidates_1d": ({"candidates": lambda a: a["candidates"][0]}, "candidates"),
        "candidates_3d": ({"candidates": lambda a: a["candidates"][None]}, "candidates"),
        "baseline_2d": ({"baseline": lambda a: a["baseline"][None]}, "baseline"),
        "baseline_short": ({"baseline": lambda a: a["baseline"][:-1]}, "baseline"),
        "candidates_nan": ({"candidates": lambda a: np.where(np.arange(12) == 3, np.nan,
                                                             a["candidates"])}, "candidates"),
        "candidates_inf": ({"candidates": lambda a: np.where(np.arange(12) == 3, np.inf,
                                                             a["candidates"])}, "candidates"),
        "baseline_nan": ({"baseline": lambda a: np.where(np.arange(12) == 3, np.nan,
                                                         a["baseline"])}, "baseline"),
        "above_one": ({"candidates": lambda a: a["candidates"] * 3}, "candidates"),
        "below_zero": ({"candidates": lambda a: a["candidates"] - 1.0}, "candidates"),
        "no_queries": ({"candidates": lambda a: a["candidates"][:, :0],
                        "baseline": lambda a: a["baseline"][:0]}, "no queries"),
        "no_candidates": ({"candidates": lambda a: a["candidates"][:0],
                           "families": lambda a: [], "bytes_per_doc": lambda a: a["bytes_per_doc"][:0],
                           "vectors_per_doc": lambda a: a["vectors_per_doc"][:0]}, "candidate"),
        "zero_baseline": ({"baseline": lambda a: a["baseline"] * 0.0}, "baseline"),
        "bool_candidates": ({"candidates": lambda a: a["candidates"] > 0.5}, "candidates"),
        "complex_candidates": ({"candidates": lambda a: a["candidates"].astype(complex)},
                               "candidates"),
        "object_candidates": ({"candidates": lambda a: a["candidates"].astype(object)},
                              "candidates"),
        "string_candidates": ({"candidates": lambda a: a["candidates"].astype(str)}, "candidates"),
        "families_one_string": ({"families": lambda a: "abab"}, "families"),
        "families_short": ({"families": lambda a: ["a", "a", "b"]}, "families"),
        "families_ints": ({"families": lambda a: [1, 1, 2, 2]}, "family"),
        "size_nan": ({"bytes_per_doc": lambda a: np.array([np.nan, 1, 1, 1])}, "bytes_per_doc"),
        "size_negative": ({"bytes_per_doc": lambda a: -a["bytes_per_doc"]}, "bytes_per_doc"),
        "size_short": ({"bytes_per_doc": lambda a: a["bytes_per_doc"][:3]}, "bytes_per_doc"),
        "vectors_short": ({"vectors_per_doc": lambda a: a["vectors_per_doc"][:3]},
                          "vectors_per_doc"),
        "target_zero": ({"target": lambda a: 0.0}, "target"),
        "target_above_one": ({"target": lambda a: 1.1}, "target"),
        "target_nan": ({"target": lambda a: float("nan")}, "target"),
        "target_array": ({"target": lambda a: np.array([0.9, 0.95])}, "target"),
        "target_bool": ({"target": lambda a: True}, "target"),
        "target_text": ({"target": lambda a: "0.9"}, "target"),
        "error_zero": ({"error": lambda a: 0.0}, "error"),
        "error_one": ({"error": lambda a: 1.0}, "error"),
        "error_list": ({"error": lambda a: [0.05]}, "error"),
    }

    @staticmethod
    def bad_arguments(change):
        a = TestBadEvidenceInputs.arrays()
        b = dict(a)
        for key, fn in change.items():
            b[key] = fn(a)
        return b

    @pytest.mark.parametrize("case", sorted(BAD))
    @pytest.mark.parametrize("entry", ["validate_evidence_arrays", "FixedSequenceBetting.decide",
                                       "BonferroniBetting.decide"])
    def test_a_rule_refuses_bad_arrays_naming_them(self, case, entry):
        change, name = self.BAD[case]
        args = self.bad_arguments(change)
        call = {"validate_evidence_arrays": validate_evidence_arrays,
                "FixedSequenceBetting.decide": FixedSequenceBetting().decide,
                "BonferroniBetting.decide": BonferroniBetting().decide}[entry]
        with pytest.raises((ValueError, TypeError), match=re.compile(name, re.IGNORECASE)):
            call(args["candidates"], args["baseline"], args["families"], args["bytes_per_doc"],
                 args["vectors_per_doc"], args["target"], args["error"])

    def test_the_documented_widening_of_validate_evidence_arrays(self):
        """Integer, float16 and float32 input is widened to float64 exactly (documented); the
        values are then the same numbers. Everything else is refused (``test_a_rule_refuses``)."""
        a = self.arrays()
        ints = (a["candidates"] > 0.5).astype(np.int64)
        for candidates in (ints, a["candidates"].astype(np.float32), a["candidates"].astype(np.float16)):
            checked = validate_evidence_arrays(candidates, a["baseline"], a["families"],
                                               a["bytes_per_doc"], a["vectors_per_doc"], 0.9, 0.05)
            assert checked.candidates.dtype == np.float64
            assert np.array_equal(checked.candidates, candidates.astype(np.float64))

    @pytest.mark.parametrize(
        ("field", "value", "text"),
        [("candidates", lambda a: a.astype(np.float32), "float64"),
         ("candidates", lambda a: (a > 0.5).astype(np.int64), "float64"),
         ("baseline", lambda a: a.astype(np.float16), "float64"),
         ("bytes_per_doc", lambda a: a.astype(np.float32), "float64"),
         ("candidates", lambda a: a[0], "dimension"),
         ("query_ids", lambda a: "q0q1", "sequence of strings"),
         ("metric", lambda a: "", "non-empty")],
        ids=["float32", "int", "float16", "float32-size", "1-D", "ids-as-text", "no-metric"])
    def test_an_evidence_refuses_what_it_would_have_to_convert(self, field, value, text):
        rng = np.random.default_rng(0)
        b = rng.random(5)
        args = {
            "metric": "ndcg@5", "construct": "labels", "baseline": b,
            "candidates": np.clip(b[None, :] * rng.random((2, 5)), 0, 1),
            "query_ids": tuple(f"q{i}" for i in range(5)),
            "bytes_per_doc": np.array([2.0, 1.0]), "vectors_per_doc": np.ones(2),
            "space_identity": "x" * 64,
        }
        args[field] = value(args[field]) if field not in ("query_ids", "metric") else value(None)
        with pytest.raises(EvidenceFormatError, match=text):
            Evidence(**args)


class TestDtypesAreNeverChangedSilently:
    def test_a_float64_corpus_keeps_its_dtype_in_identities_and_checks(self, fx):
        c32 = fx.corpus
        c64 = MultiVectorCorpus(c32.vectors.astype(np.float64), c32.offsets, ids=c32.ids)
        assert corpus_identity(c64).dtype == "float64" and corpus_identity(c32).dtype == "float32"
        assert corpus_identity(c64).vectors_sha256 != corpus_identity(c32).vectors_sha256
        with pytest.raises(ContractError, match=r"float32.*float64|float64.*float32"):
            validate_corpus(c64, expect_dtype=np.float32)
        assert array_digest(c64.vectors) != array_digest(c32.vectors)

    def test_float64_input_is_narrowed_by_the_scorers_as_documented_and_nowhere_else(self, fx):
        """``exact.py`` documents that float64 documents and queries are narrowed to float32
        by the scorer; the narrowing is exactly ``astype(float32)``. The result type is float32."""
        c32 = fx.corpus
        c64 = MultiVectorCorpus(c32.vectors.astype(np.float64), c32.offsets, ids=c32.ids)
        q64 = MultiVectorCorpus(fx.queries.vectors.astype(np.float64), fx.queries.offsets,
                                ids=fx.queries.ids)
        assert c64.dtype == np.float64 and q64.dtype == np.float64  # kept by the container
        got = ExactRetriever(c64).search(q64, 5)
        assert got.scores.dtype == np.float32
        assert got == ExactRetriever(c32).search(fx.queries, 5)

    def test_certify_records_the_dtype_it_was_given(self, fx):
        c64 = MultiVectorCorpus(fx.corpus.vectors.astype(np.float64), fx.corpus.offsets,
                                ids=fx.corpus.ids)
        result = certify(c64, fx.queries, fx.qrels, space=fx.space, target=0.5, error=0.2)
        assert result.certificate.to_dict()["corpus"]["dtype"] == "float64"
        if result.compressed is not None:
            assert result.compressed.stats["native_dtype_before"] == "float64"

    def test_rankings_and_evidence_refuse_other_dtypes(self):
        scores = np.array([[3.0, 2.0, 1.0]])
        with pytest.raises(TypeError, match="float32"):
            Ranking(ids=[["a", "b", "c"]], scores=scores)
        with pytest.raises(TypeError, match="int64"):
            Ranking(ids=[["a", "b", "c"]], scores=scores.astype(np.float32),
                    indices=np.array([[0, 1, 2]], dtype=np.int32))
        with pytest.raises(TypeError, match="integer"):
            metrics_from_ranking(np.array([[0.0, 1.0, 2.0]]), [np.array([0])], ["ndcg@3"])

    @pytest.mark.parametrize("scores", [[[np.nan]], [[np.nan, 1.0]], [[2.0, np.nan]],
                                        [[3.0, np.nan, 1.0]]], ids=str)
    def test_a_ranking_refuses_nan_scores_by_name_at_any_width(self, scores):
        """Kills the mutation that drops the NaN check: with one column the order check has
        nothing to compare, so only the NaN check stands between a NaN score and a Ranking."""
        array = np.array(scores, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            Ranking(ids=[["a", "b", "c"][: array.shape[1]]], scores=array)

    def test_a_retriever_over_nothing_but_nan_scores_refuses_to_rank_them(self):
        """Every document holds a NaN, so every score is NaN: depth one has a single column."""
        base = make_corpus(6, 8, tokens=(3, 4), seed=0)
        v = np.array(base.vectors, copy=True)
        v[base.offsets[:-1], 0] = np.nan  # the first vector of every document
        queries, _ = make_queries(base, 3, seed=0)
        with pytest.raises(ValueError, match="NaN"):
            ExactRetriever(MultiVectorCorpus(v, base.offsets, ids=base.ids)).search(queries, 1)

    def test_strings_digest_stringifies_non_strings(self):
        """OBSERVATION: ``strings_digest([1, 2])`` equals ``strings_digest(['1', '2'])`` because
        every item goes through ``str``; corpus ids are always strings, so no corpus is affected."""
        assert strings_digest([1, 2]) == strings_digest(["1", "2"])
        with pytest.raises(TypeError, match="not one string"):
            strings_digest("12")

    @pytest.mark.xfail(strict=True, reason=(
        "DEFECT: RankingReference.relevant does not check that the baseline scores are finite; "
        "NaN scores give an index-order 'reference' (the first documents) without a message"))
    def test_the_agreement_reference_refuses_nan_scores(self):
        scores = np.full((2, 5), np.nan)
        with pytest.raises((ValueError, TypeError), match="(?i)nan|finite"):
            RankingReference(1).relevant([str(i) for i in range(5)], ["q0", "q1"], scores)
