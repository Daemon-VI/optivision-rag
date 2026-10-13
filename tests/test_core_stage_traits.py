"""Declared stage traits, checked against what the stages do on synthetic data.

A trait is a statement about code, not about retrieval quality. For every declared level the
test applies a transformation ``G`` to all document and query vectors, builds the same
pipeline fresh on the result (so it is fitted again), scores it with the library's own path
(``calibration.score_candidate``) and compares the ``[n_queries, n_docs]`` score matrices:

* ORTHOGONAL: equal under random orthogonal ``G`` and under signed permutations;
* SIGNED_PERMUTATION: equal under signed permutations, and a witness that a random rotation
  changes the scores by clearly more than rounding (so the label is not vacuous);
* NONE: a witness that a plain permutation of the axes changes the scores.

The corpus is anisotropic on purpose (per-axis lognormal scales, a nonzero mean in the stored
basis, topic structure inside documents so that merging stages have something to merge).

Tolerances are fixed from float32 arithmetic and are measured against the per-query spread of
the uncompressed scores over the documents. A score is a sum of a handful of inner products;
one float32 spacing at that size is about 5e-7, and a few spacings against a spread of at
least 0.7 is about 1e-6, so ``EQUAL`` allows ten times that. The float32 error of an inner
product is proportional to the product of the norms, and a sign code decodes to norm
``sqrt(d)``, so the tolerance is multiplied by the largest norm of a decoded vector (at least
1). A control run (the float32 pipeline alone) must stay under a third of ``EQUAL``, so the
tolerance cannot hide a real change. A witness must exceed ``EQUAL`` by another factor of ten.

Two kinds of pipeline are not compared end to end because a fine quantiser can move one level
when a value sits within float32 rounding of a boundary: integer codecs after PCA or after a
rotation. For those the conditional claim is checked in two parts that do not have that
problem: the coordinates the codec receives are the same up to the sign of each component
(continuous check), and the codec is invariant under sign flips and permutations (exact check
on the stored basis).
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from collections import namedtuple
from typing import ClassVar

import numpy as np
import pytest

from optivision.calibration import score_candidate
from optivision.compose import Pipeline
from optivision.core.stages import (
    STAGES,
    AdaptiveMerge,
    BasisInvariance,
    BinaryQuantizer,
    DimensionProjector,
    DocView,
    Float16Quantizer,
    Float32Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RandomProjector,
    RandomPruner,
    Reduction,
    RedundancyPruner,
    RotationProjector,
    StageTraits,
    TokenReducer,
    TruncateProjector,
    pipeline_traits,
    register_traits,
    traits_of,
)
from optivision.core.stages import traits as traits_module
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix

O, S, N, U = (
    BasisInvariance.ORTHOGONAL,
    BasisInvariance.SIGNED_PERMUTATION,
    BasisInvariance.NONE,
    BasisInvariance.UNKNOWN,
)

D = 48
SEEDS = (0, 1, 2)
#: scores are "equal" when they differ by at most this fraction of the per-query score span
EQUAL = 1e-5
#: pipelines with an eigendecomposition: eigenvectors of a matrix built from float32-rounded
#: rotated inputs move by about (rounding noise in the matrix) / (eigenvalue gap); the gaps of
#: the test data are asserted below
EQUAL_EIGEN = 5e-5
#: a witness must move the scores by at least this fraction of the span
WITNESS = 10 * EQUAL
#: "well separated": a relative eigenvalue gap of at least 1% keeps the eigenvector error that
#: float32 noise (about 1e-7 in the second-moment matrix) causes near 1e-5, below EQUAL_EIGEN
MIN_GAP = 0.01


# ------------------------------------------------------------- synthetic data


def make_anisotropic(d=D, n_docs=40, n_queries=12, seed=0, topics=4, doc_rows=None):
    """Unit-norm rows, per-axis lognormal scales, a nonzero mean, topic structure per document."""
    rng = np.random.default_rng(seed)
    scale = np.exp(rng.normal(0.0, 1.0, d))
    mean = 0.6 * scale

    def rows(n):
        centres = rng.standard_normal((topics, d)) * scale * 0.8
        which = rng.integers(0, topics, size=n)
        x = mean + centres[which] + 0.35 * rng.standard_normal((n, d)) * scale
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    counts = doc_rows if doc_rows is not None else rng.integers(10, 31, size=n_docs)
    docs = [rows(int(n)) for n in counts]
    queries = [rows(int(n)) for n in rng.integers(4, 9, size=n_queries)]
    return MultiVectorCorpus.from_arrays(docs), MultiVectorCorpus.from_arrays(queries)


def hierarchical(*args, **kwargs):
    """``HierarchicalMerge`` needs SciPy: the test that builds one is skipped without it."""
    pytest.importorskip("scipy")
    return HierarchicalMerge(*args, **kwargs)


def signed_permutation(d, seed, signed=True):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d)
    signs = rng.integers(0, 2, size=d) * 2 - 1 if signed else np.ones(d, dtype=np.int64)
    return perm, signs


def random_rotation(d, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)


def apply_signed_permutation(corpus, perm, signs):
    """Exact: a gather and sign flips. The result is made C-contiguous: a fancy-indexed array is
    F-ordered, and ``mean(axis=0)`` then sums in another order, which can move a quantisation
    boundary by one float32 rounding."""
    moved = corpus.vectors[:, perm] * signs.astype(np.float32)
    return corpus.with_vectors(np.ascontiguousarray(moved))


def apply_rotation(corpus, g):
    """``x -> x @ g`` in float64, stored as float32 (C-contiguous)."""
    return corpus.with_vectors(
        (np.asarray(corpus.vectors, dtype=np.float64) @ g).astype(np.float32)
    )


def signed_permutation_map(corpus, seed):
    return apply_signed_permutation(corpus, *signed_permutation(corpus.dimension, seed))


def permutation_map(corpus, seed):
    return apply_signed_permutation(
        corpus, *signed_permutation(corpus.dimension, seed, signed=False)
    )


def rotation_map(corpus, seed):
    return apply_rotation(corpus, random_rotation(corpus.dimension, 100 + seed))


def pipeline_scores(factory, docs, queries, fit_queries=False):
    """Scores of a freshly built pipeline: the library's own path (``score_candidate``)."""
    pipe = Pipeline(factory())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Int8Quantizer('fixed') warns when it clips
        if fit_queries:  # no library path passes queries to fit; same steps as score_candidate
            pipe.fit(docs, queries)
            return maxsim_matrix(pipe.transform_queries(queries), pipe.compress(docs))
        return score_candidate(pipe, docs, queries).scores


def noise_scale(factory, docs, queries, fit_queries=False):
    """Largest norm of a decoded vector, at least 1: float32 error scales with the norms."""
    pipe = Pipeline(factory())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if fit_queries:
            pipe.fit(docs, queries)
        compressed = pipe.compress(docs)
    rows = compressed.decode_rows(0, compressed.num_vectors)
    return max(1.0, float(np.linalg.norm(rows, axis=1).max()))


def relative_change(a, b, reference):
    """Largest absolute score difference, per query, over that query's span of ``reference``."""
    span = np.ptp(reference, axis=1)
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).max(axis=1)
    return float(np.max(diff / span))


World = namedtuple("World", "docs queries reference")


@pytest.fixture(scope="module")
def world():
    docs, queries = make_anisotropic()
    reference = maxsim_matrix(queries, docs)
    assert np.ptp(reference, axis=1).min() > 0.5  # the spans the tolerances are quoted against
    return World(docs, queries, reference)


def changes(factory, world, transform, fit_queries=False):
    base = pipeline_scores(factory, world.docs, world.queries, fit_queries)
    return [
        relative_change(
            base,
            pipeline_scores(
                factory, transform(world.docs, s), transform(world.queries, s), fit_queries
            ),
            world.reference,
        )
        for s in SEEDS
    ]


def check_equal(factory, world, transform, tol, fit_queries=False):
    limit = tol * noise_scale(factory, world.docs, world.queries, fit_queries)
    worst = max(changes(factory, world, transform, fit_queries))
    assert worst <= limit, (
        f"{transform.__name__}: scores moved by {worst:.2e} of the span (limit {limit:.1e})"
    )


def check_witness(factory, world, transform, fit_queries=False):
    least = min(changes(factory, world, transform, fit_queries))
    assert least >= WITNESS, (
        f"{transform.__name__}: scores moved only {least:.2e} of the span (need {WITNESS:.0e})"
    )


def assert_reduces(factory, world):
    pipe = Pipeline(factory()).fit(world.docs)
    assert pipe.transform(world.docs).num_vectors < world.docs.num_vectors


def eigen_gaps(vectors, k, centre=False, queries=None, weight=0.5):
    """Relative gaps between consecutive eigenvalues of the second-moment matrix PCA uses,
    for the top ``k``: ``(l_i - l_{i+1}) / l_i``. The last entry is the gap at the cut."""
    x = np.asarray(vectors, dtype=np.float64)
    xc = x - x.mean(axis=0) if centre else x
    m = xc.T @ xc / x.shape[0]
    if queries is not None:
        q = np.asarray(queries, dtype=np.float64)
        m = (1.0 - weight) * m + weight * (q.T @ q / q.shape[0])
    vals = np.sort(np.linalg.eigvalsh(m))[::-1]
    return (vals[:k] - vals[1 : k + 1]) / vals[:k]


def check_canonical_coordinates(factory, world):
    """What reaches the codec is the same up to the sign of each column, for rotated data too."""

    def coordinates(corpus):
        stages = factory()
        return Pipeline(stages).fit(corpus).transform(corpus).vectors.astype(np.float64)

    y = coordinates(world.docs)
    smallest, worst = np.abs(y).min(), 0.0
    for seed in SEEDS:
        y2 = coordinates(rotation_map(world.docs, seed))
        assert y2.shape == y.shape
        signs = np.where((y * y2).sum(axis=0) < 0, -1.0, 1.0)
        for j in range(y.shape[1]):
            assert np.linalg.norm(y[:, j] - signs[j] * y2[:, j]) <= 1e-4 * np.linalg.norm(y[:, j])
        worst = max(worst, np.abs(y - signs * y2).max())
    return smallest, worst


# ------------------------------------------------------- the cases per level

ORTHOGONAL_CASES = {
    "float32": lambda: [Float32Quantizer()],
    "no stages": list,
    "redundancy": lambda: [RedundancyPruner(0.9)],
    "redundancy(merge=False)": lambda: [RedundancyPruner(0.9, merge=False)],
    "adaptive_merge": lambda: [AdaptiveMerge(radius=0.9)],
    "adaptive_merge(center)": lambda: [AdaptiveMerge(radius=0.9, center="mean")],
    "adaptive_merge(representative=center)": lambda: [
        AdaptiveMerge(radius=0.9, representative="center")
    ],
    "adaptive_merge(refine)": lambda: [AdaptiveMerge(radius=0.9, refine=2)],
    "adaptive_merge(ratio)": lambda: [AdaptiveMerge(radius=None, ratio=0.5)],
    "hierarchical_merge(ratio)": lambda: [hierarchical(ratio=0.5)],
    "hierarchical_merge(distance)": lambda: [hierarchical(ratio=None, max_distance=0.9)],
    "random_prune": lambda: [RandomPruner(0.5, seed=3)],
    "rotation": lambda: [RotationProjector(seed=5)],
    "merge>rotation": lambda: [AdaptiveMerge(radius=0.9), RotationProjector(seed=2)],
}
EIGEN_CASES = {  # ORTHOGONAL, but through an eigendecomposition
    "pca": lambda: [PCAProjector(16)],
    "pca(center)": lambda: [PCAProjector(16, center=True)],
    "project(pca)": lambda: [DimensionProjector(16, method="pca")],
    "merge>pca": lambda: [AdaptiveMerge(radius=0.9, center="mean"), PCAProjector(16)],
    "pca>merge": lambda: [PCAProjector(16), AdaptiveMerge(radius=0.9)],
}
SIGNED_PERMUTATION_CASES = {
    "float16": lambda: [Float16Quantizer()],
    "int8(fixed)": lambda: [Int8Quantizer("fixed")],
    "int8(fixed,center)": lambda: [Int8Quantizer("fixed", center="mean")],
    "int8(per_vector)": lambda: [Int8Quantizer("per_vector")],
    "int8(per_vector,center)": lambda: [Int8Quantizer("per_vector", center="mean")],
    "int8(per_dimension)": lambda: [Int8Quantizer("per_dimension")],
    "int8(per_dimension,center)": lambda: [Int8Quantizer("per_dimension", center="mean")],
    "int4": lambda: [Int4Quantizer()],
    "int4(center)": lambda: [Int4Quantizer(center="mean")],
    "binary": lambda: [BinaryQuantizer()],
    "binary(center)": lambda: [BinaryQuantizer(center="mean")],
    "merge>binary": lambda: [AdaptiveMerge(radius=0.9), BinaryQuantizer()],
    # rows kept verbatim: a pooled row is rescaled by a float32 norm whose rounding depends on
    # the axis order, which can move a fine quantiser by one level
    "merge(center rows)>int8(per_dimension,center)": lambda: [
        AdaptiveMerge(radius=0.9, representative="center"),
        Int8Quantizer("per_dimension", center="mean"),
    ],
}
NONE_CASES = {
    "truncate": lambda: [TruncateProjector(16)],
    "random_projection": lambda: [RandomProjector(16, seed=2)],
    "project(random)": lambda: [DimensionProjector(16, method="random")],
    "project(truncate)": lambda: [DimensionProjector(16, method="truncate")],
    "lloyd2": lambda: [Lloyd2Quantizer(seed=7)],
    "rotation>binary": lambda: [RotationProjector(1), BinaryQuantizer()],
    "rotation>int8(per_vector)": lambda: [RotationProjector(1), Int8Quantizer("per_vector")],
    "random_projection>binary": lambda: [RandomProjector(32, seed=1), BinaryQuantizer()],
    "truncate>binary": lambda: [TruncateProjector(32), BinaryQuantizer()],
}


class TestControl:
    def test_float32_noise_stays_well_under_the_tolerance(self, world):
        for factory in (ORTHOGONAL_CASES["float32"], ORTHOGONAL_CASES["no stages"]):
            for transform in (signed_permutation_map, rotation_map):
                worst = max(changes(factory, world, transform))
                assert worst <= EQUAL / 3, (transform.__name__, worst)

    def test_the_test_maps_do_what_they_say(self, world):
        g = random_rotation(D, 5)
        assert np.abs(g @ g.T - np.eye(D)).max() < 1e-12
        moved_scores = maxsim_matrix(
            apply_rotation(world.queries, g), apply_rotation(world.docs, g)
        )
        assert relative_change(world.reference, moved_scores, world.reference) <= EQUAL / 3
        perm, signs = signed_permutation(D, 0)
        moved = apply_signed_permutation(world.docs, perm, signs)
        assert moved.vectors.flags["C_CONTIGUOUS"]
        assert np.array_equal(np.abs(moved.vectors), np.abs(world.docs.vectors)[:, perm])

    def test_the_corpus_is_anisotropic_with_a_nonzero_mean(self, world):
        x = world.docs.vectors
        assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
        assert x.std(axis=0).max() / x.std(axis=0).min() > 4
        assert np.linalg.norm(x.mean(axis=0)) > 0.3
        assert (x == 0).sum() == 0  # exact zeros would void the sign-code statement


class TestTheChecksCanFail:
    """A wrong label must be caught: the checks used below reject stages that do not fit them."""

    def test_the_equality_check_rejects_a_stage_that_is_not_invariant(self, world):
        with pytest.raises(AssertionError, match="scores moved"):
            check_equal(NONE_CASES["truncate"], world, permutation_map, EQUAL)
        with pytest.raises(AssertionError, match="scores moved"):
            check_equal(SIGNED_PERMUTATION_CASES["binary"], world, rotation_map, EQUAL)
        with pytest.raises(AssertionError, match="scores moved"):
            check_equal(SIGNED_PERMUTATION_CASES["float16"], world, rotation_map, EQUAL)
        with pytest.raises(AssertionError, match="scores moved"):
            check_equal(NONE_CASES["lloyd2"], world, signed_permutation_map, EQUAL)

    def test_the_witness_check_rejects_a_stage_that_is_invariant(self, world):
        with pytest.raises(AssertionError, match="scores moved only"):
            check_witness(ORTHOGONAL_CASES["redundancy"], world, rotation_map)
        with pytest.raises(AssertionError, match="scores moved only"):
            check_witness(SIGNED_PERMUTATION_CASES["binary"], world, signed_permutation_map)
        with pytest.raises(AssertionError, match="scores moved only"):
            check_witness(ORTHOGONAL_CASES["rotation"], world, signed_permutation_map)

    def test_the_conditional_check_rejects_a_pipeline_that_is_not_canonical(self, world):
        rotation_then_pca = RULE_CASES_BY_ID["pca>rotation>binary"]
        with pytest.raises(AssertionError):
            check_canonical_coordinates(lambda: rotation_then_pca()[:-1], world)


class TestOrthogonal:
    @pytest.mark.parametrize("factory", ORTHOGONAL_CASES.values(), ids=list(ORTHOGONAL_CASES))
    def test_scores_are_unchanged_by_any_orthogonal_map(self, world, factory):
        traits = pipeline_traits(Pipeline(factory()))
        assert traits.basis_invariance is O and not traits.basis_sensitive
        if any(isinstance(s, TokenReducer) for s in factory()):
            assert_reduces(factory, world)  # the equality is not the equality of a no-op
        check_equal(factory, world, rotation_map, EQUAL)
        check_equal(factory, world, signed_permutation_map, EQUAL)

    @pytest.mark.parametrize("factory", EIGEN_CASES.values(), ids=list(EIGEN_CASES))
    def test_scores_through_an_eigendecomposition(self, world, factory):
        traits = pipeline_traits(Pipeline(factory()))
        assert traits.basis_invariance is O and not traits.basis_sensitive
        check_equal(factory, world, rotation_map, EQUAL_EIGEN)
        check_equal(factory, world, signed_permutation_map, EQUAL_EIGEN)

    def test_pca_joint_is_invariant_when_documents_and_queries_move_together(self, world):
        def factory():
            return [PCAProjector(16, basis="joint")]

        assert traits_of(factory()[0]).fit_uses_queries
        assert eigen_gaps(world.docs.vectors, 16, queries=world.queries.vectors).min() > MIN_GAP
        check_equal(factory, world, rotation_map, EQUAL_EIGEN, fit_queries=True)

    def test_importance_seeding_is_invariant_too(self, world):
        # the seeds come from a per-row importance signal, which does not move with the basis
        importance = np.random.default_rng(0).random(world.docs.num_vectors).astype(np.float32)
        docs = MultiVectorCorpus(world.docs.vectors, world.docs.offsets, importance=importance)
        weighted = World(docs, world.queries, world.reference)

        def factory():
            return [AdaptiveMerge(radius=0.9, seeding="importance", n_importance_seeds=3)]

        assert_reduces(factory, weighted)
        check_equal(factory, weighted, rotation_map, EQUAL)
        check_equal(factory, weighted, signed_permutation_map, EQUAL)

    def test_the_wide_ward_path_is_invariant_too(self):
        # from 512 dimensions up the Ward input is the Gram-derived distance matrix
        docs, queries = make_anisotropic(d=512, n_docs=12, n_queries=4, seed=2)
        wide = World(docs, queries, maxsim_matrix(queries, docs))

        def factory():
            return [hierarchical(ratio=0.5)]

        assert_reduces(factory, wide)
        check_equal(factory, wide, rotation_map, EQUAL)
        check_equal(factory, wide, signed_permutation_map, EQUAL)

    def test_a_square_random_projection_keeps_the_scores_but_not_the_codec_basis(self, world):
        # dim >= input dimension: the matrix is orthogonal, so the stage alone is an isometry
        def square():
            return [RandomProjector(D, seed=3)]

        for transform in (permutation_map, rotation_map):
            check_equal(square, world, transform, EQUAL)
        assert traits_of(square()[0]).basis_invariance is N  # declared conservatively
        check_witness(lambda: [*square(), BinaryQuantizer()], world, permutation_map)


class TestSignedPermutation:
    @pytest.mark.parametrize(
        "factory", SIGNED_PERMUTATION_CASES.values(), ids=list(SIGNED_PERMUTATION_CASES)
    )
    def test_equal_under_signed_permutations_and_not_under_rotations(self, world, factory):
        traits = pipeline_traits(Pipeline(factory()))
        assert traits.basis_invariance is S and traits.basis_sensitive
        check_equal(factory, world, signed_permutation_map, EQUAL)
        check_witness(factory, world, rotation_map)

    def test_float16_is_bounded_by_its_rounding_but_not_orthogonal(self, world):
        # half precision keeps 11 significand bits: relative error 2**-11 per coordinate, so a
        # vector moves by at most 2**-11 of its norm and a score by at most 2 * 2**-11 * n_tokens
        factory = SIGNED_PERMUTATION_CASES["float16"]
        base = pipeline_scores(factory, world.docs, world.queries)
        rotated = pipeline_scores(
            factory, rotation_map(world.docs, 0), rotation_map(world.queries, 0)
        )
        bound = 2 * 2.0**-11 * np.diff(world.queries.offsets)
        assert (np.abs(base - rotated).max(axis=1) <= bound).all()
        assert relative_change(base, rotated, world.reference) >= WITNESS

    def test_exact_zeros_void_the_sign_code_statement(self):
        # a coordinate that is exactly 0 is coded as -1 whichever sign convention is used, so a
        # sign flip is not a symmetry of the code there (the note of the binary stage says so)
        rng = np.random.default_rng(0)
        flips = np.where(np.arange(16) < 8, -1.0, 1.0).astype(np.float32)

        def flipped(corpus):
            return corpus.with_vectors(np.ascontiguousarray(corpus.vectors * flips))

        def scores_change(x):
            docs = MultiVectorCorpus(x, np.arange(0, x.shape[0] + 1, 20))
            q = rng.standard_normal((12, 16)).astype(np.float32)
            queries = MultiVectorCorpus(q, np.arange(0, 13, 3))
            a = pipeline_scores(lambda: [BinaryQuantizer()], docs, queries)
            b = pipeline_scores(lambda: [BinaryQuantizer()], flipped(docs), flipped(queries))
            return float(np.abs(a - b).max())

        dense = rng.standard_normal((200, 16)).astype(np.float32)
        sparse = dense.copy()
        sparse[np.abs(sparse) < 0.7] = 0.0
        assert (dense == 0).sum() == 0 and (sparse == 0).mean() > 0.3
        assert scores_change(dense) < 1e-4
        assert scores_change(sparse) > 0.1


class TestNone:
    @pytest.mark.parametrize("factory", NONE_CASES.values(), ids=list(NONE_CASES))
    def test_a_plain_permutation_of_the_axes_changes_the_scores(self, world, factory):
        traits = pipeline_traits(Pipeline(factory()))
        assert traits.basis_invariance is N and traits.basis_sensitive
        check_witness(factory, world, permutation_map)

    @pytest.mark.parametrize("name", ["random_projection", "lloyd2", "rotation>binary"])
    def test_randomised_realisations_change_under_a_signed_permutation(self, world, name):
        # the justification for "after a seeded random map, SIGNED_PERMUTATION counts as NONE"
        check_witness(NONE_CASES[name], world, signed_permutation_map)

    def test_a_random_map_is_what_turns_a_codec_basis_dependent(self, world):
        check_equal(SIGNED_PERMUTATION_CASES["binary"], world, signed_permutation_map, EQUAL)
        check_witness(NONE_CASES["rotation>binary"], world, signed_permutation_map)
        # the rotation stage alone is an isometry: the same scores
        check_equal(ORTHOGONAL_CASES["rotation"], world, signed_permutation_map, EQUAL)

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda seed: [RandomProjector(16, seed)], id="random_projection"),
            pytest.param(
                lambda seed: [RotationProjector(seed), BinaryQuantizer()], id="rotation>binary"
            ),
            pytest.param(lambda seed: [Lloyd2Quantizer(seed)], id="lloyd2"),
        ],
    )
    def test_the_distribution_over_seeds_does_not_depend_on_the_basis(self, world, make):
        """Same law over seeds in both bases: compare the mean of one statistic over 40 seeds."""
        rotated_docs, rotated_queries = rotation_map(world.docs, 0), rotation_map(world.queries, 0)

        def statistic(docs, queries, seed):
            scores = pipeline_scores(lambda: make(seed), docs, queries)
            return relative_change(scores, world.reference, world.reference)

        seeds = range(100, 140)
        a = np.array([statistic(world.docs, world.queries, s) for s in seeds])
        b = np.array([statistic(rotated_docs, rotated_queries, s) for s in seeds])
        se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        assert abs(a.mean() - b.mean()) < 4 * se


RULE_CASES = [
    # id, factory, invariance, conditionally_invariant
    ("empty", list, O, False),
    ("float32", lambda: [Float32Quantizer()], O, False),
    ("merge>float32", lambda: [AdaptiveMerge(radius=0.9), Float32Quantizer()], O, False),
    ("float16", lambda: [Float16Quantizer()], S, False),
    ("merge>binary", lambda: [AdaptiveMerge(radius=0.9), BinaryQuantizer()], S, False),
    (
        "merge(center)>int8(per_dimension,center)",
        lambda: [
            AdaptiveMerge(radius=0.9, center="mean", representative="center"),
            Int8Quantizer("per_dimension", center="mean"),
        ],
        S,
        False,
    ),
    ("pca>binary", lambda: [PCAProjector(16), BinaryQuantizer()], S, True),
    (
        "pca>int8(per_dimension)",
        lambda: [PCAProjector(16), Int8Quantizer("per_dimension")],
        S,
        True,
    ),
    (
        "pca(center)>int4(center)",
        lambda: [PCAProjector(16, center=True), Int4Quantizer(center="mean")],
        S,
        True,
    ),
    (
        "merge>pca>int4",
        lambda: [AdaptiveMerge(radius=0.9), PCAProjector(16), Int4Quantizer()],
        S,
        True,
    ),
    (
        "pca>merge>binary",
        lambda: [PCAProjector(16), AdaptiveMerge(radius=0.9), BinaryQuantizer()],
        S,
        True,
    ),
    ("rotation", lambda: [RotationProjector(2)], O, False),
    ("rotation>float32", lambda: [RotationProjector(2), Float32Quantizer()], O, False),
    ("rotation>binary", lambda: [RotationProjector(2), BinaryQuantizer()], N, False),
    (
        "rotation>pca>binary",
        lambda: [RotationProjector(2), PCAProjector(16), BinaryQuantizer()],
        N,
        True,
    ),
    (
        "pca>rotation>binary",
        lambda: [PCAProjector(16), RotationProjector(2), BinaryQuantizer()],
        N,
        False,
    ),
    ("pca>lloyd2", lambda: [PCAProjector(16), Lloyd2Quantizer()], N, False),
    ("truncate", lambda: [TruncateProjector(16)], N, False),
    (
        "truncate>pca>binary",
        lambda: [TruncateProjector(32), PCAProjector(16), BinaryQuantizer()],
        N,
        False,
    ),
    (
        "random_projection>pca>binary",
        lambda: [RandomProjector(32, seed=1), PCAProjector(16), BinaryQuantizer()],
        N,
        False,
    ),
    ("lloyd2", lambda: [Lloyd2Quantizer()], N, False),
]


RULE_CASES_BY_ID = {case[0]: case[1] for case in RULE_CASES}


class TestPipelineRule:
    @pytest.mark.parametrize("case", RULE_CASES, ids=[c[0] for c in RULE_CASES])
    def test_rule_and_data_agree(self, world, case):
        _, factory, invariance, conditional = case
        traits = pipeline_traits(Pipeline(factory()))
        assert (traits.basis_invariance, traits.conditionally_invariant) == (
            invariance,
            conditional,
        )
        assert traits.basis_sensitive is (invariance is not O)
        assert len(traits.stages) == len(factory())
        uses_eigen = any(isinstance(s, PCAProjector) for s in factory())
        if invariance is O:
            check_equal(factory, world, rotation_map, EQUAL_EIGEN if uses_eigen else EQUAL)
        elif conditional:
            # not invariant by itself; given well-separated principal directions the codec
            # receives coordinates fixed by the data up to sign, so a rotation does not matter
            def before_codec():
                return factory()[:-1]

            smallest, worst = check_canonical_coordinates(before_codec, world)
            if isinstance(factory()[-1], BinaryQuantizer):
                # A sign code differs only if a coordinate lies within the rounding noise of
                # zero. `worst` is measured on the very rotations compared below, so a smallest
                # coordinate above it means no sign changes in these runs; the factor of 3 leaves
                # room for the signed-permutation runs, whose noise is smaller still.
                assert smallest > 3 * worst, (
                    f"smallest |coordinate| {smallest:.1e}, noise {worst:.1e}"
                )
                check_equal(factory, world, rotation_map, EQUAL_EIGEN)
                check_equal(factory, world, signed_permutation_map, EQUAL_EIGEN)
        elif invariance is S:
            check_equal(factory, world, signed_permutation_map, EQUAL)
            check_witness(factory, world, rotation_map)
        else:
            check_witness(factory, world, permutation_map)
            check_witness(factory, world, rotation_map)

    def test_canonicalisation_is_recorded_but_not_credited(self):
        t = pipeline_traits(Pipeline([PCAProjector(16), BinaryQuantizer()]))
        assert (
            t.basis_invariance is S
        )  # unconditional value: binary is only signed-permutation invariant
        assert t.conditionally_invariant and t.basis_sensitive
        assert any("conditionally invariant" in note for note in t.notes)
        assert [s.canonicalises_basis for s in t.stages] == [True, False]

    def test_a_stage_that_is_not_sign_invariant_after_pca_is_not_credited(self):
        # PCA fixes the coordinates only up to the sign of each component; lloyd2 and a later
        # rotation are not invariant under sign flips of their input, so the sign leaks through
        assert not pipeline_traits(
            Pipeline([PCAProjector(16), Lloyd2Quantizer()])
        ).conditionally_invariant
        assert not pipeline_traits(
            Pipeline([PCAProjector(16), RotationProjector(), BinaryQuantizer()])
        ).conditionally_invariant

    def test_sensitivity_flags_stochastic_and_fitted_state(self):
        t = pipeline_traits(Pipeline([AdaptiveMerge(radius=0.9), BinaryQuantizer()]))
        assert (t.has_fitted_state, t.stochastic) == (False, False)
        t = pipeline_traits(Pipeline([AdaptiveMerge(radius=0.9, center="mean"), BinaryQuantizer()]))
        assert t.has_fitted_state and t.stochastic
        assert pipeline_traits(Pipeline([RotationProjector()])).stochastic

    def test_notes_say_why(self):
        t = pipeline_traits(Pipeline([RotationProjector(), BinaryQuantizer()]))
        assert any("binary" in n and "NONE" in n for n in t.notes)
        t = pipeline_traits(Pipeline([TruncateProjector(8)]))
        assert any("axis order" in n for n in t.notes)

    def test_a_list_of_stages_is_accepted_and_to_dict_is_json(self):
        t = pipeline_traits([AdaptiveMerge(radius=0.9), BinaryQuantizer()])
        d = json.loads(json.dumps(t.to_dict()))
        assert d["basis_invariance"] == "signed_permutation" and d["basis_sensitive"] is True
        assert [s["name"] for s in d["stages"]] == ["adaptive_merge", "binary"]
        with pytest.raises(TypeError):
            pipeline_traits("binary")


class TestPcaCondition:
    def test_the_test_data_have_well_separated_eigenvalues(self, world):
        assert eigen_gaps(world.docs.vectors, 16).min() > MIN_GAP  # distinct, with a gap at the cut
        assert eigen_gaps(world.docs.vectors, 16, centre=True).min() > MIN_GAP

    def test_a_coordinate_cut_does_not_have_that_property(self, world):
        rotated = rotation_map(world.docs, 0)
        y = Pipeline([TruncateProjector(16)]).fit(world.docs).transform(world.docs).vectors
        y2 = Pipeline([TruncateProjector(16)]).fit(rotated).transform(rotated).vectors
        assert min(np.linalg.norm(y - y2), np.linalg.norm(y + y2)) > 0.5 * np.linalg.norm(y)

    @staticmethod
    def _isotropic():
        """Every signed basis vector the same number of times: the second-moment matrix is
        exactly I/d, all eigenvalues are equal and the retained directions are arbitrary."""
        d, repeats = D, 6
        rng = np.random.default_rng(0)
        rows = np.concatenate([np.eye(d), -np.eye(d)] * repeats).astype(np.float32)
        rows = rows[rng.permutation(rows.shape[0])]
        sizes, left = [], rows.shape[0]
        while left:
            n = min(left, int(rng.integers(10, 31)))
            sizes.append(n)
            left -= n
        docs = MultiVectorCorpus(rows, np.concatenate([[0], np.cumsum(sizes)]))
        q = rng.standard_normal((40, d))
        q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
        return docs, MultiVectorCorpus(q, np.arange(0, 41, 4))

    def test_witness_with_exactly_isotropic_data(self):
        docs, queries = self._isotropic()
        x = docs.vectors.astype(np.float64)
        assert np.array_equal(x.T @ x / x.shape[0], np.eye(D) / D)  # exactly isotropic
        reference = maxsim_matrix(queries, docs)
        assert np.ptp(reference, axis=1).min() > 0.05

        def cut():
            return [PCAProjector(16)]

        base = pipeline_scores(cut, docs, queries)
        moved = [
            pipeline_scores(cut, rotation_map(docs, s), rotation_map(queries, s)) for s in SEEDS
        ]
        assert min(relative_change(base, m, reference) for m in moved) > 0.1  # far above rounding

        def everything():
            return [PCAProjector(D)]

        # the cause is a cut inside a tied group, not the data: keeping every direction is invariant
        base = pipeline_scores(everything, docs, queries)
        for s in SEEDS:
            moved = pipeline_scores(everything, rotation_map(docs, s), rotation_map(queries, s))
            assert relative_change(base, moved, reference) <= EQUAL_EIGEN

    def test_the_conditional_claim_fails_without_the_condition(self):
        docs, queries = self._isotropic()
        reference = maxsim_matrix(queries, docs)

        def factory():
            return [PCAProjector(16), BinaryQuantizer()]

        assert pipeline_traits(
            Pipeline(factory())
        ).conditionally_invariant  # the label does not see the data
        base = pipeline_scores(factory, docs, queries)
        moved = pipeline_scores(factory, rotation_map(docs, 0), rotation_map(queries, 0))
        assert relative_change(base, moved, reference) > 0.1


# ------------------------------------------------------------------ the conditions


def stage_conditions(stages):
    """The pipeline's conditions that start with the name of ``stage``."""
    traits = pipeline_traits(Pipeline(stages))
    return traits, {t.name: [c for c in traits.conditions if c.startswith(f"{t.name}: ")]
                    for t in traits.stages}


def orthonormal_rows(d=16, n_docs=20, rows=4, seed=0):
    """Documents whose rows are distinct standard basis vectors: every pair is exactly
    orthogonal (cosine exactly 0, Euclidean distance exactly sqrt(2)), so every similarity the
    merge and prune stages compare is an exact tie. Queries are generic."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n_docs):
        block = np.zeros((rows, d), dtype=np.float32)
        block[np.arange(rows), rng.choice(d, rows, replace=False)] = 1.0
        docs.append(block)
    q = rng.standard_normal((10, 4, d))
    q = (q / np.linalg.norm(q, axis=2, keepdims=True)).astype(np.float32)
    return MultiVectorCorpus.from_arrays(docs), MultiVectorCorpus.from_arrays(list(q))


#: a stage that is ORTHOGONAL on generic data and what breaks it: (id, stages that tie)
TIE_CASES = [
    pytest.param(lambda: [hierarchical(ratio=0.5)], "hierarchical_merge", "Ward", id="ward"),
    pytest.param(
        lambda: [AdaptiveMerge(radius=None, ratio=0.5)], "adaptive_merge", "exact ties",
        id="adaptive_merge(ratio)",
    ),
    pytest.param(
        lambda: [AdaptiveMerge(radius=0.0)], "adaptive_merge", "radius",
        id="adaptive_merge(radius at the tie)",
    ),
    pytest.param(
        lambda: [RedundancyPruner(0.0)], "redundancy", "threshold",
        id="redundancy(threshold at the tie)",
    ),
]


class TestTheConditionsAreReportedWhereTheDeclarationBreaks:
    """The audit's counterexamples (RI-07): data on which a declared invariance fails. Each
    test shows that the scores move there and that the pipeline lists the condition."""

    def test_a_tied_pca_cut_moves_the_scores_and_the_pipeline_reports_the_condition(self):
        docs, queries = TestPcaCondition._isotropic()  # every eigenvalue equal: the cut is a tie
        reference = maxsim_matrix(queries, docs)

        def cut():
            return [PCAProjector(16)]

        base = pipeline_scores(cut, docs, queries)
        moved = [
            pipeline_scores(cut, rotation_map(docs, s), rotation_map(queries, s)) for s in SEEDS
        ]
        assert min(relative_change(base, m, reference) for m in moved) > 0.1
        traits, by_stage = stage_conditions(cut())
        assert traits.basis_invariance is O and traits.basis_sensitive is False  # as declared
        assert not traits.unconditional and traits.conditions == tuple(by_stage["pca"])
        assert any("eigenvalue gap at the cut" in c and "tie" in c for c in traits.conditions)

    def test_tied_retained_eigenvalues_break_a_codec_after_pca_but_not_pca_alone(self):
        """Eight equal eigenvalues kept, a clear gap below them: the retained subspace is fixed
        but its coordinates are an arbitrary rotation, which a sign code then sees. The rows
        are columns of a Hadamard matrix, so the tie is exact and no coordinate is zero."""
        d, kept = D, 8
        rng = np.random.default_rng(1)
        hadamard = np.ones((1, 1))
        for _ in range(8):
            hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])  # 256 x 256
        scale = np.where(np.arange(d) < kept, 1.0, 0.5)  # second moments 1 (x8) and 0.25 (x40)
        rows = (hadamard[:, :d] * scale)[rng.permutation(256)].astype(np.float32)
        x = rows.astype(np.float64)
        assert np.array_equal(x.T @ x / 256, np.diag(scale**2))  # eight equal moments, a gap
        assert (rows != 0).all()
        docs = MultiVectorCorpus(rows, np.arange(0, 257, 16))
        q = rng.standard_normal((40, d))
        queries = MultiVectorCorpus(
            (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32), np.arange(0, 41, 4)
        )
        reference = maxsim_matrix(queries, docs)
        assert np.ptp(reference, axis=1).min() > 0.05

        def alone():
            return [PCAProjector(kept)]

        def with_codec():
            return [PCAProjector(kept), BinaryQuantizer()]

        base_alone, base_codec = (pipeline_scores(f, docs, queries) for f in (alone, with_codec))
        for s in SEEDS:
            g_docs, g_queries = rotation_map(docs, s), rotation_map(queries, s)
            moved_alone = pipeline_scores(alone, g_docs, g_queries)
            moved_codec = pipeline_scores(with_codec, g_docs, g_queries)
            assert relative_change(base_alone, moved_alone, reference) <= EQUAL_EIGEN
            assert relative_change(base_codec, moved_codec, reference) > 0.1
        traits, _ = stage_conditions(with_codec())
        assert traits.conditionally_invariant and not traits.unconditional
        assert any("distinct from one another" in c and c.startswith("pca: ")
                   for c in traits.conditions)

    @pytest.mark.parametrize(("factory", "name", "key"), TIE_CASES)
    def test_exact_ties_move_the_scores_of_the_merge_and_prune_stages_and_are_reported(
        self, factory, name, key
    ):
        """Rows that are exactly equidistant (orthonormal rows) are a tie that row order
        resolves in the stored basis and float32 rounding resolves otherwise."""
        docs, queries = orthonormal_rows()
        reference = maxsim_matrix(queries, docs)
        assert np.ptp(reference, axis=1).min() > 0.1
        check_witness(factory, World(docs, queries, reference), rotation_map)
        traits, by_stage = stage_conditions(factory())
        assert traits.basis_invariance is O and traits.basis_sensitive is False  # as declared
        assert not traits.unconditional and by_stage[name]
        assert any(key in c for c in by_stage[name]), (key, by_stage[name])

    @pytest.mark.parametrize("centre", [None, "mean"])
    def test_a_zero_coordinate_moves_the_scores_of_a_sign_code_and_is_reported(self, centre):
        d = 16
        rng = np.random.default_rng(3)
        flips = np.where(np.arange(d) < d // 2, -1.0, 1.0).astype(np.float32)

        def flipped(corpus):
            return corpus.with_vectors(np.ascontiguousarray(corpus.vectors * flips))

        def move(rows):
            docs = MultiVectorCorpus(rows, np.arange(0, rows.shape[0] + 1, 20))
            q = rng.standard_normal((12, d)).astype(np.float32)
            queries = MultiVectorCorpus(q, np.arange(0, 13, 3))

            def make():
                return [BinaryQuantizer(center=centre)]

            a = pipeline_scores(make, docs, queries)
            b = pipeline_scores(make, flipped(docs), flipped(queries))
            return float(np.abs(a - b).max())

        dense = rng.standard_normal((200, d)).astype(np.float32)
        degenerate = dense.copy()
        # a coordinate that equals the (fitted) mean exactly: 0 without centring, 0.5 with it
        degenerate[:, 0] = 0.0 if centre is None else 0.5
        assert move(dense) < 1e-4 and move(degenerate) > 0.1
        traits, by_stage = stage_conditions([BinaryQuantizer(center=centre)])
        assert traits.basis_invariance is S and traits.basis_sensitive is True
        assert not traits.unconditional and len(by_stage["binary"]) == 1
        assert "exactly zero" in by_stage["binary"][0] and "center='mean'" in by_stage["binary"][0]


class TestConditionsOfPipelines:
    def test_the_conditions_are_the_union_in_stage_order_prefixed_by_the_stage_name(self):
        stages = [HierarchicalMerge(ratio=0.5), PCAProjector(8), BinaryQuantizer()]
        traits = pipeline_traits(Pipeline(stages))
        expected = tuple(f"{t.name}: {c}" for t in traits.stages for c in t.conditions)
        assert traits.conditions == expected and len(expected) == 4
        assert [c.split(":")[0] for c in expected] == [
            "hierarchical_merge", "pca", "pca", "binary"
        ]
        reversed_ = pipeline_traits(Pipeline([PCAProjector(8), HierarchicalMerge(ratio=0.5)]))
        assert [c.split(":")[0] for c in reversed_.conditions] == [
            "pca", "pca", "hierarchical_merge"
        ]

    def test_unconditional_is_true_only_when_there_are_no_conditions(self):
        for stages in (
            [], [Float32Quantizer()], [Float16Quantizer()], [Int8Quantizer("per_vector")],
            [RotationProjector(1), Int4Quantizer()], [TruncateProjector(8)], [Lloyd2Quantizer()],
            [RandomPruner(0.5), Int8Quantizer("per_dimension")],
        ):
            traits = pipeline_traits(Pipeline(stages))
            assert traits.conditions == () and traits.unconditional, stages
        for stages in (
            [BinaryQuantizer()], [PCAProjector(8)], [HierarchicalMerge(ratio=0.5)],
            [AdaptiveMerge(radius=0.9), Float32Quantizer()], [RedundancyPruner(0.9)],
            [DimensionProjector(8, "pca")],
        ):
            traits = pipeline_traits(Pipeline(stages))
            assert traits.conditions and not traits.unconditional, stages
        assert pipeline_traits(Pipeline([DimensionProjector(8, "random")])).unconditional

    def test_an_orthogonal_pipeline_with_conditions_is_a_conditional_statement(self):
        traits = pipeline_traits(Pipeline([AdaptiveMerge(radius=0.9), Float32Quantizer()]))
        assert traits.basis_invariance is O and traits.basis_sensitive is False
        assert not traits.unconditional  # "not sensitive" holds for data that meet the conditions
        sensitive = pipeline_traits(Pipeline([TruncateProjector(8)]))
        assert sensitive.basis_sensitive and sensitive.unconditional  # the cautious statement

    def test_the_new_keys_are_added_and_no_existing_key_is_removed_or_renamed(self):
        traits = pipeline_traits(Pipeline([PCAProjector(8), BinaryQuantizer()]))
        d = json.loads(json.dumps(traits.to_dict()))
        assert {
            "stages", "basis_invariance", "conditionally_invariant", "basis_sensitive",
            "has_fitted_state", "stochastic", "notes",
        } <= set(d)
        assert d["conditions"] == list(traits.conditions) and d["unconditional"] is False
        assert d["all_library_declared"] is True
        for stage in d["stages"]:
            assert {"name", "kind", "basis_invariance", "canonicalises_basis", "randomised_basis",
                    "reads_vector_values", "has_fitted_state", "fit_uses_queries", "stochastic",
                    "lossless", "notes"} <= set(stage)
            assert isinstance(stage["conditions"], list) and stage["declared_by"] == "library"
        assert d["stages"][0]["conditions"] == [
            c.removeprefix("pca: ") for c in traits.conditions if c.startswith("pca: ")
        ]

    def test_the_basis_sensitive_definition_is_unchanged_by_conditions(self):
        for stages in ([PCAProjector(8)], [BinaryQuantizer()], [RedundancyPruner(0.9)], []):
            traits = pipeline_traits(Pipeline(stages))
            assert traits.basis_sensitive is (traits.basis_invariance is not O)


# ------------------------------------------------------------ the declarations

_GENERIC = {
    # name: (kind, invariance, canonicalises, randomised, reads values, fitted state,
    #        fit uses queries, stochastic, lossless): the weakest case over the parameters
    "spatial": ("tokens", O, False, False, False, False, False, False, False),
    "redundancy": ("tokens", O, False, False, True, False, False, False, False),
    "adaptive_merge": ("tokens", O, False, False, True, True, False, True, False),
    "hierarchical_merge": ("tokens", O, False, False, True, False, False, False, False),
    "random_prune": ("tokens", O, False, False, False, False, False, True, False),
    "pca": ("dimensions", O, True, False, True, True, True, True, False),
    "random_projection": ("dimensions", N, False, True, False, True, False, True, False),
    "truncate": ("dimensions", N, False, False, False, False, False, False, False),
    "rotation": ("dimensions", O, False, True, False, True, False, True, False),
    "project": ("dimensions", N, False, True, True, True, True, True, False),
    "float32": ("bits", O, False, False, False, False, False, False, True),
    "float16": ("bits", S, False, False, False, False, False, False, False),
    "int8": ("bits", S, False, False, True, True, False, True, False),
    "int4": ("bits", S, False, False, True, True, False, True, False),
    "binary": ("bits", S, False, False, True, True, False, True, False),
    "lloyd2": ("bits", N, False, True, True, True, False, True, False),
}
_FIELDS = [
    "kind",
    "basis_invariance",
    "canonicalises_basis",
    "randomised_basis",
    "reads_vector_values",
    "has_fitted_state",
    "fit_uses_queries",
    "stochastic",
    "lossless",
]

#: (id, factory, expected fields) for traits that depend on the parameters
PARAMETER_CASES = [
    (
        "adaptive_merge",
        lambda: AdaptiveMerge(radius=0.9),
        {"has_fitted_state": False, "stochastic": False},
    ),
    (
        "adaptive_merge(center)",
        lambda: AdaptiveMerge(radius=0.9, center="mean"),
        {"has_fitted_state": True, "stochastic": True},
    ),
    (
        "int8(fixed)",
        lambda: Int8Quantizer("fixed"),
        {"has_fitted_state": False, "reads_vector_values": False, "stochastic": False},
    ),
    (
        "int8(fixed,center)",
        lambda: Int8Quantizer("fixed", center="mean"),
        {"has_fitted_state": True, "reads_vector_values": True},
    ),
    (
        "int8(per_vector)",
        lambda: Int8Quantizer("per_vector"),
        {"has_fitted_state": False, "reads_vector_values": True},
    ),
    (
        "int8(per_dimension)",
        lambda: Int8Quantizer("per_dimension"),
        {"has_fitted_state": True, "reads_vector_values": True},
    ),
    ("int4", lambda: Int4Quantizer(), {"has_fitted_state": False, "reads_vector_values": True}),
    ("int4(center)", lambda: Int4Quantizer(center="mean"), {"has_fitted_state": True}),
    (
        "binary",
        lambda: BinaryQuantizer(),
        {"has_fitted_state": False, "reads_vector_values": False, "stochastic": False},
    ),
    (
        "binary(center)",
        lambda: BinaryQuantizer(center="mean"),
        {"has_fitted_state": True, "reads_vector_values": True},
    ),
    (
        "pca",
        lambda: PCAProjector(8),
        {"fit_uses_queries": False, "canonicalises_basis": True, "has_fitted_state": True},
    ),
    ("pca(joint)", lambda: PCAProjector(8, basis="joint"), {"fit_uses_queries": True}),
    (
        "project(pca)",
        lambda: DimensionProjector(8, "pca"),
        {"basis_invariance": O, "canonicalises_basis": True, "name": "project"},
    ),
    (
        "project(pca,joint)",
        lambda: DimensionProjector(8, "pca", basis="joint"),
        {"fit_uses_queries": True},
    ),
    (
        "project(random)",
        lambda: DimensionProjector(8, "random"),
        {"basis_invariance": N, "randomised_basis": True, "has_fitted_state": True},
    ),
    (
        "project(truncate)",
        lambda: DimensionProjector(8, "truncate"),
        {
            "basis_invariance": N,
            "randomised_basis": False,
            "has_fitted_state": False,
            "stochastic": False,
        },
    ),
    (
        "rotation",
        lambda: RotationProjector(),
        {"basis_invariance": O, "randomised_basis": True, "canonicalises_basis": False},
    ),
    ("float32", lambda: Float32Quantizer(), {"lossless": True, "basis_invariance": O}),
    ("float16", lambda: Float16Quantizer(), {"lossless": False, "basis_invariance": S}),
]


class TestDeclarations:
    @pytest.mark.parametrize("name", list(_GENERIC))
    def test_bare_name_traits_are_the_weakest_case(self, name):
        t = traits_of(name)
        assert t.name == name and t.notes
        assert tuple(getattr(t, f) for f in _FIELDS) == _GENERIC[name]

    def test_every_registered_stage_of_the_package_has_traits(self):
        mine = {n for n, c in STAGES.items() if c.__module__.startswith("optivision.")}
        assert set(_GENERIC) <= mine
        for name in mine:  # a new stage must declare its traits
            assert traits_of(name).basis_invariance is not U, name

    def test_traits_agree_with_the_stage_kind(self):
        kind = {"tokens": "tokens", "dimension": "dimensions", "quantize": "bits"}
        for name in _GENERIC:
            assert traits_of(name).kind == kind[STAGES[name].kind], name

    @pytest.mark.parametrize("case", PARAMETER_CASES, ids=[c[0] for c in PARAMETER_CASES])
    def test_traits_depend_on_the_parameters(self, case):
        _, make, expected = case
        t = traits_of(make())
        for field, value in expected.items():
            assert getattr(t, field) == value, field

    def test_only_joint_pca_fits_on_queries(self):
        assert {n for n in _GENERIC if traits_of(n).fit_uses_queries} == {"pca", "project"}
        instances = [
            AdaptiveMerge(radius=0.9, center="mean"),
            PCAProjector(8),
            Int8Quantizer("per_dimension"),
            Lloyd2Quantizer(),
            RotationProjector(),
            DimensionProjector(8, "random"),
        ]
        assert not any(traits_of(s).fit_uses_queries for s in instances)
        assert traits_of(PCAProjector(8, basis="joint")).fit_uses_queries

    def test_traits_are_value_objects(self):
        t = traits_of(Int8Quantizer("per_vector"))
        assert t == dataclasses.replace(t) and hash(t) == hash(dataclasses.replace(t))
        d = json.loads(json.dumps(t.to_dict()))
        assert d["basis_invariance"] == "signed_permutation" and d["kind"] == "bits"
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.name = "other"

    def test_traits_validate_their_fields(self):
        with pytest.raises(ValueError):
            StageTraits(name="x", kind="vectors", basis_invariance=O)
        with pytest.raises(ValueError):
            StageTraits(name="", kind="bits", basis_invariance=O)
        assert StageTraits(name="x", kind="bits", basis_invariance="none").basis_invariance is N

    def test_notes_state_the_conditions(self):
        text = {n: traits_of(n).notes for n in _GENERIC}
        assert "sign of each component" in text["pca"] and "distinct" in text["pca"]
        assert "regenerated" in text["lloyd2"] and "distribution" in text["lloyd2"]
        assert "exactly 0" in text["binary"]
        assert "not evaluated by the registered experiments" in text["rotation"]
        assert "does not depend on the basis" in text["random_projection"]
        for n, t in text.items():
            for banned in ("guaranteed", "novel", "state of the art", "universal compression"):
                assert banned not in t.lower(), n


# ----------------------------------------------- stage-level facts behind the flags


def _view(vectors, index=0):
    return DocView(
        index=index,
        vectors=vectors,
        positions=None,
        importance=None,
        weights=np.ones(len(vectors), dtype=np.float32),
        metadata={},
    )


class TestFlagsAgainstBehaviour:
    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda: RandomPruner(0.5, seed=3), id="random_prune"),
            pytest.param(lambda: AdaptiveMerge(radius=0.9), id="adaptive_merge"),
            pytest.param(lambda: RedundancyPruner(0.9), id="redundancy"),
            pytest.param(lambda: hierarchical(ratio=0.5), id="hierarchical_merge"),
        ],
    )
    def test_row_choices_depend_on_values_exactly_where_declared(self, world, make):
        other, _ = make_anisotropic(seed=1, doc_rows=np.diff(world.docs.offsets))
        assert np.array_equal(other.offsets, world.docs.offsets)
        assert not np.array_equal(other.vectors, world.docs.vectors)
        stage = make()
        same = 0
        for i in range(len(world.docs)):
            a = stage.reduce(_view(world.docs[i].vectors.astype(np.float32), i)).labels
            b = stage.reduce(_view(other[i].vectors.astype(np.float32), i)).labels
            same += int(np.array_equal(a, b))
        if traits_of(stage).reads_vector_values:
            assert same < len(world.docs) // 2, stage  # the choice moves with the values
        else:
            assert same == len(world.docs), stage  # the choice does not

    def test_fitted_projections_depend_on_values_exactly_where_declared(self, world):
        other, _ = make_anisotropic(seed=1)
        for make, attr in (
            (lambda: RandomProjector(16, seed=2), "matrix"),
            (lambda: RotationProjector(seed=2), "matrix"),
            (lambda: PCAProjector(16), "components"),
        ):
            a, b = make().fit(world.docs), make().fit(other)
            equal = bool(np.array_equal(getattr(a, attr), getattr(b, attr)))
            assert equal == (not traits_of(a).reads_vector_values), a

    def test_stochastic_stages_follow_their_seed(self, world):
        def out(stage):
            return Pipeline([stage]).fit(world.docs).transform(world.docs).vectors

        for make in (
            lambda s: RandomPruner(0.5, seed=s),
            lambda s: RandomProjector(16, seed=s),
            RotationProjector,
        ):
            assert traits_of(make(0)).stochastic
            assert np.array_equal(out(make(1)), out(make(1)))
            assert not np.array_equal(out(make(1)), out(make(2)))
        a = Pipeline([Lloyd2Quantizer(1)]).compress(world.docs).codes
        assert np.array_equal(a, Pipeline([Lloyd2Quantizer(1)]).compress(world.docs).codes)
        assert not np.array_equal(a, Pipeline([Lloyd2Quantizer(2)]).compress(world.docs).codes)

    def test_lossless_means_the_float32_values_come_back(self, world):
        comp = Pipeline([Float32Quantizer()]).compress(world.docs)
        assert traits_of(Float32Quantizer()).lossless
        assert np.array_equal(comp.decode_rows(0, comp.num_vectors), world.docs.vectors)
        for q in (Float16Quantizer(), Int8Quantizer("per_vector"), BinaryQuantizer()):
            assert not traits_of(q).lossless

    def test_spatial_pruner_never_reads_vector_values(self):
        pytest.importorskip("PIL")
        from optivision.encoders import SyntheticEncoder
        from optivision.stages import SpatialPruner
        from optivision.types import PageRef

        from .conftest import make_page

        enc = SyntheticEncoder(dim=32, grid=16)
        items = [
            (PageRef(f"d{i}", 1), make_page(ink_rows=2 + i % 5, margin=40 + 10 * i))
            for i in range(4)
        ]
        encodings = [enc.encode_pages([img], [ref])[0] for ref, img in items]
        corpus = MultiVectorCorpus.from_page_encodings(encodings, images=[img for _, img in items])
        t = traits_of(SpatialPruner())
        assert (t.basis_invariance, t.reads_vector_values) == (O, False)
        base = Pipeline([SpatialPruner()]).transform(corpus)
        assert base.num_vectors < corpus.num_vectors  # it prunes something
        noise = np.random.default_rng(0).standard_normal(corpus.vectors.shape).astype(np.float32)
        other = Pipeline([SpatialPruner()]).transform(corpus.with_vectors(noise))
        assert np.array_equal(base.offsets, other.offsets)
        assert np.array_equal(base.positions, other.positions, equal_nan=True)
        g = random_rotation(corpus.dimension, 3)
        moved = Pipeline([SpatialPruner()]).transform(apply_rotation(corpus, g))
        assert np.array_equal(base.offsets, moved.offsets)
        assert np.abs(moved.vectors - apply_rotation(base, g).vectors).max() < 1e-5


# ------------------------------------------------- third-party declarations


class _Plain(TokenReducer):
    """A token reducer that keeps every row. It declares nothing."""

    name = "test_plain_reducer"

    def params(self):
        return {}

    def reduce(self, doc):
        return Reduction(doc.vectors.copy(), np.arange(doc.vectors.shape[0], dtype=np.int64))


@pytest.fixture
def clean_traits_registry():
    before = dict(traits_module._REGISTRY)
    yield
    traits_module._REGISTRY.clear()
    traits_module._REGISTRY.update(before)


class TestUnknownAndThirdParty:
    def test_an_undeclared_stage_is_unknown_and_the_pipeline_is_sensitive(self):
        t = traits_of(_Plain())
        assert t.basis_invariance is U and t.kind == "tokens" and t.fit_uses_queries
        p = pipeline_traits(Pipeline([_Plain()]))
        assert p.basis_invariance is U and p.basis_sensitive and not p.conditionally_invariant
        assert any("no declared traits" in n for n in p.notes)

    def test_an_unknown_name_is_not_an_error(self):
        t = traits_of("no_such_stage")
        assert t.basis_invariance is U and t.kind == "unknown" and t.name == "no_such_stage"

    def test_unknown_counts_as_none_in_the_rule(self):
        assert pipeline_traits(Pipeline([_Plain(), Float32Quantizer()])).basis_invariance is U
        assert pipeline_traits(Pipeline([_Plain(), TruncateProjector(8)])).basis_invariance is N
        mixed = pipeline_traits(Pipeline([PCAProjector(8), _Plain(), BinaryQuantizer()]))
        assert mixed.basis_invariance is U and mixed.conditionally_invariant is False

    def test_a_plain_stage_really_is_invariant_but_the_label_does_not_assume_it(self, world):
        check_equal(lambda: [_Plain()], world, rotation_map, EQUAL)  # behaviour
        assert pipeline_traits(Pipeline([_Plain()])).basis_sensitive  # label: nothing is assumed

    def test_register_traits_with_a_value(self, clean_traits_registry):
        declared = StageTraits(
            name=_Plain.name, kind="tokens", basis_invariance=O, reads_vector_values=False
        )
        register_traits(_Plain.name, declared)
        assert traits_of(_Plain()) == declared and traits_of(_Plain.name) == declared
        p = pipeline_traits(Pipeline([_Plain(), Float32Quantizer()]))
        assert p.basis_invariance is O and not p.basis_sensitive
        register_traits(_Plain.name, declared)  # the same value again is a no-op
        with pytest.raises(ValueError, match="already registered"):
            register_traits(_Plain.name, dataclasses.replace(declared, basis_invariance=S))
        with pytest.raises(ValueError, match="declare the name"):
            register_traits("other_name", declared)

    def test_register_traits_with_a_function_of_the_instance(self, clean_traits_registry):
        class Sized(_Plain):
            name = "test_sized_reducer"

            def __init__(self, strong=True):
                self.strong = strong

            def params(self):
                return {"strong": self.strong}

        def declare(stage):
            return StageTraits(
                name=Sized.name, kind="tokens", basis_invariance=O if stage.strong else S
            )

        register_traits(Sized.name, declare)
        assert (
            traits_of(Sized(True)).basis_invariance is O
            and traits_of(Sized(False)).basis_invariance is S
        )
        assert traits_of(Sized.name).basis_invariance is U  # the bare name needs a generic value
        with pytest.raises(TypeError):
            register_traits("x", 3)

    def test_a_generic_value_for_the_bare_name(self, clean_traits_registry):
        weak = StageTraits(name="test_generic", kind="bits", basis_invariance=N)
        register_traits("test_generic", lambda stage: weak, generic=weak)
        assert traits_of("test_generic") == weak

    def test_a_class_can_declare_traits_by_attribute_property_or_method(self):
        t = StageTraits(name="declared", kind="tokens", basis_invariance=S)

        class ByAttribute(_Plain):
            traits = t

        class ByMethod(_Plain):
            def traits(self):
                return t

        class ByProperty(_Plain):
            @property
            def traits(self):
                return t

        for cls in (ByAttribute, ByMethod, ByProperty):
            assert traits_of(cls()) is t
        assert pipeline_traits(Pipeline([ByAttribute(), Float32Quantizer()])).basis_invariance is S

        class Wrong(_Plain):
            traits = "orthogonal"

        with pytest.raises(TypeError, match="StageTraits"):
            traits_of(Wrong())

    def test_a_subclass_does_not_inherit_the_traits_of_its_parent(self):
        class MyMerge(AdaptiveMerge):
            pass

        assert (
            MyMerge().name == "adaptive_merge" and traits_of(AdaptiveMerge()).basis_invariance is O
        )
        t = traits_of(MyMerge(radius=0.9))
        assert t.basis_invariance is U and "not inherited" in t.notes

    def test_traits_of_refuses_other_objects(self):
        with pytest.raises(TypeError):
            traits_of(42)


class TestWhoDeclaredIt:
    """RI-16: a declaration made outside the library's table is recorded as declared."""

    CONDITION_KEYS: ClassVar[dict] = {
        "redundancy": ("threshold",),
        "adaptive_merge": ("exact ties", "radius"),
        "hierarchical_merge": ("Ward", "max_distance"),
        "pca": ("eigenvalue gap at the cut", "distinct"),
        "binary": ("exactly zero",),
    }

    @pytest.mark.parametrize("name", list(_GENERIC))
    def test_the_librarys_own_table_is_declared_by_the_library(self, name):
        t = traits_of(name)
        assert t.declared_by == "library"
        keys = self.CONDITION_KEYS.get(name, ())
        assert bool(t.conditions) == bool(keys)
        for key in keys:
            assert any(key in c for c in t.conditions), (name, key)
        assert all(c.isascii() and "guaranteed" not in c for c in t.conditions)

    def test_a_pipeline_of_library_stages_is_all_library_declared(self):
        t = pipeline_traits(Pipeline([AdaptiveMerge(radius=0.9), BinaryQuantizer()]))
        assert t.all_library_declared and t.to_dict()["all_library_declared"] is True
        assert pipeline_traits(Pipeline([])).all_library_declared

    def test_register_traits_is_third_party_whatever_the_value_says(self, clean_traits_registry):
        forged = StageTraits(
            name=_Plain.name, kind="tokens", basis_invariance=O, declared_by="library"
        )
        register_traits(_Plain.name, forged)
        assert traits_of(_Plain()).declared_by == "third_party"
        assert traits_of(_Plain.name).declared_by == "third_party"
        p = pipeline_traits(Pipeline([_Plain(), Float32Quantizer()]))
        assert p.basis_invariance is O and p.basis_sensitive is False  # as declared
        assert not p.all_library_declared and p.to_dict()["all_library_declared"] is False
        assert [s["declared_by"] for s in p.to_dict()["stages"]] == ["third_party", "library"]
        assert any("third-party" in n and "not examined" in n for n in p.notes)

    def test_a_function_registration_is_third_party_too(self, clean_traits_registry):
        register_traits(
            "test_fn_reducer",
            lambda stage: StageTraits(
                name="test_fn_reducer", kind="tokens", basis_invariance=O, declared_by="library"
            ),
            generic=StageTraits(name="test_fn_reducer", kind="tokens", basis_invariance=N),
        )

        class Fn(_Plain):
            name = "test_fn_reducer"

        assert traits_of(Fn()).declared_by == "third_party"
        assert traits_of("test_fn_reducer").declared_by == "third_party"

    def test_a_class_declaration_is_third_party_for_every_form(self):
        t = StageTraits(name="declared", kind="tokens", basis_invariance=O, declared_by="library")

        class ByAttribute(_Plain):
            traits = t

        class ByMethod(_Plain):
            def traits(self):
                return t

        for cls in (ByAttribute, ByMethod):
            assert traits_of(cls()).declared_by == "third_party"

    def test_a_subclass_of_a_library_stage_that_declares_itself_is_marked(self):
        """The audit's probe: a BinaryQuantizer subclass that claims ORTHOGONAL."""

        class Claims(BinaryQuantizer):
            traits = StageTraits(name="binary", kind="bits", basis_invariance=O)

        p = pipeline_traits(Pipeline([Claims()]))
        assert p.basis_sensitive is False and p.stages[0].declared_by == "third_party"
        assert not p.all_library_declared and p.unconditional  # nothing is claimed about it
        assert any("third-party" in n for n in p.notes)

    def test_an_undeclared_stage_is_not_library_declared(self):
        t = traits_of(_Plain())
        assert t.declared_by == "undeclared"
        assert not pipeline_traits(Pipeline([_Plain()])).all_library_declared

    def test_declared_by_and_conditions_are_validated(self):
        with pytest.raises(ValueError, match="declared_by"):
            StageTraits(name="x", kind="bits", basis_invariance=O, declared_by="me")
        with pytest.raises(TypeError, match="conditions"):
            StageTraits(name="x", kind="bits", basis_invariance=O, conditions="one sentence")
        t = StageTraits(name="x", kind="bits", basis_invariance=O, conditions=["a", "b"])
        assert t.conditions == ("a", "b") and t.to_dict()["conditions"] == ["a", "b"]
        assert hash(t) == hash(dataclasses.replace(t))
