"""Contract checks for each model-agnostic stage and for a few pipelines.

Serialisation round trip, determinism, no mutation of the inputs, the quantizer shape and
dtype contract, protected rows, and the property the selection rule's assumptions rely on:
fitting a stage on documents gives the same state whether or not queries are passed along
(``PCAProjector(basis="joint")`` is the one declared exception).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import optivision.core.stages as core_stages
import optivision.stages as library_stages
from optivision.compose import Pipeline
from optivision.core.identity import array_digest
from optivision.core.stages import (
    STAGES,
    AdaptiveMerge,
    BinaryQuantizer,
    DimensionProjector,
    DimensionReducer,
    Float16Quantizer,
    Float32Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    NormPreservingInt4Quantizer,
    PCAProjector,
    Quantizer,
    RandomProjector,
    RandomPruner,
    RedundancyPruner,
    RotationProjector,
    TokenReducer,
    TruncateProjector,
    fitted_state,
    fitted_state_digest,
    pipeline_fitted_identity,
    pipeline_traits,
    stage_from_dict,
    traits_of,
)
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix

from .test_core_stage_traits import hierarchical, make_anisotropic

pytestmark = pytest.mark.filterwarnings("ignore:Int8Quantizer:UserWarning")

STAGE_CASES = {
    "adaptive_merge": lambda: AdaptiveMerge(radius=0.9),
    "adaptive_merge(center)": lambda: AdaptiveMerge(radius=0.9, center="mean"),
    "adaptive_merge(ratio,representative=center)": lambda: AdaptiveMerge(
        radius=None, ratio=0.5, representative="center"
    ),
    "adaptive_merge(refine)": lambda: AdaptiveMerge(radius=0.9, refine=2),
    "hierarchical_merge": lambda: hierarchical(ratio=0.5),
    "hierarchical_merge(distance)": lambda: hierarchical(ratio=None, max_distance=0.9),
    "random_prune": lambda: RandomPruner(0.5, seed=2),
    "redundancy": lambda: RedundancyPruner(0.9),
    "redundancy(merge=False)": lambda: RedundancyPruner(0.9, merge=False),
    "pca": lambda: PCAProjector(16),
    "pca(center)": lambda: PCAProjector(16, center=True),
    "project(pca)": lambda: DimensionProjector(16, method="pca"),
    "project(random)": lambda: DimensionProjector(16, method="random", seed=2),
    "project(truncate)": lambda: DimensionProjector(16, method="truncate"),
    "random_projection": lambda: RandomProjector(16, seed=2),
    "truncate": lambda: TruncateProjector(16),
    "rotation": lambda: RotationProjector(seed=2),
    "float32": lambda: Float32Quantizer(),
    "float16": lambda: Float16Quantizer(),
    "binary": lambda: BinaryQuantizer(),
    "binary(center)": lambda: BinaryQuantizer(center="mean"),
    "int8(fixed)": lambda: Int8Quantizer("fixed"),
    "int8(per_vector)": lambda: Int8Quantizer("per_vector"),
    "int8(per_dimension)": lambda: Int8Quantizer("per_dimension"),
    "int8(per_dimension,center)": lambda: Int8Quantizer("per_dimension", center="mean"),
    "int4": lambda: Int4Quantizer(),
    "int4(center)": lambda: Int4Quantizer(center="mean"),
    "int4norm": lambda: NormPreservingInt4Quantizer(),
    "int4norm(center)": lambda: NormPreservingInt4Quantizer(center="mean"),
    "lloyd2": lambda: Lloyd2Quantizer(seed=3),
}
PIPELINE_CASES = {
    "empty": list,
    "merge>binary": lambda: [AdaptiveMerge(radius=0.9), BinaryQuantizer()],
    "ward>pca>int8": lambda: [
        hierarchical(ratio=0.5),
        PCAProjector(16),
        Int8Quantizer("per_vector"),
    ],
    "redundancy>rotation>int4(center)": lambda: [
        RedundancyPruner(0.9),
        RotationProjector(1),
        Int4Quantizer(center="mean"),
    ],
    "merge(center)>pca(center)>int8(per_dimension,center)": lambda: [
        AdaptiveMerge(radius=0.9, center="mean"),
        PCAProjector(16, center=True),
        Int8Quantizer("per_dimension", center="mean"),
    ],
    "random_prune>random_projection>lloyd2": lambda: [
        RandomPruner(0.5, seed=1),
        RandomProjector(24, seed=1),
        Lloyd2Quantizer(seed=1),
    ],
}


def joint():
    return PCAProjector(16, basis="joint")


# the kind of each case, from the stage name in its id (no factory is called at import time)
_TOKEN_NAMES = {"adaptive_merge", "hierarchical_merge", "random_prune", "redundancy"}
_QUANTIZER_NAMES = {"float32", "float16", "binary", "int8", "int4", "int4norm", "lloyd2"}
QUANTIZERS = {k: v for k, v in STAGE_CASES.items() if k.split("(")[0] in _QUANTIZER_NAMES}
TOKEN = {k: v for k, v in STAGE_CASES.items() if k.split("(")[0] in _TOKEN_NAMES}


@pytest.fixture(scope="module")
def data():
    docs, queries = make_anisotropic()
    return docs, queries


def digests(corpus):
    return [array_digest(corpus.vectors), array_digest(corpus.offsets)]


class TestRegistry:
    def test_every_agnostic_stage_of_the_package_is_covered_by_the_cases(self):
        covered = {case.split("(")[0] for case in STAGE_CASES}
        mine = {
            n
            for n, c in STAGES.items()
            if c.__module__.startswith("optivision.") and n != "spatial"
        }
        assert mine <= covered, f"no contract case for {sorted(mine - covered)}"

    @pytest.mark.parametrize("make", STAGE_CASES.values(), ids=list(STAGE_CASES))
    def test_each_stage_is_registered_and_of_exactly_one_kind(self, make):
        stage = make()
        assert STAGES[stage.name] is type(stage)
        assert [isinstance(stage, k) for k in (TokenReducer, DimensionReducer, Quantizer)].count(
            True
        ) == 1

    def test_the_facade_exports_the_objects_of_the_library_not_copies(self):
        names = [
            *("STAGES", "Stage", "TokenReducer", "DimensionReducer", "Quantizer", "register"),
            *("stage_from_dict", "DocView", "Reduction", "AdaptiveMerge", "HierarchicalMerge"),
            *("RandomPruner", "RedundancyPruner", "PCAProjector", "RandomProjector"),
            *("TruncateProjector", "DimensionProjector", "BinaryQuantizer", "Float16Quantizer"),
            *("Float32Quantizer", "Int4Quantizer", "Int8Quantizer", "Lloyd2Quantizer"),
            "NormPreservingInt4Quantizer",
        ]
        for name in names:
            assert getattr(core_stages, name) is getattr(library_stages, name), name
        from optivision.core.stages import codecs, projections, reducers

        for module in (codecs, projections, reducers):
            for name in module.__all__:
                assert getattr(module, name) is getattr(core_stages, name), name
        assert projections.RotationProjector is core_stages.RotationProjector is STAGES["rotation"]
        assert set(reducers.__all__) == {
            "AdaptiveMerge",
            "HierarchicalMerge",
            "RandomPruner",
            "RedundancyPruner",
        }
        assert len(codecs.__all__) == 7

    def test_the_page_specific_pruner_stays_out_of_core(self):
        assert not hasattr(core_stages, "SpatialPruner")
        assert "SpatialPruner" not in core_stages.__all__

    def test_every_exported_name_resolves_and_is_listed_once(self):
        assert len(core_stages.__all__) == len(set(core_stages.__all__))
        for name in core_stages.__all__:
            assert hasattr(core_stages, name), name


class TestSerialisation:
    @pytest.mark.parametrize("make", STAGE_CASES.values(), ids=list(STAGE_CASES))
    def test_to_dict_round_trip_gives_the_same_stage_description(self, make):
        stage = make()
        spec = stage.to_dict()
        assert json.loads(json.dumps(spec)) == spec  # JSON-native
        rebuilt = stage_from_dict(spec)
        assert type(rebuilt) is type(stage) and rebuilt.to_dict() == spec
        assert (
            type(stage)(**stage.params()).to_dict() == spec
        )  # params() are the constructor arguments
        assert spec["stage"] == stage.name and "stage" not in stage.params()

    @pytest.mark.parametrize("make", PIPELINE_CASES.values(), ids=list(PIPELINE_CASES))
    def test_pipelines_round_trip(self, make):
        pipe = Pipeline(make())
        again = Pipeline.from_dict(json.loads(pipe.to_json()))
        assert again.to_dict() == pipe.to_dict()
        assert [type(s) for s in again.stages] == [type(s) for s in pipe.stages]
        # a rebuilt pipeline is unfitted exactly when some stage holds fitted state
        assert again.fitted == (not pipeline_traits(again).has_fitted_state)

    def test_an_unknown_stage_name_is_refused(self):
        with pytest.raises(ValueError, match="unknown stage"):
            stage_from_dict({"stage": "no_such_stage"})


def build(make):
    stage = make()
    return Pipeline([stage])


class TestDeterminism:
    @pytest.mark.parametrize("make", STAGE_CASES.values(), ids=list(STAGE_CASES))
    def test_same_input_and_parameters_give_bitwise_identical_output(self, data, make):
        docs, queries = data
        results = []
        for _ in range(2):
            pipe = build(make)
            comp = pipe.compress(docs)
            results.append(
                (
                    pipe.transform(docs).vectors,
                    comp.codes,
                    pipe.transform_queries(queries).vectors,
                    pipeline_fitted_identity(pipe),
                )
            )
        for a, b in zip(*results, strict=True):
            assert (
                np.array_equal(a, b) and getattr(a, "dtype", None) == getattr(b, "dtype", None)
                if isinstance(a, np.ndarray)
                else a == b
            )

    @pytest.mark.parametrize("make", PIPELINE_CASES.values(), ids=list(PIPELINE_CASES))
    def test_pipelines_are_deterministic(self, data, make):
        docs, queries = data
        a, b = Pipeline(make()), Pipeline(make())
        ca, cb = a.compress(docs), b.compress(docs)
        assert np.array_equal(ca.codes, cb.codes) and np.array_equal(ca.offsets, cb.offsets)
        assert np.array_equal(
            a.transform_queries(queries).vectors, b.transform_queries(queries).vectors
        )
        assert np.array_equal(
            maxsim_matrix(a.transform_queries(queries), ca),
            maxsim_matrix(b.transform_queries(queries), cb),
        )
        assert pipeline_fitted_identity(a) == pipeline_fitted_identity(b)

    @pytest.mark.parametrize("make", STAGE_CASES.values(), ids=list(STAGE_CASES))
    def test_fitting_again_gives_the_same_state(self, data, make):
        docs, _ = data
        stage = make()
        assert stage.fit(docs) is stage and stage.fitted
        before = fitted_state_digest(stage)
        stage.fit(docs)
        assert fitted_state_digest(stage) == before


class TestInputsAreNotModified:
    @pytest.mark.parametrize(
        "make",
        list(PIPELINE_CASES.values()) + [lambda m=m: [m()] for m in STAGE_CASES.values()],
        ids=list(PIPELINE_CASES) + list(STAGE_CASES),
    )
    def test_documents_and_queries_are_unchanged_by_every_step(self, data, make):
        docs, queries = data
        before = digests(docs), digests(queries)
        pipe = Pipeline(make())
        pipe.fit(docs, queries)
        assert (digests(docs), digests(queries)) == before
        pipe.transform(docs)
        pipe.transform_queries(queries)
        comp = pipe.compress(docs, fit=False)
        comp.decode_rows(0, comp.num_vectors)
        assert (digests(docs), digests(queries)) == before

    def test_the_aligned_arrays_of_a_corpus_are_unchanged_too(self, data):
        docs, _ = data
        n = docs.num_vectors
        rng = np.random.default_rng(0)
        rich = MultiVectorCorpus(
            docs.vectors,
            docs.offsets,
            protected=np.arange(n) % 7 == 0,
            positions=rng.random((n, 2)).astype(np.float32),
            weights=np.ones(n, np.float32),
            importance=rng.random(n).astype(np.float32),
        )
        before = [
            array_digest(a) for a in (rich.protected, rich.positions, rich.weights, rich.importance)
        ]
        for make in (
            STAGE_CASES["adaptive_merge"],
            STAGE_CASES["redundancy"],
            STAGE_CASES["hierarchical_merge"],
        ):
            Pipeline([make()]).fit(rich).transform(rich)
        assert [
            array_digest(a) for a in (rich.protected, rich.positions, rich.weights, rich.importance)
        ] == before


class TestQuantizerContract:
    @pytest.mark.parametrize("dim", [48, 33])
    @pytest.mark.parametrize("make", QUANTIZERS.values(), ids=list(QUANTIZERS))
    def test_codes_have_the_declared_width_and_decoding_has_the_right_shape_and_dtype(
        self, make, dim
    ):
        docs, _ = make_anisotropic(d=dim, n_docs=8, n_queries=2)
        q = make()
        q.fit(docs)
        x = docs.vectors[:50]
        codes = q.encode(x)
        assert codes.dtype == np.uint8 and codes.shape == (x.shape[0], q.code_bytes(dim))
        out = q.decode(codes, dim)
        assert out.dtype == np.float32 and out.shape == x.shape and np.isfinite(out).all()
        assert q.bits_per_dim(dim) == 8.0 * q.code_bytes(dim) / dim
        assert isinstance(q.overhead_bytes(), int) and q.overhead_bytes() >= 0
        assert type(q.decode(codes[:0], dim)) is np.ndarray and q.decode(codes[:0], dim).shape == (
            0,
            dim,
        )

    def test_the_stored_width_formulas(self):
        d = 33
        widths = {
            "float32": 4 * d,
            "float16": 2 * d,
            "binary": (d + 7) // 8,
            "int4": (d + 1) // 2 + 2,
            "int4norm": (d + 1) // 2 + 2,
            "lloyd2": (d + 3) // 4,
        }
        for name, expected in widths.items():
            assert STAGE_CASES[name]().code_bytes(d) == expected
        assert (
            Int8Quantizer("fixed").code_bytes(d)
            == d
            == Int8Quantizer("per_dimension").code_bytes(d)
        )
        assert Int8Quantizer("per_vector").code_bytes(d) == d + 2

    def test_float_codecs_round_trip_exactly_up_to_their_precision(self, data):
        docs, _ = data
        x = docs.vectors[:50]
        assert np.array_equal(
            Float32Quantizer().decode(Float32Quantizer().encode(x), x.shape[1]), x
        )
        half = Float16Quantizer().decode(Float16Quantizer().encode(x), x.shape[1])
        assert np.array_equal(half, x.astype(np.float16).astype(np.float32))

    @pytest.mark.parametrize(
        "name",
        ["float32", "float16", "int8(fixed)", "int8(per_vector)", "int8(per_dimension)", "int4",
         "int4norm"],
    )
    def test_zero_vectors_decode_to_exact_zeros(self, data, name):
        docs, _ = data
        q = STAGE_CASES[name]()
        q.fit(docs)
        zeros = np.zeros((3, docs.dimension), np.float32)
        assert not q.decode(q.encode(zeros), docs.dimension).any()

    @pytest.mark.parametrize("make", QUANTIZERS.values(), ids=list(QUANTIZERS))
    def test_a_quantizer_does_not_modify_the_array_it_encodes(self, data, make):
        docs, _ = data
        q = make()
        q.fit(docs)
        x = docs.vectors[:40].copy()
        before = array_digest(x)
        q.encode(x)
        assert array_digest(x) == before


class TestProtectedRows:
    @pytest.fixture(scope="class")
    def protected_corpus(self):
        docs, _ = make_anisotropic()
        protected = np.zeros(docs.num_vectors, dtype=bool)
        for lo, hi in zip(docs.offsets[:-1], docs.offsets[1:], strict=True):
            protected[hi - 2 : hi] = True  # the last two rows of every document
        return MultiVectorCorpus(docs.vectors, docs.offsets, protected=protected)

    @pytest.mark.parametrize("make", TOKEN.values(), ids=list(TOKEN))
    def test_protected_rows_pass_through_verbatim(self, protected_corpus, make):
        corpus = protected_corpus
        pipe = Pipeline([make()]).fit(corpus)
        out = pipe.transform(corpus)
        assert out.num_vectors < corpus.num_vectors and out.protected is not None
        for i in range(len(corpus)):
            lo, hi = corpus.span(i)
            olo, ohi = out.span(i)
            kept_in = corpus.vectors[lo:hi][corpus.protected[lo:hi]]
            kept_out = out.vectors[olo:ohi][out.protected[olo:ohi]]
            assert np.array_equal(kept_in, kept_out), i  # same rows, same order, same bits
            free_in, free_out = (hi - lo) - len(kept_in), (ohi - olo) - len(kept_out)
            assert 1 <= free_out <= free_in
        assert out.protected.sum() == corpus.protected.sum()

    @pytest.mark.parametrize("make", TOKEN.values(), ids=list(TOKEN))
    def test_a_document_of_only_protected_rows_is_left_alone(self, make):
        docs, _ = make_anisotropic(n_docs=5)
        corpus = MultiVectorCorpus(
            docs.vectors, docs.offsets, protected=np.ones(docs.num_vectors, bool)
        )
        out = Pipeline([make()]).fit(corpus).transform(corpus)
        assert np.array_equal(out.vectors, corpus.vectors)
        assert np.array_equal(out.offsets, corpus.offsets)


class TestFitSeesDocumentsOnly:
    """The selection rule assumes stages fitted on documents: queries must change nothing."""

    @pytest.mark.parametrize("make", STAGE_CASES.values(), ids=list(STAGE_CASES))
    def test_fit_with_and_without_queries_gives_the_same_state_and_output(self, data, make):
        docs, queries = data
        a, b = make(), make()
        a.fit(docs)
        b.fit(docs, queries)
        assert not traits_of(b).fit_uses_queries
        assert fitted_state_digest(a) == fitted_state_digest(b)
        sa, sb = fitted_state(a), fitted_state(b)
        assert set(sa) == set(sb) and all(
            np.array_equal(sa[k], sb[k]) if isinstance(sa[k], np.ndarray) else sa[k] == sb[k]
            for k in sa
        )
        if isinstance(a, (TokenReducer, DimensionReducer)):
            assert np.array_equal(a.transform(docs).vectors, b.transform(docs).vectors)

    @pytest.mark.parametrize("make", PIPELINE_CASES.values(), ids=list(PIPELINE_CASES))
    def test_a_pipeline_fitted_with_queries_equals_one_fitted_without(self, data, make):
        docs, queries = data
        a, b = Pipeline(make()), Pipeline(make())
        a.fit(docs)
        b.fit(docs, queries)  # queries reach later stages through the earlier projections
        assert pipeline_fitted_identity(a) == pipeline_fitted_identity(b)
        assert np.array_equal(a.compress(docs, fit=False).codes, b.compress(docs, fit=False).codes)

    def test_joint_pca_is_the_declared_exception(self, data):
        docs, queries = data
        _, other_queries = make_anisotropic(seed=5)
        assert traits_of(joint()).fit_uses_queries
        with pytest.raises(ValueError, match="sample queries"):
            joint().fit(docs)
        a, b = joint().fit(docs, queries), joint().fit(docs, other_queries)
        assert fitted_state_digest(a) != fitted_state_digest(b)  # the queries do enter the state
        assert fitted_state_digest(a) != fitted_state_digest(PCAProjector(16).fit(docs))
        again = joint().fit(docs, queries)
        assert fitted_state_digest(a) == fitted_state_digest(
            again
        )  # still deterministic given the same queries


class TestStageLifecycle:
    @pytest.mark.parametrize(
        "name",
        [
            "pca",
            "project(pca)",
            "random_projection",
            "rotation",
            "binary(center)",
            "int8(per_dimension)",
            "int4(center)",
            "int4norm(center)",
            "lloyd2",
            "adaptive_merge(center)",
        ],
    )
    def test_a_stage_with_fitted_state_refuses_to_run_unfitted(self, data, name):
        docs, _ = data
        stage = STAGE_CASES[name]()
        assert not stage.fitted
        if isinstance(stage, Quantizer):
            with pytest.raises(RuntimeError, match="fit"):
                stage.encode(docs.vectors[:5])
        else:
            with pytest.raises(RuntimeError, match="fit"):
                stage.transform(docs)
        assert stage.fit(docs).fitted

    @pytest.mark.parametrize(
        "name",
        [
            "float32",
            "float16",
            "binary",
            "int8(fixed)",
            "int8(per_vector)",
            "int4",
            "int4norm",
            "truncate",
            "hierarchical_merge",
            "random_prune",
            "redundancy",
        ],
    )
    def test_stateless_stages_are_always_fitted(self, name):
        assert STAGE_CASES[name]().fitted


class TestDtypes:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    @pytest.mark.parametrize("make", PIPELINE_CASES.values(), ids=list(PIPELINE_CASES))
    def test_pipelines_accept_every_float_dtype(self, data, make, dtype):
        docs, queries = data
        corpus = docs.astype(dtype)
        pipe = Pipeline(make())
        comp = pipe.compress(corpus)
        scores = maxsim_matrix(pipe.transform_queries(queries.astype(dtype)), comp)
        assert scores.shape == (len(queries), len(docs)) and np.isfinite(scores).all()
        assert comp.stats["native_dtype_before"] == str(np.dtype(dtype))

    @pytest.mark.parametrize("make", STAGE_CASES.values(), ids=list(STAGE_CASES))
    def test_output_dtypes_follow_the_stage_kind(self, data, make):
        docs, _ = data
        stage = make()
        if isinstance(stage, Quantizer):
            return
        for dtype in (np.float16, np.float32):
            corpus = docs.astype(dtype)
            out = Pipeline([make()]).fit(corpus).transform(corpus)
            expected = corpus.vectors.dtype if isinstance(stage, TokenReducer) else np.float32
            assert out.vectors.dtype == expected
            assert np.isfinite(out.vectors).all()


@pytest.mark.parametrize(
    "make",
    [v for k, v in STAGE_CASES.items() if k != "int8(fixed)"],  # that one warns when it clips
    ids=[k for k in STAGE_CASES if k != "int8(fixed)"],
)
def test_no_stage_writes_warnings_on_ordinary_input(data, make):
    docs, queries = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipe = Pipeline([make()])
        comp = pipe.compress(docs)
        maxsim_matrix(pipe.transform_queries(queries), comp)
