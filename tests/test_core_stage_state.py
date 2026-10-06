"""Fitted state of stages: capture, digest, checked restore.

Every stage that holds fitted state, in every parameter variant, is fitted on a corpus; its
state is captured, a pipeline is rebuilt from ``to_dict()``, the state is restored into it,
and the restored pipeline must give the same codes and the same projected queries bit for
bit. Digests must be stable across repeated fits and must change with the data exactly where
the stage's traits say it reads the vectors. A state that is tampered with, mis-typed or
mis-shaped must raise.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import optivision.compression.lloyd2 as lloyd2_module
from optivision.compose import Pipeline
from optivision.core.identity import array_digest, digest
from optivision.core.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    DimensionProjector,
    DimensionReducer,
    Float16Quantizer,
    Float32Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RandomProjector,
    RandomPruner,
    RedundancyPruner,
    RotationProjector,
    TruncateProjector,
    fitted_state,
    fitted_state_digest,
    fitted_state_nbytes,
    pipeline_fitted_identity,
    pipeline_fitted_state,
    restore_fitted_state,
    restore_pipeline_state,
    stage_from_dict,
    traits_of,
)
from optivision.scoring import maxsim_matrix

from .test_core_stage_traits import hierarchical, make_anisotropic

# Int8Quantizer('fixed') warns when it clips; these tests use anisotropic, non-unit-scale data
pytestmark = pytest.mark.filterwarnings("ignore:Int8Quantizer:UserWarning")

#: one entry per stage with fitted state (every parameter variant), and some chains
SINGLE = {
    "adaptive_merge(center)": lambda: AdaptiveMerge(radius=0.9, center="mean"),
    "pca": lambda: PCAProjector(16),
    "pca(center)": lambda: PCAProjector(16, center=True),
    "project(pca)": lambda: DimensionProjector(16, method="pca"),
    "project(random)": lambda: DimensionProjector(16, method="random", seed=4),
    "random_projection": lambda: RandomProjector(16, seed=3),
    "rotation": lambda: RotationProjector(seed=2),
    "binary(center)": lambda: BinaryQuantizer(center="mean"),
    "int8(per_dimension)": lambda: Int8Quantizer("per_dimension"),
    "int8(per_dimension,center)": lambda: Int8Quantizer("per_dimension", center="mean"),
    "int8(fixed,center)": lambda: Int8Quantizer("fixed", center="mean"),
    "int8(per_vector,center)": lambda: Int8Quantizer("per_vector", center="mean"),
    "int4(center)": lambda: Int4Quantizer(center="mean"),
    "lloyd2": lambda: Lloyd2Quantizer(seed=5),
}
CHAINS = {
    "merge>pca>int8": lambda: [
        AdaptiveMerge(radius=0.9, center="mean"),
        PCAProjector(16, center=True),
        Int8Quantizer("per_dimension", center="mean"),
    ],
    "rotation>lloyd2": lambda: [RotationProjector(1), Lloyd2Quantizer(3)],
    "random_projection>binary(center)": lambda: [
        RandomProjector(24, seed=1),
        BinaryQuantizer(center="mean"),
    ],
    "ward>pca>int4(center)": lambda: [
        hierarchical(ratio=0.5),
        PCAProjector(16),
        Int4Quantizer(center="mean"),
    ],
}
STATELESS = {
    "float32": lambda: Float32Quantizer(),
    "float16": lambda: Float16Quantizer(),
    "binary": lambda: BinaryQuantizer(),
    "int8(fixed)": lambda: Int8Quantizer("fixed"),
    "int8(per_vector)": lambda: Int8Quantizer("per_vector"),
    "int4": lambda: Int4Quantizer(),
    "truncate": lambda: TruncateProjector(8),
    "project(truncate)": lambda: DimensionProjector(8, method="truncate"),
    "hierarchical_merge": lambda: hierarchical(ratio=0.5),
    "random_prune": lambda: RandomPruner(0.5, seed=1),
    "redundancy": lambda: RedundancyPruner(0.9),
    "adaptive_merge": lambda: AdaptiveMerge(radius=0.9),
}


@pytest.fixture(scope="module")
def data():
    docs, queries = make_anisotropic()
    other_docs, _ = make_anisotropic(seed=1)
    return docs, queries, other_docs


def copy_state(state):
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}


def fitted(make, docs):
    stage = make()
    stage.fit(docs)
    return stage


def chain_cases():
    return [pytest.param(make, id=name) for name, make in CHAINS.items()] + [
        pytest.param(lambda make=make: [make()], id=name) for name, make in SINGLE.items()
    ]


class TestCaptureAndRestore:
    @pytest.mark.parametrize("factory", chain_cases())
    def test_a_restored_pipeline_reproduces_codes_and_queries_bit_for_bit(self, data, factory):
        docs, queries, _ = data
        pipe = Pipeline(factory()).fit(docs)
        states = pipeline_fitted_state(pipe)
        identity = pipeline_fitted_identity(pipe)
        assert len(states) == len(identity) == len(pipe.stages)
        assert any(states)  # there is something to restore

        restored = Pipeline.from_dict(pipe.to_dict())
        assert not restored.fitted
        restore_pipeline_state(restored, [copy_state(s) for s in states], expected_digests=identity)
        assert restored.fitted
        assert pipeline_fitted_identity(restored) == identity

        a, b = pipe.compress(docs, fit=False), restored.compress(docs, fit=False)
        assert np.array_equal(a.codes, b.codes) and np.array_equal(a.offsets, b.offsets)
        assert a.nbytes == b.nbytes
        qa, qb = pipe.transform_queries(queries), restored.transform_queries(queries)
        assert qa.vectors.dtype == qb.vectors.dtype and np.array_equal(qa.vectors, qb.vectors)
        assert np.array_equal(pipe.transform(docs).vectors, restored.transform(docs).vectors)
        assert np.array_equal(maxsim_matrix(qa, a), maxsim_matrix(qb, b))

    def test_restored_attributes_are_the_fitted_ones(self, data):
        docs, _, _ = data
        for make in (SINGLE["pca"], SINGLE["pca(center)"]):
            stage = fitted(make, docs)
            twin = restore_fitted_state(
                stage_from_dict(stage.to_dict()), copy_state(fitted_state(stage))
            )
            for attr in (
                "components",
                "mean",
            ):  # an uncentred PCA keeps a zero mean that is never read
                a, b = getattr(stage, attr), getattr(twin, attr)
                assert a.dtype == b.dtype and np.array_equal(a, b), attr
            assert stage.explained == twin.explained

    def test_restore_does_not_alias_the_callers_arrays(self, data):
        docs, _, _ = data
        stage = fitted(SINGLE["int8(per_dimension,center)"], docs)
        state = copy_state(fitted_state(stage))
        twin = restore_fitted_state(stage_from_dict(stage.to_dict()), state)
        before = fitted_state_digest(twin)
        for v in state.values():
            v += 1.0
        assert fitted_state_digest(twin) == before

    def test_state_values_are_plain_arrays_and_json_native_scalars(self, data):
        docs, _, _ = data
        for name, make in SINGLE.items():
            state = fitted_state(fitted(make, docs))
            assert state, name
            for key, value in state.items():
                assert type(value) in (np.ndarray, float, int, str), (name, key, type(value))
            json.dumps({k: v for k, v in state.items() if not isinstance(v, np.ndarray)})

    def test_state_survives_an_npz_round_trip(self, data, tmp_path):
        docs, queries, _ = data
        pipe = Pipeline(CHAINS["merge>pca>int8"]()).fit(docs)
        states = pipeline_fitted_state(pipe)
        arrays = {
            f"{i}.{k}": v
            for i, s in enumerate(states)
            for k, v in s.items()
            if isinstance(v, np.ndarray)
        }
        scalars = [{k: v for k, v in s.items() if not isinstance(v, np.ndarray)} for s in states]
        np.savez(tmp_path / "state.npz", **arrays)
        with np.load(tmp_path / "state.npz", allow_pickle=False) as f:
            loaded = [
                {
                    **scalars[i],
                    **{k.split(".", 1)[1]: f[k] for k in f.files if k.startswith(f"{i}.")},
                }
                for i in range(len(states))
            ]
        restored = restore_pipeline_state(
            Pipeline.from_dict(json.loads(pipe.to_json())),
            loaded,
            expected_digests=pipeline_fitted_identity(pipe),
        )
        assert np.array_equal(
            restored.compress(docs, fit=False).codes, pipe.compress(docs, fit=False).codes
        )
        assert np.array_equal(
            restored.transform_queries(queries).vectors, pipe.transform_queries(queries).vectors
        )


class TestDigest:
    @pytest.mark.parametrize("factory", chain_cases())
    def test_digest_is_stable_across_fits_and_follows_the_data_where_declared(self, data, factory):
        docs, _, other = data
        a, b, c = (Pipeline(factory()).fit(corpus) for corpus in (docs, docs, other))
        ia, ib, ic = (pipeline_fitted_identity(p) for p in (a, b, c))
        assert ia == ib  # determinism
        for stage, x, y in zip(a.stages, ia, ic, strict=True):
            traits = traits_of(stage)
            if not traits.has_fitted_state:
                assert x is None and y is None, stage
            else:
                assert (x != y) == traits.reads_vector_values, (
                    stage
                )  # data-independent state stays put
                assert len(x) == 64 and len(y) == 64

    def test_digest_depends_on_the_seed_and_the_dimension_for_data_free_state(self, data):
        docs, _, _ = data
        digests = {
            fitted_state_digest(fitted(lambda s=s: RotationProjector(s), docs)) for s in (0, 1, 2)
        }
        assert len(digests) == 3
        narrow = docs.with_vectors(np.ascontiguousarray(docs.vectors[:, :32]))
        assert fitted_state_digest(fitted(SINGLE["rotation"], narrow)) != fitted_state_digest(
            fitted(SINGLE["rotation"], docs)
        )

    def test_digest_is_the_identity_digest_of_the_entries(self, data):
        docs, _, _ = data
        for name in ("lloyd2", "pca(center)"):
            stage = fitted(SINGLE[name], docs)
            state = fitted_state(stage)
            entries = {
                k: array_digest(v) if isinstance(v, np.ndarray) else v for k, v in state.items()
            }
            assert fitted_state_digest(stage) == digest(entries)

    def test_nbytes_counts_arrays_only(self, data):
        docs, _, _ = data
        d = docs.dimension
        assert fitted_state_nbytes(fitted(SINGLE["pca"], docs)) == 4 * d * 16
        assert fitted_state_nbytes(fitted(SINGLE["pca(center)"], docs)) == 4 * d * 16 + 8 * d
        assert fitted_state_nbytes(fitted(SINGLE["rotation"], docs)) == 4 * d * d
        assert fitted_state_nbytes(fitted(SINGLE["random_projection"], docs)) == 4 * d * 16
        assert fitted_state_nbytes(fitted(SINGLE["int8(per_dimension,center)"], docs)) == 8 * d
        assert fitted_state_nbytes(fitted(SINGLE["binary(center)"], docs)) == 4 * d
        assert (
            fitted_state_nbytes(fitted(SINGLE["lloyd2"], docs)) == 4 * d
        )  # the rotation is not stored


class TestStateless:
    @pytest.mark.parametrize("make", STATELESS.values(), ids=list(STATELESS))
    def test_no_state_means_empty_dict_none_and_zero_bytes(self, data, make):
        docs, _, _ = data
        stage = make()
        assert fitted_state(stage) == {} and fitted_state_digest(stage) is None  # even before fit
        stage.fit(docs)
        assert fitted_state(stage) == {} and fitted_state_digest(stage) is None
        assert fitted_state_nbytes(stage) == 0
        assert not traits_of(stage).has_fitted_state
        fresh = make()
        assert restore_fitted_state(fresh, {}) is fresh
        with pytest.raises(ValueError, match="unexpected"):
            restore_fitted_state(make(), {"mean": np.zeros(3, np.float32)})
        with pytest.raises(ValueError, match="digest"):
            restore_fitted_state(make(), {}, expected_digest="0" * 64)

    @pytest.mark.parametrize(
        "make",
        [*SINGLE.values(), *STATELESS.values()],
        ids=[*SINGLE, *(f"stateless:{k}" for k in STATELESS)],
    )
    def test_traits_and_state_agree_for_every_variant(self, data, make):
        docs, _, _ = data
        stage = fitted(make, docs)
        assert traits_of(stage).has_fitted_state == bool(fitted_state(stage)), stage

    def test_a_stage_with_state_that_is_not_fitted_is_not_stateless(self):
        for make in SINGLE.values():
            with pytest.raises(RuntimeError, match="not fitted"):
                fitted_state(make())
            with pytest.raises(RuntimeError):
                fitted_state_digest(make())
        with pytest.raises(RuntimeError):
            pipeline_fitted_identity(Pipeline(CHAINS["merge>pca>int8"]()))


def _bad_states(state):
    """Structural mutations of a valid state, and the message each must produce."""
    keys = sorted(state)
    arrays = [k for k in keys if isinstance(state[k], np.ndarray)]
    yield "missing key", {k: v for k, v in state.items() if k != keys[0]}, "missing"
    yield "extra key", {**state, "surprise": 1}, "unexpected"
    for k in arrays:
        other = np.float32 if state[k].dtype == np.float64 else np.float64
        yield f"{k}: {np.dtype(other)}", {**state, k: state[k].astype(other)}, "dtype"
        yield f"{k}: float16", {**state, k: state[k].astype(np.float16)}, "dtype"
        yield f"{k}: wrong rank", {**state, k: state[k][None]}, "-d array"
        yield f"{k}: empty", {**state, k: state[k].ravel()[:0]}, "-d array|empty"
        nan = state[k].copy()
        nan.flat[0] = np.nan
        yield f"{k}: nan", {**state, k: nan}, "non-finite"
        yield f"{k}: not an array", {**state, k: state[k].tolist()}, "numpy array"
    for k in keys:
        if isinstance(state[k], float):
            yield f"{k}: text", {**state, k: "0.5"}, "number"
            yield f"{k}: nan", {**state, k: float("nan")}, "finite"
        if isinstance(state[k], int) and not isinstance(state[k], bool):
            yield f"{k}: float", {**state, k: 1.5}, "integer"


class TestTamperedState:
    @pytest.mark.parametrize("name", list(SINGLE))
    def test_structurally_wrong_state_raises_and_leaves_the_stage_unfitted(self, data, name):
        docs, _, _ = data
        stage = fitted(SINGLE[name], docs)
        state = fitted_state(stage)
        for label, bad, message in _bad_states(state):
            fresh = stage_from_dict(stage.to_dict())
            with pytest.raises((ValueError, TypeError), match=message):
                restore_fitted_state(fresh, bad)
            assert not fresh.fitted, (name, label)
        # and the valid one still works afterwards
        assert restore_fitted_state(stage_from_dict(stage.to_dict()), copy_state(state)).fitted

    def test_mapping_is_required(self):
        with pytest.raises(TypeError, match="mapping"):
            restore_fitted_state(SINGLE["pca"](), [1, 2])

    @pytest.mark.parametrize(
        "name, mutate, message",
        [
            ("pca", lambda s: {**s, "components": s["components"] * 2}, "orthonormal"),
            ("pca", lambda s: {**s, "components": s["components"][:, :-1].copy()}, "keep"),
            ("pca", lambda s: {**s, "components": np.ascontiguousarray(s["components"].T)}, "keep"),
            ("pca", lambda s: {**s, "explained": 1.5}, "outside"),
            ("pca(center)", lambda s: {**s, "mean": s["mean"][:-1].copy()}, "mean has shape"),
            ("random_projection", lambda s: {**s, "matrix": s["matrix"] * 1.5}, "orthogonal"),
            ("random_projection", lambda s: {**s, "matrix": s["matrix"][:, :8].copy()}, "keeps"),
            ("project(random)", lambda s: {**s, "matrix": s["matrix"] * 0.5}, "orthogonal"),
            ("rotation", lambda s: {**s, "matrix": s["matrix"] * 1.01}, "not orthogonal"),
            ("rotation", lambda s: {**s, "matrix": s["matrix"][:, :-1].copy()}, "square"),
            (
                "int8(per_dimension)",
                lambda s: {**s, "ranges": np.zeros_like(s["ranges"])},
                "positive",
            ),
            (
                "int8(per_dimension,center)",
                lambda s: {**s, "mean": s["mean"][:-1].copy()},
                "disagree",
            ),
            ("lloyd2", lambda s: {**s, "seed": s["seed"] + 1}, "seed"),
            ("lloyd2", lambda s: {**s, "sigma": -1.0}, "negative"),
            ("lloyd2", lambda s: {**s, "rotation_digest": "0" * 64}, "rotation"),
        ],
    )
    def test_state_that_contradicts_the_stage_raises(self, data, name, mutate, message):
        docs, _, _ = data
        stage = fitted(SINGLE[name], docs)
        bad = mutate(copy_state(fitted_state(stage)))
        with pytest.raises(ValueError, match=message):
            restore_fitted_state(stage_from_dict(stage.to_dict()), bad)

    @pytest.mark.parametrize(
        "name",
        [
            "pca",
            "rotation",
            "adaptive_merge(center)",
            "binary(center)",
            "int8(per_dimension,center)",
            "int4(center)",
            "lloyd2",
            "random_projection",
        ],
    )
    def test_a_structurally_valid_change_is_caught_by_the_digest(self, data, name):
        docs, _, _ = data
        stage = fitted(SINGLE[name], docs)
        original = fitted_state_digest(stage)
        state = copy_state(fitted_state(stage))
        key = next(k for k in sorted(state) if isinstance(state[k], np.ndarray))
        if state[key].ndim == 2:  # keep it orthonormal: swap two columns
            state[key] = np.ascontiguousarray(state[key][:, [1, 0, *range(2, state[key].shape[1])]])
        else:
            state[key] = state[key].copy()
            state[key][0] = state[key][0] + 1e-3 * (1 + abs(state[key][0]))
        with pytest.raises(ValueError, match="digest"):
            restore_fitted_state(stage_from_dict(stage.to_dict()), state, expected_digest=original)
        twin = restore_fitted_state(
            stage_from_dict(stage.to_dict()), state
        )  # accepted without the digest
        assert fitted_state_digest(twin) != original

    def test_restoring_into_a_fitted_stage_is_refused(self, data):
        docs, _, _ = data
        stage = fitted(SINGLE["pca"], docs)
        with pytest.raises(ValueError, match="already fitted"):
            restore_fitted_state(stage, copy_state(fitted_state(stage)))

    def test_pipeline_errors_say_which_stage(self, data):
        docs, _, _ = data
        pipe = Pipeline(CHAINS["merge>pca>int8"]()).fit(docs)
        states = [copy_state(s) for s in pipeline_fitted_state(pipe)]
        states[1]["components"] = states[1]["components"] * 3
        with pytest.raises(ValueError, match=r"stage 1 \(pca\)"):
            restore_pipeline_state(Pipeline.from_dict(pipe.to_dict()), states)
        with pytest.raises(ValueError, match="2 states for a pipeline of 3 stages"):
            restore_pipeline_state(Pipeline.from_dict(pipe.to_dict()), states[:2])
        with pytest.raises(ValueError, match="1 digests for a pipeline of 3 stages"):
            restore_pipeline_state(
                Pipeline.from_dict(pipe.to_dict()),
                [copy_state(s) for s in pipeline_fitted_state(pipe)],
                expected_digests=["x"],
            )

    def test_a_subclass_of_a_stateful_stage_must_bring_its_own_state_access(self, data):
        docs, _, _ = data

        class MyPCA(PCAProjector):
            name = "test_my_pca"

        stage = MyPCA(16).fit(docs)
        with pytest.raises(TypeError, match="get_state"):
            fitted_state(stage)
        with pytest.raises(TypeError, match="get_state"):
            restore_fitted_state(MyPCA(16), {})


class TestLloyd2Rotation:
    def test_the_rotation_is_not_stored_but_its_digest_is(self, data):
        docs, _, _ = data
        stage = fitted(SINGLE["lloyd2"], docs)
        state = fitted_state(stage)
        assert set(state) == {"mu", "sigma", "seed", "rotation_digest"}
        assert not any(
            isinstance(v, np.ndarray) and v.size >= docs.dimension**2 for v in state.values()
        )
        assert state["rotation_digest"] == array_digest(stage.codec.rotation)

    def test_a_platform_whose_qr_differs_is_detected(self, data, monkeypatch):
        docs, _, _ = data
        stage = fitted(SINGLE["lloyd2"], docs)
        state = copy_state(fitted_state(stage))
        original = lloyd2_module.rotation_matrix(docs.dimension, stage.seed)
        flipped = original.copy()
        flipped[:, 0] *= -1  # the same decomposition with another sign convention: still orthogonal
        monkeypatch.setattr(lloyd2_module, "rotation_matrix", lambda dim, seed=7: flipped)
        with pytest.raises(ValueError, match="rotation") as info:
            restore_fitted_state(stage_from_dict(stage.to_dict()), state)
        assert "LAPACK" in str(info.value) and "platform" in str(info.value)
        monkeypatch.undo()
        assert restore_fitted_state(stage_from_dict(stage.to_dict()), state).fitted

    def test_restore_does_not_need_the_matrix_to_be_recomputed_for_projections(
        self, data, monkeypatch
    ):
        docs, queries, _ = data
        pipe = Pipeline([RotationProjector(3), Float32Quantizer()]).fit(docs)
        states = pipeline_fitted_state(pipe)

        def forbidden(*args, **kwargs):
            raise AssertionError("the QR routine was called while restoring")

        monkeypatch.setattr(np.linalg, "qr", forbidden)
        restored = restore_pipeline_state(
            Pipeline.from_dict(pipe.to_dict()), [copy_state(s) for s in states]
        )
        assert np.array_equal(
            restored.transform_queries(queries).vectors, pipe.transform_queries(queries).vectors
        )


class _Scaled(DimensionReducer):
    """A third-party projection with fitted state that is read through get_state/set_state."""

    name = "test_scaled"

    def __init__(self):
        self.weights = None

    def params(self):
        return {}

    @property
    def fitted(self):
        return self.weights is not None

    def output_dim(self, input_dim):
        return input_dim

    def fit(self, corpus, queries=None):
        self.weights = (1.0 / np.maximum(np.abs(corpus.vectors).max(axis=0), 1e-6)).astype(
            np.float32
        )
        return self

    def project(self, vectors):
        return np.asarray(vectors, dtype=np.float32) * self.weights

    def get_state(self):
        return {"weights": self.weights, "note": "per-axis scale", "version": np.int64(2)}

    def set_state(self, state):
        self.weights = np.asarray(state["weights"], dtype=np.float32)


class _Stateless(DimensionReducer):
    """A third-party projection without state and without accessors."""

    name = "test_stateless_third_party"

    def params(self):
        return {}

    def output_dim(self, input_dim):
        return input_dim

    def project(self, vectors):
        return np.asarray(vectors, dtype=np.float32)


class TestThirdPartyStages:
    def test_get_state_and_set_state_are_used(self, data):
        docs, queries, _ = data
        stage = _Scaled().fit(docs)
        state = fitted_state(stage)
        assert set(state) == {"weights", "note", "version"} and state["version"] == 2
        assert type(state["version"]) is int  # numpy scalars become plain ones
        digest = fitted_state_digest(stage)
        assert digest is not None and fitted_state_nbytes(stage) == state["weights"].nbytes
        twin = restore_fitted_state(_Scaled(), copy_state(state), expected_digest=digest)
        assert twin.fitted and np.array_equal(twin.weights, stage.weights)
        assert np.array_equal(
            Pipeline([twin]).transform_queries(queries).vectors,
            Pipeline([stage]).transform_queries(queries).vectors,
        )
        with pytest.raises(ValueError, match="digest"):
            restore_fitted_state(_Scaled(), copy_state(state), expected_digest="0" * 64)
        with pytest.raises(ValueError, match="already fitted"):
            restore_fitted_state(stage, copy_state(state))
        with pytest.raises(RuntimeError, match="not fitted"):
            fitted_state(_Scaled())

    def test_state_must_be_plain(self):
        class Bad(_Scaled):
            def get_state(self):
                return {"weights": object()}

        stage = Bad()
        stage.weights = np.ones(2, np.float32)
        with pytest.raises(TypeError, match="neither an array nor a plain scalar"):
            fitted_state(stage)

    def test_a_fitted_stage_without_accessors_counts_as_stateless(self):
        stage = _Stateless()
        assert stage.fitted and fitted_state(stage) == {} and fitted_state_digest(stage) is None
