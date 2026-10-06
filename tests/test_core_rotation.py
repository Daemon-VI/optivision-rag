"""RotationProjector: a seeded orthogonal map, registered as the stage ``"rotation"``."""

from __future__ import annotations

import numpy as np
import pytest

from optivision.calibration import recommended_search_space
from optivision.compose import Pipeline
from optivision.core.identity import array_digest, digest
from optivision.core.stages import (
    STAGES,
    AdaptiveMerge,
    BinaryQuantizer,
    Float32Quantizer,
    Int8Quantizer,
    PCAProjector,
    RotationProjector,
    fitted_state,
    fitted_state_digest,
    restore_fitted_state,
    seeded_rotation,
    stage_from_dict,
)
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix

from .test_core_stage_traits import EQUAL, make_anisotropic, pipeline_scores, relative_change

#: sha256 of the 40 default candidate specifications, as registered by the research harness
#: (scripts/research/x_common.py, S0_SPEC_SHA256)
REGISTERED_DEFAULT_SPACE_SHA256 = "af29b01146a90a048ffea9cfbd66f0fedf988134829cd4dc56861f1207893520"


@pytest.fixture(scope="module")
def data():
    docs, queries = make_anisotropic()
    return docs, queries, maxsim_matrix(queries, docs)


def gram_schmidt(g):
    """Modified Gram-Schmidt: the QR whose R has a positive diagonal by construction."""
    q = np.zeros_like(g)
    for j in range(g.shape[1]):
        v = g[:, j].copy()
        for i in range(j):
            v -= (q[:, i] @ v) * q[:, i]
        q[:, j] = v / np.linalg.norm(v)
    return q


class TestRegistration:
    def test_the_stage_is_registered_under_its_name(self):
        assert STAGES["rotation"] is RotationProjector
        stage = stage_from_dict({"stage": "rotation", "seed": 3})
        assert isinstance(stage, RotationProjector) and stage.seed == 3
        assert stage_from_dict({"stage": "rotation"}).seed == 0

    def test_to_dict_and_pipeline_round_trip(self):
        stage = RotationProjector(seed=4)
        assert stage.to_dict() == {"stage": "rotation", "seed": 4} and stage.params() == {"seed": 4}
        pipe = Pipeline([AdaptiveMerge(radius=0.9), RotationProjector(4), BinaryQuantizer()])
        again = Pipeline.from_dict(pipe.to_dict())
        assert again.to_dict() == pipe.to_dict() and isinstance(again.stages[1], RotationProjector)
        assert (
            Pipeline.from_dict(pipe.to_dict()).label()
            == pipe.label()
            == "adaptive_merge(0.9) > rotation(4) > binary"
        )

    def test_registering_it_changes_no_existing_pipeline_description(self):
        pytest.importorskip("scipy")  # without SciPy the default space has 22 candidates, not 40
        for dim in (128, 96):
            spec = []
            for family, steps in recommended_search_space(dim).items():
                for pipe in steps:
                    spec.append(
                        {"position": len(spec), "family": family, "pipeline": pipe.to_dict()}
                    )
            assert len(spec) == 40
            assert digest(spec) == REGISTERED_DEFAULT_SPACE_SHA256
        assert Pipeline([Int8Quantizer("per_vector")]).to_dict() == {
            "stages": [{"scale": "per_vector", "stage": "int8"}]
        }

    def test_seed_is_validated(self):
        for bad in (-1, 1.5, True, "0", None):
            with pytest.raises(ValueError, match="seed"):
                RotationProjector(bad)
        assert (
            RotationProjector(np.int64(3)).seed == 3
            and type(RotationProjector(np.int64(3)).seed) is int
        )

    def test_the_documentation_states_what_the_stage_is_and_is_not(self):
        doc = RotationProjector.__doc__
        assert "not evaluated by the registered experiments" in doc
        assert "reports/research/X_RESULTS.md" in doc and "reduce" in doc
        assert "float32" in doc and "d x d" in doc  # the memory note
        assert "improvement" in doc


class TestMatrix:
    @pytest.mark.parametrize("dim", [1, 2, 8, 33, 64, 128, 512])
    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_it_is_orthogonal(self, dim, seed):
        r = seeded_rotation(dim, seed)
        assert r.dtype == np.float32 and r.shape == (dim, dim) and r.flags["C_CONTIGUOUS"]
        r64 = r.astype(np.float64)
        assert np.abs(r64.T @ r64 - np.eye(dim)).max() < 1e-6
        assert np.abs(r64 @ r64.T - np.eye(dim)).max() < 1e-6

    def test_it_is_deterministic_and_the_seed_matters(self):
        a, b = seeded_rotation(32, 3), seeded_rotation(32, 3)
        assert np.array_equal(a, b) and array_digest(a) == array_digest(b)
        assert not np.allclose(a, seeded_rotation(32, 4), atol=1e-3)
        assert seeded_rotation(32, 3).shape != seeded_rotation(33, 3).shape

    @pytest.mark.parametrize("dim", [8, 33])
    def test_the_sign_convention_is_the_positive_diagonal_one(self, dim):
        g = np.random.default_rng(2).standard_normal((dim, dim))
        assert np.abs(seeded_rotation(dim, 2).astype(np.float64) - gram_schmidt(g)).max() < 1e-6
        r = seeded_rotation(dim, 2).astype(np.float64)
        assert (np.diag(r.T @ g) > 0).all()  # the triangular factor has a positive diagonal
        q, rr = np.linalg.qr(g)  # the convention of the registered symmetry experiment, spelled out
        assert np.array_equal(
            seeded_rotation(dim, 2), (q * np.sign(np.diag(rr))).astype(np.float32)
        )

    def test_the_matrix_does_not_depend_on_which_signs_lapack_returns(self, monkeypatch):
        real_qr = np.linalg.qr
        flips = np.where(np.random.default_rng(9).integers(0, 2, 40) == 0, -1.0, 1.0)

        def other_convention(a, *args, **kwargs):
            q, r = real_qr(a, *args, **kwargs)
            return q * flips[: q.shape[1]], flips[: r.shape[0], None] * r  # still a valid QR

        a = np.random.default_rng(0).standard_normal((40, 40))
        assert not np.array_equal(
            other_convention(a)[0], real_qr(a)[0]
        )  # the stand-in really differs
        expected = {dim: seeded_rotation(dim, 5) for dim in (8, 40)}
        monkeypatch.setattr(np.linalg, "qr", other_convention)
        for dim, matrix in expected.items():
            assert np.array_equal(seeded_rotation(dim, 5), matrix)

    def test_dim_is_validated(self):
        for bad in (0, -3, 2.5, True):
            with pytest.raises(ValueError, match="dim"):
                seeded_rotation(bad, 0)


class TestStage:
    def test_output_dimension_equals_input_dimension(self):
        for d in (1, 2, 33, 128, 4096):
            assert RotationProjector().output_dim(d) == d

    def test_fit_builds_the_matrix_from_the_dimension_and_the_seed_only(self, data):
        docs, _, _ = data
        other, _ = make_anisotropic(seed=3)
        stage = RotationProjector(seed=2)
        assert not stage.fitted
        assert stage.fit(docs) is stage and stage.fitted
        assert (
            stage.matrix.shape == (docs.dimension, docs.dimension)
            and stage.matrix.dtype == np.float32
        )
        assert np.array_equal(
            stage.matrix, RotationProjector(seed=2).fit(other).matrix
        )  # no data dependence
        assert np.array_equal(stage.matrix, seeded_rotation(docs.dimension, 2))
        narrow = MultiVectorCorpus(np.ones((4, 5), np.float32), [0, 4])
        assert RotationProjector(seed=2).fit(narrow).matrix.shape == (5, 5)
        assert stage.fit(narrow).matrix.shape == (5, 5)  # a refit follows the new dimension

    def test_documents_and_queries_go_through_the_same_matrix(self, data):
        docs, queries, _ = data
        pipe = Pipeline([RotationProjector(seed=6)]).fit(docs)
        m = pipe.stages[0].matrix
        assert np.array_equal(pipe.transform(docs).vectors, docs.vectors @ m)
        assert np.array_equal(pipe.transform_queries(queries).vectors, queries.vectors @ m)
        assert pipe.transform(docs).vectors.dtype == np.float32
        assert np.array_equal(pipe.transform(docs).offsets, docs.offsets)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_any_float_input_comes_out_as_float32(self, data, dtype):
        docs, _, _ = data
        pipe = Pipeline([RotationProjector(seed=1)]).fit(docs)
        out = pipe.transform(docs.astype(dtype))
        assert out.vectors.dtype == np.float32 and out.vectors.shape == docs.vectors.shape

    def test_norms_and_float_scores_are_preserved_up_to_rounding(self, data):
        docs, queries, reference = data
        for seed in (0, 1, 2):
            moved = Pipeline([RotationProjector(seed=seed)]).fit(docs).transform(docs).vectors
            assert (
                np.abs(np.linalg.norm(moved, axis=1) - np.linalg.norm(docs.vectors, axis=1)).max()
                < 1e-6
            )
            rotated = pipeline_scores(lambda seed=seed: [RotationProjector(seed)], docs, queries)
            assert relative_change(reference, rotated, reference) <= EQUAL
            with_codec = pipeline_scores(
                lambda seed=seed: [RotationProjector(seed), Float32Quantizer()], docs, queries
            )
            assert relative_change(reference, with_codec, reference) <= EQUAL
            merged = pipeline_scores(lambda: [AdaptiveMerge(radius=0.9)], docs, queries)
            merged_rotated = pipeline_scores(
                lambda seed=seed: [AdaptiveMerge(radius=0.9), RotationProjector(seed)],
                docs,
                queries,
            )
            assert relative_change(merged, merged_rotated, reference) <= EQUAL

    def test_a_rotation_in_front_of_a_codec_is_a_different_candidate(self, data):
        docs, queries, reference = data
        plain = pipeline_scores(lambda: [BinaryQuantizer()], docs, queries)
        rotated = pipeline_scores(lambda: [RotationProjector(0), BinaryQuantizer()], docs, queries)
        assert (
            relative_change(plain, rotated, reference) > 1e-3
        )  # a diagnostic, not a free improvement

    def test_it_composes_with_pca_without_error(self, data):
        docs, queries, _ = data
        pipe = Pipeline([PCAProjector(16), RotationProjector(1), Int8Quantizer("per_vector")])
        comp = pipe.compress(docs)
        assert comp.dim == 16 and maxsim_matrix(pipe.transform_queries(queries), comp).shape == (
            len(queries),
            len(docs),
        )

    def test_unfitted_and_mismatched_use_is_refused(self, data):
        docs, _, _ = data
        stage = RotationProjector()
        with pytest.raises(RuntimeError, match="fit"):
            stage.project(np.zeros((2, docs.dimension), np.float32))
        with pytest.raises(RuntimeError, match="fit"):
            Pipeline([RotationProjector()]).transform(docs)
        stage.fit(docs)
        with pytest.raises(ValueError, match=f"{docs.dimension}-d"):
            stage.project(np.zeros((2, 7), np.float32))
        with pytest.raises(ValueError):
            stage.project(np.zeros(docs.dimension, np.float32))

    def test_fit_ignores_queries(self, data):
        docs, queries, _ = data
        a = RotationProjector(seed=2).fit(docs)
        b = RotationProjector(seed=2).fit(docs, queries)
        assert np.array_equal(a.matrix, b.matrix)


class TestFittedState:
    def test_the_matrix_is_the_fitted_state_and_restores_without_a_qr(self, data, monkeypatch):
        docs, queries, _ = data
        stage = RotationProjector(seed=2).fit(docs)
        state = fitted_state(stage)
        assert set(state) == {"matrix"} and state["matrix"] is stage.matrix
        assert fitted_state_digest(stage) == digest({"matrix": array_digest(stage.matrix)})

        def forbidden(*args, **kwargs):
            raise AssertionError("restore recomputed the matrix")

        monkeypatch.setattr(np.linalg, "qr", forbidden)
        twin = restore_fitted_state(
            stage_from_dict(stage.to_dict()),
            {"matrix": state["matrix"].copy()},
            expected_digest=fitted_state_digest(stage),
        )
        assert twin.fitted and np.array_equal(twin.matrix, stage.matrix)
        assert np.array_equal(
            Pipeline([twin]).transform_queries(queries).vectors, queries.vectors @ stage.matrix
        )

    def test_a_restored_matrix_must_be_orthogonal(self, data):
        docs, _, _ = data
        stage = RotationProjector(seed=2).fit(docs)
        with pytest.raises(ValueError, match="orthogonal"):
            restore_fitted_state(RotationProjector(seed=2), {"matrix": stage.matrix * 2})
