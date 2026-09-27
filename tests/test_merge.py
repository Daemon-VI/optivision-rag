from __future__ import annotations

import math

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.representation import MultiVectorCorpus, MultiVectorRepresentation
from optivision.stages import AdaptiveMerge, HierarchicalMerge, RandomPruner, stage_from_dict
from optivision.stages.base import DocView


def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _clustered(rng, n_clusters=5, per=20, dim=32, noise=0.05):
    centres = _unit(rng.standard_normal((n_clusters, dim)))
    v = np.repeat(centres, per, axis=0) + noise * rng.standard_normal((n_clusters * per, dim))
    return _unit(v).astype(np.float32)


def _view(v, importance=None, weights=None):
    return DocView(index=0, vectors=v, positions=None, importance=importance,
                   weights=np.ones(v.shape[0], np.float32) if weights is None else weights, metadata={})


class TestAdaptiveMerge:
    def test_count_adapts_to_the_number_of_clusters(self, rng):
        for n_clusters in (2, 5, 9):
            v = _clustered(rng, n_clusters=n_clusters)
            red = AdaptiveMerge(radius=0.8).reduce(_view(v))
            assert red.vectors.shape[0] == n_clusters

    def test_every_member_is_within_radius_of_its_centre(self, rng):
        v = _unit(rng.standard_normal((200, 16))).astype(np.float32)
        red = AdaptiveMerge(radius=0.3, representative="center").reduce(_view(v))
        sims = np.sum(v * red.vectors[red.labels], axis=1)
        assert sims.min() >= 0.3 - 1e-6

    def test_maxsim_error_is_bounded_by_the_radius(self, rng):
        """Design rationale check: |MaxSim change| <= sqrt(2 (1 - radius)) per query token."""
        v = _unit(rng.standard_normal((300, 16))).astype(np.float32)
        radius = 0.5
        centres = AdaptiveMerge(radius=radius, representative="center").reduce(_view(v)).vectors
        q = _unit(rng.standard_normal((500, 16))).astype(np.float32)
        gap = (q @ v.T).max(axis=1) - (q @ centres.T).max(axis=1)
        assert gap.min() >= -1e-6  # centres are members, so they can only lose
        assert gap.max() <= math.sqrt(2 * (1 - radius)) + 1e-6

    def test_ratio_and_floor(self, rng):
        v = _unit(rng.standard_normal((100, 16))).astype(np.float32)
        assert AdaptiveMerge(radius=None, ratio=0.1).reduce(_view(v)).vectors.shape[0] == 10
        dup = np.repeat(v[:1], 50, axis=0)
        assert AdaptiveMerge(radius=0.9, min_vectors=1).reduce(_view(dup)).vectors.shape[0] == 1
        assert AdaptiveMerge(radius=0.99, max_vectors=7).reduce(_view(v)).vectors.shape[0] == 7

    def test_every_row_is_assigned(self, rng):
        v = _unit(rng.standard_normal((50, 8))).astype(np.float32)
        red = AdaptiveMerge(radius=0.5, refine=3).reduce(_view(v))
        assert red.labels.min() == 0 and red.labels.max() == red.vectors.shape[0] - 1
        assert np.unique(red.labels).size == red.vectors.shape[0]

    def test_merged_rows_keep_their_members_magnitude(self, rng):
        v = (_unit(rng.standard_normal((40, 8))) * 3.0).astype(np.float32)
        red = AdaptiveMerge(radius=None, ratio=0.25).reduce(_view(v))
        np.testing.assert_allclose(np.linalg.norm(red.vectors, axis=1), 3.0, rtol=1e-5)

    def test_importance_seeding_starts_from_the_most_important_rows(self, rng):
        v = _unit(rng.standard_normal((30, 8))).astype(np.float32)
        imp = np.zeros(30, np.float32)
        imp[[4, 17]] = [2.0, 1.0]
        red = AdaptiveMerge(radius=None, max_vectors=2, representative="center", seeding="importance",
                            n_importance_seeds=2).reduce(_view(v, importance=imp))
        np.testing.assert_array_equal(red.vectors, v[[4, 17]])

    def test_deterministic(self, rng):
        c = MultiVectorCorpus.from_arrays([_clustered(rng), _clustered(rng, 3)])
        a = AdaptiveMerge(radius=0.7).transform(c)
        b = AdaptiveMerge(radius=0.7).transform(c)
        np.testing.assert_array_equal(a.vectors, b.vectors)

    def test_protected_rows_survive_in_a_pipeline(self, rng):
        v = _clustered(rng, 3)
        prot = np.zeros(v.shape[0], bool)
        prot[:2] = True
        c = MultiVectorCorpus.from_documents([MultiVectorRepresentation(v, protected=prot)])
        out = Pipeline([AdaptiveMerge(radius=0.8)]).transform(c)
        assert out.protected.sum() == 2
        np.testing.assert_array_equal(out.vectors[out.protected], v[:2])

    def test_argument_validation(self):
        with pytest.raises(ValueError):
            AdaptiveMerge(radius=None, ratio=None, max_vectors=None)
        with pytest.raises(ValueError):
            AdaptiveMerge(radius=1.5)
        with pytest.raises(ValueError):
            AdaptiveMerge(representative="center", refine=2)

    def test_dict_roundtrip(self):
        s = AdaptiveMerge(radius=0.75, ratio=0.5, refine=1)
        assert stage_from_dict(s.to_dict()).params() == s.params()


class TestHierarchicalMerge:
    def test_budget_and_pooling(self, rng):
        pytest.importorskip("scipy")
        v = _unit(rng.standard_normal((90, 16))).astype(np.float32)
        red = HierarchicalMerge(ratio=1 / 3).reduce(_view(v))
        assert 1 <= red.vectors.shape[0] <= 30
        np.testing.assert_allclose(np.linalg.norm(red.vectors, axis=1), 1.0, rtol=1e-5)
        assert red.labels.min() == 0 and red.labels.max() == red.vectors.shape[0] - 1

    def test_recovers_obvious_clusters(self, rng):
        pytest.importorskip("scipy")
        v = _clustered(rng, n_clusters=4, per=10)
        red = HierarchicalMerge(ratio=0.1).reduce(_view(v))
        assert red.vectors.shape[0] == 4
        assert np.unique(red.labels.reshape(4, 10), axis=1).shape[1] == 1  # each cluster intact


class TestRandomPruner:
    def test_exact_count_and_reproducible(self, rng):
        c = MultiVectorCorpus.from_arrays([rng.standard_normal((40, 8)).astype(np.float32)] * 2)
        a = RandomPruner(ratio=0.25, seed=3).transform(c)
        b = RandomPruner(ratio=0.25, seed=3).transform(c)
        assert a.counts.tolist() == [10, 10]
        np.testing.assert_array_equal(a.vectors, b.vectors)
        assert not np.array_equal(a[0].vectors, a[1].vectors)  # per-document draws differ


class TestHierarchicalDistanceMode:
    def test_count_adapts_per_document(self, rng):
        pytest.importorskip("scipy")
        few = _clustered(rng, n_clusters=2, per=20)
        many = _clustered(rng, n_clusters=8, per=5)
        stage = HierarchicalMerge(ratio=None, max_distance=0.6)
        assert stage.reduce(_view(few)).vectors.shape[0] == 2
        assert stage.reduce(_view(many)).vectors.shape[0] == 8

    def test_ratio_caps_the_distance_cut(self, rng):
        pytest.importorskip("scipy")
        v = _unit(rng.standard_normal((50, 16))).astype(np.float32)
        red = HierarchicalMerge(ratio=0.1, max_distance=0.01).reduce(_view(v))
        assert red.vectors.shape[0] <= 5

    def test_validation_and_roundtrip(self):
        with pytest.raises(ValueError):
            HierarchicalMerge(ratio=None, max_distance=None)
        s = HierarchicalMerge(ratio=None, max_distance=0.5)
        assert stage_from_dict(s.to_dict()).params() == s.params()
