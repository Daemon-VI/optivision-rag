from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from optivision.config import PruningConfig
from optivision.encoders.synthetic import SyntheticEncoder
from optivision.pruning import TokenPruner, build_keep_mask, patch_saliency, prune_redundant
from optivision.types import PageRef

from .conftest import make_page


class TestSaliency:
    def test_blank_page_has_near_zero_saliency(self, blank_image):
        sal = patch_saliency(blank_image, 16, 16)
        assert sal.shape == (16, 16)
        assert sal.max() < 0.05

    def test_ink_scores_above_margin(self, page_image):
        sal = patch_saliency(page_image, 32, 32)
        # Text sits in the top third; the bottom half is empty paper.
        assert sal[:10].max() > 0.3
        assert sal[20:].max() < sal[:10].max()

    def test_grey_scan_is_not_all_ink(self):
        """A yellowed photocopy must not be read as a fully inked page."""
        grey = Image.new("RGB", (256, 256), (170, 165, 150))
        sal = patch_saliency(grey, 8, 8)
        assert sal.max() < 0.05

    def test_saliency_is_scale_invariant(self, page_image):
        small = patch_saliency(page_image.resize((256, 330)), 16, 16)
        large = patch_saliency(page_image.resize((1024, 1320)), 16, 16)
        assert np.allclose(small, large, atol=0.12)

    def test_rejects_bad_grid(self, page_image):
        with pytest.raises(ValueError):
            patch_saliency(page_image, 0, 8)


class TestKeepMask:
    def test_threshold_mode_drops_blank(self, page_image):
        sal = patch_saliency(page_image, 32, 32)
        mask = build_keep_mask(sal, blank_threshold=0.02, dilate=0)
        assert 0 < mask.sum() < mask.size  # something kept, something dropped

    def test_keep_ratio_is_an_exact_budget(self, page_image):
        """Dilation must not be able to blow past an explicit token budget."""
        sal = patch_saliency(page_image, 32, 32)
        mask = build_keep_mask(sal, keep_ratio=0.25, dilate=2, min_keep=0)
        assert mask.sum() == round(0.25 * 32 * 32)

    def test_min_keep_floor_on_blank_page(self, blank_image):
        sal = patch_saliency(blank_image, 16, 16)
        mask = build_keep_mask(sal, blank_threshold=0.5, min_keep=12, dilate=0)
        assert mask.sum() >= 12

    def test_dilation_only_grows(self, page_image):
        sal = patch_saliency(page_image, 32, 32)
        tight = build_keep_mask(sal, blank_threshold=0.02, dilate=0)
        grown = build_keep_mask(sal, blank_threshold=0.02, dilate=1)
        assert grown.sum() >= tight.sum()
        assert bool((grown | tight == grown).all())


class TestRedundancy:
    def test_identical_vectors_collapse_to_one(self):
        v = np.tile(np.eye(1, 8, dtype=np.float32), (10, 1))
        out, clusters = prune_redundant(v, threshold=0.99)
        assert out.shape[0] == 1
        assert len(clusters[0]) == 10

    def test_orthogonal_vectors_all_survive(self):
        v = np.eye(8, dtype=np.float32)
        out, clusters = prune_redundant(v, threshold=0.9)
        assert out.shape[0] == 8
        assert all(len(c) == 1 for c in clusters)

    def test_output_stays_normalised(self, rng):
        v = rng.standard_normal((40, 16)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        out, _ = prune_redundant(v, threshold=0.5, merge=True)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_every_input_lands_in_exactly_one_cluster(self, rng):
        v = rng.standard_normal((50, 16)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        _, clusters = prune_redundant(v, threshold=0.3)
        members = [i for c in clusters for i in c]
        assert sorted(members) == list(range(50))

    def test_empty_input(self):
        out, clusters = prune_redundant(np.zeros((0, 8), dtype=np.float32))
        assert out.shape[0] == 0 and clusters == []

    def test_blocked_path_matches_full_matrix(self, rng):
        v = rng.standard_normal((64, 16)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        a, ca = prune_redundant(v, threshold=0.4, max_pairwise=10_000)
        b, cb = prune_redundant(v, threshold=0.4, max_pairwise=1)  # forces row blocks
        assert np.allclose(a, b)
        assert ca == cb


class TestTokenPruner:
    def _encode(self, image, grid=32):
        enc = SyntheticEncoder(dim=32, grid=grid)
        return enc.encode_pages([image], [PageRef(doc_id="d", page_no=1)])[0]

    def test_mostly_blank_page_is_heavily_pruned(self):
        image = make_page()
        enc = self._encode(image)
        pruned = TokenPruner(PruningConfig()).prune(enc, image)
        assert pruned.n_kept < enc.n_tokens * 0.5
        assert pruned.n_kept > 0

    def test_disabled_pruner_is_identity(self):
        image = make_page()
        enc = self._encode(image)
        pruned = TokenPruner(PruningConfig(enabled=False)).prune(enc, image)
        assert pruned.n_kept == enc.n_tokens
        assert np.allclose(pruned.embeddings, enc.embeddings)

    def test_text_tokens_are_never_dropped(self):
        image = make_page()
        enc = self._encode(image)
        pruned = TokenPruner(PruningConfig(keep_text_tokens=True)).prune(enc, image)
        assert set(enc.text_token_index.tolist()) <= set(pruned.kept_token_index.tolist())

    def test_kept_index_length_matches_vectors(self):
        image = make_page()
        enc = self._encode(image)
        pruned = TokenPruner(PruningConfig()).prune(enc, image)
        assert len(pruned.kept_token_index) == pruned.n_kept

    def test_stats_are_consistent(self):
        image = make_page()
        enc = self._encode(image)
        pruned = TokenPruner(PruningConfig()).prune(enc, image)
        s = pruned.stats
        assert s["n_after_spatial"] <= s["n_patches"]
        assert s["n_after"] == pruned.n_kept
        assert 0.0 <= s["blank_fraction"] <= 1.0

    def test_no_image_means_no_spatial_pruning(self):
        image = make_page()
        enc = self._encode(image)
        cfg = PruningConfig(redundancy=False)
        pruned = TokenPruner(cfg).prune(enc, image=None)
        assert pruned.n_kept == enc.n_tokens
