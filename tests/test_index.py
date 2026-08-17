from __future__ import annotations

import numpy as np
import pytest

from optivision.compression import Compressor, maxsim_asymmetric
from optivision.config import CompressionConfig
from optivision.index.numpy_index import NumpyIndex
from optivision.types import PageRef, PatchGrid, PrunedPage


def _unit(rng, n, d=32):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _page(rng, doc_id, n=10, d=32, before=100):
    vectors = _unit(rng, n, d)
    pruned = PrunedPage(
        ref=PageRef(doc_id=doc_id, page_no=1),
        embeddings=vectors,
        kept_token_index=np.arange(n, dtype=np.int32),
        keep_mask=np.ones((1, n), dtype=bool),
        grid=PatchGrid.contiguous(1, n),
        n_tokens_before=before,
    )
    return Compressor(CompressionConfig(method="binary")).compress(pruned), vectors


class TestNumpyIndex:
    def test_scores_match_direct_maxsim(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        pages, _ = zip(*[_page(rng, f"d{i}") for i in range(6)])
        idx.add(list(pages))
        q = _unit(rng, 5, 32)
        scores = idx.score_all(q)
        expected = [maxsim_asymmetric(q, p.codes, 32) for p in pages]
        assert np.allclose(scores, expected, atol=1e-4)

    def test_variable_length_pages(self, rng, tmp_path):
        """reduceat segment boundaries are easy to get wrong with uneven pages."""
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        pages = [_page(rng, f"d{i}", n=n)[0] for i, n in enumerate([1, 17, 4, 33, 2])]
        idx.add(pages)
        q = _unit(rng, 3, 32)
        expected = [maxsim_asymmetric(q, p.codes, 32) for p in pages]
        assert np.allclose(idx.score_all(q), expected, atol=1e-4)

    def test_incremental_add_equals_single_add(self, rng, tmp_path):
        pages = [_page(rng, f"d{i}")[0] for i in range(5)]
        a = NumpyIndex(tmp_path / "a", dim=32)
        a.add(pages)
        b = NumpyIndex(tmp_path / "b", dim=32)
        for p in pages:
            b.add([p])
        q = _unit(rng, 4, 32)
        assert np.allclose(a.score_all(q), b.score_all(q))

    def test_search_is_ordered_and_ranked(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        idx.add([_page(rng, f"d{i}")[0] for i in range(10)])
        hits = idx.search(_unit(rng, 4, 32), top_k=4)
        assert [h.rank for h in hits] == [1, 2, 3, 4]
        assert all(hits[i].score >= hits[i + 1].score for i in range(3))

    def test_top_k_larger_than_corpus(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        idx.add([_page(rng, "only")[0]])
        assert len(idx.search(_unit(rng, 2, 32), top_k=50)) == 1

    def test_empty_index_returns_nothing(self, tmp_path, rng):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        assert idx.search(_unit(rng, 2, 32), top_k=5) == []
        assert idx.n_pages == 0

    def test_save_load_roundtrip(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        idx.add([_page(rng, f"d{i}")[0] for i in range(7)])
        q = _unit(rng, 4, 32)
        before = idx.score_all(q)
        idx.save()

        reloaded = NumpyIndex.load(tmp_path / "idx")
        assert reloaded.n_pages == 7
        assert np.allclose(reloaded.score_all(q), before)
        assert [r.page_id for r in reloaded.refs] == [r.page_id for r in idx.refs]

    def test_stats_accounting(self, rng, tmp_path):
        idx = NumpyIndex(tmp_path / "idx", dim=32)
        idx.add([_page(rng, f"d{i}", n=10, d=32, before=100)[0] for i in range(4)])
        s = idx.stats()
        assert s["n_pages"] == 4
        assert s["n_vectors"] == 40
        assert s["tokens_before"] == 400
        assert s["token_reduction"] == pytest.approx(10.0)
        # 100 tokens x 32 dims x 4 bytes raw, vs 10 vectors x 4 bytes packed
        assert s["compression_ratio"] == pytest.approx((100 * 32 * 4) / (10 * 4))
