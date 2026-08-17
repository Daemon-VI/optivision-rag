from __future__ import annotations

import numpy as np
import pytest

from optivision.compression import Compressor
from optivision.config import CompressionConfig
from optivision.types import PageRef, PatchGrid, PrunedPage

qdrant_client = pytest.importorskip("qdrant_client")

from optivision.index.qdrant_index import QdrantIndex


def _page(rng, doc_id, n=12, d=32):
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pruned = PrunedPage(
        ref=PageRef(doc_id=doc_id, page_no=1),
        embeddings=v,
        kept_token_index=np.arange(n, dtype=np.int32),
        keep_mask=np.ones((1, n), dtype=bool),
        grid=PatchGrid.contiguous(1, n),
        n_tokens_before=100,
    )
    return Compressor(CompressionConfig(method="binary")).compress(pruned), v


@pytest.fixture
def index(tmp_path):
    idx = QdrantIndex(path=tmp_path / "q", dim=32, method="binary", recreate=True)
    yield idx
    idx.close()


class TestQdrantIndex:
    def test_multivector_maxsim_ranks_the_right_page(self, index, rng):
        """Querying with a page's own vectors must return that page first."""
        pages, vecs = zip(*[_page(rng, f"doc{i}") for i in range(8)])
        index.add(list(pages))
        assert index.n_pages == 8

        hits = index.search(vecs[3][:4], top_k=3)
        assert hits[0].ref.page_id == "doc3::p1"
        assert hits[0].score > hits[1].score

    def test_payload_survives_the_roundtrip(self, index, rng):
        page, _ = _page(rng, "invoice_001")
        index.add([page])
        hit = index.search(np.eye(1, 32, dtype=np.float32), top_k=1)[0]
        assert hit.ref.doc_id == "invoice_001"
        assert hit.ref.page_no == 1

    def test_reindexing_the_same_page_does_not_duplicate(self, index, rng):
        """Point ids are derived from page_id, so a rebuild must overwrite."""
        page, _ = _page(rng, "doc0")
        index.add([page])
        index.add([page])
        assert index.n_pages == 1

    def test_stats_report_pipeline_byte_accounting(self, index, rng):
        pages = [_page(rng, f"doc{i}", n=10, d=32)[0] for i in range(4)]
        index.add(pages)
        s = index.stats()
        assert s["n_vectors"] == 40
        assert s["tokens_before"] == 400
        # 100 tokens x 32 dims x 4 B raw vs 10 vectors x 4 B of packed bits
        assert s["compression_ratio"] == pytest.approx((100 * 32 * 4) / (10 * 4))

    def test_empty_search(self, index):
        assert index.search(np.eye(1, 32, dtype=np.float32), top_k=5) == []
