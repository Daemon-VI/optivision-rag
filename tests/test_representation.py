from __future__ import annotations

import numpy as np
import pytest

from optivision.bench import EncodedCorpus
from optivision.encoders import SyntheticEncoder
from optivision.representation import (
    MultiVectorCorpus,
    MultiVectorRepresentation,
    as_corpus,
)
from optivision.types import PageRef

from .conftest import make_page


def _docs(rng, counts=(5, 0, 3, 7), dim=16, dtype=np.float32):
    return [rng.standard_normal((n, dim)).astype(dtype) for n in counts]


class TestRepresentation:
    def test_shape_properties(self, rng):
        v = rng.standard_normal((7, 12)).astype(np.float32)
        r = MultiVectorRepresentation(v, doc_id="a")
        assert (r.num_vectors, r.dimension, r.dtype) == (7, 12, np.float32)
        assert r.nbytes == v.nbytes

    def test_keeps_float16(self, rng):
        r = MultiVectorRepresentation(rng.standard_normal((3, 4)).astype(np.float16))
        assert r.dtype == np.float16

    def test_rejects_integer_vectors(self):
        with pytest.raises(TypeError):
            MultiVectorRepresentation(np.zeros((3, 4), dtype=np.int8))

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            MultiVectorRepresentation(np.zeros(4, dtype=np.float32))

    def test_aligned_arrays_must_match(self, rng):
        v = rng.standard_normal((3, 4)).astype(np.float32)
        with pytest.raises(ValueError):
            MultiVectorRepresentation(v, protected=np.zeros(2, dtype=bool))
        with pytest.raises(ValueError):
            MultiVectorRepresentation(v, positions=np.zeros((3, 3)))

    def test_torch_bfloat16_is_widened(self):
        torch = pytest.importorskip("torch")
        t = torch.randn(5, 8, dtype=torch.bfloat16)
        r = MultiVectorRepresentation(t)
        assert r.dtype == np.float32 and r.vectors.shape == (5, 8)


class TestCorpus:
    def test_from_arrays_offsets_and_counts(self, rng):
        docs = _docs(rng)
        c = MultiVectorCorpus.from_arrays(docs)
        assert len(c) == 4
        assert c.counts.tolist() == [5, 0, 3, 7]
        assert c.offsets.tolist() == [0, 5, 5, 8, 15]
        for i, d in enumerate(docs):
            np.testing.assert_array_equal(c[i].vectors, d)

    def test_indexing_returns_views_not_copies(self, rng):
        c = MultiVectorCorpus.from_arrays(_docs(rng))
        assert np.shares_memory(c[2].vectors, c.vectors)

    def test_negative_index_and_bounds(self, rng):
        c = MultiVectorCorpus.from_arrays(_docs(rng))
        np.testing.assert_array_equal(c[-1].vectors, c[3].vectors)
        with pytest.raises(IndexError):
            c[4]

    def test_offsets_validation(self, rng):
        v = rng.standard_normal((5, 4)).astype(np.float32)
        with pytest.raises(ValueError):
            MultiVectorCorpus(v, [0, 3])  # does not end at 5
        with pytest.raises(ValueError):
            MultiVectorCorpus(v, [1, 5])  # does not start at 0
        with pytest.raises(ValueError):
            MultiVectorCorpus(v, [0, 4, 3, 5])  # decreasing

    def test_mixed_dimensions_rejected(self, rng):
        with pytest.raises(ValueError):
            MultiVectorCorpus.from_arrays([np.zeros((2, 4), np.float32), np.zeros((2, 5), np.float32)])

    def test_dtype_promotes_to_common_type(self, rng):
        c = MultiVectorCorpus.from_arrays([np.zeros((2, 4), np.float16), np.zeros((1, 4), np.float32)])
        assert c.dtype == np.float32

    def test_select_reorders_and_carries_aligned_arrays(self, rng):
        docs = [
            MultiVectorRepresentation(rng.standard_normal((n, 4)).astype(np.float32), doc_id=f"d{n}",
                                      protected=np.arange(n) == 0)
            for n in (2, 3, 4)
        ]
        c = MultiVectorCorpus.from_documents(docs)
        s = c.select([2, 0])
        assert s.ids == ["d4", "d2"]
        np.testing.assert_array_equal(s[0].vectors, docs[2].vectors)
        assert s.protected.tolist() == [True, False, False, False, True, False]

    def test_from_documents_fills_missing_aligned_arrays(self, rng):
        a = MultiVectorRepresentation(np.ones((2, 3), np.float32), weights=np.array([2.0, 3.0]))
        b = MultiVectorRepresentation(np.ones((1, 3), np.float32))
        c = MultiVectorCorpus.from_documents([a, b])
        assert c.weights.tolist() == [2.0, 3.0, 1.0]
        assert c.protected is None

    def test_with_vectors_keeps_structure(self, rng):
        c = MultiVectorCorpus.from_arrays(_docs(rng, dim=8))
        p = c.with_vectors(c.vectors[:, :4].copy())
        assert p.dimension == 4 and p.offsets.tolist() == c.offsets.tolist() and p.ids == c.ids
        with pytest.raises(ValueError):
            c.with_vectors(c.vectors[:3])

    def test_as_corpus_accepts_lists(self, rng):
        assert len(as_corpus(_docs(rng))) == 4
        with pytest.raises(TypeError):
            as_corpus("nope")

    def test_summary(self, rng):
        c = MultiVectorCorpus.from_arrays(_docs(rng))
        s = c.summary()
        assert s["n_docs"] == 4 and s["num_vectors"] == 15 and s["empty_docs"] == 1
        assert s["dimension"] == 16 and s["vectors_per_doc_max"] == 7


class TestSerialisation:
    def test_roundtrip_preserves_everything(self, rng, tmp_path):
        docs = [
            MultiVectorRepresentation(
                rng.standard_normal((n, 6)).astype(np.float16),
                doc_id=f"doc-{n}",
                metadata={"n": n, "tag": "x"},
                protected=np.arange(n) % 2 == 0,
                positions=rng.random((n, 2)).astype(np.float32),
                weights=np.full(n, 2.0, np.float32),
            )
            for n in (3, 0, 5)
        ]
        c = MultiVectorCorpus.from_documents(docs, attrs={"model": "m"})
        path = c.save(tmp_path / "c.npz")
        back = MultiVectorCorpus.load(path)
        assert back.dtype == np.float16
        np.testing.assert_array_equal(back.vectors, c.vectors)
        np.testing.assert_array_equal(back.offsets, c.offsets)
        np.testing.assert_array_equal(back.protected, c.protected)
        np.testing.assert_array_equal(back.positions, c.positions)
        np.testing.assert_array_equal(back.weights, c.weights)
        assert back.importance is None
        assert back.ids == c.ids and back.metadata == c.metadata and back.attrs == {"model": "m"}

    def test_load_rejects_foreign_npz(self, tmp_path):
        np.savez(tmp_path / "x.npz", vectors=np.zeros((1, 2), np.float32))
        with pytest.raises(ValueError):
            MultiVectorCorpus.load(tmp_path / "x.npz")

    def test_mmap_load(self, rng, tmp_path):
        c = MultiVectorCorpus.from_arrays(_docs(rng))
        back = MultiVectorCorpus.load(c.save(tmp_path / "m.npz"), mmap=True)
        np.testing.assert_array_equal(back.vectors, c.vectors)


class TestPageEncodingAdapter:
    @pytest.fixture
    def encodings(self):
        enc = SyntheticEncoder(dim=32, grid=8)
        pages = [(PageRef("a", 1), make_page()), (PageRef("b", 1), make_page(ink_rows=2))]
        out = []
        for ref, img in pages:
            out.extend(enc.encode_pages([img], [ref]))
        return out, [img for _, img in pages]

    def test_rows_are_identical_and_in_order(self, encodings):
        encs, images = encodings
        c = MultiVectorCorpus.from_page_encodings(encs, images=images)
        assert c.ids == [e.ref.page_id for e in encs]
        for i, e in enumerate(encs):
            np.testing.assert_array_equal(c[i].vectors, e.embeddings)

    def test_protected_and_positions(self, encodings):
        encs, _ = encodings
        c = MultiVectorCorpus.from_page_encodings(encs)
        e = encs[0]
        doc = c[0]
        assert doc.protected[e.text_token_index].all()
        assert doc.protected.sum() == e.text_token_index.size
        grid_rows = e.grid.token_index.reshape(-1)
        assert np.isfinite(doc.positions[grid_rows]).all()
        assert np.isnan(doc.positions[e.text_token_index]).all()
        # the top-left cell sits at the centre of the first grid square
        first = e.grid.token_index[0, 0]
        np.testing.assert_allclose(doc.positions[first], [0.5 / e.grid.rows, 0.5 / e.grid.cols])
        assert doc.metadata["grid_rows"] == e.grid.rows

    def test_legacy_cache_roundtrip(self, encodings, tmp_path):
        encs, images = encodings
        gray = [im.convert("L").resize((64, 64)) for im in images]
        EncodedCorpus(encs, gray).save(tmp_path / "cache.npz")
        c = MultiVectorCorpus.from_legacy_cache(tmp_path / "cache.npz")
        assert len(c) == 2 and c.images is not None
        np.testing.assert_array_equal(c[1].vectors, encs[1].embeddings)
        assert c.attrs["format"] == "legacy-encode-cache"
