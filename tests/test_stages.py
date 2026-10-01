from __future__ import annotations

import numpy as np
import pytest

from optivision.compose import CompressedCorpus, Pipeline, load_compressed, save_compressed
from optivision.compression import Compressor, fit_lloyd2
from optivision.config import CompressionConfig, PruningConfig
from optivision.encoders import SyntheticEncoder
from optivision.index.numpy_index import NumpyIndex
from optivision.pruning import TokenPruner
from optivision.representation import MultiVectorCorpus, MultiVectorRepresentation
from optivision.scoring import maxsim_matrix
from optivision.stages import (
    BinaryQuantizer,
    Float16Quantizer,
    Float32Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    RedundancyPruner,
    SpatialPruner,
    stage_from_dict,
)
from optivision.types import PageRef

from .conftest import make_page


@pytest.fixture(scope="module")
def pages():
    enc = SyntheticEncoder(dim=32, grid=16)
    items = [(PageRef(f"d{i}", 1), make_page(ink_rows=2 + i % 5, margin=40 + 10 * i)) for i in range(6)]
    encodings = [enc.encode_pages([img], [ref])[0] for ref, img in items]
    images = [img for _, img in items]
    queries = enc.encode_queries(["tax invoice", "office memorandum", "record of rights", "lab"])
    return encodings, images, queries


def _legacy(encodings, images, method="binary", codec=None, **pruning):
    pruner = TokenPruner(PruningConfig(**pruning))
    compressor = Compressor(CompressionConfig(method=method), codec=codec)
    pruned = [pruner.prune(e, im) for e, im in zip(encodings, images, strict=True)]
    return pruned, [compressor.compress(p) for p in pruned]


class TestLegacyEquivalence:
    """The refactored stages must reproduce the original pipeline exactly."""

    def test_spatial_then_redundancy_matches_token_pruner(self, pages):
        encodings, images, _ = pages
        pruned, _ = _legacy(encodings, images)
        corpus = MultiVectorCorpus.from_page_encodings(encodings, images=images)
        out = Pipeline([SpatialPruner(), RedundancyPruner()]).transform(corpus)
        for i, p in enumerate(pruned):
            np.testing.assert_array_equal(out[i].vectors, p.embeddings)

    @pytest.mark.parametrize("keep_ratio", [0.3, None])
    def test_budget_mode_matches(self, pages, keep_ratio):
        encodings, images, _ = pages
        pruned, _ = _legacy(encodings, images, keep_ratio=keep_ratio, redundancy_threshold=0.85)
        corpus = MultiVectorCorpus.from_page_encodings(encodings, images=images)
        out = Pipeline([SpatialPruner(keep_ratio=keep_ratio), RedundancyPruner(0.85)]).transform(corpus)
        for i, p in enumerate(pruned):
            np.testing.assert_array_equal(out[i].vectors, p.embeddings)

    @pytest.mark.parametrize("method,quantizer", [("binary", BinaryQuantizer), ("int8", Int8Quantizer),
                                                   ("none", Float32Quantizer)])
    def test_codes_and_scores_match(self, pages, tmp_path, method, quantizer):
        encodings, images, queries = pages
        _, compressed_old = _legacy(encodings, images, method=method)
        corpus = MultiVectorCorpus.from_page_encodings(encodings, images=images)
        new = Pipeline([SpatialPruner(), RedundancyPruner(), quantizer()]).compress(corpus)
        np.testing.assert_array_equal(new.codes, np.concatenate([c.codes for c in compressed_old]))

        idx = NumpyIndex(tmp_path, dim=corpus.dimension, method=method)
        idx.add(compressed_old)
        got = maxsim_matrix(MultiVectorCorpus.from_arrays(queries), new)
        for qi, q in enumerate(queries):
            np.testing.assert_allclose(got[qi], idx.score_all(q), rtol=1e-5, atol=1e-5)

    def test_lloyd2_matches_with_the_same_fit(self, pages):
        encodings, images, _ = pages
        corpus = MultiVectorCorpus.from_page_encodings(encodings, images=images)
        pipe = Pipeline([Lloyd2Quantizer(seed=7)])
        new = pipe.compress(corpus)
        codec = fit_lloyd2(np.asarray(corpus.vectors, np.float32), seed=7)
        _, old = _legacy(encodings, images, method="lloyd2", codec=codec, enabled=False)
        np.testing.assert_array_equal(new.codes, np.concatenate([c.codes for c in old]))


class TestTokenReducerContract:
    def test_protected_rows_pass_through_and_weights_add_up(self, rng):
        v = rng.standard_normal((6, 8)).astype(np.float32)
        v[1] = v[0]  # an exact duplicate to merge
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        doc = MultiVectorRepresentation(v, protected=np.array([0, 0, 0, 0, 1, 1], bool))
        out = RedundancyPruner(threshold=0.999).transform(MultiVectorCorpus.from_documents([doc]))
        assert out.num_vectors == 5
        np.testing.assert_array_equal(out.vectors[-2:], v[4:])  # protected, verbatim, last
        assert out.protected.tolist() == [False, False, False, True, True]
        assert out.weights[:3].sum() == 4.0 and out.weights.tolist()[0] == 2.0

    def test_positions_of_a_merge_are_the_mean(self):
        v = np.array([[1, 0], [1, 0], [0, 1]], np.float32)
        doc = MultiVectorRepresentation(v, positions=np.array([[0.1, 0.1], [0.3, 0.5], [0.9, 0.9]], np.float32))
        out = RedundancyPruner(threshold=0.99).transform(MultiVectorCorpus.from_documents([doc]))
        np.testing.assert_allclose(out.positions[0], [0.2, 0.3])

    def test_float16_corpus_stays_float16(self, rng):
        c = MultiVectorCorpus.from_arrays([rng.standard_normal((5, 8)).astype(np.float16)])
        assert RedundancyPruner().transform(c).dtype == np.float16

    def test_empty_document_survives(self, rng):
        c = MultiVectorCorpus.from_arrays([np.zeros((0, 4), np.float32), rng.standard_normal((3, 4)).astype(np.float32)])
        out = RedundancyPruner().transform(c)
        assert out.counts[0] == 0 and len(out) == 2

    def test_spatial_pruner_explains_what_it_needs(self, rng):
        c = MultiVectorCorpus.from_arrays([rng.standard_normal((4, 8)).astype(np.float32)])
        with pytest.raises(ValueError, match="page images"):
            SpatialPruner().transform(c)


class TestQuantizers:
    @pytest.mark.parametrize("q,expected_bytes", [(Float32Quantizer(), 64), (Float16Quantizer(), 32),
                                                   (Int8Quantizer(), 16), (BinaryQuantizer(), 2)])
    def test_code_sizes(self, rng, q, expected_bytes):
        v = rng.standard_normal((3, 16)).astype(np.float32)
        codes = q.encode(v)
        assert codes.shape == (3, expected_bytes) == (3, q.code_bytes(16))
        assert q.decode(codes, 16).shape == (3, 16)

    def test_float_roundtrips(self, rng):
        v = rng.standard_normal((4, 8)).astype(np.float32)
        np.testing.assert_array_equal(Float32Quantizer().decode(Float32Quantizer().encode(v), 8), v)
        np.testing.assert_allclose(Float16Quantizer().decode(Float16Quantizer().encode(v), 8), v, rtol=1e-3)

    def test_lloyd2_requires_fit(self, rng):
        with pytest.raises(RuntimeError):
            Lloyd2Quantizer().encode(rng.standard_normal((2, 8)).astype(np.float32))


class TestPipeline:
    def test_quantizer_must_be_last_and_single(self):
        with pytest.raises(ValueError):
            Pipeline([BinaryQuantizer(), RedundancyPruner()])
        with pytest.raises(ValueError):
            Pipeline([Int8Quantizer(), BinaryQuantizer()])

    def test_empty_pipeline_is_the_float32_baseline(self, rng):
        docs = [rng.standard_normal((n, 8)).astype(np.float32) for n in (3, 5)]
        c = MultiVectorCorpus.from_arrays(docs)
        comp = Pipeline().compress(c)
        assert comp.report()["compression_vs_float32"] == pytest.approx(1.0)
        np.testing.assert_array_equal(comp.decode_rows(0, 8), c.vectors)

    def test_dict_roundtrip(self):
        p = Pipeline([RedundancyPruner(0.9, merge=False), Int8Quantizer()])
        q = Pipeline.from_dict(p.to_dict())
        assert q.to_dict() == p.to_dict()
        assert stage_from_dict({"stage": "binary"}).name == "binary"
        assert "redundancy(0.9)" in p.label()

    def test_accounting(self, rng):
        c = MultiVectorCorpus.from_arrays([rng.standard_normal((10, 16)).astype(np.float16)])
        comp = Pipeline([BinaryQuantizer()]).compress(c)
        r = comp.report()
        assert r["compression_vs_float32"] == pytest.approx(32.0)
        assert r["compression_vs_native"] == pytest.approx(16.0)
        assert r["bits_per_dim"] == 1.0

    def test_save_load_compressed(self, rng, tmp_path):
        c = MultiVectorCorpus.from_arrays([rng.standard_normal((n, 16)).astype(np.float32) for n in (2, 0, 4)])
        comp = Pipeline([BinaryQuantizer()]).compress(c)
        back = load_compressed(save_compressed(comp, tmp_path / "c.npz"))
        assert isinstance(back, CompressedCorpus) and back.ids == comp.ids
        np.testing.assert_array_equal(back.decode_rows(0, 6), comp.decode_rows(0, 6))
