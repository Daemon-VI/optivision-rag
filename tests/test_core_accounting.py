"""Byte accounting: the library's size key, and what it leaves out."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from optivision.calibration import score_candidate
from optivision.compose import CompressedCorpus, Pipeline, load_compressed, save_compressed
from optivision.core.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    ByteAccounting,
    DimensionProjector,
    DimensionReducer,
    Float16Quantizer,
    Float32Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RandomProjector,
    RotationProjector,
    TruncateProjector,
    byte_accounting,
    fitted_state_nbytes,
)

from .test_core_stage_traits import hierarchical, make_anisotropic

pytestmark = pytest.mark.filterwarnings("ignore:Int8Quantizer:UserWarning")

D = 48
K = 16

#: id -> (stages, serving state bytes, append-only state bytes), all at d = 48, k = 16
CASES = {
    "empty": (list, 0, 0),
    "float32": (lambda: [Float32Quantizer()], 0, 0),
    "float16": (lambda: [Float16Quantizer()], 0, 0),
    "binary": (lambda: [BinaryQuantizer()], 0, 0),
    "int8(per_vector)": (lambda: [Int8Quantizer("per_vector")], 0, 0),
    "int8(per_dimension,center)": (lambda: [Int8Quantizer("per_dimension", center="mean")], 0, 0),
    "int4(center)": (lambda: [Int4Quantizer(center="mean")], 0, 0),
    "lloyd2": (
        lambda: [Lloyd2Quantizer()],
        0,
        0,
    ),  # mu and sigma are codec overhead; the rotation is regenerated
    "ward>binary": (lambda: [hierarchical(ratio=0.5), BinaryQuantizer()], 0, 0),
    "truncate>int8": (lambda: [TruncateProjector(K), Int8Quantizer("per_vector")], 0, 0),
    "pca>binary": (lambda: [PCAProjector(K), BinaryQuantizer()], 4 * D * K, 0),
    "pca(center)>int8": (
        lambda: [PCAProjector(K, center=True), Int8Quantizer("per_dimension")],
        4 * D * K,
        8 * D,
    ),
    "project(pca)": (lambda: [DimensionProjector(K, method="pca")], 4 * D * K, 0),
    "project(random)>int4": (
        lambda: [DimensionProjector(K, method="random"), Int4Quantizer()],
        4 * D * K,
        0,
    ),
    "random_projection": (lambda: [RandomProjector(K, seed=1)], 4 * D * K, 0),
    "rotation>binary": (lambda: [RotationProjector(1), BinaryQuantizer()], 4 * D * D, 0),
    "merge(center)>binary": (
        lambda: [AdaptiveMerge(radius=0.9, center="mean"), BinaryQuantizer()],
        0,
        4 * D,
    ),
    "merge(center)>pca(center)>int8(per_dimension,center)": (
        lambda: [
            AdaptiveMerge(radius=0.9, center="mean"),
            PCAProjector(K, center=True),
            Int8Quantizer("per_dimension", center="mean"),
        ],
        4 * D * K,
        4 * D + 8 * D,
    ),
}


@pytest.fixture(scope="module")
def data():
    docs, queries = make_anisotropic(d=D)
    return docs, queries


def account(factory, docs):
    pipe = Pipeline(factory())
    comp = pipe.compress(docs)
    return pipe, comp, byte_accounting(comp, pipe)


@pytest.mark.parametrize("name", list(CASES))
class TestSizeKey:
    def test_selection_bytes_are_the_libraries_size_key(self, data, name):
        docs, queries = data
        factory = CASES[name][0]
        _, comp, acc = account(factory, docs)
        assert acc.selection_bytes_per_doc == comp.nbytes / len(comp)
        assert acc.selection_bytes_per_doc == comp.report()["bytes_per_doc"]
        # what the selection rule ranks by: ScoredCandidate.bytes_per_doc -> Candidate -> _rank_key
        assert (
            acc.selection_bytes_per_doc
            == score_candidate(Pipeline(factory()), docs, queries).bytes_per_doc
        )

    def test_the_parts_add_up(self, data, name):
        docs, _ = data
        factory, serving, append = CASES[name]
        pipe, comp, acc = account(factory, docs)
        assert (
            acc.code_bytes
            == comp.codes.nbytes
            == comp.num_vectors * pipe.quantizer.code_bytes(comp.dim)
        )
        assert acc.codec_overhead_bytes == pipe.quantizer.overhead_bytes()
        assert acc.code_bytes + acc.codec_overhead_bytes == comp.nbytes
        assert acc.fitted_state_bytes == serving and acc.append_state_bytes == append
        assert acc.offsets_bytes == 8 * (len(docs) + 1)
        assert acc.float32_bytes_before == docs.num_vectors * D * 4
        assert acc.n_docs == len(docs)
        total = (
            acc.code_bytes + acc.codec_overhead_bytes + acc.fitted_state_bytes + acc.offsets_bytes
        )
        assert acc.total_bytes_per_doc == total / len(docs)
        assert acc.total_bytes_per_doc >= acc.selection_bytes_per_doc

    def test_it_is_a_frozen_json_value(self, data, name):
        docs, _ = data
        _, _, acc = account(CASES[name][0], docs)
        assert isinstance(acc, ByteAccounting)
        with pytest.raises(dataclasses.FrozenInstanceError):
            acc.code_bytes = 0
        d = json.loads(json.dumps(acc.to_dict()))
        assert d["selection_bytes_per_doc"] == acc.selection_bytes_per_doc and d["n_docs"] == len(
            docs
        )
        assert d["total_bytes_per_doc"] == acc.total_bytes_per_doc


class TestWhatTheKeyOmits:
    def test_the_key_ignores_projection_matrices_and_the_new_total_does_not(self, data):
        docs, _ = data
        _, _, acc_plain = account(lambda: [TruncateProjector(K), BinaryQuantizer()], docs)
        _, _, acc_pca = account(lambda: [PCAProjector(K), BinaryQuantizer()], docs)
        assert (
            acc_plain.selection_bytes_per_doc == acc_pca.selection_bytes_per_doc
        )  # same codes, same key
        assert acc_pca.total_bytes_per_doc - acc_plain.total_bytes_per_doc == pytest.approx(
            4 * D * K / len(docs)
        )

    def test_a_rotation_costs_a_dxd_matrix_the_key_does_not_see(self, data):
        docs, _ = data
        _, _, plain = account(lambda: [BinaryQuantizer()], docs)
        _, _, rotated = account(lambda: [RotationProjector(0), BinaryQuantizer()], docs)
        assert rotated.selection_bytes_per_doc == plain.selection_bytes_per_doc
        assert rotated.fitted_state_bytes == 4 * D * D and plain.fitted_state_bytes == 0

    def test_lloyd2_counts_mu_and_sigma_but_not_the_regenerated_rotation(self, data):
        docs, _ = data
        pipe, _, acc = account(lambda: [Lloyd2Quantizer()], docs)
        assert acc.codec_overhead_bytes == 4 * D + 4
        assert acc.fitted_state_bytes == 0
        assert (
            pipe.quantizer.overhead_bytes() == fitted_state_nbytes(pipe.quantizer) + 4
        )  # the scalar sigma

    def test_codec_overhead_equals_the_array_state_for_mean_and_range_codecs(self, data):
        docs, _ = data
        for factory in (
            lambda: Int8Quantizer("per_dimension", center="mean"),
            lambda: Int4Quantizer(center="mean"),
            lambda: BinaryQuantizer(center="mean"),
        ):
            quantizer = factory()
            Pipeline([quantizer]).fit(docs)
            assert quantizer.overhead_bytes() == fitted_state_nbytes(quantizer) > 0

    def test_append_only_state_is_not_needed_to_score_stored_codes(self, data):
        docs, queries = data
        pipe, _, acc = account(CASES["pca(center)>int8"][0], docs)
        assert acc.append_state_bytes == 8 * D
        # queries are projected without the mean: scores do not read it
        before = pipe.transform_queries(queries).vectors.copy()
        pipe.stages[0].mean = np.full_like(pipe.stages[0].mean, 123.0)
        assert np.array_equal(pipe.transform_queries(queries).vectors, before)

    def test_a_projection_of_a_float16_corpus_is_counted_against_float32(self, data):
        docs, _ = data
        pipe = Pipeline([PCAProjector(K), BinaryQuantizer()])
        comp = pipe.compress(docs.astype(np.float16))
        acc = byte_accounting(comp, pipe)
        assert acc.float32_bytes_before == docs.num_vectors * D * 4  # float32 convention
        assert comp.stats["native_bytes_before"] == docs.num_vectors * D * 2


class TestChecks:
    def test_a_corpus_from_another_pipeline_is_refused(self, data):
        docs, _ = data
        _, comp, _ = account(lambda: [BinaryQuantizer()], docs)
        with pytest.raises(ValueError, match="not produced by this pipeline"):
            byte_accounting(comp, Pipeline([Int8Quantizer("per_vector")]))

    def test_an_unfitted_stage_with_state_is_an_error(self, data):
        docs, _ = data
        pipe, comp, _ = account(lambda: [PCAProjector(K), BinaryQuantizer()], docs)
        with pytest.raises(RuntimeError, match="not fitted"):
            byte_accounting(comp, Pipeline.from_dict(pipe.to_dict()))

    def test_a_loaded_corpus_is_accounted_the_same(self, data, tmp_path):
        docs, _ = data
        pipe, comp, acc = account(lambda: [PCAProjector(K), Int8Quantizer("per_vector")], docs)
        loaded = load_compressed(save_compressed(comp, tmp_path / "c.npz"))
        assert byte_accounting(loaded, pipe) == acc

    def test_a_corpus_without_a_recorded_pipeline_is_accepted(self, data):
        docs, _ = data
        pipe, comp, acc = account(lambda: [BinaryQuantizer()], docs)
        bare = CompressedCorpus(
            comp.codes, comp.offsets, comp.dim, comp.quantizer, comp.ids, stats=comp.stats
        )
        assert byte_accounting(bare, pipe) == acc

    def test_no_documents_do_not_divide_by_zero(self):
        empty = CompressedCorpus(np.zeros((0, 4), np.uint8), [0], 1, Float32Quantizer(), [])
        acc = byte_accounting(empty, Pipeline())
        assert (
            acc.n_docs == 0
            and acc.selection_bytes_per_doc == 0.0
            and acc.total_bytes_per_doc == 8.0
        )
        assert acc.selection_bytes_per_doc == empty.report()["bytes_per_doc"]


class _Weighted(DimensionReducer):
    """A third-party projection that exposes its state through get_state/set_state."""

    name = "test_weighted"

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
        self.weights = np.ones(corpus.dimension, dtype=np.float32)
        return self

    def project(self, vectors):
        return np.asarray(vectors, dtype=np.float32) * self.weights

    def get_state(self):
        return {"weights": self.weights}

    def set_state(self, state):
        self.weights = state["weights"]


def test_third_party_array_state_counts_as_serving_state(data):
    docs, _ = data
    _, _, acc = account(lambda: [_Weighted(), BinaryQuantizer()], docs)
    assert acc.fitted_state_bytes == 4 * D and acc.append_state_bytes == 0
