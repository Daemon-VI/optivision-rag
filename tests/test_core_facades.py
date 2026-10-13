"""The core facades are the library's own objects, plus the identities added around them."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from optivision import compose, representation, scoring
from optivision.core import identity, pipeline, similarity, vectors
from optivision.core.pipeline import pipeline_identity, pipeline_spec
from optivision.representation import MultiVectorCorpus
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    Int8Quantizer,
    PCAProjector,
    Quantizer,
)

# ------------------------------------------------------------ same objects


@pytest.mark.parametrize("name", ["MultiVectorCorpus", "MultiVectorRepresentation", "as_corpus"])
def test_vectors_re_exports_the_representation_objects(name):
    assert getattr(vectors, name) is getattr(representation, name)


@pytest.mark.parametrize(
    "name", ["Pipeline", "CompressedCorpus", "save_compressed", "load_compressed"]
)
def test_pipeline_re_exports_the_compose_objects(name):
    assert getattr(pipeline, name) is getattr(compose, name)


@pytest.mark.parametrize("name", [
    "maxsim_matrix", "maxsim_pair", "rank", "segment_reduce", "DEFAULT_MAX_BLOCK_BYTES",
    "VectorStore",
])
def test_similarity_re_exports_the_scoring_objects(name):
    assert getattr(similarity, name) is getattr(scoring, name)


def test_the_re_exports_are_the_package_level_classes_too():
    import optivision

    assert vectors.MultiVectorCorpus is optivision.MultiVectorCorpus
    assert pipeline.Pipeline is optivision.Pipeline
    assert pipeline.CompressedCorpus is optivision.CompressedCorpus
    assert similarity.maxsim_matrix is optivision.maxsim_matrix


@pytest.mark.parametrize("module", [vectors, pipeline, similarity])
def test_every_listed_name_exists(module):
    assert all(hasattr(module, name) for name in module.__all__)


# -------------------------------------------------------- pipeline identity


def _pipe():
    return pipeline.Pipeline([AdaptiveMerge(radius=None, ratio=0.5), Int8Quantizer("per_vector")])


def test_pipeline_identity_is_the_digest_of_to_dict():
    p = _pipe()
    assert pipeline_identity(p) == identity.digest(p.to_dict())
    text = json.dumps(p.to_dict(), sort_keys=True, separators=(",", ":"))
    assert pipeline_identity(p) == hashlib.sha256(text.encode()).hexdigest()


def test_known_pipeline_identities_are_recomputable_without_the_library():
    spec = json.dumps({"stages": [{"stage": "float16"}]}, sort_keys=True, separators=(",", ":"))
    assert pipeline_identity(pipeline.Pipeline([Float16Quantizer()])) == (
        hashlib.sha256(spec.encode()).hexdigest())
    empty = hashlib.sha256(b'{"stages":[]}').hexdigest()
    assert pipeline_identity(pipeline.Pipeline([])) == empty


def test_pipeline_identity_ignores_fitted_state():
    corpus = _corpus()
    p = pipeline.Pipeline([PCAProjector(4), BinaryQuantizer(center="mean")])
    before = pipeline_identity(p)
    p.fit(corpus)
    assert p.fitted and pipeline_identity(p) == before
    other = pipeline.Pipeline([PCAProjector(4), BinaryQuantizer(center="mean")])
    other.fit(corpus.select([0, 1]))
    assert pipeline_identity(other) == before  # same specification, different fitted state


def test_pipeline_identity_follows_parameters_order_and_stages():
    base = pipeline_identity(_pipe())
    assert pipeline_identity(pipeline.Pipeline([AdaptiveMerge(radius=None, ratio=0.4),
                                                Int8Quantizer("per_vector")])) != base
    assert pipeline_identity(pipeline.Pipeline([AdaptiveMerge(radius=None, ratio=0.5),
                                                Int8Quantizer("per_dimension")])) != base
    assert pipeline_identity(pipeline.Pipeline([AdaptiveMerge(radius=None, ratio=0.5)])) != base
    swapped = pipeline.Pipeline([PCAProjector(4), AdaptiveMerge(radius=None, ratio=0.5)])
    reverse = pipeline.Pipeline([AdaptiveMerge(radius=None, ratio=0.5), PCAProjector(4)])
    assert pipeline_identity(swapped) != pipeline_identity(reverse)


def test_equal_specifications_have_one_identity():
    a = pipeline.Pipeline([AdaptiveMerge(radius=0.9), Int8Quantizer("per_vector")])
    b = pipeline.Pipeline.from_dict(a.to_dict())
    assert pipeline_identity(a) == pipeline_identity(b)


def test_pipeline_spec_is_a_deep_json_native_copy():
    p = _pipe()
    spec = pipeline_spec(p)
    assert spec == json.loads(json.dumps(p.to_dict()))
    json.dumps(spec)  # JSON-native
    spec["stages"][0]["ratio"] = 0.99
    spec["stages"].append({"stage": "float16"})
    assert p.to_dict()["stages"][0]["ratio"] == 0.5 and len(p.to_dict()["stages"]) == 2
    assert pipeline_spec(p) is not pipeline_spec(p)


def test_pipeline_spec_turns_numpy_values_into_python_ones():
    p = pipeline.Pipeline([PCAProjector(np.int64(4))])  # a caller may pass a NumPy integer
    assert isinstance(p.to_dict()["stages"][0]["dim"], np.integer)
    spec = pipeline_spec(p)
    assert type(spec["stages"][0]["dim"]) is int and spec["stages"][0]["dim"] == 4
    assert pipeline_identity(p) == pipeline_identity(pipeline.Pipeline([PCAProjector(4)]))


class _OddQuantizer(Quantizer):
    """A stage with a free-form parameter. Never registered, so the registry stays as it was."""

    name = "odd"

    def __init__(self, value):
        self.value = value

    def params(self):
        return {"value": self.value}

    def encode(self, vectors):
        return np.ascontiguousarray(vectors, dtype=np.float32).view(np.uint8)

    def decode(self, codes, dim):
        return codes.view(np.float32).reshape(-1, dim)

    def code_bytes(self, dim):
        return 4 * dim


@pytest.mark.parametrize(("value", "error"), [
    (float("nan"), ValueError), (float("inf"), ValueError), ({1, 2}, TypeError),
    (b"bytes", TypeError), ({1: "a"}, TypeError),
])
def test_a_parameter_without_a_json_form_is_refused(value, error):
    odd = pipeline.Pipeline([_OddQuantizer(value)])
    with pytest.raises(error):
        pipeline_spec(odd)
    with pytest.raises(error):
        pipeline_identity(odd)


def test_a_free_form_parameter_with_a_json_form_is_accepted():
    odd = pipeline.Pipeline([_OddQuantizer({"a": [1, (2, 3)], "b": np.float32(0.5)})])
    value = {"a": [1, [2, 3]], "b": 0.5}
    assert pipeline_spec(odd) == {"stages": [{"stage": "odd", "value": value}]}


def test_pipeline_spec_rebuilds_the_same_pipeline():
    p = _pipe()
    rebuilt = pipeline.Pipeline.from_dict(pipeline_spec(p))
    assert rebuilt.to_dict() == p.to_dict()
    assert pipeline_identity(rebuilt) == pipeline_identity(p)


def test_a_compressed_corpus_carries_the_identity_of_its_pipeline(tmp_path):
    corpus = _corpus()
    p = pipeline.Pipeline([Int8Quantizer("per_vector")])
    compressed = p.compress(corpus)
    assert compressed.pipeline == pipeline_spec(p)
    path = pipeline.save_compressed(compressed, tmp_path / "c.npz")
    again = pipeline.load_compressed(path)
    assert identity.digest(again.pipeline) == pipeline_identity(p)


# ---------------------------------------------------------------- similarity


def _corpus(seed=0, n_docs=6, dim=8):
    rng = np.random.default_rng(seed)
    counts = rng.integers(2, 6, n_docs)
    docs = [rng.standard_normal((int(n), dim)).astype(np.float32) for n in counts]
    return MultiVectorCorpus.from_arrays(docs, ids=[f"d{i}" for i in range(n_docs)])


def test_scores_are_float32_whatever_the_stored_dtype():
    docs, queries = _corpus(), _corpus(seed=1, n_docs=3)
    for dtype in (np.float16, np.float32, np.float64):
        scores = similarity.maxsim_matrix(queries.astype(dtype), docs.astype(dtype))
        assert scores.dtype == np.float32 and scores.shape == (3, 6)


def test_scores_are_the_textbook_definition():
    docs, queries = _corpus(), _corpus(seed=1, n_docs=3)
    scores = similarity.maxsim_matrix(queries, docs)
    expected = [[similarity.maxsim_pair(q.vectors, d.vectors) for d in docs] for q in queries]
    np.testing.assert_allclose(scores, np.array(expected, dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_scores_over_codes_are_exact_over_the_stored_representation_not_the_float_index():
    docs, queries = _corpus(dim=32), _corpus(seed=1, n_docs=3, dim=32)
    compressed = pipeline.Pipeline([BinaryQuantizer()]).compress(docs)
    scores = similarity.maxsim_matrix(queries, compressed)
    decoded = compressed.to_corpus()
    over_decoded = np.array([[similarity.maxsim_pair(q.vectors, d.vectors) for d in decoded]
                             for q in queries], dtype=np.float32)
    # exact over what is stored, and different from the float index it came from
    np.testing.assert_allclose(scores, over_decoded, rtol=1e-6, atol=1e-6)
    assert not np.allclose(scores, similarity.maxsim_matrix(queries, docs), atol=1e-3)


def test_rank_orders_by_descending_score_and_breaks_ties_by_ascending_position():
    scores = np.array([1.0, 3.0, 3.0, 2.0, 3.0], dtype=np.float32)
    assert similarity.rank(scores).tolist() == [1, 2, 4, 3, 0]
    assert similarity.rank(scores, k=2).tolist() == [1, 2]
    batch = np.stack([scores, scores[::-1]])
    assert similarity.rank(batch).tolist() == [[1, 2, 4, 3, 0], [0, 2, 3, 1, 4]]


def test_documents_with_equal_scores_keep_store_order_in_a_real_ranking():
    one = np.random.default_rng(3).standard_normal((4, 8)).astype(np.float32)
    docs = MultiVectorCorpus.from_arrays([one, one.copy(), one * 0.5, one.copy()])  # 0, 1, 3 tie
    queries = MultiVectorCorpus.from_arrays([one[:2]])
    scores = similarity.maxsim_matrix(queries, docs)
    assert scores[0, 0] == scores[0, 1] == scores[0, 3]
    order = similarity.rank(scores[0]).tolist()
    assert [i for i in order if i in (0, 1, 3)] == [0, 1, 3]


def test_empty_documents_score_minus_infinity_and_empty_queries_score_zero():
    ones, none = np.ones((2, 4), np.float32), np.zeros((0, 4), np.float32)
    docs = MultiVectorCorpus.from_arrays([ones, none])
    queries = MultiVectorCorpus.from_arrays([ones[:1], none])
    scores = similarity.maxsim_matrix(queries, docs)
    assert scores[0, 0] == 4.0 and scores[0, 1] == -np.inf
    assert scores[1, 0] == 0.0 and scores[1, 1] == -np.inf
    assert similarity.maxsim_pair(np.ones((1, 4)), np.zeros((0, 4))) == -np.inf
    assert similarity.rank(scores[0]).tolist() == [0, 1]


def test_the_block_budget_does_not_change_the_scores():
    docs, queries = _corpus(n_docs=9), _corpus(seed=2, n_docs=4)
    whole = similarity.maxsim_matrix(queries, docs)
    tight = similarity.maxsim_matrix(queries, docs, max_block_bytes=1)
    np.testing.assert_allclose(tight, whole, rtol=1e-6, atol=1e-6)
    assert similarity.DEFAULT_MAX_BLOCK_BYTES == 256 * 1024 * 1024


def test_a_compressed_corpus_and_a_float_corpus_are_both_vector_stores():
    corpus = _corpus()
    compressed = pipeline.Pipeline([Float16Quantizer()]).compress(corpus)
    for store in (corpus, compressed):
        assert hasattr(store, "offsets") and hasattr(store, "decode_rows") and len(store) == 6
