"""``optivision.testing``: synthetic data, basis transforms, the probe and the conformance checks.

Everything here is synthetic. The numbers describe the helpers and the code paths they
exercise; none of them is a statement about a real encoder.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from optivision import testing
from optivision.adapters import ArrayAdapter, CallableAdapter
from optivision.compose import Pipeline
from optivision.core.vectors import (
    EncoderIdentity,
    MultiVectorCorpus,
    corpus_identity,
    validate_corpus,
)
from optivision.evaluation import per_query_metrics, relevant_from_qrels
from optivision.scoring import maxsim_matrix
from optivision.stages import (
    STAGES,
    AdaptiveMerge,
    BinaryQuantizer,
    DimensionProjector,
    DimensionReducer,
    Float16Quantizer,
    Float32Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    Quantizer,
    RandomProjector,
    RandomPruner,
    RedundancyPruner,
    TruncateProjector,
)
from optivision.testing import (
    apply_basis,
    basis_probe,
    check_adapter,
    check_stage,
    make_corpus,
    make_queries,
    random_orthogonal,
    reorder_rows,
    signed_permutation,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _float_scores_close(a: np.ndarray, b: np.ndarray, tolerance: float = 1e-5) -> bool:
    """Equal up to float32 rounding, relative to the largest score."""
    return float(np.max(np.abs(a - b))) <= tolerance * float(np.max(np.abs(a)))


# ======================================================================= synthetic


def test_a_corpus_has_the_requested_shape_and_marks_itself_synthetic():
    corpus = make_corpus(40, 12, tokens=(5, 9), seed=0)
    assert len(corpus) == 40 and corpus.dimension == 12 and corpus.dtype == np.float32
    assert corpus.counts.min() >= 5 and corpus.counts.max() <= 9
    assert set(corpus.counts.tolist()) == {5, 6, 7, 8, 9}  # both ends of the range are reachable
    assert corpus.ids[:2] == ["d000000", "d000001"] and len(set(corpus.ids)) == 40
    assert corpus.attrs["synthetic"] is True and corpus.attrs["seed"] == 0
    validate_corpus(corpus)


def test_an_int_is_a_fixed_token_count_and_empty_documents_are_possible():
    assert set(make_corpus(10, 4, tokens=7, seed=1).counts.tolist()) == {7}
    sparse = make_corpus(60, 4, tokens=(0, 2), seed=1)
    assert (sparse.counts == 0).any() and sparse.counts.max() == 2
    validate_corpus(sparse)


def test_a_corpus_is_deterministic_per_seed_and_differs_between_seeds():
    a, b = make_corpus(20, 8, seed=5), make_corpus(20, 8, seed=5)
    assert corpus_identity(a) == corpus_identity(b)
    assert corpus_identity(a) != corpus_identity(make_corpus(20, 8, seed=6))


def test_vectors_are_unit_norm_unless_asked_not_to_be():
    unit = make_corpus(30, 16, seed=0, anisotropy=2.0, mean_shift=0.5)
    assert np.allclose(np.linalg.norm(unit.vectors, axis=1), 1.0, atol=1e-6)
    raw = make_corpus(30, 16, seed=0, unit_norm=False)
    assert not np.allclose(np.linalg.norm(raw.vectors, axis=1), 1.0, atol=1e-2)
    assert raw.attrs["unit_norm"] is False


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_the_dtype_is_the_one_asked_for(dtype):
    assert make_corpus(5, 4, seed=0, dtype=dtype).dtype == dtype


def test_the_draws_do_not_depend_on_the_options():
    """Same seed, different anisotropy or shift: the same underlying draws, so the signs agree."""
    plain = make_corpus(30, 16, seed=3)
    skewed = make_corpus(30, 16, seed=3, anisotropy=4.0)
    assert np.array_equal(np.sign(plain.vectors), np.sign(skewed.vectors))
    assert np.array_equal(plain.offsets, skewed.offsets)


def test_anisotropy_sets_the_spread_of_the_axes():
    corpus = make_corpus(4000, 8, tokens=10, seed=0, anisotropy=2.0, unit_norm=False)
    spread = corpus.vectors.std(axis=0)
    assert np.all(np.diff(spread) < 0)  # the leading coordinates carry the most variance
    assert spread[0] / spread[-1] == pytest.approx(np.exp(2.0), rel=0.1)
    flat = make_corpus(4000, 8, tokens=10, seed=0, unit_norm=False).vectors.std(axis=0)
    assert flat.max() / flat.min() < 1.1


def test_mean_shift_adds_a_common_direction():
    shifted = make_corpus(4000, 8, tokens=10, seed=0, mean_shift=2.0, unit_norm=False)
    assert np.linalg.norm(shifted.vectors.mean(axis=0)) == pytest.approx(2.0, rel=0.05)
    centred = make_corpus(4000, 8, tokens=10, seed=0, unit_norm=False)
    assert np.linalg.norm(centred.vectors.mean(axis=0)) < 0.1


@pytest.mark.parametrize("arguments", [
    {"n_docs": 0, "dim": 4}, {"n_docs": 4, "dim": 0}, {"n_docs": 4.5, "dim": 4},
    {"n_docs": 4, "dim": 4, "tokens": (5, 3)}, {"n_docs": 4, "dim": 4, "tokens": (0, 0)},
    {"n_docs": 4, "dim": 4, "tokens": "ab"}, {"n_docs": 4, "dim": 4, "anisotropy": -1.0},
    {"n_docs": 4, "dim": 4, "mean_shift": float("nan")},
    {"n_docs": 4, "dim": 4, "dtype": np.int32},
])
def test_bad_arguments_are_refused(arguments):
    with pytest.raises((ValueError, TypeError)):
        make_corpus(seed=0, **arguments)


def test_the_seed_is_required():
    with pytest.raises(TypeError):
        make_corpus(4, 4)  # no hidden randomness


def _ndcg(corpus, queries, qrels, metric="ndcg@5"):
    relevant = relevant_from_qrels(qrels, queries.ids, corpus.ids)
    scores = maxsim_matrix(queries, corpus)
    return float(np.mean(per_query_metrics(scores, relevant, [metric])[metric]))


def test_queries_come_with_labels_in_the_form_the_evaluation_accepts():
    corpus = make_corpus(60, 16, tokens=(8, 14), seed=0)
    queries, qrels = make_queries(corpus, 25, tokens=(3, 6), seed=1)
    assert len(queries) == 25 and queries.ids[0] == "q00000" and queries.dimension == 16
    assert set(qrels) == set(queries.ids)
    assert all(len(v) == 1 and next(iter(v)) in corpus.ids for v in qrels.values())
    relevant = relevant_from_qrels(qrels, queries.ids, corpus.ids)
    assert all(r.size == 1 for r in relevant)
    assert queries.counts.min() >= 3 and queries.counts.max() <= 6
    assert queries.attrs["synthetic"] is True


def test_without_noise_the_float_baseline_retrieves_the_target():
    corpus = make_corpus(60, 16, tokens=(8, 14), seed=0)
    queries, qrels = make_queries(corpus, 40, tokens=(3, 6), noise=0.0, seed=1)
    scores = maxsim_matrix(queries, corpus)
    relevant = relevant_from_qrels(qrels, queries.ids, corpus.ids)
    top1 = np.argmax(scores, axis=1)
    assert np.mean([top1[i] in relevant[i] for i in range(len(queries))]) >= 0.95
    assert _ndcg(corpus, queries, qrels) >= 0.95


def test_noise_makes_retrieval_harder():
    corpus = make_corpus(60, 16, tokens=(8, 14), seed=0)
    clean = _ndcg(corpus, *make_queries(corpus, 60, tokens=(2, 3), noise=0.0, seed=1))
    noisy = _ndcg(corpus, *make_queries(corpus, 60, tokens=(2, 3), noise=3.0, seed=1))
    assert noisy < clean


def test_noise_free_queries_are_rows_of_the_target_document():
    corpus = make_corpus(20, 8, tokens=(5, 9), seed=2)
    queries, qrels = make_queries(corpus, 20, tokens=(2, 4), noise=0.0, seed=3)
    for j, qid in enumerate(queries.ids):
        target = corpus[corpus.ids.index(next(iter(qrels[qid])))].vectors
        for row in queries[j].vectors:
            assert any(np.array_equal(row, t) for t in target)


def test_the_same_seed_gives_the_same_targets_whatever_the_noise():
    corpus = make_corpus(30, 8, seed=0)
    _, clean = make_queries(corpus, 12, noise=0.0, seed=4)
    _, noisy = make_queries(corpus, 12, noise=0.7, seed=4)
    assert clean == noisy


def test_targets_are_distinct_while_documents_last_and_repeat_after():
    corpus = make_corpus(10, 8, seed=0)
    _, few = make_queries(corpus, 10, seed=1)
    assert len({next(iter(v)) for v in few.values()}) == 10
    _, many = make_queries(corpus, 25, seed=1)
    assert len(many) == 25 and len({next(iter(v)) for v in many.values()}) <= 10


def test_queries_are_deterministic_per_seed():
    corpus = make_corpus(30, 8, seed=0)
    a, qa = make_queries(corpus, 10, noise=0.2, seed=9)
    b, qb = make_queries(corpus, 10, noise=0.2, seed=9)
    assert corpus_identity(a) == corpus_identity(b) and qa == qb
    c, _ = make_queries(corpus, 10, noise=0.2, seed=10)
    assert corpus_identity(a) != corpus_identity(c)


def test_noisy_queries_keep_the_norm_of_their_source_and_the_dtype_of_the_corpus():
    corpus = make_corpus(20, 16, seed=0, dtype=np.float16)
    queries, _ = make_queries(corpus, 10, noise=1.0, seed=1)
    assert queries.dtype == np.float16
    assert np.allclose(np.linalg.norm(queries.vectors.astype(np.float32), axis=1), 1.0, atol=5e-3)


def test_queries_are_never_cut_from_an_empty_document():
    corpus = make_corpus(40, 4, tokens=(0, 3), seed=0)
    _, qrels = make_queries(corpus, 20, tokens=(1, 2), seed=1)
    empty = {corpus.ids[i] for i in np.flatnonzero(corpus.counts == 0)}
    assert empty and not empty & {next(iter(v)) for v in qrels.values()}


@pytest.mark.parametrize("arguments", [{"n_queries": 0}, {"n_queries": 3, "noise": -0.1},
                                       {"n_queries": 3, "tokens": (0, 0)},
                                       {"n_queries": 3, "tokens": (0, 4)}])  # no empty queries
def test_bad_query_arguments_are_refused(arguments):
    with pytest.raises(ValueError):
        make_queries(make_corpus(5, 4, seed=0), seed=0, **arguments)


def test_queries_need_a_corpus_with_vectors():
    with pytest.raises(TypeError, match="MultiVectorCorpus"):
        make_queries([np.zeros((2, 2), np.float32)], 2, seed=0)
    nothing = MultiVectorCorpus(np.zeros((0, 4), np.float32), [0, 0, 0])
    with pytest.raises(ValueError, match="no document with vectors"):
        make_queries(nothing, 2, seed=0)


# ====================================================================== transforms


@pytest.mark.parametrize("dim", [1, 2, 7, 32])
def test_random_orthogonal_is_orthogonal_float64_and_deterministic(dim):
    q = random_orthogonal(dim, 3)
    assert q.shape == (dim, dim) and q.dtype == np.float64
    assert np.allclose(q @ q.T, np.eye(dim), atol=1e-12)
    assert np.array_equal(q, random_orthogonal(dim, 3))
    if dim > 1:
        assert not np.array_equal(q, random_orthogonal(dim, 4))


def test_random_orthogonal_follows_the_documented_convention():
    """QR of a standard normal matrix, columns signed by the diagonal of R."""
    g = np.random.default_rng(11).standard_normal((9, 9))
    q, r = np.linalg.qr(g)
    expected = q * np.where(np.diag(r) < 0, -1.0, 1.0)
    assert np.allclose(random_orthogonal(9, 11), expected, atol=1e-12)
    # equivalently: R of the signed factor has a positive diagonal
    assert np.all(np.diag(random_orthogonal(9, 11).T @ g) > 0)


def test_random_orthogonal_matches_the_rotation_stage_up_to_float32():
    seeded_rotation = pytest.importorskip("optivision.core.stages").seeded_rotation
    assert np.allclose(random_orthogonal(24, 5), seeded_rotation(24, 5), atol=1e-6)


@pytest.mark.parametrize("dim", [1, 3, 16])
def test_a_signed_permutation_has_one_plus_or_minus_one_per_row_and_column(dim):
    m = signed_permutation(dim, 2)
    assert m.shape == (dim, dim) and m.dtype == np.float64
    assert np.array_equal(np.abs(m).sum(axis=0), np.ones(dim))
    assert np.array_equal(np.abs(m).sum(axis=1), np.ones(dim))
    assert set(np.unique(m)) <= {-1.0, 0.0, 1.0}
    assert np.array_equal(m @ m.T, np.eye(dim))  # exactly orthogonal
    assert np.array_equal(m, signed_permutation(dim, 2))


def test_a_signed_permutation_permutes_and_flips_coordinates_exactly():
    m = signed_permutation(12, 0)
    x = np.arange(1.0, 13.0)[None, :]
    y = (x @ m)[0]
    assert sorted(np.abs(y)) == sorted(x[0]) and (y < 0).any()
    assert not np.array_equal(m, np.eye(12))


def test_an_orthogonal_basis_leaves_maxsim_scores_unchanged_within_float32_rounding():
    corpus = make_corpus(40, 32, tokens=(8, 16), seed=0, anisotropy=1.5)
    queries, _ = make_queries(corpus, 20, noise=0.3, seed=1)
    base = maxsim_matrix(queries, corpus)
    for matrix in (random_orthogonal(32, 1), random_orthogonal(32, 2), signed_permutation(32, 0)):
        moved = maxsim_matrix(apply_basis(queries, matrix), apply_basis(corpus, matrix))
        assert _float_scores_close(base, moved)
    # a matrix that is not orthogonal does change them (the test can fail)
    assert not _float_scores_close(base, maxsim_matrix(apply_basis(queries, 2 * np.eye(32)),
                                                       apply_basis(corpus, np.eye(32))))


def test_apply_basis_returns_a_new_corpus_and_leaves_the_input_alone():
    corpus = make_corpus(10, 8, seed=0)
    before = corpus_identity(corpus)
    moved = apply_basis(corpus, random_orthogonal(8, 0))
    assert corpus_identity(corpus) == before
    assert moved is not corpus and not np.shares_memory(moved.vectors, corpus.vectors)
    assert not np.allclose(moved.vectors, corpus.vectors)
    assert moved.dtype == corpus.dtype and moved.ids == corpus.ids
    assert np.array_equal(moved.offsets, corpus.offsets) and moved.offsets is not corpus.offsets


def test_apply_basis_keeps_the_aligned_arrays_and_does_not_share_them():
    rng = np.random.default_rng(0)
    corpus = MultiVectorCorpus(
        rng.standard_normal((7, 4)).astype(np.float32), [0, 3, 7], ids=["a", "b"],
        metadata=[{"k": 1}, {"k": 2}], protected=np.array([1, 0, 0, 0, 0, 1, 0], bool),
        positions=rng.random((7, 2)).astype(np.float32), weights=np.ones(7, np.float32),
        importance=rng.random(7).astype(np.float32), images=["image-a", "image-b"],
        attrs={"model": "m"},
    )
    moved = apply_basis(corpus, signed_permutation(4, 1))
    for key in ("protected", "positions", "weights", "importance"):
        assert np.array_equal(getattr(moved, key), getattr(corpus, key))
        assert not np.shares_memory(getattr(moved, key), getattr(corpus, key))
    assert moved.metadata == corpus.metadata and moved.images == corpus.images
    assert moved.attrs == {"model": "m"}
    moved.protected[:] = True
    assert corpus.protected.sum() == 2


@pytest.mark.parametrize(("dtype", "tolerance"), [(np.float64, 1e-12), (np.float32, 1e-5),
                                                  (np.float16, 5e-2)])
def test_apply_basis_computes_in_float64_and_casts_back(dtype, tolerance):
    corpus = make_corpus(12, 16, seed=0, dtype=dtype)
    matrix = random_orthogonal(16, 0)
    moved = apply_basis(corpus, matrix)
    assert moved.dtype == dtype
    expected = corpus.vectors.astype(np.float64) @ matrix
    assert np.max(np.abs(moved.vectors.astype(np.float64) - expected)) <= tolerance
    back = apply_basis(moved, matrix.T)
    assert np.max(np.abs(back.vectors.astype(np.float64) - corpus.vectors)) <= 2 * tolerance


def test_apply_basis_gives_the_same_result_in_any_block_size(monkeypatch):
    from optivision.testing import transforms

    corpus = make_corpus(30, 16, seed=0)
    matrix = random_orthogonal(16, 2)
    whole = apply_basis(corpus, matrix).vectors
    monkeypatch.setattr(transforms, "_BLOCK_BYTES", 8 * 16 * 3)  # three rows at a time
    assert np.array_equal(apply_basis(corpus, matrix).vectors, whole)


def test_apply_basis_refuses_what_it_cannot_apply():
    corpus = make_corpus(5, 8, seed=0)
    with pytest.raises(ValueError, match=r"\[8, 8\]"):
        apply_basis(corpus, np.eye(7))
    with pytest.raises(ValueError, match="non-finite"):
        apply_basis(corpus, np.full((8, 8), np.nan))
    with pytest.raises(TypeError, match="MultiVectorCorpus"):
        apply_basis(np.zeros((3, 8)), np.eye(8))
    assert np.array_equal(apply_basis(corpus, np.eye(8)).vectors, corpus.vectors)


def test_reordering_rows_keeps_every_document_as_a_set_and_every_score():
    corpus = make_corpus(20, 8, tokens=(4, 9), seed=0)
    queries, _ = make_queries(corpus, 10, seed=1)
    moved = reorder_rows(corpus, 7)
    assert np.array_equal(moved.offsets, corpus.offsets) and moved.ids == corpus.ids
    assert not np.array_equal(moved.vectors, corpus.vectors)
    for i in range(len(corpus)):
        a, b = corpus[i].vectors, moved[i].vectors
        assert np.array_equal(a[np.lexsort(a.T)], b[np.lexsort(b.T)])
    assert np.allclose(maxsim_matrix(queries, corpus), maxsim_matrix(queries, moved), atol=1e-6)


def test_reordering_rows_carries_the_row_annotations_with_the_rows():
    base = make_corpus(6, 4, tokens=(5, 8), seed=0)
    tagged = MultiVectorCorpus(
        base.vectors, base.offsets, ids=base.ids, weights=base.vectors[:, 0].copy(),
        importance=base.vectors[:, 1].copy(), protected=base.vectors[:, 2] > 0,
        positions=np.stack([base.vectors[:, 0], base.vectors[:, 3]], axis=1),
    )
    moved = reorder_rows(tagged, 1)
    assert not np.array_equal(moved.vectors, tagged.vectors)
    assert np.array_equal(moved.weights, moved.vectors[:, 0])
    assert np.array_equal(moved.importance, moved.vectors[:, 1])
    assert np.array_equal(moved.protected, moved.vectors[:, 2] > 0)
    assert np.array_equal(moved.positions, np.stack([moved.vectors[:, 0], moved.vectors[:, 3]], 1))


def test_reordering_is_deterministic_leaves_the_input_alone_and_handles_empty_documents():
    corpus = make_corpus(30, 4, tokens=(0, 5), seed=0)
    before = corpus_identity(corpus)
    a, b = reorder_rows(corpus, 3), reorder_rows(corpus, 3)
    assert corpus_identity(corpus) == before and corpus_identity(a) == corpus_identity(b)
    assert corpus_identity(a) != corpus_identity(reorder_rows(corpus, 4))
    assert (a.counts == 0).any()
    with pytest.raises(TypeError, match="MultiVectorCorpus"):
        reorder_rows("corpus", 0)


# ========================================================================= probes


def _data(anisotropy=3.0, dtype=np.float32):
    corpus = make_corpus(60, 32, tokens=(10, 18), seed=0, anisotropy=anisotropy, dtype=dtype)
    queries, qrels = make_queries(corpus, 30, tokens=(4, 8), noise=0.3, seed=1)
    return corpus, queries, qrels


def test_a_sign_code_moves_more_than_a_float_pipeline_on_anisotropic_data():
    corpus, queries, _ = _data()
    float_report = basis_probe(Pipeline([]), corpus, queries)
    sign_report = basis_probe(Pipeline([BinaryQuantizer()]), corpus, queries)
    f, s = float_report["summary"]["rotation"], sign_report["summary"]["rotation"]
    assert float_report["valid_symmetry"] and sign_report["valid_symmetry"]
    assert s["max_relative_score_change"] > 100 * max(f["max_relative_score_change"], 1e-9)
    assert s["mean_top_k_overlap"] < f["mean_top_k_overlap"]
    assert f["mean_top_k_overlap"] == 1.0


def test_a_signed_permutation_does_not_move_a_sign_code_but_a_rotation_does():
    corpus, queries, _ = _data()
    summary = basis_probe(Pipeline([BinaryQuantizer()]), corpus, queries)["summary"]
    permuted, rotated = summary["signed_permutation"], summary["rotation"]
    assert permuted["max_relative_score_change"] < rotated["max_relative_score_change"] / 100
    assert permuted["mean_top_k_overlap"] == 1.0 and rotated["min_top_k_overlap"] < 1.0


def test_the_report_is_complete_json_and_deterministic():
    corpus, queries, _ = _data()
    pipeline = Pipeline([Int4Quantizer()])
    report = basis_probe(pipeline, corpus, queries, seeds=(0, 1), k=5)
    assert json.loads(json.dumps(report)) == report
    assert basis_probe(pipeline, corpus, queries, seeds=(0, 1), k=5) == report
    assert report["k"] == 5 and report["n_documents"] == 60 and report["n_queries"] == 30
    assert report["dimension"] == 32 and report["dtype"] == "float32"
    assert report["pipeline"] == pipeline.to_dict() and report["label"] == "int4"
    assert [(r["transform"], r["seed"]) for r in report["runs"]] == [
        ("signed_permutation", 0), ("signed_permutation", 1), ("rotation", 0), ("rotation", 1)]
    assert set(report["runs"][0]) == {"transform", "seed", "max_relative_score_change",
                                     "top_k_overlap", "float_baseline_max_relative_change",
                                     "float_baseline_unchanged"}
    assert report["score_span"] > 0 and report["valid_symmetry"] is True
    assert "note" not in report


def test_k_is_capped_by_the_number_of_documents():
    corpus, queries, _ = _data()
    small = corpus.select(range(4))
    assert basis_probe(Pipeline([]), small, queries, k=10, seeds=(0,))["k"] == 4


def test_the_pipeline_and_the_data_are_not_touched():
    corpus, queries, _ = _data()
    before = (corpus_identity(corpus), corpus_identity(queries))
    pipeline = Pipeline([Int8Quantizer(scale="per_dimension")])
    basis_probe(pipeline, corpus, queries, seeds=(0,))
    assert (corpus_identity(corpus), corpus_identity(queries)) == before
    assert not pipeline.fitted and pipeline.stages[0].to_dict() == Int8Quantizer(
        scale="per_dimension").to_dict()


def test_a_fitted_pipeline_is_rebuilt_not_reused():
    corpus, queries, _ = _data()
    pipeline = Pipeline([PCAProjector(16), Int8Quantizer(scale="per_vector")]).fit(corpus)
    components = pipeline.stages[0].matrix.copy() if hasattr(pipeline.stages[0], "matrix") else None
    report = basis_probe(pipeline, corpus, queries, seeds=(0,), transforms=("rotation",))
    assert pipeline.fitted
    if components is not None:
        assert np.array_equal(pipeline.stages[0].matrix, components)
    # a refit PCA is canonical up to sign, so the change in the pipeline's scores is small
    # next to a sign code's (no ordering is claimed beyond that here)
    assert report["valid_symmetry"] is True


def test_the_declared_invariance_is_reported_next_to_the_measured_change():
    corpus, queries, _ = _data()
    float_report = basis_probe(Pipeline([]), corpus, queries, seeds=(0,))
    sign_report = basis_probe(Pipeline([BinaryQuantizer()]), corpus, queries, seeds=(0,))
    assert float_report["declared_basis_invariance"] == "orthogonal"
    assert sign_report["declared_basis_invariance"] == "signed_permutation"
    assert float_report["declared_conditions"] == []
    assert any(c.startswith("binary: ") for c in sign_report["declared_conditions"])


def test_the_probe_works_without_the_traits_module(monkeypatch):
    corpus, queries, _ = _data()
    monkeypatch.setitem(sys.modules, "optivision.core.stages", None)  # an import now fails
    report = basis_probe(Pipeline([]), corpus, queries, seeds=(0,))
    assert report["declared_basis_invariance"] is None and report["valid_symmetry"] is True
    assert report["declared_conditions"] is None


def test_a_transform_that_is_not_a_symmetry_is_caught_by_the_float_baseline(monkeypatch):
    from optivision.testing import probes

    corpus, queries, _ = _data()
    monkeypatch.setitem(probes._TRANSFORMS, "rotation",
                        lambda dim, seed: np.random.default_rng(seed).standard_normal((dim, dim)))
    report = basis_probe(Pipeline([]), corpus, queries, seeds=(0,))
    assert report["valid_symmetry"] is False and "should not be interpreted" in report["note"]
    assert report["summary"]["rotation"]["float_baseline_unchanged"] is False
    assert report["summary"]["signed_permutation"]["float_baseline_unchanged"] is True


def test_the_float_tolerance_decides_whether_rounding_counts_as_a_change():
    """float16 vectors are rounded again after a rotation, so the float scores move by more
    than float32 rounding; a tolerance tighter than that calls the transform invalid."""
    kwargs = {"seeds": (0,), "transforms": ("rotation",)}
    half = basis_probe(Pipeline([]), *_data(dtype=np.float16)[:2], **kwargs)
    single = basis_probe(Pipeline([]), *_data()[:2], **kwargs)
    change = "float_baseline_max_relative_change"
    assert half["runs"][0][change] > 100 * single["runs"][0][change]
    assert half["valid_symmetry"] and single["valid_symmetry"]  # both below the default 1e-4
    tight = basis_probe(Pipeline([]), *_data(dtype=np.float16)[:2], float_tolerance=1e-6, **kwargs)
    assert tight["valid_symmetry"] is False and "note" in tight
    assert basis_probe(Pipeline([]), *_data()[:2], float_tolerance=1e-6, **kwargs)["valid_symmetry"]


@pytest.mark.parametrize("arguments", [
    {"transforms": ("shear",)}, {"transforms": ()}, {"seeds": ()}, {"seeds": (-1,)},
    {"seeds": (0.5,)}, {"k": 0}, {"float_tolerance": -1.0}, {"float_tolerance": float("nan")},
])
def test_the_probe_refuses_bad_arguments(arguments):
    corpus, queries, _ = _data()
    with pytest.raises(ValueError):
        basis_probe(Pipeline([]), corpus, queries, **arguments)


def test_the_probe_refuses_what_it_cannot_compare():
    corpus, queries, _ = _data()
    with pytest.raises(TypeError, match="Pipeline"):
        basis_probe(BinaryQuantizer(), corpus, queries)
    with pytest.raises(TypeError, match="MultiVectorCorpus"):
        basis_probe(Pipeline([]), corpus.vectors, queries)
    with pytest.raises(ValueError, match="dimension"):
        basis_probe(Pipeline([]), corpus, make_corpus(5, 8, seed=0))
    single = make_corpus(1, 32, tokens=1, seed=0)
    one, _ = make_queries(single, 1, tokens=1, seed=0)
    with pytest.raises(ValueError, match="zero span"):
        basis_probe(Pipeline([]), single, one)


# =================================================================== check_adapter


def _arrays(n=3, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        v = rng.standard_normal((2 + i, dim)).astype(np.float32)
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return out


class _Adapter:
    """A configurable adapter for the checks to find fault with."""

    def __init__(self, dim=8, **faults):
        self.dim, self.faults, self.calls = dim, faults, 0

    def _corpus(self, inputs, ids):
        self.calls += 1
        arrays = _arrays(len(inputs), self.dim)
        if self.faults.get("jitter"):
            arrays = [a + np.float32(self.calls * 1e-3) for a in arrays]
        if "zero_row" in self.faults:
            arrays[0] = np.vstack([arrays[0], np.zeros((1, self.dim), np.float32)])
        if "nan" in self.faults:
            arrays[0][0, 0] = np.nan
        if self.faults.get("drop"):
            arrays = arrays[:-1]
        corpus = MultiVectorCorpus.from_arrays(arrays, ids=None if self.faults.get("drop") else ids)
        if self.faults.get("ignore_ids"):
            corpus.ids = [f"fixed{i}" for i in range(len(corpus))]
        if self.faults.get("duplicate_ids"):
            corpus.ids = ["same"] * len(corpus)
        return corpus

    def encode_documents(self, documents, ids=None):
        return self._corpus(documents, ids)

    def encode_queries(self, queries, ids=None):
        return self._corpus(queries, ids)

    def identity(self):
        if "identity" in self.faults:
            return self.faults["identity"]
        return None

    @property
    def dimension(self):
        return self.faults.get("dimension")


def _check(adapter, **options):
    return check_adapter(adapter, ["a", "b", "c"], ["x", "y"], **options)


def test_a_conforming_adapter_has_no_problems():
    identity = EncoderIdentity(model_id="x/y", dimension=8, modality="text", normalized=True)
    adapter = ArrayAdapter(identity)
    assert check_adapter(adapter, _arrays(3), _arrays(2, seed=1)) == []
    assert _check(_Adapter(identity=identity, dimension=8)) == []
    assert _check(_Adapter()) == []


def test_the_bring_your_own_adapters_conform_to_their_own_checks():
    docs, queries = _arrays(4), _arrays(3, seed=1)
    assert check_adapter(ArrayAdapter(), docs, queries) == []
    adapter = CallableAdapter(lambda items: _arrays(len(items)), lambda items: _arrays(len(items)))
    assert check_adapter(adapter, ["a", "b"], ["q"]) == []


@pytest.mark.parametrize(("faults", "fragment"), [
    ({"nan": True}, "non-finite value"),
    ({"duplicate_ids": True}, "is used by documents 0 and 1"),
    ({"drop": True}, "returned 2 documents for 3 inputs"),
    ({"ignore_ids": True}, "does not honour ids="),
    ({"jitter": True}, "is not deterministic"),
    ({"zero_row": True}, "vectors are all zero"),
])
def test_each_fault_is_found_and_named(faults, fragment):
    problems = _check(_Adapter(**faults))
    assert any(fragment in p for p in problems), problems


def test_several_problems_are_reported_together_and_ids_and_values_separately():
    problems = _check(_Adapter(nan=True, duplicate_ids=True))
    assert any("non-finite" in p for p in problems) and any("is used by" in p for p in problems)


def test_a_tolerance_lets_a_slightly_non_reproducible_model_pass():
    jittery = _Adapter(jitter=True)
    assert any("not deterministic" in p for p in _check(jittery))
    assert not any("not deterministic" in p for p in _check(_Adapter(jitter=True), atol=1.0))


def test_a_non_corpus_and_a_failure_to_encode_are_problems_not_crashes():
    class Lists(_Adapter):
        def encode_documents(self, documents, ids=None):
            return _arrays(len(documents))

    class Raises(_Adapter):
        def encode_queries(self, queries, ids=None):
            raise RuntimeError("out of memory")

    assert any("returned a list, not a MultiVectorCorpus" in p for p in _check(Lists()))
    assert any("encode_queries raised RuntimeError: out of memory" in p for p in _check(Raises()))
    assert check_adapter(object(), ["a"], ["b"]) == [
        "the adapter has no callable encode_documents()",
        "the adapter has no callable encode_queries()",
    ]


def test_a_floating_dtype_is_required_and_a_broken_corpus_is_reported_once():
    class Integers(_Adapter):
        def encode_documents(self, documents, ids=None):
            corpus = super().encode_documents(documents, ids)
            corpus.vectors = (corpus.vectors * 100).astype(np.int16)
            return corpus

    problems = _check(Integers())
    assert any("vectors must be floating point" in p for p in problems), problems
    assert not any("non-finite" in p for p in problems)


def test_documents_and_queries_must_share_a_dimension():
    class Mixed(_Adapter):
        def encode_queries(self, queries, ids=None):
            self.dim = 6
            return super().encode_queries(queries, ids)

    assert any("documents have dimension 8 but queries have 6" in p for p in _check(Mixed()))


def test_identity_and_dimension_must_agree_with_the_vectors():
    wrong = EncoderIdentity(model_id="x/y", dimension=16, modality="text")
    problems = _check(_Adapter(identity=wrong, dimension=4))
    assert any("identity() says 16 but the documents have dimension 8" in p for p in problems)
    assert any("dimension says 4 but the queries have dimension 8" in p for p in problems)
    assert any("identity() returned a str" in p for p in _check(_Adapter(identity="x")))

    class Broken(_Adapter):
        def identity(self):
            raise RuntimeError("no model loaded")

    assert any("identity() raised RuntimeError" in p for p in _check(Broken()))


def test_a_claim_of_unit_norm_is_checked_against_the_vectors():
    claim = EncoderIdentity(model_id="x/y", dimension=8, modality="text", normalized=True)

    class Scaled(_Adapter):
        def _corpus(self, inputs, ids):
            corpus = super()._corpus(inputs, ids)
            corpus.vectors = corpus.vectors * np.float32(1.5)
            return corpus

    scaled = Scaled(identity=claim)
    assert any("normalized=True" in p and "documents" in p for p in _check(scaled))
    no_claim = Scaled(identity=EncoderIdentity(model_id="x/y", dimension=8, modality="text"))
    assert not any("normalized" in p for p in _check(no_claim))
    assert not any("normalized" in p for p in _check(Scaled(identity=claim), norm_tolerance=1.0))


def test_there_must_be_something_to_encode():
    with pytest.raises(ValueError, match="at least one document and one query"):
        check_adapter(ArrayAdapter(), [], _arrays(1))


# ==================================================================== check_stage


def _stage_data():
    corpus = make_corpus(14, 16, tokens=(6, 12), seed=3, anisotropy=1.0)
    queries, _ = make_queries(corpus, 8, tokens=(3, 5), noise=0.2, seed=4)
    return corpus, queries


BUILT_IN = [
    Float32Quantizer(), Float16Quantizer(), BinaryQuantizer(), BinaryQuantizer(center="mean"),
    Int8Quantizer(scale="per_vector"), Int8Quantizer(scale="per_dimension"), Int4Quantizer(),
    Lloyd2Quantizer(), PCAProjector(8), RandomProjector(8), TruncateProjector(8),
    DimensionProjector(8, method="truncate"), AdaptiveMerge(radius=0.5),
    HierarchicalMerge(ratio=0.5), RandomPruner(ratio=0.5), RedundancyPruner(threshold=0.9),
]


@pytest.mark.parametrize("stage", BUILT_IN, ids=[repr(s)[:40] for s in BUILT_IN])
def test_the_built_in_stages_conform(stage):
    corpus, queries = _stage_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert check_stage(stage, corpus) == []
        assert check_stage(stage, corpus, queries) == []


def test_the_rotation_stage_conforms():
    rotation = pytest.importorskip("optivision.core.stages").RotationProjector(seed=2)
    corpus, queries = _stage_data()
    assert check_stage(rotation, corpus, queries) == []


def test_a_stage_that_needs_queries_to_fit_says_so_and_conforms_when_given_them():
    joint = PCAProjector(8, basis="joint")
    corpus, queries = _stage_data()
    assert check_stage(joint, corpus, queries) == []
    problems = check_stage(joint, corpus)
    assert len(problems) == 1 and "needs sample queries" in problems[0]


def test_the_stage_passed_in_is_not_fitted_or_changed():
    corpus, queries = _stage_data()
    stage = PCAProjector(8)
    assert check_stage(stage, corpus, queries) == []
    assert not stage.fitted


class _Scale(DimensionReducer):
    """A well-behaved stage to break one property at a time."""

    name = "test_scale"
    traits = None  # replaced by the tests that declare traits

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def params(self):
        return {"factor": self.factor}

    def output_dim(self, input_dim):
        return input_dim

    def project(self, vectors):
        return vectors * np.float32(self.factor)


def _declared(cls=_Scale, **overrides):
    """``cls`` with declared traits (orthogonal-invariant unless overridden)."""
    core = pytest.importorskip("optivision.core.stages")
    fields = {"name": cls.name, "kind": "dimensions",
              "basis_invariance": core.BasisInvariance.NONE, **overrides}
    return type(cls.__name__, (cls,), {"traits": core.StageTraits(**fields), "name": cls.name})


@pytest.fixture
def registered(monkeypatch):
    """Register a stage class under its own name for one test."""

    def register(cls):
        monkeypatch.setitem(STAGES, cls.name, cls)
        return cls

    return register


def test_a_well_formed_new_stage_conforms(registered):
    cls = registered(_declared())
    corpus, queries = _stage_data()
    assert check_stage(cls(factor=3.0), corpus, queries) == []


def test_an_unregistered_stage_is_reported():
    corpus, _ = _stage_data()
    problems = check_stage(_declared()(), corpus)
    assert any("'test_scale' is not registered" in p for p in problems)


def test_a_name_taken_by_another_class_is_reported(registered):
    registered(_Scale)
    corpus, _ = _stage_data()
    problems = check_stage(_declared()(), corpus)  # same name, a subclass nobody registered
    assert any("is registered for _Scale, not for _Scale" in p for p in problems)


def test_a_stage_with_no_declared_traits_is_reported(registered):
    registered(_Scale)
    corpus, _ = _stage_data()
    problems = check_stage(_Scale(), corpus)
    assert len(problems) == 1 and "no basis traits are declared for 'test_scale'" in problems[0]
    assert "register_traits" in problems[0]


def test_traits_of_the_wrong_kind_are_reported(registered):
    cls = registered(_declared(kind="bits"))
    corpus, _ = _stage_data()
    problems = check_stage(cls(), corpus)
    assert problems == ["the traits say kind 'bits' but this is a 'dimensions' stage"]


def test_a_stage_that_is_not_deterministic_is_reported(registered):
    class Noisy(_Scale):
        name = "test_noisy"

        def project(self, vectors):
            noise = np.random.default_rng().standard_normal(vectors.shape)
            return vectors + noise.astype(np.float32)

    cls = registered(_declared(Noisy))
    corpus, _ = _stage_data()
    problems = check_stage(cls(), corpus)
    assert any("not deterministic" in p for p in problems), problems


def test_a_stage_that_changes_its_input_is_reported(registered):
    class Writes(_Scale):
        name = "test_writes"

        def project(self, vectors):
            vectors *= np.float32(2.0)  # in place, on a view of the caller's data
            return vectors

        def transform(self, corpus):
            return corpus.with_vectors(self.project(corpus.vectors))

    cls = registered(_declared(Writes))
    corpus, queries = _stage_data()
    problems = check_stage(cls(), corpus, queries)
    assert "the stage modified the corpus it was given" in problems
    assert "the stage modified the queries it was given" in problems


def test_a_description_that_loses_a_parameter_is_reported(registered):
    class Forgets(_Scale):
        name = "test_forgets"

        def params(self):
            return {}  # the factor is not written down

    cls = registered(_declared(Forgets))
    corpus, _ = _stage_data()
    problems = check_stage(cls(factor=5.0), corpus)
    assert any("rebuilt by stage_from_dict behaves differently" in p for p in problems), problems


def test_a_description_that_is_not_json_is_reported(registered):
    class Opaque(_Scale):
        name = "test_opaque"

        def params(self):
            return {"factor": {self.factor}}  # a set

    cls = registered(_declared(Opaque))
    corpus, _ = _stage_data()
    problems = check_stage(cls(), corpus)
    assert any("to_dict() is not JSON-native" in p for p in problems), problems


def test_a_stage_that_raises_is_reported_with_its_message(registered):
    class Fails(_Scale):
        name = "test_fails"

        def project(self, vectors):
            raise RuntimeError("needs page images")

    cls = registered(_declared(Fails))
    corpus, _ = _stage_data()
    assert check_stage(cls(), corpus) == [
        "fitting and applying the stage raised RuntimeError: needs page images"]


def test_reading_the_queries_at_fit_time_must_be_declared(registered):
    class Peeks(_Scale):
        name = "test_peeks"

        def fit(self, corpus, queries=None):
            self.shift = 0.0 if queries is None else float(queries.vectors.mean())
            return self

        def project(self, vectors):
            return vectors + np.float32(getattr(self, "shift", 0.0))

    corpus, queries = _stage_data()
    undeclared = registered(_declared(Peeks, fit_uses_queries=False))
    problems = check_stage(undeclared(), corpus, queries)
    assert any("do not declare fit_uses_queries" in p for p in problems), problems
    declared = _declared(Peeks, fit_uses_queries=True)
    registered(declared)
    assert check_stage(declared(), corpus, queries) == []


def test_the_traits_checks_are_skipped_with_a_warning_when_the_module_is_missing(monkeypatch):
    corpus, _ = _stage_data()
    monkeypatch.setitem(sys.modules, "optivision.core.stages", None)
    with pytest.warns(UserWarning, match="declared-traits checks were skipped"):
        assert check_stage(Int8Quantizer(scale="per_vector"), corpus) == []


def test_check_stage_refuses_things_that_are_not_stages_or_corpora():
    corpus, _ = _stage_data()
    with pytest.raises(TypeError, match="Stage"):
        check_stage("int8", corpus)
    with pytest.raises(TypeError, match="corpus"):
        check_stage(Int8Quantizer(), corpus.vectors)
    with pytest.raises(TypeError, match="queries"):
        check_stage(Int8Quantizer(), corpus, queries=[1, 2])


def test_a_quantizer_written_for_the_test_is_exercised_like_any_other(registered):
    class Round(Quantizer):
        name = "test_round"

        def params(self):
            return {}

        def encode(self, vectors):
            return np.clip(np.round(vectors * 100) + 128, 0, 255).astype(np.uint8)

        def decode(self, codes, dim):
            return (codes.astype(np.float32) - 128) / 100

        def code_bytes(self, dim):
            return dim

    cls = registered(_declared(Round, kind="bits"))
    corpus, queries = _stage_data()
    assert check_stage(cls(), corpus, queries) == []


# ======================================================================= the package


def test_the_package_exports_exactly_what_it_documents():
    assert sorted(testing.__all__) == sorted([
        "apply_basis", "basis_probe", "check_adapter", "check_stage", "make_corpus",
        "make_queries", "random_orthogonal", "reorder_rows", "signed_permutation"])
    assert all(callable(getattr(testing, name)) for name in testing.__all__)


def test_it_imports_without_pytest_torch_or_any_model_library():
    code = (
        "import sys\n"
        "for name in ('pytest', 'torch', 'transformers', 'qdrant_client', 'scipy'):\n"
        "    sys.modules[name] = None\n"
        "import optivision.testing as t\n"
        "corpus = t.make_corpus(5, 4, seed=0)\n"
        "queries, qrels = t.make_queries(corpus, 2, seed=1)\n"
        "assert len(queries) == 2 and corpus.dimension == 4\n"
        "print('ok')\n"
    )
    path = os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])
    done = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": path},
                          capture_output=True, text=True, check=False, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
