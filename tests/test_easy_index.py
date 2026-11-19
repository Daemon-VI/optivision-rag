"""Exact segmented search (optivision.easy._index): a multi-segment store ranks exactly like one
``CompressedRetriever`` over the whole corpus compressed in one go, ties included, and
``rank_float`` ranks candidate layouts the same way before they are stored (FINAL_DESIGN 9,
"Exactness"). Synthetic corpora only; nothing here measures a model."""

from __future__ import annotations

import functools
import itertools
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.core.retrieve.exact import CompressedRetriever, ExactRetriever
from optivision.easy._index import (
    SEARCH_BLOCK_BYTES,
    Searcher,
    as_rankings,
    merge_topk,
    rank_float,
    rank_float_many,
)
from optivision.easy._store import Store, settle, split
from optivision.representation import MultiVectorCorpus
from optivision.scoring import rank
from optivision.testing.synthetic import make_corpus, make_queries

LADDER = (
    "float16",
    "float32",
    "int8(per_vector)",
    "int4(mean)",
    "binary",
    "hierarchical_merge(0.33) > int8(per_vector)",
    "hierarchical_merge(0.33) > int4(mean)",
    "hierarchical_merge(0.33) > binary",
    "hierarchical_merge(0.25) > int8(per_vector)",
    "hierarchical_merge(0.25) > int4(mean)",
    "hierarchical_merge(0.25) > binary",
)


@functools.lru_cache(maxsize=1)
def _catalogue() -> dict[str, dict]:
    data = json.loads(resources.files("optivision").joinpath("presets", "measured.json").read_text(encoding="utf-8"))
    found: dict[str, dict] = {}

    def walk(o: object) -> None:
        if isinstance(o, dict):
            if "pipeline" in o and "label" in o:
                found.setdefault(o["label"], o["pipeline"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    return found


def _pipeline(label: str, corpus: MultiVectorCorpus) -> Pipeline:
    sample = np.sort(np.random.default_rng(0).choice(len(corpus), size=len(corpus) // 2, replace=False))
    return Pipeline.from_dict(json.loads(json.dumps(_catalogue()[label]))).fit(corpus.select(sample))


@functools.lru_cache(maxsize=2)
def _data(n_docs: int = 173) -> tuple[MultiVectorCorpus, MultiVectorCorpus]:
    corpus = make_corpus(n_docs, 32, tokens=(2, 12), seed=21, dtype=np.float16, anisotropy=1.0, mean_shift=0.5)
    queries, _ = make_queries(corpus, 15, tokens=(1, 6), noise=0.4, seed=22)
    return corpus, queries


def _store(path: Path, corpus: MultiVectorCorpus, p: Pipeline, pages: int) -> Store:
    store = Store.create(path, pipeline=p)
    for seg, codes in zip(split(corpus, pages), settle(split(corpus, pages), p), strict=True):
        store.append(codes, float16=seg)
    store.commit()
    store.close()
    return Store.open(path)


def _ids(results: list[list[tuple[str, float]]]) -> list[list[str]]:
    return [[pid for pid, _ in row] for row in results]


def _scores(results: list[list[tuple[str, float]]]) -> np.ndarray:
    return np.array([[s for _, s in row] for row in results], dtype=np.float32)


def _same(a: list[list[tuple[str, float]]], b: list[list[tuple[str, float]]]) -> None:
    """Same pages in the same order; scores equal to float32 rounding (different segment
    shapes are different BLAS calls; with equal shapes the tests compare with ==)."""
    assert _ids(a) == _ids(b)
    np.testing.assert_allclose(_scores(a), _scores(b), rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------- exactness


@pytest.mark.parametrize("label", LADDER)
def test_segmented_search_equals_one_retriever_over_the_whole_corpus(tmp_path, label):
    corpus, queries = _data()
    p = _pipeline(label, corpus)
    whole = CompressedRetriever(p.compress(corpus, fit=False), p, max_block_bytes=SEARCH_BLOCK_BYTES)
    with _store(tmp_path / "i.ovx", corpus, p, pages=37) as store:  # five uneven segments
        assert store.n_segments == 5
        searcher = Searcher(store)
        for k in (1, 7, len(corpus) + 5):
            want = whole.search(queries, k)
            got = searcher.search(queries, k)
            assert _ids(got) == want.ids
            np.testing.assert_allclose(_scores(got), want.scores, rtol=1e-6, atol=1e-6)
            positions, _ = searcher.search_positions(queries, k)
            np.testing.assert_array_equal(positions, want.indices)
        np.testing.assert_allclose(searcher.score(queries), whole.score(queries), rtol=1e-6, atol=1e-6)
        assert searcher.search(queries, 3) == searcher.search(queries, 3)  # repeatable, bitwise


def test_ties_are_broken_by_store_position_across_segments(tmp_path):
    # dyadic vectors make every product and sum exact, so duplicated pages tie exactly
    rng = np.random.default_rng(3)
    base = [rng.integers(-2, 3, size=(3, 8)).astype(np.float16) / np.float16(4) for _ in range(30)]
    arrays = base + base + base  # page j, j + 30 and j + 60 are copies, in three different segments
    corpus = MultiVectorCorpus.from_arrays(arrays, ids=[f"p{i:03d}" for i in range(90)])
    sources = (0, 7, 29)
    queries = MultiVectorCorpus.from_arrays([np.asarray(base[i][:2], dtype=np.float32) for i in sources])
    p = Pipeline.from_dict(_catalogue()["float32"])
    with _store(tmp_path / "t.ovx", corpus, p, pages=25) as store:
        got = Searcher(store).search(queries, 90)
    whole = CompressedRetriever(p.compress(corpus, fit=False), p).search(queries, 90)
    assert _ids(got) == whole.ids
    np.testing.assert_array_equal(_scores(got), whole.scores)  # exact arithmetic: bitwise everywhere
    for qi, row in zip(sources, got, strict=True):
        position = {pid: i for i, (pid, _) in enumerate(row)}
        copies = [f"p{qi:03d}", f"p{qi + 30:03d}", f"p{qi + 60:03d}"]
        assert len({dict(row)[c] for c in copies}) == 1  # the three copies tie
        assert position[copies[0]] < position[copies[1]] < position[copies[2]]
        for (a, sa), (b, sb) in itertools.pairwise(row):
            assert sa > sb or (sa == sb and a < b)  # every tie, in store order


def test_pages_without_vectors_sort_last_in_store_order(tmp_path):
    corpus = make_corpus(60, 16, tokens=(0, 4), seed=5, dtype=np.float16)
    empty = [corpus.ids[i] for i in np.flatnonzero(corpus.counts == 0)]
    assert len(empty) >= 3
    queries, _ = make_queries(corpus, 4, tokens=(1, 3), seed=6)
    p = _pipeline("int8(per_vector)", corpus)
    with _store(tmp_path / "e.ovx", corpus, p, pages=11) as store:
        got = Searcher(store).search(queries, len(corpus))
    whole = CompressedRetriever(p.compress(corpus, fit=False), p).search(queries, len(corpus))
    assert _ids(got) == whole.ids
    for row in got:
        assert [pid for pid, _ in row[-len(empty):]] == empty and all(s == -np.inf for _, s in row[-len(empty):])


def test_merge_topk_equals_one_rank_over_the_whole_matrix():
    rng = np.random.default_rng(9)
    scores = rng.integers(0, 6, size=(7, 103)).astype(np.float32)  # many exact ties
    scores[:, ::17] = -np.inf
    for k in (1, 5, 40, 200):
        best = None
        for lo in range(0, 103, 13):
            part = scores[:, lo:lo + 13]
            top = rank(part, min(k, part.shape[1]))
            best = merge_topk(best, (top + lo, np.take_along_axis(part, top, axis=1)), k)
        want = rank(scores, k)
        np.testing.assert_array_equal(best[0], want)
        np.testing.assert_array_equal(best[1], np.take_along_axis(scores, want, axis=1))


# ----------------------------------------------------------------------------- rank_float


@pytest.mark.parametrize("label", ["int4(mean)", "hierarchical_merge(0.25) > int8(per_vector)", "binary", "float16"])
def test_rank_float_ranks_a_candidate_exactly_like_the_stored_index(tmp_path, label):
    corpus, queries = _data()
    p = _pipeline(label, corpus)
    with _store(tmp_path / "r.ovx", corpus, p, pages=50) as store:
        stored = Searcher(store).search(queries, 8)
        assert rank_float(store.float16_segments(), queries, 8, p) == stored  # the kept copy, streamed
    assert rank_float(corpus, queries, 8, p, segment_pages=50) == stored  # one corpus, cut into views
    _same(rank_float(corpus, queries, 8, p, segment_pages=64), rank_float(corpus, queries, 8, p, segment_pages=7))


def test_rank_float_without_a_pipeline_is_the_float_reference():
    corpus, queries = _data()
    want = ExactRetriever(corpus).search(queries, 6)
    got = rank_float(corpus, queries, 6, segment_pages=40)
    assert _ids(got) == want.ids
    np.testing.assert_allclose(_scores(got), want.scores, rtol=1e-6, atol=1e-6)


def test_rank_float_many_shares_one_pass_and_matches_single_calls():
    corpus, queries = _data()
    labels = ("hierarchical_merge(0.33) > int8(per_vector)", "hierarchical_merge(0.33) > int4(mean)",
              "int8(per_vector)")
    pipelines = [_pipeline(label, corpus) for label in labels]
    many = rank_float_many(corpus, queries, 5, [*pipelines, None], segment_pages=45)
    assert many == [rank_float(corpus, queries, 5, p, segment_pages=45) for p in [*pipelines, None]]


def test_rank_float_refuses_an_unfitted_or_foreign_pipeline():
    corpus, queries = _data()
    with pytest.raises(RuntimeError, match="unfitted"):
        rank_float(corpus, queries, 5, Pipeline.from_dict(_catalogue()["int4(mean)"]))
    with pytest.raises(TypeError):
        rank_float(corpus, queries, 5, "int8")
    with pytest.raises(ValueError, match="k must be"):
        rank_float(corpus, queries, 0)


# ----------------------------------------------------------------------------- the searcher's surface


def test_rank_all_and_as_rankings_key_by_query_id(tmp_path):
    corpus, queries = _data()
    p = _pipeline("int8(per_vector)", corpus)
    with _store(tmp_path / "a.ovx", corpus, p, pages=60) as store:
        searcher = Searcher(store)
        ranked = searcher.rank_all(queries, 4)
        assert list(ranked) == queries.ids
        assert ranked == as_rankings(searcher.search(queries, 4), queries)
        assert all(len(v) == 4 for v in ranked.values())


def test_a_writer_searches_its_staged_segments_and_readers_see_commits(tmp_path):
    corpus, queries = _data()
    p = _pipeline("binary", corpus)
    head, tail = corpus.select(range(100)), corpus.select(range(100, len(corpus)))
    path = tmp_path / "w.ovx"
    with Store.create(path, pipeline=p) as s:
        s.append(p.compress(head, fit=False))
        s.commit()
    reader = Store.open(path)
    try:
        with Store.open(path, mode="a") as writer:
            writer.append(p.compress(tail, fit=False))
            assert Searcher(writer).search(queries, 5) == rank_float([head, tail], queries, 5, p)
            assert Searcher(reader).search(queries, 5) == rank_float([head], queries, 5, p)
            writer.commit()
        reader.reload()
        assert reader.n_pages == len(corpus)
        assert Searcher(reader).search(queries, 5) == rank_float([head, tail], queries, 5, p)
    finally:
        reader.close()


def test_searcher_input_checks_and_an_empty_index(tmp_path):
    corpus, queries = _data()
    p = _pipeline("int8(per_vector)", corpus)
    path = tmp_path / "x.ovx"
    with Store.create(path, pipeline=p) as s:
        s.commit()
    with Store.open(path) as s:
        searcher = Searcher(s)
        assert searcher.search(queries, 3) == [[] for _ in range(len(queries))]
        assert searcher.score(queries).shape == (len(queries), 0)
    with pytest.raises(TypeError):
        Searcher("not a store")
    with _store(tmp_path / "y.ovx", corpus, p, pages=80) as s:
        searcher = Searcher(s)
        with pytest.raises(ValueError, match="k must be"):
            searcher.search(queries, 0)
        wrong = MultiVectorCorpus.from_arrays([np.ones((2, 7), dtype=np.float32)])
        with pytest.raises(ValueError, match="dimension"):
            searcher.search(wrong, 3)
        with pytest.raises(ValueError, match="max_block_bytes"):
            Searcher(s, max_block_bytes=0)
        small = Searcher(s, max_block_bytes=4096)  # a tiny block budget changes nothing but speed
        assert _ids(small.search(queries, 5)) == _ids(searcher.search(queries, 5))
