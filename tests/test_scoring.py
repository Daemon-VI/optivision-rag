from __future__ import annotations

import numpy as np
import pytest

from optivision.index.numpy_index import NumpyIndex
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, maxsim_pair, rank, segment_reduce
from optivision.types import CompressedPage, PageRef


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _brute(queries, docs):
    return np.array([[maxsim_pair(q, d) for d in docs] for q in queries], dtype=np.float64)


@pytest.fixture
def case(rng):
    # empty documents and empty queries at the start, middle and end on purpose
    docs = [_unit(rng, n, 16) for n in (0, 4, 1, 0, 9, 3, 0)]
    queries = [_unit(rng, n, 16) for n in (0, 3, 5, 0, 1, 0)]
    return queries, docs


class TestMaxSimMatrix:
    def test_matches_brute_force(self, case):
        queries, docs = case
        got = maxsim_matrix(MultiVectorCorpus.from_arrays(queries), MultiVectorCorpus.from_arrays(docs))
        want = _brute(queries, docs)
        finite = np.isfinite(want)
        np.testing.assert_allclose(got[finite], want[finite], rtol=1e-5, atol=1e-5)
        assert np.all(np.isneginf(got[:, [0, 3, 6]]))  # empty documents can never win

    def test_empty_queries_score_zero_on_real_documents(self, case):
        queries, docs = case
        got = maxsim_matrix(MultiVectorCorpus.from_arrays(queries), MultiVectorCorpus.from_arrays(docs))
        assert np.all(got[[0, 3, 5]][:, [1, 2, 4, 5]] == 0.0)

    @pytest.mark.parametrize("budget", [64, 700, 5_000, 10**9])
    def test_block_budget_never_changes_the_answer(self, case, budget):
        queries, docs = case
        q, d = MultiVectorCorpus.from_arrays(queries), MultiVectorCorpus.from_arrays(docs)
        np.testing.assert_allclose(
            maxsim_matrix(q, d, max_block_bytes=budget), maxsim_matrix(q, d), rtol=1e-6, atol=1e-6
        )

    def test_float16_documents_are_scored_in_float32(self, rng):
        docs = [_unit(rng, n, 8) for n in (3, 5)]
        queries = [_unit(rng, 4, 8)]
        half = MultiVectorCorpus.from_arrays([d.astype(np.float16) for d in docs])
        got = maxsim_matrix(MultiVectorCorpus.from_arrays(queries), half)
        want = _brute(queries, [d.astype(np.float16).astype(np.float32) for d in docs])
        np.testing.assert_allclose(got, want, rtol=1e-5)

    def test_query_transform_is_applied(self, rng):
        docs = [_unit(rng, 3, 8)]
        queries = [_unit(rng, 2, 8)]
        got = maxsim_matrix(
            MultiVectorCorpus.from_arrays(queries), MultiVectorCorpus.from_arrays(docs),
            query_transform=lambda q: -q,
        )
        np.testing.assert_allclose(got, _brute([-queries[0]], docs), rtol=1e-5)

    def test_agrees_with_the_legacy_numpy_index(self, rng, tmp_path):
        """Retrieval equivalence: the new scorer and the original index must agree."""
        docs = [_unit(rng, n, 32) for n in (5, 12, 1, 30, 7)]
        idx = NumpyIndex(tmp_path, dim=32, method="none")
        idx.add([
            CompressedPage(ref=PageRef(f"d{i}", 1), codes=d.view(np.uint8).reshape(d.shape[0], -1),
                           dim=32, n_tokens_before=d.shape[0], n_tokens_after=d.shape[0])
            for i, d in enumerate(docs)
        ])
        queries = [_unit(rng, n, 32) for n in (3, 20, 8)]
        got = maxsim_matrix(MultiVectorCorpus.from_arrays(queries), MultiVectorCorpus.from_arrays(docs))
        for qi, q in enumerate(queries):
            np.testing.assert_allclose(got[qi], idx.score_all(q), rtol=1e-5)


class TestSegmentReduce:
    def test_trailing_empty_segment_does_not_truncate_the_previous_one(self):
        values = np.array([[1.0, 2.0, 9.0]])
        out = segment_reduce(values, np.array([0, 3, 3]), np.maximum, axis=1, fill=-np.inf)
        assert out.tolist() == [[9.0, -np.inf]]

    def test_leading_and_middle_empties(self):
        values = np.arange(5, dtype=float)[:, None]
        out = segment_reduce(values, np.array([0, 0, 2, 2, 5]), np.add, axis=0, fill=0.0)
        assert out[:, 0].tolist() == [0.0, 1.0, 0.0, 9.0]


class TestRank:
    def test_ties_break_by_index(self):
        assert rank(np.array([1.0, 3.0, 3.0, 2.0])).tolist() == [1, 2, 3, 0]

    def test_top_k_rows(self):
        s = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.1]])
        assert rank(s, k=2).tolist() == [[1, 2], [0, 1]]
