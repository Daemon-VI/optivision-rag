"""The small-memory layout (``optivision.easy._tiered``, FINAL_DESIGN 5.7 step 6).

The identity half of V9 at test size: the memory-mapped two-tier search over several segments returns
the rankings (ids, order and scores) of the campaign's in-RAM path
``TieredIndex(ExactIndex(hot), ExactIndex(int8 per vector), C)`` over one concatenated store. The memory
half runs at 20,000 pages in ``scripts/easy_validation/v9_small_memory.py`` (``reports/easy/v9/``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.easy import _ledger
from optivision.easy import _tiered as T
from optivision.easy._types import FeatureOff
from optivision.representation import MultiVectorCorpus
from optivision.storage import ExactIndex, TieredIndex
from optivision.testing.synthetic import make_corpus, make_queries

pytest.importorskip("scipy")  # hm(0.25) merges with SciPy's Ward linkage

N, DIM = 330, 48
SIZES = (128, 128, 74)


@pytest.fixture(scope="module")
def data() -> tuple[MultiVectorCorpus, MultiVectorCorpus]:
    corpus = make_corpus(N, DIM, tokens=(12, 40), seed=21, anisotropy=0.8)
    queries, _ = make_queries(corpus, 36, tokens=(4, 10), noise=0.6, seed=22)
    return corpus, queries


def _segments(corpus: MultiVectorCorpus, hot: str, tmp, mmap_cold: bool = True):
    hot_dirs, cold_dirs, lo = [], [], 0
    for s, n in enumerate(SIZES):
        h, c = T.encode_tiers(corpus.select(range(lo, lo + n)), hot)
        hot_dirs.append(T.write_segment(tmp / "hot" / f"{s:03d}", h))
        cold_dirs.append(T.write_segment(tmp / "cold" / f"{s:03d}", c))
        lo += n
    hot_segs = [T.open_segment(d, mmap=False) for d in hot_dirs]
    cold_segs = [T.open_segment(d, mmap=mmap_cold) for d in cold_dirs]
    return hot_segs, cold_segs


@pytest.mark.parametrize("hot", list(T.HOT_SPECS))
@pytest.mark.parametrize("k", [1, 10, 50])
def test_memory_mapped_two_tier_equals_the_campaign_tieredindex(data, tmp_path, hot, k):
    corpus, queries = data
    hot_segs, cold_segs = _segments(corpus, hot, tmp_path)
    searcher = T.TieredSearcher(hot_segs, cold_segs)
    assert searcher.candidates == 50 and searcher.n_pages == N
    got = searcher.ranking(queries, k)

    hot_all, cold_all = T.encode_tiers(corpus, hot)  # one store, in RAM, as the campaign scored it
    top, scores = TieredIndex(ExactIndex(hot_all), ExactIndex(cold_all), searcher.candidates).search(queries, k)
    assert np.array_equal(got.indices, top)
    assert np.array_equal(got.scores, scores)  # the same rescoring arithmetic, bit for bit
    assert got.ids == [[corpus.ids[i] for i in row] for row in top]
    assert searcher.search(queries, k)[0][0] == (corpus.ids[int(top[0, 0])], float(scores[0, 0]))


def test_the_cold_copy_really_stays_on_disk(data, tmp_path):
    corpus, _ = data
    _, cold = _segments(corpus, "binary", tmp_path)
    for seg in cold:
        assert not seg.codes.flags.owndata  # a view of the memory map, not a copy
        assert isinstance(seg.codes.base, np.memmap) or isinstance(seg.codes, np.memmap)


def test_a_segment_round_trips_bit_for_bit(data, tmp_path):
    corpus, _ = data
    hot, cold = T.encode_tiers(corpus.select(range(40)), "hm(0.25) > binary")
    for name, cc in (("h", hot), ("c", cold)):
        back = T.open_segment(T.write_segment(tmp_path / name, cc), mmap=True)
        assert np.array_equal(back.codes, cc.codes) and np.array_equal(back.offsets, cc.offsets)
        assert back.ids == cc.ids and back.dim == cc.dim
        assert np.array_equal(back.decode_rows(0, back.num_vectors), cc.decode_rows(0, cc.num_vectors))
        header = json.loads((tmp_path / name / "tier.json").read_text(encoding="utf-8"))
        assert header["format"] == "optivision-easy-tier" and not (tmp_path / name / "tier.json.tmp").exists()


def test_a_segment_chain_decodes_like_one_store(data):
    corpus, _ = data
    cold = Pipeline.from_dict(T.COLD_SPEC)
    whole = cold.compress(corpus)
    chain = T.SegmentChain([cold.compress(corpus.select(range(a, a + n)))
                            for a, n in zip(np.cumsum((0,) + SIZES[:-1]), SIZES, strict=True)])
    assert len(chain) == len(whole) and chain.ids == whole.ids and chain.num_vectors == whole.num_vectors
    assert np.array_equal(chain.offsets, whole.offsets) and chain.nbytes == whole.nbytes
    rng = np.random.default_rng(0)
    cuts = [(0, whole.num_vectors), (0, 1), (whole.num_vectors - 1, whole.num_vectors), (5, 5)]
    seg_end = int(chain._row_starts[1])
    cuts += [(seg_end - 3, seg_end + 3), (seg_end, seg_end + 1)]
    cuts += [tuple(sorted(rng.integers(0, whole.num_vectors, 2))) for _ in range(20)]
    for lo, hi in cuts:
        assert np.array_equal(chain.decode_rows(lo, hi), whole.decode_rows(lo, hi)), (lo, hi)


def test_candidates_follow_the_rule():
    assert T.candidates_for(0) == 50 and T.candidates_for(500) == 50 and T.candidates_for(2500) == 50
    assert T.candidates_for(2501) == 51 and T.candidates_for(20000) == 400
    with pytest.raises(ValueError):
        T.candidates_for(-1)


def test_the_memory_bound_is_hot_plus_one_shortlist_plus_the_allowance(data, tmp_path):
    corpus, _ = data
    hot, cold = _segments(corpus, "binary", tmp_path)
    s = T.TieredSearcher(hot, cold, candidates=60)
    b = s.memory_bound()
    assert b.hot_bytes == sum(h.nbytes for h in hot) == s.hot_bytes
    mean_vectors = corpus.num_vectors / N
    assert b.shortlist_decode_bytes == int(np.ceil(60 * mean_vectors * DIM * 4))
    assert b.total == b.hot_bytes + b.shortlist_decode_bytes + 128 * 10**6
    assert s.cold_bytes == sum(c.nbytes for c in cold)


def test_k_beyond_the_shortlist_is_refused(data, tmp_path):
    corpus, queries = data
    hot, cold = _segments(corpus, "binary", tmp_path)
    with pytest.raises(ValueError, match="shortlist"):
        T.TieredSearcher(hot, cold).ranking(queries, 51)


def test_mismatched_or_unsuitable_copies_are_refused(data, tmp_path):
    corpus, _ = data
    hot, cold = T.encode_tiers(corpus, "binary")
    other_hot, _ = T.encode_tiers(corpus.select(range(N - 1)), "binary")
    with pytest.raises(T.SmallMemoryRefused, match="same pages"):
        T.TieredSearcher(other_hot, cold)
    with pytest.raises(T.SmallMemoryRefused, match="int8"):
        T.TieredSearcher(hot, hot)
    with pytest.raises(T.SmallMemoryRefused, match="1-bit"):
        T.TieredSearcher(cold, cold)
    with pytest.raises(T.SmallMemoryRefused):
        T.hot_pipeline("hm(0.25) > int8")
    centred = Pipeline.from_dict({"stages": [{"stage": "binary", "center": "mean"}]}).fit(corpus).compress(corpus)
    with pytest.raises(T.SmallMemoryRefused, match="fitted state"):
        T.write_segment(tmp_path / "x", centred)


def test_the_layout_is_switched_off_and_refuses_text(monkeypatch, tmp_path):
    assert not _ledger.is_on("small_memory_layout")
    with pytest.raises(FeatureOff, match="small-memory layout is switched off"):
        T.require("image")
    with pytest.raises(FeatureOff):
        T.open_searcher([tmp_path], [tmp_path])
    monkeypatch.setattr(_ledger, "require", lambda name: None)  # this test only; the ledger is unchanged
    T.require("image")
    with pytest.raises(T.SmallMemoryRefused, match="page models only"):
        T.require("text")


def test_open_searcher_loads_hot_and_maps_cold_once_switched_on(data, tmp_path, monkeypatch):
    corpus, queries = data
    monkeypatch.setattr(_ledger, "require", lambda name: None)
    lo, hot_dirs, cold_dirs = 0, [], []
    for s, n in enumerate(SIZES):
        h, c = T.encode_tiers(corpus.select(range(lo, lo + n)), "binary")
        hot_dirs.append(T.write_segment(tmp_path / "h" / str(s), h))
        cold_dirs.append(T.write_segment(tmp_path / "c" / str(s), c))
        lo += n
    s = T.open_searcher(hot_dirs, cold_dirs)
    assert all(seg.codes.flags.owndata or not isinstance(seg.codes.base, np.memmap) for seg in s.hot_store.segments)
    assert s.ranking(queries, 5).ids[0]
