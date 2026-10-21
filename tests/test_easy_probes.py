"""Easy mode, Tier A practice questions (``optivision.easy._probes``; FINAL_DESIGN 5.3.1).

The pages are synthetic PageRecords (no rendering), so word statistics are known by
construction; see ``tests/easy_docs_fixtures.py``.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from optivision.easy import _docs, _probes
from optivision.easy._probes import (
    FAMILIES,
    HARD_CAP,
    HARD_MIN,
    HardSet,
    Probe,
    encode_probes,
    hard_set,
    hardness,
    is_content,
    make_probes,
    pool_size,
    tokens,
)

from .easy_docs_fixtures import HEADER, page_records, synthetic_texts, word_adapter

N_PAGES = 120


@pytest.fixture(scope="module")
def pages():
    return page_records(synthetic_texts(N_PAGES, seed=0), duplicates={10: 9}, blanks=(20,), images=(30,))


@pytest.fixture(scope="module")
def prof(pages):
    return _docs.profile(pages)


@pytest.fixture(scope="module")
def probes(pages, prof):
    return make_probes(pages, profile=prof)


def _content_words(text: str) -> set[str]:
    return {w.casefold() for w in tokens(text) if is_content(w.casefold())}


# ---------------------------------------------------------------------------- generation


def test_generation_is_deterministic(pages, prof, probes):
    assert make_probes(pages, profile=prof) == probes
    assert make_probes(pages) == probes  # the profile is computed when not given
    other = make_probes(pages, seed=1, profile=prof)
    assert other != probes
    assert [p.id for p in probes] == [f"P{i:05d}" for i in range(len(probes))]


def test_three_per_page_from_the_five_families(probes):
    per_page = Counter(p.page_id for p in probes)
    assert max(per_page.values()) == 3
    assert set(Counter(p.family for p in probes)) == set(FAMILIES)
    # repeated, blank and image-only pages give no probes
    assert not {"docs/report.pdf::p11", "docs/report.pdf::p21", "docs/report.pdf::p31"} & set(per_page)
    assert len(per_page) == N_PAGES - 3


def test_family_shapes(probes):
    for p in probes:
        n = len(tokens(p.text))
        if p.family == "span":
            assert 8 <= n <= 25
        elif p.family == "anchor_dropped":
            assert 6 <= n <= 23
        elif p.family == "keywords":
            assert 2 <= n <= 5
        elif p.family == "bigram":
            assert n == 2 and 2 <= len(p.relevant) <= 15
        elif p.family == "caption":
            assert 3 <= n <= 25


def test_caption_probes_come_from_caption_lines(pages, probes):
    by_id = {p.id: p for p in pages}
    for p in probes:
        if p.family == "caption":
            line = next(ln for ln in by_id[p.page_id].text.split("\n") if ln.startswith("Figure"))
            assert line.split(": ", 1)[1] == p.text


def test_anchor_dropped_removes_the_two_rarest_words_of_its_span(pages, prof, probes):
    corpus = _probes._Corpus([p for p in pages if p.id not in prof.aliases and not p.blank and p.has_text],
                             frozenset(prof.boilerplate))
    for p in probes:
        if p.family != "anchor_dropped":
            continue
        page = next(x for x in pages if x.id == p.page_id)
        spans = _probes._Maker(corpus).ranked_spans(corpus.content_text(page))
        span = spans[1] if len(spans) > 1 else spans[0]
        content = sorted({w.casefold() for w in span if is_content(w.casefold())},
                         key=lambda w: (-float(corpus.idf[corpus.vocab[w]]), w))
        assert [w for w in span if w.casefold() not in set(content[:2])] == p.text.split(" ")


def test_boilerplate_number_only_and_duplicate_probes_are_dropped(probes):
    header_words = {w.casefold() for w in tokens(HEADER)}
    for p in probes:
        words = {w.casefold() for w in tokens(p.text)}
        assert not words & header_words, p
        assert "page" not in words
        assert _content_words(p.text), p  # never numbers or function words only
    keys = [" ".join(w.casefold() for w in tokens(p.text)) for p in probes]
    assert len(keys) == len(set(keys))


def test_relevant_sets_follow_the_eighty_percent_rule_exactly(pages, prof, probes):
    """Brute force over every unique text page agrees with the postings shortcut."""
    canonical = [p for p in pages if p.id not in prof.aliases and not p.blank and p.has_text]
    page_words = {}
    for page in canonical:
        text = "\n".join(ln for ln in page.text.split("\n") if _docs.line_key(ln) not in prof.boilerplate)
        page_words[page.id] = _content_words(text)
    copies = {}
    for alias, first in prof.aliases.items():
        copies.setdefault(first, set()).add(alias)
    for p in probes:
        words = _content_words(p.text)
        need = math.ceil(0.8 * len(words) - 1e-9)
        expected = {pid for pid, have in page_words.items() if len(words & have) >= need}
        assert p.page_id in expected
        assert len(expected) <= 15
        for pid in list(expected):
            expected |= copies.get(pid, set())
        assert p.relevant == frozenset(expected), p


def test_aliases_join_relevant_sets(probes):
    from_p10 = [p for p in probes if p.page_id == "docs/report.pdf::p10"]
    assert from_p10 and all("docs/report.pdf::p11" in p.relevant for p in from_p10)


def test_probes_relevant_to_more_than_fifteen_pages_are_dropped():
    # every page: sixteen shared words plus one word of its own, on one line that is not
    # boilerplate (the own word differs), so every span holds 16 of 17 words of 40 pages
    own = [f"own{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(40)]
    shared = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa"
    pages = page_records([f"{shared} {own[i]}" for i in range(40)])
    corpus = _probes._Corpus(pages, frozenset())
    assert corpus.relevant(corpus.ids_of(tokens(shared + " " + own[0]))) is None  # 40 pages qualify
    probes = make_probes(pages, max_pages=40)
    assert probes  # keywords with the page's own word still make probes
    assert all(len(p.relevant) <= 15 for p in probes)
    assert not [p for p in probes if p.family in ("span", "anchor_dropped")]


def test_max_pages_and_per_page(pages, prof):
    few = make_probes(pages, profile=prof, max_pages=10, per_page=2)
    assert len({p.page_id for p in few}) <= 10
    assert max(Counter(p.page_id for p in few).values()) <= 2
    with pytest.raises(ValueError):
        make_probes(pages, per_page=0)


def test_no_text_pages_gives_no_probes():
    assert make_probes(page_records(["", ""], images=(0, 1))) == []


# ---------------------------------------------------------------------------- encoding


def test_probes_are_encoded_as_queries(probes):
    adapter = word_adapter(documents_allowed=False)
    out = encode_probes(adapter, probes[:7])
    assert out.ids == [p.id for p in probes[:7]]
    assert len(out) == 7

    class Encoder:  # the easy-mode Encoder takes texts only
        def encode_queries(self, texts):
            return adapter.encode_queries(texts)

    again = encode_probes(Encoder(), probes[:7])
    assert again.ids == out.ids and np.array_equal(again.vectors, out.vectors)


# ---------------------------------------------------------------------------- hard set


def _probe(i: int, page: int | None = None, relevant: tuple[str, ...] | None = None) -> Probe:
    pid = f"d/x.pdf::p{page if page is not None else i}"
    return Probe(f"P{i:05d}", f"probe text {i}", "span", pid, frozenset(relevant or (pid,)))


def _ranking(relevant_rank: int, s_rel: float, s_out: float, rel: str, depth: int = 20) -> list[tuple[str, float]]:
    """A ranking with ``rel`` at ``relevant_rank`` (1-based) scoring ``s_rel`` and other pages
    scoring ``s_out`` and below."""
    others = [(f"other{j}", s_out - 0.001 * j) for j in range(depth)]
    out = others[:relevant_rank - 1] + [(rel, s_rel)] + others[relevant_rank - 1:depth - 1]
    return out


def test_keep_rule_is_top_ten_without_a_top_one_filter():
    probes = [_probe(0), _probe(1), _probe(2)]
    ref = {
        "P00000": _ranking(1, 10.0, 8.0, probes[0].page_id),
        "P00001": _ranking(10, 7.0, 9.0, probes[1].page_id),  # rank 10: kept, and hard
        "P00002": _ranking(11, 7.0, 9.0, probes[2].page_id),  # rank 11: dropped
    }
    hs = hard_set(probes, ref)
    assert hs.n_candidates == 3 and hs.n_kept == 2
    assert hs.ids == ["P00001", "P00000"]  # hardest first
    assert hs.hardness["P00001"] == pytest.approx((7.0 - 9.0) / 7.0)
    assert hs.hardness["P00000"] == pytest.approx((10.0 - 8.0) / 10.0)
    assert not hs.checked


def test_hardness_formula_and_edges():
    rel = {"a"}.__contains__
    assert hardness(["a", "b"], [2.0, 1.0], rel) == pytest.approx(0.5)
    assert hardness(["b", "a"], [3.0, 2.0], rel) == pytest.approx(-0.5)
    assert hardness(["a"], [2.0], rel) == math.inf  # no other page listed
    assert hardness(["b"], [2.0], rel) == math.inf  # no relevant page listed
    assert hardness(["a", "b"], [-2.0, -3.0], rel) == pytest.approx(0.5)


def test_one_hardest_probe_per_page_and_examples():
    probes = [_probe(0, page=1), _probe(1, page=1), _probe(2, page=2)]
    ref = {p.id: _ranking(1, 10.0, s, p.page_id) for p, s in zip(probes, (5.0, 9.0, 6.0))}
    hs = hard_set(probes, ref)
    assert hs.ids == ["P00001", "P00002"]  # P00001 is harder than P00000 on page 1
    assert [p.id for p in hs.examples] == ["P00000"]
    assert hs.clusters == {"P00001": "d/x.pdf::p1", "P00002": "d/x.pdf::p2"}


@pytest.mark.parametrize("m, size", [(40, 40), (90, 60), (180, 60), (240, 80), (600, 200), (1200, 300)])
def test_hardest_third_or_sixty_capped_at_three_hundred(m, size):
    probes = [_probe(i) for i in range(m)]
    ref = {p.id: _ranking(1, 10.0, 10.0 - 5.0 * (i + 1) / m, p.page_id) for i, p in enumerate(probes)}
    hs = hard_set(probes, ref)
    assert hs.n == size == min(m, HARD_CAP, max(math.ceil(m / 3), HARD_MIN))
    assert hs.checked == (size >= HARD_MIN)
    assert hs.ids == [f"P{i:05d}" for i in range(size)]  # the smallest gaps are the hardest
    g = [hs.hardness[pid] for pid in hs.ids]
    assert g == sorted(g)
    assert len(hs.examples) == m - size


def test_aliases_in_rankings_count_as_relevant():
    p = _probe(0, relevant=("d/x.pdf::p0", "d/x.pdf::p9"))
    ranking = [("d/x.pdf::p9", 5.0)] + [(f"o{j}", 4.0 - j) for j in range(12)]
    assert hard_set([p], {p.id: ranking}).n == 1
    q = _probe(1)
    ranking = [("d/x.pdf::p7", 5.0)] + [(f"o{j}", 4.0 - j) for j in range(12)]
    assert hard_set([q], {q.id: ranking}).n == 0
    assert hard_set([q], {q.id: ranking}, aliases={"d/x.pdf::p7": "d/x.pdf::p1"}).n == 1


def test_hard_set_is_frozen_and_hashed():
    probes = [_probe(i) for i in range(70)]
    ref = {p.id: _ranking(1, 10.0, 9.0 - 0.01 * i, p.page_id) for i, p in enumerate(probes)}
    hs = hard_set(probes, ref)
    assert hs.checked and len(hs.sha256) == 64
    assert hard_set(probes, ref).sha256 == hs.sha256
    hs.verify()
    with pytest.raises(AttributeError):  # dataclasses.FrozenInstanceError
        hs.probes = ()
    again = HardSet.from_dict(hs.as_dict())
    assert again.sha256 == hs.sha256 and again.ids == hs.ids and again.relevant == hs.relevant
    assert again.ref_ids() == hs.ref_ids()
    tampered = hs.as_dict()
    tampered["probes"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="changed after it was frozen"):
        HardSet.from_dict(tampered)
    other_ref = dict(ref)
    other_ref["P00000"] = list(reversed(ref["P00000"][:12])) + ref["P00000"][12:]
    assert hard_set(probes, other_ref).sha256 != hs.sha256


def test_strata_count_families_and_documents():
    probes = [Probe(f"P{i:05d}", f"t {i}", "span" if i % 2 else "keywords", f"d/{i % 3}.pdf::p{i}",
                    frozenset([f"d/{i % 3}.pdf::p{i}"])) for i in range(6)]
    hs = hard_set(probes, {p.id: _ranking(1, 2.0, 1.0, p.page_id) for p in probes})
    assert hs.strata() == {"family": {"span": 3, "keywords": 3}, "document": {"d/0.pdf": 2, "d/1.pdf": 2,
                                                                           "d/2.pdf": 2}}


# ---------------------------------------------------------------------------- cost control


def test_pool_size_shrinks_only_past_a_quarter_of_encoding_time():
    # 600 pages x 3 probes x 2 rankings x 0.01 s = 36 s; budget 25% of 200 s = 50 s
    plan = pool_size(0.01, 200.0)
    assert (plan.pages, plan.shrunk) == (600, False) and plan.line() is None
    plan = pool_size(0.01, 100.0)  # budget 25 s -> 416 pages
    assert (plan.pages, plan.shrunk) == (416, True)
    assert plan.projected_s <= plan.budget_s
    assert "416 pages" in plan.line().render() and plan.line().render().endswith("[counted]")
    assert pool_size(1.0, 100.0).pages == 200  # never below 200 pages
    assert pool_size(1.0, 100.0, n_text_pages=150) == _probes.PoolPlan(150, False, 900.0, 25.0)
    assert pool_size(0.01, 100.0, n_text_pages=300).pages == 300  # 18 s fits in 25 s
