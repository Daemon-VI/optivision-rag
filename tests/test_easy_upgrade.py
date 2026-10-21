"""index.upgrade(): the unchanged certify() on real questions (optivision.easy._upgrade;
FINAL_DESIGN 5.7 step 5, 14).

Runs on a fake index (tests/easy_check_fakes.py) over a synthetic corpus; one test runs
certify() for real. Synthetic data exercise code paths only.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import optivision.api
from optivision.compose import Pipeline
from optivision.core.evaluate.reference import LabelReference, RankingReference
from optivision.core.selection.statistics import betting_pvalues, to_y
from optivision.easy import _check, _claim, _querylog, _upgrade
from optivision.easy._types import FeatureOff

from . import easy_check_fakes as F
from .test_easy_check import _say


@pytest.fixture
def fakes(monkeypatch):
    F.install_fakes(monkeypatch)
    monkeypatch.setattr(_check, "gpu_available", lambda: True)
    monkeypatch.setattr(_check, "page_images", lambda index, pairs: [None] * len(pairs))


@pytest.fixture
def qs():
    return F.question_texts(F.synthetic(), 220, seed=5)


class Recorder:
    """A stand-in for certify() that records its call and certifies nothing."""

    def __init__(self, select: int | None = None, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.select = select
        self.fail = fail

    def __call__(self, vectors, queries, qrels=None, **kw):
        self.calls.append({"vectors": vectors, "queries": queries, "qrels": qrels, **kw})
        if self.fail:
            raise RuntimeError("certify crashed")
        cert = types.SimpleNamespace(digest="d" * 64, save=lambda path: None)
        return types.SimpleNamespace(selected=None, certified=False, certificate=cert, evidence=None,
                                     pipeline=None, compressed=None)


# ============================================================================ the arithmetic


def test_first_crossing_matches_betting_pvalues():
    rng = np.random.default_rng(1)
    T, level = 0.97, 0.025
    m0 = T / (1 + T)
    c = np.where(rng.random((12, 600)) < 0.02, 1 / np.log2(3), 1.0)
    y = to_y(c, np.ones_like(c), T)
    first = _upgrade.first_crossing(y, m0, level)
    assert (first > 0).all()
    for row, n in zip(y, first):
        assert betting_pvalues(row[:n], m0) <= level
        assert betting_pvalues(row[:n - 1], m0) > level


def test_queries_needed_are_recomputed_from_the_betting_test():
    # FINAL_DESIGN 5.7 step 5: at 0.97, 150 if no top result changes, 175 on attempt 2 (one family)
    assert _upgrade.queries_needed(0.97, _upgrade.attempt_error(1)) == 150
    assert _upgrade.queries_needed(0.97, _upgrade.attempt_error(2)) == 175
    # docs/package/guide/SELECTION_AND_CERTIFICATION.md 7.1: no candidate that never scores above
    # the baseline can pass with fewer than 547, 180, 107 at level 0.05 / 7 (the bet at its cap on
    # every question); the bet as implemented ramps up, so it needs at least that many
    from optivision.core.selection.statistics import BET_CAP

    for t, floor in ((0.99, 547), (0.97, 180), (0.95, 107)):
        step = np.log1p(BET_CAP * (1 - t) / t)
        assert int(np.ceil(np.log(7 / 0.05) / step)) == floor
        assert _upgrade.queries_needed(t, 0.05 / 7) >= floor
    # "median 248 at mean agreement 0.99": a seeded simulation, so a range
    median = _upgrade.queries_median(0.97, _upgrade.attempt_error(1))
    assert 220 <= median <= 270
    assert _upgrade.queries_median(0.97, _upgrade.attempt_error(1)) == median      # seeded


def test_error_spending_keeps_the_total_at_or_under_five_percent():
    errors = [_upgrade.attempt_error(k) for k in range(1, 60)]
    assert errors[0] == 0.025 and errors[1] == 0.0125
    assert sum(errors) <= _upgrade.TOTAL_ERROR
    with pytest.raises(ValueError):
        _upgrade.attempt_error(0)


def test_candidate_families_are_larger_allowed_rungs_lightest_first():
    rows = F.fake_select("fake/model", 0.97, 40).rows
    fam = _upgrade.candidate_families(rows, F.fake_row("float16"))
    assert {k: [r.name for r in v] for k, v in fam.items()} == {
        "8-bit": ["int8(per_vector)"], "centred 4-bit": ["int4(mean)"]}         # 1-bit shut: unchecked
    rows = F.fake_select("fake/model", 0.97, 40, checked=True).rows
    fam = _upgrade.candidate_families(rows, F.fake_row("int8(per_vector)"))
    assert list(fam) == ["centred 4-bit", "1-bit"]                                # below target stays in
    assert _upgrade.candidate_families(rows, F.fake_row("binary")) == {}


# ============================================================================ the certify call


def test_upgrade_calls_the_unchanged_certify_as_designed(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path, texts=vectors, layout="float16")
    report = _upgrade.run_upgrade(index, texts, out=None)
    (call,) = rec.calls
    assert call["vectors"] is index._store.float16_corpus()
    assert call["qrels"] is None
    assert call["target"] == 0.97 and call["error"] == 0.025 and call["order_seed"] == 0
    assert call["metric"] == "ndcg@5"
    assert isinstance(call["reference"], RankingReference) and call["reference"].depth == 1
    assert call["fallback"] == index.layout.spec
    space = call["space"]
    assert space.families == ("8-bit", "centred 4-bit")
    assert [s.pipeline["stages"][0]["stage"] for s in space] == ["int8", "int4"]
    assert [s.to_dict()["pipeline"] for s in space] == [
        Pipeline.from_dict(F.fake_row(n).spec).to_dict() for n in ("int8(per_vector)", "int4(mean)")]
    assert len(call["queries"]) == 220
    assert report.outcome == "kept" and report.attempt == 1 and report.error == 0.025
    # two families: level 0.025 / 2, so the least number is 175 (as attempt 2 with one family)
    assert report.needed == 175


def test_attempts_spend_halving_error_levels(tmp_path, fakes, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    corpus = F.synthetic()
    t1, v1, _ = F.question_texts(corpus, 200, seed=5, prefix="first batch")
    t2, v2, _ = F.question_texts(corpus, 200, seed=6, prefix="second batch")
    index = F.FakeIndex(tmp_path, texts={**v1, **v2})
    r1 = _upgrade.run_upgrade(index, t1, out=None)
    r2 = _upgrade.run_upgrade(index, t2, out=None)
    assert [c["error"] for c in rec.calls] == [0.025, 0.0125]
    assert (r1.attempt, r2.attempt) == (1, 2)
    book = index._store.manifest["upgrade"]
    assert book["attempts"] == 2
    assert [h["attempt"] for h in book["history"]] == [1, 2]
    assert [h["outcome"] for h in book["history"]] == ["kept", "kept"]
    assert sum(h["error"] for h in book["history"]) <= 0.05
    assert "attempt 2 on this index, error level 1.25%" in str(r2)


def test_the_attempt_is_recorded_before_certify_runs(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    monkeypatch.setattr(optivision.api, "certify", Recorder(fail=True))
    index = F.FakeIndex(tmp_path, texts=vectors)
    with pytest.raises(RuntimeError, match="certify crashed"):
        _upgrade.run_upgrade(index, texts, out=None)
    book = index._store.manifest["upgrade"]
    assert book["attempts"] == 1 and book["history"][-1]["outcome"] == "started"
    assert {_querylog.query_hash(t) for t in texts} <= set(book["used_sha256"])
    assert {_querylog.vector_hash(vectors[t]) for t in texts} <= set(book["used_sha256"])


def test_a_read_only_index_refuses_before_spending(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path, texts=vectors)
    index._store.writable = False
    report = _upgrade.run_upgrade(index, texts, out=None)
    assert report.outcome == "not run" and report.reason == "cannot record the attempt"
    assert rec.calls == [] and "upgrade" not in index._store.manifest


# ============================================================================ questions


def test_repeats_are_removed_and_used_questions_never_come_back(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path, texts=vectors)
    repeated = texts + [texts[0].upper(), "  " + texts[1] + " "]
    report = _upgrade.run_upgrade(index, repeated, out=None)
    assert report.n_given == 222 and report.n_repeats == 2 and report.n_tested == 220
    assert len(rec.calls[0]["queries"]) == 220
    again = _upgrade.run_upgrade(index, texts, out=None)
    assert again.outcome == "not run" and again.reason == "no new questions"
    assert again.n_reused == 220 and len(rec.calls) == 1
    assert index._store.manifest["upgrade"]["attempts"] == 1                 # nothing spent
    assert "a question is tested once per index; nothing was spent" in str(again)


def test_duplicate_vectors_from_different_texts_are_repeats(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    vectors = dict(vectors, **{"the same question in other words": vectors[texts[0]]})
    index = F.FakeIndex(tmp_path, texts=vectors)
    report = _upgrade.run_upgrade(index, texts + ["the same question in other words"], out=None)
    assert report.n_repeats == 1 and len(rec.calls[0]["queries"]) == 220


def test_vectors_are_accepted_and_hashed(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path)
    arrays = [vectors[t] for t in texts]
    _upgrade.run_upgrade(index, arrays, out=None)
    again = _upgrade.run_upgrade(index, arrays[:200], out=None)
    assert again.reason == "no new questions" and len(rec.calls) == 1


def test_a_question_given_as_text_then_as_vectors_is_reused(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path, texts=vectors)
    _upgrade.run_upgrade(index, texts, out=None)
    again = _upgrade.run_upgrade(index, [vectors[t] for t in texts], out=None)
    assert again.reason == "no new questions" and again.n_reused == 220 and len(rec.calls) == 1


def test_labels_without_any_answer_in_the_index_spend_nothing(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path, texts=vectors)
    report = _upgrade.run_upgrade(index, texts, answers=[None] * len(texts), out=None)
    assert report.reason == "no answered questions" and rec.calls == []
    assert "upgrade" not in index._store.manifest


def test_too_few_questions_spend_nothing(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path, texts=vectors)
    report = _upgrade.run_upgrade(index, texts[:100], out=None)
    assert report.outcome == "not run" and report.reason == "too few questions"
    assert report.needed == 150 and rec.calls == [] and "upgrade" not in index._store.manifest
    text = str(report)
    assert "at least 150 questions if no top result changes; median" in text
    assert "[estimate: the betting test's own arithmetic" in text
    assert "nothing was spent" in text


def test_without_float16_it_explains_the_re_encode_route(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path, texts=vectors, keep_float16=False)
    report = _upgrade.run_upgrade(index, texts, out=None)
    assert report.outcome == "not run" and report.reason == "no float16 copy"
    assert rec.calls == [] and index.loads == 0
    text = str(report)
    assert "rebuilding with keep_float16=True re-encodes every page" in text
    assert text.splitlines()[-1].startswith("Next: ov.build(..., keep_float16=True, rebuild=True)")


def test_no_larger_layout_spends_nothing(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    monkeypatch.setattr(optivision.api, "certify", Recorder())
    index = F.FakeIndex(tmp_path, texts=vectors, layout="int4(mean)")
    report = _upgrade.run_upgrade(index, texts, out=None)                   # 1-bit is shut (unchecked)
    assert report.reason == "no larger layout" and "upgrade" not in index._store.manifest


def test_the_query_log_feeds_upgrade(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path, texts=vectors)
    log = _querylog.open_log(index.path, True, out=None)
    log.record(texts + texts[:30])
    report = _upgrade.run_upgrade(index, None, out=None)                    # reads the log on disk
    assert report.n_tested == 220 and len(rec.calls) == 1
    again = _upgrade.run_upgrade(index, log, out=None)
    assert again.reason == "no new questions"


def test_upgrade_without_questions_or_log_says_how_to_get_them(tmp_path, fakes):
    with pytest.raises(ValueError, match="log_queries=True"):
        _upgrade.run_upgrade(F.FakeIndex(tmp_path), None, out=None)


# ============================================================================ wording


def test_agreement_and_label_wording(tmp_path, fakes, qs, monkeypatch):
    texts, vectors, _ = qs
    rec = Recorder()
    monkeypatch.setattr(optivision.api, "certify", rec)
    index = F.FakeIndex(tmp_path, texts=vectors)
    r = _upgrade.run_upgrade(index, texts[:200], out=None)
    assert r.reference == "agreement" and r.as_dict()["kind"] == "agreement with full precision, not relevance"
    assert "agreement with the full-precision top page (not relevance)" in str(r)

    t2, v2, a2 = F.question_texts(F.synthetic(), 200, seed=9, prefix="labelled")
    index.encoder.texts.update(v2)
    answers = [a2[t] for t in t2]
    answers[0] = None                                                        # an unanswered question
    lab = _upgrade.run_upgrade(index, t2, answers=answers, out=None)
    call = rec.calls[-1]
    assert isinstance(call["reference"], LabelReference) and call["qrels"] is None
    assert lab.reference == "labels" and lab.as_dict()["kind"] == "search quality"
    # some synthetic questions of the second set repeat a vector of the first: they count as used
    assert lab.n_unanswered == 1 and lab.n_tested == 200 - 1 - lab.n_reused - lab.n_repeats
    assert "your answers (search quality)" in str(lab)
    for report in (r, lab):
        _say(str(report))
        for line in report.lines:
            _claim.check_template(line.template)
            if "agreement" in line.text():
                assert "quality" not in line.render()


def test_wording_follows_the_ledger(tmp_path, fakes, qs, monkeypatch):
    from optivision.easy import _ledger

    texts, vectors, _ = qs
    monkeypatch.setattr(optivision.api, "certify", Recorder())
    real = _ledger.setting
    monkeypatch.setattr(_ledger, "setting", lambda name, key, default=None: "rare"
                        if (name, key) == ("upgrade", "wording") else real(name, key, default))
    report = _upgrade.run_upgrade(F.FakeIndex(tmp_path, texts=vectors), texts, out=None)
    assert "upgrades are rare: most attempts keep the current layout" in str(report)


def test_upgrade_switched_off_in_the_ledger(tmp_path, fakes, monkeypatch):
    from optivision.easy import _ledger

    monkeypatch.setattr(_ledger, "is_on", lambda name: False)
    monkeypatch.setattr(_ledger, "require", lambda name: (_ for _ in ()).throw(FeatureOff("upgrade() is off")))
    with pytest.raises(FeatureOff):
        _upgrade.run_upgrade(F.FakeIndex(tmp_path), ["q"], out=None)


# ============================================================================ certify for real


def test_real_certify_selects_a_larger_layout_and_apply_settles_it(tmp_path, fakes, qs):
    texts, vectors, _ = qs
    index = F.FakeIndex(tmp_path, texts=vectors, layout="int8(per_vector)")
    printed: list[str] = []
    report = _upgrade.run_upgrade(index, texts, out=printed.append)
    assert report.outcome == "certified" and report.certified
    assert report.result.certified and report.layout.name == "int4(mean)"
    assert report.measured >= 0.97
    assert index.resettles == [] and index.layout.name == "int8(per_vector)"     # nothing without apply
    text = str(report)
    assert "passed: centred 4-bit (int4(mean))" in text
    assert "is at most 2.5%  [your questions: agreement with full precision, not relevance]" in text
    assert (tmp_path / "docs.ovx" / "upgrade" / "attempt_001.certificate.json").exists()
    _say(text)

    report.apply(out=printed.append)
    (call,) = index.resettles
    assert call["pipeline"] is report.result.pipeline and call["compressed"] is report.result.compressed
    assert call["pipeline"].to_dict() == Pipeline.from_dict(F.fake_row("int4(mean)").spec).to_dict()
    assert call["updates"]["upgraded"]["attempt"] == 1
    assert index.layout.name == "int4(mean)"
    assert "no re-encoding: the codes come from the kept float16 copy, as tested" in printed[-1]
    with pytest.raises(Exception, match="applied already"):
        report.apply(out=None)
    # the index now searches the certified codes
    hits = index._searcher.search(index.encoder.encode_queries(texts[:5]), 3)
    assert len(hits) == 5 and all(len(h) == 3 for h in hits)


def test_real_certify_with_apply_true(tmp_path, fakes, qs):
    texts, vectors, _ = qs
    index = F.FakeIndex(tmp_path, texts=vectors, layout="int8(per_vector)")
    report = _upgrade.run_upgrade(index, texts, apply=True, out=None)
    assert report.applied and index.layout.name == "int4(mean)"
    assert index._store.manifest["upgraded"]["layout"] == "int4(mean)"
    assert str(report).splitlines()[-1].startswith("Next: index.search(...)")
