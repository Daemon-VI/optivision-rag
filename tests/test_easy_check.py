"""index.check(), the spot check, the query log and max_disk= (optivision.easy._check,
_spotcheck, _querylog, _budget; FINAL_DESIGN 4.4, 5.5.3, 5.7).

Runs on a fake index (tests/easy_check_fakes.py) over a synthetic corpus with the real
Pipeline and CompressedRetriever. Synthetic data exercise code paths only.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from optivision.easy import _budget, _check, _claim, _querylog, _spotcheck
from optivision.easy._claim import Line

from . import easy_check_fakes as F
from .test_docs_examples import BANNED_PHRASES, BANNED_WORDS

EXTRA_BANNED = ("lossless", "universal", "guaranteed", "first", "sota", "state of the art", "veto")


@pytest.fixture
def fakes(monkeypatch):
    F.install_fakes(monkeypatch)
    monkeypatch.setattr(_check, "gpu_available", lambda: True)
    monkeypatch.setattr(_check, "page_images", lambda index, pairs: [None] * len(pairs))


@pytest.fixture
def questions():
    texts, vectors, answer = F.question_texts(F.synthetic(), 80, seed=5)
    return texts, vectors, answer


def _say(text: str) -> None:
    """Assert user-facing text is ASCII and free of the banned words and phrases."""
    assert text.isascii()
    low = text.lower()
    for phrase in BANNED_PHRASES:
        assert phrase not in low, phrase
    assert BANNED_WORDS.search(low) is None
    for word in EXTRA_BANNED:
        assert re.search(rf"\b{re.escape(word)}\b", low) is None, word


def _line(report, label: str) -> Line:
    return next(ln for ln in report.lines if ln.label == label)


def _text(report) -> str:
    return str(report)


# ============================================================================ report fields


def test_healthy_index_is_ok_and_the_report_has_its_fields(tmp_path, fakes):
    index = F.FakeIndex(tmp_path)
    printed: list[str] = []
    report = _check.run_check(index, out=printed.append)
    assert report.ok and report.complete
    assert report.record.matches and report.record.model_compared
    assert report.drift.checked and not report.drift.drifted
    assert report.size.qualifies and report.size.status == "pick"
    assert report.practice.ran and report.practice.passed
    assert report.advice is None and not report.refit
    d = report.as_dict()
    for key in ("ok", "complete", "record", "drift", "size", "practice", "growth", "questions", "spot_check",
                "advice", "refit", "block"):
        assert key in d
    assert all(isinstance(ln, Line) for ln in report.lines)
    text = _text(report)
    assert text.splitlines()[-1].startswith("Next: ")
    assert "Result     all clear" in text
    _say(text)
    for p in printed:
        _say(p)


def test_every_number_in_the_report_is_a_claim(tmp_path, fakes, questions):
    texts, vectors, answer = questions
    index = F.FakeIndex(tmp_path, texts=vectors, n_built=10)
    report = _check.run_check(index, texts, answers=[answer[t] for t in texts], out=None)
    for line in report.lines:
        _claim.check_template(line.template)       # a digit outside a claim raises
        assert len({(c.kind, c.source) for c in line.claims}) <= 1


def test_check_changes_nothing_but_the_practice_run_count(tmp_path, fakes, questions):
    texts, vectors, _ = questions
    index = F.FakeIndex(tmp_path, texts=vectors)
    before = dict(index._store.manifest)
    _check.run_check(index, texts, out=None)
    after = dict(index._store.manifest)
    assert after.pop("practice_runs") == 2
    assert after == before
    assert index.rebuilds == [] and index.resettles == []
    assert [c for c in index._store.commits if set(c) != {"practice_runs"}] == []


# ============================================================================ build record


def test_a_changed_library_version_is_named(tmp_path, fakes):
    rec = {"model": "fake/model", "dtype": "float16", "versions": {"numpy": "0.0.1"}}
    index = F.FakeIndex(tmp_path, record=rec)
    index.encoder.record = dict(rec, versions={"numpy": np.__version__})
    report = _check.run_check(index, out=None)
    assert not report.ok
    assert ("numpy", "0.0.1", np.__version__) in report.record.differences
    assert "numpy differs: 0.0.1 at build" in _text(report)
    assert _text(report).splitlines()[-1] == "Next: ov.doctor('fake/model')"


def test_without_the_model_only_versions_are_compared(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(_check, "gpu_available", lambda: False)
    index = F.FakeIndex(tmp_path)
    report = _check.run_check(index, out=None)
    assert index.loads == 0
    assert report.record.compared == ["numpy"] and not report.record.model_compared
    assert ("model settings (commit, dtype, attention, page size, batch) are compared when the model loads"
            in _text(report))


def test_text_questions_let_the_model_settings_be_compared(tmp_path, fakes, monkeypatch, questions):
    monkeypatch.setattr(_check, "gpu_available", lambda: False)
    texts, vectors, _ = questions
    index = F.FakeIndex(tmp_path, texts=vectors)
    index.encoder.record = dict(index.record, dtype="float32")
    report = _check.run_check(index, texts, out=None)
    assert report.record.model_compared
    assert ("dtype", "float16", "float32") in report.record.differences and not report.ok
    vectors_only = F.FakeIndex(tmp_path / "v", texts=vectors)
    _check.run_check(vectors_only, [vectors[t] for t in texts], out=None)
    assert vectors_only.loads == 0                    # questions as vectors never load the model


def test_record_comparison_reads_installed_versions():
    built = {"versions": {"numpy": np.__version__, "python": "0.0", "no-such-package-xyz": "1.0"}}
    rc = _check.compare_record(built, None)
    assert ("python", "0.0", _check.installed_version("python")) in rc.differences
    assert ("no-such-package-xyz", "1.0", "not installed") in rc.differences
    assert not any(d[0] == "numpy" for d in rc.differences)


# ============================================================================ drift


def test_drift_is_detected_with_an_encoder_that_perturbs_vectors(tmp_path, fakes):
    index = F.FakeIndex(tmp_path, perturb=0.6)
    report = _check.run_check(index, out=None)
    assert report.drift.checked and report.drift.drifted
    assert report.drift.max_change > _check.DRIFT_REL_TOL
    assert not report.ok
    assert "the four fixed pages score differently from build" in _text(report)


def test_no_drift_with_the_same_encoder(tmp_path, fakes):
    report = _check.run_check(F.FakeIndex(tmp_path), out=None)
    assert report.drift.checked and not report.drift.drifted
    assert report.drift.max_change < 1e-3


def test_drift_not_checked_without_a_gpu(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(_check, "gpu_available", lambda: False)
    index = F.FakeIndex(tmp_path, perturb=0.6)
    report = _check.run_check(index, out=None)
    assert not report.drift.checked and report.drift.reason == "drift not checked (no GPU)"
    assert "drift not checked (no GPU)" in _text(report)
    assert report.ok and not report.complete
    assert index.loads == 0


def test_drift_true_runs_without_a_gpu(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(_check, "gpu_available", lambda: False)
    report = _check.run_check(F.FakeIndex(tmp_path, perturb=0.6), drift=True, out=None)
    assert report.drift.checked and report.drift.drifted


def test_drift_prints_its_time_before_encoding(tmp_path, fakes):
    index = F.FakeIndex(tmp_path)
    report = _check.run_check(index, out=lambda s: index.log.append(("print", s)))
    kinds = [e[0] for e in index.log]
    assert kinds.index("print") < kinds.index("load") < kinds.index("encode_pages")
    assert "about to re-encode the four fixed pages" in index.log[kinds.index("print")][1]
    assert report.drift.seconds is not None


def test_drift_false_and_vectors_builds(tmp_path, fakes):
    assert _check.run_check(F.FakeIndex(tmp_path), drift=False, out=None).drift.reason == \
        "drift not checked (drift=False)"
    report = _check.run_check(F.FakeIndex(tmp_path / "v", with_files=False), out=None)
    assert report.drift.reason.startswith("drift not checked (built from vectors")
    with pytest.raises(ValueError):
        _check.run_check(F.FakeIndex(tmp_path / "w"), drift="sometimes", out=None)


def test_drift_pages_are_fixed_and_the_record_round_trips(tmp_path, fakes):
    ids = F.page_ids()
    assert _check.pick_drift_pages(ids) == _check.pick_drift_pages(list(reversed(ids)))
    assert len(_check.pick_drift_pages(ids)) == 4
    assert _check.pick_drift_pages(ids[:2]) == sorted(ids[:2])
    index = F.FakeIndex(tmp_path)
    rec = index._store.manifest["drift"]
    pos = {pid: i for i, pid in enumerate(index.corpus16.ids)}
    again = _check.stored_scores(index._store, index.corpus16.select([pos[p] for p in rec["ids"]]))
    assert again == rec["scores"]
    assert rec["files"][0].endswith("report00.pdf") and rec["pages"][0] == 1


# ============================================================================ size and growth


def test_layout_that_no_longer_qualifies_is_advised_down_and_apply_prints_its_time_first(tmp_path, fakes):
    index = F.FakeIndex(tmp_path, n_pages=1500, n_built=1000, keep_float16=False)
    report = _check.run_check(index, out=None)
    assert not report.size.qualifies and report.size.status == "below target"
    assert not report.ok
    assert report.advice is not None and report.advice.name == "float16"
    text = _text(report)
    assert "the layout no longer qualifies at this size (below target)" in text
    assert "Next: index.check().apply()" in text
    assert "[our benchmarks: ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages]" in text
    assert index.rebuilds == []                                      # check never applies itself
    report.apply(out=lambda s: index.log.append(("print", s)))
    kinds = [e[0] for e in index.log]
    assert kinds.index("print") < kinds.index("rebuild")
    assert "re-encodes every page" in index.log[kinds.index("print")][1]
    assert index.rebuilds == [{"layout": "float16", "refit": True}]


def test_apply_with_the_float16_copy_needs_no_re_encoding(tmp_path, fakes):
    index = F.FakeIndex(tmp_path, n_pages=1500, n_built=1000)
    printed: list[str] = []
    _check.run_check(index, out=None).apply(out=printed.append)
    assert "no re-encoding: the layout settles from the kept float16 copy" in printed[0]


def test_apply_without_advice_does_nothing(tmp_path, fakes):
    index = F.FakeIndex(tmp_path)
    printed: list[str] = []
    assert _check.run_check(index, out=None).apply(out=printed.append) is None
    assert index.rebuilds == [] and "nothing to apply" in printed[0]


def test_hand_picked_layout_that_never_qualified_does_not_fail_the_check(tmp_path, fakes):
    index = F.FakeIndex(tmp_path, layout="int4(mean)", practice=None)
    from dataclasses import replace

    index.selection = replace(index.selection, hand_picked=True)
    report = _check.run_check(index, out=None)
    assert report.size.status == "below target" and not report.size.counts_against
    assert report.ok and report.advice is None
    assert "chosen by you" in _text(report)


def test_growth_above_twice_the_build_size_advises_a_refit(tmp_path, fakes):
    index = F.FakeIndex(tmp_path, n_built=10)
    report = _check.run_check(index, out=None)
    assert report.growth == pytest.approx(4.0)
    assert report.refit and report.advice is None
    text = _text(report)
    assert ("In our two tests (500 -> 2,500 pages, distractors from other domains), our layouts lost 0.006 "
            "and 0.009 of retention on average, at worst 0.028") in text
    assert "[our benchmarks: follow-up units f6, f7" in text
    assert "Status" in text and "not yet been checked on new documents" in text
    assert report.ok                                    # growth alone is advice, not a failure
    report.apply(out=None)
    assert index.rebuilds == [{"layout": "int8(per_vector)", "refit": True}]


def test_no_growth_line_below_twice(tmp_path, fakes):
    report = _check.run_check(F.FakeIndex(tmp_path, n_built=25), out=None)
    assert report.growth == pytest.approx(1.6) and not report.refit
    assert "Growth" not in _text(report)


# ============================================================================ practice rerun


def test_practice_rerun_fires_on_collapsed_codes_and_counts_runs(tmp_path, fakes):
    index = F.FakeIndex(tmp_path, code_noise=3.0)
    report = _check.run_check(index, out=None)
    assert report.practice.ran and not report.practice.passed
    assert not report.ok
    assert report.advice is not None and report.advice.name == "float16"
    assert report.practice.runs == 2 and report.practice.runs_saved
    text = _text(report)
    assert "fell below the target on 30 hard questions made from your PDFs  [questions from your PDFs]" in text
    assert "practice check run 2 on this index; every rerun is another look" in text
    again = _check.run_check(index, out=None)
    assert again.practice.runs == 3


def test_practice_rerun_count_not_saved_on_a_read_only_store(tmp_path, fakes):
    index = F.FakeIndex(tmp_path)
    index._store.writable = False
    report = _check.run_check(index, out=None)
    assert report.practice.ran and not report.practice.runs_saved
    assert "count not saved" in _text(report)


def test_practice_warning_mode_does_not_fail_the_check(tmp_path, fakes, monkeypatch):
    from optivision.easy import _ledger

    real = _ledger.setting
    monkeypatch.setattr(_ledger, "setting",
                        lambda name, key, default=None: "warning" if (name, key) == ("practice_check", "mode")
                        else real(name, key, default))
    report = _check.run_check(F.FakeIndex(tmp_path, code_noise=3.0), out=None)
    assert not report.practice.passed and report.ok and report.advice is None
    assert "(a warning only)" in _text(report)


def test_unchecked_index_has_no_practice_rerun(tmp_path, fakes):
    report = _check.run_check(F.FakeIndex(tmp_path, practice=None), out=None)
    assert not report.practice.ran and report.ok and not report.complete
    assert "not rerun: no practice questions were made for this index (unchecked)" in _text(report)


# ============================================================================ your questions


def test_queries_give_agreement_with_the_float16_copy(tmp_path, fakes, questions):
    texts, vectors, _ = questions
    report = _check.run_check(F.FakeIndex(tmp_path, texts=vectors), texts, out=None)
    q = report.questions
    assert q.n == 80 and q.agreement > 0.95
    line = _line(report, "Questions")
    assert line.tag == "[your questions: agreement with full precision, not relevance]"
    assert "quality" not in line.render().lower()
    assert report.as_dict()["questions"]["agreement_kind"] == "agreement with full precision, not relevance"


def test_agreement_falls_on_noisy_codes(tmp_path, fakes, questions):
    texts, vectors, _ = questions
    report = _check.run_check(F.FakeIndex(tmp_path, texts=vectors, code_noise=3.0, practice=None),
                              texts, out=None)
    assert report.questions.agreement < 0.9
    assert report.questions.fell_below and not report.ok
    assert report.advice is not None and report.advice_reason == "your questions fell below the target"


def test_queries_without_float16_need_keep_float16_and_name_the_spot_check(tmp_path, fakes, questions):
    texts, vectors, _ = questions
    report = _check.run_check(F.FakeIndex(tmp_path, texts=vectors, keep_float16=False), texts, out=None)
    assert report.questions.needs_float16 and report.questions.agreement is None
    text = _text(report)
    assert "agreement needs the float16 copy: build with keep_float16=True" in text
    assert "spot_check=True" in text and "it can only flag" in text


def test_answers_in_every_form_give_search_quality(tmp_path, fakes, questions):
    texts, vectors, answer = questions
    index = F.FakeIndex(tmp_path, texts=vectors)
    forms = []
    for i, t in enumerate(texts):
        pid = answer[t]
        doc, _, page = pid.rpartition("::p")
        if i % 4 == 0:
            forms.append(pid)                                            # full page id
        elif i % 4 == 1:
            forms.append(pid.split("/", 1)[1])                           # shortened to its unique end
        elif i % 4 == 2:
            forms.append((index.files[pid][0], int(page)))               # (file, page) with the full path
        else:
            forms.append({pid, (doc, int(page))})                        # a set of either
    report = _check.run_check(index, texts, answers=forms, out=None)
    q = report.questions
    assert q.n_answered == 80 and q.quality > 0.95 and q.retention == pytest.approx(1.0, abs=0.05)
    line = _line(report, "Answers")
    assert line.tag == "[your questions: search quality]"
    assert "nDCG@5" in line.render()


def test_answers_without_float16_give_quality_but_no_retention(tmp_path, fakes, questions):
    texts, vectors, answer = questions
    report = _check.run_check(F.FakeIndex(tmp_path, texts=vectors, keep_float16=False), texts,
                              answers=[answer[t] for t in texts], out=None)
    assert report.questions.quality is not None and report.questions.retention is None
    assert "retention against full precision needs the float16 copy (keep_float16=True)" in _text(report)


def test_answer_errors():
    ids = ["a/x/report.pdf::p1", "b/x/report.pdf::p1", "a/y.pdf::p2"]
    with pytest.raises(ValueError, match="several pages"):
        _check.answer_sets(ids, ["x/report.pdf::p1"], 1)
    with pytest.raises(ValueError, match="not a page of this index"):
        _check.answer_sets(ids, ["zz.pdf::p1"], 1)
    with pytest.raises(ValueError, match="one per query"):
        _check.answer_sets(ids, ["y.pdf::p2"], 2)
    with pytest.raises(TypeError):
        _check.answer_sets(ids, [3.5], 1)
    assert _check.answer_sets(ids, ["y.pdf::p2", None, ("a/x/report.pdf", 1)], 3) == [
        {"a/y.pdf::p2"}, None, {"a/x/report.pdf::p1"}]
    assert _check.answer_sets(["0", "1"], ["1"], 1) == [{"1"}]          # an index built from vectors


def test_answers_need_queries(tmp_path, fakes):
    with pytest.raises(ValueError, match="answers= needs queries="):
        _check.run_check(F.FakeIndex(tmp_path), answers=["x"], out=None)


# ============================================================================ spot check


def test_spot_check_flags_disagreement_and_never_changes_the_layout(tmp_path, fakes, questions):
    texts, vectors, _ = questions
    index = F.FakeIndex(tmp_path, texts=vectors, keep_float16=False, code_noise=3.0, practice=None)
    printed: list[str] = []
    report = _check.run_check(index, texts, spot_check=True, drift=False, out=printed.append)
    spot = report.spot
    assert spot.n == 80 and spot.flagged > 0
    assert spot.bound < 1.0 and spot.below_target
    assert spot.pages_encoded == index.encoder.pages_encoded
    assert not report.ok and report.advice is not None
    assert index.rebuilds == [] and index.resettles == [] and index.layout.name == "int8(per_vector)"
    assert any("about to re-encode" in p and "top pages of each question" in p for p in printed)
    flagged_line = next(ln for ln in spot.lines if ln.template.startswith("flagged"))
    assert flagged_line.tag.startswith("[your questions: spot check")
    _say(str(spot))


def test_spot_check_on_good_codes_can_only_flag(tmp_path, fakes, questions):
    texts, vectors, _ = questions
    index = F.FakeIndex(tmp_path, texts=vectors, keep_float16=False, layout="float16")
    spot = _spotcheck.spot_check(index, texts, out=None)
    assert spot.flagged == 0 and spot.bound == pytest.approx(1.0) and not spot.below_target
    assert "it can only flag: an unflagged question is not shown to agree" in str(spot)


def test_spot_check_bound_is_an_upper_limit_of_agreement(tmp_path, fakes, questions):
    """With the float16 copy the exact agreement is known; the spot check's limit is never below it."""
    texts, vectors, _ = questions
    index = F.FakeIndex(tmp_path, texts=vectors, code_noise=1.5, practice=None)
    exact = _check.run_check(index, texts, drift=False, out=None).questions.agreement
    spot = _spotcheck.spot_check(index, texts, out=None)
    assert spot.bound >= exact - 1e-12
    assert spot.flagged > 0


def test_spot_check_needs_source_files(tmp_path, fakes, questions):
    texts, vectors, _ = questions
    index = F.FakeIndex(tmp_path, texts=vectors, with_files=False)
    spot = _spotcheck.spot_check(index, texts, out=None)
    assert spot.n == 80 and spot.bound is None and index.encoder.pages_encoded == 0
    assert "this index was built from vectors" in str(spot)


def test_spot_check_skips_questions_whose_top_file_is_gone(tmp_path, fakes, questions):
    texts, vectors, _ = questions
    index = F.FakeIndex(tmp_path, texts=vectors)
    for f, _page in list(index.files.values())[:20]:
        Path(f).unlink(missing_ok=True)
    spot = _spotcheck.spot_check(index, texts, out=None)
    assert spot.skipped > 0 and spot.n + spot.skipped == 80


# ============================================================================ query log


def test_query_log_is_off_by_default(tmp_path):
    assert _querylog.open_log(tmp_path / "x.ovx") is None
    assert _querylog.open_log(tmp_path / "x.ovx", False) is None
    assert _querylog.record(None, ["what was revenue?"]) == 0
    assert not (tmp_path / "x.ovx" / "log").exists()
    with pytest.raises(TypeError):
        _querylog.open_log(tmp_path / "x.ovx", "yes")


def test_query_log_prints_its_privacy_notice_once_and_stays_local(tmp_path):
    printed: list[str] = []
    log = _querylog.open_log(tmp_path / "a.ovx", True, out=printed.append)
    again = _querylog.open_log(tmp_path / "a.ovx", True, out=printed.append)
    assert len(printed) == 1 and isinstance(log, _querylog.QueryLog) and again is not None
    notice = printed[0]
    assert "on this machine only. It is never sent anywhere." in notice
    assert str(tmp_path / "a.ovx" / "log") in notice and notice.splitlines()[-1].startswith("Next: ")
    _say(notice.replace(str(tmp_path), ""))
    assert not log.folder.exists()                       # nothing is written before a search
    source = Path(_querylog.__file__).read_text(encoding="utf-8")
    for module in ("socket", "urllib", "http", "requests", "httpx"):
        assert not re.search(rf"^\s*(import|from)\s+{module}\b", source, re.MULTILINE), module


def test_query_log_records_dedupes_excludes_and_deletes(tmp_path):
    log = _querylog.open_log(tmp_path / "b.ovx", True, out=None)
    assert _querylog.record(log, ["What was revenue?", "  what WAS   revenue? ", "Who signed?"]) == 3
    assert _querylog.record(log, [np.zeros((2, 4))]) == 0                 # vectors carry no text
    assert len(log) == 3
    assert log.queries() == ["What was revenue?", "Who signed?"]
    assert log.queries(exclude={_querylog.query_hash("who signed?")}) == ["What was revenue?"]
    with open(log.file, "a", encoding="utf-8") as fh:
        fh.write('{"q": "half a li')                                       # a crash mid-write
    assert len(log) == 3
    assert _querylog.existing_log(tmp_path / "b.ovx") is not None
    assert log.delete() and not log.folder.exists() and not log.delete()
    assert _querylog.existing_log(tmp_path / "b.ovx") is None


def test_query_hash_ignores_case_and_spacing():
    assert _querylog.query_hash("A  b\tC") == _querylog.query_hash("a b c")
    assert _querylog.query_hash("a b c") != _querylog.query_hash("a b d")
    a = np.ones((2, 3), dtype=np.float32)
    assert _querylog.vector_hash(a) == _querylog.vector_hash(a.astype(np.float64))
    assert _querylog.vector_hash(a) != _querylog.vector_hash(a.reshape(3, 2))


# ============================================================================ max_disk


def test_highest_target_that_fits(fakes):
    target, row = _budget.highest_target_that_fits("fake/model", 1000, 600_000, False)
    assert target == 0.96 and row.name == "int4(mean)"
    target, row = _budget.highest_target_that_fits("fake/model", 1000, "1.1 MB", False)
    assert target == 0.98 and row.name == "int8(per_vector)"
    target, row = _budget.highest_target_that_fits("fake/model", 1000, "5 GB", False)
    assert target == 1.0 and row.name == "float32"
    assert _budget.highest_target_that_fits("fake/model", 1000, 1_000, False) == (None, None)
    target, row = _budget.highest_target_that_fits("fake/model", 1000, 150_000, True)
    assert target == 0.93 and row.name == "binary"                     # 1-bit only when checked


def test_highest_target_counts_overheads_and_steps_in_hundredths(fakes):
    target, _ = _budget.highest_target_that_fits("fake/model", 1000, 600_000, False, overhead_bytes_per_page=100)
    assert target is None or target < 0.96
    assert _budget._targets(1.0, 0.9) == [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9]


def test_rows_without_catalogue_bytes():
    from dataclasses import replace

    row = replace(F.fake_row("int8(per_vector)"), bytes_per_page=None)
    assert _budget.bytes_per_page(row) == (None, "no measured size for this layout and no vectors to count")
    per_page, basis = _budget.bytes_per_page(row, vectors_per_page=100, dim=128)
    assert basis == "your vectors" and per_page == pytest.approx(100 * (128 + 2))
    merged = replace(row, spec={"stages": [{"stage": "hierarchical_merge", "ratio": 0.25}, {"stage": "int8"}]})
    assert _budget.bytes_per_page(merged, vectors_per_page=100, dim=128)[0] is None


def test_parse_size():
    assert _budget.parse_size("2 GB") == 2_000_000_000
    assert _budget.parse_size("1.5GB") == 1_500_000_000
    assert _budget.parse_size("800 MiB") == 800 * 2**20
    assert _budget.parse_size(1234) == 1234
    for bad in ("two GB", "5 parsecs", -1, 0):
        with pytest.raises(ValueError):
            _budget.parse_size(bad)
    with pytest.raises(TypeError):
        _budget.parse_size(True)
