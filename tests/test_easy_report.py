"""The report of an index (FINAL_DESIGN 4.4): sections of lines, every number with its tag, saved as .html, .md, .json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _easy_lint import banned_hits, non_ascii

from optivision.easy import _claim
from optivision.easy._claim import Block, Line, MixedEvidence, claim, lit
from optivision.easy._report import Report

SRC = "ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages"


def _build_block() -> Block:
    block = Block([], title="Build")
    block.add("Pages", "{0} pages from {1} files", claim(412, "counted"), claim(37, "counted"))
    block.add("Layout", "merged + 8-bit, lowest kept {0}",
              claim(0.9728, "catalogue", SRC, "nDCG@5 retention", _claim.pct(0.9728)))
    block.add("Size", "{0} on disk, {1} smaller than float16",
              claim(10.6e6, "counted", text=_claim.size(10.6e6)), claim(7.9, "counted", text=_claim.ratio(7.9)))
    block.add("Encoding", "{0} on one T4", claim(5400, "estimate", "our Kaggle run, 0.61-0.65 s/page",
                                                       text=_claim.duration(5400)))
    return block


def _skipped_block() -> Block:
    block = Block([])
    block.add("report.pdf", "skipped: {0}", lit("encrypted"))
    block.add("", "{0} files skipped", claim(2, "counted"))
    return block


def _report(**kw) -> Report:
    meta = {"model": "vidore/colqwen2-v1.0-hf", "index": lit("my_pdfs.ovx"), "built": "2026-10-21"}
    return Report([("Build", _build_block()), ("Skipped files", _skipped_block())], meta, **kw)


def test_lines_start_with_the_title_and_end_with_next():
    report = _report()
    lines = report.lines
    assert lines[0] == "OptiVision report"
    assert "  model: vidore/colqwen2-v1.0-hf" in lines and "  index: my_pdfs.ovx" in lines
    assert lines[-1].startswith("Next: ")
    assert str(report) == "\n".join(lines)
    assert repr(report) == str(report)


def test_every_line_with_a_number_carries_its_tag():
    report = _report()
    lines = report.lines
    for needle, tag in (("412 pages from 37 files", "[counted]"),
                        ("lowest kept 97.2%", f"[our benchmarks: {SRC}]"),
                        ("10.6 MB on disk", "[counted]"),
                        ("about 1.5 h on one T4", "[estimate: our Kaggle run, 0.61-0.65 s/page]")):
        line = next(ln for ln in lines if needle in ln)
        assert line.endswith(tag), line
    assert all(c.tag.startswith("[") and c.tag.endswith("]") for c in report.claims())
    assert len(report.claims()) == 7


def test_two_kinds_on_one_line_is_refused_when_building_a_section():
    block = Block([])
    with pytest.raises(MixedEvidence):
        block.add("Mixed", "{0} of {1}", claim(0.97, "catalogue", SRC), claim(137, "probe"))
    with pytest.raises(_claim.BareNumber):
        block.add("Bare", "kept 97.2% of the quality")


def test_float32_and_geometry_slots_are_always_listed():
    report = _report()
    titles = [t for t, _ in report.sections]
    assert titles[:2] == ["Build", "Skipped files"]
    assert titles[-2:] == [Report.FLOAT32, Report.GEOMETRY]
    assert "Not recorded for this index." in str(report.section(Report.FLOAT32))
    text = str(report)
    assert Report.FLOAT32 in text and Report.GEOMETRY in text


def test_supplied_slots_replace_the_placeholder_and_keep_their_place():
    f32 = Block([])
    f32.add("8-bit", "{0} smaller than float32", claim(3.9, "counted", text=_claim.ratio(3.9)))
    report = Report([(Report.FLOAT32, f32), ("Build", _build_block())], {})
    titles = [t for t, _ in report.sections]
    assert titles == [Report.FLOAT32, "Build", Report.GEOMETRY]
    assert "3.9x smaller than float32" in str(report)
    assert str(report).count(Report.FLOAT32) == 1
    assert report.section("ratios against float32") is f32, "titles are matched without regard to case"
    with pytest.raises(KeyError):
        report.section("nothing")


def test_bad_sections_and_bare_numbers_in_meta_are_refused():
    with pytest.raises(TypeError):
        Report([("Build", "just text")])
    with pytest.raises(TypeError):
        Report([("Build",)])
    with pytest.raises(TypeError):
        Report([], {"pages": 412})
    with pytest.raises(TypeError):
        Report([], {"ratio": 7.9})
    Report([], {"pages": claim(412, "counted"), "flag": True, "none": None})      # a claim carries its tag


def test_meta_claims_print_with_their_tag():
    report = Report([], {"pages": claim(412, "counted")})
    assert "  pages: 412  [counted]" in report.lines
    assert report.as_dict()["meta"]["pages"]["tag"] == "[counted]"


def test_as_dict_is_json_with_every_claim_tagged():
    report = _report()
    data = json.loads(json.dumps(report.as_dict()))
    assert data["title"] == "OptiVision report" and data["next"].startswith("index.check()")
    assert [s["title"] for s in data["sections"]][-2:] == [Report.FLOAT32, Report.GEOMETRY]
    claims = [c for s in data["sections"] for ln in s["lines"] for c in ln["claims"]]
    assert len(claims) == len(report.claims())
    assert all(c["tag"] and c["kind"] in _claim.KINDS for c in claims)
    layout = next(ln for s in data["sections"] for ln in s["lines"] if ln["label"] == "Layout")
    assert layout["tag"] == f"[our benchmarks: {SRC}]" and layout["claims"][0]["construct"] == "nDCG@5 retention"


def test_save_html_is_self_contained_a4_and_ascii(tmp_path):
    report = _report(title="Report for <my_pdfs> café")
    path = report.save(tmp_path / "out" / "report.html")
    assert path == tmp_path / "out" / "report.html" and path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "@page{size:A4" in text and "@media print" in text and "break-inside:avoid" in text
    assert non_ascii(text) == []
    assert "&lt;my_pdfs&gt; caf&#233;" in text, "markup is escaped and non-ASCII becomes a character reference"
    assert not re.search(r"https?://", text) and "<script" not in text and "<link" not in text and "@import" not in text
    assert text.count('<section class="sec"') == 4
    for tag in ("[counted]", f"[our benchmarks: {SRC}]", "[estimate: our Kaggle run, 0.61-0.65 s/page]"):
        assert tag in text
    assert "k-catalogue" in text and "k-counted" in text and "k-estimate" in text
    assert "prefers-color-scheme: dark" in text
    assert text.rstrip().endswith("</html>")
    assert banned_hits(text) == []


def test_save_markdown_and_json(tmp_path):
    report = _report()
    md = report.save(tmp_path / "r.md").read_text(encoding="utf-8")
    assert md.startswith("# OptiVision report\n")
    assert "## Build" in md and "## Skipped files" in md and f"## {Report.GEOMETRY}" in md
    assert "- **Pages** 412 pages from 37 files `[counted]`" in md
    assert md.rstrip().splitlines()[-1].startswith("Next: ")
    assert "\r" not in md
    data = json.loads(report.save(tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data == json.loads(json.dumps(report.as_dict()))
    assert banned_hits(md) == []


def test_save_follows_the_suffix_and_refuses_others(tmp_path):
    report = _report()
    for name in ("a.htm", "a.HTML", "a.markdown", "a.MD"):
        assert report.save(tmp_path / name).exists()
    for name in ("a.txt", "a", "a.pdf"):
        with pytest.raises(ValueError, match=r"\.html, \.md or \.json"):
            report.save(tmp_path / name)
    assert not (tmp_path / "a.txt").exists()
    assert isinstance(report.save(str(tmp_path / "s.json")), Path)


def test_html_escapes_labels_and_literals_and_stays_ascii():
    block = Block([])
    block.add("<b>label</b>", "{0}", lit("a & b <i>é</i> →"))
    text = Report([("S", block)], {}).to_html()
    assert "&lt;b&gt;label&lt;/b&gt;" in text and "a &amp; b &lt;i&gt;&#233;&lt;/i&gt; &#8594;" in text
    assert non_ascii(text) == []


def test_repr_html_is_a_plain_block_that_leaks_no_style():
    html = _report()._repr_html_()
    assert html.startswith("<pre") and "<style" not in html and "OptiVision report" in html


def test_from_index_is_a_thin_hook():
    class Stub:
        def _report_parts(self):
            return [("Build", _build_block())], {"model": "m"}

    report = Report.from_index(Stub())
    assert isinstance(report, Report) and "412 pages" in str(report)
    with pytest.raises(NotImplementedError, match="_report_parts"):
        Report.from_index(object())


def test_empty_report_still_lists_both_slots_and_a_next_line():
    report = Report([], {})
    assert [t for t, _ in report.sections] == [Report.FLOAT32, Report.GEOMETRY]
    assert report.lines[-1].startswith("Next: ")
    assert Report([], {}, next="ov.load('x.ovx')").lines[-1] == "Next: ov.load('x.ovx')"


def test_glossary_note_can_ride_in_a_section():
    from optivision.easy import _glossary

    block = Block([])
    for note in _glossary.notes_for("lowest kept and close call", set()):
        key, meaning = note.split(": ", 1)
        block.add(key, "{0}", lit(meaning))
    text = str(Report([("Terms", block)], {}))
    assert "Bottom of the 95% range we measured" in text and "Cleared the target by under half a point" in text
    assert len(Line("x", "{0}", lit("y")).claims) == 0
