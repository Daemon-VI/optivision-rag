"""``optivision.easy.options()`` (FINAL_DESIGN 4.1 and the journeys of section 2): its output,
its numbers (each from the rule's rows, through ``_claim``), the plan for a folder, Start here,
and the chart hooks. ``_docs``, ``_encode`` and ``_chart`` are replaced by fakes that follow
docs/easy/CONTRACT.md."""

from __future__ import annotations

import json
import re
import sys
import types
from collections import Counter, namedtuple
from pathlib import Path

import pytest

import optivision.easy as ov
from optivision.easy import _claim, _ledger, _levels, _options
from optivision.easy._types import AmbiguousModel, UnknownModel

PageCount = namedtuple("PageCount", "n_pages n_files skipped")
Estimate = namedtuple("Estimate", "lo_s hi_s basis gpu")

BANNED_PHRASES = ("universal compression", "works with every model", "guaranteed", "state of the art",
                  "state-of-the-art", "novel", "better than", "lossless", "faster", "scales to")
BANNED_WORDS = re.compile(r"\b(simply|just|universal|first|sota|veto)\b", re.I)
FD_A = "vidore/colqwen2-v1.0-hf"


def _install(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(f"optivision.easy.{name}")
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(ov, name, module, raising=False)
    return module


@pytest.fixture
def no_rates(monkeypatch):
    def estimate(model_id, n_pages):
        raise KeyError(model_id)

    return _install(monkeypatch, "_encode", estimate=estimate)


@pytest.fixture
def rates(monkeypatch):
    calls = []

    def estimate(model_id, n_pages):
        calls.append((model_id, n_pages))
        return Estimate(0.61 * n_pages, 0.65 * n_pages, "our Kaggle run, 0.61-0.65 s/page", "one T4")

    _install(monkeypatch, "_encode", estimate=estimate)
    return calls


@pytest.fixture
def folder(monkeypatch, tmp_path):
    src = tmp_path / "my_pdfs"
    src.mkdir()
    seen = []

    def count_pages(source, progress=None):
        seen.append(source)
        return PageCount(8400, 412, [(str(src / "locked.pdf"), "encrypted"), (str(src / "bad.pdf"), "damaged")])

    _install(monkeypatch, "_docs", count_pages=count_pages)
    return src, seen


@pytest.fixture
def chart(monkeypatch):
    calls = []

    def svg(selection, *, n_pages=None):
        calls.append(("svg", selection, n_pages))
        return "<svg data-pick='%s'></svg>" % selection.pick.name

    def html(selection):
        calls.append(("html", selection))
        return "<!doctype html><title>chart</title>"

    _install(monkeypatch, "_chart", svg=svg, html=html)
    return calls


def _honest(text: str) -> None:
    lines = text.splitlines()
    assert lines[-1].startswith("Next: ")
    assert all(ord(c) < 128 for c in text), [c for c in text if ord(c) >= 128]
    low = text.lower()
    assert not [p for p in BANNED_PHRASES if p in low]
    assert not BANNED_WORDS.findall(text)
    assert "validated" not in low  # in-sample numbers are never called validated


def _claims(o) -> list[_claim.Claim]:
    return o.block.claims()


# ------------------------------------------------------------------------------ journey (a)


def test_journey_a_known_model_and_target():
    o = ov.options(FD_A, target=0.97)
    text = str(o)
    _honest(text)
    sel = o.selection
    pick, safer = sel.pick, sel.safer
    src = "[our benchmarks: ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages]"
    assert text.splitlines()[0].startswith(f"{FD_A}, target 97%, collections up to 500 pages")
    assert re.search(r"^pick +merged \(1 in 4 vectors\) \+ 8-bit, layout=\"hm\(0\.25\) > int8\"; close call$", text, re.M)
    assert f"23.9-24.4 MB per thousand pages, {_claim.ratio(pick.ratio_f16)} smaller than float16  {src}" in text
    assert f"97.2% of full-precision search quality (nDCG@5)  {src}" in text
    assert "clears 97% by under half a point. We use it by default." in text
    assert (f'Safer: layout="int4(mean)" (centred 4-bit), 48.3-49.4 MB, {_claim.ratio(safer.ratio_f16)}, '
            f"lowest kept 98.0%  {src}") in text
    larger = next(line for line in text.splitlines() if line.startswith("larger"))
    assert "1,000 pages: centred 4-bit" in larger and "2,500 pages: centred 4-bit" in larger
    assert "20,000 pages: 8-bit" in larger and larger.endswith(src)
    assert f"status       {_ledger.status()}" in text
    assert text.endswith(f'Next: ov.build("my_pdfs/", model="{FD_A}", target=0.97)')
    assert "layouts" not in text  # the ladder table is for options(model) without a target
    assert len(o.rows) == 11 and o.pick is pick and o.model == FD_A and o.target == 0.97
    # every number is a claim: the size, ratio and lowest kept come from the pick's rows
    values = [c.value for c in _claims(o) if c.kind == "catalogue"]
    assert pick.bytes_per_page[0] in values and pick.ratio_f16 in values and pick.lowest_kept in values


def test_every_catalogue_tag_names_datasets_with_sizes():
    for model, t in ((FD_A, 0.97), ("vidore/colpali-v1.3", 0.95), ("colbert-ir/colbertv2.0", 0.97),
                     ("vidore/colSmol-256M", 0.97), ("vidore/colqwen2-v1.0", 0.98)):
        o = ov.options(model, target=t)
        _honest(str(o))
        for c in _claims(o):
            assert c.kind in ("catalogue", "counted")
            if c.kind == "catalogue":
                assert re.search(r"\d[\d,]* (pages|documents)", c.source), c.source
        for line in o.block.lines:
            assert len({(c.kind, c.source) for c in line.claims}) <= 1


def test_text_models_say_documents_and_ndcg_at_10():
    o = ov.options("colbert-ir/colbertv2.0", target=0.97)
    text = str(o)
    assert "collections up to 15,183 documents" in text and "(nDCG@10)" in text
    assert "MB per thousand documents" in text and "above 15,183 documents: 8-bit" in text
    assert "BEIR SciFact 5,183 documents, NFCorpus 3,633 documents, SciDocs 25,657 documents" in text


# ------------------------------------------------------------------------------ journey (b)


def test_journey_b_table_at_the_default_target_and_the_chart(chart, tmp_path):
    o = ov.options(FD_A)
    text = str(o)
    _honest(text)
    assert o.target == 0.97 and "target 97% (the default)" in text
    for row in o.rows:
        assert row.plain in text and row.status in text
    assert "needs practice questions" in text and "below target (below its needed line)" in text
    assert o.table().splitlines()[0].startswith("layouts")
    html = o.save(tmp_path / "colqwen2.html")
    svg = o.save(tmp_path / "colqwen2.svg")
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>") and svg.suffix == ".svg"
    assert [c[0] for c in chart] == ["html", "svg"] and chart[0][1] is o.selection
    assert o._repr_html_().startswith("<svg data-pick='hierarchical_merge(0.25) > int8(per_vector)'>")
    with pytest.raises(ValueError, match=r"\.html or \.svg"):
        o.save(tmp_path / "colqwen2.txt")


def test_repr_html_without_the_chart_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "optivision.easy._chart", None)
    monkeypatch.delattr(ov, "_chart", raising=False)
    html = ov.options(FD_A, target=0.95)._repr_html_()
    assert html.startswith("<pre>") and "&quot;" in html


# ------------------------------------------------------------------------------ journey (c)


def test_journey_c_plan_for_a_folder(folder, rates, monkeypatch):
    src, seen = folder
    monkeypatch.setattr(_options, "_on_colab", lambda: False)
    work = src.parent / "ov_work"
    o = ov.options(FD_A, target=0.97, pages=str(src), work_dir=work)
    text = str(o)
    _honest(text)
    assert seen == [str(src)] and rates == [(FD_A, 8400)]
    assert o.n_pages == 8400 and o.selection.n_pages == 8400
    assert "8,400 pages in 412 files; 2 files skipped  [counted]" in text
    assert "locked.pdf: encrypted" in text
    assert re.search(r"^layout +centred 4-bit, layout=\"int4\(mean\)\" \(larger folders get bigger safety margins\); "
                     r"close call  \[our benchmarks: ", text, re.M)
    est = "[estimate: our bytes per page x your pages]"
    assert f"about 414.9 MB of vectors, {_claim.ratio(o.pick.ratio_f16)} smaller than float16  {est}" in text
    assert "about 1.4-1.5 h on one T4  [estimate: our Kaggle run, 0.61-0.65 s/page]" in text
    assert "not in our benchmark runs (their pages were images)" in text
    assert f"1.6 GB for the float16 copy in {work}  {est}" in text
    assert re.search(r"^free +\d+\.\d [KMG]?B reported by that drive  \[counted\]$", text, re.M)
    assert f"keep_float16=True keeps the 1.6 GB copy so a change needs none  {est}" in text
    assert text.endswith(f'Next: ov.build("{src}", model="{FD_A}", target=0.97, work_dir="{work}")')
    assert o.plan["temporary_bytes"] == pytest.approx(8400 * o.selection.rows[1].bytes_per_page[1])
    assert o.plan["encode_seconds"] == pytest.approx([0.61 * 8400, 0.65 * 8400])


def test_colab_warning_and_default_work_dir(folder, rates, monkeypatch):
    src, _ = folder
    monkeypatch.setattr(_options, "_on_colab", lambda: True)
    text = str(ov.options(FD_A, target=0.97, pages=str(src)))
    assert f"for the float16 copy in {Path('/content/.ov_work')}  [estimate" in text
    assert "A Colab disconnect loses the temporary copy in /content. Pass work_dir= on Drive" in text
    monkeypatch.setattr(_options, "_on_colab", lambda: False)
    assert _options.default_work_dir(str(src)) == src.resolve().parent / "my_pdfs.ovx.work"
    assert _options.default_work_dir(str(src / "report.pdf")).name == "report.ovx.work"


def test_not_enough_disk_is_said_before_any_work(folder, rates, monkeypatch):
    src, _ = folder
    monkeypatch.setattr(_options, "_on_colab", lambda: False)
    monkeypatch.setattr(_options, "_free_bytes", lambda path: (1.0e6, path))
    text = str(ov.options(FD_A, target=0.97, pages=str(src)))
    assert "1.0 MB reported by that drive  [counted]" in text and "not enough free disk" in text


def test_a_page_count_without_a_folder(no_rates, monkeypatch):
    monkeypatch.setattr(_options, "_on_colab", lambda: False)
    o = ov.options(FD_A, target=0.97, pages=2500)
    text = str(o)
    _honest(text)
    assert f"{FD_A}, target 97%, 2,500 pages (as given)" in text
    assert not re.search(r"^(folder|rendering) ", text, re.M)
    assert "not timed: our runs recorded no speed for this model" in text
    assert o.pick.name == "int4(mean)"  # FINAL_DESIGN 5.1.3: centred 4-bit at 2,500 pages
    big = str(ov.options(FD_A, target=0.97, pages=25000))
    assert ("more than 8x larger than the largest collection we measured (2,500 pages); only 8-bit and centred "
            "4-bit layouts  [our benchmarks: ViDoRe V1 DocVQA 500 pages with 2,000 pages") in big
    assert "more than 8x" not in str(ov.options(FD_A, target=0.97, pages=20000))
    for bad in (0, -3):
        with pytest.raises(ValueError):
            ov.options(FD_A, pages=bad)
    with pytest.raises(TypeError):
        ov.options(FD_A, pages=True)


# ------------------------------------------------------------------------------ target and accuracy


def test_accuracy_prints_what_target_means(capsys):
    o = ov.options(FD_A, accuracy=0.97)
    out = capsys.readouterr().out
    assert out.strip() == ("accuracy=0.97 read as target=0.97: keep at least 97% of full-precision search quality "
                           "(nDCG@5); not 97% of answers correct.")
    assert o.target == 0.97 and out.strip() in o.notes
    with pytest.raises(ValueError, match="disagree"):
        ov.options(FD_A, target=0.95, accuracy=0.97)


def test_target_range_and_notes():
    for bad in (0.85, 1.2):
        with pytest.raises(ValueError, match="between 0.9 and 1.0"):
            ov.options(FD_A, target=bad)
    assert "92% is less tested" in str(ov.options(FD_A, target=0.92))
    assert "cannot separate layouts this close to 100%" in str(ov.options(FD_A, target=0.995))
    assert "less tested" not in str(ov.options(FD_A, target=0.95))


# ------------------------------------------------------------------------------ resolution and bases


def test_unknown_and_ambiguous_models():
    with pytest.raises(UnknownModel) as err:
        ov.options("vidore/colqwen2-v9")
    assert "Closest ids:" in str(err.value)
    with pytest.raises(AmbiguousModel, match="colbert-ir/colbertv2.0 .*lightonai/colbertv2.0"):
        ov.options("colbertv2.0")
    assert ov.options("colqwen2-v1.0-hf", target=0.97).model == FD_A


def test_measured_via_unmeasured_and_pending_rows_are_said():
    via = str(ov.options("vidore/colqwen2-v1.0", target=0.97))
    assert "vidore/colqwen2-v1.0 has no rows of its own; it gives the same vectors as vidore/colqwen2-v1.0-merged" in via
    un = ov.options("vidore/colSmol-256M", target=0.97)
    text = str(un)
    assert "was not measured; results have not been shown to carry over to new models" in text
    assert "97.8% of full-precision search quality (nDCG@5) on the worst model we measured" in text
    assert "worst of the 40 page models we measured" in text and un.pick.name == "int8(per_vector)"
    assert "your model's bytes per page were not measured" in text
    if _levels.data()["models"]["vidore/colqwen2.5-v0.2"].get("pending_b0"):
        assert "re-measurement (B0)" in str(ov.options("vidore/colqwen2.5-v0.2", target=0.97))


def test_as_dict_is_json():
    for o in (ov.options(FD_A, target=0.97), ov.options()):
        d = json.loads(json.dumps(o.as_dict()))
        assert d["lines"]["next"] == o.block.next
    d = ov.options(FD_A, target=0.97).as_dict()
    assert d["pick"] == "hierarchical_merge(0.25) > int8(per_vector)" and d["selection"]["close_call"] is True
    assert d["selection"]["sequence"][-1] == "float16" and len(d["rows"]) == 11


# ------------------------------------------------------------------------------ journey (0)


#: FINAL_DESIGN 2 (0): Start here, full-precision nDCG@5 (DocVQA / InfoVQA), s/page, pick at 97%
FD_START = [
    ("tencent/EVIE-4.5B", (0.673, 0.947), (1.18, 1.21), "int8(per_vector)", "2.0x"),
    ("VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1", (0.660, 0.940), (1.20, 1.45),
     "hierarchical_merge(0.33) > int4(mean)", "11.7x"),
    ("VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1", (0.649, 0.926), (0.75, 0.86),
     "hierarchical_merge(0.33) > int4(mean)", "11.7x"),
    ("vidore/colqwen2-v1.0-hf", (0.604, 0.915), (0.61, 0.65), "hierarchical_merge(0.25) > int8(per_vector)", "7.9x"),
]


def test_journey_0_every_registry_model_and_start_here():
    from optivision.adapters import MODELS

    o = ov.options()
    text = str(o)
    _honest(text)
    assert o.selection is None and o.rows == [] and o.pick is None
    assert len(o.models) == len(MODELS) == 65
    assert Counter(m["licence"] == "not recorded" for m in o.models)[True] == 13
    assert Counter(m["status"] for m in o.models) == {"full rows": 55, "no rows": 2, "not measured": 8}
    assert "no rows (measured via vidore/colqwen2-v1.0-merged)" in text
    got = [(s["model"], tuple(round(s["baselines"][d], 3) for d in s["datasets"]),
            tuple(round(x, 2) for x in s["seconds_per_page"]), s["pick"], f"{s['ratio_f16']:.1f}x")
           for s in o.start_here]
    assert got == FD_START
    assert [s["quickest"] for s in o.start_here] == [False, False, False, True]
    assert text.splitlines()[0].startswith("Start here:")
    assert "vidore/colqwen2-v1.0-hf (quickest)" in text and "0.61-0.65 s/page" in text
    assert "ran on one T4, float16, 0.61-0.65 s/page" in text and "ran on 2x T4, float32" in text
    assert text.endswith('Next: ov.options("vidore/colqwen2-v1.0-hf", target=0.97)')
    with pytest.raises(ValueError):
        o.save("x.html")
    with pytest.raises(ValueError):
        o.table()


def test_start_here_at_another_target():
    o = ov.options(target=0.95)
    picks = {s["model"]: s["pick"] for s in o.start_here}
    assert picks["vidore/colqwen2-v1.0-hf"] == "hierarchical_merge(0.25) > int4(mean)"
    assert "at 95%:" in str(o)
