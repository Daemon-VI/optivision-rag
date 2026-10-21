"""The chart of easy mode: the static SVG, the self-contained page, and the page's rule against the Python rule.

The SVG and page tests run on ``Selection`` objects built by hand (``tests/_easy_presentation_fixtures.py``).
The page's script holds the selection rule of FINAL_DESIGN 5.1.2; it is run under ``node`` and compared

* with ``reference_select``, a literal reading of the design text kept in the fixtures (runs now), and
* with ``optivision.easy._levels.select`` for every model of the registry x four targets x three page
  counts (runs once the rule module exists).
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _easy_lint import banned_hits, non_ascii
from _easy_presentation_fixtures import (
    PAGE_SPECS,
    POOLED_SPECS,
    TEXT_SPECS,
    need_of,
    random_specs,
    reference_select,
    selection_for,
)

from optivision.easy import _chart
from optivision.easy._types import Selection

SVG_NS = "{http://www.w3.org/2000/svg}"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed; the page's rule cannot be run")

TARGETS_GRID = [round(0.90 + 0.005 * i, 3) for i in range(21)]
PAGES_GRID = [1, 250, 499, 500, 501, 1000, 2500, 5244, 15183, 15184, 20000, 20001, 50000]


# ---------------------------------------------------------------------------------------- helpers


def _marks(root: ET.Element) -> list[ET.Element]:
    """The marks of the plot (the ones with a tooltip), in row order."""
    return [g for g in root.iter(SVG_NS + "g") if "ov-mark" in g.get("class", "") and g.find(SVG_NS + "title") is not None]


def _texts(root: ET.Element) -> list[str]:
    return ["".join(t.itertext()) for t in root.iter(SVG_NS + "text")]


def _all_text(svg: str) -> str:
    return "\n".join(_texts(ET.fromstring(svg)))


def _sel(target: float = 0.97, n: int | None = 500, **kw) -> Selection:
    return selection_for(PAGE_SPECS, target, n, **kw)


# ---------------------------------------------------------------------------------------- the SVG


def test_svg_is_well_formed_accessible_ascii_and_offline():
    text = _chart.svg(_sel())
    root = ET.fromstring(text)
    assert root.tag == SVG_NS + "svg"
    assert root.get("role") == "img"
    title, desc = root.find(SVG_NS + "title"), root.find(SVG_NS + "desc")
    assert title is not None and "target 97%" in "".join(title.itertext())
    assert desc is not None and "Pick:" in "".join(desc.itertext())
    assert root.get("aria-labelledby") == f"{title.get('id')} {desc.get('id')}"
    assert non_ascii(text) == []
    # nothing is fetched: the only URL is the SVG namespace
    assert re.findall(r"https?://[^\"' ]+", text) == ["http://www.w3.org/2000/svg"]
    for forbidden in ("<script", "href=", "url(", "@import", "<image", "<foreignObject"):
        assert forbidden not in text
    assert banned_hits(text) == []


def test_svg_has_both_themes_inside_itself():
    text = _chart.svg(_sel())
    assert "prefers-color-scheme: dark" in text
    assert ".ov-bg{fill:#fcfcfb}" in text and ".ov-bg{fill:#1a1a19}" in text
    root = ET.fromstring(text)
    # a background rect, so the chart reads on any page colour
    assert any("ov-bg" in r.get("class", "") for r in root.iter(SVG_NS + "rect"))


def test_one_mark_per_row_filled_when_it_qualifies_hollow_when_refused():
    sel = _sel()
    root = ET.fromstring(_chart.svg(sel))
    marks = _marks(root)
    assert len(marks) == len(sel.rows)
    for row, mark in zip(sel.rows, marks):
        title = "".join(mark.find(SVG_NS + "title").itertext())
        assert title.startswith(row.plain + ":")
        classes = mark.get("class", "").split()
        qualifies = row.status in ("pick", "qualifies", "close call")
        assert ("ov-yes" in classes) == qualifies, row.name
        assert ("ov-no" in classes) == (not qualifies), row.name
    statuses = {r.name: r.status for r in sel.rows}
    assert statuses["hm(0.25) > int8"] == "pick"
    assert statuses["hm(0.33) > int4(mean)"] == "below target"
    assert statuses["binary"] == "needs practice questions"


def test_refused_rows_say_why_in_plain_words():
    text = "\n".join(_texts(ET.fromstring(_chart.svg(_sel()))))
    assert "below its needed line" in text
    assert "needs practice questions" in text
    # the family-stopped, not-measured and size-range words are the design's wording
    assert _chart.REASONS["family stopped"] == "a less compressed layout in this family already fell short"
    assert _chart.REASONS["not measured"] == "not measured"
    assert _chart.REASONS["outside size range"] == "outside the sizes we measured"


def test_every_reason_of_the_design_reaches_the_chart():
    specs = [dict(sp) for sp in PAGE_SPECS]
    for sp in specs:                                   # no rows for the middle of the 1-bit chain
        if sp["name"] == "hm(0.33) > binary":
            sp["low"], sp["worst"] = None, None
    sel = selection_for(specs, 0.97, 25000, checked=True)
    statuses = {r.name: r.status for r in sel.rows}
    assert statuses["hm(0.33) > binary"] == "not measured"
    assert statuses["hm(0.25) > binary"] == "outside size range"      # merged, above 20,000 pages
    text = _all_text(_chart.svg(sel))
    for words in ("not measured", "outside the sizes we measured"):
        assert words in text
    sel = selection_for(PAGE_SPECS, 0.97, 500, checked=True, blocked=frozenset({"hm(0.33) > int4(mean)"}))
    assert {r.name: r.status for r in sel.rows}["hm(0.33) > int4(mean)"] == "not allowed"
    assert "the page vectors per page are below 70% of the catalogue mean" in _all_text(_chart.svg(sel))
    stopped = [dict(sp, low=0.999) if sp["name"] == "hm(0.33) > int8" else sp for sp in PAGE_SPECS]
    sel = selection_for(stopped, 0.98, 500, checked=True)          # 8-bit falls short, the merge above it clears
    assert {r.name: r.status for r in sel.rows}["hm(0.33) > int8"] == "family stopped"
    assert _chart.REASONS["family stopped"] in _all_text(_chart.svg(sel))


def test_the_pick_is_ringed_once_and_the_needed_ticks_match_the_rows():
    sel = _sel()
    root = ET.fromstring(_chart.svg(sel))
    rings = [c for c in root.iter(SVG_NS + "circle") if c.get("class") == "ov-ring" and c.get("r") == "12"]
    assert len(rings) == 1
    ticks = [ln for ln in root.iter(SVG_NS + "line") if ln.get("class") == "ov-need" and float(ln.get("x2")) - float(ln.get("x1")) == 26.0]
    assert len(ticks) == sum(1 for r in sel.rows if r.need is not None and r.name != "float32")
    # a tick sits above a mark exactly when the row clears its need
    plot_marks = {r.name: m for r, m in zip(sel.rows, _marks(root))}
    assert plot_marks["hm(0.25) > int8"].get("class").endswith("ov-yes")


def test_a_ticks_height_follows_its_need():
    sel = _sel(0.97, 500)
    sel2 = _sel(0.97, 20000)
    ys = [sorted(float(ln.get("y1")) for ln in ET.fromstring(_chart.svg(s)).iter(SVG_NS + "line")
                 if ln.get("class") == "ov-need" and float(ln.get("x2")) - float(ln.get("x1")) == 26.0)
          for s in (sel, sel2)]
    assert ys[0] != ys[1], "a larger collection moves the needed lines up"
    assert min(ys[1]) <= min(ys[0])


def test_axis_labels_name_the_evidence_and_the_unit():
    text = _chart.svg(_sel())
    assert "[our benchmarks: ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages]" in text
    assert "MB per 1,000 pages, log scale" in text
    assert "lowest kept, share of full-precision nDCG@5 [our benchmarks]" in text
    assert "These numbers come from our tests" in text            # the ledger's status line


def test_legend_names_every_mark_kind():
    text = "\n".join(_texts(ET.fromstring(_chart.svg(_sel()))))
    for word in ("qualifies", "refused (reason in the list)", "the pick", "needed line: target plus margin",
                 "uncompressed", "8-bit", "centred 4-bit", "1-bit", "bar: from lowest kept to worst point"):
        assert word in text


def test_rows_without_rows_of_their_own_sit_in_a_labelled_band():
    sel = _sel()
    text = "\n".join(_texts(ET.fromstring(_chart.svg(sel))))
    assert "no measured rows" in text                 # float32 has no measured low


def test_hand_pick_is_labelled_chosen_by_you():
    sel = _sel(hand="int4(mean)")
    text = _chart.svg(sel)
    assert "chosen by you" in text and "Chosen by you: centred 4-bit" in text


def test_close_call_names_the_safer_layout():
    sel = _sel()
    assert sel.close_call
    text = _chart.svg(sel)
    assert "the pick (close call)" in text
    desc = ET.fromstring(text).find(SVG_NS + "desc")
    assert "Hollow marks were refused" in "".join(desc.itertext())
    assert "Close call" in _chart._summary(sel) and "Safer: merged (1 in 3 vectors) + 8-bit" in _chart._summary(sel)


def test_numbers_round_the_safe_way():
    assert _chart._pct_up(0.9701) == "97.1%"
    assert _chart._pct_up(0.97) == "97.0%"
    assert _chart._pct_up(0.97 + 0.0175 * 2.2920296742201793) == "101.1%"
    assert _chart._pct_target(0.97) == "97%" and _chart._pct_target(0.975) == "97.5%"
    from optivision.easy import _claim
    assert _claim.pct(0.97289) == "97.2%"


def test_svg_ends_with_a_next_line():
    text = _all_text(_chart.svg(_sel()))
    assert text.splitlines()[-1] == 'Next: ov.build("my_pdfs/", model="vendor/colqwen2-like", target=0.97)'
    assert _chart.html(_sel()).count("Next: ov.build(") == 1, "the page has one Next line; its svg has none"


def test_svg_is_deterministic():
    assert _chart.svg(_sel()) == _chart.svg(_sel())


def test_n_pages_labels_the_chart():
    sel = selection_for(PAGE_SPECS, 0.97, None)
    text = _chart.svg(sel, n_pages=500)
    assert "collections up to 500 pages" in text
    assert _chart.svg(sel, n_pages=None) == text


def test_another_page_count_runs_the_rule_again():
    try:
        from optivision.easy import _levels
    except ImportError:
        sel = selection_for(PAGE_SPECS, 0.97, None)
        with pytest.raises(ValueError, match="another page count"):
            _chart.svg(sel, n_pages=20000)          # without the rule module there is nothing to run
        return
    sel = _levels.select(None, 0.97, 500)
    assert _chart.svg(sel, n_pages=20000) == _chart.svg(_levels.select(None, 0.97, 20000))
    assert "collections up to 20,000 pages" in _chart.svg(sel, n_pages=20000)
    assert _chart.html(sel, n_pages=2500) == _chart.html(_levels.select(None, 0.97, 2500))


def test_empty_selection_is_refused():
    sel = _sel()
    empty = Selection(**{**sel.__dict__, "rows": ()})
    with pytest.raises(ValueError):
        _chart.svg(empty)


def test_text_ladder_and_unmeasured_basis():
    sel = selection_for(TEXT_SPECS, 0.97, 500, modality="text", model=None,
                        basis="unmeasured: pooled minimum over 15 text models")
    text = _chart.svg(sel)
    root = ET.fromstring(text)
    assert len(_marks(root)) == len(TEXT_SPECS)
    assert "unmeasured: pooled minimum over 15 text models" in text
    assert banned_hits(text) == []


def test_a_model_without_bytes_is_placed_by_its_size_against_float16():
    sel = selection_for(POOLED_SPECS, 0.97, 500, model=None, pooled=True)
    text = _chart.svg(sel)
    root = ET.fromstring(text)
    assert "size as a share of float16, log scale" in text and "MB per 1,000 pages, log scale" not in text
    measured = [r for r in sel.rows if r.ratio_f16 is not None]
    assert len(_marks(root)) == len(measured) == 4, "rows without rows of their own are listed, not plotted"
    labels = [t for t in _texts(root) if t.endswith("%") and t[:-1].isdigit()]
    assert "100%" in labels and "50%" in labels
    all_text = _all_text(text)
    assert "not measured" in all_text and "needs practice questions" in all_text     # int4 unchecked, pooled
    assert "A model without rows of its own: layouts at target 97%" in all_text
    assert banned_hits(text) == [] and non_ascii(text) == []


def test_the_selections_notes_reach_the_footer():
    note = "unchecked: no practice questions, so no 1-bit layouts"
    sel = selection_for(PAGE_SPECS, 0.97, 500, notes=(note,))
    assert "Unchecked: no practice questions, so no 1-bit layouts." in _all_text(_chart.svg(sel))
    assert "Unchecked: no practice questions, so no 1-bit layouts." in _chart.html(sel)


# ---------------------------------------------------------------------------------------- the page


def _page(sel: Selection) -> tuple[str, str, dict]:
    page = _chart.html(sel)
    script, data = _chart.extract_rule_and_data(page)
    return page, script, data


def test_page_is_self_contained_and_ascii():
    page, _script, data = _page(_sel())
    assert page.startswith("<!doctype html>")
    assert non_ascii(page) == []
    urls = re.findall(r"https?://[^\"' <]+", page)
    assert set(urls) == {"http://www.w3.org/2000/svg"}
    assert not re.search(r"<script[^>]*\ssrc=", page) and "<link" not in page and "@import" not in page
    assert "<noscript>" in page
    assert banned_hits(page) == []
    ids = set(re.findall(r'id="(ov-[^"]+)"', page))
    assert {"ov-t", "ov-n", "ov-sum", "ov-next", "ov-ring", "ov-tg", "ov-sub", "ov-title", "ov-data", "ov-rule"} <= ids
    assert data["plot"]["ymax"] > data["plot"]["ymin"]
    assert [r["name"] for r in data["rows"]] == [r.name for r in _sel().rows]


def test_page_controls_start_at_the_selection():
    page, _, _ = _page(_sel(0.95, 2500))
    assert 'min="0.9"' in page and 'max="1"' in page and 'step="0.01"' in page and 'value="0.95"' in page
    assert 'id="ov-n"' in page and 'value="2500"' in page
    assert "does not mean this share of answers is right" in page
    assert "Next: ov.build(" in page and "target=0.95" in page
    page2, _, _ = _page(selection_for(PAGE_SPECS, 0.97, None))
    assert 'value="500"' in page2


def test_page_has_a_print_layout_and_both_themes():
    page, _, _ = _page(_sel())
    assert "@page{size:A4" in page and "@media print" in page
    assert "prefers-color-scheme: dark" in page


def test_the_rule_script_is_small_and_touches_no_network():
    lines = [ln for ln in _chart.RULE_JS.strip().splitlines() if ln.strip()]
    assert 60 <= len(lines) <= 120
    for forbidden in ("fetch(", "XMLHttpRequest", "import(", "eval(", "innerHTML", "document.write", "http"):
        assert forbidden not in _chart.RULE_JS
    assert non_ascii(_chart.RULE_JS) == []
    assert banned_hits(_chart.RULE_JS) == []


def test_data_carries_what_the_rule_needs():
    _, _, data = _page(_sel())
    assert set(data["margins"]) >= {"one_bit", "scale", "codecs_above", "lowest_target", "text", "text_max", "close"}
    assert data["margins"]["one_bit"] == 0.025 and data["margins"]["codecs_above"] == 20000
    by = {r["name"]: r for r in data["rows"]}
    assert by["hm(0.25) > binary"]["size"] == 0.04 and by["binary"]["size"] == 0.0175
    assert by["hm(0.25) > int8"]["size"] == 0.0175 and by["int8(per_vector)"]["size"] == 0.005
    assert by["float16"]["size"] == 0 and by["float32"]["ref"] is True and by["float16"]["ref"] is False
    assert [by[n]["ord"] for n in ("int8(per_vector)", "hm(0.33) > int8", "hm(0.25) > int8")] == [0, 1, 2]
    assert by["binary"]["one_bit"] and by["hm(0.33) > binary"]["merged"]
    assert by["hm(0.25) > int8"]["L"] == 0.9728 and by["float32"]["L"] is None
    assert data["checked"] is False and data["modality"] == "page"


# ---------------------------------------------------------------------------------------- node runners

RULE_RUNNER = r"""
const fs = require("fs"), vm = require("vm");
const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const ctx = {};
vm.createContext(ctx);
vm.runInContext(input.script, ctx);
const out = input.items.map(it => it.cases.map(c => {
  const r = ctx.ovRule(it.data, c[0], c[1], c[2]);
  return [r.pick, r.close, r.safer, r.q, r.status, r.need];
}));
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""

DOM_RUNNER = r"""
const fs = require("fs"), vm = require("vm");
const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const els = {}, listeners = {};
function el(id) {
  if (!els[id]) {
    els[id] = {value: "", textContent: "", attrs: {}, setAttribute(k, v) { this.attrs[k] = String(v); },
               addEventListener(t, f) { listeners[id + ":" + t] = f; }};
  }
  return els[id];
}
const present = new Set(input.ids);
const ctx = {document: {getElementById(id) {
  if (id === "ov-data") { return {textContent: JSON.stringify(input.data)}; }
  return present.has(id) ? el(id) : null;
}}};
vm.createContext(ctx);
vm.runInContext(input.script, ctx);
const results = input.cases.map(c => {
  el("ov-t").value = String(c[0]); el("ov-n").value = String(c[1]);
  listeners["ov-t:input"]();
  const out = {sum: els["ov-sum"].textContent, next: els["ov-next"].textContent, sub: els["ov-sub"].textContent,
               ring: Object.assign({}, els["ov-ring"].attrs), rows: [], ticks: []};
  for (let i = 0; i < input.data.rows.length; i++) {
    out.rows.push([els["ov-s" + i] ? els["ov-s" + i].textContent : null, els["ov-m" + i] ? els["ov-m" + i].attrs["class"] : null]);
    out.ticks.push(els["ov-k" + i] ? els["ov-k" + i].attrs["y1"] : null);
  }
  return out;
});
fs.writeFileSync(process.argv[3], JSON.stringify(results));
"""


def _node(tmp_path: Path, runner: str, payload: dict) -> object:
    (tmp_path / "runner.js").write_text(runner, encoding="utf-8")
    (tmp_path / "in.json").write_text(json.dumps(payload), encoding="utf-8")
    proc = subprocess.run([NODE, str(tmp_path / "runner.js"), str(tmp_path / "in.json"), str(tmp_path / "out.json")],
                          capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))


def _fixtures() -> list[tuple[str, list[dict], str, frozenset[str], bool]]:
    """(label, row specs, modality, blocked names, pooled): the hand fixtures of the standalone comparison."""
    nonmono = [dict(sp) for sp in PAGE_SPECS]
    for sp in nonmono:                                   # EVIE-like: the lighter 1-bit rungs fall short, a heavier one clears
        if sp["name"] == "binary":
            sp["low"] = 0.955
        if sp["name"] == "hm(0.33) > binary":
            sp["low"] = 0.961
        if sp["name"] == "hm(0.25) > binary":
            sp["low"] = 0.9999
    gaps = [dict(sp) for sp in PAGE_SPECS]
    for sp in gaps:                                      # unmeasured rungs in the middle of chains
        if sp["name"] in ("hm(0.33) > int8", "hm(0.33) > binary"):
            sp["low"], sp["worst"] = None, None
    edge = [dict(sp) for sp in PAGE_SPECS]
    for sp in edge:                                      # lowest kept exactly at the target: clears at 500 pages
        if sp["name"] == "int4(mean)":
            sp["low"] = 0.975
        if sp["name"] == "float16":
            sp["low"] = 0.99
    return [
        ("page ladder", PAGE_SPECS, "page", frozenset(), False),
        ("page ladder, shape gate on one merge", PAGE_SPECS, "page", frozenset({"hm(0.25) > int8"}), False),
        ("non-monotone 1-bit chain", nonmono, "page", frozenset(), False),
        ("unmeasured middle rungs", gaps, "page", frozenset(), False),
        ("boundary values", edge, "page", frozenset(), False),
        ("text ladder", TEXT_SPECS, "text", frozenset(), False),
        ("model without rows of its own", POOLED_SPECS, "page", frozenset(), True),
    ]


def _compare(js, ref, spec_names, label):
    pick, close, safer, q, status, _need = js
    assert (spec_names[pick] if pick >= 0 else None) == ref["pick"], label
    assert close is ref["close"] or close == ref["close"], label
    assert (spec_names[safer] if safer >= 0 else None) == ref["safer"], label
    for i, name in enumerate(spec_names):
        assert bool(q[i]) == ref["qual"][name], (label, name)
        assert status[i] == ref["status"][name], (label, name, status[i], ref["status"][name])


def _cases(specs, modality, blocked, targets, pages, checks, pooled=False):
    cases, refs = [], []
    for t in targets:
        for n in pages:
            for ch in checks:
                cases.append([t, n, ch])
                refs.append(reference_select(specs, t, n, checked=ch, modality=modality, blocked=blocked,
                                             pooled=pooled))
    return cases, refs


@needs_node
def test_js_rule_matches_the_design_on_hand_fixtures(tmp_path):
    items, expected = [], []
    script = None
    for label, specs, modality, blocked, pooled in _fixtures():
        sel = selection_for(specs, 0.97, 500, modality=modality, blocked=blocked, pooled=pooled)
        script_i, data = _chart.extract_rule_and_data(_chart.html(sel))
        script = script or script_i
        assert script_i == script
        cases, refs = _cases(specs, modality, blocked, TARGETS_GRID, PAGES_GRID, (False, True), pooled)
        items.append({"data": data, "cases": cases})
        expected.append((label, specs, modality, cases, refs))
    out = _node(tmp_path, RULE_RUNNER, {"script": script, "items": items})
    checked_cases = 0
    for js_cases, (label, specs, modality, cases, refs) in zip(out, expected):
        names = [sp["name"] for sp in specs]
        for js, case, ref in zip(js_cases, cases, refs):
            _compare(js, ref, names, (label, case))
            for i, sp in enumerate(specs):                       # the needed line, exactly
                want = need_of(sp, case[0], case[1], case[2], modality)
                assert js[5][i] == want, (label, case, sp["name"])
            checked_cases += 1
    assert checked_cases == len(_fixtures()) * len(TARGETS_GRID) * len(PAGES_GRID) * 2


@needs_node
def test_js_rule_matches_the_design_on_random_ladders(tmp_path):
    rng = random.Random(20261021)
    items, expected = [], []
    script = None
    for k in range(120):
        modality = "page" if k % 4 else "text"
        specs = random_specs(rng, modality)
        sel = selection_for(specs, 0.97, 500, modality=modality)
        script_i, data = _chart.extract_rule_and_data(_chart.html(sel))
        script = script or script_i
        targets = sorted(rng.sample(TARGETS_GRID, 9))
        pages = sorted(rng.sample(PAGES_GRID, 6))
        cases, refs = _cases(specs, modality, frozenset(), targets, pages, (False, True))
        items.append({"data": data, "cases": cases})
        expected.append((f"random {k} {modality}", specs, cases, refs))
    out = _node(tmp_path, RULE_RUNNER, {"script": script, "items": items})
    for js_cases, (label, specs, cases, refs) in zip(out, expected):
        names = [sp["name"] for sp in specs]
        for js, case, ref in zip(js_cases, cases, refs):
            _compare(js, ref, names, (label, case))


@needs_node
def test_js_always_has_a_pick_and_falls_back_to_float32(tmp_path):
    specs = [dict(sp) for sp in PAGE_SPECS]
    sel = selection_for(specs, 0.97, 500)
    script, data = _chart.extract_rule_and_data(_chart.html(sel))
    cases = [[0.995, 500, False], [1.0, 500, True], [0.90, 500, False]]
    out = _node(tmp_path, RULE_RUNNER, {"script": script, "items": [{"data": data, "cases": cases}]})[0]
    names = [sp["name"] for sp in specs]
    assert names[out[0][0]] == "float32"            # float16's lowest kept (0.9865) is below 0.995
    assert names[out[1][0]] == "float32"
    assert names[out[2][0]] != "float32"


@needs_node
def test_page_script_redraws_with_the_words_python_uses(tmp_path):
    """Move the controls to the selection's own values: every text the script writes equals the static page."""
    cases = [
        ("page 0.97 500", selection_for(PAGE_SPECS, 0.97, 500)),
        ("page 0.97 2500 checked", selection_for(PAGE_SPECS, 0.97, 2500, checked=True)),
        ("page 0.90 500 checked", selection_for(PAGE_SPECS, 0.90, 500, checked=True)),
        ("page 0.99 500", selection_for(PAGE_SPECS, 0.99, 500)),
        ("page 0.93 500", selection_for(PAGE_SPECS, 0.93, 500)),
        ("page 0.97 25000 checked", selection_for(PAGE_SPECS, 0.97, 25000, checked=True)),
        ("text 0.97 500", selection_for(TEXT_SPECS, 0.97, 500, modality="text")),
        ("pooled 0.97 500", selection_for(POOLED_SPECS, 0.97, 500, model=None, pooled=True)),
        ("pooled 0.95 2500 checked", selection_for(POOLED_SPECS, 0.95, 2500, model=None, pooled=True, checked=True)),
    ]
    for label, sel in cases:
        page = _chart.html(sel)
        script, data = _chart.extract_rule_and_data(page)
        ids = sorted(set(re.findall(r'id="(ov-[^"]+)"', page)))
        t, n = sel.target, sel.n_pages or 500
        out = _node(tmp_path, DOM_RUNNER, {"script": script, "data": data, "ids": ids, "cases": [[t, n]]})[0]
        assert out["sum"] == _chart._summary(sel), label
        assert out["next"].startswith("Next: ov.build(") and f"target={t:.2f}" in out["next"]
        assert out["sub"].startswith(f"collections up to {n:,} pages"), label
        for i, row in enumerate(sel.rows):
            text, klass = out["rows"][i]
            assert text == _chart._row_line(sel, row), (label, row.name, text)
            if klass is not None:
                assert ("ov-yes" in klass.split()) == _chart._filled(sel, row), (label, row.name)
        assert out["ring"].get("visibility") in ("visible", "hidden")


@needs_node
def test_page_script_moves_the_needed_lines_and_the_ring(tmp_path):
    sel = selection_for(PAGE_SPECS, 0.97, 500)
    page = _chart.html(sel)
    script, data = _chart.extract_rule_and_data(page)
    ids = sorted(set(re.findall(r'id="(ov-[^"]+)"', page)))
    out = _node(tmp_path, DOM_RUNNER, {"script": script, "data": data, "ids": ids,
                                       "cases": [[0.97, 500], [0.97, 20000], [0.90, 500], [1.0, 500]]})
    low, big, easy, top = out
    k = [r.name for r in sel.rows].index("hm(0.25) > int8")
    assert float(big["ticks"][k]) < float(low["ticks"][k])      # a larger collection: the needed line is higher
    assert float(easy["ticks"][k]) > float(low["ticks"][k])     # a lower target: lower
    assert "Our benchmarks cannot separate layouts this close to 100%." in top["sum"] or "float32" in top["sum"]
    assert "Targets below 95% are less tested." not in low["sum"]
    assert "Targets below 95% are less tested." in easy["sum"]          # 0.90 is below 0.95
    # the pick moves when the target does
    assert low["sum"].split(",")[0] != easy["sum"].split(",")[0]
    assert float(low["ring"]["cx"]) != float(easy["ring"]["cx"])


# ---------------------------------------------------------------------------------------- against the Python rule


def _levels_or_skip():
    try:
        from optivision.easy import _levels
    except ImportError:
        pytest.skip("optivision.easy._levels is not importable yet; the Python rule is written by the rule agent. "
                    "Run this test once it lands: it compares the page's rule with _levels.select.")
    return _levels


def _catalogue_models() -> list[str | None]:
    from optivision.adapters import MODELS, measured

    ids = set(MODELS) | set(measured.measured_models())
    return [None, *sorted(ids)]


@needs_node
def test_js_rule_matches_python(tmp_path):
    """Every model of the catalogue (and the model-less pooled rule) x targets 0.90/0.95/0.97/0.98 x pages
    500/2,500/20,000, unchecked and checked: the page's pick, qualifying set, close-call flag and safer layout
    equal ``_levels.select``'s."""
    levels = _levels_or_skip()
    targets, pages = (0.90, 0.95, 0.97, 0.98), (500, 2500, 20000)
    items, expected = [], []
    script = None
    for model in _catalogue_models():
        for checked in (False, True):
            base = levels.select(model, 0.97, 500, checked=checked)      # a refusal here is a finding, not a skip
            script_i, data = _chart.extract_rule_and_data(_chart.html(base))
            script = script or script_i
            cases = [[t, n, checked] for t in targets for n in pages]
            py = [levels.select(model, t, n, checked=checked) for t, n, _ in cases]
            items.append({"data": data, "cases": cases})
            expected.append((model, checked, base, cases, py))
    out = _node(tmp_path, RULE_RUNNER, {"script": script, "items": items})
    problems = []
    for js_cases, (model, checked, base, cases, py) in zip(out, expected):
        names = [r.name for r in base.rows]
        for js, case, sel in zip(js_cases, cases, py):
            pick, close, safer, q, _status, _need = js
            want_q = {r.name: r.status in ("pick", "qualifies", "close call") for r in sel.rows}
            got_q = {names[i]: bool(q[i]) for i in range(len(names)) if names[i] in want_q}
            got = (names[pick] if pick >= 0 else None, bool(close), names[safer] if safer >= 0 else None)
            want = (sel.pick.name, bool(sel.close_call), sel.safer.name if sel.safer else None)
            if got != want or got_q != want_q:
                problems.append(f"{model} checked={checked} target={case[0]} pages={case[1]}: page rule {got}, "
                                f"_levels {want}; qualifying differs on "
                                f"{sorted(k for k in want_q if want_q[k] != got_q.get(k))}")
    assert not problems, f"{len(problems)} disagreements, first ones:\n" + "\n".join(problems[:15])


@needs_node
def test_page_script_redraws_selections_of_the_rule_module(tmp_path):
    """The same check on real selections of ``_levels``: statuses, reasons and verdict lines."""
    levels = _levels_or_skip()
    cases = [(None, 0.97, 500, False, None), (None, 0.95, 2500, True, None),
             ("vidore/colqwen2-v1.0-hf", 0.97, 500, False, None), ("vidore/colqwen2-v1.0-hf", 0.98, 20000, True, None),
             ("vidore/colqwen2-v1.0-hf", 0.95, 25000, True, None), ("vidore/colqwen2-v1.0-hf", 0.97, 500, False, "int4(mean)"),
             ("vidore/colqwen2-v1.0-hf", 0.90, 2500, True, None)]
    for model, t, n, checked, layout in cases:
        sel = levels.select(model, t, n, checked=checked, layout=layout)
        page = _chart.html(sel)
        assert banned_hits(page) == [] and non_ascii(page) == [], (model, t, n)
        ET.fromstring(_chart.svg(sel))
        script, data = _chart.extract_rule_and_data(page)
        ids = sorted(set(re.findall(r'id="(ov-[^"]+)"', page)))
        out = _node(tmp_path, DOM_RUNNER, {"script": script, "data": data, "ids": ids, "cases": [[t, n]]})[0]
        if layout is None:
            assert out["sum"] == _chart._summary(sel), (model, t, n)
        for i, row in enumerate(sel.rows):
            text, klass = out["rows"][i]
            assert text == _chart._row_line(sel, row), (model, t, n, row.name, text)
            if klass is not None:
                assert ("ov-yes" in klass.split()) == _chart._filled(sel, row), (model, t, n, row.name)


def test_the_evidence_source_is_worded_by_the_rule_module():
    levels = _levels_or_skip()
    sel = levels.select("vidore/colqwen2-v1.0-hf", 0.97, 500)
    assert _chart._source(sel) == "ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages"
    assert _chart._evidence(sel).tag == "[our benchmarks: ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages]"
    pooled = levels.select(None, 0.97, 500)
    assert _chart._source(pooled).startswith("worst of the ")
    rule = _chart._rule_data()
    assert rule["size_coefficients"] == levels.data()["rule"]["size_coefficients"]
