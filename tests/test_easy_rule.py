"""The query-free selection rule of easy mode (FINAL_DESIGN 5.1): its data file, its golden
fixture, the FINAL_DESIGN 5.1.3 tables it reproduces, each rule step, the resolver, and the
light import."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from optivision.compose import Pipeline
from optivision.easy import _ledger, _levels, _resolve
from optivision.easy._types import AmbiguousModel, LayoutRow, Selection, UnknownModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BUILDER = ROOT / "scripts" / "make_easy_levels.py"
GOLDEN = ROOT / "tests" / "data" / "easy_rule_golden.json"
HAS_REPORTS = BUILDER.exists() and (ROOT / "reports" / "universal" / "followup").exists()
needs_reports = pytest.mark.skipif(not HAS_REPORTS, reason="needs the repository's reports (not in a wheel)")

PAGE = [m for m, e in _levels.data()["models"].items() if e["modality"] == "page"]
TEXT = [m for m, e in _levels.data()["models"].items() if e["modality"] == "text"]
PENDING = {m for m, e in _levels.data()["models"].items() if e.get("pending_b0")}


def _builder():
    spec = importlib.util.spec_from_file_location("make_easy_levels", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}


def _ratio(x: float) -> str:  # FINAL_DESIGN prints ratios to the nearest tenth
    return f"{x:.1f}x"


# ------------------------------------------------------------------------------ the data file


@needs_reports
def test_the_generator_check_passes_on_the_committed_files():
    done = subprocess.run([sys.executable, str(BUILDER), "--check"], capture_output=True, text=True, env=_env(),
                          check=False, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr


@needs_reports
def test_the_generator_check_exits_1_on_a_stale_file(tmp_path, monkeypatch):
    builder = _builder()
    stale = tmp_path / "easy_levels.json"
    stale.write_text((ROOT / "src" / "optivision" / "presets" / "easy_levels.json").read_text(encoding="utf-8")
                     .replace('"close_call":0.005', '"close_call":0.004'), encoding="utf-8")
    monkeypatch.setattr(builder, "OUT", stale)
    assert builder.main(["--check"]) == 1
    stale.write_text("", encoding="utf-8")
    assert builder.main(["--check"]) == 1


def test_the_data_file_records_a_sha256_for_every_source():
    data = _levels.data()
    catalogue = json.loads((SRC / "optivision" / "presets" / "measured.json").read_text(encoding="utf-8"))
    assert "src/optivision/presets/measured.json" in data["sources"]
    assert all(len(v) == 64 and int(v, 16) >= 0 for v in data["sources"].values())
    for entry in data["models"].values():
        for info in entry["datasets"].values():
            # a follow-up report is hashed here; a catalogue row's report is hashed by measured.json
            assert set(info["sources"]) <= set(data["sources"]) | set(catalogue["sources"]), info["sources"]
    assert len(data["models"]) == 55 and len(PAGE) == 40 and len(TEXT) == 15


def test_follow_up_datasets_add_evidence_but_never_set_sizes():
    entry = _levels.data()["models"]["tencent/EVIE-4.5B"]
    roles = {d: v["role"] for d, v in entry["datasets"].items()}
    assert roles["vidore/tabfquad_test_subsampled"] == roles["vidore/arxivqa_test_subsampled"] == "follow-up"
    row = _levels.row("tencent/EVIE-4.5B", "int4(mean)", 0.97)
    tab = entry["rungs"]["int4(mean)"]["vidore/tabfquad_test_subsampled"]
    assert row.lowest_kept == tab["low"] == 0.9291  # FINAL_DESIGN 5.1.4: the worst page int4(mean) unit
    cat = [entry["rungs"]["int4(mean)"][d]["bytes_per_page"] for d in ("vidore/docvqa_test_subsampled",
                                                                         "vidore/infovqa_test_subsampled")]
    assert row.bytes_per_page == (min(cat), max(cat))


def test_scale_units_are_not_evidence():
    for entry in _levels.data()["models"].values():
        for info in entry["datasets"].values():
            assert not any("_scale" in s or "plus" in Path(s).stem for s in info["sources"])


def test_b0_models_are_pending_and_their_qwen2_grid_legacy_merges_are_excluded():
    data = _levels.data()["models"]
    assert PENDING <= {"vidore/colqwen2.5-v0.2", "vidore/colqwen2-v1.0-merged", "vidore/colpali-v1.3-merged"}
    assert set(_golden()["pending_b0"]) == PENDING
    for model in PENDING & {"vidore/colqwen2.5-v0.2", "vidore/colqwen2-v1.0-merged"}:
        legacy = [d for d, v in data[model]["datasets"].items() if v.get("legacy")]
        assert legacy
        for rung in _levels.LADDER["page"]:
            if rung.merged:
                assert not set(data[model]["rungs"].get(rung.name, {})) & set(legacy), (model, rung.name)
    if "vidore/colpali-v1.3-merged" in PENDING:  # its legacy merge matches the campaign's (263.0 vectors)
        rows = data["vidore/colpali-v1.3-merged"]["rungs"]["hierarchical_merge(0.25) > int8(per_vector)"]
        assert rows["vidore/docvqa_test_subsampled"]["vectors_per_page"] == 263.0
    # campaign-format replacements already in place: ColQwen2.5 v0.2 DocVQA (f10), answerai SciFact (f3b)
    q = data["vidore/colqwen2.5-v0.2"]["datasets"]["vidore/docvqa_test_subsampled"]
    assert q["sources"][0].startswith("reports/universal/followup/f10_") and "replaces" in q
    a = data["answerdotai/answerai-colbert-small-v1"]["datasets"]["BeIR/scifact"]
    assert a["sources"][0].startswith("reports/universal/followup/f3b_") and not a.get("legacy")


@needs_reports
def test_b0_reports_replace_the_legacy_rows_when_they_land(tmp_path, monkeypatch):
    """A campaign-format report under followup/f14a_.../<org>--<name>/wide/ replaces the legacy
    entry of that dataset; with both datasets replaced the model is no longer pending."""
    builder = _builder()
    fake = tmp_path / "followup"
    src = ROOT / "reports" / "universal" / "campaign" / "vidore--colqwen2-v1.0-hf" / "wide"
    dst = fake / "f14a_easy_b0_colqwen" / "vidore--colqwen2-v1.0-merged" / "wide"
    dst.mkdir(parents=True)
    for tail in ("docvqa_test_subsampled", "infovqa_test_subsampled"):
        (dst / f"colqwen2-v1.0-merged_{tail}.json").write_bytes((src / f"colqwen2-v1.0-hf_{tail}.json").read_bytes())
    monkeypatch.setattr(builder, "FOLLOWUP", fake)
    monkeypatch.setattr(builder, "ROOT", ROOT)
    levels = builder.build()
    entry = levels["models"]["vidore/colqwen2-v1.0-merged"]
    assert entry["pending_b0"] is False and "legacy_datasets" not in entry
    catalogue = [v for v in entry["datasets"].values() if v["role"] == "catalogue"]
    assert len(catalogue) == 2
    for info in catalogue:
        assert info["sources"][0].startswith("reports/universal/followup/f14a_easy_b0_colqwen/")
        assert not info.get("legacy") and info["replaces"].startswith("reports/universal/frontier/")
    assert "hierarchical_merge(0.25) > int8(per_vector)" in entry["rungs"]
    assert levels["models"]["vidore/colqwen2.5-v0.2"]["pending_b0"] is True  # its InfoVQA is still legacy


# ------------------------------------------------------------------------------ the ladder


def test_every_rung_spec_round_trips_and_equals_the_catalogue_rows():
    catalogue = json.loads((SRC / "optivision" / "presets" / "measured.json").read_text(encoding="utf-8"))["models"]
    seen = set()
    for modality, rungs in _levels.LADDER.items():
        assert [r.name for r in rungs[:2]] == ["float32", "float16"]
        for rung in rungs:
            assert Pipeline.from_dict(rung.spec).to_dict() == rung.spec, rung.name
            for model, datasets in catalogue.items():
                for entry in datasets.values():
                    for row in entry["rows"]:
                        if row["label"] == rung.name:
                            assert row["pipeline"] == rung.spec, (model, rung.name)
                            seen.add(rung.name)
    assert seen == {r.name for rungs in _levels.LADDER.values() for r in rungs}
    page = _levels.LADDER["page"]
    assert [r.chain for r in page[2:]] == ["8-bit"] * 3 + ["centred 4-bit"] * 3 + ["1-bit"] * 3
    assert [r.name for r in _levels.LADDER["text"]] == ["float32", "float16", "int8(per_vector)", "int4(mean)"]


def test_hand_pick_names():
    assert _levels.find_rung("centred 4-bit").name == "int4(mean)"
    assert _levels.find_rung("int4(mean)").name == "int4(mean)"
    assert _levels.find_rung("hm(0.25) > int8").name == "hierarchical_merge(0.25) > int8(per_vector)"
    assert _levels.find_rung("merged (1 in 3 vectors) + 1-bit").name == "hierarchical_merge(0.33) > binary"
    assert _levels.find_rung("8-bit").name == _levels.find_rung("int8").name == "int8(per_vector)"
    with pytest.raises(ValueError, match="no layout"):
        _levels.find_rung("project(64) > int8")
    with pytest.raises(ValueError):
        _levels.find_rung("1-bit", "text")


# ------------------------------------------------------------------------------ margins


def test_size_term_and_needed_values():
    assert _levels.s_of_n(None) == _levels.s_of_n(500) == _levels.s_of_n(100) == 0.0
    assert round(_levels.s_of_n(1000), 2) == 0.43 and _levels.s_of_n(2500) == pytest.approx(1.0)
    assert round(_levels.s_of_n(20000), 2) == 2.29
    rung = {r.name: r for r in _levels.LADDER["page"]}
    t = 0.97
    assert _levels.need(t, rung["float16"], 20000) == t
    assert _levels.need(t, rung["int8(per_vector)"], 2500) == pytest.approx(t + 0.005)
    assert _levels.need(t, rung["hierarchical_merge(0.25) > int4(mean)"], 2500) == pytest.approx(t + 0.0175)
    assert _levels.need(t, rung["binary"], 2500) == pytest.approx(t + 0.0175 + 0.025)
    assert _levels.need(t, rung["hierarchical_merge(0.33) > binary"], 2500) == pytest.approx(t + 0.04 + 0.025)
    assert _levels.need(t, rung["binary"], 500) == pytest.approx(t + 0.025)
    assert _levels.need(t, rung["int4(mean)"], 20000, modality="text") == pytest.approx(t + 0.01)


def test_margins_are_read_from_the_ledger(monkeypatch):
    real = _ledger.feature

    def fake(name):
        entry = real(name)
        if name == "margins":
            entry.update(one_bit_margin=0.0, size_coefficient_scale=0.5, extra_compressed=0.01,
                         unchecked_merge_penalty=0.01, codecs_only_above=1000, lowest_target=0.95)
        return entry

    monkeypatch.setattr(_ledger, "feature", fake)
    rung = {r.name: r for r in _levels.LADDER["page"]}
    assert _levels.need(0.97, rung["binary"], 500, checked=True) == pytest.approx(0.98)
    assert _levels.need(0.97, rung["hierarchical_merge(0.25) > int8(per_vector)"], 2500) == \
        pytest.approx(0.97 + 0.01 + 0.5 * 0.0175 + 0.01)
    with pytest.raises(ValueError, match="between 0.95"):
        _levels.select("vidore/colqwen2-v1.0-hf", 0.94)
    sel = _levels.select("vidore/colqwen2-v1.0-hf", 0.97, 1500)
    assert all(r.status == "outside size range" for r in sel.rows if r.merged or r.one_bit)


def test_target_range():
    for bad in (0.89, 1.01, "high", float("nan")):
        with pytest.raises(ValueError):
            _levels.select("vidore/colqwen2-v1.0-hf", bad)
    assert _levels.select("vidore/colqwen2-v1.0-hf", 1.0).pick.name in ("float16", "float32")
    assert _levels.select("vidore/colqwen2-v1.0-hf", 0.9).target == 0.9
    with pytest.raises(ValueError):
        _levels.select("vidore/colqwen2-v1.0-hf", 0.97, 0)


# ------------------------------------------------------------------------------ the rule's steps


def _by_name(sel: Selection) -> dict[str, LayoutRow]:
    return {r.name: r for r in sel.rows}


def test_selection_shape_and_close_call():
    sel = _levels.select("vidore/colqwen2-v1.0-hf", 0.97)
    assert isinstance(sel, Selection) and sel.basis == "own rows" and sel.modality == "page"
    assert [r.name for r in sel.rows] == [r.name for r in _levels.LADDER["page"]]
    assert sel.pick.name == "hierarchical_merge(0.25) > int8(per_vector)" and sel.pick.status == "pick"
    assert sel.close_call and 0 <= sel.pick.slack < 0.005 and sel.safer.name == "int4(mean)"
    assert sel.safer.slack >= 0.005
    def ints(pair):
        return tuple(round(x) for x in pair)

    # FINAL_DESIGN 2 (a), sources: bytes per page on InfoVQA / DocVQA and the lows
    assert sel.pick.lowest_kept == 0.9728 and ints(sel.pick.bytes_per_page) == (23851, 24357)
    assert ints(sel.safer.bytes_per_page) == (48344, 49389) and sel.safer.lowest_kept == 0.9802
    assert ints(_by_name(sel)["float16"].bytes_per_page) == (187513, 191567)
    assert sum(r.status == "pick" for r in sel.rows) == 1
    # the fixed sequence of the practice check: the pick, safer qualifying layouts by falling ratio, float16
    seq = [r.name for r in sel.sequence]
    assert seq[0] == sel.pick.name and seq[-1] == "float16"
    ratios = [r.ratio_f16 for r in sel.sequence]
    assert ratios == sorted(ratios, reverse=True)
    assert set(seq[1:-1]) == {r.name for r in sel.rows if r.status in ("qualifies", "close call")
                              and r.chain != "float" and r.ratio_f16 < sel.pick.ratio_f16}


def test_chain_rule_stops_a_family_at_its_first_failure():
    # EVIE-4.5B on TabFQuAD: merge(0.25) > binary kept 0.972, its lighter 1-bit rungs 0.961 and 0.955
    sel = _levels.select("tencent/EVIE-4.5B", 0.93, checked=True)
    rows = _by_name(sel)
    assert rows["binary"].status == "pick"
    assert rows["hierarchical_merge(0.33) > binary"].status == "below target"
    assert rows["hierarchical_merge(0.25) > binary"].status == "family stopped"
    assert rows["hierarchical_merge(0.25) > binary"].slack > 0  # it clears on its own; the chain refuses it
    for sel in (_levels.select(m, t, n, checked=c) for m in PAGE[:12] for t in (0.97, 0.9) for n in (500, 2500)
                for c in (True, False)):
        for chain in ("8-bit", "centred 4-bit", "1-bit"):
            ok = [r.status in ("pick", "qualifies", "close call") for r in sel.rows if r.chain == chain]
            assert ok == sorted(ok, reverse=True), (sel.model, chain)  # qualifying rungs form a prefix


def test_allowed_sets():
    unchecked = _levels.select("vidore/colpali-v1.3", 0.95)
    assert all(r.status == "needs practice questions" for r in unchecked.rows if r.one_bit)
    assert not unchecked.pick.one_bit
    big = _levels.select("vidore/colpali-v1.3", 0.95, 20001, checked=True)
    assert all(r.status == "outside size range" for r in big.rows if r.merged or r.one_bit)
    assert big.pick.name in ("int8(per_vector)", "int4(mean)", "float16")
    at = _levels.select("vidore/colpali-v1.3", 0.90, 20000, checked=True)  # the limit itself still allows them
    assert not any(r.status == "outside size range" for r in at.rows) and at.pick.merged
    text = _levels.select("colbert-ir/colbertv2.0", 0.97, 15184)
    assert _by_name(text)["int4(mean)"].status == "outside size range" and text.pick.name == "int8(per_vector)"
    assert _levels.select("colbert-ir/colbertv2.0", 0.97, 15183).pick.name == "int4(mean)"


def test_shape_gate_removes_merged_rungs_only_below_seventy_percent():
    entry = _levels.data()["models"]["vidore/colqwen2-v1.0-hf"]
    floor = 0.7 * min(v["vectors_per_page"] for v in entry["datasets"].values())
    low = _levels.select("vidore/colqwen2-v1.0-hf", 0.97, vectors_per_page=floor - 1, checked=True)
    assert all(r.status == "not allowed" for r in low.rows if r.merged)
    assert low.pick.name == "int4(mean)" and any("fewer vectors" in n for n in low.notes)
    unchecked = _levels.select("vidore/colqwen2-v1.0-hf", 0.97, vectors_per_page=floor - 1)
    assert {r.status for r in unchecked.rows if r.merged} == {"not allowed", "needs practice questions"}
    ok = _levels.select("vidore/colqwen2-v1.0-hf", 0.97, vectors_per_page=floor + 1)
    assert ok.pick == _levels.select("vidore/colqwen2-v1.0-hf", 0.97).pick


def test_float16_and_float32_floor():
    # nemotron-4b's float16 lowest kept is 0.9865 (TabFQuAD): float16 up to there, float32 above
    assert _levels.row("nvidia/nemotron-colembed-vl-4b-v2", "float16").lowest_kept == 0.9865
    sel = _levels.select("nvidia/nemotron-colembed-vl-4b-v2", 0.99)
    assert _by_name(sel)["float16"].status == "below target"
    assert sel.sequence[-1].name == "float32"
    lateon = _levels.select("lightonai/LateOn", 0.97)
    assert lateon.pick.name == "float16" and [r.name for r in lateon.sequence] == ["float16"]
    one = _levels.select("vidore/colqwen2-v1.0-hf", 1.0)
    assert one.sequence[-1].name in ("float16", "float32")
    # FINAL_DESIGN 5.6: above 0.99, float16 if its lowest kept clears T, else float32 (8-bit kept 0.9971 here)
    close = _levels.select("vidore/colqwen2-v1.0-hf", 0.995)
    assert close.pick.name == "float16" and _by_name(close)["int8(per_vector)"].status == "not allowed"
    assert _by_name(close)["int8(per_vector)"].reason == _levels.TOO_CLOSE
    assert _levels.select("vidore/colqwen2-v1.0-hf", 0.99).pick.name == "int8(per_vector)"


def test_hand_pick_keeps_every_status():
    rule = _levels.select("vidore/colqwen2-v1.0-hf", 0.97)
    hand = _levels.select("vidore/colqwen2-v1.0-hf", 0.97, layout="centred 4-bit")
    assert hand.hand_picked and not rule.hand_picked
    assert hand.pick.name == "int4(mean)" and hand.pick.reason.startswith("chosen by you")
    assert [r.status for r in hand.rows] == [r.status for r in rule.rows]
    assert hand.sequence[0].name == "int4(mean)" and hand.sequence[-1].name == "float16"
    below = _levels.select("vidore/colqwen2-v1.0-hf", 0.97, layout="hierarchical_merge(0.25) > int4(mean)")
    assert below.pick.status == "below target" and not below.close_call
    assert below.sequence[1].name == rule.pick.name  # the rule's pick is the next, safer layout


def test_unmeasured_models_use_the_pooled_minimum():
    """FINAL_DESIGN 5.1.4."""
    pooled = _levels.data()["pooled"]
    page, text = pooled["page"]["rungs"], pooled["text"]["rungs"]
    assert (page["float16"]["low"], page["int8(per_vector)"]["low"], page["int4(mean)"]["low"]) == (0.9865, 0.9783, 0.9291)
    assert page["int8(per_vector)"]["worst_model"] == "nvidia/nemotron-colembed-vl-8b-v2"
    assert (page["int4(mean)"]["worst_model"], page["int4(mean)"]["n_models"]) == ("tencent/EVIE-4.5B", 37)
    assert (text["int8(per_vector)"]["low"], text["int4(mean)"]["low"]) == (0.9655, 0.8965)
    assert text["int8(per_vector)"]["worst_model"] == text["int4(mean)"]["worst_model"] == "lightonai/LateOn"

    def pick(t, n=None, checked=False, modality="page"):
        return _levels.select(None, t, n, checked=checked, modality=modality).pick.name

    assert pick(0.98) == "float16" and pick(0.99) == "float32"
    assert pick(0.97) == pick(0.95) == "int8(per_vector)" and pick(0.97, 7500) == "float16"
    assert pick(0.90, checked=True) == "int4(mean)" and pick(0.90) == "int8(per_vector)"
    assert pick(0.96, modality="text") == "float16" and pick(0.95, modality="text") == "int8(per_vector)"
    sel = _levels.select("vidore/colSmol-256M", 0.97)  # registered, measured on generated corpora only
    assert sel.basis.startswith("unmeasured: pooled minimum over 40 page models")
    assert all(r.status == "not measured" for r in sel.rows if r.merged or r.one_bit)


# ------------------------------------------------------------------------------ the golden fixture


def test_the_golden_fixture_matches_the_rule():
    golden = _golden()
    assert golden["levels_sha256"]
    for model, picks in golden["picks"].items():
        for key, want in picks.items():
            t, n, path = key.split("|")
            s = _levels.select(model, float(t), int(n), checked=path == "c")
            got = [s.pick.name, round(s.pick.ratio_f16, 4), None if s.pick.slack is None else round(s.pick.slack, 4),
                   s.close_call, None if s.safer is None else s.safer.name]
            assert got == want, (model, key)
    for modality, picks in golden["pooled"].items():
        for key, want in picks.items():
            t, n, path = key.split("|")
            s = _levels.select(None, float(t), int(n), checked=path == "c", modality=modality)
            assert [s.pick.name, s.close_call] == [want[0], want[3]], (modality, key)


#: FINAL_DESIGN 5.1.3, golden picks at 0.97 (pages: <= 500 / 2,500 / 20,000), ratios against float16
FD_GOLDEN = {
    "vidore/colqwen2-v1.0-hf": [("hierarchical_merge(0.25) > int8(per_vector)", "7.9x"), ("int4(mean)", "3.9x"),
                                ("int8(per_vector)", "2.0x")],
    "vidore/colpali-v1.3": [("hierarchical_merge(0.33) > int8(per_vector)", "5.9x"), ("int4(mean)", "3.9x"),
                            ("int4(mean)", "3.9x")],
    "vidore/colSmol-500M": [("hierarchical_merge(0.25) > int8(per_vector)", "5.5x"), ("int4(mean)", "3.9x"),
                            ("int4(mean)", "3.9x")],
    "nvidia/nemotron-colembed-vl-4b-v2": [("hierarchical_merge(0.33) > int8(per_vector)", "6.0x"),
                                          ("int4(mean)", "4.0x"), ("int4(mean)", "4.0x")],
    "tencent/EVIE-4.5B": [("int8(per_vector)", "2.0x")] * 3,
}
#: the B0 models' row in that table reads "int8 2.0x ... recomputed from B0": their picks wait for B0
FD_B0 = ("vidore/colqwen2.5-v0.2", "vidore/colqwen2-v1.0-merged", "vidore/colpali-v1.3-merged")
FD_TEXT = {"colbert-ir/colbertv2.0": ("0.97", "int4(mean)", "3.9x"), "lightonai/LateOn": ("0.97", "float16", "1.0x"),
           "lightonai/LateOn@0.95": ("0.95", "int8(per_vector)", "2.0x")}


def test_fd_golden_picks():
    picks = _golden()["picks"]
    for model, want in FD_GOLDEN.items():
        got = [(picks[model][f"0.97|{n}|c"][0], _ratio(picks[model][f"0.97|{n}|c"][1])) for n in (500, 2500, 20000)]
        assert got == want, model
    for model in FD_B0:  # in the fixture, and marked pending while their legacy rows remain
        assert model in picks
        assert (model in _golden()["pending_b0"]) == bool(_levels.data()["models"][model].get("legacy_datasets"))
    # before B0 the two Qwen2-grid models get 8-bit everywhere, as the table says
    for model in set(FD_B0[:2]) & PENDING:
        assert {picks[model][f"0.97|{n}|c"][0] for n in (500, 2500, 20000)} == {"int8(per_vector)"}
    assert picks["vidore/colqwen2-v1.0-hf"]["0.97|500|c"][3] is True  # "close call at <= 500"
    assert not any(picks["vidore/colqwen2-v1.0-hf"][f"0.97|{n}|c"][3] for n in (2500, 20000))
    for key, (t, name, ratio) in FD_TEXT.items():
        model = key.split("@")[0]
        got = picks[model][f"{t}|500|c"]
        assert (got[0], _ratio(got[1])) == (name, ratio), key


#: FINAL_DESIGN 5.1.3, median ratio against float16 over the 40 page models, checked path, by pages;
#: the last column is the unchecked path at <= 500 pages
FD_MEDIANS = {
    "0.98": (["4.0", "3.9", "3.9", "2.9", "2.0"], "4.0"),
    "0.97": (["11.5", "5.6", "3.9", "3.9", "3.9"], "11.5"),
    "0.95": (["15.5", "15.2", "11.7", "5.8", "3.9"], "15.2"),
    "0.90": (["62.7", "32.2", "16.0", "16.0", "15.5"], "15.5"),
}


def test_fd_median_table():
    golden = _golden()
    summary = golden["summary"]
    assert golden["page_models"] == 40
    for t, (checked, unchecked) in FD_MEDIANS.items():
        got = [f"{summary[f'{t}|{n}|c']['median']:.1f}" for n in (500, 1000, 2500, 5244, 20000)]
        assert got == checked, t
        assert f"{summary[f'{t}|500|u']['median']:.1f}" == unchecked, t
    assert [f"{summary['0.97|500|c']['iqr'][i]:.1f}" for i in (0, 1)][1] == "15.2"
    assert [f"{summary['0.95|500|c']['iqr'][i]:.1f}" for i in (0, 1)] == ["14.8", "15.8"]
    assert summary["0.95|500|c"]["one_bit"] == 7  # "1-bit picks at 0.95: 7 of 40"
    # FINAL_DESIGN 1: without the practice check (unchecked) 4.0x 11.5x 15.2x 15.5x
    assert [f"{summary[f'{t}|500|u']['median']:.1f}" for t in ("0.98", "0.97", "0.95", "0.90")] == \
        ["4.0", "11.5", "15.2", "15.5"]


def test_the_reference_exclusion_reproduces_the_fd_iqr_and_close_calls(monkeypatch):
    """FINAL_DESIGN 5.1.3 was computed with every legacy merge row excluded; its text (5.1.2 step 1)
    narrows that to the Qwen2-grid models, so ColPali v1.3 merged keeps its matching merge rows. That
    is the only difference: excluding them again gives the published IQR 5.6-15.2 and 10 close calls."""
    if "vidore/colpali-v1.3-merged" not in PENDING:
        pytest.skip("B0 rows of colpali-v1.3-merged have landed")
    data = copy.deepcopy(_levels.data())
    rungs = data["models"]["vidore/colpali-v1.3-merged"]["rungs"]
    for name in [n for n in rungs if "hierarchical_merge" in n]:
        del rungs[name]
    monkeypatch.setattr(_levels, "data", lambda: data)
    sels = [_levels.select(m, 0.97, 500, checked=True) for m in PAGE]
    xs = sorted(s.pick.ratio_f16 for s in sels)
    q = statistics.quantiles(xs, n=4)
    assert (f"{q[0]:.1f}", f"{q[2]:.1f}", sum(s.close_call for s in sels)) == ("5.6", "15.2", 10)
    sels = [_levels.select(m, 0.95, 2500, checked=True) for m in PAGE]
    assert f"{statistics.quantiles(sorted(s.pick.ratio_f16 for s in sels), n=4)[0]:.1f}" == "6.1"


# ------------------------------------------------------------------------------ the resolver


def test_resolve_ids():
    assert _resolve.resolve(None) is None
    assert _resolve.resolve("vidore/colqwen2-v1.0-hf").model_id == "vidore/colqwen2-v1.0-hf"
    assert _resolve.resolve("VIDORE/ColQwen2-v1.0-HF").model_id == "vidore/colqwen2-v1.0-hf"
    assert _resolve.resolve("colqwen2-v1.0-hf").model_id == "vidore/colqwen2-v1.0-hf"
    with pytest.raises(AmbiguousModel) as err:
        _resolve.resolve("colbertv2.0")
    text = str(err.value)
    assert "colbert-ir/colbertv2.0" in text and "lightonai/colbertv2.0" in text
    assert "sentence-transformers" in text and "licence not recorded" in text and "licence mit" in text
    with pytest.raises(UnknownModel) as err:
        _resolve.resolve("vidore/colqwen2-v9")
    assert isinstance(err.value, KeyError) and str(err.value).count("vidore/colqwen2") >= 3
    assert len(_resolve.closest("colpaly-v1.3")) == 3 and "vidore/colpali-v1.3" in _resolve.closest("colpaly-v1.3")
    with pytest.raises(TypeError):
        _resolve.resolve(3)


def test_rows_model():
    assert _resolve.rows_model("vidore/colqwen2-v1.0-hf") == ("vidore/colqwen2-v1.0-hf", "own rows")
    assert _resolve.rows_model("vidore/colqwen2-v1.0") == ("vidore/colqwen2-v1.0-merged",
                                                           "measured via vidore/colqwen2-v1.0-merged")
    assert _resolve.rows_model("vidore/colSmol-256M") == (None, "unmeasured")
    via = _levels.select("vidore/colqwen2-v1.0", 0.97)
    assert via.model == "vidore/colqwen2-v1.0" and via.basis == "measured via vidore/colqwen2-v1.0-merged"


# ------------------------------------------------------------------------------ light import


def test_easy_rule_modules_import_without_heavy_dependencies():
    # PIL is not blocked: the core package (optivision.pruning) imports it already
    blocked = ["torch", "scipy", "transformers", "sentence_transformers", "colpali_engine", "pypdfium2", "peft",
               "accelerate", "huggingface_hub", "datasets", "qdrant_client", "reportlab"]
    code = (
        "import json, sys\n"
        "for name in json.loads(sys.argv[1]):\n"
        "    sys.modules[name] = None\n"
        "import optivision.easy as ov\n"
        "from optivision.easy import _levels, _resolve, _options\n"
        "s = _levels.select('vidore/colqwen2-v1.0-hf', 0.97)\n"
        "o = ov.options('vidore/colqwen2-v1.0-hf', target=0.95)\n"
        "assert str(o).splitlines()[-1].startswith('Next:')\n"
        "print(json.dumps({'pick': s.pick.name, 'loaded': sorted(n for n in json.loads(sys.argv[1]) "
        "if sys.modules.get(n) is not None)}))\n"
    )
    done = subprocess.run([sys.executable, "-c", code, json.dumps(blocked)], capture_output=True, text=True,
                          env=_env(), check=False, timeout=300)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report == {"pick": "hierarchical_merge(0.25) > int8(per_vector)", "loaded": []}
