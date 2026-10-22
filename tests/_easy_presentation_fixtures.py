"""Hand-built ladders and a literal reading of FINAL_DESIGN 5.1.2 for the chart, report and glossary tests.

``reference_select`` is written from the design text and does not import ``_chart``; the page's
script is compared against it, and (when it exists) against ``_levels.select``. The numbers are
shaped like the colqwen2 example of the design (FINAL_DESIGN 2(a)); they are fixtures for the code,
not measurements.
"""

from __future__ import annotations

import math
import random
from typing import Any

from optivision.easy._types import LayoutRow, Selection

DOCVQA = "ViDoRe V1 DocVQA 500 pages"
INFOVQA = "InfoVQA 500 pages"

#: size class by (one_bit, merged): the drop rounded up to 0.0025 (FINAL_DESIGN 5.1.2)
SIZE = {(False, False): 0.005, (False, True): 0.0175, (True, False): 0.0175, (True, True): 0.04}
ONE_BIT_MARGIN = 0.025
CODECS_ONLY_ABOVE = 20000
TEXT_MARGIN = 0.01
TEXT_MAX = 15183
CLOSE = 0.005

PAGE_SPECS: list[dict[str, Any]] = [
    # name, plain, chain, one_bit, merged, bytes (smallest, largest), ratio vs float16, lowest kept, worst point
    {"name": "float32", "plain": "float32 (full precision)", "chain": "float", "one_bit": False, "merged": False,
         "bytes": (383134.0, 375026.0), "r16": 0.5, "low": None, "worst": None},
    {"name": "float16", "plain": "float16", "chain": "float", "one_bit": False, "merged": False,
         "bytes": (187513.0, 191567.0), "r16": 1.0, "low": 0.9865, "worst": 0.9961},
    {"name": "int8(per_vector)", "plain": "8-bit", "chain": "8-bit", "one_bit": False, "merged": False,
         "bytes": (94000.0, 96000.0), "r16": 1.99, "low": 0.9783, "worst": 0.9902},
    {"name": "hm(0.33) > int8", "plain": "merged (1 in 3 vectors) + 8-bit", "chain": "8-bit", "one_bit": False, "merged": True,
         "bytes": (31700.0, 32300.0), "r16": 5.93, "low": 0.9811, "worst": 0.9890},
    {"name": "hm(0.25) > int8", "plain": "merged (1 in 4 vectors) + 8-bit", "chain": "8-bit", "one_bit": False, "merged": True,
         "bytes": (23851.0, 24357.0), "r16": 7.86, "low": 0.9728, "worst": 0.9850},
    {"name": "int4(mean)", "plain": "centred 4-bit", "chain": "centred 4-bit", "one_bit": False, "merged": False,
         "bytes": (48344.0, 49389.0), "r16": 3.88, "low": 0.9802, "worst": 0.9921},
    {"name": "hm(0.33) > int4(mean)", "plain": "merged (1 in 3 vectors) + centred 4-bit", "chain": "centred 4-bit",
         "one_bit": False, "merged": True, "bytes": (16300.0, 16600.0), "r16": 11.54, "low": 0.9689, "worst": 0.9801},
    {"name": "hm(0.25) > int4(mean)", "plain": "merged (1 in 4 vectors) + centred 4-bit", "chain": "centred 4-bit",
         "one_bit": False, "merged": True, "bytes": (12200.0, 12500.0), "r16": 15.32, "low": 0.9501, "worst": 0.9710},
    {"name": "binary", "plain": "1-bit", "chain": "1-bit", "one_bit": True, "merged": False,
         "bytes": (11700.0, 12000.0), "r16": 15.96, "low": 0.9610, "worst": 0.9780},
    {"name": "hm(0.33) > binary", "plain": "merged (1 in 3 vectors) + 1-bit", "chain": "1-bit", "one_bit": True, "merged": True,
         "bytes": (3900.0, 4000.0), "r16": 47.9, "low": 0.9550, "worst": 0.9690},
    {"name": "hm(0.25) > binary", "plain": "merged (1 in 4 vectors) + 1-bit", "chain": "1-bit", "one_bit": True, "merged": True,
         "bytes": (2950.0, 3050.0), "r16": 62.6, "low": 0.9720, "worst": 0.9801},
]

TEXT_SPECS: list[dict[str, Any]] = [
    {"name": "float32", "plain": "float32 (full precision)", "chain": "float", "one_bit": False, "merged": False,
         "bytes": (1200.0, 1200.0), "r16": 0.5, "low": None, "worst": None},
    {"name": "float16", "plain": "float16", "chain": "float", "one_bit": False, "merged": False,
         "bytes": (600.0, 600.0), "r16": 1.0, "low": 0.9990, "worst": 1.0010},
    {"name": "int8(per_vector)", "plain": "8-bit", "chain": "8-bit", "one_bit": False, "merged": False,
         "bytes": (300.0, 300.0), "r16": 2.0, "low": 0.9890, "worst": 0.9970},
    {"name": "int4(mean)", "plain": "centred 4-bit", "chain": "centred 4-bit", "one_bit": False, "merged": False,
         "bytes": (160.0, 160.0), "r16": 3.9, "low": 0.9721, "worst": 0.9930},
]


def _no_rows(spec: dict[str, Any]) -> dict[str, Any]:
    return dict(spec, bytes=None, r16=None, low=None, worst=None)


def _no_bytes(spec: dict[str, Any]) -> dict[str, Any]:
    return dict(spec, bytes=None)


#: a model without rows of its own: pooled minima, no bytes per page, no merged or 1-bit rows (FINAL_DESIGN 5.1.4)
POOLED_SPECS: list[dict[str, Any]] = [
    _no_bytes(sp) if sp["name"] in ("float32", "float16", "int8(per_vector)", "int4(mean)") else _no_rows(sp)
    for sp in PAGE_SPECS
]


def make_row(spec: dict[str, Any], *, status: str = "not measured", reason: str = "", need: float | None = None,
             slack: float | None = None) -> LayoutRow:
    low = spec["low"]
    return LayoutRow(
        name=spec["name"], plain=spec["plain"], spec={"stages": [spec["name"]]}, chain=spec["chain"],
        bytes_per_page=spec["bytes"], ratio_f16=spec["r16"],
        ratio_f32=None if spec["r16"] is None else spec["r16"] * 2.0,
        lowest_kept=low, worst_point=spec["worst"], datasets=(DOCVQA, INFOVQA) if low is not None else (),
        status=status, reason=reason, need=need, slack=slack, one_bit=spec["one_bit"], merged=spec["merged"])


# ---------------------------------------------------------------------------- the rule, from the text of 5.1.2


def s_of_n(n: int) -> float:
    return max(0.0, math.log(n / 500) / math.log(5))


def need_of(spec: dict[str, Any], target: float, n: int, checked: bool, modality: str = "page") -> float | None:
    """target + size_class * s(N) + 0.025 * [1-bit]; text: target + 0.01 for compressed rungs."""
    if spec["name"] == "float32":
        return None
    if spec["chain"] == "float":
        return target
    if modality == "text":
        return target + TEXT_MARGIN + 0.0
    return target + SIZE[(spec["one_bit"], spec["merged"])] * s_of_n(n) + (ONE_BIT_MARGIN if spec["one_bit"] else 0.0)


def reference_select(specs: list[dict[str, Any]], target: float, n: int, *, checked: bool = False,
                     modality: str = "page", blocked: frozenset[str] = frozenset(),
                     pooled: bool = False) -> dict[str, Any]:
    """FINAL_DESIGN 5.1.2 steps 2 to 5 on a list of row specs. Returns qualifying flags by row name, the pick,
    the close-call flag and the safer rung.

    A row clears its needed line when ``lowest kept - need`` is not below zero at six decimals (a value equal to
    its need, as written in decimals, clears it whatever the binary rounding). Above 20,000 pages only the
    8-bit and centred 4-bit rungs of page models remain; a model without rows of its own (``pooled``) gets
    centred 4-bit only when checked.
    """
    why: dict[str, str] = {}
    slack: dict[str, float | None] = {}
    clear: dict[str, bool] = {}
    for sp in specs:
        nm = sp["name"]
        if nm == "float32":
            clear[nm], why[nm], slack[nm] = True, "", None
            continue
        if sp["low"] is None:
            why[nm], slack[nm] = "not measured", None
            clear[nm] = False
            continue
        slack[nm] = round(sp["low"] - need_of(sp, target, n, checked, modality), 6)
        if modality == "text" and sp["chain"] == "centred 4-bit" and n > TEXT_MAX or modality == "page" and (sp["merged"] or sp["one_bit"]) and n > CODECS_ONLY_ABOVE:
            why[nm] = "outside size range"
        elif sp["one_bit"] and not checked or pooled and sp["chain"] == "centred 4-bit" and not checked:
            why[nm] = "needs practice questions"
        elif nm in blocked:
            why[nm] = "not allowed"
        else:
            why[nm] = ""
        clear[nm] = why[nm] == "" and slack[nm] >= 0
    qual: dict[str, bool] = {}
    status: dict[str, str] = {}
    by_chain: dict[str, list[dict[str, Any]]] = {}
    for sp in specs:
        by_chain.setdefault(sp["chain"], []).append(sp)
    for chain, rungs in by_chain.items():
        ok = True
        for sp in rungs:
            nm = sp["name"]
            if chain == "float":
                qual[nm] = clear[nm]
            else:
                qual[nm] = clear[nm] and ok
                ok = ok and clear[nm]
            if qual[nm]:
                status[nm] = "qualifies"
            elif why[nm]:
                status[nm] = why[nm]
            elif slack[nm] < 0:
                status[nm] = "below target"
            else:
                status[nm] = "family stopped"
    pick = None
    for sp in specs:
        if qual[sp["name"]] and (pick is None or sp["r16"] > pick["r16"]):
            pick = sp
    room = {sp["name"]: (math.inf if sp["name"] == "float32" else slack[sp["name"]]) for sp in specs
            if qual[sp["name"]]}
    close = pick is not None and room[pick["name"]] < CLOSE
    safer = None
    if close:
        for sp in specs:
            if qual[sp["name"]] and room[sp["name"]] >= CLOSE and (safer is None or sp["r16"] > safer["r16"]):
                safer = sp
    if pick is not None:
        status[pick["name"]] = "pick"
    return {"qual": qual, "status": status, "pick": pick["name"] if pick else None, "close": close,
            "safer": safer["name"] if safer else None, "slack": room}


def selection_for(specs: list[dict[str, Any]], target: float = 0.97, n: int | None = None, *, checked: bool = False,
                  modality: str = "page", model: str | None = "vendor/colqwen2-like", basis: str = "own rows",
                  blocked: frozenset[str] = frozenset(), hand: str | None = None,
                  notes: tuple[str, ...] = (), pooled: bool = False) -> Selection:
    """A ``Selection`` whose rows carry the statuses, needs and slacks of :func:`reference_select`."""
    n_eff = 500 if n is None else n
    if pooled:
        basis = "unmeasured: pooled minimum over 15 page models"
    ref = reference_select(specs, target, n_eff, checked=checked, modality=modality, blocked=blocked, pooled=pooled)
    rows = []
    for sp in specs:
        nm = sp["name"]
        need = need_of(sp, target, n_eff, checked, modality)
        status = ref["status"][nm]
        if sp["name"] in blocked and status == "not allowed":
            reason = "the page vectors per page are below 70% of the catalogue mean"
        else:
            reason = ""
        sl = None if sp["low"] is None or need is None else round(sp["low"] - need, 6)
        rows.append(make_row(sp, status=status, reason=reason, need=need, slack=sl))
    by_name = {r.name: r for r in rows}
    pick = by_name[hand] if hand else by_name[ref["pick"]]
    return Selection(
        model=model, modality=modality, target=target, n_pages=n, checked=checked, rows=tuple(rows), pick=pick,
        sequence=(pick,), close_call=ref["close"], safer=by_name[ref["safer"]] if ref["safer"] else None,
        basis=basis, notes=notes, hand_picked=hand is not None)


def random_specs(rng: random.Random, modality: str = "page") -> list[dict[str, Any]]:
    """A random ladder with the shape of the page or text ladder; lowest kept values spread around the
    thresholds so every branch of the rule is reached."""
    base = PAGE_SPECS if modality == "page" else TEXT_SPECS
    specs = []
    for sp in base:
        sp = dict(sp)
        if sp["low"] is not None:
            sp["low"] = round(rng.uniform(0.86, 1.0), 4) if rng.random() < 0.85 else None
            if sp["low"] is not None:
                sp["worst"] = round(min(1.02, sp["low"] + rng.uniform(0.0, 0.03)), 4)
            else:
                sp["worst"] = None
        specs.append(sp)
    return specs
