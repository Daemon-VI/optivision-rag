"""The query-free selection rule: which layout a model gets at a target and a size (FINAL_DESIGN 5.1).

The rule reads ``presets/easy_levels.json``, which ``scripts/make_easy_levels.py`` writes from the
measured catalogue and the follow-up reports, with the sha256 of every file it read. For model
``m``, target ``T`` and ``N`` pages:

1. **Evidence.** A rung needs a row on every catalogue dataset of ``m``. Its lowest kept value
   ``L`` is the smallest 95% interval low over the catalogue datasets and the follow-up datasets
   (ArxivQA, TabFQuAD, NFCorpus, SciDocs). Its size and ratio come from the catalogue datasets.
2. **Margin.** ``need = T + size(rung) * s(N) + one_bit_margin * [1-bit]``, with
   ``s(N) = max(0, log(N / 500) / log 5)``; text models use ``T + 0.01`` up to the largest
   collection measured and 8-bit or float16 above it. The adjustable constants come from the
   ledger entry ``margins`` (``presets/easy_ledger.json``), the size coefficients from the JSON.
3. **Chain rule.** A rung qualifies only if every lighter rung of its chain does.
4. **Allowed sets.** Unchecked folders lose the 1-bit chain; above ``codecs_only_above`` pages
   only 8-bit and centred 4-bit remain; ``vectors_per_page`` far below what the model's rows saw
   removes the merged rungs (the shape gate, FINAL_DESIGN 5.5); above a target of 0.99 only
   float16 and float32 remain (FINAL_DESIGN 5.6).
5. **Pick.** The qualifying rung with the highest ratio (its smallest per-dataset ratio against
   float16). Float16 is the floor when its own lowest kept clears ``T``, else float32. A pick
   whose slack ``L - need`` is under 0.005 is a close call; the safer layout named is the largest
   other rung with slack of at least 0.005.

A model without rows of its own gets the same formula over pooled minima: ``L_u(rung)`` is the
smallest ``L`` over every measured model of the same modality, codecs only, and centred 4-bit
only for checked folders (FINAL_DESIGN 5.1.4).

The numbers are in-sample development measurements on public benchmarks; nothing here says
they hold on another collection. Importing this module loads no torch or SciPy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources
from typing import Any

from . import _ledger
from ._types import LayoutRow, Selection

__all__ = [
    "LADDER",
    "RungDef",
    "data",
    "details",
    "find_rung",
    "model_source",
    "need",
    "pooled_source",
    "row",
    "s_of_n",
    "select",
    "short_name",
    "source",
]

# ---------------------------------------------------------------------------- the ladder


@dataclass(frozen=True)
class RungDef:
    """One layout of the ladder.

    ``name`` is the catalogue label, ``spec`` the catalogue ``pipeline`` dict it is built from
    (``Pipeline.from_dict``; never ``parse_pipeline``, which reads ``"int8"`` as ``int8(fixed)``),
    ``chain`` one of ``_types.CHAINS`` and ``size_class`` the key of its size coefficient
    (``float``, ``codec``, ``merge``, ``binary``, ``merge-binary``).
    """

    name: str
    plain: str
    chain: str
    spec: dict[str, Any]
    one_bit: bool
    merged: bool
    size_class: str


def _merge(ratio: float) -> dict[str, Any]:
    return {"max_distance": None, "min_vectors": 1, "ratio": ratio, "stage": "hierarchical_merge"}


_F32 = RungDef("float32", "float32", "float", {"stages": []}, False, False, "float")
_F16 = RungDef("float16", "float16", "float", {"stages": [{"stage": "float16"}]}, False, False, "float")
_CODECS = {
    "int8(per_vector)": ("8-bit", "8-bit", {"scale": "per_vector", "stage": "int8"}, False),
    "int4(mean)": ("centred 4-bit", "centred 4-bit", {"center": "mean", "stage": "int4"}, False),
    "binary": ("1-bit", "1-bit", {"stage": "binary"}, True),
}


def _chain(codec: str) -> list[RungDef]:
    plain, chain, stage, one_bit = _CODECS[codec]
    rungs = [RungDef(codec, plain, chain, {"stages": [dict(stage)]}, one_bit, False,
                     "binary" if one_bit else "codec")]
    for ratio, kept in ((0.33, 3), (0.25, 4)):
        rungs.append(RungDef(f"hierarchical_merge({ratio}) > {codec}", f"merged (1 in {kept} vectors) + {plain}",
                             chain, {"stages": [_merge(ratio), dict(stage)]}, one_bit, True,
                             "merge-binary" if one_bit else "merge"))
    return rungs


#: the ladder per modality, float rows first, then each chain lightest rung first (FINAL_DESIGN 5.1.1)
LADDER: dict[str, list[RungDef]] = {
    "page": [_F32, _F16, *_chain("int8(per_vector)"), *_chain("int4(mean)"), *_chain("binary")],
    "text": [_F32, _F16, _chain("int8(per_vector)")[0], _chain("int4(mean)")[0]],
}

MODALITIES = tuple(LADDER)


def short_name(name: str) -> str:
    """``hierarchical_merge(0.25) > int8(per_vector)`` -> ``hm(0.25) > int8``."""
    return name.replace("hierarchical_merge", "hm").replace("(per_vector)", "")


def _norm(text: str) -> str:
    return " ".join(str(text).lower().replace("centered", "centred").replace(" + ", " > ").split())


def find_rung(name: str, modality: str = "page") -> RungDef:
    """The rung a hand pick names: a plain name (``"centred 4-bit"``), a catalogue label
    (``"int4(mean)"``) or its short form (``"hm(0.25) > int8"``). Raises ``ValueError``."""
    key = _norm(name)
    ladder = LADDER[modality]
    for rung in ladder:
        names = {rung.name, rung.plain, short_name(rung.name), rung.name.replace("hierarchical_merge", "merge")}
        if rung.name == "int8(per_vector)":
            names |= {"int8"}
        if rung.name == "binary":
            names |= {"1 bit"}
        if key in {_norm(n) for n in names}:
            return rung
    known = ", ".join(f'"{r.plain}" ({r.name})' for r in ladder)
    raise ValueError(f"no layout {name!r} on the {modality} ladder; known: {known}")


# ---------------------------------------------------------------------------- the data


@lru_cache(maxsize=1)
def data() -> dict[str, Any]:
    """``presets/easy_levels.json`` (generated; see ``scripts/make_easy_levels.py``)."""
    text = resources.files("optivision").joinpath("presets", "easy_levels.json").read_text(encoding="utf-8")
    loaded = json.loads(text)
    if loaded.get("schema") != 1:
        raise RuntimeError(f"unknown easy_levels.json schema {loaded.get('schema')!r}")
    return loaded


def _rule(key: str) -> Any:
    return data()["rule"][key]


def _margins() -> dict[str, Any]:
    return _ledger.feature("margins")


def lowest_target() -> float:
    return float(_margins()["lowest_target"])


def check_target(target: float) -> float:
    """``target`` as a float in [lowest_target, 1.0]; ``ValueError`` outside."""
    try:
        t = float(target)
    except (TypeError, ValueError):
        raise ValueError(f"target must be a number such as 0.97, got {target!r}") from None
    low = lowest_target()
    if not (low - 1e-12 <= t <= 1.0 + 1e-12) or math.isnan(t):
        raise ValueError(f"target must be between {low} and 1.0 (a share of full-precision search quality), "
                         f"got {target!r}")
    return t


def s_of_n(n: int | None) -> float:
    """The size term: 0 up to the measured collection size, 1 at five times it, log-shaped
    (an assumption resting on two models at two sizes, FINAL_DESIGN 5.1.2)."""
    if n is None:
        return 0.0
    base, log_base = float(_rule("size_base_pages")), float(_rule("size_log_base"))
    return max(0.0, math.log(max(int(n), 1) / base) / math.log(log_base))


def need(target: float, rung: RungDef, n_pages: int | None, *, modality: str = "page",
         checked: bool = False) -> float:
    """The lowest kept value ``rung`` must reach at ``target`` and ``n_pages``."""
    t = float(target)
    if rung.size_class == "float":
        return t
    m = _margins()
    extra = float(m.get("extra_compressed", 0.0))
    if modality == "text":
        return t + float(_rule("text_margin")) + extra
    coef = float(_rule("size_coefficients")[rung.size_class]) * float(m.get("size_coefficient_scale", 1.0))
    out = t + extra + coef * s_of_n(n_pages)
    if rung.one_bit:
        out += float(m.get("one_bit_margin", 0.0))
    if rung.merged and not checked:
        out += float(m.get("unchecked_merge_penalty", 0.0))
    return out


def source(datasets: Sequence[str]) -> str:
    """The evidence source of catalogue claims: ``"ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages"``."""
    table = data()["datasets"]
    parts, family = [], None
    for name in datasets:
        d = table[name]
        size = f"{d['pages']:,} {d['unit']}"
        parts.append(f"{d['family']} {d['label']} {size}" if d["family"] != family else f"{d['label']} {size}")
        family = d["family"]
    return ", ".join(parts)


def model_source(model_id: str) -> str:
    """The source naming every dataset whose rows the rule reads for ``model_id``."""
    entry = data()["models"][model_id]
    return source(_ordered(entry))


def pooled_source(modality: str) -> str:
    pool = data()["pooled"][modality]
    return f"worst of the {pool['n_models']:,} {modality} models we measured, {source(pool['datasets'])}"


def _ordered(entry: dict[str, Any]) -> list[str]:
    ds = entry["datasets"]
    return ([d for d in ds if ds[d]["role"] == "catalogue"] + [d for d in ds if ds[d]["role"] != "catalogue"])


def details(model_id: str, name: str) -> list[dict[str, Any]]:
    """Per-dataset figures of one rung of a model with rows: dataset, role, bytes_per_page,
    ratio_f16, ratio_f32, low, point, vectors_per_page (an empty list when not measured)."""
    entry = data()["models"][model_id]
    recs = entry["rungs"].get(name, {})
    return [{"dataset": d, "role": entry["datasets"][d]["role"], **recs[d]} for d in _ordered(entry) if d in recs]


# ---------------------------------------------------------------------------- the rule


def _resolve_model(model: str | None, modality: str | None) -> tuple[str | None, str | None, str, str]:
    """(registry id, rows id, basis, modality) for ``model``."""
    from . import _resolve

    if model is None:
        mod = _modality(modality or "page")
        return None, None, "unmeasured", mod
    info = _resolve.resolve(model)
    model_id = info.model_id
    rows_id, basis = _resolve.rows_model(model_id)
    mod = _modality(info.modality)
    if modality is not None and _modality(modality) != mod:
        raise ValueError(f"{model_id} is a {mod} model, not {modality}")
    return model_id, rows_id, basis, mod


def _modality(name: str) -> str:
    m = {"image": "page", "pages": "page", "page": "page", "text": "text"}.get(str(name).lower())
    if m is None:
        raise ValueError(f"modality must be 'page' or 'text', got {name!r}")
    return m


def _measured_row(rung: RungDef, entry: dict[str, Any]) -> dict[str, Any] | None:
    """Sizes, ratios, lowest kept and worst point of ``rung`` in a model entry, or None when a
    catalogue dataset lacks it."""
    recs = entry["rungs"].get(rung.name)
    if not recs:
        return None
    cat = [d for d, v in entry["datasets"].items() if v["role"] == "catalogue"]
    if any(d not in recs for d in cat):
        return None
    used = [d for d in _ordered(entry) if d in recs]
    sizes = [recs[d]["bytes_per_page"] for d in cat]
    return {
        "bytes_per_page": (min(sizes), max(sizes)),
        "ratio_f16": min(recs[d]["ratio_f16"] for d in cat),
        "ratio_f32": min(recs[d]["ratio_f32"] for d in cat),
        "lowest_kept": min(recs[d]["low"] for d in used),
        "worst_point": min(recs[d]["point"] for d in used),
        "datasets": tuple(used),
    }


def _pooled_row(rung: RungDef, pool: dict[str, Any]) -> dict[str, Any] | None:
    rec = pool["rungs"].get(rung.name)
    if rec is None:
        return None
    return {"bytes_per_page": None, "ratio_f16": rec["ratio_f16"], "ratio_f32": rec["ratio_f32"],
            "lowest_kept": rec["low"], "worst_point": rec["point"], "datasets": tuple(pool["datasets"])}


REASONS = {
    "pick": "the most compressed layout that clears its needed line",
    "close call": "clears its needed line by under half a point",
    "qualifies": "clears its needed line",
    "below target": "below its needed line",
    "family stopped": "a less compressed layout in this family already fell short",
    "not measured": "not measured",
    "outside size range": "outside the sizes we measured",
    "needs practice questions": "needs practice questions",
    "not allowed": "your pages have far fewer vectors than the pages we measured merging on",
}
#: why a rung is not allowed, when the cause is not the shape gate
TOO_CLOSE = "our benchmarks cannot separate layouts this close to full precision"


def select(model: str | None, target: float, n_pages: int | None = None, *, checked: bool = False,
           modality: str | None = None, vectors_per_page: float | None = None,
           layout: str | None = None) -> Selection:
    """What the rule picks for ``model`` at ``target`` with ``n_pages`` (FINAL_DESIGN 5.1).

    ``n_pages=None`` means a collection no larger than the ones measured (no size margin).
    ``checked`` says whether practice questions can check the layout (only then are 1-bit
    layouts allowed). ``vectors_per_page`` (median of the user's pages) drives the shape gate.
    ``layout=`` is a hand pick: the Selection carries that row as ``pick`` with
    ``hand_picked=True``; every status stays as the rule set it.
    """
    t = check_target(target)
    if n_pages is not None and int(n_pages) < 1:
        raise ValueError(f"n_pages must be a positive count, got {n_pages!r}")
    n = None if n_pages is None else int(n_pages)
    model_id, rows_id, basis, mod = _resolve_model(model, modality)
    margins = _margins()
    codecs_only_above = int(margins.get("codecs_only_above", 20000))
    text_max = int(_rule("text_max_documents"))
    close = float(_rule("close_call"))
    notes: list[str] = []

    pooled = rows_id is None
    if pooled:
        _ledger.require("unmeasured_rules")  # FeatureOff with the ledger's reason when switched off
        pool = data()["pooled"][mod]  # codecs only: merged and 1-bit rungs have no pooled rows
        basis = f"unmeasured: pooled minimum over {pool['n_models']} {mod} models"
        measured = {r.name: _pooled_row(r, pool) for r in LADDER[mod]}
        notes.append("your model was not measured; results have not been shown to carry over to new models, "
                     "so it gets float16, 8-bit or centred 4-bit only")
        gate_floor = None
    else:
        entry = data()["models"][rows_id]
        measured = {r.name: _measured_row(r, entry) for r in LADDER[mod]}
        vpps = [v["vectors_per_page"] for v in entry["datasets"].values() if v.get("vectors_per_page")]
        gate_floor = float(_rule("shape_fraction")) * min(vpps) if vpps else None
        if entry.get("pending_b0"):
            notes.append("these rows predate our campaign format; some layouts wait for the B0 re-measurement")

    shape_gate = (vectors_per_page is not None and gate_floor is not None and mod == "page"
                  and float(vectors_per_page) < gate_floor)
    if shape_gate:
        notes.append("your pages have far fewer vectors than the pages we measured merging on, "
                     "so merged layouts are not used")
    if mod == "page" and n is not None and n > codecs_only_above:
        notes.append("far larger than the largest collection we measured, so only 8-bit and centred 4-bit "
                      "layouts are used")
    if mod == "page" and not checked:
        notes.append("unchecked: no practice questions, so no 1-bit layouts")
    if mod == "text" and n is not None and n > text_max:
        notes.append("more documents than the largest text collection we measured, so 8-bit or float16 only")

    too_close = t > float(_rule("too_close_above"))  # FINAL_DESIGN 5.6: float16 if it clears T, else float32

    def disallowed(rung: RungDef) -> str | None:
        if rung.size_class == "float":
            return None
        if too_close:
            return "not allowed"
        if mod == "text" and rung.name == "int4(mean)" and n is not None and n > text_max:
            return "outside size range"
        if mod == "page" and n is not None and n > codecs_only_above and (rung.merged or rung.one_bit):
            return "outside size range"
        if rung.one_bit and not checked:
            return "needs practice questions"
        if pooled and rung.name == "int4(mean)" and not checked:
            return "needs practice questions"
        if rung.merged and shape_gate:
            return "not allowed"
        return None

    rows: list[LayoutRow] = []
    stopped: dict[str, bool] = {}
    for rung in LADDER[mod]:
        m = measured[rung.name]
        nd = need(t, rung, n, modality=mod, checked=checked)
        base = dict(name=rung.name, plain=rung.plain, spec=rung.spec, chain=rung.chain, one_bit=rung.one_bit,
                    merged=rung.merged, need=round(nd, 6))
        if m is None:
            status = "not measured"
            rows.append(LayoutRow(bytes_per_page=None, ratio_f16=None, ratio_f32=None, lowest_kept=None,
                                  worst_point=None, status=status, reason=REASONS[status], **base))
            stopped[rung.chain] = True
            continue
        slack = round(m["lowest_kept"] - nd, 6)
        own = disallowed(rung)
        if rung.name == "float32":
            status, slack = "qualifies", round(1.0 - t, 6)
        elif own is not None:
            status = own
        elif slack < 0:
            status = "below target"
        elif stopped.get(rung.chain):
            status = "family stopped"
        else:
            status = "qualifies"
        if status != "qualifies" and rung.size_class != "float":
            stopped[rung.chain] = True
        reason = TOO_CLOSE if (own == "not allowed" and too_close) else REASONS[status]
        rows.append(LayoutRow(status=status, reason=reason, slack=slack, **m, **base))

    by_name = {r.name: r for r in rows}
    floor = by_name["float16"] if by_name["float16"].status == "qualifies" else by_name["float32"]
    rule_pick = floor
    for r in rows:  # strict ">": an equal ratio keeps the earlier (lighter-chain) rung, as the reference does
        if r.status == "qualifies" and r.chain != "float" and r.ratio_f16 > rule_pick.ratio_f16:
            rule_pick = r

    hand_picked = layout is not None
    pick = by_name[find_rung(layout, mod).name] if hand_picked else rule_pick
    if hand_picked:
        notes.append(f"chosen by you: {pick.plain}; the rule's own pick is {rule_pick.plain}")

    def ratio(r: LayoutRow) -> float:
        return r.ratio_f16 if r.ratio_f16 is not None else 0.0

    is_close = (pick.status == "qualifies" and pick.name != "float32" and pick.slack is not None
                and pick.slack < close)
    safer_name = None
    if is_close:
        better = [r for r in rows if r.status == "qualifies" and r is not pick and r.name != "float32"
                  and r.slack is not None and r.slack >= close and ratio(r) < ratio(pick)]
        safer_name = max(better, key=ratio).name if better else "float32"

    final: list[LayoutRow] = []
    for r in rows:
        status, reason = r.status, r.reason
        if r is rule_pick:
            status, reason = "pick", REASONS["pick"]
        elif status == "qualifies" and r.name != "float32" and r.slack is not None and r.slack < close:
            status, reason = "close call", REASONS["close call"]
        if r is rule_pick and is_close and not hand_picked:
            reason = REASONS["close call"]
        if hand_picked and r is pick:
            reason = f"chosen by you; {reason}"
        final.append(replace(r, status=status, reason=reason))
    by_name = {r.name: r for r in final}
    pick, floor = by_name[pick.name], by_name[floor.name]
    safer = by_name[safer_name] if safer_name else None

    # the practice check's fixed order: the pick, each safer qualifying layout by falling ratio, the floor
    lower = [r for r in final if r is not pick and r.chain != "float" and r.status in ("pick", "qualifies", "close call")
             and (pick.ratio_f16 is None or ratio(r) < pick.ratio_f16)]
    seq = [pick, *sorted(lower, key=lambda r: -ratio(r))]
    if pick is not floor and pick.name != "float32":
        seq.append(floor)

    if t < float(_rule("less_tested_below")):
        notes.append("less tested: most of our checks are at higher targets")
    if too_close:
        notes.append(TOO_CLOSE + ", so the layout is float16 or float32")
    return Selection(model=model_id, modality=mod, target=t, n_pages=n, checked=bool(checked), rows=tuple(final),
                     pick=pick, sequence=tuple(seq), close_call=is_close, safer=safer, basis=basis,
                     notes=tuple(notes), hand_picked=hand_picked)


def row(model: str | None, name: str, target: float | None = None, n_pages: int | None = None, *,
        checked: bool = False, modality: str | None = None) -> LayoutRow:
    """The ladder row ``name`` of ``model`` with its status at ``target`` (default target if None)."""
    t = float(_rule("default_target")) if target is None else target
    sel = select(model, t, n_pages, checked=checked, modality=modality)
    rung = find_rung(name, sel.modality)
    return next(r for r in sel.rows if r.name == rung.name)
