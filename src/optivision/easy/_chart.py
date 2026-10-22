"""The chart of the layout ladder: a static SVG and a self-contained page with a target slider
(FINAL_DESIGN 2(b), 6.1).

``svg(selection)`` draws one mark per ladder row.

* x is MB per 1,000 pages on a log scale, from ``LayoutRow.bytes_per_page`` (the mark sits at the
  middle of the per-dataset sizes; a thin line shows their range). A model without rows of its own has
  no sizes, so its marks are placed by their size as a share of float16.
* y is *lowest kept*, with a bar to the worst point estimate.
* Each row has a "needed" tick, ``LayoutRow.need`` (target plus margins). A mark above its tick
  clears it.
* A **filled** mark qualifies. A **hollow** mark was refused, and the list under the plot gives the
  reason in plain words (``LayoutRow.reason``). The **pick** is ringed.
* Axis labels name the evidence (``[our benchmarks: ...]``), and ``<title>`` / ``<desc>`` describe
  the chart in words. The selection's own notes and the ledger's status sentence close the chart,
  followed by a ``Next:`` line.

``html(selection)`` wraps the same SVG in a page with a target slider (0.90 to 1.00) and a page
count box. The page recomputes every status in the browser with the rule of FINAL_DESIGN 5.1.2,
the same rule as ``_levels.select`` (``tests/test_easy_chart.py`` runs both and compares them).
The rule is the function ``ovRule`` in :data:`RULE_JS`; it touches no page element, so it can be
run by ``node``. Nothing is loaded from a network: no CDN, no external font, script or image.

Colours are the data-visualisation reference palette: the first three categorical slots, which
validate all-pairs, for the 8-bit, centred 4-bit and 1-bit families, and a neutral ink for the
uncompressed rows. A shape per family carries identity as well, and the list under the plot repeats
every value, so no mark relies on colour alone. Both themes are in the SVG's own style block.

Rounding: *lowest kept* rounds down, a *needed* value rounds up, so the two numbers shown never
suggest more than the rule used.
"""

from __future__ import annotations

import html as _html
import json
import math
import re
import zlib
from typing import Any

from . import _claim, _glossary, _ledger
from ._types import LayoutRow, Selection

__all__ = ["QUALIFYING", "REASONS", "RULE_JS", "extract_rule_and_data", "html", "margins", "svg"]

#: statuses whose mark is filled (the layout qualifies)
QUALIFYING = ("pick", "qualifies", "close call")

#: plain-words reasons for a refused mark (FINAL_DESIGN 2(b))
REASONS = {
    "below target": "below its needed line",
    "family stopped": "a less compressed layout in this family already fell short",
    "not measured": "not measured",
    "outside size range": "outside the sizes we measured",
    "needs practice questions": "needs practice questions",
    "not allowed": "not allowed for these pages",
}

#: the rule's constants as FINAL_DESIGN 5.1.2 states them. ``_levels`` reads the same numbers from its generated
#: ``presets/easy_levels.json``; when that module is importable its values are used, so the page cannot drift from it.
DEFAULT_RULE: dict[str, Any] = {
    "size_coefficients": {"float": 0.0, "codec": 0.005, "merge": 0.0175, "binary": 0.0175, "merge-binary": 0.04},
    "size_base_pages": 500, "size_log_base": 5, "text_margin": 0.01, "text_max_documents": 15183, "close_call": 0.005,
}
#: the slider and the box have these bounds
MAX_TARGET = 1.0
DEFAULT_PAGES = 500

_CHAIN_CLASS = {"float": 0, "8-bit": 1, "centred 4-bit": 2, "1-bit": 3}
_CHAIN_TEXT = {"float": "uncompressed", "8-bit": "8-bit", "centred 4-bit": "centred 4-bit", "1-bit": "1-bit"}

# geometry of the SVG (user units)
W = 760
LEFT, RIGHT, TOP, PH, BAND = 66, 24, 78, 262, 28
KEY_PITCH = 30


# ---------------------------------------------------------------------------------- the rule's inputs


def _rule_data() -> dict[str, Any]:
    try:
        from . import _levels

        rule = _levels.data()["rule"]
        return {k: rule.get(k, v) for k, v in DEFAULT_RULE.items()}
    except (ImportError, KeyError, OSError, RuntimeError, ValueError):
        return dict(DEFAULT_RULE)


def size_key(row: LayoutRow) -> str:
    """The key of the row's size coefficient: float, codec, merge, binary or merge-binary."""
    if row.chain == "float":
        return "float"
    if row.one_bit:
        return "merge-binary" if row.merged else "binary"
    return "merge" if row.merged else "codec"


def size_class(row: LayoutRow, rule: dict[str, Any] | None = None) -> float:
    """The size class of a ladder row (FINAL_DESIGN 5.1.2): 0 for float rows."""
    explicit = getattr(row, "size_class", None)
    if isinstance(explicit, (int, float)):
        return float(explicit)
    table = (rule or _rule_data())["size_coefficients"]
    return float(table[size_key(row)])


def margins() -> dict[str, float]:
    """The margin constants the page's rule uses: the ledger's ``margins`` entry plus the rule's own."""
    m = _ledger.feature("margins")
    rule = _rule_data()
    return {
        "extra": float(m.get("extra_compressed", 0.0)),
        "unchecked_merge": float(m.get("unchecked_merge_penalty", 0.0)),
        "one_bit": float(m.get("one_bit_margin", 0.025)),
        "scale": float(m.get("size_coefficient_scale", 1.0)),
        "codecs_above": float(m.get("codecs_only_above", 20000)),
        "lowest_target": float(m.get("lowest_target", 0.9)),
        "text": float(rule["text_margin"]),
        "text_max": float(rule["text_max_documents"]),
        "close": float(rule["close_call"]),
        "base_pages": float(rule["size_base_pages"]),
        "log_base": float(rule["size_log_base"]),
    }


def _is_ref(row: LayoutRow) -> bool:
    """float32 is the reference every other layout is measured against; it always qualifies."""
    return row.chain == "float" and row.name.lower().startswith("float32")


def _ratio(row: LayoutRow) -> float:
    if row.ratio_f16 is not None:
        return float(row.ratio_f16)
    if row.ratio_f32 is not None:
        return float(row.ratio_f32) / 2.0
    return 0.0


def _mid_mb(row: LayoutRow) -> float | None:
    """MB per 1,000 pages at the middle (geometric) of the per-dataset sizes."""
    if not row.bytes_per_page:
        return None
    lo, hi = (float(b) / 1000.0 for b in row.bytes_per_page)
    return math.sqrt(lo * hi) if lo > 0 and hi > 0 else None


def _rel(row: LayoutRow) -> float | None:
    """The row's size as a share of float16, from its ratio (used when bytes per page are not known)."""
    r = _ratio(row)
    return 1.0 / r if r > 0 else None


def _mb_text(row: LayoutRow) -> str:
    """``"23.9-24.4 MB"`` (per 1,000 pages), built from one row."""
    if not row.bytes_per_page:
        return "not known for this model"
    lo, hi = (_claim.size(float(b) * 1000.0) for b in sorted(row.bytes_per_page))
    if lo == hi:
        return lo
    (lv, lu), (hv, hu) = lo.split(), hi.split()
    return f"{lv}-{hv} {hu}" if lu == hu else f"{lo}-{hi}"


def _ratio_text(r: float) -> str:
    if r <= 0:
        return ""
    if abs(r - 1.0) < 0.05:
        return "the size baseline"
    if r > 1.0:
        return f"{_claim.ratio(r)} smaller than float16"
    return f"{_claim.ratio(1.0 / r)} larger than float16"


def _pct_up(x: float, digits: int = 1) -> str:
    """A needed value, rounded up, so the shown number is never below what the rule used."""
    scale = 10 ** digits
    return f"{math.ceil(100 * x * scale - 1e-9) / scale:.{digits}f}%"


def _pct_target(t: float) -> str:
    return f"{round(100 * t, 1):g}%"


def _esc(s: Any) -> str:
    """XML-escape and keep the output ASCII."""
    return _html.escape(str(s), quote=True).encode("ascii", "xmlcharrefreplace").decode("ascii")


def _f(x: float) -> str:
    return f"{x:.1f}"


def _source(sel: Selection) -> str:
    """The datasets, with page counts, whose rows the selection used. ``LayoutRow.datasets`` holds the rule module's
    dataset keys; ``_levels.source`` turns them into the words of the tag. Rows that already hold words are used
    as they are."""
    seen: list[str] = []
    for row in sel.rows:
        for d in row.datasets:
            if d not in seen:
                seen.append(d)
    if not seen:
        return "no measured rows"
    try:
        from . import _levels

        if sel.basis.startswith("unmeasured"):
            return _levels.pooled_source(sel.modality)
        return _levels.source(seen)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError, TypeError):
        return ", ".join(seen)


def _evidence(sel: Selection) -> _claim.Claim:
    return _claim.claim(None, "catalogue", source=_source(sel))


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


# ---------------------------------------------------------------------------------- per-row view


def _reason(row: LayoutRow) -> str:
    return row.reason or REASONS.get(row.status, row.status)


def _rows_data(sel: Selection) -> list[dict[str, Any]]:
    """What the page's script needs per row, plus the strings that never change."""
    rule = _rule_data()
    order: dict[str, int] = {}
    out = []
    for row in sel.rows:
        k = order.get(row.chain, 0)
        order[row.chain] = k + 1
        out.append({
            "name": row.name, "plain": row.plain, "chain": row.chain, "c": _CHAIN_CLASS.get(row.chain, 0),
            "ord": k, "one_bit": bool(row.one_bit), "merged": bool(row.merged), "size": size_class(row, rule),
            "ref": _is_ref(row),
            "L": None if row.lowest_kept is None else float(row.lowest_kept),
            "W": None if row.worst_point is None else float(row.worst_point),
            "ratio": _ratio(row),
            "bytes": None if not row.bytes_per_page else [float(b) for b in row.bytes_per_page],
            "blocked": _reason(row) if row.status == "not allowed" else "",
            "lt": "" if row.lowest_kept is None else _claim.pct(row.lowest_kept),
            "wt": "" if row.worst_point is None else _claim.pct(row.worst_point),
            "mbt": _mb_text(row), "rt": _ratio_text(_ratio(row)),
        })
    return out


class _Geo:
    """Scales and positions, shared by the SVG and the data the page script reads."""

    def __init__(self, sel: Selection, key: bool) -> None:
        rows = sel.rows
        #: a selection without bytes per page (a model without rows of its own) is placed by size against float16
        self.rel = not any(r.bytes_per_page for r in rows)
        mids = [m for m in (self.xval(r) for r in rows) if m]
        edges = [float(b) / 1000.0 for r in rows if r.bytes_per_page for b in r.bytes_per_page if b]
        lo, hi = (min(mids + edges), max(mids + edges)) if (mids or edges) else (1.0, 10.0)
        if hi / lo < 1.5:
            lo, hi = lo / 1.5, hi * 1.5
        self.xmin, self.xmax = math.log10(lo / 1.25), math.log10(hi * 1.25)
        vals = [v for r in rows for v in (r.lowest_kept, r.worst_point) if v is not None]
        floor_ = min(vals + [0.9]) - 0.01
        self.ymin = math.floor(floor_ / 0.05) * 0.05
        self.ymax = 1.02
        self.band = any(r.lowest_kept is None for r in rows)
        self.x0, self.x1 = LEFT, W - RIGHT
        self.y0 = TOP
        self.y1 = TOP + PH
        self.axis_y = self.y1 + (BAND + 8 if self.band else 0)
        self.band_y = self.y1 + 8 + BAND / 2
        self.key = key
        self.n = len(rows)

    def xval(self, row: LayoutRow) -> float | None:
        return _rel(row) if self.rel else _mid_mb(row)

    def X(self, mb: float) -> float:
        return self.x0 + (math.log10(mb) - self.xmin) / (self.xmax - self.xmin) * (self.x1 - self.x0)

    def Y(self, v: float) -> float:
        y = self.y0 + (self.ymax - v) / (self.ymax - self.ymin) * PH
        return min(self.y1, max(self.y0, y))

    def pos(self, row: LayoutRow) -> tuple[float, float] | None:
        mb = self.xval(row)
        if mb is None:
            return None
        if row.lowest_kept is None:
            return (self.X(mb), self.band_y) if self.band else None
        return self.X(mb), self.Y(row.lowest_kept)

    def plot_dict(self) -> dict[str, float]:
        return {"top": self.y0, "h": PH, "ymin": self.ymin, "ymax": self.ymax}


def _nice_step(span: float, most: int = 8) -> float:
    for step in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        if span / step <= most:
            return step
    return 1.0


def _x_ticks(geo: _Geo) -> list[float]:
    ticks = []
    for k in range(math.floor(geo.xmin) - 1, math.ceil(geo.xmax) + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** k
            if geo.xmin <= math.log10(v) <= geo.xmax:
                ticks.append(v)
    if len(ticks) > 12:
        ticks = [t for t in ticks if abs(math.log10(t) - round(math.log10(t))) < 1e-9]
    return ticks


def _tick_label(v: float, rel: bool = False) -> str:
    if rel:
        return f"{v * 100:g}%"
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


# ---------------------------------------------------------------------------------- shapes and style


def _shape(chain: str, x: float, y: float, r: float = 6.0) -> str:
    """The mark's shape: diamond (uncompressed), circle (8-bit), square (centred 4-bit), triangle (1-bit)."""
    if chain == "centred 4-bit":
        return f'<rect class="ov-shape" x="{_f(x - r)}" y="{_f(y - r)}" width="{_f(2 * r)}" height="{_f(2 * r)}"/>'
    if chain == "1-bit":
        pts = f"{_f(x)},{_f(y - r * 1.25)} {_f(x + r * 1.2)},{_f(y + r * 0.9)} {_f(x - r * 1.2)},{_f(y + r * 0.9)}"
        return f'<polygon class="ov-shape" points="{pts}"/>'
    if chain == "float":
        pts = f"{_f(x)},{_f(y - r * 1.3)} {_f(x + r * 1.3)},{_f(y)} {_f(x)},{_f(y + r * 1.3)} {_f(x - r * 1.3)},{_f(y)}"
        return f'<polygon class="ov-shape" points="{pts}"/>'
    return f'<circle class="ov-shape" cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}"/>'


# light first, then the dark theme; the first three categorical slots of the reference palette
_STYLE = """
.ov-svg{font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;font-size:11px}
.ov-bg{fill:#fcfcfb}
.ov-ink{fill:#0b0b0b}.ov-ink2{fill:#52514e}
.ov-h1{font-size:14px;font-weight:600}.ov-b{font-weight:600}
.ov-grid{stroke:#e1e0d9;stroke-width:1;fill:none}.ov-axis{stroke:#c3c2b7;stroke-width:1;fill:none}
.ov-tg{stroke:#52514e;stroke-width:1;fill:none;opacity:.7}.ov-band{fill:#f1f0ec}
.ov-need{stroke:#0b0b0b;stroke-width:2.5}.ov-wh{stroke:#898781;stroke-width:1}
.ov-ring{fill:none;stroke:#0b0b0b;stroke-width:2.5}
.ov-c0{color:#52514e}.ov-c1{color:#2a78d6}.ov-c2{color:#eb6834}.ov-c3{color:#1baf7a}
.ov-bar{stroke:currentColor;stroke-width:2;opacity:.5}
.ov-yes .ov-shape{fill:currentColor;stroke:#fcfcfb;stroke-width:2}
.ov-no .ov-shape{fill:#fcfcfb;stroke:currentColor;stroke-width:2.2}
@media (prefers-color-scheme: dark){
.ov-bg{fill:#1a1a19}.ov-ink{fill:#f0efec}.ov-ink2{fill:#c3c2b7}
.ov-grid{stroke:#2c2c2a}.ov-axis{stroke:#383835}.ov-tg{stroke:#c3c2b7}.ov-band{fill:#222221}
.ov-need{stroke:#f0efec}.ov-wh{stroke:#898781}.ov-ring{stroke:#f0efec}
.ov-c0{color:#c3c2b7}.ov-c1{color:#3987e5}.ov-c2{color:#d95926}.ov-c3{color:#199e70}
.ov-yes .ov-shape{stroke:#1a1a19}.ov-no .ov-shape{fill:#1a1a19}
}
"""


def _t(x: float, y: float, text: str, cls: str = "ov-ink2", anchor: str = "start", extra: str = "") -> str:
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'<text class="{cls}" x="{_f(x)}" y="{_f(y)}"{a}{extra}>{_esc(text)}</text>'


# ---------------------------------------------------------------------------------- summary text


def _quoted_model(sel: Selection) -> str:
    return f'"{sel.model}"' if sel.model else '"<your model id>"'


def _name(sel: Selection) -> str:
    return sel.model or "A model without rows of its own"


def _target_notes(target: float) -> list[str]:
    notes = []
    if target < 0.95 - 1e-9:
        notes.append("Targets below 95% are less tested.")
    if target > 0.99 + 1e-9:
        notes.append("Our benchmarks cannot separate layouts this close to 100%.")
    return notes


def _say(row: LayoutRow, size: bool) -> str:
    """``"merged (1 in 4 vectors) + 8-bit, 7.9x smaller than float16 (23.9-24.4 MB per 1,000 pages)"``."""
    ratio = _ratio_text(_ratio(row))
    return row.plain + (f", {ratio}" if ratio else "") + (
        f" ({_mb_text(row)} per 1,000 pages)" if size and row.bytes_per_page else "")


def _summary(sel: Selection) -> str:
    """The sentence under the controls. The page script rebuilds the same words."""
    tag = _evidence(sel).tag
    head = "Chosen by you" if sel.hand_picked else "Pick"
    parts = [f"{head}: {_say(sel.pick, True)} {tag}."]
    if sel.close_call:
        parts.append("Close call: it clears the target by under half a point.")
        if sel.safer is not None:
            parts.append(f"Safer: {_say(sel.safer, False)} {tag}.")
    parts += _target_notes(sel.target)
    return " ".join(parts)


def _row_line(sel: Selection, row: LayoutRow) -> str:
    """The second line of a key row: the numbers and the verdict in words."""
    is_pick = row.name == sel.pick.name
    filled = _filled(sel, row)
    close = sel.close_call and is_pick
    if sel.hand_picked and is_pick:
        verdict = "chosen by you" + ("" if filled else "; " + _reason(row))
    elif filled:
        if sel.hand_picked:
            verdict = "the rule's pick" if row.status == "pick" else "qualifies"
        else:
            verdict = "the pick" + (" (close call)" if close else "") if is_pick else "qualifies"
    else:
        verdict = _reason(row)
    if row.lowest_kept is None or row.status == "not measured":
        return verdict
    need = "" if row.need is None or _is_ref(row) else f", needed {_pct_up(row.need)}"
    return f"lowest kept {_claim.pct(row.lowest_kept)}{need}: {verdict}"


def _describe(sel: Selection, n: int) -> str:
    qual = [r.plain for r in sel.rows if _filled(sel, r)]
    refused = [f"{r.plain} ({_reason(r)})" for r in sel.rows if not _filled(sel, r)]
    desc = (f"Layouts for {sel.model or 'a model without rows of its own'} at target {_pct_target(sel.target)}, "
            f"collections up to {_claim.count(n)} pages. Horizontal axis: megabytes per 1,000 pages on a log "
            f"scale. Vertical axis: lowest kept, the share of full-precision nDCG@5 on the worst dataset we "
            f"measured. "
            f"{'Chosen by you' if sel.hand_picked else 'Pick'}: {sel.pick.plain}. "
            f"Filled marks qualify: {'; '.join(qual) or 'none'}. "
            f"Hollow marks were refused: {'; '.join(refused) or 'none'}.")
    return desc


def _filled(sel: Selection, row: LayoutRow) -> bool:
    """A mark is filled when the layout qualifies. The rule's pick qualifies; a hand pick keeps its own status."""
    return row.status in QUALIFYING or (row.name == sel.pick.name and not sel.hand_picked)


# ---------------------------------------------------------------------------------- the SVG


def _resolve(sel: Selection, n_pages: int | None) -> Selection:
    """The selection to draw. ``n_pages`` only labels the chart, unless it differs from the page count
    the selection was made for; then the rule is run again for that count."""
    if not sel.rows:
        raise ValueError("a chart needs a selection with at least one row")
    if n_pages is None or n_pages == sel.n_pages or (sel.n_pages is None and n_pages <= DEFAULT_PAGES):
        return sel
    try:
        from . import _levels
    except ImportError as exc:
        raise ValueError("the selection was made for another page count; make it again with "
                         "_levels.select(..., n_pages=...)") from exc
    kwargs: dict[str, Any] = {"checked": sel.checked, "modality": sel.modality}
    if sel.hand_picked:
        kwargs["layout"] = sel.pick.name
    return _levels.select(sel.model, sel.target, n_pages, **kwargs)


def _build_svg(sel: Selection, n: int, *, key: bool, ids: bool) -> tuple[str, _Geo]:
    geo = _Geo(sel, key)
    rows = sel.rows
    ev = _evidence(sel)
    pick_name = sel.pick.name
    uid = f"ov{zlib.crc32(repr((sel.model, sel.target, n, len(rows))).encode()) % 10**6}"
    n_key = len(rows) if key else 0
    tag_lines = _wrap(ev.tag, 104)
    leg_y = geo.axis_y + 56 + 13 * len(tag_lines)
    key_y = leg_y + 46
    foot_lines = _wrap(f"Evidence {ev.tag}. {_ledger.status()}", 118)
    for note in sel.notes:
        foot_lines += _wrap(note[:1].upper() + note[1:] + ("" if note.endswith(".") else "."), 118)
    if key:                                        # the page has its own Next: line under the controls
        foot_lines.append(f'Next: ov.build("my_pdfs/", model={_quoted_model(sel)}, target={sel.target:.2f})')
    foot_y = (key_y + n_key * KEY_PITCH + 6) if key else (leg_y + 44)
    height = int(foot_y + len(foot_lines) * 14 + 8)

    o: list[str] = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" class="ov-svg" role="img" viewBox="0 0 {W} {height}" '
             f'width="{W}" height="{height}" aria-labelledby="{uid}-t {uid}-d" preserveAspectRatio="xMidYMin meet">')
    title = f"{_name(sel)}: layouts at target {_pct_target(sel.target)}"
    o.append(f'<title id="{uid}-t">{_esc(title)}</title>')
    o.append(f'<desc id="{uid}-d">{_esc(_describe(sel, n))}</desc>')
    o.append(f"<style>{_STYLE}</style>")
    o.append(f'<rect class="ov-bg" x="0" y="0" width="{W}" height="{height}"/>')
    o.append(_t(LEFT - 42, 24, title, "ov-ink ov-h1", extra=' id="ov-title"' if ids else ""))
    basis = sel.basis if sel.basis != "own rows" else "rows of this model"
    sub = f"collections up to {_claim.count(n)} pages; {basis}"
    o.append(_t(LEFT - 42, 42, sub, extra=' id="ov-sub"' if ids else ""))
    o.append(_t(LEFT - 42, 58, "Filled = qualifies, hollow = refused, ring = the pick; the short dash is the "
                "needed line for that layout."))

    # y grid and labels
    step = _nice_step(geo.ymax - geo.ymin)
    v = math.ceil(geo.ymin / step - 1e-9) * step
    digits = 1 if step >= 0.01 and abs(step * 100 - round(step * 100)) < 1e-9 else 2
    while v <= 1.0 + 1e-9:
        y = geo.Y(v)
        o.append(f'<line class="ov-grid" x1="{geo.x0}" x2="{geo.x1}" y1="{_f(y)}" y2="{_f(y)}"/>')
        o.append(_t(geo.x0 - 8, y + 4, f"{round(v * 100, digits):g}%", anchor="end"))
        v += step
    # x grid and labels
    for tv in _x_ticks(geo):
        x = geo.X(tv)
        o.append(f'<line class="ov-grid" x1="{_f(x)}" x2="{_f(x)}" y1="{geo.y0}" y2="{geo.axis_y}"/>')
        o.append(_t(x, geo.axis_y + 15, _tick_label(tv, geo.rel), anchor="middle"))
    o.append(f'<line class="ov-axis" x1="{geo.x0}" x2="{geo.x1}" y1="{geo.axis_y}" y2="{geo.axis_y}"/>')
    o.append(f'<line class="ov-axis" x1="{geo.x0}" x2="{geo.x0}" y1="{geo.y0}" y2="{geo.axis_y}"/>')
    if geo.band:
        o.append(f'<rect class="ov-band" x="{geo.x0}" y="{geo.y1 + 6}" width="{geo.x1 - geo.x0}" height="{BAND}"/>')
        o.append(_t(geo.x0 + 6, geo.band_y + 4, "no measured rows"))
    # target line
    ty = geo.Y(sel.target)
    o.append(f'<line class="ov-tg" id="ov-tg" x1="{geo.x0}" x2="{geo.x1}" y1="{_f(ty)}" y2="{_f(ty)}"/>' if ids else
             f'<line class="ov-tg" x1="{geo.x0}" x2="{geo.x1}" y1="{_f(ty)}" y2="{_f(ty)}"/>')
    # axis labels
    xlabel = ("size as a share of float16, log scale (the size of this model's pages is not known)" if geo.rel
              else "MB per 1,000 pages, log scale (mark at the middle of the per-dataset sizes)")
    o.append(_t((geo.x0 + geo.x1) / 2, geo.axis_y + 32, xlabel, anchor="middle"))
    for j, line in enumerate(tag_lines):
        o.append(_t((geo.x0 + geo.x1) / 2, geo.axis_y + 45 + 13 * j, line, anchor="middle"))
    ymid = (geo.y0 + geo.y1) / 2
    o.append(f'<text class="ov-ink2" transform="rotate(-90 14 {_f(ymid)})" x="14" y="{_f(ymid)}" '
             f'text-anchor="middle">lowest kept, share of full-precision nDCG@5 '
             f'[{_claim.KINDS["catalogue"]}]</text>')

    # bars, ticks, marks
    positions: list[tuple[float, float] | None] = []
    for i, row in enumerate(rows):
        pos = geo.pos(row)
        positions.append(pos)
        if pos is None:
            continue
        x, y = pos
        cls = f"ov-c{_CHAIN_CLASS.get(row.chain, 0)}"
        if row.lowest_kept is not None and row.worst_point is not None:
            y2 = geo.Y(row.worst_point)
            if abs(y2 - y) >= 1:
                o.append(f'<line class="ov-bar {cls}" x1="{_f(x)}" x2="{_f(x)}" y1="{_f(y)}" y2="{_f(y2)}"/>')
        if row.bytes_per_page:
            xa, xb = (geo.X(b / 1000.0) for b in sorted(row.bytes_per_page) if b)
            if xb - xa >= 4:
                o.append(f'<line class="ov-wh" x1="{_f(xa)}" x2="{_f(xb)}" y1="{_f(y)}" y2="{_f(y)}"/>')
        if row.need is not None and row.lowest_kept is not None and not _is_ref(row):
            ny = geo.Y(row.need)
            tid = f' id="ov-k{i}"' if ids else ""
            o.append(f'<line class="ov-need"{tid} x1="{_f(x - 13)}" x2="{_f(x + 13)}" y1="{_f(ny)}" y2="{_f(ny)}"/>')
    for i, row in enumerate(rows):
        pos = positions[i]
        if pos is None:
            continue
        x, y = pos
        yes = _filled(sel, row)
        tip = f"{row.plain}: {_mb_text(row)} per 1,000 pages, {_ratio_text(_ratio(row))}"
        if row.lowest_kept is not None:
            tip += f"; lowest kept {_claim.pct(row.lowest_kept)}"
            if row.worst_point is not None:
                tip += f", worst point {_claim.pct(row.worst_point)}"
        tip += f" {ev.tag}"
        mid = f' id="ov-m{i}"' if ids else ""
        o.append(f'<g{mid} class="ov-mark ov-c{_CHAIN_CLASS.get(row.chain, 0)} {"ov-yes" if yes else "ov-no"}">'
                 f"<title>{_esc(tip)}</title>{_shape(row.chain, x, y)}</g>")
    pi = next((i for i, r in enumerate(rows) if r.name == pick_name), None)
    if pi is not None and positions[pi] is not None:
        px, py = positions[pi]
        o.append(f'<circle class="ov-ring" id="ov-ring" cx="{_f(px)}" cy="{_f(py)}" r="12"/>')
    elif ids:
        o.append('<circle class="ov-ring" id="ov-ring" cx="0" cy="0" r="12" visibility="hidden"/>')
    for i, row in enumerate(rows):
        if positions[i] is not None:
            x, y = positions[i]
            o.append(_t(x + 9, y - 9, str(i + 1), "ov-ink", extra=' font-weight="600"'))

    # legend
    lx, ly = geo.x0 - 42, leg_y
    for state, label in (("yes", "qualifies"), ("no", "refused (reason in the list)")):
        o.append(f'<g class="ov-mark ov-c1 ov-{state}">{_shape("8-bit", lx + 6, ly - 3, 5)}</g>')
        o.append(_t(lx + 18, ly, label))
        lx += 30 + len(label) * 5.8 + 10
    o.append(f'<circle class="ov-ring" cx="{_f(lx + 6)}" cy="{_f(ly - 3)}" r="8"/>')
    o.append(_t(lx + 20, ly, "the pick"))
    lx += 30 + 8 * 5.8 + 10
    o.append(f'<line class="ov-need" x1="{_f(lx)}" x2="{_f(lx + 14)}" y1="{_f(ly - 3)}" y2="{_f(ly - 3)}"/>')
    o.append(_t(lx + 20, ly, "needed line: target plus margin"))
    lx += 30 + 31 * 5.8 + 10
    ly += 18
    lx = geo.x0 - 42
    for chain in ("float", "8-bit", "centred 4-bit", "1-bit"):
        c = _CHAIN_CLASS[chain]
        o.append(f'<g class="ov-mark ov-c{c} ov-yes">{_shape(chain, lx + 6, ly - 3, 5)}</g>')
        label = _CHAIN_TEXT[chain]
        o.append(_t(lx + 18, ly, label))
        lx += 30 + len(label) * 5.8 + 6
    o.append(_t(lx + 4, ly, "bar: from lowest kept to worst point; thin line: range of sizes"))

    # key
    if key:
        for i, row in enumerate(rows):
            y0 = key_y + i * KEY_PITCH
            is_pick = row.name == pick_name
            yes = _filled(sel, row)
            cls = f"ov-c{_CHAIN_CLASS.get(row.chain, 0)}"
            o.append(f'<g class="ov-mark {cls} {"ov-yes" if yes else "ov-no"}">{_shape(row.chain, 22, y0 - 3, 5)}</g>')
            hand = "  (chosen by you)" if sel.hand_picked and is_pick else ""
            o.append(_t(40, y0, f"{i + 1}  {row.plain}{hand}", "ov-ink", extra=' font-weight="600"'))
            bits = ([f"{_mb_text(row)} per 1,000 pages"] if row.bytes_per_page else []) + (
                [_ratio_text(_ratio(row))] if _ratio(row) > 0 else [])
            size_txt = "; ".join(bits) or "size not measured"
            o.append(_t(W - RIGHT, y0, size_txt, anchor="end"))
            sid = f' id="ov-s{i}"' if ids else ""
            o.append(_t(40, y0 + 13, _row_line(sel, row), extra=sid))
    for j, line in enumerate(foot_lines):
        o.append(_t(geo.x0 - 42, foot_y + j * 14, line))
    o.append("</svg>")
    return "\n".join(o), geo


def svg(selection: Selection, *, n_pages: int | None = None) -> str:
    """A static SVG of the selection: one mark per ladder row (see the module docstring).

    ``n_pages`` labels the chart. The selection's own page count is used when it has one; if
    ``n_pages`` names another count, the rule is run again for it (``_levels.select``).
    """
    sel = _resolve(selection, n_pages)
    n = sel.n_pages if sel.n_pages is not None else (n_pages or DEFAULT_PAGES)
    return _build_svg(sel, n, key=True, ids=False)[0]


# ---------------------------------------------------------------------------------- the rule in JS

RULE_JS = r"""
/* The selection rule of FINAL_DESIGN 5.1.2, as optivision.easy._levels.select applies it.
   ovRule touches no page element; tests/test_easy_chart.py runs it under node. */
function ovSize(d, n) { return Math.max(0, Math.log(n / d.margins.base_pages) / Math.log(d.margins.log_base)); }
function ovRound(x) { return Math.round(x * 1e6) / 1e6; }
function ovNeed(d, r, T, n, checked) {
  var m = d.margins;
  if (r.ref || r.chain === "float") { return T; }
  if (d.modality === "text") { return T + m.text + m.extra; }
  return T + m.extra + m.scale * r.size * ovSize(d, n) + (r.one_bit ? m.one_bit : 0) +
    (r.merged && !checked ? m.unchecked_merge : 0);
}
function ovWhy(d, r, n, checked) {
  var m = d.margins;
  if (r.ref) { return ""; }
  if (r.L === null) { return "not measured"; }
  if (d.modality === "text" && r.chain === "centred 4-bit" && n > m.text_max) { return "outside size range"; }
  if (d.modality === "page" && (r.merged || r.one_bit) && n > m.codecs_above) { return "outside size range"; }
  if (r.one_bit && !checked) { return "needs practice questions"; }
  if (d.pooled && r.chain === "centred 4-bit" && !checked) { return "needs practice questions"; }
  return r.blocked ? "not allowed" : "";
}
function ovRule(d, T, n, checked) {
  var R = d.rows, close = d.margins.close, need = [], st = [], q = [], slack = [], chainOk = {};
  var i, r, why, clear, best = -1, safer = -1;
  for (i = 0; i < R.length; i++) {
    r = R[i];
    need[i] = r.ref ? null : ovNeed(d, r, T, n, checked);
    why = ovWhy(d, r, n, checked);
    slack[i] = r.ref ? Infinity : r.L === null ? null : ovRound(r.L - need[i]);
    clear = r.ref || (why === "" && slack[i] >= 0);
    if (r.chain === "float") { q[i] = clear; } else {
      if (chainOk[r.chain] === undefined) { chainOk[r.chain] = true; }
      q[i] = clear && chainOk[r.chain];
      if (!clear) { chainOk[r.chain] = false; }
    }
    st[i] = q[i] ? "qualifies" : why !== "" ? why : slack[i] < 0 ? "below target" : "family stopped";
    if (q[i] && (best < 0 || r.ratio > R[best].ratio)) { best = i; }
  }
  if (best >= 0 && slack[best] < close) {
    for (i = 0; i < R.length; i++) {
      if (q[i] && slack[i] >= close && (safer < 0 || R[i].ratio > R[safer].ratio)) { safer = i; }
    }
  }
  if (best >= 0) { st[best] = "pick"; }
  return { need: need, status: st, q: q, pick: best, close: best >= 0 && slack[best] < close, safer: safer };
}
(function () {
  if (typeof document === "undefined") { return; }
  var d = JSON.parse(document.getElementById("ov-data").textContent), P = d.plot;
  function $(id) { return document.getElementById(id); }
  function up(x) { return (Math.ceil(x * 1000 - 1e-9) / 10).toFixed(1) + "%"; }
  function py(v) { return Math.min(P.top + P.h, Math.max(P.top, P.top + (P.ymax - v) / (P.ymax - P.ymin) * P.h)); }
  function say(r, size) {
    return r.plain + (r.rt ? ", " + r.rt : "") + (size && r.bytes ? " (" + r.mbt + " per 1,000 pages)" : "");
  }
  function verdict(r, i, why) {
    if (i === d.hand) { return "chosen by you" + (r.q[i] ? "" : "; " + why); }
    if (!r.q[i]) { return why; }
    return i !== r.pick ? "qualifies" : d.hand >= 0 ? "the rule's pick" : "the pick" + (r.close ? " (close call)" : "");
  }
  function render() {
    var T = parseFloat($("ov-t").value), n = Math.max(1, Math.round(parseFloat($("ov-n").value) || 500));
    var r = ovRule(d, T, n, d.checked), tp = Math.round(T * 1000) / 10 + "%", i, row, el, v, ring = $("ov-ring");
    $("ov-tv").textContent = tp;
    for (i = 0; i < d.rows.length; i++) {
      row = d.rows[i];
      if ((el = $("ov-m" + i))) { el.setAttribute("class", "ov-mark ov-c" + row.c + (r.q[i] ? " ov-yes" : " ov-no")); }
      if ((el = $("ov-k" + i)) && r.need[i] !== null) {
        v = py(r.need[i]); el.setAttribute("y1", v); el.setAttribute("y2", v);
      }
      if ((el = $("ov-s" + i))) {
        v = r.status[i] === "not allowed" ? row.blocked : d.reasons[r.status[i]];
        el.textContent = (row.L === null ? "" : "lowest kept " + row.lt +
          (r.need[i] === null ? "" : ", needed " + up(r.need[i])) + ": ") + verdict(r, i, v);
      }
    }
    v = py(T); $("ov-tg").setAttribute("y1", v); $("ov-tg").setAttribute("y2", v);
    $("ov-sub").textContent = "collections up to " + n.toLocaleString("en-US") + " pages; " + d.basis;
    $("ov-title").textContent = d.title + tp;
    row = d.rows[d.hand >= 0 ? d.hand : r.pick];
    if (ring) {
      ring.setAttribute("visibility", row && row.x !== null ? "visible" : "hidden");
      if (row && row.x !== null) { ring.setAttribute("cx", row.x); ring.setAttribute("cy", row.y); }
    }
    v = r.pick < 0 ? "No layout qualifies at this target." :
      (d.hand >= 0 ? "Chosen by you: " + row.plain + ". The rule picks: " : "Pick: ") +
      say(d.rows[r.pick], true) + " " + d.tag + ".";
    if (r.close) { v += " Close call: it clears the target by under half a point."; }
    if (r.close && r.safer >= 0) { v += " Safer: " + say(d.rows[r.safer], false) + " " + d.tag + "."; }
    if (T < 0.95 - 1e-9) { v += " Targets below 95% are less tested."; }
    if (T > 0.99 + 1e-9) { v += " Our benchmarks cannot separate layouts this close to 100%."; }
    $("ov-sum").textContent = v;
    $("ov-next").textContent = "Next: " + d.next.replace("{t}", T.toFixed(2));
  }
  $("ov-t").addEventListener("input", render);
  $("ov-n").addEventListener("input", render);
})();
"""


def _json_script(obj: Any) -> str:
    """JSON safe inside a ``<script>`` element and ASCII."""
    text = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    return text.replace("</", "<\\/").replace("<!--", "<\\!--")


_PAGE_CSS = """
:root{color-scheme:light dark;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--card:#ffffff}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--card:#222221}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:820px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 6px}
p{margin:6px 0}.muted{color:var(--ink2)}
form.controls{display:flex;flex-wrap:wrap;gap:12px 28px;align-items:center;margin:14px 0;padding:12px 14px;
 border:1px solid var(--line);border-radius:8px;background:var(--card)}
form.controls label{display:flex;align-items:center;gap:10px}
input[type=range]{width:200px}input[type=number]{width:110px;padding:4px 6px;font:inherit;color:inherit;
 background:transparent;border:1px solid var(--line);border-radius:4px}
.chart{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--bg)}
.chart svg{display:block;min-width:620px;width:100%;height:auto}
table.key{border-collapse:collapse;width:100%;margin-top:14px;font-size:13px}
table.key th,table.key td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
dl.terms{margin:6px 0;font-size:13px}
dl.terms dt{font-weight:600;float:left;clear:left;width:11em}dl.terms dd{margin:0 0 4px 11em}
.next{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;margin-top:18px;word-break:break-all}
@page{size:A4;margin:14mm}
@media print{body{background:#fff;color:#000}.controls{display:none}.chart{border:0;overflow:visible}
 .chart svg{min-width:0}main{max-width:none;padding:0}}
"""


def _key_table(sel: Selection, n: int) -> str:
    rows = []
    for i, row in enumerate(sel.rows):
        rows.append(
            f"<tr><td>{i + 1}</td><td>{_esc(row.plain)}</td>"
            f"<td>{_esc(_mb_text(row))}</td><td>{_esc(_ratio_text(_ratio(row)))}</td>"
            f'<td id="ov-s{i}">{_esc(_row_line(sel, row))}</td></tr>')
    return ('<table class="key"><thead><tr><th>#</th><th>Layout</th><th>Per 1,000 pages</th><th>Against float16</th>'
            "<th>Lowest kept and verdict</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def html(selection: Selection, *, n_pages: int | None = None, title: str | None = None) -> str:
    """A self-contained page: the chart, a target slider and a page-count box (see the module docstring).

    The statuses of the first view are the selection's own; they change only when a control moves.
    """
    sel = _resolve(selection, n_pages)
    n = sel.n_pages if sel.n_pages is not None else (n_pages or DEFAULT_PAGES)
    body, geo = _build_svg(sel, n, key=False, ids=True)
    ev = _evidence(sel)
    m = margins()
    rows = _rows_data(sel)
    for row, data in zip(sel.rows, rows):
        pos = geo.pos(row)
        data["x"], data["y"] = (None, None) if pos is None else (round(pos[0], 1), round(pos[1], 1))
    model = sel.model or "a model without rows of its own"
    build_model = _quoted_model(sel)
    data = {
        "model": sel.model, "modality": sel.modality, "checked": bool(sel.checked),
        "pooled": sel.basis.startswith("unmeasured"),
        "hand": next((i for i, r in enumerate(sel.rows) if r.name == sel.pick.name), -1) if sel.hand_picked else -1,
        "tag": ev.tag, "margins": m, "reasons": REASONS,
        "plot": geo.plot_dict(), "rows": rows, "title": f"{_name(sel)}: layouts at target ",
        "basis": sel.basis if sel.basis != "own rows" else "rows of this model",
        "next": f'ov.build("my_pdfs/", model={build_model}, target={{t}})',
    }
    page_title = title or f"Layouts for {model}"
    seen: set[str] = set()
    notes = _glossary.notes_for(
        "target lowest kept close call full precision nDCG@5 float16 layout merged unchecked", seen)
    terms = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in (n_.split(": ", 1) for n_ in notes))
    t0 = round(sel.target, 3)
    slider_min = m["lowest_target"]
    nxt = data["next"].replace("{t}", f"{sel.target:.2f}")
    lead = ("Target means: keep at least this share of the model's full-precision search quality (nDCG@5). "
            "It does not mean this share of answers is right.")
    extra = " ".join(note[:1].upper() + note[1:] + ("" if note.endswith(".") else ".") for note in sel.notes)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(page_title)}</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<main>
<h1>{_esc(page_title)}</h1>
<p class="muted">{_esc(lead)}</p>
<form class="controls" onsubmit="return false">
<label>Target <input id="ov-t" type="range" min="{slider_min:g}" max="{MAX_TARGET:g}" step="0.01" value="{t0:g}"
 aria-label="Target share of full-precision search quality">
 <output id="ov-tv" for="ov-t">{_pct_target(sel.target)}</output></label>
<label>Pages <input id="ov-n" type="number" min="1" step="1" value="{n}"
 aria-label="Number of pages in the collection"></label>
</form>
<p id="ov-sum" aria-live="polite">{_esc(_summary(sel))}</p>
<div class="chart">
{body}
</div>
{_key_table(sel, n)}
<p class="muted">{_esc(extra)}</p>
<h2>Terms</h2>
<dl class="terms">{terms}</dl>
<p class="muted">Evidence {_esc(ev.tag)}. {_esc(_ledger.status())}</p>
<p class="next" id="ov-next">Next: {_esc(nxt)}</p>
<noscript><p class="muted">The slider needs JavaScript; the chart above shows the selection as made.</p></noscript>
</main>
<script type="application/json" id="ov-data">{_json_script(data)}</script>
<script id="ov-rule">{RULE_JS}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------------- helpers for tests


_SCRIPT = re.compile(r'<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>', re.DOTALL)


def extract_rule_and_data(page: str) -> tuple[str, dict[str, Any]]:
    """The rule script and the embedded data of a page made by :func:`html`."""
    script, data = "", {}
    for m in _SCRIPT.finditer(page):
        if 'id="ov-rule"' in m.group("attrs"):
            script = m.group("body")
        elif 'id="ov-data"' in m.group("attrs"):
            data = json.loads(m.group("body"))
    if not script or not data:
        raise ValueError("not a page made by _chart.html")
    return script, data
