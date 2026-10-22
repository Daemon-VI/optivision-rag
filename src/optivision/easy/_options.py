"""``options()``: the plan before any GPU work, and the chart of the rule (FINAL_DESIGN 4.1, 2).

    ov.options()                                             # every registry model, Start here
    ov.options("vidore/colqwen2-v1.0-hf", target=0.97)       # the pick and the ladder
    ov.options("vidore/colqwen2-v1.0-hf", pages="my_pdfs/")  # plus size, disk, time for that folder
    ov.options("vidore/colqwen2-v1.0-hf").save("colqwen2.html")

Every number shown is a :mod:`._claim` claim with one evidence tag; catalogue lines name the
datasets, with page counts, that the rule used (``_levels.model_source``). Nothing here loads a
model, downloads anything or needs a GPU. Page counting (``_docs.count_pages``), the time
estimate (``_encode.estimate``) and the chart (``_chart``) are imported when used.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import _claim, _ledger, _levels
from ._claim import Block, Line, claim, lit
from ._types import LayoutRow, Selection

__all__ = ["Options", "options"]

_EST_BYTES = "our bytes per page x your pages"


def _pct(x: float) -> str:
    """A target as typed in percent: 0.97 -> ``97%``, 0.975 -> ``97.5%``."""
    v = round(100 * float(x), 2)
    return f"{v:g}%"


def _pct_up(x: float) -> str:
    """A needed value in percent, rounded up to a tenth (the bar a layout must clear)."""
    return f"{math.ceil(round(1000 * float(x), 6)) / 10:.1f}%"


def _secs(x: float) -> str:
    """Encode seconds per page: two decimals, three below a tenth of a second (text models)."""
    return f"{float(x):.3f}" if float(x) < 0.1 else f"{float(x):.2f}"


def _span(lo: float, hi: float, fmt: Any, src: str, first: int) -> tuple[str, list[_claim.Claim]]:
    """``"{i}-{j}"`` and its two claims, or ``"{i}"`` and one claim when both ends read the same."""
    a, b = fmt(lo), fmt(hi)
    if a == b:
        return "{%d}" % first, [claim(lo, "catalogue", src, "", a)]
    return "{%d}-{%d}" % (first, first + 1), [claim(lo, "catalogue", src, "", a), claim(hi, "catalogue", src, "", b)]


def _mb(nbytes_per_page: float) -> str:
    """Bytes per page as MB per thousand pages, to a tenth."""
    return f"{float(nbytes_per_page) / 1000:.1f}"


def _unit(modality: str) -> str:
    return "documents" if modality == "text" else "pages"


def _metric(modality: str) -> str:
    return "nDCG@10" if modality == "text" else "nDCG@5"


def _catalogue_datasets(model_id: str) -> list[str]:
    ds = _levels.data()["models"][model_id]["datasets"]
    return [d for d, v in ds.items() if v["role"] == "catalogue"]


@dataclass
class Options:
    """What ``options()`` found. ``str(options)`` is the printed plan (a ``_claim.Block``).

    ``rows`` holds every ladder layout with its status at ``target`` (``LayoutRow``), ``pick`` the
    layout the rule picks and ``selection`` the full ``_types.Selection``; all three are empty
    for ``options()`` without a model, which fills ``models`` (one dict per registry model) and
    ``start_here`` instead.
    """

    model: str | None
    target: float
    n_pages: int | None
    block: Block
    selection: Selection | None = None
    rows: list[LayoutRow] = field(default_factory=list)
    pick: LayoutRow | None = None
    models: list[dict[str, Any]] = field(default_factory=list)
    start_here: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    _table_block: Block | None = field(default=None, repr=False)

    def __str__(self) -> str:
        return str(self.block)

    def __repr__(self) -> str:
        return str(self.block)

    def table(self) -> str:
        """Every ladder layout of the model with its status at this target (FINAL_DESIGN 4.1)."""
        if self._table_block is None:
            raise ValueError("the layout table needs a model: ov.options(model).table()")
        return str(self._table_block)

    def as_dict(self) -> dict[str, Any]:
        sel = self.selection
        return {
            "model": self.model,
            "target": self.target,
            "n_pages": self.n_pages,
            "pick": None if self.pick is None else self.pick.name,
            "rows": [asdict(r) for r in self.rows],
            "selection": None if sel is None else {
                "basis": sel.basis, "modality": sel.modality, "checked": sel.checked,
                "close_call": sel.close_call, "safer": None if sel.safer is None else sel.safer.name,
                "sequence": [r.name for r in sel.sequence], "notes": list(sel.notes),
                "hand_picked": sel.hand_picked},
            "models": self.models,
            "start_here": self.start_here,
            "plan": self.plan,
            "notes": self.notes,
            "lines": self.block.as_dict(),
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the chart: ``.html`` (self-contained page, no CDN) or ``.svg``."""
        if self.selection is None:
            raise ValueError("the chart needs a model: ov.options(model).save(path)")
        from . import _chart

        out = Path(path)
        suffix = out.suffix.lower()
        if suffix == ".html":
            text = _chart.html(self.selection)
        elif suffix == ".svg":
            text = _chart.svg(self.selection, n_pages=self.n_pages)
        else:
            raise ValueError(f"save() writes .html or .svg, not {out.suffix or 'a path without a suffix'!r}")
        out.write_text(text, encoding="utf-8")
        return out

    def _repr_html_(self) -> str:
        import html

        text = f"<pre>{html.escape(str(self))}</pre>"
        if self.selection is None:
            return text
        try:
            from . import _chart

            return _chart.svg(self.selection, n_pages=self.n_pages) + text
        except ImportError:
            return text


# ---------------------------------------------------------------------------- options()


def options(model: str | None = None, target: float | None = None, *, pages: int | str | os.PathLike[str] | None = None,
            accuracy: float | None = None, work_dir: str | os.PathLike[str] | None = None) -> Options:
    """The plan for ``model`` at ``target`` (default 0.97), or every registry model when
    ``model`` is None (FINAL_DESIGN 4.1).

    ``target`` is the share of the model's full-precision search quality (nDCG@5; nDCG@10 for
    text models) to keep, in [0.90, 1.0]; it is not a share of answers that are right.
    ``accuracy=`` is an alias that says so. ``pages`` is a page count or a folder / PDF whose
    pages are counted (no rendering); it adds the size, the temporary disk and where it goes,
    the free space there, the encode time and the cost of re-encoding. ``work_dir`` is where
    ``build`` would put its temporary float16 copy (default: beside the folder; on Colab,
    ``/content/.ov_work``).
    """
    notes: list[str] = []
    if accuracy is not None:
        if target is not None and float(target) != float(accuracy):
            raise ValueError(f"target={target} and accuracy={accuracy} disagree; pass one of them")
        target = accuracy
        a = float(accuracy)
        sentence = (f"accuracy={accuracy} read as target={accuracy}: keep at least {_pct(a)} of full-precision "
                    f"search quality (nDCG@5); not {_pct(a)} of answers correct.")
        print(sentence)
        notes.append(sentence)
    given = target is not None
    t = _levels.check_target(target if given else _levels.data()["rule"]["default_target"])
    if model is None:
        return _overview(t, notes)
    return _plan(model, t, given, pages, work_dir, notes)


# ---------------------------------------------------------------------------- one model


def _plan(model: str, t: float, given: bool, pages: Any, work_dir: Any, notes: list[str]) -> Options:
    from . import _resolve

    info = _resolve.resolve(model)
    model_id = info.model_id
    rows_id, basis = _resolve.rows_model(model_id)
    count = None
    if pages is None:
        n = None
    elif isinstance(pages, bool):
        raise TypeError("pages must be a page count or a folder path")
    elif isinstance(pages, int):
        if pages < 1:
            raise ValueError(f"pages must be a positive count, got {pages}")
        n = pages
    else:
        from . import _docs

        count = _docs.count_pages(pages)
        n = max(int(count.n_pages), 1)
    sel = _levels.select(model_id, t, n)
    mod = sel.modality
    src = _levels.model_source(rows_id) if rows_id else _levels.pooled_source(mod)

    def cat(value: Any, text: str, construct: str = "") -> _claim.Claim:
        return claim(value, "catalogue", src, construct, text)

    pick = sel.pick
    unit = _unit(mod)
    block = Block([], title=None)
    head = f"{model_id}, target {_pct(t)}" + ("" if given else " (the default)")
    if n is None:
        base = (_levels.data()["rule"]["text_max_documents"] if mod == "text"
                else _levels.data()["rule"]["size_base_pages"])
        cat_src = _levels.source(_catalogue_datasets(rows_id)) if rows_id else _levels.pooled_source(mod)
        block.add("", "{0}, collections up to {1} " + unit, lit(head),
                  claim(base, "catalogue", cat_src if mod == "page" else _text_scale_source(), "collection size",
                        _claim.count(base)))
    elif count is not None:
        block.add("", "{0}, plan for {1} " + unit, lit(head), claim(n, "counted", text=_claim.count(n)))
    else:
        block.add("", "{0}, {1}", lit(head), lit(f"{_claim.count(n)} {unit} (as given)"))

    if basis.startswith("measured via"):
        block.add("rows", "{0} has no rows of its own; it gives the same vectors as {1}, whose rows are used",
                  lit(model_id), lit(rows_id))
    elif rows_id is None:
        block.add("rows", "{0} was not measured; results have not been shown to carry over to new models",
                  lit(model_id))
    elif _levels.data()["models"][rows_id].get("pending_b0"):
        block.add("rows", "older-format rows; some layouts wait for their re-measurement (B0)")

    close = "; close call" if sel.close_call else ""
    plan: dict[str, Any] = {}
    if n is None:  # journey (a): the pick, its size and lowest kept, the close call, larger collections
        block.add("pick", "{0}, layout={1}{2}", lit(pick.plain), lit(f'"{_levels.short_name(pick.name)}"'), lit(close))
        _size_line(block, pick, cat, unit)
        _kept_line(block, pick, cat, mod, rows_id is None)
        if sel.close_call and sel.safer is not None:
            _close_lines(block, sel, cat, t)
        _larger_line(block, model_id, t, mod, cat)
    else:  # journey (c): the plan for this many pages
        bigger = (" (larger folders get bigger safety margins)"
                  if mod == "page" and n > _levels.data()["rule"]["size_base_pages"] else "")
        block.add("layout", "{0}, layout={1}{2}{3}", cat(pick.name, pick.plain, "layout the rule picks"),
                  lit(f'"{_levels.short_name(pick.name)}"'), lit(bigger), lit(close))
        _size_range_line(block, n, mod)
        _kept_line(block, pick, cat, mod, rows_id is None)
        if sel.close_call and sel.safer is not None:
            _close_lines(block, sel, cat, t)
        plan = _pages_lines(block, model_id, sel, n, count, pages, work_dir)
    _checked_line(block, model_id, t, n, sel, cat)
    _target_lines(block, t)
    block.add("status", "{0}", lit(_ledger.status()))
    table = Block([])
    _table(table, sel, cat, mod)
    if not given and n is None:  # journey (b): no target given, so the whole ladder at the default
        block.lines.extend(table.lines)

    source = str(pages) if (pages is not None and not isinstance(pages, int)) else "my_pdfs/"
    call = f'ov.build("{source}", model="{model_id}", target={t:g}'
    if work_dir is not None:
        call += f', work_dir="{work_dir}"'
    block.next = call + ")"
    table.next = block.next
    return Options(model=model_id, target=t, n_pages=n, block=block, selection=sel, rows=list(sel.rows), pick=pick,
                   plan=plan, notes=notes + list(sel.notes), _table_block=table)


def _text_scale_source() -> str:
    return str(_levels.data()["rule"]["text_max_source"])


def _size_range_line(block: Block, n: int, mod: str) -> None:
    """FINAL_DESIGN 5.1.2 step 4: above the codecs-only limit, say how far past our largest
    measured collection the folder is."""
    limit = int(_ledger.setting("margins", "codecs_only_above", 20000))
    rule = _levels.data()["rule"]
    if mod != "page" or n <= limit:
        return
    largest = int(rule["largest_measured_pages"])
    block.add("size range", "more than {0} larger than the largest collection we measured ({1}); only 8-bit and "
              "centred 4-bit layouts", lit(f"{limit // largest}x"),
              claim(largest, "catalogue", str(rule["largest_measured_source"]), "pages", f"{_claim.count(largest)} pages"))


def _size_line(block: Block, row: LayoutRow, cat: Any, unit: str) -> None:
    if row.bytes_per_page is None:
        if row.ratio_f16 is not None and row.name not in ("float16", "float32"):
            block.add("size", "{0} smaller than float16 on the models we measured; your model's bytes per page "
                      "were not measured", cat(row.ratio_f16, _claim.ratio(row.ratio_f16), "ratio against float16"))
        return
    lo, hi = row.bytes_per_page
    span = [cat(lo, _mb(lo), "MB per thousand " + unit)] + ([cat(hi, _mb(hi), "MB per thousand " + unit)]
                                                         if _mb(lo) != _mb(hi) else [])
    rng = "{0}-{1}" if len(span) == 2 else "{0}"
    k = len(span)
    if row.name == "float16":
        block.add("size", rng + " MB per thousand " + unit + ", the float16 baseline", *span)
    elif row.name == "float32":
        block.add("size", rng + " MB per thousand " + unit + ", full precision (twice the float16 size)", *span)
    else:
        block.add("size", rng + " MB per thousand " + unit + ", {" + str(k) + "} smaller than float16", *span,
                  cat(row.ratio_f16, _claim.ratio(row.ratio_f16), "ratio against float16"))


def _kept_line(block: Block, row: LayoutRow, cat: Any, mod: str, pooled: bool) -> None:
    if row.lowest_kept is None:
        return
    text = _claim.pct(row.lowest_kept)
    if pooled:
        block.add("lowest kept", "{0} of full-precision search quality ({1}) on the worst model we measured",
                  cat(row.lowest_kept, text, f"{_metric(mod)} retention, interval low"), lit(_metric(mod)))
    else:
        block.add("lowest kept", "{0} of full-precision search quality ({1})",
                  cat(row.lowest_kept, text, f"{_metric(mod)} retention, interval low"), lit(_metric(mod)))


def _close_lines(block: Block, sel: Selection, cat: Any, t: float) -> None:
    s = sel.safer
    block.add("close call", "clears {0} by under half a point. We use it by default.", lit(_pct(t)))
    parts: list[Any] = [lit(f'"{_levels.short_name(s.name)}"'), lit(s.plain)]
    template = "Safer: layout={0} ({1})"
    i = 2
    if s.bytes_per_page is not None:
        lo, hi = s.bytes_per_page
        if _mb(lo) != _mb(hi):
            template += ", {%d}-{%d} MB" % (i, i + 1)
            parts += [cat(lo, _mb(lo)), cat(hi, _mb(hi))]
            i += 2
        else:
            template += ", {%d} MB" % i
            parts.append(cat(lo, _mb(lo)))
            i += 1
    if s.ratio_f16 is not None and s.name not in ("float16", "float32"):
        template += ", {%d}" % i
        parts.append(cat(s.ratio_f16, _claim.ratio(s.ratio_f16)))
        i += 1
    if s.lowest_kept is not None:
        template += ", lowest kept {%d}" % i
        parts.append(cat(s.lowest_kept, _claim.pct(s.lowest_kept)))
    block.add(" ", template, *parts)  # a blank label keeps the line under the value column


def _larger_line(block: Block, model_id: str, t: float, mod: str, cat: Any) -> None:
    if mod == "text":
        big = int(_levels.data()["rule"]["text_max_documents"]) + 1
        s = _levels.select(model_id, t, big)
        block.add("larger", "above {0}: {1} {2}", lit(f"{_claim.count(big - 1)} documents"), lit(s.pick.plain),
                  cat(s.pick.ratio_f16, _ratio_text(s.pick)))
        return
    parts: list[Any] = []
    pieces = []
    for i, n in enumerate((1000, 2500, 20000)):
        s = _levels.select(model_id, t, n)
        pieces.append("{%d}: {%d} {%d}" % (3 * i, 3 * i + 1, 3 * i + 2))
        parts += [lit(f"{_claim.count(n)} pages"), lit(s.pick.plain), cat(s.pick.ratio_f16, _ratio_text(s.pick))]
    block.add("larger", "; ".join(pieces), *parts)


def _ratio_text(row: LayoutRow) -> str:
    if row.name == "float32":
        return "full precision"
    return _claim.ratio(row.ratio_f16) if row.ratio_f16 is not None else "ratio not measured"


def _checked_line(block: Block, model_id: str, t: float, n: int | None, sel: Selection, cat: Any) -> None:
    """Say what practice questions could allow, without promising it (1-bit needs a checked folder)."""
    if sel.modality != "page":
        return
    checked = _levels.select(model_id, t, n, checked=True)
    if checked.pick.name != sel.pick.name:
        block.add("checked", "when practice questions from your PDFs can check the layout, the rule allows {0}, {1}",
                  lit(checked.pick.plain), cat(checked.pick.ratio_f16, _ratio_text(checked.pick)))


def _target_lines(block: Block, t: float) -> None:
    rule = _levels.data()["rule"]
    if t < float(rule["less_tested_below"]):
        block.add("target", "{0} is less tested; most of our checks are at {1} and above",
                  lit(_pct(t)), lit(_pct(rule["less_tested_below"])))
    if t > float(rule["too_close_above"]):
        block.add("target", "our benchmarks cannot separate layouts this close to {0}", lit("100%"))


def _table(block: Block, sel: Selection, cat: Any, mod: str) -> None:
    """One line per ladder layout: plain name, spec, MB per thousand pages, ratio, lowest kept,
    worst point, what it needs at this target and size, status."""
    unit = "pages" if mod == "page" else "documents"
    block.add("layouts", "at target {0}: MB per thousand {1}, ratio against float16, lowest kept, worst point, "
              "needed, status", lit(_pct(sel.target)), lit(unit))
    width = max(len(r.plain) for r in sel.rows) + 2
    for r in sel.rows:
        name = lit(f"{r.plain:<{width}}{_levels.short_name(r.name):<24}")
        status = r.status + ("" if not (sel.hand_picked and r is sel.pick) else ", chosen by you")
        if r.reason and r.reason != r.status and r.status not in ("pick", "qualifies"):
            status += f" ({r.reason})"
        if r.lowest_kept is None:
            block.add("", "{0}{1}", name, lit(f"{'':<50}{status}"))
            continue
        parts: list[Any] = [name]
        if r.bytes_per_page is not None:
            lo, hi = r.bytes_per_page
            size = f"{_mb(lo)}-{_mb(hi)}" if _mb(lo) != _mb(hi) else _mb(lo)
            parts.append(cat(r.bytes_per_page, f"{size:<12}"))
        else:
            parts.append(lit(f"{'':<12}"))
        parts.append(cat(r.ratio_f16, f"{_claim.ratio(r.ratio_f16):<7}"))
        parts.append(cat(r.lowest_kept, f"{_claim.pct(r.lowest_kept):<8}"))
        parts.append(cat(r.worst_point, f"{_claim.retention(r.worst_point):<14}"))
        parts.append(lit(f"{_pct_up(r.need) if r.need is not None else '':<7}  {status}"))
        block.add("", "{0}{1}{2}{3}{4}{5}", *parts)


# ---------------------------------------------------------------------------- pages= planning


def _on_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def default_work_dir(source: Any) -> Path:
    """Where ``build`` keeps its temporary float16 copy when ``work_dir`` is not given:
    ``/content/.ov_work`` on Colab, else ``<folder>.ovx.work`` beside the folder (FINAL_DESIGN 4.2)."""
    if _on_colab():
        return Path("/content/.ov_work")
    if source is None or isinstance(source, int):
        return Path.cwd() / ".ov_work"
    p = Path(source).expanduser()
    stem = p.name if p.is_dir() or not p.suffix else p.stem
    return p.resolve().parent / f"{stem}.ovx.work"


def _free_bytes(path: Path) -> tuple[float, Path] | None:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            return None
        probe = probe.parent
    try:
        return float(shutil.disk_usage(probe).free), probe
    except OSError:
        return None


def _pages_lines(block: Block, model_id: str, sel: Selection, n: int, count: Any, pages: Any,
                 work_dir: Any) -> dict[str, Any]:
    plan: dict[str, Any] = {"n_pages": n}
    pick = sel.pick
    if count is not None:
        skipped = list(getattr(count, "skipped", []) or [])
        n_files = int(getattr(count, "n_files", 0))
        if skipped:
            block.add("folder", "{0} pages in {1} files; {2} files skipped",
                      claim(n, "counted", text=_claim.count(n)), claim(n_files, "counted", text=_claim.count(n_files)),
                      claim(len(skipped), "counted", text=_claim.count(len(skipped))))
            for path, reason in skipped[:5]:
                block.add(" ", "skipped {0}: {1}", lit(path), lit(reason))
        else:
            block.add("folder", "{0} pages in {1} files", claim(n, "counted", text=_claim.count(n)),
                      claim(n_files, "counted", text=_claim.count(n_files)))
        plan.update(n_files=n_files, skipped=[list(s) for s in skipped])

    def est(value: Any, text: str, basis: str = _EST_BYTES) -> _claim.Claim:
        return claim(value, "estimate", basis, "", text)

    f16 = next(r for r in sel.rows if r.name == "float16")
    index_bytes = None if pick.bytes_per_page is None else n * pick.bytes_per_page[1]
    temp_bytes = None if f16.bytes_per_page is None else n * f16.bytes_per_page[1]
    if index_bytes is not None:
        if pick.name in ("float16", "float32"):
            block.add("size", "about {0} of vectors", est(index_bytes, _claim.size(index_bytes)))
        else:
            block.add("size", "about {0} of vectors, {1} smaller than float16", est(index_bytes, _claim.size(index_bytes)),
                      est(pick.ratio_f16, _claim.ratio(pick.ratio_f16)))
    else:
        block.add("size", "not estimated: your model's bytes per page were not measured")
    plan.update(index_bytes=index_bytes, temporary_bytes=temp_bytes)

    estimate = None
    try:
        from . import _encode

        estimate = _encode.estimate(model_id, n)
    except (ImportError, KeyError, LookupError, ValueError):
        estimate = None
    if estimate is not None and getattr(estimate, "known", True):
        gpu = getattr(estimate, "gpu", None) or "a GPU like ours"
        lo_s, hi_s = float(estimate.lo_s), float(estimate.hi_s)
        block.add("encoding", "{0} on {1}", est((lo_s, hi_s), _claim.duration(lo_s, hi_s), str(estimate.basis)), lit(gpu))
        plan.update(encode_seconds=[lo_s, hi_s], encode_basis=str(estimate.basis))
    else:
        block.add("encoding", "not timed: our runs recorded no speed for this model")
    if count is not None:
        block.add("rendering", "not in our benchmark runs (their pages were images); build times it on your pages")

    where = Path(work_dir) if work_dir is not None else default_work_dir(pages)
    plan["work_dir"] = str(where)
    if temp_bytes is not None:
        block.add("temporary", "{0} for the float16 copy in {1}", est(temp_bytes, _claim.size(temp_bytes)), lit(where))
    else:
        block.add("temporary", "the float16 copy's size is not known for an unmeasured model; it goes to {0}", lit(where))
    free = _free_bytes(where)
    if free is not None:
        block.add("free", "{0} reported by that drive", claim(free[0], "counted", text=_claim.size(free[0])))
        plan["free_bytes"] = free[0]
        needed = (temp_bytes or 0.0) + (index_bytes or 0.0)
        if needed and free[0] < needed:
            block.add("disk", "not enough free disk there for the temporary copy and the index; build stops before "
                      "encoding unless work_dir= points to a larger drive")
    if temp_bytes is not None:
        block.add("if wrong", "re-encoding costs the same time; keep_float16=True keeps the {0} copy so a change "
                  "needs none", est(temp_bytes, _claim.size(temp_bytes)))
    posix = where.as_posix()
    if (_on_colab() and estimate is not None and float(estimate.hi_s) > 3600
            and posix.startswith("/content") and not posix.startswith("/content/drive")):
        block.add("warning", "A Colab disconnect loses the temporary copy in /content. Pass work_dir= on Drive to "
                  "resume in a new session.")
    return plan


# ---------------------------------------------------------------------------- model=None


def _overview(t: float, notes: list[str]) -> Options:
    from ..adapters import MODELS
    from . import _resolve

    data = _levels.data()
    runs = data["runs"]
    models: list[dict[str, Any]] = []
    for model_id, info in MODELS.items():
        rows_id, basis = _resolve.rows_model(model_id)
        own = model_id in data["models"]
        status = ("full rows" if own else "no rows" if info.status == "measured" else "not measured")
        rec: dict[str, Any] = {"model": model_id, "modality": "text" if info.modality == "text" else "page",
                               "loader": info.loader, "status": status, "licence": _resolve.licence(info),
                               "registry_status": info.status, "basis": basis, "measured_via": None}
        if basis.startswith("measured via"):
            rec["measured_via"] = rows_id
        if own:
            entry = data["models"][model_id]
            cats = _catalogue_datasets(model_id)
            rec["baselines"] = {d: entry["datasets"][d]["baseline"] for d in cats}
            f16 = [entry["rungs"]["float16"][d]["bytes_per_page"] for d in cats]
            rec["float16_bytes_per_page"] = [min(f16), max(f16)]
            rec["datasets"] = cats
        run = runs.get(model_id)
        if run is not None:
            rec["run"] = run
        models.append(rec)

    start = _start_here(models, t)
    block = Block([], title=None)
    rule = ("Start here: measured page models with the sentence-transformers loader, a recorded apache-2.0 or "
            "MIT licence, encoded in float16 on one GPU in our runs; the best by full-precision quality, plus "
            "the quickest to encode")
    block.add("", "{0}", lit(rule))
    for s in start:
        _start_line(block, s, t)
    block.add("", "{0}", lit(f"All {len(models)} registry models (id, modality, loader, licence, status):"))
    for rec in models:
        _model_lines(block, rec)
    block.add("status", "{0}", lit(_ledger.status()))
    quick = next((s for s in start if s.get("quickest")), start[0] if start else None)
    block.next = (f'ov.options("{quick["model"]}", target={t:g})' if quick else f"ov.options(model, target={t:g})")
    return Options(model=None, target=t, n_pages=None, block=block, models=models, start_here=start, notes=notes)


def _quality(rec: dict[str, Any]) -> float:
    b = list(rec.get("baselines", {}).values())
    return sum(b) / len(b) if b else 0.0


def _speed(rec: dict[str, Any]) -> float:
    s = (rec.get("run") or {}).get("seconds_per_unit")
    return sum(s) / len(s) if s else float("inf")


def _start_here(models: list[dict[str, Any]], t: float) -> list[dict[str, Any]]:
    """FINAL_DESIGN 2 (0): measured page models on the sentence-transformers loader with a recorded
    apache-2.0 or MIT licence, encoded in float16 without multi-GPU in our runs; the top three by
    full-precision quality (mean over the catalogue datasets), plus the quickest to encode."""
    ok = [r for r in models if r["status"] == "full rows" and r["modality"] == "page"
          and r["loader"] == "sentence-transformers" and r["licence"].lower() in ("apache-2.0", "mit")
          and (r.get("run") or {}).get("dtype") == "float16" and not (r.get("run") or {}).get("multi_gpu", True)
          and (r.get("run") or {}).get("seconds_per_unit")]
    top = sorted(ok, key=lambda r: (-_quality(r), r["model"]))[:3]
    quick = min(ok, key=lambda r: (_speed(r), r["model"])) if ok else None
    picked = list(top) + ([quick] if quick is not None and quick not in top else [])
    out = []
    for r in picked:
        sel = _levels.select(r["model"], t, None)
        out.append({"model": r["model"], "baselines": r["baselines"], "datasets": r["datasets"],
                    "seconds_per_page": r["run"]["seconds_per_unit"], "gpu": r["run"].get("gpu"),
                    "pick": sel.pick.name, "pick_plain": sel.pick.plain, "ratio_f16": sel.pick.ratio_f16,
                    "close_call": sel.close_call, "quickest": quick is not None and r is quick})
    return out


def _run_source(datasets: list[str]) -> str:
    return f"{_levels.source(datasets)}; encode times from our Kaggle runs"


def _start_line(block: Block, s: dict[str, Any], t: float) -> None:
    src = _run_source(s["datasets"])

    def c(value: Any, text: str) -> _claim.Claim:
        return claim(value, "catalogue", src, "", text)

    base = [c(s["baselines"][d], f"{s['baselines'][d]:.3f}") for d in s["datasets"]]
    lo, hi = s["seconds_per_page"]
    name = s["model"] + (" (quickest)" if s["quickest"] else "")
    k = len(base)
    secs, sparts = _span(lo, hi, _secs, src, k + 1)
    j = k + 1 + len(sparts)
    tmpl = "{0} nDCG@5 " + " / ".join("{%d}" % (i + 1) for i in range(k))
    tmpl += ", " + secs + " s/page, at {%d}: {%d} {%d}" % (j, j + 1, j + 2)
    block.add("", tmpl, lit(f"  {name:<48}"), *base, *sparts, lit(_pct(t)),
              lit(s["pick_plain"]), c(s["ratio_f16"], _claim.ratio(s["ratio_f16"])))


def _model_lines(block: Block, rec: dict[str, Any]) -> None:
    status = rec["status"] + (f" (measured via {rec['measured_via']})" if rec.get("measured_via") else "")
    block.add("", "{0}", lit(f"  {rec['model']}  {rec['modality']}, {rec['loader']}, licence {rec['licence']}, {status}"))
    if "baselines" in rec:
        src = _levels.source(rec["datasets"])
        metric = _metric(rec["modality"])

        def c(value: Any, text: str) -> _claim.Claim:
            return claim(value, "catalogue", src, "", text)

        tbl = _levels.data()["datasets"]
        parts: list[Any] = []
        pieces = []
        for i, d in enumerate(rec["datasets"]):
            pieces.append("{%d} {%d}" % (2 * i, 2 * i + 1))
            parts += [lit(tbl[d]["label"]), c(rec["baselines"][d], f"{rec['baselines'][d]:.3f}")]
        lo, hi = rec["float16_bytes_per_page"]
        unit = _unit(rec["modality"])
        j = len(parts)
        size = "{%d}-{%d}" % (j, j + 1) if _mb(lo) != _mb(hi) else "{%d}" % j
        parts += [c(lo, _mb(lo))] + ([c(hi, _mb(hi))] if _mb(lo) != _mb(hi) else [])
        block.add("", "    full precision " + metric + ": " + ", ".join(pieces) + "; float16 " + size
                  + " MB per thousand " + unit, *parts)
    run = rec.get("run")
    if run and run.get("seconds_per_unit"):
        src = f"our Kaggle encoding logs, {_levels.source(rec.get('datasets', []))}" if rec.get("datasets") else \
            "our Kaggle encoding logs"
        lo, hi = run["seconds_per_unit"]
        where = (f"2x {run['gpu']}" if run.get("multi_gpu") else f"one {run['gpu']}") if run.get("gpu") else \
            "GPU not recorded"
        dtype = run.get("dtype") or "dtype not recorded"
        per = "page" if run.get("unit") == "page" else "document"
        secs, sparts = _span(lo, hi, _secs, src, 2)
        block.add("", "    ran on {0}, {1}, " + secs + " s/" + per, lit(where), lit(dtype), *sparts)
    else:
        block.add("", "    ran on: not recorded")
