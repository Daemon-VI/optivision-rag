"""The report of an index (FINAL_DESIGN 4.4): every number with its evidence tag, saved as a page, notes or data.

A :class:`Report` is a list of titled sections. Each section is a :class:`~optivision.easy._claim.Block`,
so every number in it is a :class:`~optivision.easy._claim.Claim` and every line prints with the one
tag of its claims. Two sections have a fixed place:

* ``Report.FLOAT32`` ("Ratios against float32"): the README of the package states ratios against float32,
  easy mode against float16; this section gives the float32 ones so the two can be compared;
* ``Report.GEOMETRY`` ("Geometry diagnostics"): the sm4 and nsr diagnostics, which flag risk for plain
  int4 only and never change a layout.

A report always lists both. A section that was not supplied reads "Not recorded for this index."

    report = Report([("Build", block), ("Skipped files", block2)], meta={"model": "vidore/colqwen2-v1.0-hf"})
    print(report)
    report.save("report.html")      # one self-contained page, A4 when printed
    report.save("report.md")
    report.save("report.json")

``meta`` holds identification (model id, index path, versions, dates), as text. A bare number in it is
refused: wrap it in ``claim(...)`` so that it carries a tag.
"""

from __future__ import annotations

import html as _html
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._claim import Block, Claim, Line, Literal, lit

SUFFIXES = {".html": "html", ".htm": "html", ".md": "md", ".markdown": "md", ".json": "json"}

DEFAULT_NEXT = "index.check()   (re-tests the index; nothing changes unless you ask)"

_CSS = """
:root{color-scheme:light dark;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--chip:#f1f0ec}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--chip:#2a2a28}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 6px}h2{font-size:17px;margin:28px 0 6px;padding-bottom:4px;border-bottom:1px solid var(--line)}
dl.meta{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;margin:10px 0;color:var(--ink2);font-size:14px}
dl.meta dt{font-weight:600}dl.meta dd{margin:0;overflow-wrap:anywhere}
table.lines{border-collapse:collapse;width:100%;font-size:14px}
table.lines th,table.lines td{text-align:left;vertical-align:top;padding:4px 8px 4px 0;border-bottom:1px solid var(--line)}
table.lines th{width:11em;font-weight:600;color:var(--ink2)}
.tag{display:inline-block;padding:0 6px;border-radius:4px;background:var(--chip);color:var(--ink2);font-size:12px;
 font-family:ui-monospace,Menlo,Consolas,monospace;overflow-wrap:anywhere}
.next{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;margin-top:24px;overflow-wrap:anywhere}
@page{size:A4;margin:15mm}
@media print{body{background:#fff;color:#000;font-size:11pt}main{max-width:none;padding:0}
 section.sec{break-inside:avoid;page-break-inside:avoid}h2{break-after:avoid;page-break-after:avoid}
 .tag{background:none;border:1px solid #888;color:#000}dl.meta,table.lines th{color:#000}}
"""


def _ascii(text: str) -> str:
    return text.encode("ascii", "xmlcharrefreplace").decode("ascii")


def _esc(value: Any) -> str:
    return _ascii(_html.escape(str(value), quote=True))


def _meta_text(key: str, value: Any) -> str:
    """The text of a meta value; a number must be a claim so that it has a tag."""
    if isinstance(value, Claim):
        return f"{value.text}  {value.tag}"
    if isinstance(value, Literal):
        return value.text
    if value is None or isinstance(value, (str, bool)):
        return str(value)
    raise TypeError(f"meta[{key!r}] is a {type(value).__name__}; a number needs its evidence tag, "
                    f"so pass claim(value, kind, ...) or the text of it")


class Report:
    """Titled sections of lines, each number with its tag; ``save`` writes .html, .md or .json."""

    FLOAT32 = "Ratios against float32"
    GEOMETRY = "Geometry diagnostics"
    SLOTS = (FLOAT32, GEOMETRY)

    def __init__(self, sections: Iterable[tuple[str, Block]], meta: dict[str, Any] | None = None, *,
                 title: str = "OptiVision report", next: str | None = None) -> None:
        given = list(sections)
        for item in given:
            if (not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
                    or not isinstance(item[1], Block)):
                raise TypeError("a section is a (title, Block) pair; build the Block with Line/Block.add so that "
                                "every number is a claim")
        self._given: list[tuple[str, Block]] = given
        self.meta: dict[str, Any] = dict(meta or {})
        for key, value in self.meta.items():
            _meta_text(key, value)                       # refuses a bare number now, not at save time
        self.title = title
        self.next = DEFAULT_NEXT if next is None else next

    # -- structure

    @property
    def sections(self) -> list[tuple[str, Block]]:
        """The supplied sections in order, then the fixed slots that were not supplied."""
        have = {t.lower() for t, _ in self._given}
        extra = [(slot, Block([Line("", "{0}", lit("Not recorded for this index."))]))
                 for slot in self.SLOTS if slot.lower() not in have]
        return [*self._given, *extra]

    def section(self, title: str) -> Block:
        for t, block in self.sections:
            if t.lower() == title.lower():
                return block
        raise KeyError(f"no section {title!r}; sections: " + ", ".join(t for t, _ in self.sections))

    def claims(self) -> list[Claim]:
        return [c for _, block in self.sections for c in block.claims()]

    # -- text

    @property
    def lines(self) -> list[str]:
        out = [self.title]
        for key, value in self.meta.items():
            out.append(f"  {key}: {_meta_text(key, value)}")
        for title, block in self.sections:
            out += ["", title, "-" * len(title)]
            width = max([len(ln.label) for ln in block.lines if ln.label] + [9]) + 2
            out += [ln.render(width) for ln in block.lines]
        out += ["", f"Next: {self.next}"]
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines)

    __repr__ = __str__

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "meta": {k: ({"text": v.text, "tag": v.tag, "claim": v.as_dict()} if isinstance(v, Claim)
                         else _meta_text(k, v)) for k, v in self.meta.items()},
            "sections": [{"title": t, "lines": block.as_dict()["lines"]} for t, block in self.sections],
            "next": self.next,
        }

    # -- files

    def to_html(self) -> str:
        """One self-contained page: no script, no font or image from a network; A4 when printed."""
        rows = [f"<h1>{_esc(self.title)}</h1>"]
        if self.meta:
            rows.append("<dl class=\"meta\">" + "".join(
                f"<dt>{_esc(k)}</dt><dd>{_esc(_meta_text(k, v))}</dd>" for k, v in self.meta.items()) + "</dl>")
        for i, (title, block) in enumerate(self.sections, 1):
            body = []
            for ln in block.lines:
                tag = ""
                if ln.claims:
                    kind = ln.claims[0].kind
                    tag = f'<span class="tag k-{_esc(kind)}">{_esc(ln.tag)}</span>'
                label = f"<th scope=\"row\">{_esc(ln.label)}</th>" if ln.label else "<th></th>"
                body.append(f"<tr>{label}<td>{_esc(ln.text())}</td><td>{tag}</td></tr>")
            rows.append(f'<section class="sec" id="s{i}"><h2>{_esc(title)}</h2>'
                        f'<table class="lines">{"".join(body)}</table></section>')
        rows.append(f'<p class="next">Next: {_esc(self.next)}</p>')
        return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                f"<title>{_esc(self.title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
                + "\n".join(rows) + "\n</main>\n</body>\n</html>\n")

    def to_markdown(self) -> str:
        out = [f"# {self.title}", ""]
        for key, value in self.meta.items():
            out.append(f"- {key}: {_meta_text(key, value)}")
        if self.meta:
            out.append("")
        for title, block in self.sections:
            out += [f"## {title}", ""]
            for ln in block.lines:
                label = f"**{ln.label}** " if ln.label else ""
                tag = f" `{ln.tag}`" if ln.tag else ""
                out.append(f"- {label}{ln.text()}{tag}")
            out.append("")
        out.append(f"Next: {self.next}")
        return "\n".join(out) + "\n"

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=True) + "\n"

    def _repr_html_(self) -> str:
        return f'<pre style="white-space:pre-wrap">{_esc(str(self))}</pre>'

    def save(self, path: str | Path) -> Path:
        """Write the report; the format follows the suffix (.html, .md or .json). Returns the path."""
        target = Path(path)
        kind = SUFFIXES.get(target.suffix.lower())
        if kind is None:
            raise ValueError(f"cannot save a report as {target.suffix or target.name!r}; use .html, .md or .json")
        text = {"html": self.to_html, "md": self.to_markdown, "json": self.to_json}[kind]()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return target

    # -- hook

    @classmethod
    def from_index(cls, index: Any) -> Report:
        """The report of an index. The index supplies ``_report_parts() -> (sections, meta)``; this class only
        lays them out, so what a report says is decided where the index's records are read."""
        parts = getattr(index, "_report_parts", None)
        if parts is None:
            raise NotImplementedError("this index has no _report_parts(); the build module supplies the sections "
                                      "and meta that Report lays out")
        sections, meta = parts()
        return cls(sections, meta)
