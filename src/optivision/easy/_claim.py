"""The one formatter every easy-mode number passes through (FINAL_DESIGN section 7).

A number shown to a user is a :class:`Claim`: a value with exactly one evidence kind.

==================  =============================  ===========================
kind                tag shown                      may change the layout
==================  =============================  ===========================
``catalogue``       ``[our benchmarks: <source>]`` sets the ceiling
``probe``           ``[questions from your PDFs]`` safer only
``user-queries``    ``[your questions: <what>]``   safer; more only by upgrade
``counted``         ``[counted]``                  never
``estimate``        ``[estimate: <basis>]``        never
==================  =============================  ===========================

A :class:`Line` renders a template whose ``{0}``, ``{1}``... placeholders are filled with
claims or literals. All claims on a line must share one kind and one source; a line that
would mix them raises :class:`MixedEvidence` (split it into two lines). A template may not
contain a digit outside :data:`ALLOWED_DIGIT_TOKENS`, so a number cannot reach the user
without a tag (``tests/test_easy_honesty.py`` runs every template through this).

Rounding: retention and other quality numbers round *down*; a retention point above 1 whose
low is below 1 reads "no loss seen; the data allow down to X%". Sizes round to the nearest
tenth. Templates are ASCII.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

KINDS: dict[str, str] = {
    "catalogue": "our benchmarks",
    "probe": "questions from your PDFs",
    "user-queries": "your questions",
    "counted": "counted",
    "estimate": "estimate",
}

#: words with digits a template may contain without a claim: metric and codec names, GPU names,
#: test ids, page-id suffixes and the like
ALLOWED_DIGIT_TOKENS = re.compile(
    r"nDCG@\d+|recall@\d+|ndcg@\d+|\d-bit|float16|float32|bfloat16|int8|int4|fp16|fp32|"
    r"T4|A100|L4|V100|H100|PREREG_EASY|V\d+[a-z]?(?:-unchecked|u|t)?|B\d|1 in \d|ViDoRe V\d|"
    r"\{\d+\}|\[\d+/\d+\]|::p\d+|x\d+|\d+x(?= |$)"
)


class MixedEvidence(ValueError):
    """A line would carry claims of two kinds or two sources."""


class BareNumber(ValueError):
    """A template holds a digit that is not inside a claim or an allowed token."""


@dataclass(frozen=True)
class Claim:
    """A value with one evidence kind.

    ``source`` names where it comes from: datasets with page counts for ``catalogue``
    (``"ViDoRe V1 DocVQA 500 pages, InfoVQA 500 pages"``), the basis of an ``estimate``
    (``"our Kaggle run, 0.61-0.65 s/page"``), or what ``user-queries`` measured
    (``"agreement with full precision, not relevance"`` or ``"search quality"``).
    ``construct`` is what the number measures (``"nDCG@5 retention"``); it is kept for
    ``report()`` and never shown inline. ``text`` is the formatted value.
    """

    value: Any
    kind: str
    source: str = ""
    construct: str = ""
    text: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown evidence kind {self.kind!r}; known: {sorted(KINDS)}")
        if self.kind in ("catalogue", "estimate") and not self.source:
            raise ValueError(f"a {self.kind} claim needs a source")
        if not self.text:
            object.__setattr__(self, "text", _default_text(self.value))

    @property
    def tag(self) -> str:
        name = KINDS[self.kind]
        if self.kind in ("counted", "probe") or not self.source:
            return f"[{name}]"
        return f"[{name}: {self.source}]"

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "kind": self.kind, "source": self.source, "construct": self.construct,
                "text": self.text, "tag": self.tag}


@dataclass(frozen=True)
class Literal:
    """Text substituted into a template untouched and without a tag: a path, a model id, code."""

    text: str


def claim(value: Any, kind: str, source: str = "", construct: str = "", text: str = "") -> Claim:
    return Claim(value, kind, source, construct, text)


def lit(text: Any) -> Literal:
    return Literal(str(text))


# ---------------------------------------------------------------------------- number formats


def _floor(x: float, digits: int) -> float:
    scale = 10**digits
    return math.floor(x * scale + 1e-9) / scale


def pct(x: float, digits: int = 1) -> str:
    """A share as a percentage, rounded down: 0.97289 -> ``"97.2%"``."""
    return f"{_floor(100 * float(x), digits):.{digits}f}%"


def retention(point: float, low: float | None = None, digits: int = 1) -> str:
    """A retention, rounded down; above 100% with a low below it, "no loss seen"."""
    if point > 1.0 and low is not None and low < 1.0:
        return f"no loss seen; the data allow down to {pct(low, digits)}"
    if point > 1.0:
        return "no loss seen"
    return pct(point, digits)


def ratio(r: float) -> str:
    """A compression ratio, rounded down to a tenth: ``"7.9x"``."""
    return f"{_floor(float(r), 1):.1f}x"


def size(nbytes: float) -> str:
    """Bytes in decimal units, to a tenth: ``"10.6 MB"``."""
    n = float(nbytes)
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{int(n)} B"


def count(n: int) -> str:
    """An integer with thousands separators: ``"8,400"``."""
    return f"{int(n):,}"


def duration(lo: float, hi: float | None = None) -> str:
    """Seconds as a readable span: ``"about 1.4-1.5 h"``, ``"about 5 min"``, ``"0.2 s"``."""
    hi = lo if hi is None else hi

    def one(s: float, unit: str) -> str:
        if unit == "h":
            return f"{s / 3600:.1f}"
        if unit == "min":
            return f"{max(1, round(s / 60))}"
        return f"{s:.1f}"

    unit = "h" if hi >= 5400 else "min" if hi >= 90 else "s"
    a, b = one(lo, unit), one(hi, unit)
    if unit == "s" and a == b:
        return f"{a} s"
    return f"about {a} {unit}" if a == b else f"about {a}-{b} {unit}"


def _default_text(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return count(value)
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


# ---------------------------------------------------------------------------- lines and blocks


def check_template(template: str) -> None:
    """Raise :class:`BareNumber` if the template holds a digit outside an allowed token."""
    stripped = ALLOWED_DIGIT_TOKENS.sub("", template)
    if re.search(r"\d", stripped):
        raise BareNumber(f"template {template!r} holds a number that is not a claim; pass it as a claim")


def _kinds(parts: tuple[Claim | Literal, ...]) -> list[Claim]:
    claims = [p for p in parts if isinstance(p, Claim)]
    if len({(c.kind, c.source) for c in claims}) > 1:
        raise MixedEvidence("one line, two kinds of evidence: " + "; ".join(sorted({c.tag for c in claims})))
    return claims


@dataclass(frozen=True)
class Line:
    """``label`` (left column), ``template`` with ``{n}`` placeholders, and its parts.

    Rendered as ``label  text  [tag]``; the tag of the line's claims goes once at the end.
    """

    label: str
    template: str
    parts: tuple[Claim | Literal, ...] = ()

    def __init__(self, label: str, template: str, *parts: Claim | Literal | str) -> None:
        check_template(template)
        clean = tuple(lit(p) if isinstance(p, str) else p for p in parts)
        for p in clean:
            if not isinstance(p, (Claim, Literal)):
                raise TypeError(f"a line part must be a Claim or a Literal, got {type(p).__name__}; use claim() or lit()")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "parts", clean)
        _kinds(clean)

    @property
    def claims(self) -> list[Claim]:
        return [p for p in self.parts if isinstance(p, Claim)]

    @property
    def tag(self) -> str:
        claims = self.claims
        return claims[0].tag if claims else ""

    def text(self) -> str:
        return self.template.format(*[p.text for p in self.parts])

    def render(self, width: int = 11) -> str:
        body = self.text()
        tag = self.tag
        out = f"{self.label:<{width}}{body}" if self.label else body
        return f"{out}  {tag}" if tag else out


@dataclass
class Block:
    """Lines plus the closing ``Next:`` line every easy-mode output ends with."""

    lines: list[Line]
    next: str | None = None
    title: str | None = None

    def add(self, label: str, template: str, *parts: Claim | Literal | str) -> Block:
        self.lines.append(Line(label, template, *parts))
        return self

    def claims(self) -> list[Claim]:
        return [c for line in self.lines for c in line.claims]

    def __str__(self) -> str:
        width = max([len(line.label) for line in self.lines if line.label] + [9]) + 2
        out = [self.title] if self.title else []
        out += [line.render(width) for line in self.lines]
        if self.next:
            out.append(f"Next: {self.next}")
        return "\n".join(out)

    def as_dict(self) -> dict[str, Any]:
        return {"title": self.title, "next": self.next,
                "lines": [{"label": ln.label, "text": ln.text(), "tag": ln.tag,
                           "claims": [c.as_dict() for c in ln.claims]} for ln in self.lines]}
