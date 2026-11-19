"""The claim lint of easy mode (FINAL_DESIGN 7, rule 5), shared by the honesty, chart, report and glossary tests.

Two lists of banned wording:

* ``ALWAYS``: the unambiguous words and phrases. They are flagged wherever they appear as words, so
  a path or an identifier that merely contains one (``optivision-universal``, ``reports/universal/``,
  ``_veto.py``, ``first_page``) passes.
* ``CLAIM_ONLY``: ambiguous words, flagged only inside a claim pattern ("the first", "first to"), so
  ordinary sentences ("the first 50 pages are timed") are not flagged.

The words come from ``tests/test_docs_examples.py`` (``BANNED_PHRASES``, ``BANNED_WORDS``) and the
brief for easy mode ("lossless", "universal", "guaranteed", "first", "SOTA", "state of the art", "veto").
"""

from __future__ import annotations

import re

# a word edge that treats path and identifier characters as part of the word
_B = r"(?<![\w/\\.-])"
_E = r"(?![\w/\\-])(?!\.\w)"

ALWAYS: dict[str, re.Pattern[str]] = {
    name: re.compile(_B + pattern + _E, re.IGNORECASE)
    for name, pattern in {
        "lossless": r"lossless",
        "universal": r"universal(?:ly)?",
        "guaranteed": r"guarantee[ds]?",
        "SOTA": r"sota",
        "state of the art": r"state[- ]of[- ]the[- ]art",
        "veto": r"vetoe?[sd]?",
        "novel": r"novel",
        "better than": r"better than",
        "works with every model": r"works with every model",
        "scales to": r"scales to",
        "faster": r"faster",
        "simply": r"simply",
        "just": r"just",
    }.items()
}

CLAIM_ONLY: dict[str, re.Pattern[str]] = {
    "first (claim)": re.compile(
        r"\b(?:the|our|a|an) first (?:tool|method|system|package|library|compressor|compression|approach|solution|"
        r"framework|encoder|layer|release|one)\b|\bfirst (?:to|ever|of its kind)\b|\bfirst[- ]ever\b|"
        r"\b(?:world|industry)'?s first\b",
        re.IGNORECASE),
}


def banned_hits(text: str) -> list[str]:
    """Every banned word or claim phrase in ``text``, as ``"<name>: ...snippet..."``."""
    hits = []
    for table in (ALWAYS, CLAIM_ONLY):
        for name, pattern in table.items():
            for m in pattern.finditer(text):
                lo, hi = max(0, m.start() - 20), min(len(text), m.end() + 20)
                hits.append(f"{name}: ...{text[lo:hi]!r}...")
    return hits


def non_ascii(text: str) -> list[str]:
    return sorted({ch for ch in text if ord(ch) > 127})
