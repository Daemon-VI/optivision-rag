"""The glossary of easy mode (FINAL_DESIGN 6.5): fourteen terms, each with one fixed meaning.

Every output explains a term in one line when the term first appears. ``TERMS`` holds the meanings
exactly as the design table words them; nothing else in the package may define them again.

    >>> from optivision.easy import glossary
    >>> print(glossary("target"))          # one term
    >>> print(glossary())                  # all of them

``note(term, seen)`` is the one-line note an output prints once; ``notes_for(text, seen)`` finds the
known terms in a piece of output and returns the notes that are still due. The caller owns
``seen`` (a set), so two outputs in one session do not silence each other unless they share it.

Meanings hold constants of the design (a 95% range, the top 5, 60 text pages), not measured
values, so they are passed to the formatter as literals; measured numbers never go in here.
"""

from __future__ import annotations

import re

from ._claim import Block, Line, lit

#: term -> meaning shown to users (FINAL_DESIGN 6.5, in the table's order)
TERMS: dict[str, str] = {
    "layout": "How stored vectors are compressed",
    "target": "Share of full-precision search quality (nDCG@5) to keep; not \"answers correct\"",
    "nDCG@5": "0-1 score of how high the right pages rank in the top 5",
    "full precision": "The model's original vectors",
    "float16": "Uncompressed, 2 bytes per number; our size baseline",
    "8-bit / 4-bit / 1-bit": "Fewer bits per number; centred 4-bit subtracts the average first",
    "merged (1 in 4)": "Similar vectors on a page combined, keeping about a quarter",
    "lowest kept": "Bottom of the 95% range we measured, on our worst dataset",
    "close call": "Cleared the target by under half a point",
    "practice questions": "Made from your PDFs' text; easier than real ones; only make us pick safer",
    "practice check": "Testing a layout on practice questions before saving",
    "unchecked": "No practice check possible (scans, images, under 60 text pages, vectors); no 1-bit layouts",
    "drift": "Your model now producing different vectors than at build time",
    "build record": "Model, versions and settings saved with the index",
}

#: other spellings that resolve to a term (lower case)
ALIASES: dict[str, str] = {
    "ndcg": "nDCG@5",
    "ndcg@5": "nDCG@5",
    "full-precision": "full precision",
    "float32": "full precision",
    "8-bit": "8-bit / 4-bit / 1-bit",
    "4-bit": "8-bit / 4-bit / 1-bit",
    "1-bit": "8-bit / 4-bit / 1-bit",
    "centred 4-bit": "8-bit / 4-bit / 1-bit",
    "centered 4-bit": "8-bit / 4-bit / 1-bit",
    "int8": "8-bit / 4-bit / 1-bit",
    "int4": "8-bit / 4-bit / 1-bit",
    "merged": "merged (1 in 4)",
    "merge": "merged (1 in 4)",
    "merged (1 in 3)": "merged (1 in 4)",
    "practice question": "practice questions",
    "accuracy": "target",
    "lowest": "lowest kept",
}

#: the words a text is scanned for in ``notes_for`` (a term and its aliases, longest first)
_PHRASES: list[tuple[str, str]] = sorted(
    [(t.lower(), t) for t in TERMS if "/" not in t and "(" not in t]
    + [(a, t) for a, t in ALIASES.items()]
    + [("8-bit", "8-bit / 4-bit / 1-bit"), ("merged (1 in", "merged (1 in 4)")],
    key=lambda p: -len(p[0]),
)


class Glossary(Block):
    """A :class:`Block` that prints as itself in a notebook cell and a terminal."""

    def __repr__(self) -> str:
        return str(self)


def _key(term: str) -> str:
    """The key of ``TERMS`` for a user's spelling, or raise ``KeyError`` naming the known terms."""
    if term in TERMS:
        return term
    low = " ".join(str(term).lower().split())
    for known in TERMS:
        if known.lower() == low:
            return known
    if low in ALIASES:
        return ALIASES[low]
    raise KeyError(f"no glossary term {term!r}; known terms: " + ", ".join(sorted(TERMS)))


def meaning(term: str) -> str:
    """The meaning of one term (``KeyError`` if unknown)."""
    return TERMS[_key(term)]


def glossary(term: str | None = None) -> Glossary:
    """The meaning of ``term``, or of every term when it is None, as a printable block."""
    keys = list(TERMS) if term is None else [_key(term)]
    block = Glossary(lines=[], title="Terms used in easy mode" if term is None else None)
    for key in keys:
        block.lines.append(Line(key, "{0}", lit(TERMS[key])))
    block.next = ('ov.options("MODEL_ID") shows the layouts of a model' if term is None
                  else "ov.glossary() lists every term")
    return block


def note(term: str, seen: set[str] | None = None) -> str:
    """The one-line note for a term's first appearance: ``"lowest kept: Bottom of ..."``.

    With a ``seen`` set the note is returned once and ``""`` afterwards; without one it is
    always returned.
    """
    key = _key(term)
    if seen is not None:
        if key in seen:
            return ""
        seen.add(key)
    return f"{key}: {TERMS[key]}"


def notes_for(text: str, seen: set[str] | None = None) -> list[str]:
    """Notes for the glossary terms that appear in ``text`` and are not yet in ``seen``, in the order
    they appear."""
    low = text.lower()
    found: list[tuple[int, str]] = []
    for phrase, key in _PHRASES:
        m = re.search(r"(?<![\w-])" + re.escape(phrase) + r"s?(?![\w-])", low)
        if m:
            found.append((m.start(), key))
    out: list[str] = []
    done: set[str] = set()
    for _, key in sorted(found):
        if key in done:
            continue
        done.add(key)
        line = note(key, seen)
        if line:
            out.append(line)
    return out

