"""Pieces shared by the optional practice-question tiers (OCR, local VLM, API LLM).

Tier A (text-layer questions, ``_probes.py``) writes ``Probe`` objects; the tiers here write the
same type, so every tier feeds the one practice check, which can only move a layout to a safer one
(FINAL_DESIGN 5.3, 5.4). What the tiers share:

* a seeded page sample, so the same folder and seed give the same pages;
* a question style plan (type x format) in the vocabulary of ViDoRe V3 (arXiv 2601.08620, Table 5):
  seven query types and three formats. The proportions below are our choice, not the benchmark's
  counts; one page cannot support the multi-hop type, so it is not planned;
* the filters: a cap on the share of a question's content words that the page text already
  contains (probes that copy page words read easier than real questions, FINAL_DESIGN 5.4 item 4),
  duplicates, and questions that name their own source ("this page");
* the relevant set: the page, its aliases (identical renders) and, where text exists, the other
  pages that hold at least 80% of the question's content words, as in Tier A.

Nothing here imports torch, PIL or the ``_probes`` module at module level.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._claim import Block, claim

#: pages with fewer text-layer characters than this count as "no text layer" (FINAL_DESIGN 5.2)
NO_TEXT_CHARS = 20

#: the single-page query types of ViDoRe V3 and the weights we plan with (see the module docstring)
QUERY_TYPES: dict[str, str] = {
    "extractive": "asks for one specific fact that the page states",
    "open-ended": "asks for an explanation or a description of something the page covers",
    "numerical": "asks for a number, an amount, a date or a percentage that the page gives",
    "boolean": "is a yes or no question that the page settles",
    "enumerative": "asks for a list of the items that share a property on the page",
    "compare-contrast": "asks how two things on the page differ or compare",
}
DEFAULT_TYPE_WEIGHTS: dict[str, int] = {
    "extractive": 30, "open-ended": 25, "numerical": 15, "boolean": 10, "enumerative": 10, "compare-contrast": 10,
}
QUERY_FORMATS: dict[str, str] = {
    "question": "a complete question that ends with a question mark",
    "keyword": "a short search phrase of two to six words, without a verb",
    "instruction": "an instruction to a search system, such as 'List ...' or 'Explain ...'",
}
DEFAULT_FORMAT_WEIGHTS: dict[str, int] = {"question": 60, "keyword": 25, "instruction": 15}

#: a yes or no question cannot be a keyword phrase
_FORCED_FORMAT = {"boolean": "question"}

STOPWORDS = frozenset(
    "a about above after again all also am an and any are as at be because been before being below between both but by "
    "can could did do does doing down during each few for from further had has have having he her here hers him his "
    "how i if in into is it its itself me more most my no nor not of off on once only or other our out over own same "
    "she should so some such than that the their them then there these they this those through to too under until up "
    "very was we were what when where which while who whom why will with would you your yours".split()
)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SELF_REFERENCE = re.compile(
    r"\b(?:the|this|that|above|following)\s+(?:summary|page|image|picture|document|passage|text|screenshot|scan)\b"
    r"|\bimage\s+\d\b|\bpage\s+\d\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------- the Probe type


def probe_class() -> Any:
    """The ``Probe`` type of Tier A, imported when needed (that module is written elsewhere)."""
    from ._probes import Probe

    return Probe


def make_probe(*, id: str, text: str, family: str, page_id: str, relevant: Iterable[str]) -> Any:
    return probe_class()(id=id, text=text, family=family, page_id=page_id, relevant=frozenset(relevant))


def stable_id(prefix: str, *parts: object) -> str:
    """A probe id that depends only on its parts (never on Python's randomised ``hash``)."""
    digest = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


# ---------------------------------------------------------------------------- pages


def page_id(page: Any) -> str:
    return str(page.id)


def page_text(page: Any, texts: Mapping[str, str] | None = None) -> str:
    """The text we know for a page: an explicit mapping (OCR text) wins over the text layer."""
    if texts is not None:
        found = texts.get(page_id(page))
        if found:
            return found
    return str(getattr(page, "text", "") or "")


def has_text_layer(page: Any) -> bool:
    chars = getattr(page, "n_chars", None)
    if chars is None:
        chars = len(str(getattr(page, "text", "") or ""))
    return int(chars) >= NO_TEXT_CHARS


def is_blank(page: Any) -> bool:
    return bool(getattr(page, "blank", False))


def get_image(page: Any, images: Any = None) -> Any:
    """The page image: from a mapping (page id -> image), a callable (page -> image) or the source file.

    With ``images=None`` the page is re-rendered from ``page.file`` and ``page.page`` by ``_docs.render_page``.
    """
    if images is None:
        from . import _docs

        return _docs.render_page(page.file, page.page)
    if callable(images):
        return images(page)
    return images[page_id(page)]


def prepare_image(image: Any, max_side: int) -> Any:
    """RGB and no longer than ``max_side`` pixels on the long edge (what a model or provider receives)."""
    from ..ingest import downscale

    return downscale(image.convert("RGB"), max_side)


def downscaled_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """The size ``prepare_image`` gives an image of this size, without resizing anything."""
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, int(width * scale)), max(1, int(height * scale))


def select_pages(pages: Sequence[Any], *, max_pages: int, seed: int,
                 predicate: Callable[[Any], bool] | None = None) -> list[Any]:
    """A seeded sample of at most ``max_pages`` eligible pages, kept in their original order.

    The sample is drawn over indices with ``random.Random(f"select:{seed}")``; string seeds are hashed
    with SHA-512 by the standard library, so the choice does not depend on the process.
    """
    eligible = [p for p in pages if predicate is None or predicate(p)]
    if max_pages <= 0:
        return []
    if len(eligible) <= max_pages:
        return eligible
    rng = random.Random(f"select:{seed}")
    keep = sorted(rng.sample(range(len(eligible)), max_pages))
    return [eligible[i] for i in keep]


def group_names(page_ids: Iterable[str]) -> list[str]:
    """Page ids as compact lines, one per file: ``"my_pdfs/a.pdf: pages 1-4, 9"``.

    An id is ``<doc_id>::p<page>`` (CONTRACT, "Ids and sources"); anything else is listed whole.
    """
    by_doc: dict[str, list[int]] = defaultdict(list)
    loose: list[str] = []
    for pid in page_ids:
        doc, sep, tail = str(pid).rpartition("::p")
        if sep and tail.isdigit():
            by_doc[doc].append(int(tail))
        else:
            loose.append(str(pid))
    lines = []
    for doc in sorted(by_doc):
        pages = sorted(set(by_doc[doc]))
        runs: list[str] = []
        start = prev = pages[0]
        for n in pages[1:]:
            if n == prev + 1:
                prev = n
                continue
            runs.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = n
        runs.append(str(start) if start == prev else f"{start}-{prev}")
        lines.append(f"{doc}: page{'s' if len(pages) > 1 else ''} {', '.join(runs)}")
    lines.extend(sorted(set(loose)))
    return lines


# ---------------------------------------------------------------------------- words and overlap


def content_words(text: str) -> list[str]:
    """Distinct lower-case words of a text without stop words, in order of appearance."""
    seen: dict[str, None] = {}
    for w in _WORD.findall(text.lower()):
        if len(w) >= 2 and w not in STOPWORDS:
            seen.setdefault(w, None)
    return list(seen)


def overlap(question: str, text: str) -> float | None:
    """Share of the question's content words that appear in ``text``; None if either side has none."""
    q = content_words(question)
    if not q or not text.strip():
        return None
    t = set(content_words(text))
    if not t:
        return None
    return sum(1 for w in q if w in t) / len(q)


class WordIndex:
    """Inverted index over page texts, to find the pages that hold most of a question's words."""

    def __init__(self, texts: Mapping[str, str]) -> None:
        self._words: dict[str, frozenset[str]] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        for pid, text in texts.items():
            words = frozenset(content_words(text))
            if not words:
                continue
            self._words[pid] = words
            for w in words:
                self._postings[w].add(pid)

    def holding(self, question: str, share: float = 0.8, cap: int = 15) -> list[str] | None:
        """Pages holding at least ``share`` of the question's content words (sorted), or None if more than ``cap``.

        None mirrors Tier A, which drops a question that is relevant to more than 15 pages.
        """
        words = content_words(question)
        if len(words) < 3:
            return []
        need = math.ceil(share * len(words) - 1e-9)
        rarest = min(words, key=lambda w: len(self._postings.get(w, ())))
        found = []
        for pid in sorted(self._postings.get(rarest, ())):
            have = self._words[pid]
            if sum(1 for w in words if w in have) >= need:
                found.append(pid)
        return None if len(found) > cap else found


def relevant_set(pid: str, aliases: Mapping[str, str] | None, extra: Iterable[str] = ()) -> frozenset[str]:
    """The page, every page with the same canonical id (identical renders), and ``extra`` pages."""
    out = {pid, *extra}
    if aliases:
        canon = aliases.get(pid, pid)
        out.add(canon)
        out.update(a for a, c in aliases.items() if c == canon)
    return frozenset(out)


# ---------------------------------------------------------------------------- question text


_QUOTES = "\"'`\u201c\u201d\u2018\u2019* "
_LABEL = re.compile(
    r"^(?:\d+[.)]\s*|[-*]\s+|(?:search\s+)?(?:question|query|phrase|instruction)\s*\d*\s*:\s*)", re.IGNORECASE)


def clean_question(raw: str | None, fmt: str = "question") -> str | None:
    """One clean line from a model reply, or None when no question can be read from it.

    Small models wrap the line in quotes, put a label in front ("Search query: ...") or add a full stop after
    the closing quote; all of that is removed, and the ending is made to fit the format.
    """
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.S | re.I).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    line = _LABEL.sub("", lines[0].strip(_QUOTES)).strip(_QUOTES)
    line = line.rstrip(".!:; ").strip(_QUOTES)
    if not 2 <= len(line.split()) <= 40:
        return None
    if fmt == "question":
        line = line.rstrip("?. ") + "?"
    elif fmt == "keyword":
        line = line.rstrip("?!:;. ")
    else:
        line = line.rstrip("? ")
    return line or None


def refers_to_source(question: str) -> bool:
    """True when the question names its own source ("the summary", "this page", "image 2")."""
    return bool(_SELF_REFERENCE.search(question))


def norm_key(question: str) -> str:
    return " ".join(_WORD.findall(question.lower()))


# ---------------------------------------------------------------------------- style plan and quotas


@dataclass(frozen=True)
class Style:
    type: str
    format: str


def allocate(n: int, weights: Mapping[str, int]) -> dict[str, int]:
    """Split ``n`` among keys by weight with largest remainders (ties by key order); sums to ``n``."""
    keys = sorted(weights)
    total = sum(weights[k] for k in keys)
    if n <= 0 or total <= 0:
        return {k: 0 for k in keys}
    raw = {k: n * weights[k] / total for k in keys}
    out = {k: int(math.floor(raw[k] + 1e-9)) for k in keys}
    short = n - sum(out.values())
    for k in sorted(keys, key=lambda k: (-(raw[k] - out[k]), k))[:short]:
        out[k] += 1
    return out


def plan_styles(n: int, seed: int, type_weights: Mapping[str, int] | None = None,
                format_weights: Mapping[str, int] | None = None) -> list[Style]:
    """``n`` (type, format) pairs whose counts follow the weights, in a seeded random order."""
    tw = dict(type_weights or DEFAULT_TYPE_WEIGHTS)
    fw = dict(format_weights or DEFAULT_FORMAT_WEIGHTS)
    for name in tw:
        if name not in QUERY_TYPES:
            raise ValueError(f"unknown question type {name!r}; known: {sorted(QUERY_TYPES)}")
    for name in fw:
        if name not in QUERY_FORMATS:
            raise ValueError(f"unknown question format {name!r}; known: {sorted(QUERY_FORMATS)}")
    types = [k for k, c in sorted(allocate(n, tw).items()) for _ in range(c)]
    formats = [k for k, c in sorted(allocate(n, fw).items()) for _ in range(c)]
    rng = random.Random(f"styles:{seed}")
    rng.shuffle(types)
    rng.shuffle(formats)
    return [Style(t, _FORCED_FORMAT.get(t, f)) for t, f in zip(types, formats, strict=True)]


def apply_quotas(items: Sequence[tuple[str, Any]], n_target: int, weights: Mapping[str, int] | None,
                 seed: int) -> list[Any]:
    """Choose at most ``n_target`` payloads so that question types follow the weights.

    ``items`` are ``(type, payload)`` pairs. Each type gets its share of ``n_target``; slots a type
    cannot fill (too few candidates) pass to the others in weight order. The result keeps the order
    of ``items``. With ``n_target`` at or above the number of items nothing is dropped.
    """
    if n_target >= len(items):
        return [p for _, p in items]
    w = dict(weights or DEFAULT_TYPE_WEIGHTS)
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, (t, _) in enumerate(items):
        by_type[t].append(i)
        w.setdefault(t, 1)
    rng = random.Random(f"quotas:{seed}")
    for t in sorted(by_type):
        rng.shuffle(by_type[t])
    slots = allocate(n_target, w)
    chosen: dict[str, list[int]] = {t: by_type[t][: slots.get(t, 0)] for t in by_type}
    left = n_target - sum(len(v) for v in chosen.values())
    order = sorted(by_type, key=lambda t: (-w[t], t))
    while left > 0:
        moved = False
        for t in order:
            if left > 0 and len(chosen[t]) < len(by_type[t]):
                chosen[t].append(by_type[t][len(chosen[t])])
                left -= 1
                moved = True
        if not moved:
            break
    keep = sorted(i for v in chosen.values() for i in v)
    return [items[i][1] for i in keep]


# ---------------------------------------------------------------------------- filters


@dataclass
class Rejections:
    """Counts of questions dropped, by reason (reported, never hidden)."""

    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, reason: str, n: int = 1) -> None:
        self.counts[reason] += n

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))

    def total(self) -> int:
        return sum(self.counts.values())


def check_question(question: str | None, *, text: str, max_overlap: float | None, seen: set[str],
                   rejected: Rejections, min_cap_words: int = 4) -> bool:
    """Apply the shared filters to one cleaned question; record the reason when it fails.

    ``text`` is the page text (text layer or OCR); with none, the overlap cap cannot be applied and
    the question passes it (the caller counts those pages). The cap also does not apply to a question with
    fewer than ``min_cap_words`` content words: a short on-topic question ("What is the visitor policy?")
    reuses the page's words without copying anything. Pass 1 to apply the cap to every question.
    """
    if question is None:
        rejected.add("no readable question")
        return False
    if refers_to_source(question):
        rejected.add("refers to its source")
        return False
    if max_overlap is not None and len(content_words(question)) >= min_cap_words:
        share = overlap(question, text)
        if share is not None and share > max_overlap:
            rejected.add("copies the page")
            return False
    key = norm_key(question)
    if key in seen:
        rejected.add("duplicate")
        return False
    seen.add(key)
    return True


# ---------------------------------------------------------------------------- result


@dataclass
class ProbeBatch:
    """What a tier returns: the probes, how many pages were tried, what was dropped and why.

    ``tier`` is ``"ocr"``, ``"vlm"`` or ``"api"``; the tier is also in every probe's family name
    (``ocr:span``, ``vlm:extractive``, ``api:numerical``), so reports can separate them.
    ``config`` is what the build record stores (model, revision, seed, prompt hash). ``plan`` is the API tier's
    :class:`ApiPlan` (also on a dry run); ``texts`` is the page text a tier read (OCR text, by page id), which the
    other tiers can take as their ``texts=``.
    """

    tier: str
    probes: list[Any]
    n_pages: int
    rejected: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    report: Block | None = None
    plan: Any = None
    texts: dict[str, str] = field(default_factory=dict)

    def families(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for p in self.probes:
            out[str(p.family)] += 1
        return dict(sorted(out.items()))

    def __str__(self) -> str:
        return str(self.report) if self.report is not None else f"{len(self.probes)} {self.tier} practice questions"


def add_counts(block: Block, label: str, counts: Mapping[str, int], lead: str | None = None) -> None:
    """One line of counts by name ("3 extractive, 2 numerical"), each count a tagged claim.

    With ``lead`` the line reads "<total> <lead> (3 extractive, 2 numerical)".
    """
    if not counts:
        return
    names = sorted(counts)
    if lead is None:
        template = ", ".join(f"{{{i}}} {name}" for i, name in enumerate(names))
        parts = [claim(int(counts[n]), "counted") for n in names]
    else:
        inner = ", ".join(f"{{{i + 1}}} {name}" for i, name in enumerate(names))
        template = f"{{0}} {lead} ({inner})"
        total = claim(sum(int(v) for v in counts.values()), "counted")
        parts = [total, *[claim(int(counts[n]), "counted") for n in names]]
    block.add(label, template, *parts)


TIERS = ("ocr", "vlm", "api")


def make_tier_probes(tier: str, pages: Sequence[Any], **kwargs: Any) -> ProbeBatch:
    """Run one optional tier by name: ``"ocr"``, ``"vlm"`` or ``"api"`` (the last needs ``allow_upload=True``)."""
    if tier == "ocr":
        from ._ocr import make_ocr_probes

        return make_ocr_probes(pages, **kwargs)
    if tier == "vlm":
        from ._probes_vlm import generate_vlm_probes

        return generate_vlm_probes(pages, **kwargs)
    if tier == "api":
        from ._probes_api import generate_api_probes

        return generate_api_probes(pages, **kwargs)
    raise ValueError(f"unknown practice-question tier {tier!r}; known: {list(TIERS)}")


def say(text: str) -> None:
    """Print a block; a console that cannot show a character in a file name gets "?" there instead of an error."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


#: label of a continuation line in a block: it indents the line under the label column
CONT = " "


def prompt_hash(*prompts: str) -> str:
    """Short hash of the prompt texts, stored in the build record so a probe set can be told apart."""
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
