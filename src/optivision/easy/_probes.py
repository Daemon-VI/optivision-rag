"""Tier A practice questions: made from a folder's text layer, on the CPU, with no language model
(FINAL_DESIGN 5.3.1, 5.4; docs/easy/CONTRACT.md, ``_probes.py``).

Practice questions copy words from the pages, so they are easier than real questions (R7 in
UNIVERSAL.md: fragment probes scored about 100% where real queries scored 81.6-93.7%). That is
why they are used one way only: a practice check can move a layout to a safer one, never to a
more compressed one, and a pass is never shown as confirmation.

Pipeline, all deterministic for one ``seed``:

1. :func:`make_probes` samples up to 600 unique text pages and makes up to 3 candidates per
   page from five families: a salient 8-25-word span (``span``), a span with its 2
   highest-IDF words removed keeping at least 6 (``anchor_dropped``), 2-5 keywords
   (``keywords``), a figure or table caption (``caption``), and a bigram shared by 2-15 pages
   (``bigram``). Header/footer lines, number-only spans and duplicates are dropped. A probe's
   relevant set is every page holding at least 80% of its content words, plus their
   repeated copies (aliases); a probe relevant to more than 15 pages is dropped.
2. Probes are encoded as queries (:func:`encode_probes` calls ``encode_queries``), never as
   documents: the R7 failure came from encoding fragments as documents.
3. :func:`hard_set` keeps a probe when one of its relevant pages is in the full-precision
   top 10 (no top-1 filter: that would delete the hard ones), scores its hardness
   ``g = (s_rel - s_out) / s_rel`` (best relevant score against the best other score; lower
   is harder), keeps the hardest probe of each page, and takes the hardest third, or 60 if
   that is more, capped at 300. The set is frozen and hashed before any layout is scored.
   The probes it does not use become ``index.examples``.

:func:`pool_size` is the cost control: if scoring would exceed 25% of the encoding time, the
pool shrinks, to no fewer than 200 pages x 3.
"""

from __future__ import annotations

import inspect
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from . import _docs
from ._docs import FolderProfile, PageRecord

FAMILIES = ("span", "anchor_dropped", "keywords", "caption", "bigram")

MAX_PAGES = 600
PER_PAGE = 3
SPAN_WORDS = (8, 25)
SPAN_STRIDE = 12
ANCHORS = 2
ANCHOR_KEEP = 6
KEYWORDS = (2, 5)
BIGRAM_PAGES = (2, 15)
#: share of a probe's content words a page must hold to count as relevant
COVER = 0.8
#: probes relevant to more pages than this are dropped
MAX_RELEVANT = 15
#: keep rule depth: a relevant page must be in the full-precision top K_KEEP
K_KEEP = 10
HARD_SHARE = 1.0 / 3.0
HARD_MIN = 60
HARD_CAP = 300
#: scoring may take at most this share of the encoding time before the pool shrinks
COST_SHARE = 0.25
MIN_POOL_PAGES = 200

#: function words left out of content words (English; other languages keep theirs, which only
#: makes the 80% relevance rule stricter)
STOPWORDS = frozenset("""
a about above after again against all also am among an and any are as at be because been before
being below between both but by can could did do does doing down during each either etc few for
from further had has have having he her here hers herself him himself his how however i if in
into is it its itself let may me might more most must my myself neither no nor not of off on
once only onto or other ought our ours ourselves out over own per same shall she should so some
such than that the their theirs them themselves then there therefore these they this those
though through thus to too under until up upon us very via was we were what when where whether
which while who whom whose why will with within without would yet you your yours yourself
yourselves
""".split())  # noqa: SIM905 - a block of words is easier to read and edit here

_TOKEN = re.compile(r"[^\W_]+(?:['\u2019\-][^\W_]+)*")
_LETTER = re.compile(r"[^\W\d_]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CAPTION = re.compile(
    r"^\s*(?:fig(?:ure)?|table|tab|chart|exhibit|graph|diagram|map|plate|photo|illustration|scheme)"
    r"\.?\s*(?:[0-9]+(?:[.\-][0-9]+)*|[ivxlc]+)?[a-z]?\s*[:.\u2013\u2014-]\s*(.+)$",
    re.IGNORECASE,
)


def is_content(word: str) -> bool:
    """A lower-cased token counts as a content word: two or more characters, a letter, not a
    stop word."""
    return len(word) >= 2 and word not in STOPWORDS and _LETTER.search(word) is not None


def tokens(text: str) -> list[str]:
    """Word tokens of ``text`` (letters and digits, joined by inner apostrophes or hyphens)."""
    return _TOKEN.findall(text)


# ---------------------------------------------------------------------------- probes


@dataclass(frozen=True)
class Probe:
    """One practice question. ``id`` is ``"P00000"``... in generation order; ``family`` is one
    of :data:`FAMILIES`; ``page_id`` is the page it was made from; ``relevant`` holds every
    page id that counts as a right answer (pages with 80% of its content words, and their
    aliases)."""

    id: str
    text: str
    family: str
    page_id: str
    relevant: frozenset[str]

    @property
    def doc_id(self) -> str:
        return self.page_id.rsplit("::p", 1)[0]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "family": self.family, "page_id": self.page_id,
                "relevant": sorted(self.relevant)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Probe:
        return cls(str(d["id"]), str(d["text"]), str(d["family"]), str(d["page_id"]),
                   frozenset(str(r) for r in d["relevant"]))


class _Corpus:
    """Word statistics of the unique text pages: vocabulary, per-page content-word sets
    (sorted int32 ids), token sequences, document frequency and postings."""

    def __init__(self, pages: Sequence[PageRecord], boilerplate: frozenset[str]) -> None:
        self.pages = list(pages)
        self.boilerplate = boilerplate
        self.vocab: dict[str, int] = {}
        content_flags: list[bool] = []
        self.seq: list[np.ndarray] = []
        self.words: list[np.ndarray] = []
        for page in self.pages:
            ids = []
            for tok in tokens(self.content_text(page)):
                low = tok.casefold()
                wid = self.vocab.get(low)
                if wid is None:
                    wid = self.vocab[low] = len(self.vocab)
                    content_flags.append(is_content(low))
                ids.append(wid)
            seq = np.asarray(ids, dtype=np.int32)
            self.seq.append(seq)
            uniq = np.unique(seq).tolist()
            self.words.append(np.asarray([w for w in uniq if content_flags[w]], dtype=np.int32))
        self.content = np.asarray(content_flags, dtype=bool)
        n_vocab = len(self.vocab)
        flat = np.concatenate(self.words) if self.words else np.zeros(0, dtype=np.int32)
        owners = np.repeat(np.arange(len(self.pages), dtype=np.int32), [w.size for w in self.words])
        order = np.argsort(flat, kind="stable")
        self.post_pages = owners[order]
        self.df = np.bincount(flat, minlength=n_vocab).astype(np.int64)
        self.post_ptr = np.concatenate([[0], np.cumsum(self.df)])
        n = len(self.pages)
        self.idf = np.log((n + 1.0) / (self.df + 1.0)) + 1.0

    def content_text(self, page: PageRecord) -> str:
        if not self.boilerplate:
            return page.text
        return "\n".join(line for line in page.text.split("\n")
                         if _docs.line_key(line) not in self.boilerplate)

    def posting(self, wid: int) -> np.ndarray:
        return self.post_pages[self.post_ptr[wid]:self.post_ptr[wid + 1]]

    def ids_of(self, words: Iterable[str]) -> np.ndarray:
        """Distinct content-word ids of lower-cased ``words`` (unknown words are left out)."""
        out = {self.vocab[w] for w in words if w in self.vocab and self.content[self.vocab[w]]}
        return np.asarray(sorted(out), dtype=np.int32)

    def relevant(self, wids: np.ndarray, limit: int = MAX_RELEVANT) -> list[int] | None:
        """Indices of the pages holding at least ``COVER`` of ``wids``; None when more than
        ``limit`` pages qualify. A qualifying page misses at most ``m - need`` words, so it holds
        one of the ``m - need + 1`` rarest: only their postings are searched."""
        m = int(wids.size)
        if m == 0:
            return []
        need = math.ceil(COVER * m - 1e-9)
        rarest = wids[np.argsort(self.df[wids], kind="stable")][: m - need + 1]
        cand = np.unique(np.concatenate([self.posting(int(w)) for w in rarest]))
        hits: list[int] = []
        for j in cand.tolist():
            have = self.words[j]
            pos = np.searchsorted(have, wids)
            pos[pos >= have.size] = 0
            if have.size and int(np.count_nonzero(have[pos] == wids)) >= need:
                hits.append(j)
                if len(hits) > limit:
                    return None
        return hits

    def adjacent_count(self, pages: Iterable[int], w1: int, w2: int) -> int:
        """How many of ``pages`` hold ``w1`` directly followed by ``w2``."""
        n = 0
        for j in pages:
            s = self.seq[j]
            if s.size > 1 and bool(np.any((s[:-1] == w1) & (s[1:] == w2))):
                n += 1
        return n


def _windows(words: list[str]) -> list[list[str]]:
    hi = SPAN_WORDS[1]
    if len(words) <= hi:
        return [words]
    starts = list(range(0, len(words) - hi + 1, SPAN_STRIDE))
    if starts[-1] != len(words) - hi:
        starts.append(len(words) - hi)
    return [words[s:s + hi] for s in starts]


def _spans(text: str) -> list[list[str]]:
    """8-25-word spans: sentences, short ones joined to the next, long ones cut into windows."""
    lo = SPAN_WORDS[0]
    out: list[list[str]] = []
    buffer: list[str] = []
    for sentence in _SENTENCE.split(text.replace("\n", " ")):
        words = tokens(sentence)
        if not words:
            continue
        buffer = buffer + words
        if len(buffer) >= lo:
            out.extend(_windows(buffer))
            buffer = []
    if len(buffer) >= lo:
        out.extend(_windows(buffer))
    return out


class _Maker:
    """Builds the candidates of one page, family by family."""

    def __init__(self, corpus: _Corpus) -> None:
        self.c = corpus

    def _score(self, words: Sequence[str]) -> tuple[float, int]:
        content = {w.casefold() for w in words if is_content(w.casefold())}
        ids = self.c.ids_of(content)
        return (float(self.c.idf[ids].sum()) / max(len(words), 1), int(ids.size))

    def ranked_spans(self, text: str) -> list[list[str]]:
        scored = []
        for i, span in enumerate(_spans(text)):
            score, n_content = self._score(span)
            if n_content >= 3:
                scored.append((-score, i, span))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [s for _, _, s in scored]

    def span(self, ctx: dict[str, Any]) -> list[str] | None:
        spans = ctx["spans"]
        return spans[0] if spans else None

    def anchor_dropped(self, ctx: dict[str, Any]) -> list[str] | None:
        spans = ctx["spans"]
        if not spans:
            return None
        span = spans[1] if len(spans) > 1 else spans[0]
        content = sorted({w.casefold() for w in span if is_content(w.casefold())},
                         key=lambda w: (-float(self.c.idf[self.c.vocab[w]]), w))
        if len(content) < ANCHORS + 2:
            return None
        anchors = set(content[:ANCHORS])
        rest = [w for w in span if w.casefold() not in anchors]
        return rest if len(rest) >= ANCHOR_KEEP else None

    def keywords(self, ctx: dict[str, Any]) -> list[str] | None:
        tf = Counter(w.casefold() for w in ctx["words"] if is_content(w.casefold()))
        ranked = sorted(tf, key=lambda w: (-tf[w] * float(self.c.idf[self.c.vocab[w]]), w))
        k = ctx["n_keywords"]
        return ranked[:k] if len(ranked) >= k else None

    def caption(self, ctx: dict[str, Any]) -> list[str] | None:
        best = None
        for line in ctx["text"].split("\n"):
            m = _CAPTION.match(line)
            if not m:
                continue
            words = tokens(m.group(1))[: SPAN_WORDS[1]]
            score, n_content = self._score(words)
            if len(words) >= 3 and n_content >= 2 and (best is None or score > best[0]):
                best = (score, words)
        return best[1] if best else None

    def bigram(self, ctx: dict[str, Any]) -> tuple[list[str], list[int]] | None:
        words = ctx["words"]
        lows = [w.casefold() for w in words]
        seen = set()
        cands = []
        for i in range(len(lows) - 1):
            a, b = lows[i], lows[i + 1]
            if a == b or (a, b) in seen or not (is_content(a) and is_content(b)):
                continue
            seen.add((a, b))
            wa, wb = self.c.vocab[a], self.c.vocab[b]
            cands.append((-(float(self.c.idf[wa]) + float(self.c.idf[wb])), i, wa, wb))
        cands.sort()
        lo, hi = BIGRAM_PAGES
        for _, i, wa, wb in cands[:25]:
            pages = self.c.relevant(np.asarray(sorted({wa, wb}), dtype=np.int32), limit=hi)
            if pages is None or len(pages) < lo:
                continue
            if lo <= self.c.adjacent_count(pages, wa, wb) <= hi:
                return [words[i], words[i + 1]], pages
        return None


def _dedupe_key(words: Sequence[str]) -> str:
    return " ".join(w.casefold() for w in words)


def make_probes(pages: Sequence[PageRecord], *, seed: int = 0, max_pages: int = MAX_PAGES,
                per_page: int = PER_PAGE, profile: FolderProfile | None = None) -> list[Probe]:
    """Tier A practice questions from the text layer of ``pages`` (FINAL_DESIGN 5.3.1).

    Up to ``max_pages`` unique, non-blank text pages are sampled with ``seed``; each gives at
    most ``per_page`` probes, its families tried in an order drawn with the same generator.
    Word statistics (IDF, relevance) use every unique text page. ``profile`` (from
    ``_docs.profile(pages)``) supplies the aliases and header/footer lines; it is computed when
    not given. The same pages and seed give the same probes, ids included.
    """
    if per_page < 1 or max_pages < 1:
        raise ValueError("per_page and max_pages must be at least 1")
    pages = list(pages)
    prof = profile if profile is not None else _docs.profile(pages)
    aliases = dict(prof.aliases)
    text_pages = [p for p in pages if p.id not in aliases and not p.blank and p.has_text]
    if not text_pages:
        return []
    copies: dict[str, list[str]] = {}
    for alias, canonical in aliases.items():
        copies.setdefault(canonical, []).append(alias)
    corpus = _Corpus(text_pages, frozenset(prof.boilerplate))
    maker = _Maker(corpus)

    rng = np.random.default_rng(seed)
    n_pick = min(int(max_pages), len(text_pages))
    picked = np.sort(rng.choice(len(text_pages), size=n_pick, replace=False))
    orders = [rng.permutation(len(FAMILIES)) for _ in range(n_pick)]
    n_keywords = rng.integers(KEYWORDS[0], KEYWORDS[1] + 1, size=n_pick)

    def relevant_ids(js: Iterable[int]) -> frozenset[str]:
        out = set()
        for j in js:
            pid = text_pages[j].id
            out.add(pid)
            out.update(copies.get(pid, ()))
        return frozenset(out)

    seen: set[str] = set()
    made: list[tuple[str, str, str, frozenset[str]]] = []
    for slot, j in enumerate(picked.tolist()):
        page = text_pages[j]
        text = corpus.content_text(page)
        ctx = {"text": text, "words": tokens(text), "n_keywords": int(n_keywords[slot])}
        ctx["spans"] = maker.ranked_spans(text)
        n_made = 0
        for f in orders[slot].tolist():
            if n_made >= per_page:
                break
            family = FAMILIES[f]
            result = getattr(maker, family)(ctx)
            if result is None:
                continue
            if family == "bigram":
                words, rel_pages = result
            else:
                words = result
                wids = corpus.ids_of(w.casefold() for w in words)
                if wids.size == 0:  # number-only or stop words only
                    continue
                rel_pages = corpus.relevant(wids)
                if rel_pages is None or j not in rel_pages:
                    continue
            key = _dedupe_key(words)
            if key in seen:
                continue
            seen.add(key)
            made.append((" ".join(words), family, page.id, relevant_ids(rel_pages)))
            n_made += 1
    return [Probe(f"P{i:05d}", text, family, page_id, rel)
            for i, (text, family, page_id, rel) in enumerate(made)]


# ---------------------------------------------------------------------------- encoding


def encode_probes(encoder: Any, probes: Sequence[Probe]) -> Any:
    """The probes as query vectors: ``encoder.encode_queries`` on their texts (an
    ``EncoderAdapter``, or the easy-mode ``Encoder``), never ``encode_documents``. The result
    is a ``MultiVectorCorpus`` whose ids are the probe ids."""
    from ..representation import MultiVectorCorpus

    texts = [p.text for p in probes]
    ids = [p.id for p in probes]
    fn = encoder.encode_queries
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    takes_ids = "ids" in params or len([p for p in params.values()
                                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 2
    out = fn(texts, ids) if takes_ids else fn(texts)
    if len(out) != len(ids):
        raise ValueError(f"encode_queries returned {len(out)} queries for {len(ids)} probes")
    if list(out.ids) != ids:
        out = MultiVectorCorpus(out.vectors, out.offsets, ids=ids, attrs=dict(out.attrs))
    return out


# ---------------------------------------------------------------------------- hard set

Ranking = Sequence[tuple[str, float]]


def _digest(obj: Any) -> str:
    from ..core.identity import digest

    return digest(obj)


@dataclass(frozen=True)
class HardSet:
    """The frozen hard practice set (FINAL_DESIGN 5.3.1).

    ``probes`` are the hard probes, hardest first; ``hardness`` their ``g``; ``ref`` the ids of
    each one's full-precision ranking, as given to :func:`hard_set`. ``examples`` are probes
    that passed the keep rule but were not used (``index.examples``). ``n_candidates`` probes
    were offered and ``n_kept`` passed the keep rule. ``sha256`` covers the rule, the probes,
    their relevant sets and their reference rankings; :meth:`verify` recomputes it, and
    :meth:`sequence` refuses a set that changed after it was frozen.

    ``checked`` holds when there are at least 60 hard probes; otherwise the folder counts as
    unchecked (no 1-bit layouts) and the practice check does not run.
    """

    probes: tuple[Probe, ...]
    hardness: Mapping[str, float]
    ref: Mapping[str, tuple[str, ...]]
    examples: tuple[Probe, ...]
    n_candidates: int
    n_kept: int
    sha256: str
    rule: Mapping[str, Any]
    aliases: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def n(self) -> int:
        return len(self.probes)

    @property
    def checked(self) -> bool:
        return self.n >= HARD_MIN

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.probes]

    @property
    def relevant(self) -> dict[str, frozenset[str]]:
        return {p.id: p.relevant for p in self.probes}

    @property
    def clusters(self) -> dict[str, str]:
        """Probe -> its source page: the unit the page-cluster bootstrap resamples."""
        return {p.id: p.page_id for p in self.probes}

    def ref_ids(self) -> dict[str, list[str]]:
        return {pid: list(r) for pid, r in self.ref.items()}

    def strata(self) -> dict[str, dict[str, int]]:
        """Where the hard probes came from: counts per family and per document (warnings only)."""
        return {"family": dict(Counter(p.family for p in self.probes)),
                "document": dict(Counter(p.doc_id for p in self.probes))}

    def _content(self) -> dict[str, Any]:
        return {"rule": dict(self.rule), "probes": [p.as_dict() for p in self.probes],
                "ref": {pid: list(self.ref[pid]) for pid in self.ids}}

    def verify(self) -> None:
        """Raise ``ValueError`` if the set no longer matches its frozen digest."""
        if _digest(self._content()) != self.sha256:
            raise ValueError("the hard practice set changed after it was frozen; rebuild it with hard_set()")

    def check(self, cand: Mapping[str, Sequence[str]], target: float, **kw: Any) -> Any:
        """``_veto.practice_check`` of candidate rankings on this set."""
        from . import _veto

        return _veto.practice_check(cand, self.ref_ids(), self.relevant, self.clusters, target,
                                    aliases=self.aliases or None, **kw)

    def sequence(self, sequence: Sequence[Any], rank_fn: Callable[[Any], Mapping[str, Sequence[str]]],
                 target: float, **kw: Any) -> Any:
        """``_veto.fixed_sequence`` on this set, after checking the frozen digest."""
        from . import _veto

        self.verify()
        return _veto.fixed_sequence(sequence, rank_fn, self.ref_ids(), self.relevant, self.clusters,
                                    target, aliases=self.aliases or None, **kw)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form for the index's ``reference/`` folder (probe texts and relevant sets;
        query vectors and reference scores are stored beside it by the storage module)."""
        return {**self._content(), "hardness": {pid: self.hardness[pid] for pid in self.ids},
                "examples": [p.as_dict() for p in self.examples], "n_candidates": self.n_candidates,
                "n_kept": self.n_kept, "sha256": self.sha256, "aliases": dict(self.aliases)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> HardSet:
        probes = tuple(Probe.from_dict(p) for p in d["probes"])
        out = cls(probes=probes,
                  hardness=MappingProxyType({k: float(v) for k, v in d["hardness"].items()}),
                  ref=MappingProxyType({k: tuple(v) for k, v in d["ref"].items()}),
                  examples=tuple(Probe.from_dict(p) for p in d.get("examples", ())),
                  n_candidates=int(d["n_candidates"]), n_kept=int(d["n_kept"]), sha256=str(d["sha256"]),
                  rule=MappingProxyType(dict(d["rule"])),
                  aliases=MappingProxyType(dict(d.get("aliases", {}))))
        out.verify()
        return out


def _ids_scores(ranking: Iterable[Any]) -> tuple[list[str], list[float]]:
    ids, scores = [], []
    for item in ranking:
        doc, score = item[0], item[1]
        ids.append(str(doc))
        scores.append(float(score))
    return ids, scores


def hardness(ids: Sequence[str], scores: Sequence[float], is_relevant: Callable[[str], bool]) -> float:
    """``g = (s_rel - s_out) / |s_rel|`` over one ranking: the best relevant score against the best
    score of any other page. Lower is harder; negative means another page outranks every
    relevant one. ``inf`` (easiest) when the ranking lists no other page or no relevant one,
    so pass rankings deeper than the largest relevant set (20 is enough: at most 15 pages)."""
    rel = [s for d, s in zip(ids, scores) if is_relevant(d)]
    out = [s for d, s in zip(ids, scores) if not is_relevant(d)]
    if not rel or not out:
        return math.inf
    s_rel, s_out = max(rel), max(out)
    if s_rel == 0:
        return math.inf
    return (s_rel - s_out) / abs(s_rel)


def hard_set(probes: Sequence[Probe], ref_rankings: Mapping[str, Ranking], *, k_keep: int = K_KEEP,
             aliases: Mapping[str, str] | None = None) -> HardSet:
    """Freeze the hard practice set from full-precision (float16) rankings.

    ``ref_rankings`` maps a probe id to its ranking, best first, as ``(page id, score)`` pairs;
    probes without one are not kept. ``aliases`` (``FolderProfile.aliases``) maps repeated page
    ids to their first copy, for rankings that list either. Rule (FINAL_DESIGN 5.3.1): keep a
    probe when a relevant page is in the top ``k_keep``; keep the hardest probe of each source
    page; take the hardest third, or 60 if more, capped at 300; ties by probe id.
    """
    alias = dict(aliases or {})
    kept: list[tuple[float, str, Probe]] = []
    for p in probes:
        ranking = ref_rankings.get(p.id)
        if not ranking:
            continue
        ids, scores = _ids_scores(ranking)

        def is_rel(d: str, rel: frozenset[str] = p.relevant) -> bool:
            return d in rel or alias.get(d, d) in rel

        if not any(is_rel(d) for d in ids[:k_keep]):
            continue
        kept.append((hardness(ids, scores, is_rel), p.id, p))
    best: dict[str, tuple[float, str, Probe]] = {}
    for item in kept:
        page = item[2].page_id
        if page not in best or item[:2] < best[page][:2]:
            best[page] = item
    ranked = sorted(best.values(), key=lambda t: (t[0], t[1]))
    m = len(ranked)
    size = min(m, HARD_CAP, max(math.ceil(m * HARD_SHARE), HARD_MIN))
    chosen = ranked[:size]
    hard_ids = {pid for _, pid, _ in chosen}
    hard = tuple(p for _, _, p in chosen)
    rule = {"k_keep": int(k_keep), "share": "1/3", "min": HARD_MIN, "cap": HARD_CAP,
            "one_per_page": True, "hardness": "(s_rel - s_out) / |s_rel|", "cover": COVER,
            "max_relevant": MAX_RELEVANT}
    ref = {p.id: tuple(_ids_scores(ref_rankings[p.id])[0]) for p in hard}
    content = {"rule": rule, "probes": [p.as_dict() for p in hard], "ref": {pid: list(ref[pid]) for pid in ref}}
    return HardSet(
        probes=hard,
        hardness=MappingProxyType({pid: g for g, pid, _ in chosen}),
        ref=MappingProxyType(ref),
        examples=tuple(sorted((p for _, _, p in kept if p.id not in hard_ids), key=lambda p: p.id)),
        n_candidates=len(probes),
        n_kept=len(kept),
        sha256=_digest(content),
        rule=MappingProxyType(rule),
        aliases=MappingProxyType(alias),
    )


# ---------------------------------------------------------------------------- cost control


@dataclass(frozen=True)
class PoolPlan:
    """How many pages the practice-question pool samples. ``projected_s`` is the scoring time
    projected for that pool, ``budget_s`` 25% of the encoding time; ``shrunk`` says the pool
    was cut below what the folder offers."""

    pages: int
    shrunk: bool
    projected_s: float
    budget_s: float

    def line(self) -> Any:
        """The summary line when the pool was cut, else None."""
        if not self.shrunk:
            return None
        from ._claim import Line, claim

        return Line("Practice", "practice questions came from {0} pages, so that scoring stays under a quarter "
                    "of the encoding time", claim(self.pages, "counted"))


def pool_size(seconds_per_probe: float, encode_seconds: float, *, n_rankings: int = 2,
              n_text_pages: int | None = None, max_pages: int = MAX_PAGES, per_page: int = PER_PAGE,
              min_pages: int = MIN_POOL_PAGES, share: float = COST_SHARE) -> PoolPlan:
    """The pool size under the cost rule (FINAL_DESIGN 5.3.1, Cost).

    ``seconds_per_probe`` is one probe's scoring time against the whole collection, timed on
    20 probes; ``n_rankings`` how many full rankings each probe needs (the reference plus the
    layouts the fixed sequence may test). If ``seconds_per_probe * pages * per_page *
    n_rankings`` exceeds ``share * encode_seconds``, the pool shrinks to fit, never below
    ``min_pages``.
    """
    offered = int(max_pages if n_text_pages is None else min(max_pages, n_text_pages))
    per_page_cost = max(float(seconds_per_probe), 0.0) * per_page * max(int(n_rankings), 1)
    budget = share * max(float(encode_seconds), 0.0)
    projected = per_page_cost * offered
    if projected <= budget or offered <= min_pages:
        return PoolPlan(offered, False, projected, budget)
    fit = int(budget // per_page_cost) if per_page_cost > 0 else offered
    pages = max(min(fit, offered), min(min_pages, offered))
    return PoolPlan(pages, pages < offered, per_page_cost * pages, budget)


__all__ = [
    "ANCHORS",
    "ANCHOR_KEEP",
    "BIGRAM_PAGES",
    "COST_SHARE",
    "COVER",
    "FAMILIES",
    "HARD_CAP",
    "HARD_MIN",
    "HARD_SHARE",
    "KEYWORDS",
    "K_KEEP",
    "MAX_PAGES",
    "MAX_RELEVANT",
    "MIN_POOL_PAGES",
    "PER_PAGE",
    "SPAN_WORDS",
    "STOPWORDS",
    "HardSet",
    "PoolPlan",
    "Probe",
    "encode_probes",
    "hard_set",
    "hardness",
    "is_content",
    "make_probes",
    "pool_size",
    "tokens",
]
