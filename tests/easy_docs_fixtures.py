"""Synthetic documents and a lexical stand-in encoder for the easy-mode documents tests
(test_easy_docs, test_easy_probes, test_easy_veto).

Everything is seeded. The text is made of generated words, so its statistics (which words
are rare, which pages share a phrase) are known by construction; the encoder maps each word
to a fixed random vector, so MaxSim scores count shared words. Results on this data exercise
code paths and say nothing about any real encoder.
"""

from __future__ import annotations

import hashlib
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from optivision.easy._docs import PageRecord
from optivision.easy._probes import tokens

HEADER = "Northwind Traders quarterly review"
SYLLABLES = ("ka", "lo", "mi", "ra", "ten", "vo", "shi", "del", "pra", "nu", "zor", "qui", "bel", "mon",
             "tas", "fer", "gol", "hin", "jur", "wex", "pol", "sam", "dri", "yel")
FUNCTION_WORDS = ("the", "of", "and", "to", "in", "for", "with", "on", "by", "is", "was", "that", "from", "as")


def _words(rng: np.random.Generator, n: int, taken: set[str]) -> list[str]:
    out = []
    while len(out) < n:
        k = int(rng.integers(2, 4))
        w = "".join(SYLLABLES[int(i)] for i in rng.integers(0, len(SYLLABLES), size=k))
        if w not in taken:
            taken.add(w)
            out.append(w)
    return out


def synthetic_texts(n_pages: int, *, seed: int = 0, group: int = 4, captions: bool = True,
                    numbers_page: bool = True) -> list[str]:
    """Page texts: a header and a "Page i of n" footer on every page (boilerplate), six
    sentences mixing function words, shared words, words of the page's group (``group``
    consecutive pages share a phrase) and words of the page alone; a caption on every third
    page; one page holding a table of numbers."""
    rng = np.random.default_rng(seed)
    taken: set[str] = set()
    common = _words(rng, 300, taken)
    n_groups = (n_pages + group - 1) // group
    group_words = [_words(rng, 4, taken) for _ in range(n_groups)]
    texts = []
    for i in range(n_pages):
        own = _words(rng, 8, taken)
        shared = group_words[i // group]
        lines = [HEADER]
        for s in range(6):
            length = int(rng.integers(12, 19))
            words = []
            for _ in range(length):
                u = rng.random()
                if u < 0.30:
                    words.append(FUNCTION_WORDS[int(rng.integers(len(FUNCTION_WORDS)))])
                elif u < 0.62:
                    words.append(common[int(rng.integers(len(common)))])
                elif u < 0.72:
                    words.append(shared[int(rng.integers(len(shared)))])
                else:
                    words.append(own[int(rng.integers(len(own)))])
            if s == 2:  # the group's phrase: an adjacent pair shared by the group's pages
                words[3:3] = [shared[0], shared[1]]
            sentence = " ".join(words)
            lines.append(sentence[0].upper() + sentence[1:] + ".")
        if captions and i % 3 == 0:
            lines.append(f"Figure {i + 1}: " + " ".join([own[0], "of", own[1], common[i % 300], own[2]]))
        if numbers_page and i == 1:
            lines.append("2019 2020 2021 14.5 16.2 18.9 21.0")
        lines.append(f"Page {i + 1} of {n_pages}")
        texts.append("\n".join(lines))
    return texts


def page_records(texts: list[str], *, doc_id: str = "docs/report.pdf", duplicates: dict[int, int] | None = None,
                 blanks: tuple[int, ...] = (), images: tuple[int, ...] = ()) -> list[PageRecord]:
    """PageRecords of ``texts`` as the pages of one PDF. ``duplicates`` maps a page index to the
    index whose render it repeats (same sha256 and text); ``blanks`` are blank pages; ``images``
    are pages without a text layer that are not blank."""
    duplicates = duplicates or {}
    out = []
    for i, text in enumerate(texts):
        src = duplicates.get(i, i)
        body = texts[src]
        if i in blanks:
            body = ""
        if i in images:
            body = ""
        digest = hashlib.sha256((f"render {src} " + ("blank" if i in blanks else body)).encode()).hexdigest()
        if i in images:
            digest = hashlib.sha256(f"image {i}".encode()).hexdigest()
        n_chars = sum(1 for ch in body if not ch.isspace())
        out.append(PageRecord(id=f"{doc_id}::p{i + 1}", doc_id=doc_id, file=f"/x/{doc_id}", relpath="report.pdf",
                              page=i + 1, text=body, n_chars=n_chars, blank=i in blanks, sha256=digest,
                              landscape=False, width=1240, height=1754))
    return out


def make_pdf(path: Path, pages: list[dict[str, Any]], *, password: str | None = None) -> Path:
    """A PDF written with reportlab. Each page is a dict: ``lines`` (text lines; none and no
    ``shapes`` gives a blank page), ``landscape`` (bool), ``shapes`` (draw filled boxes, no text)."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas

    kwargs = {}
    if password is not None:
        from reportlab.lib import pdfencrypt

        kwargs["encrypt"] = pdfencrypt.StandardEncryption(password, ownerPassword=password + "-owner")
    c = canvas.Canvas(str(path), pagesize=A4, **kwargs)
    for spec in pages:
        size = landscape(A4) if spec.get("landscape") else A4
        c.setPageSize(size)
        y = size[1] - 72
        for line in spec.get("lines", ()):
            c.drawString(72, y, line)
            y -= 16
        if spec.get("shapes"):
            for k in range(6):
                c.rect(72 + 60 * k, 300 + 30 * k, 50, 120, fill=1)
        c.showPage()
    c.save()
    return path


# ---------------------------------------------------------------------------- stand-in encoder


def word_vector(word: str, dim: int = 48, shift: float = 0.0) -> np.ndarray:
    """A fixed unit vector per word (seeded by the word's CRC32), plus ``shift`` times one
    common direction before normalising, which moves all vectors closer together."""
    v = np.random.default_rng(zlib.crc32(word.encode("utf-8"))).standard_normal(dim)
    v /= np.linalg.norm(v)
    if shift:
        common = np.random.default_rng(12345).standard_normal(dim)
        v = v + shift * common / np.linalg.norm(common)
        v /= np.linalg.norm(v)
    return v


def encode_text(text: str, dim: int = 48, shift: float = 0.0) -> np.ndarray:
    words = [w.casefold() for w in tokens(text)] or ["<empty>"]
    return np.stack([word_vector(w, dim, shift) for w in words]).astype(np.float32)


def word_adapter(dim: int = 48, shift: float = 0.0, *, documents_allowed: bool = True) -> Any:
    """A ``CallableAdapter`` whose vectors are one word vector per token. With
    ``documents_allowed=False`` encoding documents raises, so a test can show probes are
    encoded as queries."""
    from optivision.adapters import CallableAdapter

    def queries(items: list[str]) -> list[np.ndarray]:
        return [encode_text(t, dim, shift) for t in items]

    def documents(items: list[str]) -> list[np.ndarray]:
        if not documents_allowed:
            raise AssertionError("practice questions must be encoded with encode_queries")
        return [encode_text(t, dim, shift) for t in items]

    return CallableAdapter(documents, queries)
