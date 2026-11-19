"""What the integrations share: install lines, evidence metadata, page ids, searches without the log.

Metadata values are JSON types only (str, int, float, bool, None, lists of str), because frameworks copy
metadata into vector stores and prompts. The evidence an index carries (``Index.evidence``: a
``_claim.Block``, a list of claims or lines, a dict or text) is rendered once per search into
``optivision_evidence`` (whole lines, each with its tag) and ``optivision_tags`` (the distinct tags).
"""

from __future__ import annotations

import contextlib
import inspect
import threading
from collections.abc import Iterator
from typing import Any

from .. import _ledger
from .._types import MissingExtra

#: metadata keys that describe the index rather than the page (kept out of LLM and embedding text)
INDEX_KEYS = ("optivision_layout", "optivision_layout_plain", "optivision_target", "optivision_evidence",
              "optivision_tags", "optivision_status", "optivision_score_kind")

#: attributes an Index may use for its opt-in query log; paused while the demo searches
_LOG_ATTRIBUTES = ("_querylog", "_query_log", "_log")
_LOG_LOCK = threading.Lock()


def missing_extra(what: str, package: str, extra: str) -> MissingExtra:
    """``MissingExtra`` naming the package and the pip line of the extra that brings it."""
    return MissingExtra(f'{what} needs {package}. Install it with: pip install "optivision-rag[{extra}]"')


# ---------------------------------------------------------------------------- evidence


def _render(item: Any) -> str:
    if hasattr(item, "render") and callable(item.render):
        return str(item.render()).strip()
    tag, text = getattr(item, "tag", None), getattr(item, "text", None)
    if isinstance(tag, str) and text is not None:
        text = text() if callable(text) else text
        return f"{text}  {tag}".strip() if tag else str(text)
    return str(item).strip()


def evidence_lines(evidence: Any) -> list[str]:
    """The index's evidence as rendered lines, each carrying its own tag."""
    if evidence is None:
        return []
    if isinstance(evidence, str):
        return [line for line in evidence.splitlines() if line.strip()]
    lines = getattr(evidence, "lines", None)
    if isinstance(lines, list):  # a _claim.Block
        out = [_render(line) for line in lines]
        nxt = getattr(evidence, "next", None)
        return out + ([f"Next: {nxt}"] if nxt else [])
    if isinstance(evidence, dict):
        return [f"{key}: {_render(value)}" for key, value in evidence.items()]
    if isinstance(evidence, (list, tuple)):
        return [_render(item) for item in evidence]
    return [_render(evidence)]


def _tags_of(item: Any, out: list[str]) -> None:
    tag = getattr(item, "tag", None)
    if isinstance(tag, str) and tag:
        out.append(tag)
    for claim in getattr(item, "claims", None) or []:  # a Line's claims, if it has no tag of its own
        if not tag and isinstance(getattr(claim, "tag", None), str):
            out.append(claim.tag)


def evidence_tags(evidence: Any) -> list[str]:
    """The distinct evidence tags (``[our benchmarks: ...]``, ``[counted]``, ...), in the order they appear."""
    found: list[str] = []
    if evidence is None or isinstance(evidence, str):
        pass
    elif isinstance(getattr(evidence, "lines", None), list):
        for line in evidence.lines:
            _tags_of(line, found)
    elif isinstance(evidence, dict):
        for value in evidence.values():
            _tags_of(value, found)
    elif isinstance(evidence, (list, tuple)):
        for item in evidence:
            _tags_of(item, found)
    else:
        _tags_of(evidence, found)
    return list(dict.fromkeys(found))


def index_metadata(index: Any) -> dict[str, Any]:
    """What every document from this index carries about the index itself."""
    layout = getattr(index, "layout", None)
    target = getattr(index, "target", None)
    evidence = getattr(index, "evidence", None)
    return {
        "optivision_layout": None if layout is None else str(getattr(layout, "name", layout)),
        "optivision_layout_plain": None if layout is None else str(getattr(layout, "plain", "") or ""),
        "optivision_target": None if target is None else float(target),
        "optivision_evidence": evidence_lines(evidence),
        "optivision_tags": evidence_tags(evidence),
        "optivision_status": _ledger.status(),
        "optivision_score_kind": "MaxSim inner product over the stored layout; compare scores within one index",
    }


# ---------------------------------------------------------------------------- pages


def split_page_id(page_id: str) -> tuple[str, int | None]:
    """``("my_pdfs/a.pdf", 3)`` from ``"my_pdfs/a.pdf::p3"``; ``(id, None)`` for an id without a page."""
    doc, sep, page = str(page_id).rpartition("::p")
    if sep and page.isdigit():
        return doc, int(page)
    return str(page_id), None


def hit_metadata(hit: Any, rank: int) -> dict[str, Any]:
    file = getattr(hit, "file", None)
    page = getattr(hit, "page", None)
    return {
        "id": str(hit.id),
        "page_id": str(hit.id),
        "file": None if file is None else str(file),
        "page": None if page is None else int(page),
        "score": float(hit.score),
        "rank": int(rank),
        "source": str(file) if file is not None else str(hit.id),
    }


def documents(index: Any, query: str, k: int) -> list[dict[str, Any]]:
    """``index.search(query, k)`` as framework-neutral documents.

    Each is ``{"id", "page_content", "score", "metadata"}``: ``page_content`` is the page's text layer
    (``""`` when the index kept none or the page has none; the page image is ``Hit.image()``).
    """
    if not isinstance(query, str):
        raise TypeError(f"a retriever query is one string, got {type(query).__name__}")
    hits = index.search(query, k=int(k))
    shared = index_metadata(index)
    out = []
    for rank, hit in enumerate(hits, start=1):
        meta = {**hit_metadata(hit, rank), **shared}
        out.append({"id": str(hit.id), "page_content": str(getattr(hit, "text", "") or ""),
                    "score": float(hit.score), "metadata": meta})
    return out


# ---------------------------------------------------------------------------- no query log


@contextlib.contextmanager
def _log_paused(index: Any) -> Iterator[None]:
    saved = {}
    with _LOG_LOCK:
        for name in _LOG_ATTRIBUTES:
            if getattr(index, name, None) is not None:
                saved[name] = getattr(index, name)
                setattr(index, name, None)
        try:
            yield
        finally:
            for name, value in saved.items():
                setattr(index, name, value)


def search_unlogged(index: Any, query: str, k: int) -> list[Any]:
    """``index.search`` that never reaches the opt-in query log.

    Uses ``index.search(..., log=False)`` when the Index accepts it; otherwise the log attributes an
    Index may hold are set to ``None`` for the call and restored afterwards.
    """
    try:
        accepts_log = "log" in inspect.signature(index.search).parameters
    except (TypeError, ValueError):
        accepts_log = False
    if accepts_log:
        return list(index.search(query, k=k, log=False))
    with _log_paused(index):
        return list(index.search(query, k=k))
