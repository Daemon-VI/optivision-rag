"""The opt-in local query log of an index (FINAL_DESIGN 5.7 step 5, 14; CONTRACT `_querylog`).

Off unless ``load(path, log_queries=True)`` turns it on; nothing in easy mode switches it on by
itself. When on, the text of each string search is appended to ``<index>.ovx/log/queries.jsonl``
on this machine. Nothing is sent anywhere: this module opens no socket and imports no network
library. A privacy notice is printed the first time the log is opened for an index in a
process. ``QueryLog.delete()`` (or deleting the ``log/`` folder) removes everything.

What the log is for: ``Index.upgrade()`` needs real questions it has never used, and a log of
real searches is the plainest source of them. Only the text is stored, with its hash; no time,
user or result is recorded.

The query hash (:func:`query_hash`) is shared with ``_upgrade``: it is what "never used" and
"de-duplicated" compare. Two texts are the same question when they are equal after collapsing
runs of white space and ignoring case.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

#: folder of the log inside the index folder, and the file of entries
LOG_DIR = "log"
LOG_FILE = "queries.jsonl"

#: index paths whose notice was printed in this process
_NOTICE_SHOWN: set[str] = set()


def normalize_query(text: str) -> str:
    """The form two questions are compared in: white space collapsed, case folded."""
    if not isinstance(text, str):
        raise TypeError(f"a question must be a string, got {type(text).__name__}")
    return " ".join(text.split()).casefold()


def query_hash(text: str) -> str:
    """SHA-256 (hex) of :func:`normalize_query` of ``text``, UTF-8."""
    return hashlib.sha256(normalize_query(text).encode("utf-8")).hexdigest()


def vector_hash(vectors: Any) -> str:
    """SHA-256 (hex) of one query's vectors as contiguous float32 bytes plus their shape.

    Used for queries given as vectors, where there is no text to compare."""
    import numpy as np

    a = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    h = hashlib.sha256(repr(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def notice(path: str | Path) -> str:
    """The privacy notice, as an easy-mode block (ASCII, no numbers, ends with ``Next:``)."""
    from ._claim import Block, lit

    folder = Path(path) / LOG_DIR
    block = Block([], next="index.search(...) as usual; later, index.upgrade() with these questions")
    block.add("Query log", "on: the text of each search on this index is saved in {0}", lit(folder))
    block.add("", "on this machine only. It is never sent anywhere.")
    block.add("", "It is off unless load(..., log_queries=True) is given.")
    block.add("Delete", "delete that folder, or call index.query_log.delete()")
    return str(block)


class QueryLog:
    """The query log of one index folder. Create it with :func:`open_log`."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.folder = self.index_path / LOG_DIR
        self.file = self.folder / LOG_FILE

    # ------------------------------------------------------------ writing

    def record(self, texts: str | Iterable[str]) -> int:
        """Append string queries; returns how many were written. Non-strings are skipped
        (a search by vectors has no text to keep)."""
        if isinstance(texts, str):
            texts = [texts]
        rows = [t for t in texts if isinstance(t, str) and t.strip()]
        if not rows:
            return 0
        self.folder.mkdir(parents=True, exist_ok=True)
        with open(self.file, "a", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps({"q": t, "sha256": query_hash(t)}, ensure_ascii=True) + "\n" for t in rows)
            fh.flush()
        return len(rows)

    # ------------------------------------------------------------ reading

    def entries(self) -> list[dict[str, str]]:
        """Every entry in the order written; a damaged line (a crash mid-write) is skipped."""
        if not self.file.exists():
            return []
        out = []
        with open(self.file, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("q"), str):
                    out.append({"q": row["q"], "sha256": query_hash(row["q"])})
        return out

    def queries(self, exclude: Iterable[str] = ()) -> list[str]:
        """The distinct logged questions in first-seen order, without those whose hash is in
        ``exclude`` (for example the hashes an earlier ``upgrade()`` used)."""
        seen = set(exclude)
        out = []
        for row in self.entries():
            if row["sha256"] in seen:
                continue
            seen.add(row["sha256"])
            out.append(row["q"])
        return out

    def __len__(self) -> int:
        return len(self.entries())

    # ------------------------------------------------------------ deleting

    def delete(self) -> bool:
        """Remove the log folder and everything in it; True if there was something to remove."""
        if not self.folder.exists():
            return False
        shutil.rmtree(self.folder)
        return True

    def __repr__(self) -> str:
        return f"QueryLog({str(self.folder)!r})"


def open_log(index_path: str | Path, enabled: bool = False, *,
             out: Callable[[str], Any] | None = print) -> QueryLog | None:
    """The log of an index when ``enabled`` (what ``load(..., log_queries=True)`` passes), else
    None. Nothing is created until the first question is recorded. The privacy notice goes to
    ``out`` the first time a log is opened for this index in this process."""
    if enabled is not True:
        if enabled not in (False, None):
            raise TypeError(f"log_queries must be True or False, got {enabled!r}")
        return None
    log = QueryLog(index_path)
    key = str(Path(index_path).resolve())
    if key not in _NOTICE_SHOWN:
        _NOTICE_SHOWN.add(key)
        if out is not None:
            out(notice(index_path))
    return log


def existing_log(index_path: str | Path) -> QueryLog | None:
    """The log of an index if one was written earlier (logging need not be on now), else None.
    Reading a log never turns logging on."""
    log = QueryLog(index_path)
    return log if log.file.exists() else None


def record(log: QueryLog | None, queries: str | Sequence[Any]) -> int:
    """What ``Index.search`` calls: a no-op when the log is off (``log is None``)."""
    if log is None:
        return 0
    return log.record(queries)
