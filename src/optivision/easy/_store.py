"""The ``.ovx`` index folder: one compressed copy of a collection, written so that a crash never
breaks it (docs/easy/CONTRACT.md; FINAL_DESIGN 5.7 and 9).

Layout::

    <name>.ovx/
        manifest.json               the commit point: written last, through os.replace
        segments/NNN/codes.npy      uint8 [vectors, code bytes], memory-mapped when read
        segments/NNN/offsets.npy    int64 [pages + 1]
        segments/NNN/ids.json       page ids of segment NNN, in store order
        segments/NNN/before.npy     int64 [pages]: vectors per page before the pipeline (when known)
        state/pipeline.json         Pipeline.to_dict(), fitted-state digests and scalar state
        state/fitted.npz            the fitted arrays (the centre of centred 4-bit, ...), if any
        text/NNN.zlib.json          zlib-compressed JSON {page id: text layer} of segment NNN
        float16/NNN/vectors.npy     optional float16 copy of segment NNN, before the pipeline
        float16/NNN/offsets.npy
        aliases.json                {duplicate page id: id of the stored copy}
        build.json                  the build record
        reference/reference.npz     the practice-check reference: an opaque dict with arrays
        log/                        reserved for the opt-in query log
        write.lock                  present while a writer holds the index

Why not ``compose.save_compressed``: it refuses any codec with fitted state (compose.py:264),
and centred 4-bit has one, its centre. Here the fitted state goes through
``core.stages.state`` and is restored with its digests checked.

Crash safety. Segment files never change once written. ``append`` writes a new numbered
segment; ``remove`` and ``compact`` write replacement segments under new numbers. Only
``manifest.json`` says which segments make up the index, and ``commit()`` replaces it last,
after every file it names has been flushed and fsynced, with ``os.replace`` (atomic on one
volume). A crash at any earlier moment leaves the previous manifest, and with it the previous
index, valid. The unreferenced ("orphan") segment folders such a crash leaves behind are
ignored by readers and deleted the next time a writer opens the index. The small side files
(aliases, build record, reference) are each replaced atomically inside ``commit()``, just
before the manifest, which records their sha256.

Writers. One writer at a time: ``Store.create`` and ``Store.open(path, mode="a")`` take the
writer lock, a file created with ``os.open(O_CREAT | O_EXCL)`` that holds the pid, host and
time, and ``close()`` releases it. The lock is re-entrant within one thread, so a build can
hold ``writer_lock(path)`` across encoding and create the store inside it. Readers
(``mode="r"``) never lock and see the last committed manifest.

Windows. A memory-mapped file cannot be deleted or replaced. The store keeps every map it
opens in caches that ``close()`` drops (then collects garbage), so that after ``close()`` the
folder can be deleted, provided the caller holds no segment array of its own.

Sizes. ``sizes()`` adds up the size of every file in the folder, so the printed total is the
bytes on disk (file sizes, not allocated clusters); the categories add up to the total. The
writer lock is the one file left out, because it exists only while a writer works.
"""

from __future__ import annotations

import calendar
import copy
import gc
import hashlib
import io
import json
import os
import shutil
import socket
import sys
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np

from ..compose import CompressedCorpus, Pipeline
from ..core.identity import array_digest
from ..representation import MultiVectorCorpus
from ._types import IndexLocked

__all__ = [
    "COMPACT_ABOVE",
    "FORMAT",
    "FORMAT_VERSION",
    "LOCK_NAME",
    "SEGMENT_PAGES",
    "Store",
    "clear_stale_lock",
    "lock_holder",
    "settle",
    "settle_many",
    "split",
    "writer_lock",
]

FORMAT = "optivision-easy-index"
FORMAT_VERSION = 1
STATE_FORMAT = "optivision-easy-pipeline"
REFERENCE_FORMAT = "optivision-easy-reference"
MANIFEST = "manifest.json"
LOCK_NAME = "write.lock"
#: ``compact()`` merges segments while there are more than this many (FINAL_DESIGN 9, ``_index``)
COMPACT_ABOVE = 8
#: pages per segment when a corpus is split for settling (the build's shard size, FINAL_DESIGN 5.7)
SEGMENT_PAGES = 500
#: bytes copied per step when segments are merged, so compaction never holds a whole segment
_COPY_BYTES = 64 * 1024 * 1024
#: decompressed text segments kept in memory
_TEXT_CACHE = 4

#: manifest keys the store owns; ``commit(**updates)`` and ``info`` may not set them
RESERVED_KEYS = frozenset({
    "format", "version", "created", "updated", "generation", "dim", "pipeline", "segments",
    "next_segment", "n_pages", "n_vectors", "float16", "files",
})
#: what an index folder may contain; anything else makes create() refuse to clear the folder
_KNOWN_ENTRIES = frozenset({
    MANIFEST, "segments", "state", "text", "float16", "reference", "log", "aliases.json",
    "build.json", LOCK_NAME,
})
_SIDE_FILES = ("aliases.json", "build.json", "reference/reference.npz")
#: additive per-segment statistics (``Pipeline.encode`` stats) summed when segments merge
_ADDITIVE_STATS = ("vectors_before", "float32_bytes_before", "native_bytes_before")


# ----------------------------------------------------------------------------- files


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fsync_dir(path: Path) -> None:
    """Make a rename inside ``path`` durable (POSIX); Windows has no directory handles for it."""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_bytes(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _save_npy(path: Path, array: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.save(f, np.ascontiguousarray(array), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())


def _fsync_file(path: Path) -> None:
    with open(path, "rb+") as f:
        os.fsync(f.fileno())


def _replace(src: Path, dst: Path, tries: int = 20) -> None:
    """``os.replace`` with a few retries: on Windows a reader that has ``dst`` open for a moment
    makes the replace fail with PermissionError."""
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so that a reader sees the old file or the new one, never a mix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    _write_bytes(tmp, data)
    _replace(tmp, path)
    _fsync_dir(path.parent)


def _remove_tree(path: Path) -> None:
    """Best effort: a file another process still maps (Windows) stays, and the next writer
    that opens the index removes it."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        with suppress(OSError):
            path.unlink()


def _load_npy(path: Path, mmap: bool) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def _seg_name(number: int) -> str:
    return f"{number:03d}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any, what: str) -> Any:
    """``value`` with NumPy scalars made plain, after proving it serialises as strict JSON."""

    def plain(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, Mapping):
            return {str(k): plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [plain(v) for v in o]
        return o

    out = plain(value)
    try:
        json.dumps(out, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be JSON (strings, numbers, lists, dicts; no NaN): {exc}") from None
    return out


def _check_updates(updates: Mapping[str, Any], what: str) -> dict[str, Any]:
    bad = sorted(set(updates) & RESERVED_KEYS)
    if bad:
        raise ValueError(f"{what} may not set {bad}: the store maintains those manifest keys itself")
    return _jsonable(dict(updates), what)


# ----------------------------------------------------------------------------- writer lock

_HELD: dict[str, dict[str, Any]] = {}
_HELD_GUARD = threading.Lock()


def _key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _pid_alive(pid: int) -> bool | None:
    """Whether process ``pid`` runs on this machine; None when that cannot be told.

    ``os.kill(pid, 0)`` is not used on Windows, where it would terminate the process.
    """
    if pid == os.getpid():
        return True
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return ctypes.get_last_error() == 5  # access denied: it exists; else there is none
            try:
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                return code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except (OSError, AttributeError):  # pragma: no cover - no kernel32
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:  # pragma: no cover
        return None
    return True


def lock_holder(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """What the lock file of the index at ``path`` says (pid, host, time), or None when unlocked.

    ``{"unreadable": "<reason>"}`` when the file exists but cannot be read (a writer that
    stopped while writing it).
    """
    lock = Path(path) / LOCK_NAME
    try:
        text = lock.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        return {"unreadable": str(exc)}
    try:
        info = json.loads(text)
    except ValueError as exc:
        return {"unreadable": f"{exc}"}
    return info if isinstance(info, dict) else {"unreadable": "not a JSON object"}


def _age(stamp: str) -> str:
    try:
        then = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, OverflowError):
        return "unknown time"
    seconds = max(0, int(time.time() - then))
    if seconds < 120:
        return f"{seconds} s ago"
    if seconds < 7200:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h ago"


def _locked_message(path: Path) -> str:
    lock = path / LOCK_NAME
    info = lock_holder(path)
    if info is None:
        return f"{path} was locked a moment ago; retry"
    if "unreadable" in info:
        return (f"{path} is locked, but {lock} cannot be read ({info['unreadable']}). If no build or "
                f"add() is writing this index, delete {lock} and retry.")
    pid, host, stamp = info.get("pid"), info.get("host"), str(info.get("time", "?"))
    when = f"since {stamp}, {_age(stamp)}"
    if host == socket.gethostname() and isinstance(pid, int):
        alive = _pid_alive(pid)
        if alive is False:
            return (f"{path} has a stale writer lock: process {pid} on this machine is no longer "
                    f"running (it stopped while writing, {when}). An unfinished write is never "
                    f"published, so the last saved index is intact. Clear the lock with "
                    f"optivision.easy._store.clear_stale_lock({str(path)!r}) or delete {lock}, then retry.")
        if alive is True:
            return (f"{path} is being written by process {pid} on this machine ({when}); wait for "
                    f"that build or add() to finish, then retry.")
    return (f"{path} is locked by process {pid} on {host} ({when}). If no build or add() is "
            f"running there any more, delete {lock} and retry.")


def acquire_lock(path: str | os.PathLike[str]) -> str:
    """Take the writer lock of the index folder at ``path`` (created if missing); returns its token.

    Re-entrant within one thread. Raises ``IndexLocked`` when another process or thread holds it.
    """
    key = _key(path)
    folder = Path(path)
    with _HELD_GUARD:
        held = _HELD.get(key)
        if held is not None:
            if held["thread"] == threading.get_ident():
                held["depth"] += 1
                return str(held["token"])
            raise IndexLocked(f"{folder} is being written by another thread of this process")
        created = not folder.exists()
        folder.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        info = {"pid": os.getpid(), "host": socket.gethostname(), "time": _now(), "token": token,
                "python": sys.executable}
        try:
            fd = os.open(folder / LOCK_NAME, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise IndexLocked(_locked_message(folder)) from None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(info, f)
            f.flush()
            os.fsync(f.fileno())
        _HELD[key] = {"token": token, "thread": threading.get_ident(), "depth": 1, "created": created}
        return token


def release_lock(path: str | os.PathLike[str]) -> None:
    """Undo one ``acquire_lock`` by this thread; the outermost release deletes the lock file."""
    key = _key(path)
    with _HELD_GUARD:
        held = _HELD.get(key)
        if held is None or held["thread"] != threading.get_ident():
            return
        held["depth"] -= 1
        if held["depth"] > 0:
            return
        del _HELD[key]
        folder = Path(path)
        info = lock_holder(folder)
        if info is not None and info.get("token") == held["token"]:
            with suppress(OSError):
                (folder / LOCK_NAME).unlink()
        if held["created"]:
            with suppress(OSError):
                folder.rmdir()  # only an empty folder this lock created: a build that wrote nothing


def holds_lock(path: str | os.PathLike[str]) -> bool:
    """True when this thread holds the writer lock of ``path``."""
    with _HELD_GUARD:
        held = _HELD.get(_key(path))
        return held is not None and held["thread"] == threading.get_ident()


@contextmanager
def writer_lock(path: str | os.PathLike[str]) -> Iterator[Path]:
    """Hold the writer lock of the index folder at ``path`` for the ``with`` block.

    The build takes it before encoding, so that two builds of one folder never interleave;
    ``Store.create`` and ``Store.open(path, "a")`` inside the block re-enter it.
    """
    acquire_lock(path)
    try:
        yield Path(path)
    finally:
        release_lock(path)


def clear_stale_lock(path: str | os.PathLike[str]) -> bool:
    """Delete the lock of ``path`` if its process ran on this machine and is gone; True if deleted.

    A lock taken on another machine (a shared drive, an earlier Colab runtime) is never
    cleared here, because whether its process still runs cannot be told from this machine.
    """
    info = lock_holder(path)
    if info is None or "unreadable" in info:
        return False
    pid = info.get("pid")
    if info.get("host") != socket.gethostname() or not isinstance(pid, int) or _pid_alive(pid) is not False:
        return False
    try:
        (Path(path) / LOCK_NAME).unlink()
    except FileNotFoundError:
        return False
    return True


# ----------------------------------------------------------------------------- fitted pipeline


def _state_payload(pipeline: Pipeline) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from ..core.stages.state import pipeline_fitted_identity, pipeline_fitted_state

    states = pipeline_fitted_state(pipeline)
    stages: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for i, state in enumerate(states):
        entry: dict[str, Any] = {"arrays": {}, "scalars": {}}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                name = f"s{i}__{key}"
                arrays[name] = value
                entry["arrays"][key] = name
            else:
                entry["scalars"][key] = value
        stages.append(entry)
    doc = {
        "format": STATE_FORMAT,
        "version": 1,
        "pipeline": pipeline.to_dict(),
        "label": pipeline.label(),
        "digests": pipeline_fitted_identity(pipeline),
        "stages": stages,
    }
    return _jsonable(doc, "the pipeline state"), arrays


def _restore_pipeline(doc: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> Pipeline:
    """A fitted pipeline from its stored state; ``restore_pipeline_state`` checks every digest."""
    from ..core.stages.state import restore_pipeline_state

    pipeline = Pipeline.from_dict(doc["pipeline"])
    states = []
    for entry in doc["stages"]:
        state: dict[str, Any] = {key: arrays[name] for key, name in entry["arrays"].items()}
        state.update(entry["scalars"])
        states.append(state)
    return restore_pipeline_state(pipeline, states, expected_digests=doc["digests"])


def _has_token_stage(pipeline: Pipeline) -> bool:
    return any(getattr(s, "kind", None) == "tokens" for s in pipeline.stages)


# ----------------------------------------------------------------------------- reference packing


def _pack_reference(value: Mapping[str, Any]) -> bytes:
    """One ``.npz`` (a single file, so it is replaced atomically): a JSON header with
    placeholders, plus the arrays. Dicts, lists, tuples (read back as lists), sets (read back
    as sets), strings, numbers, None, NumPy arrays and ``MultiVectorCorpus`` values are kept."""
    if not isinstance(value, Mapping):
        raise TypeError(f"the reference must be a dict, got {type(value).__name__}")
    arrays: dict[str, np.ndarray] = {}

    def array(a: np.ndarray) -> dict[str, str]:
        if a.dtype.hasobject:
            raise TypeError("object arrays cannot be stored in the reference")
        key = f"a{len(arrays)}"
        arrays[key] = np.ascontiguousarray(a)
        return {"__array__": key}

    def enc(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return array(o)
        if isinstance(o, MultiVectorCorpus):
            return {"__corpus__": {"vectors": array(np.asarray(o.vectors)), "offsets": array(o.offsets),
                                   "ids": list(o.ids)}}
        if isinstance(o, Mapping):
            if any(not isinstance(k, str) for k in o):
                raise TypeError("reference dict keys must be strings")
            if {"__array__", "__corpus__", "__set__"} & set(o):
                raise ValueError("'__array__', '__corpus__' and '__set__' are reserved reference keys")
            return {k: enc(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [enc(v) for v in o]
        if isinstance(o, (set, frozenset)):
            return {"__set__": sorted((enc(v) for v in o), key=repr)}
        if isinstance(o, np.generic):
            return o.item()
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        raise TypeError(f"cannot store a {type(o).__name__} in the reference")

    header = {"format": REFERENCE_FORMAT, "version": 1, "value": enc(value)}
    try:
        blob = json.dumps(header, allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise ValueError(f"the reference holds a NaN or infinite scalar; put such values in arrays ({exc})") from None
    buf = io.BytesIO()
    np.savez(buf, __header__=np.frombuffer(blob, dtype=np.uint8), **arrays)
    return buf.getvalue()


def _unpack_reference(data: bytes) -> dict[str, Any]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    header = json.loads(bytes(arrays.pop("__header__")).decode("utf-8"))
    if header.get("format") != REFERENCE_FORMAT:
        raise ValueError("reference/reference.npz is not an easy-mode reference")

    def dec(o: Any) -> Any:
        if isinstance(o, dict):
            if set(o) == {"__array__"}:
                return arrays[o["__array__"]]
            if set(o) == {"__corpus__"}:
                c = o["__corpus__"]
                return MultiVectorCorpus(dec(c["vectors"]), dec(c["offsets"]), ids=c["ids"])
            if set(o) == {"__set__"}:
                return {dec(v) for v in o["__set__"]}
            return {k: dec(v) for k, v in o.items()}
        if isinstance(o, list):
            return [dec(v) for v in o]
        return o

    return dec(header["value"])


# ----------------------------------------------------------------------------- settling


def _slice(corpus: MultiVectorCorpus, lo: int, hi: int) -> MultiVectorCorpus:
    """Documents ``lo:hi`` as views (no copy, so a memory-mapped corpus stays on disk)."""
    off = corpus.offsets
    a, b = int(off[lo]), int(off[hi])

    def take(x: np.ndarray | None) -> np.ndarray | None:
        return None if x is None else x[a:b]

    return MultiVectorCorpus(
        corpus.vectors[a:b],
        off[lo:hi + 1] - a,
        ids=corpus.ids[lo:hi],
        metadata=corpus.metadata[lo:hi],
        protected=take(corpus.protected),
        positions=take(corpus.positions),
        weights=take(corpus.weights),
        importance=take(corpus.importance),
        images=None if corpus.images is None else corpus.images[lo:hi],
        attrs=corpus.attrs,
    )


def split(corpus: MultiVectorCorpus, segment_pages: int = SEGMENT_PAGES) -> Iterator[MultiVectorCorpus]:
    """Consecutive segments of ``segment_pages`` documents (the last one shorter), as views."""
    if not isinstance(corpus, MultiVectorCorpus):
        raise TypeError(f"expected a MultiVectorCorpus, got {type(corpus).__name__}")
    if isinstance(segment_pages, bool) or not isinstance(segment_pages, int) or segment_pages < 1:
        raise ValueError(f"segment_pages must be a positive integer, got {segment_pages!r}")
    n = len(corpus)
    for lo in range(0, n, segment_pages):
        yield _slice(corpus, lo, min(lo + segment_pages, n))


def _segments_of(source: Any, segment_pages: int) -> Iterator[MultiVectorCorpus]:
    if isinstance(source, MultiVectorCorpus):
        yield from split(source, segment_pages)
        return
    if not isinstance(source, Iterable):
        raise TypeError(f"expected a MultiVectorCorpus or an iterable of them, got {type(source).__name__}")
    for seg in source:
        if not isinstance(seg, MultiVectorCorpus):
            raise TypeError(f"every segment must be a MultiVectorCorpus, got {type(seg).__name__}")
        yield seg


def _require_fitted(pipeline: Any) -> Pipeline:
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"expected an optivision Pipeline, got {type(pipeline).__name__}")
    if not pipeline.fitted:
        raise RuntimeError(f"{pipeline.label()} has unfitted stages: fit it on a sample of the pages "
                           "(Pipeline.fit) before settling or ranking with it")
    return pipeline


def settle(segments: MultiVectorCorpus | Iterable[MultiVectorCorpus], pipeline: Pipeline, *,
           segment_pages: int = SEGMENT_PAGES) -> Iterator[CompressedCorpus]:
    """Compress segment by segment with an already fitted pipeline (FINAL_DESIGN 5.7, step 2).

    ``segments`` is an iterable of float16 segments, or one corpus (memory-mapped, say) that is
    cut into ``segment_pages``-page views. Each segment goes through
    ``pipeline.compress(segment, fit=False)``, so a segment's codes are exactly what the
    pipeline gives for it, and only one segment is in memory at a time. Every stage works on
    one page or one vector at a time once fitted, so the segments' codes put together equal
    the codes of the whole corpus compressed in one go.
    """
    _require_fitted(pipeline)
    for seg in _segments_of(segments, segment_pages):
        yield pipeline.compress(seg, fit=False)


def _vector_key(pipeline: Pipeline) -> str:
    from ..core.stages.state import fitted_state_digest

    stages = pipeline.vector_stages
    return json.dumps([[s.to_dict() for s in stages], [fitted_state_digest(s) for s in stages]],
                      sort_keys=True, default=str)


def settle_many(segments: MultiVectorCorpus | Iterable[MultiVectorCorpus], pipelines: Sequence[Pipeline], *,
                segment_pages: int = SEGMENT_PAGES) -> Iterator[list[CompressedCorpus]]:
    """Several fitted pipelines over the same segments, one list per segment (aligned with
    ``pipelines``). Pipelines whose token and dimension stages are equal (same parameters,
    same fitted state) share one transform per segment, and each codec then encodes it:
    a merge depth is computed once, not once per codec (FINAL_DESIGN 5.7, step 2). The codes
    equal ``pipeline.compress(segment, fit=False)`` for each pipeline."""
    from ..core.retrieve.checks import first_non_finite_row

    pipelines = [_require_fitted(p) for p in pipelines]
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(pipelines):
        groups.setdefault(_vector_key(p), []).append(i)
    for seg in _segments_of(segments, segment_pages):
        row = first_non_finite_row(seg.vectors) if seg.num_vectors else None
        if row is not None:
            doc = int(np.searchsorted(seg.offsets, row, side="right")) - 1
            raise ValueError(f"documents contain non-finite values (first at vector {row}, document {doc})")
        out: list[CompressedCorpus | None] = [None] * len(pipelines)
        for members in groups.values():
            transformed = pipelines[members[0]].transform(seg)
            for i in members:
                out[i] = pipelines[i].encode(transformed, original=seg)
        yield out  # type: ignore[misc]


# ----------------------------------------------------------------------------- the store


def _clear_folder(path: Path) -> None:
    """Remove what an index folder holds, except its lock; refuse a folder with other content."""
    if not path.exists():
        return
    unknown = sorted(p.name for p in path.iterdir() if p.name not in _KNOWN_ENTRIES and not p.name.endswith(".tmp"))
    if unknown:
        shown = ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        raise ValueError(f"{path} holds files that are not part of an index ({shown}); "
                         "choose another path for the index")
    for p in path.iterdir():
        if p.name != LOCK_NAME:
            _remove_tree(p)


def _plan_groups(sizes: Sequence[int], max_segments: int) -> list[list[int]]:
    """Adjacent groups of segments, at most ``max_segments`` of them: repeatedly join the
    neighbouring pair with the fewest pages (the leftmost pair on a tie). Adjacent only, so
    store order, and with it every tie-break, is unchanged."""
    groups = [[i] for i in range(len(sizes))]
    pages = [int(s) for s in sizes]
    while len(groups) > max_segments:
        sums = [pages[j] + pages[j + 1] for j in range(len(groups) - 1)]
        j = int(np.argmin(sums))
        groups[j:j + 2] = [groups[j] + groups[j + 1]]
        pages[j:j + 2] = [sums[j]]
    return groups


def _rows_of(offsets: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Vector rows of the given documents, in order."""
    starts = offsets[docs]
    counts = offsets[docs + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(total, dtype=np.int64) - first + np.repeat(starts, counts)


def _concat_npy(path: Path, parts: Sequence[np.ndarray], dtype: Any) -> None:
    """Write the row-wise concatenation of ``parts`` (memory-mapped, say) in bounded steps."""
    width = parts[0].shape[1] if parts else 0
    total = sum(int(p.shape[0]) for p in parts)
    if total == 0:
        _save_npy(path, np.zeros((0, width), dtype=dtype))
        return
    out = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(total, width))
    try:
        row = 0
        for part in parts:
            step = max(1, _COPY_BYTES // max(1, int(part.shape[1]) * np.dtype(dtype).itemsize))
            for lo in range(0, int(part.shape[0]), step):
                block = np.asarray(part[lo:lo + step], dtype=dtype)
                out[row:row + block.shape[0]] = block
                row += block.shape[0]
        out.flush()
    finally:
        del out
    _fsync_file(path)


class Store:
    """An ``.ovx`` index folder. See the module docstring for the layout and the guarantees.

    Construct with :meth:`create` or :meth:`open`. A writable store (``mode="a"``) holds the
    writer lock until :meth:`close`; changes are staged and become the index only at
    :meth:`commit`. The store's own view (``segments()``, ``ids``, ``manifest``) includes
    staged changes, so a writer can search what it is about to commit.
    """

    def __init__(self, path: Path, mode: str) -> None:  # use Store.create / Store.open
        self.path = Path(path)
        self.mode = mode
        self._manifest: dict[str, Any] = {}
        self._committed: dict[str, Any] = {}
        self._pipeline: Pipeline | None = None
        self._segments: dict[str, CompressedCorpus] = {}
        self._f16: dict[str, MultiVectorCorpus] = {}
        self._seg_ids: dict[str, list[str]] = {}
        self._texts: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._id_pos: dict[str, int] | None = None
        self._id_list: list[str] | None = None
        self._pending: dict[str, bytes | None] = {}
        self._side_cache: dict[str, Any] = {}
        self._garbage: set[str] = set()
        self._locked = False
        self._closed = False

    # ------------------------------------------------------------------ construction

    @classmethod
    def create(cls, path: str | os.PathLike[str], *, pipeline: Pipeline, info: Mapping[str, Any] | None = None,
               overwrite: bool = False) -> Store:
        """A new, empty, writable index at ``path`` for codes of the fitted ``pipeline``.

        ``info`` goes into the manifest (target, layout, model, ...; JSON only, not the
        store's own keys). Nothing is an index until :meth:`commit`. An existing index at
        ``path`` raises ``FileExistsError`` unless ``overwrite=True``, which deletes it (a
        rebuild discards the old index). A folder holding files that do not belong to an index
        is never cleared.
        """
        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"pipeline must be an optivision Pipeline, got {type(pipeline).__name__}")
        if not pipeline.fitted:
            raise ValueError(f"{pipeline.label()} has unfitted stages: fit it on a sample of the pages "
                             "(Pipeline.fit) before creating the store")
        extra = _check_updates(info or {}, "info")
        store = cls(Path(path), "a")
        store._acquire()
        try:
            if (store.path / MANIFEST).exists() and not overwrite:
                raise FileExistsError(f"{store.path} already holds an index: Store.open(path, mode='a') "
                                      "adds to it, overwrite=True replaces it")
            _clear_folder(store.path)
            doc, arrays = _state_payload(pipeline)
            restored = _restore_pipeline(json.loads(json.dumps(doc)), arrays)  # proves the round trip
            if restored.to_dict() != pipeline.to_dict():
                raise ValueError(f"{pipeline.label()} does not rebuild from its to_dict(); it cannot be stored")
            state = store.path / "state"
            state.mkdir(parents=True, exist_ok=True)
            files: dict[str, str] = {}
            if arrays:
                buf = io.BytesIO()
                np.savez(buf, **arrays)
                _write_bytes(state / "fitted.npz", buf.getvalue())
                files["fitted.npz"] = hashlib.sha256(buf.getvalue()).hexdigest()
            blob = json.dumps(doc, indent=1, sort_keys=True).encode("utf-8")
            _write_bytes(state / "pipeline.json", blob)
            files["pipeline.json"] = hashlib.sha256(blob).hexdigest()
            stamp = _now()
            store._manifest = {
                "format": FORMAT,
                "version": FORMAT_VERSION,
                "created": stamp,
                "updated": stamp,
                "generation": 0,
                "dim": None,
                "pipeline": {"spec": doc["pipeline"], "label": doc["label"], "digests": doc["digests"],
                             "state_sha256": files},
                "segments": [],
                "next_segment": 0,
                "n_pages": 0,
                "n_vectors": 0,
                "float16": None,
                "files": {},
                **extra,
            }
            store._committed = {}
            store._pipeline = restored
        except BaseException:
            store._release()
            raise
        return store

    @classmethod
    def open(cls, path: str | os.PathLike[str], mode: str = "r") -> Store:
        """The committed index at ``path``: read-only (``"r"``, no lock) or writable (``"a"``).

        Opening for writing takes the writer lock (``IndexLocked`` if another writer holds it)
        and deletes what an interrupted write left behind.
        """
        if mode not in ("r", "a"):
            raise ValueError(f"mode must be 'r' or 'a', got {mode!r}")
        path = Path(path)
        if not (path / MANIFEST).is_file():
            hint = (" (an interrupted build leaves a folder like this; run the build again to finish it)"
                    if path.is_dir() else "")
            raise FileNotFoundError(f"{path} is not an index: it has no {MANIFEST}{hint}")
        store = cls(path, mode)
        if mode == "a":
            store._acquire()
        try:
            store._read_manifest()
            if mode == "a":
                store._clean_orphans()
        except BaseException:
            store._release()
            raise
        return store

    @staticmethod
    def exists(path: str | os.PathLike[str]) -> bool:
        """True when ``path`` holds a committed index."""
        return (Path(path) / MANIFEST).is_file()

    def _read_manifest(self) -> None:
        raw = (self.path / MANIFEST).read_text(encoding="utf-8")
        manifest = json.loads(raw)
        if manifest.get("format") != FORMAT:
            raise ValueError(f"{self.path / MANIFEST} is not an easy-mode index manifest")
        if int(manifest.get("version", 0)) > FORMAT_VERSION:
            raise ValueError(f"{self.path} was written by a newer optivision (index format "
                             f"{manifest['version']}); upgrade the package to read it")
        self._committed = manifest
        self._manifest = copy.deepcopy(manifest)

    # ------------------------------------------------------------------ lock and lifetime

    def _acquire(self) -> None:
        acquire_lock(self.path)
        self._locked = True

    def _release(self) -> None:
        if self._locked:
            self._locked = False
            release_lock(self.path)

    @contextmanager
    def lock(self) -> Iterator[Store]:
        """Hold the writer lock for the ``with`` block (re-entrant; a writable store holds it
        already). Raises ``IndexLocked`` when another writer holds it."""
        self._check_open()
        acquire_lock(self.path)
        try:
            yield self
        finally:
            release_lock(self.path)

    def close(self) -> None:
        """Drop every memory map, discard uncommitted changes, release the lock.

        After ``close()`` the folder can be deleted, also on Windows, as long as the caller
        holds no array obtained from the store.
        """
        if self._closed:
            return
        try:
            if self.mode == "a" and self._locked and self._has_staged():
                self._discard()
        finally:
            self._drop_maps()
            self._closed = True
            self._release()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # a forgotten writer must not keep the lock for the process lifetime
        try:
            if not getattr(self, "_closed", True) and getattr(self, "_locked", False):
                self._locked = False
                release_lock(self.path)
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    def reload(self) -> None:
        """Re-read the committed manifest (another process may have committed since)."""
        self._check_open()
        if self.mode == "a" and self._has_staged():
            raise RuntimeError("the store has uncommitted changes; commit() or close() before reload()")
        self._drop_maps()
        self._pipeline = None
        self._read_manifest()

    def _drop_maps(self) -> None:
        self._segments.clear()
        self._f16.clear()
        self._texts.clear()
        self._seg_ids.clear()
        self._id_pos = None
        self._id_list = None
        gc.collect()

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError(f"the store at {self.path} is closed")

    def _need_writer(self) -> None:
        self._check_open()
        if self.mode != "a":
            raise PermissionError(f"{self.path} was opened read-only; Store.open(path, mode='a') to change it")
        if not self._locked or not holds_lock(self.path):
            raise IndexLocked(f"this store no longer holds the writer lock of {self.path}")

    def _has_staged(self) -> bool:
        return bool(self._pending) or self._manifest != self._committed

    # ------------------------------------------------------------------ what it holds

    @property
    def manifest(self) -> dict[str, Any]:
        """A copy of the manifest, staged changes included."""
        return copy.deepcopy(self._manifest)

    @property
    def writable(self) -> bool:
        return self.mode == "a" and not self._closed

    @property
    def dirty(self) -> bool:
        """True when there are changes that :meth:`commit` has not published yet."""
        return self._has_staged()

    @property
    def n_pages(self) -> int:
        return int(self._manifest["n_pages"])

    @property
    def n_vectors(self) -> int:
        return int(self._manifest["n_vectors"])

    @property
    def n_segments(self) -> int:
        return len(self._manifest["segments"])

    @property
    def dim(self) -> int | None:
        """Dimension of the stored (decoded) vectors; None before the earliest append."""
        dim = self._manifest["dim"]
        return None if dim is None else int(dim)

    @property
    def has_float16(self) -> bool:
        return bool(self._manifest["float16"])

    @property
    def log_dir(self) -> Path:
        """Where the opt-in query log lives (not created here)."""
        return self.path / "log"

    def _entries(self) -> list[dict[str, Any]]:
        return self._manifest["segments"]

    def _ids_of(self, entry: Mapping[str, Any]) -> list[str]:
        name = entry["name"]
        ids = self._seg_ids.get(name)
        if ids is None:
            ids = json.loads((self.path / "segments" / name / "ids.json").read_text(encoding="utf-8"))
            if len(ids) != entry["n_pages"]:
                raise ValueError(f"segment {name} is damaged: {len(ids)} ids for {entry['n_pages']} pages")
            self._seg_ids[name] = ids
        return ids

    def _all_ids(self) -> list[str]:
        if self._id_list is None:
            out: list[str] = []
            for entry in self._entries():
                out.extend(self._ids_of(entry))
            self._id_list = out
        return self._id_list

    def _positions(self) -> dict[str, int]:
        if self._id_pos is None:
            self._id_pos = {pid: i for i, pid in enumerate(self._all_ids())}
        return self._id_pos

    def _invalidate_ids(self) -> None:
        self._id_pos = None
        self._id_list = None

    @property
    def ids(self) -> list[str]:
        """Every stored page id, in store order (the order ties are broken in)."""
        self._check_open()
        return list(self._all_ids())

    def position(self, page_id: str) -> int:
        """The store position of ``page_id`` (``KeyError`` if it is not stored)."""
        self._check_open()
        return self._positions()[page_id]

    def __contains__(self, page_id: object) -> bool:
        return isinstance(page_id, str) and page_id in self._positions()

    def __len__(self) -> int:
        return self.n_pages

    def locate(self, page_id: str) -> tuple[int, int]:
        """``(segment index, position inside the segment)`` of ``page_id``."""
        pos = self.position(page_id)
        for s, entry in enumerate(self._entries()):
            if pos < entry["n_pages"]:
                return s, pos
            pos -= entry["n_pages"]
        raise KeyError(page_id)  # pragma: no cover - positions and entries agree

    def pipeline(self) -> Pipeline:
        """The fitted pipeline the codes were made with (restored, digests checked; shared by
        the segments, so do not fit it again)."""
        self._check_open()
        if self._pipeline is None:
            self._pipeline = self._read_pipeline()
        return self._pipeline

    def _read_pipeline(self) -> Pipeline:
        state = self.path / "state"
        blob = (state / "pipeline.json").read_bytes()
        want = self._manifest["pipeline"]
        sha = want.get("state_sha256", {})
        if sha.get("pipeline.json") and hashlib.sha256(blob).hexdigest() != sha["pipeline.json"]:
            raise ValueError(f"{state / 'pipeline.json'} does not belong to this index (sha256 differs)")
        doc = json.loads(blob.decode("utf-8"))
        if doc.get("format") != STATE_FORMAT:
            raise ValueError(f"{state / 'pipeline.json'} is not an easy-mode pipeline state")
        if doc["pipeline"] != want["spec"] or doc["digests"] != want["digests"]:
            raise ValueError(f"{state / 'pipeline.json'} describes another pipeline than the manifest")
        arrays: dict[str, np.ndarray] = {}
        if (state / "fitted.npz").exists():
            raw = (state / "fitted.npz").read_bytes()
            if sha.get("fitted.npz") and hashlib.sha256(raw).hexdigest() != sha["fitted.npz"]:
                raise ValueError(f"{state / 'fitted.npz'} does not belong to this index (sha256 differs)")
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        return _restore_pipeline(doc, arrays)

    # ------------------------------------------------------------------ segments

    def _load_segment(self, entry: Mapping[str, Any], pipeline: Pipeline) -> CompressedCorpus:
        name = entry["name"]
        d = self.path / "segments" / name
        n_pages, n_vectors, width = int(entry["n_pages"]), int(entry["n_vectors"]), int(entry["code_bytes"])
        codes = _load_npy(d / "codes.npy", mmap=n_vectors > 0)
        offsets = _load_npy(d / "offsets.npy", mmap=False)
        if codes.dtype != np.uint8 or codes.shape != (n_vectors, width):
            raise ValueError(f"segment {name} is damaged: codes {codes.dtype}{codes.shape}, "
                             f"expected uint8({n_vectors}, {width})")
        if (offsets.dtype != np.int64 or offsets.shape != (n_pages + 1,) or int(offsets[0]) != 0
                or int(offsets[-1]) != n_vectors or bool(np.any(np.diff(offsets) < 0))):
            raise ValueError(f"segment {name} is damaged: offsets do not describe {n_pages} pages "
                             f"and {n_vectors} vectors")
        return CompressedCorpus(codes, offsets, int(self._manifest["dim"]), pipeline.quantizer, self._ids_of(entry),
                                stats=dict(entry["stats"]), pipeline=copy.deepcopy(self._manifest["pipeline"]["spec"]))

    def _segment(self, entry: Mapping[str, Any]) -> CompressedCorpus:
        seg = self._segments.get(entry["name"])
        if seg is None:
            seg = self._load_segment(entry, self.pipeline())
            self._segments[entry["name"]] = seg
        return seg

    def segments(self) -> list[CompressedCorpus]:
        """The segments in store order, codes memory-mapped (staged ones included)."""
        self._check_open()
        return [self._segment(e) for e in self._entries()]

    def segment_bases(self) -> list[int]:
        """Store position of each segment's leading page."""
        bases, at = [], 0
        for entry in self._entries():
            bases.append(at)
            at += int(entry["n_pages"])
        return bases

    def _float16_segment(self, entry: Mapping[str, Any]) -> MultiVectorCorpus:
        name = entry["name"]
        seg = self._f16.get(name)
        if seg is None:
            d = self.path / "float16" / name
            offsets = _load_npy(d / "offsets.npy", mmap=False)
            vectors = _load_npy(d / "vectors.npy", mmap=int(offsets[-1]) > 0)
            seg = MultiVectorCorpus(vectors, offsets, ids=self._ids_of(entry))
            self._f16[name] = seg
        return seg

    def float16_segments(self) -> list[MultiVectorCorpus] | None:
        """The kept float16 copy, one memory-mapped corpus per segment; None if not kept."""
        self._check_open()
        if not self._manifest["float16"]:
            return None
        return [self._float16_segment(e) for e in self._entries()]

    def float16_corpus(self) -> MultiVectorCorpus | None:
        """The kept float16 copy as one corpus; None if not kept. With several segments this
        reads the copy into memory: ``float16_segments()`` streams it instead."""
        segs = self.float16_segments()
        if segs is None:
            return None
        if len(segs) == 1:
            return segs[0]
        if not segs:  # every page was removed
            return MultiVectorCorpus(np.zeros((0, 0), dtype=np.float16), np.zeros(1, dtype=np.int64))
        vectors = np.concatenate([np.asarray(s.vectors) for s in segs], axis=0)
        offsets, ids, at = [np.zeros(1, dtype=np.int64)], [], 0
        for s in segs:
            offsets.append(s.offsets[1:] + at)
            at += s.num_vectors
            ids.extend(s.ids)
        return MultiVectorCorpus(vectors, np.concatenate(offsets), ids=ids)

    # ------------------------------------------------------------------ text, aliases, record, reference

    def _text_dict(self, entry: Mapping[str, Any]) -> dict[str, str]:
        name = entry["name"]
        if not entry.get("text"):
            return {}
        cached = self._texts.get(name)
        if cached is None:
            raw = (self.path / "text" / f"{name}.zlib.json").read_bytes()
            cached = json.loads(zlib.decompress(raw).decode("utf-8"))
            self._texts[name] = cached
            while len(self._texts) > _TEXT_CACHE:
                self._texts.popitem(last=False)
        else:
            self._texts.move_to_end(name)
        return cached

    def text(self, page_id: str) -> str:
        """The stored text layer of ``page_id``, or ``""`` when none was kept."""
        s, _ = self.locate(page_id)
        return self._text_dict(self._entries()[s]).get(page_id, "")

    def _side(self, rel: str) -> Any:
        if rel in self._pending:
            data = self._pending[rel]
        elif rel in self._side_cache:
            return self._side_cache[rel]
        else:
            p = self.path / rel
            data = p.read_bytes() if p.exists() else None
        if data is None:
            value = None
        elif rel.endswith(".npz"):
            value = _unpack_reference(data)
        else:
            value = json.loads(data.decode("utf-8"))
        if rel not in self._pending:
            self._side_cache[rel] = value
        return value

    def _stage_side(self, rel: str, data: bytes | None) -> None:
        self._need_writer()
        self._pending[rel] = data
        self._side_cache.pop(rel, None)

    @property
    def aliases(self) -> dict[str, str]:
        """``{duplicate page id: id of the stored copy}`` (a copy; assign to change it)."""
        self._check_open()
        return dict(self._side("aliases.json") or {})

    @aliases.setter
    def aliases(self, value: Mapping[str, str]) -> None:
        if not isinstance(value, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()
        ):
            raise TypeError("aliases must map page ids (str) to page ids (str)")
        self._stage_side("aliases.json", json.dumps(dict(sorted(value.items())), indent=0).encode("utf-8"))

    @property
    def record(self) -> dict[str, Any] | None:
        """The build record (``build.json``), or None."""
        self._check_open()
        value = self._side("build.json")
        return copy.deepcopy(value)

    @record.setter
    def record(self, value: Mapping[str, Any] | None) -> None:
        if value is None:
            self._stage_side("build.json", None)
            return
        if not isinstance(value, Mapping):
            raise TypeError(f"the build record must be a dict, got {type(value).__name__}")
        plain = _jsonable(value, "the build record")
        self._stage_side("build.json", json.dumps(plain, indent=1, sort_keys=True, allow_nan=False).encode("utf-8"))

    @property
    def reference(self) -> dict[str, Any] | None:
        """The practice-check reference written by the build (probe texts, query vectors,
        relevant sets, float16 scores): an opaque dict whose arrays come back as arrays.
        Treat it as read-only; assign a new dict to change it. None when there is none."""
        self._check_open()
        return self._side("reference/reference.npz")

    @reference.setter
    def reference(self, value: Mapping[str, Any] | None) -> None:
        self._stage_side("reference/reference.npz", None if value is None else _pack_reference(value))

    # ------------------------------------------------------------------ statistics and sizes

    def stats(self) -> dict[str, Any]:
        """Totals over the segments: pages, vectors before and after the pipeline, code and
        codec-state bytes, and the float32 / float16 / native sizes of the vectors before the
        pipeline. ``exact`` is False when a removal had to estimate a segment's "before" sizes
        (its per-page vector counts were not known)."""
        self._check_open()
        entries = self._entries()
        before = {k: sum(int(e["stats"].get(k, 0)) for e in entries) for k in _ADDITIVE_STATS}
        dim_before = next((int(e["stats"]["dim_before"]) for e in entries if "dim_before" in e["stats"]), None)
        dtypes = sorted({str(e["stats"]["native_dtype_before"]) for e in entries
                         if "native_dtype_before" in e["stats"]})
        codes = sum(int(e["n_vectors"]) * int(e["code_bytes"]) for e in entries)
        overhead = int(self.pipeline().quantizer.overhead_bytes())
        return {
            "n_pages": self.n_pages,
            "n_segments": len(entries),
            "vectors": self.n_vectors,
            "dim": self.dim,
            "vectors_before": before["vectors_before"],
            "dim_before": dim_before,
            "codes_bytes": codes,
            "overhead_bytes": overhead,
            "bytes": codes + overhead,
            "float32_bytes_before": before["float32_bytes_before"],
            "float16_bytes_before": before["vectors_before"] * (dim_before or 0) * 2,
            "native_bytes_before": before["native_bytes_before"],
            "native_dtype_before": dtypes[0] if len(dtypes) == 1 else (dtypes or None),
            "exact": all(e.get("stats_exact", True) for e in entries),
        }

    def _category(self, parts: tuple[str, ...], live: set[str]) -> str:
        head = parts[0]
        if len(parts) == 1:
            return "record" if head in (MANIFEST, "build.json", "aliases.json") else "other"
        if head == "segments":
            if len(parts) == 3 and parts[1] in live:
                return "vectors" if parts[2] in ("codes.npy", "offsets.npy") else "record"
            return "other"
        if head == "state":
            return "vectors" if parts[-1] == "fitted.npz" else ("record" if parts[-1] == "pipeline.json" else "other")
        if head == "text":
            stem = parts[-1][: -len(".zlib.json")] if parts[-1].endswith(".zlib.json") else None
            return "text" if len(parts) == 2 and stem in live else "other"
        if head == "float16":
            return "float16" if len(parts) == 3 and parts[1] in live else "other"
        if head == "reference":
            return "reference" if parts[-1] == "reference.npz" else "other"
        if head == "log":
            return "log"
        return "other"

    def sizes(self) -> dict[str, int]:
        """Bytes on disk by category: ``vectors`` (codes, offsets, fitted codec state),
        ``text``, ``record`` (manifest, build record, aliases, ids, pipeline description),
        ``reference``, ``float16``, ``log``, ``other`` (temporary and orphan files) and
        ``total``, the sum of every file's size in the folder. The writer lock is left out:
        it exists only while a writer works, so the total is what stays on disk afterwards.
        Call it after :meth:`commit`: staged side files are not on disk before."""
        self._check_open()
        live = {e["name"] for e in self._entries()}
        out = dict.fromkeys(("vectors", "text", "record", "reference", "float16", "log", "other"), 0)
        for root, _dirs, files in os.walk(self.path):
            for name in files:
                p = Path(root) / name
                if name == LOCK_NAME and Path(root) == self.path:
                    continue
                try:
                    size = p.stat().st_size
                except FileNotFoundError:
                    continue
                out[self._category(p.relative_to(self.path).parts, live)] += size
        out["total"] = sum(out.values())
        return out

    def verify(self) -> list[str]:
        """Re-check everything against the manifest (fitted-state digests, code digests,
        shapes, side-file sha256); returns the problems found, empty when none."""
        self._check_open()
        problems: list[str] = []
        try:
            pipeline = self._read_pipeline()
        except Exception as exc:  # a report, not a control path: every failure is listed
            return [f"fitted pipeline: {exc}"]
        for entry in self._entries():
            name = entry["name"]
            try:
                seg = self._load_segment(entry, pipeline)
                if array_digest(seg.codes) != entry["codes_digest"]:
                    problems.append(f"segment {name}: codes differ from the digest recorded at write time")
                del seg
                if entry.get("float16"):
                    d = self.path / "float16" / name
                    offsets = _load_npy(d / "offsets.npy", mmap=False)
                    vectors = _load_npy(d / "vectors.npy", mmap=int(offsets[-1]) > 0)
                    if offsets.shape != (entry["n_pages"] + 1,) or int(offsets[-1]) != vectors.shape[0]:
                        problems.append(f"float16 copy of segment {name}: shapes disagree")
                    del vectors
                if entry.get("text"):
                    self._texts.pop(name, None)
                    self._text_dict(entry)
            except Exception as exc:  # listed, not raised
                problems.append(f"segment {name}: {exc}")
        for rel, meta in self._committed.get("files", {}).items():
            p = self.path / rel
            if not p.exists():
                problems.append(f"{rel}: missing")
            elif _sha256_file(p) != meta["sha256"]:
                problems.append(f"{rel}: differs from the sha256 in the manifest")
        gc.collect()
        return problems

    # ------------------------------------------------------------------ writing

    def _next_name(self) -> str:
        name = _seg_name(int(self._manifest["next_segment"]))
        self._manifest["next_segment"] = int(self._manifest["next_segment"]) + 1
        return name

    def _write_segment(self, name: str, codes: np.ndarray, offsets: np.ndarray, ids: list[str],
                       before: np.ndarray | None, texts: Mapping[str, str] | None,
                       f16: tuple[np.ndarray, np.ndarray] | None) -> None:
        d = self.path / "segments" / name
        if d.exists():  # an orphan of an earlier crash under the same number: never referenced
            _remove_tree(d)
        d.mkdir(parents=True)
        _save_npy(d / "codes.npy", codes)
        _save_npy(d / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        _write_bytes(d / "ids.json", json.dumps(ids, ensure_ascii=False).encode("utf-8"))
        if before is not None:
            _save_npy(d / "before.npy", np.asarray(before, dtype=np.int64))
        if texts:
            t = self.path / "text"
            t.mkdir(exist_ok=True)
            blob = json.dumps(dict(texts), ensure_ascii=False, sort_keys=True).encode("utf-8")
            _write_bytes(t / f"{name}.zlib.json", zlib.compress(blob, 9))
        if f16 is not None:
            f = self.path / "float16" / name
            if f.exists():
                _remove_tree(f)
            f.mkdir(parents=True)
            _save_npy(f / "vectors.npy", np.asarray(f16[0], dtype=np.float16))
            _save_npy(f / "offsets.npy", np.asarray(f16[1], dtype=np.int64))
        _fsync_dir(d)

    def _recount(self) -> None:
        entries = self._entries()
        self._manifest["n_pages"] = sum(int(e["n_pages"]) for e in entries)
        self._manifest["n_vectors"] = sum(int(e["n_vectors"]) for e in entries)
        self._invalidate_ids()

    def append(self, compressed: CompressedCorpus, texts: Mapping[str, str] | None = None,
               float16: MultiVectorCorpus | None = None, *,
               vectors_before: Sequence[int] | np.ndarray | None = None) -> str:
        """Add ``compressed`` as a new segment (staged until :meth:`commit`); returns its name.

        ``compressed`` must come from this store's pipeline (same stages, same fitted codec
        state: compress with ``store.pipeline()``); page ids must be new. ``texts`` maps some
        or all of its ids to their text layer. ``float16`` is the segment before the pipeline
        (same ids), kept as the float16 copy: every segment has one or none does.
        ``vectors_before`` gives each page's vector count before the pipeline, so that a
        later ``remove`` keeps the size statistics exact; it is taken from ``float16`` when
        given, and equals the stored counts for a pipeline without token stages.
        """
        from ..core.stages.state import fitted_state_digest

        self._need_writer()
        if not isinstance(compressed, CompressedCorpus):
            raise TypeError(f"expected a CompressedCorpus, got {type(compressed).__name__}")
        n = len(compressed)
        if n == 0:
            raise ValueError("cannot append a segment without pages")
        pipeline = self.pipeline()
        spec = self._manifest["pipeline"]["spec"]
        if compressed.pipeline is not None and compressed.pipeline != spec:
            got = Pipeline.from_dict(compressed.pipeline).label()
            raise ValueError(f"the codes come from {got}; this index stores {self._manifest['pipeline']['label']}")
        want_q, got_q = pipeline.quantizer, compressed.quantizer
        if got_q.to_dict() != want_q.to_dict():
            raise ValueError(f"the codes use the {got_q.to_dict()} codec; this index stores {want_q.to_dict()}")
        if fitted_state_digest(got_q) != fitted_state_digest(want_q):
            raise ValueError(f"the codes were encoded with another fitted {want_q.name} state (another centre) "
                             "than this index's pipeline; compress the pages with store.pipeline()")
        dim = self._manifest["dim"]
        if dim is not None and int(dim) != compressed.dim:
            raise ValueError(f"the codes hold {compressed.dim}-dimensional vectors; this index holds {dim}")
        width = want_q.code_bytes(compressed.dim)
        if compressed.codes.ndim != 2 or compressed.codes.shape[1] != width:
            raise ValueError(f"codes are {compressed.codes.shape} bytes; {want_q.name} at dimension "
                             f"{compressed.dim} has {width} bytes per vector")
        ids = [str(i) for i in compressed.ids]
        if len(set(ids)) != n:
            raise ValueError("the segment repeats page ids")
        clash = [i for i in ids if i in self._positions()]
        if clash:
            raise ValueError(f"{len(clash)} page ids are stored already, e.g. {clash[:3]}; remove() them before "
                             "adding new versions")
        if texts:
            if not isinstance(texts, Mapping):
                raise TypeError("texts must map page ids to their text")
            unknown = sorted(set(texts) - set(ids))
            if unknown:
                raise ValueError(f"texts name pages that are not in the segment, e.g. {unknown[:3]}")
            if not all(isinstance(v, str) for v in texts.values()):
                raise TypeError("every text must be a string")
            texts = {k: v for k, v in texts.items() if v}
        have16 = self._manifest["float16"]
        if have16 is not None and self._entries() and bool(have16) != (float16 is not None):
            raise ValueError("this index keeps a float16 copy of every page: pass float16=" if have16 else
                             "this index keeps no float16 copy: do not pass float16=")
        f16 = None
        if float16 is not None:
            if not isinstance(float16, MultiVectorCorpus):
                raise TypeError(f"float16 must be a MultiVectorCorpus, got {type(float16).__name__}")
            if list(float16.ids) != ids:
                raise ValueError("the float16 copy must hold the same pages, in the same order, as the codes")
            dim_before = compressed.stats.get("dim_before")
            if dim_before is not None and float16.dimension != int(dim_before):
                raise ValueError(f"the float16 copy is {float16.dimension}-dimensional; the codes were made "
                                 f"from {dim_before}-dimensional vectors")
            f16 = (np.asarray(float16.vectors), np.asarray(float16.offsets))
        before: np.ndarray | None
        if vectors_before is not None:
            before = np.asarray(vectors_before, dtype=np.int64).reshape(-1)
            if before.shape != (n,) or bool(np.any(before < 0)):
                raise ValueError(f"vectors_before must hold {n} counts >= 0")
        elif float16 is not None:
            before = float16.counts.astype(np.int64)
        elif not _has_token_stage(pipeline):
            before = compressed.counts.astype(np.int64)
        else:
            before = None
        stats = _jsonable(dict(compressed.stats), "the segment statistics")
        if before is not None and "vectors_before" in stats and int(before.sum()) != int(stats["vectors_before"]):
            raise ValueError(f"vectors_before adds up to {int(before.sum())} but the codes were made from "
                             f"{stats['vectors_before']} vectors")
        name = self._next_name()
        self._write_segment(name, compressed.codes, compressed.offsets, ids, before, texts, f16)
        entry = {
            "name": name,
            "n_pages": n,
            "n_vectors": int(compressed.num_vectors),
            "code_bytes": int(width),
            "codes_digest": array_digest(compressed.codes),
            "text": bool(texts),
            "float16": f16 is not None,
            "before": before is not None,
            "stats_exact": True,
            "stats": stats,
        }
        if dim is None:
            self._manifest["dim"] = int(compressed.dim)
        self._manifest["float16"] = f16 is not None
        self._entries().append(entry)
        self._recount()
        return name

    def remove(self, ids: Iterable[str], *, missing_ok: bool = False) -> int:
        """Delete pages (prune; staged until :meth:`commit`). Each segment that holds one is
        rewritten without it under a new number; a segment left empty goes. Store order of the
        remaining pages does not change. Returns how many pages were removed; ids that are not
        stored raise ``KeyError`` unless ``missing_ok``."""
        self._need_writer()
        if isinstance(ids, str):
            ids = [ids]
        want = {str(i) for i in ids}
        positions = self._positions()
        unknown = sorted(want - positions.keys())
        if unknown and not missing_ok:
            raise KeyError(f"{len(unknown)} page ids are not in the index, e.g. {unknown[:3]}")
        want &= positions.keys()
        if not want:
            return 0
        kept: list[dict[str, Any]] = []
        for entry in self._entries():
            seg_ids = self._ids_of(entry)
            drop = np.array([i for i, pid in enumerate(seg_ids) if pid in want], dtype=np.int64)
            if drop.size == 0:
                kept.append(entry)
                continue
            self._garbage.add(entry["name"])
            keep = np.setdiff1d(np.arange(len(seg_ids), dtype=np.int64), drop)
            if keep.size:
                kept.append(self._rewrite(entry, keep))
            self._forget(entry["name"])
        self._manifest["segments"] = kept
        self._recount()
        return len(want)

    def _forget(self, name: str) -> None:
        self._segments.pop(name, None)
        self._f16.pop(name, None)
        self._texts.pop(name, None)
        self._seg_ids.pop(name, None)

    def _rewrite(self, entry: Mapping[str, Any], keep: np.ndarray) -> dict[str, Any]:
        """A new segment holding the pages ``keep`` (positions inside ``entry``)."""
        seg = self._segment(entry)
        seg_ids = self._ids_of(entry)
        rows = _rows_of(seg.offsets, keep)
        codes = np.asarray(seg.codes[rows])
        counts = seg.counts[keep]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        ids = [seg_ids[i] for i in keep]
        texts = None
        if entry.get("text"):
            whole = self._text_dict(entry)
            texts = {pid: whole[pid] for pid in ids if pid in whole}
        f16 = None
        if entry.get("float16"):
            f = self._float16_segment(entry)
            frows = _rows_of(f.offsets, keep)
            fcounts = f.counts[keep]
            f16 = (np.asarray(f.vectors[frows]), np.concatenate([[0], np.cumsum(fcounts)]).astype(np.int64))
        stats = dict(entry["stats"])
        exact = bool(entry.get("stats_exact", True))
        before = None
        if entry.get("before"):
            before = _load_npy(self.path / "segments" / entry["name"] / "before.npy", mmap=False)[keep]
            n_before = int(before.sum())
        else:
            # per-page counts unknown: scale by the share of stored vectors kept, and say so
            share = (rows.size / seg.num_vectors) if seg.num_vectors else (keep.size / len(seg_ids))
            n_before = round(int(stats.get("vectors_before", 0)) * share)
            exact = False
        if "vectors_before" in stats:
            old = int(stats["vectors_before"])
            stats["vectors_before"] = n_before
            for key in ("float32_bytes_before", "native_bytes_before"):
                if key in stats and old:
                    per_vector = int(stats[key]) // old
                    stats[key] = n_before * per_vector
        name = self._next_name()
        self._write_segment(name, codes, offsets, ids, before, texts, f16)
        del seg
        return {
            "name": name,
            "n_pages": len(ids),
            "n_vectors": int(codes.shape[0]),
            "code_bytes": int(entry["code_bytes"]),
            "codes_digest": array_digest(codes),
            "text": bool(texts),
            "float16": f16 is not None,
            "before": before is not None,
            "stats_exact": exact,
            "stats": stats,
        }

    def compact(self, max_segments: int = COMPACT_ABOVE) -> bool:
        """Merge neighbouring segments while there are more than ``max_segments`` (staged until
        :meth:`commit`); returns whether anything changed. Store order is unchanged, so search
        results are identical. Codes and the float16 copy are copied in bounded steps."""
        self._need_writer()
        if isinstance(max_segments, bool) or not isinstance(max_segments, int) or max_segments < 1:
            raise ValueError(f"max_segments must be a positive integer, got {max_segments!r}")
        entries = self._entries()
        if len(entries) <= max_segments:
            return False
        groups = _plan_groups([int(e["n_pages"]) for e in entries], max_segments)
        merged: list[dict[str, Any]] = []
        for group in groups:
            if len(group) == 1:
                merged.append(entries[group[0]])
                continue
            members = [entries[i] for i in group]
            merged.append(self._merge(members))
            for e in members:
                self._garbage.add(e["name"])
                self._forget(e["name"])
        self._manifest["segments"] = merged
        self._recount()
        gc.collect()
        return True

    def _merge(self, members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        name = self._next_name()
        d = self.path / "segments" / name
        if d.exists():
            _remove_tree(d)
        d.mkdir(parents=True)
        width = int(members[0]["code_bytes"])
        segs = [self._segment(e) for e in members]
        _concat_npy(d / "codes.npy", [s.codes for s in segs], np.uint8)
        offsets, at = [np.zeros(1, dtype=np.int64)], 0
        ids: list[str] = []
        for s, e in zip(segs, members, strict=True):
            offsets.append(np.asarray(s.offsets[1:], dtype=np.int64) + at)
            at += s.num_vectors
            ids.extend(self._ids_of(e))
        _save_npy(d / "offsets.npy", np.concatenate(offsets))
        _write_bytes(d / "ids.json", json.dumps(ids, ensure_ascii=False).encode("utf-8"))
        del segs
        has_before = all(e.get("before") for e in members)
        if has_before:
            before = np.concatenate([_load_npy(self.path / "segments" / e["name"] / "before.npy", mmap=False)
                                     for e in members])
            _save_npy(d / "before.npy", before)
        texts: dict[str, str] = {}
        for e in members:
            texts.update(self._text_dict(e))
        if texts:
            t = self.path / "text"
            t.mkdir(exist_ok=True)
            blob = json.dumps(texts, ensure_ascii=False, sort_keys=True).encode("utf-8")
            _write_bytes(t / f"{name}.zlib.json", zlib.compress(blob, 9))
        has16 = all(e.get("float16") for e in members)
        if has16:
            f = self.path / "float16" / name
            if f.exists():
                _remove_tree(f)
            f.mkdir(parents=True)
            fsegs = [self._float16_segment(e) for e in members]
            _concat_npy(f / "vectors.npy", [s.vectors for s in fsegs], np.float16)
            foff, at = [np.zeros(1, dtype=np.int64)], 0
            for s in fsegs:
                foff.append(np.asarray(s.offsets[1:], dtype=np.int64) + at)
                at += s.num_vectors
            _save_npy(f / "offsets.npy", np.concatenate(foff))
            del fsegs
        _fsync_dir(d)
        stats: dict[str, Any] = {}
        for key in _ADDITIVE_STATS:
            if all(key in e["stats"] for e in members):
                stats[key] = sum(int(e["stats"][key]) for e in members)
        for key in set().union(*(e["stats"] for e in members)) - set(_ADDITIVE_STATS):
            values = {json.dumps(e["stats"].get(key), sort_keys=True) for e in members}
            if len(values) == 1:
                stats[key] = members[0]["stats"].get(key)
        n_vectors = sum(int(e["n_vectors"]) for e in members)
        codes = _load_npy(d / "codes.npy", mmap=n_vectors > 0)
        digest = array_digest(codes)
        del codes
        return {
            "name": name,
            "n_pages": len(ids),
            "n_vectors": n_vectors,
            "code_bytes": width,
            "codes_digest": digest,
            "text": bool(texts),
            "float16": has16,
            "before": has_before,
            "stats_exact": all(e.get("stats_exact", True) for e in members),
            "stats": stats,
        }

    def commit(self, **updates: Any) -> None:
        """Publish every staged change: side files, then ``manifest.json`` (``os.replace``).

        ``updates`` are merged into the manifest (JSON; not the store's own keys), e.g.
        ``commit(target=0.97, upgrade_attempts=1)``. Segments no longer referenced are deleted
        afterwards (best effort; leftovers go when a writer next opens the index).
        """
        self._need_writer()
        extra = _check_updates(updates, "commit()")
        manifest = copy.deepcopy(self._manifest)
        manifest.update(extra)
        files = dict(manifest.get("files", {}))
        for rel, data in self._pending.items():
            target = self.path / rel
            if data is None:
                files.pop(rel, None)
                _remove_tree(target)
                continue
            _atomic_write(target, data)
            files[rel] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest["files"] = files
        manifest["generation"] = int(manifest.get("generation", 0)) + 1
        manifest["updated"] = _now()
        blob = json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
        _atomic_write(self.path / MANIFEST, blob)
        self._manifest = manifest
        self._committed = copy.deepcopy(manifest)
        for rel in list(self._pending):
            self._side_cache.pop(rel, None)
        self._pending.clear()
        live = {e["name"] for e in self._entries()}
        garbage = sorted(self._garbage - live)
        self._garbage.clear()
        if garbage:
            for name in garbage:
                self._forget(name)
            gc.collect()
            for name in garbage:
                self._delete_segment_files(name)

    def _delete_segment_files(self, name: str) -> None:
        _remove_tree(self.path / "segments" / name)
        _remove_tree(self.path / "text" / f"{name}.zlib.json")
        _remove_tree(self.path / "float16" / name)

    def _discard(self) -> None:
        """Forget staged changes and delete the files they wrote."""
        self._pending.clear()
        self._side_cache.clear()
        self._garbage.clear()
        self._drop_maps()
        if not self._committed:  # created and never committed: there is no index to keep
            _clear_folder(self.path)
            self._manifest = {}
            return
        self._manifest = copy.deepcopy(self._committed)
        self._clean_orphans()

    def _clean_orphans(self) -> None:
        """Delete what the committed manifest does not name (needs the writer lock)."""
        if not self._locked:
            return
        live = {e["name"] for e in self._committed.get("segments", [])} | {e["name"] for e in self._entries()}
        seg_root = self.path / "segments"
        if seg_root.is_dir():
            for d in seg_root.iterdir():
                if d.name not in live:
                    _remove_tree(d)
        text_root = self.path / "text"
        if text_root.is_dir():
            for f in text_root.iterdir():
                stem = f.name[: -len(".zlib.json")] if f.name.endswith(".zlib.json") else None
                if stem not in live:
                    _remove_tree(f)
        f16_root = self.path / "float16"
        if f16_root.is_dir():
            for d in f16_root.iterdir():
                if d.name not in live:
                    _remove_tree(d)
        for root in (self.path, self.path / "reference", self.path / "state"):
            if root.is_dir():
                for f in root.iterdir():
                    if f.name.endswith(".tmp"):
                        _remove_tree(f)

    def __repr__(self) -> str:
        state = "closed" if self._closed else ("writable" if self.mode == "a" else "read-only")
        label = self._manifest.get("pipeline", {}).get("label", "?") if self._manifest else "?"
        return f"Store({str(self.path)!r}, {state}, {self._manifest.get('n_pages', 0)} pages, {label})"
