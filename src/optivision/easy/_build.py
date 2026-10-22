"""The build pipeline behind ``optivision.easy.build`` and ``Index.add`` (FINAL_DESIGN 4.2, 5.5-5.7).

Order of one build (docs/easy/CONTRACT.md, ``_api.py, _build.py``):

1. classify the source, resolve the model, check the target, find the index path; an existing
   index is synced instead (``TargetMismatch`` when target, layout or model differ without
   ``rebuild=True``);
2. take the writer lock (``<path>.lock``, ``O_CREAT | O_EXCL``);
3. list the sources and diff them against the index: key ``(path, size, mtime)``, sha256 only
   when that key changed (strings and vectors: sha256 always, it is cheap);
4. plan: count pages, estimate time and disk, check free disk (``NotEnoughDisk``) and print the
   plan with the location of ``work_dir``;
5. load the encoder (only when something has to be encoded) and run the wiring check;
6. encode new and changed pages into float16 shards in ``work_dir``: 500 pages per shard, each
   shard written under a temporary name and renamed, then the shard manifest replaced
   (``os.replace``), so a killed build resumes where it stopped, also in a new session. Blank
   pages are counted and skipped, identical pages are encoded once (aliases), and a file that
   cannot be read is skipped with its reason;
7. profile the pages; practice questions from their text (``probes="text"``), ranked on the
   float16 copy, give the hard set; 60 or more make the folder "checked";
8. selection (``_levels.select``: catalogue ceiling, chain rule, margins, shape gate, unchecked
   folders without 1-bit);
9. practice check (``_veto.fixed_sequence``) and real-query gate (``queries=``); each can only
   move along the fixed sequence towards float16;
10. fit the chosen pipeline on a seeded page sample (at most 200,000 vectors, seed 0) and settle
    each shard with ``compress(segment, fit=False)`` into the store, with page text, aliases and
    the practice reference;
11. time 20 sample queries, write the manifest, print the summary, delete the shards.

The loading order differs from the contract's in one place: the encoder is loaded after the
diff, so re-running ``build`` on an unchanged folder loads no model.

Every call into another easy-mode module sits in one small adapter function in the section
"adapters" below, so a signature that differs from the contract is fixed in one place. The
manifest keys written here (read by ``_api``, ``_check``, ``_upgrade``, ``_report``) are in
:data:`MANIFEST_KEYS`.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import json
import os
import shutil
import socket
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import _claim, _ledger
from ._claim import Block, Line, claim, lit
from ._types import (
    EasyError,
    IndexLocked,
    LayoutRow,
    NotEnoughDisk,
    Selection,
    SourceMissing,
    SyncReport,
    TargetMismatch,
)

#: pages per float16 shard and per stored segment (FINAL_DESIGN 5.7 step 2)
SEGMENT_PAGES = 500
#: the fit sample: at most this many vectors, drawn page by page with this seed
FIT_ROWS = 200_000
FIT_SEED = 0
#: pages handed to the encoder per call (the encoder halves its own chunk on CUDA out-of-memory)
ENCODE_CHUNK = 16
#: wiring check pages, from different files, and pages scanned per file to find one
WIRING_PAGES = 12
WIRING_SCAN = 3
#: the real-query gate needs this many queries (FINAL_DESIGN 5.5.2)
MIN_QUERIES = 60
#: sample queries timed after the build
TIMING_QUERIES = 20
#: the practice and real-query gates rank this deep
RANK_DEPTH = 10
#: an estimate above this (seconds) on Colab with a local work_dir prints the disconnect warning
COLAB_WARN_SECONDS = 3600.0
#: the default target of a new index
DEFAULT_TARGET = 0.97
#: a lock untouched this long, held on another machine (an old Colab session), counts as stale
STALE_LOCK_SECONDS = 6 * 3600.0
#: practice questions kept as ``Index.examples``
MAX_EXAMPLES = 50
#: the store compacts above this many segments (CONTRACT ``_store.compact``)
COMPACT_ABOVE = 8

MANIFEST_SCHEMA = 1
MANIFEST_KIND = "optivision-easy-index"

#: what this module writes into the store manifest (``store.commit(**...)``)
MANIFEST_KEYS = {
    "schema": "manifest schema of the easy index (1)",
    "kind": "'optivision-easy-index'",
    "optivision": "package version that wrote it",
    "model": "registry id, or None (vectors without a model, or your own adapter)",
    "adapter": "True when built with an EncoderAdapter object (string queries then need it again)",
    "modality": "'image' or 'text', as the rule used it",
    "source_kind": "'files' (folders, PDFs, images), 'text' (strings) or 'vectors'",
    "target": "the target the layout was chosen for",
    "layout": "the stored LayoutRow as a dict",
    "selection": "the Selection as a dict (rows, pick, sequence, basis, close call)",
    "checked": "whether the practice check could run (60 or more hard practice questions)",
    "unchecked_reason": "why not, in words",
    "sources": "{namespace: absolute root folder}",
    "files": "{doc_id: {path, relpath, ns, kind, size, mtime, sha256, pages, stored, f16_bytes, partial}}",
    "aliases": "{page id: the identical page id that is stored instead}",
    "blank": "[page ids of blank pages, not stored]",
    "sha_index": "{sha256 of a stored page render, text or vectors: its page id}",
    "skipped": "[[path, reason]] of files that could not be read",
    "n_pages": "pages stored (aliases and blank pages not included)",
    "dim": "vector width",
    "float16_bytes": "bytes of the float16 vectors of the stored pages",
    "keep_text": "page text kept (zlib) for Hit.text",
    "keep_float16": "float16 copy kept in the index",
    "record": "the encoder's build record (model, revision, dtype, device, versions, ...)",
    "trust_remote_code": "whether the build allowed the model's own code",
    "practice": "the practice check: ran, mode, n_probes, n_hard, sha256, steps, tested, chosen, fired",
    "queries_gate": "the real-query gate: n, ran, kind, steps, tested, chosen, fired",
    "search_seconds": "measured seconds per query, on 20 sample queries",
    "examples": "practice questions the check did not use, as examples",
    "summary": "the printed summary block (Block.as_dict())",
    "evidence": "every claim of the summary (Claim.as_dict())",
    "upgrade_attempts": "upgrade() attempts so far (error 0.05 * 2**-k)",
    "work_dir": "the shard folder the build used",
    "built": "UTC time of the build",
    "updated": "UTC time of the last sync",
}


# ============================================================================ adapters
#
# One function per call into a module written by another engineer, with the signature assumed
# from docs/easy/CONTRACT.md in its docstring.


def _mod(name: str) -> Any:
    """An easy-mode module by name, imported on use (most of them pull in heavy libraries)."""
    return importlib.import_module(f"optivision.easy.{name}")


def _resolve_model(model: str | None) -> Any:
    """``_resolve.resolve(model: str | None) -> ModelInfo | None``."""
    return _mod("_resolve").resolve(model)


def _select(model_id: str | None, target: float, n_pages: int | None, *, checked: bool, modality: str | None,
            vectors_per_page: float | None, layout: str | None) -> Selection:
    """``_levels.select(model, target, n_pages=None, *, checked, modality, vectors_per_page, layout) -> Selection``."""
    return _mod("_levels").select(model_id, target, n_pages, checked=checked, modality=modality,
                                  vectors_per_page=vectors_per_page, layout=layout)


def _list_source(source: Path) -> tuple[list[Any], list[tuple[str, str]]]:
    """``_docs.list_source(source) -> (files: list[FileRecord], skipped: [(path, reason)])``."""
    files, skipped = _mod("_docs").list_source(source)
    return list(files), [(str(p), str(r)) for p, r in skipped]


def _count_pages(source: Path) -> int:
    """``_docs.count_pages(source, progress=None) -> PageCount(n_pages, n_files, skipped)``."""
    return int(_mod("_docs").count_pages(source).n_pages)


def _iter_pages(record: Any) -> Iterator[tuple[Any, Any]]:
    """``_docs.iter_pages(file: FileRecord, *, dpi=150, max_side=1536) -> Iterator[(PageRecord, image)]``."""
    return iter(_mod("_docs").iter_pages(record))


def _profile(pages: Sequence[Any], vectors_per_page: Sequence[int] | None) -> Any:
    """``_docs.profile(pages, vectors_per_page=None) -> FolderProfile(..., checkable, reason)``."""
    return _mod("_docs").profile(pages, vectors_per_page)


def _make_probes(kind: str, pages: Sequence[Any], *, allow_upload: bool) -> list[Any]:
    """Tier A ``_probes.make_probes(pages, *, seed=0, max_pages=600, per_page=3) -> list[Probe]``; the
    opt-in tiers assumed as ``_probes_vlm.make_probes(pages, seed=0)`` and
    ``_probes_api.make_probes(pages, seed=0, allow_upload=...)``."""
    if kind == "text":
        return list(_mod("_probes").make_probes(pages, seed=0, max_pages=600, per_page=3))
    if kind == "vlm":
        return list(_mod("_probes_vlm").make_probes(pages, seed=0))
    if kind == "api":
        return list(_mod("_probes_api").make_probes(pages, seed=0, allow_upload=allow_upload))
    raise ValueError(f"probes must be 'text', 'vlm', 'api' or None, got {kind!r}")


def _hard_set(probes: Sequence[Any], ref: dict[str, list[tuple[str, float]]]) -> Any:
    """``_probes.hard_set(probes, ref_rankings, *, k_keep=10) -> HardSet``."""
    return _mod("_probes").hard_set(probes, ref, k_keep=RANK_DEPTH)


def _hard_probes(hard: Any) -> list[Any]:
    """The probes of a ``HardSet``, assumed as attribute ``probes``."""
    return list(hard.probes)


def _hard_sha(hard: Any) -> str:
    """The frozen sha256 of a ``HardSet``, assumed as attribute ``sha256``."""
    return str(getattr(hard, "sha256", ""))


def _fixed_sequence(sequence: Sequence[LayoutRow], rank_fn: Callable[[LayoutRow], dict[str, list[str]]],
                    ref: dict[str, list[str]], relevant: dict[str, frozenset[str]], clusters: dict[str, str],
                    target: float) -> Any:
    """``_veto.fixed_sequence(sequence, rank_fn, ref, relevant, clusters, target)
    -> SequenceResult(chosen: LayoutRow, steps: [(row, CheckStat(R, U, n, passed))])``."""
    return _mod("_veto").fixed_sequence(list(sequence), rank_fn, ref, relevant, clusters, target)


def _strength(n: int, target: float) -> dict[str, float] | None:
    """The strength line (``presets/veto_strength.json``), assumed as ``_veto.strength(n, target)
    -> {"fall": 0.90, "caught": 0.74} | None``; the line is left out when there is none."""
    fn = getattr(_mod("_veto"), "strength", None)
    out = None if fn is None else fn(n, target)
    return dict(out) if out else None


def _store_create(path: Path, pipeline: Any, info: dict[str, Any]) -> Any:
    """``_store.Store.create(path, *, pipeline: fitted Pipeline, info: dict) -> Store``."""
    return _mod("_store").Store.create(path, pipeline=pipeline, info=info)


def _store_open(path: Path, mode: str = "r") -> Any:
    """``_store.Store.open(path, mode="r" | "a")``."""
    return _mod("_store").Store.open(path, mode=mode)


def _store_append(store: Any, compressed: Any, texts: dict[str, str] | None, float16: Any) -> None:
    """``store.append(compressed: CompressedCorpus, texts: dict | None, float16: MultiVectorCorpus | None)``."""
    store.append(compressed, texts, float16)


def _store_set_reference(store: Any, reference: dict[str, Any]) -> None:
    """``store.reference = {...}`` (probe texts, query vectors, relevant sets, float16 reference scores)."""
    store.reference = reference


def _store_lock(store: Any) -> Any:
    """``store.lock()``: the store's own writer lock, a context manager (``IndexLocked`` when held)."""
    return store.lock()


def _searcher(store: Any) -> Any:
    """``_index.Searcher(store)`` with ``.search(queries: MultiVectorCorpus, k) -> list[list[(id, score)]]``."""
    return _mod("_index").Searcher(store)


def _rank_float(corpus: Any, queries: Any, k: int, pipeline: Any = None) -> list[list[tuple[str, float]]]:
    """``_index.rank_float(corpus, queries, k, pipeline=None) -> list[list[(id, score)]]``: exact top-k of
    a float16 corpus, compressed first with ``pipeline`` (a fitted candidate) when one is given."""
    return _mod("_index").rank_float(corpus, queries, k, pipeline=pipeline)


def _load_encoder(model: Any, *, device: str, dtype: str, trust_remote_code: bool) -> Any:
    """``_encode.load_encoder(model, *, device="auto", dtype="recorded", trust_remote_code=False) -> Encoder``
    with ``.adapter``, ``.info``, ``.record``, ``.encode_pages(images, ids, progress=None)`` and
    ``.encode_queries(texts)``."""
    return _mod("_encode").load_encoder(model, device=device, dtype=dtype, trust_remote_code=trust_remote_code)


def _encode_pages(encoder: Any, payloads: list[Any], ids: list[str]) -> Any:
    """``encoder.encode_pages(images, ids, progress=None) -> MultiVectorCorpus`` (strings for text models)."""
    return encoder.encode_pages(payloads, ids)


def _encode_queries(encoder: Any, texts: Sequence[str]) -> Any:
    """``encoder.encode_queries(texts) -> MultiVectorCorpus``."""
    return encoder.encode_queries(list(texts))


def _wiring(encoder: Any, pages: list[tuple[Any, Any]]) -> Any:
    """``_encode.wiring_check(encoder, pages) -> Wiring(ok, hits, n, used_sample, message)``; it raises
    ``WiringError`` itself when the user's pages and the sample PDF both fail."""
    return _mod("_encode").wiring_check(encoder, pages)


def _estimate(model_id: str | None, n_pages: int) -> Any:
    """``_encode.estimate(model_id, n_pages) -> Estimate(lo_s, hi_s, basis, gpu)``; None without a model id."""
    if not model_id or n_pages <= 0:
        return None
    try:
        return _mod("_encode").estimate(model_id, n_pages)
    except (KeyError, LookupError, ValueError):
        return None


def _recorded_dtype(model_id: str | None) -> str | None:
    """``_encode.recorded_dtype(model_id) -> str | None``."""
    if not model_id:
        return None
    fn = getattr(_mod("_encode"), "recorded_dtype", None)
    return None if fn is None else fn(model_id)


def _budget_target(model_id: str | None, n_pages: int, max_disk: int, *, checked: bool, modality: str | None,
                   vectors_per_page: float | None) -> float | None:
    """The checks agent's ``max_disk=`` helper, assumed as ``_budget.highest_target_that_fits(model, n_pages,
    max_disk_bytes, *, checked, modality, vectors_per_page) -> float | None`` (None: nothing fits)."""
    return _mod("_budget").highest_target_that_fits(model_id, n_pages, max_disk, checked=checked, modality=modality,
                                                    vectors_per_page=vectors_per_page)


def run_check(index: Any, queries: Any, *, answers: Any, drift: str) -> Any:
    """``_check.run_check(index, queries=None, *, answers=None, drift="auto") -> CheckReport``."""
    return _mod("_check").run_check(index, queries, answers=answers, drift=drift)


def run_upgrade(index: Any, queries: Any, *, answers: Any, apply: bool) -> Any:
    """``_upgrade.run_upgrade(index, queries, *, answers=None, apply=False) -> UpgradeReport``."""
    return _mod("_upgrade").run_upgrade(index, queries, answers=answers, apply=apply)


def make_report(index: Any, sections: Callable[[], list[tuple[str, Block]]], meta: Callable[[], dict[str, Any]]) -> Any:
    """``_report.Report.from_index(index)`` (CONTRACT), else ``_report.Report(sections, meta)`` built here."""
    report_cls = _mod("_report").Report
    from_index = getattr(report_cls, "from_index", None)
    if from_index is not None:
        return from_index(index)
    return report_cls(sections(), meta())


def open_query_log(index: Any) -> Any:
    """The opt-in query log, assumed as ``_querylog.open_log(index) -> log`` with
    ``log.record(query: str, hits: list[Hit])`` (local files only)."""
    return _mod("_querylog").open_log(index)


def launch_demo(index: Any) -> Any:
    """``integrations.demo.launch(index)`` (extra ``easy-ui``)."""
    return importlib.import_module("optivision.easy.integrations.demo").launch(index)


def export_qdrant(index: Any, **kwargs: Any) -> Any:
    """``integrations.qdrant.export(index, ...)``; gated by ``qdrant_export``."""
    return importlib.import_module("optivision.easy.integrations.qdrant").export(index, **kwargs)


# ============================================================================ values


@dataclass
class PageMeta:
    """One page as the build keeps it: the fields of ``_docs.PageRecord`` plus the build's own.

    ``id`` is ``PageRef(doc_id, page).page_id`` for file sources and the item id for strings and
    vectors. ``alias_of`` names the identical page stored instead; ``file_key`` is the
    ``size:mtime`` of the file version the page came from; ``n_vectors`` is set once encoded.
    It is duck-compatible with ``PageRecord``, so ``_docs.profile`` and ``_probes.make_probes``
    take it.
    """

    id: str
    doc_id: str
    file: str | None = None
    relpath: str | None = None
    page: int | None = None
    text: str = ""
    n_chars: int = 0
    blank: bool = False
    sha256: str = ""
    landscape: bool = False
    alias_of: str | None = None
    file_key: str = ""
    n_vectors: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PageMeta:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def stored(self) -> bool:
        return bool(self.n_vectors) and not self.blank and self.alias_of is None


@dataclass
class Unit:
    """One thing to sync: a file (all its pages), a string, or a vector document."""

    doc_id: str
    kind: str  # "pdf" | "image" | "text" | "vectors"
    path: str | None = None
    relpath: str | None = None
    ns: str = ""
    size: int = 0
    mtime: int = 0
    record: Any = None  # the _docs FileRecord
    payload: Any = None  # the string, or the [n, d] array
    sha: str | None = None
    max_pages: int | None = None  # sample=: encode only the leading pages

    @property
    def key(self) -> str:
        return f"{self.size}:{self.mtime}"

    def sha256(self) -> str:
        if self.sha is None:
            if self.kind == "text":
                self.sha = hashlib.sha256(str(self.payload).encode("utf-8")).hexdigest()
            elif self.kind == "vectors":
                self.sha = _array_sha(self.payload)
            else:
                self.sha = file_sha256(Path(self.path or ""))
        return self.sha

    def entry(self) -> dict[str, Any]:
        return {"path": self.path, "relpath": self.relpath, "ns": self.ns, "kind": self.kind, "size": self.size,
                "mtime": self.mtime, "sha256": self.sha}


@dataclass
class Source:
    """What ``build`` was given, classified."""

    kind: str  # "files" | "text" | "vectors"
    root: Path | None = None
    single: Path | None = None
    units: list[Unit] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notices: list[Line] = field(default_factory=list)
    dropped_rows: int = 0
    dim: int | None = None


@dataclass
class BuildOptions:
    """The keyword arguments of ``build`` (and ``add``) that steer the pipeline."""

    model: Any = None
    target: float | None = None
    path: Path | None = None
    ids: Sequence[str] | None = None
    layout: str | None = None
    keep_float16: bool = False
    keep_text: bool = True
    work_dir: str | Path | None = None
    prune: bool = False
    rebuild: bool = False
    trust_remote_code: bool = False
    device: str = "auto"
    dtype: str = "recorded"
    progress: bool = True
    sample: int | None = None
    probes: str | None = "text"
    queries: Any = None
    answers: Any = None
    max_disk: Any = None
    allow_upload: bool = False
    #: an encoder already loaded (``Index.add`` passes its own, so the model loads once)
    encoder: Any = None


@dataclass
class BuildResult:
    """What a build hands to ``_api.Index``."""

    path: Path
    encoder: Any = None
    adapter: Any = None
    summary: Block | None = None
    sync: SyncReport | None = None
    blocks: list[Block] = field(default_factory=list)


class _Out:
    """Prints blocks when ``progress`` is on; keeps every block either way."""

    def __init__(self, on: bool) -> None:
        self.on = bool(on)
        self.blocks: list[Block] = []

    def block(self, block: Block) -> None:
        self.blocks.append(block)
        if self.on:
            _print(str(block))

    def line(self, line: Line) -> None:
        if self.on:
            _print(line.render(11))


def _print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:  # a cp1252 console and a non-ASCII file name
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)


def render(*lines: Line) -> str:
    """Lines as one aligned text, for exception messages."""
    width = max([len(ln.label) for ln in lines if ln.label] + [9]) + 2
    return "\n".join(ln.render(width) for ln in lines)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def target_text(target: float) -> str:
    """``0.97 -> "97%"``, ``0.975 -> "97.5%"``: a target is the user's own number, shown as given."""
    s = f"{100 * float(target):.2f}".rstrip("0").rstrip(".")
    return f"{s}%"


def seconds_text(s: float) -> str:
    if s < 0.01:
        return f"{1000 * s:.1f} ms"
    if s < 0.1:
        return f"{1000 * s:.0f} ms"
    return _claim.duration(s)


def check_target(target: float | None, accuracy: float | None, out: _Out) -> float | None:
    """The target in [lowest, 1.0] (``ValueError`` outside); ``accuracy=`` is an alias and says what it means."""
    if accuracy is not None:
        if target is not None and abs(float(target) - float(accuracy)) > 1e-12:
            raise ValueError("pass target= or accuracy=, not both with different values")
        target = float(accuracy)
        out.block(Block([Line("accuracy", "{0} read as {1}: keep at least {2} of full-precision search quality "
                                          "(nDCG@5); not {2} of answers correct.",
                              lit(f"accuracy={accuracy}"), lit(f"target={accuracy}"), lit(target_text(target)))],
                        next="the build goes on with this target"))
    if target is None:
        return None
    t = float(target)
    lowest = float(_ledger.setting("margins", "lowest_target", 0.9))
    if not (lowest - 1e-12 <= t <= 1.0 + 1e-12):
        raise ValueError(f"target must be in [{lowest}, 1.0], got {target!r}")
    return t


# ============================================================================ hashing, sizes, places


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """sha256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}:{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def parse_bytes(value: Any) -> int:
    """``max_disk=``: bytes as a number, or a size such as ``"500MB"`` or ``"2 GB"`` (decimal units)."""
    if isinstance(value, bool):
        raise TypeError("max_disk must be bytes or a size such as '2 GB'")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    text = str(value).strip().upper().replace(" ", "")
    for unit, scale in (("TB", 10**12), ("GB", 10**9), ("MB", 10**6), ("KB", 10**3), ("B", 1)):
        if text.endswith(unit):
            return int(float(text[: -len(unit)]) * scale)
    return int(float(text))


def _existing_ancestor(path: Path) -> Path:
    p = Path(path).resolve()
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(_existing_ancestor(path)).free)


def _same_device(a: Path, b: Path) -> bool:
    try:
        return os.stat(_existing_ancestor(a)).st_dev == os.stat(_existing_ancestor(b)).st_dev
    except OSError:
        return False


def on_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG")) \
        or bool(os.environ.get("COLAB_GPU"))


def work_root(path: Path, work_dir: str | Path | None, kind: str) -> Path | None:
    """Where the float16 shards go: ``<work_dir>/<index name>`` when given, ``/content/.ov_work/<index
    name>`` on Colab, else ``<path>.work`` beside the index. Vectors you give need none."""
    if kind == "vectors":
        return None
    if work_dir is not None:
        return Path(work_dir).expanduser().resolve() / path.name
    if on_colab():
        return Path("/content/.ov_work") / path.name
    return Path(str(path) + ".work")


# ============================================================================ sources


def _is_tensor(x: Any) -> bool:
    return hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "shape")


def _to_numpy(x: Any) -> np.ndarray:
    if _is_tensor(x):
        t = x.detach().cpu()
        if str(getattr(t, "dtype", "")) == "torch.bfloat16":
            t = t.float()
        return t.numpy()
    return np.asarray(x)


def _looks_vectors(source: Any) -> bool:
    from ..representation import MultiVectorCorpus

    if isinstance(source, (MultiVectorCorpus, np.ndarray)) or _is_tensor(source):
        return True
    if isinstance(source, (list, tuple)) and source:
        first = source[0]
        return _is_tensor(first) or isinstance(first, np.ndarray)
    return False


def vectors_source(source: Any, ids: Sequence[str] | None) -> Source:
    """Page vectors: a ``MultiVectorCorpus``, a list of ``(n_i, d)`` arrays or tensors, or a padded batch
    ``(n, L, d)``. Torch tensors are converted (said in a notice); all-zero rows are padding, dropped
    and counted."""
    from ..representation import MultiVectorCorpus

    notices: list[Line] = []
    given_ids: list[str] | None = None
    if isinstance(source, MultiVectorCorpus):
        arrays = [np.asarray(source.vectors[lo:hi]) for lo, hi in zip(source.offsets[:-1], source.offsets[1:])]
        given_ids = list(source.ids)
    else:
        torch_in = _is_tensor(source) or (isinstance(source, (list, tuple)) and any(_is_tensor(a) for a in source))
        if _is_tensor(source) or isinstance(source, np.ndarray):
            batch = _to_numpy(source)
            if batch.ndim == 2:
                batch = batch[None]
            if batch.ndim != 3:
                raise ValueError(f"vectors must be a list of (n_i, d) arrays or a padded (n, L, d) batch, "
                                 f"got shape {batch.shape}")
            arrays = [batch[i] for i in range(batch.shape[0])]
        else:
            arrays = [_to_numpy(a) for a in source]
        if torch_in:
            notices.append(Line("Vectors", "torch tensors were converted to NumPy arrays on the CPU "
                                           "(bfloat16 widened to float32)"))
    if not arrays:
        raise ValueError("no vectors given")
    if any(a.ndim != 2 for a in arrays) or len({a.shape[1] for a in arrays}) != 1:
        raise ValueError("every document must be a 2-D (n_i, d) array of one shared width d")
    if not all(np.issubdtype(a.dtype, np.floating) for a in arrays):
        raise TypeError("vectors must be floating point")
    dim = int(arrays[0].shape[1])
    dropped = 0
    clean: list[np.ndarray] = []
    for a in arrays:
        keep = np.any(a != 0, axis=1)
        dropped += int(a.shape[0] - int(keep.sum()))
        clean.append(np.ascontiguousarray(a[keep]))
    if dropped:
        notices.append(Line("Padding", "{0} all-zero rows dropped as padding", claim(dropped, "counted")))
    n = len(clean)
    id_list = [str(i) for i in ids] if ids is not None else (given_ids or [str(i) for i in range(n)])
    if len(id_list) != n:
        raise ValueError(f"{len(id_list)} ids for {n} documents")
    if len(set(id_list)) != n:
        raise ValueError("ids must be unique")
    units = [Unit(doc_id=i, kind="vectors", payload=a, size=int(a.nbytes)) for i, a in zip(id_list, clean)]
    return Source(kind="vectors", units=units, notices=notices, dropped_rows=dropped, dim=dim)


def text_source(texts: Sequence[str], ids: Sequence[str] | None) -> Source:
    """Strings, one document each, for text models."""
    id_list = [str(i) for i in ids] if ids is not None else [str(i) for i in range(len(texts))]
    if len(id_list) != len(texts):
        raise ValueError(f"{len(id_list)} ids for {len(texts)} texts")
    if len(set(id_list)) != len(id_list):
        raise ValueError("ids must be unique")
    units = [Unit(doc_id=i, kind="text", payload=str(t), size=len(str(t).encode("utf-8")))
             for i, t in zip(id_list, texts)]
    return Source(kind="text", units=units)


def classify(source: Any, ids: Sequence[str] | None) -> Source:
    """Sort ``source`` into files (folder, PDF, image), strings or vectors."""
    if isinstance(source, (str, os.PathLike)):
        p = Path(source).expanduser()
        if not p.exists():
            raise SourceMissing(f"{p} does not exist (give a folder, a PDF or an image; strings go in a list)")
        if ids is not None:
            raise ValueError("ids= is for strings and vectors; files are named by their path")
        p = p.resolve()
        return Source(kind="files", root=p if p.is_dir() else p.parent, single=None if p.is_dir() else p)
    if _looks_vectors(source):
        return vectors_source(source, ids)
    if isinstance(source, (list, tuple)) and source and all(isinstance(s, str) for s in source):
        return text_source(source, ids)
    raise TypeError(f"cannot build from {type(source).__name__}: give a folder, a PDF, an image, a list of strings, "
                    "a list of (n_i, d) arrays or a MultiVectorCorpus")


def default_path(src: Source) -> Path:
    """``<folder>.ovx`` beside the folder; ``<name>.ovx`` beside a single file; ``vectors.ovx`` or
    ``texts.ovx`` in the working folder otherwise."""
    if src.kind == "files":
        if src.single is not None:
            return src.single.parent / f"{src.single.stem}.ovx"
        assert src.root is not None
        return src.root.parent / f"{src.root.name}.ovx"
    return Path.cwd() / ("vectors.ovx" if src.kind == "vectors" else "texts.ovx")


def namespace_for(root: Path, sources: Mapping[str, str]) -> str:
    """The id namespace of a source folder: its name, or ``name~2``, ``name~3``... when another folder
    of the same name is already in the index (CONTRACT, Ids and sources)."""
    root_s = str(root)
    for ns, r in sources.items():
        if r == root_s:
            return ns
    base = root.name or "source"
    if base not in sources:
        return base
    k = 2
    while f"{base}~{k}" in sources:
        k += 1
    return f"{base}~{k}"


def _with_doc_id(record: Any, doc_id: str) -> Any:
    """A copy of a ``_docs.FileRecord`` carrying our namespaced ``doc_id``."""
    import dataclasses

    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        return dataclasses.replace(record, doc_id=doc_id)
    if hasattr(record, "_replace"):
        return record._replace(doc_id=doc_id)
    with contextlib.suppress(AttributeError, TypeError):
        record.doc_id = doc_id
    return record


def file_units(src: Source, sources: Mapping[str, str]) -> tuple[list[Unit], str]:
    """List a file source (``_docs.list_source``) into units with namespaced doc ids, sorted."""
    assert src.root is not None
    ns = namespace_for(src.root, sources)
    files, skipped = _list_source(src.single or src.root)
    src.skipped.extend(skipped)
    units: list[Unit] = []
    for rec in files:
        p = Path(str(rec.path)).resolve()
        try:
            rel = p.relative_to(src.root).as_posix()
        except ValueError:
            rel = str(getattr(rec, "relpath", p.name)).replace("\\", "/")
        doc_id = f"{ns}/{rel}"
        try:
            st = p.stat()
        except OSError as exc:
            src.skipped.append((str(p), f"cannot be read: {exc.strerror or exc}"))
            continue
        kind = str(getattr(rec, "kind", None) or ("pdf" if p.suffix.lower() == ".pdf" else "image"))
        units.append(Unit(doc_id=doc_id, kind=kind, path=str(p), relpath=rel, ns=ns, size=int(st.st_size),
                          mtime=int(st.st_mtime_ns), record=_with_doc_id(rec, doc_id)))
    units.sort(key=lambda u: u.doc_id)
    return units, ns


# ============================================================================ sync diff


@dataclass
class Diff:
    """Units to encode (in listing order), unchanged units, and the doc ids that disappeared."""

    new: list[Unit] = field(default_factory=list)
    changed: list[Unit] = field(default_factory=list)
    unchanged: list[Unit] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    touched: list[Unit] = field(default_factory=list)  # new (size, mtime), same sha256: only the key is stored

    @property
    def to_encode(self) -> list[Unit]:
        pos = {d: i for i, d in enumerate(self.order)}
        return sorted([*self.new, *self.changed], key=lambda u: pos.get(u.doc_id, len(pos)))


def diff_units(units: list[Unit], known: Mapping[str, Mapping[str, Any]], ns: str | None,
               full_listing: bool) -> Diff:
    """Compare listed units with the manifest's ``files``. A file with the same ``(size, mtime)`` is
    unchanged; a changed key is checked by sha256 before it counts as changed. Strings and vectors are
    compared by sha256. Missing: known units of the same namespace not listed (folders only)."""
    d = Diff(order=[u.doc_id for u in units])
    seen = set()
    for u in units:
        seen.add(u.doc_id)
        old = known.get(u.doc_id)
        if old is None:
            d.new.append(u)
        elif u.kind in ("text", "vectors"):
            (d.unchanged if old.get("sha256") == u.sha256() else d.changed).append(u)
        elif int(old.get("size", -1)) == u.size and int(old.get("mtime", -1)) == u.mtime:
            u.sha = old.get("sha256") or u.sha
            d.unchanged.append(u)
        elif old.get("sha256") and u.sha256() == old.get("sha256"):
            d.unchanged.append(u)
            d.touched.append(u)
        else:
            d.changed.append(u)
    if full_listing:
        d.missing = [doc_id for doc_id, entry in known.items()
                     if doc_id not in seen and (ns is None or entry.get("ns", "") == ns)]
    return d


# ============================================================================ the writer lock


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class BuildLock:
    """The writer lock of an index path: ``<path>.lock``, created with ``O_CREAT | O_EXCL``.

    It covers a build from before its store exists; the store's own ``lock()`` is held as well
    while the store is written. A lock left by a process that is gone (same machine), or untouched
    for six hours on another machine (an old Colab session), is taken over; a running build touches
    it at every shard.
    """

    def __init__(self, path: Path) -> None:
        self.file = Path(str(path) + ".lock")
        self.held = False

    def _stale(self) -> bool:
        try:
            info = json.loads(self.file.read_text(encoding="utf-8"))
            age = time.time() - self.file.stat().st_mtime
        except (OSError, ValueError):
            return True
        if info.get("host") == socket.gethostname():
            return int(info.get("pid", -1)) != os.getpid() and not _pid_alive(int(info.get("pid", -1)))
        return age > STALE_LOCK_SECONDS

    def acquire(self) -> BuildLock:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                fd = os.open(str(self.file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if attempt == 0 and self._stale():
                    with contextlib.suppress(OSError):
                        self.file.unlink()
                    continue
                raise IndexLocked(render(
                    Line("Locked", "{0} is being written by another build or add()", lit(self.file.with_suffix(""))),
                    Line("", "if none is running, delete {0}", lit(self.file)))) from None
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started": _now()}, f)
            self.held = True
            return self
        raise IndexLocked(f"{self.file} could not be taken")  # pragma: no cover

    def touch(self) -> None:
        if self.held:
            with contextlib.suppress(OSError):
                os.utime(self.file, None)

    def release(self) -> None:
        if self.held:
            with contextlib.suppress(OSError):
                self.file.unlink()
            self.held = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


# ============================================================================ shards


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True)
        f.flush()
        with contextlib.suppress(OSError):
            os.fsync(f.fileno())
    os.replace(tmp, path)


def rmtree_retry(path: Path, tries: int = 5) -> bool:
    """Delete a folder. On Windows a memory map closes only when it is collected, so collect and retry."""
    for i in range(tries):
        if not path.exists():
            return True
        gc.collect()
        try:
            shutil.rmtree(path)
            return True
        except OSError:
            time.sleep(0.05 * (i + 1))
    return not path.exists()


class ShardSet:
    """float16 shards of encoded pages (FINAL_DESIGN 5.7 step 2).

    On disk (``root`` given): ``shards.json`` (shard names and per-file state, replaced atomically)
    and one folder per shard, ``s00000/{vectors.npy, offsets.npy, pages.json}``. A shard is written
    under a temporary name, renamed, and only then listed in ``shards.json``, so a killed build leaves
    the old state or the new one, never half a shard. ``pages.json`` holds the :class:`PageMeta` of
    every page the shard covers, blank and identical ones included, so a resumed build neither encodes
    them again nor loses their text. A file counts as complete once its last page is in a listed shard.

    In memory (``root=None``) for vectors the caller already holds.

    ``identity`` (model, dtype, revision, source kind) is compared on open: shards made by another
    model are discarded.
    """

    STATE = "shards.json"

    def __init__(self, root: Path | None, identity: dict[str, Any]) -> None:
        self.root = root
        self.identity = dict(identity)
        self.meta: dict[str, PageMeta] = {}  # page id -> its latest valid meta
        self.where: dict[str, str] = {}  # stored page id -> shard name
        self.files: dict[str, dict[str, Any]] = {}  # doc id -> {size, mtime, sha256, complete}
        self.names: list[str] = []
        self._ids: dict[str, list[str]] = {}  # shard name -> its encoded page ids, in order
        self._memory: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pend_meta: list[PageMeta] = []
        self._pend_vec: list[np.ndarray | None] = []
        self._pend_done: list[str] = []
        self.on_flush: Callable[[], None] | None = None
        if root is not None:
            self._open()

    # ------------------------------------------------------------ state

    def _open(self) -> None:
        assert self.root is not None
        state: dict[str, Any] = {}
        state_file = self.root / self.STATE
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state = {}
        if state and state.get("identity") != self.identity:
            rmtree_retry(self.root)  # shards of another model or dtype are of no use
            state = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = {k: dict(v) for k, v in state.get("files", {}).items()}
        for name in state.get("shards", []):
            try:
                pages = json.loads((self.root / name / "pages.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # a shard folder that vanished: its pages are encoded again
            self._index(name, [PageMeta.from_dict(d) for d in pages])

    def _index(self, name: str, metas: list[PageMeta]) -> None:
        self.names.append(name)
        self._ids[name] = [m.id for m in metas if m.stored]
        for m in metas:
            self.meta[m.id] = m
            if m.stored:
                self.where[m.id] = name
            else:
                self.where.pop(m.id, None)

    def _save_state(self) -> None:
        if self.root is not None:
            _write_json_atomic(self.root / self.STATE, {"schema": 1, "identity": self.identity,
                                                        "shards": self.names, "files": self.files})

    # ------------------------------------------------------------ per file

    def begin(self, unit: Unit) -> tuple[set[str], bool]:
        """Pages of ``unit`` already in shards for this version of it, and whether it is complete."""
        rec = self.files.get(unit.doc_id)
        same = rec is not None and int(rec.get("size", -1)) == unit.size and int(rec.get("mtime", -1)) == unit.mtime
        if rec is not None and not same and rec.get("sha256") and unit.sha256() == rec["sha256"]:
            same = True
            rec.update(size=unit.size, mtime=unit.mtime)
        if not same:
            self.files[unit.doc_id] = {"size": unit.size, "mtime": unit.mtime, "sha256": unit.sha, "complete": False}
            for pid in [p for p, m in self.meta.items() if m.doc_id == unit.doc_id]:
                self.meta.pop(pid, None)
                self.where.pop(pid, None)
            return set(), False
        assert rec is not None
        return {m.id for m in self.meta.values() if m.doc_id == unit.doc_id}, bool(rec.get("complete"))

    def add(self, meta: PageMeta, vectors: np.ndarray | None) -> None:
        meta.n_vectors = 0 if vectors is None else int(vectors.shape[0])
        self._pend_meta.append(meta)
        self._pend_vec.append(None if vectors is None else np.asarray(vectors, dtype=np.float16))

    def end(self, unit: Unit, sha: str | None = None) -> None:
        """Every page of ``unit`` is added; it becomes complete when those pages are flushed."""
        rec = self.files.setdefault(unit.doc_id, {"size": unit.size, "mtime": unit.mtime})
        if sha:
            rec["sha256"] = sha
        self._pend_done.append(unit.doc_id)

    def drop(self, unit: Unit) -> None:
        """Forget the unflushed pages of a file that failed half way; it stays incomplete."""
        keep = [i for i, m in enumerate(self._pend_meta) if m.doc_id != unit.doc_id]
        self._pend_meta = [self._pend_meta[i] for i in keep]
        self._pend_vec = [self._pend_vec[i] for i in keep]
        self._pend_done = [d for d in self._pend_done if d != unit.doc_id]
        if unit.doc_id in self.files:
            self.files[unit.doc_id]["complete"] = False

    def complete(self, doc_id: str) -> bool:
        return bool(self.files.get(doc_id, {}).get("complete"))

    def maybe_flush(self) -> None:
        if sum(1 for v in self._pend_vec if v is not None and v.shape[0]) >= SEGMENT_PAGES:
            self.flush()

    def flush(self) -> None:
        """Write the pending pages as one shard and list it; mark the files that ended complete."""
        if not self._pend_meta:
            if self._pend_done:
                for d in self._pend_done:
                    self.files.setdefault(d, {})["complete"] = True
                self._pend_done = []
                self._save_state()
            return
        k = len(self.names)
        while f"s{k:05d}" in self._ids:
            k += 1
        name = f"s{k:05d}"
        metas, vecs = self._pend_meta, self._pend_vec
        enc = [v for m, v in zip(metas, vecs) if m.stored and v is not None]
        if enc:
            vectors = np.concatenate(enc, axis=0)
            offsets = np.concatenate([[0], np.cumsum([v.shape[0] for v in enc])]).astype(np.int64)
        else:
            vectors, offsets = np.zeros((0, 0), dtype=np.float16), np.zeros(1, dtype=np.int64)
        if self.root is None:
            self._memory[name] = (vectors, offsets)
        else:
            final, tmp = self.root / name, self.root / f"{name}.tmp"
            for stale in (tmp, final):
                if stale.exists():
                    rmtree_retry(stale)
            tmp.mkdir(parents=True)
            np.save(tmp / "vectors.npy", vectors)
            np.save(tmp / "offsets.npy", offsets)
            (tmp / "pages.json").write_text(json.dumps([m.as_dict() for m in metas], ensure_ascii=True),
                                            encoding="utf-8")
            os.replace(tmp, final)
        self._index(name, metas)
        for d in self._pend_done:
            self.files.setdefault(d, {})["complete"] = True
        self._pend_meta, self._pend_vec, self._pend_done = [], [], []
        self._save_state()
        if self.on_flush is not None:
            self.on_flush()

    # ------------------------------------------------------------ reading

    def load(self, name: str, ids: Sequence[str] | None = None) -> Any:
        """The float16 corpus of shard ``name`` restricted to ``ids`` (in shard order)."""
        from ..representation import MultiVectorCorpus

        all_ids = self._ids[name]
        if self.root is None:
            vectors, offsets = self._memory[name]
        else:
            vectors = np.load(self.root / name / "vectors.npy", mmap_mode="r")
            offsets = np.load(self.root / name / "offsets.npy")
        want = set(all_ids if ids is None else ids)
        pick = [i for i, pid in enumerate(all_ids) if pid in want and self.where.get(pid) == name]
        dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
        parts = [np.array(vectors[offsets[i]:offsets[i + 1]], dtype=np.float16) for i in pick]
        del vectors  # closes the memory map once the slices above are copies
        vec = np.concatenate(parts, axis=0) if parts else np.zeros((0, dim), dtype=np.float16)
        off = np.concatenate([[0], np.cumsum([p.shape[0] for p in parts])]).astype(np.int64)
        return MultiVectorCorpus(vec, off, ids=[all_ids[i] for i in pick])

    def encoded(self, ids: set[str]) -> list[tuple[str, list[str]]]:
        """``[(shard name, [page ids])]`` holding the given stored pages, in shard order."""
        out = []
        for name in self.names:
            pids = [p for p in self._ids.get(name, []) if p in ids and self.where.get(p) == name]
            if pids:
                out.append((name, pids))
        return out

    def dim(self) -> int | None:
        for name in self.names:
            if self._ids.get(name):
                return int(self.load(name, self._ids[name][:1]).dimension)
        return None

    def delete(self) -> bool:
        self._memory.clear()
        return True if self.root is None else rmtree_retry(self.root)


# ============================================================================ encoding


class _FileError(Exception):
    def __init__(self, unit: Unit, exc: BaseException) -> None:
        super().__init__(str(exc))
        self.unit = unit
        self.exc = exc


def skip_reason(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    low = text.lower()
    if "password" in low or "encrypt" in low:
        return "encrypted"
    return f"unreadable ({text[:160]})"


def _meta_from_record(rec: Any, unit: Unit) -> PageMeta:
    from ..types import PageRef

    page = int(rec.page)
    text = str(getattr(rec, "text", "") or "")
    return PageMeta(id=PageRef(unit.doc_id, page).page_id, doc_id=unit.doc_id, file=unit.path, relpath=unit.relpath,
                    page=page, text=text, n_chars=int(getattr(rec, "n_chars", len(text)) or 0),
                    blank=bool(getattr(rec, "blank", False)), sha256=str(getattr(rec, "sha256", "") or ""),
                    landscape=bool(getattr(rec, "landscape", False)), file_key=unit.key)


def _pages_of(unit: Unit) -> Iterator[tuple[PageMeta, Any]]:
    """``(meta, payload)`` per page: a PIL image for files, the string itself for text. Errors that come
    from reading the file are raised as :class:`_FileError` (skip the file), never the encoder's."""
    if unit.kind == "text":
        text = str(unit.payload)
        yield PageMeta(id=unit.doc_id, doc_id=unit.doc_id, text=text, n_chars=len(text), blank=not text.strip(),
                       sha256=unit.sha256(), file_key=unit.key), text
        return
    try:
        it = _iter_pages(unit.record)
    except Exception as exc:
        raise _FileError(unit, exc) from exc
    try:
        while True:
            try:
                rec, image = next(it)
            except StopIteration:
                return
            except Exception as exc:
                raise _FileError(unit, exc) from exc
            yield _meta_from_record(rec, unit), image
    finally:
        close = getattr(it, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()  # a half-read PDF is released at once (Windows keeps open files locked)


class Encoding:
    """The encode loop over units into a :class:`ShardSet`: blank pages counted, identical pages
    aliased, unreadable files skipped, encoder calls batched across files."""

    def __init__(self, shards: ShardSet, encoder: Any, out: _Out, sha_seen: Mapping[str, str], total: int) -> None:
        self.shards = shards
        self.encoder = encoder
        self.out = out
        self.sha_seen = dict(sha_seen)
        self.batch: list[tuple[PageMeta, Any, Unit]] = []
        self.ended: list[tuple[Unit, str | None]] = []
        self.skipped: list[tuple[str, str]] = []
        self.total = max(int(total), 0)
        self.encoded = 0
        self._shown = 0
        self.dim: int | None = None

    def run(self, units: Sequence[Unit]) -> None:
        try:
            for unit in units:
                self._one(unit)
            self._encode_batch()
        finally:
            self.shards.flush()  # whatever was encoded survives an interruption
        if self.encoded and self.encoded != self._shown:
            self._progress()

    def _one(self, unit: Unit) -> None:
        done, complete = self.shards.begin(unit)
        if complete:
            return
        sha = None
        if unit.kind in ("pdf", "image"):
            try:
                sha = unit.sha256()
            except OSError as exc:
                self._skip(unit, exc)
                return
        seen = 0
        try:
            for meta, payload in _pages_of(unit):
                seen += 1
                if unit.max_pages is not None and seen > unit.max_pages:
                    return  # sample=: the rest of this file waits for the full build (not complete)
                if meta.id in done:
                    continue
                if meta.blank:
                    self.shards.add(meta, None)
                    continue
                canon = self.sha_seen.get(meta.sha256) if meta.sha256 else None
                if canon is not None and canon != meta.id:
                    meta.alias_of = canon
                    self.shards.add(meta, None)
                    continue
                if meta.sha256:
                    self.sha_seen[meta.sha256] = meta.id
                self.batch.append((meta, payload, unit))
                if len(self.batch) >= ENCODE_CHUNK:
                    self._encode_batch()
        except _FileError as err:
            self._skip(unit, err.exc)
            return
        if any(b[2] is unit for b in self.batch):
            self.ended.append((unit, sha))
        else:
            self.shards.end(unit, sha)

    def _skip(self, unit: Unit, exc: BaseException) -> None:
        self.batch = [b for b in self.batch if b[2] is not unit]
        self.ended = [e for e in self.ended if e[0] is not unit]
        self.shards.drop(unit)
        prefix = unit.doc_id + "::"
        self.sha_seen = {s: p for s, p in self.sha_seen.items() if not p.startswith(prefix)}
        self.skipped.append((unit.path or unit.doc_id, skip_reason(exc)))

    def _encode_batch(self) -> None:
        if self.batch:
            metas = [b[0] for b in self.batch]
            corpus = _encode_pages(self.encoder, [b[1] for b in self.batch], [m.id for m in metas])
            if len(corpus) != len(metas):
                raise RuntimeError(f"the encoder returned {len(corpus)} documents for {len(metas)} pages")
            self.dim = int(corpus.dimension)
            for i, meta in enumerate(metas):
                lo, hi = int(corpus.offsets[i]), int(corpus.offsets[i + 1])
                vec = np.asarray(corpus.vectors[lo:hi], dtype=np.float16)
                if not np.isfinite(vec).all():
                    raise ValueError(f"{meta.id}: the vectors are not finite in float16 (values above 65504?)")
                if vec.shape[0] == 0:
                    meta.blank = True  # an encoder that returns no vectors: nothing to search, counted as blank
                    self.shards.add(meta, None)
                else:
                    self.shards.add(meta, vec)
            self.encoded += len(metas)
            self.batch = []
            if self.encoded // SEGMENT_PAGES > self._shown // SEGMENT_PAGES:
                self._progress()
        for unit, sha in self.ended:
            self.shards.end(unit, sha)
        self.ended = []
        self.shards.maybe_flush()

    def _progress(self) -> None:
        self._shown = self.encoded
        self.out.line(Line("Encoding", "{0} of {1} pages", claim(self.encoded, "counted"),
                           claim(max(self.total, self.encoded), "counted")))


def load_vectors(units: Sequence[Unit], shards: ShardSet, sha_seen: Mapping[str, str]) -> None:
    """Vectors the caller gave go straight into (in-memory) shards; identical documents are aliases and
    documents without vectors count as blank."""
    seen = dict(sha_seen)
    for u in units:
        shards.begin(u)
        sha = u.sha256()
        meta = PageMeta(id=u.doc_id, doc_id=u.doc_id, sha256=sha, file_key=u.key, blank=u.payload.shape[0] == 0)
        if meta.blank:
            shards.add(meta, None)
        elif sha in seen and seen[sha] != u.doc_id:
            meta.alias_of = seen[sha]
            shards.add(meta, None)
        else:
            seen[sha] = u.doc_id
            shards.add(meta, u.payload)
        shards.end(u, sha)
        shards.maybe_flush()
    shards.flush()


def wiring_pages(units: Sequence[Unit]) -> list[tuple[Any, Any]]:
    """Up to 12 text pages from different files for the wiring check; with fewer files, more pages of
    the same files fill up to 12."""
    picked: list[tuple[Any, Any]] = []
    extra: list[tuple[Any, Any]] = []
    for unit in units:
        if len(picked) >= WIRING_PAGES:
            break
        if unit.kind not in ("pdf", "image"):
            continue
        it = None
        try:
            it = _iter_pages(unit.record)
            got = False
            for _ in range(WIRING_SCAN):
                rec, image = next(it)
                if getattr(rec, "blank", False) or int(getattr(rec, "n_chars", 0) or 0) < 20:
                    continue
                if not got:
                    picked.append((rec, image))
                    got = True
                elif len(extra) < WIRING_PAGES:
                    extra.append((rec, image))
        except Exception:  # noqa: BLE001, S110 - end of file, or unreadable: skipped later with its reason
            pass
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    close()
    return (picked + extra)[:WIRING_PAGES]


# ============================================================================ ranking over shards


def rank_shards(shards: ShardSet, ids: set[str], queries: Any, k: int,
                pipeline: Any = None) -> list[list[tuple[str, float]]]:
    """Exact top-``k`` per query over the given stored pages, shard by shard, merged. A page's score does
    not depend on the other pages, so the merge is the global top-k; ties go to the earlier page."""
    per_query: list[list[tuple[float, int, str]]] = [[] for _ in range(len(queries))]
    pos = 0
    for name, page_ids in shards.encoded(ids):
        corpus = shards.load(name, page_ids)
        order = {pid: pos + i for i, pid in enumerate(corpus.ids)}
        pos += len(corpus)
        res = _rank_float(corpus, queries, min(k, len(corpus)), pipeline)
        for q, hits in enumerate(res):
            per_query[q].extend((float(s), order[str(i)], str(i)) for i, s in hits)
        del corpus
    out = []
    for hits in per_query:
        hits.sort(key=lambda t: (-t[0], t[1]))
        out.append([(i, s) for s, _, i in hits[:k]])
    return out


def fit_sample(shards: ShardSet, ids: set[str]) -> Any:
    """A seeded page sample (seed 0) of at most :data:`FIT_ROWS` vectors, in shard order
    (FINAL_DESIGN 5.7: ``_fit_sample`` caps rows at 200,000)."""
    from ..representation import MultiVectorCorpus

    pages = [(name, pid) for name, pids in shards.encoded(ids) for pid in pids]
    order = np.random.default_rng(FIT_SEED).permutation(len(pages))
    take: list[int] = []
    rows = 0
    for i in order:
        n = shards.meta[pages[int(i)][1]].n_vectors
        if take and rows + n > FIT_ROWS:
            continue
        take.append(int(i))
        rows += n
        if rows >= FIT_ROWS:
            break
    by: dict[str, list[str]] = {}
    for i in sorted(take):
        by.setdefault(pages[i][0], []).append(pages[i][1])
    parts = [shards.load(name, pids) for name, pids in by.items()]
    vec = np.concatenate([p.vectors for p in parts], axis=0)
    off = np.concatenate([[0], np.cumsum(np.concatenate([p.counts for p in parts]))]).astype(np.int64)
    return MultiVectorCorpus(vec, off, ids=[i for p in parts for i in p.ids])


def timing_queries(shards: ShardSet, ids: set[str], probe_queries: Any = None) -> Any:
    """20 sample queries: practice questions when there are some, else the leading vectors of 20 seeded
    pages (the time depends on shapes, not on content)."""
    from ..representation import MultiVectorCorpus

    if probe_queries is not None and len(probe_queries):
        return probe_queries.select(range(min(TIMING_QUERIES, len(probe_queries))))
    pages = sorted(ids)
    if not pages:
        return None
    pick = np.random.default_rng(FIT_SEED).choice(len(pages), size=min(TIMING_QUERIES, len(pages)), replace=False)
    arrays = []
    for name, pids in shards.encoded({pages[int(i)] for i in pick}):
        c = shards.load(name, pids)
        for j in range(len(c)):
            lo, hi = int(c.offsets[j]), int(c.offsets[j + 1])
            arrays.append(np.asarray(c.vectors[lo:min(hi, lo + 16)], dtype=np.float32))
    arrays = [a for a in arrays if a.shape[0]]
    return MultiVectorCorpus.from_arrays(arrays, ids=[f"t{i}" for i in range(len(arrays))]) if arrays else None


# ============================================================================ selection helpers


def row_to_dict(row: LayoutRow | None) -> dict[str, Any] | None:
    return None if row is None else asdict(row)


def row_from_dict(d: Mapping[str, Any] | None) -> LayoutRow | None:
    if d is None:
        return None
    names = {f.name for f in fields(LayoutRow)}
    kw = {k: v for k, v in d.items() if k in names}
    if kw.get("bytes_per_page") is not None:
        kw["bytes_per_page"] = tuple(kw["bytes_per_page"])
    kw["datasets"] = tuple(kw.get("datasets") or ())
    return LayoutRow(**kw)


def selection_to_dict(sel: Selection | None) -> dict[str, Any] | None:
    if sel is None:
        return None
    return {"model": sel.model, "modality": sel.modality, "target": sel.target, "n_pages": sel.n_pages,
            "checked": sel.checked, "rows": [row_to_dict(r) for r in sel.rows], "pick": row_to_dict(sel.pick),
            "sequence": [row_to_dict(r) for r in sel.sequence], "close_call": sel.close_call,
            "safer": row_to_dict(sel.safer), "basis": sel.basis, "notes": list(sel.notes),
            "hand_picked": sel.hand_picked}


def selection_from_dict(d: Mapping[str, Any] | None) -> Selection | None:
    if not d or d.get("pick") is None:
        return None
    rows = tuple(r for r in (row_from_dict(x) for x in d.get("rows", [])) if r is not None)
    seq = tuple(r for r in (row_from_dict(x) for x in d.get("sequence", [])) if r is not None)
    pick = row_from_dict(d["pick"])
    assert pick is not None
    return Selection(model=d.get("model"), modality=d.get("modality", "image"), target=float(d["target"]),
                     n_pages=d.get("n_pages"), checked=bool(d.get("checked")), rows=rows, pick=pick, sequence=seq,
                     close_call=bool(d.get("close_call")), safer=row_from_dict(d.get("safer")),
                     basis=d.get("basis", ""), notes=tuple(d.get("notes", ())), hand_picked=bool(d.get("hand_picked")))


def _row_index(sequence: Sequence[LayoutRow], row: LayoutRow) -> int:
    return next((i for i, r in enumerate(sequence) if r.name == row.name), -1)


def safer_of(sequence: Sequence[LayoutRow], current: LayoutRow, proposed: LayoutRow) -> LayoutRow:
    """A gate may only move along the fixed sequence towards float16, never back (FINAL_DESIGN 5.5)."""
    i, j = _row_index(sequence, current), _row_index(sequence, proposed)
    return proposed if j >= 0 and j >= i else current


def float_row(sel: Selection | None, name: str = "float16") -> LayoutRow | None:
    return None if sel is None else next((r for r in sel.rows if r.name == name), None)


def catalogue_source(sel: Selection, row: LayoutRow) -> str:
    """The source of a catalogue claim: the datasets the rule used (FINAL_DESIGN 7, rule 3)."""
    return ", ".join(row.datasets) if row.datasets else (sel.basis or "our catalogue")


def matches_layout(row: LayoutRow | None, layout: str | None) -> bool:
    if layout is None or row is None:
        return True
    want = layout.strip().lower()
    return want in (row.name.lower(), row.plain.lower())


# ============================================================================ the build context


class _Ctx:
    """State of one build or sync, filled step by step and read by the summary."""

    def __init__(self, opts: BuildOptions, out: _Out) -> None:
        self.opts = opts
        self.out = out
        self.src: Source = Source(kind="files")
        self.path = Path()
        self.model_id: str | None = None
        self.info: Any = None
        self.adapter: Any = None
        self.modality: str = "image"
        self.target = DEFAULT_TARGET
        self.encoder: Any = None
        self.wiring: Any = None
        self.manifest: dict[str, Any] = {}
        self.existing = False
        self.diff = Diff()
        self.shards: ShardSet | None = None
        self.work: Path | None = None
        self.lock: BuildLock | None = None
        self.ns: str | None = None
        self.skipped: list[tuple[str, str]] = []
        self.notices: list[Line] = []
        self.profile: Any = None
        self.selection: Selection | None = None
        self.chosen: LayoutRow | None = None
        self.practice: dict[str, Any] = {"ran": False}
        self.queries_gate: dict[str, Any] = {"ran": False}
        self.hard: dict[str, Any] | None = None
        self.examples: list[str] = []
        self.probe_queries: Any = None
        self.checked = False
        self.unchecked_reason = ""
        self.search_seconds: float | None = None
        self.timed_queries = 0
        self.estimate: Any = None
        self.plan_pages = 0
        self.keep: list[PageMeta] = []
        self.keep_ids: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.blank: list[str] = []
        self.dim: int | None = None
        self.fitted: dict[str, Any] = {}
        self.sample: Any = None
        self.summary: Block | None = None
        self.sync = SyncReport()


def _model_label(ctx: _Ctx) -> str:
    if ctx.model_id:
        return ctx.model_id
    return "your own adapter" if ctx.adapter is not None else "no model named"


def _resolve(ctx: _Ctx) -> None:
    from ..adapters import EncoderAdapter

    model = ctx.opts.model
    if isinstance(model, EncoderAdapter):
        ctx.adapter = model
    elif isinstance(model, str):
        ctx.info = _resolve_model(model)
        ctx.model_id = None if ctx.info is None else str(ctx.info.model_id)
    elif model is not None:
        raise TypeError(f"model must be a registry id, an EncoderAdapter or None, got {type(model).__name__}")
    if ctx.info is not None:
        ctx.modality = str(ctx.info.modality)
    else:
        ctx.modality = "text" if ctx.src.kind == "text" else "image"


def _open_existing(ctx: _Ctx) -> None:
    """An index at the path: read its manifest; refuse a different model, target or layout
    (``TargetMismatch``) unless ``rebuild=True``."""
    path = ctx.path
    if not path.exists():
        return
    if not (path / "manifest.json").exists():
        if path.is_dir() and (not any(path.iterdir()) or (path / "segments").exists() or (path / "state").exists()):
            rmtree_retry(path)  # a build that stopped before writing its manifest: start the store again
            return
        raise ValueError(f"{path} exists and is not an optivision index; pass path= somewhere else")
    store = _store_open(path, "r")
    try:
        ctx.manifest = dict(store.manifest)
    finally:
        store.close()
    ctx.existing = True
    m = ctx.manifest
    if ctx.opts.rebuild:
        return
    stored_model = m.get("model")
    if ctx.model_id and stored_model and ctx.model_id != stored_model:
        raise TargetMismatch(_mismatch(ctx, f"built with {stored_model}", f"with {ctx.model_id}"))
    if ctx.opts.target is not None and abs(float(ctx.opts.target) - float(m.get("target", DEFAULT_TARGET))) > 1e-9:
        raise TargetMismatch(_mismatch(ctx, f"built at {m.get('target')}", f"at {ctx.opts.target}"))
    row = row_from_dict(m.get("layout"))
    if ctx.opts.layout is not None and not matches_layout(row, ctx.opts.layout):
        raise TargetMismatch(_mismatch(ctx, f"built with layout {row.name if row else None!r}",
                                       f"with layout {ctx.opts.layout!r}"))
    if m.get("source_kind") and m["source_kind"] != ctx.src.kind:
        raise ValueError(f"{path} was built from {m['source_kind']}; it cannot take {ctx.src.kind} as well")
    ctx.target = float(m.get("target", DEFAULT_TARGET))
    ctx.modality = str(m.get("modality") or ctx.modality)
    if ctx.model_id is None and stored_model and ctx.adapter is None:
        ctx.info = _resolve_model(stored_model)
        ctx.model_id = stored_model


def _mismatch(ctx: _Ctx, was: str, now: str) -> str:
    n = int(ctx.manifest.get("n_pages", 0))
    lines = [Line("Index", "{0} was {1}; pass rebuild=True to build it {2}", lit(ctx.path), lit(was), lit(now))]
    est = _estimate(ctx.manifest.get("model"), n)
    if ctx.manifest.get("keep_float16") and ctx.manifest.get("source_kind") != "vectors":
        lines.append(Line("Rebuild", "the index keeps its float16 copy, so that rebuild encodes nothing again"))
    elif est is not None:
        lines.append(Line("Rebuild", "re-encoding takes {0}", claim((est.lo_s, est.hi_s), "estimate",
                                                                   source=str(est.basis),
                                                                   text=_claim.duration(est.lo_s, est.hi_s))))
    return render(*lines)


# ============================================================================ plan and disk


def _plan(ctx: _Ctx) -> Block:
    """Count, estimate and check disk before encoding (FINAL_DESIGN 2 (c)); printed with ``Next:``."""
    lines: list[Line] = []
    to_encode = ctx.diff.to_encode
    if ctx.src.kind == "files":
        n_pages = 0
        for u in to_encode:
            try:
                n = _count_pages(Path(u.path or ""))
            except Exception:  # noqa: BLE001, S112 - counted one at a time; an unreadable file is skipped later
                continue
            n_pages += n if u.max_pages is None else min(n, u.max_pages)
        lines.append(Line("Folder", "{0} pages to encode in {1} files; {2} files skipped (listed in report())",
                          claim(n_pages, "counted"), claim(len(to_encode), "counted"),
                          claim(len(_unique(ctx.skipped)), "counted")))
    else:
        n_pages = len(to_encode)
        lines.append(Line("Input", "{0} {1} to add", claim(n_pages, "counted"),
                          lit("texts" if ctx.src.kind == "text" else "documents of vectors")))
    ctx.plan_pages = n_pages
    total = n_pages + int(ctx.manifest.get("n_pages", 0))
    plan_sel = None
    with contextlib.suppress(Exception):  # the plan is advice; the build selects again on the real pages
        plan_sel = _select(ctx.model_id, ctx.target, max(total, 1), checked=False, modality=ctx.modality,
                           vectors_per_page=None, layout=ctx.opts.layout)
    if ctx.src.kind != "vectors":
        ctx.estimate = _estimate(ctx.model_id, n_pages)
    est = ctx.estimate
    if est is not None:
        lines.append(Line("Encoding", "{0} on {1}", claim((est.lo_s, est.hi_s), "estimate", source=str(est.basis),
                                                          text=_claim.duration(est.lo_s, est.hi_s)), lit(est.gpu)))
    elif ctx.src.kind != "vectors" and n_pages:
        lines.append(Line("Encoding", "time not known for this model; it is shown as pages are encoded"))
    f16 = float_row(plan_sel, "float16")
    f16_page = f16.bytes_per_page[1] if f16 is not None and f16.bytes_per_page is not None else None
    temp_need = int(f16_page * n_pages) if f16_page is not None and ctx.work is not None else 0
    if temp_need:
        lines.append(Line("Temporary", "{0} for the float16 copy",
                          claim(temp_need, "estimate", source="our bytes per page x your pages",
                                text=_claim.size(temp_need))))
    if ctx.work is not None:
        lines.append(Line("Location", "{0}", lit(ctx.work)))
        free = free_bytes(ctx.work)
        lines.append(Line("Free", "{0} reported by that drive", claim(free, "counted", text=_claim.size(free))))
    index_need = 0
    if plan_sel is not None and plan_sel.pick.bytes_per_page is not None:
        index_need = int(plan_sel.pick.bytes_per_page[1] * n_pages)
    if ctx.opts.keep_float16 and f16_page is not None:
        index_need += int(f16_page * n_pages)
        lines.append(Line("Keep copy", "keep_float16=True keeps the float16 copy in the index: {0} more",
                          claim(int(f16_page * n_pages), "estimate", source="our bytes per page x your pages",
                                text=_claim.size(f16_page * n_pages))))
    disk_check(ctx.work, ctx.path.parent, temp_need, index_need)
    if est is not None and est.hi_s > COLAB_WARN_SECONDS and on_colab() and ctx.work is not None \
            and not str(ctx.work).replace("\\", "/").startswith("/content/drive"):
        lines.append(Line("Colab", "A Colab disconnect loses the temporary copy in /content. Pass work_dir= on "
                                   "Drive to resume in a new session."))
    if n_pages and ctx.src.kind != "vectors" and not ctx.opts.keep_float16:
        lines.append(Line("If wrong", "re-encoding costs the same time; keep_float16=True keeps the copy so a "
                                      "change needs none"))
    nxt = "encoding now; rerun the same line to resume after an interruption" if ctx.src.kind != "vectors" \
        else "compressing now"
    return Block(lines, next=nxt, title="Plan")


def disk_check(work: Path | None, index_dir: Path, temp_need: int, index_need: int) -> None:
    """``NotEnoughDisk`` before encoding when the float16 copy or the index will not fit."""
    checks: list[tuple[Path, int]] = []
    if work is not None and temp_need:
        if _same_device(work, index_dir):
            checks.append((work, temp_need + index_need))
        else:
            checks.append((work, temp_need))
            if index_need:
                checks.append((index_dir, index_need))
    elif index_need:
        checks.append((index_dir, index_need))
    for where, need in checks:
        free = free_bytes(where)
        if need > free:
            raise NotEnoughDisk(render(
                Line("Needed", "{0} in {1}", claim(need, "estimate", source="our bytes per page x your pages",
                                                   text=_claim.size(need)), lit(where)),
                Line("Free", "{0} reported by that drive", claim(free, "counted", text=_claim.size(free))),
                Line("", "free some space, or pass work_dir= (or path=) on a drive with room")))


def _flush_check(ctx: _Ctx) -> None:
    """At every shard: touch the lock; the bytes written so far, scaled to the pages left, must fit."""
    if ctx.lock is not None:
        ctx.lock.touch()
    shards = ctx.shards
    if shards is None or shards.root is None:
        return
    done = sum(1 for m in shards.meta.values() if m.stored)
    left = max(ctx.plan_pages - done, 0)
    if not done or not left:
        return
    used = sum(f.stat().st_size for f in shards.root.glob("s*/vectors.npy"))
    need = int(used / done * left)
    free = free_bytes(shards.root)
    if need > free:
        raise NotEnoughDisk(render(
            Line("Needed", "{0} more in {1}", claim(need, "estimate", source="bytes written so far x pages left",
                                                     text=_claim.size(need)), lit(shards.root)),
            Line("Free", "{0} reported by that drive", claim(free, "counted", text=_claim.size(free))),
            Line("", "free some space and rerun the same line: the pages encoded so far are kept")))


# ============================================================================ encode step


def _shard_identity(ctx: _Ctx) -> dict[str, Any]:
    rec = getattr(ctx.encoder, "record", None)
    rec = rec if isinstance(rec, Mapping) else {}
    return {"model": ctx.model_id or ("adapter" if ctx.adapter is not None else None), "dtype": rec.get("dtype"),
            "revision": rec.get("revision"), "kind": ctx.src.kind}


def _ensure_encoder(ctx: _Ctx) -> None:
    if ctx.encoder is not None:
        return
    model = ctx.adapter if ctx.adapter is not None else ctx.model_id
    if model is None:
        if ctx.manifest.get("adapter"):
            raise EasyError(f"{ctx.path} was built with your own adapter object; pass it again as model=")
        raise ValueError("pass model= (a registry id or an EncoderAdapter) to encode pages or questions")
    dtype = ctx.opts.dtype
    if ctx.existing and dtype == "recorded" and (ctx.manifest.get("record") or {}).get("dtype"):
        dtype = str(ctx.manifest["record"]["dtype"])  # a sync encodes as the build did
    ctx.encoder = _load_encoder(model, device=ctx.opts.device, dtype=dtype,
                                trust_remote_code=ctx.opts.trust_remote_code)


def _encode_step(ctx: _Ctx) -> None:
    """Wiring, then the shard loop (resumable). Vectors go into in-memory shards."""
    units = ctx.diff.to_encode
    if ctx.src.kind == "vectors":
        ctx.shards = ShardSet(None, {"kind": "vectors"})
        load_vectors(units, ctx.shards, ctx.manifest.get("sha_index", {}))
        ctx.dim = ctx.src.dim
        return
    if not units:
        return
    _ensure_encoder(ctx)
    if ctx.src.kind == "files":
        ctx.wiring = _wiring(ctx.encoder, wiring_pages(units))
    ctx.shards = ShardSet(ctx.work, _shard_identity(ctx))
    ctx.shards.on_flush = lambda: _flush_check(ctx)
    enc = Encoding(ctx.shards, ctx.encoder, ctx.out, ctx.manifest.get("sha_index", {}), ctx.plan_pages)
    enc.run(units)
    ctx.skipped.extend(enc.skipped)
    ctx.dim = enc.dim or ctx.shards.dim()


def _run_pages(ctx: _Ctx) -> list[PageMeta]:
    """Metas of this run's units that finished (or, for ``sample=``, that have pages), in order."""
    shards = ctx.shards
    if shards is None:
        return []
    want = [u.doc_id for u in ctx.diff.to_encode
            if shards.complete(u.doc_id) or (u.max_pages is not None and u.doc_id in shards.files)]
    pos = {d: i for i, d in enumerate(want)}
    metas = [m for m in shards.meta.values() if m.doc_id in pos]
    metas.sort(key=lambda m: (pos[m.doc_id], m.page or 0))
    return metas


def split_pages(metas: list[PageMeta], stored: set[str]) -> tuple[list[PageMeta], dict[str, str], list[str],
                                                                 set[str]]:
    """Stored pages, aliases, blank pages; an alias of a page that is not stored (its file was skipped)
    is dropped and its id returned."""
    keep = [m for m in metas if m.stored]
    have = stored | {m.id for m in keep}
    aliases: dict[str, str] = {}
    dropped: set[str] = set()
    for m in metas:
        if m.alias_of is not None:
            if m.alias_of in have:
                aliases[m.id] = m.alias_of
            else:
                dropped.add(m.id)
    return keep, aliases, [m.id for m in metas if m.blank], dropped


# ============================================================================ practice and gates


def _practice_questions(ctx: _Ctx) -> None:
    """Practice questions from the pages' text, ranked on the float16 copy; the hard set decides
    whether the folder is checked (FINAL_DESIGN 5.3.1)."""
    opts = ctx.opts
    keep = ctx.keep
    if ctx.src.kind == "vectors":
        ctx.unchecked_reason = "built from vectors, so no practice questions can be made"
        return
    if opts.probes is None or opts.probes == "none":
        ctx.unchecked_reason = "practice questions switched off (probes=None)"
        return
    if not _ledger.is_on("practice_check"):
        ctx.unchecked_reason = "the practice check is switched off: " + str(
            _ledger.feature("practice_check").get("reason", ""))
    elif opts.probes == "text" and ctx.profile is not None and not bool(getattr(ctx.profile, "checkable", True)):
        ctx.unchecked_reason = str(getattr(ctx.profile, "reason", "") or "too little text for practice questions")
    if ctx.unchecked_reason and opts.probes != "text":
        return  # a model or an API call per page only for examples: not worth it
    probes = _make_probes(str(opts.probes), keep, allow_upload=opts.allow_upload)
    ctx.practice.update(kind=opts.probes, n_probes=len(probes))
    if not probes:
        ctx.unchecked_reason = ctx.unchecked_reason or "no practice questions could be made from the text"
        return
    if ctx.unchecked_reason:  # questions serve as examples only
        ctx.examples = [str(p.text) for p in probes][:MAX_EXAMPLES]
        return
    from ..representation import MultiVectorCorpus

    _ensure_encoder(ctx)
    pids = [str(p.id) for p in probes]
    raw = _encode_queries(ctx.encoder, [str(p.text) for p in probes])
    qv = MultiVectorCorpus(raw.vectors, raw.offsets, ids=pids)
    assert ctx.shards is not None
    ref = dict(zip(pids, rank_shards(ctx.shards, ctx.keep_ids, qv, RANK_DEPTH)))
    hard = _hard_set(probes, ref)
    hard_probes = _hard_probes(hard)
    hard_ids = [str(p.id) for p in hard_probes]
    ctx.examples = [str(p.text) for p in probes if str(p.id) not in set(hard_ids)][:MAX_EXAMPLES]
    ctx.practice.update(n_hard=len(hard_probes), sha256=_hard_sha(hard))
    index_of = {pid: i for i, pid in enumerate(pids)}
    ctx.probe_queries = qv.select([index_of[h] for h in hard_ids]) if hard_ids else None
    doc_of = {m.id: m.doc_id for m in keep}
    ctx.hard = {
        "probes": hard_probes,
        "ref": {h: [i for i, _ in ref[h]] for h in hard_ids},
        "ref_scores": {h: ref[h] for h in hard_ids},
        "relevant": {str(p.id): frozenset(ctx.aliases.get(str(r), str(r)) for r in p.relevant) for p in hard_probes},
        # page-cluster bootstrap: practice questions of one file move together
        "clusters": {str(p.id): doc_of.get(str(p.page_id), str(p.page_id)) for p in hard_probes},
    }
    if len(hard_probes) >= MIN_QUERIES:
        ctx.checked = True
    else:
        ctx.unchecked_reason = "fewer hard practice questions than the check needs"


def _fitted(ctx: _Ctx, row: LayoutRow) -> Any:
    """The pipeline of ``row`` (built from its catalogue dict, never parsed) fitted on the seeded sample."""
    from ..compose import Pipeline

    if row.name not in ctx.fitted:
        if ctx.sample is None:
            assert ctx.shards is not None
            ctx.sample = fit_sample(ctx.shards, ctx.keep_ids)
        ctx.fitted[row.name] = Pipeline.from_dict(row.spec).fit(ctx.sample)
    return ctx.fitted[row.name]


def _rank_fn(ctx: _Ctx, queries: Any, qids: list[str]) -> Callable[[LayoutRow], dict[str, list[str]]]:
    def rank(row: LayoutRow) -> dict[str, list[str]]:
        assert ctx.shards is not None
        res = rank_shards(ctx.shards, ctx.keep_ids, queries, RANK_DEPTH, _fitted(ctx, row))
        return {q: [i for i, _ in hits] for q, hits in zip(qids, res)}

    return rank


def _gate_steps(result: Any) -> list[dict[str, Any]]:
    return [{"layout": row.name, "plain": row.plain, "R": float(stat.R), "U": float(stat.U), "n": int(stat.n),
             "passed": bool(stat.passed)} for row, stat in getattr(result, "steps", [])]


def _practice_gate(ctx: _Ctx) -> None:
    """The practice check: the fixed sequence from the current layout towards float16 (FINAL_DESIGN 5.5.1).
    Ledger mode ``step-down`` acts on it; any other mode only reports it."""
    if not ctx.checked or ctx.hard is None or not _ledger.is_on("practice_check"):
        return
    assert ctx.selection is not None and ctx.chosen is not None
    mode = str(_ledger.setting("practice_check", "mode", "step-down"))
    seq = list(ctx.selection.sequence)
    start = max(_row_index(seq, ctx.chosen), 0)
    qids = [str(p.id) for p in ctx.hard["probes"]]
    result = _fixed_sequence(seq[start:], _rank_fn(ctx, ctx.probe_queries, qids), ctx.hard["ref"],
                             ctx.hard["relevant"], ctx.hard["clusters"], ctx.target)
    tested = ctx.chosen
    if mode == "step-down":
        ctx.chosen = safer_of(seq, ctx.chosen, result.chosen)
    ctx.practice.update(ran=True, mode=mode, steps=_gate_steps(result), tested=tested.name, chosen=ctx.chosen.name,
                        fired=result.chosen.name != tested.name)


def answers_relevant(answers: Sequence[Any], files: Mapping[str, Mapping[str, Any]], aliases: Mapping[str, str],
                     stored: set[str]) -> list[frozenset[str]]:
    """``answers``: one entry per query, a page id, a ``(file, page)`` pair, or a set of either."""
    from ..types import PageRef

    def doc_of(file: Any) -> str:
        f = str(file).replace("\\", "/")
        if f in files:
            return f
        with contextlib.suppress(OSError):
            p = str(Path(file).expanduser().resolve())
            for doc_id, e in files.items():
                if e.get("path") == p:
                    return doc_id
        hits = [d for d, e in files.items() if e.get("relpath") == f or d.endswith("/" + f)]
        if len(hits) == 1:
            return hits[0]
        raise ValueError(f"answers: {file!r} is not a file of this index" if not hits else
                         f"answers: {file!r} matches several files: {hits[:3]}")

    def one(a: Any) -> str:
        if isinstance(a, tuple) and len(a) == 2 and not isinstance(a[1], str):
            pid = PageRef(doc_of(a[0]), int(a[1])).page_id
        else:
            pid = str(a)
        pid = aliases.get(pid, pid)
        if pid not in stored:
            raise ValueError(f"answers: {pid!r} is not a stored page of this index")
        return pid

    return [frozenset(one(x) for x in (list(a) if isinstance(a, (set, frozenset, list)) else [a])) for a in answers]


def _query_gate(ctx: _Ctx) -> None:
    """The real-query gate (FINAL_DESIGN 5.5.2): agreement with the float16 top page without answers,
    search quality with them; at least 60 queries; only towards safer layouts."""
    from ..representation import MultiVectorCorpus

    queries = ctx.opts.queries
    if queries is None:
        return
    assert ctx.selection is not None and ctx.chosen is not None and ctx.shards is not None
    if isinstance(queries, MultiVectorCorpus):
        raw = queries
    else:
        texts = list(dict.fromkeys(str(q) for q in queries))  # de-duplicated, in order
        _ensure_encoder(ctx)
        raw = _encode_queries(ctx.encoder, texts)
    n = len(raw)
    qids = [f"u{i}" for i in range(n)]
    qv = MultiVectorCorpus(raw.vectors, raw.offsets, ids=qids)
    ctx.queries_gate.update(n=n)
    if n < MIN_QUERIES:
        ctx.queries_gate.update(ran=False, reason="too few queries")
        return
    ref = {q: [i for i, _ in hits] for q, hits in zip(qids, rank_shards(ctx.shards, ctx.keep_ids, qv, RANK_DEPTH))}
    if ctx.opts.answers is not None:
        if len(ctx.opts.answers) != n:
            raise ValueError(f"{len(ctx.opts.answers)} answers for {n} queries (one per query, duplicates removed)")
        files = {**ctx.manifest.get("files", {}), **{u.doc_id: u.entry() for u in ctx.diff.to_encode}}
        relevant = dict(zip(qids, answers_relevant(ctx.opts.answers, files, ctx.aliases, ctx.keep_ids)))
        kind = "search quality"
    else:
        relevant = {q: frozenset(ref[q][:1]) for q in qids}
        kind = "agreement with full precision, not relevance"
    seq = list(ctx.selection.sequence)
    start = max(_row_index(seq, ctx.chosen), 0)
    result = _fixed_sequence(seq[start:], _rank_fn(ctx, qv, qids), ref, relevant, {q: q for q in qids}, ctx.target)
    tested = ctx.chosen
    ctx.chosen = safer_of(seq, ctx.chosen, result.chosen)
    ctx.queries_gate.update(ran=True, kind=kind, steps=_gate_steps(result), tested=tested.name,
                            chosen=ctx.chosen.name, fired=ctx.chosen.name != tested.name)


def _select_step(ctx: _Ctx) -> None:
    n_pages = len(ctx.keep)
    counts = [m.n_vectors for m in ctx.keep if m.n_vectors]
    vpp = float(np.median(counts)) if counts else None
    if ctx.opts.max_disk is not None:
        fit = _budget_target(ctx.model_id, n_pages, parse_bytes(ctx.opts.max_disk), checked=ctx.checked,
                             modality=ctx.modality, vectors_per_page=vpp)
        if fit is None:
            raise NotEnoughDisk(render(
                Line("Budget", "no layout fits {0} for {1} pages", lit(f"max_disk={ctx.opts.max_disk!r}"),
                     claim(n_pages, "counted")),
                Line("", "raise max_disk= or build from fewer pages")))
        ctx.target = float(fit)
        ctx.notices.append(Line("Budget", "max_disk= picked target {0}, the highest that fits",
                                lit(target_text(ctx.target))))
    ctx.selection = _select(ctx.model_id, ctx.target, n_pages, checked=ctx.checked, modality=ctx.modality,
                            vectors_per_page=vpp, layout=ctx.opts.layout)
    ctx.chosen = ctx.selection.pick


# ============================================================================ build


def build(source: Any, opts: BuildOptions) -> BuildResult:
    """Build or sync the index of ``source`` (the order is in the module docstring)."""
    out = _Out(opts.progress)
    ctx = _Ctx(opts, out)
    ctx.encoder = opts.encoder
    if opts.max_disk is not None and opts.target is not None:
        raise ValueError("pass target= or max_disk=, not both: max_disk= picks the target")
    ctx.src = classify(source, opts.ids)
    _resolve(ctx)
    ctx.path = (Path(opts.path).expanduser() if opts.path is not None else default_path(ctx.src)).resolve()
    if opts.sample is not None and opts.path is None:
        ctx.path = ctx.path.with_name(ctx.path.stem + ".sample.ovx")
    if opts.target is not None:
        ctx.target = float(opts.target)
    _open_existing(ctx)
    if ctx.existing and opts.rebuild:
        return _rebuild(source, ctx)
    # a trial (sample=) writes the shards the full build of the same folder will look for
    full_path = ctx.path.with_name(ctx.path.name.replace(".sample.ovx", ".ovx")) if opts.sample is not None         else ctx.path
    ctx.work = work_root(full_path, opts.work_dir, ctx.src.kind)
    ctx.lock = BuildLock(ctx.path)
    with ctx.lock:
        _list_and_diff(ctx)
        if opts.sample is not None:
            _apply_sample(ctx, int(opts.sample))
        if ctx.diff.to_encode:
            out.block(_plan(ctx))
        elif not ctx.existing:
            raise ValueError("nothing to index: no readable pages" +
                             (f" ({len(ctx.skipped)} files skipped: {ctx.skipped[:3]})" if ctx.skipped else ""))
        if ctx.existing:
            _sync(ctx)
        else:
            _encode_step(ctx)
            _first_build(ctx)
    return BuildResult(path=ctx.path, encoder=ctx.encoder, adapter=ctx.adapter, summary=ctx.summary, sync=ctx.sync,
                       blocks=out.blocks)


def _list_and_diff(ctx: _Ctx) -> None:
    known = ctx.manifest.get("files", {})
    if ctx.src.kind == "files":
        units, ns = file_units(ctx.src, ctx.manifest.get("sources", {}))
        ctx.ns = ns
        ctx.skipped.extend(ctx.src.skipped)
        ctx.diff = diff_units(units, known, ns, full_listing=ctx.src.single is None)
    else:
        ctx.diff = diff_units(ctx.src.units, known, None, full_listing=True)
    ctx.notices.extend(ctx.src.notices)


def _apply_sample(ctx: _Ctx, n: int) -> None:
    """``sample=n``: a trial on about ``n`` pages, the leading pages of files taken in a seeded order. Its
    shards are those of the full build (same ``work_dir``), which then encodes only the rest."""
    if ctx.existing:
        raise ValueError(f"sample= makes a trial index, and {ctx.path} exists; pass another path=")
    if n < 1:
        raise ValueError("sample must be a positive number of pages")
    units = ctx.diff.to_encode
    order = np.random.default_rng(FIT_SEED).permutation(len(units))
    if ctx.src.kind == "files":
        per_file = max(1, -(-n // max(1, min(len(units), 30))))
        chosen: list[Unit] = []
        for i in order:
            u = units[int(i)]
            u.max_pages = per_file
            chosen.append(u)
            if len(chosen) * per_file >= n:
                break
        ctx.notices.append(Line("Sample", "a trial on up to {0} pages ({1} per file); the full build reuses its "
                                          "encoded pages", claim(len(chosen) * per_file, "counted"),
                                claim(per_file, "counted")))
    else:
        chosen = [units[int(i)] for i in sorted(order[: min(n, len(units))])]
    ids = {u.doc_id for u in chosen}
    ctx.diff = Diff(new=[u for u in units if u.doc_id in ids], order=ctx.diff.order)


def _first_build(ctx: _Ctx) -> None:
    """Profile, practice questions, selection, gates, settle into a new store, summary."""
    metas = _run_pages(ctx)
    keep, aliases, blank, dropped = split_pages(metas, set())
    if not keep:
        raise ValueError("no page could be encoded: every file was skipped or blank" +
                         (f" ({_unique(ctx.skipped)[:3]})" if ctx.skipped else ""))
    ctx.keep, ctx.keep_ids, ctx.aliases, ctx.blank = keep, {m.id for m in keep}, aliases, blank
    if ctx.src.kind != "vectors":
        profiled = [m for m in metas if m.id not in dropped]
        ctx.profile = _profile(profiled, [m.n_vectors for m in profiled])
    _practice_questions(ctx)
    _select_step(ctx)
    _practice_gate(ctx)
    _query_gate(ctx)
    assert ctx.chosen is not None
    pipeline = _fitted(ctx, ctx.chosen)
    info = {"model": ctx.model_id, "target": ctx.target, "layout": ctx.chosen.name, "record": _record(ctx),
            "created": _now()}
    store = _store_create(ctx.path, pipeline, info)
    ok = False
    try:
        with _store_lock(store):
            f16_bytes = _settle(ctx, store, pipeline, keep)
            _set_reference(ctx, store)
            store.commit(**_manifest(ctx, metas, f16_bytes))
            ctx.search_seconds = _time_search(ctx, store)
            ctx.summary = _summary(ctx, store, f16_bytes)
            store.commit(search_seconds=ctx.search_seconds, summary=ctx.summary.as_dict(),
                         evidence=[c.as_dict() for c in ctx.summary.claims()])
        ok = True
    finally:
        store.close()
        if not ok:  # no half index: the shards stay, so a rerun settles again without encoding
            rmtree_retry(ctx.path)
    assert ctx.shards is not None
    ctx.sync = SyncReport(added=[(u.path or u.doc_id, "new") for u in ctx.diff.to_encode
                                 if ctx.shards.complete(u.doc_id) or u.max_pages is not None],
                          skipped=_unique(ctx.skipped), aliases=len(aliases), n_pages=len(keep), message="built")
    ctx.out.block(ctx.summary)
    _cleanup(ctx)


def _unique(pairs: Sequence[Sequence[str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out = []
    for p in pairs:
        if p[0] not in seen:
            seen.add(p[0])
            out.append((str(p[0]), str(p[1])))
    return out


def _cleanup(ctx: _Ctx) -> None:
    """The shards go once the store holds the codes (their maps closed first); a trial (``sample=``)
    keeps them for the full build."""
    ctx.sample = None
    ctx.fitted.clear()
    if ctx.shards is None or ctx.opts.sample is not None:
        return
    if not ctx.shards.delete():
        ctx.out.line(Line("Temporary", "could not delete {0} (a file is still open); delete it by hand",
                          lit(ctx.shards.root)))


def _record(ctx: _Ctx) -> dict[str, Any]:
    rec = getattr(ctx.encoder, "record", None)
    if isinstance(rec, Mapping):
        return dict(rec)
    if ctx.manifest.get("record"):
        return dict(ctx.manifest["record"])
    return {"model": ctx.model_id}


def _settle(ctx: _Ctx, store: Any, pipeline: Any, keep: list[PageMeta]) -> int:
    """``compress(segment, fit=False)`` shard by shard into the store; returns the float16 bytes settled.
    The float16 copy of 20,000 pages never sits in memory at once."""
    assert ctx.shards is not None
    text = {m.id: m.text for m in keep}
    total = 0
    for name, pids in ctx.shards.encoded({m.id for m in keep}):
        seg = ctx.shards.load(name, pids)
        compressed = pipeline.compress(seg, fit=False)
        texts = {i: text[i] for i in seg.ids} if ctx.opts.keep_text else None
        _store_append(store, compressed, texts, seg if ctx.opts.keep_float16 else None)
        total += int(seg.num_vectors) * int(seg.dimension) * 2
        del seg, compressed
    with contextlib.suppress(AttributeError):
        if len(store.segments()) > COMPACT_ABOVE:
            store.compact()
    return total


def _set_reference(ctx: _Ctx, store: Any) -> None:
    if ctx.hard is None or ctx.probe_queries is None:
        return
    hard = ctx.hard
    _store_set_reference(store, {
        "target": ctx.target, "sha256": ctx.practice.get("sha256", ""),
        "probes": [{"id": str(p.id), "text": str(p.text), "family": str(getattr(p, "family", "")),
                    "page_id": str(p.page_id), "relevant": sorted(hard["relevant"][str(p.id)])}
                   for p in hard["probes"]],
        "clusters": dict(hard["clusters"]), "queries": ctx.probe_queries,
        "ref_top": {q: [[i, float(s)] for i, s in hits] for q, hits in hard["ref_scores"].items()},
    })


def _files_entries(ctx: _Ctx, metas: list[PageMeta], dim: int | None) -> dict[str, Any]:
    """Manifest ``files`` entries of this run's units."""
    assert ctx.shards is not None
    by_doc: dict[str, list[PageMeta]] = {}
    for m in metas:
        by_doc.setdefault(m.doc_id, []).append(m)
    out = {}
    for u in ctx.diff.to_encode:
        pages = by_doc.get(u.doc_id)
        if pages is None:
            continue
        e = u.entry()
        e["sha256"] = ctx.shards.files.get(u.doc_id, {}).get("sha256") or u.sha
        e["pages"] = [m.id for m in pages]
        e["stored"] = [m.id for m in pages if m.id in ctx.keep_ids]
        e["f16_bytes"] = sum(m.n_vectors for m in pages if m.id in ctx.keep_ids) * int(dim or 0) * 2
        e["partial"] = not ctx.shards.complete(u.doc_id)
        if e["partial"]:  # a trial's file: a later sync of this index encodes the rest of it
            e.update(size=-1, sha256=None)
        out[u.doc_id] = e
    return out


def _manifest(ctx: _Ctx, metas: list[PageMeta], f16_bytes: int) -> dict[str, Any]:
    from .. import __version__

    assert ctx.chosen is not None
    sources = dict(ctx.manifest.get("sources", {}))
    if ctx.src.kind == "files" and ctx.ns is not None and ctx.src.root is not None:
        sources[ctx.ns] = str(ctx.src.root)
    return {
        "schema": MANIFEST_SCHEMA, "kind": MANIFEST_KIND, "optivision": __version__, "model": ctx.model_id,
        "adapter": ctx.adapter is not None, "modality": ctx.modality, "source_kind": ctx.src.kind,
        "target": ctx.target, "layout": row_to_dict(ctx.chosen), "selection": selection_to_dict(ctx.selection),
        "checked": ctx.checked, "unchecked_reason": ctx.unchecked_reason, "sources": sources,
        "files": _files_entries(ctx, metas, ctx.dim), "aliases": dict(ctx.aliases), "blank": list(ctx.blank),
        "sha_index": {m.sha256: m.id for m in ctx.keep if m.sha256}, "skipped": [list(p) for p in _unique(ctx.skipped)],
        "n_pages": len(ctx.keep), "dim": ctx.dim, "float16_bytes": int(f16_bytes),
        "keep_text": bool(ctx.opts.keep_text), "keep_float16": bool(ctx.opts.keep_float16), "record": _record(ctx),
        "trust_remote_code": bool(ctx.opts.trust_remote_code), "practice": _jsonable(ctx.practice),
        "queries_gate": _jsonable(ctx.queries_gate), "examples": list(ctx.examples),
        "upgrade_attempts": int(ctx.manifest.get("upgrade_attempts", 0)),
        "work_dir": None if ctx.work is None else str(ctx.work), "built": _now(), "updated": _now(),
    }


def _jsonable(d: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(d, default=str))


def _time_search(ctx: _Ctx, store: Any) -> float | None:
    """Seconds per query of the stored index, on 20 sample queries (one warm-up query first)."""
    assert ctx.shards is not None
    q = timing_queries(ctx.shards, ctx.keep_ids, ctx.probe_queries)
    if q is None or not len(q):
        return None
    searcher = _searcher(store)
    try:
        searcher.search(q.select([0]), 5)
        t0 = time.perf_counter()
        searcher.search(q, 5)
        dt = time.perf_counter() - t0
    finally:
        close = getattr(searcher, "close", None)
        if close is not None:
            close()
    ctx.timed_queries = len(q)
    return dt / len(q)


# ============================================================================ sync


def _sync(ctx: _Ctx) -> None:
    """Re-running ``build`` (or ``Index.add``) on an existing index: new and changed files go through the
    frozen fitted pipeline; missing files are listed and kept unless ``prune=True``."""
    m = ctx.manifest
    # new pages are stored the way the index stores the others
    ctx.opts.keep_text = bool(m.get("keep_text", ctx.opts.keep_text))
    ctx.opts.keep_float16 = bool(m.get("keep_float16", ctx.opts.keep_float16))
    known: dict[str, dict[str, Any]] = {k: dict(v) for k, v in m.get("files", {}).items()}
    old_aliases: dict[str, str] = dict(m.get("aliases", {}))
    removed: set[str] = set()
    for u in ctx.diff.changed:
        removed.update(known.get(u.doc_id, {}).get("stored", []))
    if ctx.opts.prune:
        for d in ctx.diff.missing:
            removed.update(known.get(d, {}).get("stored", []))
    _reencode_orphans(ctx, known, old_aliases, removed)
    to_encode = ctx.diff.to_encode
    if to_encode and ctx.src.kind != "vectors":
        _ensure_encoder(ctx)
        for d in record_differences(m.get("record") or {}, _record(ctx)):
            ctx.notices.append(Line("Record", "{0}", lit(d)))
    if to_encode:
        _encode_step(ctx)
    metas = _run_pages(ctx)
    stored_before = {p for e in known.values() for p in e.get("stored", [])}
    keep, new_aliases, blank, _ = split_pages(metas, stored_before - removed)
    ctx.keep, ctx.keep_ids = keep, {x.id for x in keep}
    gone_docs = {u.doc_id for u in to_encode} | (set(ctx.diff.missing) if ctx.opts.prune else set())
    ctx.aliases = {a: c for a, c in old_aliases.items() if c not in removed and _doc_of(a) not in gone_docs}
    ctx.aliases.update(new_aliases)
    if to_encode or removed or ctx.diff.touched:
        store = _store_open(ctx.path, "a")
        try:
            with _store_lock(store):
                if removed:
                    store.remove(sorted(removed))
                if keep:
                    _settle(ctx, store, store.pipeline(), keep)
                store.commit(**_sync_manifest(ctx, known, metas, removed, blank))
        finally:
            store.close()
    n_after = len(stored_before - removed) + len(keep)
    ctx.sync = _sync_report(ctx, len(new_aliases), n_after, bool(keep or removed))
    block = _sync_block(ctx, n_after, int(m.get("n_pages", 0)))
    ctx.sync.message = str(block)  # the printed lines, so a caller sees what the console showed
    ctx.out.block(block)
    _cleanup(ctx)


def _doc_of(page_id: str) -> str:
    return page_id.rsplit("::p", 1)[0] if "::p" in page_id else page_id


def _reencode_orphans(ctx: _Ctx, known: Mapping[str, Mapping[str, Any]], aliases: Mapping[str, str],
                      removed: set[str]) -> None:
    """Pages of other files stored as aliases of a removed page lose their vectors: encode those files
    again in this sync."""
    busy = {u.doc_id for u in ctx.diff.changed} | set(ctx.diff.missing)
    for d in sorted({_doc_of(a) for a, c in aliases.items() if c in removed} - busy):
        e = known.get(d)
        if e is None or not e.get("path") or not Path(e["path"]).exists():
            continue
        files, _ = _list_source(Path(e["path"]))
        if not files:
            continue
        p = Path(e["path"])
        st = p.stat()
        ctx.diff.changed.append(Unit(doc_id=d, kind=str(e.get("kind", "pdf")), path=str(p), relpath=e.get("relpath"),
                                     ns=str(e.get("ns", "")), size=int(st.st_size), mtime=int(st.st_mtime_ns),
                                     record=_with_doc_id(files[0], d)))
        removed.update(e.get("stored", []))


def record_differences(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """``"dtype: 'float16' at build, 'float32' now"`` for each key both records hold that differ
    (FINAL_DESIGN 4.3: versions, resolved commit, dtype, attention, dpi/max_side, batch size)."""
    skip = {"device", "time", "created", "batch_size_final"}
    return [f"{k}: {old[k]!r} at build, {new[k]!r} now" for k in sorted(set(old) & set(new))
            if k not in skip and old[k] != new[k]]


def _sync_manifest(ctx: _Ctx, known: dict[str, dict[str, Any]], metas: list[PageMeta], removed: set[str],
                   blank: list[str]) -> dict[str, Any]:
    files = dict(known)
    for d in (ctx.diff.missing if ctx.opts.prune else []):
        files.pop(d, None)
    for u in ctx.diff.changed:
        files.pop(u.doc_id, None)
    if ctx.shards is not None:
        files.update(_files_entries(ctx, metas, ctx.dim or int(ctx.manifest.get("dim") or 0)))
    for u in ctx.diff.touched:  # a touched file with the same content keeps its new key
        if u.doc_id in files:
            files[u.doc_id].update(size=u.size, mtime=u.mtime)
    redone = {u.doc_id for u in ctx.diff.changed}
    sha_index = {s: p for s, p in ctx.manifest.get("sha_index", {}).items() if p not in removed}
    sha_index.update({x.sha256: x.id for x in ctx.keep if x.sha256})
    old_blank = [b for b in ctx.manifest.get("blank", []) if _doc_of(b) in files and _doc_of(b) not in redone]
    sources = dict(ctx.manifest.get("sources", {}))
    if ctx.src.kind == "files" and ctx.ns is not None and ctx.src.root is not None:
        sources[ctx.ns] = str(ctx.src.root)
    return {"files": files, "aliases": dict(ctx.aliases), "blank": old_blank + list(blank), "sha_index": sha_index,
            "n_pages": sum(len(e.get("stored", [])) for e in files.values()),
            "float16_bytes": sum(int(e.get("f16_bytes", 0)) for e in files.values()), "sources": sources,
            "skipped": [list(p) for p in _unique([*ctx.manifest.get("skipped", []), *ctx.skipped])],
            "updated": _now(), "dim": ctx.dim or ctx.manifest.get("dim")}


def _sync_report(ctx: _Ctx, n_aliases: int, n_after: int, changed: bool) -> SyncReport:
    shards = ctx.shards
    done = {u.doc_id for u in ctx.diff.to_encode if shards is not None and shards.complete(u.doc_id)}
    known = ctx.manifest.get("files", {})
    added = [(u.path or u.doc_id, "new") for u in ctx.diff.new if u.doc_id in done]
    redone = [(u.path or u.doc_id, "changed" if u.doc_id in known and u.doc_id not in
               {x.doc_id for x in ctx.diff.new} else "re-encoded") for u in ctx.diff.changed if u.doc_id in done]
    missing = [(known.get(d, {}).get("path") or d, "removed (prune=True)" if ctx.opts.prune else "missing; kept")
               for d in ctx.diff.missing]
    still, message = still_qualifies(ctx.manifest, n_after) if changed else (True, "")
    return SyncReport(added=added, changed=redone, missing=missing, skipped=_unique(ctx.skipped), aliases=n_aliases,
                      n_pages=n_after, layout_still_qualifies=still, message=message)


def still_qualifies(manifest: Mapping[str, Any], n_pages: int) -> tuple[bool, str]:
    """Whether the stored layout still qualifies at ``n_pages`` under the same rule (FINAL_DESIGN 2 (e))."""
    row = row_from_dict(manifest.get("layout"))
    if row is None or row.chain == "float":
        return True, ""
    try:
        sel = _select(manifest.get("model"), float(manifest.get("target", DEFAULT_TARGET)), n_pages,
                      checked=bool(manifest.get("checked")), modality=manifest.get("modality"),
                      vectors_per_page=None, layout=None)
    except Exception:  # noqa: BLE001 - the rule cannot say (its catalogue is absent): no claim either way
        return True, ""
    status = next((r.status for r in sel.rows if r.name == row.name), None)
    if status in ("pick", "qualifies", "close call"):
        return True, ""
    return False, f"{row.name} no longer qualifies at this size; index.check().apply() re-encodes"


def _sync_block(ctx: _Ctx, n_after: int, n_before: int) -> Block:
    s = ctx.sync
    lines = list(ctx.notices)
    if not (s.added or s.changed or s.missing):
        lines.append(Line("Sync", "nothing new: {0} pages, unchanged", claim(n_after, "counted")))
    else:
        lines.append(Line("Sync", "{0} files added, {1} changed, {2} missing", claim(len(s.added), "counted"),
                          claim(len(s.changed), "counted"), claim(len(s.missing), "counted")))
        if s.missing:
            lines.append(Line("Missing", "{0}", lit("removed (prune=True)" if ctx.opts.prune else
                                                    "kept in the index; prune=True removes them")))
        lines.append(Line("Pages", "{0} stored now, {1} before", claim(n_after, "counted"),
                          claim(n_before, "counted")))
        if s.aliases:
            lines.append(Line("Identical", "{0} new pages are identical to stored ones; stored once",
                              claim(s.aliases, "counted")))
    if s.skipped:
        lines.append(Line("Skipped", "{0} files could not be read, see report()", claim(len(s.skipped), "counted")))
    if n_before and n_after > n_before * 1.02 and ctx.manifest.get("checked"):
        grew = (n_after - n_before) / n_before
        lines.append(Line("Practice", "the collection grew by {0}; index.check() re-runs the practice check",
                          claim(grew, "counted", text=_claim.pct(grew))))
    if not s.layout_still_qualifies:
        row = row_from_dict(ctx.manifest.get("layout"))
        lines.append(Line("Layout", "{0} no longer qualifies at {1} pages; nothing changed by itself",
                          lit(row.plain if row else "the layout"), claim(n_after, "counted")))
        est = _estimate(ctx.manifest.get("model"), n_after)
        if est is not None:
            lines.append(Line("Re-encode", "{0} with index.check().apply()",
                              claim((est.lo_s, est.hi_s), "estimate", source=str(est.basis),
                                    text=_claim.duration(est.lo_s, est.hi_s))))
    s.message = s.message or "\n".join(ln.render(11) for ln in lines)
    return Block(lines, next="index.search(...) as before" if s.layout_still_qualifies else "index.check()",
                 title="Sync")


# ============================================================================ rebuild


def _rebuild(source: Any, ctx: _Ctx) -> BuildResult:
    """``rebuild=True``: a new index beside the old one, swapped in when complete. With the same model,
    a float16 copy kept in the old index is reused, so nothing is encoded again."""
    old = ctx.path
    new_path = old.with_name(old.name + ".rebuild")
    if new_path.exists():
        rmtree_retry(new_path)
    m = ctx.manifest
    opts = BuildOptions(**{**ctx.opts.__dict__, "path": new_path, "rebuild": False})
    if opts.target is None and opts.max_disk is None:
        opts.target = float(m.get("target", DEFAULT_TARGET))
    if opts.model is None and m.get("model"):
        opts.model = m["model"]
    same_model = (ctx.model_id or m.get("model")) == m.get("model")
    if same_model and m.get("keep_float16") and m.get("source_kind") != "vectors":
        work = work_root(new_path, opts.work_dir, ctx.src.kind)
        if work is not None:
            _seed_from_float16(old, m, work)
    with BuildLock(old):
        result = build(source, opts)
        backup = old.with_name(old.name + ".old")
        if backup.exists():
            rmtree_retry(backup)
        gc.collect()
        os.replace(old, backup)
        os.replace(new_path, old)
        rmtree_retry(backup)
    result.path = old
    return result


def _seed_from_float16(old: Path, manifest: Mapping[str, Any], work: Path) -> None:
    """Write the old index's float16 copy as shards keyed by its files, for the rebuild to reuse. The
    identity matches what the rebuild's encoder will report only when the record is unchanged."""
    store = _store_open(old, "r")
    try:
        corpus = store.float16_corpus()
        if corpus is None:
            return
        rec = dict(manifest.get("record") or {})
        shards = ShardSet(work, {"model": manifest.get("model") or ("adapter" if manifest.get("adapter") else None),
                                 "dtype": rec.get("dtype"), "revision": rec.get("revision"),
                                 "kind": manifest.get("source_kind")})
        aliases, blank = manifest.get("aliases", {}), set(manifest.get("blank", []))
        sha_of = {p: s for s, p in manifest.get("sha_index", {}).items()}
        row_of = {pid: i for i, pid in enumerate(corpus.ids)}
        for doc_id, e in manifest.get("files", {}).items():
            u = Unit(doc_id=doc_id, kind=str(e.get("kind", "pdf")), path=e.get("path"), relpath=e.get("relpath"),
                     ns=str(e.get("ns", "")), size=int(e.get("size", 0)), mtime=int(e.get("mtime", 0)),
                     sha=e.get("sha256"))
            shards.begin(u)
            for pid in e.get("pages", []):
                text = ""
                with contextlib.suppress(Exception):
                    text = store.text(pid) or ""
                page = int(pid.rsplit("::p", 1)[1]) if "::p" in pid else None
                meta = PageMeta(id=pid, doc_id=doc_id, file=e.get("path"), relpath=e.get("relpath"), page=page,
                                text=text, n_chars=len(text), blank=pid in blank, alias_of=aliases.get(pid),
                                sha256=sha_of.get(pid, ""), file_key=u.key)
                if pid in row_of and not meta.blank and meta.alias_of is None:
                    i = row_of[pid]
                    shards.add(meta, np.asarray(corpus.vectors[int(corpus.offsets[i]):int(corpus.offsets[i + 1])]))
                else:
                    shards.add(meta, None)
            shards.end(u, e.get("sha256"))
            shards.maybe_flush()
        shards.flush()
        del corpus
    finally:
        store.close()


# ============================================================================ summary


def _summary(ctx: _Ctx, store: Any, f16_bytes: int) -> Block:
    """The printed summary of a build, in the style of FINAL_DESIGN 4.2; one evidence kind per line."""
    assert ctx.selection is not None and ctx.chosen is not None
    sel, row = ctx.selection, ctx.chosen
    lines: list[Line] = list(ctx.notices)
    lines += _model_lines(ctx)
    lines += _pages_lines(ctx)
    lines += _layout_lines(ctx, sel, row)
    lines += _gate_lines(ctx, sel, row)
    lines += _size_lines(ctx, store, f16_bytes)
    if ctx.search_seconds is not None:
        lines.append(Line("Search", "{0} per query on this machine, timed on {1} sample queries",
                          claim(ctx.search_seconds, "counted", text=seconds_text(ctx.search_seconds)),
                          claim(ctx.timed_queries, "counted")))
    if len(ctx.keep) > int(_ledger.setting("margins", "codecs_only_above", 20000)):
        lines.append(Line("Large", "codecs only at this size; for a search server see index.export_qdrant()"))
    lines.append(Line("Saved", "{0}    reload: {1}", lit(ctx.path), lit(f'ov.load("{ctx.path.as_posix()}")')))
    lines.append(Line("Status", "{0}", lit(_ledger.status())))
    if ctx.src.kind == "vectors":
        nxt = "index.search_vectors(query_vectors)"
    else:
        nxt = f"index.search({json.dumps(ctx.examples[0] if ctx.examples else 'your question')})"
    return Block(lines, next=nxt)


def _model_lines(ctx: _Ctx) -> list[Line]:
    label = _model_label(ctx)
    if ctx.src.kind == "vectors":
        return [Line("Model", "{0}; vectors you gave, not encoded here", lit(label))]
    rec = _record(ctx)
    loader = str(getattr(ctx.info, "loader", "") or rec.get("loader", "") or "adapter")
    dtype = rec.get("dtype")
    ours = _recorded_dtype(ctx.model_id)
    if dtype is None:
        note = "loader default dtype"
    elif ctx.model_id is None:
        note = str(dtype)
    elif ours is None:
        note = f"{dtype}; our run's dtype was not recorded"
    elif str(ours) == str(dtype):
        note = f"{dtype} as in our run"
    else:
        note = f"{dtype}; dtype differs from our measurement ({ours})"
    device = str(rec.get("device") or "device not recorded")
    lines = [Line("Model", "{0}, {1}, {2}; {3}", lit(label), lit(loader), lit(note),
                  claim(device, "counted", text=device))]
    w = ctx.wiring
    if w is not None:
        if bool(getattr(w, "ok", False)):
            lines.append(Line("Wiring", "{0} pages, each found by its own sentence: {1} of {0}. OK.",
                              claim(int(w.n), "counted"), claim(int(w.hits), "counted")))
        else:
            lines.append(Line("Wiring", "your pages look alike; wiring is fine on our sample"))
    elif ctx.src.kind == "text":
        lines.append(Line("Wiring", "not run for text input; ov.doctor() checks the install"))
    return lines


def _pages_lines(ctx: _Ctx) -> list[Line]:
    keep = ctx.keep
    skipped = _unique(ctx.skipped)
    if ctx.src.kind == "vectors":
        lines = [Line("Pages", "{0} documents of vectors", claim(len(keep), "counted"))]
    elif ctx.src.kind == "text":
        lines = [Line("Pages", "{0} texts", claim(len(keep), "counted"))]
    else:
        lines = [Line("Pages", "{0} from {1} files; {2} have a text layer; {3} files skipped, see report()",
                      claim(len(keep), "counted"), claim(len({m.doc_id for m in keep}), "counted"),
                      claim(sum(1 for m in keep if m.n_chars >= 20), "counted"), claim(len(skipped), "counted"))]
    if ctx.blank or ctx.aliases:
        lines.append(Line("Repeats", "{0} blank pages skipped; {1} identical pages stored once",
                          claim(len(ctx.blank), "counted"), claim(len(ctx.aliases), "counted")))
    for path, reason in skipped[:5]:
        lines.append(Line("Skipped", "{0}: {1}", lit(Path(path).name), lit(reason)))
    return lines


def _layout_lines(ctx: _Ctx, sel: Selection, row: LayoutRow) -> list[Line]:
    t = lit(target_text(ctx.target))
    lines: list[Line] = []
    kept = None if row.lowest_kept is None else claim(row.lowest_kept, "catalogue", source=catalogue_source(sel, row),
                                                      construct="nDCG@5 retention", text=_claim.pct(row.lowest_kept))
    if sel.hand_picked:
        lines.append(Line("Layout", "chosen by you: {0}. Lowest kept on our benchmarks {1}", lit(row.plain), kept)
                     if kept is not None else Line("Layout", "chosen by you: {0}; no rows of ours for it",
                                                   lit(row.plain)))
    elif kept is not None:
        lines.append(Line("Layout", "target {0}: {1}. Lowest kept on our benchmarks {2}", t, lit(row.plain), kept))
    else:
        lines.append(Line("Layout", "target {0}: {1}; no rows of ours for it", t, lit(row.plain)))
    if row.name != sel.pick.name:
        lines.append(Line("", "the catalogue pick was {0}; the checks below chose the safer layout",
                          lit(sel.pick.plain)))
    elif sel.close_call and sel.safer is not None:
        safer = sel.safer
        if safer.ratio_f16 is not None:
            lines.append(Line("Close call", "cleared {0} by under half a point; kept as the default. Safer: {1}, "
                                            "{2} smaller than float16", t, lit(f'layout="{safer.name}", rebuild=True'),
                              claim(safer.ratio_f16, "catalogue", source=catalogue_source(sel, safer),
                                    text=_claim.ratio(safer.ratio_f16))))
        else:
            lines.append(Line("Close call", "cleared {0} by under half a point; kept as the default. Safer: {1}", t,
                              lit(f'layout="{safer.name}", rebuild=True')))
        est = _estimate(ctx.model_id, len(ctx.keep))
        if ctx.opts.keep_float16:
            lines.append(Line("", "the float16 copy is kept, so that rebuild encodes nothing again"))
        elif est is not None:
            lines.append(Line("", "that rebuild re-encodes in {0}",
                              claim((est.lo_s, est.hi_s), "estimate", source=str(est.basis),
                                    text=_claim.duration(est.lo_s, est.hi_s))))
    if sel.basis.startswith("unmeasured"):
        lines.append(Line("Basis", "{0}. Your model was not measured; results have not been shown to carry over "
                                   "to new models.", lit(sel.basis)))
    elif sel.basis and sel.basis != "own rows":
        lines.append(Line("Basis", "{0}", lit(sel.basis)))
    if ctx.target > 0.99:
        lines.append(Line("Note", "our benchmarks cannot separate layouts this close to full precision"))
    lines += [Line("Note", "{0}", lit(n)) for n in sel.notes]
    return lines


def _plain_of(sel: Selection, name: Any) -> str:
    return next((r.plain for r in sel.rows if r.name == name), str(name))


def _gate_lines(ctx: _Ctx, sel: Selection, row: LayoutRow) -> list[Line]:
    lines: list[Line] = []
    t = lit(target_text(ctx.target))
    p = ctx.practice
    n_hard = int(p.get("n_hard", 0) or 0)
    if p.get("ran"):
        if not p.get("fired"):
            lines.append(Line("Practice", "passed: {0} hard questions made from {1}", claim(n_hard, "probe"),
                              lit("your texts" if ctx.src.kind == "text" else "your PDFs' text")))
        elif p.get("mode") == "step-down":
            lines.append(Line("Practice", "{0} fell below {1} on {2} hard questions from your PDFs; we used the "
                                          "safer {3}", lit(_plain_of(sel, p.get("tested"))), t, claim(n_hard, "probe"),
                              lit(row.plain)))
            int8_failed = any(not s["passed"] and s["layout"].startswith("int8") for s in p.get("steps", []))
            if row.chain == "float" and int8_failed:
                lines.append(Line("", "the model or the questions may be at fault; see report() and ov.doctor()"))
        else:
            lines.append(Line("Practice", "{0} fell below {1} on {2} hard questions from your PDFs; kept, because "
                                          "the practice check only warns for now",
                              lit(_plain_of(sel, p.get("tested"))), t, claim(n_hard, "probe")))
        strength = _strength(n_hard, ctx.target)
        if strength:
            lines.append(Line("Strength", "at {0} questions a simulated fall to {1} of full-precision quality is "
                                          "caught in {2} of runs; whether real losses show up in practice questions "
                                          "is untested (V5)",
                              claim(n_hard, "estimate", source="simulation"),
                              claim(strength["fall"], "estimate", source="simulation",
                                    text=_claim.pct(strength["fall"], 0)),
                              claim(strength["caught"], "estimate", source="simulation",
                                    text=_claim.pct(strength["caught"], 0))))
    elif n_hard:
        lines.append(Line("Practice", "{0} hard questions made from your PDFs; too few for the check",
                          claim(n_hard, "probe")))
    if not ctx.checked:
        lines.append(Line("Unchecked", "{0}; 1-bit layouts are not used", lit(ctx.unchecked_reason or "no check ran")))
    g = ctx.queries_gate
    if g.get("n") is not None:
        n = int(g["n"])
        if not g.get("ran"):
            lines.append(Line("Questions", "{0} given, fewer than the check needs ({1}); not used",
                              claim(n, "counted"), lit("len(queries) >= 60")))
        elif g.get("fired"):
            lines.append(Line("Questions", "{0} fell below {1} on {2} of your questions; we used the safer {3}",
                              lit(_plain_of(sel, g.get("tested"))), t,
                              claim(n, "user-queries", source=str(g.get("kind"))), lit(row.plain)))
        else:
            lines.append(Line("Questions", "passed: {0} of your questions",
                              claim(n, "user-queries", source=str(g.get("kind")))))
    missing = []
    if not g.get("ran"):
        missing.append("your real questions (queries=)")
    if not ctx.checked and ctx.src.kind != "vectors":
        missing.append("your pages (no practice check)")
    if missing:
        lines.append(Line("Not checked", "{0}", lit("; ".join(missing))))
    return lines


def _size_lines(ctx: _Ctx, store: Any, f16_bytes: int) -> list[Line]:
    sizes = dict(store.sizes())
    total = int(sizes.get("total", 0))
    vectors = int(sizes.get("vectors", 0))
    f16_kept = int(sizes.get("float16", 0))
    other = max(total - vectors - f16_kept, 0)
    lines = [Line("Size", "{0} on disk in total: vectors {1}, page text, build record and the rest {2}",
                  claim(total, "counted", text=_claim.size(total)),
                  claim(vectors, "counted", text=_claim.size(vectors)),
                  claim(other, "counted", text=_claim.size(other)))]
    if vectors and f16_bytes:
        lines.append(Line("Vectors", "{0} smaller than the float16 vectors ({1})",
                          claim(f16_bytes / vectors, "counted", text=_claim.ratio(f16_bytes / vectors)),
                          claim(f16_bytes, "counted", text=_claim.size(f16_bytes))))
    if ctx.opts.keep_float16:
        lines.append(Line("Kept", "the float16 copy, {0} of the total, for step-downs and upgrade()",
                          claim(f16_kept, "counted", text=_claim.size(f16_kept))))
    return lines
