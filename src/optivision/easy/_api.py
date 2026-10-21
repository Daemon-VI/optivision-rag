"""``build``, ``load`` and ``Index``: the easy-mode lifecycle (FINAL_DESIGN 4.2-4.4).

    index = ov.build("my_pdfs/", model="vidore/colqwen2-v1.0-hf", target=0.97)
    hits = index.search("What drove operating costs in 2023?", k=5)
    index = ov.load("my_pdfs.ovx")                 # later; the model loads when a string query needs it
    index.add("more_pdfs/")                        # same as re-running build: new and changed files only

The pipeline itself is in ``_build``; this module holds the public signatures and the ``Index``
object. Every number an ``Index`` prints goes through ``_claim``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import _build, _claim, _ledger
from ._claim import Block, Claim, Line, claim, lit
from ._types import EasyError, Hit, LayoutRow, Selection, SyncReport

__all__ = ["Index", "build", "load"]

#: the per-segment block budget of the exact search (CONTRACT ``_index``)
MAX_BLOCK_BYTES = 32 * 2**20


def build(
    source: Any,
    model: Any = None,
    target: float | None = None,
    *,
    path: str | Path | None = None,
    ids: Sequence[str] | None = None,
    layout: str | None = None,
    keep_float16: bool = False,
    keep_text: bool = True,
    work_dir: str | Path | None = None,
    prune: bool = False,
    rebuild: bool = False,
    trust_remote_code: bool = False,
    device: str = "auto",
    dtype: str = "recorded",
    progress: bool = True,
    accuracy: float | None = None,
    sample: int | None = None,
    probes: str | None = "text",
    queries: Any = None,
    answers: Any = None,
    max_disk: Any = None,
    allow_upload: bool = False,
) -> Index:
    """Build a compressed, searchable index of ``source``, or sync the one at ``path``.

    Args:
        source: a folder, a PDF, an image, a list of strings (text models), a list of ``(n_i, d)``
            arrays or tensors, a padded ``(n, L, d)`` batch, or a ``MultiVectorCorpus``.
        model: a registry id or an ``EncoderAdapter`` (an unregistered adapter gets the rule for
            models without rows of their own). Optional for vectors.
        target: share of full-precision search quality (nDCG@5) to keep, in [0.90, 1.0]; 0.97 for a
            new index, the stored one for an existing index. ``accuracy=`` is an alias.
        path: the index folder; default ``<folder>.ovx`` beside the folder. An existing index syncs:
            only new and changed files are encoded. A different target, layout or model raises
            ``TargetMismatch`` unless ``rebuild=True``.
        ids: names of strings or vector documents (default ``"0"`` .. ``"n-1"``).
        layout: a hand pick ("centred 4-bit" or a row label such as "int4(mean)"); every check still runs.
        keep_float16: keep the float16 copy in the index, for step-downs and ``upgrade()``.
        keep_text: keep the page text (zlib) for ``Hit.text``; counted in the size.
        work_dir: the temporary float16 shards (default ``/content/.ov_work`` on Colab, else
            ``<path>.work``); on Google Drive a build resumes in a new session.
        prune: remove files that disappeared from the source (they are kept otherwise).
        trust_remote_code: needed by models that run their repository's own code.
        device, dtype: where and how the model runs; ``"recorded"`` is our run's dtype.
        progress: print the plan, progress and summary.
        sample: a trial on about this many pages, whose shards the full build reuses.
        probes: practice questions: ``"text"`` (from the text layer, default), ``"vlm"``, ``"api"``
            (needs ``allow_upload=True``) or None.
        queries, answers: your real questions (60 or more), with optional answers (page ids,
            ``(file, page)`` pairs, or sets of them); they can only make the layout safer.
        max_disk: bytes or a size such as ``"2 GB"``: the highest target, in 0.01 steps, that fits.

    Raises:
        UnknownModel, AmbiguousModel, RemoteCodeRefused, MissingExtra, WiringError, NotEnoughDisk,
        TargetMismatch, IndexLocked; ``ValueError`` for a target outside [0.90, 1.0].
    """
    out = _build._Out(progress)
    t = _build.check_target(target, accuracy, out)
    opts = _build.BuildOptions(
        model=model, target=t, path=None if path is None else Path(path), ids=ids, layout=layout,
        keep_float16=keep_float16, keep_text=keep_text, work_dir=work_dir, prune=prune, rebuild=rebuild,
        trust_remote_code=trust_remote_code, device=device, dtype=dtype, progress=progress, sample=sample,
        probes=probes, queries=queries, answers=answers, max_disk=max_disk, allow_upload=allow_upload)
    result = _build.build(source, opts)
    index = Index(result.path, device=device, encoder=result.encoder, adapter=result.adapter, progress=progress)
    index.last_sync = result.sync
    if result.summary is not None:
        index.summary = result.summary
    return index


def load(path: str | Path, *, device: str = "auto", log_queries: bool = False) -> Index:
    """Open an index. The model loads when a string query needs it, at the recorded revision and dtype;
    any difference from the build record is named then. ``log_queries=True`` keeps a local query log
    (never on by default)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    return Index(p, device=device, log_queries=log_queries)


def _doc_of(page_id: str) -> str:
    return page_id.rsplit("::p", 1)[0] if "::p" in page_id else page_id


def _page_of(page_id: str) -> int | None:
    if "::p" not in page_id:
        return None
    tail = page_id.rsplit("::p", 1)[1]
    return int(tail) if tail.isdigit() else None


def _block_from_dict(d: Mapping[str, Any] | None) -> Block | None:
    """A stored summary as a Block again: each line keeps its text and its one evidence kind."""
    if not d:
        return None
    lines = []
    for ln in d.get("lines", []):
        claims = ln.get("claims") or []
        text = str(ln.get("text", ""))
        if claims:
            c = claims[0]
            part: Claim | Any = claim(c.get("value"), c.get("kind", "counted"), c.get("source", ""),
                                      c.get("construct", ""), text=text)
        else:
            part = lit(text)
        lines.append(Line(str(ln.get("label", "")), "{0}", part))
    return Block(lines, next=d.get("next"), title=d.get("title"))


def _as_queries(q: Any) -> Any:
    """Query vectors: a ``MultiVectorCorpus``, a list of ``(n_i, d)`` arrays, one ``(n, d)`` array (one
    query) or a padded ``(n, L, d)`` batch (all-zero rows dropped)."""
    import numpy as np

    from ..representation import MultiVectorCorpus

    if isinstance(q, MultiVectorCorpus):
        return q
    src = _build.vectors_source(q, None)
    arrays = [u.payload for u in src.units]
    if any(a.shape[0] == 0 for a in arrays):
        raise ValueError("a query has no vectors")
    return MultiVectorCorpus.from_arrays([np.asarray(a) for a in arrays], ids=[f"q{i}" for i in range(len(arrays))])


class Index:
    """A built index: search it, sync it, check it, explain it (FINAL_DESIGN 4.4).

    Attributes: ``target``, ``layout`` (``LayoutRow``), ``size`` (bytes on disk), ``n_pages``,
    ``examples`` (practice questions the check did not use), ``evidence`` (every claim of the
    summary, with its tag), ``pipeline`` (the fitted ``Pipeline``), ``retriever`` (the exact
    searcher), ``path``, ``selection`` (``Selection``), ``record`` (the build record),
    ``summary`` (the build summary ``Block``), ``last_sync`` (``SyncReport`` of the last build or add).

    For other easy-mode modules: ``index._store``, ``index._searcher``, ``index._encoder()`` (loads
    the model at the recorded revision and dtype), ``index._manifest`` and ``index._reload()``
    (re-open after the store was written).
    """

    def __init__(self, path: str | Path, *, device: str = "auto", log_queries: bool = False, encoder: Any = None,
                 adapter: Any = None, progress: bool = True) -> None:
        self.path = Path(path)
        self._device = device
        self._encoder_obj = encoder
        self._adapter = adapter
        self._progress = progress
        self._searcher_obj: Any = None
        self._store: Any = _build._store_open(self.path, "r")
        self._closed = False
        self._mcache: dict[str, Any] = dict(self._store.manifest)
        self.record_differences: list[str] = []
        self.last_sync: SyncReport | None = None
        m = self._manifest
        if m.get("kind") not in (None, _build.MANIFEST_KIND):
            self._store.close()
            raise ValueError(f"{self.path} is not an optivision easy index (kind {m.get('kind')!r})")
        self.summary: Block | None = _block_from_dict(m.get("summary"))
        self._log: Any = None
        if log_queries:
            self._log = _build.open_query_log(self)
            self._say(Block([Line("Query log", "on: your searches are kept in {0}, on this machine only",
                                  lit(self.path / "log"))], next="index.search(...)"))

    # ------------------------------------------------------------ internals

    @property
    def _manifest(self) -> dict[str, Any]:
        """The store manifest as read at open (``_reload()`` reads it again)."""
        self._check_open()
        return self._mcache

    @property
    def _searcher(self) -> Any:
        self._check_open()
        if self._searcher_obj is None:
            self._searcher_obj = _build._searcher(self._store)
        return self._searcher_obj

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError(f"the index {self.path} is closed")

    def _say(self, block: Block) -> None:
        """What the user asked for (explain, relink) or must see (record differences, the query-log
        notice) prints whatever ``progress`` was at build time."""
        _build._print(str(block))

    def _encoder(self) -> Any:
        """The encoder, loaded on first use at the recorded revision and dtype; differences from the build
        record are named (FINAL_DESIGN 4.3)."""
        if self._encoder_obj is not None:
            return self._encoder_obj
        m = self._manifest
        model = self._adapter if self._adapter is not None else m.get("model")
        if model is None:
            if m.get("adapter"):
                raise EasyError("this index was built with your own adapter object: encode queries with it and call "
                                "search_vectors(), or pass it to add(..) through build(.., model=adapter)")
            raise EasyError("this index was built from vectors without a model: use search_vectors(), or rebuild "
                            "with model= to search with text")
        rec = dict(m.get("record") or {})
        enc = _build._load_encoder(model, device=self._device, dtype=str(rec.get("dtype") or "recorded"),
                                   trust_remote_code=bool(m.get("trust_remote_code")))
        now = getattr(enc, "record", None)
        self.record_differences = _build.record_differences(rec, now if isinstance(now, Mapping) else {})
        if self.record_differences:
            self._say(Block([Line("Record", "{0}", lit(d)) for d in self.record_differences],
                            next="index.check() compares the build record and re-tests",
                            title="The model now differs from the build record"))
        self._encoder_obj = enc
        return enc

    def _reload(self) -> None:
        """Re-open the store after another module wrote it (``check().apply()``, ``upgrade(apply=True)``)."""
        self._release()
        self._reopen()
        self.summary = _block_from_dict(self._manifest.get("summary")) or self.summary

    def _reopen(self) -> None:
        self._store = _build._store_open(self.path, "r")
        self._closed = False
        self._mcache = dict(self._store.manifest)

    def _release(self) -> None:
        if self._searcher_obj is not None:
            close = getattr(self._searcher_obj, "close", None)
            if close is not None:
                close()
            self._searcher_obj = None
        if self._store is not None and not self._closed:
            self._store.close()
        self._closed = True

    def _hit(self, page_id: str, score: float) -> Hit:
        m = self._files_cache()
        file = None
        page = None
        if self._source_kind == "files":
            entry = m.get(_doc_of(page_id))
            file = None if entry is None else entry.get("path")
            page = _page_of(page_id)
        text = ""
        if self._keep_text:
            text = self._store.text(page_id) or ""
        return Hit(id=page_id, file=file, page=page, score=float(score), text=text, _index=self)

    def _files_cache(self) -> dict[str, Any]:
        return self._manifest.get("files", {})

    @property
    def _source_kind(self) -> str:
        return str(self._manifest.get("source_kind", "files"))

    @property
    def _keep_text(self) -> bool:
        return bool(self._manifest.get("keep_text", True))

    # ------------------------------------------------------------ attributes

    @property
    def target(self) -> float:
        return float(self._manifest.get("target", _build.DEFAULT_TARGET))

    @property
    def layout(self) -> LayoutRow:
        row = _build.row_from_dict(self._manifest.get("layout"))
        assert row is not None
        return row

    @property
    def selection(self) -> Selection | None:
        return _build.selection_from_dict(self._manifest.get("selection"))

    @property
    def size(self) -> int:
        self._check_open()
        return int(dict(self._store.sizes()).get("total", 0))

    @property
    def n_pages(self) -> int:
        return int(self._manifest.get("n_pages", 0))

    @property
    def examples(self) -> list[str]:
        return list(self._manifest.get("examples", []))

    @property
    def evidence(self) -> list[dict[str, Any]]:
        return list(self._manifest.get("evidence", []))

    @property
    def record(self) -> dict[str, Any]:
        return dict(self._manifest.get("record") or {})

    @property
    def pipeline(self) -> Any:
        self._check_open()
        return self._store.pipeline()

    @property
    def retriever(self) -> Any:
        return self._searcher

    # ------------------------------------------------------------ search

    def search(self, query: str | Sequence[str], k: int = 5) -> list[Hit] | list[list[Hit]]:
        """Encode the question(s) and scan the stored codes exactly: ``list[Hit]`` for a string,
        ``list[list[Hit]]`` for a list of strings."""
        single = isinstance(query, str)
        texts = [query] if single else list(query)
        if not texts or not all(isinstance(t, str) for t in texts):
            raise TypeError("search() takes a string or a list of strings; use search_vectors() for vectors")
        q = _build._encode_queries(self._encoder(), texts)
        results = self.search_vectors(q, k)
        if self._log is not None:
            for text, hits in zip(texts, results):
                self._log.record(text, hits)
        return results[0] if single else results

    def search_vectors(self, q: Any, k: int = 5) -> list[list[Hit]]:
        """Pre-encoded queries (a ``MultiVectorCorpus``, a list of ``(n_i, d)`` arrays, one ``(n, d)`` array)."""
        if int(k) < 1:
            raise ValueError("k must be at least 1")
        queries = _as_queries(q)
        raw = self._searcher.search(queries, int(k))
        return [[self._hit(str(i), float(s)) for i, s in hits] for hits in raw]

    # ------------------------------------------------------------ lifecycle

    def add(self, source: Any, ids: Sequence[str] | None = None, *, prune: bool = False) -> SyncReport:
        """Add or update documents, as re-running ``build`` on this index does: new and changed files go
        through the frozen fitted pipeline; missing files are kept unless ``prune=True``."""
        self._check_open()
        m = self._manifest
        model = self._adapter if self._adapter is not None else m.get("model")
        opts = _build.BuildOptions(model=model, path=self.path, ids=ids, prune=prune, progress=self._progress,
                                   device=self._device, trust_remote_code=bool(m.get("trust_remote_code")),
                                   keep_text=bool(m.get("keep_text", True)),
                                   keep_float16=bool(m.get("keep_float16", False)), encoder=self._encoder_obj)
        self._release()
        try:
            result = _build.build(source, opts)
        finally:
            self._reopen()
        if result.encoder is not None:
            self._encoder_obj = result.encoder
        self.last_sync = result.sync
        return result.sync if result.sync is not None else SyncReport()

    def check(self, queries: Any = None, *, answers: Any = None, drift: str = "auto") -> Any:
        """Re-test the index without changing it (FINAL_DESIGN 5.5.3): build record, drift, size, practice
        re-run; ``queries=`` for agreement or, with ``answers=``, search quality. ``.apply()`` acts."""
        self._check_open()
        return _build.run_check(self, queries, answers=answers, drift=drift)

    def upgrade(self, queries: Any, *, answers: Any = None, apply: bool = False) -> Any:
        """The only route to more compression: your real questions through the unchanged ``certify()``."""
        self._check_open()
        _ledger.require("upgrade")
        return _build.run_upgrade(self, queries, answers=answers, apply=apply)

    def report(self) -> Any:
        """Every number with its tag: skipped files, close calls, the ladder at this target, the gates,
        float32 ratios, sizes, the build record; ``.save(path)`` writes .html, .md or .json."""
        self._check_open()
        return _build.make_report(self, self.report_sections, self.report_meta)

    def report_sections(self) -> list[tuple[str, Block]]:
        """The report's sections as ``(title, Block)`` (used when ``_report`` builds from sections)."""
        m = self._manifest
        sections: list[tuple[str, Block]] = []
        if self.summary is not None:
            sections.append(("Summary", self.summary))
        sel = self.selection
        if sel is not None:
            rows = []
            for r in sel.rows:
                if r.lowest_kept is not None:
                    rows.append(Line(r.name, "{0}: {1}; lowest kept {2}", lit(r.plain), lit(r.status),
                                     claim(r.lowest_kept, "catalogue", source=_build.catalogue_source(sel, r),
                                           construct="nDCG@5 retention",
                                           text=_claim.pct(r.lowest_kept))))
                else:
                    rows.append(Line(r.name, "{0}: {1}", lit(r.plain), lit(r.status)))
                if r.ratio_f16 is not None and r.ratio_f32 is not None:
                    rows.append(Line("", "{0} smaller than float16, {1} smaller than float32",
                                     claim(r.ratio_f16, "catalogue", source=_build.catalogue_source(sel, r),
                                           text=_claim.ratio(r.ratio_f16)),
                                     claim(r.ratio_f32, "catalogue", source=_build.catalogue_source(sel, r),
                                           text=_claim.ratio(r.ratio_f32))))
            sections.append(("Layouts at this target", Block(rows, next="index.explain()")))
        gates = []
        for name, key, kind in (("Practice", "practice", "probe"), ("Questions", "queries_gate", "user-queries")):
            g = m.get(key) or {}
            for s in g.get("steps", []):
                src = str(g.get("kind", "")) if kind == "user-queries" else ""
                gates.append(Line(name, "{0}: retention {1}, upper bound {2}, on {3} questions; {4}",
                                  lit(s.get("plain", s.get("layout"))),
                                  claim(s["R"], kind, source=src, text=_claim.pct(s["R"])),
                                  claim(s["U"], kind, source=src, text=_claim.pct(s["U"])),
                                  claim(int(s["n"]), kind, source=src), lit("passed" if s["passed"] else "fell below")))
        if gates:
            sections.append(("Checks", Block(gates, next="index.check()")))
        skipped = [Line("Skipped", "{0}: {1}", lit(p), lit(r)) for p, r in m.get("skipped", [])]
        if skipped:
            sections.append(("Files skipped", Block(skipped, next="fix or remove them, then index.add(folder)")))
        sizes = dict(self._store.sizes())
        sections.append(("Size", Block([Line(k, "{0}", claim(int(v), "counted", text=_claim.size(v)))
                                        for k, v in sizes.items()], next="index.report().save('report.html')")))
        rec = [Line(str(k), "{0}", lit(json.dumps(v, default=str))) for k, v in sorted((m.get("record") or {}).items())]
        if rec:
            sections.append(("Build record", Block(rec, next="index.check()")))
        return sections

    def report_meta(self) -> dict[str, Any]:
        m = self._manifest
        return {"path": str(self.path), "model": m.get("model"), "target": m.get("target"),
                "layout": (m.get("layout") or {}).get("name"), "n_pages": m.get("n_pages"), "built": m.get("built"),
                "updated": m.get("updated"), "status": _ledger.status(), "checked": m.get("checked"),
                "unchecked_reason": m.get("unchecked_reason"), "aliases": len(m.get("aliases", {})),
                "blank": len(m.get("blank", [])), "skipped": m.get("skipped", [])}

    def explain(self) -> str:
        """The expert-API code equivalent to this index (``Pipeline.from_dict``, ``CompressedRetriever``,
        and the ``certify`` call that ``upgrade()`` makes); printed, and returned."""
        code = self._explain_code()
        self._say(Block([Line("", "{0}", lit(code))], next="index.report()", title="Expert code for this index"))
        return code

    def _explain_code(self) -> str:
        m = self._manifest
        row = self.layout
        model = m.get("model")
        kind = self._source_kind
        out = ["import numpy as np",
               "from optivision.compose import Pipeline",
               "from optivision.core.retrieve.exact import CompressedRetriever",
               "from optivision.representation import MultiVectorCorpus", ""]
        if kind == "vectors":
            out += ["pages = MultiVectorCorpus.from_arrays(page_vectors, ids=page_ids).astype(np.float16)"]
        else:
            what = "page_images" if kind == "files" else "texts"
            if model:
                out += ["from optivision.adapters import load_adapter", f"adapter = load_adapter({model!r})"]
            else:
                out += ["adapter = your_adapter   # the EncoderAdapter this index was built with"]
            out += [(f"pages = adapter.encode_documents({what}, ids=page_ids).astype(np.float16)"
                     "   # ids such as 'folder/file.pdf::p1'")]
        out += [f"pipeline = Pipeline.from_dict({row.spec!r})   # {row.name}: {row.plain}",
                (f"pipeline.fit(pages)   # the build fits on a seeded page sample: at most {_build.FIT_ROWS} "
                 f"vectors, seed {_build.FIT_SEED}"),
                (f"store = pipeline.compress(pages, fit=False)   # the build settles {_build.SEGMENT_PAGES} pages "
                 "at a time"),
                f"retriever = CompressedRetriever(store, pipeline, max_block_bytes={MAX_BLOCK_BYTES})"]
        if kind == "vectors":
            out += ["ranking = retriever.search(query_vectors, k=5)"]
        else:
            out += ['query_vectors = adapter.encode_queries(["your question"])',
                    "ranking = retriever.search(query_vectors, k=5)"]
        families = self._upgrade_families(row)
        attempts = int(m.get("upgrade_attempts", 0))
        out += ["", "# more compression comes only from your real questions (index.upgrade), through certify():"]
        if not families:
            out += ["# no larger layout is open to upgrade() at this size, so it has nothing to try"]
            return "\n".join(out)
        out += ["from optivision.api import certify",
                "from optivision.core.evaluate.reference import RankingReference   # with answers: LabelReference",
                "from optivision.core.selection.spaces import CandidateSpace",
                "space = CandidateSpace.from_families({"]
        for chain, specs in families.items():
            items = ", ".join(f"Pipeline.from_dict({s!r})" for s in specs)
            out.append(f"    {chain!r}: [{items}],")
        out += ["})",
                f"result = certify(pages, query_vectors, None, target={self.target}, error=0.05 * 2**-{attempts},",
                "                 reference=RankingReference(depth=1), space=space, fallback=pipeline, order_seed=0)"]
        return "\n".join(out)

    def _upgrade_families(self, row: LayoutRow) -> dict[str, list[dict[str, Any]]]:
        sel = self.selection
        out: dict[str, list[dict[str, Any]]] = {}
        if sel is None or row.ratio_f16 is None:
            return out
        for r in sel.rows:
            if r.ratio_f16 is None or r.ratio_f16 <= row.ratio_f16 or r.chain == "float":
                continue
            if r.status in ("not measured", "outside size range", "not allowed"):
                continue
            out.setdefault(r.chain, []).append(r.spec)
        return out

    def relink(self, folder: str | Path) -> dict[str, str]:
        """Find files that moved by their sha256 under ``folder`` and record their new paths; returns
        ``{doc_id: new path}``."""
        self._check_open()
        m = self._manifest
        root = Path(folder).expanduser().resolve()
        missing = {d: e for d, e in m.get("files", {}).items()
                   if e.get("path") and not Path(e["path"]).exists() and e.get("sha256")}
        found: dict[str, str] = {}
        if missing and root.is_dir():
            by_size: dict[int, list[Path]] = {}
            for p in root.rglob("*"):
                if p.is_file():
                    by_size.setdefault(p.stat().st_size, []).append(p)
            for doc_id, e in missing.items():
                for cand in by_size.get(int(e.get("size", -1)), []):
                    if _build.file_sha256(cand) == e["sha256"]:
                        found[doc_id] = str(cand.resolve())
                        break
        if found:
            files = {d: dict(e) for d, e in m.get("files", {}).items()}
            for doc_id, p in found.items():
                st = Path(p).stat()
                files[doc_id].update(path=p, size=int(st.st_size), mtime=int(st.st_mtime_ns))
            self._release()
            try:
                with _build.BuildLock(self.path):
                    store = _build._store_open(self.path, "a")
                    try:
                        with _build._store_lock(store):
                            store.commit(files=files, updated=_build._now())
                    finally:
                        store.close()
            finally:
                self._reopen()
        self._say(Block([Line("Relinked", "{0} of {1} missing files found by content in {2}",
                              claim(len(found), "counted"), claim(len(missing), "counted"), lit(root))],
                        next="index.search(...); hits now render from the new paths"))
        return found

    def demo(self) -> Any:
        """A local Gradio search box (extra ``easy-ui``); it never logs searches."""
        self._check_open()
        return _build.launch_demo(self)

    def export_qdrant(self, **kwargs: Any) -> Any:
        """Export to Qdrant (switched off until its top-10 equality test passes)."""
        self._check_open()
        _ledger.require("qdrant_export")
        return _build.export_qdrant(self, **kwargs)

    def close(self) -> None:
        """Close the memory maps (on Windows files are deleted only after this)."""
        if not self._closed:
            self._release()
        self._encoder_obj = None if self._adapter is None else self._encoder_obj

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._closed:
            return f"Index({str(self.path)!r}, closed)"
        line = Line("", "Index at {0}: {1} pages, layout {2}", lit(self.path), claim(self.n_pages, "counted"),
                    lit(self.layout.plain))
        return line.render()

    def __str__(self) -> str:
        return str(self.summary) if self.summary is not None else repr(self)
