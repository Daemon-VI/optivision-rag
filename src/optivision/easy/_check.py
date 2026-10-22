"""``index.check()``: is the index still what it was built to be? (FINAL_DESIGN 4.4, 5.5.3)

Six steps, in the order of FINAL_DESIGN 5.5.3:

1. **Build record.** Model, resolved commit, dtype, attention implementation, ``dpi`` /
   ``max_side``, batch size and library versions, compared with the record saved at build.
   Without loading the model only the library versions can be compared; the model's own
   settings are compared when the drift check loads it.
2. **Drift.** Four fixed pages are re-encoded and scored against their stored codes; every
   score must stay within 1% (relative) of the score saved at build (an engineering
   threshold). It needs a GPU unless ``drift=True``, and prints its time first. Otherwise
   the report says "drift not checked (no GPU)".
3. **Size.** Does the layout still qualify at today's page count (``_levels.select``)?
4. **Practice rerun.** Exact, from the stored codes and ``reference/``; it prints how many
   times the practice check has run on this index, because every rerun is another look.
5. **Growth.** Above twice the build size it advises a refit and gives the growth figures of
   our two size tests ``[our benchmarks]``.
6. **Your questions.** ``queries=`` gives exact agreement with the float16 copy when the index
   keeps one (``keep_float16=True``); without it the report says so and names the spot check
   (``spot_check=True``, :mod:`._spotcheck`), which can only flag. ``answers=`` gives search
   quality (nDCG@5); retention against full precision also needs the float16 copy.

``CheckReport.ok`` holds when the build record matches, there is no drift, the practice check
is quiet and the layout still qualifies at today's size; when questions were given, also when
they did not fall below the target. Steps that could not run do not make ``ok`` false;
``complete`` says whether all four ran. ``check()`` changes nothing in the index (it keeps a
count of practice reruns in the manifest when it can write); ``CheckReport.apply()`` makes the
advised change, printing its time first, and never runs on its own.

Agreement with full precision is never called quality: the two have different tags and are
never on one line.

The helpers at the top (``index_*``) are the only places this module, ``_spotcheck`` and
``_upgrade`` touch an index's internals (CONTRACT: ``index._store``, ``index._searcher``,
``index._encoder()``, ``index._manifest``, ``target``, ``layout``, ``selection``, ``record``).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import _claim
from ._claim import Block, Claim, claim, lit
from ._types import EasyError, LayoutRow

#: row statuses under which a layout counts as qualifying (FINAL_DESIGN 4.1)
QUALIFYING = ("pick", "qualifies", "close call")

#: drift: how many fixed pages, and the relative tolerance on their scores (engineering threshold)
DRIFT_PAGES = 4
DRIFT_REL_TOL = 0.01

#: growth: above this many times the build size a refit is advised (FINAL_DESIGN 5.5.3 step 5)
GROWTH_FACTOR = 2.0

#: the growth sentence of FINAL_DESIGN 5.5.3 step 5 (computed there on the ladder rungs of the
#: follow-up units f6/f7): sizes of the two tests, mean losses, worst loss. Move these into
#: presets/easy_levels.json if the generator computes them.
GROWTH_TEST_PAGES = (500, 2500)
GROWTH_MEAN_LOSS = (0.006, 0.009)
GROWTH_WORST_LOSS = 0.028
GROWTH_SOURCE = "follow-up units f6, f7: ViDoRe V1 DocVQA 500 pages, then 2,500 pages with distractors"

#: questions below this count get no bootstrap bound (FINAL_DESIGN 5.5.1, 5.5.2)
MIN_GATE_QUESTIONS = 60

#: build-record fields compared, each with the keys it may be stored under
RECORD_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("model", ("model", "model_id")),
    ("resolved commit", ("resolved_sha", "resolved_commit", "commit_sha", "commit", "revision")),
    ("dtype", ("dtype",)),
    ("attention", ("attn_implementation", "attention_implementation", "attention_impl", "attention")),
    ("dpi", ("dpi",)),
    ("max_side", ("max_side",)),
    ("batch size", ("batch_size", "batch")),
)
VERSION_KEYS = ("versions", "library_versions")

AGREEMENT = "agreement with full precision, not relevance"
QUALITY = "search quality"


# ============================================================================ index access


def index_manifest(index: Any) -> dict[str, Any]:
    m = getattr(index, "_manifest", None)
    if m is None:
        m = getattr(getattr(index, "_store", None), "manifest", None)
    return dict(m or {})


def index_store(index: Any) -> Any:
    store = getattr(index, "_store", None)
    if store is None:
        raise EasyError("this index has no store (it was closed?); reload it with ov.load(path)")
    return store


def index_record(index: Any) -> dict[str, Any]:
    rec = getattr(index, "record", None)
    if rec is None:
        m = index_manifest(index)
        rec = m.get("record") or m.get("build") or {}
    return dict(rec)


def index_selection(index: Any) -> Any:
    return getattr(index, "selection", None)


def index_model(index: Any) -> str | None:
    sel = index_selection(index)
    model = getattr(sel, "model", None)
    if model:
        return str(model)
    rec = index_record(index)
    value = rec.get("model") or rec.get("model_id")
    return str(value) if value else None


def index_target(index: Any) -> float:
    target = getattr(index, "target", None)
    if target is None:
        target = getattr(index_selection(index), "target", None)
    if target is None:
        raise EasyError("this index records no target")
    return float(target)


def index_n_pages(index: Any) -> int:
    n = getattr(index, "n_pages", None)
    if n is None:
        n = index_store(index).n_pages
    return int(n)


def index_built_n_pages(index: Any) -> int | None:
    """The page count the layout was chosen and fitted at (the build or the last refit)."""
    m = index_manifest(index)
    for key in ("n_pages_at_build", "built_n_pages"):
        if m.get(key):
            return int(m[key])
    n = getattr(index_selection(index), "n_pages", None)
    if n:
        return int(n)
    n = index_record(index).get("n_pages")
    return int(n) if n else None


def index_path(index: Any) -> Path | None:
    path = getattr(index, "path", None) or getattr(index_store(index), "path", None)
    return Path(path) if path is not None else None


def index_float16(index: Any) -> Any:
    """The kept float16 copy (a ``MultiVectorCorpus``), or None without ``keep_float16=True``."""
    fn = getattr(index_store(index), "float16_corpus", None)
    return fn() if callable(fn) else None


def index_reference(index: Any) -> Any:
    return getattr(index_store(index), "reference", None)


def index_vectors_per_page(index: Any) -> float | None:
    m = index_manifest(index)
    value = m.get("vectors_per_page") or (m.get("profile") or {}).get("vectors_per_page")
    return float(value) if value else None


def index_commit(index: Any, **updates: Any) -> bool:
    """Write manifest entries under the store's writer lock; False when that is not possible
    (a read-only store, another writer holding the lock). Never raises."""
    try:
        store = index_store(index)
        lock = store.lock() if hasattr(store, "lock") else nullcontext()
        with lock:
            store.commit(**updates)
        manifest = getattr(index, "_manifest", None)
        if isinstance(manifest, dict) and manifest is not getattr(store, "manifest", None):
            manifest.update(updates)
        return True
    except Exception:  # noqa: BLE001 - bookkeeping only; the caller says it was not saved
        return False


def ranking_lists(result: Any) -> tuple[list[list[str]], list[list[float]]]:
    """Ids and scores from a searcher result: ``list[list[(id, score)]]``, ``list[list[Hit]]``
    or a core ``Ranking``."""
    if hasattr(result, "ids") and hasattr(result, "scores") and not isinstance(result, list):
        ids = [list(map(str, row)) for row in result.ids]
        scores = [list(map(float, np.asarray(result.scores)[i, :len(row)])) for i, row in enumerate(ids)]
        return ids, scores
    ids, scores = [], []
    for row in result:
        r_ids, r_sc = [], []
        for item in row:
            if hasattr(item, "id"):
                r_ids.append(str(item.id))
                r_sc.append(float(item.score))
            else:
                r_ids.append(str(item[0]))
                r_sc.append(float(item[1]))
        ids.append(r_ids)
        scores.append(r_sc)
    return ids, scores


def index_rank(index: Any, queries: Any, k: int) -> list[list[str]]:
    """Ids of the stored-code ranking, ``k`` per query (exact top-k over every segment)."""
    ids, _ = ranking_lists(index._searcher.search(queries, k))
    return ids


def as_query_corpus(index: Any, queries: Any) -> tuple[Any, list[str] | None]:
    """Queries as a ``MultiVectorCorpus`` with ids ``q00000``, ...; and their texts when given as
    strings (encoded with the index's model, which loads on first use)."""
    from optivision.representation import MultiVectorCorpus, as_corpus

    if isinstance(queries, str):
        queries = [queries]
    texts = None
    if isinstance(queries, MultiVectorCorpus):
        q = queries
    elif isinstance(queries, Sequence) and queries and all(isinstance(t, str) for t in queries):
        texts = list(queries)
        q = index._encoder().encode_queries(texts)
    else:
        if isinstance(queries, Sequence) and not queries:
            raise ValueError("queries is empty")
        q = as_corpus(queries)
    if len(q) == 0:
        raise ValueError("queries is empty")
    ids = [f"q{i:05d}" for i in range(len(q))]
    return MultiVectorCorpus(q.vectors, q.offsets, ids=ids), texts


def gpu_available() -> bool:
    """True when torch is installed and sees a CUDA device (torch is imported only here)."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken torch install means no usable GPU
        return False


def encode_estimate(index: Any, n_pages: int) -> Claim | None:
    """The encode-time estimate for ``n_pages`` pages of this index's model, or None."""
    model = index_model(index)
    if model is None:
        return None
    try:
        from . import _encode

        est = _encode.estimate(model, int(n_pages))
    except Exception:  # noqa: BLE001 - no rate recorded for this model, or no estimator
        return None
    if est is None:
        return None
    lo, hi = float(est.lo_s), float(est.hi_s)
    return claim((lo, hi), "estimate", source=str(est.basis), construct="encode time",
                 text=_claim.duration(lo, hi))


def page_images(index: Any, pages: Sequence[tuple[str, int]]) -> list[Any]:
    """Render ``(file, page)`` pairs as the build did: the recorded ``dpi`` (150 by default),
    then ``ingest.downscale`` to the recorded ``max_side`` (1536 by default)."""
    from optivision.ingest import downscale

    from . import _docs

    rec = index_record(index)
    dpi = int(rec.get("dpi") or 150)
    max_side = int(rec.get("max_side") or 1536)
    return [downscale(_docs.render_page(file, int(page), dpi=dpi), max_side) for file, page in pages]


def _row_ratio(row: LayoutRow) -> float:
    for value in (row.ratio_f16, row.ratio_f32):
        if value:
            return float(value)
    return 0.0


def find_row(rows: Sequence[LayoutRow], name: str) -> LayoutRow | None:
    return next((r for r in rows if r.name == name), None)


# ============================================================================ build record


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if mapping.get(key) not in (None, ""):
            return mapping[key]
    return None


def installed_version(package: str) -> str:
    """The installed version of a package, read from its metadata (nothing is imported)."""
    name = package.lower()
    if name == "python":
        return platform.python_version()
    if name in ("optivision", "optivision-rag", "optivision_rag"):
        import optivision

        return str(optivision.__version__)
    for candidate in (package, package.replace("_", "-"), package.replace("-", "_")):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not installed"


@dataclass
class RecordCheck:
    """``differences`` holds ``(field, at build, now)``; ``compared`` the fields compared."""

    differences: list[tuple[str, str, str]]
    compared: list[str]
    model_compared: bool

    @property
    def matches(self) -> bool:
        return not self.differences


def compare_record(built: Mapping[str, Any], now: Mapping[str, Any] | None) -> RecordCheck:
    """Compare a build record with the record of the encoder loaded now (``now``), or, when the
    model was not loaded (``now is None``), only the library versions with what is installed."""
    diffs: list[tuple[str, str, str]] = []
    compared: list[str] = []
    if now is not None:
        for label, keys in RECORD_FIELDS:
            b, n = _first(built, keys), _first(now, keys)
            if b is None and n is None:
                continue
            compared.append(label)
            if str(b) != str(n):
                diffs.append((label, str(b), str(n)))
    built_versions = dict(_first(built, VERSION_KEYS) or {})
    now_versions = dict(_first(now, VERSION_KEYS) or {}) if now is not None else {}
    for package in sorted(built_versions):
        b = str(built_versions[package])
        n = str(now_versions[package]) if package in now_versions else installed_version(package)
        compared.append(package)
        if b != n:
            diffs.append((package, b, n))
    return RecordCheck(diffs, compared, now is not None)


# ============================================================================ drift


def pick_drift_pages(ids: Sequence[str], n: int = DRIFT_PAGES) -> list[str]:
    """The fixed drift pages: ``n`` ids evenly spaced over the ids in sorted order."""
    ordered = sorted(map(str, ids))
    if len(ordered) <= n:
        return ordered
    picks = [ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n)]
    return list(dict.fromkeys(picks))


def stored_scores(store: Any, vectors: Any) -> list[float]:
    """For each page of ``vectors``: its vectors, used as one query, scored against that page's
    stored codes (exact MaxSim through ``CompressedRetriever.rescore`` with the fitted pipeline).

    The build calls this on the float16 vectors of the drift pages and saves the scores
    (:func:`make_drift_record`); ``check()`` calls it on the re-encoded vectors."""
    from optivision.core.retrieve.exact import CompressedRetriever

    pipeline = store.pipeline()
    want = {pid: i for i, pid in enumerate(vectors.ids)}
    out: list[float | None] = [None] * len(want)
    for seg in store.segments():
        pos = {pid: j for j, pid in enumerate(seg.ids) if pid in want}
        if not pos:
            continue
        rows = [want[pid] for pid in pos]
        q = vectors.select(rows)
        cand = np.array([[pos[vectors.ids[i]]] for i in rows], dtype=np.int64)
        scores = CompressedRetriever(seg, pipeline).rescore(q, cand)
        for r, i in enumerate(rows):
            out[i] = float(scores[r, 0])
    missing = [pid for pid, i in want.items() if out[i] is None]
    if missing:
        raise KeyError(f"pages not in the index: {missing}")
    return [float(s) for s in out]  # type: ignore[arg-type]


def make_drift_record(store: Any, vectors: Any, sources: Mapping[str, tuple[str, int]] | None) -> dict[str, Any]:
    """What the build saves as ``manifest["drift"]``: the drift pages' ids, source files and page
    numbers, and their :func:`stored_scores` from the float16 vectors. ``vectors`` holds the
    pages of :func:`pick_drift_pages`; ``sources`` maps id -> (file, page), None from vectors."""
    sources = sources or {}
    return {
        "ids": list(vectors.ids),
        "files": [str(sources[i][0]) if i in sources else None for i in vectors.ids],
        "pages": [int(sources[i][1]) if i in sources else None for i in vectors.ids],
        "scores": stored_scores(store, vectors),
        "rel_tol": DRIFT_REL_TOL,
    }


@dataclass
class DriftCheck:
    checked: bool
    reason: str = ""
    drifted: bool = False
    max_change: float | None = None
    changes: dict[str, float] = field(default_factory=dict)
    seconds: float | None = None
    encoder_record: dict[str, Any] | None = None


def _run_drift(index: Any, drift: Any, out: Callable[[str], Any] | None) -> DriftCheck:
    if drift is False:
        return DriftCheck(False, "drift not checked (drift=False)")
    if drift not in ("auto", True):
        raise ValueError(f"drift must be 'auto', True or False, got {drift!r}")
    rec = index_manifest(index).get("drift") or {}
    ids = list(rec.get("ids") or [])
    if not ids or not rec.get("scores"):
        return DriftCheck(False, "drift not checked (this index saved no drift pages)")
    files, pages = list(rec.get("files") or []), list(rec.get("pages") or [])
    if len(files) != len(ids) or any(f is None for f in files):
        return DriftCheck(False, "drift not checked (built from vectors: there are no pages to re-encode)")
    gone = [f for f in files if not Path(f).exists()]
    if gone:
        return DriftCheck(False, f"drift not checked ({gone[0]} is gone; Index.relink(folder) finds moved files)")
    if drift == "auto" and not gpu_available():
        return DriftCheck(False, "drift not checked (no GPU)")
    if out is not None:
        est = encode_estimate(index, len(ids))
        block = Block([], next="the check runs now")
        if est is None:
            block.add("Drift", "about to re-encode the four fixed pages; no time estimate for this model")
        else:
            block.add("Drift", "about to re-encode the four fixed pages: {0}, plus loading the model", est)
        out(str(block))
    from optivision.representation import MultiVectorCorpus

    start = time.perf_counter()
    encoder = index._encoder()
    images = page_images(index, list(zip(files, pages)))
    encoded = encoder.encode_pages(images, ids)
    vectors = MultiVectorCorpus(encoded.vectors, encoded.offsets, ids=ids)
    now = stored_scores(index_store(index), vectors)
    seconds = time.perf_counter() - start
    tol = float(rec.get("rel_tol") or DRIFT_REL_TOL)
    changes = {pid: abs(a - b) / max(abs(b), 1e-12) for pid, a, b in zip(ids, now, rec["scores"])}
    worst = max(changes.values())
    record = getattr(encoder, "record", None)
    return DriftCheck(True, "", worst > tol, worst, changes, seconds, dict(record) if record else None)


# ============================================================================ answers


def _id_tables(ids: Sequence[str]) -> tuple[set[str], dict[str, set[str]]]:
    exact = set(ids)
    suffix: dict[str, set[str]] = {}
    for pid in ids:
        doc, sep, page = pid.rpartition("::p")
        if not sep:
            continue
        parts = doc.split("/")
        for i in range(1, len(parts)):
            suffix.setdefault("/".join(parts[i:]) + sep + page, set()).add(pid)
    return exact, suffix


def _resolve_id(text: str, page: int | None, exact: set[str], suffix: dict[str, set[str]]) -> str:
    raw = str(text).replace("\\", "/")
    if page is not None:
        raw = f"{raw}::p{int(page)}"
    if raw in exact:
        return raw
    doc, sep, pg = raw.rpartition("::p")
    if not sep:
        raise ValueError(f"answer {text!r} is not a page id of this index (ids look like 'folder/file.pdf::p3')")
    parts = [p for p in doc.split("/") if p not in ("", ".")]
    for i in range(len(parts)):
        key = "/".join(parts[i:]) + sep + pg
        if key in exact:
            return key
        found = suffix.get(key)
        if found and len(found) == 1:
            return next(iter(found))
        if found:
            raise ValueError(f"answer {text!r} matches several pages: {sorted(found)[:3]}; give more of the path")
    raise ValueError(f"answer {text!r} is not a page of this index")


def answer_sets(ids: Sequence[str], answers: Sequence[Any], n_queries: int) -> list[set[str] | None]:
    """One set of page ids per query from ``answers`` (FINAL_DESIGN 2(f)): a page id, a
    ``(file, page)`` pair, or a set of either; None for a query without an answer. A page id
    may be shortened to its unique end (``"annual.pdf::p4"``)."""
    if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence):
        raise TypeError("answers must be a list with one entry per query")
    if len(answers) != n_queries:
        raise ValueError(f"answers has {len(answers)} entries for {n_queries} queries; give one per query")
    exact, suffix = _id_tables(list(ids))

    def one(item: Any) -> str:
        if isinstance(item, (str, Path)):
            return _resolve_id(str(item), None, exact, suffix)
        if (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (str, Path))
                and isinstance(item[1], (int, np.integer)) and not isinstance(item[1], bool)):
            return _resolve_id(str(item[0]), int(item[1]), exact, suffix)
        raise TypeError(f"an answer is a page id, a (file, page) pair or a set of them, got {item!r}")

    out: list[set[str] | None] = []
    for entry in answers:
        if entry is None:
            out.append(None)
        elif isinstance(entry, (set, frozenset, list)):
            out.append({one(x) for x in entry} or None)
        else:
            out.append({one(entry)})
    return out


# ============================================================================ questions


@dataclass
class QuestionCheck:
    n: int
    agreement: float | None = None
    agreement_stat: Any = None
    quality: float | None = None
    quality_float16: float | None = None
    retention: float | None = None
    retention_stat: Any = None
    n_answered: int = 0
    needs_float16: bool = False
    fell_below: bool = False


def _bound(cand: dict[str, list[str]], ref: dict[str, list[str]], relevant: dict[str, set[str]],
           target: float) -> Any:
    """The bootstrap bound of FINAL_DESIGN 5.5.1 (``_veto.practice_check``), one cluster per
    question; None below 60 questions or without ``_veto``."""
    if len(cand) < MIN_GATE_QUESTIONS:
        return None
    try:
        from . import _veto
    except ImportError:
        return None
    clusters = {q: q for q in cand}
    return _veto.practice_check(cand, ref, relevant, clusters, target)


def _questions(index: Any, queries: Any, answers: Any, target: float) -> QuestionCheck:
    from optivision.core.retrieve.exact import ExactRetriever
    from optivision.metrics import ndcg_at_k

    q, _ = as_query_corpus(index, queries)
    sets = None
    if answers is not None:
        sets = answer_sets(list(index_store(index).ids), answers, len(q))
    compressed = index_rank(index, q, 10)
    f16 = index_float16(index)
    out = QuestionCheck(n=len(q), needs_float16=f16 is None)
    floats = None
    if f16 is not None:
        floats = ranking_lists(ExactRetriever(f16).search(q, 10))[0]
        cand = {qid: compressed[i] for i, qid in enumerate(q.ids)}
        ref = {qid: floats[i] for i, qid in enumerate(q.ids)}
        top = {qid: {floats[i][0]} for i, qid in enumerate(q.ids) if floats[i]}
        c = [ndcg_at_k(cand[qid], top[qid], 5) for qid in top]
        out.agreement = float(np.mean(c)) if c else None
        out.agreement_stat = _bound({k: cand[k] for k in top}, ref, top, target)
    if sets is not None:
        answered = [i for i, s in enumerate(sets) if s]
        out.n_answered = len(answered)
        if answered:
            c = [ndcg_at_k(compressed[i], sets[i], 5) for i in answered]
            out.quality = float(np.mean(c))
            if floats is not None:
                b = [ndcg_at_k(floats[i], sets[i], 5) for i in answered]
                out.quality_float16 = float(np.mean(b))
                out.retention = float(np.sum(c) / np.sum(b)) if np.sum(b) > 0 else None
                rel = {q.ids[i]: sets[i] for i in answered}
                out.retention_stat = _bound({q.ids[i]: compressed[i] for i in answered},
                                            {q.ids[i]: floats[i] for i in answered}, rel, target)
    for stat in (out.agreement_stat, out.retention_stat):
        if stat is not None and not bool(getattr(stat, "passed", True)):
            out.fell_below = True
    return out


# ============================================================================ practice rerun


def _ref_get(ref: Any, *names: str) -> Any:
    for name in names:
        if isinstance(ref, Mapping) and ref.get(name) is not None:
            return ref[name]
        if not isinstance(ref, Mapping) and getattr(ref, name, None) is not None:
            return getattr(ref, name)
    return None


def _as_id_lists(value: Any, ids: Sequence[str]) -> dict[str, list[str]]:
    if isinstance(value, Mapping):
        items = value.items()
    else:
        items = zip(ids, value)
    out = {}
    for key, row in items:
        out[str(key)] = [str(x[0]) if isinstance(x, (tuple, list)) else str(x) for x in row]
    return out


@dataclass
class PracticeCheck:
    ran: bool
    reason: str = ""
    n: int = 0
    R: float | None = None
    U: float | None = None
    passed: bool = True
    runs: int | None = None
    runs_saved: bool = False
    mode: str = "step-down"


def _practice(index: Any, target: float) -> PracticeCheck:
    from . import _ledger

    try:
        if not _ledger.is_on("practice_check"):
            return PracticeCheck(False, "practice check switched off: " + _ledger.feature("practice_check")["reason"])
        mode = str(_ledger.setting("practice_check", "mode", "step-down"))
    except KeyError:
        mode = "step-down"
    ref = index_reference(index)
    queries = _ref_get(ref, "queries", "query_vectors") if ref else None
    if queries is None or len(queries) == 0:
        return PracticeCheck(False, "no practice questions were made for this index (unchecked)")
    relevant = {str(k): set(map(str, v)) for k, v in dict(_ref_get(ref, "relevant", "relevant_sets") or {}).items()}
    clusters = {str(k): str(v) for k, v in dict(_ref_get(ref, "clusters") or {}).items()}
    rankings = _ref_get(ref, "rankings", "ref", "ref_rankings", "float16_rankings")
    if not relevant or rankings is None:
        return PracticeCheck(False, "the saved practice questions lack their answers or reference rankings")
    try:
        from . import _veto
    except ImportError:
        return PracticeCheck(False, "the practice check is not part of this install")
    ids = list(queries.ids)
    ref_lists = _as_id_lists(rankings, ids)
    cand_lists = dict(zip(ids, index_rank(index, queries, 10)))
    keep = [i for i in ids if i in relevant and i in ref_lists]
    stat = _veto.practice_check({i: cand_lists[i] for i in keep}, {i: ref_lists[i] for i in keep},
                                {i: relevant[i] for i in keep}, {i: clusters.get(i, i) for i in keep}, target)
    runs = int(index_manifest(index).get("practice_runs") or 1) + 1
    saved = index_commit(index, practice_runs=runs)
    return PracticeCheck(True, "", int(stat.n), float(stat.R), float(stat.U), bool(stat.passed), runs, saved, mode)


# ============================================================================ size


@dataclass
class SizeCheck:
    n_pages: int
    n_built: int | None
    status: str
    qualifies: bool
    counts_against: bool
    today: Any = None
    row: LayoutRow | None = None


def _size(index: Any, n_now: int) -> SizeCheck:
    from . import _levels

    sel = index_selection(index)
    layout: LayoutRow = index.layout
    today = _levels.select(getattr(sel, "model", None), index_target(index), n_now,
                           checked=bool(getattr(sel, "checked", False)),
                           modality=getattr(sel, "modality", None),
                           vectors_per_page=index_vectors_per_page(index))
    row = find_row(today.rows, layout.name)
    status = row.status if row is not None else "not in the ladder"
    qualifies = status in QUALIFYING
    built_row = find_row(getattr(sel, "rows", ()) or (), layout.name)
    never = bool(getattr(sel, "hand_picked", False)) and (built_row is None or built_row.status not in QUALIFYING)
    return SizeCheck(n_now, index_built_n_pages(index), status, qualifies, not qualifies and not never, today, row)


# ============================================================================ the report


@dataclass
class CheckReport:
    """What ``index.check()`` found. ``lines`` are the ``_claim.Line`` objects of the printed
    block; ``str(report)`` renders it; ``apply()`` makes the advised change (prints its time
    first) and is never called by ``check()`` itself."""

    ok: bool
    complete: bool
    record: RecordCheck
    drift: DriftCheck
    size: SizeCheck
    practice: PracticeCheck
    growth: float | None
    questions: QuestionCheck | None
    spot: Any
    advice: LayoutRow | None
    advice_reason: str
    refit: bool
    block: Block
    _index: Any = field(default=None, repr=False, compare=False)

    @property
    def lines(self) -> list[_claim.Line]:
        return self.block.lines

    def __str__(self) -> str:
        return str(self.block)

    def __repr__(self) -> str:
        return str(self.block)

    def as_dict(self) -> dict[str, Any]:
        q = self.questions
        return {
            "ok": self.ok,
            "complete": self.complete,
            "record": {"matches": self.record.matches, "model_compared": self.record.model_compared,
                       "compared": list(self.record.compared),
                       "differences": [list(d) for d in self.record.differences]},
            "drift": {"checked": self.drift.checked, "reason": self.drift.reason, "drifted": self.drift.drifted,
                      "max_change": self.drift.max_change, "changes": dict(self.drift.changes),
                      "seconds": self.drift.seconds},
            "size": {"n_pages": self.size.n_pages, "n_pages_at_build": self.size.n_built,
                     "layout": getattr(self.size.row, "name", None), "status": self.size.status,
                     "qualifies": self.size.qualifies,
                     "pick_today": getattr(getattr(self.size.today, "pick", None), "name", None)},
            "practice": {"ran": self.practice.ran, "reason": self.practice.reason, "n": self.practice.n,
                         "R": self.practice.R, "U": self.practice.U, "passed": self.practice.passed,
                         "runs": self.practice.runs, "mode": self.practice.mode},
            "growth": self.growth,
            "questions": None if q is None else {
                "n": q.n, "agreement": q.agreement, "agreement_kind": AGREEMENT, "quality": q.quality,
                "quality_float16": q.quality_float16, "retention": q.retention, "quality_kind": QUALITY,
                "n_answered": q.n_answered, "needs_float16": q.needs_float16, "fell_below": q.fell_below},
            "spot_check": None if self.spot is None else self.spot.as_dict(),
            "advice": None if self.advice is None else {"layout": self.advice.name, "plain": self.advice.plain,
                                                         "reason": self.advice_reason},
            "refit": self.refit,
            "block": self.block.as_dict(),
        }

    def apply(self, *, out: Callable[[str], Any] | None = print) -> Any:
        """Make the advised change: print its time, then re-settle or re-encode through the
        index's rebuild path. Returns the rebuilt index, or None when nothing is advised."""
        index = self._index
        if self.advice is None and not self.refit:
            if out is not None:
                block = Block([], next="index.search(...)")
                block.add("Apply", "nothing to apply: the check advised no change")
                out(str(block))
            return None
        layout = self.advice.name if self.advice is not None else index.layout.name
        if out is not None:
            out(str(_time_block(index, layout)))
        return rebuild(index, layout, refit=True)


def _time_block(index: Any, layout: str) -> Block:
    block = Block([], next="the change runs now; the index stays usable until it is replaced")
    block.add("Change", "new layout {0}", lit(layout))
    if index_float16(index) is not None:
        block.add("Time", "no re-encoding: the layout settles from the kept float16 copy")
    else:
        est = encode_estimate(index, index_n_pages(index))
        if est is None:
            block.add("Time", "re-encodes every page; no time estimate for this model")
        else:
            block.add("Time", "re-encodes every page: {0}", est)
    return block


def rebuild(index: Any, layout: str, *, refit: bool = True) -> Any:
    """Re-settle the index in ``layout`` through its rebuild path (``Index._rebuild``), from the
    float16 copy when it is kept, else by re-encoding every page."""
    hook = getattr(index, "_rebuild", None)
    if callable(hook):
        return hook(layout=layout, refit=refit)
    from . import _api

    sources = index_manifest(index).get("sources") or []
    if len(sources) != 1:
        raise EasyError("this index cannot rebuild itself here; run ov.build(...) again with "
                        f"layout={layout!r}, rebuild=True and the same path")
    src = sources[0] if isinstance(sources[0], str) else sources[0].get("path")
    return _api.build(src, model=index_model(index), target=index_target(index), path=index_path(index),
                      layout=layout, rebuild=True, keep_float16=index_float16(index) is not None)


# ============================================================================ run_check


def _catalogue_source(row: LayoutRow | None) -> str:
    if row is not None and row.datasets:
        return ", ".join(row.datasets)
    return "our measured catalogue"


def run_check(index: Any, queries: Any = None, *, answers: Any = None, drift: Any = "auto",
              spot_check: bool = False, out: Callable[[str], Any] | None = print) -> CheckReport:
    """FINAL_DESIGN 5.5.3 on ``index``; see the module docstring. Changes nothing in the index
    except the count of practice reruns; ``out`` receives the time lines printed before slow
    steps (``print`` by default; None for silence)."""
    if answers is not None and queries is None:
        raise ValueError("answers= needs queries=: one answer per query")
    if spot_check and queries is None:
        raise ValueError("spot_check=True needs queries=: it re-encodes each question's top pages")
    target = index_target(index)
    layout: LayoutRow = index.layout
    n_now = index_n_pages(index)

    drift_result = _run_drift(index, drift, out)
    size = _size(index, n_now)
    practice = _practice(index, target)
    growth = (n_now / size.n_built) if size.n_built else None
    questions = _questions(index, queries, answers, target) if queries is not None else None
    spot = None
    if spot_check:
        from . import _spotcheck

        spot = _spotcheck.spot_check(index, queries, out=out)
    # the model's own settings can be compared once something loaded it (drift, text questions,
    # the spot check); the build record is never a reason to load it
    now = drift_result.encoder_record
    model_loaded = (spot is not None and spot.pages_encoded > 0) or (queries is not None and (
        isinstance(queries, str) or (isinstance(queries, Sequence) and bool(queries)
                                     and all(isinstance(t, str) for t in queries))))
    if now is None and model_loaded:
        loaded = getattr(index._encoder(), "record", None)
        now = dict(loaded) if loaded else None
    record = compare_record(index_record(index), now)

    # ------------------------------------------------------------ advice
    advice, reason, refit = None, "", False
    practice_fired = practice.ran and not practice.passed and practice.mode == "step-down"
    if size.counts_against and size.today is not None:
        advice, reason = size.today.pick, "the layout no longer qualifies at this size"
    if practice_fired:
        safer = _next_safer(index_selection(index), layout)
        if safer is not None and (advice is None or _row_ratio(safer) < _row_ratio(advice)):
            advice, reason = safer, "the practice check fell below the target"
    questions_below = (questions is not None and questions.fell_below) or bool(spot is not None and spot.below_target)
    if questions_below:
        safer = _next_safer(index_selection(index), layout)
        if safer is not None and (advice is None or _row_ratio(safer) < _row_ratio(advice)):
            advice, reason = safer, "your questions fell below the target"
    if advice is not None and advice.name == layout.name:
        advice = None
    if advice is None and growth is not None and growth > GROWTH_FACTOR:
        refit, reason = True, "the collection more than doubled since the layout was fitted"

    ok = (record.matches and not drift_result.drifted and not practice_fired and not size.counts_against
          and not questions_below)
    complete = record.model_compared and drift_result.checked and practice.ran

    block = _render(index, record, drift_result, size, practice, growth, questions, spot,
                    advice, reason, refit, ok, complete, layout, target)
    return CheckReport(ok, complete, record, drift_result, size, practice, growth, questions, spot,
                       advice, reason, refit, block, index)


def _next_safer(selection: Any, layout: LayoutRow) -> LayoutRow | None:
    """The next layout of the fixed test order after ``layout`` (FINAL_DESIGN 5.5.1), or the first
    one less compressed than it when the layout is not in that order."""
    sequence = list(getattr(selection, "sequence", ()) or ())
    names = [r.name for r in sequence]
    if layout.name in names:
        i = names.index(layout.name)
        return sequence[i + 1] if i + 1 < len(sequence) else None
    ratio = _row_ratio(layout)
    return next((r for r in sequence if _row_ratio(r) < ratio), None)


def _render(index: Any, record: RecordCheck, drift: DriftCheck, size: SizeCheck, practice: PracticeCheck,
            growth: float | None, questions: QuestionCheck | None, spot: Any, advice: LayoutRow | None,
            reason: str, refit: bool, ok: bool, complete: bool, layout: LayoutRow, target: float) -> Block:
    from . import _ledger

    b = Block([])
    path = index_path(index)
    if path is not None:
        b.add("Index", "{0}", lit(path))
    b.add("Layout", "{0} ({1}), target {2}", lit(layout.plain), lit(layout.name),
          claim(target, "counted", text=_claim.pct(target, 0)))

    # 1 build record
    if record.differences:
        for name, built, now in record.differences:
            b.add("Record", "{0} differs: {1} at build, {2} now", lit(name), lit(built), lit(now))
    elif record.compared:
        b.add("Record", "matches the build record: {0}", lit(", ".join(record.compared)))
    else:
        b.add("Record", "this index saved no build record to compare")
    if not record.model_compared:
        b.add("", "model settings (commit, dtype, attention, page size, batch) are compared when the model loads")

    # 2 drift
    if not drift.checked:
        b.add("Drift", "{0}", lit(drift.reason))
    elif drift.drifted:
        b.add("Drift", "the four fixed pages score differently from build: up to {0} apart (allowed {1})",
              claim(drift.max_change, "counted", text=_claim.pct(drift.max_change, 1)),
              claim(DRIFT_REL_TOL, "counted", text=_claim.pct(DRIFT_REL_TOL, 0)))
        b.add("", "your model now produces different vectors than at build time; see ov.doctor()")
    else:
        b.add("Drift", "none: the four fixed pages score within {0} of build", claim(
            DRIFT_REL_TOL, "counted", text=_claim.pct(DRIFT_REL_TOL, 0)))

    # 3 size
    if size.n_built:
        b.add("Pages", "{0} today, {1} when the layout was fitted", claim(size.n_pages, "counted"),
              claim(size.n_built, "counted"))
    else:
        b.add("Pages", "{0} today", claim(size.n_pages, "counted"))
    src = _catalogue_source(size.row)
    if size.qualifies:
        b.add("Size rule", "the layout still qualifies at this size ({0})", lit(size.status))
    elif not size.counts_against:
        b.add("Size rule", "chosen by you; the rule does not qualify it ({0}), as at build", lit(size.status))
    else:
        b.add("Size rule", "the layout no longer qualifies at this size ({0})", lit(size.status))
        if size.row is not None and size.row.lowest_kept is not None and size.row.need is not None:
            b.add("", "lowest kept {0}, needed at this size {1}",
                  claim(size.row.lowest_kept, "catalogue", source=src, text=_claim.pct(size.row.lowest_kept)),
                  claim(size.row.need, "catalogue", source=src, text=_claim.pct(size.row.need)))

    # 4 practice rerun
    if not practice.ran:
        b.add("Practice", "not rerun: {0}", lit(practice.reason))
    else:
        verdict = "passed" if practice.passed else (
            "fell below the target" if practice.mode == "step-down" else "fell below the target (a warning only)")
        b.add("Practice", "rerun from the stored codes: {0} on {1} hard questions made from your PDFs",
              lit(verdict), claim(practice.n, "probe"))
        b.add("", "that is {0} of full-precision quality on them", claim(
            practice.R, "probe", construct="nDCG@5 retention", text=_claim.retention(practice.R)))
        if practice.runs is not None:
            saved = "" if practice.runs_saved else " (count not saved: the index is read-only or locked)"
            b.add("Runs", "practice check run {0} on this index; every rerun is another look{1}",
                  claim(practice.runs, "counted"), lit(saved))
        b.add("", "practice questions are easier than real ones; a pass cannot confirm the target on your searches")

    # 5 growth
    if growth is not None and growth > GROWTH_FACTOR:
        b.add("Growth", "more than {0} the size the layout was fitted at; a refit is advised",
              claim(GROWTH_FACTOR, "counted", text="twice"))
        s = GROWTH_SOURCE
        b.add("", "In our two tests ({0} -> {1} pages, distractors from other domains), our layouts lost {2} "
              "and {3} of retention on average, at worst {4}",
              claim(GROWTH_TEST_PAGES[0], "catalogue", source=s), claim(GROWTH_TEST_PAGES[1], "catalogue", source=s),
              claim(GROWTH_MEAN_LOSS[0], "catalogue", source=s, text=f"{GROWTH_MEAN_LOSS[0]:.3f}"),
              claim(GROWTH_MEAN_LOSS[1], "catalogue", source=s, text=f"{GROWTH_MEAN_LOSS[1]:.3f}"),
              claim(GROWTH_WORST_LOSS, "catalogue", source=s, text=f"{GROWTH_WORST_LOSS:.3f}"))
    if (growth is not None and growth > GROWTH_FACTOR) or (size.counts_against and size.row is not None
                                                            and size.row.lowest_kept is not None):
        b.add("Status", "{0}", lit(_ledger.status()))

    # 6 your questions
    if questions is not None:
        _render_questions(b, questions, target)
    if spot is not None:
        for line in spot.lines:
            b.lines.append(line)

    # result and next step
    if ok and complete:
        b.add("Result", "all clear")
    elif ok:
        b.add("Result", "nothing wrong found; some steps did not run (see above)")
    else:
        b.add("Result", "needs attention")
    if advice is not None:
        named = advice.plain if advice.plain == advice.name else f"{advice.plain} ({advice.name})"
        b.add("Advice", "{0}: use {1}", lit(reason), lit(named))
        b.next = "index.check().apply()  (prints its time before it starts; nothing changes until you run it)"
    elif refit:
        b.add("Advice", "{0}: refit the same layout", lit(reason))
        b.next = "index.check().apply()  (prints its time before it starts; nothing changes until you run it)"
    elif record.differences or drift.drifted:
        b.next = f"ov.doctor({index_model(index)!r})"
    elif questions is None:
        b.next = "index.check(queries=[...]) with your real questions"
    else:
        b.next = "index.search(...)"
    return b


def _render_questions(b: Block, q: QuestionCheck, target: float) -> None:
    if q.agreement is not None:
        b.add("Questions", "on {0} of your questions, agreement with the full-precision top page was {1}",
              claim(q.n, "user-queries", source=AGREEMENT),
              claim(q.agreement, "user-queries", source=AGREEMENT, construct="nDCG@5 agreement@1",
                    text=_claim.retention(q.agreement)))
        if q.agreement_stat is not None:
            verdict = "does not show it below" if q.agreement_stat.passed else "puts it below"
            b.add("", "the bound of the practice check {0} the target of {1}", lit(verdict),
                  claim(target, "user-queries", source=AGREEMENT, text=_claim.pct(target, 0)))
    elif q.needs_float16:
        b.add("Questions", "agreement needs the float16 copy: build with keep_float16=True")
        b.add("", "without it: index.check(queries, spot_check=True) re-encodes each question's top pages; "
              "it can only flag")
    if q.n_answered:
        b.add("Answers", "search quality on {0} answered questions: nDCG@5 {1}",
              claim(q.n_answered, "user-queries", source=QUALITY),
              claim(q.quality, "user-queries", source=QUALITY, construct="nDCG@5", text=f"{_floor3(q.quality)}"))
        if q.retention is not None:
            b.add("", "full precision scores {0}; retention {1}",
                  claim(q.quality_float16, "user-queries", source=QUALITY, construct="nDCG@5",
                        text=f"{_floor3(q.quality_float16)}"),
                  claim(q.retention, "user-queries", source=QUALITY, construct="nDCG@5 retention",
                        text=_claim.retention(q.retention)))
            if q.retention_stat is not None:
                verdict = "does not show it below" if q.retention_stat.passed else "puts it below"
                b.add("", "the bound of the practice check {0} the target of {1}", lit(verdict),
                      claim(target, "user-queries", source=QUALITY, text=_claim.pct(target, 0)))
        else:
            b.add("", "retention against full precision needs the float16 copy (keep_float16=True)")


def _floor3(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{np.floor(float(x) * 1000 + 1e-9) / 1000:.3f}"
