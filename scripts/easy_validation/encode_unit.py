"""Encode the easy-mode validation units on a GPU and save what the CPU analyses need.

    python scripts/easy_validation/encode_unit.py jobs.json --out results_<batch> --deadline-hours 11

``jobs.json`` is ``{"batch": "<name>", "jobs": [<Job.as_dict()>, ...]}`` (``units.py``). For
every corpus named by a job the script first writes ``_corpus/<key>/`` once: the page list
(id, document, page number, blank flag, sha256 of the RGB bytes, orientation), the page text
(PDF text layer where the dataset ships its PDFs, and the dataset's OCR text) and the Tier A
practice questions made from each text source with ``optivision.easy._probes.make_probes``
(seed 0, at most 600 pages, 3 per page). Then, per model, it runs ``check_retriever`` on 12
labelled pairs (nothing is encoded if it fails) and writes ``<model>__<corpus>/``:

    docs_f16/shard_NNNNNN.npz     page vectors in float16, 32 pages a shard (vectors, offsets, ids)
    docs_f32/shard_NNNNNN.npz     the same in float32, only for corpora in ``keep_float32`` (V3)
    queries_f16.npz               the corpus's English-original queries
    probes_<source>_f16.npz       the practice questions of each text source, as queries
    run.json                      what ran, how long, versions, revision, dtype, outcome

Shards are written atomically (``os.replace``) and skipped when present, so a second session
started with the first one's output resumes where it stopped. A job that runs out of time is
recorded as ``partial``. Labels: ``_corpus/<key>/qrels.json`` (``{qid: {page_id: grade}}``,
grade > 0) and ``queries.json``.

The analyses never need the model again: every layout is rescored on a CPU from the float16
vectors (FINAL_DESIGN section 8).
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import time
from collections.abc import Iterator, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

from easy_validation import units  # noqa: E402

SHARD_PAGES = 32
TEXT_SHARD_DOCS = 2000
RESERVE_SECONDS = 20 * 60  # kept for zipping the results
CHECK_PAIRS = 12


# --------------------------------------------------------------------------- session


class Session:
    def __init__(self, deadline_hours: float) -> None:
        self.t0 = time.time()
        self.deadline = self.t0 + deadline_hours * 3600

    def remaining(self) -> float:
        return self.deadline - time.time() - RESERVE_SECONDS


def _script(name: str):
    """A sibling script of ``scripts/`` as a module (encode_vectors, encode_text_vectors)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "colpali-engine", "sentence-transformers", "datasets", "numpy",
                 "pypdfium2", "optivision-rag", "accelerate", "peft"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    try:
        import torch

        if torch.cuda.is_available():
            out["gpus"] = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    except ImportError:
        pass
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def save_arrays(path: Path, vectors: Any, offsets: Any, ids: Sequence[str], dtype: str = "float16") -> None:
    """One shard: ``vectors`` (``dtype``), ``offsets`` (int64), ``ids`` (str); written atomically."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, vectors=np.asarray(vectors, dtype=dtype), offsets=np.asarray(offsets, dtype=np.int64),
             ids=np.asarray(list(ids), dtype=str))
    os.replace(tmp, path)


def save_corpus(path: Path, corpus: Any, dtype: str = "float16") -> None:
    save_arrays(path, corpus.vectors, corpus.offsets, corpus.ids, dtype)


# --------------------------------------------------------------------------- sources


@dataclasses.dataclass
class Source:
    """A corpus read from Hugging Face: page or document ids, labels and page text.

    ``item(i)`` returns page ``i`` as a PIL image (pages) or its text (documents), decoding it
    only then, so a 5,244-page corpus is never held in memory.
    """

    key: str
    ids: list[str]
    meta: list[dict]
    queries: list[tuple[str, str]]
    qrels: dict[str, dict[str, int]]
    kind: str = "image"
    _rows: Any = None
    _column: str = "image"
    texts: dict[str, dict[str, str | None]] = dataclasses.field(default_factory=dict)
    info: dict = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    def item(self, i: int) -> Any:
        row = self._rows[int(self.meta[i]["row"])]
        value = row[self._column]
        return value.convert("RGB") if self.kind == "image" else value

    def items(self, lo: int, hi: int) -> list[Any]:
        return [self.item(i) for i in range(lo, hi)]


def _load(hf_id: str, config: str | None, split: str, revision: str | None):
    from datasets import load_dataset

    kwargs = {"revision": revision} if revision else {}
    return load_dataset(hf_id, config, split=split, **kwargs) if config else load_dataset(hf_id, split=split, **kwargs)


def _unique(ids: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i in ids:
        n = seen.get(i, 0)
        seen[i] = n + 1
        out.append(i if n == 0 else f"{i}~{n + 1}")
    return out


def read_vidore_v3(c: units.Corpus, data: Path, limit: int | None = None, text: bool = True) -> Source:
    """ViDoRe V3: corpus / queries / qrels / documents_metadata configs; English-original queries only."""
    corpus = _load(c.hf_id, "corpus", "test", c.revision)
    queries = _load(c.hf_id, "queries", "test", c.revision)
    qrels = _load(c.hf_id, "qrels", "test", c.revision)
    docs = _load(c.hf_id, "documents_metadata", "test", c.revision)
    small = corpus.select_columns(["corpus_id", "doc_id", "page_number_in_doc"]
                                  + (["markdown"] if "markdown" in corpus.column_names else []))
    n = len(small) if limit is None else min(limit, len(small))
    ids, meta, by_corpus_id, ocr = [], [], {}, {}
    for i in range(n):
        r = small[i]
        pid = f"{r['doc_id']}::p{int(r['page_number_in_doc']) + 1}"
        ids.append(pid)
        meta.append({"row": i, "corpus_id": int(r["corpus_id"]), "doc_id": r["doc_id"],
                     "page": int(r["page_number_in_doc"]) + 1})
        ocr[pid] = r.get("markdown")
    ids = _unique(ids)
    for pid, m in zip(ids, meta, strict=True):
        m["id"] = pid
        by_corpus_id[m["corpus_id"]] = pid
    english = {}
    for r in queries.select_columns(["query_id", "query", "language"]):
        if str(r["language"]).lower().startswith("en"):
            english[str(r["query_id"])] = r["query"]
    labels: dict[str, dict[str, int]] = {}
    for r in qrels.select_columns(["query_id", "corpus_id", "score"]):
        q, d, s = str(r["query_id"]), int(r["corpus_id"]), int(r["score"])
        if q in english and s > 0 and d in by_corpus_id:
            labels.setdefault(q, {})[by_corpus_id[d]] = s
    qs = [(q, english[q]) for q in sorted(english, key=int) if q in labels]
    src = Source(c.key, ids, meta, qs, labels, "image", corpus, "image",
                 info={"hf_id": c.hf_id, "revision": c.revision, "english_queries": len(english),
                       "labelled_english_queries": len(qs), "pages": len(ids)})
    files = {r["doc_id"]: r["file_name"] for r in docs}
    src.texts = {pid: {"ocr": ocr.get(pid), "text_layer": None} for pid in ids}
    if text:
        src.info["text_layer"] = add_text_layer(src, c, files, data)
    return src


def add_text_layer(src: Source, c: units.Corpus, files: dict[str, str], data: Path) -> dict:
    """The PDF text layer of every page (pypdfium2), from the PDFs the dataset ships."""
    from huggingface_hub import hf_hub_download

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        return {"status": f"skipped: {exc}"}
    by_doc: dict[str, list[dict]] = {}
    for m in src.meta:
        by_doc.setdefault(m["doc_id"], []).append(m)
    done, missing, errors = 0, [], []
    for doc_id, pages in by_doc.items():
        name = files.get(doc_id)
        if name is None:
            missing.append(doc_id)
            continue
        try:
            path = hf_hub_download(c.hf_id, f"pdfs/{name}", repo_type="dataset", revision=c.revision,
                                   cache_dir=str(data / "pdfs"))
        except Exception as exc:  # noqa: BLE001 -- a missing PDF leaves its pages without a text layer
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
            continue
        pdf = pdfium.PdfDocument(path)
        try:
            n_pdf = len(pdf)
            for m in pages:
                index = m["page"] - 1
                if not 0 <= index < n_pdf:
                    continue
                page = pdf[index]
                try:
                    tp = page.get_textpage()
                    try:
                        src.texts[m["id"]]["text_layer"] = tp.get_text_range()
                    finally:
                        tp.close()
                finally:
                    page.close()
                done += 1
        finally:
            pdf.close()
    return {"status": "ok", "pages_with_text_layer_read": done, "documents_without_pdf": missing,
            "errors": errors[:10]}


def read_beir_image(c: units.Corpus, data: Path, limit: int | None = None, text: bool = True) -> Source:
    """BEIR-layout page corpus (REAL-MM-RAG ``*_BEIR``): corpus / queries / qrels configs."""
    corpus = _load(c.hf_id, "corpus", "test", c.revision)
    queries = _load(c.hf_id, "queries", "test", c.revision)
    qrels = _load(c.hf_id, "qrels", "test", c.revision)
    small = corpus.select_columns(["corpus-id", "image_filename", "doc-id"])
    n = len(small) if limit is None else min(limit, len(small))
    ids, meta, by_corpus = [], [], {}
    for i in range(n):
        r = small[i]
        m = re.search(r"_page_(\d+)\.\w+$", str(r["image_filename"]))
        page = int(m.group(1)) if m else 1
        doc = str(r["doc-id"]) if r["doc-id"] else f"{int(r['corpus-id']):05d}"
        ids.append(f"{doc}::p{page}")
        meta.append({"row": i, "corpus_id": int(r["corpus-id"]), "doc_id": doc, "page": page,
                     "image_filename": r["image_filename"]})
    ids = _unique(ids)
    for pid, m in zip(ids, meta, strict=True):
        m["id"] = pid
        by_corpus[m["corpus_id"]] = pid
    cols = [x for x in ("query-id", "query", "language") if x in queries.column_names]
    text_of = {str(r["query-id"]): r["query"] for r in queries.select_columns(cols)
               if str(r.get("language", "english")).lower().startswith("en")}
    labels: dict[str, dict[str, int]] = {}
    for r in qrels.select_columns(["query-id", "corpus-id", "score"]):
        q, d, s = str(r["query-id"]), int(r["corpus-id"]), int(r["score"])
        if q in text_of and s > 0 and d in by_corpus:
            labels.setdefault(q, {})[by_corpus[d]] = s
    qs = [(q, text_of[q]) for q in sorted(text_of, key=lambda x: int(x) if x.isdigit() else x) if q in labels]
    src = Source(c.key, ids, meta, qs, labels, "image", corpus, "image",
                 info={"hf_id": c.hf_id, "revision": c.revision, "queries": len(qs), "pages": len(ids)})
    src.texts = {pid: {"ocr": None, "text_layer": None} for pid in ids}
    return src


def read_vidore_v1(c: units.Corpus, data: Path, limit: int | None = None, text: bool = True) -> Source:
    """ViDoRe V1 rows (``image`` + ``query``): page ``i`` is ``f"{i:05d}::p1"``, each distinct query
    relevant to the first row it appears on (``scripts/encode_vectors.py``)."""
    ds = _load(c.hf_id, c.config, c.split, c.revision)
    n = len(ds) if limit is None else min(limit, len(ds))
    cols = [x for x in ("query", "text_description", "text", "ocr") if x in ds.column_names]
    small = ds.select_columns(cols) if cols else None
    ids, meta, qs, labels, seen, texts = [], [], [], {}, {}, {}
    for i in range(n):
        pid = f"{i:05d}::p1"
        ids.append(pid)
        meta.append({"row": i, "id": pid, "doc_id": f"{i:05d}", "page": 1})
        r = small[i] if small is not None else {}
        q = r.get("query")
        ocr = r.get("text_description") or r.get("ocr") or r.get("text")
        texts[pid] = {"ocr": ocr, "text_layer": None}
        if q and q not in seen:
            qid = f"q{len(qs):04d}"
            seen[q] = qid
            qs.append((qid, q))
            labels[qid] = {pid: 1}
    src = Source(c.key, ids, meta, qs, labels, "image", ds, "image",
                 info={"hf_id": c.hf_id, "revision": c.revision, "queries": len(qs), "pages": len(ids)})
    src.texts = texts
    return src


def read_beir_text(c: units.Corpus, data: Path, limit: int | None = None, text: bool = True) -> Source:
    """BEIR text corpus at the pinned revisions; FiQA is subsampled to ``c.subsample`` keeping positives."""
    import numpy as np

    name = c.hf_id.split("/")[-1]
    corpus = _load(c.hf_id, "corpus", "corpus", c.revision)
    queries = _load(c.hf_id, "queries", "queries", c.revision)
    qrels = _load(c.qrels_id or f"BeIR/{name}-qrels", None, "test", c.qrels_revision)
    labels: dict[str, dict[str, int]] = {}
    for r in qrels:
        if int(r["score"]) > 0:
            labels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    all_ids = [str(x) for x in corpus["_id"]]
    keep = np.arange(len(all_ids))
    if c.subsample and c.subsample < len(all_ids):
        positive = {d for docs in labels.values() for d in docs}
        pos_idx = [i for i, d in enumerate(all_ids) if d in positive]
        rest = np.array([i for i, d in enumerate(all_ids) if d not in positive])
        rng = np.random.default_rng(0)
        extra = rng.choice(rest, size=c.subsample - len(pos_idx), replace=False)
        keep = np.sort(np.concatenate([np.array(pos_idx), extra]))
    if limit:
        keep = keep[:limit]
    keep_set = {all_ids[i] for i in keep}
    ids = [all_ids[i] for i in keep]
    meta = [{"row": int(i), "id": all_ids[i], "doc_id": all_ids[i], "page": 1} for i in keep]
    qtext = {str(r["_id"]): r["text"] for r in queries}
    labels = {q: {d: s for d, s in docs.items() if d in keep_set} for q, docs in labels.items() if q in qtext}
    labels = {q: docs for q, docs in labels.items() if docs}
    qs = [(q, qtext[q]) for q in sorted(labels, key=lambda x: int(x) if x.isdigit() else x)]

    class _Rows:
        def __getitem__(self, i: int) -> dict:
            r = corpus[int(i)]
            return {"text": (r["title"] + " " + r["text"]).strip()}

    src = Source(c.key, ids, meta, qs, labels, "text", _Rows(), "text",
                 info={"hf_id": c.hf_id, "revision": c.revision, "qrels": c.qrels_id, "qrels_revision": c.qrels_revision,
                       "documents": len(ids), "queries": len(qs), "subsample_seed": 0 if c.subsample else None,
                       "self_match_excluded": c.key in units.SELF_MATCH_EXCLUDED})
    return src


READERS = {"vidore_v3": read_vidore_v3, "beir_image": read_beir_image, "vidore_v1": read_vidore_v1,
           "beir_text": read_beir_text}


def read_source(key: str, data: Path, limit: int | None = None, text: bool = True) -> Source:
    c = units.corpus(key)
    return READERS[c.format](c, data, limit, text)


# --------------------------------------------------------------------------- pages, text, probes


def page_record_dicts(src: Source) -> list[dict]:
    """Per page: blank flag (``optivision.demo.looks_blank``), sha256 of the RGB bytes, orientation."""
    from optivision.demo import looks_blank

    out = []
    for i, m in enumerate(src.meta):
        im = src.item(i)
        out.append({**{k: v for k, v in m.items() if k != "row"}, "width": im.width, "height": im.height,
                    "landscape": im.width > im.height, "blank": bool(looks_blank(im)),
                    "sha256": hashlib.sha256(im.tobytes()).hexdigest()})
    return out


def build_page_records(pages: list[dict], texts: dict[str, dict], source: str) -> list[Any]:
    """``optivision.easy._docs.PageRecord`` objects with the page text of one source."""
    from optivision.easy import _docs

    fields = {f.name for f in dataclasses.fields(_docs.PageRecord)}
    records = []
    for p in pages:
        text = (texts.get(p["id"]) or {}).get(source) or ""
        values = {"id": p["id"], "doc_id": p["doc_id"], "file": None, "relpath": p["doc_id"], "page": p["page"],
                  "text": text, "n_chars": len(text.strip()), "blank": p["blank"], "sha256": p["sha256"],
                  "landscape": p["landscape"]}
        values.update({f: None for f in fields - set(values)})
        records.append(_docs.PageRecord(**{k: v for k, v in values.items() if k in fields}))
    return records


def probe_dict(p: Any) -> dict:
    relevant = getattr(p, "relevant", ()) or ()
    return {"id": str(p.id), "text": str(p.text), "family": getattr(p, "family", None),
            "page_id": getattr(p, "page_id", None), "relevant": sorted(str(x) for x in relevant)}


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, (set, frozenset, tuple)):
        return [_asdict(x) for x in obj]
    return obj


def prepare_corpus(src: Source, out: Path, with_probes: bool) -> dict:
    """``_corpus/<key>/``: pages, labels, page text and the Tier A practice questions (once per corpus)."""
    folder = out / "_corpus" / src.key
    done = folder / "prepared.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "queries.json", [{"id": q, "text": t} for q, t in src.queries])
    write_json(folder / "qrels.json", src.qrels)
    record: dict[str, Any] = {"corpus": src.key, **src.info, "probes": {}}
    if src.kind == "image":
        pages = page_record_dicts(src)
        write_json(folder / "pages.json", pages)
        with gzip.open(folder / "texts.json.gz", "wt", encoding="utf-8") as f:
            json.dump(src.texts, f)
        if with_probes:
            for source in ("text_layer", "ocr"):
                if not any((t or {}).get(source) for t in src.texts.values()):
                    record["probes"][source] = {"status": "no page text from this source"}
                    continue
                try:
                    from optivision.easy import _docs, _probes

                    recs = build_page_records(pages, src.texts, source)
                    profile = _docs.profile(recs)
                    probes = _probes.make_probes(recs, seed=units.PROBE_SEED, max_pages=units.PROBE_MAX_PAGES,
                                                 per_page=units.PROBE_PER_PAGE)
                    write_json(folder / f"probes_{source}.json", [probe_dict(p) for p in probes])
                    write_json(folder / f"profile_{source}.json", _asdict(profile))
                    record["probes"][source] = {"status": "ok", "n": len(probes),
                                                "text_pages": sum(1 for r in recs if r.n_chars >= 20)}
                except Exception as exc:  # noqa: BLE001 -- the corpus is still encoded without probes
                    record["probes"][source] = {"status": f"failed: {type(exc).__name__}: {exc}"[:500]}
    else:
        write_json(folder / "pages.json", [{k: v for k, v in m.items() if k != "row"} for m in src.meta])
    write_json(done, record)
    return record


def check_pairs(src: Source, n: int = CHECK_PAIRS) -> list[tuple[int, str]]:
    """Up to ``n`` (page index, query) pairs: each query takes its most relevant page, pages are
    distinct, and no chosen query is relevant to another chosen page (``encode_text_vectors``)."""
    index = {pid: i for i, pid in enumerate(src.ids)}
    chosen: list[tuple[int, str]] = []
    used_pages: set[str] = set()
    used_queries: list[str] = []
    for qid, text in src.queries:
        rel = src.qrels.get(qid, {})
        if not rel:
            continue
        page = max(sorted(rel), key=lambda p: rel[p])
        if page in used_pages or set(rel) & used_pages or any(page in src.qrels[q] for q in used_queries):
            continue
        if page not in index:
            continue
        chosen.append((index[page], text))
        used_pages.add(page)
        used_queries.append(qid)
        if len(chosen) == n:
            break
    return chosen


# --------------------------------------------------------------------------- models


def gpu_budgets() -> dict[int, int]:
    return _script("encode_vectors")._gpu_budgets()


def load_model(job: dict) -> Any:
    """The package's own loader with the options ``scripts/encode_vectors.py`` /
    ``encode_text_vectors.py`` pass for this spec (dtype, multi-GPU, revision pin, batch size)."""
    from optivision.adapters import MODELS, load_adapter

    model = job["model"]
    info = MODELS.get(model)
    loader = job.get("loader") or (info.loader if info else "sentence-transformers")
    options: dict[str, Any] = {}
    if job.get("kind") == "text":
        options["batch_size"] = int(job.get("batch_size", 32))
        if loader == "sentence-transformers":
            options["encode_kwargs"] = {"show_progress_bar": False}
            if info is not None and info.revision:
                options["revision"] = info.revision
        if job.get("trust_remote_code"):
            options["trust_remote_code"] = True
        return load_adapter(model, loader=job.get("loader"), **options)
    dtype = job.get("dtype", "auto")
    if loader == "nemotron" or loader.startswith(("colpali-engine:", "remote-code:")):
        options.update(dtype=dtype, multi_gpu=bool(job.get("multi_gpu")))
    if loader.startswith("remote-code:"):
        options.update(trust_remote_code=bool(job.get("trust_remote_code")), revision=job.get("revision"))
    if loader == "sentence-transformers":
        options.update(batch_size=int(job.get("batch_size", 4)), encode_kwargs={"show_progress_bar": False})
        if dtype != "auto":
            options["dtype"] = dtype
        if job.get("trust_remote_code"):
            options["trust_remote_code"] = True
        revision = job.get("revision") or (info.revision if info else None)
        if revision and revision != "latest":
            options["revision"] = revision
        if job.get("multi_gpu"):
            options["model_kwargs"] = {"device_map": "auto", "max_memory": gpu_budgets()}
        if job.get("processor_kwargs"):
            options["processor_kwargs"] = job["processor_kwargs"]
    return load_adapter(model, loader=job.get("loader"), **options)


def encoder_info(adapter: Any, job: dict) -> dict:
    encoder = getattr(adapter, "encoder", adapter)
    return {"loader": job.get("loader"), "device": str(getattr(encoder, "device", None)),
            "dtype": str(getattr(encoder, "torch_dtype", job.get("dtype"))),
            "revision": getattr(encoder, "revision", None) or job.get("revision"),
            "loading_report": getattr(encoder, "loading_report", None),
            "placement": getattr(encoder, "placement", None),
            "max_num_visual_tokens": getattr(encoder, "max_num_visual_tokens", None)}


def encode_items(adapter: Any, items: list[Any], ids: list[str], kind: str) -> Any:
    from optivision.adapters import PageEncoderAdapter
    from optivision.types import PageRef

    if kind == "image" and isinstance(adapter, PageEncoderAdapter):
        return adapter.encode_documents([(PageRef(pid.split("::")[0], 1), im) for pid, im in zip(ids, items, strict=True)])
    return adapter.encode_documents(items, ids=ids)


def run_check(adapter: Any, src: Source, model: str) -> dict:
    from optivision.adapters import MODELS, PageEncoderAdapter, check_retriever
    from optivision.types import PageRef

    info = MODELS.get(model)
    if src.kind == "text":
        text = _script("encode_text_vectors")
        ids, docs, qtext, qrels = text.read_beir("scifact", None)
        by_id = dict(zip(ids, docs, strict=True))
        pairs = [(by_id[d], qtext[q]) for d, q in text.choose_check_pairs(qrels, qtext, CHECK_PAIRS)]
        return check_retriever(adapter, [d for d, _ in pairs], [q for _, q in pairs],
                               expected_dim=info.dimension if info else None)
    pairs = check_pairs(src)
    if len(pairs) < 6:
        return {"ok": False, "reason": f"only {len(pairs)} distinct labelled pairs"}
    pages = [src.item(i) for i, _ in pairs]
    if isinstance(adapter, PageEncoderAdapter):
        pages = [(PageRef(f"check{i}", 1), im) for i, im in enumerate(pages)]
    report = check_retriever(adapter, pages, [q for _, q in pairs], expected_dim=info.dimension if info else None)
    report["pages"] = [src.ids[i] for i, _ in pairs]
    return report


def encode_job(adapter: Any, src: Source, folder: Path, session: Session, keep_float32: bool, log: list) -> str:
    """Encode every page/document into shards (resuming), then the queries and the probes."""
    shard = SHARD_PAGES if src.kind == "image" else TEXT_SHARD_DOCS
    timings = []
    for lo in range(0, len(src), shard):
        hi = min(lo + shard, len(src))
        path = folder / "docs_f16" / f"shard_{lo:06d}.npz"
        if path.exists() and (not keep_float32 or (folder / "docs_f32" / path.name).exists()):
            continue
        need = 2 * (sum(timings) / len(timings)) if timings else 120
        if session.remaining() < need:
            log.append(f"stopped at page {lo} of {len(src)}: {session.remaining():.0f} s left")
            return "partial"
        t = time.time()
        corpus = encode_items(adapter, src.items(lo, hi), src.ids[lo:hi], src.kind)
        if list(corpus.ids) != src.ids[lo:hi]:
            corpus.ids = src.ids[lo:hi]
        if keep_float32:
            save_corpus(folder / "docs_f32" / path.name, corpus, "float32")
        save_corpus(path, corpus, "float16")
        timings.append(time.time() - t)
        if (lo // shard) % 10 == 0:
            print(f"   {src.key}: {hi}/{len(src)} ({timings[-1] / (hi - lo):.2f} s/item)", flush=True)
    qpath = folder / "queries_f16.npz"
    if not qpath.exists():
        qs = adapter.encode_queries([t for _, t in src.queries], ids=[q for q, _ in src.queries])
        save_corpus(qpath, qs)
    corpus_dir = folder.parent / "_corpus" / src.key
    for source in units.PROBE_SOURCES:
        probes_file = corpus_dir / f"probes_{source}.json"
        out = folder / f"probes_{source}_f16.npz"
        if not probes_file.exists() or out.exists():
            continue
        probes = json.loads(probes_file.read_text(encoding="utf-8"))
        if probes:
            qs = adapter.encode_queries([p["text"] for p in probes], ids=[p["id"] for p in probes])
            save_corpus(out, qs)
    return "complete"


def job_folder(out: Path, model: str, corpus_key: str) -> Path:
    return out / f"{model.split('/')[-1]}__{corpus_key}"


def free_model(model: str) -> None:
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    shutil.rmtree(hub / f"models--{model.replace('/', '--')}", ignore_errors=True)


def run_model(job: dict, adapter: Any, sources: dict[str, Source], out: Path, session: Session) -> list[dict]:
    """Every corpus of one job with an already loaded adapter; one ``run.json`` per corpus."""
    records = []
    check: dict | None = None
    for key in job["corpora"]:
        src = sources[key]
        folder = job_folder(out, job["model"], key)
        folder.mkdir(parents=True, exist_ok=True)
        record = {"model": job["model"], "corpus": key, "job": job, "started_utc": _now(), "versions": versions(),
                  "encoder": encoder_info(adapter, job), "log": []}
        t0 = time.time()
        try:
            if check is None:  # once per loaded model, on its first corpus
                check = run_check(adapter, src, job["model"])
                write_json(folder / "check_retriever.json", check)
            record["check_retriever"] = check
            if not check.get("ok"):
                record["outcome"] = "check failed"
            else:
                record["outcome"] = encode_job(adapter, src, folder, session, key in job.get("keep_float32", []),
                                               record["log"])
        except Exception as exc:  # noqa: BLE001 -- one corpus failing must not end the batch
            record["outcome"] = f"error: {type(exc).__name__}: {exc}"[:800]
        record["seconds"] = round(time.time() - t0, 1)
        record["finished_utc"] = _now()
        record["shards"] = len(list((folder / "docs_f16").glob("shard_*.npz")))
        write_json(folder / "run.json", record)
        print(f"   {job['model']} on {key}: {record['outcome']} in {record['seconds'] / 60:.1f} min", flush=True)
        records.append(record)
        if check is not None and not check.get("ok"):
            for rest in job["corpora"][job["corpora"].index(key) + 1:]:
                records.append({"model": job["model"], "corpus": rest, "outcome": "check failed"})
            break
    return records


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def iter_sources(jobs: list[dict], data: Path, limit: int | None) -> Iterator[tuple[str, Source]]:
    seen = set()
    for job in jobs:
        for key in job.get("corpora", []):
            if key not in seen:
                seen.add(key)
                yield key, read_source(key, data, limit)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/tmp/optdata"))
    ap.add_argument("--deadline-hours", type=float, default=units.DEADLINE_HOURS)
    ap.add_argument("--limit", type=int, default=None, help="pages / documents per corpus, for a dry run")
    a = ap.parse_args(argv)
    spec = json.loads(a.jobs.read_text(encoding="utf-8"))
    jobs = spec["jobs"]
    session = Session(a.deadline_hours)
    a.out.mkdir(parents=True, exist_ok=True)
    a.data.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    sources: dict[str, Source] = {}
    for key, src in iter_sources(jobs, a.data, a.limit):
        sources[key] = src
        with_probes = units.corpus(key).text is not None and src.kind == "image"
        rec = prepare_corpus(src, a.out, with_probes)
        print(f"== corpus {key}: {len(src)} items, {len(src.queries)} queries; probes {rec.get('probes')}", flush=True)
    for job in jobs:
        if session.remaining() < 900:
            summary.append({"model": job["model"], "outcome": "not started (no time left)"})
            continue
        print(f"== {job['model']} ({job.get('role', 'encode')}); {session.remaining() / 3600:.2f} h left", flush=True)
        try:
            if job.get("role") == "heldout":
                from easy_validation import heldout

                summary += heldout.run(job, sources, a.out, session)
            elif job.get("role") == "wiring":
                from easy_validation import wiring_v6

                summary += wiring_v6.run(job, sources, a.out, session)
            elif job.get("role") == "vlm":
                from easy_validation import vlm_probes

                summary += vlm_probes.run(job, sources, a.out, session)
            else:
                for package in job.get("pip") or []:
                    os.system(f"{sys.executable} -m pip install -q {package}")
                adapter = load_model(job)
                summary += run_model(job, adapter, sources, a.out, session)
                del adapter
        except Exception as exc:  # noqa: BLE001 -- one model's failure must not end the batch
            summary.append({"model": job["model"], "outcome": f"error: {type(exc).__name__}: {exc}"[:800]})
            print(f"   -> error: {exc}", flush=True)
        free_model(job["model"])
        write_json(a.out / "batch_summary.json", summary)
    write_json(a.out / "batch_summary.json", summary)
    print(json.dumps(summary, indent=1, default=str)[:4000])
    print(f"session used {(time.time() - session.t0) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
