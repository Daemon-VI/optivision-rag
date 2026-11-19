"""Small stand-ins for the easy-mode modules written in parallel (docs/easy/CONTRACT.md).

``tests/test_easy_build.py`` and ``tests/test_easy_api.py`` test the orchestration in ``_build`` and
``_api``; the modules it calls (``_levels``, ``_docs``, ``_store``, ...) are replaced here by fakes that
keep to the contract signatures ``_build``'s adapters assume. ``install(monkeypatch)`` puts them in
``sys.modules``; the integrator's end-to-end tests run the real modules.

The fakes are deliberately plain: the store keeps segments as ``.npy`` files with the fitted pipeline
state (through the unchanged ``pipeline_fitted_state``), the searcher is ``CompressedRetriever`` per
segment, ``_docs`` renders with pypdfium2 at a low resolution, and the stub encoder embeds each word of
a page's text layer (handed over in ``image.info["ov_text"]``) as a seeded random unit vector, so a
query finds the page that holds its words.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import types
from collections import namedtuple
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from optivision.adapters import CallableAdapter, ModelInfo
from optivision.compose import CompressedCorpus, Pipeline
from optivision.easy._types import IndexLocked, LayoutRow, Selection, UnknownModel
from optivision.representation import MultiVectorCorpus

DIM = 16
WORD = re.compile(r"[a-z0-9]{2,}")


# ============================================================================ stub encoder


def word_vector(word: str, dim: int = DIM) -> np.ndarray:
    seed = int(hashlib.sha256(word.encode()).hexdigest()[:12], 16)
    v = np.random.default_rng(seed).standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def text_vectors(text: str, cap: int = 48) -> np.ndarray:
    words = list(dict.fromkeys(WORD.findall(text.lower())))[:cap]
    if not words:
        words = ["emptypage"]
    return np.stack([word_vector(w) for w in words])


def _payload_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    info = getattr(item, "info", {}) or {}
    if "ov_text" in info:
        return str(info["ov_text"]) or "image" + hashlib.sha256(item.tobytes()).hexdigest()[:8]
    return "image" + hashlib.sha256(item.tobytes()).hexdigest()[:8]


@dataclass
class Calls:
    documents: int = 0
    pages: int = 0
    queries: int = 0
    fail_after: int | None = None  # raise on the document call after this many


def stub_adapter(calls: Calls | None = None) -> CallableAdapter:
    calls = calls if calls is not None else Calls()

    def docs(items: list[Any]) -> list[np.ndarray]:
        if calls.fail_after is not None and calls.documents >= calls.fail_after:
            raise RuntimeError("simulated crash in the middle of encoding")
        calls.documents += 1
        calls.pages += len(items)
        return [text_vectors(_payload_text(x)) for x in items]

    def queries(items: list[Any]) -> list[np.ndarray]:
        calls.queries += 1
        return [text_vectors(str(q)) for q in items]

    adapter = CallableAdapter(docs, queries)
    adapter.calls = calls  # type: ignore[attr-defined]
    return adapter


# ============================================================================ _resolve

TEXT_MODEL = ModelInfo("fake/text-model", "fake", "text", "callable", DIM, "measured", "fake rows")
PAGE_MODEL = ModelInfo("fake/page-model", "fake", "image", "callable", DIM, "measured", "fake rows")


def make_resolve() -> types.ModuleType:
    mod = types.ModuleType("optivision.easy._resolve")
    models = {m.model_id: m for m in (TEXT_MODEL, PAGE_MODEL)}

    def resolve(model: str | None) -> ModelInfo | None:
        if model is None:
            return None
        if model in models:
            return models[model]
        raise UnknownModel(f"unknown model {model!r}; closest: {sorted(models)}")

    mod.resolve = resolve  # type: ignore[attr-defined]
    return mod


# ============================================================================ _levels

SPECS = {
    "float32": {"stages": []},
    "float16": {"stages": [{"stage": "float16"}]},
    "int8(per_vector)": {"stages": [{"scale": "per_vector", "stage": "int8"}]},
    "hierarchical_merge(0.25) > int8(per_vector)": {"stages": [
        {"max_distance": None, "min_vectors": 1, "ratio": 0.25, "stage": "hierarchical_merge"},
        {"scale": "per_vector", "stage": "int8"}]},
    "int4(mean)": {"stages": [{"center": "mean", "stage": "int4"}]},
    "binary": {"stages": [{"stage": "binary"}]},
}
#: name: (plain, chain, ratio_f16, lowest_kept, one_bit, merged)
LADDER = {
    "float32": ("float32", "float", 0.5, 1.0, False, False),
    "float16": ("float16", "float", 1.0, 1.0, False, False),
    "int8(per_vector)": ("8-bit", "8-bit", 2.0, 0.99, False, False),
    "hierarchical_merge(0.25) > int8(per_vector)": ("merged (1 in 4 vectors) + 8-bit", "8-bit", 7.9, 0.972,
                                                    False, True),
    "int4(mean)": ("centred 4-bit", "centred 4-bit", 3.9, 0.975, False, False),
    "binary": ("1-bit", "1-bit", 16.0, 0.95, True, False),
}
DATASETS = ("ViDoRe V1 DocVQA 500 pages", "InfoVQA 500 pages")


def make_levels(**overrides: float) -> types.ModuleType:
    """``select`` with a fake six-row ladder; ``overrides`` replace lowest-kept values by row name."""
    mod = types.ModuleType("optivision.easy._levels")
    mod.calls = []  # type: ignore[attr-defined]

    def rows_at(target: float, checked: bool, vpp: float | None) -> list[LayoutRow]:
        rows = []
        for name, (plain, chain, ratio, low, one_bit, merged) in LADDER.items():
            low = overrides.get(name, low)
            need = target + (0.025 if one_bit else 0.0)
            status, reason = ("qualifies", "") if low >= need else ("below target", "below its needed line")
            if one_bit and not checked:
                status, reason = "not allowed", "needs practice questions"
            if merged and vpp is not None and vpp < 3:
                status, reason = "outside size range", "outside the sizes we measured"
            rows.append(LayoutRow(name=name, plain=plain, spec=SPECS[name], chain=chain,
                                  bytes_per_page=(1000.0 / ratio, 1100.0 / ratio), ratio_f16=ratio,
                                  ratio_f32=2 * ratio, lowest_kept=low, worst_point=low + 0.01, datasets=DATASETS,
                                  status=status, reason=reason, need=need, slack=low - need, one_bit=one_bit,
                                  merged=merged))
        return rows

    def select(model: str | None, target: float, n_pages: int | None = None, *, checked: bool = False,
               modality: str | None = None, vectors_per_page: float | None = None,
               layout: str | None = None) -> Selection:
        mod.calls.append({"model": model, "target": target, "n_pages": n_pages, "checked": checked,
                          "modality": modality, "vectors_per_page": vectors_per_page, "layout": layout})
        rows = rows_at(target, checked, vectors_per_page)
        ok = [r for r in rows if r.status == "qualifies" and r.chain != "float"]
        f16 = next(r for r in rows if r.name == "float16")
        pick = max(ok, key=lambda r: r.ratio_f16) if ok else f16
        hand = False
        if layout is not None:
            pick = next(r for r in rows if layout.lower() in (r.name.lower(), r.plain.lower()))
            hand = True
        close = pick.slack is not None and pick.slack < 0.005 and not hand and pick.chain != "float"
        safer_rows = sorted([r for r in ok if r.slack >= 0.005 and r.ratio_f16 < pick.ratio_f16],
                            key=lambda r: -r.ratio_f16)
        safer = safer_rows[0] if close and safer_rows else None
        seq = [pick] + sorted([r for r in ok if r.ratio_f16 < pick.ratio_f16], key=lambda r: -r.ratio_f16)
        if seq[-1].name != "float16":
            seq.append(f16)
        rows = [replace(r, status="pick") if r.name == pick.name else r for r in rows]
        pick = next(r for r in rows if r.name == pick.name)
        seq[0] = pick
        return Selection(model=model, modality=modality or "image", target=target, n_pages=n_pages, checked=checked,
                         rows=tuple(rows), pick=pick, sequence=tuple(seq), close_call=close, safer=safer,
                         basis="own rows" if model else "unmeasured: pooled minimum over 40 page models",
                         hand_picked=hand)

    mod.select = select  # type: ignore[attr-defined]
    return mod


# ============================================================================ _docs


@dataclass(frozen=True)
class FileRecord:
    path: str
    relpath: str
    doc_id: str
    size: int
    mtime: float
    kind: str


@dataclass(frozen=True)
class PageRecord:
    id: str
    doc_id: str
    file: str
    relpath: str
    page: int
    text: str
    n_chars: int
    blank: bool
    sha256: str
    landscape: bool


PageCount = namedtuple("PageCount", "n_pages n_files skipped")
FolderProfile = namedtuple("FolderProfile", "n_pages n_text_pages n_blank aliases boilerplate landscape_share "
                                            "text_share checkable reason")
IMAGES = (".png", ".jpg", ".jpeg")


def make_docs() -> types.ModuleType:
    mod = types.ModuleType("optivision.easy._docs")
    mod.opened = []  # type: ignore[attr-defined]

    def list_source(source: Any) -> tuple[list[FileRecord], list[tuple[str, str]]]:
        p = Path(source)
        root = p if p.is_dir() else p.parent
        paths = sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
        files, skipped = [], []
        for q in paths:
            if q.suffix.lower() not in (".pdf", *IMAGES):
                continue
            st = q.stat()
            if st.st_size == 0:
                skipped.append((str(q), "empty file"))
                continue
            rel = q.relative_to(root).as_posix()
            files.append(FileRecord(str(q), rel, f"{root.name}/{rel}", st.st_size, st.st_mtime,
                                    "pdf" if q.suffix.lower() == ".pdf" else "image"))
        return files, skipped

    def count_pages(source: Any, progress: Any = None) -> PageCount:
        import pypdfium2 as pdfium

        files, skipped = list_source(source)
        n = 0
        for f in files:
            if f.kind == "image":
                n += 1
                continue
            try:
                doc = pdfium.PdfDocument(f.path)
            except Exception as exc:  # noqa: BLE001 - any unreadable PDF is listed as skipped
                skipped.append((f.path, str(exc)))
                continue
            n += len(doc)
            doc.close()
        return PageCount(n, len(files), skipped)

    def iter_pages(file: FileRecord, *, dpi: int = 150, max_side: int = 1536) -> Any:
        import pypdfium2 as pdfium
        from PIL import Image

        mod.opened.append(file.path)
        if file.kind == "image":
            img = Image.open(file.path).convert("RGB")
            img.info["ov_text"] = ""
            sha = hashlib.sha256(img.tobytes()).hexdigest()
            yield PageRecord(f"{file.doc_id}::p1", file.doc_id, file.path, file.relpath, 1, "", 0, False, sha,
                             img.width > img.height), img
            return
        doc = pdfium.PdfDocument(file.path)  # raises on a corrupt or encrypted file
        try:
            for i in range(len(doc)):
                page = doc[i]
                text = page.get_textpage().get_text_range().strip()
                img = page.render(scale=24 / 72).to_pil().convert("RGB")
                img.info["ov_text"] = text
                arr = np.asarray(img)
                blank = not text and float((arr < 128).mean()) < 0.001
                sha = hashlib.sha256(img.tobytes()).hexdigest()
                yield PageRecord(f"{file.doc_id}::p{i + 1}", file.doc_id, file.path, file.relpath, i + 1, text,
                                 len(text), blank, sha, img.width > img.height), img
        finally:
            doc.close()

    def render_page(path: Any, page: int, dpi: int = 150) -> Any:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
        try:
            return doc[page - 1].render(scale=dpi / 72).to_pil()
        finally:
            doc.close()

    def profile(pages: Any, vectors_per_page: Any = None) -> FolderProfile:
        pages = list(pages)
        n = len(pages)
        text = sum(1 for p in pages if p.n_chars >= 20)
        blank = sum(1 for p in pages if p.blank)
        share = text / max(1, n - blank)
        ok = text >= 60 and share >= 0.5
        return FolderProfile(n, text, blank, {}, [], 0.0, share, ok,
                             "" if ok else "under 60 text pages, so no practice check")

    for fn in (list_source, count_pages, iter_pages, render_page, profile):
        setattr(mod, fn.__name__, fn)
    return mod


# ============================================================================ _probes, _veto

Probe = namedtuple("Probe", "id text family page_id relevant")
HardSet = namedtuple("HardSet", "probes sha256")
CheckStat = namedtuple("CheckStat", "R U n passed")
SequenceResult = namedtuple("SequenceResult", "chosen steps")


def make_probes_mod() -> types.ModuleType:
    mod = types.ModuleType("optivision.easy._probes")

    def make_probes(pages: Any, *, seed: int = 0, max_pages: int = 600, per_page: int = 3) -> list[Probe]:
        pages = [p for p in pages if p.n_chars >= 20]
        words = {p.id: set(WORD.findall(p.text.lower())) for p in pages}
        out = []
        for p in pages[:max_pages]:
            w = list(dict.fromkeys(WORD.findall(p.text.lower())))[:4]
            rel = frozenset(q for q, ws in words.items() if set(w) <= ws)
            out.append(Probe(f"probe{len(out)}", " ".join(w), "keywords", p.id, rel))
        return out

    def hard_set(probes: Any, ref_rankings: Any, *, k_keep: int = 10) -> HardSet:
        kept = [p for p in probes if any(i in p.relevant for i, _ in ref_rankings[p.id][:k_keep])]
        return HardSet(kept, hashlib.sha256(json.dumps([p.id for p in kept]).encode()).hexdigest())

    mod.make_probes = make_probes  # type: ignore[attr-defined]
    mod.hard_set = hard_set  # type: ignore[attr-defined]
    return mod


def ndcg_at(ranking_ids: Any, relevant: Any, k: int = 5) -> float:
    gains = [1.0 / np.log2(i + 2) for i, d in enumerate(list(ranking_ids)[:k]) if d in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return float(sum(gains) / ideal) if ideal else 0.0


def make_veto(fail: tuple[str, ...] = ()) -> types.ModuleType:
    """``fixed_sequence`` testing rows in order; rows named in ``fail`` are made to fall below the target."""
    mod = types.ModuleType("optivision.easy._veto")
    mod.sequences = []  # type: ignore[attr-defined]

    def practice_check(cand: Any, ref: Any, relevant: Any, clusters: Any, target: float, *, B: int = 2000,
                       seed: int = 0) -> CheckStat:
        c = sum(ndcg_at(cand[q], relevant[q]) for q in cand)
        b = sum(ndcg_at(ref[q], relevant[q]) for q in cand)
        r = c / b if b else 1.0
        return CheckStat(r, r, len(cand), r >= target)

    def fixed_sequence(sequence: Any, rank_fn: Any, ref: Any, relevant: Any, clusters: Any, target: float) -> Any:
        mod.sequences.append([r.name for r in sequence])
        steps = []
        for row in sequence:
            stat = practice_check(rank_fn(row), ref, relevant, clusters, target)
            if row.name in fail:
                stat = CheckStat(0.5, 0.6, stat.n, False)
            steps.append((row, stat))
            if stat.passed:
                return SequenceResult(row, steps)
        return SequenceResult(sequence[-1], steps)

    def strength(n: int, target: float) -> dict[str, float]:
        return {"fall": 0.90, "caught": 0.74}

    mod.practice_check = practice_check  # type: ignore[attr-defined]
    mod.fixed_sequence = fixed_sequence  # type: ignore[attr-defined]
    mod.ndcg_at = ndcg_at  # type: ignore[attr-defined]
    mod.strength = strength  # type: ignore[attr-defined]
    return mod


# ============================================================================ _encode

Wiring = namedtuple("Wiring", "ok hits n used_sample message")
Estimate = namedtuple("Estimate", "lo_s hi_s basis gpu")


@dataclass
class Encoder:
    adapter: Any
    info: Any = None
    record: dict[str, Any] = field(default_factory=dict)

    def encode_pages(self, images: Any, ids: Any, progress: Any = None) -> MultiVectorCorpus:
        return self.adapter.encode_documents(list(images), ids=list(ids))

    def encode_queries(self, texts: Any) -> MultiVectorCorpus:
        texts = list(texts)
        return self.adapter.encode_queries(texts, ids=[f"q{i}" for i in range(len(texts))])


def make_encode(adapters: dict[str, Any] | None = None) -> types.ModuleType:
    """``load_encoder`` over an adapter object, or over a fake registry id mapped in ``adapters``."""
    mod = types.ModuleType("optivision.easy._encode")
    mod.loads = []  # type: ignore[attr-defined]
    mod.wirings = []  # type: ignore[attr-defined]
    mod.dtype = "float32"  # type: ignore[attr-defined]

    def load_encoder(model: Any, *, device: str = "auto", dtype: str = "recorded",
                     trust_remote_code: bool = False) -> Encoder:
        mod.loads.append({"model": model, "device": device, "dtype": dtype})
        adapter = (adapters or {}).get(model, model) if isinstance(model, str) else model
        if isinstance(adapter, str):
            raise KeyError(f"no fake adapter for {model!r}")
        return Encoder(adapter, None, {"model": model if isinstance(model, str) else "adapter",
                                       "dtype": mod.dtype, "device": "cpu", "loader": "callable",
                                       "revision": "abc123"})

    def wiring_check(encoder: Any, pages: Any) -> Wiring:
        mod.wirings.append(len(pages))
        return Wiring(True, len(pages), len(pages), False, "")

    def estimate(model_id: str, n_pages: int) -> Estimate:
        return Estimate(0.61 * n_pages + 60, 0.65 * n_pages + 60, "our Kaggle run, 0.61-0.65 s/page", "one T4")

    def recorded_dtype(model_id: str) -> str | None:
        return "float32"

    for fn in (load_encoder, wiring_check, estimate, recorded_dtype):
        setattr(mod, fn.__name__, fn)
    return mod


# ============================================================================ _store


def _save_state(folder: Path, pipeline: Pipeline) -> None:
    from optivision.core.stages.state import pipeline_fitted_state

    folder.mkdir(parents=True, exist_ok=True)
    states = pipeline_fitted_state(pipeline)
    arrays, scalars = {}, []
    for i, st in enumerate(states):
        sc = {}
        for k, v in st.items():
            if isinstance(v, np.ndarray):
                arrays[f"{i}.{k}"] = v
            else:
                sc[k] = v
        scalars.append(sc)
    (folder / "pipeline.json").write_text(json.dumps({"spec": pipeline.to_dict(), "scalars": scalars}),
                                          encoding="utf-8")
    np.savez(folder / "fitted.npz", **arrays)


def _load_state(folder: Path) -> Pipeline:
    from optivision.core.stages.state import restore_pipeline_state

    d = json.loads((folder / "pipeline.json").read_text(encoding="utf-8"))
    p = Pipeline.from_dict(d["spec"])
    with np.load(folder / "fitted.npz") as z:
        arrays = {k: z[k] for k in z.files}
    states = []
    for i, sc in enumerate(d["scalars"]):
        st = dict(sc)
        st.update({k.split(".", 1)[1]: v for k, v in arrays.items() if k.split(".", 1)[0] == str(i)})
        states.append(st)
    return restore_pipeline_state(p, states)


class FakeStore:
    """An on-disk store: ``manifest.json`` written last (``os.replace``); segments listed only after commit."""

    def __init__(self, path: Path, mode: str) -> None:
        self.path = Path(path)
        self.mode = mode
        m = self.path / "manifest.json"
        self.manifest: dict[str, Any] = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
        self._segments: list[str] = list(self.manifest.get("_segments", []))
        self._pipeline: Pipeline | None = None
        self._reference: Any = None
        self.closed = False

    @classmethod
    def create(cls, path: Any, *, pipeline: Pipeline, info: dict[str, Any]) -> FakeStore:
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        _save_state(path / "state", pipeline)
        (path / "build.json").write_text(json.dumps(info, default=str), encoding="utf-8")
        return cls(path, "a")

    @classmethod
    def open(cls, path: Any, mode: str = "r") -> FakeStore:
        path = Path(path)
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"{path} has no manifest")
        return cls(path, mode)

    def _need_write(self) -> None:
        if self.mode != "a":
            raise PermissionError("store opened read-only")

    def append(self, compressed: CompressedCorpus, texts: Any, float16: Any) -> None:
        self._need_write()
        name = f"{len(os.listdir(self.path / 'segments')) if (self.path / 'segments').exists() else 0:03d}"
        seg = self.path / "segments" / name
        seg.mkdir(parents=True)
        np.save(seg / "codes.npy", compressed.codes)
        np.save(seg / "offsets.npy", compressed.offsets)
        (seg / "ids.json").write_text(json.dumps({"ids": compressed.ids, "dim": compressed.dim}), encoding="utf-8")
        if texts is not None:
            (seg / "text.json").write_text(json.dumps(texts), encoding="utf-8")
        if float16 is not None:
            np.save(seg / "f16.npy", np.asarray(float16.vectors, dtype=np.float16))
            np.save(seg / "f16_offsets.npy", float16.offsets)
        self._segments.append(name)

    def remove(self, ids: Any) -> None:
        self._need_write()
        gone = set(ids)
        pipe = self.pipeline()
        keep = []
        for name in self._segments:
            seg = self.path / "segments" / name
            meta = json.loads((seg / "ids.json").read_text(encoding="utf-8"))
            sel = [i for i, x in enumerate(meta["ids"]) if x not in gone]
            if len(sel) == len(meta["ids"]):
                keep.append(name)
                continue
            if not sel:
                continue
            codes, off = np.load(seg / "codes.npy"), np.load(seg / "offsets.npy")
            rows = [codes[off[i]:off[i + 1]] for i in sel]
            new = seg.parent / f"{name}r"
            new.mkdir()
            np.save(new / "codes.npy", np.concatenate(rows))
            np.save(new / "offsets.npy", np.concatenate([[0], np.cumsum([r.shape[0] for r in rows])]))
            (new / "ids.json").write_text(json.dumps({"ids": [meta["ids"][i] for i in sel], "dim": meta["dim"]}),
                                          encoding="utf-8")
            if (seg / "text.json").exists():
                t = json.loads((seg / "text.json").read_text(encoding="utf-8"))
                (new / "text.json").write_text(json.dumps({meta["ids"][i]: t[meta["ids"][i]] for i in sel}),
                                               encoding="utf-8")
            keep.append(new.name)
        self._segments = keep
        del pipe

    def commit(self, **updates: Any) -> None:
        self._need_write()
        bad = sorted(set(updates) & {"format", "version", "created", "updated", "generation", "dim", "pipeline",
                                     "segments", "next_segment", "n_pages", "n_vectors", "float16", "files"})
        if bad:  # the real Store refuses these: it maintains them itself
            raise ValueError(f"commit() may not set {bad}: the store maintains those manifest keys itself")
        self.manifest.update(updates)
        self.manifest["_segments"] = list(self._segments)
        self.manifest["n_pages"] = self.n_pages
        dims = [json.loads((self.path / "segments" / n / "ids.json").read_text(encoding="utf-8"))["dim"]
                for n in self._segments]
        self.manifest["dim"] = dims[0] if dims else self.manifest.get("dim")
        tmp = self.path / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, default=str), encoding="utf-8")
        os.replace(tmp, self.path / "manifest.json")

    @property
    def ids(self) -> list[str]:
        out = []
        for name in self._segments:
            out += json.loads((self.path / "segments" / name / "ids.json").read_text(encoding="utf-8"))["ids"]
        return out

    @property
    def n_pages(self) -> int:
        return len(self.ids)

    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = _load_state(self.path / "state")
        return self._pipeline

    def segments(self) -> list[CompressedCorpus]:
        pipe = self.pipeline()
        out = []
        for name in self._segments:
            seg = self.path / "segments" / name
            meta = json.loads((seg / "ids.json").read_text(encoding="utf-8"))
            out.append(CompressedCorpus(np.load(seg / "codes.npy"), np.load(seg / "offsets.npy"), meta["dim"],
                                        pipe.quantizer, meta["ids"]))
        return out

    def float16_corpus(self) -> MultiVectorCorpus | None:
        parts = []
        for name in self._segments:
            seg = self.path / "segments" / name
            if not (seg / "f16.npy").exists():
                return None
            meta = json.loads((seg / "ids.json").read_text(encoding="utf-8"))
            parts.append(MultiVectorCorpus(np.load(seg / "f16.npy"), np.load(seg / "f16_offsets.npy"),
                                           ids=meta["ids"]))
        if not parts:
            return None
        vec = np.concatenate([p.vectors for p in parts])
        off = np.concatenate([[0], np.cumsum(np.concatenate([p.counts for p in parts]))])
        return MultiVectorCorpus(vec, off, ids=[i for p in parts for i in p.ids])

    def text(self, page_id: str) -> str:
        for name in self._segments:
            f = self.path / "segments" / name / "text.json"
            if f.exists():
                t = json.loads(f.read_text(encoding="utf-8"))
                if page_id in t:
                    return t[page_id]
        return ""

    @property
    def reference(self) -> Any:
        return self._reference

    @reference.setter
    def reference(self, value: Any) -> None:
        self._reference = value
        (self.path / "reference").mkdir(exist_ok=True)
        (self.path / "reference" / "probes.json").write_text(json.dumps(value["probes"]), encoding="utf-8")

    def sizes(self) -> dict[str, int]:
        out = {"vectors": 0, "text": 0, "float16": 0, "record": 0, "reference": 0}
        for f in self.path.rglob("*"):
            if not f.is_file():
                continue
            n = f.stat().st_size
            if f.name in ("codes.npy", "offsets.npy", "fitted.npz"):
                out["vectors"] += n
            elif f.name == "text.json":
                out["text"] += n
            elif f.name.startswith("f16"):
                out["float16"] += n
            elif "reference" in f.parts:
                out["reference"] += n
            else:
                out["record"] += n
        out["total"] = sum(out.values())
        return out

    @contextlib.contextmanager
    def lock(self) -> Any:
        f = self.path / ".writer.lock"
        try:
            fd = os.open(str(f), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise IndexLocked(f"{self.path} is locked") from None
        os.close(fd)
        try:
            yield self
        finally:
            f.unlink()

    def compact(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def make_store() -> types.ModuleType:
    mod = types.ModuleType("optivision.easy._store")
    mod.Store = FakeStore  # type: ignore[attr-defined]
    return mod


# ============================================================================ _index


def _ranking_lists(ranking: Any) -> list[list[tuple[str, float]]]:
    return [[(ranking.ids[q][j], float(ranking.scores[q][j])) for j in range(len(ranking.ids[q]))]
            for q in range(ranking.n_queries)]


def make_index() -> types.ModuleType:
    from optivision.core.retrieve.exact import CompressedRetriever, ExactRetriever

    mod = types.ModuleType("optivision.easy._index")

    class Searcher:
        def __init__(self, store: FakeStore) -> None:
            self.pipeline = store.pipeline()
            self.segments = store.segments()

        def search(self, queries: MultiVectorCorpus, k: int) -> list[list[tuple[str, float]]]:
            merged: list[list[tuple[float, int, str]]] = [[] for _ in range(len(queries))]
            pos = 0
            for seg in self.segments:
                r = CompressedRetriever(seg, self.pipeline, max_block_bytes=32 * 2**20).search(queries,
                                                                                            min(k, len(seg)))
                for q, hits in enumerate(_ranking_lists(r)):
                    merged[q] += [(s, pos + seg.ids.index(i), i) for i, s in hits]
                pos += len(seg)
            return [[(i, s) for s, _, i in sorted(h, key=lambda t: (-t[0], t[1]))[:k]] for h in merged]

        rank_all = search

    def rank_float(corpus: MultiVectorCorpus, queries: MultiVectorCorpus, k: int,
                   pipeline: Pipeline | None = None) -> list[list[tuple[str, float]]]:
        if pipeline is None:
            return _ranking_lists(ExactRetriever(corpus).search(queries, k))
        store = pipeline.compress(corpus, fit=False)
        return _ranking_lists(CompressedRetriever(store, pipeline).search(queries, k))

    mod.Searcher = Searcher  # type: ignore[attr-defined]
    mod.rank_float = rank_float  # type: ignore[attr-defined]
    return mod


# ============================================================================ delegated modules


def make_recorder(name: str, attr: str, result: Any = "done") -> types.ModuleType:
    mod = types.ModuleType(f"optivision.easy.{name}")
    mod.calls = []  # type: ignore[attr-defined]

    def fn(*args: Any, **kwargs: Any) -> Any:
        mod.calls.append((args, kwargs))
        return result

    setattr(mod, attr, fn)
    return mod


class FakeReport:
    def __init__(self, sections: Any, meta: Any) -> None:
        self.sections = sections
        self.meta = meta

    def save(self, path: Any) -> Path:
        Path(path).write_text("\n\n".join(f"{t}\n{b}" for t, b in self.sections), encoding="utf-8")
        return Path(path)


def install(monkeypatch: Any, *, levels: types.ModuleType | None = None, veto: types.ModuleType | None = None,
            encode: types.ModuleType | None = None) -> dict[str, types.ModuleType]:
    """Put every fake in ``sys.modules`` for the test; returns them by short name."""
    report = types.ModuleType("optivision.easy._report")
    report.Report = FakeReport  # type: ignore[attr-defined]
    budget = types.ModuleType("optivision.easy._budget")
    budget.highest_target_that_fits = lambda model, n_pages, max_disk, **kw: 0.95  # type: ignore[attr-defined]
    integ = types.ModuleType("optivision.easy.integrations")
    integ.__path__ = []  # type: ignore[attr-defined]
    mods = {
        "_resolve": make_resolve(), "_levels": levels or make_levels(), "_docs": make_docs(),
        "_probes": make_probes_mod(), "_veto": veto or make_veto(), "_store": make_store(), "_index": make_index(),
        "_encode": encode or make_encode(), "_check": make_recorder("_check", "run_check", "check report"),
        "_upgrade": make_recorder("_upgrade", "run_upgrade", "upgrade report"), "_report": report,
        "_budget": budget, "_querylog": make_recorder("_querylog", "open_log", None),
        "integrations": integ, "integrations.demo": make_recorder("integrations.demo", "launch", "demo"),
        "integrations.qdrant": make_recorder("integrations.qdrant", "export", "qdrant"),
    }
    for short, mod in mods.items():
        monkeypatch.setitem(sys.modules, f"optivision.easy.{short}", mod)
    return mods


# ============================================================================ PDFs


def make_pdf(path: Path, pages: list[str], *, encrypt: str | None = None) -> Path:
    """A PDF with one sentence per page ("" gives a blank page)."""
    from reportlab.lib.pagesizes import A6
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A6, encrypt=encrypt)
    for text in pages:
        if text:
            c.setFont("Helvetica", 9)
            y = 380
            for chunk in [text[i:i + 45] for i in range(0, len(text), 45)]:
                c.drawString(18, y, chunk)
                y -= 12
        c.showPage()
    c.save()
    return path


SENTENCES = [
    "Operating costs rose because freight prices doubled in the second quarter",
    "The board approved a new dividend policy for shareholders next year",
    "Annual revenue grew modestly while marketing spending stayed flat",
    "Warehouse automation reduced picking errors across northern sites",
    "A cyber incident delayed invoicing systems during the spring",
    "Employee training hours increased after the safety review",
]
