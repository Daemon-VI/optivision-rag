"""Small fakes for the tests of optivision.easy._check, _upgrade, _spotcheck and _budget.

The index modules (_index, _store, _levels, _veto, _encode) are written against the contract by
other modules; these fakes implement only the parts of it the checks use, over a synthetic
corpus with the real ``Pipeline`` and ``CompressedRetriever``. Synthetic data exercise code
paths; nothing here says anything about a real encoder.
"""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import optivision.easy as easy_pkg
from optivision.compose import Pipeline
from optivision.core.retrieve.exact import CompressedRetriever
from optivision.easy._types import Hit, LayoutRow, Selection
from optivision.metrics import ndcg_at_k
from optivision.representation import MultiVectorCorpus
from optivision.testing.synthetic import make_corpus, make_queries

# ---------------------------------------------------------------------------- fake ladder

#: name, plain, spec, chain, ratio vs float16, lowest kept, bytes per page (lo, hi)
RUNGS = (
    ("float32", "float32", {"stages": []}, "float", 0.5, 1.0, (4000.0, 4200.0)),
    ("float16", "float16", {"stages": [{"stage": "float16"}]}, "float", 1.0, 0.999, (2000.0, 2100.0)),
    ("int8(per_vector)", "8-bit", {"stages": [{"scale": "per_vector", "stage": "int8"}]}, "8-bit", 1.97,
     0.985, (1000.0, 1050.0)),
    ("int4(mean)", "centred 4-bit", {"stages": [{"center": "mean", "stage": "int4"}]}, "centred 4-bit", 3.9,
     0.96, (500.0, 525.0)),
    ("binary", "1-bit", {"stages": [{"stage": "binary"}]}, "1-bit", 15.9, 0.93, (130.0, 140.0)),
)
SIZE_STEP = 1000      # above this many pages the fake rule adds 0.02 to every compressed rung's need
DATASETS = ("ViDoRe V1 DocVQA 500 pages", "InfoVQA 500 pages")


def fake_select(model: str | None, target: float, n_pages: int | None = None, *, checked: bool = False,
                modality: str | None = None, vectors_per_page: float | None = None,
                layout: str | None = None) -> Selection:
    rows = []
    for name, plain, spec, chain, ratio, low, bpp in RUNGS:
        need = target + (0.02 if (n_pages or 0) > SIZE_STEP and chain != "float" else 0.0)
        if chain == "1-bit" and not checked:
            status = "needs practice questions"
        else:
            status = "qualifies" if low >= need else "below target"
        rows.append(LayoutRow(name, plain, spec, chain, bpp, ratio, ratio * 2, low, low - 0.01, DATASETS,
                              status, "", need, low - need, chain == "1-bit", False))
    qualifying = [r for r in rows if r.status == "qualifies"]
    pick = max(qualifying, key=lambda r: r.ratio_f16) if qualifying else rows[0]
    rows = [replace(r, status="pick") if r.name == pick.name else r for r in rows]
    pick = next(r for r in rows if r.name == pick.name)
    safer = sorted((r for r in rows if r.status in ("pick", "qualifies") and r.ratio_f16 < pick.ratio_f16
                    and r.chain != "float"), key=lambda r: -r.ratio_f16)
    f16 = next(r for r in rows if r.name == "float16")
    sequence = tuple([pick, *safer] + ([f16] if pick.name != "float16" else []))
    return Selection(model, modality or "page", target, n_pages, checked, tuple(rows), pick, sequence)


def fake_row(name: str) -> LayoutRow:
    return next(r for r in fake_select("fake/model", 0.97, 40).rows if r.name == name)


def fake_practice_check(cand, ref, relevant, clusters, target, *, B=2000, seed=0):
    """R = sum of candidate nDCG@5 over sum of reference nDCG@5; passes when R >= target (the
    real one adds a page-cluster bootstrap bound)."""
    c = sum(ndcg_at_k(cand[q], relevant[q], 5) for q in cand)
    b = sum(ndcg_at_k(ref[q], relevant[q], 5) for q in cand)
    r = c / b if b else 0.0
    return types.SimpleNamespace(R=r, U=r, n=len(cand), passed=r >= target)


def install_fakes(monkeypatch) -> None:
    """Put the fake _levels and _veto in place of whatever is installed (sys.modules and the
    package attribute, which ``from . import _levels`` reads first)."""
    levels = types.ModuleType("optivision.easy._levels")
    levels.select = fake_select
    veto = types.ModuleType("optivision.easy._veto")
    veto.practice_check = fake_practice_check
    for name, module in (("_levels", levels), ("_veto", veto)):
        monkeypatch.setitem(sys.modules, f"optivision.easy.{name}", module)
        monkeypatch.setattr(easy_pkg, name, module, raising=False)


# ---------------------------------------------------------------------------- corpus


def page_ids(n_files: int = 10, pages: int = 4) -> list[str]:
    return [f"docs/report{j:02d}.pdf::p{p}" for j in range(n_files) for p in range(1, pages + 1)]


def synthetic(n_files: int = 10, pages: int = 4, dim: int = 32, seed: int = 0) -> MultiVectorCorpus:
    base = make_corpus(n_files * pages, dim, tokens=(8, 14), seed=seed)
    return MultiVectorCorpus(base.vectors, base.offsets, ids=page_ids(n_files, pages))


def as_float16(corpus: MultiVectorCorpus) -> MultiVectorCorpus:
    return MultiVectorCorpus(corpus.vectors.astype(np.float16), corpus.offsets, ids=list(corpus.ids))


def question_texts(corpus: MultiVectorCorpus, n: int, *, seed: int, noise: float = 0.0,
                   prefix: str = "question") -> tuple[list[str], dict[str, np.ndarray], dict[str, str]]:
    """``n`` texts, the vectors the fake encoder returns for each, and each one's planted page."""
    q, qrels = make_queries(corpus, n, tokens=(3, 6), noise=noise, seed=seed)
    texts, vectors, answer = [], {}, {}
    for i, qid in enumerate(q.ids):
        text = f"{prefix} {seed} {i}"
        texts.append(text)
        vectors[text] = np.asarray(q.vectors[q.offsets[i]:q.offsets[i + 1]], dtype=np.float32)
        answer[text] = next(iter(qrels[qid]))
    return texts, vectors, answer


# ---------------------------------------------------------------------------- the fakes


class FakeEncoder:
    def __init__(self, corpus16: MultiVectorCorpus, texts: dict[str, np.ndarray], *, perturb: float = 0.0,
                 record: dict[str, Any] | None = None, log: list | None = None) -> None:
        self.corpus16 = corpus16
        self.texts = texts
        self.perturb = perturb
        self.record = dict(record or {})
        self.log = log if log is not None else []
        self.pages_encoded = 0

    def encode_queries(self, texts):
        self.log.append(("encode_queries", len(texts)))
        arrays = []
        for t in texts:
            if t in self.texts:
                arrays.append(self.texts[t])
            else:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.standard_normal((4, self.corpus16.dimension))
                arrays.append((v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32))
        return MultiVectorCorpus.from_arrays(arrays)

    def encode_pages(self, images, ids, progress=None):
        self.log.append(("encode_pages", len(ids)))
        self.pages_encoded += len(ids)
        pos = {pid: i for i, pid in enumerate(self.corpus16.ids)}
        arrays = []
        rng = np.random.default_rng(7)
        off = self.corpus16.offsets
        for pid in ids:
            i = pos[pid]
            v = np.asarray(self.corpus16.vectors[off[i]:off[i + 1]], dtype=np.float32)
            if self.perturb:
                v = v + self.perturb * rng.standard_normal(v.shape)
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
            arrays.append(v.astype(np.float32))
        return MultiVectorCorpus.from_arrays(arrays, ids=list(ids))


class FakeStore:
    def __init__(self, path: Path, corpus16: MultiVectorCorpus, spec: dict, *, keep_float16: bool = True,
                 code_noise: float = 0.0, reference: Any = None, manifest: dict | None = None) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._f16 = corpus16 if keep_float16 else None
        self._all16 = corpus16
        self.reference = reference
        self.manifest = dict(manifest or {})
        self.commits: list[dict] = []
        self.writable = True
        self.settle(Pipeline.from_dict(spec), code_noise=code_noise)

    def settle(self, pipeline: Pipeline, compressed=None, *, code_noise: float = 0.0) -> None:
        source = MultiVectorCorpus(np.asarray(self._all16.vectors, dtype=np.float32), self._all16.offsets,
                                   ids=list(self._all16.ids))
        if code_noise:
            rng = np.random.default_rng(3)
            v = source.vectors + code_noise * rng.standard_normal(source.vectors.shape).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            source = MultiVectorCorpus(v, source.offsets, ids=list(source.ids))
        if compressed is None:
            pipeline.fit(source)
            compressed = pipeline.compress(source, fit=False)
        self._pipeline = pipeline
        self._seg = compressed

    @property
    def ids(self) -> list[str]:
        return list(self._all16.ids)

    @property
    def n_pages(self) -> int:
        return len(self._all16)

    def float16_corpus(self):
        return self._f16

    def segments(self):
        return [self._seg]

    def pipeline(self) -> Pipeline:
        return self._pipeline

    def lock(self):
        return nullcontext()

    def commit(self, **updates) -> None:
        if not self.writable:
            raise PermissionError("read-only store")
        self.manifest.update(updates)
        self.commits.append(dict(updates))


class FakeSearcher:
    def __init__(self, store: FakeStore) -> None:
        self.store = store

    def search(self, queries, k):
        r = CompressedRetriever(self.store.segments()[0], self.store.pipeline()).search(queries, k)
        return [[(pid, float(r.scores[i, j])) for j, pid in enumerate(row)] for i, row in enumerate(r.ids)]

    def rank_all(self, queries, k):
        return [[pid for pid, _ in row] for row in self.search(queries, k)]


class FakeIndex:
    """What ``_check`` / ``_upgrade`` / ``_spotcheck`` read of an index, per the contract."""

    def __init__(self, tmp: Path, *, layout: str = "int8(per_vector)", target: float = 0.97,
                 keep_float16: bool = True, code_noise: float = 0.0, perturb: float = 0.0,
                 texts: dict[str, np.ndarray] | None = None, record: dict | None = None,
                 with_files: bool = True, practice: str | None = "match", n_pages: int | None = None,
                 n_built: int | None = None, checked: bool = False, dim: int = 32) -> None:
        self.path = Path(tmp) / "docs.ovx"
        self.corpus = synthetic(dim=dim)
        self.corpus16 = as_float16(self.corpus)
        self.target = target
        self.selection = fake_select("fake/model", target, n_built or len(self.corpus), checked=checked)
        self.layout = next(r for r in self.selection.rows if r.name == layout)
        self.files: dict[str, tuple[str, int]] = {}
        if with_files:
            folder = Path(tmp) / "docs"
            folder.mkdir(parents=True, exist_ok=True)
            for pid in self.corpus.ids:
                doc, _, page = pid.rpartition("::p")
                f = folder / Path(doc).name
                f.touch()
                self.files[pid] = (str(f), int(page))
        self.record = dict(record if record is not None else
                           {"model": "fake/model", "dtype": "float16", "dpi": 150, "max_side": 1536,
                            "batch_size": 4, "versions": {"numpy": np.__version__}})
        manifest = {"n_pages_at_build": n_built or len(self.corpus)}
        self._store = FakeStore(self.path, self.corpus16, dict(self.layout.spec), keep_float16=keep_float16,
                                code_noise=code_noise, manifest=manifest)
        self._searcher = FakeSearcher(self._store)
        self._n_pages = n_pages
        self.log: list = []
        self.encoder = FakeEncoder(self.corpus16, texts or {}, perturb=perturb, record=self.record, log=self.log)
        self.loads = 0
        self.rebuilds: list[dict] = []
        self.resettles: list[dict] = []
        if practice is not None:
            self._store.reference = self._practice_reference(practice)
        self._store.manifest["drift"] = self._drift_record() if with_files else {
            "ids": ["docs/report00.pdf::p1"], "files": [None], "pages": [None], "scores": [1.0]}

    # contract surface ---------------------------------------------------------
    @property
    def _manifest(self) -> dict:
        return self._store.manifest

    @property
    def n_pages(self) -> int:
        return self._n_pages if self._n_pages is not None else self._store.n_pages

    def _encoder(self):
        self.loads += 1
        self.log.append(("load",))
        return self.encoder

    def search_vectors(self, queries, k=5):
        from optivision.representation import as_corpus

        rows = self._searcher.search(as_corpus(queries), k)
        return [[Hit(pid, *(self.files.get(pid, (None, None))), score) for pid, score in row] for row in rows]

    def _rebuild(self, layout: str, refit: bool = True):
        self.log.append(("rebuild", layout))
        self.rebuilds.append({"layout": layout, "refit": refit})
        return self

    def _resettle(self, pipeline, compressed, *, layout, certificate, updates):
        self.log.append(("resettle", layout.name))
        self.resettles.append({"pipeline": pipeline, "compressed": compressed, "layout": layout,
                               "certificate": certificate, "updates": updates})
        self._store.settle(pipeline, compressed)
        self._store.commit(**updates)
        self.layout = layout
        return self

    # helpers -------------------------------------------------------------------
    def _drift_record(self) -> dict:
        from optivision.easy._check import make_drift_record, pick_drift_pages

        picks = pick_drift_pages(self.corpus16.ids)
        pos = {pid: i for i, pid in enumerate(self.corpus16.ids)}
        vectors = self.corpus16.select([pos[p] for p in picks])
        return make_drift_record(self._store, vectors, self.files)

    def _practice_reference(self, mode: str) -> dict:
        from optivision.core.retrieve.exact import ExactRetriever

        q, qrels = make_queries(self.corpus, 30, tokens=(3, 6), noise=0.2, seed=11)
        probes = MultiVectorCorpus(q.vectors, q.offsets, ids=[f"probe{i:03d}" for i in range(len(q))])
        rel = {f"probe{i:03d}": set(qrels[qid]) for i, qid in enumerate(q.ids)}
        r = ExactRetriever(self.corpus16).search(probes, 10)
        rankings = {pid: list(row) for pid, row in zip(probes.ids, r.ids)}
        return {"queries": probes, "relevant": rel, "clusters": {p: next(iter(s)) for p, s in rel.items()},
                "rankings": rankings}
