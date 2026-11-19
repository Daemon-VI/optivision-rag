"""Read the units a validation notebook saved (``encode_unit.py``), on a CPU.

``data/easy/<batch>/`` holds what ``ingest.py`` unpacked from ``results_<batch>.zip``. A unit
(model x corpus) can be spread over batches (its pages from B1, its Tier B probes from the
VLM batch), so :func:`load_unit` searches every batch folder given and merges the files.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from optivision.representation import MultiVectorCorpus

from easy_validation import units


def read_arrays(path: Path) -> MultiVectorCorpus:
    with np.load(path, allow_pickle=False) as z:
        return MultiVectorCorpus(z["vectors"], z["offsets"], ids=[str(i) for i in z["ids"]])


def read_shards(folder: Path) -> MultiVectorCorpus | None:
    shards = sorted(folder.glob("shard_*.npz"))
    if not shards:
        return None
    parts = [read_arrays(p) for p in shards]
    vectors = np.concatenate([p.vectors for p in parts])
    counts = np.concatenate([np.diff(p.offsets) for p in parts])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    ids = [i for p in parts for i in p.ids]
    return MultiVectorCorpus(vectors, offsets, ids=ids)


@dataclass
class Unit:
    """One model on one corpus: float16 pages, queries with labels, and the practice questions."""

    model: str
    corpus: str
    docs: MultiVectorCorpus
    queries: MultiVectorCorpus
    qrels: dict[str, dict[str, int]]
    pages: list[dict] = field(default_factory=list)
    probes: dict[str, list[dict]] = field(default_factory=dict)
    probe_queries: dict[str, MultiVectorCorpus] = field(default_factory=dict)
    profiles: dict[str, dict] = field(default_factory=dict)
    docs32: MultiVectorCorpus | None = None
    run: dict = field(default_factory=dict)
    modality: str = "page"
    heldout: bool = False

    @property
    def n_pages(self) -> int:
        return len(self.docs)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def scored(self) -> bool:
        return self.n_queries >= units.MIN_QUERIES

    @property
    def vectors_per_page(self) -> float:
        counts = np.diff(self.docs.offsets)
        return float(np.median(counts)) if counts.size else 0.0

    @property
    def tag(self) -> str:
        return f"{self.model.split('/')[-1]}__{self.corpus}"

    def text_checkable(self, source: str) -> bool:
        """The profile's checkability (text layer on at least half the non-blank pages, 60 text pages)."""
        prof = self.profiles.get(source) or {}
        if "checkable" in prof:
            return bool(prof["checkable"])
        return bool(self.probes.get(source))


def unit_dirs(roots: list[Path], model: str, corpus: str) -> list[Path]:
    name = f"{model.split('/')[-1]}__{corpus}"
    return [r / name for r in roots if (r / name).is_dir()]


def corpus_dirs(roots: list[Path], corpus: str) -> list[Path]:
    return [r / "_corpus" / corpus for r in roots if (r / "_corpus" / corpus).is_dir()]


def _json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def load_unit(roots: list[Path], model: str, corpus: str, *, heldout: bool = False) -> Unit | None:
    """The unit's files merged over ``roots``; None when no batch encoded all its pages."""
    docs = docs32 = queries = None
    probe_q: dict[str, MultiVectorCorpus] = {}
    run: dict = {}
    for d in unit_dirs(roots, model, corpus):
        if (d / "run.json").exists():
            r = _json(d / "run.json")
            if docs is None and r.get("outcome") == "complete":
                run = r
                docs = read_shards(d / "docs_f16")
                if (d / "docs_f32").is_dir():
                    docs32 = read_shards(d / "docs_f32")
                if (d / "queries_f16.npz").exists():
                    queries = read_arrays(d / "queries_f16.npz")
        for source in units.PROBE_SOURCES:
            p = d / f"probes_{source}_f16.npz"
            if p.exists() and source not in probe_q:
                probe_q[source] = read_arrays(p)
    if docs is None or queries is None:
        return None
    qrels: dict = {}
    pages: list = []
    probes: dict[str, list[dict]] = {}
    profiles: dict[str, dict] = {}
    for c in corpus_dirs(roots, corpus):
        if not qrels and (c / "qrels.json").exists():
            qrels = _json(c / "qrels.json")
        if not pages and (c / "pages.json").exists():
            pages = _json(c / "pages.json")
        for source in units.PROBE_SOURCES:
            if source not in probes and (c / f"probes_{source}.json").exists():
                probes[source] = _json(c / f"probes_{source}.json")
            if source not in profiles and (c / f"profile_{source}.json").exists():
                profiles[source] = _json(c / f"profile_{source}.json")
    modality = units.corpus(corpus).modality
    probes = {s: p for s, p in probes.items() if s in probe_q}
    return Unit(model, corpus, docs, queries, qrels, pages, probes, probe_q, profiles, docs32, run, modality, heldout)


def batch_roots(data: Path) -> list[Path]:
    """Every ingested batch folder under ``data`` (``data/easy/<batch>/``)."""
    return sorted(p for p in data.iterdir() if p.is_dir() and not p.name.startswith("_")) if data.exists() else []


def heldout_models(roots: list[Path]) -> list[str]:
    for r in roots:
        f = r / "heldout_selected.json"
        if f.exists():
            return [s["model"] for s in _json(f).get("selected", [])]
    return []
