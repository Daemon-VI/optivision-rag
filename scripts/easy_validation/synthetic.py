"""A tiny synthetic stand-in for the ingested Kaggle results, in the layout ``encode_unit.py`` writes.

``write_tree(root)`` writes, for every unit the analyses look for (design models on the V1,
TechSlides and industrial corpora, two held-out encoders, the text models), float16 page
vectors with topic structure, queries that are noisy copies of their relevant pages, practice
questions with relevant sets, float32 copies for the hr corpus, ``heldout_selected.json`` and
``wiring.json``. Sizes are a few dozen pages, so the whole programme runs in seconds; the
numbers it produces mean nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from easy_validation import units

HELDOUT = ("synthetic/heldout-a", "synthetic/heldout-b")


def _save(path: Path, arrays: list[np.ndarray], ids: list[str], dtype: str = "float16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = [len(a) for a in arrays]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.savez(path, vectors=np.concatenate(arrays).astype(dtype), offsets=offsets, ids=np.array(ids, dtype=str))


def _unit_vectors(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def corpus_spec(key: str, n_pages: int, n_queries: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pages = [f"doc{i // 10:02d}::p{i % 10 + 1}" for i in range(n_pages)]
    qrels = {}
    for q in range(n_queries):
        rel = rng.choice(n_pages, size=int(rng.integers(1, 3)), replace=False)
        qrels[f"{q}"] = {pages[int(j)]: int(rng.integers(1, 3)) for j in rel}
    probes = []
    for i, p in enumerate(pages):
        for k in range(2):
            probes.append({"id": f"{key}-p{i}-{k}", "text": f"probe {i} {k}", "family": ["span", "keywords"][k],
                           "page_id": p, "relevant": [p]})
    return {"pages": pages, "qrels": qrels, "probes": probes}


def write_unit(root: Path, model: str, key: str, spec: dict, seed: int, dim: int = 16, vecs: int = 6,
               keep32: bool = False, modality: str = "page") -> None:
    rng = np.random.default_rng(seed)
    pages = spec["pages"]
    topics = _unit_vectors(rng.normal(size=(len(pages), dim)))
    docs = [_unit_vectors(topics[i] + 0.35 * rng.normal(size=(vecs + i % 3, dim))) for i in range(len(pages))]
    pos = {p: i for i, p in enumerate(pages)}

    def query_like(relevant: list[str], noise: float) -> np.ndarray:
        centre = topics[[pos[p] for p in relevant]].mean(axis=0)
        return _unit_vectors(centre + noise * rng.normal(size=(4, dim)))

    folder = root / f"{model.split('/')[-1]}__{key}"
    shard = 8
    for lo in range(0, len(pages), shard):
        hi = min(lo + shard, len(pages))
        _save(folder / "docs_f16" / f"shard_{lo:06d}.npz", docs[lo:hi], pages[lo:hi])
        if keep32:
            _save(folder / "docs_f32" / f"shard_{lo:06d}.npz", docs[lo:hi], pages[lo:hi], "float32")
    qids = list(spec["qrels"])
    _save(folder / "queries_f16.npz", [query_like(list(spec["qrels"][q]), 0.9) for q in qids], qids)
    if modality == "page" and units.corpus(key).text is not None:
        probes = spec["probes"]
        _save(folder / "probes_text_layer_f16.npz", [query_like(p["relevant"], 0.6) for p in probes],
              [p["id"] for p in probes])
    run = {"model": model, "corpus": key, "outcome": "complete", "check_retriever": {"ok": True}, "synthetic": True}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")


def write_corpus(root: Path, key: str, spec: dict) -> None:
    c = root / "_corpus" / key
    c.mkdir(parents=True, exist_ok=True)
    (c / "qrels.json").write_text(json.dumps(spec["qrels"]), encoding="utf-8")
    (c / "pages.json").write_text(json.dumps([{"id": p, "doc_id": p.split("::")[0]} for p in spec["pages"]]),
                                  encoding="utf-8")
    if units.corpus(key).text is not None and units.corpus(key).modality == "page":
        (c / "probes_text_layer.json").write_text(json.dumps(spec["probes"]), encoding="utf-8")
        (c / "profile_text_layer.json").write_text(json.dumps({"checkable": True}), encoding="utf-8")


def write_tree(root: Path, n_pages: int = 64, n_queries: int = 160) -> Path:
    """The synthetic results of every validation batch, as one batch folder under ``root``."""
    batch = root / "synthetic_all"
    page_corpora = units.V1_CORPORA + units.UNCHECKED_CORPORA + units.V2_CORPORA
    specs = {k: corpus_spec(k, n_pages + (16 if k in units.V2_CORPORA else 0), n_queries, i)
             for i, k in enumerate(page_corpora + units.TEXT_CORPORA)}
    for k, spec in specs.items():
        write_corpus(batch, k, spec)
    seed = 100
    for m in units.DESIGN:
        for k in units.V1_CORPORA + units.UNCHECKED_CORPORA:
            seed += 1
            write_unit(batch, m, k, specs[k], seed, keep32=k == "v3_hr")
    for m in units.V2_MODELS:
        seed += 1
        write_unit(batch, m, "v3_industrial", specs["v3_industrial"], seed)
    for m in HELDOUT:
        for k in units.V1U_CORPORA:
            seed += 1
            write_unit(batch, m, k, specs[k], seed)
    for m in units.TEXT_MODELS:
        for k in units.TEXT_CORPORA:
            seed += 1
            write_unit(batch, m, k, specs[k], seed, modality="text")
    (batch / "heldout_selected.json").write_text(json.dumps({"selected": [{"model": m} for m in HELDOUT]}),
                                                 encoding="utf-8")
    loaders = {name: {"model": s["model"], "loaded": True, "wiring": {"ok": True}} for name, s in units.V6_LOADERS.items()}
    faults = {f"{m}: {f}": {"ok": False} for m in units.V6_FAULT_MODELS
              for f in ("re-initialised norms", "query/document dtype mismatch")}
    (batch / "wiring.json").write_text(json.dumps({"loaders": loaders, "faults": faults}), encoding="utf-8")
    return root
