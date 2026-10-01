"""Benchmark matrix for the universal layer: datasets x pipelines -> measured rows.

Everything here is exact MaxSim over the whole corpus (see :mod:`optivision.scoring`),
quality is reported as absolute metrics *and* as retention against the float
baseline of the same run with a paired bootstrap interval, and every row
records the pipeline that produced it as JSON so it can be rebuilt.

Datasets come from the encode caches the original ``bench`` writes, or from
the pickle-free ``MultiVectorCorpus`` format for anything else.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .compose import Pipeline
from .evaluation import (
    DEFAULT_METRICS,
    measure,
    per_query_metrics,
    relevant_from_baseline,
    relevant_from_qrels,
    retention,
)
from .representation import MultiVectorCorpus
from .stages import (
    BinaryQuantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    RedundancyPruner,
    SpatialPruner,
)


@dataclass
class Dataset:
    """Documents, queries and (optionally) labels, all as vectors."""

    name: str
    corpus: MultiVectorCorpus
    queries: MultiVectorCorpus
    qrels: dict[str, set[str]] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def relevant(self) -> list[np.ndarray]:
        if self.qrels is None:
            raise ValueError(f"{self.name} has no labels; use the baseline reference")
        return relevant_from_qrels(self.qrels, self.queries.ids, self.corpus.ids)

    def select_queries(self, indices: Sequence[int]) -> Dataset:
        q = self.queries.select(indices)
        qrels = None if self.qrels is None else {i: self.qrels[i] for i in q.ids if i in self.qrels}
        return Dataset(self.name, self.corpus, q, qrels, dict(self.attrs))


def load_vector_dataset(prefix: str | Path, name: str | None = None) -> Dataset:
    """Vector files written by ``scripts/encode_vectors.py``.

    ``prefix`` is the shared path stem: ``<prefix>_docs.npz``, ``<prefix>_queries.npz``
    and ``<prefix>_qrels.json`` (``{qid: [doc ids]}``). All three are pickle-free.
    """
    prefix = Path(prefix)
    corpus = MultiVectorCorpus.load(f"{prefix}_docs.npz")
    queries = MultiVectorCorpus.load(f"{prefix}_queries.npz")
    qrels = {k: set(v) for k, v in json.loads(Path(f"{prefix}_qrels.json").read_text(encoding="utf-8")).items()}
    missing = sorted(q for q in qrels if q not in set(queries.ids))
    if missing:
        raise ValueError(f"{prefix}: {len(missing)} labelled queries have no vectors (e.g. {missing[0]})")
    return Dataset(name or prefix.name, corpus, queries, qrels, attrs=dict(corpus.attrs))


def load_legacy_dataset(cache: str | Path, queries_json: str | Path, name: str | None = None,
                        with_images: bool = True) -> Dataset:
    """An encode cache from ``optivision bench --cache`` plus its ``queries.json``.

    The query cache (``<cache>.queries.npz``) stores vectors in encode order
    keyed by the query text; labels come from ``queries.json`` and are matched by
    that text, so a reordered labels file cannot silently misalign them.
    """
    cache = Path(cache)
    corpus = MultiVectorCorpus.from_legacy_cache(cache, with_images=with_images)
    qcache = cache.with_suffix(".queries.npz")
    zq = np.load(qcache, allow_pickle=True)  # legacy format: texts are a pickled object
    texts = list(json.loads(str(zq["texts"])))
    vectors = [np.asarray(zq[f"q_{i}"], dtype=np.float32) for i in range(len(texts))]
    rows = json.loads(Path(queries_json).read_text(encoding="utf-8"))
    by_text = {r["query"]: r for r in rows}
    missing = [t for t in texts if t not in by_text]
    if missing:
        raise ValueError(f"{len(missing)} cached queries have no label in {queries_json}, e.g. {missing[0]!r}")
    qids = [by_text[t]["qid"] for t in texts]
    indexed = set(corpus.ids)
    qrels = {by_text[t]["qid"]: set(by_text[t]["relevant"]) & indexed for t in texts}
    queries = MultiVectorCorpus.from_arrays(vectors, ids=qids, attrs={"source": str(qcache)})
    return Dataset(name or cache.stem, corpus, queries, qrels, attrs=dict(corpus.attrs))


def legacy_variants() -> dict[str, Pipeline]:
    """The original ``bench.default_variants()`` table, rebuilt from stages."""
    return {
        "baseline-float32": Pipeline(),
        "binary-only": Pipeline([BinaryQuantizer()]),
        "int8-only": Pipeline([Int8Quantizer()]),
        "lloyd2-only": Pipeline([Lloyd2Quantizer(seed=7)]),
        "spatial-only": Pipeline([SpatialPruner()]),
        "spatial+redundancy": Pipeline([SpatialPruner(), RedundancyPruner()]),
        "prune+int8": Pipeline([SpatialPruner(), RedundancyPruner(), Int8Quantizer()]),
        "optivision": Pipeline([SpatialPruner(), RedundancyPruner(), BinaryQuantizer()]),
        "optivision-aggressive": Pipeline(
            [SpatialPruner(keep_ratio=0.25), RedundancyPruner(threshold=0.85), BinaryQuantizer()]
        ),
    }


def run_matrix(
    dataset: Dataset,
    pipelines: dict[str, Pipeline],
    metrics: Sequence[str] = DEFAULT_METRICS,
    reference: str = "labels",
    baseline_depth: int = 1,
    n_boot: int = 1000,
    progress: Any = None,
) -> dict[str, Any]:
    """Measure every pipeline on one dataset.

    ``reference="labels"`` scores against the dataset's qrels; ``"baseline"``
    against the float baseline's own top-``baseline_depth`` pages. Either way
    retention is ``metric(pipeline) / metric(float baseline)`` with a paired
    bootstrap interval over queries, and the float baseline is always row zero.
    """
    t_start = time.perf_counter()
    if reference not in {"labels", "baseline"}:
        raise ValueError("reference must be 'labels' or 'baseline'")
    # The float baseline is scored once: its scores define the label-free
    # reference, and its metrics are row zero and every retention denominator.
    probe = [np.zeros(0, np.int64)] * len(dataset.queries)
    base_m, base_scores = measure(Pipeline(), dataset.corpus, dataset.queries, probe, metrics=metrics,
                                  label="baseline-float32")
    relevant = (dataset.relevant() if reference == "labels"
                else relevant_from_baseline(base_scores, depth=baseline_depth))
    base_m.per_query = per_query_metrics(base_scores, relevant, metrics)
    base_pq = base_m.per_query

    rows: list[dict[str, Any]] = []
    todo = {name: pipe for name, pipe in pipelines.items() if name != "baseline-float32"}
    for name, pipe in {"baseline-float32": None, **todo}.items():
        if pipe is None:
            m = base_m
        else:
            m, _ = measure(pipe, dataset.corpus, dataset.queries, relevant, metrics=metrics, label=name)
        row = m.summary()
        row["pipeline"] = m.pipeline
        for metric in metrics:
            point, lo, hi = retention(m.per_query[metric], base_pq[metric], n_boot=n_boot)
            row[f"retention:{metric}"] = point
            row[f"retention_ci:{metric}"] = [lo, hi]
        rows.append(row)
        if progress is not None:
            progress(name, row)
    return {
        "dataset": dataset.name,
        "attrs": {k: v for k, v in dataset.attrs.items() if isinstance(v, (str, int, float))},
        "n_docs": len(dataset.corpus),
        "n_queries": len(dataset.queries),
        "reference": reference if reference == "labels" else f"baseline@{baseline_depth}",
        "metrics": list(metrics),
        "rows": rows,
        "machine": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__},
        "seconds": time.perf_counter() - t_start,
    }


def to_markdown(result: dict[str, Any], metric: str = "ndcg@5") -> str:
    head = (f"**{result['dataset']}** — {result['n_docs']} docs, {result['n_queries']} queries, "
            f"reference: {result['reference']}\n\n")
    cols = ["pipeline", "vec/doc", "dim", "bits", "KB/doc", "x float32", metric, "retention [95% CI]", "q ms"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in result["rows"]:
        lo, hi = r[f"retention_ci:{metric}"]
        lines.append(
            f"| {r['label']} | {r['vectors_per_doc']:.1f} | {r['dim']} | {r['bits_per_dim']:.3g} | "
            f"{r['bytes_per_doc'] / 1e3:.2f} | {r['compression_vs_float32']:.1f}x | {r[metric]:.4f} | "
            f"{r[f'retention:{metric}']:.1%} [{lo:.1%}, {hi:.1%}] | {r['query_ms']:.1f} |"
        )
    return head + "\n".join(lines) + "\n"


def save_result(result: dict[str, Any], out_dir: str | Path, stem: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or result["dataset"]
    path = out / f"{stem}.json"
    path.write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    (out / f"{stem}.md").write_text(to_markdown(result), encoding="utf-8")
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
