"""Score any layout on a saved unit, on a CPU, against the float16 vectors the notebook kept.

Every arm of the validation is a layout (a catalogue ``pipeline`` dict) fitted and compressed
here with the package's own ``Pipeline`` and searched with ``CompressedRetriever``: exact
MaxSim over the decoded codes, ties broken by page order. Retention is the ratio of mean
nDCG (nDCG@5 for pages, nDCG@10 for text; graded labels count as relevant, as in the
catalogue) against the same model's float16 index, with the catalogue's paired bootstrap
interval over queries (``optivision.evaluation.retention``, 1,000 resamples, seed 0).

Fits follow a build (FINAL_DESIGN 5.7): ``fit="sample"`` fits on a seeded page sample capped at
200,000 vectors and compresses every page with ``fit=False``; ``fit="full"`` fits on all pages
(the catalogue's way, V3 compares the two); ``fit="growth"`` fits on a seeded half and reuses
that state on all pages (V3's growth arm). Results are cached per unit, so the many arms that
share a layout pay for it once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from optivision.compose import Pipeline
from optivision.core.identity import canonical_json
from optivision.core.retrieve.exact import CompressedRetriever
from optivision.evaluation import per_query_metrics, relevant_from_qrels, retention
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, rank

from easy_validation import units
from easy_validation.data import Unit

TOP = 50  # depth of the stored rankings (hard-set hardness needs scores past the top 10)


def stateful(spec: dict) -> bool:
    """Whether a layout fits state on the corpus (a centre, a PCA basis): only then do fit variants differ."""
    return not Pipeline.from_dict(spec).fitted


def fit_sample(corpus: MultiVectorCorpus, rows: int = units.FIT_SAMPLE_ROWS, seed: int = units.FIT_SEED) -> np.ndarray:
    """Seeded page sample whose vectors total at most ``rows`` (all pages when they fit)."""
    counts = np.diff(corpus.offsets)
    if counts.sum() <= rows:
        return np.arange(len(corpus))
    order = np.random.default_rng(seed).permutation(len(corpus))
    keep = order[np.cumsum(counts[order]) <= rows]
    return np.sort(keep if keep.size else order[:1])


def growth_sample(n: int, fraction: float = units.GROWTH_FRACTION, seed: int = units.GROWTH_SEED) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[: max(1, int(round(n * fraction)))])


class UnitScorer:
    """Score matrices, rankings and retention of layouts on one unit."""

    def __init__(self, unit: Unit) -> None:
        self.unit = unit
        self.metric = units.PAGE_METRIC if unit.modality == "page" else units.TEXT_METRIC
        self._scores: dict[tuple, np.ndarray] = {}
        self._compressed: dict[tuple, tuple[Any, Pipeline, float]] = {}
        self._transformed: dict[tuple, MultiVectorCorpus] = {}
        self._rel: dict[tuple, list] = {}
        self._pq: dict[tuple, tuple] = {}
        self._self_match = unit.corpus in units.SELF_MATCH_EXCLUDED

    # ------------------------------------------------------------ query sets

    def query_set(self, which: str, ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        qs = self.unit.queries if which == "queries" else self.unit.probe_queries[which]
        if ids is None:
            return qs
        pos = {q: i for i, q in enumerate(qs.ids)}
        return qs.select([pos[i] for i in ids])

    def relevant(self, which: str, qids: Sequence[str], doc_ids: Sequence[str]) -> list[np.ndarray]:
        if which == "queries":
            labels = {q: list(self.unit.qrels.get(q, {})) for q in qids}
        else:
            by_id = {p["id"]: p for p in self.unit.probes[which]}
            labels = {q: list(by_id[q]["relevant"]) for q in qids}
        return relevant_from_qrels(labels, list(qids), list(doc_ids))

    # ------------------------------------------------------------ corpora

    def docs(self, variant: str = "f16", subset: tuple[int, ...] | None = None) -> MultiVectorCorpus:
        base = self.unit.docs if variant == "f16" else self.unit.docs32
        if base is None:
            raise LookupError(f"{self.unit.tag} has no {variant} vectors")
        return base if subset is None else base.select(list(subset))

    def compressed(self, spec: dict, *, fit: str = "sample", variant: str = "f16",
                   subset: tuple[int, ...] | None = None) -> tuple[Any, Pipeline, float]:
        """(store, fitted pipeline, float16 bytes / stored bytes) of a layout; cached."""
        fit = fit if stateful(spec) else "full"
        key = (canonical_json(spec), fit, variant, subset)
        if key not in self._compressed:
            docs = self.docs(variant, subset)
            p = Pipeline.from_dict(spec)
            sample = (fit_sample(docs) if fit == "sample" else growth_sample(len(docs)) if fit == "growth"
                      else np.arange(len(docs)))
            vector_stages = Pipeline(p.vector_stages)
            if vector_stages.fitted:  # merges carry no state: transform every page once, share it
                full_key = (canonical_json(vector_stages.to_dict()), variant, None)
                if full_key not in self._transformed:
                    self._transformed[full_key] = vector_stages.transform(self.docs(variant))
                transformed = self._transformed[full_key]
                if subset is not None:  # per-page stages: a subset's transform is the subset of the transform
                    transformed = transformed.select(list(subset))
                if not p.quantizer.fitted:
                    p.quantizer.fit(transformed.select(list(sample)) if len(sample) < len(docs) else transformed)
                store = p.encode(transformed, original=docs)
            else:
                p.fit(docs.select(list(sample)) if len(sample) < len(docs) else docs)
                store = p.compress(docs, fit=False)
            f16 = docs.num_vectors * docs.dimension * 2
            ratio = f16 / max(1, int(store.nbytes))
            self._compressed[key] = (store, p, ratio)
        return self._compressed[key]

    # ------------------------------------------------------------ scores

    def scores(self, spec: dict | None, which: str = "queries", *, qids: Sequence[str] | None = None,
               fit: str = "sample", variant: str = "f16", subset: tuple[int, ...] | None = None) -> np.ndarray:
        """``[n_queries, n_pages]`` MaxSim scores; ``spec=None`` is the float16 index itself."""
        if spec is not None and not stateful(spec):
            fit = "full"
        key = (None if spec is None else canonical_json(spec), which, None if qids is None else tuple(qids), fit,
               variant, subset)
        if key in self._scores:
            return self._scores[key]
        qs = self.query_set(which, qids)
        if spec is None or spec.get("stages") == [] or (spec.get("stages") == [{"stage": "float16"}] and variant == "f16"):
            s = maxsim_matrix(qs, self.docs(variant, subset))
        elif subset is not None and not stateful(spec):
            full = self.scores(spec, which, qids=qids, fit=fit, variant=variant)
            s = full[:, list(subset)]
        else:
            store, p, _ = self.compressed(spec, fit=fit, variant=variant, subset=subset)
            s = CompressedRetriever(store, p).score(qs)
        s = np.asarray(s, dtype=np.float32)
        if self._self_match and which == "queries":
            s = s.copy()
            doc_ids = self.docs(variant, subset).ids
            pos = {d: i for i, d in enumerate(doc_ids)}
            for qi, q in enumerate(qs.ids):
                if q in pos:
                    s[qi, pos[q]] = -np.inf
        self._scores[key] = s
        return s

    def random_scores(self, which: str, qids: Sequence[str], seed: int = units.FAULT_SEED) -> np.ndarray:
        """The "random scores" fault: an independent uniform score per (query, page)."""
        rng = np.random.default_rng(seed)
        return rng.random((len(qids), self.unit.n_pages)).astype(np.float32)

    def per_query(self, s: np.ndarray, which: str = "queries", qids: Sequence[str] | None = None,
                  subset: tuple[int, ...] | None = None, metric: str | None = None) -> np.ndarray:
        m = metric or self.metric
        pkey = (id(s), which, None if qids is None else tuple(qids), subset, m)
        hit = self._pq.get(pkey)
        if hit is not None and hit[0] is s:
            return hit[1]
        rkey = (which, None if qids is None else tuple(qids), subset)
        if rkey not in self._rel:
            ids = list(self.query_set(which, qids).ids)
            self._rel[rkey] = self.relevant(which, ids, self.docs("f16", subset).ids)
        out = per_query_metrics(s, self._rel[rkey], [m])[m]
        self._pq[pkey] = (s, out)
        return out

    def retention(self, spec: dict | None, *, fit: str = "sample", variant: str = "f16",
                  subset: tuple[int, ...] | None = None, qids: Sequence[str] | None = None,
                  base_variant: str | None = None) -> dict:
        """Point, interval low and high of a layout's retention on the real queries."""
        base = self.per_query(self.scores(None, qids=qids, variant=base_variant or variant, subset=subset),
                              qids=qids, subset=subset)
        if spec is None:
            c = base
        else:
            c = self.per_query(self.scores(spec, qids=qids, fit=fit, variant=variant, subset=subset),
                               qids=qids, subset=subset)
        point, lo, hi = retention(c, base, n_boot=units.N_BOOT, seed=units.BOOT_SEED)
        out = {"point": point, "low": lo, "high": hi, "n_queries": int(np.isfinite(base).sum()),
               "baseline": float(np.nanmean(base))}
        if spec is not None:
            out["ratio_f16"] = round(self.compressed(spec, fit=fit, variant=variant, subset=subset)[2], 3)
        return out

    # ------------------------------------------------------------ rankings

    def ranking(self, s: np.ndarray, ids: Sequence[str], k: int = 10, subset: tuple[int, ...] | None = None,
                with_scores: bool = False) -> dict[str, list]:
        doc_ids = self.docs("f16", subset).ids
        top = rank(s, min(k, s.shape[1]))
        out = {}
        for qi, q in enumerate(ids):
            if with_scores:
                out[q] = [(doc_ids[j], float(s[qi, j])) for j in top[qi]]
            else:
                out[q] = [doc_ids[j] for j in top[qi]]
        return out
