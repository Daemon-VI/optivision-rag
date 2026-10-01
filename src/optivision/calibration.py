"""Quality-aware selection: the smallest configuration that meets a retention target.

    result = calibrate(corpus, queries, qrels, quality_target=0.97, metric="ndcg@5")

1. Queries are split once into a *calibration* part and a *held-out* part.
2. Every candidate pipeline is measured on the calibration queries, as retention
   against the float baseline on the same queries (labels if given, else the
   baseline's own top results).
3. The feasible candidates are those whose calibration retention meets the
   target (``safety="point"``) or whose bootstrap lower bound does
   (``safety="lower_ci"``). The smallest feasible one wins (bytes per document,
   then query time).
4. The winner -- and only the winner -- is reported with its retention on the
   held-out queries, which played no part in choosing it.

Step 4 is the point. A configuration chosen because it scored well on some
queries is optimistically biased on those queries; the held-out figure is the
honest estimate, and it can miss the target. The result says so when it does.

Each candidate is scored once against *all* queries and its score matrix is
kept (queries x documents float32 -- about 1 MB for 500 x 500), so the held-out
number costs nothing extra and :func:`select` can re-run the choice for any
target, metric, reference or split without re-scoring (see
:func:`score_space`, used by the benchmark scripts to repeat splits).

Candidates are grouped into *families* ordered from least to most aggressive
(one token-reduction knob swept at a fixed codec). Within a family the search
stops at the first infeasible step when ``assume_monotone=True``; that is an
assumption about the data, not a guarantee, and it can be switched off.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .compose import Pipeline
from .evaluation import (
    per_query_metrics,
    relevant_from_baseline,
    relevant_from_qrels,
    retention,
    retention_lower_bound,
    split_queries,
)
from .representation import MultiVectorCorpus
from .scoring import maxsim_matrix
from .stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    DimensionProjector,
    Float16Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Quantizer,
)

#: Below this many calibration queries the held-out outcome was unreliable in
#: the calibration study (36-query splits met the target in as few as 67% of
#: splits even with the lower-bound rule).
MIN_RELIABLE_CALIBRATION_QUERIES = 100


# ------------------------------------------------------------ candidates


@dataclass
class ScoredCandidate:
    """One pipeline, compressed once and scored against every query."""

    family: str
    label: str
    pipeline: dict[str, Any]
    report: dict[str, Any]
    scores: np.ndarray  # float32 [n_queries, n_docs]
    compress_seconds: float
    score_seconds: float

    @property
    def bytes_per_doc(self) -> float:
        return float(self.report["bytes_per_doc"])

    @property
    def query_ms(self) -> float:
        return 1000.0 * self.score_seconds / max(1, self.scores.shape[0])


#: Byte budget for the transform cache. Each entry is a whole transformed corpus:
#: negligible at 128-d, but ~10 GB for one 2,560-d split of ViDoRe pages.
TRANSFORM_CACHE_BYTES = int(os.environ.get("OPTIVISION_CACHE_BYTES", str(4 * 1024**3)))


def _cache_put(cache: dict, key: str, value: tuple) -> None:
    """Insert, then evict least-recently used entries beyond the byte budget.

    Every stage is deterministic, so an evicted prefix recomputes to the same output.
    """
    cache[key] = value
    total = sum(v[1].vectors.nbytes for v in cache.values())
    while total > TRANSFORM_CACHE_BYTES and len(cache) > 1:
        oldest = next(iter(cache))
        total -= cache.pop(oldest)[1].vectors.nbytes


def _compress(pipe: Pipeline, corpus: MultiVectorCorpus, cache: dict | None) -> Any:
    """Compress, reusing the token/dimension stages' output across candidates.

    Candidates that differ only in their codec share one merge or projection:
    the first time a stage prefix is seen it is fitted and applied, and later
    candidates get the fitted stages and their output (so their queries are
    projected by the same fitted map).
    """
    if cache is None:
        return pipe.compress(corpus)
    prefix = pipe.vector_stages
    key = json.dumps([s.to_dict() for s in prefix], sort_keys=True, default=str)
    if key in cache:
        cache[key] = cache.pop(key)  # mark as most recently used
    else:
        fitted = Pipeline(prefix).fit(corpus)
        _cache_put(cache, key, (fitted.stages, fitted.transform(corpus)))
    stages, transformed = cache[key]
    pipe.stages = [*stages, *[s for s in pipe.stages if isinstance(s, Quantizer)]]
    return pipe.encode(transformed, original=corpus)


def score_candidate(pipe: Pipeline, corpus: MultiVectorCorpus, queries: MultiVectorCorpus,
                    family: str = "", label: str | None = None, cache: dict | None = None) -> ScoredCandidate:
    t0 = time.perf_counter()
    compressed = _compress(pipe, corpus, cache)
    t1 = time.perf_counter()
    scores = maxsim_matrix(pipe.transform_queries(queries), compressed)
    t2 = time.perf_counter()
    return ScoredCandidate(family, label or pipe.label(), pipe.to_dict(), compressed.report(),
                           scores, t1 - t0, t2 - t1)


def score_space(corpus: MultiVectorCorpus, queries: MultiVectorCorpus,
                space: dict[str, Sequence[Pipeline]], progress: Any = None,
                cache_transforms: bool = True) -> list[ScoredCandidate]:
    """Score every candidate in ``space`` (no early stopping). For experiments."""
    cache: dict | None = {} if cache_transforms else None
    out = []
    for family, steps in space.items():
        for pipe in steps:
            sc = score_candidate(pipe, corpus, queries, family=family, cache=cache)
            out.append(sc)
            if progress is not None:
                progress(sc)
    return out


@dataclass
class Candidate:
    """A scored candidate's standing on the calibration split."""

    family: str
    label: str
    pipeline: dict[str, Any]
    bytes_per_doc: float
    vectors_per_doc: float
    dim: int
    bits_per_dim: float
    compression_vs_float32: float
    retention: float
    retention_lo: float
    retention_hi: float
    query_ms: float
    compress_seconds: float
    feasible: bool = False

    def row(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _judge(sc: ScoredCandidate, relevant: Sequence[np.ndarray], idx: np.ndarray, metric: str,
           base_pq: np.ndarray, target: float, safety: str, max_bytes: float | None,
           n_boot: int, seed: int, confidence: float = 0.975, margin: float = 0.0) -> Candidate:
    """Judge one candidate on the calibration queries.

    ``safety="lower_ci"`` compares the one-sided ``confidence`` lower bound with
    ``target + margin``; ``"point"`` compares the point estimate. The reported
    ``retention_lo/hi`` stay the ordinary 95% interval.
    """
    pq = per_query_metrics(sc.scores[idx], [relevant[i] for i in idx], [metric])[metric]
    point, lo, hi = retention(pq, base_pq, n_boot=n_boot, seed=seed)
    if safety == "point":
        score = point
    else:
        score = retention_lower_bound(pq, base_pq, confidence, n_boot=n_boot, seed=seed)
    target = target + margin
    return Candidate(
        family=sc.family, label=sc.label, pipeline=sc.pipeline,
        bytes_per_doc=sc.bytes_per_doc,
        vectors_per_doc=float(sc.report["vectors_per_doc"]),
        dim=int(sc.report["dim"]),
        bits_per_dim=float(sc.report["bits_per_dim"]),
        compression_vs_float32=float(sc.report["compression_vs_float32"]),
        retention=point, retention_lo=lo, retention_hi=hi,
        query_ms=sc.query_ms, compress_seconds=sc.compress_seconds,
        feasible=bool(score >= target) and (max_bytes is None or sc.bytes_per_doc <= max_bytes),
    )


def _rank_key(c: Candidate, position: int) -> tuple:
    """Smallest stored size wins; exact ties go to fewer vectors, then to the
    earlier (less aggressive) candidate. Measured scan time is not used: it is
    timing noise, and it made the choice between byte-identical candidates (a
    stage that changed nothing) differ from run to run."""
    return (c.bytes_per_doc, c.vectors_per_doc, position)


def _per_candidate_confidence(confidence: float, multiplicity: str, n_candidates: int) -> float:
    if multiplicity == "none":
        return confidence
    if multiplicity == "bonferroni":
        return 1.0 - (1.0 - confidence) / max(1, n_candidates)
    raise ValueError("multiplicity must be 'none' or 'bonferroni'")


def select(scored: Sequence[ScoredCandidate], base_scores: np.ndarray, relevant: Sequence[np.ndarray],
           cal_idx: np.ndarray, hold_idx: np.ndarray, target: float, metric: str = "ndcg@5",
           safety: str = "point", max_bytes_per_doc: float | None = None, n_boot: int = 1000,
           seed: int = 0, confidence: float = 0.975, multiplicity: str = "none",
           margin: float = 0.0) -> tuple[list[Candidate], Candidate | None, dict[str, Any] | None]:
    """Choose among already-scored candidates for one target and split."""
    base_cal = per_query_metrics(base_scores[cal_idx], [relevant[i] for i in cal_idx], [metric])[metric]
    conf = _per_candidate_confidence(confidence, multiplicity, len(scored))
    judged = [_judge(sc, relevant, cal_idx, metric, base_cal, target, safety, max_bytes_per_doc, n_boot, seed,
                     confidence=conf, margin=margin)
              for sc in scored]
    feasible = [i for i, c in enumerate(judged) if c.feasible]
    if not feasible:
        return judged, None, None
    best = min(feasible, key=lambda i: _rank_key(judged[i], i))
    return judged, judged[best], _holdout(scored[best], base_scores, relevant, hold_idx, metric, n_boot, seed)


def _holdout(sc: ScoredCandidate, base_scores: np.ndarray, relevant: Sequence[np.ndarray],
             hold_idx: np.ndarray, metric: str, n_boot: int, seed: int) -> dict[str, Any]:
    rel = [relevant[i] for i in hold_idx]
    base = per_query_metrics(base_scores[hold_idx], rel, [metric])[metric]
    pq = per_query_metrics(sc.scores[hold_idx], rel, [metric])[metric]
    point, lo, hi = retention(pq, base, n_boot=n_boot, seed=seed)
    return {
        "retention": point,
        "retention_ci": [lo, hi],
        metric: float(np.nanmean(pq)),
        "baseline_" + metric: float(np.nanmean(base)),
        "bytes_per_doc": sc.bytes_per_doc,
        "query_ms": sc.query_ms,
    }


# ----------------------------------------------------------- the one call


@dataclass
class CalibrationResult:
    target: float
    metric: str
    reference: str
    safety: str
    n_calibration_queries: int
    n_holdout_queries: int
    selected: Candidate | None
    holdout: dict[str, Any] | None
    candidates: list[Candidate] = field(default_factory=list)
    baseline: dict[str, Any] = field(default_factory=dict)

    @property
    def met_target_on_holdout(self) -> bool | None:
        if self.holdout is None:
            return None
        return bool(self.holdout["retention"] >= self.target)

    @property
    def pipeline(self) -> Pipeline | None:
        return None if self.selected is None else Pipeline.from_dict(self.selected.pipeline)

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "metric": self.metric,
            "reference": self.reference,
            "safety": self.safety,
            "n_calibration_queries": self.n_calibration_queries,
            "n_holdout_queries": self.n_holdout_queries,
            "baseline": self.baseline,
            "selected": None if self.selected is None else self.selected.row(),
            "holdout": self.holdout,
            "met_target_on_holdout": self.met_target_on_holdout,
            "n_candidates_evaluated": len(self.candidates),
        }


def basic_search_space(dim: int) -> dict[str, list[Pipeline]]:
    """The first search space: codecs x (uncentred) adaptive-merge radii.

    Kept because the first calibration study (docs/UNIVERSAL.md, R7) was run
    on it. It lacks the families that won the measured frontier, so it is no
    longer the default.
    """
    radii = [None, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]
    codecs = {"float16": Float16Quantizer, "int8": Int8Quantizer, "binary": BinaryQuantizer}
    families: dict[str, list[Pipeline]] = {}
    for cname, cls in codecs.items():
        steps = []
        for r in radii:
            stages = [] if r is None else [AdaptiveMerge(radius=r)]
            steps.append(Pipeline([*stages, cls()]))
        families[f"adaptive_merge+{cname}"] = steps
    return families


def _scipy_available() -> bool:
    try:
        import scipy.cluster.hierarchy  # noqa: F401
    except ImportError:
        return False
    return True


def recommended_search_space(dim: int, use_scipy: bool | None = None) -> dict[str, list[Pipeline]]:
    """Families built from what won the measured frontier (docs/UNIVERSAL.md, R5).

    Token reduction x codec, every family ordered from least to most
    aggressive:

    * Ward merging to a *fraction* of each document (needs SciPy), and
    * adaptive merging on *centred* vectors (numpy only),

    both relative to the data rather than absolute cosine thresholds, which the
    text ColBERT showed do not transfer between encoders; each with per-vector
    int8, centred int4 and binary codes, plus float16 alone as the near-lossless
    floor.
    """
    use_scipy = _scipy_available() if use_scipy is None else use_scipy
    # int4 is centred: measured never worse on ColPali beyond noise and far better
    # on the anisotropic text model (SciFact 89.7% -> 98.6%). Binary is *not*
    # centred: that collapsed the same text model (94.1% -> 1.1%).
    codecs = {"int8": lambda: Int8Quantizer("per_vector"), "int4": lambda: Int4Quantizer(center="mean"),
              "binary": BinaryQuantizer}
    families: dict[str, list[Pipeline]] = {"float16": [Pipeline([Float16Quantizer()])]}
    for cname, make in codecs.items():
        if use_scipy:
            families[f"ward+{cname}"] = [Pipeline(([HierarchicalMerge(ratio=f)] if f else []) + [make()])
                                         for f in (None, 0.5, 0.33, 0.25, 0.15, 0.1)]
        families[f"adaptive_centred+{cname}"] = [
            Pipeline(([AdaptiveMerge(radius=r, center="mean")] if r else []) + [make()])
            for r in (None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
        ]
    return families


def projection_targets(dim: int) -> list[int]:
    """Widths to project to, widest first: d/2, d/4, d/8, d/16 and 128 (each kept if < d)."""
    targets = {dim // 2, dim // 4, dim // 8, dim // 16, 128}
    return sorted((t for t in targets if 64 <= t < dim), reverse=True)


def projection_search_space(dim: int, use_scipy: bool | None = None) -> dict[str, list[Pipeline]]:
    """EXPERIMENTAL, opt-in: families that also reduce the vector width (PCA fitted on
    the documents; queries are projected with the same map). Not part of the default:
    it was added for the 2,560-4,096-d study and has no measured history yet.

    Each family varies the width from none to the narrowest target at a fixed codec
    and token stage, least aggressive first, like the default families.
    """
    use_scipy = _scipy_available() if use_scipy is None else use_scipy
    widths = [None, *projection_targets(dim)]
    codecs = {"int8": lambda: Int8Quantizer("per_vector"), "int4": lambda: Int4Quantizer(center="mean"),
              "binary": BinaryQuantizer}
    families: dict[str, list[Pipeline]] = {}
    for cname, make in codecs.items():
        families[f"pca+{cname}"] = [Pipeline(([DimensionProjector(w)] if w else []) + [make()]) for w in widths]
    if use_scipy:
        for ratio in (0.33, 0.25):
            for cname in ("int8", "int4"):
                families[f"ward{ratio}+pca+{cname}"] = [
                    Pipeline([HierarchicalMerge(ratio=ratio), *([DimensionProjector(w)] if w else []), codecs[cname]()])
                    for w in widths]
    return families


def wide_search_space(dim: int, use_scipy: bool | None = None) -> dict[str, list[Pipeline]]:
    """EXPERIMENTAL: the default space plus :func:`projection_search_space`."""
    return {**recommended_search_space(dim, use_scipy), **projection_search_space(dim, use_scipy)}


def default_search_space(dim: int) -> dict[str, list[Pipeline]]:
    """The search space ``calibrate()`` and ``optimize()`` use unless told otherwise."""
    return recommended_search_space(dim)


def reference_relevance(corpus: MultiVectorCorpus, queries: MultiVectorCorpus, qrels: dict[str, Any] | None,
                        reference: str, base_scores: np.ndarray, baseline_depth: int = 1) -> list[np.ndarray]:
    if reference == "labels":
        if qrels is None:
            raise ValueError("reference='labels' needs qrels")
        return relevant_from_qrels(qrels, queries.ids, corpus.ids)
    return relevant_from_baseline(base_scores, depth=baseline_depth)


def calibrate(
    corpus: MultiVectorCorpus,
    queries: MultiVectorCorpus,
    qrels: dict[str, Any] | None = None,
    quality_target: float = 0.97,
    metric: str = "ndcg@5",
    search_space: dict[str, Sequence[Pipeline]] | None = None,
    calibration_fraction: float = 0.5,
    seed: int = 0,
    reference: str = "auto",
    baseline_depth: int = 1,
    safety: str = "lower_ci",
    max_bytes_per_doc: float | None = None,
    assume_monotone: bool = True,
    n_boot: int = 1000,
    progress: Any = None,
    cache_transforms: bool = True,
    confidence: float = 0.999,
    multiplicity: str = "none",
    margin: float = 0.0,
) -> CalibrationResult:
    """Pick the smallest pipeline meeting ``quality_target`` on calibration queries,
    then report it on held-out queries. See the module docstring.

    ``safety="lower_ci"`` (the default) requires the one-sided ``confidence``
    lower bound of the calibration retention to meet the target. Choosing by
    the point estimate (``"point"``) is a winner's-curse selection: with the
    default search space it met a 0.97 target on held-out ColPali DocVQA queries
    in 20% of 20 random splits, against 19 of 20 for the defaults
    (docs/UNIVERSAL.md, R7b). Stricter is safer and chooses less compression.

    The bound is approximate (bootstrap standard error) and holds per
    configuration, not for the one chosen among many; it assumes future queries
    and the corpus look like the ones given here. Nothing guarantees the target
    (docs/UNIVERSAL.md, R10): the held-out figure in the result is the check.
    """
    if not 0.0 < quality_target <= 1.0:
        raise ValueError("quality_target is a retention fraction in (0, 1]")
    if safety not in {"point", "lower_ci"}:
        raise ValueError("safety must be 'point' or 'lower_ci'")
    if reference == "auto":
        reference = "labels" if qrels is not None else "baseline"
    if reference not in {"labels", "baseline"}:
        raise ValueError("reference must be 'auto', 'labels' or 'baseline'")
    if len(queries) < 4:
        raise ValueError("need at least 4 queries to split into calibration and held-out halves")
    cal_idx, hold_idx = split_queries(len(queries), calibration_fraction, seed)
    t_base = time.perf_counter()
    base_scores = maxsim_matrix(queries, corpus)
    base_query_ms = 1000.0 * (time.perf_counter() - t_base) / max(1, len(queries))
    relevant = reference_relevance(corpus, queries, qrels, reference, base_scores, baseline_depth)
    base_cal = per_query_metrics(base_scores[cal_idx], [relevant[i] for i in cal_idx], [metric])[metric]
    base_hold = per_query_metrics(base_scores[hold_idx], [relevant[i] for i in hold_idx], [metric])[metric]
    # Only queries the float index scores above zero say anything about retention (0 / 0 drops out of the
    # ratio), and the bootstrap cannot see a failure no calibration query happened to show.
    n_informative = int(np.sum(np.nan_to_num(base_cal) > 0))
    if n_informative < MIN_RELIABLE_CALIBRATION_QUERIES:
        warnings.warn(
            f"calibrating on {len(cal_idx)} queries ({n_informative} with a nonzero baseline {metric}, the only "
            f"ones that carry information): with fewer than {MIN_RELIABLE_CALIBRATION_QUERIES} the chosen "
            "configuration missed its target on held-out queries in up to a third of splits even with "
            "safety='lower_ci' (36-query splits, docs/UNIVERSAL.md); use more queries or a lower target",
            stacklevel=2,
        )

    if search_space is None and not _scipy_available():
        warnings.warn(
            "SciPy is not installed, so the default search space has no Ward merging families. The measured "
            "behaviour of the default (docs/UNIVERSAL.md, R7b) is for the full space: "
            "pip install 'optivision-rag[merge]'",
            stacklevel=2,
        )
    space = search_space if search_space is not None else default_search_space(corpus.dimension)
    conf = _per_candidate_confidence(confidence, multiplicity, sum(len(v) for v in space.values()))
    cache: dict | None = {} if cache_transforms else None
    scored: list[ScoredCandidate] = []
    judged: list[Candidate] = []
    for family, steps in space.items():
        for pipe in steps:
            sc = score_candidate(pipe, corpus, queries, family=family, cache=cache)
            cand = _judge(sc, relevant, cal_idx, metric, base_cal, quality_target, safety,
                          max_bytes_per_doc, n_boot, seed, confidence=conf, margin=margin)
            scored.append(sc)
            judged.append(cand)
            if progress is not None:
                progress(cand)
            if assume_monotone and not cand.feasible and (
                max_bytes_per_doc is None or cand.bytes_per_doc <= max_bytes_per_doc
            ):
                break  # quality failed; later steps in this family are more aggressive

    feasible = [i for i, c in enumerate(judged) if c.feasible]
    selected = holdout = None
    if feasible:
        best = min(feasible, key=lambda i: _rank_key(judged[i], i))
        selected = judged[best]
        holdout = _holdout(scored[best], base_scores, relevant, hold_idx, metric, n_boot, seed)

    return CalibrationResult(
        target=quality_target,
        metric=metric,
        reference=reference if reference == "labels" else f"baseline@{baseline_depth}",
        safety=safety,
        n_calibration_queries=len(cal_idx),
        n_holdout_queries=len(hold_idx),
        selected=selected,
        holdout=holdout,
        candidates=judged,
        baseline={
            "bytes_per_doc": corpus.num_vectors * corpus.dimension * 4 / max(1, len(corpus)),
            "vectors_per_doc": corpus.num_vectors / max(1, len(corpus)),
            "dim": corpus.dimension,
            "query_ms": base_query_ms,
            "calibration_" + metric: float(np.nanmean(base_cal)),
            "informative_calibration_queries": n_informative,
            "per_candidate_confidence": conf if safety == "lower_ci" else None,
            "holdout_" + metric: float(np.nanmean(base_hold)),
        },
    )


# ----------------------------------------------------- proxy queries (experimental)


def pseudo_queries(corpus: MultiVectorCorpus, n_queries: int = 200, tokens: int = 20, noise: float = 0.0,
                   seed: int = 0) -> tuple[MultiVectorCorpus, dict[str, set[str]]]:
    """EXPERIMENTAL: fragments of documents used as stand-in queries.

    Each pseudo-query is ``tokens`` random non-protected vectors of one random
    document (plus optional Gaussian noise), labelled relevant to that document.
    Whether calibrating on these predicts quality on real queries is an open
    empirical question -- document vectors are not query vectors -- and it has
    to be measured before it is trusted (see docs/UNIVERSAL.md).
    """
    rng = np.random.default_rng(seed)
    eligible = [i for i in range(len(corpus)) if corpus.counts[i] > 0]
    picks = rng.choice(eligible, size=min(n_queries, len(eligible)), replace=len(eligible) < n_queries)
    arrays, ids, qrels = [], [], {}
    for j, i in enumerate(picks):
        doc = corpus[int(i)]
        rows = np.arange(doc.num_vectors)
        if doc.protected is not None and (~doc.protected).any():
            rows = rows[~doc.protected]
        take = rng.choice(rows, size=min(tokens, rows.size), replace=False)
        v = np.asarray(doc.vectors[take], dtype=np.float32)
        if noise > 0:
            v = v + noise * rng.standard_normal(v.shape).astype(np.float32) / np.sqrt(v.shape[1])
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        qid = f"pseudo{j:05d}"
        arrays.append(v)
        ids.append(qid)
        qrels[qid] = {corpus.ids[int(i)]}
    return MultiVectorCorpus.from_arrays(arrays, ids=ids, attrs={"pseudo_queries": True}), qrels
