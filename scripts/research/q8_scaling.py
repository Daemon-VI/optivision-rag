"""Q8 validation timings (reports/research/Q8/PLAN.md section 5). TIMING ONLY.

The real ColQwen2 DocVQA corpus (500 pages) is tiled m in {1, 2, 4, 8} times; every copy
is an identical duplicate, so these synthetic corpora are used for wall-time scaling only
and no retrieval metric is computed on them.

    V1  ExactIndex.score time vs corpus size, six configurations (codes compressed once, tiled)
    V2  old per-query vs batched ExactIndex.rescore with 50 random candidates per query
        drawn from the m-tiled corpus (overlap between queries falls as m grows)

Protocol: one untimed warm-up run per (configuration, m), then 5 timed rounds round-robin.
Writes reports/research/Q8/validation.json.

    OPTIVISION_DATA=../optivision-rag-v2/data PYTHONPATH="src;scripts/research" python scripts/research/q8_scaling.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

import e1_common as E
import numpy as np

from optivision.compose import CompressedCorpus, Pipeline
from optivision.representation import MultiVectorCorpus
from optivision.scoring import segment_reduce
from optivision.stages import BinaryQuantizer, DimensionProjector, HierarchicalMerge, Int8Quantizer
from optivision.storage import ExactIndex

OUT = E.ROOT / "reports" / "research" / "Q8"
TILES = (1, 2, 4, 8)
REPEATS = 5
CANDIDATES = 50


def tile(store, m):
    off = np.asarray(store.offsets, dtype=np.int64)
    n_vec = int(off[-1])
    offs = np.concatenate([[0], np.concatenate([off[1:] + j * n_vec for j in range(m)])])
    if isinstance(store, CompressedCorpus):
        return CompressedCorpus(np.tile(store.codes, (m, 1)), offs, store.dim, store.quantizer,
                                [f"{i}" for i in range(len(offs) - 1)], dict(store.stats), store.pipeline)
    return MultiVectorCorpus(np.tile(np.asarray(store.vectors), (m, 1)), offs)


class OldExactIndex(ExactIndex):
    """The previous per-query rescore body (verbatim copy, as in scripts/engineering/batched_rescore.py)."""

    def rescore(self, queries, candidates):
        q = self._q(queries)
        cand = np.asarray(candidates, dtype=np.int64)
        out = np.full(cand.shape, -np.inf, dtype=np.float32)
        offsets = self.store.offsets
        for qi in range(cand.shape[0]):
            lo_q, hi_q = int(q.offsets[qi]), int(q.offsets[qi + 1])
            if hi_q == lo_q:
                out[qi] = 0.0
                continue
            qv = q.decode_rows(lo_q, hi_q)
            spans = [(int(offsets[d]), int(offsets[d + 1])) for d in cand[qi]]
            blocks = [self.store.decode_rows(lo, hi) for lo, hi in spans]
            counts = np.array([b.shape[0] for b in blocks], dtype=np.int64)
            if counts.sum() == 0:
                continue
            local = np.concatenate([[0], np.cumsum(counts)])
            per_token = segment_reduce(qv @ np.concatenate(blocks, axis=0).T, local, np.maximum, axis=1, fill=-np.inf)
            out[qi] = per_token.sum(axis=0)
        return out


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while timing (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); no-op elsewhere.

    Added after the first run was invalidated: the laptop entered standby overnight mid-run.
    """
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def main():
    keep_awake(True)
    t_start = time.time()
    ds, _ = E.load_s7b().dataset(E.dataset_arg("colqwen2_docvqa"))
    corpus, queries = ds.corpus, ds.queries
    nq = len(queries)
    configs = {
        "float32": None,
        "int8(per_vector)": Pipeline([Int8Quantizer("per_vector")]),
        "binary": Pipeline([BinaryQuantizer()]),
        "hierarchical_merge(0.25)": Pipeline([HierarchicalMerge(ratio=0.25)]),
        "hierarchical_merge(0.25) > binary": Pipeline([HierarchicalMerge(ratio=0.25), BinaryQuantizer()]),
        "project(64)": Pipeline([DimensionProjector(64)]),
    }
    v1 = {}
    for name, pipe in configs.items():
        base = corpus if pipe is None else pipe.compress(corpus)
        stores = {m: tile(base, m) for m in TILES}
        idx = {m: ExactIndex(stores[m], pipe if pipe is not None and pipe.vector_stages else None) for m in TILES}
        warm = {m: timed(lambda m=m, ix=idx: ix[m].score(queries)) for m in TILES}
        times = {m: [] for m in TILES}
        for r in range(REPEATS):
            for m in TILES[r % len(TILES):] + TILES[:r % len(TILES)]:
                times[m].append(timed(lambda m=m, ix=idx: ix[m].score(queries)))
        vecs = {m: int(stores[m].offsets[-1]) for m in TILES}
        med = {m: 1000 * float(np.median(times[m])) / nq for m in TILES}
        x = np.array([vecs[m] for m in TILES], float)
        y = np.array([med[m] for m in TILES])
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (slope * x + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ratio8 = med[8] / (8 * med[1])
        v1[name] = {"pages": {m: 500 * m for m in TILES}, "vectors": vecs,
                    "ms_per_query_median": med,
                    "ms_per_query_all": {m: [1000 * t / nq for t in times[m]] for m in TILES},
                    "warmup_ms_per_query": {m: 1000 * warm[m] / nq for m in TILES},
                    "linear_fit_ms_per_query": {"slope_per_vector": float(slope), "intercept": float(intercept),
                                                "r2": float(r2)},
                    "ratio_m8_over_8x_m1": float(ratio8),
                    "linear_by_plan_criterion": bool(abs(ratio8 - 1) <= 0.15 and r2 >= 0.98)}
        print("V1", name, {m: round(v, 2) for m, v in med.items()}, "r2", round(float(r2), 4),
              "ratio8", round(float(ratio8), 3), f"{time.time() - t_start:.0f}s", flush=True)
        del stores, idx, base
    cold = Pipeline([Int8Quantizer("per_vector")]).compress(corpus)
    v2 = {}
    for m in TILES:
        store = tile(cold, m)
        n_pages = 500 * m
        rng = np.random.default_rng(0)
        cand = np.stack([rng.choice(n_pages, CANDIDATES, replace=False) for _ in range(nq)])
        new_i, old_i = ExactIndex(store), OldExactIndex(store)
        same = bool(np.array_equal(new_i.rescore(queries, cand), old_i.rescore(queries, cand)))
        fns = {"old": lambda o=old_i, c=cand: o.rescore(queries, c), "batched": lambda n=new_i, c=cand: n.rescore(queries, c)}
        warm = {k: timed(f) for k, f in fns.items()}
        times = {k: [] for k in fns}
        for r in range(REPEATS):
            for k in (("old", "batched") if r % 2 == 0 else ("batched", "old")):
                times[k].append(timed(fns[k]))
        med = {k: 1000 * float(np.median(v)) / nq for k, v in times.items()}
        v2[m] = {"pages": n_pages, "distinct_candidate_pages_per_batch": int(np.unique(cand).size),
                 "candidate_slots": int(cand.size), "outputs_bitwise_equal": same,
                 "ms_per_query_median": med, "ms_per_query_all": {k: [1000 * t / nq for t in v] for k, v in times.items()},
                 "warmup_ms_per_query": {k: 1000 * w / nq for k, w in warm.items()},
                 "old_over_batched": med["old"] / med["batched"]}
        print("V2", m, v2[m]["distinct_candidate_pages_per_batch"], {k: round(v, 2) for k, v in med.items()},
              round(v2[m]["old_over_batched"], 2), same, f"{time.time() - t_start:.0f}s", flush=True)
        del store
    out = {"label": "MEASURED wall time on synthetic tiled corpora (identical duplicate pages); timing only, "
                    "no retrieval metric", "dataset": "colqwen2_docvqa tiled", "n_queries": nq, "repeats": REPEATS,
           "V1_scan": v1, "V2_rescore": v2,
           "environment": E.environment() | {"processor": platform.processor(), "cpus": os.cpu_count()},
           "wall_clock": {"start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
                          "end": time.strftime("%Y-%m-%d %H:%M:%S")},
           "seconds": time.time() - t_start}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("done", f"{time.time() - t_start:.0f}s", flush=True)
    keep_awake(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
