"""V9 on CPU: the small-memory layout (FINAL_DESIGN 5.7 step 6 and section 8).

Prediction V9: identical rankings; resident memory <= hot copy + shortlist decode + 128 MB at 20,000 pages.

What this script measures, on a synthetic corpus (random unit vectors from
``optivision.testing.make_corpus``; the numbers say nothing about any real encoder):

1. **identity**: the memory-mapped two-tier search (``optivision.easy._tiered.TieredSearcher``:
   ``TwoTierRetriever`` over a ``SegmentChain`` of memory-mapped int8 segments) against the
   campaign's in-RAM path ``TieredIndex(ExactIndex(hot), ExactIndex(int8 per vector), C)``
   (``scripts/universal_study/s13_wide_study.py``), top-10 ids and scores of every query;
2. **agreement with the plain 8-bit search**: exact ``CompressedRetriever`` over the whole int8 copy,
   top-1 / top-5 / top-10 (list equality and set overlap);
3. **memory**: a fresh process loads the hot copy, maps the 8-bit copy and answers the queries; its peak
   private bytes (commit charge: memory that must live in RAM or the page file) and peak working set
   (which also counts the 8-bit file's pages the OS keeps cached after reading them) are compared with
   the bound ``hot bytes + C x mean vectors per page x dim x 4 + 128 MB``.

Usage (writes ``reports/easy/v9/v9_<hot>_<pages>.json``)::

    python scripts/easy_validation/v9_small_memory.py --pages 20000 --work D:/tmp/v9
    python scripts/easy_validation/v9_small_memory.py --pages 2000 --hot binary --keep

Phases run in separate processes (``generate``, then one ``measure`` per hot copy) so the memory figures
of ``measure`` do not include the generator's float arrays. Results at fewer than 20,000 pages are
extrapolated to 20,000 only as an estimate, and labelled so.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
#: pages per generated segment (small, so the float64 draws of one segment stay near 75 MB)
SEGMENT_PAGES = 100
HOT_NAMES = {"binary": "binary", "hm25": "hm(0.25) > binary"}


def _src_on_path() -> None:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


# ---------------------------------------------------------------------------- generate


def _segment(args: tuple) -> dict:
    """Generate segment ``s`` (float vectors in this process only) and write its tiers."""
    _src_on_path()
    from optivision.easy import _tiered as T
    from optivision.representation import MultiVectorCorpus
    from optivision.testing.synthetic import make_corpus

    s, n_pages, start, dim, lo, hi, seed, work, hots = args
    raw = make_corpus(n_pages, dim, tokens=(lo, hi), seed=seed * 1_000_003 + s)
    ids = [f"p{start + i:06d}" for i in range(n_pages)]
    corpus = MultiVectorCorpus(raw.vectors, raw.offsets, ids=ids)
    out = {"segment": s, "pages": n_pages, "vectors": corpus.num_vectors}
    work = Path(work)
    if not (work / "cold" / f"{s:04d}" / "tier.json").exists():
        T.write_segment(work / "cold" / f"{s:04d}", T.cold_pipeline().compress(corpus))
    for key in hots:
        folder = work / f"hot_{key}" / f"{s:04d}"
        if not (folder / "tier.json").exists():
            T.write_segment(folder, T.hot_pipeline(HOT_NAMES[key]).compress(corpus))
    return out


def _queries(work: Path, n_pages: int, n_queries: int, dim: int, lo: int, hi: int, seed: int,
             q_tokens: tuple[int, int], noise: float, seg_pages: int) -> None:
    """Queries cut from known pages, as ``make_queries`` does (noisy copies of a page's vectors)."""
    _src_on_path()
    from optivision.representation import MultiVectorCorpus
    from optivision.testing.synthetic import make_corpus

    rng = np.random.default_rng(seed + 17)
    targets = np.sort(rng.choice(n_pages, size=min(n_queries, n_pages), replace=False))
    vecs, counts, target_ids = [], [], []
    cached: tuple[int, MultiVectorCorpus] | None = None
    for t in targets:  # sorted, so each segment is regenerated once
        s, i = divmod(int(t), seg_pages)
        if cached is None or cached[0] != s:
            n_seg = min(seg_pages, n_pages - s * seg_pages)
            cached = (s, make_corpus(n_seg, dim, tokens=(lo, hi), seed=seed * 1_000_003 + s))
        page = cached[1]
        a, b = page.span(i)
        doc = np.asarray(page.vectors[a:b], dtype=np.float64)
        m = int(min(rng.integers(q_tokens[0], q_tokens[1] + 1), doc.shape[0]))
        rows = doc[rng.choice(doc.shape[0], size=m, replace=False)]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        noisy = rows + rng.standard_normal(rows.shape) * noise * norms / math.sqrt(dim)
        noisy *= norms / np.maximum(np.linalg.norm(noisy, axis=1, keepdims=True), 1e-300)
        vecs.append(noisy.astype(np.float32))
        counts.append(m)
        target_ids.append(f"p{int(t):06d}")
    q = MultiVectorCorpus(np.concatenate(vecs), np.concatenate([[0], np.cumsum(counts)]),
                          ids=[f"q{i:05d}" for i in range(len(counts))])
    q.save(work / "queries.npz")
    (work / "query_targets.json").write_text(json.dumps(target_ids), encoding="utf-8")


def generate(a: argparse.Namespace) -> None:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    sp = a.segment_pages
    n_seg = math.ceil(a.pages / sp)
    hots = [h for h in a.hot.split(",") if h]
    jobs = [(s, min(sp, a.pages - s * sp), s * sp, a.dim, a.vectors[0],
             a.vectors[1], a.seed, str(work), hots) for s in range(n_seg)]
    t0 = time.perf_counter()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            done = list(pool.map(_segment, jobs))
    else:
        done = [_segment(j) for j in jobs]
    _queries(work, a.pages, a.queries, a.dim, a.vectors[0], a.vectors[1], a.seed, tuple(a.query_tokens), a.noise, sp)
    meta = {"pages": a.pages, "segments": n_seg, "segment_pages": sp, "dim": a.dim,
            "vectors_per_page_range": list(a.vectors), "vectors": int(sum(d["vectors"] for d in done)),
            "queries": a.queries, "query_tokens": list(a.query_tokens), "noise": a.noise, "seed": a.seed,
            "hots": hots, "generate_seconds": time.perf_counter() - t0}
    (work / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)


# ---------------------------------------------------------------------------- measure (fresh process)


def _mem() -> dict:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is in the dev environment
        return {}
    p = psutil.Process()
    info = p.memory_info()._asdict()
    out = {k: int(v) for k, v in info.items()}
    try:
        out["uss"] = int(p.memory_full_info().uss)
    except Exception as exc:  # noqa: BLE001 - uss needs extra rights on some systems
        out["uss_error"] = type(exc).__name__
    return out


class _Sampler:
    """Polls the private working set (USS) and the working set (RSS) on a thread.

    USS counts the pages only this process holds: the hot copy, Python, NumPy and scan blocks. Pages of
    the memory-mapped 8-bit file are file-backed and shareable, so they are not in USS; the OS may drop
    them at any time and read them again. RSS (the working set) includes them. The commit charge is not
    used for the bound because OpenBLAS commits per-thread buffers it never touches.
    """

    def __init__(self, interval: float = 0.01) -> None:
        import threading

        import psutil

        self._p = psutil.Process()
        self.interval = interval
        self.peak_uss = 0
        self.peak_rss = 0
        self.n = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def sample(self) -> None:
        m = self._p.memory_full_info()
        self.peak_uss = max(self.peak_uss, int(m.uss))
        self.peak_rss = max(self.peak_rss, int(m.rss))
        self.n += 1

    def start(self) -> None:
        self.sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.sample()


def measure(a: argparse.Namespace) -> None:
    mem0 = _mem()
    _src_on_path()
    from optivision.easy import _tiered as T
    from optivision.representation import MultiVectorCorpus

    work = Path(a.work)
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    hot_key = a.measure_hot
    queries = MultiVectorCorpus.load(work / "queries.npz")
    mem_imports = _mem()

    k = a.k
    tiered_ids, tiered_scores, samples = [], [], []
    sampler = _Sampler(a.sample_interval)
    sampler.start()
    try:
        t0 = time.perf_counter()
        cold_dirs = sorted((work / "cold").iterdir())
        hot_dirs = sorted((work / f"hot_{hot_key}").iterdir())
        hot = [T.open_segment(d, mmap=False) for d in hot_dirs]          # the copy that stays in RAM
        cold = [T.open_segment(d, mmap=True) for d in cold_dirs]         # read for the shortlist only
        searcher = T.TieredSearcher(hot, cold)
        open_s = time.perf_counter() - t0
        mem_open = _mem()
        t1 = time.perf_counter()
        for b0 in range(0, len(queries), a.batch):
            part = queries.select(range(b0, min(b0 + a.batch, len(queries))))
            r = searcher.ranking(part, k)
            tiered_ids += [list(row) for row in r.ids]
            tiered_scores.append(np.asarray(r.scores))
            samples.append(_mem())
        search_s = time.perf_counter() - t1
    finally:
        sampler.stop()
    mem_search = _mem()                      # OS peaks so far cover open + two-tier search only
    bound = searcher.memory_bound()

    # ---- identity against the campaign's in-RAM path (TieredIndex over the same codes)
    from optivision.scoring import rank  # noqa: F401  (TieredIndex ranks with it)
    from optivision.storage import ExactIndex, TieredIndex

    t2 = time.perf_counter()
    ref = TieredIndex(ExactIndex(searcher.hot_store, None, T.MAX_BLOCK_BYTES),
                      ExactIndex(searcher.cold_store, None, T.MAX_BLOCK_BYTES), searcher.candidates)
    ref_ids, ref_scores = [], []
    names = np.asarray(searcher.cold_store.ids, dtype=object)
    for b0 in range(0, len(queries), a.batch):
        part = queries.select(range(b0, min(b0 + a.batch, len(queries))))
        top, sc = ref.search(part, k)
        ref_ids += [names[row].tolist() for row in top]
        ref_scores.append(sc)
    ref_s = time.perf_counter() - t2
    tiered_scores_a, ref_scores_a = np.concatenate(tiered_scores), np.concatenate(ref_scores)
    identical = [a_ == b_ for a_, b_ in zip(tiered_ids, ref_ids, strict=True)]

    # ---- the plain 8-bit search (exact over every int8 vector, memory-mapped)
    from optivision.core.retrieve.exact import CompressedRetriever

    t3 = time.perf_counter()
    plain = CompressedRetriever(searcher.cold_store, None, T.MAX_BLOCK_BYTES)
    pr = plain.search(queries, k)
    plain_ids = [list(row) for row in pr.ids]
    plain_s = time.perf_counter() - t3
    targets = json.loads((work / "query_targets.json").read_text(encoding="utf-8"))

    def agree(depth: int) -> dict:
        lists = [t[:depth] == p[:depth] for t, p in zip(tiered_ids, plain_ids, strict=True)]
        sets = [len(set(t[:depth]) & set(p[:depth])) / depth for t, p in zip(tiered_ids, plain_ids, strict=True)]
        return {"queries_identical": int(sum(lists)), "share_identical": float(np.mean(lists)),
                "mean_set_overlap": float(np.mean(sets))}

    in_shortlist = []
    sl = searcher.hot_retriever.search(queries, min(searcher.candidates, searcher.n_pages))
    for row_sl, row_plain in zip(sl.ids, plain_ids, strict=True):
        s_ = set(row_sl)
        in_shortlist.append(sum(d in s_ for d in row_plain) / len(row_plain))

    peak_private = int(sampler.peak_uss)
    out = {
        "test": "V9 (small-memory layout), CPU, synthetic corpus",
        "hot": HOT_NAMES[hot_key],
        "cold": "int8(per_vector), memory-mapped",
        "pages": searcher.n_pages,
        "candidates": searcher.candidates,
        "k": k,
        "queries": len(queries),
        "batch": a.batch,
        "corpus": meta,
        "bytes": {"hot": searcher.hot_bytes, "cold": searcher.cold_bytes,
                  "hot_per_page": searcher.hot_bytes / searcher.n_pages,
                  "cold_per_page": searcher.cold_bytes / searcher.n_pages},
        "bound": {"hot_bytes": bound.hot_bytes, "shortlist_decode_bytes": bound.shortlist_decode_bytes,
                  "allowance_bytes": bound.allowance_bytes, "total": bound.total},
        "memory": {"process_start": mem0, "after_imports": mem_imports, "after_open": mem_open,
                   "after_search": mem_search, "per_batch": samples,
                   "peak_private_bytes": peak_private,
                   "peak_private_samples": sampler.n,
                   "sample_interval_s": a.sample_interval,
                   "peak_working_set_bytes": mem_search.get("peak_wset"),
                   "peak_commit_bytes": mem_search.get("peak_pagefile"),
                   "metric_note": "peak_private_bytes is the largest sampled private working set (USS) from "
                                  "opening the copies to the end of the two-tier searches: pages only this "
                                  "process holds (hot copy, Python, NumPy, scan blocks). Pages read through the "
                                  "memory map of the 8-bit file are file-backed and not in it. "
                                  "peak_working_set_bytes (the OS peak) also counts those file pages, which the "
                                  "OS keeps after a read and may drop at any time. peak_commit_bytes includes "
                                  "buffers OpenBLAS commits and never touches, so it is not a resident figure."},
        "identity_vs_tieredindex": {
            "queries_identical_top_k": int(sum(identical)), "share": float(np.mean(identical)),
            "max_abs_score_difference": float(np.max(np.abs(tiered_scores_a - ref_scores_a))),
            "scores_bitwise_equal": bool(np.array_equal(tiered_scores_a, ref_scores_a)),
        },
        "agreement_vs_plain_int8": {"top1": agree(1), "top5": agree(5), "top10": agree(10),
                                    "plain_top10_in_shortlist_mean": float(np.mean(in_shortlist)),
                                    "plain_top1_is_target": float(np.mean([p[0] == t for p, t in zip(plain_ids, targets,
                                                                                                     strict=True)])),
                                    "tiered_top1_is_target": float(np.mean([p[0] == t for p, t in zip(tiered_ids, targets,
                                                                                                      strict=True)]))},
        "seconds": {"open": open_s, "two_tier_search": search_s, "two_tier_per_query": search_s / len(queries),
                    "tieredindex_reference": ref_s, "plain_int8_search": plain_s,
                    "note": "wall time on a laptop CPU shared with other jobs; not a speed claim"},
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "numpy": np.__version__, "cpus": os.cpu_count()},
    }
    out["pass"] = {
        "identical_rankings_vs_tieredindex": bool(all(identical)),
        "identical_rankings_vs_plain_int8_top10": bool(out["agreement_vs_plain_int8"]["top10"]["share_identical"] == 1.0),
        "resident_within_bound_private": bool(peak_private <= bound.total),
        "resident_within_bound_working_set": (None if mem_search.get("peak_wset") is None
                                               else bool(mem_search["peak_wset"] <= bound.total)),
    }
    print(json.dumps(out), flush=True)


# ---------------------------------------------------------------------------- run


def _estimate_20k(res: dict) -> dict | None:
    """Linear extrapolation to 20,000 pages (an estimate, never a measurement)."""
    n = res["pages"]
    if n >= 20000 or res["memory"]["peak_private_bytes"] is None:
        return None
    hot_pp = res["bytes"]["hot_per_page"]
    vpp = res["corpus"]["vectors"] / n
    overhead = res["memory"]["peak_private_bytes"] - res["bytes"]["hot"]
    c = max(50, math.ceil(0.02 * 20000))
    bound = hot_pp * 20000 + c * vpp * res["corpus"]["dim"] * 4 + 128e6
    est = hot_pp * 20000 + overhead
    return {"kind": "estimate", "basis": f"hot bytes per page x 20,000 + the overhead measured at {n} pages",
            "peak_private_bytes": est, "bound": bound, "within": bool(est <= bound)}


def run(a: argparse.Namespace) -> None:
    work = Path(a.work)
    meta_file = work / "meta.json"
    if meta_file.exists() and json.loads(meta_file.read_text())["pages"] != a.pages:
        shutil.rmtree(work)  # another size: start again
    if not meta_file.exists():  # a partial work folder resumes: finished segments are skipped
        cmd = [sys.executable, __file__, "generate", "--work", str(work), "--pages", str(a.pages), "--dim", str(a.dim),
               "--vectors", *map(str, a.vectors), "--queries", str(a.queries), "--query-tokens",
               *map(str, a.query_tokens), "--noise", str(a.noise), "--seed", str(a.seed), "--hot", a.hot,
               "--workers", str(a.workers), "--segment-pages", str(a.segment_pages)]
        subprocess.run(cmd, check=True)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in [h for h in a.hot.split(",") if h]:
        cmd = [sys.executable, __file__, "measure", "--work", str(work), "--measure-hot", key, "--k", str(a.k),
               "--batch", str(a.batch), "--sample-interval", str(a.sample_interval)]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        res = json.loads(proc.stdout.strip().splitlines()[-1])
        res["estimate_at_20000_pages"] = _estimate_20k(res)
        path = out_dir / f"v9_{key}_{a.pages}.json"
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"{key}: identical vs TieredIndex {res['pass']['identical_rankings_vs_tieredindex']}; "
              f"top-10 identical vs plain int8 {res['agreement_vs_plain_int8']['top10']['share_identical']:.3f}; "
              f"peak private {res['memory']['peak_private_bytes']} vs bound {res['bound']['total']} -> {path}",
              flush=True)
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("phase", nargs="?", default="run", choices=["run", "generate", "measure"])
    p.add_argument("--pages", type=int, default=20000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--vectors", type=int, nargs=2, default=[700, 800], help="vectors per page, inclusive range")
    p.add_argument("--queries", type=int, default=64)
    p.add_argument("--query-tokens", type=int, nargs=2, default=[12, 24])
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hot", default="binary,hm25", help="comma list of binary, hm25")
    p.add_argument("--measure-hot", default="binary")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--segment-pages", type=int, default=SEGMENT_PAGES)
    p.add_argument("--sample-interval", type=float, default=0.01, help="seconds between memory samples")
    p.add_argument("--work", default=str(ROOT / ".v9_work"))
    p.add_argument("--out", default=str(ROOT / "reports" / "easy" / "v9"))
    p.add_argument("--keep", action="store_true", help="keep the generated tiers")
    a = p.parse_args(argv)
    {"run": run, "generate": generate, "measure": measure}[a.phase](a)


if __name__ == "__main__":
    main()
