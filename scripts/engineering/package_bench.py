"""Engineering benchmarks of the package layer (writes docs/package/ENGINEERING_BENCHMARKS.md).

NEW ENGINEERING MEASUREMENTS of code overhead on one machine, on SYNTHETIC data only: not
research results, nothing about retrieval quality or any encoder, not to be compared with the
Q4/Q5/Q6/Q8 measurements or anything under reports/research/ (other data, other purpose). No
model output and no research store is read; nothing needs a network.

    PYTHONPATH=src python scripts/engineering/package_bench.py [--load-note TEXT]
    PYTHONPATH=src python scripts/engineering/package_bench.py --quick --out SCRATCH_DIR
    PYTHONPATH=src python scripts/engineering/package_bench.py --no-run \
        --write-doc docs/package/ENGINEERING_BENCHMARKS.md

Writes bench.json (a flat list of measurement records, sizes, repeats) and environment.json to
--out; the document is generated from them, so a re-run refreshes every figure. A section whose
module cannot be imported is recorded as "not available at measurement time" and is picked up by
a later re-run. File round trips use a temporary directory that is removed. Exit status 1 if a
section failed.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import functools
import gc
import hashlib
import importlib
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

from optivision.backends import NumpyBackend, NumpyIndex
from optivision.compose import Pipeline
from optivision.core import retrieve as rt
from optivision.core import selection as sel
from optivision.core import stages as st
from optivision.core.evaluate import Evidence, LabelReference
from optivision.core.identity import canonical_json, digest
from optivision.core.vectors import as_host_array, corpus_identity, validate_corpus
from optivision.scoring import maxsim_matrix, rank
from optivision.storage import ExactIndex, TieredIndex, to_legacy_pages
from optivision.testing import make_corpus, make_queries

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "engineering" / "package_benchmarks"
BASE = "91564fd"
FROZEN = [f"src/optivision/{p}" for p in (
    "calibration.py", "compose.py", "scoring.py", "evaluation.py", "representation.py", "stages",
    "compression", "storage.py", "optimize.py")]
#: What a difference from the base commit in one of the FROZEN paths is, where it is known and
#: confined to imports and docstrings (``regression_check`` verifies that part on every run).
KNOWN_DIFFERENCES = {
    "src/optivision/storage.py": (
        "the hardening pass restored the v0.3.0 import of `segment_reduce` and added a sentence "
        "to the docstring of `ExactIndex.rescore`"),
}
SEED, NOISE, TARGET, ERROR, CREATED = 20261006, 2.5, 0.97, 0.05, "2026-10-06T00:00:00Z"
TOKENS, QUERY_TOKENS = {"full": (30, 60), "quick": (6, 12)}, (4, 8)  # vectors per document / query
SIZES = {"full": {"narrow": (128, 500, 450), "wide": (1024, 200, 200), "e2e": (128, 200, 200)},
         "quick": {"narrow": (32, 40, 30), "wide": (64, 30, 20), "e2e": (32, 40, 30)}}
# (dim, documents, queries) per configuration; repeats for cheap calls, retrieval and pipelines,
# and heavy calls; the numbers of queries for the selection arrays
REPEATS = {"full": (7, 5, 3), "quick": (2, 2, 1)}
SELECTION_N = {"full": (50, 225, 450, 3600), "quick": (20, 50)}
PIPELINES = {
    "binary": lambda dim: [st.BinaryQuantizer()],
    "int4, centred": lambda dim: [st.Int4Quantizer(center="mean")],
    "ward 1/4 + int8": lambda dim: [st.HierarchicalMerge(ratio=0.25),
                                    st.Int8Quantizer("per_vector")],
    "pca (d/2) + int8": lambda dim: [st.PCAProjector(dim=dim // 2), st.Int8Quantizer("per_vector")],
    "rotation + binary": lambda dim: [st.RotationProjector(seed=0), st.BinaryQuantizer()]}
BACKEND_CASE = "numpy backend (up to 100 queries)"
LAYER = ["optivision.core.stages", "optivision.core.retrieve", "optivision.core.selection",
         "optivision.core.evaluate", "optivision.core.artifact", "optivision.backends",
         "optivision.api"]
IMPORT_CODE = """
import json, sys, time
def third(mods):
    tops = {m.split(".")[0] for m in mods}
    tops -= set(sys.stdlib_module_names) | {"optivision", "cython_runtime"}
    return sorted(m for m in tops if not m.startswith("_"))
before = set(sys.modules)
t0 = time.perf_counter()
import optivision
t1 = time.perf_counter()
middle, missing = set(sys.modules), []
for name in %s:
    try:
        __import__(name)
    except ImportError:
        missing.append(name)
t2 = time.perf_counter()
print(json.dumps({"import": t1 - t0, "layer": t2 - t1, "third_party": third(middle - before),
                  "layer_third_party": third(set(sys.modules) - middle), "missing": missing}))
"""


class Unavailable(Exception):
    """A module or name that a section needs cannot be imported (yet)."""


def need(module, *names):
    """The named attributes of ``module``; ``Unavailable`` if it or one of them is missing."""
    try:
        mod = importlib.import_module(module)
        return tuple(getattr(mod, n) for n in names)
    except (ImportError, AttributeError) as exc:
        raise Unavailable(f"{type(exc).__name__}: {exc}") from None


# ------------------------------------------------------------------ measuring


def peak_of(fn):
    """Peak bytes that tracemalloc sees (Python and NumPy allocations) during one call."""
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def measure(fn, repeats, setup=None, memory=False):
    """Seconds per call of ``fn(*setup())``: one untimed warm-up call (traced for peak memory when
    ``memory``), then ``repeats`` timed repeats. A call under ~20 ms is looped inside a repeat;
    with ``setup`` every call gets fresh arguments and is never looped."""

    def once(number, traced=False):
        args = setup() if setup else ()
        gc.collect()
        if traced:
            tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(number):
            fn(*args)
        seconds = (time.perf_counter() - t0) / number
        peak = None
        if traced:
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        return seconds, peak

    warm, peak = once(1, memory)
    number = 1 if setup else max(1, min(5000, int(0.02 / max(warm, 1e-9))))
    times = [once(number)[0] for _ in range(repeats)]
    out = {"best": min(times), "median": statistics.median(times), "repeats": repeats,
           "number": number, "all": times}
    return {**out, "peak_bytes": peak} if memory else out


def compare(fns, repeats):
    """Round-robin timing, rotating the start, of callables (the leading one is the reference),
    after a traced warm-up of each. ``ratio`` is the median over rounds of the within-round time
    ratio to the reference, so drift between rounds cancels, unlike in two separate medians."""
    names = list(fns)
    peaks = {n: peak_of(fns[n]) for n in names}
    times = {n: [] for n in names}
    for r in range(repeats):
        for n in names[r % len(names):] + names[:r % len(names)]:
            gc.collect()
            t0 = time.perf_counter()
            fns[n]()
            times[n].append(time.perf_counter() - t0)
    ref = times[names[0]]
    return {n: {"best": min(t), "median": statistics.median(t), "repeats": repeats, "all": t,
                "peak_bytes": peaks[n],
                "ratio": statistics.median(a / b for a, b in zip(t, ref, strict=True))}
            for n, t in times.items()}


def probe():
    """Best of 5 of one 768x768 float32 matrix product, in ms: a CPU speed indicator."""
    a = np.random.default_rng(0).standard_normal((768, 768), dtype=np.float32)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        np.dot(a, a)
        times.append(time.perf_counter() - t0)
    return 1e3 * min(times)


def positions(result):
    """Top-k store positions from any of the result types compared here."""
    return result[0] if isinstance(result, tuple) else getattr(result, "indices", result)


def agreement(a, b):
    """Fraction of queries whose top-k positions are identical in two [n_q, k] arrays."""
    return float(np.mean((a == b).all(axis=1))) if a.shape == b.shape else 0.0


def add_record(records, section, group, name, t, **info):
    """One record: a timing (a ``measure`` or ``compare`` row) plus numbers that describe it."""
    records.append({"section": section, "group": group, "name": name, **t, **info})


def build_data(mode):
    data = {}
    for name, (dim, n_docs, n_queries) in SIZES[mode].items():
        corpus = make_corpus(n_docs, dim, tokens=TOKENS[mode], seed=SEED)
        queries, qrels = make_queries(corpus, n_queries, tokens=QUERY_TOKENS, noise=NOISE,
                                      seed=SEED + 1)
        data[name] = {"dim": dim, "corpus": corpus, "queries": queries, "qrels": qrels}
    return data


# ------------------------------------------------------------------ sections


def s_interface(ctx, add):
    fast = ctx["fast"]
    cand = [spec.to_dict() for spec in sel.CandidateSpace.default(ctx["data"]["narrow"]["dim"])]
    add("candidate list", "canonical_json", measure(lambda: canonical_json(cand), fast),
        nbytes=len(canonical_json(cand)))
    add("candidate list", "json.dumps(sort_keys, compact)", measure(
        lambda: json.dumps(cand, sort_keys=True, separators=(",", ":")), fast))
    add("candidate list", "digest = SHA-256 of canonical_json", measure(lambda: digest(cand), fast))
    for name in ("narrow", "wide"):
        interface_costs(add, ctx["data"][name], fast)


def interface_costs(add, d, fast):
    corpus, queries, v, g = d["corpus"], d["queries"], d["corpus"].vectors, f"d={d['dim']}"
    mb, qmb = v.nbytes / 1e6, queries.vectors.nbytes / 1e6
    add(g, "validate_corpus (documents)", measure(lambda: validate_corpus(corpus), fast), mb=mb)
    add(g, "validate_corpus (queries)", measure(lambda: validate_corpus(queries), fast), mb=qmb)
    add(g, "corpus_identity (documents)", measure(lambda: corpus_identity(corpus), fast), mb=mb)
    add(g, "hashlib.sha256 over the buffer", measure(lambda: hashlib.sha256(v).hexdigest(), fast),
        mb=mb)
    add(g, "as_host_array(ndarray)", measure(lambda: as_host_array(v), fast),
        same_object=as_host_array(v) is v)


def s_facade(ctx, add):
    for name in ("narrow", "wide"):
        facade_costs(add, ctx["data"][name], ctx["fast"])


def facade_costs(add, d, reps):
    corpus, queries, dim, k, g = d["corpus"], d["queries"], d["dim"], 10, f"d={d['dim']} "
    # a two-tier ranking has no order beyond its shortlist, so the shortlist must reach k
    cands = max(k, min(50, len(corpus) // 4))
    bin_p, hot_p, cold_p = (Pipeline([st.BinaryQuantizer()]), Pipeline([
        st.HierarchicalMerge(ratio=0.25), st.BinaryQuantizer()]),
        Pipeline([st.Int8Quantizer("per_vector")]))
    binary, hot, cold = (p.compress(corpus) for p in (bin_p, hot_p, cold_p))
    direct, exact = ExactIndex(corpus), rt.ExactRetriever(corpus)
    over_codes, codes = ExactIndex(binary, bin_p), rt.CompressedRetriever(binary, bin_p)
    tiered = TieredIndex(ExactIndex(hot, hot_p), ExactIndex(cold, cold_p), cands)
    hot_r, cold_r = rt.CompressedRetriever(hot, hot_p), rt.CompressedRetriever(cold, cold_p)
    tier_r, two_r = (rt.TieredRetriever(hot_r, cold_r, cands),
                     rt.TwoTierRetriever(hot_r, cold_r, cands))
    cases = {
        "float32 corpus": {
            "ExactIndex.search": lambda: direct.search(queries, k),
            "maxsim_matrix + rank": lambda: rank(maxsim_matrix(queries, corpus), k),
            "ExactRetriever.search": lambda: exact.search(queries, k)},
        "binary codes": {
            "ExactIndex.search": lambda: over_codes.search(queries, k),
            "maxsim_matrix + rank": lambda: rank(
                maxsim_matrix(bin_p.transform_queries(queries), binary), k),
            "CompressedRetriever.search": lambda: codes.search(queries, k)},
        "two tiers": {
            "TieredIndex.search": lambda: tiered.search(queries, k),
            "TieredRetriever.search": lambda: tier_r.search(queries, k),
            "TwoTierRetriever.search": lambda: two_r.search(queries, k)}}
    results = {case: compare(fns, reps) for case, fns in cases.items()}
    for case, fns in cases.items():
        ref = positions(next(iter(fns.values()))())
        for name, fn in fns.items():
            add(g + case, name, results[case][name], same_topk=agreement(ref, positions(fn())))
    backend_costs(add, g + BACKEND_CASE, binary,
                  queries.select(np.arange(min(100, len(queries)))), dim, k, reps)
    top, scores = direct.search(queries, k)
    top100, scores100 = direct.search(queries, min(100, len(corpus)))
    names = np.empty(len(corpus), dtype=object)
    names[:] = exact.ids
    base = results["float32 corpus"]["ExactIndex.search"]["median"]
    for name, fn in (
            ("query check (as_queries)", lambda: rt.as_queries(queries, dim)),
            ("Ranking construction, k=10", lambda: rt.ranking_from_positions(
                top, scores, names, rt.SCORE_MAXSIM_IP)),
            ("Ranking construction, k=100", lambda: rt.ranking_from_positions(
                top100, scores100, names, rt.SCORE_MAXSIM_IP))):
        t = measure(fn, reps)
        add(g + "seam parts", name, t, share=t["median"] / base)


def backend_costs(add, group, binary, queries, dim, k, reps):
    """Legacy ``NumpyIndex.search`` loop against ``NumpyBackend.search_many`` on one index.

    ``NumpyIndex`` builds a float32 expansion of its codes on the first search and keeps it, so a
    peak-memory figure depends on which path ran first. ``peak_bytes`` is therefore measured on a
    fresh index for each path (cold cache, the same state for both) and ``peak_bytes_warm`` on a
    shared index whose cache the timing rounds have already built.
    """
    pages = to_legacy_pages(binary)

    def search_loop(index):
        return [index.search(queries.decode_rows(*queries.span(i)), k)
                for i in range(len(queries))]

    with tempfile.TemporaryDirectory() as tmp:  # NumpyIndex touches its path only on save()
        def fresh():
            index = NumpyIndex(tmp, dim, method="binary")
            index.add(pages)
            return index

        cold = {}
        index = fresh()
        cold["legacy NumpyIndex.search loop"] = peak_of(lambda: search_loop(index))
        backend = NumpyBackend(fresh())
        cold["NumpyBackend.search_many"] = peak_of(lambda: backend.search_many(queries, k))

        index = fresh()  # shared by both paths for the timing rounds and the warm peaks
        backend = NumpyBackend(index)
        rows = compare({"legacy NumpyIndex.search loop": lambda: search_loop(index),
                        "NumpyBackend.search_many": lambda: backend.search_many(queries, k)}, reps)
        for name, fn in (("legacy NumpyIndex.search loop", lambda: search_loop(index)),
                         ("NumpyBackend.search_many", lambda: backend.search_many(queries, k))):
            rows[name]["peak_bytes_warm"] = peak_of(fn)  # the cache exists now
            rows[name]["peak_bytes"] = cold[name]
        old = [[hit.ref.page_id for hit in hits] for hits in search_loop(index)]
        new = backend.search_many(queries, k).ids
        add(group, "legacy NumpyIndex.search loop", rows["legacy NumpyIndex.search loop"],
            same_topk=1.0)
        add(group, "NumpyBackend.search_many", rows["NumpyBackend.search_many"],
            same_topk=float(np.mean([a == b for a, b in zip(old, new, strict=True)])))


def s_pipeline(ctx, add):
    for name in ("narrow", "wide"):
        d = ctx["data"][name]
        for label, stages in PIPELINES.items():
            pipeline_costs(add, f"d={d['dim']}, {label}", stages, d, ctx["mid"])


def build(stages, dim):
    return Pipeline(stages(dim))


def pipeline_costs(add, g, stages, d, reps):
    corpus, make = d["corpus"], functools.partial(build, stages, d["dim"])
    pipe = make()
    spec = pipe.to_dict()
    compressed = pipe.compress(corpus)
    states, digests = st.pipeline_fitted_state(pipe), st.pipeline_fitted_identity(pipe)
    add(g, "compress", measure(lambda p: p.compress(corpus), reps, setup=lambda: (make(),),
                               memory=True),
        state_bytes=sum(st.fitted_state_nbytes(s) for s in pipe.stages),
        code_bytes=int(compressed.codes.nbytes))
    add(g, "pipeline_traits", measure(lambda: st.pipeline_traits(pipe), reps))
    add(g, "fitted_state", measure(lambda: st.pipeline_fitted_state(pipe), reps))
    add(g, "fitted_state_digest", measure(lambda: st.pipeline_fitted_identity(pipe), reps))
    add(g, "restore (checked)", measure(
        lambda p: st.restore_pipeline_state(p, states, expected_digests=digests), reps,
        setup=lambda: (Pipeline.from_dict(spec),)))
    add(g, "byte_accounting", measure(lambda: st.byte_accounting(compressed, pipe), reps))


def synthetic_evidence(space, n):
    """Arrays shaped like per-query evidence for ``space``; they mean nothing about any pipeline."""
    rng = np.random.default_rng(SEED + 2)
    k = len(space)
    base = rng.uniform(0.5, 1.0, n)
    drop = rng.uniform(0.0, 1.0, (k, n)) * np.linspace(0.0, 0.3, k)[:, None]
    return {"metric": "ndcg@5", "construct": "labels", "baseline": base,
            "candidates": np.clip(base[None, :] - drop, 0.0, 1.0),
            "query_ids": tuple(f"q{i:05d}" for i in range(n)),
            "bytes_per_doc": np.linspace(4000.0, 100.0, k),
            "vectors_per_doc": np.linspace(50.0, 5.0, k), "space_identity": space.identity}


def s_selection(ctx, add):
    space = sel.CandidateSpace.default(ctx["data"]["narrow"]["dim"])
    with tempfile.TemporaryDirectory() as tmp:
        for n in ctx["ns"]:
            selection_costs(add, space, n, ctx["fast"], Path(tmp))


def selection_costs(add, space, n, fast, tmp):
    f, g = synthetic_evidence(space, n), f"n={n}"
    base, cand, size, vec = f["baseline"], f["candidates"], f["bytes_per_doc"], f["vectors_per_doc"]
    rule, m0 = sel.FixedSequenceBetting(), TARGET / (1.0 + TARGET)
    y = sel.to_y(cand, base[None, :], TARGET)
    stored, path = Evidence(**f), tmp / f"evidence_{n}.npz"
    stored.to_npz(path)  # also caches the digest of ``stored``
    decision = sel.select(space, Evidence(**f), target=TARGET, error=ERROR)
    # select() digests the evidence it is given, so every call needs a new object
    fresh = iter([Evidence(**f) for _ in range(fast + 2)])
    rows = compare({
        "betting_pvalues (K=40)": lambda: sel.betting_pvalues(y, m0),
        "FixedSequenceBetting.decide": lambda: rule.decide(
            cand, base, space.candidate_families, size, vec, TARGET, ERROR),
        "select()": lambda: sel.select(space, next(fresh), target=TARGET, error=ERROR)}, fast)
    for name, row in rows.items():
        extra = {"outcome": decision.outcome.value,
                 "n_certified": len(decision.certified_positions)}
        add(g, name, row, **(extra if name == "select()" else {}))
    add(g, "Evidence()", measure(lambda: Evidence(**f), fast))
    add(g, "Evidence.digest() (fresh)", measure(lambda e: e.digest(), fast,
                                                setup=lambda: (Evidence(**f),)))
    add(g, "to_npz", measure(lambda: stored.to_npz(path), fast), nbytes=path.stat().st_size)
    add(g, "from_npz (digest checked)", measure(lambda: Evidence.from_npz(path), fast))


def instrumented(module, run):
    """Run ``run()`` with timers around the library calls ``module`` makes; ``(seconds per
    component, wall seconds, result)``. ``score_candidate`` reports its own compression and
    scoring seconds."""
    acc = dict.fromkeys(("compression", "scoring", "per-query metrics", "baseline scoring",
                         "fitted-state digests"), 0.0)
    timers = {"per_query_metrics": "per-query metrics", "maxsim_matrix": "baseline scoring",
              "pipeline_fitted_identity": "fitted-state digests"}
    saved = {n: getattr(module, n) for n in (*timers, "score_candidate") if hasattr(module, n)}

    def wrap(name, fn):
        def timed(*a, **kw):
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            if name == "score_candidate":
                acc["compression"] += out.compress_seconds
                acc["scoring"] += out.score_seconds
            else:
                acc[timers[name]] += time.perf_counter() - t0
            return out
        return timed

    for n, fn in saved.items():
        setattr(module, n, wrap(n, fn))
    try:
        t0 = time.perf_counter()
        result = run()
        wall = time.perf_counter() - t0
    finally:
        for n, fn in saved.items():
            setattr(module, n, fn)
    return {**acc, "other (validation, space, Evidence)": wall - sum(acc.values())}, wall, result


def s_collect(ctx, add):
    (collect_evidence,) = need("optivision.core.evaluate.collect", "collect_evidence")
    d = ctx["data"]["e2e"]
    corpus, queries = d["corpus"], d["queries"]
    space, ref = sel.CandidateSpace.default(d["dim"]), LabelReference(d["qrels"])
    g = f"{len(space)} candidates, {len(queries)} queries, {len(corpus)} documents, d={d['dim']}"

    def run():
        return collect_evidence(corpus, queries, space, ref)

    add(g, "collect_evidence", measure(run, ctx["heavy"], memory=True), n_families=space.n_families)
    parts, wall, evidence = instrumented(
        importlib.import_module("optivision.core.evaluate.collect"), run)
    for name, seconds in parts.items():
        add(g, f"{name} (one instrumented call)", {"seconds": seconds}, share=seconds / wall)
    add(g, "evidence arrays (nbytes)", {}, nbytes=int(evidence.candidates.nbytes))
    ctx.update(evidence=evidence, space=space, reference=ref)


def timed_verify(add, verify, cert, level, kwargs, reps):
    report = verify(cert, **kwargs)
    add("certificate", f"verify_certificate ({level})",
        measure(lambda: verify(cert, **kwargs), reps), ok=report.ok,
        level_reached=report.level, n_checks=len(report.checks))


def s_artifact(ctx, add):
    create, verify, save, load = need("optivision.core.artifact", "create_certificate",
                                      "verify_certificate", "save_artifact", "load_artifact")
    (compress_selected,) = need("optivision.core.evaluate.collect", "compress_selected")
    if "evidence" not in ctx:
        raise Unavailable("section 5 (evidence collection) did not run, so there is no decision")
    d, reps, ev, space = ctx["data"]["e2e"], ctx["mid"], ctx["evidence"], ctx["space"]
    decision = sel.select(space, ev, target=TARGET, error=ERROR)
    pipe = comp = None
    if decision.selected is not None:
        pipe, comp = compress_selected(d["corpus"], decision.selected,
                                       expected_fitted=ev.fitted[decision.selected.position])
    live = {"corpus": d["corpus"], "queries": d["queries"], "reference": ctx["reference"],
            "pipeline": pipe, "compressed": comp}

    def make():
        return create(decision, evidence=ev, space=space, created=CREATED, **live)

    cert = make()
    add("certificate", "create_certificate", measure(make, reps), outcome=decision.outcome.value,
        nbytes=len(cert.to_json()))
    for level, kwargs in (("integrity", {}), ("replay", {"evidence": ev}),
                          ("binding", {"evidence": ev, **live})):
        timed_verify(add, verify, cert, level, kwargs, reps)
    codes = None if comp is None else int(comp.codes.nbytes)
    with tempfile.TemporaryDirectory() as tmp:
        count = itertools.count()
        objects = {"certificate": cert, "compressed": comp, "pipeline": pipe, "evidence": ev}
        add("bundle", "save_artifact", measure(
            lambda p: save(p, **objects), reps, setup=lambda: (Path(tmp) / f"b{next(count)}",)))
        bundle = save(Path(tmp) / "kept", **objects)
        add("bundle", "load_artifact (verify=True)", measure(lambda: load(bundle), reps))
        add("bundle", "load_artifact (verify=False)",
            measure(lambda: load(bundle, verify=False), reps))
        for p in sorted(bundle.iterdir()):
            add("bundle files", p.name, {}, nbytes=p.stat().st_size, codes_bytes=codes)


def find_certify():
    """``(certify, dotted name)`` found in optivision or optivision.api, else ``Unavailable``."""
    for module in ("optivision", "optivision.api"):
        try:
            found = getattr(importlib.import_module(module), "certify", None)
        except ImportError:
            found = None
        if found is not None:
            return found, f"{module}.certify"
    raise Unavailable("neither optivision.certify nor optivision.api.certify can be imported")


def s_end_to_end(ctx, add):
    (optimize,) = need("optivision.optimize", "optimize")
    (collect_evidence,) = need("optivision.core.evaluate.collect", "collect_evidence")
    certify, where = find_certify()
    d, heavy = ctx["data"]["e2e"], ctx["heavy"]
    corpus, queries, qrels, box = d["corpus"], d["queries"], d["qrels"], {}
    space, ref = sel.CandidateSpace.default(d["dim"]), LabelReference(qrels)
    g = f"{len(queries)} queries, {len(corpus)} documents, d={d['dim']}"

    def run_certify():
        box["result"] = certify(corpus, queries, qrels, target=TARGET, error=ERROR)

    def run_optimize():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # calibrate warns when it has few queries
            box["optimized"] = optimize(corpus, queries=queries, qrels=qrels, quality_target=TARGET)

    rows = compare({f"{where}()": run_certify, "collect_evidence alone (a part of certify)":
                    lambda: collect_evidence(corpus, queries, space, ref)}, heavy)
    for name, row in rows.items():
        extra = {"outcome": box["result"].outcome.value} if name.endswith("()") else {}
        add(g, name, row, **extra)
    add(g, "optivision.optimize()", measure(run_optimize, heavy, memory=True),
        label=box["optimized"].pipeline.label())


def s_import(ctx, add):
    reps = ctx["fast"]

    def run(code):
        t0 = time.perf_counter()
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              check=True, cwd=ROOT)
        return time.perf_counter() - t0, done.stdout

    code = IMPORT_CODE % json.dumps(LAYER)
    run(code)  # warm-up: compiles any stale .pyc files
    runs = [json.loads(run(code)[1]) for _ in range(reps)]

    def summary(values):
        return {"best": min(values), "median": statistics.median(values), "repeats": reps,
                "all": values}

    add("fresh interpreter", "bare interpreter (python -c pass, wall time)",
        summary([run("pass")[0] for _ in range(reps)]))
    add("fresh interpreter", "import optivision", summary([r["import"] for r in runs]),
        third_party=runs[0]["third_party"])
    add("fresh interpreter", "then import the layer (core.*, backends, api)",
        summary([r["layer"] for r in runs]), third_party=runs[0]["layer_third_party"],
        missing=runs[0]["missing"])


def s_vector_memory(ctx, add):
    for name in ("narrow", "wide"):
        d = ctx["data"][name]
        for label, stages in (("float32", []), ("float16", [st.Float16Quantizer()]),
                              ("int8 (per vector)", [st.Int8Quantizer("per_vector")]),
                              ("binary", [st.BinaryQuantizer()])):
            pipe = Pipeline(stages)
            compressed = pipe.compress(d["corpus"])
            add(f"d={d['dim']}", label, {}, code_bytes_per_vector=int(compressed.codes.shape[1]),
                vectors_per_doc=d["corpus"].num_vectors / len(d["corpus"]),
                **st.byte_accounting(compressed, pipe).to_dict())


SECTIONS = (("interface", s_interface), ("facade", s_facade), ("pipeline", s_pipeline),
            ("selection", s_selection), ("collect", s_collect), ("artifact", s_artifact),
            ("end_to_end", s_end_to_end), ("import", s_import), ("vector_memory", s_vector_memory))


# ------------------------------------------------------------------ environment


def git(*args):
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return done.stdout.strip()


def code_only(source):
    """``ast.dump`` of a module with its import statements and docstrings removed."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        kept = [n for n in body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        if (kept and isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                       ast.AsyncFunctionDef))
                and isinstance(kept[0], ast.Expr) and isinstance(kept[0].value, ast.Constant)
                and isinstance(kept[0].value.value, str)):
            kept = kept[1:]
        node.body = kept
    return ast.dump(tree)


def regression_check():
    """Whether the pre-existing compute paths differ from the base commit (not re-timed here).

    For every file that differs, ``imports_and_docstrings_only`` says whether the two syntax
    trees are equal once import statements and docstrings are removed, i.e. whether the code
    that runs is the base commit's.
    """
    head = git("diff", BASE, "HEAD", "--stat", "--", *FROZEN)
    tree = git("diff", BASE, "--stat", "--", *FROZEN)
    names = sorted(set(git("diff", BASE, "HEAD", "--name-only", "--", *FROZEN).splitlines())
                   | set(git("diff", BASE, "--name-only", "--", *FROZEN).splitlines()))
    changed = {}
    for name in names:
        try:
            same = code_only(git("show", f"{BASE}:{name}")) == code_only(
                (ROOT / name).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            same = False
        changed[name] = {"imports_and_docstrings_only": same}
    return {"base": BASE, "paths": FROZEN, "head_diff_stat": head, "worktree_diff_stat": tree,
            "identical": head == "" and tree == "", "changed": changed}


def regression_text(r):
    """The paragraph on the pre-existing compute paths, from a ``regression_check`` record."""
    command = f"`git diff {r['base']} HEAD --stat -- {' '.join(r['paths'])}` printed "
    if r["identical"]:
        return command + (
            "nothing, and the same command against the working tree printed nothing: those "
            "files are byte-identical to the base commit, so the compute paths that existed "
            "before the package layer were not modified by it and were not timed again here.")
    stat = "\n".join(x for x in dict.fromkeys((r["head_diff_stat"], r["worktree_diff_stat"])) if x)
    changed = r.get("changed")
    if changed is None:  # a record written before the per-file comparison existed
        return command + ("the following, so those files differ from the base commit and their "
                          f"timings may differ too:\n\n```\n{stat}\n```")
    quiet = [n for n, c in changed.items() if c["imports_and_docstrings_only"]]
    loud = [n for n in changed if n not in quiet]
    out = command + ("the following; every listed path that does not appear in it is "
                     f"byte-identical to the base commit:\n\n```\n{stat}\n```\n")
    if quiet:
        notes = "; ".join(f"`{n}`" + (f" ({KNOWN_DIFFERENCES[n]})" if n in KNOWN_DIFFERENCES
                                      else "") for n in quiet)
        out += ("\nDiffering in import statements and docstrings only (the two syntax trees are "
                f"equal once those are removed; checked when this file was generated): {notes}. "
                "The code that runs in those files is the base commit's.")
    if loud:
        out += ("\n\nDiffering in code as well, so timings of these paths may differ from the "
                "base commit's and were NOT re-measured here: " + ", ".join(f"`{n}`" for n in loud)
                + ".")
    else:
        out += (" The compute paths that existed before the package layer therefore run the same "
                "code as at the base commit and were not timed again here.")
    return out


def cpu_model():
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, ImportError):
        pass
    return platform.processor() or None


def ram_and_load():
    """``(total RAM in bytes, system CPU percent over 1 s)``; ``None`` where psutil is absent."""
    try:
        import psutil
    except ImportError:
        return None, None
    return int(psutil.virtual_memory().total), psutil.cpu_percent(interval=1.0)


def blas_info():
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return {"name": blas.get("name"), "version": blas.get("version"),
                "configuration": blas.get("openblas configuration")}
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def environment(note):
    ram, load = ram_and_load()
    changed = git("status", "--porcelain").splitlines()
    try:
        scipy_version = importlib.import_module("scipy").__version__
    except ImportError:
        scipy_version = None
    return {
        "python": sys.version, "numpy": np.__version__, "scipy": scipy_version, "blas": blas_info(),
        "cpu_model": cpu_model(), "logical_cores": os.cpu_count(), "ram_bytes": ram,
        "cpu_percent_at_start": load, "os": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(changed), "git_changed_paths": len(changed),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "machine_load_note": note, "env": {k: os.environ.get(k) for k in (
            "OPTIVISION_SCORE_DEVICE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS")}}


def jsonable(obj):
    """numpy scalars as Python numbers, anything else as text."""
    return obj.item() if hasattr(obj, "item") else str(obj)


def run_all(args):
    mode = "quick" if args.quick else "full"
    fast, mid, heavy = REPEATS[mode]
    data = build_data(mode)
    ctx = {"data": data, "fast": fast, "mid": mid, "heavy": heavy, "ns": SELECTION_N[mode]}
    env = environment(args.load_note)
    bench = {"mode": mode, "repeats": {"fast": fast, "mid": mid, "heavy": heavy},
             "data": {"tokens_per_document": TOKENS[mode], "tokens_per_query": QUERY_TOKENS,
                      "noise": NOISE, "seeds": [SEED, SEED + 1, SEED + 2]},
             "sizes": {n: {"dim": d["dim"], "n_docs": len(d["corpus"]),
                           "n_vectors": d["corpus"].num_vectors, "n_queries": len(d["queries"]),
                           "n_query_vectors": d["queries"].num_vectors} for n, d in data.items()},
             "probes": [{"at": "start", "ms": probe()}], "status": {}, "records": []}
    failed = False
    for key, fn in SECTIONS:
        t0 = time.perf_counter()
        try:
            fn(ctx, functools.partial(add_record, bench["records"], key))
            bench["status"][key] = {"ok": True}
        except Unavailable as exc:
            bench["status"][key] = {"ok": False, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001 - one failed section must not lose the others
            bench["status"][key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()}
            failed = True
        bench["status"][key]["seconds"] = time.perf_counter() - t0
        bench["probes"].append({"at": f"after {key}", "ms": probe()})
        print(f"{key}: {bench['status'][key]}"[:200], flush=True)
    bench["regression_check"] = regression_check()
    env["cpu_percent_at_end"] = ram_and_load()[1]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("bench.json", bench), ("environment.json", env)):
        text = json.dumps(obj, indent=1, default=jsonable)
        (args.out / name).write_text(text + "\n", encoding="utf-8")
    return 1 if failed else 0


# ------------------------------------------------------------------- document


def sig(x):
    return f"{x:.3g}" if abs(x) < 1000 else f"{x:.0f}"


def fs(x):
    """Seconds as text in a sensible unit."""
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if x >= scale:
            return f"{sig(x / scale)} {unit}"
    return f"{sig(x * 1e9)} ns"


def fb(n):
    return f"{sig(n / 1e6)} MB" if n >= 1e6 else f"{sig(n / 1e3)} KB" if n >= 1e3 else f"{n} B"


def pct(x):
    return f"{x * 100:.1f}%" if abs(x) < 10 else f"{x * 100:.0f}%"


def span(values):
    lo, hi = sig(min(values)), sig(max(values))
    return lo if lo == hi else f"{lo} to {hi}"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *("| " + " | ".join(str(c) for c in r) + " |" for r in rows), ""]


def timing(r):
    if "median" in r:
        return f"{fs(r['best'])} / {fs(r['median'])}"
    return fs(r["seconds"]) if "seconds" in r else "-"


def get(B, section, group, name, key="median"):
    """One value of the record (section, group, name)."""
    return next(r for r in B["records"] if (r["section"], r["group"], r["name"]) == (
        section, group, name))[key]


def dims(B):
    return B["sizes"]["narrow"]["dim"], B["sizes"]["wide"]["dim"]


COLUMNS = {  # record key -> (column header, text of the cell for a record that has the key)
    "mb": ("throughput", lambda r: f"{sig(r['mb'] / r['median'])} MB/s"),
    "ratio": ("ratio", lambda r: f"{r['ratio']:.3f}"),
    "share": ("share", lambda r: pct(r["share"])),
    "peak_bytes": ("peak memory", lambda r: fb(r["peak_bytes"])),
    "peak_bytes_warm": ("peak memory, warm cache", lambda r: fb(r["peak_bytes_warm"])),
    "same_topk": ("same top-k", lambda r: pct(r["same_topk"])),
    "vectors_per_doc": ("vectors per doc", lambda r: sig(r["vectors_per_doc"])),
    "code_bytes_per_vector": ("bytes per vector", lambda r: r["code_bytes_per_vector"]),
    "selection_bytes_per_doc": ("bytes per doc, size key",
                                lambda r: sig(r["selection_bytes_per_doc"])),
    "total_bytes_per_doc": ("bytes per doc, total", lambda r: sig(r["total_bytes_per_doc"]))}
NOTES = {  # sparse facts, merged into one column: record key -> text
    "nbytes": lambda r: f"size {fb(r['nbytes'])}",
    "codes_bytes": lambda r: f"{pct(r['nbytes'] / r['codes_bytes'])} of the in-memory codes"
    if r["codes_bytes"] else "no codes",
    "level_reached": lambda r: f"level reached {r['level_reached']}, ok={r['ok']}, "
    f"{r['n_checks']} checks",
    "outcome": lambda r: f"outcome {r['outcome']}" + (f" ({r['n_certified']} certified)"
                                                      if "n_certified" in r else ""),
    "label": lambda r: f"chose {r['label']}",
    "third_party": lambda r: "third-party packages: " + (", ".join(r["third_party"]) or "none")}
PIPELINE_OPS = ("pipeline_traits", "fitted_state", "fitted_state_digest", "restore (checked)",
                "byte_accounting")


def section_table(rs):
    """One row per record; a column for every kind of number that any record of the section has."""
    keys = [k for k in COLUMNS if any(k in r for r in rs)]
    timed = any("median" in r or "seconds" in r for r in rs)
    noted = any(k in r for r in rs for k in NOTES)
    head = ["case", "item", *(["best / median"] if timed else []), *(COLUMNS[k][0] for k in keys),
            *(["notes"] if noted else [])]
    rows, previous = [], None
    for r in rs:
        rows.append([r["group"] if r["group"] != previous else "", r["name"],
                     *([timing(r)] if timed else []), *(COLUMNS[k][1](r) if k in r else "-"
                                                        for k in keys),
                     *(["; ".join(NOTES[k](r) for k in NOTES if k in r)] if noted else [])])
        previous = r["group"]
    return table(head, rows)


def pipeline_table(rs):
    by = {(r["group"], r["name"]): r for r in rs}
    rows = []
    for g in dict.fromkeys(r["group"] for r in rs):
        c = by[g, "compress"]
        rows.append([g, timing(c), fb(c["peak_bytes"]), *(
            f"{fs(by[g, o]['median'])} ({pct(by[g, o]['median'] / c['median'])})"
            for o in PIPELINE_OPS), fb(c["state_bytes"]), fb(c["code_bytes"])])
    return table(["case", "compress best / median", "compress peak", *PIPELINE_OPS, "fitted state",
                  "codes"], rows)


def shows_interface(B):
    m, ds = functools.partial(get, B, "interface"), dims(B)
    valid, ident, sha = ("validate_corpus (documents)", "corpus_identity (documents)",
                         "hashlib.sha256 over the buffer")
    rate = [m(f"d={d}", valid, "mb") / m(f"d={d}", valid) for d in ds]
    ratio = [m(f"d={d}", ident) / m(f"d={d}", sha) for d in ds]
    host = fs(m(f"d={ds[0]}", "as_host_array(ndarray)"))
    plain = "json.dumps(sort_keys, compact)"
    cj = m("candidate list", "canonical_json") / m("candidate list", plain)
    return (f"validate_corpus scanned the document vectors at {span(rate)} MB/s. corpus_identity "
            f"took {span(ratio)}x the time of one hashlib SHA-256 over the same buffer (it also "
            f"copies each block to bytes before hashing). as_host_array on an ndarray took {host} "
            f"and returns the same object. canonical_json of the candidate list took {cj:.1f}x "
            "plain json.dumps with the same options.")


def shows_facade(B):
    m, out = functools.partial(get, B, "facade"), []
    exact, comp, tier = (("float32 corpus", "ExactRetriever.search"),
                         ("binary codes", "CompressedRetriever.search"),
                         ("two tiers", "TieredRetriever.search"))
    two, back = "TwoTierRetriever.search", "NumpyBackend.search_many"
    for d in dims(B):
        g = f"d={d} "
        direct = m(g + "float32 corpus", "ExactIndex.search")
        own = sum(m(g + "seam parts", n) for n in ("query check (as_queries)",
                                                   "Ranking construction, k=10"))
        facades = [m(g + case, name, "ratio") for case, name in (exact, comp, tier)]
        out.append(
            f"At d={d} the three facades took {span(facades)}x the time of the direct calls "
            "(median of within-round ratios); the seam's own parts, measured alone (query check "
            f"and Ranking at k=10), took {fs(own)}, {pct(own / direct)} of the direct float call "
            f"({fs(direct)}). TwoTierRetriever.search took {m(g + 'two tiers', two, 'ratio'):.3f}x "
            "the time of TieredIndex.search (a different code path on the same stores, same top-k "
            f"for {pct(m(g + 'two tiers', two, 'same_topk'))} of the queries) and "
            f"NumpyBackend.search_many {m(g + BACKEND_CASE, back, 'ratio'):.3f}x the legacy "
            "per-query loop (same top-k page ids for "
            f"{pct(m(g + BACKEND_CASE, back, 'same_topk'))}).")
    direct_rows = [r for r in B["records"] if r["section"] == "facade" and r["name"] in (
        "ExactIndex.search", "TieredIndex.search")]
    scatter = max(r["median"] / r["best"] - 1 for r in direct_rows)
    pairs = [(("float32 corpus", "ExactIndex.search"), exact),
             (("binary codes", "ExactIndex.search"), comp),
             (("two tiers", "TieredIndex.search"), tier)]
    peak = max(abs(m(f"d={d} " + c2, n2, "peak_bytes") / m(f"d={d} " + c1, n1, "peak_bytes") - 1)
               for d in dims(B) for (c1, n1), (c2, n2) in pairs)
    sim = [B["sizes"][n]["n_query_vectors"] * B["sizes"][n]["n_vectors"] * 4 / 1e6
           for n in ("narrow", "wide")]
    return " ".join(out) + (
        f" In this run the median of a repeated direct call exceeded its best by up to "
        f"{pct(scatter)}, more than the seam adds, so the within-round ratios cannot resolve the "
        "seam; the measured parts are the estimate of what it adds, and a ratio below 1 is timing "
        "noise, since a facade performs the direct call plus its own parts. Peak traced memory of "
        f"each facade was within {pct(peak)} of its direct call's, about the size of the "
        "query-vector by document-vector similarity block that maxsim_matrix forms "
        f"({sig(sim[0])} MB at d={dims(B)[0]}, {sig(sim[1])} MB at d={dims(B)[1]}).")


def shows_pipeline(B):
    rs = [r for r in B["records"] if r["section"] == "pipeline"]
    med = {(r["group"], r["name"]): r["median"] for r in rs}
    groups = list(dict.fromkeys(r["group"] for r in rs))
    parts = [f"{span([100 * med[g, o] / med[g, 'compress'] for g in groups])}% for {o}"
             for o in PIPELINE_OPS]
    return ("Relative to Pipeline.compress the median cost was " + ", ".join(parts) + " across the "
            "five pipelines and both widths. They read, copy and hash arrays that already exist, "
            "and say nothing about any other pipeline.")


def shows_selection(B):
    m = functools.partial(get, B, "selection")
    g = [r["group"] for r in B["records"] if r["name"] == "select()"][-1]
    n = int(g[2:])
    bp = m(g, "betting_pvalues (K=40)")
    dc, sl = (m(g, o, "ratio") for o in ("FixedSequenceBetting.decide", "select()"))
    npz = m(g, "to_npz") + m(g, "from_npz (digest checked)")
    return (f"betting_pvalues is a Python loop over the queries: {fs(bp / n)} per query step at "
            f"n={n} ({fs(bp)} in all). In rounds that call all three, decide, which adds input "
            f"validation and the fixed-sequence step, took {dc:.3f}x the time of betting_pvalues "
            f"and select(), which adds the Decision and the evidence digest, {sl:.3f}x (medians of "
            f"within-round ratios). At n={n} the evidence digest took "
            f"{fs(m(g, 'Evidence.digest() (fresh)'))} and the npz round trip {fs(npz)} "
            f"({fb(m(g, 'to_npz', 'nbytes'))}).")


def shows_collect(B):
    rs = [r for r in B["records"] if r["section"] == "collect"]
    share = {r["name"].split(" (")[0]: pct(r["share"]) for r in rs if "share" in r}
    return (f"Collecting evidence for {rs[0]['group'].split(' ')[0]} candidates took "
            f"{fs(rs[0]['median'])} (median). In the instrumented call compression was "
            f"{share['compression']}, scoring {share['scoring']}, per-query metrics "
            f"{share['per-query metrics']}, baseline scoring {share['baseline scoring']} and the "
            f"new bookkeeping, the fitted-state digests, {share['fitted-state digests']}.")


def shows_artifact(B):
    m = functools.partial(get, B, "artifact", "certificate")
    b = functools.partial(get, B, "artifact", "bundle")
    files = [r for r in B["records"] if r["group"] == "bundle files"]
    codes, total = files[0]["codes_bytes"], sum(r["nbytes"] for r in files)
    return (f"create_certificate took {fs(m('create_certificate'))}; verify_certificate took "
            f"{fs(m('verify_certificate (integrity)'))} at integrity level, "
            f"{fs(m('verify_certificate (replay)'))} with the evidence and "
            f"{fs(m('verify_certificate (binding)'))} with every live object. save_artifact took "
            f"{fs(b('save_artifact'))}; load_artifact took "
            f"{fs(b('load_artifact (verify=True)'))} verified and "
            f"{fs(b('load_artifact (verify=False)'))} without verification. The bundle holds "
            f"{len(files)} files, {fb(total)} in all"
            + (f", {pct(total / codes)} of the in-memory codes." if codes else
               " (the decision was an abstention, so there are no codes)."))


def shows_end_to_end(B):
    cert, coll, opt = (r for r in B["records"] if r["section"] == "end_to_end")
    share = coll["ratio"]
    return (f"certify() took {fs(cert['median'])} and optimize() {fs(opt['median'])} (medians) on "
            f"this data. Evidence collection, called in the same rounds as certify(), took "
            f"{fs(coll['median'])}, {pct(share)} of the certify() time (median of within-round "
            "ratios)" + (": every candidate of the list is compressed and scored on every query, "
                         "which makes it the dominant cost of a certify() call at these sizes"
                         if share > 0.5 else "")
            + (" (a share at or above 100% is timing noise, since certify() does the same "
               "collection plus its other steps)." if share >= 1 else ".")
            + certify_memory(B, cert))


def certify_memory(B, cert):
    """The sentence on the traced peak of certify() against the float32 corpus it was given."""
    if cert.get("peak_bytes") is None:
        return ""
    size = B["sizes"]["e2e"]
    corpus, block = size["n_vectors"] * size["dim"] * 4, size["n_vectors"] * size["n_query_vectors"] * 4
    return (f" Peak traced memory during certify() was {fb(cert['peak_bytes'])}, "
            f"{cert['peak_bytes'] / corpus:.1f} times the float32 corpus ({fb(corpus)}). The "
            "multiple belongs to these sizes and is not a constant: the peak holds the transform "
            "cache that the candidates share (one whole transformed corpus per distinct merge or "
            "projection prefix; cap `OPTIVISION_CACHE_BYTES`, 4 GiB by default), the live block of "
            f"query-vector by document-vector similarities ({fb(block)} if all of it is one block, "
            "as it is at this size; capped by `max_block_bytes`, 256 MiB by default), and smaller "
            "terms.")


def shows_import(B):
    bare, imp, layer = (r for r in B["records"] if r["section"] == "import")
    more = f" and loaded {', '.join(layer['third_party'])}" if layer["third_party"] else ""
    miss = f"; not importable at measurement time: {', '.join(layer['missing'])}" if layer[
        "missing"] else ""
    return (f"`import optivision` took {fs(imp['median'])} (median) after a bare interpreter "
            f"start of {fs(bare['median'])}; it loaded {', '.join(imp['third_party'])}. Importing "
            f"the layer afterwards took {fs(layer['median'])}{more}{miss}.")


def shows_memory(B):
    def size(d, name, key="selection_bytes_per_doc"):
        return get(B, "vector_memory", f"d={d}", name, key)

    ratios = "; ".join(
        f"d={d}: " + ", ".join(f"{n.split(' ')[0]} {size(d, 'float32') / size(d, n):.2f}x"
                               for n in ("float16", "int8 (per vector)", "binary"))
        for d in dims(B))
    offsets = size(dims(B)[0], "float32", "total_bytes_per_doc") - size(dims(B)[0], "float32")
    fitted = sum(r["fitted_state_bytes"] for r in B["records"] if r["section"] == "vector_memory")
    return ("The sizes follow from the formats: 4 bytes per coordinate for float32, 2 for float16, "
            "one plus a 2-byte scale per vector for per-vector int8, and one bit for binary. "
            f"Float32 bytes over the size key: {ratios}. The total adds {sig(offsets)} bytes per "
            "document (the document offsets) to the size key"
            + ("; none of these formats has fitted state." if fitted == 0 else "."))


DOC = (
    ("interface", "1. Interface overhead",
     "Per-call figures; MB = 10^6 bytes; throughput is the size of the data over the median time.",
     shows_interface),
    ("facade", "2. Facade overhead",
     ("Variants of one case run round-robin on the same data; `ratio` is the median over rounds of "
      "the within-round time ratio to the leading row of the case. All queries of a set are scored "
      "as one batch, k=10, shortlist min(50, documents/4) and at least k. `same top-k`: share of "
      "queries whose top-k positions (page ids for the NumPy case) equal the leading row's. The "
      "NumPy case scores one query at a time and uses at most 100 queries. A seam part's `share` "
      "is its time over the direct float call's."), shows_facade),
    ("pipeline", "3. Pipeline overhead",
     ("Median time of each call with its share of the median of `Pipeline.compress` (fit, "
      "transform, encode) in brackets; `restore` is `restore_pipeline_state` into a fresh pipeline "
      "with the digest check."), shows_pipeline),
    ("selection", "4. Selection",
     ("Synthetic arrays for the default 40-candidate space, target 0.97, error 0.05; the case is "
      "the number of queries n. The `select()` outcome is an output of the code on random arrays."),
     shows_selection),
    ("collect", "5. Evidence collection",
     ("`collect_evidence` over the default list with the label reference (the case names the "
      "sizes). The split is from timers wrapped around the library calls of one further call; its "
      "`share` is of that call."), shows_collect),
    ("artifact", "6. Certificate and artifact",
     ("One decision on the evidence of section 5; `verify_certificate` is timed at each of the "
      "three levels of `CERTIFICATE_SPEC.md`."), shows_artifact),
    ("end_to_end", "7. End to end",
     ("One synthetic corpus, default candidate list, target 0.97, error 0.05, label reference. "
      "`optimize()` is for context only: it is a different procedure (it splits the queries into "
      "calibration and held-out halves, judges candidates by a bootstrap lower bound and may skip "
      "candidates) and returns different things, so the two times are not a comparison of speed "
      "or of results."), shows_end_to_end),
    ("import", "8. Import",
     ("Fresh interpreters: the in-process time of `import optivision`, then of importing the layer "
      "(`optivision.core.*`, backends, api) in the same process, after one warm-up run."),
     shows_import),
    ("vector_memory", "9. Vector memory",
     ("Bytes per document from `byte_accounting`: pure arithmetic of each format (vectors per "
      "document times bytes per vector, plus the codec's shared state). A size, not a speed or a "
      "quality figure."), shows_memory))
LIMITS = (
    ("other hardware, BLAS builds, thread counts, or Python and NumPy versions; this laptop's two "
     "CPU speed states mean that even on this machine absolute times differ between runs"),
    ("real corpora or real encoder output: the vectors are Gaussian, so merging, codecs and "
     "selection behave as on random data, and a selection outcome here is an output of the code, "
     "not a finding"),
    ("larger scale: sizes stop at a few hundred documents, and the dense score matrix and the "
     "decoded blocks grow with the corpus; nothing here is a statement about large corpora"),
    ("memory of `certify()` on a large corpus: section 7 gives one traced peak at one small size; "
     "the transform cache alone can hold several whole transformed corpora until its cap, beyond "
     "which it evicts and recomputes (slower, same results), and that regime was not measured"),
    "a GPU or `OPTIVISION_SCORE_DEVICE` (all scoring ran on the CPU through NumPy)",
    "single-query latency: no timing here is of one query alone",
    "concurrency, memory outside tracemalloc's view, or disk speed beyond one temporary directory",
    "retrieval quality, agreement with the uncompressed ranking, or anything about an encoder",
    "any relation to the Q4, Q5/Q6 or Q8 measurements or to the registered experiments")


def environment_rows(B, E):
    probes, blas = [p["ms"] for p in B["probes"]], E["blas"]
    return [
        ["Python", E["python"].split()[0]], ["NumPy / SciPy", f"{E['numpy']} / {E['scipy']}"],
        ["BLAS", blas.get("error")
         or f"{blas['name']} {blas['version']} ({blas['configuration']})"],
        ["CPU", f"{E['cpu_model']}, {E['logical_cores']} logical cores"],
        ["RAM", fb(E["ram_bytes"]) if E["ram_bytes"] else "not recorded"], ["OS", E["os"]],
        ["commit", f"`{E['git_commit']}` on `{E['git_branch']}`, working tree " + (
            f"DIRTY ({E['git_changed_paths']} paths changed or untracked)" if E["git_dirty"]
            else "clean")],
        ["run (UTC)", E["utc_timestamp"]],
        ["system CPU use over 1 s", (f"at start {E['cpu_percent_at_start']}%, at end "
                                     f"{E.get('cpu_percent_at_end')}%")],
        ["CPU speed probe (768x768 float32 product, best of 5)",
         (f"start {sig(probes[0])} ms, end {sig(probes[-1])} ms; range {sig(min(probes))} to "
          f"{sig(max(probes))} ms ({max(probes) / min(probes):.2f}x)")],
        ["environment variables", ", ".join(f"{k}={'unset' if v is None else v}"
                                            for k, v in E["env"].items())],
        ["machine load note", E["machine_load_note"]]]


def build_doc(B, E):
    d, t, q = B["data"], B["data"]["tokens_per_document"], B["data"]["tokens_per_query"]
    note = E["machine_load_note"].strip()
    sizes = [[n, s["dim"], s["n_docs"], s["n_vectors"], s["n_queries"], s["n_query_vectors"], used]
             for (n, s), used in zip(B["sizes"].items(), ("1, 2, 3, 9", "1, 2, 3, 9", "5, 6, 7"),
                                     strict=True)]
    out = [
        "# Engineering benchmarks of the package layer", "",
        ("**Every number in this document is a NEW ENGINEERING MEASUREMENT on SYNTHETIC data on "
         "ONE laptop. It is not a research result, it is not comparable with the research "
         "reports (`reports/research/`, Q4, Q5/Q6, Q8 or any registered experiment: different "
         "data, different purpose), and it is not evidence about retrieval quality or about any "
         "encoder.**"), "",
        *(["**SMOKE RUN (`--quick`): tiny sizes, not for reporting.**", ""] if B["mode"] == "quick"
          else []),
        ("Generated by `scripts/engineering/package_bench.py --write-doc` from "
         "`reports/engineering/package_benchmarks/bench.json` and `environment.json`; do not edit "
         "by hand. To refresh every number, run `PYTHONPATH=src python "
         "scripts/engineering/package_bench.py --write-doc docs/package/ENGINEERING_BENCHMARKS.md "
         "--load-note '<what else was running>'` (it measures, then rewrites this file). "
         f"Mode `{B['mode']}`."), "",
        (f"**Status of these numbers.** {note}{'' if note.endswith('.') else '.'} Absolute times "
         "are indicative only; the robust quantities are ratios inside one run (a facade against "
         "the direct call, bookkeeping against the call it accompanies). This laptop also has two "
         "CPU speed states about 2.8x apart, so absolute times are comparable only within one run; "
         "the speed probe below shows whether the speed moved during this run."), "",
        "## Environment", "", *table(["item", "value"], environment_rows(B, E)),
        "## Method", "",
        ("- **Time.** `time.perf_counter`; best and median of N repeats of one call after one "
         "untimed warm-up call (N is stated in each section). A call under about 20 ms is looped "
         "inside a repeat and reported per call; `gc.collect()` runs before each repeat. "
         "Comparisons run in rounds, every variant once per round with a rotating start (sections "
         "2, 4 and 7); the shares in sections 1, 3 and 5 compare calls timed one after the "
         "other and carry the drift of the run."),
        ("- **Memory.** Peak of `tracemalloc` during the warm-up call (Python and NumPy "
         "allocations, not BLAS-internal buffers), and array `nbytes`."),
        (f"- **Data.** `optivision.testing.make_corpus` and `make_queries`, seeds {d['seeds'][0]} "
         f"and {d['seeds'][1]}: {t[0]} to {t[1]} unit-norm Gaussian vectors per document, {q[0]} "
         f"to {q[1]} per query cut from a labelled document with noise {d['noise']}. No model "
         "output and no research store is read."),
        ("- **Scope.** One process, default BLAS threading, CPU only; every call is given a whole "
         "query set (the NumPy index scores its queries one at a time inside the call). Nothing "
         "here evaluates retrieval quality."), "",
        *table(["configuration", "dim", "documents", "document vectors", "queries", "query vectors",
                "used in sections"], sizes)]
    for key, title, intro, shows in DOC:
        status, rs = B["status"][key], [r for r in B["records"] if r["section"] == key]
        out += [f"## {title}", ""]
        if not status["ok"]:
            out += [f"FAILED at measurement time: {status['error']}" if "error" in status else
                    f"Not available at measurement time ({status['reason']}).", ""]
            continue
        reps = sorted({r["repeats"] for r in rs if "repeats" in r})
        out += [intro + (f" N = {'/'.join(map(str, reps))} repeats (rounds) after one warm-up."
                         if reps else ""), "",
                *(pipeline_table(rs) if key == "pipeline" else section_table(rs)),
                f"**What this shows.** {shows(B)}", ""]
    r = B["regression_check"]
    out += ["## Pre-existing compute paths: not re-measured", "",
            regression_text(r), "",
            "## What these numbers do not show", "", *(f"- {x}" for x in LIMITS), ""]
    return "\n".join(out)


def write_doc(out_dir, path):
    bench = json.loads((out_dir / "bench.json").read_text(encoding="utf-8"))
    env = json.loads((out_dir / "environment.json").read_text(encoding="utf-8"))
    text = build_doc(bench, env)
    text.encode("ascii")  # the document is ASCII only; fail loudly otherwise
    Path(path).write_text(text, encoding="ascii", newline="\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke run")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="directory for bench.json/environment.json")
    ap.add_argument("--write-doc", type=Path, help="write the Markdown document from the JSON")
    ap.add_argument("--no-run", action="store_true", help="do not measure; only write the document")
    ap.add_argument("--load-note", default="Machine load during this run: not stated.")
    args = ap.parse_args()
    if args.quick and args.out == OUT:
        ap.error("--quick needs its own --out, so that a smoke run cannot replace real numbers")
    status = 0 if args.no_run else run_all(args)
    if args.write_doc:
        write_doc(args.out, args.write_doc)
    return status


if __name__ == "__main__":
    sys.exit(main())
