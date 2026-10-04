"""X3: coordinate and symmetry test (NEXT_PHASE_PREREGISTRATION.md, section 12).

Four registered maps are applied to the stored vectors of a dataset: a signed coordinate
permutation (T1), two dense rotations (T2a, T2b) and a reordering of the rows inside each
document (T3). Each leaves every MaxSim score unchanged and keeps the vectors unit-norm.
T0 is the identity. The per-query store is rebuilt on the transformed vectors with the
calls E1.0 used, and P0 is run on it. H5 is a statement about these four maps on the four
primary datasets within the registered tolerances, and about nothing wider.

    python scripts/research/x3_symmetry.py                            dry run: validates, computes nothing
    python scripts/research/x3_symmetry.py --run                      the primary set (D1, D4, D5, D7)
    python scripts/research/x3_symmetry.py --run --dataset scifact    one dataset, its five transformations
    python scripts/research/x3_symmetry.py --run --extension          D2, D3, D6: only after the primary set

One (dataset, transformation) is one unit of work, tens of minutes long, with at most three
attempts. Finished units are kept; the analysis and the report are rebuilt from them.
Outputs go to reports/research/X3/. Run one dataset at a time on a small machine:

    OPTIVISION_DATA=../optivision-rag-v2/data OPTIVISION_CACHE_BYTES=750000000 \\
        PYTHONPATH="src;scripts/research" python scripts/research/x3_symmetry.py --run --dataset scifact
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e1_common as E  # noqa: E402
import x_common as X  # noqa: E402

from optivision.calibration import recommended_search_space, score_space  # noqa: E402
from optivision.evaluation import per_query_metrics, relevant_from_baseline  # noqa: E402
from optivision.representation import MultiVectorCorpus  # noqa: E402
from optivision.scoring import maxsim_matrix, rank  # noqa: E402

EXPERIMENT, PREFIX = "X3", "x3"
TRANSFORMATIONS = {"T0": None, "T1": 0, "T2a": 1, "T2b": 2, "T3": 3}  # id: registered seed (section 12.3)
KIND = {"T0": "identity", "T1": "signed coordinate permutation", "T2a": "dense random rotation",
        "T2b": "dense random rotation", "T3": "reordering of the rows of each document"}
ROW_ANNOTATIONS = ("protected", "positions", "weights", "importance")
ROW_BLOCK = 65536
MAX_ATTEMPTS = 3
SCORE_TOLERANCE, METRIC_IDENTICAL, MEAN_TOLERANCE = 1e-4, 0.99, 0.001  # symmetry check, section 12.5 step 3
T0_RETENTION_TOLERANCE, T0_IDENTICAL = 0.001, 0.995  # T0 gate fallback, section 12.5 step 2
INVARIANT, MATERIAL = 0.005, 0.02  # classes of a candidate's change, section 12.6
RATIO, RATE_DIFFERENCE = 2.0, 0.25  # materiality of the procedure outcome, section 12.6
ARRAYS = {"labels": ("base_labels", "cand_labels"), "agreement": ("base_b1", "cand_b1")}
N11 = "The result for {subject} on dataset {dataset} is not invariant under the registered rotation."
N12 = ("The implementation's result for {subject} on dataset {dataset} changes with the order of "
       "coordinates or of vectors.")


def unit(name: str, tid: str) -> str:
    return f"{name}__{tid}"


# ------------------------------------------------------------ transformations


def signed_permutation(dim: int, seed: int) -> np.ndarray:
    """T1 as a matrix M: coordinate j of M v is signs[j] * v[perm[j]] (open point OP1)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dim)
    signs = rng.integers(0, 2, size=dim) * 2 - 1
    m = np.zeros((dim, dim))
    m[np.arange(dim), perm] = signs
    return m


def rotation(dim: int, seed: int) -> np.ndarray:
    """T2: the orthogonal factor of a QR decomposition of a d x d standard normal matrix, each
    column multiplied by the sign of the corresponding diagonal entry of R (section 12.3)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


def apply_matrix(vectors: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Every row v becomes M v: applied in float64, stored as float32, not re-normalised (OP3)."""
    out = np.empty(vectors.shape, dtype=np.float32)
    for lo in range(0, vectors.shape[0], ROW_BLOCK):
        out[lo:lo + ROW_BLOCK] = np.asarray(vectors[lo:lo + ROW_BLOCK], dtype=np.float64) @ m.T
    return out


def row_order(offsets: np.ndarray, seed: int) -> np.ndarray:
    """T3: a new order of the rows inside each document; rows never leave their document (OP2)."""
    rng = np.random.default_rng(seed)
    parts = [int(lo) + rng.permutation(int(hi) - int(lo)) for lo, hi in zip(offsets[:-1], offsets[1:], strict=True)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def transform(corpus: MultiVectorCorpus, queries: MultiVectorCorpus, tid: str) -> tuple:
    """(corpus, queries, record) after one registered transformation. The record carries the hash of
    the matrix or of the row order (section 22.2)."""
    seed = TRANSFORMATIONS[tid]
    record = {"id": tid, "kind": KIND[tid], "seed": seed}
    if tid == "T0":
        return corpus, queries, {**record, "applied_to": "nothing", "sha256": None}
    if tid == "T3":
        rows = row_order(corpus.offsets, seed)
        moved = {a: None if getattr(corpus, a) is None else getattr(corpus, a)[rows] for a in ROW_ANNOTATIONS}
        new = MultiVectorCorpus(corpus.vectors[rows], corpus.offsets, ids=corpus.ids, metadata=corpus.metadata,
                                images=corpus.images, attrs=corpus.attrs, **moved)
        return new, queries, {**record, "applied_to": "documents",
                              "hashed": "the row order, little-endian int64",
                              "sha256": X.sha256_bytes(rows.astype("<i8").tobytes())}
    m = signed_permutation(corpus.dimension, seed) if tid == "T1" else rotation(corpus.dimension, seed)
    record.update({"applied_to": "documents and queries", "hashed": "the matrix, little-endian float64, C order",
                   "sha256": X.sha256_bytes(np.ascontiguousarray(m, dtype="<f8").tobytes()),
                   "orthogonality_error": float(np.abs(m @ m.T - np.eye(m.shape[0])).max())})
    return (corpus.with_vectors(apply_matrix(corpus.vectors, m)),
            queries.with_vectors(apply_matrix(queries.vectors, m)), record)


# --------------------------------------------------------- store, gates, P0


def load_dataset(name: str) -> tuple:
    """(dataset, metric, vector files) for a registered store, loaded as E1.0 loaded it."""
    s7b = E.load_s7b()
    ds, metric = s7b.dataset(E.dataset_arg(name))
    return ds, metric, {pathlib.Path(p).name: X.sha256_file(p) for p in vector_files(name, s7b)}


def vector_files(name: str, s7b) -> list:
    """The files ``s7b.dataset`` reads for a store (same path rules as that function)."""
    arg = E.dataset_arg(name)
    if arg.startswith("vec:"):
        stem = f"{s7b.D}/vectors/{arg[4:]}"
        return [f"{stem}_docs.npz", f"{stem}_queries.npz", f"{stem}_qrels.json"]
    if arg == "text-scifact":
        stem = f"{s7b.D}/vectors/scifact"
        return [f"{stem}_answerai-colbert-small-v1_docs.npz", f"{stem}_answerai-colbert-small-v1_queries.npz",
                f"{stem}_qrels_test.json"]
    cache, queries_json = s7b.LEGACY[arg]
    return [cache, str(pathlib.Path(cache).with_suffix(".queries.npz")), queries_json]


def build_store(ds, metric: str, corpus, queries, base: np.ndarray) -> tuple:
    """The per-query store of one (transformed) dataset: the calls of ``e1_0_rebuild.build``."""
    rel_labels = ds.relevant()
    rel_base = relevant_from_baseline(base, 1)
    space = recommended_search_space(corpus.dimension)
    scored = score_space(corpus, queries, space,
                         progress=lambda sc: print(f"  scored {sc.label:48s} x{sc.report['compression_vs_float32']:7.1f}",
                                                   flush=True))
    arrays = {
        "base_labels": per_query_metrics(base, rel_labels, [metric])[metric],
        "base_b1": per_query_metrics(base, rel_base, [metric])[metric],
        "cand_labels": np.stack([per_query_metrics(sc.scores, rel_labels, [metric])[metric] for sc in scored]),
        "cand_b1": np.stack([per_query_metrics(sc.scores, rel_base, [metric])[metric] for sc in scored]),
    }
    n_rel = [int(len(r)) for r in rel_labels]
    meta = {
        "dataset": ds.name, "metric": metric, "n_docs": len(corpus), "n_queries": len(queries),
        "dim": corpus.dimension, "relevant_per_query": {str(k): n_rel.count(k) for k in sorted(set(n_rel))},
        "candidates": [{"position": i, "family": sc.family, "label": sc.label, "pipeline": sc.pipeline,
                        "bytes_per_doc": sc.bytes_per_doc,
                        "vectors_per_doc": float(sc.report["vectors_per_doc"]),
                        "compression_vs_float32": float(sc.report["compression_vs_float32"]),
                        "dim": int(sc.report["dim"]), "bits_per_dim": float(sc.report["bits_per_dim"])}
                       for i, sc in enumerate(scored)],
    }
    return arrays, meta, scored


def symmetry_check(base0: np.ndarray, base: np.ndarray, metric0: np.ndarray, metric: np.ndarray) -> dict:
    """Section 12.5 step 3, on the uncompressed vectors. A failure means the map is implemented wrongly."""
    finite = np.isfinite(base0)  # an empty document scores -inf under every map
    same = bool(np.array_equal(finite, np.isfinite(base)))
    scale = float(np.abs(base0[finite]).max())
    worst = float(np.abs(base[finite].astype(np.float64) - base0[finite]).max()) / scale if same else float("inf")
    identical = float((metric == metric0).mean())
    mean_difference = float(abs(metric.mean() - metric0.mean()))
    return {"max_abs_score_difference_over_max_abs_score": worst, "score_tolerance": SCORE_TOLERANCE,
            "share_of_queries_with_identical_metric": identical, "share_required": METRIC_IDENTICAL,
            "pool_mean_difference": mean_difference, "mean_tolerance": MEAN_TOLERANCE,
            "passed": bool(worst <= SCORE_TOLERANCE and identical >= METRIC_IDENTICAL
                           and mean_difference <= MEAN_TOLERANCE)}


def t0_gate(arrays: dict, meta: dict, committed: dict) -> dict:
    """Section 12.5 step 2: the T0 store must equal the committed E1 store. If it does not, the gate
    still passes when all 40 pool retentions agree within 0.001 and at least 99.5% of the per-query
    entries are identical (open point OP4: both constructs, each of the four arrays)."""
    keys = [k for pair in ARRAYS.values() for k in pair]
    same_shape = all(arrays[k].shape == committed[k].shape for k in keys)
    sizes = ("bytes_per_doc", "vectors_per_doc", "compression_vs_float32")
    out = {
        "candidate_specification_equal": X.spec_sha256(X.candidate_spec(meta["candidates"]))
        == X.spec_sha256(X.candidate_spec(committed["meta"]["candidates"])),
        "size_metadata_equal": all(a[s] == b[s] for a, b in zip(meta["candidates"], committed["meta"]["candidates"],
                                                               strict=False) for s in sizes),
        "arrays_equal": bool(same_shape and all(np.array_equal(arrays[k], committed[k]) for k in keys)),
    }
    out["equal"] = bool(out["arrays_equal"] and out["size_metadata_equal"] and out["candidate_specification_equal"])
    if out["equal"] or not same_shape:
        out["max_abs_difference"] = 0.0 if out["equal"] else None
        out["passed"] = out["equal"]
        return out
    out["max_abs_difference"] = float(max(np.abs(arrays[k] - committed[k]).max() for k in keys))
    retention = {c: float(np.abs(arrays[ck].sum(axis=1) / arrays[bk].sum()
                                 - committed[ck].sum(axis=1) / committed[bk].sum()).max())
                 for c, (bk, ck) in ARRAYS.items()}
    identical = {k: float((arrays[k] == committed[k]).mean()) for k in keys}
    out["fallback"] = {"max_pool_retention_difference": retention, "share_of_identical_entries": identical,
                       "passed": bool(all(v <= T0_RETENTION_TOLERANCE for v in retention.values())
                                      and all(v >= T0_IDENTICAL for v in identical.values()))}
    out["passed"] = bool(out["candidate_specification_equal"] and out["fallback"]["passed"])
    return out


def procedure(arrays: dict, meta: dict, reg: X.Registry) -> tuple:
    """P0 on one store with the X2 resampling protocol: labels, three thresholds, delta = 0.05,
    n = 225 and the pool-capped budget (section 12.5 step 4)."""
    st = E.Store({"meta": meta, **arrays}, "labels")
    prob = X.problem_for(st)
    n_pool = st.pq.shape[1]
    budgets = tuple(sorted({reg.x3_budget, reg.pool_capped_budget(n_pool)}))
    f16 = X.first_position(meta, X.FLOAT16_PIPELINE)
    idx, nested = X.sample_indices(n_pool, range(reg.n_seeds), budgets, reg)
    sel = X.simulate(st.pq, st.base, prob, idx, X.PRIMARY_THRESHOLDS, deltas=(X.PRIMARY_DELTA,), methods=("C",))
    X.check_monotone(sel, prob, budgets, X.PRIMARY_THRESHOLDS, nested, meta["dataset"], deltas=(X.PRIMARY_DELTA,))
    rows = []
    for n in budgets:
        for target in X.PRIMARY_THRESHOLDS:
            k = sel[X.sel_key(n, target, f"C@{X.PRIMARY_DELTA}")]
            flags = X.deployed_flags(k, prob, target, f16)
            rows.append({**X.cell_identity("labels", target, X.PRIMARY_DELTA, n, n_pool, "P0"),
                         "seeds": int(k.shape[0]), "deployed_rate": float(flags["deployed"].mean()),
                         "nontrivial_rate": float(flags["nontrivial"].mean()),
                         "compression_median": float(np.median(X.compression_of(k, prob)))})
    return sel, rows, nested


def run_unit(name: str, tid: str, loaded: tuple, base0: np.ndarray, reg: X.Registry, log: X.RunLog,
             validation: dict) -> bool:
    """One (dataset, transformation): transform, check, rebuild the store, run P0, write. True if valid."""
    ds, metric, files = loaded
    u = unit(name, tid)
    entry, timer = log.begin(u, X.git_state(reg.root).get("commit")), X.Timer()

    def stop(error, message: str, gates: dict):
        log.end(entry, "invalid", reason=message, gates=gates, **timer.record())
        raise error(f"X3 {u}: {message}")

    corpus, queries, record = transform(ds.corpus, ds.queries, tid)
    base = base0 if tid == "T0" else maxsim_matrix(queries, corpus)
    gates = {}
    if tid != "T0":
        metric0 = per_query_metrics(base0, ds.relevant(), [metric])[metric]
        gates["symmetry"] = symmetry_check(base0, base, metric0, per_query_metrics(base, ds.relevant(), [metric])[metric])
        if not gates["symmetry"]["passed"]:
            stop(X.GateFailure, "the symmetry check failed: the transformation is implemented incorrectly", gates)
    arrays, meta, scored = build_store(ds, metric, corpus, queries, base)
    meta["transformation"] = record
    meta["evidence"] = X.MEASURED if tid == "T0" else f"{X.MEASURED}, transformed vectors"
    gates["candidate_specification"] = {"sha256": X.spec_sha256(X.candidate_spec(meta["candidates"])),
                                        "registered": reg.s0_spec_sha256}
    gates["candidate_specification"]["passed"] = (gates["candidate_specification"]["sha256"] == reg.s0_spec_sha256)
    if not gates["candidate_specification"]["passed"]:
        stop(X.RegisteredInputMismatch, "the generated candidate list does not hash to the registered value", gates)
    gates["values"] = X.gate_values(arrays)
    if not gates["values"]["passed"]:
        stop(X.GateFailure, "a per-query value is not finite or not in [0, 1]", gates)
    lists = None
    if tid == "T0":
        gates["T0"] = t0_gate(arrays, meta, X.load_store(name, reg))
        if not gates["T0"]["passed"]:
            stop(X.GateFailure, "the T0 store does not reproduce the committed E1 store", gates)
        depth = min(10, len(corpus))  # the lists behind overlap@5 and overlap@10 (section 10.6)
        lists = {"reference_top10": rank(base, depth).astype(np.int32),
                 "candidate_top10": np.stack([rank(sc.scores, depth) for sc in scored]).astype(np.int32)}
    del scored, corpus, queries
    gc.collect()
    sel, rows, nested = procedure(arrays, meta, reg)
    if tid == "T0" and gates["T0"]["equal"] and name in reg.e1_selections and reg.x3_budget in reg.e1_budgets:
        gates["P0_reproduces_E1"] = X.gate_reproduction(name, sel, reg, [f"C@{X.PRIMARY_DELTA}"], (reg.x3_budget,))
        if not gates["P0_reproduces_E1"]["passed"]:
            stop(X.GateFailure, "P0 on the T0 store does not reproduce the selections stored by E1", gates)
    rec = timer.record()
    if rec["slept"]:
        log.end(entry, "invalid", reason="the machine slept during the run", gates=gates, **rec)
        return False
    files_out = {f"store/{u}.npz": arrays, f"store/{u}.json": meta, f"selections/{u}.npz": sel}
    if lists is not None:
        files_out[f"lists/{u}_top10.npz"] = lists
    outputs = {f: (X.write_npz if f.endswith(".npz") else X.write_json)(X.out_path(reg, EXPERIMENT, *f.split("/")), c)
               for f, c in files_out.items()}
    X.write_json(X.out_path(reg, EXPERIMENT, "units", f"{u}.json"), {
        "header": X.header(EXPERIMENT, reg, validation), "dataset": name, "transformation": record,
        "evidence": meta["evidence"], "vector_files_sha256": files, "gates": gates,
        "P0": {"evidence": "SIMULATED: resamples of the pool of this store; both budgets are pool-supported",
               "nested_across_budgets": nested, "rows": rows},
        "outputs": outputs})
    log.end(entry, "valid", outputs=outputs, **rec)
    print(f"X3 {u}: done in {rec['seconds']:.0f}s", flush=True)
    return True


def run_dataset(name: str, reg: X.Registry, loader, log: X.RunLog, validation: dict) -> None:
    """The units of one dataset that are still missing, T0 first. A failed gate stops X3."""
    todo = [t for t in TRANSFORMATIONS if log.valid(unit(name, t)) is None
            and len(log.attempts(unit(name, t))) < MAX_ATTEMPTS]
    if not todo:
        return
    loaded = loader(name)
    base0 = maxsim_matrix(loaded[0].queries, loaded[0].corpus)
    for tid in todo:
        if tid != "T0" and log.valid(unit(name, "T0")) is None:
            print(f"X3 {name}: no valid T0 unit, so its other transformations are not run", flush=True)
            break
        run_unit(name, tid, loaded, base0, reg, log, validation)
        gc.collect()


# ------------------------------------------------------------------- analysis


def classify(delta: float, lo: float, hi: float) -> str:
    if abs(delta) <= INVARIANT:
        return "invariant"
    if abs(delta) >= MATERIAL and (lo > 0.0 or hi < 0.0):
        return "material"
    return "small"


def candidate_deltas(t: dict, t0: dict, n_boot: int) -> dict:
    """Per candidate and construct: pool retention after the map minus before, with a 95% interval
    from a paired bootstrap over queries (seed 0, percentile; section 12.6)."""
    n = t0["base_labels"].shape[0]
    idx = np.random.default_rng(0).integers(0, n, size=(n_boot, n))  # as optivision.evaluation.retention draws
    out = {}
    for construct, (bk, ck) in ARRAYS.items():
        bt, b0 = t[bk][idx].sum(axis=1), t0[bk][idx].sum(axis=1)
        rows = []
        for k in range(t[ck].shape[0]):
            after, before = float(t[ck][k].sum() / t[bk].sum()), float(t0[ck][k].sum() / t0[bk].sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                boot = t[ck][k][idx].sum(axis=1) / bt - t0[ck][k][idx].sum(axis=1) / b0
            lo, hi = (float(v) for v in np.nanquantile(boot, [0.025, 0.975]))
            rows.append({"position": k, "retention_T0": before, "retention": after, "delta": after - before,
                         "ci95": [lo, hi], "class": classify(after - before, lo, hi)})
        out[construct] = rows
    return out


def overlap(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    """Size of the intersection of two top-k lists, divided by k, averaged over queries (section 10.6)."""
    hits = (candidate[:, :k, None] == reference[:, None, :k]).any(axis=2).sum(axis=1)
    return float((hits / k).mean())


def analyse_dataset(name: str, reg: X.Registry) -> dict | None:
    """Everything H5 needs for one dataset, rebuilt from its unit files. None until T0 exists."""
    def path(*parts):
        return X.out_path(reg, EXPERIMENT, *parts)

    units = {t: X.read_json(path("units", f"{unit(name, t)}.json")) for t in TRANSFORMATIONS
             if path("units", f"{unit(name, t)}.json").is_file()}
    if "T0" not in units:
        return None
    stores = {t: dict(np.load(path("store", f"{unit(name, t)}.npz"), allow_pickle=False)) for t in units}
    meta = X.read_json(path("store", f"{unit(name, 'T0')}.json"))
    distinct, labels = X.distinct_candidates(meta), [c["label"] for c in meta["candidates"]]
    reference = {(r["n"], r["threshold"]): r for r in units["T0"]["P0"]["rows"]}
    pairs = {}
    for tid in units:
        if tid == "T0":
            continue
        deltas = candidate_deltas(stores[tid], stores["T0"], reg.n_boot)
        for rows in deltas.values():
            for r in rows:
                r["label"] = labels[r["position"]]
        # a pipeline that opens two chains has the same values in both and is counted once (section 16.7)
        material = [r["position"] for r in deltas["labels"] if r["class"] == "material" and r["position"] in distinct]
        outcome = []
        for r in units[tid]["P0"]["rows"]:
            ref = reference[(r["n"], r["threshold"])]
            ratio = r["compression_median"] / ref["compression_median"]
            difference = r["nontrivial_rate"] - ref["nontrivial_rate"]
            outcome.append({"n": r["n"], "threshold": r["threshold"], "delta": r["delta"],
                            "budget_class": r["budget_class"], "evidence": r["evidence"],
                            "median_compression_T0": ref["compression_median"],
                            "median_compression": r["compression_median"], "ratio_of_medians": ratio,
                            "nontrivial_rate_T0": ref["nontrivial_rate"], "nontrivial_rate": r["nontrivial_rate"],
                            "rate_difference": difference,
                            "material": bool(ratio >= RATIO or ratio <= 1.0 / RATIO
                                             or abs(difference) >= RATE_DIFFERENCE)})
        pairs[tid] = {
            "transformation": units[tid]["transformation"],
            "candidates": deltas,
            "class_counts_distinct": {c: {cls: sum(1 for r in rows if r["position"] in distinct and r["class"] == cls)
                                          for cls in ("invariant", "small", "material")}
                                      for c, rows in deltas.items()},
            "max_abs_delta": {c: float(max(abs(r["delta"]) for r in rows)) for c, rows in deltas.items()},
            "material_candidates": material, "material_candidate_labels": [labels[k] for k in material],
            "procedure_outcome": outcome, "procedure_material": any(o["material"] for o in outcome),
        }
        pairs[tid]["holds"] = bool(not material and not pairs[tid]["procedure_material"])
    result = {"header": X.header(EXPERIMENT, reg), "dataset": name,
              "set": "primary" if name in reg.x3_primary else "extension",
              "evidence": {"candidates": f"{X.MEASURED}, transformed vectors",
                           "procedure_outcome": "SIMULATED, pool-supported"},
              "units": {t: {"transformation": u["transformation"], "gates": u["gates"],
                            "code_commit": u["header"]["code_commit"]} for t, u in units.items()},
              "pairs": pairs}
    if "T2a" in pairs and "T2b" in pairs:  # secondary; not required for a verdict
        a, b = pairs["T2a"]["candidates"]["labels"], pairs["T2b"]["candidates"]["labels"]
        result["rotations_compared"] = {
            "max_abs_difference_in_R_L": float(max(abs(x["retention"] - y["retention"]) for x, y in zip(a, b, strict=True))),
            "candidates_with_the_same_class": sum(x["class"] == y["class"] for x, y in zip(a, b, strict=True)),
            "candidates": len(a)}
    lists = path("lists", f"{unit(name, 'T0')}_top10.npz")
    if lists.is_file():  # exploratory; no hypothesis depends on it
        z = np.load(lists, allow_pickle=False)
        result["overlap_exploratory"] = {"evidence": X.MEASURED, "rows": [
            {"position": k, "label": labels[k],
             **{f"overlap@{d}": overlap(z["reference_top10"], z["candidate_top10"][k], d)
                for d in (5, 10) if d <= z["reference_top10"].shape[1]}} for k in distinct]}
    return result


def summarise(datasets: dict, status: dict, reg: X.Registry) -> dict:
    """H5 over the 16 primary pairs. There is no 'undecided' outcome (section 12.6)."""
    failing = [k for k, v in status.items() if v == "does not hold"]
    incomplete = [k for k, v in status.items() if v == "incomplete"]
    sentences = []
    for key in failing:
        name, tid = key.split("__")
        pair = datasets[name]["pairs"][tid]
        subjects = [f"candidate `{label}`" for label in pair["material_candidate_labels"]]
        subjects += ["the outcome of P0"] if pair["procedure_material"] else []
        sentences += [(N11 if tid in ("T2a", "T2b") else N12).format(subject=s, dataset=name) for s in subjects]
    extension = {unit(n, t): ("holds" if p["holds"] else "does not hold")
                 for n in reg.x3_extension if n in datasets for t, p in datasets[n]["pairs"].items()}
    return {
        "hypotheses": {"H5": {
            "pairs": status, "failing_pairs": failing, "incomplete_pairs": incomplete,
            "verdict": "fails" if failing else ("not assessable" if incomplete else "supported"),
            "if_supported": "invariance under these four maps, on these datasets, within these tolerances. Two "
                            "rotations cannot establish invariance under every orthogonal map."}},
        "extension_set": {"note": "reported separately; never substituted for a primary dataset", "pairs": extension},
        "negative_result_sentences": sentences,
        "scope": "four registered maps; 96-d and 128-d development encoders; says nothing about model-blindness "
                 "(sections 12.8 and 15.6)",
    }


def report(summary: dict, datasets: dict) -> str:
    h5 = summary["hypotheses"]["H5"]
    out = [
        "# X3: coordinate and symmetry test", "",
        f"Specification: `{X.PREREG_PATH}` section 12, commit `{X.PREREG_COMMIT}`. "
        f"Code commit `{summary['header']['code_commit']}`. Written by `scripts/research/x3_symmetry.py`; "
        "every number is in the JSON files beside this report.", "",
        "**What is tested.** Four fixed maps with registered seeds: a signed coordinate permutation (T1), two "
        "rotations (T2a, T2b) and a reordering of the rows of each document (T3). Per-candidate retentions "
        "are MEASURED on transformed vectors; the outcome of P0 is SIMULATED at pool-supported budgets. "
        "Support would mean invariance under these four maps on these datasets within the registered "
        "tolerances. It would not show that results are independent of the coordinate basis in general.", "",
        "## Summary", "",
        f"- **H5: {h5['verdict']}** ({sum(v == 'holds' for v in h5['pairs'].values())} of {len(h5['pairs'])} "
        "primary pairs hold).",
    ]
    out += [f"- {s}" for s in summary["negative_result_sentences"]]
    out += [f"- Incomplete after three attempts: {k}." for k in h5["incomplete_pairs"]]
    out += ["", "## Pairs", "",
            "Classes of the change in R_L over the 37 distinct candidates: invariant (at most 0.005), material "
            "(at least 0.02 with an interval excluding 0), small (anything else).", "",
            X.md_table(["dataset", "set", "map", "invariant", "small", "material", "largest change in R_L",
                        "P0 outcome material", "holds"],
                       [[n, d["set"], t, *(p["class_counts_distinct"]["labels"][c] for c in ("invariant", "small", "material")),
                         p["max_abs_delta"]["labels"], p["procedure_material"], p["holds"]]
                        for n, d in datasets.items() for t, p in d["pairs"].items()]),
            "", "## Gates", "",
            X.md_table(["dataset", "map", "matrix or row-order sha256", "T0 equals the committed store",
                        "symmetry check: score difference / largest score", "identical per-query metric"],
                       [[n, t, (u["transformation"]["sha256"] or "–")[:16], u["gates"].get("T0", {}).get("equal"),
                         None if t == "T0" else
                         f"{u['gates']['symmetry']['max_abs_score_difference_over_max_abs_score']:.1e}",
                         u["gates"].get("symmetry", {}).get("share_of_queries_with_identical_metric")]
                        for n, d in datasets.items() for t, u in d["units"].items()]),
            "", "## Secondary", "",
            X.md_table(["dataset", "map", "R_A invariant", "R_A small", "R_A material", "largest change in R_A"],
                       [[n, t, *(p["class_counts_distinct"]["agreement"][c] for c in ("invariant", "small", "material")),
                         p["max_abs_delta"]["agreement"]] for n, d in datasets.items() for t, p in d["pairs"].items()]),
            "", "The two rotations compared, and overlap@5 and overlap@10 from the identity run (exploratory), are "
            "in each dataset's JSON file.", ""]
    return "\n".join(out)


def finish(reg: X.Registry, log: X.RunLog) -> dict:
    """Per-dataset analysis from the unit files; summary and report once every primary pair is settled."""
    out_dir = X.out_path(reg, EXPERIMENT)
    datasets = {}
    for name in (*reg.x3_primary, *reg.x3_extension):
        result = analyse_dataset(name, reg)
        if result is not None:
            datasets[name] = result
            X.write_json(out_dir / f"{PREFIX}_{name}.json", result)
    status, pending = {}, []
    for name in reg.x3_primary:
        for tid in list(TRANSFORMATIONS)[1:]:
            pair = datasets.get(name, {}).get("pairs", {}).get(tid)
            exhausted = any(log.valid(unit(name, t)) is None and len(log.attempts(unit(name, t))) >= MAX_ATTEMPTS
                            for t in ("T0", tid))
            if pair is not None:
                status[unit(name, tid)] = "holds" if pair["holds"] else "does not hold"
            elif exhausted:
                status[unit(name, tid)] = "incomplete"
            else:
                pending.append(unit(name, tid))
    if pending:
        print(f"X3: summary not written; {len(pending)} primary pairs are still to run: {', '.join(pending)}",
              flush=True)
        return {"finished": False, "pending": pending}
    summary = {"header": X.header(EXPERIMENT, reg), **summarise(datasets, status, reg)}
    X.write_json(out_dir / f"{PREFIX}_summary.json", summary)
    X.write_text(out_dir / f"{EXPERIMENT}_REPORT.md", report(summary, datasets))
    X.write_json(out_dir / f"{PREFIX}_provenance.json", {
        "experiment": EXPERIMENT, "environment": X.environment(), "git": X.git_state(reg.root),
        "run_log": "attempts.json",
        "outputs": {p.relative_to(out_dir).as_posix(): X.sha256_file(p) for p in sorted(out_dir.rglob("*"))
                    if p.is_file() and p.name != f"{PREFIX}_provenance.json"}})
    print(f"X3: summary, report and provenance written to {out_dir}", flush=True)
    return {"finished": True, "summary": summary}


def run(reg: X.Registry = X.REGISTERED, datasets=None, loader=load_dataset) -> dict:
    names = list(datasets or reg.x3_primary)
    validation = X.require_run_allowed(EXPERIMENT, reg, names)
    if reg.registered and os.environ.get("OPTIVISION_SCORE_DEVICE"):
        raise X.RunNotAllowed("X3: OPTIVISION_SCORE_DEVICE is set; the stores were built on the numpy path")
    log = X.RunLog(X.out_path(reg, EXPERIMENT, "attempts.json"))
    primary_done = all(log.valid(unit(n, t)) is not None or len(log.attempts(unit(n, t))) >= MAX_ATTEMPTS
                       for n in reg.x3_primary for t in TRANSFORMATIONS)
    if any(n in reg.x3_extension for n in names) and not primary_done:
        raise X.RunNotAllowed("X3: the extension set is run only after the whole primary set is complete")
    X.keep_awake(True)
    try:
        for name in names:
            run_dataset(name, reg, loader, log, validation)
    finally:
        X.keep_awake(False)
    return finish(reg, log)


# -------------------------------------------------------------------- dry run


def self_test(dim: int, seed: int = 12345) -> dict:
    """The four maps on small synthetic unit-norm vectors: determinism, norms, MaxSim, row annotations."""
    rng = np.random.default_rng(seed)

    def unit_rows(n):
        v = rng.standard_normal((n, dim))
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    offsets = np.concatenate([[0], np.cumsum(rng.integers(8, 20, size=6))])
    n = int(offsets[-1])
    corpus = MultiVectorCorpus(unit_rows(n), offsets, protected=rng.random(n) < 0.1,
                               positions=rng.random((n, 2)).astype(np.float32))
    queries = MultiVectorCorpus(unit_rows(40), np.arange(0, 41, 5))
    base = maxsim_matrix(queries, corpus)
    out = {}
    for tid in list(TRANSFORMATIONS)[1:]:
        c, q, record = transform(corpus, queries, tid)
        scores = maxsim_matrix(q, c)
        out[tid] = {
            "sha256": record["sha256"], "deterministic": transform(corpus, queries, tid)[2]["sha256"] == record["sha256"],
            "max_abs_score_difference_over_max_abs_score": float(np.abs(scores - base).max() / np.abs(base).max()),
            "max_norm_error": float(np.abs(np.linalg.norm(c.vectors.astype(np.float64), axis=1) - 1.0).max())}
        if tid == "T3":
            rows = row_order(offsets, TRANSFORMATIONS[tid])
            inside = all(((rows[lo:hi] >= lo) & (rows[lo:hi] < hi)).all()
                         for lo, hi in zip(offsets[:-1], offsets[1:], strict=True))
            out[tid]["rows_stay_in_their_document"] = bool(inside)
            out[tid]["annotations_follow_rows"] = bool(np.array_equal(c.protected, corpus.protected[rows])
                                                       and np.array_equal(c.positions, corpus.positions[rows])
                                                       and np.array_equal(c.vectors, corpus.vectors[rows]))
        else:
            out[tid]["orthogonality_error"] = record["orthogonality_error"]
        out[tid]["passed"] = bool(out[tid]["deterministic"] and out[tid]["max_norm_error"] <= 1e-5
                                  and out[tid]["max_abs_score_difference_over_max_abs_score"] <= SCORE_TOLERANCE
                                  and out[tid].get("orthogonality_error", 0.0) <= 1e-12
                                  and out[tid].get("rows_stay_in_their_document", True)
                                  and out[tid].get("annotations_follow_rows", True))
    return out


def dry_run(reg: X.Registry = X.REGISTERED) -> dict:
    rep = X.dry_run_common(EXPERIMENT, reg)
    rep["datasets"] = {"primary": list(reg.x3_primary), "extension": list(reg.x3_extension)}
    rep["transformations"] = {t: {"kind": KIND[t], "seed": s} for t, s in TRANSFORMATIONS.items()}
    rep["self_test_on_synthetic_vectors"] = {str(d): self_test(d) for d in sorted({s.dim for s in reg.stores.values()})}
    rep["transformations_ok"] = all(t["passed"] for d in rep["self_test_on_synthetic_vectors"].values() for t in d.values())
    vectors = {}
    if reg.registered:
        s7b = E.load_s7b()
        for name in (*reg.x3_primary, *reg.x3_extension):
            files = [pathlib.Path(p) for p in vector_files(name, s7b)]
            vectors[name] = {p.name: ({"bytes": p.stat().st_size, "sha256": X.sha256_file(p)} if p.is_file() else None)
                             for p in files}
        rep["data_directory"] = s7b.D
    rep["vector_files"] = vectors
    rep["vectors_ok"] = all(v is not None for files in vectors.values() for v in files.values())
    rep["environment"] = {"OPTIVISION_CACHE_BYTES": os.environ.get("OPTIVISION_CACHE_BYTES"),
                          "OPTIVISION_SCORE_DEVICE": os.environ.get("OPTIVISION_SCORE_DEVICE")}
    if not rep["vectors_ok"]:
        rep["run_blocked_by"].append("a vector file is missing under the data directory (OPTIVISION_DATA)")
    rep["inputs_ok"] = bool(rep["inputs_ok"] and rep["transformations_ok"] and rep["vectors_ok"])
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="X3: coordinate and symmetry test (preregistered).")
    ap.add_argument("--run", action="store_true", help="run the registered experiment (default: dry run)")
    ap.add_argument("--dataset", action="append", choices=[*X.X3_PRIMARY, *X.X3_EXTENSION],
                    help="with --run: only this dataset (repeatable)")
    ap.add_argument("--extension", action="store_true", help="with --run: the extension set D2, D3, D6")
    ap.add_argument("--json", action="store_true", help="print the full dry-run report as JSON")
    args = ap.parse_args(argv)
    if not args.run:
        rep = dry_run()
        X.print_dry_run(rep, as_json=args.json)
        if not args.json:
            for d, tests in rep["self_test_on_synthetic_vectors"].items():
                print(f"  maps on synthetic unit vectors, d = {d}: "
                      + ", ".join(f"{t} {'ok' if r['passed'] else 'FAILED'} ({r['sha256'][:12]})" for t, r in tests.items()))
            for name, files in rep["vector_files"].items():
                print(f"  vectors {name:18s} " + ", ".join(f"{f} {'ok' if v else 'MISSING'}" for f, v in files.items()))
            print(f"  inputs ok including vectors and maps: {rep['inputs_ok']}")
        return 0 if rep["inputs_ok"] else 1
    run(X.REGISTERED, args.dataset or (X.X3_EXTENSION if args.extension else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
