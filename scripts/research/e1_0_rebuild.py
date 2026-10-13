"""E1.0: rebuild per-query results for the 40 default candidates and check reproduction.

For one development dataset:
1. load it exactly as s7b does, score the default search space (s7b's own calls);
2. store per-query nDCG@5 (baseline and each candidate, labels and baseline@1
   references) plus candidate metadata in reports/research/E1/store/;
3. re-run s7b's evaluate() from the fresh scores (output outside git);
4. replay every s7b rule from the compact store alone;
5. compare 3 and 4, outcome by outcome, with the committed
   reports/universal/selection_rules/<dataset>.json. Exact equality is required:
   selected candidate, number judged, compression, calibration and held-out
   retention. Any mismatch is reported and the script exits non-zero.

    OPTIVISION_DATA=../optivision-rag-v2/data OPTIVISION_OUT=<scratch> PYTHONPATH=src \
        python scripts/research/e1_0_rebuild.py colpali_docvqa
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e1_common as E  # noqa: E402

from optivision.calibration import recommended_search_space, score_space  # noqa: E402
from optivision.evaluation import per_query_metrics, relevant_from_baseline, split_queries  # noqa: E402
from optivision.scoring import maxsim_matrix  # noqa: E402

FIELDS = ("selected", "judged", "compression", "calibration", "holdout", "holdout_labels")


def build(name: str, s7b) -> tuple:
    t0 = time.time()
    ds, metric = s7b.dataset(E.dataset_arg(name))
    corpus, queries = ds.corpus, ds.queries
    base = maxsim_matrix(queries, corpus)
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
        "dataset": ds.name, "s7b_argument": E.dataset_arg(name), "metric": metric,
        "n_docs": len(corpus), "n_queries": len(queries), "dim": corpus.dimension,
        "relevant_per_query": {str(k): n_rel.count(k) for k in sorted(set(n_rel))},
        "candidates": [{"position": i, "family": sc.family, "label": sc.label, "pipeline": sc.pipeline,
                        "bytes_per_doc": sc.bytes_per_doc,
                        "vectors_per_doc": float(sc.report["vectors_per_doc"]),
                        "compression_vs_float32": float(sc.report["compression_vs_float32"]),
                        "dim": int(sc.report["dim"]), "bits_per_dim": float(sc.report["bits_per_dim"])}
                       for i, sc in enumerate(scored)],
        "stored": {
            "base_labels": "per-query nDCG@5 of the float32 index, labels reference (n_queries,) float64",
            "base_b1": "the same, baseline@1 reference (float index's own top page as the relevant one)",
            "cand_labels": "per-query nDCG@5 of each candidate (K, n_queries) float64, labels reference",
            "cand_b1": "the same, baseline@1 reference",
        },
        "build_seconds": time.time() - t0,
    }
    E.STORE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(E.STORE / f"{name}.npz", **arrays)
    (E.STORE / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return ds, metric, base, rel_labels, rel_base, scored


def replay(name: str) -> dict:
    """Every s7b outcome, from the store alone."""
    d = E.load_store(name)
    st = {ref: E.Store(d, ref) for ref in ("labels", "baseline@1")}
    lab = st["labels"]
    n_q = lab.pq.shape[1]
    out = {}
    for ref, store in st.items():
        for rule in ["calibrate() default", *RULES]:
            for target in TARGETS:
                for seed in range(SPLITS):
                    cal, hold = split_queries(n_q, 0.5, seed)
                    if rule == "calibrate() default":
                        k, judged = E.replay_default(store, cal, target, seed=seed)
                        calib = None if k is None else E.retention(store.pq[k][cal], store.base[cal],
                                                                     n_boot=1000, seed=seed)[0]
                    else:
                        k = E.replay_select(store, cal, target, seed, n_boot=500, **RULES[rule])
                        judged = store.K
                        calib = None if k is None else E.retention(store.pq[k][cal], store.base[cal],
                                                                     n_boot=500, seed=seed)[0]
                    if k is None:
                        o = {"selected": None, "judged": judged}
                    else:
                        o = {"selected": store.labels[k], "judged": judged,
                             "compression": float(store.compression[k]), "calibration": calib,
                             "holdout": E.point_retention(store, k, hold),
                             "holdout_labels": E.point_retention(lab, k, hold)}
                    out[(ref, rule, target, seed)] = o
    return out


def outcomes(path) -> dict:
    d = json.loads(open(path, encoding="utf-8").read())
    return {(r["reference"], r["rule"], r["target"], o["seed"]): o for r in d["results"] for o in r["outcomes"]}, d


def compare(old: dict, new: dict) -> dict:
    mism, max_diff, n = [], 0.0, 0
    for key, o in old.items():
        p = new.get(key)
        n += 1
        if p is None:
            mism.append({"key": list(key), "problem": "missing in rebuilt"})
            continue
        for f in FIELDS:
            a, b = o.get(f), p.get(f)
            if isinstance(a, float) and isinstance(b, float):
                max_diff = max(max_diff, abs(a - b))
            if a != b:
                mism.append({"key": list(key), "field": f, "committed": a, "rebuilt": b,
                             "abs_diff": abs(a - b) if isinstance(a, float) and isinstance(b, float) else None})
    return {"outcomes_compared": n, "mismatches": mism, "max_abs_float_diff": max_diff,
            "selected_identical": all(m.get("field") != "selected" and "problem" not in m for m in mism)}


def main(name: str, replay_only: bool = False) -> int:
    """replay_only (E1.2 confirmation stores): skip re-running s7b's rules from fresh
    scores; the store replay is still compared with every committed outcome."""
    s7b = load = E.load_s7b()
    global RULES, TARGETS, SPLITS
    RULES, TARGETS, SPLITS = s7b.RULES, s7b.TARGETS, s7b.SPLITS
    t0 = time.time()
    ds, metric, base, rel_labels, rel_base, scored = build(name, load)
    t_build = time.time() - t0
    committed, cdoc = outcomes(E.COMMITTED / f"{ds.name}.json")
    if replay_only:
        rerun = committed
    else:
        s7b.evaluate(ds, metric, base, rel_labels, rel_base, scored, "e1_rerun", time.time())
        rerun, _ = outcomes(os.path.join(s7b.S, "e1_rerun", f"{ds.name}.json"))
    t_rerun = time.time() - t0 - t_build
    t1 = time.time()
    rep = replay(name)
    t_replay = time.time() - t1
    cand_ok = ([c["label"] for c in cdoc["candidates"]] == [sc.label for sc in scored]
               and [c["compression"] for c in cdoc["candidates"]]
               == [float(sc.report["compression_vs_float32"]) for sc in scored])
    report = {
        "experiment": "E1.0", "dataset": ds.name, "store": name, "label": "MEASURED",
        "environment": E.environment(),
        "candidates_identical_to_committed": cand_ok,
        "n_candidates": len(scored),
        "committed_vs_rerun": "skipped (replay_only)" if replay_only else compare(committed, rerun),
        "committed_vs_store_replay": compare(committed, rep),
        "store_files": {f: hashlib.sha256((E.STORE / f).read_bytes()).hexdigest()
                        for f in (f"{name}.npz", f"{name}.json")},
        "seconds": {"build": t_build, "rerun_rules": t_rerun, "replay_from_store": t_replay},
    }
    (E.E1 / f"e1_0_{name}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    ok = (cand_ok and (replay_only or not report["committed_vs_rerun"]["mismatches"])
          and not report["committed_vs_store_replay"]["mismatches"])
    print(json.dumps({k: (v if not isinstance(v, dict) or "mismatches" not in v else
                          {**v, "mismatches": v["mismatches"][:5], "n_mismatches": len(v["mismatches"])})
                      for k, v in report.items()}, indent=1, default=float))
    print("E1.0", name, "REPRODUCED" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], replay_only="--replay-only" in sys.argv))
