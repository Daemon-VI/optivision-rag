"""Shared pieces of experiment E1 (docs/RESEARCH_INVESTIGATION.md, E1).

The per-query store: for one dataset, per-query nDCG@5 of the float32 baseline and
of each of the 40 default candidates, under both references (labels, baseline@1),
plus each candidate's metadata. Every selection rule in s7b depends on the
candidates only through these arrays and the metadata, so the store is enough to
replay them exactly (E1.0 checks that).

``replay_default`` is ``calibrate()``'s own selection path (0.999 one-sided
bootstrap bound, n_boot=1000, families walked in order and each stopped at its
first infeasible step, smallest stored size wins), written against the store and
calling the library's own ``retention_lower_bound`` / ``retention``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib

import numpy as np

from optivision.calibration import _per_candidate_confidence
from optivision.evaluation import retention, retention_lower_bound

ROOT = pathlib.Path(__file__).resolve().parents[2]
E1 = ROOT / "reports" / "research" / "E1"
STORE = E1 / "store"
DATASETS = {
    # store name: s7b dataset argument
    "colpali_docvqa": "E2-colpali-docvqa",
    "colpali_infovqa": "E2-colpali-infovqa",
    "colqwen2_docvqa": "vec:colqwen2-v1.0-merged_docvqa_test_subsampled",
    "colqwen2_infovqa": "vec:colqwen2-v1.0-merged_infovqa_test_subsampled",
}
#: held back for confirmation only (not loaded by any development script)
CONFIRMATION = {
    "colqwen25_docvqa": "vec:colqwen2.5-v0.2_docvqa_test_subsampled",
    "colqwen25_infovqa": "vec:colqwen2.5-v0.2_infovqa_test_subsampled",
    "scifact": "text-scifact",
}
COMMITTED = ROOT / "reports" / "universal" / "selection_rules"


def load_s7b():
    spec = importlib.util.spec_from_file_location(
        "s7b", ROOT / "scripts" / "universal_study" / "s7b_selection_rules.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_store(name: str) -> dict:
    z = np.load(STORE / f"{name}.npz", allow_pickle=False)
    meta = json.loads((STORE / f"{name}.json").read_text(encoding="utf-8"))
    return {"meta": meta, **{k: z[k] for k in z.files}}


class Store:
    """Arrays for one reference: base (n_q,), pq (K, n_q); metadata per candidate."""

    def __init__(self, d: dict, reference: str = "labels"):
        key = "labels" if reference == "labels" else "b1"
        self.base = d[f"base_{key}"]
        self.pq = d[f"cand_{key}"]
        m = d["meta"]
        self.labels = [c["label"] for c in m["candidates"]]
        self.families = [c["family"] for c in m["candidates"]]
        self.bytes = np.array([c["bytes_per_doc"] for c in m["candidates"]])
        self.vectors = np.array([c["vectors_per_doc"] for c in m["candidates"]])
        self.compression = np.array([c["compression_vs_float32"] for c in m["candidates"]])
        self.metric = m["metric"]

    @property
    def K(self) -> int:
        return self.pq.shape[0]


def rank_key(store: Store, k: int, position: int) -> tuple:
    # calibration._rank_key: (bytes_per_doc, vectors_per_doc, position among the judged)
    return (store.bytes[k], store.vectors[k], position)


def replay_default(store: Store, cal: np.ndarray, target: float, seed: int = 0,
                   n_boot: int = 1000, confidence: float = 0.999, bounds: dict | None = None):
    """calibrate()'s selection on the calibration indices ``cal``.

    Returns (k or None, n_judged). ``bounds`` caches the per-candidate lower bound,
    which does not depend on the target.
    """
    base_cal = store.base[cal]
    judged, stopped = [], set()
    for k in range(store.K):
        fam = store.families[k]
        if fam in stopped:
            continue
        if bounds is not None and k in bounds:
            lo = bounds[k]
        else:
            lo = retention_lower_bound(store.pq[k][cal], base_cal, confidence, n_boot=n_boot, seed=seed)
            if bounds is not None:
                bounds[k] = lo
        ok = bool(lo >= target)
        judged.append((k, ok))
        if not ok:
            stopped.add(fam)
    feas = [(k, pos) for pos, (k, ok) in enumerate(judged) if ok]
    if not feas:
        return None, len(judged)
    k, _ = min(feas, key=lambda t: rank_key(store, t[0], t[1]))
    return k, len(judged)


def replay_select(store: Store, cal: np.ndarray, target: float, seed: int, n_boot: int = 500,
                  safety: str = "point", confidence: float = 0.975, multiplicity: str = "none",
                  margin: float = 0.0):
    """calibration.select() on the store (all candidates judged)."""
    base_cal = store.base[cal]
    conf = _per_candidate_confidence(confidence, multiplicity, store.K)
    feas = []
    for k in range(store.K):
        if safety == "point":
            score = retention(store.pq[k][cal], base_cal, n_boot=n_boot, seed=seed)[0]
        else:
            score = retention_lower_bound(store.pq[k][cal], base_cal, conf, n_boot=n_boot, seed=seed)
        if score >= target + margin:
            feas.append(k)
    if not feas:
        return None
    return min(feas, key=lambda k: rank_key(store, k, k))


def point_retention(store: Store, k: int, idx: np.ndarray) -> float:
    return retention(store.pq[k][idx], store.base[idx], n_boot=0)[0]


def environment() -> dict:
    import platform
    import subprocess
    import sys

    import scipy

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=False).stdout
    return {"commit": commit, "dirty_tree": bool(dirty.strip()), "python": sys.version.split()[0],
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count()}
