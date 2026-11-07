"""Score every preregistered prediction from the ingested units and propose the ledger change.

    PYTHONPATH=src python scripts/easy_validation/run_all.py --data data/easy --out reports/easy/results
    PYTHONPATH=src python scripts/easy_validation/run_all.py --synthetic --out <tmp>   # dry run, fake backend

Reads ``data/easy/<batch>/`` (``ingest.py``), scores V1a-V6, S, arm S (``predictions.py``) and Part 3
DT + practice-question allowance (``part3.py``), applies the decision rules (``decide.py``) and writes,
under ``--out``:

    predictions.json      every result with its unit rows
    decisions.json        the decision rules that fired and the proposed settings
    ledger_proposal.json  the change to presets/easy_ledger.json, for a person to review and apply
    SUMMARY.md            one line per prediction

The ledger itself is never written here. ``--synthetic`` runs the whole programme on a tiny
generated tree with ``fakes.FakeBackend``; its numbers mean nothing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

# Gentle Mode: limit background math threads by default unless overridden (e.g. on Kaggle)
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "4")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import numpy.lib.format as npfmt  # noqa: E402
from optivision.compose import Pipeline  # noqa: E402
from optivision.core.identity import canonical_json  # noqa: E402
from optivision.evaluation import retention as _eval_retention  # noqa: E402
from optivision.representation import MultiVectorCorpus  # noqa: E402
from optivision.stages.base import DimensionReducer, TokenReducer  # noqa: E402
from optivision.stages.merge import _budget, _pool, _unit, _ward_input  # noqa: E402
from easy_validation import api, data, decide, part3, predictions, scorer, units  # noqa: E402
from easy_validation.data import batch_roots, heldout_models  # noqa: E402
from easy_validation.evaluate import Context  # noqa: E402


class BoundedDict(dict):
    """Keep at most `max_items` entries in memory simultaneously to prevent RAM exhaustion."""

    def __init__(self, max_items: int = 2) -> None:
        super().__init__()
        self.max_items = max_items

    def __setitem__(self, key: object, value: object) -> None:
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        while len(self) > self.max_items:
            oldest = next(iter(self.keys()))
            if oldest == key:
                break
            del self[oldest]
        gc.collect()


def _read_vec_header(shard_path: Path) -> tuple[int, np.dtype]:
    with zipfile.ZipFile(shard_path, "r") as zf:
        name = "vectors.npy" if "vectors.npy" in zf.namelist() else next(
            n for n in zf.namelist() if n.startswith("vectors")
        )
        with zf.open(name) as f:
            ver = npfmt.read_magic(f)
            if ver == (1, 0):
                shape, _, dtype = npfmt.read_array_header_1_0(f)
            else:
                shape, _, dtype = npfmt.read_array_header_2_0(f)
            return int(shape[1]), np.dtype(dtype)


class LazyShardCorpus(MultiVectorCorpus):
    """MultiVectorCorpus backed by .npz shards that materializes .vectors on demand and bounds global RAM."""

    _LIVE: list[LazyShardCorpus] = []
    _MAX_LIVE_BYTES: int = 8 * 1024**3  # cap live raw shard vectors at 8 GiB across the process

    def __init__(self, shards: list[Path], offsets: np.ndarray, ids: list[str], dim: int, dtype: np.dtype) -> None:
        self._shards = list(shards)
        self._vectors: np.ndarray | None = None
        self._dim = int(dim)
        self._dtype = np.dtype(dtype)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        n_docs = int(self.offsets.size) - 1
        self.ids = [str(i) for i in ids]
        self.metadata = [{} for _ in range(n_docs)]
        self.protected = None
        self.positions = None
        self.weights = None
        self.importance = None
        self.images = None
        self.attrs = {}

    @property
    def num_vectors(self) -> int:
        return int(self.offsets[-1])

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self.num_vectors * self._dim * self._dtype.itemsize

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            need = self.nbytes
            while LazyShardCorpus._LIVE and (
                sum(c.nbytes for c in LazyShardCorpus._LIVE) + need > LazyShardCorpus._MAX_LIVE_BYTES
            ):
                old = LazyShardCorpus._LIVE.pop(0)
                old._vectors = None
            gc.collect()
            arr = np.empty((self.num_vectors, self._dim), dtype=self._dtype)
            cursor = 0
            for s in self._shards:
                with np.load(s, allow_pickle=False) as z:
                    v = z["vectors"]
                    n = len(v)
                    arr[cursor:cursor + n] = v
                    cursor += n
            self._vectors = arr
            LazyShardCorpus._LIVE.append(self)
        else:
            if self in LazyShardCorpus._LIVE:
                LazyShardCorpus._LIVE.remove(self)
            LazyShardCorpus._LIVE.append(self)
        return self._vectors


def _streaming_read_shards(folder: Path) -> MultiVectorCorpus | None:
    """Read shard metadata immediately and stream vectors lazily on demand."""
    shards = sorted(folder.glob("shard_*.npz"))
    if not shards:
        return None
    dim, dtype = _read_vec_header(shards[0])
    doc_counts: list[np.ndarray] = []
    all_ids: list[str] = []
    for s in shards:
        with np.load(s, allow_pickle=False) as z:
            doc_counts.append(np.diff(z["offsets"]))
            all_ids.extend(str(i) for i in z["ids"])
    counts = np.concatenate(doc_counts) if doc_counts else np.zeros(0, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return LazyShardCorpus(shards, offsets, all_ids, dim, dtype)


# Apply lazy streaming shard reader to avoid 17+ GB memory spikes on large units
data.read_shards = _streaming_read_shards

# Active cache directory (defaults to data/easy/_cache, updated per run)
_SCORES_CACHE = HERE.parents[1] / "data" / "easy" / "_cache"
_SCORES_CACHE.mkdir(parents=True, exist_ok=True)


def _norm_subset(n_pages: int, subset: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if subset is not None and len(subset) == n_pages and subset == tuple(range(n_pages)):
        return None
    return subset


# Bound per-scorer dictionaries so compressed stores and merged corpora never accumulate
_orig_scorer_init = scorer.UnitScorer.__init__


def _bounded_scorer_init(self: scorer.UnitScorer, unit: data.Unit) -> None:
    _orig_scorer_init(self, unit)
    self._compressed = BoundedDict(max_items=2)
    self._transformed = BoundedDict(max_items=2)


scorer.UnitScorer.__init__ = _bounded_scorer_init  # type: ignore[method-assign]


def _joint_hm_transform(docs: MultiVectorCorpus) -> dict[float, MultiVectorCorpus]:
    """Compute HierarchicalMerge(0.33) and HierarchicalMerge(0.25) from one Ward linkage per page in parallel."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n_docs = len(docs)
    dtype = docs.dtype
    dim = docs.dimension

    def _process_page(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = docs.span(i)
        v = np.asarray(docs.vectors[lo:hi], dtype=np.float32)
        n = v.shape[0]
        w = docs.weights[lo:hi] if docs.weights is not None else np.ones(n, np.float32)
        if n == 0:
            empty_v = np.zeros((0, dim), dtype=dtype)
            empty_w = np.zeros(0, dtype=np.float32)
            return empty_v, empty_w, empty_v, empty_w
        k33 = _budget(n, 0.33, None, 1)
        k25 = _budget(n, 0.25, None, 1)
        z = None
        if n > 1 and (k33 < n or k25 < n):
            z = linkage(_ward_input(_unit(v).astype(np.float64)), method="ward")
        outs = []
        for k in (k33, k25):
            if n <= 1 or k >= n:
                out_vec = v.copy()
                labels = np.arange(n, dtype=np.int64)
            else:
                raw = fcluster(z, t=k, criterion="maxclust")
                _, labels = np.unique(raw, return_inverse=True)
                labels = labels.astype(np.int64)
                out_vec = _pool(v, labels, int(labels.max()) + 1, w)
            out_vec = np.asarray(out_vec, dtype=np.float32).reshape(-1, dim)
            k_out = out_vec.shape[0]
            w_out = np.bincount(labels, weights=w, minlength=k_out).astype(np.float32)
            outs.append((out_vec.astype(dtype, copy=False), w_out))
        return outs[0][0], outs[0][1], outs[1][0], outs[1][1]

    workers = min(8, max(1, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_process_page, range(n_docs)))

    out_corpora: dict[float, MultiVectorCorpus] = {}
    for idx, ratio in ((0, 0.33), (2, 0.25)):
        parts = [r[idx] for r in results]
        w_parts = [r[idx + 1] for r in results]
        counts = [p.shape[0] for p in parts]
        vectors = np.concatenate(parts, axis=0) if parts else np.zeros((0, dim), dtype)
        offsets = np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))])
        weights = np.concatenate(w_parts) if w_parts else np.zeros(0, np.float32)
        protected = np.zeros(vectors.shape[0], dtype=bool)
        out_corpora[ratio] = MultiVectorCorpus(
            vectors,
            offsets,
            ids=docs.ids,
            metadata=docs.metadata,
            protected=protected,
            weights=weights,
            attrs=docs.attrs,
        )
    return out_corpora


def _tx_cache_file(tag: str, n_pages: int, variant: str, vs_json: str) -> Path:
    h = hashlib.sha256(f"{tag}|{n_pages}|{variant}|{vs_json}".encode("utf-8")).hexdigest()[:24]
    return _SCORES_CACHE / f"{tag}__tx_{variant}_{h}.npz"


def _save_tx_corpus(path: Path, corpus: MultiVectorCorpus) -> None:
    try:
        arrays: dict[str, Any] = {"vectors": corpus.vectors, "offsets": corpus.offsets}
        if corpus.weights is not None:
            arrays["weights"] = corpus.weights
        np.savez(path, **arrays)
    except Exception:
        pass


def _load_tx_corpus(path: Path, base_docs: MultiVectorCorpus) -> MultiVectorCorpus | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            vectors = z["vectors"]
            offsets = z["offsets"]
            weights = z["weights"] if "weights" in z.files else None
        return MultiVectorCorpus(
            vectors,
            offsets,
            ids=base_docs.ids,
            metadata=base_docs.metadata,
            protected=np.zeros(vectors.shape[0], dtype=bool),
            weights=weights,
            attrs=base_docs.attrs,
        )
    except Exception:
        return None


def _ensure_transformed(self: scorer.UnitScorer, vector_stages: Pipeline, variant: str) -> MultiVectorCorpus:
    vs_json = canonical_json(vector_stages.to_dict())
    full_key = (vs_json, variant, None)
    if full_key in self._transformed:
        return self._transformed[full_key]
    base_docs = self.docs(variant)
    tx_path = _tx_cache_file(self.unit.tag, self.unit.n_pages, variant, vs_json)
    loaded = _load_tx_corpus(tx_path, base_docs)
    if loaded is not None:
        self._transformed[full_key] = loaded
        return loaded

    stages = vector_stages.stages
    if (
        len(stages) == 1
        and stages[0].name == "hierarchical_merge"
        and getattr(stages[0], "max_distance", None) is None
        and getattr(stages[0], "min_vectors", 1) == 1
        and getattr(stages[0], "ratio", None) in (0.33, 0.25)
        and base_docs.protected is None
        and base_docs.positions is None
        and base_docs.importance is None
    ):
        joint = _joint_hm_transform(base_docs)
        for r_val, corp in joint.items():
            r_vs = canonical_json(
                {"stages": [{"max_distance": None, "min_vectors": 1, "ratio": r_val, "stage": "hierarchical_merge"}]}
            )
            _save_tx_corpus(_tx_cache_file(self.unit.tag, self.unit.n_pages, variant, r_vs), corp)
            self._transformed[(r_vs, variant, None)] = corp
        return self._transformed[full_key]

    corp = vector_stages.transform(base_docs)
    _save_tx_corpus(tx_path, corp)
    self._transformed[full_key] = corp
    return corp


_orig_compressed = scorer.UnitScorer.compressed


def _safe_compressed(
    self: scorer.UnitScorer,
    spec: dict,
    *,
    fit: str = "sample",
    variant: str = "f16",
    subset: tuple[int, ...] | None = None,
):
    subset = _norm_subset(self.unit.n_pages, subset)
    stages = (spec or {}).get("stages", [])
    if (not stages or stages == [{"stage": "float16"}]) and variant == "f16":
        key = (canonical_json(spec), fit if scorer.stateful(spec) else "full", variant, subset)
        if key not in self._compressed:
            p = Pipeline.from_dict(spec or {"stages": [{"stage": "float16"}]})
            self._compressed[key] = (self.docs(variant, subset), p, 1.0)
        return self._compressed[key]

    eff_fit = fit if scorer.stateful(spec) else "full"
    key = (canonical_json(spec), eff_fit, variant, subset)
    if key not in self._compressed:
        p = Pipeline.from_dict(spec)
        vector_stages = Pipeline(p.vector_stages)
        if vector_stages.fitted and vector_stages.stages:
            _ensure_transformed(self, vector_stages, variant)
    return _orig_compressed(self, spec, fit=fit, variant=variant, subset=subset)


scorer.UnitScorer.compressed = _safe_compressed  # type: ignore[method-assign]


def _fast_ratio_f16(
    self: scorer.UnitScorer,
    spec: dict,
    *,
    fit: str = "sample",
    variant: str = "f16",
    subset: tuple[int, ...] | None = None,
) -> float:
    subset = _norm_subset(self.unit.n_pages, subset)
    stages = (spec or {}).get("stages", [])
    if (not stages or stages == [{"stage": "float16"}]) and variant == "f16":
        return 1.0
    eff_fit = fit if scorer.stateful(spec) else "full"
    key = (canonical_json(spec), eff_fit, variant, subset)
    if key in self._compressed:
        return round(self._compressed[key][2], 3)

    base_docs = self.docs(variant, None)
    orig_dim = base_docs.dimension
    orig_vecs = base_docs.num_vectors if subset is None else int(base_docs.counts[list(subset)].sum())
    f16_bytes = orig_vecs * orig_dim * 2

    p = Pipeline.from_dict(spec)
    dim = orig_dim
    for s in p.vector_stages:
        if isinstance(s, DimensionReducer):
            dim = s.output_dim(dim)
    q = p.quantizer
    overhead = (
        (4 * dim if getattr(q, "scale", None) == "per_dimension" else 0)
        + (4 * dim if getattr(q, "center", None) == "mean" else 0)
        + (4 * dim + 4 if q.name == "lloyd2" else 0)
    )

    if not any(isinstance(s, TokenReducer) for s in p.vector_stages):
        store_nbytes = orig_vecs * q.code_bytes(dim) + overhead
        return round(f16_bytes / max(1, int(store_nbytes)), 3)

    vector_stages = Pipeline(p.vector_stages)
    if vector_stages.fitted:
        tx = _ensure_transformed(self, vector_stages, variant)
        tx_vecs = tx.num_vectors if subset is None else int(tx.counts[list(subset)].sum())
        store_nbytes = tx_vecs * q.code_bytes(dim) + overhead
        return round(f16_bytes / max(1, int(store_nbytes)), 3)

    return round(self.compressed(spec, fit=fit, variant=variant, subset=subset)[2], 3)


def _fast_retention(
    self: scorer.UnitScorer,
    spec: dict | None,
    *,
    fit: str = "sample",
    variant: str = "f16",
    subset: tuple[int, ...] | None = None,
    qids: Any = None,
    base_variant: str | None = None,
) -> dict:
    subset = _norm_subset(self.unit.n_pages, subset)
    base = self.per_query(
        self.scores(None, qids=qids, variant=base_variant or variant, subset=subset),
        qids=qids,
        subset=subset,
    )
    if spec is None:
        c = base
    else:
        c = self.per_query(
            self.scores(spec, qids=qids, fit=fit, variant=variant, subset=subset),
            qids=qids,
            subset=subset,
        )
    point, lo, hi = _eval_retention(c, base, n_boot=units.N_BOOT, seed=units.BOOT_SEED)
    out = {
        "point": point,
        "low": lo,
        "high": hi,
        "n_queries": int(np.isfinite(base).sum()),
        "baseline": float(np.nanmean(base)),
    }
    if spec is not None:
        out["ratio_f16"] = _fast_ratio_f16(self, spec, fit=fit, variant=variant, subset=subset)
    return out


scorer.UnitScorer.retention = _fast_retention  # type: ignore[method-assign]


_orig_scores = scorer.UnitScorer.scores
_PREWARMING = False


def _probe_hard_ids(self: scorer.UnitScorer, source: str) -> list[str] | None:
    if source not in self.unit.probes or not self.unit.text_checkable(source):
        return None
    cache_attr = "_cached_probe_hard_ids"
    mapping = getattr(self, cache_attr, None)
    if mapping is None:
        mapping = {}
        setattr(self, cache_attr, mapping)
    if source not in mapping:
        by_id = {p["id"]: p for p in self.unit.probes[source]}
        ids = list(self.query_set(source).ids)
        base = self.scores(None, source)
        ref_scores = self.ranking(base, ids, k=scorer.TOP, with_scores=True)
        b = api.backend()
        hs = b.hard_set([b.probe(by_id[i]) for i in ids], ref_scores, k_keep=10)
        hard = [i for i in api.hard_ids(hs) if i in by_id]
        mapping[source] = hard if len(hard) >= units.HARD_MIN else None
    return mapping[source]


def _score_cache_file(self: scorer.UnitScorer, spec: dict | None, which: str, kw: dict[str, Any]) -> Path:
    key = (
        self.unit.tag,
        self.unit.n_pages,
        None if spec is None else canonical_json(spec),
        which,
        tuple(sorted((k, str(v)) for k, v in kw.items())),
    )
    h = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:24]
    return _SCORES_CACHE / f"{self.unit.tag}__{h}.npy"


def _cached_scores(self: scorer.UnitScorer, spec: dict | None, which: str = "queries", **kw: Any) -> np.ndarray:
    global _PREWARMING
    if "subset" in kw:
        kw = {**kw, "subset": _norm_subset(self.unit.n_pages, kw["subset"])}
        if kw["subset"] is None:
            del kw["subset"]
    cache_file = _score_cache_file(self, spec, which, kw)
    if cache_file.exists():
        try:
            return np.load(cache_file)
        except Exception:
            pass
    s = _orig_scores(self, spec, which=which, **kw)
    try:
        np.save(cache_file, s)
    except Exception:
        pass

    # While docs or compressed store are already in RAM for one probe source, score remaining probe sources
    if not _PREWARMING and which in self.unit.probe_queries:
        _PREWARMING = True
        try:
            if spec is None and not kw:
                for other_src in self.unit.probe_queries:
                    if other_src != which and not _score_cache_file(self, None, other_src, {}).exists():
                        _cached_scores(self, None, which=other_src)
            elif (
                spec is not None
                and kw.get("fit", "sample") == "sample"
                and kw.get("variant", "f16") == "f16"
                and kw.get("subset") is None
                and kw.get("qids") is not None
            ):
                for other_src in self.unit.probe_queries:
                    if other_src != which:
                        hard = _probe_hard_ids(self, other_src)
                        if hard and not _score_cache_file(self, spec, other_src, {"qids": hard}).exists():
                            _cached_scores(self, spec, which=other_src, qids=hard)
        finally:
            _PREWARMING = False

    return s


scorer.UnitScorer.scores = _cached_scores  # type: ignore[method-assign]


def _fmt(v: object) -> str:
    return {True: "pass", False: "fail", None: "not scored"}.get(v, str(v))  # type: ignore[arg-type]


def summary_md(results: dict, decisions: dict, backend: str, roots: list[Path]) -> str:
    lines = [
        "# Easy-mode validation: results",
        "",
        f"Written by `scripts/easy_validation/run_all.py` on {time.strftime('%Y-%m-%d')} with backend "
        f"`{backend}` from {len(roots)} batch folder(s). Thresholds and decision rules: "
        "reports/easy/PREREG_EASY.md (part 1), PREREG_EASY_PART2.md and PREREG_EASY_PART3.md. "
        "Rows marked in-sample are fitted on the units they are reported on.",
        "",
        "| id | prediction | pass if | result | detail |",
        "|---|---|---|---|---|",
    ]
    for k, r in results.items():
        tag = " (in-sample)" if r.get("in_sample") else ""
        lines.append(f"| {k} | {r['prediction']} | {r['pass_if']} | {_fmt(r['passed'])}{tag} | {r['reason']} |")
    lines += ["", "## Decision rules that fired (proposed, not applied)", ""]
    if decisions["actions"]:
        for a in decisions["actions"]:
            lines.append(f"- {a['rule']}: {a['outcome']} `{json.dumps(a['set'])}` {a['note']}".rstrip())
    else:
        lines.append("- none")
    lines += ["", f"Status line this would give: \"{decisions['status']}\"", ""]
    return "\n".join(lines)


def score_all_with_progress(ctx: Context, checkpoint_path: Path | None = None) -> dict[str, dict]:
    """Score every prediction with step-by-step progress logging, checkpointing, and garbage collection."""
    predictions._RULE_CACHE.clear()
    out: dict[str, dict] = {}
    int8_v2: list[float] = []

    if checkpoint_path and checkpoint_path.exists():
        try:
            ckpt_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            out = ckpt_data.get("out", {})
            int8_v2 = ckpt_data.get("int8_v2", [])
            print(f"Loaded {len(out)} prediction(s) from checkpoint: {checkpoint_path}", flush=True)
        except Exception:
            out, int8_v2 = {}, []

    def _save() -> None:
        if checkpoint_path:
            try:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps({"out": out, "int8_v2": int8_v2}, indent=1, default=float) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass

    if "V1a" not in out:
        print("[1/14] Scoring V1a (int8 retention across all page units)...", flush=True)
        t0 = time.time()
        out["V1a"] = predictions.v1a(ctx)
        print(f"       -> {_fmt(out['V1a']['passed'])} ({time.time() - t0:.1f}s): {out['V1a']['reason']}", flush=True)
        _save()
        gc.collect()
    else:
        print("[1/14] Scoring V1a: already computed (loaded from checkpoint)", flush=True)

    if not all(k in out for k in ("V1b", "V1c", "V1d")):
        print("[2/14] Scoring V1b/V1c/V1d (rule + practice check on V1 corpora)...", flush=True)
        t0 = time.time()
        for r in predictions.v1bcd(ctx):
            out[r["id"]] = r
            print(f"       -> {r['id']} {_fmt(r['passed'])}: {r['reason']}", flush=True)
        print(f"       (completed in {time.time() - t0:.1f}s)", flush=True)
        _save()
        gc.collect()
    else:
        print("[2/14] Scoring V1b/V1c/V1d: already computed (loaded from checkpoint)", flush=True)

    if "V1-unchecked" not in out:
        print("[3/14] Scoring V1-unchecked (TechSlides & text-hidden)...", flush=True)
        t0 = time.time()
        out["V1-unchecked"] = predictions.v1_unchecked(ctx)
        print(
            f"       -> {_fmt(out['V1-unchecked']['passed'])} ({time.time() - t0:.1f}s): {out['V1-unchecked']['reason']}",
            flush=True,
        )
        _save()
        gc.collect()
    else:
        print("[3/14] Scoring V1-unchecked: already computed (loaded from checkpoint)", flush=True)

    if "V2" not in out or not int8_v2:
        print("[4/14] Scoring V2 (size rule on v3_industrial 5,244 pages)...", flush=True)
        t0 = time.time()
        out["V2"], int8_v2 = predictions.v2(ctx)
        print(f"       -> {_fmt(out['V2']['passed'])} ({time.time() - t0:.1f}s): {out['V2']['reason']}", flush=True)
        _save()
        gc.collect()
    else:
        print("[4/14] Scoring V2: already computed (loaded from checkpoint)", flush=True)

    if "V1-arms" not in out:
        print("[5/14] Scoring V1-arms (arms B, C, D across targets)...", flush=True)
        t0 = time.time()
        out["V1-arms"] = predictions.v1_arms(ctx, int8_v2)
        print(
            f"       -> {_fmt(out['V1-arms']['passed'])} ({time.time() - t0:.1f}s): {out['V1-arms']['reason']}",
            flush=True,
        )
        _save()
        gc.collect()
    else:
        print("[5/14] Scoring V1-arms: already computed (loaded from checkpoint)", flush=True)

    if "V1t" not in out:
        print("[6/14] Scoring V1t (text models on FiQA & ArguAna)...", flush=True)
        t0 = time.time()
        out["V1t"] = predictions.v1t(ctx)
        print(f"       -> {_fmt(out['V1t']['passed'])} ({time.time() - t0:.1f}s): {out['V1t']['reason']}", flush=True)
        _save()
        gc.collect()
    else:
        print("[6/14] Scoring V1t: already computed (loaded from checkpoint)", flush=True)

    if "V1u" not in out:
        print("[7/14] Scoring V1u (held-out encoders on hr & cs)...", flush=True)
        t0 = time.time()
        out["V1u"] = predictions.v1u(ctx)
        print(f"       -> {_fmt(out['V1u']['passed'])} ({time.time() - t0:.1f}s): {out['V1u']['reason']}", flush=True)
        _save()
        gc.collect()
    else:
        print("[7/14] Scoring V1u: already computed (loaded from checkpoint)", flush=True)

    if "S" not in out:
        print("[8/14] Scoring S (size study across 250, 500, 1000, full)...", flush=True)
        t0 = time.time()
        out["S"] = predictions.size_study(ctx)
        print(f"       -> {_fmt(out['S']['passed'])} ({time.time() - t0:.1f}s): {out['S']['reason']}", flush=True)
        _save()
        gc.collect()
    else:
        print("[8/14] Scoring S: already computed (loaded from checkpoint)", flush=True)

    if "arm S" not in out:
        print("[9/14] Scoring arm S (refit size coefficients checked on V2)...", flush=True)
        t0 = time.time()
        out["arm S"] = predictions.arm_s(ctx, out["S"]["coefficients"], int8_v2)
        print(
            f"       -> {_fmt(out['arm S']['passed'])} ({time.time() - t0:.1f}s): {out['arm S']['reason']}",
            flush=True,
        )
        _save()
        gc.collect()
    else:
        print("[9/14] Scoring arm S: already computed (loaded from checkpoint)", flush=True)

    if "V3" not in out:
        print("[10/14] Scoring V3 (growth and refit delta)...", flush=True)
        t0 = time.time()
        out["V3"] = predictions.v3(ctx)
        print(f"       -> {_fmt(out['V3']['passed'])} ({time.time() - t0:.1f}s): {out['V3']['reason']}", flush=True)
        _save()
        gc.collect()
    else:
        print("[10/14] Scoring V3: already computed (loaded from checkpoint)", flush=True)

    if "V4" not in out:
        print("[11/14] Scoring V4 & V5 (probes optimism & planted faults)...", flush=True)
        t0 = time.time()
        for source in units.PROBE_SOURCES:
            r = predictions.v4(ctx, source)
            if source == "text_layer" or r["units"]:
                out[r["id"]] = r
                print(f"       -> {r['id']} {_fmt(r['passed'])}: {r['reason']}", flush=True)
            for r in predictions.v5(ctx, source):
                if source == "text_layer" or r["units"]:
                    out[r["id"]] = r
                    print(f"       -> {r['id']} {_fmt(r['passed'])}: {r['reason']}", flush=True)
        print(f"       (completed in {time.time() - t0:.1f}s)", flush=True)
        _save()
        gc.collect()
    else:
        print("[11/14] Scoring V4 & V5: already computed (loaded from checkpoint)", flush=True)

    if "V6" not in out:
        print("[12/14] Scoring V6 (wiring check)...", flush=True)
        t0 = time.time()
        out["V6"] = predictions.v6(ctx)
        print(f"       -> {_fmt(out['V6']['passed'])} ({time.time() - t0:.1f}s): {out['V6']['reason']}", flush=True)
        _save()
        gc.collect()
    else:
        print("[12/14] Scoring V6: already computed (loaded from checkpoint)", flush=True)

    if "DT" not in out:
        print("[13/14] Scoring Part 3 DT (document-type presets campaign)...", flush=True)
        t0 = time.time()
        out["DT"] = part3.dt_campaign(ctx)
        print(f"       -> {_fmt(out['DT']['passed'])} ({time.time() - t0:.1f}s): {out['DT']['reason']}", flush=True)
        _save()
        gc.collect()
    else:
        print("[13/14] Scoring Part 3 DT: already computed (loaded from checkpoint)", flush=True)

    if "probe_allowance" not in out:
        print("[14/14] Scoring Part 3 probe_allowance (1-step upward allowance across probe sources)...", flush=True)
        t0 = time.time()
        for source in units.PROBE_SOURCES:
            r = part3.probe_allowance(ctx, source)
            if source == "text_layer" or r["units"]:
                out[r["id"]] = r
                print(f"       -> {r['id']} {_fmt(r['passed'])}: {r['reason']}", flush=True)
        print(f"       (completed in {time.time() - t0:.1f}s)", flush=True)
        _save()
        gc.collect()
    else:
        print("[14/14] Scoring Part 3 probe_allowance: already computed (loaded from checkpoint)", flush=True)

    return out


def run(roots: list[Path], out: Path, backend_name: str = "optivision.easy") -> dict:
    global _SCORES_CACHE
    _SCORES_CACHE = (roots[0].parent / "_cache") if roots else (out / "_cache")
    _SCORES_CACHE.mkdir(parents=True, exist_ok=True)

    ctx = Context(roots, heldout=heldout_models(roots))
    ctx._units = BoundedDict(max_items=2)
    ctx._scorers = BoundedDict(max_items=2)
    ctx._practice = BoundedDict(max_items=4)
    checkpoint_file = out / "_step_checkpoint.json"
    print(f"Scoring all predictions with Gentle Mode memory bounding across {len(roots)} batch folders...", flush=True)
    results = score_all_with_progress(ctx, checkpoint_path=checkpoint_file)

    def rerun(arm: str, target: float) -> bool | None:
        rows, missing = predictions.rule_rows(ctx, predictions.v1_units(), (target,), arm=arm)
        rows = predictions.scored(rows)
        if missing or not rows:
            return None
        crit = predictions.crit_v1b if target == 0.97 else predictions.crit_v1c
        return crit(rows)[0]

    decisions = decide.decide(results, rerun)
    decisions = part3.apply_part3_decisions(results, decisions)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(results, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "decisions.json").write_text(
        json.dumps({k: v for k, v in decisions.items() if k != "ledger_change"}, indent=1, default=float) + "\n",
        encoding="utf-8",
    )
    (out / "ledger_proposal.json").write_text(
        json.dumps(decisions["ledger_change"], indent=1, default=float) + "\n",
        encoding="utf-8",
    )
    (out / "SUMMARY.md").write_text(summary_md(results, decisions, backend_name, roots), encoding="utf-8", newline="\n")
    return {"results": results, "decisions": decisions, "missing": sorted(ctx.missing)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=HERE.parents[1] / "data" / "easy")
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "reports" / "easy" / "results")
    ap.add_argument("--synthetic", action="store_true", help="dry run on generated units with the fake backend")
    a = ap.parse_args(argv)
    if a.synthetic:
        from easy_validation import fakes, synthetic

        with tempfile.TemporaryDirectory() as tmp, api.use(fakes.FakeBackend()):
            root = synthetic.write_tree(Path(tmp))
            info = run(batch_roots(root), a.out, "fakes (synthetic)")
    else:
        info = run(batch_roots(a.data), a.out)
    for k, r in info["results"].items():
        print(f"{k:24s} {_fmt(r['passed']):11s} {r['reason'][:100]}")
    if info["missing"]:
        print(f"missing units: {len(info['missing'])} (first: {info['missing'][:3]})")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
