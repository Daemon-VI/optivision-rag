"""Shared pieces of the preregistered next-phase experiments X1, X2 and X3.

Specification: ``reports/research/NEXT_PHASE_PREREGISTRATION.md`` as committed in
``73085ab637679f30fffece6632eb7b9e92adff79`` and recorded in the registry by
``d408b96db87698f76f5f320b8ee8591a3d75ecd9``. Section numbers below refer to it.

Nothing here changes Method C or the library. The statistics are imported from
``e1_stats.py`` and ``e1_1_study.py`` (frozen at E1.2); this module only wires them to
the registered budgets, seeds and stores, validates the registered inputs, and writes
outputs that say what kind of evidence each number is.

A result-producing run goes through :func:`require_run_allowed`. The ``--dry-run`` path
of each script reads file hashes, array headers and candidate metadata only; it never
reads a per-query value.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e1_1_study as E1S  # noqa: E402  (frozen: a_bounds, clopper_pearson, Problem, A_CONF)
import e1_common as E  # noqa: E402  (frozen: Store, dataset names)
import e1_stats as S  # noqa: E402  (frozen: Method C primitives)

from optivision.evaluation import split_queries  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "reports" / "research"

# --------------------------------------------------------------------- registered

PREREG_PATH = "reports/research/NEXT_PHASE_PREREGISTRATION.md"
PREREG_COMMIT = "73085ab637679f30fffece6632eb7b9e92adff79"
PREREG_BLOB = "fa9deb73fdf380f06141d89cfaf145444f883506"
REGISTRY_COMMIT = "d408b96db87698f76f5f320b8ee8591a3d75ecd9"

#: Versions of the preregistration this harness implements. The registered text, plus
#: any later version that only appends dated amendments to its Appendix C: the commit
#: that makes an amendment adds the new blob here.
PREREG_BLOBS_IMPLEMENTED = (PREREG_BLOB,)

#: sha256 of the 40 candidate specifications (section 22.1): the JSON array of
#: {"position", "family", "pipeline"} objects, sorted keys, separators "," and ":".
S0_SPEC_SHA256 = "af29b01146a90a048ffea9cfbd66f0fedf988134829cd4dc56861f1207893520"
N_CANDIDATES = 40
N_CHAINS = 7
N_DISTINCT_CANDIDATES = 37
FLOAT16_PIPELINE = {"stages": [{"stage": "float16"}]}
INT8_PIPELINE = {"stages": [{"scale": "per_vector", "stage": "int8"}]}

#: git blob hashes of the code S0 and Method C depend on (section 22.1).
CODE_BLOBS = {
    "src/optivision/calibration.py": "58f1727ceeb200f9d84dcd28452821b19d324904",
    "src/optivision/compose.py": "85b20d449706b5a5980fa90b3e9be2bd572b9ba8",
    "src/optivision/stages/base.py": "1516638bc68f72303a1cee7f234b48be4b2b3346",
    "src/optivision/stages/merge.py": "3f17b093c4adcd18fcf29f50a8e2a6d4f55c5a0f",
    "src/optivision/stages/quantize.py": "68d8816099a0fe4f32e87fdf04226172380c3d4f",
    "src/optivision/stages/project.py": "5d32185f5cfab195de50cd2b5180e401d9d192f6",
    "src/optivision/compression/__init__.py": "49b025086b2fe87edb39ce1235c733159b401bc1",
    "src/optivision/compression/binary.py": "cc1d86825d2cf1b77de3c283840a6250da7a8b63",
    "src/optivision/scoring.py": "3e1677327445077a455afe580320d583d547aee8",
    "src/optivision/evaluation.py": "192d1b2717c34cbafbd157ac373e67b695719bf6",
    "src/optivision/representation.py": "20adcd1dde815e1cf1f42760196ab4b45c20ffe8",
    "scripts/research/e1_stats.py": "13494307a4a42cd2c1da63f34aa0952a92b33130",
    "scripts/research/e1_1_study.py": "5896e763b49d4a0f3f0cc988dc2b5ec5e6b2b9aa",
    "scripts/research/e1_common.py": "3bef181683307eb98a9c5e2d61ab02d87801bbfe",
}


@dataclass(frozen=True)
class StoreSpec:
    """One registered per-query store (sections 9.1 and 22.1)."""

    name: str
    encoder: str
    pool: str
    n_queries: int
    n_docs: int
    dim: int
    metric: str
    npz_sha256: str
    json_sha256: str  # of the sidecar as checked out with CRLF line endings (X_HARNESS.md, L2)

    @property
    def cutoff(self) -> int:
        return int(self.metric.split("@")[1])

    @property
    def vidore(self) -> bool:
        return self.pool in ("DocVQA", "InfoVQA")


_STORES = [
    StoreSpec("colpali_docvqa", "ColPali-v1.3", "DocVQA", 451, 500, 128, "ndcg@5",
              "bb301073e96b170fd8a35db6dbfa937c29cc6211659a88fd9a9b115fb0b0310b",
              "d419cdb895316af796a743c5af6f2c8dc784ea4a7888abad02c6ddbc66215283"),
    StoreSpec("colpali_infovqa", "ColPali-v1.3", "InfoVQA", 494, 500, 128, "ndcg@5",
              "cd88c233670801101728e21e30d439f1d8644703683ceff3ee9ee7e85295487f",
              "62cfd92be80687e0eb5aa82e41b4de501ae8f1313b49375584d7f433b9c02406"),
    StoreSpec("colqwen2_docvqa", "ColQwen2-v1.0", "DocVQA", 451, 500, 128, "ndcg@5",
              "511fb87d33a3485f3259d0d239658d30a9655513b18b03412731ed80dd83cb6f",
              "a68ff3a539a4454eb61ae7607a13a1910c0a93a3c86713a91dbe81dd612d6961"),
    StoreSpec("colqwen2_infovqa", "ColQwen2-v1.0", "InfoVQA", 494, 500, 128, "ndcg@5",
              "3faadf535897b5bb78b0e4adb08d7d60916912522fbd3da1c17d634dac7806f8",
              "b11bbb679f4eba4f63552b097ad3daef8f421c36071accfe63b7dfb5101147d4"),
    StoreSpec("colqwen25_docvqa", "ColQwen2.5-v0.2", "DocVQA", 451, 500, 128, "ndcg@5",
              "df50ae1b9adaab754962de2f1d2154e691b7a8919413f946958b67b0449e42ca",
              "00875fd852234007b27eef7d57b44c1921dcfb942f21674c2badffe916b29107"),
    StoreSpec("colqwen25_infovqa", "ColQwen2.5-v0.2", "InfoVQA", 494, 500, 128, "ndcg@5",
              "1aed74af53fbb24ac10eb123f463c5de4a2f77e9ece28b04ecfe99caa1a4dc50",
              "f31b69e0193efad6e186369cdce17cf39c658519b4896a1ec500014c04d4bd75"),
    StoreSpec("scifact", "answerai-colbert-small-v1", "SciFact", 300, 5183, 96, "ndcg@10",
              "e4ac7f15e8ccf652c0e928ed504f04bc65a9f94b7e11e2b68a3ea1638df90b98",
              "d75d093a4409cf1611b833848bcabb72a06e1ccc1f27ca24aba4e5a7112775e3"),
]
REGISTERED_STORES = {s.name: s for s in _STORES}

#: per-seed selections stored by E1.1 (development) and E1.2 (confirmation); gate G4.
E1_SELECTIONS = {
    **{n: f"reports/research/E1/real/{n}_selections.npz" for n in E.DATASETS},
    **{n: f"reports/research/E1/confirm/{n}_selections.npz" for n in E.CONFIRMATION},
}
E1_METHODS = tuple(E1S.METHODS)  # A, A-bonf, then B, C, C-split, D at each delta

# Budgets (sections 6 and 11.1)
COMMON_BUDGETS = (50, 100, 225, 450, 900, 1800, 3600)
POOL_CAP = 450
MAX_BUDGET = max(COMMON_BUDGETS)
E1_BUDGETS = (50, 100, 225)
AB_EXTRA_BUDGET = 900  # with the pool-capped budget: where A and B are also run (section 11.3)
AB_EXTRA_SEEDS = 500

# Thresholds, error levels, seeds (sections 10.4, 11.2, 11.3)
THRESHOLDS = {"labels": (0.99, 0.97, 0.95), "agreement": (0.99, 0.97, 0.95, 0.90)}
PRIMARY_THRESHOLDS = (0.99, 0.97, 0.95)
DELTAS = (0.05, 0.01)
PRIMARY_DELTA = 0.05
N_SEEDS = 2000
REPLICATION_SEEDS = (2000, 4000)

# Decision constants (sections 6, 7, 11.5, 12, 16.3)
RATE_THRESHOLD = 0.90
SUBSTANTIAL_COMPRESSION = 10.0
CP1_TOLERANCE = 0.01
CP2_LEVEL = 0.05
ORACLE_FACTOR = 2.0
H3_THRESHOLD = 0.97
H3_SECONDARY_THRESHOLD = 0.99
H4C_THRESHOLD = 0.95
X3_BUDGET = 225
X3_BOOTSTRAP = 2000
X3_PRIMARY = ("colpali_docvqa", "colqwen2_infovqa", "colqwen25_docvqa", "scifact")  # D1, D4, D5, D7 (section 12.4)
X3_EXTENSION = ("colpali_infovqa", "colqwen2_docvqa", "colqwen25_infovqa")  # D2, D3, D6

CONSTRUCTS = ("labels", "agreement")
CONSTRUCT_REFERENCE = {"labels": "labels", "agreement": "baseline@1"}  # e1_common.Store keys
CONSTRUCT_NOTE = {
    "labels": "labelled retention: nDCG@k against the dataset's labels",
    "agreement": "label-free agreement with the uncompressed reference: nDCG@k against the "
                 "uncompressed index's own top document. It is not a measure of retrieval quality",
}
MEASURED = "MEASURED"
SIM_POOL = "SIMULATED, pool-supported"
SIM_EXTRA = "SIMULATED, extrapolated"

#: memory target for one block of resampled arrays (bytes). Blocks hold whole seeds and a
#: multiple of 8 of them, so that every element takes the same vectorised arithmetic path
#: as in E1's single block.
CHUNK_BYTES = 96 * 1024 * 1024


@dataclass(frozen=True)
class Registry:
    """What a run may read, where it may write, and the size of its grid.

    ``REGISTERED`` is the only registered configuration. Tests build small synthetic
    registries in temporary directories. They go through the same code, and their
    outputs say ``registered_configuration: false``.
    """

    stores: dict
    store_dir: pathlib.Path
    out_roots: dict
    e1_selections: dict = field(default_factory=dict)
    q3_features: pathlib.Path | None = None
    s0_spec_sha256: str = S0_SPEC_SHA256
    code_blobs: dict = field(default_factory=dict)
    root: pathlib.Path = ROOT
    registered: bool = False
    common_budgets: tuple = COMMON_BUDGETS
    pool_cap: int = POOL_CAP
    e1_budgets: tuple = E1_BUDGETS
    ab_extra_budget: int = AB_EXTRA_BUDGET
    ab_extra_seeds: int = AB_EXTRA_SEEDS
    n_seeds: int = N_SEEDS
    replication_seeds: tuple = REPLICATION_SEEDS
    draw_size: int = MAX_BUDGET
    x3_budget: int = X3_BUDGET
    n_boot: int = X3_BOOTSTRAP
    x3_primary: tuple = X3_PRIMARY
    x3_extension: tuple = X3_EXTENSION

    def budgets(self, n_pool: int) -> tuple:
        """Common grid, plus the pool size where it is below the cap (section 11.1)."""
        grid = set(self.common_budgets)
        if n_pool < self.pool_cap:
            grid.add(int(n_pool))
        return tuple(sorted(grid))

    def pool_capped_budget(self, n_pool: int) -> int:
        return self.pool_cap if n_pool >= self.pool_cap else int(n_pool)

    def ab_seed_count(self, n: int, n_pool: int) -> int:
        """How many seeds rule A and its companion B get at budget n (section 11.3)."""
        if n in self.e1_budgets:
            return self.n_seeds
        if n in (self.pool_capped_budget(n_pool), self.ab_extra_budget):
            return min(self.ab_extra_seeds, self.n_seeds)
        return 0


REGISTERED = Registry(
    stores=REGISTERED_STORES,
    store_dir=E.STORE,
    out_roots={x: RESEARCH / x for x in ("X1", "X2", "X3")},
    e1_selections={k: ROOT / v for k, v in E1_SELECTIONS.items()},
    q3_features=RESEARCH / "Q3" / "features",
    code_blobs=CODE_BLOBS,
    registered=True,
)

# ------------------------------------------------------------ open points (not decided)

#: Points the preregistration leaves technically open. ``implemented`` is the reading
#: this harness carries. None of them is decided here: a registered run of an
#: experiment refuses to start until every point that affects it is in
#: ACCEPTED_OPEN_POINTS, which only the project owner fills (with a dated amendment in
#: Appendix C of the preregistration, committed before the run it affects).
OPEN_POINTS = {
    "OP1": {"affects": ("X3",),
            "text": "T1, the signed coordinate permutation with seed 0 (section 12.3): the text does "
                    "not say how the permutation and the signs are drawn from the seed.",
            "implemented": "rng = numpy.random.default_rng(0); perm = rng.permutation(d); "
                           "signs = rng.integers(0, 2, size=d) * 2 - 1; coordinate j of the result is "
                           "signs[j] * v[perm[j]]."},
    "OP2": {"affects": ("X3",),
            "text": "T3, the reordering of the rows of each document with seed 3 (section 12.3): the "
                    "text does not say how the per-document orders are drawn from the seed.",
            "implemented": "one generator numpy.random.default_rng(3); documents visited in corpus "
                           "order; rng.permutation(n_i) for a document of n_i rows; protected, "
                           "positions, weights and importance move with their rows."},
    "OP3": {"affects": ("X3",),
            "text": "T2a and T2b (section 12.3): the matrix U is defined, but not how it is applied to "
                    "a stored row vector. Section 12.2 writes the map as (Uq).(Ud).",
            "implemented": "every vector v becomes U v, as section 12.2 writes it: rows X become "
                           "X @ U.T, computed in float64 in blocks of 65,536 rows and stored as float32."},
    "OP4": {"affects": ("X3",),
            "text": "T0 gate, fallback (section 12.5 step 2): 'all 40 pool retentions' and 'per-query "
                    "entries' do not name the construct or the arrays.",
            "implemented": "the stricter reading: retentions under both constructs within 0.001, and "
                           "at least 99.5% identical entries in each of the four arrays."},
    "OP5": {"affects": ("X1", "X2"),
            "text": "Borderline rule (section 11.5): 'a rate that decides a verdict or an n*' does not "
                    "list the cells.",
            "implemented": "the cells that decide H3 (primary and secondary) and H4c, and, for each n*, "
                           "the cells met when the budgets are scanned from the largest downwards up to "
                           "and including the first one that fails. Cells below that one cannot change "
                           "n* and are not re-run."},
    "OP6": {"affects": ("X1",),
            "text": "A6, exploratory (section 10.5): 'agreement loss by quartile of the baseline score "
                    "margin' names neither the margin (Q3/features holds a relevance margin and a "
                    "top-2 margin), nor the candidates, nor how loss is aggregated.",
            "implemented": "both margins, each reported separately; quartiles of the margin over the "
                           "store's pool (numpy.quantile at 0.25, 0.5, 0.75; a query on an edge goes "
                           "to the upper quartile); per distinct candidate, loss = 1 - mean of a in "
                           "the quartile."},
    "OP7": {"affects": ("X1",),
            "text": "N10 (section 19): 'the pool-supported cells where the labels arm deploys "
                    "non-trivially' does not say when a cell counts as deploying non-trivially.",
            "implemented": "cells are (store, threshold in 0.99/0.97/0.95, delta, pool-supported "
                           "budget), both error levels counted together; the labels arm deploys "
                           "non-trivially when its non-trivial deployment rate is at least 0.90."},
}
ACCEPTED_OPEN_POINTS: frozenset = frozenset()


class RegisteredInputMismatch(SystemExit):
    """An input does not match its registered identity. The run stops."""


class RunNotAllowed(SystemExit):
    """A result-producing run was requested while a precondition is not met."""


class GateFailure(SystemExit):
    """A registered gate or check failed. The experiment stops (section 21, rule 3)."""


# ------------------------------------------------------------------ hashing


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def crlf_form(data: bytes) -> bytes:
    """The same text with CRLF line endings, whatever it had."""
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def lfs_pointer_oid(path) -> str | None:
    """sha256 named by a Git LFS pointer file, or None if ``path`` holds real content."""
    with open(path, "rb") as f:
        head = f.read(512)
    if not head.startswith(b"version https://git-lfs.github.com/spec/"):
        return None
    for line in head.decode("utf-8", "replace").splitlines():
        if line.startswith("oid sha256:"):
            return line.split(":", 1)[1].strip()
    return None


def git_blob_sha1(path) -> str:
    """git's blob hash of a text file as committed (line endings normalised to LF)."""
    data = pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def candidate_spec(candidates) -> list:
    return [{"position": c["position"], "family": c["family"], "pipeline": c["pipeline"]} for c in candidates]


def spec_sha256(spec) -> str:
    return sha256_bytes(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def generated_spec(dim: int) -> list:
    """The candidate list the frozen code generates (needs SciPy for the Ward chains)."""
    from optivision.calibration import recommended_search_space

    out = []
    for family, steps in recommended_search_space(dim).items():
        for pipe in steps:
            out.append({"position": len(out), "family": family, "pipeline": pipe.to_dict()})
    return out


def npz_headers(path) -> dict:
    """{array name: (shape, dtype)} from the .npy headers only. No array data is read."""
    from numpy.lib import format as npformat

    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version = npformat.read_magic(f)
                reader = npformat.read_array_header_1_0 if version == (1, 0) else npformat.read_array_header_2_0
                shape, _, dtype = reader(f)
            out[name.removesuffix(".npy")] = (tuple(int(x) for x in shape), str(dtype))
    return out


# ---------------------------------------------------------------- validation


def check_store_files(spec: StoreSpec, reg: Registry = REGISTERED) -> dict:
    """Identity of one store's two files. Reads hashes, headers and candidate metadata only.

    ``ok_for_run``: the bytes on disk are the registered bytes (gate G1 as registered).
    ``identity_ok``: the files are the registered content up to what git changes on
    checkout (an LFS pointer in place of the .npz, LF instead of CRLF in the sidecar).
    """
    npz = pathlib.Path(reg.store_dir) / f"{spec.name}.npz"
    side = pathlib.Path(reg.store_dir) / f"{spec.name}.json"
    out = {"store": spec.name, "encoder": spec.encoder, "pool": spec.pool, "problems": []}
    if not npz.is_file() or not side.is_file():
        out["problems"].append("missing file")
        return {**out, "ok_for_run": False, "identity_ok": False}

    oid = lfs_pointer_oid(npz)
    npz_raw = None if oid is not None else sha256_file(npz)
    out["npz"] = {"registered_sha256": spec.npz_sha256, "sha256": npz_raw, "lfs_pointer_oid": oid}
    npz_run = npz_raw == spec.npz_sha256
    npz_id = npz_run or oid == spec.npz_sha256
    if oid is not None:
        out["problems"].append("the .npz is a Git LFS pointer; fetch the object before a run")
    elif not npz_run:
        out["problems"].append("the .npz does not hash to the registered value")

    raw = side.read_bytes()
    side_raw, side_crlf = sha256_bytes(raw), sha256_bytes(crlf_form(raw))
    out["json"] = {"registered_sha256": spec.json_sha256, "sha256": side_raw, "sha256_crlf_form": side_crlf,
                   "git_blob": git_blob_sha1(side)}
    side_run = side_raw == spec.json_sha256
    side_id = side_run or side_crlf == spec.json_sha256
    if not side_id:
        out["problems"].append("the sidecar does not hash to the registered value in any line-ending form")
    elif not side_run:
        out["problems"].append("the sidecar matches the registered hash only in its CRLF form")

    meta = json.loads(raw.decode("utf-8"))
    cands = meta.get("candidates", [])
    families = list(dict.fromkeys(c["family"] for c in cands))
    spec_hash = spec_sha256(candidate_spec(cands)) if cands else None
    out["meta"] = {"dataset": meta.get("dataset"), "n_queries": meta.get("n_queries"), "n_docs": meta.get("n_docs"),
                   "dim": meta.get("dim"), "metric": meta.get("metric"), "n_candidates": len(cands),
                   "n_chains": len(families), "n_distinct_candidates": len(distinct_candidates(meta)),
                   "candidate_spec_sha256": spec_hash}
    meta_ok = (meta.get("n_queries") == spec.n_queries and meta.get("n_docs") == spec.n_docs
               and meta.get("dim") == spec.dim and meta.get("metric") == spec.metric
               and len(cands) == N_CANDIDATES and len(families) == N_CHAINS
               and spec_hash == reg.s0_spec_sha256)
    if not meta_ok:
        out["problems"].append("sidecar metadata or candidate specification differs from the registration")

    headers_ok = oid is not None
    if oid is None:
        heads = npz_headers(npz)
        want = {"base_labels": (spec.n_queries,), "base_b1": (spec.n_queries,),
                "cand_labels": (N_CANDIDATES, spec.n_queries), "cand_b1": (N_CANDIDATES, spec.n_queries)}
        out["arrays"] = {k: {"shape": list(v[0]), "dtype": v[1]} for k, v in heads.items()}
        headers_ok = all(k in heads and heads[k][0] == shape and heads[k][1] == "float64" for k, shape in want.items())
        if not headers_ok:
            out["problems"].append("array names, shapes or dtypes differ from the registration")

    out["ok_for_run"] = bool(npz_run and side_run and meta_ok and headers_ok)
    out["identity_ok"] = bool(npz_id and side_id and meta_ok and headers_ok)
    return out


def check_selection_file(name: str, reg: Registry = REGISTERED) -> dict:
    """The E1 per-seed selections used by gate G4: present, real content, expected keys. Headers only."""
    path = reg.e1_selections.get(name)
    if path is None or not pathlib.Path(path).is_file():
        return {"store": name, "ok": False, "problem": "missing file"}
    oid = lfs_pointer_oid(path)
    if oid is not None:
        return {"store": name, "ok": False, "lfs_pointer_oid": oid,
                "problem": "Git LFS pointer; fetch the object before a run"}
    heads = npz_headers(path)
    want = [sel_key(n, t, m) for n in reg.e1_budgets for t in PRIMARY_THRESHOLDS for m in E1_METHODS]
    missing = [k for k in want if k not in heads]
    shapes_ok = all(heads[k] == ((reg.n_seeds,), "int16") for k in want if k in heads)
    return {"store": name, "sha256": sha256_file(path), "keys": len(heads), "missing_keys": missing,
            "ok": bool(not missing and shapes_ok)}


def check_code(reg: Registry = REGISTERED) -> dict:
    """Frozen code files against their registered git blob hashes."""
    files, ok = {}, True
    for rel, blob in reg.code_blobs.items():
        path = pathlib.Path(reg.root) / rel
        got = git_blob_sha1(path) if path.is_file() else None
        files[rel] = {"registered_blob": blob, "blob": got, "match": got == blob}
        ok &= got == blob
    return {"files": files, "ok": bool(ok)}


def method_c_config() -> dict:
    """Method C as frozen in E1.2 (sections 14.2 and 22.1), read from the imported code."""
    return {"source": "scripts/research/e1_stats.py and e1_1_study.py, imported unchanged",
            "hypothesis": "H0_k: E[c_k - T b] <= 0", "statistic": "Y = (c - T b + T) / (1 + T); m0 = T / (1 + T)",
            "p_value": "exact betting p-value, aGRAPA bets", "bet_cap": S.BET_CAP,
            "multiplicity": "fixed-sequence testing inside each chain, in the chain's own order",
            "level": "delta / number of chains", "chains": N_CHAINS,
            "choice": "smallest stored size among the rejected; ties by fewer vectors, then earlier position",
            "deltas": list(DELTAS), "queries_used": "all n; no hold-out split"}


def check_method_c() -> dict:
    out = method_c_config()
    out["ok"] = bool(S.BET_CAP == 0.9 and tuple(E1S.DELTAS) == DELTAS and tuple(E1S.TARGETS) == PRIMARY_THRESHOLDS
                     and tuple(E1S.NS) == E1_BUDGETS and E1S.SEEDS == N_SEEDS and E1S.A_CONF == 0.999)
    return out


def check_generated_space(reg: Registry = REGISTERED, dims=(128, 96)) -> dict:
    """The candidate list the code generates must hash to the registered value (section 14.1)."""
    try:
        import scipy  # noqa: F401
    except ImportError:
        return {"ok": False, "problem": "SciPy is not installed: the generated space would lack the Ward chains"}
    hashes = {str(d): spec_sha256(generated_spec(d)) for d in dims}
    return {"registered": reg.s0_spec_sha256, "generated": hashes,
            "ok": all(h == reg.s0_spec_sha256 for h in hashes.values())}


def git_state(root=ROOT) -> dict:
    def git(*args):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return r.returncode, r.stdout.strip()

    code, head = git("rev-parse", "HEAD")
    if code != 0:
        return {"available": False}
    _, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    tracked_changes = git("diff", "--quiet", "HEAD")[0] != 0
    _, code_status = git("status", "--porcelain", "--untracked-files=all", "--", "scripts", "src", "tests")
    has_prereg = git("cat-file", "-e", PREREG_COMMIT + "^{commit}")[0] == 0
    _, prereg_blob = git("rev-parse", f"HEAD:{PREREG_PATH}")
    _, main = git("rev-parse", "--verify", "--quiet", "main")
    return {"available": True, "commit": head, "branch": branch, "tracked_changes": tracked_changes,
            "uncommitted_code": [ln for ln in code_status.splitlines() if ln.strip()],
            "prereg_commit_present": has_prereg, "prereg_blob_at_head": prereg_blob,
            "prereg_blob_registered": PREREG_BLOB, "main": main or None}


def validate_inputs(reg: Registry = REGISTERED, stores=None) -> dict:
    """Everything that can be checked without reading a per-query value."""
    names = list(reg.stores) if stores is None else list(stores)
    checked = {name: check_store_files(reg.stores[name], reg) for name in names}
    out = {"stores": checked, "code": check_code(reg), "method_c": check_method_c(),
           "generated_candidate_space": check_generated_space(reg, sorted({s.dim for s in reg.stores.values()}))}
    out["identity_ok"] = bool(all(s["identity_ok"] for s in checked.values()) and out["code"]["ok"]
                              and out["method_c"]["ok"] and out["generated_candidate_space"]["ok"])
    out["ok_for_run"] = bool(out["identity_ok"] and all(s["ok_for_run"] for s in checked.values()))
    return out


def open_points_for(experiment: str) -> dict:
    return {k: v for k, v in OPEN_POINTS.items() if experiment in v["affects"] and k not in ACCEPTED_OPEN_POINTS}


def require_run_allowed(experiment: str, reg: Registry = REGISTERED, stores=None) -> dict:
    """Refuse a result-producing run unless every registered precondition holds.

    Returns the validation report that goes into the run's provenance.
    """
    if reg.registered and reg is not REGISTERED:
        raise RunNotAllowed("only x_common.REGISTERED is the registered configuration")
    if reg.registered:
        blocking = open_points_for(experiment)
        if blocking:
            raise RunNotAllowed(f"{experiment}: open points not accepted by the project owner: "
                                f"{', '.join(sorted(blocking))}. See reports/research/X_HARNESS.md.")
    report = validate_inputs(reg, stores)
    if not report["ok_for_run"]:
        problems = [f"{n}: {p}" for n, s in report["stores"].items() for p in s["problems"]]
        problems += [f"code: {rel}" for rel, f in report["code"]["files"].items() if not f["match"]]
        if not report["method_c"]["ok"]:
            problems.append("Method C constants differ from the E1.2 freeze")
        if not report["generated_candidate_space"]["ok"]:
            problems.append("the generated candidate space does not hash to the registered specification")
        raise RegisteredInputMismatch(f"{experiment}: registered inputs do not match. Nothing was run.\n  "
                                      + "\n  ".join(problems))
    if experiment in ("X1", "X2"):  # gate G4 reads the selections stored by E1.1 and E1.2
        bad = [n for n in report["stores"] if not check_selection_file(n, reg)["ok"]]
        if bad:
            raise RegisteredInputMismatch(f"{experiment}: the E1 selection files of {', '.join(bad)} are missing, "
                                          "are Git LFS pointers, or lack the E1 keys. Nothing was run.")
    if reg.registered:
        g = git_state(reg.root)
        if not g.get("available"):
            raise RunNotAllowed(f"{experiment}: git is not available, so the code commit cannot be recorded")
        if g["tracked_changes"] or g["uncommitted_code"]:
            raise RunNotAllowed(f"{experiment}: uncommitted changes. Each script is committed before its "
                                "first run (section 22.2)")
        if not g["prereg_commit_present"] or g["prereg_blob_at_head"] not in PREREG_BLOBS_IMPLEMENTED:
            raise RunNotAllowed(f"{experiment}: the preregistration at HEAD is not a version this harness implements")
        report["git"] = g
    return report


# ------------------------------------------------------------------- budgets


def registered_budgets(n_pool: int) -> tuple:
    return REGISTERED.budgets(n_pool)


def pool_capped_budget(n_pool: int) -> int:
    return REGISTERED.pool_capped_budget(n_pool)


def budget_class(n: int, n_pool: int) -> str:
    return "pool-supported" if n <= n_pool else "extrapolated"


def evidence_mark(n: int, n_pool: int) -> str:
    return SIM_POOL if n <= n_pool else SIM_EXTRA


def budget_table(n_pool: int, reg: Registry = REGISTERED) -> list:
    cap = reg.pool_capped_budget(n_pool)
    return [{"n": n, "budget_class": budget_class(n, n_pool), "evidence": evidence_mark(n, n_pool),
             "pool_capped": n == cap} for n in reg.budgets(n_pool)]


# ------------------------------------------------------------------ sampling


def draw_block(n_pool: int, seeds, size: int) -> np.ndarray:
    """(len(seeds), size) query indices, i.i.d. with replacement, one generator per seed."""
    return np.stack([np.random.default_rng(int(s)).integers(0, n_pool, size=size) for s in seeds])


def prefix_equals_e1(n_pool: int, block: np.ndarray, seeds, budgets) -> bool:
    """Do the prefixes of the long draw equal E1's separate draws (section 11.2)?"""
    for row, s in zip(block, seeds, strict=True):
        for n in budgets:
            if not np.array_equal(row[:n], np.random.default_rng(int(s)).integers(0, n_pool, size=n)):
                return False
    return True


def sample_indices(n_pool: int, seeds, budgets, reg: Registry = REGISTERED) -> tuple:
    """({budget: (n_seeds, budget) indices}, nested).

    A sample is the first n entries of ``default_rng(seed).integers(0, N, size=3600)``. If
    those prefixes do not equal E1's own draws, every budget is drawn separately as in E1
    and the samples are not treated as nested.
    """
    seeds = [int(s) for s in seeds]
    block = draw_block(n_pool, seeds, reg.draw_size)
    if prefix_equals_e1(n_pool, block, seeds, reg.e1_budgets):
        return {n: block[:, :n] for n in budgets}, True
    return {n: draw_block(n_pool, seeds, n) for n in budgets}, False


# ----------------------------------------------------------- store and problem


def load_store(name: str, reg: Registry = REGISTERED) -> dict:
    """The arrays and metadata of a store, in the format ``e1_common.Store`` takes."""
    z = np.load(pathlib.Path(reg.store_dir) / f"{name}.npz", allow_pickle=False)
    meta = json.loads((pathlib.Path(reg.store_dir) / f"{name}.json").read_text(encoding="utf-8"))
    return {"meta": meta, **{k: z[k] for k in z.files}}


def problem_for(store: E.Store) -> E1S.Problem:
    """Candidate metadata and pool truth, exactly as ``e1_1_study.real`` builds them."""
    truth = store.pq.sum(axis=1) / store.base.sum()
    return E1S.Problem(store.families, store.bytes, store.vectors, store.compression, truth)


def first_position(meta: dict, pipeline: dict) -> int:
    """Position of the first candidate with this pipeline (float16, or per-vector int8)."""
    for c in meta["candidates"]:
        if c["pipeline"] == pipeline:
            return int(c["position"])
    raise ValueError(f"no candidate with pipeline {pipeline}")


def distinct_candidates(meta: dict) -> list:
    """Positions of the first occurrence of each pipeline (37 of the 40 in S0)."""
    seen, keep = set(), []
    for c in meta["candidates"]:
        key = json.dumps(c["pipeline"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            keep.append(int(c["position"]))
    return keep


def free_oracle(prob: E1S.Problem, target: float):
    """Smallest candidate whose pool truth is at least the target (as in ``e1_1_study.summarise``)."""
    feas = [int(j) for j in np.flatnonzero(prob.truth >= target)]
    return S.smallest(feas, prob.bytes, prob.vectors)


def chain_oracle(prob: E1S.Problem, target: float):
    """Smallest candidate among each chain's longest initial run with pool truth above the target."""
    passed, stopped = [], set()
    for k, fam in enumerate(prob.families):
        if fam in stopped:
            continue
        if prob.truth[k] > target:
            passed.append(k)
        else:
            stopped.add(fam)
    return S.smallest(passed, prob.bytes, prob.vectors)


def oracle_compression(prob: E1S.Problem, k) -> float:
    return 1.0 if k is None else float(prob.compression[k])


# ---------------------------------------------------------------------- gates


def gate_values(d: dict) -> dict:
    """G2: every value is finite and in [0, 1]; ``base_b1`` equals 1 for every query."""
    out, ok = {"arrays": {}}, True
    for key in ("base_labels", "base_b1", "cand_labels", "cand_b1"):
        a = np.asarray(d[key])
        finite = bool(np.isfinite(a).all())
        in_range = bool(finite and (a >= 0.0).all() and (a <= 1.0).all())
        out["arrays"][key] = {"finite": finite, "in_unit_interval": in_range}
        ok &= in_range
    out["base_b1_all_one"] = bool((np.asarray(d["base_b1"]) == 1.0).all())
    out["passed"] = bool(ok and out["base_b1_all_one"])
    return out


def gate_identity(d: dict, tol: float = 1e-12) -> dict:
    """G3 (ViDoRe stores): where ``base_labels`` = 1, ``cand_labels`` equals ``cand_b1`` within 1e-12."""
    g1 = np.asarray(d["base_labels"]) == 1.0
    diff = np.abs(d["cand_labels"][:, g1] - d["cand_b1"][:, g1])
    worst = float(diff.max()) if diff.size else 0.0
    return {"queries_with_base_labels_1": int(g1.sum()), "max_abs_difference": worst, "tolerance": tol,
            "passed": bool(worst <= tol)}


def gate_reproduction(name: str, selections: dict, reg: Registry, methods, budgets=None) -> dict:
    """G4: selections under labels at the E1 budgets equal those stored by E1.1 and E1.2, seed by seed."""
    path = pathlib.Path(reg.e1_selections[name])
    ref = np.load(path, allow_pickle=False)
    compared, mismatches = 0, []
    for n in (reg.e1_budgets if budgets is None else budgets):
        for target in PRIMARY_THRESHOLDS:
            for m in methods:
                key = sel_key(n, target, m)
                if key not in ref.files or key not in selections:
                    mismatches.append({"key": key, "problem": "missing"})
                    continue
                compared += 1
                a, b = ref[key], selections[key]
                if a.shape != b.shape or not np.array_equal(a, b):
                    mismatches.append({"key": key,
                                       "seeds_differing": int((a != b).sum()) if a.shape == b.shape else None})
    return {"file": path.name, "sha256": sha256_file(path), "methods": list(methods), "keys_compared": compared,
            "mismatches": mismatches, "passed": bool(compared > 0 and not mismatches)}


def require(gate: str, result: dict, where: str) -> dict:
    if not result["passed"]:
        raise GateFailure(f"{where}: gate {gate} failed. The experiment stops until this is explained.\n"
                          f"{json.dumps(result, indent=1, default=_json_default)[:2000]}")
    return result


# ------------------------------------------------------- procedures (wrappers)


def method_c(pvals: np.ndarray, prob: E1S.Problem, delta: float):
    """Method C on one sample's p-values: the frozen fixed-sequence test at delta / chains."""
    return S.smallest(S.fixed_sequence_by_family(pvals, prob.families, delta / prob.F), prob.bytes, prob.vectors)


def full_pool_decision(pq: np.ndarray, base: np.ndarray, prob: E1S.Problem, target: float, delta: float):
    """P0 given the whole pool once, each query once, in stored order (section 11.1). MEASURED."""
    p = S.betting_pvalues(S.to_y(pq, base[None, :], target), target / (1.0 + target))
    return method_c(p, prob, delta)


def sel_key(n: int, target: float, method: str) -> str:
    return f"n{n}_T{target}_{method}"  # the key format of the E1 selection files


def simulate(pq: np.ndarray, base: np.ndarray, prob: E1S.Problem, idx_by_budget: dict, thresholds, *,
             deltas=DELTAS, methods=("C", "D", "I8", "C-split"), i8_index: int | None = None,
             ab_seeds: dict | None = None, a_bonf_budgets=(), chunk_bytes: int = CHUNK_BYTES) -> dict:
    """Per-seed selections (-1 = abstain) for every budget, threshold and procedure.

    The orchestration mirrors ``e1_1_study.run_cell``; every statistic comes from the
    frozen functions. ``ab_seeds[n]`` is the number of leading seeds on which rule A and
    B are run at budget n; A-bonf is added at the budgets in ``a_bonf_budgets``.
    Returns {key: int16 array} with the keys of the E1 selection files.
    """
    ab_seeds = ab_seeds or {}
    sel = {}
    for n, idx in sorted(idx_by_budget.items()):
        n_seeds = idx.shape[0]
        cal, hold = split_queries(n, 0.5, 0)
        n_ab = min(int(ab_seeds.get(n, 0)), n_seeds)
        for target in thresholds:
            for d in deltas:
                for m in methods:
                    sel[sel_key(n, target, f"{m}@{d}")] = np.full(n_seeds, -1, np.int16)
        p_hold_ab = {target: np.ones((n_ab, prob.K)) for target in thresholds}
        per_seed = prob.K * n * 8
        step = n_seeds if per_seed * n_seeds <= 2 * chunk_bytes else max(8, (chunk_bytes // per_seed) // 8 * 8)
        need_hold = "C-split" in methods or n_ab > 0
        for lo in range(0, n_seeds, step):
            rows = idx[lo:lo + step]
            pq_all = np.ascontiguousarray(pq[:, rows].transpose(1, 0, 2))  # (seeds, K, n), as run_cell stacks it
            b_all = base[rows]  # (seeds, n)
            if "C-split" in methods:
                point_cal = pq_all[:, :, cal].mean(axis=2) / b_all[:, cal].mean(axis=1)[:, None]
            for target in thresholds:
                m0 = target / (1.0 + target)
                p_full = S.betting_pvalues(S.to_y(pq_all, b_all[:, None, :], target), m0)
                if need_hold:
                    p_hold = S.betting_pvalues(S.to_y(pq_all[:, :, hold], b_all[:, None, hold], target), m0)
                    if lo < n_ab:
                        take = min(lo + rows.shape[0], n_ab) - lo
                        p_hold_ab[target][lo:lo + take] = p_hold[:take]
                for d in deltas:
                    for j in range(rows.shape[0]):
                        s = lo + j
                        if "C" in methods:
                            k = method_c(p_full[j], prob, d)
                            sel[sel_key(n, target, f"C@{d}")][s] = -1 if k is None else k
                        if "D" in methods:
                            rej = [int(q) for q in np.flatnonzero(p_full[j] <= d / prob.K)]
                            k = S.smallest(rej, prob.bytes, prob.vectors)
                            sel[sel_key(n, target, f"D@{d}")][s] = -1 if k is None else k
                        if "I8" in methods:
                            sel[sel_key(n, target, f"I8@{d}")][s] = i8_index if p_full[j, i8_index] <= d else -1
                        if "C-split" in methods:
                            order = np.lexsort((np.arange(prob.K), -prob.bytes, -point_cal[j]))
                            k = S.smallest(S.fixed_sequence(p_hold[j], order, d), prob.bytes, prob.vectors)
                            sel[sel_key(n, target, f"C-split@{d}")][s] = -1 if k is None else k
            del pq_all, b_all
        if n_ab:
            bonf = n in a_bonf_budgets
            for target in thresholds:
                sel[sel_key(n, target, "A")] = np.full(n_ab, -1, np.int16)
                if bonf:
                    sel[sel_key(n, target, "A-bonf")] = np.full(n_ab, -1, np.int16)
            for s in range(n_ab):
                pq_s, b_s = np.ascontiguousarray(pq[:, idx[s]]), base[idx[s]]
                bnd = E1S.a_bounds(pq_s[:, cal], b_s[cal], E1S.A_CONF, "library")
                bnd_bonf = E1S.a_bounds(pq_s[:, cal], b_s[cal], E1S.A_BONF_CONF, "library") if bonf else None
                for target in thresholds:
                    k = S.walk_select(bnd, prob.families, prob.bytes, prob.vectors, target)
                    sel[sel_key(n, target, "A")][s] = -1 if k is None else k
                    if bonf:
                        kb = S.walk_select(bnd_bonf, prob.families, prob.bytes, prob.vectors, target)
                        sel[sel_key(n, target, "A-bonf")][s] = -1 if kb is None else kb
            for target in thresholds:
                a = sel[sel_key(n, target, "A")]
                for d in deltas:
                    b_sel = np.full(n_ab, -1, np.int16)
                    for s in range(n_ab):
                        ka = int(a[s])
                        if ka >= 0 and p_hold_ab[target][s, ka] <= d:
                            b_sel[s] = ka
                    sel[sel_key(n, target, f"B@{d}")] = b_sel
    return sel


class Replication:
    """Selections for the replication block (seeds 2000-3999), computed only for the
    cells a registered rule asks for (section 21, rule 2), and at most once each."""

    def __init__(self, pq, base, prob, n_pool: int, reg: Registry, i8_index: int | None = None):
        self.pq, self.base, self.prob, self.n_pool, self.reg, self.i8 = pq, base, prob, n_pool, reg, i8_index
        self.selections: dict = {}

    def get(self, n: int, target: float, method: str) -> np.ndarray:
        key = sel_key(n, target, method)
        if key not in self.selections:
            name, delta = method.split("@")
            idx, _ = sample_indices(self.n_pool, range(*self.reg.replication_seeds), (n,), self.reg)
            sim = simulate(self.pq, self.base, self.prob, idx, (target,), deltas=(float(delta),), methods=(name,),
                           i8_index=self.i8)
            self.selections[key] = sim[key]
        return self.selections[key]


# ------------------------------------------------------------------ summaries


def rate_with_interval(successes: int, trials: int) -> dict:
    lo, hi = E1S.clopper_pearson(int(successes), int(trials))
    return {"rate": int(successes) / int(trials), "ci95": [lo, hi], "count": int(successes), "seeds": int(trials)}


def consistent_with(level: float, misses: int, trials: int) -> bool:
    """The E1.2 criterion (section 16.3): the lower end of the 95% Clopper-Pearson interval is at most ``level``."""
    return bool(E1S.clopper_pearson(int(misses), int(trials))[0] <= level)


def compression_of(k: np.ndarray, prob: E1S.Problem) -> np.ndarray:
    """Compression deployed on each seed; 1.0 where the procedure abstains."""
    return np.where(k >= 0, prob.compression[np.maximum(k, 0)], 1.0)


def deployed_flags(k: np.ndarray, prob: E1S.Problem, target: float, float16_index: int) -> dict:
    """Per-seed booleans: deployed, non-trivial, substantial, miss against the pool truth."""
    dep = k >= 0
    comp = compression_of(k, prob)
    truth = np.where(dep, prob.truth[np.maximum(k, 0)], 1.0)
    return {"deployed": dep, "nontrivial": dep & (k != float16_index),
            "substantial": dep & (comp >= SUBSTANTIAL_COMPRESSION), "miss": dep & (truth < target)}


def summarise(k: np.ndarray, prob: E1S.Problem, target: float, float16_index: int, delta: float | None = None) -> dict:
    """One cell (section 11.4): the fields of ``e1_1_study.summarise`` plus the registered additions."""
    n_seeds = int(k.shape[0])
    f = deployed_flags(k, prob, target, float16_index)
    comp = compression_of(k, prob)
    truth = np.where(f["deployed"], prob.truth[np.maximum(k, 0)], 1.0)
    free, chain = free_oracle(prob, target), chain_oracle(prob, target)
    counts = np.bincount(k.astype(np.int64) + 1, minlength=prob.K + 1)
    common = np.lexsort((np.arange(counts.size), -counts))[:3] - 1  # most frequent first; ties by position
    n_miss = int(f["miss"].sum())
    row = {
        "seeds": n_seeds,
        "deployed_rate": float(f["deployed"].mean()), "abstain_rate": float(1.0 - f["deployed"].mean()),
        "nontrivial": rate_with_interval(int(f["nontrivial"].sum()), n_seeds),
        "substantial": rate_with_interval(int(f["substantial"].sum()), n_seeds),
        "compression_median": float(np.median(comp)),
        "compression_p10_p90": [float(np.quantile(comp, 0.1)), float(np.quantile(comp, 0.9))],
        "compression_mean_log2": float(np.mean(np.log2(comp))),
        "miss": rate_with_interval(n_miss, n_seeds),
        "worst_miss_truth": float(truth[f["miss"]].min()) if n_miss else None,
        "free_oracle_index": free, "free_oracle_compression": oracle_compression(prob, free),
        "chain_oracle_index": chain, "chain_oracle_compression": oracle_compression(prob, chain),
        "median_compression_over_free_oracle": float(np.median(comp / oracle_compression(prob, free))),
        "median_compression_over_chain_oracle": float(np.median(comp / oracle_compression(prob, chain))),
        "most_common_selection": [int(x) for x in common],
    }
    if delta is not None:
        row["consistent_with_delta"] = consistent_with(delta, n_miss, n_seeds)
    return row


def cell_identity(construct: str, target: float, delta, n: int, n_pool: int, procedure: str) -> dict:
    """What every simulated row carries: the threshold, delta, the budget and its class."""
    return {"construct": construct, "threshold": target, "delta": delta, "n": int(n),
            "budget_class": budget_class(n, n_pool), "evidence": evidence_mark(n, n_pool), "procedure": procedure}


def check_miss_rate(misses: np.ndarray, level: float, replicate) -> dict:
    """A miss rate against ``level`` with the replication rule of section 16.3.

    ``replicate()`` returns the per-seed miss flags of the replication block. It is called
    only if the first 2,000 seeds fail the check.
    """
    n = int(misses.shape[0])
    first = rate_with_interval(int(misses.sum()), n)
    out = {"level": level, "rate": first["rate"], "ci95": first["ci95"], "count": first["count"], "seeds": n,
           "replicated": False, "confirmed_failure": False}
    if first["ci95"][0] > level:
        extra = replicate()
        pooled = rate_with_interval(int(misses.sum()) + int(extra.sum()), n + int(extra.shape[0]))
        out.update({"replicated": True, "pooled_rate": pooled["rate"], "pooled_ci95": pooled["ci95"],
                    "pooled_count": pooled["count"], "pooled_seeds": pooled["seeds"],
                    "confirmed_failure": bool(pooled["ci95"][0] > level)})
    return out


def decide_rate(hits: np.ndarray, replicate) -> dict:
    """A deployment rate against 0.90 with the borderline rule of section 11.5.

    ``replicate()`` returns the per-seed flags of the replication block. It is called only
    if the 95% interval of the first 2,000 seeds contains 0.90; the pooled rate then decides.
    """
    n = int(hits.shape[0])
    first = rate_with_interval(int(hits.sum()), n)
    borderline = bool(first["ci95"][0] <= RATE_THRESHOLD <= first["ci95"][1])
    out = {"rate": first["rate"], "ci95": first["ci95"], "count": first["count"], "seeds": n,
           "borderline_rule_used": borderline, "deciding_rate": first["rate"]}
    if borderline:
        extra = replicate()
        pooled = rate_with_interval(int(hits.sum()) + int(extra.sum()), n + int(extra.shape[0]))
        out.update({"pooled_rate": pooled["rate"], "pooled_ci95": pooled["ci95"], "pooled_count": pooled["count"],
                    "pooled_seeds": pooled["seeds"], "deciding_rate": pooled["rate"]})
    out["at_least_0.90"] = bool(out["deciding_rate"] >= RATE_THRESHOLD)
    return out


def n_star(budgets, n_pool: int, decide) -> dict:
    """n* (section 6): the smallest registered budget with a rate of at least 0.90 there and
    at every larger registered budget. ``decide(n)`` returns a :func:`decide_rate` result.

    Budgets are scanned from the largest downwards and the scan stops at the first one
    that fails: no budget below it can change n*.
    """
    best, cells = None, {}
    for n in sorted(budgets, reverse=True):
        cells[str(n)] = decide(n)
        if not cells[str(n)]["at_least_0.90"]:
            break
        best = n
    decided_by = best if best is not None else max(budgets)
    return {"n_star": best, "decided_by_budget": decided_by, "budget_class": budget_class(decided_by, n_pool),
            "evidence": evidence_mark(decided_by, n_pool), "deciding_cells": cells}


def worst_by_pool(cells: dict, reg: Registry) -> dict:
    """Pools are summarised by their worst store (sections 8 and 16.4): the lowest deciding rate.

    ``cells`` maps a store to a :func:`decide_rate` result. Stores are never pooled.
    """
    out = {}
    for name, cell in cells.items():
        pool = reg.stores[name].pool
        if pool not in out or cell["deciding_rate"] < out[pool]["deciding_rate"]:
            out[pool] = {"worst_store": name, "deciding_rate": cell["deciding_rate"],
                         "at_least_0.90": cell["at_least_0.90"]}
    return out


def monotone_violations(compression_by_budget: dict) -> int:
    """Seeds whose deployed compression falls as the budget grows. For P0 on nested samples this
    must be 0 (section 11.6); anything else is an implementation error."""
    budgets = sorted(compression_by_budget)
    bad = np.zeros(compression_by_budget[budgets[0]].shape[0], bool)
    for a, b in zip(budgets, budgets[1:], strict=False):
        bad |= compression_by_budget[b] < compression_by_budget[a]
    return int(bad.sum())


def check_monotone(selections: dict, prob: E1S.Problem, budgets, thresholds, nested: bool, where: str,
                   deltas=DELTAS) -> dict:
    """The monotonicity check of section 11.6 for P0. A single violation stops the run."""
    if not nested:
        return {"checked": False, "reason": "samples are not nested across budgets"}
    for target in thresholds:
        for d in deltas:
            by_n = {n: compression_of(selections[sel_key(n, target, f"C@{d}")], prob) for n in budgets}
            bad = monotone_violations(by_n)
            if bad:
                raise GateFailure(f"{where}: P0 deployed less compression at a larger nested budget on {bad} "
                                  f"seed(s) (T = {target}, delta = {d}). This is an implementation error (section 11.6).")
    return {"checked": True, "violations": 0}


# ---------------------------------------------------------- provenance, output


def environment() -> dict:
    env = E.environment()
    env["executable"] = sys.executable
    env["cache_bytes_env"] = os.environ.get("OPTIVISION_CACHE_BYTES")
    env["score_device_env"] = os.environ.get("OPTIVISION_SCORE_DEVICE")
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        env["blas"] = {k: blas.get(k) for k in ("name", "version")}
    except (TypeError, KeyError):  # numpy older than 1.25, or a build without this record
        env["blas"] = None
    return env


def header(experiment: str, reg: Registry, validation: dict | None = None) -> dict:
    """What identifies a result file: the registration, the code, Method C, the seeds, the inputs."""
    g = git_state(reg.root)
    inputs = None
    if validation is not None:
        inputs = {n: {"npz_sha256": s["npz"]["sha256"], "json_sha256": s["json"]["sha256"],
                      "candidate_spec_sha256": s["meta"]["candidate_spec_sha256"]}
                  for n, s in validation["stores"].items()}
    return {
        "experiment": experiment,
        "registered_configuration": reg is REGISTERED,
        "preregistration": {"path": PREREG_PATH, "commit": PREREG_COMMIT, "blob": PREREG_BLOB,
                            "registry_commit": REGISTRY_COMMIT, "blob_at_head": g.get("prereg_blob_at_head")},
        "code_commit": g.get("commit"), "code_tree_clean": not (g.get("tracked_changes") or g.get("uncommitted_code")),
        "method_c": method_c_config(),
        "candidate_spec_sha256": reg.s0_spec_sha256,
        "seeds": {"resamples": [0, reg.n_seeds - 1], "replication_block": [reg.replication_seeds[0],
                                                                         reg.replication_seeds[1] - 1],
                  "sampling": f"first n entries of numpy.random.default_rng(seed).integers(0, N, size={reg.draw_size})"},
        "accepted_open_points": sorted(ACCEPTED_OPEN_POINTS),
        "inputs": inputs,
    }


def out_path(reg: Registry, experiment: str, *parts: str) -> pathlib.Path:
    """A path inside the experiment's own output directory, and nowhere else."""
    root = pathlib.Path(reg.out_roots[experiment]).resolve()
    path = root.joinpath(*parts).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{path} is outside the output directory of {experiment}")
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def dump_json(obj) -> bytes:
    """Deterministic JSON bytes: sorted keys, LF line endings on every platform."""
    return (json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + "\n").encode("utf-8")


def write_json(path: pathlib.Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dump_json(obj)
    path.write_bytes(data)
    return sha256_bytes(data)


def read_json(path: pathlib.Path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def write_npz(path: pathlib.Path, arrays: dict) -> str:
    """A .npz with fixed timestamps and member order, so the same arrays give the same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            z.writestr(info, buf.getvalue())
    return sha256_file(path)


def write_text(path: pathlib.Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.replace("\r\n", "\n").encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def md_table(columns, rows) -> str:
    """A Markdown table. Cells are converted with ``fmt``."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x).replace("|", "\\|")


def fmt_rate(r: dict) -> str:
    return f"{r['rate']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"


# ----------------------------------------------------------- long-run hygiene


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep during a run; no-op elsewhere (as in q8_scaling.py)."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def awake_clock() -> float:
    """Seconds on a clock that stops while the machine sleeps."""
    if sys.platform == "win32":
        import ctypes

        ticks = ctypes.c_ulonglong()
        ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(ticks))
        return ticks.value / 1e7
    return time.monotonic()


class Timer:
    """Start and end times of a unit of work, and whether the machine slept during it (section 20)."""

    SLEEP_TOLERANCE = 5.0

    def __init__(self):
        self.wall, self.awake = time.time(), awake_clock()

    def record(self) -> dict:
        wall, awake = time.time(), awake_clock()
        asleep = (wall - self.wall) - (awake - self.awake)
        stamp = "%Y-%m-%dT%H:%M:%SZ"
        return {"started_utc": time.strftime(stamp, time.gmtime(self.wall)),
                "finished_utc": time.strftime(stamp, time.gmtime(wall)), "seconds": wall - self.wall,
                "seconds_not_awake": asleep, "slept": bool(asleep > self.SLEEP_TOLERANCE)}


class RunLog:
    """Append-only log of attempts, one entry per unit of work (section 22.2: gates and their
    results, including failed attempts). A unit still marked "started" was killed or interrupted."""

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.entries = read_json(self.path) if self.path.is_file() else []
        for e in self.entries:
            if e["status"] == "started":
                e["status"] = "invalid"
                e["reason"] = "did not finish: the process was killed or interrupted"

    def attempts(self, unit: str) -> list:
        return [e for e in self.entries if e["unit"] == unit]

    def valid(self, unit: str) -> dict | None:
        done = [e for e in self.attempts(unit) if e["status"] == "valid"]
        return done[-1] if done else None

    def begin(self, unit: str, commit) -> dict:
        entry = {"unit": unit, "attempt": len(self.attempts(unit)) + 1, "commit": commit, "status": "started",
                 "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.entries.append(entry)
        write_json(self.path, self.entries)
        return entry

    def end(self, entry: dict, status: str, **info) -> None:
        entry.update({"status": status, **info})
        write_json(self.path, self.entries)


# ------------------------------------------------ rows, units, dry run, report


#: row name: (key in the selection files, carries an error level, validity check registered in section 11.6)
PROCEDURES = {
    "P0": ("C", True, True), "I8": ("I8", True, True), "D": ("D", True, True), "C-split": ("C-split", True, True),
    "B": ("B", True, False), "A": ("A", False, False), "A-bonf": ("A-bonf", False, False),
}
N13 = ("An error level was not kept in cell {cell}. The cause is an implementation error or a violated "
       "assumption. No usefulness claim is made from that cell.")


def simulated_rows(construct: str, thresholds, sel: dict, prob: E1S.Problem, rep: Replication, budgets,
                   n_pool: int, float16_index: int, where: str, procedures=("P0",)) -> list:
    """One row per (threshold, budget, procedure, delta) with the outcomes of section 11.4.

    Procedures with a registered validity check get it here, with the replication rule of
    section 16.3. A confirmed failure stops the experiment.
    """
    rows = []
    for target in thresholds:
        for n in budgets:
            for name in procedures:
                key, has_delta, checked = PROCEDURES[name]
                for delta in (DELTAS if has_delta else (None,)):
                    method = key if delta is None else f"{key}@{delta}"
                    k = sel.get(sel_key(n, target, method))
                    if k is None:
                        continue  # A and B are registered at some budgets only
                    row = {**cell_identity(construct, target, delta, n, n_pool, name),
                           **summarise(k, prob, target, float16_index, delta)}
                    if checked:
                        row["validity"] = check_miss_rate(
                            deployed_flags(k, prob, target, float16_index)["miss"], delta,
                            lambda n=n, target=target, method=method: deployed_flags(
                                rep.get(n, target, method), prob, target, float16_index)["miss"])
                        if row["validity"]["confirmed_failure"]:
                            cell = f"{where}, {construct}, T = {target}, delta = {delta}, n = {n}, {name}"
                            raise GateFailure(N13.format(cell=cell))
                    rows.append(row)
    return rows


def run_store_units(experiment: str, reg: Registry, names, prepare, analyse, prefix: str) -> list:
    """Gates for every selected store first (a failure stops the experiment), then one unit of
    work per store. A unit's files are written only if its checks passed and the machine did
    not sleep; every attempt is logged."""
    names = list(names)
    validation = require_run_allowed(experiment, reg)
    log = RunLog(out_path(reg, experiment, "runs.json"))
    commit = git_state(reg.root).get("commit")
    done = []
    keep_awake(True)
    try:
        gate_timer = Timer()
        entries = {name: log.begin(name, commit) for name in names}
        try:
            prepared = {name: prepare(name, reg) for name in names}
        except GateFailure as exc:
            for entry in entries.values():
                log.end(entry, "invalid", reason=str(exc)[:4000])
            raise
        gates_slept = gate_timer.record()["slept"]
        for i, name in enumerate(names):
            timer = Timer()
            try:
                result, arrays = analyse(name, prepared.pop(name), reg, validation)
            except GateFailure as exc:
                for later in names[i:]:
                    log.end(entries[later], "invalid", reason=str(exc)[:4000])
                raise
            rec = timer.record()
            if rec["slept"] or gates_slept:
                log.end(entries[name], "invalid", reason="the machine slept during the run", **rec)
                continue
            files = {f"{prefix}_{name}.json": result, f"{prefix}_{name}_selections.npz": arrays}
            outputs = {f: (write_npz if f.endswith(".npz") else write_json)(out_path(reg, experiment, f), content)
                       for f, content in files.items()}
            log.end(entries[name], "valid", outputs=outputs, **rec)
            done.append(name)
            print(f"{experiment} {name}: done in {rec['seconds']:.0f}s", flush=True)
    finally:
        keep_awake(False)
    return done


def finish_experiment(experiment: str, reg: Registry, prefix: str, summarise_fn, report_fn) -> dict:
    """Summary, report and provenance, once every registered store has a unit from this commit."""
    commit = git_state(reg.root).get("commit")
    stores, pending = {}, []
    for name in reg.stores:
        path = out_path(reg, experiment, f"{prefix}_{name}.json")
        if not path.is_file():
            pending.append(f"{name}: no valid unit yet")
            continue
        stores[name] = read_json(path)
        if reg.registered and stores[name]["header"]["code_commit"] != commit:
            pending.append(f"{name}: produced at another commit; repeat it")
    if pending:
        print(f"{experiment}: summary not written.\n  " + "\n  ".join(pending), flush=True)
        return {"finished": False, "pending": pending}
    summary = {"header": header(experiment, reg), **summarise_fn(stores, reg)}
    out_dir = out_path(reg, experiment)
    write_json(out_dir / f"{prefix}_summary.json", summary)
    write_text(out_dir / f"{experiment}_REPORT.md", report_fn(summary, stores))
    provenance = {
        "experiment": experiment, "environment": environment(), "git": git_state(reg.root),
        "validation": validate_inputs(reg), "run_log": "runs.json",
        "e1_selection_files": {n: check_selection_file(n, reg) for n in reg.stores},
        "outputs": {p.relative_to(out_dir).as_posix(): sha256_file(p) for p in sorted(out_dir.rglob("*"))
                    if p.is_file() and p.name != f"{prefix}_provenance.json"},
    }
    write_json(out_dir / f"{prefix}_provenance.json", provenance)
    print(f"{experiment}: summary, report and provenance written to {out_dir}", flush=True)
    return {"finished": True, "summary": summary}


def dry_run_common(experiment: str, reg: Registry = REGISTERED) -> dict:
    """What can be validated without reading a per-query value or writing a result."""
    validation = validate_inputs(reg)
    selections = {name: check_selection_file(name, reg) for name in reg.stores}
    plan = {}
    for name, spec in reg.stores.items():
        plan[name] = {"encoder": spec.encoder, "pool": spec.pool, "pool_queries": spec.n_queries,
                      "metric": spec.metric, "pool_capped_budget": reg.pool_capped_budget(spec.n_queries),
                      "budgets": budget_table(spec.n_queries, reg),
                      "ab_seeds": {str(n): reg.ab_seed_count(n, spec.n_queries) for n in reg.budgets(spec.n_queries)}}
    sampling = {}
    for n_pool in sorted({s.n_queries for s in reg.stores.values()}):
        sampling[str(n_pool)] = {
            "prefixes_equal_e1_draws_resample_seeds": sample_indices(n_pool, range(reg.n_seeds), (), reg)[1],
            "prefixes_equal_e1_draws_replication_seeds": sample_indices(n_pool, range(*reg.replication_seeds), (), reg)[1],
        }
    out_dir = pathlib.Path(reg.out_roots[experiment])
    inside = RESEARCH in out_dir.resolve().parents if reg.registered else True
    existing = sorted(p.name for p in out_dir.iterdir()) if out_dir.is_dir() else []
    g = git_state(reg.root)
    blocked = [f"open point {k} is not accepted" for k in sorted(open_points_for(experiment))]
    if g.get("available") and (g["tracked_changes"] or g["uncommitted_code"]):
        blocked.append("the working tree has uncommitted changes")
    if g.get("available") and g["prereg_blob_at_head"] not in PREREG_BLOBS_IMPLEMENTED:
        blocked.append("the preregistration at HEAD is not a version this harness implements")
    if not validation["ok_for_run"]:
        blocked.append("an input does not have its registered bytes on this checkout")
    # X3 reads the E1 selections only for an extra check of its identity run
    selections_ok = experiment == "X3" or all(s["ok"] or "lfs_pointer_oid" in s for s in selections.values())
    return {
        "experiment": experiment, "mode": "dry run: no per-query value read, no result computed or written",
        "git": g, "validation": validation, "e1_selection_files": selections, "plan": plan,
        "thresholds": {k: list(v) for k, v in THRESHOLDS.items()}, "deltas": list(DELTAS),
        "seeds": header(experiment, reg)["seeds"], "sampling": sampling,
        "output_dir": {"path": out_dir.as_posix(), "inside_reports_research": bool(inside), "existing_files": existing},
        "open_points": {k: v for k, v in OPEN_POINTS.items() if experiment in v["affects"]},
        "run_blocked_by": blocked,
        "inputs_ok": bool(validation["identity_ok"] and selections_ok and inside
                          and all(v for s in sampling.values() for v in s.values())),
    }


def print_dry_run(rep: dict, as_json: bool = False) -> None:
    if as_json:
        print(dump_json(rep).decode("utf-8"))
        return
    v, g = rep["validation"], rep["git"]
    print(f"{rep['experiment']} dry run. No per-query value is read and nothing is computed or written.")
    if g.get("available"):
        print(f"  branch {g['branch']} at {g['commit'][:12]}; preregistration commit present: "
              f"{g['prereg_commit_present']}; its blob at HEAD is the registered one: "
              f"{g['prereg_blob_at_head'] == PREREG_BLOB}")
    for name, s in v["stores"].items():
        state = "registered bytes" if s["ok_for_run"] else ("registered content" if s["identity_ok"] else "MISMATCH")
        sel = rep["e1_selection_files"][name]
        meta = s.get("meta", {})
        print(f"  store {name:18s} {state:18s} candidates {meta.get('n_candidates')} "
              f"({meta.get('n_distinct_candidates')} distinct), chains {meta.get('n_chains')}, "
              f"E1 selections {'ok' if sel['ok'] else sel.get('problem')}")
        for p in s["problems"]:
            print(f"      note: {p}")
    space = v["generated_candidate_space"]
    print(f"  candidate specification generated by the code equals the registered hash: {space['ok']}")
    print(f"  frozen code blobs match: {v['code']['ok']}; Method C constants match the E1.2 freeze: "
          f"{v['method_c']['ok']} (bet cap {v['method_c']['bet_cap']}, level delta/{v['method_c']['chains']})")
    for name, p in rep["plan"].items():
        cells = ", ".join(f"{b['n']}{'*' if b['pool_capped'] else ''}"
                          f"({'p' if b['budget_class'] == 'pool-supported' else 'x'})" for b in p["budgets"])
        print(f"  budgets {name:18s} N = {p['pool_queries']}: {cells}")
    print("      (p) pool-supported, (x) extrapolated, * the pool-capped budget")
    print(f"  seeds {rep['seeds']['resamples']}, replication block {rep['seeds']['replication_block']}")
    print(f"  prefixes of the long draw equal E1's draws, per pool size: "
          f"{ {k: all(s.values()) for k, s in rep['sampling'].items()} }")
    print(f"  output directory {rep['output_dir']['path']} "
          f"(existing files: {len(rep['output_dir']['existing_files'])})")
    for k, p in rep["open_points"].items():
        print(f"  {k}: {p['text']}\n      implemented, not accepted: {p['implemented']}")
    print(f"  inputs ok: {rep['inputs_ok']}")
    for b in rep["run_blocked_by"]:
        print(f"  a run is blocked: {b}")


def rate_by_budget_table(stores: dict, value) -> str:
    """Rows are stores. Pool-supported and extrapolated budgets get separate columns (section 11.5).

    ``value(store_json, n)`` returns the cell text for a budget registered on that store.
    """
    classes = {name: {b["n"]: b["budget_class"] for b in s["budgets"]} for name, s in stores.items()}
    columns = [(n, cls) for cls in ("pool-supported", "extrapolated")
               for n in sorted({n for c in classes.values() for n, k in c.items() if k == cls})]
    head = ["store"] + [f"n = {n} ({cls})" for n, cls in columns]
    rows = [[name] + [value(s, n) if classes[name].get(n) == cls else None for n, cls in columns]
            for name, s in stores.items()]
    return md_table(head, rows)
