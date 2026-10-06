"""Parity of the package's selection arithmetic with the frozen research functions.

(a) Seeded random problems: p-values bit for bit, rejected sets and selections identical,
    against ``scripts/research/e1_stats.py`` (imported by path inside the test; the package
    itself never imports research code). The rules are compared on the same problems.
(b) The method D path against the frozen orchestration ``e1_1_study.run_cell`` on synthetic
    data, for both rules.
(c) The golden-file equivalence test of ``METHOD_C_SPEC.md`` section 9.3: the stored E1
    selections of Method C and of method D are reproduced seed by seed from the stored
    per-query arrays. These inputs are registered and are read read-only, as an
    implementation-equivalence check: nothing is written, printed or analysed beyond the
    equality assertion.

By default (c) compares every fifth seed (400 of 2,000) on every store and cell. Set
``OPTIVISION_FULL_PARITY=1`` (read by this test only) to compare all 2,000 seeds.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from optivision.core.evaluate import Evidence
from optivision.core.selection import (
    BonferroniBetting,
    CandidateSpace,
    CandidateSpec,
    FixedSequenceBetting,
    Outcome,
    select,
)
from optivision.core.selection.spaces import REGISTERED_DEFAULT_SPACE_SHA256
from optivision.core.selection.statistics import (
    betting_pvalues,
    fixed_sequence_by_family,
    smallest,
    to_y,
)

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "scripts" / "research"
E1 = ROOT / "reports" / "research" / "E1"

if not (RESEARCH / "e1_stats.py").is_file():
    pytest.skip("scripts/research/e1_stats.py is absent", allow_module_level=True)

#: the registered blob of the frozen statistics module (scripts/research/x_common.py:80-82)
E1_STATS_BLOB = "13494307a4a42cd2c1da63f34aa0952a92b33130"


@pytest.fixture(scope="module")
def frozen():
    """The frozen ``e1_stats`` module, imported from its own directory."""
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(RESEARCH))
        yield importlib.import_module("e1_stats")


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def test_the_frozen_module_is_the_registered_blob():
    assert git_blob_sha1(RESEARCH / "e1_stats.py") == E1_STATS_BLOB


# ------------------------------------------------------------------ (a) random problems

NS = (1, 2, 3, 7, 25, 100, 225, 450)
TARGETS = (0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0)
DELTAS = (0.2, 0.05, 0.01)


def random_problem(rng: np.random.Generator, k: int, n: int):
    """Arrays valid as evidence, with arbitrary family interleavings and many exact size ties."""
    n_fam = int(rng.integers(1, min(k, 9) + 1))
    labels = [f"fam{j}" for j in range(n_fam)]
    families = [labels[j] for j in rng.integers(0, n_fam, size=k)]  # arbitrary interleaving
    if rng.random() < 0.5:  # b on the nDCG grid; else uniform
        grid = np.concatenate([1.0 / np.log2(np.arange(2, 7)), [0.0]])
        b = grid[rng.choice(6, size=n, p=[0.5, 0.08, 0.05, 0.03, 0.02, 0.32])]
    else:
        b = rng.random(n)
    if not b.any():
        b[0] = 1.0
    loss = rng.random(k) * rng.choice([0.0, 0.02, 0.1, 0.4])
    drop = rng.random((k, n)) < loss[:, None]
    shrink = rng.random((k, n))
    c = np.where(drop, b[None, :] * shrink, b[None, :])
    size = rng.choice([100.0, 200.0, 400.0, 800.0], size=k)  # exact ties on purpose
    vectors = rng.choice([8.0, 16.0], size=k)
    return c, b, families, size, vectors


def frozen_method_c(S, c, b, families, size, vectors, target, delta):
    """``x_common.method_c`` (scripts/research/x_common.py:940-948) over the frozen functions."""
    n_fam = len(dict.fromkeys(families))
    m0 = target / (1.0 + target)
    p = S.betting_pvalues(S.to_y(c, b[None, :], target), m0)
    rej = S.fixed_sequence_by_family(p, families, delta / n_fam)
    return S.smallest(rej, size, vectors), p, rej


def frozen_method_d(S, c, b, size, vectors, target, delta):
    """Method D exactly as scripts/research/e1_1_study.py:132-134 computes it from ``p``."""
    m0 = target / (1.0 + target)
    p = S.betting_pvalues(S.to_y(c, b[None, :], target), m0)
    rej = [int(j) for j in np.flatnonzero(p <= delta / c.shape[0])]
    return S.smallest(rej, size, vectors), p, rej


@pytest.mark.parametrize("n", NS)
def test_random_problems_match_the_frozen_functions_bit_for_bit(frozen, n):
    """44 candidate counts per sample size: 352 problems over the 8 sample sizes."""
    rng = np.random.default_rng(1000 + n)
    rule_c, rule_d = FixedSequenceBetting(), BonferroniBetting()
    for k in range(1, 45):
        c, b, families, size, vectors = random_problem(rng, k, n)
        target = float(rng.choice(TARGETS))
        delta = float(rng.choice(DELTAS))
        sel_f, p_f, rej_f = frozen_method_c(frozen, c, b, families, size, vectors, target, delta)

        p = betting_pvalues(to_y(c, b[None, :], target), target / (1.0 + target))
        assert p.tobytes() == p_f.tobytes(), (k, n, target)  # bitwise
        rej = fixed_sequence_by_family(p, families, delta / len(dict.fromkeys(families)))
        assert rej == rej_f
        assert smallest(rej, size, vectors) == sel_f

        res = rule_c.decide(c, b, families, size, vectors, target, delta)
        assert (res.selected, list(res.certified)) == (sel_f, rej_f)
        assert np.array(res.p_values).tobytes() == p_f.tobytes()
        assert res.level == delta / len(dict.fromkeys(families))

        sel_d, p_d, rej_d = frozen_method_d(frozen, c, b, size, vectors, target, delta)
        res_d = rule_d.decide(c, b, families, size, vectors, target, delta)
        assert (res_d.selected, list(res_d.certified)) == (sel_d, rej_d)
        assert np.array(res_d.p_values).tobytes() == p_d.tobytes()
        assert res_d.level == delta / k


def test_the_random_problem_set_has_every_outcome_and_exact_size_ties():
    """Guards the generator: the comparison above is not vacuous."""
    rng = np.random.default_rng(2024)
    seen = {"abstain": 0, "select": 0, "tie": 0, "multi_chain": 0}
    for n in (25, 100, 225):
        for k in range(2, 45, 3):
            c, b, families, size, vectors = random_problem(rng, k, n)
            res = FixedSequenceBetting().decide(c, b, families, size, vectors, 0.9, 0.05)
            seen["select" if res.selected is not None else "abstain"] += 1
            keys = list(zip(size, vectors, strict=True))
            seen["tie"] += int(len(set(keys)) < len(keys))
            seen["multi_chain"] += int(len(set(families)) > 1)
    assert all(v > 0 for v in seen.values()), seen


def test_boundary_cases_around_the_level_match_the_frozen_function(frozen):
    """300+ trials with the level placed at, one ulp below and one ulp above a p-value."""
    rng = np.random.default_rng(77)
    trials = 0
    for n in (7, 25, 100, 225):
        for k in range(2, 45, 4):
            c, b, families, size, vectors = random_problem(rng, k, n)
            target = float(rng.choice(TARGETS))
            p = frozen.betting_pvalues(frozen.to_y(c, b[None, :], target), target / (1.0 + target))
            for j in rng.choice(k, size=min(k, 3), replace=False):
                for level in (float(p[j]), float(np.nextafter(p[j], 0.0)),
                              float(np.nextafter(p[j], 1.0))):
                    rej_f = frozen.fixed_sequence_by_family(p, families, level)
                    assert fixed_sequence_by_family(p, families, level) == rej_f
                    assert smallest(rej_f, size, vectors) == frozen.smallest(rej_f, size, vectors)
                    trials += 1
    assert trials >= 300


@pytest.mark.parametrize("shape", [(5, 13), (3, 4, 9), (2, 3, 4, 6)])
def test_stacked_input_matches_the_frozen_function(frozen, shape):
    rng = np.random.default_rng(shape[-1])
    y = rng.random(shape)
    m0 = 0.95 / 1.95
    assert betting_pvalues(y, m0).tobytes() == frozen.betting_pvalues(y, m0).tobytes()


# ------------------------------------------------- (b) method D against run_cell (synthetic)


@pytest.fixture(scope="module")
def study():
    """The frozen orchestration ``e1_1_study`` (importing it has no side effect but sys.path)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(RESEARCH))
        yield importlib.import_module("e1_1_study")


@pytest.mark.parametrize("n", [50, 225])
def test_both_rules_match_the_frozen_run_cell_on_synthetic_data(study, n, monkeypatch):
    """``run_cell`` (scripts/research/e1_1_study.py:80-140) computes C at lines 125-127 and D at
    lines 132-134 for every seed, target and delta; here it runs on synthetic arrays only."""
    rng = np.random.default_rng(31)
    pool, k = 300, 12
    families = ["fa"] * 4 + ["fb"] * 4 + ["fc"] * 4
    size = np.tile([400.0, 300.0, 200.0, 100.0], 3)
    vectors = np.tile([16.0, 16.0, 8.0, 8.0], 3)
    b_pool = rng.choice([1.0, 0.63, 0.5, 0.43, 0.39, 0.0], size=pool,
                        p=[0.5, 0.08, 0.05, 0.03, 0.02, 0.32])
    loss = np.tile([0.0, 0.01, 0.03, 0.08], 3) * np.repeat([1.0, 2.0, 0.5], 4)
    c_pool = np.where(rng.random((k, pool)) < loss[:, None], 0.0, b_pool[None, :])

    def draw(seed):
        idx = np.random.default_rng(seed).integers(0, pool, size=n)
        return c_pool[:, idx], b_pool[idx]

    monkeypatch.setattr(study, "SEEDS", 8)
    prob = study.Problem(families, size, vectors, np.ones(k), np.ones(k))
    out = study.run_cell(prob, draw, n, "vector")
    rule_c, rule_d = FixedSequenceBetting(), BonferroniBetting()
    for target in study.TARGETS:
        for delta in study.DELTAS:
            stored_c = out["sel"][target][f"C@{delta}"]
            stored_d = out["sel"][target][f"D@{delta}"]
            for seed in range(8):
                c, b = draw(seed)
                res_c = rule_c.decide(c, b, families, size, vectors, target, delta)
                res_d = rule_d.decide(c, b, families, size, vectors, target, delta)
                assert (-1 if res_c.selected is None else res_c.selected) == stored_c[seed]
                assert (-1 if res_d.selected is None else res_d.selected) == stored_d[seed]


# ------------------------------------------------------------------ (c) golden selections

STORES = {"real": ["colpali_docvqa", "colpali_infovqa", "colqwen2_docvqa", "colqwen2_infovqa"],
          "confirm": ["colqwen25_docvqa", "colqwen25_infovqa", "scifact"]}
GOLDEN_NS = (50, 100, 225)  # scripts/research/e1_1_study.py:52
GOLDEN_TARGETS = (0.99, 0.97, 0.95)
GOLDEN_DELTAS = (0.05, 0.01)
GOLDEN_SEEDS = 2000
CHUNK = 200
RULE_SEEDS = (0, 1999)  # seeds also run through the rule classes and select()


def full_parity() -> bool:
    return os.environ.get("OPTIVISION_FULL_PARITY") == "1"


def golden_seeds() -> np.ndarray:
    return np.arange(GOLDEN_SEEDS) if full_parity() else np.arange(0, GOLDEN_SEEDS, 5)


def load_golden_store(where: str, name: str):
    """Per-query arrays, candidate metadata and the stored selections of one store, or skip."""
    npz = E1 / "store" / f"{name}.npz"
    side = E1 / "store" / f"{name}.json"
    sel = E1 / where / f"{name}_selections.npz"
    for path in (npz, side, sel):
        if not path.is_file():
            pytest.skip(f"{path.relative_to(ROOT)} is absent")
        if path.read_bytes()[:40].startswith(b"version https://git-lfs"):
            pytest.skip(f"{path.relative_to(ROOT)} is a Git LFS pointer, not the data")
    with np.load(npz, allow_pickle=False) as z:
        base, cand = z["base_labels"], z["cand_labels"]
    meta = json.loads(side.read_text(encoding="utf-8"))
    return base, cand, meta, sel


def golden_cases():
    return [pytest.param(where, name, id=name) for where, names in STORES.items()
            for name in names]


@pytest.mark.parametrize(("where", "name"), golden_cases())
def test_stored_selections_of_method_c_and_d_are_reproduced_exactly(where, name):
    base, cand, meta, sel_path = load_golden_store(where, name)
    candidates = meta["candidates"]
    families = [c["family"] for c in candidates]
    size = np.array([c["bytes_per_doc"] for c in candidates])
    vectors = np.array([c["vectors_per_doc"] for c in candidates])
    n_fam, n_cand, pool = len(dict.fromkeys(families)), len(families), base.shape[0]
    seeds = golden_seeds()
    compared = 0
    with np.load(sel_path, allow_pickle=False) as stored:
        for n in GOLDEN_NS:  # the draw of scripts/research/e1_1_study.py:215-217
            rows = np.stack([np.random.default_rng(int(s)).integers(0, pool, size=n)
                             for s in seeds])
            for target in GOLDEN_TARGETS:
                m0 = target / (1.0 + target)
                mine = {(m, d): np.full(seeds.size, -1, np.int16)
                        for m in "CD" for d in GOLDEN_DELTAS}
                for lo in range(0, seeds.size, CHUNK):  # whole seeds stay together
                    r = rows[lo:lo + CHUNK]
                    pq = np.ascontiguousarray(cand[:, r].transpose(1, 0, 2))  # (seeds, K, n)
                    p = betting_pvalues(to_y(pq, base[r][:, None, :], target), m0)
                    for d in GOLDEN_DELTAS:
                        for j in range(r.shape[0]):
                            k = smallest(fixed_sequence_by_family(p[j], families, d / n_fam),
                                         size, vectors)
                            mine["C", d][lo + j] = -1 if k is None else k
                            rej = [int(q) for q in np.flatnonzero(p[j] <= d / n_cand)]
                            k = smallest(rej, size, vectors)
                            mine["D", d][lo + j] = -1 if k is None else k
                for (method, d), got in mine.items():
                    want = stored[f"n{n}_T{target}_{method}@{d}"][seeds]
                    assert np.array_equal(got, want), (name, n, target, method, d)
                    compared += got.size
    assert compared == seeds.size * len(GOLDEN_NS) * len(GOLDEN_TARGETS) * len(GOLDEN_DELTAS) * 2


@pytest.mark.parametrize(("where", "name"), golden_cases())
def test_stored_selections_are_reproduced_through_the_rules_and_the_engine(where, name):
    """The same stored selections, on a few seeds, through ``decide`` and ``select``."""
    base, cand, meta, sel_path = load_golden_store(where, name)
    candidates = meta["candidates"]
    families = [c["family"] for c in candidates]
    size = np.array([c["bytes_per_doc"] for c in candidates])
    vectors = np.array([c["vectors_per_doc"] for c in candidates])
    space = CandidateSpace("golden", tuple(
        CandidateSpec(c["position"], c["family"], c["pipeline"]) for c in candidates))
    assert space.identity == REGISTERED_DEFAULT_SPACE_SHA256
    pool = base.shape[0]
    rules = {"C": FixedSequenceBetting(), "D": BonferroniBetting()}
    compared = 0
    with np.load(sel_path, allow_pickle=False) as stored:
        for n in GOLDEN_NS:
            for seed in RULE_SEEDS:
                idx = np.random.default_rng(seed).integers(0, pool, size=n)
                c, b = cand[:, idx], base[idx]
                evidence = Evidence(meta["metric"], "labels", b, c,
                                    tuple(f"q{i}" for i in idx), size, vectors, space.identity)
                for target in GOLDEN_TARGETS:
                    for d in GOLDEN_DELTAS:
                        for method, rule in rules.items():
                            want = int(stored[f"n{n}_T{target}_{method}@{d}"][seed])
                            res = rule.decide(c, b, families, size, vectors, target, d)
                            assert (-1 if res.selected is None else res.selected) == want
                            decision = select(space, evidence, target=target, error=d, rule=rule)
                            got = -1 if decision.selected is None else decision.selected.position
                            assert got == want
                            assert (decision.outcome is Outcome.SELECT) == (want >= 0)
                            compared += 2
    assert compared == len(GOLDEN_NS) * len(RULE_SEEDS) * len(GOLDEN_TARGETS) \
        * len(GOLDEN_DELTAS) * 2 * 2
