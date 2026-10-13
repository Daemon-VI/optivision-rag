"""The package copy of Method C's arithmetic, against the specification's numerical vectors.

Source: ``docs/package/METHOD_C_SPEC.md`` section 9.2 (TV0-TV11 and the type-I check),
section 1.5 (validation) and section 2.4 (edge cases). The S0 metadata the S0-based vectors
need is read from a sidecar JSON next to the registered stores (candidate metadata only; no
per-query value); those cases are skipped when the file is absent.

Tolerance: p-values ``rtol=1e-12, atol=0``; rejected lists, selections and booleans exactly.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from optivision.core.selection.statistics import (
    BET_CAP,
    DtypeError,
    EmptySampleError,
    ErrorLevelError,
    EvidenceError,
    NonFiniteError,
    RangeError,
    ShapeError,
    TargetError,
    ZeroBaselineError,
    betting_pvalues,
    fixed_sequence_by_family,
    smallest,
    to_y,
    validate_evidence_arrays,
)

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "reports" / "research" / "E1" / "store" / "colpali_docvqa.json"


def m0_of(target):
    return target / (1.0 + target)


def close(a, b):
    return bool(np.isclose(a, b, rtol=1e-12, atol=0.0))


def method_c(c, b, families, bytes_, vectors, target, delta):
    """The specification's reference composition, written with the package's functions."""
    c = np.asarray(c, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n_families = len(dict.fromkeys(families))
    p = betting_pvalues(to_y(c, b[None, :], target), m0_of(target))
    rejected = fixed_sequence_by_family(p, families, delta / n_families)
    return smallest(rejected, bytes_, vectors), p, rejected


def load_s0():
    """S0 candidate metadata from the sidecar, or skip the calling test."""
    if not SIDECAR.is_file():
        pytest.skip(f"{SIDECAR} is absent")
    meta = json.loads(SIDECAR.read_text(encoding="utf-8"))["candidates"]
    return SimpleNamespace(
        meta=meta,
        families=[c["family"] for c in meta],
        bytes=np.array([c["bytes_per_doc"] for c in meta]),
        vectors=np.array([c["vectors_per_doc"] for c in meta]),
        k=len(meta),
        n_families=len(dict.fromkeys(c["family"] for c in meta)),
    )


def tv9_problem(s0, n=450):
    """The deterministic end-to-end problem of TV9: no RNG, S0 metadata, ``n`` queries."""
    depth, seen = [], {}
    for f in s0.families:
        depth.append(seen.get(f, 0))
        seen[f] = seen.get(f, 0) + 1
    chain_ix = [list(dict.fromkeys(s0.families)).index(f) for f in s0.families]
    vals = np.concatenate([1.0 / np.log2(np.arange(2, 7)), [0.0]])
    cyc = np.array([0, 0, 0, 1, 2, 5, 0, 3, 5, 4])
    i = np.arange(n)
    kk = np.arange(s0.k)[:, None]
    b9 = vals[cyc[i % 10]]
    h = ((i[None, :] * 7919 + kk * 104729 + 13) % 1009) / 1009.0
    loss = (0.002 * np.maximum(np.array(depth) - 1, 0) * (1 + 0.5 * np.array(chain_ix)))[:, None]
    c9 = np.where(h < loss, 0.0, b9[None, :])
    return c9, b9


TV9_EXPECT = {  # (T, delta): (selected, certified)
    (0.99, 0.05): (None, []), (0.99, 0.01): (None, []),
    (0.97, 0.05): (28, [0, 1, 2, 3, 7, 8, 14, 15, 20, 21, 27, 28, 33, 34]),
    (0.97, 0.01): (28, [0, 1, 2, 7, 8, 14, 15, 20, 21, 27, 28, 33, 34]),
    (0.95, 0.05): (30, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 20, 21, 22, 27, 28,
                        29, 30, 33, 34, 35]),
    (0.95, 0.01): (35, [0, 1, 2, 3, 4, 7, 8, 9, 10, 14, 15, 16, 20, 21, 27, 28, 29, 33, 34, 35]),
}


# --------------------------------------------------------------------- TV0 - TV6


def test_bet_cap_is_the_design_constant():
    assert BET_CAP == 0.9


def test_tv0_m0_values():
    assert (m0_of(0.99), m0_of(0.97), m0_of(0.95), m0_of(0.90)) == (
        0.49748743718592964, 0.49238578680203043, 0.48717948717948717, 0.4736842105263158)


TV1 = {(0.95, 1): 0.9743918053777209, (0.95, 2): 0.6096967495622448, (0.95, 5): 0.1011749929010796,
       (0.95, 10): 0.0038903901009042755, (0.95, 50): 1.0298567486376658e-14,
       (0.95, 225): 2.287469300795809e-65, (0.99, 1): 0.9949751268780463,
       (0.99, 10): 0.004276708556589193, (0.99, 225): 1.792208764226269e-63,
       (0.97, 5): 0.10395635389076212, (0.90, 5): 0.09445940121038329}


@pytest.mark.parametrize(("target", "n"), sorted(TV1))
def test_tv1_all_ones_anchor_values(target, n):
    assert close(betting_pvalues(np.ones(n), m0_of(target)), TV1[(target, n)])


def test_tv2_no_evidence_gives_p_equal_one_and_empty_sample_gives_one():
    m0 = m0_of(0.95)
    assert betting_pvalues(np.zeros(50), m0) == 1.0
    assert betting_pvalues(np.full(50, m0), m0) == 1.0
    assert betting_pvalues(np.ones(0), m0) == 1.0


TV3_P225 = {0.99: 0.19649653893886854, 0.97: 0.0031521656162632767,
            0.95: 4.601695771539229e-05, 0.90: 7.635302410969988e-10}
TV3_MIN_N = {  # target: smallest n that rejects at (0.05, 0.05/7, 0.01, 0.01/7)
    0.99: (377, 592, 555, 770), 0.97: (125, 196, 183, 254), 0.95: (74, 116, 109, 151),
    0.90: (37, 57, 54, 74)}


@pytest.mark.parametrize("target", sorted(TV3_P225))
def test_tv3_perfect_agreement_p_value_at_225(target):
    y = to_y(np.ones(225), np.ones(225), target)
    assert close(betting_pvalues(y, m0_of(target)), TV3_P225[target])


@pytest.mark.parametrize("target", sorted(TV3_MIN_N))
def test_tv3_smallest_sample_size_that_rejects(target):
    y = to_y(np.ones(800), np.ones(800), target)
    m0 = m0_of(target)
    for level, n_min in zip((0.05, 0.05 / 7, 0.01, 0.01 / 7), TV3_MIN_N[target], strict=True):
        assert betting_pvalues(y[:n_min], m0) <= level < betting_pvalues(y[:n_min - 1], m0)


def test_tv4_the_p_value_depends_on_the_order_of_the_queries():
    m0 = m0_of(0.95)
    assert close(betting_pvalues(np.r_[np.ones(20), np.zeros(20)], m0), 4.968445298450026e-06)
    assert betting_pvalues(np.r_[np.zeros(20), np.ones(20)], m0) == 1.0
    assert close(betting_pvalues(np.tile([1.0, 0.0], 20), m0), 0.9743918053777209)
    assert betting_pvalues(np.tile([0.0, 1.0], 20), m0) == 1.0


def test_tv5_appending_queries_never_increases_p():
    m0 = m0_of(0.95)
    y = np.clip(np.random.default_rng(7).normal(0.62, 0.2, 225), 0, 1)
    prefix = np.array([betting_pvalues(y[:m], m0) for m in range(1, 226)])
    assert (np.diff(prefix) <= 0).all()
    assert prefix[-1] < prefix[0]


def test_tv6_to_y_corner_values():
    corners = tuple(float(to_y(c, b, 0.95)) for c, b in ((1, 1), (0, 1), (1, 0), (0, 0)))
    assert corners == (0.5128205128205129, 0.0, 1.0, 0.48717948717948717)
    assert float(to_y(0, 0, 0.95)) == m0_of(0.95)


def test_tv6_to_y_casts_to_float64_before_any_arithmetic():
    y32 = to_y(np.float32([1, 0.5]), np.float32([1, 1]), 0.95)
    assert y32.dtype == np.float64
    assert np.array_equal(y32, to_y(np.array([1.0, 0.5]), np.array([1.0, 1.0]), 0.95))
    # Python lists are accepted (the frozen function raises TypeError on them)
    assert np.array_equal(to_y([1.0, 0.5], [1.0, 1.0], 0.95), y32)


def test_to_y_keeps_the_frozen_operation_order():
    # (c - T*b + T) / (1 + T) is not bit-identical to c/(1+T) when b = 1: do not simplify
    c = np.array([0.37, 0.6180339887, 0.9])
    expected = (c - 0.95 * 1.0 + 0.95) / (1.0 + 0.95)
    assert np.array_equal(to_y(c, np.ones(3), 0.95), expected)


def test_betting_pvalues_stack_equals_row_by_row():
    rng = np.random.default_rng(11)
    y = rng.random((3, 5, 40))
    m0 = m0_of(0.97)
    stacked = betting_pvalues(y, m0)
    assert stacked.shape == (3, 5)
    for i in range(3):
        for j in range(5):
            assert close(stacked[i, j], betting_pvalues(y[i, j], m0))


@pytest.mark.parametrize("seed", range(6))
def test_p_values_lie_in_zero_one_and_are_finite(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([1, 2, 7, 50, 450]))
    target = float(rng.choice([0.5, 0.9, 0.95, 0.99, 1.0]))
    shape = (6, n)
    kinds = [rng.random(shape), np.ones(shape), np.zeros(shape),
             np.where(rng.random(shape) < 0.3, 1.0, 0.0), np.full(shape, m0_of(target))]
    for y in kinds:
        p = betting_pvalues(y, m0_of(target))
        assert np.isfinite(p).all()
        assert ((p > 0.0) & (p <= 1.0)).all()


# ----------------------------------------------------- TV7 - TV11 (S0 metadata needed)


def test_s0_metadata_has_forty_candidates_in_seven_chains():
    s0 = load_s0()
    assert (s0.k, s0.n_families) == (40, 7)


def test_tv7_decisions_from_hand_made_p_vectors():
    s0 = load_s0()
    level = 0.05 / s0.n_families
    assert level == 0.0071428571428571435

    def decide(over, default, delta=0.05):
        p = np.full(s0.k, default)
        for k, v in over.items():
            p[k] = v
        rej = fixed_sequence_by_family(p, s0.families, delta / s0.n_families)
        return smallest(rej, s0.bytes, s0.vectors), rej

    assert decide({}, 0.0)[0] == 39 and len(decide({}, 0.0)[1]) == 40
    assert decide({38: 1.0}, 0.0)[0] == 32 and len(decide({38: 1.0}, 0.0)[1]) == 38
    assert decide({1: 0.0, 7: 0.0}, 1.0) == (1, [1, 7])
    assert decide({14: 0.0, 20: 0.0}, 1.0) == (14, [14, 20])
    assert decide({27: 0.0, 33: 0.0}, 1.0) == (27, [27, 33])
    assert decide({0: 0.0}, 1.0) == (0, [0])
    assert decide({2: 0.0}, 1.0) == (None, [])
    assert decide({}, 1.0) == (None, [])
    assert decide({0: level}, 1.0) == (0, [0])
    assert decide({0: float(np.nextafter(level, 1.0))}, 1.0) == (None, [])
    assert decide({27: 0.0, 28: 0.0, 33: 0.0, 34: 0.0, 35: 0.0}, 1.0) == (
        35, [27, 28, 33, 34, 35])


def test_tv8_the_level_counts_distinct_labels_of_the_list_passed():
    p8 = np.ones(8)
    p8[0] = 0.05 / 7
    assert fixed_sequence_by_family(p8, [f"f{j}" for j in range(8)], 0.05 / 8) == []
    assert fixed_sequence_by_family(p8[:7], [f"f{j}" for j in range(7)], 0.05 / 7) == [0]


def test_tv9_end_to_end_problem_on_s0_metadata():
    s0 = load_s0()
    c9, b9 = tv9_problem(s0)
    pchk = {0.95: {0: 6.069652166871767e-06, 3: 2.5280520281851414e-05, 4: 0.00019715526301319086,
                   5: 0.003097429218431128, 6: 0.0031039904927338195, 28: 6.069652166871767e-06,
                   30: 0.005487977456782755, 35: 0.0006658290732972056, 39: 0.9719748201470284},
            0.97: {0: 0.000958527981069801, 3: 0.0037845824328897413, 4: 0.026585268871457415,
                   9: 0.038583562515132716, 10: 0.0031892869322677647, 30: 0.214876440200214,
                   39: 1.0},
            0.99: {0: 0.13320105693579673, 3: 0.5451647971928198}}
    for (target, delta), (sel_e, rej_e) in TV9_EXPECT.items():
        sel, p, rej = method_c(c9, b9, s0.families, s0.bytes, s0.vectors, target, delta)
        assert (sel, rej) == (sel_e, rej_e), (target, delta)
        for k, v in pchk[target].items():
            assert close(p[k], v), (target, k)
    # p[10] <= level at T = 0.97 but position 9 stops chain 3, so position 10 is unreachable
    assert 10 not in TV9_EXPECT[(0.97, 0.05)][1]
    for target in (0.99, 0.97, 0.95):  # certified set at 0.01 is a subset of the set at 0.05
        assert set(TV9_EXPECT[(target, 0.01)][1]) <= set(TV9_EXPECT[(target, 0.05)][1])


def test_tv9_p_vector_fingerprints_when_the_platform_reproduces_the_bits():
    s0 = load_s0()
    c9, b9 = tv9_problem(s0)
    fingerprints = {
        0.99: "59f3788406a129bbff850ce2ea3f40342958913f9377b000cf30d7f6590952ae",
        0.97: "676cced61cbb56ce2f40f2fe47ad021e37eb719254f2c72ae63c816378e410ca",
        0.95: "4f4b03e005c94dd94ebff0537fa85086a970ba6bee3ce574c71e2325c8da6674"}
    match = {t: hashlib.sha256(method_c(c9, b9, s0.families, s0.bytes, s0.vectors, t,
                                        0.05)[1].astype("<f8").tobytes()).hexdigest() == fp
             for t, fp in fingerprints.items()}
    if not all(match.values()):
        pytest.skip("bit patterns differ on this platform/numpy build (decisions are still "
                    "compared exactly by the other tests)")


def test_tv10_structure_properties():
    s0 = load_s0()
    c9, b9 = tv9_problem(s0)
    sel0, p0, rej0 = method_c(c9, b9, s0.families, s0.bytes, s0.vectors, 0.95, 0.05)
    # candidates do not interact: permuting rows permutes p bitwise
    perm = np.random.default_rng(5).permutation(s0.k)
    assert np.array_equal(betting_pvalues(to_y(c9[perm], b9[None, :], 0.95), m0_of(0.95)), p0[perm])
    # duplicated pipelines have identical p
    assert all(p0[a] == p0[b] for a, b in ((1, 7), (14, 20), (27, 33)))
    # interleaving chains (within-chain order kept) gives the same certified set
    queues = {f: [k for k in range(s0.k) if s0.families[k] == f]
              for f in dict.fromkeys(s0.families)}
    order, rr = [], np.random.default_rng(3)
    while any(queues.values()):
        f = rr.choice([f for f, q in queues.items() if q])
        order.append(queues[f].pop(0))
    order = np.array(order)
    rej_i = fixed_sequence_by_family(p0[order], [s0.families[j] for j in order], 0.05 / 7)
    assert sorted(order[rej_i]) == rej0
    # a bijective relabelling of the families changes nothing
    relabel = {f: f"zz{j}" for j, f in enumerate(dict.fromkeys(s0.families))}
    assert method_c(c9, b9, [relabel[f] for f in s0.families], s0.bytes, s0.vectors,
                    0.95, 0.05)[0] == sel0
    # reversing each chain changes the certified set: order matters for power
    rev = np.array([k for f in dict.fromkeys(s0.families)
                    for k in reversed([j for j in range(s0.k) if s0.families[j] == f])])
    rej_r = fixed_sequence_by_family(p0[rev], [s0.families[j] for j in rev], 0.05 / 7)
    assert sorted(rev[rej_r]) != rej0


def test_tv11_nested_samples_never_increase_the_selected_size():
    s0 = load_s0()
    c9, b9 = tv9_problem(s0)
    last = np.inf
    for nn in (50, 100, 150, 225, 300, 450):
        selected, _, _ = method_c(c9[:, :nn], b9[:nn], s0.families, s0.bytes, s0.vectors,
                                  0.95, 0.05)
        current = np.inf if selected is None else s0.bytes[selected]
        assert current <= last
        last = current


# ---------------------------------------------------------------------- type-I error


def test_type_i_error_at_the_null_boundary_20000_replicates():
    reps, levels = 20000, (0.05, 0.05 / 7, 0.01)
    target = 0.95
    m0 = m0_of(target)

    def bound(lv):
        return lv + 4 * (lv * (1 - lv) / reps) ** 0.5

    for n in (50, 225):  # two-point null with P(Y = 1) = m0
        y = (np.random.default_rng(2026).random((reps, n)) < m0).astype(np.float64)
        rate = [float((betting_pvalues(y, m0) <= lv).mean()) for lv in levels]
        assert all(r <= bound(lv) for r, lv in zip(rate, levels, strict=True)), (n, rate)
    vals = np.r_[1.0, 1 / np.log2([3, 4, 5, 6]), 0.0]
    pr = np.array([0.50, 0.08, 0.05, 0.03, 0.02, 0.32])
    for target, n in ((0.95, 50), (0.95, 225), (0.97, 50), (0.97, 225)):
        g = np.random.default_rng(99)  # rare total losses with retention equal to the target
        bq = vals[g.choice(6, size=(reps, n), p=pr)]
        cq = np.where(g.random((reps, n)) < 1 - target, 0.0, bq)
        p = betting_pvalues(to_y(cq, bq, target), m0_of(target))
        rate = [float((p <= lv).mean()) for lv in levels]
        assert all(r <= bound(lv) for r, lv in zip(rate, levels, strict=True)), (target, n, rate)


# -------------------------------------------------------------- the pure functions


def test_fixed_sequence_stops_a_chain_at_its_earliest_non_rejection_only():
    fam = ["a", "a", "a", "b", "b"]
    assert fixed_sequence_by_family(np.array([0.0, 0.5, 0.0, 0.0, 0.0]), fam, 0.1) == [0, 3, 4]
    # a NaN p-value is not <= level, so it stops the chain (the validation layer prevents it)
    assert fixed_sequence_by_family(np.array([0.0, np.nan, 0.0, 0.0, 0.0]), fam, 0.1) == [0, 3, 4]
    assert fixed_sequence_by_family(np.array([0.1, 0.1]), ["a", "a"], 0.1) == [0, 1]  # inclusive


def test_smallest_uses_exact_size_then_vectors_then_position():
    sizes = np.array([10.0, 5.0, 5.0, 5.0, 7.0])
    vectors = np.array([4.0, 3.0, 2.0, 2.0, 1.0])
    assert smallest([0, 1, 2, 3, 4], sizes, vectors) == 2  # 5 bytes, 2 vectors, earlier position
    assert smallest([0, 4], sizes, vectors) == 4
    assert smallest([], sizes, vectors) is None
    near = np.array([1.0, np.nextafter(1.0, 0.0)])
    assert smallest([0, 1], near, np.array([1.0, 9.0])) == 1  # no tolerance on sizes


# ----------------------------------------------------------------------- validation


def valid_args(k=3, n=5, seed=0):
    rng = np.random.default_rng(seed)
    b = 0.2 + 0.8 * rng.random(n)
    c = np.clip(b[None, :] - 0.1 * rng.random((k, n)), 0.0, 1.0)
    return {"candidates": c, "baseline": b,
            "families": ["a", "a", "b"][:k] + ["c"] * max(0, k - 3),
            "bytes_per_doc": np.arange(k, 0, -1, dtype=np.float64) * 100,
            "vectors_per_doc": np.full(k, 10.0), "target": 0.95, "error": 0.05}


def call(**over):
    args = valid_args()
    args.update(over)
    return validate_evidence_arrays(**args)


def test_valid_input_is_returned_as_float64_and_python_floats():
    out = call()
    assert out.candidates.dtype == np.float64 and out.baseline.dtype == np.float64
    assert out.bytes_per_doc.dtype == np.float64 and out.vectors_per_doc.dtype == np.float64
    assert out.families == ("a", "a", "b")
    assert type(out.target) is float and type(out.error) is float
    assert (out.target, out.error) == (0.95, 0.05)


def test_validation_never_modifies_its_input():
    args = valid_args()
    before = {k: np.array(v, copy=True) for k, v in args.items() if isinstance(v, np.ndarray)}
    validate_evidence_arrays(**args)
    for k, v in before.items():
        assert np.array_equal(args[k], v)


def test_every_evidence_error_is_a_value_error():
    for cls in (ShapeError, DtypeError, NonFiniteError, RangeError, EmptySampleError,
                ZeroBaselineError, TargetError, ErrorLevelError):
        assert issubclass(cls, EvidenceError) and issubclass(cls, ValueError)


@pytest.mark.parametrize("over", [
    {"candidates": np.ones(5)},                              # 1-D
    {"candidates": np.ones((3, 5, 2))},                      # 3-D
    {"baseline": np.ones((5, 1))},                           # 2-D baseline
    {"baseline": np.ones(4)},                                # n mismatch
    {"families": ["a", "a"]},                                # too few labels
    {"families": ["a", "a", "b", "b"]},                      # too many labels
    {"families": "aab"},                                     # a string, not a sequence
    {"bytes_per_doc": np.ones(2)},
    {"vectors_per_doc": np.ones(4)},
    {"bytes_per_doc": np.ones((3, 1))},
    {"candidates": [[0.1, 0.2], [0.3]]},                     # ragged
])
def test_wrong_shapes_raise_shape_error(over):
    with pytest.raises(ShapeError):
        call(**over)


def test_no_candidates_raises():
    with pytest.raises(ShapeError, match="K >= 1"):
        call(candidates=np.ones((0, 5)), families=[], bytes_per_doc=np.ones(0),
             vectors_per_doc=np.ones(0))


def test_zero_queries_raises_instead_of_returning_p_equal_one():
    with pytest.raises(EmptySampleError):
        call(candidates=np.ones((3, 0)), baseline=np.ones(0))
    assert betting_pvalues(np.ones((3, 0)), 0.5) .tolist() == [1.0, 1.0, 1.0]  # the frozen answer


@pytest.mark.parametrize("name", ["candidates", "baseline", "bytes_per_doc", "vectors_per_doc"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_raise(name, bad):
    args = valid_args()
    arr = np.array(args[name], copy=True)
    arr.flat[-1] = bad
    with pytest.raises(NonFiniteError):
        validate_evidence_arrays(**{**args, name: arr})


@pytest.mark.parametrize("name", ["candidates", "baseline"])
@pytest.mark.parametrize("bad", [-0.1, 1.1, -1e-300, float(np.nextafter(1.0, 2.0)), 2.0])
def test_metric_values_outside_zero_one_raise_and_are_never_clipped(name, bad):
    args = valid_args()
    arr = np.array(args[name], copy=True)
    arr.flat[0] = bad
    with pytest.raises(RangeError):
        validate_evidence_arrays(**{**args, name: arr})


def test_metric_values_exactly_zero_and_one_are_accepted():
    out = call(candidates=np.array([[0.0, 1.0, 0.5, 0.0, 1.0]] * 3),
               baseline=np.array([1.0, 1.0, 0.0, 0.5, 1.0]))
    assert out.candidates.min() == 0.0 and out.candidates.max() == 1.0


@pytest.mark.parametrize("name", ["bytes_per_doc", "vectors_per_doc"])
def test_negative_sizes_raise(name):
    with pytest.raises(RangeError):
        call(**{name: np.array([1.0, -1.0, 1.0])})


def test_zero_mean_baseline_raises_and_a_single_zero_does_not():
    with pytest.raises(ZeroBaselineError):
        call(baseline=np.zeros(5), candidates=np.zeros((3, 5)))
    out = call(baseline=np.array([0.0, 0.0, 0.0, 0.0, 0.3]))
    assert out.baseline[-1] == 0.3


@pytest.mark.parametrize("target", [0.0, -0.5, 1.0000001, 2, np.nan, np.inf, True, "0.9", None,
                                    np.array([0.9]), [0.9, 0.95]])
def test_target_outside_zero_one_or_not_one_real_number_raises(target):
    with pytest.raises(TargetError):
        call(target=target)


@pytest.mark.parametrize("target", [1.0, 1, 1e-12, np.float64(0.5), np.float32(0.5)])
def test_target_inside_zero_one_is_accepted(target):
    assert call(target=target).target == float(target)


def test_float32_target_is_taken_as_given_not_rounded_to_a_nicer_value():
    assert call(target=np.float32(0.95)).target == float(np.float32(0.95))
    assert call(target=np.float32(0.95)).target != 0.95


@pytest.mark.parametrize("error", [0.0, 1.0, 1.5, -0.1, np.nan, np.inf, True, "0.05", None,
                                   [0.05, 0.01]])
def test_error_outside_open_unit_interval_or_not_one_real_number_raises(error):
    with pytest.raises(ErrorLevelError):
        call(error=error)


@pytest.mark.parametrize("label", ["", None, 3, ("a",)])
def test_family_labels_must_be_non_empty_strings(label):
    with pytest.raises(DtypeError):
        call(families=["a", label, "b"])


def test_float32_float16_and_integer_inputs_are_widened_exactly_before_any_arithmetic():
    args = valid_args()
    for dtype in (np.float32, np.float16):
        c, b = args["candidates"].astype(dtype), args["baseline"].astype(dtype)
        out = validate_evidence_arrays(**{**args, "candidates": c, "baseline": b})
        assert out.candidates.dtype == np.float64
        assert np.array_equal(out.candidates, c.astype(np.float64))  # exact widening
        assert np.array_equal(out.baseline, b.astype(np.float64))
    ints = validate_evidence_arrays(**{**args, "candidates": np.ones((3, 5), dtype=np.int64),
                                       "baseline": np.ones(5, dtype=np.int8)})
    assert ints.candidates.dtype == np.float64 and ints.baseline.dtype == np.float64
    with pytest.raises(RangeError):  # an integer outside [0, 1] is still refused
        call(candidates=np.full((3, 5), 2, dtype=np.int64))


def test_float32_input_gives_the_same_p_values_as_its_exact_float64_widening():
    rng = np.random.default_rng(4)
    b = rng.random(60).astype(np.float32)
    c = (b * 0.97).astype(np.float32)
    m0 = m0_of(0.95)
    p32 = betting_pvalues(to_y(c, b, 0.95), m0)
    p64 = betting_pvalues(to_y(c.astype(np.float64), b.astype(np.float64), 0.95), m0)
    assert p32 == p64 and to_y(c, b, 0.95).dtype == np.float64


@pytest.mark.parametrize("bad", [np.array([True, False, True]), np.array([1 + 0j, 0j, 1j]),
                                 np.array(["0.1", "0.2", "0.3"]), np.array([None, 1, 2], dtype=object)])
def test_non_numeric_dtypes_raise_a_dtype_error_that_is_also_a_type_error(bad):
    with pytest.raises(DtypeError) as info:
        call(bytes_per_doc=bad)
    assert isinstance(info.value, TypeError) and isinstance(info.value, ValueError)


def test_floats_wider_than_64_bits_are_refused_when_the_platform_has_them():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("longdouble is 64-bit on this platform")
    with pytest.raises(DtypeError):
        call(candidates=np.ones((3, 5), dtype=np.longdouble))


def test_validation_emits_no_warnings_on_valid_input():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        call()


@pytest.mark.parametrize("target", [5e-324, 1e-300, 1e-12, 0.5, 1.0])
def test_extreme_targets_in_the_valid_domain_give_finite_p_values_without_warnings(target):
    args = valid_args()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = validate_evidence_arrays(**{**args, "target": target})
        p = betting_pvalues(to_y(out.candidates, out.baseline[None, :], out.target),
                            out.target / (1.0 + out.target))
    assert np.isfinite(p).all() and ((p > 0.0) & (p <= 1.0)).all()
