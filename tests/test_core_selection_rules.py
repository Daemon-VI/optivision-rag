"""The two selection rules: chain-stop semantics, tie-breaks, level boundary, family count.

The synthetic problems use a baseline of 1 on every query. A *strong* candidate (metric 1 on
every query) has Y = 1/(1+T) and a small p-value once n is large enough; a *weak* candidate
(metric 0 on every query) has Y = 0 and p = 1. That gives exact control over which
hypotheses are rejected without depending on any research data.
"""

from __future__ import annotations

import inspect
import json
import warnings

import numpy as np
import pytest

from optivision.core.identity import canonical_json
from optivision.core.selection import (
    STATEMENT,
    BonferroniBetting,
    FixedSequenceBetting,
    RuleResult,
    SelectionRule,
    betting_pvalues,
    to_y,
)
from optivision.core.selection.statistics import (
    EmptySampleError,
    NonFiniteError,
    RangeError,
    ZeroBaselineError,
)

from .test_core_statistics import TV9_EXPECT, load_s0, tv9_problem

N = 200
T = 0.9
ERR = 0.05
FS = FixedSequenceBetting()
BON = BonferroniBetting()


def rows(*kinds, n=N):
    """Candidate rows: 's' strong (metric 1), 'w' weak (metric 0)."""
    return np.array([np.ones(n) if kind == "s" else np.zeros(n) for kind in kinds])


def decide(rule, kinds, families, size=None, vectors=None, target=T, error=ERR, n=N):
    k = len(kinds)
    size = np.arange(k, 0, -1) * 100.0 if size is None else np.asarray(size, dtype=float)
    vectors = np.full(k, 10.0) if vectors is None else np.asarray(vectors, dtype=float)
    return rule.decide(rows(*kinds, n=n), np.ones(n), families, size, vectors, target, error)


# ------------------------------------------------------------------------- identity


@pytest.mark.parametrize("rule", [FS, BON])
def test_rules_satisfy_the_protocol_and_are_certifying(rule):
    assert isinstance(rule, SelectionRule)
    assert rule.certifying is True
    assert isinstance(rule.name, str) and rule.name
    assert FS.name == "fixed_sequence_betting" and BON.name == "bonferroni_betting"


@pytest.mark.parametrize("rule", [FS, BON])
def test_decide_takes_arrays_only_and_no_provenance(rule):
    params = list(inspect.signature(rule.decide).parameters)
    assert params == ["candidates", "baseline", "families", "bytes_per_doc", "vectors_per_doc",
                      "target", "error"]


@pytest.mark.parametrize("rule", [FS, BON])
def test_config_is_json_native_and_stable(rule):
    config = rule.config()
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    assert canonical_json(config) == canonical_json(rule.config())
    assert config is not rule.config()  # a fresh copy each call
    assert config["bet_cap"] == 0.9 and config["prior_mean"] == 0.5
    assert config["prior_variance"] == 0.25


def test_fixed_sequence_config_extends_the_research_config():
    config = FS.config()
    # the research's method_c_config() (METHOD_C_SPEC.md section 7), values unchanged
    assert config["hypothesis"] == "H0_k: E[c_k - T b] <= 0"
    assert config["statistic"] == "Y = (c - T b + T) / (1 + T); m0 = T / (1 + T)"
    assert config["p_value"] == "exact betting p-value, aGRAPA bets"
    assert config["multiplicity"] == ("fixed-sequence testing inside each chain, in the chain's "
                                      "own order")
    assert config["level"] == "delta / number of chains"
    assert config["queries_used"] == "all n; no hold-out split"
    assert "ties by fewer vectors, then earlier position" in config["choice"]
    # the fields that dict left implicit
    for key in ("prior_mean", "prior_variance", "bet", "running_maximum", "comparison",
                "level_rule"):
        assert key in config
    assert "clip(" in config["bet"] and "bet_cap / m0" in config["bet"]
    assert "starting wealth 1" in config["running_maximum"]
    assert "inclusive" in config["comparison"]
    assert "distinct family labels" in config["level_rule"]
    # the number of chains is counted from the list, not a constant of the configuration
    assert config["chains"] != 7 and not isinstance(config["chains"], int)
    assert "deltas" not in config


def test_bonferroni_config_names_its_own_multiplicity():
    config = BON.config()
    assert "Bonferroni" in config["multiplicity"] and config["level"].startswith("delta / K")
    assert config["bet_cap"] == 0.9 and "chains" not in config


def test_assumptions_are_the_six_of_the_statement():
    for rule in (FS, BON):
        a = rule.assumptions()
        assert isinstance(a, tuple) and len(a) == 6
        assert [s.split(":")[0] for s in a] == ["A1", "A2", "A3", "A4", "A5", "A6"]
        assert all(isinstance(s, str) and s.endswith(".") for s in a)
    assert "independent draws" in FS.assumptions()[0]
    assert "order inside each chain" in FS.assumptions()[2]
    assert "No order inside a family is needed" in BON.assumptions()[2]
    assert "fixed before their outcomes are seen" in FS.assumptions()[4]


def test_statement_is_the_conditional_marginal_statement_without_extra_claims():
    assert ("P(a compressed candidate is selected and its retention under Q is below T) "
            "<= delta") in STATEMENT
    assert "over the draw of the calibration queries" in STATEMENT
    assert "every sample size" in STATEMENT and "A1-A6" in STATEMENT
    assert "no statement about power" in STATEMENT
    assert "not been externally reviewed" in STATEMENT
    assert "|" not in STATEMENT  # never written as a conditional probability
    lowered = STATEMENT.lower()
    assert "guarantee" not in lowered


# ------------------------------------------------------------ fixed-sequence behaviour


def test_a_chain_stops_at_its_earliest_non_rejection_and_other_chains_continue():
    res = decide(FS, "swsss", ["a", "a", "a", "b", "b"])
    assert res.certified == (0, 3, 4)  # position 2 is strong but its chain stopped at 1
    assert res.p_values[2] < res.level  # ... which is why it is not certified
    assert res.p_values[1] == 1.0


def test_a_stopped_chain_ignores_every_later_member_however_small_its_p_value():
    res = decide(FS, "wssss", ["a"] * 5)
    assert res.certified == () and res.selected is None
    assert min(res.p_values[1:]) < res.level


def test_chain_order_decides_what_is_certified():
    forward = decide(FS, "ssw", ["a"] * 3).certified
    reverse = decide(FS, "wss", ["a"] * 3).certified
    assert forward == (0, 1) and reverse == ()


def test_selection_is_the_smallest_stored_size_among_the_certified():
    res = decide(FS, "ssss", ["a", "a", "b", "b"], size=[400, 300, 500, 200])
    assert res.certified == (0, 1, 2, 3) and res.selected == 3
    res = decide(FS, "sw", ["a", "b"], size=[400, 100])  # smaller but not certified
    assert res.selected == 0


def test_size_ties_go_to_fewer_vectors_then_to_the_earlier_position():
    kinds, fam = "sss", ["a", "b", "c"]
    assert decide(FS, kinds, fam, size=[5, 5, 5], vectors=[9, 3, 3]).selected == 1
    assert decide(FS, kinds, fam, size=[5, 5, 5], vectors=[3, 3, 3]).selected == 0
    assert decide(FS, kinds, fam, size=[5, 5, 5], vectors=[3, 4, 3]).selected == 0
    eps = float(np.nextafter(5.0, 0.0))
    assert decide(FS, kinds, fam, size=[5, 5, eps], vectors=[1, 1, 99]).selected == 2  # exact


def test_the_level_boundary_is_inclusive():
    c, b = rows("s"), np.ones(N)
    p = float(betting_pvalues(to_y(c, b[None, :], T), T / (1.0 + T))[0])
    size, vec = np.array([1.0]), np.array([1.0])
    on = FS.decide(c, b, ["a"], size, vec, T, p)  # one family: level == error exactly
    assert on.level == p and on.certified == (0,)
    below = FS.decide(c, b, ["a"], size, vec, T, float(np.nextafter(p, 0.0)))
    assert below.certified == () and below.selected is None
    above = FS.decide(c, b, ["a"], size, vec, T, float(np.nextafter(p, 1.0)))
    assert above.certified == (0,)


@pytest.mark.parametrize("n_families", [1, 2, 3, 7, 8, 11])
def test_the_level_counts_the_distinct_labels_of_the_list_passed(n_families):
    families = [f"f{j}" for j in range(n_families)] * 2  # every label twice
    res = decide(FS, "s" * len(families), families)
    assert res.level == ERR / n_families
    assert res.level == 0.05 / n_families  # true division, as the research computes it


def test_an_extra_family_can_remove_a_certification_by_lowering_the_level():
    c, b = rows("s"), np.ones(N)
    p = float(betting_pvalues(to_y(c, b[None, :], T), T / (1.0 + T))[0])
    err = min(0.9, p * 1.5)  # level err > p with one chain, err / 2 < p with two
    one = FS.decide(c, b, ["a"], [1.0], [1.0], T, err)
    two = FS.decide(np.vstack([c, rows("w")]), b, ["a", "b"], [1.0, 2.0], [1.0, 1.0], T, err)
    assert one.certified == (0,) and two.certified == ()
    assert two.level == err / 2 and two.p_values[0] == p


def test_a_one_member_chain_counts_toward_the_number_of_chains():
    res = decide(FS, "sssss", ["float16", "x", "x", "y", "y"])
    assert res.level == ERR / 3  # not 2: the single-member label counts


def test_nothing_certified_means_no_selection_not_an_error():
    res = decide(FS, "www", ["a", "b", "c"])
    assert res == RuleResult(None, (), res.p_values, ERR / 3)
    assert res.p_values == (1.0, 1.0, 1.0)


def test_duplicated_candidates_are_tested_separately_and_never_merged():
    res = decide(FS, "ss", ["a", "b"], size=[7, 7], vectors=[1, 1])
    assert res.certified == (0, 1) and res.selected == 0
    assert res.p_values[0] == res.p_values[1]


def test_relabelling_the_families_changes_nothing():
    rng = np.random.default_rng(8)
    for _ in range(10):
        k = int(rng.integers(2, 12))
        c = np.clip(1.0 - rng.random((k, 80)) * rng.random((k, 1)) * 0.2, 0, 1)
        b = np.ones(80)
        fam = [f"f{j}" for j in rng.integers(0, 4, size=k)]
        size = rng.choice([1.0, 2.0, 3.0], size=k)
        mapping = {f: f"encoder-{j}-{f[::-1]}" for j, f in enumerate(dict.fromkeys(fam))}
        for rule in (FS, BON):
            base = rule.decide(c, b, fam, size, size, 0.95, 0.05)
            other = rule.decide(c, b, [mapping[f] for f in fam], size, size, 0.95, 0.05)
            assert base == other


def test_permuting_candidates_within_the_constraints_keeps_the_certified_set():
    rng = np.random.default_rng(21)
    for _ in range(15):
        k = int(rng.integers(3, 14))
        fam = [f"f{j}" for j in rng.integers(0, 3, size=k)]
        c = np.clip(1.0 - rng.random((k, 100)) * rng.random((k, 1)) * 0.1, 0, 1)
        b = np.ones(100)
        size = rng.permutation(k).astype(float) + 1.0  # distinct sizes: no tie-break involved
        res = FS.decide(c, b, fam, size, size, 0.95, 0.1)
        queues = {f: [j for j in range(k) if fam[j] == f] for f in dict.fromkeys(fam)}
        order = []
        while any(queues.values()):  # interleave chains; every chain keeps its own order
            f = rng.choice([f for f, q in queues.items() if q])
            order.append(queues[f].pop(0))
        order = np.array(order)
        moved = FS.decide(c[order], b, [fam[j] for j in order], size[order], size[order],
                          0.95, 0.1)
        assert sorted(order[list(moved.certified)]) == sorted(res.certified)
        if res.selected is not None:
            assert order[moved.selected] == res.selected


def test_the_certified_set_at_a_smaller_error_is_a_subset_and_p_values_do_not_move():
    rng = np.random.default_rng(5)
    hits = 0
    for _ in range(30):
        k = int(rng.integers(2, 16))
        fam = [f"f{j}" for j in rng.integers(0, 4, size=k)]
        c = np.clip(1.0 - rng.random((k, 150)) * rng.random((k, 1)) * 0.15, 0, 1)
        b = np.ones(150)
        size = rng.random(k)
        for rule in (FS, BON):
            loose = rule.decide(c, b, fam, size, size, 0.95, 0.05)
            tight = rule.decide(c, b, fam, size, size, 0.95, 0.01)
            assert set(tight.certified) <= set(loose.certified)
            assert tight.p_values == loose.p_values
            hits += len(loose.certified) > len(tight.certified)
    assert hits > 0  # the comparison is not vacuous


def test_more_queries_never_shrink_the_certified_set_or_grow_the_selected_size():
    rng = np.random.default_rng(13)
    for _ in range(10):
        k = 8
        fam = ["a"] * 4 + ["b"] * 4
        size = np.array([8.0, 6.0, 4.0, 2.0, 7.0, 5.0, 3.0, 1.0])
        loss = np.array([0, 0.01, 0.03, 0.06, 0, 0.02, 0.04, 0.08])
        b = rng.choice([1.0, 0.63, 0.5, 0.0], size=450, p=[0.5, 0.1, 0.1, 0.3])
        c = np.where(rng.random((k, 450)) < loss[:, None], 0.0, b[None, :])
        last_set, last_size = set(), np.inf
        for n in (50, 100, 225, 450):
            res = FS.decide(c[:, :n], b[:n], fam, size, size, 0.95, 0.05)
            assert last_set <= set(res.certified)
            current = np.inf if res.selected is None else size[res.selected]
            assert current <= last_size
            last_set, last_size = set(res.certified), current


def test_the_result_depends_on_the_order_of_the_queries():
    n = 40
    b = np.ones(n)
    c_good_late = np.r_[np.zeros(20), np.ones(20)][None, :]
    c_good_early = np.r_[np.ones(20), np.zeros(20)][None, :]
    late = FS.decide(c_good_late, b, ["a"], [1.0], [1.0], 0.5, 0.5)
    early = FS.decide(c_good_early, b, ["a"], [1.0], [1.0], 0.5, 0.5)
    assert late.p_values != early.p_values


def test_the_s0_problem_through_the_rule():
    s0 = load_s0()
    c9, b9 = tv9_problem(s0)
    for (target, delta), (sel, certified) in TV9_EXPECT.items():
        res = FS.decide(c9, b9, s0.families, s0.bytes, s0.vectors, target, delta)
        assert (res.selected, list(res.certified)) == (sel, certified)
        assert res.level == delta / 7


def test_dropping_the_single_member_chain_would_change_the_level_so_it_is_counted():
    s0 = load_s0()
    c9, b9 = tv9_problem(s0)
    with_float16 = FS.decide(c9, b9, s0.families, s0.bytes, s0.vectors, 0.95, 0.05)
    without = FS.decide(c9[1:], b9, s0.families[1:], s0.bytes[1:], s0.vectors[1:], 0.95, 0.05)
    assert with_float16.level == 0.05 / 7 and without.level == 0.05 / 6


# ------------------------------------------------------------------------- Bonferroni


def test_bonferroni_tests_every_candidate_at_delta_over_k_with_no_chain_stop():
    res = decide(BON, "sws", ["a", "a", "a"])
    assert res.level == ERR / 3
    assert res.certified == (0, 2)  # position 2 follows a failure in its chain: still certified
    assert decide(FS, "sws", ["a", "a", "a"]).certified == (0,)


def test_bonferroni_counts_every_candidate_and_ignores_the_labels_entirely():
    kinds = "ssws"
    a = decide(BON, kinds, ["x", "x", "y", "y"])
    b = decide(BON, kinds, ["only"] * 4)
    c = decide(BON, kinds, ["p", "q", "r", "s"])
    assert a == b == c
    assert a.level == ERR / 4  # K = 4 candidates, not the number of labels


def test_bonferroni_selection_uses_the_same_size_key():
    res = decide(BON, "sss", ["a", "b", "c"], size=[5, 5, 5], vectors=[3, 2, 2])
    assert res.certified == (0, 1, 2) and res.selected == 1


def test_the_rules_spend_the_error_level_differently():
    c, b = rows("s", "s"), np.ones(N)
    p = float(betting_pvalues(to_y(c, b[None, :], T), T / (1.0 + T))[0])
    err = 1.5 * p  # p lies between err / 2 and err
    one_chain = ["a", "a"]
    fs = FS.decide(c, b, one_chain, [2.0, 1.0], [1.0, 1.0], T, err)
    bon = BON.decide(c, b, one_chain, [2.0, 1.0], [1.0, 1.0], T, err)
    assert fs.p_values == bon.p_values
    assert fs.level == err and bon.level == err / 2
    assert fs.certified == (0, 1) and bon.certified == ()
    # one family per candidate: F == K, so the two rules coincide
    two_chains = ["a", "b"]
    assert (FS.decide(c, b, two_chains, [2.0, 1.0], [1.0, 1.0], T, err)
            == BON.decide(c, b, two_chains, [2.0, 1.0], [1.0, 1.0], T, err))


def test_there_is_no_float16_or_other_default_when_nothing_is_certified():
    for rule in (FS, BON):
        res = decide(rule, "www", ["float16", "int8", "binary"])
        assert res.selected is None and res.certified == ()


# --------------------------------------------------------------------------- results


def test_rule_result_is_frozen_and_checks_that_the_selection_is_certified():
    res = RuleResult(1, [0, 1], [0.1, 0.2], 0.05)
    assert res.certified == (0, 1) and res.p_values == (0.1, 0.2)
    with pytest.raises(AttributeError):
        res.selected = 0  # type: ignore[misc]
    with pytest.raises(ValueError, match="member of the certified set"):
        RuleResult(2, (0, 1), (0.1, 0.2, 0.3), 0.05)


@pytest.mark.parametrize("rule", [FS, BON])
def test_decide_is_deterministic_random_free_and_does_not_touch_its_input(rule, monkeypatch):
    rng = np.random.default_rng(3)
    c = np.clip(1.0 - rng.random((6, 90)) * 0.1, 0, 1)
    b = np.ones(90)
    fam = list("aabbcc")
    size = rng.random(6)
    snapshot = (c.copy(), b.copy(), size.copy())

    def forbidden(*_a, **_k):
        raise AssertionError("a selection rule must not draw random numbers")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "seed", forbidden)
    before = rule.decide(c, b, fam, size, size, 0.95, 0.05)
    again = rule.decide(c, b, fam, size, size, 0.95, 0.05)
    assert before == again
    assert np.array_equal(c, snapshot[0]) and np.array_equal(b, snapshot[1])
    assert np.array_equal(size, snapshot[2])


@pytest.mark.parametrize("rule", [FS, BON])
def test_float32_input_is_widened_exactly_and_gives_the_float64_result(rule):
    rng = np.random.default_rng(6)
    c = np.clip(1.0 - rng.random((5, 70)) * 0.1, 0, 1).astype(np.float32)
    b = np.ones(70, dtype=np.float32)
    size = np.arange(5, 0, -1, dtype=np.float32)
    fam = list("aabbb")
    low = rule.decide(c, b, fam, size, size, 0.9, 0.05)
    exact = rule.decide(c.astype(np.float64), b.astype(np.float64), fam, size.astype(np.float64),
                        size.astype(np.float64), 0.9, 0.05)
    assert low == exact


@pytest.mark.parametrize("rule", [FS, BON])
def test_invalid_evidence_raises_and_is_never_repaired(rule):
    ok = {"candidates": rows("s", "s"), "baseline": np.ones(N), "families": ["a", "b"],
          "bytes_per_doc": [2.0, 1.0], "vectors_per_doc": [1.0, 1.0], "target": T, "error": ERR}
    rule.decide(**ok)

    def bad(**over):
        return {**ok, **over}

    with pytest.raises(EmptySampleError):
        rule.decide(**bad(candidates=np.ones((2, 0)), baseline=np.ones(0)))
    with pytest.raises(NonFiniteError):
        rule.decide(**bad(baseline=np.r_[np.nan, np.ones(N - 1)]))
    with pytest.raises(RangeError):
        rule.decide(**bad(candidates=rows("s", "s") * 1.5))
    with pytest.raises(ZeroBaselineError):
        rule.decide(**bad(baseline=np.zeros(N)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rule.decide(**ok)


def test_a_recorded_rule_name_can_be_turned_back_into_the_rule():
    from optivision.core.selection import rule_from_name

    for rule in (FS, BON):
        assert rule_from_name(rule.name) == rule
        assert rule_from_name(rule.name).config() == rule.config()
    for bad in ("nope", "", None, ["fixed_sequence_betting"]):
        with pytest.raises(ValueError, match="unknown selection rule"):
            rule_from_name(bad)  # type: ignore[arg-type]
