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
    assumptions_of,
    betting_pvalues,
    builtin_rule_from_record,
    is_builtin_rule,
    rule_record,
    to_y,
)
from optivision.core.selection.rules import DECLARED_PREFIX
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


def test_assumptions_are_the_six_of_the_statement_and_the_four_conditions_of_use():
    for rule in (FS, BON):
        a = rule.assumptions()
        assert isinstance(a, tuple) and len(a) == 10
        assert [s.split(":")[0] for s in a] == ["A1", "A2", "A3", "A4", "A5", "A6",
                                                "C1", "C2", "C3", "C4"]
        assert all(isinstance(s, str) and s.endswith(".") for s in a)
    assert "independent draws" in FS.assumptions()[0]
    assert "order inside each chain" in FS.assumptions()[2]
    assert "No order inside a family is needed" in BON.assumptions()[2]
    assert "fixed before their outcomes are seen" in FS.assumptions()[4]


def test_the_strengthened_assumptions_say_what_the_reviews_asked_them_to_say():
    a1, a3, a5 = FS.assumptions()[0], FS.assumptions()[2], FS.assumptions()[4]
    # A1: repeated queries, one session or author, and a sample sorted or filtered by difficulty
    assert "distribution of the queries the deployed index will serve" in a1
    assert "Repeated or near-duplicate queries" in a1 and "one session or one author" in a1
    assert "sorted or filtered by anything related to difficulty" in a1
    assert "are not independent draws" in a1
    # A3: nothing fitted may use the calibration queries (the library fits on documents only)
    assert "no fitted stage uses the calibration queries" in a3
    assert "stages are fitted on documents only" in a3
    assert "no fitted stage uses the calibration queries" in BON.assumptions()[2]
    # A5: fixed before the outcomes AND unrelated to them
    assert "and is unrelated to them" in a5
    assert "arrival order of independent draws" in a5 and "random permutation" in a5
    assert "sorted by any query attribute" in a5 and "picked among several orders" in a5
    # the unchanged ones stay as they were, so that A1-A6 stay traceable to the registration
    assert FS.assumptions()[1].startswith("A2: the corpus is fixed")
    assert FS.assumptions()[3].startswith("A4: the per-query metric lies in [0, 1]")
    assert FS.assumptions()[5].startswith("A6: the reference is taken as given")
    assert FS.assumptions()[:2] == BON.assumptions()[:2]
    assert FS.assumptions()[3:] == BON.assumptions()[3:]


def test_the_conditions_of_use_are_the_ones_the_research_protocol_satisfied_by_construction():
    c1, c2, c3, c4 = FS.assumptions()[6:]
    assert "one look" in c1 and "one target, one error level, one rule, one metric" in c1.lower()
    assert "one reference and one candidate list" in c1 and "run once on this sample" in c1
    assert "other choices on the same queries is outside the statement" in c1
    assert "baseline and every candidate are scored on the same queries" in c2
    assert "same metric and the same reference" in c2
    assert "fixed without regard to the candidates' results" in c3
    assert "the index that is deployed is the one that was evaluated" in c4
    assert "same corpus" in c4 and "same fitted state" in c4 and "same scoring procedure" in c4
    # nothing is said about the sample size or about stopping: that is not a condition added here
    text = " ".join(FS.assumptions()).lower()
    assert "optional stopping" not in text and "sample size" not in text


def test_the_statement_is_the_joint_statement_with_its_disclaimers_in_the_stored_text():
    assert ("P(a compressed candidate is selected and its retention under Q is below T) "
            "<= delta") in STATEMENT
    assert "over the draw of the calibration queries" in STATEMENT
    assert "every sample size" in STATEMENT
    assert "assumptions A1-A6 and conditions C1-C4" in STATEMENT
    assert "sufficient conditions" in STATEMENT and "nothing is claimed if one" in STATEMENT
    assert "only under" not in STATEMENT  # the statement is not false when an assumption fails
    assert "This is a joint probability" in STATEMENT
    assert "NOT the probability, given that a candidate was selected" in STATEMENT
    assert "not a statement about the retention measured on the calibration queries" in STATEMENT
    assert "not a statement about a finite batch of future queries" in STATEMENT
    assert "not a statement about power" in STATEMENT and "always abstains" in STATEMENT
    assert "an application of published tools" in STATEMENT
    assert "this project's own and has not been externally reviewed" in STATEMENT
    assert "|" not in STATEMENT  # never written as a conditional probability
    lowered = STATEMENT.lower()
    assert "guarantee" not in lowered and "conditional on" not in lowered
    STATEMENT.encode("ascii")


# ------------------------------------------------------------ only the library's own rules


class Declaring:
    """A rule of someone else's that declares itself certifying and tests nothing."""

    name = "declaring"
    certifying = True

    def config(self):
        return {"name": self.name}

    def assumptions(self):
        return ("none: trust me.",)

    def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc, target,
               error):
        k = int(np.argmin(bytes_per_doc))
        return RuleResult(k, (k,), tuple(1.0 for _ in bytes_per_doc), error)


class SneakyFixedSequence(FixedSequenceBetting):
    """A subclass that keeps the built-in name and overrides the decision."""

    def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc, target,
               error):
        k = int(np.argmin(bytes_per_doc))
        return RuleResult(k, (k,), tuple(0.0 for _ in bytes_per_doc), error / 3)


def test_only_the_exact_types_of_the_two_library_rules_are_built_in():
    assert is_builtin_rule(FS) and is_builtin_rule(BON)
    assert is_builtin_rule(FixedSequenceBetting()) and is_builtin_rule(BonferroniBetting())
    sneaky = SneakyFixedSequence()
    assert isinstance(sneaky, FixedSequenceBetting) and sneaky.name == FS.name
    assert not is_builtin_rule(sneaky)  # a subclass is not built in, nor is a matching name
    assert not is_builtin_rule(Declaring()) and not is_builtin_rule("fixed_sequence_betting")
    assert not is_builtin_rule(FixedSequenceBetting)  # the class is not an instance


def test_the_record_of_a_builtin_rule_names_its_class_and_says_it_is_built_in():
    for rule, cls in ((FS, "FixedSequenceBetting"), (BON, "BonferroniBetting")):
        record = rule_record(rule)
        assert record == {"name": rule.name, "class": f"optivision.core.selection.rules.{cls}",
                          "builtin": True, "certifying": True, "declared_certifying": True,
                          "config": rule.config()}
        assert assumptions_of(rule) == rule.assumptions()


def test_the_record_of_any_other_rule_is_never_certifying_whatever_it_declares():
    for rule in (Declaring(), SneakyFixedSequence()):
        record = rule_record(rule)
        assert record["builtin"] is False and record["certifying"] is False
        assert record["declared_certifying"] is True  # the declaration is kept, as the author's
        assert record["class"].endswith(type(rule).__qualname__)
        assert record["class"].startswith(type(rule).__module__)
        assert record["name"] == rule.name and record["config"] == rule.config()
    quiet = Declaring()
    quiet.certifying = False
    assert rule_record(quiet)["declared_certifying"] is False


def test_the_assumptions_of_another_rule_are_labelled_as_its_authors_and_not_examined():
    sentences = assumptions_of(Declaring())
    assert sentences == (DECLARED_PREFIX + "none: trust me.",)
    assert "Declared by the rule's author, not examined by this library" in sentences[0]
    # a subclass of a built-in rule gets the same treatment: its inherited sentences are labelled
    assert all(s.startswith(DECLARED_PREFIX) for s in assumptions_of(SneakyFixedSequence()))
    assert len(assumptions_of(SneakyFixedSequence())) == 10


def test_a_record_resolves_to_a_builtin_rule_only_by_name_class_and_flag_together():
    good = rule_record(FS)
    assert builtin_rule_from_record(good) == FS
    assert builtin_rule_from_record(rule_record(BON)) == BON
    for edit in ({"name": "fixed_sequence_betting_v2"}, {"class": "my.module.MyRule"},
                 {"builtin": False}, {"name": "bonferroni_betting"},
                 {"class": "optivision.core.selection.rules.BonferroniBetting"}):
        assert builtin_rule_from_record({**good, **edit}) is None, edit
    assert builtin_rule_from_record(rule_record(Declaring())) is None
    assert builtin_rule_from_record(rule_record(SneakyFixedSequence())) is None
    assert builtin_rule_from_record({"name": ["unhashable"], "builtin": True}) is None
    assert builtin_rule_from_record({}) is None


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
