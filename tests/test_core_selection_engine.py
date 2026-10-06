"""``select``: outcomes, fallback, budget, order, purity, errors, the decision record."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.core.evaluate import Evidence
from optivision.core.identity import array_digest
from optivision.core.selection import (
    BonferroniBetting,
    CandidateSpace,
    CandidateSpec,
    Decision,
    FixedSequenceBetting,
    Outcome,
    QueryOrderError,
    RuleResult,
    SpaceMismatchError,
    select,
)
from optivision.core.selection.statistics import (
    EmptySampleError,
    ErrorLevelError,
    EvidenceError,
    NonFiniteError,
    RangeError,
    TargetError,
    ZeroBaselineError,
)
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    Int4Quantizer,
    Int8Quantizer,
)

from .test_core_statistics import TV9_EXPECT, load_s0, tv9_problem

N = 200
T = 0.9
ERR = 0.05
HAVE_SCIPY = True
try:
    import scipy.cluster.hierarchy  # noqa: F401
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_SCIPY = False


def space4():
    return CandidateSpace.from_families({
        "a": [Pipeline([Float16Quantizer()]), Pipeline([Int8Quantizer("per_vector")])],
        "b": [Pipeline([AdaptiveMerge(radius=0.9, center="mean"), Int4Quantizer(center="mean")]),
              Pipeline([BinaryQuantizer()])],
    }, name="four")


SIZES = np.array([400.0, 300.0, 200.0, 100.0])
VECTORS = np.array([10.0, 10.0, 10.0, 10.0])


def evidence_for(space, kinds, n=N, **over):
    """Baseline 1 everywhere; a strong candidate has metric 1, a weak one metric 0."""
    cand = np.array([np.ones(n) if kind == "s" else np.zeros(n) for kind in kinds])
    args = {"metric": "ndcg@5", "construct": "labels", "baseline": np.ones(n),
            "candidates": cand, "query_ids": tuple(f"q{i}" for i in range(n)),
            "bytes_per_doc": SIZES, "vectors_per_doc": VECTORS,
            "space_identity": space.identity}
    args.update(over)
    return Evidence(**args)


def random_evidence(space, seed=0, n=300):
    rng = np.random.default_rng(seed)
    b = rng.choice([1.0, 0.63, 0.5, 0.0], size=n, p=[0.5, 0.1, 0.1, 0.3])
    loss = np.array([0.0, 0.01, 0.02, 0.05])
    c = np.where(rng.random((4, n)) < loss[:, None], 0.0, b[None, :])
    return Evidence("ndcg@5", "labels", b, c, tuple(f"q{i}" for i in range(n)), SIZES, VECTORS,
                    space.identity)


# ------------------------------------------------------------------------ outcomes


def test_select_returns_the_smallest_certified_candidate_and_records_everything():
    space, rule = space4(), FixedSequenceBetting()
    ev = evidence_for(space, "ssss")
    d = select(space, ev, target=T, error=ERR)
    assert d.outcome is Outcome.SELECT and d.certified is True
    assert d.selected == space[3] and d.selected.position == 3
    assert d.certified_positions == (0, 1, 2, 3)
    assert d.level == ERR / 2 and (d.target, d.error) == (T, ERR)
    assert len(d.p_values) == 4 and all(0.0 < p <= 1.0 for p in d.p_values)
    assert d.n_queries == N and d.metric == "ndcg@5" and d.construct == "labels"
    assert d.space_identity == space.identity and d.evidence_digest == ev.digest()
    assert d.rule == {"name": rule.name, "certifying": True, "config": rule.config()}
    assert d.assumptions == rule.assumptions() and len(d.assumptions) == 6
    assert d.query_order["kind"] == "given"


def test_nothing_certified_is_an_abstention_not_an_error():
    space = space4()
    d = select(space, evidence_for(space, "wwww"), target=T, error=ERR)
    assert d.outcome is Outcome.ABSTAIN
    assert d.selected is None and d.certified is False and d.certified_positions == ()
    assert d.p_values == (1.0, 1.0, 1.0, 1.0)


def test_the_rule_may_abstain_even_when_every_candidate_equals_the_baseline():
    space = space4()
    perfect = evidence_for(space, "ssss", n=225)
    high = select(space, perfect, target=0.99, error=0.05)
    assert high.outcome is Outcome.ABSTAIN  # too little evidence for this target and level
    assert all(abs(p - 0.19649653893886854) < 1e-12 for p in high.p_values)
    assert select(space, perfect, target=0.9, error=0.05).outcome is Outcome.SELECT


def test_an_explicit_fallback_is_returned_but_never_certified():
    space = space4()
    ev = evidence_for(space, "wwww")
    fallback = Pipeline([Float16Quantizer()])
    for given in (fallback, fallback.to_dict()):
        d = select(space, ev, target=T, error=ERR, fallback=given)
        assert d.outcome is Outcome.FALLBACK and d.certified is False
        assert d.certified_positions == ()
        assert d.selected.position == -1 and d.selected.family == "fallback"
        assert d.selected.to_dict()["pipeline"] == fallback.to_dict()
        assert d.selected not in list(space)  # not a member of the space
        assert d.p_values == (1.0,) * 4


def test_a_fallback_is_ignored_when_a_candidate_is_certified():
    space = space4()
    d = select(space, evidence_for(space, "ssss"), target=T, error=ERR,
               fallback=Pipeline([Float16Quantizer()]))
    assert d.outcome is Outcome.SELECT and d.selected.position == 3 and d.certified


def test_there_is_no_default_fallback_even_when_float16_is_in_the_list():
    space = space4()
    assert space[0].to_pipeline().to_dict() == Pipeline([Float16Quantizer()]).to_dict()
    d = select(space, evidence_for(space, "wwww"), target=T, error=ERR)
    assert d.selected is None and d.outcome is Outcome.ABSTAIN


def test_a_malformed_fallback_is_refused():
    space, ev = space4(), evidence_for(space4(), "wwww")
    with pytest.raises(TypeError, match="fallback"):
        select(space, ev, target=T, error=ERR, fallback="float16")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown stage"):
        select(space, ev, target=T, error=ERR, fallback={"stages": [{"stage": "nope"}]})


def test_the_rule_parameter_selects_the_rule_and_is_recorded():
    space = space4()
    ev = evidence_for(space, "swsw")
    fs = select(space, ev, target=T, error=ERR)
    bon = select(space, ev, target=T, error=ERR, rule=BonferroniBetting())
    assert fs.rule["name"] == "fixed_sequence_betting"
    assert bon.rule["name"] == "bonferroni_betting" and bon.level == ERR / 4
    assert fs.certified_positions == (0, 2) and bon.certified_positions == (0, 2)
    assert bon.assumptions == BonferroniBetting().assumptions()


def test_a_certified_set_at_a_smaller_error_is_a_subset():
    space = space4()
    ev = random_evidence(space)
    for rule in (FixedSequenceBetting(), BonferroniBetting()):
        loose = select(space, ev, target=0.97, error=0.05, rule=rule)
        tight = select(space, ev, target=0.97, error=0.01, rule=rule)
        assert set(tight.certified_positions) <= set(loose.certified_positions)
        assert tight.p_values == loose.p_values


class PointRetentionRule:
    """A toy non-certifying rule: every candidate whose mean retention reaches the target."""

    name = "point_retention"
    certifying = False

    def config(self):
        return {"name": self.name}

    def assumptions(self):
        return ("none",)

    def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc, target,
               error):
        retention = candidates.mean(axis=1) / baseline.mean()
        ok = tuple(int(k) for k in np.flatnonzero(retention >= target))
        selected = min(ok, key=lambda k: (bytes_per_doc[k], vectors_per_doc[k], k)) if ok else None
        return RuleResult(selected, ok, tuple(float(r) for r in retention), error)


def test_a_non_certifying_rule_never_produces_a_certified_decision():
    space = space4()
    d = select(space, evidence_for(space, "ssss"), target=T, error=ERR, rule=PointRetentionRule())
    assert d.outcome is Outcome.SELECT and d.selected.position == 3
    assert d.certified is False and d.rule["certifying"] is False


def test_a_rule_that_returns_a_position_outside_the_space_is_refused():
    class Broken(PointRetentionRule):
        def decide(self, *args):
            return RuleResult(9, (9,), (0.0,) * 4, 0.05)

    space = space4()
    with pytest.raises(ValueError, match="outside the candidate space"):
        select(space, evidence_for(space, "ssss"), target=T, error=ERR, rule=Broken())


def test_family_labels_are_grouping_keys_only():
    base = space4()
    relabelled = CandidateSpace.from_families({
        "encoder-x": [p for p in base.pipelines()[:2]],
        "float16": [p for p in base.pipelines()[2:]]})
    assert relabelled.identity != base.identity
    one = select(base, evidence_for(base, "swss"), target=T, error=ERR)
    two = select(relabelled, evidence_for(relabelled, "swss"), target=T, error=ERR)
    assert (one.selected.position, one.certified_positions, one.p_values, one.level) == (
        two.selected.position, two.certified_positions, two.p_values, two.level)


# ----------------------------------------------------------------- order and budget


def test_the_default_order_is_the_given_order_and_is_recorded():
    space = space4()
    d = select(space, random_evidence(space), target=0.95, error=0.05)
    assert d.query_order == {"kind": "given", "seed": None, "budget": None,
                             "n_available": 300, "n_used": 300,
                             "indices_sha256": array_digest(np.arange(300, dtype=np.int64))}


def test_a_budget_takes_a_prefix_of_the_given_order():
    space = space4()
    ev = random_evidence(space)
    for m in (1, 17, 150, 300):
        budgeted = select(space, ev, target=0.95, error=0.05, query_budget=m)
        direct = select(space, ev.subset(np.arange(m)), target=0.95, error=0.05)
        assert budgeted.n_queries == m
        assert (budgeted.p_values, budgeted.certified_positions, budgeted.selected) == (
            direct.p_values, direct.certified_positions, direct.selected)
        assert budgeted.query_order["budget"] == m and budgeted.query_order["n_used"] == m
        assert budgeted.evidence_digest == ev.digest()  # bound to the evidence as given


@pytest.mark.parametrize("budget", [0, -3, 301, 10_000, 2.5, "5", True, 1.0])
def test_a_budget_that_cannot_be_used_raises(budget):
    space = space4()
    with pytest.raises(QueryOrderError):
        select(space, random_evidence(space), target=0.95, error=0.05, query_budget=budget)


def test_the_error_for_a_budget_beyond_the_pool_says_it_never_resamples():
    space = space4()
    with pytest.raises(QueryOrderError, match="never samples with replacement"):
        select(space, random_evidence(space), target=0.95, error=0.05, query_budget=301)
    assert issubclass(QueryOrderError, ValueError)


def test_a_seeded_order_is_a_data_independent_permutation_that_is_recorded():
    space = space4()
    ev = random_evidence(space)
    d = select(space, ev, target=0.95, error=0.05, order_seed=11)
    perm = np.random.default_rng(11).permutation(300)
    assert d.query_order["kind"] == "seeded_permutation" and d.query_order["seed"] == 11
    assert d.query_order["indices_sha256"] == array_digest(perm)
    direct = select(space, ev.subset(perm), target=0.95, error=0.05)
    assert (d.p_values, d.certified_positions, d.selected) == (
        direct.p_values, direct.certified_positions, direct.selected)
    assert d.evidence_digest == ev.digest() != direct.evidence_digest


def test_a_seeded_order_is_reproducible_and_depends_on_the_seed_only():
    space = space4()
    ev = random_evidence(space)
    run_a = select(space, ev, target=0.95, error=0.05, order_seed=5)
    np.random.seed(123)  # the legacy global state plays no part
    run_b = select(space, ev, target=0.95, error=0.05, order_seed=5)
    assert run_a == run_b and run_a.to_dict() == run_b.to_dict()
    other = select(space, ev, target=0.95, error=0.05, order_seed=6)
    assert other.query_order["indices_sha256"] != run_a.query_order["indices_sha256"]
    assert other.p_values != run_a.p_values  # the p-value depends on the order


def test_a_budget_applies_to_the_prefix_of_the_seeded_order():
    space = space4()
    ev = random_evidence(space)
    d = select(space, ev, target=0.95, error=0.05, order_seed=3, query_budget=120)
    prefix = np.random.default_rng(3).permutation(300)[:120]
    assert d.query_order["indices_sha256"] == array_digest(prefix)
    assert d.query_order["budget"] == 120 and d.query_order["n_used"] == 120
    direct = select(space, ev.subset(prefix), target=0.95, error=0.05)
    assert d.p_values == direct.p_values and d.n_queries == 120


@pytest.mark.parametrize("seed", [-1, 1.5, "3", True])
def test_a_bad_order_seed_raises(seed):
    space = space4()
    with pytest.raises(QueryOrderError):
        select(space, random_evidence(space), target=0.95, error=0.05, order_seed=seed)


def test_more_queries_from_one_order_never_increase_the_selected_size():
    space = space4()
    for seed in (None, 1, 2):
        ev = random_evidence(space, seed=7, n=450)
        last, last_set = np.inf, set()
        for budget in (50, 100, 150, 225, 300, 450):
            d = select(space, ev, target=0.95, error=0.05, query_budget=budget, order_seed=seed)
            size = np.inf if d.selected is None else SIZES[d.selected.position]
            assert size <= last and last_set <= set(d.certified_positions)
            last, last_set = size, set(d.certified_positions)


# --------------------------------------------------------------- purity and errors


def test_select_is_pure_it_changes_nothing_and_draws_nothing_without_a_seed(monkeypatch):
    space = space4()
    ev = random_evidence(space)
    digests = (ev.digest(), space.identity)

    def forbidden(*_a, **_k):
        raise AssertionError("no random number may be drawn without order_seed")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    run_a = select(space, ev, target=0.95, error=0.05)
    run_b = select(space, ev, target=0.95, error=0.05)
    assert run_a == run_b and (ev.digest(), space.identity) == digests
    assert not ev.candidates.flags.writeable


def test_evidence_from_another_space_is_refused():
    space = space4()
    ev = evidence_for(space, "ssss", space_identity="0" * 64)
    with pytest.raises(SpaceMismatchError, match="different candidate space"):
        select(space, ev, target=T, error=ERR)
    other = CandidateSpace.from_families({"a": [Pipeline([Float16Quantizer()])]})
    with pytest.raises(SpaceMismatchError):
        select(other, evidence_for(space, "ssss"), target=T, error=ERR)
    assert issubclass(SpaceMismatchError, ValueError)


def test_the_number_of_candidates_must_match_even_if_the_identity_is_claimed():
    space = space4()
    short = Evidence("ndcg@5", "labels", np.ones(N), np.ones((3, N)),
                     tuple(f"q{i}" for i in range(N)), SIZES[:3], VECTORS[:3], space.identity)
    with pytest.raises(SpaceMismatchError, match="candidates"):
        select(space, short, target=T, error=ERR)


def test_the_arguments_are_type_checked():
    space = space4()
    ev = evidence_for(space, "ssss")
    with pytest.raises(TypeError, match="space"):
        select({"a": []}, ev, target=T, error=ERR)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="evidence"):
        select(space, {"candidates": 1}, target=T, error=ERR)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="rule"):
        select(space, ev, target=T, error=ERR, rule="fixed")  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [0.0, -0.1, 1.5, np.nan, [0.9, 0.95], "0.9", None])
def test_exactly_one_valid_target_per_call(target):
    space = space4()
    with pytest.raises(TargetError):
        select(space, evidence_for(space, "ssss"), target=target, error=ERR)


@pytest.mark.parametrize("error", [0.0, 1.0, 2.0, np.nan, [0.05, 0.01], "0.05", None])
def test_exactly_one_valid_error_level_per_call(error):
    space = space4()
    with pytest.raises(ErrorLevelError):
        select(space, evidence_for(space, "ssss"), target=T, error=error)


def test_target_and_error_are_keyword_only():
    space = space4()
    with pytest.raises(TypeError):
        select(space, evidence_for(space, "ssss"), T, ERR)  # type: ignore[misc]


@pytest.mark.parametrize(("over", "error"), [
    ({"candidates": np.full((4, N), 1.5)}, RangeError),
    ({"candidates": np.full((4, N), np.nan)}, NonFiniteError),
    ({"baseline": np.zeros(N)}, ZeroBaselineError),
    ({"bytes_per_doc": np.array([1.0, np.inf, 1.0, 1.0])}, NonFiniteError),
    ({"bytes_per_doc": np.array([1.0, -1.0, 1.0, 1.0])}, RangeError),
])
def test_invalid_values_in_the_evidence_raise_and_are_never_repaired(over, error):
    space = space4()
    ev = evidence_for(space, "ssss", **over)
    with pytest.raises(error):
        select(space, ev, target=T, error=ERR)
    assert issubclass(error, EvidenceError) and issubclass(error, ValueError)


def test_an_evidence_with_no_queries_raises_instead_of_abstaining_silently():
    space = space4()
    ev = evidence_for(space, "ssss", n=0, query_ids=())
    with pytest.raises(EmptySampleError):
        select(space, ev, target=T, error=ERR)


# ------------------------------------------------------------------- the decision


def decisions():
    space = space4()
    return [
        select(space, evidence_for(space, "ssss"), target=T, error=ERR),
        select(space, evidence_for(space, "wwww"), target=T, error=ERR),
        select(space, evidence_for(space, "wwww"), target=T, error=ERR,
               fallback=Pipeline([Int8Quantizer("per_vector")])),
        select(space, random_evidence(space), target=0.95, error=0.05, order_seed=4,
               query_budget=100, rule=BonferroniBetting()),
    ]


@pytest.mark.parametrize("decision", decisions(), ids=["select", "abstain", "fallback", "seeded"])
def test_a_decision_round_trips_through_json_text_exactly(decision):
    text = json.dumps(decision.to_dict(), allow_nan=False)
    back = Decision.from_dict(json.loads(text))
    assert back == decision and hash(back) == hash(decision)
    assert back.to_dict() == decision.to_dict()
    assert back.outcome is decision.outcome and back.p_values == decision.p_values
    assert isinstance(decision.to_dict()["outcome"], str)


def test_to_dict_is_a_fresh_plain_copy():
    d = decisions()[0]
    plain = d.to_dict()
    plain["selected"]["pipeline"]["stages"].clear()
    plain["rule"]["config"]["bet_cap"] = 7
    plain["query_order"]["kind"] = "x"
    assert d.to_dict()["selected"]["pipeline"]["stages"] and d.rule["config"]["bet_cap"] == 0.9
    assert d.query_order["kind"] == "given"


def test_a_decision_is_immutable():
    d = decisions()[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.outcome = Outcome.ABSTAIN  # type: ignore[misc]
    with pytest.raises(TypeError):
        d.rule["name"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        d.query_order["seed"] = 3  # type: ignore[index]
    assert isinstance(d.certified_positions, tuple) and isinstance(d.p_values, tuple)
    assert isinstance(d.assumptions, tuple)


def test_from_dict_refuses_foreign_missing_or_extra_content():
    record = decisions()[0].to_dict()
    with pytest.raises(ValueError, match="format"):
        Decision.from_dict({**record, "format": "other"})
    with pytest.raises(ValueError, match="version"):
        Decision.from_dict({**record, "version": 9})
    missing = {k: v for k, v in record.items() if k != "level"}
    with pytest.raises(ValueError, match="missing"):
        Decision.from_dict(missing)
    with pytest.raises(ValueError, match="unexpected"):
        Decision.from_dict({**record, "note": "x"})
    with pytest.raises(ValueError):
        Decision.from_dict({**record, "outcome": "maybe"})


def test_the_outcome_values_are_the_documented_strings():
    assert [o.value for o in Outcome] == ["select", "abstain", "fallback"]
    assert Outcome("select") is Outcome.SELECT and Outcome.SELECT == "select"
    for text in ("SELECT", "ABSTAIN", "FALLBACK"):
        assert text.lower() in (Outcome.__doc__ or "").lower()
    doc = " ".join(Outcome.__doc__.split())
    assert "NOT certified and NOT covered" in doc and "There is no default fallback" in doc
    assert "safe outcome, not an error" in doc


def good(**over):
    base = decisions()[0].to_dict()
    base.update(over)
    return base


@pytest.mark.parametrize(("over", "message"), [
    ({"certified": False}, "certifying flag"),
    ({"certified_positions": []}, "member of"),
    ({"selected": None}, "member of"),
    ({"outcome": "abstain"}, "not certified|ABSTAIN"),
    ({"n_queries": 0}, "n_queries"),
    ({"metric": ""}, "metric"),
    ({"query_order": {"kind": "random"}}, "query_order"),
    ({"rule": {"name": "x"}}, "rule must hold"),
])
def test_an_inconsistent_decision_cannot_be_built(over, message):
    with pytest.raises((ValueError, TypeError), match=message):
        Decision.from_dict(good(**over))


def test_abstain_and_fallback_records_are_checked_for_consistency():
    abstain = decisions()[1].to_dict()
    fallback = decisions()[2].to_dict()
    with pytest.raises(ValueError):
        Decision.from_dict({**abstain, "selected": decisions()[0].to_dict()["selected"]})
    with pytest.raises(ValueError):
        Decision.from_dict({**abstain, "certified": True})
    with pytest.raises(ValueError, match="FALLBACK"):
        Decision.from_dict({**fallback, "selected": None})
    moved = {**fallback, "selected": {**fallback["selected"], "position": 0}}
    with pytest.raises(ValueError, match="FALLBACK"):
        Decision.from_dict(moved)
    with pytest.raises(ValueError):
        Decision.from_dict({**fallback, "certified_positions": [0]})


def test_a_decision_states_what_it_is_conditional_on():
    d = decisions()[0]
    assert len(d.assumptions) == 6 and d.assumptions[0].startswith("A1")
    assert d.rule["config"]["bet_cap"] == 0.9
    assert d.space_identity and d.evidence_digest and d.metric and d.construct


# ---------------------------------------------------------- the S0 list, end to end


@pytest.mark.skipif(not HAVE_SCIPY, reason="the registered default needs SciPy")
def test_the_s0_end_to_end_problem_through_select():
    s0 = load_s0()
    space = CandidateSpace.default(128)
    assert tuple(space.candidate_families) == tuple(s0.families)
    c9, b9 = tv9_problem(s0)
    ev = Evidence("ndcg@5", "labels", b9, c9, tuple(f"q{i}" for i in range(450)), s0.bytes,
                  s0.vectors, space.identity)
    for (target, delta), (selected, certified) in TV9_EXPECT.items():
        d = select(space, ev, target=target, error=delta)
        assert list(d.certified_positions) == certified
        assert (None if d.selected is None else d.selected.position) == selected
        assert d.level == delta / 7 and d.certified == (selected is not None)
        assert d.outcome is (Outcome.ABSTAIN if selected is None else Outcome.SELECT)


@pytest.mark.skipif(not HAVE_SCIPY, reason="the registered default needs SciPy")
def test_nested_budgets_on_s0_never_increase_the_selected_size():
    s0 = load_s0()
    space = CandidateSpace.default(128)
    c9, b9 = tv9_problem(s0)
    ev = Evidence("ndcg@5", "labels", b9, c9, tuple(f"q{i}" for i in range(450)), s0.bytes,
                  s0.vectors, space.identity)
    last = np.inf
    for budget in (50, 100, 150, 225, 300, 450):
        d = select(space, ev, target=0.95, error=0.05, query_budget=budget)
        current = np.inf if d.selected is None else s0.bytes[d.selected.position]
        assert current <= last
        last = current


def test_candidate_specs_in_a_decision_are_members_of_the_space_by_value():
    space = space4()
    d = select(space, evidence_for(space, "ssss"), target=T, error=ERR)
    assert d.selected in list(space) and isinstance(d.selected, CandidateSpec)
    assert d.selected.to_pipeline().to_dict() == space[3].to_pipeline().to_dict()


def test_decisions_survive_copy_deepcopy_and_pickle():
    import copy
    import pickle

    for decision in decisions():
        for clone in (copy.deepcopy(decision), pickle.loads(pickle.dumps(decision))):
            assert clone == decision and clone.to_dict() == decision.to_dict()
            with pytest.raises(TypeError):
                clone.rule["name"] = "x"  # type: ignore[index]


def test_selected_must_be_a_candidate_spec_or_none():
    d = decisions()[0]
    fields = {f.name: getattr(d, f.name) for f in dataclasses.fields(d)}
    with pytest.raises(TypeError, match="selected"):
        Decision(**{**fields, "selected": {"position": 3}})
