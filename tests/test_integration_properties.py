"""Property tests of the selection path on synthetic evidence, over many seeds.

Seeded NumPy loops and ``pytest.mark.parametrize`` (hypothesis is not installed). They exercise
``select`` end to end (``validate_evidence_arrays``, the betting p-values, fixed-sequence testing,
the size key, the ``Decision`` record) on the candidate lists the package uses: a small custom
list and the 40-candidate default list (7 chains).

What is checked is the implementation, on synthetic per-query metric values. Nothing here is a
research experiment, no registered experiment was re-run, and a rate computed from simulated data
says nothing about any encoder. Parity of the arithmetic with the frozen research functions is
``tests/test_core_statistics_parity.py``.

Two groups:

* invariants that must hold whatever the data (membership, nesting in the error level, the
  monotonicity of ``METHOD_C_SPEC.md`` section 2.3 and TV11, permutation and relabelling
  invariance, dependence on the arrays only);
* the statistical sanity of the whole decision path at the null boundary, with the bound stated
  in each test.
"""

from __future__ import annotations

import itertools
import math
from typing import NamedTuple

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.core.evaluate import Evidence
from optivision.core.selection import (
    BonferroniBetting,
    CandidateSpace,
    CandidateSpec,
    FixedSequenceBetting,
    Outcome,
    betting_pvalues,
    fixed_sequence_by_family,
    select,
    to_y,
)
from optivision.stages import Float16Quantizer

try:
    import scipy.cluster.hierarchy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

#: per-query nDCG@5 values: rank 1..5 and a miss (the grid of the research stores)
GRID = np.concatenate([1.0 / np.log2(np.arange(2, 7)), [0.0]])
GRID_P = np.array([0.50, 0.08, 0.05, 0.03, 0.02, 0.32])

SMALL_LABELS = ["a", "a", "a", "b", "b", "c", "d", "d"]  # 4 chains, 8 candidates


# ====================================================================== the generator


def synthetic_space(labels: list[str], name: str = "synthetic") -> CandidateSpace:
    """A space with the given family labels (positions in list order); the pipelines are only
    labels here: ``select`` never builds them."""
    specs = tuple(
        CandidateSpec(k, label, {"stages": [{"stage": "truncate", "dim": k + 1}]})
        for k, label in enumerate(labels)
    )
    return CandidateSpace(name, specs)


class Problem(NamedTuple):
    space: CandidateSpace
    evidence: Evidence
    retention: np.ndarray  # the true retention of every candidate (what the data were drawn from)
    target: float
    error: float


def draw_values(rng: np.random.Generator, retention: np.ndarray, n: int):
    """Baseline and candidate metrics: a candidate loses the whole per-query value with
    probability ``1 - retention`` (the 'rare total loss' construction of ``METHOD_C_SPEC.md``
    section 9.2), so its true retention under the drawing distribution is exactly ``retention``."""
    b = GRID[rng.choice(6, size=n, p=GRID_P)]
    lost = rng.random((retention.size, n)) < (1.0 - retention)[:, None]
    return b, np.where(lost, 0.0, b[None, :])


def make_problem(seed: int, labels: list[str], space: CandidateSpace | None = None) -> Problem:
    rng = np.random.default_rng(seed)
    n = int(rng.choice([40, 80, 160, 300]))
    space = synthetic_space(labels) if space is None else space
    families = list(dict.fromkeys(labels))
    retention = np.empty(len(labels))
    sizes = np.empty(len(labels))
    vectors = np.empty(len(labels))
    for family in families:
        at = [k for k, label in enumerate(labels) if label == family]
        start = float(rng.choice([1.0, 0.999, 0.99, 0.97]))
        drop = np.sort(rng.random(len(at))) * float(rng.choice([0.02, 0.1, 0.4]))
        retention[at] = start - drop  # least to most aggressive: retention falls along a chain
        sizes[at] = np.sort(rng.choice([800.0, 400.0, 200.0, 100.0, 50.0], size=len(at)))[::-1]
        vectors[at] = np.sort(rng.choice([16.0, 8.0], size=len(at)))[::-1]
    b, c = draw_values(rng, retention, n)
    evidence = Evidence("ndcg@5", "labels", b, c, tuple(f"q{i}" for i in range(n)), sizes, vectors,
                        space.identity)
    return Problem(space, evidence, retention, float(rng.choice([0.9, 0.95])),
                   float(rng.choice([0.05, 0.1])))


def default_labels() -> list[str]:
    return list(CandidateSpace.default(128).candidate_families)


KINDS = ["small"] + (["default"] if HAVE_SCIPY else [])


def problem(seed: int, kind: str) -> Problem:
    if kind == "small":
        return make_problem(seed, SMALL_LABELS)
    space = CandidateSpace.default(128)
    return make_problem(seed, list(space.candidate_families), space)


def decide(p: Problem, **options):
    return select(p.space, p.evidence, target=options.pop("target", p.target),
                  error=options.pop("error", p.error), **options)


def size_key(evidence: Evidence, position: int) -> tuple[float, float, int]:
    return (float(evidence.bytes_per_doc[position]), float(evidence.vectors_per_doc[position]),
            position)


SEEDS = range(60)


# ============================================================ invariants of the decision


class TestTheDecisionIsConsistent:
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_generator_reaches_every_outcome(self, kind):
        """Guards the properties below against being vacuous: both outcomes occur, with chains
        that stop early and chains that run to the end."""
        outcomes = {Outcome.SELECT: 0, Outcome.ABSTAIN: 0}
        partial = full = 0
        for seed in SEEDS:
            p = problem(seed, kind)
            d = decide(p)
            outcomes[d.outcome] += 1
            for family in dict.fromkeys(p.space.candidate_families):
                at = [k for k, f in enumerate(p.space.candidate_families) if f == family]
                got = [k for k in d.certified_positions if k in at]
                partial += 0 < len(got) < len(at)
                full += len(got) == len(at)
        assert min(outcomes.values()) >= 5, outcomes
        assert partial >= 5 and full >= 5, (partial, full)

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_selected_candidate_is_a_certified_member_with_the_smallest_key(self, kind, seed):
        p = problem(seed, kind)
        d = decide(p)
        assert len(d.p_values) == len(p.space) and d.level == d.error / p.space.n_families
        if d.outcome is Outcome.SELECT:
            assert d.certified is True and d.selected is not None
            assert d.selected.position in d.certified_positions
            key = size_key(p.evidence, d.selected.position)
            assert key == min(size_key(p.evidence, k) for k in d.certified_positions)
            assert d.selected == p.space[d.selected.position]
        else:
            assert d.outcome is Outcome.ABSTAIN
            assert d.certified is False and d.selected is None and d.certified_positions == ()
        # a fallback never turns an abstention into a certified choice, and never replaces a pick
        spare = Pipeline([Float16Quantizer()])
        with_fallback = decide(p, fallback=spare)
        if d.outcome is Outcome.SELECT:
            assert with_fallback.to_dict() == d.to_dict()
        else:
            assert with_fallback.outcome is Outcome.FALLBACK
            assert with_fallback.certified is False and with_fallback.certified_positions == ()
            assert with_fallback.selected.position == -1

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_every_chain_certifies_a_prefix_and_the_rule_matches_a_scalar_transcription(
            self, kind, seed):
        """The certified set is what ``METHOD_C_SPEC.md`` section 3 says (a prefix of each chain,
        stopping at the first p-value above the level) and the p-values are those of a plain
        Python transcription of section 2.2, written from the formulae and not from the code."""
        p = problem(seed, kind)
        d = decide(p)
        families = p.space.candidate_families
        for family in dict.fromkeys(families):
            at = [k for k, f in enumerate(families) if f == family]
            got = [k for k in d.certified_positions if k in at]
            assert got == at[: len(got)]
            if len(got) < len(at):
                assert d.p_values[at[len(got)]] > d.level  # the chain stopped on a real failure
            assert all(d.p_values[k] <= d.level for k in got)
        if seed % 6 == 0:  # the scalar loop is slow: a sample of the problems
            b, c = p.evidence.baseline, p.evidence.candidates
            m0 = p.target / (1.0 + p.target)
            for k in range(0, len(p.space), 3):
                y = [(float(ci) - p.target * float(bi) + p.target) / (1.0 + p.target)
                     for ci, bi in zip(c[k], b, strict=True)]
                assert math.isclose(d.p_values[k], scalar_p_value(y, m0), rel_tol=1e-9, abs_tol=0)

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("seed", range(30))
    def test_lowering_the_error_level_never_enlarges_the_certified_set(self, kind, seed):
        p = problem(seed, kind)
        decisions = [decide(p, error=e) for e in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001)]
        for wider, narrower in itertools.pairwise(decisions):
            assert set(narrower.certified_positions) <= set(wider.certified_positions)
            assert narrower.level < wider.level
            assert narrower.p_values == wider.p_values  # p-values do not depend on the error level
        if decisions[-1].outcome is Outcome.SELECT:  # a certified pick at 0.001 stays certified
            for d in decisions:
                assert decisions[-1].selected.position in d.certified_positions

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("seed", range(40))
    def test_a_longer_prefix_of_the_queries_never_selects_a_larger_candidate(self, kind, seed):
        """``METHOD_C_SPEC.md`` section 2.3 and TV11, stated as they are: on nested samples (the
        first m queries, m growing) the p-value of every candidate never increases and the
        selected stored size never increases (an abstention is the largest size). The p-value
        depends on the order of the queries, so this is a statement about prefixes of one
        order; it does not hold for two arbitrary samples (see the witness below)."""
        p = problem(seed, kind)
        n = p.evidence.n_queries
        marks = sorted({max(10, n // 8 * j) for j in range(1, 9)} | {n})
        previous_p = np.full(len(p.space), np.inf)
        previous_size = np.inf
        previous_set: set[int] = set()
        for m in marks:
            d = decide(p, query_budget=m)
            assert d.n_queries == m and d.query_order["budget"] == m
            now = np.array(d.p_values)
            assert (now <= previous_p).all()
            size = np.inf if d.selected is None else p.evidence.bytes_per_doc[d.selected.position]
            assert size <= previous_size
            assert previous_set <= set(d.certified_positions)  # a consequence at a fixed level
            previous_p, previous_size, previous_set = now, size, set(d.certified_positions)

    def test_non_nested_samples_do_not_have_that_monotonicity(self):
        """A witness that the property above is about prefixes only. The p-value depends on the
        order of the queries (TV4: the same values, another order, p = 1 instead of below
        1e-5), so a *longer* sample that does not begin with the shorter one can select a
        *larger* candidate."""
        y = np.concatenate([np.ones(20), np.zeros(20)])
        m0 = 0.95 / 1.95
        assert betting_pvalues(y, m0) < 1e-5 and betting_pvalues(y[::-1].copy(), m0) == 1.0

        space = synthetic_space(["f", "f"])
        sizes = np.array([200.0, 100.0])  # candidate 1 is the smaller one
        lossy = np.vstack([np.ones(60), np.r_[np.ones(30), np.zeros(30)]])

        def evidence(candidates: np.ndarray) -> Evidence:
            n = candidates.shape[1]
            return Evidence("ndcg@5", "labels", np.ones(n), candidates,
                            tuple(f"q{i}" for i in range(n)), sizes, np.ones(2), space.identity)

        short = select(space, evidence(lossy[:, :30]), target=0.9, error=0.2)
        longer_other_order = select(space, evidence(lossy[:, ::-1]), target=0.9, error=0.2)
        prefix_extension = select(space, evidence(lossy), target=0.9, error=0.2)
        assert short.selected.position == 1  # the smaller candidate is certified on 30 queries
        assert prefix_extension.selected.position == 1  # nested: the larger sample keeps it
        assert longer_other_order.selected.position == 0  # 60 queries, loss first: a larger pick
        assert sizes[longer_other_order.selected.position] > sizes[short.selected.position]

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("seed", range(40))
    def test_permuting_candidates_across_families_keeps_the_certified_set(self, kind, seed):
        """Interleave the chains at random, keeping each chain's own order: the same candidates
        are certified, with the same p-values, and the same stored size is chosen."""
        p = problem(seed, kind)
        labels = list(p.space.candidate_families)
        rng = np.random.default_rng(seed + 1000)
        queues = {f: [k for k, label in enumerate(labels) if label == f]
                  for f in dict.fromkeys(labels)}
        order = []
        while any(queues.values()):
            f = rng.choice([f for f, q in queues.items() if q])
            order.append(queues[f].pop(0))
        order = np.array(order)
        space2 = synthetic_space([labels[k] for k in order], "interleaved")
        ev = p.evidence
        ev2 = Evidence(ev.metric, ev.construct, ev.baseline, ev.candidates[order], ev.query_ids,
                       ev.bytes_per_doc[order], ev.vectors_per_doc[order], space2.identity)
        d1 = decide(p)
        d2 = select(space2, ev2, target=p.target, error=p.error)
        assert sorted(int(order[k]) for k in d2.certified_positions) == list(d1.certified_positions)
        assert np.array_equal(np.array(d2.p_values), np.array(d1.p_values)[order])
        assert d2.level == d1.level and d2.outcome == d1.outcome
        if d1.selected is not None:
            key1 = size_key(ev, d1.selected.position)[:2]
            key2 = size_key(ev2, d2.selected.position)[:2]
            assert key1 == key2
            ties = [k for k in d1.certified_positions if size_key(ev, k)[:2] == key1]
            if len(ties) == 1:
                assert int(order[d2.selected.position]) == d1.selected.position

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("seed", range(40))
    def test_relabelling_the_families_bijectively_does_not_change_the_decision(self, kind, seed):
        p = problem(seed, kind)
        labels = list(p.space.candidate_families)
        distinct = list(dict.fromkeys(labels))
        rng = np.random.default_rng(seed + 2000)
        new = {f: f"zz{j}" for j, f in zip(rng.permutation(len(distinct)), distinct, strict=True)}
        space2 = synthetic_space([new[f] for f in labels], "relabelled")
        ev = p.evidence
        ev2 = Evidence(ev.metric, ev.construct, ev.baseline, ev.candidates, ev.query_ids,
                       ev.bytes_per_doc, ev.vectors_per_doc, space2.identity)
        d1, d2 = decide(p), select(space2, ev2, target=p.target, error=p.error)
        assert space2.identity != p.space.identity
        assert (d1.outcome, d1.certified_positions, d1.p_values, d1.level) == (
            d2.outcome, d2.certified_positions, d2.p_values, d2.level)
        assert (None if d1.selected is None else d1.selected.position) == (
            None if d2.selected is None else d2.selected.position)

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("seed", range(25))
    def test_the_decision_depends_on_the_arrays_and_nothing_else(self, kind, seed):
        """Different query ids, metric name, construct, dropped-query list and fitted-state
        digests, identical arrays: the same p-values, certified set and choice. (The record of
        the decision is bound to the evidence it was made from, so its digest differs.)"""
        p = problem(seed, kind)
        ev = p.evidence
        other = Evidence(
            metric="recall@10", construct="agreement@1", baseline=ev.baseline,
            candidates=ev.candidates, query_ids=tuple(f"other-{i}" for i in range(ev.n_queries)),
            bytes_per_doc=ev.bytes_per_doc, vectors_per_doc=ev.vectors_per_doc,
            space_identity=ev.space_identity, dropped_queries=("x", "y"),
            fitted=tuple(("0" * 64,) for _ in range(ev.n_candidates)))
        d1, d2 = decide(p), select(p.space, other, target=p.target, error=p.error)
        assert other.digest() != ev.digest() and d1.evidence_digest != d2.evidence_digest
        assert d2.metric == "recall@10" and d2.construct == "agreement@1"
        assert (d1.outcome, d1.certified_positions, d1.p_values, d1.level, d1.n_queries) == (
            d2.outcome, d2.certified_positions, d2.p_values, d2.level, d2.n_queries)
        assert (None if d1.selected is None else d1.selected.position) == (
            None if d2.selected is None else d2.selected.position)

    @pytest.mark.parametrize("seed", range(15))
    def test_a_seeded_order_is_the_same_as_permuting_the_arrays_first(self, seed):
        p = problem(seed, "small")
        n = p.evidence.n_queries
        order = np.random.default_rng(seed + 7).permutation(n)
        shuffled = p.evidence.subset(order)
        via_seed = decide(p, order_seed=seed + 7)
        via_arrays = select(p.space, shuffled, target=p.target, error=p.error)
        assert via_seed.p_values == via_arrays.p_values
        assert via_seed.certified_positions == via_arrays.certified_positions
        assert via_seed.query_order["kind"] == "seeded_permutation"
        assert via_seed.query_order["seed"] == seed + 7
        assert via_arrays.query_order["kind"] == "given"

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("seed", range(20))
    def test_the_bonferroni_rule_uses_the_same_p_values_at_a_level_over_the_candidates(
            self, kind, seed):
        p = problem(seed, kind)
        chain, union = decide(p), decide(p, rule=BonferroniBetting())
        assert chain.p_values == union.p_values
        assert union.level == p.error / len(p.space) <= chain.level
        assert union.certified_positions == tuple(
            k for k, v in enumerate(union.p_values) if v <= union.level)
        assert chain.rule["name"] == "fixed_sequence_betting"
        assert union.rule["name"] == "bonferroni_betting"
        chain_rule = FixedSequenceBetting().decide(
            p.evidence.candidates, p.evidence.baseline, p.space.candidate_families,
            p.evidence.bytes_per_doc, p.evidence.vectors_per_doc, p.target, p.error)
        assert tuple(chain_rule.certified) == chain.certified_positions


class TestTheLevelIsInclusiveAndExactlyTheErrorOverTheChains:
    """``p <= level`` rejects (``METHOD_C_SPEC.md`` section 2.2 and 10, item 12) and the level
    of the chain rule is ``error / F`` with ``F`` the number of distinct families, of the
    union-bound rule ``error / K``. A p-value is placed exactly on the level: the number of
    candidates is a power of two, so ``error = K * p`` and ``error / K == p`` hold exactly in
    floating point, and one ulp below ``error`` must reject nothing."""

    TARGET = 0.9

    @classmethod
    def identical_to_baseline(cls, n_candidates: int):
        """Candidates whose per-query metric equals the baseline's on every query (retention 1
        by construction), and their common p-value, which lies in (0, 1/8)."""
        m0 = cls.TARGET / (1.0 + cls.TARGET)
        for n in range(1, 400):
            ones = np.ones(n)
            p = float(betting_pvalues(to_y(ones, ones, cls.TARGET), m0))
            if 0.0 < p < 0.125:
                return np.ones((n_candidates, n)), np.ones(n), p
        raise AssertionError("no sample size gives a p-value in (0, 1/8)")  # pragma: no cover

    @pytest.mark.parametrize("n_chains", [1, 2, 4, 8])
    def test_the_chain_rule_rejects_exactly_at_error_over_the_number_of_families(self, n_chains):
        c, b, p = self.identical_to_baseline(8)
        labels = [f"f{k % n_chains}" for k in range(8)]
        order = sorted(range(8), key=lambda k: labels[k])  # chains contiguous, in list order
        labels = [labels[k] for k in order]
        size, vec = np.arange(8.0, 0.0, -1.0), np.ones(8)
        on_the_level = FixedSequenceBetting().decide(c, b, labels, size, vec, self.TARGET,
                                                     n_chains * p)
        assert on_the_level.level == p and on_the_level.p_values == (p,) * 8
        assert on_the_level.certified == tuple(range(8)) and on_the_level.selected == 7
        just_below = FixedSequenceBetting().decide(
            c, b, labels, size, vec, self.TARGET, float(np.nextafter(n_chains * p, 0.0)))
        assert just_below.level < p and just_below.certified == () and just_below.selected is None

    def test_the_union_bound_rule_rejects_exactly_at_error_over_the_number_of_candidates(self):
        c, b, p = self.identical_to_baseline(8)
        labels = ["a", "a", "b", "b", "c", "c", "d", "d"]  # four chains: the level is error / 8
        size, vec = np.arange(8.0, 0.0, -1.0), np.ones(8)
        rule = BonferroniBetting()
        on_the_level = rule.decide(c, b, labels, size, vec, self.TARGET, 8 * p)
        assert on_the_level.level == p and on_the_level.certified == tuple(range(8))
        just_below = rule.decide(c, b, labels, size, vec, self.TARGET,
                                 float(np.nextafter(8 * p, 0.0)))
        assert just_below.certified == () and just_below.selected is None
        # the chain rule on the same labels spends error / 4 and so rejects at error = 4 p
        chain = FixedSequenceBetting().decide(c, b, labels, size, vec, self.TARGET, 4 * p)
        assert chain.level == p and chain.certified == tuple(range(8))

    @pytest.mark.parametrize("rule", [FixedSequenceBetting(), BonferroniBetting()],
                             ids=["chain", "union"])
    def test_select_applies_the_same_comparison_and_records_the_level(self, rule):
        c, b, p = self.identical_to_baseline(8)
        labels = ["a", "a", "b", "b", "c", "c", "d", "d"]
        space = synthetic_space(labels)
        evidence = Evidence("ndcg@5", "labels", b, c, tuple(f"q{i}" for i in range(b.size)),
                            np.arange(8.0, 0.0, -1.0), np.ones(8), space.identity)
        divisor = 4 if rule.name == "fixed_sequence_betting" else 8
        d = select(space, evidence, target=self.TARGET, error=divisor * p, rule=rule)
        assert d.level == p and d.outcome is Outcome.SELECT and d.selected.position == 7
        below = select(space, evidence, target=self.TARGET,
                       error=float(np.nextafter(divisor * p, 0.0)), rule=rule)
        assert below.outcome is Outcome.ABSTAIN and below.certified_positions == ()


def scalar_p_value(y: list[float], m0: float) -> float:
    """``METHOD_C_SPEC.md`` section 2.2 with plain floats and ``math``."""
    cap = 0.9 / m0
    s1 = s2 = log_k = best = 0.0
    for t, yt in enumerate(y):
        mu = (0.5 + s1) / (t + 1)
        var = (0.25 + s2) / (t + 1)
        d = mu - m0
        lam = min(max(d / (var + d * d), 0.0), cap)
        log_k += math.log1p(lam * (yt - m0))
        best = max(best, log_k)
        s1 += yt
        s2 += (yt - (0.5 + s1) / (t + 2)) ** 2
    return min(1.0, math.exp(-best))


# ======================================================== statistical sanity of the decision

REPS = 2000
DELTA = 0.05
#: the largest rate allowed, in binomial standard errors above the error level
SIGMAS = 4.0


def bound(level: float, reps: int = REPS) -> float:
    return level + SIGMAS * math.sqrt(level * (1.0 - level) / reps)


def simulate(retention: np.ndarray, labels: list[str], sizes: np.ndarray, n: int, target: float,
             seed: int, reps: int = REPS):
    """``reps`` independent calibration samples; the full decision path (``select``) on each.

    Returns the decisions and the stacked arrays (for the arithmetic-only control).
    """
    space = synthetic_space(labels, "boundary")
    rng = np.random.default_rng(seed)
    ids = tuple(f"q{i}" for i in range(n))
    vectors = np.ones(len(labels))
    decisions, bs, cs = [], [], []
    for _ in range(reps):
        b, c = draw_values(rng, retention, n)
        evidence = Evidence("ndcg@5", "labels", b, c, ids, sizes, vectors, space.identity)
        decisions.append(select(space, evidence, target=target, error=DELTA))
        bs.append(b)
        cs.append(c)
    return decisions, np.stack(bs), np.stack(cs)


# four chains of two candidates. The first candidate of chains a and b is just below the target
# (0.948 against 0.95), that of chains c and d exactly at it; the second candidates are well below.
# Every candidate is a true null (retention <= target), so a certified candidate is a false
# rejection, and a selected one of chains a and b is a "miss" in the sense of the statement.
LABELS = ["a", "a", "b", "b", "c", "c", "d", "d"]
TARGET = 0.95
RETENTION = np.array([0.948, 0.90, 0.948, 0.90, 0.95, 0.90, 0.95, 0.90])
SIZES = np.array([400.0, 200.0] * 4)

# one candidate equals the baseline on every query (retention 1 by construction); the
# others are bad (retention far below the target)
GOOD_LABELS = ["a", "a", "b", "b", "c"]
GOOD_RETENTION = np.array([1.0, 0.6, 0.6, 0.3, 0.8])
GOOD_SIZES = np.array([400.0, 200.0, 150.0, 50.0, 80.0])


class TestStatisticalSanityOfTheDecisionPath:
    """An implementation check on synthetic data, not a research result and not a power study.

    The rule's statement (``core.selection.STATEMENT``) bounds the probability that a compressed
    candidate is selected although its retention is below the target by the error level, over
    the draw of the queries. Simulated here with the "rare total loss" construction of
    ``METHOD_C_SPEC.md`` section 9.2, 2,000 seeded samples per configuration, and a tolerance of
    four binomial standard errors above the error level. The rates are conservative by design
    (the observed rates in the spec are well below the level); the negative control below shows
    that this test would still notice a rule whose level is not split over the chains."""

    @pytest.fixture(scope="class")
    def boundary(self):
        return simulate(RETENTION, LABELS, SIZES, n=225, target=TARGET,
                        seed=20261006)

    def test_a_candidate_below_the_target_is_selected_at_most_as_often_as_the_error_level(
            self, boundary):
        decisions, _, _ = boundary
        below = RETENTION < TARGET
        miss = np.mean([d.selected is not None and below[d.selected.position] for d in decisions])
        any_certified = np.mean([bool(d.certified_positions) for d in decisions])
        assert len(decisions) == REPS
        assert miss <= bound(DELTA), (miss, bound(DELTA))
        # a certified candidate here is a false rejection of a true null (retention <= target):
        # the family-wise error over the whole list is bounded by the same level
        assert any_certified <= bound(DELTA), (any_certified, bound(DELTA))
        # the decision records the level it used: the error split over the four chains
        assert {d.level for d in decisions} == {DELTA / 4}
        assert all(d.outcome in (Outcome.SELECT, Outcome.ABSTAIN) for d in decisions)

    def test_the_same_test_would_notice_a_level_that_is_not_split_over_the_chains(self, boundary):
        """Negative control, arithmetic only (the same samples, the same p-values, the level
        ``error`` instead of ``error / 4``): the family-wise error rate leaves the bound."""
        decisions, b, c = boundary
        m0 = TARGET / (1.0 + TARGET)
        p = betting_pvalues(to_y(c, b[:, None, :], TARGET), m0)
        families = LABELS
        wrong = np.mean([bool(fixed_sequence_by_family(p[j], families, DELTA))
                         for j in range(REPS)])
        right = np.mean([bool(fixed_sequence_by_family(p[j], families, DELTA / 4))
                         for j in range(REPS)])
        assert right == np.mean([bool(d.certified_positions) for d in decisions])
        assert wrong > bound(DELTA), (wrong, bound(DELTA))
        assert right < wrong

    def test_with_a_candidate_equal_to_the_baseline_only_bad_ones_are_never_selected(self):
        decisions, _, _ = simulate(GOOD_RETENTION, GOOD_LABELS, GOOD_SIZES,
                                   n=225, target=0.9, seed=99)
        bad = np.mean([d.selected is not None and d.selected.position != 0 for d in decisions])
        certified_bad = np.mean([any(k != 0 for k in d.certified_positions) for d in decisions])
        found = np.mean([d.selected is not None and d.selected.position == 0 for d in decisions])
        assert bad <= bound(DELTA) and certified_bad <= bound(DELTA), (bad, certified_bad)
        # not vacuous: a rule that always abstains would satisfy the line above
        assert found >= 0.8, found
        assert all(d.certified for d in decisions if d.outcome is Outcome.SELECT)
