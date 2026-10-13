"""``select``: a frozen candidate list, per-query evidence, a target and an error level in;
a decision out, which may be to abstain.

``select`` is a pure function of its arguments: no clock, no environment, no random draw
other than the permutation seeded by ``order_seed``. It never forces compression. When no
candidate is certified the outcome is ``ABSTAIN`` (the uncompressed index is to be used);
an explicit fallback pipeline supplied by the caller turns that into ``FALLBACK``, which is
recorded as not certified and is not covered by the rule's statement. There is no default
fallback.

Exactly one ``(target, error)`` pair per call. The rule's statement is about one pair on one
sample; looking at the same sample under several targets or error levels, or after changing
the list, order or seed, spends error level that the statement does not account for
(``docs/package/METHOD_C_SPEC.md`` section 6.3; in simulations such looks multiplied the
false-selection probability by up to 14, ``docs/package/STATISTICAL_REVIEW.md`` section 7.7). A
caller who needs several targets needs fresh calibration queries for each, or a rule built for
that.

``Decision.certified`` is True only for a selection by one of the library's own certifying
rules (``rules.is_builtin_rule``: the exact type). Any other rule is recorded with its class
and what it declared about itself, its selection is never certified, and its assumptions are
stored as its author's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from optivision.compose import Pipeline
from optivision.core.evaluate.evidence import Evidence
from optivision.core.identity import array_digest

from .decision import Decision, Outcome
from .rules import FixedSequenceBetting, SelectionRule, assumptions_of, rule_record
from .spaces import NOT_IN_SPACE, CandidateSpace, CandidateSpec
from .statistics import validate_evidence_arrays

#: Family label recorded for an explicit fallback pipeline, which belongs to no family.
FALLBACK_FAMILY = "fallback"

#: The seed of the random order that the high-level entry points (``certify``, the command
#: line) use unless told otherwise. ``select`` itself defaults to the order given.
DEFAULT_ORDER_SEED = 0


class SpaceMismatchError(ValueError):
    """The evidence was not collected on the candidate space it is being used with."""


class QueryOrderError(ValueError):
    """A query budget or an order seed that cannot be used."""


def _int_argument(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise QueryOrderError(f"{name} must be an integer, got {type(value).__name__}")
    if int(value) < minimum:
        raise QueryOrderError(f"{name} must be >= {minimum}, got {int(value)}")
    return int(value)


def _query_indices(n: int, query_budget: int | None,
                   order_seed: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    """The positions of the queries to test, in test order, and the record of how."""
    if order_seed is None:
        order = np.arange(n, dtype=np.int64)
    else:
        seed = _int_argument("order_seed", order_seed, 0)
        order = np.random.default_rng(seed).permutation(n).astype(np.int64, copy=False)
    if query_budget is not None:
        budget = _int_argument("query_budget", query_budget, 1)
        if budget > n:
            raise QueryOrderError(f"query_budget {budget} exceeds the {n} queries in the "
                                  "evidence; a budget takes a prefix of the order and "
                                  "never samples with replacement")
        order = order[:budget]
    record = {
        "kind": "given" if order_seed is None else "seeded_permutation",
        "seed": None if order_seed is None else int(order_seed),
        "budget": None if query_budget is None else int(query_budget),
        "n_available": int(n),
        "n_used": int(order.size),
        "indices_sha256": array_digest(order),
    }
    return order, record


def _fallback_spec(fallback: Pipeline | Mapping[str, Any] | None) -> CandidateSpec | None:
    if fallback is None:
        return None
    if isinstance(fallback, Pipeline):
        description = fallback.to_dict()
    elif isinstance(fallback, Mapping):
        description = Pipeline.from_dict(dict(fallback)).to_dict()  # refuses unknown stages
    else:
        raise TypeError("fallback must be None, a Pipeline or a Pipeline.to_dict() mapping, "
                        f"got {type(fallback).__name__}")
    return CandidateSpec(NOT_IN_SPACE, FALLBACK_FAMILY, description)


def select(space: CandidateSpace, evidence: Evidence, *, target: float, error: float,
           rule: SelectionRule | None = None, fallback: Pipeline | Mapping[str, Any] | None = None,
           query_budget: int | None = None, order_seed: int | None = None) -> Decision:
    """Select a candidate of ``space`` from ``evidence``, or abstain.

    ``evidence`` must have been collected on ``space`` (same identity, same number of
    candidates). Queries are tested in the order they have in ``evidence``; if ``order_seed``
    is given that order is replaced by ``np.random.default_rng(order_seed).permutation(n)``,
    a permutation independent of the data, and the seed is recorded. ``query_budget``
    then keeps the leading ``query_budget`` queries of the order and raises if it exceeds the
    number available. The arrays are validated (see ``validate_evidence_arrays``) and the
    rule, ``FixedSequenceBetting()`` by default, decides.

    The default is the order given (``order_seed=None``), because this layer is exercised
    against the research code. A given order is valid for the rule's statement only if it is
    unrelated to the outcomes (assumption A5): the arrival order of independent draws, never
    an order sorted by difficulty, date, source or any other query attribute. ``certify()``
    and the command line default to ``order_seed=DEFAULT_ORDER_SEED`` (0) so that a sorted
    input cannot break A5. Choose the seed before you look and use it once: trying several
    seeds and reporting one is outside the statement (``docs/package/STATISTICAL_REVIEW.md``,
    sections 7.1 and 8, measured how far).

    Outcomes: ``SELECT`` when the rule rejected the hypothesis for at least one candidate (the
    smallest by size is chosen); ``ABSTAIN`` when it did so for none; ``FALLBACK`` when it did
    so for none and ``fallback`` (a ``Pipeline`` or its ``to_dict()``) was given, in which case
    the result is not certified. ``fallback`` is ignored when a candidate is selected. The
    selection is certified (``Decision.certified``) only when the rule is one of the library's
    own certifying rules; a rule of anyone else's never gives a certified decision.

    ``target`` is the retention to be exceeded, in (0, 1]; ``error`` is the level of the
    rule's statement, in (0, 1). Exactly one of each per call (see the module docstring).
    """
    if not isinstance(space, CandidateSpace):
        raise TypeError(f"space must be a CandidateSpace, got {type(space).__name__}")
    if not isinstance(evidence, Evidence):
        raise TypeError(f"evidence must be an Evidence, got {type(evidence).__name__}")
    rule = FixedSequenceBetting() if rule is None else rule
    if not isinstance(rule, SelectionRule):
        raise TypeError("rule must provide name, certifying, config(), assumptions() and "
                        "decide()")
    if evidence.space_identity != space.identity:
        raise SpaceMismatchError(
            "the evidence was collected on a different candidate space: evidence identity "
            f"{evidence.space_identity[:16]}..., space identity {space.identity[:16]}...")
    if evidence.n_candidates != len(space):
        raise SpaceMismatchError(f"the evidence has {evidence.n_candidates} candidates, the "
                                 f"space has {len(space)}")
    spec = _fallback_spec(fallback)

    order, record = _query_indices(evidence.n_queries, query_budget, order_seed)
    tested = evidence if record["kind"] == "given" and record["budget"] is None \
        else evidence.subset(order)
    families = space.candidate_families
    checked = validate_evidence_arrays(tested.candidates, tested.baseline, families,
                                       tested.bytes_per_doc, tested.vectors_per_doc,
                                       target, error)
    result = rule.decide(checked.candidates, checked.baseline, checked.families,
                         checked.bytes_per_doc, checked.vectors_per_doc,
                         checked.target, checked.error)
    n_candidates = len(space)
    if any(not 0 <= k < n_candidates for k in result.certified) or (
            result.selected is not None and not 0 <= result.selected < n_candidates):
        raise ValueError("the rule returned a position outside the candidate space")
    if len(result.p_values) != n_candidates:
        raise ValueError("the rule returned a p-value count that differs from the space")

    record_of_rule = rule_record(rule)
    if result.selected is not None:
        outcome, chosen = Outcome.SELECT, space[result.selected]
        certified = bool(record_of_rule["certifying"])  # built in and declared, never a custom rule
    elif spec is not None:
        outcome, chosen, certified = Outcome.FALLBACK, spec, False
    else:
        outcome, chosen, certified = Outcome.ABSTAIN, None, False
    return Decision(
        outcome=outcome,
        selected=chosen,
        certified=certified,
        certified_positions=result.certified,
        p_values=result.p_values,
        level=result.level,
        target=checked.target,
        error=checked.error,
        rule=record_of_rule,
        assumptions=assumptions_of(rule),
        n_queries=tested.n_queries,
        metric=evidence.metric,
        construct=evidence.construct,
        space_identity=space.identity,
        evidence_digest=evidence.digest(),
        query_order=record,
    )
