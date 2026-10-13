"""Selection: a frozen candidate space, per-query evidence, a rule, a decision.

    space = CandidateSpace.default(dim)              # frozen, development-informed list
    decision = select(space, evidence, target=0.97, error=0.05)
    decision.outcome                                 # SELECT, ABSTAIN or FALLBACK

``select`` may abstain; it never forces compression. The rule that certifies is an
application of published tools (testing by betting, fixed-sequence testing, a union bound),
not a new statistical method; its statement is :data:`STATEMENT`, claimed only under the
assumptions and conditions in ``Decision.assumptions``, and only for the library's own rules.
The package is named ``selection`` rather than ``select`` because
``optivision.calibration.select`` already exists.
"""

from __future__ import annotations

from .decision import Decision, Outcome
from .engine import (
    DEFAULT_ORDER_SEED,
    FALLBACK_FAMILY,
    QueryOrderError,
    SpaceMismatchError,
    select,
)
from .rules import (
    STATEMENT,
    BonferroniBetting,
    FixedSequenceBetting,
    RuleResult,
    SelectionRule,
    assumptions_of,
    builtin_rule_from_record,
    is_builtin_rule,
    rule_from_name,
    rule_record,
)
from .spaces import (
    NOT_IN_SPACE,
    REGISTERED_DEFAULT_SPACE_SHA256,
    CandidateSpace,
    CandidateSpec,
)
from .statistics import (
    BET_CAP,
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

__all__ = [
    "BET_CAP",
    "DEFAULT_ORDER_SEED",
    "FALLBACK_FAMILY",
    "NOT_IN_SPACE",
    "REGISTERED_DEFAULT_SPACE_SHA256",
    "STATEMENT",
    "BonferroniBetting",
    "CandidateSpace",
    "CandidateSpec",
    "Decision",
    "EmptySampleError",
    "ErrorLevelError",
    "EvidenceError",
    "FixedSequenceBetting",
    "NonFiniteError",
    "Outcome",
    "QueryOrderError",
    "RangeError",
    "RuleResult",
    "SelectionRule",
    "ShapeError",
    "SpaceMismatchError",
    "TargetError",
    "ZeroBaselineError",
    "assumptions_of",
    "betting_pvalues",
    "builtin_rule_from_record",
    "fixed_sequence_by_family",
    "is_builtin_rule",
    "rule_from_name",
    "rule_record",
    "select",
    "smallest",
    "to_y",
    "validate_evidence_arrays",
]
