"""The record of one selection: what was decided, from which evidence, under which assumptions.

A :class:`Decision` is a value. It carries everything needed to say what it covers and what
it does not: the rule and its configuration, the assumptions the rule's statement is claimed
under, the target and error level, the p-values, the identities of the candidate space and of
the evidence, and how the queries were ordered. Nothing in it is a promise about retention on
other queries, other corpora or other encoders.

``certified`` is True only when the outcome is ``SELECT`` and the rule is one of the library's
own certifying rules (``rules.is_builtin_rule``). A rule written by anyone else is recorded
with the class it came from and what it declared about itself, its selection is never called
certified, and its assumptions are stored as the author's, not examined by the library.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from optivision.core.identity import digest

from .rules import builtin_rule_from_record
from .spaces import NOT_IN_SPACE, CandidateSpec, freeze_json, thaw_json

_FORMAT = "optivision-decision"
_VERSION = 1
_RULE_KEYS = ("name", "class", "builtin", "certifying", "declared_certifying", "config")


class Outcome(str, Enum):
    """The three things a selection can end in.

    ``SELECT``    a candidate was chosen from those the rule rejected the hypothesis for.
                  ``Decision.certified`` is True when the rule is one of the library's own
                  certifying rules, and the rule's statement (claimed only under its
                  assumptions) applies to the procedure that made the choice.
    ``ABSTAIN``   nothing was certified, so the uncompressed index is to be used. This is the
                  safe outcome, not an error: the statement also holds if the rule always
                  abstains.
    ``FALLBACK``  nothing was certified and the CALLER supplied an explicit fallback
                  pipeline, which is returned instead. It is NOT certified and NOT covered by
                  the rule's statement. There is no default fallback: without one the outcome
                  is ``ABSTAIN``.
    """

    SELECT = "select"
    ABSTAIN = "abstain"
    FALLBACK = "fallback"


_KEYS = frozenset({
    "format", "version", "outcome", "selected", "certified", "certified_positions", "p_values",
    "level", "target", "error", "rule", "assumptions", "n_queries", "metric", "construct",
    "space_identity", "evidence_digest", "query_order",
})


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class Decision:
    """What a call to ``select`` decided.

    ``selected`` is ``None`` on ``ABSTAIN``, the chosen member of the space on ``SELECT``, and
    on ``FALLBACK`` the caller's pipeline with ``position == -1`` (not a member of the space).
    ``certified`` is True only when ``outcome is SELECT`` and the rule is a built-in
    certifying one; ``certified_positions`` lists every member the rule rejected the
    hypothesis for, in list order. ``p_values`` has one entry per candidate and ``level`` is
    the level they were compared with. ``rule`` holds ``{"name", "class", "builtin",
    "certifying", "declared_certifying", "config"}``: ``class`` is the qualified class name,
    ``builtin`` says the library's own rule ran, ``declared_certifying`` is what the rule
    says about itself and ``certifying`` is what the library accepts (``builtin`` and
    declared). ``assumptions`` are the sentences the rule's statement is claimed under (for a
    rule that is not built in, labelled as declared by its author). ``n_queries`` is the
    number of queries tested, after any budget. ``query_order`` records how they were ordered
    and how many were used. The decision is bound to its evidence by ``evidence_digest`` and
    to its candidate space by ``space_identity``.
    """

    outcome: Outcome
    selected: CandidateSpec | None
    certified: bool
    certified_positions: tuple[int, ...]
    p_values: tuple[float, ...]
    level: float
    target: float
    error: float
    rule: Mapping[str, Any]
    assumptions: tuple[str, ...]
    n_queries: int
    metric: str
    construct: str
    space_identity: str
    evidence_digest: str
    query_order: Mapping[str, Any]

    def __post_init__(self) -> None:
        outcome = Outcome(self.outcome)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "certified_positions",
                           tuple(int(k) for k in self.certified_positions))
        object.__setattr__(self, "p_values", tuple(float(p) for p in self.p_values))
        object.__setattr__(self, "assumptions", tuple(str(a) for a in self.assumptions))
        object.__setattr__(self, "level", float(self.level))
        object.__setattr__(self, "target", float(self.target))
        object.__setattr__(self, "error", float(self.error))
        object.__setattr__(self, "rule", freeze_json(self.rule))
        object.__setattr__(self, "query_order", freeze_json(self.query_order))
        if not isinstance(self.certified, bool):
            raise TypeError("certified must be a bool")
        if self.selected is not None and not isinstance(self.selected, CandidateSpec):
            raise TypeError("selected must be a CandidateSpec or None")
        if isinstance(self.n_queries, bool) or not isinstance(self.n_queries, int) \
                or self.n_queries < 1:
            raise ValueError(f"n_queries must be a positive int, got {self.n_queries!r}")
        for name in ("metric", "construct", "space_identity", "evidence_digest"):
            _text(name, getattr(self, name))
        if not all(key in self.rule for key in _RULE_KEYS):
            raise ValueError(f"rule must hold {', '.join(_RULE_KEYS)}")
        for key in ("builtin", "certifying", "declared_certifying"):
            if not isinstance(self.rule[key], bool):
                raise TypeError(f"rule[{key!r}] must be a bool")
        for key in ("name", "class"):
            _text(f"rule[{key!r}]", self.rule[key])
        if self.rule["certifying"] != (self.rule["builtin"] and self.rule["declared_certifying"]):
            raise ValueError("rule['certifying'] must be true exactly when the rule is built in "
                             "and declares itself certifying: the library certifies only its "
                             "own rules")
        if self.rule["builtin"] and builtin_rule_from_record(self.rule) is None:
            raise ValueError("a rule recorded as built in must be one of the library's own "
                             f"rules; name {self.rule['name']!r} and class "
                             f"{self.rule['class']!r} are not")
        if self.query_order.get("kind") not in ("given", "seeded_permutation"):
            raise ValueError("query_order['kind'] must be 'given' or 'seeded_permutation'")
        self._check_consistency(outcome)

    def _check_consistency(self, outcome: Outcome) -> None:
        if outcome is Outcome.SELECT:
            if self.selected is None or self.selected.position not in self.certified_positions:
                raise ValueError("SELECT needs a selected candidate that is a member of "
                                 "certified_positions")
            if self.certified != self.rule["certifying"]:
                raise ValueError("certified must equal the rule's certifying flag on SELECT")
            return
        if self.certified or self.certified_positions:
            raise ValueError(f"{outcome.name} is not certified and has no certified candidate")
        if outcome is Outcome.ABSTAIN and self.selected is not None:
            raise ValueError("ABSTAIN selects nothing")
        if outcome is Outcome.FALLBACK and (
                self.selected is None or self.selected.position != NOT_IN_SPACE):
            raise ValueError("FALLBACK needs the caller's pipeline with position -1")

    def __hash__(self) -> int:
        return hash(digest(self.to_dict()))

    def __reduce__(self) -> tuple[Any, ...]:
        return (Decision.from_dict, (self.to_dict(),))

    def to_dict(self) -> dict[str, Any]:
        """JSON-native, fresh copy; ``Decision.from_dict(d.to_dict()) == d``."""
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "outcome": self.outcome.value,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "certified": self.certified,
            "certified_positions": list(self.certified_positions),
            "p_values": list(self.p_values),
            "level": self.level,
            "target": self.target,
            "error": self.error,
            "rule": thaw_json(self.rule),
            "assumptions": list(self.assumptions),
            "n_queries": self.n_queries,
            "metric": self.metric,
            "construct": self.construct,
            "space_identity": self.space_identity,
            "evidence_digest": self.evidence_digest,
            "query_order": thaw_json(self.query_order),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Decision:
        if data.get("format") != _FORMAT:
            raise ValueError(f"not a decision record (format {data.get('format')!r})")
        if data.get("version") != _VERSION:
            raise ValueError(f"unsupported decision version {data.get('version')!r}")
        if set(data) != _KEYS:
            raise ValueError(f"a decision record has exactly the keys {sorted(_KEYS)}; "
                             f"missing {sorted(_KEYS - set(data))}, "
                             f"unexpected {sorted(set(data) - _KEYS)}")
        selected = data["selected"]
        return cls(
            outcome=Outcome(data["outcome"]),
            selected=None if selected is None else CandidateSpec.from_dict(selected),
            certified=data["certified"],
            certified_positions=tuple(data["certified_positions"]),
            p_values=tuple(data["p_values"]),
            level=data["level"],
            target=data["target"],
            error=data["error"],
            rule=data["rule"],
            assumptions=tuple(data["assumptions"]),
            n_queries=data["n_queries"],
            metric=data["metric"],
            construct=data["construct"],
            space_identity=data["space_identity"],
            evidence_digest=data["evidence_digest"],
            query_order=data["query_order"],
        )
