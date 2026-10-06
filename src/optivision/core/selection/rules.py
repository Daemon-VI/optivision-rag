"""Selection rules: from per-query evidence to a chosen candidate, or none.

A rule sees arrays and nothing else: per-query metric values of the candidates and of the
uncompressed index, one family label per candidate, one stored size per candidate, a target
and an error level. It receives no encoder name, no dimension, no dataset name and no
provenance, and it must not look for any; family labels are grouping keys compared for
equality only. What a rule returns is a :class:`RuleResult`; turning that into a decision
that can abstain is the engine's job (``engine.select``).

Both rules are applications of published tools (testing by betting, fixed-sequence testing,
a union bound), not new statistical methods. Both test, for every candidate ``k``, the
hypothesis ``H0_k: retention(k) <= target`` with the same betting p-values, where retention
under the query distribution Q is ``E_Q[metric of the candidate] / E_Q[metric of the
uncompressed index]``. They differ only in how the error level is spent. A candidate is
called *certified* when, and only when, its hypothesis is rejected.

"Certified" is only ever asserted for these two classes, which the library derived and tested
(:func:`is_builtin_rule` compares the exact type: a subclass is not built in). A rule written
by anyone else may declare ``certifying = True``; the library does not examine it, records the
declaration as the author's own, and never calls its selection certified.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .statistics import (
    BET_CAP,
    CheckedInputs,
    betting_pvalues,
    fixed_sequence_by_family,
    smallest,
    to_y,
    validate_evidence_arrays,
)

#: The statement a built-in certifying rule makes, with its conditions. It is an application of
#: the cited tools; the derivation is the project's own and has not been externally reviewed
#: (``docs/package/METHOD_C_SPEC.md`` section 6). It is never to be quoted without
#: assumptions A1-A6 and conditions C1-C4 (see :meth:`FixedSequenceBetting.assumptions`).
#: Reviewed in ``docs/package/STATISTICAL_REVIEW.md`` (section 6, wording W1).
STATEMENT = (
    "For the target T and the error level delta of this decision: "
    "P(a compressed candidate is selected and its retention under Q is below T) <= delta. "
    "The probability is over the draw of the calibration queries; the corpus, the candidate "
    "list and its order, T and delta are fixed, not random. Q is the distribution the "
    "calibration queries are drawn from, and retention under Q is E_Q[metric of the "
    "candidate] / E_Q[metric of the uncompressed index]. "
    "This is a joint probability. It is NOT the probability, given that a candidate was "
    "selected, that its retention is below T: a rule that rarely selects can satisfy it "
    "while most of its selections are below T. It is not a statement about the retention "
    "measured on the calibration queries, not a statement about a finite batch of future "
    "queries, and not a statement about power (it also holds if the rule always abstains). "
    "The project derived it for every sample size, with no large-sample approximation, under "
    "assumptions A1-A6 and conditions C1-C4. They are sufficient conditions: nothing is "
    "claimed if one of them fails. "
    "It is an application of published tools; the derivation is this project's own and has "
    "not been externally reviewed."
)

#: The sentence recorded for every assumption a custom rule declares: the library has not
#: looked at them.
DECLARED_PREFIX = "Declared by the rule's author, not examined by this library: "

_ASSUMPTIONS = (
    ("A1: the calibration queries are independent draws from one distribution Q, the "
     "distribution of the queries the deployed index will serve. Repeated or near-duplicate "
     "queries, queries from one session or one author, and a sample sorted or filtered by "
     "anything related to difficulty are not independent draws."),
    ("A2: the corpus is fixed; retention moves with corpus size, so the statement is about "
     "this corpus."),
    ("A3: the candidate list, its family labels and chains, and the order inside each chain "
     "are fixed before the calibration queries are seen, and no fitted stage uses the "
     "calibration queries (in this library stages are fitted on documents only)."),
    ("A4: the per-query metric lies in [0, 1] and the expected metric of the uncompressed "
     "index under Q is positive."),
    ("A5: the order in which queries enter the sequential test is fixed before their outcomes "
     "are seen and is unrelated to them: the arrival order of independent draws, or a random "
     "permutation drawn without reference to the data. An order sorted by any query "
     "attribute, or picked among several orders after looking, is not."),
    ("A6: the reference is taken as given (the labels under the label construct, the "
     "uncompressed index's own top document under the agreement construct); errors in the "
     "reference are not modelled."),
    ("C1: one look. One target, one error level, one rule, one metric, one reference and one "
     "candidate list are fixed before the outcomes are seen, and the procedure is run once on "
     "this sample; re-running it with other choices on the same queries is outside the "
     "statement."),
    ("C2: the baseline and every candidate are scored on the same queries with the same "
     "metric and the same reference."),
    ("C3: the reference (the labels, or the uncompressed ranking) is fixed without regard to "
     "the candidates' results."),
    ("C4: the index that is deployed is the one that was evaluated: the same corpus, the same "
     "pipeline, the same fitted state and the same scoring procedure."),
)

_ASSUMPTIONS_UNORDERED = (
    _ASSUMPTIONS[0],
    _ASSUMPTIONS[1],
    ("A3: the candidate list is fixed before the calibration queries are seen, and no fitted "
     "stage uses the calibration queries (in this library stages are fitted on documents "
     "only). No order inside a family is needed."),
    *_ASSUMPTIONS[3:],
)


@dataclass(frozen=True)
class RuleResult:
    """What a rule decided on one sample.

    ``certified`` holds the positions whose hypothesis was rejected, in list order;
    ``selected`` is the one of them with the smallest ``(bytes, vectors, position)``, or
    ``None`` when none was rejected. ``p_values`` has one entry per candidate and ``level``
    is the level each of them was compared with.
    """

    selected: int | None
    certified: tuple[int, ...]
    p_values: tuple[float, ...]
    level: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", tuple(int(k) for k in self.certified))
        object.__setattr__(self, "p_values", tuple(float(p) for p in self.p_values))
        if self.selected is not None and self.selected not in self.certified:
            raise ValueError("a selected candidate must be a member of the certified set")


@runtime_checkable
class SelectionRule(Protocol):
    """A rule that maps evidence arrays to a :class:`RuleResult`.

    ``certifying`` is what the rule *declares* about itself: that it makes an error statement
    about the candidate it selects. The library certifies only for its own rules
    (:func:`is_builtin_rule`); for any other class ``Decision.certified`` is False whatever
    this says, and the declaration is recorded as the author's. ``config()`` is JSON-native
    and identifies the rule; ``assumptions()`` are the sentences its statement is claimed
    under. ``decide`` validates its inputs and raises a ``ValueError`` subclass on invalid
    evidence; it never coerces or drops anything.
    """

    name: str
    certifying: bool

    def config(self) -> dict[str, Any]: ...

    def assumptions(self) -> tuple[str, ...]: ...

    def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc,
               target, error) -> RuleResult: ...


def _betting_pvalues(v: CheckedInputs) -> np.ndarray:
    m0 = v.target / (1.0 + v.target)  # scripts/research/x_common.py:947
    # scripts/research/e1_1_study.py:116: one p-value per candidate, queries in the order given
    return betting_pvalues(to_y(v.candidates, v.baseline[None, :], v.target), m0)


@dataclass(frozen=True)
class FixedSequenceBetting:
    """Method C of the research: betting p-values, fixed-sequence testing per chain.

    Every candidate gets an exact betting p-value (Waudby-Smith and Ramdas, J. R. Stat. Soc. B
    86(1), 2024; the citations of this module were given from memory and have not been checked
    against the sources, ``docs/package/STATISTICAL_REVIEW.md`` section 12). Inside each family, in list order, a candidate is rejected when
    ``p <= error / F`` and the family stops at its earliest non-rejection (fixed-sequence
    testing); ``F`` is the number of distinct family labels *in the list passed*, never a
    constant. The chosen candidate is the certified one with the smallest
    ``(bytes_per_doc, vectors_per_doc, position)``, compared exactly; if none is certified
    there is no choice. This is an application of those published tools together with a
    union bound over chains. It is not a new statistical method.

    The statement (:data:`STATEMENT`) is claimed only under assumptions A1-A6 and conditions
    C1-C4, holds for every sample size, and says nothing about power: the rule may abstain on
    any sample, and abstaining is the safe outcome, not an error. Validity does not need the
    order inside a family to be monotone; power depends on it. The p-values depend on the
    order of the queries, so that order must be unrelated to the outcomes (A5). Nothing is
    random.
    """

    name = "fixed_sequence_betting"
    certifying = True

    def config(self) -> dict[str, Any]:
        """The research's ``method_c_config()`` plus the fields it left implicit."""
        return {
            "name": self.name,
            "port_of": "scripts/research/e1_stats.py and e1_1_study.py (Method C)",
            "hypothesis": "H0_k: E[c_k - T b] <= 0",
            "statistic": "Y = (c - T b + T) / (1 + T); m0 = T / (1 + T)",
            "p_value": "exact betting p-value, aGRAPA bets",
            "bet_cap": BET_CAP,
            "multiplicity": "fixed-sequence testing inside each chain, in the chain's own order",
            "level": "delta / number of chains",
            "choice": ("smallest stored size among the certified; ties by fewer vectors, "
                       "then earlier position"),
            "queries_used": "all n; no hold-out split",
            "prior_mean": 0.5,
            "prior_variance": 0.25,
            "bet": ("lam_t = clip((mu_t - m0) / (var_t + (mu_t - m0)^2), 0, bet_cap / m0), "
                    "computed from the observations before t"),
            "running_maximum": "maximum over t of the log wealth, starting wealth 1 included",
            "comparison": "p <= level (inclusive)",
            "level_rule": "error / F, F = number of distinct family labels in the list passed",
            "chains": "the number of distinct family labels in the list passed",
        }

    def assumptions(self) -> tuple[str, ...]:
        return _ASSUMPTIONS

    def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc,
               target, error) -> RuleResult:
        v = validate_evidence_arrays(candidates, baseline, families, bytes_per_doc,
                                     vectors_per_doc, target, error)
        n_families = len(dict.fromkeys(v.families))  # scripts/research/e1_1_study.py:70
        level = v.error / n_families  # scripts/research/x_common.py:942
        p = _betting_pvalues(v)
        rejected = fixed_sequence_by_family(p, v.families, level)
        selected = smallest(rejected, v.bytes_per_doc, v.vectors_per_doc)
        return RuleResult(selected, tuple(rejected), tuple(float(x) for x in p), level)


@dataclass(frozen=True)
class BonferroniBetting:
    """Method D of the research's E1 comparison: the same p-values, a union bound over K.

    Every candidate is tested at ``error / K`` (``K`` = number of candidates) and every
    candidate with ``p <= error / K`` is certified; there is no chain and no stopping. The
    chosen candidate is the certified one with the smallest
    ``(bytes_per_doc, vectors_per_doc, position)``. This is a union bound over betting
    p-values, an application of published tools, not a new method.

    For candidate lists without a meaningful least-to-most-aggressive order. On a list whose
    families are ordered it usually has less power than :class:`FixedSequenceBetting`, because
    it spends error level on every candidate instead of along each chain; no power statement
    is made here. The statement and the assumptions are those of the fixed-sequence rule,
    except that no order inside a family is needed (A3). Family labels are not used at all.
    """

    name = "bonferroni_betting"
    certifying = True

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "port_of": "scripts/research/e1_1_study.py, method D (lines 132-134)",
            "hypothesis": "H0_k: E[c_k - T b] <= 0",
            "statistic": "Y = (c - T b + T) / (1 + T); m0 = T / (1 + T)",
            "p_value": "exact betting p-value, aGRAPA bets",
            "bet_cap": BET_CAP,
            "multiplicity": "Bonferroni over all K candidates; no ordering, no stopping",
            "level": "delta / K, K = number of candidates in the list passed",
            "choice": ("smallest stored size among the certified; ties by fewer vectors, "
                       "then earlier position"),
            "queries_used": "all n; no hold-out split",
            "prior_mean": 0.5,
            "prior_variance": 0.25,
            "bet": ("lam_t = clip((mu_t - m0) / (var_t + (mu_t - m0)^2), 0, bet_cap / m0), "
                    "computed from the observations before t"),
            "running_maximum": "maximum over t of the log wealth, starting wealth 1 included",
            "comparison": "p <= level (inclusive)",
        }

    def assumptions(self) -> tuple[str, ...]:
        return _ASSUMPTIONS_UNORDERED

    def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc,
               target, error) -> RuleResult:
        v = validate_evidence_arrays(candidates, baseline, families, bytes_per_doc,
                                     vectors_per_doc, target, error)
        level = v.error / v.candidates.shape[0]  # scripts/research/e1_1_study.py:132
        p = _betting_pvalues(v)
        rejected = [int(j) for j in np.flatnonzero(p <= level)]
        selected = smallest(rejected, v.bytes_per_doc, v.vectors_per_doc)
        return RuleResult(selected, tuple(rejected), tuple(float(x) for x in p), level)


_BUILTIN_RULES = MappingProxyType({
    FixedSequenceBetting.name: FixedSequenceBetting,
    BonferroniBetting.name: BonferroniBetting,
})
_BUILTIN_TYPES = frozenset(_BUILTIN_RULES.values())


def rule_from_name(name: str) -> SelectionRule:
    """The built-in rule called ``name`` (the string a ``Decision`` records in ``rule["name"]``).

    Lets a record be replayed from its evidence. Unknown names raise ``ValueError``. A name
    alone does not make a record's rule built in: see :func:`builtin_rule_from_record`.
    """
    try:
        return _BUILTIN_RULES[name]()
    except (KeyError, TypeError):
        raise ValueError(f"unknown selection rule {name!r}; built-in rules: "
                         f"{sorted(_BUILTIN_RULES)}") from None


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def is_builtin_rule(rule: object) -> bool:
    """True when ``rule`` is an instance of exactly :class:`FixedSequenceBetting` or
    :class:`BonferroniBetting`.

    The exact type is compared, not ``isinstance`` and not ``rule.name``: a subclass that
    overrides ``decide`` and keeps the name is a different rule, and the library has not
    examined it.
    """
    return type(rule) in _BUILTIN_TYPES


def rule_record(rule: SelectionRule) -> dict[str, Any]:
    """What a decision and a certificate record about the rule that decided.

    ``name`` and ``class`` (the qualified class name) say which rule it was; ``builtin`` that it
    is one of the library's own (:func:`is_builtin_rule`); ``declared_certifying`` what the rule
    says about itself; ``certifying`` what the library accepts: true only for a built-in rule
    that declares it. ``config`` is the rule's own configuration.
    """
    builtin = is_builtin_rule(rule)
    declared = bool(rule.certifying)
    return {
        "name": rule.name,
        "class": _qualified(type(rule)),
        "builtin": builtin,
        "certifying": builtin and declared,
        "declared_certifying": declared,
        "config": rule.config(),
    }


def assumptions_of(rule: SelectionRule) -> tuple[str, ...]:
    """The sentences a decision records for ``rule``.

    A built-in rule's own assumptions, verbatim. For any other rule the sentences it declares,
    each prefixed with :data:`DECLARED_PREFIX`: the library has not examined them.
    """
    sentences = tuple(str(a) for a in rule.assumptions())
    if is_builtin_rule(rule):
        return sentences
    return tuple(DECLARED_PREFIX + a for a in sentences)


def builtin_rule_from_record(record: Mapping[str, Any]) -> SelectionRule | None:
    """The library's own rule that ``record`` (a ``rule`` entry) names, or ``None``.

    The record counts only if its name, its qualified class name and its ``builtin`` flag all
    describe the same built-in rule. A record that names a built-in rule but fails any of the
    three is not that rule: the verification of a certificate refuses to treat it as one.
    """
    try:
        cls = _BUILTIN_RULES.get(record.get("name"))
    except TypeError:  # an unhashable name in a malformed record
        return None
    if cls is None or record.get("builtin") is not True:
        return None
    if record.get("class") != _qualified(cls):
        return None
    return cls()
