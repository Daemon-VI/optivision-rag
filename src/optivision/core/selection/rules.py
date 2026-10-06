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
"""

from __future__ import annotations

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

#: The statement a certifying rule makes, with its conditions. It is an application of the
#: cited tools; the derivation is the project's own and has not been externally reviewed
#: (``docs/package/METHOD_C_SPEC.md`` section 6). It is never to be quoted without
#: assumptions A1-A6 (see :meth:`FixedSequenceBetting.assumptions`).
STATEMENT = (
    "P(a compressed candidate is selected and its retention under Q is below T) <= delta, "
    "where the probability is over the draw of the calibration queries and retention under Q "
    "is E_Q[metric of the candidate] / E_Q[metric of the uncompressed index]. "
    "It holds for every sample size, with no large-sample approximation, and only under "
    "assumptions A1-A6. It makes no statement about power: it also holds if the rule always "
    "abstains. It is an application of published tools; the derivation is this project's own "
    "and has not been externally reviewed."
)

_ASSUMPTIONS = (
    ("A1: the calibration queries are independent draws from one distribution Q, the "
     "deployment query distribution."),
    ("A2: the corpus is fixed; retention moves with corpus size, so the statement is about "
     "this corpus."),
    ("A3: the candidate list, its family labels and chains, and the order inside each chain "
     "are fixed before the calibration queries are seen, and every fitted stage is fitted on "
     "documents only."),
    ("A4: the per-query metric lies in [0, 1] and the expected metric of the uncompressed "
     "index under Q is positive."),
    ("A5: the order in which queries enter the sequential test is fixed before their outcomes "
     "are seen."),
    ("A6: the reference is taken as given (the labels under the label construct, the "
     "uncompressed index's own top document under the agreement construct); errors in the "
     "reference are not modelled."),
)

_ASSUMPTIONS_UNORDERED = (
    _ASSUMPTIONS[0],
    _ASSUMPTIONS[1],
    ("A3: the candidate list is fixed before the calibration queries are seen, and every "
     "fitted stage is fitted on documents only. No order inside a family is needed."),
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

    ``certifying`` says whether the rule makes an error statement about the candidate it
    selects. ``config()`` is JSON-native and identifies the rule; ``assumptions()`` are the
    sentences its statement is conditional on. ``decide`` validates its inputs and raises a
    ``ValueError`` subclass on invalid evidence; it never coerces or drops anything.
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
    86(1), 2024). Inside each family, in list order, a candidate is rejected when
    ``p <= error / F`` and the family stops at its earliest non-rejection (fixed-sequence
    testing); ``F`` is the number of distinct family labels *in the list passed*, never a
    constant. The chosen candidate is the certified one with the smallest
    ``(bytes_per_doc, vectors_per_doc, position)``, compared exactly; if none is certified
    there is no choice. This is an application of those published tools together with a
    union bound over chains. It is not a new statistical method.

    The statement (:data:`STATEMENT`) is conditional on assumptions A1-A6, holds for every
    sample size, and says nothing about power: the rule may abstain on any sample, and
    abstaining is the safe outcome, not an error. Validity does not need the order inside a
    family to be monotone; power depends on it. The p-values depend on the order of the
    queries, so that order must not be chosen after seeing the outcomes. Nothing is random.
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


def rule_from_name(name: str) -> SelectionRule:
    """The built-in rule called ``name`` (the string a ``Decision`` records in ``rule["name"]``).

    Lets a record be replayed from its evidence. Unknown names raise ``ValueError``.
    """
    try:
        return _BUILTIN_RULES[name]()
    except (KeyError, TypeError):
        raise ValueError(f"unknown selection rule {name!r}; built-in rules: "
                         f"{sorted(_BUILTIN_RULES)}") from None
