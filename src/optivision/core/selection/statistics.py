"""The arithmetic of the selection rule: betting p-values, fixed-sequence testing, the size key.

These functions are the package copy of the research's Method C pieces
(``scripts/research/e1_stats.py``). They keep the research's operation order and
dtype handling line for line, so that parity with the frozen functions can be tested
bit for bit. They do no input validation; :func:`validate_evidence_arrays` does, and
the rules call it before they call anything here.

This is an application of published tools, not a new statistical method:

* exact one-sided p-values by *testing by betting* (Waudby-Smith and Ramdas,
  "Estimating means of bounded random variables by betting", J. R. Stat. Soc. B 86(1),
  2024), with the aGRAPA bet, a bet cap fixed at design time, and the running maximum of
  the wealth process (anytime-valid through Ville's inequality);
* *fixed-sequence* multiple testing in a pre-specified order (Westfall and Krishen,
  J. Statist. Plann. Inference 99, 2001; the same device is used by Learn then Test,
  Angelopoulos et al., 2021, arXiv:2110.01052);
* a union bound over chains, which splits the error level equally.

The citations above are given from memory and were not checked against the sources; the
research record cites Waudby-Smith and Ramdas and "learn-then-test-style" testing
(``reports/research/Q10/REPORT.md``) and does not cite the other two.

The derivation that combines them for this setting is the project's own and has not
been externally reviewed (see ``METHOD_C_SPEC.md`` section 6).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

#: Cap on the bet, as a fraction of the largest admissible bet ``1 / m0``. Validity holds for
#: any cap in (0, 1); 0.9 is the research's design-time constant and is not a parameter.
BET_CAP = 0.9


# ----------------------------------------------------------------- the frozen arithmetic


def to_y(c, b, target):
    """``Y = (c - target * b + target) / (1 + target)``, in [0, 1] for metric values in [0, 1].

    ``c`` and ``b`` are cast to float64 *before* any arithmetic (the research's functions
    receive float64 arrays; a float32 input would otherwise stay float32). The expression
    is evaluated in exactly this order: algebraically equivalent forms differ in the last
    bits and are not bit-identical to the frozen function.

    ``E_Q[Y] <= m0 = target / (1 + target)`` is equivalent to retention ``E_Q[c] / E_Q[b]``
    not exceeding ``target`` when ``E_Q[b] > 0``, which turns "the candidate keeps ``target``
    of the baseline's metric" into a statement about the mean of a bounded variable, the
    setting of Waudby-Smith and Ramdas, "Estimating means of bounded random variables by
    betting", J. R. Stat. Soc. B 86(1), 2024. This is an application of that published
    setting, not a new method.
    """
    c = np.asarray(c, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (c - target * b + target) / (1.0 + target)


def betting_pvalues(y, m0):
    """One-sided p-values for ``H0: E[Y] <= m0``, one per row, time on the LAST axis.

    The wealth ``K_t = prod_s (1 + lam_s (Y_s - m0))`` uses a predictable bet ``lam_s`` in
    ``[0, BET_CAP / m0]`` (aGRAPA: ``(mu - m0) / (var + (mu - m0)^2)`` from the regularised
    running mean and variance, prior mean 1/2 and prior variance 1/4 counted as one
    pseudo-observation). Under H0 and independent, identically distributed observations
    ``K_t`` is a non-negative supermartingale, so by Ville's inequality
    ``P(max_t K_t >= 1/a) <= a`` and ``p = min(1, 1 / max_t K_t)`` is a valid p-value for
    every sample size. The running maximum includes the starting wealth 1.

    The recursion is sequential in time and the p-value depends on the order of the
    observations; rows (candidates) are independent, so leading axes may be stacked
    freely. A 1-D input returns a scalar; an empty sample returns 1.0.

    Source: Waudby-Smith and Ramdas, "Estimating means of bounded random variables by
    betting", J. R. Stat. Soc. B 86(1), 2024 (aGRAPA bet, regularised running mean and
    variance), with Ville's inequality for the running maximum. This is an application of
    those published tools, not a new statistical method. The bet cap is the design-time
    constant ``BET_CAP``: validity holds for any cap in (0, 1), and it is not tuned here.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[-1]
    lead = y.shape[:-1]
    log_k = np.zeros(lead)
    log_kmax = np.zeros(lead)
    s1 = np.zeros(lead)
    s2 = np.zeros(lead)
    cap = BET_CAP / m0
    for t in range(n):
        mu = (0.5 + s1) / (t + 1)
        var = (0.25 + s2) / (t + 1)
        d = mu - m0
        lam = np.clip(d / (var + d * d), 0.0, cap)
        yt = y[..., t]
        log_k = log_k + np.log1p(lam * (yt - m0))
        log_kmax = np.maximum(log_kmax, log_k)
        s1 = s1 + yt
        mu_t = (0.5 + s1) / (t + 2)
        s2 = s2 + (yt - mu_t) ** 2
    return np.minimum(1.0, np.exp(-log_kmax))


def fixed_sequence_by_family(p, families, level):
    """Reject along each family's list order until that family's earliest non-rejection.

    A candidate is rejected when ``p <= level`` (inclusive), and only while its family has
    not stopped; the earliest non-rejection of a family stops that family only, and later
    members are ignored however small their p-value. Families are compared for equality
    only. Returns the rejected positions in list order. These are the "certified" members.

    Source: fixed-sequence multiple testing in a pre-specified order (Westfall and Krishen,
    J. Statist. Plann. Inference 99, 2001; also the device of Learn then Test, Angelopoulos
    et al., 2021, arXiv:2110.01052). Inside one family a false rejection needs the earliest
    true null of the order to be rejected, which happens with probability at most ``level``
    under arbitrary dependence; a union bound over F families at ``level = error / F``
    controls the family-wise error over all candidates at ``error``. This is an application
    of those published tools, not a new statistical method. The order inside a family must
    be fixed before the data are seen.
    """
    rejected, stopped = [], set()
    for k, pk in enumerate(p):
        f = families[k]
        if f in stopped:
            continue
        if pk <= level:
            rejected.append(k)
        else:
            stopped.add(f)
    return rejected


def smallest(cands, bytes_, vectors):
    """The member of ``cands`` with the smallest ``(bytes, vectors, position)``, else ``None``.

    Exact comparisons, no tolerance. ``position`` is the list index, so the key is unique.
    The choice among certified candidates is deterministic and uses no statistics: any rule
    for picking a member of the certified set keeps the statement of the rules that certify.
    """
    return min(cands, key=lambda k: (bytes_[k], vectors[k], k)) if cands else None


# ------------------------------------------------------------------------- validation


class EvidenceError(ValueError):
    """The arrays handed to a selection rule are not valid evidence."""


class ShapeError(EvidenceError):
    """A wrong shape, a wrong length, or a ragged input."""


class DtypeError(EvidenceError, TypeError):
    """A value that is not a real number array (or a real scalar where one is needed)."""


class NonFiniteError(EvidenceError):
    """NaN or infinity in the evidence."""


class RangeError(EvidenceError):
    """A per-query metric value outside [0, 1], or a negative size."""


class EmptySampleError(EvidenceError):
    """No queries: nothing can be tested (the research's functions return p = 1 silently)."""


class ZeroBaselineError(EvidenceError):
    """The mean of the uncompressed index's metric is zero: retention is undefined."""


class TargetError(EvidenceError):
    """The target is not in (0, 1]."""


class ErrorLevelError(EvidenceError):
    """The error level is not in (0, 1)."""


class CheckedInputs(NamedTuple):
    """What :func:`validate_evidence_arrays` returns: float64 arrays and Python floats."""

    candidates: np.ndarray
    baseline: np.ndarray
    families: tuple[str, ...]
    bytes_per_doc: np.ndarray
    vectors_per_doc: np.ndarray
    target: float
    error: float


def _float64(name: str, value: Any, ndim: int) -> np.ndarray:
    try:
        a = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ShapeError(f"{name} could not be read as a regular numeric array ({exc})") from exc
    if a.dtype.kind not in "fiu" or (a.dtype.kind == "f" and a.dtype.itemsize > 8):
        raise DtypeError(f"{name} must hold real numbers of at most 64 bits, got dtype {a.dtype}")
    if a.ndim != ndim:
        raise ShapeError(f"{name} must have {ndim} dimension(s), got shape {a.shape}")
    return a.astype(np.float64, copy=False)


def _where(mask: np.ndarray) -> str:
    index = tuple(int(i) for i in np.argwhere(mask)[0])
    return f"e.g. at index {index}"


def _real_scalar(name: str, value: Any, error: type[EvidenceError]) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise error(f"{name} must be one real number (exactly one per call), "
                    f"got {type(value).__name__}")
    x = float(value)
    if not math.isfinite(x):
        raise error(f"{name} must be finite, got {x}")
    return x


def validate_evidence_arrays(candidates, baseline, families, bytes_per_doc, vectors_per_doc,
                             target, error) -> CheckedInputs:
    """Check the inputs of a selection rule and return them as float64 / Python floats.

    Raises a subclass of ``ValueError`` (:class:`EvidenceError`) on: a wrong shape or
    length (``candidates`` is ``[K, n]``, ``baseline`` is ``[n]``, ``families``,
    ``bytes_per_doc`` and ``vectors_per_doc`` have length K), ``K < 1``, ``n = 0``, NaN or
    infinity anywhere, a metric value outside [0, 1], a negative size, a zero mean baseline
    (retention undefined), a family label that is not a non-empty string, a target outside
    (0, 1], or an error level outside (0, 1).

    Nothing is dropped, clipped or replaced. The only conversion is the exact widening of
    float16, float32 and integer input to float64, which happens before any arithmetic;
    wider-than-64-bit floats, booleans and non-numeric dtypes are refused. Queries without
    a relevant document are removed by the caller, for every candidate and the baseline
    together; this function never drops a query. ``target`` and ``error`` are single real
    numbers: one ``(target, error)`` per call, because every further look at the same sample
    consumes error level that the statement does not account for.
    """
    c = _float64("candidates", candidates, 2)
    b = _float64("baseline", baseline, 1)
    n_cand, n = c.shape
    if n_cand < 1:
        raise ShapeError("candidates must hold at least one candidate (K >= 1)")
    if b.shape != (n,):
        raise ShapeError(f"baseline must have shape ({n},) to match candidates {c.shape}, "
                         f"got {b.shape}")
    if n == 0:
        raise EmptySampleError("there are no queries (n = 0): nothing can be tested; "
                               "the sample must hold at least one query")
    if isinstance(families, (str, bytes)) or not isinstance(families, (Sequence, np.ndarray)):
        raise ShapeError("families must be a sequence of K labels, one per candidate")
    labels = tuple(families)
    if len(labels) != n_cand:
        raise ShapeError(f"families has {len(labels)} labels for {n_cand} candidates")
    for k, label in enumerate(labels):
        if not isinstance(label, str) or not label:
            raise DtypeError(f"family label {k} must be a non-empty string, got {label!r}")
    size = _float64("bytes_per_doc", bytes_per_doc, 1)
    vectors = _float64("vectors_per_doc", vectors_per_doc, 1)
    for name, arr in (("bytes_per_doc", size), ("vectors_per_doc", vectors)):
        if arr.shape != (n_cand,):
            raise ShapeError(f"{name} must have shape ({n_cand},), got {arr.shape}")
    for name, arr in (("candidates", c), ("baseline", b), ("bytes_per_doc", size),
                      ("vectors_per_doc", vectors)):
        bad = ~np.isfinite(arr)
        if bad.any():
            raise NonFiniteError(f"{name} contains NaN or infinity ({_where(bad)}); remove "
                                 "queries without a relevant document for every candidate "
                                 "and the baseline together before testing")
    for name, arr in (("candidates", c), ("baseline", b)):
        bad = (arr < 0.0) | (arr > 1.0)
        if bad.any():
            raise RangeError(f"{name} must lie in [0, 1] ({_where(bad)}); the test needs a "
                             "bounded per-query metric")
    for name, arr in (("bytes_per_doc", size), ("vectors_per_doc", vectors)):
        if (arr < 0.0).any():
            raise RangeError(f"{name} must not be negative ({_where(arr < 0.0)})")
    if float(b.mean()) == 0.0:
        raise ZeroBaselineError("the baseline metric is zero on every query: retention is "
                                "undefined (the test needs a positive mean baseline)")
    t = _real_scalar("target", target, TargetError)
    if not 0.0 < t <= 1.0:
        raise TargetError(f"target must be in (0, 1], got {t}")
    e = _real_scalar("error", error, ErrorLevelError)
    if not 0.0 < e < 1.0:
        raise ErrorLevelError(f"error must be in (0, 1), got {e}")
    return CheckedInputs(c, b, labels, size, vectors, t, e)
