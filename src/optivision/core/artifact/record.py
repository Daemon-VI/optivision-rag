"""What a certificate record contains, and the values that are derived from other values.

Shared by creation (``certificate.py``) and verification (``verification.py``), so that a
value is computed in one place: the content digest, the HMAC, the fixed texts that depend on
the outcome and the construct, the order in which queries were tested, and the descriptive
statistics of the selected candidate on that sample.

A certificate is a JSON object (``docs/package/CERTIFICATE_SPEC.md``). Its ``digest`` is the
SHA-256 of the canonical JSON of every field except ``digest`` and ``signature``. That digest is
an integrity check, not security: anyone who can edit a certificate can recompute it. The
optional ``signature`` is an HMAC over the digest; HMAC is symmetric, so whoever can verify can
also sign, and it gives no non-repudiation.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from typing import Any

import numpy as np

from optivision.core.evaluate.evidence import Evidence
from optivision.core.identity import array_digest, digest
from optivision.core.selection.rules import STATEMENT

SCHEMA = "optivision-certificate"
SCHEMA_VERSION = 1
SIGNATURE_ALGORITHM = "HMAC-SHA256"

#: Top-level fields, in the order they are written to a file. The digest does not depend on it.
FIELD_ORDER = (
    "schema", "schema_version", "created_utc", "package", "code", "encoder", "corpus", "queries",
    "reference", "space", "sizes", "rule", "statement", "assumptions", "caveats", "target",
    "error", "level", "evidence", "result", "sample", "candidate", "digest", "signature",
)
#: Fields the content digest does not cover.
UNCOVERED = ("digest", "signature")

OUTCOMES = ("select", "abstain", "fallback")

STATEMENT_NOT_CERTIFYING = (
    "The selection rule that produced this record is not a certifying rule: it makes no error "
    "statement about the candidate it selected, and nothing in this record is certified."
)
CAVEAT_AGREEMENT = (
    "Retention here means agreement with the uncompressed ranking, which is not retrieval "
    "quality. The uncompressed index's own top document is taken as the relevant one, so the "
    "measure is fidelity to that ranking, including its mistakes, and no statement in this "
    "record is a statement about relevance."
)
CAVEAT_FALLBACK = (
    "The deployed pipeline is a fallback supplied by the caller. It is not certified, and the "
    "statement above does not cover it."
)
CAVEAT_NOT_CERTIFIED = (
    "The selected candidate is not certified: the rule that chose it makes no error statement."
)
SAMPLE_NOTE = (
    "Descriptive statistics of the selected candidate on the queries that selected it. They are "
    "optimistically biased by the selection and are not certified quantities."
)


def content_digest(fields: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of every field except ``digest`` and ``signature``."""
    return digest({k: v for k, v in fields.items() if k not in UNCOVERED})


def signature_value(key: bytes, digest_hex: str) -> str:
    """Hex HMAC-SHA256 of the ASCII digest string under ``key``."""
    return hmac.new(bytes(key), digest_hex.encode("ascii"), hashlib.sha256).hexdigest()


def statement_for(certifying: bool) -> str:
    """The rule's statement for a certifying rule; a plain refusal for any other rule."""
    return STATEMENT if certifying else STATEMENT_NOT_CERTIFYING


def caveats_for(outcome: str, construct: str, certifying: bool) -> list[str]:
    """Sentences a reader must not miss, derived from the outcome, construct and rule."""
    out = []
    if construct.startswith("agreement@"):
        out.append(CAVEAT_AGREEMENT)
    if outcome == "fallback":
        out.append(CAVEAT_FALLBACK)
    if outcome == "select" and not certifying:
        out.append(CAVEAT_NOT_CERTIFIED)
    return out


def tested_indices(order: Mapping[str, Any], n_available: int) -> np.ndarray:
    """Positions (in the evidence) of the queries that were tested, in test order.

    Reproduces the order that ``select`` documents: the evidence order, or
    ``np.random.default_rng(seed).permutation(n)`` for a recorded seed, cut to the recorded
    budget. The result is checked against the digest of the indices that ``select`` recorded,
    so a change in how the engine orders queries is an error here, not a silent difference.
    """
    if order["kind"] == "given":
        idx = np.arange(n_available, dtype=np.int64)
    elif order["kind"] == "seeded_permutation":
        seed = order["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"the recorded order seed must be a non-negative integer, got {seed!r}")
        idx = np.random.default_rng(seed).permutation(n_available).astype(np.int64, copy=False)
    else:
        raise ValueError(f"unknown query order kind {order['kind']!r}")
    budget = order.get("budget")
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= n_available:
            raise ValueError(f"the recorded query budget {budget!r} does not fit {n_available} "
                             "queries")
        idx = idx[:budget]
    if array_digest(idx) != order.get("indices_sha256"):
        raise ValueError("the recorded query order cannot be reproduced from its kind, seed and "
                         "budget")
    return idx


def sample_record(evidence: Evidence, position: int, order: Mapping[str, Any]) -> dict[str, Any]:
    """Descriptive statistics of candidate ``position`` on the tested queries.

    ``mean`` is its mean metric, ``baseline_mean`` that of the uncompressed index on the same
    queries, ``retention_point`` their ratio. Optimistically biased by selection; not certified.
    """
    idx = tested_indices(order, evidence.n_queries)
    mean = float(np.mean(evidence.candidates[position][idx]))
    base = float(np.mean(evidence.baseline[idx]))
    if not base > 0.0:
        raise ValueError("the mean baseline metric on the tested queries is not positive")
    return {"n": int(idx.size), "mean": mean, "baseline_mean": base,
            "retention_point": mean / base, "note": SAMPLE_NOTE}
