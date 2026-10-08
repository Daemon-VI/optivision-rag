"""What a certificate record contains, and the values that are derived from other values.

Shared by creation (``certificate.py``) and verification (``verification.py``), so that a
value is computed in one place: the content digest, the HMAC, the fixed texts that depend on
the rule, the outcome, the construct, the order and the inputs, the order in which queries
were tested, the descriptive statistics of the selected candidate on that sample, and two
properties of the inputs that the statement needs and no stage checks: how many queries repeat
an earlier one exactly, and whether the vectors have unit L2 norm.

A certificate is a JSON object (``docs/package/CERTIFICATE_SPEC.md``). Its ``digest`` is the
SHA-256 of the canonical JSON of every field except ``digest`` and ``signature``. That digest is
an integrity check, not security: anyone who can edit a certificate can recompute it. The
optional ``signature`` is an HMAC over the digest; HMAC is symmetric, so whoever can verify can
also sign, and it gives no non-repudiation.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from optivision.core.evaluate.evidence import Evidence
from optivision.core.identity import array_digest, digest
from optivision.core.selection.engine import DEFAULT_ORDER_SEED
from optivision.core.selection.rules import STATEMENT
from optivision.representation import MultiVectorCorpus

SCHEMA = "optivision-certificate"
SCHEMA_VERSION = 1
SIGNATURE_ALGORITHM = "HMAC-SHA256"

#: Top-level fields, in the order they are written to a file. The digest does not depend on it.
FIELD_ORDER = (
    "schema", "schema_version", "created_utc", "package", "code", "encoder", "corpus", "queries",
    "norms", "reference", "space", "sizes", "rule", "statement", "assumptions", "caveats",
    "target", "error", "level", "evidence", "result", "sample", "candidate", "digest",
    "signature",
)
#: Fields the content digest does not cover.
UNCOVERED = ("digest", "signature")

OUTCOMES = ("select", "abstain", "fallback")

#: A vector counts as unit-norm when its L2 norm is within this distance of 1.
NORM_TOLERANCE = 1e-3
_NORM_BLOCK_ELEMENTS = 1 << 21  # rows are normed in blocks of about 16 MB of float64

STATEMENT_UNEXAMINED = (
    "This record was produced by a selection rule that is not part of this library. The library "
    "has not examined that rule and makes no statement about it, and nothing in this record is "
    "certified. Any statement the rule makes is its author's."
)
CAVEAT_UNEXAMINED_RULE = (
    "The selection rule that produced this record is not part of this library. The library has "
    "not examined it, so a selected candidate is not certified, whatever the rule declares about "
    "itself, and the assumptions recorded here are the rule author's, not the library's."
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
CAVEAT_ORDER_GIVEN = (
    "The queries were tested in the order supplied. The statement requires that order to be "
    "unrelated to the outcomes (assumption A5): the arrival order of independent draws "
    "qualifies; an order sorted by difficulty, date, source or any other query attribute does "
    "not."
)
CAVEAT_ORDER_SEED = (
    "The order seed ({seed}) was chosen by the caller, not left at the default. Choosing a seed "
    "after looking at results, or trying several seeds and keeping one, is outside the "
    "statement."
)
CAVEAT_DUPLICATES = (
    "{n} of the {m} queries tested repeat an earlier query exactly (the same vectors). Repeated "
    "queries count as independent draws only if they really are; if they come from failed "
    "de-duplication, logs replayed twice or augmentation, assumption A1 does not hold and the "
    "stated error level does not apply."
)
CAVEAT_DUPLICATES_CONDITIONAL = (
    "{n} of the {m} queries tested repeat an earlier query exactly (the same vectors). "
    "{who} If they are copies (failed de-duplication, logs replayed twice or augmentation), "
    "assumption A1 does not hold and the stated error level does not apply."
)
DUPLICATES_ASSERTED = "The caller asserted that they are independent draws of the query distribution."
DUPLICATES_NOT_ASSERTED = "Whether they are independent draws was not asserted by the caller."
CAVEAT_NORMS = (
    "Not every vector has unit L2 norm (tolerance {tolerance:g}): {which}. The research studied "
    "only MaxSim over L2-normalised vectors, and nothing is known outside that scope."
)
CAVEAT_DROPPED = (
    "{k} of the {n} queries given have no relevant document under this reference and were "
    "removed before testing, for the baseline and for every candidate alike. The statement is "
    "about the queries that have a relevant document."
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


def statement_for(rule: Mapping[str, Any]) -> str:
    """The statement recorded for ``rule`` (an entry as ``rules.rule_record`` writes it).

    The library's ``STATEMENT`` only for one of its own certifying rules. For any other rule a
    fixed text that says the library has not examined it and makes no statement about it.
    """
    return STATEMENT if rule["certifying"] else STATEMENT_UNEXAMINED


def _norm_sentence(norms: Mapping[str, Any]) -> str | None:
    def show(x: Any) -> str:
        return "?" if x is None else f"{x:.6g}"

    parts = []
    for key, label in (("documents", "document"), ("queries", "query")):
        entry = norms[key]
        if not entry["unit_norm"]:
            parts.append(f"{label} vectors have norms from {show(entry['norm_min'])} to "
                         f"{show(entry['norm_max'])}")
    if not parts:
        return None
    return CAVEAT_NORMS.format(tolerance=norms["tolerance"], which=" and ".join(parts))


def caveats_for(d: Mapping[str, Any]) -> list[str]:
    """Sentences a reader must not miss, derived from the record's own fields.

    ``d`` is a certificate record (or the fields of one that is being written): the sentences
    depend on the rule, the outcome, the construct, how the queries were ordered, how many
    repeat one another, the norms of the vectors and the queries that were removed. Because they
    follow from fields the record carries, verification can recompute them.
    """
    out = []
    if not d["rule"]["builtin"]:
        out.append(CAVEAT_UNEXAMINED_RULE)
    if d["reference"]["construct"].startswith("agreement@"):
        out.append(CAVEAT_AGREEMENT)
    if d["result"]["outcome"] == "fallback":
        out.append(CAVEAT_FALLBACK)
    order = d["queries"]["order"]
    if order["kind"] == "given":
        out.append(CAVEAT_ORDER_GIVEN)
    elif order["seed"] != DEFAULT_ORDER_SEED:
        out.append(CAVEAT_ORDER_SEED.format(seed=order["seed"]))
    if d["queries"]["exact_duplicates"] > 0:
        asserted = d["queries"].get("duplicates_asserted_independent")
        if asserted is None:  # a record written before the assertion was recorded
            out.append(CAVEAT_DUPLICATES.format(n=d["queries"]["exact_duplicates"],
                                                m=d["queries"]["n_used"]))
        else:
            out.append(CAVEAT_DUPLICATES_CONDITIONAL.format(
                n=d["queries"]["exact_duplicates"], m=d["queries"]["n_used"],
                who=DUPLICATES_ASSERTED if asserted else DUPLICATES_NOT_ASSERTED))
    sentence = _norm_sentence(d["norms"])
    if sentence is not None:
        out.append(sentence)
    dropped = d["evidence"]["dropped_queries"]
    if dropped:
        out.append(CAVEAT_DROPPED.format(k=len(dropped), n=d["queries"]["n_pool"]))
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


# ---------------------------------------------------------- properties of the inputs


def exact_duplicates(queries: MultiVectorCorpus, dropped: Sequence[str],
                     order: Mapping[str, Any]) -> int:
    """How many of the queries the rule tested repeat an earlier tested query exactly.

    Two queries are exact duplicates when their vector arrays are identical (same dtype, shape
    and values: the digests of the per-query rows are compared); a repeated query id counts as
    well, although ``validate_corpus`` refuses repeated ids, so that only matters for objects
    built by hand. ``dropped`` are the ids removed for having no relevant document and ``order``
    is the decision's order record: the count is over the queries the rule saw, in the order
    it saw them, so a budget leaves the rest out. Near-duplicates are not detected.
    """
    gone = set(dropped)
    kept = [i for i, qid in enumerate(queries.ids) if qid not in gone]
    seen_vectors: set[str] = set()
    seen_ids: set[str] = set()
    repeats = 0
    for j in tested_indices(order, len(kept)).tolist():
        i = kept[j]
        rows = array_digest(queries.vectors[queries.offsets[i]:queries.offsets[i + 1]])
        qid = queries.ids[i]
        repeats += int(rows in seen_vectors or qid in seen_ids)
        seen_vectors.add(rows)
        seen_ids.add(qid)
    return repeats


def norm_record(corpus: MultiVectorCorpus) -> dict[str, Any]:
    """Whether every vector of ``corpus`` has unit L2 norm within ``NORM_TOLERANCE``.

    ``{"unit_norm", "norm_min", "norm_max"}``: the extremes of the row norms and whether both
    lie within the tolerance of 1 (``None`` extremes and ``True`` for a corpus without
    vectors). Rows are normed in blocks, in float64, so nothing as large as the corpus is
    allocated; the corpus is neither changed nor converted.
    """
    vectors = corpus.vectors
    low, high = float("inf"), float("-inf")
    step = max(1, _NORM_BLOCK_ELEMENTS // max(1, vectors.shape[1]))
    for start in range(0, vectors.shape[0], step):
        block = np.asarray(vectors[start:start + step], dtype=np.float64)
        norms = np.sqrt(np.einsum("ij,ij->i", block, block))
        low, high = min(low, float(norms.min())), max(high, float(norms.max()))
    if low > high:
        return {"unit_norm": True, "norm_min": None, "norm_max": None}
    unit = bool(low >= 1.0 - NORM_TOLERANCE and high <= 1.0 + NORM_TOLERANCE)
    return {"unit_norm": unit, "norm_min": low, "norm_max": high}


def norms_record(corpus: MultiVectorCorpus, queries: MultiVectorCorpus) -> dict[str, Any]:
    """The ``norms`` field: the tolerance, and :func:`norm_record` of the documents and queries."""
    return {"tolerance": NORM_TOLERANCE, "documents": norm_record(corpus),
            "queries": norm_record(queries)}
