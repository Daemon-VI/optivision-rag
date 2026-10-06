"""Verification of a certificate record: integrity, replay from evidence, binding to objects.

``run_checks`` works on the plain dict of a certificate (``Certificate.to_dict()``) and the
live objects the caller supplies. It never raises on a bad certificate: a check that cannot
be evaluated is reported as failed. Argument types are checked by ``verify_certificate``.

Level 1, integrity (the certificate alone): structure, content digest, HMAC (with a key),
internal consistency (everything that can be recomputed from the certificate's own fields),
the texts and traits the running library would write for the same rule and pipeline, and the
code identity. Level 2, replay (needs the evidence): the evidence digest, the certificate's
copies of the evidence's values, and a re-run of ``select`` with the recorded target, error,
rule, fallback, query budget and order seed. Level 3, binding (each object optional): the
corpus, the queries, the reference, the fitted pipeline, the compressed corpus.

What is and is not detected is tabulated in ``docs/package/CERTIFICATE_SPEC.md`` (section 4.1).
In short: an edit is caught by the digest unless the digest is recomputed; with the digest
recomputed it is caught by the consistency checks, the replay or the binding, *except* where the
edited value is not tied to anything (the code identity, descriptive fields, the counts and
norms that only the live queries and vectors can confirm), and a party who rewrites the
certificate and also supplies matching forged evidence and objects is not detected. Only an
HMAC with a key that party does not hold closes that gap, and HMAC is symmetric: it gives no
non-repudiation.

The library certifies only for its own rules. The checks that recompute the level, the decision
step and the replay, and the comparison of the rule's text, need the rule: when the recorded
rule is not one of the library's own (by its name, its qualified class name and its ``builtin``
flag together) a record that claims ``certified: true`` fails them, and a record that claims
nothing is certified skips them and says so in the report's notes.
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from optivision.compose import CompressedCorpus, Pipeline
from optivision.core.evaluate.evidence import Evidence
from optivision.core.evaluate.reference import Reference
from optivision.core.identity import array_digest, canonical_json, digest, strings_digest
from optivision.core.pipeline import pipeline_identity
from optivision.core.selection.engine import select
from optivision.core.selection.rules import (
    DECLARED_PREFIX,
    assumptions_of,
    builtin_rule_from_record,
    rule_record,
)
from optivision.core.selection.spaces import (
    NOT_IN_SPACE,
    REGISTERED_DEFAULT_SPACE_SHA256,
    CandidateSpace,
    CandidateSpec,
)
from optivision.core.selection.statistics import fixed_sequence_by_family, smallest
from optivision.core.stages.accounting import byte_accounting
from optivision.core.stages.state import pipeline_fitted_identity
from optivision.core.stages.traits import pipeline_traits
from optivision.core.vectors import corpus_identity
from optivision.representation import MultiVectorCorpus

from .codeid import code_identity
from .record import (
    OUTCOMES,
    SCHEMA,
    SCHEMA_VERSION,
    SIGNATURE_ALGORITHM,
    STATEMENT_UNEXAMINED,
    caveats_for,
    content_digest,
    exact_duplicates,
    norm_record,
    sample_record,
    signature_value,
    statement_for,
    tested_indices,
)
from .report import (
    BINDING,
    DIFFERS,
    FAIL,
    INTEGRITY,
    PASS,
    REPLAY,
    SKIPPED,
    Check,
    VerificationReport,
)

#: Relative tolerance of the replay comparison of p-values and of recomputed means. Everything
#: else is compared exactly. Bitwise equality across platforms is not promised.
RTOL = 1e-12

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CREATED = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

Result = tuple[str, str]


# ------------------------------------------------------------------ structure


def _real(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


_KINDS: dict[str, Callable[[Any], bool]] = {
    "str": lambda x: isinstance(x, str),
    "str?": lambda x: x is None or isinstance(x, str),
    "int": _int,
    "int?": lambda x: x is None or _int(x),
    "real": _real,
    "real?": lambda x: x is None or _real(x),
    "bool": lambda x: isinstance(x, bool),
    "dict": lambda x: isinstance(x, dict),
    "dict?": lambda x: x is None or isinstance(x, dict),
    "list": lambda x: isinstance(x, list),
    "strs": lambda x: isinstance(x, list) and all(isinstance(i, str) for i in x),
    "strs?": lambda x: isinstance(x, list) and all(i is None or isinstance(i, str) for i in x),
    "ints": lambda x: isinstance(x, list) and all(_int(i) for i in x),
    "reals": lambda x: isinstance(x, list) and all(_real(i) for i in x),
}

_IDENTITY = {
    "n_docs": "int", "n_vectors": "int", "dimension": "int", "dtype": "str",
    "vectors_sha256": "str", "offsets_sha256": "str", "ids_sha256": "str",
    "protected_sha256": "str?", "positions_sha256": "str?", "weights_sha256": "str?",
    "importance_sha256": "str?", "digest": "str",
}
_TOP = {
    "schema": "str", "schema_version": "int", "created_utc": "str", "package": "dict",
    "code": "dict", "encoder": "dict?", "corpus": "dict", "queries": "dict", "reference": "dict",
    "norms": "dict", "space": "dict", "sizes": "dict", "rule": "dict", "statement": "str",
    "assumptions": "strs", "caveats": "strs", "target": "real", "error": "real", "level": "real",
    "evidence": "dict", "result": "dict", "sample": "dict?", "candidate": "dict?",
    "digest": "str", "signature": "dict?",
}
_NESTED: dict[str, dict[str, str]] = {
    "package": {"name": "str", "version": "str"},
    "code": {"digest": "str?", "files": "dict", "python": "str", "numpy": "str", "scipy": "str?"},
    "corpus": _IDENTITY,
    "queries": {"n_pool": "int", "n_used": "int", "identity": "dict", "order": "dict",
                "exact_duplicates": "int"},
    "norms": {"tolerance": "real", "documents": "dict", "queries": "dict"},
    "reference": {"construct": "str", "metric": "str", "identity": "dict"},
    "space": {"name": "str", "identity": "str", "n_candidates": "int", "n_families": "int",
              "families": "strs", "registered_default": "bool", "candidates": "list"},
    "sizes": {"bytes_per_doc": "reals", "vectors_per_doc": "reals"},
    "rule": {"name": "str", "class": "str", "builtin": "bool", "certifying": "bool",
             "declared_certifying": "bool", "config": "dict"},
    "evidence": {"digest": "str", "n_queries": "int", "dropped_queries": "strs",
                 "baseline_mean": "real"},
    "result": {"outcome": "str", "certified": "bool", "selected": "int?",
               "certified_positions": "ints", "p_values": "reals"},
}
_NORM = {"unit_norm": "bool", "norm_min": "real?", "norm_max": "real?"}
_ORDER = {"kind": "str", "seed": "int?", "budget": "int?", "n_available": "int", "n_used": "int",
          "indices_sha256": "str"}
_CANDIDATE = {"pipeline": "dict", "pipeline_sha256": "str", "fitted_state": "strs?",
              "traits": "dict", "bytes": "dict", "codes_sha256": "str", "offsets_sha256": "str"}
_SAMPLE = {"n": "int", "mean": "real", "baseline_mean": "real", "retention_point": "real",
           "note": "str"}
_SIGNATURE = {"algorithm": "str", "key_id": "str?", "value": "str"}
_BYTES = ("code_bytes", "codec_overhead_bytes", "fitted_state_bytes", "append_state_bytes",
          "offsets_bytes", "float32_bytes_before", "n_docs")


def _require(node: Any, table: Mapping[str, str], where: str, problems: list[str]) -> bool:
    if not isinstance(node, dict):
        problems.append(f"{where} must be an object")
        return False
    ok = True
    for key, kind in table.items():
        if key not in node:
            problems.append(f"{where}.{key} is missing")
            ok = False
        elif not _KINDS[kind](node[key]):
            problems.append(f"{where}.{key} has the wrong type or content (expected {kind})")
            ok = False
    return ok


def structure_problems(d: Mapping[str, Any]) -> list[str]:
    """Everything wrong with the structure and value domains of a certificate record."""
    problems: list[str] = []
    if not isinstance(d, dict):
        return ["the record is not an object"]
    unknown = sorted(set(d) - set(_TOP))
    if unknown:
        problems.append(f"unexpected field(s): {unknown}")
    if not _require(d, _TOP, "certificate", problems):
        return problems
    if d["schema"] != SCHEMA:
        problems.append(f"schema is {d['schema']!r}, expected {SCHEMA!r}")
    if d["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version {d['schema_version']} is not supported (this code "
                        f"reads version {SCHEMA_VERSION})")
    if not _CREATED.match(d["created_utc"]):
        problems.append("created_utc is not of the form YYYY-MM-DDTHH:MM:SSZ")
    if not _HEX64.match(d["digest"]):
        problems.append("digest is not 64 lower-case hex characters")
    if not 0.0 < d["target"] <= 1.0:
        problems.append(f"target {d['target']} is outside (0, 1]")
    if not 0.0 < d["error"] < 1.0:
        problems.append(f"error {d['error']} is outside (0, 1)")
    if not 0.0 < d["level"] <= 1.0:
        problems.append(f"level {d['level']} is outside (0, 1]")
    for key, table in _NESTED.items():
        _require(d[key], table, key, problems)
    if isinstance(d["queries"], dict):
        _require(d["queries"].get("identity"), _IDENTITY, "queries.identity", problems)
        _require(d["queries"].get("order"), _ORDER, "queries.order", problems)
    if isinstance(d["norms"], dict):
        for key in ("documents", "queries"):
            _require(d["norms"].get(key), _NORM, f"norms.{key}", problems)
    if (isinstance(d["code"], dict) and isinstance(d["code"].get("files"), dict)
            and not all(isinstance(v, str) for v in d["code"]["files"].values())):
        problems.append("code.files must map paths to digests")
    if isinstance(d["space"], dict) and isinstance(d["space"].get("candidates"), list):
        for i, c in enumerate(d["space"]["candidates"]):
            _require(c, {"position": "int", "family": "str", "pipeline": "dict"},
                     f"space.candidates[{i}]", problems)
    if d["candidate"] is not None:
        _require(d["candidate"], _CANDIDATE, "candidate", problems)
    if d["sample"] is not None:
        _require(d["sample"], _SAMPLE, "sample", problems)
    if d["signature"] is not None:
        _require(d["signature"], _SIGNATURE, "signature", problems)
    if isinstance(d["result"], dict) and d["result"].get("outcome") not in OUTCOMES:
        problems.append(f"result.outcome {d['result'].get('outcome')!r} is not one of {OUTCOMES}")
    return problems


# --------------------------------------------------------------------- context


class _Ctx:
    """What the checks share: the record, the strictness, the code identities, the space."""

    def __init__(self, d: dict[str, Any], strict: bool) -> None:
        self.d = d
        self.strict = strict
        self.running = code_identity()
        issued = d["code"].get("digest")
        self.code_same = issued is not None and issued == self.running["digest"]
        self._space: CandidateSpace | None = None
        #: the library's own rule the record names (name, class and flag together), or None
        self.builtin = builtin_rule_from_record(d["rule"])

    @property
    def outcome(self) -> str:
        return self.d["result"]["outcome"]

    @property
    def claims_certified(self) -> bool:
        return bool(self.d["result"]["certified"])

    @property
    def space(self) -> CandidateSpace:
        if self._space is None:
            s = self.d["space"]
            self._space = CandidateSpace(
                s["name"], tuple(CandidateSpec.from_dict(c) for c in s["candidates"]))
        return self._space

    def text_status(self, equal: bool) -> str:
        """A text or trait that the running library would write differently: decisive (fail)
        when the code is the issuing code, otherwise only a difference."""
        if equal:
            return PASS
        return FAIL if (self.code_same or self.strict) else DIFFERS


def _canon(x: Any) -> bytes:
    return canonical_json(x)


def _summarise(items: Sequence[Any], limit: int = 6) -> str:
    shown = ", ".join(str(i) for i in items[:limit])
    return shown + (f", ... ({len(items)} in all)" if len(items) > limit else "")


def _verdict(problems: list[str], ok_detail: str) -> Result:
    return (FAIL, "; ".join(problems)) if problems else (PASS, ok_detail)


def _unexamined(ctx: _Ctx, what: str) -> Result:
    """The result of a check that needs the rule when the record's rule is not built in.

    The library certifies only its own rules. A record that says ``certified: true`` for any
    other rule fails the check, whatever else in it is consistent: the claim cannot be true
    and is never skipped. A record that claims nothing is certified may skip it.
    """
    rule = ctx.d["rule"]
    named = (f"rule {rule['name']!r} (class {rule['class']!r}, builtin flag {rule['builtin']!r})"
             " is not one of this library's own rules")
    if ctx.claims_certified:
        return FAIL, (f"the record says certified: true, but {named}, the only rules it "
                      f"certifies for: {what}")
    return SKIPPED, f"{named}: {what}; nothing in this record is certified"


# --------------------------------------------------------------- level 1 checks


def _check_schema(d: Mapping[str, Any]) -> Result:
    problems = structure_problems(d)
    if problems:
        return FAIL, "; ".join(problems)
    return PASS, f"{SCHEMA} version {d['schema_version']}, all required fields present"


def _check_digest(d: Mapping[str, Any]) -> Result:
    recomputed = content_digest(d)
    stored = d.get("digest")
    if recomputed == stored:
        return PASS, "the content digest matches (an integrity check: anyone can recompute it)"
    return FAIL, (f"the content digest does not match the fields: stored {str(stored)[:16]}..., "
                  f"recomputed {recomputed[:16]}...; the certificate was modified or is corrupt")


def _check_signature(d: Mapping[str, Any], key: bytes | None) -> Result:
    sig = d.get("signature")
    if key is None:
        if sig is None:
            return SKIPPED, "the certificate is not signed"
        return SKIPPED, "the certificate is signed; no key was supplied, so it was not checked"
    if sig is None:
        return FAIL, "a key was supplied but the certificate carries no signature"
    if sig.get("algorithm") != SIGNATURE_ALGORITHM:
        return FAIL, f"unsupported signature algorithm {sig.get('algorithm')!r}"
    expected = signature_value(key, d["digest"]).encode("ascii")
    if hmac.compare_digest(expected, str(sig.get("value")).encode("utf-8")):
        return PASS, (f"HMAC-SHA256 over the digest verifies (key id {sig.get('key_id')!r}); HMAC "
                      "is symmetric: whoever holds the key can also sign")
    return FAIL, "the signature does not verify under this key"


def _check_space(ctx: _Ctx) -> Result:
    d = ctx.d
    s = d["space"]
    problems: list[str] = []
    space = ctx.space
    if space.identity != s["identity"]:
        problems.append("space.identity is not the digest of space.candidates")
    if s["n_candidates"] != len(space):
        problems.append(f"space.n_candidates is {s['n_candidates']} for {len(space)} candidates")
    if s["n_families"] != space.n_families:
        problems.append(f"space.n_families is {s['n_families']}, the list has {space.n_families}")
    if s["families"] != list(space.families):
        problems.append("space.families does not match the families of the candidate list")
    if s["registered_default"] != (s["identity"] == REGISTERED_DEFAULT_SPACE_SHA256):
        problems.append("space.registered_default does not match the space identity")
    return _verdict(problems, f"{len(space)} candidates in {space.n_families} families; identity "
                    "is the digest of the list")


def _check_identities(ctx: _Ctx) -> Result:
    d = ctx.d
    problems: list[str] = []
    for where, ident in (("corpus", d["corpus"]), ("queries.identity", d["queries"]["identity"])):
        body = {k: v for k, v in ident.items() if k != "digest"}
        if digest(body) != ident["digest"]:
            problems.append(f"{where}.digest is not the digest of its fields")
    code = d["code"]
    if code["files"]:
        if code["digest"] != digest(code["files"]):
            problems.append("code.digest is not the digest of code.files")
    elif code["digest"] is not None:
        problems.append("code.digest is set but code.files is empty")
    ref = d["reference"]
    if ref["identity"].get("construct") != ref["construct"]:
        problems.append("reference.identity names a different construct than reference.construct")
    if d["queries"]["identity"]["n_docs"] != d["queries"]["n_pool"]:
        problems.append("queries.n_pool differs from the number of queries in queries.identity")
    return _verdict(problems, "combined digests of corpus, queries and code are consistent")


def _check_counts(ctx: _Ctx) -> Result:
    d = ctx.d
    k = d["space"]["n_candidates"]
    q, order, ev = d["queries"], d["queries"]["order"], d["evidence"]
    problems: list[str] = []
    for name, values in (("sizes.bytes_per_doc", d["sizes"]["bytes_per_doc"]),
                         ("sizes.vectors_per_doc", d["sizes"]["vectors_per_doc"]),
                         ("result.p_values", d["result"]["p_values"])):
        if len(values) != k:
            problems.append(f"{name} has {len(values)} entries for {k} candidates")
    for name in ("bytes_per_doc", "vectors_per_doc"):
        values = np.asarray(d["sizes"][name], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            problems.append(f"sizes.{name} must be finite and not negative")
    p = np.asarray(d["result"]["p_values"], dtype=np.float64)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        problems.append("result.p_values must lie in [0, 1]")
    if q["n_pool"] != ev["n_queries"] + len(ev["dropped_queries"]):
        problems.append("queries.n_pool is not evidence.n_queries plus the dropped queries")
    if order["n_available"] != ev["n_queries"]:
        problems.append("queries.order.n_available differs from evidence.n_queries")
    if order["n_used"] != q["n_used"]:
        problems.append("queries.order.n_used differs from queries.n_used")
    expected_used = ev["n_queries"] if order["budget"] is None else order["budget"]
    if q["n_used"] != expected_used:
        problems.append(f"queries.n_used is {q['n_used']}, the budget implies {expected_used}")
    if order["kind"] == "given" and order["seed"] is not None:
        problems.append("a query order 'given' cannot carry a seed")
    if order["kind"] == "seeded_permutation" and (order["seed"] is None or order["seed"] < 0):
        problems.append("a seeded query order needs a non-negative seed")
    if order["kind"] not in ("given", "seeded_permutation"):
        problems.append(f"unknown query order kind {order['kind']!r}")
    else:
        try:  # the recorded digest of the tested positions follows from kind, seed and budget
            tested_indices(order, order["n_available"])
        except ValueError as exc:
            problems.append(f"queries.order: {exc}")
    if len(set(ev["dropped_queries"])) != len(ev["dropped_queries"]):
        problems.append("evidence.dropped_queries repeats an id")
    if not 0 <= q["exact_duplicates"] < max(1, q["n_used"]):
        problems.append(f"queries.exact_duplicates is {q['exact_duplicates']}: it counts tested "
                        f"queries that repeat an earlier one, so it lies in [0, {q['n_used']})")
    problems += _norm_record_problems(d["norms"])
    return _verdict(problems, "sizes, p-values, query counts, the order record, the duplicate "
                    "count and the norm records are consistent")


def _norm_record_problems(norms: Mapping[str, Any]) -> list[str]:
    problems = []
    tol = norms["tolerance"]
    if not 0.0 < tol < 1.0:
        problems.append(f"norms.tolerance {tol} is outside (0, 1)")
        return problems
    for key in ("documents", "queries"):
        entry = norms[key]
        low, high = entry["norm_min"], entry["norm_max"]
        if (low is None) != (high is None):
            problems.append(f"norms.{key} has one extreme and not the other")
        elif low is None:
            if not entry["unit_norm"]:
                problems.append(f"norms.{key} has no vectors, so unit_norm cannot be false")
        elif not 0.0 <= low <= high:
            problems.append(f"norms.{key} extremes are not ordered: {low} > {high}")
        elif entry["unit_norm"] != (low >= 1.0 - tol and high <= 1.0 + tol):
            problems.append(f"norms.{key}.unit_norm does not follow from its extremes and the "
                            "tolerance")
    return problems


def _expected_level(ctx: _Ctx) -> float | None:
    d = ctx.d
    if ctx.builtin is None:
        return None
    if ctx.builtin.name == "fixed_sequence_betting":
        return d["error"] / ctx.space.n_families
    return d["error"] / len(ctx.space)  # bonferroni_betting


def _check_level(ctx: _Ctx) -> Result:
    expected = _expected_level(ctx)
    if expected is None:
        return _unexamined(ctx, "its level cannot be derived here")
    if ctx.d["level"] != expected:
        return FAIL, (f"level is {ctx.d['level']!r}; the rule's configuration derives "
                      f"{expected!r} from error {ctx.d['error']!r} and the candidate list")
    return PASS, f"level {expected!r} follows from error {ctx.d['error']!r} and the list"


def _check_outcome(ctx: _Ctx) -> Result:
    d = ctx.d
    r = d["result"]
    outcome = ctx.outcome
    k = d["space"]["n_candidates"]
    problems: list[str] = []
    rule = d["rule"]
    if rule["certifying"] != (rule["builtin"] and rule["declared_certifying"]):
        problems.append("rule.certifying must be true exactly when the rule is built in and "
                        "declares itself certifying: the library certifies only its own rules")
    if rule["builtin"] and ctx.builtin is None:
        problems.append("rule.builtin is true, but rule.name and rule.class are not those of one "
                        "of the library's own rules")
    if any(not 0 <= p < k for p in r["certified_positions"]) or (
            r["certified_positions"] != sorted(set(r["certified_positions"]))):
        problems.append("result.certified_positions must be distinct positions in list order")
    if outcome == "select":
        sel = r["selected"]
        if sel is None or not 0 <= sel < k or sel not in r["certified_positions"]:
            problems.append("a select outcome needs a selected position that is certified")
        if r["certified"] != d["rule"]["certifying"]:
            problems.append("result.certified must equal rule.certifying for a select outcome")
        if d["candidate"] is None:
            problems.append("a select outcome needs a candidate record")
        if d["sample"] is None:
            problems.append("a select outcome needs sample statistics")
    else:
        if r["certified"] or r["certified_positions"]:
            problems.append(f"a {outcome} outcome cannot be certified or list certified positions")
        if d["sample"] is not None:
            problems.append(f"a {outcome} outcome has no sample statistics")
        if outcome == "abstain":
            if r["selected"] is not None or d["candidate"] is not None:
                problems.append("an abstain outcome selects nothing and has no candidate record")
        elif r["selected"] != NOT_IN_SPACE or d["candidate"] is None:
            problems.append("a fallback outcome needs selected == -1 and a candidate record")
    return _verdict(problems, f"outcome {outcome!r} agrees with selection, certification and "
                    "candidate record")


def _check_decision(ctx: _Ctx) -> Result:
    d = ctx.d
    p = d["result"]["p_values"]
    families = [c["family"] for c in d["space"]["candidates"]]
    if ctx.builtin is None:
        return _unexamined(ctx, "the decision step cannot be recomputed from the p-values")
    if ctx.builtin.name == "fixed_sequence_betting":
        rejected = fixed_sequence_by_family(p, families, d["level"])
    else:  # bonferroni_betting
        rejected = [k for k, pk in enumerate(p) if pk <= d["level"]]
    chosen = smallest(rejected, d["sizes"]["bytes_per_doc"], d["sizes"]["vectors_per_doc"])
    r = d["result"]
    problems = []
    if list(r["certified_positions"]) != list(rejected):
        problems.append(f"certified_positions {r['certified_positions']} differ from the "
                        f"{rejected} that the stored p-values give at level {d['level']!r}")
    if ctx.outcome == "select":
        if r["selected"] != chosen:
            problems.append(f"selected {r['selected']} is not the smallest certified candidate "
                            f"({chosen}) by (bytes, vectors, position)")
    elif rejected:
        problems.append(f"the stored p-values certify {rejected}, which a {ctx.outcome} "
                        "outcome does not allow")
    return _verdict(problems, "certified positions and the selection follow from the stored "
                    "p-values, families, sizes and level")


def _check_candidate(ctx: _Ctx) -> Result:
    d = ctx.d
    cand = d["candidate"]
    if cand is None:
        return SKIPPED, "no candidate (abstain)"
    problems: list[str] = []
    if digest(cand["pipeline"]) != cand["pipeline_sha256"]:
        problems.append("candidate.pipeline_sha256 is not the digest of candidate.pipeline")
    stages = cand["pipeline"].get("stages")
    if not isinstance(stages, list):
        problems.append("candidate.pipeline has no list of stages")
    elif len(cand["fitted_state"]) != len(stages):
        problems.append(f"candidate.fitted_state has {len(cand['fitted_state'])} entries for "
                        f"{len(stages)} stages")
    sel = d["result"]["selected"]
    if ctx.outcome == "select" and isinstance(sel, int) and 0 <= sel < len(ctx.space):
        listed = d["space"]["candidates"][sel]["pipeline"]
        if _canon(cand["pipeline"]) != _canon(listed):
            problems.append("candidate.pipeline is not the pipeline of the selected candidate "
                            f"in the list (position {sel})")
    acc = cand["bytes"]
    missing = [k for k in _BYTES if not _int(acc.get(k))]
    if missing:
        problems.append(f"candidate.bytes lacks integer field(s) {missing}")
    else:
        n_docs = max(1, acc["n_docs"])
        selection = (acc["code_bytes"] + acc["codec_overhead_bytes"]) / n_docs
        if acc.get("selection_bytes_per_doc") != selection:
            problems.append("candidate.bytes.selection_bytes_per_doc is not (code_bytes + "
                            "codec_overhead_bytes) / n_docs")
        total = (acc["code_bytes"] + acc["codec_overhead_bytes"] + acc["fitted_state_bytes"]
                 + acc["offsets_bytes"]) / n_docs
        if acc.get("total_bytes_per_doc") != total:
            problems.append("candidate.bytes.total_bytes_per_doc does not follow from its parts")
        corpus = d["corpus"]
        if acc["n_docs"] != corpus["n_docs"]:
            problems.append("candidate.bytes.n_docs differs from the corpus identity")
        if acc["offsets_bytes"] != 8 * (corpus["n_docs"] + 1):
            problems.append("candidate.bytes.offsets_bytes is not 8 bytes per offset of the corpus")
        if acc["float32_bytes_before"] != corpus["n_vectors"] * corpus["dimension"] * 4:
            problems.append("candidate.bytes.float32_bytes_before is not the float32 size of the "
                            "corpus identity")
        if (ctx.outcome == "select" and isinstance(sel, int) and 0 <= sel < len(ctx.space)
                and acc.get("selection_bytes_per_doc") != d["sizes"]["bytes_per_doc"][sel]):
            problems.append("candidate.bytes.selection_bytes_per_doc differs from "
                            "sizes.bytes_per_doc of the selected candidate")
    return _verdict(problems, "pipeline digest, selected pipeline, fitted-state list and byte "
                    "figures are consistent")


def _check_sample(ctx: _Ctx) -> Result:
    s = ctx.d["sample"]
    if s is None:
        return SKIPPED, f"no sample statistics ({ctx.outcome})"
    problems = []
    if not s["baseline_mean"] > 0:
        problems.append("sample.baseline_mean is not positive")
    elif not np.isclose(s["retention_point"], s["mean"] / s["baseline_mean"], rtol=RTOL, atol=0.0):
        problems.append("sample.retention_point is not sample.mean / sample.baseline_mean")
    if s["n"] != ctx.d["queries"]["n_used"]:
        problems.append("sample.n differs from queries.n_used")
    return _verdict(problems, "sample statistics are internally consistent (descriptive, "
                    "optimistically biased, not certified)")


def _check_rule_text(ctx: _Ctx) -> Result:
    d = ctx.d
    rule = ctx.builtin
    if rule is None:
        return _unexamined_texts(ctx)
    record = rule_record(rule)
    wanted = {
        "rule": record,
        "statement": statement_for(record),
        "assumptions": list(assumptions_of(rule)),
        "caveats": caveats_for(d),
    }
    differing = [k for k, v in wanted.items() if _canon(d[k]) != _canon(v)]
    status = ctx.text_status(not differing)
    if status == PASS:
        return PASS, "rule, statement, assumptions and caveats are those of the running library"
    why = ("the code is the code that issued the certificate, so a different text is not a "
           "version difference" if ctx.code_same else "the running code differs from the issuing "
           "code, so this may be a version difference")
    return status, f"{', '.join(differing)} differ from what the running library writes ({why})"


def _unexamined_texts(ctx: _Ctx) -> Result:
    """``rule.text`` for a rule that is not built in: only what the library itself writes.

    The rule's configuration and its assumptions cannot be compared (the library has not
    examined the rule), but the statement is a fixed text, every assumption must carry the
    label that says whose it is, and the caveats follow from the record's own fields.
    """
    d = ctx.d
    if ctx.claims_certified:
        return _unexamined(ctx, "its statement and texts cannot be those of a certified rule")
    differing = []
    if d["statement"] != STATEMENT_UNEXAMINED:
        differing.append("statement")
    if not all(a.startswith(DECLARED_PREFIX) for a in d["assumptions"]):
        differing.append("assumptions")
    if _canon(d["caveats"]) != _canon(caveats_for(d)):
        differing.append("caveats")
    status = ctx.text_status(not differing)
    if status == PASS:
        return PASS, ("the rule is not one of this library's own, so only the fixed statement, "
                      "the label on the assumptions and the caveats were compared; the rule's "
                      "configuration was not examined and nothing in this record is certified")
    why = ("the code is the code that issued the certificate, so a different text is not a "
           "version difference" if ctx.code_same else "the running code differs from the "
           "issuing code, so this may be a version difference")
    return status, f"{', '.join(differing)} differ from what the running library writes ({why})"


def _check_traits(ctx: _Ctx) -> Result:
    cand = ctx.d["candidate"]
    if cand is None:
        return SKIPPED, "no candidate (abstain)"
    try:
        pipeline = Pipeline.from_dict(cand["pipeline"])
    except (ValueError, TypeError, KeyError) as exc:  # a stage this process does not know
        return SKIPPED, f"the pipeline cannot be rebuilt here ({type(exc).__name__}: {exc})"
    expected = pipeline_traits(pipeline).to_dict()
    status = ctx.text_status(_canon(cand["traits"]) == _canon(expected))
    if status == PASS:
        return PASS, "the recorded structural traits are those the running library derives"
    return status, ("the recorded traits (basis behaviour of the pipeline) differ from what the "
                    "running library derives from the pipeline specification")


def _check_code_files(ctx: _Ctx) -> Result:
    issued, running = ctx.d["code"], ctx.running
    if issued["digest"] is not None and issued["digest"] == running["digest"]:
        return PASS, f"the {len(issued['files'])} hashed source files are those that issued it"
    status = FAIL if ctx.strict else DIFFERS
    if issued["digest"] is None or running["digest"] is None:
        which = "the certificate" if issued["digest"] is None else "the running package"
        return status, (f"the source identity of {which} is unavailable "
                        f"({issued.get('note') or running.get('note') or 'unreadable sources'}); "
                        "it cannot be compared")
    a, b = issued["files"], running["files"]
    changed = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
    added = sorted(b.keys() - a.keys())
    removed = sorted(a.keys() - b.keys())
    parts = []
    for label, names in (("changed", changed), ("added since", added), ("removed since", removed)):
        if names:
            parts.append(f"{label}: {_summarise(names, 4)}")
    return status, ("the running code differs from the code that issued the certificate ("
                    + "; ".join(parts) + "); a different version is not tampering by itself, but "
                    "a replay under different code is weaker evidence")


def _check_code_libraries(ctx: _Ctx) -> Result:
    issued, running = ctx.d["code"], ctx.running
    diffs = [f"{k} {issued[k]} (issued) vs {running[k]} (running)"
             for k in ("python", "numpy", "scipy") if issued[k] != running[k]]
    if diffs:
        return DIFFERS, ("; ".join(diffs) + "; the code identity covers source files, not the "
                         "libraries beneath them, which is why replay compares p-values at a "
                         "relative tolerance")
    return PASS, "Python, NumPy and SciPy versions are those recorded"


# --------------------------------------------------------------- level 2 checks


def _check_evidence_digest(ctx: _Ctx, evidence: Evidence) -> Result:
    got = evidence.digest()
    if got == ctx.d["evidence"]["digest"]:
        return PASS, "the supplied evidence is the evidence the decision was computed from"
    return FAIL, (f"the supplied evidence has digest {got[:16]}..., the certificate records "
                  f"{ctx.d['evidence']['digest'][:16]}...")


def _check_evidence_fields(ctx: _Ctx, evidence: Evidence) -> Result:
    d = ctx.d
    ev = d["evidence"]
    problems: list[str] = []
    if ev["n_queries"] != evidence.n_queries:
        problems.append("evidence.n_queries differs from the evidence")
    if ev["dropped_queries"] != list(evidence.dropped_queries):
        problems.append("evidence.dropped_queries differs from the evidence")
    if not np.isclose(ev["baseline_mean"], float(np.mean(evidence.baseline)), rtol=RTOL, atol=0.0):
        problems.append("evidence.baseline_mean differs from the evidence")
    if d["sizes"]["bytes_per_doc"] != evidence.bytes_per_doc.tolist():
        problems.append("sizes.bytes_per_doc differs from the evidence")
    if d["sizes"]["vectors_per_doc"] != evidence.vectors_per_doc.tolist():
        problems.append("sizes.vectors_per_doc differs from the evidence")
    if d["reference"]["construct"] != evidence.construct:
        problems.append(f"reference.construct {d['reference']['construct']!r} differs from the "
                        f"evidence's {evidence.construct!r}")
    if d["reference"]["metric"] != evidence.metric:
        problems.append("reference.metric differs from the evidence")
    if d["space"]["identity"] != evidence.space_identity:
        problems.append("space.identity differs from the space the evidence was collected on")
    if d["space"]["n_candidates"] != evidence.n_candidates:
        problems.append("space.n_candidates differs from the evidence")
    sel = d["result"]["selected"]
    if (ctx.outcome == "select" and evidence.fitted and isinstance(sel, int)
            and 0 <= sel < len(evidence.fitted)
            and d["candidate"]["fitted_state"] != list(evidence.fitted[sel])):
        problems.append("candidate.fitted_state differs from the fitted-state digests the "
                        "evidence recorded for the selected candidate")
    return _verdict(problems, "the certificate's copies of the evidence's values equal the "
                    "supplied evidence")


def _check_evidence_sample(ctx: _Ctx, evidence: Evidence) -> Result:
    s = ctx.d["sample"]
    if s is None:
        return SKIPPED, f"no sample statistics ({ctx.outcome})"
    sel = ctx.d["result"]["selected"]
    fresh = sample_record(evidence, sel, ctx.d["queries"]["order"])
    problems = []
    if s["n"] != fresh["n"]:
        problems.append("sample.n differs from the evidence")
    for key in ("mean", "baseline_mean", "retention_point"):
        if not np.isclose(s[key], fresh[key], rtol=RTOL, atol=0.0):
            problems.append(f"sample.{key} differs from the evidence (stored {s[key]!r}, "
                            f"recomputed {fresh[key]!r})")
    return _verdict(problems, "sample statistics equal their recomputation from the evidence")


def _check_replay(ctx: _Ctx, evidence: Evidence) -> Result:
    d = ctx.d
    rule = ctx.builtin
    if rule is None:
        return _unexamined(ctx, "the decision cannot be replayed")
    order = d["queries"]["order"]
    fallback = d["candidate"]["pipeline"] if ctx.outcome == "fallback" else None
    try:
        replay = select(ctx.space, evidence, target=d["target"], error=d["error"], rule=rule,
                        fallback=fallback, query_budget=order["budget"],
                        order_seed=order["seed"])
    except (ValueError, TypeError) as exc:
        return FAIL, ("select() could not be re-run on the supplied evidence: "
                      f"{type(exc).__name__}: {exc}")
    r = d["result"]
    problems: list[str] = []

    def same(label: str, got: Any, stored: Any) -> None:
        if got != stored:
            problems.append(f"{label}: replay {got!r}, certificate {stored!r}")

    same("outcome", replay.outcome.value, r["outcome"])
    same("certified", replay.certified, r["certified"])
    same("selected", None if replay.selected is None else replay.selected.position, r["selected"])
    same("certified_positions", list(replay.certified_positions), r["certified_positions"])
    same("level", replay.level, d["level"])
    same("n_queries", replay.n_queries, d["queries"]["n_used"])
    same("query_order", dict(replay.query_order), order)
    p_new = np.asarray(replay.p_values, dtype=np.float64)
    p_old = np.asarray(r["p_values"], dtype=np.float64)
    if p_new.shape != p_old.shape:
        problems.append(f"p_values: replay has {p_new.size}, certificate {p_old.size}")
    else:
        close = np.isclose(p_new, p_old, rtol=RTOL, atol=0.0)
        if not close.all():
            where = np.flatnonzero(~close)
            worst = float(np.max(np.abs(p_new[where] - p_old[where]) / np.abs(p_old[where]).clip(1e-300)))
            problems.append(f"p_values differ at positions {_summarise(where.tolist())} "
                            f"(relative difference up to {worst:.3g}; tolerance {RTOL})")
    suffix = "" if ctx.code_same else " (the running code differs from the issuing code)"
    if problems:
        return FAIL, "the decision recomputed from the evidence differs: " + "; ".join(
            problems) + suffix
    return PASS, ("select() re-run on the evidence gives the recorded decision (p-values at "
                  f"relative tolerance {RTOL}, everything else exactly)" + suffix)


# --------------------------------------------------------------- level 3 checks


def _identity_diff(live: dict[str, Any], stored: Mapping[str, Any]) -> list[str]:
    body = {k: v for k, v in stored.items() if k != "digest"}
    return sorted(k for k in set(live) | set(body) if live.get(k) != body.get(k))


def _check_binding_identity(live_corpus: MultiVectorCorpus, stored: Mapping[str, Any],
                            what: str) -> Result:
    ident = corpus_identity(live_corpus)
    bad = _identity_diff(ident.to_dict(), stored)
    if bad or ident.digest() != stored["digest"]:
        return FAIL, (f"the supplied {what} is not the one certified: differs in "
                      f"{bad or ['combined digest']}")
    return PASS, f"the supplied {what} has the recorded identity ({ident.n_docs} items)"


def _live_norm_problems(live: MultiVectorCorpus, stored: Mapping[str, Any], where: str) -> list[str]:
    """The recorded norm entry against the one the supplied vectors give (they are the
    certified vectors, since their identity matched, so the two must agree)."""
    now = norm_record(live)
    problems = []
    if now["unit_norm"] != stored["unit_norm"]:
        problems.append(f"norms.{where}.unit_norm is {stored['unit_norm']}, the supplied "
                        f"vectors give {now['unit_norm']}")
    for key in ("norm_min", "norm_max"):
        a, b = now[key], stored[key]
        if (a is None) != (b is None) or (a is not None and not np.isclose(
                a, b, rtol=1e-9, atol=0.0)):
            problems.append(f"norms.{where}.{key} is {b!r}, the supplied vectors give {a!r}")
    return problems


def _check_binding_corpus(ctx: _Ctx, corpus: MultiVectorCorpus) -> Result:
    d = ctx.d
    status, detail = _check_binding_identity(corpus, d["corpus"], "corpus")
    if status != PASS:
        return status, detail
    problems = _live_norm_problems(corpus, d["norms"]["documents"], "documents")
    return _verdict(problems, detail + "; its norm record is the recorded one")


def _check_binding_queries(ctx: _Ctx, queries: MultiVectorCorpus) -> Result:
    d = ctx.d
    status, detail = _check_binding_identity(queries, d["queries"]["identity"], "query set")
    if status != PASS:
        return status, detail
    problems = _live_norm_problems(queries, d["norms"]["queries"], "queries")
    try:
        repeats = exact_duplicates(queries, d["evidence"]["dropped_queries"],
                                   d["queries"]["order"])
    except ValueError as exc:
        problems.append(f"the exact-duplicate count could not be recomputed: {exc}")
    else:
        if repeats != d["queries"]["exact_duplicates"]:
            problems.append(f"queries.exact_duplicates is {d['queries']['exact_duplicates']}, the "
                            f"supplied queries give {repeats}")
    return _verdict(problems, detail + "; its exact-duplicate count and norm record are the "
                    "recorded ones")


def _check_binding_reference(reference: Reference, d: Mapping[str, Any]) -> Result:
    problems = []
    if reference.construct != d["reference"]["construct"]:
        problems.append(f"construct {reference.construct!r} differs from "
                        f"{d['reference']['construct']!r}")
    if _canon(reference.identity()) != _canon(d["reference"]["identity"]):
        problems.append("the reference's identity (labels digest or depth) differs from the "
                        "certified one")
    return _verdict(problems, "the supplied reference has the recorded identity")


def _check_binding_pipeline(ctx: _Ctx, pipeline: Pipeline) -> Result:
    cand = ctx.d["candidate"]
    if cand is None:
        return FAIL, (f"the certificate records no deployed candidate ({ctx.outcome}); a "
                      "pipeline was supplied")
    problems = []
    if pipeline_identity(pipeline) != cand["pipeline_sha256"]:
        problems.append("the pipeline's specification differs from the certified one")
    try:
        fitted = pipeline_fitted_identity(pipeline)
    except RuntimeError as exc:
        problems.append(f"the pipeline is not fitted ({exc})")
    else:
        if fitted != list(cand["fitted_state"]):
            bad = [i for i, (a, b) in enumerate(zip(fitted, cand["fitted_state"])) if a != b]
            problems.append("the per-stage fitted-state digests differ from the certified ones"
                            + (f" (stage(s) {bad})" if len(fitted) == len(cand["fitted_state"])
                               else f" ({len(fitted)} stages, {len(cand['fitted_state'])} certified)"))
    return _verdict(problems, "the supplied pipeline has the certified specification and "
                    "fitted state")


def _check_binding_codes(ctx: _Ctx, compressed: CompressedCorpus) -> Result:
    cand = ctx.d["candidate"]
    if cand is None:
        return FAIL, (f"the certificate records no deployed candidate ({ctx.outcome}); a "
                      "compressed corpus was supplied")
    problems = []
    if array_digest(compressed.codes) != cand["codes_sha256"]:
        problems.append("the codes differ from the certified codes")
    if array_digest(compressed.offsets) != cand["offsets_sha256"]:
        problems.append("the offsets differ from the certified offsets")
    if strings_digest(compressed.ids) != ctx.d["corpus"]["ids_sha256"]:
        problems.append("the document ids differ from those of the certified corpus")
    if compressed.pipeline is not None and _canon(compressed.pipeline) != _canon(cand["pipeline"]):
        problems.append("the compressed corpus records a different pipeline than the certified one")
    return _verdict(problems, "the supplied compressed corpus has the certified codes, offsets "
                    "and document ids")


def _check_binding_bytes(ctx: _Ctx, pipeline: Pipeline, compressed: CompressedCorpus) -> Result:
    cand = ctx.d["candidate"]
    if cand is None:
        return FAIL, "the certificate records no deployed candidate"
    live = byte_accounting(compressed, pipeline).to_dict()
    bad = sorted(k for k in set(live) | set(cand["bytes"]) if live.get(k) != cand["bytes"].get(k))
    if bad:
        return FAIL, f"the byte figures differ from those recomputed from the objects: {bad}"
    return PASS, "the certified byte figures equal their recomputation from the objects"


def _check_evidence_queries(ctx: _Ctx, queries: MultiVectorCorpus, evidence: Evidence) -> Result:
    dropped = set(evidence.dropped_queries)
    ids = list(queries.ids)
    problems = []
    if not dropped <= set(ids):
        problems.append("the evidence lists dropped queries that are not among the queries")
    if tuple(q for q in ids if q not in dropped) != evidence.query_ids:
        problems.append("the evidence's query ids are not the supplied queries without the "
                        "dropped ones, in order")
    return _verdict(problems, "the evidence was collected on the supplied queries")


# ------------------------------------------------------------------ the driver


def _run(out: list[Check], name: str, level: str, fn: Callable[[], Result]) -> None:
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a malformed certificate is reported, never raised
        status, detail = FAIL, f"could not be evaluated: {type(exc).__name__}: {exc}"
    out.append(Check(name, status, detail, level))


def _skip(out: list[Check], name: str, level: str, why: str) -> None:
    out.append(Check(name, SKIPPED, why, level))


def run_checks(
    d: dict[str, Any],
    *,
    evidence: Evidence | None = None,
    corpus: MultiVectorCorpus | None = None,
    queries: MultiVectorCorpus | None = None,
    reference: Reference | None = None,
    pipeline: Pipeline | None = None,
    compressed: CompressedCorpus | None = None,
    key: bytes | None = None,
    strict_code: bool = False,
) -> VerificationReport:
    """All checks for the certificate record ``d``; see the module docstring."""
    out: list[Check] = []
    _run(out, "schema", INTEGRITY, lambda: _check_schema(d))
    _run(out, "digest", INTEGRITY, lambda: _check_digest(d))
    _run(out, "signature", INTEGRITY, lambda: _check_signature(d, key))
    if structure_problems(d):
        _skip(out, "remaining", INTEGRITY, "the structure is invalid: no further check was run")
        return VerificationReport(tuple(out))
    ctx = _Ctx(d, strict_code)

    _run(out, "consistency.space", INTEGRITY, lambda: _check_space(ctx))
    _run(out, "consistency.identities", INTEGRITY, lambda: _check_identities(ctx))
    _run(out, "consistency.counts", INTEGRITY, lambda: _check_counts(ctx))
    _run(out, "consistency.level", INTEGRITY, lambda: _check_level(ctx))
    _run(out, "consistency.outcome", INTEGRITY, lambda: _check_outcome(ctx))
    _run(out, "consistency.decision", INTEGRITY, lambda: _check_decision(ctx))
    _run(out, "consistency.candidate", INTEGRITY, lambda: _check_candidate(ctx))
    _run(out, "consistency.sample", INTEGRITY, lambda: _check_sample(ctx))
    _run(out, "rule.text", INTEGRITY, lambda: _check_rule_text(ctx))
    _run(out, "candidate.traits", INTEGRITY, lambda: _check_traits(ctx))
    _run(out, "code.files", INTEGRITY, lambda: _check_code_files(ctx))
    _run(out, "code.libraries", INTEGRITY, lambda: _check_code_libraries(ctx))

    if evidence is None:
        for name in ("evidence.digest", "evidence.fields", "evidence.sample", "replay.decision"):
            _skip(out, name, REPLAY, "no evidence supplied")
    else:
        _run(out, "evidence.digest", REPLAY, lambda: _check_evidence_digest(ctx, evidence))
        _run(out, "evidence.fields", REPLAY, lambda: _check_evidence_fields(ctx, evidence))
        _run(out, "evidence.sample", REPLAY, lambda: _check_evidence_sample(ctx, evidence))
        _run(out, "replay.decision", REPLAY, lambda: _check_replay(ctx, evidence))

    if corpus is None:
        _skip(out, "binding.corpus", BINDING, "no corpus supplied")
    else:
        _run(out, "binding.corpus", BINDING, lambda: _check_binding_corpus(ctx, corpus))
    if queries is None:
        _skip(out, "binding.queries", BINDING, "no queries supplied")
    else:
        _run(out, "binding.queries", BINDING, lambda: _check_binding_queries(ctx, queries))
    if queries is None or evidence is None:
        _skip(out, "binding.evidence_queries", BINDING, "needs both the queries and the evidence")
    else:
        _run(out, "binding.evidence_queries", BINDING,
             lambda: _check_evidence_queries(ctx, queries, evidence))
    if reference is None:
        _skip(out, "binding.reference", BINDING, "no reference supplied")
    else:
        _run(out, "binding.reference", BINDING, lambda: _check_binding_reference(reference, d))
    if pipeline is None:
        _skip(out, "binding.pipeline", BINDING, "no pipeline supplied")
    else:
        _run(out, "binding.pipeline", BINDING, lambda: _check_binding_pipeline(ctx, pipeline))
    if compressed is None:
        _skip(out, "binding.codes", BINDING, "no compressed corpus supplied")
    else:
        _run(out, "binding.codes", BINDING, lambda: _check_binding_codes(ctx, compressed))
    if pipeline is None or compressed is None:
        _skip(out, "binding.bytes", BINDING, "needs both the pipeline and the compressed corpus")
    else:
        _run(out, "binding.bytes", BINDING,
             lambda: _check_binding_bytes(ctx, pipeline, compressed))
    return VerificationReport(tuple(out), tuple(_notes(ctx)))


def _notes(ctx: _Ctx) -> list[str]:
    """Sentences about the whole record that the checks alone do not say."""
    if ctx.builtin is not None:
        return []
    cls = ctx.d["rule"]["class"]
    if ctx.claims_certified:
        return [(f"this record says certified: true for a rule (class {cls}) that is not one of "
                 "this library's own; the library certifies only its own rules, so the claim is "
                 "refused")]
    return [(f"this record names a rule (class {cls}) that is not one of this library's own. The "
             "library has not examined it: nothing in this record is certified, and the checks "
             "that recompute the level and the decision, and the replay, were skipped")]


__all__ = ["RTOL", "run_checks", "structure_problems"]
