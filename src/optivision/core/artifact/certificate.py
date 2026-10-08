"""The certificate: a record of one selection decision, with checks on it.

``create_certificate`` turns a :class:`~optivision.core.selection.Decision` and the objects it
was computed from into an immutable :class:`Certificate`; ``verify_certificate`` checks one;
``sign_certificate`` adds an HMAC. The format is specified in
``docs/package/CERTIFICATE_SPEC.md`` (schema ``optivision-certificate``, version 1).

What a certificate is, and is not
---------------------------------
It answers: *was this compression decision certified under the stated configuration, and does
this record match the index it was issued for?* It carries the selection rule's own statement
and its assumptions A1-A6 and conditions C1-C4 verbatim, and nothing wider. It is not a
statement that the compressed index keeps retrieval quality, and it does not transfer to another
corpus, query distribution or candidate list. Under the ``agreement@...`` construct "retention"
means agreement with the uncompressed ranking, which is not retrieval quality; the certificate
says so.

* The content ``digest`` is an integrity check, not security: anyone who can edit the file can
  recompute it.
* The optional ``signature`` is an HMAC-SHA256 over the digest. It is symmetric (whoever can
  verify can also sign) and gives no non-repudiation. Without a key the format gives integrity
  checking only.
* A party who rewrites a certificate *and* supplies matching forged evidence and objects is not
  detected by any level of verification; only a signature under a key that party lacks closes
  that gap.

A certificate is issued for every outcome (``select``, ``abstain``, ``fallback``); an
abstention is as checkable as a selection. ``certified`` is true only for ``select`` under one
of the library's own certifying rules: a rule written by anyone else is recorded with its class,
the library makes no statement about it, and the certificate says so and is never ``certified``.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from optivision.compose import CompressedCorpus, Pipeline
from optivision.core.evaluate.evidence import Evidence
from optivision.core.evaluate.reference import Reference
from optivision.core.identity import array_digest, canonical_json
from optivision.core.pipeline import pipeline_identity, pipeline_spec
from optivision.core.selection.decision import Decision, Outcome
from optivision.core.selection.spaces import (
    NOT_IN_SPACE,
    CandidateSpace,
    CandidateSpec,
    freeze_json,
    thaw_json,
)
from optivision.core.stages.accounting import byte_accounting
from optivision.core.stages.state import pipeline_fitted_identity
from optivision.core.stages.traits import pipeline_traits
from optivision.core.vectors import EncoderIdentity, corpus_identity, validate_corpus
from optivision.representation import MultiVectorCorpus

from .codeid import code_identity, package_info
from .record import (
    FIELD_ORDER,
    SCHEMA,
    SCHEMA_VERSION,
    SIGNATURE_ALGORITHM,
    caveats_for,
    content_digest,
    exact_duplicates,
    norms_record,
    sample_record,
    signature_value,
    statement_for,
)
from .report import Check, VerificationReport
from .verification import run_checks

__all__ = [
    "Certificate",
    "CertificateFormatError",
    "Check",
    "VerificationReport",
    "create_certificate",
    "sign_certificate",
    "verify_certificate",
]

_CREATED_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class CertificateFormatError(ValueError):
    """A file or mapping cannot be read as a certificate record, or a field is missing."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON in a certificate")


@dataclass(frozen=True, eq=False)
class Certificate:
    """An immutable certificate record. ``data`` is its read-only content.

    Build one with :func:`create_certificate`, :meth:`from_dict`, :meth:`from_json` or
    :meth:`load`. Reading a record does not check it: an altered or malformed record loads, and
    :func:`verify_certificate` reports what is wrong. The typed accessors raise
    :class:`CertificateFormatError` for a field that is missing.
    """

    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.data, Mapping):
            raise CertificateFormatError(
                f"a certificate is a JSON object, got {type(self.data).__name__}")
        try:
            frozen = freeze_json(self.data)
        except (TypeError, ValueError) as exc:
            raise CertificateFormatError(f"not a JSON-native certificate record: {exc}") from exc
        except RecursionError:
            raise CertificateFormatError("not a certificate record: nested too deeply") from None
        object.__setattr__(self, "data", frozen)

    # ------------------------------------------------------------------ access

    def _at(self, *path: str | int) -> Any:
        node: Any = self.data
        for key in path:
            try:
                node = node[key]
            except (KeyError, IndexError, TypeError):
                where = ".".join(str(p) for p in path)
                raise CertificateFormatError(f"the certificate has no field {where}") from None
        return node

    @property
    def schema_version(self) -> int:
        return int(self._at("schema_version"))

    @property
    def created_utc(self) -> str:
        return str(self._at("created_utc"))

    @property
    def digest(self) -> str:
        """The content digest (SHA-256 of every field but ``digest`` and ``signature``)."""
        return str(self._at("digest"))

    @property
    def signed(self) -> bool:
        return self.data.get("signature") is not None

    @property
    def signature(self) -> dict[str, Any] | None:
        sig = self.data.get("signature")
        return None if sig is None else thaw_json(sig)

    @property
    def outcome(self) -> Outcome:
        return Outcome(self._at("result", "outcome"))

    @property
    def certified(self) -> bool:
        """True only for a ``select`` outcome under one of the library's own certifying rules."""
        return bool(self._at("result", "certified"))

    @property
    def target(self) -> float:
        return float(self._at("target"))

    @property
    def error(self) -> float:
        return float(self._at("error"))

    @property
    def level(self) -> float:
        """The per-test level the rule derived from ``error`` and the candidate list."""
        return float(self._at("level"))

    @property
    def metric(self) -> str:
        return str(self._at("reference", "metric"))

    @property
    def construct(self) -> str:
        """``labels`` or ``agreement@1``: what the metric was measured against."""
        return str(self._at("reference", "construct"))

    @property
    def rule_name(self) -> str:
        return str(self._at("rule", "name"))

    @property
    def space_identity(self) -> str:
        return str(self._at("space", "identity"))

    @property
    def n_queries(self) -> int:
        """The number of queries the rule tested."""
        return int(self._at("queries", "n_used"))

    @property
    def p_values(self) -> tuple[float, ...]:
        return tuple(float(p) for p in self._at("result", "p_values"))

    @property
    def certified_positions(self) -> tuple[int, ...]:
        return tuple(int(p) for p in self._at("result", "certified_positions"))

    @property
    def selected_position(self) -> int | None:
        """The list position of the deployed candidate; ``-1`` for a fallback; ``None`` for an
        abstention."""
        position = self._at("result", "selected")
        return None if position is None else int(position)

    @property
    def selected(self) -> CandidateSpec | None:
        """The deployed candidate as a ``CandidateSpec`` (``None`` on abstention). For a
        fallback it has position ``-1`` and family ``"fallback"``."""
        position = self.selected_position
        if position is None:
            return None
        if position == NOT_IN_SPACE:
            return CandidateSpec(NOT_IN_SPACE, "fallback",
                                 thaw_json(self._at("candidate", "pipeline")))
        return CandidateSpec.from_dict(thaw_json(self._at("space", "candidates", position)))

    @property
    def basis_sensitive(self) -> bool | None:
        """Whether the deployed pipeline is sensitive to the coordinate basis (``None`` on
        abstention). A structural property of the code, not a statement about quality."""
        candidate = self.data.get("candidate")
        return None if candidate is None else bool(candidate["traits"]["basis_sensitive"])

    @property
    def statement(self) -> str:
        return str(self._at("statement"))

    @property
    def assumptions(self) -> tuple[str, ...]:
        return tuple(str(a) for a in self._at("assumptions"))

    @property
    def caveats(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self._at("caveats"))

    # ------------------------------------------------------------------ dunder

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Certificate) and self.data == other.data

    def __hash__(self) -> int:
        return hash(str(self.data.get("digest")))

    def __repr__(self) -> str:
        try:
            return (f"Certificate(outcome={self.outcome.value!r}, certified={self.certified}, "
                    f"target={self.target}, error={self.error}, digest={self.digest[:12]!r}"
                    f"{', signed' if self.signed else ''})")
        except (CertificateFormatError, ValueError, TypeError):
            return "Certificate(<malformed record>)"

    # ----------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        """The record as plain JSON-native dicts and lists (a fresh deep copy)."""
        return thaw_json(self.data)

    def to_json(self, indent: int | None = 2) -> str:
        """The record as JSON text. Fields come in a readable order, nested keys sorted; the
        digest does not depend on this layout (it is computed over the canonical JSON)."""
        plain = self.to_dict()
        keys = [k for k in FIELD_ORDER if k in plain] + sorted(set(plain) - set(FIELD_ORDER))
        ordered = {k: json.loads(canonical_json(plain[k])) for k in keys}
        return json.dumps(ordered, indent=indent, ensure_ascii=True, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Certificate:
        """A certificate from a mapping, without checking it (see :func:`verify_certificate`)."""
        return cls(data)

    @classmethod
    def from_json(cls, text: str | bytes) -> Certificate:
        """Parse JSON text. Duplicate keys and NaN or Infinity are refused."""
        try:
            data = json.loads(text, object_pairs_hook=_no_duplicate_keys,
                              parse_constant=_reject_constant)
        except ValueError as exc:  # JSONDecodeError and the refusals above
            raise CertificateFormatError(f"not a readable certificate: {exc}") from exc
        except RecursionError:
            raise CertificateFormatError("not a readable certificate: nested too deeply") from None
        return cls(data)

    def save(self, path: str | Path) -> Path:
        """Write ``to_json()`` as UTF-8 with LF line endings and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((self.to_json() + "\n").encode("utf-8"))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Certificate:
        """Read a certificate file. It is not verified; call :func:`verify_certificate`."""
        try:
            text = Path(path).read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CertificateFormatError(f"{path} is not UTF-8 text: {exc}") from exc
        return cls.from_json(text)


# ------------------------------------------------------------------- creation


def _created_string(created: str | _dt.datetime | None) -> str:
    """The ``created_utc`` text. The clock is read here, and only when ``created`` is not given."""
    if created is None:
        created = _dt.datetime.now(_dt.timezone.utc)
    if isinstance(created, _dt.datetime):
        if created.tzinfo is None:
            raise ValueError("created is a naive datetime: pass an aware one or a string like "
                             "'2026-10-06T12:00:00Z'")
        return created.astimezone(_dt.timezone.utc).strftime(_CREATED_FORMAT)
    if isinstance(created, str):
        try:
            if not created.endswith("Z"):
                raise ValueError(created)
            _dt.datetime.strptime(created, "%Y-%m-%dT%H:%M:%S%z")  # %z reads the Z as UTC
        except ValueError:
            raise ValueError(f"created must look like '2026-10-06T12:00:00Z', got {created!r}"
                             ) from None
        return created
    raise TypeError(f"created must be a string, a datetime or None, got {type(created).__name__}")


def _identity_record(corpus: MultiVectorCorpus) -> dict[str, Any]:
    ident = corpus_identity(corpus)
    return {**ident.to_dict(), "digest": ident.digest()}


def _encoder_record(encoder: EncoderIdentity | Mapping[str, Any] | None,
                    corpus: MultiVectorCorpus) -> dict[str, Any] | None:
    if encoder is None:
        return None
    if isinstance(encoder, Mapping):
        encoder = EncoderIdentity.from_dict(encoder)
    if not isinstance(encoder, EncoderIdentity):
        raise TypeError("encoder must be an EncoderIdentity, a mapping or None, got "
                        f"{type(encoder).__name__}")
    if encoder.dimension != corpus.dimension:
        raise ValueError(f"the declared encoder has dimension {encoder.dimension}, the corpus "
                         f"has {corpus.dimension}")
    return encoder.to_dict()


def _check_types(decision: Any, evidence: Any, space: Any, corpus: Any, queries: Any,
                 reference: Any, pipeline: Any, compressed: Any) -> None:
    for name, value, kind in (
        ("decision", decision, Decision), ("evidence", evidence, Evidence),
        ("space", space, CandidateSpace), ("corpus", corpus, MultiVectorCorpus),
        ("queries", queries, MultiVectorCorpus), ("reference", reference, Reference),
    ):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__}, got {type(value).__name__}")
    if pipeline is not None and not isinstance(pipeline, Pipeline):
        raise TypeError(f"pipeline must be a Pipeline or None, got {type(pipeline).__name__}")
    if compressed is not None and not isinstance(compressed, CompressedCorpus):
        raise TypeError("compressed must be a CompressedCorpus or None, got "
                        f"{type(compressed).__name__}")


def _check_inputs(decision: Decision, evidence: Evidence, space: CandidateSpace,
                  corpus: MultiVectorCorpus, queries: MultiVectorCorpus,
                  reference: Reference) -> None:
    if decision.space_identity != space.identity:
        raise ValueError("the decision was made on a different candidate space: decision "
                         f"{decision.space_identity[:16]}..., space {space.identity[:16]}...")
    if evidence.space_identity != space.identity or evidence.n_candidates != len(space):
        raise ValueError("the evidence was not collected on the candidate space given")
    if decision.evidence_digest != evidence.digest():
        raise ValueError("the decision was computed from different evidence: decision "
                         f"{decision.evidence_digest[:16]}..., evidence {evidence.digest()[:16]}...")
    if decision.metric != evidence.metric or decision.construct != evidence.construct:
        raise ValueError("the decision's metric or construct differs from the evidence's")
    if reference.construct != decision.construct:
        raise ValueError(f"the reference measures {reference.construct!r}, the decision used "
                         f"{decision.construct!r}")
    if len(decision.p_values) != len(space):
        raise ValueError("the decision has a different number of p-values than the space has "
                         "candidates")
    validate_corpus(corpus, name="corpus")
    validate_corpus(queries, name="queries")
    if corpus.dimension != queries.dimension:
        raise ValueError(f"queries have dimension {queries.dimension}, the corpus has "
                         f"{corpus.dimension}")
    dropped = set(evidence.dropped_queries)
    if not dropped <= set(queries.ids) or (
            tuple(q for q in queries.ids if q not in dropped) != evidence.query_ids):
        raise ValueError("the queries given are not the queries the evidence was collected on "
                         "(its query ids must be the queries without the dropped ones, in order)")


def _check_candidate_objects(decision: Decision, evidence: Evidence, space: CandidateSpace,
                             corpus: MultiVectorCorpus, pipeline: Pipeline | None,
                             compressed: CompressedCorpus | None) -> None:
    outcome, selected = decision.outcome, decision.selected
    if outcome is Outcome.ABSTAIN:
        if pipeline is not None or compressed is not None:
            raise ValueError("an abstention deploys the uncompressed index: no pipeline or "
                             "compressed corpus can be bound to it")
        return
    if pipeline is None or compressed is None:
        raise ValueError(f"a {outcome.value} outcome needs the fitted pipeline and the "
                         "compressed corpus it deployed (see compress_selected)")
    position = selected.position
    if outcome is Outcome.SELECT and (not 0 <= position < len(space) or space[position] != selected):
        raise ValueError("the selected candidate is not the one at its position in the space")
    if _canonical_pipeline(pipeline.to_dict()) != _canonical_pipeline(thaw_json(selected.pipeline)):
        raise ValueError("the pipeline given is not the selected candidate's pipeline")
    if not pipeline.fitted:
        raise ValueError("the pipeline is not fitted: pass the pipeline that produced `compressed`")
    stats = compressed.stats
    if (len(compressed) != len(corpus) or list(compressed.ids) != list(corpus.ids)
            or compressed.num_vectors > corpus.num_vectors
            or stats.get("vectors_before") != corpus.num_vectors
            or stats.get("dim_before") != corpus.dimension):
        raise ValueError("the compressed corpus does not match the corpus: documents, ids, "
                         "vector count or dimension differ")
    if compressed.pipeline is not None and (
            _canonical_pipeline(compressed.pipeline) != _canonical_pipeline(pipeline.to_dict())):
        raise ValueError("the compressed corpus records a different pipeline than the one given")
    if outcome is not Outcome.SELECT:
        return
    if evidence.fitted and tuple(pipeline_fitted_identity(pipeline)) != evidence.fitted[position]:
        raise ValueError("the pipeline's fitted state is not the one that was evaluated for this "
                         "candidate (the evidence recorded different fitted-state digests); "
                         "use compress_selected(..., expected_fitted=evidence.fitted[position])")
    report = compressed.report()
    accounting = byte_accounting(compressed, pipeline)
    if (accounting.selection_bytes_per_doc != float(evidence.bytes_per_doc[position])
            or float(report["vectors_per_doc"]) != float(evidence.vectors_per_doc[position])):
        raise ValueError("the compressed corpus has a different size than the evidence recorded "
                         "for the selected candidate: it is not the index that was evaluated")


def _canonical_pipeline(spec: Mapping[str, Any]) -> bytes:
    return canonical_json(spec)


def _candidate_record(pipeline: Pipeline, compressed: CompressedCorpus) -> dict[str, Any]:
    return {
        "pipeline": pipeline_spec(pipeline),
        "pipeline_sha256": pipeline_identity(pipeline),
        "fitted_state": pipeline_fitted_identity(pipeline),
        "traits": json.loads(canonical_json(pipeline_traits(pipeline).to_dict())),
        "bytes": byte_accounting(compressed, pipeline).to_dict(),
        "codes_sha256": array_digest(compressed.codes),
        "offsets_sha256": array_digest(compressed.offsets),
    }


def create_certificate(
    decision: Decision,
    *,
    evidence: Evidence,
    space: CandidateSpace,
    corpus: MultiVectorCorpus,
    queries: MultiVectorCorpus,
    reference: Reference,
    pipeline: Pipeline | None = None,
    compressed: CompressedCorpus | None = None,
    encoder: EncoderIdentity | Mapping[str, Any] | None = None,
    created: str | _dt.datetime | None = None,
) -> Certificate:
    """The certificate of ``decision``, bound to the objects it was computed from.

    Every input is cross-checked before anything is written, and a mismatch raises
    ``ValueError``: the decision must come from this space and this evidence; the queries must
    be those the evidence was collected on; for ``select`` the ``pipeline`` must be the selected
    candidate's, fitted exactly as evaluated, and ``compressed`` must be its output on ``corpus``
    with the size the evidence recorded. ``select`` and ``fallback`` need both ``pipeline`` and
    ``compressed`` (use ``compress_selected``); an ``abstain`` outcome takes neither.

    ``certified`` is true only for ``select`` under one of the library's own certifying rules.
    For such a rule the statement and the assumptions are its own, verbatim; for any other rule
    the statement is a fixed text saying that the library has not examined it, and its
    assumptions are recorded as its author's. Sentences a reader must not miss (agreement is not
    retrieval quality; a fallback is not certified; the order, repeated queries, vectors that
    are not unit-norm, queries removed for having no relevant document) are recorded in
    ``caveats``. ``queries.exact_duplicates`` counts the tested queries that repeat an earlier one
    exactly, and ``norms`` records whether the document and query vectors have unit L2 norm.
    ``encoder`` is descriptive metadata and is never verified. ``created`` is an ISO string
    ending in ``Z`` or an aware ``datetime``; the clock is read only when it is not given, so
    with ``created`` the same inputs give a byte-identical certificate.

    The certificate is an integrity record, not a security token: see the module docstring.
    """
    _check_types(decision, evidence, space, corpus, queries, reference, pipeline, compressed)
    _check_inputs(decision, evidence, space, corpus, queries, reference)
    _check_candidate_objects(decision, evidence, space, corpus, pipeline, compressed)
    created_utc = _created_string(created)
    encoder_record = _encoder_record(encoder, corpus)

    outcome = decision.outcome
    selected = decision.selected
    sample = None
    candidate = None
    if outcome is not Outcome.ABSTAIN:
        candidate = _candidate_record(pipeline, compressed)
    if outcome is Outcome.SELECT:
        sample = sample_record(evidence, selected.position, decision.query_order)

    fields: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": created_utc,
        "package": package_info(),
        "code": code_identity(),
        "encoder": encoder_record,
        "corpus": _identity_record(corpus),
        "queries": {
            "n_pool": len(queries),
            "n_used": decision.n_queries,
            "identity": _identity_record(queries),
            "order": thaw_json(decision.query_order),
            "exact_duplicates": exact_duplicates(queries, evidence.dropped_queries,
                                                 decision.query_order),
        },
        "norms": norms_record(corpus, queries),
        "reference": {
            "construct": decision.construct,
            "metric": decision.metric,
            "identity": json.loads(canonical_json(reference.identity())),
        },
        "space": {
            "name": space.name,
            "identity": space.identity,
            "n_candidates": len(space),
            "n_families": space.n_families,
            "families": list(space.families),
            "registered_default": space.is_registered_default,
            "candidates": [spec.to_dict() for spec in space.candidates],
        },
        "sizes": {"bytes_per_doc": evidence.bytes_per_doc.tolist(),
                  "vectors_per_doc": evidence.vectors_per_doc.tolist()},
        "rule": thaw_json(decision.rule),
        "statement": statement_for(thaw_json(decision.rule)),
        "assumptions": list(decision.assumptions),
        "caveats": [],
        "target": decision.target,
        "error": decision.error,
        "level": decision.level,
        "evidence": {
            "digest": evidence.digest(),
            "n_queries": evidence.n_queries,
            "dropped_queries": list(evidence.dropped_queries),
            "baseline_mean": float(np.mean(evidence.baseline)),
        },
        "result": {
            "outcome": outcome.value,
            "certified": bool(decision.certified),
            "selected": None if selected is None else selected.position,
            "certified_positions": list(decision.certified_positions),
            "p_values": list(decision.p_values),
        },
        "sample": sample,
        "candidate": candidate,
    }
    fields["caveats"] = caveats_for(fields)
    fields["digest"] = content_digest(fields)
    fields["signature"] = None
    return Certificate(fields)


# -------------------------------------------------------------------- signing


def _check_key(key: Any) -> bytes:
    if isinstance(key, str) or not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes (encode a passphrase explicitly); a str is refused "
                        "because its encoding would be a guess")
    if len(key) == 0:
        raise ValueError("key must not be empty")
    return bytes(key)


def sign_certificate(cert: Certificate, key: bytes, key_id: str | None = None) -> Certificate:
    """A copy of ``cert`` carrying an HMAC-SHA256 of its digest under ``key``.

    HMAC is symmetric: whoever can verify the signature can also create it, so it protects
    against modification by parties without the key and gives no non-repudiation. Use a random
    key of at least 32 bytes. ``key_id`` is a free label (for example which key to look up); it
    is not authenticated by the signature. A certificate whose digest does not match its content
    is refused: signing it would vouch for a modified record.
    """
    if not isinstance(cert, Certificate):
        raise TypeError(f"cert must be a Certificate, got {type(cert).__name__}")
    secret = _check_key(key)
    if key_id is not None and not isinstance(key_id, str):
        raise TypeError("key_id must be a string or None")
    plain = cert.to_dict()
    if content_digest(plain) != plain.get("digest"):
        raise ValueError("the certificate's digest does not match its content; refusing to sign "
                         "a modified record")
    plain["signature"] = {"algorithm": SIGNATURE_ALGORITHM, "key_id": key_id,
                          "value": signature_value(secret, plain["digest"])}
    return Certificate(plain)


# --------------------------------------------------------------- verification


def verify_certificate(
    cert: Certificate,
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
    """Check a certificate, and report what was checked, what failed and what was not run.

    Level 1 (integrity) needs only the certificate: structure, content digest, HMAC (with
    ``key``), consistency of everything that can be recomputed from the record's own fields
    (candidate list identity, level, the decision step from the p-values, byte figures), the
    texts and structural traits the running library writes for the same rule and pipeline, and
    the code identity. Level 2 (replay) needs ``evidence``: its digest, and a re-run of
    ``select`` with the recorded target, error, rule, fallback, budget and order seed; p-values
    are compared at relative tolerance 1e-12 and everything else exactly. Level 3 (binding)
    compares the live ``corpus``, ``queries``, ``reference``, fitted ``pipeline`` and
    ``compressed`` corpus with the certified identities; each is optional.

    A differing code identity is reported as ``differs`` and does not fail the report unless
    ``strict_code=True``. A certificate that has a signature but is checked without ``key`` has
    its signature ``skipped``; with a ``key`` an unsigned certificate fails.

    This never raises on a bad certificate; it raises ``TypeError`` for wrong argument types.
    ``ok`` means that no check failed. It does not make the certificate true: it is as good as
    the queries, labels and corpus it was issued for, and the digest is not security (see the
    module docstring).
    """
    if not isinstance(cert, Certificate):
        raise TypeError(f"cert must be a Certificate, got {type(cert).__name__}")
    for name, value, kind in (
        ("evidence", evidence, Evidence), ("corpus", corpus, MultiVectorCorpus),
        ("queries", queries, MultiVectorCorpus), ("reference", reference, Reference),
        ("pipeline", pipeline, Pipeline), ("compressed", compressed, CompressedCorpus),
    ):
        if value is not None and not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__} or None, got "
                            f"{type(value).__name__}")
    if key is not None:
        key = _check_key(key)
    if not isinstance(strict_code, bool):
        raise TypeError("strict_code must be a bool")
    return run_checks(cert.to_dict(), evidence=evidence, corpus=corpus, queries=queries,
                      reference=reference, pipeline=pipeline, compressed=compressed, key=key,
                      strict_code=strict_code)
