"""``certify()``: choose a compression candidate under a stated error level, or abstain.

    from optivision import certify

    result = certify(doc_vectors, query_vectors, qrels, target=0.97, error=0.05)
    result.outcome                  # Outcome.SELECT, Outcome.ABSTAIN or Outcome.FALLBACK
    result.retriever().search(query_vectors, k=10)      # ranked ids from the chosen index
    result.save("artifact/")        # certificate, codes, fitted state, evidence

``certify`` is a second entry point beside :func:`optivision.optimize`, not a replacement for
it. The two do different things and say different things:

* ``optimize()`` / ``calibrate()`` choose on one half of the queries by a bootstrap lower bound,
  report the retention on the other half, and always return a compressed index when any
  candidate looked feasible. They state nothing about the chosen configuration beyond that
  held-out check. They are unchanged.
* ``certify()`` tests a candidate list that was fixed in advance, on all the queries, with the
  selection rule of :mod:`optivision.core.selection`; the rule selects one candidate or
  abstains, and a certificate records what was decided and under which assumptions. When
  nothing is certified the answer is the uncompressed index (``Outcome.ABSTAIN``): that is the
  normal, safe outcome, not an error.

What a certificate says is the rule's own statement (``core.selection.STATEMENT``), which is
conditional on the assumptions it lists. It is not a statement that the compressed index keeps
retrieval quality in general, and with the agreement reference it is a statement about
agreement with the uncompressed ranking, which is not retrieval quality. This module adds no
selection logic of its own: it validates, calls ``collect_evidence``, ``select``,
``compress_selected`` and ``create_certificate`` in that order, and hands the pieces back.

Model-blind by construction: nothing here branches on the encoder, the dimension (beyond the
library's own default candidate list), the modality or any name. ``encoder`` (or the identity
an adapter wrote into the vectors' ``attrs``) is recorded in the certificate and used for
nothing else.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from .compose import CompressedCorpus, Pipeline
from .core.artifact import (
    ArtifactError,
    Certificate,
    Check,
    VerificationReport,
    create_certificate,
    load_artifact,
    save_artifact,
    verify_certificate,
)
from .core.artifact.report import FAIL, INTEGRITY, PASS
from .core.evaluate import Evidence, LabelReference, RankingReference, Reference
from .core.evaluate.collect import collect_evidence, compress_selected
from .core.retrieve import CompressedRetriever, ExactRetriever
from .core.selection import (
    CandidateSpace,
    CandidateSpec,
    Decision,
    Outcome,
    QueryOrderError,
    SelectionRule,
    select,
)
from .core.selection.statistics import validate_evidence_arrays
from .core.similarity import DEFAULT_MAX_BLOCK_BYTES
from .core.vectors import (
    ContractError,
    EncoderIdentity,
    MultiVectorCorpus,
    as_corpus,
    validate_corpus,
)

__all__ = ["CertifiedResult", "certify", "verify"]

_ABSTAIN_NOTE = (
    "Nothing was certified at this target and error level on these queries, so the uncompressed "
    "index is the outcome. This is the safe outcome, not an error. It does not say that no "
    "candidate would pass: with more queries, or a lower target, a later run may certify one. "
    "Decide on that before looking again: every further look at the same queries, with another "
    "target, error level or list, is outside the statement of the rule."
)
_FALLBACK_NOTE = (
    "Nothing was certified. The compressed index is the fallback pipeline the caller supplied: "
    "the rule did not test it and its statement does not cover it."
)


# --------------------------------------------------------------------------- result


@dataclass(frozen=True, eq=False, repr=False)
class CertifiedResult:
    """What :func:`certify` returns: the decision, its record, and the index to use.

    ``decision`` is the rule's :class:`~optivision.core.selection.Decision` and ``certificate``
    the record of it (``certificate.digest`` identifies it); ``evidence`` holds the per-query
    arrays the decision was computed from and ``corpus`` is the corpus the call ran on.
    ``selected`` is the chosen candidate (``None`` on ABSTAIN), ``pipeline`` its fitted pipeline
    and ``compressed`` the corpus compressed with it; both are ``None`` on ABSTAIN, where nothing
    is compressed and no copy of the corpus is made.

    ``certified`` is True only for ``Outcome.SELECT`` under a certifying rule. ``Outcome.FALLBACK``
    (the caller supplied a fallback pipeline and nothing was certified) is not certified.
    No attribute is a retention figure. What a selection certifies is the rule's statement
    (``certificate.statement``), which is conditional on ``decision.assumptions``.
    """

    decision: Decision
    certificate: Certificate
    evidence: Evidence
    corpus: MultiVectorCorpus
    pipeline: Pipeline | None = None
    compressed: CompressedCorpus | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Decision):
            raise TypeError(f"decision must be a Decision, got {type(self.decision).__name__}")
        nothing = self.decision.outcome is Outcome.ABSTAIN
        if nothing and (self.pipeline is not None or self.compressed is not None):
            raise ValueError("an ABSTAIN result holds no pipeline and no compressed corpus")
        if not nothing and (self.pipeline is None or self.compressed is None):
            raise ValueError(f"a {self.decision.outcome.name} result needs the fitted pipeline "
                             "and the compressed corpus")

    # ------------------------------------------------------------------ decision

    @property
    def outcome(self) -> Outcome:
        return self.decision.outcome

    @property
    def certified(self) -> bool:
        """True only when the rule certified the selected candidate (``Outcome.SELECT``)."""
        return self.decision.certified

    @property
    def selected(self) -> CandidateSpec | None:
        """The chosen candidate: ``None`` on ABSTAIN, the caller's pipeline (position -1) on FALLBACK."""
        return self.decision.selected

    # ------------------------------------------------------------------ the index

    @property
    def store(self) -> MultiVectorCorpus | CompressedCorpus:
        """What to search: ``compressed`` on SELECT and FALLBACK, the original corpus on ABSTAIN."""
        return self.corpus if self.compressed is None else self.compressed

    @property
    def compression_ratio(self) -> float:
        """Float32 bytes of the input vectors over the stored bytes of ``compressed``.

        The same definition as ``OptimizationResult.compression_ratio`` (the library's
        ``compression_vs_float32``). It is 1.0 when nothing was compressed (ABSTAIN), whatever
        the dtype of the input. It is a size, not a quality figure.
        """
        if self.compressed is None:
            return 1.0
        return float(self.compressed.report()["compression_vs_float32"])

    def retriever(self, max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES) -> ExactRetriever:
        """A retriever over :attr:`store`: queries in, a ``Ranking`` of document ids out.

        A ``CompressedRetriever`` (queries are projected by the fitted pipeline, scoring decodes
        the codes block by block) on SELECT and FALLBACK; an ``ExactRetriever`` over the
        original corpus on ABSTAIN. ``max_block_bytes`` bounds the live similarity block.
        """
        if self.compressed is None:
            return ExactRetriever(self.corpus, max_block_bytes)
        return CompressedRetriever(self.compressed, self.pipeline, max_block_bytes)

    # ------------------------------------------------------------------ records

    def save(self, path: str | Path, *, include_evidence: bool = True,
             overwrite: bool = False) -> Path:
        """Write the artifact directory (certificate, codes, fitted state, evidence).

        ``include_evidence=False`` leaves out the per-query evidence, which a later replay check
        (``verify`` level 2) needs. An existing non-empty directory is refused unless
        ``overwrite=True``, which replaces the files of an earlier artifact and leaves any other
        file in the directory alone.
        """
        return save_artifact(
            path,
            certificate=self.certificate,
            compressed=self.compressed,
            pipeline=self.pipeline,
            evidence=self.evidence if include_evidence else None,
            overwrite=overwrite,
        )

    def report(self) -> dict[str, Any]:
        """A JSON-native summary of what was asked, decided and certified.

        Every number in it is a setting, a count, a size or the rule's own level; none is a
        measured retention. ``statement`` and ``assumptions`` are the rule's, verbatim, and the
        statement is conditional on the assumptions. ``caveats`` are the sentences the
        certificate records so that a reader cannot miss them (agreement with the uncompressed
        ranking is not retrieval quality; a fallback is not certified), and
        ``not_certified_note`` says in words what an ABSTAIN or a FALLBACK means.
        """
        d, cert = self.decision, self.certificate
        record = cert.to_dict()
        chosen = None
        if self.selected is not None:
            chosen = {
                "position": self.selected.position,
                "family": self.selected.family,
                "label": None if self.pipeline is None else self.pipeline.label(),
                "pipeline": self.selected.to_dict()["pipeline"],
            }
        traits = (record["candidate"] or {}).get("traits") or {}
        order = dict(d.query_order)
        note = {Outcome.ABSTAIN: _ABSTAIN_NOTE, Outcome.FALLBACK: _FALLBACK_NOTE}.get(d.outcome)
        return {
            "outcome": d.outcome.value,
            "certified": d.certified,
            "construct": d.construct,
            "metric": d.metric,
            "target": d.target,
            "error": d.error,
            "level": d.level,
            "rule": {"name": d.rule["name"], "certifying": d.rule["certifying"]},
            "n_queries": d.n_queries,
            "n_queries_given": self.evidence.n_queries + len(self.evidence.dropped_queries),
            "n_queries_without_relevant_document": len(self.evidence.dropped_queries),
            "query_order": {k: order.get(k) for k in ("kind", "seed", "budget")},
            "space": {key: record["space"][key] for key in (
                "name", "identity", "n_candidates", "n_families", "registered_default")},
            "selected": chosen,
            "compression_ratio": self.compression_ratio,
            "basis_sensitive": cert.basis_sensitive,
            "basis_invariance": traits.get("basis_invariance"),
            "statement": cert.statement,
            "assumptions": list(cert.assumptions),
            "caveats": list(cert.caveats),
            "not_certified_note": note,
            "certificate_digest": cert.digest,
        }

    def __repr__(self) -> str:
        d = self.decision
        if d.outcome is Outcome.ABSTAIN:
            what = "nothing certified: use the uncompressed index"
        else:
            what = f"{self.pipeline.label()!r}, {self.compression_ratio:.3g}x vs float32"
            if not d.certified:
                what += ", NOT certified"
        construct = d.construct
        if construct.startswith("agreement"):
            construct += " (agreement with the uncompressed ranking, not retrieval quality)"
        return (f"CertifiedResult(outcome={d.outcome.value!r}, certified={d.certified}, {what}; "
                f"reference={construct}, metric={d.metric}, target={d.target:g}, "
                f"error={d.error:g}, n_queries={d.n_queries})")


# --------------------------------------------------------------------------- inputs


def _as_corpus(value: Any, name: str) -> MultiVectorCorpus:
    if isinstance(value, (list, tuple)) and len(value) == 0:
        raise ContractError(f"{name}: nothing was given (an empty list)")
    corpus = as_corpus(value)
    validate_corpus(corpus, name=name)
    return corpus


def _check_target_and_error(target: Any, error: Any) -> None:
    """The check ``select`` makes of ``target`` and ``error``, made before the expensive step.

    The selection rule's own validator is called on a one-query dummy so that the exception
    type and message are the ones a late failure would give.
    """
    validate_evidence_arrays([[0.5]], [0.5], ["_"], [1.0], [1.0], target, error)


def _check_query_order(query_budget: Any, order_seed: Any, n_given: int) -> None:
    """The checks ``select`` makes of ``query_budget`` and ``order_seed``, made before the work.

    The budget can only be compared with the queries *given* here; ``select`` compares it again
    with the queries that have a relevant document.
    """
    for name, value, minimum in (("query_budget", query_budget, 1), ("order_seed", order_seed, 0)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise QueryOrderError(f"{name} must be an integer, got {type(value).__name__}")
        if value < minimum:
            raise QueryOrderError(f"{name} must be >= {minimum}, got {int(value)}")
    if query_budget is not None and query_budget > n_given:
        raise QueryOrderError(f"query_budget {query_budget} exceeds the {n_given} queries given; "
                              "a budget takes a prefix of the order and never samples with "
                              "replacement")


def _resolve_reference(reference: Any, qrels: Mapping[Any, Any] | None) -> Reference:
    if isinstance(reference, str):
        if reference not in ("auto", "labels", "agreement"):
            raise ValueError("reference must be 'auto', 'labels', 'agreement' or a reference "
                             f"object (LabelReference, RankingReference), got {reference!r}")
        kind = ("labels" if qrels is not None else "agreement") if reference == "auto" else reference
        if kind == "labels":
            if qrels is None:
                raise ValueError("reference='labels' needs qrels {query id: [relevant document "
                                 "ids]}; pass qrels=..., or reference='agreement' to test "
                                 "agreement with the uncompressed ranking (which is not "
                                 "retrieval quality)")
            return LabelReference(qrels)
        if qrels is not None:
            raise ValueError("qrels were given but reference='agreement' does not use them; use "
                             "reference='labels' (or 'auto') to test against the labels, or drop "
                             "qrels")
        return RankingReference()
    if isinstance(reference, Reference):
        if qrels is not None:
            raise ValueError("pass either qrels or a reference object, not both")
        return reference
    raise TypeError("reference must be 'auto', 'labels', 'agreement' or a reference object with "
                    f"construct, relevant() and identity(), got {type(reference).__name__}")


def _resolve_space(space: Any, dim: int) -> CandidateSpace:
    if space is None:
        try:
            return CandidateSpace.default(dim)
        except ImportError as exc:
            raise ImportError(
                "certify() tests the library's default candidate list, whose Ward-merge "
                "families need SciPy. Install it with pip install 'optivision-rag[merge]'. To "
                "run without SciPy, pass space=CandidateSpace.default(dim, allow_reduced=True) "
                "(a smaller list: a different space with a different identity and a different "
                "test level, both recorded in the certificate) or a candidate list of your own "
                "(a CandidateSpace, or a dict {family: [Pipeline, ...]} ordered from least to "
                "most aggressive). The list is never reduced silently."
            ) from exc
    if isinstance(space, CandidateSpace):
        return space
    if isinstance(space, Mapping):
        return CandidateSpace.from_families(space)
    raise TypeError("space must be None (the default list), a CandidateSpace or a dict "
                    f"{{family: [Pipeline, ...]}}, got {type(space).__name__}")


def _check_fallback(fallback: Any) -> None:
    if fallback is None or isinstance(fallback, Pipeline):
        return
    if isinstance(fallback, Mapping):
        try:
            Pipeline.from_dict(dict(fallback))  # refuses a bad description now, not after the run
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fallback is not a valid Pipeline.to_dict() description: {exc}") from exc
        return
    raise TypeError("fallback must be None, a Pipeline or a Pipeline.to_dict() mapping, "
                    f"got {type(fallback).__name__}")


def _declared_encoder(encoder: Any, corpus: MultiVectorCorpus) -> EncoderIdentity | None:
    """The identity to record in the certificate: ``encoder``, else what the vectors carry.

    The adapters that build a corpus write ``attrs["encoder_identity"]`` (a plain dict); an
    explicit ``encoder`` argument wins over it. This only chooses which descriptive metadata is
    written down: nothing is decided from it, and it is not checked against anything here.
    """
    if encoder is None:
        encoder, where = corpus.attrs.get("encoder_identity"), "vectors.attrs['encoder_identity']"
        if encoder is None:
            return None
    else:
        where = "encoder"
    if isinstance(encoder, EncoderIdentity):
        return encoder
    if not isinstance(encoder, Mapping):
        raise TypeError(f"{where} must be an EncoderIdentity or a dict, got {type(encoder).__name__}")
    try:
        return EncoderIdentity.from_dict(encoder)
    except ValueError as exc:  # ContractError is one
        raise ContractError(f"{where} is not a valid encoder identity: {exc}"
                            + ("" if where == "encoder" else "; fix it, or pass encoder=... instead")
                            ) from exc


# --------------------------------------------------------------------------- the entry point


def certify(
    vectors: Any,
    queries: Any,
    qrels: Mapping[Any, Any] | None = None,
    *,
    target: float = 0.97,
    error: float = 0.05,
    metric: str = "ndcg@5",
    reference: Any = "auto",
    space: CandidateSpace | Mapping[str, Sequence[Pipeline]] | None = None,
    rule: SelectionRule | None = None,
    query_budget: int | None = None,
    order_seed: int | None = None,
    fallback: Pipeline | Mapping[str, Any] | None = None,
    encoder: EncoderIdentity | Mapping[str, Any] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    created: Any = None,
) -> CertifiedResult:
    """Select the smallest candidate the rule certifies at ``(target, error)``, or abstain.

    Every candidate of the list is compressed and scored on every query, and the rule tests
    "retention is at most ``target``" for each, where retention is the mean metric of the
    candidate over the mean metric of the uncompressed index. If it rejects that for some
    candidates, the smallest of them is selected; if for none, the outcome is ``ABSTAIN`` and
    the uncompressed index is to be used. The rule's statement, which a certified selection
    carries and which is conditional on its assumptions (``result.decision.assumptions``), is:
    the probability that a compressed candidate is selected and its retention under the query
    distribution is below ``target`` is at most ``error``, over the draw of the queries. The
    verbatim text is ``optivision.core.selection.STATEMENT``; it makes no statement about power,
    and it also holds if the rule always abstains.

    Args:
        vectors: the documents, as ``optimize()`` takes them: a ``MultiVectorCorpus`` or a list
            of ``[n_i, d]`` float arrays. Not modified.
        queries: sample queries, same forms; at least one. The statement is about the
            distribution these are drawn from (assumption A1).
        qrels: ``{query id: [relevant document ids]}``. Query and document ids are the ids of
            the two corpora (``"0"``, ``"1"``, ... for a list of arrays).
        target: the retention to be exceeded, in (0, 1].
        error: the level of the statement, in (0, 1).
        metric: ``ndcg@k``, ``recall@k``, ``mrr@k`` or ``hit@k``. Only nDCG@k was evaluated by
            the registered experiments.
        reference: ``"auto"`` (labels when ``qrels`` is given, otherwise agreement),
            ``"labels"``, ``"agreement"``, or a ``LabelReference`` / ``RankingReference``.
            Agreement means agreement with the uncompressed ranking's own top document. It is
            not retrieval quality; the result and the certificate say which was tested.
        space: ``None`` for the library's default list (needs SciPy; raises ``ImportError``
            with advice otherwise), a ``CandidateSpace``, or a dict ``{family: [Pipeline,
            ...]}`` as ``calibrate(search_space=...)`` takes it. Each family is ordered from
            least to most aggressive.
        rule: a selection rule; ``FixedSequenceBetting()`` by default.
        query_budget: test only a prefix of ``query_budget`` queries of the order used.
        order_seed: if given, test the queries in a seeded permutation (independent of the
            data) instead of the order given; the seed is recorded.
        fallback: a ``Pipeline`` (or its ``to_dict()``) to return when nothing is certified.
            It is recorded as NOT certified (``Outcome.FALLBACK``). There is no default: with
            none, nothing certified means the uncompressed index.
        encoder: an ``EncoderIdentity`` (or its dict) recorded in the certificate. Descriptive:
            it is neither verified nor used by any decision. If omitted, the identity the
            adapters write into ``vectors.attrs["encoder_identity"]`` is recorded instead, when
            there is one; an explicit ``encoder`` wins.
        progress: optional ``progress(position, n_candidates, family)``, called after each
            candidate is evaluated.
        created: the creation time to record (a string like ``"2026-10-06T12:00:00Z"`` or an
            aware ``datetime``; checked when the certificate is made, after the evidence); the
            clock when omitted. With the same inputs and ``created``, the certificate is
            identical.

    Obligations that no code can check, and which the statement assumes: the candidate list,
    its families and its order were fixed before these queries were looked at (this function
    never lets the queries influence the list); the queries are independent draws from the
    deployment distribution; the order they are tested in is fixed in advance. Call this once
    per ``(target, error)``: trying several targets on the same queries and reporting one is
    outside the statement. If you want another look, collect new queries.

    Nothing certified is the normal, safe outcome and is returned like any other: no exception,
    no warning. ``result.outcome is Outcome.ABSTAIN`` and ``result.store`` is the original
    corpus.

    Raises ``ValueError`` (``ContractError`` for invalid vectors) on invalid input,
    ``ImportError`` if the default list needs SciPy, and ``CandidateEvaluationError`` if a
    candidate cannot be evaluated (a candidate is never skipped or replaced).
    """
    if queries is None:
        raise ValueError(
            "certify() needs sample queries to test the candidates against; pass queries=[...] "
            "(real queries from the distribution the index will serve; labels are optional)"
        )
    _check_target_and_error(target, error)
    corpus = _as_corpus(vectors, "vectors")
    query_corpus = _as_corpus(queries, "queries")
    if query_corpus.dimension != corpus.dimension:
        raise ContractError(f"queries have dimension {query_corpus.dimension} but the vectors "
                            f"have dimension {corpus.dimension}")
    _check_query_order(query_budget, order_seed, len(query_corpus))
    if rule is not None and not isinstance(rule, SelectionRule):
        raise TypeError("rule must provide name, certifying, config(), assumptions() and decide()")
    ref = _resolve_reference(reference, qrels)
    candidate_space = _resolve_space(space, corpus.dimension)
    _check_fallback(fallback)
    identity = _declared_encoder(encoder, corpus)

    evidence = collect_evidence(corpus, query_corpus, candidate_space, ref, metric=metric,
                                progress=progress)
    decision = select(candidate_space, evidence, target=target, error=error, rule=rule,
                      fallback=fallback, query_budget=query_budget, order_seed=order_seed)

    pipeline = compressed = None
    if decision.outcome is not Outcome.ABSTAIN:
        expected = None
        if decision.outcome is Outcome.SELECT and evidence.fitted:
            expected = evidence.fitted[decision.selected.position]
        pipeline, compressed = compress_selected(corpus, decision.selected,
                                                 expected_fitted=expected)
    certificate = create_certificate(
        decision, evidence=evidence, space=candidate_space, corpus=corpus, queries=query_corpus,
        reference=ref, pipeline=pipeline, compressed=compressed, encoder=identity,
        created=created)
    return CertifiedResult(decision=decision, certificate=certificate, evidence=evidence,
                           corpus=corpus, pipeline=pipeline, compressed=compressed)


# --------------------------------------------------------------------------- verification


def verify(
    source: Certificate | str | Path,
    *,
    evidence: Evidence | None = None,
    corpus: Any = None,
    queries: Any = None,
    reference: Reference | None = None,
    pipeline: Pipeline | None = None,
    compressed: CompressedCorpus | None = None,
    key: bytes | None = None,
    strict_code: bool = False,
) -> VerificationReport:
    """Check a certificate, a ``certificate.json`` file or an artifact directory.

    ``source`` is a ``Certificate``, the path of a certificate file, or the path of a directory
    written by ``CertifiedResult.save``. For a directory the report starts with one check,
    ``artifact files`` (the manifest matches every file, the stored pipeline is the certified
    one and its fitted state restores to the certified digests), and the certificate is then
    checked against the stored evidence (replay) and the stored pipeline and codes (binding).
    Objects passed explicitly take the place of the stored ones. Pass the live ``corpus``,
    ``queries`` and ``reference`` you actually serve to check that the certificate was issued
    for *them*: a stored artifact cannot show that by itself.

    A bad certificate is reported, not raised; ``report.ok`` means that no check failed, and it
    does not make the certificate true. An unreadable directory raises ``ArtifactError`` or
    ``FileNotFoundError``. See ``verify_certificate`` for the levels and ``key`` / ``strict_code``.
    """
    objects: dict[str, Any] = {"evidence": evidence, "corpus": corpus, "queries": queries,
                               "reference": reference, "pipeline": pipeline,
                               "compressed": compressed}
    for name in ("corpus", "queries"):
        if objects[name] is not None:
            objects[name] = as_corpus(objects[name])
    leading: tuple[Check, ...] = ()
    if isinstance(source, Certificate):
        certificate = source
    elif Path(source).is_file():
        certificate = Certificate.load(source)
    else:
        try:
            stored = load_artifact(source, verify=True, key=key, strict_code=strict_code)
            leading = (Check("artifact files", PASS, "manifest, stored pipeline and fitted state "
                             "match the certificate", INTEGRITY),)
        except ArtifactError as exc:  # report it, then read what can be read to check the rest
            stored = load_artifact(source, verify=False)
            leading = (Check("artifact files", FAIL, str(exc).splitlines()[0], INTEGRITY),)
        certificate = stored.certificate
        for name in ("evidence", "pipeline", "compressed"):
            if objects[name] is None:
                objects[name] = getattr(stored, name)
    report = verify_certificate(certificate, key=key, strict_code=strict_code, **objects)
    return VerificationReport((*leading, *report.checks)) if leading else report
