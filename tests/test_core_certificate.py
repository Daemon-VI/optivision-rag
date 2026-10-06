"""Certificates: creation, verification at every level, signing, serialisation, determinism.

A certificate is issued for every outcome (select, abstain, fallback) from a decision and the
objects it was computed from. These tests issue certificates on small seeded synthetic data and
check what is recorded, what is refused at creation, what verification reports at each level,
and that nothing in the record is wider than the rule's own statement. They assert structure,
never a retention number: the data are synthetic.

``issue`` and ``Issued`` are shared with the tamper and bundle tests.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.core import artifact
from optivision.core.artifact import (
    Certificate,
    CertificateFormatError,
    Check,
    VerificationReport,
    create_certificate,
    sign_certificate,
    verify_certificate,
)
from optivision.core.artifact import verification as verification_module
from optivision.core.artifact.record import (
    CAVEAT_DROPPED,
    CAVEAT_ORDER_GIVEN,
    CAVEAT_ORDER_SEED,
    FIELD_ORDER,
    NORM_TOLERANCE,
    STATEMENT_UNEXAMINED,
    exact_duplicates,
    norm_record,
)
from optivision.core.evaluate import (
    LabelReference,
    RankingReference,
    collect_evidence,
    compress_selected,
)
from optivision.core.identity import array_digest, canonical_json, digest
from optivision.core.selection import (
    STATEMENT,
    BonferroniBetting,
    Outcome,
    RuleResult,
    select,
)
from optivision.core.stages import (
    BinaryQuantizer,
    Float16Quantizer,
    Int8Quantizer,
    byte_accounting,
    pipeline_fitted_identity,
    pipeline_traits,
)
from optivision.core.vectors import EncoderIdentity, corpus_identity
from optivision.representation import MultiVectorCorpus

from .test_core_collect_evidence import make_problem, small_space

CREATED = "2026-10-06T12:00:00Z"
KEY = b"k" * 32
HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: The fields of schema version 1, in file order.
SCHEMA_V1_FIELDS = (
    "schema", "schema_version", "created_utc", "package", "code", "encoder", "corpus", "queries",
    "norms", "reference", "space", "sizes", "rule", "statement", "assumptions", "caveats",
    "target", "error", "level", "evidence", "result", "sample", "candidate", "digest",
    "signature",
)


# ---------------------------------------------------------------- shared helpers


@dataclass(frozen=True)
class Issued:
    """A certificate with every object it was issued from."""

    decision: Any
    evidence: Any
    space: Any
    corpus: Any
    queries: Any
    reference: Any
    pipeline: Any
    compressed: Any
    certificate: Certificate

    def objects(self) -> dict[str, Any]:
        """Every object that ``verify_certificate`` can bind to (``None`` where there is none)."""
        return {"evidence": self.evidence, "corpus": self.corpus, "queries": self.queries,
                "reference": self.reference, "pipeline": self.pipeline,
                "compressed": self.compressed}

    def verify(self, certificate=None, **override) -> VerificationReport:
        kwargs = {**self.objects(), **override}
        return verify_certificate(self.certificate if certificate is None else certificate,
                                  **kwargs)


def issue(problem, space, *, kind="labels", target=0.8, error=0.2, rule=None, fallback=None,
          query_budget=None, order_seed=None, created=CREATED, encoder=None) -> Issued:
    """Collect evidence, select, deploy the choice and issue its certificate."""
    corpus, queries, qrels = problem
    reference = LabelReference(qrels) if kind == "labels" else RankingReference()
    evidence = collect_evidence(corpus, queries, space, reference)
    decision = select(space, evidence, target=target, error=error, rule=rule, fallback=fallback,
                      query_budget=query_budget, order_seed=order_seed)
    pipeline = compressed = None
    if decision.outcome is not Outcome.ABSTAIN:
        expected = evidence.fitted[decision.selected.position] \
            if decision.outcome is Outcome.SELECT else None
        pipeline, compressed = compress_selected(corpus, decision.selected,
                                                 expected_fitted=expected)
    certificate = create_certificate(
        decision, evidence=evidence, space=space, corpus=corpus, queries=queries,
        reference=reference, pipeline=pipeline, compressed=compressed, encoder=encoder,
        created=created)
    return Issued(decision, evidence, space, corpus, queries, reference, pipeline, compressed,
                  certificate)


class DeclaresCertifying:
    """A rule of someone else's that declares itself certifying and tests nothing: every
    p-value is 1 and the smallest candidate is selected."""

    name = "declares_certifying"
    certifying = True

    def config(self):
        return {"name": self.name}

    def assumptions(self):
        return ("none: trust me.",)

    def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc, target,
               error):
        k = int(np.argmin(bytes_per_doc))
        return RuleResult(k, (k,), tuple(1.0 for _ in bytes_per_doc), error)


def statuses(report, status):
    return sorted(c.name for c in report.checks if c.status == status)


@pytest.fixture(scope="module")
def problem():
    return make_problem()


@pytest.fixture(scope="module")
def space():
    return small_space()


@pytest.fixture(scope="module")
def selected(problem, space):
    """Labels, target 0.8: several candidates certified, a stateful one selected."""
    out = issue(problem, space)
    assert out.decision.outcome is Outcome.SELECT and len(out.decision.certified_positions) >= 2
    assert len(out.decision.certified_positions) < len(space)
    return out


@pytest.fixture(scope="module")
def agreement(problem, space):
    out = issue(problem, space, kind="agreement", target=0.9)
    assert out.decision.outcome is Outcome.SELECT and out.decision.construct == "agreement@1"
    return out


@pytest.fixture(scope="module")
def abstained(problem, space):
    out = issue(problem, space, target=0.95, error=0.3)
    assert out.decision.outcome is Outcome.ABSTAIN
    return out


@pytest.fixture(scope="module")
def fell_back(problem, space):
    fallback = Pipeline([Int8Quantizer("per_dimension", center="mean")])
    out = issue(problem, space, target=1.0, fallback=fallback)
    assert out.decision.outcome is Outcome.FALLBACK
    return out


# ------------------------------------------------------------------ what is recorded


class TestFieldsOfTheRecord:
    def test_the_field_list_is_schema_version_1(self, selected):
        assert set(selected.certificate.to_dict()) == set(SCHEMA_V1_FIELDS)
        assert FIELD_ORDER == SCHEMA_V1_FIELDS
        assert list(json.loads(selected.certificate.to_json())) == list(SCHEMA_V1_FIELDS)

    def test_schema_package_and_time(self, selected):
        c = selected.certificate
        assert c.data["schema"] == "optivision-certificate" and c.schema_version == 1
        assert c.created_utc == CREATED
        assert c.data["package"] == {"name": "optivision", "version": "0.3.0"}

    def test_the_decision_is_recorded_as_it_was_made(self, selected):
        c, d = selected.certificate, selected.decision
        assert c.outcome is Outcome.SELECT and c.certified is True
        assert c.target == d.target == 0.8 and c.error == d.error == 0.2
        assert c.level == d.level
        assert c.p_values == d.p_values
        assert c.certified_positions == d.certified_positions
        assert c.selected_position == d.selected.position
        assert c.selected == d.selected
        assert c.metric == "ndcg@5" and c.construct == "labels"
        assert c.rule_name == "fixed_sequence_betting"
        assert c.space_identity == selected.space.identity
        assert c.n_queries == d.n_queries == selected.evidence.n_queries

    def test_the_statement_and_assumptions_are_the_rules_own_verbatim(self, selected):
        c = selected.certificate
        assert c.statement == STATEMENT
        assert c.assumptions == selected.decision.assumptions
        assert len(c.assumptions) == 10
        assert [a[:3] for a in c.assumptions] == [f"{k}{i}:" for k, n in (("A", 6), ("C", 4))
                                                  for i in range(1, n + 1)]
        assert c.data["rule"]["certifying"] is True
        assert c.data["rule"]["config"]["bet_cap"] == 0.9

    def test_the_rule_is_recorded_with_its_class_and_whether_it_is_the_librarys_own(self,
                                                                                   selected):
        rule = selected.certificate.to_dict()["rule"]
        assert set(rule) == {"name", "class", "builtin", "certifying", "declared_certifying",
                             "config"}
        assert rule["class"] == "optivision.core.selection.rules.FixedSequenceBetting"
        assert rule["builtin"] is True and rule["declared_certifying"] is True
        assert rule == selected.decision.to_dict()["rule"]

    def test_the_space_is_recorded_whole(self, selected):
        s = selected.certificate.to_dict()["space"]
        assert s["identity"] == selected.space.identity and s["name"] == "small"
        assert s["n_candidates"] == 7 and s["n_families"] == 4
        assert s["families"] == ["float16", "merge+int8", "pca+int4", "binary"]
        assert s["registered_default"] is False
        assert s["candidates"] == [spec.to_dict() for spec in selected.space.candidates]

    def test_sizes_are_the_values_the_choice_used(self, selected):
        s = selected.certificate.to_dict()["sizes"]
        assert s["bytes_per_doc"] == selected.evidence.bytes_per_doc.tolist()
        assert s["vectors_per_doc"] == selected.evidence.vectors_per_doc.tolist()

    def test_evidence_summary(self, selected):
        e = selected.certificate.to_dict()["evidence"]
        ev = selected.evidence
        assert e["digest"] == ev.digest() == selected.decision.evidence_digest
        assert e["n_queries"] == ev.n_queries and e["dropped_queries"] == ["q0", "q1", "q2"]
        assert e["baseline_mean"] == float(np.mean(ev.baseline))

    def test_identities_of_corpus_and_queries(self, selected):
        d = selected.certificate.to_dict()
        for record, obj in ((d["corpus"], selected.corpus),
                            (d["queries"]["identity"], selected.queries)):
            ident = corpus_identity(obj)
            assert {k: v for k, v in record.items() if k != "digest"} == ident.to_dict()
            assert record["digest"] == ident.digest()
        assert d["queries"]["n_pool"] == len(selected.queries) == 80
        assert d["queries"]["n_used"] == 77
        assert d["queries"]["order"]["kind"] == "given"
        assert d["queries"]["exact_duplicates"] == 0

    def test_reference_identity(self, selected, agreement):
        r = selected.certificate.to_dict()["reference"]
        assert r == {"construct": "labels", "metric": "ndcg@5",
                     "identity": selected.reference.identity()}
        assert agreement.certificate.to_dict()["reference"]["identity"] == {
            "construct": "agreement@1", "depth": 1}

    def test_the_candidate_record(self, selected):
        c = selected.certificate.to_dict()["candidate"]
        pipe, comp = selected.pipeline, selected.compressed
        assert c["pipeline"] == json.loads(canonical_json(pipe.to_dict()))
        assert c["pipeline_sha256"] == digest(pipe.to_dict())
        assert c["fitted_state"] == pipeline_fitted_identity(pipe)
        assert c["traits"] == json.loads(canonical_json(pipeline_traits(pipe).to_dict()))
        assert c["bytes"] == byte_accounting(comp, pipe).to_dict()
        assert c["codes_sha256"] == array_digest(comp.codes)
        assert c["offsets_sha256"] == array_digest(comp.offsets)
        assert HEX64.match(c["codes_sha256"]) and HEX64.match(c["pipeline_sha256"])
        assert c["pipeline"] == selected.certificate.to_dict()["space"]["candidates"][
            selected.decision.selected.position]["pipeline"]

    def test_traits_carry_the_basis_properties(self, selected):
        traits = selected.certificate.to_dict()["candidate"]["traits"]
        assert {"basis_invariance", "conditionally_invariant", "basis_sensitive"} <= set(traits)
        assert selected.certificate.basis_sensitive is traits["basis_sensitive"]
        assert selected.certificate.basis_sensitive is True  # a coordinate codec after a merge

    def test_sample_statistics_are_labelled_as_biased_and_uncertified(self, selected):
        s = selected.certificate.to_dict()["sample"]
        pos = selected.decision.selected.position
        ev = selected.evidence
        assert s["mean"] == float(np.mean(ev.candidates[pos]))
        assert s["baseline_mean"] == float(np.mean(ev.baseline))
        assert s["retention_point"] == pytest.approx(s["mean"] / s["baseline_mean"], rel=1e-12)
        assert s["n"] == ev.n_queries
        assert "optimistically biased" in s["note"] and "not certified" in s["note"]

    def test_signature_is_empty_when_unsigned(self, selected):
        assert selected.certificate.signed is False
        assert selected.certificate.signature is None
        assert selected.certificate.to_dict()["signature"] is None

    def test_the_digest_covers_every_field_but_digest_and_signature(self, selected):
        d = selected.certificate.to_dict()
        body = {k: v for k, v in d.items() if k not in ("digest", "signature")}
        assert d["digest"] == hashlib.sha256(canonical_json(body)).hexdigest()
        assert selected.certificate.digest == d["digest"]

    def test_the_encoder_is_descriptive_metadata_only(self, problem, space):
        enc = EncoderIdentity(model_id="my/model", revision="r1", dimension=16, modality="text",
                              normalized=True, extra={"pooling": "none"})
        out = issue(problem, space, encoder=enc)
        assert out.certificate.to_dict()["encoder"] == enc.to_dict()
        assert out.verify().ok
        again = issue(problem, space, encoder=enc.to_dict())  # a mapping is accepted too
        assert again.certificate == out.certificate
        assert issue(problem, space).certificate.to_dict()["encoder"] is None
        with pytest.raises(ValueError, match="dimension 20"):
            issue(problem, space, encoder=EncoderIdentity(model_id="m", dimension=20,
                                                          modality="text"))
        with pytest.raises(TypeError, match="encoder"):
            issue(problem, space, encoder="my/model")


class TestOutcomes:
    def test_abstain_is_certified_for_nothing_and_has_no_candidate(self, abstained):
        c = abstained.certificate
        assert c.outcome is Outcome.ABSTAIN and c.certified is False
        assert c.selected is None and c.selected_position is None
        assert c.certified_positions == () and c.basis_sensitive is None
        d = c.to_dict()
        assert d["candidate"] is None and d["sample"] is None
        assert d["result"]["certified"] is False
        assert c.statement == STATEMENT and len(c.assumptions) == 10  # an abstention is checkable
        assert c.caveats == (CAVEAT_ORDER_GIVEN, CAVEAT_DROPPED.format(k=3, n=80))

    def test_fallback_is_recorded_and_never_certified(self, fell_back):
        c = fell_back.certificate
        assert c.outcome is Outcome.FALLBACK and c.certified is False
        assert c.selected_position == -1 and c.selected.family == "fallback"
        assert c.selected.pipeline["stages"][0]["stage"] == "int8"
        assert c.certified_positions == ()
        d = c.to_dict()
        assert d["sample"] is None and d["candidate"] is not None
        assert d["result"]["certified"] is False and d["result"]["selected"] == -1
        assert any("fallback" in s and "not certified" in s for s in c.caveats)
        assert d["candidate"]["pipeline"] == c.selected.to_dict()["pipeline"]

    def test_fallback_state_is_in_the_record(self, fell_back):
        # a per-dimension int8 codec keeps ranges and a mean: both are fitted state
        assert fell_back.certificate.to_dict()["candidate"]["fitted_state"][0] is not None

    @pytest.mark.parametrize("which", ["selected", "agreement", "abstained", "fell_back"])
    def test_certified_is_true_only_for_a_select_outcome(self, request, which):
        out = request.getfixturevalue(which)
        c = out.certificate
        assert c.certified == (c.outcome is Outcome.SELECT) == out.decision.certified
        assert c.to_dict()["result"]["certified"] is c.certified

    @pytest.mark.parametrize("which", ["selected", "agreement", "abstained", "fell_back"])
    def test_each_outcome_verifies_at_every_level(self, request, which):
        out = request.getfixturevalue(which)
        report = out.verify()
        assert report.ok, report.summary()
        assert report.level == "binding" and report.levels == ("integrity", "replay", "binding")
        assert statuses(report, "fail") == []
        assert statuses(report, "differs") == []
        expected_skipped = ["signature"]
        if which == "abstained":
            expected_skipped += ["binding.bytes", "binding.codes", "binding.pipeline",
                                 "candidate.traits", "consistency.candidate"]
        if which in ("abstained", "fell_back"):
            expected_skipped += ["consistency.sample", "evidence.sample"]
        assert statuses(report, "skipped") == sorted(expected_skipped)


class TestTheAgreementConstruct:
    def test_the_caveat_says_agreement_is_not_retrieval_quality(self, agreement):
        c = agreement.certificate
        assert c.construct == "agreement@1"
        assert any("agreement with the uncompressed ranking" in s
                   and "not retrieval quality" in s for s in c.caveats)
        assert c.statement == STATEMENT  # the rule's statement is unchanged; the caveat is beside it

    def test_labels_runs_carry_no_such_caveat(self, selected):
        assert not any("agreement" in c for c in selected.certificate.caveats)
        assert selected.certificate.caveats == (CAVEAT_ORDER_GIVEN, CAVEAT_DROPPED.format(k=3, n=80))

    def test_the_baseline_of_the_agreement_construct_is_one(self, agreement):
        assert agreement.certificate.to_dict()["evidence"]["baseline_mean"] == 1.0

    def test_a_fallback_under_agreement_carries_both_caveats(self, problem, space):
        out = issue(problem, space, kind="agreement", target=1.0,
                    fallback=Pipeline([Float16Quantizer()]))
        assert out.certificate.outcome is Outcome.FALLBACK
        text = " ".join(out.certificate.caveats)
        assert "not retrieval quality" in text and "fallback" in text
        assert out.verify().ok


class TestOtherRulesAndOrders:
    def test_bonferroni_rule_round_trips(self, problem, space):
        out = issue(problem, space, rule=BonferroniBetting())
        c = out.certificate
        assert c.rule_name == "bonferroni_betting" and c.level == pytest.approx(0.2 / 7)
        assert c.certified and out.verify().ok
        assert c.assumptions == BonferroniBetting().assumptions()

    def test_a_seeded_order_and_a_budget_are_recorded_and_replayed(self, problem, space):
        out = issue(problem, space, order_seed=7, query_budget=50)
        order = out.certificate.to_dict()["queries"]["order"]
        assert order["kind"] == "seeded_permutation" and order["seed"] == 7
        assert order["budget"] == 50 and order["n_used"] == 50
        assert order["n_available"] == out.evidence.n_queries
        assert out.certificate.n_queries == 50
        report = out.verify()
        assert report.ok, report.summary()
        assert report.status("replay.decision") == "pass"
        if out.certificate.certified:  # sample statistics are on the tested queries
            pos = out.certificate.selected_position
            idx = np.random.default_rng(7).permutation(out.evidence.n_queries)[:50]
            assert out.certificate.to_dict()["sample"]["mean"] == float(
                np.mean(out.evidence.candidates[pos][idx]))
            assert out.certificate.to_dict()["sample"]["n"] == 50

    @pytest.mark.parametrize("declared", [False, True], ids=["declares-nothing", "declares-certifying"])
    def test_a_rule_of_someone_elses_is_never_certified_whatever_it_declares(self, problem, space,
                                                                            declared):
        class PointRetention:
            name = "test_point_retention"
            certifying = declared

            def config(self):
                return {"name": self.name}

            def assumptions(self):
                return ("This rule makes no error statement.",)

            def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc,
                       target, error):
                retention = candidates.mean(axis=1) / baseline.mean()
                ok = [k for k in range(len(retention)) if retention[k] >= target]
                best = min(ok, key=lambda k: (bytes_per_doc[k], vectors_per_doc[k], k)) \
                    if ok else None
                return RuleResult(best, tuple(ok), tuple(1.0 for _ in retention), error)

        out = issue(problem, space, rule=PointRetention(), target=0.8)
        c = out.certificate
        assert c.outcome is Outcome.SELECT and out.decision.outcome is Outcome.SELECT
        assert c.certified is False and c.to_dict()["result"]["certified"] is False
        assert out.decision.certified is False
        rule = c.to_dict()["rule"]
        assert rule["builtin"] is False and rule["certifying"] is False
        assert rule["declared_certifying"] is declared  # kept: it is what the author declared
        assert rule["class"].endswith("PointRetention")
        # the library's statement is never recorded for it, and the text says why
        assert c.statement == STATEMENT_UNEXAMINED != STATEMENT
        assert "not part of this library" in c.statement and "makes no statement" in c.statement
        assert any("not part of this library" in s and "not certified" in s for s in c.caveats)
        assert c.assumptions == (
            ("Declared by the rule's author, not examined by this library: "
             "This rule makes no error statement."),)
        report = out.verify()
        assert report.ok, report.summary()
        # what the library cannot examine is skipped and said so; what it can is checked
        for name in ("replay.decision", "consistency.decision", "consistency.level"):
            assert report.status(name) == "skipped", name
        assert report.status("rule.text") == "pass"
        assert "not examined" in report.get("rule.text").detail
        assert "nothing in this record is certified" in report.get("replay.decision").detail
        assert any("not one of this library's own" in n and "nothing in this record is certified"
                   in n for n in report.notes)
        assert "note: this record names a rule" in report.summary()


# ------------------------------------------- what is recorded about the order and the inputs


def with_copies(problem):
    """The queries followed by an exact copy of each under a new id (the labels are copied too)."""
    corpus, queries, qrels = problem
    arrays = [queries.vectors[queries.offsets[i]:queries.offsets[i + 1]]
              for i in range(len(queries))]
    ids = [*queries.ids, *[f"{q}_copy" for q in queries.ids]]
    labels = {**qrels, **{f"{q}_copy": docs for q, docs in qrels.items()}}
    return corpus, MultiVectorCorpus.from_arrays(arrays + arrays, ids=ids), labels


class TestTheOrderIsRecordedWithWhatItRequires:
    def test_a_given_order_carries_the_sentence_that_it_must_be_unrelated_to_the_outcomes(
            self, selected):
        assert selected.certificate.to_dict()["queries"]["order"]["kind"] == "given"
        assert CAVEAT_ORDER_GIVEN in selected.certificate.caveats
        assert "unrelated to the outcomes" in CAVEAT_ORDER_GIVEN
        assert "sorted by difficulty, date, source" in CAVEAT_ORDER_GIVEN

    def test_the_default_seed_carries_no_order_sentence(self, problem, space):
        out = issue(problem, space, order_seed=0)
        order = out.certificate.to_dict()["queries"]["order"]
        assert order["kind"] == "seeded_permutation" and order["seed"] == 0
        assert not any("order" in c and "seed" in c for c in out.certificate.caveats)
        assert CAVEAT_ORDER_GIVEN not in out.certificate.caveats
        assert out.verify().ok

    @pytest.mark.parametrize("seed", [1, 7, 12345])
    def test_another_seed_says_that_the_caller_chose_it_and_that_looking_first_is_outside(
            self, problem, space, seed):
        out = issue(problem, space, order_seed=seed)
        assert CAVEAT_ORDER_SEED.format(seed=seed) in out.certificate.caveats
        text = CAVEAT_ORDER_SEED.format(seed=seed)
        assert "chosen by the caller" in text and "trying several seeds" in text
        assert "outside the statement" in text
        assert CAVEAT_ORDER_GIVEN not in out.certificate.caveats
        assert out.verify().ok

    def test_the_sentences_are_recomputed_from_the_record_so_an_edit_of_the_order_fails(
            self, problem, space):
        out = issue(problem, space, order_seed=7)
        d = out.certificate.to_dict()
        d["caveats"] = [c for c in d["caveats"] if "order seed" not in c]
        d["digest"] = digest({k: v for k, v in d.items() if k not in ("digest", "signature")})
        report = out.verify(Certificate.from_dict(d))
        assert report.status("rule.text") == "fail" and "caveats" in report.get("rule.text").detail


class TestExactDuplicateQueriesAreCounted:
    def test_distinct_queries_give_zero_and_no_sentence(self, problem, space):
        corpus, queries, qrels = make_problem(n_unlabelled=0)
        out = issue((corpus, queries, qrels), space, order_seed=0)
        assert out.certificate.to_dict()["queries"]["exact_duplicates"] == 0
        assert not any("repeat an earlier query" in c for c in out.certificate.caveats)
        assert out.verify().ok

    def test_every_query_twice_gives_the_number_of_copies_and_the_sentence(self, space):
        corpus, doubled, labels = with_copies(make_problem(n_unlabelled=0))
        assert len(doubled) == 160
        out = issue((corpus, doubled, labels), space, order_seed=0, target=0.7)
        record = out.certificate.to_dict()
        assert record["queries"]["exact_duplicates"] == 80 and record["queries"]["n_used"] == 160
        sentence = next(c for c in record["caveats"] if "repeat an earlier query" in c)
        assert sentence.startswith("80 of the 160 queries tested repeat an earlier query exactly")
        assert "assumption A1 does not hold" in sentence
        assert "de-duplication" in sentence and "logs replayed twice" in sentence
        assert "augmentation" in sentence and "stated error level does not apply" in sentence
        report = out.verify()
        assert report.ok, report.summary()
        assert report.status("binding.queries") == "pass"
        assert "exact-duplicate count" in report.get("binding.queries").detail

    def test_the_count_is_the_same_whatever_the_order_and_is_over_the_queries_tested(self, space):
        corpus, doubled, labels = with_copies(make_problem(n_unlabelled=0))
        full = {issue((corpus, doubled, labels), space, order_seed=s, target=0.7)
                .certificate.to_dict()["queries"]["exact_duplicates"] for s in (None, 0, 5)}
        assert full == {80}
        out = issue((corpus, doubled, labels), space, order_seed=3, query_budget=60, target=0.7)
        recorded = out.certificate.to_dict()["queries"]["exact_duplicates"]
        tested = [out.evidence.query_ids[i]
                  for i in np.random.default_rng(3).permutation(160)[:60]]
        distinct = {q.removesuffix("_copy") for q in tested}  # a copy and its original are one
        assert recorded == len(tested) - len(distinct) and 0 < recorded < 60
        assert out.verify().ok

    def test_queries_removed_for_having_no_relevant_document_are_not_counted(self, space):
        corpus, doubled, labels = with_copies(make_problem(n_unlabelled=3))
        for q in ("q0", "q1", "q2"):  # no labels for the originals and no labels for their copies
            labels.pop(f"{q}_copy", None)
        out = issue((corpus, doubled, labels), space, order_seed=0, target=0.7)
        assert out.certificate.to_dict()["evidence"]["dropped_queries"] == [
            "q0", "q1", "q2", "q0_copy", "q1_copy", "q2_copy"]
        assert out.certificate.to_dict()["queries"]["exact_duplicates"] == 77

    def test_the_helper_compares_the_vectors_and_also_counts_a_repeated_id(self, problem):
        _, queries, _ = problem
        order = {"kind": "given", "seed": None, "budget": None, "n_available": 80,
                 "n_used": 80, "indices_sha256": array_digest(np.arange(80, dtype=np.int64))}
        assert exact_duplicates(queries, (), {**order, "n_available": 80}) == 0
        arrays = [queries.vectors[queries.offsets[i]:queries.offsets[i + 1]] for i in range(3)]
        same_vectors = MultiVectorCorpus.from_arrays([arrays[0], arrays[1], arrays[0]],
                                                     ids=["a", "b", "c"])
        short = {**order, "n_available": 3, "n_used": 3,
                 "indices_sha256": array_digest(np.arange(3, dtype=np.int64))}
        assert exact_duplicates(same_vectors, (), short) == 1
        same_id = MultiVectorCorpus.from_arrays(arrays, ids=["a", "b", "a"])  # not a valid corpus
        assert exact_duplicates(same_id, (), short) == 1
        near = arrays[0].copy()
        near[0, 0] += 1e-7  # a near-duplicate is not detected: that stays a documentation matter
        almost = MultiVectorCorpus.from_arrays([arrays[0], near, arrays[2]], ids=["a", "b", "c"])
        assert exact_duplicates(almost, (), short) == 0


class TestTheNormsOfTheVectorsAreRecorded:
    def test_unit_vectors_are_recorded_as_unit_with_their_extremes(self, selected):
        norms = selected.certificate.to_dict()["norms"]
        assert set(norms) == {"tolerance", "documents", "queries"}
        assert norms["tolerance"] == NORM_TOLERANCE == 1e-3
        for side in ("documents", "queries"):
            assert norms[side]["unit_norm"] is True
            assert abs(norms[side]["norm_min"] - 1.0) < 1e-6
            assert abs(norms[side]["norm_max"] - 1.0) < 1e-6
        assert not any("unit L2 norm" in c for c in selected.certificate.caveats)

    def test_documents_that_are_not_unit_norm_say_so_in_a_caveat(self, problem, space):
        corpus, queries, qrels = problem
        scaled = MultiVectorCorpus(corpus.vectors * np.float32(1.5), corpus.offsets,
                                   ids=corpus.ids)
        out = issue((scaled, queries, qrels), space)
        norms = out.certificate.to_dict()["norms"]
        assert norms["documents"]["unit_norm"] is False and norms["queries"]["unit_norm"] is True
        assert norms["documents"]["norm_min"] == pytest.approx(1.5, abs=1e-5)
        sentence = next(c for c in out.certificate.caveats if "unit L2 norm" in c)
        assert "document vectors have norms from 1.5 to 1.5" in sentence
        assert "query vectors" not in sentence and "tolerance 0.001" in sentence
        assert "only MaxSim over L2-normalised vectors" in sentence
        assert "nothing is known outside that scope" in sentence
        assert out.verify().ok

    def test_queries_that_are_not_unit_norm_say_so_too(self, problem, space):
        corpus, queries, qrels = problem
        wide = MultiVectorCorpus(queries.vectors * np.float32(0.5), queries.offsets,
                                 ids=queries.ids)
        out = issue((corpus, wide, qrels), space)
        sentence = next(c for c in out.certificate.caveats if "unit L2 norm" in c)
        assert "query vectors have norms from 0.5 to 0.5" in sentence
        assert "document vectors" not in sentence
        both = MultiVectorCorpus(corpus.vectors * np.float32(2.0), corpus.offsets, ids=corpus.ids)
        out = issue((both, wide, qrels), space)
        sentence = next(c for c in out.certificate.caveats if "unit L2 norm" in c)
        assert "document vectors" in sentence and "and query vectors" in sentence

    def test_the_tolerance_is_one_in_a_thousand_and_float16_unit_vectors_pass_it(self):
        base = np.eye(4, dtype=np.float32)
        inside = MultiVectorCorpus(base * np.float32(1.0009), np.array([0, 4]), ids=["a"])
        outside = MultiVectorCorpus(base * np.float32(1.0011), np.array([0, 4]), ids=["a"])
        assert norm_record(inside)["unit_norm"] is True
        assert norm_record(outside)["unit_norm"] is False
        rng = np.random.default_rng(0)
        x = rng.standard_normal((500, 128))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        half = MultiVectorCorpus(x.astype(np.float16), np.array([0, 500]), ids=["a"])
        assert norm_record(half)["unit_norm"] is True

    def test_the_norms_are_computed_in_blocks_and_equal_the_direct_computation(self, monkeypatch):
        from optivision.core.artifact import record as record_module

        rng = np.random.default_rng(4)
        x = (rng.standard_normal((1000, 32)) * rng.uniform(0.5, 1.5, (1000, 1))).astype(np.float32)
        corpus = MultiVectorCorpus(x, np.array([0, 1000]), ids=["a"])
        direct = np.linalg.norm(x.astype(np.float64), axis=1)
        whole = norm_record(corpus)
        monkeypatch.setattr(record_module, "_NORM_BLOCK_ELEMENTS", 32 * 7)  # seven rows a block
        blocks = norm_record(corpus)
        assert whole == blocks
        assert blocks["norm_min"] == pytest.approx(direct.min(), rel=1e-12)
        assert blocks["norm_max"] == pytest.approx(direct.max(), rel=1e-12)
        asked = []
        real = np.asarray

        def spy(a, *args, **kwargs):
            if kwargs.get("dtype") == np.float64:
                asked.append(len(a))
            return real(a, *args, **kwargs)

        monkeypatch.setattr(record_module.np, "asarray", spy)
        norm_record(corpus)
        assert max(asked) == 7 and len(asked) == 143  # float64 only ever for a block of rows

    def test_a_corpus_with_no_vectors_is_vacuously_unit(self):
        empty = MultiVectorCorpus(np.zeros((0, 4), dtype=np.float32), np.array([0, 0]), ids=["a"])
        assert norm_record(empty) == {"unit_norm": True, "norm_min": None, "norm_max": None}


# --------------------------------------------------------- creation refuses mismatches


class TestCreationCrossChecks:
    def kwargs(self, out: Issued, **change):
        base = {"evidence": out.evidence, "space": out.space, "corpus": out.corpus,
                "queries": out.queries, "reference": out.reference, "pipeline": out.pipeline,
                "compressed": out.compressed, "created": CREATED}
        base.update(change)
        return base

    def test_the_decision_must_come_from_this_space(self, selected, problem):
        other = small_space("renamed")  # same list, different name: same identity, accepted
        create_certificate(selected.decision, **self.kwargs(selected, space=other))
        different = type(selected.space)(
            "other", selected.space.candidates[:-1])  # one candidate fewer: another identity
        with pytest.raises(ValueError, match="different candidate space"):
            create_certificate(selected.decision, **self.kwargs(selected, space=different))

    def test_the_decision_must_come_from_this_evidence(self, selected, agreement):
        with pytest.raises(ValueError, match="different evidence"):
            create_certificate(selected.decision,
                               **self.kwargs(selected, evidence=agreement.evidence))

    def test_the_reference_must_measure_the_same_construct(self, selected):
        with pytest.raises(ValueError, match="measures 'agreement@1'"):
            create_certificate(selected.decision,
                               **self.kwargs(selected, reference=RankingReference()))

    def test_the_queries_must_be_those_the_evidence_was_collected_on(self, selected, problem):
        _, queries, _ = problem
        fewer = queries.select(list(range(1, len(queries))))
        with pytest.raises(ValueError, match="not the queries the evidence"):
            create_certificate(selected.decision, **self.kwargs(selected, queries=fewer))

    def test_dimension_mismatch_between_corpus_and_queries(self, selected, problem):
        _, queries, _ = problem
        wide = MultiVectorCorpus(np.zeros((queries.num_vectors, 20), np.float32), queries.offsets,
                                 ids=queries.ids)
        with pytest.raises(ValueError, match="dimension"):
            create_certificate(selected.decision, **self.kwargs(selected, queries=wide))

    def test_a_select_outcome_needs_the_pipeline_and_the_compressed_corpus(self, selected):
        with pytest.raises(ValueError, match="needs the fitted pipeline"):
            create_certificate(selected.decision, **self.kwargs(selected, pipeline=None))
        with pytest.raises(ValueError, match="needs the fitted pipeline"):
            create_certificate(selected.decision, **self.kwargs(selected, compressed=None))

    def test_an_abstention_takes_no_pipeline_and_no_compressed_corpus(self, abstained, selected):
        with pytest.raises(ValueError, match="abstention"):
            create_certificate(abstained.decision,
                               **self.kwargs(abstained, pipeline=selected.pipeline))
        with pytest.raises(ValueError, match="abstention"):
            create_certificate(abstained.decision,
                               **self.kwargs(abstained, compressed=selected.compressed))

    def test_the_pipeline_must_be_the_selected_candidates(self, selected, problem):
        wrong = Pipeline([Float16Quantizer()])
        wrong.compress(problem[0])
        with pytest.raises(ValueError, match="not the selected candidate's pipeline"):
            create_certificate(selected.decision, **self.kwargs(selected, pipeline=wrong))

    def test_the_pipeline_must_be_fitted(self, selected, space):
        unfitted = space.pipelines()[selected.decision.selected.position]
        with pytest.raises(ValueError, match="not fitted"):
            create_certificate(selected.decision, **self.kwargs(selected, pipeline=unfitted))

    def test_the_compressed_corpus_must_fit_the_corpus(self, selected, problem):
        corpus, _, _ = problem
        small = corpus.select(range(10))
        _, compressed = compress_selected(small, selected.decision.selected)
        with pytest.raises(ValueError, match="does not match the corpus"):
            create_certificate(selected.decision, **self.kwargs(selected, compressed=compressed))

    def test_the_compressed_corpus_must_belong_to_the_pipeline(self, selected, problem):
        corpus, _, _ = problem
        _, other = compress_selected(corpus, Pipeline([BinaryQuantizer()]))
        with pytest.raises(ValueError, match="different pipeline"):
            create_certificate(selected.decision, **self.kwargs(selected, compressed=other))

    def test_the_fit_must_be_the_one_that_was_evaluated(self, selected):
        other_corpus, _, _ = make_problem(seed=5)  # same shape and ids, different vectors
        pipeline, compressed = compress_selected(other_corpus, selected.decision.selected)
        with pytest.raises(ValueError, match="not the one that was evaluated"):
            create_certificate(selected.decision,
                               **self.kwargs(selected, pipeline=pipeline, compressed=compressed))

    def test_wrong_argument_types(self, selected):
        with pytest.raises(TypeError, match="decision"):
            create_certificate("select", **self.kwargs(selected))
        with pytest.raises(TypeError, match="evidence"):
            create_certificate(selected.decision, **self.kwargs(selected, evidence=3))
        with pytest.raises(TypeError, match="pipeline"):
            create_certificate(selected.decision, **self.kwargs(selected, pipeline="x"))
        with pytest.raises(TypeError, match="compressed"):
            create_certificate(selected.decision, **self.kwargs(selected, compressed="x"))
        with pytest.raises(TypeError, match="reference"):
            create_certificate(selected.decision, **self.kwargs(selected, reference={}))


# ----------------------------------------------------------------- serialisation


class TestSerialisation:
    @pytest.mark.parametrize("which", ["selected", "agreement", "abstained", "fell_back"])
    def test_json_round_trip_keeps_every_value_and_the_digest_is_stable(self, request, which):
        out = request.getfixturevalue(which)
        c = out.certificate
        for indent in (2, None, 0):
            again = Certificate.from_json(c.to_json(indent=indent))
            assert again == c and again.digest == c.digest
            assert again.to_dict() == c.to_dict()
        assert Certificate.from_dict(c.to_dict()) == c
        assert out.verify(Certificate.from_json(c.to_json())).ok

    def test_floats_survive_exactly(self, selected):
        c = selected.certificate
        again = Certificate.from_json(c.to_json())
        assert again.p_values == c.p_values and again.level == c.level
        assert content_digest_of(again) == again.digest

    def test_save_and_load_then_verify(self, selected, tmp_path):
        path = selected.certificate.save(tmp_path / "sub" / "certificate.json")
        assert path.is_file()
        raw = path.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"}\n")
        raw.decode("ascii")  # ASCII JSON: non-ASCII text would be escaped
        loaded = Certificate.load(path)
        assert loaded == selected.certificate
        report = selected.verify(loaded)
        assert report.ok and report.level == "binding"

    def test_the_file_is_in_a_readable_order(self, selected):
        text = selected.certificate.to_json()
        assert text.startswith('{\n  "schema": "optivision-certificate"')
        keys = list(json.loads(text))
        assert keys[-2:] == ["digest", "signature"]

    def test_a_loaded_certificate_with_a_crlf_conversion_still_parses(self, selected, tmp_path):
        text = selected.certificate.to_json().replace("\n", "\r\n")
        assert Certificate.from_json(text) == selected.certificate

    def test_duplicate_keys_and_non_finite_numbers_are_refused(self, selected):
        text = selected.certificate.to_json(indent=None)
        with pytest.raises(CertificateFormatError, match="duplicate key"):
            Certificate.from_json(text[:-1] + ', "target": 0.5}')
        poisoned = text.replace('"target": 0.8', '"target": NaN')
        assert poisoned != text
        with pytest.raises(CertificateFormatError, match="NaN"):
            Certificate.from_json(poisoned)
        with pytest.raises(CertificateFormatError, match="not a readable certificate"):
            Certificate.from_json("{not json")
        with pytest.raises(CertificateFormatError, match="JSON object"):
            Certificate.from_json("[1, 2]")
        with pytest.raises(CertificateFormatError, match="NaN or infinity"):
            Certificate.from_dict({"target": float("nan")})

    def test_a_file_that_is_not_utf8_is_refused(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_bytes(b"\xff\xfe\x00")
        with pytest.raises(CertificateFormatError, match="UTF-8"):
            Certificate.load(path)

    def test_the_record_is_immutable(self, selected):
        c = selected.certificate
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.data = {}
        with pytest.raises(TypeError):
            c.data["target"] = 0.1
        with pytest.raises(TypeError):
            c.data["result"]["p_values"] = ()
        assert isinstance(c.data["result"]["p_values"], tuple)
        copy = c.to_dict()
        copy["target"] = 0.1
        copy["result"]["p_values"].append(0.5)
        assert c.target == 0.8 and len(c.p_values) == 7  # to_dict is a deep copy

    def test_equality_and_hash(self, selected, abstained):
        c = selected.certificate
        twin = Certificate.from_dict(c.to_dict())
        assert c == twin and hash(c) == hash(twin) and c != abstained.certificate
        assert c != "certificate" and c != sign_certificate(c, KEY)
        assert len({c, twin, abstained.certificate}) == 2

    def test_accessors_on_a_malformed_record_raise_a_clear_error(self):
        broken = Certificate.from_dict({"schema": "optivision-certificate"})
        with pytest.raises(CertificateFormatError, match="no field result.outcome"):
            _ = broken.outcome
        assert "malformed" in repr(broken)

    def test_the_repr_is_short(self, selected):
        text = repr(selected.certificate)
        assert "select" in text and "certified=True" in text and len(text) < 160


def content_digest_of(cert: Certificate) -> str:
    d = cert.to_dict()
    return digest({k: v for k, v in d.items() if k not in ("digest", "signature")})


# ----------------------------------------------------------------- determinism


class TestDeterminism:
    def test_the_same_inputs_and_created_give_byte_identical_certificates(self, problem, space):
        a = issue(problem, space)
        b = issue(problem, space)
        assert a.certificate.to_json() == b.certificate.to_json()
        assert a.certificate.digest == b.certificate.digest
        assert a.certificate == b.certificate

    def test_created_can_be_a_string_or_an_aware_datetime(self, problem, space):
        as_string = issue(problem, space, created="2026-10-06T12:00:00Z").certificate
        as_dt = issue(problem, space, created=dt.datetime(2026, 10, 6, 12, 0, 0,
                                                          tzinfo=dt.timezone.utc)).certificate
        offset = dt.timezone(dt.timedelta(hours=2))
        as_local = issue(problem, space, created=dt.datetime(2026, 10, 6, 14, 0, 0, 999,
                                                             tzinfo=offset)).certificate
        assert as_string == as_dt == as_local and as_local.created_utc == CREATED

    @pytest.mark.parametrize("bad", ["2026-10-06", "2026-10-06T12:00:00", "yesterday",
                                     "2026-10-06T12:00:00+02:00", "2026-13-06T12:00:00Z", ""])
    def test_a_malformed_created_string_is_refused(self, problem, space, bad):
        with pytest.raises(ValueError, match="created must look like"):
            issue(problem, space, created=bad)

    def test_a_naive_datetime_and_other_types_are_refused(self, problem, space):
        with pytest.raises(ValueError, match="naive datetime"):
            issue(problem, space, created=dt.datetime(2026, 10, 6, 12, 0, 0))  # noqa: DTZ001
        with pytest.raises(TypeError, match="created"):
            issue(problem, space, created=20261006)

    def test_without_created_the_clock_gives_a_utc_second(self, problem, space):
        before = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        out = issue(problem, space, created=None)
        after = dt.datetime.now(dt.timezone.utc)
        stamp = out.certificate.created_utc
        assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", stamp)
        parsed = dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z")
        assert before <= parsed <= after
        assert out.verify().ok

    def test_inputs_are_not_modified(self, problem, space):
        corpus, queries, _ = problem
        before = (corpus_identity(corpus), corpus_identity(queries), space.identity)
        issue(problem, space)
        assert (corpus_identity(corpus), corpus_identity(queries), space.identity) == before

    def test_a_different_target_gives_a_different_certificate(self, problem, space, selected):
        other = issue(problem, space, target=0.7)
        assert other.certificate.digest != selected.certificate.digest


# --------------------------------------------------------------------- signing


class TestSigning:
    def test_a_signed_certificate_verifies_with_the_key_only(self, selected):
        signed = sign_certificate(selected.certificate, KEY, key_id="k1")
        assert signed.signed and signed.digest == selected.certificate.digest
        assert signed.signature == {
            "algorithm": "HMAC-SHA256", "key_id": "k1",
            "value": hmac.new(KEY, signed.digest.encode("ascii"), hashlib.sha256).hexdigest()}
        report = selected.verify(signed, key=KEY)
        assert report.ok and report.status("signature") == "pass"
        assert "symmetric" in report.get("signature").detail

    def test_a_signed_certificate_without_a_key_reports_the_signature_as_skipped(self, selected):
        signed = sign_certificate(selected.certificate, KEY)
        report = selected.verify(signed)
        assert report.ok and report.status("signature") == "skipped"
        assert "signed" in report.get("signature").detail

    def test_a_wrong_key_fails(self, selected):
        signed = sign_certificate(selected.certificate, KEY)
        report = selected.verify(signed, key=b"x" * 32)
        assert not report.ok and report.status("signature") == "fail"
        assert [c.name for c in report.failed()] == ["signature"]

    def test_an_unsigned_certificate_fails_when_a_key_is_supplied(self, selected):
        report = selected.verify(selected.certificate, key=KEY)
        assert not report.ok and report.status("signature") == "fail"
        assert "no signature" in report.get("signature").detail

    def test_the_comparison_is_constant_time(self, selected, monkeypatch):
        seen = []
        real = hmac.compare_digest

        def spy(a, b):
            seen.append((type(a), type(b)))
            return real(a, b)

        monkeypatch.setattr(verification_module.hmac, "compare_digest", spy)
        selected.verify(sign_certificate(selected.certificate, KEY), key=KEY)
        assert seen == [(bytes, bytes)]

    def test_an_unknown_algorithm_fails(self, selected):
        d = sign_certificate(selected.certificate, KEY).to_dict()
        d["signature"]["algorithm"] = "HMAC-MD5"
        report = selected.verify(Certificate.from_dict(d), key=KEY)
        assert report.status("signature") == "fail" and "algorithm" in report.get(
            "signature").detail

    def test_signing_does_not_change_what_is_hashed(self, selected):
        signed = sign_certificate(selected.certificate, KEY)
        assert content_digest_of(signed) == signed.digest

    def test_key_must_be_non_empty_bytes(self, selected):
        with pytest.raises(TypeError, match="bytes"):
            sign_certificate(selected.certificate, "secret")
        with pytest.raises(ValueError, match="empty"):
            sign_certificate(selected.certificate, b"")
        with pytest.raises(TypeError, match="key"):
            selected.verify(key="secret")
        with pytest.raises(TypeError, match="key_id"):
            sign_certificate(selected.certificate, KEY, key_id=7)
        with pytest.raises(TypeError, match="Certificate"):
            sign_certificate({}, KEY)
        sign_certificate(selected.certificate, bytearray(KEY))  # bytes-like is accepted

    def test_a_modified_record_is_not_signed(self, selected):
        d = selected.certificate.to_dict()
        d["target"] = 0.5
        with pytest.raises(ValueError, match="does not match its content"):
            sign_certificate(Certificate.from_dict(d), KEY)

    def test_re_signing_replaces_the_signature(self, selected):
        signed_a = sign_certificate(selected.certificate, KEY, key_id="a")
        signed_b = sign_certificate(signed_a, b"z" * 32, key_id="b")
        assert signed_b.signature["key_id"] == "b" and signed_b.signature != signed_a.signature
        assert selected.verify(signed_b, key=b"z" * 32).ok

    def test_a_signed_certificate_survives_save_and_load(self, selected, tmp_path):
        signed = sign_certificate(selected.certificate, KEY, key_id="k1")
        loaded = Certificate.load(signed.save(tmp_path / "c.json"))
        assert loaded == signed and selected.verify(loaded, key=KEY).ok

    def test_the_key_id_is_a_label_the_signature_does_not_cover(self, selected):
        d = sign_certificate(selected.certificate, KEY, key_id="a").to_dict()
        d["signature"]["key_id"] = "b"
        assert selected.verify(Certificate.from_dict(d), key=KEY).ok  # documented: not covered


# ----------------------------------------------------------- levels and the report


class TestLevelsAreReportedByWhatIsSupplied:
    def test_the_certificate_alone_is_level_integrity(self, selected):
        report = verify_certificate(selected.certificate)
        assert report.ok and report.level == "integrity" and report.levels == ("integrity",)
        for name in ("evidence.digest", "evidence.fields", "evidence.sample", "replay.decision",
                     "binding.corpus", "binding.queries", "binding.reference",
                     "binding.pipeline", "binding.codes", "binding.bytes",
                     "binding.evidence_queries"):
            assert report.status(name) == "skipped", name
        assert report.status("schema") == report.status("digest") == "pass"

    def test_evidence_adds_the_replay_level(self, selected):
        report = verify_certificate(selected.certificate, evidence=selected.evidence)
        assert report.ok and report.level == "replay" and report.levels == ("integrity", "replay")
        assert [report.status(n) for n in ("evidence.digest", "evidence.fields",
                                           "evidence.sample", "replay.decision")] == ["pass"] * 4

    @pytest.mark.parametrize("name,kwargs,check", [
        ("corpus", {"corpus": "corpus"}, "binding.corpus"),
        ("queries", {"queries": "queries"}, "binding.queries"),
        ("reference", {"reference": "reference"}, "binding.reference"),
        ("pipeline", {"pipeline": "pipeline"}, "binding.pipeline"),
        ("compressed", {"compressed": "compressed"}, "binding.codes"),
    ])
    def test_each_live_object_is_checked_on_its_own(self, selected, name, kwargs, check):
        objects = {k: getattr(selected, v) for k, v in kwargs.items()}
        report = verify_certificate(selected.certificate, **objects)
        assert report.ok and report.status(check) == "pass"
        assert report.levels == ("integrity", "binding")  # binding reached without a replay
        others = [c.name for c in report.checks
                  if c.level == "binding" and c.status != "skipped"]
        assert others == [check]

    def test_both_pipeline_and_compressed_add_the_byte_figures(self, selected):
        report = verify_certificate(selected.certificate, pipeline=selected.pipeline,
                                    compressed=selected.compressed)
        assert report.status("binding.bytes") == "pass"

    def test_queries_and_evidence_together_check_their_ids(self, selected):
        report = verify_certificate(selected.certificate, queries=selected.queries,
                                    evidence=selected.evidence)
        assert report.status("binding.evidence_queries") == "pass"

    def test_everything_is_level_binding(self, selected):
        report = selected.verify()
        assert report.levels == ("integrity", "replay", "binding") and report.level == "binding"

    def test_abstention_fails_a_binding_to_objects_that_cannot_belong_to_it(self, abstained,
                                                                            selected):
        report = verify_certificate(abstained.certificate, pipeline=selected.pipeline,
                                    compressed=selected.compressed)
        assert not report.ok
        assert report.status("binding.pipeline") == "fail"
        assert report.status("binding.codes") == "fail"
        assert "abstain" in report.get("binding.pipeline").detail


class TestTheReport:
    def test_ok_failed_get_and_status(self, selected):
        report = selected.verify()
        assert report.ok and report.failed() == () and report.differing() == ()
        assert isinstance(report.get("digest"), Check)
        assert report.status("digest") == "pass"
        with pytest.raises(KeyError, match="no check named"):
            report.get("nonexistent")

    def test_the_summary_is_readable(self, selected):
        text = selected.verify().summary()
        assert text.startswith("Certificate verification: OK")
        assert "level reached: binding" in text
        for name in ("schema", "digest", "replay.decision", "binding.pipeline"):
            assert name in text
        assert "does not make the certificate true" in text
        assert str(selected.verify()) == text

    def test_a_failed_report_says_so(self, selected):
        d = selected.certificate.to_dict()
        d["target"] = 0.5
        report = selected.verify(Certificate.from_dict(d))
        assert not report.ok and report.summary().startswith("Certificate verification: FAILED")
        assert report.failed()[0].name == "digest"
        assert report.to_dict()["ok"] is False
        assert {"name", "status", "detail", "level"} == set(report.to_dict()["checks"][0])

    def test_a_check_validates_its_status_and_level(self):
        with pytest.raises(ValueError, match="status"):
            Check("x", "maybe")
        with pytest.raises(ValueError, match="level"):
            Check("x", "pass", "", "deep")
        assert Check("x", "differs").status == "differs"

    def test_the_report_is_immutable(self, selected):
        report = selected.verify()
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.checks = ()
        assert isinstance(report.checks, tuple)


class TestVerificationNeverRaisesOnABadCertificate:
    def test_wrong_argument_types_are_misuse(self, selected):
        with pytest.raises(TypeError, match="Certificate"):
            verify_certificate("certificate.json")
        with pytest.raises(TypeError, match="evidence"):
            verify_certificate(selected.certificate, evidence=selected.pipeline)
        with pytest.raises(TypeError, match="corpus"):
            verify_certificate(selected.certificate, corpus=selected.compressed)
        with pytest.raises(TypeError, match="reference"):
            verify_certificate(selected.certificate, reference="labels")
        with pytest.raises(TypeError, match="pipeline"):
            verify_certificate(selected.certificate, pipeline=selected.compressed)
        with pytest.raises(TypeError, match="compressed"):
            verify_certificate(selected.certificate, compressed=selected.pipeline)
        with pytest.raises(TypeError, match="strict_code"):
            verify_certificate(selected.certificate, strict_code="yes")

    def test_an_empty_or_foreign_record_is_reported(self):
        for record in ({}, {"schema": "something-else"}, {"schema": "optivision-certificate"}):
            report = verify_certificate(Certificate.from_dict(record))
            assert not report.ok and report.status("schema") == "fail"
            assert report.status("remaining") == "skipped"

    @pytest.mark.parametrize("field", SCHEMA_V1_FIELDS)
    @pytest.mark.parametrize("mutation", ["delete", "none", "text", "list"])
    def test_a_damaged_top_level_field_is_reported_not_raised(self, selected, field, mutation):
        original = selected.certificate.to_dict()
        d = selected.certificate.to_dict()
        if mutation == "delete":
            del d[field]
        else:
            d[field] = {"none": None, "text": "x", "list": []}[mutation]
        if field not in ("digest", "signature"):
            d["digest"] = content_digest_of_dict(d)  # even with the digest made consistent
        report = verify_certificate(Certificate.from_dict(d))
        unchanged = field in d and d[field] == original[field]  # e.g. caveats [] -> []
        if not unchanged:
            assert not report.ok, (field, mutation)

    def test_unexpected_extra_fields_are_reported(self, selected):
        d = selected.certificate.to_dict()
        d["note"] = "approved"
        report = verify_certificate(Certificate.from_dict(d))
        assert report.status("schema") == "fail" and "unexpected field" in report.get(
            "schema").detail
        d["digest"] = content_digest_of_dict(d)
        assert verify_certificate(Certificate.from_dict(d)).status("schema") == "fail"


def content_digest_of_dict(d: dict) -> str:
    return digest({k: v for k, v in d.items() if k not in ("digest", "signature")})


# ------------------------------------------------------------- code and libraries


class TestTheCodeIdentity:
    def reissue(self, out: Issued, **change) -> Certificate:
        d = out.certificate.to_dict()
        for key, value in change.items():
            section, _, name = key.partition("__")
            d[section][name] = value
        d["digest"] = content_digest_of_dict(d)
        return Certificate.from_dict(d)

    def test_the_issuing_code_is_the_running_code(self, selected):
        report = selected.verify()
        assert report.status("code.files") == "pass" and report.status("code.libraries") == "pass"
        code = selected.certificate.to_dict()["code"]
        assert set(code) == {"digest", "files", "python", "numpy", "scipy"}
        assert HEX64.match(code["digest"]) and code["digest"] == digest(code["files"])

    def test_a_different_code_digest_is_reported_as_differs_not_as_a_failure(self, selected):
        other = self.reissue(selected, code__digest="0" * 64)
        report = selected.verify(other)
        # the table no longer hashes to code.digest: that inconsistency is a failure
        assert report.status("consistency.identities") == "fail"
        files = selected.certificate.to_dict()["code"]["files"]
        files = {**files, "core/identity.py": "0" * 64}
        other = self.reissue(selected, code__files=files, code__digest=digest(files))
        report = selected.verify(other)
        assert report.ok, report.summary()
        assert report.status("code.files") == "differs"
        assert "changed: core/identity.py" in report.get("code.files").detail
        assert statuses(report, "fail") == []

    def test_strict_code_turns_the_difference_into_a_failure(self, selected):
        files = {**selected.certificate.to_dict()["code"]["files"], "core/identity.py": "0" * 64}
        other = self.reissue(selected, code__files=files, code__digest=digest(files))
        report = selected.verify(other, strict_code=True)
        assert not report.ok and [c.name for c in report.failed()][:1] == ["code.files"]

    def test_added_and_removed_files_are_named(self, selected):
        files = dict(selected.certificate.to_dict()["code"]["files"])
        files.pop("core/identity.py")
        files["core/older_module.py"] = "1" * 64
        other = self.reissue(selected, code__files=files, code__digest=digest(files))
        detail = selected.verify(other).get("code.files").detail
        assert "added since: core/identity.py" in detail
        assert "removed since: core/older_module.py" in detail

    def test_unreadable_sources_at_issue_time_cannot_be_compared(self, selected):
        other = self.reissue(selected, code__files={}, code__digest=None,
                             code__note="source files could not be read: test")
        report = selected.verify(other)
        assert report.ok and report.status("code.files") == "differs"
        assert "unavailable" in report.get("code.files").detail
        assert selected.verify(other, strict_code=True).status("code.files") == "fail"

    def test_library_versions_are_information_only(self, selected):
        other = self.reissue(selected, code__numpy="0.0.1", code__python="2.7.18")
        report = selected.verify(other, strict_code=True)
        assert report.status("code.libraries") == "differs"
        detail = report.get("code.libraries").detail
        assert "numpy 0.0.1 (issued)" in detail and "python 2.7.18 (issued)" in detail
        assert report.ok  # the code digest is unchanged, and libraries never fail a report


class TestPublicSurface:
    def test_the_artifact_package_exports_the_documented_names(self):
        for name in ("Artifact", "ArtifactError", "Certificate", "CertificateFormatError", "Check",
                     "VerificationReport", "code_identity", "create_certificate",
                     "load_artifact", "package_info", "save_artifact", "sign_certificate",
                     "verify_certificate"):
            assert name in artifact.__all__ and hasattr(artifact, name), name
