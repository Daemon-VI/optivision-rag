"""``optivision.certify``: end to end on small seeded synthetic data.

The data say nothing about any encoder (``optivision.testing`` makes them); these tests check
that the pieces fit together and that the result says what it is: a selection, an abstention or
an uncertified fallback, with the reference it was tested against and a certificate that checks.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import inspect
import json
import warnings

import numpy as np
import pytest

import optivision
from optivision import api as api_module
from optivision import certify, load_artifact, verify, verify_certificate
from optivision.adapters import corpus_from_arrays
from optivision.compose import CompressedCorpus, Pipeline
from optivision.core.artifact import ArtifactError, sign_certificate
from optivision.core.evaluate import LabelReference, RankingReference
from optivision.core.identity import array_digest
from optivision.core.retrieve import CompressedRetriever, ExactRetriever, Ranking
from optivision.core.selection import (
    DEFAULT_ORDER_SEED,
    BonferroniBetting,
    CandidateSpace,
    ErrorLevelError,
    Outcome,
    QueryOrderError,
    TargetError,
)
from optivision.core.selection import spaces as spaces_module
from optivision.core.vectors import ContractError, EncoderIdentity, corpus_identity
from optivision.representation import MultiVectorCorpus
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    Float32Quantizer,
    Int8Quantizer,
)
from optivision.testing import make_corpus, make_queries

CREATED = "2026-10-06T00:00:00Z"

try:
    import scipy.cluster.hierarchy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAVE_SCIPY, reason="the default candidate space needs SciPy")


def small_space():
    """Four families, ordered from least to most aggressive; fast to evaluate."""
    return {
        "exact": [Pipeline([Float32Quantizer()])],
        "float16": [Pipeline([Float16Quantizer()])],
        "int8": [Pipeline([Int8Quantizer("per_vector")])],
        "merge+binary": [Pipeline([AdaptiveMerge(radius=r), BinaryQuantizer()]) for r in (0.9, 0.7, 0.4)],
    }


def run(corpus, queries, qrels=None, **kwargs):
    kwargs.setdefault("target", 0.9)
    kwargs.setdefault("space", small_space())
    kwargs.setdefault("created", CREATED)
    return certify(corpus, queries, qrels, **kwargs)


@pytest.fixture(scope="module")
def data():
    corpus = make_corpus(40, 16, tokens=(8, 12), seed=3)
    queries, qrels = make_queries(corpus, 120, noise=1.0, seed=4)
    return corpus, queries, qrels


@pytest.fixture(scope="module")
def selected(data):
    corpus, queries, qrels = data
    return run(corpus, queries, qrels)


@pytest.fixture(scope="module")
def abstained(data):
    corpus, queries, qrels = data
    return run(corpus, queries.select(range(8)), qrels, target=0.999)


# ------------------------------------------------------------------------ outcomes


def test_a_generous_target_with_enough_queries_selects_and_certifies(selected, data):
    corpus, queries, _ = data
    r = selected
    assert r.outcome is Outcome.SELECT and r.decision.outcome is Outcome.SELECT
    assert r.certified is True and r.decision.certified is True
    assert r.selected is not None and r.selected.position >= 0
    assert isinstance(r.compressed, CompressedCorpus)
    assert isinstance(r.pipeline, Pipeline) and r.pipeline.fitted
    assert r.store is r.compressed and r.corpus is corpus
    assert r.compression_ratio > 1.0
    assert r.certificate.certified and r.certificate.outcome is Outcome.SELECT
    assert r.certificate.n_queries == len(queries) == r.decision.n_queries
    assert r.evidence.n_queries == len(queries)
    assert r.pipeline.to_dict() == r.selected.to_dict()["pipeline"]


def test_the_selected_candidate_is_the_smallest_the_rule_certified(selected):
    d = selected.decision
    sizes = selected.evidence.bytes_per_doc
    assert d.selected.position in d.certified_positions
    assert sizes[d.selected.position] == min(sizes[k] for k in d.certified_positions)


def test_nothing_certified_abstains_without_compressing_anything(abstained, data):
    corpus, queries, _ = data
    r = abstained
    assert r.outcome is Outcome.ABSTAIN and r.certified is False
    assert r.selected is None and r.decision.selected is None
    assert r.compressed is None and r.pipeline is None
    assert r.store is r.corpus is corpus  # the uncompressed index is the outcome: no copy
    assert r.compression_ratio == 1.0
    assert r.certificate.outcome is Outcome.ABSTAIN and r.certificate.certified is False
    assert r.certificate.to_dict()["candidate"] is None
    assert verify(r.certificate, evidence=r.evidence, corpus=corpus, queries=queries.select(range(8)),
                  reference=LabelReference(data[2])).ok


def test_abstaining_is_silent(data):
    corpus, queries, qrels = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = run(corpus, queries.select(range(8)), qrels, target=0.999)
    assert r.outcome is Outcome.ABSTAIN


def test_the_abstention_says_what_it_means_and_what_it_does_not(abstained):
    rep = abstained.report()
    assert rep["outcome"] == "abstain" and rep["certified"] is False
    assert rep["selected"] is None and rep["compression_ratio"] == 1.0
    assert rep["basis_sensitive"] is None
    note = rep["not_certified_note"]
    assert "uncompressed index" in note and "safe outcome" in note
    # more queries means NEW queries, and another target is a decision taken before looking
    assert "new, independent queries" in note and "chosen before they are looked at" in note
    assert "Every further look at the same queries" in note and "outside the statement" in note
    assert "order or seed" in note and "up to 14" in note
    assert "a lower target" not in note and "more queries" not in note
    assert "nothing certified: use the uncompressed index" in repr(abstained)


def test_an_explicit_fallback_is_returned_and_is_not_certified(data):
    corpus, queries, qrels = data
    r = run(corpus, queries.select(range(8)), qrels, target=0.999, fallback=Pipeline([Float16Quantizer()]))
    assert r.outcome is Outcome.FALLBACK and r.certified is False
    assert r.selected.position == -1 and r.selected.family == "fallback"
    assert isinstance(r.compressed, CompressedCorpus) and r.pipeline.fitted
    assert r.store is r.compressed
    assert r.certificate.outcome is Outcome.FALLBACK and r.certificate.certified is False
    assert r.decision.certified_positions == ()
    rep = r.report()
    assert rep["not_certified_note"] and "fallback" in rep["not_certified_note"]
    assert any("fallback" in c and "not certified" in c for c in rep["caveats"])
    assert "NOT certified" in repr(r)
    assert verify(r.certificate, evidence=r.evidence).ok


def test_a_fallback_may_be_given_as_a_dict_and_is_ignored_when_something_is_certified(data, selected):
    corpus, queries, qrels = data
    as_dict = run(corpus, queries, qrels, fallback=Pipeline([Float16Quantizer()]).to_dict())
    assert as_dict.outcome is Outcome.SELECT
    assert as_dict.selected == selected.selected and as_dict.decision.certified is True


@pytest.mark.parametrize("bad", [3, "float16", [Pipeline([])]])
def test_a_fallback_of_the_wrong_type_is_refused_before_any_work(data, monkeypatch, bad):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(TypeError, match="fallback"):
        run(corpus, queries, qrels, fallback=bad)


def test_a_fallback_description_with_an_unknown_stage_is_refused_before_any_work(data, monkeypatch):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(ValueError, match="fallback.*unknown stage"):
        run(corpus, queries, qrels, fallback={"stages": [{"stage": "no_such_stage"}]})


# ---------------------------------------------------------------------- the reference


def test_without_qrels_the_reference_is_agreement_and_the_result_says_so(data):
    corpus, queries, _ = data
    r = run(corpus, queries)
    assert r.decision.construct == "agreement@1" and r.certificate.construct == "agreement@1"
    rep = r.report()
    assert rep["construct"] == "agreement@1"
    assert any("agreement with the uncompressed ranking" in c and "not retrieval quality" in c
               for c in rep["caveats"])
    assert "agreement with the uncompressed ranking, not retrieval quality" in repr(r)
    assert any("not retrieval quality" in c for c in r.certificate.caveats)
    assert verify(r.certificate, evidence=r.evidence).ok


def test_with_qrels_the_reference_is_the_labels_and_nothing_is_said_about_agreement(selected):
    assert selected.decision.construct == "labels"
    assert selected.report()["caveats"] == []
    assert "agreement" not in repr(selected)


def test_reference_objects_and_strings_are_all_accepted(data):
    corpus, queries, qrels = data
    assert run(corpus, queries, qrels, reference="labels").decision.construct == "labels"
    assert run(corpus, queries, reference="agreement").decision.construct == "agreement@1"
    assert run(corpus, queries, reference=LabelReference(qrels)).decision.construct == "labels"
    deeper = run(corpus, queries, reference=RankingReference(depth=2), target=0.5)
    assert deeper.decision.construct == "agreement@2"


def test_labels_without_qrels_are_refused_before_any_work(data, monkeypatch):
    corpus, queries, _ = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(ValueError, match="reference='labels' needs qrels"):
        run(corpus, queries, reference="labels")


def test_agreement_with_qrels_is_refused_rather_than_ignoring_the_labels(data, monkeypatch):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(ValueError, match="does not use them"):
        run(corpus, queries, qrels, reference="agreement")
    with pytest.raises(ValueError, match="either qrels or a reference object"):
        run(corpus, queries, qrels, reference=LabelReference(qrels))


@pytest.mark.parametrize("bad", ["relevance", "Labels", 3, None, ["labels"]])
def test_an_unknown_reference_is_refused_before_any_work(data, monkeypatch, bad):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises((ValueError, TypeError), match="reference must be"):
        run(corpus, queries, qrels, reference=bad)


def test_queries_without_a_relevant_document_are_dropped_and_counted(data):
    corpus, queries, qrels = data
    labelled = {k: v for k, v in qrels.items() if k not in set(queries.ids[:10])}
    r = run(corpus, queries, labelled)
    rep = r.report()
    assert rep["n_queries"] == r.decision.n_queries == 110
    assert rep["n_queries_given"] == 120 and rep["n_queries_without_relevant_document"] == 10
    assert r.evidence.dropped_queries == tuple(queries.ids[:10])
    assert r.certificate.to_dict()["evidence"]["dropped_queries"] == list(queries.ids[:10])


def test_untested_queries_are_reported_by_a_warning_that_names_the_counts(data):
    """Labels whose ids match only some queries must not shrink the sample silently."""
    corpus, queries, qrels = data
    labelled = {k: v for k, v in qrels.items() if k not in set(queries.ids[:10])}
    with pytest.warns(UserWarning, match=r"10 of the 120 queries given have no relevant "
                                         r"document.*\(110 were\).*ids"):
        run(corpus, queries, labelled)
    with warnings.catch_warnings(record=True) as caught:  # every query labelled: nothing to say
        warnings.simplefilter("always")
        run(corpus, queries, qrels)
    assert not [w for w in caught if "have no relevant document" in str(w.message)]


def test_labels_that_match_no_query_are_an_error_not_an_abstention(data):
    corpus, queries, _ = data
    with pytest.raises(ValueError, match="relevant document"):
        run(corpus, queries, {"nobody": {"nothing"}})


# ----------------------------------------------------------------- the candidate list


def test_a_dict_of_pipelines_becomes_a_recorded_custom_space(selected):
    record = selected.certificate.to_dict()["space"]
    expected = CandidateSpace.from_families(small_space())
    assert record["name"] == "custom" and record["registered_default"] is False
    assert record["identity"] == expected.identity == selected.decision.space_identity
    assert record["n_candidates"] == 6 and record["n_families"] == 4
    assert record["families"] == ["exact", "float16", "int8", "merge+binary"]


def test_a_candidate_space_object_gives_the_same_decision_as_the_dict(data, selected):
    corpus, queries, qrels = data
    again = run(corpus, queries, qrels, space=CandidateSpace.from_families(small_space()))
    assert again.certificate.digest == selected.certificate.digest


@needs_scipy
def test_the_default_space_is_the_registered_list(data):
    corpus, queries, qrels = data
    r = certify(corpus, queries, qrels, target=0.9, created=CREATED)
    record = r.certificate.to_dict()["space"]
    assert record["name"] == "default" and record["registered_default"] is True
    assert record["n_candidates"] == 40 and record["n_families"] == 7
    assert r.outcome is Outcome.SELECT and r.certified
    assert verify(r.certificate, evidence=r.evidence, corpus=corpus, queries=queries).ok


def test_without_scipy_the_default_list_is_refused_with_advice_and_never_reduced(data, monkeypatch):
    corpus, queries, qrels = data
    monkeypatch.setattr(spaces_module, "_have_scipy", lambda: False)
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(ImportError) as info:
        certify(corpus, queries, qrels, target=0.9)
    message = str(info.value)
    assert "SciPy" in message and "optivision-rag[merge]" in message
    assert "allow_reduced=True" in message and "never reduced silently" in message


def test_the_reduced_list_can_be_chosen_explicitly_and_is_recorded_as_such(data, monkeypatch):
    corpus, queries, qrels = data
    monkeypatch.setattr(spaces_module, "_have_scipy", lambda: False)
    r = certify(corpus, queries, qrels, target=0.9, created=CREATED,
                space=CandidateSpace.default(corpus.dimension, allow_reduced=True))
    record = r.certificate.to_dict()["space"]
    assert record["name"] == "default-reduced" and record["registered_default"] is False
    assert record["n_families"] == 4


@pytest.mark.parametrize("bad", [3, "default", ["exact"], {"exact": Pipeline([])}, {}])
def test_a_space_of_the_wrong_kind_is_refused_before_any_work(data, monkeypatch, bad):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises((TypeError, ValueError)):
        run(corpus, queries, qrels, space=bad)


# ------------------------------------------------------------- query order and budget


def test_query_budget_and_order_seed_are_recorded(data):
    corpus, queries, qrels = data
    r = run(corpus, queries, qrels, query_budget=60, order_seed=5)
    order = dict(r.decision.query_order)
    assert (order["kind"], order["seed"], order["budget"]) == ("seeded_permutation", 5, 60)
    assert r.decision.n_queries == r.certificate.n_queries == 60
    assert r.report()["query_order"] == {"kind": "seeded_permutation", "seed": 5, "budget": 60}
    recorded = r.certificate.to_dict()["queries"]
    assert recorded["order"]["seed"] == 5 and recorded["n_pool"] == 120 and recorded["n_used"] == 60
    assert r.evidence.n_queries == 120  # the evidence keeps every query; the rule tests a prefix


def test_the_queries_are_tested_in_a_seeded_random_order_unless_the_given_order_is_asked_for(
        selected, data):
    corpus, queries, qrels = data
    order = dict(selected.decision.query_order)
    assert (order["kind"], order["seed"]) == ("seeded_permutation", 0)  # the default: seed 0
    assert order["indices_sha256"] == array_digest(
        np.random.default_rng(0).permutation(len(queries)).astype(np.int64))
    assert order["indices_sha256"] != array_digest(np.arange(len(queries), dtype=np.int64))
    assert selected.report()["query_order"] == {"kind": "seeded_permutation", "seed": 0,
                                                "budget": None}
    given = run(corpus, queries, qrels, order_seed=None)  # None: the order the file has
    assert dict(given.decision.query_order)["kind"] == "given"
    assert given.report()["query_order"] == {"kind": "given", "seed": None, "budget": None}
    assert given.decision.p_values != selected.decision.p_values  # the order matters to the test


def test_what_the_order_requires_is_recorded_and_said_where_the_caller_reads(data, selected):
    corpus, queries, qrels = data
    assert selected.report()["caveats"] == []  # the default seed asks nothing of the caller
    given = run(corpus, queries, qrels, order_seed=None)
    assert any("order supplied" in c and "unrelated to the outcomes" in c
               for c in given.report()["caveats"])
    chosen = run(corpus, queries, qrels, order_seed=7)
    assert any("order seed (7) was chosen by the caller" in c and "outside the statement" in c
               for c in chosen.report()["caveats"])
    doc = " ".join((certify.__doc__ or "").split())
    assert "The default shuffle exists so that a sorted input file cannot break assumption A5" in doc
    assert "valid only if that order is unrelated to the outcomes" in doc
    assert "never an order sorted by difficulty, date, source or any other query attribute" in doc
    assert "trying several seeds and reporting one is outside the statement" in doc
    assert inspect.signature(certify).parameters["order_seed"].default == DEFAULT_ORDER_SEED == 0


def test_a_budget_larger_than_the_queries_given_is_refused_before_any_work(data, monkeypatch):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(QueryOrderError, match="exceeds"):
        run(corpus, queries, qrels, query_budget=121)


def test_a_budget_larger_than_the_labelled_queries_is_refused_after_the_evidence(data):
    corpus, queries, qrels = data
    labelled = {k: v for k, v in qrels.items() if k not in set(queries.ids[:30])}
    with pytest.raises(QueryOrderError, match="exceeds"):
        run(corpus, queries, labelled, query_budget=100)


def test_the_progress_callback_sees_every_candidate_once(data):
    corpus, queries, qrels = data
    seen = []
    run(corpus, queries, qrels, progress=lambda position, total, family: seen.append((position, total, family)))
    assert [s[0] for s in seen] == [1, 2, 3, 4, 5, 6] and {s[1] for s in seen} == {6}
    assert [s[2] for s in seen] == ["exact", "float16", "int8", "merge+binary", "merge+binary", "merge+binary"]


def test_the_rule_can_be_chosen(data):
    corpus, queries, qrels = data
    r = run(corpus, queries, qrels, rule=BonferroniBetting())
    assert r.certificate.rule_name == "bonferroni_betting"
    assert r.report()["rule"] == {
        "name": "bonferroni_betting", "class": "optivision.core.selection.rules.BonferroniBetting",
        "builtin": True, "certifying": True, "declared_certifying": True}
    assert r.decision.level == pytest.approx(0.05 / 6)
    assert verify(r.certificate, evidence=r.evidence).ok


# ---------------------------------------------------------------------- determinism


def test_the_same_inputs_and_creation_time_give_the_same_certificate(data, selected):
    corpus, queries, qrels = data
    again = run(corpus, queries, qrels)
    assert again.certificate.digest == selected.certificate.digest
    assert again.certificate == selected.certificate
    assert again.certificate.to_json() == selected.certificate.to_json()
    assert again.decision.to_dict() == selected.decision.to_dict()


def test_another_creation_time_changes_only_the_record_of_the_time(data, selected):
    corpus, queries, qrels = data
    later = run(corpus, queries, qrels, created="2027-01-01T00:00:00Z")
    assert later.certificate.digest != selected.certificate.digest
    assert later.decision.to_dict() == selected.decision.to_dict()
    assert later.certificate.created_utc == "2027-01-01T00:00:00Z"


def test_a_list_of_arrays_and_a_corpus_are_both_accepted_and_agree(data):
    corpus, queries, qrels = data
    doc_arrays = [corpus[i].vectors for i in range(len(corpus))]
    query_arrays = [queries[i].vectors for i in range(len(queries))]
    position = {doc_id: str(i) for i, doc_id in enumerate(corpus.ids)}
    labels = {str(j): {position[d] for d in qrels[qid]} for j, qid in enumerate(queries.ids)}
    from_lists = run(doc_arrays, query_arrays, labels)
    from_corpora = run(MultiVectorCorpus.from_arrays(doc_arrays), MultiVectorCorpus.from_arrays(query_arrays),
                       labels)
    assert from_lists.outcome is Outcome.SELECT
    assert from_lists.certificate.digest == from_corpora.certificate.digest


def test_a_float16_corpus_works_and_is_recorded_as_float16(data):
    corpus = make_corpus(40, 16, tokens=(8, 12), seed=3, dtype=np.float16)
    queries, qrels = make_queries(corpus, 120, noise=1.0, seed=4)
    r = run(corpus, queries, qrels)
    assert r.outcome is Outcome.SELECT and r.certificate.to_dict()["corpus"]["dtype"] == "float16"
    assert r.compressed.stats["native_dtype_before"] == "float16"
    assert r.corpus.vectors.dtype == np.float16  # not converted
    assert r.compression_ratio == pytest.approx(r.compressed.report()["compression_vs_float32"])
    assert r.retriever().search(queries, 3).n_queries == len(queries)
    assert verify(r.certificate, evidence=r.evidence, corpus=corpus, queries=queries).ok


def test_inputs_are_not_modified(data):
    corpus, queries, qrels = data
    before = (corpus_identity(corpus).digest(), corpus_identity(queries).digest(), copy.deepcopy(qrels))
    run(corpus, queries, qrels)
    run(corpus, queries.select(range(8)), qrels, target=0.999)
    assert (corpus_identity(corpus).digest(), corpus_identity(queries).digest(), qrels) == before


# ------------------------------------------------------------------ save and verify


def test_save_load_and_verify_at_all_three_levels(selected, data, tmp_path):
    corpus, queries, qrels = data
    path = selected.save(tmp_path / "artifact")
    assert sorted(p.name for p in path.iterdir()) == [
        "MANIFEST.json", "certificate.json", "codes.npz", "evidence.npz"]

    stored = load_artifact(path)  # verify=True raises on any failed check
    assert stored.certificate == selected.certificate and stored.verification.ok
    assert stored.evidence == selected.evidence
    assert stored.pipeline.to_dict() == selected.pipeline.to_dict()
    assert np.array_equal(stored.compressed.codes, selected.compressed.codes)

    integrity = verify(selected.certificate)  # level 1: the certificate alone
    assert integrity.ok and integrity.level == "integrity" and integrity.levels == ("integrity",)
    replay = verify(selected.certificate, evidence=selected.evidence)  # level 2
    assert replay.ok and replay.level == "replay" and replay.status("replay.decision") == "pass"
    binding = verify(path, corpus=corpus, queries=queries, reference=LabelReference(qrels))  # level 3
    assert binding.ok and binding.level == "binding"
    for name in ("artifact files", "binding.corpus", "binding.queries", "binding.reference",
                 "binding.pipeline", "binding.codes", "replay.decision"):
        assert binding.status(name) == "pass", name


def test_a_loaded_artifact_retrieves_exactly_as_the_original_does(selected, data, tmp_path):
    _, queries, _ = data
    stored = load_artifact(selected.save(tmp_path / "artifact"))
    again = CompressedRetriever(stored.compressed, stored.pipeline).search(queries, 5)
    assert again == selected.retriever().search(queries, 5)


def test_an_abstention_is_saved_and_verified_like_a_selection(abstained, data, tmp_path):
    path = abstained.save(tmp_path / "abstained")
    assert sorted(p.name for p in path.iterdir()) == ["MANIFEST.json", "certificate.json", "evidence.npz"]
    report = verify(path)
    assert report.ok
    assert load_artifact(path).compressed is None and load_artifact(path).pipeline is None


def test_saving_twice_needs_overwrite_and_evidence_is_optional(selected, tmp_path):
    path = selected.save(tmp_path / "artifact")
    with pytest.raises(FileExistsError, match="overwrite"):
        selected.save(path)
    (path / "notes.txt").write_text("keep me", encoding="utf-8")
    selected.save(path, overwrite=True, include_evidence=False)
    assert not (path / "evidence.npz").exists() and (path / "notes.txt").read_text(encoding="utf-8") == "keep me"
    report = verify(path)
    assert report.ok and "replay" not in report.levels  # nothing to replay from


def test_a_modified_artifact_is_reported_not_trusted(selected, tmp_path):
    path = selected.save(tmp_path / "artifact")
    cert = path / "certificate.json"
    cert.write_text(cert.read_text(encoding="utf-8").replace('"target": 0.9', '"target": 0.5'),
                    encoding="utf-8")
    with pytest.raises(ArtifactError, match="manifest"):
        load_artifact(path)
    report = verify(path)
    assert not report.ok
    assert report.status("artifact files") == "fail" and report.status("digest") == "fail"


def test_a_certificate_file_can_be_verified_on_its_own(selected, tmp_path):
    saved = selected.certificate.save(tmp_path / "certificate.json")
    report = verify(saved)
    assert report.ok and report.level == "integrity"


def test_a_signed_certificate_checks_with_its_key_only(selected):
    key = bytes(range(32))
    signed = sign_certificate(selected.certificate, key, key_id="test")
    assert verify(signed, key=key).status("signature") == "pass"
    assert verify(signed).status("signature") == "skipped"
    assert not verify(signed, key=bytes(32)).ok
    assert verify_certificate(signed, key=key).ok


def test_verify_checks_the_certificate_against_the_live_objects_it_is_given(selected, data):
    corpus, queries, _ = data
    other = make_corpus(40, 16, tokens=(8, 12), seed=99)
    report = verify(selected.certificate, corpus=other)
    assert not report.ok and report.status("binding.corpus") == "fail"
    assert verify(selected.certificate, corpus=corpus, queries=queries).ok
    wrong_labels = verify(selected.certificate, reference=LabelReference({"q00000": {"d000001"}}))
    assert wrong_labels.status("binding.reference") == "fail"


# ----------------------------------------------------------------------- the index


def test_the_retriever_of_a_selection_returns_ranked_ids(selected, data):
    corpus, queries, _ = data
    retriever = selected.retriever()
    assert isinstance(retriever, CompressedRetriever)
    ranking = retriever.search(queries, 5)
    assert isinstance(ranking, Ranking) and ranking.n_queries == len(queries) and ranking.width == 5
    assert {doc for row in ranking.ids for doc in row} <= set(corpus.ids)
    assert all(len(row) == 5 for row in ranking.ids)
    small_blocks = selected.retriever(max_block_bytes=4096).search(queries, 5)
    assert isinstance(small_blocks, Ranking) and small_blocks.width == 5  # equal to float32 rounding only


def test_the_retriever_of_an_abstention_is_the_exact_one_over_the_original_corpus(abstained, data):
    corpus, queries, _ = data
    retriever = abstained.retriever()
    assert type(retriever) is ExactRetriever and retriever.store is corpus
    assert retriever.search(queries, 5) == ExactRetriever(corpus).search(queries, 5)


def test_a_float32_only_selection_ranks_exactly_as_the_uncompressed_index_does(data):
    corpus, queries, qrels = data
    r = run(corpus, queries, qrels, target=0.5, space={"exact": [Pipeline([Float32Quantizer()])]})
    assert r.outcome is Outcome.SELECT and r.compression_ratio == pytest.approx(1.0)
    assert r.retriever().search(queries, 5) == ExactRetriever(corpus).search(queries, 5)


# ---------------------------------------------------------------- what the result is


def test_the_report_is_json_native_and_holds_no_retention_figure(selected, abstained):
    expected_keys = {
        "outcome", "certified", "construct", "metric", "target", "error", "level", "rule", "n_queries",
        "n_queries_given", "n_queries_without_relevant_document", "n_exact_duplicate_queries",
        "query_order", "space", "selected", "compression_ratio", "basis_sensitive",
        "basis_invariance", "statement", "assumptions", "caveats", "not_certified_note",
        "certificate_digest"}
    for result in (selected, abstained):
        rep = result.report()
        assert set(rep) == expected_keys
        assert json.loads(json.dumps(rep, allow_nan=False)) == rep
        assert not [k for k in rep if "retention" in k or "ndcg" in k]
        assert rep["certificate_digest"] == result.certificate.digest
    rep = selected.report()
    assert rep["outcome"] == "select" and rep["certified"] is True and rep["not_certified_note"] is None
    assert rep["metric"] == "ndcg@5" and rep["target"] == 0.9 and rep["error"] == 0.05
    assert rep["selected"]["label"] == selected.pipeline.label()
    assert rep["selected"]["pipeline"] == selected.pipeline.to_dict()
    assert isinstance(rep["basis_sensitive"], bool)
    assert rep["basis_sensitive"] == selected.certificate.basis_sensitive
    assert rep["statement"] == selected.certificate.statement and len(rep["assumptions"]) == 10
    assert rep["n_exact_duplicate_queries"] == 0
    assert rep["compression_ratio"] == selected.compression_ratio


def test_the_statement_comes_with_its_assumptions_in_every_report(selected, abstained):
    for result in (selected, abstained):
        rep = result.report()
        assert "assumptions A1-A6 and conditions C1-C4" in rep["statement"]
        assert "not a statement about power" in rep["statement"]
        assert "NOT the probability, given that a candidate was selected" in rep["statement"]
        assert [a[:2] for a in rep["assumptions"]] == [
            "A1", "A2", "A3", "A4", "A5", "A6", "C1", "C2", "C3", "C4"]


# ------------------------------------------------- repeated queries and rules of other people


def doubled_queries(queries, qrels):
    arrays = [queries.vectors[queries.offsets[i]:queries.offsets[i + 1]] for i in range(len(queries))]
    ids = [*queries.ids, *[f"{q}_copy" for q in queries.ids]]
    labels = {**qrels, **{f"{q}_copy": docs for q, docs in qrels.items()}}
    return MultiVectorCorpus.from_arrays(arrays + arrays, ids=ids), labels


def test_every_query_twice_is_refused_by_default(data):
    corpus, queries, qrels = data
    doubled, labels = doubled_queries(queries, qrels)
    with pytest.raises(ValueError) as exc:
        run(corpus, doubled, labels, target=0.5)
    message = str(exc.value)
    assert message.startswith("120 of the 240 queries tested repeat an earlier query exactly")
    assert "assumption A1" in message and "allow_duplicate_queries=True" in message


def test_every_query_twice_with_the_flag_is_counted_recorded_asserted_and_warned_about_once(data):
    corpus, queries, qrels = data
    doubled, labels = doubled_queries(queries, qrels)
    with pytest.warns(UserWarning) as caught:
        r = run(corpus, doubled, labels, target=0.5, allow_duplicate_queries=True)
    repeats = [w for w in caught if "repeat an earlier query" in str(w.message)]
    assert len(repeats) == 1 and issubclass(repeats[0].category, UserWarning)
    message = str(repeats[0].message)
    assert message.startswith("120 of the 240 queries tested repeat an earlier query exactly")
    assert "the caller asserted" in message and "does not apply" in message
    assert repeats[0].filename == __file__  # attributed to the caller of certify()
    record = r.certificate.to_dict()
    assert record["queries"]["exact_duplicates"] == 120
    assert record["queries"]["duplicates_asserted_independent"] is True
    assert r.report()["n_exact_duplicate_queries"] == 120
    sentence = next(c for c in r.report()["caveats"] if "repeat an earlier query" in c)
    assert sentence.startswith("120 of the 240 queries tested repeat an earlier query exactly")
    assert "The caller asserted that they are independent draws" in sentence
    assert "If they are copies" in sentence
    assert verify(r.certificate, evidence=r.evidence, corpus=corpus, queries=doubled).ok


def test_a_second_look_at_the_same_queries_warns_and_changes_nothing(data):
    corpus, queries, qrels = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = run(corpus, queries, qrels, target=0.9, created=CREATED)
        again = run(corpus, queries, qrels, target=0.9, created=CREATED)  # the same look
    assert again.certificate.digest == first.certificate.digest
    for change in ({"target": 0.8}, {"error": 0.1}, {"order_seed": 5}):
        with pytest.warns(UserWarning, match="spends error level") as caught:
            other = run(corpus, queries, qrels, **{"target": 0.9, "created": CREATED, **change})
        message = str(caught[0].message)
        assert "STATISTICAL_REVIEW.md" in message and "7.7" in message
        assert "Fresh queries are needed for each target" in message
        assert caught[0].filename == __file__
        assert other.decision.outcome is not None
    # other queries are a first look
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(corpus, queries.select(range(60)), qrels, target=0.8)


def test_distinct_queries_give_zero_no_caveat_and_no_warning(data):
    corpus, queries, qrels = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = run(corpus, queries, qrels)
    assert r.report()["n_exact_duplicate_queries"] == 0
    assert not any("repeat an earlier query" in c for c in r.report()["caveats"])


def test_a_graded_qrels_mapping_with_a_zero_grade_is_refused(data):
    """LabelReference refuses a grade of 0 or below (RI-14); the older warning is not reached."""
    corpus, queries, qrels = data
    few = queries.select(range(8))
    positive = {q: {d: 1 for d in docs} for q, docs in qrels.items()}  # read correctly: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(corpus, few, positive, target=0.999)
    first = next(iter(qrels))
    zero = {**positive, first: {**positive[first], "d000039": 0}}
    with pytest.raises(ValueError, match="Drop the entries that are not relevant"):
        run(corpus, few, zero, target=0.999)


def test_a_rule_of_someone_elses_is_not_certified_through_certify_either(data):
    corpus, queries, qrels = data

    class AlwaysSmallest:
        name = "always_smallest"
        certifying = True

        def config(self):
            return {"name": self.name}

        def assumptions(self):
            return ("none: trust me",)

        def decide(self, candidates, baseline, families, bytes_per_doc, vectors_per_doc, target,
                   error):
            from optivision.core.selection import RuleResult

            k = int(np.argmin(bytes_per_doc))
            return RuleResult(k, (k,), tuple(1.0 for _ in bytes_per_doc), error)

    r = run(corpus, queries, qrels, rule=AlwaysSmallest(), target=0.999)
    assert r.outcome is Outcome.SELECT and r.certified is False
    assert r.decision.certified is False and r.certificate.certified is False
    rep = r.report()
    assert rep["certified"] is False and rep["rule"]["builtin"] is False
    assert rep["rule"]["certifying"] is False and rep["rule"]["declared_certifying"] is True
    assert "not part of this library" in rep["statement"] and "STATEMENT" not in rep["statement"]
    assert rep["assumptions"] == [("Declared by the rule's author, not examined by this library: "
                                   "none: trust me")]
    assert "NOT certified" in repr(r) and "certified=False" in repr(r)
    report = verify(r.certificate, evidence=r.evidence, corpus=corpus, queries=queries,
                    reference=LabelReference(qrels), pipeline=r.pipeline, compressed=r.compressed)
    assert report.ok and report.status("replay.decision") == "skipped"
    assert report.notes and "nothing in this record is certified" in report.notes[0]


def test_the_repr_names_the_outcome_the_reference_and_the_settings(selected, abstained):
    text = repr(selected)
    assert text.startswith("CertifiedResult(outcome='select', certified=True")
    assert selected.pipeline.label() in text and "reference=labels" in text
    assert "target=0.9" in text and "error=0.05" in text and "n_queries=120" in text
    assert "retention" not in text
    assert "outcome='abstain', certified=False" in repr(abstained)


def test_the_result_is_read_only(selected):
    with pytest.raises(dataclasses.FrozenInstanceError):
        selected.decision = None
    with pytest.raises(AttributeError):
        selected.outcome = Outcome.ABSTAIN
    with pytest.raises(AttributeError):
        selected.certified = False


def test_a_result_that_contradicts_its_decision_is_refused(selected, abstained):
    with pytest.raises(ValueError, match="ABSTAIN result holds no pipeline"):
        dataclasses.replace(abstained, pipeline=selected.pipeline, compressed=selected.compressed)
    with pytest.raises(ValueError, match="needs the fitted pipeline"):
        dataclasses.replace(selected, pipeline=None, compressed=None)
    with pytest.raises(TypeError):
        dataclasses.replace(selected, decision=None)


# ------------------------------------------------------------------ model-blindness


def _encoder_dict(**changes):
    return {"model_id": "org/model", "dimension": 16, "modality": "text", **changes}


def test_the_encoder_is_recorded_and_changes_no_decision(data, selected):
    corpus, queries, qrels = data
    plain = selected.certificate.to_dict()
    assert plain["encoder"] is None
    for encoder in (_encoder_dict(), EncoderIdentity(model_id="other/model", dimension=16, modality="image"),
                    _encoder_dict(model_id="x" * 40, revision="r1", normalized=True)):
        r = run(corpus, queries, qrels, encoder=encoder)
        expected = encoder.to_dict() if isinstance(encoder, EncoderIdentity) \
            else EncoderIdentity.from_dict(encoder).to_dict()
        assert r.certificate.to_dict()["encoder"] == expected
        assert r.decision.to_dict() == selected.decision.to_dict()
        assert r.decision.p_values == selected.decision.p_values
        assert r.certificate.digest != selected.certificate.digest  # the record differs, nothing else


def test_the_identity_the_vectors_carry_is_recorded_when_no_encoder_is_given(data, selected):
    corpus, queries, qrels = data
    identity = EncoderIdentity(model_id="org/model", revision="abc123", dimension=16, modality="text",
                               normalized=True, inference_dtype="float32", extra={"pooling": "none"})
    carrying = corpus_from_arrays([corpus[i].vectors for i in range(len(corpus))], ids=corpus.ids,
                                  identity=identity)
    assert carrying.attrs["encoder_identity"] == identity.to_dict()
    r = run(carrying, queries, qrels)
    assert r.certificate.to_dict()["encoder"] == identity.to_dict()
    assert r.decision.to_dict() == selected.decision.to_dict()  # descriptive metadata: nothing depends on it
    assert verify(r.certificate, evidence=r.evidence, corpus=carrying, queries=queries).ok


def test_an_explicit_encoder_wins_over_the_identity_the_vectors_carry(data):
    corpus, queries, qrels = data
    carried = EncoderIdentity(model_id="org/carried", dimension=16, modality="text")
    carrying = corpus_from_arrays([corpus[i].vectors for i in range(len(corpus))], ids=corpus.ids,
                                  identity=carried)
    explicit = EncoderIdentity(model_id="org/explicit", dimension=16, modality="image")
    r = run(carrying, queries, qrels, encoder=explicit)
    assert r.certificate.to_dict()["encoder"] == explicit.to_dict()
    as_dict = run(carrying, queries, qrels, encoder={"model_id": "org/dict", "dimension": 16, "modality": "other"})
    assert as_dict.certificate.to_dict()["encoder"]["model_id"] == "org/dict"


def test_an_unusable_identity_in_the_vectors_is_reported_with_its_origin(data):
    corpus, queries, qrels = data
    broken = MultiVectorCorpus(corpus.vectors, corpus.offsets, ids=corpus.ids,
                               attrs={"encoder_identity": {"bogus": 1}})
    with pytest.raises(ContractError, match=r"attrs\['encoder_identity'\].*pass encoder="):
        run(broken, queries, qrels)
    with pytest.raises(ContractError, match="encoder is not a valid encoder identity"):
        run(corpus, queries, qrels, encoder={"bogus": 1})
    with pytest.raises(TypeError, match="encoder"):
        run(corpus, queries, qrels, encoder="org/model")


def test_a_declared_dimension_that_does_not_match_the_vectors_is_refused(data):
    corpus, queries, qrels = data
    with pytest.raises(ValueError, match="dimension"):
        run(corpus, queries, qrels, encoder=_encoder_dict(dimension=99))


# ------------------------------------------------------------------------- errors


def _must_not_run(*args, **kwargs):
    raise AssertionError("the evidence was collected although the arguments were invalid")


def test_queries_are_required(data, monkeypatch):
    corpus, _, _ = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(ValueError, match="needs sample queries"):
        certify(corpus, None)
    with pytest.raises(TypeError, match="queries"):
        certify(corpus)  # type: ignore[call-arg]


@pytest.mark.parametrize("empty", [[], ()])
def test_an_empty_list_of_queries_or_documents_is_a_clear_error(data, empty):
    corpus, queries, _ = data
    with pytest.raises(ContractError, match="queries: nothing was given"):
        certify(corpus, empty)
    with pytest.raises(ContractError, match="vectors: nothing was given"):
        certify(empty, queries)


def test_an_empty_query_corpus_is_a_clear_error(data):
    corpus, _, _ = data
    none = MultiVectorCorpus(np.zeros((0, 16), dtype=np.float32), [0])
    with pytest.raises(ContractError, match="no queries"):
        certify(corpus, none, space=small_space())


def test_queries_of_another_dimension_are_refused_naming_both(data, monkeypatch):
    corpus, _, _ = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    other = MultiVectorCorpus.from_arrays([np.ones((3, 8), dtype=np.float32)])
    with pytest.raises(ContractError, match=r"queries have dimension 8 but the vectors have dimension 16"):
        certify(corpus, other, space=small_space())


def test_invalid_vectors_are_refused_by_the_contract(data):
    corpus, queries, _ = data
    bad = corpus.vectors.copy()
    bad[5, 3] = np.nan
    broken = MultiVectorCorpus(bad, corpus.offsets, ids=corpus.ids)
    with pytest.raises(ContractError, match="vectors: .*non-finite"):
        certify(broken, queries, space=small_space())
    duplicated = MultiVectorCorpus(corpus.vectors, corpus.offsets, ids=["d"] * len(corpus))
    with pytest.raises(ContractError, match="used by documents"):
        certify(duplicated, queries, space=small_space())


@pytest.mark.parametrize("target", [0, 0.0, -0.1, 1.0001, 2, float("nan"), float("inf"), True, "0.9", None])
def test_a_target_outside_zero_one_is_refused_before_any_work(data, monkeypatch, target):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(TargetError):
        certify(corpus, queries, qrels, target=target, space=small_space())


@pytest.mark.parametrize("error", [0, 0.0, -0.05, 1, 1.0, 1.5, float("nan"), True, "0.05", None])
def test_an_error_level_outside_zero_one_is_refused_before_any_work(data, monkeypatch, error):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(ErrorLevelError):
        certify(corpus, queries, qrels, error=error, space=small_space())


def test_the_boundaries_are_valid_and_one_target_is_one_call(data):
    corpus, queries, qrels = data
    assert run(corpus, queries.select(range(8)), qrels, target=1.0).outcome is Outcome.ABSTAIN
    assert run(corpus, queries, qrels, error=0.999).outcome in (Outcome.SELECT, Outcome.ABSTAIN)
    with pytest.raises(TargetError, match="exactly one per call"):
        run(corpus, queries, qrels, target=[0.9, 0.95])  # several targets: outside the statement
    with pytest.raises(ErrorLevelError, match="exactly one per call"):
        run(corpus, queries, qrels, error=(0.05, 0.01))


@pytest.mark.parametrize(("name", "value"), [
    ("query_budget", 0), ("query_budget", -3), ("query_budget", True), ("query_budget", 2.5),
    ("order_seed", -1), ("order_seed", 1.5), ("order_seed", False), ("order_seed", "7"),
])
def test_a_bad_budget_or_seed_is_refused_before_any_work(data, monkeypatch, name, value):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(QueryOrderError, match=name):
        run(corpus, queries, qrels, **{name: value})


def test_numpy_integers_are_fine_as_a_budget_and_a_seed(data):
    corpus, queries, qrels = data
    r = run(corpus, queries, qrels, query_budget=np.int64(60), order_seed=np.int32(5))
    assert dict(r.decision.query_order)["budget"] == 60 and dict(r.decision.query_order)["seed"] == 5


@pytest.mark.parametrize(("created", "error", "message"), [
    ("2026-10-06", ValueError, "created must look like"),
    (datetime.datetime(2026, 10, 6, 12, 0, 0), ValueError, "naive datetime"),  # noqa: DTZ001
    (20261006, TypeError, "created must be"),
])
def test_a_bad_creation_time_is_refused_when_the_certificate_is_made(data, created, error, message):
    corpus, queries, qrels = data
    with pytest.raises(error, match=message):
        run(corpus, queries.select(range(8)), qrels, created=created)


def test_an_aware_datetime_is_recorded_as_utc(data):
    corpus, queries, qrels = data
    zone = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    r = run(corpus, queries, qrels, created=datetime.datetime(2026, 10, 6, 5, 30, 0, tzinfo=zone))
    assert r.certificate.created_utc == "2026-10-06T00:00:00Z"


@pytest.mark.parametrize("rule", ["fixed_sequence_betting", 3, object()])
def test_a_rule_that_is_not_a_rule_is_refused_before_any_work(data, monkeypatch, rule):
    corpus, queries, qrels = data
    monkeypatch.setattr(api_module, "collect_evidence", _must_not_run)
    with pytest.raises(TypeError, match="rule must provide"):
        run(corpus, queries, qrels, rule=rule)


def test_a_metric_that_does_not_exist_is_refused(data):
    corpus, queries, qrels = data
    with pytest.raises(ValueError, match="metric"):
        run(corpus, queries, qrels, metric="map@10")


def test_certify_is_importable_where_the_design_says(selected):
    assert optivision.certify is certify and certify.__module__ == "optivision.api"
    assert isinstance(selected, optivision.CertifiedResult)
    assert isinstance(selected.decision, optivision.Decision)
    assert isinstance(selected.certificate, optivision.Certificate)


def test_verify_says_when_ok_does_not_mean_a_certified_index(selected, abstained, data):
    # ok means the record is intact; for an ABSTAIN or FALLBACK it must not read as "certified".
    corpus, queries, qrels = data
    fallback = run(corpus, queries.select(range(8)), qrels, target=0.999,
                   fallback=Pipeline([Float16Quantizer()]))
    for result, word in ((abstained, "ABSTAIN"), (fallback, "FALLBACK")):
        report = verify(result.certificate, evidence=result.evidence)
        assert report.ok
        assert report.notes and word in report.notes[0] and "not that" in report.notes[0]
        assert word in report.summary() and word in report.to_dict()["notes"][0]
    assert verify(selected.certificate, evidence=selected.evidence).notes == ()


@pytest.mark.parametrize("bad", [[("q00000", ["d000001"])], ["d000001"], 3])
def test_qrels_of_the_wrong_type_are_refused_by_name_not_with_an_attribute_error(data, bad):
    corpus, queries, _ = data
    with pytest.raises(TypeError, match="qrels must be a mapping"):
        certify(corpus, queries, bad, target=0.9, space=CandidateSpace.from_families(
            {"f": [Pipeline([Float16Quantizer()])]}))
