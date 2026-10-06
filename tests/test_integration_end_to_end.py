"""End to end: vectors -> ``certify`` -> ``save`` -> ``load_artifact`` -> ``verify_certificate``
-> ``retriever().search``, on seeded synthetic corpora.

The question every scenario answers is the one that matters for a certificate: *is the index
that was certified the index that is shipped?* The per-query metric values the evidence
records for the selected candidate are recomputed from the artifact as it is loaded from disk
(restored fitted state, stored codes, queries projected by the restored pipeline) and must
be equal to the recorded ones exactly; so must the metric computed from the ranking the
deployed retriever returns.

Scenarios: a text-like corpus (d = 96, 20 to 60 vectors per document, no aligned arrays), a
vision-like one (d = 128, about 100 vectors per document, ``protected`` rows and ``positions``),
float16 input, the agreement construct (no labels), an abstention and a fallback. Synthetic data
exercise code paths; none of the numbers says anything about an encoder, and no registered
experiment was re-run.
"""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from optivision.api import CertifiedResult, certify, verify
from optivision.compose import Pipeline
from optivision.core.artifact import (
    ArtifactError,
    Certificate,
    load_artifact,
    sign_certificate,
    verify_certificate,
)
from optivision.core.evaluate import (
    LabelReference,
    RankingReference,
    metrics_from_ranking,
    per_query_metrics,
)
from optivision.core.retrieve import CompressedRetriever, ExactRetriever
from optivision.core.selection import BonferroniBetting, CandidateSpace, Outcome
from optivision.core.similarity import maxsim_matrix
from optivision.core.stages import pipeline_fitted_identity
from optivision.core.vectors import EncoderIdentity, corpus_identity
from optivision.representation import MultiVectorCorpus
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    PCAProjector,
)
from optivision.testing import make_corpus, make_queries

from .test_integration_layers import bitwise_equal, build_corpus, small_space

CREATED = "2026-01-01T00:00:00Z"
METRIC = "ndcg@5"


def space(wide: bool = True) -> CandidateSpace:
    """Four chains, least to most aggressive, ordered before any query is looked at."""
    return CandidateSpace.from_families({
        "float16": [Pipeline([Float16Quantizer()])],
        "int8": [Pipeline([Int8Quantizer("per_vector")])],
        "int4+merge": [Pipeline([Int4Quantizer(center="mean")]),
                       Pipeline([AdaptiveMerge(radius=0.9, center="mean"),
                                 Int4Quantizer(center="mean")]),
                       Pipeline([AdaptiveMerge(radius=0.7, center="mean"),
                                 Int4Quantizer(center="mean")])],
        "binary": [Pipeline([BinaryQuantizer()]), Pipeline([PCAProjector(48), BinaryQuantizer()]),
                   Pipeline([PCAProjector(16), BinaryQuantizer()])],
    }, name="end-to-end")


def merge_space() -> CandidateSpace:
    return CandidateSpace.from_families({
        "merge+int8": [Pipeline([AdaptiveMerge(radius=0.5, center="mean"),
                                 Int8Quantizer("per_vector")]),
                       Pipeline([AdaptiveMerge(radius=0.3, center="mean"),
                                 Int8Quantizer("per_vector")])],
    }, name="merge-only")


def text_corpus(dtype: object = np.float32, dim: int = 96, n_docs: int = 60) -> MultiVectorCorpus:
    """A text-like corpus: 20 to 60 vectors per document, no aligned arrays."""
    return make_corpus(n_docs, dim, tokens=(20, 60), seed=11, anisotropy=0.5, dtype=dtype)


def vision_corpus(n_docs: int = 24, dim: int = 128, side: int = 10, n_protected: int = 4):
    """A vision-like corpus: about 100 vectors per document, the first ``n_protected`` rows of
    each document protected (instruction tokens) and the others on a ``side x side`` grid
    (``positions``; NaN for protected rows and for rows beyond the grid)."""
    base = make_corpus(n_docs, dim, tokens=(side * side, side * side + 8), seed=21,
                       anisotropy=0.5)
    n = base.num_vectors
    protected = np.zeros(n, dtype=bool)
    positions = np.full((n, 2), np.nan, dtype=np.float32)
    for i in range(len(base)):
        lo, hi = base.span(i)
        protected[lo : lo + n_protected] = True
        for j in range(n_protected, hi - lo):
            if j - n_protected < side * side:
                positions[lo + j] = [((j - n_protected) // side + 0.5) / side,
                                     ((j - n_protected) % side + 0.5) / side]
    return MultiVectorCorpus(base.vectors, base.offsets, ids=base.ids, protected=protected,
                             positions=positions, attrs={"synthetic": True})


@dataclass
class Scenario:
    name: str
    corpus: MultiVectorCorpus
    queries: MultiVectorCorpus
    qrels: dict | None
    reference: Any  # the reference object the certificate was issued under
    result: CertifiedResult
    root: Path  # where the artifact was saved
    options: dict


def run_scenario(name: str, tmp_root: Path, corpus: MultiVectorCorpus, *, n_queries: int = 200,
                 noise: float = 1.4, labels: bool = True, the_space: CandidateSpace | None = None,
                 unlabelled: bool = False, **options: Any) -> Scenario:
    queries, qrels = make_queries(corpus, n_queries, tokens=(4, 8), noise=noise, seed=12)
    if unlabelled:  # every fifth query has no label at all, every fifth+1 only an unknown document
        qrels = {q: ({"no-such-document"} if i % 5 == 1 else docs)
                 for i, (q, docs) in enumerate(qrels.items()) if i % 5 != 0}
    reference = LabelReference(qrels) if labels else RankingReference()
    settings = {"target": 0.9, "error": 0.1, "created": CREATED, **options}
    result = certify(corpus, queries, qrels if labels else None,
                     space=space() if the_space is None else the_space, metric=METRIC, **settings)
    root = result.save(tmp_root / name)
    return Scenario(name, corpus, queries, qrels if labels else None, reference, result, root,
                    settings)


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory) -> dict[str, Scenario]:
    root = tmp_path_factory.mktemp("end-to-end")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        made = {
            "text": run_scenario("text", root, text_corpus()),
            "vision": run_scenario("vision", root, vision_corpus(), n_queries=150),
            "float16": run_scenario("float16", root, text_corpus(np.float16, dim=64, n_docs=40)),
            "agreement": run_scenario("agreement", root, text_corpus(), labels=False),
            "unlabelled": run_scenario("unlabelled", root, text_corpus(), unlabelled=True),
            "edge": run_scenario("edge", root, build_corpus(7, 21), n_queries=120, noise=1.0,
                                 the_space=small_space(), target=0.7, error=0.2),
            "bonferroni": run_scenario("bonferroni", root, text_corpus(), rule=BonferroniBetting()),
            "seeded": run_scenario("seeded", root, text_corpus(), order_seed=5, query_budget=120),
            "merge": run_scenario("merge", root, vision_corpus(), n_queries=150, noise=1.0,
                                  the_space=merge_space(), target=0.5, error=0.2),
            "abstain": run_scenario("abstain", root, text_corpus(), query_budget=10, target=0.99,
                                    error=0.01),
            "fallback": run_scenario("fallback", root, text_corpus(), query_budget=10,
                                     target=0.99, error=0.01,
                                     fallback=Pipeline([Float16Quantizer()])),
        }
    return made


SELECTED = ["text", "vision", "float16", "agreement", "unlabelled", "edge", "bonferroni",
            "seeded", "merge"]
ALL = [*SELECTED, "abstain", "fallback"]


def baseline_scores(sc: Scenario) -> np.ndarray:
    return maxsim_matrix(sc.queries, sc.corpus)


def relevant_sets(sc: Scenario) -> list[np.ndarray]:
    return sc.reference.relevant(sc.corpus.ids, sc.queries.ids, baseline_scores(sc))


def kept_queries(sc: Scenario) -> list[int]:
    return [i for i, rel in enumerate(relevant_sets(sc)) if rel.size]


# =========================================================== the certified thing is the shipped thing


class TestTheCertifiedIndexIsTheShippedIndex:
    @pytest.mark.parametrize("name", SELECTED)
    def test_the_loaded_artifact_reproduces_the_recorded_per_query_metrics_exactly(
            self, scenarios, name):
        sc = scenarios[name]
        assert sc.result.outcome is Outcome.SELECT, sc.result
        art = load_artifact(sc.root)  # verify=True: manifest, restored state, all levels with evidence
        position = art.certificate.selected_position
        assert position == sc.result.selected.position
        evidence = art.evidence
        keep = kept_queries(sc)
        relevant = relevant_sets(sc)

        base = per_query_metrics(baseline_scores(sc), relevant, [METRIC])[METRIC]
        assert bitwise_equal(base[keep], evidence.baseline)

        projected = art.pipeline.transform_queries(sc.queries)
        shipped = maxsim_matrix(projected, art.compressed)
        got = per_query_metrics(shipped, relevant, [METRIC])[METRIC][keep]
        assert bitwise_equal(got, evidence.candidates[position])

        ranking = CompressedRetriever(art.compressed, art.pipeline).search(sc.queries, 5)
        via_ranking = metrics_from_ranking(ranking.indices, relevant, [METRIC])[METRIC][keep]
        assert bitwise_equal(via_ranking, evidence.candidates[position])

        # the in-memory result and the loaded artifact are one index
        assert bitwise_equal(art.compressed.codes, sc.result.compressed.codes)
        assert bitwise_equal(art.compressed.offsets, sc.result.compressed.offsets)
        assert art.compressed.ids == sc.result.compressed.ids == sc.corpus.ids
        assert art.pipeline.to_dict() == sc.result.pipeline.to_dict()
        assert pipeline_fitted_identity(art.pipeline) == list(evidence.fitted[position])
        assert pipeline_fitted_identity(art.pipeline) == pipeline_fitted_identity(sc.result.pipeline)
        assert sc.result.retriever().search(sc.queries, 10) == \
            CompressedRetriever(art.compressed, art.pipeline).search(sc.queries, 10)
        assert bitwise_equal(shipped, maxsim_matrix(
            sc.result.pipeline.transform_queries(sc.queries), sc.result.compressed))

    @pytest.mark.parametrize("name", SELECTED)
    def test_the_certificate_describes_what_was_asked_and_decided(self, scenarios, name):
        sc = scenarios[name]
        cert = sc.result.certificate
        data = cert.to_dict()
        d = sc.result.decision
        assert cert.certified is True and cert.outcome is Outcome.SELECT
        assert (cert.target, cert.error, cert.metric) == (sc.options["target"],
                                                          sc.options["error"], METRIC)
        assert cert.level == d.level and cert.p_values == d.p_values
        assert cert.certified_positions == d.certified_positions
        assert cert.space_identity == sc.result.evidence.space_identity
        assert data["corpus"]["dtype"] == str(sc.corpus.dtype)
        assert data["evidence"]["digest"] == sc.result.evidence.digest()
        assert data["candidate"]["pipeline"] == sc.result.selected.to_dict()["pipeline"]
        assert data["result"]["selected"] == sc.result.selected.position
        assert data["created_utc"] == CREATED
        assert cert.statement and cert.assumptions and isinstance(cert.basis_sensitive, bool)
        assert cert.basis_sensitive is True  # every selected pipeline here ends in a coordinate codec
        construct = "labels" if sc.qrels is not None else "agreement@1"
        assert cert.construct == construct == d.construct

    @pytest.mark.parametrize("name", SELECTED)
    def test_verification_passes_at_every_level_with_the_live_objects(self, scenarios, name):
        sc = scenarios[name]
        art = load_artifact(sc.root)
        cert = art.certificate
        integrity = verify_certificate(cert)
        replay = verify_certificate(cert, evidence=art.evidence)
        binding = verify_certificate(
            cert, evidence=art.evidence, corpus=sc.corpus, queries=sc.queries,
            reference=sc.reference, pipeline=art.pipeline, compressed=art.compressed)
        assert (integrity.ok, replay.ok, binding.ok) == (True, True, True), binding.summary()
        assert integrity.levels == ("integrity",)
        assert replay.levels == ("integrity", "replay")
        assert binding.levels == ("integrity", "replay", "binding") and binding.level == "binding"
        statuses = {c.name: c.status for c in binding.checks}
        assert statuses["replay.decision"] == "pass" and statuses["binding.codes"] == "pass"
        assert statuses["binding.pipeline"] == "pass" and statuses["binding.bytes"] == "pass"
        assert statuses["binding.corpus"] == statuses["binding.queries"] == "pass"
        assert statuses["binding.reference"] == statuses["binding.evidence_queries"] == "pass"
        assert [c for c in binding.checks if c.status == "fail"] == []
        assert statuses["signature"] == "skipped"  # unsigned: integrity, not security
        # the package-level helper reads the directory and does the same
        assert verify(sc.root, corpus=sc.corpus, queries=sc.queries,
                      reference=sc.reference).ok

    @pytest.mark.parametrize("name", SELECTED)
    def test_the_result_the_artifact_and_the_report_agree(self, scenarios, name):
        sc = scenarios[name]
        result = sc.result
        report = result.report()
        json.dumps(report)  # JSON-native
        assert report["outcome"] == "select" and report["certified"] is True
        assert report["certificate_digest"] == result.certificate.digest
        assert report["selected"]["position"] == result.selected.position
        assert report["compression_ratio"] == result.compression_ratio > 1.0
        assert report["n_queries"] == result.decision.n_queries
        assert report["statement"] == result.certificate.statement
        assert "retention" not in " ".join(k for k in report if k != "statement")  # no retention figure
        assert result.store is result.compressed
        text = repr(result)
        assert "certified=True" in text
        if report["construct"].startswith("agreement"):
            assert "not retrieval quality" in text

    def test_a_text_like_corpus_is_certified_through_a_projection(self, scenarios):
        """The selected pipeline carries fitted state (a PCA basis) that only the bundle can
        store: it is restored from file and used to project the queries."""
        sc = scenarios["text"]
        assert [s.name for s in sc.result.pipeline.stages] == ["pca", "binary"]
        art = load_artifact(sc.root)
        assert art.pipeline.fitted and (sc.root / "fitted.npz").is_file()
        assert art.compressed.dim == 48 and sc.corpus.dimension == 96
        assert art.certificate.basis_sensitive is True
        assert sc.result.compression_ratio == sc.result.compressed.report()["compression_vs_float32"]

    def test_a_vision_like_corpus_keeps_protected_rows_verbatim(self, scenarios):
        sc = scenarios["merge"]
        assert sc.corpus.protected is not None and sc.corpus.positions is not None
        art = load_artifact(sc.root)
        transformed = art.pipeline.transform(sc.corpus)
        assert transformed.num_vectors < sc.corpus.num_vectors  # tokens were merged
        for i in range(len(sc.corpus)):
            lo, hi = sc.corpus.span(i)
            protected = sc.corpus.vectors[lo:hi][sc.corpus.protected[lo:hi]]
            t_lo, t_hi = transformed.span(i)
            kept = transformed.vectors[t_lo:t_hi][transformed.protected[t_lo:t_hi]]
            assert np.array_equal(kept, protected)  # the instruction rows pass through verbatim
        assert sc.result.compressed.stats["vectors_before"] == sc.corpus.num_vectors
        assert sc.result.compressed.report()["vector_reduction"] > 1.0

    def test_float16_input_is_recorded_and_certified_as_float16(self, scenarios):
        sc = scenarios["float16"]
        assert sc.corpus.dtype == np.float16
        stats = sc.result.compressed.stats
        assert stats["native_dtype_before"] == "float16"
        assert stats["float32_bytes_before"] == 2 * stats["native_bytes_before"]
        report = sc.result.compressed.report()
        assert report["compression_vs_native"] < report["compression_vs_float32"]
        assert load_artifact(sc.root).certificate.to_dict()["corpus"]["dtype"] == "float16"

    def test_the_agreement_construct_is_named_and_never_taken_for_labels(self, scenarios):
        sc = scenarios["agreement"]
        cert = sc.result.certificate
        assert cert.construct == "agreement@1" and sc.qrels is None
        assert any("agreement" in c and "retrieval quality" in c for c in cert.caveats)
        art = load_artifact(sc.root)
        ok = verify_certificate(cert, evidence=art.evidence, reference=RankingReference())
        assert ok.ok and ok.status("binding.reference") == "pass"
        # binding the same certificate to labels is refused: the two constructs are not merged
        labels = LabelReference({q: {sc.corpus.ids[0]} for q in sc.queries.ids})
        wrong = verify_certificate(cert, evidence=art.evidence, reference=labels)
        assert not wrong.ok and wrong.status("binding.reference") == "fail"
        assert sc.result.report()["construct"] == "agreement@1"
        assert sc.result.report()["caveats"]
        # and the baseline of the agreement evidence is exactly 1 on every kept query
        assert (sc.result.evidence.baseline == 1.0).all()

    def test_the_query_order_and_the_rule_are_part_of_the_certificate_and_replay(self, scenarios):
        seeded = scenarios["seeded"]
        order = seeded.result.certificate.to_dict()["queries"]["order"]
        assert order["kind"] == "seeded_permutation" and order["seed"] == 5
        assert order["budget"] == 120 and seeded.result.decision.n_queries == 120
        art = load_artifact(seeded.root)
        assert verify_certificate(art.certificate, evidence=art.evidence).ok
        union = scenarios["bonferroni"].result
        assert union.decision.rule["name"] == "bonferroni_betting"
        assert union.certificate.to_dict()["rule"]["name"] == "bonferroni_betting"
        art = load_artifact(scenarios["bonferroni"].root)
        assert verify_certificate(art.certificate, evidence=art.evidence).status(
            "replay.decision") == "pass"


class TestUnlabelledQueriesAndTheFilesOfAnArtifact:
    def test_queries_without_a_relevant_document_are_dropped_for_every_candidate_together(
            self, scenarios):
        sc = scenarios["unlabelled"]
        evidence = sc.result.evidence
        n = len(sc.queries)
        expected_dropped = {sc.queries.ids[i] for i in range(n) if i % 5 in (0, 1)}
        assert set(evidence.dropped_queries) == expected_dropped
        assert evidence.n_queries == n - len(expected_dropped)
        assert set(evidence.query_ids).isdisjoint(expected_dropped)
        assert evidence.candidates.shape == (len(sc.result.decision.p_values), evidence.n_queries)
        assert not np.isnan(evidence.candidates).any() and not np.isnan(evidence.baseline).any()
        report = sc.result.report()
        assert report["n_queries_without_relevant_document"] == len(expected_dropped)
        assert report["n_queries_given"] == n
        art = load_artifact(sc.root)
        assert set(art.certificate.to_dict()["evidence"]["dropped_queries"]) == expected_dropped
        ok = verify_certificate(art.certificate, evidence=art.evidence, queries=sc.queries,
                                reference=sc.reference)
        assert ok.ok and ok.status("binding.evidence_queries") == "pass"
        # the same queries minus one kept query no longer match the evidence
        fewer = sc.queries.select(
            [i for i in range(n) if sc.queries.ids[i] != evidence.query_ids[0]])
        assert verify_certificate(art.certificate, evidence=art.evidence, queries=fewer,
                                  reference=sc.reference).status("binding.queries") == "fail"

    def test_a_corpus_with_an_empty_and_a_single_vector_document_survives_the_whole_path(
            self, scenarios):
        sc = scenarios["edge"]
        assert sc.corpus.counts[1] == 1 and sc.corpus.counts[2] == 0
        art = load_artifact(sc.root)
        assert art.compressed.counts[2] == 0 and len(art.compressed) == len(sc.corpus)
        ranking = CompressedRetriever(art.compressed, art.pipeline).search(sc.queries, len(sc.corpus))
        assert all(row[-1] == sc.corpus.ids[2] for row in ranking.ids)  # the empty document is last
        assert np.isneginf(ranking.scores[:, -1]).all()
        assert ranking == sc.result.retriever().search(sc.queries, len(sc.corpus))

    def test_certify_takes_the_list_of_arrays_that_optimize_takes_and_changes_no_input(
            self, scenarios):
        sc = scenarios["text"]
        before = (corpus_identity(sc.corpus), corpus_identity(sc.queries))
        again = certify(sc.corpus, sc.queries, sc.qrels, space=space(), target=0.9, error=0.1,
                        created=CREATED)
        assert (corpus_identity(sc.corpus), corpus_identity(sc.queries)) == before
        assert again.certificate == sc.result.certificate  # same inputs and created: same record
        docs = [np.array(sc.corpus.vectors[lo:hi]) for lo, hi in
                (sc.corpus.span(i) for i in range(len(sc.corpus)))]
        queries = [np.array(sc.queries.vectors[lo:hi]) for lo, hi in
                   (sc.queries.span(i) for i in range(len(sc.queries)))]
        ids = {q: sc.corpus.ids.index(next(iter(d))) for q, d in sc.qrels.items()}
        qrels = {str(i): {str(ids[q])} for i, q in enumerate(sc.queries.ids)}
        listed = certify(docs, queries, qrels, space=space(), target=0.9, error=0.1,
                         created=CREATED)
        assert listed.corpus.ids == [str(i) for i in range(len(docs))]
        assert listed.outcome is Outcome.SELECT
        assert listed.selected.position == sc.result.selected.position
        assert listed.decision.p_values == sc.result.decision.p_values  # same arrays, same p

    @pytest.mark.parametrize("metric", ["recall@5", "mrr@10", "hit@3", "NDCG@10"])
    def test_other_metrics_are_certified_and_reproduced_the_same_way(
            self, scenarios, tmp_path, metric):
        """Only nDCG@k was evaluated by the registered experiments; the others are an engineering
        option, and the path (evidence, decision, bundle, replay, shipped metrics) is the same."""
        sc = scenarios["text"]
        result = certify(sc.corpus, sc.queries, sc.qrels, space=space(), metric=metric,
                         target=0.9, error=0.1, created=CREATED)
        lowered = metric.lower()
        assert result.evidence.metric == lowered == result.decision.metric
        assert result.certificate.metric == lowered
        art = load_artifact(result.save(tmp_path / "m"))
        assert art.certificate.metric == lowered
        if art.pipeline is not None:  # something was certified: the shipped index gives the record
            row = art.evidence.candidates[art.certificate.selected_position]
            scores = maxsim_matrix(art.pipeline.transform_queries(sc.queries), art.compressed)
            shipped = per_query_metrics(scores, relevant_sets(sc), [lowered])[lowered]
            assert bitwise_equal(shipped[kept_queries(sc)], row)
        assert verify_certificate(art.certificate, evidence=art.evidence, corpus=sc.corpus,
                                  queries=sc.queries, reference=sc.reference).ok

    def test_the_declared_encoder_is_recorded_and_never_changes_a_decision(self, scenarios):
        """The procedure is model-blind: an ``EncoderIdentity`` is descriptive metadata. Whatever
        is declared (or nothing), the evidence, the p-values, the certified set and the choice
        are the same, and only the record of the encoder differs."""
        sc = scenarios["text"]
        declared = [
            EncoderIdentity(model_id="synthetic-a", dimension=96, modality="text", normalized=True),
            {"model_id": "synthetic-b", "dimension": 96, "modality": "image", "revision": "r1",
             "extra": {"pooling": "none"}},
        ]
        for encoder in declared:
            result = certify(sc.corpus, sc.queries, sc.qrels, space=space(), target=0.9,
                             error=0.1, created=CREATED, encoder=encoder)
            assert result.evidence.digest() == sc.result.evidence.digest()
            assert result.decision.to_dict() == sc.result.decision.to_dict()
            record = result.certificate.to_dict()["encoder"]
            assert record["model_id"] == (encoder.model_id if isinstance(encoder, EncoderIdentity)
                                          else encoder["model_id"])
            assert result.certificate.digest != sc.result.certificate.digest  # the record differs
            assert sc.result.certificate.to_dict()["encoder"] is None
            art = load_artifact(result.save(Path(sc.root).parent / f"enc-{record['model_id']}"))
            assert verify_certificate(art.certificate, evidence=art.evidence).ok
        with pytest.raises(ValueError, match="dimension"):
            certify(sc.corpus, sc.queries, sc.qrels, space=space(), created=CREATED,
                    encoder=EncoderIdentity(model_id="x", dimension=64, modality="text"))

    def test_a_directory_is_not_overwritten_unless_asked_and_other_files_survive(
            self, scenarios, tmp_path):
        sc = scenarios["text"]
        root = sc.result.save(tmp_path / "a")
        with pytest.raises(FileExistsError, match="overwrite"):
            sc.result.save(root)
        (root / "notes.txt").write_text("keep me", encoding="utf-8")
        again = sc.result.save(root, overwrite=True)
        assert again == root and (root / "notes.txt").read_text(encoding="utf-8") == "keep me"
        assert load_artifact(root).certificate == sc.result.certificate
        # the same files byte for byte: the manifest is a function of the content only
        other = sc.result.save(tmp_path / "b")
        for name in ("MANIFEST.json", "certificate.json", "codes.npz", "fitted.npz", "evidence.npz"):
            assert (root / name).read_bytes() == (other / name).read_bytes(), name

    def test_an_artifact_without_evidence_loads_and_verifies_without_the_replay(
            self, scenarios, tmp_path):
        sc = scenarios["text"]
        root = sc.result.save(tmp_path / "lean", include_evidence=False)
        assert not (root / "evidence.npz").exists()
        art = load_artifact(root)
        assert art.evidence is None and art.verification.ok
        report = verify_certificate(art.certificate, corpus=sc.corpus, queries=sc.queries,
                                    reference=sc.reference, pipeline=art.pipeline,
                                    compressed=art.compressed)
        assert report.ok and report.levels == ("integrity", "binding")
        assert report.status("replay.decision") == "skipped"
        assert verify(root, corpus=sc.corpus, queries=sc.queries, reference=sc.reference).ok

    def test_a_directory_that_is_not_an_artifact_is_refused_by_name(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a directory"):
            load_artifact(tmp_path / "missing")
        (tmp_path / "empty").mkdir()
        with pytest.raises(ArtifactError, match="MANIFEST"):
            load_artifact(tmp_path / "empty")


# =========================================================================== nothing certified


class TestAbstainingAndFallingBack:
    def test_an_abstention_ships_the_uncompressed_corpus_and_says_so(self, scenarios):
        sc = scenarios["abstain"]
        result = sc.result
        assert result.outcome is Outcome.ABSTAIN and result.certified is False
        assert result.selected is None and result.pipeline is None and result.compressed is None
        assert result.store is sc.corpus and result.compression_ratio == 1.0
        assert isinstance(result.retriever(), ExactRetriever)
        assert result.retriever().search(sc.queries, 10) == ExactRetriever(sc.corpus).search(
            sc.queries, 10)
        report = result.report()
        assert report["not_certified_note"] and report["selected"] is None
        assert "uncompressed" in report["not_certified_note"]
        assert "nothing certified" in repr(result)
        assert result.decision.certified_positions == ()

    def test_an_abstention_artifact_holds_a_certificate_and_no_codes(self, scenarios):
        sc = scenarios["abstain"]
        files = sorted(p.name for p in sc.root.iterdir())
        assert files == ["MANIFEST.json", "certificate.json", "evidence.npz"]
        art = load_artifact(sc.root)
        assert art.pipeline is None and art.compressed is None and art.evidence is not None
        cert = art.certificate
        assert cert.outcome is Outcome.ABSTAIN and cert.certified is False
        assert cert.to_dict()["candidate"] is None and cert.selected_position is None
        report = verify_certificate(cert, evidence=art.evidence, corpus=sc.corpus,
                                    queries=sc.queries, reference=sc.reference)
        assert report.ok and report.levels == ("integrity", "replay", "binding")
        assert report.status("binding.pipeline") == "skipped"
        assert report.status("binding.codes") == "skipped"

    def test_a_fallback_is_shipped_but_never_certified(self, scenarios):
        sc = scenarios["fallback"]
        result = sc.result
        assert result.outcome is Outcome.FALLBACK and result.certified is False
        assert result.selected.position == -1 and result.decision.certified_positions == ()
        assert result.pipeline.to_dict() == Pipeline([Float16Quantizer()]).to_dict()
        assert result.compression_ratio > 1.0
        assert "NOT certified" in repr(result) and result.report()["not_certified_note"]
        art = load_artifact(sc.root)
        assert art.certificate.outcome is Outcome.FALLBACK and art.certificate.certified is False
        assert any("not certified" in c.lower() for c in art.certificate.caveats)
        report = verify_certificate(
            art.certificate, evidence=art.evidence, corpus=sc.corpus, queries=sc.queries,
            reference=sc.reference, pipeline=art.pipeline, compressed=art.compressed)
        assert report.ok
        assert result.retriever().search(sc.queries, 5) == CompressedRetriever(
            art.compressed, art.pipeline).search(sc.queries, 5)


# ============================================================================= tampering


def edit_json(path: Path, change) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def rehash(root: Path) -> None:
    """What someone who can write the directory can do: make the manifest match again."""
    import hashlib

    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name in manifest["files"]:
        data = (root / name).read_bytes()
        if name.endswith(".json"):
            data = data.replace(b"\r\n", b"\n")
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")


class TestTamperingIsNoticed:
    @pytest.fixture
    def copy(self, scenarios, tmp_path):
        sc = scenarios["text"]
        target = tmp_path / "artifact"
        shutil.copytree(sc.root, target)
        return sc, target

    def test_a_flipped_byte_in_the_codes_fails_the_manifest(self, copy):
        _, root = copy
        data = bytearray((root / "codes.npz").read_bytes())
        data[len(data) // 2] ^= 0xFF
        (root / "codes.npz").write_bytes(bytes(data))
        with pytest.raises(ArtifactError, match="manifest"):
            load_artifact(root)

    def test_an_edited_certificate_fails_the_manifest_and_then_its_own_digest(self, copy):
        _, root = copy
        edit_json(root / "certificate.json", lambda d: d.__setitem__("target", 0.5))
        with pytest.raises(ArtifactError, match="manifest"):
            load_artifact(root)
        rehash(root)  # an attacker who can write the directory fixes the manifest too ...
        with pytest.raises(ArtifactError, match="digest"):  # ... but not the certificate digest
            load_artifact(root)
        read = load_artifact(root, verify=False).certificate  # readable, and reported as wrong
        report = verify_certificate(read)
        assert not report.ok and report.status("digest") == "fail"

    def test_swapped_evidence_fails_replay(self, copy, scenarios):
        _, root = copy
        other = scenarios["agreement"].result.evidence
        other.to_npz(root / "evidence.npz")
        rehash(root)
        with pytest.raises(ArtifactError, match="evidence"):
            load_artifact(root)

    @pytest.mark.parametrize("what", ["corpus", "queries", "reference", "pipeline", "codes"])
    def test_a_live_object_that_is_not_the_certified_one_fails_binding(self, scenarios, what):
        sc = scenarios["text"]
        art = load_artifact(sc.root)
        live: dict[str, Any] = {"corpus": sc.corpus, "queries": sc.queries,
                                "reference": sc.reference, "pipeline": art.pipeline,
                                "compressed": art.compressed}
        if what == "corpus":
            v = np.array(sc.corpus.vectors, copy=True)
            v[0, 0] += np.float32(0.01)
            live["corpus"] = MultiVectorCorpus(v, sc.corpus.offsets, ids=sc.corpus.ids)
            failing = "binding.corpus"
        elif what == "queries":
            live["queries"] = sc.queries.select(range(len(sc.queries) - 1))
            failing = "binding.queries"
        elif what == "reference":
            live["reference"] = LabelReference({q: {sc.corpus.ids[0]} for q in sc.queries.ids})
            failing = "binding.reference"
        elif what == "pipeline":
            other = Pipeline.from_dict(art.pipeline.to_dict())
            other.fit(make_corpus(30, 96, tokens=(20, 40), seed=99, anisotropy=0.5))  # other data
            live["pipeline"] = other
            failing = "binding.pipeline"
        else:
            codes = np.array(art.compressed.codes, copy=True)
            codes[0, 0] ^= 0xFF
            live["compressed"] = type(art.compressed)(
                codes, art.compressed.offsets, art.compressed.dim, art.compressed.quantizer,
                art.compressed.ids, stats=art.compressed.stats, pipeline=art.compressed.pipeline)
            failing = "binding.codes"
        report = verify_certificate(art.certificate, evidence=art.evidence, **live)
        assert not report.ok
        assert report.status(failing) == "fail", report.summary()
        others = [c.name for c in report.checks if c.status == "fail" and c.name != failing]
        assert all(name.startswith("binding.") for name in others), others
        assert report.status("replay.decision") == "pass"  # the recorded decision is still sound

    def test_a_signature_needs_the_key_and_a_wrong_key_fails(self, scenarios):
        art = load_artifact(scenarios["text"].root)
        signed = sign_certificate(art.certificate, b"k" * 32, key_id="test")
        assert signed.signed and signed.digest == art.certificate.digest
        assert verify_certificate(signed, key=b"k" * 32).status("signature") == "pass"
        assert verify_certificate(signed).status("signature") == "skipped"
        wrong = verify_certificate(signed, key=b"x" * 32)
        assert not wrong.ok and wrong.status("signature") == "fail"
        assert verify_certificate(art.certificate, key=b"k" * 32).status("signature") == "fail"
        reread = Certificate.from_json(signed.to_json())
        assert reread == signed and verify_certificate(reread, key=b"k" * 32).ok
