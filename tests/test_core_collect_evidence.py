"""``collect_evidence`` and ``compress_selected``: evidence the selection rule is allowed to see.

The evidence of every candidate must equal an independent recomputation with the library's own
functions (``calibration.score_space``, ``evaluation.per_query_metrics``); queries without a
relevant document are dropped for all candidates together; the collector is deterministic and
leaves its inputs alone; a stage fitted on queries is refused; the chosen pipeline can be
fitted again and must reproduce the fitted state that was evaluated.

This module also hosts the small seeded problem generator and candidate space that the
certificate and bundle tests share. Nothing here asserts a retention number as if it meant
anything: the data are synthetic and only the structure of the results is checked.
"""

from __future__ import annotations

import numpy as np
import pytest

import optivision.core.evaluate.collect as collect_module
from optivision.calibration import score_space
from optivision.compose import Pipeline
from optivision.core.evaluate import (
    Evidence,
    LabelReference,
    RankingReference,
    collect_evidence,
    compress_selected,
)
from optivision.core.evaluate.collect import CandidateEvaluationError, FittedStateMismatchError
from optivision.core.selection import CandidateSpace, Outcome, select
from optivision.core.stages import (
    STAGES,
    AdaptiveMerge,
    BasisInvariance,
    BinaryQuantizer,
    DimensionProjector,
    Float16Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    PCAProjector,
    Reduction,
    StageTraits,
    TokenReducer,
)
from optivision.core.vectors import ContractError, corpus_identity
from optivision.evaluation import per_query_metrics, relevant_from_baseline, relevant_from_qrels
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix

try:
    import scipy.cluster.hierarchy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_SCIPY = False


# ----------------------------------------------------------- shared test fixtures


def make_problem(seed=0, n_docs=40, per_doc=8, dim=16, n_queries=80, n_unlabelled=3,
                 q_tokens=5, group=4, spread=0.9, qnoise=0.7):
    """A small seeded corpus, queries and qrels.

    Documents come in groups that share a direction, so a query is sometimes ambiguous between
    documents of its group and the per-query metric takes several values. The leading
    ``n_unlabelled`` queries have no qrels entry. ``RandomState`` is used because its stream is
    frozen across NumPy versions.
    """
    rs = np.random.RandomState(seed)
    group_dirs = rs.standard_normal((n_docs // group + 1, dim))
    doc_dirs = group_dirs[np.arange(n_docs) // group] + 0.6 * rs.standard_normal((n_docs, dim))
    docs = doc_dirs[:, None, :] + spread * rs.standard_normal((n_docs, per_doc, dim))
    docs = (docs / np.linalg.norm(docs, axis=2, keepdims=True)).astype(np.float32)
    corpus = MultiVectorCorpus(docs.reshape(-1, dim), np.arange(n_docs + 1) * per_doc,
                               ids=[f"d{i}" for i in range(n_docs)])
    blocks, qrels = [], {}
    for j in range(n_queries):
        i = int(rs.randint(n_docs))
        rows = docs[i][rs.choice(per_doc, q_tokens, replace=False)]
        rows = rows + qnoise * rs.standard_normal((q_tokens, dim)).astype(np.float32)
        blocks.append((rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32))
        if j >= n_unlabelled:
            qrels[f"q{j}"] = [f"d{i}"]
    queries = MultiVectorCorpus(np.concatenate(blocks), np.arange(n_queries + 1) * q_tokens,
                                ids=[f"q{j}" for j in range(n_queries)])
    return corpus, queries, qrels


def small_space(name="small"):
    """Seven candidates in four families, with and without fitted state."""
    return CandidateSpace.from_families({
        "float16": [Pipeline([Float16Quantizer()])],
        "merge+int8": [
            Pipeline([Int8Quantizer("per_vector")]),
            Pipeline([AdaptiveMerge(radius=0.9, center="mean"), Int8Quantizer("per_vector")]),
            Pipeline([AdaptiveMerge(radius=0.7, center="mean"), Int8Quantizer("per_vector")]),
        ],
        "pca+int4": [
            Pipeline([PCAProjector(12, center=True), Int4Quantizer(center="mean")]),
            Pipeline([PCAProjector(8, center=True), Int4Quantizer(center="mean")]),
        ],
        "binary": [Pipeline([BinaryQuantizer()])],
    }, name=name)


@pytest.fixture(scope="module")
def problem():
    return make_problem()


@pytest.fixture(scope="module")
def space():
    return small_space()


@pytest.fixture(scope="module")
def labels_evidence(problem, space):
    corpus, queries, qrels = problem
    return collect_evidence(corpus, queries, space, LabelReference(qrels))


@pytest.fixture(scope="module")
def agreement_evidence(problem, space):
    corpus, queries, _ = problem
    return collect_evidence(corpus, queries, space, RankingReference())


def independent(corpus, queries, space, relevant_of, metric="ndcg@5"):
    """The evidence arrays computed straight from the library, without the collector."""
    groups: dict[str, list[Pipeline]] = {}
    for spec, pipe in zip(space.candidates, space.pipelines(), strict=True):
        groups.setdefault(spec.family, []).append(pipe)
    scored = score_space(corpus, queries, groups)
    base = maxsim_matrix(queries, corpus)
    relevant = relevant_of(base)
    base_pq = per_query_metrics(base, relevant, [metric])[metric]
    keep = ~np.isnan(base_pq)
    rows = [per_query_metrics(sc.scores, relevant, [metric])[metric][keep] for sc in scored]
    return {
        "baseline": base_pq[keep],
        "candidates": np.stack(rows),
        "bytes": np.array([sc.bytes_per_doc for sc in scored]),
        "vectors": np.array([float(sc.report["vectors_per_doc"]) for sc in scored]),
        "keep": keep,
    }


# --------------------------------------------------------------------- the evidence


class TestAgainstIndependentRecomputation:
    def test_labels(self, problem, space, labels_evidence):
        corpus, queries, qrels = problem
        ref = independent(corpus, queries, space,
                          lambda base: relevant_from_qrels(qrels, queries.ids, corpus.ids))
        ev = labels_evidence
        assert np.array_equal(ev.baseline, ref["baseline"])
        assert np.array_equal(ev.candidates, ref["candidates"])
        assert np.array_equal(ev.bytes_per_doc, ref["bytes"])
        assert np.array_equal(ev.vectors_per_doc, ref["vectors"])
        assert ev.construct == "labels" and ev.metric == "ndcg@5"
        assert ev.space_identity == space.identity

    def test_agreement(self, problem, space, agreement_evidence):
        corpus, queries, _ = problem
        ref = independent(corpus, queries, space, lambda base: relevant_from_baseline(base, 1))
        ev = agreement_evidence
        assert np.array_equal(ev.baseline, ref["baseline"])
        assert np.array_equal(ev.candidates, ref["candidates"])
        assert ev.construct == "agreement@1"

    @pytest.mark.parametrize("metric", ["ndcg@10", "recall@5", "mrr@10", "hit@5"])
    def test_other_metrics_follow_the_library(self, problem, space, metric):
        corpus, queries, qrels = problem
        ev = collect_evidence(corpus, queries, space, LabelReference(qrels), metric=metric)
        ref = independent(corpus, queries, space,
                          lambda base: relevant_from_qrels(qrels, queries.ids, corpus.ids),
                          metric=metric)
        assert ev.metric == metric
        assert np.array_equal(ev.candidates, ref["candidates"])

    def test_metric_name_is_stored_in_lower_case(self, problem, space):
        corpus, queries, qrels = problem
        ev = collect_evidence(corpus, queries, space, LabelReference(qrels), metric="NDCG@5")
        assert ev.metric == "ndcg@5"

    def test_arrays_are_float64_and_the_sizes_are_the_librarys(self, labels_evidence, space):
        ev = labels_evidence
        for array in (ev.baseline, ev.candidates, ev.bytes_per_doc, ev.vectors_per_doc):
            assert array.dtype == np.float64
        assert ev.candidates.shape == (len(space), ev.n_queries)
        assert ev.baseline.shape == (ev.n_queries,)
        # float16 stores 2 bytes per coordinate: 8 vectors x 16 dims x 2
        assert ev.bytes_per_doc[0] == 8 * 16 * 2
        assert ev.vectors_per_doc[0] == 8.0

    def test_fitted_state_digests_are_recorded_per_candidate_and_stage(self, labels_evidence,
                                                                         space):
        ev = labels_evidence
        assert len(ev.fitted) == len(space)
        for entry, spec in zip(ev.fitted, space.candidates, strict=True):
            assert len(entry) == len(spec.pipeline["stages"])
        assert ev.fitted[0] == (None,)  # float16: no state
        assert ev.fitted[4][0] is not None and ev.fitted[4][1] is not None  # pca, centred int4


class TestAgreementConstruct:
    def test_baseline_is_exactly_one_everywhere(self, agreement_evidence):
        assert np.all(agreement_evidence.baseline == 1.0)

    def test_candidates_lie_on_the_ndcg_grid(self, agreement_evidence):
        grid = np.concatenate([[0.0], 1.0 / np.log2(np.arange(2, 7))])
        assert np.isin(agreement_evidence.candidates, grid).all()
        assert agreement_evidence.candidates.max() <= 1.0

    def test_an_uncompressed_looking_candidate_agrees_with_itself(self, problem):
        corpus, queries, _ = problem
        one = CandidateSpace.from_families({"f32": [Pipeline([])]}, name="identity")
        ev = collect_evidence(corpus, queries, one, RankingReference())
        assert np.all(ev.candidates == 1.0)

    def test_depth_changes_the_construct_name(self, problem, space):
        corpus, queries, _ = problem
        ev = collect_evidence(corpus, queries, space, RankingReference(depth=3))
        assert ev.construct == "agreement@3"


class TestUnlabelledQueries:
    def test_dropped_for_the_baseline_and_all_candidates_together(self, problem, labels_evidence):
        _, queries, qrels = problem
        ev = labels_evidence
        assert ev.dropped_queries == ("q0", "q1", "q2")
        assert ev.query_ids == tuple(q for q in queries.ids if q in qrels)
        assert ev.n_queries == len(queries) - 3
        assert ev.candidates.shape[1] == ev.baseline.shape[0] == len(ev.query_ids)
        assert not np.isnan(ev.baseline).any() and not np.isnan(ev.candidates).any()

    def test_queries_whose_documents_are_missing_from_the_corpus_are_dropped_too(self, problem,
                                                                                 space):
        corpus, queries, qrels = problem
        odd = dict(qrels)
        odd["q10"] = ["no-such-document"]
        ev = collect_evidence(corpus, queries, space, LabelReference(odd))
        assert "q10" in ev.dropped_queries and "q10" not in ev.query_ids

    def test_the_agreement_reference_never_drops(self, agreement_evidence, problem):
        assert agreement_evidence.dropped_queries == ()
        assert agreement_evidence.n_queries == len(problem[1])

    def test_no_labelled_query_at_all_is_an_error(self, problem, space):
        corpus, queries, _ = problem
        with pytest.raises(ValueError, match="no query has a relevant document"):
            collect_evidence(corpus, queries, space, LabelReference({}))

    def test_a_nan_on_a_kept_query_for_any_candidate_raises(self, problem, space, monkeypatch):
        corpus, queries, qrels = problem
        real = collect_module.per_query_metrics
        calls = []

        def poisoned(scores, relevant, metrics):
            out = real(scores, relevant, metrics)
            calls.append(1)
            if len(calls) == 3:  # the baseline is call 1; this is the second candidate
                key = next(iter(out))
                kept = [i for i, rel in enumerate(relevant) if len(rel)]
                out[key][kept[5]] = np.nan
            return out

        monkeypatch.setattr(collect_module, "per_query_metrics", poisoned)
        with pytest.raises(CandidateEvaluationError, match="NaN metric"):
            collect_evidence(corpus, queries, space, LabelReference(qrels))


class TestDeterminismAndPurity:
    def test_two_runs_give_the_same_digest(self, problem, space, labels_evidence):
        corpus, queries, qrels = problem
        again = collect_evidence(corpus, queries, space, LabelReference(qrels))
        assert again.digest() == labels_evidence.digest()
        assert again == labels_evidence

    def test_inputs_are_not_modified(self, problem, space):
        corpus, queries, qrels = problem
        before = (corpus_identity(corpus), corpus_identity(queries))
        vectors = (corpus.vectors.copy(), queries.vectors.copy())
        collect_evidence(corpus, queries, space, LabelReference(qrels))
        collect_evidence(corpus, queries, space, RankingReference())
        assert (corpus_identity(corpus), corpus_identity(queries)) == before
        assert np.array_equal(corpus.vectors, vectors[0])
        assert np.array_equal(queries.vectors, vectors[1])

    def test_the_space_is_untouched_and_its_pipelines_are_rebuilt_fresh(self, problem):
        corpus, queries, qrels = problem
        space = small_space()
        identity = space.identity
        collect_evidence(corpus, queries, space, LabelReference(qrels))
        assert space.identity == identity
        fresh = space.pipelines()[4]  # pca + centred int4: both stages hold state once fitted
        assert not fresh.stages[0].fitted and not fresh.stages[1].fitted

    def test_progress_counts_from_one(self, problem, space):
        corpus, queries, qrels = problem
        seen = []
        collect_evidence(corpus, queries, space, LabelReference(qrels),
                         progress=lambda i, n, family: seen.append((i, n, family)))
        assert [s[0] for s in seen] == list(range(1, len(space) + 1))
        assert {s[1] for s in seen} == {len(space)}
        assert [s[2] for s in seen] == list(space.candidate_families)


class TestInputValidation:
    def test_a_non_finite_corpus_is_refused(self, problem, space):
        corpus, queries, qrels = problem
        vectors = corpus.vectors.copy()
        vectors[3, 2] = np.nan
        bad = MultiVectorCorpus(vectors, corpus.offsets, ids=corpus.ids)
        with pytest.raises(ContractError, match="non-finite"):
            collect_evidence(bad, queries, space, LabelReference(qrels))

    def test_non_finite_queries_are_refused(self, problem, space):
        corpus, queries, qrels = problem
        vectors = queries.vectors.copy()
        vectors[0, 0] = np.inf
        bad = MultiVectorCorpus(vectors, queries.offsets, ids=queries.ids)
        with pytest.raises(ContractError, match="queries"):
            collect_evidence(corpus, bad, space, LabelReference(qrels))

    def test_duplicate_query_ids_are_refused(self, problem, space):
        corpus, queries, qrels = problem
        ids = list(queries.ids)
        ids[1] = ids[0]
        bad = MultiVectorCorpus(queries.vectors, queries.offsets, ids=ids)
        with pytest.raises(ContractError, match="used by"):
            collect_evidence(corpus, bad, space, LabelReference(qrels))

    def test_a_dimension_mismatch_is_refused(self, problem, space):
        corpus, queries, qrels = problem
        wide = MultiVectorCorpus(np.zeros((queries.num_vectors, 20), np.float32), queries.offsets,
                                 ids=queries.ids)
        with pytest.raises(ContractError, match="dimension"):
            collect_evidence(corpus, wide, space, LabelReference(qrels))

    def test_wrong_argument_types(self, problem, space):
        corpus, queries, qrels = problem
        with pytest.raises(TypeError, match="CandidateSpace"):
            collect_evidence(corpus, queries, {"a": []}, LabelReference(qrels))
        with pytest.raises(TypeError, match="reference"):
            collect_evidence(corpus, queries, space, qrels)
        with pytest.raises(TypeError, match="progress"):
            collect_evidence(corpus, queries, space, LabelReference(qrels), progress=3)

    def test_an_unsupported_metric_is_refused(self, problem, space):
        corpus, queries, qrels = problem
        with pytest.raises(ValueError, match="unsupported metric"):
            collect_evidence(corpus, queries, space, LabelReference(qrels), metric="map@5")

    def test_a_misbehaving_reference_is_refused(self, problem, space):
        corpus, queries, _ = problem

        class Bad:
            construct = "labels"

            def relevant(self, corpus_ids, query_ids, base_scores):
                return [np.array([10**6])] * len(query_ids)

            def identity(self):
                return {"construct": "labels"}

        with pytest.raises(ValueError, match="outside"):
            collect_evidence(corpus, queries, space, Bad())


# --------------------------------------------------- stages fitted on documents only


class _Keep(TokenReducer):
    """Keeps every row; used to test what the collector does with custom stages."""

    name = "test_keep_all"

    def params(self):
        return {}

    def reduce(self, doc):
        return Reduction(doc.vectors.copy(), np.arange(doc.vectors.shape[0]))


class _KeepDeclared(_Keep):
    name = "test_keep_all_declared"
    traits = StageTraits(name="test_keep_all_declared", kind="tokens",
                         basis_invariance=BasisInvariance.ORTHOGONAL)


class _Boom(_KeepDeclared):
    name = "test_boom"
    traits = StageTraits(name="test_boom", kind="tokens",
                         basis_invariance=BasisInvariance.ORTHOGONAL)

    def fit(self, corpus, queries=None):
        raise RuntimeError("boom")


@pytest.fixture
def custom_stages(monkeypatch):
    for cls in (_Keep, _KeepDeclared, _Boom):
        monkeypatch.setitem(STAGES, cls.name, cls)


class TestStagesAreFittedOnDocumentsOnly:
    @pytest.mark.parametrize("make", [
        lambda: Pipeline([PCAProjector(8, basis="joint"), Int8Quantizer("per_vector")]),
        lambda: Pipeline([DimensionProjector(8, method="pca", basis="joint"),
                          Int4Quantizer(center="mean")]),
    ], ids=["pca-joint", "project-pca-joint"])
    def test_a_stage_that_reads_queries_is_refused_before_any_work(self, problem, make):
        corpus, queries, qrels = problem
        space = CandidateSpace.from_families({
            "ok": [Pipeline([Float16Quantizer()])], "joint": [make()]}, name="joint")
        progress = []
        with pytest.raises(ValueError) as caught:
            collect_evidence(corpus, queries, space, LabelReference(qrels),
                             progress=lambda *a: progress.append(a))
        message = str(caught.value)
        assert "documents only" in message and "A3" in message and "candidate 1" in message
        assert progress == []  # the guard ran before any candidate was scored

    def test_the_documents_basis_is_accepted(self, problem):
        corpus, queries, qrels = problem
        space = CandidateSpace.from_families({
            "docs": [Pipeline([PCAProjector(8, basis="documents"), Int8Quantizer("per_vector")])]},
            name="docs")
        assert collect_evidence(corpus, queries, space, LabelReference(qrels)).n_candidates == 1

    def test_a_stage_without_declared_traits_is_refused_with_the_remedy(self, problem,
                                                                        custom_stages):
        corpus, queries, qrels = problem
        space = CandidateSpace.from_families({"x": [Pipeline([_Keep(), Float16Quantizer()])]})
        with pytest.raises(ValueError, match="register_traits"):
            collect_evidence(corpus, queries, space, LabelReference(qrels))

    def test_a_stage_that_declares_documents_only_is_accepted(self, problem, custom_stages):
        corpus, queries, qrels = problem
        space = CandidateSpace.from_families(
            {"x": [Pipeline([_KeepDeclared(), Float16Quantizer()])]})
        ev = collect_evidence(corpus, queries, space, LabelReference(qrels))
        assert np.array_equal(ev.candidates[0], ev.baseline)  # keeping every row changes nothing

    def test_a_candidate_that_fails_raises_and_is_never_skipped(self, problem, custom_stages):
        corpus, queries, qrels = problem
        space = CandidateSpace.from_families({
            "a": [Pipeline([Float16Quantizer()])],
            "b": [Pipeline([_Boom(), Float16Quantizer()]), Pipeline([BinaryQuantizer()])]})
        with pytest.raises(CandidateEvaluationError, match="candidate 1") as caught:
            collect_evidence(corpus, queries, space, LabelReference(qrels))
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "never skipped" in str(caught.value)


# ------------------------------------------------------------------- compress_selected


class TestCompressSelected:
    def test_reproduces_the_fitted_state_recorded_for_every_candidate(self, problem, space,
                                                                      labels_evidence):
        corpus, _, _ = problem
        ev = labels_evidence
        for spec in space.candidates:
            pipeline, compressed = compress_selected(
                corpus, spec, expected_fitted=ev.fitted[spec.position])
            assert pipeline.fitted
            report = compressed.report()
            assert report["bytes_per_doc"] == ev.bytes_per_doc[spec.position]
            assert report["vectors_per_doc"] == ev.vectors_per_doc[spec.position]

    def test_a_wrong_expected_digest_raises(self, problem, space, labels_evidence):
        corpus, _, _ = problem
        spec = space[4]
        good = labels_evidence.fitted[4]
        flipped = ("0" * 64, good[1])
        with pytest.raises(FittedStateMismatchError, match="stage 0 .*pca"):
            compress_selected(corpus, spec, expected_fitted=flipped)
        with pytest.raises(FittedStateMismatchError, match="no state"):
            compress_selected(corpus, spec, expected_fitted=(good[0], None))
        with pytest.raises(FittedStateMismatchError, match="stage digest"):
            compress_selected(corpus, spec, expected_fitted=good[:1])

    def test_a_different_corpus_does_not_reproduce_the_fit(self, space, labels_evidence):
        other, _, _ = make_problem(seed=5)
        with pytest.raises(FittedStateMismatchError, match="not be the one that was evaluated"):
            compress_selected(other, space[4], expected_fitted=labels_evidence.fitted[4])

    def test_without_expected_digests_nothing_is_compared(self, problem, space):
        corpus, _, _ = problem
        pipeline, compressed = compress_selected(corpus, space[4])
        assert len(compressed) == len(corpus) and pipeline.fitted

    def test_accepts_a_spec_a_mapping_or_a_pipeline_without_touching_them(self, problem, space):
        corpus, _, _ = problem
        spec = space[2]
        original = Pipeline.from_dict(dict(spec.pipeline.items()))
        for given in (spec, spec.to_pipeline().to_dict(), original):
            pipeline, _ = compress_selected(corpus, given)
            assert pipeline is not original and pipeline.fitted
        assert not original.fitted  # a Pipeline passed in is rebuilt, never fitted in place

    def test_refuses_the_wrong_types_and_queries_fitted_stages(self, problem):
        corpus, _, _ = problem
        with pytest.raises(TypeError, match="CandidateSpec"):
            compress_selected(corpus, "float16")
        joint = Pipeline([PCAProjector(8, basis="joint"), Int8Quantizer("per_vector")])
        with pytest.raises(ValueError, match="documents only"):
            compress_selected(corpus, joint)

    def test_the_input_corpus_is_not_modified(self, problem, space):
        corpus, _, _ = problem
        before = corpus_identity(corpus)
        compress_selected(corpus, space[5])
        assert corpus_identity(corpus) == before


# ------------------------------------------------- the evidence feeds the selection rule


class TestSelectionOnCollectedEvidence:
    def test_an_easy_target_selects_something_consistent(self, space, labels_evidence):
        d = select(space, labels_evidence, target=0.5, error=0.2)
        assert d.outcome is Outcome.SELECT and d.certified
        assert d.selected.position in d.certified_positions
        smallest = min(d.certified_positions,
                       key=lambda k: (labels_evidence.bytes_per_doc[k],
                                      labels_evidence.vectors_per_doc[k], k))
        assert d.selected.position == smallest
        assert d.evidence_digest == labels_evidence.digest()
        assert d.n_queries == labels_evidence.n_queries

    def test_agreement_evidence_selects_under_its_own_construct(self, space, agreement_evidence):
        d = select(space, agreement_evidence, target=0.9, error=0.2)
        assert d.outcome is Outcome.SELECT and d.construct == "agreement@1"

    def test_target_one_with_few_queries_abstains(self, space):
        corpus, queries, qrels = make_problem(n_queries=14, n_unlabelled=0)
        ev = collect_evidence(corpus, queries, space, LabelReference(qrels))
        d = select(space, ev, target=1.0, error=0.05)
        assert d.outcome is Outcome.ABSTAIN and not d.certified and d.certified_positions == ()

    def test_the_evidence_round_trips_through_its_file(self, labels_evidence, tmp_path):
        path = labels_evidence.to_npz(tmp_path / "e.npz")
        assert Evidence.from_npz(path) == labels_evidence


# ----------------------------------------------------------- the registered default space


@pytest.mark.skipif(not HAVE_SCIPY, reason="the registered default space needs SciPy")
class TestDefaultSpace:
    @pytest.fixture(scope="class")
    def default_space(self, problem):
        return CandidateSpace.default(problem[0].dimension)

    @pytest.fixture(scope="class")
    def default_evidence(self, problem, default_space):
        corpus, queries, qrels = problem
        return collect_evidence(corpus, queries, default_space, LabelReference(qrels))

    def test_it_is_the_registered_space(self, default_space):
        assert default_space.is_registered_default and len(default_space) == 40
        assert default_space.n_families == 7

    def test_evidence_equals_an_independent_recomputation(self, problem, default_space,
                                                         default_evidence):
        corpus, queries, qrels = problem
        ref = independent(corpus, queries, default_space,
                          lambda base: relevant_from_qrels(qrels, queries.ids, corpus.ids))
        ev = default_evidence
        assert np.array_equal(ev.baseline, ref["baseline"])
        assert np.array_equal(ev.candidates, ref["candidates"])
        assert np.array_equal(ev.bytes_per_doc, ref["bytes"])
        assert np.array_equal(ev.vectors_per_doc, ref["vectors"])

    def test_compress_selected_reproduces_every_recorded_fit(self, problem, default_space,
                                                             default_evidence):
        corpus, _, _ = problem
        for spec in default_space.candidates:
            _, compressed = compress_selected(
                corpus, spec, expected_fitted=default_evidence.fitted[spec.position])
            assert compressed.report()["bytes_per_doc"] == \
                default_evidence.bytes_per_doc[spec.position]

    def test_selection_runs_end_to_end_on_the_registered_space(self, default_space,
                                                              default_evidence):
        d = select(default_space, default_evidence, target=0.5, error=0.2)
        assert d.outcome is Outcome.SELECT
        assert d.level == pytest.approx(0.2 / 7)
        assert len(d.p_values) == 40
