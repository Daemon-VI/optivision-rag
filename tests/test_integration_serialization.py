"""Save and load preserve meaning: corpora, compressed corpora (the v1 file) and evidence.

For each object the check is the same: what comes back has the same identity (the digests of
``core.identity``) and gives the same rankings as what went in, bit for bit, and a file that
cannot be read back faithfully is refused at save time or at load time, never accepted.

Synthetic data only; nothing here is a research experiment. The artifact bundle written by
``certify(...).save`` has its own round-trip tests in ``test_integration_end_to_end.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from optivision.compose import Pipeline, load_compressed, save_compressed
from optivision.core.evaluate import (
    Evidence,
    EvidenceFormatError,
    LabelReference,
    RankingReference,
)
from optivision.core.evaluate.collect import collect_evidence
from optivision.core.retrieve import CompressedRetriever, ExactRetriever
from optivision.core.selection import Decision, select
from optivision.core.similarity import maxsim_matrix
from optivision.core.stages import (
    fitted_state,
    fitted_state_digest,
    pipeline_fitted_identity,
    pipeline_fitted_state,
    restore_fitted_state,
    restore_pipeline_state,
)
from optivision.core.vectors import corpus_identity
from optivision.representation import MultiVectorCorpus
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    Float32Quantizer,
    HierarchicalMerge,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RedundancyPruner,
    TruncateProjector,
)
from optivision.stages.base import stage_from_dict

from .test_integration_layers import (
    DTYPE_IDS,
    DTYPES,
    bitwise_equal,
    build_corpus,
    build_queries,
    make_pipeline,
    needs_scipy,
    pipeline_params,
    small_space,
)


def rich_corpus(seed: int, dtype: object, dim: int = 13) -> MultiVectorCorpus:
    """A corpus with every optional field: protected rows, positions (NaN for off-grid rows),
    weights, importance, metadata and attributes, an empty document and unicode ids."""
    base = build_corpus(seed, dim, dtype)
    rng = np.random.default_rng(seed + 50)
    n = base.num_vectors
    positions = rng.random((n, 2)).astype(np.float32)
    positions[::5] = np.nan
    protected = np.zeros(n, dtype=bool)
    protected[::7] = True
    ids = [f"{doc_id}-é中" if i % 4 == 0 else doc_id for i, doc_id in enumerate(base.ids)]
    return MultiVectorCorpus(
        base.vectors, base.offsets, ids=ids,
        metadata=[{"page": i, "tag": f"t{i}"} for i in range(len(base))],
        protected=protected, positions=positions,
        weights=(rng.random(n) + 0.5).astype(np.float32),
        importance=rng.random(n).astype(np.float32),
        attrs={"source": "synthetic", "n_pages": len(base), "ratio": 0.5})


class TestCorpusFiles:
    @pytest.mark.parametrize("how", ["plain", "compressed", "mmap"])
    @pytest.mark.parametrize("dtype", [*DTYPES, np.float64], ids=[*DTYPE_IDS, "float64"])
    @pytest.mark.parametrize("seed", (0, 1))
    def test_a_saved_corpus_comes_back_with_the_same_identity_and_rankings(
            self, tmp_path, how, dtype, seed):
        corpus = rich_corpus(seed, dtype)
        path = corpus.save(tmp_path / "corpus", compressed=how == "compressed")
        assert path.suffix == ".npz" and path.is_file()
        loaded = MultiVectorCorpus.load(path, mmap=how == "mmap")
        assert corpus_identity(loaded) == corpus_identity(corpus)
        assert corpus_identity(loaded).digest() == corpus_identity(corpus).digest()
        assert loaded.dtype == corpus.dtype and loaded.ids == corpus.ids
        assert loaded.metadata == corpus.metadata and loaded.attrs == corpus.attrs
        for field in ("protected", "positions", "weights", "importance"):
            a, b = getattr(corpus, field), getattr(loaded, field)
            assert bitwise_equal(np.asarray(a), np.asarray(b)), field  # NaN bits included
        queries, _ = build_queries(corpus, seed)
        assert ExactRetriever(loaded).search(queries, 10) == ExactRetriever(corpus).search(queries, 10)
        assert bitwise_equal(maxsim_matrix(queries, loaded), maxsim_matrix(queries, corpus))

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_compressing_a_loaded_corpus_gives_the_codes_of_compressing_the_original(
            self, tmp_path, dtype):
        """The aligned arrays survive the file with their meaning: a token reducer that treats
        protected rows verbatim and averages weights gives the same output on both."""
        corpus = rich_corpus(3, dtype)
        loaded = MultiVectorCorpus.load(corpus.save(tmp_path / "c.npz"))
        for make in (lambda: Pipeline([AdaptiveMerge(radius=0.5), Int8Quantizer("per_vector")]),
                     lambda: Pipeline([RedundancyPruner(0.6), BinaryQuantizer()]),
                     lambda: Pipeline([PCAProjector(8), Int4Quantizer()]),
                     lambda: Pipeline([])):
            a, b = make().compress(corpus), make().compress(loaded)
            assert bitwise_equal(a.codes, b.codes) and bitwise_equal(a.offsets, b.offsets)
            assert a.ids == b.ids and a.stats == b.stats

    def test_the_file_of_one_corpus_is_the_same_every_time(self, tmp_path):
        corpus = rich_corpus(2, np.float32)
        first = corpus.save(tmp_path / "a.npz").read_bytes()
        again = corpus.save(tmp_path / "b.npz").read_bytes()
        assert first == again

    @pytest.mark.parametrize("damage", ["missing-header", "other-format", "newer-version"])
    def test_a_file_that_is_not_a_corpus_is_refused_by_name(self, tmp_path, damage):
        corpus = build_corpus(0, 13)
        path = corpus.save(tmp_path / "c.npz")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        header = json.loads(bytes(arrays["header"]).decode("utf-8"))
        if damage == "missing-header":
            del arrays["header"]
        elif damage == "other-format":
            header["format"] = "something-else"
        else:
            header["version"] = 99
        if damage != "missing-header":
            arrays["header"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
        np.savez(tmp_path / "bad.npz", **arrays)
        with pytest.raises(ValueError, match="format|header|version|newer"):
            MultiVectorCorpus.load(tmp_path / "bad.npz")


# codecs the v1 compressed file can hold (no fitted state of its own) with the stages before them
V1_PIPELINES = {
    "float32": lambda: Pipeline([]),
    "float32-explicit": lambda: Pipeline([Float32Quantizer()]),
    "float16": lambda: Pipeline([Float16Quantizer()]),
    "binary": lambda: Pipeline([BinaryQuantizer()]),
    "int8-fixed": lambda: Pipeline([Int8Quantizer("fixed")]),
    "int8-per-vector": lambda: Pipeline([Int8Quantizer("per_vector")]),
    "int4": lambda: Pipeline([Int4Quantizer()]),
    "merge+int8": lambda: Pipeline([AdaptiveMerge(radius=0.5), Int8Quantizer("per_vector")]),
    "redundancy+binary": lambda: Pipeline([RedundancyPruner(0.6), BinaryQuantizer()]),
    "truncate+int4": lambda: Pipeline([TruncateProjector(9), Int4Quantizer()]),
    "ward+float16": pytest.param(
        lambda: Pipeline([HierarchicalMerge(ratio=0.5), Float16Quantizer()]), marks=needs_scipy),
}

#: codecs that carry fitted state: ``save_compressed`` refuses them instead of writing a file
#: that cannot be decoded
V1_REFUSED = {
    "lloyd2": lambda: Pipeline([Lloyd2Quantizer(seed=7)]),
    "int8-per-dimension": lambda: Pipeline([Int8Quantizer("per_dimension")]),
    "binary-centred": lambda: Pipeline([BinaryQuantizer(center="mean")]),
    "int4-centred": lambda: Pipeline([Int4Quantizer(center="mean")]),
    "int8-centred": lambda: Pipeline([Int8Quantizer("per_vector", center="mean")]),
}


def v1_params():
    return [pytest.param(name, id=name, marks=getattr(make, "marks", ()))
            for name, make in V1_PIPELINES.items()]


def build_v1(name: str) -> Pipeline:
    entry = V1_PIPELINES[name]
    return (entry.values[0] if hasattr(entry, "values") else entry)()


class TestCompressedFilesV1:
    @pytest.mark.filterwarnings("ignore:Int8Quantizer:UserWarning")  # the fixed scale clips a little
    @pytest.mark.parametrize("name", v1_params())
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_a_saved_compressed_corpus_decodes_and_ranks_like_the_original(
            self, tmp_path, name, dtype):
        corpus = rich_corpus(1, dtype)
        queries, _ = build_queries(corpus, 1)
        pipeline = build_v1(name)
        compressed = pipeline.compress(corpus)
        path = save_compressed(compressed, tmp_path / "store.npz")
        loaded = load_compressed(path)
        assert bitwise_equal(loaded.codes, compressed.codes)
        assert bitwise_equal(loaded.offsets, compressed.offsets)
        assert loaded.dim == compressed.dim and loaded.ids == compressed.ids
        assert loaded.pipeline == compressed.pipeline == pipeline.to_dict()
        assert loaded.stats == compressed.stats and loaded.nbytes == compressed.nbytes
        assert loaded.quantizer.to_dict() == compressed.quantizer.to_dict()
        assert bitwise_equal(loaded.decode_rows(0, loaded.num_vectors),
                             compressed.decode_rows(0, compressed.num_vectors))
        # the file stores the description but not fitted state: a pipeline rebuilt from it
        # projects queries as the original did only when no projection has to be fitted
        rebuilt = Pipeline.from_dict(loaded.pipeline)
        for retriever in (CompressedRetriever(loaded, rebuilt),
                          CompressedRetriever(compressed, pipeline)):
            assert retriever.search(queries, 8) == CompressedRetriever(
                compressed, pipeline).search(queries, 8)
        assert bitwise_equal(
            maxsim_matrix(rebuilt.transform_queries(queries), loaded),
            maxsim_matrix(pipeline.transform_queries(queries), compressed))

    @pytest.mark.parametrize("name", sorted(V1_REFUSED))
    def test_a_codec_with_fitted_state_is_refused_and_no_file_is_written(self, tmp_path, name):
        compressed = V1_REFUSED[name]().compress(build_corpus(0, 13))
        with pytest.raises(NotImplementedError, match="not supported"):
            save_compressed(compressed, tmp_path / "store.npz")
        assert not (tmp_path / "store.npz").exists()

    def test_the_v1_file_does_not_hold_stage_state_so_a_projection_cannot_be_restored(
            self, tmp_path):
        """Why the artifact bundle exists (design decision D13): a PCA basis is not in the v1
        file, so a pipeline rebuilt from it is unfitted and refuses to project queries."""
        corpus = build_corpus(0, 13)
        pipeline = Pipeline([PCAProjector(8), Int4Quantizer()])
        compressed = pipeline.compress(corpus)
        loaded = load_compressed(save_compressed(compressed, tmp_path / "pca.npz"))
        rebuilt = Pipeline.from_dict(loaded.pipeline)
        assert not rebuilt.fitted
        queries, _ = build_queries(corpus, 0)
        with pytest.raises(RuntimeError, match="fit"):
            rebuilt.transform_queries(queries)

    def test_a_file_that_is_not_a_compressed_corpus_is_refused(self, tmp_path):
        path = build_corpus(0, 13).save(tmp_path / "corpus.npz")
        with pytest.raises(ValueError, match="not a compressed"):
            load_compressed(path)


class TestEvidenceFiles:
    @pytest.fixture(scope="class")
    def evidence_and_space(self):
        corpus = build_corpus(5, 21)
        queries, qrels = build_queries(corpus, 5, n=12)
        space = small_space()
        return {"labels": collect_evidence(corpus, queries, space, LabelReference(qrels)),
                "agreement": collect_evidence(corpus, queries, space, RankingReference())}, space

    @pytest.mark.parametrize("construct", ["labels", "agreement"])
    def test_a_loaded_evidence_has_the_same_identity_and_gives_the_same_decisions(
            self, tmp_path, evidence_and_space, construct):
        evidences, space = evidence_and_space
        evidence = evidences[construct]
        loaded = Evidence.from_npz(evidence.to_npz(tmp_path / "evidence"))
        assert loaded == evidence and loaded.digest() == evidence.digest()
        for field in ("baseline", "candidates", "bytes_per_doc", "vectors_per_doc"):
            assert bitwise_equal(getattr(loaded, field), getattr(evidence, field)), field
        assert loaded.query_ids == evidence.query_ids
        assert loaded.dropped_queries == evidence.dropped_queries
        assert loaded.fitted == evidence.fitted and loaded.construct == evidence.construct
        for target in (0.3, 0.6, 0.9):
            for error in (0.05, 0.2):
                a = select(space, evidence, target=target, error=error)
                b = select(space, loaded, target=target, error=error)
                assert a.to_dict() == b.to_dict()
                assert Decision.from_dict(json.loads(json.dumps(a.to_dict()))) == a

    def test_an_altered_file_is_refused_not_read(self, tmp_path, evidence_and_space):
        evidence = evidence_and_space[0]["labels"]
        path = evidence.to_npz(tmp_path / "evidence")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["candidates"] = np.array(arrays["candidates"], copy=True)
        arrays["candidates"][0, 0] = np.nextafter(arrays["candidates"][0, 0], 2.0)  # one ulp
        np.savez(tmp_path / "altered.npz", **arrays)
        with pytest.raises(EvidenceFormatError, match="digest"):
            Evidence.from_npz(tmp_path / "altered.npz")
        del arrays["baseline"]
        np.savez(tmp_path / "missing.npz", **arrays)
        with pytest.raises(EvidenceFormatError, match="missing"):
            Evidence.from_npz(tmp_path / "missing.npz")

    def test_the_file_of_one_evidence_is_the_same_every_time(self, tmp_path, evidence_and_space):
        evidence = evidence_and_space[0]["labels"]
        first = evidence.to_npz(tmp_path / "a").read_bytes()
        again = evidence.to_npz(tmp_path / "b").read_bytes()
        assert first == again


class TestFittedStateRestores:
    """What a bundle stores (``fitted_state``) is enough to rebuild the fitted pipeline: the
    restored stages give the same codes, the same query projection and the same digests."""

    @pytest.mark.parametrize("name", pipeline_params())
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_a_pipeline_rebuilt_from_its_description_and_its_state_behaves_as_fitted(
            self, name, dtype):
        corpus = build_corpus(2, 21, dtype)
        queries, _ = build_queries(corpus, 2)
        fitted = make_pipeline(name)
        compressed = fitted.compress(corpus)
        states = pipeline_fitted_state(fitted)
        digests = pipeline_fitted_identity(fitted)
        fresh = Pipeline.from_dict(fitted.to_dict())
        assert not fresh.fitted or all(s is None for s in digests)  # nothing was carried over
        restore_pipeline_state(fresh, states, expected_digests=digests)
        assert fresh.fitted and pipeline_fitted_identity(fresh) == digests
        again = fresh.compress(corpus, fit=False)  # no refit: the restored state is used
        assert bitwise_equal(again.codes, compressed.codes)
        assert bitwise_equal(again.offsets, compressed.offsets)
        assert bitwise_equal(fresh.transform_queries(queries).vectors,
                             fitted.transform_queries(queries).vectors)
        assert bitwise_equal(maxsim_matrix(fresh.transform_queries(queries), again),
                             maxsim_matrix(fitted.transform_queries(queries), compressed))
        wrong = list(digests)
        stateful = [i for i, d in enumerate(wrong) if d is not None]
        if stateful:  # a digest that does not match what is restored is refused, stage named
            wrong[stateful[0]] = "0" * 64
            with pytest.raises(ValueError, match="digest"):
                restore_pipeline_state(Pipeline.from_dict(fitted.to_dict()), states,
                                       expected_digests=wrong)

    def test_a_boolean_is_not_a_number_in_a_restored_state(self):
        """A state that says ``seed=True`` is not a state that says ``seed=1``."""
        corpus = build_corpus(0, 13)
        stage = Lloyd2Quantizer(seed=1)
        Pipeline([stage]).fit(corpus)
        state = fitted_state(stage)
        for key in ("seed", "sigma"):
            fresh = stage_from_dict(stage.to_dict())
            with pytest.raises(TypeError, match="number"):
                restore_fitted_state(fresh, {**state, key: True})
        fresh = stage_from_dict(stage.to_dict())
        restore_fitted_state(fresh, state, expected_digest=fitted_state_digest(stage))
        assert fitted_state_digest(fresh) == fitted_state_digest(stage)
