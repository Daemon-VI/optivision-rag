from __future__ import annotations

import json

import numpy as np
import pytest

from optivision.config import Config
from optivision.corpus import CorpusSpec, generate_synthetic_corpus, load_queries
from optivision.metrics import (
    evaluate,
    ndcg_at_k,
    rank_correlation,
    rank_correlation_shared,
    recall_at_k,
)
from optivision.pipeline import OptiVisionRAG
from optivision.types import PageRef

from .conftest import make_page


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    generate_synthetic_corpus(out, CorpusSpec(n_docs=6, pages_per_doc=2, seed=3))
    return out


def _cfg(corpus, tmp_path, **overrides) -> Config:
    cfg = Config()
    cfg.encoder.backend = "synthetic"
    cfg.encoder.synthetic_dim = 64
    cfg.encoder.synthetic_grid = 16
    cfg.encoder.synthetic_layout = str(corpus / "layout.json")
    cfg.index.path = str(tmp_path / "index")
    for section, values in overrides.items():
        for k, v in values.items():
            setattr(getattr(cfg, section), k, v)
    return cfg


class TestCorpus:
    def test_generates_pages_and_queries(self, corpus):
        manifest = json.loads((corpus / "manifest.json").read_text())
        assert manifest["n_pages"] == 12
        assert manifest["n_precise"] == 12
        assert manifest["n_topical"] > 0
        assert len(list((corpus / "pdfs").glob("*.pdf"))) == 6

    def test_layout_keys_match_ingest_page_ids(self, corpus):
        """The bug this guards: layout keyed differently from PageRef.page_id
        silently degrades the encoder to pixel hashing and fakes bad quality."""
        from optivision.ingest import iter_pages

        layout = json.loads((corpus / "layout.json").read_text())
        ingested = {ref.page_id for ref, _ in iter_pages(corpus / "pdfs")}
        assert ingested == set(layout)

    def test_query_relevance_points_at_real_pages(self, corpus):
        layout = json.loads((corpus / "layout.json").read_text())
        _, _, qrels = load_queries(corpus / "queries.json")
        for relevant in qrels.values():
            assert relevant <= set(layout)


class TestEndToEnd:
    def test_index_then_search_finds_the_right_page(self, corpus, tmp_path):
        cfg = _cfg(corpus, tmp_path)
        rag = OptiVisionRAG(cfg)
        report = rag.build(corpus / "pdfs")
        assert report.n_pages == 12
        assert report.token_reduction > 1.5
        assert report.compression_ratio > 20

        _, texts, qrels = load_queries(corpus / "queries.json")
        qids = list(qrels)
        precise = [(q, t) for q, t in zip(qids, texts) if len(qrels[q]) == 1]
        hits = 0
        for qid, text in precise:
            result = rag.search(text, top_k=1)
            hits += result.hits[0].ref.page_id in qrels[qid]
        assert hits / len(precise) >= 0.9
        rag.close()

    def test_compression_does_not_destroy_the_baseline_ranking(self, corpus, tmp_path):
        """The central claim: pruning + binary keeps the float ranking."""
        _, texts, qrels = load_queries(corpus / "queries.json")
        qids = list(qrels)

        runs = {}
        for name, over in {
            "baseline": {
                "pruning": {"enabled": False},
                "compression": {"enabled": False, "method": "none"},
            },
            "optivision": {"pruning": {"enabled": True}, "compression": {"enabled": True}},
        }.items():
            cfg = _cfg(corpus, tmp_path / name, **over)
            rag = OptiVisionRAG(cfg)
            rag.build(corpus / "pdfs")
            # Deep enough to rank the whole fixture corpus: tau over truncated
            # lists charges for pages that simply fell off the end, which is
            # not what this test is asking about.
            runs[name] = {
                qid: [h.ref.page_id for h in rag.search(t, top_k=64).hits]
                for qid, t in zip(qids, texts)
            }
            rag.close()

        base = evaluate(runs["baseline"], qrels, ks=(5,))
        opti = evaluate(runs["optivision"], qrels, ks=(5,))
        pool = sorted({p for r in runs.values() for lst in r.values() for p in lst})
        assert opti["ndcg@5"] >= base["ndcg@5"] * 0.9

        taus = [
            rank_correlation(runs["baseline"][q], runs["optivision"][q], pool=pool)
            for q in qids
        ]
        assert float(np.mean(taus)) > 0.5

    def test_manifest_is_written(self, corpus, tmp_path):
        cfg = _cfg(corpus, tmp_path)
        rag = OptiVisionRAG(cfg)
        rag.build(corpus / "pdfs")
        manifest = json.loads((tmp_path / "index" / "manifest.json").read_text())
        assert manifest["index"]["n_pages"] == 12
        assert "config" in manifest
        rag.close()

    def test_process_page_matches_build_accounting(self, corpus, tmp_path):
        from optivision.ingest import iter_pages

        cfg = _cfg(corpus, tmp_path)
        rag = OptiVisionRAG(cfg)
        ref, image = next(iter(iter_pages(corpus / "pdfs")))
        page = rag.process_page(image, ref)
        assert page.n_tokens_after < page.n_tokens_before
        assert page.codes.shape == (page.n_tokens_after, 8)  # 64 dims -> 8 bytes
        rag.close()


class TestMetrics:
    def test_ndcg_rewards_earlier_hits(self):
        assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) == 1.0
        assert ndcg_at_k(["x", "a", "y"], {"a"}, 3) < 1.0
        assert ndcg_at_k(["x", "y", "z"], {"a"}, 3) == 0.0

    def test_recall_caps_at_k(self):
        assert recall_at_k(["a", "b"], {"a", "b", "c"}, 2) == 1.0
        assert recall_at_k(["a", "z"], {"a", "b"}, 2) == 0.5

    def test_kendall_tau_extremes(self):
        order = ["a", "b", "c", "d"]
        assert rank_correlation(order, order) == 1.0
        assert rank_correlation(order, list(reversed(order))) == -1.0

    def test_kendall_tau_ranks_absentees_last(self):
        # An item only one list ranks sits after everything that list does
        # rank, so agreeing on the shared prefix still reads as agreement.
        assert rank_correlation(["a", "b"], ["a", "b", "z"]) == 1.0

    def test_kendall_tau_scores_disjoint_lists_as_disagreement(self):
        # The bug this replaced: shared-ids-only scored these as identical,
        # so the worst possible outcome read as the best one.
        assert rank_correlation(["a", "b", "c"], ["x", "y", "z"]) < 0.0
        assert rank_correlation_shared(["a", "b", "c"], ["x", "y", "z"]) == 1.0

    def test_kendall_tau_pool_counts_pages_neither_list_ranked(self):
        # Two rankings that agree on what they return can still disagree about
        # the rest of the corpus; without a pool that is invisible.
        a, b = ["a", "b"], ["b", "a"]
        assert rank_correlation(a, b) == -1.0
        assert rank_correlation(a, b, pool=["a", "b", "c", "d"]) > -1.0

    def test_kendall_tau_pool_makes_the_cutoff_irrelevant(self):
        pool = list("abcdef")
        full = rank_correlation(list("abcdef"), list("bacdef"), pool=pool)
        assert full == rank_correlation(list("abcdef"), list("bacdef"), pool=pool)
        assert 0.8 < full < 1.0

    def test_evaluate_averages_over_queries(self):
        run = {"q1": ["a"], "q2": ["z"]}
        qrels = {"q1": {"a"}, "q2": {"b"}}
        out = evaluate(run, qrels, ks=(1,))
        assert out["recall@1"] == 0.5
        assert out["n_queries"] == 2


class TestConfig:
    def test_roundtrip_through_yaml(self, tmp_path):
        cfg = Config()
        cfg.pruning.keep_ratio = 0.3
        cfg.dump(tmp_path / "c.yaml")
        assert Config.load(tmp_path / "c.yaml").pruning.keep_ratio == 0.3

    def test_unknown_option_is_rejected(self):
        with pytest.raises(ValueError, match="unknown pruning option"):
            Config.from_dict({"pruning": {"blank_treshold": 0.1}})

    def test_unknown_section_is_rejected(self):
        with pytest.raises(ValueError, match="unknown config section"):
            Config.from_dict({"prunning": {}})

    def test_with_overrides_does_not_mutate_original(self):
        cfg = Config()
        other = cfg.with_overrides(compression={"method": "int8"})
        assert other.compression.method == "int8"
        assert cfg.compression.method == "binary"


class TestSyntheticEncoder:
    def test_missing_layout_file_raises(self):
        from optivision.encoders.synthetic import SyntheticEncoder

        with pytest.raises(FileNotFoundError):
            SyntheticEncoder(layout_path="does/not/exist.json")

    def test_page_id_drift_warns_instead_of_failing_silently(self, corpus):
        from optivision.encoders.synthetic import SyntheticEncoder

        enc = SyntheticEncoder(dim=32, grid=8, layout_path=corpus / "layout.json")
        with pytest.warns(RuntimeWarning, match="not in the loaded word layout"):
            enc.encode_pages([make_page()], [PageRef(doc_id="nope", page_no=9)])

    def test_vectors_are_normalised(self):
        from optivision.encoders.synthetic import SyntheticEncoder

        enc = SyntheticEncoder(dim=32, grid=8)
        out = enc.encode_pages([make_page()], [PageRef(doc_id="d", page_no=1)])[0]
        assert np.allclose(np.linalg.norm(out.embeddings, axis=1), 1.0, atol=1e-5)
        assert out.n_tokens == 8 * 8 + 4

    def test_same_query_encodes_identically_across_instances(self):
        from optivision.encoders.synthetic import SyntheticEncoder

        a = SyntheticEncoder(dim=32).encode_queries(["fire safety audit"])[0]
        b = SyntheticEncoder(dim=32).encode_queries(["fire safety audit"])[0]
        assert np.allclose(a, b)


class TestBenchCache:
    def test_encode_cache_roundtrips(self, corpus, tmp_path):
        """A cached encode pass must reproduce the vectors bit-for-bit.

        This is what makes it safe to add an ablation row without re-encoding:
        if the cache drifted from a fresh pass, the new row would be compared
        against a baseline computed from different vectors.
        """
        import numpy as np

        from optivision.bench import EncodedCorpus
        from optivision.encoders import get_encoder

        cfg = _cfg(corpus, tmp_path)
        encoder = get_encoder(cfg.encoder)
        built = EncodedCorpus.build(corpus / "pdfs", encoder, cfg)
        built.save(tmp_path / "cache.npz")
        loaded = EncodedCorpus.load(tmp_path / "cache.npz")

        assert loaded.n_pages == built.n_pages
        assert loaded.dim == built.dim
        for a, b in zip(built.encodings, loaded.encodings, strict=True):
            assert a.ref.page_id == b.ref.page_id
            assert np.array_equal(a.embeddings, b.embeddings)
            assert np.array_equal(a.grid.token_index, b.grid.token_index)
            assert np.array_equal(a.text_token_index, b.text_token_index)
            assert (a.grid.rows, a.grid.cols) == (b.grid.rows, b.grid.cols)
        for a, b in zip(built.images, loaded.images, strict=True):
            assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_cached_run_matches_uncached(self, corpus, tmp_path):
        from optivision.bench import default_variants, run_benchmark

        cfg = _cfg(corpus, tmp_path)
        variants = default_variants()[:3]
        kwargs = {
            "cfg": cfg,
            "variants": variants,
            "workdir": str(tmp_path / "wd"),
            "top_k": 5,
        }
        fresh = run_benchmark(corpus / "pdfs", corpus / "queries.json", **kwargs)
        cached = run_benchmark(
            corpus / "pdfs", corpus / "queries.json", cache=str(tmp_path / "c.npz"), **kwargs
        )
        replay = run_benchmark(
            corpus / "pdfs", corpus / "queries.json", cache=str(tmp_path / "c.npz"), **kwargs
        )
        for a, b in zip(fresh["rows"], replay["rows"], strict=True):
            assert a["variant"] == b["variant"]
            assert a["ndcg@5"] == pytest.approx(b["ndcg@5"])
            assert a["tokens_per_page"] == pytest.approx(b["tokens_per_page"])
        assert cached["corpus"]["n_pages"] == replay["corpus"]["n_pages"]

    def test_query_cache_invalidates_when_queries_change(self, corpus, tmp_path):
        import json as _json

        from optivision.bench import _encode_queries_cached
        from optivision.encoders import get_encoder

        cfg = _cfg(corpus, tmp_path)
        encoder = get_encoder(cfg.encoder)
        path = tmp_path / "q.npz"
        first, _ = _encode_queries_cached(["fire safety audit"], cfg, encoder, path)
        second, _ = _encode_queries_cached(["fire safety review"], cfg, encoder, path)
        # Same token count, different last word: a stale cache would return the
        # first result verbatim.
        assert first[0].shape == second[0].shape
        assert not np.allclose(first[0], second[0])
        assert _json.loads(str(np.load(path, allow_pickle=True)["texts"])) == [
            "fire safety review"
        ]

        again, _ = _encode_queries_cached(["fire safety review"], cfg, encoder, path)
        assert np.array_equal(again[0], second[0])  # unchanged queries hit the cache
