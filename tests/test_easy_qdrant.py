"""Qdrant export (``optivision.easy.integrations.qdrant``): the test that gates ``qdrant_export``.

Gate (presets/easy_ledger.json, FINAL_DESIGN 14): top-10 equality 100% between Qdrant's MaxSim over the
exported collection and the index's own exact search, here on synthetic corpora in qdrant-client's local
mode (``QdrantClient(":memory:")`` and a local folder). Local mode stores float32 and scans every point,
so this checks what the export writes and how Qdrant's MaxSim/DOT scores it; a server's ``exact=True``
scan does the same arithmetic in float32. The float16 option is not covered by the gate: a server rounds
query numbers to float16 there (see the module docstring), which local mode does not reproduce.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient

from optivision.easy import _ledger
from optivision.easy._types import FeatureOff
from optivision.easy.integrations import qdrant as Q
from optivision.representation import MultiVectorCorpus
from optivision.testing.synthetic import make_corpus, make_queries

from ._easy_index_fakes import (
    FakeIndex,
    FakeStore,
    evidence_block,
    layout_row,
    page_ids,
    segmented,
    with_ids,
)

HM25 = {"stage": "hierarchical_merge", "ratio": 0.25, "max_distance": None, "min_vectors": 1}
HM33 = {"stage": "hierarchical_merge", "ratio": 0.33, "max_distance": None, "min_vectors": 1}

#: every codec of the ladder, plain and merged (catalogue specs, presets/measured.json)
LAYOUTS = {
    "float32": {"stages": []},
    "float16": {"stages": [{"stage": "float16"}]},
    "int8(per_vector)": {"stages": [{"stage": "int8", "scale": "per_vector"}]},
    "int4": {"stages": [{"stage": "int4"}]},
    "int4(mean)": {"stages": [{"stage": "int4", "center": "mean"}]},
    "lloyd2(7)": {"stages": [{"stage": "lloyd2", "seed": 7}]},
    "binary": {"stages": [{"stage": "binary"}]},
    "hm(0.25) > float16": {"stages": [HM25, {"stage": "float16"}]},
    "hm(0.25) > int8(per_vector)": {"stages": [HM25, {"stage": "int8", "scale": "per_vector"}]},
    "hm(0.33) > int4(mean)": {"stages": [HM33, {"stage": "int4", "center": "mean"}]},
    "hm(0.25) > binary": {"stages": [HM25, {"stage": "binary"}]},
}
FLOAT16_EXACT = ("float16", "binary", "hm(0.25) > float16", "hm(0.25) > binary")

N_PAGES, DIM, K = 240, 64, 10
SIZES = (100, 100, 40)


@pytest.fixture(scope="module")
def data() -> tuple[MultiVectorCorpus, MultiVectorCorpus]:
    raw = make_corpus(N_PAGES, DIM, tokens=(8, 40), seed=11, anisotropy=1.0, mean_shift=0.3)
    corpus = with_ids(raw, page_ids(N_PAGES))
    queries, _ = make_queries(corpus, 48, tokens=(4, 12), noise=0.8, seed=12)
    return corpus, queries


def _index(corpus: MultiVectorCorpus, name: str) -> FakeIndex:
    segments, pipeline = segmented(corpus, LAYOUTS[name], SIZES)
    return FakeIndex(FakeStore(segments, pipeline), layout=layout_row(LAYOUTS[name], name=name),
                     evidence=evidence_block())


def _own_top(index: FakeIndex, queries: MultiVectorCorpus, k: int = K) -> list[list[str]]:
    return [[h.id for h in row] for row in index.search_vectors(queries, k)]


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_top10_equals_the_index_own_exact_search(data, name):
    """The gate: every query's top-10 list is identical (ids and order), on every ladder codec."""
    corpus, queries = data
    index = _index(corpus, name)
    client = QdrantClient(":memory:")
    report = Q._export(index, client, "pages")
    assert report.n_points == N_PAGES and report.datatype == "float32"
    got = Q.search(client, "pages", queries, k=K)
    own = index.search_vectors(queries, K)
    assert [[d for d, _ in row] for row in got] == [[h.id for h in row] for row in own]
    # the scores are the same MaxSim inner products, to float32 summation order
    for row_q, row_o in zip(got, own, strict=True):
        np.testing.assert_allclose([s for _, s in row_q], [h.score for h in row_o], rtol=2e-5, atol=2e-5)


def test_a_local_folder_collection_gives_the_same_lists(data, tmp_path):
    corpus, queries = data
    index = _index(corpus, "hm(0.25) > int8(per_vector)")
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    Q._export(index, client, "pages")
    assert [[d for d, _ in row] for row in Q.search(client, "pages", queries, k=K)] == _own_top(index, queries)
    client.close()


@pytest.mark.parametrize("name", FLOAT16_EXACT)
def test_float16_storage_is_allowed_where_every_stored_number_is_a_float16(data, name):
    corpus, queries = data
    index = _index(corpus, name)
    client = QdrantClient(":memory:")
    report = Q._export(index, client, "half", datatype="float16")
    assert report.datatype == "float16" and "rounds query numbers" in report.exact
    assert report.bytes_sent == report.n_vectors * DIM * 2
    # local mode does not round queries, so here the lists still match
    assert [[d for d, _ in row] for row in Q.search(client, "half", queries, k=K)] == _own_top(index, queries)


@pytest.mark.parametrize("name", ["int8(per_vector)", "int4(mean)", "lloyd2(7)", "float32"])
def test_float16_storage_is_refused_where_it_would_round_stored_numbers(data, name):
    corpus, _ = data
    with pytest.raises(Q.ExportRefused, match="not float16 numbers"):
        Q._export(_index(corpus, name), QdrantClient(":memory:"), "half", datatype="float16")


def test_uint8_and_unknown_datatypes_are_refused(data):
    corpus, _ = data
    index = _index(corpus, "binary")
    with pytest.raises(Q.ExportRefused, match="uint8"):
        Q.plan(index, "uint8")
    with pytest.raises(Q.ExportRefused, match="datatype must be"):
        Q.plan(index, "bfloat16")


def test_a_projected_layout_is_refused(data):
    corpus, _ = data
    spec = {"stages": [HM25, {"stage": "project", "dim": 16, "method": "pca"}, {"stage": "int8", "scale": "per_vector"}]}
    segments, pipeline = segmented(corpus, spec, SIZES)
    index = FakeIndex(FakeStore(segments, pipeline))
    with pytest.raises(Q.ExportRefused, match="projection"):
        Q._export(index, QdrantClient(":memory:"), "p")


def test_empty_pages_are_skipped_and_listed():
    raw = make_corpus(30, 16, tokens=(3, 6), seed=5)
    counts = raw.counts.copy()
    counts[[4, 17]] = 0  # two pages without vectors
    keep = np.concatenate([np.arange(raw.offsets[i], raw.offsets[i] + counts[i]) for i in range(30)])
    corpus = MultiVectorCorpus(raw.vectors[keep], np.concatenate([[0], np.cumsum(counts)]), ids=page_ids(30))
    segments, pipeline = segmented(corpus, LAYOUTS["float16"], (30,))
    index = FakeIndex(FakeStore(segments, pipeline))
    client = QdrantClient(":memory:")
    report = Q._export(index, client, "e")
    assert report.skipped_empty == [corpus.ids[4], corpus.ids[17]]
    assert report.n_points == 28 and "Skipped" in str(report)
    queries, _ = make_queries(corpus, 6, tokens=(2, 3), seed=6)
    assert [[d for d, _ in row] for row in Q.search(client, "e", queries, k=10)] == _own_top(index, queries)


def test_payload_carries_the_page_and_the_evidence_and_ids_are_stable(data):
    corpus, _ = data
    index = _index(corpus, "binary")
    index.files = {corpus.ids[0]: "C:/data/my_pdfs/report.pdf"}
    client = QdrantClient(":memory:")
    Q._export(index, client, "pay")
    point = client.retrieve("pay", ids=[Q.point_id(corpus.ids[0])], with_payload=True)[0]
    assert point.payload["page_id"] == corpus.ids[0]
    assert point.payload["doc_id"] == "my_pdfs/report.pdf" and point.payload["page"] == 1
    assert point.payload["file"] == "C:/data/my_pdfs/report.pdf"
    meta = point.payload["optivision"]
    assert meta["optivision_layout"] == "binary" and meta["optivision_target"] == 0.97
    assert any("[our benchmarks: ViDoRe V1 DocVQA 500 pages]" in line for line in meta["optivision_evidence"])
    assert meta["optivision_status"] == _ledger.status()
    json.dumps(point.payload)
    assert Q.point_id(corpus.ids[0]) == Q.point_id(corpus.ids[0]) != Q.point_id(corpus.ids[1])


def test_payload_text_is_opt_in(data):
    corpus, _ = data
    segments, pipeline = segmented(corpus, LAYOUTS["float16"], SIZES)
    index = FakeIndex(FakeStore(segments, pipeline, texts={corpus.ids[3]: "Revenue rose in 2023."}))
    client = QdrantClient(":memory:")
    Q._export(index, client, "t")
    assert "text" not in client.retrieve("t", ids=[Q.point_id(corpus.ids[3])], with_payload=True)[0].payload
    Q._export(index, client, "t", recreate=True, payload_text=True)
    assert client.retrieve("t", ids=[Q.point_id(corpus.ids[3])], with_payload=True)[0].payload["text"] == (
        "Revenue rose in 2023.")


def test_an_existing_collection_needs_recreate(data):
    corpus, _ = data
    index = _index(corpus, "binary")
    client = QdrantClient(":memory:")
    Q._export(index, client, "c")
    with pytest.raises(Q.ExportRefused, match="recreate=True"):
        Q._export(index, client, "c")
    assert Q._export(index, client, "c", recreate=True).n_points == N_PAGES


def test_the_summary_tags_every_number_and_ends_with_next(data):
    corpus, _ = data
    report = Q._export(_index(corpus, "int8(per_vector)"), QdrantClient(":memory:"), "s")
    text = str(report)
    assert text.splitlines()[-1].startswith("Next: ")
    assert "[counted]" in text and text.isascii()
    assert report.bytes_sent == report.n_vectors * DIM * 4 > report.index_vector_bytes


def test_export_is_switched_off_until_the_gate_is_recorded(data):
    corpus, _ = data
    assert not _ledger.is_on("qdrant_export")
    with pytest.raises(FeatureOff, match="Qdrant export is switched off"):
        Q.export(_index(corpus, "binary"), ":memory:", "off")


def test_the_public_export_runs_once_switched_on(data, monkeypatch):
    corpus, queries = data
    monkeypatch.setattr(_ledger, "require", lambda name: None)  # this test only; the ledger is unchanged
    index = _index(corpus, "hm(0.25) > binary")
    client = QdrantClient(":memory:")
    Q.export(index, client, "on")
    assert [[d for d, _ in row] for row in Q.search(client, "on", queries, k=K)] == _own_top(index, queries)


def test_a_url_or_memory_target_builds_the_client(data):
    corpus, _ = data
    report = Q._export(_index(corpus, "binary"), ":memory:", "m")
    assert report.n_points == N_PAGES
