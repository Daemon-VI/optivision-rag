"""byaldi import (``optivision.easy.integrations.byaldi``): the format test that gates ``byaldi_import``.

``write_byaldi`` below writes a folder the way byaldi 0.0.x does (``byaldi/colpali.py``,
``ColPaliModel._export_index``: ``torch.save`` of 500-page lists of tensors, ``srsly.write_gzip_json``
of the config and maps, which is gzip-compressed JSON). The round trip checks that every page's vectors
come back bit for bit, with their document, page, file and metadata, in byaldi's embed-id order (files
sorted by number, so ``embeddings_1000.pt`` after ``embeddings_500.pt``), and that a search over the
import ranks pages as MaxSim over byaldi's own tensors does.
"""

from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optivision.core.retrieve.exact import ExactRetriever
from optivision.easy import _ledger
from optivision.easy._types import FeatureOff, MissingExtra
from optivision.easy.integrations import byaldi as B
from optivision.representation import MultiVectorCorpus

DIM = 32


def _gzip_json(path: Path, data: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)


def write_byaldi(root: Path, embeddings: list, embed_map: dict, files: dict | None, metadata: dict | None,
                 model_name: str = "vidore/colqwen2-v1.0", images: dict | None = None) -> Path:
    """The folder byaldi's ``_export_index`` writes (key types as JSON makes them: strings)."""
    root.mkdir(parents=True, exist_ok=True)
    emb = root / "embeddings"
    emb.mkdir(exist_ok=True)
    for i in range(0, len(embeddings), 500):
        torch.save(embeddings[i:i + 500], emb / f"embeddings_{i}.pt")
    _gzip_json(root / "index_config.json.gz", {
        "model_name": model_name, "full_document_collection": images is not None,
        "highest_doc_id": max(v["doc_id"] for v in embed_map.values()), "resize_stored_images": False,
        "max_image_width": None, "max_image_height": None, "library_version": "0.0.7"})
    _gzip_json(root / "embed_id_to_doc_id.json.gz", {str(k): v for k, v in embed_map.items()})
    if files is not None:
        _gzip_json(root / "doc_ids_to_file_names.json.gz", {str(k): v for k, v in files.items()})
    if metadata is not None:
        _gzip_json(root / "metadata.json.gz", {str(k): v for k, v in metadata.items()})
    if images is not None:
        coll = root / "collection"
        coll.mkdir(exist_ok=True)
        items = list(images.items())
        for i in range(0, len(items), 500):
            _gzip_json(coll / f"{i}.json.gz", {str(k): v for k, v in items[i:i + 500]})
    return root


def synthetic_pages(n_docs: int, pages_per_doc: int, seed: int = 0) -> tuple[list, dict]:
    """bfloat16 page tensors (unit rows, 3-9 tokens) and byaldi's embed_id -> (doc_id, page_id) map."""
    g = torch.Generator().manual_seed(seed)
    embeddings, embed_map = [], {}
    for d in range(n_docs):
        for p in range(1, pages_per_doc + 1):
            n = int(torch.randint(3, 10, (1,), generator=g))
            t = torch.randn(n, DIM, generator=g)
            embed_map[len(embeddings)] = {"doc_id": d, "page_id": p}
            embeddings.append((t / t.norm(dim=1, keepdim=True)).to(torch.bfloat16))
    return embeddings, embed_map


def torch_maxsim_top(queries: list, embeddings: list, k: int) -> list[list[int]]:
    """MaxSim over byaldi's own tensors in float32: for each query token the best page vector, summed."""
    out = []
    for q in queries:
        scores = torch.stack([(q.float() @ e.float().T).max(dim=1).values.sum() for e in embeddings])
        order = sorted(range(len(embeddings)), key=lambda i: (-float(scores[i]), i))
        out.append(order[:k])
    return out


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> tuple[Path, list, dict, dict, dict]:
    embeddings, embed_map = synthetic_pages(n_docs=7, pages_per_doc=150)  # 1,050 pages: three .pt files
    files = {0: "docs/annual.pdf", 1: "docs/q3.pdf", 2: "In-memory Image", 3: "scans/annual.pdf",
             4: "other/docs/annual.pdf", 5: "C:\\data\\memo.pdf", 6: "single.png"}
    metadata = {0: {"year": 2023}, 3: {"kind": "scan"}}
    root = write_byaldi(tmp_path_factory.mktemp("byaldi") / ".byaldi" / "my_index", embeddings, embed_map,
                        files, metadata, images={0: "iVBORw0KGgo=", 1: "iVBORw0KGgo="})
    return root, embeddings, embed_map, files, metadata


def test_the_vectors_come_back_bit_for_bit_in_embed_id_order(folder):
    root, embeddings, embed_map, _, _ = folder
    assert sorted(p.name for p in (root / "embeddings").iterdir()) == [
        "embeddings_0.pt", "embeddings_1000.pt", "embeddings_500.pt"]
    imported = B._read(root)
    c = imported.corpus
    assert len(c) == len(embeddings) == 1050 and c.dimension == DIM
    for e, t in enumerate(embeddings):
        lo, hi = c.span(e)
        assert np.array_equal(c.vectors[lo:hi], t.float().numpy()), e
        assert imported.pages[e].embed_id == e
        assert imported.pages[e].doc_id == embed_map[e]["doc_id"] and imported.pages[e].page == embed_map[e]["page_id"]
    assert imported.source_dtype == "bfloat16" and c.attrs["source"] == "byaldi"


def test_page_ids_files_and_metadata(folder):
    root, _, _, files, metadata = folder
    imported = B._read(root)
    by_doc = {p.doc_id: p for p in imported.pages if p.page == 2}
    assert by_doc[0].page_id == "docs/annual.pdf::p2" and by_doc[0].file == "docs/annual.pdf"
    assert by_doc[1].page_id == "docs/q3.pdf::p2"
    assert by_doc[2].page_id == "byaldi/doc2::p2" and by_doc[2].file is None
    assert by_doc[3].page_id == "scans/annual.pdf::p2"
    assert by_doc[4].page_id == "docs/annual.pdf~4::p2"           # same parent/name as document 0
    assert by_doc[5].page_id == "data/memo.pdf::p2"              # a Windows path, read with / separators
    assert by_doc[6].page_id == "single.png::p2"
    assert by_doc[0].metadata == metadata[0] and by_doc[3].metadata == metadata[3] and by_doc[1].metadata == {}
    assert len(set(imported.ids)) == len(imported.ids)
    assert imported.files[by_doc[0].page_id] == files[0]
    assert imported.model_name == "vidore/colqwen2-v1.0" and imported.n_images == 2


def test_a_search_over_the_import_ranks_as_maxsim_over_byaldi_tensors(folder):
    root, embeddings, _, _, _ = folder
    imported = B._read(root)
    g = torch.Generator().manual_seed(9)
    picks = [3, 77, 160, 499, 500, 501, 777, 1000, 1049]
    queries = []
    for i in picks:  # noisy copies of a few tokens of known pages, in bfloat16 as byaldi's model gives them
        e = embeddings[i].float()
        q = e[: min(3, e.shape[0])] + 0.3 * torch.randn(min(3, e.shape[0]), DIM, generator=g) / DIM ** 0.5
        queries.append(q.to(torch.bfloat16))
    want = torch_maxsim_top(queries, embeddings, k=10)
    qc = MultiVectorCorpus.from_arrays([q.float().numpy() for q in queries])
    got = ExactRetriever(imported.corpus).search(qc, 10)
    assert [list(map(int, row)) for row in got.indices] == want
    assert [row[0] for row in got.ids] == [imported.ids[i] for i in picks]


def test_an_index_from_before_byaldi_0_0_2_has_no_file_map(tmp_path):
    embeddings, embed_map = synthetic_pages(2, 3, seed=1)
    root = write_byaldi(tmp_path / "old", embeddings, embed_map, files=None, metadata=None)
    imported = B._read(root)
    assert imported.ids == [f"byaldi/doc{d}::p{p}" for d in range(2) for p in range(1, 4)]
    assert imported.files == {} and imported.n_images == 0


def test_a_stacked_chunk_and_a_leading_batch_axis_are_read(tmp_path):
    root = tmp_path / "stacked"
    (root / "embeddings").mkdir(parents=True)
    stacked = torch.randn(4, 5, DIM).to(torch.bfloat16)
    torch.save(stacked, root / "embeddings" / "embeddings_0.pt")
    torch.save([torch.randn(1, 6, DIM)], root / "embeddings" / "embeddings_4.pt")
    _gzip_json(root / "index_config.json.gz", {"model_name": "vidore/colpali-v1.2"})
    _gzip_json(root / "embed_id_to_doc_id.json.gz", {str(i): {"doc_id": 0, "page_id": i + 1} for i in range(5)})
    imported = B._read(root)
    assert [int(n) for n in imported.corpus.counts] == [5, 5, 5, 5, 6]
    assert imported.source_dtype == "bfloat16,float32"


@pytest.mark.parametrize("damage", ["no_config", "no_map", "count", "dims", "name", "twice", "not_tensor"])
def test_damaged_folders_are_refused(tmp_path, damage):
    embeddings, embed_map = synthetic_pages(1, 4, seed=2)
    files = {0: "a/b.pdf"}
    if damage == "dims":
        embeddings[2] = torch.randn(4, DIM + 1).to(torch.bfloat16)
    if damage == "count":
        embed_map[4] = {"doc_id": 0, "page_id": 5}
    if damage == "twice":
        embed_map[3] = {"doc_id": 0, "page_id": 1}
    root = write_byaldi(tmp_path / damage, embeddings, embed_map, files, None)
    if damage == "no_config":
        (root / "index_config.json.gz").unlink()
    if damage == "no_map":
        (root / "embed_id_to_doc_id.json.gz").unlink()
    if damage == "name":
        (root / "embeddings" / "embeddings_0.pt").rename(root / "embeddings" / "embeddings_x.pt")
    if damage == "not_tensor":
        torch.save([{"not": "a tensor"}] * 4, root / "embeddings" / "embeddings_0.pt")
    with pytest.raises(B.ByaldiFormatError):
        B._read(root)


def test_torch_is_needed_and_named(folder, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(MissingExtra, match=r'pip install "optivision-rag\[byaldi\]"'):
        B._read(folder[0])


def test_the_summary_tags_its_counts_and_ends_with_next(folder):
    text = str(B._read(folder[0]).summary())
    assert "1,050 pages from 7 documents" in text and "[counted]" in text
    assert "bfloat16" in text and text.splitlines()[-1].startswith("Next: ") and text.isascii()


def test_import_is_switched_off_until_the_gate_is_recorded(folder):
    assert not _ledger.is_on("byaldi_import")
    with pytest.raises(FeatureOff, match="byaldi import is switched off"):
        B.import_index(folder[0])
    with pytest.raises(FeatureOff):
        B.read_index(folder[0])


def test_import_hands_the_vectors_and_ids_to_build(folder, monkeypatch, capsys):
    """With the switch forced open for this test only, import_index calls ov.build on the read vectors."""
    root = folder[0]
    monkeypatch.setattr(_ledger, "require", lambda name: None)
    seen = {}

    def build(source, model=None, target=None, **kw):
        seen.update(source=source, model=model, target=target, **kw)
        return "an Index"

    fake = types.ModuleType("optivision.easy._api")
    fake.build = build
    monkeypatch.setitem(sys.modules, "optivision.easy._api", fake)
    assert B.import_index(root, target=0.95) == "an Index"
    imported = B._read(root)
    assert seen["model"] == "vidore/colqwen2-v1.0" and seen["target"] == 0.95
    assert seen["ids"] == imported.ids and seen["source"].ids == imported.ids
    assert np.array_equal(seen["source"].vectors, imported.corpus.vectors)
    assert seen["path"] == root.with_name("my_index.ovx")
    assert "1,050 pages" in capsys.readouterr().out


def test_an_unknown_byaldi_model_imports_without_a_model(folder, monkeypatch):
    assert B._registry_model("someone/not-a-model") is None
    assert B._registry_model("VIDORE/COLQWEN2-V1.0") == "vidore/colqwen2-v1.0"
    assert B._registry_model(None) is None


def test_end_to_end_once_the_build_module_exists(folder, monkeypatch, tmp_path):
    """Runs only when ``optivision.easy._api`` is present: the built Index searches the imported vectors."""
    pytest.importorskip("optivision.easy._api")
    monkeypatch.setattr(_ledger, "require", lambda name: None)
    imported = B._read(folder[0])
    index = B.import_index(folder[0], out=tmp_path / "imported.ovx", model=None, progress=False)
    try:
        assert index.n_pages == len(imported.pages)
        picks = [5, 600, 1049]
        rows = index.search_vectors(imported.corpus.select(picks), k=10)
        assert [row[0].id for row in rows] == [imported.ids[i] for i in picks]  # each page finds itself
    finally:
        index.close()
