"""Files written by v0.3.0 must keep reading back, with the same meaning, in later versions.

``tests/golden/`` holds small fixtures written once by the v0.3.0 library (see
``tests/golden/make_golden.py``) and ``expected_v030.json``, which records what each file
must read back as: array digests, ids, metadata, pipelines, a few decoded rows, MaxSim scores
for a fixed query set and the rankings they give. These tests load the committed files with
the installed library. Arrays, ids, metadata, pipelines and rankings must match exactly;
scores to 1e-6. A change that stops the library reading a v0.3.0 file, or reads it as
something else, fails here. Nothing is unpickled.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from optivision.compose import Pipeline, load_compressed, save_compressed
from optivision.index.numpy_index import NumpyIndex
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, rank

GOLDEN = Path(__file__).resolve().parent / "golden"
EXPECTED = "expected_v030.json"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
FIXTURE_FILES = [
    "corpus_f32_v030.npz", "corpus_f16_v030.npz", "queries_v030.npz",
    "compressed_binary_v030.npz", "compressed_int4_v030.npz", "compressed_int8_per_vector_v030.npz",
    "compressed_float16_v030.npz", "compressed_float32_empty_v030.npz",
    "compressed_merge_int8_per_vector_v030.npz",
    "index_binary_v030/codes.npy", "index_binary_v030/offsets.npy", "index_binary_v030/meta.json",
]
SCORE_TOLERANCE = {"rtol": 1e-6, "atol": 1e-6}


# ------------------------------------------------------------------- helpers


def sha256(a: np.ndarray) -> str:
    """dtype string, shape and bytes: the scheme make_golden.py used (no optivision.core)."""
    a = np.ascontiguousarray(a)
    head = f"{a.dtype.str}|{','.join(str(n) for n in a.shape)}|".encode("ascii")
    return hashlib.sha256(head + a.tobytes()).hexdigest()


def floats(nested) -> np.ndarray:
    """Recorded numbers as float32; ``null`` stands for -inf (an empty document's score)."""

    def convert(x):
        if isinstance(x, list):
            return [convert(item) for item in x]
        return -np.inf if x is None else x

    return np.asarray(convert(nested), dtype=np.float32)


def load_expected(directory: Path) -> dict:
    return json.loads((directory / EXPECTED).read_text(encoding="utf-8"))


def only(entries: dict):
    (name, entry), = entries.items()
    return name, entry


# --------------------------------------------------------------------- checks


def check_corpus_content(c: MultiVectorCorpus, entry: dict) -> None:
    assert str(c.dtype) == entry["dtype"] and list(c.vectors.shape) == entry["shape"]
    assert c.ids == entry["ids"]
    assert c.metadata == entry["metadata"]
    assert c.attrs == entry["attrs"]
    assert c.counts.tolist() == entry["counts"]
    assert c.images is None
    assert sha256(c.vectors) == entry["sha256"]["vectors"]
    assert sha256(c.offsets) == entry["sha256"]["offsets"]
    for name in ("protected", "positions", "weights", "importance"):
        array, digest = getattr(c, name), entry["sha256"][name]
        assert (array is None) == (digest is None), name
        if array is not None:
            assert sha256(array) == digest, name
    for row, values in entry["rows"].items():
        np.testing.assert_array_equal(c.vectors[int(row)].astype(np.float32), floats(values))


def check_queries(directory: Path, expected: dict) -> MultiVectorCorpus:
    name, entry = only(expected["queries"])
    queries = MultiVectorCorpus.load(directory / name)
    check_corpus_content(queries, entry)
    return queries


def check_scores(scores: np.ndarray, entry: dict) -> None:
    np.testing.assert_allclose(scores, floats(entry["scores"]), **SCORE_TOLERANCE)
    assert rank(scores).tolist() == entry["ranking"]  # exact, ties by position


def check_corpora(directory: Path, expected: dict, queries: MultiVectorCorpus) -> None:
    assert sorted(expected["corpora"]) == ["corpus_f16_v030.npz", "corpus_f32_v030.npz"]
    for name, entry in expected["corpora"].items():
        c = MultiVectorCorpus.load(directory / name)
        check_corpus_content(c, entry)
        mapped = MultiVectorCorpus.load(directory / name, mmap=True)
        np.testing.assert_array_equal(mapped.vectors, c.vectors)
        check_scores(maxsim_matrix(queries, c), entry)


def check_compressed(directory: Path, expected: dict, queries: MultiVectorCorpus) -> None:
    assert len(expected["compressed"]) >= 6
    for name, entry in expected["compressed"].items():
        c = load_compressed(directory / name)
        assert (len(c), c.num_vectors, c.dim) == (
            entry["n_docs"], entry["num_vectors"], entry["dim"])
        assert c.codes.dtype == np.uint8 and c.codes.shape == (c.num_vectors, entry["code_bytes"])
        assert sha256(c.codes) == entry["sha256"]["codes"]
        assert sha256(c.offsets) == entry["sha256"]["offsets"]
        assert c.counts.tolist() == entry["counts"] and c.ids == entry["ids"]
        assert c.stats == entry["stats"]
        assert c.pipeline == entry["pipeline"]
        assert c.quantizer.to_dict() == entry["quantizer"] and c.quantizer.fitted
        assert c.nbytes == entry["nbytes"]
        assert Pipeline.from_dict(c.pipeline).to_dict() == entry["pipeline"]
        for row, values in entry["decoded_rows"].items():
            decoded = c.decode_rows(int(row), int(row) + 1)[0]
            np.testing.assert_allclose(decoded, floats(values), rtol=1e-6, atol=1e-7)
        check_scores(maxsim_matrix(queries, c), entry)


def check_index(directory: Path, expected: dict, queries: MultiVectorCorpus) -> None:
    name, entry = only(expected["index"])
    path = directory / name
    index = NumpyIndex.load(path)
    assert (index.n_pages, index.dim, index.method) == (
        entry["n_pages"], entry["dim"], entry["method"])
    assert [r.page_id for r in index.refs] == entry["page_ids"]
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    assert sorted(meta) == entry["meta_keys"] and sorted(meta["refs"][0]) == entry["ref_keys"]
    assert meta["codec"] == entry["codec"]
    assert sha256(np.load(path / "codes.npy", allow_pickle=False)) == entry["sha256"]["codes"]
    assert sha256(np.load(path / "offsets.npy", allow_pickle=False)) == entry["sha256"]["offsets"]
    stats = index.stats()
    assert sorted(stats) == sorted(entry["stats"])
    for key, value in entry["stats"].items():
        assert stats[key] == pytest.approx(value, rel=1e-9), key
    want = floats(entry["scores"])
    for j, query in enumerate(queries):
        vectors = np.asarray(query.vectors, dtype=np.float32)
        np.testing.assert_allclose(index.score_all(vectors), want[j], **SCORE_TOLERANCE)
        hits = index.search(vectors, top_k=5)
        assert [(h.ref.page_id, h.rank) for h in hits] == [
            (h["page_id"], h["rank"]) for h in entry["search"][j]]
        np.testing.assert_allclose(
            [h.score for h in hits], [h["score"] for h in entry["search"][j]], **SCORE_TOLERANCE)


def check_round_trips(directory: Path, expected: dict, scratch: Path) -> None:
    """What the current library reads from a v0.3.0 file it can write again and read back."""
    for name in (*expected["corpora"], *expected["queries"]):
        c = MultiVectorCorpus.load(directory / name)
        back = MultiVectorCorpus.load(c.save(scratch / name))
        check_corpus_content(back, (expected["corpora"] | expected["queries"])[name])
    for name, entry in expected["compressed"].items():
        c = load_compressed(directory / name)
        save_compressed(c, scratch / name)
        back = load_compressed(scratch / name)
        assert sha256(back.codes) == entry["sha256"]["codes"]
        assert back.ids == c.ids and back.stats == c.stats and back.pipeline == c.pipeline
        assert back.quantizer.to_dict() == c.quantizer.to_dict() and back.dim == c.dim
    name, entry = only(expected["index"])
    index = NumpyIndex.load(directory / name)
    index.path = scratch / name
    index.save()
    again = NumpyIndex.load(scratch / name)
    assert [r.page_id for r in again.refs] == entry["page_ids"]
    probe = np.ones((2, entry["dim"]), dtype=np.float32)
    np.testing.assert_array_equal(again.score_all(probe), index.score_all(probe))


def check_everything(directory: Path, expected: dict, scratch: Path) -> None:
    queries = check_queries(directory, expected)
    check_corpora(directory, expected, queries)
    check_compressed(directory, expected, queries)
    check_index(directory, expected, queries)
    check_round_trips(directory, expected, scratch)


@pytest.fixture(scope="module")
def expected() -> dict:
    return load_expected(GOLDEN)


@pytest.fixture(scope="module")
def queries(expected) -> MultiVectorCorpus:
    return check_queries(GOLDEN, expected)


# ---------------------------------------------------------------------- tests


def test_fixtures_are_present_and_are_not_lfs_pointers():
    for name in [*FIXTURE_FILES, EXPECTED]:
        path = GOLDEN / name
        assert path.is_file(), f"missing fixture {name}; run tests/golden/make_golden.py once"
        assert not path.read_bytes().startswith(LFS_POINTER), (
            f"{name} is a Git LFS pointer: tests/golden/.gitattributes must keep the fixtures "
            "out of LFS, and CI checks out without it")


def test_fixtures_stay_small():
    files = [p for p in GOLDEN.rglob("*") if p.is_file() and p.suffix in {".npz", ".npy", ".json"}]
    assert sum(p.stat().st_size for p in files) < 100_000


def test_the_files_have_the_v030_layout():
    """Raw layout, read without optivision: member names, header keys, dtypes, no pickle."""
    for name in ("corpus_f32_v030.npz", "corpus_f16_v030.npz", "queries_v030.npz"):
        with np.load(GOLDEN / name, allow_pickle=False) as data:
            assert {"vectors", "offsets", "header"} <= set(data.files)
            assert set(data.files) <= {"vectors", "offsets", "header", "protected", "positions",
                                       "weights", "importance"}
            header = json.loads(bytes(data["header"]).decode("utf-8"))
            assert data["offsets"].dtype == np.int64 and data["header"].dtype == np.uint8
        assert header["format"] == "optivision-multivector" and header["version"] == 1
        assert sorted(header) == ["attrs", "format", "ids", "metadata", "version"]
    for name in FIXTURE_FILES:
        if name.startswith("compressed_"):
            with np.load(GOLDEN / name, allow_pickle=False) as data:
                assert sorted(data.files) == ["codes", "header", "offsets"]
                header = json.loads(bytes(data["header"]).decode("utf-8"))
                assert data["codes"].dtype == np.uint8 and data["offsets"].dtype == np.int64
            assert header["format"] == "optivision-compressed" and header["version"] == 1
            assert sorted(header) == ["dim", "format", "ids", "pipeline", "quantizer", "stats",
                                      "version"]
    index = GOLDEN / "index_binary_v030"
    assert sorted(p.name for p in index.iterdir()) == ["codes.npy", "meta.json", "offsets.npy"]
    assert np.load(index / "codes.npy", allow_pickle=False).dtype == np.uint8
    assert np.load(index / "offsets.npy", allow_pickle=False).dtype == np.int64
    meta = json.loads((index / "meta.json").read_text(encoding="utf-8"))
    assert sorted(meta) == ["backend", "codec", "dim", "method", "n_pages", "page_stats", "refs"]
    assert meta["backend"] == "numpy" and meta["method"] == "binary" and meta["codec"] is None
    assert sorted(meta["refs"][0]) == ["doc_id", "image_path", "page_no", "source_path"]


def test_the_fixtures_cover_what_they_are_meant_to_cover(expected, queries):
    entry = expected["corpora"]["corpus_f32_v030.npz"]
    c = MultiVectorCorpus.load(GOLDEN / "corpus_f32_v030.npz")
    half = MultiVectorCorpus.load(GOLDEN / "corpus_f16_v030.npz")
    assert c.dtype == np.float32 and half.dtype == np.float16
    assert (c.counts == 0).any(), "an empty document"
    assert np.isnan(c.positions).any() and c.protected.any() and (c.weights != 1).any()
    assert c.importance is None and c.dimension % 8 != 0
    assert c.metadata[1]["title"] == "caf" + chr(0xE9), "non-ASCII text in the header"
    assert isinstance(c.attrs["encoder"], dict)
    assert len(entry["ids"]) == len(set(entry["ids"])) == 12 and len(queries) == 4
    chains = {tuple(s["stage"] for s in e["pipeline"]["stages"])
              for e in expected["compressed"].values()}
    assert chains == {("binary",), ("int4",), ("int8",), ("float16",), (),
                      ("adaptive_merge", "int8")}
    quantizers = {e["quantizer"]["stage"] for e in expected["compressed"].values()}
    assert quantizers == {"binary", "int4", "int8", "float16", "float32"}


def test_a_v030_corpus_reads_back_as_recorded(expected, queries):
    check_corpora(GOLDEN, expected, queries)


def test_v030_queries_read_back_as_recorded(expected):
    check_queries(GOLDEN, expected)


def test_v030_compressed_corpora_read_back_as_recorded(expected, queries):
    check_compressed(GOLDEN, expected, queries)


def test_a_v030_numpy_index_reads_back_as_recorded(expected, queries):
    check_index(GOLDEN, expected, queries)


def test_what_is_read_can_be_written_and_read_again(expected, tmp_path):
    check_round_trips(GOLDEN, expected, tmp_path)


def test_the_index_and_the_compressed_corpus_store_the_same_binary_codes(expected):
    """Two generations, one codec: the legacy Compressor and BinaryQuantizer agree byte for byte."""
    index_codes = np.load(GOLDEN / "index_binary_v030" / "codes.npy", allow_pickle=False)
    compressed = load_compressed(GOLDEN / "compressed_binary_v030.npz")
    np.testing.assert_array_equal(index_codes, compressed.codes)  # the empty page has no rows
    ids = [i for i, n in zip(compressed.ids, compressed.counts, strict=True) if n]
    assert ids == only(expected["index"])[1]["page_ids"]


def test_the_scores_are_those_of_the_stored_representation(expected, queries):
    """Binary codes decode to +/-1: scoring the decoded vectors by hand gives the same numbers."""
    compressed = load_compressed(GOLDEN / "compressed_binary_v030.npz")
    decoded = compressed.to_corpus()
    assert set(np.unique(decoded.vectors)) == {-1.0, 1.0}
    by_hand = maxsim_matrix(queries, decoded)
    np.testing.assert_allclose(by_hand, maxsim_matrix(queries, compressed), **SCORE_TOLERANCE)
    check_scores(by_hand, expected["compressed"]["compressed_binary_v030.npz"])


# ------------------------------------------------------------- the generator


def load_generator(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # no __pycache__ inside tests/golden
    spec = importlib.util.spec_from_file_location("make_golden", GOLDEN / "make_golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_generator_still_runs_and_what_it_writes_passes_the_same_checks(tmp_path, monkeypatch):
    fresh = tmp_path / "fresh"
    recorded = load_generator(monkeypatch).main(fresh, strict=False)
    assert load_expected(fresh) == json.loads(json.dumps(recorded))
    check_everything(fresh, recorded, tmp_path / "scratch")


def test_the_generator_refuses_to_overwrite_fixtures(tmp_path, monkeypatch):
    copy = tmp_path / "golden"
    shutil.copytree(GOLDEN, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = (copy / EXPECTED).read_bytes()
    generator = load_generator(monkeypatch)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        generator.main(copy)
    assert (copy / EXPECTED).read_bytes() == before
    written = generator.main(copy, force=True, strict=False)  # an explicit request is honoured
    assert set(written) == set(load_expected(GOLDEN))


# ------------------------------------------------------- persistent identities


QDRANT_NAMESPACE = "6f2a1f0e-4f8a-4d3e-9a1b-2c7d5e8f0a11"
QDRANT_POINT_IDS = {
    "doc0::p1": "09710ec3-816b-54c5-a1bf-04ac59e06179",
    "doc1::p2": "af9a8b02-dff1-54ff-91ad-ca3d9c425ee8",
}
QDRANT_PAYLOAD_KEYS = {
    "page_id", "doc_id", "page_no", "source_path", "image_path", "n_tokens_before",
    "n_tokens_after", "nbytes", "raw_nbytes",
}


def test_qdrant_point_ids_and_payload_keys_are_stable(tmp_path):
    """A collection written by v0.3.0 holds uuid5(namespace, page_id) points with this payload.

    Re-indexing the same page must land on the same point, so neither the namespace nor the
    derivation may change; the payload keys are what ``QdrantIndex`` reads back.
    """
    pytest.importorskip("qdrant_client")
    import uuid

    from optivision.compression import Compressor
    from optivision.config import CompressionConfig
    from optivision.index import qdrant_index
    from optivision.types import PageRef, PatchGrid, PrunedPage

    assert str(qdrant_index._NAMESPACE) == QDRANT_NAMESPACE
    for page_id, point_id in QDRANT_POINT_IDS.items():
        assert str(uuid.uuid5(qdrant_index._NAMESPACE, page_id)) == point_id

    vectors = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    ref = PageRef(doc_id="doc0", page_no=1, source_path="pdfs/doc0.pdf")
    page = Compressor(CompressionConfig(method="binary")).compress(PrunedPage(
        ref=ref, embeddings=vectors, kept_token_index=np.arange(4, dtype=np.int32),
        keep_mask=np.ones((1, 4), dtype=bool), grid=PatchGrid.contiguous(1, 4), n_tokens_before=40))
    index = qdrant_index.QdrantIndex(path=tmp_path / "q", dim=16, method="binary", recreate=True)
    try:
        index.add([page])
        found = index.client.retrieve(index.collection, ids=[QDRANT_POINT_IDS["doc0::p1"]],
                                      with_payload=True)
    finally:
        index.close()
    assert len(found) == 1
    assert set(found[0].payload) == QDRANT_PAYLOAD_KEYS
    assert found[0].payload["page_id"] == "doc0::p1" and found[0].payload["n_tokens_before"] == 40
