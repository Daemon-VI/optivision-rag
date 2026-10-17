"""The measurement scripts agree on the vector-file names a follow-up batch reads and writes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts"

pytestmark = pytest.mark.skipif(not (ROOT / "measure_models.py").exists(), reason="needs the repository's scripts")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_three_scripts_name_vector_files_alike():
    measure, pages, text = _load("measure_models"), _load("encode_vectors"), _load("encode_text_vectors")
    for dataset in ("vidore/docvqa_test_subsampled", "vidore/arxivqa_test_subsampled"):
        for count in (0, 2000):
            assert measure.tag_for("colSmol-500M", dataset, count) == pages.tag_for("vidore/colSmol-500M", dataset, count)
    for dataset in ("scifact", "nfcorpus"):
        for count in (0, 10000):
            assert measure.tag_for("colbertv2.0", dataset, count) == text.tag_for("colbert-ir/colbertv2.0", dataset, count)
    # the names every earlier campaign run used are unchanged
    assert text.tag_for("lightonai/LateOn") == "LateOn_scifact"
    assert pages.tag_for("vidore/colpali", "vidore/docvqa_test_subsampled") == "colpali_docvqa_test_subsampled"
    assert measure.tag_for("colpali", "vidore/infovqa_test_subsampled", 2000) == "colpali_infovqa_test_subsampled_plus2000"


def test_image_step_names_keep_the_two_default_splits():
    split_key = _load("measure_models").split_key
    assert split_key("vidore/docvqa_test_subsampled") == "docvqa"
    assert split_key("vidore/infovqa_test_subsampled") == "infovqa"
    assert split_key("vidore/arxivqa_test_subsampled") == "arxivqa"
    assert split_key("vidore/tabfquad_test_subsampled") == "tabfquad"


def test_check_pairs_never_pair_a_query_with_a_document_another_query_also_needs():
    choose = _load("encode_text_vectors").choose_check_pairs
    qrels = {"q1": ["d1", "d2"], "q2": ["d2"], "q3": ["d3", "d1"], "q4": ["d4"], "q5": ["d4", "d5"]}
    text = {q: q for q in qrels}
    pairs = choose(qrels, text, 10)
    assert pairs == [("d1", "q1"), ("d4", "q4")]  # q2's d2 and q3's d1 belong to q1; q5's d4 is q4's
    for doc, query in pairs:
        others = {d for d, q in pairs if q != query}
        assert doc in qrels[query] and not set(qrels[query]) & others
    assert choose({"1": ["a"], "2": ["b"]}, {"1": "x"}, 5) == [("a", "1")]  # a query without text is skipped


def _archive(path: Path, batch: str, model: str, short: str) -> Path:
    import json
    import zipfile

    top = f"results_{batch}"
    run = {"model": model, "kind": "text", "spec": {"model": model, "datasets": ["nfcorpus"]}, "steps": {},
           "outcome": "measured", "check_retriever": {"ok": True}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{top}/batch_summary.json", json.dumps([{"model": model, "outcome": "measured"}]))
        z.writestr(f"{top}/commit.txt", "abc123\n")
        z.writestr(f"{top}/{short}/run.json", json.dumps(run))
        z.writestr(f"{top}/{short}/wide/{short}_nfcorpus.json", "{}")
        z.writestr(f"{top}/{short}/geometry/{short}_nfcorpus.json", "{}")
        z.writestr(f"{top}/{short}/evil.py", "print('never copied')")
    return path


def test_a_followup_archive_lands_in_its_own_folder_and_leaves_the_campaign_alone(tmp_path, monkeypatch):
    ingest = _load("ingest_campaign")
    monkeypatch.setattr(ingest, "CAMPAIGN", tmp_path / "campaign")
    monkeypatch.setattr(ingest, "FOLLOWUP", tmp_path / "followup")
    assert ingest.is_followup("f1_text_nfcorpus_a") and ingest.is_followup("f8_geometry")
    assert not ingest.is_followup("14a_text_retry") and not ingest.is_followup("fake")
    outcomes: dict = {}
    zip_path = _archive(tmp_path / "results_f1_text_nfcorpus_a.zip", "f1_text_nfcorpus_a", "colbert-ir/colbertv2.0",
                        "colbertv2.0")
    lines = ingest.ingest(zip_path, outcomes)
    model_dir = tmp_path / "followup" / "f1_text_nfcorpus_a" / "colbert-ir--colbertv2.0"
    assert (model_dir / "run.json").exists() and (model_dir / "geometry" / "colbertv2.0_nfcorpus.json").exists()
    assert not (model_dir / "evil.py").exists()
    assert (tmp_path / "followup" / "f1_text_nfcorpus_a" / "_batch" / "commit.txt").exists()
    assert not (tmp_path / "campaign").exists()
    assert set(outcomes) == {"colbert-ir/colbertv2.0 @ f1_text_nfcorpus_a"}
    assert outcomes["colbert-ir/colbertv2.0 @ f1_text_nfcorpus_a"]["spec"] == {"datasets": ["nfcorpus"]}
    assert any("measured" in line for line in lines)


def test_distractor_pages_are_appended_a_chunk_at_a_time_without_labels(monkeypatch):
    import sys
    import types

    import numpy as np

    from optivision.representation import MultiVectorCorpus

    pages = _load("encode_vectors")

    class Page:
        def __init__(self, k: int) -> None:
            self.k = k

        def convert(self, mode: str):
            return self

    class Adapter:  # one 2 x 4 array per page, filled with the page's number
        def __init__(self) -> None:
            self.calls: list[int] = []

        def encode_documents(self, images, ids=None):
            self.calls.append(len(images))
            return MultiVectorCorpus.from_arrays([np.full((2, 4), float(im.k), np.float32) for im in images], ids=ids)

    stub = types.ModuleType("datasets")
    stub.load_dataset = lambda name, split: [{"image": Page(100 + i if name == "a" else 200 + i)} for i in range(5)]
    monkeypatch.setitem(sys.modules, "datasets", stub)
    adapter = Adapter()
    own = MultiVectorCorpus.from_arrays([np.zeros((3, 4), np.float32)], ids=["00000::p1"], attrs={"model": "m"})
    out = pages._with_distractors(adapter, own, ["a", "b"], 7, "test", chunk=3)
    assert out.ids == ["00000::p1", *[f"x{j:05d}::p1" for j in range(7)]]
    assert adapter.calls == [3, 3, 1]  # chunks, never all pages at once
    firsts = [float(out.vectors[out.offsets[i]][0]) for i in range(len(out))]
    assert firsts == [0.0, 100, 101, 102, 103, 104, 200, 201] and out.attrs["model"] == "m"
    with pytest.raises(SystemExit, match="only 10 distractor pages"):
        pages._with_distractors(adapter, own, ["a", "b"], 11, "test", chunk=3)
