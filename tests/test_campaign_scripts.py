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


def test_a_spec_names_known_studies_and_an_unknown_one_fails_before_encoding(tmp_path):
    measure = _load("measure_models")
    assert measure.STUDIES["s13"] == "s13_wide_study.py" and (ROOT / "universal_study" / measure.STUDIES["s17"]).exists()
    with pytest.raises(ValueError, match="unknown study"):
        measure.measure({"model": "colbert-ir/colbertv2.0", "study": ["s99"]}, tmp_path, tmp_path,
                        measure.Session(1.0), False, "python")
    assert not any(tmp_path.iterdir())


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


def test_prediction_status_waits_for_pending_units_but_fails_early():
    score = _load("followup_predictions")

    def unit(state, value=None):
        u = score.Unit.__new__(score.Unit)
        u.model, u.key, u.batch, u.state = "org/m", "nfcorpus", "f1", state
        u.result = {"v": value} if state == "ran" else None
        return u

    ok = lambda r: r["v"] >= 0.95
    assert score.each([unit("ran", 0.99), unit("pending")], ok)[0] == "pending"
    assert score.each([unit("ran", 0.99), unit("ran", 0.96)], ok)[0] == "held"
    assert score.each([unit("ran", 0.90), unit("pending")], ok)[0] == "failed"  # one miss sinks "every unit"
    assert score.each([unit("ran", 0.90), unit("ran", 0.99), unit("pending")], ok, need=2)[0] == "pending"
    assert score.each([unit("ran", 0.90), unit("ran", 0.99), unit("not run")], ok, need=2)[0] == "not testable"
    assert score.each([unit("ran", 0.99), unit("ran", 0.99), unit("pending")], ok, need=2)[0] == "held"
    assert score.combine([("held", "a"), ("pending", "b")])[0] == "pending"
    assert score.combine([("pending", "a"), ("failed", "b")])[0] == "failed"


def test_spearman_handles_ties_and_order():
    rho = _load("followup_predictions").spearman
    assert rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert rho([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, abs=1e-6)
    assert rho([1, 1, 1], [1, 2, 3]) == 0.0


def test_int4norm_scidocs_flag_rule_excuses_only_flagged_models(monkeypatch):
    score = _load("int4norm_predictions")

    def unit(model, int4):
        u = score.Unit.__new__(score.Unit)
        u.model, u.key, u.batch, u.state = model, "scidocs", "f12b_text_scidocs", "ran"
        rows = [{"label": lab, "retention:ndcg@5": v, "retention_ci:ndcg@5": [v - 0.01, v + 0.01]}
                for lab, v in (("int8(per_vector)", 0.99), ("int4", int4), ("int4norm(mean)", 0.99))]
        u.result = {"dataset": f"{model.split('/')[1]}_scidocs", "rows": rows}
        return u

    def units(control_int4):
        return [unit(m, 0.10 if m in score.FLAGGED else control_int4) for m in score.SCIDOCS]

    assert len(score.SCIDOCS) == 8 and {"colbert-ir/colbertv2.0", "jinaai/jina-colbert-v2"} == set(score.SCIDOCS) - score.FLAGGED
    monkeypatch.setattr(score, "scidocs_units", lambda: units(0.95))
    assert score.q9()[0] == "held"  # the flagged models' collapse is allowed
    monkeypatch.setattr(score, "scidocs_units", lambda: units(0.85))
    assert score.q9()[0] == "failed"  # an unflagged model below 0.90 sinks the clause
