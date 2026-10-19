"""The measured catalogue: equal to the reports it copies, honest about what it holds, and the
``models`` command and ``encode_and_certify`` built on the registry."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import optivision
from optivision import encode_and_certify
from optivision.adapters import (
    MEASURED,
    MODELS,
    EncoderAdapter,
    corpus_from_arrays,
    measured_models,
    measured_results,
    measured_shortlist,
    supported_models,
)
from optivision.cli import app
from optivision.compose import Pipeline
from optivision.core.selection import Outcome
from optivision.core.vectors import EncoderIdentity
from optivision.stages import Float16Quantizer, Float32Quantizer, Int8Quantizer
from optivision.testing import make_corpus, make_queries

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_measured_catalog.py"

#: measured models without fixed-pipeline rows, and why
NO_ROWS = {
    "vidore/colSmol-256M": "measured on generated corpora only (E1/E4), which have no frontier study",
    "vidore/colqwen2-v1.0": "same vectors as vidore/colqwen2-v1.0-merged, which holds the rows",
}


def _builder():
    spec = importlib.util.spec_from_file_location("build_measured_catalog", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------ the catalogue


@pytest.mark.skipif(not BUILDER.exists() or not (ROOT / "reports" / "universal").exists(),
                    reason="needs the repository's reports (not in an installed wheel)")
def test_the_shipped_catalogue_equals_what_the_reports_give():
    builder = _builder()
    shipped = (ROOT / "src" / "optivision" / "presets" / "measured.json").read_text(encoding="utf-8")
    assert shipped.replace("\r\n", "\n") == builder.render(builder.build())


def test_every_catalogued_model_is_registered_as_measured_and_the_others_are_explained():
    measured = {m.model_id for m in supported_models(MEASURED)}
    with_rows = set(measured_models())
    assert with_rows <= measured
    assert measured - with_rows == set(NO_ROWS)


def test_rows_hold_a_rebuildable_pipeline_and_plausible_figures():
    for model_id in measured_models():
        for dataset, entry in measured_results(model_id).items():
            assert entry["source"].startswith("reports/universal/") and entry["n_queries"] > 0
            assert 0.0 < entry["baseline"] <= 1.0, (model_id, dataset)
            labels = [r["label"] for r in entry["rows"]]
            # 45: the 48- and 64-d mxbai-edge models skip projections no narrower than themselves (49 rows)
            assert len(labels) == len(set(labels)) >= 45
            for row in entry["rows"]:
                lo, hi = row["retention_interval"]
                assert lo <= row["retention"] <= hi and row["compression"] > 0
                assert isinstance(Pipeline.from_dict(row["pipeline"]), Pipeline)
            baseline = [r for r in entry["rows"] if not r["pipeline"]["stages"]]
            assert baseline and all(r["retention"] == 1.0 and r["compression"] == 1.0 for r in baseline)


def test_the_result_is_a_copy_and_an_unmeasured_model_is_an_error():
    first = measured_results("vidore/colqwen2.5-v0.2")
    next(iter(first.values()))["rows"].clear()
    assert next(iter(measured_results("vidore/colqwen2.5-v0.2").values()))["rows"]
    with pytest.raises(KeyError, match="no measured"):
        measured_results("Metric-AI/ColQwen2.5-7b-multilingual-v1.0")
    with pytest.raises(KeyError, match="was not measured on"):
        measured_results("vidore/colqwen2.5-v0.2", "BeIR/scifact")
    assert list(measured_results("vidore/colqwen2.5-v0.2", "vidore/docvqa_test_subsampled")) == [
        "vidore/docvqa_test_subsampled"]


def test_the_shortlist_keeps_only_pipelines_that_held_on_every_dataset():
    model_id = "nvidia/nemotron-colembed-vl-4b-v2"
    strict = measured_shortlist(model_id, 0.97)
    loose = measured_shortlist(model_id, 0.97, on="point")
    assert strict and len(loose) >= len(strict)
    for row in strict:
        assert row["pipeline"]["stages"] and set(row["datasets"]) == set(measured_results(model_id))
        assert all(d["retention_interval"][0] >= 0.97 for d in row["datasets"].values())
    smallest = [min(d["compression"] for d in row["datasets"].values()) for row in strict]
    assert smallest == sorted(smallest, reverse=True)
    assert measured_shortlist(model_id, 1.2) == []
    with pytest.raises(ValueError, match="on must be"):
        measured_shortlist(model_id, 0.97, on="hope")
    with pytest.raises(ValueError, match="min_retention"):
        measured_shortlist(model_id, 97)


def test_plain_int4_is_absent_from_the_wide_shortlist_where_it_collapsed():
    """A known measured fact (R12): plain int4 keeps almost nothing on the 2,560-d model."""
    rows = next(iter(measured_results("nvidia/nemotron-colembed-vl-4b-v2").values()))["rows"]
    int4 = next(r for r in rows if r["label"] == "int4")
    assert int4["retention"] < 0.1
    assert "int4" not in {r["label"] for r in measured_shortlist("nvidia/nemotron-colembed-vl-4b-v2", 0.9)}


# ---------------------------------------------------------------------------- optivision models


def test_the_models_command_lists_every_model_and_filters_by_status():
    runner = CliRunner()
    listed = json.loads(runner.invoke(app, ["models", "--json"]).output)
    assert {m["model_id"] for m in listed} == set(MODELS)
    measured = json.loads(runner.invoke(app, ["models", "--status", "measured", "--json"]).output)
    assert {m["model_id"] for m in measured} == {m.model_id for m in supported_models(MEASURED)}
    assert {m["model_id"] for m in measured if m["measured_rows"]} == set(measured_models())
    assert runner.invoke(app, ["models"]).exit_code == 0
    assert runner.invoke(app, ["models", "--status", "blessed"]).exit_code == 2
    bad = runner.invoke(app, ["models", "vidore/colpali-v1.3", "--min-retention", "2"])
    assert bad.exit_code == 2 and "--min-retention" in bad.output and "Traceback" not in bad.output


def test_the_nemotron_4b_entry_declares_its_licence_and_the_revision_its_loader_pins():
    from optivision.encoders.nemotron import KNOWN

    info = MODELS["nvidia/nemotron-colembed-vl-4b-v2"]
    assert info.license == "cc-by-nc-4.0" and info.revision == KNOWN[info.model_id][1]


def test_the_models_command_shows_one_model_with_its_caveat():
    runner = CliRunner()
    shown = runner.invoke(app, ["models", "vidore/colqwen2.5-v0.2"])
    assert shown.exit_code == 0 and "in-sample" in shown.output and "certify" in shown.output
    detail = json.loads(runner.invoke(app, ["models", "vidore/colqwen2.5-v0.2", "--json"]).output)
    assert detail["status"] == "measured" and detail["shortlist"] and len(detail["measured"]) == 2
    unmeasured = runner.invoke(app, ["models", "Metric-AI/ColQwen2.5-7b-multilingual-v1.0"])
    assert unmeasured.exit_code == 0 and "No fixed-pipeline measurements" in unmeasured.output
    assert runner.invoke(app, ["models", "nobody/nothing"]).exit_code == 2


# --------------------------------------------------------------------------- encode_and_certify


class _Adapter(EncoderAdapter):
    """Returns seeded synthetic vectors for whatever it is asked to encode."""

    def __init__(self, n_docs: int, n_queries: int) -> None:
        self.docs = make_corpus(n_docs=n_docs, dim=16, seed=3)
        self.queries, self.qrels = make_queries(self.docs, n_queries=n_queries, seed=4)
        self.calls: list[str] = []

    def identity(self) -> EncoderIdentity:
        return EncoderIdentity(model_id="test/synthetic", revision="r1", dimension=16, modality="text")

    def _corpus(self, source, ids, name):
        arrays = [np.asarray(source[i].vectors) for i in range(len(source))]
        return corpus_from_arrays(arrays, ids, identity=self.identity(), name=name)

    def encode_documents(self, documents, ids=None):
        self.calls.append(f"documents:{len(documents)}")
        return self._corpus(self.docs, ids or self.docs.ids, "documents")

    def encode_queries(self, queries, ids=None):
        self.calls.append(f"queries:{len(queries)}")
        return self._corpus(self.queries, ids or self.queries.ids, "queries")


def _space():
    return {"exact": [Pipeline([Float32Quantizer()])], "float16": [Pipeline([Float16Quantizer()])],
            "int8": [Pipeline([Int8Quantizer("per_vector")])]}


def test_encode_and_certify_encodes_then_certifies_and_records_the_model():
    adapter = _Adapter(n_docs=30, n_queries=120)
    result = encode_and_certify(adapter, ["d"] * 30, ["q"] * 120, adapter.qrels, target=0.9,
                                space=_space(), created="2026-10-12T00:00:00Z")
    assert adapter.calls == ["documents:30", "queries:120"]
    assert result.outcome in (Outcome.SELECT, Outcome.ABSTAIN)
    recorded = result.certificate.to_dict()
    assert "test/synthetic" in json.dumps(recorded) and '"r1"' in json.dumps(recorded)
    same = optivision.certify(adapter.encode_documents([], None), adapter.encode_queries([], None),
                              adapter.qrels, target=0.9, space=_space(), created="2026-10-12T00:00:00Z")
    assert same.certificate.digest == result.certificate.digest


def test_encode_and_certify_refuses_what_does_not_belong_to_it():
    adapter = _Adapter(n_docs=6, n_queries=6)
    with pytest.raises(TypeError, match="model id or an EncoderAdapter"):
        encode_and_certify(42, [], [])
    with pytest.raises(TypeError, match="adapter_options"):
        encode_and_certify(adapter, [], [], adapter_options={"device": "cpu"})
    with pytest.raises(TypeError, match="'encoder'"):
        encode_and_certify(adapter, ["d"], ["q"], encoder={"model_id": "x"})


def test_encode_and_certify_builds_the_adapter_of_a_model_id(monkeypatch):
    from optivision import adapters

    adapter = _Adapter(n_docs=30, n_queries=120)
    seen = {}

    def fake_load(model_id, **options):
        seen.update(model_id=model_id, **options)
        return adapter

    monkeypatch.setattr(adapters, "load_adapter", fake_load)
    result = encode_and_certify("org/model", ["d"] * 30, ["q"] * 120, adapter.qrels, target=0.9,
                                space=_space(), adapter_options={"device": "cpu"})
    assert seen == {"model_id": "org/model", "device": "cpu"} and result.certificate.digest


# ------------------------------------------------------------------------------ the models page


@pytest.mark.skipif(not (ROOT / "scripts" / "build_models_doc.py").exists()
                    or not (ROOT / "docs" / "package" / "guide").exists(),
                    reason="needs the repository's scripts and docs (not in an installed wheel)")
def test_the_models_page_equals_what_the_registry_and_the_catalogue_give():
    spec = importlib.util.spec_from_file_location("build_models_doc", ROOT / "scripts" / "build_models_doc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    page = (ROOT / "docs" / "package" / "guide" / "MODELS.md").read_text(encoding="utf-8")
    assert page.replace("\r\n", "\n") == module.render()
    for model_id in MODELS:
        assert f"`{model_id}`" in page


# ------------------------------------------------------------------ 0.4.1 regressions


def test_encode_and_certify_records_the_identity_of_an_adapter_that_does_not_write_it():
    class _Silent(_Adapter):
        def _corpus(self, source, ids, name):  # like a 0.4.0 page / sentence-transformers adapter
            arrays = [np.asarray(source[i].vectors) for i in range(len(source))]
            return corpus_from_arrays(arrays, ids, name=name)

    adapter = _Silent(n_docs=30, n_queries=120)
    assert "encoder_identity" not in adapter.encode_documents([], None).attrs
    result = encode_and_certify(adapter, ["d"] * 30, ["q"] * 120, adapter.qrels, target=0.9,
                                space=_space(), created="2026-10-12T00:00:00Z")
    recorded = json.dumps(result.certificate.to_dict())
    assert "test/synthetic" in recorded and '"r1"' in recorded


def test_the_models_listing_never_cuts_a_model_id_at_80_columns(monkeypatch):
    from rich.console import Console

    from optivision import cli_models

    monkeypatch.setattr(cli_models, "console", Console(width=80, force_terminal=False))
    with cli_models.console.capture() as captured:
        cli_models._listing(None, False)
    output = captured.get()
    assert "…" not in output
    for model_id in MODELS:
        assert model_id in output
    wide = Console(width=200, force_terminal=False)
    monkeypatch.setattr(cli_models, "console", wide)
    with wide.capture() as captured:
        cli_models._listing(None, False)
    assert "colpali-engine:colqwen3.5" in captured.get()


def test_measured_models_without_rows_explain_how_they_were_measured():
    """M4: no "measured" label followed by "No fixed-pipeline measurements ... are stored"."""
    runner = CliRunner(env={"COLUMNS": "200"})
    for model_id in NO_ROWS:
        out = runner.invoke(app, ["models", model_id])
        assert out.exit_code == 0 and "measured)" in out.output
        assert "No fixed-pipeline measurements" not in out.output
    twin = runner.invoke(app, ["models", "vidore/colqwen2-v1.0"]).output
    assert "Measured via equivalence with vidore/colqwen2-v1.0-merged" in " ".join(twin.split())
    detail = json.loads(runner.invoke(app, ["models", "vidore/colqwen2-v1.0", "--json"]).output)
    assert detail["measured_via"] == "vidore/colqwen2-v1.0-merged"
    smol = " ".join(runner.invoke(app, ["models", "vidore/colSmol-256M"]).output.split())
    assert "Measured outside the fixed-pipeline study" in smol
    assert json.loads(runner.invoke(app, ["models", "vidore/colSmol-256M", "--json"]).output)["measured_via"] is None
    never = next(m for m, info in MODELS.items() if info.status != MEASURED and m not in measured_models())
    unmeasured = runner.invoke(app, ["models", never]).output
    assert "No fixed-pipeline measurements" in unmeasured


def test_measured_via_names_the_sibling_whose_rows_apply():
    """L6: the CLI and MODELS.md explain measured models without rows by one rule."""
    from optivision.adapters.measured import _measured_via

    assert _measured_via("vidore/colqwen2-v1.0") == "vidore/colqwen2-v1.0-merged"
    assert _measured_via("vidore/colSmol-256M") is None  # measured, but outside the fixed-pipeline study
    assert _measured_via("vidore/colqwen2-v1.0-merged") is None  # has rows of its own


def test_the_builder_refuses_label_sets_that_differ_across_datasets():
    """L7: a label present in one dataset only would make measured_shortlist drop it silently."""
    builder = _builder()
    rows = [{"label": "float16"}, {"label": "int8"}]
    builder.check_labels({"m": {"a": {"rows": rows}, "b": {"rows": list(rows)}}})
    with pytest.raises(ValueError, match="label sets differ"):
        builder.check_labels({"m": {"a": {"rows": rows}, "b": {"rows": [{"label": "float16"}, {"label": "x"}]}}})

