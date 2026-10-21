"""``optivision.easy.load`` and the ``Index`` surface (FINAL_DESIGN 4.3, 4.4).

The modules ``Index`` delegates to (``_check``, ``_upgrade``, ``_report``, the integrations) and the
ones the build calls are contract-shaped fakes (``tests/easy_build_fakes.py``).
"""

from __future__ import annotations

import shutil
import types
from pathlib import Path

import numpy as np
import pytest

from optivision.easy import _build
from optivision.easy._types import (
    EasyError,
    FeatureOff,
    Hit,
    LayoutRow,
    NoSource,
    Selection,
    SourceMissing,
    SyncReport,
)
from optivision.testing.synthetic import make_corpus, make_queries

from . import easy_build_fakes as F

MODEL = "fake/page-model"


@pytest.fixture
def fakes(monkeypatch):
    return F.install(monkeypatch, encode=F.make_encode({MODEL: F.stub_adapter()}))


@pytest.fixture
def ov(fakes):
    import optivision.easy as ov

    return ov


def _arrays(corpus):
    return [np.asarray(corpus.vectors[lo:hi]) for lo, hi in zip(corpus.offsets[:-1], corpus.offsets[1:])]


@pytest.fixture
def pdf_index(ov, tmp_path):
    root = tmp_path / "my_pdfs"
    F.make_pdf(root / "report.pdf", F.SENTENCES[:3])
    F.make_pdf(root / "sub" / "notes.pdf", F.SENTENCES[3:5])
    index = ov.build(root, model=MODEL, progress=False)
    index.close()
    return index.path, root


def test_load_loads_the_model_only_when_a_string_query_needs_it(ov, fakes, pdf_index):
    path, _ = pdf_index
    loads = len(fakes["_encode"].loads)
    index = ov.load(path)
    assert len(fakes["_encode"].loads) == loads  # opening loads no model
    hits = index.search("dividend policy shareholders", k=2)
    assert len(fakes["_encode"].loads) == loads + 1
    assert fakes["_encode"].loads[-1]["model"] == MODEL and fakes["_encode"].loads[-1]["dtype"] == "float32"
    assert hits[0].id == "my_pdfs/report.pdf::p2" and hits[0].page == 2
    index.search("again")
    assert len(fakes["_encode"].loads) == loads + 1  # once per Index
    assert index.record_differences == []


def test_load_names_differences_from_the_build_record(ov, fakes, pdf_index, capsys):
    path, _ = pdf_index
    fakes["_encode"].dtype = "float16"
    index = ov.load(path)
    index.search("freight")
    assert index.record_differences == ["dtype: 'float32' at build, 'float16' now"]
    out = capsys.readouterr().out
    assert "The model now differs from the build record" in out and "Next: index.check()" in out


def test_search_takes_a_string_or_a_list(ov, pdf_index):
    index = ov.load(pdf_index[0])
    one = index.search("freight prices doubled", k=3)
    many = index.search(["freight prices doubled", "warehouse automation picking"], k=3)
    assert isinstance(one, list) and all(isinstance(h, Hit) for h in one)
    assert [h.id for h in many[0]] == [h.id for h in one]
    assert many[1][0].id == "my_pdfs/sub/notes.pdf::p1"
    assert len(one) == 3 and one[0].score >= one[1].score >= one[2].score
    with pytest.raises(TypeError):
        index.search(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        index.search_vectors(np.ones((2, F.DIM), np.float32), k=0)


def test_hit_image_source_missing_and_relink(ov, tmp_path, pdf_index):
    path, root = pdf_index
    index = ov.load(path)
    hit = index.search("freight prices doubled")[0]
    assert hit.image().size[0] > 0
    moved = tmp_path / "elsewhere" / "deep"
    moved.mkdir(parents=True)
    shutil.move(str(root / "report.pdf"), str(moved / "renamed.pdf"))
    with pytest.raises(SourceMissing, match="relink"):
        index.search("freight prices doubled")[0].image()
    found = index.relink(tmp_path / "elsewhere")
    assert found == {"my_pdfs/report.pdf": str((moved / "renamed.pdf").resolve())}
    hit = index.search("freight prices doubled")[0]
    assert Path(hit.file) == (moved / "renamed.pdf").resolve() and hit.image().size[0] > 0
    assert ov.load(path).search("freight prices doubled")[0].file == hit.file  # recorded in the manifest


def test_add_is_a_sync(ov, tmp_path, pdf_index):
    path, _ = pdf_index
    more = tmp_path / "more_pdfs"
    F.make_pdf(more / "report.pdf", [F.SENTENCES[5]])
    index = ov.load(path)
    report = index.add(more)
    assert isinstance(report, SyncReport)
    assert [Path(p).name for p, _ in report.added] == ["report.pdf"] and report.n_pages == 6
    assert index.n_pages == 6
    assert index.search("employee training safety review")[0].id == "more_pdfs/report.pdf::p1"
    assert index.add(more).added == []  # again: nothing new


def test_check_upgrade_report_demo_and_qdrant_delegate(ov, fakes, pdf_index, tmp_path):
    index = ov.load(pdf_index[0])
    assert index.check() == "check report"
    args, kwargs = fakes["_check"].calls[-1]
    assert args == (index, None) and kwargs == {"answers": None, "drift": "auto"}
    assert index.check(["q"], answers=["a"], drift="off") == "check report"
    assert fakes["_check"].calls[-1] == ((index, ["q"]), {"answers": ["a"], "drift": "off"})
    assert index.upgrade(["q"] * 3, apply=True) == "upgrade report"
    assert fakes["_upgrade"].calls[-1] == ((index, ["q"] * 3), {"answers": None, "apply": True})
    rep = index.report()
    titles = [t for t, _ in rep.sections]
    assert titles[:2] == ["Summary", "Layouts at this target"] and "Size" in titles and "Build record" in titles
    assert rep.meta["model"] == MODEL and rep.meta["n_pages"] == 5
    for _, block in rep.sections:
        assert block.next
        for line in block.lines:
            assert not line.claims or line.render().endswith(line.tag)
    assert rep.save(tmp_path / "r.md").read_text(encoding="utf-8").startswith("Summary")
    assert index.demo() == "demo"
    assert fakes["integrations.demo"].calls[-1][0] == (index,)
    with pytest.raises(FeatureOff, match="Qdrant export is switched off"):
        index.export_qdrant(url="http://localhost:6333")


def test_report_uses_from_index_when_the_report_module_has_it(ov, fakes, pdf_index):
    class R:
        @classmethod
        def from_index(cls, index):
            return ("from_index", index.path)

    fakes["_report"].Report = R
    index = ov.load(pdf_index[0])
    assert index.report() == ("from_index", index.path)


def test_attributes(ov, pdf_index):
    index = ov.load(pdf_index[0])
    assert index.target == 0.97 and isinstance(index.layout, LayoutRow)
    assert isinstance(index.selection, Selection) and index.selection.pick.name == index.layout.name
    assert index.n_pages == 5 and index.size == index._store.sizes()["total"] > 0
    assert isinstance(index.examples, list)
    assert index.evidence and all(set(c) >= {"value", "kind", "tag"} for c in index.evidence)
    assert index.pipeline.fitted and hasattr(index.retriever, "search")
    assert index.record["model"] == MODEL and index.path == pdf_index[0]
    assert index._manifest["kind"] == _build.MANIFEST_KIND
    assert "5 pages" in repr(index) and repr(index).endswith("[counted]")


def test_summary_survives_a_reload_with_its_tags(ov, tmp_path):
    corpus = make_corpus(12, 8, seed=0)
    built = ov.build(_arrays(corpus), path=tmp_path / "v.ovx", progress=False)
    loaded = ov.load(tmp_path / "v.ovx")
    assert [ln.render() for ln in loaded.summary.lines] == [ln.render() for ln in built.summary.lines]
    assert loaded.summary.next == built.summary.next
    assert str(loaded) == str(built.summary)


def test_close_and_context_manager(ov, pdf_index):
    with ov.load(pdf_index[0]) as index:
        assert index.search("freight")
    with pytest.raises(ValueError, match="closed"):
        index.search("freight")
    assert "closed" in repr(index)
    index.close()  # twice is fine


def test_an_index_from_vectors(ov, tmp_path):
    corpus = make_corpus(20, 8, seed=1)
    index = ov.build(_arrays(corpus), path=tmp_path / "v.ovx", progress=False)
    with pytest.raises(EasyError, match="search_vectors"):
        index.search("text needs a model")
    queries, _ = make_queries(corpus, 3, seed=2)
    hit = index.search_vectors(queries)[0][0]
    assert hit.file is None and hit.page is None
    with pytest.raises(NoSource):
        hit.image()
    one = index.search_vectors(np.asarray(queries.vectors[queries.offsets[0]:queries.offsets[1]]))
    assert len(one) == 1 and one[0][0].id == hit.id  # one (n, d) array is one query


def test_explain_prints_code_that_reproduces_the_search(ov, tmp_path, capsys):
    corpus = make_corpus(30, 16, seed=4)
    ids = [f"page{i}" for i in range(30)]
    index = ov.build(_arrays(corpus), ids=ids, path=tmp_path / "v.ovx", progress=False)
    code = index.explain()
    out = capsys.readouterr().out
    assert "Expert code for this index" in out and out.rstrip().endswith("Next: index.report()")
    assert "Pipeline.from_dict(" in code and "CompressedRetriever(" in code
    queries, _ = make_queries(corpus, 4, seed=5)
    scope = {"page_vectors": _arrays(corpus), "page_ids": ids, "query_vectors": queries}
    exec(compile(code, "explain", "exec"), scope)  # noqa: S102  (our own generated code)
    expected = [[h.id for h in r] for r in index.search_vectors(queries, k=5)]
    assert scope["ranking"].ids == expected


def test_explain_names_the_certify_call_when_larger_layouts_are_open(ov, monkeypatch, tmp_path):
    F.install(monkeypatch, levels=F.make_levels(**{"hierarchical_merge(0.25) > int8(per_vector)": 0.95}))
    index = ov.build(_arrays(make_corpus(10, 8, seed=0)), path=tmp_path / "v.ovx", progress=False)
    code = index._explain_code()
    assert index.layout.name == "int4(mean)"
    assert "certify(pages, query_vectors, None, target=0.97, error=0.05 * 2**-0," in code
    assert "RankingReference(depth=1)" in code and "CandidateSpace.from_families({" in code
    assert "'8-bit': [Pipeline.from_dict(" in code and "fallback=pipeline, order_seed=0" in code


def test_query_log_is_opt_in(ov, fakes, monkeypatch, pdf_index, capsys):
    records = []
    log = types.SimpleNamespace(record=lambda q, hits: records.append((q, [h.id for h in hits])))
    qlog = types.ModuleType("optivision.easy._querylog")
    qlog.open_log = lambda index: log
    monkeypatch.setitem(__import__("sys").modules, "optivision.easy._querylog", qlog)
    ov.load(pdf_index[0]).search("freight")
    assert records == []  # never on by default
    logged = ov.load(pdf_index[0], log_queries=True)
    assert "on this machine only" in capsys.readouterr().out
    logged.search(["freight", "dividend"])
    assert [q for q, _ in records] == ["freight", "dividend"]


def test_load_refuses_what_is_not_an_index(ov, tmp_path):
    with pytest.raises(FileNotFoundError):
        ov.load(tmp_path / "nothing.ovx")


def test_public_names_resolve_lazily(ov):
    assert ov.build is __import__("optivision.easy._api", fromlist=["build"]).build
    assert ov.Index.__name__ == "Index" and callable(ov.load)
