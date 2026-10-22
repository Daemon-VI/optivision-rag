"""LangChain and LlamaIndex retrievers, the Gradio demo and the shared helpers (``optivision.easy.integrations``).

The development environment has neither langchain-core nor llama-index-core, so each retriever is
checked twice: against small stand-ins of the framework classes (always), and against the real classes
when the framework is installed (``pytest.importorskip``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from optivision.easy import _ledger
from optivision.easy._claim import Block, Line, claim, lit
from optivision.easy._types import MissingExtra
from optivision.easy.integrations import _common
from optivision.easy.integrations import langchain as lc
from optivision.easy.integrations import llamaindex as li
from optivision.testing.synthetic import make_corpus, make_queries

from ._easy_index_fakes import (
    FakeIndex,
    FakeStore,
    evidence_block,
    hashing_encoder,
    layout_row,
    page_ids,
    segmented,
    with_ids,
)
from .test_docs_examples import BANNED_PHRASES, BANNED_WORDS

ROOT = Path(__file__).resolve().parents[1]
SPEC = {"stages": [{"stage": "int8", "scale": "per_vector"}]}
TEXTS = [f"question {i}" for i in range(6)]
#: the brief's words, beyond test_docs_examples' lists
BRIEF_WORDS = ("lossless", "universal", "guaranteed", "first", "sota", "state of the art", "veto")


def _index(files: bool = True, **kw) -> FakeIndex:
    raw = make_corpus(40, 24, tokens=(5, 12), seed=31)
    corpus = with_ids(raw, page_ids(40))
    queries, _ = make_queries(corpus, len(TEXTS), tokens=(3, 5), noise=0.4, seed=32)
    segments, pipeline = segmented(corpus, SPEC, (25, 15))
    texts = {corpus.ids[i]: f"text of page {i + 1}" for i in range(0, 40, 2)}
    fmap = {i: "C:/data/my_pdfs/report.pdf" for i in corpus.ids} if files else {}
    return FakeIndex(FakeStore(segments, pipeline, texts), encode=hashing_encoder(queries, TEXTS), files=fmap,
                     layout=layout_row(SPEC, name="int8(per_vector)", plain="8-bit"), evidence=evidence_block(), **kw)


def _clean(text: str) -> None:
    low = text.lower()
    for phrase in BANNED_PHRASES + BRIEF_WORDS:
        assert phrase not in low, phrase
    assert BANNED_WORDS.search(text) is None
    assert text.isascii()


# ---------------------------------------------------------------------------- light import


def test_importing_the_integrations_loads_no_framework():
    code = (
        "import sys\n"
        "import optivision.easy.integrations.langchain, optivision.easy.integrations.llamaindex\n"
        "import optivision.easy.integrations.qdrant, optivision.easy.integrations.byaldi\n"
        "import optivision.easy.integrations.demo, optivision.easy._tiered\n"
        "heavy = ['langchain_core', 'llama_index', 'gradio', 'torch', 'qdrant_client', 'transformers', 'scipy']\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n"
    )
    env = {"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


# ---------------------------------------------------------------------------- shared helpers


def test_evidence_lines_and_tags_from_every_shape():
    block = evidence_block()
    lines = _common.evidence_lines(block)
    assert lines[0].startswith("Layout") and lines[0].endswith("[our benchmarks: ViDoRe V1 DocVQA 500 pages]")
    assert lines[-1] == "Next: index.search(...)"
    assert _common.evidence_tags(block) == ["[our benchmarks: ViDoRe V1 DocVQA 500 pages]", "[counted]"]
    c = claim(3, "counted")
    assert _common.evidence_tags([c, c]) == ["[counted]"]
    assert _common.evidence_lines([Line("Pages", "{0} pages", c)]) == ["Pages      3 pages  [counted]"]
    assert _common.evidence_lines({"size": c}) == ["size: 3  [counted]"]
    assert _common.evidence_lines("a\n\nb") == ["a", "b"] and _common.evidence_tags("a") == []
    assert _common.evidence_lines(None) == [] and _common.evidence_tags(None) == []


def test_page_ids_split_into_document_and_page():
    assert _common.split_page_id("my_pdfs/sub/a.pdf::p12") == ("my_pdfs/sub/a.pdf", 12)
    assert _common.split_page_id("7") == ("7", None)
    assert _common.split_page_id("odd::pX") == ("odd::pX", None)


def test_documents_carry_page_score_and_evidence_as_json():
    index = _index()
    docs = _common.documents(index, TEXTS[0], 4)
    assert len(docs) == 4
    m = docs[0]["metadata"]
    assert m["id"] == m["page_id"] == docs[0]["id"] and m["file"] == "C:/data/my_pdfs/report.pdf"
    assert m["page"] == int(m["id"].rsplit("::p", 1)[1]) and m["rank"] == 1 and m["source"] == m["file"]
    assert m["optivision_layout"] == "int8(per_vector)" and m["optivision_layout_plain"] == "8-bit"
    assert m["optivision_target"] == 0.97 and m["optivision_status"] == _ledger.status()
    assert "[our benchmarks: ViDoRe V1 DocVQA 500 pages]" in m["optivision_tags"]
    assert [d["metadata"]["score"] for d in docs] == sorted((d["metadata"]["score"] for d in docs), reverse=True)
    json.dumps(docs)
    vectors_only = _common.documents(_index(files=False), TEXTS[0], 2)[0]["metadata"]
    assert vectors_only["file"] is None and vectors_only["page"] is None and vectors_only["source"] == vectors_only["id"]
    with pytest.raises(TypeError):
        _common.documents(index, ["two", "queries"], 3)


def test_missing_extra_names_the_install_line():
    err = _common.missing_extra("Thing", "pkg", "extra")
    assert isinstance(err, ImportError) and str(err) == 'Thing needs pkg. Install it with: pip install "optivision-rag[extra]"'


# ---------------------------------------------------------------------------- LangChain


@pytest.fixture
def fresh_langchain(monkeypatch):
    monkeypatch.setattr(lc, "_CLASS", None)
    yield
    lc._CLASS = None


def test_langchain_missing_raises_missing_extra(monkeypatch, fresh_langchain):
    for name in ("langchain_core", "langchain_core.callbacks", "langchain_core.documents", "langchain_core.retrievers"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingExtra, match=r'pip install "optivision-rag\[langchain\]"'):
        from optivision.easy.integrations.langchain import OptiVisionRetriever  # noqa: F401
    with pytest.raises(MissingExtra):
        lc.as_retriever(_index())
    with pytest.raises(AttributeError):
        lc.NotAName  # noqa: B018


def _fake_langchain(monkeypatch) -> None:
    pydantic = pytest.importorskip("pydantic")
    pkg = types.ModuleType("langchain_core")
    pkg.__path__ = []
    callbacks = types.ModuleType("langchain_core.callbacks")
    documents = types.ModuleType("langchain_core.documents")
    retrievers = types.ModuleType("langchain_core.retrievers")

    class CallbackManagerForRetrieverRun:
        pass

    class Document(pydantic.BaseModel):
        page_content: str
        metadata: dict = {}

    class BaseRetriever(pydantic.BaseModel):
        def invoke(self, query: str) -> list:
            return self._get_relevant_documents(query, run_manager=CallbackManagerForRetrieverRun())

    callbacks.CallbackManagerForRetrieverRun = CallbackManagerForRetrieverRun
    documents.Document = Document
    retrievers.BaseRetriever = BaseRetriever
    for name, mod in (("langchain_core", pkg), ("langchain_core.callbacks", callbacks),
                      ("langchain_core.documents", documents), ("langchain_core.retrievers", retrievers)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_langchain_retriever_on_stand_in_classes(monkeypatch, fresh_langchain):
    _fake_langchain(monkeypatch)
    index = _index()
    retriever = lc.as_retriever(index, k=3)
    docs = retriever.invoke(TEXTS[1])
    want = [h.id for h in index.search_vectors(index.encode([TEXTS[1]]), 3)[0]]
    assert [d.metadata["id"] for d in docs] == want
    assert docs[0].page_content == index.store.text(want[0])
    assert docs[0].metadata["optivision_tags"][0].startswith("[our benchmarks")
    cls = lc.OptiVisionRetriever
    assert cls.__name__ == "OptiVisionRetriever" and cls.__module__ == lc.__name__
    assert cls(index=index).k == 4


def test_langchain_retriever_on_the_real_classes(fresh_langchain):
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    from optivision.easy.integrations.langchain import OptiVisionRetriever

    assert issubclass(OptiVisionRetriever, BaseRetriever)
    index = _index()
    docs = OptiVisionRetriever(index=index, k=3).invoke(TEXTS[2])
    assert len(docs) == 3 and all(isinstance(d, Document) for d in docs)
    assert docs[0].metadata["page_id"] == index.search_vectors(index.encode([TEXTS[2]]), 1)[0][0].id
    assert "optivision_evidence" in docs[0].metadata


# ---------------------------------------------------------------------------- LlamaIndex


@pytest.fixture
def fresh_llamaindex(monkeypatch):
    monkeypatch.setattr(li, "_CLASS", None)
    yield
    li._CLASS = None


def test_llamaindex_missing_raises_missing_extra(monkeypatch, fresh_llamaindex):
    for name in ("llama_index", "llama_index.core", "llama_index.core.retrievers", "llama_index.core.schema"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingExtra, match=r'pip install "optivision-rag\[llamaindex\]"'):
        from optivision.easy.integrations.llamaindex import OptiVisionRetriever  # noqa: F401


def _fake_llamaindex(monkeypatch) -> None:
    pkg = types.ModuleType("llama_index")
    pkg.__path__ = []
    core = types.ModuleType("llama_index.core")
    core.__path__ = []
    retrievers = types.ModuleType("llama_index.core.retrievers")
    schema = types.ModuleType("llama_index.core.schema")

    class QueryBundle:
        def __init__(self, query_str: str) -> None:
            self.query_str = query_str

    class TextNode:
        def __init__(self, text: str, id_: str, metadata: dict, excluded_llm_metadata_keys=(),
                     excluded_embed_metadata_keys=()) -> None:
            self.text, self.id_, self.metadata = text, id_, metadata
            self.excluded_llm_metadata_keys = list(excluded_llm_metadata_keys)
            self.excluded_embed_metadata_keys = list(excluded_embed_metadata_keys)

    class NodeWithScore:
        def __init__(self, node: TextNode, score: float) -> None:
            self.node, self.score = node, score

    class BaseRetriever:
        def __init__(self, callback_manager=None, verbose: bool = False) -> None:
            self.callback_manager, self.verbose = callback_manager, verbose

        def retrieve(self, query):
            return self._retrieve(query if isinstance(query, QueryBundle) else QueryBundle(query))

    schema.QueryBundle, schema.TextNode, schema.NodeWithScore = QueryBundle, TextNode, NodeWithScore
    retrievers.BaseRetriever = BaseRetriever
    for name, mod in (("llama_index", pkg), ("llama_index.core", core), ("llama_index.core.retrievers", retrievers),
                      ("llama_index.core.schema", schema)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_llamaindex_retriever_on_stand_in_classes(monkeypatch, fresh_llamaindex):
    _fake_llamaindex(monkeypatch)
    index = _index()
    retriever = li.as_retriever(index, k=5, verbose=True)
    nodes = retriever.retrieve(TEXTS[3])
    want = index.search_vectors(index.encode([TEXTS[3]]), 5)[0]
    assert [n.node.id_ for n in nodes] == [h.id for h in want]
    assert [n.score for n in nodes] == [h.score for h in want]
    node = nodes[0].node
    assert node.metadata["file"] == "C:/data/my_pdfs/report.pdf" and "optivision_status" in node.metadata
    assert set(_common.INDEX_KEYS) <= set(node.excluded_llm_metadata_keys)
    assert set(_common.INDEX_KEYS) <= set(node.excluded_embed_metadata_keys)
    assert "file" not in node.excluded_llm_metadata_keys
    assert retriever.verbose and retriever.index is index and retriever.k == 5


def test_llamaindex_retriever_on_the_real_classes(fresh_llamaindex):
    pytest.importorskip("llama_index.core")
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore

    from optivision.easy.integrations.llamaindex import OptiVisionRetriever

    assert issubclass(OptiVisionRetriever, BaseRetriever)
    index = _index()
    nodes = OptiVisionRetriever(index, k=3).retrieve(TEXTS[4])
    assert len(nodes) == 3 and all(isinstance(n, NodeWithScore) for n in nodes)
    assert nodes[0].node.node_id == index.search_vectors(index.encode([TEXTS[4]]), 1)[0][0].id
    assert "optivision_tags" not in nodes[0].node.get_content(metadata_mode="llm")


# ---------------------------------------------------------------------------- the demo


def test_demo_results_show_images_captions_and_never_reach_the_log():
    from optivision.easy.integrations import demo

    index = _index()
    gallery, table = demo.results(index, TEXTS[0], k=3)
    assert index._querylog == []  # the opt-in log was on, and stayed empty
    assert len(gallery) == 3 and len(table) == 3
    image, text = gallery[0]
    assert image.size == (40, 52) and text.startswith("report.pdf, page ") and text.endswith("[counted]")
    assert table[0][4] == "shown" and table[0][0] == index.search_vectors(index.encode([TEXTS[0]]), 1)[0][0].id
    index.search(TEXTS[0], k=1)
    assert index._querylog == [TEXTS[0]]  # a normal search does log: the demo's bypass is what kept it empty


def test_demo_uses_the_log_keyword_when_the_index_has_one():
    from optivision.easy.integrations import demo

    index = _index(accept_log_keyword=True)
    demo.results(index, TEXTS[1], k=2)
    assert index._querylog == []


def test_demo_lists_pages_without_images_for_an_index_built_from_vectors():
    from optivision.easy.integrations import demo

    gallery, table = demo.results(_index(files=False), TEXTS[2], k=2)
    assert gallery == [] and len(table) == 2
    assert table[0][4].startswith("not shown") and table[0][1] == "" and table[0][2] == ""
    assert demo.results(_index(), "   ", k=2) == ([], [])


def test_demo_text_is_tagged_plain_and_ends_with_next():
    from optivision.easy.integrations import demo

    index = _index()
    head = str(demo.header(index))
    assert "40 pages; layout 8-bit  [counted]" in head and head.splitlines()[-1].startswith("Next: ")
    assert "not logged" in head
    _clean(head)
    hit = index.search_vectors(index.encode([TEXTS[0]]), 1)[0][0]
    _clean(demo.caption(hit))


def test_demo_needs_gradio_and_names_the_extra(monkeypatch):
    from optivision.easy.integrations import demo

    monkeypatch.setitem(sys.modules, "gradio", None)
    with pytest.raises(MissingExtra, match=r'pip install "optivision-rag\[easy-ui\]"'):
        demo.build_demo(_index())


def test_demo_builds_with_every_gradio_record_switched_off(monkeypatch, capsys):
    gr = pytest.importorskip("gradio")
    from optivision.easy.integrations import demo

    app = demo.build_demo(_index(), k=3)
    assert isinstance(app, gr.Blocks) and app.analytics_enabled is False
    assert __import__("os").environ["GRADIO_ANALYTICS_ENABLED"] == "False"

    seen = {}

    def fake_launch(self, **kwargs):
        seen.update(kwargs)
        self.local_url = "http://127.0.0.1:7860/"

    blocked = []
    monkeypatch.setattr(gr.Blocks, "launch", fake_launch)
    monkeypatch.setattr(gr.Blocks, "block_thread", lambda self: blocked.append(True))
    demo.launch(_index(), prevent_thread_lock=True)
    assert seen["run_history"] is False and seen["enable_monitoring"] is False and seen["mcp_server"] is False
    assert seen["share"] is False and seen["server_name"] == "127.0.0.1" and blocked == []
    out = capsys.readouterr().out
    assert "http://127.0.0.1:7860/" in out and "searches are not logged" in out and "Next: " in out
    demo.launch(_index(), share=True, prevent_thread_lock=False)
    assert blocked == [True] and "share servers" in capsys.readouterr().out


# ---------------------------------------------------------------------------- refusal messages are plain


def test_refusal_and_summary_messages_avoid_the_banned_words():
    from optivision.easy import _tiered
    from optivision.easy.integrations import byaldi, qdrant

    texts = [str(_common.missing_extra("X", "y", "z"))]
    for fn, arg in ((qdrant.plan, "uint8"), (qdrant.plan, "nope")):
        try:
            raw = make_corpus(4, 8, tokens=2, seed=1)
            from optivision.compose import Pipeline

            fn(Pipeline.from_dict({"stages": [{"stage": "binary"}]}).compress(raw), arg)
        except qdrant.ExportRefused as exc:
            texts.append(str(exc))
    try:
        _tiered.check_modality("text")
    except _tiered.SmallMemoryRefused as exc:
        texts.append(str(exc))
    for name in ("small_memory_layout", "qdrant_export", "byaldi_import"):
        texts.append(_ledger.feature(name)["title"])
    texts.append(byaldi.__doc__.split("\n\n")[0] + qdrant.__doc__.split("\n\n")[0])
    for t in texts:
        _clean(t)


def test_a_block_with_mixed_kinds_cannot_be_built_here_either():
    with pytest.raises(ValueError):
        Block([]).add("x", "{0} {1}", claim(1, "counted"), claim(0.9, "catalogue", source="s"))
    assert str(Block([], next="n").add("x", "{0}", lit("y"))) == "x          y\nNext: n"
