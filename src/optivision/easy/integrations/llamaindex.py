"""A LlamaIndex retriever over an ``optivision.easy`` Index (extra ``llamaindex``).

    from optivision.easy.integrations.llamaindex import OptiVisionRetriever
    retriever = OptiVisionRetriever(ov.load("my_pdfs.ovx"), k=4)
    nodes = retriever.retrieve("What drove operating costs in 2023?")     # list[NodeWithScore]

Each ``NodeWithScore`` wraps a ``TextNode`` whose ``id_`` is the page id, whose text is the page's text
layer (``""`` if none was kept; the page image is ``Hit.image()``) and whose metadata holds ``file``,
``page`` (1-based), ``score``, ``rank``, ``source`` and the index's evidence (``optivision_layout``,
``optivision_target``, ``optivision_evidence``, ``optivision_tags``, ``optivision_status``). The evidence
keys are excluded from the text LlamaIndex shows an LLM or an embedding model, so they travel with the
node without entering prompts. ``NodeWithScore.score`` is the MaxSim inner product over the stored
layout: it orders this index's pages and means nothing across indexes.

The retriever calls ``Index.search``, so a query log the user switched on records these searches too;
it is off by default. llama-index-core is imported when ``OptiVisionRetriever`` is used; without it
the name raises ``MissingExtra`` with the install line.
"""

from __future__ import annotations

from typing import Any

from ._common import INDEX_KEYS, documents, missing_extra

__all__ = ["OptiVisionRetriever", "as_retriever", "retriever_class"]

#: annotated, never assigned: the name is resolved by the module ``__getattr__`` below
OptiVisionRetriever: Any

_CLASS: Any = None


def _build_class() -> Any:
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    except ImportError as exc:
        raise missing_extra("OptiVisionRetriever", "llama-index-core", "llamaindex") from exc

    class OptiVisionRetriever(BaseRetriever):
        """Top ``k`` pages of an ``optivision.easy`` Index for a text query, as LlamaIndex nodes."""

        def __init__(self, index: Any, k: int = 4, **kwargs: Any) -> None:
            self._ov_index = index
            self._k = int(k)
            super().__init__(**kwargs)

        @property
        def index(self) -> Any:
            return self._ov_index

        @property
        def k(self) -> int:
            return self._k

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            out = []
            for d in documents(self._ov_index, query_bundle.query_str, self._k):
                node = TextNode(text=d["page_content"], id_=d["id"], metadata=d["metadata"],
                                excluded_llm_metadata_keys=list(INDEX_KEYS),
                                excluded_embed_metadata_keys=list(INDEX_KEYS))
                out.append(NodeWithScore(node=node, score=d["score"]))
            return out

    OptiVisionRetriever.__module__ = __name__
    OptiVisionRetriever.__qualname__ = "OptiVisionRetriever"
    return OptiVisionRetriever


def retriever_class() -> Any:
    """The ``OptiVisionRetriever`` class, built when it is needed (raises ``MissingExtra`` without llama-index-core)."""
    global _CLASS
    if _CLASS is None:
        _CLASS = _build_class()
    return _CLASS


def as_retriever(index: Any, k: int = 4, **kwargs: Any) -> Any:
    """``OptiVisionRetriever(index, k=k, **kwargs)``."""
    return retriever_class()(index, k=k, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "OptiVisionRetriever":
        return retriever_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
