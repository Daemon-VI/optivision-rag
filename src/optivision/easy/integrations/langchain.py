"""A LangChain retriever over an ``optivision.easy`` Index (extra ``langchain``).

    from optivision.easy.integrations.langchain import OptiVisionRetriever
    retriever = OptiVisionRetriever(index=ov.load("my_pdfs.ovx"), k=4)
    docs = retriever.invoke("What drove operating costs in 2023?")

Each ``Document`` holds the page's text layer as ``page_content`` (``""`` if none was kept; the page
image is ``Hit.image()``) and, in ``metadata``: ``id`` / ``page_id``, ``file``, ``page`` (1-based),
``score``, ``rank``, ``source`` (the file, else the page id), and the index's evidence: ``optivision_layout``,
``optivision_target``, ``optivision_evidence`` (lines with their tags), ``optivision_tags``,
``optivision_status``. Scores are MaxSim inner products over the stored layout: they order this index's
pages and mean nothing across indexes.

The retriever calls ``Index.search``, so a query log the user switched on (``ov.load(...,
log_queries=True)``) records these searches too; it is off by default.

langchain-core is imported when ``OptiVisionRetriever`` is used; without it the name raises
``MissingExtra`` with the install line, and the rest of optivision is unaffected.
"""

from __future__ import annotations

from typing import Any

from ._common import documents, missing_extra

__all__ = ["OptiVisionRetriever", "as_retriever", "retriever_class"]

#: annotated, never assigned: the name is resolved by the module ``__getattr__`` below
OptiVisionRetriever: Any

_CLASS: Any = None


def _build_class() -> Any:
    try:
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:
        raise missing_extra("OptiVisionRetriever", "langchain-core", "langchain") from exc
    try:
        from pydantic import ConfigDict
    except ImportError as exc:  # pragma: no cover - langchain-core depends on pydantic 2
        raise missing_extra("OptiVisionRetriever", "pydantic 2", "langchain") from exc

    class OptiVisionRetriever(BaseRetriever):
        """Top ``k`` pages of an ``optivision.easy`` Index for a text query, as LangChain documents."""

        index: Any
        k: int = 4

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            return [Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in documents(self.index, query, self.k)]

    OptiVisionRetriever.__module__ = __name__
    OptiVisionRetriever.__qualname__ = "OptiVisionRetriever"
    return OptiVisionRetriever


def retriever_class() -> Any:
    """The ``OptiVisionRetriever`` class, built when it is needed (raises ``MissingExtra`` without langchain-core)."""
    global _CLASS
    if _CLASS is None:
        _CLASS = _build_class()
    return _CLASS


def as_retriever(index: Any, k: int = 4, **kwargs: Any) -> Any:
    """``OptiVisionRetriever(index=index, k=k, **kwargs)``."""
    return retriever_class()(index=index, k=k, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "OptiVisionRetriever":
        return retriever_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
