"""Bridges from an ``optivision.easy`` Index to other tools (FINAL_DESIGN 14, CONTRACT ``integrations/``).

``langchain``   ``OptiVisionRetriever``, a langchain-core ``BaseRetriever`` (extra ``langchain``)
``llamaindex``  ``OptiVisionRetriever``, a llama-index-core ``BaseRetriever`` (extra ``llamaindex``)
``qdrant``      ``export(index, client_or_url, collection)``: the stored layout as Qdrant multivectors
                (switched off until its equality test passes; ``presets/easy_ledger.json``)
``byaldi``      ``import_index(path)``: a byaldi index folder into an Index, on CPU (switched off until
                its format test passes)
``demo``        ``launch(index)``: a local Gradio search box (extra ``easy-ui``) that never logs searches

Importing this package, or any of its modules, imports none of those libraries; each one is imported
when it is needed, and a missing one raises ``MissingExtra`` with the install line. Every
document or node handed to a framework carries the page id, file, page, score and the index's evidence
tags in its metadata, so the tags travel with the results.
"""

from __future__ import annotations

__all__ = ["byaldi", "demo", "langchain", "llamaindex", "qdrant"]
