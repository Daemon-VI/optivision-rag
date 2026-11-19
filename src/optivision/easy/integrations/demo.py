"""A local search box for an ``optivision.easy`` Index (Gradio, extra ``easy-ui``).

    index.demo()                                  # = optivision.easy.integrations.demo.launch(index)

One text box, the top pages as images with their file, page and score, and the index's evidence lines
with their tags. It runs on ``127.0.0.1`` and is meant for a person trying their own index.

Searches are never logged:

* the search bypasses the Index's opt-in query log (``index.search(..., log=False)`` when the Index takes
  it, else the log is paused for the call);
* nothing here prints, writes or keeps a query;
* Gradio's own records are switched off: telemetry (``analytics_enabled=False`` and
  ``GRADIO_ANALYTICS_ENABLED=False``), the per-browser run history (``run_history=False``, which also
  deletes runs an earlier version saved), the ``/monitoring`` endpoint and the MCP server.

``share=True`` asks Gradio for a public link, which sends every search through Gradio's share servers;
we cannot say what those keep, and the launch line says so. Page images are rendered from the source
files (``Hit.image()``); an index built from vectors has none, and its pages are listed without images.
"""

from __future__ import annotations

import os
from typing import Any

from .._claim import Block, Line, claim, lit
from .._types import EasyError
from ._common import evidence_lines, missing_extra, search_unlogged

__all__ = ["build_demo", "launch", "results"]

_HEADERS = ["page id", "file", "page", "score", "image"]


def _gradio() -> Any:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # before the import: no telemetry
    try:
        import gradio as gr
    except ImportError as exc:
        raise missing_extra("index.demo()", "gradio", "easy-ui") from exc
    return gr


def _name(hit: Any) -> str:
    if getattr(hit, "file", None):
        return os.path.basename(str(hit.file))
    return str(hit.id)


def caption(hit: Any) -> str:
    """``report.pdf, page 3, score 12.345  [counted]``: every number tagged (``_claim``)."""
    score = claim(float(hit.score), "counted", construct="MaxSim score", text=f"{float(hit.score):.3f}")
    if getattr(hit, "page", None) is not None:
        line = Line("", "{0}, page {1}, score {2}", lit(_name(hit)), claim(int(hit.page), "counted"), score)
    else:
        line = Line("", "{0}, score {1}", lit(_name(hit)), score)
    return line.render()


def _image(hit: Any, dpi: int) -> tuple[Any, str]:
    try:
        return hit.image(dpi=dpi), "shown"
    except (EasyError, ImportError, OSError, ValueError) as exc:
        return None, f"not shown: {exc}"


def results(index: Any, query: str, k: int = 5, *, dpi: int = 100) -> tuple[list[tuple[Any, str]], list[list[Any]]]:
    """``(gallery, table)`` for one query: images with captions, and one table row per hit."""
    query = (query or "").strip()
    if not query:
        return [], []
    hits = search_unlogged(index, query, k)
    gallery, table = [], []
    for hit in hits:
        image, status = _image(hit, dpi)
        text = caption(hit)
        if image is not None:
            gallery.append((image, text))
        table.append([str(hit.id), "" if hit.file is None else str(hit.file),
                      "" if hit.page is None else str(hit.page), text.rsplit("score ", 1)[-1], status])
    return gallery, table


def header(index: Any) -> Block:
    """What the page says above the search box (every number through ``_claim``)."""
    b = Block([], next="type a question and press Search; close the tab and press Ctrl+C in the terminal to stop")
    n = getattr(index, "n_pages", None)
    layout = getattr(index, "layout", None)
    plain = getattr(layout, "plain", None) or getattr(layout, "name", None) or "not recorded"
    if n is not None:
        b.add("Index", "{0} pages; layout {1}", claim(int(n), "counted"), lit(plain))
    else:
        b.add("Index", "layout {0}", lit(plain))
    b.add("Privacy", "searches typed here are not logged, printed or saved")
    return b


def build_demo(index: Any, *, k: int = 5, title: str = "OptiVision search", dpi: int = 100) -> Any:
    """The Gradio ``Blocks`` app, not launched (tests and notebooks use it directly)."""
    gr = _gradio()
    head = header(index)
    evidence = evidence_lines(getattr(index, "evidence", None))
    with gr.Blocks(title=title, analytics_enabled=False) as app:
        gr.Markdown("```\n" + "\n".join([str(head)] + evidence) + "\n```")
        box = gr.Textbox(label="Question", placeholder="What drove operating costs in 2023?", lines=1)
        button = gr.Button("Search", variant="primary")
        gallery = gr.Gallery(label="Pages", columns=3, height="auto", allow_preview=True)
        table = gr.Dataframe(headers=_HEADERS, interactive=False, wrap=True)

        def run(query: str) -> tuple[list[tuple[Any, str]], list[list[Any]]]:
            return results(index, query, k, dpi=dpi)

        button.click(run, inputs=[box], outputs=[gallery, table], show_progress="minimal")
        box.submit(run, inputs=[box], outputs=[gallery, table], show_progress="minimal")
    return app


def launch(index: Any, share: bool = False, *, k: int = 5, server_name: str = "127.0.0.1",
           server_port: int | None = None, inbrowser: bool = False, prevent_thread_lock: bool = False,
           dpi: int = 100) -> Any:
    """Start the search box (``index.demo()``). Returns the Gradio app; ``app.close()`` stops it."""
    app = build_demo(index, k=k, dpi=dpi)
    app.launch(share=share, server_name=server_name, server_port=server_port, inbrowser=inbrowser,
               prevent_thread_lock=True, quiet=True, show_error=True, run_history=False,
               enable_monitoring=False, mcp_server=False, footer_links=["gradio"])
    url = getattr(app, "share_url", None) if share else getattr(app, "local_url", None)
    b = Block([], next="open the address; app.close() or Ctrl+C stops it")
    b.add("Demo", "running at {0}; searches are not logged", lit(url or f"http://{server_name}"))
    if share:
        b.add("Share", "a public link sends every search through Gradio's share servers; we cannot say what they keep")
    print(b)
    if not prevent_thread_lock:
        app.block_thread()
    return app
