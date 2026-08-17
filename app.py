"""OptiVision RAG — Gradio demonstration for Hugging Face Spaces.

Single-page demo of the Stage-I pipeline:

    document page -> ColSmol -> visual tokens -> spatial pruning
                  -> redundancy pruning -> binary quantization -> KiB

This file is **presentation only**. Every number it shows is read off the real
arrays produced by ``optivision.demo.DemoPipeline``, which in turn calls the
research modules in ``src/optivision``. Nothing here recomputes or hard-codes a
result.

Run locally:

    python app.py
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))  # importable without `pip install -e .`

from optivision.config import Config
from optivision.demo import (
    DemoError,
    DemoPipeline,
    load_page,
    pick_device,
    store_in_qdrant,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("optivision.app")

CONFIG_PATH = os.environ.get("OPTIVISION_CONFIG", str(ROOT / "configs" / "colsmol.yaml"))

# ---------------------------------------------------------------- ZeroGPU ---
# `spaces` only exists on a Hugging Face Space. Locally the decorator has to be
# a no-op so the same file runs unchanged.
try:  # pragma: no cover - environment dependent
    import spaces

    GPU_DECORATOR = spaces.GPU(duration=120)
    ZERO_GPU = True
except Exception:  # noqa: BLE001

    def GPU_DECORATOR(fn):  # type: ignore[misc]
        return fn

    ZERO_GPU = False


# ----------------------------------------------------------------- pipeline -

_PIPELINE: DemoPipeline | None = None


def get_pipeline() -> DemoPipeline:
    """Build the pipeline once per process; the checkpoint load is the slow bit."""
    global _PIPELINE
    if _PIPELINE is None:
        cfg = Config.load(CONFIG_PATH)
        log.info("loading encoder backend=%s", cfg.encoder.backend)
        if cfg.encoder.backend == "synthetic":
            # The synthetic encoder is a regression harness, not a model. Showing
            # its output as a demo result would misrepresent the project.
            raise DemoError(
                "This demo requires the real ColSmol encoder; the configured "
                "backend is 'synthetic', which exists only for offline tests."
            )
        _PIPELINE = DemoPipeline(cfg)
        _PIPELINE.warm_up()
        log.info("encoder ready: %s", _PIPELINE.model_name)
    return _PIPELINE


@GPU_DECORATOR
def run_pipeline(image, page_id: str):
    """The one GPU-eligible step: encode, then prune/quantize on CPU-cheap numpy.

    ZeroGPU exposes CUDA only inside this function, so the device is resolved
    here rather than at import time.
    """
    pipeline = get_pipeline()
    messages: list[str] = []
    result = pipeline.compress(
        image, page_id=page_id, device=pick_device(), progress=messages.append
    )
    return result, messages


# ------------------------------------------------------------------- render -

CSS = """
.ov-title { text-align:center; margin-bottom:0.2rem; }
.ov-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.ov-card { border:1px solid var(--border-color-primary); border-radius:10px;
           padding:14px 16px; background:var(--background-fill-secondary); }
.ov-card .k { font-size:0.75rem; letter-spacing:.06em; text-transform:uppercase;
              opacity:.72; margin-bottom:6px; }
.ov-card .v { font-size:1.75rem; font-weight:650; line-height:1.15; }
.ov-card .s { font-size:0.78rem; opacity:.68; margin-top:4px; }
.ov-hero { font-size:2.4rem; font-weight:700; text-align:center; margin:6px 0 2px; }
.ov-bar-row { margin:14px 0; }
.ov-bar-label { display:flex; justify-content:space-between; font-size:0.85rem;
                margin-bottom:5px; }
.ov-bar { height:26px; border-radius:5px; background:var(--background-fill-secondary);
          overflow:hidden; }
.ov-bar > div { height:100%; border-radius:5px; }
.ov-flow { text-align:center; font-size:0.92rem; line-height:2.0; opacity:.9; }
.ov-flow b { font-size:1.05rem; }
table.ov-tech { width:100%; border-collapse:collapse; font-size:0.9rem; }
table.ov-tech td { padding:6px 10px; border-bottom:1px solid var(--border-color-primary); }
table.ov-tech td:first-child { opacity:.72; width:52%; }
table.ov-tech td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
.ov-note { font-size:0.82rem; opacity:.72; }
"""


QDRANT_OFF_HTML = (
    '<p class="ov-note">Off by default. The compression demonstration does not require '
    "Qdrant, and the app will not fail if it is unavailable.</p>"
)


def _card(key: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="ov-card"><div class="k">{key}</div><div class="v">{value}</div>{sub_html}</div>'


def status_block(lines: list[str], done: bool = False) -> str:
    icon = "✓" if done else "→"
    body = "<br>".join(f"{icon} {ln}" for ln in lines) if lines else "Waiting for a document."
    return f'<div class="ov-note" style="line-height:1.9">{body}</div>'


def metrics_block(r) -> str:
    return f"""<div class="ov-metrics">
{_card("Original visual tokens", f"{r.original_tokens:,}", f"{r.grid_patches:,} page patches + {r.non_grid_tokens} kept verbatim")}
{_card("Compressed visual tokens", f"{r.final_tokens:,}", f"{r.grid_rows}x{r.grid_cols} patch grid")}
{_card("Token reduction", f"{r.token_reduction_percent:.1f}%", f"{r.original_tokens:,} → {r.final_tokens:,} vectors")}
{_card("Original representation", f"{r.original_kib:,.1f} KiB", f"{r.original_tokens:,} x {r.embedding_dim} x 4 B (float32)")}
{_card("Compressed representation", f"{r.compressed_kib:,.2f} KiB", f"{r.final_tokens:,} x {r.embedding_dim} bits (1-bit)")}
{_card("Raw representation reduction", f"{r.storage_reduction_factor:,.1f}x", "embedding bytes, not disk footprint")}
</div>"""


def comparison_block(r) -> str:
    # Linear widths on purpose: the compressed bar *should* be a sliver. That is
    # the entire point, and a log scale would hide it.
    frac = r.compressed_bytes / max(1, r.original_bytes)
    compressed_pct = max(frac * 100.0, 0.35)  # keep it visible at ~120x
    return f"""
<div class="ov-hero">~{r.storage_reduction_factor:,.0f}x Smaller</div>
<div class="ov-note" style="text-align:center;margin-bottom:14px">
  raw embedding representation for this page
</div>

<div class="ov-bar-row">
  <div class="ov-bar-label"><span>Original — float32 multi-vector</span>
    <b>{r.original_kib:,.1f} KiB</b></div>
  <div class="ov-bar"><div style="width:100%;background:#94a3b8"></div></div>
</div>

<div class="ov-bar-row">
  <div class="ov-bar-label"><span>OptiVision — pruned + binary</span>
    <b>{r.compressed_kib:,.2f} KiB</b></div>
  <div class="ov-bar"><div style="width:{compressed_pct:.3f}%;background:#2e9f43"></div></div>
</div>

<div class="ov-note">
  Bars are linear and to scale: the green bar is {frac * 100:.2f}% of the grey one
  (drawn at a {compressed_pct:.2f}% minimum so it stays visible).
</div>
"""


def flow_block(r) -> str:
    return f"""<div class="ov-flow">
DOCUMENT PAGE<br>↓<br><b>{r.model_name}</b><br>↓<br>
<b>{r.original_tokens:,}</b> VISUAL TOKENS &nbsp;({r.original_kib:,.1f} KiB float32)<br>↓<br>
SPATIAL + REDUNDANCY PRUNING<br>↓<br>
<b>{r.final_tokens:,}</b> TOKENS &nbsp;(−{r.token_reduction_percent:.1f}%)<br>↓<br>
BINARY QUANTIZATION &nbsp;(1 bit / dimension)<br>↓<br>
<b>{r.compressed_kib:,.2f} KiB</b> REPRESENTATION<br>↓<br>
<b>~{r.storage_reduction_factor:,.0f}x</b> RAW REPRESENTATION REDUCTION
</div>"""


def tech_block(r) -> str:
    rows = [
        ("Model", r.model_name),
        ("Device used for encoding", r.device),
        ("Page image size", f"{r.image_size[0]} x {r.image_size[1]} px"),
        ("Patch grid", f"{r.grid_rows} x {r.grid_cols} = {r.grid_patches:,} patches"),
        ("Original tokens (model output)", f"{r.original_tokens:,}"),
        ("&nbsp;&nbsp;· page-grid patches", f"{r.grid_patches:,}"),
        ("&nbsp;&nbsp;· thumbnail + instruction tokens", f"{r.non_grid_tokens:,}"),
        ("After spatial pruning (patches)", f"{r.spatial_tokens:,}"),
        ("After redundancy pruning (patches)", f"{r.redundancy_tokens:,}"),
        ("Kept verbatim (never pruned)", f"{r.non_grid_tokens:,}"),
        ("<b>Final stored vectors</b>", f"<b>{r.final_tokens:,}</b>"),
        ("Embedding dimension", f"{r.embedding_dim}"),
        ("Original datatype", r.original_dtype),
        ("Compressed datatype", r.compressed_dtype),
        ("Bits per dimension (measured)", f"{r.bits_per_dimension:.3f}"),
        ("Original bytes", f"{r.original_bytes:,} B  ({r.original_kib:,.2f} KiB)"),
        ("Compressed bytes", f"{r.compressed_bytes:,} B  ({r.compressed_kib:,.2f} KiB)"),
        ("Compression factor", f"{r.storage_reduction_factor:,.2f}x"),
        ("Encoding time", f"{r.encoding_seconds:.2f} s"),
        ("Pruning time", f"{r.pruning_ms:.2f} ms"),
        ("Quantization time", f"{r.quantization_ms:.3f} ms"),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"""<table class="ov-tech">{body}</table>
<p class="ov-note">Final stored vectors = patches after redundancy pruning
({r.redundancy_tokens:,}) + tokens kept verbatim ({r.non_grid_tokens:,}) =
{r.final_tokens:,}. Byte figures are the actual sizes of the arrays the pipeline
produced, not estimates.</p>"""


# ----------------------------------------------------------------- handlers -


def _render_page(file_path, page_number):
    """Shared preview logic. Returns (image, note, total_pages or 0 on failure)."""
    try:
        image, total = load_page(file_path, int(page_number or 1))
    except DemoError as exc:
        return None, f"⚠️ {exc}", 0
    except Exception:  # noqa: BLE001 - any read failure becomes a friendly message
        log.error("preview failed\n%s", traceback.format_exc())
        return None, "⚠️ That file could not be read.", 0

    note = ""
    if total > 1:
        note = (
            f"This document has **{total} pages**. The demo compresses one page at a time — "
            f"showing page **{int(page_number or 1)}**. Change the page number to pick another."
        )
    return image, note, total


def on_upload(file_path, page_number):
    """Preview a newly uploaded file and reveal the page picker if it is multi-page."""
    if not file_path:
        return None, gr.update(visible=False, value=1), ""
    image, note, total = _render_page(file_path, 1)
    if total > 1:
        return image, gr.update(visible=True, value=1, maximum=total), note
    return image, gr.update(visible=False, value=1), note


def on_page_change(file_path, page_number):
    """Re-render after the user picks a different page (never touches the picker)."""
    if not file_path:
        return None, ""
    image, note, _ = _render_page(file_path, page_number)
    return image, note


def on_compress(file_path, page_number, use_qdrant, progress=gr.Progress()):  # noqa: B008 - Gradio's documented way to receive a progress handle
    """Stream stage updates, then render the results.

    Yields tuples matching the 8 output components; ``gr.update()`` leaves a
    component untouched on the intermediate yields.
    """
    hold = (gr.update(),) * 7  # everything except the status panel

    if not file_path:
        raise gr.Error("Upload a single-page PDF or an image first.")

    try:
        progress(0.05, desc="Preprocessing page")
        image, total = load_page(file_path, int(page_number or 1))
        page_id = Path(file_path).stem
    except DemoError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        log.error("preprocessing failed\n%s", traceback.format_exc())
        raise gr.Error("The page could not be prepared. Try a different file.") from exc

    lines = [f"Preprocessed page {int(page_number or 1)} of {total} ({image.size[0]}x{image.size[1]} px)."]
    yield (status_block(lines), *hold)

    try:
        progress(0.25, desc="Encoding with the vision-language model")
        lines.append("Encoding page with ColSmol (first run also loads the checkpoint)...")
        yield (status_block(lines), *hold)

        result, messages = run_pipeline(image, page_id)
    except DemoError as exc:
        raise gr.Error(str(exc)) from exc
    except (MemoryError, RuntimeError) as exc:
        log.error("pipeline failed\n%s", traceback.format_exc())
        detail = str(exc).lower()
        if "memory" in detail or "alloc" in detail:
            raise gr.Error(
                "Ran out of memory while encoding this page. Try a smaller or lower-resolution page."
            ) from exc
        raise gr.Error("The model failed to run on this page. Please try another document.") from exc
    except Exception as exc:
        log.error("pipeline failed\n%s", traceback.format_exc())
        raise gr.Error("Something went wrong while compressing this page.") from exc

    progress(0.9, desc="Rendering results")
    lines = [lines[0]]
    lines.extend(messages)
    lines.append(
        f"Stored {result.final_tokens:,} vectors — {result.compressed_kib:,.2f} KiB "
        f"({result.storage_reduction_factor:,.1f}x smaller than float32)."
    )
    if result.blank_page:
        lines.append(
            "⚠️ This page looks almost blank, so the compression ratio is unusually flattering."
        )

    qdrant_html = QDRANT_OFF_HTML
    if use_qdrant:
        info = store_in_qdrant(result.compressed_page)
        if info.get("ok"):
            qdrant_html = (
                f'<table class="ov-tech">'
                f'<tr><td>Collection</td><td>{info["collection"]}</td></tr>'
                f'<tr><td>Points stored</td><td>{info["n_points"]}</td></tr>'
                f'<tr><td>Vectors in this page\'s multivector</td><td>{info["n_vectors"]:,}</td></tr>'
                f'<tr><td>Comparator</td><td>{info["comparator"]}</td></tr>'
                f"</table>"
                f'<p class="ov-note">Written to a local embedded Qdrant collection. This confirms '
                f"the compressed page is storable as a multivector; it is not a measurement of "
                f"Qdrant's on-disk footprint.</p>"
            )
        else:
            qdrant_html = (
                f'<p class="ov-note">Qdrant storage unavailable — the compression demo above is '
                f'unaffected.<br>Reason: {info.get("error", "unknown")}</p>'
            )

    yield (
        status_block(lines, done=True),
        metrics_block(result),
        flow_block(result),
        result.original_image,
        result.retained_regions_image,
        comparison_block(result),
        tech_block(result),
        qdrant_html,
    )


def on_clear():
    """Reset every output back to its empty state."""
    return (
        None,  # file_in
        None,  # preview
        gr.update(visible=False, value=1),  # page_num
        "",  # page_note
        status_block([]),  # status
        "",  # metrics
        "",  # flow
        None,  # orig_img
        None,  # kept_img
        "",  # comparison
        "",  # tech
        QDRANT_OFF_HTML,  # qdrant_out
    )


# ---------------------------------------------------------------------- UI --

INTRO = """
<div class="ov-title">
  <h1 style="margin-bottom:0">OptiVision RAG</h1>
  <h3 style="margin-top:4px;font-weight:500;opacity:.85">
    Extreme Token Compression for Vision-Language Models</h3>
  <p style="opacity:.8;max-width:760px;margin:10px auto 0">
    Demonstration of visual-token pruning and binary quantization for efficient
    document retrieval.</p>
</div>
"""

PREMISE = """
> **We are not compressing the document image.** We are reducing and compressing the
> *visual token representation* that the vision-language model produces for that page —
> the vectors a retrieval index has to store. The page itself is untouched.
"""


# Gradio 6 moved `theme` and `css` from Blocks() to launch(); Gradio 5 only
# accepts them on Blocks(). The Space's resolved version is not fully under our
# control, so pass them wherever this install expects them.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0]) if gr.__version__[0].isdigit() else 6
_STYLE_ON_LAUNCH = _GRADIO_MAJOR >= 6
STYLE_KWARGS = {"theme": gr.themes.Soft(), "css": CSS}


def build_demo() -> gr.Blocks:
    blocks_kwargs = {} if _STYLE_ON_LAUNCH else STYLE_KWARGS
    with gr.Blocks(title="OptiVision RAG", **blocks_kwargs) as demo:
        gr.HTML(INTRO)
        gr.Markdown(PREMISE)

        with gr.Row():
            # ---------------------------------------------------- section 1
            with gr.Column(scale=1):
                gr.Markdown("### 1 · Document")
                file_in = gr.File(
                    label="Upload a single-page PDF or image",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"],
                    type="filepath",
                )
                page_num = gr.Number(
                    label="Page", value=1, minimum=1, step=1, visible=False, precision=0
                )
                page_note = gr.Markdown("")
                preview = gr.Image(label="Page preview", type="pil", height=380)
                with gr.Row():
                    run_btn = gr.Button("Compress Document", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", scale=1)

            # ---------------------------------------------------- section 2
            with gr.Column(scale=1):
                gr.Markdown("### 2 · Pipeline")
                status = gr.HTML(status_block([]))
                gr.Markdown("### Representation flow")
                flow = gr.HTML("")

        # -------------------------------------------------------- section 3
        gr.Markdown("### 3 · Results")
        metrics = gr.HTML("")

        # -------------------------------------------------------- section 4
        gr.Markdown("### 4 · Retained visual regions")
        gr.Markdown(
            "Green patches are the vectors that reach the index; red patches were "
            "dropped as blank or redundant. This is the pipeline's actual keep-mask, "
            "rendered over the page."
        )
        with gr.Row():
            orig_img = gr.Image(label="Original page", type="pil", height=520)
            kept_img = gr.Image(label="Retained visual regions", type="pil", height=520)

        # -------------------------------------------------------- section 5
        gr.Markdown("### 5 · Storage comparison")
        comparison = gr.HTML("")

        # -------------------------------------------------------- section 6
        with gr.Accordion("6 · Technical details", open=False):
            tech = gr.HTML("")

        with gr.Accordion("Optional · Qdrant storage", open=False):
            qdrant_toggle = gr.Checkbox(
                label="Also store this compressed page in a local Qdrant collection",
                value=False,
            )
            qdrant_out = gr.HTML(QDRANT_OFF_HTML)

        gr.Markdown(
            "---\n"
            "*Project Stage-I (CD753PC) · B.Tech CSE (Data Science). "
            "Figures shown are the raw embedding representation for a single page, "
            "computed from the actual arrays the pipeline produced — not a measured "
            "Qdrant on-disk footprint.*"
        )

        # ------------------------------------------------------------ wiring
        file_in.change(on_upload, [file_in, page_num], [preview, page_num, page_note])
        page_num.change(on_page_change, [file_in, page_num], [preview, page_note])

        run_btn.click(
            on_compress,
            [file_in, page_num, qdrant_toggle],
            [status, metrics, flow, orig_img, kept_img, comparison, tech, qdrant_out],
            concurrency_limit=1,
        )

        clear_btn.click(
            on_clear,
            None,
            [
                file_in, preview, page_num, page_note, status, metrics,
                flow, orig_img, kept_img, comparison, tech, qdrant_out,
            ],
        )

    return demo


if __name__ == "__main__":
    log.info(
        "gradio=%s config=%s zerogpu=%s device=%s",
        gr.__version__, CONFIG_PATH, ZERO_GPU, pick_device(),
    )
    launch_kwargs = STYLE_KWARGS if _STYLE_ON_LAUNCH else {}
    build_demo().queue(max_size=8).launch(**launch_kwargs)
