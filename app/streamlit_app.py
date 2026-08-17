"""OptiVision RAG demo.

    streamlit run app/streamlit_app.py -- --config configs/synthetic.yaml

Two tabs: a search box over a built index, and a per-page inspector that shows
exactly which patches were dropped and what that saved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import streamlit as st

from optivision.config import Config
from optivision.ingest import iter_pages
from optivision.pipeline import OptiVisionRAG
from optivision.viz import overlay_mask, overlay_saliency


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/synthetic.yaml")
    p.add_argument("--corpus", default="data/corpus/pdfs")
    known, _ = p.parse_known_args(argv)
    return known


ARGS = parse_args()

st.set_page_config(page_title="OptiVision RAG", page_icon="🗜️", layout="wide")


@st.cache_resource(show_spinner="loading encoder and index...")
def load_rag(config_path: str) -> OptiVisionRAG:
    return OptiVisionRAG(Config.load(config_path))


@st.cache_data(show_spinner=False)
def load_manifest(index_path: str) -> dict | None:
    path = Path(index_path) / "manifest.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


st.title("OptiVision RAG")
st.caption("Extreme token compression for vision-language document retrieval")

config_path = st.sidebar.text_input("config", ARGS.config)
corpus_path = st.sidebar.text_input("corpus", ARGS.corpus)

try:
    rag = load_rag(config_path)
except Exception as exc:  # noqa: BLE001 # config or model problems should be readable, not a traceback wall
    st.error(f"could not load the pipeline from `{config_path}`: {exc}")
    st.stop()

manifest = load_manifest(rag.cfg.index.path)

with st.sidebar:
    st.subheader("index")
    if manifest is None:
        st.warning("no index yet — run `optivision index <corpus>`")
    else:
        s = manifest["index"]
        st.metric("pages", f"{s['n_pages']:,}")
        st.metric("compression", f"{s['compression_ratio']:.0f}x")
        st.metric("tokens/page", f"{s['vectors_per_page']:.0f}")
        st.metric("index size", f"{s['index_bytes'] / 1e6:.2f} MB")
        st.caption(f"raw would be {s['raw_bytes'] / 1e6:.1f} MB")

search_tab, inspect_tab = st.tabs(["search", "what got pruned"])

with search_tab:
    query = st.text_input("query", placeholder="renewal of vehicle insurance policy")
    top_k = st.slider("results", 1, 20, 5)
    if query:
        if manifest is None:
            st.error("build an index first")
        else:
            result = rag.search(query, top_k=top_k)
            st.caption(
                f"{result.latency_ms:.1f} ms over {result.candidates_scored} pages "
                f"({rag.cfg.compression.method} codes)"
            )
            for hit in result.hits:
                col_a, col_b = st.columns([1, 4])
                col_a.metric(f"#{hit.rank}", f"{hit.score:.2f}")
                col_b.write(f"**{hit.ref.page_id}**")
                col_b.caption(hit.ref.source_path or "")

with inspect_tab:
    st.write(
        "Every green cell is a patch vector that reaches the index. "
        "Red cells are dropped before anything is stored."
    )
    if not Path(corpus_path).exists():
        st.error(f"corpus `{corpus_path}` not found")
    else:
        pages = list(iter_pages(corpus_path, rag.cfg.ingest))[:50]
        labels = [ref.page_id for ref, _ in pages]
        choice = st.selectbox("page", labels)
        ref, image = pages[labels.index(choice)]

        with st.spinner("encoding and pruning..."):
            enc = rag.encoder.encode_pages([image], [ref])[0]
            pruned = rag.pruner.prune(enc, image)
            compressed = rag.compressor.compress(pruned)

        c1, c2, c3 = st.columns(3)
        c1.image(image, caption="page", use_container_width=True)
        if pruned.saliency is not None:
            c2.image(
                overlay_saliency(image, pruned.saliency),
                caption="saliency",
                use_container_width=True,
            )
        c3.image(
            overlay_mask(image, pruned.keep_mask), caption="kept / dropped", use_container_width=True
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("vectors", f"{pruned.n_kept}", f"-{enc.n_tokens - pruned.n_kept}")
        m2.metric("token reduction", f"{enc.n_tokens / max(1, pruned.n_kept):.1f}x")
        m3.metric("stored", f"{compressed.nbytes / 1024:.1f} KB")
        m4.metric(
            "vs float32",
            f"{compressed.raw_nbytes() / max(1, compressed.nbytes):.0f}x",
            f"was {compressed.raw_nbytes() / 1024:.0f} KB",
        )
        st.json(pruned.stats, expanded=False)
