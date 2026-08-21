"""Ablation harness.

The experiment this project has to answer is narrow: *how much of the index can
we throw away before retrieval quality moves?* So the harness encodes the corpus
exactly once, then replays every pruning/quantization setting over those cached
vectors. Nothing but the compression changes between rows of the table, and the
uncompressed float32 run is always present as row zero.

Quality is reported two ways:

    absolute   nDCG@5 / Recall@1 against the corpus ground truth
    relative   Kendall tau against the *baseline's own ranking*

The second one is the honest measure of compression damage: it does not care
whether the model was right, only whether compressing changed its mind.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .compression import Compressor
from .config import Config
from .corpus import load_queries
from .encoders import BaseEncoder, get_encoder
from .index.numpy_index import NumpyIndex
from .ingest import iter_pages
from .metrics import (
    evaluate,
    rank_correlation,
    rank_correlation_shared,
    storage_summary,
)
from .pruning import TokenPruner
from .types import PageEncoding, PageRef, PatchGrid


@dataclass
class Variant:
    name: str
    pruning: dict = field(default_factory=dict)
    compression: dict = field(default_factory=dict)
    note: str = ""


def default_variants() -> list[Variant]:
    """The table that goes in the report."""
    return [
        Variant(
            "baseline-float32",
            pruning={"enabled": False},
            compression={"enabled": False, "method": "none"},
            note="ColPali as published: every patch, full precision",
        ),
        Variant(
            "binary-only",
            pruning={"enabled": False},
            compression={"enabled": True, "method": "binary"},
            note="quantization alone (32x)",
        ),
        Variant(
            "int8-only",
            pruning={"enabled": False},
            compression={"enabled": True, "method": "int8"},
            note="scalar quantization alone (4x)",
        ),
        Variant(
            "spatial-only",
            pruning={"enabled": True, "spatial": True, "redundancy": False},
            compression={"enabled": False, "method": "none"},
            note="blank-patch pruning alone",
        ),
        Variant(
            "spatial+redundancy",
            pruning={"enabled": True, "spatial": True, "redundancy": True},
            compression={"enabled": False, "method": "none"},
            note="both pruning stages, full precision",
        ),
        Variant(
            "prune+int8",
            pruning={"enabled": True, "spatial": True, "redundancy": True},
            compression={"enabled": True, "method": "int8"},
            note="pruning with the cheaper quantizer — the quality-first option",
        ),
        Variant(
            "optivision",
            pruning={"enabled": True, "spatial": True, "redundancy": True},
            compression={"enabled": True, "method": "binary"},
            note="full pipeline: prune + binary",
        ),
        Variant(
            "optivision-aggressive",
            pruning={
                "enabled": True,
                "spatial": True,
                "redundancy": True,
                "keep_ratio": 0.25,
                "redundancy_threshold": 0.85,
            },
            compression={"enabled": True, "method": "binary"},
            note="fixed 25% token budget",
        ),
    ]


def keep_ratio_sweep(ratios: Sequence[float] = (0.5, 0.4, 0.3, 0.2, 0.1)) -> list[Variant]:
    return [
        Variant(
            f"keep-{int(r * 100)}pct",
            pruning={"enabled": True, "spatial": True, "redundancy": True, "keep_ratio": r},
            compression={"enabled": True, "method": "binary"},
            note=f"top {int(r * 100)}% most salient patches, binary",
        )
        for r in ratios
    ]


# --------------------------------------------------------------------- cache


class EncodedCorpus:
    """Encode every page once; hold the vectors and a downscaled image copy.

    The image copy is needed because spatial pruning reads pixels, and re-reading
    the PDF for every ablation row would dominate the runtime.
    """

    def __init__(self, encodings: list[PageEncoding], images: list[Image.Image]) -> None:
        self.encodings = encodings
        self.images = images

    @property
    def dim(self) -> int:
        return self.encodings[0].dim if self.encodings else 0

    @property
    def n_pages(self) -> int:
        return len(self.encodings)

    @classmethod
    def build(
        cls,
        source: str | Path,
        encoder: BaseEncoder,
        cfg: Config,
        limit: int | None = None,
        progress=None,
    ) -> EncodedCorpus:
        encodings: list[PageEncoding] = []
        images: list[Image.Image] = []
        for i, (ref, image) in enumerate(iter_pages(source, cfg.ingest)):
            if limit is not None and i >= limit:
                break
            enc = encoder.encode_pages([image], [ref])[0]
            encodings.append(enc)
            # 512px is plenty for an 8x8..32x32 saliency grid and keeps the
            # whole corpus in RAM on a laptop.
            images.append(image.copy().convert("L").resize((512, 512), Image.BILINEAR))
            if progress is not None:
                progress(i + 1, ref)
        return cls(encodings, images)

    # ------------------------------------------------------------ disk cache

    def save(self, path: str | Path) -> None:
        """Persist the encode pass so new ablation rows cost seconds, not an hour.

        Encoding is ~30 s/page on CPU and every variant replays the same vectors,
        so adding one row to the table should not mean re-encoding the corpus.
        Stores the vectors, the grid layout and the grayscale page copies.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        meta = []
        for i, (enc, img) in enumerate(zip(self.encodings, self.images, strict=True)):
            payload[f"emb_{i}"] = enc.embeddings
            payload[f"grid_{i}"] = enc.grid.token_index
            payload[f"text_{i}"] = enc.text_token_index
            payload[f"img_{i}"] = np.asarray(img, dtype=np.uint8)
            meta.append(
                {
                    "ref": enc.ref.__dict__,
                    "rows": enc.grid.rows,
                    "cols": enc.grid.cols,
                    "image_size": list(enc.image_size),
                    "meta": enc.meta,
                }
            )
        payload["meta"] = np.array(json.dumps(meta), dtype=object)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> EncodedCorpus:
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        encodings, images = [], []
        for i, m in enumerate(meta):
            grid = PatchGrid(rows=m["rows"], cols=m["cols"], token_index=data[f"grid_{i}"])
            encodings.append(
                PageEncoding(
                    ref=PageRef(**m["ref"]),
                    embeddings=data[f"emb_{i}"],
                    grid=grid,
                    image_size=tuple(m["image_size"]),
                    text_token_index=data[f"text_{i}"],
                    meta=m["meta"],
                )
            )
            images.append(Image.fromarray(data[f"img_{i}"], mode="L"))
        return cls(encodings, images)


# ----------------------------------------------------------------- the runner


def run_variant(
    corpus: EncodedCorpus,
    cfg: Config,
    variant: Variant,
    query_vectors: list[np.ndarray],
    qids: Sequence[str],
    qrels: dict[str, set[str]],
    top_k: int = 10,
    workdir: str | Path = "data/bench",
    tau_depth: int | None = None,
) -> dict:
    vcfg = cfg.with_overrides(pruning=variant.pruning, compression=variant.compression)
    pruner = TokenPruner(vcfg.pruning)
    compressor = Compressor(vcfg.compression)
    method = vcfg.compression.method if vcfg.compression.enabled else "none"

    t0 = time.perf_counter()
    index = NumpyIndex(Path(workdir) / variant.name, dim=corpus.dim, method=method)
    compressed = []
    for enc, image in zip(corpus.encodings, corpus.images, strict=True):
        pruned = pruner.prune(enc, image)
        compressed.append(compressor.compress(pruned))
    index.add(compressed)
    build_s = time.perf_counter() - t0

    # Rank agreement is measured over the whole candidate set rather than the
    # top-k hit list. A page that falls out of the list entirely is the damage
    # most worth counting, and a truncated list cannot see it.
    #
    # Both come from one scoring pass. MaxSim over the corpus is the whole cost
    # of a search, so the timed section covers exactly what a deployment pays
    # and the full ordering is taken afterwards for free.
    depth = tau_depth or len(corpus.encodings)
    refs = index.refs
    run: dict[str, list[str]] = {}
    deep: dict[str, list[str]] = {}
    latencies = []
    for qid, qv in zip(qids, query_vectors, strict=True):
        t1 = time.perf_counter()
        scores = index.score_all(qv)
        k = min(top_k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        latencies.append((time.perf_counter() - t1) * 1000.0)

        run[qid] = [refs[int(i)].page_id for i in top]
        deep[qid] = [refs[int(i)].page_id for i in np.argsort(-scores)[:depth]]

    stats = index.stats()
    metrics = evaluate(run, qrels, ks=(1, 3, 5, 10))
    row = {
        "variant": variant.name,
        "note": variant.note,
        **metrics,
        **storage_summary(stats),
        "tokens_per_page": stats["vectors_per_page"],
        "tokens_per_page_raw": stats["tokens_before"] / max(1, stats["n_pages"]),
        "token_reduction": stats["token_reduction"],
        "index_build_s": build_s,
        "query_ms_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "query_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
    }
    return {"row": row, "run": run, "deep": deep}


def _encode_queries_cached(
    texts: Sequence[str],
    cfg: Config,
    encoder: BaseEncoder | None,
    cache_path: Path | None,
) -> tuple[list[np.ndarray], float]:
    """Query vectors, reused from disk when the query set is unchanged.

    Keyed on the query strings themselves, so editing queries.json correctly
    invalidates the cache rather than silently scoring the old questions.
    """
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if list(json.loads(str(data["texts"]))) == list(texts):
            vectors = [data[f"q_{i}"] for i in range(len(texts))]
            return vectors, float(data["encode_ms"])

    if encoder is None:
        encoder = get_encoder(cfg.encoder)
    t0 = time.perf_counter()
    vectors = encoder.encode_queries(list(texts))
    encode_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(texts))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"q_{i}": v for i, v in enumerate(vectors)}
        payload["texts"] = np.array(json.dumps(list(texts)), dtype=object)
        payload["encode_ms"] = np.array(encode_ms)
        np.savez_compressed(cache_path, **payload)
    return vectors, encode_ms


def run_benchmark(
    source: str | Path,
    queries_path: str | Path,
    cfg: Config | None = None,
    variants: list[Variant] | None = None,
    limit: int | None = None,
    top_k: int = 10,
    workdir: str | Path = "data/bench",
    progress=None,
    cache: str | Path | None = None,
    tau_depth: int | None = None,
) -> dict:
    """Run every variant over one encode pass.

    ``cache`` points at an .npz of page encodings: reused if it exists, written
    if it does not. Query vectors are cached alongside it, keyed by the query
    text, so adding a row to the table costs seconds instead of a fresh
    half-hour encode of the whole corpus.
    """
    cfg = cfg or Config()
    variants = variants or default_variants()

    qids, texts, qrels = load_queries(queries_path)

    cache_path = Path(cache) if cache else None
    queries_cache = cache_path.with_suffix(".queries.npz") if cache_path else None
    reuse = cache_path is not None and cache_path.exists()

    encoder: BaseEncoder | None = None
    if reuse:
        corpus = EncodedCorpus.load(cache_path)
    else:
        encoder = get_encoder(cfg.encoder)
        corpus = EncodedCorpus.build(source, encoder, cfg, limit=limit, progress=progress)
        if cache_path is not None and corpus.n_pages:
            corpus.save(cache_path)
    if corpus.n_pages == 0:
        raise RuntimeError(f"no pages found under {source}")

    # Restrict ground truth to pages that were actually indexed (``limit``).
    indexed = {e.ref.page_id for e in corpus.encodings}
    keep = [i for i, qid in enumerate(qids) if qrels[qid] & indexed]
    qids = [qids[i] for i in keep]
    texts = [texts[i] for i in keep]
    qrels = {qid: (qrels[qid] & indexed) for qid in qids}
    if not qids:
        raise RuntimeError("no queries have a relevant page inside the indexed subset")

    query_vectors, query_encode_ms = _encode_queries_cached(
        texts, cfg, encoder, queries_cache
    )

    rows = []
    runs: dict[str, dict] = {}
    shallow: dict[str, dict] = {}
    for variant in variants:
        out = run_variant(
            corpus, cfg, variant, query_vectors, qids, qrels, top_k=top_k,
            workdir=workdir, tau_depth=tau_depth,
        )
        rows.append(out["row"])
        runs[variant.name] = out["deep"]
        shallow[variant.name] = out["run"]

    # Rank agreement with the uncompressed baseline, over the full candidate
    # pool. Reported alongside the pool size, because a tau is only comparable
    # to another tau taken over a comparable fraction of its corpus.
    page_ids = [e.ref.page_id for e in corpus.encodings]
    # A truncated deep search cannot speak for pages it never returned, so the
    # pool falls back to the union of the two lists in that case.
    pool = page_ids if tau_depth is None else None
    base_name = variants[0].name
    base_run = runs[base_name]
    base_shallow = shallow[base_name]
    for row in rows:
        this = runs[row["variant"]]
        taus = [rank_correlation(base_run[q], this[q], pool=pool) for q in qids]
        row["kendall_tau_vs_baseline"] = float(np.mean(taus))
        row["kendall_tau_pool"] = len(pool) if pool is not None else int(tau_depth)

        # Also carry the superseded statistic. Reports predating the fix have
        # only this one, and the paper's tables quote it across all three
        # experiments; recording it keeps those tables regenerable in one pass
        # instead of stranding whichever experiment was re-run first.
        that = shallow[row["variant"]]
        legacy = [rank_correlation_shared(base_shallow[q], that[q]) for q in qids]
        row["kendall_tau_shared_topk"] = float(np.mean(legacy))
        row["kendall_tau_shared_k"] = top_k
        base_ndcg = rows[0].get("ndcg@5", 0.0)
        row["ndcg5_retention"] = (row.get("ndcg@5", 0.0) / base_ndcg) if base_ndcg else 0.0

    return {
        "config": cfg.to_dict(),
        "corpus": {"source": str(source), "n_pages": corpus.n_pages, "n_queries": len(qids)},
        "encoder": {"backend": cfg.encoder.backend, "dim": corpus.dim},
        "query_encode_ms": query_encode_ms,
        "rows": rows,
    }


# ------------------------------------------------------------------ reporting

TABLE_COLUMNS = [
    ("variant", "Variant", "{}"),
    ("tokens_per_page", "Tok/pg", "{:.1f}"),
    ("kb_per_page", "KB/pg", "{:.2f}"),
    ("compression_ratio", "Compr.", "{:.1f}x"),
    ("ndcg@5", "nDCG@5", "{:.4f}"),
    ("recall@1", "R@1", "{:.4f}"),
    ("hit@5", "Hit@5", "{:.4f}"),
    ("ndcg5_retention", "Retain", "{:.1%}"),
    ("kendall_tau_vs_baseline", "Tau", "{:.3f}"),
    ("query_ms_p50", "q ms", "{:.2f}"),
]


def to_markdown(report: dict) -> str:
    rows = report["rows"]
    header = "| " + " | ".join(label for _, label, _ in TABLE_COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _, fmt in TABLE_COLUMNS:
            value = row.get(key)
            cells.append(fmt.format(value) if value is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    corpus = report["corpus"]
    enc = report["encoder"]
    preamble = (
        f"**Corpus**: {corpus['n_pages']} pages, {corpus['n_queries']} queries  \n"
        f"**Encoder**: {enc['backend']} (dim {enc['dim']})  \n"
        f"**Tau pool**: {rows[0].get('kendall_tau_pool', corpus['n_pages'])} candidates "
        f"— rank agreement is comparable across runs only at a comparable pool  \n"
        f"**Query encode**: {report['query_encode_ms']:.1f} ms/query\n"
    )
    notes = "\n".join(f"- `{r['variant']}` — {r['note']}" for r in rows if r.get("note"))
    return f"{preamble}\n" + "\n".join(lines) + "\n\n" + notes + "\n"


def save_report(report: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark.json"
    md_path = out_dir / "benchmark.md"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(report))
    return json_path, md_path
