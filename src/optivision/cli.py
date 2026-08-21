"""``optivision`` command line.

    optivision make-corpus data/corpus --docs 20
    optivision index data/corpus/pdfs --config configs/synthetic.yaml
    optivision search "office memorandum 2024" --config configs/synthetic.yaml
    optivision bench data/corpus/pdfs data/corpus/queries.json --config configs/synthetic.yaml
    optivision explain data/corpus/pdfs --out reports/figures
    optivision stats
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import Config

app = typer.Typer(add_completion=False, help="OptiVision RAG — extreme token compression for VLM retrieval")
console = Console()


def _load_cfg(config: str | None) -> Config:
    cfg = Config.load(config)
    return cfg


def _progress():
    """The live bar used when stdout is a terminal."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


class _PlainProgress:
    """Minimal stand-in used when output is redirected: one line per step."""

    def __init__(self, every: int = 1) -> None:
        self.every = max(1, every)
        self._total = 0

    def add_task(self, description: str, total: int | None = None) -> int:
        self._total = total or 0
        print(f"{description}: 0/{self._total}", flush=True)
        return 0

    def update(self, task: int, completed: int | None = None, description: str = "") -> None:
        if completed is None:
            return
        if completed % self.every == 0 or completed == self._total:
            suffix = f"  {description}" if description else ""
            print(f"  {completed}/{self._total}{suffix}", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


def _progress_for(total: int):
    """Pick a live bar or plain lines depending on where output is going.

    Indexing a corpus on CPU takes tens of minutes, and rich suppresses live
    updates when stdout is not a tty. Piping to `tee` or running in the
    background would then print nothing at all for half an hour, which is
    indistinguishable from a hang — so fall back to periodic plain lines.
    """
    if console.is_terminal:
        return _progress()
    return _PlainProgress(every=max(1, total // 20))


# ------------------------------------------------------------------- corpus


@app.command("make-corpus")
def make_corpus(
    out: str = typer.Argument(..., help="output directory"),
    docs: int = typer.Option(20, help="number of documents"),
    pages: int = typer.Option(2, help="pages per document"),
    seed: int = typer.Option(7),
) -> None:
    """Generate a synthetic scanned-document corpus with ground-truth queries."""
    from .corpus import CorpusSpec, generate_synthetic_corpus

    spec = CorpusSpec(n_docs=docs, pages_per_doc=pages, seed=seed)
    with console.status(f"rendering {docs * pages} pages..."):
        manifest = generate_synthetic_corpus(out, spec)
    console.print(f"[green]wrote[/] {manifest['n_pages']} pages, {manifest['n_queries']} queries")
    console.print(f"  pdfs    {manifest['pdf_dir']}")
    console.print(f"  queries {Path(out) / 'queries.json'}")
    console.print(f"  layout  {Path(out) / 'layout.json'}  (for the synthetic encoder)")


@app.command("fetch-vidore")
def fetch_vidore(
    dataset: str = typer.Option("vidore/syntheticDocQA_energy_test", help="HF dataset id"),
    out: str = typer.Option("data/vidore"),
    limit: int = typer.Option(100),
) -> None:
    """Download a ViDoRe benchmark split as images + queries.json."""
    from .corpus import load_vidore_subset

    manifest = load_vidore_subset(dataset, out, limit)
    console.print(f"[green]{manifest['n_pages']} pages, {manifest['n_queries']} queries[/] -> {out}")


# -------------------------------------------------------------------- index


@app.command()
def index(
    source: str = typer.Argument(..., help="folder of PDFs / images, or one file"),
    config: str | None = typer.Option(None, "--config", "-c"),
    append: bool = typer.Option(False, help="add to the existing index instead of recreating"),
    float_cache: str | None = typer.Option(None, help="also write float vectors here (for reranking)"),
    limit: int | None = typer.Option(None, help="stop after N pages"),
) -> None:
    """Encode, prune, compress and index a document collection."""
    from .ingest import count_pages, iter_pages
    from .pipeline import OptiVisionRAG

    cfg = _load_cfg(config)
    rag = OptiVisionRAG(cfg)
    total = count_pages(source, cfg.ingest)
    if limit:
        total = min(total, limit)

    console.print(f"[bold]encoder[/] {cfg.encoder.backend}   [bold]index[/] {cfg.index.backend} -> {cfg.index.path}")

    def pages():
        for i, item in enumerate(iter_pages(source, cfg.ingest)):
            if limit is not None and i >= limit:
                return
            yield item

    with _progress_for(total) as prog:
        task = prog.add_task("indexing", total=total)
        report = rag.build(
            pages(),
            recreate=not append,
            progress=lambda n, ref: prog.update(task, completed=n, description=f"indexing {ref.doc_id}"),
            float_cache=float_cache,
        )
    rag.close()

    d = report.as_dict()
    table = Table(title="index built", show_header=False)
    table.add_row("pages", f"{d['n_pages']}")
    table.add_row("tokens/page", f"{d['tokens_per_page_before']:.1f} -> {d['tokens_per_page_after']:.1f}")
    table.add_row("token reduction", f"{d['token_reduction']:.2f}x")
    table.add_row("index size", f"{d['index_bytes'] / 1e6:.2f} MB  (raw {d['raw_bytes'] / 1e6:.2f} MB)")
    table.add_row("compression", f"{d['compression_ratio']:.1f}x")
    table.add_row("per page", f"{d['bytes_per_page'] / 1e3:.2f} KB")
    table.add_row("encode", f"{d['encode_seconds']:.1f} s")
    table.add_row("prune+compress", f"{d['prune_seconds'] + d['compress_seconds']:.2f} s")
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(...),
    config: str | None = typer.Option(None, "--config", "-c"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Query the index."""
    from .pipeline import OptiVisionRAG

    cfg = _load_cfg(config)
    rag = OptiVisionRAG(cfg)
    result = rag.search(query, top_k=top_k)

    table = Table(title=f'"{query}"  ({result.latency_ms:.1f} ms over {result.candidates_scored} pages)')
    table.add_column("#", justify="right")
    table.add_column("page")
    table.add_column("score", justify="right")
    table.add_column("source", overflow="fold")
    for hit in result.hits:
        table.add_row(str(hit.rank), hit.ref.page_id, f"{hit.score:.3f}", hit.ref.source_path or "")
    console.print(table)
    rag.close()


@app.command()
def stats(config: str | None = typer.Option(None, "--config", "-c")) -> None:
    """Show what is currently in the index."""
    cfg = _load_cfg(config)
    manifest_path = Path(cfg.index.path) / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]no index at {cfg.index.path}[/] — run `optivision index` first")
        raise typer.Exit(1)
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    s = manifest["index"]
    table = Table(title=f"index @ {cfg.index.path}", show_header=False)
    for key in (
        "backend", "method", "dim", "n_pages", "n_vectors", "vectors_per_page",
        "index_bytes", "raw_bytes", "compression_ratio", "token_reduction",
    ):
        if key in s:
            value = s[key]
            table.add_row(key, f"{value:,.2f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value))
    console.print(table)


# ---------------------------------------------------------------- benchmark


@app.command()
def bench(
    source: str = typer.Argument(..., help="corpus folder"),
    queries: str = typer.Argument(..., help="queries.json"),
    config: str | None = typer.Option(None, "--config", "-c"),
    out: str = typer.Option("reports", help="where to write benchmark.json/.md"),
    limit: int | None = typer.Option(None, help="only index the first N pages"),
    sweep: bool = typer.Option(False, help="add the keep-ratio sweep rows"),
    top_k: int = typer.Option(10),
    cache: str | None = typer.Option(
        None, help="reuse/write an encode cache (.npz) so new rows skip re-encoding"
    ),
    codebook: bool = typer.Option(
        False, help="add retrieval-space saliency rows at matched token budgets"
    ),
) -> None:
    """Run the ablation table: baseline vs pruning vs quantization vs both."""
    from .bench import (
        codebook_sweep,
        default_variants,
        keep_ratio_sweep,
        run_benchmark,
        save_report,
        to_markdown,
    )
    from .ingest import count_pages

    cfg = _load_cfg(config)
    variants = default_variants() + (keep_ratio_sweep() if sweep else [])
    if codebook:
        # Needs the keep-N% rows to compare against; a matched-budget claim with
        # nothing to match is not a comparison.
        if not sweep:
            variants += keep_ratio_sweep()
        variants += codebook_sweep()
    total = count_pages(source, cfg.ingest)
    if limit:
        total = min(total, limit)

    with _progress_for(total) as prog:
        task = prog.add_task("encoding corpus (once)", total=total)
        report = run_benchmark(
            source,
            queries,
            cfg=cfg,
            variants=variants,
            limit=limit,
            top_k=top_k,
            workdir=str(Path(out) / "bench_indexes"),
            progress=lambda n, ref: prog.update(task, completed=n),
            cache=cache,
        )

    json_path, md_path = save_report(report, out)
    console.print(to_markdown(report))
    console.print(f"[green]wrote[/] {json_path}  {md_path}")


@app.command()
def explain(
    source: str = typer.Argument(..., help="folder or file to visualise"),
    config: str | None = typer.Option(None, "--config", "-c"),
    out: str = typer.Option("reports/figures"),
    limit: int = typer.Option(4, help="how many pages to render"),
) -> None:
    """Render original | saliency | keep-mask figures for the report."""
    from .ingest import iter_pages
    from .pipeline import OptiVisionRAG
    from .viz import explain_page

    cfg = _load_cfg(config)
    rag = OptiVisionRAG(cfg)
    out_dir = Path(out)
    written = []
    for i, (ref, image) in enumerate(iter_pages(source, cfg.ingest)):
        if i >= limit:
            break
        enc = rag.encoder.encode_pages([image], [ref])[0]
        pruned = rag.pruner.prune(enc, image)
        path = out_dir / f"{ref.page_id.replace('/', '_').replace('::', '_')}.png"
        explain_page(image, pruned, path)
        written.append((path, pruned))
        console.print(
            f"{ref.page_id}: kept {pruned.n_kept}/{pruned.n_tokens_before} vectors "
            f"({pruned.keep_ratio:.0%})  -> {path}"
        )
    if not written:
        console.print("[red]no pages found[/]")
    rag.close()


@app.command("init-config")
def init_config(
    out: str = typer.Argument("configs/my.yaml"),
    backend: str = typer.Option("colsmol", help="colsmol | colpali | colqwen2 | synthetic"),
) -> None:
    """Write a config file pre-filled with the defaults."""
    cfg = Config()
    cfg.encoder.backend = backend
    cfg.dump(out)
    console.print(f"[green]wrote[/] {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
