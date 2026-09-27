"""CLI commands for the universal layer: inspect, compress, calibrate, benchmark, compare.

They operate on vectors, not models, so they work for any encoder whose output
has been saved as a ``MultiVectorCorpus`` (``.npz``) -- and on the encode caches
written by ``optivision bench --cache``, which are detected automatically.

Queries come in one of two forms:

* a legacy ``queries.json`` (``[{qid, query, relevant}]``) next to a legacy
  cache, whose query vectors live in ``<cache>.queries.npz``;
* a ``MultiVectorCorpus`` ``.npz`` of query vectors, with optional labels in a
  JSON file mapping ``qid -> [doc ids]`` (``--qrels``).

Pipelines are given as JSON (inline or a file) or as a shorthand:
``"adaptive_merge(radius=0.8) > int8(scale=per_vector)"``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

console = Console()

# ------------------------------------------------------------------ loading


def _is_legacy_cache(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as data:
        return "meta" in data.files and "header" not in data.files


def _is_compressed(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as data:
        if "header" not in data.files or "codes" not in data.files:
            return False
        return json.loads(bytes(data["header"]).decode("utf-8")).get("format") == "optivision-compressed"


def load_corpus(path: str | Path, with_images: bool = False):
    from .representation import MultiVectorCorpus

    p = Path(path)
    if _is_legacy_cache(p):
        return MultiVectorCorpus.from_legacy_cache(p, with_images=with_images)
    return MultiVectorCorpus.load(p)


def load_task(corpus_path: str, queries: str | None, qrels: str | None, with_images: bool = False):
    """(corpus, queries or None, qrels or None) from command-line paths."""
    from .benchmark import load_legacy_dataset
    from .representation import MultiVectorCorpus

    cp = Path(corpus_path)
    if queries is not None and queries.endswith(".json") and _is_legacy_cache(cp):
        ds = load_legacy_dataset(cp, queries, with_images=with_images)
        return ds.corpus, ds.queries, ds.qrels
    corpus = load_corpus(cp, with_images=with_images)
    if queries is None:
        return corpus, None, None
    qc = MultiVectorCorpus.load(queries)
    labels = None
    if qrels is not None:
        raw = json.loads(Path(qrels).read_text(encoding="utf-8"))
        labels = {str(k): set(v) for k, v in raw.items()}
    return corpus, qc, labels


# ------------------------------------------------------------ pipeline spec

_STAGE = re.compile(r"^\s*([a-z0-9_]+)\s*(?:\((.*)\))?\s*$")


def _value(text: str) -> Any:
    text = text.strip()
    if text in {"None", "none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_pipeline(spec: str):
    """JSON (inline or file) or ``name(k=v, ...) > name > ...`` shorthand."""
    from .compose import Pipeline
    from .stages import stage_from_dict

    text = spec.strip()
    path = Path(text)
    if not text.startswith(("{", "[")) and path.suffix == ".json" and path.is_file():
        text = path.read_text(encoding="utf-8")
    if text.startswith("{"):
        return Pipeline.from_dict(json.loads(text))
    if text.startswith("["):
        return Pipeline.from_dict({"stages": json.loads(text)})
    if text in {"", "float32", "none"}:
        return Pipeline()
    stages = []
    for part in re.split(r"\s*>\s*", text):
        m = _STAGE.match(part)
        if not m:
            raise typer.BadParameter(f"cannot parse stage {part!r}")
        kwargs = {}
        if m.group(2):
            for item in filter(None, (x.strip() for x in m.group(2).split(","))):
                key, _, val = item.partition("=")
                kwargs[key.strip()] = _value(val)
        stages.append(stage_from_dict({"stage": m.group(1), **kwargs}))
    return Pipeline(stages)


def _kb(x: float) -> str:
    return f"{x / 1e3:,.2f} KB"


# ------------------------------------------------------------------ commands


def inspect(path: str = typer.Argument(..., help="corpus .npz, legacy encode cache, or compressed .npz")) -> None:
    """Describe a vector file: documents, vectors, dimension, dtype, bytes."""
    from .compose import load_compressed

    p = Path(path)
    if _is_compressed(p):
        report = load_compressed(p).report()
        title = f"compressed corpus @ {p}"
    else:
        report = load_corpus(p).summary()
        title = f"{'legacy encode cache' if _is_legacy_cache(p) else 'multi-vector corpus'} @ {p}"
    table = Table(title=title, show_header=False)
    for key, value in report.items():
        if isinstance(value, float):
            value = f"{value:,.4g}"
        elif isinstance(value, dict):
            value = json.dumps(value)[:120]
        table.add_row(key, str(value))
    console.print(table)


def compress(
    corpus: str = typer.Argument(..., help="corpus .npz or legacy encode cache"),
    pipeline: str = typer.Option(..., "--pipeline", "-p", help="JSON, a .json file, or shorthand"),
    out: str = typer.Option(..., "--out", "-o", help="where to write the compressed .npz"),
) -> None:
    """Compress a corpus with an explicit pipeline and write the codes."""
    from .compose import save_compressed

    pipe = parse_pipeline(pipeline)
    needs_images = any(s.name == "spatial" for s in pipe.stages)
    docs = load_corpus(corpus, with_images=needs_images)
    compressed = pipe.compress(docs)
    path = save_compressed(compressed, out)
    r = compressed.report()
    console.print(f"[green]wrote[/] {path}  ({pipe.label()})")
    console.print(f"  vectors/doc {r['vectors_per_doc']:.1f}  dim {r['dim']}  bits/dim {r['bits_per_dim']:.3g}  "
                  f"{_kb(r['bytes_per_doc'])}/doc  {r['compression_vs_float32']:.1f}x vs float32")


def calibrate(
    corpus: str = typer.Argument(..., help="corpus .npz or legacy encode cache"),
    queries: str = typer.Option(..., "--queries", "-q", help="queries .npz, or legacy queries.json"),
    qrels: str | None = typer.Option(None, help="labels JSON {qid: [doc ids]} for a queries .npz"),
    target: float = typer.Option(0.97, "--target", help="retention target, e.g. 0.97"),
    metric: str = typer.Option("ndcg@5", help="ndcg@k | recall@k | mrr@k | hit@k"),
    reference: str = typer.Option("auto", help="labels | baseline | auto"),
    safety: str = typer.Option("lower_ci", help="lower_ci (default, conservative) | point"),
    calibration_fraction: float = typer.Option(0.5, help="share of queries used to choose"),
    seed: int = typer.Option(0),
    out: str | None = typer.Option(None, help="write the full result as JSON"),
) -> None:
    """Choose the smallest configuration meeting a retention target; report it on held-out queries."""
    from .calibration import calibrate as run_calibration

    docs, qs, labels = load_task(corpus, queries, qrels)
    if reference == "labels" and labels is None:
        raise typer.BadParameter("reference=labels needs labels (--qrels or a legacy queries.json)")

    def progress(c):
        mark = "[green]ok[/]" if c.feasible else "[red]--[/]"
        console.print(f"  {mark} {c.label:44s} {_kb(c.bytes_per_doc):>12s}/doc  retention {c.retention:.3f}")

    console.print(f"[bold]calibrating[/] {len(docs)} docs, {len(qs)} queries, target {metric} >= {target:.3f}")
    res = run_calibration(docs, qs, labels, quality_target=target, metric=metric, reference=reference,
                          safety=safety, calibration_fraction=calibration_fraction, seed=seed, progress=progress)
    b = res.baseline
    table = Table(title="calibration result", show_header=False)
    table.add_row("[bold]Baseline[/]", "")
    table.add_row("  vectors / doc", f"{b['vectors_per_doc']:.1f}")
    table.add_row("  dimension", str(b["dim"]))
    table.add_row("  memory / doc (float32)", _kb(b["bytes_per_doc"]))
    if res.selected is None:
        table.add_row("[bold red]Selected[/]", "nothing met the target on the calibration queries")
    else:
        s, h = res.selected, res.holdout
        table.add_row("[bold]Selected[/]", s.label)
        table.add_row("  vectors / doc", f"{s.vectors_per_doc:.1f}")
        table.add_row("  dimension", str(s.dim))
        table.add_row("  quantization", f"{s.bits_per_dim:.3g} bits / dim")
        table.add_row("  compression vs float32", f"{s.compression_vs_float32:.1f}x")
        table.add_row("[bold]Measured quality retention[/]", f"reference: {res.reference}")
        table.add_row(f"  calibration ({res.n_calibration_queries} queries)",
                      f"{s.retention:.3f} [{s.retention_lo:.3f}, {s.retention_hi:.3f}]")
        ok = "[green]met[/]" if res.met_target_on_holdout else "[red]MISSED[/]"
        table.add_row(f"  held-out ({res.n_holdout_queries} queries)",
                      f"{h['retention']:.3f} [{h['retention_ci'][0]:.3f}, {h['retention_ci'][1]:.3f}]  {ok}")
        table.add_row("RAM / storage reduction", f"{1 - s.bytes_per_doc / b['bytes_per_doc']:.1%} "
                                                 f"({_kb(s.bytes_per_doc)}/doc)")
        change = h["query_ms"] / b["query_ms"] - 1.0 if b.get("query_ms") else float("nan")
        table.add_row("query time (exact scan, this machine)",
                      f"{b['query_ms']:.1f} -> {h['query_ms']:.1f} ms/query ({change:+.0%})")
    console.print(table)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        payload = res.summary() | {"candidates": [c.row() for c in res.candidates]}
        Path(out).write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
        console.print(f"[green]wrote[/] {out}")


SUITES = ("legacy", "merge", "quantize")


def _suite(name: str) -> dict[str, Any]:
    from .benchmark import legacy_variants
    from .compose import Pipeline
    from .stages import (
        AdaptiveMerge,
        BinaryQuantizer,
        Float16Quantizer,
        Int4Quantizer,
        Int8Quantizer,
        Lloyd2Quantizer,
        RandomPruner,
        RedundancyPruner,
    )

    if name == "legacy":
        v = legacy_variants()
        v.pop("baseline-float32")
        return v
    if name == "merge":
        out: dict[str, Any] = {}
        for r in (0.9, 0.8, 0.7, 0.6, 0.5):
            out[f"adaptive r={r}"] = Pipeline([AdaptiveMerge(radius=r)])
        for t in (0.92, 0.85):
            out[f"redundancy t={t}"] = Pipeline([RedundancyPruner(threshold=t)])
        out["random 25%"] = Pipeline([RandomPruner(ratio=0.25)])
        return out
    if name == "quantize":
        return {
            "float16": Pipeline([Float16Quantizer()]),
            "int8 fixed": Pipeline([Int8Quantizer()]),
            "int8 per-vector": Pipeline([Int8Quantizer("per_vector")]),
            "int4": Pipeline([Int4Quantizer()]),
            "lloyd2": Pipeline([Lloyd2Quantizer()]),
            "binary": Pipeline([BinaryQuantizer()]),
            "binary centred": Pipeline([BinaryQuantizer(center="mean")]),
        }
    raise typer.BadParameter(f"unknown suite {name!r}; choose from {SUITES}")


def benchmark(
    corpus: str = typer.Argument(..., help="corpus .npz or legacy encode cache"),
    queries: str = typer.Option(..., "--queries", "-q"),
    qrels: str | None = typer.Option(None),
    suite: str = typer.Option("merge", help=f"one of {', '.join(SUITES)}"),
    reference: str = typer.Option("labels", help="labels | baseline"),
    out: str = typer.Option("reports/universal", help="output directory"),
) -> None:
    """Measure a suite of pipelines against the float baseline (exact MaxSim)."""
    from .benchmark import Dataset, run_matrix, save_result, to_markdown

    pipes = _suite(suite)
    docs, qs, labels = load_task(corpus, queries, qrels, with_images=suite == "legacy")
    ds = Dataset(Path(corpus).stem, docs, qs, labels)
    res = run_matrix(ds, pipes, reference=reference,
                     progress=lambda name, row: console.print(f"  {name:28s} retention "
                                                              f"{row['retention:ndcg@5']:.3f}"))
    path = save_result(res, out, f"{ds.name}-{suite}")
    console.print(to_markdown(res))
    console.print(f"[green]wrote[/] {path}")


def compare(
    corpus: str = typer.Argument(..., help="corpus .npz or legacy encode cache"),
    queries: str = typer.Option(..., "--queries", "-q"),
    pipeline: list[str] = typer.Option(..., "--pipeline", "-p", help="repeat for each pipeline to compare"),  # noqa: B008
    qrels: str | None = typer.Option(None),
    reference: str = typer.Option("labels", help="labels | baseline"),
) -> None:
    """Side-by-side retention / size of explicit pipelines."""
    from .benchmark import Dataset, run_matrix, to_markdown

    pipes = {p: parse_pipeline(p) for p in pipeline}
    needs_images = any(s.name == "spatial" for pp in pipes.values() for s in pp.stages)
    docs, qs, labels = load_task(corpus, queries, qrels, with_images=needs_images)
    res = run_matrix(Dataset(Path(corpus).stem, docs, qs, labels), pipes, reference=reference, n_boot=500)
    console.print(to_markdown(res))


def register(app: typer.Typer) -> None:
    """Attach the universal-layer commands to the main ``optivision`` app."""
    for command in (inspect, compress, calibrate, benchmark, compare):
        app.command()(command)
