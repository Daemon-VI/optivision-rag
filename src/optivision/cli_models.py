"""CLI command for the model registry: ``optivision models``.

    optivision models                          # every known model, with its status
    optivision models --status measured        # only the measured ones
    optivision models vidore/colqwen2.5-v0.2   # one model: how it loads, what was measured
    optivision models vidore/colqwen2.5-v0.2 --min-retention 0.97

The status of a model says how much of its path this repository has exercised: ``measured``
(compression measured on its real output), ``untested`` (loading path known to work: shared with a
measured sibling, or checked here with ``check_retriever``; compression not measured here) or ``unverified`` (written against the model
card, never run here). The measured rows are in-sample development figures for the named
datasets; they show what is plausible for a model and are not a guarantee for another
corpus. ``optivision certify`` on your own queries is what decides.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

console = Console()
_err = Console(stderr=True)

STATUSES = ("measured", "untested", "unverified")


def _listing(status: str | None, as_json: bool) -> None:
    from .adapters import supported_models
    from .adapters.measured import measured_models

    with_rows = set(measured_models())
    models = supported_models(status)
    if as_json:
        typer.echo(json.dumps([{"model_id": m.model_id, "family": m.family, "modality": m.modality,
                                "loader": m.loader, "dimension": m.dimension, "status": m.status,
                                "license": m.license, "measured_rows": m.model_id in with_rows,
                                "evidence": m.evidence} for m in models], indent=1))
        return
    table = Table(title=f"{len(models)} models" + (f" with status {status}" if status else ""))
    for column in ("model", "modality", "dim", "loader", "status", "rows"):
        table.add_column(column)
    for m in models:
        table.add_row(m.model_id, m.modality, "?" if m.dimension is None else str(m.dimension), m.loader,
                      m.status, "yes" if m.model_id in with_rows else "")
    console.print(table)
    console.print("status: measured = compression measured here on real output; untested = loading path known to "
                  "work (measured sibling, or checked here), compression not measured; unverified = written against the model card, never run here.\n"
                  "rows = fixed-pipeline measurements available: optivision models <model id>")


def _detail(model_id: str, min_retention: float, as_json: bool) -> None:
    from .adapters import MODELS
    from .adapters.measured import measured_models, measured_results, measured_shortlist

    info = MODELS.get(model_id)
    if info is None:
        _err.print(f"[red]{model_id} is not in the registry.[/red] Any late-interaction model can still be used: "
                   "pass its vectors as arrays, or load it with load_adapter(model_id, loader=...).")
        raise typer.Exit(2)
    has_rows = model_id in measured_models()
    shortlist = measured_shortlist(model_id, min_retention) if has_rows else []
    if as_json:
        typer.echo(json.dumps({"model_id": info.model_id, "family": info.family, "modality": info.modality,
                               "loader": info.loader, "dimension": info.dimension, "status": info.status,
                               "license": info.license, "revision": info.revision, "notes": info.notes,
                               "evidence": info.evidence,
                               "measured": measured_results(model_id) if has_rows else {},
                               "shortlist": shortlist, "min_retention": min_retention}, indent=1))
        return
    console.print(f"[bold]{info.model_id}[/bold]  ({info.modality}, {info.dimension or '?'}-d, {info.status})")
    console.print(f"loader: {info.loader}" + (f"   licence: {info.license}" if info.license else ""))
    if info.revision:
        console.print(f"revision of the facts above: {info.revision}")
    if info.notes:
        console.print(f"notes: {info.notes}")
    console.print(f"evidence: {info.evidence}")
    if not has_rows:
        console.print("\nNo fixed-pipeline measurements of this model are stored. Encode your pages and run "
                      "`optivision certify` on your own queries.")
        return
    results = measured_results(model_id)
    for dataset, entry in results.items():
        note = f"  [{entry['note']}]" if entry.get("note") else ""
        console.print(f"\n{dataset}: {entry['n_docs']} documents, {entry['n_queries']} queries, float "
                      f"{entry['metric']} {entry['baseline']:.4f}, {len(entry['rows'])} pipelines measured{note}")
    table = Table(title=f"pipelines whose interval stayed at or above {min_retention:.3g} on every measured dataset")
    table.add_column("pipeline")
    for dataset in results:
        table.add_column(dataset.split("/")[-1] + "\nx float32 / retention [interval]")
    for row in shortlist[:15]:
        cells = [f"{d['compression']:.1f}x / {d['retention']:.3f} [{d['retention_interval'][0]:.3f}, "
                 f"{d['retention_interval'][1]:.3f}]" for d in (row["datasets"][name] for name in results)]
        table.add_row(row["label"], *cells)
    console.print(table)
    if len(shortlist) > 15:
        console.print(f"({len(shortlist) - 15} more; --json lists all)")
    console.print("These are in-sample development measurements on the datasets named, and the largest one is "
                  "the best of many on the same queries. They show what to try, not what will hold on your "
                  "corpus: `optivision certify` on your own queries decides.")


def models(
    model_id: str = typer.Argument(None, help="a model id, to show that model; omit to list every model"),
    status: str = typer.Option(None, "--status", help="list only models with this status: " + ", ".join(STATUSES)),
    min_retention: float = typer.Option(0.97, "--min-retention",
                                        help="shortlist pipelines whose interval stayed at or above this"),
    as_json: bool = typer.Option(False, "--json", help="print JSON instead of tables"),
) -> None:
    """List the known models, or show how one loads and what was measured for it."""
    if status is not None and status not in STATUSES:
        _err.print(f"[red]--status must be one of {', '.join(STATUSES)}[/red]")
        raise typer.Exit(2)
    if model_id is None:
        _listing(status, as_json)
    else:
        _detail(model_id, min_retention, as_json)


def register(app: typer.Typer) -> None:
    """Attach ``models`` to the main ``optivision`` app."""
    app.command("models")(models)
