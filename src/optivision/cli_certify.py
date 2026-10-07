"""CLI commands for certified selection: ``optivision certify`` and ``optivision verify``.

    optivision certify docs.npz queries.npz --qrels qrels.json --target 0.97 --out artifact/
    optivision verify artifact/ --vectors docs.npz --queries queries.npz --qrels qrels.json

``certify`` tests the library's fixed candidate list on the queries and prints one of three
outcomes: ``select`` (the rule rejected "retention <= target" for at least one candidate and the
smallest was chosen), ``abstain`` (nothing was certified; the uncompressed index is the answer)
or ``fallback`` (only with an explicit fallback pipeline, which the CLI does not offer).
Abstaining is a normal result and exits with status 0, like a selection; read ``outcome`` in the
``--json`` output to branch on it. Invalid input exits 2. The queries are tested in a seeded
random order (``--order-seed``, default 0) unless ``--given-order`` asks for the order of the
file, which is valid only if that order is unrelated to the outcomes.

``verify`` prints every check of a certificate and exits 1 when any check failed. A passing
report means that no check failed: it does not make the certificate true, and it is only as good
as the queries, labels and corpus the certificate was issued for.

Vectors are read as in ``cli_universal``: a ``MultiVectorCorpus`` ``.npz`` (or a legacy encode
cache with ``--trust-pickle``). Labels are a JSON file ``{query id: [document ids]}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.text import Text

from .cli_universal import TRUST_PICKLE_HELP, load_corpus, load_task

console = Console()
_err = Console(stderr=True)

REFERENCES = ("auto", "labels", "agreement")
SPACES = ("default", "wide")

_STATUS_STYLE = {"pass": "green", "fail": "bold red", "skipped": "dim", "differs": "yellow"}


# ------------------------------------------------------------------ helpers


def _fail(message: str, code: int = 2) -> typer.Exit:
    """Print ``message`` on stderr and return the ``Exit`` to raise."""
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def _need_file(path: str, what: str) -> None:
    if not Path(path).is_file():
        raise typer.BadParameter(f"{what} {path!r} does not exist or is not a file")


def _read_qrels(path: str) -> dict[str, set[str]]:
    _need_file(path, "the labels file")
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"cannot read the labels file {path!r}: {exc}") from exc
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"{path!r} must hold a JSON object {{query id: [document ids]}}")
    for query_id, relevant in raw.items():
        if isinstance(relevant, dict):
            raise typer.BadParameter(
                f"{path!r}: the labels of query {query_id!r} are an object {{document: grade}}. "
                "Graded labels are not supported: every listed document would count as relevant, "
                "grade 0 included. List the relevant document ids, {query id: [document ids]}")
    try:
        return {str(k): set(v) for k, v in raw.items()}
    except TypeError as exc:
        raise typer.BadParameter(f"{path!r}: every value must be a list of document ids") from exc


def _say(label: str, value: Any, style: str = "") -> None:
    """One ``label  value`` line; values are never read as markup."""
    console.print(Text.assemble((f"  {label:<15}", "bold"), (str(value), style)), soft_wrap=True,
                  highlight=False)


def _paragraph(text: str, style: str = "", indent: int = 2) -> None:
    console.print(Text(" " * indent + text, style=style), soft_wrap=True, highlight=False)


# ------------------------------------------------------------------ certify


def _render_certify(report: dict[str, Any], n_docs: int, dim: int, artifact: Path | None) -> None:
    outcome = report["outcome"]
    space = report["space"]
    order = report["query_order"]
    console.print(Text(f"certify: {n_docs} documents, dimension {dim}", style="bold"),
                  soft_wrap=True, highlight=False)
    _say("reference", report["construct"])
    _say("metric", f"{report['metric']}   target {report['target']:g}   error "
                   f"{report['error']:g}   (per-test level {report['level']:.4g})")
    _say("candidate list", f"{space['name']}: {space['n_candidates']} candidates in "
                           f"{space['n_families']} families"
                           + ("; the default list registered for the development stores, not "
                              "validated for other encoders" if space["registered_default"] else ""))
    order_text = ("given (the order of the file)" if order["kind"] == "given"
                  else f"seeded permutation (seed {order['seed']})")
    _say("queries", f"{report['n_queries']} tested of {report['n_queries_given']} given "
                    f"({report['n_queries_without_relevant_document']} without a relevant "
                    f"document), order: {order_text}"
                    + ("" if order["budget"] is None else f", budget {order['budget']}")
                    + ("" if not report["n_exact_duplicate_queries"]
                       else f"; {report['n_exact_duplicate_queries']} repeat an earlier query "
                            "exactly"))
    console.print()

    if outcome == "select" and report["certified"]:
        head = ("select (the rule rejected 'retention <= target' for this candidate at the "
                f"stated error level, {report['error']:g}; claimed only under the assumptions "
                "and conditions below)")
        if report["construct"].startswith("agreement@"):
            head += "; agreement with the uncompressed ranking, not retrieval quality"
        style = "bold green"
    elif outcome == "select":
        head, style = "select (NOT certified)", "bold yellow"
    elif outcome == "abstain":
        head, style = "abstain (nothing certified; use the uncompressed index)", "bold"
    else:
        head, style = "fallback (NOT certified)", "bold yellow"
    console.print(Text.assemble(("outcome: ", "bold"), (head, style)), soft_wrap=True,
                  highlight=False)
    if report["selected"] is not None:
        chosen = report["selected"]
        where = "" if chosen["position"] < 0 else f" [position {chosen['position']}, family " \
                                                  f"{chosen['family']}]"
        _say("pipeline", f"{chosen['label']}{where}")
        _say("compression", f"{report['compression_ratio']:.3g}x vs float32 "
                            "(a size, not a quality figure)")
        _say("basis", ("sensitive to the coordinate basis" if report["basis_sensitive"]
                       else "not sensitive to the coordinate basis")
             + " (a property of the code, not a quality figure)")
    if report["not_certified_note"]:
        _paragraph(report["not_certified_note"])
    for caveat in report["caveats"]:
        _paragraph(caveat, style="yellow")
    console.print()
    console.print(Text("what the rule states (claimed only if every assumption and condition below "
                       "holds):", style="bold"), soft_wrap=True, highlight=False)
    _paragraph(report["statement"])
    for assumption in report["assumptions"]:
        _paragraph(assumption, style="dim", indent=4)
    _say("certificate", report["certificate_digest"])
    if artifact is not None:
        _say("artifact", str(artifact))


def certify(
    vectors: str = typer.Argument(..., help="document vectors: corpus .npz or legacy encode cache"),
    queries: str = typer.Argument(..., help="query vectors: queries .npz, or a legacy queries.json"),
    qrels: str | None = typer.Option(None, "--qrels",
                                     help="labels JSON {query id: \\[doc ids]}: binary, every "
                                          "listed document is relevant (graded labels are "
                                          "refused)"),
    target: float = typer.Option(0.97, "--target", help="retention to be exceeded, in (0, 1]"),
    error: float = typer.Option(0.05, "--error", help="error level of the statement, in (0, 1)"),
    metric: str = typer.Option("ndcg@5", "--metric", help="ndcg@k | recall@k | mrr@k | hit@k"),
    reference: str = typer.Option("auto", "--reference",
                                  help="auto | labels | agreement (agreement is not retrieval quality)"),
    space: str = typer.Option("default", "--space",
                              help="default | wide (wide is an engineering option the registered "
                                   "experiments did not evaluate)"),
    budget: int | None = typer.Option(None, "--budget", help="test only a prefix of N queries of the order"),
    order_seed: int | None = typer.Option(
        None, "--order-seed",
        help="seed of the random order in which the queries are tested (default 0, recorded). "
             "Choose it before you look and use it once: trying several seeds and reporting one "
             "is outside the statement"),
    given_order: bool = typer.Option(
        False, "--given-order",
        help="test the queries in the order of the file instead of a random order; valid only if "
             "that order is unrelated to the outcomes (never sorted by difficulty, date, source "
             "or any query attribute)"),
    out: str | None = typer.Option(None, "--out", help="write the artifact to this new directory"),
    as_json: bool = typer.Option(False, "--json", help="print the report as JSON on stdout"),
    trust_pickle: bool = typer.Option(False, "--trust-pickle", help=TRUST_PICKLE_HELP),
) -> None:
    """Select the smallest candidate the rule certifies at (target, error), or abstain.

    Abstaining (nothing certified, use the uncompressed index) is a normal result and exits 0.

    Run it once per (target, error): another look at the same queries is outside the statement.

    Queries are tested in a seeded random order; a sorted file cannot break the order assumption.

    Exact duplicate queries are counted and reported.
    """
    from .api import certify as run_certify
    from .core.selection import DEFAULT_ORDER_SEED, CandidateSpace
    from .evaluation import parse_metric

    if reference not in REFERENCES:
        raise typer.BadParameter(f"reference must be one of {', '.join(REFERENCES)}; got {reference!r}",
                                 param_hint="--reference")
    if space not in SPACES:
        raise typer.BadParameter(f"space must be one of {', '.join(SPACES)}; got {space!r}",
                                 param_hint="--space")
    try:
        parse_metric(metric)  # before any candidate list is built
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--metric") from exc
    if reference == "agreement" and qrels is not None:
        raise typer.BadParameter("--qrels is not used by reference=agreement; drop it, or use "
                                 "reference=labels", param_hint="--qrels")
    if given_order and order_seed is not None:
        raise typer.BadParameter("--given-order tests the queries in the order of the file and "
                                 "cannot be combined with --order-seed", param_hint="--given-order")
    seed = None if given_order else DEFAULT_ORDER_SEED if order_seed is None else order_seed
    _need_file(vectors, "the vectors file")
    _need_file(queries, "the queries file")
    if qrels is not None:
        _need_file(qrels, "the labels file")
        _read_qrels(qrels)  # refuses graded labels before any candidate is scored
    if out is not None and Path(out).exists() and (not Path(out).is_dir() or any(Path(out).iterdir())):
        raise typer.BadParameter(f"{out!r} already exists and is not empty; choose a new directory "
                                 "(refused now, before any candidate is scored)", param_hint="--out")

    try:
        docs, qs, labels = load_task(vectors, queries, qrels, trust_pickle=trust_pickle)
    except (OSError, ValueError, KeyError) as exc:
        raise _fail(f"cannot read the inputs: {exc}") from exc
    if reference == "labels" and labels is None:
        raise typer.BadParameter("reference=labels needs labels (--qrels or a legacy queries.json)")
    if reference == "agreement":
        labels = None  # a legacy queries.json carries labels; the user asked not to use them

    progress = None
    bar = None
    if not as_json:
        bar = Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                       TimeElapsedColumn(), console=_err, transient=True)
        task = bar.add_task("scoring candidates", total=None)

        def progress(position: int, total: int, family: str) -> None:
            bar.update(task, total=total, completed=position,
                       description=f"candidate {position}/{total} ({family})")

        bar.start()
    try:
        chosen_space = None if space == "default" else CandidateSpace.wide(docs.dimension)
        result = run_certify(docs, qs, labels, target=target, error=error, metric=metric,
                             reference=reference, space=chosen_space, query_budget=budget,
                             order_seed=seed, progress=progress)
    except ImportError as exc:
        raise _fail(str(exc), code=1) from exc
    except ValueError as exc:  # invalid input, including every ContractError
        raise _fail(str(exc)) from exc
    except RuntimeError as exc:  # a candidate that could not be evaluated
        raise _fail(str(exc), code=1) from exc
    finally:
        if bar is not None:
            bar.stop()

    artifact = None
    if out is not None:
        try:
            artifact = result.save(out)
        except (OSError, ValueError) as exc:
            raise _fail(f"cannot write the artifact: {exc}", code=1) from exc
    report = result.report()
    if as_json:
        payload = dict(report)
        payload["artifact"] = None if artifact is None else str(artifact)
        typer.echo(json.dumps(payload, indent=2, allow_nan=False))
    else:
        _render_certify(report, len(docs), docs.dimension, artifact)


# ------------------------------------------------------------------ verify


def verify(
    path: str = typer.Argument(..., help="an artifact directory (from certify --out) or a certificate.json"),
    vectors: str | None = typer.Option(None, "--vectors", help="corpus the certificate was issued for"),
    queries: str | None = typer.Option(None, "--queries", help="queries it was issued for"),
    qrels: str | None = typer.Option(None, "--qrels", help="labels it was issued for"),
    evidence: str | None = typer.Option(None, "--evidence", help="evidence .npz, for the replay check"),
    key_file: str | None = typer.Option(None, "--key-file",
                                        help="file whose exact bytes are the HMAC key"),
    strict_code: bool = typer.Option(False, "--strict-code",
                                     help="fail, instead of reporting 'differs', when the code differs"),
    as_json: bool = typer.Option(False, "--json", help="print the report as JSON on stdout"),
    trust_pickle: bool = typer.Option(False, "--trust-pickle", help=TRUST_PICKLE_HELP),
) -> None:
    """Check a certificate: integrity, replay from the evidence, binding to the live objects.

    Prints every check and exits 1 if any failed. Pass --vectors, --queries and --qrels to check
    that the certificate was issued for exactly those; an artifact alone cannot show that. A
    pass means that no check failed; it does not make the certificate true.
    """
    from .api import verify as run_verify
    from .core.evaluate import Evidence, LabelReference

    target = Path(path)
    if not target.exists():
        raise typer.BadParameter(f"{path!r} does not exist", param_hint="PATH")
    objects: dict[str, Any] = {}
    try:
        if vectors is not None:
            _need_file(vectors, "the vectors file")
            objects["corpus"] = load_corpus(vectors, trust_pickle=trust_pickle)
        if queries is not None:
            _need_file(queries, "the queries file")
            objects["queries"] = load_corpus(queries, trust_pickle=trust_pickle)
        if qrels is not None:
            objects["reference"] = LabelReference(_read_qrels(qrels))
        if evidence is not None:
            _need_file(evidence, "the evidence file")
            objects["evidence"] = Evidence.from_npz(evidence)
        key = None
        if key_file is not None:
            _need_file(key_file, "the key file")
            key = Path(key_file).read_bytes()
    except (OSError, ValueError) as exc:
        raise _fail(f"cannot read the inputs: {exc}") from exc
    try:
        report = run_verify(target, key=key, strict_code=strict_code, **objects)
    except (OSError, ValueError) as exc:  # an unreadable artifact or certificate is a failed check
        message = f"cannot load {path!r}: {exc}"
        if as_json:
            typer.echo(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            typer.echo(f"FAILED: {message}", err=True)
        raise typer.Exit(1) from exc

    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, allow_nan=False))
    else:
        width = max((len(c.name) for c in report.checks), default=0)
        for check in report.checks:
            console.print(Text.assemble(
                ("[", ""), (f"{check.status:<7}", _STATUS_STYLE[check.status]), ("] ", ""),
                (f"{check.name:<{width}}", "bold"), (f"  {check.detail}" if check.detail else "", "")),
                soft_wrap=True, highlight=False)
        for note in report.notes:
            _paragraph(f"note: {note}", style="yellow", indent=0)
        verdict = "OK: no check failed" if report.ok else f"FAILED: {len(report.failed())} check(s) failed"
        console.print(Text(f"{verdict} (level reached: {report.level})",
                           style="bold green" if report.ok else "bold red"),
                      soft_wrap=True, highlight=False)
        _paragraph("A pass means that no check failed. It does not make the certificate true: it is "
                   "as good as the queries, labels and corpus it was issued for. 'skipped' checks "
                   "were not run; 'differs' reports different code or libraries.", style="dim", indent=0)
    if not report.ok:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    """Attach ``certify`` and ``verify`` to the main ``optivision`` app."""
    app.command("certify")(certify)
    app.command("verify")(verify)
