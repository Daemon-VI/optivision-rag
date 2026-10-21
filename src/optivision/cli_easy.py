"""Easy-mode commands: ``options``, ``build``, ``find``, ``check``, ``doctor`` and ``quickstart``.

    optivision options vidore/colqwen2-v1.0-hf --target 0.97 --pages my_pdfs/
    optivision build my_pdfs/ --model vidore/colqwen2-v1.0-hf --target 0.97
    optivision find my_pdfs.ovx "What drove operating costs in 2023?"
    optivision check my_pdfs.ovx
    optivision doctor --model vidore/colqwen2-v1.0-hf
    optivision quickstart my_pdfs/

A thin layer over :mod:`optivision.easy` (FINAL_DESIGN 4.5 and 6.2). Each command imports the package
when it runs, so ``optivision --help`` and the older commands start as fast as before, and the
easy-mode modules stay free to import heavy libraries only when a command needs them. Every command
sits in the "Easy mode" help panel.

Conventions. Text output ends with a ``Next:`` line. ``--json`` prints one JSON document on stdout
(the object's ``as_dict()``) and sends anything else the library prints to stderr. Exit codes: 0 done,
1 a refusal or a failed check (``doctor`` and ``check`` use it to say "not fine"), 2 input that cannot
be used (an unknown model, a missing path, a target outside the allowed range). Standard output is
reconfigured with ``errors="replace"`` so a non-ASCII file name cannot crash a cp1252 console.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import typer

from .easy import _claim
from .easy._claim import Line, claim, lit

PANEL = "Easy mode"
DEFAULT_TARGET = 0.97
_STEPS = 6  # the quickstart wizard: FINAL_DESIGN 6.2

_UNITS = {"": 1, "b": 1, "k": 10**3, "kb": 10**3, "m": 10**6, "mb": 10**6, "g": 10**9, "gb": 10**9,
          "t": 10**12, "tb": 10**12, "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}


# --------------------------------------------------------------------------- small helpers


def _easy() -> Any:
    """``optivision.easy``, imported when a command runs (the package itself loads no heavy library)."""
    import optivision.easy as ov

    return ov


def _safe_streams() -> None:
    """Make printing a file name that the console cannot encode print ``?`` instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # a closed or detached stream
            pass


def _echo(text: str = "") -> None:
    typer.echo(text)


def _fail(message: str, code: int = 2) -> typer.Exit:
    """Print ``error: message`` on stderr and return the exit to raise (``raise _fail(...)``)."""
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def _quiet_if(flag: bool):
    """With ``--json`` whatever the library prints goes to stderr, so stdout holds the JSON alone."""
    return contextlib.redirect_stdout(sys.stderr) if flag else contextlib.nullcontext()


@contextlib.contextmanager
def _refusals(ov: Any):
    """Turn the library's refusals into one ``error:`` line and an exit code (2 input, 1 everything else)."""
    try:
        yield
    except (ov.UnknownModel, ov.AmbiguousModel, ov.TargetMismatch, ValueError, FileNotFoundError) as exc:
        raise _fail(str(exc) or type(exc).__name__, 2) from exc
    except ov.EasyError as exc:
        raise _fail(str(exc) or type(exc).__name__, 1) from exc


def _jsonable(obj: Any) -> Any:
    """Plain data for ``json.dumps``: ``as_dict()`` when the object has one, else its fields."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [_jsonable(v) for v in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                if not f.name.startswith("_")}
    item = getattr(obj, "item", None)  # a NumPy scalar
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return str(obj)


def _print_json(data: Any) -> None:
    """One JSON document on stdout. ASCII-only (``\\u`` escapes), so any console can show it."""
    typer.echo(json.dumps(_jsonable(data), indent=1))


def _render(obj: Any) -> str:
    """``str(obj)``; for a report that has ``lines`` but no readable ``__str__``, the lines joined."""
    text = str(obj)
    if text.startswith("<") and " object at 0x" in text and hasattr(obj, "lines"):
        text = "\n".join(line.render() if hasattr(line, "render") else str(line) for line in obj.lines)
    return text


def _strip_next(text: str) -> str:
    """Drop a closing ``Next:`` line: the wizard embeds library output and writes its own."""
    lines = text.rstrip().splitlines()
    if lines and lines[-1].startswith("Next:"):
        lines = lines[:-1]
    return "\n".join(lines).rstrip()


def _quote(arg: str) -> str:
    """An argument as it is typed in a shell: double quotes when it holds a space or a quote."""
    if arg and not re.search(r"[\s\"'&|<>^]", arg):
        return arg
    return '"' + arg.replace('"', '\\"') + '"'


def _close(obj: Any) -> None:
    closer = getattr(obj, "close", None)
    if callable(closer):
        closer()


def _parse_size(text: str) -> int:
    """``"50GB"``, ``"1.5 gb"``, ``"500000000"`` -> bytes (decimal units, as the sizes printed elsewhere)."""
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*", text)
    unit = _UNITS.get(match.group(2).lower()) if match else None
    if match is None or unit is None:
        raise _fail(f"cannot read the size {text!r}; write it like 50GB, 800MB or 12000000000", 2)
    return int(float(match.group(1)) * unit)


def _parse_target(text: str) -> float:
    """``97``, ``97%`` and ``0.97`` all mean 0.97."""
    raw = text.strip().rstrip("%").strip()
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"cannot read {text!r} as a target; write 97, 97% or 0.97") from None
    return value / 100 if value > 1 else value


def _pages_argument(text: str | None) -> int | str | None:
    """``--pages``: a folder (or file) that exists, else a page count."""
    if text is None:
        return None
    if Path(text).exists():
        return text
    if text.strip().isdigit():
        return int(text)
    raise _fail(f"--pages {text!r} is neither an existing folder nor a page count", 2)


def _read_text(path: str, what: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise _fail(f"{what} file {path} not found", 2)
    try:
        return p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise _fail(f"{what} file {path} is not UTF-8 text", 2) from None


def _read_queries(path: str) -> list[str]:
    """One question per line (blank lines skipped), or a ``.json`` list of strings."""
    text = _read_text(path, "queries")
    if Path(path).suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise _fail(f"{path} is not valid JSON: {exc}", 2) from exc
        if not isinstance(data, list) or not all(isinstance(q, str) for q in data):
            raise _fail(f"{path} must hold a JSON list of question strings", 2)
        queries = [q.strip() for q in data if q.strip()]
    else:
        queries = [line.strip() for line in text.splitlines() if line.strip()]
    if not queries:
        raise _fail(f"{path} holds no questions", 2)
    return queries


def _answer(entry: Any) -> Any:
    """One JSON answer: a page id, a ``[file, page]`` pair, or a list of either (a set of pages)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str) and isinstance(entry[1], int):
        return (entry[0], entry[1])
    if isinstance(entry, list) and entry:
        return {_answer(e) for e in entry if isinstance(_answer(e), (str, tuple))}
    raise ValueError(f"cannot read the answer {entry!r}")


def _read_answers(path: str, n_queries: int) -> list[Any]:
    """Answers, one per question and in the same order.

    A ``.txt`` file has one line per question: a page id such as ``my_pdfs/annual.pdf::p4``, or several
    separated by ``|`` when more than one page answers it. A ``.json`` file holds a list whose entries are
    a page id, a ``[file, page]`` pair, or a list of either.
    """
    text = _read_text(path, "answers")
    if Path(path).suffix.lower() == ".json":
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("expected a list")
            answers = [_answer(e) for e in data]
        except ValueError as exc:
            raise _fail(f"{path}: {exc}", 2) from exc
    else:
        lines = text.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        answers = []
        for number, line in enumerate(lines, 1):
            pages = [p.strip() for p in line.split("|") if p.strip()]
            if not pages:
                raise _fail(f"{path}, line {number}: the answer is empty", 2)
            answers.append(pages[0] if len(pages) == 1 else set(pages))
    if len(answers) != n_queries:
        raise _fail(f"{path} holds {len(answers)} answers for {n_queries} questions; give one per question", 2)
    return answers


def _question_kwargs(queries: str | None, answers: str | None) -> dict[str, Any]:
    """``queries=`` / ``answers=`` for ``build`` or ``check`` from the two files (answers need questions)."""
    if answers is not None and queries is None:
        raise _fail("--answers needs --queries: the answers are matched to the questions line by line", 2)
    if queries is None:
        return {}
    questions = _read_queries(queries)
    out: dict[str, Any] = {"queries": questions}
    if answers is not None:
        out["answers"] = _read_answers(answers, len(questions))
    return out


class _HoldNext:
    """Wraps stdout while a build prints: lets everything through except a line that starts ``Next:``.

    ``ov.build`` closes its summary with a Python call (``Next: index.search(...)``), which is of no use
    at a shell prompt. The command swallows that line and prints its own ``Next:`` (a ``find`` call)
    once the build returns, so the output still ends with exactly one. Anything that is not at the
    start of a line passes through at once, so progress written with carriage returns is not delayed.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._line_start = True
        self._swallowing = False
        self.swallowed: list[str] = []

    def write(self, text: str) -> int:
        for part in text.splitlines(keepends=True):
            ended = part.endswith(("\n", "\r"))
            if self._swallowing:
                self.swallowed[-1] += part
                self._swallowing = not ended
                self._line_start = ended
                continue
            if self._line_start and part.startswith("Next:"):
                self.swallowed.append(part)
                self._swallowing = not ended
                self._line_start = ended
                continue
            self._stream.write(part)
            self._line_start = ended
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:  # isatty, encoding, buffer ...
        return getattr(self._stream, name)


def _do_build(ov: Any, source: str, kwargs: dict[str, Any], *, as_json: bool) -> Any:
    """Run ``ov.build``. With ``--json`` its printing goes to stderr; otherwise its closing ``Next:`` is held."""
    if as_json:
        with _quiet_if(True):
            return ov.build(source, progress=False, **kwargs)
    with contextlib.redirect_stdout(_HoldNext(sys.stdout)):
        return ov.build(source, **kwargs)


def _index_summary(index: Any) -> dict[str, Any]:
    as_dict = getattr(index, "as_dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())
    keys = ("path", "target", "layout", "size", "n_pages", "evidence")
    return {k: _jsonable(getattr(index, k)) for k in keys if hasattr(index, k)}


def _find_call(index_path: Any) -> str:
    return f'optivision find {_quote(str(index_path))} "your question"'


# --------------------------------------------------------------------------- options


def options_command(
    model: str | None = typer.Argument(None, metavar="MODEL", help="a model id or short name; omit to list all"),
    target: float = typer.Option(DEFAULT_TARGET, "--target", help="share of full-precision search quality "
                                 "(nDCG@5) to keep, from 0.90 to 1.0"),
    pages: str | None = typer.Option(None, "--pages", help="a folder of PDFs or images, or a number of pages: "
                                     "adds size, disk and time"),
    html: str | None = typer.Option(None, "--html", help="also write the chart to FILE (.html or .svg)"),
    as_json: bool = typer.Option(False, "--json", help="print JSON instead of text"),
) -> None:
    """Plan an index: the layouts a model supports, what each keeps and how big it is.

    With no MODEL, lists every model we know, then a Start here block.

    With a MODEL, lists its layouts and marks the one that --target would use.

    --pages (a folder, or a number) adds the size, the temporary disk and the encode time.

    Nothing is encoded or downloaded.
    """
    _safe_streams()
    ov = _easy()
    pages_arg = _pages_argument(pages)
    with _refusals(ov):
        plan = ov.options(model, target=target, pages=pages_arg)
        if html is not None:
            plan.save(html)
    if as_json:
        _print_json(plan.as_dict())
        if html is not None:
            typer.echo(f"wrote {html}", err=True)
        return
    if html is not None:
        _echo(f"Chart      wrote {html}")
    _echo(str(plan))


# --------------------------------------------------------------------------- build


def build_command(
    source: str = typer.Argument(..., metavar="SOURCE", help="a folder of PDFs or images, or one file"),
    model: str = typer.Option(..., "--model", help="a model id or short name (see: optivision options)"),
    target: float | None = typer.Option(None, "--target", help="share of full-precision search quality "
                                        "(nDCG@5) to keep; 0.97 for a new index, the stored target for an "
                                        "existing one"),
    path: str | None = typer.Option(None, "--path", help="where the index goes; default: NAME.ovx next to "
                                    "the folder. An existing index is brought up to date"),
    work_dir: str | None = typer.Option(None, "--work-dir", help="folder for the temporary float16 copy; "
                                        "use one that survives a restart (Drive on Colab) to resume"),
    keep_float16: bool = typer.Option(False, "--keep-float16", help="keep the float16 copy so a later "
                                      "step-down needs no re-encoding"),
    sample: int | None = typer.Option(None, "--sample", min=1, help="build from a trial of N pages; the "
                                      "full build reuses the encoded pages"),
    queries: str | None = typer.Option(None, "--queries", help="a file of your real questions, one per line "
                                       "(or a JSON list); used to check the layout"),
    answers: str | None = typer.Option(None, "--answers", help="the page that answers each question, one "
                                       "line per question; several pages separated by |"),
    max_disk: str | None = typer.Option(None, "--max-disk", help="total disk the index may use, such as "
                                        "50GB; the highest target that fits is used"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="allow a model that runs "
                                           "code from its own repository"),
    as_json: bool = typer.Option(False, "--json", help="print a JSON summary instead of text"),
) -> None:
    """Build an index of a folder of PDFs or images, or bring an existing one up to date.

    Encodes the pages, picks the most compressed layout that keeps at least the target, and saves one folder (NAME.ovx).

    The same line run again adds new and changed files.

    Run `optivision options` beforehand to see the size, the disk and the time.
    """
    _safe_streams()
    ov = _easy()
    if not Path(source).exists():
        raise _fail(f"{source} does not exist", 2)
    kwargs: dict[str, Any] = {"model": model}
    kwargs.update(_question_kwargs(queries, answers))
    for key, value in (("target", target), ("path", path), ("work_dir", work_dir), ("sample", sample)):
        if value is not None:
            kwargs[key] = value
    if max_disk is not None:
        kwargs["max_disk"] = _parse_size(max_disk)
    if keep_float16:
        kwargs["keep_float16"] = True
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    with _refusals(ov):
        index = _do_build(ov, source, kwargs, as_json=as_json)
        try:
            summary = _index_summary(index)
            where = getattr(index, "path", None) or path or "INDEX"
        finally:
            _close(index)
    if as_json:
        _print_json(summary)
        return
    _echo(f"Next: {_find_call(where)}")


# --------------------------------------------------------------------------- find


def _resolve_index_path(text: str) -> str:
    """The index folder; ``my_pdfs`` finds ``my_pdfs.ovx`` when only that exists."""
    p = Path(text)
    if p.exists():
        return text
    alt = p.with_name(p.name + ".ovx")
    if alt.exists():
        return str(alt)
    raise _fail(f"no index at {text}. Build one with: optivision build FOLDER --model MODEL_ID", 2)


def _hit_dict(hit: Any) -> dict[str, Any]:
    as_dict = getattr(hit, "as_dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())
    return {name: _jsonable(getattr(hit, name, None)) for name in ("id", "file", "page", "score", "text")}


def _snippet(text: str, width: int = 100) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 3].rstrip() + "..."


def find_command(
    index: str = typer.Argument(..., metavar="INDEX", help="the index folder (NAME.ovx)"),
    query: str = typer.Argument(..., metavar="QUERY", help="your question, in quotes"),
    k: int = typer.Option(5, "-k", "--top-k", min=1, help="how many pages to return"),
    as_json: bool = typer.Option(False, "--json", help="print JSON instead of text"),
) -> None:
    """Search an index and print the best pages.

    The model loads on every run; a script with many questions should use `optivision.easy.load` once instead.
    """
    _safe_streams()
    ov = _easy()
    if not query.strip():
        raise _fail("the question is empty", 2)
    location = _resolve_index_path(index)
    with _refusals(ov):
        opened = ov.load(location)
        try:
            with _quiet_if(as_json):
                hits = list(opened.search(query, k=k))
        finally:
            _close(opened)
    if as_json:
        _print_json({"index": location, "query": query, "k": k, "hits": [_hit_dict(h) for h in hits]})
        return
    if not hits:
        _echo(Line("Results", "none for this question in {0}", lit(location)).render())
    else:
        _echo(Line("Results", "{0} pages in {1}; page numbers and scores are exact values from this index",
                   claim(len(hits), "counted"), lit(location)).render())
        for rank, hit in enumerate(hits, 1):
            page = getattr(hit, "page", None)
            score = float(getattr(hit, "score", 0.0))
            text = Line(
                "", "{0}. {1}{2}  score {3}", claim(rank, "counted"),
                lit(getattr(hit, "file", None) or getattr(hit, "id", "?")),
                claim(page, "counted", text=f", page {page}") if page is not None else lit(""),
                claim(score, "counted", text=f"{score:.2f}"),
            ).text()
            _echo(f"  {text}")
            snippet = _snippet(getattr(hit, "text", "") or "")
            if snippet:
                _echo(f"     {snippet}")
    _echo(f'Next: optivision find {_quote(location)} "another question" -k {2 * k}')


# --------------------------------------------------------------------------- check


def check_command(
    index: str = typer.Argument(..., metavar="INDEX", help="the index folder (NAME.ovx)"),
    queries: str | None = typer.Option(None, "--queries", help="a file of your real questions, one per line "
                                       "(or a JSON list); needs the float16 copy (--keep-float16 at build)"),
    answers: str | None = typer.Option(None, "--answers", help="the page that answers each question, one "
                                       "line per question; several pages separated by |"),
    apply: bool = typer.Option(False, "--apply", help="carry out the advised step-down (its time is "
                               "printed before it starts)"),
    as_json: bool = typer.Option(False, "--json", help="print JSON instead of text"),
) -> None:
    """Check that an index is still accurate: its build record, drift, size and layout.

    With --queries, also compares the stored layout with full precision on your questions.

    Add --answers to score search quality.

    Exits 1 when the check does not pass. Nothing changes unless you add --apply.
    """
    _safe_streams()
    ov = _easy()
    location = _resolve_index_path(index)
    kwargs = _question_kwargs(queries, answers)
    applied = False
    with _refusals(ov):
        opened = ov.load(location)
        try:
            with _quiet_if(as_json):
                report = opened.check(**kwargs)
            ok = bool(getattr(report, "ok", False))
            if as_json:
                _print_json(report.as_dict() if hasattr(report, "as_dict") else report)
            else:
                if apply and ok:
                    _echo("Nothing to apply: the check passed.")
                _echo(_render(report))
            if apply and not ok:
                with _quiet_if(as_json):
                    report.apply()
                applied = True
        finally:
            _close(opened)
    if not ok and not applied:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- doctor


def _doctor_ok(report: Any) -> bool:
    ok = getattr(report, "ok", None)
    if ok is None and hasattr(report, "as_dict"):
        ok = report.as_dict().get("ok")
    return True if ok is None else bool(ok)


def doctor_command(
    model: str | None = typer.Option(None, "--model", help="also check this model and print its install line"),
    as_json: bool = typer.Option(False, "--json", help="print JSON instead of text"),
) -> None:
    """Check this machine: Python, PyTorch and CUDA, GPU, disk, library versions and known problems.

    Exits 0 when nothing needs fixing and 1 when something does, so a script can stop early.
    """
    _safe_streams()
    ov = _easy()
    with _refusals(ov):
        report = ov.doctor(model)
    if as_json:
        _print_json(report.as_dict() if hasattr(report, "as_dict") else report)
    else:
        _echo(_render(report))
    if not _doctor_ok(report):
        raise typer.Exit(1)


# --------------------------------------------------------------------------- quickstart


def _is_interactive() -> bool:
    """True when questions can be asked: stdin is a terminal (a Colab ``!`` cell is not)."""
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return False
        if sys.platform == "win32":  # pragma: no cover - Windows only
            # the NUL device reports itself as a terminal; only a console answers GetConsoleMode
            import ctypes
            import msvcrt

            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            mode = ctypes.c_uint32()
            return bool(ctypes.windll.kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)))
        return True
    except (ValueError, AttributeError, OSError):
        return False


def _step(number: int, title: str) -> None:
    _echo(f"\n[{number}/{_STEPS}] {title}")


def _count_pages(folder: str) -> tuple[Any, str]:
    """``(PageCount, "")`` from the documents module, or ``(None, why)`` when it cannot be used here."""
    try:
        from .easy import _docs

        return _docs.count_pages(folder), ""
    except Exception as exc:  # no PDF reader installed, or a folder it cannot read: the plan still prints
        return None, str(exc) or type(exc).__name__


def _start_here_default(plan: Any) -> str | None:
    """The leading model of the Start here block, if the plan's data names one."""
    try:
        data = plan.as_dict()
    except Exception:
        return None
    entries = data.get("start_here") if isinstance(data, dict) else None
    for entry in entries or []:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            for key in ("model", "model_id", "id"):
                if isinstance(entry.get(key), str):
                    return entry[key]
    return None


def _float16_note(plan: Any, n_pages: int | None) -> str:
    """What keeping the temporary float16 copy costs, as a tagged line; empty when it cannot be worked out."""
    rows = getattr(getattr(plan, "selection", None), "rows", None) or ()
    row = next((r for r in rows if getattr(r, "name", "") == "float16"), None)
    per_page = getattr(row, "bytes_per_page", None)
    if not per_page or not n_pages:
        return ""
    nbytes = max(per_page) * n_pages
    cost = claim(nbytes, "estimate", "our bytes per page x your pages", text=_claim.size(nbytes))
    return Line("Float16", "keeping the temporary copy takes {0} on disk", cost).render()


def _build_call(folder: str | None, model: str | None, target: float, keep: bool = False) -> str:
    parts = ["optivision", "build", _quote(folder or "FOLDER"), "--model", _quote(model or "MODEL_ID"),
             "--target", f"{target:g}"]
    if keep:
        parts.append("--keep-float16")
    return " ".join(parts)


def quickstart_command(
    folder: str | None = typer.Argument(None, metavar="FOLDER", help="a folder of PDFs or images"),
    yes: bool = typer.Option(False, "--yes", "-y", help="ask nothing and build; needs FOLDER and --model"),
    model: str | None = typer.Option(None, "--model", help="a model id or short name"),
    target: float | None = typer.Option(None, "--target", help="share of full-precision search quality "
                                        "(nDCG@5) to keep; default 0.97"),
) -> None:
    """Guided setup: check this machine, count the pages, pick a model and a target, then build.

    Six steps, with the plan (size, disk, time) shown before anything is written.

    Where nobody can answer questions (a Colab `!` cell, a pipe) it prints the plan and the `build` line instead.

    --yes with FOLDER and --model builds with no questions at all.
    """
    _safe_streams()
    ov = _easy()
    if yes and (folder is None or model is None):
        raise _fail("--yes asks nothing, so it needs FOLDER and --model ID", 2)
    if folder is not None and not Path(folder).exists():
        raise _fail(f"{folder} does not exist", 2)
    interactive = _is_interactive() and not yes
    chosen_target = DEFAULT_TARGET if target is None else target
    _echo("OptiVision quickstart. The plan is shown before anything is written.")

    with _refusals(ov):
        # 1. this machine
        _step(1, "Checking this machine")
        report = ov.doctor(model)
        _echo(_strip_next(_render(report)))
        if not _doctor_ok(report):
            if yes:
                raise _fail("doctor found problems; fix them, or run `optivision build` directly", 1)
            if interactive and not typer.confirm("Continue anyway?", default=False):
                raise typer.Exit(1)

        if folder is None and interactive:
            folder = typer.prompt("Folder of PDFs or images")
            if not Path(folder).exists():
                raise _fail(f"{folder} does not exist", 2)

        # 2. the pages
        _step(2, "Counting pages" + (f" in {folder}" if folder else ""))
        n_pages: int | None = None
        if folder is None:
            _echo("No folder given; the plan below is for models only.")
        else:
            found, why = _count_pages(folder)
            if found is None:
                _echo(f"Pages are counted in the plan below ({why}).")
            else:
                n_pages = int(found.n_pages)
                _echo(Line("Pages", "{0} in {1} files", claim(n_pages, "counted"),
                           claim(int(found.n_files), "counted")).render())
                for path_, reason in list(getattr(found, "skipped", ()) or ())[:5]:
                    _echo(f"           skipped {path_}: {reason}")

        # 3. the model
        _step(3, "Choosing a model")
        if model is not None:
            _echo(f"Model      {model}")
        else:
            table = ov.options(None, target=chosen_target, pages=n_pages if n_pages else folder)
            _echo(_strip_next(str(table)))
            suggestion = _start_here_default(table)
            if interactive:
                while model is None:
                    answer = typer.prompt("Model id", default=suggestion) if suggestion else typer.prompt("Model id")
                    try:
                        ov.options(answer, target=chosen_target)
                        model = answer
                    except (ov.UnknownModel, ov.AmbiguousModel) as exc:
                        typer.echo(f"error: {exc}", err=True)
            else:
                model = suggestion

        # 4. the target and the plan
        _step(4, "Target and size")
        if interactive and target is None:
            while True:
                answer = typer.prompt("Target: share of full-precision search quality to keep",
                                      default=_claim.pct(DEFAULT_TARGET, 0))
                try:
                    chosen_target = _parse_target(answer)
                    ov.options(model, target=chosen_target)
                    break
                except ValueError as exc:
                    typer.echo(f"error: {exc}", err=True)
        _echo("The target is a share of the model's own full-precision search quality, not the share of "
              "answers that are right.")
        keep = False
        if model is not None:
            plan = ov.options(model, target=chosen_target, pages=folder if folder else None)
            _echo(_strip_next(str(plan)))
            note = _float16_note(plan, n_pages)
            if note:
                _echo(note)
            if interactive:
                keep = typer.confirm("Keep the temporary float16 copy, so a later step-down needs no "
                                     "re-encoding?", default=False)

        # 5. the build
        call = _build_call(folder, model, chosen_target, keep)
        if folder is None or model is None:
            reason = "no folder or no model was given"
        elif yes:
            reason = ""
        elif interactive:
            reason = "" if typer.confirm("Build now?", default=True) else "you chose not to"
        else:
            reason = "no one can answer questions here (a pipe, or a notebook cell)"
        if reason:
            _step(5, "Not building")
            _echo(f"Nothing was written: {reason}.")
            _step(6, "Next")
            _echo(f"Next: {call}")
            return
        _step(5, "Building the index")
        kwargs: dict[str, Any] = {"model": model}
        if not (yes and target is None):  # unattended without --target: the library's rule applies
            kwargs["target"] = chosen_target
        if keep:
            kwargs["keep_float16"] = True
        built = _do_build(ov, folder, kwargs, as_json=False)
        try:
            where = getattr(built, "path", None) or "INDEX"
        finally:
            _close(built)

        # 6. done
        _step(6, "Done")
        _echo(f"Next: {_find_call(where)}")


# --------------------------------------------------------------------------- registration

#: command name -> function; the panel and the names are checked by tests/test_easy_cli.py
COMMANDS = (
    ("options", options_command),
    ("build", build_command),
    ("find", find_command),
    ("check", check_command),
    ("doctor", doctor_command),
    ("quickstart", quickstart_command),
)


def _registered_names(app: typer.Typer) -> set[str]:
    names = set()
    for info in app.registered_commands:
        callback_name = getattr(info.callback, "__name__", "")
        names.add(info.name or callback_name.replace("_", "-"))
    return names


def register(app: typer.Typer) -> None:
    """Attach the six easy-mode commands to the main ``optivision`` app, in the "Easy mode" help panel.

    Typer replaces a command registered twice under one name without a word, so a name that is
    already taken (``models`` for one) raises here instead.
    """
    clash = _registered_names(app) & {name for name, _ in COMMANDS}
    if clash:
        raise RuntimeError(f"cannot register easy-mode commands over existing ones: {sorted(clash)}")
    for name, function in COMMANDS:
        app.command(name, rich_help_panel=PANEL)(function)
