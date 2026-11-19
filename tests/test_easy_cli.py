"""The easy-mode commands (``optivision options / build / find / check / doctor / quickstart``).

The functions behind them (``optivision.easy.options``, ``build``, ``load``, ``doctor``) are written by
other modules, so these tests put small fakes in ``optivision.easy`` and check what the command line does
with them: which arguments reach the library, what is printed, the exit codes, ``--json``, and that the 16
older commands are untouched. Nothing here encodes a page or needs a GPU.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import typer
import typer.main
from typer.testing import CliRunner

import optivision.easy
from optivision import cli_easy
from optivision.cli import app
from optivision.easy import EasyError, Hit, LayoutRow, UnknownModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
runner = CliRunner()

#: the 16 commands that existed before easy mode (typer replaces a duplicate name without a word)
EXISTING = ["make-corpus", "fetch-vidore", "index", "search", "stats", "bench", "explain", "init-config",
            "inspect", "compress", "calibrate", "benchmark", "compare", "certify", "verify", "models"]
EASY = ["options", "build", "find", "check", "doctor", "quickstart"]

#: words and phrases no user-facing text may hold (docs/easy/CONTRACT.md; tests/test_docs_examples.py)
BANNED = ("universal", "lossless", "guaranteed", "state of the art", "state-of-the-art", "sota", "veto",
          "works with every model", "novel", "better than", "faster", "scales to")
BANNED_WORDS = ("simply", "just", "first")

MODEL = "vidore/colqwen2-v1.0-hf"


def _commands(typer_app) -> dict:
    return typer.main.get_command(typer_app).commands


def _lint(text: str) -> None:
    low = text.lower()
    for phrase in BANNED:
        assert phrase not in low, phrase
    words = set(low.replace(",", " ").replace(".", " ").replace(":", " ").split())
    for word in BANNED_WORDS:
        assert word not in words, word


# --------------------------------------------------------------------------------- fakes


class FakeOptions:
    def __init__(self, text: str, data: dict | None = None, rows: tuple = ()):
        self.text, self.data = text, data or {"rows": []}
        self.selection = types.SimpleNamespace(rows=rows)
        self.saved: list[str] = []

    def __str__(self) -> str:
        return self.text

    def as_dict(self) -> dict:
        return self.data

    def save(self, path) -> None:
        self.saved.append(str(path))
        Path(path).write_text("<svg></svg>", encoding="utf-8")


class FakeIndex:
    def __init__(self, path="my_pdfs.ovx", hits=(), report=None):
        self.path = path
        self.target = 0.97
        self.layout = LayoutRow(name="hm(0.25) > int8", plain="merged + 8-bit", spec={"stages": []}, chain="8-bit",
                                bytes_per_page=(23851.0, 24357.0), ratio_f16=7.9, ratio_f32=15.8,
                                lowest_kept=0.972, worst_point=0.99)
        self.size = {"total": 10_600_000, "vectors": 10_000_000}
        self.n_pages = 412
        self.evidence = {"catalogue": "ViDoRe V1"}
        self.hits, self.report = list(hits), report
        self.closed = False
        self.searched: list[tuple] = []
        self.checked: list[dict] = []

    def search(self, query, k=5):
        self.searched.append((query, k))
        return self.hits[:k]

    def check(self, **kwargs):
        self.checked.append(kwargs)
        return self.report

    def close(self):
        self.closed = True


class FakeReport:
    def __init__(self, ok=True, lines=("Record     matches", "Drift      none"), applied=None):
        self.ok, self.lines, self.applied = ok, list(lines), applied if applied is not None else []

    def __str__(self) -> str:
        return "\n".join([*self.lines, "Next: index.check(queries=...)"])

    def as_dict(self) -> dict:
        return {"ok": self.ok, "lines": self.lines}

    def apply(self) -> None:
        self.applied.append(True)
        print("re-encoding takes about 5 min")


class FakeDoctor:
    def __init__(self, ok=True):
        self.ok = ok

    def __str__(self) -> str:
        if self.ok:
            return "Python     3.11\nTorch      CPU build\nNext: ov.options()"
        return "Torch      broken\nNext: pip install x"

    def as_dict(self) -> dict:
        return {"ok": self.ok}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Install fakes for ``options``, ``build``, ``load`` and ``doctor`` in ``optivision.easy``.

    The working directory becomes a temporary one holding an empty ``my_pdfs.ovx`` folder, which
    stands in for an index (the fake ``load`` ignores what is in it).
    """
    (tmp_path / "my_pdfs.ovx").mkdir()
    monkeypatch.chdir(tmp_path)
    state = types.SimpleNamespace(
        options_calls=[], build_calls=[], load_calls=[], doctor_calls=[],
        options_text=f"{MODEL}, target 97%\npick   merged + 8-bit\nNext: ov.build(...)",
        index=FakeIndex(), doctor=FakeDoctor(), build_print="Size       10.6 MB\nNext: index.search(\"question\")",
        options_error=None, build_error=None,
    )

    def options(model=None, target=None, *, pages=None, accuracy=None):
        state.options_calls.append({"model": model, "target": target, "pages": pages})
        if state.options_error is not None:
            raise state.options_error
        if model == "nope":
            raise UnknownModel("nope is not a model we know; closest: a, b, c")
        return FakeOptions(state.options_text, {"model": model, "target": target, "rows": [{"name": "float16"}]})

    def build(source, model=None, target=None, **kwargs):
        state.build_calls.append({"source": source, "model": model, "target": target, **kwargs})
        if state.build_error is not None:
            raise state.build_error
        print(state.build_print)
        return state.index

    def load(path, **kwargs):
        state.load_calls.append(path)
        return state.index

    def doctor(model=None):
        state.doctor_calls.append(model)
        if model == "nope":
            raise UnknownModel("nope is not a model we know")
        return state.doctor

    namespace = vars(optivision.easy)
    for name, function in (("options", options), ("build", build), ("load", load), ("doctor", doctor)):
        monkeypatch.setitem(namespace, name, function)
    return state


def invoke(*args, input=None):
    return runner.invoke(app, list(args), input=input)


# --------------------------------------------------------------------------------- registration


def test_the_sixteen_older_commands_are_present_before_and_after_registration():
    # "before": the app as it was built without the easy-mode line (a fresh process with register() disabled)
    script = (
        "import json, optivision.cli_easy as e\n"
        "e.register = lambda app: None\n"
        "import optivision.cli as c\n"
        "print(json.dumps(sorted(cmd.name or cmd.callback.__name__.replace('_', '-') "
        "for cmd in c.app.registered_commands)))\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": str(SRC)}, check=False)
    assert done.returncode == 0, done.stderr
    before = set(json.loads(done.stdout))
    assert before == set(EXISTING)
    after = set(_commands(app))
    assert after == before | set(EASY)
    assert not before & set(EASY)


def test_no_command_name_is_registered_twice():
    names = [info.name or info.callback.__name__.replace("_", "-") for info in app.registered_commands]
    assert len(names) == len(set(names)) == len(EXISTING) + len(EASY)


def test_models_is_not_registered_again():
    assert [i.name for i in app.registered_commands].count("models") == 1


def test_the_new_commands_sit_in_the_easy_mode_panel():
    commands = _commands(app)
    for name in EASY:
        assert commands[name].rich_help_panel == "Easy mode", name
    for name in EXISTING:
        assert getattr(commands[name], "rich_help_panel", None) != "Easy mode", name


def test_register_refuses_a_name_that_is_taken():
    other = typer.Typer()

    @other.command("doctor")
    def _taken() -> None:  # pragma: no cover - never run
        pass

    with pytest.raises(RuntimeError, match="doctor"):
        cli_easy.register(other)


def test_register_adds_exactly_the_six_commands():
    fresh = typer.Typer()
    cli_easy.register(fresh)
    assert sorted(_commands(fresh)) == sorted(EASY)


def test_importing_the_cli_loads_no_heavy_library():
    script = ("import sys, optivision.cli, optivision.cli_easy\n"
              "bad = [m for m in ('torch','scipy','transformers','sentence_transformers','colpali_engine',"
              "'pypdfium2') if m in sys.modules]\n"
              "print(bad)\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": str(SRC)}, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


# --------------------------------------------------------------------------------- help


def test_the_main_help_shows_the_easy_mode_panel_with_the_six_commands():
    r = invoke("--help")
    assert r.exit_code == 0
    assert "Easy mode" in r.output
    for name in EASY:
        assert name in r.output


@pytest.mark.parametrize("name, flags", [
    ("options", ["--target", "--pages", "--html", "--json"]),
    ("build", ["--model", "--target", "--path", "--work-dir", "--keep-float16", "--sample", "--queries",
               "--answers", "--max-disk", "--trust-remote-code", "--json"]),
    ("find", ["-k", "--json"]),
    ("check", ["--queries", "--answers", "--apply"]),
    ("doctor", ["--model"]),
    ("quickstart", ["--yes", "--model", "--target"]),
])
def test_each_command_documents_its_options(name, flags):
    r = invoke(name, "--help")
    assert r.exit_code == 0, r.output
    for flag in flags:
        assert flag in r.output, (name, flag)
    _lint(r.output)


def test_the_help_of_the_main_app_is_free_of_banned_words():
    _lint(" ".join(c.help or "" for n, c in _commands(app).items() if n in EASY))


def test_help_does_not_call_the_library(monkeypatch):
    # --help is answered by the command line alone: no easy-mode function is looked up
    def boom(*args, **kwargs):
        raise AssertionError("the library was called")

    for name in ("options", "build", "load", "doctor"):
        monkeypatch.setitem(vars(optivision.easy), name, boom)
    for name in EASY:
        assert invoke(name, "--help").exit_code == 0


# --------------------------------------------------------------------------------- options


def test_options_prints_the_text_and_passes_the_arguments(fake):
    r = invoke("options", MODEL, "--target", "0.95")
    assert r.exit_code == 0, r.output
    assert fake.options_calls == [{"model": MODEL, "target": 0.95, "pages": None}]
    assert "merged + 8-bit" in r.stdout
    assert r.stdout.rstrip().splitlines()[-1].startswith("Next:")


def test_options_default_target_is_97_percent(fake):
    invoke("options", MODEL)
    assert fake.options_calls[0]["target"] == 0.97


def test_options_without_a_model_lists_all(fake):
    r = invoke("options")
    assert r.exit_code == 0
    assert fake.options_calls[0]["model"] is None


def test_options_pages_is_a_count_or_a_folder(fake, tmp_path):
    invoke("options", MODEL, "--pages", "500")
    invoke("options", MODEL, "--pages", str(tmp_path))
    assert fake.options_calls[0]["pages"] == 500
    assert fake.options_calls[1]["pages"] == str(tmp_path)
    r = invoke("options", MODEL, "--pages", "no_such_folder")
    assert r.exit_code == 2 and "neither an existing folder nor a page count" in r.stderr


def test_options_json_is_the_objects_as_dict_alone_on_stdout(fake):
    r = invoke("options", MODEL, "--json")
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert data["model"] == MODEL and data["rows"] == [{"name": "float16"}]


def test_options_html_saves_the_chart(fake, tmp_path):
    out = tmp_path / "chart.html"
    r = invoke("options", MODEL, "--html", str(out))
    assert r.exit_code == 0, r.output
    assert out.read_text(encoding="utf-8") == "<svg></svg>"
    assert "chart.html" in r.stdout
    assert r.stdout.rstrip().splitlines()[-1].startswith("Next:")
    r = invoke("options", MODEL, "--html", str(out), "--json")
    assert json.loads(r.stdout)["model"] == MODEL  # stdout stays JSON


def test_options_unknown_model_is_an_input_error(fake):
    r = invoke("options", "nope")
    assert r.exit_code == 2
    assert "error: nope is not a model we know" in r.stderr and r.stdout == ""


def test_options_target_out_of_range_is_an_input_error(fake):
    fake.options_error = ValueError("target must be between 0.90 and 1.0, got 0.5")
    r = invoke("options", MODEL, "--target", "0.5")
    assert r.exit_code == 2 and "between 0.90 and 1.0" in r.stderr


# --------------------------------------------------------------------------------- build


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "my_pdfs"
    d.mkdir()
    return d


def test_build_passes_only_what_was_given(fake, folder):
    r = invoke("build", str(folder), "--model", MODEL)
    assert r.exit_code == 0, r.output
    assert fake.build_calls == [{"source": str(folder), "model": MODEL, "target": None}]


def test_build_passes_every_flag(fake, folder, tmp_path):
    queries = tmp_path / "q.txt"
    queries.write_text("what drove costs?\n\nwhat was revenue?\n", encoding="utf-8")
    answers = tmp_path / "a.txt"
    answers.write_text("my_pdfs/a.pdf::p4\nmy_pdfs/a.pdf::p2 | my_pdfs/b.pdf::p1\n", encoding="utf-8")
    r = invoke("build", str(folder), "--model", MODEL, "--target", "0.95", "--path", "x.ovx", "--work-dir", "w",
               "--keep-float16", "--sample", "300", "--queries", str(queries), "--answers", str(answers),
               "--max-disk", "2GB", "--trust-remote-code")
    assert r.exit_code == 0, r.output
    call = fake.build_calls[0]
    assert call["target"] == 0.95 and call["path"] == "x.ovx" and call["work_dir"] == "w"
    assert call["keep_float16"] is True and call["sample"] == 300 and call["trust_remote_code"] is True
    assert call["max_disk"] == 2_000_000_000
    assert call["queries"] == ["what drove costs?", "what was revenue?"]
    assert call["answers"] == ["my_pdfs/a.pdf::p4", {"my_pdfs/a.pdf::p2", "my_pdfs/b.pdf::p1"}]


def test_build_prints_the_summary_and_ends_with_a_find_call(fake, folder):
    r = invoke("build", str(folder), "--model", MODEL)
    lines = r.stdout.rstrip().splitlines()
    assert lines[0] == "Size       10.6 MB"
    assert "index.search" not in r.stdout  # the library's python-flavoured closing line is held back
    assert lines[-1] == 'Next: optivision find my_pdfs.ovx "your question"'
    assert sum(line.startswith("Next:") for line in lines) == 1
    assert fake.index.closed


def test_build_keeps_a_summary_that_has_no_next_line(fake, folder):
    fake.build_print = "Size       10.6 MB"
    r = invoke("build", str(folder), "--model", MODEL)
    assert r.stdout.rstrip().splitlines() == ["Size       10.6 MB", 'Next: optivision find my_pdfs.ovx "your question"']


def test_build_json_holds_the_summary_alone_on_stdout(fake, folder):
    r = invoke("build", str(folder), "--model", MODEL, "--json")
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["path"] == "my_pdfs.ovx" and data["n_pages"] == 412 and data["target"] == 0.97
    assert data["layout"]["name"] == "hm(0.25) > int8" and data["layout"]["bytes_per_page"] == [23851.0, 24357.0]
    assert fake.build_calls[0]["progress"] is False
    assert "Size       10.6 MB" in r.stderr  # whatever the library prints goes to stderr


def test_build_needs_a_model_and_an_existing_source(fake, folder):
    assert invoke("build", str(folder)).exit_code == 2
    r = invoke("build", str(folder / "missing"), "--model", MODEL)
    assert r.exit_code == 2 and "does not exist" in r.stderr
    assert fake.build_calls == []


def test_build_answers_need_queries_and_must_match_them(fake, folder, tmp_path):
    answers = tmp_path / "a.txt"
    answers.write_text("a::p1\n", encoding="utf-8")
    queries = tmp_path / "q.txt"
    queries.write_text("one\ntwo\n", encoding="utf-8")
    r = invoke("build", str(folder), "--model", MODEL, "--answers", str(answers))
    assert r.exit_code == 2 and "--answers needs --queries" in r.stderr
    r = invoke("build", str(folder), "--model", MODEL, "--queries", str(queries), "--answers", str(answers))
    assert r.exit_code == 2 and "1 answers for 2 questions" in r.stderr
    assert fake.build_calls == []


def test_build_bad_max_disk_is_an_input_error(fake, folder):
    r = invoke("build", str(folder), "--model", MODEL, "--max-disk", "lots")
    assert r.exit_code == 2 and "cannot read the size" in r.stderr


@pytest.mark.parametrize("error, code", [
    (optivision.easy.WiringError("wiring failed: 3 of 12 pages found"), 1),
    (optivision.easy.NotEnoughDisk("needs 3.2 GB, 1.1 GB free"), 1),
    (optivision.easy.RemoteCodeRefused("runs its own code; pass --trust-remote-code"), 1),
    (optivision.easy.MissingExtra('pip install "optivision-rag[easy-st]"'), 1),
    (optivision.easy.FeatureOff("small-memory layout is switched off: V9 not run"), 1),
    (optivision.easy.TargetMismatch("built at 0.95; pass rebuild=True"), 2),
    (UnknownModel("nope"), 2),
    (ValueError("target must be at least 0.90"), 2),
])
def test_build_refusals_become_one_error_line_and_an_exit_code(fake, folder, error, code):
    fake.build_error = error
    r = invoke("build", str(folder), "--model", MODEL)
    assert r.exit_code == code, r.output
    assert r.stderr.startswith("error: ") and str(error) in r.stderr
    assert "Traceback" not in r.output


def test_a_bug_in_the_library_is_not_hidden(fake, folder):
    fake.build_error = RuntimeError("boom")
    r = invoke("build", str(folder), "--model", MODEL)
    assert r.exit_code == 1 and isinstance(r.exception, RuntimeError)


# --------------------------------------------------------------------------------- find


@pytest.fixture
def hits():
    return [
        Hit(id="my_pdfs/annual.pdf::p4", file="my_pdfs/annual.pdf", page=4, score=12.3456, text="Operating costs rose"),
        Hit(id="my_pdfs/q3.pdf::p2", file="my_pdfs/q3.pdf", page=2, score=9.5, text=""),
        Hit(id="7", file=None, page=None, score=3.25),
    ]


def test_find_prints_the_pages_and_ends_with_next(fake, hits):
    fake.index = FakeIndex(hits=hits)
    r = invoke("find", "my_pdfs.ovx", "What drove costs?")
    assert r.exit_code == 0, r.output
    assert fake.index.searched == [("What drove costs?", 5)]
    out = r.stdout
    assert "1. my_pdfs/annual.pdf, page 4  score 12.35" in out
    assert "Operating costs rose" in out
    assert "2. my_pdfs/q3.pdf, page 2  score 9.50" in out
    assert "3. 7  score 3.25" in out  # built from vectors: the id, no page
    assert "[counted]" in out
    assert out.rstrip().splitlines()[-1].startswith("Next: optivision find")
    assert fake.index.closed
    _lint(out)


def test_find_k_is_forwarded(fake, hits):
    fake.index = FakeIndex(hits=hits)
    invoke("find", "my_pdfs.ovx", "q", "-k", "2")
    assert fake.index.searched == [("q", 2)]
    assert invoke("find", "my_pdfs.ovx", "q", "-k", "0").exit_code == 2


def test_find_json(fake, hits):
    fake.index = FakeIndex(hits=hits)
    r = invoke("find", "my_pdfs.ovx", "q", "--json")
    data = json.loads(r.stdout)
    assert data["query"] == "q" and data["k"] == 5 and len(data["hits"]) == 3
    assert data["hits"][0] == {"id": "my_pdfs/annual.pdf::p4", "file": "my_pdfs/annual.pdf", "page": 4,
                               "score": 12.3456, "text": "Operating costs rose"}
    assert data["hits"][2]["file"] is None and data["hits"][2]["page"] is None


def test_find_without_results(fake):
    r = invoke("find", "my_pdfs.ovx", "q")
    assert r.exit_code == 0 and "none for this question" in r.stdout
    assert r.stdout.rstrip().splitlines()[-1].startswith("Next:")


def test_find_a_missing_index_is_an_input_error_and_the_ovx_suffix_is_optional(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = invoke("find", "nothing_here", "q")
    assert r.exit_code == 2 and "no index at nothing_here" in r.stderr
    _lint(r.stderr)
    (tmp_path / "docs.ovx").mkdir()
    r = invoke("find", "docs", "q")
    assert r.exit_code == 0, r.output
    assert fake.load_calls == ["docs.ovx"]


def test_find_an_empty_question_is_an_input_error(fake):
    assert invoke("find", "my_pdfs.ovx", "  ").exit_code == 2


# --------------------------------------------------------------------------------- non-ASCII on a cp1252 console


def _cp1252_stream():
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)


def test_a_file_name_a_cp1252_console_cannot_show_does_not_crash_find(fake, monkeypatch):
    name = "my_pdfs/\u5831\u544a\u66f8 \u2713 caf\u00e9.pdf"
    with pytest.raises(UnicodeEncodeError):  # the situation being guarded against
        name.encode("cp1252")
    fake.index = FakeIndex(hits=[Hit(id=name + "::p1", file=name, page=1, score=1.0, text="\u2713 ok")])
    raw, out = _cp1252_stream()
    _, err = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    cli_easy.find_command(index="my_pdfs.ovx", query="q", k=5, as_json=False)
    out.flush()
    text = raw.getvalue().decode("cp1252")
    assert "caf\u00e9.pdf" in text and "?" in text  # the unencodable characters became ?
    assert "Next:" in text


def test_the_whole_cli_survives_a_strict_cp1252_stdout_in_a_real_process(tmp_path):
    script = (
        "import sys, types\n"
        "import optivision.easy as ov\n"
        "from optivision.easy import Hit\n"
        "name = 'docs/\\u5831\\u544a\\u66f8.pdf'\n"
        "class Idx:\n"
        "    def search(self, q, k=5): return [Hit(id=name + '::p1', file=name, page=1, score=2.0, text='\\u2713')]\n"
        "    def close(self): pass\n"
        "ov.load = lambda path, **kw: Idx()\n"
        "from optivision.cli import app\n"
        "app(['find', sys.argv[1], 'q'])\n"
    )
    (tmp_path / "docs.ovx").mkdir()
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "cp1252", "TERM": "dumb"}
    done = subprocess.run([sys.executable, "-c", script, str(tmp_path / "docs.ovx")], capture_output=True,
                          env=env, timeout=120, check=False)
    assert done.returncode == 0, done.stderr.decode("cp1252", "replace")
    assert b"docs/" in done.stdout and b"Next:" in done.stdout


# --------------------------------------------------------------------------------- check


def test_check_prints_the_report_and_exits_0_when_ok(fake):
    fake.index = FakeIndex(report=FakeReport(ok=True))
    r = invoke("check", "my_pdfs.ovx")
    assert r.exit_code == 0, r.output
    assert "Record     matches" in r.stdout
    assert fake.index.checked == [{}]
    assert fake.index.closed


def test_check_exits_1_when_the_check_does_not_pass(fake):
    fake.index = FakeIndex(report=FakeReport(ok=False, lines=("Drift      found",)))
    r = invoke("check", "my_pdfs.ovx")
    assert r.exit_code == 1 and "Drift      found" in r.stdout


def test_check_forwards_queries_and_answers(fake, tmp_path):
    fake.index = FakeIndex(report=FakeReport())
    q = tmp_path / "q.json"
    q.write_text(json.dumps(["one", "two", "three"]), encoding="utf-8")
    a = tmp_path / "a.json"
    a.write_text(json.dumps(["x.pdf::p1", ["y.pdf", 3], ["z.pdf::p1", ["w.pdf", 2]]]), encoding="utf-8")
    r = invoke("check", "my_pdfs.ovx", "--queries", str(q), "--answers", str(a))
    assert r.exit_code == 0, r.output
    assert fake.index.checked == [{"queries": ["one", "two", "three"],
                                   "answers": ["x.pdf::p1", ("y.pdf", 3), {"z.pdf::p1", ("w.pdf", 2)}]}]


def test_check_apply_runs_the_stepdown_only_when_the_check_fails(fake):
    report = FakeReport(ok=False)
    fake.index = FakeIndex(report=report)
    r = invoke("check", "my_pdfs.ovx", "--apply")
    assert r.exit_code == 0, r.output
    assert report.applied == [True] and "re-encoding takes about 5 min" in r.stdout
    ok = FakeReport(ok=True)
    fake.index = FakeIndex(report=ok)
    r = invoke("check", "my_pdfs.ovx", "--apply")
    assert r.exit_code == 0 and ok.applied == [] and "Nothing to apply" in r.stdout


def test_check_json(fake):
    fake.index = FakeIndex(report=FakeReport(ok=False))
    r = invoke("check", "my_pdfs.ovx", "--json")
    assert r.exit_code == 1
    assert json.loads(r.stdout) == {"ok": False, "lines": ["Record     matches", "Drift      none"]}


def test_check_a_missing_index_is_an_input_error(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert invoke("check", "nothing_here").exit_code == 2


# --------------------------------------------------------------------------------- doctor


def test_doctor_exit_codes(fake):
    fake.doctor = FakeDoctor(ok=True)
    r = invoke("doctor")
    assert r.exit_code == 0 and "Python     3.11" in r.stdout
    fake.doctor = FakeDoctor(ok=False)
    r = invoke("doctor")
    assert r.exit_code == 1 and "Torch      broken" in r.stdout


def test_doctor_passes_the_model_and_rejects_an_unknown_one(fake):
    invoke("doctor", "--model", MODEL)
    assert fake.doctor_calls == [MODEL]
    r = invoke("doctor", "--model", "nope")
    assert r.exit_code == 2 and "error:" in r.stderr


def test_doctor_json(fake):
    fake.doctor = FakeDoctor(ok=False)
    r = invoke("doctor", "--json")
    assert r.exit_code == 1 and json.loads(r.stdout) == {"ok": False}


def test_doctor_with_a_report_that_has_no_ok_is_fine(fake):
    fake.doctor = "Python 3.11"  # a plain string report
    r = invoke("doctor")
    assert r.exit_code == 0 and "Python 3.11" in r.stdout


# --------------------------------------------------------------------------------- quickstart


@pytest.fixture
def counted(monkeypatch):
    monkeypatch.setattr(cli_easy, "_count_pages", lambda folder: (
        types.SimpleNamespace(n_pages=412, n_files=37, skipped=[("scan.pdf", "encrypted")]), ""))


def test_quickstart_without_a_terminal_prints_the_plan_and_the_build_call(fake, folder, counted):
    r = invoke("quickstart", str(folder), "--model", MODEL)
    assert r.exit_code == 0, r.output
    assert fake.build_calls == []  # nothing is built
    out = r.stdout
    for step in range(1, 7):
        assert f"[{step}/6]" in out
    assert "412 in 37 files" in out and "[counted]" in out
    assert "skipped scan.pdf: encrypted" in out
    assert "Nothing was written: no one can answer questions here" in out
    last = out.rstrip().splitlines()[-1]
    assert last == f"Next: optivision build {folder} --model {MODEL} --target 0.97"
    _lint(out)


def test_quickstart_without_a_terminal_and_without_a_model_names_the_placeholder(fake, folder, counted):
    r = invoke("quickstart", str(folder))
    assert r.exit_code == 0, r.output
    assert r.stdout.rstrip().splitlines()[-1].endswith("--model MODEL_ID --target 0.97")
    assert fake.options_calls[0]["model"] is None  # the Start here table


def test_quickstart_uses_the_start_here_pick_for_the_printed_call(fake, folder, counted, monkeypatch):
    fake.options_text = "Start here\nNext: ov.options(...)"
    original = optivision.easy.options

    def with_start_here(model=None, target=None, *, pages=None, accuracy=None):
        plan = original(model, target, pages=pages)
        if model is None:
            plan.data["start_here"] = [{"model": "vendor/small-model"}, {"model": "vendor/other"}]
        return plan

    monkeypatch.setitem(vars(optivision.easy), "options", with_start_here)
    r = invoke("quickstart", str(folder))
    assert "--model vendor/small-model --target 0.97" in r.stdout.splitlines()[-1]


def test_quickstart_shows_the_float16_cost_with_its_tag(fake, folder, counted, monkeypatch):
    f16 = LayoutRow(name="float16", plain="float16", spec={}, chain="float", bytes_per_page=(187513.0, 191567.0),
                    ratio_f16=1.0, ratio_f32=2.0, lowest_kept=1.0, worst_point=1.0)
    original = optivision.easy.options

    def with_rows(model=None, target=None, *, pages=None, accuracy=None):
        plan = original(model, target, pages=pages)
        plan.selection = types.SimpleNamespace(rows=(f16,))
        return plan

    monkeypatch.setitem(vars(optivision.easy), "options", with_rows)
    r = invoke("quickstart", str(folder), "--model", MODEL)
    assert "keeping the temporary copy takes 78.9 MB on disk  [estimate: our bytes per page x your pages]" in r.stdout


def test_quickstart_yes_builds_without_questions(fake, folder, counted):
    r = invoke("quickstart", str(folder), "--yes", "--model", MODEL, "--target", "0.95")
    assert r.exit_code == 0, r.output
    assert fake.build_calls == [{"source": str(folder), "model": MODEL, "target": 0.95}]
    lines = r.stdout.rstrip().splitlines()
    assert lines[-1] == 'Next: optivision find my_pdfs.ovx "your question"'
    assert sum(line.startswith("Next:") for line in lines) == 1  # the plan's own Next lines are dropped
    assert "[5/6]" in r.stdout and "[6/6]" in r.stdout and "Size       10.6 MB" in r.stdout


def test_quickstart_yes_without_a_target_leaves_it_to_the_library(fake, folder, counted):
    r = invoke("quickstart", str(folder), "--yes", "--model", MODEL)
    assert r.exit_code == 0, r.output
    assert fake.build_calls == [{"source": str(folder), "model": MODEL, "target": None}]


def test_quickstart_yes_needs_a_folder_and_a_model(fake, folder):
    for args in (["--yes"], ["--yes", str(folder)], ["--yes", "--model", MODEL]):
        r = invoke("quickstart", *args)
        assert r.exit_code == 2 and "--yes asks nothing" in r.stderr
    assert fake.build_calls == []


def test_quickstart_yes_stops_when_the_machine_check_fails(fake, folder, counted):
    fake.doctor = FakeDoctor(ok=False)
    r = invoke("quickstart", str(folder), "--yes", "--model", MODEL)
    assert r.exit_code == 1 and "doctor found problems" in r.stderr
    assert fake.build_calls == []


def test_quickstart_a_missing_folder_is_an_input_error(fake):
    r = invoke("quickstart", "no_such_folder")
    assert r.exit_code == 2 and "does not exist" in r.stderr


def test_quickstart_interactive_asks_then_builds(fake, folder, counted, monkeypatch):
    monkeypatch.setattr(cli_easy, "_is_interactive", lambda: True)
    # model id, target, keep float16?, build now?
    r = invoke("quickstart", str(folder), input=f"{MODEL}\n95%\ny\ny\n")
    assert r.exit_code == 0, r.output
    assert fake.build_calls == [{"source": str(folder), "model": MODEL, "target": 0.95, "keep_float16": True}]
    out = r.stdout
    assert out.index("[1/6]") < out.index("[2/6]") < out.index("[3/6]") < out.index("[4/6]") < out.index("[5/6]")
    assert "[6/6]" in out and "Model id" in out and "Target" in out and "Build now?" in out
    assert out.rstrip().splitlines()[-1] == 'Next: optivision find my_pdfs.ovx "your question"'
    # the target and the model were checked while asking
    assert {"model": MODEL, "target": 0.95, "pages": str(folder)} in fake.options_calls
    _lint(out)


def test_quickstart_interactive_asks_again_after_a_bad_model_or_target(fake, folder, counted, monkeypatch):
    monkeypatch.setattr(cli_easy, "_is_interactive", lambda: True)
    r = invoke("quickstart", str(folder), input=f"nope\n{MODEL}\nabc\n97\nn\ny\n")
    assert r.exit_code == 0, r.output
    assert "nope is not a model we know" in r.stderr
    assert "cannot read 'abc' as a target" in r.stderr
    assert fake.build_calls == [{"source": str(folder), "model": MODEL, "target": 0.97}]


def test_quickstart_interactive_can_decline_the_build(fake, folder, counted, monkeypatch):
    monkeypatch.setattr(cli_easy, "_is_interactive", lambda: True)
    r = invoke("quickstart", str(folder), "--model", MODEL, "--target", "0.97", input="n\nn\n")
    assert r.exit_code == 0 and fake.build_calls == []
    assert r.stdout.rstrip().splitlines()[-1] == f"Next: optivision build {folder} --model {MODEL} --target 0.97"


def test_quickstart_interactive_asks_for_the_folder(fake, folder, counted, monkeypatch):
    monkeypatch.setattr(cli_easy, "_is_interactive", lambda: True)
    r = invoke("quickstart", "--model", MODEL, "--target", "0.97", input=f"{folder}\nn\ny\n")
    assert r.exit_code == 0, r.output
    assert fake.build_calls[0]["source"] == str(folder)


def test_quickstart_when_the_page_counter_is_unavailable_the_plan_still_prints(fake, folder, monkeypatch):
    monkeypatch.setattr(cli_easy, "_count_pages", lambda f: (None, "No module named 'pypdfium2'"))
    r = invoke("quickstart", str(folder), "--model", MODEL)
    assert r.exit_code == 0 and "No module named 'pypdfium2'" in r.stdout


# --------------------------------------------------------------------------------- helpers


def test_hold_next_swallows_only_a_next_line_at_the_start_of_a_line():
    sink = io.StringIO()
    hold = cli_easy._HoldNext(sink)
    hold.write("Size 10 MB\n")
    hold.write("Next:")
    hold.write(" index.search(...)")
    hold.write("\n")
    hold.write("after the summary, Next: stays\n")
    hold.write("progress 1/3\rprogress 2/3\r")
    assert sink.getvalue() == "Size 10 MB\nafter the summary, Next: stays\nprogress 1/3\rprogress 2/3\r"
    assert hold.swallowed == ["Next: index.search(...)\n"]


def test_hold_next_passes_through_the_attributes_of_the_stream():
    hold = cli_easy._HoldNext(io.StringIO())
    assert hold.isatty() is False


@pytest.mark.parametrize("text, expected", [("97", 0.97), ("97%", 0.97), ("0.97", 0.97), (" 95 % ", 0.95),
                                            ("1", 1.0), ("100", 1.0)])
def test_parse_target(text, expected):
    assert cli_easy._parse_target(text) == pytest.approx(expected)


@pytest.mark.parametrize("text, expected", [("50GB", 50 * 10**9), ("1.5 gb", 1_500_000_000), ("800MB", 800 * 10**6),
                                            ("12345", 12345), ("2GiB", 2 * 2**30), ("1k", 1000)])
def test_parse_size(text, expected):
    assert cli_easy._parse_size(text) == expected


def test_quote_adds_double_quotes_only_when_needed():
    assert cli_easy._quote("my_pdfs/") == "my_pdfs/"
    assert cli_easy._quote("my docs/") == '"my docs/"'
    assert cli_easy._quote("") == '""'


def test_jsonable_handles_the_library_types():
    layout = FakeIndex().layout
    out = cli_easy._jsonable({"layout": layout, "ids": {"b", "a"}, "x": float("nan"), "p": Path("a/b"),
                              "pair": (1, 2)})
    assert out["layout"]["name"] == "hm(0.25) > int8" and out["ids"] == ["a", "b"] and out["x"] is None
    assert out["p"] == str(Path("a/b")) and out["pair"] == [1, 2]
    json.dumps(out)


def test_json_output_is_ascii_so_any_console_can_show_it(fake):
    fake.index = FakeIndex(hits=[Hit(id="d/\u5831.pdf::p1", file="d/\u5831.pdf", page=1, score=1.0)])
    r = invoke("find", "my_pdfs.ovx", "q", "--json")
    assert r.stdout.isascii()
    assert json.loads(r.stdout)["hits"][0]["file"] == "d/\u5831.pdf"


def test_the_library_errors_the_cli_catches_are_easy_errors():
    # _refusals relies on this hierarchy (docs/easy/CONTRACT.md: _types.py)
    for name in ("UnknownModel", "AmbiguousModel", "WiringError", "RemoteCodeRefused", "TargetMismatch",
                 "MissingExtra", "NotEnoughDisk", "FeatureOff"):
        assert issubclass(getattr(optivision.easy, name), EasyError)
