"""The documented examples run, so that they cannot rot.

Every script under ``docs/package/guide/examples/`` and every ``python`` block of the pages in
``docs/package/guide/`` is executed in a subprocess, with the interpreter that runs the tests,
from an empty working directory, with this checkout's ``src`` first on ``PYTHONPATH``. The blocks
of one page run in order in one session, so a page may build on its earlier blocks.

They use synthetic data from ``optivision.testing`` or stand-ins made in place. A pass shows that
the documented code works. It says nothing about any encoder or about retrieval quality.

The files of the guide are also checked for ASCII, for the wording rules of the package, and for
relative links that point nowhere.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GUIDE = ROOT / "docs" / "package" / "guide"
EXAMPLES = GUIDE / "examples"

#: The default candidate list has Ward-merge families, which need SciPy.
NEEDS_SCIPY = {"certify_synthetic.py", "QUICKSTART.md", "BASIS_DEPENDENCE.md"}
TIMEOUT = 900

BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
BANNED_PHRASES = ("universal compression", "works with every model", "guaranteed", "state of the art",
                  "state-of-the-art", "novel", "better than", "lossless", "faster", "scales to")
BANNED_WORDS = re.compile(r"\b(simply|just)\b")


def _run(script: Path, workdir: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, str(script)], cwd=workdir, env=env,
                          capture_output=True, text=True, timeout=TIMEOUT, check=False)


def _pages() -> list[Path]:
    return sorted(p for p in GUIDE.glob("*.md") if BLOCK.search(p.read_text(encoding="utf-8")))


def _guide_files() -> list[Path]:
    return sorted([*GUIDE.glob("*.md"), *EXAMPLES.glob("*.py")])


def _empty_dir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def test_the_guide_has_the_documented_files():
    expected = {"README.md", "CONCEPTS.md", "SELECTION_AND_CERTIFICATION.md", "BASIS_DEPENDENCE.md",
                "LIMITATIONS.md", "ARCHITECTURE.md", "QUICKSTART.md", "BACKENDS_GUIDE.md",
                "ADAPTERS_GUIDE.md"}
    assert expected <= {p.name for p in GUIDE.glob("*.md")}
    assert len(list(EXAMPLES.glob("*.py"))) >= 3


@pytest.mark.parametrize("script", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_example_script_runs(script, tmp_path):
    if script.name in NEEDS_SCIPY:
        pytest.importorskip("scipy")
    text = script.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 60, "an example stays short"
    assert "synthetic" in text.lower() or "stand-in" in text.lower(), (
        "an example says that its data only exercise the code")
    cwd = _empty_dir(tmp_path, "cwd")
    proc = _run(script, cwd)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip(), "an example prints a short result"
    assert not list(cwd.iterdir()), "an example writes only to a temporary directory"


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.name)
def test_python_blocks_of_a_page_run_in_order(page, tmp_path):
    if page.name in NEEDS_SCIPY:
        pytest.importorskip("scipy")
    blocks = BLOCK.findall(page.read_text(encoding="utf-8"))
    assert blocks
    script = _empty_dir(tmp_path, "script") / "blocks.py"
    script.write_text("\n\n".join(blocks), encoding="utf-8")
    cwd = _empty_dir(tmp_path, "cwd")
    proc = _run(script, cwd)
    assert proc.returncode == 0, f"{page.name}\n{proc.stdout}\n{proc.stderr}"
    assert not list(cwd.iterdir()), "a snippet writes only to a temporary directory"


def test_guide_files_are_ascii():
    for path in _guide_files():
        data = path.read_bytes()
        try:
            data.decode("ascii")
        except UnicodeDecodeError as exc:
            line = data[: exc.start].count(b"\n") + 1
            pytest.fail(f"{path.name}: non-ASCII byte at line {line}")


def test_wording_rules_hold_in_the_guide():
    for path in _guide_files():
        text = path.read_text(encoding="utf-8").lower()
        for phrase in BANNED_PHRASES:
            assert phrase not in text, f"{path.name}: {phrase!r}"
        found = BANNED_WORDS.search(text)
        assert found is None, f"{path.name}: {found.group(0)!r}"


def test_relative_links_in_the_guide_resolve():
    for path in sorted(GUIDE.glob("*.md")):
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            file = target.split("#", 1)[0]
            assert (path.parent / file).exists(), f"{path.name}: {target} does not exist"
