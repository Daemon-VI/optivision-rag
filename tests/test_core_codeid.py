"""``code_identity``: SHA-256 of the decision-relevant source files.

The identity must be stable, change exactly when a hashed file's content changes, ignore line
endings (a Windows checkout with CRLF and a wheel built from the same sources agree) and
files outside the set, and report an unreadable installation as ``digest: None`` with a note
instead of raising. Tests that change files work on a temporary copy of the tree, through the
``root`` parameter; the real source files are only read.
"""

from __future__ import annotations

import platform
import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

import optivision
from optivision.core.artifact import code_identity, codeid, package_info
from optivision.core.identity import digest

HEX64 = re.compile(r"^[0-9a-f]{64}$")
PACKAGE_DIR = Path(optivision.__file__).resolve().parent


@pytest.fixture
def tree(tmp_path):
    """A copy of exactly the hashed files of the real package, plus files that are not hashed."""
    root = tmp_path / "optivision"
    for rel, path in codeid.source_files().items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (root / "cli.py").write_text("# not part of the decision path\n", encoding="utf-8")
    return root


@pytest.fixture
def mini(tmp_path):
    """A tiny tree with the same layout (one file per hashed kind): fast to build and change."""
    root = tmp_path / "mini"
    for rel in ("calibration.py", "compose.py", "scoring.py", "evaluation.py",
                "representation.py", "core/__init__.py", "core/identity.py",
                "core/selection/rules.py", "core/selection/engine.py", "stages/base.py",
                "compression/binary.py"):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"# {rel}\nx = 1\n".encode())
    (root / "cli.py").write_text("# not part of the decision path\n", encoding="utf-8")
    return root


class TestTheRealIdentity:
    def test_it_is_stable_across_calls_and_each_call_is_a_fresh_copy(self):
        a = code_identity()
        b = code_identity()
        assert a == b
        a["files"].clear()
        a["digest"] = "tampered"
        assert code_identity() == b

    def test_its_fields_and_the_digest_over_the_table(self):
        ident = code_identity()
        assert set(ident) == {"digest", "files", "python", "numpy", "scipy"}
        assert HEX64.match(ident["digest"])
        assert ident["digest"] == digest(ident["files"])
        assert all(HEX64.match(v) for v in ident["files"].values())
        assert list(ident["files"]) == sorted(ident["files"])
        assert "\\" not in "".join(ident["files"])  # posix separators on every platform

    def test_the_file_set_is_the_decision_path_and_nothing_else(self):
        files = set(code_identity()["files"])
        for must in ("calibration.py", "compose.py", "scoring.py", "evaluation.py",
                     "representation.py", "core/identity.py", "core/selection/rules.py",
                     "core/artifact/codeid.py", "core/artifact/certificate.py",
                     "stages/base.py", "stages/quantize.py", "compression/binary.py"):
            assert must in files, must
        for never in ("cli.py", "storage.py", "api.py", "pipeline.py", "config.py"):
            assert never not in files, never
        assert not any(f.startswith(("adapters/", "backends/", "ext/", "legacy/", "encoders/"))
                       for f in files)
        assert not any("__pycache__" in f or f.endswith(".pyc") for f in files)

    def test_the_versions(self):
        ident = code_identity()
        assert ident["python"] == platform.python_version()
        assert ident["numpy"] == np.__version__
        try:
            installed = metadata.version("scipy")
        except metadata.PackageNotFoundError:
            installed = None
        assert ident["scipy"] == installed

    def test_the_readable_package_has_no_note(self):
        assert "note" not in code_identity()

    def test_the_package_info_names_the_importable_package(self):
        assert package_info() == {"name": "optivision", "version": optivision.__version__}

    def test_the_default_identity_is_computed_once_per_process(self, monkeypatch):
        calls = []
        real = codeid._compute
        monkeypatch.setattr(codeid, "_CACHE", {})
        monkeypatch.setattr(codeid, "_compute", lambda base: calls.append(base) or real(base))
        assert code_identity() == code_identity()
        assert len(calls) == 1

    def test_an_explicit_root_is_never_cached(self, mini, monkeypatch):
        calls = []
        real = codeid._compute
        monkeypatch.setattr(codeid, "_compute", lambda base: calls.append(base) or real(base))
        code_identity(mini)
        code_identity(mini)
        assert len(calls) == 2


class TestOnACopyOfTheTree:
    def test_an_identical_copy_has_the_identity_of_the_original(self, tree):
        copy = code_identity(root=tree)
        real = code_identity(root=PACKAGE_DIR)  # read fresh, not from the process cache
        assert copy["files"] == real["files"] and copy["digest"] == real["digest"]

    def test_the_root_may_be_a_string(self, mini):
        assert code_identity(root=str(mini))["digest"] == code_identity(root=mini)["digest"]

    def test_a_crlf_checkout_of_the_real_tree_agrees_with_the_lf_one(self, tree):
        lf = code_identity(root=PACKAGE_DIR)  # LF or CRLF as checked out, hashed as LF
        for path in tree.rglob("*.py"):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        assert code_identity(root=tree)["digest"] == lf["digest"]

    def test_a_changed_byte_changes_that_entry_and_the_digest(self, mini):
        before = code_identity(root=mini)
        target = mini / "core" / "identity.py"
        target.write_bytes(target.read_bytes() + b"# one more line\n")
        after = code_identity(root=mini)
        assert after["digest"] != before["digest"]
        changed = [k for k in before["files"] if before["files"][k] != after["files"][k]]
        assert changed == ["core/identity.py"]
        assert after["files"].keys() == before["files"].keys()

    @pytest.mark.parametrize("name", ["calibration.py", "compose.py", "scoring.py",
                                      "evaluation.py", "representation.py", "stages/base.py",
                                      "compression/binary.py", "core/selection/engine.py"])
    def test_every_kind_of_hashed_file_counts(self, mini, name):
        before = code_identity(root=mini)["digest"]
        path = mini / name
        path.write_bytes(path.read_bytes() + b"\n")
        assert code_identity(root=mini)["digest"] != before

    def test_crlf_and_lf_copies_hash_the_same(self, mini):
        lf = code_identity(root=mini)
        for path in mini.rglob("*.py"):
            data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            path.write_bytes(data)
        assert b"\r\n" in (mini / "core" / "identity.py").read_bytes()
        crlf = code_identity(root=mini)
        assert crlf["files"] == lf["files"] and crlf["digest"] == lf["digest"]

    def test_only_line_endings_are_normalised(self, mini):
        before = code_identity(root=mini)["digest"]
        path = mini / "scoring.py"
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n") + b"  \n")  # trailing blanks
        assert code_identity(root=mini)["digest"] != before

    def test_files_outside_the_set_do_not_matter(self, mini):
        before = code_identity(root=mini)
        (mini / "cli.py").write_text("# changed\n", encoding="utf-8")
        (mini / "storage.py").write_text("x = 1\n", encoding="utf-8")
        (mini / "adapters").mkdir()
        (mini / "adapters" / "extra.py").write_text("y = 2\n", encoding="utf-8")
        (mini / "core" / "__pycache__").mkdir()
        (mini / "core" / "__pycache__" / "stray.py").write_text("z = 3\n", encoding="utf-8")
        (mini / "core" / "notes.txt").write_text("not python\n", encoding="utf-8")
        assert code_identity(root=mini) == before

    def test_a_new_python_file_in_a_hashed_directory_changes_the_identity(self, mini):
        before = code_identity(root=mini)
        (mini / "core" / "extra.py").write_text("x = 1\n", encoding="utf-8")
        (mini / "stages" / "extra.py").write_text("x = 1\n", encoding="utf-8")
        after = code_identity(root=mini)
        assert after["digest"] != before["digest"]
        assert {"core/extra.py", "stages/extra.py"} <= set(after["files"])

    def test_nested_core_directories_are_included(self, mini):
        (mini / "core" / "deep" / "er").mkdir(parents=True)
        (mini / "core" / "deep" / "er" / "x.py").write_text("x = 1\n", encoding="utf-8")
        assert "core/deep/er/x.py" in code_identity(root=mini)["files"]

    def test_a_removed_file_changes_the_identity(self, mini):
        before = code_identity(root=mini)["digest"]
        (mini / "core" / "selection" / "rules.py").unlink()
        assert code_identity(root=mini)["digest"] != before


class TestUnreadableSources:
    @pytest.mark.parametrize("missing", ["calibration.py", "compose.py", "scoring.py",
                                         "evaluation.py", "representation.py"])
    def test_a_missing_required_module_gives_no_digest_and_a_note(self, mini, missing):
        (mini / missing).unlink()
        ident = code_identity(root=mini)
        assert ident["digest"] is None and ident["files"] == {}
        assert missing in ident["note"] and "could not be read" in ident["note"]
        assert ident["python"] == platform.python_version()  # the rest is still recorded

    @pytest.mark.parametrize("directory", ["core", "stages", "compression"])
    def test_a_missing_directory_gives_no_digest_and_a_note(self, mini, directory):
        shutil.rmtree(mini / directory)
        ident = code_identity(root=mini)
        assert ident["digest"] is None and directory in ident["note"]

    def test_a_directory_without_any_source_gives_no_digest_and_never_raises(self, tmp_path):
        ident = code_identity(root=tmp_path / "does-not-exist")
        assert ident["digest"] is None and ident["files"] == {} and ident["note"]

    def test_a_file_that_cannot_be_read_gives_no_digest(self, mini, monkeypatch):
        victim = mini / "scoring.py"
        real = type(victim).read_bytes

        def read_bytes(self):
            if self.name == "scoring.py":
                raise PermissionError("denied")
            return real(self)

        monkeypatch.setattr(type(victim), "read_bytes", read_bytes)
        ident = code_identity(root=mini)
        assert ident["digest"] is None and "denied" in ident["note"]


class TestLibraryVersions:
    def test_scipy_is_none_when_it_is_not_installed(self, monkeypatch, mini):
        monkeypatch.delitem(sys.modules, "scipy", raising=False)

        def missing(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(codeid.metadata, "version", missing)
        assert code_identity(root=mini)["scipy"] is None

    def test_scipy_is_read_from_the_loaded_module_first(self, monkeypatch, mini):
        class Fake:
            __version__ = "9.9.9"

        monkeypatch.setitem(sys.modules, "scipy", Fake)
        assert code_identity(root=mini)["scipy"] == "9.9.9"
