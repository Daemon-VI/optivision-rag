"""Identity of the code a decision ran through.

A certificate records which source files produced its p-values and its choice, so that a
reader can tell whether the code that checks it now is the code that issued it. The
identity is SHA-256 of every decision-relevant source file, a digest over that table, and the
versions of Python, NumPy and SciPy. It identifies source files only, not the numerical
libraries or the machine beneath them; that is why replay compares p-values at a relative
tolerance.

The file set (paths relative to the ``optivision`` package directory):

* every ``.py`` file under ``core/`` (the model-agnostic layer, this module included);
* the pinned modules the decision path runs through: ``calibration.py``, ``compose.py``,
  ``scoring.py``, ``evaluation.py``, ``representation.py``;
* every ``.py`` file directly under ``stages/`` and under ``compression/``.

Everything else (page encoders, the CLI, backends, adapters, storage) does not take part in
a selection or in the deployed index and is not hashed.

Files are located from this module's own ``__file__``, so the same code works in a source
tree and in an installed wheel. Line endings are normalised to LF before hashing: a Windows
checkout with CRLF and a wheel built from the same sources agree. Nothing else is normalised.
If the sources cannot be read (no ``.py`` files in the installation, a missing required
module) the identity has ``digest`` ``None`` and a ``note`` saying why; it never raises.

The identity of the running package is computed once per process. An explicit ``root`` is
never cached.
"""

from __future__ import annotations

import copy
import hashlib
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from optivision.core.identity import digest

__all__ = ["code_identity", "package_info", "source_files"]

#: Single modules (relative to the package directory) that the decision path runs through.
_MODULES = ("calibration.py", "compose.py", "scoring.py", "evaluation.py", "representation.py")
#: ``(directory, recursive)``: every ``.py`` file of these directories is hashed.
_TREES = (("core", True), ("stages", False), ("compression", False))

_CACHE: dict[str, dict[str, Any]] = {}


def _package_root() -> Path:
    """The ``optivision`` package directory: this file is ``optivision/core/artifact/codeid.py``."""
    return Path(__file__).resolve().parents[2]


def source_files(root: str | Path | None = None) -> dict[str, Path]:
    """``{relative posix path: path}`` of the hashed files, sorted by path.

    Raises ``FileNotFoundError`` when a required module or a whole directory is missing.
    """
    base = _package_root() if root is None else Path(root)
    found: dict[str, Path] = {}
    missing: list[str] = []
    for name in _MODULES:
        path = base / name
        if path.is_file():
            found[name] = path
        else:
            missing.append(name)
    for directory, recursive in _TREES:
        folder = base / directory
        pattern = "**/*.py" if recursive else "*.py"
        paths = sorted(folder.glob(pattern)) if folder.is_dir() else []
        paths = [p for p in paths if "__pycache__" not in p.parts]
        if not paths:
            missing.append(f"{directory}/{pattern}")
        for path in paths:
            found[path.relative_to(base).as_posix()] = path
    if missing:
        raise FileNotFoundError("no source files for: " + ", ".join(missing))
    return dict(sorted(found.items()))


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _installed_version(name: str) -> str | None:
    module = sys.modules.get(name)
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    try:
        return metadata.version(name)
    except (metadata.PackageNotFoundError, OSError, ValueError):
        return None  # not installed, or unreadable metadata: the identity must not raise


def _compute(base: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "digest": None,
        "files": {},
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "scipy": _installed_version("scipy"),
    }
    try:
        files = {rel: _file_digest(path) for rel, path in source_files(base).items()}
    except OSError as exc:  # FileNotFoundError is an OSError
        out["note"] = f"source files could not be read: {exc}"
        return out
    out["files"] = files
    out["digest"] = digest(files)
    return out


def code_identity(root: str | Path | None = None) -> dict[str, Any]:
    """The identity of the decision-relevant code, as a fresh JSON-native dict.

    ``{"digest", "files", "python", "numpy", "scipy"}``: ``files`` maps each relative path
    to the SHA-256 of its content with CRLF turned into LF; ``digest`` is the SHA-256 of the
    canonical JSON of that table, or ``None`` (with a ``note``) when the sources cannot be
    read. ``scipy`` is ``None`` when it is not installed.

    ``root`` is the directory that holds ``core/``, ``stages/`` and the pinned modules; it
    exists for tests and for checking a copy of the tree. By default the running package is
    described and the result is cached for the process.

    The identity is an integrity check on which source files ran. It is not a signature:
    anyone who can edit a certificate can write any identity into it.
    """
    if root is not None:
        return _compute(Path(root))
    cached = _CACHE.get("package")
    if cached is None:
        cached = _CACHE["package"] = _compute(_package_root())
    return copy.deepcopy(cached)


def package_info() -> dict[str, str]:
    """``{"name", "version"}`` of the importable package this module belongs to.

    ``name`` is the import name (``optivision``), ``version`` its ``__version__``; the
    package object is already loaded when any of its submodules runs.
    """
    name = __name__.partition(".")[0]
    version = getattr(sys.modules.get(name), "__version__", None)
    return {"name": name, "version": "unknown" if version is None else str(version)}
