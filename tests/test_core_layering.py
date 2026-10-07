"""Dependency direction: ``optivision.core`` must not import outward, and the package
must not import research scripts.

This is a source-level check (``ast``). It does not see what ``optivision/__init__``
loads when any submodule is imported; see docs/package/UNIVERSAL_PACKAGE_DESIGN.md.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "optivision"
CORE = PACKAGE / "core"
RESEARCH = Path(__file__).resolve().parents[1] / "scripts" / "research"

#: Modules outside ``optivision.core`` that core code may import. They are the pinned,
#: array-only modules of the library; core is a set of facades and new code over them.
ALLOWED_OPTIVISION = {
    "optivision.representation",
    "optivision.compose",
    "optivision.scoring",
    "optivision.evaluation",
    "optivision.calibration",
    "optivision.stages",
    "optivision.stages.base",
    "optivision.stages.merge",
    "optivision.stages.project",
    "optivision.stages.quantize",
    "optivision.storage",
}
#: Names core may take from modules that also hold page- or legacy-specific code.
RESTRICTED_NAMES = {
    "optivision.storage": {"ExactIndex", "TieredIndex"},
}
FORBIDDEN_NAMES = {
    "optivision.stages": {"SpatialPruner"},
}
ALWAYS_ALLOWED_THIRD_PARTY = {"numpy"}
LAZY_ONLY_THIRD_PARTY = {"scipy", "torch"}


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(path: Path) -> str:
    name = _module_name(path)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def _imports(path: Path):
    """Yield ``(module, names, lazy, lineno)`` for every import statement of a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _package_of(path)

    def visit(node: ast.AST, lazy: bool):
        for child in ast.iter_child_nodes(node):
            inner = lazy or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                for alias in child.names:
                    yield alias.name, (), lazy, child.lineno
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    base = package.split(".")
                    base = base[: len(base) - (child.level - 1)]
                    module = ".".join([*base, child.module] if child.module else base)
                else:
                    module = child.module or ""
                yield module, tuple(a.name for a in child.names), lazy, child.lineno
            yield from visit(child, inner)

    yield from visit(tree, False)


def _core_files() -> list[Path]:
    return sorted(CORE.rglob("*.py"))


def _is_module(name: str) -> bool:
    path = SRC.joinpath(*name.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


@pytest.mark.skipif(not CORE.is_dir(), reason="the repository layout (src/) is absent")
def test_core_exists():
    assert (CORE / "__init__.py").is_file()
    assert _core_files()


def test_core_imports_only_inward():
    problems = []
    for path in _core_files():
        for module, names, lazy, line in _imports(path):
            where = f"{path.relative_to(SRC)}:{line}"
            top = module.split(".")[0]
            if top in sys.stdlib_module_names or top == "__future__":
                continue
            if top in ALWAYS_ALLOWED_THIRD_PARTY:
                continue
            if top in LAZY_ONLY_THIRD_PARTY:
                if not lazy:
                    problems.append(f"{where}: {module} must be imported inside a function")
                continue
            if top != "optivision":
                problems.append(f"{where}: third-party import {module}")
                continue
            # `from optivision import x` / `from . import x`: resolve each name to a module if it is one
            targets = [(module, names)]
            if module in {"optivision", "optivision.core"} or (names and all(
                    _is_module(f"{module}.{n}") for n in names)):
                targets = [(f"{module}.{n}", ()) for n in names] if names else targets
            for target, target_names in targets:
                if target == "optivision.core" or target.startswith("optivision.core."):
                    continue
                if target not in ALLOWED_OPTIVISION:
                    problems.append(f"{where}: core imports {target}")
                    continue
                allowed = RESTRICTED_NAMES.get(target)
                if allowed is not None and not set(target_names) <= allowed:
                    problems.append(f"{where}: core takes {sorted(set(target_names) - allowed)} from {target}")
                banned = FORBIDDEN_NAMES.get(target, set()) & set(target_names)
                if banned:
                    problems.append(f"{where}: core takes {sorted(banned)} from {target}")
    assert not problems, "\n".join(problems)


def test_package_does_not_import_research_scripts():
    stems = {p.stem for p in RESEARCH.glob("*.py")} if RESEARCH.is_dir() else set()
    stems |= {"x_common", "e1_stats", "e1_common", "e1_1_study", "x1_agreement", "x2_budget", "x3_symmetry"}
    problems = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for module, _names, _lazy, line in _imports(path):
            top = module.split(".")[0]
            if top in stems or top == "scripts":
                problems.append(f"{path.relative_to(SRC)}:{line}: imports research code {module}")
    assert not problems, "\n".join(problems)
