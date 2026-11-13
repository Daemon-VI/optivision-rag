"""Every extra that an easy-mode install hint names is defined in pyproject.toml.

The hints were written module by module; a user who follows ``pip install "optivision-rag[langchain]"`` for an
extra that does not exist gets a pip warning and still lacks the library."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_extra_the_code_names_is_defined():
    defined = set(tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"])
    named: set[str] = set()
    for path in (ROOT / "src" / "optivision").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for group in re.findall(r"optivision-rag\[([A-Za-z,-]+)\]", text):
            named |= set(group.split(","))
        named |= set(re.findall(r'missing_extra\([^)]*?"([a-z-]+)"\)', text))
    assert named, "the hints were not found"
    assert sorted(named - defined) == [], f"install hints name extras that pyproject.toml does not define: {sorted(named - defined)}"
