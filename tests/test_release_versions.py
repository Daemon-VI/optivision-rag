"""The npm launcher installs the PyPI release with its own version number."""

import json
from pathlib import Path

from optivision import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_npm_version_matches_python():
    pkg = json.loads((ROOT / "npm" / "package.json").read_text(encoding="utf-8"))
    assert pkg["version"] == __version__
