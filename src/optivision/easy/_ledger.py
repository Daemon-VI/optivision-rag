"""Status labels and test-gated switches, read from ``presets/easy_ledger.json``.

A feature whose preregistered test has not passed ships switched off and says why
(FINAL_DESIGN section 14). Outcomes are appended to the ledger only from recorded results
(``reports/easy/``); the status line a user sees comes from here, never from code.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from ._types import FeatureOff


@lru_cache(maxsize=1)
def ledger() -> dict[str, Any]:
    text = resources.files("optivision").joinpath("presets", "easy_ledger.json").read_text(encoding="utf-8")
    return json.loads(text)


def status() -> str:
    """The status sentence of every catalogue number (FINAL_DESIGN 7, rule 4)."""
    return str(ledger()["status"])


def feature(name: str) -> dict[str, Any]:
    features = ledger()["features"]
    if name not in features:
        raise KeyError(f"no feature {name!r} in easy_ledger.json; known: {sorted(features)}")
    return dict(features[name])


def is_on(name: str) -> bool:
    return bool(feature(name).get("on"))


def require(name: str) -> None:
    """Raise :class:`FeatureOff` with the ledger's reason unless the feature is on."""
    entry = feature(name)
    if not entry.get("on"):
        raise FeatureOff(f"{entry.get('title', name)} is switched off: {entry['reason']}")


def setting(name: str, key: str, default: Any = None) -> Any:
    """A decision-rule value of a feature (a margin, a mode), as the ledger records it."""
    return feature(name).get(key, default)


def practice_difficulty() -> str:
    """How practice questions compare with real ones: the wording V4's outcome records."""
    return str(setting("practice_check", "wording", "easier than real questions"))


def practice_role() -> str:
    """What the practice check may do, after "They": it moves a layout only in step-down mode."""
    if str(setting("practice_check", "mode", "step-down")) == "step-down":
        return "can only move a layout to a safer one and never confirm the target"
    return "only warn: they do not change the layout and never confirm the target"
