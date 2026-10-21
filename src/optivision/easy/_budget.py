"""``max_disk=``: the highest target whose layout fits a disk budget (FINAL_DESIGN 5.7 step 4).

``build(max_disk=...)`` and ``options(...)`` call :func:`highest_target_that_fits`. It walks the
targets from 1.00 down to the ledger's lowest target in steps of 0.01 (whole hundredths, so no
float drift) and returns the first whose pick, under the same rule ``_levels.select`` applies
everywhere else, fits. Nothing here chooses a layout by itself: the rule picks, the budget only
says which target to ask the rule for.

Sizes come from one row each:

* a catalogue row's ``bytes_per_page`` (the larger end of its per-dataset range, so the budget
  is not exceeded on the dataset with the bigger pages) ``[our benchmarks]``;
* for a row without catalogue bytes (a model without rows of its own), the exact code size of a
  codec-only layout on the user's own vectors, ``vectors_per_page * code_bytes(dim)``, when both
  are given. A merged layout without rows has no size we could state without an estimate, and an
  estimate may not set a layout (FINAL_DESIGN 7), so such a row is skipped.
"""

from __future__ import annotations

import math
import re
from typing import Any

from ._types import LayoutRow

_UNITS = {"": 1, "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
          "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}
_SIZE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$")


def parse_size(value: Any) -> int:
    """Bytes from an int, a float or a string such as ``"2 GB"``, ``"1.5GB"``, ``"800 MiB"``.

    Decimal units (GB = 10**9 bytes) as ``_claim.size`` prints them; binary units with ``iB``."""
    if isinstance(value, bool):
        raise TypeError("max_disk must be a number of bytes or a size such as '2 GB'")
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"max_disk must be a positive size, got {value!r}")
        return int(value)
    if isinstance(value, str):
        m = _SIZE.match(value)
        unit = m.group(2).lower() if m else None
        if m is None or unit not in _UNITS:
            raise ValueError(f"cannot read {value!r} as a size; write it like '2 GB' or '800 MB'")
        n = float(m.group(1)) * _UNITS[unit]
        if n <= 0:
            raise ValueError(f"max_disk must be a positive size, got {value!r}")
        return int(n)
    raise TypeError(f"max_disk must be a number of bytes or a size such as '2 GB', got {type(value).__name__}")


def _has_merge(spec: dict[str, Any]) -> bool:
    return any(s.get("stage") in ("hierarchical_merge", "adaptive_merge") for s in spec.get("stages", []))


def bytes_per_page(row: LayoutRow, *, vectors_per_page: float | None = None,
                   dim: int | None = None) -> tuple[float | None, str]:
    """``(bytes, basis)`` for one page in this layout; ``(None, reason)`` when it has no size.

    ``basis`` is ``"catalogue"`` (the row's larger per-dataset bytes) or ``"your vectors"``
    (exact code bytes of a codec-only layout on the given vectors per page and width)."""
    if row.bytes_per_page:
        return float(max(row.bytes_per_page)), "catalogue"
    if vectors_per_page is None or dim is None:
        return None, "no measured size for this layout and no vectors to count"
    if _has_merge(row.spec):
        return None, "a merged layout without measured rows has no size we can state"
    from optivision.compose import Pipeline

    code = Pipeline.from_dict(row.spec).quantizer.code_bytes(int(dim))
    return float(vectors_per_page) * float(code), "your vectors"


def disk_needed(row: LayoutRow, n_pages: int, *, vectors_per_page: float | None = None,
                dim: int | None = None, overhead_bytes_per_page: float = 0.0,
                overhead_bytes: float = 0.0) -> float | None:
    """Bytes the index would take in this layout: vectors plus the given overheads (text,
    record, reference), or None when the layout has no size (see :func:`bytes_per_page`)."""
    per_page, _ = bytes_per_page(row, vectors_per_page=vectors_per_page, dim=dim)
    if per_page is None:
        return None
    return float(n_pages) * (per_page + float(overhead_bytes_per_page)) + float(overhead_bytes)


def _targets(highest: float, lowest: float) -> list[float]:
    hi, lo = round(highest * 100), math.ceil(lowest * 100 - 1e-9)
    return [k / 100 for k in range(hi, lo - 1, -1)]


def highest_target_that_fits(model: str | None, n_pages: int, max_disk_bytes: Any, checked: bool, *,
                             modality: str | None = None, vectors_per_page: float | None = None,
                             dim: int | None = None, overhead_bytes_per_page: float = 0.0,
                             overhead_bytes: float = 0.0, lowest: float | None = None,
                             highest: float = 1.0) -> tuple[float | None, LayoutRow | None]:
    """The highest target, in steps of 0.01 from ``highest`` (1.00) down to ``lowest`` (the
    ledger's ``margins.lowest_target``), whose pick fits ``max_disk_bytes``; and that pick.

    ``(None, None)`` when no target down to the lowest fits: the caller says so and stops
    (``build``) or shows it (``options``). ``max_disk_bytes`` may be a size string
    (:func:`parse_size`). Each target asks ``_levels.select(model, target, n_pages, checked=...,
    modality=..., vectors_per_page=...)``, so the pick is exactly what ``build(target=...)``
    would take at that target.
    """
    from . import _ledger, _levels

    budget = parse_size(max_disk_bytes)
    if isinstance(n_pages, bool) or not isinstance(n_pages, int) or n_pages < 1:
        raise ValueError(f"n_pages must be a positive integer, got {n_pages!r}")
    if lowest is None:
        lowest = float(_ledger.setting("margins", "lowest_target", 0.9))
    for target in _targets(highest, lowest):
        sel = _levels.select(model, target, n_pages, checked=checked, modality=modality,
                             vectors_per_page=vectors_per_page)
        need = disk_needed(sel.pick, n_pages, vectors_per_page=vectors_per_page, dim=dim,
                           overhead_bytes_per_page=overhead_bytes_per_page, overhead_bytes=overhead_bytes)
        if need is not None and need <= budget:
            return target, sel.pick
    return None, None
