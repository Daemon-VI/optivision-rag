"""Pareto frontiers over measured configurations.

A configuration is worth keeping only if no other one is at least as cheap on
every cost *and* at least as good, and strictly better on one of them. Costs
default to stored bytes per document; add ``"query_ms"`` (or anything else a
row carries) to trade latency as well.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def dominates(a: dict[str, Any], b: dict[str, Any], costs: Sequence[str], gain: str) -> bool:
    no_worse = all(a[c] <= b[c] for c in costs) and a[gain] >= b[gain]
    better = any(a[c] < b[c] for c in costs) or a[gain] > b[gain]
    return no_worse and better


def pareto_front(
    rows: Sequence[dict[str, Any]],
    costs: Sequence[str] = ("bytes_per_doc",),
    gain: str = "retention",
) -> list[dict[str, Any]]:
    """Non-dominated rows, sorted by the first cost (cheapest first)."""
    front = [r for r in rows if not any(dominates(o, r, costs, gain) for o in rows if o is not r)]
    # identical (cost, gain) points are all non-dominated; keep the first of each
    seen: set[tuple] = set()
    unique = []
    for r in sorted(front, key=lambda r: tuple(r[c] for c in costs) + (-r[gain],)):
        key = tuple(r[c] for c in costs) + (r[gain],)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def choose(
    rows: Sequence[dict[str, Any]],
    quality_target: float | None = None,
    max_cost: float | None = None,
    cost: str = "bytes_per_doc",
    gain: str = "retention",
) -> dict[str, Any] | None:
    """The cheapest row meeting ``quality_target``; or, with only ``max_cost``,
    the best row within budget. None if nothing qualifies."""
    ok = [r for r in rows if (max_cost is None or r[cost] <= max_cost)]
    if quality_target is not None:
        ok = [r for r in ok if r[gain] >= quality_target]
        return min(ok, key=lambda r: (r[cost], -r[gain])) if ok else None
    return max(ok, key=lambda r: (r[gain], -r[cost])) if ok else None
