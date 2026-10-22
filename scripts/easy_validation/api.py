"""The easy-mode functions the analyses score, behind one switchable backend.

The validation scores the frozen code, so by default every call goes to the package:
``optivision.easy._levels.select`` (the rule), ``optivision.easy._probes`` (``make_probes``,
``hard_set``, ``ndcg_at``) and ``optivision.easy._veto`` (``practice_check``,
``fixed_sequence``), with the signatures of docs/easy/CONTRACT.md. Tests and dry runs swap
in ``fakes.FakeBackend`` with :func:`use`. Small adapters here absorb attribute-name details
the contract leaves open (a hard set's probe ids, a selection's modality name), so the
analyses never reach into a module's internals.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from easy_validation import units


class Backend:
    """The package's own functions, imported on first use."""

    name = "optivision.easy"

    def _mod(self, name: str) -> Any:
        return importlib.import_module(f"optivision.easy.{name}")

    def select(self, model: str | None, target: float, n_pages: int | None = None, **kw: Any) -> Any:
        return self._mod("_levels").select(model, target, n_pages, **kw)

    def ladder(self, modality: str) -> list[Any] | None:
        lad = getattr(self._mod("_levels"), "LADDER", None)
        if not lad:
            return None
        for key in (modality, "image" if modality == "page" else "page" if modality == "image" else modality):
            if key in lad:
                return list(lad[key])
        return None

    def hard_set(self, probes: Sequence[Any], ref_rankings: dict, k_keep: int = 10) -> Any:
        return self._mod("_probes").hard_set(probes, ref_rankings, k_keep=k_keep)

    def probe(self, d: dict) -> Any:
        cls = getattr(self._mod("_probes"), "Probe", None)
        values = {"id": d["id"], "text": d["text"], "family": d.get("family"), "page_id": d.get("page_id"),
                  "relevant": frozenset(d.get("relevant") or ())}
        if cls is None:
            return SimpleNamespace(**values)
        try:
            return cls(**values)
        except TypeError:
            import dataclasses

            names = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set(values)
            return cls(**{k: v for k, v in values.items() if k in names})

    def practice_check(self, cand: dict, ref: dict, relevant: dict, clusters: dict, target: float) -> Any:
        return self._mod("_veto").practice_check(cand, ref, relevant, clusters, target,
                                                 B=units.PRACTICE_B, seed=units.PRACTICE_SEED)

    def fixed_sequence(self, sequence: Sequence[Any], rank_fn: Callable[[Any], dict], ref: dict, relevant: dict,
                       clusters: dict, target: float) -> Any:
        return self._mod("_veto").fixed_sequence(sequence, rank_fn, ref, relevant, clusters, target)

    def ndcg_at(self, ranking_ids: Sequence[str], relevant: Any, k: int = 5) -> float:
        return float(self._mod("_probes").ndcg_at(ranking_ids, relevant, k=k))


_BACKEND: list[Any] = [Backend()]


def backend() -> Any:
    return _BACKEND[-1]


@contextlib.contextmanager
def use(b: Any) -> Iterator[Any]:
    """Run the analyses against another backend (``fakes.FakeBackend()`` in tests)."""
    _BACKEND.append(b)
    try:
        yield b
    finally:
        _BACKEND.pop()


# --------------------------------------------------------------------------- views


@dataclass(frozen=True)
class Rung:
    """One ladder layout as the analyses need it (from a ``LayoutRow`` or the ladder)."""

    name: str
    spec: dict
    chain: str
    one_bit: bool
    merged: bool
    ratio: float | None = None
    lowest_kept: float | None = None
    status: str = ""
    row: Any = None

    @property
    def key(self) -> str:
        from optivision.core.identity import canonical_json

        return canonical_json(self.spec)

    @property
    def is_float(self) -> bool:
        return self.chain == "float"

    @property
    def size_class(self) -> str:
        if self.one_bit:
            return "merge>binary" if self.merged else "binary"
        return "merge>codec" if self.merged else "codec"


def rung(row: Any) -> Rung:
    chain = getattr(row, "chain", "") or ""
    spec = dict(getattr(row, "spec", {}) or {})
    stages = [s.get("stage") for s in spec.get("stages", [])]
    one_bit = bool(getattr(row, "one_bit", "binary" in stages))
    merged = bool(getattr(row, "merged", any(s and "merge" in s for s in stages)))
    if not chain:
        chain = "float" if all(s in (None, "float16") for s in stages) else "1-bit" if one_bit else "8-bit"
    ratio = getattr(row, "ratio_f32", None) or getattr(row, "ratio_f16", None)
    return Rung(str(getattr(row, "name", "")), spec, chain, one_bit, merged, ratio,
                getattr(row, "lowest_kept", None), str(getattr(row, "status", "")), row)


def select(model: str | None, target: float, n_pages: int | None, *, checked: bool, modality: str,
           vectors_per_page: float | None = None) -> Any:
    """``_levels.select``; for ``model=None`` the modality name the rule module accepts."""
    kw: dict[str, Any] = {"checked": checked}
    if vectors_per_page is not None:
        kw["vectors_per_page"] = vectors_per_page
    if model is not None:
        return backend().select(model, target, n_pages, **kw)
    last: Exception | None = None
    names = ("page", "image") if modality in ("page", "image") else (modality,)
    for name in names:
        try:
            return backend().select(None, target, n_pages, modality=name, **kw)
        except (KeyError, ValueError) as exc:
            last = exc
    raise last if last else ValueError(modality)


def rows(selection: Any) -> list[Rung]:
    return [rung(r) for r in selection.rows]


def pick(selection: Any) -> Rung:
    return rung(selection.pick)


def sequence(selection: Any) -> list[Any]:
    return list(selection.sequence)


def hard_ids(hs: Any) -> list[str]:
    """The probe ids of a hard set, whatever the attribute is called."""
    for name in ("ids", "probe_ids"):
        value = getattr(hs, name, None)
        if value is not None:
            return [str(x) for x in value]
    probes = getattr(hs, "probes", None)
    if probes is not None:
        return [str(getattr(p, "id", p)) for p in probes]
    if isinstance(hs, (list, tuple)):
        return [str(getattr(p, "id", p)) for p in hs]
    raise AttributeError(f"cannot read probe ids from {type(hs).__name__}")


def hard_digest(hs: Any) -> str | None:
    for name in ("sha256", "digest", "frozen_sha256"):
        value = getattr(hs, name, None)
        if value:
            return str(value)
    return None


def stat(cs: Any) -> dict:
    """A practice-check result as a dict: ``R``, ``U``, ``n``, ``passed`` (fires = not passed)."""
    out = {"R": float(cs.R), "U": float(cs.U), "n": int(cs.n), "passed": bool(cs.passed)}
    out["fires"] = not out["passed"]
    return out


def chosen(result: Any) -> Any:
    return result.chosen


def steps(result: Any) -> list[dict]:
    out = []
    for row, cs in getattr(result, "steps", []):
        out.append({"layout": rung(row).name, **stat(cs)})
    return out
