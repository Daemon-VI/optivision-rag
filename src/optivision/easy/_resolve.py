"""Model ids for easy mode: exact, case-insensitive or a unique short name (FINAL_DESIGN 4.1).

``resolve("colqwen2-v1.0-hf")`` returns the registry entry of ``vidore/colqwen2-v1.0-hf``; an
unknown id raises :class:`UnknownModel` naming the three closest ids, and a short name that two
organisations publish raises :class:`AmbiguousModel` listing both with loader and licence, so a
beginner never gets a model they did not mean. ``rows_model`` says whose catalogue rows the rule
reads: the model's own, a sibling's that gives the same vectors (``_measured_via``), or none.

The registry is imported inside the functions, so importing this module stays light.
"""

from __future__ import annotations

import difflib
from typing import Any

from ._types import AmbiguousModel, UnknownModel

__all__ = ["resolve", "rows_model", "licence"]


def _models() -> dict[str, Any]:
    from ..adapters import MODELS

    return MODELS


def licence(info: Any) -> str:
    """The licence the registry records, or ``"not recorded"``."""
    return info.license if getattr(info, "license", None) else "not recorded"


def resolve(model: str | None) -> Any:
    """The registry ``ModelInfo`` of ``model``, or None for ``model=None``.

    Tried in order: the exact id, the id ignoring case, then the part after the slash
    (``"colqwen2-v1.0-hf"``) when exactly one registry id ends in it.
    """
    if model is None:
        return None
    if not isinstance(model, str):
        raise TypeError(f"model must be a registry id such as 'vidore/colqwen2-v1.0-hf', got {type(model).__name__}")
    models = _models()
    name = model.strip()
    if name in models:
        return models[name]
    lower = {k.lower(): k for k in models}
    if name.lower() in lower:
        return models[lower[name.lower()]]
    short = name.lower().rsplit("/", 1)[-1]
    matches = sorted(k for k in models if k.lower().rsplit("/", 1)[-1] == short)
    if len(matches) == 1:
        return models[matches[0]]
    if len(matches) > 1:
        listing = "; ".join(f"{k} ({models[k].loader}, licence {licence(models[k])})" for k in matches)
        raise AmbiguousModel(f"{model!r} matches {len(matches)} registry models: {listing}. Pass the full id.")
    raise UnknownModel(f"{model!r} is not in the model registry. Closest ids: {', '.join(closest(name))}. "
                       "ov.options() lists every registered model.")


def closest(name: str, n: int = 3) -> list[str]:
    """The ``n`` registry ids closest to ``name`` (difflib), comparing whole ids and short names."""
    models = sorted(_models())
    key = name.lower()
    scored = []
    for k in models:
        whole = difflib.SequenceMatcher(None, key, k.lower()).ratio()
        part = difflib.SequenceMatcher(None, key.rsplit("/", 1)[-1], k.lower().rsplit("/", 1)[-1]).ratio()
        scored.append((-max(whole, part), k))
    return [k for _, k in sorted(scored)[:n]]


def rows_model(model_id: str) -> tuple[str | None, str]:
    """``(catalogue id whose rows the rule uses, basis)``.

    The basis is ``"own rows"``, ``"measured via <id>"`` (a registry entry with the same vectors
    as a measured sibling, e.g. ``vidore/colqwen2-v1.0`` via its pre-merged copy) or
    ``"unmeasured"``.
    """
    from ..adapters.measured import _measured_via
    from ._levels import data

    rows = data()["models"]
    if model_id in rows:
        return model_id, "own rows"
    via = _measured_via(model_id)
    if via is not None and via in rows:
        return via, f"measured via {via}"
    return None, "unmeasured"
