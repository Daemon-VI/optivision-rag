"""What this repository measured for each model: fixed pipelines, their size and retention.

The catalogue (``optivision/presets/measured.json``) is a copy of the fixed-pipeline rows of
the committed study reports under ``reports/universal``; ``scripts/build_measured_catalog.py``
writes it and a test keeps it equal to them. For a measured model it answers "which
compressions were tried on this model's vectors, how much smaller were they, and how much
of the float baseline's quality did they keep on that dataset".

Read these figures for what they are. Each retention is the pipeline's mean metric over the
float baseline's on all queries of one dataset (500 pages of ViDoRe V1 DocVQA or InfoVQA for
the image models, BEIR SciFact for the text model), with the report's bootstrap interval.
They are development measurements made in-sample on those datasets. The largest compression
that met a target there is the best of many candidates on the same queries, so it is
optimistic for that dataset and is not a statement about another corpus. Use them to see
what is plausible for a model and to choose what to try; use
:func:`optivision.certify` on your own queries to decide.

Nothing in the library reads this catalogue to choose a compression: selection is
model-blind, and the candidate list of ``certify`` does not depend on the model.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

__all__ = ["measured_models", "measured_results", "measured_shortlist"]


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    text = resources.files("optivision").joinpath("presets", "measured.json").read_text(encoding="utf-8")
    catalog = json.loads(text)
    if catalog.get("schema") != 1:
        raise RuntimeError(f"unknown measured-catalogue schema {catalog.get('schema')!r}")
    return catalog


def measured_models() -> list[str]:
    """Model ids that have fixed-pipeline rows in the catalogue."""
    return sorted(_catalog()["models"])


def _measured_via(model_id: str) -> str | None:
    """The catalogue model whose rows a measured id without rows of its own is measured through.

    That is the longest catalogue id its registry evidence names (``vidore/colqwen2-v1.0`` names
    its pre-merged copy); None when the id has rows, is not measured, or names none.
    """
    from . import MODELS

    info = MODELS.get(model_id)
    rows = measured_models()
    if info is None or info.status != "measured" or model_id in rows:
        return None
    named = [m for m in rows if m != model_id and m in info.evidence]
    return max(named, key=len) if named else None


def measured_results(model_id: str, dataset: str | None = None) -> dict[str, dict[str, Any]]:
    """The catalogue entries of one model: ``{dataset: entry}``.

    An entry holds ``source`` (the report it was copied from), ``n_docs``, ``n_queries``,
    ``metric`` (the metric the retention is of), ``baseline`` (the float baseline's value),
    an optional ``note``, and ``rows``. A row holds ``label``, ``pipeline`` (a specification
    ``Pipeline.from_dict`` accepts), ``compression`` (float32 bytes over stored bytes),
    ``vectors_per_doc``, ``bytes_per_doc``, ``retention`` and ``retention_interval``.

    Raises ``KeyError`` for a model without rows: a registered model can be loadable and
    still unmeasured, and a model measured only on a generated corpus has no rows here.
    """
    models = _catalog()["models"]
    if model_id not in models:
        raise KeyError(f"{model_id!r} has no measured fixed-pipeline rows; measured: {sorted(models)}")
    entries = models[model_id]
    if dataset is None:
        return json.loads(json.dumps(entries))  # a copy the caller may change
    if dataset not in entries:
        raise KeyError(f"{model_id!r} was not measured on {dataset!r}; measured on: {sorted(entries)}")
    return {dataset: json.loads(json.dumps(entries[dataset]))}


def measured_shortlist(model_id: str, min_retention: float = 0.97, *, on: str = "interval") -> list[dict[str, Any]]:
    """Pipelines that kept at least ``min_retention`` on every dataset the model was measured on.

    ``on="interval"`` asks for the lower end of the bootstrap interval to reach
    ``min_retention`` on each dataset, ``on="point"`` for the point estimate. A row of the
    result holds ``label``, ``pipeline``, and per-dataset ``compression``, ``retention`` and
    ``retention_interval``; rows are sorted by their smallest compression, largest first.
    Only pipelines run on every one of the model's datasets under the same label are
    compared.

    This is a shortlist of what to try, not a selection: the figures are in-sample for the
    measured datasets (see the module docstring). The pipelines can be handed to
    ``certify(..., space=...)`` as candidates and tested on your own queries.
    """
    if on not in ("interval", "point"):
        raise ValueError(f"on must be 'interval' or 'point', got {on!r}")
    if not 0.0 < float(min_retention) <= 1.5:
        raise ValueError(f"min_retention must be a retention such as 0.97, got {min_retention!r}")
    entries = measured_results(model_id)
    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, entry in entries.items():
        for row in entry["rows"]:
            by_label.setdefault(row["label"], {})[dataset] = row
    shortlist = []
    for label, rows in by_label.items():
        if len(rows) != len(entries):
            continue
        value = (lambda r: r["retention_interval"][0]) if on == "interval" else (lambda r: r["retention"])
        if not all(value(r) >= float(min_retention) for r in rows.values()):
            continue
        first = next(iter(rows.values()))
        if not first["pipeline"]["stages"]:
            continue  # the uncompressed baseline is not a compression
        shortlist.append({
            "label": label,
            "pipeline": first["pipeline"],
            "datasets": {d: {k: r[k] for k in ("compression", "retention", "retention_interval")}
                         for d, r in rows.items()},
        })
    shortlist.sort(key=lambda s: -min(d["compression"] for d in s["datasets"].values()))
    return shortlist
