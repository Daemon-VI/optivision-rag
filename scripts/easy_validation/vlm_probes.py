"""Tier B practice questions (local VLM, FINAL_DESIGN 5.3.2) on the V3 units, for V4/V5 scored apart.

For each corpus the package's own generator, ``optivision.easy._probes_vlm``, writes the
questions once (blind and two-stage: summarise the page, then a question from the summary
without the image; answerability filter, 60% overlap cap, type quotas; seed 0). They go to
``_corpus/<key>/probes_vlm.json`` with the same fields as Tier A. Then every model whose pages
B1/B2 encoded (the four design models, and the two held-out encoders named by
``heldout_selected.json`` on hr and computer_science) encodes them as queries into
``<model>__<corpus>/probes_vlm_f16.npz``; the CPU analyses merge them with that unit's pages.

The generator's entry point is looked up by name (``make_vlm_probes``, ``make_probes_vlm`` or
``make_probes``) and called with the page records, a lazy sequence of page images and the
seed; the notebook refuses to start when the module or the entry point is missing.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from easy_validation import units
from easy_validation.encode_unit import (
    ROOT,
    build_page_records,
    free_model,
    job_folder,
    load_model,
    page_record_dicts,
    probe_dict,
    save_corpus,
    write_json,
)

ENTRY_POINTS = ("make_vlm_probes", "make_probes_vlm", "make_probes")


class LazyImages(Sequence):
    """Page images decoded on access, so 600 pages are never all in memory."""

    def __init__(self, src: Any) -> None:
        self.src = src

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return [self.src.item(j) for j in range(*i.indices(len(self)))]
        return self.src.item(int(i))


def generator() -> Any:
    from optivision.easy import _probes_vlm

    for name in ENTRY_POINTS:
        fn = getattr(_probes_vlm, name, None)
        if callable(fn):
            return fn
    raise AttributeError(f"optivision.easy._probes_vlm has none of {ENTRY_POINTS}")


def generate(src: Any, out: Path) -> dict:
    folder = out / "_corpus" / src.key
    target = folder / "probes_vlm.json"
    if target.exists():
        return {"status": "present"}
    t0 = time.time()
    pages = page_record_dicts(src)
    records = build_page_records(pages, src.texts, "text_layer")
    fn = generator()
    params = inspect.signature(fn).parameters
    kwargs: dict[str, Any] = {}
    for name, value in (("images", LazyImages(src)), ("seed", units.PROBE_SEED), ("max_pages", units.PROBE_MAX_PAGES),
                        ("model", units.VLM_MODEL["model"]), ("device", "cuda")):
        if name in params:
            kwargs[name] = value
    probes = fn(records, **kwargs)
    write_json(target, [probe_dict(p) for p in probes])
    return {"status": "ok", "n": len(probes), "minutes": round((time.time() - t0) / 60, 1), "generator": fn.__name__,
            "vlm": units.VLM_MODEL}


def run(job: dict, sources: dict, out: Path, session: Any) -> list[dict]:
    records: list[dict] = []
    gen = {}
    for key in job["corpora"]:
        if session.remaining() < 1800:
            gen[key] = {"status": "not run (no time left)"}
            continue
        try:
            gen[key] = generate(sources[key], out)
        except Exception as exc:  # noqa: BLE001 -- recorded; the other corpora still run
            gen[key] = {"status": f"failed: {type(exc).__name__}: {exc}"[:600]}
        free_model(units.VLM_MODEL["model"])
        write_json(out / "vlm_generation.json", gen)
    heldout = []
    selected = ROOT / units.HELDOUT_SELECTED
    if selected.exists():
        heldout = [c["model"] for c in json.loads(selected.read_text(encoding="utf-8"))["selected"]]
    encoders = [(m, dict(spec), job["corpora"]) for m, spec in units.DESIGN.items()]
    encoders += [(m, {}, [c for c in job["corpora"] if c in units.V1U_CORPORA]) for m in heldout]
    for model, spec, corpora in encoders:
        if session.remaining() < 900:
            records.append({"model": model, "outcome": "not started (no time left)"})
            continue
        try:
            if model in units.DESIGN:
                adapter = load_model({"model": model, **spec})
            else:
                from easy_validation import heldout as ho

                adapter = ho.load(next(c for c in units.HELDOUT_CANDIDATES if c.model == model))
            for key in corpora:
                probes_file = out / "_corpus" / key / "probes_vlm.json"
                if not probes_file.exists():
                    continue
                probes = json.loads(probes_file.read_text(encoding="utf-8"))
                folder = job_folder(out, model, key)
                qs = adapter.encode_queries([p["text"] for p in probes], ids=[p["id"] for p in probes])
                save_corpus(folder / "probes_vlm_f16.npz", qs)
                write_json(folder / "run_vlm.json", {"model": model, "corpus": key, "n": len(probes),
                                                     "role": "vlm probes", "outcome": "complete"})
                records.append({"model": model, "corpus": key, "outcome": "vlm probes encoded"})
            del adapter
        except Exception as exc:  # noqa: BLE001
            records.append({"model": model, "outcome": f"error: {type(exc).__name__}: {exc}"[:600]})
        free_model(model)
    return records
