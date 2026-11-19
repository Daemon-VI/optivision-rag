"""V6: the build's wiring check on every registry loader that loads on Kaggle, and two planted faults.

Prediction (FINAL_DESIGN section 8): the wiring check (G-wire, FD 5.5: 12 text pages from
different files, each paired with a sentence unique among them, ``allowed_misses=2``, the
bundled sample PDF as a fallback) passes on every loader that loads, and fails on

- a re-initialised norm: every normalisation layer of the model reset to its initial values
  (``reset_parameters``: gain 1, bias 0), what a checkpoint whose norm weights did not load
  looks like;
- a query/document dtype mismatch: pages encoded at the recorded dtype, queries after casting
  the model to another one (float16 for a float32 model, float32 for a float16 one).

The check is the package's own ``optivision.easy._encode.wiring_check`` through
``load_encoder(adapter)``; ``wiring.json`` records each loader's result, and the fault runs.
Pages come from ViDoRe V3 hr (14 PDFs with a text layer).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from easy_validation import units
from easy_validation.encode_unit import build_page_records, free_model, load_model, write_json

SENTENCE = re.compile(r"[^.!?\n]{40,240}[.!?]")


def wiring_pages(src: Any, per_doc: int = 1, min_chars: int = 400) -> list[tuple[Any, Any]]:
    """One text page per PDF (the first with ``min_chars`` of text layer), as (PageRecord, image)."""
    pages, seen = [], set()
    for i, m in enumerate(src.meta):
        text = (src.texts.get(m["id"]) or {}).get("text_layer") or ""
        if m["doc_id"] in seen or len(text.strip()) < min_chars:
            continue
        seen.add(m["doc_id"])
        im = src.item(i)
        rec = build_page_records([{"id": m["id"], "doc_id": m["doc_id"], "page": m["page"], "blank": False,
                                   "sha256": "", "landscape": im.width > im.height}], src.texts, "text_layer")[0]
        pages.append((rec, im))
    return pages


def unique_sentences(pages: list[tuple[Any, Any]], n: int = 12) -> list[tuple[Any, str]]:
    """Fallback construction (only when ``_encode.wiring_check`` is missing): per page a sentence
    that occurs on no other page of the set."""
    out = []
    texts = [p[0].text for p in pages]
    for i, (rec, im) in enumerate(pages):
        for s in SENTENCE.findall(rec.text):
            s = " ".join(s.split())
            if len(s.split()) >= 8 and not any(s in t for j, t in enumerate(texts) if j != i):
                out.append((im, s))
                break
        if len(out) == n:
            break
    return out


def wiring(adapter: Any, pages: list[tuple[Any, Any]]) -> dict:
    """``_encode.wiring_check`` on these pages; a refusal (``WiringError``) is a failed check."""
    t0 = time.time()
    try:
        from optivision.easy import _encode
        from optivision.easy._types import WiringError
    except ImportError as exc:
        return fallback_wiring(adapter, pages, f"optivision.easy._encode missing: {exc}")
    if not hasattr(_encode, "wiring_check") or not hasattr(_encode, "load_encoder"):
        return fallback_wiring(adapter, pages, "optivision.easy._encode has no wiring_check / load_encoder")
    try:
        encoder = _encode.load_encoder(adapter)
        w = _encode.wiring_check(encoder, pages)
        return {"method": "optivision.easy._encode.wiring_check", "ok": bool(w.ok), "hits": getattr(w, "hits", None),
                "n": getattr(w, "n", None), "used_sample": getattr(w, "used_sample", None),
                "message": str(getattr(w, "message", "")), "seconds": round(time.time() - t0, 1)}
    except WiringError as exc:
        return {"method": "optivision.easy._encode.wiring_check", "ok": False, "refused": True,
                "message": str(exc)[:500], "seconds": round(time.time() - t0, 1)}


def fallback_wiring(adapter: Any, pages: list[tuple[Any, Any]], why: str) -> dict:
    from optivision.adapters import check_retriever

    pairs = unique_sentences(pages)
    report = check_retriever(adapter, [im for im, _ in pairs], [s for _, s in pairs], allowed_misses=2)
    return {"method": "check_retriever on 12 unique sentences (fallback)", "why": why, "ok": bool(report["ok"]),
            "report": report}


def torch_module(adapter: Any) -> Any:
    """The ``torch.nn.Module`` behind an adapter (searched one or two attributes deep)."""
    import torch

    stack = [adapter]
    for _ in range(3):
        nxt = []
        for obj in stack:
            if isinstance(obj, torch.nn.Module):
                return obj
            for name in ("model", "encoder", "_model", "module", "_encoder", "st_model"):
                child = getattr(obj, name, None)
                if child is not None:
                    nxt.append(child)
        stack = nxt
    raise LookupError("no torch module found behind the adapter")


def reinit_norms(model: Any) -> tuple[int, list]:
    """Reset every normalisation layer; returns how many and their saved state (for restoring)."""
    import torch

    saved = []
    for name, m in model.named_modules():
        cls = type(m).__name__
        if "Norm" not in cls or not any(True for _ in m.parameters(recurse=False)):
            continue
        saved.append((m, {k: v.detach().clone() for k, v in m.named_parameters(recurse=False)}))
        with torch.no_grad():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            else:
                for k, p in m.named_parameters(recurse=False):
                    p.fill_(1.0 if "weight" in k else 0.0)
    return len(saved), saved


def restore(saved: list) -> None:
    import torch

    with torch.no_grad():
        for m, state in saved:
            for k, p in m.named_parameters(recurse=False):
                p.copy_(state[k])


def mismatched(adapter: Any, model: Any, recorded: str) -> Any:
    """Pages at the recorded dtype, queries after a cast to another dtype (cast back afterwards)."""
    import torch

    from optivision.adapters import CallableAdapter
    from easy_validation.encode_unit import encode_items

    other = torch.float16 if recorded == "float32" else torch.float32
    original = next(model.parameters()).dtype

    def documents(items: list[Any]) -> list[Any]:
        c = encode_items(adapter, items, [f"w{i}" for i in range(len(items))], "image")
        return [c[i].vectors for i in range(len(c))]

    def queries(texts: list[str]) -> list[Any]:
        model.to(other)
        try:
            c = adapter.encode_queries(texts)
        finally:
            model.to(original)
        return [c[i].vectors for i in range(len(c))]

    return CallableAdapter(documents, queries)


def run(job: dict, sources: dict, out: Path, session: Any) -> list[dict]:
    src = sources[job["corpora"][0]]
    pages = wiring_pages(src)
    results: dict[str, Any] = {"pages": [p[0].id for p in pages], "loaders": {}, "faults": {}}
    for loader, spec in units.V6_LOADERS.items():
        if session.remaining() < 1200:
            results["loaders"][loader] = {"model": spec["model"], "outcome": "not run (no time left)"}
            continue
        entry: dict[str, Any] = {"model": spec["model"]}
        print(f"== V6 {loader}: {spec['model']}", flush=True)
        try:
            job_spec = {**spec, "corpora": []}
            fault_run = spec["model"] in units.V6_FAULT_MODELS
            if fault_run:
                job_spec["multi_gpu"] = False  # one GPU, so the model can be cast in place
            adapter = load_model(job_spec)
            entry["loaded"] = True
            entry["wiring"] = wiring(adapter, pages)
            if fault_run:
                model = torch_module(adapter)
                n, saved = reinit_norms(model)
                try:
                    results["faults"][f"{spec['model']}: re-initialised norms"] = {"layers": n, **wiring(adapter, pages)}
                finally:
                    restore(saved)
                recorded = spec.get("dtype", "float32")
                results["faults"][f"{spec['model']}: query/document dtype mismatch"] = {
                    "recorded": recorded, **wiring(mismatched(adapter, model, recorded), pages)}
            del adapter
        except Exception as exc:  # noqa: BLE001 -- a loader that does not load on Kaggle is recorded, not a V6 failure
            entry.setdefault("loaded", False)
            entry["error"] = f"{type(exc).__name__}: {exc}"[:600]
        free_model(spec["model"])
        results["loaders"][loader] = entry
        write_json(out / "wiring.json", results)
    write_json(out / "wiring.json", results)
    loaded = [e for e in results["loaders"].values() if e.get("loaded")]
    return [{"model": "V6 wiring", "outcome": f"{sum(1 for e in loaded if e.get('wiring', {}).get('ok'))} of "
                                               f"{len(loaded)} loaded loaders passed; faults {len(results['faults'])}"}]
