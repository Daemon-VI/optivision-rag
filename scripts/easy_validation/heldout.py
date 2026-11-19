"""V1u: choose the two held-out page encoders on Kaggle and encode them (FINAL_DESIGN section 8).

The rule, frozen in PREREG_EASY part 1: "the 2 most-downloaded Hugging Face late-interaction
page retrievers not in the registry at freeze that pass ``check_retriever`` and fit on 2xT4".
``units.HELDOUT_CANDIDATES`` is the ranked list read from the Hub on 2026-10-21 (exclusions in
``units.HELDOUT_EXCLUDED``). The notebook tries them in order: a candidate that does not load
on two T4s or fails ``check_retriever`` (12 pairs, at most 2 misses) is recorded and the next
one is tried, until two are encoded. ``heldout_selected.json`` names them; batch e3b (the
computer_science half) and the Tier B batch read that file and run the same two, without
choosing again.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from easy_validation import units
from easy_validation.encode_unit import ROOT, free_model, run_model, write_json

N_SELECTED = 2


def _strip_padding(arrays: list[Any]) -> list[Any]:
    """Drop all-zero rows: the masked padding of a batch (MaxSim never needs them)."""
    import numpy as np

    out = []
    for a in arrays:
        a = np.asarray(a.float().cpu().numpy() if hasattr(a, "float") else a, dtype=np.float32)
        keep = np.linalg.norm(a, axis=1) > 1e-6
        out.append(a[keep])
    return out


def ops_colqwen3_adapter(candidate: units.HeldOut) -> Any:
    """The model card's own loader (``AutoModel``/``AutoProcessor`` with ``trust_remote_code``),
    at ``dims`` dimensions, SDPA attention (no flash-attention on a T4), wrapped as a
    ``CallableAdapter``. Vectors are taken as the model returns them (no extra normalisation),
    so ``check_retriever`` judges the model as published."""
    import torch
    from transformers import AutoModel, AutoProcessor

    from optivision.adapters import CallableAdapter

    spec = candidate.spec
    dtype = getattr(torch, spec.get("dtype", "float16"))
    model = AutoModel.from_pretrained(candidate.model, dims=int(spec.get("dims", 128)), trust_remote_code=True,
                                      dtype=dtype, device_map="cuda:0", attn_implementation="sdpa",
                                      revision=candidate.revision or None)
    model.eval()
    processor = AutoProcessor.from_pretrained(candidate.model, trust_remote_code=True,
                                              revision=candidate.revision or None)
    batch = int(spec.get("batch_size", 2))

    def run(inputs: dict) -> list[Any]:
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        with torch.no_grad():
            return list(model(**inputs))

    def documents(images: list[Any]) -> list[Any]:
        out: list[Any] = []
        for lo in range(0, len(images), batch):
            out += _strip_padding(run(processor.process_images([im.convert("RGB") for im in images[lo:lo + batch]])))
        return out

    def queries(texts: list[str]) -> list[Any]:
        out: list[Any] = []
        for lo in range(0, len(texts), 16):
            out += _strip_padding(run(processor.process_queries(texts[lo:lo + 16])))
        return out

    adapter = CallableAdapter(documents, queries)
    adapter.encoder = model  # for encoder_info and the V6 fault helpers
    adapter.revision = candidate.revision
    return adapter


def load(candidate: units.HeldOut) -> Any:
    from easy_validation.encode_unit import load_model

    if candidate.loader == "remote-code:ops-colqwen3":
        return ops_colqwen3_adapter(candidate)
    job = {"model": candidate.model, "loader": candidate.loader, "revision": candidate.revision or None,
           **{k: v for k, v in candidate.spec.items() if k not in ("pip", "dims")}}
    return load_model(job)


def run(job: dict, sources: dict, out: Path, session: Any) -> list[dict]:
    """Choose (or reuse) the held-out encoders and encode ``job["corpora"]`` with each."""
    records: list[dict] = []
    if job.get("selected_from"):
        chosen = json.loads((ROOT / job["selected_from"]).read_text(encoding="utf-8"))["selected"]
        by_id = {c.model: c for c in units.HELDOUT_CANDIDATES}
        candidates = [by_id[c["model"]] for c in chosen]
        reuse = True
    else:
        candidates = list(units.HELDOUT_CANDIDATES)
        reuse = False
    selected, attempts = [], []
    for candidate in candidates:
        if len(selected) == N_SELECTED:
            break
        if session.remaining() < 1800:
            attempts.append({"model": candidate.model, "outcome": "not tried (no time left)"})
            break
        for package in candidate.spec.get("pip") or []:
            os.system(f"{sys.executable} -m pip install -q '{package}'")
        print(f"== held-out candidate {candidate.model} ({candidate.downloads} downloads)", flush=True)
        spec = {"model": candidate.model, "corpora": list(job["corpora"]), "kind": "image", "role": "heldout",
                "loader": candidate.loader, "revision": candidate.revision, **candidate.spec, "keep_float32": []}
        try:
            adapter = load(candidate)
        except Exception as exc:  # noqa: BLE001 -- a candidate that does not load is skipped, as the rule says
            attempts.append({"model": candidate.model, "outcome": f"did not load: {type(exc).__name__}: {exc}"[:600]})
            free_model(candidate.model)
            continue
        recs = run_model(spec, adapter, sources, out, session)
        del adapter
        free_model(candidate.model)
        records += recs
        check = (recs[0].get("check_retriever") or {}) if recs else {}
        attempts.append({"model": candidate.model, "outcome": recs[0]["outcome"] if recs else "no record",
                         "check_ok": check.get("ok"), "own_first": check.get("own_first", check.get("hits"))})
        if check.get("ok"):
            selected.append({"model": candidate.model, "revision": candidate.revision, "loader": candidate.loader,
                             "downloads": candidate.downloads, "read_on": candidate.read_on, "spec": candidate.spec})
    name = "heldout_used.json" if reuse else "heldout_selected.json"
    write_json(out / name, {"rule": "the 2 most-downloaded Hugging Face late-interaction page retrievers not in the "
                                    "registry at freeze that pass check_retriever and fit on 2xT4",
                            "query": units.HELDOUT_QUERY, "selected": selected, "attempts": attempts,
                            "excluded_before_ranking": units.HELDOUT_EXCLUDED, "reused": reuse,
                            "complete": len(selected) == N_SELECTED})
    return records or [{"model": "heldout", "outcome": "no candidate passed"}]
