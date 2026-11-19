"""Build ``src/optivision/presets/encode_rates.json`` from the committed Kaggle run records.

Easy mode prints how long encoding a folder will take and loads each model in the dtype of
our run (FINAL_DESIGN 4.2, ``dtype="recorded"``). Both come from here: for every model a
Kaggle run encoded, the settings it ran with (dtype, batch size, processor arguments, how
many GPUs the model spanned), the GPU, the library versions of the session, and per encode
step the pages, the step's seconds (``run.json``, model load included) and the page and
query seconds the encode log printed. Nothing is computed beyond division and rounding.

Sources, in order of preference:

* ``reports/universal/campaign/<model>/run.json`` (the measurement campaign), with the
  batch's ``gpu.txt`` and ``pip_freeze.txt`` (``campaign/outcomes.json`` names the batch);
* ``reports/universal/followup/<batch>/<model>/run.json`` for a model the campaign has no
  run of (ColQwen2.5 v0.2, the two Nemotron ColEmbed v2 models, answerai, and the easy-mode
  B0 batches once ingested), with ``<batch>/_batch/``.

A step counts only if ``run.json`` says ``ok`` and its log holds the timing line.

    python scripts/make_encode_rates.py            # rewrite the file
    python scripts/make_encode_rates.py --check    # exit 1 if it is out of date

``tests/test_easy_encode.py`` runs the check, so the shipped file cannot drift from the runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "universal"
CAMPAIGN = REPORTS / "campaign"
FOLLOWUP = REPORTS / "followup"
OUT = ROOT / "src" / "optivision" / "presets" / "encode_rates.json"

#: distributions whose versions the session recorded (pip_freeze.txt), as easy mode names them
LIBRARIES = ("torch", "transformers", "sentence-transformers", "colpali-engine", "peft", "accelerate")

NOTE = ("Encode timings and settings of this repository's own Kaggle runs, one entry per model. "
        "s_per_page is the page-encoding time per page (page_seconds / pages, from the encode log); "
        "step_s_per_page is the whole encode step per page (step_seconds / pages, from run.json: model "
        "download and load, the 12-pair check, page and query encoding); setup_s is the step minus page "
        "and query encoding. A text model's unit is a document. gpus is how many GPUs the model spanned. "
        "Times on other hardware differ; easy mode prints them as estimates with this basis.")


def _sha256_lf(path: Path) -> str:
    """Hash of the file with LF line endings, so a CRLF checkout gives the same value."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _floor(x: float, digits: int) -> float:
    scale = 10**digits
    return math.floor(x * scale + 1e-9) / scale


def _ceil(x: float, digits: int) -> float:
    scale = 10**digits
    return math.ceil(x * scale - 1e-9) / scale


def _timing(log: Path) -> dict[str, Any] | None:
    """The encode log's timing line (``{"tag": ..., "docs": ..., "doc_seconds": ...}``), or None."""
    if not log.is_file():
        return None
    found = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{") and '"doc_seconds"' in line:
            try:
                found = json.loads(line)
            except json.JSONDecodeError:
                continue
    return found


def _dtype_name(value: Any) -> str | None:
    return None if value in (None, "", "auto") else str(value).removeprefix("torch.")


def _gpu(batch_dir: Path) -> tuple[str | None, int]:
    """The GPU name and count of the session (nvidia-smi CSV: a header, then ``name, driver, memory``)."""
    path = batch_dir / "gpu.txt"
    if not path.is_file():
        return None, 0
    rows = [line.split(",")[0].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = [name for name in rows if name and name != "name"]
    return (names[0] if names else None), len(names)


def _versions(batch_dir: Path) -> dict[str, str]:
    path = batch_dir / "pip_freeze.txt"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        name = name.strip().lower().replace("_", "-")
        if name in LIBRARIES:
            out[name] = version.strip()
    return out


def _steps(run: dict[str, Any], folder: Path, sources: dict[str, str]) -> list[dict[str, Any]]:
    steps = []
    for key, step in sorted(run.get("steps", {}).items()):
        if not key.startswith("encode_") or step.get("status") != "ok":
            continue
        log = folder / "logs" / f"{key}.log"
        timing = _timing(log)
        if timing is None or not timing.get("docs"):
            continue
        sources[_rel(log)] = _sha256_lf(log)
        placement = timing.get("placement")
        devices = placement.get("devices") if isinstance(placement, dict) else None
        steps.append({
            "dataset": key.removeprefix("encode_"),
            "run": _rel(folder / "run.json"),
            "pages": int(timing["docs"]),
            "step_seconds": round(float(step["seconds"]), 1),
            "page_seconds": round(float(timing["doc_seconds"]), 1),
            "query_seconds": round(float(timing.get("query_seconds") or 0.0), 1),
            "gpus": max(1, len(devices)) if isinstance(devices, list) else 1,
            "logged_dtype": _dtype_name(timing.get("dtype")),
        })
    return steps


def _settings(run: dict[str, Any]) -> dict[str, Any]:
    spec = run.get("spec", {})
    out: dict[str, Any] = {"dtype": _dtype_name(spec.get("dtype")), "batch_size": spec.get("batch_size")}
    if spec.get("processor_kwargs"):
        out["processor_kwargs"] = spec["processor_kwargs"]
    if spec.get("trust_remote_code"):
        out["trust_remote_code"] = True
    return out


def _entry(model: str, kind: str, origin: str, batch: str, runs: list[tuple[dict, Path]], batch_dir: Path,
           sources: dict[str, str]) -> dict[str, Any] | None:
    steps: list[dict[str, Any]] = []
    for run, folder in runs:
        sources[_rel(folder / "run.json")] = _sha256_lf(folder / "run.json")
        steps.extend(_steps(run, folder, sources))
    if not steps:
        return None
    settings = _settings(runs[-1][0])
    if settings["dtype"] is None:  # a spec without a dtype ran at the loader's default; the log may name it
        logged = {s["logged_dtype"] for s in steps} - {None}
        settings["dtype"] = logged.pop() if len(logged) == 1 else None
    gpu_name, session_gpus = _gpu(batch_dir)
    for name in ("gpu.txt", "pip_freeze.txt"):
        if (batch_dir / name).is_file():
            sources[_rel(batch_dir / name)] = _sha256_lf(batch_dir / name)
    per_page = [s["page_seconds"] / s["pages"] for s in steps]
    per_step = [s["step_seconds"] / s["pages"] for s in steps]
    setup = [max(0.0, s["step_seconds"] - s["page_seconds"] - s["query_seconds"]) for s in steps]
    gpus = max(s["gpus"] for s in steps)
    for s in steps:
        del s["logged_dtype"]
    return {
        "modality": "text" if kind == "text" else "image",
        "unit": "document" if kind == "text" else "page",
        "origin": origin,
        "batch": batch,
        **settings,
        "gpus": gpus,
        "multi_gpu": gpus > 1,
        "gpu_name": gpu_name,
        "session_gpus": session_gpus,
        "versions": _versions(batch_dir),
        "s_per_page": [_floor(min(per_page), 4), _ceil(max(per_page), 4)],
        "step_s_per_page": [_floor(min(per_step), 4), _ceil(max(per_step), 4)],
        "setup_s": [_floor(min(setup), 1), _ceil(max(setup), 1)],
        "steps": steps,
    }


def build() -> dict[str, Any]:
    sources: dict[str, str] = {}
    models: dict[str, Any] = {}
    outcomes_path = CAMPAIGN / "outcomes.json"
    outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
    sources[_rel(outcomes_path)] = _sha256_lf(outcomes_path)
    for path in sorted(CAMPAIGN.glob("*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        model = run["model"]
        batch = outcomes.get(model, {}).get("batch", "")
        entry = _entry(model, run.get("kind", "image"), "campaign", batch, [(run, path.parent)],
                       CAMPAIGN / "batches" / batch, sources)
        if entry is not None:
            models[model] = entry
    # follow-up runs, for the models the campaign has no timing of
    followups: dict[str, list[tuple[dict, Path]]] = {}
    for path in sorted(FOLLOWUP.glob("*/*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["model"] not in models:
            followups.setdefault(run["model"], []).append((run, path.parent))
    for model, runs in sorted(followups.items()):
        # one entry, one set of settings: the runs that share the latest run's settings
        runs.sort(key=lambda r: r[0].get("started_utc", ""))
        latest = _settings(runs[-1][0])
        same = [r for r in runs if _settings(r[0]) == latest]
        batch_dir = same[-1][1].parent / "_batch"
        entry = _entry(model, same[-1][0].get("kind", "image"), "followup", same[-1][1].parent.name, same,
                       batch_dir, sources)
        if entry is not None:
            models[model] = entry
    return {"schema": 1, "note": NOTE, "sources": dict(sorted(sources.items())), "models": models}


def render(rates: dict[str, Any]) -> str:
    return json.dumps(rates, indent=1, sort_keys=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed file instead of writing")
    a = ap.parse_args()
    text = render(build())
    if a.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current.replace("\r\n", "\n") != text:
            print(f"{OUT.relative_to(ROOT)} is out of date; run scripts/make_encode_rates.py")
            return 1
        print("encode rates are up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(json.loads(text)['models'])} models, {len(text)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
