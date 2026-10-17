"""Copy the results of a Kaggle measurement batch into the repository.

    python scripts/ingest_campaign.py ~/Downloads/results_01_colpali.zip [more zips ...]

The archives are the ``results_<batch>.zip`` files that ``notebooks/kaggle_batches`` write
(``scripts/measure_models.py`` inside). Each model's folder goes to
``reports/universal/campaign/<org>--<name>/``: the organisation keeps apart two models whose
names differ only in case (``tsystems/colqwen2.5-3b-...`` and ``Metric-AI/ColQwen2.5-3b-...``),
which a case-insensitive file system would otherwise merge. The batch's environment files
(``pip_freeze.txt``, ``gpu.txt``, ``commit.txt``, ``batch_summary.json``) go to
``reports/universal/campaign/batches/<batch>/``, and every model's outcome to
``reports/universal/campaign/outcomes.json``, which ``scripts/build_measured_catalog.py``
reads. A later batch that measures a model again replaces its folder and its outcome; the
earlier attempt stays listed under ``attempts``.

A follow-up batch (named ``f<n>_...``, from ``notebooks/kaggle_followup``) measures models the
campaign already holds, on other datasets or corpus sizes. Its archive goes to
``reports/universal/followup/<batch>/<org>--<name>/`` instead, its environment files to
``reports/universal/followup/<batch>/_batch/``, and its outcomes to
``reports/universal/followup/outcomes.json`` under ``"<model> @ <batch>"``. The campaign's
folders, outcomes and catalogue are not touched.

Members are read straight from the archive. Only ``.json``, ``.log`` and ``.txt`` files under
the batch folder are taken, never a path that leaves it, and nothing is executed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports" / "universal" / "campaign"
OUTCOMES = CAMPAIGN / "outcomes.json"
FOLLOWUP = ROOT / "reports" / "universal" / "followup"
FOLLOWUP_OUTCOMES = FOLLOWUP / "outcomes.json"
ALLOWED = {".json", ".log", ".txt"}
MAX_BYTES = 20 * 1024**2


def folder_of(model_id: str) -> str:
    return model_id.replace("/", "--")


def _hours(record: dict) -> float:
    return round(sum(float(s.get("seconds", 0)) for s in record.get("steps", {}).values()) / 3600, 2)


def _selection_file(rest: tuple[str, ...]) -> bool:
    return rest[0] == "selection_rules" or (rest[0] == "logs" and rest[-1].startswith("s7b_"))


def _ran_selection(record: dict) -> bool:
    return any(k.startswith("s7b ") for k in record.get("steps", {}))


def is_followup(batch: str) -> bool:
    return re.fullmatch(r"f\d+[a-z]?_[A-Za-z0-9_]+", batch) is not None


def batch_of(archive: Path) -> str:
    with zipfile.ZipFile(archive) as z:
        tops = {PurePosixPath(m.filename).parts[0] for m in z.infolist() if not m.is_dir()}
    if len(tops) != 1 or not next(iter(tops)).startswith("results_"):
        raise SystemExit(f"{archive}: expected one results_<batch>/ folder, found {sorted(tops)}")
    return next(iter(tops))[len("results_"):]


def ingest(archive: Path, outcomes: dict) -> list[str]:
    followup = is_followup(batch_of(archive))
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        tops = {PurePosixPath(m.filename).parts[0] for m in members}
        if len(tops) != 1 or not next(iter(tops)).startswith("results_"):
            raise SystemExit(f"{archive}: expected one results_<batch>/ folder, found {sorted(tops)}")
        top = tops.pop()
        batch = top[len("results_"):]
        runs = {}
        for m in members:
            parts = PurePosixPath(m.filename).parts
            if len(parts) == 3 and parts[2] == "run.json":
                runs[parts[1]] = json.loads(z.read(m))
        summary = json.loads(z.read(f"{top}/batch_summary.json"))
        commit = z.read(f"{top}/commit.txt").decode().strip() if f"{top}/commit.txt" in z.namelist() else None

        models_root = FOLLOWUP / batch if followup else CAMPAIGN
        batch_dir = models_root / "_batch" if followup else CAMPAIGN / "batches" / batch
        destinations = {short: folder_of(r["model"]) for short, r in runs.items()}
        if len({d.lower() for d in destinations.values()}) != len(destinations):
            raise SystemExit(f"{archive}: two models would share a folder: {destinations}")
        for dest in destinations.values():
            shutil.rmtree(models_root / dest, ignore_errors=True)
        shutil.rmtree(batch_dir, ignore_errors=True)

        skipped, orphans = [], []
        for m in members:
            path = PurePosixPath(m.filename)
            parts = path.parts
            if any(p in ("", ".", "..") for p in parts) or path.suffix not in ALLOWED or m.file_size > MAX_BYTES:
                skipped.append(m.filename)
                continue
            if len(parts) == 2:
                target = batch_dir / parts[1]
            elif parts[1] in destinations and _selection_file(parts[2:]) and not _ran_selection(runs[parts[1]]):
                # a selection study this folder's run never ran: left by another model with the same
                # short name (lightonai/colbertv2.0 and colbert-ir/colbertv2.0 in batch 11_text)
                target = batch_dir / "orphans" / parts[1] / Path(*parts[2:])
                orphans.append(m.filename)
            elif parts[1] in destinations:
                target = models_root / destinations[parts[1]] / Path(*parts[2:])
            else:
                skipped.append(m.filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(m))

    lines = []
    for item in summary:
        model = item["model"]
        record = next((r for r in runs.values() if r["model"] == model), None)
        outcome = item["outcome"]
        if record is None and outcome == "measured":
            outcome = "incomplete: its results are not in the archive (overwritten by a model with the same short name)"
        entry = {"batch": batch, "outcome": outcome, "commit": commit}
        if record is not None:
            check = record.get("check_retriever") or {}
            entry.update(folder=folder_of(model), kind=record["kind"], dtype=record["spec"].get("dtype", "auto"),
                         multi_gpu=bool(record["spec"].get("multi_gpu")), started_utc=record.get("started_utc"),
                         finished_utc=record.get("finished_utc"), gpu_hours=_hours(record),
                         check_retriever_ok=check.get("ok"),
                         selection_study=_ran_selection(record))
        if followup:
            entry["spec"] = {k: v for k, v in (record or {}).get("spec", {}).items() if k != "model"}
            outcomes[f"{model} @ {batch}"] = entry
            lines.append(f"  {model}: {outcome}")
            continue
        previous = outcomes.get(model)
        attempts = (previous or {}).get("attempts", [])
        if previous is not None and previous["batch"] != batch:  # the same archive again changes nothing
            attempts = [*attempts, {"batch": previous["batch"], "outcome": previous["outcome"]}]
        outcomes[model] = {**entry, **({"attempts": attempts} if attempts else {})}
        lines.append(f"  {model}: {outcome}")
    if orphans:
        lines.append(f"  set aside {len(orphans)} selection-study file(s) their folder's run never made: {orphans[:3]}")
    if skipped:
        lines.append(f"  skipped {len(skipped)} member(s): {skipped[:5]}")
    return [f"{archive.name} -> batch {batch}", *lines]


def write_readme(outcomes: dict) -> None:
    measured = sorted(m for m, o in outcomes.items() if o["outcome"] == "measured")
    lines = [
        "# Kaggle measurement campaign",
        "",
        "Written by `scripts/ingest_campaign.py` from the batches' `results_<batch>.zip`; do not edit by hand.",
        "",
        "Each model ran in a `notebooks/kaggle_batches` notebook on two Kaggle T4s: encode ViDoRe V1",
        "DocVQA + InfoVQA (500 pages each; BEIR SciFact for text models) with the package's own",
        "loader after `check_retriever` passed on 12 pairs, then the fixed-pipeline study",
        "(`s13_wide_study.py`, in `wide/`) and, when the session had time, the selection study",
        "(`s7b_selection_rules.py`, in `selection_rules/`). The figures are development measurements",
        "on those datasets, not a guarantee for another corpus. The vectors were not kept.",
        "",
        f"{len(measured)} of {len(outcomes)} models attempted are measured.",
        "",
        "| model | batch | outcome | inference dtype | GPU hours | selection study |",
        "|---|---|---|---|---|---|",
    ]
    for model, o in sorted(outcomes.items(), key=lambda kv: (kv[1]["batch"], kv[0].lower())):
        lines.append(f"| `{model}` | {o['batch']} | {o['outcome']} | {o.get('dtype', '')} | "
                     f"{o.get('gpu_hours', '')} | {'yes' if o.get('selection_study') else 'no'} |")
    lines += ["", ("Why a run failed is in the model's `logs/` and in its registry evidence "
                   "(`optivision models <id>`)."), ""]
    (CAMPAIGN / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archives", nargs="+", type=Path)
    a = ap.parse_args()
    followups = [x for x in a.archives if is_followup(batch_of(x))]
    campaign = [x for x in a.archives if x not in followups]
    if campaign:
        CAMPAIGN.mkdir(parents=True, exist_ok=True)
        outcomes = json.loads(OUTCOMES.read_text(encoding="utf-8")) if OUTCOMES.exists() else {}
        for archive in campaign:
            print("\n".join(ingest(archive, outcomes)))
        OUTCOMES.write_text(json.dumps(dict(sorted(outcomes.items())), indent=1) + "\n", encoding="utf-8", newline="\n")
        write_readme(outcomes)
        measured = sum(1 for o in outcomes.values() if o["outcome"] == "measured")
        print(f"{OUTCOMES.relative_to(ROOT)}: {len(outcomes)} models, {measured} measured")
    if followups:  # their own folder and outcomes: the campaign's record is not touched
        FOLLOWUP.mkdir(parents=True, exist_ok=True)
        runs = json.loads(FOLLOWUP_OUTCOMES.read_text(encoding="utf-8")) if FOLLOWUP_OUTCOMES.exists() else {}
        for archive in followups:
            print("\n".join(ingest(archive, runs)))
        FOLLOWUP_OUTCOMES.write_text(json.dumps(dict(sorted(runs.items())), indent=1) + "\n", encoding="utf-8",
                                     newline="\n")
        print(f"{FOLLOWUP_OUTCOMES.relative_to(ROOT)}: {len(runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
