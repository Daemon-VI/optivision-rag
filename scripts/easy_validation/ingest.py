"""Copy the results of an easy-mode validation batch out of its Kaggle archive.

    python scripts/easy_validation/ingest.py ~/Downloads/results_e1a_easy_b1_design.zip [more zips ...]

The archives are the ``results_<batch>.zip`` files the ``notebooks/kaggle_easy/batch_e*``
notebooks write (``encode_unit.py`` inside); ``scripts/ingest_campaign.py`` hands such
archives here too. Two destinations, so that the repository never holds vectors:

- ``reports/easy/<batch>/``: the records (``run.json``, ``check_retriever.json``,
  ``batch_summary.json``, ``heldout_selected.json``, ``wiring.json``, the environment files,
  labels and practice-question texts): ``.json``, ``.txt`` and ``.log`` files up to 20 MB;
- ``data/easy/<batch>/`` (git-ignored): everything the CPU analyses read, vectors included
  (``.npz``, ``.json``, ``.json.gz``), which ``run_all.py`` scores.

The campaign's and the follow-ups' folders, outcomes and catalogue are never touched, and
``reports/easy/outcomes.json`` lists each batch's model outcomes. Members are read straight
from the archive: never a path that leaves the batch folder, and nothing is executed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "easy"
DATA = ROOT / "data" / "easy"
OUTCOMES = REPORTS / "outcomes.json"
RECORD_SUFFIXES = {".json", ".txt", ".log"}
DATA_SUFFIXES = {".npz", ".json", ".gz"}
MAX_RECORD_BYTES = 20 * 1024**2
#: page text and per-page lists stay out of the repository (large, and data/ has them)
DATA_ONLY = ("texts.json.gz", "pages.json")


def is_easy_validation(batch: str) -> bool:
    """``e<n>[a-z]_...``: the easy-mode validation notebooks (never a campaign or follow-up batch)."""
    return re.fullmatch(r"e\d+[a-z]?_[A-Za-z0-9_]+", batch) is not None


def batch_of(archive: Path) -> str:
    with zipfile.ZipFile(archive) as z:
        tops = {PurePosixPath(m.filename).parts[0] for m in z.infolist() if not m.is_dir()}
    if len(tops) != 1 or not next(iter(tops)).startswith("results_"):
        raise SystemExit(f"{archive}: expected one results_<batch>/ folder, found {sorted(tops)}")
    return next(iter(tops))[len("results_"):]


def _safe(parts: tuple[str, ...]) -> bool:
    return all(p not in ("", ".", "..") and not p.startswith("/") and ":" not in p for p in parts)


def ingest(archive: Path, outcomes: dict, reports: Path = REPORTS, data: Path = DATA) -> list[str]:
    batch = batch_of(archive)
    if not is_easy_validation(batch):
        raise SystemExit(f"{archive}: batch {batch!r} is not an easy-mode validation batch (e<n>_...)")
    rec_dir, data_dir = reports / batch, data / batch
    shutil.rmtree(rec_dir, ignore_errors=True)
    shutil.rmtree(data_dir, ignore_errors=True)
    n_rec = n_data = 0
    skipped = []
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for m in members:
            path = PurePosixPath(m.filename)
            parts = path.parts[1:]
            if not parts or not _safe(path.parts):
                skipped.append(m.filename)
                continue
            rel = Path(*parts)
            name = path.name
            suffix = ".gz" if name.endswith(".json.gz") else path.suffix
            if suffix in DATA_SUFFIXES and not name.endswith(".tmp.npz"):
                target = data_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 16 * 1024**2)
                n_data += 1
            if suffix in RECORD_SUFFIXES and m.file_size <= MAX_RECORD_BYTES and name not in DATA_ONLY:
                target = rec_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(m))
                n_rec += 1
            elif suffix not in DATA_SUFFIXES:
                skipped.append(m.filename)
    summary_path = rec_dir / "batch_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    commit_path = rec_dir / "commit.txt"
    commit = commit_path.read_text(encoding="utf-8").strip() if commit_path.exists() else None
    lines = [f"{archive.name} -> reports/easy/{batch}/ ({n_rec} records), data/easy/{batch}/ ({n_data} files)"]
    for item in summary:
        key = f"{item.get('model')} @ {item.get('corpus', '-')} @ {batch}"
        outcomes[key] = {"batch": batch, "outcome": item.get("outcome"), "commit": commit}
        lines.append(f"  {item.get('model')} {item.get('corpus', '')}: {item.get('outcome')}")
    if skipped:
        lines.append(f"  skipped {len(skipped)} member(s): {skipped[:5]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archives", nargs="+", type=Path)
    a = ap.parse_args(argv)
    REPORTS.mkdir(parents=True, exist_ok=True)
    outcomes = json.loads(OUTCOMES.read_text(encoding="utf-8")) if OUTCOMES.exists() else {}
    for archive in a.archives:
        print("\n".join(ingest(archive, outcomes)))
    OUTCOMES.write_text(json.dumps(dict(sorted(outcomes.items())), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{OUTCOMES.relative_to(ROOT)}: {len(outcomes)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
