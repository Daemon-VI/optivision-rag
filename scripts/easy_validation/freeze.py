"""Freeze a part of the easy-mode preregistration: record the SHA-256 of every file it fixes.

    python scripts/easy_validation/freeze.py --part 1            # before 0.6.0 / B0
    python scripts/easy_validation/freeze.py --part 2            # before B1
    python scripts/easy_validation/freeze.py --part 3            # before the DT campaign
    python scripts/easy_validation/freeze.py --verify 1 2        # what the notebooks check

``reports/easy/PREREG_EASY.frozen.json`` gets one entry per part: the UTC time, the commit, and
``{path: sha256}`` of each file, hashed after turning CRLF into LF (as
``optivision.core.artifact.codeid._file_digest``). Part 1 also records the ledger's margins
as they stand (the ledger itself keeps changing as outcomes are appended, so its file is not
hashed). A part is written once: freezing it again is refused. A preregistration text that
still says "DRAFT" in its first lines cannot be frozen.

Every validation notebook verifies this file right after cloning, before any GPU work: the
parts it needs must be present and every listed file must still have its recorded digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "reports" / "easy" / "PREREG_EASY.frozen.json"

#: files each part fixes; optional ones are hashed when they exist
PARTS: dict[str, dict[str, list[str]]] = {
    "1": {"required": ["reports/easy/PREREG_EASY.md", "src/optivision/presets/easy_levels.json",
                       "src/optivision/easy/_levels.py", "scripts/easy_validation/units.py",
                       "scripts/easy_validation/rule.py", "scripts/easy_validation/predictions.py",
                       "scripts/easy_validation/decide.py", "scripts/easy_validation/scorer.py",
                       "scripts/easy_validation/evaluate.py", "scripts/easy_validation/api.py"],
          "optional": ["scripts/make_easy_levels.py", "src/optivision/easy/_resolve.py"]},
    "2": {"required": ["reports/easy/PREREG_EASY_PART2.md", "src/optivision/easy/_probes.py",
                       "src/optivision/easy/_veto.py", "src/optivision/easy/_docs.py",
                       "scripts/easy_validation/encode_unit.py", "scripts/easy_validation/data.py",
                       "scripts/easy_validation/heldout.py", "scripts/easy_validation/wiring_v6.py"],
          "optional": ["src/optivision/presets/veto_strength.json", "src/optivision/easy/_encode.py",
                       "src/optivision/easy/_probes_vlm.py", "scripts/easy_validation/vlm_probes.py"]},
    "3": {"required": ["reports/easy/PREREG_EASY_PART3.md", "scripts/easy_validation/dt.py"],
          "optional": []},
}
DRAFT_MARK = "DRAFT"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else {"schema": 1, "parts": {}}


def freeze(part: str, root: Path = ROOT) -> dict:
    record = load()
    if part in record["parts"]:
        raise SystemExit(f"part {part} is already frozen ({record['parts'][part]['frozen_utc']}); a frozen part "
                         "is never rewritten: amend it in a new, separately frozen part")
    files = {}
    for rel in PARTS[part]["required"]:
        p = root / rel
        if not p.exists():
            raise SystemExit(f"cannot freeze part {part}: {rel} is missing")
        files[rel] = digest(p)
    for rel in PARTS[part]["optional"]:
        if (root / rel).exists():
            files[rel] = digest(root / rel)
    for rel in files:
        if rel.endswith(".md") and DRAFT_MARK in "\n".join((root / rel).read_text(encoding="utf-8").splitlines()[:8]):
            raise SystemExit(f"{rel} is still marked {DRAFT_MARK}: remove the mark when the text is final")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                check=False).stdout.strip() or None
    except OSError:
        commit = None
    entry = {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": commit, "files": files}
    if part == "1":
        ledger = json.loads((root / "src/optivision/presets/easy_ledger.json").read_text(encoding="utf-8"))
        entry["ledger_margins"] = ledger["features"]["margins"]
    record["parts"][part] = entry
    return record


def verify(parts: list[str], root: Path = ROOT, frozen: Path | None = None) -> list[str]:
    """Problems that stop a notebook (empty when it may run)."""
    path = frozen or root / "reports" / "easy" / "PREREG_EASY.frozen.json"
    if not path.exists():
        return [f"{path.relative_to(root).as_posix()} is missing: the preregistration is not frozen"]
    record = json.loads(path.read_text(encoding="utf-8"))
    problems = [f"preregistration part {p} is not frozen" for p in parts if p not in record.get("parts", {})]
    for entry in record.get("parts", {}).values():
        for rel, sha in entry.get("files", {}).items():
            p = root / rel
            if not p.exists():
                problems.append(f"frozen file {rel} is missing")
            elif digest(p) != sha:
                problems.append(f"{rel} changed after it was frozen")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--part", choices=sorted(PARTS))
    g.add_argument("--verify", nargs="+", choices=sorted(PARTS))
    a = ap.parse_args(argv)
    if a.verify:
        problems = verify(a.verify)
        print("\n".join(problems) if problems else f"parts {', '.join(a.verify)} frozen and unchanged")
        return 1 if problems else 0
    record = freeze(a.part)
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"froze part {a.part}: {len(record['parts'][a.part]['files'])} files -> {FROZEN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
