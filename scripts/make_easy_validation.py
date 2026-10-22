"""Write the Kaggle notebooks of the easy-mode validation programme (FINAL_DESIGN section 8).

    python scripts/make_easy_validation.py        # writes notebooks/kaggle_easy/batch_e*.ipynb

One notebook per batch of ``scripts/easy_validation/units.py`` (B1-B3, the DT campaign and
the Tier B practice-question batch), each sized for one 11-hour Kaggle session with resume.
Every notebook clones the ``easy-mode`` branch and, before installing or encoding anything,
refuses to run unless the branch carries the frozen preregistration
(``reports/easy/PREREG_EASY.frozen.json``) with the parts it needs, every frozen file still
has its recorded SHA-256, and the scripts and inputs it needs are present. It then runs
``scripts/easy_validation/encode_unit.py`` (float16 vectors, labels, practice questions,
run records) and zips ``results_<batch>/``; ``scripts/easy_validation/ingest.py`` (or
``scripts/ingest_campaign.py``) takes the archive. The B0 notebooks in the same folder come
from ``scripts/make_kaggle_batches.py`` and are not touched here.

The output is deterministic: the same commit always writes the same bytes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from easy_validation import units  # noqa: E402
from optivision.adapters import MODELS  # noqa: E402

OUT = ROOT / "notebooks" / "kaggle_easy"
BRANCH = "easy-mode"
REPO_URL = "https://github.com/Daemon-VI/optivision-rag.git"

#: the refusal, run right after the clone; tests execute this cell against a temporary tree
GUARD = '''
# Preregistration guard (reports/easy/PREREG_EASY.md): stop here, before any install or GPU work,
# unless the cloned branch carries the frozen preregistration this batch needs, unchanged.
import hashlib, json, pathlib
def _digest(p):
    return hashlib.sha256(p.read_bytes().replace(b"\\r\\n", b"\\n")).hexdigest()
_repo = pathlib.Path(REPO)
_frozen = _repo / "reports" / "easy" / "PREREG_EASY.frozen.json"
_problems = []
if not _frozen.exists():
    _problems.append("reports/easy/PREREG_EASY.frozen.json is missing: the preregistration is not frozen")
else:
    _record = json.loads(_frozen.read_text(encoding="utf-8"))
    _parts = _record.get("parts", {})
    _problems += [f"preregistration part {p} is not frozen" for p in REQUIRED_PARTS if str(p) not in _parts]
    _listed = {}
    for _entry in _parts.values():
        _listed.update(_entry.get("files", {}))
    for _rel, _sha in sorted(_listed.items()):
        _p = _repo / _rel
        if not _p.exists():
            _problems.append(f"frozen file {_rel} is missing")
        elif _digest(_p) != _sha:
            _problems.append(f"{_rel} changed after it was frozen")
        elif _rel.endswith(".md") and "NOT FROZEN" in _p.read_text(encoding="utf-8"):
            _problems.append(f"{_rel} is still a draft")
    _problems += [f"{_rel} is not among the frozen files" for _rel in REQUIRED_FROZEN if _rel not in _listed]
_problems += [f"{_rel} is missing from the {BRANCH} branch" for _rel in REQUIRED_FILES if not (_repo / _rel).exists()]
if _problems:
    raise SystemExit("REFUSING TO RUN: nothing was installed or encoded.\\n- " + "\\n- ".join(_problems))
print(f"preregistration parts {list(REQUIRED_PARTS)} frozen and unchanged; {len(REQUIRED_FILES)} required files present")
'''

RESUME = '''
# Resume: a previous run of this notebook added as input (Add data -> Notebook output) gives back its
# finished shards; encode_unit.py skips every shard that is already there.
import glob, os, shutil, zipfile
_found = 0
for _z in sorted(glob.glob(f"/kaggle/input/**/results_{NAME}.zip", recursive=True)):
    with zipfile.ZipFile(_z) as _f:
        for _m in _f.infolist():
            _t = os.path.join(WORK, _m.filename)
            if _m.is_dir() or ".." in _m.filename or os.path.exists(_t) or _m.filename.endswith((".tmp", ".tmp.npz")):
                continue
            os.makedirs(os.path.dirname(_t), exist_ok=True)
            with _f.open(_m) as _src, open(_t, "wb") as _dst:
                shutil.copyfileobj(_src, _dst)
            _found += 1
for _d in sorted(glob.glob(f"/kaggle/input/**/results_{NAME}", recursive=True)):
    shutil.copytree(_d, RESULTS, dirs_exist_ok=True)
    _found += 1
print(f"resume: {_found} file(s) or folder(s) from earlier runs" if _found else "resume: starting fresh")
'''


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def required_files(b: units.Batch) -> list[str]:
    files = list(units.COMMON_FILES)
    roles = {j.role for j in b.jobs}
    if "heldout" in roles:
        files.append("scripts/easy_validation/heldout.py")
    if "wiring" in roles:
        files.append("scripts/easy_validation/wiring_v6.py")
    if "vlm" in roles:
        files.append("scripts/easy_validation/vlm_probes.py")
    if b.name.startswith("e4"):
        files.append("scripts/easy_validation/dt.py")
    return sorted(set(files) | set(b.needs_files))


def packages(b: units.Batch) -> list[str]:
    """The batch's extra pip packages, plus sentence-transformers 6 when an encode job loads through it.

    ``units.py`` (frozen) left it out of the batches that run colqwen2-v1.0-hf, which then failed
    to load (e3a, 2026-10-22). The environment is not part of any analysis, so it is added here
    (reports/easy/DEVIATIONS.md, D1)."""
    extra = list(b.packages)
    needs_st = any(j.role == "encode" and j.model in MODELS and MODELS[j.model].loader == "sentence-transformers"
                   for j in b.jobs)
    if needs_st and units.ST not in extra:
        extra.insert(0, units.ST)
    return extra


def describe(b: units.Batch) -> str:
    lo, hi = units.batch_hours(b)
    lines = []
    for j in b.jobs:
        corpora = ", ".join(f"{c} ({units.corpus(c).pages:,} {'documents' if units.corpus(c).modality == 'text' else 'pages'})"
                            for c in j.corpora)
        jl, jh = units.job_hours(j)
        what = {"heldout": "held-out encoders (V1u)", "wiring": "wiring test (V6) on every registry loader",
                "vlm": f"Tier B practice questions with {units.VLM_MODEL['model']}"}.get(j.role, f"`{j.model}`")
        lines.append(f"- {what} on {corpora}: {jl:.1f}-{jh:.1f} h")
    return "\n".join(lines) + f"\n\n**Estimate:** {lo:.1f}-{hi:.1f} GPU hours (session limit {units.DEADLINE_HOURS:.0f} h)."


def notebook(b: units.Batch) -> dict:
    jobs = [j.as_dict() for j in b.jobs]
    remote = ("\n**Remote code:** some models here run Python from their own Hugging Face repositories inside "
              "this session (`trust_remote_code`).\n" if b.remote_code else "")
    cells = [
        md(f"""
# OptiVision easy-mode validation, batch {b.name} ({b.prereg_batch})

{b.title}.

{describe(b)}

**Before it may run:** {b.needs_text}. The guard cell after the clone stops the notebook, before any install
or GPU work, if the `{BRANCH}` branch lacks `reports/easy/PREREG_EASY.frozen.json` with preregistration
part(s) {", ".join(str(p) for p in b.parts)}, if any frozen file changed since, or if a required file is missing.

**Run it:** *Settings -> Accelerator*: **GPU T4 x2**, *Internet*: **On**. Then *Save Version -> Save & Run All
(Commit)*. Download `results_{b.name}.zip` from the version's *Output* tab and ingest it with
`python scripts/easy_validation/ingest.py results_{b.name}.zip`.

**Resume:** if the session stopped at its deadline (`batch_summary.json` lists a job as `partial`), add this
notebook's output as input (*Add data -> Notebook output*) and run it again: finished shards are kept.
""" + remote),
        code(f'''
import json as _json
BRANCH = "{BRANCH}"
NAME = "{b.name}"
JOBS = _json.loads({json.dumps(json.dumps(jobs, sort_keys=True))})
EXTRA_PIP = {json.dumps(packages(b))}
DEADLINE_HOURS = {units.DEADLINE_HOURS}
REQUIRED_PARTS = {json.dumps([str(p) for p in b.parts])}
REQUIRED_FILES = {json.dumps(required_files(b), indent=1)}
REQUIRED_FROZEN = {json.dumps(sorted(b.needs_frozen), indent=1)}
'''),
        code('''
import json, os, shutil, subprocess, sys
if shutil.which("nvidia-smi") is None:
    raise SystemExit("NO GPU IN THIS SESSION. Settings (right panel) -> Accelerator -> GPU T4 x2, then Save Version "
                     "-> Save & Run All (Commit) again.")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
WORK = "/kaggle/working"
REPO = f"{WORK}/optivision-rag"
RESULTS = f"{WORK}/results_{NAME}"
os.environ["HF_HOME"] = "/tmp/hf"
!rm -rf {REPO} && git clone -q --depth 1 --branch {BRANCH} ''' + REPO_URL + ''' {REPO}
!cd {REPO} && git log --oneline -1
'''),
        code(GUARD),
        code('''
!pip install -q -e "{REPO}[bench,merge,vlm]" 2>&1 | tail -2
!pip install -q -U colpali-engine pypdfium2 2>&1 | tail -1
for package in EXTRA_PIP:
    !pip install -q "{package}" 2>&1 | tail -1
# Kaggle preinstalls torchao 0.10; peft refuses to work while an incompatible torchao is importable
!pip uninstall -y -q torchao
os.makedirs(RESULTS, exist_ok=True)
json.dump({"batch": NAME, "jobs": JOBS}, open(f"{WORK}/jobs.json", "w"), indent=1)
'''),
        code(RESUME),
        code('''
cmd = [sys.executable, f"{REPO}/scripts/easy_validation/encode_unit.py", f"{WORK}/jobs.json", "--out", RESULTS,
       "--data", "/tmp/optdata", "--deadline-hours", str(DEADLINE_HOURS)]
subprocess.run(cmd, check=False)
'''),
        code('''
!pip freeze > {RESULTS}/pip_freeze.txt
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > {RESULTS}/gpu.txt
!cd {REPO} && git rev-parse HEAD > {RESULTS}/commit.txt
!cat {RESULTS}/batch_summary.json
# store, do not deflate: float16 vectors barely compress; the folder goes once the archive tests sound
!cd {WORK} && rm -f results_{NAME}.zip && zip -q -r -0 results_{NAME}.zip results_{NAME} && zip -T -q results_{NAME}.zip && rm -rf results_{NAME} && ls -la results_{NAME}.zip
'''),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def render(b: units.Batch) -> str:
    return json.dumps(notebook(b), indent=1) + "\n"


def main(out: Path = OUT) -> int:
    out.mkdir(parents=True, exist_ok=True)
    batches = units.all_batches()
    names = [b.name for b in batches]
    assert len(set(names)) == len(names), "two batches share a name"
    for b in batches:
        assert units.batch_hours(b)[1] <= units.DEADLINE_HOURS, f"{b.name}: estimate above the session limit"
        (out / b.notebook).write_text(render(b), encoding="utf-8", newline="\n")
    total = [sum(units.batch_hours(b)[i] for b in batches) for i in (0, 1)]
    print(f"wrote {len(batches)} notebooks to {out}; {total[0]:.1f}-{total[1]:.1f} GPU hours in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
