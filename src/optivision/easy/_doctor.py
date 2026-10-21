"""``ov.doctor()``: is this environment ready to encode with a model? (FINAL_DESIGN 4.4)

Installing is the riskiest step of easy mode (a CPU-only torch on Windows, a library upgraded
after the notebook imported it, a version a model is known to fail with), so the doctor reports,
without loading any model:

* OS and Python; the torch build (``torch.version.cuda is None`` is a CPU-only wheel, the
  default on Windows, which leaves an NVIDIA GPU unused); GPUs and their memory; free disk;
* the versions of the libraries that shape the vectors;
* combinations that broke a load in our own runs (``presets/known_issues.json``), each with
  where it was recorded, the date and the fix;
* with a model: its install line, our run's settings, whether its loader is installed and
  whether it runs its own repository code;
* "restart the runtime" when a library on disk is not the version this process imported.

This module imports no torch at module level; ``doctor()`` imports torch (when installed) to read
the CUDA facts. ``DoctorReport.exit_code()`` is 0 when nothing needs fixing, else 1.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

from ._claim import Block, Line, claim, lit, size
from ._encode import (
    _loader,
    _model_matches,
    _version_tuple,
    install_line,
    installed_version,
    known_issues,
    missing_packages,
    needs_remote_code,
    rate,
)

__all__ = ["DoctorReport", "doctor", "run"]

#: libraries listed by the doctor: (distribution, module)
LIBRARIES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"), ("scipy", "scipy"), ("torch", "torch"), ("transformers", "transformers"),
    ("sentence-transformers", "sentence_transformers"), ("colpali-engine", "colpali_engine"), ("peft", "peft"),
    ("accelerate", "accelerate"), ("huggingface-hub", "huggingface_hub"), ("pillow", "PIL"),
    ("pypdfium2", "pypdfium2"), ("torchao", "torchao"),
)

#: free disk below this is worth a note (a float16 copy of 2,500 pages of a 128-d model is about 0.5 GB)
LOW_DISK_BYTES = 2_000_000_000

TORCH_INSTALL_URL = "https://pytorch.org/get-started/locally/"


# ---------------------------------------------------------------------------- facts (test seams)


def _system_facts() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "colab": "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ,
        "kaggle": "KAGGLE_KERNEL_RUN_TYPE" in os.environ,
    }


def _nvidia_smi() -> bool:
    """Whether the NVIDIA driver's tool is on PATH: a sign of an NVIDIA GPU torch may not see."""
    return shutil.which("nvidia-smi") is not None


def _torch_facts() -> dict[str, Any]:
    """The torch build and the GPUs it sees; ``installed`` False when torch does not import."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 -- absent or broken: both are reported
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}".strip()}
    facts: dict[str, Any] = {"installed": True, "version": str(torch.__version__),
                             "cuda": getattr(torch.version, "cuda", None), "gpus": []}
    try:
        facts["cuda_available"] = bool(torch.cuda.is_available())
        for i in range(torch.cuda.device_count() if facts["cuda_available"] else 0):
            props = torch.cuda.get_device_properties(i)
            gpu = {"name": torch.cuda.get_device_name(i), "total": int(props.total_memory),
                   "capability": [int(props.major), int(props.minor)], "free": None}
            with contextlib.suppress(Exception):  # older drivers cannot answer
                gpu["free"] = int(torch.cuda.mem_get_info(i)[0])
            facts["gpus"].append(gpu)
    except Exception as exc:  # noqa: BLE001 -- a broken CUDA install
        facts["cuda_available"] = False
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _versions() -> dict[str, str | None]:
    return {dist: installed_version(dist) for dist, _ in LIBRARIES}


def _same_version(imported: str, installed: str) -> bool:
    """Equal, or equal apart from a local label only one side carries (torch reports ``2.11.0+cpu``
    while a CPU wheel's metadata says ``2.11.0``)."""
    if imported == installed:
        return True
    if ("+" in imported) != ("+" in installed):
        return imported.split("+")[0] == installed.split("+")[0]
    return False


def _changed_since_import() -> list[tuple[str, str, str]]:
    """``(distribution, imported, installed)`` for each library whose files on disk are not the
    version this process imported (a ``pip install -U`` after the import, as in a notebook)."""
    out = []
    for dist, module in LIBRARIES:
        loaded = sys.modules.get(module)
        imported = getattr(loaded, "__version__", None) if loaded is not None else None
        installed = installed_version(dist)
        if imported is not None and installed is not None and not _same_version(str(imported), installed):
            out.append((dist, str(imported), installed))
    return out


def _disk(path: str | None) -> tuple[str, int | None]:
    where = os.path.abspath(path or os.getcwd())
    probe = where
    while probe and not os.path.exists(probe):  # a folder the build will create: ask its parent
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return where, int(shutil.disk_usage(probe).free)
    except OSError:
        return where, None


# ---------------------------------------------------------------------------- report


@dataclass
class DoctorReport:
    """What ``doctor()`` found. ``ok`` is True when ``problems`` is empty; ``lines`` are the
    printed lines (``_claim.Line``), ``facts`` the raw values, ``str(report)`` the printout."""

    model: str | None
    lines: list[Line]
    problems: list[str]
    notes: list[str]
    facts: dict[str, Any] = field(default_factory=dict)
    next: str = "ov.options()"

    @property
    def ok(self) -> bool:
        return not self.problems

    def exit_code(self) -> int:
        """0 when nothing needs fixing, else 1 (the CLI's exit status)."""
        return 0 if self.ok else 1

    def block(self) -> Block:
        return Block(list(self.lines), next=self.next, title="optivision doctor" + (f": {self.model}" if self.model else ""))

    def __str__(self) -> str:
        return str(self.block())

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "ok": self.ok, "problems": list(self.problems), "notes": list(self.notes),
                "facts": self.facts, **self.block().as_dict()}

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:  # IPython shows the printout
        printer.text(str(self))


class _Lines:
    """Collects lines, problems and notes; a problem's line is labelled ``Problem``."""

    def __init__(self) -> None:
        self.lines: list[Line] = []
        self.problems: list[str] = []
        self.notes: list[str] = []

    def add(self, label: str, template: str, *parts: Any) -> None:
        self.lines.append(Line(label, template, *parts))

    def problem(self, template: str, *parts: Any) -> None:
        line = Line("Problem", template, *parts)
        self.lines.append(line)
        self.problems.append(line.text())

    def note(self, template: str, *parts: Any) -> None:
        line = Line("Note", template, *parts)
        self.lines.append(line)
        self.notes.append(line.text())


def _issue_applies(issue: dict[str, Any], model_id: str | None, info: Any, versions: dict[str, str | None],
                   torch_facts: dict[str, Any]) -> bool | None:
    """True if the issue's conditions hold here; None if it concerns models and no model was named."""
    when = issue.get("when", {})
    if any(versions.get(dist) is None for dist in when.get("installed", ())):
        return False
    for dist, bound in when.get("below", {}).items():
        if versions.get(dist) is None or _version_tuple(versions[dist]) >= _version_tuple(bound):
            return False
    for dist, bound in when.get("at_least", {}).items():
        if versions.get(dist) is None or _version_tuple(versions[dist]) < _version_tuple(bound):
            return False
    if "gpu_capability_below" in when:
        capable = [g for g in torch_facts.get("gpus", []) if g["capability"][0] >= int(when["gpu_capability_below"])]
        if capable:
            return False
    model_keys = {"models", "model_prefix", "loaders"} & set(when)
    if model_id is None:
        if {"models", "model_prefix"} & model_keys:
            return None
        return True
    return _model_matches(when, model_id, info)


def doctor(model: str | None = None, *, path: str | None = None) -> DoctorReport:
    """Report whether this environment can encode with ``model`` (FINAL_DESIGN 4.4).

    ``path`` is where the index and its temporary copy will go (default: the working directory),
    for the free-disk line. No model is loaded and nothing is downloaded.
    """
    info = None
    model_id = model
    if model is not None:
        from ._encode import _resolve_info

        info = _resolve_info(model)  # UnknownModel / AmbiguousModel for an id we cannot place
        model_id = info.model_id
    system = _system_facts()
    torch_facts = _torch_facts()
    versions = _versions()
    out = _Lines()

    where = " (Google Colab)" if system["colab"] else " (Kaggle)" if system["kaggle"] else ""
    out.add("System", "{0}, Python {1}{2}", lit(system["os"]), lit(system["python"]), lit(where))

    smi = _nvidia_smi()
    if not torch_facts["installed"]:
        text = "torch is not installed (it comes with the model's extra; a CUDA build is needed for a GPU)"
        if model_id is not None:
            out.problem(text)
        else:
            out.note(text)
    else:
        if torch_facts.get("cuda") is None:
            out.add("torch", "{0}, a CPU-only build", lit(torch_facts["version"]))
            if smi:
                out.problem("this torch is a CPU-only build but an NVIDIA driver is present, so the GPU will not "
                            "be used. Install a CUDA build of torch: pick the command at {0}", lit(TORCH_INSTALL_URL))
            elif system["system"] == "Windows":
                out.note("CPU-only torch is the default wheel on Windows; with an NVIDIA GPU install a CUDA "
                         "build from {0}", lit(TORCH_INSTALL_URL))
        else:
            out.add("torch", "{0}, CUDA build {1}", lit(torch_facts["version"]), lit(torch_facts["cuda"]))
            if not torch_facts.get("cuda_available") and smi:
                out.problem("torch has CUDA but sees no GPU although an NVIDIA driver is present: the driver may be "
                            "older than this torch's CUDA. {0}", lit(torch_facts.get("error", "")))
    gpus = torch_facts.get("gpus", [])
    for gpu in gpus:
        if gpu.get("free") is not None:
            out.add("GPU", "{0}: {1} memory, {2} free", lit(gpu["name"]),
                    claim(gpu["total"], "counted", construct="GPU memory", text=size(gpu["total"])),
                    claim(gpu["free"], "counted", construct="GPU memory free", text=size(gpu["free"])))
        else:
            out.add("GPU", "{0}: {1} memory", lit(gpu["name"]),
                    claim(gpu["total"], "counted", construct="GPU memory", text=size(gpu["total"])))
    if torch_facts["installed"] and not gpus:
        out.add("GPU", "none visible to torch; page models would encode on the CPU, which is slow")

    folder, free = _disk(path)
    if free is None:
        out.add("Disk", "free space at {0} could not be read", lit(folder))
    else:
        out.add("Disk", "{0} free at {1}", claim(free, "counted", construct="free disk", text=size(free)), lit(folder))
        if free < LOW_DISK_BYTES:
            out.note("little free disk: a build needs room for a float16 copy of the vectors while it runs")

    present = [f"{dist} {v}" for dist, v in versions.items() if v is not None]
    absent = [dist for dist, v in versions.items() if v is None and dist != "torchao"]
    out.add("Libraries", "{0}", lit(", ".join(present)))
    if absent:
        out.add("Not found", "{0}", lit(", ".join(absent)))

    for dist, imported, installed in _changed_since_import():
        out.problem("restart the runtime: {0} on disk is {1} but this process imported {2}; restart so the "
                    "new version loads", lit(dist), lit(installed), lit(imported))

    for issue in known_issues():
        applies = _issue_applies(issue, model_id, info, versions, torch_facts)
        if not applies:
            continue
        fix = issue.get("fix") or issue.get("advice") or "no fix recorded"
        recorded = f"recorded {issue.get('recorded')} in {', '.join(issue.get('sources', []))}"
        upstream = issue.get("upstream")
        if upstream:
            recorded += f"; upstream: {upstream}"
        if model_id is None and "loaders" in issue.get("when", {}):
            out.note("{0} (affects models loaded through {1}). Fix: {2} ({3})", lit(issue["title"]),
                     lit(", ".join(p.rstrip(":") for p in issue["when"]["loaders"])), lit(fix), lit(recorded))
        else:
            out.problem("{0}. Fix: {1} ({2})", lit(issue["title"]), lit(fix), lit(recorded))

    if info is not None:
        out.add("Model", "{0}: loader {1}, licence {2}, registry status {3}", lit(model_id), lit(_loader(info)),
                lit(info.license or "not recorded"), lit(info.status))
        out.add("Install", "{0}", lit(install_line(model_id)))
        entry = rate(model_id)
        if entry is not None:
            ran = entry.get("versions", {})
            libs = ", ".join(f"{k} {ran[k]}" for k in ("torch", "transformers", "sentence-transformers",
                                                       "colpali-engine") if k in ran)
            gpu = entry.get("gpu_name") or "a GPU"
            spread = "split over two of them" if entry.get("multi_gpu") else "on one"
            out.add("Our run", "{0} on {1} ({2}); {3}", lit(entry.get("dtype") or "dtype not recorded"), lit(gpu),
                    lit(spread), lit(libs))
        else:
            out.add("Our run", "none recorded in our timings for this model")
        multi = bool(entry and entry.get("multi_gpu")) and len(gpus) > 1
        missing = missing_packages(model_id, info, multi_gpu=multi)
        if missing:
            out.problem("{0} needs {1}: run {2}", lit(model_id), lit(", ".join(missing)), lit(install_line(model_id)))
        if needs_remote_code(info):
            out.note("{0} runs code from its own repository when it loads; read it, then pass "
                     "trust_remote_code=True", lit(model_id))
        if info.status == "unverified":
            out.note("{0} is not measured here (registry status unverified); see its evidence with "
                     "optivision models {0}", lit(model_id))

    if system["colab"] or system["kaggle"]:
        out.note("in a notebook, install with %pip, then restart the runtime before importing optivision")

    n = len(out.problems)
    if n:
        out.add("Status", "{0} to fix", claim(n, "counted", construct="problems", text=f"{n} problem" + ("s" if n > 1 else "")))
        nxt = "fix the lines marked Problem, then run ov.doctor() again"
    else:
        out.add("Status", "nothing to fix")
        nxt = f'ov.options("{model_id}")' if model_id else "ov.options()"
    facts = {"system": system, "torch": torch_facts, "nvidia_smi": smi, "versions": versions,
             "disk": {"path": folder, "free": free}, "install": install_line(model_id)}
    return DoctorReport(model_id, out.lines, out.problems, out.notes, facts, nxt)


def run(model: str | None = None, *, path: str | None = None) -> int:
    """Print the doctor's report and return the CLI exit status (0 nothing to fix, 1 otherwise)."""
    report = doctor(model, path=path)
    print(report)
    return report.exit_code()
