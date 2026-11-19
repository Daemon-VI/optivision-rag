"""optivision.easy._doctor: the environment report (FINAL_DESIGN 4.4).

Offline and without a GPU. The facts the doctor reads (torch build, GPUs, driver, versions) are
replaced where a test needs a particular machine; one test runs it for real on this machine and
one runs it in a fresh interpreter with torch and the loaders blocked.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from optivision.adapters import MODELS
from optivision.easy import _doctor, _encode
from optivision.easy._types import UnknownModel

ROOT = Path(__file__).resolve().parents[1]

T4 = {"name": "Tesla T4", "total": 15_828_320_256, "free": 15_000_000_000, "capability": [7, 5]}
A100 = {"name": "NVIDIA A100-SXM4-40GB", "total": 42_505_273_344, "free": 42_000_000_000, "capability": [8, 0]}
VERSIONS = {"numpy": "2.4.6", "scipy": "1.17.1", "torch": "2.11.0+cu128", "transformers": "5.16.1",
            "sentence-transformers": "6.1.0", "colpali-engine": "0.3.18", "peft": None, "accelerate": "1.14.0",
            "huggingface-hub": "1.27.0", "pillow": "12.3.0", "pypdfium2": "5.13.0", "torchao": None}


@pytest.fixture
def machine(monkeypatch):
    """A Linux machine with one T4, a CUDA torch, the campaign's versions and every loader installed."""
    state = {
        "system": {"os": "Linux-6.1-x86_64", "system": "Linux", "python": "3.11.9", "colab": False, "kaggle": False},
        "torch": {"installed": True, "version": "2.11.0+cu128", "cuda": "12.8", "cuda_available": True, "gpus": [T4]},
        "smi": True,
        "versions": dict(VERSIONS),
        "changed": [],
        "missing": [],
    }
    monkeypatch.setattr(_doctor, "_system_facts", lambda: dict(state["system"]))
    monkeypatch.setattr(_doctor, "_torch_facts", lambda: json.loads(json.dumps(state["torch"])))
    monkeypatch.setattr(_doctor, "_nvidia_smi", lambda: state["smi"])
    monkeypatch.setattr(_doctor, "_versions", lambda: dict(state["versions"]))
    monkeypatch.setattr(_doctor, "_changed_since_import", lambda: list(state["changed"]))
    monkeypatch.setattr(_doctor, "missing_packages", lambda *a, **k: list(state["missing"]))
    monkeypatch.setattr(_doctor, "_disk", lambda path: (path or "/content", 50_000_000_000))
    return state


def _labels(report) -> list[str]:
    return [line.label for line in report.lines]


def _line(report, label: str) -> str:
    return next(line.render() for line in report.lines if line.label == label)


# ------------------------------------------------------------------------------ this machine, for real


def test_doctor_runs_on_this_machine_without_a_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(_doctor, "_nvidia_smi", lambda: False)
    monkeypatch.setattr(_doctor, "_changed_since_import", list)
    report = _doctor.doctor(path=str(tmp_path / "index.ovx"))
    text = str(report)
    assert text.splitlines()[0] == "optivision doctor" and text.splitlines()[-1].startswith("Next: ")
    assert {"System", "Disk", "Libraries", "Status"} <= set(_labels(report))
    assert report.exit_code() == (0 if report.ok else 1)
    assert report.facts["disk"]["free"] is not None and str(tmp_path) in report.facts["disk"]["path"]
    assert report.as_dict()["ok"] is report.ok and report.as_dict()["next"] == report.next


def test_doctor_imports_light_and_runs_with_torch_and_the_loaders_blocked():
    code = (
        "import sys\n"
        "for name in ('torch', 'scipy', 'transformers', 'sentence_transformers', 'colpali_engine', 'pypdfium2'):\n"
        "    sys.modules[name] = None\n"
        "import optivision.easy as ov\n"
        "import optivision.easy._doctor as d\n"
        "assert ov.doctor is d.doctor and ov.DoctorReport is d.DoctorReport\n"
        "loaded = [m for m in ('torch', 'scipy', 'transformers') if sys.modules.get(m) is not None]\n"
        "assert not loaded, loaded\n"
        "code = d.run('vidore/colqwen2-v1.0-hf')\n"
        "print('EXIT', code)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120,
                            check=False)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "torch is not installed" in out and "Problem" in out and "EXIT 1" in out
    assert 'pip install "optivision-rag[easy-st]"' in out


# ------------------------------------------------------------------------------ torch, GPU, disk


def test_a_ready_machine_has_nothing_to_fix(machine):
    report = _doctor.doctor()
    assert report.ok and report.exit_code() == 0 and report.problems == []
    assert _line(report, "torch").endswith("2.11.0+cu128, CUDA build 12.8")
    assert _line(report, "GPU") == "GPU        Tesla T4: 15.8 GB memory, 15.0 GB free  [counted]"
    assert _line(report, "Disk").endswith("50.0 GB free at /content  [counted]")
    assert "transformers 5.16.1" in _line(report, "Libraries")
    assert _line(report, "Status").endswith("nothing to fix") and report.next == "ov.options()"


def test_a_cpu_only_windows_wheel_beside_an_nvidia_driver_is_a_problem(machine):
    machine["system"]["system"] = "Windows"
    machine["torch"].update(version="2.11.0+cpu", cuda=None, cuda_available=False, gpus=[])
    report = _doctor.doctor()
    assert not report.ok and report.exit_code() == 1
    assert "a CPU-only build" in _line(report, "torch")
    assert any("CPU-only build but an NVIDIA driver is present" in p and _doctor.TORCH_INSTALL_URL in p
               for p in report.problems)
    assert "none visible to torch" in _line(report, "GPU")
    assert report.next.startswith("fix the lines marked Problem")
    assert re.search(r"Status\s+1 problem to fix  \[counted\]", str(report))


def test_a_cpu_only_windows_wheel_without_an_nvidia_gpu_is_a_note(machine):
    machine["system"]["system"] = "Windows"
    machine["smi"] = False
    machine["torch"].update(version="2.11.0+cpu", cuda=None, cuda_available=False, gpus=[])
    report = _doctor.doctor()
    assert report.ok and any("default wheel on Windows" in n for n in report.notes)


def test_a_cuda_torch_that_sees_no_gpu_beside_a_driver_is_a_problem(machine):
    machine["torch"].update(cuda_available=False, gpus=[], error="RuntimeError: driver too old")
    report = _doctor.doctor()
    assert any("sees no GPU" in p and "driver too old" in p for p in report.problems)


def test_missing_torch_is_a_note_without_a_model_and_a_problem_with_one(machine):
    machine["torch"] = {"installed": False, "error": "ModuleNotFoundError: No module named 'torch'"}
    assert _doctor.doctor().ok
    report = _doctor.doctor("vidore/colqwen2-v1.0-hf")
    assert any("torch is not installed" in p for p in report.problems)


def test_a_library_upgraded_after_import_asks_for_a_restart(machine):
    machine["changed"] = [("transformers", "5.16.1", "5.19.0")]
    report = _doctor.doctor()
    assert any(p.startswith("restart the runtime: transformers on disk is 5.19.0 but this process imported 5.16.1")
               for p in report.problems)


def test_changed_since_import_compares_the_imported_version_with_the_files_on_disk(monkeypatch):
    installed = _encode.installed_version("pypdfium2")
    if installed is None:
        pytest.skip("pypdfium2 is not installed")
    stale = types.ModuleType("pypdfium2")
    stale.__version__ = "0.0.1"
    monkeypatch.setitem(sys.modules, "pypdfium2", stale)
    assert ("pypdfium2", "0.0.1", installed) in _doctor._changed_since_import()


def test_a_local_version_label_alone_is_not_a_change():
    assert _doctor._same_version("2.11.0+cpu", "2.11.0")  # torch's CPU wheel on Windows
    assert _doctor._same_version("5.16.1", "5.16.1")
    assert not _doctor._same_version("2.11.0+cpu", "2.11.0+cu128")
    assert not _doctor._same_version("5.16.1", "5.19.0")


def test_little_free_disk_is_noted(machine, monkeypatch):
    monkeypatch.setattr(_doctor, "_disk", lambda path: ("/content", 1_000_000_000))
    assert any("little free disk" in n for n in _doctor.doctor().notes)


def test_a_notebook_is_told_to_restart_after_installing(machine):
    machine["system"]["colab"] = True
    report = _doctor.doctor()
    assert "(Google Colab)" in _line(report, "System")
    assert any("%pip" in n and "restart the runtime" in n for n in report.notes)


# ------------------------------------------------------------------------------ a model


def test_a_model_gets_its_install_line_and_our_runs_settings(machine):
    report = _doctor.doctor("vidore/colqwen2-v1.0-hf")
    assert report.ok and report.model == "vidore/colqwen2-v1.0-hf"
    assert _line(report, "Model").endswith("vidore/colqwen2-v1.0-hf: loader sentence-transformers, licence "
                                           "apache-2.0, registry status measured")
    assert _line(report, "Install").endswith('pip install "optivision-rag[easy-st]"')
    ours = _line(report, "Our run")
    assert "float16 on Tesla T4 (on one)" in ours and "transformers 5.16.1" in ours
    assert report.next == 'ov.options("vidore/colqwen2-v1.0-hf")'
    assert str(report).splitlines()[0] == "optivision doctor: vidore/colqwen2-v1.0-hf"


def test_a_model_whose_loader_is_missing_is_a_problem_with_the_pip_line(machine):
    machine["missing"] = ["sentence-transformers"]
    report = _doctor.doctor("vidore/colqwen2-v1.0-hf")
    assert any('needs sentence-transformers: run pip install "optivision-rag[easy-st]"' in p for p in report.problems)


def test_a_remote_code_model_and_an_unverified_one_get_notes(machine):
    report = _doctor.doctor("TomoroAI/tomoro-colqwen3-embed-4b")
    assert any("trust_remote_code=True" in n for n in report.notes)
    report = _doctor.doctor("webAI-Official/webAI-ColVec1.1-4b")
    assert any("registry status unverified" in n for n in report.notes)
    assert "none recorded in our timings" in _line(report, "Our run")


def test_an_unknown_model_is_refused(machine):
    with pytest.raises(UnknownModel):
        _doctor.doctor("no-such-org/no-such-model")


# ------------------------------------------------------------------------------ known issues


def test_torchao_with_peft_is_a_note_in_general_and_a_problem_for_a_colpali_engine_model(machine):
    machine["versions"].update(peft="0.19.1", torchao="0.10.0")
    general = _doctor.doctor()
    assert general.ok and any("torchao" in n and "colpali-engine" in n for n in general.notes)
    report = _doctor.doctor("vidore/colpali-v1.3")
    problem = next(p for p in report.problems if "torchao" in p)
    assert "Fix: pip uninstall -y torchao" in problem and "recorded 2026-10-02" in problem
    assert "docs/GPU_RUN.md" in problem
    assert not any("torchao" in p for p in _doctor.doctor("vidore/colqwen2-v1.0-hf").problems)
    machine["versions"]["torchao"] = None
    assert not any("torchao" in p for p in _doctor.doctor("vidore/colpali-v1.3").problems)


def test_neomme_with_an_older_transformers_is_a_problem_with_its_fix(machine):
    report = _doctor.doctor("Hcompany/NeoMME-800M-Retriever-ST-late")  # transformers 5.16.1
    assert any('pip install -U "transformers>=5.17"' in p for p in report.problems)
    machine["versions"]["transformers"] = "5.19.0"
    assert _doctor.doctor("Hcompany/NeoMME-800M-Retriever-ST-late").ok


def test_remote_code_models_that_fail_on_transformers_5_are_flagged(machine):
    report = _doctor.doctor("jinaai/jina-embeddings-v4")
    assert any("fails under transformers 5" in p and "no fix is recorded here" in p for p in report.problems)
    machine["versions"]["transformers"] = "4.50.0"
    assert not any("transformers 5" in p for p in _doctor.doctor("jinaai/jina-embeddings-v4").problems)


def test_flash_attn_models_need_an_ampere_gpu(machine):
    model = "nvidia/llama-nemoretriever-colembed-1b-v1"
    assert any("flash_attn" in p for p in _doctor.doctor(model).problems)  # a T4 is compute capability 7.5
    machine["torch"]["gpus"] = [A100]
    assert not any("flash_attn" in p for p in _doctor.doctor(model).problems)


def test_known_issues_cite_files_of_this_repository_and_dates():
    issues = _encode.known_issues()
    assert len({i["id"] for i in issues}) == len(issues) >= 4
    for issue in issues:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", issue["recorded"]), issue["id"]
        assert "upstream" in issue and (issue["fix"] or issue.get("advice")), issue["id"]
        for source in issue["sources"]:
            assert (ROOT / source.split(" (")[0]).exists(), source
        for model in issue["when"].get("models", []):
            assert model in MODELS, model
        assert set(issue["when"]) <= {"installed", "below", "at_least", "models", "model_prefix", "loaders",
                                      "gpu_capability_below"}


def test_doctor_text_avoids_the_banned_words(machine):
    from tests.test_docs_examples import BANNED_PHRASES, BANNED_WORDS

    machine["versions"].update(peft="0.19.1", torchao="0.10.0")
    texts = [str(_doctor.doctor(m)) for m in (None, "vidore/colpali-v1.3", "jinaai/jina-embeddings-v4",
                                               "TomoroAI/tomoro-colqwen3-embed-4b")]
    texts.append((ROOT / "src" / "optivision" / "presets" / "known_issues.json").read_text(encoding="utf-8"))
    texts.append(str(_doctor.doctor("Hcompany/NeoMME-800M-Retriever-ST-late")))
    for text in texts:
        assert text.isascii()
        low = re.sub(r"\S*/\S*", " ", text.lower())  # paths (reports/universal/...) are not claims
        for phrase in (*BANNED_PHRASES, "lossless", "universal", "guaranteed", "sota", "state of the art", "veto"):
            assert phrase not in low, phrase
        assert BANNED_WORDS.search(low) is None and re.search(r"\bfirst\b", low) is None


def test_run_prints_the_report_and_returns_the_exit_code(machine, capsys):
    assert _doctor.run() == 0
    assert "optivision doctor" in capsys.readouterr().out
    machine["changed"] = [("torch", "2.10.0", "2.11.0")]
    assert _doctor.run() == 1
