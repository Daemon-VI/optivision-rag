"""The easy-mode quickstart notebook, the guide and the README, run or checked offline.

``docs/notebooks/easy_quickstart.ipynb`` is a Colab notebook that builds the bundled 20-page sample PDF
with a real model. Here its code cells run on a CPU, with no network and no model download: the one
seam that loads a model (``_encode._build_adapter``) returns a small stand-in encoder, and the cells run
as written, with the real model id. The stand-in gives every page one vector per distinct word of its
text (it finds the text of a rendered page by the page's pixels), so a question that shares words with a
page finds it. It says nothing about any real model.

The Python blocks of ``docs/package/guide/EASY.md`` run the same way, in order, in one session. The guide
tags the blocks that need a model ``py`` (so ``tests/test_docs_examples.py``, which runs ``python`` blocks
with the real library only, leaves them alone) and the blocks that run with no model ``python``.

The README checks compare its model counts with the registry and the measured catalogue, so they cannot
go stale again.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from optivision.adapters import MODELS, CallableAdapter
from optivision.adapters import measured as catalogue

from ._easy_lint import banned_hits

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs" / "notebooks" / "easy_quickstart.ipynb"
GUIDE = ROOT / "docs" / "package" / "guide" / "EASY.md"
README = ROOT / "README.md"

pytestmark = pytest.mark.skipif(not NOTEBOOK.is_file(), reason="the repository layout (docs/) is absent")

DIM = 16
WORD = re.compile(r"[a-z0-9]{2,}")
FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.DOTALL | re.MULTILINE)


# ------------------------------------------------------------------------------------ the stand-in encoder


def _word_vector(word: str) -> np.ndarray:
    seed = int(hashlib.sha256(word.encode()).hexdigest()[:12], 16)
    v = np.random.default_rng(seed).standard_normal(DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _text_vectors(text: str) -> np.ndarray:
    words = list(dict.fromkeys(WORD.findall(text.lower())))[:48] or ["emptypage"]
    return np.stack([_word_vector(w) for w in words])


@lru_cache(maxsize=1)
def _sample_texts() -> dict[str, str]:
    """The text of each page of the bundled sample, keyed by the digest of the page as easy mode renders it."""
    from optivision.easy import _encode

    records = _encode._sample_records()
    images = _encode._render_sample([r.page for r in records])
    return {hashlib.sha256(i.tobytes()).hexdigest(): r.text for i, r in zip(images, records, strict=True)}


def stand_in_adapter() -> CallableAdapter:
    texts = _sample_texts()

    def documents(items):
        return [_text_vectors(texts.get(hashlib.sha256(i.tobytes()).hexdigest(), "unknown page")) for i in items]

    return CallableAdapter(documents, lambda queries: [_text_vectors(q) for q in queries])


def _use_stand_in(monkeypatch) -> None:
    """Every model the notebook or the guide names loads the stand-in; nothing is downloaded."""
    from optivision.easy import _encode

    adapter = stand_in_adapter()
    monkeypatch.setattr(_encode, "_build_adapter", lambda model_id, kwargs: adapter)
    monkeypatch.setattr(_encode, "missing_packages", lambda *a, **k: [])


def test_a_real_build_and_report_run_end_to_end_with_no_fakes(monkeypatch, tmp_path):
    """The build, the profile, the practice questions and the report were written as separate modules against a
    written contract; this runs them together on the bundled PDF (only the encoder is a stand-in)."""
    import optivision.easy as ov
    from optivision.easy import _encode

    _use_stand_in(monkeypatch)
    pdf = tmp_path / "pdfs"
    pdf.mkdir()
    sample = Path(_encode.__file__).parent / "data" / "sample.pdf"
    (pdf / "sample.pdf").write_bytes(sample.read_bytes())
    index = ov.build(pdf, model="vidore/colqwen2-v1.0-hf", path=tmp_path / "t.ovx", progress=False)
    assert index.n_pages == 20 and index.search("committee")[0].id.endswith("::p" + str(index.search("committee")[0].page))
    report = index.report()
    assert [t for t, _ in report.sections][:2] == ["Summary", "Layouts at this target"]
    saved = report.save(tmp_path / "r.md").read_text(encoding="utf-8")
    assert saved.startswith("# OptiVision report") and "Summary" in saved and "Layouts at this target" in saved
    assert "path" in report.to_markdown() and report.to_html() and report.to_json()


# ------------------------------------------------------------------------------------------- the notebook


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _tags(cell: dict) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def test_notebook_is_valid_nbformat_4():
    nb = _load()
    assert nb["nbformat"] == 4 and isinstance(nb["nbformat_minor"], int)
    assert isinstance(nb["metadata"], dict) and nb["cells"]
    ids = []
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], (str, list)) and all(isinstance(s, str) for s in cell["source"])
        if cell["cell_type"] == "code":
            assert "execution_count" in cell and isinstance(cell["outputs"], list)
        else:
            assert "outputs" not in cell and "execution_count" not in cell
        ids.append(cell.get("id"))
    if all(ids):  # nbformat 4.5 cell ids are unique
        assert len(set(ids)) == len(ids)
    with contextlib.suppress(ImportError):
        import nbformat

        nbformat.validate(nbformat.reads(NOTEBOOK.read_text(encoding="utf-8"), as_version=4))


def test_notebook_names_a_registry_model_and_marks_the_cells_that_need_it():
    cells = [c for c in _load()["cells"] if c["cell_type"] == "code"]
    model_line = [c for c in cells if re.search(r"^MODEL = \"([^\"]+)\"", _source(c), re.MULTILINE)]
    assert len(model_line) == 1
    model = re.search(r"^MODEL = \"([^\"]+)\"", _source(model_line[0]), re.MULTILINE).group(1)
    assert model in MODELS and MODELS[model].status == "measured"
    needs = [c for c in cells if "needs-model" in _tags(c)]
    assert model_line[0] in needs and len(needs) >= 4
    for cell in cells:
        if re.search(r"\bMODEL\b", _source(cell)):
            assert "needs-model" in _tags(cell)


def test_notebook_wording_passes_the_claim_lint():
    text = "\n".join(_source(c) for c in _load()["cells"])
    assert banned_hits(text) == []
    assert "veto" not in text.lower()
    assert ROOT.joinpath("docs", "package", "guide", "EASY.md").is_file()
    assert "docs/package/guide/EASY.md" in text


def test_notebook_runs_on_a_cpu_with_a_stand_in_encoder(tmp_path, monkeypatch):
    _use_stand_in(monkeypatch)
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    namespace: dict = {"__name__": "notebook"}
    ran = []
    out = io.StringIO()
    for cell in _load()["cells"]:
        if cell["cell_type"] != "code" or "colab-only" in _tags(cell):
            continue
        code = _source(cell)
        if "work-dir" in _tags(cell):
            replacement = f"WORK_DIR = {work.as_posix()!r}"
            code = re.sub(r"^WORK_DIR = .*$", lambda _m: replacement, code, flags=re.MULTILINE)
        assert not re.search(r"^\s*[%!]", code, re.MULTILINE), "a shell or magic line outside a colab-only cell"
        with contextlib.redirect_stdout(out):
            exec(compile(code, f"<cell {len(ran)}>", "exec"), namespace)
        ran.append(code)
    assert len(ran) >= 7
    index = namespace["index"]
    assert index.n_pages == 20 and Path("my_pdfs.ovx").is_dir()
    hits = namespace["hits"]
    assert [h.page for h in hits][0] == 20 and hits[0].id == "my_pdfs/sample.pdf::p20"  # the page about dyes
    assert namespace["page"].size[0] <= 600
    assert Path("my_report.md").is_file() and "Summary" in namespace["report"].sections[0][0]
    from optivision.easy import _ledger

    assert "Next:" in out.getvalue()

    assert _ledger.status() in out.getvalue()  # the plan and the build print the status sentence
    index.close()


# ----------------------------------------------------------------------------------------------- the guide


def _blocks(path: Path, tags: tuple[str, ...] = ("py", "python")) -> list[str]:
    return [code for lang, code in FENCE.findall(path.read_text(encoding="utf-8")) if lang in tags]


def test_the_guide_runs_in_order_with_the_stand_in(tmp_path, monkeypatch):
    _use_stand_in(monkeypatch)
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "guide"}
    blocks = _blocks(GUIDE)
    assert len(blocks) >= 9
    ran = 0
    for number, code in enumerate(blocks):
        if "qdrant" in code:
            pytest.importorskip("qdrant_client")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, f"<guide block {number}>", "exec"), namespace)
        ran += 1
    assert ran == len(blocks)
    assert namespace["index"].n_pages == 20
    assert namespace["hits"][0].page == 20 and namespace["export"].n_points == 20
    namespace["index"].close()


def test_blocks_that_need_no_model_run_without_a_stand_in(tmp_path, monkeypatch):
    """The ``python`` blocks are the ones ``tests/test_docs_examples.py`` runs: plain, offline, no files."""
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "guide"}
    blocks = _blocks(GUIDE, ("python",))
    assert blocks
    for number, code in enumerate(blocks):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, f"<python block {number}>", "exec"), namespace)
    assert not list(tmp_path.iterdir())


def test_the_guide_quotes_the_status_sentence_and_the_ledger_as_it_is():
    from optivision.easy import _ledger

    text = GUIDE.read_text(encoding="utf-8")
    assert _ledger.status() in text
    assert _ledger.practice_difficulty() in re.sub(r"\s+", " ", text)
    ledger = _ledger.ledger()["features"]
    for feature, label in (("qdrant_export", "**Qdrant export: "), ("byaldi_import", "**byaldi import: ")):
        state = "on" if ledger[feature]["on"] else "off"
        assert f"{label}{state}.**" in text, feature
    off = {"small_memory_layout": "Small-memory layout", "probe_allowance": "Practice questions raising the layout one step",
           "type_presets": "Document-type presets"}
    for feature, label in off.items():
        row = next(ln for ln in text.splitlines() if ln.startswith(f"| {label}"))
        assert ("Off" in row) == (not ledger[feature]["on"]), feature
    assert "**not** been run against the byaldi package" in text.replace("\n", " ")


def test_the_guide_lists_the_results_that_did_not_pass():
    text = GUIDE.read_text(encoding="utf-8")
    for name in ("V1t", "V3", "V4", "V5b", "V5c", "V6"):
        assert re.search(rf"^\| {name} ", text, re.MULTILINE), name
    assert "82.4%" in text and "33%" in text and "35%" in text and "45%" in text
    assert "reports/easy/DEVIATIONS.md" in text and "reports/easy/results/SUMMARY.md" in text
    summary = (ROOT / "reports" / "easy" / "results" / "SUMMARY.md")
    if summary.is_file():
        s = summary.read_text(encoding="utf-8")
        for fragment in ("82.4% of 102 pairs", "33% of 54 units", "35% of 54 units", "45% of 31 units",
                         "fires on 11% of 325 pairs", "fires on 43% of 30 pairs", "fires on 0% of 7 pairs",
                         "32/32 units"):
            assert fragment in s or fragment.replace("fires on ", "") in s, fragment


def test_the_guide_passes_the_claim_lint_and_its_links_resolve():
    text = GUIDE.read_text(encoding="utf-8")
    assert banned_hits(text) == [] and "veto" not in text.lower()
    assert text.isascii()
    for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        assert (GUIDE.parent / target.split("#", 1)[0]).exists(), target


# ---------------------------------------------------------------------------------------------- the README


def test_readme_model_counts_match_the_registry_and_the_catalogue():
    text = README.read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", text)
    modality = Counter(m.modality for m in MODELS.values())
    status = Counter(m.status for m in MODELS.values())
    m = re.search(r"lists (\d+) late-interaction models: (\d+) that take page images and (\d+) that take text", flat)
    assert m, "the model-coverage sentence is missing"
    assert tuple(map(int, m.groups())) == (len(MODELS), modality["image"], modality["text"])
    for name in ("measured", "untested", "unverified"):
        found = re.search(rf"`{name}` \((\d+)\)", flat)
        assert found, name
        assert int(found.group(1)) == status.get(name, 0), name
    cat = catalogue.measured_models()
    kinds = Counter(MODELS[x].modality for x in cat)
    c = re.search(r"holds fixed-pipeline measurements for (\d+) models: (\d+) page models .* and (\d+) text models", flat)
    assert c, "the catalogue sentence is missing"
    assert tuple(map(int, c.groups())) == (len(cat), kinds["image"], kinds["text"])
    dims = sorted(MODELS[x].dimension for x in cat)
    d = re.search(r"(\d+) to ([\d,]+) dimensions", flat)
    assert d and (int(d.group(1)), int(d.group(2).replace(",", ""))) == (dims[0], dims[-1])


def test_readme_has_an_easy_mode_section_that_links_the_guide():
    text = README.read_text(encoding="utf-8")
    assert re.search(r"^## Easy mode", text, re.MULTILINE)
    assert "docs/package/guide/EASY.md" in text
    section = text.split("## Easy mode", 1)[1].split("\n## ", 1)[0]
    assert banned_hits(section) == [] and "veto" not in section.lower()
    assert "have not been checked on new documents" in section or "not been checked on new documents" in section
    assert os.path.exists(ROOT / "docs" / "package" / "guide" / "EASY.md")
