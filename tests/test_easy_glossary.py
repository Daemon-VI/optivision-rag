"""The glossary of easy mode (FINAL_DESIGN 6.5): fourteen terms with their exact meanings."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _easy_lint import banned_hits, non_ascii

from optivision.easy import _glossary
from optivision.easy._claim import Block

#: the table of FINAL_DESIGN 6.5, word for word
DESIGN_TABLE = {
    "layout": "How stored vectors are compressed",
    "target": "Share of full-precision search quality (nDCG@5) to keep; not \"answers correct\"",
    "nDCG@5": "0-1 score of how high the right pages rank in the top 5",
    "full precision": "The model's original vectors",
    "float16": "Uncompressed, 2 bytes per number; our size baseline",
    "8-bit / 4-bit / 1-bit": "Fewer bits per number; centred 4-bit subtracts the average first",
    "merged (1 in 4)": "Similar vectors on a page combined, keeping about a quarter",
    "lowest kept": "Bottom of the 95% range we measured, on our worst dataset",
    "close call": "Cleared the target by under half a point",
    "practice questions": "Made from your PDFs' text; easier than real ones; only make us pick safer",
    "practice check": "Testing a layout on practice questions before saving",
    "unchecked": "No practice check possible (scans, images, under 60 text pages, vectors); no 1-bit layouts",
    "drift": "Your model now producing different vectors than at build time",
    "build record": "Model, versions and settings saved with the index",
}


def test_terms_are_the_design_table_exactly():
    assert dict(_glossary.TERMS) == DESIGN_TABLE
    assert list(_glossary.TERMS) == list(DESIGN_TABLE), "the order of the table is kept"
    assert len(_glossary.TERMS) == 14


def test_terms_are_ascii_and_pass_the_claim_lint():
    for term, meaning in _glossary.TERMS.items():
        assert non_ascii(term + meaning) == [], term
        assert banned_hits(term + " " + meaning) == [], term


def test_the_target_meaning_never_says_answers_are_correct():
    meaning = _glossary.TERMS["target"]
    assert "not \"answers correct\"" in meaning and "nDCG@5" in meaning
    assert "accuracy" not in meaning.lower()


def test_glossary_of_one_term_prints_with_a_next_line():
    block = _glossary.glossary("lowest kept")
    assert isinstance(block, Block)
    text = str(block)
    assert text.splitlines()[0].startswith("lowest kept")
    assert DESIGN_TABLE["lowest kept"] in text
    assert text.splitlines()[-1].startswith("Next: ")
    assert repr(block) == text


def test_glossary_of_everything_lists_every_term():
    block = _glossary.glossary()
    labels = [ln.label for ln in block.lines]
    assert labels == list(DESIGN_TABLE)
    assert str(block).splitlines()[-1].startswith("Next: ")
    assert len(block.as_dict()["lines"]) == 14


@pytest.mark.parametrize("spelling, key", [
    ("Target", "target"), ("  close  call ", "close call"), ("ndcg@5", "nDCG@5"), ("NDCG", "nDCG@5"),
    ("8-bit", "8-bit / 4-bit / 1-bit"), ("centred 4-bit", "8-bit / 4-bit / 1-bit"), ("1-bit", "8-bit / 4-bit / 1-bit"),
    ("merged", "merged (1 in 4)"), ("accuracy", "target"), ("practice question", "practice questions"),
])
def test_spellings_resolve(spelling, key):
    assert _glossary.meaning(spelling) == DESIGN_TABLE[key]
    assert str(_glossary.glossary(spelling)).startswith(key)


def test_unknown_term_names_the_known_ones():
    with pytest.raises(KeyError) as exc:
        _glossary.glossary("recall")
    assert "no glossary term 'recall'" in str(exc.value) and "lowest kept" in str(exc.value)


def test_note_is_one_line_and_once_per_set():
    seen: set[str] = set()
    first = _glossary.note("lowest kept", seen)
    assert first == "lowest kept: " + DESIGN_TABLE["lowest kept"]
    assert "\n" not in first
    assert _glossary.note("Lowest Kept", seen) == ""
    assert _glossary.note("lowest kept") == first, "without a seen set the note is always given"
    assert _glossary.note("close call", seen).startswith("close call: ")


def test_notes_for_follow_the_text_and_skip_what_was_seen():
    seen: set[str] = set()
    text = "Pick: 8-bit layout; a close call. The lowest kept is below the target."
    notes = _glossary.notes_for(text, seen)
    keys = [n.split(":")[0] for n in notes]
    assert keys == ["layout", "8-bit / 4-bit / 1-bit", "close call", "lowest kept", "target"] or \
        keys == ["8-bit / 4-bit / 1-bit", "layout", "close call", "lowest kept", "target"]
    assert _glossary.notes_for(text, seen) == []
    assert _glossary.notes_for("nothing to explain here", set()) == []
    assert _glossary.notes_for("ov.options() lists layouts", set())[0].startswith("layout: ")


def test_presentation_modules_import_light():
    """After ``import optivision`` (which itself loads NumPy and PIL), importing the glossary, chart and report
    adds no heavy library."""
    import os
    import subprocess

    code = """
import sys
import optivision
before = set(sys.modules)
import optivision.easy as ov
assert 'optivision.easy._glossary' not in sys.modules, 'the glossary loads lazily'
g = ov.glossary('drift')
assert 'drift' in str(g)
from optivision.easy import _chart, _report
new = set(sys.modules) - before
heavy = {'torch', 'scipy', 'transformers', 'sentence_transformers', 'colpali_engine', 'pypdfium2', 'PIL'}
bad = sorted(m for m in new if m.split('.')[0] in heavy)
assert not bad, bad
"""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
