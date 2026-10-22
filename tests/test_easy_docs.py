"""Easy mode, document analysis (``optivision.easy._docs``; FINAL_DESIGN 5.2).

PDFs are written with reportlab in ``tmp_path``: text pages, a blank page, a repeated page, a
landscape page, a page of shapes without text, an encrypted PDF, a corrupt one and an empty
one. Each file must be closed after use: on Windows an open file cannot be deleted, so the
tests delete PDFs right after reading them.
"""

from __future__ import annotations

import contextlib
import gc
import os
from pathlib import Path

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("pypdfium2")

from PIL import Image

from optivision.easy import _docs
from optivision.easy._docs import (
    MIN_TEXT_PAGES,
    FileRecord,
    UnreadableFile,
    count_pages,
    iter_pages,
    iter_source,
    list_source,
    profile,
    render_page,
    source_name,
)
from optivision.easy._types import Hit

from .easy_docs_fixtures import HEADER, make_pdf, page_records, synthetic_texts

TEXTS = synthetic_texts(4, seed=3)


def _text_lines(i: int) -> list[str]:
    return [line[:90] for line in TEXTS[i].split("\n")]


def _report_pages() -> list[dict]:
    """Seven pages: text, text, blank, repeat of page 1, landscape text, shapes only, text."""
    return [
        {"lines": _text_lines(0)},
        {"lines": _text_lines(1)},
        {},
        {"lines": _text_lines(0)},
        {"lines": _text_lines(2), "landscape": True},
        {"shapes": True},
        {"lines": _text_lines(3)},
    ]


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "my_pdfs"
    (root / "sub").mkdir(parents=True)
    make_pdf(root / "report.pdf", _report_pages())
    make_pdf(root / "sub" / "notes.pdf", [{"lines": ["A short note about the warehouse inventory levels."]}])
    make_pdf(root / "locked.pdf", [{"lines": ["secret figures"]}], password="pw")
    (root / "broken.pdf").write_bytes(b"%PDF-1.4\nthis is not a PDF at all\n")
    (root / "empty.pdf").write_bytes(b"")
    Image.new("RGB", (300, 200), "white").save(root / "sub" / "scan.png")
    (root / "readme.txt").write_text("not a document", encoding="utf-8")
    (root / ".hidden.pdf").write_bytes(b"ignored")
    (root / "Thumbs.db").write_bytes(b"ignored")
    (root / "old.ovx").mkdir()
    make_pdf(root / "old.ovx" / "inside.pdf", [{"lines": ["never listed"]}])
    return root


# ---------------------------------------------------------------------------- listing and ids


def test_list_source_is_sorted_recursive_and_ids_follow_the_contract(folder: Path):
    files, skipped = list_source(folder)
    assert [f.relpath for f in files] == ["broken.pdf", "empty.pdf", "locked.pdf", "report.pdf", "sub/notes.pdf",
                                          "sub/scan.png"]
    assert [f.doc_id for f in files][3:] == ["my_pdfs/report.pdf", "my_pdfs/sub/notes.pdf", "my_pdfs/sub/scan.png"]
    assert {f.kind for f in files} == {"pdf", "image"}
    assert all(Path(f.path).is_absolute() and f.size >= 0 and f.mtime > 0 for f in files)
    # unsupported files are listed with a reason; hidden files, Thumbs.db and *.ovx folders are not
    assert [(Path(p).name, r) for p, r in skipped] == [("readme.txt", "not a PDF or an image (.txt)")]


def test_single_file_source_takes_its_parent_folder_name(folder: Path):
    files, skipped = list_source(folder / "sub" / "notes.pdf")
    assert skipped == []
    assert [(f.relpath, f.doc_id) for f in files] == [("notes.pdf", "sub/notes.pdf")]
    assert _docs.file_record(folder / "report.pdf").doc_id == "my_pdfs/report.pdf"


def test_two_sources_with_one_name_are_told_apart(folder: Path):
    assert source_name(folder) == "my_pdfs"
    assert source_name(folder, taken={"my_pdfs"}) == "my_pdfs~2"
    assert source_name(folder, taken={"my_pdfs", "my_pdfs~2"}) == "my_pdfs~3"
    files, _ = list_source(folder, namespace="my_pdfs~2")
    assert files[3].doc_id == "my_pdfs~2/report.pdf"


def test_missing_source_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list_source(tmp_path / "nope")


# ---------------------------------------------------------------------------- counting


def test_count_pages_counts_without_rendering_and_skips_bad_files_with_reasons(folder: Path):
    seen = []
    count = count_pages(folder, progress=lambda done, total, rel: seen.append((done, total, rel)))
    assert count.n_pages == 7 + 1 + 1  # report, notes, scan
    assert (count.n_files, count.n_pdfs, count.n_images) == (3, 2, 1)
    reasons = {Path(p).name: r for p, r in count.skipped}
    assert reasons["locked.pdf"] == "encrypted (needs a password)"
    assert reasons["broken.pdf"].startswith("not a readable PDF")
    assert reasons["empty.pdf"] == "empty file"
    assert reasons["readme.txt"].startswith("not a PDF or an image")
    assert [s[0] for s in seen] == [1, 2, 3, 4, 5, 6] and all(s[1] == 6 for s in seen)
    # nothing is left open: every file can be deleted at once
    for p in folder.rglob("*.pdf"):
        os.remove(p)


# ---------------------------------------------------------------------------- pages


def test_iter_pages_records_and_signals(folder: Path):
    rec = _docs.file_record(folder / "report.pdf")
    with iter_pages(rec) as stream:
        pages = list(stream)
    records = [r for r, _ in pages]
    assert [r.id for r in records] == [f"my_pdfs/report.pdf::p{i}" for i in range(1, 8)]
    assert [r.page for r in records] == list(range(1, 8))
    assert records[0].text.split("\n")[0] == HEADER and records[0].has_text
    assert [r.blank for r in records] == [False, False, True, False, False, False, False]
    assert records[5].n_chars == 0 and not records[5].blank  # shapes only: no text layer, not blank
    assert records[3].sha256 == records[0].sha256  # the repeated page renders identically
    assert len({r.sha256 for r in records}) == 6
    assert [r.landscape for r in records] == [False, False, False, False, True, False, False]
    for r, image in pages:
        assert image.mode == "RGB" and max(image.size) <= 1536
        assert (r.width, r.height) == image.size
    # 150 dpi of A4 is 1240 x 1754; the long side is capped at 1536
    assert max(pages[0][1].size) == 1536
    os.remove(folder / "report.pdf")


def test_iter_pages_respects_dpi_and_max_side(folder: Path):
    with iter_pages(folder / "report.pdf", dpi=72, max_side=400) as stream:
        record, image = next(stream)
    assert max(image.size) == 400
    assert record.id == "my_pdfs/report.pdf::p1"
    os.remove(folder / "report.pdf")


def test_partly_read_pdf_is_released_for_deletion(folder: Path):
    """Windows keeps an open file locked; every way of stopping early must close the PDF."""
    path = folder / "report.pdf"
    with iter_pages(path) as stream:
        next(stream)
    os.remove(path)

    make_pdf(path, _report_pages())
    with contextlib.closing(iter_pages(path)) as stream:
        next(stream)
        next(stream)
    os.remove(path)

    make_pdf(path, _report_pages())
    stream = iter_pages(path)
    next(stream)
    del stream
    gc.collect()
    os.remove(path)

    make_pdf(path, _report_pages())
    pages = iter_source([_docs.file_record(path)])
    next(pages)
    pages.close()
    os.remove(path)


def test_image_file_is_one_page_without_text(folder: Path):
    rec = _docs.file_record(folder / "sub" / "scan.png")
    with iter_pages(rec) as stream:
        [(record, image)] = list(stream)
    assert record.id == "sub/scan.png::p1" and record.page == 1
    assert record.text == "" and record.n_chars == 0 and record.blank
    assert image.size == (300, 200) and record.landscape
    os.remove(folder / "sub" / "scan.png")


def test_unreadable_pdf_raises_on_its_own_and_is_skipped_in_a_source(folder: Path):
    locked = FileRecord(str(folder / "locked.pdf"), "locked.pdf", "my_pdfs/locked.pdf", 1, 1.0, "pdf")
    with pytest.raises(UnreadableFile) as err:
        next(iter_pages(locked))
    assert err.value.reason == "encrypted (needs a password)"

    files, _ = list_source(folder)
    skipped: list[tuple[str, str]] = []
    ids = [r.id for r, _ in iter_source(files, skipped=skipped)]
    assert ids == [f"my_pdfs/report.pdf::p{i}" for i in range(1, 8)] + ["my_pdfs/sub/notes.pdf::p1",
                                                                         "my_pdfs/sub/scan.png::p1"]
    reasons = {Path(p).name: r for p, r in skipped}
    assert set(reasons) == {"locked.pdf", "broken.pdf", "empty.pdf"}
    assert reasons["locked.pdf"].startswith("encrypted")
    for p in folder.rglob("*.pdf"):
        os.remove(p)


def test_a_page_that_fails_to_render_is_skipped_and_the_rest_follow(folder: Path, monkeypatch):
    real = _docs._render_pdf_page

    def flaky(doc, index, dpi):
        if index == 1:
            raise RuntimeError("bad page")
        return real(doc, index, dpi)

    monkeypatch.setattr(_docs, "_render_pdf_page", flaky)
    with iter_pages(folder / "report.pdf") as stream:
        pages = [r.page for r, _ in stream]
        reason = "page 2: could not be read (RuntimeError: bad page)"
        assert stream.skipped == [(str((folder / "report.pdf").resolve()), reason)]
    assert pages == [1, 3, 4, 5, 6, 7]
    os.remove(folder / "report.pdf")


def test_render_page_matches_the_stream_and_closes_the_file(folder: Path):
    path = folder / "report.pdf"
    with iter_pages(path) as stream:
        pages = list(stream)
    image = render_page(path, 5, max_side=1536)
    assert image.tobytes() == pages[4][1].tobytes()
    width, height = render_page(path, 1, dpi=72).size  # A4 is 595.3 x 841.9 points
    assert width in (595, 596) and height in (841, 842)
    with pytest.raises(IndexError):
        render_page(path, 8)
    with pytest.raises(IndexError):
        render_page(folder / "sub" / "scan.png", 2)
    os.remove(path)


def test_hit_image_renders_the_page(folder: Path):
    hit = Hit(id="my_pdfs/report.pdf::p2", file=str(folder / "report.pdf"), page=2, score=1.0)
    assert hit.image(dpi=72).size[1] in (841, 842)
    os.remove(folder / "report.pdf")


# ---------------------------------------------------------------------------- profile


def test_profile_of_real_pages(folder: Path):
    with iter_pages(folder / "report.pdf") as stream:
        records = [r for r, _ in stream]
    prof = profile(records, vectors_per_page=[700, 760, 740, 720])
    assert prof.n_pages == 7 and prof.n_unique == 6 and prof.n_blank == 1
    assert prof.aliases == {"my_pdfs/report.pdf::p4": "my_pdfs/report.pdf::p1"}
    assert prof.n_text_pages == 4  # p1, p2, p5, p7 (p6 is shapes only)
    assert prof.text_share == pytest.approx(4 / 5)
    assert prof.landscape_share == pytest.approx(1 / 7)
    assert _docs.line_key(HEADER) in prof.boilerplate
    assert "page # of #" in prof.boilerplate
    assert not prof.checkable and prof.reason_code == "few_text_pages"
    assert prof.median_vectors == 730.0 and prof.n_vector_sample == 4
    os.remove(folder / "report.pdf")


def test_profile_checkability_rules():
    texts = synthetic_texts(MIN_TEXT_PAGES + 10, seed=1)
    pages = page_records(texts, duplicates={5: 4}, blanks=(7,))
    prof = profile(pages)
    assert prof.checkable and prof.reason == "" and prof.n_text_pages == MIN_TEXT_PAGES + 8
    assert prof.aliases == {pages[5].id: pages[4].id}

    few = profile(page_records(texts[:MIN_TEXT_PAGES - 1]))
    assert not few.checkable and few.reason_code == "few_text_pages"

    # enough text pages, but most non-blank pages are scans without a text layer
    many = synthetic_texts(150, seed=2)
    scans = tuple(range(70, 150))
    low = profile(page_records(many, images=scans))
    assert low.n_text_pages == 70 and not low.checkable and low.reason_code == "low_text_share"
    assert low.text_share == pytest.approx(70 / 150)

    # blank pages do not count against the share
    blanks = tuple(range(70, 150))
    ok = profile(page_records(many, blanks=blanks))
    assert ok.checkable and ok.n_blank == 80


def test_boilerplate_needs_more_than_thirty_percent_of_pages():
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    texts = [f"Shared line here\nreport page {i} of ten\nonly {names[i]} here" for i in range(10)]
    for i in range(4, 10):
        texts[i] = texts[i].replace("Shared line here", f"other {names[i]} line")
    # "page 3 of ten" and "page 7 of ten" are one line once digits are folded
    assert _docs.find_boilerplate(texts) == ("report page # of ten", "shared line here")
    texts[3] = texts[3].replace("Shared line here", "gamma")
    assert _docs.find_boilerplate(texts) == ("report page # of ten",)  # 3 of 10 is not more than 30%


def test_profile_lines_are_counted_claims(folder: Path):
    with iter_pages(folder / "report.pdf") as stream:
        records = [r for r, _ in stream]
    lines = profile(records).lines()
    rendered = [line.render() for line in lines]
    assert rendered[0] == "Pages      7 pages; 4 with a text layer; 1 blank; 1 repeated (stored once)  [counted]"
    assert rendered[1].startswith("Unchecked") and "no 1-bit layouts" in rendered[1]
    os.remove(folder / "report.pdf")


def test_clean_text_normalises_whitespace_and_controls():
    assert _docs.clean_text("a\r\nb  \t c\x02\n\n  d\u00a0e \r") == "a\nb c\nd e"
