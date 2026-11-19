"""Document analysis for easy mode: list, count, render and profile a folder of PDFs and images
(FINAL_DESIGN 5.2; docs/easy/CONTRACT.md, ``_docs.py``).

Everything here runs on the CPU with pypdfium2 and Pillow, and nothing here encodes, so
``options(pages="folder/")`` can plan without a GPU or a model download.

Ids (CONTRACT "Ids and sources"). A folder source ``.../my_pdfs/`` gives every file the
document id ``"my_pdfs/<path relative to the folder, posix>"``; a single file
``.../x/report.pdf`` gives ``"x/report.pdf"``. Page ids are ``PageRef(doc_id, page).page_id``,
``f"{doc_id}::p{page}"`` with ``page`` 1-based; an image file is one page. When two sources
share a folder name, the build passes ``namespace="name~2"`` (see :func:`source_name`).

Files are read one at a time inside ``try``/``except``: an encrypted, corrupt or unreadable
file is listed in ``skipped`` with a reason and never stops a count or a build. A PDF is
closed in ``finally`` and :func:`iter_pages` returns a :class:`PageStream` that closes the
file when the stream is closed, garbage collected or used as a context manager, so a
partly read PDF is never left locked on Windows (``contextlib.closing`` works too).

What a page's signals drive (FINAL_DESIGN 5.2): checkability, the shape gate and the
practice-question strata. There is no document-type classifier.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: rendering defaults, as the encoders see pages (FINAL_DESIGN 5.2)
DPI = 150
MAX_SIDE = 1536
#: fewer text-layer characters than this means "no text layer"
TEXT_CHARS = 20
#: a normalised line on more than this share of the text pages is a header or footer
BOILERPLATE_SHARE = 0.3
#: a folder is unchecked below this many pages with a text layer (FINAL_DESIGN 5.1.2 step 4)
MIN_TEXT_PAGES = 60
#: ... or when under this share of its non-blank pages have a text layer
MIN_TEXT_SHARE = 0.5

PDF_SUFFIXES = (".pdf",)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")
#: files a folder commonly holds that are not documents; never listed as skipped
_JUNK = {"thumbs.db", "desktop.ini", ".ds_store"}
#: folders this package writes next to (or inside) a source; never read as documents
_OWN_DIR_SUFFIXES = (".ovx", ".ovx.work")

Progress = Callable[[int, int, str], Any]


class UnreadableFile(OSError):
    """A whole file could not be opened or holds no page; ``reason`` is the plain-words cause."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


# ---------------------------------------------------------------------------- records


@dataclass(frozen=True)
class FileRecord:
    """One document file of a source. ``path`` is absolute; ``relpath`` is posix and relative
    to the source folder (the file name for a single-file source); ``kind`` is ``"pdf"`` or
    ``"image"``. ``size`` and ``mtime`` are what sync compares before hashing."""

    path: str
    relpath: str
    doc_id: str
    size: int
    mtime: float
    kind: str


@dataclass(frozen=True)
class PageRecord:
    """One rendered page and its signals.

    ``text`` is the PDF text layer with whitespace normalised (``""`` for an image file),
    ``n_chars`` its count of non-whitespace characters. ``blank`` holds when the page has no
    text layer and :func:`optivision.demo.looks_blank` finds almost no ink (a page with a text
    layer is never called blank). ``sha256`` is the digest of ``b"<width>x<height>\\n"`` plus
    the rendered RGB bytes; the size prefix keeps a blank portrait page from matching a blank
    landscape page of the same pixel count. ``landscape`` means wider than tall.
    """

    id: str
    doc_id: str
    file: str
    relpath: str
    page: int
    text: str
    n_chars: int
    blank: bool
    sha256: str
    landscape: bool
    width: int = 0
    height: int = 0

    @property
    def has_text(self) -> bool:
        return self.n_chars >= TEXT_CHARS


@dataclass(frozen=True)
class PageCount:
    """What :func:`count_pages` found without rendering: pages in readable files, the number
    of readable files (``n_pdfs`` + ``n_images``), and ``(path, reason)`` for every file left
    out."""

    n_pages: int
    n_files: int
    skipped: tuple[tuple[str, str], ...] = ()
    n_pdfs: int = 0
    n_images: int = 0


@dataclass(frozen=True)
class FolderProfile:
    """The signals of a set of pages (FINAL_DESIGN 5.2).

    Counts: ``n_pages`` all pages; ``n_unique`` pages left once identical renders are folded
    into their first copy; ``n_blank`` blank pages; ``n_text_pages`` unique, non-blank pages
    with a text layer. ``aliases`` maps each repeated page id to the id of its first copy
    (``{id: canonical id}``; canonical pages are not keys). ``boilerplate`` holds the
    normalised lines (lower case, digits as ``#``) that appear on more than 30% of the text
    pages: headers, footers, page numbers. ``landscape_share`` is over all pages;
    ``text_share`` is ``n_text_pages`` over the unique non-blank pages.

    ``checkable`` is False when there are fewer than 60 text pages or a text layer on under
    half the unique non-blank pages; ``reason`` then says why in words without numbers (so
    a template can carry the counts as claims) and ``reason_code`` is ``"few_text_pages"`` or
    ``"low_text_share"``. A checkable folder still needs 60 hard practice questions
    (``_probes.HardSet.checked``) before it counts as checked.
    ``median_vectors`` is the median of the ``vectors_per_page`` sample given (the shape gate),
    or None.
    """

    n_pages: int
    n_unique: int
    n_text_pages: int
    n_blank: int
    aliases: dict[str, str]
    boilerplate: tuple[str, ...]
    landscape_share: float
    text_share: float
    checkable: bool
    reason: str = ""
    reason_code: str = ""
    median_vectors: float | None = None
    n_vector_sample: int = 0

    def lines(self) -> list[Any]:
        """The profile as ``_claim.Line`` objects, every number counted."""
        from ._claim import Line, claim

        out = [Line("Pages", "{0} pages; {1} with a text layer; {2} blank; {3} repeated (stored once)",
                    claim(self.n_pages, "counted"), claim(self.n_text_pages, "counted"),
                    claim(self.n_blank, "counted"), claim(len(self.aliases), "counted"))]
        if not self.checkable:
            out.append(Line("Unchecked", "{0}: no practice questions, so no 1-bit layouts", self.reason))
        return out


# ---------------------------------------------------------------------------- sources


def _kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return None


def _base_name(source: Path) -> str:
    resolved = source.resolve()
    folder = resolved if resolved.is_dir() else resolved.parent
    return folder.name or (folder.drive.rstrip(":\\/") or "files")


def source_name(source: str | Path, taken: Iterable[str] = ()) -> str:
    """The id namespace of a source: its folder name (a single file: its parent folder's
    name), told apart from names already ``taken`` by ``"~2"``, ``"~3"``, ..."""
    base = _base_name(Path(source))
    used = set(taken)
    if base not in used:
        return base
    n = 2
    while f"{base}~{n}" in used:
        n += 1
    return f"{base}~{n}"


def _record(path: Path, relpath: str, namespace: str, kind: str) -> FileRecord:
    st = path.stat()
    return FileRecord(path=str(path.resolve()), relpath=relpath, doc_id=f"{namespace}/{relpath}",
                      size=int(st.st_size), mtime=float(st.st_mtime), kind=kind)


def list_source(source: str | Path, *, namespace: str | None = None
                ) -> tuple[list[FileRecord], list[tuple[str, str]]]:
    """Every PDF and image file of ``source`` (a folder, searched recursively, or one file),
    sorted by relative path, and ``(path, reason)`` for the files left out.

    Hidden files and folders, ``Thumbs.db``/``desktop.ini``, and this package's own ``*.ovx``
    and ``*.ovx.work`` folders are passed over silently; any other file that is not a PDF or
    an image is listed with a reason. Symbolic links to folders are not followed. Raises
    ``FileNotFoundError`` if ``source`` does not exist.
    """
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(f"{source} does not exist")
    ns = namespace if namespace is not None else source_name(root)
    files: list[FileRecord] = []
    skipped: list[tuple[str, str]] = []

    def consider(path: Path, relpath: str) -> None:
        kind = _kind(path)
        if kind is None:
            skipped.append((str(path), f"not a PDF or an image ({path.suffix.lower() or 'no extension'})"))
            return
        try:
            files.append(_record(path, relpath, ns, kind))
        except OSError as exc:
            skipped.append((str(path), f"cannot be read ({exc.strerror or type(exc).__name__})"))

    if root.is_file():
        consider(root, root.name)
        return files, skipped

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and not d.lower().endswith(_OWN_DIR_SUFFIXES))
        for name in sorted(filenames):
            if name.startswith(".") or name.lower() in _JUNK:
                continue
            path = Path(dirpath) / name
            consider(path, path.relative_to(root).as_posix())
    files.sort(key=lambda f: f.relpath)
    return files, skipped


def file_record(path: str | Path, *, namespace: str | None = None) -> FileRecord:
    """The :class:`FileRecord` of one file, with the single-file id rule."""
    files, skipped = list_source(path, namespace=namespace)
    if not files:
        raise UnreadableFile(str(path), skipped[0][1] if skipped else "not a PDF or an image")
    return files[0]


# ---------------------------------------------------------------------------- reasons


def _reason(exc: BaseException) -> str:
    """A plain-words reason for a file or page that could not be read."""
    if isinstance(exc, UnreadableFile):
        return exc.reason
    text = str(exc)
    low = text.lower()
    name = type(exc).__name__
    if name == "PdfiumError" or "pdfium" in low:
        if "password" in low:
            return "encrypted (needs a password)"
        detail = text.split("(PDFium:", 1)[-1].rstrip(").").strip() if "(PDFium:" in text else text
        return f"not a readable PDF ({detail.lower() or 'unknown error'})"
    if name == "UnidentifiedImageError":
        return "not a readable image"
    if isinstance(exc, MemoryError):
        return "too large to render in memory"
    if isinstance(exc, OSError):
        return f"cannot be read ({exc.strerror or text or name})"
    return f"could not be read ({name}: {text})" if text else f"could not be read ({name})"


def _open_pdf(path: str) -> Any:
    import pypdfium2 as pdfium

    if os.path.getsize(path) == 0:
        raise UnreadableFile(path, "empty file")
    try:
        return pdfium.PdfDocument(path)
    except Exception as exc:  # pdfium raises PdfiumError (a RuntimeError) for every load failure
        raise UnreadableFile(path, _reason(exc)) from exc


def _count_one(rec: FileRecord) -> int:
    if rec.kind == "pdf":
        doc = _open_pdf(rec.path)
        try:
            n = len(doc)
        finally:
            doc.close()
        if n == 0:
            raise UnreadableFile(rec.path, "the PDF has no pages")
        return n
    from PIL import Image

    if os.path.getsize(rec.path) == 0:
        raise UnreadableFile(rec.path, "empty file")
    with Image.open(rec.path) as im:  # reads the header only
        if im.size[0] < 1 or im.size[1] < 1:
            raise UnreadableFile(rec.path, "the image has no pixels")
    return 1


def count_pages(source: str | Path, progress: Progress | None = None, *,
                namespace: str | None = None) -> PageCount:
    """Pages in ``source`` without rendering any: each PDF is opened, counted and closed, each
    image's header is read. A file that fails is added to ``skipped`` with its reason.

    ``progress``, if given, is called as ``progress(files_done, files_total, relpath)`` after
    each file.
    """
    files, skipped = list_source(source, namespace=namespace)
    n_pages = n_pdfs = n_images = 0
    for i, rec in enumerate(files, 1):
        try:
            n = _count_one(rec)
        except Exception as exc:  # noqa: BLE001 - one bad file never stops a count
            skipped.append((rec.path, _reason(exc)))
        else:
            n_pages += n
            if rec.kind == "pdf":
                n_pdfs += 1
            else:
                n_images += 1
        if progress is not None:
            progress(i, len(files), rec.relpath)
    return PageCount(n_pages=n_pages, n_files=n_pdfs + n_images, skipped=tuple(skipped),
                     n_pdfs=n_pdfs, n_images=n_images)


# ---------------------------------------------------------------------------- rendering

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\ufffe\uffff]")
_SPACES = re.compile(r"[ \t\u00a0\u2000-\u200b\u3000]+")


def clean_text(raw: str) -> str:
    """Text-layer text with line ends unified, control characters removed and runs of
    whitespace collapsed; empty lines dropped."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _n_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def _render_pdf_page(doc: Any, index: int, dpi: int) -> tuple[str, Any]:
    page = doc[index]
    try:
        try:
            textpage = page.get_textpage()
            try:
                raw = textpage.get_text_bounded()
            finally:
                textpage.close()
        except Exception:  # noqa: BLE001 - a broken text layer leaves the page usable as an image
            raw = ""
        bitmap = page.render(scale=dpi / 72.0)
        try:
            image = bitmap.to_pil().convert("RGB")  # a copy: the bitmap's buffer is freed below
        finally:
            bitmap.close()
    finally:
        page.close()
    return clean_text(raw), image


def _open_image(path: str) -> Any:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im) or im
        return im.convert("RGB")  # loads the pixels, so the file can close


def _page_record(rec: FileRecord, page: int, text: str, image: Any) -> PageRecord:
    from ..types import PageRef

    width, height = image.size
    n_chars = _n_chars(text)
    blank = False
    if n_chars < TEXT_CHARS:
        from ..demo import looks_blank

        blank = bool(looks_blank(image))
    digest = hashlib.sha256(f"{width}x{height}\n".encode("ascii"))
    digest.update(image.tobytes())
    return PageRecord(id=PageRef(rec.doc_id, page).page_id, doc_id=rec.doc_id, file=rec.path,
                      relpath=rec.relpath, page=page, text=text, n_chars=n_chars, blank=blank,
                      sha256=digest.hexdigest(), landscape=width > height, width=width, height=height)


class PageStream:
    """The pages of one file as ``(PageRecord, PIL.Image)`` pairs, yielded one at a time.

    Iterating opens the file; :meth:`close` (also on ``with`` exit, ``contextlib.closing`` or
    garbage collection) closes it even when iteration stopped early. A file that cannot be
    opened raises :class:`UnreadableFile` on the first ``next()``. A single page that fails to
    render is left out and recorded in :attr:`skipped` as ``(path, "page N: reason")``, so one
    bad page does not drop the rest of the file.
    """

    def __init__(self, file: FileRecord, dpi: int = DPI, max_side: int = MAX_SIDE) -> None:
        self.file = file
        self.dpi = int(dpi)
        self.max_side = int(max_side)
        self.skipped: list[tuple[str, str]] = []
        self._gen = self._pages()

    def _pages(self) -> Iterator[tuple[PageRecord, Any]]:
        from ..ingest import downscale

        rec = self.file
        if rec.kind == "image":
            try:
                image = downscale(_open_image(rec.path), self.max_side)
            except Exception as exc:
                raise UnreadableFile(rec.path, _reason(exc)) from exc
            yield _page_record(rec, 1, "", image), image
            return
        doc = _open_pdf(rec.path)
        try:
            n = len(doc)
            if n == 0:
                raise UnreadableFile(rec.path, "the PDF has no pages")
            for i in range(n):
                try:
                    text, image = _render_pdf_page(doc, i, self.dpi)
                    image = downscale(image, self.max_side)
                    record = _page_record(rec, i + 1, text, image)
                except Exception as exc:  # noqa: BLE001 - one bad page never stops the file
                    self.skipped.append((rec.path, f"page {i + 1}: {_reason(exc)}"))
                    continue
                yield record, image
        finally:
            doc.close()

    def __iter__(self) -> PageStream:
        return self

    def __next__(self) -> tuple[PageRecord, Any]:
        return next(self._gen)

    def close(self) -> None:
        self._gen.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_pages(file: FileRecord | str | Path, *, dpi: int = DPI, max_side: int = MAX_SIDE) -> PageStream:
    """The pages of one file at ``dpi`` with the long side capped at ``max_side`` (via
    ``optivision.ingest.downscale``), as a :class:`PageStream`. A path is turned into a
    :class:`FileRecord` with the single-file id rule."""
    rec = file if isinstance(file, FileRecord) else file_record(file)
    return PageStream(rec, dpi=dpi, max_side=max_side)


def iter_source(files: Iterable[FileRecord], *, skipped: list[tuple[str, str]] | None = None,
                dpi: int = DPI, max_side: int = MAX_SIDE,
                progress: Progress | None = None) -> Iterator[tuple[PageRecord, Any]]:
    """The pages of many files in order. Per file, errors are caught: an unreadable file and
    any page that failed are appended to ``skipped`` (when given) and the next file follows.
    Every file is closed before the next is opened. ``progress(files_done, files_total,
    relpath)`` is called after each file."""
    files = list(files)
    sink = skipped if skipped is not None else []
    for i, rec in enumerate(files, 1):
        stream = PageStream(rec, dpi=dpi, max_side=max_side)
        try:
            yield from stream
        except UnreadableFile as exc:
            sink.append((rec.path, exc.reason))
        except Exception as exc:  # noqa: BLE001 - anything else wrong with this file skips it
            sink.append((rec.path, _reason(exc)))
        finally:
            stream.close()
            sink.extend(stream.skipped)
        if progress is not None:
            progress(i, len(files), rec.relpath)


def render_page(path: str | Path, page: int, dpi: int = DPI, *, max_side: int | None = None) -> Any:
    """One page of a PDF (``page`` 1-based) or an image file (``page`` 1) as an RGB
    ``PIL.Image.Image``. ``max_side`` caps the long edge (None: the full ``dpi`` render).
    The file is closed before returning."""
    p = str(path)
    page = int(page)
    kind = _kind(Path(p))
    if kind is None:
        raise UnreadableFile(p, "not a PDF or an image")
    if kind == "image":
        if page != 1:
            raise IndexError(f"{p} is an image file, which has page 1 only (asked for page {page})")
        image = _open_image(p)
    else:
        doc = _open_pdf(p)
        try:
            n = len(doc)
            if not 1 <= page <= n:
                raise IndexError(f"{p} has {n} pages (asked for page {page})")
            _, image = _render_pdf_page(doc, page - 1, int(dpi))
        finally:
            doc.close()
    if max_side is not None:
        from ..ingest import downscale

        image = downscale(image, int(max_side))
    return image


# ---------------------------------------------------------------------------- profile

_DIGITS = re.compile(r"\d+")
_LETTER = re.compile(r"[^\W\d_]")


def line_key(line: str) -> str:
    """A text line normalised for header/footer matching: lower case, whitespace collapsed,
    each run of digits as ``#`` (so "Page 3 of 10" matches "Page 4 of 10")."""
    return _DIGITS.sub("#", " ".join(line.split()).casefold()).strip(" .,;:-|")


def _line_keys(text: str) -> set[str]:
    keys = set()
    for line in text.split("\n"):
        key = line_key(line)
        if len(key) >= 3 and _LETTER.search(key):
            keys.add(key)
    return keys


def find_aliases(pages: Sequence[PageRecord]) -> dict[str, str]:
    """``{id: id of its first identical copy}`` for every page whose render repeats an earlier
    page's (same sha256), in the order given."""
    first: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for p in pages:
        if p.sha256 in first:
            aliases[p.id] = first[p.sha256]
        else:
            first[p.sha256] = p.id
    return aliases


def find_boilerplate(texts: Sequence[str], share: float = BOILERPLATE_SHARE) -> tuple[str, ...]:
    """Normalised lines found on more than ``share`` of the given page texts (and on at least
    two of them), sorted."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_line_keys(text))
    limit = share * len(texts)
    return tuple(sorted(k for k, c in counts.items() if c >= 2 and c > limit))


def profile(pages: Sequence[PageRecord], vectors_per_page: Sequence[int] | None = None) -> FolderProfile:
    """The :class:`FolderProfile` of ``pages`` (in build order; the first copy of a repeated
    page is the canonical one). ``vectors_per_page`` is the sample of vector counts per page
    the shape gate reads (FINAL_DESIGN 5.5, G-shape); only its median is kept."""
    pages = list(pages)
    aliases = find_aliases(pages)
    unique = [p for p in pages if p.id not in aliases]
    nonblank = [p for p in unique if not p.blank]
    text_pages = [p for p in nonblank if p.has_text]
    n_text = len(text_pages)
    text_share = n_text / len(nonblank) if nonblank else 0.0
    if n_text < MIN_TEXT_PAGES:
        checkable, code, reason = False, "few_text_pages", "too few pages with a text layer for practice questions"
    elif text_share < MIN_TEXT_SHARE:
        checkable, code, reason = False, "low_text_share", "a text layer on under half of the pages"
    else:
        checkable, code, reason = True, "", ""
    median = None
    n_sample = 0
    if vectors_per_page is not None:
        counts = sorted(int(v) for v in vectors_per_page)
        n_sample = len(counts)
        if counts:
            mid = n_sample // 2
            median = float(counts[mid]) if n_sample % 2 else (counts[mid - 1] + counts[mid]) / 2.0
    return FolderProfile(
        n_pages=len(pages),
        n_unique=len(unique),
        n_text_pages=n_text,
        n_blank=sum(1 for p in pages if p.blank),
        aliases=aliases,
        boilerplate=find_boilerplate([p.text for p in text_pages]),
        landscape_share=(sum(1 for p in pages if p.landscape) / len(pages)) if pages else 0.0,
        text_share=text_share,
        checkable=checkable,
        reason=reason,
        reason_code=code,
        median_vectors=median,
        n_vector_sample=n_sample,
    )


__all__ = [
    "BOILERPLATE_SHARE",
    "DPI",
    "IMAGE_SUFFIXES",
    "MAX_SIDE",
    "MIN_TEXT_PAGES",
    "MIN_TEXT_SHARE",
    "PDF_SUFFIXES",
    "TEXT_CHARS",
    "FileRecord",
    "FolderProfile",
    "PageCount",
    "PageRecord",
    "PageStream",
    "UnreadableFile",
    "clean_text",
    "count_pages",
    "file_record",
    "find_aliases",
    "find_boilerplate",
    "iter_pages",
    "iter_source",
    "line_key",
    "list_source",
    "profile",
    "render_page",
    "source_name",
]
