"""OCR text for pages without a text layer, so scanned folders can get practice questions (extra ``easy-ocr``).

Why: Tier A writes questions from the text layer (FINAL_DESIGN 5.3.1). A folder of scans has none, so it
would stay unchecked. Reading the scans with an OCR engine gives text that goes through the same
generator; those questions are tagged ``ocr:<family>`` and reported on their own, because OCR text can
hold reading mistakes. Like every practice question they can only make the layout safer.

Engine (licence check, 2026-10-21): RapidOCR ``rapidocr`` 3.10.0 on ONNX Runtime.

* code: Apache-2.0 (PyPI metadata and the wheel's LICENSE);
* models bundled in the wheel: PP-OCRv6 small detection and recognition and the legacy
  ``ch_ppocr_mobile_v2.0_cls`` angle classifier, Apache-2.0 per the wheel's MODEL_LICENSES.md
  (weights by Baidu / PaddleOCR, which is Apache-2.0);
* install: ``pip install rapidocr onnxruntime`` worked in a fresh Python 3.11 virtual environment on
  Windows 11 with no system package and no model download (the wheel is 27 MB and holds the models);
  its dependencies are wheels (opencv-python-headless, shapely, pyclipper, numpy, pillow).
* speed: 5.5-9.3 s per A4 page at 150 dpi on one laptop CPU (10 text lines read). Slow on a laptop,
  so the default sample is 200 pages.

Nothing leaves the machine: the engine runs locally on the rendered page.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ._claim import Block, claim, duration
from ._probes_common import (
    CONT,
    ProbeBatch,
    add_counts,
    get_image,
    has_text_layer,
    is_blank,
    make_probe,
    page_id,
    select_pages,
)
from ._types import MissingExtra

#: engine facts for the build record and for ``report()``
ENGINE_INFO: dict[str, str] = {
    "name": "rapidocr",
    "package": "rapidocr 3.10.0 on onnxruntime",
    "licence": "Apache-2.0",
    "models": "PP-OCRv6 small detection and recognition, ch_ppocr_mobile_v2.0 angle classifier",
    "models_licence": "Apache-2.0 (wheel MODEL_LICENSES.md)",
    "checked": "2026-10-21",
    "source": "https://github.com/RapidAI/RapidOCR",
}

#: seconds per A4 page at 150 dpi on a laptop CPU: one test gave 5.5-9.3 s; the range below is wider
SECONDS_PER_PAGE = (4.0, 10.0)
_TIME_BASIS = "one test on a laptop CPU, 5.5-9.3 s per A4 page at 150 dpi; your machine will differ"


def install_line() -> str:
    return 'pip install "optivision-rag[easy-ocr]"'


def _missing(exc: BaseException) -> MissingExtra:
    return MissingExtra(f"OCR needs the easy-ocr extra ({exc}). Install it with: {install_line()}")


@dataclass(frozen=True)
class OcrResult:
    """The text an engine read from one page.

    ``confidence`` is the engine's own line scores, averaged with each line weighted by its length;
    it says how sure the engine is of its reading, not how right it is. ``text`` joins the kept lines
    with newlines, as a PDF text layer does.
    """

    text: str
    confidence: float
    n_lines: int
    n_chars: int


class OcrEngine(Protocol):
    name: str

    def recognise(self, image: Any) -> OcrResult: ...


class RapidOcrEngine:
    """RapidOCR on ONNX Runtime, loaded when it is used.

    ``min_line_score`` drops lines the engine itself doubts: in a test, lines with numbers in brackets
    came back as noise at scores of 0.61-0.77 while clean lines scored 0.99 or more.
    """

    name = "rapidocr"

    def __init__(self, *, min_line_score: float = 0.8) -> None:
        self.min_line_score = float(min_line_score)
        self._engine: Any = None

    def _load(self) -> Any:
        if self._engine is None:
            params = {"Global.log_level": "critical"}
            try:
                from rapidocr import RapidOCR

                self._engine = RapidOCR(params=params)
            except ImportError as exc:
                try:  # the older package name, if that is what is installed
                    from rapidocr_onnxruntime import RapidOCR as LegacyRapidOCR

                    self._engine = LegacyRapidOCR()
                except ImportError:
                    raise _missing(exc) from exc
        return self._engine

    def recognise(self, image: Any) -> OcrResult:
        import numpy as np

        engine = self._load()
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        out = engine(bgr)
        if isinstance(out, tuple):  # rapidocr_onnxruntime: (result, timings), result = [[box, text, score], ...]
            rows = out[0] or []
            texts = [r[1] for r in rows]
            scores = [float(r[2]) for r in rows]
        else:
            texts = list(getattr(out, "txts", None) or ())
            scores = [float(s) for s in (getattr(out, "scores", None) or ())]
        return _assemble(texts, scores, self.min_line_score)


def _assemble(texts: Sequence[str | None], scores: Sequence[float], min_line_score: float) -> OcrResult:
    kept = [(str(t).strip(), s) for t, s in zip(texts, scores, strict=False)
            if t is not None and str(t).strip() and s >= min_line_score]
    if not kept:
        return OcrResult("", 0.0, 0, 0)
    chars = sum(len(t) for t, _ in kept)
    conf = sum(len(t) * s for t, s in kept) / max(chars, 1)
    return OcrResult("\n".join(t for t, _ in kept), float(conf), len(kept), chars)


def load_engine(name: str = "rapidocr", **kwargs: Any) -> OcrEngine:
    """The named engine. Only ``"rapidocr"`` is built in; pass your own object with ``recognise(image)`` otherwise."""
    if name != "rapidocr":
        raise ValueError(f"unknown OCR engine {name!r}; built in: 'rapidocr'. "
                         "Any object with recognise(image) -> OcrResult works.")
    engine = RapidOcrEngine(**kwargs)
    engine._load()  # fail here, with the install line, rather than midway through a run
    return engine


# ---------------------------------------------------------------------------- reading a folder


def needs_ocr(page: Any) -> bool:
    """A page with no text layer that is not blank."""
    return not has_text_layer(page) and not is_blank(page)


def estimate(n_pages: int) -> tuple[float, float]:
    """Seconds to read ``n_pages`` pages (low, high); an estimate, shown with its basis."""
    lo, hi = SECONDS_PER_PAGE
    return lo * n_pages, hi * n_pages


@dataclass
class OcrRun:
    """What reading the scanned pages gave. ``texts`` holds only pages that passed the filters."""

    engine: str
    texts: dict[str, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    n_candidates: int = 0
    n_attempted: int = 0
    seconds: float = 0.0

    @property
    def n_read(self) -> int:
        return len(self.confidence)


def ocr_pages(pages: Sequence[Any], *, engine: OcrEngine | None = None, images: Any = None, max_pages: int = 200,
              seed: int = 0, min_confidence: float = 0.75, min_words: int = 8,
              progress: Callable[[int, int], None] | None = None) -> OcrRun:
    """Read up to ``max_pages`` seeded pages that have no text layer.

    A page is kept when the engine read at least ``min_words`` words at a confidence of at least
    ``min_confidence``; the others are listed in ``skipped`` with the reason. ``images`` is a mapping
    (page id -> image), a callable (page -> image) or None to re-render the page from its file.
    """
    candidates = [p for p in pages if needs_ocr(p)]
    chosen = select_pages(candidates, max_pages=max_pages, seed=seed)
    if not chosen:  # nothing to read, so no engine is loaded and no extra is needed
        return OcrRun(engine=getattr(engine, "name", "rapidocr"), n_candidates=len(candidates))
    eng = engine if engine is not None else load_engine()
    run = OcrRun(engine=getattr(eng, "name", type(eng).__name__), n_candidates=len(candidates),
                 n_attempted=len(chosen))
    t0 = time.perf_counter()
    for i, page in enumerate(chosen):
        if progress is not None:
            progress(i, len(chosen))
        pid = page_id(page)
        try:
            result = eng.recognise(get_image(page, images))
        except MissingExtra:
            raise
        except Exception as exc:  # one bad page must not stop the run
            run.skipped[pid] = f"error: {type(exc).__name__}"
            continue
        run.confidence[pid] = result.confidence
        if len(result.text.split()) < min_words:
            run.skipped[pid] = "too little text"
        elif result.confidence < min_confidence:
            run.skipped[pid] = "low confidence"
        else:
            run.texts[pid] = result.text
    if progress is not None:
        progress(len(chosen), len(chosen))
    run.seconds = time.perf_counter() - t0
    return run


def _with_text(page: Any, text: str) -> Any:
    if dataclasses.is_dataclass(page) and not isinstance(page, type):
        fields = {f.name for f in dataclasses.fields(page)}
        changes: dict[str, Any] = {"text": text}
        if "n_chars" in fields:
            changes["n_chars"] = len(text)
        return dataclasses.replace(page, **changes)
    clone = type("OcrPage", (), dict(vars(page)))()
    clone.text, clone.n_chars = text, len(text)
    return clone


def make_ocr_probes(pages: Sequence[Any], *, engine: OcrEngine | None = None, images: Any = None,
                    run: OcrRun | None = None, seed: int = 0, max_pages: int = 200, per_page: int = 3,
                    min_confidence: float = 0.75, min_words: int = 8,
                    progress: Callable[[int, int], None] | None = None) -> ProbeBatch:
    """Practice questions for scanned pages: OCR text through the Tier A generator, tagged ``ocr:<family>``.

    Pass ``run`` (an earlier :func:`ocr_pages` result) to reuse OCR text you already have; the same text
    is useful to Tiers B and C as the page text for their overlap filter and neighbour search.
    The generator sees the whole folder, so a scanned page's relevant set also names text pages that
    hold its words; only the questions whose page was read by OCR are kept.
    """
    from ._probes import make_probes

    if run is None:
        run = ocr_pages(pages, engine=engine, images=images, max_pages=max_pages, seed=seed,
                        min_confidence=min_confidence, min_words=min_words, progress=progress)
    combined = [_with_text(p, run.texts[page_id(p)]) if page_id(p) in run.texts else p for p in pages]
    made = make_probes(combined, seed=seed, max_pages=len(combined), per_page=per_page) if run.texts else []
    probes = [
        make_probe(id=f"ocr-{p.id}", text=p.text, family=f"ocr:{p.family}", page_id=p.page_id, relevant=p.relevant)
        for p in made
        if p.page_id in run.texts
    ]
    skipped: dict[str, int] = {}
    for reason in run.skipped.values():
        key = reason.split(":")[0]
        skipped[key] = skipped.get(key, 0) + 1
    batch = ProbeBatch(
        tier="ocr", probes=probes, n_pages=run.n_read, rejected=skipped,
        usage={"seconds": round(run.seconds, 1), "n_candidates": run.n_candidates}, texts=dict(run.texts),
        config={"engine": run.engine, "engine_info": dict(ENGINE_INFO), "seed": seed, "min_confidence": min_confidence,
                "min_words": min_words, "per_page": per_page},
    )
    if run.n_candidates > run.n_attempted:
        batch.notes.append("only a sample of the scanned pages was read; raise max_pages to read more")
    batch.report = report_block(run, batch)
    return batch


def report_block(run: OcrRun, batch: ProbeBatch) -> Block:
    """The OCR tier on its own lines, apart from the text-layer questions."""
    n_pages_with_probes = len({p.page_id for p in batch.probes})
    block = Block([], title=f"Practice questions from scanned pages (OCR: {run.engine})")
    block.add("Scans", "{0} pages without a text layer were found; {1} were read", claim(run.n_candidates, "counted"),
              claim(run.n_read, "counted"))
    if run.skipped:
        add_counts(block, "Left out", batch.rejected, lead="pages were left out")
    block.add("Questions", "{0} questions made from OCR text on {1} pages", claim(len(batch.probes), "counted"),
              claim(n_pages_with_probes, "counted"))
    block.add(CONT, "OCR text can hold reading mistakes, so these questions are reported apart from the "
                    "text-layer ones and tagged ocr: in their family.")
    block.next = "give them to the practice check with the text-layer questions; they can only make the layout safer"
    return block


def time_line(n_pages: int) -> Block:
    """The time to read ``n_pages`` pages, as an estimate with its basis."""
    lo, hi = estimate(n_pages)
    block = Block([], title="Reading scanned pages with OCR")
    block.add("Pages", "{0} pages to read", claim(n_pages, "counted"))
    block.add("Time", "{0} on a laptop CPU", claim(duration(lo, hi), "estimate", _TIME_BASIS))
    block.next = "run make_ocr_probes(pages) or lower max_pages to read fewer pages"
    return block


__all__ = [
    "ENGINE_INFO",
    "OcrEngine",
    "OcrResult",
    "OcrRun",
    "RapidOcrEngine",
    "estimate",
    "install_line",
    "load_engine",
    "make_ocr_probes",
    "needs_ocr",
    "ocr_pages",
    "report_block",
    "time_line",
]

