"""Evaluation corpora.

Two sources, deliberately:

``synthetic``
    Generated locally with reportlab. Realistic office paperwork — invoices,
    lab reports, memos, land records — with the whitespace profile that motivates
    this project (wide margins, short tables, half-empty pages). Because we draw
    every glyph, we know each word's box, which gives the offline encoder real
    content to work with and gives us ground-truth queries for free. Runs with
    no network and no GPU, so the pipeline is always testable.

``vidore``
    The real benchmark used by the ColPali paper. Needs ``datasets`` and a
    download. Use this for any number that goes in the report.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------- content

DOC_KINDS = [
    ("invoice", "TAX INVOICE", ["Invoice No", "Order Ref", "GSTIN", "Due Date", "Amount"]),
    ("lab_report", "PATHOLOGY REPORT", ["Patient ID", "Referred By", "Sample", "Collected"]),
    ("land_record", "RECORD OF RIGHTS", ["Survey No", "Village", "Extent", "Pattadar"]),
    ("memo", "OFFICE MEMORANDUM", ["File No", "Subject", "Issued On", "Circulated To"]),
    ("loan_form", "LOAN APPLICATION", ["Application No", "Branch", "Scheme", "Sanctioned"]),
    ("transcript", "STATEMENT OF MARKS", ["Roll No", "Programme", "Semester", "SGPA"]),
]

SUBJECTS = [
    "quarterly maintenance of transformer bay",
    "haemoglobin and platelet count review",
    "mutation of agricultural land parcel",
    "revision of contract labour wage rates",
    "collateral valuation for working capital",
    "reassessment of internal examination marks",
    "procurement of laboratory reagents",
    "digitisation of legacy case files",
    "annual fire safety audit compliance",
    "renewal of vehicle insurance policy",
    "settlement of pending travel claims",
    "installation of rooftop solar panels",
]

BODY_WORDS = ["the", "said", "amount", "shall", "be", "payable", "within", "thirty", "days", "from", "the", "date", "of", "receipt", "subject", "to", "verification", "by", "the", "competent", "authority", "as", "per", "the", "terms", "recorded", "in", "the", "annexure", "attached", "herewith", "all", "corrections", "must", "be", "attested", "and", "countersigned", "before", "submission", "to", "the", "concerned", "section", "for", "further", "necessary", "action", "and", "record"]

CITIES = ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam", "Nalgonda"]


@dataclass
class CorpusSpec:
    n_docs: int = 20
    pages_per_doc: int = 2
    seed: int = 7
    page_size: str = "A4"
    # Font multiplier applied to the unique code's glyphs only (the value on
    # the first field line, and the footer copy). The field label, every other
    # line, the line pitch and the RNG sequence are untouched, so a corpus at
    # code_scale 3.0 or 0.4 has byte-identical pages except for those glyphs.
    # A value above 1 is drawn right-aligned in the blank right half of the
    # field block instead of in the text flow: a taller glyph in the flow would
    # push every line below it down by ~2 patch rows, which re-encodes half the
    # page and is a layout change, not a legibility change (measured: cos 0.35
    # to 0.55 between the 1x and shifted-3x tokens of the moved rows). This is
    # the manipulation that tests whether the one-bit codec's cost follows the
    # legibility of the discriminative evidence (docs/REVIEW-2026-08-21.md).
    code_scale: float = 1.0


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_synthetic_corpus(out_dir: str | Path, spec: CorpusSpec | None = None) -> dict:
    """Render a corpus of PDFs plus queries.json and layout.json.

    Returns a manifest dict; also written to ``out_dir/manifest.json``.
    """
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.pdfgen import canvas

    spec = spec or CorpusSpec()
    out_dir = Path(out_dir)
    pdf_dir = out_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    page_w, page_h = A4 if spec.page_size.upper() == "A4" else letter
    rng = _rng(spec.seed)

    layout: dict[str, list] = {}
    page_index: list[dict] = []

    for d in range(spec.n_docs):
        kind, heading, fields = DOC_KINDS[d % len(DOC_KINDS)]
        doc_id = f"{kind}_{d:03d}"
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))

        for p in range(spec.pages_per_doc):
            # Page ids must match what ``ingest.iter_pages`` derives when the
            # corpus is indexed at its pdfs/ directory, which is how the CLI
            # examples and the benchmark invoke it.
            page_id = f"{doc_id}::p{p + 1}"
            words: list = []

            def put(
                text: str,
                x: float,
                y: float,
                size: float,
                font: str = "Helvetica",
                _c=c,
                _words=words,
            ) -> None:
                """Draw a line and record a normalised word box (top-left origin).

                ``_c``/``_words`` are bound as defaults rather than captured, so
                the closure always writes to the page it was defined for.
                """
                _draw_line(_c, _words, text, x, y, size, font, page_w, page_h)

            margin = 60
            y = page_h - margin

            # Header block
            put(heading, margin, y, 16, "Helvetica-Bold")
            y -= 22
            put(f"{CITIES[rng.randrange(len(CITIES))]} Regional Office", margin, y, 10)
            y -= 10
            c.line(margin, y, page_w - margin, y)
            y -= 26

            # Field block — the retrievable identifiers
            unique_code = f"{kind[:3].upper()}{spec.seed:02d}{d:03d}{p + 1}"
            subject = SUBJECTS[(d * spec.pages_per_doc + p) % len(SUBJECTS)]
            for i, field in enumerate(fields):
                value = unique_code if i == 0 else _field_value(field, rng)
                if i == 0 and spec.code_scale == 1.0:
                    put(f"{field}: {value}", margin, y, 11, "Helvetica-Bold")
                    y -= 18
                elif i == 0:
                    from reportlab.pdfbase import pdfmetrics

                    size = 11 * spec.code_scale
                    label = f"{field}:"
                    put(label, margin, y, 11, "Helvetica-Bold")
                    if size < 11:  # shrink in place, right after the label
                        x = margin + pdfmetrics.stringWidth(label + " ", "Helvetica-Bold", 11)
                        put(value, x, y, size, "Helvetica-Bold")
                    else:  # grow into the blank right half, one line lower
                        x = page_w - margin - pdfmetrics.stringWidth(value, "Helvetica-Bold", size)
                        put(value, x, y - 18, size, "Helvetica-Bold")
                    y -= 18
                else:
                    put(f"{field}: {value}", margin, y, 11, "Helvetica")
                    y -= 18
            y -= 8
            put(f"Subject: {subject}", margin, y, 11)
            y -= 24

            # Body — a few short paragraphs, leaving the page mostly empty,
            # which is exactly the whitespace profile OptiVision exploits.
            n_lines = rng.randint(6, 14)
            for _ in range(n_lines):
                n_words = rng.randint(8, 13)
                start = rng.randrange(0, max(1, len(BODY_WORDS) - n_words))
                put(" ".join(BODY_WORDS[start : start + n_words]), margin, y, 10)
                y -= 14

            # Footer
            put(f"Page {p + 1} of {spec.pages_per_doc}", margin, margin * 0.6, 8)
            put(unique_code, page_w - margin - 80 * max(spec.code_scale, 1.0), margin * 0.6, 8 * spec.code_scale)

            layout[page_id] = words
            page_index.append(
                {"page_id": page_id, "kind": kind, "subject": subject, "code": unique_code}
            )
            c.showPage()
        c.save()

    queries = _build_queries(page_index)
    manifest = {
        "spec": spec.__dict__,
        "n_docs": spec.n_docs,
        "n_pages": spec.n_docs * spec.pages_per_doc,
        "n_queries": len(queries),
        "n_precise": sum(1 for q in queries if q["type"] == "precise"),
        "n_topical": sum(1 for q in queries if q["type"] == "topical"),
        "pdf_dir": str(pdf_dir),
    }
    _dump(out_dir / "layout.json", layout)
    _dump(out_dir / "queries.json", queries)
    _dump(out_dir / "manifest.json", manifest)
    return manifest


def _draw_line(
    c,
    words: list,
    text: str,
    x: float,
    y: float,
    size: float,
    font: str,
    page_w: float,
    page_h: float,
) -> None:
    """Draw one line of text and record a normalised box for each word.

    Boxes are (x0, y0, x1, y1) in [0, 1] with a top-left origin — PDF space has
    its origin at the bottom left, so y is flipped here rather than at read time.
    """
    from reportlab.pdfbase import pdfmetrics

    c.setFont(font, size)
    c.drawString(x, y, text)
    space = pdfmetrics.stringWidth(" ", font, size)
    cursor = x
    for token in text.split(" "):
        if not token:
            cursor += space
            continue
        w = pdfmetrics.stringWidth(token, font, size)
        clean = "".join(ch for ch in token.lower() if ch.isalnum())
        if clean:
            words.append(
                [
                    clean,
                    [
                        cursor / page_w,
                        (page_h - y - size) / page_h,
                        (cursor + w) / page_w,
                        (page_h - y + size * 0.3) / page_h,
                    ],
                ]
            )
        cursor += w + space


def _build_queries(pages: list[dict]) -> list[dict]:
    """Two query families, so the benchmark can actually separate variants.

    ``precise``  subject + the page's unique code. Exactly one relevant page.
        A single rare token dominates the match, so every variant tends to get
        these right — they check that compression did not break retrieval at all.

    ``topical``  the subject phrase alone. Several pages share a subject, so all
        of them are relevant and the metric depends on *ordering* a group of
        near-identical candidates. This is where pruning and quantization damage
        actually shows up; without it the table saturates at 1.0 and measures
        nothing.
    """
    by_subject: dict[str, list[str]] = {}
    for page in pages:
        by_subject.setdefault(page["subject"], []).append(page["page_id"])

    queries: list[dict] = []
    for page in pages:
        queries.append(
            {
                "qid": f"q{len(queries):04d}",
                "query": f"{page['subject']} {page['code']}",
                "relevant": [page["page_id"]],
                "kind": page["kind"],
                "type": "precise",
            }
        )
    for subject, page_ids in sorted(by_subject.items()):
        queries.append(
            {
                "qid": f"q{len(queries):04d}",
                "query": subject,
                "relevant": sorted(page_ids),
                "kind": "topical",
                "type": "topical",
            }
        )
    return queries


def _field_value(field: str, rng: random.Random) -> str:
    if "Date" in field or "On" in field or "Collected" in field:
        return f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-202{rng.randint(3, 6)}"
    if "Amount" in field or "Sanctioned" in field or "Extent" in field:
        return f"{rng.randint(12, 980)}.{rng.randint(10, 99)}"
    if "SGPA" in field:
        return f"{rng.uniform(6.0, 9.9):.2f}"
    return f"{rng.randrange(1000, 9999)}"


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def load_queries(path: str | Path) -> tuple[list[str], list[str], dict[str, set[str]]]:
    """Return (qids, query strings, qrels) from a queries.json."""
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    qids = [r["qid"] for r in rows]
    texts = [r["query"] for r in rows]
    qrels = {r["qid"]: set(r["relevant"]) for r in rows}
    return qids, texts, qrels


# ----------------------------------------------------------------- ViDoRe


def load_vidore_subset(
    name: str = "vidore/syntheticDocQA_energy_test",
    out_dir: str | Path = "data/vidore",
    limit: int | None = 100,
) -> dict:
    """Materialise a ViDoRe test split as page images + queries.json.

    Kept deliberately thin: the benchmark runner only ever sees a folder of
    images and a queries file, so the synthetic and real corpora are
    interchangeable.
    """
    from datasets import load_dataset  # optional dependency

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(name, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    queries = []
    seen: dict[str, str] = {}
    for i, row in enumerate(ds):
        page_id = f"{i:05d}::p1"  # matches iter_pages() over out_dir/images
        img_path = img_dir / f"{i:05d}.png"
        if not img_path.exists():
            row["image"].convert("RGB").save(img_path)
        query = row.get("query")
        if query and query not in seen:
            seen[query] = page_id
            queries.append(
                {"qid": f"q{len(queries):04d}", "query": query, "relevant": [page_id]}
            )

    _dump(out_dir / "queries.json", queries)
    manifest = {
        "dataset": name,
        "n_pages": len(ds),
        "n_queries": len(queries),
        "pdf_dir": str(img_dir),
    }
    _dump(out_dir / "manifest.json", manifest)
    return manifest
