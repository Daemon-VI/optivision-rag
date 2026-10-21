"""The optional practice-question tiers: OCR text for scans, a local VLM, and an API model (FINAL_DESIGN 5.3.2).

Everything runs offline with fakes: a fake OCR engine, a fake VLM backend that records what each request
received, a fake ``generate(prompt, images)`` and a fake Anthropic client. The tests pin the promises the
tiers make: nothing is sent without ``allow_upload``, the pages are named and the cost is printed before
the first request, the question stage of the VLM never receives the image, the overlap cap works, scanned
pages fall back to OCR, and seeds make a run repeatable.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from optivision.easy import _ocr, _probes_api, _probes_common, _probes_vlm
from optivision.easy._claim import MixedEvidence  # noqa: F401  (import check: lines never mix kinds)
from optivision.easy._types import EasyError, MissingExtra

# ---------------------------------------------------------------------------- Tier A stand-in


def _install_probes_stand_in() -> bool:
    """Define a compatible ``optivision.easy._probes`` when the Tier A module is not in the tree yet."""
    try:
        import optivision.easy._probes  # noqa: F401
    except ImportError:
        pass
    else:
        return False
    mod = types.ModuleType("optivision.easy._probes")

    @dataclass(frozen=True)
    class Probe:
        id: str
        text: str
        family: str
        page_id: str
        relevant: frozenset

    def make_probes(pages, *, seed=0, max_pages=600, per_page=3):
        out = []
        for page in pages:
            if len(page.text) < 20:
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", page.text) if len(s.split()) >= 8]
            for k, sentence in enumerate(sentences[:per_page]):
                words = _probes_common.content_words(sentence)
                relevant = {
                    p.id for p in pages
                    if words and sum(w in set(_probes_common.content_words(p.text)) for w in words) >= 0.8 * len(words)
                }
                out.append(Probe(id=f"A-{page.id}-{k}", text=sentence, family="span", page_id=page.id,
                                 relevant=frozenset(relevant | {page.id})))
        return out

    mod.Probe, mod.make_probes = Probe, make_probes
    sys.modules["optivision.easy._probes"] = mod
    return True


STAND_IN = _install_probes_stand_in()

# ---------------------------------------------------------------------------- fixtures


@dataclass(frozen=True)
class Page:
    id: str
    text: str = ""
    n_chars: int = -1
    blank: bool = False
    file: str | None = None
    page: int | None = None

    def __post_init__(self) -> None:
        if self.n_chars < 0:
            object.__setattr__(self, "n_chars", len(self.text))


TOPICS = [
    "elevator repairs and the cooling system on the second floor",
    "the fire safety inspection contract with an external vendor",
    "the generator service contract and the diesel storage tank",
    "visitor registration at the front desk and the digital sign in system",
    "roof waterproofing for the east wing before the monsoon season",
    "the annual audit of electrical panels in the basement parking",
    "canteen supplier changes and the new menu for night shift staff",
    "security camera replacement across the main gate and loading dock",
]


def text_page(i: int, doc: str = "docs/a.pdf") -> Page:
    topic = TOPICS[i % len(TOPICS)]
    text = (f"Report section {i}. The facilities team reviewed {topic} during the quarter. "
            f"Costs for {topic} were approved by the finance committee in meeting {i}.")
    return Page(id=f"{doc}::p{i + 1}", text=text, file=f"/data/{doc}", page=i + 1)


def scan_page(i: int, doc: str = "scans/b.pdf") -> Page:
    return Page(id=f"{doc}::p{i + 1}", text="", file=f"/data/{doc}", page=i + 1)


def images_for(pages, size=(560, 840)) -> dict[str, Image.Image]:
    """One flat-colour image per page; the colour carries the page index so fakes can tell pages apart."""
    return {p.id: Image.new("RGB", size, (k % 250, 0, 0)) for k, p in enumerate(pages)}


def mark(image: Image.Image) -> int:
    return image.getpixel((0, 0))[0]


@pytest.fixture
def folder():
    pages = [text_page(i) for i in range(6)] + [scan_page(i) for i in range(10)]
    return pages, images_for(pages)


# ---------------------------------------------------------------------------- shared helpers


def test_group_names_compacts_page_runs():
    ids = ["x/a.pdf::p1", "x/a.pdf::p2", "x/a.pdf::p3", "x/a.pdf::p9", "x/b.pdf::p4", "loose-id"]
    lines = _probes_common.group_names(ids)
    assert lines[0] == "x/a.pdf: pages 1-3, 9"
    assert lines[1] == "x/b.pdf: page 4"
    assert lines[2] == "loose-id"


def test_overlap_and_content_words():
    assert _probes_common.content_words("The cooling system of the building") == ["cooling", "system", "building"]
    assert _probes_common.overlap("cooling system repairs", "the cooling system was replaced") == pytest.approx(2 / 3)
    assert _probes_common.overlap("anything here", "") is None


def test_plan_styles_follow_the_weights_and_the_seed():
    plan = _probes_common.plan_styles(100, seed=3)
    counts = {}
    for s in plan:
        counts[s.type] = counts.get(s.type, 0) + 1
    assert counts == {"extractive": 30, "open-ended": 25, "numerical": 15, "boolean": 10, "enumerative": 10,
                      "compare-contrast": 10}
    assert all(s.format == "question" for s in plan if s.type == "boolean")  # a yes/no query is never a keyword phrase
    assert plan == _probes_common.plan_styles(100, seed=3)
    assert plan != _probes_common.plan_styles(100, seed=4)
    with pytest.raises(ValueError):
        _probes_common.plan_styles(5, seed=0, type_weights={"nonsense": 1})


def test_allocate_sums_and_apply_quotas_keep_the_mix():
    assert sum(_probes_common.allocate(37, _probes_common.DEFAULT_TYPE_WEIGHTS).values()) == 37
    items = [("extractive", f"e{i}") for i in range(20)] + [("numerical", f"n{i}") for i in range(20)]
    kept = _probes_common.apply_quotas(items, 10, {"extractive": 1, "numerical": 1}, seed=0)
    assert len(kept) == 10 and sum(k.startswith("e") for k in kept) == 5
    short = _probes_common.apply_quotas([("extractive", "e")] * 2 + [("numerical", f"n{i}") for i in range(8)], 6,
                                        {"extractive": 1, "numerical": 1}, seed=0)
    assert len(short) == 6  # slots a type cannot fill pass to the others
    assert _probes_common.apply_quotas(items, 100, None, seed=0) == [p for _, p in items]


def test_word_index_finds_pages_holding_most_of_the_words():
    index = _probes_common.WordIndex({"a": "the cooling system on the second floor was replaced",
                                      "b": "elevator repairs and the fire safety contract",
                                      "c": "cooling system second floor budget"})
    assert index.holding("cooling system second floor") == ["a", "c"]
    assert index.holding("generator diesel storage tank") == []
    many = _probes_common.WordIndex({str(i): "alpha beta gamma delta epsilon" for i in range(20)})
    assert many.holding("alpha beta gamma delta") is None  # more than 15 pages: too general to be a practice question


def test_clean_question_and_self_reference():
    assert _probes_common.clean_question('Question: "what drives the budget rise" ') == "what drives the budget rise?"
    assert _probes_common.clean_question("budget rise 2024?", "keyword") == "budget rise 2024"
    assert _probes_common.clean_question("", "question") is None
    assert _probes_common.clean_question("one", "question") is None
    assert _probes_common.refers_to_source("What does this page say about repairs?")
    assert _probes_common.refers_to_source("According to image 2, what is the cost?")
    assert not _probes_common.refers_to_source("What is the repair budget?")


def test_select_pages_is_seeded_and_ordered():
    pages = [text_page(i) for i in range(40)]
    a = _probes_common.select_pages(pages, max_pages=10, seed=1)
    assert a == _probes_common.select_pages(pages, max_pages=10, seed=1)
    assert a != _probes_common.select_pages(pages, max_pages=10, seed=2)
    assert a == sorted(a, key=pages.index)
    assert _probes_common.select_pages(pages, max_pages=100, seed=1) == pages


# ---------------------------------------------------------------------------- OCR (extra easy-ocr)


class FakeOcr:
    name = "fake-ocr"

    def __init__(self, results):
        self.results, self.seen = results, []

    def recognise(self, image):
        self.seen.append(mark(image))
        return self.results.get(mark(image), _ocr.OcrResult("", 0.0, 0, 0))


GOOD = ("The facilities team reviewed the elevator repairs during the quarter and approved the cost.\n"
        "The finance committee asked for three quotes before signing the generator service contract.")


def test_ocr_reads_only_pages_without_a_text_layer(folder):
    pages, images = folder
    pages = [*pages, Page(id="docs/blank.pdf::p1", text="", blank=True, file="/x", page=1)]
    images["docs/blank.pdf::p1"] = Image.new("RGB", (560, 840), "white")
    engine = FakeOcr({mark(images[p.id]): _ocr.OcrResult(GOOD, 0.95, 2, len(GOOD)) for p in pages})
    run = _ocr.ocr_pages(pages, engine=engine, images=images, max_pages=100)
    scanned = [p for p in pages if not p.text and not p.blank]
    assert len(engine.seen) == len(scanned) == run.n_read
    assert set(run.texts) == {p.id for p in scanned}
    assert not any(p.id in run.texts for p in pages if p.text or p.blank)


def test_ocr_leaves_out_low_confidence_and_short_pages(folder):
    pages, images = folder
    scans = [p for p in pages if not p.text]
    results = {
        mark(images[scans[0].id]): _ocr.OcrResult(GOOD, 0.95, 2, len(GOOD)),
        mark(images[scans[1].id]): _ocr.OcrResult(GOOD, 0.40, 2, len(GOOD)),
        mark(images[scans[2].id]): _ocr.OcrResult("page 3", 0.99, 1, 6),
    }
    run = _ocr.ocr_pages(scans[:4], engine=FakeOcr(results), images=images, max_pages=10)
    assert set(run.texts) == {scans[0].id}
    assert run.skipped[scans[1].id] == "low confidence"
    assert run.skipped[scans[2].id] == "too little text"
    assert run.skipped[scans[3].id] == "too little text"  # the engine read nothing at all


def test_ocr_error_on_one_page_does_not_stop_the_run(folder):
    pages, images = folder
    scans = [p for p in pages if not p.text][:3]

    class Flaky(FakeOcr):
        def recognise(self, image):
            if mark(image) == mark(images[scans[1].id]):
                raise RuntimeError("bad image")
            return _ocr.OcrResult(GOOD, 0.9, 2, len(GOOD))

    run = _ocr.ocr_pages(scans, engine=Flaky({}), images=images)
    assert set(run.texts) == {scans[0].id, scans[2].id}
    assert run.skipped[scans[1].id].startswith("error")


def test_ocr_sample_is_seeded(folder):
    pages, images = folder
    results = {mark(images[p.id]): _ocr.OcrResult(GOOD, 0.95, 2, len(GOOD)) for p in pages}
    a = _ocr.ocr_pages(pages, engine=FakeOcr(results), images=images, max_pages=4, seed=5)
    b = _ocr.ocr_pages(pages, engine=FakeOcr(results), images=images, max_pages=4, seed=5)
    c = _ocr.ocr_pages(pages, engine=FakeOcr(results), images=images, max_pages=4, seed=6)
    assert list(a.texts) == list(b.texts) and list(a.texts) != list(c.texts)
    assert a.n_attempted == 4 and a.n_candidates == 10


def test_scanned_folder_falls_back_to_ocr_probes(folder):
    """A folder of scans has no text layer, so Tier A has nothing to read; OCR text lets it make questions."""
    pages, images = folder
    scans = [p for p in pages if not p.text]
    results = {mark(images[p.id]): _ocr.OcrResult(GOOD + f"\nSection {k} covers topic {k}.", 0.93, 3, len(GOOD))
               for k, p in enumerate(scans)}
    batch = _ocr.make_ocr_probes(scans, engine=FakeOcr(results), images=images, max_pages=20, per_page=2)
    assert batch.tier == "ocr"
    if STAND_IN:
        assert len(batch.probes) >= len(scans)
    scan_ids = {p.id for p in scans}
    for probe in batch.probes:
        assert probe.family.startswith("ocr:")  # tagged, and reported apart from text-layer questions
        assert probe.id.startswith("ocr-")
        assert probe.page_id in scan_ids
        assert probe.page_id in probe.relevant
    text = str(batch.report)
    assert "OCR text can hold reading mistakes" in text and "Next:" in text
    assert batch.config["engine"] == "fake-ocr"


def test_ocr_probes_only_for_pages_that_were_read(folder):
    pages, images = folder
    scans = [p for p in pages if not p.text]
    results = {mark(images[scans[0].id]): _ocr.OcrResult(GOOD, 0.95, 2, len(GOOD))}
    batch = _ocr.make_ocr_probes(pages, engine=FakeOcr(results), images=images)
    assert {p.page_id for p in batch.probes} <= {scans[0].id}  # text-layer pages keep their Tier A questions
    assert batch.rejected  # the unread scans are counted with their reason


def test_ocr_without_the_extra_names_the_install_line(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    with pytest.raises(MissingExtra) as err:
        _ocr.load_engine()
    assert 'pip install "optivision-rag[easy-ocr]"' in str(err.value)
    with pytest.raises(MissingExtra):
        _ocr.ocr_pages([scan_page(0)], images={scan_page(0).id: Image.new("RGB", (50, 50))})
    run = _ocr.ocr_pages([text_page(0), text_page(1)])  # no scanned page: no engine is needed, so no extra
    assert run.texts == {} and run.n_candidates == 0


def test_ocr_unknown_engine_is_a_plain_error():
    with pytest.raises(ValueError, match="rapidocr"):
        _ocr.load_engine("nothing")


def test_rapidocr_wrapper_reads_both_result_shapes():
    class New:
        txts = ("Quarterly budget review", "  ", "noise", None)
        scores = (0.99, 0.9, 0.3, 0.9)

    engine = _ocr.RapidOcrEngine(min_line_score=0.7)
    engine._engine = lambda arr: New()
    got = engine.recognise(Image.new("RGB", (40, 40)))
    assert got.text == "Quarterly budget review" and got.n_lines == 1 and got.confidence == pytest.approx(0.99)

    legacy_rows = [[[0, 0], "Annual report", 0.9], [[0, 1], "smudge", 0.2]]
    engine._engine = lambda arr: (legacy_rows, [0.1])
    assert engine.recognise(Image.new("RGB", (40, 40))).text == "Annual report"
    engine._engine = lambda arr: (None, [0.1])  # the legacy engine returns None when it finds nothing
    assert engine.recognise(Image.new("RGB", (40, 40))).text == ""


def test_real_rapidocr_reads_a_rendered_line():
    pytest.importorskip("rapidocr")
    pytest.importorskip("onnxruntime")
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(img).text((30, 60), "Quarterly maintenance budget", fill="black", font=ImageFont.load_default(36))
    got = _ocr.load_engine().recognise(img)
    assert "maintenance" in got.text.lower() and got.confidence > 0.5


def test_ocr_time_line_is_an_estimate_with_its_basis():
    text = str(_ocr.time_line(200))
    assert "[estimate: one test on a laptop CPU" in text and text.rstrip().splitlines()[-1].startswith("Next:")


# ---------------------------------------------------------------------------- local VLM (Tier B, extra easy-gen)


class FakeVlm:
    """Records every request: its kind, whether it carried an image, and the prompt."""

    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    revision = "test-revision"

    def __init__(self, question=None, answerable="yes"):
        self.calls: list[dict] = []
        self.question = question or (lambda summary, attempt, page: f"What was decided about budget item {page}?")
        self.answerable = answerable
        self.attempts: dict[str, int] = {}

    def generate(self, prompt, image=None, *, max_new_tokens=256, seed=None, temperature=0.0):
        if prompt.startswith("Summarise"):
            kind = "summary"
            reply = f"A report page numbered marker-{mark(image)} about maintenance spending and committee approvals."
        elif prompt.startswith("Below is a summary"):
            kind = "question"
            summary = prompt.split("Summary:\n", 1)[1]
            page = int(re.search(r"marker-(\d+)", summary).group(1))
            attempt = self.attempts.get(summary, 0)
            self.attempts[summary] = attempt + 1
            reply = self.question(summary, attempt, page)
        else:
            kind = "answerable"
            reply = self.answerable
        self.calls.append({"kind": kind, "image": image, "prompt": prompt, "seed": seed, "temperature": temperature})
        return reply

    def of(self, kind):
        return [c for c in self.calls if c["kind"] == kind]


def test_vlm_question_stage_never_receives_the_image(folder):
    pages, images = folder
    backend = FakeVlm()
    batch = _probes_vlm.generate_vlm_probes(pages[:8], backend=backend, images=images, max_pages=8, out=None)
    assert len(batch.probes) == 8
    assert len(backend.of("summary")) == 8 and len(backend.of("question")) == 8
    assert all(c["image"] is not None for c in backend.of("summary"))
    assert all(c["image"] is None for c in backend.of("question"))  # the blind stage
    page_words = {w for p in pages for w in _probes_common.content_words(p.text) if len(w) > 6}
    for call in backend.of("question"):  # and it saw no page text: only the summary and the instructions
        assert "Summary:" in call["prompt"]
        assert not page_words & set(_probes_common.content_words(call["prompt"].split("Summary:")[0]))
        for p in pages:
            if p.text:
                assert p.text[:40] not in call["prompt"]
    assert all(c["image"] is not None for c in backend.of("answerable"))  # the check may look at the page


def test_vlm_overlap_cap_drops_questions_that_copy_the_page(folder):
    pages, images = folder
    copy = lambda summary, attempt, page: pages[page].text.split(". ")[1] if pages[page].text else "x"  # noqa: E731
    backend = FakeVlm(question=copy)
    batch = _probes_vlm.generate_vlm_probes(pages[:6], backend=backend, images=images, max_pages=6, max_attempts=1,
                                            out=None)
    assert batch.probes == [] and batch.rejected["copies the page"] == 6
    relaxed = _probes_vlm.generate_vlm_probes(pages[:6], backend=FakeVlm(question=copy), images=images, max_pages=6,
                                              max_attempts=1, max_overlap=None, out=None)
    assert len(relaxed.probes) == 6


def test_vlm_retry_samples_with_a_seed_and_keeps_the_better_question(folder):
    pages, images = folder
    page_text = {i: p.text for i, p in enumerate(pages)}

    def question(summary, attempt, page):
        if attempt == 0:  # the greedy reply copies the page; the sampled one is a paraphrase
            return page_text[page].split(". ")[1] if page_text[page] else "x"
        return f"Who approved spending on topic {page} last quarter?"

    run1 = FakeVlm(question=question)
    batch = _probes_vlm.generate_vlm_probes(pages[:6], backend=run1, images=images, max_pages=6, seed=9, out=None)
    assert len(batch.probes) == 6
    retried = [c for c in run1.of("question") if c["temperature"] > 0]
    assert retried and all(isinstance(c["seed"], int) for c in retried)
    run2 = FakeVlm(question=question)
    _probes_vlm.generate_vlm_probes(pages[:6], backend=run2, images=images, max_pages=6, seed=9, out=None)
    assert [c["seed"] for c in run1.of("question")] == [c["seed"] for c in run2.of("question")]


def test_vlm_answerability_filter_drops_unanswerable_questions(folder):
    pages, images = folder
    batch = _probes_vlm.generate_vlm_probes(pages[:5], backend=FakeVlm(answerable="No."), images=images, max_pages=5,
                                            max_attempts=1, out=None)
    assert batch.probes == [] and batch.rejected["not answerable from the page"] == 5
    skipped = FakeVlm()
    _probes_vlm.generate_vlm_probes(pages[:5], backend=skipped, images=images, max_pages=5, check_answerable=False,
                                    out=None)
    assert skipped.of("answerable") == []


def test_vlm_questions_name_no_source_and_are_not_repeated(folder):
    pages, images = folder
    same = FakeVlm(question=lambda s, a, p: "What does this page say about repairs?")
    got = _probes_vlm.generate_vlm_probes(pages[:4], backend=same, images=images, max_attempts=1, out=None)
    assert got.probes == []
    dup = FakeVlm(question=lambda s, a, p: "Which contractor handles the generator service?")
    batch = _probes_vlm.generate_vlm_probes(pages[:4], backend=dup, images=images, max_attempts=1, out=None)
    assert len(batch.probes) == 1 and batch.rejected["duplicate"] == 3


def test_vlm_probes_are_the_tier_a_type_and_tagged(folder):
    pages, images = folder
    scans = [p for p in pages if not p.text]
    batch = _probes_vlm.generate_vlm_probes(scans, backend=FakeVlm(), images=images, max_pages=10, out=None)
    Probe = _probes_common.probe_class()
    assert batch.probes and all(isinstance(p, Probe) for p in batch.probes)
    for p in batch.probes:
        assert re.fullmatch(r"vlm:[a-z-]+", p.family) and p.id.startswith("vlm-")
        assert p.page_id in p.relevant and isinstance(p.relevant, frozenset)
    assert batch.config["revision"] == "test-revision" and batch.config["tier"] == "vlm"
    assert {k.split(":")[1] for k in batch.families()} <= set(_probes_common.QUERY_TYPES)


def test_vlm_aliases_join_the_relevant_set(folder):
    pages, images = folder
    aliases = {pages[7].id: pages[6].id}
    batch = _probes_vlm.generate_vlm_probes(pages[6:8], backend=FakeVlm(), images=images, aliases=aliases, out=None)
    by_page = {p.page_id: p for p in batch.probes}
    assert {pages[6].id, pages[7].id} <= by_page[pages[6].id].relevant


def test_vlm_text_layer_pages_in_the_relevant_set(folder):
    pages, images = folder
    twin = Page(id="docs/z.pdf::p1", text=pages[0].text, file="/x", page=1)
    images[twin.id] = Image.new("RGB", (560, 840), (200, 0, 0))
    q = lambda s, a, p: "Which committee approved the elevator repairs and cooling system costs?"  # noqa: E731
    batch = _probes_vlm.generate_vlm_probes([pages[0], twin], backend=FakeVlm(question=q), images=images,
                                            max_overlap=None, check_answerable=False, out=None)
    assert batch.probes and {pages[0].id, twin.id} <= batch.probes[0].relevant


def test_vlm_seeded_and_repeatable(folder):
    pages, images = folder
    many = [text_page(i) for i in range(30)]
    imgs = images_for(many)

    def run(seed):
        return _probes_vlm.generate_vlm_probes(many, backend=FakeVlm(), images=imgs, max_pages=8, seed=seed, out=None)

    a, b, c = run(1), run(1), run(2)
    assert [p.id for p in a.probes] == [p.id for p in b.probes]
    assert [p.page_id for p in a.probes] != [p.page_id for p in c.probes]


def test_vlm_max_probes_trims_by_the_type_plan():
    many = [text_page(i) for i in range(40)]
    imgs = images_for(many)
    full = _probes_vlm.generate_vlm_probes(many, backend=FakeVlm(), images=imgs, max_pages=40, out=None)
    assert len(full.probes) == 40
    cut = _probes_vlm.generate_vlm_probes(many, backend=FakeVlm(), images=imgs, max_pages=40, max_probes=20, out=None)
    counts = {}
    for p in cut.probes:
        counts[p.family] = counts.get(p.family, 0) + 1
    assert counts == {"vlm:extractive": 6, "vlm:open-ended": 5, "vlm:numerical": 3, "vlm:boolean": 2,
                      "vlm:enumerative": 2, "vlm:compare-contrast": 2}


def test_progress_counts_pages_done_up_to_the_total(folder, api_folder):
    pages, images = folder
    seen: list[tuple[int, int]] = []
    _probes_vlm.generate_vlm_probes(pages[:5], backend=FakeVlm(), images=images, out=None,
                                    progress=lambda a, b: seen.append((a, b)))
    assert seen[0] == (0, 5) and seen[-1] == (5, 5) and [a for a, _ in seen] == sorted(a for a, _ in seen)
    api_pages, api_images = api_folder
    seen.clear()
    run_api(api_pages, api_images, generate=FakeApi(), max_pages=3, progress=lambda a, b: seen.append((a, b)))
    assert seen[0] == (0, 3) and seen[-1] == (3, 3)


def test_ocr_text_feeds_the_other_tiers(folder):
    pages, images = folder
    scans = [p for p in pages if not p.text]
    results = {mark(images[p.id]): _ocr.OcrResult(GOOD, 0.93, 2, len(GOOD)) for p in scans}
    ocr = _ocr.make_ocr_probes(scans, engine=FakeOcr(results), images=images)
    assert set(ocr.texts) == {p.id for p in scans}
    copy = FakeVlm(question=lambda s, a, p: "The facilities team reviewed the elevator repairs during the quarter")
    plain = _probes_vlm.generate_vlm_probes(scans[:3], backend=copy, images=images, max_attempts=1, out=None)
    assert len(plain.probes) == 1 and any("overlap cap" in n for n in plain.notes)  # no text: nothing to compare with
    capped = _probes_vlm.generate_vlm_probes(scans[:3], backend=copy, images=images, texts=ocr.texts, max_attempts=1,
                                             out=None)
    assert capped.probes == [] and capped.rejected["copies the page"] == 3  # with the OCR text the copy is caught


def test_make_tier_probes_dispatches_by_name(folder):
    pages, images = folder
    with pytest.raises(ValueError, match="unknown practice-question tier"):
        _probes_common.make_tier_probes("nope", pages)
    with pytest.raises(_probes_api.UploadNotAllowed):
        _probes_common.make_tier_probes("api", pages, generate=FakeApi(), images=images, out=None)
    batch = _probes_common.make_tier_probes("vlm", pages[:3], backend=FakeVlm(), images=images, out=None)
    assert batch.tier == "vlm" and len(batch.probes) == 3
    assert _probes_common.TIERS == ("ocr", "vlm", "api")


def test_say_survives_a_console_that_cannot_show_a_file_name(monkeypatch):
    import io

    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="ascii", errors="strict"))
    _probes_common.say("pages: café/報告.pdf")
    sys.stdout.flush()
    assert raw.getvalue().decode("ascii").strip() == "pages: caf?/??.pdf"


def test_vlm_time_estimate_is_a_labelled_research_estimate():
    block = str(_probes_vlm.estimate_block(1000, "cuda"))
    assert "about 35-70 min" in block and "[estimate: research estimate" in block and "T4" in block
    cpu = str(_probes_vlm.estimate_block(1000, "cpu"))
    assert "min" not in cpu and "not measured on a CPU" in cpu


def test_vlm_report_prints_time_before_questions_and_model_licence(folder):
    pages, images = folder
    lines: list[str] = []
    batch = _probes_vlm.generate_vlm_probes(pages[:4], backend=FakeVlm(), images=images, out=lines.append)
    assert "[estimate:" in lines[0] and "Questions" in lines[1]
    assert "Apache-2.0" in str(batch.report)


def test_vlm_models_are_permissively_licensed_and_pinned():
    assert _probes_vlm.DEFAULT_MODEL in _probes_vlm.VLM_MODELS
    for info in _probes_vlm.VLM_MODELS.values():
        assert info.licence == "Apache-2.0" and re.fullmatch(r"[0-9a-f]{40}", info.revision)
        assert info.checked and info.licence_source.startswith("https://")
    with pytest.raises(ValueError, match="licence-checked"):
        _probes_vlm.HFVlmBackend("Qwen/Qwen2.5-VL-3B-Instruct")  # qwen-research licence, not offered


def test_vlm_without_the_extra_names_the_install_line(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    backend = _probes_vlm.HFVlmBackend()
    with pytest.raises(MissingExtra) as err:
        backend.resolve_device()
    assert 'pip install "optivision-rag[easy-gen]"' in str(err.value)


# ---------------------------------------------------------------------------- API LLM (Tier C, allow_upload)


class FakeApi:
    """A ``generate(prompt, images)`` that records requests and answers like a model would."""

    def __init__(self, events=None, judge="[]", question=None):
        self.requests: list[dict] = []
        self.events = events if events is not None else []
        self.judge = judge
        self.question = question or (lambda marks: f"Which budget line covers item {marks[0]} approvals?")
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    def __call__(self, prompt, images):
        self.events.append("request")
        kind = "write" if "target page" in prompt else "judge"
        marks = [mark(i) for i in images]
        self.requests.append({"kind": kind, "prompt": prompt, "marks": marks, "n_images": len(images)})
        self.usage["requests"] += 1
        return self.question(marks) if kind == "write" else self.judge


@pytest.fixture
def api_folder():
    pages = [text_page(i) for i in range(4)] + [scan_page(i) for i in range(6)]
    return pages, images_for(pages)


def run_api(pages, images, **kw):
    kw.setdefault("out", None)
    kw.setdefault("allow_upload", True)
    kw.setdefault("max_pages", 4)
    return _probes_api.generate_api_probes(pages, images=images, **kw)


def test_no_upload_without_allow_upload(api_folder):
    pages, images = api_folder
    fake = FakeApi()
    rendered: list[str] = []

    def spying_images(page):
        rendered.append(page.id)
        return images[page.id]

    with pytest.raises(_probes_api.UploadNotAllowed) as err:
        _probes_api.generate_api_probes(pages, generate=fake, images=spying_images, out=None)
    assert fake.requests == [] and rendered == []  # nothing sent, not even a page rendered
    assert "allow_upload=True" in str(err.value) and "dry_run=True" in str(err.value)
    assert isinstance(err.value, (EasyError, PermissionError))
    with pytest.raises(_probes_api.UploadNotAllowed):  # the same for the built-in backend: no client is built
        _probes_api.generate_api_probes(pages, images=images, out=None)


def test_no_upload_without_allow_upload_even_for_a_user_callable_that_looks_local(api_folder):
    pages, images = api_folder
    fake = FakeApi()
    with pytest.raises(_probes_api.UploadNotAllowed):
        _probes_api.generate_api_probes(pages, generate=fake, images=images, allow_upload=False, dry_run=False,
                                        out=None)
    assert fake.requests == []


def test_dry_run_prints_the_plan_and_sends_nothing(api_folder):
    pages, images = api_folder
    fake = FakeApi()
    shown: list[str] = []
    batch = _probes_api.generate_api_probes(pages, generate=fake, images=images, dry_run=True, max_pages=3,
                                            out=shown.append)
    assert fake.requests == [] and batch.probes == [] and "dry run" in batch.notes[0]
    plan = batch.plan
    assert len(plan.targets) == 3 and set(plan.targets) <= set(plan.sent_pages)
    assert len(shown) == 1 and "Next: to send these pages call again with allow_upload=True" in shown[0]
    for line in _probes_common.group_names(plan.sent_pages):
        assert line in shown[0]  # every page that would be sent is named


def test_dry_run_with_allow_upload_still_sends_nothing(api_folder):
    pages, images = api_folder
    fake = FakeApi()
    _probes_api.generate_api_probes(pages, generate=fake, images=images, dry_run=True, allow_upload=True, out=None)
    assert fake.requests == []


def test_pages_are_named_and_cost_printed_before_the_first_request(api_folder):
    pages, images = api_folder
    events: list = []
    fake = FakeApi(events=events)
    batch = run_api(pages, images, generate=fake, out=lambda text: events.append(("print", text)),
                    provider="anthropic", model="claude-haiku-5-5")
    assert batch.probes
    first_print = next(i for i, e in enumerate(events) if isinstance(e, tuple))
    first_request = events.index("request")
    assert first_print < first_request
    plan_text = events[first_print][1]
    assert "docs/a.pdf" in plan_text or "scans/b.pdf" in plan_text  # pages named
    assert "would leave this machine" in plan_text
    assert "[estimate: prices recorded 2026-10-21; check the provider's page]" in plan_text
    assert re.search(r"Cost\s+about \$[0-9.]+, at most \$[0-9.]+", plan_text) or "under $0.01" in plan_text


def test_image_token_rule_matches_the_providers_examples():
    price = _probes_api.price_for("anthropic", "claude-haiku-5-5")
    standard = None  # the 1568 px / 1568 token tier
    assert _probes_api.image_tokens(1000, 1000, standard) == 1296
    assert _probes_api.image_tokens(200, 200, standard) == 64
    assert _probes_api.image_tokens(1920, 1080, standard) == 1560
    assert _probes_api.image_tokens(1920, 1080, price) == 2691  # high-resolution tier, not resized
    assert _probes_api.image_tokens(3840, 2160, price) == 4784  # resized to 2576x1449
    assert _probes_api.image_tokens(1080, 1920, price) == 2691  # portrait is the same as landscape


def test_cost_comes_from_the_images_that_would_be_sent_and_the_dated_price(api_folder):
    pages, images = api_folder
    plan = _probes_api.plan_api_probes(pages, images=images, max_pages=2, max_side=1568, seed=0)
    price = plan.price
    assert price is not None and price.recorded == "2026-10-21"
    per_image = _probes_api.image_tokens(560, 840, price)  # 20 x 30 patches
    assert per_image == 600
    write = [c for c in plan.calls if c.kind == "write"][0]
    judge = [c for c in plan.calls if c.kind == "judge"][0]
    assert len(write.sent) == 3 and len(judge.sent) == 2
    assert 3 * per_image < write.input_tokens < 3 * per_image + 500
    assert 2 * per_image < judge.input_tokens < 2 * per_image + 400
    expected = sum((c.input_tokens * 0.10 + c.expected_output_tokens * 0.50) / 1e6 for c in plan.calls)
    upper = sum((c.input_tokens * 0.10 + c.max_output_tokens * 0.50) / 1e6 for c in plan.calls)
    assert plan.cost_expected == pytest.approx(expected) and plan.cost_upper == pytest.approx(upper)
    assert plan.cost_upper > plan.cost_expected
    smaller = _probes_api.plan_api_probes(pages, images=images, max_pages=2, max_side=280, seed=0)
    assert smaller.input_tokens < plan.input_tokens  # a lower max_side is cheaper


def test_long_prompt_rates_apply_above_the_threshold():
    price = _probes_api.price_for("anthropic", "claude-haiku-5-5")
    assert price.call_cost(100_000, 0) == pytest.approx(0.01)
    assert price.call_cost(100_001, 0) == pytest.approx(100_001 * 0.50 / 1e6)
    assert price.call_cost(1000, 1000) == pytest.approx((1000 * 0.10 + 1000 * 0.50) / 1e6)


def test_price_preset_has_the_dated_fields():
    rows = _probes_api.load_prices()
    assert rows
    for row in rows:
        assert row.provider and row.model and row.input_per_mtok > 0 and row.output_per_mtok > 0
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row.recorded) and row.source_url.startswith("https://")
        assert row.basis == f"prices recorded {row.recorded}; check the provider's page"
    assert _probes_api.DEFAULT_MODEL == "claude-haiku-5-5" and _probes_api.price_for("anthropic", "claude-haiku-5-5")
    raw = json.loads(Path(_probes_api.__file__).parents[1].joinpath("presets", "api_prices.json").read_text("utf-8"))
    assert {"provider", "model", "input_per_mtok", "output_per_mtok", "recorded", "source_url"} <= set(raw["prices"][0])


def test_unknown_model_has_no_price_and_says_so(api_folder):
    pages, images = api_folder
    plan = _probes_api.plan_api_probes(pages, images=images, max_pages=2, provider="custom", model="my-model")
    assert plan.price is None and plan.cost_expected is None
    text = str(plan)
    assert "Cost" in text and "not known" in text and "$" not in text
    given = _probes_api.plan_api_probes(pages, images=images, max_pages=2, provider="custom", model="my-model",
                                        price={"input_per_mtok": 1.0, "output_per_mtok": 2.0, "recorded": "2026-01-01",
                                               "source_url": "https://example.com/prices"})
    assert given.cost_expected is not None and "prices recorded 2026-01-01" in str(given)


def test_contrastive_two_requests_and_judged_neighbours_join_the_relevant_set(api_folder):
    pages, images = api_folder
    fake = FakeApi(judge="[1]")
    near = {p.id: [pages[0].id, pages[1].id] for p in pages}
    batch = run_api(pages, images, generate=fake, max_pages=3, neighbours=near)
    writes = [r for r in fake.requests if r["kind"] == "write"]
    judges = [r for r in fake.requests if r["kind"] == "judge"]
    assert writes and judges and all(r["n_images"] == 3 for r in writes) and all(r["n_images"] == 2 for r in judges)
    by_page = {p.page_id: p for p in batch.probes}
    for target in batch.plan.targets:
        probe = by_page.get(target)
        if probe is None:
            continue
        neighbours = [n for n in (pages[0].id, pages[1].id) if n != target][:2]
        assert target in probe.relevant and neighbours[0] in probe.relevant  # the judged neighbour (reply "[1]")
    assert all(p.family.startswith("api:") and p.id.startswith("api-") for p in batch.probes)


def test_judge_sees_only_the_neighbours_not_the_target(api_folder):
    pages, images = api_folder
    fake = FakeApi()
    batch = run_api(pages, images, generate=fake, max_pages=2)
    markers = {p.id: mark(images[p.id]) for p in pages}
    for target in batch.plan.targets:
        write = next(r for r in fake.requests if r["kind"] == "write" and r["marks"][0] == markers[target])
        judge = fake.requests[fake.requests.index(write) + 1]
        assert judge["kind"] == "judge" and markers[target] not in judge["marks"]
        assert judge["marks"] == write["marks"][1:]


def test_parse_judged():
    assert _probes_api.parse_judged("[2]", 2) == [2]
    assert _probes_api.parse_judged("[1, 2]", 2) == [1, 2]
    assert _probes_api.parse_judged("[]", 2) == []
    assert _probes_api.parse_judged("None of them.", 2) == []
    assert _probes_api.parse_judged("Image 2 answers it", 2) == [2]
    assert _probes_api.parse_judged("[5, 1]", 2) == [1]  # out of range numbers are ignored
    assert _probes_api.parse_judged("", 2) == []


def test_aliases_are_never_sent_as_neighbours_and_join_the_relevant_set(api_folder):
    pages, images = api_folder
    aliases = {pages[1].id: pages[0].id}
    near = _probes_api.find_neighbours(pages, [pages[0]], k=2, aliases=aliases)
    assert pages[1].id not in near[pages[0].id] and pages[0].id not in near[pages[0].id]
    batch = run_api(pages, images, generate=FakeApi(), aliases=aliases, max_pages=10)
    first = next(p for p in batch.probes if p.page_id == pages[0].id)
    assert {pages[0].id, pages[1].id} <= first.relevant


def test_neighbours_prefer_similar_text_then_folder_order():
    # text pages 0 and 8 share a topic
    pages = [text_page(0), text_page(1), text_page(8), scan_page(0), scan_page(1), scan_page(2)]
    near = _probes_api.find_neighbours(pages, [pages[0], pages[3]], k=2)
    assert near[pages[0].id][0] == pages[2].id and len(near[pages[0].id]) == 2
    assert near[pages[3].id] == [pages[4].id, pages[2].id]  # a page with no text gets its neighbours in folder order
    by_hand = _probes_api.find_neighbours(pages, [pages[0]], k=2, neighbours=lambda pid, k: [pages[5].id])
    assert by_hand[pages[0].id][0] == pages[5].id
    outside = _probes_api.find_neighbours(pages, [pages[0]], k=2, neighbours={pages[0].id: ["not/in/folder::p1"]})
    assert "not/in/folder::p1" not in outside[pages[0].id]
    blank = Page(id="x::p1", text="", blank=True)
    assert blank.id not in _probes_api.find_neighbours([*pages, blank], [pages[3]], k=5)[pages[3].id]


def test_nothing_identifying_is_sent_to_the_model(api_folder):
    pages, images = api_folder
    fake = FakeApi()
    run_api(pages, images, generate=fake, max_pages=3)
    for request in fake.requests:
        assert "docs/" not in request["prompt"] and "scans/" not in request["prompt"] and "::p" not in request["prompt"]
        assert pages[0].text[:30] not in request["prompt"]


def test_max_cost_refuses_before_anything_is_sent(api_folder):
    pages, images = api_folder
    fake = FakeApi()
    shown: list[str] = []
    price = {"input_per_mtok": 1.0, "output_per_mtok": 1.0, "recorded": "2026-01-01", "source_url": "https://x"}
    with pytest.raises(_probes_api.ApiProbeError, match="nothing was sent"):
        run_api(pages, images, generate=fake, max_cost=1e-9, price=price, out=shown.append)
    assert fake.requests == [] and shown  # the plan was printed, then the run was refused
    with pytest.raises(_probes_api.ApiProbeError, match="cannot be priced"):
        run_api(pages, images, generate=FakeApi(), max_cost=5.0, provider="custom", model="unpriced")


def test_confirm_can_decline(api_folder):
    pages, images = api_folder
    fake = FakeApi()
    seen_plans = []
    batch = run_api(pages, images, generate=fake, confirm=lambda plan: seen_plans.append(plan) or False)
    assert fake.requests == [] and batch.probes == [] and "declined" in batch.notes[0] and seen_plans


def test_overlap_cap_applies_to_api_questions(api_folder):
    pages, images = api_folder
    texts = {pages[0].id: "approved spending elevator repairs cooling second floor budget committee"}
    copy = FakeApi(question=lambda marks: "approved spending elevator repairs cooling second floor budget committee?")
    batch = run_api([pages[0]], images, generate=copy, max_pages=1, texts=texts)
    assert batch.probes == [] and batch.rejected["copies the page"] == 1
    free = run_api([pages[0]], images, generate=FakeApi(question=lambda marks: "Who signed off the lift upgrade?"),
                   max_pages=1, texts=texts)
    assert len(free.probes) == 1


def test_scanned_pages_use_ocr_text_for_the_overlap_cap(api_folder):
    pages, images = api_folder
    scan = pages[5]
    ocr_text = {scan.id: "the generator service contract needs three quotes from suppliers before signing"}
    copy = FakeApi(question=lambda marks: "generator service contract needs three quotes from suppliers before signing")
    batch = run_api([scan], images, generate=copy, max_pages=1, texts=ocr_text)
    assert batch.rejected.get("copies the page") == 1
    no_text = run_api([scan], images, generate=FakeApi(), max_pages=1)
    assert any("overlap cap could not be applied" in n for n in no_text.notes)


def test_seed_makes_the_run_repeatable(api_folder):
    pages, images = api_folder
    many = [text_page(i) for i in range(24)]
    imgs = images_for(many)

    def plan(seed):
        return _probes_api.plan_api_probes(many, images=imgs, max_pages=6, seed=seed)

    assert plan(1).targets == plan(1).targets and plan(1).targets != plan(2).targets
    a = run_api(many, imgs, generate=FakeApi(), max_pages=6, seed=1)
    b = run_api(many, imgs, generate=FakeApi(), max_pages=6, seed=1)
    assert [(p.id, p.text, p.family) for p in a.probes] == [(p.id, p.text, p.family) for p in b.probes]


def test_cache_means_a_rerun_pays_nothing_again(api_folder, tmp_path):
    pages, images = api_folder
    cache = tmp_path / "api_probes.jsonl"
    first = FakeApi()
    a = run_api(pages, images, generate=first, cache_path=cache, max_pages=3)
    assert first.requests and cache.exists()
    cache.write_text(cache.read_text("utf-8") + '{"half written', encoding="utf-8")  # a crash mid-write is tolerated
    second = FakeApi()
    shown: list[str] = []
    b = run_api(pages, images, generate=second, cache_path=cache, max_pages=3, out=shown.append)
    assert second.requests == []
    assert [p.id for p in a.probes] == [p.id for p in b.probes]
    assert "already have saved answers" in shown[0] and b.plan.cached_pages


def test_failures_stop_the_run_rather_than_billing_into_the_void(api_folder):
    pages, images = api_folder
    calls = []

    def broken(prompt, imgs):
        calls.append(1)
        raise RuntimeError("service unavailable")

    batch = run_api(pages, images, generate=broken, max_pages=6)
    assert len(calls) == _probes_api.MAX_CONSECUTIVE_ERRORS and batch.probes == []
    assert "stopped after repeated failures" in batch.notes[-1]

    class Unauthorised(Exception):
        status_code = 401

    def bad_key(prompt, imgs):
        calls.append(1)
        raise Unauthorised("invalid x-api-key")

    calls.clear()
    run_api(pages, images, generate=bad_key, max_pages=6)
    assert len(calls) == 1  # a rejected key is not retried page after page


def test_one_failed_page_costs_that_page_only(api_folder):
    pages, images = api_folder
    state = {"n": 0}
    fake = FakeApi()

    def flaky(prompt, imgs):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("timeout")
        return fake(prompt, imgs)

    batch = run_api(pages, images, generate=flaky, max_pages=4)
    assert batch.rejected["request failed"] == 1 and len(batch.probes) == 3


def test_final_report_separates_counted_tokens_from_estimated_cost(api_folder):
    pages, images = api_folder
    fake = FakeApi()
    fake.usage = {"requests": 0, "input_tokens": 12345, "output_tokens": 678}
    batch = run_api(pages, images, generate=fake, max_pages=2, provider="anthropic", model="claude-haiku-5-5")
    text = str(batch.report)
    tokens = next(ln for ln in text.splitlines() if ln.startswith("Tokens"))
    cost = next(ln for ln in text.splitlines() if ln.startswith("Cost"))
    assert "[counted]" in tokens and "12,345" in tokens
    assert "[estimate: prices recorded 2026-10-21; check the provider's page]" in cost


# ---- the built-in Anthropic backend, with a fake client


class FakeAnthropic:
    def __init__(self, text="Which budget line covers lift repairs?", stop_reason="end_turn"):
        self.calls: list[dict] = []
        self.messages = self
        self._text, self._stop = text, stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text=self._text)
        usage = types.SimpleNamespace(input_tokens=1000, output_tokens=20)
        return types.SimpleNamespace(content=[block], usage=usage, stop_reason=self._stop)


def test_anthropic_backend_request_shape_and_privacy():
    client = FakeAnthropic()
    backend = _probes_api.AnthropicBackend(client=client)
    images = [Image.new("RGB", (100, 140), "white"), Image.new("RGB", (100, 140), "black")]
    assert backend("Write a query.", images) == "Which budget line covers lift repairs?"
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5" and call["output_config"] == {"effort": "low"}
    assert "temperature" not in call and "top_p" not in call and "top_k" not in call  # rejected by this model family
    content = call["messages"][0]["content"]
    kinds = [b["type"] for b in content]
    assert kinds == ["text", "image", "text", "image", "text"] and content[-1]["text"] == "Write a query."
    assert content[1]["source"]["media_type"] == "image/jpeg" and content[1]["source"]["type"] == "base64"
    assert [b["text"] for b in content if b["type"] == "text"][:2] == ["Image 1:", "Image 2:"]
    assert backend.usage == {"requests": 1, "input_tokens": 1000, "output_tokens": 20, "refusals": 0}


def test_anthropic_backend_reports_a_refusal_as_no_text():
    backend = _probes_api.AnthropicBackend(client=FakeAnthropic(text="ignored", stop_reason="refusal"))
    assert backend("p", [Image.new("RGB", (10, 10))]) == "" and backend.usage["refusals"] == 1


def test_anthropic_backend_without_the_sdk_names_the_install_line(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(MissingExtra) as err:
        _probes_api.AnthropicBackend()("p", [])
    assert 'pip install "optivision-rag[easy-gen]"' in str(err.value)


def test_built_in_backend_fails_before_sending_when_the_sdk_is_missing(api_folder, monkeypatch):
    pages, images = api_folder
    monkeypatch.setitem(sys.modules, "anthropic", None)
    shown: list[str] = []
    with pytest.raises(MissingExtra):
        run_api(pages, images, max_pages=2, out=shown.append)  # no generate=: the Anthropic backend is built
    assert shown and "would leave this machine" in shown[0]  # the plan was shown, nothing was sent


def test_end_to_end_with_the_built_in_backend_and_a_fake_client(api_folder, monkeypatch):
    pages, images = api_folder
    client = FakeAnthropic(text="Who approved the lift repair spending?")
    monkeypatch.setattr(_probes_api.AnthropicBackend, "_get_client", lambda self: client)
    batch = run_api(pages, images, max_pages=2)
    assert batch.probes and client.calls
    assert batch.usage["input_tokens"] == 1000 * len(client.calls)
    for call in client.calls:
        assert "my_pdfs" not in json.dumps({k: v for k, v in call.items() if k != "messages"})
        text_blocks = [b["text"] for b in call["messages"][0]["content"] if b["type"] == "text"]
        assert not any("::p" in t or "docs/" in t or "scans/" in t for t in text_blocks)


# ---------------------------------------------------------------------------- house rules


def test_nothing_in_the_tiers_imports_heavy_packages_at_import_time():
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import optivision.easy._ocr, optivision.easy._probes_vlm, optivision.easy._probes_api, "
        "optivision.easy._probes_common\n"
        "new = set(sys.modules) - before\n"
        "heavy = {'torch', 'transformers', 'anthropic', 'rapidocr', 'onnxruntime', 'scipy', 'pypdfium2', 'cv2'}\n"
        "bad = sorted(m for m in new if m.split('.')[0] in heavy)\n"
        "print('BAD', bad) if bad else print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                         env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    assert out.stdout.strip() == "OK", out.stdout + out.stderr


BANNED = ("lossless", "universal", "guaranteed", "state of the art", "state-of-the-art", "sota", "veto", "novel",
          "better than", "faster", "scales to", "works with every model")
BANNED_WORD = re.compile(r"\b(simply|just|first)\b", re.IGNORECASE)


def _sources():
    root = Path(_probes_api.__file__).parent
    return [root / "_ocr.py", root / "_probes_vlm.py", root / "_probes_api.py", root / "_probes_common.py",
            root.parent / "presets" / "api_prices.json"]


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_user_facing_text_avoids_the_banned_words(path):
    text = path.read_text(encoding="utf-8").lower()
    for phrase in BANNED:
        assert phrase not in text, f"{path.name}: {phrase!r}"
    found = BANNED_WORD.search(text)
    assert found is None, f"{path.name}: {found.group(0)!r}"
    try:
        from tests.test_docs_examples import BANNED_PHRASES, BANNED_WORDS
    except ImportError:  # run outside the tests package
        return
    assert not [ph for ph in BANNED_PHRASES if ph in text]
    assert BANNED_WORDS.search(text) is None


def test_rendered_blocks_are_ascii_and_end_with_next(folder, api_folder):
    pages, images = folder
    api_pages, api_images = api_folder
    texts = [
        str(_probes_vlm.estimate_block(500)),
        str(_probes_vlm.generate_vlm_probes(pages[:3], backend=FakeVlm(), images=images, out=None).report),
        str(_probes_api.plan_api_probes(api_pages, images=api_images, max_pages=2)),
        str(run_api(api_pages, api_images, generate=FakeApi(), max_pages=2).report),
        str(_ocr.time_line(50)),
        str(_ocr.make_ocr_probes(pages, engine=FakeOcr({}), images=images).report),
    ]
    for text in texts:
        assert text.isascii(), text
        assert text.rstrip().splitlines()[-1].startswith("Next:"), text
    # every figure carries a tag: no digit appears on a line that has none, except in names and page lists
    for text in texts:
        for line in text.splitlines():
            named = re.search(r"\.pdf|claude-|Qwen|SmolVLM|Apache|checked|licence", line)
            if re.search(r"\d", line) and "[" not in line and not named:
                raise AssertionError(f"a number without a tag: {line!r}")
