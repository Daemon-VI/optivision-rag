"""Tier B practice questions: a local vision-language model writes them (``probes="vlm"``, extra ``easy-gen``).

Nothing leaves the machine; the model runs on the user's GPU (a CPU works for the small model, slowly).

Blind two-stage generation (FINAL_DESIGN 5.3.2). A question written while looking at a page tends to copy
the page's words, and such questions are easier than real ones (FINAL_DESIGN 5.4). So:

1. the model sees the page image and writes a short summary of it;
2. a second request gets only the summary, no image and no page text, and writes the question. This
   stage cannot copy wording from the page, because it never receives it;
3. the page text (text layer or OCR text, where there is any) is compared with the question; a question
   that holds more than 60% of its content words in the page text is dropped as a copy;
4. an answerability request shows the model the page and the question and asks whether the page answers it;
   questions it says no to are dropped.

Question types follow a plan (ViDoRe V3 type and format vocabulary, ``_probes_common``) so a folder does not
get only one kind of question.

Models (licence check, 2026-10-21, from the Hugging Face model cards and LICENSE files):

* ``Qwen/Qwen2-VL-2B-Instruct``, Apache-2.0: the default. About 2.2 billion parameters, about 4.4 GB in 16-bit.
* ``HuggingFaceTB/SmolVLM-500M-Instruct``, Apache-2.0: smaller (about 1 GB), for machines without a GPU.
* ``Qwen/Qwen2.5-VL-3B-Instruct`` is not offered: its licence is ``qwen-research``, which is not Apache-2.0
  and limits commercial use.

Weights are pinned to the commit recorded in ``VLM_MODELS`` (the build record stores it), and
``trust_remote_code`` is never turned on: both models are supported by ``transformers`` itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ._claim import Block, claim, duration, lit
from ._probes_common import (
    CONT,
    QUERY_FORMATS,
    QUERY_TYPES,
    ProbeBatch,
    Rejections,
    Style,
    WordIndex,
    add_counts,
    apply_quotas,
    check_question,
    clean_question,
    get_image,
    is_blank,
    make_probe,
    page_id,
    page_text,
    plan_styles,
    prepare_image,
    prompt_hash,
    relevant_set,
    say,
    select_pages,
    stable_id,
)
from ._types import MissingExtra


@dataclass(frozen=True)
class VlmInfo:
    """A model we have checked: where it comes from, at which commit, and under which licence."""

    model_id: str
    revision: str
    licence: str
    licence_source: str
    checked: str
    kind: str
    weights_gb: float
    note: str = ""


VLM_MODELS: dict[str, VlmInfo] = {
    "Qwen/Qwen2-VL-2B-Instruct": VlmInfo(
        model_id="Qwen/Qwen2-VL-2B-Instruct", revision="895c3a49bc3fa70a340399125c650a463535e71c",
        licence="Apache-2.0",
        licence_source="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct (model card and LICENSE)",
        checked="2026-10-21", kind="qwen2_vl", weights_gb=4.4,
        note="the default; 16-bit weights fit a 16 GB T4"),
    "HuggingFaceTB/SmolVLM-500M-Instruct": VlmInfo(
        model_id="HuggingFaceTB/SmolVLM-500M-Instruct", revision="a7da5b986cb59b408707209984f360a5f4ad7e47",
        licence="Apache-2.0", licence_source="https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct (model card)",
        checked="2026-10-21", kind="idefics3", weights_gb=1.0,
        note="smaller and weaker; usable without a GPU"),
}
DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

#: seconds per page on a T4 for summary + question + answerability: 35-70 minutes per 1,000 pages. This is a
#: research estimate (FINAL_DESIGN 5.3.2), not a measurement of this code, and is always shown as one.
T4_SECONDS_PER_PAGE = (2.1, 4.2)
_T4_BASIS = "research estimate for a 2B-class model on a T4, 35-70 min per 1,000 pages; not measured on your machine"

SUMMARY_PROMPT = (
    "Summarise this document page in three to five sentences. Name the main topic, the key terms, names, "
    "figures and numbers, and say what kind of page it is (for example a table, a form, a chart or running text). "
    "Reply with the summary only."
)
QUESTION_PROMPT = (
    "Below is a summary of one page from a collection of documents. You cannot see the page. "
    "Write one search query that a person looking for this page could type into a search box. "
    "The query {type_text}. Write it as {format_text}. "
    "Use your own words where you can instead of copying phrases from the summary. "
    "The query must make sense on its own and must not mention the summary, the page or this task. "
    "Reply with the query only, on one line.\n\nSummary:\n{summary}"
)
ANSWERABLE_PROMPT = (
    "Does this page contain the information needed to answer the following query? "
    "Reply with yes or no only.\n\nQuery: {question}"
)


class VlmBackend(Protocol):
    """What the generator needs: ``image=None`` means a text-only request."""

    def generate(self, prompt: str, image: Any = None, *, max_new_tokens: int = 256, seed: int | None = None,
                 temperature: float = 0.0) -> str: ...


def install_line() -> str:
    return 'pip install "optivision-rag[easy-gen]"'


def estimate(n_pages: int) -> tuple[float, float]:
    """Seconds on a T4 for ``n_pages`` pages (low, high); a research estimate."""
    lo, hi = T4_SECONDS_PER_PAGE
    return lo * n_pages, hi * n_pages


def estimate_block(n_pages: int, device: str = "cuda") -> Block:
    """The time line printed before generation starts."""
    block = Block([], title="Practice questions from a local VLM")
    block.add("Pages", "{0} pages to read", claim(n_pages, "counted"))
    if not str(device).startswith(("cpu", "mps")):
        lo, hi = estimate(n_pages)
        block.add("Time", "{0} on a T4; other GPUs are not measured", claim(duration(lo, hi), "estimate", _T4_BASIS))
    else:
        block.add("Time", "not measured on a CPU; this tier is sized for a GPU, so expect it to be several times "
                          "slower")
    block.next = "lower max_pages to read fewer pages, or use probes=\"text\" (no model, no GPU)"
    return block


# ---------------------------------------------------------------------------- the Hugging Face backend


class HFVlmBackend:
    """A ``transformers`` image-text-to-text model, loaded when it is used.

    ``device="auto"`` picks CUDA when there is one. ``dtype="auto"`` is float16 on CUDA (a T4 has no bfloat16)
    and float32 on CPU. Greedy decoding is the default; ``temperature > 0`` samples with the given seed.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, revision: str | None = None, device: str = "auto",
                 dtype: str = "auto", max_pixels: int | None = None) -> None:
        info = VLM_MODELS.get(model)
        if info is None:
            raise ValueError(f"{model!r} has not been licence-checked for practice questions; "
                             f"checked models: {sorted(VLM_MODELS)}. Pass your own backend= to use another.")
        self.info = info
        self.model_id = model
        self.revision = revision or info.revision
        self.device_name = device
        self.dtype_name = dtype
        default_pixels = 1024 * 28 * 28 if info.kind == "qwen2_vl" else None
        self.max_pixels = max_pixels if max_pixels is not None else default_pixels
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device = "cpu"

    @property
    def device(self) -> str:
        return self._device

    def resolve_device(self) -> str:
        """The device that will be used, without loading the model (needs torch)."""
        try:
            import torch
        except ImportError as exc:
            raise MissingExtra(f"local VLM questions need torch and transformers ({exc}). Install them with: "
                               f"{install_line()}") from exc
        if self.device_name != "auto":
            return self.device_name
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise MissingExtra(f"local VLM questions need torch and transformers ({exc}). Install them with: "
                               f"{install_line()}") from exc
        self._torch = torch
        self._device = self.resolve_device()
        dtype = {"auto": torch.float16 if self._device.startswith("cuda") else torch.float32,
                 "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype_name]
        proc_kwargs: dict[str, Any] = {"revision": self.revision}
        if self.max_pixels is not None:
            proc_kwargs["max_pixels"] = self.max_pixels
        self._processor = AutoProcessor.from_pretrained(self.model_id, **proc_kwargs)
        try:  # transformers 4.56 and later call it dtype; earlier versions torch_dtype
            model = AutoModelForImageTextToText.from_pretrained(self.model_id, revision=self.revision, dtype=dtype)
        except TypeError:
            model = AutoModelForImageTextToText.from_pretrained(self.model_id, revision=self.revision,
                                                                torch_dtype=dtype)
        self._model = model.to(self._device).eval()

    def close(self) -> None:
        """Free the weights (and the GPU memory) once the questions are made."""
        self._model = self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def generate(self, prompt: str, image: Any = None, *, max_new_tokens: int = 256, seed: int | None = None,
                 temperature: float = 0.0) -> str:
        self._load()
        torch = self._torch
        content: list[dict[str, Any]] = [{"type": "image"}] if image is not None else []
        content.append({"type": "text", "text": prompt})
        text = self._processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [image]
        inputs = self._processor(**kwargs).to(self._device)
        gen: dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": temperature > 0}
        if temperature > 0:
            gen.update(temperature=float(temperature), top_p=0.9)
            if seed is not None:
                torch.manual_seed(int(seed))
        with torch.inference_mode():
            output = self._model.generate(**inputs, **gen)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return str(self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]).strip()


# ---------------------------------------------------------------------------- generation


def question_prompt(summary: str, style: Style) -> str:
    """The stage 2 prompt: the summary and the style, nothing from the page itself."""
    return QUESTION_PROMPT.format(type_text=QUERY_TYPES[style.type], format_text=QUERY_FORMATS[style.format],
                                  summary=summary.strip())


def is_yes(reply: str | None) -> bool:
    return bool(reply) and reply.strip().lower().lstrip("*\"'` ").startswith("yes")


def generate_vlm_probes(pages: Sequence[Any], *, backend: VlmBackend | None = None, model: str = DEFAULT_MODEL,
                        images: Any = None, texts: Mapping[str, str] | None = None,
                        aliases: Mapping[str, str] | None = None, seed: int = 0, max_pages: int = 300,
                        per_page: int = 1, max_overlap: float | None = 0.6, max_probes: int | None = None,
                        check_answerable: bool = True,
                        max_attempts: int = 2, max_side: int = 1280, type_weights: Mapping[str, int] | None = None,
                        device: str = "auto", dtype: str = "auto",
                        out: Callable[[str], None] | None = say,
                        progress: Callable[[int, int], None] | None = None) -> ProbeBatch:
    """Make practice questions with a local VLM, two blind stages per question.

    ``pages`` are page records (``id``, ``text``, ``n_chars``, ``blank``, ``file``, ``page``); ``images`` is a
    mapping (page id -> image), a callable (page -> image) or None to re-render from the file. ``texts`` (page id
    -> text, such as OCR text) gives the overlap cap text for scanned pages. ``backend`` replaces the built-in
    Hugging Face model (any object with ``generate(prompt, image=None, *, max_new_tokens, seed, temperature)``).

    A retry (``max_attempts``) samples with a seed derived from ``seed``, the page and the attempt, so the same
    ``seed`` gives the same pages, the same type plan and the same sampling seeds. GPU kernels can still differ
    in the last digit between machines, so identical text is not promised across hardware.
    """
    per_page = max(1, min(int(per_page), 4))
    emit = out if out is not None else (lambda _text: None)
    chosen = select_pages(pages, max_pages=max_pages, seed=seed, predicate=lambda p: not is_blank(p))
    styles = plan_styles(len(chosen) * per_page, seed, type_weights)
    owns_backend = backend is None
    if backend is None:
        backend = HFVlmBackend(model, device=device, dtype=dtype)
    info = VLM_MODELS.get(getattr(backend, "model_id", model))
    resolved_device = device
    if isinstance(backend, HFVlmBackend):
        resolved_device = backend.resolve_device()
    emit(str(estimate_block(len(chosen), resolved_device)))

    by_id = {page_id(p): p for p in pages}
    all_texts = {pid: page_text(p, texts) for pid, p in by_id.items()}
    known_texts = {k: v for k, v in all_texts.items() if v.strip()}
    index = WordIndex(known_texts) if known_texts else None
    rejected = Rejections()
    seen: set[str] = set()
    made: list[tuple[str, Any]] = []
    no_text = 0
    t0 = time.perf_counter()
    try:
        for n, page in enumerate(chosen):
            if progress is not None:
                progress(n, len(chosen))
            pid = page_id(page)
            image = prepare_image(get_image(page, images), max_side)
            summary = backend.generate(SUMMARY_PROMPT, image, max_new_tokens=256)
            if not summary or not summary.strip():
                rejected.add("no summary")
                continue
            text = all_texts.get(pid, "")
            if max_overlap is not None and not text.strip():
                no_text += 1
            for j in range(per_page):
                style = styles[n * per_page + j]
                probe = _one_question(backend, page, image, summary, style, text=text, index=index, aliases=aliases,
                                      seed=seed, slot=n * per_page + j, max_attempts=max_attempts,
                                      max_overlap=max_overlap, check_answerable=check_answerable, seen=seen,
                                      rejected=rejected, by_id=by_id)
                if probe is not None:
                    made.append((style.type, probe))
        if progress is not None:
            progress(len(chosen), len(chosen))
    finally:
        seconds = time.perf_counter() - t0
        if owns_backend and hasattr(backend, "close"):
            backend.close()
    batch = ProbeBatch(
        tier="vlm", probes=apply_quotas(made, len(made) if max_probes is None else max_probes, type_weights, seed),
        n_pages=len(chosen),
        rejected=rejected.as_dict(), usage={"seconds": round(seconds, 1)},
        config={"tier": "vlm", "model": getattr(backend, "model_id", model),
                "revision": getattr(backend, "revision", info.revision if info else None),
                "licence": info.licence if info else None, "seed": seed, "per_page": per_page,
                "max_overlap": max_overlap, "check_answerable": check_answerable, "max_attempts": max_attempts,
                "prompt_hash": prompt_hash(SUMMARY_PROMPT, QUESTION_PROMPT, ANSWERABLE_PROMPT)},
    )
    if no_text:
        batch.notes.append(f"the overlap cap could not be applied to {no_text} pages without any text; "
                           "run OCR beforehand and pass texts= to apply it")
    batch.report = _report(batch, getattr(backend, "model_id", model), info)
    emit(str(batch.report))
    return batch


def _one_question(backend: VlmBackend, page: Any, image: Any, summary: str, style: Style, *, text: str,
                  index: WordIndex | None, aliases: Mapping[str, str] | None, seed: int, slot: int,
                  max_attempts: int, max_overlap: float | None, check_answerable: bool, seen: set[str],
                  rejected: Rejections, by_id: Mapping[str, Any]) -> Any:
    """Stage 2 (blind) and the filters for one question; retries with a sampling seed when a filter drops it."""
    pid = page_id(page)
    for attempt in range(max(1, max_attempts)):
        temperature = 0.0 if attempt == 0 else 0.7
        sampling_seed = None if attempt == 0 else int(stable_id("s", seed, pid, slot, attempt)[2:10], 16)
        # the question request carries no image: image=None, and the prompt holds only the summary
        reply = backend.generate(question_prompt(summary, style), None, max_new_tokens=64, seed=sampling_seed,
                                 temperature=temperature)
        question = clean_question(reply, style.format)
        trial_seen = set(seen)
        if not check_question(question, text=text, max_overlap=max_overlap, seen=trial_seen, rejected=rejected):
            continue
        assert question is not None
        lexical: list[str] = []
        if index is not None:
            found = index.holding(question)
            if found is None:
                rejected.add("matches many pages")
                continue
            lexical = [f for f in found if f in by_id]
        if check_answerable and not is_yes(backend.generate(ANSWERABLE_PROMPT.format(question=question), image,
                                                            max_new_tokens=8)):
            rejected.add("not answerable from the page")
            continue
        seen.update(trial_seen)
        return make_probe(id=stable_id("vlm", pid, question), text=question, family=f"vlm:{style.type}", page_id=pid,
                          relevant=relevant_set(pid, aliases, lexical))
    return None


def _report(batch: ProbeBatch, model: str, info: VlmInfo | None) -> Block:
    block = Block([], title=f"Practice questions from a local VLM: {model}")
    block.add("Pages", "{0} pages read", claim(batch.n_pages, "counted"))
    block.add("Questions", "{0} questions kept on {1} pages", claim(len(batch.probes), "counted"),
              claim(len({p.page_id for p in batch.probes}), "counted"))
    add_counts(block, "Types", {family.split(":", 1)[1]: n for family, n in batch.families().items()})
    if batch.rejected:
        add_counts(block, "Dropped", batch.rejected, lead="questions were dropped")
    block.add("Time", "{0} on this machine", claim(duration(batch.usage.get("seconds", 0.0)), "counted"))
    if info is not None:
        block.add("Model", "{0}, licence {1}, checked {2}", lit(info.model_id), lit(info.licence), lit(info.checked))
    block.add(CONT, "The model wrote each question from a summary without seeing the page, but the questions are still "
                  "easier than real ones. They can only move a layout to a safer one and never confirm the target.")
    block.next = "give them to the practice check with the other questions"
    return block


__all__ = [
    "DEFAULT_MODEL",
    "VLM_MODELS",
    "HFVlmBackend",
    "VlmBackend",
    "VlmInfo",
    "estimate",
    "estimate_block",
    "generate_vlm_probes",
    "install_line",
    "question_prompt",
]
