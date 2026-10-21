"""Tier C practice questions: a hosted LLM writes them (``probes="api"``, ``allow_upload=True``, extra ``easy-gen``).

This is the only tier that sends anything off the machine, so the rules are strict and enforced in code:

* nothing is sent unless the caller passed ``allow_upload=True``; without it :class:`UploadNotAllowed`
  is raised before any page is rendered or any client is built;
* before any request the plan is printed: every page that would be sent (grouped by file), the
  number of requests, the tokens counted from the images that would be sent, and the cost at a dated
  price record. ``dry_run=True`` stops there and sends nothing, with or without ``allow_upload``;
* only page images and the instructions leave the machine, never file names, page ids or the text
  layer (images are labelled "Image 1", "Image 2");
* ``max_cost`` refuses a run whose upper cost bound is above it, and ``confirm(plan)`` can say no.

How a question is made (contrastive, FINAL_DESIGN 5.3.2). For a page, the page and its two nearest
neighbours are sent and the model writes a question that the page answers and the neighbours do not.
A second request shows the model only the neighbours and the question, and any neighbour that also
answers is added to the question's relevant set. So a question that two pages answer is scored as
having two right answers rather than counting the second as a miss.

Tokens and cost. An image costs ceil(width/28) x ceil(height/28) tokens after it is scaled to fit the
model's limits (the provider's documented rule, in ``presets/api_prices.json``); we apply it to the
images we would send, after our own downscale to ``max_side`` pixels. Text is counted at a
conservative characters-per-token figure. The cost is arithmetic on those counts and the recorded
prices, so it is shown as ``[estimate: prices recorded <date>; check the provider's page]``; the tokens
the provider reports afterwards are shown as counted.

Provider neutral: pass ``generate(prompt, images) -> str`` (images are PIL images; encode them as your
provider wants). The built-in backend is the Anthropic Messages API (default model ``claude-haiku-5-5``),
with the ``anthropic`` SDK imported when it is used.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from ._claim import Block, claim, lit
from ._probes_common import (
    CONT,
    DEFAULT_TYPE_WEIGHTS,
    QUERY_FORMATS,
    QUERY_TYPES,
    STOPWORDS,
    ProbeBatch,
    Rejections,
    Style,
    WordIndex,
    add_counts,
    apply_quotas,
    check_question,
    clean_question,
    downscaled_size,
    get_image,
    group_names,
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
from ._types import EasyError, MissingExtra

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_PROVIDER = "anthropic"

#: tokens a reply is expected to use (a question; a list of numbers), and the cap we ask for
EXPECTED_OUTPUT_TOKENS = {"write": 80, "judge": 20}
MAX_OUTPUT_TOKENS = 1024

#: stop after this many failed requests in a row (a wrong key fails every request)
MAX_CONSECUTIVE_ERRORS = 4

WRITE_PROMPT = (
    "You are helping to test a document search system. {n_images} images follow. Image 1 is the target page. "
    "{others}"
    "Write one search query that the target page answers and that the other images do not answer. "
    "The query {type_text}. Write it as {format_text}. "
    "A person types it into a search box to find the target page in a large collection, so it must make sense on its "
    "own and must not mention images, pages, numbers of images or this task. "
    "Use your own words where you can instead of copying long phrases from the page. "
    "Reply with the query only, on one line."
)
_OTHERS_TEXT = "The other images are different pages from the same collection. "
_NO_OTHERS_TEXT = ""

JUDGE_PROMPT = (
    "Search query: {question}\n"
    "{n_images} images follow, numbered 1 to {n_images}. For each image, decide whether it holds enough information "
    "to answer the query. Reply with a JSON list of the numbers of the images that answer it, for example [2], "
    "or [] if none does. Reply with the list only."
)


class UploadNotAllowed(EasyError, PermissionError):
    """A run that would send pages off the machine, without ``allow_upload=True``."""


class ApiProbeError(EasyError, RuntimeError):
    """The API tier stopped before sending (cost above ``max_cost``) or while sending (repeated failures)."""


# ---------------------------------------------------------------------------- prices and tokens


@dataclass(frozen=True)
class Price:
    """One dated price record (``presets/api_prices.json``). USD per million tokens."""

    provider: str
    model: str
    input_per_mtok: float
    output_per_mtok: float
    recorded: str
    source_url: str
    long_prompt_tokens: int | None = None
    long_input_per_mtok: float | None = None
    long_output_per_mtok: float | None = None
    image_patch_px: int = 28
    image_max_edge_px: int = 1568
    image_max_tokens: int = 1568
    chars_per_token: float = 3.0

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Price:
        known = set(cls.__dataclass_fields__)
        required = ("provider", "model", "input_per_mtok", "output_per_mtok", "recorded", "source_url")
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"a price record needs {', '.join(required)}; missing {', '.join(missing)}. "
                             "Copy the figures and the date from the provider's pricing page.")
        return cls(**{k: v for k, v in row.items() if k in known})

    @property
    def basis(self) -> str:
        """The source text of every estimate made from this record."""
        return f"prices recorded {self.recorded}; check the provider's page"

    def call_cost(self, input_tokens: int, output_tokens: int) -> float:
        """USD for one request: the long-prompt rates apply when the prompt is over the long-prompt threshold."""
        long_ = self.long_prompt_tokens is not None and input_tokens > self.long_prompt_tokens
        inp = self.long_input_per_mtok if long_ and self.long_input_per_mtok is not None else self.input_per_mtok
        out = self.long_output_per_mtok if long_ and self.long_output_per_mtok is not None else self.output_per_mtok
        return (input_tokens * inp + output_tokens * out) / 1e6


def load_prices() -> list[Price]:
    """Every record in ``presets/api_prices.json``."""
    text = resources.files("optivision").joinpath("presets", "api_prices.json").read_text(encoding="utf-8")
    return [Price.from_dict(row) for row in json.loads(text)["prices"]]


def price_for(provider: str, model: str) -> Price | None:
    for row in load_prices():
        if row.provider == provider and row.model == model:
            return row
    return None


def image_tokens(width: int, height: int, price: Price | None = None) -> int:
    """Tokens of one image under the provider's patch rule: ceil(w/28) x ceil(h/28) after scaling to the limits.

    Scaling keeps the aspect ratio and finds the largest size where neither side (rounded up to a whole
    patch) exceeds the edge limit and the patch count stays within the token limit.
    """
    patch = price.image_patch_px if price else 28
    max_edge = price.image_max_edge_px if price else 1568
    max_tokens = price.image_max_tokens if price else 1568

    def tokens(w: int, h: int) -> int:
        return math.ceil(w / patch) * math.ceil(h / patch)

    def fits(w: int, h: int) -> bool:
        return math.ceil(w / patch) * patch <= max_edge and math.ceil(h / patch) * patch <= max_edge \
            and tokens(w, h) <= max_tokens

    if fits(width, height):
        return tokens(width, height)
    if height > width:
        return image_tokens(height, width, price)
    ratio = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / ratio), 1)):
            lo = mid
        else:
            hi = mid
    return tokens(lo, max(round(lo / ratio), 1))


def text_tokens(text: str, price: Price | None = None) -> int:
    """A conservative count of the tokens in ``text`` (characters / chars_per_token, rounded up)."""
    per = price.chars_per_token if price else 3.0
    return math.ceil(len(text) / per)


# ---------------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class PlannedCall:
    """One request: ``kind`` is ``"write"`` or ``"judge"``; ``sent`` lists the pages whose images go in it."""

    kind: str
    page_id: str
    sent: tuple[str, ...]
    input_tokens: int
    expected_output_tokens: int
    max_output_tokens: int


@dataclass
class ApiPlan:
    """What a run would send and what it would cost; built without sending anything."""

    provider: str
    model: str
    price: Price | None
    targets: list[str]
    calls: list[PlannedCall]
    sent_pages: list[str]
    cached_pages: list[str] = field(default_factory=list)
    max_side: int = 1568
    seed: int = 0

    @property
    def n_requests(self) -> int:
        return len(self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def expected_output_tokens(self) -> int:
        return sum(c.expected_output_tokens for c in self.calls)

    @property
    def max_output_tokens(self) -> int:
        return sum(c.max_output_tokens for c in self.calls)

    def _cost(self, output: str) -> float | None:
        if self.price is None:
            return None
        return sum(self.price.call_cost(c.input_tokens, getattr(c, output)) for c in self.calls)

    @property
    def cost_expected(self) -> float | None:
        return self._cost("expected_output_tokens")

    @property
    def cost_upper(self) -> float | None:
        """Cost if every reply used its full output cap; an upper bound at the recorded prices."""
        return self._cost("max_output_tokens")

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "model": self.model, "targets": list(self.targets),
            "sent_pages": list(self.sent_pages), "cached_pages": list(self.cached_pages),
            "n_requests": self.n_requests, "input_tokens": self.input_tokens,
            "expected_output_tokens": self.expected_output_tokens, "max_output_tokens": self.max_output_tokens,
            "cost_expected_usd": self.cost_expected, "cost_upper_usd": self.cost_upper,
            "price_recorded": self.price.recorded if self.price else None,
            "price_source": self.price.source_url if self.price else None,
        }

    def block(self, *, sending: bool = False, max_name_lines: int = 40) -> Block:
        """The plan as the user sees it: pages named, requests, tokens, cost, then what to do next."""
        where = f"{self.provider} ({self.model})"
        block = Block([], title=f"Practice questions from an API model: {where}")
        if not self.sent_pages:
            block.add("Upload", "no pages need to be sent")
            if self.cached_pages:
                block.add("Reused", "{0} pages already have saved answers and are not sent again",
                          claim(len(self.cached_pages), "counted"))
            block.next = "nothing leaves the machine for this run"
            return block
        block.add("Upload", "{0} pages would leave this machine as images, in {1} requests to {2}",
                  claim(len(self.sent_pages), "counted"), claim(self.n_requests, "counted"), lit(where))
        if self.cached_pages:
            block.add("Reused", "{0} pages already have saved answers and are not sent again",
                      claim(len(self.cached_pages), "counted"))
        names = group_names(self.sent_pages)
        shown = names[:max_name_lines]
        for i, line in enumerate(shown):
            block.add("Pages" if i == 0 else CONT, "{0}", lit(line))
        if len(names) > len(shown):
            block.add(CONT, "and {0} more files; the full list is in plan.sent_pages",
                      claim(len(names) - len(shown), "counted"))
        block.add("Not sent", "file names, page ids and text layers stay here; only the page images and the "
                              "instructions are sent")
        block.add("Tokens", "{0} tokens in, at most {1} out", claim(self.input_tokens, "estimate", _TOKENS_BASIS),
                  claim(self.max_output_tokens, "estimate", _TOKENS_BASIS))
        if self.price is None:
            block.add("Cost", "not known: no price is recorded for {0}; pass price= to see a figure, or check the "
                              "provider's page", lit(where))
        else:
            block.add("Cost", "expected {0}, at most {1}", claim(self.cost_expected, "estimate", self.price.basis,
                                                              text=_money(self.cost_expected)),
                      claim(self.cost_upper, "estimate", self.price.basis, text=_money(self.cost_upper)))
        if sending:
            block.next = "sending now; stop the run to cancel before any request is made"
        else:
            block.next = "to send these pages call again with allow_upload=True; change max_pages to send fewer"
        return block

    def __str__(self) -> str:
        return str(self.block())


_TOKENS_BASIS = ("image tokens by the provider's published rule on the images we would send, text at a conservative "
                 "figure; the provider's own count will differ")


def _money(x: float | None) -> str:
    """Dollars rounded up to the cent; amounts under a cent read "under $0.01"."""
    if x is None:
        return "unknown"
    if x < 0.01:
        return "under $0.01"
    return f"${math.ceil(x * 100 - 1e-9) / 100:.2f}"


# ---------------------------------------------------------------------------- neighbours


def _tfidf_neighbours(ids: Sequence[str], texts: Mapping[str, str], targets: Sequence[str], k: int,
                      exclude: Mapping[str, set[str]]) -> dict[str, list[str]]:
    """Top ``k`` pages by TF-IDF cosine over the page texts, for each target; pages with no text get none."""
    token = re.compile(r"[^\W_]+", re.UNICODE)
    docs: dict[str, Counter[str]] = {}
    df: Counter[str] = Counter()
    for pid in ids:
        words = [w for w in token.findall(texts.get(pid, "").lower()) if len(w) >= 2 and w not in STOPWORDS]
        if len(words) < 5:
            continue
        docs[pid] = Counter(words)
        df.update(docs[pid].keys())
    n = max(len(docs), 1)
    vectors: dict[str, dict[str, float]] = {}
    postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for pid, tf in docs.items():
        weights = {w: c * (math.log((n + 1) / (df[w] + 1)) + 1.0) for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
        vectors[pid] = {w: v / norm for w, v in weights.items()}
        for w, v in vectors[pid].items():
            postings[w].append((pid, v))
    out: dict[str, list[str]] = {}
    for pid in targets:
        vec = vectors.get(pid)
        if vec is None:
            out[pid] = []
            continue
        scores: dict[str, float] = defaultdict(float)
        for w, v in vec.items():
            for other, v2 in postings[w]:
                scores[other] += v * v2
        banned = exclude.get(pid, {pid}) | {pid}
        ranked = sorted(((s, o) for o, s in scores.items() if o not in banned and s > 0), key=lambda t: (-t[0], t[1]))
        out[pid] = [o for _, o in ranked[:k]]
    return out


def _adjacent(order: Sequence[str], pid: str, k: int, banned: set[str]) -> list[str]:
    """Pages nearest to ``pid`` in folder order (the same file's neighbours come next to it in a normal listing)."""
    try:
        i = order.index(pid)
    except ValueError:
        return []
    found: list[str] = []
    for step in range(1, len(order)):
        for j in (i + step, i - step):
            if 0 <= j < len(order) and order[j] not in banned and order[j] not in found:
                found.append(order[j])
                if len(found) == k:
                    return found
    return found


def find_neighbours(pages: Sequence[Any], targets: Sequence[Any], *, k: int = 2,
                    neighbours: Mapping[str, Sequence[str]] | Callable[[str, int], Sequence[str]] | None = None,
                    texts: Mapping[str, str] | None = None,
                    aliases: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    """The ``k`` nearest other pages for each target page id.

    Order of preference: the caller's ``neighbours`` (a mapping page id -> ids, or a function
    ``(page_id, k) -> ids``, for example from the model's own page vectors); TF-IDF cosine over the page
    text (text layer or OCR); then the pages next to it in folder order. Pages that are the target's
    aliases (identical renders), blank, or not in ``pages`` are never used.
    """
    usable = [p for p in pages if not is_blank(p)]
    order = [page_id(p) for p in usable]
    known = set(order)
    full_texts = {page_id(p): page_text(p, texts) for p in usable}
    same: dict[str, set[str]] = {}
    for p in targets:
        pid = page_id(p)
        same[pid] = set(relevant_set(pid, aliases))
    tids = [page_id(p) for p in targets]
    by_text = _tfidf_neighbours(order, full_texts, tids, k, same)
    out: dict[str, list[str]] = {}
    for pid in tids:
        chosen: list[str] = []
        supplied: Sequence[str] = ()
        if callable(neighbours):
            supplied = neighbours(pid, k)
        elif neighbours is not None:
            supplied = neighbours.get(pid, ())
        for source in (supplied, by_text.get(pid, ())):
            for o in source:
                if o in known and o not in same[pid] and o not in chosen and len(chosen) < k:
                    chosen.append(o)
        if len(chosen) < k:
            chosen += _adjacent(order, pid, k - len(chosen), same[pid] | set(chosen))
        out[pid] = chosen[:k]
    return out


# ---------------------------------------------------------------------------- backends


def jpeg_base64(image: Any, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class AnthropicBackend:
    """``generate(prompt, images) -> str`` on the Anthropic Messages API.

    The ``anthropic`` SDK is imported when a request is made; if it is missing, :class:`MissingExtra`
    carries the install line. Credentials come from the SDK's usual places (``ANTHROPIC_API_KEY`` or a
    profile). Thinking is left at the model's default and ``effort`` is ``"low"``, because the job is to write
    one line; no sampling parameters are sent (this model family rejects non-default ones). ``usage`` holds
    the token counts the API reports, summed over requests.
    """

    provider = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, *, api_key: str | None = None, client: Any = None,
                 max_tokens: int = MAX_OUTPUT_TOKENS, effort: str | None = "low") -> None:
        self.model = model
        self.max_tokens = int(max_tokens)
        self.effort = effort
        self._api_key = api_key
        self._client = client
        self.usage: dict[str, int] = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "refusals": 0}

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise MissingExtra('the API tier needs the anthropic SDK. Install it with: pip install '
                                   '"optivision-rag[easy-gen]"') from exc
            self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        return self._client

    def ready(self) -> None:
        """Build the client now (raises :class:`MissingExtra` or the SDK's own error), so a run fails before sending."""
        self._get_client()

    def __call__(self, prompt: str, images: Sequence[Any]) -> str:
        client = self._get_client()
        content: list[dict[str, Any]] = []
        for i, image in enumerate(images, 1):
            content.append({"type": "text", "text": f"Image {i}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                       "data": jpeg_base64(image)}})
        content.append({"type": "text", "text": prompt})
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                  "messages": [{"role": "user", "content": content}]}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        response = client.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        self.usage["requests"] += 1
        self.usage["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        self.usage["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
        if getattr(response, "stop_reason", None) == "refusal":
            self.usage["refusals"] += 1
            return ""
        return "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", None) == "text")


# ---------------------------------------------------------------------------- planning


def _write_prompt(style: Style, n_neighbours: int) -> str:
    return WRITE_PROMPT.format(n_images=1 + n_neighbours, others=_OTHERS_TEXT if n_neighbours else _NO_OTHERS_TEXT,
                               type_text=QUERY_TYPES[style.type], format_text=QUERY_FORMATS[style.format])


def _judge_prompt(question: str, n_neighbours: int) -> str:
    return JUDGE_PROMPT.format(question=question, n_images=n_neighbours)


class _Sizes:
    """Image sizes after our downscale, rendered once per page while planning (the images are not kept)."""

    def __init__(self, by_id: Mapping[str, Any], images: Any, max_side: int) -> None:
        self._by_id, self._images, self._max_side = by_id, images, max_side
        self._memo: dict[str, tuple[int, int]] = {}

    def __call__(self, pid: str) -> tuple[int, int]:
        if pid not in self._memo:
            width, height = get_image(self._by_id[pid], self._images).size
            self._memo[pid] = downscaled_size(width, height, self._max_side)
        return self._memo[pid]


@dataclass
class _Item:
    page: Any
    style: Style
    neighbours: list[str]
    key: str


def _cache_key(pid: str, neighbours: Sequence[str], model: str, style: Style, seed: int, phash: str) -> str:
    return stable_id("k", pid, ",".join(neighbours), model, style.type, style.format, seed, phash)


def _load_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if path is None or not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            out[str(row["key"])] = row
        except (ValueError, KeyError, TypeError):
            continue  # a half-written last line after a crash
    return out


def plan_api_probes(pages: Sequence[Any], *, provider: str | None = None, model: str | None = None,
                    generate: Callable[[str, Sequence[Any]], str] | None = None,
                    price: Price | Mapping[str, Any] | None = None, images: Any = None,
                    texts: Mapping[str, str] | None = None, aliases: Mapping[str, str] | None = None,
                    neighbours: Mapping[str, Sequence[str]] | Callable[[str, int], Sequence[str]] | None = None,
                    n_neighbours: int = 2, seed: int = 0, max_pages: int = 300, max_side: int = 1568,
                    type_weights: Mapping[str, int] | None = None, cache_path: str | Path | None = None) -> ApiPlan:
    """The requests a run would make, built without sending anything. Pages are rendered here to size them."""
    plan, _ = _plan(pages, provider=provider, model=model, generate=generate, price=price, images=images, texts=texts,
                    aliases=aliases, neighbours=neighbours, n_neighbours=n_neighbours, seed=seed, max_pages=max_pages,
                    max_side=max_side, type_weights=type_weights, cache_path=cache_path)
    return plan


def _resolve_price(provider: str, model: str, price: Price | Mapping[str, Any] | None) -> Price | None:
    if isinstance(price, Price):
        return price
    if price is not None:
        row = dict(price)
        row.setdefault("provider", provider)
        row.setdefault("model", model)
        return Price.from_dict(row)
    return price_for(provider, model)


def _plan(pages: Sequence[Any], *, provider: str | None, model: str | None, generate: Any, price: Any, images: Any,
          texts: Mapping[str, str] | None, aliases: Mapping[str, str] | None, neighbours: Any, n_neighbours: int,
          seed: int, max_pages: int, max_side: int, type_weights: Mapping[str, int] | None,
          cache_path: str | Path | None) -> tuple[ApiPlan, list[_Item]]:
    provider = provider or ("custom" if generate is not None else DEFAULT_PROVIDER)
    model = model or (DEFAULT_MODEL if generate is None else "your own generate function")
    resolved = _resolve_price(provider, model, price)
    targets = select_pages(pages, max_pages=max_pages, seed=seed, predicate=lambda p: not is_blank(p))
    styles = plan_styles(len(targets), seed, type_weights)
    near = find_neighbours(pages, targets, k=n_neighbours, neighbours=neighbours, texts=texts, aliases=aliases)
    phash = prompt_hash(WRITE_PROMPT, JUDGE_PROMPT)
    cache = _load_cache(Path(cache_path) if cache_path else None)
    by_id = {page_id(p): p for p in pages}
    sizes = _Sizes(by_id, images, max_side)
    items = [_Item(p, s, near[page_id(p)], _cache_key(page_id(p), near[page_id(p)], model, s, seed, phash))
             for p, s in zip(targets, styles, strict=True)]
    calls: list[PlannedCall] = []
    sent: dict[str, None] = {}
    cached: list[str] = []
    for item in items:
        pid = page_id(item.page)
        if item.key in cache:
            cached.append(pid)
            continue
        group = (pid, *item.neighbours)
        prompt = _write_prompt(item.style, len(item.neighbours))
        in_tokens = sum(image_tokens(*sizes(g), resolved) for g in group) + text_tokens(prompt, resolved) \
            + 6 * len(group)
        calls.append(PlannedCall("write", pid, group, in_tokens, EXPECTED_OUTPUT_TOKENS["write"], MAX_OUTPUT_TOKENS))
        sent.update(dict.fromkeys(group))
        if item.neighbours:
            judge_prompt = _judge_prompt("x" * 120, len(item.neighbours))
            j_tokens = sum(image_tokens(*sizes(g), resolved) for g in item.neighbours) \
                + text_tokens(judge_prompt, resolved) + 6 * len(item.neighbours)
            calls.append(PlannedCall("judge", pid, tuple(item.neighbours), j_tokens, EXPECTED_OUTPUT_TOKENS["judge"],
                                     MAX_OUTPUT_TOKENS))
    plan = ApiPlan(provider=provider, model=model, price=resolved, targets=[page_id(i.page) for i in items],
                   calls=calls, sent_pages=list(sent), cached_pages=cached, max_side=max_side, seed=seed)
    return plan, items


# ---------------------------------------------------------------------------- the run


def parse_judged(reply: str, k: int) -> list[int]:
    """The 1-based image numbers a judge reply names, within ``1..k``; an empty list when none answers."""
    match = re.search(r"\[([^\]]*)\]", reply or "")
    if match:
        numbers = re.findall(r"\d+", match.group(1))
    elif re.search(r"\bnone\b|\bno image", (reply or "").lower()):
        numbers = []
    else:
        numbers = re.findall(r"\b\d+\b", reply or "")
    return sorted({int(n) for n in numbers if 1 <= int(n) <= k})


def _is_fatal(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in (401, 403, 404)


class _ImageCache:
    """The last few prepared images, because neighbouring targets share neighbours."""

    def __init__(self, by_id: Mapping[str, Any], images: Any, max_side: int, size: int = 24) -> None:
        self._by_id, self._images, self._max_side, self._size = by_id, images, max_side, size
        self._items: OrderedDict[str, Any] = OrderedDict()

    def __call__(self, pid: str) -> Any:
        if pid in self._items:
            self._items.move_to_end(pid)
            return self._items[pid]
        image = prepare_image(get_image(self._by_id[pid], self._images), self._max_side)
        self._items[pid] = image
        while len(self._items) > self._size:
            self._items.popitem(last=False)
        return image


def generate_api_probes(pages: Sequence[Any], *, generate: Callable[[str, Sequence[Any]], str] | None = None,
                        provider: str | None = None, model: str | None = None,
                        price: Price | Mapping[str, Any] | None = None, allow_upload: bool = False,
                        dry_run: bool = False, images: Any = None, texts: Mapping[str, str] | None = None,
                        aliases: Mapping[str, str] | None = None,
                        neighbours: Mapping[str, Sequence[str]] | Callable[[str, int], Sequence[str]] | None = None,
                        n_neighbours: int = 2, seed: int = 0, max_pages: int = 300, max_side: int = 1568,
                        max_overlap: float | None = 0.6, min_cap_words: int = 4, max_probes: int | None = None,
                        type_weights: Mapping[str, int] | None = None,
                        max_cost: float | None = None, confirm: Callable[[ApiPlan], bool] | None = None,
                        cache_path: str | Path | None = None, out: Callable[[str], None] | None = say,
                        progress: Callable[[int, int], None] | None = None) -> ProbeBatch:
    """Make practice questions with a hosted model, after naming the pages and the cost.

    ``pages`` are page records (``id``, ``text``, ``n_chars``, ``blank``, ``file``, ``page``). ``generate`` is your
    own ``(prompt, images) -> str``; without it the built-in Anthropic backend is used. Whatever sends, it counts
    as an upload: ``allow_upload=True`` is required, and it is a refusal otherwise. ``dry_run=True`` prints the
    plan (pages, requests, tokens, cost) and returns without sending.

    ``texts`` (page id -> text, e.g. OCR text) feeds the overlap cap and the neighbour search; ``neighbours`` lets
    you supply the nearest pages yourself (e.g. from page vectors). ``cache_path`` is a JSON-lines file: answers
    already paid for are reused on a re-run and not sent again. ``max_probes`` keeps at most that many questions,
    the type plan deciding which. ``max_cost`` (USD) refuses a run whose upper cost bound is higher, or whose cost
    is unknown. ``confirm(plan)`` returning False cancels before anything is sent.
    """
    if not allow_upload and not dry_run:
        raise UploadNotAllowed(
            "The API tier sends page images to a hosted model, and allow_upload=True was not given, so nothing was "
            "sent. Use dry_run=True to see the pages and the cost before sending.")
    emit = out if out is not None else (lambda _text: None)
    plan, items = _plan(pages, provider=provider, model=model, generate=generate, price=price, images=images,
                        texts=texts, aliases=aliases, neighbours=neighbours, n_neighbours=n_neighbours, seed=seed,
                        max_pages=max_pages, max_side=max_side, type_weights=type_weights, cache_path=cache_path)
    config = {"tier": "api", "provider": plan.provider, "model": plan.model, "seed": seed, "n_neighbours": n_neighbours,
              "max_overlap": max_overlap, "max_side": max_side,
              "prompt_hash": prompt_hash(WRITE_PROMPT, JUDGE_PROMPT),
              "price_recorded": plan.price.recorded if plan.price else None}
    batch = ProbeBatch(tier="api", probes=[], n_pages=len(items), config=config, plan=plan)
    if dry_run:
        batch.report = plan.block()
        batch.notes.append("dry run: nothing was sent")
        emit(str(batch.report))
        return batch
    if not plan.calls and not plan.cached_pages:
        batch.report = plan.block()
        emit(str(batch.report))
        return batch

    # the plan is printed, and the guards run, before any request is made
    emit(str(plan.block(sending=bool(plan.calls))))
    if max_cost is not None:
        upper = plan.cost_upper
        if upper is None or upper > max_cost:
            what = "cannot be priced" if upper is None else f"could cost {_money(upper)}"
            raise ApiProbeError(f"This run {what}, above max_cost={max_cost}; nothing was sent. "
                                "Lower max_pages or raise max_cost.")
    if confirm is not None and plan.calls and not confirm(plan):
        batch.notes.append("not sent: declined at the confirm step")
        batch.report = plan.block()
        return batch

    backend = generate
    if backend is None and plan.calls:
        backend = AnthropicBackend(plan.model)
        backend.ready()  # a missing SDK or key shows up here, before any page is rendered for sending
    by_id = {page_id(p): p for p in pages}
    fetch = _ImageCache(by_id, images, max_side)
    cache = _load_cache(Path(cache_path) if cache_path else None)
    sink = open(cache_path, "a", encoding="utf-8") if cache_path else None  # noqa: SIM115 - closed in finally
    all_texts = {pid: page_text(p, texts) for pid, p in by_id.items()}
    known_texts = {pid: t for pid, t in all_texts.items() if t.strip()}
    index = WordIndex(known_texts) if known_texts else None
    rejected = Rejections()
    no_text = 0
    seen: set[str] = set()
    made: list[tuple[str, Any]] = []
    errors = 0
    consecutive = 0
    last_error = ""
    try:
        for n, item in enumerate(items):
            if progress is not None:
                progress(n, len(items))
            pid = page_id(item.page)
            row = cache.get(item.key)
            if row is None:
                try:
                    row = _ask(backend, item, fetch)
                    consecutive = 0
                except MissingExtra:
                    raise
                except Exception as exc:  # a failed request costs one page, not the run
                    errors += 1
                    consecutive += 1
                    last_error = f"{type(exc).__name__}: {exc}"
                    if _is_fatal(exc) or consecutive >= MAX_CONSECUTIVE_ERRORS:
                        batch.notes.append(f"stopped after repeated failures; last error {last_error}")
                        break
                    continue
                row["key"], row["page_id"] = item.key, pid
                if sink is not None:
                    sink.write(json.dumps(row, sort_keys=True) + "\n")
                    sink.flush()
            question = clean_question(row.get("question"), item.style.format)
            if max_overlap is not None and not all_texts.get(pid, "").strip():
                no_text += 1
            if not check_question(question, text=all_texts.get(pid, ""), max_overlap=max_overlap, seen=seen,
                                  rejected=rejected, min_cap_words=min_cap_words):
                continue
            assert question is not None
            lexical: list[str] = []
            if index is not None:
                found = index.holding(question)
                if found is None:
                    rejected.add("matches many pages")
                    continue
                lexical = found
            judged = [pid_ for pid_ in row.get("judged", []) if pid_ in by_id]
            relevant = relevant_set(pid, aliases, [*judged, *lexical])
            probe = make_probe(id=stable_id("api", pid, question), text=question, family=f"api:{item.style.type}",
                               page_id=pid, relevant=relevant)
            made.append((item.style.type, probe))
        if progress is not None:
            progress(len(items), len(items))
    finally:
        if sink is not None:
            sink.close()
    batch.probes = apply_quotas(made, len(made) if max_probes is None else max_probes,
                                type_weights or DEFAULT_TYPE_WEIGHTS, seed)
    if no_text:
        batch.notes.append(f"the overlap cap could not be applied to {no_text} pages without any text; "
                           "run OCR beforehand and pass texts= to apply it")
    batch.rejected = rejected.as_dict()
    if errors:
        batch.rejected["request failed"] = errors
    usage = getattr(backend, "usage", None)
    batch.usage = dict(usage) if isinstance(usage, Mapping) else {}
    batch.report = _final_block(plan, batch, errors)
    emit(str(batch.report))
    return batch


def _ask(backend: Callable[[str, Sequence[Any]], str] | None, item: _Item,
         fetch: Callable[[str], Any]) -> dict[str, Any]:
    """The two requests for one page: write the question (target + neighbours), then judge the neighbours."""
    assert backend is not None
    pid = page_id(item.page)
    group = [fetch(pid), *[fetch(n) for n in item.neighbours]]
    question = clean_question(backend(_write_prompt(item.style, len(item.neighbours)), group), item.style.format)
    row: dict[str, Any] = {"question": question, "type": item.style.type, "format": item.style.format, "judged": []}
    if question and item.neighbours:
        reply = backend(_judge_prompt(question, len(item.neighbours)), group[1:])
        row["judged"] = [item.neighbours[i - 1] for i in parse_judged(reply, len(item.neighbours))]
    return row


def _final_block(plan: ApiPlan, batch: ProbeBatch, errors: int) -> Block:
    block = Block([], title=f"Practice questions from an API model: {plan.provider} ({plan.model})")
    block.add("Sent", "{0} pages were sent in {1} requests", claim(len(plan.sent_pages), "counted"),
              claim(plan.n_requests, "counted"))
    block.add("Questions", "{0} questions kept on {1} pages", claim(len(batch.probes), "counted"),
              claim(len({p.page_id for p in batch.probes}), "counted"))
    shared = sum(1 for p in batch.probes if len(p.relevant) > 1)
    block.add(CONT, "{0} of them are answered by more than one page, and all of those pages count as right",
              claim(shared, "counted"))
    if batch.rejected:
        add_counts(block, "Dropped", batch.rejected, lead="questions were dropped")
    used = batch.usage
    if used.get("input_tokens"):
        block.add("Tokens", "the provider counted {0} tokens in and {1} out",
                  claim(int(used["input_tokens"]), "counted"), claim(int(used.get("output_tokens", 0)), "counted"))
        if plan.price is not None:
            spent = plan.price.call_cost(int(used["input_tokens"]), int(used.get("output_tokens", 0)))
            block.add("Cost", "about {0} at the recorded prices",
                      claim(spent, "estimate", plan.price.basis, text=_money(spent)))
    block.add(CONT, "These questions are written from the pages, so they are easier than real ones. They can only "
                    "move a layout to a safer one and never confirm the target.")
    block.next = "give them to the practice check with the other questions"
    return block


__all__ = [
    "DEFAULT_MODEL",
    "AnthropicBackend",
    "ApiPlan",
    "ApiProbeError",
    "PlannedCall",
    "Price",
    "UploadNotAllowed",
    "find_neighbours",
    "generate_api_probes",
    "image_tokens",
    "load_prices",
    "parse_judged",
    "plan_api_probes",
    "price_for",
    "text_tokens",
]
