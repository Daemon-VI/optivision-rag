"""Late-interaction encoders without a patch grid: token vectors out, nothing else.

:class:`~optivision.encoders.colvlm.ColVLMEncoder` knows the patch grid of the ColPali,
ColQwen2, ColQwen2.5 and ColSmol processors, which the pixel-space stages need. The model
families here are loaded the way their own model cards load them and return only what every
vector-only stage needs: one ``[n_tokens, dim]`` float32 array per page and per query, with
the padding rows dropped.

Three encoders, one contract (``encode_images``, ``encode_queries``, ``info``):

``ColEngineTokenEncoder``
    Any ``Col*`` class of ``colpali-engine``, chosen by family name
    (:data:`COLENGINE_FAMILIES`): ``processor.process_images`` / ``process_queries`` and a
    forward pass that returns ``[batch, tokens, dim]``.
``RemoteCodeEncoder``
    A checkpoint that ships its own modelling code (``trust_remote_code``), driven by a
    :class:`Recipe` that names the processor or model methods of its model card. The remote
    code runs on load, so it is never loaded without ``trust_remote_code=True`` from the
    caller and a pinned revision.
``PyLateEncoder``
    Text ColBERT models through ``pylate.models.ColBERT``.

What "exists here" means: the ``tomoro`` recipe and the ``colqwen3.5`` family reproduce the
calls of ``reports/universal/wide/logs/*/enc_generic.py``, the script that encoded the two
measured models of those families on Kaggle T4s. Every other recipe and family is written
against the model card and has not been run by this repository; the registry
(:data:`optivision.adapters.MODELS`) says which is which, and
:func:`optivision.adapters.check_retriever` is the test to run before trusting any of them.

Nothing here chooses or alters compression settings by model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .colvlm import _from_pretrained, _load_model, _resolve_device, _resolve_dtype, place_model

__all__ = [
    "COLENGINE_FAMILIES",
    "RECIPES",
    "ColEngineTokenEncoder",
    "PyLateEncoder",
    "Recipe",
    "RemoteCodeEncoder",
]

#: family -> (model class, processor class) of ``colpali_engine.models``
COLENGINE_FAMILIES: dict[str, tuple[str, str]] = {
    "colpali": ("ColPali", "ColPaliProcessor"),
    "colidefics3": ("ColIdefics3", "ColIdefics3Processor"),
    "colqwen2": ("ColQwen2", "ColQwen2Processor"),
    "colqwen2.5": ("ColQwen2_5", "ColQwen2_5_Processor"),
    "colqwen2.5-omni": ("ColQwen2_5Omni", "ColQwen2_5OmniProcessor"),
    "colqwen3": ("ColQwen3", "ColQwen3Processor"),
    "colqwen3.5": ("ColQwen3_5", "ColQwen3_5Processor"),
    "colmodernvbert": ("ColModernVBert", "ColModernVBertProcessor"),
    "colgemma3": ("ColGemma3", "ColGemmaProcessor3"),
}

#: rows whose norm is below this are padding the model zeroed, not vectors
_ZERO_ROW = 1e-6


def _rows(emb: Any, mask: Any) -> list[np.ndarray]:
    """``[batch, tokens, dim]`` (or one item's ``[tokens, dim]``) to per-item float32 arrays.

    A row is dropped when the attention mask says it is padding, and when its norm is zero
    (these models multiply their output by the mask, so a padded row is exactly zero).
    """
    import torch

    emb = torch.as_tensor(emb).detach().to("cpu", torch.float32)  # numpy output too
    if emb.ndim == 2:
        emb = emb.unsqueeze(0)
    if emb.ndim != 3:
        raise ValueError(f"expected [batch, tokens, dim] embeddings, got shape {tuple(emb.shape)}")
    if not torch.isfinite(emb).all():
        raise ValueError("the model returned non-finite embeddings")
    if mask is not None:
        mask = torch.as_tensor(mask).detach().to("cpu")
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) != tuple(emb.shape[:2]):
            mask = None  # a mask of another length (prefix tokens removed by the model) is not applied
    out = []
    for i in range(emb.shape[0]):
        item = emb[i] if mask is None else emb[i][mask[i].bool()]
        item = item[item.norm(dim=-1) > _ZERO_ROW]
        out.append(np.ascontiguousarray(item.numpy()))
    return out


def _as_list(value: Any) -> list[np.ndarray]:
    """Per-item arrays from whatever a model method returned (list of tensors, tensor, arrays)."""
    if hasattr(value, "ndim") and getattr(value, "ndim", 0) == 3:
        return _rows(value, None)
    out = []
    for item in value:
        if hasattr(item, "detach"):
            item = item.detach().float().cpu().numpy()
        item = np.asarray(item, dtype=np.float32)
        if item.ndim != 2:
            raise ValueError(f"expected one [tokens, dim] array per item, got shape {item.shape}")
        if not np.isfinite(item).all():
            raise ValueError("the model returned non-finite embeddings")
        out.append(np.ascontiguousarray(item[np.linalg.norm(item, axis=1) > _ZERO_ROW]))
    return out


def _rgb(image: Any) -> Any:
    convert = getattr(image, "convert", None)
    return convert("RGB") if callable(convert) else image


def _resolve_revision(model_name: str, revision: str | None, known: str | None) -> str:
    """The commit to load. ``"latest"`` asks the Hub and is recorded; None means the known pin."""
    if revision is None:
        revision = known
    if revision is None:
        raise ValueError(
            f"{model_name}: pin a revision (pass revision='<commit hash>', or revision='latest' to "
            "load the current commit and record its hash). The checkpoint's own code runs on load.")
    if revision != "latest":
        return revision
    from huggingface_hub import model_info

    return str(model_info(model_name).sha)


def _adopt_model_placement(enc: Any, model: Any) -> None:
    """Take device and dtype from an injected model's parameters, so inputs go where the model is."""
    try:
        param = next(iter(model.parameters()))
    except Exception:  # noqa: BLE001 - an injected stand-in may have no parameters
        return
    device = getattr(param, "device", None)
    if device is not None:
        enc.device = str(device)
    if getattr(param, "dtype", None) is not None:
        enc.torch_dtype = param.dtype


class _TokenEncoder:
    """Shared state and reporting of the encoders below."""

    loader = ""
    checkpoint: str
    revision: str | None = None
    dim: int | None = None
    expected_dim: int | None = None
    device: str = "cpu"
    torch_dtype: Any = None
    loading_report: Any = None
    adapter_merge_check: Any = None
    placement: Any = None
    page_batch_size: int = 1
    query_batch_size: int = 1

    def _note_width(self, arrays: Sequence[np.ndarray]) -> None:
        for a in arrays:
            width = int(a.shape[1])
            if self.expected_dim is not None and width != self.expected_dim:
                raise RuntimeError(f"{self.checkpoint}: got {width}-d vectors, expected {self.expected_dim}")
            if self.dim is None:
                self.dim = width
            elif width != self.dim:
                raise RuntimeError(f"{self.checkpoint}: got {width}-d vectors after {self.dim}-d ones")

    def encode_images(self, images: Sequence[Any]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        step = self.page_batch_size
        for start in range(0, len(images), step):
            out.extend(self._images([_rgb(im) for im in images[start : start + step]]))
        self._note_width(out)
        return out

    def encode_queries(self, queries: Sequence[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        step = self.query_batch_size
        for start in range(0, len(queries), step):
            out.extend(self._queries([str(q) for q in queries[start : start + step]]))
        self._note_width(out)
        return out

    def info(self) -> dict[str, Any]:
        return {"model": self.checkpoint, "revision": self.revision, "dim": self.dim, "loader": self.loader,
                "device": self.device, "dtype": str(self.torch_dtype), "loading_report": self.loading_report,
                "adapter_merge_check": self.adapter_merge_check, "placement": self.placement}

    def _images(self, images: list[Any]) -> list[np.ndarray]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _queries(self, queries: list[str]) -> list[np.ndarray]:  # pragma: no cover - abstract
        raise NotImplementedError


class ColEngineTokenEncoder(_TokenEncoder):
    """A ``colpali-engine`` ``Col*`` model, by family name, without a patch grid.

    Full checkpoints load with every weight checked; an adapter-only repository is loaded as
    its base model plus the adapter, merged explicitly, and refused if the merge changed
    nothing (:func:`optivision.encoders.colvlm._load_model`). ``model`` and ``processor``
    inject already-built objects (that is how the tests run without weights).
    """

    def __init__(self, model_name: str, family: str, device: str = "auto", dtype: str = "auto",
                 multi_gpu: bool = False, expected_dim: int | None = None, page_batch_size: int = 1,
                 query_batch_size: int = 8, model: Any = None, processor: Any = None,
                 revision: str | None = None) -> None:
        if family not in COLENGINE_FAMILIES:
            raise ValueError(f"unknown colpali-engine family {family!r}; choose from {sorted(COLENGINE_FAMILIES)}")
        self.loader = f"colpali-engine:{family}"
        model_injected = model is not None
        self.family = family
        self.checkpoint = model_name
        self.revision = revision  # the commit loaded; info() and the adapter identity record it
        self.expected_dim = expected_dim
        self.page_batch_size = max(1, int(page_batch_size))
        self.query_batch_size = max(1, int(query_batch_size))
        pin: dict[str, Any] = {} if revision is None else {"revision": revision}
        if model is None or processor is None:
            try:
                from colpali_engine import models as cpm
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError("this model needs colpali-engine (pip install 'optivision-rag[vlm]')") from exc
            model_cls_name, proc_cls_name = COLENGINE_FAMILIES[family]
            missing = [n for n in (model_cls_name, proc_cls_name) if not hasattr(cpm, n)]
            if missing:
                raise ImportError(f"the installed colpali-engine has no {missing[0]}; the {family} family "
                                  "needs a newer release (pip install -U colpali-engine)")
            self.device = _resolve_device(device)
            self.torch_dtype = _resolve_dtype(dtype, self.device)
            if model is None:
                model = _load_model(getattr(cpm, model_cls_name), model_name, self.torch_dtype, **pin)
                self.loading_report = getattr(model, "loading_report", None)
                self.adapter_merge_check = getattr(model, "adapter_merge_check", None)
                unloaded = [n for n, p in model.named_parameters() if p.device.type == "meta"]
                if unloaded:
                    raise RuntimeError(f"{model_name}: {len(unloaded)} parameters never loaded (e.g. {unloaded[0]})")
                model, self.device, self.placement = place_model(model, self.device, multi_gpu)
                model = model.eval()
            if processor is None:
                processor = getattr(cpm, proc_cls_name).from_pretrained(model_name, **pin)
        if model_injected:
            _adopt_model_placement(self, model)
        self.model = model
        self.processor = processor

    def _forward(self, batch: Any) -> list[np.ndarray]:
        import torch

        batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in dict(batch).items()}
        with torch.inference_mode():
            out = self.model(**batch)
        return _rows(getattr(out, "embeddings", out), batch.get("attention_mask"))

    def _images(self, images: list[Any]) -> list[np.ndarray]:
        return self._forward(self.processor.process_images(images))

    def _queries(self, queries: list[str]) -> list[np.ndarray]:
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None  # the Qwen3-VL model cards reset this before text-only input
        return self._forward(self.processor.process_queries(queries))


@dataclass(frozen=True)
class Recipe:
    """How one family of remote-code checkpoints is driven, as its model card does it.

    ``style="processor"``: ``processor.<image_method>(...)`` and ``processor.<query_method>(...)``
    build a batch, the model is called on it, and the token vectors are the output's
    ``output`` attribute (the output itself when the attribute is absent). The image and
    query lists are passed under ``image_arg`` / ``query_arg``, or positionally when those
    are None.

    ``style="model"``: ``model.<image_method>(...)`` and ``model.<query_method>(...)`` take the
    lists directly and return per-item token vectors; ``image_kwargs`` / ``query_kwargs`` are
    the extra arguments of the card's example.
    """

    style: str
    image_method: str
    query_method: str
    image_arg: str | None = None
    query_arg: str | None = None
    image_kwargs: Mapping[str, Any] = field(default_factory=dict)
    query_kwargs: Mapping[str, Any] = field(default_factory=dict)
    processor_kwargs: Mapping[str, Any] = field(default_factory=dict)
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    output: str = "embeddings"
    needs_processor: bool = True
    exercised: bool = False  # True only if a run of this repository used these calls

    def __post_init__(self) -> None:
        if self.style not in ("processor", "model"):
            raise ValueError(f"style must be 'processor' or 'model', got {self.style!r}")


#: recipe name -> the calls of that family's model card
RECIPES: dict[str, Recipe] = {
    # TomoroAI/tomoro-colqwen3-embed-*: the calls of the measured run (enc_generic.py, loader "tomoro")
    "tomoro": Recipe(style="processor", image_method="process_images", query_method="process_texts",
                     image_arg="images", query_arg="texts",
                     processor_kwargs={"max_num_visual_tokens": 1280}, exercised=True),
    # The recipes below repeat the calls of a model card and have not been run by this repository.
    # processor.process_images(images) / processor.process_queries(queries), then model(**batch):
    # ibm-granite/granite-vision-3.3-2b-embedding, 5CD-AI/Vintern-Embedding-1B
    "processor": Recipe(style="processor", image_method="process_images", query_method="process_queries",
                        output=""),
    # webAI-Official/webAI-ColVec1-*: keyword arguments, queries under ``texts``
    "colvec": Recipe(style="processor", image_method="process_images", query_method="process_queries",
                     image_arg="images", query_arg="texts"),
    # jinaai/jina-embeddings-v4: single-vector by default; these arguments ask for token vectors
    "jina-v4": Recipe(style="model", image_method="encode_image", query_method="encode_text",
                      image_arg="images", query_arg="texts",
                      image_kwargs={"task": "retrieval", "return_multivector": True},
                      query_kwargs={"task": "retrieval", "prompt_name": "query", "return_multivector": True},
                      needs_processor=False),
    # nvidia/llama-nemoretriever-colembed-*-v1, nvidia/llama-nemotron-colembed-vl-3b-v2
    "nvidia-colembed": Recipe(style="model", image_method="forward_images", query_method="forward_queries",
                              needs_processor=False),
}


class RemoteCodeEncoder(_TokenEncoder):
    """A checkpoint that ships its own modelling code, driven by a :class:`Recipe`.

    ``trust_remote_code=True`` must be passed by the caller: loading runs Python from the
    model repository on this machine. The revision is pinned for the same reason (see
    :func:`_resolve_revision`); the commit that was loaded is in :meth:`info`.
    """

    def __init__(self, model_name: str, recipe: str | Recipe, revision: str | None = None,
                 trust_remote_code: bool = False, device: str = "auto", dtype: str = "auto",
                 multi_gpu: bool = False, expected_dim: int | None = None,
                 attn_implementation: str | None = "sdpa", page_batch_size: int = 1,
                 query_batch_size: int = 8, known_revision: str | None = None,
                 model: Any = None, processor: Any = None) -> None:
        if isinstance(recipe, str):
            if recipe not in RECIPES:
                raise ValueError(f"unknown recipe {recipe!r}; choose from {sorted(RECIPES)} or pass a Recipe")
            self.recipe_name, recipe = recipe, RECIPES[recipe]
        else:
            self.recipe_name = "custom"
        self.recipe = recipe
        self.loader = f"remote-code:{self.recipe_name}"
        self.checkpoint = model_name
        self.expected_dim = expected_dim
        self.attn_implementation = attn_implementation
        self.page_batch_size = max(1, int(page_batch_size))
        self.query_batch_size = max(1, int(query_batch_size))
        if recipe.style == "model":
            # a model method returns no attention mask, so a padded batch could not be unpadded here
            self.page_batch_size = self.query_batch_size = 1
        model_injected = model is not None
        injected = model is not None and (processor is not None or not recipe.needs_processor)
        if injected:
            self.revision = revision
        else:
            if not trust_remote_code:
                raise ValueError(
                    f"{model_name} ships its own modelling code, which runs on this machine when the "
                    "model is loaded. Read it, then pass trust_remote_code=True to load it.")
            self.revision = _resolve_revision(model_name, revision, known_revision)
            from transformers import AutoModel, AutoProcessor

            self.device = _resolve_device(device)
            self.torch_dtype = _resolve_dtype(dtype, self.device)
            if model is None:
                extra: dict[str, Any] = {"revision": self.revision, "trust_remote_code": True,
                                         **dict(recipe.model_kwargs)}
                if attn_implementation:
                    extra["attn_implementation"] = attn_implementation
                model = _from_pretrained(AutoModel, model_name, self.torch_dtype, **extra)
                self.loading_report = getattr(model, "loading_report", None)
                unloaded = [n for n, p in model.named_parameters() if p.device.type == "meta"]
                if unloaded:
                    raise RuntimeError(f"{model_name}: {len(unloaded)} parameters never loaded (e.g. {unloaded[0]})")
                model, self.device, self.placement = place_model(model, self.device, multi_gpu)
                model = model.eval()
            if processor is None and recipe.needs_processor:
                processor = AutoProcessor.from_pretrained(model_name, revision=self.revision,
                                                          trust_remote_code=True, **dict(recipe.processor_kwargs))
        if model_injected:
            _adopt_model_placement(self, model)
        self.model = model
        self.processor = processor

    def _call(self, target: Any, method: str, arg: str | None, items: list[Any], kwargs: Mapping[str, Any]) -> Any:
        fn = getattr(target, method, None)
        if not callable(fn):
            raise TypeError(f"{self.checkpoint}: {type(target).__name__} has no {method}(); the "
                               f"{self.recipe_name!r} recipe does not fit this checkpoint's code")
        return fn(items, **dict(kwargs)) if arg is None else fn(**{arg: items}, **dict(kwargs))

    def _run(self, method: str, arg: str | None, items: list[Any], kwargs: Mapping[str, Any]) -> list[np.ndarray]:
        import torch

        recipe = self.recipe
        if recipe.style == "model":
            with torch.inference_mode():
                return _as_list(self._call(self.model, method, arg, items, kwargs))
        batch = dict(self._call(self.processor, method, arg, items, kwargs))
        batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.inference_mode():
            out = self.model(**batch)
        return _rows(getattr(out, recipe.output, out) if recipe.output else out, batch.get("attention_mask"))

    def _images(self, images: list[Any]) -> list[np.ndarray]:
        r = self.recipe
        return self._run(r.image_method, r.image_arg, images, r.image_kwargs)

    def _queries(self, queries: list[str]) -> list[np.ndarray]:
        r = self.recipe
        return self._run(r.query_method, r.query_arg, queries, r.query_kwargs)

    def info(self) -> dict[str, Any]:
        return {**super().info(), "recipe": self.recipe_name, "recipe_exercised": self.recipe.exercised,
                "attn_implementation": self.attn_implementation}


class PyLateEncoder(_TokenEncoder):
    """A text ColBERT model through ``pylate.models.ColBERT``.

    ``encode_documents`` takes strings; ``encode_images`` is refused. It calls
    ``encode(sentences, is_query=...)`` and takes one array per item; ``check_retriever``
    passed through it here with PyLate 1.6.0 on two models
    (``reports/universal/loader_checks/2026-10-12.json``). PyLate pins its own
    sentence-transformers version, so it may need an environment of its own.
    """

    loader = "pylate"

    def __init__(self, model_name: str, device: str | None = None, expected_dim: int | None = None,
                 batch_size: int = 32, model: Any = None, **kwargs: Any) -> None:
        self.checkpoint = model_name
        self.expected_dim = expected_dim
        self.batch_size = max(1, int(batch_size))
        if model is None:
            try:
                from pylate import models
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError("this model needs PyLate (pip install pylate). If it is installed, it "
                                  f"failed to import in this environment: {exc}") from exc
            model = models.ColBERT(model_name_or_path=model_name, device=device, **kwargs)
        self.model = model
        self.device = str(device or getattr(model, "device", "cpu"))

    def _encode(self, texts: Sequence[str], is_query: bool) -> list[np.ndarray]:
        out = _as_list(self.model.encode([str(t) for t in texts], is_query=is_query,
                                         batch_size=self.batch_size, show_progress_bar=False))
        self._note_width(out)
        return out

    def encode_documents(self, documents: Sequence[str]) -> list[np.ndarray]:
        return self._encode(documents, is_query=False)

    def encode_queries(self, queries: Sequence[str]) -> list[np.ndarray]:
        return self._encode(queries, is_query=True)

    def encode_images(self, images: Sequence[Any]) -> list[np.ndarray]:
        raise TypeError(f"{self.checkpoint} is a text model; pass strings to encode_documents")
