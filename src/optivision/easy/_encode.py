"""Loading and driving the encoder of an easy-mode build (FINAL_DESIGN 4.2, 5.5 G-wire, 9).

``load_encoder`` turns a registry id (or an :class:`~optivision.adapters.EncoderAdapter` the
caller built) into an :class:`Encoder`: the adapter of :func:`optivision.adapters.load_adapter`,
called with the settings of our own Kaggle run of that model (``presets/encode_rates.json``:
dtype, batch size, processor arguments, GPU split) and pinned to the registry's revision, plus
the build record a later ``check()`` compares against. It refuses a model that runs its
repository's own code unless ``trust_remote_code=True`` (:class:`RemoteCodeRefused`) and a
model whose loader is not installed (:class:`MissingExtra`, with the pip line).

``Encoder.encode_pages`` sends pages to the adapter a chunk at a time; on CUDA out-of-memory
it halves the chunk, down to one page, and records the size it ended at. ``wiring_check``
runs :func:`optivision.adapters.check_retriever` on 12 of the user's pages, each searched with a
sentence of its own, and on the bundled sample PDF when that fails; it refuses only if both
fail. ``estimate`` gives encode time from our runs' timings, as an estimate with its basis.

Module level imports only the standard library and the easy-mode formatter, so
``import optivision.easy._doctor`` (which uses ``install_line`` and the rates) stays light;
the adapters, torch and the loaders are imported inside the functions that need them.
"""

from __future__ import annotations

import contextlib
import difflib
import gc
import json
import math
import platform
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata, resources, util
from pathlib import Path
from typing import Any

from . import _claim
from ._claim import Block, Claim, Line, claim, lit
from ._types import MissingExtra, RemoteCodeRefused, UnknownModel, WiringError

__all__ = ["DPI", "MAX_SIDE", "Encoder", "Estimate", "Wiring", "estimate", "install_line", "known_issues",
           "load_encoder", "rates", "recorded_dtype", "sample_path", "wiring_check"]

#: page rendering of easy mode (FINAL_DESIGN 5.2); recorded in the build record
DPI = 150
MAX_SIDE = 1536

#: the wiring check (FINAL_DESIGN 5.5, G-wire): pairs, allowed misses, and the fewest pairs it runs on
WIRING_PAIRS = 12
ALLOWED_MISSES = 2
MIN_PAIRS = 2 * ALLOWED_MISSES + 2

#: the pip lines of the loader extras (pyproject.toml)
EXTRA_LINES = {
    "easy": 'pip install "optivision-rag[easy]"',
    "easy-st": 'pip install "optivision-rag[easy-st]"',
    "easy-colpali": 'pip install "optivision-rag[easy-colpali]"',
}

#: distributions recorded in the build record and compared by check() and doctor(): (distribution, module)
LIBRARIES: tuple[tuple[str, str], ...] = (
    ("torch", "torch"), ("transformers", "transformers"), ("sentence-transformers", "sentence_transformers"),
    ("colpali-engine", "colpali_engine"), ("peft", "peft"), ("accelerate", "accelerate"),
    ("huggingface-hub", "huggingface_hub"), ("numpy", "numpy"), ("optivision-rag", "optivision"),
)

_REMOTE_CODE_NOTE = "pass trust_remote_code=True"


# ---------------------------------------------------------------------------- presets


@lru_cache(maxsize=1)
def rates() -> dict[str, Any]:
    """``presets/encode_rates.json`` (written by ``scripts/make_encode_rates.py``)."""
    text = resources.files("optivision").joinpath("presets", "encode_rates.json").read_text(encoding="utf-8")
    data = json.loads(text)
    if data.get("schema") != 1:
        raise RuntimeError(f"unknown encode_rates schema {data.get('schema')!r}")
    return data


@lru_cache(maxsize=1)
def known_issues() -> list[dict[str, Any]]:
    """``presets/known_issues.json``: library and hardware combinations that broke a load in our runs."""
    text = resources.files("optivision").joinpath("presets", "known_issues.json").read_text(encoding="utf-8")
    data = json.loads(text)
    if data.get("schema") != 1:
        raise RuntimeError(f"unknown known_issues schema {data.get('schema')!r}")
    return list(data["issues"])


def rate(model_id: str | None) -> dict[str, Any] | None:
    """Our run's timing and settings of a model, or None if no run of ours timed it."""
    if model_id is None:
        return None
    entry = rates()["models"].get(model_id)
    return json.loads(json.dumps(entry)) if entry is not None else None


def recorded_dtype(model_id: str | None) -> str | None:
    """The inference dtype of our run of the model (``"float16"``, ``"float32"``), or None if unrecorded."""
    entry = rate(model_id)
    return None if entry is None else entry.get("dtype")


def sample_path() -> Path:
    """The bundled 20-page sample PDF (``scripts/make_sample_pdf.py``; MIT, written for this package)."""
    return Path(__file__).resolve().parent / "data" / "sample.pdf"


# ---------------------------------------------------------------------------- registry facts


def _registry() -> dict[str, Any]:
    from ..adapters import MODELS

    return MODELS


def _info(model_id: str | None) -> Any:
    return None if model_id is None else _registry().get(model_id)


def _resolve_info(model: str) -> Any:
    """The registry entry of ``model``: exact id, else ``_resolve.resolve`` (case, short names)."""
    info = _info(model)
    if info is not None:
        return info
    try:
        from . import _resolve  # the rule agent's resolver, when present
    except ImportError:
        _resolve = None
    if _resolve is not None:
        found = _resolve.resolve(model)
        if found is not None:
            return found
    models = _registry()
    lowered = {m.lower(): m for m in models}
    if model.lower() in lowered:
        return models[lowered[model.lower()]]
    short = [m for m in models if m.split("/")[-1].lower() == model.lower()]
    if len(short) == 1:
        return models[short[0]]
    close = difflib.get_close_matches(model, list(models), n=3, cutoff=0.0)
    raise UnknownModel(f"{model!r} is not in the registry; closest ids: {', '.join(close)}")


def needs_remote_code(info: Any) -> bool:
    """True if loading the model runs Python from its own repository (FINAL_DESIGN 4.2)."""
    if info is None:
        return False
    return str(info.loader).startswith("remote-code:") or _REMOTE_CODE_NOTE in (info.notes or "")


def _loader(info: Any) -> str:
    return info.loader if info is not None else "sentence-transformers"


def _extra(loader: str) -> str:
    if loader == "sentence-transformers":
        return "easy-st"
    if loader.startswith(("colpali-engine:", "remote-code:")) or loader == "nemotron":
        return "easy-colpali"  # the campaign environment: colpali-engine with its transformers and peft
    return "easy"


def _requirements(model_id: str | None, info: Any) -> list[str]:
    """Extra pip requirements of one model: known issues with a requirement, notes, a GPU split."""
    out: list[str] = []
    for issue in known_issues():
        when = issue.get("when", {})
        req = issue.get("requirement")
        if req and model_id and _model_matches(when, model_id, info):
            out.append(req)
    if info is not None and "needs einops" in (info.notes or ""):
        out.append("einops")
    entry = rate(model_id)
    if entry is not None and entry.get("multi_gpu"):
        out.append("accelerate")
    return out


def install_line(model_id: str | None) -> str:
    """The pip line that installs what ``model_id`` loads through (``pip install "optivision-rag[easy-st]"``).

    torch is not part of it: a CPU-only torch is the default wheel on Windows, and ``ov.doctor()``
    says when a CUDA build is needed. ``None`` gives the planning-only line.
    """
    if model_id is None:
        return EXTRA_LINES["easy"]
    info = _info(model_id)
    loader = _loader(info)
    base = EXTRA_LINES[_extra(loader)]
    if loader == "pylate":
        base += " pylate"
    return base + "".join(f' "{req}"' for req in _requirements(model_id, info))


def _model_matches(when: dict[str, Any], model_id: str | None, info: Any) -> bool:
    """Whether the model conditions of a known issue hold (True when it has none)."""
    if "models" in when and model_id not in when["models"]:
        return False
    if "model_prefix" in when and not (model_id or "").startswith(when["model_prefix"]):
        return False
    return "loaders" not in when or any(_loader(info).startswith(p) for p in when["loaders"])


# ---------------------------------------------------------------------------- versions and devices


def _version_tuple(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()
    return tuple(int(x) for x in re.findall(r"\d+", version.split("+")[0])[:3])


def installed_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def library_versions() -> dict[str, str | None]:
    """Versions of the libraries that shape the vectors: the imported module's if loaded, else installed."""
    out: dict[str, str | None] = {"python": platform.python_version()}
    for dist, module in LIBRARIES:
        loaded = sys.modules.get(module)
        version = getattr(loaded, "__version__", None) if loaded is not None else None
        out[dist] = str(version) if version is not None else installed_version(dist)
    return out


def _torch_if_loaded() -> Any:
    return sys.modules.get("torch")


def has_module(name: str) -> bool:
    """Whether ``import name`` would find the package, without importing it."""
    try:
        return util.find_spec(name) is not None
    except (ValueError, ImportError):  # in sys.modules without a spec (a stand-in): it is there
        return sys.modules.get(name) is not None


def _cuda_count() -> int:
    if not has_module("torch"):
        return 0
    import torch

    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 -- a broken CUDA install answers with errors; treat as no GPU
        return 0


def _bf16_native() -> bool:
    torch = _torch_if_loaded()
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8


def _is_oom(exc: BaseException) -> bool:
    """CUDA (or MPS) out-of-memory, whatever class this torch build raises it as."""
    torch = _torch_if_loaded()
    oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None) if torch is not None else None
    if isinstance(oom, type) and isinstance(exc, oom):
        return True
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_device_memory() -> None:
    gc.collect()
    torch = _torch_if_loaded()
    if torch is not None:
        with contextlib.suppress(Exception):  # freeing is best effort
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _gpu_budgets() -> dict[int, int]:
    """``max_memory`` for a sentence-transformers model spread with ``device_map="auto"``.

    As ``scripts/encode_vectors.py`` ran it: GPU 0 also holds the inputs and every page's
    outputs, so it gets a smaller share of the weights (tencent/EVIE-8B ran GPU 0 out of memory
    with an even split, Kaggle 2026-10-16).
    """
    import torch

    return {i: int(torch.cuda.get_device_properties(i).total_memory * (0.45 if i == 0 else 0.85))
            for i in range(torch.cuda.device_count())}


def _dtype_name(dtype: Any) -> str | None:
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _walk_models(adapter: Any) -> list[Any]:
    """The model objects behind an adapter (its encoder's ``model``, its own ``model``), for facts."""
    out = []
    for owner in (getattr(adapter, "encoder", None), adapter):
        model = getattr(owner, "model", None) if owner is not None else None
        if model is not None:
            out.append(model)
    return out


def _config_value(adapter: Any, key: str) -> Any:
    """A value from the earliest transformers config found behind the adapter (``_commit_hash``, ...)."""
    for model in _walk_models(adapter):
        candidates = [model]
        modules = getattr(model, "modules", None)
        if callable(modules):
            with contextlib.suppress(Exception):  # an arbitrary object may not iterate
                candidates.extend(list(modules())[:50])
        for obj in candidates:
            value = getattr(getattr(obj, "config", None), key, None)
            if isinstance(value, str) and value:
                return value
    return None


def _cached_sha(model_id: str) -> str | None:
    """The commit of the cached ``config.json`` of the default branch, read from its snapshot path."""
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model_id, "config.json")
    except Exception:  # noqa: BLE001 -- no hub library or no cache: unknown
        return None
    if isinstance(path, str):
        match = re.search(r"snapshots[\\/]([0-9a-f]{40})[\\/]", path)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------- estimate


def _rate_text(lo: float, hi: float) -> str:
    """``lo-hi`` with two significant figures on ``lo``, the low end rounded down and the high end up."""
    digits = 2 if lo >= 0.1 or lo <= 0 else max(2, 1 - math.floor(math.log10(lo)))
    scale = 10**digits
    a = math.floor(lo * scale + 1e-9) / scale
    b = math.ceil(hi * scale - 1e-9) / scale
    return f"{a:.{digits}f}" if a == b else f"{a:.{digits}f}-{b:.{digits}f}"


def _gpu_text(entry: dict[str, Any]) -> str | None:
    name = entry.get("gpu_name")
    if not name:
        return None
    short = str(name).split()[-1]
    gpus = int(entry.get("gpus") or 1)
    return f"one {short}" if gpus == 1 else f"{gpus}x {short}"


@dataclass(frozen=True)
class Estimate:
    """Encode time of ``n_pages`` from our run's timings (FINAL_DESIGN 2 (c), 4.1).

    ``lo_s`` / ``hi_s`` are the whole encode step in seconds: ``encode_s`` (pages times our
    per-page range) plus ``setup_s`` (our runs' model download and load, and the 12-pair check).
    ``basis`` is the source text of the ``[estimate: ...]`` tag (``"our Kaggle run, 0.61-0.65
    s/page"``) and ``gpu`` what that ran on (``"one T4"``, ``"2x T4"``). For a model no run of
    ours timed, the seconds are None and ``basis`` says so. A text model's unit is a document.
    The figures were measured on page images; rendering a PDF is not in them.
    """

    lo_s: float | None
    hi_s: float | None
    basis: str
    gpu: str | None
    model: str | None = None
    unit: str = "page"
    s_per_page: tuple[float, float] | None = None
    encode_s: tuple[float, float] | None = None
    setup_s: tuple[float, float] | None = None
    dtype: str | None = None
    multi_gpu: bool = False

    @property
    def known(self) -> bool:
        return self.lo_s is not None and self.hi_s is not None

    def claim(self) -> Claim | None:
        """The total as an ``estimate`` claim (``"about 1.4-1.5 h"``), or None when not timed."""
        if not self.known:
            return None
        return claim((self.lo_s, self.hi_s), "estimate", source=self.basis, construct="encode time",
                     text=_claim.duration(float(self.lo_s), float(self.hi_s)))

    def setup_claim(self) -> Claim | None:
        """The model load and set-up part (``"about 1-2 min"``) as an ``estimate`` claim, same basis."""
        if self.setup_s is None:
            return None
        lo, hi = (float(x) for x in self.setup_s)
        if hi < 60:
            text = "under a minute"
        else:
            a, b = max(1, round(lo / 60)), max(1, math.ceil(hi / 60))
            text = f"about {a} min" if a == b else f"about {a}-{b} min"
        return claim((lo, hi), "estimate", source=self.basis, construct="model load", text=text)


def estimate(model_id: str | None, n_pages: int) -> Estimate:
    """How long encoding ``n_pages`` takes, from ``presets/encode_rates.json``.

    ``lo_s = n * fastest page rate + shortest set-up`` and ``hi_s = n * slowest + longest``,
    over the encode steps of our run of that model (DocVQA and InfoVQA for page models).
    """
    entry = rate(model_id)
    if entry is None:
        return Estimate(None, None, "not timed in our runs; a build times its own pages", None, model=model_id)
    n = max(0, int(n_pages))
    s_lo, s_hi = (float(x) for x in entry["s_per_page"])
    setup_lo, setup_hi = (float(x) for x in entry["setup_s"])
    unit = entry.get("unit", "page")
    return Estimate(
        lo_s=n * s_lo + setup_lo,
        hi_s=n * s_hi + setup_hi,
        basis=f"our Kaggle run, {_rate_text(s_lo, s_hi)} s/{unit}",
        gpu=_gpu_text(entry),
        model=model_id,
        unit=unit,
        s_per_page=(s_lo, s_hi),
        encode_s=(n * s_lo, n * s_hi),
        setup_s=(setup_lo, setup_hi),
        dtype=entry.get("dtype"),
        multi_gpu=bool(entry.get("multi_gpu")),
    )


# ---------------------------------------------------------------------------- loading


def _modules_for(loader: str) -> list[tuple[str, str]]:
    """(module, distribution) pairs a loader imports."""
    mods = [("torch", "torch")]
    if loader == "sentence-transformers":
        mods.append(("sentence_transformers", "sentence-transformers"))
    elif loader.startswith("colpali-engine:"):
        mods.append(("colpali_engine", "colpali-engine"))
    elif loader.startswith("remote-code:") or loader == "nemotron":
        mods.append(("transformers", "transformers"))
    elif loader == "pylate":
        mods.append(("pylate", "pylate"))
    return mods


def missing_packages(model_id: str | None, info: Any = None, *, multi_gpu: bool = False) -> list[str]:
    """What the model's loader needs that is absent (or too old), as pip requirement names."""
    info = info if info is not None else _info(model_id)
    loader = _loader(info)
    missing = [dist for module, dist in _modules_for(loader) if not has_module(module)]
    for req in _requirements(model_id, info):
        name, _, bound = req.partition(">=")
        name = name.strip()
        if name == "accelerate" and not multi_gpu:
            continue
        installed = installed_version(name)
        if installed is None:
            missing.append(req)
        elif bound and _version_tuple(installed) < _version_tuple(bound):
            missing.append(f"{req} (installed: {installed})")
    return missing


def _missing_extra(model_id: str, loader: str, missing: Sequence[str], cause: str = "") -> MissingExtra:
    line = install_line(model_id)
    what = ", ".join(missing)
    text = (f"{model_id} loads through {loader}, which needs {what}.{(' ' + cause) if cause else ''}\n"
            f"Install: {line}\n"
            f"In a notebook, run %{line} and restart the runtime; ov.doctor(\"{model_id}\") checks the rest.")
    return MissingExtra(text)


def _remote_code_refused(info: Any) -> RemoteCodeRefused:
    revision = info.revision or "its default branch (unpinned)"
    licence = info.license or "not recorded"
    url = f"https://huggingface.co/{info.model_id}/tree/{info.revision or 'main'}"
    return RemoteCodeRefused(
        f"{info.model_id} runs Python code from its own repository on this machine when it loads.\n"
        f"Repository {info.model_id}, revision {revision}, licence {licence}.\n"
        f"Read that code ({url}); if you accept it, pass trust_remote_code=True.")


def _loader_kwargs(info: Any, entry: dict[str, Any] | None, dtype: str | None, device: str, multi_gpu: bool,
                   trust_remote_code: bool) -> dict[str, Any]:
    """Keyword arguments for ``load_adapter``, as ``scripts/encode_vectors.py`` ran each loader.

    The registry revision is passed on the sentence-transformers path and to ``ColVLMEncoder``,
    where ``load_adapter`` does not add it (FINAL_DESIGN 9, Revisions); the other loaders take
    it from the registry themselves.
    """
    loader = _loader(info)
    kw: dict[str, Any] = {}
    remote = trust_remote_code and needs_remote_code(info)
    if loader == "sentence-transformers":
        recorded_batch = (entry or {}).get("batch_size")
        kw["batch_size"] = int(recorded_batch) if recorded_batch else (32 if info.modality == "text" else 4)
        kw["encode_kwargs"] = {"show_progress_bar": False}
        if dtype is not None:
            kw["dtype"] = dtype
        if remote:
            kw["trust_remote_code"] = True
        if info.revision:
            kw["revision"] = info.revision
        if multi_gpu:
            kw["model_kwargs"] = {"device_map": "auto", "max_memory": _gpu_budgets()}
        elif device != "auto":
            kw["device"] = device
        if (entry or {}).get("processor_kwargs"):
            kw["processor_kwargs"] = entry["processor_kwargs"]
    elif loader.startswith(("colpali-engine:", "remote-code:")) or loader == "nemotron":
        kw["dtype"] = dtype or "auto"
        kw["multi_gpu"] = multi_gpu
        kw["device"] = device
        if loader.startswith("colpali-engine:") and info.revision:
            from ..encoders.colvlm import BACKENDS

            if loader.split(":", 1)[1] in BACKENDS:
                kw["revision"] = info.revision
        if loader.startswith("remote-code:") and remote:
            kw["trust_remote_code"] = True
    elif loader == "pylate":
        if device != "auto":
            kw["device"] = device
    return kw


def _build_adapter(model_id: str, kwargs: dict[str, Any]) -> Any:
    from .. import adapters

    return adapters.load_adapter(model_id, **kwargs)


def _fallback_dtype(chosen: str | None) -> str | None:
    """A smaller dtype to retry a load that did not fit, or None (already half precision, or no GPU)."""
    if chosen in ("float16", "bfloat16"):
        return None
    if _cuda_count() == 0:
        return None
    return "bfloat16" if _bf16_native() else "float16"


def _is_adapter(obj: Any) -> bool:
    return callable(getattr(obj, "encode_documents", None)) and callable(getattr(obj, "encode_queries", None))


def _modality_of(adapter: Any, info: Any) -> str:
    if info is not None and getattr(info, "modality", None):
        return str(info.modality)
    if isinstance(getattr(adapter, "modality", None), str):
        return adapter.modality
    identity = None
    try:
        identity = adapter.identity() if callable(getattr(adapter, "identity", None)) else None
    except Exception:  # noqa: BLE001 -- an arbitrary adapter's identity may fail
        identity = None
    if identity is not None and getattr(identity, "modality", None) in ("image", "text"):
        return identity.modality
    return "image"


def load_encoder(model: Any, *, device: str = "auto", dtype: str | None = "recorded",
                 trust_remote_code: bool = False, batch_size: int | None = None, echo: bool = True) -> Encoder:
    """Load the encoder of a registry model (or wrap an adapter you built) for an easy-mode build.

    ``dtype="recorded"`` loads in our run's inference dtype (``recorded_dtype``); a model whose
    run recorded none loads at its loader's default, and the record says so. If the model does
    not fit the GPU in that dtype, it is loaded once more in half precision and the record and
    output say "dtype differs from our measurement". Any other value (``"float16"``,
    ``"float32"``, ``"bfloat16"``, ``"auto"``) is used as given. A model our run spread over two
    GPUs is spread again when this machine has two. ``batch_size`` overrides the page chunk sent
    to the adapter per call; ``echo`` prints the notes as they arise (they are kept in
    ``Encoder.notes`` and the record either way).

    An :class:`~optivision.adapters.EncoderAdapter` instance is used as it is: no registry
    entry, no refusal, no dtype; a build treats it as an unmeasured model.

    Raises :class:`UnknownModel`, :class:`RemoteCodeRefused` and :class:`MissingExtra`.
    """
    if not isinstance(model, str):
        if not _is_adapter(model):
            raise TypeError(f"model must be a registry id or an EncoderAdapter, got {type(model).__name__}")
        return Encoder._wrap(model, batch_size=batch_size)
    info = _resolve_info(model)
    model_id = info.model_id
    loader = _loader(info)
    if needs_remote_code(info) and not trust_remote_code:
        raise _remote_code_refused(info)
    entry = rate(model_id)
    gpus_here = _cuda_count()
    multi = bool(entry and entry.get("multi_gpu")) and gpus_here > 1 and device in ("auto", "cuda")
    missing = missing_packages(model_id, info, multi_gpu=multi)
    if missing:
        raise _missing_extra(model_id, loader, missing)
    notes: list[str] = []
    ours = recorded_dtype(model_id)
    if dtype == "recorded":
        chosen = ours
        dtype_source = "as in our run" if ours else "loader default; our run recorded none"
    else:
        chosen = None if dtype in (None, "auto") else str(dtype)
        dtype_source = "as given"
    if entry and entry.get("multi_gpu") and not multi:
        notes.append(f"our run spread {model_id} over {_gpu_text(entry)}; this machine has "
                     f"{'one GPU' if gpus_here == 1 else 'no GPU' if gpus_here == 0 else 'its GPUs unused'}")
    kwargs = _loader_kwargs(info, entry, chosen, device, multi, trust_remote_code)
    loaded_dtype = chosen
    adapter = None
    fallback = None
    try:
        adapter = _build_adapter(model_id, kwargs)
    except ImportError as exc:
        raise _missing_extra(model_id, loader, [str(exc).split("\n")[0]]) from exc
    except Exception as exc:
        fallback = _fallback_dtype(chosen) if _is_oom(exc) else None
        if fallback is None:
            raise
    if adapter is None:  # out of memory in the chosen dtype; retried outside the handler so the
        _free_device_memory()  # traceback no longer holds the half-loaded model
        kwargs = _loader_kwargs(info, entry, fallback, device, multi, trust_remote_code)
        adapter = _build_adapter(model_id, kwargs)
        loaded_dtype = fallback
        dtype_source = f"fallback: {chosen or 'the loader default'} did not fit this GPU"
        notes.append(f"dtype differs from our measurement: loaded in {fallback} because "
                     f"{chosen or 'the loader default'} did not fit (ours: {ours or 'not recorded'})")
    if dtype != "recorded" and chosen is not None and ours is not None and chosen != ours:
        notes.append(f"dtype differs from our measurement: {chosen} as given (ours: {ours})")
    if dtype == "recorded" and ours is None:
        notes.append(f"our run of {model_id} recorded no dtype; loaded at the loader's default")
    encoder = Encoder(adapter, info=info, modality=info.modality, chunk_size=batch_size, notes=notes)
    encoder.record.update(_record_for(encoder, model_id, info, loader, kwargs, dtype, loaded_dtype, ours,
                                      dtype_source, multi, trust_remote_code))
    if echo:
        for note in notes:
            print(note)
    return encoder


def _record_for(encoder: Encoder, model_id: str, info: Any, loader: str, kwargs: dict[str, Any],
                dtype_requested: Any, loaded_dtype: str | None, ours: str | None, dtype_source: str,
                multi: bool, trust_remote_code: bool) -> dict[str, Any]:
    adapter = encoder.adapter
    inner = getattr(adapter, "encoder", None)
    actual = _dtype_name(getattr(inner, "torch_dtype", None)) or _dtype_name(getattr(adapter, "dtype", None))
    for model in _walk_models(adapter) if actual is None else ():
        with contextlib.suppress(Exception):  # a stand-in model may have no parameters
            actual = _dtype_name(next(iter(model.parameters())).dtype)
        if actual is not None:
            break
    pinned = info.revision
    resolved = pinned or getattr(inner, "revision", None) or _config_value(adapter, "_commit_hash") \
        or _cached_sha(model_id)
    attn = getattr(inner, "attn_implementation", None) or _config_value(adapter, "_attn_implementation")
    return {
        "model": model_id,
        "registered": True,
        "status": info.status,
        "licence": info.license,
        "loader": loader,
        "revision": pinned,
        "resolved_revision": resolved,
        "revision_source": "registry pin" if pinned else ("resolved at load" if resolved else "not known"),
        "dtype": actual or loaded_dtype,
        "dtype_requested": dtype_requested,
        "dtype_ours": ours,
        "dtype_source": dtype_source,
        "dtype_matches_ours": None if ours is None else (actual or loaded_dtype) == ours,
        "attn_implementation": attn,
        "batch_size": kwargs.get("batch_size", getattr(inner, "page_batch_size", None)),
        "processor_kwargs": kwargs.get("processor_kwargs"),
        "load_options": dict(info.load_options),
        "multi_gpu": multi,
        "placement": getattr(adapter, "placement", None) or getattr(inner, "placement", None),
        "trust_remote_code": bool(trust_remote_code and needs_remote_code(info)),
    }


# ---------------------------------------------------------------------------- the encoder


def _split_id(page_id: str) -> tuple[str, int]:
    doc, sep, page = str(page_id).rpartition("::p")
    if sep and page.isdigit():
        return doc, int(page)
    return str(page_id), 1


class Encoder:
    """A loaded encoder: ``adapter``, ``info`` (registry entry or None), ``record``, ``notes``.

    ``encode_pages`` and ``encode_queries`` return a
    :class:`~optivision.representation.MultiVectorCorpus` with the given ids. Pages go to the
    adapter ``chunk_size`` at a time; a CUDA out-of-memory error halves the chunk (down to one
    page) and the run carries on with the smaller size, which ``record["page_chunk_final"]`` keeps.
    """

    def __init__(self, adapter: Any, *, info: Any = None, modality: str | None = None,
                 chunk_size: int | None = None, notes: list[str] | None = None) -> None:
        self.adapter = adapter
        self.info = info
        self.modality = modality or _modality_of(adapter, info)
        self.notes: list[str] = list(notes or [])
        self.chunk_size = max(1, int(chunk_size)) if chunk_size else self._default_chunk()
        self.query_chunk = 32
        self.oom_halvings = 0
        self._pairs = self._wants_pairs()
        self.record: dict[str, Any] = {
            "model": None, "registered": False, "loader": None, "revision": None, "resolved_revision": None,
            "dtype": None, "attn_implementation": None,
            "dpi": DPI, "max_side": MAX_SIDE,
            "page_chunk": self.chunk_size, "page_chunk_final": self.chunk_size, "oom_halvings": 0,
            **_device_facts(adapter),
            "versions": library_versions(),
            "notes": self.notes,
        }

    @classmethod
    def _wrap(cls, adapter: Any, *, batch_size: int | None = None) -> Encoder:
        info = getattr(adapter, "info", None)
        identity = None
        try:
            identity = adapter.identity() if callable(getattr(adapter, "identity", None)) else None
        except Exception:  # noqa: BLE001
            identity = None
        name = getattr(identity, "model_id", None) or type(adapter).__name__
        notes = ["an adapter you built: not in our measurements, so a build treats it as an unmeasured model"]
        encoder = cls(adapter, info=None, chunk_size=batch_size, notes=notes)
        encoder.record.update({
            "model": name, "registered": False, "status": None, "licence": None,
            "loader": f"your adapter ({type(adapter).__name__})",
            "revision": getattr(identity, "revision", None), "resolved_revision": getattr(identity, "revision", None),
            "revision_source": "the adapter's identity" if getattr(identity, "revision", None) else "not known",
            "dtype": getattr(identity, "inference_dtype", None), "dtype_requested": None, "dtype_ours": None,
            "dtype_source": "the adapter's own", "dtype_matches_ours": None,
            "registry_model": getattr(info, "model_id", None),
        })
        return encoder

    # ------------------------------------------------------------------ internals

    def _default_chunk(self) -> int:
        size = getattr(self.adapter, "batch_size", None)
        if isinstance(size, int) and not isinstance(size, bool) and size > 0:
            return size  # one call is one batch of the adapter's own
        return 8

    def _wants_pairs(self) -> bool:
        """Whether the adapter takes ``(PageRef, image)`` pairs (the repository's page and token adapters)."""
        if self.modality == "text":
            return False
        try:
            from ..adapters import PageEncoderAdapter, TokenEncoderAdapter
            from ..adapters.nemotron import NemotronAdapter
        except ImportError:  # pragma: no cover - the adapters always import
            return False
        return isinstance(self.adapter, (PageEncoderAdapter, TokenEncoderAdapter, NemotronAdapter))

    def _documents(self, items: Sequence[Any], ids: Sequence[str]) -> list[Any]:
        if self.modality == "text":
            return [str(x) for x in items]
        if self._pairs:
            from ..types import PageRef

            return [(PageRef(*_split_id(i)), item) for i, item in zip(ids, items, strict=True)]
        return list(items)

    def _chunked(self, call: Callable[[list[Any], list[str]], Any], items: list[Any], ids: list[str],
                 chunk: int, what: str, progress: Callable[[int, int], Any] | None = None) -> tuple[Any, int]:
        from ..representation import MultiVectorCorpus

        parts = []
        done = 0
        total = len(items)
        while done < total:
            size = min(chunk, total - done)
            out_of_memory = False
            try:
                part = call(items[done:done + size], ids[done:done + size])
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                if size == 1:
                    raise RuntimeError(
                        f"the GPU ran out of memory on one {what} at a time ({self.record.get('model')}, "
                        f"dtype {self.record.get('dtype')}). Load it in half precision "
                        f"(dtype=\"float16\") or on a larger GPU; ov.doctor() shows the memory free.") from exc
                out_of_memory = True
            if out_of_memory:  # freed outside the handler, once the traceback has let go of the activations
                _free_device_memory()
                chunk = max(1, size // 2)
                self.oom_halvings += 1
                self.record["oom_halvings"] = self.oom_halvings
                continue
            if len(part) != size:
                raise RuntimeError(f"the adapter returned {len(part)} {what}s for {size} inputs")
            parts.append(part)
            done += size
            if progress is not None:
                progress(done, total)
        if not parts:
            raise ValueError(f"no {what}s to encode")
        if len(parts) == 1 and getattr(parts[0], "images", None) is None:
            return parts[0], chunk
        reps = [rep for part in parts for rep in part]
        return MultiVectorCorpus.from_documents(reps, ids=ids, attrs=parts[0].attrs), chunk

    # ------------------------------------------------------------------ public

    def encode_pages(self, images: Sequence[Any], ids: Sequence[str],
                     progress: Callable[[int, int], Any] | None = None) -> Any:
        """Encode pages (PIL images; strings for a text model) under ``ids``; ``progress(done, total)``."""
        items, ids = list(images), [str(i) for i in ids]
        if len(items) != len(ids):
            raise ValueError(f"{len(items)} pages for {len(ids)} ids")

        def call(chunk_items: list[Any], chunk_ids: list[str]) -> Any:
            return self.adapter.encode_documents(self._documents(chunk_items, chunk_ids), ids=chunk_ids)

        corpus, self.chunk_size = self._chunked(call, items, ids, self.chunk_size, "page", progress)
        self.record["page_chunk_final"] = self.chunk_size
        return corpus

    def encode_queries(self, texts: Sequence[str], ids: Sequence[str] | None = None) -> Any:
        """Encode query strings (ids ``"0"``, ``"1"``, ... unless given)."""
        items = [str(t) for t in texts]
        ids = [str(i) for i in ids] if ids is not None else [str(i) for i in range(len(items))]
        if len(items) != len(ids):
            raise ValueError(f"{len(items)} queries for {len(ids)} ids")

        def call(chunk_items: list[Any], chunk_ids: list[str]) -> Any:
            return self.adapter.encode_queries(chunk_items, ids=chunk_ids)

        corpus, self.query_chunk = self._chunked(call, items, ids, self.query_chunk, "query")
        return corpus

    def line(self) -> Line:
        """The ``Model`` line of a build summary: id, loader, dtype against ours, device."""
        rec = self.record
        dtype = rec.get("dtype") or "dtype unknown"
        ours = rec.get("dtype_ours")
        if not rec.get("registered"):
            how = f"{dtype}, as your adapter runs it"
        elif ours is None:
            how = f"{dtype}, the loader default (our run recorded none)"
        elif rec.get("dtype_matches_ours"):
            how = f"{dtype} as in our run"
        else:
            how = f"{dtype}; dtype differs from our measurement ({ours})"
        device = rec.get("gpu") or rec.get("device") or "CPU"
        return Line("Model", "{0}, {1}, {2}; {3}", lit(rec.get("model")), lit(rec.get("loader")), lit(how),
                    claim(device, "counted", construct="device", text=str(device)))


def _device_facts(adapter: Any) -> dict[str, Any]:
    inner = getattr(adapter, "encoder", None)
    device = getattr(inner, "device", None) or getattr(getattr(adapter, "model", None), "device", None)
    facts: dict[str, Any] = {"device": None if device is None else str(device), "gpu": None, "n_gpus": 0}
    torch = _torch_if_loaded()
    if torch is not None:
        with contextlib.suppress(Exception):  # a broken CUDA install: no GPU facts
            if torch.cuda.is_available():
                facts["n_gpus"] = int(torch.cuda.device_count())
                facts["gpu"] = torch.cuda.get_device_name(0)
    return facts


# ---------------------------------------------------------------------------- wiring check

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _segments(text: str) -> list[str]:
    """The page's text as runs of lines; a short line without closing punctuation (a heading, a
    table row) stands alone, so it is not glued to the sentence after it."""
    segments: list[str] = []
    run: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            if run:
                segments.append(" ".join(run))
                run = []
            continue
        if len(line.split()) < 8 and not line.endswith((".", "!", "?", ",", ";", ":")):
            if run:
                segments.append(" ".join(run))
                run = []
            segments.append(line)
            continue
        run.append(line)
    if run:
        segments.append(" ".join(run))
    return segments


def _sentences(text: str) -> list[str]:
    """Candidate query sentences of a page: 6 to 30 words, mostly words rather than numbers."""
    out = []
    for segment in _segments(text):
        for sentence in _SENTENCE_END.split(re.sub(r"\s+", " ", segment).strip()):
            sentence = sentence.strip()
            tokens = sentence.split()
            words = _WORD.findall(sentence)
            if 6 <= len(tokens) <= 30 and len(words) >= 0.7 * len(tokens):
                out.append(sentence)
    return out


def _preference(sentence: str) -> tuple[bool, bool, int, str]:
    """Sort key: sentences without digits, of 8 to 25 words, longest, then alphabetical."""
    n = len(sentence.split())
    return (bool(re.search(r"\d", sentence)), not 8 <= n <= 25, -len(sentence), sentence)


def _group(record: Any) -> str:
    for attr in ("doc_id", "file", "relpath"):
        value = getattr(record, attr, None)
        if value:
            return str(value)
    return _split_id(str(getattr(record, "id", "")))[0]


def _usable(record: Any) -> bool:
    if getattr(record, "blank", False):
        return False
    return len((getattr(record, "text", "") or "").strip()) >= 20


def _spread(n: int, m: int) -> list[int]:
    """``m`` indices spread evenly over ``range(n)`` (all of them when ``m >= n``)."""
    if m >= n:
        return list(range(n))
    if m == 1:
        return [n // 2]
    return sorted({round(i * (n - 1) / (m - 1)) for i in range(m)})


def select_pairs(pages: Sequence[tuple[Any, Any]], k: int = WIRING_PAIRS) -> list[tuple[Any, Any, str]]:
    """Up to ``k`` pages, each with a sentence of its own that no other chosen page holds.

    One page per file to begin with (files in sorted order, the middle page of each); when there are
    fewer than ``k`` files, more pages of the same files, spread evenly. A page joins with its
    preferred sentence (no digits, 8 to 25 words, longest) that no chosen page holds, and only
    if it holds no chosen sentence, so each query has one right answer among the pairs.
    """
    usable = [(rec, img) for rec, img in pages if _usable(rec)]
    by_file: dict[str, list[tuple[Any, Any]]] = {}
    for rec, img in usable:
        by_file.setdefault(_group(rec), []).append((rec, img))
    files = sorted(by_file)
    middle = {name: by_file[name][len(by_file[name]) // 2] for name in files}
    order: list[tuple[Any, Any]] = [middle[name] for name in files]  # one page per file, from its middle
    rest = [p for name in files for p in by_file[name] if p is not middle[name]]
    need = k - len(order)
    if rest and need > 0:  # then more pages of the same files, spread evenly, the others as reserves
        picks = _spread(len(rest), need)
        order.extend([rest[i] for i in picks] + [p for i, p in enumerate(rest) if i not in picks])
    else:
        order.extend(rest)
    chosen: list[tuple[Any, Any, str]] = []
    texts: list[str] = []
    for rec, img in order:
        if len(chosen) >= k:
            break
        text = _normal(rec.text)
        if any(_normal(s) in text for _, _, s in chosen):
            continue
        candidates = sorted(_sentences(rec.text), key=_preference)
        pick = next((s for s in candidates if not any(_normal(s) in t for t in texts)), None)
        if pick is None:
            continue
        chosen.append((rec, img, pick))
        texts.append(text)
    return chosen


@dataclass(frozen=True)
class _SampleRecord:
    id: str
    doc_id: str
    page: int
    text: str
    blank: bool = False


def _sample_records() -> list[_SampleRecord]:
    """The text layer of every page of the sample PDF (pypdfium2; no rendering)."""
    import pypdfium2 as pdfium

    path = sample_path()
    pdf = pdfium.PdfDocument(str(path))
    out = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_range()
            finally:
                textpage.close()
                page.close()
            out.append(_SampleRecord(f"sample.pdf::p{i + 1}", "sample.pdf", i + 1, text))
    finally:
        pdf.close()
    return out


def _render_sample(pages: Sequence[int], dpi: int = DPI, max_side: int = MAX_SIDE) -> list[Any]:
    """Render pages (1-based) of the sample PDF as easy mode renders a user's pages."""
    import pypdfium2 as pdfium

    from ..ingest import downscale

    pdf = pdfium.PdfDocument(str(sample_path()))
    images = []
    try:
        for number in pages:
            page = pdf[number - 1]
            try:
                image = page.render(scale=dpi / 72.0).to_pil().convert("RGB")
            finally:
                page.close()
            images.append(downscale(image, max_side))
    finally:
        pdf.close()
    return images


def _sample_pairs(encoder: Encoder) -> list[tuple[Any, Any, str]]:
    records = _sample_records()
    chosen = select_pairs([(r, None) for r in records])
    if encoder.modality == "text":
        return [(rec, rec.text, sentence) for rec, _, sentence in chosen]
    images = _render_sample([rec.page for rec, _, _ in chosen])
    return [(rec, img, sentence) for (rec, _, sentence), img in zip(chosen, images, strict=True)]


def _as_adapter(encoder: Encoder) -> Any:
    """The encoder behind the adapter interface ``check_retriever`` calls, OOM halving included."""
    from ..adapters import EncoderAdapter

    class _Wired(EncoderAdapter):
        info = encoder.info

        def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> Any:
            ids = list(ids) if ids is not None else [f"wiring/{i}::p1" for i in range(len(documents))]
            return encoder.encode_pages(list(documents), ids)

        def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> Any:
            return encoder.encode_queries(list(queries), ids)

    return _Wired()


def _run_check(encoder: Encoder, pairs: list[tuple[Any, Any, str]]) -> dict[str, Any]:
    from ..adapters import check_retriever

    documents = [rec.text if encoder.modality == "text" else img for rec, img, _ in pairs]
    queries = [sentence for _, _, sentence in pairs]
    report = check_retriever(_as_adapter(encoder), documents, queries, allowed_misses=ALLOWED_MISSES)
    report["pages"] = [str(getattr(rec, "id", "")) for rec, _, _ in pairs]
    report["queries"] = queries
    return report


@dataclass(frozen=True)
class Wiring:
    """The outcome of the wiring check (FINAL_DESIGN 5.5, G-wire).

    ``ok`` is True when the model found its own pages on the user's pages or, failing that, on
    the bundled sample. ``hits`` of ``n`` are the pairs whose sentence ranked its own page above
    every other on the run that decided (the sample when ``used_sample``). ``user`` and
    ``sample`` hold the two ``check_retriever`` reports (None for a run that did not happen);
    ``files`` is how many files the user's pairs came from. ``message`` is the line as printed.
    """

    ok: bool
    hits: int
    n: int
    used_sample: bool
    message: str
    files: int = 0
    user: dict[str, Any] | None = None
    sample: dict[str, Any] | None = None
    lines: tuple[Line, ...] = field(default=(), repr=False, compare=False)

    def line(self) -> Line:
        return self.lines[0]


def _counted(n: int) -> Claim:
    return claim(int(n), "counted")


def _failure_detail(report: dict[str, Any]) -> list[Line]:
    out = []
    if not report.get("finite"):
        out.append(Line("", "some vectors are not finite numbers (NaN or infinite)"))
    elif report.get("unit_norm_max_error", 0.0) >= 1e-2:
        err = float(report["unit_norm_max_error"])
        out.append(Line("", "vectors are not unit length: off by up to {0}",
                        claim(err, "counted", construct="norm error", text=f"{err:.3g}")))
    if report.get("dimension") != report.get("query_dimension"):
        out.append(Line("", "page vectors are {0} wide, query vectors {1}",
                        _counted(report.get("dimension") or 0), _counted(report.get("query_dimension") or 0)))
    return out


def wiring_check(encoder: Encoder, pages: Sequence[tuple[Any, Any]]) -> Wiring:
    """Check that the model finds pages by their own words before hours of encoding.

    ``pages`` are ``(PageRecord, image)`` pairs (``_docs.iter_pages``; a record needs ``id``,
    ``text`` and a file name in ``doc_id``/``file``; for a text model the image may be None).
    Twelve text pages from different files are each searched with a sentence unique among them,
    through :func:`optivision.adapters.check_retriever` with two misses allowed. On failure the
    check runs once more on the bundled sample PDF; only if that fails too is
    :class:`WiringError` raised, naming what went wrong. Nothing is written either way.
    """
    pairs = select_pairs(pages)
    files = len({_group(rec) for rec, _, _ in pairs})
    user = _run_check(encoder, pairs) if len(pairs) >= MIN_PAIRS else None
    if user is not None and user["ok"]:
        hits, n = int(user["query_ranks_own_document_first"]), int(user["pairs"])
        line = Line("Wiring", "{0} pages from {1} files, each searched with a sentence of its own: "
                    "{2} of {3} found their own page. OK.", _counted(n), _counted(files), _counted(hits), _counted(n))
        return Wiring(True, hits, n, False, line.render(), files=files, user=user, lines=(line,))
    sample = _run_check(encoder, _sample_pairs(encoder))
    s_hits, s_n = int(sample["query_ranks_own_document_first"]), int(sample["pairs"])
    if sample["ok"]:
        if user is None:
            line = Line("Wiring", "too few pages with a text layer to check on; on our sample "
                        "{0} of {1} pages found their own page. OK.", _counted(s_hits), _counted(s_n))
        else:
            u_hits, u_n = int(user["query_ranks_own_document_first"]), int(user["pairs"])
            line = Line("Wiring", "{0} of {1} of your pages found by a sentence of their own; on our sample "
                        "{2} of {3}: your pages look alike; wiring is fine on our sample.",
                        _counted(u_hits), _counted(u_n), _counted(s_hits), _counted(s_n))
        return Wiring(True, s_hits, s_n, True, line.render(), files=files, user=user, sample=sample, lines=(line,))
    block = Block([], title="Wiring check failed: the model does not find pages by their own words.")
    if user is not None:
        block.add("Your pages", "{0} of {1} found their own page", _counted(user["query_ranks_own_document_first"]),
                  _counted(user["pairs"]))
        block.lines.extend(_failure_detail(user))
    block.add("Our sample", "{0} of {1} found their own page", _counted(s_hits), _counted(s_n))
    block.lines.extend(_failure_detail(sample))
    block.add("Likely", "wrong weights, dtype or loader version for this model; nothing was written")
    model = encoder.record.get("model")
    block.next = f'ov.doctor("{model}")' if encoder.record.get("registered") else "ov.doctor()"
    raise WiringError(str(block))
