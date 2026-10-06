"""Encoders in, :class:`~optivision.representation.MultiVectorCorpus` out.

The compression layer never talks to a model. It takes vectors. This module is
the thin, replaceable bridge from a model to those vectors, and it is explicit
about how much of each bridge has actually been exercised.

Every known model carries one of three statuses:

``measured``
    Retrieval quality with and without compression has been measured by this
    repository on real encoder output (the numbers live in ``reports/``).
``untested``
    The loading code exists and uses a library path known to work for a sibling
    model, but this repository has not encoded or benchmarked this checkpoint.
``unverified``
    The adapter is written against a library's documented API, but that library
    has never been installed and run here. Treat it as a starting point.

The most general path needs no adapter at all: hand
:meth:`MultiVectorCorpus.from_arrays` the ``[n_i, d]`` arrays any model produced.

Three ways in
-------------
``ArrayAdapter`` takes arrays some model already produced and checks them;
``CallableAdapter`` wraps two functions that produce such arrays; ``load_adapter`` builds
the adapter of a model family this repository has a loader for. A custom adapter can reuse
the checked conversion, :func:`optivision.adapters.arrays.corpus_from_arrays`.

What an adapter is, and is not
------------------------------
An adapter encodes, returns a :class:`~optivision.representation.MultiVectorCorpus`, and
can describe the model behind the vectors (:meth:`EncoderAdapter.identity`,
:attr:`EncoderAdapter.dimension`). Adapters produce vectors and descriptive identity
only. An adapter never chooses or alters compression settings by model, never moves data
between devices or changes its dtype unless the caller said so, and nothing in the core
branches on the identity it reports. That an adapter exists for a model says nothing about
how well any pipeline compresses that model's vectors: that has to be measured on those
vectors, in the basis they are in.

Where layout lives
------------------
Model-specific layout (patch grids, protected instruction tokens, page images) is not part
of the adapter interface. ``PageEncoderAdapter`` carries it into the corpus it returns: the
aligned arrays ``positions`` and ``protected``, the per-document ``metadata`` and ``images``.
The core reads those arrays and never sees a ``PatchGrid`` or a page type.
"""

from __future__ import annotations

import abc
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.vectors import MODALITIES, EncoderIdentity
from ..representation import MultiVectorCorpus

MEASURED = "measured"
UNTESTED = "untested"
UNVERIFIED = "unverified"


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    family: str
    modality: str  # "image" (page screenshots) or "text"
    loader: str  # how the adapter obtains vectors
    dimension: int | None
    status: str
    evidence: str


_MODELS: list[ModelInfo] = [
    ModelInfo("vidore/colpali-v1.3-merged", "colpali", "image", "colpali-engine:colpali", 128,
              MEASURED, "E2/E3: reports/colpali_*; replayed 2026-09-27 (docs/AUDIT-2026-09-27.md)"),
    ModelInfo("vidore/colSmol-256M", "colsmol", "image", "colpali-engine:colsmol", 128,
              MEASURED, "E1/E4: reports/colsmol*; replayed 2026-09-27"),
    ModelInfo("vidore/colSmol-500M", "colsmol", "image", "colpali-engine:colsmol-500m", 128,
              UNTESTED, "same colpali-engine class as ColSmol-256M; never encoded here"),
    ModelInfo("vidore/colqwen2-v1.0-merged", "colqwen", "image", "colpali-engine:colqwen2", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each), encoded 2026-10-01 on a Kaggle T4 in "
              "float32; reports/universal/*/colqwen2-v1.0-merged_* (docs/UNIVERSAL.md, R11)"),
    ModelInfo("vidore/colqwen2-v1.0", "colqwen", "image", "colpali-engine:colqwen2", 128,
              MEASURED, "adapter-only repo, merged explicitly on load; gives the same vectors as the "
              "benchmarked vidore/colqwen2-v1.0-merged (median cosine 0.99999 on 6 pages, R11)"),
    ModelInfo("vidore/colqwen2.5-v0.2", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each), adapter merged explicitly over "
              "vidore/colqwen2.5-base, float32 split over two T4s, 2026-10-01; "
              "reports/universal/*/colqwen2.5-v0.2_* (docs/UNIVERSAL.md, R11)"),
    ModelInfo("nvidia/nemotron-colembed-vl-4b-v2", "nemotron-colembed", "image", "nemotron", 2560,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~750 vectors/page), float32 over two "
              "Kaggle T4s, pinned revision 0ed152d9, 2026-10-02; reports/universal/wide (docs/UNIVERSAL.md, R12); "
              "CC-BY-NC-4.0"),
    ModelInfo("nvidia/nemotron-colembed-vl-8b-v2", "nemotron-colembed", "image", "nemotron", 4096,
              UNTESTED, "Qwen3-VL-8B, same code as the 4B, 4,096-d; CC-BY-NC-4.0; pinned revision 34b64061; "
              "not encoded here"),
    ModelInfo("nomic-ai/colnomic-embed-multimodal-7b", "colqwen", "image", "sentence-transformers", 128,
              UNVERIFIED, "Qwen2.5-VL based; loading path not exercised here"),
    ModelInfo("colbert-ir/colbertv2.0", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "text late interaction; loading path not exercised here"),
    ModelInfo("answerdotai/answerai-colbert-small-v1", "colbert", "text", "sentence-transformers", 96,
              MEASURED, "BEIR SciFact (5,183 docs / 300 queries) via sentence-transformers 6.1.0 on "
              "2026-09-27; float nDCG@10 74.56 vs 74.77 on the model card; reports/universal/text"),
    ModelInfo("lightonai/GTE-ModernColBERT-v1", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "PyLate / sentence-transformers model; not exercised here"),
]

MODELS: dict[str, ModelInfo] = {m.model_id: m for m in _MODELS}


def supported_models(status: str | None = None) -> list[ModelInfo]:
    """Known models, optionally filtered by status. Unknown ids can still be used
    through :class:`SentenceTransformersAdapter` or precomputed vectors."""
    return [m for m in _MODELS if status is None or m.status == status]


# ------------------------------------------------------------------ adapters


def _positive_int(value: Any) -> int | None:
    """``value`` as a Python int if it is a positive integer (bool excluded), else None."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    return int(value) if value > 0 else None


def _text(value: Any) -> str | None:
    """``value`` if it is a non-empty string, else None."""
    return value if isinstance(value, str) and value else None


def _dtype_name(dtype: Any) -> str | None:
    """A dtype object as a plain name (``torch.float32`` -> ``"float32"``), or None."""
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _identity_from_info(info: ModelInfo | None) -> EncoderIdentity | None:
    """What a registry entry states about a model, or None without a known dimension.

    The revision, the normalisation flag and the inference dtype are not in the registry
    (its evidence text is prose and is never parsed), so they stay None. The registry
    status is not part of an identity: it says how well a model has been tested here, not
    which model it is.
    """
    if info is None or _positive_int(info.dimension) is None:
        return None
    return EncoderIdentity(
        model_id=info.model_id,
        dimension=int(info.dimension),
        modality=info.modality if info.modality in MODALITIES else "other",
        extra={"family": info.family, "loader": info.loader},
    )


class EncoderAdapter(abc.ABC):
    """Anything that turns documents and queries into multi-vector corpora.

    ``encode_documents`` and ``encode_queries`` are the contract: each returns a
    :class:`~optivision.representation.MultiVectorCorpus` with one document per input,
    with the padding rows of a batch already dropped. ``identity()`` and ``dimension``
    describe the model behind the vectors; they are concrete, with defaults built from
    ``info``, so a subclass written without them keeps working.
    """

    info: ModelInfo | None = None

    @abc.abstractmethod
    def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> MultiVectorCorpus: ...

    @abc.abstractmethod
    def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> MultiVectorCorpus: ...

    def identity(self) -> EncoderIdentity | None:
        """Descriptive identity of the model behind the vectors, or None when not known.

        The default is built from ``info`` when it names a model with a known dimension.
        Subclasses override it with whatever they can state; a field they cannot state
        stays None, and a revision or a normalisation flag is never guessed. Nothing in
        the core branches on the result.
        """
        return _identity_from_info(self.info)

    @property
    def dimension(self) -> int | None:
        """Vector dimension the adapter produces, or None when it is not known up front.

        Taken from :meth:`identity` unless the subclass assigned ``self.dimension``.
        """
        declared = _positive_int(getattr(self, "_declared_dimension", None))
        if declared is not None:
            return declared
        identity = self.identity()
        return None if identity is None else identity.dimension

    @dimension.setter
    def dimension(self, value: int | None) -> None:
        self._declared_dimension = value


class PageEncoderAdapter(EncoderAdapter):
    """Wraps any :class:`~optivision.encoders.BaseEncoder` (ColPali family, synthetic).

    Documents are ``(PageRef, PIL.Image)`` pairs, as produced by
    :func:`optivision.ingest.iter_pages`. Grid positions, protected instruction
    tokens and the page images are carried into the corpus, so both the
    vector-only stages and the pixel-space pruner can run on the result.
    """

    def __init__(self, encoder: Any, info: ModelInfo | None = None) -> None:
        self.encoder = encoder
        self.info = info

    def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        refs = [ref for ref, _ in documents]
        images = [img for _, img in documents]
        encodings = []
        for ref, img in zip(refs, images, strict=True):
            encodings.extend(self.encoder.encode_pages([img], [ref]))
        corpus = MultiVectorCorpus.from_page_encodings(encodings, images=images, attrs=self._attrs())
        if ids is not None:
            if len(ids) != len(corpus):
                raise ValueError(f"got {len(ids)} ids for {len(corpus)} encoded pages")
            corpus.ids = [str(i) for i in ids]
        return corpus

    def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        vectors = self.encoder.encode_queries(list(queries))
        return MultiVectorCorpus.from_arrays(vectors, ids=ids, attrs=self._attrs())

    def _attrs(self) -> dict[str, Any]:
        attrs = {"encoder": getattr(self.encoder, "name", type(self.encoder).__name__)}
        checkpoint = getattr(self.encoder, "checkpoint", None)
        if checkpoint:
            attrs["model"] = checkpoint
        if self.info is not None:
            attrs["model_status"] = self.info.status
        return attrs

    def identity(self) -> EncoderIdentity | None:
        """From the wrapped encoder where it states a fact, else from ``info``.

        Model id: the encoder's ``checkpoint``, else ``info.model_id``, else its ``name``.
        Dimension: the encoder's ``dim``, else ``info.dimension``. Modality: ``info`` if
        given, else ``"image"`` (this adapter encodes page images). Revision and inference
        dtype are read only if the encoder carries them (``revision``, ``torch_dtype``).
        The normalisation flag stays None: the ``BaseEncoder`` contract asks for unit-norm
        rows but an arbitrary wrapped encoder is not checked here.
        """
        enc, info = self.encoder, self.info
        model_id = _text(getattr(enc, "checkpoint", None)) or (info.model_id if info else None)
        name = _text(getattr(enc, "name", None))
        model_id = model_id or name
        try:
            dim = _positive_int(enc.dim)
        except Exception:  # noqa: BLE001 -- a property of an arbitrary encoder may fail to answer
            dim = None
        dim = dim or (_positive_int(info.dimension) if info else None)
        if model_id is None or dim is None:
            return None
        modality = ("image" if info is None
                    else info.modality if info.modality in MODALITIES else "other")
        extra = {"encoder": name} if name else {}
        if info is not None:
            extra["family"] = info.family
        return EncoderIdentity(
            model_id=model_id,
            revision=_text(getattr(enc, "revision", None)),
            dimension=dim,
            modality=modality,
            inference_dtype=_dtype_name(getattr(enc, "torch_dtype", None)),
            extra=extra,
        )


class SentenceTransformersAdapter(EncoderAdapter):
    """sentence-transformers ``MultiVectorEncoder`` (>= 6.0).

    Written against the API the model cards document -- ``encode_document(inputs)``
    and ``encode_query(texts)`` returning one ``[n_i, d]`` tensor or array per
    input. Run for real once, with sentence-transformers 6.1.0 on text
    (answerai-colbert-small-v1 on SciFact, whose float baseline matched the model
    card); the image path and every other model remain unverified. Pass ``model``
    to inject an already-constructed encoder (that is how the tests exercise the
    glue without the library).
    """

    def __init__(self, model_id: str, model: Any = None, **kwargs: Any) -> None:
        self.model_id = model_id
        self.info = MODELS.get(model_id)
        if self.info is None or self.info.status != MEASURED:
            warnings.warn(
                f"{model_id}: the sentence-transformers adapter is unverified in this "
                "repository; check retrieval quality before relying on it",
                stacklevel=2,
            )
        if model is None:
            try:
                from sentence_transformers import MultiVectorEncoder  # type: ignore[attr-defined]
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "SentenceTransformersAdapter needs sentence-transformers>=6.0 "
                    "(pip install 'sentence-transformers[image]>=6')"
                ) from exc
            model = MultiVectorEncoder(model_id, **kwargs)
        self.model = model

    def _to_corpus(self, outputs: Sequence[Any], ids: Sequence[str] | None) -> MultiVectorCorpus:
        arrays = [np.asarray(o.detach().cpu().float().numpy() if hasattr(o, "detach") else o) for o in outputs]
        attrs = {"model": self.model_id, "loader": "sentence-transformers",
                 "model_status": self.info.status if self.info else UNVERIFIED}
        return MultiVectorCorpus.from_arrays(arrays, ids=ids, attrs=attrs)

    def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        return self._to_corpus(self.model.encode_document(list(documents)), ids)

    def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        return self._to_corpus(self.model.encode_query(list(queries)), ids)

    def identity(self) -> EncoderIdentity | None:
        """From the registry entry only. The wrapped model's API is unverified in this
        repository, so it is not asked for a dimension or a revision."""
        return _identity_from_info(self.info)


def load_adapter(model_id: str, **kwargs: Any) -> EncoderAdapter:
    """Build the adapter a known model id calls for.

    colpali-engine models go through the repository's own encoder (which knows
    the patch grid); the NVIDIA ColEmbed models (loader ``"nemotron"``) go through
    :class:`~optivision.adapters.nemotron.NemotronAdapter`, with the registry's width as
    the width the checkpoint must have; everything else goes through sentence-transformers.
    """
    info = MODELS.get(model_id)
    if info is not None and info.loader.startswith("colpali-engine:"):
        from ..encoders.colvlm import ColVLMEncoder  # needs torch + colpali-engine

        backend = info.loader.split(":", 1)[1]
        return PageEncoderAdapter(ColVLMEncoder(backend=backend, model_name=model_id, **kwargs), info=info)
    if info is not None and info.loader == "nemotron":
        from ..encoders.nemotron import NemotronColEmbedEncoder  # needs torch + transformers

        kwargs.setdefault("expected_dim", info.dimension)
        return NemotronAdapter(NemotronColEmbedEncoder(model_id, **kwargs), info=info)
    return SentenceTransformersAdapter(model_id, **kwargs)


# The adapters below subclass ``EncoderAdapter``, so they are imported once it exists.
from .arrays import ArrayAdapter, CallableAdapter, corpus_from_arrays  # noqa: F401
from .nemotron import NemotronAdapter
