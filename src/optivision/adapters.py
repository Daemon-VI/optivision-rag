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
"""

from __future__ import annotations

import abc
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .representation import MultiVectorCorpus

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


class EncoderAdapter(abc.ABC):
    """Anything that turns documents and queries into multi-vector corpora."""

    info: ModelInfo | None = None

    @abc.abstractmethod
    def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> MultiVectorCorpus: ...

    @abc.abstractmethod
    def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> MultiVectorCorpus: ...


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


def load_adapter(model_id: str, **kwargs: Any) -> EncoderAdapter:
    """Build the adapter a known model id calls for.

    colpali-engine models go through the repository's own encoder (which knows
    the patch grid); everything else goes through sentence-transformers.
    """
    info = MODELS.get(model_id)
    if info is not None and info.loader.startswith("colpali-engine:"):
        from .encoders.colvlm import ColVLMEncoder  # needs torch + colpali-engine

        backend = info.loader.split(":", 1)[1]
        return PageEncoderAdapter(ColVLMEncoder(backend=backend, model_name=model_id, **kwargs), info=info)
    return SentenceTransformersAdapter(model_id, **kwargs)
