"""NVIDIA ColEmbed (Nemotron) page encoders as an adapter.

:class:`~optivision.encoders.nemotron.NemotronColEmbedEncoder` returns one
``[n_tokens, dim]`` float32 array per page and per query, with the padded positions
already dropped and every row L2-normalised by its own code. It has no ``encode_pages``,
no patch grid and no ``PageEncoding``, so ``PageEncoderAdapter`` cannot wrap it. This
adapter hands those arrays to the same checked conversion as ``ArrayAdapter``
(:func:`optivision.adapters.arrays.corpus_from_arrays`), so what comes out has passed the
vector contract whatever the encoder did.

``load_adapter`` had no branch for the registry loader ``"nemotron"`` and fell through to
the sentence-transformers adapter; it now builds this one.

What the corpus does not carry: no ``positions``, no ``protected`` rows and no page
images, because the encoder does not expose a patch grid. A pixel-space stage such as
``spatial`` therefore cannot run on it; vector-only stages can. Nothing here fabricates a
layout.

Adapters produce vectors and descriptive identity only: nothing here chooses or alters
compression settings by model. That this adapter exists says nothing about how well any
pipeline compresses these vectors; the registry status of a model changes only with a
committed measurement, never with an adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.vectors import ContractError, EncoderIdentity, MultiVectorCorpus
from . import EncoderAdapter, ModelInfo, _dtype_name, _positive_int, _text
from .arrays import corpus_from_arrays

__all__ = ["NemotronAdapter"]


def _split_pages(documents: Sequence[Any]) -> tuple[list[Any] | None, list[Any]]:
    """``(refs, images)`` from page images or from ``(PageRef, image)`` pairs.

    The pair form is what :func:`optivision.ingest.iter_pages` yields and what
    ``PageEncoderAdapter`` takes, so one loop can feed either adapter. A call must use one
    form throughout; ``refs`` is None for bare images.
    """
    items = list(documents)
    pairs = [isinstance(d, tuple) and len(d) == 2 for d in items]
    if any(pairs) and not all(pairs):
        raise ContractError("documents must be all page images or all (PageRef, image) pairs")
    if items and all(pairs):
        for i, (ref, _) in enumerate(items):
            if not isinstance(getattr(ref, "page_id", None), str):
                raise ContractError(f"documents[{i}][0] must be a PageRef (it has no page_id), "
                                    f"got {type(ref).__name__}")
        return [d[0] for d in items], [d[1] for d in items]
    return None, items


class NemotronAdapter(EncoderAdapter):
    """Wraps a ``NemotronColEmbedEncoder`` (or anything with its two ``encode_*`` methods).

    ``documents`` are page images, or ``(PageRef, image)`` pairs as ``iter_pages`` yields
    them (the page id then becomes the document id unless ``ids`` is given, and the page
    reference goes into the document's metadata). ``queries`` are strings. The corpus
    ``attrs`` carry the model, its pinned revision, the loader, the registry status when
    ``info`` is given, the identity as a dict, and what the encoder reports about the run
    (``encoder_info``: device, dtype, attention implementation, loading report).

    ``info`` is the registry entry; it is passed explicitly (``load_adapter`` does it) and
    never looked up from the checkpoint name, so a status is only attached on purpose.
    """

    def __init__(self, encoder: Any, info: ModelInfo | None = None) -> None:
        for method in ("encode_images", "encode_queries"):
            if not callable(getattr(encoder, method, None)):
                raise TypeError(f"{type(encoder).__name__} has no {method}() method")
        self.encoder = encoder
        self.info = info

    def identity(self) -> EncoderIdentity | None:
        """From the encoder: checkpoint, pinned revision, width, inference dtype.

        ``normalized`` is True only for a real ``NemotronColEmbedEncoder``, whose
        ``_embed`` normalises every row (``F.normalize``) after masking; for any other
        wrapped object it stays None. The dimension falls back to the registry entry.
        """
        from ..encoders.nemotron import NemotronColEmbedEncoder

        enc, info = self.encoder, self.info
        model_id = _text(getattr(enc, "checkpoint", None)) or (info.model_id if info else None)
        dim = _positive_int(getattr(enc, "dim", None))
        dim = dim or (_positive_int(info.dimension) if info else None)
        if model_id is None or dim is None:
            return None
        extra = {"family": "nemotron-colembed"}
        attention = _text(getattr(enc, "attn_implementation", None))
        if attention:
            extra["attn_implementation"] = attention
        return EncoderIdentity(
            model_id=model_id,
            revision=_text(getattr(enc, "revision", None)),
            dimension=dim,
            modality="image",
            normalized=True if isinstance(enc, NemotronColEmbedEncoder) else None,
            inference_dtype=_dtype_name(getattr(enc, "torch_dtype", None)),
            extra=extra,
        )

    def _attrs(self, identity: EncoderIdentity | None) -> dict[str, Any]:
        attrs: dict[str, Any] = {"loader": "nemotron"}
        checkpoint = _text(getattr(self.encoder, "checkpoint", None))
        if checkpoint:
            attrs["model"] = checkpoint
        revision = _text(getattr(self.encoder, "revision", None))
        if revision:
            attrs["revision"] = revision
        if self.info is not None:
            attrs["model_status"] = self.info.status
        report = getattr(self.encoder, "info", None)
        if callable(report):
            attrs["encoder_info"] = report()
        if identity is not None:
            attrs["encoder_identity"] = identity.to_dict()
        return attrs

    def encode_documents(
        self, documents: Sequence[Any], ids: Sequence[str] | None = None
    ) -> MultiVectorCorpus:
        refs, images = _split_pages(documents)
        identity = self.identity()
        if ids is None and refs is not None:
            ids = [ref.page_id for ref in refs]
        arrays = self.encoder.encode_images(images)
        return corpus_from_arrays(
            arrays, ids, identity=identity, name="documents", attrs=self._attrs(identity),
            metadata=None if refs is None else [{"page_ref": dict(vars(ref))} for ref in refs],
        )

    def encode_queries(
        self, queries: Sequence[str], ids: Sequence[str] | None = None
    ) -> MultiVectorCorpus:
        identity = self.identity()
        arrays = self.encoder.encode_queries(list(queries))
        return corpus_from_arrays(arrays, ids, identity=identity, name="queries",
                                  attrs=self._attrs(identity))
