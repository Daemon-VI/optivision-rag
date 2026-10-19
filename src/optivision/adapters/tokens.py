"""Adapter for encoders that return token vectors and nothing else, and a check to run on any adapter.

:class:`TokenEncoderAdapter` wraps the encoders of :mod:`optivision.encoders.late` (or any
object with ``encode_queries`` and ``encode_images`` or ``encode_documents``) and hands their
arrays to the same checked conversion as ``ArrayAdapter``
(:func:`optivision.adapters.arrays.corpus_from_arrays`). The corpus carries no patch grid,
no protected rows and no page images, so the pixel-space ``spatial`` stage cannot run on it;
every vector-only stage can.

:func:`check_retriever` is the test to run before trusting a loading path this repository
has not exercised: it encodes a few document/query pairs and reports whether the vectors
are finite, of one width, unit-norm, and whether each query ranks its own document first.
Passing it shows that the model was loaded and driven correctly on those pairs. It says
nothing about how well any pipeline compresses the vectors.

Adapters produce vectors and descriptive identity only: nothing here chooses or alters
compression settings by model, and the registry status of a model changes only with a
committed measurement, never with an adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.vectors import MODALITIES, EncoderIdentity, MultiVectorCorpus
from . import EncoderAdapter, ModelInfo, _dtype_name, _positive_int, _text
from .arrays import corpus_from_arrays
from .nemotron import _split_pages

__all__ = ["TokenEncoderAdapter", "check_retriever"]


class TokenEncoderAdapter(EncoderAdapter):
    """Wraps an encoder that returns one ``[n_tokens, dim]`` array per item.

    For an image model ``documents`` are page images, or ``(PageRef, image)`` pairs as
    ``iter_pages`` yields them (the page id then becomes the document id unless ``ids`` is
    given); for a text model they are strings. ``queries`` are strings. The encoder's
    ``encode_documents`` is used when it has one, else ``encode_images``.

    ``info`` is the registry entry; it is passed explicitly (``load_adapter`` does it) and
    never looked up from the checkpoint name, so a status is only attached on purpose.
    """

    def __init__(self, encoder: Any, info: ModelInfo | None = None, modality: str | None = None) -> None:
        if not callable(getattr(encoder, "encode_queries", None)):
            raise TypeError(f"{type(encoder).__name__} has no encode_queries() method")
        if not any(callable(getattr(encoder, m, None)) for m in ("encode_documents", "encode_images")):
            raise TypeError(f"{type(encoder).__name__} has neither encode_documents() nor encode_images()")
        if modality is None and info is not None:
            modality = info.modality
        elif modality is None:  # a PyLate encoder is a text model
            pylate = getattr(encoder, "loader", None) == "pylate" and callable(getattr(encoder, "encode_documents", None))
            modality = "text" if pylate else "image"
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        self.encoder = encoder
        self.info = info
        self.modality = modality

    def identity(self) -> EncoderIdentity | None:
        """From the encoder: checkpoint, the revision it loaded, the width it produced, its dtype.

        The width is the encoder's ``dim`` (known after the first call for most of them), else
        the registry entry's. ``normalized`` stays None: unit norm is checked by
        :func:`check_retriever`, not assumed here.
        """
        enc, info = self.encoder, self.info
        model_id = _text(getattr(enc, "checkpoint", None)) or (info.model_id if info else None)
        dim = _positive_int(getattr(enc, "dim", None))
        dim = dim or (_positive_int(info.dimension) if info else None)
        if model_id is None or dim is None:
            return None
        extra = {}
        loader = _text(getattr(enc, "loader", None))
        if loader:
            extra["loader"] = loader
        if info is not None:
            extra["family"] = info.family
        return EncoderIdentity(
            model_id=model_id,
            revision=_text(getattr(enc, "revision", None)),
            dimension=dim,
            modality=self.modality,
            inference_dtype=_dtype_name(getattr(enc, "torch_dtype", None)),
            extra=extra,
        )

    def _attrs(self, identity: EncoderIdentity | None) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        for key, name in (("loader", "loader"), ("model", "checkpoint"), ("revision", "revision")):
            value = _text(getattr(self.encoder, name, None))
            if value:
                attrs[key] = value
        if self.info is not None:
            attrs["model_status"] = self.info.status
        report = getattr(self.encoder, "info", None)
        if callable(report):
            attrs["encoder_info"] = report()
        if identity is not None:
            attrs["encoder_identity"] = identity.to_dict()
        return attrs

    def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        refs = None
        encode = getattr(self.encoder, "encode_documents", None)
        if self.modality == "text" and callable(encode):
            arrays = encode([str(d) for d in documents])
        else:
            refs, images = _split_pages(documents)
            if ids is None and refs is not None:
                ids = [ref.page_id for ref in refs]
            arrays = (encode if callable(encode) else self.encoder.encode_images)(images)
        identity = self.identity()
        return corpus_from_arrays(
            arrays, ids, identity=identity, name="documents", attrs=self._attrs(identity),
            metadata=None if refs is None else [{"page_ref": dict(vars(ref))} for ref in refs],
        )

    def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        arrays = self.encoder.encode_queries([str(q) for q in queries])
        identity = self.identity()
        return corpus_from_arrays(arrays, ids, identity=identity, name="queries", attrs=self._attrs(identity))


def check_retriever(adapter: EncoderAdapter, documents: Sequence[Any], queries: Sequence[str], *,
                    expected_dim: int | None = None, allowed_misses: int = 2,
                    unit_norm_tolerance: float = 1e-2) -> dict[str, Any]:
    """Encode matched pairs and report whether the adapter behaves like a retriever.

    ``queries[i]`` must be a query whose answer is ``documents[i]``; a dozen pairs are
    enough. The report holds the width, the vectors per document and per query, whether
    every value is finite, the largest deviation of a row norm from 1, how many queries
    rank their own document first under exact MaxSim, and ``ok``: finite, one width (equal
    to ``expected_dim`` when given), unit-norm within ``unit_norm_tolerance``, and at most
    ``allowed_misses`` queries that do not rank their own document first. A tie with another
    document is a miss, and at least ``2 * allowed_misses + 2`` pairs are required.

    ``ok`` shows that the model was loaded and driven correctly on these pairs. It is not a
    measurement of retrieval quality and says nothing about compression.
    """
    if len(documents) != len(queries):
        raise ValueError(f"got {len(documents)} documents for {len(queries)} queries; pass matched pairs")
    # With few pairs, allowed_misses would let an encoder that ranks nothing pass: require
    # more than twice as many pairs as allowed misses, so a pass needs a majority of hits.
    min_pairs = max(2, 2 * int(allowed_misses) + 2)
    if len(documents) < min_pairs:
        raise ValueError(f"pass at least {min_pairs} document/query pairs for allowed_misses={allowed_misses}")
    docs = adapter.encode_documents(list(documents))
    qs = adapter.encode_queries(list(queries))
    doc_arrays = [np.asarray(docs[i].vectors, dtype=np.float32) for i in range(len(docs))]
    query_arrays = [np.asarray(qs[i].vectors, dtype=np.float32) for i in range(len(qs))]
    everything = np.concatenate([*doc_arrays, *query_arrays])
    finite = bool(np.isfinite(everything).all())
    norm_error = float(np.abs(np.linalg.norm(everything, axis=1) - 1.0).max()) if finite else float("nan")
    scores = np.array([[float((q @ d.T).max(axis=1).sum()) for d in doc_arrays] for q in query_arrays])
    # A hit is a query whose own document scores strictly above every other document; a tie
    # (as from a constant or collapsed encoder) is a miss, not argmax's first index.
    idx = np.arange(len(query_arrays))
    rivals = np.where(np.eye(len(idx), dtype=bool), -np.inf, scores).max(axis=1)
    own = int((scores[idx, idx] > rivals).sum())
    dim = int(docs.dimension)
    same_width = int(qs.dimension) == dim and (expected_dim is None or dim == int(expected_dim))
    report: dict[str, Any] = {
        "pairs": len(doc_arrays),
        "dimension": dim,
        "query_dimension": int(qs.dimension),
        "vectors_per_document": [int(d.shape[0]) for d in doc_arrays],
        "vectors_per_query": [int(q.shape[0]) for q in query_arrays],
        "finite": finite,
        "unit_norm_max_error": norm_error,
        "query_ranks_own_document_first": own,
    }
    report["ok"] = bool(finite and same_width and norm_error < unit_norm_tolerance
                        and own >= len(query_arrays) - int(allowed_misses))
    return report
