"""The opt-in small-memory layout, pages only (FINAL_DESIGN 5.7 step 6).

A *hot* copy that stays in RAM (one sign bit per number: ``binary``, or ``hm(0.25) > binary``)
shortlists ``candidates = max(50, ceil(0.02 * N))`` pages per query, and a memory-mapped 8-bit copy of
every vector (``int8(per_vector)``) rescores the shortlist only::

    TwoTierRetriever(CompressedRetriever(hot), CompressedRetriever(memory-mapped int8), candidates)

This is the campaign's two-tier row served from disk. The campaign measured it in RAM as
``TieredIndex(ExactIndex(hot), ExactIndex(int8 per vector), candidates=50)``
(``scripts/universal_study/s13_wide_study.py``); ``TwoTierRetriever`` reproduces ``TieredIndex.search``
for the top ``k <= candidates`` ranks, ties included (``core/retrieve/tiered.py``), so the catalogue's
tiered rows describe this layout only if the memory-mapped search returns the same rankings. That,
and the memory it holds, is what test V9 checks (``scripts/easy_validation/v9_small_memory.py``).

Why it saves memory. The hot copy is what has to be resident; the 8-bit copy sits on disk and only
the shortlisted pages are read from it, one page at a time (``ExactIndex.rescore``). It does not save
disk: both copies are stored. Nothing here claims a speed.

Refusals. The feature is switched off until V9 passes (``presets/easy_ledger.json``,
``small_memory_layout``): :func:`require` raises ``FeatureOff`` with the ledger's reason. Text models
are refused: a 1-bit shortlist lost most of the search quality on text in our benchmarks.

Every pipeline used here is stateless (no fitted mean, range or basis), so each segment of an index can
be encoded on its own and a segment's codes decode with nothing but the codes. Importing this module
loads NumPy and the core retrievers, not torch or SciPy (``hm(0.25)`` needs SciPy when it encodes).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import _ledger
from ._types import EasyError

#: the hot copies FINAL_DESIGN 5.7 allows, by catalogue label (campaign: "binary", "ward 1/4 > binary")
HOT_SPECS: dict[str, dict[str, Any]] = {
    "binary": {"stages": [{"stage": "binary"}]},
    "hm(0.25) > binary": {
        "stages": [
            {"stage": "hierarchical_merge", "ratio": 0.25, "max_distance": None, "min_vectors": 1},
            {"stage": "binary"},
        ]
    },
}

#: the rescoring copy: every vector, 8 bits per number with a float16 scale per vector
COLD_SPEC: dict[str, Any] = {"stages": [{"stage": "int8", "scale": "per_vector"}]}

#: the scan budget the index uses (CONTRACT: CompressedRetriever(..., max_block_bytes=32 MiB))
MAX_BLOCK_BYTES = 32 * 1024 * 1024

#: the fixed allowance of the V9 memory line (interpreter, NumPy, scan blocks, score matrix)
ALLOWANCE_BYTES = 128 * 10**6

_FORMAT = "optivision-easy-tier"


class SmallMemoryRefused(EasyError, ValueError):
    """The small-memory layout does not apply here (a text model, a stateful codec, misaligned copies)."""


# ---------------------------------------------------------------------------- the rule


def candidates_for(n_pages: int) -> int:
    """Shortlist size: ``max(50, ceil(0.02 * N))`` (FINAL_DESIGN 5.7 step 6)."""
    n = int(n_pages)
    if n < 0:
        raise ValueError(f"n_pages must be >= 0, got {n_pages!r}")
    return max(50, math.ceil(0.02 * n))


def require(modality: str = "image") -> None:
    """The gate every user-facing entry point calls before anything else.

    Raises ``FeatureOff`` while ``small_memory_layout`` is off in the ledger, then
    :class:`SmallMemoryRefused` for a text model.
    """
    _ledger.require("small_memory_layout")
    check_modality(modality)


def check_modality(modality: str) -> None:
    if str(modality).lower() == "text":
        raise SmallMemoryRefused(
            "the small-memory layout is for page models only: on text models a 1-bit shortlist lost most "
            "of the search quality in our benchmarks; use a one-copy layout"
        )


def hot_pipeline(hot: str = "binary") -> Any:
    """A fresh ``Pipeline`` for the hot copy (``"binary"`` or ``"hm(0.25) > binary"``)."""
    from ..compose import Pipeline

    if hot not in HOT_SPECS:
        raise SmallMemoryRefused(f"hot copy {hot!r} is not one of {sorted(HOT_SPECS)}")
    return Pipeline.from_dict(HOT_SPECS[hot])


def cold_pipeline() -> Any:
    """A fresh ``Pipeline`` for the memory-mapped 8-bit copy."""
    from ..compose import Pipeline

    return Pipeline.from_dict(COLD_SPEC)


def encode_tiers(corpus: Any, hot: str = "binary") -> tuple[Any, Any]:
    """``(hot, cold)`` ``CompressedCorpus`` of one float segment, aligned page for page.

    Both pipelines are stateless, so segments may be encoded one at a time and in any order.
    """
    hot_cc = hot_pipeline(hot).compress(corpus)
    cold_cc = cold_pipeline().compress(corpus)
    return hot_cc, cold_cc


# ---------------------------------------------------------------------------- one segment on disk


def write_segment(folder: str | Path, compressed: Any) -> Path:
    """Write a stateless ``CompressedCorpus`` as ``codes.npy``, ``offsets.npy`` and ``tier.json``.

    ``tier.json`` is written last through ``os.replace``, so a folder without it is incomplete.
    """
    if compressed.quantizer.overhead_bytes():
        raise SmallMemoryRefused(
            f"{compressed.quantizer.name} keeps fitted state; the small-memory copies must decode from codes alone"
        )
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "codes.npy", np.ascontiguousarray(compressed.codes, dtype=np.uint8))
    np.save(folder / "offsets.npy", np.asarray(compressed.offsets, dtype=np.int64))
    header = {
        "format": _FORMAT,
        "version": 1,
        "dim": int(compressed.dim),
        "ids": list(compressed.ids),
        "quantizer": compressed.quantizer.to_dict(),
        "pipeline": compressed.pipeline,
        "stats": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                  for k, v in dict(compressed.stats).items()},
    }
    tmp = folder / "tier.json.tmp"
    tmp.write_text(json.dumps(header, sort_keys=True), encoding="utf-8")
    os.replace(tmp, folder / "tier.json")
    return folder


def open_segment(folder: str | Path, *, mmap: bool = True) -> Any:
    """A ``CompressedCorpus`` over a folder written by :func:`write_segment`.

    With ``mmap=True`` (the default) the codes stay on disk and pages are read when scored;
    ``mmap=False`` loads them into RAM (the hot copy is meant to be loaded).
    """
    from ..compose import CompressedCorpus
    from ..stages.base import Quantizer, stage_from_dict

    folder = Path(folder)
    header = json.loads((folder / "tier.json").read_text(encoding="utf-8"))
    if header.get("format") != _FORMAT:
        raise ValueError(f"{folder} is not a small-memory tier folder")
    quantizer = stage_from_dict(header["quantizer"])
    if not isinstance(quantizer, Quantizer):
        raise TypeError(f"{folder}: tier.json names a non-quantizer stage")
    codes = np.load(folder / "codes.npy", mmap_mode="r" if mmap else None)
    offsets = np.load(folder / "offsets.npy")
    return CompressedCorpus(codes, offsets, header["dim"], quantizer, header["ids"],
                            stats=header.get("stats"), pipeline=header.get("pipeline"))


# ---------------------------------------------------------------------------- many segments as one store


class SegmentChain:
    """Several ``CompressedCorpus`` segments read as one store, without copying their codes.

    Implements the store interface the scorers read (``offsets``, ``decode_rows``, ``len``, ``ids``,
    ``dim``), so ``CompressedRetriever`` scans or rescores the chain exactly as it would one
    concatenated store: every codec here decodes row by row, so a row decodes to the same float32
    values whichever segment holds it. Segments must share the dimension and the codec.
    """

    def __init__(self, segments: Sequence[Any]) -> None:
        segments = list(segments)
        if not segments:
            raise ValueError("a segment chain needs at least one segment")
        dims = {int(s.dim) for s in segments}
        if len(dims) != 1:
            raise ValueError(f"segments have different dimensions: {sorted(dims)}")
        codecs = {json.dumps(s.quantizer.to_dict(), sort_keys=True) for s in segments}
        if len(codecs) != 1:
            raise ValueError(f"segments use different codecs: {sorted(codecs)}")
        self.segments = segments
        self.dim = dims.pop()
        self.quantizer = segments[0].quantizer
        counts = [np.diff(np.asarray(s.offsets, dtype=np.int64)) for s in segments]
        self.offsets = np.concatenate([[0], np.cumsum(np.concatenate(counts))]).astype(np.int64)
        self._row_starts = np.cumsum([0] + [int(s.num_vectors) for s in segments]).astype(np.int64)
        self.ids: list[str] = [i for s in segments for i in s.ids]
        if len(self.ids) != len(self.offsets) - 1:
            raise ValueError("segment ids and offsets disagree")

    def __len__(self) -> int:
        return int(self.offsets.size) - 1

    @property
    def num_vectors(self) -> int:
        return int(self._row_starts[-1])

    @property
    def nbytes(self) -> int:
        return int(sum(int(s.nbytes) for s in self.segments))

    def decode_rows(self, lo: int, hi: int) -> np.ndarray:
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            return np.zeros((0, self.dim), dtype=np.float32)
        s = int(np.searchsorted(self._row_starts, lo, side="right")) - 1
        parts = []
        while s < len(self.segments) and int(self._row_starts[s]) < hi:
            base = int(self._row_starts[s])
            a, b = max(lo, base) - base, min(hi, int(self._row_starts[s + 1])) - base
            if b > a:
                parts.append(self.segments[s].decode_rows(a, b))
            s += 1
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


def as_store(x: Any) -> Any:
    """One ``CompressedCorpus`` as is; a list of segments as a :class:`SegmentChain`."""
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) == 1 else SegmentChain(x)
    return x


# ---------------------------------------------------------------------------- the searcher


@dataclass(frozen=True)
class MemoryBound:
    """The V9 memory line: resident <= hot copy + shortlist decode + 128 MB (decimal bytes)."""

    hot_bytes: int
    shortlist_decode_bytes: int
    allowance_bytes: int = ALLOWANCE_BYTES

    @property
    def total(self) -> int:
        return int(self.hot_bytes + self.shortlist_decode_bytes + self.allowance_bytes)


class TieredSearcher:
    """Hot shortlist, memory-mapped 8-bit rescoring, exact over the stored codes.

    Args:
        hot: the hot copy, one ``CompressedCorpus`` or a list of segments (loaded in RAM).
        cold: the 8-bit copy, aligned page for page with ``hot`` (normally memory-mapped).
        hot_pipeline, cold_pipeline: the pipelines that made them (used to check and project
            queries; both are stateless here). Default: :data:`HOT_SPECS` / :data:`COLD_SPEC` rebuilt.
        candidates: shortlist size; default :func:`candidates_for` of the page count.
        max_block_bytes: scan budget of both tiers.

    This class does not consult the feature switch, so tests and the V9 script can run it while the
    feature is off; user-facing entry points call :func:`require` before it.
    """

    def __init__(
        self,
        hot: Any,
        cold: Any,
        *,
        hot_pipeline: Any = None,
        cold_pipeline: Any = None,
        candidates: int | None = None,
        max_block_bytes: int = MAX_BLOCK_BYTES,
    ) -> None:
        from ..core.retrieve.exact import CompressedRetriever
        from ..core.retrieve.tiered import TwoTierRetriever

        hot_store, cold_store = as_store(hot), as_store(cold)
        cold_codec = cold_store.quantizer.to_dict()
        if cold_codec != COLD_SPEC["stages"][0]:
            raise SmallMemoryRefused(f"the rescoring copy must be int8(per_vector), got {cold_codec}")
        if hot_store.quantizer.name != "binary":
            raise SmallMemoryRefused(f"the hot copy must be 1-bit, got {hot_store.quantizer.name}")
        hot_ids, cold_ids = list(hot_store.ids), list(cold_store.ids)
        if hot_ids != cold_ids:
            raise SmallMemoryRefused("the hot and 8-bit copies must hold the same pages in the same order")
        self.hot_store, self.cold_store = hot_store, cold_store
        self.n_pages = len(cold_store)
        self.candidates = int(candidates) if candidates is not None else candidates_for(self.n_pages)
        self.max_block_bytes = int(max_block_bytes)
        hp = hot_pipeline if hot_pipeline is not None else _rebuilt(hot_store)
        cp = cold_pipeline if cold_pipeline is not None else _rebuilt(cold_store)
        self.hot_retriever = CompressedRetriever(hot_store, hp, self.max_block_bytes)
        self.cold_retriever = CompressedRetriever(cold_store, cp, self.max_block_bytes)
        self.retriever = TwoTierRetriever(self.hot_retriever, self.cold_retriever, self.candidates)

    # ------------------------------------------------------------ search

    def ranking(self, queries: Any, k: int = 5) -> Any:
        """The ``Ranking`` of ``TwoTierRetriever.search`` (``k`` may not exceed the shortlist)."""
        return self.retriever.search(queries, k)

    def search(self, queries: Any, k: int = 5) -> list[list[tuple[str, float]]]:
        """Top ``k`` ``(page id, score)`` pairs per query, scores on the 8-bit copy's scale."""
        r = self.ranking(queries, k)
        return [[(doc, float(r.scores[q, j])) for j, doc in enumerate(row)] for q, row in enumerate(r.ids)]

    # ------------------------------------------------------------ sizes

    @property
    def hot_bytes(self) -> int:
        return int(self.hot_store.nbytes)

    @property
    def cold_bytes(self) -> int:
        return int(self.cold_store.nbytes)

    def memory_bound(self) -> MemoryBound:
        """Hot copy + decoding one shortlist of mean-size pages to float32 + the 128 MB allowance."""
        vectors_per_page = self.cold_store.num_vectors / max(1, self.n_pages)
        decode = math.ceil(min(self.candidates, self.n_pages) * vectors_per_page * self.cold_store.dim * 4)
        return MemoryBound(self.hot_bytes, decode)


def _rebuilt(store: Any) -> Any:
    from ..compose import Pipeline

    spec = getattr(store, "pipeline", None)
    if spec is None and isinstance(store, SegmentChain):
        spec = store.segments[0].pipeline
    return Pipeline.from_dict(spec) if spec else None


def open_searcher(hot_folders: Sequence[str | Path], cold_folders: Sequence[str | Path], *,
                  modality: str = "image", candidates: int | None = None) -> TieredSearcher:
    """The user-facing entry point: gate, load the hot copy, memory-map the 8-bit copy."""
    require(modality)
    hot = [open_segment(f, mmap=False) for f in hot_folders]
    cold = [open_segment(f, mmap=True) for f in cold_folders]
    return TieredSearcher(hot, cold, candidates=candidates)
