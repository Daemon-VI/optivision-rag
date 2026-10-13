"""Per-query evidence: what a selection rule is allowed to see.

An :class:`Evidence` holds, for one corpus, one query set, one metric and one reference,
the per-query metric value of the uncompressed index and of every candidate of a frozen
candidate space, in the order the queries will enter the test, plus the stored size of each
candidate. Nothing else reaches a rule: no encoder name, no dimension, no dataset name.

``construct`` names what the metric was measured against (``"labels"`` or
``"agreement@1"``). The two constructs are different quantities and are never merged:
agreement with the uncompressed ranking measures fidelity to that ranking, including its
mistakes, and is not retrieval quality.

An Evidence is immutable: the arrays are private read-only copies. It is built by
``collect_evidence`` (``core/evaluate/collect.py``) or directly from arrays. Structure is
checked here (shapes, float64, lengths); the values (finite, in [0, 1], positive mean
baseline) are checked when a rule runs, where an invalid value is an error and is never
repaired.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from optivision.core.identity import array_digest, canonical_json, digest, strings_digest

_FORMAT = "optivision-evidence"
_VERSION = 1


class EvidenceFormatError(ValueError):
    """The structure of an Evidence (shape, dtype, length) or of its file is wrong."""


def _array(name: str, value: Any, ndim: int) -> np.ndarray:
    a = np.array(value, copy=True, order="C")
    if a.dtype != np.float64:
        raise EvidenceFormatError(
            f"{name} must be a native float64 array (no silent dtype conversion), "
            f"got dtype {a.dtype}")
    if a.ndim != ndim:
        raise EvidenceFormatError(f"{name} must have {ndim} dimension(s), got shape {a.shape}")
    a.setflags(write=False)
    return a


def _strings(name: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
        raise EvidenceFormatError(f"{name} must be a sequence of strings")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str):
            raise EvidenceFormatError(f"{name} must hold strings, got {type(item).__name__}")
    return out


def _fitted(value: Any, n_candidates: int) -> tuple[tuple[str | None, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
        raise EvidenceFormatError("fitted must be a sequence of per-candidate digest sequences")
    out = []
    for k, entry in enumerate(value):
        if isinstance(entry, (str, bytes, dict)) or not isinstance(entry, Iterable):
            raise EvidenceFormatError(
                f"fitted[{k}] must be a sequence of str | None digests, one per stage, "
                f"got {type(entry).__name__}")
        items = tuple(entry)
        for item in items:
            if item is not None and not isinstance(item, str):
                raise EvidenceFormatError(f"fitted[{k}] must hold str | None, "
                                          f"got {type(item).__name__}")
        out.append(items)
    if out and len(out) != n_candidates:
        raise EvidenceFormatError(f"fitted has {len(out)} entries for {n_candidates} "
                                  "candidates; use an empty tuple when unknown")
    return tuple(out)


@dataclass(frozen=True, eq=False, repr=False)
class Evidence:
    """Per-query metric values of the baseline and of K candidates, in test order.

    ``baseline`` is ``[n]`` and ``candidates`` is ``[K, n]``, both float64, the same metric
    on the same queries; ``query_ids`` (length n) are in the order the queries enter the
    test. ``bytes_per_doc`` and ``vectors_per_doc`` (``[K]``) are the library's stored size
    per document and vectors kept per document; fitted-state bytes are not in the key.
    ``space_identity`` binds the evidence to the candidate space it was collected on.
    ``dropped_queries`` are the queries removed, for every candidate and the baseline
    together, by a rule that looks only at the reference (no relevant document).
    ``fitted`` has, per candidate, one fitted-state digest per pipeline stage (``None`` for a
    stage without state), or is empty when unknown.

    Two Evidence objects are equal when their digests are equal.
    """

    metric: str
    construct: str
    baseline: np.ndarray
    candidates: np.ndarray
    query_ids: tuple[str, ...]
    bytes_per_doc: np.ndarray
    vectors_per_doc: np.ndarray
    space_identity: str
    dropped_queries: tuple[str, ...] = ()
    fitted: tuple[tuple[str | None, ...], ...] = ()

    def __post_init__(self) -> None:
        for name in ("metric", "construct", "space_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise EvidenceFormatError(f"{name} must be a non-empty string, got {value!r}")
        baseline = _array("baseline", self.baseline, 1)
        candidates = _array("candidates", self.candidates, 2)
        size = _array("bytes_per_doc", self.bytes_per_doc, 1)
        vectors = _array("vectors_per_doc", self.vectors_per_doc, 1)
        ids = _strings("query_ids", self.query_ids)
        dropped = _strings("dropped_queries", self.dropped_queries)
        n_cand, n = candidates.shape
        if n_cand < 1:
            raise EvidenceFormatError("candidates must hold at least one candidate")
        if baseline.shape != (n,):
            raise EvidenceFormatError(f"baseline must have shape ({n},) to match candidates "
                                      f"{candidates.shape}, got {baseline.shape}")
        if len(ids) != n:
            raise EvidenceFormatError(f"query_ids has {len(ids)} entries for {n} queries")
        for name, arr in (("bytes_per_doc", size), ("vectors_per_doc", vectors)):
            if arr.shape != (n_cand,):
                raise EvidenceFormatError(f"{name} must have shape ({n_cand},), got {arr.shape}")
        fitted = _fitted(self.fitted, n_cand)
        for name, value in (("baseline", baseline), ("candidates", candidates),
                            ("bytes_per_doc", size), ("vectors_per_doc", vectors),
                            ("query_ids", ids), ("dropped_queries", dropped), ("fitted", fitted)):
            object.__setattr__(self, name, value)

    # ------------------------------------------------------------------ views

    @property
    def n_queries(self) -> int:
        return int(self.baseline.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.shape[0])

    def digest(self) -> str:
        """SHA-256 over every field: metric, construct, arrays, query ids, space identity,
        fitted digests and dropped queries. Any change to any of them changes it."""
        cached = self.__dict__.get("_digest")
        if cached is None:
            cached = digest({
                "format": _FORMAT,
                "version": _VERSION,
                "metric": self.metric,
                "construct": self.construct,
                "baseline": array_digest(self.baseline),
                "candidates": array_digest(self.candidates),
                "bytes_per_doc": array_digest(self.bytes_per_doc),
                "vectors_per_doc": array_digest(self.vectors_per_doc),
                "query_ids": strings_digest(self.query_ids),
                "space_identity": self.space_identity,
                "fitted": [list(entry) for entry in self.fitted],
                "dropped_queries": strings_digest(self.dropped_queries),
            })
            object.__setattr__(self, "_digest", cached)
        return cached

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Evidence) and self.digest() == other.digest()

    def __hash__(self) -> int:
        return hash(self.digest())

    def __copy__(self) -> Evidence:
        return self  # immutable: the arrays are read-only and owned by this object

    def __deepcopy__(self, memo: dict) -> Evidence:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (Evidence, (self.metric, self.construct, self.baseline, self.candidates,
                           self.query_ids, self.bytes_per_doc, self.vectors_per_doc,
                           self.space_identity, self.dropped_queries, self.fitted))

    def __repr__(self) -> str:
        return (f"Evidence(metric={self.metric!r}, construct={self.construct!r}, "
                f"n_candidates={self.n_candidates}, n_queries={self.n_queries}, "
                f"digest={self.digest()[:12]!r})")

    def subset(self, indices: Any) -> Evidence:
        """The same evidence on the queries ``indices``, in that order (a reorder or a prefix).

        ``indices`` is a 1-D integer array of positions in ``[0, n)``; negative values are
        refused rather than wrapped, and repeated positions are allowed. A repeated position
        counts the same query again: the test then sees dependent draws, assumption A1 of the
        rule's statement does not hold, and resampling with replacement is outside it (the
        caller's decision). Candidates, sizes, fitted digests and dropped queries are
        unchanged. The result has its own digest.
        """
        idx = np.asarray(indices)
        if idx.ndim != 1 or (idx.size and idx.dtype.kind not in "iu"):
            raise EvidenceFormatError("indices must be a 1-D array of integers")
        idx = idx.astype(np.int64, copy=False)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self.n_queries):
            raise EvidenceFormatError(
                f"indices must lie in [0, {self.n_queries}), got range "
                f"[{int(idx.min())}, {int(idx.max())}]")
        return Evidence(
            metric=self.metric,
            construct=self.construct,
            baseline=self.baseline[idx],
            candidates=self.candidates[:, idx],
            query_ids=tuple(self.query_ids[i] for i in idx.tolist()),
            bytes_per_doc=self.bytes_per_doc,
            vectors_per_doc=self.vectors_per_doc,
            space_identity=self.space_identity,
            dropped_queries=self.dropped_queries,
            fitted=self.fitted,
        )

    # ------------------------------------------------------------------ files

    def to_npz(self, path: str | Path) -> Path:
        """Write a pickle-free ``.npz``: four float64 arrays and a canonical-JSON header."""
        path = Path(path)
        target = path if path.suffix == ".npz" else Path(str(path) + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "format": _FORMAT,
            "version": _VERSION,
            "metric": self.metric,
            "construct": self.construct,
            "query_ids": list(self.query_ids),
            "space_identity": self.space_identity,
            "dropped_queries": list(self.dropped_queries),
            "fitted": [list(entry) for entry in self.fitted],
            "digest": self.digest(),
        }
        np.savez(
            target,
            baseline=self.baseline,
            candidates=self.candidates,
            bytes_per_doc=self.bytes_per_doc,
            vectors_per_doc=self.vectors_per_doc,
            header=np.frombuffer(canonical_json(header), dtype=np.uint8),
        )
        return target

    @classmethod
    def from_npz(cls, path: str | Path) -> Evidence:
        """Read a file written by :meth:`to_npz`. Never unpickles; checks the stored digest."""
        with np.load(path, allow_pickle=False) as data:
            missing = {"baseline", "candidates", "bytes_per_doc", "vectors_per_doc",
                       "header"} - set(data.files)
            if missing:
                raise EvidenceFormatError(f"{path} is not an {_FORMAT} file (missing "
                                          f"{sorted(missing)})")
            header = json.loads(bytes(data["header"]).decode("utf-8"))
            arrays = {name: data[name] for name in
                      ("baseline", "candidates", "bytes_per_doc", "vectors_per_doc")}
        if header.get("format") != _FORMAT:
            raise EvidenceFormatError(f"{path}: unexpected format {header.get('format')!r}")
        if header.get("version") != _VERSION:
            raise EvidenceFormatError(f"{path}: unsupported version {header.get('version')!r}")
        evidence = cls(
            metric=header["metric"],
            construct=header["construct"],
            baseline=arrays["baseline"],
            candidates=arrays["candidates"],
            query_ids=tuple(header["query_ids"]),
            bytes_per_doc=arrays["bytes_per_doc"],
            vectors_per_doc=arrays["vectors_per_doc"],
            space_identity=header["space_identity"],
            dropped_queries=tuple(header["dropped_queries"]),
            fitted=tuple(tuple(entry) for entry in header["fitted"]),
        )
        if evidence.digest() != header.get("digest"):
            raise EvidenceFormatError(f"{path}: the stored digest does not match the content; "
                                      "the file was altered or is corrupt")
        return evidence
