"""Candidate spaces: a frozen, ordered, hashed list of pipelines grouped into families.

A selection rule is only as meaningful as the list it tests. The list, its order, its
family labels and the order inside each family are fixed *before* any calibration query
is seen (assumption A3 of the rule's statement), so a space is an immutable value with an
identity: the SHA-256 of the canonical JSON of the ordered ``{position, family, pipeline}``
list. That is the form the registered experiments hashed, so the library's default list
at any width hashes to :data:`REGISTERED_DEFAULT_SPACE_SHA256` when SciPy is installed.

Nothing here de-duplicates or reorders candidates. Duplicated pipelines are tested as
the separate candidates they are, and the number of families (which sets the level each
chain is tested at) is counted from the list actually passed. Changing the list changes the
number of families and therefore the test level.

The default list is a frozen, development-informed default: it was shaped by results on
the development encoders. It is neither neutral nor chosen for a particular encoder, and
using it on an unseen encoder is an untested transfer, not a finding.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Any

import numpy as np

from optivision.calibration import recommended_search_space, wide_search_space
from optivision.compose import Pipeline
from optivision.core.identity import digest

#: SHA-256 of the default list the registered experiments used (``scripts/research/x_common.py``,
#: ``S0_SPEC_SHA256``). It exists only with SciPy installed: without it the Ward chains are
#: missing and the list is a different, reduced space.
REGISTERED_DEFAULT_SPACE_SHA256 = "af29b01146a90a048ffea9cfbd66f0fedf988134829cd4dc56861f1207893520"

#: ``CandidateSpec.position`` of a pipeline that is not a member of any space (an explicit
#: fallback recorded in a decision).
NOT_IN_SPACE = -1

_FORMAT = "optivision-candidate-space"
_VERSION = 1


def freeze_json(obj: Any) -> Any:
    """Deep, read-only copy of a JSON-native value: dicts become mappings, lists tuples.

    Shared with ``decision.py``. Numpy scalars become Python numbers; NaN, infinity,
    non-string keys and unknown types raise.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if not math.isfinite(value):
            raise ValueError("a JSON value cannot be NaN or infinity")
        return value
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON keys must be strings, got {type(key).__name__}")
            out[key] = freeze_json(value)
        return MappingProxyType(out)
    if isinstance(obj, (list, tuple)):
        return tuple(freeze_json(v) for v in obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-native")


def thaw_json(obj: Any) -> Any:
    """Plain mutable dicts and lists from a value made by :func:`freeze_json` (a fresh copy)."""
    if isinstance(obj, Mapping):
        return {k: thaw_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [thaw_json(v) for v in obj]
    return obj


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate: its place in the list, its family label and its pipeline description.

    ``pipeline`` is ``Pipeline.to_dict()``, held as a read-only mapping so that a space
    cannot change under a decision that refers to it. ``position`` is the index in the
    flattened list and is the last component of the selection's size key.
    """

    position: int
    family: str
    pipeline: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or not isinstance(self.position, (int, np.integer)):
            raise TypeError(f"position must be an int, got {type(self.position).__name__}")
        if int(self.position) < NOT_IN_SPACE:
            raise ValueError(f"position must be >= {NOT_IN_SPACE}, got {self.position}")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError(f"family must be a non-empty string, got {self.family!r}")
        if not isinstance(self.pipeline, Mapping):
            raise TypeError("pipeline must be a mapping (Pipeline.to_dict())")
        object.__setattr__(self, "position", int(self.position))
        object.__setattr__(self, "pipeline", freeze_json(self.pipeline))

    def __hash__(self) -> int:
        return hash((self.position, self.family, digest(thaw_json(self.pipeline))))

    def __reduce__(self) -> tuple[Any, ...]:
        return (CandidateSpec, (self.position, self.family, thaw_json(self.pipeline)))

    def to_dict(self) -> dict[str, Any]:
        """``{"position", "family", "pipeline"}`` as plain JSON-native values (a fresh copy)."""
        return {"position": self.position, "family": self.family,
                "pipeline": thaw_json(self.pipeline)}

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> CandidateSpec:
        if set(spec) != {"position", "family", "pipeline"}:
            raise ValueError("a candidate needs exactly position, family and pipeline, "
                             f"got {sorted(spec)}")
        return cls(spec["position"], spec["family"], spec["pipeline"])

    def to_pipeline(self) -> Pipeline:
        """A fresh, unfitted :class:`~optivision.compose.Pipeline` for this candidate."""
        return Pipeline.from_dict(thaw_json(self.pipeline))


def _have_scipy() -> bool:
    try:
        import scipy.cluster.hierarchy  # noqa: F401
    except ImportError:
        return False
    return True


def _check_dim(dim: int) -> None:
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or int(dim) < 1:
        raise ValueError(f"dim must be a positive integer, got {dim!r}")


@dataclass(frozen=True)
class CandidateSpace:
    """An immutable, ordered list of candidates, grouped into families by label.

    Families are grouping keys compared for equality only. The order inside a family is the
    pre-specified order of the fixed-sequence test (least to most aggressive); validity does
    not need it to be monotone, power does. Positions are ``0 .. K-1`` in list order.
    ``name`` is a label for humans and is not part of the identity.
    """

    name: str
    candidates: tuple[CandidateSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a candidate space needs a non-empty name")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("a candidate space must hold at least one candidate")
        for k, spec in enumerate(candidates):
            if not isinstance(spec, CandidateSpec):
                raise TypeError(f"candidate {k} is a {type(spec).__name__}, not a CandidateSpec")
            if spec.position != k:
                raise ValueError(f"candidate {k} has position {spec.position}; positions must "
                                 "be 0..K-1 in list order")
        object.__setattr__(self, "candidates", candidates)

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_families(cls, families: Mapping[str, Sequence[Pipeline]],
                      name: str = "custom") -> CandidateSpace:
        """Flatten ``{family: [pipelines, least to most aggressive]}`` in mapping order.

        Duplicated pipelines are kept as separate candidates. A family without a pipeline is
        refused: it would silently change the number of families, and with it the level.
        """
        if not isinstance(families, Mapping) or not families:
            raise ValueError("families must be a non-empty mapping {label: [pipelines]}")
        specs: list[CandidateSpec] = []
        for label, pipelines in families.items():
            if not isinstance(label, str) or not label:
                raise ValueError(f"family labels must be non-empty strings, got {label!r}")
            if isinstance(pipelines, (Pipeline, Mapping, str)) or not isinstance(pipelines, Sequence):
                raise TypeError(f"family {label!r} must be a sequence of Pipeline objects")
            if len(pipelines) == 0:
                raise ValueError(f"family {label!r} has no candidates")
            for pipe in pipelines:
                if not isinstance(pipe, Pipeline):
                    raise TypeError(f"family {label!r} holds a {type(pipe).__name__}, "
                                    "not a Pipeline")
                specs.append(CandidateSpec(len(specs), label, pipe.to_dict()))
        return cls(name, tuple(specs))

    @classmethod
    def default(cls, dim: int, *, allow_reduced: bool = False) -> CandidateSpace:
        """The library's default list (``calibration.recommended_search_space``), 7 chains.

        A frozen, development-informed default (see the module docstring), not a neutral or
        encoder-specific choice. With SciPy installed it hashes to
        :data:`REGISTERED_DEFAULT_SPACE_SHA256` at any ``dim`` (the list does not depend on
        it). Without SciPy the Ward chains cannot be built: this raises ``ImportError``
        unless ``allow_reduced=True``, which returns the 4-chain list that is *not* the
        registered space (``is_registered_default`` is False, the name says so).
        """
        _check_dim(dim)
        scipy = _have_scipy()
        if not scipy and not allow_reduced:
            raise ImportError(
                "the default candidate space needs SciPy for its Ward-merge chains; install "
                "scipy, or pass allow_reduced=True for the reduced 4-chain list (a different "
                "space, with a different identity and a different test level)")
        space = recommended_search_space(int(dim), use_scipy=scipy)
        return cls.from_families(space, name="default" if scipy else "default-reduced")

    @classmethod
    def wide(cls, dim: int, *, allow_reduced: bool = False) -> CandidateSpace:
        """The default list plus width-reduction chains (``calibration.wide_search_space``).

        Engineering capability; not evaluated by the executed registered experiments as a
        selection space. The list depends on ``dim``. ``allow_reduced`` behaves as in
        :meth:`default`.
        """
        _check_dim(dim)
        scipy = _have_scipy()
        if not scipy and not allow_reduced:
            raise ImportError("the wide candidate space needs SciPy for its Ward-merge chains; "
                              "install scipy, or pass allow_reduced=True")
        space = wide_search_space(int(dim), use_scipy=scipy)
        return cls.from_families(space, name="wide" if scipy else "wide-reduced")

    # ---------------------------------------------------------------- the list

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Iterator[CandidateSpec]:
        return iter(self.candidates)

    def __getitem__(self, position: int) -> CandidateSpec:
        return self.candidates[position]

    @property
    def candidate_families(self) -> tuple[str, ...]:
        """The family label of every candidate, in list order (what a rule receives)."""
        return tuple(spec.family for spec in self.candidates)

    @property
    def families(self) -> tuple[str, ...]:
        """The distinct family labels, in order of appearance."""
        return tuple(dict.fromkeys(self.candidate_families))

    @property
    def n_families(self) -> int:
        return len(self.families)

    @cached_property
    def identity(self) -> str:
        """SHA-256 of the canonical JSON of the ordered ``{position, family, pipeline}`` list."""
        return digest([spec.to_dict() for spec in self.candidates])

    @property
    def is_registered_default(self) -> bool:
        return self.identity == REGISTERED_DEFAULT_SPACE_SHA256

    def pipelines(self) -> list[Pipeline]:
        """Fresh, unfitted pipelines in list order; mutating them never touches the space."""
        return [spec.to_pipeline() for spec in self.candidates]

    # ---------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {"format": _FORMAT, "version": _VERSION, "name": self.name,
                "identity": self.identity,
                "candidates": [spec.to_dict() for spec in self.candidates]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateSpace:
        if data.get("format") != _FORMAT:
            raise ValueError(f"not a candidate space (format {data.get('format')!r})")
        if data.get("version") != _VERSION:
            raise ValueError(f"unsupported candidate space version {data.get('version')!r}")
        space = cls(data["name"], tuple(CandidateSpec.from_dict(c) for c in data["candidates"]))
        if data.get("identity") != space.identity:
            raise ValueError("the stored identity does not match the candidate list: the "
                             "record was altered or is corrupt")
        return space
