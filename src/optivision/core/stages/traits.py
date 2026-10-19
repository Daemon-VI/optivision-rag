"""Declared structural traits of compression stages, and the pipeline rule built on them.

The question a trait answers is narrow: if every document and query vector is multiplied
by the same orthogonal matrix ``G`` and the stage is fitted again on the result, do the
scores that come out stay the same? ``BasisInvariance`` records for which matrices ``G``
the answer is yes. The answer is a statement about the code, read in exact arithmetic and
checked on synthetic data by ``tests/test_core_stage_traits.py``; it says nothing about
retrieval quality, and in float32 a value that sits within rounding of a quantisation
boundary can still change level.

A declared level holds for data that meet the stage's ``conditions``: data without the
exact ties and degeneracies at which the stage's code decides by row position, by sign
convention or by rounding (equal eigenvalues at a PCA cut, rows that are exactly
equidistant for Ward clustering, a coordinate that is exactly zero for a sign code). The
conditions are plain sentences. Nothing in the package checks them against the data, and
they are carried to ``PipelineTraits``, so ``ORTHOGONAL`` means "for data that meet the
listed conditions" and ``basis_sensitive=False`` with ``unconditional=False`` is a
conditional statement.

Nothing in the package may assume ``codec(x)`` behaves like ``codec(xG)``. Most codecs act
on the stored coordinates (sign bits, fixed or per-vector steps, a leading-coordinate cut),
so a pipeline is called ``basis_sensitive`` unless every stage is declared invariant under
any orthogonal ``G``. A stage without declared traits is ``UNKNOWN`` and is treated as
basis dependent.

Traits are registered by stage name (``register_traits``) or declared by the stage class
through a ``traits`` attribute or method; ``traits_of`` reads both. Only the table at the end
of this module is the library's own (``declared_by="library"``). A declaration made by any
other code is recorded as declared (``declared_by="third_party"``) and is not examined by
the library. The trait registry is the only global state here, as the stage registry is in
``optivision.stages.base``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from optivision.stages.base import STAGES, DimensionReducer, Quantizer, Stage, TokenReducer

if TYPE_CHECKING:
    from optivision.compose import Pipeline

__all__ = [
    "BasisInvariance",
    "PipelineTraits",
    "StageTraits",
    "pipeline_traits",
    "register_traits",
    "traits_of",
]


class BasisInvariance(str, Enum):
    """For which orthogonal changes of basis a stage's scores stay the same (after refit).

    Each level is a statement for data that meet the stage's ``conditions``
    (``StageTraits.conditions``): ``ORTHOGONAL`` means "unchanged under any orthogonal
    matrix, for data that meet the listed conditions", and the other levels read the same
    way. A stage that lists no condition makes its statement for all data, in exact
    arithmetic.
    """

    ORTHOGONAL = "orthogonal"  # any orthogonal matrix
    SIGNED_PERMUTATION = "signed_permutation"  # only axis permutations and sign flips
    NONE = "none"  # not even that (for example: keeps the leading k coordinates)
    UNKNOWN = "unknown"  # undeclared: counted as NONE


_O, _S, _N, _U = (
    BasisInvariance.ORTHOGONAL,
    BasisInvariance.SIGNED_PERMUTATION,
    BasisInvariance.NONE,
    BasisInvariance.UNKNOWN,
)
_RANK = {_O: 2, _S: 1, _N: 0, _U: 0}
_KINDS = ("tokens", "dimensions", "bits", "unknown")
#: values of ``StageTraits.declared_by``
LIBRARY = "library"
THIRD_PARTY = "third_party"
UNDECLARED = "undeclared"
_DECLARED_BY = (LIBRARY, THIRD_PARTY, UNDECLARED)


@dataclass(frozen=True)
class StageTraits:
    """What is declared about one stage, for the parameters it was built with.

    The declared ``basis_invariance`` is a statement for data that meet ``conditions``; with
    no condition listed it is made for all data, in exact arithmetic. Nothing here checks
    the data. A declaration that did not come from the library's own table
    (``declared_by="third_party"``) is recorded as declared and is not examined by the
    library: the library does not test that the stage behaves as the declaration says.

    Attributes:
        name: registry name of the stage.
        kind: ``"tokens"`` (fewer vectors), ``"dimensions"`` (projection) or ``"bits"``
            (quantizer); ``"unknown"`` only for a name that has no declaration.
        basis_invariance: see :class:`BasisInvariance`.
        canonicalises_basis: the output coordinates are fixed by the data, not by the basis
            of the input (PCA). Recorded, never credited as invariance on its own.
        randomised_basis: applies an internal seeded orthogonal map, so the realisation
            depends on the input basis while the distribution over seeds does not.
        reads_vector_values: the stage's decisions or parameters are computed from the
            vector values (clusters, fitted statistics, per-vector scales). ``False`` for
            a stage that depends only on counts, positions, pixels, seeds or dimension, and
            for a fixed element-wise map.
        has_fitted_state: ``fit`` stores arrays that are needed to apply the stage again.
            Depends on the parameters; read it from ``traits_of(instance)``.
        fit_uses_queries: ``fit`` reads the queries it is given. The selection rule assumes
            stages fitted on documents only, so this matters.
        stochastic: the stage's code can draw from a seeded generator for these
            parameters, including a fixed-seed row sample taken only above a size cap.
            Same seed and same input always give the same output.
        lossless: stores its input exactly (float32 input to the float32 codec).
        notes: the reasons behind the declaration, in plain words.
        conditions: sentences that name what must hold for the declared invariance: no exact
            ties in the quantities that decide a merge or a kept row, distinct eigenvalues at
            a PCA cut, no coordinate that is exactly zero for a sign code. Empty when the
            declaration needs none. They describe the data; nothing in the package checks
            them.
        declared_by: who made the declaration. ``"library"``: the library's own table at the
            end of this module, which ``tests/test_core_stage_traits.py`` exercises on
            synthetic data. ``"third_party"`` (the default for a value you build): a
            ``register_traits`` call or a ``traits`` attribute or method of a stage class,
            made outside that table; recorded as declared and not examined by the library.
            ``"undeclared"``: nothing was declared (``UNKNOWN``). ``traits_of`` sets it, and
            replaces what a registered or class-declared value says about itself.
    """

    name: str
    kind: str
    basis_invariance: BasisInvariance
    canonicalises_basis: bool = False
    randomised_basis: bool = False
    reads_vector_values: bool = True
    has_fitted_state: bool = False
    fit_uses_queries: bool = False
    stochastic: bool = False
    lossless: bool = False
    notes: str = ""
    conditions: tuple[str, ...] = ()
    declared_by: str = THIRD_PARTY

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a stage's traits need its registry name")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "basis_invariance", BasisInvariance(self.basis_invariance))
        if isinstance(self.conditions, (str, bytes)) or not all(
            isinstance(c, str) and c for c in self.conditions
        ):
            raise TypeError("conditions must be a sequence of non-empty sentences")
        object.__setattr__(self, "conditions", tuple(self.conditions))
        if self.declared_by not in _DECLARED_BY:
            raise ValueError(f"declared_by must be one of {_DECLARED_BY}, got {self.declared_by!r}")

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["basis_invariance"] = self.basis_invariance.value
        out["conditions"] = list(self.conditions)
        return out


# ------------------------------------------------------------------ registry


@dataclass(frozen=True)
class _Entry:
    generic: StageTraits  # what ``traits_of(name)`` returns: the weakest case over the parameters
    for_stage: Callable[[Stage], StageTraits] | None  # None: ``generic`` applies to every instance
    declared_by: str  # applied to whatever the entry hands out


_REGISTRY: dict[str, _Entry] = {}


def _declared_by(value: StageTraits, who: str) -> StageTraits:
    """``value`` as declared by ``who``: what a value says about its own origin is ignored."""
    return value if value.declared_by == who else dataclasses.replace(value, declared_by=who)


def _register(
    name: str,
    traits: StageTraits | Callable[[Stage], StageTraits],
    generic: StageTraits | None,
    who: str,
) -> None:
    if isinstance(traits, StageTraits):
        if traits.name != name:
            raise ValueError(f"traits registered under {name!r} declare the name {traits.name!r}")
        entry = _Entry(_declared_by(traits, who), None, who)
    elif callable(traits):
        bare = (
            _declared_by(generic, who)
            if generic is not None
            else _unknown(name, "unknown", _PER_INSTANCE)
        )
        entry = _Entry(bare, traits, who)
    else:
        raise TypeError("traits must be a StageTraits or a function of the stage instance")
    old = _REGISTRY.get(name)
    if old is not None and old != entry:
        raise ValueError(f"traits for stage {name!r} are already registered")
    _REGISTRY[name] = entry


def register_traits(
    name: str,
    traits: StageTraits | Callable[[Stage], StageTraits],
    *,
    generic: StageTraits | None = None,
) -> None:
    """Declare the traits of the stage registered under ``name``.

    ``traits`` is one :class:`StageTraits`, or a function of the stage instance when the
    traits depend on its parameters. ``generic`` is what ``traits_of(name)`` should return
    for the bare name (the weakest case over the parameters); without it a function leaves
    the bare name ``UNKNOWN``. Registering the same value again is a no-op; a different one
    for the same name raises, as for stage names.

    A declaration made through this function is third-party: ``traits_of`` and
    ``pipeline_traits`` record it as ``declared_by="third_party"`` whatever the value says,
    as declared and not examined by the library. List in ``conditions`` the data for which
    the declared ``basis_invariance`` holds.
    """
    _register(name, traits, generic, THIRD_PARTY)


_PER_INSTANCE = "traits depend on the parameters: ask for them with a stage instance"
_NOT_DECLARED = "no basis traits are declared for this stage; nothing is assumed"


def _unknown(name: str, kind: str, why: str) -> StageTraits:
    return StageTraits(
        name=name,
        kind=kind,
        basis_invariance=_U,
        fit_uses_queries=True,
        notes=f"{why}. {_NOT_DECLARED}.",
        declared_by=UNDECLARED,
    )


def _kind_of(stage: Stage) -> str:
    if isinstance(stage, TokenReducer):
        return "tokens"
    if isinstance(stage, DimensionReducer):
        return "dimensions"
    if isinstance(stage, Quantizer):
        return "bits"
    return "unknown"


def _declared(stage: Stage) -> StageTraits | None:
    """Traits a stage class declares itself: a ``traits`` attribute, property or method."""
    attr = getattr(stage, "traits", None)
    if attr is None:
        return None
    value = attr() if callable(attr) else attr
    if not isinstance(value, StageTraits):
        raise TypeError(
            f"{type(stage).__name__}.traits must give a StageTraits, got {type(value).__name__}"
        )
    return value


def traits_of(stage: Stage | str) -> StageTraits:
    """Traits of a stage instance (they can depend on its parameters) or of a registered name.

    For a name the result is the weakest case over the parameters. A stage that declares
    nothing, or a subclass of a registered class that was not registered itself, gets
    ``BasisInvariance.UNKNOWN`` with ``declared_by="undeclared"``; this is never an error.
    A ``traits`` attribute or method of the stage class, and a ``register_traits`` call,
    give ``declared_by="third_party"``: recorded as declared, not examined by the library.
    Only the library's own table gives ``"library"``.
    """
    if isinstance(stage, str):
        entry = _REGISTRY.get(stage)
        return (
            entry.generic
            if entry is not None
            else _unknown(stage, "unknown", "no traits registered")
        )
    if not isinstance(stage, Stage):
        raise TypeError(f"traits_of expects a Stage or a stage name, got {type(stage).__name__}")
    declared = _declared(stage)
    if declared is not None:
        return _declared_by(declared, THIRD_PARTY)
    entry = _REGISTRY.get(stage.name)
    registered = STAGES.get(stage.name)
    if entry is not None and registered in (None, type(stage)):
        if entry.for_stage is None:
            return entry.generic
        value = entry.for_stage(stage)
        if not isinstance(value, StageTraits):
            raise TypeError(
                f"the traits registered for {stage.name!r} must give a StageTraits, "
                f"got {type(value).__name__}"
            )
        return _declared_by(value, entry.declared_by)
    why = "traits are not inherited by a subclass" if entry is not None else "no traits registered"
    unknown = _unknown(stage.name, _kind_of(stage), why)
    return dataclasses.replace(
        unknown, has_fitted_state=callable(getattr(stage, "get_state", None))
    )


# ----------------------------------------------------------- pipeline rule


@dataclass(frozen=True)
class PipelineTraits:
    """Basis behaviour of a whole pipeline.

    ``basis_invariance`` is the weakest level the stages declare; canonicalisation (PCA) is
    recorded but not credited in it. Every level is a statement for data that meet
    ``conditions``: the conditions of every stage, in stage order, each prefixed with the
    stage name. ``unconditional`` is True only when there is none. So ``basis_sensitive=False``
    with ``unconditional=False`` is a conditional statement ("not sensitive, for data that
    meet the listed conditions"), not a property of every data set. ``basis_sensitive=True``
    is the cautious statement and needs no condition. Nothing checks the conditions against
    the data.

    ``conditionally_invariant`` is True when the pipeline is not ORTHOGONAL but every stage
    that is not ORTHOGONAL is a SIGNED_PERMUTATION stage that receives coordinates fixed by
    the data (up to the sign of each component) from a canonicalising stage (PCA) before it,
    with no seeded random map in between. That holds only if the retained principal
    directions are well separated, which is among the ``conditions``. ``basis_sensitive`` is
    True unless the pipeline is ORTHOGONAL.

    ``all_library_declared`` is True when every stage's traits come from the library's own
    table (``StageTraits.declared_by``). A third-party declaration is recorded as declared
    and is not examined by the library; the flag is False for a pipeline that holds one, and
    for a stage that declares nothing.
    """

    stages: tuple[StageTraits, ...]
    basis_invariance: BasisInvariance
    conditionally_invariant: bool
    basis_sensitive: bool
    has_fitted_state: bool
    stochastic: bool
    notes: tuple[str, ...]
    conditions: tuple[str, ...] = ()

    @property
    def unconditional(self) -> bool:
        """True only when no stage lists a condition."""
        return not self.conditions

    @property
    def all_library_declared(self) -> bool:
        """True when every stage's traits are the library's own declarations."""
        return all(t.declared_by == LIBRARY for t in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [t.to_dict() for t in self.stages],
            "basis_invariance": self.basis_invariance.value,
            "conditionally_invariant": self.conditionally_invariant,
            "basis_sensitive": self.basis_sensitive,
            "has_fitted_state": self.has_fitted_state,
            "stochastic": self.stochastic,
            "notes": list(self.notes),
            "conditions": list(self.conditions),
            "unconditional": self.unconditional,
            "all_library_declared": self.all_library_declared,
        }


def _stages_of(pipeline: Pipeline | Iterable[Stage]) -> list[Stage]:
    stages = getattr(pipeline, "stages", pipeline)
    if isinstance(stages, (str, bytes)):
        raise TypeError("pipeline_traits expects a Pipeline or a sequence of stages")
    return list(stages)


def _conditionally_invariant(stages: tuple[StageTraits, ...]) -> bool:
    """Every weaker-than-ORTHOGONAL stage must be a SIGNED_PERMUTATION stage reached through
    canonical coordinates. A stage that is not invariant under sign flips (a NONE stage, or
    anything after a seeded random map) would pass the arbitrary signs of PCA components on
    into its output, so it cannot be credited."""
    canonical = False
    weak = False
    for t in stages:
        if t.basis_invariance is _U:
            return False
        if t.basis_invariance is not _O:
            weak = True
            if not canonical or t.basis_invariance is not _S:
                return False
        if t.canonicalises_basis:
            canonical = True
        elif t.randomised_basis:
            canonical = False
    return weak


def pipeline_traits(pipeline: Pipeline | Iterable[Stage]) -> PipelineTraits:
    """Apply the pipeline rule to a pipeline (or a list of stages).

    Walk the stages in order in the state ``input`` or ``mixed``, starting in ``input``. A
    stage's declared invariance counts as declared, except that in state ``mixed`` a
    SIGNED_PERMUTATION stage counts as NONE: a fixed seeded orthogonal map was applied
    before it, so even a permutation of the input axes changes what it sees. A stage with
    ``randomised_basis`` switches the state to ``mixed``. The pipeline's invariance is the
    weakest value met (ORTHOGONAL > SIGNED_PERMUTATION > NONE); an undeclared stage counts
    as NONE and, when no declared stage is NONE, the pipeline is reported as UNKNOWN.

    The result is a statement for data that meet its ``conditions`` (those of every stage,
    prefixed with the stage name): ``ORTHOGONAL`` means "for data that meet the listed
    conditions", and ``basis_sensitive=False`` with ``unconditional=False`` is a
    conditional statement. Nothing checks the conditions against the data. A stage's
    traits are those of ``traits_of``: declarations made outside the library's own table
    are recorded as third-party (``all_library_declared`` is then False, and a note says
    so) and are not examined by the library.
    """
    stages = tuple(traits_of(s) for s in _stages_of(pipeline))
    mixed = False
    effective: list[BasisInvariance] = []
    notes: list[str] = []
    for t in stages:
        inv = t.basis_invariance
        if t.declared_by == THIRD_PARTY and inv is not _U:
            notes.append(
                f"{t.name}: traits declared by third-party code; recorded as declared, "
                "not examined by the library"
            )
        if inv is _U:
            eff = _N
            notes.append(
                f"{t.name}: no declared traits; counted as dependent on the coordinate basis"
            )
        elif mixed and inv is _S:
            eff = _N
            notes.append(f"{t.name}: counted as NONE because a seeded random map comes before it")
        else:
            eff = inv
            if inv is _S:
                notes.append(f"{t.name}: invariant under signed permutations of the axes only")
            elif inv is _N:
                notes.append(f"{t.name}: its result changes with the axis order")
        effective.append(eff)
        mixed = mixed or t.randomised_basis
    weakest = min(effective, key=_RANK.__getitem__, default=_O)
    known_none = any(e is _N and t.basis_invariance is not _U for t, e in zip(stages, effective))
    overall = _U if weakest is _N and not known_none else weakest
    conditional = _conditionally_invariant(stages)
    if conditional:
        notes.append(
            "conditionally invariant: the codec after PCA sees coordinates fixed by the data "
            "up to the sign of each component, provided the retained eigenvalues are distinct "
            "and the last one retained exceeds the next; a condition on the data, not checked"
        )
    return PipelineTraits(
        stages=stages,
        basis_invariance=overall,
        conditionally_invariant=conditional,
        basis_sensitive=overall is not _O,
        has_fitted_state=any(t.has_fitted_state for t in stages),
        stochastic=any(t.stochastic for t in stages),
        notes=tuple(notes),
        conditions=tuple(f"{t.name}: {c}" for t in stages for c in t.conditions),
    )


# ------------------------------------------------------- built-in declarations


def _builtin(
    name: str,
    kind: str,
    invariance: BasisInvariance,
    notes: str,
    refine: Callable[[Stage, StageTraits], StageTraits] | None = None,
    **flags: Any,
) -> None:
    generic = StageTraits(
        name=name,
        kind=kind,
        basis_invariance=invariance,
        notes=notes,
        declared_by=LIBRARY,
        **flags,
    )
    if refine is None:
        _register(name, generic, None, LIBRARY)
    else:
        _register(name, lambda stage, g=generic: refine(stage, g), generic, LIBRARY)


def _centred(stage: Stage) -> bool:
    return getattr(stage, "center", None) == "mean"


# The conditions of the declarations below. Each is a sentence about the data; a witness test
# in tests/test_core_stage_traits.py builds data that break it and shows the scores move.
_TIES_OVERTURNED = (
    "ties go to row order, which float32 rounding after a change of basis can overturn"
)
_REDUNDANCY_CONDITIONS = (
    (
        "no cosine between two rows of one document lies within float32 rounding of the "
        "threshold (an exact tie with the threshold is the extreme case: rounding after a change "
        "of basis can move it to either side)"
    ),
)
_ADAPTIVE_MERGE_CONDITIONS = (
    (
        "no exact ties among the cosines that choose a seed, the next centre or the nearest "
        f"centre of a row ({_TIES_OVERTURNED}), and no cosine within float32 rounding of the "
        "radius"
    ),
)
_HIERARCHICAL_MERGE_CONDITIONS = (
    (
        "no exact ties among the Ward merge distances of a document (rows that are exactly "
        f"equidistant, for example orthonormal rows; {_TIES_OVERTURNED}), and no merge height "
        "within float32 rounding of max_distance"
    ),
)
_PCA_CONDITIONS = (
    (
        "eigenvalue gap at the cut: the last retained eigenvalue is larger than the next one, by "
        "more than float32 rounding in the second-moment matrix (with a tie the retained subspace "
        "is arbitrary)"
    ),
    (
        "the retained eigenvalues are distinct from one another; this matters to a coordinate "
        "stage after this one (with a tie its input coordinates are an arbitrary rotation within "
        "the tied group)"
    ),
)
_SIGN_CODE_CONDITIONS = (
    (
        "no coordinate of a stored vector is exactly zero (exactly equal to the fitted mean when "
        "center='mean'): a zero is coded as -1 whichever sign convention is used"
    ),
)


_builtin(
    "spatial",
    "tokens",
    _O,
    "Chooses rows from page pixels and grid positions and never reads the vector values, so "
    "which rows survive does not depend on the coordinate basis. Needs page images; the class "
    "is page-specific and lives outside the model-agnostic layer.",
    reads_vector_values=False,
)

_builtin(
    "redundancy",
    "tokens",
    _O,
    "Compares the vectors of one document only through inner products and pools a cluster by "
    "its mean, so one orthogonal matrix applied to every vector keeps the same rows and "
    "rotates the pooled ones. Exact only in exact arithmetic: a pair whose cosine lies within "
    "float32 rounding of the threshold can fall on the other side.",
    conditions=_REDUNDANCY_CONDITIONS,
)


def _adaptive_merge(stage: Stage, g: StageTraits) -> StageTraits:
    centred = _centred(stage)
    return dataclasses.replace(g, has_fitted_state=centred, stochastic=centred)


_builtin(
    "adaptive_merge",
    "tokens",
    _O,
    "Measures similarity by cosines (of the vectors, or of the vectors minus a fitted mean "
    "when center='mean') and pools members by their mean; all of these follow an orthogonal "
    "change of basis, so the same rows are merged. Exact ties go to the lowest row index, "
    "which holds in exact arithmetic and not after float32 rounding (see the conditions). "
    "With center='mean' the fitted mean (of a fixed-seed row sample above the fit-sample cap) "
    "rotates with the data.",
    refine=_adaptive_merge,
    has_fitted_state=True,
    stochastic=True,
    conditions=_ADAPTIVE_MERGE_CONDITIONS,
)

_builtin(
    "hierarchical_merge",
    "tokens",
    _O,
    "Ward clustering of unit vectors depends only on pairwise Euclidean distances (taken from "
    "the vectors, or from their Gram matrix from 512 dimensions up), which an orthogonal "
    "change of basis leaves unchanged; clusters are pooled by their mean. Needs SciPy. Exact "
    "ties between merge heights are resolved by row order, which holds in exact arithmetic "
    "and not after float32 rounding (see the conditions).",
    conditions=_HIERARCHICAL_MERGE_CONDITIONS,
)

_builtin(
    "random_prune",
    "tokens",
    _O,
    "Keeps a seeded random subset of rows. The draw depends on the seed and the position of "
    "the document only, not on the vector values, so a change of basis keeps the same rows. "
    "Reordering or subsetting the documents changes the draw.",
    reads_vector_values=False,
    stochastic=True,
)


def _pca(stage: Stage, g: StageTraits) -> StageTraits:
    joint = getattr(stage, "basis", "documents") == "joint"
    return dataclasses.replace(g, fit_uses_queries=joint)


_builtin(
    "pca",
    "dimensions",
    _O,
    "Canonical up to the sign of each component, and only when the retained eigenvalues are "
    "distinct and the last one retained is strictly larger than the next: the retained "
    "directions are then fixed by the data rather than by the stored basis, so scores do not "
    "change when one orthogonal matrix is applied to documents and queries. With equal "
    "eigenvalues at the cut the retained subspace is arbitrary; with equal retained "
    "eigenvalues the output coordinates are an arbitrary rotation within the tied group. The "
    "signs of the output coordinates come from the eigensolver, not from the data. "
    "basis='joint' also reads the queries at fit time. Rows are sampled with `seed` above "
    "`sample` rows; center=True subtracts the document mean on the document side only.",
    refine=_pca,
    canonicalises_basis=True,
    has_fitted_state=True,
    stochastic=True,
    fit_uses_queries=True,
    conditions=_PCA_CONDITIONS,
)

_builtin(
    "random_projection",
    "dimensions",
    _N,
    "Draws a seeded Gaussian matrix in the stored coordinates and orthonormalises it, so the "
    "realisation depends on the stored basis: even a permutation of the axes changes it. The "
    "distribution of scores over seeds does not depend on the basis. When dim is at least "
    "the input dimension the matrix is square and orthogonal and the stage alone keeps all "
    "scores, but a codec after it then sees a random rotation of the stored coordinates.",
    randomised_basis=True,
    reads_vector_values=False,
    has_fitted_state=True,
    stochastic=True,
)

_builtin(
    "truncate",
    "dimensions",
    _N,
    "Keeps coordinates 0..k-1 of the stored basis, so what is kept depends on the axis order: "
    "a permutation of the axes changes the result. Meaningful only when the leading "
    "coordinates carry the information (nested, Matryoshka-trained encoders); for other "
    "encoders it is a control.",
    reads_vector_values=False,
)

_builtin(
    "rotation",
    "dimensions",
    _O,
    "Multiplies documents and queries by one seeded orthogonal matrix, so all inner products, "
    "and with them float scores, are unchanged up to float32 rounding. The output coordinates "
    "are a rotation of the stored ones and depend on the seed, so a codec after this stage "
    "sees a different basis than without it (the pipeline rule counts a later "
    "SIGNED_PERMUTATION stage as NONE). The matrix depends on the seed and the dimension only. "
    "Engineering capability and diagnostic; not evaluated by the registered experiments.",
    randomised_basis=True,
    reads_vector_values=False,
    has_fitted_state=True,
    stochastic=True,
)


def _project(stage: Stage, g: StageTraits) -> StageTraits:
    inner = getattr(stage, "inner", None)
    if inner is None:
        return g
    t = traits_of(inner)
    return dataclasses.replace(
        t, name=g.name, notes=f"method={getattr(stage, 'method', '?')!r}: {t.notes}"
    )


_builtin(
    "project",
    "dimensions",
    _N,
    "Facade over pca, random_projection and truncate: the traits are those of the chosen "
    "method (ask with an instance). The bare-name values are the weakest case over the three.",
    refine=_project,
    randomised_basis=True,
    has_fitted_state=True,
    stochastic=True,
    fit_uses_queries=True,
)

_builtin(
    "float32",
    "bits",
    _O,
    "Stores float32 values unchanged (float64 input is cast to float32 before storage), so it "
    "adds no coordinate-dependent error.",
    reads_vector_values=False,
    lossless=True,
)

_builtin(
    "float16",
    "bits",
    _S,
    "Rounds each coordinate to half precision. The rounding acts on each coordinate alone and "
    "is symmetric in sign, so signed permutations of the axes give the same scores. A general "
    "rotation changes which rounding errors occur (the relative bound per coordinate is the "
    "same in every basis), so scores move by about the size of the half-precision rounding "
    "error and the stage is not declared ORTHOGONAL.",
    reads_vector_values=False,
)

_CENTRE = (
    " center='mean' subtracts a fitted per-axis mean before quantising (a fixed-seed row sample "
    "above the fit-sample cap), which follows the axes."
)
_INT8 = {
    "fixed": (
        "Fixed absolute step (0.5/127 per component) with clipping at +-0.5, applied to each "
        "coordinate alone: signed permutations of the axes give the same codes up to the "
        "permutation, but a rotation changes which components clip and where they fall on the "
        "grid."
    ),
    "per_vector": (
        "The step is the vector's largest absolute component over 127 (scale stored as "
        "float16). That maximum is unchanged by permuting or flipping axes and changes under a "
        "rotation."
    ),
    "per_dimension": (
        "Per-axis range fitted as the 99.99th percentile of |component|: the ranges follow the "
        "axes under signed permutations; under a rotation the ranges, the clipping and the "
        "rounding all change."
    ),
}


def _int8(stage: Stage, g: StageTraits) -> StageTraits:
    scale = getattr(stage, "scale", "fixed")
    centred = _centred(stage)
    fitted = scale == "per_dimension" or centred
    return dataclasses.replace(
        g,
        has_fitted_state=fitted,
        stochastic=fitted,
        reads_vector_values=scale != "fixed" or centred,
        notes=_INT8.get(scale, "") + (_CENTRE if centred else ""),
    )


_builtin(
    "int8",
    "bits",
    _S,
    " ".join(_INT8.values()) + _CENTRE,
    refine=_int8,
    has_fitted_state=True,
    stochastic=True,
)


def _int4(stage: Stage, g: StageTraits) -> StageTraits:
    centred = _centred(stage)
    return dataclasses.replace(
        g,
        has_fitted_state=centred,
        stochastic=centred,
        notes=g.notes.removesuffix(_CENTRE) + (_CENTRE if centred else ""),
    )


_builtin(
    "int4",
    "bits",
    _S,
    "Per-vector scale (largest absolute component, float16) and levels -7..7: unchanged by "
    "permuting or flipping axes, changed by a rotation." + _CENTRE,
    refine=_int4,
    has_fitted_state=True,
    stochastic=True,
)


_builtin(
    "int4norm",
    "bits",
    _S,
    "int4's levels -7..7 of the largest absolute component, with the float16 scale chosen so that "
    "each decoded vector keeps the norm of the vector it codes (with the mean added back when "
    "centred): unchanged by permuting or flipping axes, changed by a rotation." + _CENTRE,
    refine=_int4,
    has_fitted_state=True,
    stochastic=True,
)


def _binary(stage: Stage, g: StageTraits) -> StageTraits:
    centred = _centred(stage)
    return dataclasses.replace(
        g,
        has_fitted_state=centred,
        stochastic=centred,
        reads_vector_values=centred,
        notes=g.notes.removesuffix(_CENTRE) + (_CENTRE if centred else ""),
    )


_builtin(
    "binary",
    "bits",
    _S,
    "Keeps one sign bit per coordinate (v > 0) and decodes to +-1, so it stores the orthant of "
    "the vector in the stored basis: signed permutations of the axes map codes to codes, a "
    "rotation does not. A coordinate that is exactly 0 is coded as -1 whichever sign "
    "convention is used, so the signed-permutation statement excludes exact zeros." + _CENTRE,
    refine=_binary,
    has_fitted_state=True,
    stochastic=True,
    conditions=_SIGN_CODE_CONDITIONS,
)

_builtin(
    "lloyd2",
    "bits",
    _N,
    "Applies an internal seeded orthogonal rotation and then 2-bit levels on each coordinate, "
    "so the realisation depends on the input basis (even a signed permutation changes the "
    "codes) while the distribution of results over seeds does not. The rotation is "
    "regenerated from (dim, seed) with a LAPACK QR and is not stored; the fitted state "
    "carries a digest of it.",
    randomised_basis=True,
    has_fitted_state=True,
    stochastic=True,
)
