"""Fitted stage state as plain arrays and scalars, with digests and a checked restore.

``Pipeline.to_dict`` stores only constructor arguments, so a rebuilt pipeline is unfitted.
This module reads what ``fit`` left on a stage into a dict of NumPy arrays and
JSON-native scalars, and puts such a dict back into a stage built from
``stage_from_dict(stage.to_dict())``.

The built-in stage classes are byte-pinned, so their fitted attributes are read through a
table per class (attribute names, dtypes, shapes), including how the parameters decide
which attributes exist: ``Int8Quantizer(scale="per_dimension")`` has ``ranges``, a centred
codec has ``mean``, an uncentred PCA has no ``mean`` to store. A stage outside the table
can implement ``get_state()`` and ``set_state(state)`` instead; its ``fitted`` property must
be False until ``set_state`` has run.

``restore_fitted_state`` checks keys, dtypes, shapes and finiteness, and a few structural
properties the fitting code produces (orthonormal PCA components, an orthogonal rotation).
Those checks catch corruption and mix-ups; they are not an integrity proof. Integrity is
the digest: ``fitted_state_digest`` of the restored stage must equal the digest recorded
when the state was captured, and ``expected_digest`` makes the restore check exactly that.

The ``lloyd2`` rotation is not stored. It is regenerated from ``(dim, seed)`` by a LAPACK QR
(``compression.lloyd2.rotation_matrix``), so the state carries a digest of the matrix that
was used, and a restore on a platform that regenerates a different matrix raises.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from optivision.core.identity import array_digest, digest
from optivision.stages.base import Stage
from optivision.stages.merge import AdaptiveMerge
from optivision.stages.project import DimensionProjector, PCAProjector, RandomProjector
from optivision.stages.quantize import (
    BinaryQuantizer,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Codec,
    Lloyd2Quantizer,
)

from .rotation import RotationProjector

if TYPE_CHECKING:
    from optivision.compose import Pipeline

__all__ = [
    "fitted_state",
    "fitted_state_digest",
    "fitted_state_nbytes",
    "pipeline_fitted_identity",
    "pipeline_fitted_state",
    "restore_fitted_state",
    "restore_pipeline_state",
]

#: Largest entry of ``C^T C - I`` accepted for stored orthonormal float32 matrices. Float32
#: rounding of a fitted matrix stays orders of magnitude below it; a scaled, truncated or
#: foreign matrix does not.
_ORTHO_TOL = 1e-4


@dataclass(frozen=True)
class _Field:
    kind: str  # "array" | "float" | "int" | "str"
    dtype: str | None = None
    ndim: int = 0


_F32_1, _F32_2 = _Field("array", "float32", 1), _Field("array", "float32", 2)
_F64_1 = _Field("array", "float64", 1)
_FLOAT, _INT, _STR = _Field("float"), _Field("int"), _Field("str")


def _no_check(stage: Any, values: dict[str, Any]) -> None:
    return None


@dataclass(frozen=True)
class _Handler:
    fields: dict[str, _Field]
    read: Callable[[Any], dict[str, Any]]  # values by key; None for what is not fitted yet
    write: Callable[[Any, dict[str, Any]], None]
    check: Callable[[Any, dict[str, Any]], None] = _no_check


def _attributes(
    fields: dict[str, _Field],
    check: Callable[[Any, dict[str, Any]], None] = _no_check,
    extra: Callable[[Any, dict[str, Any]], None] | None = None,
) -> _Handler:
    def read(stage: Any) -> dict[str, Any]:
        return {key: getattr(stage, key) for key in fields}

    def write(stage: Any, values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(stage, key, value)
        if extra is not None:
            extra(stage, values)

    return _Handler(fields, read, write, check)


def _gram_error(m: np.ndarray, scale: float = 1.0) -> float:
    """Largest entry of ``m^T m - scale * I``, relative to ``scale``. Float32 products: their
    rounding is about 1e-6 for the widths in use, well under the tolerances applied."""
    gram = (m.T @ m).astype(np.float64)
    return float(np.abs(gram - scale * np.eye(gram.shape[0])).max()) / scale


# ------------------------------------------------------------ per-class tables


def _pca(stage: PCAProjector) -> _Handler:
    fields = {"components": _F32_2, "explained": _FLOAT}
    if stage.center:
        fields["mean"] = _F64_1

    def check(s: PCAProjector, v: dict[str, Any]) -> None:
        d_in, k = v["components"].shape
        if k != min(s.dim, d_in):
            raise ValueError(
                f"pca: components keep {k} directions but dim={s.dim} of {d_in} dimensions"
            )
        if "mean" in v and v["mean"].shape != (d_in,):
            raise ValueError(f"pca: mean has shape {v['mean'].shape}, components need ({d_in},)")
        if not -1e-6 <= v["explained"] <= 1.0 + 1e-6:
            raise ValueError(f"pca: explained variance {v['explained']} is outside [0, 1]")
        if _gram_error(v["components"]) > _ORTHO_TOL:
            raise ValueError("pca: components are not orthonormal")

    def unset_mean(s: PCAProjector, v: dict[str, Any]) -> None:
        if not s.center:  # what fit leaves behind for an uncentred PCA; project() never reads it
            s.mean = np.zeros(v["components"].shape[0])

    return _attributes(fields, check, unset_mean)


def _random_projection(stage: RandomProjector) -> _Handler:
    def check(s: RandomProjector, v: dict[str, Any]) -> None:
        d, k = v["matrix"].shape
        if k != min(s.dim, d):
            raise ValueError(
                f"random_projection: matrix keeps {k} columns but dim={s.dim} of {d} dimensions"
            )
        if _gram_error(v["matrix"], scale=d / k) > 1e-3:
            raise ValueError("random_projection: columns are not orthogonal with squared norm d/k")

    return _attributes({"matrix": _F32_2}, check)


def _rotation(stage: RotationProjector) -> _Handler:
    def check(s: RotationProjector, v: dict[str, Any]) -> None:
        d0, d1 = v["matrix"].shape
        if d0 != d1:
            raise ValueError(f"rotation: matrix must be square, got shape {v['matrix'].shape}")
        if _gram_error(v["matrix"]) > _ORTHO_TOL:
            raise ValueError("rotation: matrix is not orthogonal")

    return _attributes({"matrix": _F32_2}, check)


def _mean_only(stage: Any) -> _Handler:
    return _attributes({"mean": _F32_1} if stage.center else {})


def _int8(stage: Int8Quantizer) -> _Handler:
    fields: dict[str, _Field] = {}
    if stage.scale == "per_dimension":
        fields["ranges"] = _F32_1
    if stage.center:
        fields["mean"] = _F32_1

    def check(s: Int8Quantizer, v: dict[str, Any]) -> None:
        if "ranges" in v and not (v["ranges"] > 0).all():
            raise ValueError("int8: per-dimension ranges must be positive")
        if "ranges" in v and "mean" in v and v["ranges"].shape != v["mean"].shape:
            raise ValueError(
                f"int8: ranges {v['ranges'].shape} and mean {v['mean'].shape} disagree"
            )

    return _attributes(fields, check)


_LLOYD2_FIELDS = {"mu": _F32_1, "sigma": _FLOAT, "seed": _INT, "rotation_digest": _STR}


def _lloyd2_codec(values: dict[str, Any]) -> Lloyd2Codec:
    return Lloyd2Codec(
        mu=values["mu"], sigma=values["sigma"], seed=values["seed"], dim=int(values["mu"].shape[0])
    )


def _lloyd2(stage: Lloyd2Quantizer) -> _Handler:
    def read(s: Lloyd2Quantizer) -> dict[str, Any]:
        codec = s.codec
        if codec is None:
            return dict.fromkeys(_LLOYD2_FIELDS)
        return {
            "mu": codec.mu,
            "sigma": float(codec.sigma),
            "seed": int(codec.seed),
            "rotation_digest": array_digest(codec.rotation),
        }

    def check(s: Lloyd2Quantizer, v: dict[str, Any]) -> None:
        if v["seed"] != s.seed:
            raise ValueError(
                f"lloyd2: state was fitted with seed {v['seed']}, the stage has seed {s.seed}"
            )
        if v["sigma"] < 0:
            raise ValueError("lloyd2: sigma must not be negative")
        here = array_digest(_lloyd2_codec(v).rotation)
        if here != v["rotation_digest"]:
            raise ValueError(
                "lloyd2: the rotation regenerated on this machine differs from the one the "
                f"state was fitted with (digest {here[:16]}... here, "
                f"{v['rotation_digest'][:16]}... recorded). The rotation is regenerated from "
                "(dim, seed) with the local LAPACK QR, which does not give the same matrix on "
                "this platform, so codes written elsewhere cannot be decoded here. Restore on "
                "the platform that wrote the state, or re-encode the corpus here."
            )

    def write(s: Lloyd2Quantizer, v: dict[str, Any]) -> None:
        s.codec = _lloyd2_codec(v)

    return _Handler(dict(_LLOYD2_FIELDS), read, write, check)


_BUILDERS: dict[type, Callable[[Any], _Handler]] = {
    AdaptiveMerge: _mean_only,
    PCAProjector: _pca,
    RandomProjector: _random_projection,
    RotationProjector: _rotation,
    BinaryQuantizer: _mean_only,
    Int8Quantizer: _int8,
    Int4Quantizer: _mean_only,
    Lloyd2Quantizer: _lloyd2,
}
_FACADES = (DimensionProjector,)


def _unwrap(stage: Stage) -> Stage:
    """``DimensionProjector`` holds its state in the projector it delegates to."""
    return stage.inner if type(stage) is DimensionProjector else stage


def _handler(stage: Stage) -> _Handler | None:
    builder = _BUILDERS.get(type(stage))
    if builder is not None:
        return builder(stage)
    if isinstance(stage, tuple(_BUILDERS) + _FACADES):
        raise TypeError(
            f"{type(stage).__name__} subclasses a built-in stage that holds fitted state; give it "
            "get_state() and set_state(state) so that its own state is stored as well"
        )
    return None


# ------------------------------------------------------- generic state values


def _plain_value(owner: str, key: str, value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{owner}.{key}: object arrays cannot be stored")
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"{owner}.{key}: {type(value).__name__} is neither an array nor a plain scalar")


def _custom_state(owner: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{owner}: state must be a mapping, got {type(raw).__name__}")
    out = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise TypeError(f"{owner}: state keys must be strings, got {key!r}")
        out[key] = _plain_value(owner, key, value)
    return out


def _coerce(owner: str, key: str, field: _Field, value: Any) -> Any:
    if field.kind == "array":
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{owner}.{key}: expected a numpy array, got {type(value).__name__}")
        if value.dtype != np.dtype(field.dtype):
            raise ValueError(f"{owner}.{key}: expected dtype {field.dtype}, got {value.dtype}")
        if value.ndim != field.ndim:
            raise ValueError(
                f"{owner}.{key}: expected a {field.ndim}-d array, got shape {value.shape}"
            )
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"{owner}.{key}: array is empty or holds non-finite values")
        return np.array(value, order="C", copy=True)
    if field.kind == "str":
        if not isinstance(value, str):
            raise TypeError(f"{owner}.{key}: expected a string, got {type(value).__name__}")
        return value
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{owner}.{key}: expected a number, got {type(value).__name__}")
    if field.kind == "int":
        if isinstance(value, (float, np.floating)):
            raise TypeError(f"{owner}.{key}: expected an integer, got {value!r}")
        return int(value)
    if not np.isfinite(value):
        raise ValueError(f"{owner}.{key}: expected a finite number, got {value!r}")
    return float(value)


def _entries_digest(state: Mapping[str, Any]) -> str:
    return digest(
        {k: array_digest(v) if isinstance(v, np.ndarray) else v for k, v in state.items()}
    )


# ------------------------------------------------------------------ public API


def fitted_state(stage: Stage) -> dict[str, Any]:
    """What ``fit`` left on ``stage``: arrays and JSON-native scalars; ``{}`` if it holds none.

    The arrays are the stage's own; do not modify them. Whether a stage holds state can
    depend on its parameters (a centred codec has a mean, an uncentred one does not). A
    stage that has state but is not fitted yet raises ``RuntimeError``, so that "no state"
    and "not fitted" are never confused. A stage outside the built-in table without a
    ``get_state()`` method is taken to hold no state if it is fitted.
    """
    getter = getattr(stage, "get_state", None)
    if callable(getter):
        if not stage.fitted:
            raise RuntimeError(f"{stage.name} is not fitted")
        return _custom_state(stage.name, getter())
    target = _unwrap(stage)
    handler = _handler(target)
    if handler is None:
        if stage.fitted:
            return {}
        raise RuntimeError(
            f"{stage.name} is not fitted and has no get_state(), so its state cannot be read"
        )
    values = handler.read(target)
    if handler.fields and any(v is None for v in values.values()):
        raise RuntimeError(f"{stage.name} is not fitted")
    return values


def restore_fitted_state(
    stage: Stage, state: Mapping[str, Any], *, expected_digest: str | None = None
) -> Stage:
    """Put ``state`` into ``stage`` and return it, fitted.

    ``stage`` must be an unfitted instance of the class that produced the state, built
    from ``stage_from_dict(original.to_dict())``. Keys must match exactly; arrays need
    exactly the expected dtype (nothing is converted), shape and finite values; the
    structural checks described in the module docstring apply. With ``expected_digest`` the
    digest of ``state`` must equal it. Anything wrong raises ``ValueError`` or ``TypeError``
    and leaves the stage unchanged.
    """
    owner = stage.name
    setter = getattr(stage, "set_state", None)
    if callable(setter):
        if stage.fitted:
            raise ValueError(
                f"{owner} is already fitted; restore into a fresh stage_from_dict(...) instance"
            )
        clean = _custom_state(owner, state)
        _verify_digest(owner, _entries_digest(clean), expected_digest)
        setter(dict(clean))
        if not stage.fitted:
            raise ValueError(f"{owner}.set_state did not leave the stage fitted")
        return stage
    target = _unwrap(stage)
    handler = _handler(target)
    fields = handler.fields if handler is not None else {}
    if not isinstance(state, Mapping):
        raise TypeError(f"{owner}: state must be a mapping, got {type(state).__name__}")
    missing, extra = sorted(set(fields) - set(state)), sorted(set(state) - set(fields))
    if missing or extra:
        raise ValueError(
            f"{owner}: state keys do not match the stage: missing {missing}, unexpected {extra}"
        )
    if not fields:
        _verify_digest(owner, None, expected_digest)
        return stage
    if stage.fitted:
        raise ValueError(
            f"{owner} is already fitted; restore into a fresh stage_from_dict(...) instance"
        )
    values = {key: _coerce(owner, key, field, state[key]) for key, field in fields.items()}
    handler.check(target, values)
    _verify_digest(owner, _entries_digest(values), expected_digest)
    handler.write(target, values)
    return stage


def _verify_digest(owner: str, actual: str | None, expected: str | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(
            f"{owner}: fitted-state digest {actual} does not match the expected {expected}"
        )


def fitted_state_digest(stage: Stage) -> str | None:
    """SHA-256 identity of the fitted state, or ``None`` when the stage holds none.

    The digest covers the array digests (dtype, shape, values) and scalar values of every
    entry under its key, and nothing else: stage name and parameters are identified
    separately by ``to_dict()``.
    """
    state = fitted_state(stage)
    return _entries_digest(state) if state else None


def fitted_state_nbytes(stage: Stage) -> int:
    """Bytes held by the arrays of the fitted state; scalar entries are not counted."""
    return sum(int(v.nbytes) for v in fitted_state(stage).values() if isinstance(v, np.ndarray))


def pipeline_fitted_identity(pipeline: Pipeline) -> list[str | None]:
    """``fitted_state_digest`` of every element of ``pipeline.stages``, in order.

    The quantizer is the last element of ``pipeline.stages`` when there is one. A pipeline
    without a quantizer stores float32 through an implicit stateless codec that has no entry.
    """
    return [fitted_state_digest(stage) for stage in pipeline.stages]


def pipeline_fitted_state(pipeline: Pipeline) -> list[dict[str, Any]]:
    """``fitted_state`` of every element of ``pipeline.stages``, in order."""
    return [fitted_state(stage) for stage in pipeline.stages]


def restore_pipeline_state(
    pipeline: Pipeline,
    states: Sequence[Mapping[str, Any]],
    *,
    expected_digests: Sequence[str | None] | None = None,
) -> Pipeline:
    """Restore one state per stage into a pipeline built with ``Pipeline.from_dict``.

    On an error the pipeline may be partly restored and should be discarded.
    """
    stages = pipeline.stages
    if len(states) != len(stages):
        raise ValueError(f"{len(states)} states for a pipeline of {len(stages)} stages")
    if expected_digests is not None and len(expected_digests) != len(stages):
        raise ValueError(f"{len(expected_digests)} digests for a pipeline of {len(stages)} stages")
    for i, (stage, state) in enumerate(zip(stages, states, strict=True)):
        want = None if expected_digests is None else expected_digests[i]
        try:
            restore_fitted_state(stage, state, expected_digest=want)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"stage {i} ({stage.name}): {exc}") from exc
    return pipeline
