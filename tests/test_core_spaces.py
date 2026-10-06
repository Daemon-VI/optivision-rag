"""Candidate spaces: identity, ordering, immutability, validation, fresh pipelines."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.core.identity import digest
from optivision.core.selection import (
    NOT_IN_SPACE,
    REGISTERED_DEFAULT_SPACE_SHA256,
    CandidateSpace,
    CandidateSpec,
)
from optivision.core.selection import spaces as spaces_module
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    Int4Quantizer,
    Int8Quantizer,
)

ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = ROOT / "reports" / "research" / "E1" / "store"
REGISTERED = "af29b01146a90a048ffea9cfbd66f0fedf988134829cd4dc56861f1207893520"

try:
    import scipy.cluster.hierarchy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAVE_SCIPY, reason="the registered default needs SciPy")


def small_families():
    return {
        "a": [Pipeline([Float16Quantizer()]), Pipeline([Int8Quantizer("per_vector")])],
        "b": [Pipeline([AdaptiveMerge(radius=0.9, center="mean"), Int4Quantizer(center="mean")]),
              Pipeline([BinaryQuantizer()])],
    }


def small_space(name="custom"):
    return CandidateSpace.from_families(small_families(), name=name)


# ------------------------------------------------------------------------ identity


def test_the_registered_constant_is_the_value_the_experiments_hashed():
    assert REGISTERED_DEFAULT_SPACE_SHA256 == REGISTERED


@needs_scipy
@pytest.mark.parametrize("dim", [128, 96])
def test_the_default_space_hashes_to_the_registered_value(dim):
    space = CandidateSpace.default(dim)
    assert space.identity == REGISTERED
    assert space.is_registered_default
    assert (len(space), space.n_families, space.name) == (40, 7, "default")
    assert space.families == ("float16", "ward+int8", "adaptive_centred+int8", "ward+int4",
                              "adaptive_centred+int4", "ward+binary", "adaptive_centred+binary")


@needs_scipy
def test_the_default_list_does_not_depend_on_the_width():
    assert CandidateSpace.default(2560).identity == CandidateSpace.default(128).identity


@needs_scipy
def test_the_default_space_keeps_the_duplicated_pipelines_and_the_chain_order():
    space = CandidateSpace.default(128)
    distinct = {digest(spec.to_dict()["pipeline"]) for spec in space}
    assert len(distinct) == 37  # 40 candidates, three exact duplicates kept as candidates
    for a, b in ((1, 7), (14, 20), (27, 33)):
        assert space[a].pipeline == space[b].pipeline and space[a].family != space[b].family
    assert [spec.position for spec in space] == list(range(40))
    assert space.candidate_families[:7] == ("float16",) + ("ward+int8",) * 6


def test_the_identity_is_the_digest_of_the_ordered_position_family_pipeline_list():
    space = small_space()
    expected = digest([{"position": k, "family": fam, "pipeline": pipe.to_dict()}
                       for k, (fam, pipe) in enumerate(
                           (f, p) for f, ps in small_families().items() for p in ps)])
    assert space.identity == expected
    assert not space.is_registered_default


def test_the_name_is_not_part_of_the_identity_but_everything_else_is():
    base = small_space("one")
    assert small_space("two").identity == base.identity
    families = small_families()
    relabelled = CandidateSpace.from_families({"x": families["a"], "b": families["b"]})
    assert relabelled.identity != base.identity
    reordered = CandidateSpace.from_families({"b": families["b"], "a": families["a"]})
    assert reordered.identity != base.identity
    swapped = CandidateSpace.from_families({"a": families["a"][::-1], "b": families["b"]})
    assert swapped.identity != base.identity
    changed = small_families()
    changed["a"][0] = Pipeline([Int8Quantizer("per_dimension")])
    assert CandidateSpace.from_families(changed).identity != base.identity


@pytest.mark.parametrize("store", ["colpali_docvqa", "colpali_infovqa", "colqwen2_docvqa",
                                   "colqwen2_infovqa", "colqwen25_docvqa", "colqwen25_infovqa",
                                   "scifact"])
def test_the_registered_candidate_lists_in_the_sidecars_hash_to_the_registered_value(store):
    side = STORE_DIR / f"{store}.json"
    if not side.is_file():
        pytest.skip(f"{side} is absent")
    candidates = json.loads(side.read_text(encoding="utf-8"))["candidates"]  # metadata only
    space = CandidateSpace("sidecar", tuple(
        CandidateSpec(c["position"], c["family"], c["pipeline"]) for c in candidates))
    assert space.identity == REGISTERED and space.is_registered_default


# --------------------------------------------------------------------- construction


def test_from_families_flattens_in_mapping_order_with_running_positions():
    space = small_space()
    assert [(s.position, s.family) for s in space] == [(0, "a"), (1, "a"), (2, "b"), (3, "b")]
    assert space.families == ("a", "b") and space.n_families == 2 and len(space) == 4
    assert space.candidate_families == ("a", "a", "b", "b")
    assert space.name == "custom" and small_space("mine").name == "mine"
    assert space[2].to_dict()["pipeline"] == small_families()["b"][0].to_dict()


def test_nothing_is_deduplicated_or_reordered():
    p = Pipeline([Float16Quantizer()])
    space = CandidateSpace.from_families({"a": [p, p, Pipeline([Float16Quantizer()])],
                                          "b": [p]})
    assert len(space) == 4 and space.n_families == 2
    assert len({digest(s.to_dict()["pipeline"]) for s in space}) == 1
    assert [s.position for s in space] == [0, 1, 2, 3]


def test_the_number_of_families_is_counted_from_the_list_not_assumed():
    one = CandidateSpace.from_families({"only": [Pipeline([Float16Quantizer()])]})
    many = CandidateSpace.from_families({f"f{j}": [Pipeline([Float16Quantizer()])]
                                         for j in range(11)})
    assert (one.n_families, many.n_families) == (1, 11)


@pytest.mark.parametrize("bad", [
    {},
    {"a": []},
    {"": [Pipeline([Float16Quantizer()])]},
    {3: [Pipeline([Float16Quantizer()])]},
    {"a": [{"stages": []}]},
    {"a": Pipeline([Float16Quantizer()])},
    {"a": "pipeline"},
    {"a": [Pipeline([Float16Quantizer()]), None]},
])
def test_from_families_refuses_malformed_input(bad):
    with pytest.raises((ValueError, TypeError)):
        CandidateSpace.from_families(bad)


def test_an_empty_family_is_refused_because_it_would_change_the_level():
    with pytest.raises(ValueError, match="no candidates"):
        CandidateSpace.from_families({"a": [Pipeline([Float16Quantizer()])], "b": []})


def test_direct_construction_validates_positions_and_content():
    spec = [CandidateSpec(k, "a", {"stages": []}) for k in range(3)]
    assert len(CandidateSpace("ok", tuple(spec))) == 3
    with pytest.raises(ValueError, match="at least one"):
        CandidateSpace("empty", ())
    with pytest.raises(ValueError, match="positions must be 0..K-1"):
        CandidateSpace("bad", (spec[0], spec[2]))
    with pytest.raises(ValueError, match="positions must be 0..K-1"):
        CandidateSpace("bad", (spec[1], spec[0], spec[2]))
    with pytest.raises(TypeError):
        CandidateSpace("bad", (spec[0], {"position": 1}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        CandidateSpace("", tuple(spec))


@needs_scipy
def test_default_and_wide_validate_the_width():
    for bad in (0, -1, 2.5, True, "128", None):
        with pytest.raises(ValueError):
            CandidateSpace.default(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            CandidateSpace.wide(bad)  # type: ignore[arg-type]


@needs_scipy
def test_the_wide_space_extends_the_default_and_is_not_the_registered_list():
    wide, default = CandidateSpace.wide(128), CandidateSpace.default(128)
    assert wide.name == "wide" and not wide.is_registered_default
    assert len(wide) > len(default) and wide.n_families > default.n_families
    assert [s.to_dict() for s in wide][:len(default)] == [s.to_dict() for s in default]
    assert CandidateSpace.wide(2560).identity != wide.identity  # depends on the width


def test_without_scipy_the_default_raises_unless_the_reduced_list_is_requested(monkeypatch):
    monkeypatch.setattr(spaces_module, "_have_scipy", lambda: False)
    with pytest.raises(ImportError, match="SciPy"):
        CandidateSpace.default(128)
    with pytest.raises(ImportError, match="SciPy"):
        CandidateSpace.wide(128)
    reduced = CandidateSpace.default(128, allow_reduced=True)
    assert reduced.name == "default-reduced" and reduced.n_families == 4
    assert reduced.families == ("float16", "adaptive_centred+int8", "adaptive_centred+int4",
                                "adaptive_centred+binary")
    assert not reduced.is_registered_default and reduced.identity != REGISTERED
    wide = CandidateSpace.wide(128, allow_reduced=True)
    assert wide.name == "wide-reduced" and len(wide) > len(reduced)


# --------------------------------------------------------------------- immutability


def test_specs_and_spaces_are_frozen():
    space = small_space()
    with pytest.raises(dataclasses.FrozenInstanceError):
        space.name = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        space.candidates = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        space[0].family = "z"  # type: ignore[misc]
    assert isinstance(space.candidates, tuple)


def test_the_pipeline_mapping_is_read_only_at_every_depth():
    spec = small_space()[2]
    with pytest.raises(TypeError):
        spec.pipeline["stages"] = ()  # type: ignore[index]
    stage = spec.pipeline["stages"][0]
    with pytest.raises(TypeError):
        stage["radius"] = 0.1  # type: ignore[index]
    assert isinstance(spec.pipeline["stages"], tuple)
    with pytest.raises(AttributeError):
        spec.pipeline["stages"].append(1)  # type: ignore[attr-defined]


def test_a_spec_is_a_deep_copy_of_what_it_was_built_from():
    description = Pipeline([AdaptiveMerge(radius=0.9, center="mean")]).to_dict()
    spec = CandidateSpec(0, "a", description)
    before = spec.to_dict()
    description["stages"][0]["radius"] = 0.1
    description["stages"].append({"stage": "binary"})
    assert spec.to_dict() == before


def test_to_dict_returns_a_fresh_plain_copy_each_time():
    spec = small_space()[2]
    plain = spec.to_dict()
    plain["pipeline"]["stages"][0]["radius"] = 0.1
    plain["family"] = "z"
    assert spec.to_dict()["pipeline"]["stages"][0]["radius"] == 0.9
    assert spec.to_dict()["family"] == "b"
    assert isinstance(plain["pipeline"]["stages"], list)


def test_specs_hash_and_compare_by_content():
    a, b = small_space()[1], small_space()[1]
    assert a == b and hash(a) == hash(b) and a is not b
    assert a != small_space()[0]
    assert hash(small_space()) == hash(small_space()) and small_space() == small_space()
    assert len({small_space(), small_space("other")}) == 2  # the name is a field of the value


@pytest.mark.parametrize(("args", "error"), [
    ((True, "a", {}), TypeError),
    ((1.0, "a", {}), TypeError),
    ((-2, "a", {}), ValueError),
    ((0, "", {}), ValueError),
    ((0, 3, {}), ValueError),
    ((0, "a", [1]), TypeError),
    ((0, "a", {"x": float("nan")}), ValueError),
    ((0, "a", {"x": object()}), TypeError),
    ((0, "a", {1: "x"}), TypeError),
])
def test_candidate_spec_validates_its_fields(args, error):
    with pytest.raises(error):
        CandidateSpec(*args)


def test_a_spec_outside_any_space_may_carry_position_minus_one():
    assert CandidateSpec(NOT_IN_SPACE, "fallback", {"stages": []}).position == -1
    assert CandidateSpec(np.int64(3), "a", {"stages": []}).position == 3


# ---------------------------------------------------------------------- pipelines()


def test_pipelines_returns_fresh_unfitted_objects_each_call():
    space = small_space()
    left, right = space.pipelines(), space.pipelines()
    assert len(left) == 4 and all(isinstance(p, Pipeline) for p in left)
    for a, b in zip(left, right, strict=True):
        assert a is not b and a.to_dict() == b.to_dict()
    assert [p.to_dict() for p in left] == [s.to_dict()["pipeline"] for s in space]
    assert not any(p.fitted for p in left[2:3])  # the merge stage is not fitted
    left[0].stages.append(Int8Quantizer())  # mutating a returned pipeline...
    assert space.pipelines()[0].to_dict() == small_families()["a"][0].to_dict()  # ...is local
    assert space[0].to_pipeline() is not space[0].to_pipeline()


# ----------------------------------------------------------------- serialisation


def test_to_dict_from_dict_round_trip_keeps_the_identity_through_json_text():
    space = small_space("round-trip")
    text = json.dumps(space.to_dict(), allow_nan=False)
    back = CandidateSpace.from_dict(json.loads(text))
    assert back == space and back.identity == space.identity and back.name == "round-trip"


@needs_scipy
def test_the_default_space_round_trips_and_stays_registered():
    space = CandidateSpace.default(128)
    back = CandidateSpace.from_dict(json.loads(json.dumps(space.to_dict())))
    assert back.is_registered_default and back == space


def test_from_dict_detects_tampering_and_foreign_formats():
    record = small_space().to_dict()
    tampered = json.loads(json.dumps(record))
    tampered["candidates"][0]["family"] = "z"
    with pytest.raises(ValueError, match="identity"):
        CandidateSpace.from_dict(tampered)
    with pytest.raises(ValueError, match="format"):
        CandidateSpace.from_dict({**record, "format": "other"})
    with pytest.raises(ValueError, match="version"):
        CandidateSpace.from_dict({**record, "version": 2})
    extra = json.loads(json.dumps(record))
    extra["candidates"][0]["note"] = "x"
    with pytest.raises(ValueError, match="exactly position, family and pipeline"):
        CandidateSpace.from_dict(extra)


def test_the_documentation_states_the_default_is_a_development_informed_default():
    doc = spaces_module.__doc__ + CandidateSpace.default.__doc__
    assert "development-informed" in doc
    assert "neither neutral" in doc or "not a neutral" in doc
    assert "changes the number of families" in doc.replace("\n", " ")


def test_specs_and_spaces_survive_copy_deepcopy_and_pickle():
    import copy
    import pickle

    space = small_space("copies")
    for clone in (copy.deepcopy(space), pickle.loads(pickle.dumps(space))):
        assert clone == space and clone.identity == space.identity
        with pytest.raises(TypeError):
            clone[0].pipeline["stages"] = ()  # type: ignore[index]
    spec = space[2]
    assert copy.deepcopy(spec) == spec and pickle.loads(pickle.dumps(spec)) == spec
