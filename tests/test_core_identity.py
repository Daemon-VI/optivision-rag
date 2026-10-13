"""Canonical JSON, content digests, and the identity of the registered candidate space."""

from __future__ import annotations

import datetime
import decimal
import enum
import hashlib
import json
import pathlib
import random

import numpy as np
import pytest

from optivision.core import identity

#: sha256 of the 40 default candidates (d = 96 and d = 128): scripts/research/x_common.py:60
REGISTERED_SPACE_SHA256 = "af29b01146a90a048ffea9cfbd66f0fedf988134829cd4dc56861f1207893520"


def _dumps(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# ------------------------------------------------------------ canonical JSON


_STRINGS = [
    "", "a", "Z", chr(0xE9), chr(0x65E5) + chr(0x672C), "line\nbreak", 'q"uote', "\\",
    chr(0x1F600),
]


def _random_json(rng: random.Random, depth: int = 0):
    kinds = ["int", "float", "str", "bool", "none"] + (["list", "dict"] if depth < 3 else [])
    kind = rng.choice(kinds)
    if kind == "int":
        return rng.choice([0, 1, -1, 2**40, -(2**70), rng.randrange(-10**6, 10**6)])
    if kind == "float":
        return rng.choice([0.0, -0.0, 0.5, 0.1, 1e-300, 1.5e300, 5e-324, rng.uniform(-1e3, 1e3)])
    if kind == "str":
        return rng.choice(_STRINGS)
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "none":
        return None
    if kind == "list":
        return [_random_json(rng, depth + 1) for _ in range(rng.randrange(0, 5))]
    keys = ["b", "a", "\u00e9", "Z", "key with space", "", "10", "9", "\u2603"]
    return {rng.choice(keys): _random_json(rng, depth + 1) for _ in range(rng.randrange(0, 6))}


@pytest.mark.parametrize("seed", range(60))
def test_canonical_json_equals_json_dumps_for_json_native_objects(seed):
    obj = _random_json(random.Random(seed))
    assert identity.canonical_json(obj) == _dumps(obj)


def test_canonical_json_does_not_depend_on_insertion_order():
    a = {"x": 1, "y": [1, 2, {"q": 1, "p": 2}]}
    b = {"y": [1, 2, {"p": 2, "q": 1}], "x": 1}
    assert identity.canonical_json(a) == identity.canonical_json(b)
    assert identity.digest(a) == identity.digest(b)


class _Color(enum.Enum):
    RED = "red"
    ONE = 1


class _Level(str, enum.Enum):
    LOW = "low"


class _Count(enum.IntEnum):
    TWO = 2


def test_numpy_values_tuples_and_enums_become_json_values():
    obj = {
        "f32": np.float32(0.5),
        "f16": np.float16(0.5),
        "i64": np.int64(3),
        "u8": np.uint8(255),
        "bool": np.bool_(True),
        "array": np.arange(3, dtype=np.int16),
        "matrix": np.array([[1.5, 2.5]]),
        "tuple": (1, (2, 3)),
        "enum": _Color.RED,
        "enum_int": _Color.ONE,
        "str_enum": _Level.LOW,
        "int_enum": _Count.TWO,
        "np_str": np.str_("x"),
    }
    expected = {
        "f32": 0.5, "f16": 0.5, "i64": 3, "u8": 255, "bool": True, "array": [0, 1, 2],
        "matrix": [[1.5, 2.5]], "tuple": [1, [2, 3]], "enum": "red", "enum_int": 1,
        "str_enum": "low", "int_enum": 2, "np_str": "x",
    }
    assert identity.canonical_json(obj) == _dumps(expected)


def test_a_float32_is_widened_to_the_double_it_equals():
    value = np.float32(0.1)
    assert identity.canonical_json(value) == json.dumps(float(value)).encode()
    assert identity.canonical_json(value) != b"0.1"


def test_output_is_ascii_without_whitespace():
    out = identity.canonical_json({"k": "\u00e9\u2603\U0001f600", "a": [1, 2]})
    assert out == b'{"a":[1,2],"k":"\\u00e9\\u2603\\ud83d\\ude00"}'
    assert json.loads(out) == {"k": "\u00e9\u2603\U0001f600", "a": [1, 2]}


def test_json_booleans_stay_distinct_from_numbers():
    assert identity.canonical_json([True, 1, 1.0]) == b"[true,1,1.0]"
    assert identity.digest({"a": 1}) != identity.digest({"a": 1.0}) != identity.digest({"a": True})


@pytest.mark.parametrize("value", [
    float("nan"), float("inf"), float("-inf"), np.nan, np.float32("nan"), np.float16("inf"),
    np.array([1.0, np.nan]), {"x": [float("inf")]}, [[0.0, -np.inf]],
])
def test_non_finite_numbers_are_refused(value):
    with pytest.raises(ValueError, match="NaN or infinity"):
        identity.canonical_json(value)


@pytest.mark.parametrize("value", [
    {1: "a"}, {None: 1}, {1.5: 1}, {("a",): 1}, {b"a": 1}, {"ok": {2: "nested"}}, [{3: 3}],
])
def test_non_string_keys_are_refused(value):
    with pytest.raises(TypeError, match="string keys"):
        identity.canonical_json(value)


@pytest.mark.parametrize("value", [
    b"abc", bytearray(b"x"), np.bytes_(b"x"), [b"inside"], {"k": b"v"},
])
def test_bytes_are_refused(value):
    with pytest.raises(TypeError):
        identity.canonical_json(value)


@pytest.mark.parametrize("value", [
    object(), {1, 2}, frozenset({"a"}), 1 + 2j, np.complex64(1), pathlib.Path("x"), range(3),
    decimal.Decimal("1.5"), datetime.date(2026, 10, 6), np.datetime64("2026-10-06"), lambda: 0,
    (i for i in range(2)),
])
def test_unknown_objects_are_refused(value):
    with pytest.raises(TypeError, match="cannot hold"):
        identity.canonical_json(value)


def test_sha256_hex_is_sha256():
    assert identity.sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert identity.sha256_hex(b"abc").startswith("ba7816bf")


def test_digest_is_sha256_of_canonical_json():
    obj = {"b": [1, 2.5, None], "a": "x"}
    assert identity.digest(obj) == hashlib.sha256(_dumps(obj)).hexdigest()


# --------------------------------------------- the registered candidate space


def _default_space_spec(dim):
    from optivision.calibration import recommended_search_space

    spec = []
    for family, pipelines in recommended_search_space(dim).items():
        for pipe in pipelines:
            spec.append({"position": len(spec), "family": family, "pipeline": pipe.to_dict()})
    return spec


@pytest.mark.parametrize("dim", [128, 96])
def test_default_candidate_space_reproduces_the_registered_hash(dim):
    """The research hashed the ordered {position, family, pipeline} list (x_common.py:453-461)."""
    pytest.importorskip("scipy")  # without SciPy the default space has 22 candidates, not 40
    spec = _default_space_spec(dim)
    assert len(spec) == 40
    assert identity.digest(spec) == REGISTERED_SPACE_SHA256
    # the research's own formula (x_common.py:449-450), spelled out
    assert hashlib.sha256(_dumps(spec)).hexdigest() == REGISTERED_SPACE_SHA256


def _change_a_parameter(spec):
    spec[2]["pipeline"]["stages"][0]["ratio"] = 0.6


def _rename_a_family(spec):
    spec[5]["family"] = "other"


def _swap_two_candidates(spec):
    spec[3], spec[4] = spec[4], spec[3]


def _drop_a_candidate(spec):
    del spec[-1]


@pytest.mark.parametrize("change", [
    _change_a_parameter, _rename_a_family, _swap_two_candidates, _drop_a_candidate,
])
def test_any_change_to_the_candidate_list_changes_its_hash(change):
    pytest.importorskip("scipy")
    spec = _default_space_spec(128)
    change(spec)
    assert identity.digest(spec) != REGISTERED_SPACE_SHA256


# ------------------------------------------------------------- array digests


def test_array_digest_scheme_is_dtype_shape_then_little_endian_bytes():
    a = np.arange(6, dtype="<i4").reshape(2, 3)
    assert identity.array_digest(a) == hashlib.sha256(b"<i4|2,3|" + a.tobytes()).hexdigest()
    b = np.arange(3, dtype="<f4")
    assert identity.array_digest(b) == hashlib.sha256(b"<f4|3|" + b.tobytes()).hexdigest()
    zero_d = np.float64(1.5)
    expected = hashlib.sha256(b"<f8||" + zero_d.tobytes()).hexdigest()
    assert identity.array_digest(zero_d) == expected


def test_array_digest_is_sensitive_to_dtype_and_shape():
    zeros = {
        "f32": np.zeros(4, np.float32), "f64": np.zeros(4, np.float64),
        "i32": np.zeros(4, np.int32), "u32": np.zeros(4, np.uint32),
        "f16": np.zeros(4, np.float16),
    }
    digests = {k: identity.array_digest(v) for k, v in zeros.items()}
    assert len(set(digests.values())) == len(digests)  # same bytes, different dtype
    shapes = [np.zeros((2, 3)), np.zeros((3, 2)), np.zeros(6), np.zeros((1, 6)), np.zeros((6, 1))]
    assert len({identity.array_digest(s) for s in shapes}) == len(shapes)


@pytest.mark.parametrize("index", [(0, 0), (3, 6), (999, 6), (500, 2)])
def test_array_digest_is_sensitive_to_one_changed_value(index):
    a = np.random.default_rng(0).standard_normal((1000, 7)).astype(np.float32)
    b = a.copy()
    b[index] = np.nextafter(b[index], np.float32(10))  # the smallest possible change
    assert identity.array_digest(a) != identity.array_digest(b)


def test_array_digest_tells_zero_from_negative_zero_and_hashes_nan_bits_as_stored():
    zero, neg = np.array([0.0], np.float32), np.array([-0.0], np.float32)
    assert identity.array_digest(zero) != identity.array_digest(neg)
    quiet = np.array([0x7FC00000], np.uint32).view(np.float32)
    other = np.array([0xFFC00000], np.uint32).view(np.float32)
    assert np.isnan(quiet).all() and np.isnan(other).all()
    assert identity.array_digest(quiet) != identity.array_digest(other)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "i2", "i4", "i8", "u2", "u4", "u8"])
def test_array_digest_normalises_byte_order(dtype):
    a = np.arange(24).reshape(4, 6).astype("<" + dtype)
    big = a.astype(">" + dtype)
    assert big.dtype.byteorder == ">" and big.tobytes() != a.tobytes()
    assert identity.array_digest(big) == identity.array_digest(a)
    assert identity.array_digest(a.astype("=" + dtype)) == identity.array_digest(a)


@pytest.mark.parametrize("shape", [(0,), (1,), (37,), (13, 5), (4, 3, 5)])
@pytest.mark.parametrize("block", [1, 7, 64, 1000, 10**9])
def test_array_digest_does_not_depend_on_the_block_size(monkeypatch, shape, block):
    a = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    whole = identity.array_digest(a)
    monkeypatch.setattr(identity, "_BLOCK_BYTES", block)
    assert identity.array_digest(a) == whole


def test_array_digest_ignores_memory_layout():
    a = np.random.default_rng(2).standard_normal((6, 8)).astype(np.float32)
    assert identity.array_digest(np.asfortranarray(a)) == identity.array_digest(a)
    view = a[:, ::2]
    assert not view.flags.c_contiguous
    assert identity.array_digest(view) == identity.array_digest(np.ascontiguousarray(view))
    assert identity.array_digest(a.T) == identity.array_digest(np.ascontiguousarray(a.T))


@pytest.mark.parametrize("shape", [(0,), (0, 5), (5, 0), (0, 0, 3)])
def test_array_digest_of_empty_arrays(shape):
    empty = np.zeros(shape, np.float32)
    digest = identity.array_digest(empty)
    assert isinstance(digest, str) and len(digest) == 64
    assert digest == identity.array_digest(np.zeros(shape, np.float32))
    assert digest != identity.array_digest(np.zeros(shape, np.float64))


def test_empty_arrays_of_different_shape_differ():
    shapes = [(0,), (0, 5), (5, 0), (0, 0)]
    assert len({identity.array_digest(np.zeros(s, np.float32)) for s in shapes}) == len(shapes)


def test_array_digest_of_a_memory_mapped_array(tmp_path, monkeypatch):
    path = tmp_path / "a.npy"
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(50, 9))
    mm[:] = np.random.default_rng(3).standard_normal((50, 9)).astype(np.float32)
    mm.flush()
    expected = identity.array_digest(np.array(mm))
    assert identity.array_digest(mm) == expected
    loaded = np.load(path, mmap_mode="r")
    assert isinstance(loaded, np.memmap)
    assert identity.array_digest(loaded) == expected
    monkeypatch.setattr(identity, "_BLOCK_BYTES", 100)
    assert identity.array_digest(loaded) == expected


def test_array_digest_copies_one_block_at_a_time(monkeypatch):
    import tracemalloc

    monkeypatch.setattr(identity, "_BLOCK_BYTES", 256 * 1024)
    a = np.zeros((8000, 256), np.float32)  # 8 MB, 32 blocks
    tracemalloc.start()
    try:
        identity.array_digest(a)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < a.nbytes / 8


@pytest.mark.parametrize("array", [
    np.array([1, "a"], dtype=object),
    np.array([None, 1.0], dtype=object),
    np.zeros(2, dtype=[("x", object)]),
])
def test_array_digest_rejects_object_arrays(array):
    with pytest.raises(TypeError, match="numeric, boolean or fixed-width string"):
        identity.array_digest(array)


def test_array_digest_accepts_bool_string_and_list_input():
    digest = identity.array_digest
    assert digest(np.array([True, False])) != digest(np.array([False, True]))
    assert digest(np.array(["ab", "c"])) == digest(np.array(["ab", "c"]))
    assert digest(np.array(["ab", "c"])) != digest(np.array(["ab", "cd"]))
    assert digest([1.0, 2.0]) == digest(np.array([1.0, 2.0]))


def test_structured_dtypes_with_different_fields_do_not_collide():
    a = np.zeros(2, dtype=[("x", "<i4"), ("y", "<i4")])
    b = np.zeros(2, dtype=[("a", "<i8")])
    try:
        same = identity.array_digest(a) == identity.array_digest(b)
    except TypeError:  # refusing them is a fix too
        return
    assert not same


# ------------------------------------------------------------ strings digest


def test_strings_digest_is_the_digest_of_the_list():
    assert identity.strings_digest(["a", "b"]) == identity.digest(["a", "b"])
    assert identity.strings_digest(("a", "b")) == identity.strings_digest(["a", "b"])


def test_strings_digest_is_ordered_and_unambiguous():
    assert identity.strings_digest(["a", "b"]) != identity.strings_digest(["b", "a"])
    assert identity.strings_digest(["a,b"]) != identity.strings_digest(["a", "b"])
    assert identity.strings_digest(["ab"]) != identity.strings_digest(["a", "b"])
    assert identity.strings_digest([]) == identity.digest([])
    assert identity.strings_digest([""]) != identity.strings_digest([])


def test_strings_digest_turns_items_into_strings():
    assert identity.strings_digest([1, 2]) == identity.strings_digest(["1", "2"])


def test_a_bare_string_is_not_a_sequence_of_strings():
    try:
        same = identity.strings_digest("ab") == identity.strings_digest(["a", "b"])
    except TypeError:  # refusing it is a fix too
        return
    assert not same
