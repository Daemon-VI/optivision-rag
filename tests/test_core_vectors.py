"""The vector contract: corpus identity, validation, host conversion, encoder identity."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from optivision.core import identity
from optivision.core import vectors as vectors_module
from optivision.core.vectors import (
    ContractError,
    CorpusIdentity,
    EncoderIdentity,
    MultiVectorCorpus,
    MultiVectorRepresentation,
    as_host_array,
    corpus_identity,
    validate_corpus,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _full_corpus(dtype=np.float32) -> MultiVectorCorpus:
    """Nine vectors in four documents (the second is empty); every aligned array present."""
    rng = np.random.default_rng(0)
    positions = rng.random((9, 2)).astype(np.float32)
    positions[[0, 3, 7]] = np.nan  # the first row of each non-empty document is off the grid
    protected = np.zeros(9, bool)
    protected[[0, 3, 7]] = True
    return MultiVectorCorpus(
        rng.standard_normal((9, 6)).astype(dtype),
        [0, 3, 3, 7, 9],
        ids=["d0", "d1", "d2", "d3"],
        metadata=[{"n": 3}, {}, {"n": 4}, {"n": 2}],
        protected=protected,
        positions=positions,
        weights=rng.integers(1, 4, 9).astype(np.float32),
        importance=rng.random(9).astype(np.float32),
        attrs={"model": "synthetic"},
    )


def _rebuild(c: MultiVectorCorpus, **changes) -> MultiVectorCorpus:
    """A new corpus from copies of ``c``'s arrays, with some arguments replaced."""

    def copied(a):
        return None if a is None else a.copy()

    args = {
        "vectors": c.vectors.copy(), "offsets": c.offsets.copy(), "ids": list(c.ids),
        "metadata": c.metadata, "protected": copied(c.protected), "positions": copied(c.positions),
        "weights": copied(c.weights), "importance": copied(c.importance), "attrs": c.attrs,
    }
    args.update(changes)
    return MultiVectorCorpus(**args)


def _tamper(c: MultiVectorCorpus, **attributes) -> MultiVectorCorpus:
    """A shallow copy of ``c`` whose attributes are reassigned, bypassing the constructor."""
    clone = copy.copy(c)
    for key, value in attributes.items():
        setattr(clone, key, value)
    return clone


def _run(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    path = os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=False,
                          env={**os.environ, "PYTHONPATH": path}, capture_output=True,
                          text=True, timeout=120)


# ------------------------------------------------------------- corpus identity


def test_identity_reports_the_shape_of_the_corpus():
    ident = corpus_identity(_full_corpus())
    assert (ident.n_docs, ident.n_vectors, ident.dimension, ident.dtype) == (4, 9, 6, "float32")
    assert corpus_identity(_full_corpus(np.float16)).dtype == "float16"


def test_identity_is_a_frozen_value_with_a_digest_of_its_dict():
    ident = corpus_identity(_full_corpus())
    assert isinstance(ident, CorpusIdentity)
    with pytest.raises(AttributeError):
        ident.n_docs = 5
    assert ident.digest() == identity.digest(ident.to_dict())
    assert ident == corpus_identity(_full_corpus())
    assert set(ident.to_dict()) == {
        "n_docs", "n_vectors", "dimension", "dtype", "vectors_sha256", "offsets_sha256",
        "ids_sha256", "protected_sha256", "positions_sha256", "weights_sha256",
        "importance_sha256",
    }
    json.loads(identity.canonical_json(ident.to_dict()))  # JSON-native, None included


def test_identity_can_be_recomputed_from_the_arrays_alone():
    """The scheme is documented, not hidden: dtype, shape and bytes, then the canonical JSON."""
    c = MultiVectorCorpus(
        (np.arange(12, dtype=np.float32) / 4).reshape(6, 2), [0, 2, 2, 6], ids=["a", "b", "c"],
        protected=np.array([True, False, False, False, False, False]),
    )

    def sha(prefix: bytes, array: np.ndarray) -> str:
        return hashlib.sha256(prefix + array.tobytes()).hexdigest()

    expected = {
        "n_docs": 3, "n_vectors": 6, "dimension": 2, "dtype": "float32",
        "vectors_sha256": sha(b"<f4|6,2|", c.vectors),
        "offsets_sha256": sha(b"<i8|4|", c.offsets),
        "ids_sha256": hashlib.sha256(b'["a","b","c"]').hexdigest(),
        "protected_sha256": sha(b"|b1|6|", c.protected),
        "positions_sha256": None, "weights_sha256": None, "importance_sha256": None,
    }
    ident = corpus_identity(c)
    assert ident.to_dict() == expected
    text = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    assert ident.digest() == hashlib.sha256(text).hexdigest()


@pytest.mark.parametrize("compressed", [False, True])
def test_identity_is_stable_across_save_and_load(tmp_path, compressed):
    c = _full_corpus()  # includes NaN in positions: the bits survive save and load
    assert np.isnan(c.positions).any()
    path = c.save(tmp_path / "c.npz", compressed=compressed)
    assert corpus_identity(MultiVectorCorpus.load(path)) == corpus_identity(c)
    assert corpus_identity(MultiVectorCorpus.load(path, mmap=True)) == corpus_identity(c)


def test_identity_of_a_float16_corpus_is_stable_across_save_and_load(tmp_path):
    c = _full_corpus(np.float16)
    back = MultiVectorCorpus.load(c.save(tmp_path / "h.npz"))
    assert back.dtype == np.float16
    assert corpus_identity(back) == corpus_identity(c)


def _more_vectors(c):
    v = c.vectors.copy()
    v[5, 2] += 1.0
    return {"vectors": v}


def _moved_boundary(c):
    return {"offsets": np.array([0, 2, 3, 7, 9])}  # one row moves from document 0 to document 1


def _other_id(c):
    return {"ids": ["d0", "d1", "other", "d3"]}


def _flipped_protection(c):
    p = c.protected.copy()
    p[4] = ~p[4]
    return {"protected": p}


def _moved_position(c):
    p = c.positions.copy()
    p[5, 0] += 0.25
    return {"positions": p}


def _other_weight(c):
    w = c.weights.copy()
    w[6] += 1
    return {"weights": w}


def _other_importance(c):
    i = c.importance.copy()
    i[1] += 0.5
    return {"importance": i}


@pytest.mark.parametrize(("tweak", "component"), [
    (_more_vectors, "vectors_sha256"),
    (_moved_boundary, "offsets_sha256"),
    (_other_id, "ids_sha256"),
    (_flipped_protection, "protected_sha256"),
    (_moved_position, "positions_sha256"),
    (_other_weight, "weights_sha256"),
    (_other_importance, "importance_sha256"),
])
def test_identity_changes_with_exactly_the_array_that_changed(tweak, component):
    base = _full_corpus()
    before, after = corpus_identity(base), corpus_identity(_rebuild(base, **tweak(base)))
    changed = {k for k, v in before.to_dict().items() if after.to_dict()[k] != v}
    assert changed == {component}
    assert before.digest() != after.digest()


@pytest.mark.parametrize("key", ["protected", "positions", "weights", "importance"])
def test_an_absent_aligned_array_differs_from_an_all_default_one(key):
    base = _full_corpus()
    defaults = {
        "protected": np.zeros(9, bool), "positions": np.full((9, 2), np.nan, np.float32),
        "weights": np.ones(9, np.float32), "importance": np.zeros(9, np.float32),
    }
    absent = corpus_identity(_rebuild(base, **{key: None}))
    default = corpus_identity(_rebuild(base, **{key: defaults[key]}))
    assert absent.to_dict()[f"{key}_sha256"] is None
    assert default.to_dict()[f"{key}_sha256"] is not None
    assert absent.digest() != default.digest() != corpus_identity(base).digest()


def test_equal_corpora_built_in_different_ways_have_one_identity():
    rng = np.random.default_rng(1)
    docs = [rng.standard_normal((n, 5)).astype(np.float32) for n in (3, 0, 4, 1)]
    ids = ["a", "b", "c", "d"]
    one = MultiVectorCorpus.from_arrays(docs, ids=ids)
    two = MultiVectorCorpus.from_documents(
        [MultiVectorRepresentation(d, doc_id=i) for d, i in zip(docs, ids, strict=True)])
    three = MultiVectorCorpus(np.concatenate(docs), [0, 3, 3, 7, 8], ids=ids)
    four = one.select(range(len(one)))
    five = one.with_vectors(one.vectors.copy())
    assert len({corpus_identity(c).digest() for c in (one, two, three, four, five)}) == 1


def test_a_float16_corpus_is_not_the_float32_corpus_it_was_rounded_from():
    values = (np.arange(24).reshape(6, 4) / 4).astype(np.float32)  # exact in float16
    c32 = MultiVectorCorpus.from_arrays([values])
    c16 = c32.astype(np.float16)
    a, b = corpus_identity(c32), corpus_identity(c16)
    assert a.dtype != b.dtype and a.vectors_sha256 != b.vectors_sha256 and a.digest() != b.digest()
    assert a.offsets_sha256 == b.offsets_sha256 and a.ids_sha256 == b.ids_sha256
    assert corpus_identity(c16.astype(np.float32)) == a  # content, not history


def test_images_metadata_and_attrs_are_not_part_of_the_identity():
    base = _full_corpus()
    other = _rebuild(base, metadata=[{"x": 1}, {"y": 2}, {}, {"z": [1, 2]}],
                     attrs={"model": "other"})
    other.images = ["page-0", "page-1", "page-2", "page-3"]
    assert corpus_identity(other) == corpus_identity(base)


def test_row_order_inside_the_vectors_matters():
    base = _full_corpus()
    swapped = base.vectors.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert corpus_identity(_rebuild(base, vectors=swapped)) != corpus_identity(base)


def test_nan_bits_in_positions_are_hashed_as_stored():
    """Documented: the identity is a function of bytes, so another NaN is another digest."""
    base = _full_corpus()
    quiet = base.positions.copy()
    quiet[0, :] = np.array([0x7FC00000], np.uint32).view(np.float32)
    signed = quiet.copy()
    signed[0, :] = np.array([0xFFC00000], np.uint32).view(np.float32)
    assert np.isnan(quiet[0]).all() and np.isnan(signed[0]).all()
    a = corpus_identity(_rebuild(base, positions=quiet))
    b = corpus_identity(_rebuild(base, positions=signed))
    assert a.positions_sha256 != b.positions_sha256
    assert a.vectors_sha256 == b.vectors_sha256


def test_the_library_writes_the_same_nan_every_time():
    """So identities of corpora built from page encodings or documents agree."""
    docs = [
        MultiVectorRepresentation(np.ones((2, 3), np.float32), positions=np.full((2, 2), np.nan)),
        MultiVectorRepresentation(np.ones((1, 3), np.float32)),
    ]
    filled = MultiVectorCorpus.from_documents(docs)  # the 2nd document's NaN is the library's
    explicit = MultiVectorCorpus(np.ones((3, 3), np.float32), [0, 2, 3],
                                 positions=np.full((3, 2), np.nan, np.float32))
    assert corpus_identity(filled) == corpus_identity(explicit)


def test_identity_of_memory_mapped_vectors_equals_that_of_the_same_values_in_memory(tmp_path):
    mm = np.lib.format.open_memmap(tmp_path / "v.npy", mode="w+", dtype=np.float32, shape=(40, 7))
    mm[:] = np.random.default_rng(2).standard_normal((40, 7)).astype(np.float32)
    mapped = MultiVectorCorpus(mm, [0, 10, 40])
    in_memory = MultiVectorCorpus(np.array(mm), [0, 10, 40])
    assert corpus_identity(mapped) == corpus_identity(in_memory)


def test_identity_needs_a_corpus():
    with pytest.raises(TypeError, match="MultiVectorCorpus"):
        corpus_identity([np.zeros((2, 2), np.float32)])


# -------------------------------------------------------------- validation


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
def test_a_valid_corpus_passes(dtype):
    validate_corpus(_full_corpus(dtype))
    validate_corpus(_full_corpus(dtype), expect_dim=6, expect_dtype=dtype, allow_empty_docs=True)


def test_corpora_without_vectors_pass_and_have_an_identity():
    for offsets, n_docs in (([0, 0], 1), ([0], 0)):  # one empty document, and no documents
        c = MultiVectorCorpus(np.zeros((0, 4), np.float32), offsets)
        validate_corpus(c)
        ident = corpus_identity(c)
        assert (ident.n_docs, ident.n_vectors, ident.dimension) == (n_docs, 0, 4)
    assert corpus_identity(MultiVectorCorpus(np.zeros((0, 4), np.float32), [0])) != corpus_identity(
        MultiVectorCorpus(np.zeros((0, 5), np.float32), [0]))
    with pytest.raises(ContractError, match="document 0 .* has no vectors"):
        validate_corpus(MultiVectorCorpus(np.zeros((0, 4), np.float32), [0, 0]),
                        allow_empty_docs=False)


def test_validation_changes_nothing_and_converts_nothing():
    c = _full_corpus(np.float16)
    before = corpus_identity(c).digest()
    arrays = {k: getattr(c, k) for k in ("vectors", "offsets", "protected", "positions")}
    validate_corpus(c, expect_dtype="float16")
    assert corpus_identity(c).digest() == before
    assert c.dtype == np.float16
    assert all(getattr(c, k) is v for k, v in arrays.items())
    with pytest.raises(ContractError):
        validate_corpus(c, expect_dtype="float32")  # a mismatch is reported, not converted
    assert c.dtype == np.float16 and corpus_identity(c).digest() == before


def test_a_failing_validation_leaves_the_corpus_alone_too():
    c = _full_corpus()
    c.vectors[3, 1] = np.nan
    before = [c.vectors.copy(), c.offsets.copy()]
    with pytest.raises(ContractError):
        validate_corpus(c)
    assert np.array_equal(c.vectors, before[0], equal_nan=True)
    assert np.array_equal(c.offsets, before[1])


def _with_vector(c, row, col, value, dtype=None):
    v = np.array(c.vectors, dtype=dtype or c.vectors.dtype)
    v[row, col] = value
    return _tamper(c, vectors=v)


def _with_aligned(c, key, row, value):
    a = getattr(c, key).copy()
    a[row] = value
    return _tamper(c, **{key: a})


_CASES = [
    # (id, corpus factory, keyword arguments, message pattern)
    ("not-a-corpus", lambda c: [np.zeros((2, 2), np.float32)], {},
     "expected a MultiVectorCorpus, got list"),
    ("vectors-1d", lambda c: _tamper(c, vectors=np.zeros(9, np.float32)), {},
     "vectors must be a 2-D"),
    ("vectors-int", lambda c: _tamper(c, vectors=np.zeros((9, 6), np.int8)), {},
     "vectors must be floating point, got int8"),
    ("vectors-list", lambda c: _tamper(c, vectors=[[0.0] * 6] * 9), {},
     "vectors must be a 2-D array"),
    ("expected-dim", lambda c: c, {"expect_dim": 7}, "expected dimension 7, got 6"),
    ("expected-dtype", lambda c: c, {"expect_dtype": "float16"},
     "expected dtype float16, got float32"),
    ("offsets-start", lambda c: _tamper(c, offsets=np.array([1, 3, 3, 7, 9])), {},
     "offsets must start at 0, got 1"),
    ("offsets-decrease", lambda c: _tamper(c, offsets=np.array([0, 5, 3, 7, 9])), {},
     r"offsets decrease at document 1: offsets\[1\]=5 > offsets\[2\]=3"),
    ("offsets-end", lambda c: _tamper(c, offsets=np.array([0, 3, 3, 7, 8])), {},
     "offsets end at 8 but there are 9 vectors"),
    ("offsets-dtype", lambda c: _tamper(c, offsets=c.offsets.astype(np.int32)), {}, "int64"),
    ("offsets-2d", lambda c: _tamper(c, offsets=c.offsets.reshape(1, -1)), {}, "1-D int64"),
    ("offsets-empty", lambda c: _tamper(c, offsets=np.zeros(0, np.int64)), {},
     r"n_docs \+ 1 entries"),
    ("ids-count", lambda c: _tamper(c, ids=["a", "b"]), {}, "2 ids for 4 documents"),
    ("ids-type", lambda c: _tamper(c, ids=["a", 1, "c", "d"]), {},
     "the id of document 1 is int, not str"),
    ("ids-duplicate", lambda c: _tamper(c, ids=["a", "b", "a", "d"]), {},
     "id 'a' is used by documents 0 and 2"),
    ("metadata-count", lambda c: _tamper(c, metadata=[{}]), {},
     "1 metadata entries for 4 documents"),
    ("images-count", lambda c: _tamper(c, images=["x", "y"]), {}, "2 images for 4 documents"),
    ("protected-shape", lambda c: _tamper(c, protected=np.zeros(8, bool)), {},
     r"protected must have shape \(9,\), got \(8,\)"),
    ("protected-dtype", lambda c: _tamper(c, protected=np.zeros(9, np.int64)), {},
     "protected must be bool, got int64"),
    ("positions-shape", lambda c: _tamper(c, positions=np.zeros((9, 3), np.float32)), {},
     r"positions must have shape \(9, 2\), got \(9, 3\)"),
    ("positions-dtype", lambda c: _tamper(c, positions=np.zeros((9, 2), np.int32)), {},
     "positions must be floating point, got int32"),
    ("weights-2d", lambda c: _tamper(c, weights=np.ones((9, 1), np.float32)), {},
     r"weights must have shape \(9,\), got \(9, 1\)"),
    ("importance-length", lambda c: _tamper(c, importance=np.ones(10, np.float32)), {},
     r"importance must have shape \(9,\), got \(10,\)"),
    ("empty-document", lambda c: c, {"allow_empty_docs": False},
     r"document 1 \(id 'd1'\) has no vectors"),
    ("nan-vector", lambda c: _with_vector(c, 5, 2, np.nan), {},
     r"vector 5 \(document 2, id 'd2'\) has a non-finite value, nan, at column 2"),
    ("inf-vector", lambda c: _with_vector(c, 8, 0, -np.inf), {},
     r"vector 8 \(document 3, id 'd3'\) has a non-finite value, -inf, at column 0"),
    ("nan-float16-vector", lambda c: _with_vector(c, 4, 1, np.nan, np.float16), {},
     r"vector 4 \(document 2, id 'd2'\) has a non-finite value, nan"),
    ("first-row-after-an-empty-document", lambda c: _with_vector(c, 3, 0, np.inf), {},
     r"vector 3 \(document 2, id 'd2'\)"),
    ("nan-weight", lambda c: _with_aligned(c, "weights", 6, np.nan), {},
     r"weights\[6\] \(document 2, id 'd2'\) is a non-finite value, nan"),
    ("inf-importance", lambda c: _with_aligned(c, "importance", 0, np.inf), {},
     r"importance\[0\] \(document 0, id 'd0'\) is a non-finite value, inf"),
    ("inf-position", lambda c: _with_aligned(c, "positions", 2, np.inf), {},
     r"positions\[2\] \(document 0, id 'd0'\) is an infinite value, inf"),
    ("named", lambda c: _tamper(c, ids=["a"]), {"name": "queries"},
     "^queries: 1 ids for 4 documents"),
]


@pytest.mark.parametrize(
    ("make", "kwargs", "pattern"), [case[1:] for case in _CASES], ids=[case[0] for case in _CASES]
)
def test_each_violation_raises_a_contract_error_that_names_the_offender(make, kwargs, pattern):
    with pytest.raises(ContractError, match=pattern):
        validate_corpus(make(_full_corpus()), **kwargs)


def test_contract_error_is_a_value_error():
    assert issubclass(ContractError, ValueError)


def test_the_first_violation_in_the_documented_order_is_the_one_reported():
    c = _tamper(_full_corpus(), ids=["a", "a"], vectors=np.zeros(9, np.float32))
    with pytest.raises(ContractError, match="vectors must be a 2-D"):  # vectors before ids
        validate_corpus(c)
    c = _tamper(_full_corpus(), ids=["a", "a", "b", "c"])
    c.vectors[0, 0] = np.nan
    with pytest.raises(ContractError, match="id 'a' is used"):  # ids before finiteness
        validate_corpus(c)


def test_optional_checks_can_be_switched_off():
    dup = _tamper(_full_corpus(), ids=["a", "a", "b", "c"])
    validate_corpus(dup, require_unique_ids=False)
    bad = _with_vector(_with_aligned(_full_corpus(), "weights", 1, np.inf), 2, 2, np.nan)
    validate_corpus(bad, require_finite=False)
    validate_corpus(_full_corpus(), allow_empty_docs=True)


def test_nan_positions_are_the_off_grid_marker_and_pass():
    c = _full_corpus()
    assert np.isnan(c.positions).any()
    validate_corpus(c)


def test_a_non_finite_value_after_the_first_block_is_located_globally(monkeypatch):
    monkeypatch.setattr(vectors_module, "_BLOCK_BYTES", 48)  # two rows of six float32 per block
    c = _with_vector(_full_corpus(), 7, 3, np.nan)
    with pytest.raises(ContractError, match=r"vector 7 \(document 3, id 'd3'\).*column 3"):
        validate_corpus(c)


def test_validation_scans_in_blocks_without_widening_the_array(monkeypatch):
    monkeypatch.setattr(vectors_module, "_BLOCK_BYTES", 256 * 1024)
    n, d = 400_000, 8
    big = MultiVectorCorpus(np.ones((n, d), np.float16), [0, n])  # 6.4 MB; float64 would be 25.6 MB
    tracemalloc.start()
    try:
        validate_corpus(big)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < big.nbytes / 4


# --------------------------------------------------------------- host arrays


class _Device:
    def __init__(self, kind):
        self.type = kind

    def __str__(self):
        return "cpu" if self.type == "cpu" else f"{self.type}:0"


class _FakeTensor:
    """Looks like a torch tensor to ``as_host_array``: only the module name and methods count."""

    def __init__(self, array, device="cpu", numpy_dtype_ok=True):
        self.array, self.device, self.ok, self.calls = array, _Device(device), numpy_dtype_ok, []
        self.dtype = "torch.float32" if numpy_dtype_ok else "torch.bfloat16"

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        moved = _FakeTensor(self.array, "cpu", self.ok)
        moved.calls = self.calls
        return moved

    def numpy(self):
        if not self.ok:
            raise TypeError("Got unsupported ScalarType BFloat16")
        return self.array

    def float(self):
        self.calls.append("float")
        return _FakeTensor(self.array.astype(np.float32))


_FakeTensor.__module__ = "torch"


def test_a_numpy_array_comes_back_as_the_same_object():
    for a in (np.zeros((2, 3), np.float16), np.arange(6).reshape(2, 3), np.zeros((4, 4))[:, ::2]):
        assert as_host_array(a) is a


def test_a_memory_map_is_not_loaded_or_copied(tmp_path):
    mm = np.lib.format.open_memmap(tmp_path / "m.npy", mode="w+", dtype=np.float32, shape=(5, 2))
    assert as_host_array(mm) is mm


def test_an_object_with_array_is_converted_without_changing_its_dtype():
    class Wrapper:
        def __array__(self, dtype=None, copy=None):
            return np.ones((2, 2), np.float16)

    out = as_host_array(Wrapper())
    assert isinstance(out, np.ndarray) and out.dtype == np.float16


@pytest.mark.parametrize("value", [[[1.0, 2.0]], ((1.0, 2.0),), "abc", None, 3.5, {"a": 1}])
def test_anything_else_is_a_type_error_that_asks_for_an_explicit_conversion(value):
    with pytest.raises(TypeError, match="np.asarray"):
        as_host_array(value, name="queries")


def test_the_name_is_used_in_messages():
    with pytest.raises(TypeError, match="^queries:"):
        as_host_array([1.0], name="queries")
    with pytest.raises(ContractError, match="^queries is a torch tensor"):
        as_host_array(_FakeTensor(np.zeros(2), "cuda"), name="queries")


def test_a_tensor_on_an_accelerator_is_refused_unless_the_transfer_is_allowed():
    t = _FakeTensor(np.zeros((2, 3), np.float32), device="cuda")
    with pytest.raises(ContractError, match=r"device 'cuda:0'.*allow_device_transfer=True"):
        as_host_array(t)
    assert t.calls == []  # nothing was moved before refusing
    out = as_host_array(t, allow_device_transfer=True)
    assert isinstance(out, np.ndarray) and t.calls == ["detach", "cpu"]


def test_a_cpu_tensor_needs_no_permission_and_is_not_copied():
    data = np.zeros((2, 3), np.float16)
    t = _FakeTensor(data)
    out = as_host_array(t)
    assert out is data and out.dtype == np.float16 and t.calls == ["detach"]


def test_a_dtype_numpy_cannot_hold_is_refused_unless_widening_is_allowed():
    t = _FakeTensor(np.ones((2, 2), np.float32), numpy_dtype_ok=False)
    with pytest.raises(ContractError, match=r"torch.bfloat16.*allow_dtype_widening=True"):
        as_host_array(t)
    widened = as_host_array(t, allow_dtype_widening=True)
    assert widened.dtype == np.float32 and "float" in t.calls


def test_widening_is_not_applied_to_dtypes_numpy_can_hold():
    t = _FakeTensor(np.ones((2, 2), np.float16))
    out = as_host_array(t, allow_dtype_widening=True)
    assert out.dtype == np.float16 and "float" not in t.calls


def test_a_real_cpu_tensor_is_viewed(tmp_path):
    torch = pytest.importorskip("torch")
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = as_host_array(t)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert np.shares_memory(out, t.numpy())
    t16 = torch.ones(2, 2, dtype=torch.float16, requires_grad=False)
    assert as_host_array(t16).dtype == np.float16
    grad = torch.ones(2, 2, requires_grad=True)
    assert as_host_array(grad).dtype == np.float32  # detach is not a device or dtype change
    param = torch.nn.Parameter(torch.ones(2, 2))
    assert as_host_array(param).shape == (2, 2)


def test_a_real_bfloat16_tensor_is_refused_then_widened():
    torch = pytest.importorskip("torch")
    t = torch.ones(2, 3, dtype=torch.bfloat16)
    with pytest.raises(ContractError, match="bfloat16"):
        as_host_array(t)
    out = as_host_array(t, allow_dtype_widening=True)
    assert out.dtype == np.float32 and (out == 1).all()


def test_a_real_tensor_off_the_cpu_is_refused():
    torch = pytest.importorskip("torch")
    meta = torch.empty(2, 3, device="meta")  # a non-CPU device that exists in every build
    with pytest.raises(ContractError, match="meta"):
        as_host_array(meta)


def test_the_documented_gap_in_the_pinned_converter_is_what_this_function_closes():
    """as_float_matrix widens bfloat16 silently; as_host_array refuses (see its docstring)."""
    torch = pytest.importorskip("torch")
    from optivision.representation import as_float_matrix

    t = torch.ones(2, 3, dtype=torch.bfloat16)
    assert as_float_matrix(t).dtype == np.float32
    with pytest.raises(ContractError):
        as_host_array(t)


def test_as_host_array_never_imports_torch(tmp_path):
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"  # an import of torch would raise ImportError
        "import numpy as np\n"
        "from optivision.core.vectors import as_host_array\n"
        "assert as_host_array(np.zeros((2, 2), np.float16)).dtype == np.float16\n"
    )
    done = _run(code, tmp_path)
    assert done.returncode == 0, done.stderr


# -------------------------------------------------------------- encoder identity


def _encoder(**changes):
    args = {"model_id": "org/model", "dimension": 128, "modality": "text"}
    args.update(changes)
    return EncoderIdentity(**args)


def test_encoder_identity_round_trips_through_a_dict():
    e = _encoder(revision="abc123", normalized=True, inference_dtype="bfloat16",
                 extra={"pooling": "none", "prefix": "query: "})
    assert EncoderIdentity.from_dict(e.to_dict()) == e
    assert EncoderIdentity.from_dict(json.loads(identity.canonical_json(e.to_dict()))) == e
    assert e.to_dict()["extra"] == {"pooling": "none", "prefix": "query: "}
    assert _encoder().to_dict() == {
        "model_id": "org/model", "revision": None, "dimension": 128, "modality": "text",
        "normalized": None, "inference_dtype": None, "extra": {},
    }


@pytest.mark.parametrize("modality", ["text", "image", "other"])
def test_encoder_identity_accepts_the_three_modalities(modality):
    assert _encoder(modality=modality).modality == modality


@pytest.mark.parametrize("bad", [
    {"dimension": 0}, {"dimension": -4}, {"dimension": 12.5}, {"dimension": "128"},
    {"dimension": True}, {"dimension": None}, {"modality": "audio"}, {"modality": ""},
    {"modality": None}, {"model_id": ""}, {"model_id": 3}, {"revision": 7},
    {"inference_dtype": 16}, {"normalized": "yes"}, {"extra": {"k": 1}}, {"extra": {1: "v"}},
    {"extra": ["a"]},
])
def test_encoder_identity_refuses_invalid_fields(bad):
    with pytest.raises(ContractError):
        _encoder(**bad)


def test_encoder_identity_takes_numpy_scalars_and_stores_python_ones():
    e = _encoder(dimension=np.int64(96), normalized=np.bool_(True))
    assert type(e.dimension) is int and type(e.normalized) is bool
    assert json.dumps(e.to_dict())


def test_encoder_identity_is_frozen_hashable_and_copies_extra():
    extra = {"a": "1"}
    e = _encoder(extra=extra)
    extra["a"] = "changed"
    assert e.extra == {"a": "1"}
    with pytest.raises(AttributeError):
        e.dimension = 64
    assert hash(e) == hash(_encoder(extra={"a": "1"})) and e == _encoder(extra={"a": "1"})
    assert len({e, _encoder(extra={"a": "1"})}) == 1


def test_encoder_identity_is_keyword_only():
    with pytest.raises(TypeError):
        EncoderIdentity("org/model", None, 128, "text")


def test_encoder_identity_from_dict_rejects_unknown_and_missing_fields():
    with pytest.raises(ContractError, match=r"unknown.*\['colour'\]"):
        EncoderIdentity.from_dict({**_encoder().to_dict(), "colour": "red"})
    with pytest.raises(ContractError, match=r"needs \['dimension'\]"):
        EncoderIdentity.from_dict({"model_id": "m", "modality": "text"})
    minimal = EncoderIdentity.from_dict({"model_id": "m", "dimension": 4, "modality": "other"})
    assert minimal.revision is None and minimal.extra == {}
