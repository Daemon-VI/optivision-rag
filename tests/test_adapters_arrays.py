"""Bring-your-own-vectors adapters: what is accepted, what is refused, what is kept as given."""

from __future__ import annotations

import numpy as np
import pytest

from optivision.adapters import (
    MEASURED,
    ArrayAdapter,
    CallableAdapter,
    EncoderAdapter,
    corpus_from_arrays,
)
from optivision.core.vectors import (
    ContractError,
    EncoderIdentity,
    MultiVectorCorpus,
    corpus_identity,
)

DIM = 6


def _arrays(counts=(3, 1, 4), dim=DIM, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, dim)).astype(dtype) for n in counts]


def _identity(**changes):
    args = {"model_id": "org/some-model", "dimension": DIM, "modality": "text"}
    args.update(changes)
    return EncoderIdentity(**args)


class _FakeTensor:
    """Looks like a torch tensor to ``as_host_array``; records what was done to it."""

    class _Device:
        def __init__(self, kind):
            self.type = kind

        def __str__(self):
            return "cpu" if self.type == "cpu" else f"{self.type}:0"

    def __init__(self, array, device="cpu"):
        self.array, self.device, self.calls = array, self._Device(device), []
        self.dtype = "torch.float32"

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        moved = _FakeTensor(self.array)
        moved.calls = self.calls
        return moved

    def numpy(self):
        return self.array


_FakeTensor.__module__ = "torch"


# ------------------------------------------------------------------ ArrayAdapter


def test_it_is_an_encoder_adapter_and_returns_a_corpus():
    adapter = ArrayAdapter()
    assert isinstance(adapter, EncoderAdapter)
    corpus = adapter.encode_documents(_arrays())
    assert isinstance(corpus, MultiVectorCorpus)
    assert corpus.counts.tolist() == [3, 1, 4] and corpus.dimension == DIM
    assert corpus.ids == ["0", "1", "2"]
    queries = adapter.encode_queries(_arrays((2, 2)), ids=["a", "b"])
    assert queries.ids == ["a", "b"] and queries.counts.tolist() == [2, 2]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_the_dtype_is_kept_and_the_values_are_not_rounded(dtype):
    arrays = _arrays(dtype=dtype)
    corpus = ArrayAdapter().encode_documents(arrays)
    assert corpus.dtype == dtype
    assert np.array_equal(corpus.vectors, np.concatenate(arrays))


def test_the_library_would_promote_mixed_dtypes_and_the_adapter_refuses_to():
    """MultiVectorCorpus.from_arrays promotes to a common dtype; the adapter says no instead."""
    f32 = _arrays((2,))[0]
    f16 = _arrays((2,), dtype=np.float16)[0]
    f64 = _arrays((2,), dtype=np.float64)[0]
    assert MultiVectorCorpus.from_arrays([f32, f16]).dtype == np.float32
    assert MultiVectorCorpus.from_arrays([f32, f64]).dtype == np.float64
    message = r"documents\[1\] has dtype float16 but documents\[0\] has float32"
    with pytest.raises(ContractError, match=message):
        ArrayAdapter().encode_documents([f32, f16])
    with pytest.raises(ContractError, match="promoted to one common dtype"):
        ArrayAdapter().encode_queries([f32, f64])


def test_the_inputs_are_not_modified_and_the_corpus_does_not_share_their_memory():
    arrays = _arrays()
    copies = [a.copy() for a in arrays]
    corpus = ArrayAdapter().encode_documents(arrays)
    assert all(np.array_equal(a, c) for a, c in zip(arrays, copies, strict=True))
    corpus.vectors[:] = 0.0
    assert all(np.array_equal(a, c) for a, c in zip(arrays, copies, strict=True))
    assert not any(np.shares_memory(corpus.vectors, a) for a in arrays)


def test_the_output_is_deterministic():
    arrays = _arrays()
    a = corpus_identity(ArrayAdapter().encode_documents(arrays))
    b = corpus_identity(ArrayAdapter().encode_documents(arrays))
    assert a == b


def test_empty_documents_are_kept_as_empty_documents():
    arrays = [np.zeros((0, DIM), np.float32), *_arrays((2,))]
    corpus = ArrayAdapter().encode_documents(arrays)
    assert corpus.counts.tolist() == [0, 2]


# ----------------------------------------------------------------------- identity


def test_without_an_identity_nothing_is_claimed():
    adapter = ArrayAdapter()
    assert adapter.identity() is None and adapter.dimension is None
    assert "encoder_identity" not in adapter.encode_documents(_arrays()).attrs


def test_the_identity_is_recorded_as_a_dict_and_checked_against_the_dimension():
    identity = _identity(revision="abc123", normalized=False, extra={"pooling": "none"})
    adapter = ArrayAdapter(identity)
    assert adapter.identity() is identity and adapter.dimension == DIM
    corpus = adapter.encode_documents(_arrays())
    assert corpus.attrs["encoder_identity"] == identity.to_dict()
    assert EncoderIdentity.from_dict(corpus.attrs["encoder_identity"]) == identity
    assert corpus.attrs["model"] == "org/some-model" and corpus.attrs["loader"] == "arrays"
    with pytest.raises(ContractError, match=r"documents\[0\] has dimension 9 but the identity of "
                                            r"'org/some-model' says 6"):
        adapter.encode_documents(_arrays(dim=9))


def test_the_identity_survives_a_save_and_a_load(tmp_path):
    identity = _identity(inference_dtype="float32")
    corpus = ArrayAdapter(identity).encode_documents(_arrays())
    path = corpus.save(tmp_path / "docs.npz")
    assert MultiVectorCorpus.load(path).attrs["encoder_identity"] == identity.to_dict()


def test_a_registered_model_name_in_the_identity_never_earns_a_registry_status():
    """The adapter looks nothing up: that the arrays came from a model is the caller's word."""
    from optivision.adapters import MODELS

    measured = next(m for m in MODELS.values() if m.status == MEASURED)
    identity = _identity(model_id=measured.model_id, dimension=measured.dimension or DIM)
    corpus = ArrayAdapter(identity).encode_documents(_arrays(dim=identity.dimension))
    assert "model_status" not in corpus.attrs


def test_a_non_identity_is_refused():
    with pytest.raises(TypeError, match="EncoderIdentity"):
        ArrayAdapter({"model_id": "x"})
    with pytest.raises(TypeError, match="EncoderIdentity"):
        CallableAdapter(lambda x: x, lambda x: x, identity="x")


# --------------------------------------------------------------------- refusals


def test_dimensions_must_agree_across_items():
    message = r"documents\[1\] has dimension 4 but documents\[0\] has 6"
    with pytest.raises(ContractError, match=message):
        ArrayAdapter().encode_documents([*_arrays((2,)), *_arrays((2,), dim=4)])


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint8, bool])
def test_integer_and_boolean_vectors_are_refused(dtype):
    codes = np.ones((2, DIM), dtype)
    with pytest.raises(ContractError, match="must be floating point"):
        ArrayAdapter().encode_documents([codes])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_refused_and_located(bad):
    arrays = _arrays()
    arrays[2][1, 3] = bad
    with pytest.raises(ContractError, match=r"documents: vector 5 \(document 2, id '2'\) has a "
                                            r"non-finite value.*column 3"):
        ArrayAdapter().encode_documents(arrays)


def test_items_must_be_two_dimensional():
    with pytest.raises(ContractError, match=r"documents\[1\] must be 2-D .* got shape \(4,\)"):
        ArrayAdapter().encode_documents([*_arrays((2,)), np.zeros(4, np.float32)])
    with pytest.raises(ContractError, match=r"queries\[0\] must be 2-D .* got shape \(2, 3, 6\)"):
        ArrayAdapter().encode_queries([np.zeros((2, 3, DIM), np.float32)])


def test_a_zero_width_array_is_refused():
    with pytest.raises(ContractError, match="dimension must be at least 1"):
        ArrayAdapter().encode_documents([np.zeros((2, 0), np.float32)])


def test_a_3d_array_is_refused_because_a_padded_batch_looks_like_one():
    batch = np.ones((4, 7, DIM), np.float32)
    with pytest.raises(ContractError, match="expected a sequence of") as caught:
        ArrayAdapter().encode_documents(batch)
    assert "padded batch" in str(caught.value) and "padding rows" in str(caught.value)
    # the caller who knows no row is padding says so by passing a list
    assert ArrayAdapter().encode_documents(list(batch)).counts.tolist() == [7, 7, 7, 7]


def test_a_2d_array_is_refused_with_a_hint():
    with pytest.raises(ContractError, match="pass a list with one"):
        ArrayAdapter().encode_documents(np.ones((4, DIM), np.float32))


@pytest.mark.parametrize("junk", ["abc", b"abc", {"a": 1}])
def test_a_string_or_mapping_is_not_a_list_of_arrays(junk):
    with pytest.raises(ContractError, match="expected a sequence of"):
        ArrayAdapter().encode_queries(junk)


def test_a_list_of_lists_asks_for_an_explicit_conversion():
    with pytest.raises(TypeError, match="np.asarray"):
        ArrayAdapter().encode_documents([[[0.1, 0.2], [0.3, 0.4]]])


def test_nothing_to_convert_is_an_error():
    with pytest.raises(ContractError, match="no items"):
        ArrayAdapter().encode_documents([])


def test_ids_are_checked():
    arrays = _arrays()
    with pytest.raises(ContractError, match="2 ids for 3 items"):
        ArrayAdapter().encode_documents(arrays, ids=["a", "b"])
    with pytest.raises(TypeError, match="not one string"):
        ArrayAdapter().encode_documents(arrays, ids="abc")
    with pytest.raises(ContractError, match="id 'a' is used by documents 0 and 2"):
        ArrayAdapter().encode_documents(arrays, ids=["a", "b", "a"])


# --------------------------------------------------------------- tensors and devices


def test_a_cpu_tensor_is_taken_without_a_transfer():
    data = _arrays((3,))[0]
    tensor = _FakeTensor(data)
    corpus = ArrayAdapter().encode_documents([tensor])
    assert tensor.calls == ["detach"] and np.array_equal(corpus.vectors, data)


def test_a_tensor_on_an_accelerator_is_not_moved_unless_the_adapter_was_told_to():
    tensor = _FakeTensor(_arrays((3,))[0], device="cuda")
    with pytest.raises(ContractError, match=r"documents\[0\] is a torch tensor on device 'cuda:0'"):
        ArrayAdapter().encode_documents([tensor])
    assert tensor.calls == []  # refused before anything was moved
    corpus = ArrayAdapter(allow_device_transfer=True).encode_documents([tensor])
    assert tensor.calls == ["detach", "cpu"] and corpus.counts.tolist() == [3]


def test_real_torch_tensors_follow_the_same_rules():
    torch = pytest.importorskip("torch")
    cpu = torch.ones(2, DIM, dtype=torch.float16)
    corpus = ArrayAdapter().encode_documents([cpu, cpu])
    assert corpus.dtype == np.float16  # kept, not widened
    with pytest.raises(ContractError, match="torch dtype"):
        ArrayAdapter().encode_documents([torch.ones(2, DIM, dtype=torch.bfloat16)])  # not widened
    with pytest.raises(ContractError, match="device 'meta'"):
        ArrayAdapter().encode_documents([torch.empty(2, DIM, device="meta")])
    with pytest.raises(ContractError, match="padded batch"):
        ArrayAdapter().encode_documents(torch.ones(2, 3, DIM))


# ----------------------------------------------------------------- CallableAdapter


def test_a_callable_adapter_runs_the_functions_and_checks_what_they_return():
    seen = {}

    def docs(items):
        seen["documents"] = items
        return [np.full((len(str(x)), DIM), 0.5, np.float32) for x in items]

    def queries(items):
        seen["queries"] = items
        return [np.ones((2, DIM), np.float32) for _ in items]

    adapter = CallableAdapter(docs, queries, _identity())
    out = adapter.encode_documents(iter(["a", "bbb"]), ids=["x", "y"])  # a generator is accepted
    assert seen["documents"] == ["a", "bbb"] and isinstance(seen["documents"], list)
    assert out.ids == ["x", "y"] and out.counts.tolist() == [1, 3]
    assert out.attrs["loader"] == "callable"
    assert out.attrs["encoder_identity"] == _identity().to_dict()
    assert adapter.encode_queries(["q"]).counts.tolist() == [2]
    assert adapter.identity() == _identity() and adapter.dimension == DIM


def test_the_callable_adapter_applies_the_same_checks_as_the_array_adapter():
    def good(items):
        return [np.ones((2, DIM), np.float32) for _ in items]

    def wrong_count(items):
        return [np.ones((2, DIM), np.float32)]

    def wrong_dim(items):
        return [np.ones((2, 3), np.float32) for _ in items]

    def mixed(items):
        return [np.ones((2, DIM), t) for t in (np.float32, np.float16)][: len(items)]

    def not_finite(items):
        return [np.full((2, DIM), np.nan, np.float32) for _ in items]

    def a_corpus(items):
        return MultiVectorCorpus.from_arrays([np.ones((2, DIM), np.float32) for _ in items])

    def padded(items):
        return np.zeros((len(items), 5, DIM), np.float32)

    with pytest.raises(ContractError, match="returned 1 arrays for 2 inputs"):
        CallableAdapter(wrong_count, good).encode_documents(["a", "b"])
    with pytest.raises(ContractError, match="identity of 'org/some-model' says 6"):
        CallableAdapter(wrong_dim, good, _identity()).encode_documents(["a"])
    with pytest.raises(ContractError, match="promoted to one common dtype"):
        CallableAdapter(mixed, good).encode_documents(["a", "b"])
    with pytest.raises(ContractError, match="non-finite"):
        CallableAdapter(not_finite, good).encode_documents(["a"])
    with pytest.raises(ContractError, match="returned a MultiVectorCorpus"):
        CallableAdapter(a_corpus, good).encode_documents(["a"])
    with pytest.raises(ContractError, match="padded batch"):
        CallableAdapter(padded, good).encode_documents(["a", "b"])


def test_the_callable_adapter_refuses_what_is_not_callable():
    with pytest.raises(TypeError, match="encode_queries must be callable"):
        CallableAdapter(lambda x: x, "not a function")


def test_the_functions_may_return_tensors_on_the_host_only():
    tensor = _FakeTensor(np.ones((2, DIM), np.float32), device="cuda")
    adapter = CallableAdapter(lambda items: [tensor], lambda items: [tensor])
    with pytest.raises(ContractError, match="device 'cuda:0'"):
        adapter.encode_documents(["a"])
    allowed = CallableAdapter(lambda items: [tensor], lambda items: [tensor],
                              allow_device_transfer=True)
    assert allowed.encode_queries(["a"]).counts.tolist() == [2]


# ------------------------------------------------------- the shared conversion itself


def test_corpus_from_arrays_is_the_one_place_that_checks_and_takes_metadata_and_attrs():
    corpus = corpus_from_arrays(_arrays((2, 3)), ["a", "b"], name="pages",
                                metadata=[{"n": 2}, {"n": 3}], attrs={"source": "test"})
    assert corpus.metadata == [{"n": 2}, {"n": 3}] and corpus.attrs == {"source": "test"}
    with pytest.raises(ContractError, match="pages: 1 metadata entries for 2 items"):
        corpus_from_arrays(_arrays((2, 3)), name="pages", metadata=[{}])
    with pytest.raises(ContractError, match="^pages\\[0\\] must be 2-D"):
        corpus_from_arrays([np.zeros(3, np.float32)], name="pages")


def test_fortran_ordered_items_give_the_same_memory_layout_as_c_ordered_ones():
    # Same values and the same array_digest, but a Fortran-ordered corpus makes float reductions
    # differ in the last bits, which changed the norms and so the certificate digest.
    c_items = _arrays()
    f_items = [np.asfortranarray(a) for a in c_items]
    assert not all(a.flags.c_contiguous for a in f_items)
    c_corpus = ArrayAdapter().encode_documents(c_items)
    f_corpus = ArrayAdapter().encode_documents(f_items)
    assert f_corpus.vectors.flags.c_contiguous
    assert np.array_equal(f_corpus.vectors, c_corpus.vectors)
