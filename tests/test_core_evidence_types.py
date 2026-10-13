"""The Evidence type: structure checks, immutability, digest, subset, npz round trip."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from optivision.core import evaluate
from optivision.core.evaluate import Evidence, EvidenceFormatError
from optivision.core.identity import canonical_json

K, N = 3, 6
SPACE = "a" * 64


def make(**over) -> Evidence:
    rng = np.random.default_rng(0)
    base = {
        "metric": "ndcg@5",
        "construct": "labels",
        "baseline": 0.2 + 0.8 * rng.random(N),
        "candidates": rng.random((K, N)),
        "query_ids": tuple(f"q{i}" for i in range(N)),
        "bytes_per_doc": np.array([300.0, 200.0, 100.0]),
        "vectors_per_doc": np.array([10.0, 10.0, 5.0]),
        "space_identity": SPACE,
        "dropped_queries": ("d0", "d1"),
        "fitted": ((None,), ("ab" * 32, None), ("cd" * 32,)),
    }
    base.update(over)
    return Evidence(**base)


# ------------------------------------------------------------------------ structure


def test_a_valid_evidence_exposes_its_sizes_and_a_short_repr():
    ev = make()
    assert (ev.n_queries, ev.n_candidates) == (N, K)
    assert ev.baseline.shape == (N,) and ev.candidates.shape == (K, N)
    text = repr(ev)
    assert "ndcg@5" in text and "labels" in text and ev.digest()[:12] in text
    assert len(text) < 200  # never the arrays


def test_dropped_queries_and_fitted_default_to_empty():
    ev = Evidence("ndcg@5", "labels", np.ones(2), np.ones((1, 2)), ("a", "b"), np.ones(1),
                  np.ones(1), SPACE)
    assert ev.dropped_queries == () and ev.fitted == ()


def test_arrays_are_private_read_only_copies():
    base = np.linspace(0.1, 0.9, N)
    cand = np.full((K, N), 0.5)
    sizes, vecs = np.array([3.0, 2.0, 1.0]), np.array([2.0, 2.0, 1.0])
    ev = make(baseline=base, candidates=cand, bytes_per_doc=sizes, vectors_per_doc=vecs)
    for stored, given in ((ev.baseline, base), (ev.candidates, cand), (ev.bytes_per_doc, sizes),
                          (ev.vectors_per_doc, vecs)):
        assert not stored.flags.writeable and stored.dtype == np.float64
        assert not np.shares_memory(stored, given)
        with pytest.raises(ValueError, match="read-only"):
            stored.flat[0] = 0.0
    before = ev.digest()
    base[:] = 0.0
    cand[:] = 0.0
    assert ev.digest() == before  # later changes to the caller's arrays have no effect


def test_the_evidence_is_frozen():
    ev = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.metric = "ndcg@10"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.baseline = np.ones(N)  # type: ignore[misc]


def test_a_list_of_floats_is_accepted_and_becomes_a_float64_array():
    ev = make(baseline=[0.5] * N)
    assert ev.baseline.dtype == np.float64


@pytest.mark.parametrize("name", ["baseline", "candidates", "bytes_per_doc", "vectors_per_doc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int64, np.bool_, ">f8"])
def test_only_native_float64_is_accepted_there_is_no_silent_conversion(name, dtype):
    given = np.asarray(getattr(make(), name)).astype(dtype)
    with pytest.raises(EvidenceFormatError, match="float64"):
        make(**{name: given})


@pytest.mark.parametrize(("over", "message"), [
    ({"baseline": np.ones((N, 1))}, "dimension"),
    ({"baseline": np.ones(N + 1)}, "baseline must have shape"),
    ({"candidates": np.ones(N)}, "dimension"),
    ({"candidates": np.ones((K, N + 1))}, "baseline must have shape"),
    ({"candidates": np.ones((0, N)), "bytes_per_doc": np.ones(0),
      "vectors_per_doc": np.ones(0), "fitted": ()}, "at least one candidate"),
    ({"query_ids": tuple("abcde")}, "query_ids has 5 entries"),
    ({"bytes_per_doc": np.ones(K + 1)}, "bytes_per_doc must have shape"),
    ({"vectors_per_doc": np.ones(K - 1)}, "vectors_per_doc must have shape"),
    ({"fitted": ((None,), (None,))}, "fitted has 2 entries"),
    ({"fitted": "abc"}, "fitted"),
    ({"fitted": ({"x": None}, (None,), (None,))}, "fitted"),
    ({"fitted": ((1,), (None,), (None,))}, "str | None"),
    ({"query_ids": "abcdef"}, "sequence of strings"),
    ({"query_ids": (1, 2, 3, 4, 5, 6)}, "strings"),
    ({"dropped_queries": (1,)}, "strings"),
    ({"dropped_queries": "d0"}, "sequence of strings"),
    ({"metric": ""}, "metric"),
    ({"construct": None}, "construct"),
    ({"space_identity": ""}, "space_identity"),
])
def test_structure_errors_raise_a_value_error(over, message):
    with pytest.raises(EvidenceFormatError, match=message):
        make(**over)
    assert issubclass(EvidenceFormatError, ValueError)


def test_zero_queries_are_representable_but_zero_candidates_are_not():
    ev = make(baseline=np.ones(0), candidates=np.ones((K, 0)), query_ids=())
    assert ev.n_queries == 0 and ev.n_candidates == K and len(ev.digest()) == 64


def test_values_are_checked_when_a_rule_runs_not_at_construction():
    ev = make(candidates=np.full((K, N), np.nan))
    assert np.isnan(ev.candidates).all() and len(ev.digest()) == 64


# --------------------------------------------------------------------------- digest


def test_the_digest_is_deterministic_and_equal_for_equal_content():
    a, b = make(), make()
    assert a is not b and a.digest() == b.digest() and len(a.digest()) == 64
    assert a == b and hash(a) == hash(b) and a != "x"


def test_the_digest_is_cached_but_never_stale():
    ev = make()
    before = ev.digest()
    assert ev.digest() == before
    assert dataclasses.replace(ev, metric="ndcg@10").digest() != before
    assert ev.digest() == before  # the original is untouched


def perturbed_variants(ev: Evidence) -> dict[str, Evidence]:
    def bumped(arr, index):
        out = np.array(arr, copy=True)
        out.flat[index] = np.nextafter(out.flat[index], 2.0)  # one ulp
        return out

    swapped_ids = list(ev.query_ids)
    swapped_ids[0], swapped_ids[1] = swapped_ids[1], swapped_ids[0]
    return {
        "metric": dataclasses.replace(ev, metric="ndcg@10"),
        "construct": dataclasses.replace(ev, construct="agreement@1"),
        "baseline": dataclasses.replace(ev, baseline=bumped(ev.baseline, 3)),
        "candidates": dataclasses.replace(ev, candidates=bumped(ev.candidates, 7)),
        "bytes_per_doc": dataclasses.replace(ev, bytes_per_doc=bumped(ev.bytes_per_doc, 1)),
        "vectors_per_doc": dataclasses.replace(ev, vectors_per_doc=bumped(ev.vectors_per_doc, 2)),
        "query_ids rename": dataclasses.replace(
            ev, query_ids=(*ev.query_ids[:-1], "renamed")),
        "query_ids order": dataclasses.replace(ev, query_ids=tuple(swapped_ids)),
        "space_identity": dataclasses.replace(ev, space_identity="b" * 64),
        "dropped add": dataclasses.replace(ev, dropped_queries=(*ev.dropped_queries, "d2")),
        "dropped remove": dataclasses.replace(ev, dropped_queries=ev.dropped_queries[:1]),
        "dropped order": dataclasses.replace(ev, dropped_queries=ev.dropped_queries[::-1]),
        "fitted digest": dataclasses.replace(
            ev, fitted=(*ev.fitted[:2], ("ef" * 32,))),
        "fitted none": dataclasses.replace(ev, fitted=((None,), (None, None), ("cd" * 32,))),
        "fitted empty": dataclasses.replace(ev, fitted=()),
    }


def test_the_digest_is_sensitive_to_every_field_and_to_one_ulp():
    ev = make()
    variants = perturbed_variants(ev)
    digests = {name: v.digest() for name, v in variants.items()}
    assert all(d != ev.digest() for d in digests.values()), digests
    assert len(set(digests.values())) == len(digests)  # and they differ from one another
    assert all(v != ev for v in variants.values())


def test_the_digest_distinguishes_a_swap_of_two_queries_from_the_original():
    ev = make()
    swapped = ev.subset(np.array([1, 0, 2, 3, 4, 5]))
    assert swapped.digest() != ev.digest()
    assert swapped.subset(np.array([1, 0, 2, 3, 4, 5])).digest() == ev.digest()


# --------------------------------------------------------------------------- subset


def test_subset_reorders_and_takes_prefixes_with_everything_aligned():
    ev = make()
    order = np.array([4, 0, 5, 1, 3, 2])
    reordered = ev.subset(order)
    assert np.array_equal(reordered.baseline, ev.baseline[order])
    assert np.array_equal(reordered.candidates, ev.candidates[:, order])
    assert reordered.query_ids == tuple(ev.query_ids[i] for i in order)
    prefix = reordered.subset(np.arange(3))
    assert prefix.n_queries == 3 and prefix.query_ids == reordered.query_ids[:3]
    assert np.array_equal(prefix.candidates, ev.candidates[:, order[:3]])
    for kept in ("metric", "construct", "bytes_per_doc", "vectors_per_doc", "space_identity",
                 "dropped_queries", "fitted"):
        a, b = getattr(ev, kept), getattr(prefix, kept)
        assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b


def test_subset_of_everything_in_order_is_the_same_evidence():
    ev = make()
    assert ev.subset(np.arange(N)) == ev
    assert ev.subset(list(range(N))).digest() == ev.digest()


def test_subset_allows_empty_and_repeated_positions():
    ev = make()
    assert ev.subset([]).n_queries == 0
    again = ev.subset([2, 2, 2, 0])
    assert again.query_ids == ("q2", "q2", "q2", "q0")
    assert np.array_equal(again.baseline, ev.baseline[[2, 2, 2, 0]])


@pytest.mark.parametrize("bad", [[-1], [N], [0, 1, 99], np.ones((2, 2), dtype=np.int64),
                                 np.array([0.0, 1.0]), "ab"])
def test_subset_refuses_bad_indices_instead_of_wrapping(bad):
    with pytest.raises(EvidenceFormatError):
        make().subset(bad)


def test_a_subset_is_itself_immutable_and_independent():
    ev = make()
    sub = ev.subset([0, 1])
    assert not sub.candidates.flags.writeable and not np.shares_memory(sub.candidates,
                                                                      ev.candidates)


# ----------------------------------------------------------------------------- npz


def test_npz_round_trip_preserves_every_field_and_the_digest(tmp_path):
    ev = make(query_ids=("q0", "é", "q2", "日本", "q4", "q\n5"))
    path = ev.to_npz(tmp_path / "evidence.npz")
    assert path == tmp_path / "evidence.npz" and path.is_file()
    back = Evidence.from_npz(path)
    assert back == ev and back.digest() == ev.digest()
    assert back.query_ids == ev.query_ids and back.fitted == ev.fitted
    assert back.dropped_queries == ev.dropped_queries
    for name in ("baseline", "candidates", "bytes_per_doc", "vectors_per_doc"):
        assert np.array_equal(getattr(back, name), getattr(ev, name))
        assert getattr(back, name).dtype == np.float64
    assert back.metric == ev.metric and back.construct == ev.construct
    assert back.space_identity == ev.space_identity


def test_npz_adds_the_suffix_like_the_other_formats_and_makes_directories(tmp_path):
    ev = make()
    path = ev.to_npz(tmp_path / "deep" / "dir" / "evidence")
    assert path.name == "evidence.npz" and Evidence.from_npz(path) == ev


def test_npz_holds_plain_arrays_and_a_canonical_json_header_and_no_pickle(tmp_path):
    ev = make()
    path = ev.to_npz(tmp_path / "e.npz")
    with np.load(path, allow_pickle=False) as data:  # raises if any array needs pickle
        assert sorted(data.files) == ["baseline", "bytes_per_doc", "candidates", "header",
                                      "vectors_per_doc"]
        header_bytes = bytes(data["header"])
        assert data["header"].dtype == np.uint8
    header = json.loads(header_bytes.decode("ascii"))
    assert header_bytes == canonical_json(header)
    assert header["format"] == "optivision-evidence" and header["digest"] == ev.digest()
    assert header["fitted"][0] == [None]


def test_npz_round_trips_empty_fitted_and_zero_queries(tmp_path):
    ev = make(fitted=(), baseline=np.ones(0), candidates=np.ones((K, 0)), query_ids=())
    assert Evidence.from_npz(ev.to_npz(tmp_path / "z.npz")) == ev


def test_npz_round_trips_nan_values_without_touching_the_header(tmp_path):
    ev = make(candidates=np.full((K, N), np.nan))
    back = Evidence.from_npz(ev.to_npz(tmp_path / "nan.npz"))
    assert np.isnan(back.candidates).all() and back == ev


def rewrite(path, **changes):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays.update(changes)
    np.savez(path, **arrays)


def test_a_tampered_file_is_refused_by_the_digest_check(tmp_path):
    path = make().to_npz(tmp_path / "e.npz")
    with np.load(path, allow_pickle=False) as data:
        altered = np.array(data["candidates"])
    altered[0, 0] += 1e-9
    rewrite(path, candidates=altered)
    with pytest.raises(EvidenceFormatError, match="digest"):
        Evidence.from_npz(path)


@pytest.mark.parametrize("edit", [{"format": "other"}, {"version": 2}])
def test_a_foreign_format_or_version_is_refused(tmp_path, edit):
    path = make().to_npz(tmp_path / "e.npz")
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(bytes(data["header"]).decode("utf-8"))
    header.update(edit)
    rewrite(path, header=np.frombuffer(canonical_json(header), dtype=np.uint8))
    with pytest.raises(EvidenceFormatError, match="format|version"):
        Evidence.from_npz(path)


def test_a_file_without_the_expected_arrays_is_refused(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, vectors=np.ones((2, 2)))
    with pytest.raises(EvidenceFormatError, match="missing"):
        Evidence.from_npz(path)


def test_an_object_array_file_cannot_be_loaded_because_pickle_is_never_enabled(tmp_path):
    path = tmp_path / "pickled.npz"
    np.savez(path, baseline=np.array([None, 1], dtype=object))
    with pytest.raises(ValueError):
        Evidence.from_npz(path)


# ----------------------------------------------------------------------- the package


def test_the_evaluate_package_exports_and_resolves_unknown_names_as_missing():
    for name in ("Evidence", "LabelReference", "RankingReference", "metrics_from_ranking",
                 "per_query_metrics", "parse_metric", "mean_metrics"):
        assert hasattr(evaluate, name) and name in evaluate.__all__
    assert "collect_evidence" not in evaluate.__all__  # exported lazily from collect.py
    with pytest.raises(AttributeError):
        getattr(evaluate, "no_such_name")  # noqa: B009


# ------------------------------------------------------ copies and pickling (in process)


def test_evidence_survives_copy_deepcopy_and_pickle_as_the_same_immutable_value():
    import copy
    import pickle

    ev = make()
    assert copy.copy(ev) is ev and copy.deepcopy(ev) is ev
    back = pickle.loads(pickle.dumps(ev))
    assert back == ev and not back.candidates.flags.writeable
    assert not np.shares_memory(back.candidates, ev.candidates)
