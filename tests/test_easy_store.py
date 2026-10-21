"""The ``.ovx`` store (optivision.easy._store): exact round trips for every ladder layout,
segment settling, crash safety, the writer lock, removal, compaction and Windows deletion
(FINAL_DESIGN 9, "Exactness"). Synthetic corpora only; nothing here measures a model."""

from __future__ import annotations

import functools
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

import optivision
from optivision.compose import Pipeline
from optivision.core.retrieve.exact import CompressedRetriever
from optivision.core.stages.state import pipeline_fitted_identity
from optivision.easy import _store
from optivision.easy._index import SEARCH_BLOCK_BYTES, Searcher
from optivision.easy._store import (
    LOCK_NAME,
    Store,
    clear_stale_lock,
    lock_holder,
    settle,
    settle_many,
    split,
    writer_lock,
)
from optivision.easy._types import IndexLocked
from optivision.representation import MultiVectorCorpus
from optivision.testing.synthetic import make_corpus, make_queries

SRC = Path(optivision.__file__).resolve().parents[1]

#: the ladder of FINAL_DESIGN 5.1.1: float rows, then each chain's rungs
LADDER = (
    "float16",
    "float32",
    "int8(per_vector)",
    "int4(mean)",
    "binary",
    "hierarchical_merge(0.33) > int8(per_vector)",
    "hierarchical_merge(0.33) > int4(mean)",
    "hierarchical_merge(0.33) > binary",
    "hierarchical_merge(0.25) > int8(per_vector)",
    "hierarchical_merge(0.25) > int4(mean)",
    "hierarchical_merge(0.25) > binary",
)

DATA = {"n_docs": 150, "dim": 24, "tokens": (3, 10), "seed": 11, "anisotropy": 1.0, "mean_shift": 0.5}


@functools.lru_cache(maxsize=1)
def _catalogue() -> dict[str, dict]:
    """label -> pipeline dict, as the measured catalogue rows give them (never parse_pipeline)."""
    data = json.loads(resources.files("optivision").joinpath("presets", "measured.json").read_text(encoding="utf-8"))
    found: dict[str, dict] = {}

    def walk(o: object) -> None:
        if isinstance(o, dict):
            if "pipeline" in o and "label" in o:
                found.setdefault(o["label"], o["pipeline"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    return found


def _spec(label: str) -> dict:
    return json.loads(json.dumps(_catalogue()[label]))


@functools.lru_cache(maxsize=4)
def _data(n_docs: int = DATA["n_docs"]) -> tuple[MultiVectorCorpus, MultiVectorCorpus]:
    kw = dict(DATA, n_docs=n_docs)
    corpus = make_corpus(kw.pop("n_docs"), kw.pop("dim"), dtype=np.float16, **kw)
    queries, _ = make_queries(corpus, 12, tokens=(2, 6), noise=0.3, seed=12)
    return corpus, queries


def _fitted(label: str, corpus: MultiVectorCorpus, seed: int = 0) -> Pipeline:
    """Fit on a seeded page sample, as the build does, then compress with fit=False."""
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(corpus), size=len(corpus) // 2, replace=False))
    return Pipeline.from_dict(_spec(label)).fit(corpus.select(sample))


def _texts(seg: MultiVectorCorpus) -> dict[str, str]:
    return {pid: f"text of {pid} é" for pid in seg.ids[::3]}


def _build(path: Path, label: str, *, pages: int = 40, float16: bool = True, texts: bool = True,
           corpus: MultiVectorCorpus | None = None) -> Pipeline:
    corpus = corpus if corpus is not None else _data()[0]
    p = _fitted(label, corpus)
    with Store.create(path, pipeline=p, info={"layout": label}) as store:
        for seg, codes in zip(split(corpus, pages), settle(split(corpus, pages), p), strict=True):
            store.append(codes, texts=_texts(seg) if texts else None, float16=seg if float16 else None,
                         vectors_before=seg.counts)
        store.commit()
    return p


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
    return env


def _dead_pid() -> int:
    """The pid of a process that has exited (the interpreter's own pid, which a venv launcher
    on Windows does not share)."""
    out = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True,
                         text=True, env=_env(), timeout=120, check=True)
    return int(out.stdout.strip())


# ----------------------------------------------------------------------------- catalogue rungs


def test_every_ladder_rung_comes_from_a_catalogue_row_and_rebuilds():
    for label in LADDER:
        spec = _spec(label)
        assert Pipeline.from_dict(spec).to_dict() == spec, label


# ----------------------------------------------------------------------------- settling


@pytest.mark.parametrize("label", LADDER)
def test_settled_codes_equal_the_pipeline_output_and_the_whole_corpus(label):
    corpus, _ = _data()
    p = _fitted(label, corpus)
    whole = p.compress(corpus, fit=False)
    settled = list(settle(corpus, p, segment_pages=40))  # one corpus, cut into views
    assert [len(c) for c in settled] == [40, 40, 40, 30]
    for seg, codes in zip(split(corpus, 40), settled, strict=True):
        alone = p.compress(corpus.select([corpus.ids.index(i) for i in seg.ids]), fit=False)
        np.testing.assert_array_equal(codes.codes, alone.codes)
        np.testing.assert_array_equal(codes.offsets, alone.offsets)
        assert codes.ids == alone.ids and codes.stats == alone.stats and codes.pipeline == p.to_dict()
    np.testing.assert_array_equal(np.concatenate([c.codes for c in settled]), whole.codes)
    counts = np.concatenate([c.counts for c in settled])
    np.testing.assert_array_equal(counts, whole.counts)
    assert [pid for c in settled for pid in c.ids] == whole.ids
    assert sum(c.stats["vectors_before"] for c in settled) == whole.stats["vectors_before"]


def test_settle_refuses_an_unfitted_pipeline():
    corpus, _ = _data()
    with pytest.raises(RuntimeError, match="unfitted"):
        next(settle(corpus, Pipeline.from_dict(_spec("int4(mean)"))))


def test_settle_many_transforms_each_merge_depth_once_and_matches_compress(monkeypatch):
    corpus, _ = _data()
    labels = ["hierarchical_merge(0.25) > int8(per_vector)", "hierarchical_merge(0.25) > int4(mean)",
              "hierarchical_merge(0.25) > binary", "int8(per_vector)", "binary"]
    pipelines = [_fitted(label, corpus) for label in labels]
    calls = []
    original = Pipeline.transform

    def counting(self, x):
        calls.append(self.label())
        return original(self, x)

    monkeypatch.setattr(Pipeline, "transform", counting)
    out = list(settle_many(corpus, pipelines, segment_pages=50))
    monkeypatch.setattr(Pipeline, "transform", original)
    assert len(out) == 3 and len(calls) == 3 * 2  # one merged transform and one plain one per segment
    for seg, codes in zip(split(corpus, 50), out, strict=True):
        for p, c in zip(pipelines, codes, strict=True):
            ref = p.compress(seg, fit=False)
            np.testing.assert_array_equal(c.codes, ref.codes)
            np.testing.assert_array_equal(c.offsets, ref.offsets)
            assert c.stats == ref.stats and c.pipeline == ref.pipeline


def test_split_gives_views_of_a_memory_mapped_corpus(tmp_path):
    corpus, _ = _data()
    np.save(tmp_path / "v.npy", np.asarray(corpus.vectors))
    mapped = MultiVectorCorpus(np.load(tmp_path / "v.npy", mmap_mode="r"), corpus.offsets, ids=corpus.ids)
    parts = list(split(mapped, 64))
    assert [len(s) for s in parts] == [64, 64, 22]
    assert all(np.shares_memory(s.vectors, mapped.vectors) for s in parts)
    del parts, mapped


# ----------------------------------------------------------------------------- round trip


@pytest.mark.parametrize("label", LADDER)
def test_store_round_trip_has_zero_score_difference_for_every_ladder_layout(tmp_path, label):
    corpus, queries = _data()
    p = _fitted(label, corpus)
    segs = list(split(corpus, 40))
    settled = list(settle(segs, p))
    path = tmp_path / "i.ovx"
    store = Store.create(path, pipeline=p, info={"layout": label})
    for seg, codes in zip(segs, settled, strict=True):
        store.append(codes, texts=_texts(seg), float16=seg)
    store.commit()
    store.close()

    with Store.open(path) as s:
        restored = s.pipeline()
        assert restored.to_dict() == _spec(label)
        assert pipeline_fitted_identity(restored) == pipeline_fitted_identity(p)
        assert s.n_pages == len(corpus) and s.ids == corpus.ids and s.manifest["layout"] == label
        for codes, loaded in zip(settled, s.segments(), strict=True):
            assert isinstance(loaded.codes.base, np.memmap) or loaded.num_vectors == 0
            np.testing.assert_array_equal(loaded.codes, codes.codes)
            np.testing.assert_array_equal(loaded.offsets, codes.offsets)
            assert loaded.ids == codes.ids and loaded.stats == codes.stats
            a = CompressedRetriever(codes, p, max_block_bytes=SEARCH_BLOCK_BYTES).score(queries)
            b = CompressedRetriever(loaded, restored, max_block_bytes=SEARCH_BLOCK_BYTES).score(queries)
            assert np.array_equal(a, b)  # bitwise: zero score difference
        for seg, kept in zip(segs, s.float16_segments(), strict=True):
            np.testing.assert_array_equal(np.asarray(kept.vectors), np.asarray(seg.vectors))
            np.testing.assert_array_equal(kept.offsets, seg.offsets)
        whole16 = s.float16_corpus()
        np.testing.assert_array_equal(np.asarray(whole16.vectors), np.asarray(corpus.vectors))
        del whole16
        for seg in segs:
            texts = _texts(seg)
            for pid in seg.ids:
                assert s.text(pid) == texts.get(pid, "")
        assert s.verify() == []


def test_the_fitted_centre_of_centred_4bit_is_stored_and_checked(tmp_path):
    _build(tmp_path / "i.ovx", "int4(mean)")
    assert (tmp_path / "i.ovx" / "state" / "fitted.npz").is_file()
    with np.load(tmp_path / "i.ovx" / "state" / "fitted.npz") as z:
        assert z.files == ["s0__mean"] and z["s0__mean"].dtype == np.float32
    # a different centre in place of the stored one is refused, not decoded
    with np.load(tmp_path / "i.ovx" / "state" / "fitted.npz") as z:
        mean = z["s0__mean"] + np.float32(0.25)
    np.savez(tmp_path / "i.ovx" / "state" / "fitted.npz", s0__mean=mean)
    with Store.open(tmp_path / "i.ovx") as s:
        with pytest.raises(ValueError, match="does not belong"):
            s.pipeline()
        assert s.verify() and "fitted pipeline" in s.verify()[0]


# ----------------------------------------------------------------------------- crash safety


def test_a_crash_after_the_segment_write_and_before_the_manifest_keeps_the_previous_index(tmp_path):
    corpus, queries = _data()
    p = _fitted("int4(mean)", corpus)
    segs = list(split(corpus, 60))
    path = tmp_path / "c.ovx"
    with Store.create(path, pipeline=p) as store:
        store.append(p.compress(segs[0], fit=False))
        store.commit()
    with Store.open(path) as s:
        before = Searcher(s).search(queries, 5)
        manifest = s.manifest

    child = tmp_path / "crash.py"
    child.write_text(
        "import os, sys\n"
        "import numpy as np\n"
        "from optivision.easy._store import Store, split\n"
        "from optivision.testing.synthetic import make_corpus\n"
        f"kw = {DATA!r}\n"
        "corpus = make_corpus(kw.pop('n_docs'), kw.pop('dim'), dtype=np.float16, **kw)\n"
        "segs = list(split(corpus, 60))\n"
        "store = Store.open(sys.argv[1], mode='a')\n"
        "store.append(store.pipeline().compress(segs[1], fit=False), texts={segs[1].ids[0]: 'x'})\n"
        "os._exit(17)  # dies here: segment files on disk, manifest not replaced, lock left behind\n",
        encoding="utf-8",
    )
    done = subprocess.run([sys.executable, str(child), str(path)], env=_env(), capture_output=True,
                          text=True, timeout=300, check=False)
    assert done.returncode == 17, done.stderr
    assert (path / "segments" / "001" / "codes.npy").is_file()  # the orphan
    assert (path / "text" / "001.zlib.json").is_file()

    with Store.open(path) as s:  # readers see the previous index, unchanged
        assert s.manifest == manifest and s.n_pages == len(segs[0]) and s.n_segments == 1
        assert Searcher(s).search(queries, 5) == before
        assert s.sizes()["other"] > 0  # the orphan is on disk, counted apart from the index

    with pytest.raises(IndexLocked, match="stale writer lock"):
        Store.open(path, mode="a")
    assert clear_stale_lock(path) and lock_holder(path) is None

    with Store.open(path, mode="a") as s:  # the next writer removes what the crash left
        assert not (path / "segments" / "001").exists() and not (path / "text" / "001.zlib.json").exists()
        assert s.append(p.compress(segs[1], fit=False)) == "001"
        s.commit()
    with Store.open(path) as s:
        assert s.n_pages == len(segs[0]) + len(segs[1])
        assert s.verify() == []


def test_a_failed_manifest_write_leaves_the_previous_manifest(tmp_path, monkeypatch):
    _, queries = _data()
    path = tmp_path / "f.ovx"
    p = _build(path, "int8(per_vector)", pages=50, float16=False)
    with Store.open(path) as s:
        before = (s.manifest, Searcher(s).search(queries, 4))
    real = _store._atomic_write

    def failing(target, data):
        if Path(target).name == "manifest.json":
            raise OSError("disk full (simulated)")
        return real(target, data)

    monkeypatch.setattr(_store, "_atomic_write", failing)
    extra = make_corpus(20, DATA["dim"], seed=99, dtype=np.float16)
    extra.ids = [f"x{i}" for i in range(20)]
    with Store.open(path, mode="a") as s:
        s.append(p.compress(extra, fit=False))
        with pytest.raises(OSError, match="disk full"):
            s.commit()
        assert s.dirty
    monkeypatch.setattr(_store, "_atomic_write", real)
    with Store.open(path) as s:
        assert (s.manifest, Searcher(s).search(queries, 4)) == before
    assert sorted(d.name for d in (path / "segments").iterdir()) == ["000", "001", "002"]  # orphan cleaned on close


def test_uncommitted_changes_are_discarded_on_close(tmp_path):
    corpus, _ = _data()
    path = tmp_path / "u.ovx"
    p = _build(path, "binary", pages=75, float16=False)
    extra = make_corpus(10, DATA["dim"], seed=5, dtype=np.float16)
    extra.ids = [f"new{i}" for i in range(10)]
    with Store.open(path, mode="a") as s:
        s.append(p.compress(extra, fit=False), texts={"new0": "t"})
        s.aliases = {"a": "new0"}
        assert s.n_pages == len(corpus) + 10 and "new0" in s
    assert sorted(d.name for d in (path / "segments").iterdir()) == ["000", "001"]
    assert not (path / "aliases.json").exists()
    with Store.open(path) as s:
        assert s.n_pages == len(corpus) and "new0" not in s


def test_create_then_close_without_commit_leaves_no_index(tmp_path):
    corpus, _ = _data()
    path = tmp_path / "n.ovx"
    p = _fitted("int8(per_vector)", corpus)
    with Store.create(path, pipeline=p) as s:
        s.append(p.compress(corpus, fit=False))
    assert not path.exists()
    with pytest.raises(FileNotFoundError, match="not an index"):
        Store.open(path)


# ----------------------------------------------------------------------------- writer lock


def test_lock_contention_between_processes(tmp_path):
    path = tmp_path / "l.ovx"
    _build(path, "int8(per_vector)", pages=75, float16=False, texts=False)
    code = ("import os, sys\n"
            "from optivision.easy._store import writer_lock\n"
            "with writer_lock(sys.argv[1]):\n"
            "    print(os.getpid(), flush=True)\n"
            "    sys.stdin.readline()\n")
    child = subprocess.Popen([sys.executable, "-c", code, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=_env())
    try:
        line = child.stdout.readline().strip()
        assert line.isdigit(), child.stderr.read()
        holder = int(line)
        with pytest.raises(IndexLocked) as info:
            Store.open(path, mode="a")
        assert f"process {holder}" in str(info.value) and "being written" in str(info.value)
        with pytest.raises(IndexLocked), writer_lock(path):
            pass
        assert lock_holder(path)["pid"] == holder
        with Store.open(path) as s:  # readers never wait for the writer
            assert s.n_pages == DATA["n_docs"]
        assert not clear_stale_lock(path)  # the holder is alive
    finally:
        _, err = child.communicate("\n", timeout=120)
    assert child.returncode == 0, err
    assert lock_holder(path) is None
    with Store.open(path, mode="a") as s:
        assert s.writable


def test_a_stale_lock_names_the_dead_process_and_can_be_cleared(tmp_path):
    path = tmp_path / "s.ovx"
    _build(path, "binary", pages=75, float16=False, texts=False)
    pid = _dead_pid()
    (path / LOCK_NAME).write_text(json.dumps({"pid": pid, "host": socket.gethostname(),
                                              "time": "2026-10-21T10:00:00Z"}), encoding="utf-8")
    with pytest.raises(IndexLocked) as info:
        Store.open(path, mode="a")
    message = str(info.value)
    assert f"process {pid}" in message and "no longer running" in message and "clear_stale_lock" in message
    assert clear_stale_lock(path)
    with Store.open(path, mode="a"):
        assert lock_holder(path)["pid"] == os.getpid()
    assert lock_holder(path) is None


def test_locks_from_another_machine_or_unreadable_are_never_cleared(tmp_path):
    path = tmp_path / "o.ovx"
    _build(path, "binary", pages=75, float16=False, texts=False)
    (path / LOCK_NAME).write_text(json.dumps({"pid": 1234, "host": "elsewhere", "time": "2026-10-21T10:00:00Z"}),
                                  encoding="utf-8")
    with pytest.raises(IndexLocked, match="on elsewhere"):
        Store.open(path, mode="a")
    assert not clear_stale_lock(path)
    (path / LOCK_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexLocked, match="cannot be read"):
        Store.open(path, mode="a")
    assert not clear_stale_lock(path)
    (path / LOCK_NAME).unlink()


def test_the_lock_is_reentrant_in_one_thread_and_exclusive_across_threads(tmp_path):
    corpus, _ = _data()
    path = tmp_path / "r.ovx"
    p = _fitted("int8(per_vector)", corpus)
    errors = []
    with writer_lock(path):  # the build holds it across encoding, then creates the store inside
        with Store.create(path, pipeline=p) as store, store.lock():
            store.append(p.compress(corpus, fit=False))
            store.commit()

        def other():
            try:
                with writer_lock(path):
                    pass
            except IndexLocked as exc:
                errors.append(str(exc))

        t = threading.Thread(target=other)
        t.start()
        t.join()
        assert (path / LOCK_NAME).exists()  # still held by the outer block
    assert errors and "another thread" in errors[0]
    assert not (path / LOCK_NAME).exists()


def test_read_only_and_closed_stores_refuse_writes(tmp_path):
    corpus, _ = _data()
    path = tmp_path / "ro.ovx"
    p = _build(path, "int8(per_vector)", pages=75, float16=False)
    codes = p.compress(corpus.select([0]), fit=False)
    with Store.open(path) as s:
        with pytest.raises(PermissionError, match="read-only"):
            s.append(codes)
        with pytest.raises(PermissionError):
            s.aliases = {"a": "b"}
    with pytest.raises(ValueError, match="closed"):
        s.segments()


# ----------------------------------------------------------------------------- create and append checks


def test_create_refuses_an_existing_index_unless_overwrite_and_never_clears_foreign_files(tmp_path):
    path = tmp_path / "e.ovx"
    p = _build(path, "binary", pages=75)
    with pytest.raises(FileExistsError, match="already holds an index"):
        Store.create(path, pipeline=p)
    assert lock_holder(path) is None
    with Store.create(path, pipeline=p, overwrite=True) as s:
        s.commit()
    with Store.open(path) as s:
        assert s.n_pages == 0 and s.ids == []
    mine = tmp_path / "my_pdfs"
    mine.mkdir()
    (mine / "report.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="not part of an index"):
        Store.create(mine, pipeline=p)
    assert (mine / "report.pdf").read_bytes() == b"%PDF-1.4" and not (mine / LOCK_NAME).exists()
    with pytest.raises(ValueError, match="unfitted"):
        Store.create(tmp_path / "u.ovx", pipeline=Pipeline.from_dict(_spec("int4(mean)")))
    with pytest.raises(ValueError, match="may not set"):
        Store.create(tmp_path / "k.ovx", pipeline=p, info={"segments": []})


def test_append_refuses_codes_the_index_cannot_decode(tmp_path):
    corpus, _ = _data()
    path = tmp_path / "a.ovx"
    p = _fitted("int4(mean)", corpus)
    other_centre = _fitted("int4(mean)", corpus, seed=1)
    seg0, seg1 = list(split(corpus, 75))
    with Store.create(path, pipeline=p) as s:
        s.append(p.compress(seg0, fit=False), float16=seg0)
        with pytest.raises(ValueError, match="another fitted int4 state"):
            s.append(other_centre.compress(seg1, fit=False), float16=seg1)
        with pytest.raises(ValueError, match="the codes come from"):
            s.append(_fitted("int8(per_vector)", corpus).compress(seg1, fit=False), float16=seg1)
        with pytest.raises(ValueError, match="stored already"):
            s.append(p.compress(seg0, fit=False), float16=seg0)
        with pytest.raises(ValueError, match="texts name pages"):
            s.append(p.compress(seg1, fit=False), texts={"nope": "x"}, float16=seg1)
        with pytest.raises(ValueError, match="float16 copy of every page"):
            s.append(p.compress(seg1, fit=False))
        with pytest.raises(ValueError, match="same pages"):
            s.append(p.compress(seg1, fit=False), float16=seg0)
        with pytest.raises(ValueError, match="vectors_before"):
            s.append(p.compress(seg1, fit=False), float16=seg1, vectors_before=[1] * len(seg1))
        assert s.n_segments == 1
        s.append(p.compress(seg1, fit=False), float16=seg1)
        s.commit()
    with Store.open(path) as s:
        assert s.n_pages == len(corpus) and s.verify() == []


# ----------------------------------------------------------------------------- side data


def test_reference_aliases_record_and_manifest_updates_round_trip(tmp_path):
    corpus, queries = _data()
    path = tmp_path / "x.ovx"
    _build(path, "int8(per_vector)", pages=75)
    reference = {
        "probes": [{"id": "p0", "text": "What is é?", "family": "phrase", "page_id": corpus.ids[0]}],
        "relevant": {"p0": {corpus.ids[0], corpus.ids[3]}},
        "query_vectors": queries,
        "scores": np.array([[1.5, -np.inf]], dtype=np.float16),
        "depth": 10,
        "nested": {"pair": (1, 2), "none": None},
    }
    with Store.open(path, mode="a") as s:
        s.reference = reference
        s.aliases = {corpus.ids[1] + "~dup": corpus.ids[1]}
        s.record = {"model": "synthetic", "versions": {"numpy": np.__version__}, "seconds": np.float64(1.25)}
        assert s.reference["depth"] == 10  # staged values are visible to the writer
        s.commit(target=0.97, upgrade_attempts=1)
        with pytest.raises(ValueError, match="may not set"):
            s.commit(n_pages=3)
        with pytest.raises(TypeError, match="JSON"):
            s.commit(bad=object())
        with pytest.raises(TypeError):
            s.reference = {"bad": object()}
    with Store.open(path) as s:
        ref = s.reference
        assert ref["probes"] == reference["probes"] and ref["relevant"] == {"p0": {corpus.ids[0], corpus.ids[3]}}
        assert ref["query_vectors"].ids == queries.ids
        np.testing.assert_array_equal(ref["query_vectors"].vectors, queries.vectors)
        assert ref["scores"].dtype == np.float16 and np.isneginf(ref["scores"][0, 1])
        assert ref["nested"] == {"pair": [1, 2], "none": None}
        assert s.aliases == {corpus.ids[1] + "~dup": corpus.ids[1]}
        assert s.record == {"model": "synthetic", "versions": {"numpy": np.__version__}, "seconds": 1.25}
        m = s.manifest
        assert m["target"] == 0.97 and m["upgrade_attempts"] == 1 and m["layout"] == "int8(per_vector)"
        assert set(m["files"]) == {"aliases.json", "build.json", "reference/reference.npz"}
        assert s.verify() == []
    with Store.open(path, mode="a") as s:
        s.reference = None
        s.commit()
    with Store.open(path) as s:
        assert s.reference is None and not (path / "reference" / "reference.npz").exists()


def test_sizes_count_every_byte_on_disk(tmp_path):
    corpus, _ = _data()
    path = tmp_path / "z.ovx"
    _build(path, "hierarchical_merge(0.25) > int4(mean)", pages=40)
    with Store.open(path, mode="a") as s:
        s.reference = {"scores": np.zeros(30, dtype=np.float16)}
        s.aliases = {"dup": corpus.ids[0]}
        s.record = {"model": "synthetic"}
        s.commit()
    (path / "log").mkdir()
    (path / "log" / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    with Store.open(path) as s:
        sizes = s.sizes()
        files = [f for f in path.rglob("*") if f.is_file()]
        assert sizes["total"] == sum(f.stat().st_size for f in files)
        assert sizes["total"] == sum(v for k, v in sizes.items() if k != "total")

        def size_of(pattern: str) -> int:
            return sum(f.stat().st_size for f in path.glob(pattern) if f.is_file())

        assert sizes["vectors"] == size_of("segments/*/codes.npy") + size_of("segments/*/offsets.npy") + size_of(
            "state/fitted.npz")
        assert sizes["text"] == size_of("text/*") and sizes["float16"] == size_of("float16/*/*")
        assert sizes["reference"] == size_of("reference/*") and sizes["log"] == size_of("log/*")
        assert sizes["other"] == 0
        stats = s.stats()
        assert stats["vectors_before"] == corpus.num_vectors and stats["exact"]
        assert stats["float16_bytes_before"] == corpus.num_vectors * DATA["dim"] * 2
        assert stats["bytes"] == stats["codes_bytes"] + DATA["dim"] * 4  # codes + the fitted centre


# ----------------------------------------------------------------------------- remove and compact


def test_remove_rewrites_the_affected_segments_and_keeps_everything_else_exact(tmp_path):
    corpus, queries = _data()
    path = tmp_path / "p.ovx"
    p = _build(path, "hierarchical_merge(0.33) > int8(per_vector)", pages=40)
    segs = list(split(corpus, 40))
    gone = [segs[0].ids[1], segs[0].ids[5], *segs[2].ids, segs[3].ids[-1]]
    with Store.open(path, mode="a") as s:
        with pytest.raises(KeyError, match="not in the index"):
            s.remove(["nope"])
        assert s.remove(["nope"], missing_ok=True) == 0
        assert s.remove(gone) == len(gone)
        assert s.n_segments == 3 and s.n_pages == len(corpus) - len(gone)
        s.commit()
    assert sorted(d.name for d in (path / "segments").iterdir()) == ["001", "004", "005"]
    keep = [i for i, pid in enumerate(corpus.ids) if pid not in set(gone)]
    rest = corpus.select(keep)
    with Store.open(path) as s:
        assert s.ids == rest.ids
        whole = CompressedRetriever(p.compress(rest, fit=False), p, max_block_bytes=SEARCH_BLOCK_BYTES)
        want = whole.search(queries, 6)
        got = Searcher(s).search(queries, 6)
        assert [[pid for pid, _ in row] for row in got] == want.ids
        np.testing.assert_allclose(np.array([[x for _, x in row] for row in got], dtype=np.float32), want.scores,
                                   rtol=1e-6, atol=1e-6)
        kept16 = s.float16_corpus()
        np.testing.assert_array_equal(np.asarray(kept16.vectors), np.asarray(rest.vectors))
        del kept16
        with pytest.raises(KeyError):
            s.text(gone[0])
        survivor = segs[0].ids[3]
        assert s.text(survivor) == _texts(segs[0]).get(survivor, "")
        stats = s.stats()
        assert stats["exact"] and stats["vectors_before"] == rest.num_vectors
        assert s.verify() == []


def test_removal_without_per_page_counts_estimates_the_sizes_and_says_so(tmp_path):
    corpus, _ = _data()
    path = tmp_path / "q.ovx"
    p = _fitted("hierarchical_merge(0.25) > binary", corpus)
    with Store.create(path, pipeline=p) as s:
        for codes in settle(corpus, p, segment_pages=75):
            s.append(codes)  # a merge, no float16 copy, no vectors_before: per-page counts unknown
        s.commit()
    with Store.open(path, mode="a") as s:
        assert s.stats()["exact"]
        s.remove([corpus.ids[0]])
        assert not s.stats()["exact"]


def test_compaction_keeps_search_results_identical(tmp_path):
    corpus, queries = _data()
    path = tmp_path / "k.ovx"
    _build(path, "hierarchical_merge(0.33) > binary", pages=12)  # 13 segments
    with Store.open(path) as s:
        assert s.n_segments == 13
        before = Searcher(s).search(queries, 10)
        dense = Searcher(s).score(queries)
        texts = {pid: s.text(pid) for pid in s.ids}
    with Store.open(path, mode="a") as s:
        assert s.compact()
        assert s.n_segments == 8
        s.commit()
        assert not s.compact()
    assert len(list((path / "segments").iterdir())) == 8 and len(list((path / "float16").iterdir())) == 8
    with Store.open(path) as s:
        after = Searcher(s).search(queries, 10)
        assert [[pid for pid, _ in row] for row in after] == [[pid for pid, _ in row] for row in before]
        scores_after = np.array([[x for _, x in r] for r in after])
        np.testing.assert_allclose(scores_after, np.array([[x for _, x in r] for r in before]), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(Searcher(s).score(queries), dense, rtol=1e-6, atol=1e-6)
        assert s.ids == corpus.ids and {pid: s.text(pid) for pid in s.ids} == texts
        whole16 = s.float16_corpus()
        np.testing.assert_array_equal(np.asarray(whole16.vectors), np.asarray(corpus.vectors))
        del whole16
        assert s.stats()["vectors_before"] == corpus.num_vectors and s.verify() == []


def test_compaction_plan_merges_neighbours_with_the_fewest_pages():
    assert _store._plan_groups([500] * 10, 8) == [[0, 1], [2, 3], [4], [5], [6], [7], [8], [9]]
    assert _store._plan_groups([500, 10, 10, 500, 500], 3) == [[0, 1, 2], [3], [4]]  # tie: leftmost pair
    assert _store._plan_groups([500, 10, 10, 400, 500], 3) == [[0], [1, 2, 3], [4]]
    assert _store._plan_groups([1, 2, 3], 8) == [[0], [1], [2]]


# ----------------------------------------------------------------------------- Windows


def test_open_close_then_delete_the_folder(tmp_path):
    corpus, queries = _data()
    path = tmp_path / "w.ovx"
    _build(path, "int4(mean)", pages=40)
    s = Store.open(path)
    Searcher(s).search(queries, 3)
    assert s.float16_segments() is not None and s.text(corpus.ids[0]) != ""
    assert len(s.segments()) == 4 and s.verify() == []
    s.close()
    shutil.rmtree(path)  # no memory map is left open (Windows refuses to delete mapped files)
    assert not path.exists()

    _build(path, "int4(mean)", pages=40)
    with Store.open(path, mode="a") as s:
        Searcher(s).search(queries, 3)
        s.remove([corpus.ids[0]])
        s.compact(max_segments=2)
        s.commit()
        Searcher(s).search(queries, 3)
    shutil.rmtree(path)
    assert not path.exists()
