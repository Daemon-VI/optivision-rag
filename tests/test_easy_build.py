"""The easy-mode build pipeline (``optivision.easy._build``; FINAL_DESIGN 4.2, 5.5-5.7).

The modules the build calls but does not own (``_levels``, ``_docs``, ``_store``, ``_encode``, ...)
are replaced by the contract-shaped fakes of ``tests/easy_build_fakes.py``; what is tested here is
the orchestration: sources, ids, sync, shards and resume, skipping, locks, disk checks, the gates'
direction, settling and the printed summary.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

from optivision.easy import _build, _claim
from optivision.easy._types import IndexLocked, NotEnoughDisk, SourceMissing, TargetMismatch
from optivision.testing.synthetic import make_corpus, make_queries

from . import easy_build_fakes as F


@pytest.fixture
def fakes(monkeypatch):
    return F.install(monkeypatch)


@pytest.fixture
def ov(fakes):
    import optivision.easy as ov

    return ov


def _arrays(corpus):
    return [np.asarray(corpus.vectors[lo:hi]) for lo, hi in zip(corpus.offsets[:-1], corpus.offsets[1:])]


def _folder(tmp_path: Path, name: str = "my_pdfs", files: dict[str, list[str]] | None = None) -> Path:
    root = tmp_path / name
    files = files or {"report.pdf": F.SENTENCES[:3]}
    for rel, pages in files.items():
        F.make_pdf(root / rel, pages)
    return root


def _ids(index) -> list[str]:
    return sorted(index._store.ids)


def _all_lines(block):
    return list(block.lines)


# ============================================================================ end to end


def test_build_from_vectors_end_to_end(ov, tmp_path):
    corpus = make_corpus(40, 16, seed=0)
    index = ov.build(_arrays(corpus), path=tmp_path / "v.ovx", progress=False)
    assert index.n_pages == 40
    assert _ids(index) == sorted(str(i) for i in range(40))
    queries, qrels = make_queries(corpus, 12, seed=1)
    results = index.search_vectors(queries, k=3)
    assert len(results) == 12 and all(len(r) == 3 for r in results)
    names = {f"d{i:06d}": str(i) for i in range(40)}
    found = sum(names[next(iter(qrels[q]))] == r[0].id for q, r in zip(queries.ids, results))
    assert found >= 10  # exact search over 8-bit codes of the copied vectors
    assert all(h.file is None and h.page is None for r in results for h in r)
    m = index._manifest
    assert m["source_kind"] == "vectors" and m["checked"] is False
    assert "1-bit" not in index.layout.plain  # unchecked: no 1-bit layouts
    assert not (tmp_path / "v.ovx.work").exists()  # vectors need no shards on disk


def test_build_from_a_three_page_pdf(ov, tmp_path):
    root = _folder(tmp_path)
    adapter = F.stub_adapter()
    index = ov.build(root, model=adapter, progress=False)
    assert index.path == (tmp_path / "my_pdfs.ovx").resolve()
    assert _ids(index) == [f"my_pdfs/report.pdf::p{i}" for i in (1, 2, 3)]
    hit = index.search("freight prices doubled", k=2)[0]
    assert hit.id == "my_pdfs/report.pdf::p1"
    assert Path(hit.file) == (root / "report.pdf").resolve() and hit.page == 1
    assert "freight" in hit.text.replace("\r", "").replace("\n", "")
    assert adapter.calls.pages == 3
    assert not Path(str(index.path) + ".work").exists()  # shards deleted after the build
    assert not Path(str(index.path) + ".lock").exists()  # lock released


def test_path_defaults_beside_the_folder_and_file(ov, tmp_path):
    root = _folder(tmp_path)
    one = ov.build(root / "report.pdf", model=F.stub_adapter(), progress=False)
    assert one.path == (root / "report.ovx").resolve()
    assert _ids(one)[0] == "my_pdfs/report.pdf::p1"  # a single file is named by its folder too


def test_an_image_and_a_multivector_corpus_are_sources(ov, tmp_path):
    from PIL import Image

    folder = tmp_path / "scans"
    folder.mkdir()
    Image.new("RGB", (40, 60), (200, 10, 10)).save(folder / "scan.png")
    one = ov.build(folder / "scan.png", model=F.stub_adapter(), progress=False)
    assert _ids(one) == ["scans/scan.png::p1"] and one.path == (folder / "scan.ovx").resolve()
    corpus = make_corpus(8, 8, seed=0)
    corpus.ids = [f"c{i}" for i in range(8)]
    index = ov.build(corpus, path=tmp_path / "c.ovx", progress=False)
    assert _ids(index) == sorted(corpus.ids)


def test_keep_text_false_stores_no_text(ov, tmp_path):
    index = ov.build(_folder(tmp_path), model=F.stub_adapter(), keep_text=False, progress=False)
    assert index.search("freight prices")[0].text == ""
    assert index._manifest["keep_text"] is False


def test_a_wiring_failure_writes_nothing(ov, tmp_path, monkeypatch):
    from optivision.easy._types import WiringError

    enc = F.make_encode()

    def failing(encoder, pages):
        raise WiringError("pages and their sentences do not match, also on the sample PDF; see ov.doctor()")

    enc.wiring_check = failing
    F.install(monkeypatch, encode=enc)
    adapter = F.stub_adapter()
    root = _folder(tmp_path)
    with pytest.raises(WiringError):
        ov.build(root, model=adapter, progress=False)
    assert adapter.calls.pages == 0 and not (tmp_path / "my_pdfs.ovx").exists()
    assert not Path(str(tmp_path / "my_pdfs.ovx") + ".lock").exists()


def test_a_target_above_99_percent_says_the_benchmarks_cannot_separate(ov, tmp_path):
    index = ov.build(_arrays(make_corpus(6, 8, seed=0)), target=0.995, path=tmp_path / "v.ovx", progress=False)
    assert index.layout.chain == "float"
    assert "cannot separate layouts this close to full precision" in str(index.summary)


def test_strings_for_a_text_model(ov, tmp_path):
    texts = [f"{s} item {i}" for i, s in enumerate(F.SENTENCES)]
    index = ov.build(texts, model=F.stub_adapter(), path=tmp_path / "t.ovx", ids=[f"doc{i}" for i in range(6)],
                     progress=False)
    assert _ids(index) == sorted(f"doc{i}" for i in range(6))
    assert index._manifest["modality"] == "text"
    hit = index.search("dividend policy shareholders")[0]
    assert hit.id == "doc1" and hit.file is None and hit.text.startswith("The board")


def test_torch_tensors_are_cast_and_padding_dropped(ov, tmp_path):
    torch = pytest.importorskip("torch")
    corpus = make_corpus(12, 16, tokens=6, seed=3)
    batch = np.zeros((12, 9, 16), dtype=np.float32)
    for i, a in enumerate(_arrays(corpus)):
        batch[i, : a.shape[0]] = a
    index = ov.build(torch.from_numpy(batch), path=tmp_path / "p.ovx", progress=False)
    assert index.n_pages == 12
    text = str(index.summary)
    assert "torch tensors were converted" in text
    assert "36 all-zero rows dropped as padding" in text
    lines = [ln for ln in index.summary.lines if ln.label == "Padding"]
    assert lines and lines[0].tag == "[counted]"


# ============================================================================ sync


def test_sync_is_idempotent(ov, tmp_path):
    root = _folder(tmp_path)
    adapter = F.stub_adapter()
    first = ov.build(root, model=adapter, progress=False)
    ids, size = _ids(first), first.size
    first.close()
    again = ov.build(root, model=adapter, progress=False)
    assert adapter.calls.pages == 3  # nothing encoded the second time
    assert _ids(again) == ids and again.size == size
    s = again.last_sync
    assert not (s.added or s.changed or s.missing) and s.n_pages == 3
    assert "nothing new" in s.message


def test_sync_new_changed_missing_and_prune(ov, tmp_path):
    root = _folder(tmp_path, files={"a.pdf": F.SENTENCES[:2], "b.pdf": F.SENTENCES[2:4]})
    adapter = F.stub_adapter()
    ov.build(root, model=adapter, progress=False).close()
    F.make_pdf(root / "c.pdf", F.SENTENCES[4:6])  # new
    F.make_pdf(root / "a.pdf", [F.SENTENCES[0], "a completely different second page about tea"])  # changed
    os.utime(root / "a.pdf", ns=(10**9, 10**9))
    (root / "b.pdf").unlink()  # missing
    index = ov.build(root, model=adapter, progress=False)
    s = index.last_sync
    assert [Path(p).name for p, _ in s.added] == ["c.pdf"]
    assert [Path(p).name for p, _ in s.changed] == ["a.pdf"]
    assert [Path(p).name for p, _ in s.missing] == ["b.pdf"] and "kept" in s.missing[0][1]
    assert "my_pdfs/b.pdf::p1" in _ids(index)  # kept until prune
    assert index.search("different second page tea")[0].id == "my_pdfs/a.pdf::p2"
    assert len(_ids(index)) == len(set(_ids(index))) == 6
    index.close()
    pruned = ov.build(root, model=adapter, prune=True, progress=False)
    assert not any(i.startswith("my_pdfs/b.pdf") for i in _ids(pruned))
    assert "my_pdfs/b.pdf" not in pruned._manifest["docs"]
    assert pruned.n_pages == 4


def test_touched_but_unchanged_file_is_not_encoded_again(ov, tmp_path):
    root = _folder(tmp_path)
    adapter = F.stub_adapter()
    ov.build(root, model=adapter, progress=False).close()
    os.utime(root / "report.pdf", ns=(5 * 10**9, 5 * 10**9))  # new mtime, same bytes: sha256 decides
    index = ov.build(root, model=adapter, progress=False)
    assert adapter.calls.pages == 3
    assert index._manifest["docs"]["my_pdfs/report.pdf"]["mtime"] == 5 * 10**9


def test_ids_distinct_across_sources_with_the_same_names(ov, tmp_path):
    a = _folder(tmp_path / "a", "docs", {"report.pdf": F.SENTENCES[:2]})
    b = _folder(tmp_path / "b", "docs", {"report.pdf": F.SENTENCES[2:4]})
    more = _folder(tmp_path, "more_pdfs", {"report.pdf": F.SENTENCES[4:6]})
    adapter = F.stub_adapter()
    index = ov.build(a, model=adapter, path=tmp_path / "all.ovx", progress=False)
    index.add(b)
    index.add(more)
    ids = _ids(index)
    assert ids == sorted(["docs/report.pdf::p1", "docs/report.pdf::p2", "docs~2/report.pdf::p1",
                          "docs~2/report.pdf::p2", "more_pdfs/report.pdf::p1", "more_pdfs/report.pdf::p2"])
    assert index._manifest["sources"] == {"docs": str(a.resolve()), "docs~2": str(b.resolve()),
                                          "more_pdfs": str(more.resolve())}
    index.add(b)  # the same folder again keeps its namespace
    assert _ids(index) == ids


# ============================================================================ shards and resume


def test_kill_resume_gives_the_same_index(ov, tmp_path, monkeypatch):
    monkeypatch.setattr(_build, "SEGMENT_PAGES", 2)
    monkeypatch.setattr(_build, "ENCODE_CHUNK", 1)
    files = {"a.pdf": F.SENTENCES[:3], "b.pdf": F.SENTENCES[3:6]}
    clean_root = _folder(tmp_path / "clean", "docs", files)
    clean = ov.build(clean_root, model=F.stub_adapter(), progress=False)

    root = _folder(tmp_path / "run", "docs", files)
    work = tmp_path / "drive_work"
    crashing = F.stub_adapter(F.Calls(fail_after=3))
    with pytest.raises(RuntimeError, match="simulated crash"):
        ov.build(root, model=crashing, work_dir=work, progress=False)
    state = json.loads((work / "docs.ovx" / "shards.json").read_text())
    assert state["shards"]  # pages encoded before the crash are kept
    assert not (tmp_path / "run" / "docs.ovx").exists()  # no half index
    assert not Path(str(tmp_path / "run" / "docs.ovx") + ".lock").exists()

    resumed_adapter = F.stub_adapter()  # a new session
    resumed = ov.build(root, model=resumed_adapter, work_dir=work, progress=False)
    assert crashing.calls.pages + resumed_adapter.calls.pages == 6  # each page encoded once
    assert resumed_adapter.calls.pages < 6
    ids = resumed._store.ids
    assert len(ids) == len(set(ids)) == 6
    assert sorted(ids) == _ids(clean)
    q = ["freight prices doubled", "safety review training"]
    assert [[h.id for h in r] for r in resumed.search(q, k=3)] == [[h.id for h in r] for r in clean.search(q, k=3)]
    assert not (work / "docs.ovx").exists()  # shards deleted once settled


def test_shards_of_another_model_are_not_reused(ov, tmp_path, fakes):
    root = _folder(tmp_path)
    work = tmp_path / "w"
    shards = _build.ShardSet(work / "my_pdfs.ovx", {"model": "someone/else", "dtype": "x", "revision": None,
                                                    "kind": "files"})
    shards.flush()
    (work / "my_pdfs.ovx" / "shards.json").write_text(json.dumps({"identity": {"model": "someone/else"},
                                                                   "shards": [], "files": {}}))
    adapter = F.stub_adapter()
    ov.build(root, model=adapter, work_dir=work, progress=False)
    assert adapter.calls.pages == 3


def test_segment_settle_equals_whole_corpus_settle(fakes, tmp_path, monkeypatch):
    from optivision.compose import Pipeline

    monkeypatch.setattr(_build, "SEGMENT_PAGES", 7)
    corpus = make_corpus(30, 16, seed=5).astype(np.float16)
    shards = _build.ShardSet(tmp_path / "s", {"kind": "files"})
    for i in range(len(corpus)):
        u = _build.Unit(doc_id=f"d{i}", kind="text", payload="x")
        shards.begin(u)
        lo, hi = corpus.offsets[i], corpus.offsets[i + 1]
        shards.add(_build.PageMeta(id=f"d{i}", doc_id=f"d{i}"), np.asarray(corpus.vectors[lo:hi]))
        shards.end(u)
        shards.maybe_flush()
    shards.flush()
    assert len(shards.names) == 5
    ids = {f"d{i}" for i in range(30)}
    for spec in F.SPECS.values():
        pipe = Pipeline.from_dict(spec).fit(_build.fit_sample(shards, ids))
        whole = pipe.compress(corpus.with_vectors(corpus.vectors, ids=[f"d{i}" for i in range(30)]), fit=False)
        parts = [pipe.compress(shards.load(n, p), fit=False) for n, p in shards.encoded(ids)]
        assert np.array_equal(np.concatenate([p.codes for p in parts]), whole.codes)
        assert [i for p in parts for i in p.ids] == whole.ids


def test_fit_sample_is_seeded_and_capped(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(_build, "FIT_ROWS", 100)
    corpus = make_corpus(50, 8, tokens=(5, 15), seed=2)
    shards = _build.ShardSet(None, {"kind": "vectors"})
    for i in range(50):
        u = _build.Unit(doc_id=f"d{i}", kind="vectors", payload=None)
        shards.begin(u)
        shards.add(_build.PageMeta(id=f"d{i}", doc_id=f"d{i}"),
                   np.asarray(corpus.vectors[corpus.offsets[i]:corpus.offsets[i + 1]]))
        shards.end(u)
    shards.flush()
    ids = {f"d{i}" for i in range(50)}
    a, b = _build.fit_sample(shards, ids), _build.fit_sample(shards, ids)
    assert a.num_vectors <= 100 and a.ids == b.ids and np.array_equal(a.vectors, b.vectors)
    assert 0 < len(a) < 50


# ============================================================================ files that cannot be read


def test_corrupt_encrypted_and_empty_pdfs_are_skipped(ov, tmp_path):
    root = _folder(tmp_path, files={"good.pdf": F.SENTENCES[:2]})
    (root / "corrupt.pdf").write_bytes(b"%PDF-1.4\nthis is not a pdf at all")
    F.make_pdf(root / "locked.pdf", ["a secret page"], encrypt="secret")
    (root / "empty.pdf").write_bytes(b"")
    index = ov.build(root, model=F.stub_adapter(), progress=False)
    assert _ids(index) == ["my_pdfs/good.pdf::p1", "my_pdfs/good.pdf::p2"]
    reasons = {Path(p).name: r for p, r in index._manifest["skipped"]}
    assert reasons["locked.pdf"] == "encrypted"
    assert reasons["corrupt.pdf"].startswith("unreadable")
    assert reasons["empty.pdf"] == "empty file"
    assert {Path(p).name for p, _ in index.last_sync.skipped} == {"locked.pdf", "corrupt.pdf", "empty.pdf"}
    assert "3 files skipped" in str(index.summary)


def test_blank_pages_skipped_and_identical_pages_stored_once(ov, tmp_path):
    root = _folder(tmp_path, files={"a.pdf": [F.SENTENCES[0], "", F.SENTENCES[1]],
                                    "b.pdf": [F.SENTENCES[1], F.SENTENCES[2]]})
    adapter = F.stub_adapter()
    index = ov.build(root, model=adapter, progress=False)
    m = index._manifest
    assert m["blank"] == ["my_pdfs/a.pdf::p2"]
    assert m["aliases"] == {"my_pdfs/b.pdf::p1": "my_pdfs/a.pdf::p3"}
    assert adapter.calls.pages == 3 and index.n_pages == 3
    assert "1 blank pages skipped; 1 identical pages stored once" in str(index.summary)


# ============================================================================ refusals, locks, disk


def test_target_mismatch_names_the_rebuild(ov, tmp_path):
    root = _folder(tmp_path)
    adapter = F.stub_adapter()
    ov.build(root, model=adapter, target=0.97, progress=False).close()
    with pytest.raises(TargetMismatch) as err:
        ov.build(root, model=adapter, target=0.95, progress=False)
    text = str(err.value)
    assert "built at 0.97" in text and "rebuild=True" in text
    with pytest.raises(TargetMismatch, match="layout"):
        ov.build(root, model=adapter, layout="centred 4-bit", progress=False)
    same = ov.build(root, model=adapter, target=0.97, layout="merged (1 in 4 vectors) + 8-bit", progress=False)
    same.close()
    rebuilt = ov.build(root, model=adapter, target=0.95, rebuild=True, progress=False)
    assert rebuilt.target == 0.95 and rebuilt.path == (tmp_path / "my_pdfs.ovx").resolve()
    assert _ids(rebuilt) == [f"my_pdfs/report.pdf::p{i}" for i in (1, 2, 3)]
    assert not (tmp_path / "my_pdfs.ovx.rebuild").exists() and not (tmp_path / "my_pdfs.ovx.old").exists()


def test_target_mismatch_with_a_registry_model_prints_the_reencode_time(ov, tmp_path, monkeypatch):
    F.install(monkeypatch, encode=F.make_encode({"fake/page-model": F.stub_adapter()}))
    root = _folder(tmp_path)
    ov.build(root, model="fake/page-model", progress=False).close()
    with pytest.raises(TargetMismatch) as err:
        ov.build(root, model="fake/page-model", target=0.95, progress=False)
    assert "re-encoding takes about" in str(err.value) and "[estimate: our Kaggle run" in str(err.value)


def test_rebuild_reuses_a_kept_float16_copy(ov, tmp_path):
    root = _folder(tmp_path)
    adapter = F.stub_adapter()
    ov.build(root, model=adapter, keep_float16=True, progress=False).close()
    assert adapter.calls.pages == 3
    rebuilt = ov.build(root, model=adapter, layout="centred 4-bit", rebuild=True, progress=False)
    assert adapter.calls.pages == 3  # nothing encoded again
    assert rebuilt.layout.name == "int4(mean)"
    assert rebuilt.search("freight prices doubled")[0].id == "my_pdfs/report.pdf::p1"


def test_target_outside_range_and_accuracy_alias(ov, tmp_path, capsys):
    with pytest.raises(ValueError, match="target must be in"):
        ov.build(make_corpus(5, 8, seed=0).vectors[None], target=0.5, path=tmp_path / "x.ovx")
    corpus = make_corpus(10, 8, seed=0)
    index = ov.build(_arrays(corpus), accuracy=0.98, path=tmp_path / "a.ovx")
    out = capsys.readouterr().out
    assert "accuracy=0.98 read as target=0.98: keep at least 98% of full-precision search quality (nDCG@5); " \
           "not 98% of answers correct." in out
    assert index.target == 0.98


def test_missing_source_and_bad_inputs(ov, tmp_path):
    with pytest.raises(SourceMissing):
        ov.build(tmp_path / "nowhere", model=F.stub_adapter())
    with pytest.raises(TypeError):
        ov.build(42)
    with pytest.raises(ValueError, match="ids"):
        ov.build(_folder(tmp_path), ids=["a"])
    with pytest.raises(ValueError, match="max_disk"):
        ov.build(_arrays(make_corpus(5, 8, seed=0)), target=0.97, max_disk="1 GB", path=tmp_path / "m.ovx")


def test_writer_lock_refuses_a_running_build_and_takes_over_a_dead_one(ov, tmp_path):
    corpus = make_corpus(10, 8, seed=0)
    path = tmp_path / "v.ovx"
    lock = Path(str(path) + ".lock")
    lock.write_text(json.dumps({"pid": os.getpid() + 0, "host": socket.gethostname(), "started": "now"}))
    # our own pid is alive: refused (a second build in this process waits for nobody)
    with pytest.raises(IndexLocked, match="delete"):
        _build.BuildLock(path).acquire()
    lock.write_text(json.dumps({"pid": 2**22 + 12345, "host": socket.gethostname(), "started": "then"}))
    index = ov.build(_arrays(corpus), path=path, progress=False)  # a dead process's lock is taken over
    assert index.n_pages == 10 and not lock.exists()


def test_not_enough_disk_stops_before_encoding(ov, tmp_path, monkeypatch):
    monkeypatch.setattr(_build, "free_bytes", lambda path: 10)
    adapter = F.stub_adapter()
    with pytest.raises(NotEnoughDisk) as err:
        ov.build(_folder(tmp_path), model=adapter, progress=False)
    assert adapter.calls.pages == 0
    text = str(err.value)
    assert "[estimate: our bytes per page x your pages]" in text and "reported by that drive  [counted]" in text


def test_work_dir_defaults_and_the_colab_warning(ov, tmp_path, monkeypatch, capsys):
    p = (tmp_path / "x.ovx").resolve()
    assert _build.work_root(p, None, "files") == Path(str(p) + ".work")
    assert _build.work_root(p, tmp_path / "drive", "files") == (tmp_path / "drive").resolve() / "x.ovx"
    assert _build.work_root(p, None, "vectors") is None
    monkeypatch.setattr(_build, "on_colab", lambda: True)
    assert _build.work_root(p, None, "files") == Path("/content/.ov_work") / "x.ovx"
    monkeypatch.setattr(_build, "work_root", lambda path, wd, kind: Path(str(path) + ".work"))
    monkeypatch.setattr(_build, "COLAB_WARN_SECONDS", 1.0)
    F.install(monkeypatch, encode=F.make_encode({"fake/page-model": F.stub_adapter()}))
    index = ov.build(_folder(tmp_path), model="fake/page-model")
    out = capsys.readouterr().out
    assert "A Colab disconnect loses the temporary copy in /content. Pass work_dir= on Drive" in out
    assert "Location" in out and str(index.path) + ".work" in out


# ============================================================================ trial, budget, gates


def test_sample_trial_shards_are_reused_by_the_full_build(ov, tmp_path):
    files = {f"f{i}.pdf": [f"{s} file {i}" for s in F.SENTENCES[:3]] for i in range(3)}
    root = _folder(tmp_path, files=files)
    adapter = F.stub_adapter()
    trial = ov.build(root, model=adapter, sample=2, progress=False)
    assert trial.path == (tmp_path / "my_pdfs.sample.ovx").resolve()
    trial_pages = adapter.calls.pages
    assert 0 < trial_pages < 9 and trial.n_pages == trial_pages
    assert (tmp_path / "my_pdfs.ovx.work").exists()  # kept for the full build
    full = ov.build(root, model=adapter, progress=False)
    assert adapter.calls.pages == 9  # the trial's pages were not encoded again
    assert full.n_pages == 9 and len(set(full._store.ids)) == 9


def test_sample_trial_with_a_work_dir_shares_it_with_the_full_build(ov, tmp_path):
    files = {f"f{i}.pdf": [f"{s} file {i}" for s in F.SENTENCES[:2]] for i in range(2)}
    root = _folder(tmp_path, files=files)
    adapter = F.stub_adapter()
    ov.build(root, model=adapter, sample=2, work_dir=tmp_path / "drive", progress=False)
    assert (tmp_path / "drive" / "my_pdfs.ovx" / "shards.json").exists()
    ov.build(root, model=adapter, work_dir=tmp_path / "drive", progress=False)
    assert adapter.calls.pages == 4


def test_a_sync_says_when_the_layout_no_longer_qualifies(ov, tmp_path, monkeypatch, fakes):
    root = _folder(tmp_path, files={"a.pdf": F.SENTENCES[:2]})
    adapter = F.stub_adapter()
    ov.build(root, model=adapter, progress=False).close()
    real = fakes["_levels"].select

    def grown(model, target, n_pages=None, **kw):
        sel = real(model, target, n_pages, **kw)
        if n_pages and n_pages > 2:
            from dataclasses import replace

            rows = tuple(replace(r, status="below target") if r.merged else r for r in sel.rows)
            sel = replace(sel, rows=rows)
        return sel

    monkeypatch.setattr(fakes["_levels"], "select", grown)
    F.make_pdf(root / "b.pdf", F.SENTENCES[2:4])
    index = ov.build(root, model=adapter, progress=False)
    s = index.last_sync
    assert s.layout_still_qualifies is False and "index.check().apply()" in s.message
    assert "no longer qualifies at 4 pages; nothing changed by itself" in s.message
    assert index.layout.merged  # nothing changed automatically


def test_max_disk_picks_the_target(ov, tmp_path, fakes):
    index = ov.build(_arrays(make_corpus(10, 8, seed=0)), max_disk="2 GB", path=tmp_path / "m.ovx",
                     progress=False)
    assert index.target == 0.95
    assert "max_disk= picked target 95%, the highest that fits" in str(index.summary)
    assert _build.parse_bytes("500MB") == 500_000_000 and _build.parse_bytes(1234) == 1234


def _texts(n: int) -> list[str]:
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(4000)]
    return [" ".join(rng.choice(vocab, size=12, replace=False)) for _ in range(n)]


def test_practice_check_passes_on_a_checked_collection(ov, tmp_path, fakes):
    index = ov.build(_texts(90), model=F.stub_adapter(), path=tmp_path / "t.ovx", progress=False)
    m = index._manifest
    assert m["checked"] is True and m["practice"]["ran"] and not m["practice"]["fired"]
    assert m["practice"]["n_hard"] >= 60
    assert fakes["_veto"].sequences[0][0] == m["layout"]["name"]
    lines = {ln.label: ln for ln in index.summary.lines}
    assert lines["Practice"].tag == "[questions from your PDFs]"
    assert lines["Strength"].tag == "[estimate: simulation]"


def test_gates_only_ever_pick_safer(ov, tmp_path, monkeypatch):
    veto = F.make_veto(fail=("hierarchical_merge(0.25) > int8(per_vector)", "int4(mean)"))
    mods = F.install(monkeypatch, veto=veto)
    index = ov.build(_texts(90), model=F.stub_adapter(), path=tmp_path / "t.ovx", progress=False)
    seq = [r.name for r in index.selection.sequence]
    assert index.layout.name == "int8(per_vector)"
    assert seq.index(index.layout.name) > seq.index(index.selection.pick.name)
    assert mods["_veto"].sequences[0][0] == index.selection.pick.name
    text = str(index.summary)
    assert "fell below 97% on" in text and "we used the safer 8-bit" in text


def test_practice_check_fired_down_to_float16_says_who_may_be_at_fault(ov, tmp_path, monkeypatch):
    F.install(monkeypatch, veto=F.make_veto(fail=("hierarchical_merge(0.25) > int8(per_vector)", "int4(mean)",
                                                  "int8(per_vector)")))
    index = ov.build(_texts(90), model=F.stub_adapter(), path=tmp_path / "t.ovx", progress=False)
    assert index.layout.name == "float16"
    assert "the model or the questions may be at fault; see report() and ov.doctor()" in str(index.summary)


def test_real_query_gate(ov, tmp_path, monkeypatch):
    texts = _texts(90)
    queries = [" ".join(t.split()[:3]) for t in texts[:70]]
    index = ov.build(texts, model=F.stub_adapter(), path=tmp_path / "t.ovx", queries=queries, progress=False)
    g = index._manifest["queries_gate"]
    assert g["ran"] and g["n"] == 70 and g["kind"] == "agreement with full precision, not relevance"
    line = next(ln for ln in index.summary.lines if ln.label == "Questions")
    assert line.tag == "[your questions: agreement with full precision, not relevance]"
    answered = ov.build(texts, model=F.stub_adapter(), path=tmp_path / "a.ovx", queries=queries,
                        answers=[str(i) for i in range(70)], progress=False)
    assert answered._manifest["queries_gate"]["kind"] == "search quality"
    few = ov.build(texts, model=F.stub_adapter(), path=tmp_path / "f.ovx", queries=queries[:10], progress=False)
    assert not few._manifest["queries_gate"]["ran"]
    assert "fewer than the check needs" in str(few.summary)


def test_answers_accept_page_ids_file_page_pairs_and_sets(fakes):
    files = {"my_pdfs/a.pdf": {"path": "/x/my_pdfs/a.pdf", "relpath": "a.pdf"}}
    stored = {"my_pdfs/a.pdf::p1", "my_pdfs/a.pdf::p2"}
    got = _build.answers_relevant(["my_pdfs/a.pdf::p1", ("a.pdf", 2), {("my_pdfs/a.pdf", 1), "my_pdfs/a.pdf::p3"}],
                                  files, {"my_pdfs/a.pdf::p3": "my_pdfs/a.pdf::p2"}, stored)
    assert got == [frozenset({"my_pdfs/a.pdf::p1"}), frozenset({"my_pdfs/a.pdf::p2"}),
                   frozenset({"my_pdfs/a.pdf::p1", "my_pdfs/a.pdf::p2"})]
    with pytest.raises(ValueError, match="not a file"):
        _build.answers_relevant([("b.pdf", 1)], files, {}, stored)


def test_shape_gate_gets_the_median_vectors_per_page(ov, tmp_path, fakes):
    ov.build(_arrays(make_corpus(9, 8, tokens=(4, 6), seed=0)), path=tmp_path / "v.ovx", progress=False)
    call = fakes["_levels"].calls[-1]
    assert 4 <= call["vectors_per_page"] <= 6 and call["n_pages"] == 9 and call["checked"] is False


# ============================================================================ the printed summary


def test_summary_lines_pass_check_template_and_carry_tags(ov, tmp_path):
    root = _folder(tmp_path, files={"a.pdf": [F.SENTENCES[0], "", F.SENTENCES[1]], "b.pdf": [F.SENTENCES[1]]})
    (root / "bad.pdf").write_bytes(b"%PDF-1.4 nope")
    indexes = [ov.build(root, model=F.stub_adapter(), progress=False),
               ov.build(_arrays(make_corpus(10, 8, seed=0)), path=tmp_path / "v.ovx", progress=False),
               ov.build(_texts(90), model=F.stub_adapter(), path=tmp_path / "t.ovx", progress=False,
                        queries=[" ".join(t.split()[:3]) for t in _texts(90)[:65]])]
    banned = ("lossless", "universal", "guaranteed", "first", "sota", "state of the art", "veto", "faster", "novel",
              "better than", "scales to")
    for index in indexes:
        block = index.summary
        assert block.next  # every output ends with Next:
        assert str(block).splitlines()[-1].startswith("Next: ")
        for line in block.lines:
            _claim.check_template(line.template)
            line.text().encode("ascii")  # templates and our parts are ASCII here
            if line.claims:
                assert line.tag.startswith("[") and line.render().endswith(line.tag)
                assert len({(c.kind, c.source) for c in line.claims}) == 1
            else:
                stripped = _claim.ALLOWED_DIGIT_TOKENS.sub("", line.template)
                assert not any(ch.isdigit() for ch in stripped)
            low = line.template.lower()
            assert not any(word in low for word in banned), line.template
            assert " just " not in f" {low} " and " simply " not in f" {low} "
        assert index.evidence and all("tag" in c for c in index.evidence)


def test_plan_is_printed_before_encoding(ov, tmp_path, capsys):
    ov.build(_folder(tmp_path), model=F.stub_adapter())
    out = capsys.readouterr().out
    plan, rest = out.split("Next: encoding now", 1)
    assert "Folder     3 pages to encode in 1 files" in plan and "Location" in plan and "Free" in plan
    assert "Saved" in rest and rest.rstrip().splitlines()[-1].startswith("Next: index.search(")
