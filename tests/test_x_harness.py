"""The preregistered X1-X3 harness (scripts/research/x*.py), on synthetic fixtures.

No test reads a per-query value of a research store. The registered stores are touched
only through their hashes, array headers and candidate metadata, and the experiments run
here use synthetic stores written to a temporary directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import pathlib
import re
import shutil
import sys
import types

import numpy as np
import pytest

from optivision.benchmark import Dataset
from optivision.calibration import recommended_search_space
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix

pytest.importorskip("scipy")  # the registered search space needs the Ward chains

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
E1S, E, S, X1, X2, X3, X = (importlib.import_module(m) for m in (
    "e1_1_study", "e1_common", "e1_stats", "x1_agreement", "x2_budget", "x3_symmetry", "x_common"))

BITS = {"float16": 16.0, "int8": 8.125, "int4": 4.125, "binary": 1.0}
LOSS = {"float16": 0.0, "int8": 0.0, "int4": 0.01, "binary": 0.04}
KEEP = {"ward": [1.0, 0.5, 0.33, 0.25, 0.15, 0.1],
        "adaptive_centred": [1.0, 0.57, 0.31, 0.19, 0.127, 0.09, 0.0655]}
GRID = {"common_budgets": (8, 16, 24, 48, 96), "pool_cap": 48, "e1_budgets": (8, 16, 24),
        "ab_extra_budget": 96, "ab_extra_seeds": 16, "n_seeds": 32, "replication_seeds": (32, 64),
        "draw_size": 96, "x3_budget": 24, "n_boot": 200}


# ------------------------------------------------------------------ fixtures


def discount(rank, cutoff):
    return np.where(rank <= cutoff, 1.0 / np.log2(1.0 + rank), 0.0)


def synthetic_store(directory, name, pool, n_queries, cutoff, seed):
    """A store in the E1 format with the registered 40 candidates and made-up per-query values."""
    rng = np.random.default_rng(seed)
    base_rank = rng.choice([1, 2, 3, cutoff + 2], size=n_queries, p=[0.62, 0.14, 0.08, 0.16])
    cands, values = [], {}
    for family, steps in recommended_search_space(128).items():
        codec = family.split("+")[-1]
        keep = KEEP.get(family.split("+")[0], [1.0])
        for j, pipe in enumerate(steps):
            key = json.dumps(pipe.to_dict(), sort_keys=True)
            if key not in values:  # a pipeline that opens two chains has one set of values
                p = LOSS[codec] + 0.015 * j
                shift = rng.integers(1, 4, n_queries) * (rng.random(n_queries) < p)
                other = rng.integers(1, 4, n_queries) * (rng.random(n_queries) < p)
                values[key] = (discount(base_rank + shift, cutoff),
                               discount(np.where(base_rank == 1, 1 + shift, 1 + other), cutoff))
            vectors = 1000.0 * keep[j]
            cands.append({"position": len(cands), "family": family, "label": pipe.label(),
                          "pipeline": pipe.to_dict(), "bytes_per_doc": vectors * 128 * BITS[codec] / 8,
                          "vectors_per_doc": vectors, "compression_vs_float32": 32.0 / BITS[codec] / keep[j],
                          "dim": 128, "bits_per_dim": BITS[codec], "key": key})
    arrays = {"base_labels": discount(base_rank, cutoff), "base_b1": np.ones(n_queries),
              "cand_labels": np.stack([values[c["key"]][0] for c in cands]),
              "cand_b1": np.stack([values[c["key"]][1] for c in cands])}
    for c in cands:
        del c["key"]
    meta = {"dataset": f"synthetic-{name}", "metric": f"ndcg@{cutoff}", "n_docs": 500,
            "n_queries": n_queries, "dim": 128, "candidates": cands}
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{name}.npz", **arrays)
    (directory / f"{name}.json").write_bytes(json.dumps(meta, indent=1).encode("utf-8"))
    spec = X.StoreSpec(name, "synthetic encoder", pool, n_queries, 500, 128, f"ndcg@{cutoff}",
                       X.sha256_file(directory / f"{name}.npz"), X.sha256_file(directory / f"{name}.json"))
    features = {"base_rank": base_rank, "base_margin_rel": rng.normal(size=n_queries),
                "base_margin_12": rng.random(n_queries)}
    return spec, features


def write_e1_selections(reg, name):
    """What E1.1 stored for a store: the frozen ``run_cell`` with ``e1_1_study.real``'s own draws."""
    st = E.Store(X.load_store(name, reg), "labels")
    prob = X.problem_for(st)
    n_pool = st.pq.shape[1]
    saved, E1S.SEEDS = E1S.SEEDS, reg.n_seeds
    try:
        sels = {}
        for n in reg.e1_budgets:
            def draw(seed, n=n):
                idx = np.random.default_rng(seed).integers(0, n_pool, size=n)
                return st.pq[:, idx], st.base[idx]

            res = E1S.run_cell(prob, draw, n, "library")
            for target in E1S.TARGETS:
                for m in E1S.METHODS:
                    sels[f"n{n}_T{target}_{m}"] = res["sel"][target][m]
    finally:
        E1S.SEEDS = saved
    path = pathlib.Path(reg.e1_selections[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sels)


def build_registry(tmp):
    stores, features = {}, {}
    for name, pool, n_q, cutoff, seed in (("syn_doc", "DocVQA", 60, 5, 11), ("syn_text", "SciFact", 40, 10, 12)):
        stores[name], features[name] = synthetic_store(tmp / "store", name, pool, n_q, cutoff, seed)
    (tmp / "q3").mkdir()
    np.savez_compressed(tmp / "q3" / "syn_doc.npz", **features["syn_doc"])
    reg = X.Registry(stores=stores, store_dir=tmp / "store",
                     out_roots={x: tmp / "out" / x for x in ("X1", "X2", "X3")},
                     e1_selections={n: tmp / "e1" / f"{n}_selections.npz" for n in stores},
                     q3_features=tmp / "q3", **GRID)
    for name in stores:
        write_e1_selections(reg, name)
    return reg


def copy_registry(reg, tmp):
    """The same synthetic inputs in a fresh directory, with empty outputs."""
    src = pathlib.Path(reg.store_dir).parent
    for sub in ("store", "e1", "q3"):
        shutil.copytree(src / sub, tmp / sub)
    return X.Registry(stores=reg.stores, store_dir=tmp / "store",
                      out_roots={x: tmp / "out" / x for x in ("X1", "X2", "X3")},
                      e1_selections={n: tmp / "e1" / f"{n}_selections.npz" for n in reg.stores},
                      q3_features=tmp / "q3", **GRID)


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    return build_registry(tmp_path_factory.mktemp("x_inputs"))


@pytest.fixture(scope="module")
def x1_run(registry):
    out = X1.run(registry)
    assert out["finished"]
    return pathlib.Path(registry.out_roots["X1"])


@pytest.fixture(scope="module")
def x2_run(registry):
    out = X2.run(registry)
    assert out["finished"]
    return pathlib.Path(registry.out_roots["X2"])


def vector_dataset(seed=7, dim=16, n_docs=30, n_queries=40):
    rng = np.random.default_rng(seed)

    def unit_rows(v):
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    counts = rng.integers(18, 30, size=n_docs)
    centres = rng.standard_normal((n_docs, dim))
    docs = unit_rows(np.concatenate([c + 0.9 * rng.standard_normal((k, dim))
                                     for c, k in zip(centres, counts, strict=True)]))
    n = docs.shape[0]
    corpus = MultiVectorCorpus(docs, np.concatenate([[0], np.cumsum(counts)]),
                               ids=[f"d{i}" for i in range(n_docs)], protected=rng.random(n) < 0.05,
                               positions=rng.random((n, 2)).astype(np.float32))
    target = rng.integers(0, n_docs, size=n_queries)
    q = unit_rows(np.concatenate([centres[t] + 1.4 * rng.standard_normal((6, dim)) for t in target]))
    queries = MultiVectorCorpus(q, np.arange(0, 6 * n_queries + 1, 6), ids=[f"q{i}" for i in range(n_queries)])
    return Dataset("synthetic-vectors", corpus, queries, {f"q{i}": {f"d{t}"} for i, t in enumerate(target)})


def vector_registry(tmp, ds, extension=()):
    """A registry whose 'committed' store is built from the synthetic vectors, as E1.0 would."""
    arrays, meta, _ = X3.build_store(ds, "ndcg@5", ds.corpus, ds.queries, maxsim_matrix(ds.queries, ds.corpus))
    stores = {}
    for name in ("syn_vec", *extension):
        X.write_npz(tmp / "store" / f"{name}.npz", arrays)
        X.write_json(tmp / "store" / f"{name}.json", meta)
        stores[name] = X.StoreSpec(name, "synthetic encoder", "DocVQA", len(ds.queries), len(ds.corpus), 16,
                                   "ndcg@5", X.sha256_file(tmp / "store" / f"{name}.npz"),
                                   X.sha256_file(tmp / "store" / f"{name}.json"))
    return X.Registry(stores=stores, store_dir=tmp / "store",
                      out_roots={x: tmp / "out" / x for x in ("X1", "X2", "X3")},
                      x3_primary=("syn_vec",), x3_extension=tuple(extension), **GRID)


@pytest.fixture(scope="module")
def vectors():
    return vector_dataset()


@pytest.fixture(scope="module")
def x3_run(tmp_path_factory, vectors):
    reg = vector_registry(tmp_path_factory.mktemp("x3"), vectors)
    out = X3.run(reg, loader=lambda name: (vectors, "ndcg@5", {"synthetic": "none"}))
    assert out["finished"]
    return reg


def problem(truth, families, compression):
    compression = np.asarray(compression, float)
    return E1S.Problem(families, 1.0 / compression, 1.0 / compression, compression, truth)


def research_snapshot():
    """sha256 of every file under reports/research and of the frozen E1 scripts: the stores, the
    E1 and Q1-Q10 artifacts, the preregistration, the registry, and any X1-X3 output."""
    files = [p for p in sorted((ROOT / "reports" / "research").rglob("*")) if p.is_file()]
    files += sorted((ROOT / "scripts" / "research").glob("e1_*.py"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


# ------------------------------------------------- 1. candidate-space identity


@pytest.mark.parametrize("dim", [128, 96])
def test_generated_space_is_the_registered_one(dim):
    spec = X.generated_spec(dim)
    assert X.spec_sha256(spec) == X.S0_SPEC_SHA256
    assert len(spec) == X.N_CANDIDATES == 40
    assert list(dict.fromkeys(c["family"] for c in spec)) == [
        "float16", "ward+int8", "adaptive_centred+int8", "ward+int4", "adaptive_centred+int4",
        "ward+binary", "adaptive_centred+binary"]
    distinct = X.distinct_candidates({"candidates": spec})
    assert len(distinct) == X.N_DISTINCT_CANDIDATES == 37
    assert sorted(set(range(40)) - set(distinct)) == [7, 20, 33]  # the codec-only pipelines' second chains
    assert spec[0]["pipeline"] == X.FLOAT16_PIPELINE and spec[1]["pipeline"] == X.INT8_PIPELINE


def test_registered_stores_carry_the_registered_space():
    assert list(X.REGISTERED.stores) == ["colpali_docvqa", "colpali_infovqa", "colqwen2_docvqa",
                                         "colqwen2_infovqa", "colqwen25_docvqa", "colqwen25_infovqa", "scifact"]
    for spec in X.REGISTERED.stores.values():
        check = X.check_store_files(spec)  # hashes, array headers and candidate metadata only
        assert check["identity_ok"], check["problems"]
        assert check["meta"]["candidate_spec_sha256"] == X.S0_SPEC_SHA256
        assert (check["meta"]["n_candidates"], check["meta"]["n_chains"]) == (40, 7)
        assert check["meta"]["n_distinct_candidates"] == 37
        assert check["meta"]["n_queries"] == spec.n_queries


def test_frozen_code_is_the_registered_code():
    code = X.check_code()
    assert code["ok"], [f for f, v in code["files"].items() if not v["match"]]
    assert set(code["files"]) >= {"scripts/research/e1_stats.py", "scripts/research/e1_1_study.py",
                                  "src/optivision/calibration.py", "src/optivision/scoring.py"}


# ---------------------------------------------------- 2. input hash validation


def test_a_changed_store_stops_the_run(registry, tmp_path):
    reg = copy_registry(registry, tmp_path)
    assert X.validate_inputs(reg)["ok_for_run"]
    arrays = dict(np.load(reg.store_dir / "syn_doc.npz"))
    arrays["cand_labels"][3, 0] = 0.123
    np.savez_compressed(reg.store_dir / "syn_doc.npz", **arrays)
    check = X.check_store_files(reg.stores["syn_doc"], reg)
    assert not check["ok_for_run"] and not check["identity_ok"]
    for module in (X1, X2):
        with pytest.raises(X.RegisteredInputMismatch, match="do not match"):
            module.run(reg)
    assert not (tmp_path / "out").exists()  # nothing was computed or written


def test_a_changed_candidate_list_stops_the_run(registry, tmp_path):
    reg = copy_registry(registry, tmp_path)
    side = reg.store_dir / "syn_text.json"
    meta = json.loads(side.read_text(encoding="utf-8"))
    meta["candidates"][2]["pipeline"]["stages"][0]["ratio"] = 0.6
    side.write_bytes(json.dumps(meta, indent=1).encode("utf-8"))
    check = X.check_store_files(reg.stores["syn_text"], reg)
    assert check["meta"]["candidate_spec_sha256"] != X.S0_SPEC_SHA256
    assert not check["identity_ok"]
    with pytest.raises(X.RegisteredInputMismatch):
        X.require_run_allowed("X1", reg)


def test_lfs_pointers_and_line_endings_are_recognised_not_accepted(registry, tmp_path):
    reg = copy_registry(registry, tmp_path)
    spec = reg.stores["syn_doc"]
    lf = (reg.store_dir / "syn_doc.json").read_bytes()
    assert b"\r\n" not in lf
    (reg.store_dir / "syn_doc.json").write_bytes(X.crlf_form(lf))  # as a CRLF checkout would hold it
    check = X.check_store_files(spec, reg)
    assert not check["ok_for_run"] and not check["identity_ok"]  # registered as LF here: other bytes
    crlf_spec = dataclasses.replace(spec, json_sha256=X.sha256_bytes(X.crlf_form(lf)))
    (reg.store_dir / "syn_doc.json").write_bytes(lf)  # the registered hash is of the CRLF form
    check = X.check_store_files(crlf_spec, reg)
    assert check["identity_ok"] and not check["ok_for_run"]
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{spec.npz_sha256}\nsize 1\n"
    (reg.store_dir / "syn_doc.npz").write_bytes(pointer.encode())
    assert X.lfs_pointer_oid(reg.store_dir / "syn_doc.npz") == spec.npz_sha256
    check = X.check_store_files(spec, reg)
    assert check["identity_ok"] and not check["ok_for_run"]
    with pytest.raises(X.RegisteredInputMismatch):
        X.require_run_allowed("X2", reg)


def test_git_blob_hash_ignores_line_endings(tmp_path):
    (tmp_path / "lf.py").write_bytes(b"a = 1\nb = 2\n")
    (tmp_path / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    expected = hashlib.sha1(b"blob 12\0a = 1\nb = 2\n").hexdigest()
    assert X.git_blob_sha1(tmp_path / "lf.py") == X.git_blob_sha1(tmp_path / "crlf.py") == expected


# ------------------------------- 3, 4, 5. budgets, the pool cap, their labels


def test_registered_budget_grid_and_classes():
    for n_pool in (451, 494):
        assert X.registered_budgets(n_pool) == (50, 100, 225, 450, 900, 1800, 3600)
        assert X.pool_capped_budget(n_pool) == 450
        assert [X.budget_class(n, n_pool) for n in X.registered_budgets(n_pool)] == (
            ["pool-supported"] * 4 + ["extrapolated"] * 3)
    assert X.registered_budgets(300) == (50, 100, 225, 300, 450, 900, 1800, 3600)
    assert X.pool_capped_budget(300) == 300
    assert X.budget_class(300, 300) == "pool-supported" and X.budget_class(450, 300) == "extrapolated"
    used = {n for s in X.REGISTERED.stores.values() for n in X.registered_budgets(s.n_queries)}
    assert used == {50, 100, 225, 300, 450, 900, 1800, 3600}  # no other budget may be added


def test_simulated_budgets_are_never_called_measured():
    for n_pool in (300, 451, 494):
        for row in X.budget_table(n_pool):
            assert row["evidence"].startswith("SIMULATED")
            assert (row["evidence"] == X.SIM_POOL) == (row["n"] <= n_pool)
            assert row["pool_capped"] == (row["n"] == X.pool_capped_budget(n_pool))
    for n in (900, 1800, 3600):
        assert all(X.evidence_mark(n, s.n_queries) == X.SIM_EXTRA for s in X.REGISTERED.stores.values())


def test_rule_a_and_b_get_their_registered_budgets_and_seeds():
    reg = X.REGISTERED
    assert {n: reg.ab_seed_count(n, 451) for n in reg.budgets(451)} == {
        50: 2000, 100: 2000, 225: 2000, 450: 500, 900: 500, 1800: 0, 3600: 0}
    assert {n: reg.ab_seed_count(n, 300) for n in reg.budgets(300)} == {
        50: 2000, 100: 2000, 225: 2000, 300: 500, 450: 0, 900: 500, 1800: 0, 3600: 0}


def test_every_simulated_row_carries_its_budget_class(registry, x1_run, x2_run):
    for name, spec in registry.stores.items():
        x2 = X.read_json(x2_run / f"x2_{name}.json")
        x1 = X.read_json(x1_run / f"x1_{name}.json")
        rows = x2["cells"]["rows"] + x1["A3"]["rows"] + x1["A4"]["rows"] + x1["A5"]["rows"]
        assert rows
        for row in rows:
            expected = "pool-supported" if row["n"] <= spec.n_queries else "extrapolated"
            assert row["budget_class"] == expected
            assert row["evidence"] == f"SIMULATED, {expected}"
            assert {"threshold", "delta", "n", "procedure", "construct"} <= set(row)
        assert {r["n"] for r in x2["cells"]["rows"]} == set(registry.budgets(spec.n_queries))
        measured = x2["full_pool"]["rows"]
        assert all(r["evidence"] == "MEASURED" and r["queries"] == spec.n_queries for r in measured)
        assert len(measured) == (3 + 4) * 2  # per construct and threshold, at both error levels
        for row in x2["n_star"]["rows"]:
            assert row["budget_class"] == X.budget_class(row["decided_by_budget"], spec.n_queries)


# --------------------------------------------------------- 6. seed determinism


def test_samples_are_prefixes_of_the_registered_draw():
    idx, nested = X.sample_indices(451, range(4), (50, 225, 3600))
    assert nested
    for s in range(4):
        long = np.random.default_rng(s).integers(0, 451, size=3600)
        for n in (50, 225, 3600):
            assert np.array_equal(idx[n][s], long[:n])
        assert np.array_equal(idx[50][s], np.random.default_rng(s).integers(0, 451, size=50))  # E1's draw
    again, _ = X.sample_indices(451, range(4), (50,))
    assert np.array_equal(again[50], idx[50])
    other, _ = X.sample_indices(451, range(2000, 2004), (50,))
    assert not np.array_equal(other[50], idx[50])
    assert (X.N_SEEDS, X.REPLICATION_SEEDS) == (2000, (2000, 4000))


def test_a_run_is_reproducible_byte_for_byte(registry, x1_run, tmp_path):
    reg = copy_registry(registry, tmp_path)
    assert X1.run(reg)["finished"]
    for name in ("x1_syn_doc.json", "x1_syn_doc_selections.npz", "x1_syn_text.json", "x1_summary.json",
                 "X1_REPORT.md"):
        assert (tmp_path / "out" / "X1" / name).read_bytes() == (x1_run / name).read_bytes(), name


def test_chunked_simulation_equals_one_block(registry):
    st = E.Store(X.load_store("syn_doc", registry), "labels")
    prob = X.problem_for(st)
    idx, _ = X.sample_indices(st.pq.shape[1], range(32), (24, 96), registry)
    one = X.simulate(st.pq, st.base, prob, idx, (0.97, 0.95), i8_index=1, ab_seeds={24: 32, 96: 16})
    chunked = X.simulate(st.pq, st.base, prob, idx, (0.97, 0.95), i8_index=1, ab_seeds={24: 32, 96: 16},
                         chunk_bytes=40 * 24 * 8)  # forces blocks of 8 seeds
    assert set(one) == set(chunked)
    assert all(np.array_equal(one[k], chunked[k]) for k in one)
    assert one["n96_T0.95_A"].shape == (16,) and one["n24_T0.95_A"].shape == (32,)


# ---------------------------------------------- 7. transformation determinism


def test_registered_transformations_and_datasets():
    assert X3.TRANSFORMATIONS == {"T0": None, "T1": 0, "T2a": 1, "T2b": 2, "T3": 3}
    assert X.X3_PRIMARY == ("colpali_docvqa", "colqwen2_infovqa", "colqwen25_docvqa", "scifact")
    assert X.X3_EXTENSION == ("colpali_infovqa", "colqwen2_docvqa", "colqwen25_infovqa")
    assert set(X.X3_PRIMARY) | set(X.X3_EXTENSION) == set(X.REGISTERED.stores)
    assert (X3.INVARIANT, X3.MATERIAL, X3.RATIO, X3.RATE_DIFFERENCE, X3.MAX_ATTEMPTS) == (0.005, 0.02, 2.0, 0.25, 3)
    assert (X.X3_BUDGET, X.X3_BOOTSTRAP) == (225, 2000)


@pytest.mark.parametrize("dim", [128, 96])
def test_maps_are_orthogonal_and_deterministic(dim):
    t1 = X3.signed_permutation(dim, 0)
    assert np.array_equal(np.abs(t1).sum(axis=0), np.ones(dim)) and np.array_equal(np.abs(t1).sum(axis=1), np.ones(dim))
    assert set(np.unique(t1)) <= {-1.0, 0.0, 1.0}
    rotations = [X3.rotation(dim, seed) for seed in (1, 2)]
    for u in rotations:
        assert np.abs(u @ u.T - np.eye(dim)).max() < 1e-12
    assert not np.allclose(rotations[0], rotations[1])
    assert np.array_equal(X3.rotation(dim, 1), rotations[0]) and np.array_equal(X3.signed_permutation(dim, 0), t1)
    # the registered definition: Q of a QR of a standard normal matrix, columns signed by diag(R)
    q, r = np.linalg.qr(np.random.default_rng(1).standard_normal((dim, dim)))
    assert np.array_equal(rotations[0], q * np.sign(np.diag(r)))


def test_maps_keep_scores_norms_and_row_annotations(vectors):
    corpus, queries = vectors.corpus, vectors.queries
    base = maxsim_matrix(queries, corpus)
    same = X3.transform(corpus, queries, "T0")
    assert same[0] is corpus and same[1] is queries and same[2]["sha256"] is None
    for tid in ("T1", "T2a", "T2b", "T3"):
        c, q, record = X3.transform(corpus, queries, tid)
        assert record["seed"] == X3.TRANSFORMATIONS[tid]
        assert record["sha256"] == X3.transform(corpus, queries, tid)[2]["sha256"]
        assert c.vectors.dtype == np.float32 and np.array_equal(c.offsets, corpus.offsets) and c.ids == corpus.ids
        assert np.abs(np.linalg.norm(c.vectors.astype(np.float64), axis=1) - 1.0).max() < 1e-5  # not re-normalised
        assert np.abs(maxsim_matrix(q, c) - base).max() / np.abs(base).max() < X3.SCORE_TOLERANCE
    c, q, _ = X3.transform(corpus, queries, "T1")
    m = X3.signed_permutation(corpus.dimension, 0)
    perm, signs = np.abs(m).argmax(axis=1), m.sum(axis=1)
    assert np.array_equal(c.vectors, corpus.vectors[:, perm] * signs.astype(np.float32))  # exact
    assert np.array_equal(q.vectors, queries.vectors[:, perm] * signs.astype(np.float32))
    c, q, _ = X3.transform(corpus, queries, "T3")
    rows = X3.row_order(corpus.offsets, 3)
    assert q is queries and not np.array_equal(rows, np.arange(rows.size))
    assert np.array_equal(np.sort(rows), np.arange(rows.size))
    for lo, hi in zip(corpus.offsets[:-1], corpus.offsets[1:], strict=True):
        assert rows[lo:hi].min() == lo and rows[lo:hi].max() == hi - 1  # rows stay in their document
    assert np.array_equal(c.vectors, corpus.vectors[rows])
    assert np.array_equal(c.protected, corpus.protected[rows]) and np.array_equal(c.positions, corpus.positions[rows])


# ------------------------------------------ 8. Method C is the frozen Method C


def test_method_c_parameters_are_the_frozen_ones():
    assert S.BET_CAP == 0.9
    config = X.check_method_c()
    assert config["ok"] and config["bet_cap"] == 0.9 and config["chains"] == 7
    assert X.DELTAS == (0.05, 0.01) and X.PRIMARY_DELTA == 0.05
    assert X.THRESHOLDS == {"labels": (0.99, 0.97, 0.95), "agreement": (0.99, 0.97, 0.95, 0.90)}
    assert (X.RATE_THRESHOLD, X.SUBSTANTIAL_COMPRESSION, X.CP1_TOLERANCE, X.CP2_LEVEL) == (0.90, 10.0, 0.01, 0.05)
    assert X.CODE_BLOBS["scripts/research/e1_stats.py"] == "13494307a4a42cd2c1da63f34aa0952a92b33130"
    assert X.PREREG_COMMIT == "73085ab637679f30fffece6632eb7b9e92adff79"


def test_method_c_wrapper_is_the_frozen_rule(rng):
    families = [c["family"] for c in X.generated_spec(128)]
    compression = rng.uniform(2, 300, size=40)
    prob = problem(np.ones(40), families, compression)
    assert prob.F == 7
    for delta in X.DELTAS:
        for _ in range(50):
            p = rng.uniform(0, 3 * delta / 7, size=40)
            rejected = S.fixed_sequence_by_family(p, families, delta / 7)  # level delta / 7, chain order
            assert X.method_c(p, prob, delta) == S.smallest(rejected, prob.bytes, prob.vectors)
    assert X.method_c(np.ones(40), prob, 0.05) is None  # abstains when nothing is rejected
    # a later candidate in a chain is not reachable past a non-rejection, however small it is
    p = np.ones(40)
    p[0], p[2] = 0.0, 0.0
    assert X.method_c(p, prob, 0.05) == 0
    # ties in stored size: fewer vectors, then the earlier position
    tied = E1S.Problem(["a", "b", "c"], [5.0, 5.0, 5.0], [9.0, 4.0, 4.0], [2.0, 2.0, 2.0], np.ones(3))
    assert X.method_c(np.zeros(3), tied, 0.05) == 1


def test_the_engine_reproduces_the_frozen_run_cell(registry, x1_run, x2_run):
    """Gate G4 on synthetic stores: selections equal those of ``e1_1_study.run_cell``, seed by seed."""
    for name in registry.stores:
        gates = X.read_json(x2_run / f"x2_{name}.json")["gates"]
        assert gates["G4"]["passed"] and gates["G4"]["keys_compared"] == 3 * 3 * 2
        assert gates["E1_comparators"]["passed"] and gates["E1_comparators"]["keys_compared"] == 3 * 3 * 10
        assert X.read_json(x1_run / f"x1_{name}.json")["gates"]["G4"]["passed"]


def test_full_pool_decision_is_one_pass_in_stored_order(registry):
    st = E.Store(X.load_store("syn_doc", registry), "agreement")
    prob = X.problem_for(st)
    p = S.betting_pvalues(S.to_y(st.pq, st.base[None, :], 0.9), 0.9 / 1.9)
    expected = S.smallest(S.fixed_sequence_by_family(p, prob.families, 0.05 / 7), prob.bytes, prob.vectors)
    assert X.full_pool_decision(st.pq, st.base, prob, 0.9, 0.05) == expected


# ----------------------------------------------------- decision rules, gates


def test_oracles():
    families = ["a", "a", "a", "b", "b"]
    prob = problem([0.99, 0.94, 0.98, 0.951, 0.96], families, [2, 4, 8, 3, 6])
    assert X.free_oracle(prob, 0.95) == 2  # the smallest candidate at or above the target, anywhere
    assert X.chain_oracle(prob, 0.95) == 4  # chain a stops at its second candidate
    assert X.chain_oracle(prob, 0.96) == 0  # strictly above the target: 0.96 does not pass
    assert X.free_oracle(prob, 0.96) == 2
    assert X.free_oracle(prob, 0.995) is None and X.oracle_compression(prob, None) == 1.0


def test_borderline_rule_uses_extra_seeds_only_when_the_interval_contains_090():
    def never():
        raise AssertionError("the replication block must not be drawn")

    clear = X.decide_rate(np.arange(2000) < 1900, never)  # 0.95: the interval is above 0.90
    assert clear["at_least_0.90"] and not clear["borderline_rule_used"] and clear["seeds"] == 2000
    low = X.decide_rate(np.arange(2000) < 1000, never)
    assert not low["at_least_0.90"] and not low["borderline_rule_used"]
    close = X.decide_rate(np.arange(2000) < 1805, lambda: np.arange(2000) < 1780)
    assert close["borderline_rule_used"] and close["pooled_seeds"] == 4000
    assert close["rate"] > 0.90 and close["deciding_rate"] == (1805 + 1780) / 4000 and not close["at_least_0.90"]


def test_n_star_is_the_smallest_budget_from_which_the_rate_stays_high():
    def decide(rates):
        return lambda n: X.decide_rate(np.arange(1000) < 1000 * rates[n], lambda: np.zeros(1000, bool))

    budgets = (50, 100, 225, 450, 900)
    found = X.n_star(budgets, 451, decide({50: 0.1, 100: 0.99, 225: 0.5, 450: 0.99, 900: 1.0}))
    assert found["n_star"] == 450 and found["budget_class"] == "pool-supported"
    assert set(found["deciding_cells"]) == {"900", "450", "225"}  # nothing below the first failure is used
    late = X.n_star(budgets, 451, decide({50: 0.1, 100: 0.1, 225: 0.1, 450: 0.2, 900: 1.0}))
    assert late["n_star"] == 900 and late["evidence"] == X.SIM_EXTRA
    none = X.n_star(budgets, 451, decide({n: 0.3 for n in budgets}))
    assert none["n_star"] is None and none["decided_by_budget"] == 900 and none["budget_class"] == "extrapolated"


def test_pools_are_summarised_by_their_worst_store():
    rates = {"colpali_docvqa": 0.95, "colqwen2_docvqa": 0.91, "colqwen25_docvqa": 0.99, "colpali_infovqa": 0.97,
             "colqwen2_infovqa": 0.98, "colqwen25_infovqa": 0.80, "scifact": 0.93}
    cells = {n: {"deciding_rate": r, "at_least_0.90": r >= 0.90} for n, r in rates.items()}
    assert X.worst_by_pool(cells, X.REGISTERED) == {
        "DocVQA": {"worst_store": "colqwen2_docvqa", "deciding_rate": 0.91, "at_least_0.90": True},
        "InfoVQA": {"worst_store": "colqwen25_infovqa", "deciding_rate": 0.80, "at_least_0.90": False},
        "SciFact": {"worst_store": "scifact", "deciding_rate": 0.93, "at_least_0.90": True}}


def test_replication_rule_for_a_miss_rate():
    def never():
        raise AssertionError("the replication block must not be drawn")

    fine = X.check_miss_rate(np.arange(2000) < 110, 0.05, never)  # 0.055: the lower end is below 0.05
    assert not fine["replicated"] and not fine["confirmed_failure"]
    cleared = X.check_miss_rate(np.arange(2000) < 130, 0.05, lambda: np.arange(2000) < 80)
    assert cleared["replicated"] and cleared["pooled_seeds"] == 4000 and not cleared["confirmed_failure"]
    failed = X.check_miss_rate(np.arange(2000) < 130, 0.05, lambda: np.arange(2000) < 130)
    assert failed["confirmed_failure"]
    assert X.consistent_with(0.05, 110, 2000) and not X.consistent_with(0.05, 130, 2000)


def test_a_drop_in_compression_on_nested_samples_stops_the_run():
    prob = problem(np.ones(3), ["a", "b", "c"], [2, 4, 8])
    grows = {X.sel_key(n, t, "C@0.05"): np.array(k, np.int16)
             for t in (0.97,) for n, k in ((50, [-1, 0]), (100, [0, 1]), (225, [2, 1]))}
    assert X.check_monotone(grows, prob, (50, 100, 225), (0.97,), True, "t", deltas=(0.05,))["violations"] == 0
    assert not X.check_monotone(grows, prob, (50, 100, 225), (0.97,), False, "t", deltas=(0.05,))["checked"]
    grows[X.sel_key(225, 0.97, "C@0.05")] = np.array([2, 0], np.int16)
    with pytest.raises(X.GateFailure, match="implementation error"):
        X.check_monotone(grows, prob, (50, 100, 225), (0.97,), True, "t", deltas=(0.05,))


def test_value_and_identity_gates(registry):
    d = X.load_store("syn_doc", registry)
    assert X.gate_values(d)["passed"] and X.gate_identity(d)["passed"]
    for key, value in (("cand_labels", np.nan), ("cand_b1", 1.5), ("base_labels", -0.1), ("base_b1", 0.5)):
        bad = {**d, key: d[key].copy()}
        bad[key].flat[0] = value
        assert not X.gate_values(bad)["passed"], key
    lost = {**d, "cand_b1": d["cand_b1"].copy()}
    first = int(np.flatnonzero(d["base_labels"] == 1.0)[0])
    lost["cand_b1"][5, first] = 0.0 if d["cand_labels"][5, first] else 1.0
    assert not X.gate_identity(lost)["passed"]
    with pytest.raises(X.GateFailure, match="gate G3"):
        X.require("G3", X.gate_identity(lost), "syn_doc")


def test_a_failed_reproduction_gate_stops_the_experiment(registry, tmp_path):
    reg = copy_registry(registry, tmp_path)
    path = reg.e1_selections["syn_text"]
    sels = dict(np.load(path))
    sels["n16_T0.95_C@0.05"] = sels["n16_T0.95_C@0.05"].copy()
    sels["n16_T0.95_C@0.05"][3] = 5 if sels["n16_T0.95_C@0.05"][3] != 5 else 6
    np.savez_compressed(path, **sels)
    with pytest.raises(X.GateFailure, match="gate G4"):
        X1.run(reg)
    log = X.read_json(tmp_path / "out" / "X1" / "runs.json")
    assert [e["status"] for e in log] == ["invalid", "invalid"]  # gates come first: no store was analysed
    assert sorted(p.name for p in (tmp_path / "out" / "X1").iterdir()) == ["runs.json"]


def test_an_interrupted_unit_is_logged_as_invalid(tmp_path):
    log = X.RunLog(tmp_path / "runs.json")
    first = log.begin("store", "abc")
    assert first["attempt"] == 1 and log.valid("store") is None
    again = X.RunLog(tmp_path / "runs.json")  # the process died before the unit ended
    assert again.attempts("store")[0]["status"] == "invalid" and "killed" in again.attempts("store")[0]["reason"]
    second = again.begin("store", "abc")
    again.end(second, "valid", outputs={})
    assert second["attempt"] == 2 and X.RunLog(tmp_path / "runs.json").valid("store")["attempt"] == 2
    record = X.Timer().record()
    assert not record["slept"] and abs(record["seconds_not_awake"]) < X.Timer.SLEEP_TOLERANCE


# ------------------------------------------- 9, 10. output schema, provenance


def test_x1_outputs(registry, x1_run):
    assert sorted(p.name for p in x1_run.iterdir()) == [
        "X1_REPORT.md", "runs.json", "x1_provenance.json", "x1_summary.json", "x1_syn_doc.json",
        "x1_syn_doc_selections.npz", "x1_syn_text.json", "x1_syn_text_selections.npz"]
    for name, spec in registry.stores.items():
        out = X.read_json(x1_run / f"x1_{name}.json")
        assert {"header", "store", "budgets", "gates", "A1", "A2", "A3", "A4", "A5", "A6", "H4c", "N10",
                "labels_arm", "checks", "constructs"} <= set(out)
        assert out["A1"]["evidence"] == "MEASURED" and len(out["A1"]["rows"]) == 37
        assert [r["threshold"] for r in out["A2"]["rows"]] == [0.99, 0.97, 0.95, 0.9]
        assert "cp1_holds" in out["A2"]["rows"][0] and "cp1_holds" not in out["A2"]["rows"][3]
        budgets = registry.budgets(spec.n_queries)
        assert len(out["A3"]["rows"]) == 4 * 2 * len(budgets)  # agreement: four thresholds, two levels
        assert len(out["labels_arm"]["rows"]) == 3 * 2 * len(budgets)
        assert len(out["A5"]["rows"]) == 3 * 2 * len(budgets)
        assert sum(r["cp2_cell"] for r in out["A4"]["rows"]) == 3 * len(budgets)
        row = out["A3"]["rows"][0]
        assert {"deployed_rate", "abstain_rate", "nontrivial", "substantial", "compression_median",
                "compression_p10_p90", "compression_mean_log2", "miss", "validity",
                "median_compression_over_chain_oracle", "median_compression_over_free_oracle"} <= set(row)
        assert out["H4c"]["n"] == registry.pool_capped_budget(spec.n_queries)
        assert out["H4c"]["threshold"] == 0.95 and out["H4c"]["budget_class"] == "pool-supported"
        assert "not a measure of retrieval quality" in out["constructs"]["agreement"]
        sel = np.load(x1_run / f"x1_{name}_selections.npz")
        key = f"A_n{budgets[-1]}_T0.9_C@0.05"
        assert sel[key].shape == (registry.n_seeds,) and sel[key].dtype == np.int16
    assert X.read_json(x1_run / "x1_syn_doc.json")["A6"]["computed"]
    assert set(X.read_json(x1_run / "x1_syn_doc.json")["A6"]["margins"]) == {"relevance margin", "top-2 margin"}
    assert not X.read_json(x1_run / "x1_syn_text.json")["A6"]["computed"]  # features exist for ViDoRe only
    summary = X.read_json(x1_run / "x1_summary.json")
    assert set(summary["hypotheses"]) == {"H4a", "H4b", "H4c"}
    assert all(h["verdict"] in ("supported", "fails") for h in summary["hypotheses"].values())
    report = (x1_run / "X1_REPORT.md").read_text(encoding="utf-8")
    assert "It is not retrieval quality" in report and "extrapolated" in report
    assert all(s in report for s in summary["negative_result_sentences"])


def test_x2_outputs(registry, x2_run):
    for name, spec in registry.stores.items():
        out = X.read_json(x2_run / f"x2_{name}.json")
        assert {"header", "store", "budgets", "gates", "cells", "full_pool", "n_star", "H3", "checks"} <= set(out)
        rows, budgets = out["cells"]["rows"], registry.budgets(spec.n_queries)
        cap = registry.pool_capped_budget(spec.n_queries)
        by = {(r["construct"], r["threshold"], r["delta"], r["n"], r["procedure"]) for r in rows}
        assert len(by) == len(rows)
        for procedure in ("P0", "I8", "D", "C-split"):  # every registered budget, both arms and levels
            assert sum(r["procedure"] == procedure for r in rows) == (3 + 4) * 2 * len(budgets)
        a_budgets = {r["n"] for r in rows if r["procedure"] == "A"}
        assert a_budgets == {*registry.e1_budgets, cap, registry.ab_extra_budget}
        assert {r["n"] for r in rows if r["procedure"] == "A-bonf"} == set(registry.e1_budgets)
        seeds = {r["n"]: r["seeds"] for r in rows if r["procedure"] == "B"}
        assert seeds[cap] == registry.ab_extra_seeds and seeds[registry.e1_budgets[0]] == registry.n_seeds
        assert all("validity" in r for r in rows if r["procedure"] in ("P0", "I8", "D", "C-split"))
        assert all("validity" not in r and r["delta"] is None for r in rows if r["procedure"] == "A")
        assert set(out["H3"]) == {"primary", "secondary_T0.99", "secondary_substantial",
                                  "secondary_T0.99_substantial"}
        primary = out["H3"]["primary"]
        assert (primary["construct"], primary["threshold"], primary["delta"], primary["n"]) == ("labels", 0.97, 0.05, cap)
        assert primary["budget_class"] == "pool-supported"
        assert len(out["n_star"]["rows"]) == (3 + 4) * 2 * 2
        assert out["checks"]["monotonicity"]["labels"] == {"checked": True, "violations": 0}
    summary = X.read_json(x2_run / "x2_summary.json")
    h3 = summary["hypotheses"]["H3"]
    assert h3["primary"]["verdict"] in ("supported", "fails") and h3["primary"]["evidence"] == X.SIM_POOL
    assert h3["extension_beyond_the_pools"]["verdict"] in ("holds in simulation", "fails in simulation")
    report = (x2_run / "X2_REPORT.md").read_text(encoding="utf-8")
    assert "(pool-supported)" in report and "(extrapolated)" in report and "MEASURED" in report
    assert all(s in report for s in summary["negative_result_sentences"])


def test_reports_say_which_cells_used_the_borderline_rule(registry, x1_run, x2_run):
    stores = {n: X.read_json(x2_run / f"x2_{n}.json") for n in registry.stores}
    summary = X.read_json(x2_run / "x2_summary.json")
    assert set(summary["hypotheses"]["H3"]["primary"]["by_pool"]) == {"DocVQA", "SciFact"}
    assert "## Cells that used the borderline rule" in X2.report(summary, stores)
    row = stores["syn_doc"]["n_star"]["rows"][0]
    budget, cell = next(iter(row["deciding_cells"].items()))
    cell.update({"borderline_rule_used": True, "pooled_rate": 0.905, "pooled_seeds": 64, "at_least_0.90": True})
    text = X2.report(summary, stores)
    expected = (f"| syn_doc | {row['construct']} | {row['threshold']:.3f} | {row['delta']:.3f} | "
                f"{row['deployment']} | {budget} |")
    assert expected in text and "| 0.905 | yes |" in text
    x1 = X.read_json(x1_run / "x1_summary.json")
    assert set(x1["hypotheses"]["H4c"]["by_pool"]) == {"DocVQA", "SciFact"}
    assert "borderline rule used" in (x1_run / "X1_REPORT.md").read_text(encoding="utf-8")


def test_outputs_carry_their_provenance(registry, x1_run, x2_run):
    for folder, prefix, experiment in ((x1_run, "x1", "X1"), (x2_run, "x2", "X2")):
        for name, spec in registry.stores.items():
            head = X.read_json(folder / f"{prefix}_{name}.json")["header"]
            assert head["experiment"] == experiment and head["registered_configuration"] is False
            assert head["preregistration"]["commit"] == X.PREREG_COMMIT
            assert head["preregistration"]["blob"] == X.PREREG_BLOB
            assert head["method_c"]["bet_cap"] == 0.9 and head["method_c"]["level"] == "delta / number of chains"
            assert head["candidate_spec_sha256"] == X.S0_SPEC_SHA256
            assert head["seeds"]["resamples"] == [0, registry.n_seeds - 1]
            assert head["inputs"][name] == {"npz_sha256": spec.npz_sha256, "json_sha256": spec.json_sha256,
                                            "candidate_spec_sha256": X.S0_SPEC_SHA256}
            assert head["preregistration"]["blob_implemented"] == X.PREREG_BLOB_IMPLEMENTED
            assert head["preregistration"]["amendments"] == ["Amendment 1, 2026-10-04"]
            assert head["code_identity"] == X.code_identity(experiment)
            assert head["auxiliary_inputs"]["e1_selection_files"][name] == X.sha256_file(registry.e1_selections[name])
            assert head["accepted_open_points"] == [f"OP{i}" for i in range(1, 8)]
        provenance = X.read_json(folder / f"{prefix}_provenance.json")
        assert provenance["code_identity"] == X.code_identity(experiment) and provenance["auxiliary"]["ok"]
        assert {"python", "numpy", "scipy", "platform"} <= set(provenance["environment"])
        for file, digest in provenance["outputs"].items():
            assert X.sha256_file(folder / file) == digest
        log = X.read_json(folder / "runs.json")
        assert [e["status"] for e in log] == ["valid", "valid"] and all("started_utc" in e for e in log)


def test_results_stay_inside_their_own_directory(registry):
    assert {k: v.relative_to(ROOT).as_posix() for k, v in X.REGISTERED.out_roots.items()} == {
        "X1": "reports/research/X1", "X2": "reports/research/X2", "X3": "reports/research/X3"}
    assert X.out_path(registry, "X1", "x1_summary.json").parent == pathlib.Path(registry.out_roots["X1"]).resolve()
    for parts in (("..", "E1", "store", "scifact.npz"), ("..", "..", "README.md"), ("..", "X2", "a.json")):
        with pytest.raises(ValueError, match="outside the output directory"):
            X.out_path(registry, "X1", *parts)


def test_written_files_are_deterministic(tmp_path):
    arrays = {"b": np.arange(5, dtype=np.int16), "a": np.ones((2, 3))}
    assert X.write_npz(tmp_path / "1.npz", arrays) == X.write_npz(tmp_path / "2.npz", dict(reversed(arrays.items())))
    assert X.dump_json({"b": 1, "a": np.float64(0.5)}) == b'{\n "a": 0.5,\n "b": 1\n}\n'
    assert X.write_text(tmp_path / "r.md", "a\r\nb\n") == X.sha256_bytes(b"a\nb\n")


# ------------------------- 11, 12. no change to source stores or old artifacts


def test_runs_leave_their_inputs_unchanged(registry, x1_run, x2_run):
    assert X.validate_inputs(registry)["ok_for_run"]  # the store files still have their registered bytes
    for name, spec in registry.stores.items():
        assert X.sha256_file(registry.store_dir / f"{name}.npz") == spec.npz_sha256
        assert X.sha256_file(registry.store_dir / f"{name}.json") == spec.json_sha256


def test_dry_runs_change_nothing_in_the_repository():
    before = research_snapshot()
    assert len(before) > 100
    reports = {m.EXPERIMENT: m.dry_run() for m in (X1, X2, X3)}
    assert research_snapshot() == before  # nothing changed and no result was written
    for rep in reports.values():
        assert rep["validation"]["identity_ok"]
        assert rep["validation"]["generated_candidate_space"]["ok"] and rep["validation"]["code"]["ok"]
        assert rep["output_dir"]["inside_reports_research"]
        assert all(all(s.values()) for s in rep["sampling"].values())
        json.loads(X.dump_json(rep))
    assert reports["X1"]["inputs_ok"] and reports["X2"]["inputs_ok"]
    assert reports["X3"]["transformations_ok"]
    assert set(reports["X3"]["self_test_on_synthetic_vectors"]) == {"96", "128"}


# ------------------------------------------------------------ the run gate


@pytest.mark.parametrize("module", [X1, X2, X3])
def test_a_registered_run_is_refused_while_open_points_are_open(module, monkeypatch):
    monkeypatch.setattr(X, "ACCEPTED_OPEN_POINTS", frozenset())
    assert X.open_points_for(module.EXPERIMENT)
    before = research_snapshot()
    with pytest.raises(X.RunNotAllowed, match="open points not accepted"):
        module.main(["--run"])
    assert research_snapshot() == before  # nothing was computed or written


def test_the_seven_open_points_are_accepted_as_implemented(monkeypatch):
    assert set(X.OPEN_POINTS) == {f"OP{i}" for i in range(1, 8)}
    assert all(set(p) == {"affects", "text", "implemented"} for p in X.OPEN_POINTS.values())
    assert X.ACCEPTED_OPEN_POINTS == frozenset(X.OPEN_POINTS)  # Amendment 1, part 1
    assert all(X.open_points_for(experiment) == {} for experiment in ("X1", "X2", "X3"))
    monkeypatch.setattr(X, "ACCEPTED_OPEN_POINTS", frozenset())  # the gate itself is still there
    assert set(X.open_points_for("X3")) == {"OP1", "OP2", "OP3", "OP4"}
    assert set(X.open_points_for("X2")) == {"OP5"} and set(X.open_points_for("X1")) == {"OP5", "OP6", "OP7"}


def test_a_registered_run_needs_the_amended_text_committed(monkeypatch):
    unamended = {"available": True, "commit": "c" * 40, "branch": "b", "tracked_changes": False,
                 "uncommitted_code": [], "prereg_commit_present": True, "prereg_blob_at_head": X.PREREG_BLOB,
                 "prereg_blob_registered": X.PREREG_BLOB, "main": None}
    monkeypatch.setattr(X, "git_state", lambda root=X.ROOT: unamended)
    with pytest.raises((X.RunNotAllowed, X.RegisteredInputMismatch)) as caught:
        X.require_run_allowed("X2")  # checks only; it runs nothing
    if isinstance(caught.value, X.RunNotAllowed):  # a full checkout: the inputs are fine, the text at HEAD is not
        assert "not the amended text" in str(caught.value)


def test_only_the_registered_registry_can_claim_registration(registry):
    fake = X.Registry(stores=registry.stores, store_dir=registry.store_dir, out_roots=registry.out_roots,
                      registered=True)
    with pytest.raises(X.RunNotAllowed, match="only x_common.REGISTERED"):
        X.require_run_allowed("X1", fake)
    assert X.REGISTERED.registered and not registry.registered
    assert (X.REGISTERED.n_seeds, X.REGISTERED.draw_size, X.REGISTERED.replication_seeds) == (2000, 3600, (2000, 4000))


# --------------------------------------------------------------- Amendment 1


def amendment_parts():
    text = (ROOT / X.PREREG_PATH).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    frozen, _, amendment = text.partition(X.PREREG_FROZEN_UNTIL)
    return frozen + X.PREREG_FROZEN_UNTIL, amendment


def test_the_amendment_only_adds_text_below_the_registered_preregistration():
    frozen, amendment = amendment_parts()
    assert X.sha256_bytes(frozen.encode("utf-8")) == X.PREREG_FROZEN_SHA256  # the registered text, byte for byte
    assert X.git_blob_sha1(ROOT / X.PREREG_PATH) == X.PREREG_BLOB_IMPLEMENTED != X.PREREG_BLOB
    assert X.check_preregistration() == {**X.check_preregistration(), "frozen_part_unchanged": True, "ok": True}
    assert amendment.lstrip().startswith("### Amendment 1, 2026-10-04.")
    for phrase in ("The registered text is unchanged", "before any X1, X2 or X3 computation",
                   "No per-query outcome value was read", "73085ab637679f30fffece6632eb7b9e92adff79",
                   "always reported alongside the primary N10 count", "diagnostic only and does not affect",
                   "The error ends that unit", "for X3 only", "F1 does not apply to XH", "more than 5 seconds",
                   "only when the same code produced them", "No figure is required", "scripts/research/x_common.py",
                   "updated by hand after each actual run", "No other reporting convention is introduced",
                   "provenance and input-integrity checks only"):
        assert phrase in amendment, phrase
    assert all(f"| OP{i} |" in amendment for i in range(1, 8))
    assert all(f"| R{i} |" in amendment for i in range(1, 14)) and "| R14 |" not in amendment
    assert all(f"| F{i} |" in amendment for i in range(1, 6)) and all(f"| O{i} |" in amendment for i in range(1, 5))
    marks = {"**clarification**", "**implementation**", "**diagnostic**", "**amendment**"}
    assert marks <= set(re.findall(r"\*\*[a-z]+\*\*", amendment)) and "**Four kinds of item.**" in amendment
    assert re.findall(r"\| ([A-Z]+\d+) \| amendment \|", amendment) == ["F1"]  # the one changed registered rule


def test_the_hashes_in_the_amendment_are_the_ones_the_harness_checks():
    rows = dict(re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", amendment_parts()[1]))
    assert len(rows) == 7 + 6 + 21
    assert {X.E1_SELECTIONS[n].removeprefix("reports/research/E1/"): h for n, h in X.E1_SELECTION_SHA256.items()} == {
        k: v for k, v in rows.items() if k.endswith("_selections.npz")}
    assert all(rows[f"{name}.npz"] == digest for name, digest in X.Q3_FEATURE_SHA256.items())
    vectors = {file: digest for files in X.VECTOR_SHA256.values() for file, digest in files.items()}
    assert len(vectors) == 21 and all(rows[file] == digest for file, digest in vectors.items())
    assert len(X.E1_SELECTION_SHA256) == 7 and len(X.Q3_FEATURE_SHA256) == 6
    assert set(X.VECTOR_SHA256) == set(X.REGISTERED.stores) == set(X.E1_SELECTION_SHA256)
    assert set(X.Q3_FEATURE_SHA256) == {n for n, s in X.REGISTERED.stores.items() if s.vidore}
    s7b = E.load_s7b()  # the registered paths are the ones the loader reads
    for name, registered in X.VECTOR_SHA256.items():
        assert {os.path.relpath(p, s7b.D).replace(os.sep, "/") for p in X3.vector_files(name, s7b)} == set(registered)


def test_registered_auxiliary_files_have_their_registered_hash():
    for experiment in ("X1", "X2", "X3"):
        aux = X.check_auxiliary(experiment)  # hashes and array headers only
        assert aux["identity_ok"], aux["problems"]
        assert len(aux["e1_selection_files"]) == 7
    assert len(X.check_auxiliary("X1")["q3_feature_files"]) == 6 and not X.check_auxiliary("X2")["q3_feature_files"]


def test_an_auxiliary_file_with_another_hash_stops_the_run(registry, tmp_path):
    reg = copy_registry(registry, tmp_path)
    good = {n: X.sha256_file(reg.e1_selections[n]) for n in reg.stores}
    features = {"syn_doc": X.sha256_file(reg.q3_features / "syn_doc.npz")}
    ok = dataclasses.replace(reg, selection_sha256=good, feature_sha256=features)
    assert X.check_auxiliary("X1", ok)["ok"] and X.check_auxiliary("X2", ok)["ok"]
    for module, field, wrong in ((X1, "selection_sha256", {**good, "syn_text": "0" * 64}),
                                 (X1, "feature_sha256", {"syn_doc": "0" * 64}),
                                 (X2, "selection_sha256", {**good, "syn_doc": "0" * 64})):
        with pytest.raises(X.RegisteredInputMismatch, match="auxiliary input does not match"):
            module.run(dataclasses.replace(ok, **{field: wrong}))
    assert X.check_auxiliary("X2", dataclasses.replace(ok, feature_sha256={"syn_doc": "0" * 64}))["ok"]  # not read by X2
    assert not (tmp_path / "out").exists()  # nothing was computed or written


def test_a_vector_file_with_another_hash_stops_x3(tmp_path, monkeypatch):
    files = {"vectors/a_docs.npz": b"docs", "vectors/a_queries.npz": b"queries", "vectors/a_qrels.json": b"{}"}
    (tmp_path / "vectors").mkdir()
    for rel, content in files.items():
        (tmp_path / rel).write_bytes(content)
    s7b = types.SimpleNamespace(D=str(tmp_path))
    monkeypatch.setattr(X3, "vector_files", lambda name, s: [f"{s.D}/{rel}" for rel in files])
    hashes = {rel: X.sha256_bytes(content) for rel, content in files.items()}
    reg = X.Registry(stores={}, store_dir=tmp_path, out_roots={}, vector_sha256={"a": hashes})
    assert X3.require_vectors("a", reg, s7b) == hashes
    (tmp_path / "vectors" / "a_queries.npz").write_bytes(b"other")
    with pytest.raises(X.RegisteredInputMismatch, match="does not hash to its registered value"):
        X3.require_vectors("a", reg, s7b)
    (tmp_path / "vectors" / "a_queries.npz").unlink()
    with pytest.raises(X.RegisteredInputMismatch, match="missing file"):
        X3.require_vectors("a", reg, s7b)


def test_units_are_combined_only_when_the_same_code_produced_them(registry, tmp_path, monkeypatch):
    reg = copy_registry(registry, tmp_path)
    assert X1.run(reg)["finished"]
    identity = X.code_identity("X1")
    assert set(identity) == {*X.EXPERIMENT_CODE["X1"], "src/optivision", "python", "numpy", "scipy"}
    assert {"scripts/research/x_common.py", "scripts/research/x1_agreement.py", "scripts/research/e1_stats.py",
            "scripts/research/e1_1_study.py", "scripts/research/e1_common.py"} == set(X.EXPERIMENT_CODE["X1"])
    assert "scripts/universal_study/s7b_selection_rules.py" in X.EXPERIMENT_CODE["X3"]
    assert X.read_json(tmp_path / "out" / "X1" / "x1_syn_doc.json")["header"]["code_identity"] == identity
    summary = tmp_path / "out" / "X1" / "x1_summary.json"
    summary.unlink()
    for changed in ("numpy", "scripts/research/x_common.py", "src/optivision"):
        monkeypatch.setattr(X, "code_identity", lambda experiment, root=X.ROOT, c=changed: {**identity, c: "other"})
        out = X.finish_experiment("X1", reg, "x1", X1.summarise, X1.report)
        assert not out["finished"] and len(out["pending"]) == 2
        assert all("produced by other code" in p for p in out["pending"]) and not summary.exists()
    monkeypatch.setattr(X, "code_identity", lambda experiment, root=X.ROOT: dict(identity))
    assert X.finish_experiment("X1", reg, "x1", X1.summarise, X1.report)["finished"] and summary.exists()


def test_n10_diagnostic_count_decides_nothing(registry, x1_run):
    stores = {n: X.read_json(x1_run / f"x1_{n}.json") for n in registry.stores}
    for s in stores.values():
        other = s["N10"]["diagnostic_only"]
        assert other["reading"] == "more than half of the resamples deploy non-trivially"
        assert other["eligible_cells"] >= s["N10"]["eligible_cells"]  # a rate of 0.90 is above one half
        assert set(other["by_delta"]) == set(s["N10"]["by_delta"]) == {"0.05", "0.01"}
    before = X1.summarise(stores, registry)
    for s in stores.values():  # make the diagnostic count as large as it can be
        s["N10"]["diagnostic_only"].update({"eligible_cells": 50, "agreement_at_most_half": 50})
    after = X1.summarise(stores, registry)
    assert after["N10"]["diagnostic_only"]["agreement_at_most_half"] == 100
    assert after["N10"]["applies"] == before["N10"]["applies"]  # no verdict, sentence or decision moves
    assert after["negative_result_sentences"] == before["negative_result_sentences"]
    assert after["hypotheses"] == before["hypotheses"]
    for s in stores.values():  # the count that does decide
        s["N10"].update({"eligible_cells": 10, "agreement_at_most_half": 6})
    decided = X1.summarise(stores, registry)
    assert decided["N10"]["applies"] and X1.N10 in decided["negative_result_sentences"]
    for s in stores.values():  # no eligible cell: N10 does not apply
        s["N10"].update({"eligible_cells": 0, "agreement_at_most_half": 0})
    absent = X1.summarise(stores, registry)
    assert not absent["N10"]["applies"] and X1.N10 not in absent["negative_result_sentences"]
    assert "Diagnostic only, deciding nothing" in (x1_run / "X1_REPORT.md").read_text(encoding="utf-8")
    header = X.read_json(x1_run / "x1_summary.json")["header"]
    for summary in (before, after, decided, absent):  # always beside the primary count, whatever it decides
        primary, other = summary["N10"], summary["N10"]["diagnostic_only"]
        assert other["reading"] == X1.DIAGNOSTIC_READING
        text = X1.report({"header": header, **summary}, stores)
        assert f"N10: in {primary['agreement_at_most_half']} of {primary['eligible_cells']} pool-supported" in text
        assert f"the count is {other['agreement_at_most_half']} of {other['eligible_cells']}." in text


# ------------------------------------------------------------------------ X3


def test_x3_outputs(x3_run):
    out = pathlib.Path(x3_run.out_roots["X3"])
    for tid in X3.TRANSFORMATIONS:
        record = X.read_json(out / "units" / f"syn_vec__{tid}.json")
        assert record["transformation"]["seed"] == X3.TRANSFORMATIONS[tid]
        assert record["evidence"] == ("MEASURED" if tid == "T0" else "MEASURED, transformed vectors")
        assert record["gates"]["candidate_specification"]["passed"] and record["gates"]["values"]["passed"]
        assert len(record["P0"]["rows"]) == 2 * 3  # two budgets, three thresholds, delta = 0.05
        assert {r["n"] for r in record["P0"]["rows"]} == {24, 40}
        assert all(r["budget_class"] == "pool-supported" and r["delta"] == 0.05 for r in record["P0"]["rows"])
        heads = X.npz_headers(out / "store" / f"syn_vec__{tid}.npz")
        assert heads["cand_labels"] == ((40, 40), "float64") and heads["base_b1"] == ((40,), "float64")
        for file, digest in record["outputs"].items():
            assert X.sha256_file(out / file) == digest
    t0 = X.read_json(out / "units" / "syn_vec__T0.json")
    assert t0["gates"]["T0"] == {**t0["gates"]["T0"], "equal": True, "passed": True, "max_abs_difference": 0.0}
    for tid in ("T1", "T2a", "T2b", "T3"):
        gate = X.read_json(out / "units" / f"syn_vec__{tid}.json")["gates"]["symmetry"]
        assert gate["passed"] and gate["max_abs_score_difference_over_max_abs_score"] <= 1e-4
    lists = np.load(out / "lists" / "syn_vec__T0_top10.npz")
    assert lists["reference_top10"].shape == (40, 10) and lists["candidate_top10"].shape == (40, 40, 10)
    result = X.read_json(out / "x3_syn_vec.json")
    assert set(result["pairs"]) == {"T1", "T2a", "T2b", "T3"} and result["set"] == "primary"
    for pair in result["pairs"].values():
        assert len(pair["candidates"]["labels"]) == 40 and len(pair["procedure_outcome"]) == 6
        assert sum(pair["class_counts_distinct"]["labels"].values()) == 37
        assert pair["holds"] == (not pair["material_candidates"] and not pair["procedure_material"])
        assert all(r["class"] == X3.classify(r["delta"], *r["ci95"]) for r in pair["candidates"]["labels"])
    assert len(result["overlap_exploratory"]["rows"]) == 37
    h5 = X.read_json(out / "x3_summary.json")["hypotheses"]["H5"]
    assert set(h5["pairs"]) == {f"syn_vec__{t}" for t in ("T1", "T2a", "T2b", "T3")}
    assert h5["verdict"] in ("supported", "fails") and "undecided" not in json.dumps(h5)
    assert h5["verdict"] == ("fails" if h5["failing_pairs"] else "supported")
    log = X.read_json(out / "attempts.json")
    assert [(e["unit"], e["attempt"], e["status"]) for e in log] == [
        (f"syn_vec__{t}", 1, "valid") for t in X3.TRANSFORMATIONS]


def test_x3_classes_and_decision_rule():
    assert X3.classify(0.005, -0.01, 0.02) == "invariant" and X3.classify(-0.004, -0.01, -0.001) == "invariant"
    assert X3.classify(0.03, 0.01, 0.05) == "material" and X3.classify(-0.02, -0.05, -0.001) == "material"
    assert X3.classify(0.03, -0.01, 0.05) == "small"  # large, but the interval contains 0
    assert X3.classify(0.01, 0.005, 0.015) == "small"  # the interval excludes 0, but below 0.02
    status = {"a__T1": "holds", "a__T2a": "holds"}
    assert X3.summarise({}, status, X.REGISTERED)["hypotheses"]["H5"]["verdict"] == "supported"
    status["a__T3"] = "incomplete"
    assert X3.summarise({}, status, X.REGISTERED)["hypotheses"]["H5"]["verdict"] == "not assessable"
    pair = {"material_candidate_labels": ["binary"], "procedure_material": True}
    out = X3.summarise({"a": {"pairs": {"T2b": pair, "T1": pair}}},
                       {**status, "a__T2b": "does not hold", "a__T1": "does not hold"}, X.REGISTERED)
    assert out["hypotheses"]["H5"]["verdict"] == "fails"  # a failure is a failure, whatever is incomplete
    order = "changes with the order of coordinates or of vectors."
    assert out["negative_result_sentences"] == [
        f"The implementation's result for candidate `binary` on dataset a {order}",
        f"The implementation's result for the outcome of P0 on dataset a {order}",
        "The result for candidate `binary` on dataset a is not invariant under the registered rotation.",
        "The result for the outcome of P0 on dataset a is not invariant under the registered rotation."]


def test_x3_paired_bootstrap_interval(rng):
    n = 300
    base = rng.choice([1.0, 0.5, 0.0], size=n, p=[0.6, 0.2, 0.2])
    t0 = {"base_labels": base, "cand_labels": np.stack([base, base * (rng.random(n) > 0.1)]),
          "base_b1": np.ones(n), "cand_b1": np.ones((2, n))}
    t = {**t0, "cand_labels": np.stack([base, base * (rng.random(n) > 0.4)])}
    rows = X3.candidate_deltas(t, t0, 2000)["labels"]
    assert rows[0]["delta"] == 0.0 and rows[0]["ci95"] == [0.0, 0.0] and rows[0]["class"] == "invariant"
    assert rows[1]["class"] == "material" and rows[1]["ci95"][1] < 0.0 and rows[1]["delta"] <= -0.02
    idx = np.random.default_rng(0).integers(0, n, size=(2000, n))  # seed 0, the library's own draw
    boot = (t["cand_labels"][1][idx].sum(axis=1) - t0["cand_labels"][1][idx].sum(axis=1)) / base[idx].sum(axis=1)
    assert np.allclose(rows[1]["ci95"], np.quantile(boot, [0.025, 0.975]))
    assert X3.candidate_deltas(t, t0, 2000)["agreement"][1]["delta"] == 0.0


def test_x3_t0_gate(x3_run, vectors):
    committed = X.load_store("syn_vec", x3_run)
    arrays = {k: committed[k].copy() for k in ("base_labels", "base_b1", "cand_labels", "cand_b1")}
    assert X3.t0_gate(arrays, committed["meta"], committed) == {
        "candidate_specification_equal": True, "size_metadata_equal": True, "arrays_equal": True, "equal": True,
        "max_abs_difference": 0.0, "passed": True}
    nudged = {**arrays, "cand_labels": arrays["cand_labels"].copy()}
    nudged["cand_labels"][4, 0] += 1e-9  # one entry in 1,600: the environment changed a rounding
    gate = X3.t0_gate(nudged, committed["meta"], committed)
    assert not gate["equal"] and gate["fallback"]["passed"] and gate["passed"]
    broken = {**arrays, "cand_labels": arrays["cand_labels"].copy()}
    broken["cand_labels"][4, :20] = 0.0  # half of a candidate's queries
    gate = X3.t0_gate(broken, committed["meta"], committed)
    assert not gate["passed"] and gate["fallback"]["share_of_identical_entries"]["cand_labels"] < 0.995


def test_x3_stops_when_a_map_does_not_keep_the_scores(tmp_path, vectors, monkeypatch):
    reg = vector_registry(tmp_path, vectors)
    monkeypatch.setattr(X3, "rotation", lambda dim, seed: 1.5 * np.eye(dim))  # not an orthogonal map
    with pytest.raises(X.GateFailure, match="symmetry check failed"):
        X3.run(reg, loader=lambda name: (vectors, "ndcg@5", {}))
    log = X.read_json(tmp_path / "out" / "X3" / "attempts.json")
    assert [(e["unit"], e["status"]) for e in log] == [
        ("syn_vec__T0", "valid"), ("syn_vec__T1", "valid"), ("syn_vec__T2a", "invalid")]
    assert not log[2]["gates"]["symmetry"]["passed"]
    assert not (tmp_path / "out" / "X3" / "store" / "syn_vec__T2a.npz").exists()
    assert not (tmp_path / "out" / "X3" / "x3_summary.json").exists()  # no verdict from a stopped run


def test_x3_gives_a_pair_three_attempts_and_then_reports_it_incomplete(tmp_path, vectors, monkeypatch):
    reg = vector_registry(tmp_path, vectors)
    log = X.RunLog(X.out_path(reg, "X3", "attempts.json"))
    for _ in range(X3.MAX_ATTEMPTS):
        log.end(log.begin("syn_vec__T2b", "abc"), "invalid", reason="the machine slept during the run")
    out = X3.run(reg, loader=lambda name: (vectors, "ndcg@5", {}))
    h5 = out["summary"]["hypotheses"]["H5"]
    assert h5["pairs"]["syn_vec__T2b"] == "incomplete" and h5["incomplete_pairs"] == ["syn_vec__T2b"]
    assert h5["verdict"] in ("not assessable", "fails")
    assert len(X.RunLog(X.out_path(reg, "X3", "attempts.json")).attempts("syn_vec__T2b")) == 3  # not run again


def test_x3_extension_waits_for_the_primary_set(tmp_path, vectors):
    reg = vector_registry(tmp_path, vectors, extension=("syn_ext",))
    with pytest.raises(X.RunNotAllowed, match="only after the whole primary set"):
        X3.run(reg, datasets=["syn_ext"], loader=lambda name: (vectors, "ndcg@5", {}))
    assert not (tmp_path / "out" / "X3" / "units").exists()


def test_x3_candidate_that_fails_to_run_ends_the_unit_and_counts_as_an_attempt(tmp_path, vectors, monkeypatch):
    """Amendment 1, F1 and F2: no per-candidate handling; three attempts; then the pair is incomplete."""
    reg = vector_registry(tmp_path, vectors)
    real = X3.score_space

    def failing(corpus, queries, space, progress=None):
        if corpus is vectors.corpus:  # the identity run is left alone
            return real(corpus, queries, space, progress=progress)
        scored = []

        def stop_at_the_sixth(sc):
            progress(sc)
            scored.append(sc.label)
            if len(scored) == 5:
                raise RuntimeError("the sixth candidate could not be fitted")

        return real(corpus, queries, space, progress=stop_at_the_sixth)

    def loader(name):
        return vectors, "ndcg@5", {}

    monkeypatch.setattr(X3, "score_space", failing)
    for attempt in (1, 2, 3):
        with pytest.raises(RuntimeError, match="could not be fitted"):  # the error ends the unit, and the run
            X3.run(reg, loader=loader)
        last = X.read_json(tmp_path / "out" / "X3" / "attempts.json")[-1]
        assert (last["unit"], last["attempt"], last["status"]) == ("syn_vec__T1", attempt, "invalid")
        assert "a candidate failed to run after 5 of 40 were scored: RuntimeError" in last["reason"]
    assert not (tmp_path / "out" / "X3" / "store" / "syn_vec__T1.npz").exists()  # no store without all 40
    monkeypatch.setattr(X3, "score_space", real)
    out = X3.run(reg, loader=loader)  # T1 has used its three attempts and is not tried again
    log = X.read_json(tmp_path / "out" / "X3" / "attempts.json")
    assert sum(e["unit"] == "syn_vec__T1" for e in log) == 3
    h5 = out["summary"]["hypotheses"]["H5"]
    assert h5["pairs"]["syn_vec__T1"] == "incomplete" and h5["incomplete_pairs"] == ["syn_vec__T1"]
    assert h5["verdict"] == ("fails" if h5["failing_pairs"] else "not assessable")  # never "supported"


def test_x3_uses_only_units_produced_by_the_same_code(x3_run, tmp_path, vectors, monkeypatch):
    log = X.RunLog(X.out_path(x3_run, "X3", "attempts.json"))
    identity = X.code_identity("X3")
    record = X3.unit_record("syn_vec", "T1", x3_run, log, identity)
    assert record is not None and record["header"]["code_identity"] == identity
    other = {**identity, "scripts/research/x3_symmetry.py": "0" * 40}
    assert X3.unit_record("syn_vec", "T1", x3_run, log, other) is None
    assert not X3.exhausted("syn_vec", "T1", x3_run, log, other)  # one attempt so far: it would be repeated
    reg = vector_registry(tmp_path, vectors)

    def loader(name):
        return vectors, "ndcg@5", {}

    assert X3.run(reg, loader=loader)["finished"]
    monkeypatch.setattr(X, "code_identity", lambda experiment, root=X.ROOT: other)
    assert X3.analyse_dataset("syn_vec", reg, X.RunLog(X.out_path(reg, "X3", "attempts.json"))) is None
    out = X3.finish(reg, X.RunLog(X.out_path(reg, "X3", "attempts.json")))
    assert not out["finished"] and len(out["pending"]) == 4  # units of other code are not combined
    assert X3.run(reg, loader=loader)["finished"]  # each unit is repeated, and the repeat is an attempt
    attempts = X.read_json(tmp_path / "out" / "X3" / "attempts.json")
    assert [(e["unit"], e["attempt"], e["status"]) for e in attempts[5:]] == [
        (f"syn_vec__{t}", 2, "valid") for t in X3.TRANSFORMATIONS]
