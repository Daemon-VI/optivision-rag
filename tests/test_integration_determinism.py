"""Determinism of the new layers: same input and same configuration, same output.

``compute`` below (a source string, so that the very same code runs here and in a fresh
interpreter) builds seeded synthetic data and returns a digest for every public result of the
new layers: ``corpus_identity``, the fitted state of every stage (``fitted_state_digest``), the
codes and decoded vectors of ``Pipeline.compress``, query transforms, ``ByteAccounting``,
``pipeline_traits``, the candidate-space identity, ``Evidence.digest()`` (both references),
``select()`` decisions, ``Ranking`` objects from the exact, the compressed and the NumPy-backend
paths, a certificate and the manifest of the artifact bundle written for it.

Three runs are compared: twice in this process and once in a fresh subprocess whose
``PYTHONHASHSEED`` differs (the hash seed changes the iteration order of every ``set`` of
strings, which is where a hidden dependence on ordering would show). Equality is exact; the
digests are of bytes.

A fresh interpreter proves reproducibility on one machine with one NumPy and one BLAS; it says
nothing about other platforms, which is outside what these tests can claim.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import optivision
from optivision.core.identity import array_digest, canonical_json, digest, strings_digest
from optivision.core.vectors import corpus_identity
from optivision.representation import MultiVectorCorpus

from .test_integration_layers import build_corpus

SCRIPT = r'''
import json, os, tempfile
import numpy as np

from optivision.api import certify
from optivision.backends import NumpyBackend, NumpyIndex
from optivision.compose import Pipeline
from optivision.core.evaluate import Evidence, LabelReference, RankingReference
from optivision.core.evaluate.collect import collect_evidence
from optivision.core.identity import array_digest, digest
from optivision.core.pipeline import pipeline_identity
from optivision.core.retrieve import CompressedRetriever, ExactRetriever
from optivision.core.selection import (BonferroniBetting, CandidateSpace, FixedSequenceBetting,
                                       select)
from optivision.core.similarity import maxsim_matrix
from optivision.core.stages import (RotationProjector, byte_accounting, fitted_state_digest,
                                    pipeline_traits)
from optivision.core.vectors import corpus_identity
from optivision.representation import MultiVectorCorpus
from optivision.stages import (AdaptiveMerge, BinaryQuantizer, Float16Quantizer, Int4Quantizer,
                               Int8Quantizer, Lloyd2Quantizer, PCAProjector, RandomProjector,
                               RandomPruner, TruncateProjector)
from optivision.storage import to_legacy_pages
from optivision.testing import make_corpus, make_queries


def build(seed, dim, dtype):
    base = make_corpus(24, dim, tokens=(3, 9), seed=seed, dtype=dtype, anisotropy=0.7)
    arrays = [np.asarray(base.vectors[lo:hi]) for lo, hi in (base.span(i) for i in range(24))]
    arrays[1] = arrays[1][:1]
    arrays[2] = arrays[2][:0]
    return MultiVectorCorpus.from_arrays(arrays, ids=[f"{i}::p1" for i in base.ids])


PIPES = {
    "empty": lambda: Pipeline([]),
    "float16": lambda: Pipeline([Float16Quantizer()]),
    "merge+int8": lambda: Pipeline([AdaptiveMerge(radius=0.5, center="mean"),
                                    Int8Quantizer("per_vector")]),
    "prune+binary-centred": lambda: Pipeline([RandomPruner(0.5, seed=2),
                                              BinaryQuantizer(center="mean")]),
    "pca+binary": lambda: Pipeline([PCAProjector(8), BinaryQuantizer()]),
    "rotation+int4": lambda: Pipeline([RotationProjector(seed=3), Int4Quantizer()]),
    "randproj+truncate": lambda: Pipeline([RandomProjector(11, seed=2), TruncateProjector(7),
                                           Int8Quantizer("per_dimension")]),
    "lloyd2": lambda: Pipeline([Lloyd2Quantizer(seed=7)]),
}


def ranking_digest(r):
    return digest({"ids": r.ids, "scores": array_digest(r.scores), "kind": r.score_kind,
                   "tie": r.tie_break,
                   "indices": None if r.indices is None else array_digest(r.indices)})


def compute(seed):
    out = {}
    for dtype_name in ("float32", "float16"):
        corpus = build(seed, 21, np.dtype(dtype_name))
        queries, qrels = make_queries(corpus, 10, tokens=(3, 6), noise=0.3, seed=seed + 100)
        tag = dtype_name
        out[f"{tag}/corpus_identity"] = corpus_identity(corpus).digest()
        out[f"{tag}/queries_identity"] = corpus_identity(queries).digest()
        for name, make in PIPES.items():
            p = make()
            c = p.compress(corpus)
            key = f"{tag}/{name}"
            out[key + "/pipeline_identity"] = pipeline_identity(p)
            out[key + "/fitted_state"] = digest([fitted_state_digest(s) for s in p.stages])
            out[key + "/codes"] = array_digest(c.codes)
            out[key + "/offsets"] = array_digest(c.offsets)
            out[key + "/decoded"] = array_digest(c.decode_rows(0, c.num_vectors))
            projected = p.transform_queries(queries)
            out[key + "/queries"] = array_digest(projected.vectors)
            out[key + "/scores"] = array_digest(maxsim_matrix(projected, c))
            out[key + "/ranking"] = ranking_digest(CompressedRetriever(c, p).search(queries, 7))
            out[key + "/accounting"] = digest(byte_accounting(c, p).to_dict())
            out[key + "/traits"] = digest(pipeline_traits(p).to_dict())
        out[f"{tag}/exact_ranking"] = ranking_digest(ExactRetriever(corpus).search(queries, 7))

    corpus = build(seed, 21, np.float32)
    queries, qrels = make_queries(corpus, 20, tokens=(3, 6), noise=0.3, seed=seed + 100)
    space = CandidateSpace.from_families({
        "float16": [Pipeline([Float16Quantizer()])],
        "merge+int8": [Pipeline([Int8Quantizer("per_vector")]),
                       Pipeline([AdaptiveMerge(radius=0.7, center="mean"),
                                 Int8Quantizer("per_vector")])],
        "pca+binary": [Pipeline([BinaryQuantizer()]), Pipeline([PCAProjector(10), BinaryQuantizer()])],
        "rotation+int4": [Pipeline([RotationProjector(seed=1), Int4Quantizer()])],
    })
    out["space/identity"] = space.identity
    out["space/dict"] = digest(space.to_dict())
    try:
        out["space/default"] = CandidateSpace.default(128).identity
    except ImportError:
        out["space/default"] = "no scipy"
    for label, reference in (("labels", LabelReference(qrels)), ("agreement", RankingReference())):
        ev = collect_evidence(corpus, queries, space, reference)
        out[f"evidence/{label}"] = ev.digest()
        out[f"evidence/{label}/fitted"] = digest([list(e) for e in ev.fitted])
        for target, error in ((0.5, 0.2), (0.9, 0.05)):
            for rule in (FixedSequenceBetting(), BonferroniBetting()):
                d = select(space, ev, target=target, error=error, rule=rule)
                out[f"decision/{label}/{rule.name}/{target}/{error}"] = digest(d.to_dict())
        d = select(space, ev, target=0.5, error=0.2, order_seed=11, query_budget=15)
        out[f"decision/{label}/seeded"] = digest(d.to_dict())

    rng = np.random.default_rng(seed)
    b = rng.random(60)
    cand = np.clip(b[None, :] * rng.random((6, 60)) ** 0.1, 0, 1)
    labels = ["x", "x", "y", "y", "z", "z"]
    from optivision.core.selection import CandidateSpec
    syn = CandidateSpace("syn", tuple(CandidateSpec(k, labels[k], {"stages": [{"stage": "float16"}]})
                                       for k in range(6)))
    ev = Evidence("ndcg@5", "labels", b, cand, tuple(f"q{i}" for i in range(60)),
                  np.array([6., 5, 4, 3, 2, 1]), np.ones(6), syn.identity)
    out["synthetic/evidence"] = ev.digest()
    out["synthetic/decision"] = digest(select(syn, ev, target=0.8, error=0.1).to_dict())

    with tempfile.TemporaryDirectory() as tmp:
        store = Pipeline([BinaryQuantizer()]).compress(corpus)
        backend = NumpyBackend(NumpyIndex(os.path.join(tmp, "idx"), dim=21, method="binary"))
        backend.add(to_legacy_pages(store))
        out["backend/ranking"] = ranking_digest(backend.search_many(queries, 7))
        result = certify(corpus, queries, qrels, space=space, target=0.5, error=0.2,
                         created="2026-01-01T00:00:00Z")
        out["certify/outcome"] = result.outcome.value
        out["certify/certificate"] = result.certificate.digest
        out["certify/evidence"] = result.evidence.digest()
        root = result.save(os.path.join(tmp, "bundle"))
        import hashlib
        for name in sorted(os.listdir(root)):
            out["bundle/" + name] = hashlib.sha256(open(os.path.join(root, name), "rb").read()).hexdigest()
    return out
'''


def run(seed: int) -> dict[str, str]:
    """``compute(seed)`` in this process."""
    namespace: dict = {}
    exec(compile(SCRIPT, "<determinism>", "exec"), namespace)  # noqa: S102 - our own source
    return namespace["compute"](seed)


def src_directory() -> Path:
    return Path(optivision.__file__).resolve().parents[1]


def run_in_subprocess(seed: int, hash_seed: str) -> tuple[dict[str, str], str]:
    """The same ``compute(seed)`` in a fresh interpreter with ``PYTHONHASHSEED=hash_seed``."""
    code = SCRIPT + (
        "\nimport json as _json\n"
        f"print(_json.dumps({{'hash': hash('hash-seed-probe'), 'out': compute({seed})}}))\n")
    env = {k: v for k, v in os.environ.items() if k != "OPTIVISION_SCORE_DEVICE"}
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = str(src_directory())
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=600, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    payload = json.loads(done.stdout.strip().splitlines()[-1])
    return payload["out"], str(payload["hash"])


@pytest.fixture(scope="module")
def first_run():
    return run(0)


class TestSameInputSameOutput:
    def test_two_runs_in_one_process_are_identical(self, first_run):
        second = run(0)
        assert second.keys() == first_run.keys()
        different = [k for k in first_run if first_run[k] != second[k]]
        assert not different, different
        assert len(first_run) > 150  # the table is large, so a silent truncation would show

    def test_a_fresh_interpreter_with_another_hash_seed_gives_the_same_digests(self, first_run):
        here = str(hash("hash-seed-probe"))
        tested = 0
        for hash_seed in ("1", "424242"):
            there, there_hash = run_in_subprocess(0, hash_seed)
            if there_hash == here:  # the (astronomically unlikely) same seed: nothing was tested
                continue
            tested += 1
            assert there.keys() == first_run.keys()
            different = [k for k in first_run if first_run[k] != there[k]]
            assert not different, (hash_seed, different)
        assert tested >= 1

    def test_another_seed_changes_the_data_dependent_digests(self, first_run):
        """The comparison above is not vacuous: other data give other digests everywhere except
        for the digests of a configuration (pipeline identity, traits, candidate space) and of
        fitted state that does not depend on the data (a seeded rotation, no state at all)."""
        other = run(1)
        assert other.keys() == first_run.keys()
        same = {k for k in first_run if first_run[k] == other[k]}
        configuration = ("pipeline_identity", "/traits", "space/", "certify/outcome",
                         "fitted_state", "/fitted")
        unexpected = sorted(k for k in same if not any(part in k for part in configuration))
        assert not unexpected, unexpected
        assert first_run["float32/corpus_identity"] != other["float32/corpus_identity"]
        assert first_run["evidence/labels"] != other["evidence/labels"]
        assert len(first_run) - len(same) > 120

    def test_the_default_space_digest_is_the_registered_one_in_every_process(self, first_run):
        if first_run["space/default"] == "no scipy":
            pytest.skip("SciPy is not installed")
        assert first_run["space/default"] == (
            "af29b01146a90a048ffea9cfbd66f0fedf988134829cd4dc56861f1207893520")


class TestIdentitiesDoNotDependOnHowTheBytesAreHeld:
    """Digests are functions of dtype, shape and values only."""

    def test_array_digest_ignores_memory_layout_and_byte_order(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((7, 5)).astype(np.float32)
        variants = [
            np.asfortranarray(a),
            np.ascontiguousarray(a),
            np.concatenate([a, a])[:7],  # a view of a larger array
            np.pad(a, ((0, 0), (0, 3)))[:, :5],  # a strided view
            a.astype(a.dtype.newbyteorder(">")),  # big-endian bytes, same values
            a.copy(),
        ]
        assert len({array_digest(v) for v in variants}) == 1
        assert array_digest(a) != array_digest(a.astype(np.float64))  # dtype is part of it
        assert array_digest(a) != array_digest(a.reshape(5, 7))  # so is the shape
        assert array_digest(a) != array_digest(a.T.copy())  # and the order of the values
        assert array_digest(np.zeros(3)) != array_digest(np.zeros(3) * -1.0)  # 0.0 is not -0.0

    def test_canonical_json_sorts_keys_and_is_independent_of_insertion_order(self):
        a = {"b": [1, 2, {"y": 1.5, "x": None}], "a": "text", "c": {"k2": True, "k1": False}}
        b = {"c": {"k1": False, "k2": True}, "a": "text", "b": [1, 2, {"x": None, "y": 1.5}]}
        assert canonical_json(a) == canonical_json(b)
        assert canonical_json(a) == json.dumps(a, sort_keys=True, separators=(",", ":")).encode()
        assert canonical_json(a).index(b'"a"') < canonical_json(a).index(b'"b"') \
            < canonical_json(a).index(b'"c"')
        assert digest(a) == digest(b)
        assert canonical_json({"v": np.float32(0.5), "n": np.int64(3), "t": (1, 2)}) == \
            canonical_json({"t": [1, 2], "n": 3, "v": 0.5})

    def test_strings_digest_depends_on_the_order_of_the_items(self):
        assert strings_digest(["a", "b"]) != strings_digest(["b", "a"])
        assert strings_digest(["a", "b"]) == strings_digest(("a", "b"))
        assert strings_digest(["ab"]) != strings_digest(["a", "b"])

    def test_corpus_identity_ignores_what_is_not_the_corpus(self):
        a = build_corpus(0, 13)
        b = MultiVectorCorpus(np.asfortranarray(a.vectors), a.offsets.copy(), ids=list(a.ids),
                              metadata=[{"x": 1}] * len(a), attrs={"anything": "else"})
        assert corpus_identity(a) == corpus_identity(b)
        assert corpus_identity(a).digest() == corpus_identity(b).digest()
        c = MultiVectorCorpus(a.vectors, a.offsets, ids=[*a.ids[:-1], "renamed"])
        assert corpus_identity(a).ids_sha256 != corpus_identity(c).ids_sha256
        assert corpus_identity(a).vectors_sha256 == corpus_identity(c).vectors_sha256
