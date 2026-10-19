"""scripts/vector_geometry.py: the statistics separate vectors plain int4 handles from those it does not."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from optivision.representation import MultiVectorCorpus

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vector_geometry.py"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="needs the repository's scripts")


def _module():
    spec = importlib.util.spec_from_file_location("vector_geometry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _sets(dim: int = 128, n: int = 3000):
    rng = np.random.default_rng(0)
    iso = _unit(rng.standard_normal((n, dim)))
    spike = rng.standard_normal((n, dim))
    spike[:, 0] += 40.0  # one coordinate dominates every vector: plain int4's per-vector scale is set by it
    offset = rng.standard_normal((n, dim)) * 0.15 + 1.0  # a large shared offset on every coordinate
    return iso, _unit(spike), _unit(offset)


def test_a_dominant_coordinate_breaks_plain_int4_and_centring_repairs_it():
    g = _module().geometry
    iso, spike, _ = _sets()
    a, b = g(iso, iso[:500]), g(spike, spike[:500])
    assert b["peak_to_rms"]["median"] > 3 * a["peak_to_rms"]["median"]
    assert b["second_moment_share"]["top1"] > 0.5 > a["second_moment_share"]["top1"]
    plain, centred = b["codecs"]["int4"]["score_pearson"]["median"], b["codecs"]["int4(mean)"]["score_pearson"]["median"]
    assert plain < 0.9 < centred
    assert a["codecs"]["int4"]["score_pearson"]["median"] > 0.95
    assert a["codecs"]["int8(per_vector)"]["score_pearson"]["median"] > 0.99


def test_a_shared_offset_shows_in_the_mean_and_hurts_binary_more_than_centred_binary():
    g = _module().geometry
    iso, _, offset = _sets()
    a, c = g(iso, iso[:500]), g(offset, offset[:500])
    assert c["mean_vector_norm"] > 0.9 and a["mean_vector_norm"] < 0.1
    assert c["doc_cos_to_mean"]["median"] > 0.9
    assert c["codecs"]["binary"]["score_pearson"]["median"] < c["codecs"]["binary(mean)"]["score_pearson"]["median"]
    assert "score_rel_rmse" in c["codecs"]["int4"] and "score_rel_rmse" not in c["codecs"]["binary"]


def test_the_script_reads_encoded_vectors_and_writes_one_json(tmp_path, monkeypatch):
    iso, _, _ = _sets(dim=32, n=400)
    vectors = tmp_path / "data" / "vectors"
    vectors.mkdir(parents=True)
    offsets = np.arange(0, 401, 20)
    MultiVectorCorpus(iso, offsets).save(vectors / "toy_set_docs.npz")
    MultiVectorCorpus(iso[:60], np.arange(0, 61, 6)).save(vectors / "toy_set_queries.npz")
    monkeypatch.setenv("OPTIVISION_DATA", str(tmp_path / "data"))
    monkeypatch.setattr("sys.argv", ["vector_geometry.py", "vec:toy_set", "--out", str(tmp_path / "geo"), "--sample", "300"])
    assert _module().main() == 0
    written = json.loads((tmp_path / "geo" / "toy_set.json").read_text(encoding="utf-8"))
    assert written["tag"] == "toy_set" and written["n_docs"] == 20 and written["doc_vectors"] == 300
    assert set(written["codecs"]) == {"int8(per_vector)", "int4", "int4(mean)", "binary", "binary(mean)"}
