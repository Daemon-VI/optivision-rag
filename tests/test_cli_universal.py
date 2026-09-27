from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from optivision.cli import app
from optivision.cli_universal import parse_pipeline
from optivision.representation import MultiVectorCorpus

pytestmark = [pytest.mark.filterwarnings("ignore:calibrating on"), pytest.mark.filterwarnings("ignore:Int8Quantizer")]

runner = CliRunner()


def _unit(v):
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(5)
    topics = rng.standard_normal((20, 16))
    docs, queries, qrels = [], [], {}
    for i in range(16):
        v = _unit(topics[np.repeat(rng.choice(20, 3, replace=False), 8)] + 0.3 * rng.standard_normal((24, 16)))
        docs.append(v)
        queries.append(_unit(v[rng.choice(24, 4, replace=False)] + 0.2 * rng.standard_normal((4, 16))))
        qrels[f"q{i}"] = [f"d{i}"]
    corpus = MultiVectorCorpus.from_arrays(docs, ids=[f"d{i}" for i in range(16)]).save(tmp_path / "docs.npz")
    qpath = MultiVectorCorpus.from_arrays(queries, ids=[f"q{i}" for i in range(16)]).save(tmp_path / "queries.npz")
    labels = tmp_path / "qrels.json"
    labels.write_text(json.dumps(qrels), encoding="utf-8")
    return str(corpus), str(qpath), str(labels), tmp_path


def test_parse_pipeline_shorthand_and_json(tmp_path):
    p = parse_pipeline("adaptive_merge(radius=0.8, ratio=null) > int8(scale=per_vector)")
    assert [s.name for s in p.stages] == ["adaptive_merge", "int8"]
    assert p.stages[0].radius == 0.8 and p.stages[0].ratio is None
    assert p.stages[1].scale == "per_vector"
    assert parse_pipeline(json.dumps(p.to_dict())).to_dict() == p.to_dict()
    assert parse_pipeline("float32").stages == []
    spec = tmp_path / "p.json"
    spec.write_text(json.dumps(p.to_dict()), encoding="utf-8")
    assert parse_pipeline(str(spec)).to_dict() == p.to_dict()


def test_inspect_and_compress(files):
    corpus, _, _, tmp = files
    r = runner.invoke(app, ["inspect", corpus])
    assert r.exit_code == 0, r.output
    assert "num_vectors" in r.output
    out = tmp / "c.npz"
    r = runner.invoke(app, ["compress", corpus, "-p", "adaptive_merge(radius=0.7) > binary", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()
    r = runner.invoke(app, ["inspect", str(out)])
    assert r.exit_code == 0 and "compression_vs_float32" in r.output


def test_calibrate_writes_a_result(files):
    corpus, queries, qrels, tmp = files
    out = tmp / "cal.json"
    r = runner.invoke(app, ["calibrate", corpus, "-q", queries, "--qrels", qrels, "--target", "0.9",
                            "--out", str(out)])
    assert r.exit_code == 0, r.output
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["target"] == 0.9 and result["n_holdout_queries"] == 8
    assert "Baseline" in r.output


def test_compare_and_benchmark(files):
    corpus, queries, qrels, tmp = files
    r = runner.invoke(app, ["compare", corpus, "-q", queries, "--qrels", qrels,
                            "-p", "int8", "-p", "adaptive_merge(radius=0.6) > binary"])
    assert r.exit_code == 0, r.output
    assert "retention" in r.output
    r = runner.invoke(app, ["benchmark", corpus, "-q", queries, "--qrels", qrels, "--suite", "quantize",
                            "--out", str(tmp / "bench")])
    assert r.exit_code == 0, r.output
    assert list((tmp / "bench").glob("*.json"))


class _Marker:
    """Unpickling this touches a file: proof that the CLI ran code from the input."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        from pathlib import Path

        return (Path.touch, (Path(self.path),))


@pytest.mark.parametrize("command", ["inspect", "compress"])
def test_legacy_pickle_needs_explicit_trust(tmp_path, command):
    marker = tmp_path / "unpickled"
    meta = np.empty(1, dtype=object)
    meta[0] = _Marker(str(marker))
    bad = tmp_path / "crafted.npz"
    np.savez(bad, meta=meta, vectors=np.zeros((1, 4), np.float32))
    args = [command, str(bad)] + (["-p", "binary", "-o", str(tmp_path / "out.npz")] if command == "compress" else [])
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert "--trust-pickle" in result.output
    assert not marker.exists()
