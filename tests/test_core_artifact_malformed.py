"""Malformed or edited bundle files and certificate text give a clean error, not a traceback.

Each case edits one file of a valid bundle (and refreshes the manifest, as a careless editor
would) or feeds hostile text to the certificate reader. The expected result is
``ArtifactError`` / ``CertificateFormatError`` and never an ``AttributeError``, ``KeyError``,
``TypeError`` or ``RecursionError`` from inside the loader. The header fields ``dim`` and
``stats`` of ``codes.npz`` are covered by no digest; the loader checks them against the
certificate and the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from optivision.core.artifact import (
    ArtifactError,
    Certificate,
    CertificateFormatError,
    load_artifact,
)
from optivision.core.artifact import bundle as bundle_module
from optivision.core.artifact.record import content_digest

from .test_core_artifact_bundle import (  # noqa: F401  (fixtures)
    issued,
    pca,
    problem,
    refresh_manifest,
    write,
)


@pytest.fixture
def path(tmp_path, pca):  # noqa: F811
    return write(tmp_path, pca)


def edit_header(file: Path, edit) -> None:
    arrays = bundle_module._read_npz(file)
    header = json.loads(arrays["header"].tobytes().decode())
    edit(header)
    arrays["header"] = bundle_module._header_array(header)
    bundle_module._write_npz(file, arrays)


def edit_certificate(path: Path, edit) -> None:
    d = json.loads((path / "certificate.json").read_text(encoding="utf-8"))
    edit(d)
    d["digest"] = content_digest(d)
    (path / "certificate.json").write_text(json.dumps(d), encoding="utf-8")


def test_a_certificate_nested_too_deeply_is_a_format_error():
    with pytest.raises(CertificateFormatError, match="nested too deeply"):
        Certificate.from_json("[" * 100000 + "]" * 100000)
    with pytest.raises(CertificateFormatError, match="nested too deeply"):
        Certificate.from_json('{"encoder": ' + "[" * 20000 + "]" * 20000 + "}")


def test_a_deeply_nested_mapping_is_a_format_error():
    deep: list = []
    for _ in range(20000):
        deep = [deep]
    with pytest.raises(CertificateFormatError):
        Certificate.from_dict({"encoder": deep})


def test_a_manifest_nested_too_deeply(path):
    (path / "MANIFEST.json").write_text('{"files":' + "[" * 100000, encoding="utf-8")
    with pytest.raises(ArtifactError, match="MANIFEST.json is not readable JSON"):
        load_artifact(path)


@pytest.mark.parametrize("edit", [
    lambda h: h["stages"].__setitem__(0, 7),
    lambda h: h["stages"].__setitem__(0, None),
    lambda h: h["stages"][0].__setitem__("scalars", [1]),
    lambda h: h["stages"][0].__setitem__("scalars", None),
    lambda h: h["stages"][0].__setitem__("arrays", 5),
    lambda h: h["stages"][0].__setitem__("arrays", [3]),
], ids=["entry-int", "entry-null", "scalars-list", "scalars-null", "arrays-int", "array-name-int"])
def test_a_malformed_fitted_state_header(path, edit):
    edit_header(path / "fitted.npz", edit)
    refresh_manifest(path)
    with pytest.raises(ArtifactError):
        load_artifact(path)


@pytest.mark.parametrize("edit", [
    lambda h: h.pop("metric"),
    lambda h: h.__setitem__("query_ids", 5),
    lambda h: h.__setitem__("fitted", 5),
    lambda h: h.__setitem__("dropped_queries", 5),
], ids=["no-metric", "query-ids-int", "fitted-int", "dropped-int"])
def test_a_malformed_evidence_header(path, edit):
    edit_header(path / "evidence.npz", edit)
    refresh_manifest(path)
    with pytest.raises(ArtifactError, match="evidence.npz cannot be read"):
        load_artifact(path)


@pytest.mark.parametrize("edit", [
    lambda d: d.__setitem__("candidate", {}),
    lambda d: d.__setitem__("candidate", 5),
    lambda d: d["candidate"].__setitem__("fitted_state", 5),
], ids=["candidate-empty", "candidate-int", "fitted-state-int"])
def test_a_malformed_candidate_record_in_the_certificate_file(path, edit):
    edit_certificate(path, edit)
    refresh_manifest(path)
    with pytest.raises(ArtifactError):
        load_artifact(path)


@pytest.mark.parametrize("edit", [
    lambda h: h.__setitem__("dim", 16),
    lambda h: h.__setitem__("dim", 99),
    lambda h: h["stats"].__setitem__("vectors_before", 1),
    lambda h: h["stats"].__setitem__("dim_before", 3),
], ids=["dim-same-width", "dim-wider", "stats-vectors", "stats-dim"])
def test_header_fields_that_no_digest_covers_are_checked(path, edit):
    """``dim`` decides how the codes decode; with the manifest refreshed it was loaded silently."""
    edit_header(path / "codes.npz", edit)
    refresh_manifest(path)
    with pytest.raises(ArtifactError, match="does not match the certified layout"):
        load_artifact(path)
    loaded = load_artifact(path, verify=False)  # as edited, by the documented contract
    assert loaded.compressed is not None


def test_an_honest_bundle_still_loads(path, pca):  # noqa: F811
    assert load_artifact(path).verification.ok
    assert np.array_equal(load_artifact(path).compressed.codes, pca.compressed.codes)
