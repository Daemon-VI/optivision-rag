"""The artifact bundle: round trips, bit-for-bit reproduction, corruption, determinism.

``save_artifact`` writes a directory (certificate, codes, fitted stage state, optional evidence,
manifest) with no pickle; ``load_artifact`` reads it back and by default verifies it. The loaded
index must decode, project queries and score exactly like the saved one, and damage to any file
must be detected. Bundles are written under ``tmp_path`` only: ``*.npz`` is routed to Git LFS by
the repository's ``.gitattributes`` and generated bundles are never committed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from optivision.compose import Pipeline
from optivision.core.artifact import (
    Artifact,
    ArtifactError,
    Certificate,
    load_artifact,
    save_artifact,
    sign_certificate,
    verify_certificate,
)
from optivision.core.artifact import bundle as bundle_module
from optivision.core.evaluate import Evidence
from optivision.core.identity import digest
from optivision.core.selection import CandidateSpace, Outcome
from optivision.core.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    Int4Quantizer,
    Int8Quantizer,
    Lloyd2Quantizer,
    PCAProjector,
    RandomProjector,
    RotationProjector,
)
from optivision.scoring import maxsim_matrix

from .test_core_certificate import KEY, Issued, issue
from .test_core_collect_evidence import make_problem, small_space

try:
    import scipy.cluster.hierarchy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_SCIPY = False

#: Pipelines with no fitted state, a codec mean, PCA + codec, a rotation, a merge, ... Each is a
#: one-candidate space under an easy target, so it is selected and its certificate deploys it.
SCENARIOS = {
    "no-state": (lambda: Pipeline([Float16Quantizer()]), False),
    "codec-mean": (lambda: Pipeline([Int4Quantizer(center="mean")]), True),
    "pca+codec": (lambda: Pipeline([PCAProjector(12, center=True), Int4Quantizer(center="mean")]),
                  True),
    "rotation": (lambda: Pipeline([RotationProjector(seed=2), Int8Quantizer("per_dimension")]),
                 True),
    "merge": (lambda: Pipeline([AdaptiveMerge(radius=0.9, center="mean"),
                                Int8Quantizer("per_vector")]), True),
    "random-projection": (lambda: Pipeline([RandomProjector(8, seed=1), BinaryQuantizer()]),
                          True),
    "lloyd2": (lambda: Pipeline([Lloyd2Quantizer(seed=5)]), True),
    "float32-only": (lambda: Pipeline([]), False),
}


def one_candidate(name):
    make, _ = SCENARIOS[name]
    return CandidateSpace.from_families({name: [make()]}, name=name)


@pytest.fixture(scope="module")
def problem():
    return make_problem()


@pytest.fixture(scope="module")
def issued(problem):
    """Every scenario, issued once: ``issued[name]``."""
    out = {}
    for name in SCENARIOS:
        out[name] = issue(problem, one_candidate(name), target=0.3, error=0.3)
        assert out[name].decision.outcome is Outcome.SELECT, name
    return out


@pytest.fixture(scope="module")
def pca(issued):
    return issued["pca+codec"]


@pytest.fixture(scope="module")
def abstained(problem):
    out = issue(problem, small_space(), target=0.95, error=0.3)
    assert out.decision.outcome is Outcome.ABSTAIN
    return out


@pytest.fixture(scope="module")
def fell_back(problem):
    fallback = Pipeline([Int8Quantizer("per_dimension", center="mean")])
    out = issue(problem, small_space(), target=1.0, fallback=fallback)
    assert out.decision.outcome is Outcome.FALLBACK
    return out


def write(tmp_path: Path, out: Issued, *, evidence: bool = True, name: str = "bundle",
          certificate: Certificate | None = None) -> Path:
    return save_artifact(
        tmp_path / name, certificate=out.certificate if certificate is None else certificate,
        compressed=out.compressed, pipeline=out.pipeline,
        evidence=out.evidence if evidence else None)


def listing(path: Path) -> set[str]:
    return {p.name for p in path.iterdir()}


def file_record(path: Path) -> dict:
    """The manifest's rule, restated independently: raw bytes, but .json is hashed after LF."""
    data = path.read_bytes()
    if path.suffix == ".json":
        data = data.replace(b"\r\n", b"\n")
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def refresh_manifest(root: Path) -> None:
    """What a dishonest editor does after changing a file: write a manifest that matches."""
    names = [n for n in ("certificate.json", "codes.npz", "fitted.npz", "evidence.npz")
             if (root / n).is_file()]
    manifest = {"format": "optivision-artifact-manifest", "version": 1,
                "files": {n: file_record(root / n) for n in names}}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")


def flip_payload_byte(path: Path, payload: bytes, offset: int = 3) -> None:
    raw = bytearray(path.read_bytes())
    at = bytes(raw).find(payload)
    assert at > 0, "the array payload was not found in the file"
    raw[at + offset] ^= 0x01
    path.write_bytes(bytes(raw))


def assert_same_index(loaded: Artifact, out: Issued) -> None:
    """Codes, decoded vectors, projected queries and scores equal the saved ones bit for bit."""
    a, b = loaded.compressed, out.compressed
    assert a.codes.dtype == b.codes.dtype == np.uint8 and a.codes.shape == b.codes.shape
    assert a.codes.tobytes() == b.codes.tobytes()
    assert a.offsets.dtype == b.offsets.dtype and a.offsets.tobytes() == b.offsets.tobytes()
    assert a.dim == b.dim and list(a.ids) == list(b.ids) and a.stats == b.stats
    assert a.pipeline == b.pipeline
    n = a.num_vectors
    decoded_a, decoded_b = a.decode_rows(0, n), b.decode_rows(0, n)
    assert decoded_a.dtype == decoded_b.dtype and decoded_a.tobytes() == decoded_b.tobytes()
    queries_a = loaded.pipeline.transform_queries(out.queries)
    queries_b = out.pipeline.transform_queries(out.queries)
    assert queries_a.vectors.dtype == queries_b.vectors.dtype
    assert queries_a.vectors.tobytes() == queries_b.vectors.tobytes()
    scores_a, scores_b = maxsim_matrix(queries_a, a), maxsim_matrix(queries_b, b)
    assert scores_a.dtype == scores_b.dtype and np.array_equal(scores_a, scores_b)
    assert loaded.pipeline.fitted and loaded.pipeline.to_dict() == out.pipeline.to_dict()


# ------------------------------------------------------------------ round trips


class TestRoundTrip:
    @pytest.mark.parametrize("name", list(SCENARIOS))
    @pytest.mark.parametrize("with_evidence", [True, False], ids=["evidence", "no-evidence"])
    def test_the_loaded_index_is_the_saved_one(self, tmp_path, issued, name, with_evidence):
        out = issued[name]
        path = write(tmp_path, out, evidence=with_evidence)
        loaded = load_artifact(path)
        assert isinstance(loaded, Artifact) and loaded.path == path
        assert loaded.certificate == out.certificate
        assert_same_index(loaded, out)
        assert (loaded.evidence == out.evidence) if with_evidence else loaded.evidence is None
        assert loaded.verification is not None and loaded.verification.ok
        expected_levels = ("integrity", "replay", "binding") if with_evidence \
            else ("integrity", "binding")
        assert loaded.verification.levels == expected_levels

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_the_files_written(self, tmp_path, issued, name):
        out = issued[name]
        path = write(tmp_path, out)
        stateful = SCENARIOS[name][1]
        assert listing(path) == {"certificate.json", "codes.npz", "evidence.npz", "MANIFEST.json",
                                 *(["fitted.npz"] if stateful else [])}
        assert listing(write(tmp_path, out, evidence=False, name="b2")) == \
            listing(path) - {"evidence.npz"}

    @pytest.mark.parametrize("name", ["codec-mean", "rotation", "lloyd2"])
    def test_the_loaded_certificate_binds_to_the_loaded_objects(self, tmp_path, issued, name):
        out = issued[name]
        loaded = load_artifact(write(tmp_path, out))
        report = verify_certificate(
            loaded.certificate, evidence=loaded.evidence, corpus=out.corpus, queries=out.queries,
            reference=out.reference, pipeline=loaded.pipeline, compressed=loaded.compressed)
        assert report.ok and report.level == "binding", report.summary()

    def test_a_loaded_pipeline_can_compress_more_documents_like_the_original(self, tmp_path,
                                                                             issued):
        out = issued["codec-mean"]
        loaded = load_artifact(write(tmp_path, out))
        more = out.corpus.select(range(5))
        a = loaded.pipeline.compress(more, fit=False)  # fit=False: the restored state is used
        b = out.pipeline.compress(more, fit=False)
        assert a.codes.tobytes() == b.codes.tobytes()

    def test_an_abstention_is_a_certificate_only_bundle(self, tmp_path, abstained):
        for with_evidence in (True, False):
            path = save_artifact(tmp_path / f"abstain-{with_evidence}",
                                 certificate=abstained.certificate,
                                 evidence=abstained.evidence if with_evidence else None)
            assert listing(path) == {"certificate.json", "MANIFEST.json",
                                     *(["evidence.npz"] if with_evidence else [])}
            loaded = load_artifact(path)
            assert loaded.certificate == abstained.certificate
            assert loaded.pipeline is None and loaded.compressed is None
            assert loaded.verification.ok
            assert (loaded.evidence is not None) is with_evidence

    def test_a_fallback_keeps_its_fitted_state(self, tmp_path, fell_back):
        path = write(tmp_path, fell_back)
        assert "fitted.npz" in listing(path)
        loaded = load_artifact(path)
        assert loaded.certificate.outcome is Outcome.FALLBACK and not loaded.certificate.certified
        assert_same_index(loaded, fell_back)

    def test_the_certificate_file_is_the_certificate(self, tmp_path, pca):
        path = write(tmp_path, pca)
        assert Certificate.load(path / "certificate.json") == pca.certificate
        assert (path / "certificate.json").read_bytes() == \
            (pca.certificate.to_json() + "\n").encode("utf-8")

    def test_the_evidence_file_is_a_readable_evidence_file(self, tmp_path, pca):
        path = write(tmp_path, pca)
        assert Evidence.from_npz(path / "evidence.npz") == pca.evidence

    def test_a_signed_certificate_is_carried_with_its_signature(self, tmp_path, pca):
        signed = sign_certificate(pca.certificate, KEY, key_id="k1")
        path = write(tmp_path, pca, certificate=signed)
        assert load_artifact(path).verification.status("signature") == "skipped"
        loaded = load_artifact(path, key=KEY)
        assert loaded.certificate.signature["key_id"] == "k1"
        assert loaded.verification.status("signature") == "pass"
        with pytest.raises(ArtifactError, match="signature"):
            load_artifact(path, key=b"x" * 32)

    def test_a_different_code_identity_loads_unless_strict(self, tmp_path, pca):
        d = pca.certificate.to_dict()
        d["code"]["files"]["core/identity.py"] = "0" * 64
        d["code"]["digest"] = digest(d["code"]["files"])
        d["digest"] = digest({k: v for k, v in d.items() if k not in ("digest", "signature")})
        path = write(tmp_path, pca, certificate=Certificate.from_dict(d))
        assert load_artifact(path).verification.status("code.files") == "differs"
        with pytest.raises(ArtifactError, match="code.files"):
            load_artifact(path, strict_code=True)


@pytest.mark.skipif(not HAVE_SCIPY, reason="the registered default space needs SciPy")
def test_the_registered_default_space_end_to_end(tmp_path, problem):
    space = CandidateSpace.default(problem[0].dimension)
    out = issue(problem, space, target=0.5, error=0.3)
    assert out.decision.outcome is Outcome.SELECT and out.certificate.to_dict()["space"][
        "registered_default"] is True
    loaded = load_artifact(write(tmp_path, out))
    assert_same_index(loaded, out)
    assert loaded.verification.ok


# ------------------------------------------------- what is on disk, and no pickle


class TestFormat:
    def test_the_manifest_lists_sha256_and_size_of_every_other_file(self, tmp_path, pca):
        path = write(tmp_path, pca)
        manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["format"] == "optivision-artifact-manifest" and manifest["version"] == 1
        assert set(manifest["files"]) == listing(path) - {"MANIFEST.json"}
        for name, record in manifest["files"].items():
            assert record == file_record(path / name), name
        assert (path / "MANIFEST.json").read_bytes().endswith(b"}\n")

    def test_json_files_are_hashed_after_line_ending_normalisation(self, tmp_path, pca):
        path = write(tmp_path, pca)
        for name in ("certificate.json", "MANIFEST.json"):  # a checkout that converts to CRLF
            raw = (path / name).read_bytes()
            (path / name).write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        assert load_artifact(path).verification.ok

    def test_npz_files_hold_plain_arrays_and_a_canonical_json_header(self, tmp_path, pca):
        path = write(tmp_path, pca)
        with np.load(path / "codes.npz", allow_pickle=False) as data:
            assert set(data.files) == {"codes", "offsets", "header"}
            assert data["codes"].dtype == np.uint8 and data["offsets"].dtype == np.int64
            assert data["header"].dtype == np.uint8
            raw = data["header"].tobytes()
        header = json.loads(raw)
        assert raw == json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
        assert header["format"] == "optivision-artifact" and header["version"] == 1
        assert set(header) == {"format", "version", "dim", "ids", "stats", "pipeline"}
        assert header["ids"] == list(pca.corpus.ids) and header["dim"] == pca.compressed.dim
        assert header["pipeline"] == pca.certificate.to_dict()["candidate"]["pipeline"]

    def test_fitted_state_is_arrays_plus_scalars_in_the_header(self, tmp_path, pca):
        path = write(tmp_path, pca)
        with np.load(path / "fitted.npz", allow_pickle=False) as data:
            assert set(data.files) == {"header", "s0.components", "s0.mean", "s1.mean"}
            assert data["s0.components"].dtype == np.float32
            assert data["s0.mean"].dtype == np.float64 and data["s1.mean"].dtype == np.float32
            header = json.loads(data["header"].tobytes())
        assert header["format"] == "optivision-artifact-fitted"
        assert [s["stage"] for s in header["stages"]] == ["pca", "int4"]
        assert header["stages"][0]["scalars"].keys() == {"explained"}
        assert header["stages"][0]["arrays"] == ["components", "mean"]
        assert header["stages"][1] == {"stage": "int4", "arrays": ["mean"], "scalars": {}}

    def test_no_object_arrays_and_no_pickle_anywhere(self, tmp_path, pca):
        path = write(tmp_path, pca)
        for name in ("codes.npz", "fitted.npz", "evidence.npz"):
            with np.load(path / name, allow_pickle=False) as data:
                assert all(not data[k].dtype.hasobject for k in data.files), name
            with zipfile.ZipFile(path / name) as z:
                for info in z.infolist():
                    blob = z.read(info)
                    assert blob.startswith(b"\x93NUMPY"), (name, info.filename)
                    assert b"\x80\x04" not in blob[:64]  # no pickle protocol header
        source = Path(bundle_module.__file__).read_text(encoding="utf-8")
        assert "import pickle" not in source and "allow_pickle=True" not in source

    def test_the_npz_container_has_fixed_zip_metadata(self, tmp_path, pca):
        path = write(tmp_path, pca)
        for name in ("codes.npz", "fitted.npz", "evidence.npz"):
            with zipfile.ZipFile(path / name) as z:
                for info in z.infolist():
                    assert info.date_time == (1980, 1, 1, 0, 0, 0), name
                    assert info.create_system == 3 and info.compress_type == zipfile.ZIP_STORED

    def test_np_savez_leaves_the_zip_creator_to_the_operating_system(self, tmp_path):
        """Why the bundle writes its own container: the platform byte differs across systems."""
        np.savez(tmp_path / "x.npz", a=np.arange(3))
        with zipfile.ZipFile(tmp_path / "x.npz") as z:
            assert z.infolist()[0].create_system == zipfile.ZipInfo("x").create_system
        assert zipfile.ZipInfo("x").create_system == (0 if sys.platform == "win32" else 3)

    def test_the_bundle_npz_files_are_read_by_numpy_and_zipfile_alike(self, tmp_path, pca):
        path = write(tmp_path, pca)
        with np.load(path / "codes.npz", allow_pickle=False) as data:
            assert np.array_equal(data["codes"], pca.compressed.codes)
            assert np.array_equal(data["offsets"], pca.compressed.offsets)
        with zipfile.ZipFile(path / "codes.npz") as z:
            assert z.testzip() is None


class TestDeterminism:
    def test_a_bundle_written_twice_from_the_same_inputs_is_byte_identical(self, tmp_path, pca):
        a, b = write(tmp_path, pca, name="a"), write(tmp_path, pca, name="b")
        assert listing(a) == listing(b)
        for name in listing(a):
            assert (a / name).read_bytes() == (b / name).read_bytes(), name

    def test_so_is_a_bundle_from_an_independently_issued_certificate(self, tmp_path, problem):
        spaces = [one_candidate("pca+codec") for _ in range(2)]
        outs = [issue(problem, s, target=0.3, error=0.3) for s in spaces]
        a, b = write(tmp_path, outs[0], name="a"), write(tmp_path, outs[1], name="b")
        assert (a / "MANIFEST.json").read_bytes() == (b / "MANIFEST.json").read_bytes()
        for name in listing(a):
            assert (a / name).read_bytes() == (b / name).read_bytes(), name

    def test_the_manifest_changes_when_the_certificate_does(self, tmp_path, problem, pca):
        later = issue(problem, one_candidate("pca+codec"), target=0.3, error=0.3,
                      created="2026-10-07T12:00:00Z")
        a, b = write(tmp_path, pca, name="a"), write(tmp_path, later, name="b")
        assert (a / "MANIFEST.json").read_bytes() != (b / "MANIFEST.json").read_bytes()
        assert (a / "codes.npz").read_bytes() == (b / "codes.npz").read_bytes()


# ----------------------------------------------------------- refusing to overwrite


class TestDirectories:
    def test_a_non_empty_directory_is_refused_unless_overwrite(self, tmp_path, pca):
        path = write(tmp_path, pca)
        with pytest.raises(FileExistsError, match="overwrite=True"):
            write(tmp_path, pca)
        again = save_artifact(path, certificate=pca.certificate, compressed=pca.compressed,
                              pipeline=pca.pipeline, evidence=pca.evidence, overwrite=True)
        assert again == path and load_artifact(path).verification.ok

    def test_an_empty_directory_is_accepted(self, tmp_path, pca):
        (tmp_path / "bundle").mkdir()
        assert load_artifact(write(tmp_path, pca)).verification.ok

    def test_a_file_in_the_way_is_refused(self, tmp_path, pca):
        (tmp_path / "bundle").write_text("not a directory", encoding="utf-8")
        with pytest.raises(FileExistsError, match="not a directory"):
            write(tmp_path, pca)
        with pytest.raises(FileExistsError):
            write(tmp_path, pca, name="bundle")

    def test_overwrite_replaces_the_bundles_own_files_and_leaves_others_alone(self, tmp_path,
                                                                              issued):
        path = write(tmp_path, issued["pca+codec"])
        (path / "notes.txt").write_text("keep me", encoding="utf-8")
        assert "fitted.npz" in listing(path) and "evidence.npz" in listing(path)
        save_artifact(path, certificate=issued["no-state"].certificate,
                      compressed=issued["no-state"].compressed,
                      pipeline=issued["no-state"].pipeline, overwrite=True)
        assert listing(path) == {"certificate.json", "codes.npz", "MANIFEST.json", "notes.txt"}
        assert (path / "notes.txt").read_text(encoding="utf-8") == "keep me"
        assert load_artifact(path).certificate == issued["no-state"].certificate

    def test_the_directory_is_created_with_its_parents(self, tmp_path, pca):
        path = save_artifact(tmp_path / "a" / "b" / "c", certificate=pca.certificate,
                             compressed=pca.compressed, pipeline=pca.pipeline)
        assert load_artifact(path).verification.ok

    def test_a_missing_directory_cannot_be_loaded(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifact(tmp_path / "nothing")
        (tmp_path / "plain").write_text("x", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            load_artifact(tmp_path / "plain")


class TestSaveRefusesWhatDoesNotBelongTogether:
    def test_an_abstention_takes_no_objects(self, tmp_path, abstained, pca):
        with pytest.raises(ValueError, match="abstention"):
            save_artifact(tmp_path / "x", certificate=abstained.certificate,
                          compressed=pca.compressed, pipeline=pca.pipeline)
        assert not (tmp_path / "x").exists()

    def test_a_select_certificate_needs_both_objects(self, tmp_path, pca):
        with pytest.raises(ValueError, match="needs the fitted pipeline"):
            save_artifact(tmp_path / "x", certificate=pca.certificate, compressed=pca.compressed)
        with pytest.raises(ValueError, match="needs the fitted pipeline"):
            save_artifact(tmp_path / "x", certificate=pca.certificate, pipeline=pca.pipeline)
        assert not (tmp_path / "x").exists()

    def test_objects_that_are_not_the_certified_ones_are_refused_before_anything_is_written(
            self, tmp_path, issued):
        with pytest.raises(ArtifactError, match="refusing to write") as caught:
            save_artifact(tmp_path / "x", certificate=issued["pca+codec"].certificate,
                          compressed=issued["merge"].compressed, pipeline=issued["merge"].pipeline)
        assert "binding.pipeline" in str(caught.value) and "binding.codes" in str(caught.value)
        assert not (tmp_path / "x").exists()

    def test_evidence_that_is_not_the_certified_evidence_is_refused(self, tmp_path, pca,
                                                                    abstained):
        with pytest.raises(ArtifactError, match="evidence.digest"):
            save_artifact(tmp_path / "x", certificate=pca.certificate, compressed=pca.compressed,
                          pipeline=pca.pipeline, evidence=abstained.evidence)

    def test_a_modified_certificate_is_refused(self, tmp_path, pca):
        d = pca.certificate.to_dict()
        d["target"] = 0.9
        with pytest.raises(ArtifactError, match="digest"):
            save_artifact(tmp_path / "x", certificate=Certificate.from_dict(d),
                          compressed=pca.compressed, pipeline=pca.pipeline)

    def test_wrong_types(self, tmp_path, pca):
        with pytest.raises(TypeError, match="Certificate"):
            save_artifact(tmp_path / "x", certificate=pca.certificate.to_dict())
        with pytest.raises(TypeError, match="Evidence"):
            save_artifact(tmp_path / "x", certificate=pca.certificate, compressed=pca.compressed,
                          pipeline=pca.pipeline, evidence="e")


# ------------------------------------------------------------ damage is detected


class TestCorruptionIsDetected:
    @pytest.fixture
    def path(self, tmp_path, pca):
        return write(tmp_path, pca)

    def test_one_flipped_byte_of_the_codes(self, path, pca):
        flip_payload_byte(path / "codes.npz", pca.compressed.codes.tobytes())
        with pytest.raises(ArtifactError, match=r"codes\.npz does not match its manifest"):
            load_artifact(path)

    def test_one_flipped_byte_of_the_codes_with_the_manifest_refreshed(self, path, pca):
        flip_payload_byte(path / "codes.npz", pca.compressed.codes.tobytes())
        refresh_manifest(path)  # the zip's own CRC-32 still refuses the damaged entry
        with pytest.raises(ArtifactError, match=r"codes\.npz cannot be read.*Bad CRC-32"):
            load_artifact(path)

    def test_changed_codes_in_a_clean_container_with_the_manifest_refreshed(self, path, pca):
        """The manifest is not the root of trust: the certificate's codes digest is."""
        with np.load(path / "codes.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["codes"] = arrays["codes"].copy()
        arrays["codes"][7, 1] ^= 0x01
        bundle_module._write_npz(path / "codes.npz", arrays)
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match=r"binding\.codes: .*codes differ"):
            load_artifact(path)

    def test_changed_offsets_in_a_clean_container_with_the_manifest_refreshed(self, path, pca):
        with np.load(path / "codes.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["offsets"] = arrays["offsets"].copy()
        arrays["offsets"][5] -= 1
        bundle_module._write_npz(path / "codes.npz", arrays)
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="binding.codes"):
            load_artifact(path)

    def test_one_flipped_byte_of_the_offsets_with_the_manifest_refreshed(self, path, pca):
        flip_payload_byte(path / "codes.npz", pca.compressed.offsets.tobytes(), offset=8)
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="codes.npz cannot be read"):
            load_artifact(path)

    def test_one_flipped_byte_of_the_fitted_state(self, path, pca):
        flip_payload_byte(path / "fitted.npz", pca.pipeline.stages[0].components.tobytes())
        with pytest.raises(ArtifactError, match=r"fitted\.npz does not match its manifest"):
            load_artifact(path)

    def test_one_flipped_byte_of_the_fitted_state_with_the_manifest_refreshed(self, path, pca):
        flip_payload_byte(path / "fitted.npz", pca.pipeline.stages[0].components.tobytes())
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match=r"fitted\.npz cannot be read"):
            load_artifact(path)

    def test_one_changed_value_of_the_fitted_state_in_a_clean_container(self, path, pca):
        with np.load(path / "fitted.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["s1.mean"] = arrays["s1.mean"] + np.float32(1e-3)
        bundle_module._write_npz(path / "fitted.npz", arrays)
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="digest .* does not match the expected"):
            load_artifact(path)

    def test_the_certificate_is_edited(self, path):
        text = (path / "certificate.json").read_text(encoding="utf-8")
        assert '"target": 0.3' in text
        (path / "certificate.json").write_text(text.replace('"target": 0.3', '"target": 0.9'),
                                               encoding="utf-8")
        with pytest.raises(ArtifactError, match=r"certificate\.json does not match its manifest"):
            load_artifact(path)

    def test_the_certificate_is_edited_and_the_manifest_refreshed(self, path):
        text = (path / "certificate.json").read_text(encoding="utf-8")
        (path / "certificate.json").write_text(text.replace('"target": 0.3', '"target": 0.9'),
                                               encoding="utf-8")
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match=r"digest: the content digest does not match"):
            load_artifact(path)

    def test_the_certificate_is_edited_with_its_digest_and_the_manifest_refreshed(self, path,
                                                                                  pca):
        """The evidence in the bundle exposes a changed target through the replay."""
        d = pca.certificate.to_dict()
        d["target"] = 0.9
        d["digest"] = digest({k: v for k, v in d.items() if k not in ("digest", "signature")})
        Certificate.from_dict(d).save(path / "certificate.json")
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="replay.decision"):
            load_artifact(path)

    def test_a_signature_exposes_an_edit_that_fixes_the_digest_and_the_manifest(self, tmp_path,
                                                                                pca):
        path = write(tmp_path, pca, certificate=sign_certificate(pca.certificate, KEY),
                     evidence=False)
        d = Certificate.load(path / "certificate.json").to_dict()
        d["package"]["version"] = "9.9.9"  # a descriptive field no check ties to anything
        d["digest"] = digest({k: v for k, v in d.items() if k not in ("digest", "signature")})
        Certificate.from_dict(d).save(path / "certificate.json")
        refresh_manifest(path)
        assert load_artifact(path).verification.ok  # without the key the edit goes unseen
        with pytest.raises(ArtifactError, match="signature"):
            load_artifact(path, key=KEY)

    def test_the_fitted_state_file_is_deleted(self, path):
        (path / "fitted.npz").unlink()
        with pytest.raises(ArtifactError, match=r"fitted\.npz is listed but missing"):
            load_artifact(path)

    def test_the_fitted_state_file_is_deleted_and_the_manifest_refreshed(self, path):
        (path / "fitted.npz").unlink()
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="fitted.npz is missing but the certificate"):
            load_artifact(path)

    def test_the_codes_file_is_deleted(self, path):
        (path / "codes.npz").unlink()
        with pytest.raises(ArtifactError, match=r"codes\.npz is listed but missing"):
            load_artifact(path)
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match=r"codes\.npz is missing"):
            load_artifact(path)

    def test_the_certificate_file_is_deleted(self, path):
        (path / "certificate.json").unlink()
        with pytest.raises(ArtifactError, match="certificate.json"):
            load_artifact(path)
        with pytest.raises(ArtifactError, match="certificate.json is missing"):
            load_artifact(path, verify=False)

    def test_the_manifest_is_edited(self, path):
        manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
        record = manifest["files"]["codes.npz"]
        record["sha256"] = ("0" if record["sha256"][0] != "0" else "1") + record["sha256"][1:]
        (path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ArtifactError, match=r"codes\.npz does not match its manifest"):
            load_artifact(path)

    def test_a_size_in_the_manifest_is_edited(self, path):
        manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["files"]["evidence.npz"]["size"] += 1
        (path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ArtifactError, match=r"evidence\.npz has \d+ bytes, the manifest"):
            load_artifact(path)

    def test_a_file_is_missing_from_the_manifest(self, path):
        manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
        del manifest["files"]["evidence.npz"]
        (path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ArtifactError, match=r"evidence\.npz is present but not listed"):
            load_artifact(path)

    def test_the_manifest_lists_a_file_this_format_does_not_have(self, path):
        manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["files"]["../outside.bin"] = {"sha256": "0" * 64, "size": 1}
        (path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ArtifactError, match="not a file of this format"):
            load_artifact(path)

    @pytest.mark.parametrize("content", [
        "", "{not json", "[]", '{"format": "other", "version": 1}',
        ('{"format": "optivision-artifact-manifest", "version": 2, "files": {}}')])
    def test_the_manifest_is_unreadable(self, path, content):
        (path / "MANIFEST.json").write_text(content, encoding="utf-8")
        with pytest.raises(ArtifactError, match="MANIFEST.json"):
            load_artifact(path)

    def test_the_manifest_is_deleted(self, path):
        (path / "MANIFEST.json").unlink()
        with pytest.raises(ArtifactError, match="MANIFEST.json is missing"):
            load_artifact(path)

    def test_a_truncated_codes_file(self, path):
        raw = (path / "codes.npz").read_bytes()
        (path / "codes.npz").write_bytes(raw[: len(raw) // 2])
        with pytest.raises(ArtifactError, match="codes.npz"):
            load_artifact(path)

    def test_a_truncated_codes_file_with_the_manifest_refreshed(self, path):
        raw = (path / "codes.npz").read_bytes()
        (path / "codes.npz").write_bytes(raw[: len(raw) // 2])
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match=r"codes\.npz cannot be read"):
            load_artifact(path)

    def test_the_evidence_file_is_damaged(self, path, pca):
        flip_payload_byte(path / "evidence.npz", pca.evidence.baseline.tobytes())
        with pytest.raises(ArtifactError, match=r"evidence\.npz does not match its manifest"):
            load_artifact(path)
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match=r"evidence\.npz cannot be read"):
            load_artifact(path)

    def test_evidence_of_another_run_is_exposed_by_the_certificate(self, path, abstained):
        abstained.evidence.to_npz(path / "evidence.npz")
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="evidence.digest"):
            load_artifact(path)

    def test_the_certificate_file_is_not_json(self, path):
        (path / "certificate.json").write_text("{broken", encoding="utf-8")
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match=r"certificate\.json cannot be read"):
            load_artifact(path)

    def test_codes_and_fitted_state_of_another_bundle(self, path, issued, tmp_path):
        other = write(tmp_path, issued["rotation"], name="other")
        for name in ("codes.npz", "fitted.npz"):
            (path / name).write_bytes((other / name).read_bytes())
        refresh_manifest(path)
        with pytest.raises(ArtifactError):
            load_artifact(path)

    def test_stray_codes_in_an_abstention_bundle(self, tmp_path, abstained, pca):
        path = save_artifact(tmp_path / "abs", certificate=abstained.certificate)
        other = write(tmp_path, pca, name="other")
        (path / "codes.npz").write_bytes((other / "codes.npz").read_bytes())
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="records no deployed candidate"):
            load_artifact(path)

    def test_without_verification_nothing_is_checked(self, path, pca):
        """Pinned: verify=False reads the files and checks nothing about them."""
        with np.load(path / "codes.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["codes"] = arrays["codes"].copy()
        arrays["codes"][7, 1] ^= 0x01
        bundle_module._write_npz(path / "codes.npz", arrays)
        (path / "MANIFEST.json").unlink()
        loaded = load_artifact(path, verify=False)
        assert loaded.verification is None
        assert loaded.compressed.codes.tobytes() != pca.compressed.codes.tobytes()
        assert loaded.certificate == pca.certificate
        report = verify_certificate(loaded.certificate, pipeline=loaded.pipeline,
                                    compressed=loaded.compressed)
        assert report.status("binding.codes") == "fail"  # the certificate still exposes it

    def test_every_payload_file_is_covered_by_the_manifest(self, path):
        manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
        for name in manifest["files"]:
            raw = (path / name).read_bytes()
            (path / name).write_bytes(raw + b"\x00")  # trailing junk changes the file
            with pytest.raises(ArtifactError, match=name.replace(".", r"\.")):
                load_artifact(path)
            (path / name).write_bytes(raw)
        assert load_artifact(path).verification.ok


class TestCrossChecksOnLoad:
    def test_a_pipeline_description_in_the_codes_header_that_is_not_the_certified_one(
            self, tmp_path, pca):
        path = write(tmp_path, pca)
        with np.load(path / "codes.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        header = json.loads(arrays["header"].tobytes())
        header["pipeline"]["stages"][0]["dim"] = 11
        arrays["header"] = np.frombuffer(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"),
            dtype=np.uint8)
        bundle_module._write_npz(path / "codes.npz", arrays)
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="not the certified one"):
            load_artifact(path)

    def test_a_codes_file_of_another_format(self, tmp_path, pca):
        path = write(tmp_path, pca)
        np.savez(path / "codes.npz", codes=pca.compressed.codes, offsets=pca.compressed.offsets,
                 header=np.frombuffer(b'{"format": "optivision-compressed", "version": 1}',
                                      dtype=np.uint8))
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="not an 'optivision-artifact' file"):
            load_artifact(path)

    def test_a_codes_file_without_a_header(self, tmp_path, pca):
        path = write(tmp_path, pca)
        np.savez(path / "codes.npz", codes=pca.compressed.codes, offsets=pca.compressed.offsets)
        refresh_manifest(path)
        with pytest.raises(ArtifactError, match="no header"):
            load_artifact(path)
