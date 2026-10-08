"""The artifact bundle: a certificate together with the index it certifies, as plain files.

``save_artifact`` writes a directory; ``load_artifact`` reads it back and, by default,
checks it. The library's own ``optivision-compressed`` v1 file cannot hold the fitted state of
a stage (a PCA basis, a codec's mean); this bundle can, and that format stays readable and is
not changed.

======================  ======================================================================
``certificate.json``    the certificate (indented JSON, UTF-8, LF line endings)
``codes.npz``           ``codes`` (uint8), ``offsets`` (int64) and a JSON ``header`` array
                        (``format "optivision-artifact"``, ``version``, ``dim``, ``ids``,
                        ``stats``, ``pipeline``); absent for an abstention
``fitted.npz``          the fitted state of every stage (arrays ``s<i>.<name>``; scalars in the
                        ``header`` array); absent when no stage holds state
``evidence.npz``        the per-query evidence (optional; needed for a replay later)
``MANIFEST.json``       SHA-256 and size of every other file; written last
======================  ======================================================================

Nothing is pickled: arrays are read with ``allow_pickle=False`` and every header is canonical
JSON bytes in a ``uint8`` array (as ``representation.py`` writes its headers).

Manifest rule. ``.npz`` files are written by this module with fixed zip metadata (the
timestamp is the zip epoch and the creating system is a constant; ``np.savez`` leaves the
latter to the operating system). What is stable is the array content, and the digests of the
arrays that the manifest and the certificate record. The bytes of an ``.npz`` container also
depend on the Python version (zip header fields) and may depend on the NumPy version, so the
same arrays are not promised to give the same file bytes in every environment; the manifest
hashes the files as written here. A ``.json`` file is hashed
after CRLF is turned into LF, so a checkout that converts line endings does not break it.

What the manifest is. It detects accidental damage and naive edits of a file. Anyone who can
write the directory can write a matching manifest too, and anyone who can edit the certificate
can recompute its digest: without an HMAC key nothing in the bundle is a root of trust.
``load_artifact(verify=True)`` also checks the certificate (its digest, the fitted-state and
codes digests it records, an optional HMAC). ``verify=False`` reads without any integrity check.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from optivision.compose import CompressedCorpus, Pipeline
from optivision.core.evaluate.evidence import Evidence
from optivision.core.identity import canonical_json
from optivision.core.pipeline import pipeline_spec
from optivision.core.selection.spaces import thaw_json
from optivision.core.stages.state import pipeline_fitted_state, restore_pipeline_state
from optivision.stages.base import DimensionReducer

from .certificate import Certificate, CertificateFormatError, verify_certificate
from .report import VerificationReport

__all__ = ["Artifact", "ArtifactError", "load_artifact", "save_artifact"]

CERTIFICATE_FILE = "certificate.json"
CODES_FILE = "codes.npz"
FITTED_FILE = "fitted.npz"
EVIDENCE_FILE = "evidence.npz"
MANIFEST_FILE = "MANIFEST.json"
_PAYLOAD_FILES = (CERTIFICATE_FILE, CODES_FILE, FITTED_FILE, EVIDENCE_FILE)

CODES_FORMAT = "optivision-artifact"
FITTED_FORMAT = "optivision-artifact-fitted"
MANIFEST_FORMAT = "optivision-artifact-manifest"
FORMAT_VERSION = 1

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_HASH_BLOCK = 1 << 24


class ArtifactError(ValueError):
    """The bundle is incomplete, damaged, inconsistent, or failed verification."""


@dataclass(frozen=True)
class Artifact:
    """A loaded bundle. ``pipeline`` is fitted; ``pipeline`` and ``compressed`` are ``None`` for
    an abstention; ``evidence`` is ``None`` when the bundle holds none. ``verification`` is the
    report of the checks ``load_artifact`` ran (``None`` with ``verify=False``)."""

    certificate: Certificate
    pipeline: Pipeline | None
    compressed: CompressedCorpus | None
    evidence: Evidence | None
    verification: VerificationReport | None = None
    path: Path | None = None


# --------------------------------------------------------------------- npz files


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """An uncompressed ``.npz`` that ``np.load`` reads, with fixed zip metadata (see module doc)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, array in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            with archive.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.ascontiguousarray(array),
                                          allow_pickle=False)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ArtifactError(f"{path.name} cannot be read as an .npz file: {exc}") from exc


def _header_array(header: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(canonical_json(header), dtype=np.uint8)


def _read_header(arrays: Mapping[str, np.ndarray], name: str, expected_format: str) -> dict:
    if "header" not in arrays:
        raise ArtifactError(f"{name} has no header array")
    try:
        header = json.loads(arrays["header"].tobytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ArtifactError(f"the header of {name} is not JSON: {exc}") from exc
    if not isinstance(header, dict) or header.get("format") != expected_format:
        raise ArtifactError(f"{name} is not an {expected_format!r} file")
    if header.get("version") != FORMAT_VERSION:
        raise ArtifactError(f"{name} has version {header.get('version')!r}; this code reads "
                            f"version {FORMAT_VERSION}")
    return header


# -------------------------------------------------------------------- manifest


def _file_record(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        data = path.read_bytes().replace(b"\r\n", b"\n")
        return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            h.update(block)
            size += len(block)
    return {"sha256": h.hexdigest(), "size": size}


def _write_manifest(root: Path, written: list[str]) -> None:
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": FORMAT_VERSION,
        "files": {name: _file_record(root / name) for name in sorted(written)},
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (root / MANIFEST_FILE).write_bytes(text.encode("utf-8"))


def _check_manifest(root: Path) -> None:
    path = root / MANIFEST_FILE
    if not path.is_file():
        raise ArtifactError(f"{MANIFEST_FILE} is missing: the bundle is incomplete or was not "
                            "written by save_artifact")
    try:
        manifest = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ArtifactError(f"{MANIFEST_FILE} is not readable JSON: {exc}") from exc
    if (not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT
            or manifest.get("version") != FORMAT_VERSION
            or not isinstance(manifest.get("files"), dict)):
        raise ArtifactError(f"{MANIFEST_FILE} is not a version {FORMAT_VERSION} manifest")
    listed: dict[str, Any] = manifest["files"]
    problems: list[str] = []
    if CERTIFICATE_FILE not in listed:
        problems.append(f"{CERTIFICATE_FILE} is not listed")
    for name in sorted(set(listed) - set(_PAYLOAD_FILES)):
        problems.append(f"{name!r} is listed but is not a file of this format")
    for name in _PAYLOAD_FILES:
        present = (root / name).is_file()
        if name in listed and not present:
            problems.append(f"{name} is listed but missing")
        elif present and name not in listed:
            problems.append(f"{name} is present but not listed")
        elif present:
            want = listed[name]
            got = _file_record(root / name)
            if not isinstance(want, dict) or want.get("sha256") != got["sha256"]:
                problems.append(f"{name} does not match its manifest digest (the file or the "
                                "manifest was modified, or the file is damaged)")
            elif want.get("size") != got["size"]:
                problems.append(f"{name} has {got['size']} bytes, the manifest lists "
                                f"{want.get('size')}")
    if problems:
        raise ArtifactError(f"the manifest check failed: {'; '.join(problems)}")


# ------------------------------------------------------------------------ save


def _prepare_directory(root: Path, overwrite: bool) -> None:
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"{root} exists and is not a directory")
        if any(root.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"{root} is not empty; pass overwrite=True to replace the artifact's own "
                    "files (other files in the directory are left alone)")
            for name in (MANIFEST_FILE, *_PAYLOAD_FILES):  # manifest removed before the rest: a half-written bundle does not load
                (root / name).unlink(missing_ok=True)
    else:
        root.mkdir(parents=True)


def _fitted_arrays(pipeline: Pipeline) -> dict[str, np.ndarray] | None:
    states = pipeline_fitted_state(pipeline)
    if not any(states):
        return None
    arrays: dict[str, np.ndarray] = {}
    stages = []
    for i, (stage, state) in enumerate(zip(pipeline.stages, states, strict=True)):
        names, scalars = [], {}
        for key in sorted(state):
            value = state[key]
            if isinstance(value, np.ndarray):
                arrays[f"s{i}.{key}"] = value
                names.append(key)
            else:
                scalars[key] = value
        stages.append({"stage": stage.name, "arrays": names, "scalars": scalars})
    header = {"format": FITTED_FORMAT, "version": FORMAT_VERSION, "stages": stages}
    arrays["header"] = _header_array(header)
    return arrays


def save_artifact(
    path: str | Path,
    *,
    certificate: Certificate,
    compressed: CompressedCorpus | None = None,
    pipeline: Pipeline | None = None,
    evidence: Evidence | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the bundle directory ``path`` and return it.

    An abstention is a certificate only (``compressed`` and ``pipeline`` must be ``None``); a
    ``select`` or ``fallback`` certificate needs the fitted ``pipeline`` and the ``compressed``
    corpus it produced. Before anything is written the certificate is verified against the
    objects given (integrity, pipeline and codes binding, and the replay when ``evidence`` is
    given); a bundle that would fail its own verification is not written. A non-empty directory
    is refused unless ``overwrite=True``, which replaces only the bundle's own files. The manifest
    is written last.

    No pickle is written. Do not commit generated bundles: ``*.npz`` files are routed to Git LFS
    by the repository's ``.gitattributes``.
    """
    if not isinstance(certificate, Certificate):
        raise TypeError(f"certificate must be a Certificate, got {type(certificate).__name__}")
    if evidence is not None and not isinstance(evidence, Evidence):
        raise TypeError(f"evidence must be an Evidence or None, got {type(evidence).__name__}")
    deployed = certificate.data.get("candidate") is not None
    if deployed and (compressed is None or pipeline is None):
        raise ValueError("a select or fallback certificate needs the fitted pipeline and the "
                         "compressed corpus")
    if not deployed and (compressed is not None or pipeline is not None):
        raise ValueError("an abstention has no compressed corpus or pipeline to store")
    report = verify_certificate(certificate, evidence=evidence, pipeline=pipeline,
                                compressed=compressed)
    if not report.ok:
        raise ArtifactError("refusing to write a bundle that fails its own verification:\n"
                            + "\n".join(f"  {c.name}: {c.detail}" for c in report.failed()))

    root = Path(path)
    _prepare_directory(root, overwrite)
    written = [CERTIFICATE_FILE]
    certificate.save(root / CERTIFICATE_FILE)
    if deployed:
        header = {"format": CODES_FORMAT, "version": FORMAT_VERSION, "dim": compressed.dim,
                  "ids": list(compressed.ids), "stats": dict(compressed.stats),
                  "pipeline": pipeline_spec(pipeline)}
        _write_npz(root / CODES_FILE, {"codes": compressed.codes, "offsets": compressed.offsets,
                                       "header": _header_array(header)})
        written.append(CODES_FILE)
        fitted = _fitted_arrays(pipeline)
        if fitted is not None:
            _write_npz(root / FITTED_FILE, fitted)
            written.append(FITTED_FILE)
    if evidence is not None:
        with tempfile.TemporaryDirectory() as scratch:
            arrays = _read_npz(evidence.to_npz(Path(scratch) / EVIDENCE_FILE))
        _write_npz(root / EVIDENCE_FILE, arrays)  # re-packed with fixed zip metadata
        written.append(EVIDENCE_FILE)
    _write_manifest(root, written)
    return root


# ------------------------------------------------------------------------ load


def _states_from_file(root: Path, pipeline: Pipeline, expected: list[str | None]) -> list[dict]:
    path = root / FITTED_FILE
    if not path.is_file():
        if any(d is not None for d in expected):
            raise ArtifactError(f"{FITTED_FILE} is missing but the certificate records fitted "
                                "state for some stage")
        return [{} for _ in pipeline.stages]
    arrays = _read_npz(path)
    header = _read_header(arrays, FITTED_FILE, FITTED_FORMAT)
    entries = header.get("stages")
    if not isinstance(entries, list) or len(entries) != len(pipeline.stages):
        raise ArtifactError(f"{FITTED_FILE} holds state for a different number of stages than "
                            f"the pipeline has ({len(pipeline.stages)})")
    states = []
    for i, (stage, entry) in enumerate(zip(pipeline.stages, entries, strict=True)):
        if not isinstance(entry, dict) or entry.get("stage") != stage.name:
            found = entry.get("stage") if isinstance(entry, dict) else entry
            raise ArtifactError(f"{FITTED_FILE}: stage {i} is {found!r}, the "
                                f"pipeline has {stage.name!r}")
        scalars, names = entry.get("scalars", {}), entry.get("arrays", [])
        if (not isinstance(scalars, dict) or not isinstance(names, list)
                or not all(isinstance(key, str) for key in names)):
            raise ArtifactError(f"{FITTED_FILE}: the entry of stage {i} is malformed")
        state = dict(scalars)
        for key in names:
            name = f"s{i}.{key}"
            if name not in arrays:
                raise ArtifactError(f"{FITTED_FILE} lacks the array {name!r}")
            state[key] = arrays[name]
        states.append(state)
    return states


def _check_layout(cert: Certificate, pipeline: Pipeline, compressed: CompressedCorpus) -> None:
    """The header fields of ``codes.npz`` that no digest covers (``dim``, ``stats``) against what
    the certificate and the pipeline imply: a changed ``dim`` would change how the codes decode."""
    corpus = cert.data.get("corpus")
    try:
        dim_before, vectors_before = int(corpus["dimension"]), int(corpus["n_vectors"])
        dim = dim_before
        for stage in pipeline.stages:
            if isinstance(stage, DimensionReducer):
                dim = stage.output_dim(dim)
        stats = compressed.stats
        problems = []
        if compressed.dim != dim:
            problems.append(f"dim is {compressed.dim}, the certified pipeline gives {dim}")
        if compressed.codes.ndim != 2 or compressed.codes.shape[1] != pipeline.quantizer.code_bytes(dim):
            problems.append("the codes do not have the width the certified quantizer writes")
        if stats.get("vectors_before") != vectors_before or stats.get("dim_before") != dim_before:
            problems.append("stats.vectors_before or stats.dim_before differ from the certified corpus")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"{CODES_FILE} cannot be checked against the certificate: {exc!r}") from exc
    if problems:
        raise ArtifactError(f"{CODES_FILE} does not match the certified layout: " + "; ".join(problems))


def _load_deployed(root: Path, cert: Certificate, verify: bool) -> tuple[Pipeline, CompressedCorpus]:
    candidate = cert.data["candidate"]
    codes_path = root / CODES_FILE
    if not codes_path.is_file():
        raise ArtifactError(f"{CODES_FILE} is missing: a select or fallback certificate needs "
                            "the codes it certifies")
    arrays = _read_npz(codes_path)
    header = _read_header(arrays, CODES_FILE, CODES_FORMAT)
    for key in ("codes", "offsets"):
        if key not in arrays:
            raise ArtifactError(f"{CODES_FILE} lacks the array {key!r}")
    try:
        spec = thaw_json(candidate["pipeline"])
        expected = list(candidate["fitted_state"])
    except (KeyError, TypeError) as exc:
        raise ArtifactError(f"the candidate record of the certificate is malformed: {exc!r}") from exc
    if verify and canonical_json(header.get("pipeline")) != canonical_json(spec):
        raise ArtifactError(f"the pipeline recorded in {CODES_FILE} is not the certified one")
    try:
        pipeline = Pipeline.from_dict(spec)
    except Exception as exc:  # unknown stage name or bad parameters
        raise ArtifactError(f"the certified pipeline cannot be rebuilt: {exc}") from exc
    states = _states_from_file(root, pipeline, expected)
    try:
        restore_pipeline_state(pipeline, states, expected_digests=expected if verify else None)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"the fitted state cannot be restored: {exc}") from exc
    try:
        compressed = CompressedCorpus(
            arrays["codes"], arrays["offsets"], header["dim"], pipeline.quantizer,
            header["ids"], stats=header.get("stats"), pipeline=header.get("pipeline"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"the compressed corpus cannot be rebuilt: {exc}") from exc
    if verify:
        _check_layout(cert, pipeline, compressed)
    return pipeline, compressed


def _load_evidence(path: Path) -> Evidence | None:
    if not path.is_file():
        return None
    try:
        return Evidence.from_npz(path)  # checks the digest stored in the file against its content
    except (ValueError, OSError, zipfile.BadZipFile, KeyError, TypeError, AttributeError,
            RecursionError) as exc:  # EvidenceFormatError is a ValueError; the rest: a malformed header
        raise ArtifactError(f"{EVIDENCE_FILE} cannot be read: {exc}") from exc


def load_artifact(
    path: str | Path,
    *,
    verify: bool = True,
    key: bytes | None = None,
    strict_code: bool = False,
) -> Artifact:
    """Read a bundle written by :func:`save_artifact`.

    With ``verify=True`` (the default) the manifest is checked, fitted state is restored with
    the digests the certificate records, and the certificate is verified: level 1, the pipeline
    and codes binding, and (when the bundle holds evidence) the replay. Any failed check raises
    :class:`ArtifactError` listing it. ``key`` is passed on for an HMAC check and
    ``strict_code`` makes a different code identity a failure. With ``verify=False`` nothing is
    checked beyond what is needed to read the files.

    The loaded ``CompressedCorpus`` decodes bit for bit like the one that was saved, the loaded
    pipeline projects queries bit for bit alike, and ``maxsim_matrix`` over it gives the same
    scores, provided the fitted state restores exactly (a ``lloyd2`` codec regenerates its
    rotation on this machine and refuses to load where that differs).
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    if verify:
        _check_manifest(root)
    cert_path = root / CERTIFICATE_FILE
    if not cert_path.is_file():
        raise ArtifactError(f"{CERTIFICATE_FILE} is missing")
    try:
        cert = Certificate.load(cert_path)
    except CertificateFormatError as exc:
        raise ArtifactError(f"{CERTIFICATE_FILE} cannot be read: {exc}") from exc
    pipeline = compressed = None
    if cert.data.get("candidate") is not None:
        pipeline, compressed = _load_deployed(root, cert, verify)
    elif verify and ((root / CODES_FILE).exists() or (root / FITTED_FILE).exists()):
        raise ArtifactError("the certificate records no deployed candidate (abstain) but the "
                            "bundle holds codes or fitted state")
    evidence = _load_evidence(root / EVIDENCE_FILE)
    report = None
    if verify:
        report = verify_certificate(cert, evidence=evidence, pipeline=pipeline,
                                    compressed=compressed, key=key, strict_code=strict_code)
        if not report.ok:
            raise ArtifactError("the bundle failed verification:\n" + "\n".join(
                f"  {c.name}: {c.detail}" for c in report.failed()))
    return Artifact(cert, pipeline, compressed, evidence, report, root)
