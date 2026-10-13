"""Certificates and artifact bundles: what was decided, under which assumptions, for which index.

* ``create_certificate`` / ``verify_certificate`` / ``sign_certificate``: a record of one selection
  decision (``select``, ``abstain`` or ``fallback``), its checks, and an optional HMAC.
* ``save_artifact`` / ``load_artifact``: a directory with the certificate, the codes, the fitted
  stage state, optional evidence and a manifest, with no pickle.
* ``code_identity``: SHA-256 of the decision-relevant source files.

The content digest is an integrity check, not security. The optional HMAC is symmetric and gives
no non-repudiation. A certificate carries the selection rule's statement with its assumptions
and nothing wider: it is not a statement about retrieval quality, and it does not transfer to
another corpus, query distribution or candidate list. See ``docs/package/CERTIFICATE_SPEC.md``.
"""

from __future__ import annotations

from .bundle import Artifact, ArtifactError, load_artifact, save_artifact
from .certificate import (
    Certificate,
    CertificateFormatError,
    create_certificate,
    sign_certificate,
    verify_certificate,
)
from .codeid import code_identity, package_info
from .report import Check, VerificationReport

__all__ = [
    "Artifact",
    "ArtifactError",
    "Certificate",
    "CertificateFormatError",
    "Check",
    "VerificationReport",
    "code_identity",
    "create_certificate",
    "load_artifact",
    "package_info",
    "save_artifact",
    "sign_certificate",
    "verify_certificate",
]
