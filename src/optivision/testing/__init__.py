"""Helpers for people who add an adapter, a stage or a backend, and for this project's tests.

Numpy-only, importable without pytest; it imports ``optivision.core`` and the library's own
scoring path and nothing model-specific.

``make_corpus``, ``make_queries``
    Seeded synthetic late-interaction data with planted relevance. They exercise code paths;
    results on them say nothing about any real encoder.
``random_orthogonal``, ``signed_permutation``, ``apply_basis``, ``reorder_rows``
    Changes of the stored vectors that leave every MaxSim score unchanged (up to rounding).
``basis_probe``
    How much one pipeline's compressed scores and top-k move under those changes, on one
    data set. A diagnostic; engineering capability, not a research result.
``check_adapter``, ``check_stage``
    Conformance checks that return a list of problems (empty: the contract holds). They say
    nothing about retrieval quality.
"""

from __future__ import annotations

from .conformance import check_adapter, check_stage
from .probes import basis_probe
from .synthetic import make_corpus, make_queries
from .transforms import apply_basis, random_orthogonal, reorder_rows, signed_permutation

__all__ = [
    "apply_basis",
    "basis_probe",
    "check_adapter",
    "check_stage",
    "make_corpus",
    "make_queries",
    "random_orthogonal",
    "reorder_rows",
    "signed_permutation",
]
