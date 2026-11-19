"""OptiVision easy mode: give a model and a target, get a compressed, searchable index.

    import optivision.easy as ov
    ov.options("vidore/colqwen2-v1.0-hf", target=0.97, pages="my_pdfs/")   # plan: no GPU, no download
    index = ov.build("my_pdfs/", model="vidore/colqwen2-v1.0-hf", target=0.97)
    hits = index.search("What drove operating costs in 2023?", k=5)
    page = hits[0].image()                                                  # PIL image for your VLM

``target=0.97`` means "keep at least 97% of this model's full-precision search quality
(nDCG@5)"; it does not mean "97% of answers are right". The layout comes from our measured
catalogue; questions made from your PDFs can only make it safer, and only your real questions
(``Index.upgrade``) can make it smaller. Every number printed carries one evidence tag.

Importing this package loads no torch, SciPy or transformers. The expert API
(``optivision.optimize``, ``calibrate``, ``certify``) is unchanged; ``Index.explain()``
prints the expert code equivalent to an index.
"""

from __future__ import annotations

import importlib as _importlib
import typing as _typing

from ._types import (
    AmbiguousModel,
    EasyError,
    FeatureOff,
    Hit,
    IndexLocked,
    LayoutRow,
    MissingExtra,
    NoSource,
    NotEnoughDisk,
    RemoteCodeRefused,
    Selection,
    SourceMissing,
    SyncReport,
    TargetMismatch,
    UnknownModel,
    WiringError,
)

#: name -> module that defines it, imported on first use
_LAZY = {
    "options": "._options",
    "Options": "._options",
    "build": "._api",
    "load": "._api",
    "Index": "._api",
    "CheckReport": "._check",
    "UpgradeReport": "._upgrade",
    "Report": "._report",
    "doctor": "._doctor",
    "DoctorReport": "._doctor",
    "glossary": "._glossary",
}

__all__ = [
    "AmbiguousModel",
    "CheckReport",
    "DoctorReport",
    "EasyError",
    "FeatureOff",
    "Hit",
    "Index",
    "IndexLocked",
    "LayoutRow",
    "MissingExtra",
    "NoSource",
    "NotEnoughDisk",
    "Options",
    "RemoteCodeRefused",
    "Report",
    "Selection",
    "SourceMissing",
    "SyncReport",
    "TargetMismatch",
    "UnknownModel",
    "UpgradeReport",
    "WiringError",
    "build",
    "doctor",
    "glossary",
    "load",
    "options",
]

if _typing.TYPE_CHECKING:  # pragma: no cover
    from ._api import Index, build, load
    from ._check import CheckReport
    from ._doctor import DoctorReport, doctor
    from ._glossary import glossary
    from ._options import Options, options
    from ._report import Report
    from ._upgrade import UpgradeReport


def __getattr__(name: str) -> _typing.Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'optivision.easy' has no attribute {name!r}")
    value = getattr(_importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
