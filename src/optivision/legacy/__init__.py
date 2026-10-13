"""The original page recipe, named as a layer.

OptiVision began as a page-level pipeline: a YAML ``Config``, ``OptiVisionRAG`` to index a
folder of pages and search it, ``Compressor`` to bit-pack pruned pages, and a benchmark
harness (``bench``) to compare variants. The composable layer (``optivision.core``) came
afterwards and does not depend on any of this.

These names stay supported and import-compatible at their old paths. This namespace moves
no code and copies nothing: every name is the original object, and
``optivision.legacy.Config is optivision.config.Config``. It only says which layer a name
belongs to, so that new code can reach for ``optivision.core`` and know what it is leaving.

* configuration (``optivision.config``): ``Config``, ``CompressionConfig``,
  ``EncoderConfig``, ``IndexConfig``, ``IngestConfig``, ``PruningConfig``, ``SearchConfig``,
  ``available_presets``;
* the pipeline (``optivision.pipeline``): ``OptiVisionRAG``, ``IndexReport``;
* page compression (``optivision.compression``, ``optivision.types``, ``optivision.pruning``):
  ``Compressor``, ``CompressedPage``, ``TokenPruner``;
* search results (``optivision.types``): ``SearchHit``, ``SearchResult``;
* the benchmark harness (``optivision.bench``): ``EncodedCorpus``, ``Variant``,
  ``default_variants``, ``keep_ratio_sweep``, ``codebook_sweep``, ``run_variant``,
  ``run_benchmark``.

``TokenPruner`` is also listed in ``optivision.ext.vision``: it prunes pages, so it
belongs to both. Names are resolved when accessed (PEP 562) and cached, so importing this
namespace does not import the benchmark harness, torch or any other optional package.
"""

from __future__ import annotations

import importlib
from typing import Any

#: name -> (module that defines or re-exports it, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    "CompressedPage": ("optivision.types", "CompressedPage"),
    "CompressionConfig": ("optivision.config", "CompressionConfig"),
    "Compressor": ("optivision.compression", "Compressor"),
    "Config": ("optivision.config", "Config"),
    "EncodedCorpus": ("optivision.bench", "EncodedCorpus"),
    "EncoderConfig": ("optivision.config", "EncoderConfig"),
    "IndexConfig": ("optivision.config", "IndexConfig"),
    "IndexReport": ("optivision.pipeline", "IndexReport"),
    "IngestConfig": ("optivision.config", "IngestConfig"),
    "OptiVisionRAG": ("optivision.pipeline", "OptiVisionRAG"),
    "PruningConfig": ("optivision.config", "PruningConfig"),
    "SearchConfig": ("optivision.config", "SearchConfig"),
    "SearchHit": ("optivision.types", "SearchHit"),
    "SearchResult": ("optivision.types", "SearchResult"),
    "TokenPruner": ("optivision.pruning", "TokenPruner"),
    "Variant": ("optivision.bench", "Variant"),
    "available_presets": ("optivision.config", "available_presets"),
    "codebook_sweep": ("optivision.bench", "codebook_sweep"),
    "default_variants": ("optivision.bench", "default_variants"),
    "keep_ratio_sweep": ("optivision.bench", "keep_ratio_sweep"),
    "run_benchmark": ("optivision.bench", "run_benchmark"),
    "run_variant": ("optivision.bench", "run_variant"),
}

__all__ = [
    "CompressedPage",
    "CompressionConfig",
    "Compressor",
    "Config",
    "EncodedCorpus",
    "EncoderConfig",
    "IndexConfig",
    "IndexReport",
    "IngestConfig",
    "OptiVisionRAG",
    "PruningConfig",
    "SearchConfig",
    "SearchHit",
    "SearchResult",
    "TokenPruner",
    "Variant",
    "available_presets",
    "codebook_sweep",
    "default_variants",
    "keep_ratio_sweep",
    "run_benchmark",
    "run_variant",
]


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), attribute)
    globals()[name] = value  # later lookups find it directly
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
