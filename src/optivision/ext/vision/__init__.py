"""The page and vision-language surface of OptiVision, in one namespace.

Everything that knows about page images, patch grids or a vision encoder is listed here,
and nothing in ``optivision.core`` depends on it:

* page types: ``PageRef``, ``PatchGrid``, ``PageEncoding``, ``PrunedPage``;
* pixel-space and grid pruning: ``SpatialPruner`` (a stage that needs page images and
  grid positions), ``TokenPruner``, ``build_keep_mask``, ``dilate_mask``,
  ``patch_saliency``, ``codebook_saliency``, ``fit_codebook``;
* page encoders: ``BaseEncoder``, ``ColVLMEncoder`` (ColPali, ColQwen, ColSmol),
  ``NemotronColEmbedEncoder``, ``SyntheticEncoder`` (an offline stand-in whose numbers say
  nothing about any real model), ``get_encoder``;
* adapters and ingestion: ``PageEncoderAdapter``, ``NemotronAdapter``, ``iter_pages``;
* overlays: ``overlay_mask``, ``overlay_saliency``, ``explain_page``.

Vector-only redundancy pruning (``RedundancyPruner``, ``prune_redundant``) does not need a
page and belongs to the core layer.

This namespace moves no code: each name is the original object, importable from its old
path as before (``optivision.types``, ``optivision.stages``, ``optivision.pruning``,
``optivision.encoders``, ``optivision.adapters``, ``optivision.ingest``, ``optivision.viz``).
Names are resolved when accessed (PEP 562) and cached, so ``import optivision.ext.vision``
does not import torch, transformers or any other optional package; the encoders import those
only when one is constructed.
"""

from __future__ import annotations

import importlib
from typing import Any

#: name -> (module that defines or re-exports it, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    "BaseEncoder": ("optivision.encoders", "BaseEncoder"),
    "ColVLMEncoder": ("optivision.encoders.colvlm", "ColVLMEncoder"),
    "NemotronAdapter": ("optivision.adapters", "NemotronAdapter"),
    "NemotronColEmbedEncoder": ("optivision.encoders.nemotron", "NemotronColEmbedEncoder"),
    "PageEncoderAdapter": ("optivision.adapters", "PageEncoderAdapter"),
    "PageEncoding": ("optivision.types", "PageEncoding"),
    "PageRef": ("optivision.types", "PageRef"),
    "PatchGrid": ("optivision.types", "PatchGrid"),
    "PrunedPage": ("optivision.types", "PrunedPage"),
    "SpatialPruner": ("optivision.stages", "SpatialPruner"),
    "SyntheticEncoder": ("optivision.encoders", "SyntheticEncoder"),
    "TokenPruner": ("optivision.pruning", "TokenPruner"),
    "build_keep_mask": ("optivision.pruning", "build_keep_mask"),
    "codebook_saliency": ("optivision.pruning", "codebook_saliency"),
    "dilate_mask": ("optivision.pruning", "dilate_mask"),
    "explain_page": ("optivision.viz", "explain_page"),
    "fit_codebook": ("optivision.pruning", "fit_codebook"),
    "get_encoder": ("optivision.encoders", "get_encoder"),
    "iter_pages": ("optivision.ingest", "iter_pages"),
    "overlay_mask": ("optivision.viz", "overlay_mask"),
    "overlay_saliency": ("optivision.viz", "overlay_saliency"),
    "patch_saliency": ("optivision.pruning", "patch_saliency"),
}

__all__ = [
    "BaseEncoder",
    "ColVLMEncoder",
    "NemotronAdapter",
    "NemotronColEmbedEncoder",
    "PageEncoderAdapter",
    "PageEncoding",
    "PageRef",
    "PatchGrid",
    "PrunedPage",
    "SpatialPruner",
    "SyntheticEncoder",
    "TokenPruner",
    "build_keep_mask",
    "codebook_saliency",
    "dilate_mask",
    "explain_page",
    "fit_codebook",
    "get_encoder",
    "iter_pages",
    "overlay_mask",
    "overlay_saliency",
    "patch_saliency",
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
