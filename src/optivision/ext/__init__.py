"""Extensions: code that belongs to one family of models or one kind of input.

``optivision.core`` works on ``[n_vectors, dim]`` float arrays and knows nothing about a
particular model. Whatever depends on a model family lives out here, in a namespace per
family, and may import the core; the core never imports anything from this package.

``optivision.ext.vision``
    Page and vision-language specifics: page and patch-grid types, pixel-space and
    grid-based pruning, the page encoders, page ingestion and overlays.

A namespace only says which layer a name belongs to. It re-exports the original objects
(no copies, no moved code), so every old import path keeps working and the classes are the
same ones. Names are resolved when accessed and cached, so importing a namespace does not
import torch, transformers or any other optional package.
"""

from __future__ import annotations

__all__ = ["vision"]
