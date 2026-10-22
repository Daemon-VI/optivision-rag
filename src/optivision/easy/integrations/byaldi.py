"""Import a byaldi index folder into an ``optivision.easy`` Index, on CPU (switched off until its test passes).

    from optivision.easy.integrations import byaldi
    index = byaldi.import_index(".byaldi/my_index", target=0.97)   # no GPU; byaldi is not imported

The page vectors byaldi saved are read as they are and built into an Index like any other vectors
(``ov.build(vectors, model=..., ids=...)``), so no page is encoded again. The feature ``byaldi_import``
stays off in ``presets/easy_ledger.json`` until the format test (``tests/test_easy_byaldi.py``) is
recorded; :func:`import_index` and :func:`read_index` then raise ``FeatureOff`` with the reason.

The on-disk format, as byaldi writes and reads it (``byaldi/colpali.py`` on GitHub, AnswerDotAI/byaldi
``main``: ``ColPaliModel._export_index`` and the ``load_from_index`` branch of ``__init__``)::

    <index_root>/<index_name>/
      index_config.json.gz           gzip JSON: model_name, full_document_collection, highest_doc_id,
                                     resize_stored_images, max_image_width, max_image_height, library_version
      embeddings/embeddings_<i>.pt   torch.save of a list of up to 500 tensors [n_tokens, dim], one per page
                                     (the model's dtype, bfloat16 for byaldi's models); i = 0, 500, 1000, ...;
                                     byaldi loads them in numeric order of <i> and concatenates the lists,
                                     and a page's position in that list is its embed_id
      embed_id_to_doc_id.json.gz     {"<embed_id>": {"doc_id": <int>, "page_id": <1-based page>}}
      doc_ids_to_file_names.json.gz  {"<doc_id>": "<path given to index()>" or "In-memory Image"}
                                     (absent in indexes made before byaldi 0.0.2)
      metadata.json.gz               {"<doc_id>": {...}} (optional)
      collection/<i>.json.gz         {"<embed_id>": "<base64 PNG>"}, only with store_collection_with_index

Page ids follow the easy-mode convention ``<doc>::p<page>``: ``<doc>`` is the file's parent folder and name
(``docs/report.pdf``), ``byaldi/doc<doc_id>`` for in-memory images or a missing file name, and
``<doc>~<doc_id>`` where two byaldi documents would get the same name.

Reading the ``.pt`` files needs torch (byaldi itself needs it), and uses ``torch.load(weights_only=True)``,
which refuses pickled code. bfloat16 vectors are widened to float32 exactly. The vectors keep byaldi's
dtype: our benchmark rows were measured on float16 or float32 encodings, and the import says so.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from .. import _ledger
from .._claim import Block, claim, lit
from .._types import EasyError
from ._common import missing_extra

__all__ = ["ByaldiFormatError", "ByaldiIndex", "ByaldiPage", "import_index", "read_index"]

IN_MEMORY = "In-memory Image"


class ByaldiFormatError(EasyError, ValueError):
    """The folder is not a byaldi index, or its files disagree with each other."""


@dataclass(frozen=True)
class ByaldiPage:
    """One byaldi page: ``embed_id`` is its position in byaldi's embedding list."""

    embed_id: int
    doc_id: Any
    page: int
    file: str | None
    page_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ByaldiIndex:
    """A byaldi index read into memory: the vectors as a ``MultiVectorCorpus`` and one record per page."""

    path: Path
    model_name: str | None
    config: dict[str, Any]
    corpus: Any
    pages: list[ByaldiPage]
    source_dtype: str
    n_images: int = 0

    @property
    def ids(self) -> list[str]:
        return [p.page_id for p in self.pages]

    @property
    def files(self) -> dict[str, str]:
        """Page id -> source file, for pages whose file byaldi recorded."""
        return {p.page_id: p.file for p in self.pages if p.file}

    def summary(self) -> Block:
        b = Block([], next=f'byaldi.import_index("{self.path.as_posix()}")')
        b.add("byaldi", "{0} pages from {1} documents in {2}", claim(len(self.pages), "counted"),
              claim(len({p.doc_id for p in self.pages}), "counted"), lit(self.path.as_posix()))
        b.add("Model", "{0}, as recorded in index_config.json.gz", lit(self.model_name or "not recorded"))
        b.add("Vectors", "{0} as byaldi stored them; our benchmark rows used float16 or float32 encodings",
              lit(self.source_dtype))
        if self.n_images:
            b.add("Images", "{0} page images stored with the byaldi index are not imported",
                  claim(self.n_images, "counted"))
        return b


# ---------------------------------------------------------------------------- reading


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _chunk_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ByaldiFormatError(f"{path.name}: byaldi names embedding files embeddings_<number>.pt") from exc


def _load_chunk(path: Path) -> list[Any]:
    try:
        import torch
    except ImportError as exc:
        raise missing_extra("Reading a byaldi index", "torch", "byaldi") from exc
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, torch.Tensor):  # a stacked chunk: one tensor per row
        data = list(torch.unbind(data))
    if not isinstance(data, (list, tuple)):
        raise ByaldiFormatError(f"{path.name} holds {type(data).__name__}, not a list of tensors")
    return list(data)


def _as_matrix(t: Any, embed_id: int) -> tuple[np.ndarray, str]:
    import torch

    if not isinstance(t, torch.Tensor):
        raise ByaldiFormatError(f"embedding {embed_id} is a {type(t).__name__}, not a tensor")
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 2:
        raise ByaldiFormatError(f"embedding {embed_id} has shape {tuple(t.shape)}; byaldi stores [tokens, dim]")
    dtype = str(t.dtype).replace("torch.", "")
    return t.detach().to(torch.float32).contiguous().numpy(), dtype


def _doc_names(doc_ids: list[Any], files: dict[str, str]) -> dict[str, str]:
    names: dict[str, str] = {}
    used: set[str] = set()
    for d in doc_ids:
        f = files.get(str(d))
        if f and f != IN_MEMORY:
            parts = PurePath(f.replace("\\", "/")).parts
            base = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else f"byaldi/doc{d}")
        else:
            base = f"byaldi/doc{d}"
        if base in used:
            base = f"{base}~{d}"
        used.add(base)
        names[str(d)] = base
    return names


def _read(path: str | Path) -> ByaldiIndex:
    """Read a byaldi index folder (no switch; :func:`read_index` is the public, switched entry point)."""
    from ...representation import MultiVectorCorpus

    root = Path(path)
    config_file = root / "index_config.json.gz"
    if not config_file.is_file():
        raise ByaldiFormatError(f"{root} is not a byaldi index: there is no index_config.json.gz")
    config = _read_gzip_json(config_file)
    map_file = root / "embed_id_to_doc_id.json.gz"
    if not map_file.is_file():
        raise ByaldiFormatError(f"{root}: embed_id_to_doc_id.json.gz is missing")
    raw_map = {int(k): v for k, v in _read_gzip_json(map_file).items()}
    files_file = root / "doc_ids_to_file_names.json.gz"
    files = {str(k): str(v) for k, v in _read_gzip_json(files_file).items()} if files_file.is_file() else {}
    meta_file = root / "metadata.json.gz"
    metadata = {str(k): v for k, v in _read_gzip_json(meta_file).items()} if meta_file.is_file() else {}

    chunks = sorted((root / "embeddings").glob("embeddings_*.pt"), key=_chunk_number)
    tensors: list[Any] = []
    for chunk in chunks:
        tensors.extend(_load_chunk(chunk))
    if len(tensors) != len(raw_map) or sorted(raw_map) != list(range(len(tensors))):
        raise ByaldiFormatError(
            f"{root}: {len(tensors)} embeddings in {len(chunks)} files but embed_id_to_doc_id lists "
            f"{len(raw_map)} pages; the index is incomplete")

    arrays, dtypes = [], set()
    for embed_id, t in enumerate(tensors):
        m, dt = _as_matrix(t, embed_id)
        arrays.append(m)
        dtypes.add(dt)
    dims = {a.shape[1] for a in arrays}
    if len(dims) > 1:
        raise ByaldiFormatError(f"{root}: embeddings have different dimensions {sorted(dims)}")

    doc_ids = list(dict.fromkeys(raw_map[e]["doc_id"] for e in range(len(tensors))))
    names = _doc_names(doc_ids, files)
    pages: list[ByaldiPage] = []
    seen: set[str] = set()
    for e in range(len(tensors)):
        entry = raw_map[e]
        d, p = entry["doc_id"], int(entry["page_id"])
        page_id = f"{names[str(d)]}::p{p}"
        if page_id in seen:
            raise ByaldiFormatError(f"{root}: document {d!r} lists page {p} twice")
        seen.add(page_id)
        f = files.get(str(d))
        pages.append(ByaldiPage(e, d, p, None if f in (None, IN_MEMORY) else f, page_id,
                                dict(metadata.get(str(d), {}) or {})))

    dim = dims.pop() if dims else 0
    counts = [a.shape[0] for a in arrays]
    vectors = np.concatenate(arrays) if arrays else np.zeros((0, dim), dtype=np.float32)
    corpus = MultiVectorCorpus(vectors, np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                               ids=[p.page_id for p in pages],
                               attrs={"source": "byaldi", "model": config.get("model_name"),
                                      "byaldi_version": config.get("library_version"),
                                      "source_dtype": ",".join(sorted(dtypes)) or "none"})
    n_images = 0
    collection = root / "collection"
    if config.get("full_document_collection") and collection.is_dir():
        n_images = sum(len(_read_gzip_json(f)) for f in collection.glob("*.json.gz"))
    return ByaldiIndex(root, config.get("model_name"), dict(config), corpus, pages,
                       ",".join(sorted(dtypes)) or "none", n_images)


def read_index(path: str | Path) -> ByaldiIndex:
    """Read a byaldi index folder into a :class:`ByaldiIndex` (switched off until its test passes)."""
    _ledger.require("byaldi_import")
    return _read(path)


# ---------------------------------------------------------------------------- importing


def _registry_model(name: str | None) -> str | None:
    """byaldi's model id if our registry knows it (exactly or by case), else None."""
    if not name:
        return None
    try:
        from .._resolve import resolve  # the rule agent's resolver, when present
    except ImportError:
        resolve = None
    if resolve is not None:
        try:
            info = resolve(name)
        except KeyError:  # UnknownModel and AmbiguousModel are KeyErrors
            return None
        return None if info is None else str(info.model_id)
    from ...adapters import MODELS

    if name in MODELS:
        return name
    lower = {k.lower(): k for k in MODELS}
    return lower.get(name.lower())


def import_index(path: str | Path, *, out: str | Path | None = None, model: Any = "auto",
                 target: float | None = None, progress: bool = True, **build_kwargs: Any) -> Any:
    """Build an ``optivision.easy`` Index from a byaldi index folder, with no GPU and no re-encoding.

    ``out``: the Index folder (default ``<byaldi folder>.ovx`` beside it). ``model="auto"`` uses byaldi's
    recorded model when our registry knows it, else none (the rule for models without rows of their own
    applies, and text queries need ``search_vectors``). Other keywords go to ``ov.build``.
    """
    _ledger.require("byaldi_import")
    imported = _read(path)
    if progress:
        print(imported.summary())
    from .._api import build

    model_id = _registry_model(imported.model_name) if model == "auto" else model
    out = Path(out) if out is not None else imported.path.with_name(imported.path.name + ".ovx")
    return build(imported.corpus, model=model_id, target=target, path=out, ids=imported.ids,
                 progress=progress, **build_kwargs)
