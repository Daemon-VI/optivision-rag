"""Typed configuration for the OptiVision RAG pipeline.

Every knob that changes an experimental result lives here, so a benchmark run
can be reproduced from a single YAML file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml


@dataclass
class EncoderConfig:
    # "colsmol" (256M, CPU-friendly, default) | "colpali" | "colqwen2" | "synthetic"
    backend: str = "colsmol"
    model_name: str | None = None  # overrides the backend default checkpoint
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "auto"  # "auto" | "float32" | "bfloat16" | "float16"
    batch_size: int = 16  # queries per forward pass (a benchmark hands over all of them at once)
    # Pages are always encoded one at a time: a single page is already ~900
    # vectors and 13 image crops, so batching them raises peak memory a lot for
    # very little throughput. This only controls how many finished pages are
    # buffered before a write to the index.
    max_pages_in_flight: int = 4
    # --- synthetic backend only (offline tests / CI, never for reported results)
    synthetic_dim: int = 128
    synthetic_grid: int = 32
    synthetic_layout: str | None = None  # JSON of word boxes, see corpus.py


@dataclass
class PruningConfig:
    enabled: bool = True
    # Spatial (image-space) pruning
    spatial: bool = True
    # Where saliency comes from. "pixel" is ink density and edge energy, which
    # has nothing to say about a dense page. "codebook" scores a patch by how
    # many probe directions it wins, which is what MaxSim actually rewards.
    saliency: str = "pixel"  # "pixel" | "codebook"
    codebook_size: int = 256
    # "farthest" by default: probes chosen for coverage. k-means puts them
    # where patches are dense, which is where the redundant patches are, and
    # it measures far worse than either alternative (scripts/probe_eval.py).
    codebook_source: str = "farthest"  # "farthest" | "random" | "kmeans"
    codebook_sample: int = 20_000  # patches drawn corpus-wide to fit the probes
    codebook_seed: int = 7  # fixed so the probes are reproducible run to run
    ink_weight: float = 0.6  # weight of foreground-pixel density
    edge_weight: float = 0.4  # weight of local gradient energy
    blank_threshold: float = 0.02  # saliency below this is treated as blank
    keep_ratio: float | None = None  # if set, keep exactly this fraction (top-k)
    min_keep: int = 16  # never drop below this many patches
    dilate: int = 1  # grow the keep-mask by N patches (protects glyph edges)
    # Embedding-space redundancy pruning
    redundancy: bool = True
    redundancy_threshold: float = 0.92  # cosine above which neighbours merge
    redundancy_merge: bool = True  # True: average the cluster, False: drop dupes
    # Text/instruction tokens are cheap and highly informative: keep them.
    keep_text_tokens: bool = True


@dataclass
class CompressionConfig:
    enabled: bool = True
    method: str = "binary"  # "binary" | "int8" | "none"
    keep_norm: bool = False  # store a per-vector norm alongside the code
    asymmetric: bool = True  # float query vs binary doc (better ranking)


@dataclass
class IndexConfig:
    backend: str = "numpy"  # "numpy" | "qdrant"
    path: str = "data/index"  # on-disk location (numpy) or Qdrant local path
    collection: str = "optivision"
    qdrant_url: str | None = None  # set to use a Qdrant server instead of local
    qdrant_api_key: str | None = None
    on_disk: bool = True


@dataclass
class SearchConfig:
    top_k: int = 5
    prefilter_k: int = 50  # candidates pulled from the compressed index
    rerank: bool = False  # rerank with full-precision vectors (needs a float cache)
    rerank_cache: str | None = None  # path to the float-vector cache for reranking


@dataclass
class IngestConfig:
    dpi: int = 150
    max_side: int = 1536  # downscale page images beyond this before encoding
    formats: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    seed: int = 1337

    # ---------------------------------------------------------------- loading

    # Typos in a config are the cheapest possible experimental error to make and
    # the most expensive to notice — a misspelled `blank_treshold` would be
    # silently ignored and the run would look like a null result. So unknown
    # keys are rejected at both levels rather than dropped.
    SECTIONS: ClassVar[dict[str, type]] = {
        "encoder": EncoderConfig,
        "pruning": PruningConfig,
        "compression": CompressionConfig,
        "index": IndexConfig,
        "search": SearchConfig,
        "ingest": IngestConfig,
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        data = data or {}
        unknown_top = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown_top:
            raise ValueError(f"unknown config section(s): {sorted(unknown_top)}")

        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            sub_cls = cls.SECTIONS.get(name)
            if sub_cls is None:  # a scalar top-level field such as `seed`
                kwargs[name] = value
                continue
            known = {sf.name for sf in dataclasses.fields(sub_cls)}
            unknown = set(value or {}) - known
            if unknown:
                raise ValueError(f"unknown {name} option(s): {sorted(unknown)}")
            kwargs[name] = sub_cls(**(value or {}))
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def with_overrides(self, **sections: dict[str, Any]) -> Config:
        """Return a copy with per-section fields replaced (used by ablations)."""
        data = self.to_dict()
        for name, updates in sections.items():
            if name not in data:
                raise ValueError(f"unknown config section: {name}")
            data[name].update(updates)
        return Config.from_dict(data)
