"""Deterministic stand-in for a late-interaction VLM.

Downloading a 256M-3B parameter checkpoint is not always possible (CI, an
offline lab machine, a first-run smoke test), but the *rest* of the pipeline —
pruning, quantization, indexing, scoring, metrics — must still be exercisable.

This encoder produces multi-vector page encodings with the same contract as the
real ones (L2-normalised float32, one vector per grid patch, a text-token tail)
using hashed word features laid out on the page grid. Retrieval genuinely works
in this mode, so end-to-end tests are meaningful. Numbers produced here describe
the *compression machinery*, not ColPali quality: never report them as such.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from ..types import PageEncoding, PageRef, PatchGrid
from .base import BaseEncoder, l2_normalise

_WORD_TAIL = 4  # instruction-like text tokens appended to every page


def _hash_vector(token: str, dim: int) -> np.ndarray:
    """Stable pseudo-random unit vector for a token (same across processes)."""
    digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=32).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return l2_normalise(rng.standard_normal(dim).astype(np.float32))


class SyntheticEncoder(BaseEncoder):
    name = "synthetic"

    def __init__(
        self,
        dim: int = 128,
        grid: int = 32,
        layout_path: str | Path | None = None,
        noise: float = 0.05,
        seed: int = 1337,
    ) -> None:
        self._dim = int(dim)
        self._grid = int(grid)
        self._noise = float(noise)
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}
        self._layout: dict[str, list] = {}
        self._warned_missing = False
        if layout_path is not None:
            if not Path(layout_path).exists():
                raise FileNotFoundError(
                    f"synthetic_layout {layout_path!r} does not exist; "
                    "run `optivision make-corpus` first or clear the setting"
                )
            with open(layout_path, encoding="utf-8") as fh:
                self._layout = json.load(fh)

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, token: str) -> np.ndarray:
        if token not in self._cache:
            self._cache[token] = _hash_vector(token, self._dim)
        return self._cache[token]

    # ------------------------------------------------------------------ pages

    def _grid_from_layout(self, ref: PageRef) -> np.ndarray:
        """Accumulate hashed word vectors into the grid cells they fall in.

        Boxes are normalised (x0, y0, x1, y1) in [0, 1] with a top-left origin,
        so they survive any rescaling of the page image.
        """
        g = self._grid
        acc = np.zeros((g, g, self._dim), dtype=np.float32)
        for entry in self._layout.get(ref.page_id, []):
            token, (x0, y0, x1, y1) = entry[0], entry[1]
            c0, c1 = (int(np.clip(v * g, 0, g - 1)) for v in (x0, x1))
            r0, r1 = (int(np.clip(v * g, 0, g - 1)) for v in (y0, y1))
            acc[r0 : r1 + 1, c0 : c1 + 1] += self._vec(token)
        return acc

    def _grid_from_pixels(self, image: Image.Image) -> np.ndarray:
        """Fallback when no word layout is known: hash the pixel content."""
        g = self._grid
        small = np.asarray(image.convert("L").resize((g * 2, g * 2)), dtype=np.float32) / 255.0
        cells = small.reshape(g, 2, g, 2).transpose(0, 2, 1, 3).reshape(g, g, 4)
        basis = np.stack([self._vec(f"__pix{i}") for i in range(4)])  # [4, dim]
        return (1.0 - cells) @ basis

    def encode_pages(
        self, images: Sequence[Image.Image], refs: Sequence[PageRef]
    ) -> list[PageEncoding]:
        out: list[PageEncoding] = []
        for image, ref in zip(images, refs, strict=True):
            g = self._grid
            if self._layout.get(ref.page_id):
                acc = self._grid_from_layout(ref)
            else:
                # Falling back here is legitimate for an arbitrary image, but if
                # a layout was loaded and this page is missing from it, the page
                # ids have drifted and retrieval will look random. Say so once.
                if self._layout and not self._warned_missing:
                    self._warned_missing = True
                    warnings.warn(
                        f"page {ref.page_id!r} is not in the loaded word layout "
                        f"(it has {len(self._layout)} pages, e.g. "
                        f"{next(iter(self._layout))!r}); falling back to pixel "
                        "hashing, so retrieval quality will be meaningless. "
                        "Index the corpus at its pdfs/ directory.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                acc = self._grid_from_pixels(image)

            # Ink density drives the vector magnitude, mirroring how a real VLM
            # emits low-energy vectors over blank paper.
            gray = np.asarray(image.convert("L").resize((g, g)), dtype=np.float32) / 255.0
            ink = (1.0 - gray)[..., None]
            acc = acc + self._noise * ink * self._rng.standard_normal(acc.shape).astype(np.float32)

            # A cell with no ink accumulates exactly zero, which normalises to a
            # zero vector — and sign-quantising a zero vector invents an all-(-1)
            # direction that is pure noise in the index. A real VLM emits a small
            # but definite "blank paper" vector there, so do the same: every
            # blank patch gets the same well-defined direction, which the pruner
            # drops and the redundancy stage would collapse anyway.
            acc = acc + 1e-3 * self._vec("__blank")

            patches = l2_normalise(acc.reshape(g * g, self._dim))
            tail = np.stack([self._vec(f"__instr{i}") for i in range(_WORD_TAIL)])
            embeddings = np.concatenate([patches, tail], axis=0).astype(np.float32)

            out.append(
                PageEncoding(
                    ref=ref,
                    embeddings=embeddings,
                    grid=PatchGrid.contiguous(g, g),
                    image_size=image.size,
                    text_token_index=np.arange(g * g, g * g + _WORD_TAIL, dtype=np.int32),
                    meta={"encoder": self.name},
                )
            )
        return out

    # ---------------------------------------------------------------- queries

    def encode_queries(self, queries: Sequence[str]) -> list[np.ndarray]:
        out = []
        for q in queries:
            tokens = [t for t in _tokenise(q) if t]
            if not tokens:
                tokens = ["__empty"]
            vecs = np.stack([self._vec(t) for t in tokens]).astype(np.float32)
            out.append(l2_normalise(vecs))
        return out


def _tokenise(text: str) -> list[str]:
    return [
        "".join(ch for ch in tok if ch.isalnum()) for tok in text.lower().split() if tok.strip()
    ]
