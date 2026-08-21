"""Token pruning: PageEncoding -> PrunedPage."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..config import PruningConfig
from ..types import PageEncoding, PrunedPage
from .codebook import codebook_saliency, fit_codebook
from .redundancy import prune_redundant
from .saliency import patch_saliency
from .spatial import build_keep_mask, dilate_mask

__all__ = [
    "TokenPruner",
    "build_keep_mask",
    "codebook_saliency",
    "dilate_mask",
    "fit_codebook",
    "patch_saliency",
    "prune_redundant",
]


class TokenPruner:
    """Two-stage pruner: blank-region removal, then duplicate collapsing."""

    def __init__(self, cfg: PruningConfig, codebook: np.ndarray | None = None) -> None:
        self.cfg = cfg
        # Fitted once over the corpus and shared by every page: probe directions
        # are a property of the collection, not of a page.
        self.codebook = codebook

    def prune(self, enc: PageEncoding, image: Image.Image | None = None) -> PrunedPage:
        cfg = self.cfg
        grid = enc.grid
        rows, cols = grid.rows, grid.cols
        patch_idx = grid.token_index.reshape(-1)  # positions in enc.embeddings
        n_before = enc.n_tokens

        if not cfg.enabled:
            return PrunedPage(
                ref=enc.ref,
                embeddings=enc.embeddings.copy(),
                kept_token_index=np.arange(n_before, dtype=np.int32),
                keep_mask=np.ones((rows, cols), dtype=bool),
                grid=grid,
                n_tokens_before=n_before,
                saliency=None,
                stats={"stage": "disabled"},
            )

        # -------------------------------------------------- stage 1: spatial
        if cfg.spatial and cfg.saliency == "codebook" and self.codebook is not None:
            # Retrieval space rather than pixel space: score each patch by how
            # many probe directions it is the arg max for. No image needed, and
            # nothing about page density limits what it can find.
            saliency = codebook_saliency(
                enc.embeddings[patch_idx], self.codebook, rows, cols
            )
            keep_mask = build_keep_mask(
                saliency,
                blank_threshold=cfg.blank_threshold,
                keep_ratio=cfg.keep_ratio,
                min_keep=cfg.min_keep,
                dilate=cfg.dilate,
            )
        elif cfg.spatial and image is not None:
            saliency = patch_saliency(
                image, rows, cols, ink_weight=cfg.ink_weight, edge_weight=cfg.edge_weight
            )
            keep_mask = build_keep_mask(
                saliency,
                blank_threshold=cfg.blank_threshold,
                keep_ratio=cfg.keep_ratio,
                min_keep=cfg.min_keep,
                dilate=cfg.dilate,
            )
        else:
            saliency = np.ones((rows, cols), dtype=np.float32)
            keep_mask = np.ones((rows, cols), dtype=bool)

        kept_grid_positions = np.flatnonzero(keep_mask.reshape(-1))
        kept_tokens = patch_idx[kept_grid_positions].astype(np.int32)
        vectors = enc.embeddings[kept_tokens]
        n_after_spatial = int(vectors.shape[0])

        # ---------------------------------------------- stage 2: redundancy
        clusters: list[list[int]] | None = None
        if cfg.redundancy and n_after_spatial > 1:
            sal_flat = saliency.reshape(-1)[kept_grid_positions]
            order = np.argsort(-sal_flat)  # most informative patch leads its cluster
            vectors, clusters = prune_redundant(
                vectors,
                threshold=cfg.redundancy_threshold,
                order=order,
                merge=cfg.redundancy_merge,
            )
            # A merged vector is represented by its cluster leader's token id.
            kept_tokens = np.array([kept_tokens[c[0]] for c in clusters], dtype=np.int32)

        # ------------------------------------------------- text/instruction
        if cfg.keep_text_tokens and enc.text_token_index.size:
            text_vectors = enc.embeddings[enc.text_token_index]
            vectors = np.concatenate([vectors, text_vectors], axis=0)
            kept_tokens = np.concatenate([kept_tokens, enc.text_token_index])

        stats = {
            "n_before": n_before,
            "n_patches": rows * cols,
            "n_after_spatial": n_after_spatial,
            "n_after_redundancy": int(vectors.shape[0]) - int(enc.text_token_index.size)
            if cfg.keep_text_tokens
            else int(vectors.shape[0]),
            "n_after": int(vectors.shape[0]),
            "blank_fraction": 1.0 - (n_after_spatial / max(1, rows * cols)),
            "n_clusters": len(clusters) if clusters is not None else None,
        }

        return PrunedPage(
            ref=enc.ref,
            embeddings=np.ascontiguousarray(vectors, dtype=np.float32),
            kept_token_index=kept_tokens.astype(np.int32),
            keep_mask=keep_mask,
            grid=grid,
            n_tokens_before=n_before,
            saliency=saliency,
            stats=stats,
        )
