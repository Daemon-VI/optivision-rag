"""Embedding-space pruning: collapse patches that say the same thing.

Spatial pruning removes patches with *no* content. This stage removes patches
with *duplicate* content — the interior of a filled table cell, a run of
background texture inside a photograph, the middle of a thick rule — which look
salient to a pixel detector but add nothing to a MaxSim score.

Why it is nearly free in ranking terms: MaxSim takes, per query vector, the
maximum similarity over document vectors. If two document vectors are within
cosine 0.92 of each other, the max over the pair and the similarity to their
(renormalised) mean differ by a small amount bounded by their angular spread —
so replacing a tight cluster with its centroid barely moves the score, while
removing one vector per duplicate removes real bytes.
"""

from __future__ import annotations

import numpy as np


def _normalise(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def prune_redundant(
    embeddings: np.ndarray,
    threshold: float = 0.92,
    order: np.ndarray | None = None,
    merge: bool = True,
    max_pairwise: int = 4096,
) -> tuple[np.ndarray, list[list[int]]]:
    """Greedy single-pass clustering of near-duplicate vectors.

    Args:
        embeddings: float32 [n, dim], L2-normalised.
        threshold: cosine similarity at or above which vectors are duplicates.
        order: visiting order (default: as given). Pass descending saliency so
            the most informative vector becomes the cluster representative.
        merge: True averages each cluster, False keeps the representative only.
        max_pairwise: above this many vectors the n^2 similarity matrix is built
            in row blocks instead of all at once.

    Returns:
        (vectors [k, dim], clusters) where ``clusters[i]`` lists the input rows
        that vector i represents.
    """
    n = int(embeddings.shape[0])
    if n == 0:
        return embeddings.reshape(0, embeddings.shape[-1]), []
    if threshold >= 1.0 or n == 1:
        return embeddings.copy(), [[i] for i in range(n)]

    visit = np.arange(n) if order is None else np.asarray(order, dtype=int)
    assigned = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []

    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    sim_full = emb @ emb.T if n <= max_pairwise else None

    for i in visit:
        if assigned[i]:
            continue
        sims = sim_full[i] if sim_full is not None else emb @ emb[i]
        members = np.flatnonzero((sims >= threshold) & (~assigned))
        if members.size == 0:  # can only happen if i was just assigned
            members = np.array([i])
        assigned[members] = True
        # Keep the representative first so ``clusters[k][0]`` is the survivor.
        member_list = [int(i)] + [int(m) for m in members if int(m) != int(i)]
        clusters.append(member_list)

    if merge:
        vectors = np.stack([emb[c].mean(axis=0) for c in clusters])
        vectors = _normalise(vectors).astype(np.float32)
    else:
        vectors = np.stack([emb[c[0]] for c in clusters]).astype(np.float32)

    return vectors, clusters
