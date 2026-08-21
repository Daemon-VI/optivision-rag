"""Saliency in retrieval space: which patches win against a query codebook.

Spatial saliency asks a pixel-space question -- does this patch carry ink, does
it sit on an edge. That question has no answer on a dense page, which is why the
detector returns 1.03x on ``infovqa``.

A late-interaction score is a sum over query tokens of a max over patches, so a
patch contributes exactly nothing unless it is the arg max for some query token.
On E1 only 8.4% of patches ever win one (``scripts/winner_stats.py``), and a
winner-only index fitted on half the queries retains 100% of nDCG@5 on the other
half at 12.4x smaller. That is the headroom this module tries to reach without
the queries, which a deployment does not have at index time.

The stand-in is a codebook of probe directions. Query vectors live in the same
space as patch vectors -- that is what makes MaxSim work -- so directions fitted
to the corpus itself approximate where queries land. A patch scores by how many
probes it wins, which is the same criterion the oracle uses with real queries.

``random`` probes are offered as a control, and they matter: if random
directions do as well as fitted ones, the fitting is not what is working.
"""

from __future__ import annotations

import numpy as np


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def fit_codebook(
    sample: np.ndarray,
    size: int = 256,
    seed: int = 7,
    iters: int = 10,
    source: str = "kmeans",
) -> np.ndarray:
    """Probe directions in the embedding space, from unlabelled patches.

    ``sample`` is any set of patch vectors -- a few thousand drawn across the
    corpus is plenty. Spherical k-means, because the vectors are L2-normalised
    and MaxSim compares them by inner product: Lloyd's algorithm under cosine is
    assignment by max dot product and an update that renormalises the mean.

    ``source="random"`` returns random unit vectors instead, as the control that
    says whether fitting the probes to the corpus buys anything.
    """
    rng = np.random.default_rng(seed)
    sample = _normalise(np.asarray(sample, dtype=np.float32))
    dim = sample.shape[1]
    size = min(size, sample.shape[0])

    if source == "random":
        return _normalise(rng.standard_normal((size, dim)).astype(np.float32))
    if source != "kmeans":
        raise ValueError(f"unknown codebook source {source!r}")

    centres = sample[rng.choice(sample.shape[0], size, replace=False)].copy()
    for _ in range(iters):
        assign = (sample @ centres.T).argmax(axis=1)
        for k in range(size):
            members = sample[assign == k]
            if members.shape[0]:
                centres[k] = members.mean(axis=0)
            else:
                # An empty cell wastes a probe. Reseed it on a random vector
                # rather than leaving a direction nothing ever matches.
                centres[k] = sample[rng.integers(sample.shape[0])]
        centres = _normalise(centres)
    return centres


def codebook_saliency(
    patch_vectors: np.ndarray,
    codebook: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Per-patch score on the page grid: how many probes this patch wins.

    ``patch_vectors`` is [rows*cols, dim] in grid order. The score is the count
    of codebook directions for which the patch is the arg max, which mirrors
    exactly what MaxSim rewards. Counts are integral and heavily tied at zero --
    most patches win nothing -- so the largest similarity is folded in as a
    tie-break, scaled small enough never to reorder two different win counts.

    Returned on [0, 1] so it can share :func:`build_keep_mask` and its
    ``blank_threshold`` with the pixel-space detector.
    """
    sims = np.asarray(codebook, dtype=np.float32) @ np.asarray(
        patch_vectors, dtype=np.float32
    ).T  # [n_probes, n_patches]
    wins = np.bincount(sims.argmax(axis=1), minlength=patch_vectors.shape[0])

    best = sims.max(axis=0)  # strongest probe response per patch
    best = (best - best.min()) / max(float(np.ptp(best)), 1e-12)  # ndarray.ptp went in numpy 2

    score = wins.astype(np.float32)
    if score.max() > 0:
        score /= score.max()
    # Tie-break sits strictly below one win: 0.5 / n_probes is smaller than the
    # gap between consecutive normalised counts, so it orders patches inside a
    # win count without ever reordering across two.
    score += best * (0.5 / max(len(codebook), 1))
    # Rescale rather than clip. Clipping to [0, 1] would flatten every patch at
    # the top back to the same value and throw the tie-break away exactly where
    # it decides which patches are kept.
    top = float(score.max())
    if top > 0:
        score /= top
    return score.reshape(rows, cols)
