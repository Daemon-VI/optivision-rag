"""MaxSim with the per-page max on a torch device: the scores of ``scoring.maxsim_matrix``, faster.

With ``OPTIVISION_SCORE_DEVICE`` set, :func:`optivision.scoring.maxsim_matrix` runs only the
query x document product on the device and brings the whole similarity block back for numpy to
take the per-page max. On SciDocs that copy was most of the time (about 650 s per pipeline).
:func:`maxsim_matrix` here takes the max on the device too (``scatter_reduce`` with ``amax``,
which is exact), so only the page maxima come back; the per-query sum is the same numpy call.
The scores are bit-identical to ``scoring.maxsim_matrix`` with the same device
(``tests/test_followup_study_helpers.py``).

``scoring.py`` and ``calibration.py`` are byte-pinned for the X1-X3 experiments, so this lives
beside the study scripts, not in the package: :func:`use_for_calibration` makes
``calibration.score_space`` score with it, for the process that calls it (``s17_int4_norm.py``).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from optivision.scoring import (
    DEFAULT_MAX_BLOCK_BYTES,
    VectorStore,
    _page_blocks,
    _query_chunks,
    _to_device,
    segment_reduce,
)
from optivision.scoring import maxsim_matrix as _numpy_maxsim_matrix


def _page_max(qv_dev: Any, block: np.ndarray, local_d: np.ndarray, device: str) -> np.ndarray:
    """Per query token, the best document vector within each page: ``[tokens, pages]``."""
    import torch

    counts = np.diff(local_d)
    with torch.no_grad():
        sims = qv_dev @ torch.from_numpy(block).to(device).T
        page = torch.repeat_interleave(torch.arange(counts.size, device=device), torch.from_numpy(counts).to(device))
        out = torch.full((sims.shape[0], counts.size), float("-inf"), dtype=sims.dtype, device=device)
        out.scatter_reduce_(1, page.expand_as(sims), sims, reduce="amax")
        del sims
        return out.cpu().numpy()


def maxsim_matrix(
    queries: VectorStore,
    docs: VectorStore,
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
    query_transform: Any = None,
) -> np.ndarray:
    """``scoring.maxsim_matrix`` with the per-page max on ``OPTIVISION_SCORE_DEVICE``.

    Without a device it is ``scoring.maxsim_matrix`` itself. The blocks, chunks, empty-page and
    empty-query handling are the same.
    """
    device = os.environ.get("OPTIVISION_SCORE_DEVICE") or None
    if device is None:
        return _numpy_maxsim_matrix(queries, docs, max_block_bytes, query_transform)
    n_q, n_d = len(queries), len(docs)
    out = np.zeros((n_q, n_d), dtype=np.float32)
    if n_q == 0 or n_d == 0:
        return out

    q_off = np.asarray(queries.offsets, dtype=np.int64)
    d_off = np.asarray(docs.offsets, dtype=np.int64)
    empty_docs = np.diff(d_off) == 0
    total_q_tokens = int(q_off[-1])
    if total_q_tokens == 0:
        out[:, empty_docs] = -np.inf
        return out

    q_chunk_tokens = max(1, min(total_q_tokens, max_block_bytes // (4 * 4096)))
    for q0, q1 in _query_chunks(q_off, q_chunk_tokens):
        t0, t1 = int(q_off[q0]), int(q_off[q1])
        if t1 == t0:
            continue  # every query in this chunk is empty: scores stay 0
        qv = np.ascontiguousarray(queries.decode_rows(t0, t1), dtype=np.float32)
        if query_transform is not None:
            qv = np.ascontiguousarray(query_transform(qv), dtype=np.float32)
        local_q = q_off[q0 : q1 + 1] - t0
        qv_dev = _to_device(qv, device)
        rows_per_block = max(1, max_block_bytes // (4 * qv.shape[0] + 4 * qv.shape[1] + 1))
        for d0, d1 in _page_blocks(d_off, rows_per_block):
            lo, hi = int(d_off[d0]), int(d_off[d1])
            if hi == lo:
                out[q0:q1, d0:d1] = -np.inf
                continue
            block = np.ascontiguousarray(docs.decode_rows(lo, hi), dtype=np.float32)
            per_token = _page_max(qv_dev, block, d_off[d0 : d1 + 1] - lo, device)
            out[q0:q1, d0:d1] = segment_reduce(per_token, local_q, np.add, axis=0, fill=0.0)
            del per_token, block
    out[:, empty_docs] = -np.inf
    return out


def use_for_calibration() -> None:
    """Make ``calibration.score_space`` (and its single-pipeline scoring) use :func:`maxsim_matrix`."""
    from optivision import calibration

    calibration.maxsim_matrix = maxsim_matrix
