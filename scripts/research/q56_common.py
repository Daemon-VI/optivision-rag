"""Q5/Q6 experimental scorers (reports/research/Q5Q6/PLAN.md). Research code only:
nothing here is part of the optivision package, which is used read-only.

Stores (ColQwen2 DocVQA):
  hot  = HierarchicalMerge(0.25) > BinaryQuantizer      (primary two-tier hot tier)
  hotp = BinaryQuantizer over all vectors               (secondary)
  cold = Int8Quantizer("per_vector") over all vectors   (rescoring tier)

Scorers return float32 [n_queries, n_docs] MaxSim matrices (or, for rescoring, the
[n_queries, candidates] shortlist scores), ranked with optivision.scoring.rank.
"""

from __future__ import annotations

import e1_common as E
import numpy as np
import torch

from optivision.compose import Pipeline
from optivision.scoring import rank, segment_reduce
from optivision.stages import BinaryQuantizer, HierarchicalMerge, Int8Quantizer
from optivision.storage import ExactIndex, TieredIndex

CANDIDATES = 50  # Q4 / R12 TieredIndex policy (PLAN §1); not tuned
TOK_CHUNK = 1024
ROW_BLOCK = 32768
DIM = 128

# np.packbits is big-endian within a byte: element 8b+i sits at bit (7 - i)
_SIGNS = np.array([[1.0 if (k >> (7 - i)) & 1 else -1.0 for i in range(8)] for k in range(256)], dtype=np.float32)


def load():
    ds, _ = E.load_s7b().dataset(E.dataset_arg("colqwen2_docvqa"))
    return ds


def build(corpus):
    hot = Pipeline([HierarchicalMerge(ratio=0.25), BinaryQuantizer()]).compress(corpus)
    hotp = Pipeline([BinaryQuantizer()]).compress(corpus)
    cold = Pipeline([Int8Quantizer("per_vector")]).compress(corpus)
    return hot, hotp, cold


class Int8Direct:
    """The per-vector int8 store laid out as (int8 codes [n, d], decode multiplier c [n]).

    Same information and bytes as the library's interleaved 130-byte rows (128 codes +
    a float16 scale); c = float32(scale) / 127.0 is exactly the library's multiplier.
    """

    def __init__(self, cold):
        codes = cold.codes
        self.offsets = cold.offsets
        self.q = np.ascontiguousarray(codes[:, :DIM]).view(np.int8)
        scale = np.ascontiguousarray(codes[:, DIM:DIM + 2]).view(np.float16).astype(np.float32).ravel()
        self.c = scale / 127.0
        self.tq = torch.from_numpy(self.q)
        self.tc = torch.from_numpy(self.c)

    def sims(self, x, lo, hi):
        """float32 x [t, d] against rows lo:hi, without a decoded float matrix."""
        return torch.ops.aten._weight_int8pack_mm(torch.from_numpy(x), self.tq[lo:hi], self.tc[lo:hi]).numpy()

    def decoded_sims(self, x, lo, hi):
        """Control: the library's decode (float32 products q * c) then a float32 GEMM."""
        d = self.q[lo:hi].astype(np.float32)
        d *= self.c[lo:hi, None]
        return x @ d.T


def _page_blocks(offsets, max_rows):
    blocks, start, n = [], 0, len(offsets) - 1
    while start < n:
        end = start + 1
        while end < n and offsets[end + 1] - offsets[start] <= max_rows:
            end += 1
        blocks.append((start, end))
        start = end
    return blocks


def maxsim(qvec, qoff, doc_offsets, sim_fn, n_docs):
    """Generic MaxSim: sim_fn(x_chunk, lo, hi) -> float32 [tokens, rows]."""
    nq = len(qoff) - 1
    out = np.zeros((nq, n_docs), dtype=np.float32)
    qchunks = _page_blocks(qoff, TOK_CHUNK)
    dblocks = _page_blocks(doc_offsets, ROW_BLOCK)
    for q0, q1 in qchunks:
        t0, t1 = int(qoff[q0]), int(qoff[q1])
        x = np.ascontiguousarray(qvec[t0:t1], dtype=np.float32)
        loc_q = qoff[q0:q1 + 1] - t0
        for d0, d1 in dblocks:
            lo, hi = int(doc_offsets[d0]), int(doc_offsets[d1])
            s = sim_fn(x, lo, hi)
            per_page = segment_reduce(s, doc_offsets[d0:d1 + 1] - lo, np.maximum, axis=1, fill=-np.inf)
            out[q0:q1, d0:d1] = segment_reduce(per_page, loc_q, np.add, axis=0, fill=0.0)
    return out


# ------------------------------------------------------------------ binary first stage

class BinaryLUT:
    """B1: exact float-query scoring on packed sign bits via per-byte lookup tables."""

    def __init__(self, store):
        self.codes = store.codes
        self.offsets = store.offsets

    def sims(self, x, lo, hi):
        t = x.shape[0]
        nbytes = self.codes.shape[1]
        acc = np.zeros((t, hi - lo), dtype=np.float32)
        codes = self.codes[lo:hi]
        for b in range(nbytes):
            lut = (_SIGNS @ x[:, 8 * b:8 * b + 8].T).T  # [t, 256]: LUT_b[k] for every token
            acc += np.take(lut, codes[:, b], axis=1)
        return acc


class BinaryHamming:
    """B2 (approximate): binarized query bits against code bits; sim = d - 2 popcount(xor)."""

    def __init__(self, store, dim=DIM):
        self.words = np.ascontiguousarray(store.codes).view(np.uint64)
        self.offsets = store.offsets
        self.dim = dim

    def sims(self, x, lo, hi):
        qw = np.packbits(x > 0, axis=1).view(np.uint64)  # same bit rule as the documents
        pc = np.zeros((x.shape[0], hi - lo), dtype=np.int32)
        for w in range(qw.shape[1]):
            pc += np.bitwise_count(qw[:, w][:, None] ^ self.words[lo:hi, w][None, :])
        return (self.dim - 2 * pc).astype(np.float32)


# ------------------------------------------------------------------ rescoring

def rescore_grouped(queries, shortlist, i8: Int8Direct, mode: str):
    """Shortlist scores [nq, m], one kernel call per shortlisted page (no per-query loop).

    mode "direct": int8 codes with the per-row multiplier inside the kernel;
    mode "decoded": control, decode the page then float32 GEMM.
    """
    qoff = queries.offsets
    qvec = queries.vectors
    nq, m = shortlist.shape
    out = np.full((nq, m), -np.inf, dtype=np.float32)
    flat = shortlist.ravel()
    order = np.argsort(flat, kind="stable")
    pages, starts = np.unique(flat[order], return_index=True)
    ends = np.append(starts[1:], len(flat))
    fn = i8.sims if mode == "direct" else i8.decoded_sims
    for p, s0, s1 in zip(pages, starts, ends, strict=True):
        pos = order[s0:s1]  # flat positions (query * m + slot) that shortlisted page p
        qs = pos // m
        lens = (qoff[qs + 1] - qoff[qs]).astype(np.int64)
        rows = np.concatenate([np.arange(qoff[q], qoff[q + 1]) for q in qs])
        x = np.ascontiguousarray(qvec[rows], dtype=np.float32)
        lo, hi = int(i8.offsets[p]), int(i8.offsets[p + 1])
        per_token = fn(x, lo, hi).max(axis=1)
        seg = np.concatenate([[0], np.cumsum(lens)])
        sums = segment_reduce(per_token, seg, np.add, axis=0, fill=0.0)
        out[qs, pos % m] = sums
    return out


def merge_tiers(hot, shortlist, cold):
    """TieredIndex.score's construction of the final matrix (copied, unchanged)."""
    out = hot.copy()
    floor = np.nanmin(np.where(np.isfinite(cold), cold, np.nan), axis=1)
    top_hot = np.nanmax(np.where(np.isfinite(hot), hot, np.nan), axis=1)
    shift = np.where(np.isfinite(floor), top_hot - floor + 1.0, 0.0)
    out -= shift[:, None].astype(np.float32)
    np.put_along_axis(out, shortlist, cold, axis=1)
    return out


def current_two_tier(hot_store, cold_store, queries):
    idx = TieredIndex(ExactIndex(hot_store), ExactIndex(cold_store), candidates=CANDIDATES)
    return idx.score(queries), idx


def experimental_two_tier(hot_scorer, i8, queries, n_docs, rescore_mode="direct", hot_scores=None):
    hot = hot_scores if hot_scores is not None else maxsim(queries.vectors, queries.offsets, hot_scorer.offsets,
                                                           hot_scorer.sims, n_docs)
    shortlist = rank(hot, CANDIDATES)
    cold = rescore_grouped(queries, shortlist, i8, rescore_mode)
    return merge_tiers(hot, shortlist, cold), hot, shortlist
