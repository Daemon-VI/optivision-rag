"""The spot check: your questions against freshly encoded pages, without a float16 copy
(FINAL_DESIGN 5.5.3 step 6; CONTRACT `_spotcheck`).

For each question, the stored codes give a top 10. Those pages are re-encoded at full precision
from their source files (it needs the model and the files) and the question is scored against
them. When full precision puts another of those pages above the stored top page, the stored
ranking certainly disagrees with full precision for that question: it is **flagged**.

It can only flag. Full precision may prefer a page outside the top 10, which the spot check never
sees, so an unflagged question is not shown to agree. What it can state is an upper limit: the
agreement with the full-precision top page (nDCG@5 against that page, as ``check(queries=)``
measures with a float16 copy) is at most the mean of one per unflagged question and, per flagged
question, the best gain any page that could still be the full-precision top can have. When that
limit is below the target, the layout is too compressed for these questions; the report says so
and ``check().apply()`` can step down. The spot check itself never changes the layout.

Run ``check()`` (drift) first: a model that now encodes differently than at build is flagged here
for reasons that have nothing to do with the layout.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import _check, _claim
from ._check import as_query_corpus, encode_estimate, index_target
from ._claim import Block, claim, lit

#: how many stored top pages per question are re-encoded
TOP = 10

SOURCE = "spot check of each question's top pages, agreement with full precision, not relevance"


def _gain(rank: int) -> float:
    """nDCG@5 of a single relevant page at 1-based ``rank``."""
    return 1.0 / math.log2(rank + 1) if rank <= 5 else 0.0


@dataclass
class SpotCheck:
    """``flagged`` of ``n`` questions certainly disagree with full precision at the top;
    ``bound`` is the upper limit of their agreement (None when nothing could be checked)."""

    n: int
    flagged: int
    skipped: int
    bound: float | None
    below_target: bool
    pages_encoded: int
    target: float
    reason: str = ""
    seconds: float | None = None
    per_query: list[dict[str, Any]] = field(default_factory=list)
    lines: list[_claim.Line] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "flagged": self.flagged, "skipped": self.skipped, "bound": self.bound,
                "below_target": self.below_target, "pages_encoded": self.pages_encoded,
                "target": self.target, "reason": self.reason, "seconds": self.seconds,
                "kind": SOURCE, "per_query": [dict(r) for r in self.per_query]}

    def __str__(self) -> str:
        return str(Block(list(self.lines), next="index.check().apply() if a safer layout is advised"))


def _not_run(n: int, target: float, reason: str) -> SpotCheck:
    block = Block([])
    block.add("Spot check", "not run: {0}", lit(reason))
    return SpotCheck(n, 0, n, None, False, 0, target, reason, lines=block.lines)


def spot_check(index: Any, queries: Any, *, k: int = TOP,
               out: Callable[[str], Any] | None = print) -> SpotCheck:
    """Re-encode each question's stored top ``k`` pages and flag the questions whose full-precision
    order puts another page first. Prints its time before encoding. Never changes the index."""
    from optivision.core.retrieve.exact import ExactRetriever
    from optivision.representation import MultiVectorCorpus

    target = index_target(index)
    q, _ = as_query_corpus(index, queries)
    arrays = [np.asarray(q.vectors[q.offsets[i]:q.offsets[i + 1]]) for i in range(len(q))]
    hits = index.search_vectors(arrays, k=k)
    if not any(h.file for row in hits for h in row):
        return _not_run(len(q), target, "it needs the source files; this index was built from vectors")

    pages: dict[str, tuple[str, int]] = {}
    missing: set[str] = set()
    for row in hits:
        for h in row:
            if h.id in pages or h.id in missing:
                continue
            if h.file and h.page and Path(h.file).exists():
                pages[h.id] = (str(h.file), int(h.page))
            else:
                missing.add(h.id)
    if not pages:
        return _not_run(len(q), target, "none of the top pages' source files exist")

    ids = list(pages)
    if out is not None:
        est = encode_estimate(index, len(ids))
        block = Block([], next="the spot check runs now")
        block.add("Spot check", "about to re-encode {0} pages (the top pages of each question)",
                  claim(len(ids), "counted"))
        if est is not None:
            block.add("Time", "{0}, plus loading the model", est)
        else:
            block.add("Time", "no time estimate for this model")
        out(str(block))

    start = time.perf_counter()
    encoded = index._encoder().encode_pages(_check.page_images(index, [pages[i] for i in ids]), ids)
    fresh = MultiVectorCorpus(encoded.vectors, encoded.offsets, ids=ids)
    scores = ExactRetriever(fresh).score(q)
    seconds = time.perf_counter() - start
    col = {pid: j for j, pid in enumerate(ids)}

    per_query, bounds, flagged, skipped = [], [], 0, 0
    for i, row in enumerate(hits):
        shortlist = [h.id for h in row]
        if not shortlist or shortlist[0] not in col:
            skipped += 1
            per_query.append({"query": q.ids[i], "checked": False})
            continue
        avail = [(r + 1, pid) for r, pid in enumerate(shortlist) if pid in col]
        top_score = float(scores[i, col[shortlist[0]]])
        best_rank, best_pid = max(avail, key=lambda t: (float(scores[i, col[t[1]]]), -t[0]))
        is_flagged = best_pid != shortlist[0] and float(scores[i, col[best_pid]]) > top_score
        if is_flagged:
            flagged += 1
            ranks = [best_rank] + [r + 1 for r, pid in enumerate(shortlist) if pid not in col and r > 0]
            bounds.append(max(_gain(r) for r in ranks))
        else:
            bounds.append(1.0)
        per_query.append({"query": q.ids[i], "checked": True, "flagged": is_flagged,
                          "stored_top": shortlist[0], "full_precision_best": best_pid,
                          "rank_of_best": best_rank})
    n = len(bounds)
    bound = float(np.mean(bounds)) if bounds else None
    below = bound is not None and bound < target

    block = Block([])
    block.add("Spot check", "re-encoded {0} pages: the stored top {1} of {2} questions",
              claim(len(ids), "counted"), claim(k, "counted"), claim(n, "counted"))
    if n:
        block.add("", "flagged {0} of {1}: full precision ranks another of their top pages above the stored top page",
                  claim(flagged, "user-queries", source=SOURCE), claim(n, "user-queries", source=SOURCE))
        block.add("", "agreement with the full-precision top page is at most {0} on these questions",
                  claim(bound, "user-queries", source=SOURCE, construct="nDCG@5 agreement@1, upper limit",
                        text=_claim.pct(bound)))
        if below:
            block.add("", "that is below the target of {0}: a safer layout is advised",
                      claim(target, "user-queries", source=SOURCE, text=_claim.pct(target, 0)))
        else:
            block.add("", "it can only flag: an unflagged question is not shown to agree")
    if skipped:
        block.add("", "{0} questions not checked: their top page's source file is gone",
                  claim(skipped, "counted"))
    return SpotCheck(n, flagged, skipped, bound, below, len(ids), target, "", seconds, per_query, block.lines)
