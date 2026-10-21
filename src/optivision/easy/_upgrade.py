"""``index.upgrade(queries)``: the only route to a more compressed layout (FINAL_DESIGN 5.7 step 5).

The catalogue rule sets the ceiling and practice questions can only step down; real questions,
through the unchanged :func:`optivision.api.certify`, are the one way up. Each call is one
attempt, run as

    certify(float16_vectors, queries, None, target=T, error=0.05 * 2**-k,
            reference=RankingReference(depth=1) or LabelReference(qrels),
            space=CandidateSpace.from_families({chain: larger rungs allowed at N}),
            fallback=current pipeline, order_seed=0)

where ``k`` (1, 2, 3, ...) counts the attempts on this index, so the error levels of all attempts
together add up to at most 0.05. The attempt number and the hashes of the questions it uses are
written to the manifest *before* ``certify`` runs: a crash cannot hand an attempt's error level
back, and a question is never tested twice. Questions are de-duplicated (equal after collapsing
white space and case; then exact duplicate vectors) and every question an earlier attempt used
is left out.

* It needs the float16 copy (``keep_float16=True``); without one it says how to get it (a
  rebuild re-encodes every page) and spends nothing.
* Without ``answers`` the reference is agreement with the full-precision top page
  (``[your questions: agreement with full precision, not relevance]``); with ``answers`` it is
  the labels (``[your questions: search quality]``). Agreement is never called quality.
* The candidates are the ladder rungs more compressed than the current layout that the rule
  allows at today's page count, one family per chain, lightest rung first, as fixed before any
  question is looked at. "Allowed" leaves out the rows the rule shuts for this collection
  (``not allowed``, ``outside size range``, ``needs practice questions``); rows below the
  catalogue target stay in, because testing them on real questions is the point.
* The number of questions needed is computed from the betting test itself
  (``core.selection.statistics``), never typed in: the least number for a candidate that
  changes no top result, and the median for one whose top page drops to second place on 1% of
  questions. With agreement as the reference no candidate scores above full precision, and
  with fewer than the least number even one that changes no top result does not pass, so the
  attempt is refused and nothing is spent. (The refusal depends on the count only, never on
  outcomes, so it cannot bias the test. The least number is at or above the floor of
  ``docs/package/guide/SELECTION_AND_CERTIFICATION.md`` 7.1, which assumes the bet at its cap
  from the first question.)
* ``apply=True`` (or ``report.apply()``) re-settles the index into the certified layout from
  the float16 copy, with the pipeline and codes ``certify`` evaluated (condition C4: the index
  deployed is the one evaluated). Nothing else changes the layout.

How the outcome is worded follows the ledger's ``upgrade.wording`` (V8 decides it).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from . import _claim
from ._check import (
    AGREEMENT,
    QUALITY,
    answer_sets,
    as_query_corpus,
    encode_estimate,
    index_commit,
    index_float16,
    index_manifest,
    index_model,
    index_n_pages,
    index_path,
    index_selection,
    index_store,
    index_target,
    index_vectors_per_page,
)
from ._claim import Block, claim, lit
from ._querylog import QueryLog, existing_log, query_hash, vector_hash
from ._types import EasyError, LayoutRow

#: the error level all attempts on one index share (FINAL_DESIGN 5.7 step 5)
TOTAL_ERROR = 0.05

#: rows the rule shuts for this collection; never upgrade candidates
EXCLUDED_STATUSES = ("not allowed", "outside size range", "needs practice questions")

#: the "mean agreement" of the median figure, and how its misses are modelled
MEDIAN_AGREEMENT = 0.99
MEDIAN_RUNS = 400
MEDIAN_SEED = 0
HORIZON = 4000

#: manifest key of the upgrade bookkeeping
MANIFEST_KEY = "upgrade"


def attempt_error(k: int) -> float:
    """The error level of attempt ``k`` (1-based): ``0.05 * 2**-k``; the sum over all k <= 0.05."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"attempts are counted from 1, got {k!r}")
    return TOTAL_ERROR * 2.0 ** -k


# ============================================================================ queries needed


def first_crossing(y: Any, m0: float, level: float, *, stop_after: int | None = None) -> np.ndarray:
    """Per row, the first sample size at which the betting p-value of
    :func:`optivision.core.selection.statistics.betting_pvalues` is at most ``level`` (0 when it
    never is within the row, or when the pass stopped early).

    This is the recursion of ``betting_pvalues`` kept step for step (aGRAPA bet, ``BET_CAP``,
    running maximum of the wealth), evaluated once per step instead of only at the end, so the
    crossing of many simulated rows costs one pass; ``tests/test_easy_upgrade.py`` checks it
    against ``betting_pvalues`` itself. ``stop_after`` ends the pass once that many rows have
    crossed (enough for a median)."""
    from optivision.core.selection.statistics import BET_CAP

    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    lead = y.shape[:-1]
    log_k = np.zeros(lead)
    log_kmax = np.zeros(lead)
    s1 = np.zeros(lead)
    s2 = np.zeros(lead)
    cap = BET_CAP / m0
    first = np.zeros(lead, dtype=np.int64)
    for t in range(y.shape[-1]):
        mu = (0.5 + s1) / (t + 1)
        var = (0.25 + s2) / (t + 1)
        d = mu - m0
        lam = np.clip(d / (var + d * d), 0.0, cap)
        yt = y[..., t]
        log_k = log_k + np.log1p(lam * (yt - m0))
        log_kmax = np.maximum(log_kmax, log_k)
        hit = (first == 0) & (np.minimum(1.0, np.exp(-log_kmax)) <= level)
        first[hit] = t + 1
        s1 = s1 + yt
        mu_t = (0.5 + s1) / (t + 2)
        s2 = s2 + (yt - mu_t) ** 2
        crossed = int(np.count_nonzero(first))
        if crossed == first.size or (stop_after is not None and crossed >= stop_after):
            break
    return first


@lru_cache(maxsize=64)
def queries_needed(target: float, level: float, *, horizon: int = HORIZON) -> int | None:
    """The least number of questions with which a candidate that changes no top result (every
    question scores 1 for it and for full precision) reaches ``level``; None beyond ``horizon``.

    Under the agreement reference this is the best case: no candidate scores above full
    precision, and a candidate that changes no top result gives the largest statistic on every
    question."""
    from optivision.core.selection.statistics import to_y

    y = to_y(np.ones(horizon), np.ones(horizon), target)
    n = int(first_crossing(y, target / (1.0 + target), level)[0])
    return n or None


@lru_cache(maxsize=64)
def queries_median(target: float, level: float, *, agreement: float = MEDIAN_AGREEMENT,
                   runs: int = MEDIAN_RUNS, seed: int = MEDIAN_SEED, horizon: int = HORIZON) -> float | None:
    """The median number of questions at which a candidate with mean agreement ``agreement``
    reaches ``level``, simulated: on a share of questions its top page drops to second place
    (nDCG@5 of ``1 / log2(3)``), the share chosen so the mean is ``agreement``; ``runs`` seeded
    runs. None when the median run does not cross within ``horizon``."""
    from optivision.core.selection.statistics import to_y

    second = 1.0 / math.log2(3.0)
    miss = (1.0 - agreement) / (1.0 - second)
    rng = np.random.default_rng(seed)
    c = np.where(rng.random((runs, horizon)) < miss, second, 1.0)
    # the median of ``runs`` crossing times is fixed once runs // 2 + 1 rows have crossed
    first = first_crossing(to_y(c, np.ones_like(c), target), target / (1.0 + target), level,
                           stop_after=runs // 2 + 1)
    n = np.where(first == 0, np.iinfo(np.int64).max, first)
    med = float(np.median(n))
    return med if med < horizon else None


# ============================================================================ candidates


def _nominal_ratio(spec: Mapping[str, Any]) -> float:
    """Size against float16 from the stages alone (16 / bits per number / share of vectors kept),
    used only to order rungs; it is never shown."""
    bits, keep = 32.0, 1.0
    for stage in spec.get("stages", []):
        name = stage.get("stage")
        if name in ("hierarchical_merge", "adaptive_merge") and stage.get("ratio"):
            keep *= float(stage["ratio"])
        elif name == "float16":
            bits = 16.0
        elif name == "int8":
            bits = 8.0
        elif name == "int4":
            bits = 4.0
        elif name == "lloyd2":
            bits = 2.0
        elif name == "binary":
            bits = 1.0
    return 16.0 / bits / keep


def candidate_families(rows: Sequence[LayoutRow], current: LayoutRow) -> dict[str, list[LayoutRow]]:
    """``{chain: [rows]}`` of the rungs more compressed than ``current`` that the rule allows at
    this size, in ladder order (lightest first within a chain); fixed before any question."""
    now = _nominal_ratio(current.spec)
    families: dict[str, list[LayoutRow]] = {}
    for row in rows:
        if row.name == current.name or row.status in EXCLUDED_STATUSES:
            continue
        if _nominal_ratio(row.spec) <= now:
            continue
        families.setdefault(row.chain, []).append(row)
    return families


# ============================================================================ the report


@dataclass
class UpgradeReport:
    """What one ``upgrade()`` call did. ``outcome`` is ``"certified"`` (a more compressed layout
    passed), ``"kept"`` (certify ran; nothing passed, the current layout stays) or ``"not run"``
    (nothing was spent; ``reason`` says why). ``result`` is certify's ``CertifiedResult``."""

    outcome: str
    reason: str
    target: float
    current: LayoutRow
    reference: str
    attempt: int | None = None
    error: float | None = None
    n_given: int = 0
    n_repeats: int = 0
    n_reused: int = 0
    n_unanswered: int = 0
    n_tested: int = 0
    needed: int | None = None
    median: float | None = None
    families: dict[str, list[str]] = field(default_factory=dict)
    layout: LayoutRow | None = None
    measured: float | None = None
    result: Any = None
    applied: bool = False
    block: Block = field(default_factory=lambda: Block([]))
    _index: Any = field(default=None, repr=False, compare=False)

    @property
    def certified(self) -> bool:
        return self.outcome == "certified"

    @property
    def lines(self) -> list[_claim.Line]:
        return self.block.lines

    def __str__(self) -> str:
        return str(self.block)

    def __repr__(self) -> str:
        return str(self.block)

    def as_dict(self) -> dict[str, Any]:
        cert = getattr(self.result, "certificate", None)
        return {
            "outcome": self.outcome, "certified": self.certified, "reason": self.reason,
            "target": self.target, "current": self.current.name, "reference": self.reference,
            "kind": QUALITY if self.reference == "labels" else AGREEMENT,
            "attempt": self.attempt, "error": self.error, "n_given": self.n_given,
            "n_repeats": self.n_repeats, "n_reused": self.n_reused, "n_unanswered": self.n_unanswered,
            "n_tested": self.n_tested, "needed": self.needed, "median": self.median,
            "families": {k: list(v) for k, v in self.families.items()},
            "layout": None if self.layout is None else self.layout.name, "measured": self.measured,
            "certificate": None if cert is None else _digest(cert), "applied": self.applied,
            "block": self.block.as_dict(),
        }

    def apply(self, *, out: Callable[[str], Any] | None = print) -> Any:
        """Re-settle the index into the certified layout from the float16 copy (prints first).
        Returns what the index's re-settle path returns; None when nothing was certified."""
        if not self.certified:
            if out is not None:
                block = Block([], next="index.search(...)")
                block.add("Apply", "nothing to apply: no more compressed layout passed")
                out(str(block))
            return None
        if self.applied:
            raise EasyError("this upgrade was applied already")
        value = resettle(self._index, self, out=out)
        self.applied = True
        return value


def _digest(cert: Any) -> str | None:
    d = getattr(cert, "digest", None)
    return str(d() if callable(d) else d) if d is not None else None


def resettle(index: Any, report: UpgradeReport, *, out: Callable[[str], Any] | None = print) -> Any:
    """Hand certify's fitted pipeline and codes to the index (``Index._resettle``): the stored
    layout becomes exactly what was evaluated."""
    hook = getattr(index, "_resettle", None)
    if not callable(hook):
        raise EasyError("this index cannot re-settle in place; rebuild it with "
                        f"layout={report.layout.name!r} and rebuild=True")
    if out is not None:
        block = Block([], next="the change runs now")
        block.add("Change", "new layout {0} ({1})", lit(report.layout.plain), lit(report.layout.name))
        block.add("Time", "no re-encoding: the codes come from the kept float16 copy, as tested")
        out(str(block))
    result = report.result
    updates = {"upgraded": {"attempt": report.attempt, "error": report.error, "layout": report.layout.name,
                            "reference": report.reference, "n_tested": report.n_tested,
                            "certificate": _digest(result.certificate)}}
    return hook(pipeline=result.pipeline, compressed=result.compressed, layout=report.layout,
                certificate=result.certificate, updates=updates)


# ============================================================================ run_upgrade


def _bookkeeping(index: Any) -> dict[str, Any]:
    entry = dict(index_manifest(index).get(MANIFEST_KEY) or {})
    entry.setdefault("attempts", 0)
    entry.setdefault("used_sha256", [])
    entry.setdefault("history", [])
    return entry


def used_hashes(index: Any) -> set[str]:
    """Hashes of every question an earlier attempt on this index used (also those the build's
    real-query gate recorded under ``manifest["upgrade"]["used_sha256"]``)."""
    return set(_bookkeeping(index)["used_sha256"])


def _queries_from_log(index: Any, queries: Any) -> list[str]:
    log = queries if isinstance(queries, QueryLog) else (getattr(index, "query_log", None)
                                                         or _log_of(index))
    if log is None:
        raise ValueError("upgrade() needs real questions: pass queries=[...], or load the index with "
                         "log_queries=True and search for a while")
    return log.queries(exclude=used_hashes(index))


def _log_of(index: Any) -> QueryLog | None:
    path = index_path(index)
    return existing_log(path) if path is not None else None


def _not_run(index: Any, reason: str, *lines: tuple[str, str, tuple[Any, ...]], next_: str,
             **fields: Any) -> UpgradeReport:
    block = Block([], next=next_)
    for label, template, parts in lines:
        block.add(label, template, *parts)
    sel = fields.pop("reference", "agreement")
    return UpgradeReport("not run", reason, index_target(index), index.layout, sel, block=block,
                         _index=index, **fields)


def run_upgrade(index: Any, queries: Any = None, *, answers: Any = None, apply: bool = False,
                out: Callable[[str], Any] | None = print) -> UpgradeReport:
    """One upgrade attempt on ``index`` with real ``queries`` (strings, vectors, a ``QueryLog``,
    or None for the index's query log). See the module docstring."""
    from . import _ledger, _levels

    _ledger.require("upgrade")
    target = index_target(index)
    current: LayoutRow = index.layout
    reference = "labels" if answers is not None else "agreement"
    kind = QUALITY if reference == "labels" else AGREEMENT

    f16 = index_float16(index)
    if f16 is None:
        est = encode_estimate(index, index_n_pages(index))
        lines = [("Upgrade", "needs the float16 copy, which this index does not keep", ())]
        if est is not None:
            lines.append(("", "rebuilding with keep_float16=True re-encodes every page: {0}", (est,)))
        else:
            lines.append(("", "rebuilding with keep_float16=True re-encodes every page", ()))
        lines.append(("", "nothing was spent: no attempt is counted", ()))
        return _not_run(index, "no float16 copy", *lines, reference=reference,
                        next_="ov.build(..., keep_float16=True, rebuild=True), then index.upgrade(queries)")

    # ------------------------------------------------------------ questions: dedupe, never reuse
    if queries is None or isinstance(queries, QueryLog):
        if answers is not None:
            raise ValueError("answers= needs the questions they answer: pass queries=[...]")
        queries = _queries_from_log(index, queries)
    if isinstance(queries, str):
        queries = [queries]
    if isinstance(queries, Sequence) and len(queries) == 0:
        return _not_run(index, "no new questions",
                        ("Questions", ("none to test (the query log holds no question an earlier attempt "
                                       "did not use)"), ()),
                        ("", "nothing was spent", ()), reference=reference, next_="index.upgrade(new_questions)")
    texts = isinstance(queries, Sequence) and all(isinstance(t, str) for t in queries)
    if texts:
        items = list(queries)
        hashes = [query_hash(t) for t in items]
    else:
        q_all, _ = as_query_corpus(index, queries)
        items = list(range(len(q_all)))
        hashes = [vector_hash(q_all.vectors[q_all.offsets[i]:q_all.offsets[i + 1]]) for i in items]
    if answers is not None and (not isinstance(answers, Sequence) or len(answers) != len(items)):
        raise ValueError(f"answers needs one entry per query ({len(items)})")
    used = used_hashes(index)
    keep, seen, repeats, reused = [], set(), 0, 0
    for i, h in enumerate(hashes):
        if h in seen:
            repeats += 1
            continue
        seen.add(h)
        if h in used:
            reused += 1
            continue
        keep.append(i)
    n_given = len(items)

    if not keep:
        return _not_run(
            index, "no new questions",
            ("Questions", "{0} given: {1} repeats, {2} used by an earlier attempt; none new",
             (claim(n_given, "counted"), claim(repeats, "counted"), claim(reused, "counted"))),
            ("", "a question is tested once per index; nothing was spent", ()),
            reference=reference, n_given=n_given, n_repeats=repeats, n_reused=reused,
            next_="index.upgrade(new_questions)")

    if texts:
        q, _ = as_query_corpus(index, [items[i] for i in keep])
    else:
        q = q_all.select(keep)
    # exact duplicate vectors (two texts the model encodes alike, or a question given once as
    # text and once as vectors) are repeats too; both hashes of a tested question are recorded
    vec_hashes = [vector_hash(q.vectors[q.offsets[j]:q.offsets[j + 1]]) for j in range(len(q))]
    first_of: dict[str, int] = {}
    rows = []
    for j, h in enumerate(vec_hashes):
        if h in first_of:
            repeats += 1
            continue
        if h in used:
            reused += 1
            continue
        first_of[h] = j
        rows.append(j)
    if not rows:
        return _not_run(
            index, "no new questions",
            ("Questions", "{0} given: {1} repeats, {2} used by an earlier attempt; none new",
             (claim(n_given, "counted"), claim(repeats, "counted"), claim(reused, "counted"))),
            ("", "a question is tested once per index; nothing was spent", ()),
            reference=reference, n_given=n_given, n_repeats=repeats, n_reused=reused,
            next_="index.upgrade(new_questions)")
    from optivision.representation import MultiVectorCorpus

    q = q.select(rows)
    q = MultiVectorCorpus(q.vectors, q.offsets, ids=[f"q{j:05d}" for j in range(len(q))])
    kept_idx = [keep[j] for j in rows]
    kept_hashes = [{hashes[keep[j]], vec_hashes[j]} for j in rows]

    qrels = None
    unanswered = 0
    if answers is not None:
        sets = answer_sets(list(index_store(index).ids), [answers[i] for i in kept_idx], len(q))
        answered = [j for j, s in enumerate(sets) if s]
        unanswered = len(q) - len(answered)
        if not answered:
            return _not_run(index, "no answered questions",
                            ("Questions", "none of the new questions has an answer in this index", ()),
                            ("", "nothing was spent", ()), reference=reference, n_given=n_given,
                            n_repeats=repeats, n_reused=reused, n_unanswered=unanswered,
                            next_="index.upgrade(queries, answers=[...]) with answers that are pages of this index")
        q = q.select(answered)
        q = MultiVectorCorpus(q.vectors, q.offsets, ids=[f"q{j:05d}" for j in range(len(q))])
        qrels = {q.ids[n]: sorted(sets[j]) for n, j in enumerate(answered)}
        kept_hashes = [kept_hashes[j] for j in answered]
    n_tested = len(q)

    # ------------------------------------------------------------ candidates at today's size
    sel = index_selection(index)
    today = _levels.select(getattr(sel, "model", None) or index_model(index), target, index_n_pages(index),
                           checked=bool(getattr(sel, "checked", False)), modality=getattr(sel, "modality", None),
                           vectors_per_page=index_vectors_per_page(index))
    families = candidate_families(today.rows, current)
    common = {"reference": reference, "n_given": n_given, "n_repeats": repeats, "n_reused": reused,
              "n_unanswered": unanswered, "n_tested": n_tested}
    if not families:
        return _not_run(index, "no larger layout",
                        ("Upgrade", "no layout more compressed than {0} is allowed at this size",
                         (lit(current.plain),)),
                        ("", "nothing was spent: no attempt is counted", ()),
                        next_="index.search(...)", **common)

    book = _bookkeeping(index)
    attempt = int(book["attempts"]) + 1
    error = attempt_error(attempt)
    level = error / len(families)
    needed = queries_needed(target, level)
    median = queries_median(target, level)
    fam_names = {chain: [r.name for r in rows_] for chain, rows_ in families.items()}
    needed_line = _needed_line(needed, median, error)
    if reference == "agreement" and (needed is None or n_tested < needed):
        return _not_run(
            index, "too few questions",
            ("Questions", "{0} new questions to test", (claim(n_tested, "counted"),)),
            needed_line,
            ("", "with fewer, even a layout that changes no top result cannot pass; nothing was spent", ()),
            next_="collect more real questions (load(..., log_queries=True) keeps them), then index.upgrade()",
            needed=needed, median=median, families=fam_names, attempt=None, error=None, **common)

    # ------------------------------------------------------------ spend the attempt, then run
    book["attempts"] = attempt
    book["used_sha256"] = sorted(set(book["used_sha256"]).union(*kept_hashes))
    book["history"] = list(book["history"]) + [{"attempt": attempt, "error": error, "n_tested": n_tested,
                                                 "reference": reference, "outcome": "started"}]
    if not index_commit(index, **{MANIFEST_KEY: book}):
        return _not_run(index, "cannot record the attempt",
                        ("Upgrade", ("the attempt must be recorded before it runs, and this index could not "
                                     "be written (read-only, or another build or add() holds it)"), ()),
                        ("", "nothing was spent", ()), next_="close other writers, then index.upgrade() again",
                        needed=needed, median=median, families=fam_names, **common)

    from optivision.api import certify
    from optivision.compose import Pipeline
    from optivision.core.evaluate.reference import LabelReference, RankingReference
    from optivision.core.selection.spaces import CandidateSpace

    flat = [row for rows_ in families.values() for row in rows_]
    space = CandidateSpace.from_families(
        {chain: [Pipeline.from_dict(dict(r.spec)) for r in rows_] for chain, rows_ in families.items()},
        name="easy-upgrade")
    ref = LabelReference(qrels) if qrels is not None else RankingReference(depth=1)
    result = certify(f16, q, None, target=target, error=error, metric="ndcg@5", reference=ref,
                     space=space, fallback=dict(current.spec), order_seed=0)

    selected = result.selected
    certified = bool(result.certified) and selected is not None and selected.position >= 0
    layout = flat[selected.position] if certified else None
    measured = None
    if certified:
        ev = result.evidence
        base = float(np.sum(ev.baseline))
        measured = float(np.sum(ev.candidates[selected.position]) / base) if base > 0 else None
    book["history"][-1].update({"outcome": "certified" if certified else "kept",
                                "layout": layout.name if layout is not None else None,
                                "certificate": _digest(result.certificate)})
    index_commit(index, **{MANIFEST_KEY: book})
    _save_certificate(index, result, attempt)

    report = UpgradeReport("certified" if certified else "kept", "", target, current, reference, attempt, error,
                           n_given, repeats, reused, unanswered, n_tested, needed, median, fam_names, layout,
                           measured, result, False, Block([]), index)
    report.block = _render(report, needed_line, kind)
    if apply and certified:
        report.apply(out=out)
        report.block.next = "index.search(...)  (the index now uses the upgraded layout)"
    return report


def _save_certificate(index: Any, result: Any, attempt: int) -> None:
    path = index_path(index)
    if path is None:
        return
    # the digest is in the manifest; the file is a convenience, so a failure to write it is ignored
    with suppress(Exception):
        folder = path / "upgrade"
        folder.mkdir(parents=True, exist_ok=True)
        result.certificate.save(folder / f"attempt_{attempt:03d}.certificate.json")


def _needed_line(needed: int | None, median: float | None, error: float) -> tuple[str, str, tuple[Any, ...]]:
    src = "the betting test's own arithmetic at this error level; median from a seeded simulation"
    if needed is None:
        return ("Needed", "more questions than we can compute at this target and error level", ())
    parts: list[Any] = [claim(needed, "estimate", source=src)]
    template = "at least {0} questions if no top result changes"
    if median is not None:
        template += "; median {1} at mean agreement {2}"
        parts += [claim(round(median), "estimate", source=src),
                  claim(MEDIAN_AGREEMENT, "estimate", source=src, text=_claim.pct(MEDIAN_AGREEMENT, 0))]
    return ("Needed", template, tuple(parts))


def _render(r: UpgradeReport, needed_line: tuple[str, str, tuple[Any, ...]], kind: str) -> Block:
    from . import _ledger

    b = Block([])
    b.add("Upgrade", "attempt {0} on this index, error level {1}; all attempts together stay within {2}",
          claim(r.attempt, "counted"), claim(r.error, "counted", text=_pct_exact(r.error)),
          claim(TOTAL_ERROR, "counted", text=_pct_exact(TOTAL_ERROR)))
    b.add("Questions", "{0} given: {1} repeats and {2} used before left out; {3} tested",
          claim(r.n_given, "counted"), claim(r.n_repeats, "counted"), claim(r.n_reused, "counted"),
          claim(r.n_tested, "counted"))
    if r.n_unanswered:
        b.add("", "{0} questions without an answer in this index were not tested", claim(r.n_unanswered, "counted"))
    label, template, parts = needed_line
    b.add(label, template, *parts)
    n_layouts = sum(len(v) for v in r.families.values())
    b.add("Layouts", "{0} more compressed than {1}, tested in a fixed order; test families: {2}",
          claim(n_layouts, "counted"), lit(r.current.plain), claim(len(r.families), "counted"))
    b.add("Reference", "{0}", lit("your answers (search quality)" if r.reference == "labels"
                                  else "agreement with the full-precision top page (not relevance)"))
    if r.certified and r.layout is not None:
        b.add("Result", "passed: {0} ({1})", lit(r.layout.plain), lit(r.layout.name))
        if r.measured is not None:
            b.add("", "on the tested questions it kept {0}",
                  claim(r.measured, "user-queries", source=kind, construct="nDCG@5 retention",
                        text=_claim.retention(r.measured)))
        b.add("Meaning", "the chance that this procedure passes a layout that keeps less than {0} on questions "
              "like these is at most {1}",
              claim(r.target, "user-queries", source=kind, text=_claim.pct(r.target, 0)),
              claim(r.error, "user-queries", source=kind, text=_pct_exact(r.error)))
        b.add("", "it assumes your questions are a fair sample of the searches this index will serve")
        b.next = "index.upgrade(queries, apply=True), or report.apply()  (settles from the float16 copy)"
    else:
        b.add("Result", "no more compressed layout passed; the current layout stays")
        b.next = "index.search(...); a later attempt needs new questions"
    if str(_ledger.setting("upgrade", "wording", "default")) == "rare":
        b.add("Note", "upgrades are rare: most attempts keep the current layout")
    return b


def _pct_exact(x: float | None) -> str:
    """An error level as a percentage without rounding it up: 0.025 -> "2.5%", 0.0125 -> "1.25%"."""
    if x is None:
        return "n/a"
    text = f"{100 * float(x):.4f}".rstrip("0").rstrip(".")
    return f"{text}%"
