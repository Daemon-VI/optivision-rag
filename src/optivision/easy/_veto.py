"""The practice check, the fixed test sequence and the real-query gate
(FINAL_DESIGN 5.5.1, 5.5.2; docs/easy/CONTRACT.md, ``_veto.py``).

Every function here is pure: rankings in, statistics out. Rankings are lists of page ids,
best first; the build module produces them (``_index.rank_float``) and this module never
encodes or scores vectors.

Statistic (FINAL_DESIGN 5.5.1). For each hard question, ``c`` is the candidate layout's
nDCG@5 and ``b`` the full-precision (float16) one. ``R = sum(c) / sum(b)``, and ``U`` is its
one-sided 95% upper bound from a page-cluster bootstrap (B = 2,000 resamples of the clusters,
seed 0, percentile method). The layout passes when ``U >= target``: it is set aside only when
even the optimistic end of the bound is below the target. This is a single look at a fixed
number of questions, which is why a bootstrap bound is used here and the betting test is
kept for ``upgrade()``, where looks are sequential.

Fixed sequence. The pick is tested; if it fails, the next safer layout is tested, and so on
down to float16, which is never tested (it is the reference) and is the floor. Testing in a
fixed order with one bound per step needs no splitting of the error rate.

These checks can only make the layout safer. Practice questions are easier than real
questions, so a pass is never reported as confirmation of the target on real searches.

The strength line comes from ``presets/veto_strength.json`` (``scripts/make_veto_strength.py``,
a toy damage-model simulation) and is tagged ``[estimate: simulation]``; it does not claim
that real losses show up in practice questions (that is test V5).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np

#: one-sided level of the bootstrap bound
LEVEL = 0.95
#: bootstrap resamples (the design simulations used 1,000; the build uses 2,000)
B_DEFAULT = 2000
#: rank depth of the statistic: nDCG@5
K = 5
#: fewest hard practice questions for a folder to count as checked (FINAL_DESIGN 5.3.1)
MIN_PROBES = 60
#: fewest real questions for the real-query gate (FINAL_DESIGN 5.5.2)
MIN_QUERIES = 60
#: kinds of evidence a check can carry, and the ``_claim`` source of the user-query kinds
AGREEMENT = "agreement with full precision, not relevance"
SEARCH_QUALITY = "search quality"
CHECK_KINDS = ("probe", "agreement", "search quality")
_RESAMPLE_ELEMENTS = 2_000_000  # bootstrap index block size, bounded so memory stays small


# ---------------------------------------------------------------------------- nDCG


def _first(item: Any) -> str:
    return str(item[0]) if isinstance(item, (tuple, list)) else str(item)


def canonical_ranking(ranking: Iterable[Any], aliases: Mapping[str, str] | None = None) -> list[str]:
    """Page ids of a ranking (ids or ``(id, score)`` pairs), each mapped to its first copy when
    ``aliases`` is given, keeping the best-ranked occurrence of each."""
    out: list[str] = []
    seen: set[str] = set()
    for item in ranking:
        pid = _first(item)
        if aliases:
            pid = aliases.get(pid, pid)
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def ndcg_at(ranking_ids: Iterable[Any], relevant: Iterable[str], k: int = K) -> float:
    """Binary-gain nDCG@k: ``sum(1/log2(rank + 1))`` over relevant pages in the top ``k``,
    divided by the same for an ideal ranking of ``min(k, len(relevant))`` relevant pages.
    ``ranking_ids`` may hold ids or ``(id, score)`` pairs; a repeated id counts once. 0 when
    there is nothing relevant."""
    rel = set(relevant)
    if not rel or k < 1:
        return 0.0
    dcg = 0.0
    seen: set[str] = set()
    rank = 0
    for item in ranking_ids:
        pid = _first(item)
        if pid in seen:
            continue
        seen.add(pid)
        if pid in rel:
            dcg += 1.0 / math.log2(rank + 2)
        rank += 1
        if rank >= k:
            break
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / ideal


# ---------------------------------------------------------------------------- bootstrap


def bootstrap_upper(c_sums: np.ndarray, b_sums: np.ndarray, *, B: int = B_DEFAULT, seed: int = 0,
                    level: float = LEVEL) -> float:
    """One-sided ``level`` upper bound of ``sum(c) / sum(b)`` by resampling clusters with
    replacement: ``B`` resamples of all ``m`` clusters drawn from ``default_rng(seed)``,
    percentile method (``numpy.quantile``, linear). A resample whose ``b`` sum is 0 counts as
    ratio 1."""
    c = np.asarray(c_sums, dtype=np.float64)
    b = np.asarray(b_sums, dtype=np.float64)
    m = int(c.size)
    if m == 0 or B < 1:
        return math.nan
    rng = np.random.default_rng(seed)
    rows = max(1, _RESAMPLE_ELEMENTS // m)
    ratios = np.empty(B, dtype=np.float64)
    for lo in range(0, B, rows):
        hi = min(B, lo + rows)
        idx = rng.integers(0, m, size=(hi - lo, m))
        num = c[idx].sum(axis=1)
        den = b[idx].sum(axis=1)
        ratios[lo:hi] = np.divide(num, den, out=np.ones_like(num), where=den > 0)
    return float(np.quantile(ratios, level))


def _pct_text(x: float) -> str:
    text = f"{100.0 * float(x):.1f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _plain(row: Any) -> str:
    return str(getattr(row, "plain", None) or getattr(row, "name", None) or row)


def _evidence(kind: str) -> tuple[str, str]:
    """``(_claim kind, source)`` of a check kind."""
    if kind == "probe":
        return "probe", ""
    if kind == "agreement":
        return "user-queries", AGREEMENT
    if kind == "search quality":
        return "user-queries", SEARCH_QUALITY
    raise ValueError(f"unknown check kind {kind!r}; known: {CHECK_KINDS}")


@dataclass(frozen=True)
class CheckStat:
    """One practice check (or real-query gate) of one layout.

    ``R`` is the nDCG@5 retention ``sum(c) / sum(b)``, ``U`` its one-sided 95% bootstrap upper
    bound, ``n`` the number of questions and ``n_clusters`` the clusters resampled; ``passed``
    is ``U >= target``. ``kind`` is ``"probe"`` (practice questions), ``"agreement"`` (real
    questions labelled by the full-precision top page) or ``"search quality"`` (real
    questions with answers). ``informative`` is False when the full-precision nDCG was 0 on
    every question, which leaves nothing to compare (the check then passes).
    """

    R: float
    U: float
    n: int
    passed: bool
    target: float = 0.0
    n_clusters: int = 0
    kind: str = "probe"
    sum_c: float = 0.0
    sum_b: float = 0.0
    B: int = B_DEFAULT
    seed: int = 0
    k: int = K
    informative: bool = True

    def claim(self, value: Any, text: str = "", construct: str = "") -> Any:
        from ._claim import claim

        kind, source = _evidence(self.kind)
        return claim(value, kind, source, construct, text)

    def retention_line(self, label: str = "Questions") -> Any:
        """``R`` as a ``_claim.Line`` (for ``report()``; never a confirmation of the target)."""
        from ._claim import Line, retention

        what = {"probe": "on {0} hard questions made from your PDFs, retention against full precision was {1}",
                "agreement": "on {0} of your questions, retention against the full-precision top page was {1}",
                "search quality": "on {0} of your questions, search quality kept {1} of full precision"}[self.kind]
        return Line(label, what, self.claim(self.n, construct="questions"),
                    self.claim(self.R, text=retention(self.R) if self.informative else "not measurable",
                               construct="nDCG@5 retention"))

    def as_dict(self) -> dict[str, Any]:
        return {"R": self.R, "U": self.U, "n": self.n, "passed": self.passed, "target": self.target,
                "n_clusters": self.n_clusters, "kind": self.kind, "sum_c": self.sum_c, "sum_b": self.sum_b,
                "B": self.B, "seed": self.seed, "k": self.k, "level": LEVEL, "informative": self.informative}


def practice_check(cand: Mapping[str, Sequence[Any]], ref: Mapping[str, Sequence[Any]],
                   relevant: Mapping[str, Iterable[str]], clusters: Mapping[str, str] | None, target: float, *,
                   B: int = B_DEFAULT, seed: int = 0, k: int = K, aliases: Mapping[str, str] | None = None,
                   kind: str = "probe") -> CheckStat:
    """The practice check of one candidate layout (FINAL_DESIGN 5.5.1).

    ``ref`` holds the full-precision ranking of every question checked, ``cand`` the
    candidate's ranking of the same questions, ``relevant`` each question's right pages and
    ``clusters`` each question's cluster (its source page; a question without one is its own
    cluster). With ``aliases``, repeated pages are mapped to their first copy in rankings and
    relevant sets alike. Raises ``KeyError`` for a question without a candidate ranking or a
    relevant set.
    """
    _evidence(kind)
    qids = sorted(ref)
    groups: dict[str, list[float]] = {}
    total_c = total_b = 0.0
    for q in qids:
        if q not in cand:
            raise KeyError(f"no candidate ranking for question {q!r}")
        if q not in relevant:
            raise KeyError(f"no relevant set for question {q!r}")
        rel = set(relevant[q])
        if aliases:
            rel = {aliases.get(r, r) for r in rel}
        c = ndcg_at(canonical_ranking(cand[q], aliases), rel, k)
        b = ndcg_at(canonical_ranking(ref[q], aliases), rel, k)
        label = str(clusters.get(q, q)) if clusters else q
        acc = groups.setdefault(label, [0.0, 0.0])
        acc[0] += c
        acc[1] += b
        total_c += c
        total_b += b
    labels = sorted(groups)
    cs = np.array([groups[g][0] for g in labels], dtype=np.float64)
    bs = np.array([groups[g][1] for g in labels], dtype=np.float64)
    if total_b <= 0:
        return CheckStat(math.nan, math.nan, len(qids), True, float(target), len(labels), kind,
                         total_c, total_b, B, seed, k, informative=False)
    R = total_c / total_b
    U = bootstrap_upper(cs, bs, B=B, seed=seed)
    return CheckStat(R, U, len(qids), bool(U >= target), float(target), len(labels), kind, total_c, total_b,
                     B, seed, k)


# ---------------------------------------------------------------------------- fixed sequence


def is_float_row(row: Any) -> bool:
    """A full-precision row (float16 or float32): the floor of the sequence, never tested."""
    if getattr(row, "chain", None) == "float":
        return True
    return str(getattr(row, "name", "")) in ("float16", "float32")


@dataclass(frozen=True)
class SequenceResult:
    """The outcome of the fixed sequence: the layout ``chosen`` and every ``(row, CheckStat)``
    tested, in order. ``exhausted`` is True when every row failed and the sequence had no
    float row to fall back on (``chosen`` is then its last row)."""

    chosen: Any
    steps: list[tuple[Any, CheckStat]] = field(default_factory=list)
    target: float = 0.0
    kind: str = "probe"
    exhausted: bool = False

    @property
    def failed(self) -> list[Any]:
        return [row for row, stat in self.steps if not stat.passed]

    @property
    def stepped_down(self) -> bool:
        return bool(self.failed)

    @property
    def floor(self) -> bool:
        return is_float_row(self.chosen)

    @property
    def n(self) -> int:
        return self.steps[0][1].n if self.steps else 0

    def lines(self) -> list[Any]:
        """The summary lines (FINAL_DESIGN 4.2): a pass, a step-down, or the float16 fallback."""
        from ._claim import Line, claim

        if not self.steps:
            return []
        stat = self.steps[-1][1]
        ckind, source = _evidence(self.kind)
        n = claim(stat.n, ckind, source, "questions")
        target = claim(self.target, ckind, source, "target", _pct_text(self.target))
        where = "hard questions from your PDFs" if self.kind == "probe" else "of your questions"
        label = "Practice" if self.kind == "probe" else "Questions"
        if not self.stepped_down:
            if self.kind == "probe":
                return [Line(label, "passed: {0} hard questions made from your PDFs' text", n)]
            return [Line(label, "passed: {0} of your questions", n)]
        failed = ", ".join(_plain(r) for r in self.failed)
        if self.exhausted:
            return [Line(label, "every layout offered ({0}) fell below {1} on {2} " + where
                         + "; none is safer, so {3} is kept", failed, target, n, _plain(self.chosen)),
                    Line("", "the model or the questions may be at fault; see report() and ov.doctor()")]
        if self.floor:
            return [Line(label, "every compressed layout tested ({0}) fell below {1} on {2} " + where
                         + "; we stored {3}", failed, target, n, _plain(self.chosen)),
                    Line("", "the model or the questions may be at fault; see report() and ov.doctor()")]
        return [Line(label, "{0} fell below {1} on {2} " + where + "; we used the safer {3}",
                     failed, target, n, _plain(self.chosen))]

    def as_dict(self) -> dict[str, Any]:
        return {"chosen": _plain(self.chosen), "target": self.target, "kind": self.kind,
                "exhausted": self.exhausted,
                "steps": [{"layout": _plain(r), **s.as_dict()} for r, s in self.steps]}


def fixed_sequence(sequence: Sequence[Any], rank_fn: Callable[[Any], Mapping[str, Sequence[Any]]],
                   ref: Mapping[str, Sequence[Any]], relevant: Mapping[str, Iterable[str]],
                   clusters: Mapping[str, str] | None, target: float, *, B: int = B_DEFAULT, seed: int = 0,
                   k: int = K, aliases: Mapping[str, str] | None = None, kind: str = "probe") -> SequenceResult:
    """Test ``sequence`` (``Selection.sequence``: the pick, then each safer layout, ending at
    float16) in order and keep the first layout that passes (FINAL_DESIGN 5.5.1).

    ``rank_fn(row)`` returns that layout's rankings of the questions in ``ref``; it is called
    only for the rows actually tested. A float row is taken without a test. Each step uses
    the same ``seed``.
    """
    if not sequence:
        raise ValueError("the sequence of layouts is empty")
    steps: list[tuple[Any, CheckStat]] = []
    for row in sequence:
        if is_float_row(row):
            return SequenceResult(row, steps, float(target), kind)
        stat = practice_check(rank_fn(row), ref, relevant, clusters, target, B=B, seed=seed, k=k,
                              aliases=aliases, kind=kind)
        steps.append((row, stat))
        if stat.passed:
            return SequenceResult(row, steps, float(target), kind)
    return SequenceResult(sequence[-1], steps, float(target), kind, exhausted=True)


# ---------------------------------------------------------------------------- real questions


def _answer_set(entry: Any) -> frozenset[str]:
    if isinstance(entry, str):
        return frozenset([entry])
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], int):
        return frozenset([f"{entry[0]}::p{entry[1]}"])
    out: set[str] = set()
    for item in entry:
        out |= _answer_set(item)
    return frozenset(out)


def query_labels(ref: Mapping[str, Sequence[Any]], answers: Mapping[str, Any] | None = None, *,
                 aliases: Mapping[str, str] | None = None
                 ) -> tuple[dict[str, frozenset[str]], dict[str, str], str]:
    """``(relevant, clusters, kind)`` for the real-query gate (FINAL_DESIGN 5.5.2).

    Without ``answers`` each question's full-precision top page is its label (``"agreement"``:
    agreement with full precision, not relevance). With ``answers`` (question id -> a page id,
    a ``(doc_id, page)`` pair, or a set of either) the answers are the labels (``"search
    quality"``); questions without an answer are left out. Each question is its own cluster.
    """
    relevant: dict[str, frozenset[str]] = {}
    if answers is None:
        for q, ranking in ref.items():
            top = canonical_ranking(ranking, aliases)
            if top:
                relevant[q] = frozenset([top[0]])
        kind = "agreement"
    else:
        if not isinstance(answers, Mapping):
            raise TypeError("answers must map each question id to its answer pages "
                            f"(got {type(answers).__name__}); pair a list with the question ids first")
        for q in ref:
            if q in answers and answers[q] is not None:
                labels = _answer_set(answers[q])
                if labels:
                    relevant[q] = labels
        kind = "search quality"
    return relevant, {q: q for q in relevant}, kind


def real_query_check(cand: Mapping[str, Sequence[Any]], ref: Mapping[str, Sequence[Any]], target: float, *,
                     answers: Mapping[str, Any] | None = None, B: int = B_DEFAULT, seed: int = 0, k: int = K,
                     aliases: Mapping[str, str] | None = None) -> CheckStat:
    """The real-query gate statistic of one layout: the statistic and bound of the practice
    check on the user's questions (FINAL_DESIGN 5.5.2). Needs at least 60 labelled questions;
    raises ``ValueError`` below that. Use :func:`query_labels` with :func:`fixed_sequence` to
    run it over a sequence of layouts."""
    relevant, clusters, kind = query_labels(ref, answers, aliases=aliases)
    if len(relevant) < MIN_QUERIES:
        raise ValueError(f"the real-query gate needs at least {MIN_QUERIES} questions with a label; "
                         f"got {len(relevant)}")
    ref_l = {q: ref[q] for q in relevant}
    return practice_check(cand, ref_l, relevant, clusters, target, B=B, seed=seed, k=k, aliases=aliases,
                          kind=kind)


# ---------------------------------------------------------------------------- strength


@lru_cache(maxsize=1)
def strength_table() -> dict[str, Any]:
    """``presets/veto_strength.json`` (generated by ``scripts/make_veto_strength.py``)."""
    text = resources.files("optivision").joinpath("presets", "veto_strength.json").read_text(encoding="utf-8")
    return json.loads(text)


def strength(n: int, target: float) -> dict[str, Any] | None:
    """The simulated strength for ``n`` hard questions at ``target``: the summary entry of the
    largest simulated target at or below ``target`` (the smallest one when ``target`` is below
    them all) and the largest simulated count at or below ``n``. None below the smallest
    simulated count.

    Both choices understate: a check at a higher target, or with more questions, fires at
    least as often on the same fall, so the rate shown is never more than the user's check
    would reach under the same damage model."""
    summary = strength_table()["summary"]
    targets = sorted({float(e["target"]) for e in summary})
    below = [x for x in targets if x <= float(target) + 1e-9]
    t = below[-1] if below else targets[0]
    rows = [e for e in summary if float(e["target"]) == t and int(e["n"]) <= int(n)]
    if not rows:
        return None
    return max(rows, key=lambda e: int(e["n"]))


def _range_text(lo: float, hi: float) -> str:
    a, b = math.floor(100 * float(lo) + 1e-9), math.floor(100 * float(hi) + 1e-9)
    return f"{a}%" if a == b else f"{a}-{b}%"


def strength_line(n: int, target: float) -> Any:
    """The ``Strength`` line of the build summary, tagged ``[estimate: simulation]``, or None
    when there are too few questions (FINAL_DESIGN 4.2)."""
    entry = strength(n, target)
    if entry is None:
        return None
    from ._claim import Line, claim

    fall = min(entry["falls"], key=lambda f: float(f["retention"]))
    lo, hi = fall["caught"]
    src = "simulation"
    return Line("Strength", "at {0} questions a simulated fall to {1} of full-precision quality is caught in {2} "
                "of runs; whether real losses show up in practice questions is untested (V5)",
                claim(int(entry["n"]), "estimate", src, "questions simulated"),
                claim(float(fall["retention"]), "estimate", src, "simulated retention",
                      _pct_text(float(fall["retention"]))),
                claim((float(lo), float(hi)), "estimate", src, "share of simulated runs caught",
                      _range_text(lo, hi)))


__all__ = [
    "AGREEMENT",
    "B_DEFAULT",
    "CHECK_KINDS",
    "LEVEL",
    "MIN_PROBES",
    "MIN_QUERIES",
    "SEARCH_QUALITY",
    "CheckStat",
    "K",
    "SequenceResult",
    "bootstrap_upper",
    "canonical_ranking",
    "fixed_sequence",
    "is_float_row",
    "ndcg_at",
    "practice_check",
    "query_labels",
    "real_query_check",
    "strength",
    "strength_line",
    "strength_table",
]
