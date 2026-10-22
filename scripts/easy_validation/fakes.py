"""Stand-ins for the easy-mode rule, probe and practice-check functions, for tests and dry runs.

``FakeBackend`` follows the contract signatures (docs/easy/CONTRACT.md) with small, readable
implementations: a fixed ladder with made-up lowest-kept values and the frozen rule's
margins (``rule.need``), the FD 5.3.1 hard-set rule, and the FD 5.5.1 statistic with a page
cluster bootstrap. It lets the analyses run end to end on a synthetic unit before the real
modules exist; nothing it returns is evidence about the package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from easy_validation import units

MERGE = {"max_distance": None, "min_vectors": 1, "stage": "hierarchical_merge"}


def _hm(r: float, q: dict) -> dict:
    return {"stages": [{**MERGE, "ratio": r}, q]}


INT8 = {"scale": "per_vector", "stage": "int8"}
INT4M = {"center": "mean", "stage": "int4"}
BIN = {"stage": "binary"}

#: (name, chain, spec, one_bit, merged, ratio vs float32, made-up lowest kept)
PAGE_LADDER = [
    ("float32", "float", {"stages": []}, False, False, 1.0, 1.0),
    ("float16", "float", {"stages": [{"stage": "float16"}]}, False, False, 2.0, 0.999),
    ("int8", "8-bit", {"stages": [INT8]}, False, False, 3.9, 0.985),
    ("hm(0.33) > int8", "8-bit", _hm(0.33, INT8), False, True, 11.9, 0.975),
    ("hm(0.25) > int8", "8-bit", _hm(0.25, INT8), False, True, 15.7, 0.972),
    ("int4(mean)", "centred 4-bit", {"stages": [INT4M]}, False, False, 7.8, 0.968),
    ("hm(0.33) > int4(mean)", "centred 4-bit", _hm(0.33, INT4M), False, True, 23.4, 0.955),
    ("hm(0.25) > int4(mean)", "centred 4-bit", _hm(0.25, INT4M), False, True, 30.9, 0.948),
    ("binary", "1-bit", {"stages": [BIN]}, True, False, 32.0, 0.95),
    ("hm(0.33) > binary", "1-bit", _hm(0.33, BIN), True, True, 96.0, 0.93),
    ("hm(0.25) > binary", "1-bit", _hm(0.25, BIN), True, True, 126.0, 0.91),
]
TEXT_LADDER = [r for r in PAGE_LADDER if r[0] in ("float32", "float16", "int8", "int4(mean)")]


@dataclass(frozen=True)
class FakeProbe:
    id: str
    text: str
    family: str | None
    page_id: str | None
    relevant: frozenset


@dataclass(frozen=True)
class FakeStat:
    R: float
    U: float
    n: int
    passed: bool


def ndcg(ranking: list[str], relevant: frozenset, k: int = 5) -> float:
    gains = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return gains / ideal if ideal else 0.0


class FakeBackend:
    name = "fakes"

    def __init__(self, lowest_kept: dict[str, float] | None = None, hard_min: int = units.HARD_MIN,
                 boot: int = 400) -> None:
        self.lowest_kept = lowest_kept or {}
        self.hard_min = hard_min
        self.boot = boot

    # ------------------------------------------------------------ rule

    def ladder(self, modality: str) -> list[Any]:
        return self._rows(TEXT_LADDER if modality == "text" else PAGE_LADDER)

    def _rows(self, ladder: list) -> list[Any]:
        from optivision.easy._types import LayoutRow

        return [LayoutRow(name=n, plain=n, spec=s, chain=c, bytes_per_page=None, ratio_f16=r / 2, ratio_f32=r,
                          lowest_kept=self.lowest_kept.get(n, low), worst_point=self.lowest_kept.get(n, low),
                          one_bit=ob, merged=m) for n, c, s, ob, m, r, low in ladder]

    def select(self, model: str | None, target: float, n_pages: int | None = None, *, checked: bool = False,
               modality: str | None = None, vectors_per_page: float | None = None, layout: str | None = None) -> Any:
        from optivision.easy._types import Selection

        from easy_validation import api, rule

        text = modality == "text" or (model is not None and any(x in model for x in ("colbert", "LateOn", "ColBERT")))
        mod = "text" if text else "page"
        rows = self._rows(TEXT_LADDER if text else PAGE_LADDER)
        n = n_pages or 500
        out, ok_chain = [], {}
        for r in rows:
            v = api.rung(r)
            if v.is_float:
                out.append(replace(r, status="qualifies" if (r.lowest_kept or 0) >= target else "below target"))
                continue
            if v.one_bit and not checked:
                out.append(replace(r, status="needs practice questions"))
                continue
            if model is None and v.merged:
                out.append(replace(r, status="not allowed", reason="unmeasured: codecs only"))
                continue
            need = rule.need(v, target, n, "A", mod)
            ok = ok_chain.get(v.chain, True) and r.lowest_kept >= need
            status = "qualifies" if ok else ("family stopped" if not ok_chain.get(v.chain, True) else "below target")
            ok_chain[v.chain] = ok
            out.append(replace(r, status=status, need=need, slack=r.lowest_kept - need))
        sel = SimpleNamespace(rows=out)
        pick_rung, seq = rule.arm_choice(sel, target, n, "A", mod)
        pick = pick_rung.row
        out = [replace(r, status="pick") if r is pick else r for r in out]
        pick = next(r for r in out if r.name == pick.name)
        seq = [next(r for r in out if r.name == s.name) for s in seq]
        return Selection(model=model, modality=mod, target=target, n_pages=n_pages, checked=checked, rows=tuple(out),
                         pick=pick, sequence=tuple(seq), basis="own rows" if model else "unmeasured: pooled (fake)")

    # ------------------------------------------------------------ probes

    def probe(self, d: dict) -> FakeProbe:
        return FakeProbe(d["id"], d["text"], d.get("family"), d.get("page_id"), frozenset(d.get("relevant") or ()))

    def hard_set(self, probes: list, ref_rankings: dict, k_keep: int = 10) -> Any:
        kept: dict[str, tuple[float, str]] = {}
        for p in probes:
            ranking = ref_rankings[p.id]
            top = [d for d, _ in ranking[:k_keep]]
            if not set(top) & p.relevant:
                continue
            s_rel = max(s for d, s in ranking if d in p.relevant)
            s_out = max((s for d, s in ranking if d not in p.relevant), default=0.0)
            g = (s_rel - s_out) / s_rel if s_rel else 0.0
            if p.page_id not in kept or g < kept[p.page_id][0]:
                kept[p.page_id] = (g, p.id)
        order = sorted(kept.values())
        n = min(300, max(len(order) // 3, self.hard_min), len(order))
        return SimpleNamespace(ids=[pid for _, pid in order[:n]], sha256="fake")

    def ndcg_at(self, ranking_ids: list[str], relevant: Any, k: int = 5) -> float:
        return ndcg(list(ranking_ids), frozenset(relevant), k)

    def practice_check(self, cand: dict, ref: dict, relevant: dict, clusters: dict, target: float) -> FakeStat:
        ids = sorted(cand)
        c = np.array([ndcg(cand[i], relevant[i]) for i in ids])
        b = np.array([ndcg(ref[i], relevant[i]) for i in ids])
        R = float(c.sum() / b.sum()) if b.sum() else 0.0
        names = sorted({clusters.get(i, i) for i in ids})
        index = {n: k for k, n in enumerate(names)}
        k = np.array([index[clusters.get(i, i)] for i in ids])
        cs, bs = np.bincount(k, c, len(names)), np.bincount(k, b, len(names))
        counts = np.random.default_rng(units.PRACTICE_SEED).multinomial(len(names), [1 / len(names)] * len(names),
                                                                         size=self.boot)
        den = counts @ bs
        ratios = np.where(den > 0, (counts @ cs) / np.where(den > 0, den, 1), 0.0)
        U = float(np.quantile(ratios, 0.95)) if ratios.size else R
        return FakeStat(R, U, len(ids), U >= target)

    def fixed_sequence(self, sequence: list, rank_fn: Any, ref: dict, relevant: dict, clusters: dict,
                       target: float) -> Any:
        steps = []
        for row in sequence:
            st = self.practice_check(rank_fn(row), ref, relevant, clusters, target)
            steps.append((row, st))
            if st.passed:
                return SimpleNamespace(chosen=row, steps=steps)
        return SimpleNamespace(chosen=sequence[-1], steps=steps)
