"""One unit through the rule and the practice check, scored on its real labels.

:func:`run_rule` is the path a build takes (FINAL_DESIGN 5.5): the rule picks from the
catalogue (``select``, or an arm of it), and on a checked folder the practice check tests the
pick and steps down the fixed sequence (``_veto.fixed_sequence``). The real labels then score
both the rule's pick and the layout the check kept. A folder is checked when its practice
questions (from the PDF text layer by default) leave at least 60 hard ones and its profile is
checkable (FD 5.3.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from easy_validation import api, rule, units
from easy_validation.data import Unit, load_unit
from easy_validation.scorer import TOP, UnitScorer


class Practice:
    """The hard practice questions of one unit and one text source, with their float16 rankings."""

    def __init__(self, scorer: UnitScorer, source: str) -> None:
        unit = scorer.unit
        self.scorer, self.source = scorer, source
        self.by_id = {p["id"]: p for p in unit.probes[source]}
        ids = list(scorer.query_set(source).ids)
        base = scorer.scores(None, source)
        ref_scores = scorer.ranking(base, ids, k=TOP, with_scores=True)
        b = api.backend()
        hs = b.hard_set([b.probe(self.by_id[i]) for i in ids], ref_scores, k_keep=10)
        self.hard = [i for i in api.hard_ids(hs) if i in self.by_id]
        self.digest = api.hard_digest(hs)
        self.relevant = {i: frozenset(self.by_id[i]["relevant"]) for i in self.hard}
        self.clusters = {i: str(self.by_id[i].get("page_id")) for i in self.hard}
        self.ref = {i: [d for d, _ in ref_scores[i][:10]] for i in self.hard}
        self.n_probes = len(ids)

    @property
    def checked(self) -> bool:
        return len(self.hard) >= units.HARD_MIN and self.scorer.unit.text_checkable(self.source)

    def rank_fn(self, row: Any) -> dict[str, list[str]]:
        spec = api.rung(row).spec
        s = self.scorer.scores(spec, self.source, qids=self.hard)
        return self.scorer.ranking(s, self.hard, k=10)

    def fault_ranking(self, fault: str) -> dict[str, list[str]]:
        spec = units.FAULTS[fault]
        if spec is None:
            s = self.scorer.random_scores(self.source, self.hard)
        else:
            s = self.scorer.scores(spec, self.source, qids=self.hard)
        return self.scorer.ranking(s, self.hard, k=10)

    def check(self, ranking: dict[str, list[str]], target: float) -> dict:
        return api.stat(api.backend().practice_check(ranking, self.ref, self.relevant, self.clusters, target))


@dataclass
class Context:
    """Loaded units, scorers and practice sets, shared by every prediction of one run."""

    roots: list[Path]
    heldout: list[str] = field(default_factory=list)
    _units: dict = field(default_factory=dict)
    _scorers: dict = field(default_factory=dict)
    _practice: dict = field(default_factory=dict)
    missing: set = field(default_factory=set)

    def unit(self, model: str, corpus: str) -> Unit | None:
        key = (model, corpus)
        if key not in self._units:
            u = load_unit(self.roots, model, corpus, heldout=model in self.heldout)
            if u is None:
                self.missing.add(f"{model} on {corpus}")
            self._units[key] = u
        return self._units[key]

    def scorer(self, unit: Unit) -> UnitScorer:
        if unit.tag not in self._scorers:
            self._scorers[unit.tag] = UnitScorer(unit)
        return self._scorers[unit.tag]

    def practice(self, unit: Unit, source: str = "text_layer") -> Practice | None:
        key = (unit.tag, source)
        if key not in self._practice:
            self._practice[key] = Practice(self.scorer(unit), source) if source in unit.probes else None
        return self._practice[key]


def selection(unit: Unit, target: float, *, checked: bool) -> Any:
    return api.select(None if unit.heldout else unit.model, target, unit.n_pages, checked=checked,
                      modality=unit.modality, vectors_per_page=unit.vectors_per_page if unit.modality == "page" else None)


def run_rule(ctx: Context, unit: Unit, target: float, *, arm: str = "A", hide_text: bool = False,
             practice: bool = True, source: str = "text_layer", coefficients: dict[str, float] | None = None) -> dict:
    """The rule's pick and, on a checked folder, the practice check's layout, both scored."""
    sc = ctx.scorer(unit)
    pr = None if hide_text or unit.modality != "page" else ctx.practice(unit, source)
    checked = bool(pr is not None and pr.checked)
    sel = selection(unit, target, checked=checked)
    a_pick, a_seq = rule.arm_choice(sel, target, unit.n_pages, "A", unit.modality)
    frozen_pick = api.pick(sel)
    agrees = a_pick.key == frozen_pick.key
    if arm == "A":
        pick, seq = frozen_pick, api.sequence(sel)
    else:
        pick, seq = rule.arm_choice(sel, target, unit.n_pages, arm, unit.modality, coefficients)
    real_pick = sc.retention(pick.spec)
    out: dict[str, Any] = {"model": unit.model, "corpus": unit.corpus, "target": target, "arm": arm,
                           "n_pages": unit.n_pages, "n_queries": unit.n_queries, "checked": checked,
                           "text_hidden": hide_text, "probe_source": source if checked else None,
                           "n_hard": len(pr.hard) if pr else 0, "rule_agrees_with_select": agrees,
                           "basis": getattr(sel, "basis", None), "close_call": bool(getattr(sel, "close_call", False)),
                           "pick": pick.name, "pick_one_bit": pick.one_bit, "pick_point": real_pick["point"],
                           "pick_low": real_pick["low"], "pick_ratio_f16": real_pick.get("ratio_f16", 1.0)}
    chosen, real = pick, real_pick
    if checked and practice:
        result = api.backend().fixed_sequence(seq, pr.rank_fn, pr.ref, pr.relevant, pr.clusters, target)
        chosen = api.rung(api.chosen(result))
        real = sc.retention(chosen.spec)
        out["steps"] = api.steps(result)
    out.update(chosen=chosen.name, chosen_one_bit=chosen.one_bit, point=real["point"], low=real["low"],
               ratio_f16=real.get("ratio_f16", 1.0), baseline=real["baseline"])
    return out
