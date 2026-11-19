"""Apply the decision rules of FINAL_DESIGN section 8 to scored predictions.

The rules were fixed before any result was seen (PREREG_EASY part 1). This module only
*proposes*: it returns the actions and a ledger change (``presets/easy_ledger.json``:
settings to set and outcome entries to append) that a person reviews and applies; it never
writes the ledger. Any margin a rule changes is labelled "adjusted on our validation sets
(in-sample)" until a fresh, separately preregistered unit set confirms it. A rule whose
prediction was not scored (``passed`` is None) is not applied.
"""

from __future__ import annotations

import time
from typing import Any

from easy_validation import units

IN_SAMPLE = "adjusted on our validation sets (in-sample)"
CHECKED = "checked on new documents (V1)"
NOT_YET = "not yet checked on new documents (check registered)"


def _passed(results: dict, key: str) -> bool | None:
    r = results.get(key)
    return None if r is None else r.get("passed")


def decide(results: dict[str, dict], rerun: Any = None) -> dict:
    """Actions and the proposed ledger change. ``rerun(arm, target)`` re-scores V1 at one target
    under an adjusted arm (``"A+0.01"``, ``"A+0.01/codecs"``) for the V1b/V1c rule; without it
    that follow-up is left to the person applying the change."""
    actions: list[dict] = []
    settings: dict[str, Any] = {}
    margin_changed = False

    def act(rule_text: str, outcome: str, set_: dict | None = None, note: str = "") -> None:
        nonlocal margin_changed
        actions.append({"rule": rule_text, "outcome": outcome, "set": set_ or {}, "note": note})
        for k, v in (set_ or {}).items():
            settings[k] = v
            if k.startswith("features.margins") or k.startswith("features.unmeasured_rules"):
                margin_changed = True

    for key, target in (("V1b", 0.97), ("V1c", 0.95)):
        if _passed(results, key) is False:
            follow = rerun("A+0.01", target) if rerun else None
            if follow is None or follow:
                act(f"{key} fails", "+0.01 on compressed rungs at that target",
                    {f"features.margins.extra_compressed_at.{target}": 0.01},
                    "V1 rescored with +0.01: " + ("passes" if follow else "not rescored here"))
            else:
                act(f"{key} fails", "+0.01 did not repair V1: cap at codec chains at that target",
                    {f"features.margins.extra_compressed_at.{target}": 0.01,
                     f"features.margins.codec_chains_only_at.{target}": True})
    v1d = results.get("V1d") or {}
    if v1d.get("passed") is False and (v1d.get("per_target") or {}).get("0.9") is False:
        act("V1d fails at 0.90", "lowest target becomes 0.95", {"features.margins.lowest_target": 0.95})
    if _passed(results, "V1-unchecked") is False:
        act("V1-unchecked fails", "+0.01 merge penalty for unchecked folders",
            {"features.margins.unchecked_merge_penalty": 0.01})
    arms = (results.get("V1-arms") or {}).get("arms") or {}
    comparable = (results.get("V1-arms") or {}).get("comparable", False)
    if comparable and arms:
        b = arms.get("B") or {}
        if (b.get("passes") or {}).get("V1b") and (b.get("passes") or {}).get("V1c") and b.get("severe_one_bit") == 0:
            act("Arm B passes V1b, V1c, no severe 1-bit miss", "remove the 1-bit margin",
                {"features.margins.one_bit_margin": 0.0})
        c = arms.get("C") or {}
        pc = c.get("passes") or {}
        if pc.get("V1b") and pc.get("V1c") and pc.get("V1d") and c.get("V2"):
            act("Arm C passes V1, V2", "halve the size coefficients", {"features.margins.size_coefficient_scale": 0.5})
        d, a = arms.get("D") or {}, arms.get("A") or {}
        pd, pa = d.get("passes") or {}, a.get("passes") or {}
        if pd.get("V1b") and pd.get("V1c") and (pa.get("V1b") is False or pa.get("V1c") is False):
            act("Arm D passes, A fails", "flat +0.01", {"features.margins.flat": 0.01})
    arm_s = results.get("arm S") or {}
    if arm_s.get("passed") is True:
        coef = arm_s.get("coefficients") or {}
        if coef != units.SIZE_COEFFICIENTS:
            act("Arm S (V2 kept out passes)", "size coefficients refit", {"features.margins.size_coefficients": coef})
    below = ((results.get("S") or {}).get("below_500") or {}).get("coefficient_per_unit_s") or {}
    if arm_s.get("passed") is True and below and all(v for v in below.values()):
        act("Arm S, measured below 500 pages", "negative s(N) side", {"features.margins.below_500": below})
    v1u = results.get("V1u") or {}
    if v1u.get("passed") is not None:
        if (v1u.get("int8_low_min") or 0) >= 0.98:
            act("V1u int8 low >= 0.98 on all held-out page units", "unmeasured pages: int8 at 0.98",
                {"features.unmeasured_rules.int8_at": 0.98})
        if (v1u.get("int4_mean_low_min") or 0) >= 0.95:
            act("V1u int4(mean) low >= 0.95 on all held-out page units", "unmeasured pages: int4(mean) at 0.95",
                {"features.unmeasured_rules.int4_mean_at": 0.95})
        if v1u.get("passed") is False:
            act("V1u picks miss", "unmeasured: float16 above 0.95", {"features.unmeasured_rules.float16_above": 0.95})
    if _passed(results, "V2") is False:
        act("V2 fails", "codecs only above 2,500 pages", {"features.margins.codecs_only_above": 2500})
    if _passed(results, "V3") is False:
        act("V3 fails", "re-encode advised past 1.5x growth", {"features.check.reencode_advised_growth": 1.5})
    if _passed(results, "V4") is False:
        act("V4 fails", 'practice questions described as "not reliably easier or harder than real questions"',
            {"features.practice_check.wording": "not reliably easier or harder than real questions"})
    if _passed(results, "V5a") is False:
        act("V5a fails", "practice questions only as examples",
            {"features.practice_check.on": False, "features.practice_check.mode": "examples only"})
    elif _passed(results, "V5b") is False or _passed(results, "V5c") is False:
        act("V5b or V5c fails", "the practice check becomes a warning", {"features.practice_check.mode": "warning"})
    scored = {k: r.get("passed") for k, r in results.items()}
    v1_ok = all(scored.get(k) is True for k in ("V1b", "V1c", "V1d"))
    status = IN_SAMPLE if margin_changed else CHECKED if v1_ok else NOT_YET
    outcome = {"recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "source": "reports/easy/results/predictions.json",
               "predictions": {k: v for k, v in scored.items()}, "actions": [a["rule"] for a in actions],
               "status": status}
    return {"actions": actions, "settings": settings, "status": status, "margin_changed": margin_changed,
            "ledger_change": {"set": settings, "append_outcome": outcome,
                              "note": "proposed only; review, then apply to src/optivision/presets/easy_ledger.json"}}
