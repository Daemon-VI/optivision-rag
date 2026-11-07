"""Score PREREG_EASY part 3: the DT campaign (FD 5.8) and practice-question allowance (FD 5.3.2).

Kept in its own module so the frozen part 1-3 files (``predictions.py``, ``decide.py``, ``dt.py``)
stay byte-for-byte unchanged.
"""

from __future__ import annotations

from typing import Any

from easy_validation import api, decide, dt, predictions, rule, units
from easy_validation.evaluate import Context, selection

ALLOWANCE_TARGETS = (0.95, 0.97)
ALLOWANCE_MIN_CASES = 20
ALLOWANCE_POINT_SHARE = 0.90
ALLOWANCE_MAX_DROP = 0.02


def _ratio_f16(r: api.Rung) -> float:
    val = getattr(r.row, "ratio_f16", None)
    if val is not None:
        return float(val)
    if r.ratio is not None:
        return float(r.ratio) / 2.0
    if "float16" in r.name:
        return 1.0
    if "float32" in r.name:
        return 0.5
    return 0.0


def _step_up_rung(sel: Any, pick: api.Rung) -> api.Rung | None:
    """The allowed rung one step more compressed than ``pick`` on the selection's ladder."""
    p_ratio = _ratio_f16(pick)
    higher = [
        r for r in api.rows(sel)
        if not r.is_float and r.status in rule.ALLOWED and _ratio_f16(r) > p_ratio + 1e-9
    ]
    if not higher:
        return None
    higher.sort(key=_ratio_f16)
    return higher[0]


def dt_campaign(ctx: Context) -> dict:
    """Score the DT campaign (PREREG_EASY_PART3.md, FINAL_DESIGN 5.8).

    A type preset ships only if the type has >= 2 scored datasets (>= 150 queries),
    measured with >= 4 models, and low >= target on every scored unit (T in [0.9, 0.95, 0.97, 0.98]).
    Datasets under 150 queries (``dt.REPORTED_ONLY``) are scored and reported only.
    """
    saved_missing = set(ctx.missing)
    has_dt_corpora = any(
        (r / "_corpus" / c).is_dir()
        for r in ctx.roots
        for c in dt.DT_CORPORA
    )
    all_rows: list[dict] = []
    missing_scored: list[str] = []
    by_type: dict[str, dict[str, Any]] = {}

    for doc_type, datasets in dt.DT_TYPES.items():
        scored_ds = [c for c in datasets if c not in dt.REPORTED_ONLY and units.corpus(c).queries >= units.MIN_QUERIES]
        reported_ds = [c for c in datasets if c in dt.REPORTED_ONLY or units.corpus(c).queries < units.MIN_QUERIES]
        type_rows: list[dict] = []
        type_missing: list[str] = []

        for m in dt.DT_MODELS:
            for c in datasets:
                u = ctx.unit(m, c)
                is_rep = c in dt.REPORTED_ONLY or units.corpus(c).queries < units.MIN_QUERIES
                if u is None:
                    if not is_rep:
                        msg = f"{m} on {c}"
                        type_missing.append(msg)
                        missing_scored.append(msg)
                    continue
                for t in units.TARGETS:
                    key = (id(ctx), u.tag, t, '{"arm": "A", "coefficients": null, "hide_text": false, "practice": true, "source": "text_layer"}')
                    if key not in predictions._RULE_CACHE:
                        from easy_validation.evaluate import run_rule
                        predictions._RULE_CACHE[key] = run_rule(ctx, u, t)
                    row = {
                        **predictions._RULE_CACHE[key],
                        "doc_type": doc_type,
                        "scored": bool(u.scored and not is_rep),
                        "reported_only": is_rep,
                    }
                    type_rows.append(row)
                    all_rows.append(row)

        scored_rows = [r for r in type_rows if r["scored"]]
        models_on_all = [
            m for m in dt.DT_MODELS
            if scored_ds and all(
                any(r["model"] == m and r["corpus"] == c for r in scored_rows)
                for c in scored_ds
            )
        ]
        n_low_ok = sum(1 for r in scored_rows if r["low"] >= r["target"])
        low_ok = bool(scored_rows) and n_low_ok == len(scored_rows)
        worst_gap = min((r["low"] - r["target"] for r in scored_rows), default=float("nan"))
        passed = (
            len(scored_ds) >= 2
            and len(models_on_all) >= 4
            and not type_missing
            and low_ok
        ) if has_dt_corpora else None

        reasons = []
        reasons.append(f"{len(scored_ds)} scored dataset(s) (need >= 2)")
        reasons.append(f"{len(models_on_all)}/4 models complete")
        if scored_rows:
            reasons.append(f"low >= T on {n_low_ok}/{len(scored_rows)} units (worst low - T: {worst_gap:+.4f})")
        by_type[doc_type] = {
            "passed": passed,
            "scored_datasets": scored_ds,
            "reported_only_datasets": reported_ds,
            "models_complete": models_on_all,
            "n_models": len(models_on_all),
            "missing": type_missing,
            "n_scored_units": len(scored_rows),
            "n_low_ge_target": n_low_ok,
            "worst_low_minus_target": None if not scored_rows else float(worst_gap),
            "reason": ", ".join(reasons),
        }

    if not has_dt_corpora:
        ctx.missing.clear()
        ctx.missing.update(saved_missing)
        return predictions.result(
            "DT",
            "document-type presets (slides, reports, academic, tables)",
            ">= 2 scored datasets, >= 4 models, low >= T on every unit of the type",
            all_rows,
            None,
            "not scored: DT corpora not present",
            missing_scored,
            by_type=by_type,
        )

    # Restore ctx.missing so missing DT units don't alter Part 1-2 missing set if tracked separately
    ctx.missing.clear()
    ctx.missing.update(saved_missing)

    passed_types = [k for k, v in by_type.items() if v["passed"] is True]
    summary_parts = [f"{k}: {'pass' if v['passed'] else 'fail'} ({v['reason']})" for k, v in by_type.items()]
    overall_passed = bool(passed_types)
    reason = f"{len(passed_types)}/{len(by_type)} types passed; " + "; ".join(summary_parts)
    return predictions.result(
        "DT",
        "document-type presets (slides, reports, academic, tables)",
        ">= 2 scored datasets, >= 4 models, low >= T on every unit of the type",
        all_rows,
        overall_passed,
        reason,
        missing_scored,
        by_type=by_type,
        passed_types=passed_types,
    )


def probe_allowance(ctx: Context, source: str = "text_layer") -> dict:
    """Score the practice-question 1-step upward allowance test (PREREG_EASY_PART3.md, FD 5.3.2).

    On every scored V3 unit (design and held-out models; T in [0.95, 0.97]), take each case where
    the practice check passes on the rung one step more compressed than the catalogue pick.
    Passes only if there are >= 20 such cases, real point >= T in >= 90% of them, and
    real point >= T - 0.02 in all of them.
    """
    rows: list[dict] = []
    for u in predictions.checked_units(ctx, source):
        pr = ctx.practice(u, source)
        if pr is None:
            continue
        sc = ctx.scorer(u)
        for t in ALLOWANCE_TARGETS:
            sel = selection(u, t, checked=True)
            pick = api.pick(sel)
            step_up = _step_up_rung(sel, pick)
            if step_up is None:
                continue
            st = pr.check(pr.rank_fn(step_up.row), t)
            real = sc.retention(step_up.spec)
            rows.append({
                "model": u.model,
                "corpus": u.corpus,
                "heldout": u.heldout,
                "target": t,
                "pick": pick.name,
                "step_up": step_up.name,
                "probe_passed": st["passed"],
                "probe_R": st["R"],
                "probe_U": st["U"],
                "n_hard": st["n"],
                "real_point": real["point"],
                "real_low": real["low"],
                "margin_vs_target": real["point"] - t,
            })

    passing = [r for r in rows if r["probe_passed"]]
    n_cases = len(passing)
    s_ge_t = predictions.share([r["real_point"] >= r["target"] for r in passing]) if passing else float("nan")
    all_above_floor = bool(passing) and all(r["real_point"] >= r["target"] - ALLOWANCE_MAX_DROP for r in passing)
    worst_margin = min((r["real_point"] - r["target"] for r in passing), default=float("nan"))

    pid = "probe_allowance" if source == "text_layer" else f"probe_allowance ({source})"
    if not rows:
        return predictions.result(
            pid,
            "practice check raising layout one step (T in 0.95, 0.97)",
            ">= 20 probe-passing cases, real point >= T on >= 90%, >= T - 0.02 on all",
            rows,
            None,
            f"no checked V3 units for probe source {source!r}",
            probe_source=source,
            in_sample=True,
        )

    ok = n_cases >= ALLOWANCE_MIN_CASES and s_ge_t >= ALLOWANCE_POINT_SHARE and all_above_floor
    if n_cases < ALLOWANCE_MIN_CASES:
        reason = (
            f"only {n_cases} probe-passing step-up case(s) of {len(rows)} tested (need >= {ALLOWANCE_MIN_CASES})"
            + (f"; point >= T on {s_ge_t:.0%}, worst point - T = {worst_margin:+.4f}" if passing else "")
        )
    else:
        reason = (
            f"{n_cases} probe-passing step-up cases (of {len(rows)} tested): "
            f"point >= T on {s_ge_t:.0%} (need >= {ALLOWANCE_POINT_SHARE:.0%}), "
            f"worst point - T = {worst_margin:+.4f} (need >= -{ALLOWANCE_MAX_DROP:.2f})"
        )
    return predictions.result(
        pid,
        "practice check raising layout one step (T in 0.95, 0.97)",
        ">= 20 probe-passing cases, real point >= T on >= 90%, >= T - 0.02 on all",
        rows,
        ok,
        reason,
        probe_source=source,
        n_passing_cases=n_cases,
        share_point_ge_target=None if not passing else float(s_ge_t),
        all_ge_target_minus_002=all_above_floor,
        in_sample=True,
    )


def apply_part3_decisions(results: dict[str, dict], decisions: dict) -> dict:
    """Augment ``decisions`` with any PREREG_EASY_PART3 rules that passed."""
    dt_res = results.get("DT") or {}
    passed_types = dt_res.get("passed_types") or []
    if passed_types:
        setting = {"features.type_presets.on": True, "features.type_presets.enabled_types": passed_types}
        decisions["actions"].append({
            "rule": "DT campaign type rule passes",
            "outcome": f"enable type presets for {passed_types}",
            "set": setting,
            "note": "PREREG_EASY part 3",
        })
        decisions["settings"].update(setting)

    for source in units.PROBE_SOURCES:
        pid = "probe_allowance" if source == "text_layer" else f"probe_allowance ({source})"
        pa = results.get(pid) or {}
        if pa.get("passed") is True:
            key = f"features.probe_allowance.{source}"
            setting = {"features.probe_allowance.on": True, key: True}
            decisions["actions"].append({
                "rule": f"{pid} passes (>= 20 cases, >= 90% >= T, all >= T - 0.02)",
                "outcome": f"enable 1-step upward probe allowance for {source}",
                "set": setting,
                "note": decide.IN_SAMPLE,
            })
            decisions["settings"].update(setting)
            decisions["margin_changed"] = True
            decisions["status"] = decide.IN_SAMPLE

    decisions["ledger_change"]["set"] = decisions["settings"]
    decisions["ledger_change"]["append_outcome"]["predictions"] = {
        k: r.get("passed") for k, r in results.items()
    }
    decisions["ledger_change"]["append_outcome"]["actions"] = [a["rule"] for a in decisions["actions"]]
    decisions["ledger_change"]["append_outcome"]["status"] = decisions["status"]
    return decisions
