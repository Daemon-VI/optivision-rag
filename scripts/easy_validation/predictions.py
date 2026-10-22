"""Score every prediction of FINAL_DESIGN section 8 from the saved units (PREREG_EASY part 1).

Each ``v*`` function returns one result dict: the prediction and its pass rule, one row per
unit, ``passed`` (True, False, or None when a unit it needs is missing or too few pairs
exist) and the reason. Units with fewer than 150 English-original queries are listed but
not scored. Nothing here changes a margin: ``decide.py`` applies the decision rules to these
results and writes a proposed ledger change for a person to apply.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from easy_validation import api, rule, units
from easy_validation.evaluate import Context, run_rule, selection

INT8 = {"stages": [{"scale": "per_vector", "stage": "int8"}]}
INT4_MEAN = {"stages": [{"center": "mean", "stage": "int4"}]}


# --------------------------------------------------------------------------- unit lists


def v1_units() -> list[tuple[str, str]]:
    return [(m, c) for m in units.DESIGN for c in units.V1_CORPORA]


def v1u_units(ctx: Context) -> list[tuple[str, str]]:
    return [(m, c) for m in ctx.heldout for c in units.V1U_CORPORA]


def unchecked_units() -> list[tuple[str, str]]:
    return [(m, c) for m in units.DESIGN for c in units.UNCHECKED_CORPORA]


def v2_units() -> list[tuple[str, str]]:
    return [(m, c) for m in units.V2_MODELS for c in units.V2_CORPORA]


def text_units() -> list[tuple[str, str]]:
    return [(m, c) for m in units.TEXT_MODELS for c in units.TEXT_CORPORA]


def page_units(ctx: Context) -> list[tuple[str, str]]:
    return v1_units() + v1u_units(ctx) + unchecked_units() + v2_units()


def checked_units(ctx: Context, source: str = "text_layer") -> list[Any]:
    out = []
    for m, c in v1_units() + v1u_units(ctx) + v2_units():
        u = ctx.unit(m, c)
        if u is None or not u.scored:
            continue
        pr = ctx.practice(u, source)
        if pr is not None and pr.checked:
            out.append(u)
    return out


# --------------------------------------------------------------------------- helpers


def result(pid: str, prediction: str, pass_if: str, rows: list[dict], passed: bool | None, reason: str = "",
           missing: Iterable[str] = (), **extra: Any) -> dict:
    return {"id": pid, "prediction": prediction, "pass_if": pass_if, "passed": passed, "reason": reason,
            "n_units": len(rows), "missing": sorted(set(missing)), "units": rows, **extra}


def share(values: list[bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def gather(ctx: Context, pairs: list[tuple[str, str]], fn: Callable[[Any], list[dict]]) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for m, c in pairs:
        u = ctx.unit(m, c)
        if u is None:
            missing.append(f"{m} on {c}")
            continue
        if not u.scored:
            rows.append({"model": m, "corpus": c, "scored": False, "n_queries": u.n_queries})
            continue
        rows += fn(u)
    return rows, missing


def scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("scored", True)]


_RULE_CACHE: dict = {}


def rule_rows(ctx: Context, pairs: list[tuple[str, str]], targets: Iterable[float], **kw: Any) -> tuple[list, list]:
    kw = {"arm": "A", "hide_text": False, "practice": True, "source": "text_layer", "coefficients": None, **kw}

    def fn(u: Any) -> list[dict]:
        out = []
        for t in targets:
            key = (id(ctx), u.tag, t, json.dumps(kw, sort_keys=True, default=str))
            if key not in _RULE_CACHE:
                _RULE_CACHE[key] = run_rule(ctx, u, t, **kw)
            out.append(_RULE_CACHE[key])
        return out

    return gather(ctx, pairs, fn)


def crit_v1b(rows: list[dict]) -> tuple[bool, str]:
    th = units.THRESHOLDS["V1b"]
    pts = [r["point"] for r in rows]
    s = share([p >= th["point_min"] for p in pts])
    ok = s >= th["share"] and min(pts) >= th["all_point_min"]
    return ok, f"point >= {th['point_min']} on {s:.0%} of {len(pts)} units (need {th['share']:.0%}); lowest {min(pts):.4f} (need >= {th['all_point_min']})"


def crit_v1c(rows: list[dict]) -> tuple[bool, str]:
    th = units.THRESHOLDS["V1c"]
    s1 = share([r["point"] >= th["point_min"] for r in rows])
    s2 = share([r["low"] >= th["low_min"] for r in rows])
    ok = s1 >= th["share"] and s2 >= th["low_share"]
    return ok, (f"point >= {th['point_min']} on {s1:.0%}, low >= {th['low_min']} on {s2:.0%} of {len(rows)} units "
                f"(need {th['share']:.0%} each)")


def crit_share(rows: list[dict], target: float, need: float) -> tuple[bool, str]:
    s = share([r["point"] >= target for r in rows])
    return s >= need, f"point >= {target} on {s:.0%} of {len(rows)} units (need {need:.0%})"


def at(rows: list[dict], t: float) -> list[dict]:
    return [r for r in scored(rows) if abs(r.get("target", -1) - t) < 1e-9]


def _verdict(checks: list[tuple[bool, str]], missing: list[str]) -> tuple[bool | None, str]:
    reason = "; ".join(text for _, text in checks)
    if missing:
        return None, f"not scored: {len(missing)} unit(s) missing; " + reason
    return all(ok for ok, _ in checks), reason


# --------------------------------------------------------------------------- V1


def v1a(ctx: Context) -> dict:
    th = units.THRESHOLDS["V1a"]

    def fn(u: Any) -> list[dict]:
        r = ctx.scorer(u).retention(INT8)
        return [{"model": u.model, "corpus": u.corpus, "heldout": u.heldout, "point": r["point"], "low": r["low"],
                 "n_queries": r["n_queries"]}]

    rows, missing = gather(ctx, page_units(ctx), fn)
    rs = scored(rows)
    ok = bool(rs) and all(r["point"] >= th["point_min"] for r in rs)
    worst = min((r["point"] for r in rs), default=float("nan"))
    passed, reason = _verdict([(ok, f"int8 point >= 0.97 on {sum(r['point'] >= 0.97 for r in rs)} of {len(rs)} page "
                                    f"units; lowest {worst:.4f}")], missing)
    return result("V1a", "int8 is safe", "point >= 0.97 on every page unit, held-out included", rows, passed, reason,
                  missing)


def v1bcd(ctx: Context) -> list[dict]:
    rows, missing = rule_rows(ctx, v1_units(), units.TARGETS)
    out = []
    r97, r95 = at(rows, 0.97), at(rows, 0.95)
    p, why = _verdict([crit_v1b(r97)] if r97 else [(False, "no units")], missing)
    out.append(result("V1b", "rule + practice check, 0.97", "point >= 0.97 on >= 90% of units, >= 0.94 on all",
                      r97, p, why, missing))
    p, why = _verdict([crit_v1c(r95)] if r95 else [(False, "no units")], missing)
    out.append(result("V1c", "at 0.95", "point >= 0.95 on >= 90%; low >= 0.93 on >= 90%", r95, p, why, missing))
    checks = [crit_share(at(rows, t), t, units.THRESHOLDS["V1d"]["share"]) for t in units.THRESHOLDS["V1d"]["targets"]]
    p, why = _verdict(checks, missing)
    out.append(result("V1d", "at 0.98, 0.90", "point >= T on >= 90%, each", at(rows, 0.98) + at(rows, 0.90), p, why,
                      missing, per_target={str(t): ok for (ok, _), t in zip(checks, units.THRESHOLDS["V1d"]["targets"],
                                                                             strict=True)}))
    return out


def v1_unchecked(ctx: Context) -> dict:
    a, m1 = rule_rows(ctx, unchecked_units(), (0.95, 0.97))
    b, m2 = rule_rows(ctx, v1_units(), (0.95, 0.97), hide_text=True)
    rows, missing = a + b, m1 + m2
    checks = [crit_v1b(at(rows, 0.97)) if at(rows, 0.97) else (False, "no units"),
              crit_v1c(at(rows, 0.95)) if at(rows, 0.95) else (False, "no units")]
    passed, why = _verdict(checks, missing)
    return result("V1-unchecked", "unchecked rule on TechSlides and on V1 corpora with text hidden",
                  "V1b thresholds at 0.97 and V1c thresholds at 0.95", at(rows, 0.97) + at(rows, 0.95), passed, why,
                  missing)


def arm_summary(rows: list[dict], severe: float) -> dict:
    misses = [r for r in rows if r["point"] < r["target"]]
    sev = [r for r in misses if r["point"] < r["target"] - severe]
    return {"units": len(rows), "misses": len(misses), "severe": len(sev),
            "severe_one_bit": sum(1 for r in sev if r.get("chosen_one_bit")),
            "by_target": {str(t): {"misses": sum(1 for r in misses if r["target"] == t),
                                   "severe": sum(1 for r in sev if r["target"] == t)} for t in units.TARGETS}}


def arm_passes(rows: list[dict]) -> dict:
    r97, r95 = at(rows, 0.97), at(rows, 0.95)
    return {"V1b": crit_v1b(r97)[0] if r97 else None, "V1c": crit_v1c(r95)[0] if r95 else None,
            "V1d": all(crit_share(at(rows, t), t, 0.90)[0] for t in (0.98, 0.90)) if at(rows, 0.98) else None}


def v2_criteria(rows: list[dict], int8_points: list[float]) -> tuple[bool, str]:
    th = units.THRESHOLDS["V2"]
    rs = [r for r in rows if r["target"] in th["targets"]]
    ok_t = bool(rs) and all(r["point"] >= r["target"] for r in rs)
    ok_8 = bool(int8_points) and all(p >= th["int8_point_min"] for p in int8_points)
    return ok_t and ok_8, (f"point >= T in {sum(r['point'] >= r['target'] for r in rs)}/{len(rs)}; int8 point "
                           f">= 0.97 on {sum(p >= 0.97 for p in int8_points)}/{len(int8_points)}")


def v1_arms(ctx: Context, int8_v2: list[float]) -> dict:
    sev = units.THRESHOLDS["V1-arms"]["severe_below"]
    arms: dict[str, Any] = {}
    all_rows: list[dict] = []
    missing: list[str] = []
    for arm in rule.ARMS:
        rows, missing = rule_rows(ctx, v1_units(), units.TARGETS, arm=arm)
        rows = scored(rows)
        v2rows, m2 = rule_rows(ctx, v2_units(), units.THRESHOLDS["V2"]["targets"], arm=arm)
        entry = {**arm_summary(rows, sev), "passes": arm_passes(rows),
                 "V2": v2_criteria(scored(v2rows), int8_v2)[0] if scored(v2rows) and not m2 else None}
        arms[arm] = entry
        all_rows += [{**r, "arm": arm} for r in rows]
    agree = [r["rule_agrees_with_select"] for r in all_rows if r["arm"] == "A"]
    comparable = bool(agree) and all(agree)
    return result("V1-arms", "B (no 1-bit margin), C (s(N) = 0), D (flat +0.01)",
                  "reported with misses and severe misses (< T - 0.02)", all_rows,
                  None if missing else True, "reported, not pass/fail" + ("" if comparable else
                  "; arm A recomputed from the rows disagrees with select on some units: arms not comparable"),
                  missing, arms=arms, comparable=comparable)


def v1t(ctx: Context) -> dict:
    rows, missing = rule_rows(ctx, text_units(), units.THRESHOLDS["V1t"]["targets"], practice=False)
    checks = [crit_share(at(rows, t), t, units.THRESHOLDS["V1t"]["share"]) if at(rows, t) else (False, "no units")
              for t in units.THRESHOLDS["V1t"]["targets"]]
    passed, why = _verdict(checks, missing)
    return result("V1t", "text rule", "point >= T on >= 90% of text units at 0.97, 0.95", scored(rows), passed, why,
                  missing)


def v1u(ctx: Context) -> dict:
    pairs = v1u_units(ctx)
    rows, missing = rule_rows(ctx, pairs, units.TARGETS, practice=False)

    def lows(u: Any) -> list[dict]:
        sc = ctx.scorer(u)
        return [{"model": u.model, "corpus": u.corpus, "int8_low": sc.retention(INT8)["low"],
                 "int4_mean_low": sc.retention(INT4_MEAN)["low"]}]

    codec, _ = gather(ctx, pairs, lows)
    rs = scored(rows)
    if len(ctx.heldout) < 2:
        return result("V1u", "unmeasured rule, held-out encoders", "picks keep point >= T on every unit", rs, None,
                      f"only {len(ctx.heldout)} held-out encoder(s) passed check_retriever", missing, codec_lows=codec)
    ok = bool(rs) and all(r["point"] >= r["target"] for r in rs)
    passed, why = _verdict([(ok, f"point >= T on {sum(r['point'] >= r['target'] for r in rs)} of {len(rs)} units")],
                           missing)
    return result("V1u", "unmeasured rule, held-out encoders", "picks keep point >= T on every unit; int8 and "
                  "int4(mean) lows reported", rs, passed, why, missing, codec_lows=codec,
                  int8_low_min=min((c["int8_low"] for c in codec), default=None),
                  int4_mean_low_min=min((c["int4_mean_low"] for c in codec), default=None))


def v2(ctx: Context) -> tuple[dict, list[float]]:
    rows, missing = rule_rows(ctx, v2_units(), units.THRESHOLDS["V2"]["targets"])
    int8 = []
    for m, c in v2_units():
        u = ctx.unit(m, c)
        if u is not None and u.scored:
            int8.append(ctx.scorer(u).retention(INT8)["point"])
    ok, why = v2_criteria(scored(rows), int8)
    passed, why = _verdict([(ok, why)], missing)
    return result("V2", "size rule at 5,244 pages (colSmol-500M, colqwen2-v1.0-hf)",
                  "point >= T in 4/4 (0.95, 0.97); int8 >= 0.97", scored(rows), passed, why, missing,
                  int8_points=int8), int8


# --------------------------------------------------------------------------- S, V3


def ladder_rungs(ctx: Context, u: Any) -> list[api.Rung]:
    """Every compressed ladder layout of the unit's modality (from the rule's own rows)."""
    sel = selection(u, 0.97, checked=True)
    return [r for r in api.rows(sel) if not r.is_float and r.spec.get("stages")]


def size_core(u: Any, n: int = 250, seed: int = units.S_SEED) -> tuple[list[str], list[int]]:
    """Query set Q and the page order of the size study: queries in a seeded order join while
    the union of their relevant pages stays within ``n``; every other page follows in a seeded
    order, so each smaller corpus is a prefix of the next and holds every page labelled for Q."""
    rng = np.random.default_rng(seed)
    pos = {p: i for i, p in enumerate(u.docs.ids)}
    qids = [u.queries.ids[i] for i in rng.permutation(len(u.queries))]
    core: set[int] = set()
    chosen = []
    for q in qids:
        rel = {pos[p] for p in u.qrels.get(q, {}) if p in pos}
        if rel and len(core | rel) <= n:
            core |= rel
            chosen.append(q)
    rest = [i for i in rng.permutation(len(u.docs)) if i not in core]
    order = sorted(core) + [int(i) for i in rest]
    return chosen, order


def size_study(ctx: Context) -> dict:
    sizes = [s for s in units.S_SIZES if s != "full"]
    rows, missing = [], []
    drops: dict[str, list[tuple[float, float]]] = {}
    gains: dict[str, list[float]] = {}
    for m, c in v1_units() + v2_units():
        u = ctx.unit(m, c)
        if u is None:
            missing.append(f"{m} on {c}")
            continue
        sc = ctx.scorer(u)
        q, order = size_core(u)
        ns = [n for n in sizes if n < u.n_pages] + [u.n_pages]
        for r in ladder_rungs(ctx, u):
            lows = {}
            for n in ns:
                subset = tuple(sorted(order[:n]))
                res = sc.retention(r.spec, subset=subset, qids=q)
                lows[n] = res["low"]
                rows.append({"model": m, "corpus": c, "layout": r.name, "class": r.size_class, "n_pages": n,
                             "n_queries": res["n_queries"], "point": res["point"], "low": res["low"],
                             "kept_out": c in units.V2_CORPORA})
            if c in units.V2_CORPORA or 500 not in lows:
                continue
            for n, low in lows.items():
                if n > 500:
                    drops.setdefault(r.size_class, []).append((lows[500] - low, units.s_of_n(n)))
            if 250 in lows:
                gains.setdefault(r.size_class, []).append(lows[250] - lows[500])
    coefficients = rule.refit_coefficients(drops)
    below = {cls: (units.round_down(min(g) / abs(units.s_signed(250))) if g and min(g) > 0 else None)
             for cls, g in gains.items()}
    return result("S", "size study: V1/V2 vectors at N in {250, 500, 1,000, full}", "every rung scored", rows,
                  None if missing else True, "reported; arm S refits the size coefficients from the V1 units", missing,
                  coefficients=coefficients, in_sample=True,
                  below_500={"coefficient_per_unit_s": below,
                             "rule": "proposed only when every V1 unit's low rose from 500 to 250 pages "
                                     "(smallest rise per unit of |s(250)|, rounded down to 0.0025)"})


def arm_s(ctx: Context, coefficients: dict[str, float], int8_v2: list[float]) -> dict:
    rows, missing = rule_rows(ctx, v2_units(), units.THRESHOLDS["V2"]["targets"], arm="S", coefficients=coefficients)
    ok, why = v2_criteria(scored(rows), int8_v2)
    passed, why = _verdict([(ok, why)], missing)
    return result("arm S", "refit size coefficients, checked on V2 (kept out)", "V2 passes with the refit coefficients",
                  scored(rows), passed, why, missing, coefficients=coefficients, in_sample=True)


def v3(ctx: Context) -> dict:
    th = units.THRESHOLDS["V3"]
    rows, missing = [], []
    for m, c in v1_units():
        u = ctx.unit(m, c)
        if u is None:
            missing.append(f"{m} on {c}")
            continue
        sc = ctx.scorer(u)
        for r in ladder_rungs(ctx, u):
            if u.docs32 is not None:
                a = sc.retention(r.spec, variant="f16", base_variant="f32")["point"]
                b = sc.retention(r.spec, variant="f32", base_variant="f32")["point"]
                rows.append({"model": m, "corpus": c, "layout": r.name, "pair": "float16 vs float32 inputs",
                             "delta": a - b})
            from easy_validation.scorer import stateful

            if stateful(r.spec):
                full = sc.retention(r.spec, fit="full")["point"]
                rows.append({"model": m, "corpus": c, "layout": r.name, "pair": "sample fit vs full fit",
                             "delta": sc.retention(r.spec, fit="sample")["point"] - full})
                rows.append({"model": m, "corpus": c, "layout": r.name, "pair": "state reused after 2x growth vs refit",
                             "delta": sc.retention(r.spec, fit="growth")["point"] - full})
    ok_pairs = [abs(r["delta"]) <= th["abs_delta_max"] for r in rows]
    s = share(ok_pairs)
    by_pair = {p: share([abs(r["delta"]) <= th["abs_delta_max"] for r in rows if r["pair"] == p])
               for p in sorted({r["pair"] for r in rows})}
    passed, why = _verdict([(bool(rows) and s >= th["share"], f"|delta point| <= 0.005 on {s:.1%} of {len(rows)} "
                                                             f"pairs (need {th['share']:.0%})")], missing)
    return result("V3", "refit and growth", "|delta point| <= 0.005 on >= 95% of pairs", rows, passed, why, missing,
                  by_pair=by_pair)


# --------------------------------------------------------------------------- V4, V5


def v4(ctx: Context, source: str = "text_layer") -> dict:
    rows, missing = [], []
    for u in checked_units(ctx, source):
        pr = ctx.practice(u, source)
        for t in units.TARGETS:
            r = run_rule(ctx, u, t, practice=False, source=source)
            sel = selection(u, t, checked=True)
            p = api.pick(sel)
            if p.is_float:
                continue
            stat = pr.check(pr.rank_fn(p.row), t)
            rows.append({"model": u.model, "corpus": u.corpus, "target": t, "layout": p.name, "probe_R": stat["R"],
                         "real_point": r["pick_point"], "difference": stat["R"] - r["pick_point"], "n_hard": stat["n"]})
    s = share([r["difference"] > 0 for r in rows])
    th = units.THRESHOLDS["V4"]
    passed = (s >= th["share"]) if rows else None
    return result("V4" if source == "text_layer" else f"V4 ({source})", "probes are optimistic",
                  "probe minus real retention > 0 in >= 70% of units", rows, passed,
                  f"probe - real > 0 in {s:.0%} of {len(rows)} units (rule picks other than float)" if rows
                  else "no checked unit with a compressed pick", missing, probe_source=source)


def v5(ctx: Context, source: str = "text_layer") -> list[dict]:
    th = units.THRESHOLDS
    faults, forced, cost = [], [], []
    for u in checked_units(ctx, source):
        pr = ctx.practice(u, source)
        sc = ctx.scorer(u)
        real_fault = {}
        for name, spec in units.FAULTS.items():
            if spec is None:
                base = sc.per_query(sc.scores(None))
                c = sc.per_query(sc.random_scores("queries", list(u.queries.ids)))
                from optivision.evaluation import retention

                real_fault[name] = retention(c, base, n_boot=0)[0]
            else:
                real_fault[name] = sc.retention(spec)["point"]
        rankings = {name: pr.fault_ranking(name) for name in units.FAULTS}
        rungs = ladder_rungs(ctx, u)
        rung_rank = {r.key: pr.rank_fn(r.row) for r in rungs}
        rung_real = {r.key: sc.retention(r.spec) for r in rungs}
        for t in units.TARGETS:
            for name in units.FAULTS:
                st = pr.check(rankings[name], t)
                faults.append({"model": u.model, "corpus": u.corpus, "target": t, "fault": name,
                               "real_point": real_fault[name], "fires": st["fires"], "U": st["U"], "R": st["R"]})
            for r in rungs:
                st = pr.check(rung_rank[r.key], t)
                forced.append({"model": u.model, "corpus": u.corpus, "target": t, "layout": r.name,
                               "real_point": rung_real[r.key]["point"], "real_low": rung_real[r.key]["low"],
                               "fires": st["fires"], "U": st["U"]})
            with_check = run_rule(ctx, u, t, source=source)
            cost.append({"model": u.model, "corpus": u.corpus, "target": t, "pick": with_check["pick"],
                         "chosen": with_check["chosen"], "meets_with": with_check["point"] >= t,
                         "meets_without": with_check["pick_point"] >= t,
                         "ratio_with": with_check["ratio_f16"], "ratio_without": with_check["pick_ratio_f16"]})
    suffix = "" if source == "text_layer" else f" ({source})"
    out = []
    a = [r for r in faults if r["real_point"] < th["V5a"]["real_below"]]
    sa = share([r["fires"] for r in a])
    out.append(result("V5a" + suffix, "planted faults (random scores; project(16) > binary; adaptive_merge(0.6) > int4)",
                      "fires on >= 95% of (checked unit, fault) pairs with real retention < 0.80", faults,
                      (sa >= th["V5a"]["fire_share"]) if a else None,
                      f"fires on {sa:.0%} of {len(a)} pairs" if a else "no pair with real retention < 0.80",
                      probe_source=source))
    b = [r for r in forced if r["real_low"] >= r["target"]]
    sb = share([r["fires"] for r in b])
    out.append(result("V5b" + suffix, "stays quiet", "fires on <= 10% of forced pairs with real low >= T", b,
                      (sb <= th["V5b"]["fire_share_max"]) if b else None,
                      f"fires on {sb:.0%} of {len(b)} pairs" if b else "no forced pair with real low >= T",
                      probe_source=source))
    c = [r for r in forced if r["real_point"] < r["target"] - th["V5c"]["miss_below"]]
    sc_ = share([r["fires"] for r in c])
    enough = len(c) >= th["V5c"]["min_pairs"]
    out.append(result("V5c" + suffix, "catches misses", "fires on >= 50% of pairs with real point < T - 0.03 "
                      "(>= 6 pairs)", c, (sc_ >= th["V5c"]["fire_share"]) if enough else None,
                      f"fires on {sc_:.0%} of {len(c)} pairs" if enough else f"only {len(c)} pairs (need 6)",
                      probe_source=source))
    with_n = sum(r["meets_with"] for r in cost)
    without_n = sum(r["meets_without"] for r in cost)
    ratios = [r["ratio_with"] / r["ratio_without"] for r in cost if r["ratio_without"]]
    med = statistics.median(ratios) if ratios else float("nan")
    ok = with_n >= without_n and med >= th["V5d"]["median_ratio_min"]
    out.append(result("V5d" + suffix, "costs little", "units meeting T with check >= without; median ratio >= 0.8x",
                      cost, ok if cost else None, f"meeting T: {with_n} with the check, {without_n} without; median "
                      f"ratio {med:.2f}x" if cost else "no checked unit", probe_source=source))
    return out


# --------------------------------------------------------------------------- V6


def v6(ctx: Context) -> dict:
    found = [r / "wiring.json" for r in ctx.roots if (r / "wiring.json").exists()]
    if not found:
        return result("V6", "wiring", "passes on every loader that loads on Kaggle; fails on a re-initialised norm "
                      "and a dtype mismatch", [], None, "wiring.json missing (batch e1b not ingested)")
    w = json.loads(found[0].read_text(encoding="utf-8"))
    rows = [{"loader": k, **{x: v.get(x) for x in ("model", "loaded", "error")},
             "ok": (v.get("wiring") or {}).get("ok")} for k, v in w.get("loaders", {}).items()]
    loaded = [r for r in rows if r["loaded"]]
    faults = [{"fault": k, "ok": v.get("ok"), "message": v.get("message")} for k, v in w.get("faults", {}).items()]
    passes = bool(loaded) and all(r["ok"] for r in loaded)
    caught = len(faults) == 2 * len(units.V6_FAULT_MODELS) and all(f["ok"] is False for f in faults)
    return result("V6", "wiring", "passes on every loader that loads on Kaggle; fails on a re-initialised norm and a "
                  "dtype mismatch", rows, passes and caught,
                  f"{sum(bool(r['ok']) for r in loaded)} of {len(loaded)} loaded loaders passed; "
                  f"{sum(f['ok'] is False for f in faults)} of {len(faults)} planted faults failed the check",
                  faults=faults)


# --------------------------------------------------------------------------- all


def score_all(ctx: Context) -> dict[str, dict]:
    """Every prediction, keyed by id (V1a ... V6, S, arm S, and the per-source V4/V5 variants)."""
    _RULE_CACHE.clear()
    out: dict[str, dict] = {}
    out["V1a"] = v1a(ctx)
    for r in v1bcd(ctx):
        out[r["id"]] = r
    out["V1-unchecked"] = v1_unchecked(ctx)
    out["V2"], int8_v2 = v2(ctx)
    out["V1-arms"] = v1_arms(ctx, int8_v2)
    out["V1t"] = v1t(ctx)
    out["V1u"] = v1u(ctx)
    out["S"] = size_study(ctx)
    out["arm S"] = arm_s(ctx, out["S"]["coefficients"], int8_v2)
    out["V3"] = v3(ctx)
    for source in units.PROBE_SOURCES:
        r = v4(ctx, source)
        if source == "text_layer" or r["units"]:
            out[r["id"]] = r
        for r in v5(ctx, source):
            if source == "text_layer" or r["units"]:
                out[r["id"]] = r
    out["V6"] = v6(ctx)
    return out
