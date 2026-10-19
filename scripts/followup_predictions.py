"""Score the follow-up batches against the predictions written before they ran.

    python scripts/followup_predictions.py

Reads what ``scripts/ingest_campaign.py`` put in ``reports/universal/followup/`` and, for the
original-corpus side of the corpus-size predictions, ``reports/universal/campaign/``. Writes
``reports/universal/followup/PREDICTIONS_STATUS.md`` and ``predictions_status.json``.

Each prediction in ``reports/universal/campaign_analysis/PREDICTIONS.md`` gets one status:

- ``held`` or ``failed``, once every unit it needs has a result. A prediction about every unit
  is ``failed`` as soon as one unit misses, and a count ("at least 16 of 18") as soon as too many
  have missed to reach it.
- ``not testable``, when a unit it needs did not run: its batch came back and the model failed,
  or was skipped.
- ``pending``, while a batch it needs has not come back. Pooled statistics are shown as interim
  values only.

The thresholds are copied from PREDICTIONS.md and are not to be changed here.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = ROOT / "reports" / "universal" / "followup"
CAMPAIGN = ROOT / "reports" / "universal" / "campaign"

NFCORPUS = ["colbert-ir/colbertv2.0", "answerdotai/answerai-colbert-small-v1", "lightonai/LateOn",
            "mixedbread-ai/mxbai-edge-colbert-v0-17m", "lightonai/mLateOn", "lightonai/GTE-ModernColBERT-v1",
            "jinaai/jina-colbert-v2", "LiquidAI/LFM2-ColBERT-350M"]
PAGE_MODELS = ["vidore/colSmol-500M", "vidore/colpali-v1.3", "vidore/colqwen2.5-v0.2",
               "vultr/VultronRetrieverCore-Qwen3.5-4.5B", "tencent/EVIE-4.5B"]
PAGE_SETS = ["arxivqa_test_subsampled", "tabfquad_test_subsampled"]
BATCH_OF = {**{m: "f1_text_nfcorpus_a" for m in NFCORPUS[:4]}, **{m: "f2_text_nfcorpus_b" for m in NFCORPUS[4:]},
            **{m: "f4_pages_128d" for m in PAGE_MODELS[:3]}, **{m: "f5_pages_wide" for m in PAGE_MODELS[3:]}}

#: (model, dataset key) -> (batch, dataset key of the original corpus, where the original run is)
SCALE = {
    ("colbert-ir/colbertv2.0", "scifact_plus10000"): ("f3_text_scale", "scifact", "campaign"),
    ("answerdotai/answerai-colbert-small-v1", "scifact_plus10000"): ("f3_text_scale", "scifact", "f3b_answerai_scifact"),
    ("vidore/colSmol-500M", "docvqa_test_subsampled_plus2000"): ("f6_scale_colsmol", "docvqa_test_subsampled", "campaign"),
    ("vidore/colpali-v1.3", "docvqa_test_subsampled_plus2000"): ("f7_scale_colpali", "docvqa_test_subsampled", "campaign"),
}
SELECTION_BATCHES = ["f1_text_nfcorpus_a", "f2_text_nfcorpus_b", "f3_text_scale", "f4_pages_128d", "f5_pages_wide",
                     "f6_scale_colsmol", "f7_scale_colpali"]

DEGRADED = ["nvidia/nemotron-colembed-vl-4b-v2", "nvidia/nemotron-colembed-vl-8b-v2", "lightonai/LateOn",
            "lightonai/mLateOn", "mixedbread-ai/mxbai-edge-colbert-v0-17m", "mixedbread-ai/mxbai-edge-colbert-v0-32m",
            "lightonai/GTE-ModernColBERT-v1"]
CONTROLS = ["lightonai/Reason-ModernColBERT", "lightonai/ColBERT-Zero", "mixedbread-ai/mxbai-colbert-large-v1",
            "colbert-ir/colbertv2.0", "LiquidAI/LFM2-ColBERT-350M", "tencent/EVIE-4.5B", "tencent/EVIE-8B",
            "vidore/colpali-v1.3"]
STAT = ("codecs", "int4", "score_pearson", "median")

INT8, INT4, INT4M, BINARY = "int8(per_vector)", "int4", "int4(mean)", "binary"
HM25, HM10 = "hierarchical_merge(0.25)", "hierarchical_merge(0.1)"
HM25_INT8, HM25_BINARY = "hierarchical_merge(0.25) > int8(per_vector)", "hierarchical_merge(0.25) > binary"


def folder(model: str) -> str:
    return model.replace("/", "--", 1)


def short(model: str) -> str:
    return model.split("/", 1)[1]


def batch_done(batch: str) -> bool:
    return (FOLLOWUP / batch / "_batch" / "batch_summary.json").exists()


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def wide(batch: str, model: str, key: str) -> dict | None:
    if batch == "campaign":
        return read(CAMPAIGN / folder(model) / "wide" / f"{short(model)}_{key}.json")
    return read(FOLLOWUP / batch / folder(model) / "wide" / f"{short(model)}_{key}.json")


def geometry(batch: str, model: str, key: str | None = None) -> dict | None:
    """The unit's geometry file; with no key, the model's one file in the batch (f8)."""
    base = FOLLOWUP / batch / folder(model) / "geometry"
    if key is not None:
        return read(base / f"{short(model)}_{key}.json")
    files = sorted(base.glob("*.json")) if base.exists() else []
    return read(files[0]) if files else None


def row(result: dict, label: str) -> dict:
    found = [r for r in result["rows"] if r["label"] == label]
    if len(found) != 1:
        raise KeyError(f"{result['dataset']}: {len(found)} rows labelled {label!r}")
    return found[0]


def ret(result: dict, label: str) -> float:
    return row(result, label)["retention:ndcg@5"]


def lower(result: dict, label: str) -> float:
    return row(result, label)["retention_ci:ndcg@5"][0]


def tiered(result: dict) -> float:
    found = [t for t in result["tiered"] if t["hot"] == "binary" and t["candidates"] == 50
             and t["cold"].startswith("int8")]
    if len(found) != 1:
        raise KeyError(f"{result['dataset']}: {len(found)} binary + int8 tiered rows with 50 candidates")
    return found[0]["retention:ndcg@5"]


def get(d: dict, path: tuple[str, ...]) -> float:
    for k in path:
        d = d[k]
    return d


class Unit:
    """One model on one dataset: its result, or why it has none."""

    def __init__(self, model: str, key: str, batch: str) -> None:
        self.model, self.key, self.batch = model, key, batch
        self.result = wide(batch, model, key)
        self.state = "ran" if self.result else ("not run" if batch_done(batch) else "pending")

    @property
    def name(self) -> str:
        return f"{short(self.model)} / {self.key}"


def new_dataset_units() -> list[Unit]:
    units = [Unit(m, "nfcorpus", BATCH_OF[m]) for m in NFCORPUS]
    return units + [Unit(m, s, BATCH_OF[m]) for m in PAGE_MODELS for s in PAGE_SETS]


def each(units: list[Unit], test, need: int | None = None) -> tuple[str, str]:
    """Status of "the test holds on every unit" (or on at least ``need`` of them)."""
    need = len(units) if need is None else need
    ran = [u for u in units if u.state == "ran"]
    misses = [u.name for u in ran if not test(u.result)]
    hits = len(ran) - len(misses)
    missing = [u for u in units if u.state != "ran"]
    detail = f"{hits} of {len(ran)} units in so far meet it (need {need} of {len(units)})"
    if misses:
        detail += "; misses: " + ", ".join(misses)
    if len(misses) > len(units) - need:
        return "failed", detail
    if hits >= need:
        return "held", detail
    if any(u.state == "pending" for u in missing):
        return "pending", detail
    return "not testable", detail + f"; did not run: {', '.join(u.name for u in missing)}"


def pooled(units: list[Unit], value, compare, what: str) -> tuple[str, str]:
    """Status of a statistic over all the units, known only once every one is in."""
    ran = [u for u in units if u.state == "ran"]
    if not ran:
        return ("pending" if any(u.state == "pending" for u in units) else "not testable"), "no unit in yet"
    v = value(ran)
    detail = f"{what} = {v} over {len(ran)} of {len(units)} units"
    if len(ran) < len(units):
        pending = any(u.state == "pending" for u in units)
        return ("pending" if pending else "not testable"), detail + " (interim)"
    return ("held" if compare(v) else "failed"), detail


def combine(parts: list[tuple[str, str]]) -> tuple[str, str]:
    states = [s for s, _ in parts]
    for s in ("failed", "pending", "not testable"):
        if s in states:
            break
    else:
        s = "held"
    return s, " | ".join(d for _, d in parts)


def p1(units):
    return each(units, lambda r: lower(r, INT8) >= 0.95 and ret(r, INT8) >= 0.97)


def p2(units):
    return combine([each(units, lambda r: ret(r, INT4M) >= 0.93),
                    each(units, lambda r: lower(r, INT4M) >= 0.95, need=16)])


def p3(units):
    by = {(u.model, u.key): u for u in units}
    nf = lambda m: [by[(m, "nfcorpus")]]
    pages = [u for u in units if u.key in PAGE_SETS]
    return combine([each(nf("lightonai/LateOn") + nf("mixedbread-ai/mxbai-edge-colbert-v0-17m"),
                         lambda r: ret(r, INT4) < 0.6),
                    each(nf("colbert-ir/colbertv2.0") + nf("jinaai/jina-colbert-v2"), lambda r: ret(r, INT4) >= 0.9),
                    each(pages, lambda r: ret(r, INT4) >= 0.97)])


def p4(units):
    text = [u for u in units if u.key == "nfcorpus"]
    pages = [u for u in units if u.key in PAGE_SETS]
    def binary(ran):
        return round(median(ret(u.result, BINARY) for u in ran), 4)

    a = pooled(text, binary, lambda v: True, "NFCorpus median binary retention")
    b = pooled(pages, binary, lambda v: True, "page-set median binary retention")
    if a[0] == b[0] == "held":  # both complete: now the comparison itself
        return ("held" if binary(text) < binary(pages) else "failed"), f"{a[1]} | {b[1]}"
    return combine([a, b])


def p5(units):
    return each([u for u in units if u.key in PAGE_SETS], lambda r: lower(r, HM25_INT8) >= 0.95, need=8)


def p6(units):
    return each(units, lambda r: ret(r, HM10) < ret(r, HM25))


def p7(units):
    parts = []
    for key in ["nfcorpus", *PAGE_SETS]:
        group = [u for u in units if u.key == key]

        def gap(ran):
            rows = [r for u in ran for r in u.result["rows"] if r["compression_vs_float32"] >= 16]
            return (round(mean(r["retention:recall@1"] for r in rows), 4),
                    round(mean(r["retention:ndcg@5"] for r in rows), 4), len(rows))

        parts.append(pooled(group, gap, lambda v: v[0] < v[1], f"{key}: (mean recall@1, mean ndcg@5, rows)"))
    return combine(parts)


def selection_units() -> list[Unit]:
    units = []
    for u in new_dataset_units() + [Unit(m, k, b) for (m, k), (b, _, _) in SCALE.items()]:
        u.selection = read(FOLLOWUP / u.batch / folder(u.model) / "selection_rules" / f"{short(u.model)}_{u.key}.json")
        if u.state == "ran" and u.selection is None:
            u.state = "not run"  # the study was skipped for time; the unit has no selection result
        units.append(u)
    return units


def met_rate(units: list[Unit], rule: str, target: float) -> float:
    met = [o["holdout_labels"] >= target for u in units for r in u.selection["results"]
           if r["reference"] == "labels" and r["rule"] == rule and r["target"] == target
           for o in r["outcomes"] if o["selected"] is not None]
    return round(sum(met) / len(met), 4)


def p8(_units):
    units = selection_units()
    ran = [u for u in units if u.state == "ran"]
    pending = any(u.state == "pending" for u in units)
    if not ran:
        return ("pending" if pending else "not testable"), "no unit in yet"
    default = {t: met_rate(ran, "calibrate() default", t) for t in (0.95, 0.97, 0.99)}
    point = met_rate(ran, "point", 0.99)
    detail = f"default met rate {default}, point met rate at 0.99 = {point}, over {len(ran)} units"
    ok = all(v >= 0.95 for v in default.values()) and point < 0.80
    if pending:
        return "pending", detail + " (interim)"
    # pooled over the units where the study ran, as the prediction says; a skipped unit is listed
    skipped = [u.name for u in units if u.state != "ran"]
    return ("held" if ok else "failed"), detail + (f"; no study for: {', '.join(skipped)}" if skipped else "")


def scale_units() -> list[tuple[Unit, Unit]]:
    pairs = []
    for (model, key), (batch, original, where) in SCALE.items():
        pairs.append((Unit(model, key, batch), Unit(model, original, where) if where != "campaign"
                      else _campaign_unit(model, original)))
    return pairs


def _campaign_unit(model: str, key: str) -> Unit:
    u = Unit.__new__(Unit)
    u.model, u.key, u.batch = model, key, "campaign"
    u.result = wide("campaign", model, key)
    u.state = "ran" if u.result else "not run"
    return u


def paired(test) -> tuple[str, str]:
    """A test of each enlarged corpus against its original."""
    parts, states = [], []
    for big, small in scale_units():
        if big.state == "ran" and small.state == "ran":
            ok, text = test(small.result, big.result)
            states.append("held" if ok else "failed")
            parts.append(f"{short(big.model)}: {text}")
        else:
            missing = [u for u in (big, small) if u.state != "ran"]
            states.append("pending" if any(u.state == "pending" for u in missing) else "not testable")
            parts.append(f"{short(big.model)}: {', '.join(f'{u.batch} {u.state}' for u in missing)}")
    return combine(list(zip(states, parts)))


def p9(_units):
    def test(a, b):
        d = {c: round(ret(b, c) - ret(a, c), 4) for c in (INT8, INT4M)}
        return all(abs(v) < 0.01 for v in d.values()), f"change {d}"
    return paired(test)


def p10(_units):
    def test(a, b):
        d = {c: (round(ret(a, c), 4), round(ret(b, c), 4)) for c in (BINARY, HM25_BINARY)}
        return all(big < small for small, big in d.values()), f"(original, enlarged) {d}"
    return paired(test)


def p11(_units):
    def test(a, b):
        small, big = round(tiered(a), 4), round(tiered(b), 4)
        return big < small, f"tiered {small} -> {big}"
    return paired(test)


def f8_stats() -> tuple[str, dict]:
    state = "ran" if batch_done("f8_geometry") else "pending"
    stats = {m: geometry("f8_geometry", m) for m in DEGRADED + CONTROLS} if state == "ran" else {}
    return state, stats


def _need_all(stats: dict, models: list[str]) -> list[str]:
    return [short(m) for m in models if stats.get(m) is None]


def p12(_units):
    state, stats = f8_stats()
    if state == "pending":
        return "pending", "f8 has not come back"
    missing = _need_all(stats, DEGRADED + CONTROLS)
    deg = [get(stats[m], STAT) for m in DEGRADED if stats.get(m)]
    con = [get(stats[m], STAT) for m in CONTROLS if stats.get(m)]
    if not deg or not con:
        return "not testable", f"did not run: {', '.join(missing)}"
    auc = sum(1.0 if d < c else 0.5 if d == c else 0.0 for d in deg for c in con) / (len(deg) * len(con))
    detail = f"AUC = {auc:.3f} over {len(deg)} degraded and {len(con)} controls"
    if missing:
        return "not testable", detail + f"; did not run: {', '.join(missing)}"
    return ("held" if auc >= 0.9 else "failed"), detail


def p13(_units):
    state, stats = f8_stats()
    if state == "pending":
        return "pending", "f8 has not come back"
    parts = []
    for models, path, name in (
            (DEGRADED[:3], ("peak_to_rms", "median"), "peak_to_rms.median"),
            ([DEGRADED[4], DEGRADED[5], DEGRADED[3]], ("doc_cos_to_mean", "median"), "doc_cos_to_mean.median")):
        missing = _need_all(stats, models + CONTROLS)
        con = [get(stats[m], path) for m in CONTROLS if stats.get(m)]
        top = max(con) if con else None
        values = {short(m): round(get(stats[m], path), 4) for m in models if stats.get(m)}
        detail = f"{name}: {values} against the controls' highest {top if top is None else round(top, 4)}"
        if top is not None and any(v <= top for v in values.values()):
            parts.append(("failed", detail))
        elif missing:
            parts.append(("not testable", detail + f"; did not run: {', '.join(missing)}"))
        else:
            parts.append(("held", detail))
    return combine(parts)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def p14(units):
    for u in units:
        g = geometry(u.batch, u.model, u.key) if u.state == "ran" else None
        if u.state == "ran" and g is None:
            u.state = "not run"  # measured, but without a geometry file
        u.geometry = g

    def rho(ran):
        return round(spearman([get(u.geometry, STAT) for u in ran], [ret(u.result, INT4) for u in ran]), 4)

    return pooled(units, rho, lambda v: v >= 0.5, "Spearman(statistic, plain int4 retention)")


PREDICTIONS = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6, "P7": p7, "P8": p8, "P9": p9, "P10": p10,
               "P11": p11, "P12": p12, "P13": p13, "P14": p14}


def unit_table(units: list[Unit]) -> list[str]:
    lines = ["| Unit | Batch | ndcg@5 baseline | int8 (lower) | int4 | int4(mean) (lower) | binary | merge 0.25 | merge 0.1 | int4 statistic |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for u in units:
        if u.state != "ran":
            lines.append(f"| {u.name} | {u.batch} | {u.state} | | | | | | | |")
            continue
        r, g = u.result, geometry(u.batch, u.model, u.key)
        lines.append(f"| {u.name} | {u.batch} | {r['baseline']['ndcg@5']:.3f} | {ret(r, INT8):.3f} ({lower(r, INT8):.3f}) "
                     f"| {ret(r, INT4):.3f} | {ret(r, INT4M):.3f} ({lower(r, INT4M):.3f}) | {ret(r, BINARY):.3f} "
                     f"| {ret(r, HM25):.3f} | {ret(r, HM10):.3f} | {'' if g is None else f'{get(g, STAT):.3f}'} |")
    return lines


def main() -> None:
    status = {}
    for key, fn in PREDICTIONS.items():
        state, detail = fn(new_dataset_units())
        status[key] = {"status": state, "detail": detail}
    batches = {b: batch_done(b) for b in [*SELECTION_BATCHES, "f3b_answerai_scifact", "f8_geometry"]}
    (FOLLOWUP / "predictions_status.json").write_text(
        json.dumps({"batches_in": batches, "predictions": status}, indent=1) + "\n", encoding="utf-8")
    lines = ["# Follow-up predictions: status", "",
             "Written by `scripts/followup_predictions.py` from the ingested follow-up results; do not edit by hand.",
             "The predictions and their thresholds are in `reports/universal/campaign_analysis/PREDICTIONS.md`.",
             "", "Batches in: " + ", ".join(f"`{b}`" for b, done in batches.items() if done) + ".",
             "Still out: " + (", ".join(f"`{b}`" for b, done in batches.items() if not done) or "none") + ".",
             "", "| Prediction | Status | Detail |", "|---|---|---|"]
    lines += [f"| {k} | {v['status']} | {v['detail']} |" for k, v in status.items()]
    lines += ["", "## New-dataset units", "",
              "Retention of ndcg@5 against the same model's float32 run; the lower end of the 95% interval in brackets.",
              "The int4 statistic is the median per-query-token Pearson correlation of plain int4 scores.", ""]
    lines += unit_table(new_dataset_units())
    (FOLLOWUP / "PREDICTIONS_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in status.items():
        print(f"{k:4} {v['status']:13} {v['detail']}")


if __name__ == "__main__":
    main()
