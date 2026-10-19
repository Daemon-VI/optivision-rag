"""Score the int4 norm batches against the predictions written before they ran.

    python scripts/int4norm_predictions.py

The predictions are Q1 to Q10 in ``reports/universal/followup/PREDICTIONS_INT4NORM.md``; the
thresholds below are copied from there and are not to be changed here. Q1 to Q9 read the s17
results (``int4norm/``) of the rerun batches ``f9b`` to ``f12b``, which stand for the units the
predictions call ``f9`` to ``f12`` (``INT4NORM_RUN_NOTES.md``). Q10 reads the s13 results
(``wide/``) of ``f13`` against SciFact alone. Writes ``PREDICTIONS_INT4NORM_STATUS.md`` and
``int4norm_predictions_status.json`` next to the predictions.

Statuses are those of ``scripts/followup_predictions.py``: ``held``, ``failed``, ``not testable``
(a unit it needs did not run) and ``pending`` (a batch it needs has not come back).
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("followup_predictions", ROOT / "scripts" / "followup_predictions.py")
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)
FOLLOWUP, folder, short, read, batch_done = fp.FOLLOWUP, fp.folder, fp.short, fp.read, fp.batch_done
each, pooled, combine, spearman, ret, lower = fp.each, fp.pooled, fp.combine, fp.spearman, fp.ret, fp.lower

TEXT = ["answerdotai/answerai-colbert-small-v1", "lightonai/LateOn", "lightonai/mLateOn",
        "mixedbread-ai/mxbai-edge-colbert-v0-17m", "mixedbread-ai/mxbai-edge-colbert-v0-32m",
        "lightonai/GTE-ModernColBERT-v1", "LiquidAI/LFM2-ColBERT-350M", "colbert-ir/colbertv2.0",
        "lightonai/ColBERT-Zero", "lightonai/Reason-ModernColBERT", "mixedbread-ai/mxbai-colbert-large-v1",
        "jinaai/jina-colbert-v2"]
PAGES_128 = ["vidore/colpali-v1.3", "vidore/colSmol-500M", "vidore/colqwen2.5-v0.2"]
PAGES_WIDE = ["nvidia/nemotron-colembed-vl-4b-v2", "nvidia/nemotron-colembed-vl-8b-v2", "tencent/EVIE-4.5B",
              "vultr/VultronRetrieverCore-Qwen3.5-4.5B"]
SCIDOCS = ["answerdotai/answerai-colbert-small-v1", "lightonai/LateOn", "lightonai/mLateOn",
           "mixedbread-ai/mxbai-edge-colbert-v0-17m", "mixedbread-ai/mxbai-edge-colbert-v0-32m",
           "lightonai/GTE-ModernColBERT-v1", "colbert-ir/colbertv2.0", "jinaai/jina-colbert-v2"]
NORM_NOISE = ["answerdotai/answerai-colbert-small-v1", "lightonai/mLateOn", "mixedbread-ai/mxbai-edge-colbert-v0-17m",
              "mixedbread-ai/mxbai-edge-colbert-v0-32m", "lightonai/GTE-ModernColBERT-v1"]
NEMOTRON = PAGES_WIDE[:2]
#: sm4 above 0.15 in the SciFact geometry files (PREDICTIONS_INT4NORM.md, Definitions)
FLAGGED = {"answerdotai/answerai-colbert-small-v1", "lightonai/LateOn", "lightonai/mLateOn",
           "mixedbread-ai/mxbai-edge-colbert-v0-17m", "mixedbread-ai/mxbai-edge-colbert-v0-32m",
           "lightonai/GTE-ModernColBERT-v1"}
PAGE_SETS = ["docvqa_test_subsampled", "tabfquad_test_subsampled"]

#: s17 labels; PREDICTIONS_INT4NORM.md calls the last two codecs int4(norm) and int4(mean, norm)
INT8, INT4, INT4N, INT4M, INT4MN = "int8(per_vector)", "int4", "int4norm", "int4(mean)", "int4norm(mean)"


class Unit(fp.Unit):
    """One model on one dataset, from its s17 file."""

    def __init__(self, model: str, key: str, batch: str) -> None:
        self.model, self.key, self.batch = model, key, batch
        self.result = read(FOLLOWUP / batch / folder(model) / "int4norm" / f"{short(model)}_{key}.json")
        self.state = "ran" if self.result else ("not run" if batch_done(batch) else "pending")


def text_units() -> list[Unit]:
    return [Unit(m, k, "f9b_int4norm_text") for m in TEXT for k in ("scifact", "nfcorpus")]


def page_units() -> list[Unit]:
    return ([Unit(m, k, "f10b_int4norm_pages_128d") for m in PAGES_128 for k in PAGE_SETS]
            + [Unit(m, k, "f11b_int4norm_pages_wide") for m in PAGES_WIDE for k in PAGE_SETS])


def scidocs_units() -> list[Unit]:
    return [Unit(m, "scidocs", "f12b_text_scidocs") for m in SCIDOCS]


def pick(units: list[Unit], models: list[str], keys: list[str] | None = None) -> list[Unit]:
    return [u for u in units if u.model in models and (keys is None or u.key in keys)]


def q1():
    t = text_units()
    answerai = ["answerdotai/answerai-colbert-small-v1"]
    return combine([each(pick(t, answerai, ["nfcorpus"]), lambda r: ret(r, INT4) < 0.6 and ret(r, INT4N) >= 0.95),
                    each(pick(t, answerai, ["scifact"]), lambda r: ret(r, INT4N) >= 0.98)])


def q2():
    return each(pick(text_units(), NORM_NOISE), lambda r: ret(r, INT4N) >= 0.95, need=8)


def q3():
    return each(pick(text_units(), ["lightonai/LateOn"]), lambda r: ret(r, INT4N) < 0.9 and ret(r, INT4MN) >= 0.95)


def q4():
    return each(text_units(), lambda r: lower(r, INT4MN) >= 0.95, need=23)


def q5():
    def rho(ran):
        return round(spearman([r.result["norms"]["int4"]["ratio_sd"] for r in ran], [ret(r.result, INT4) for r in ran]), 4)

    return pooled(text_units(), rho, lambda v: v <= -0.5, "Spearman(int4 norm-ratio sd, plain int4 retention)")


def q6():
    units = text_units() + page_units() + scidocs_units()
    return each(units, lambda r: ret(r, INT4N) >= ret(r, INT4) - 0.01 and ret(r, INT4MN) >= ret(r, INT4M) - 0.01)


def q7():
    return each(pick(page_units(), NEMOTRON),
                lambda r: ret(r, INT4) < 0.5 and ret(r, INT4N) < 0.5 and ret(r, INT4MN) >= 0.97)


def q8():
    return each(page_units(), lambda r: lower(r, INT4MN) >= 0.95)


def q9():
    s = scidocs_units()
    flag = {u.result["dataset"]: u.model in FLAGGED for u in s if u.state == "ran"}
    return combine([each(s, lambda r: lower(r, INT8) >= 0.95),
                    each(s, lambda r: flag[r["dataset"]] or ret(r, INT4) >= 0.90),  # int4 < 0.90 only when flagged
                    each(s, lambda r: lower(r, INT4MN) >= 0.95, need=7)])


#: model -> (where SciFact alone is, f13's result) for Q10, both s13 (wide/)
Q10_PAIRS = {"colbert-ir/colbertv2.0": "campaign", "answerdotai/answerai-colbert-small-v1": "f3b_answerai_scifact"}


def q10_values(alone: dict, big: dict) -> tuple[float, float, float]:
    dim = alone["baseline"]["dim"]
    a = {r["label"]: r for r in alone["rows"]}
    b = {r["label"]: r for r in big["rows"]}
    common = [k for k in a if k in b]

    def projected(label: str) -> bool:
        ds = [int(x) for x in re.findall(r"project\((\d+)\)", label)]
        return bool(ds) and "random" not in label and min(ds) <= dim // 4

    proj = [b[k]["retention:ndcg@5"] - a[k]["retention:ndcg@5"] for k in common if projected(k)]
    codec = [b[k]["retention:ndcg@5"] - a[k]["retention:ndcg@5"] for k in common if a[k]["group"] == "codec"]
    return alone["baseline"]["ndcg@5"] - big["baseline"]["ndcg@5"], mean(proj), mean(codec)


def q10():
    parts = []
    for model, where in Q10_PAIRS.items():
        big = fp.wide("f13_text_scale_same_domain", model, "scifact_plus10000")
        alone = fp.wide(where, model, "scifact")
        if big is None or alone is None:
            parts.append(("pending" if not batch_done("f13_text_scale_same_domain") else "not testable",
                          f"{short(model)}: no result"))
            continue
        drop, proj, codec = q10_values(alone, big)
        ok = drop >= 0.02 and proj <= -0.05 and codec >= -0.015
        detail = (f"{short(model)}: baseline drop {drop:.4f} (>= 0.02), projection rows {proj:+.4f} (<= -0.05), "
                  f"codec rows {codec:+.4f} (>= -0.015)")
        parts.append(("held" if ok else "failed", detail))
    return combine(parts)


PREDICTIONS = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6, "Q7": q7, "Q8": q8, "Q9": q9, "Q10": q10}
BATCHES = ["f9b_int4norm_text", "f10b_int4norm_pages_128d", "f11b_int4norm_pages_wide", "f12b_text_scidocs",
           "f13_text_scale_same_domain"]


def unit_table(units: list[Unit]) -> list[str]:
    lines = [("| Unit | Batch | ndcg@5 baseline | int8 (lower) | int4 | int4norm | int4(mean) | int4norm(mean) (lower) "
              "| int4 norm-ratio sd |"), "|---|---|---|---|---|---|---|---|---|"]
    for u in units:
        if u.state != "ran":
            lines.append(f"| {u.name} | {u.batch} | {u.state} | | | | | | |")
            continue
        r = u.result
        lines.append(f"| {u.name} | {u.batch} | {r['baseline']['ndcg@5']:.3f} | {ret(r, INT8):.3f} ({lower(r, INT8):.3f}) "
                     f"| {ret(r, INT4):.3f} | {ret(r, INT4N):.3f} | {ret(r, INT4M):.3f} "
                     f"| {ret(r, INT4MN):.3f} ({lower(r, INT4MN):.3f}) | {r['norms']['int4']['ratio_sd']:.4f} |")
    return lines


def main() -> None:
    status = {k: dict(zip(("status", "detail"), fn(), strict=True)) for k, fn in PREDICTIONS.items()}
    batches = {b: batch_done(b) for b in BATCHES}
    (FOLLOWUP / "int4norm_predictions_status.json").write_text(
        json.dumps({"batches_in": batches, "predictions": status}, indent=1) + "\n", encoding="utf-8")
    lines = ["# int4 norm predictions: status", "",
             "Written by `scripts/int4norm_predictions.py` from the ingested results; do not edit by hand.",
             ("The predictions and their thresholds are in `PREDICTIONS_INT4NORM.md`; the run history is in "
              "`INT4NORM_RUN_NOTES.md`."), "",
             "Batches in: " + (", ".join(f"`{b}`" for b, done in batches.items() if done) or "none") + ".",
             "Still out: " + (", ".join(f"`{b}`" for b, done in batches.items() if not done) or "none") + ".",
             "", "| Prediction | Status | Detail |", "|---|---|---|"]
    lines += [f"| {k} | {v['status']} | {v['detail']} |" for k, v in status.items()]
    lines += ["", "## Units (s17)", "",
              ("Retention of ndcg@5 against the same model's float32 run; the lower end of the 95% interval in "
               "brackets."), ""]
    lines += unit_table(text_units() + page_units() + scidocs_units())
    (FOLLOWUP / "PREDICTIONS_INT4NORM_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in status.items():
        print(f"{k:4} {v['status']:13} {v['detail']}")


if __name__ == "__main__":
    main()
