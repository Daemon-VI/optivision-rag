"""Build ``src/optivision/presets/easy_levels.json``, the data of easy mode's selection rule.

FINAL_DESIGN 5.1. The file is generated, never edited by hand, and holds the sha256 (LF
line endings) of every file it was built from:

- ``presets/measured.json``: the catalogue rows of every measured model, matched to the ladder
  by their ``pipeline`` dict (so the pre-campaign labels ``none > int8/vec`` or
  ``ward ratio=0.25 > int8/vec`` join their rung without a label mapping);
- the follow-up reports ``reports/universal/followup/<batch>/<org>--<name>/{wide,int4norm}/``:
  ArxivQA, TabFQuAD, NFCorpus and SciDocs rows, which only lower a rung's lowest kept value
  (FINAL_DESIGN 5.1.2 step 1). Collection-size units (``*_scale_*`` batches, ``*_plus*`` files)
  are not evidence for the rule; they set its size coefficients below. The s17 (``int4norm/``)
  reports repeat the s13 rows of their units exactly, and add units s13 never finished;
- campaign-format replacements of the pre-campaign catalogue entries: batch B0 (``f14a``,
  ``f14b``) re-measures the three old-format page models, ColQwen2.5 v0.2's DocVQA row is
  ``f10``'s and answerai's SciFact row ``f3b``'s. A replacement is used as soon as its report is
  present, and the legacy entry of that dataset is dropped. Until every dataset of a B0 model is
  replaced, the model is marked ``pending_b0`` and the legacy merge rows of the two Qwen2-grid
  models are excluded (their merge kept 195.4 vectors against the campaign's 187.4 on the same
  748.3-vector grid; ColPali v1.3 merged matches at 263.0, so its rows stay);
- ``reports/universal/campaign``: the dtype, GPU count and encode seconds per page of each
  campaign model (``run.json``, the encode logs, the batch ``gpu.txt``), for ``options()``.

    python scripts/make_easy_levels.py            # rewrite the file
    python scripts/make_easy_levels.py --check    # exit 1 if it is out of date

``tests/test_easy_rule.py`` runs the check.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optivision.adapters import MODELS  # noqa: E402
from optivision.core.identity import canonical_json  # noqa: E402
from optivision.easy._levels import LADDER  # noqa: E402

REPORTS = ROOT / "reports" / "universal"
FOLLOWUP = REPORTS / "followup"
CAMPAIGN = REPORTS / "campaign"
CATALOG = ROOT / "src" / "optivision" / "presets" / "measured.json"
OUT = ROOT / "src" / "optivision" / "presets" / "easy_levels.json"

#: dataset id -> (benchmark family, label, unit); page counts come from the reports
DATASETS = {
    "vidore/docvqa_test_subsampled": ("ViDoRe V1", "DocVQA", "pages"),
    "vidore/infovqa_test_subsampled": ("ViDoRe V1", "InfoVQA", "pages"),
    "vidore/arxivqa_test_subsampled": ("ViDoRe V1", "ArxivQA", "pages"),
    "vidore/tabfquad_test_subsampled": ("ViDoRe V1", "TabFQuAD", "pages"),
    "BeIR/scifact": ("BEIR", "SciFact", "documents"),
    "BeIR/nfcorpus": ("BEIR", "NFCorpus", "documents"),
    "BeIR/scidocs": ("BEIR", "SciDocs", "documents"),
}
#: follow-up file-stem tail -> dataset id (the follow-up datasets of FINAL_DESIGN 5.1.2 step 1)
FOLLOWUP_DATASETS = {
    "arxivqa_test_subsampled": "vidore/arxivqa_test_subsampled",
    "tabfquad_test_subsampled": "vidore/tabfquad_test_subsampled",
    "nfcorpus": "BeIR/nfcorpus",
    "scidocs": "BeIR/scidocs",
}
METRIC = {"page": "ndcg@5", "text": "ndcg@10"}

#: the three page models whose catalogue rows predate the campaign; batch B0 re-measures them
B0_MODELS = ("vidore/colqwen2.5-v0.2", "vidore/colqwen2-v1.0-merged", "vidore/colpali-v1.3-merged")
#: the follow-up batches whose campaign-format reports replace a pre-campaign catalogue entry, in order
#: of preference: B0 itself, then ColQwen2.5 v0.2's DocVQA run (f10) and answerai's SciFact run (f3b)
REPLACEMENT_BATCHES = ("f14a_easy_b0_colqwen", "f14b_easy_b0_colpali", "f10_int4norm_pages_128d",
                       "f3b_answerai_scifact")
#: of those, the two whose legacy merge kept other vector counts than the campaign's merge
QWEN2_GRID_LEGACY = ("vidore/colqwen2.5-v0.2", "vidore/colqwen2-v1.0-merged")
LEGACY_PREFIX = "reports/universal/frontier/"

#: The rule's constants (FINAL_DESIGN 5.1.2). The size coefficient of a rung class is the
#: largest drop of the interval low from 500 to 2,500 pages over the two size units f6/f7
#: (codecs 0.0029, merge > codec 0.0151, binary 0.0167, merge > binary 0.0386), rounded up to
#: 0.0025. The ledger entry "margins" holds what a preregistered outcome may change.
RULE = {
    "size_coefficients": {"float": 0.0, "codec": 0.005, "merge": 0.0175, "binary": 0.0175, "merge-binary": 0.04},
    "size_base_pages": 500,
    "size_log_base": 5,
    "largest_measured_pages": 2500,
    "largest_measured_source": "ViDoRe V1 DocVQA 500 pages with 2,000 pages from other datasets added (follow-ups f6, f7)",
    "text_max_source": "BEIR SciFact 5,183 documents with 10,000 TREC-COVID documents added (follow-up f13b)",
    "text_margin": 0.01,
    "text_max_documents": 15183,
    "close_call": 0.005,
    "shape_fraction": 0.7,
    "default_target": 0.97,
    "less_tested_below": 0.95,
    "too_close_above": 0.99,
}

NOTE = ("Generated by scripts/make_easy_levels.py from the measured catalogue and the follow-up reports; "
        "do not edit. In-sample development measurements on public benchmarks: retention of the "
        "full-precision nDCG with the report's 95% bootstrap interval (low, point). They are not a "
        "statement about another collection.")


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _rel(path: Path) -> str:
    """A source path as the repository names it (``reports/universal/followup/...``)."""
    p = path.resolve()
    for base, name in ((FOLLOWUP, "reports/universal/followup"), (ROOT, "")):
        try:
            rest = p.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
        return f"{name}/{rest}" if name else rest
    return p.as_posix()


def _key(spec: Any) -> bytes:
    spec = ast.literal_eval(spec) if isinstance(spec, str) else spec
    return canonical_json(spec)


def _modality(model_id: str) -> str:
    return "text" if MODELS[model_id].modality == "text" else "page"


def _batch_order(path: Path) -> tuple[int, str]:
    """``f14a_easy_b0_colqwen`` -> (14, "a"): batches compare by number, then by letter."""
    batch = path.relative_to(FOLLOWUP).parts[0]
    m = re.match(r"f(\d+)([a-z]*)_", batch)
    return (int(m.group(1)), m.group(2)) if m else (10**6, batch)


def _usable(path: Path) -> bool:
    batch = path.relative_to(FOLLOWUP).parts[0]
    return "_scale" not in batch and "plus" not in path.stem


def _record(row: dict[str, Any], f16_bytes: float, metric: str, report: bool) -> dict[str, Any]:
    """One rung on one dataset; a report row is rounded as the catalogue rounds it."""
    if report:
        lo = float(row[f"retention_ci:{metric}"][0])
        point = float(row[f"retention:{metric}"])
        ratio_f32 = float(row.get("compression_vs_float32", row.get("compression")))
        nbytes = round(float(row["bytes_per_doc"]), 1)
        vpp = round(float(row["vectors_per_doc"]), 2)
        lo, point, ratio_f32 = round(lo, 4), round(point, 4), round(ratio_f32, 3)
    else:
        lo, point = row["retention_interval"][0], row["retention"]
        ratio_f32, nbytes, vpp = row["compression"], row["bytes_per_doc"], row["vectors_per_doc"]
    return {"bytes_per_page": nbytes, "ratio_f16": round(f16_bytes / nbytes, 4), "ratio_f32": ratio_f32,
            "low": lo, "point": point, "vectors_per_page": vpp}


def _report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for r in report["rows"]:
        rows.append({**r, "pipeline": ast.literal_eval(r["pipeline"]) if isinstance(r["pipeline"], str) else r["pipeline"]})
    return rows


def _rung_records(rows: list[dict[str, Any]], modality: str, metric: str, report: bool,
                  skip_merges: bool = False) -> tuple[dict[str, dict[str, Any]], float]:
    """Ladder rung -> record for one dataset's rows, and the float16 row's vectors per page."""
    by_spec = {_key(r["pipeline"]): r for r in rows}
    f16 = by_spec[_key(LADDER[modality][1].spec)]
    out = {}
    for rung in LADDER[modality]:
        if skip_merges and rung.merged:
            continue
        r = by_spec.get(_key(rung.spec))
        if r is None or (report and f"retention:{metric}" not in r):
            continue
        out[rung.name] = _record(r, float(f16["bytes_per_doc"]), metric, report)
    return out, float(f16["vectors_per_doc"])


def _followup_reports(model_id: str, tails: dict[str, str], folders: tuple[str, ...] = ("wide", "int4norm")) -> dict[str, list[Path]]:
    """dataset id -> usable report files of ``model_id`` whose stem ends in one of ``tails``."""
    folder, short = model_id.replace("/", "--"), model_id.split("/")[-1]
    found: dict[str, list[Path]] = {}
    for sub in folders:
        for path in FOLLOWUP.glob(f"*/{folder}/{sub}/{short}_*.json"):
            tail = path.stem[len(short) + 1:]
            if tail in tails and _usable(path):
                found.setdefault(tails[tail], []).append(path)
    return {d: sorted(ps, key=lambda p: (_batch_order(p), p.as_posix())) for d, ps in found.items()}


def build() -> dict[str, Any]:
    sources: dict[str, str] = {_rel(CATALOG): _sha256_lf(CATALOG)}
    catalogue = json.loads(CATALOG.read_text(encoding="utf-8"))["models"]
    datasets: dict[str, dict[str, Any]] = {}

    def note_dataset(name: str, n_docs: int) -> None:
        family, label, unit = DATASETS[name]
        seen = datasets.setdefault(name, {"family": family, "label": label, "unit": unit, "pages": int(n_docs)})
        if seen["pages"] != int(n_docs):
            raise ValueError(f"{name}: page counts differ across reports ({seen['pages']} and {n_docs})")

    models: dict[str, Any] = {}
    for model_id in sorted(catalogue):
        modality = _modality(model_id)
        metric = METRIC[modality]
        entry: dict[str, Any] = {"modality": modality, "datasets": {}, "rungs": {}}
        legacy_left = []
        for name, unit in sorted(catalogue[model_id].items()):
            tail = name.split("/")[-1]
            replacement = None
            if unit["source"].startswith(LEGACY_PREFIX):
                found = _followup_reports(model_id, {tail: name}, ("wide",)).get(name, [])
                ranked = [(REPLACEMENT_BATCHES.index(b), p) for p in found
                          if (b := p.relative_to(FOLLOWUP).parts[0]) in REPLACEMENT_BATCHES]
                replacement = min(ranked)[1] if ranked else None
            if replacement is not None:
                report = json.loads(replacement.read_text(encoding="utf-8"))
                sources[_rel(replacement)] = _sha256_lf(replacement)
                recs, vpp = _rung_records(_report_rows(report), modality, unit["metric"], True)
                info = {"role": "catalogue", "pages": int(report["n_docs"]), "queries": int(report["n_queries"]),
                        "metric": unit["metric"], "baseline": round(float(report["baseline"][unit["metric"]]), 4),
                        "vectors_per_page": round(vpp, 2), "sources": [_rel(replacement)],
                        "replaces": unit["source"]}
            else:
                legacy = unit["source"].startswith(LEGACY_PREFIX)
                if legacy:
                    legacy_left.append(name)
                skip = legacy and model_id in QWEN2_GRID_LEGACY
                recs, vpp = _rung_records(unit["rows"], modality, unit["metric"], False, skip_merges=skip)
                info = {"role": "catalogue", "pages": int(unit["n_docs"]), "queries": int(unit["n_queries"]),
                        "metric": unit["metric"], "baseline": unit["baseline"], "vectors_per_page": vpp,
                        "sources": [unit["source"]]}
                if legacy:
                    info["legacy"] = True
                if skip:
                    info["excluded"] = "legacy merge rows (another merge than the campaign's; B0 re-measures them)"
            note_dataset(name, info["pages"])
            entry["datasets"][name] = info
            for rung, rec in recs.items():
                entry["rungs"].setdefault(rung, {})[name] = rec

        for name, paths in sorted(_followup_reports(model_id, FOLLOWUP_DATASETS).items()):
            best: dict[str, tuple[dict[str, Any], str]] = {}
            info = None
            for path in paths:
                report = json.loads(path.read_text(encoding="utf-8"))
                rows = _report_rows(report)
                if not any(f"retention:{metric}" in r for r in rows):
                    continue
                sources[_rel(path)] = _sha256_lf(path)
                recs, vpp = _rung_records(rows, modality, metric, True)
                if info is None:
                    info = {"role": "follow-up", "pages": int(report["n_docs"]), "queries": int(report["n_queries"]),
                            "metric": metric, "baseline": round(float(report["baseline"][metric]), 4),
                            "vectors_per_page": round(vpp, 2), "sources": []}
                info["sources"].append(_rel(path))
                for rung, rec in recs.items():  # duplicate units reproduce; keep the lowest low
                    if rung not in best or rec["low"] < best[rung][0]["low"]:
                        best[rung] = (rec, _rel(path))
            if info is None or not best:
                continue
            note_dataset(name, info["pages"])
            entry["datasets"][name] = info
            for rung, (rec, _) in best.items():
                entry["rungs"].setdefault(rung, {})[name] = rec

        if model_id in B0_MODELS:
            entry["pending_b0"] = bool(legacy_left)
        if legacy_left:
            entry["legacy_datasets"] = legacy_left
        models[model_id] = entry

    return {"schema": 1, "note": NOTE, "rule": RULE, "datasets": dict(sorted(datasets.items())),
            "models": models, "pooled": _pooled(models), "runs": _runs(sources), "sources": dict(sorted(sources.items()))}


def _lowest(entry: dict[str, Any], rung: str) -> tuple[float, float, str, float, float] | None:
    """(L, worst point, dataset of L, smallest ratio_f16, smallest ratio_f32) of a measured rung."""
    recs = entry["rungs"].get(rung, {})
    cat = [d for d, v in entry["datasets"].items() if v["role"] == "catalogue"]
    if not recs or any(d not in recs for d in cat):
        return None
    worst = min(recs, key=lambda d: (recs[d]["low"], d))
    return (recs[worst]["low"], min(r["point"] for r in recs.values()), worst,
            min(recs[d]["ratio_f16"] for d in cat), min(recs[d]["ratio_f32"] for d in cat))


def _pooled(models: dict[str, Any]) -> dict[str, Any]:
    """FINAL_DESIGN 5.1.4: the smallest L over every measured model of a modality, per codec rung."""
    out = {}
    for modality in ("page", "text"):
        members = sorted(m for m, e in models.items() if e["modality"] == modality)
        names = sorted({d for m in members for d in models[m]["datasets"]},
                       key=lambda d: (list(DATASETS).index(d), d))
        rungs = {}
        for rung in ("float32", "float16", "int8(per_vector)", "int4(mean)"):
            vals = [(v, m) for m in members if (v := _lowest(models[m], rung)) is not None]
            if not vals:
                continue
            (low, _, dset, _, _), who = min(vals, key=lambda x: (x[0][0], x[1]))
            rungs[rung] = {"low": low, "point": min(v[1] for v, _ in vals), "worst_model": who, "worst_dataset": dset,
                           "n_models": len(vals), "ratio_f16": min(v[3] for v, _ in vals),
                           "ratio_f32": min(v[4] for v, _ in vals)}
        out[modality] = {"n_models": len(members), "datasets": names, "rungs": rungs}
    return out


def _runs(sources: dict[str, str]) -> dict[str, Any]:
    """Per campaign model: inference dtype, whether it was split over two GPUs, the GPU, and the
    encode step's seconds per page (or document) on each dataset, from the encode logs."""
    path = CAMPAIGN / "outcomes.json"
    if not path.exists():
        return {}
    sources[_rel(path)] = _sha256_lf(path)
    out = {}
    for model_id, rec in sorted(json.loads(path.read_text(encoding="utf-8")).items()):
        folder = CAMPAIGN / rec["folder"]
        run_path = folder / "run.json"
        if rec["outcome"] != "measured" or not run_path.exists() or model_id not in MODELS:
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("model") != model_id:
            continue
        sources[_rel(run_path)] = _sha256_lf(run_path)
        spec = run.get("spec", {})
        dtype = spec.get("dtype")
        per_unit = []
        for log in sorted((folder / "logs").glob("encode_*.log")):
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if '"doc_seconds"' in line:
                    j = json.loads(line[line.index("{"):])
                    per_unit.append(j["doc_seconds"] / j["docs"])
                    sources[_rel(log)] = _sha256_lf(log)
        gpu = None
        gpu_path = CAMPAIGN / "batches" / rec["batch"] / "gpu.txt"
        if gpu_path.exists():
            lines = [ln.split(",")[0].strip() for ln in gpu_path.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
            if lines:
                gpu = lines[0].replace("Tesla ", "")
                sources[_rel(gpu_path)] = _sha256_lf(gpu_path)
        out[model_id] = {
            "dtype": dtype if dtype in ("float16", "float32", "bfloat16") else None,
            "multi_gpu": bool(spec.get("multi_gpu", rec.get("multi_gpu", False))),
            "gpu": gpu,
            "unit": "page" if rec.get("kind") == "image" else "document",
            "seconds_per_unit": [round(min(per_unit), 4), round(max(per_unit), 4)] if per_unit else None,
            "source": _rel(run_path),
        }
    return out


def render(levels: dict[str, Any]) -> str:
    return json.dumps(levels, separators=(",", ":"), sort_keys=True) + "\n"


# ---------------------------------------------------------------------------- the golden fixture

GOLDEN = ROOT / "tests" / "data" / "easy_rule_golden.json"
GOLDEN_TARGETS = (0.98, 0.97, 0.95, 0.90)
GOLDEN_SIZES = (500, 1000, 2500, 5244, 20000)


def golden(levels: dict[str, Any]) -> dict[str, Any]:
    """The rule's pick for every catalogue model, target, size and path (checked or not), and the
    medians of FINAL_DESIGN 5.1.3, computed by ``_levels.select`` on ``levels``."""
    import statistics

    from optivision.easy import _levels

    saved = _levels.data
    _levels.data = lambda: levels  # the rule reads this build, not the installed file
    try:
        def pick(model: str | None, t: float, n: int, checked: bool, modality: str | None = None) -> list[Any]:
            s = _levels.select(model, t, n, checked=checked, modality=modality)
            return [s.pick.name, round(s.pick.ratio_f16, 4), None if s.pick.slack is None else round(s.pick.slack, 4),
                    s.close_call, None if s.safer is None else s.safer.name]

        picks = {}
        for model in sorted(levels["models"]):
            picks[model] = {f"{t:.2f}|{n}|{'c' if c else 'u'}": pick(model, t, n, c)
                            for t in GOLDEN_TARGETS for n in GOLDEN_SIZES for c in (True, False)}
        pooled = {mod: {f"{t:.2f}|{n}|{'c' if c else 'u'}": pick(None, t, n, c, mod)
                        for t in (0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90) for n in (500, 2500, 7000, 7500, 20000)
                        for c in (True, False)} for mod in ("page", "text")}
        page = [m for m, e in levels["models"].items() if e["modality"] == "page"]
        summary: dict[str, Any] = {}
        for path in ("c", "u"):
            for t in GOLDEN_TARGETS:
                for n in GOLDEN_SIZES:
                    xs = sorted(picks[m][f"{t:.2f}|{n}|{path}"][1] for m in page)
                    q = statistics.quantiles(xs, n=4)
                    summary[f"{t:.2f}|{n}|{path}"] = {
                        "median": round(statistics.median(xs), 4), "iqr": [round(q[0], 4), round(q[2], 4)],
                        "close_calls": sum(picks[m][f"{t:.2f}|{n}|{path}"][3] for m in page),
                        "one_bit": sum("binary" in picks[m][f"{t:.2f}|{n}|{path}"][0] for m in page)}
    finally:
        _levels.data = saved
    return {
        "note": "Generated by scripts/make_easy_levels.py --golden; tests/test_easy_rule.py checks the rule against it. "
                "Each pick is [layout, smallest ratio against float16, slack, close call, safer layout]; "
                "keys are target|pages|path (c = checked, u = unchecked).",
        "levels_sha256": hashlib.sha256(render(levels).encode("utf-8")).hexdigest(),
        "targets": list(GOLDEN_TARGETS),
        "sizes": list(GOLDEN_SIZES),
        "page_models": len(page),
        "pending_b0": sorted(m for m, e in levels["models"].items() if e.get("pending_b0")),
        "picks": picks,
        "pooled": pooled,
        "summary": summary,
    }


def render_golden(fixture: dict[str, Any]) -> str:
    """Sorted JSON with one line per model (and per summary cell), so a rule change shows as a
    readable diff."""
    def one(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    out = []
    for key in sorted(fixture):
        value = fixture[key]
        if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            inner = ",\n".join(f"  {json.dumps(k)}: {one(v)}" for k, v in sorted(value.items()))
            out.append(f" {json.dumps(key)}: {{\n{inner}\n }}")
        else:
            out.append(f" {json.dumps(key)}: {one(value)}")
    return "{\n" + ",\n".join(out) + "\n}\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed files instead of writing")
    ap.add_argument("--golden", action="store_true", help="also rewrite tests/data/easy_rule_golden.json")
    a = ap.parse_args(argv)
    levels = build()
    text = render(levels)
    if a.check:
        stale = []
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current.replace("\r\n", "\n") != text:
            stale.append(f"{_rel(OUT)} is out of date; run scripts/make_easy_levels.py")
        elif GOLDEN.exists() and GOLDEN.read_text(encoding="utf-8").replace("\r\n", "\n") != render_golden(golden(levels)):
            stale.append(f"{_rel(GOLDEN)} is out of date; run scripts/make_easy_levels.py --golden")
        if stale:
            print("\n".join(stale))
            return 1
        print("easy_levels.json is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    pending = [m for m, e in levels["models"].items() if e.get("pending_b0")]
    print(f"wrote {_rel(OUT)}: {len(levels['models'])} models, {len(levels['sources'])} source files, "
          f"{len(text):,} bytes; pending B0: {', '.join(pending) or 'none'}")
    if a.golden:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(render_golden(golden(levels)), encoding="utf-8", newline="\n")
        print(f"wrote {_rel(GOLDEN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
