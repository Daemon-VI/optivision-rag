"""Build ``src/optivision/presets/measured.json`` from the committed study reports.

The catalogue is a copy, in one small file that ships in the wheel, of the fixed-pipeline
rows of ``reports/universal/frontier`` (128-d and text models) and ``reports/universal/wide``
(wide models and the ColQwen3 generation): for every measured model and dataset, each
pipeline that was run, its compression against float32 and its retention of the float
baseline with the bootstrap interval of the report. Nothing is computed here beyond rounding.

    python scripts/build_measured_catalog.py            # rewrite the catalogue
    python scripts/build_measured_catalog.py --check    # exit 1 if it is out of date

``tests/test_measured_catalog.py`` runs the check, so the shipped file cannot drift from
the reports.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "universal"
OUT = ROOT / "src" / "optivision" / "presets" / "measured.json"

DOCVQA = "vidore/docvqa_test_subsampled"
INFOVQA = "vidore/infovqa_test_subsampled"

#: (report file under reports/universal, model id, dataset, metric the retention is of, note)
SOURCES: list[tuple[str, str, str, str, str]] = [
    ("frontier/E2-colpali-docvqa.json", "vidore/colpali-v1.3-merged", DOCVQA, "ndcg@5", ""),
    ("frontier/E2-colpali-infovqa.json", "vidore/colpali-v1.3-merged", INFOVQA, "ndcg@5", ""),
    ("frontier/colqwen2-v1.0-merged_docvqa_test_subsampled.json", "vidore/colqwen2-v1.0-merged", DOCVQA, "ndcg@5", ""),
    ("frontier/colqwen2-v1.0-merged_infovqa_test_subsampled.json", "vidore/colqwen2-v1.0-merged", INFOVQA, "ndcg@5", ""),
    ("frontier/colqwen2.5-v0.2_docvqa_test_subsampled.json", "vidore/colqwen2.5-v0.2", DOCVQA, "ndcg@5", ""),
    ("frontier/colqwen2.5-v0.2_infovqa_test_subsampled.json", "vidore/colqwen2.5-v0.2", INFOVQA, "ndcg@5", ""),
    ("frontier/text-answerai-colbert-small-v1-scifact.json", "answerdotai/answerai-colbert-small-v1",
     "BeIR/scifact", "ndcg@10", ""),
    ("wide/nemotron-colembed-vl-4b-v2_docvqa_test_subsampled.json", "nvidia/nemotron-colembed-vl-4b-v2", DOCVQA, "ndcg@5", ""),
    ("wide/nemotron-colembed-vl-4b-v2_infovqa_test_subsampled.json", "nvidia/nemotron-colembed-vl-4b-v2", INFOVQA, "ndcg@5", ""),
    ("wide/nemotron-colembed-vl-8b-v2_docvqa_test_subsampled.json", "nvidia/nemotron-colembed-vl-8b-v2", DOCVQA, "ndcg@5",
     "float16 inference; the baseline is the float16-inference encoding (no float32 encoding exists)"),
    ("wide/nemotron-colembed-vl-8b-v2_infovqa_test_subsampled.json", "nvidia/nemotron-colembed-vl-8b-v2", INFOVQA, "ndcg@5",
     "float16 inference; the baseline is the float16-inference encoding (no float32 encoding exists)"),
    ("wide/colqwen3.5-4.5B-v3_docvqa_test_subsampled.json", "athrael-soju/colqwen3.5-4.5B-v3", DOCVQA, "ndcg@5", ""),
    ("wide/colqwen3.5-4.5B-v3_infovqa_test_subsampled.json", "athrael-soju/colqwen3.5-4.5B-v3", INFOVQA, "ndcg@5", ""),
    ("wide/tomoro-colqwen3-embed-4b_docvqa_test_subsampled.json", "TomoroAI/tomoro-colqwen3-embed-4b", DOCVQA, "ndcg@5", ""),
    ("wide/tomoro-colqwen3-embed-4b_infovqa_test_subsampled.json", "TomoroAI/tomoro-colqwen3-embed-4b", INFOVQA, "ndcg@5", ""),
]

NOTE = ("Development measurements of fixed pipelines on the named dataset: retention is the "
        "pipeline's mean metric over the float baseline's, on all queries of that dataset, with "
        "the report's bootstrap interval. They are in-sample figures for these datasets, not a "
        "guarantee for another corpus or query set; use certify() on your own queries to decide.")


def _sha256_lf(path: Path) -> str:
    """Hash of the file with LF line endings, so a CRLF checkout gives the same value."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _pipeline(value: object) -> dict:
    spec = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(spec, dict) or "stages" not in spec:
        raise ValueError(f"not a pipeline specification: {value!r}")
    return spec


def _interval(value: object) -> list[float]:
    lo, hi = ast.literal_eval(value) if isinstance(value, str) else value
    return [round(float(lo), 4), round(float(hi), 4)]


def _rows(report: dict, metric: str) -> list[dict]:
    rows = []
    for r in report["rows"]:
        compression = r.get("compression_vs_float32", r.get("compression"))
        interval = r.get(f"retention_ci:{metric}", r.get(f"ci:{metric}"))
        rows.append({
            "label": r["label"],
            "pipeline": _pipeline(r["pipeline"]),
            "compression": round(float(compression), 3),
            "vectors_per_doc": round(float(r["vectors_per_doc"]), 2),
            "bytes_per_doc": round(float(r["bytes_per_doc"]), 1),
            "retention": round(float(r[f"retention:{metric}"]), 4),
            "retention_interval": _interval(interval),
        })
    # A report may give two pipelines one label (PCA and random projection to the same width
    # in the wide study). The catalogue keys rows by label, so the method is added to both.
    counts = Counter(r["label"] for r in rows)
    for r in rows:
        if counts[r["label"]] > 1:
            methods = "+".join(str(s["method"]) for s in r["pipeline"]["stages"] if "method" in s)
            r["label"] = f"{r['label']} [{methods}]"
    if len({r["label"] for r in rows}) != len(rows):
        raise ValueError(f"labels still repeat in {report['dataset']}")
    return rows


def build() -> dict:
    models: dict[str, dict] = {}
    sources = {}
    for rel, model_id, dataset, metric, note in SOURCES:
        path = REPORTS / rel
        report = json.loads(path.read_text(encoding="utf-8"))
        sources[f"reports/universal/{rel}"] = _sha256_lf(path)
        entry = {
            "source": f"reports/universal/{rel}",
            "n_docs": int(report["n_docs"]),
            "n_queries": int(report["n_queries"]),
            "metric": metric,
            "baseline": round(float(report["baseline"][metric]), 4),
            "rows": _rows(report, metric),
        }
        if note:
            entry["note"] = note
        models.setdefault(model_id, {})[dataset] = entry
    return {"schema": 1, "note": NOTE, "sources": sources, "models": models}


def render(catalog: dict) -> str:
    return json.dumps(catalog, separators=(",", ":"), sort_keys=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed file instead of writing")
    a = ap.parse_args()
    text = render(build())
    if a.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current.replace("\r\n", "\n") != text:
            print(f"{OUT.relative_to(ROOT)} is out of date; run scripts/build_measured_catalog.py")
            return 1
        print("catalogue is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    n_rows = sum(len(d["rows"]) for m in json.loads(text)["models"].values() for d in m.values())
    print(f"wrote {OUT.relative_to(ROOT)}: {len(json.loads(text)['models'])} models, {n_rows} rows, {len(text)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
