"""Score every preregistered prediction from the ingested units and propose the ledger change.

    PYTHONPATH=src python scripts/easy_validation/run_all.py --data data/easy --out reports/easy/results
    PYTHONPATH=src python scripts/easy_validation/run_all.py --synthetic --out <tmp>   # dry run, fake backend

Reads ``data/easy/<batch>/`` (``ingest.py``), scores V1a-V6, S and arm S (``predictions.py``),
applies the decision rules (``decide.py``) and writes, under ``--out``:

    predictions.json      every result with its unit rows
    decisions.json        the decision rules that fired and the proposed settings
    ledger_proposal.json  the change to presets/easy_ledger.json, for a person to review and apply
    SUMMARY.md            one line per prediction

The ledger itself is never written here. ``--synthetic`` runs the whole programme on a tiny
generated tree with ``fakes.FakeBackend``; its numbers mean nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))

from easy_validation import api, decide, predictions  # noqa: E402
from easy_validation.data import batch_roots, heldout_models  # noqa: E402
from easy_validation.evaluate import Context  # noqa: E402


def _fmt(v: object) -> str:
    return {True: "pass", False: "fail", None: "not scored"}.get(v, str(v))  # type: ignore[arg-type]


def summary_md(results: dict, decisions: dict, backend: str, roots: list[Path]) -> str:
    lines = ["# Easy-mode validation: results", "",
             f"Written by `scripts/easy_validation/run_all.py` on {time.strftime('%Y-%m-%d')} with backend "
             f"`{backend}` from {len(roots)} batch folder(s). Thresholds and decision rules: "
             "reports/easy/PREREG_EASY.md (part 1) and PREREG_EASY_PART2.md. Rows marked in-sample are fitted on "
             "the units they are reported on.", "",
             "| id | prediction | pass if | result | detail |", "|---|---|---|---|---|"]
    for k, r in results.items():
        tag = " (in-sample)" if r.get("in_sample") else ""
        lines.append(f"| {k} | {r['prediction']} | {r['pass_if']} | {_fmt(r['passed'])}{tag} | {r['reason']} |")
    lines += ["", "## Decision rules that fired (proposed, not applied)", ""]
    if decisions["actions"]:
        for a in decisions["actions"]:
            lines.append(f"- {a['rule']}: {a['outcome']} `{json.dumps(a['set'])}` {a['note']}".rstrip())
    else:
        lines.append("- none")
    lines += ["", f"Status line this would give: \"{decisions['status']}\"", ""]
    return "\n".join(lines)


def run(roots: list[Path], out: Path, backend_name: str = "optivision.easy") -> dict:
    ctx = Context(roots, heldout=heldout_models(roots))
    results = predictions.score_all(ctx)

    def rerun(arm: str, target: float) -> bool | None:
        rows, missing = predictions.rule_rows(ctx, predictions.v1_units(), (target,), arm=arm)
        rows = predictions.scored(rows)
        if missing or not rows:
            return None
        crit = predictions.crit_v1b if target == 0.97 else predictions.crit_v1c
        return crit(rows)[0]

    decisions = decide.decide(results, rerun)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(results, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "decisions.json").write_text(json.dumps({k: v for k, v in decisions.items() if k != "ledger_change"},
                                                   indent=1, default=float) + "\n", encoding="utf-8")
    (out / "ledger_proposal.json").write_text(json.dumps(decisions["ledger_change"], indent=1, default=float) + "\n",
                                              encoding="utf-8")
    (out / "SUMMARY.md").write_text(summary_md(results, decisions, backend_name, roots), encoding="utf-8", newline="\n")
    return {"results": results, "decisions": decisions, "missing": sorted(ctx.missing)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=HERE.parents[1] / "data" / "easy")
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "reports" / "easy" / "results")
    ap.add_argument("--synthetic", action="store_true", help="dry run on generated units with the fake backend")
    a = ap.parse_args(argv)
    if a.synthetic:
        from easy_validation import fakes, synthetic

        with tempfile.TemporaryDirectory() as tmp, api.use(fakes.FakeBackend()):
            root = synthetic.write_tree(Path(tmp))
            info = run(batch_roots(root), a.out, "fakes (synthetic)")
    else:
        info = run(batch_roots(a.data), a.out)
    for k, r in info["results"].items():
        print(f"{k:14s} {_fmt(r['passed']):11s} {r['reason'][:110]}")
    if info["missing"]:
        print(f"missing units: {len(info['missing'])} (first: {info['missing'][:3]})")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
