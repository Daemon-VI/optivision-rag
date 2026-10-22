"""The easy-mode validation programme: notebooks, guard, time budget, ingestion and analyses.

Offline and CPU only. The analyses run end to end on a tiny synthetic unit tree with the
fake backend (``scripts/easy_validation/fakes.py``), so they test the plumbing, not the
package's rule or practice check.
"""

from __future__ import annotations

import importlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

pytestmark = pytest.mark.skipif(not (SCRIPTS / "easy_validation" / "units.py").exists(),
                                reason="needs the repository's scripts")
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _mod(name: str):
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def units():
    return _mod("easy_validation.units")


@pytest.fixture(scope="module")
def maker():
    return _mod("make_easy_validation")


# --------------------------------------------------------------------------- units and budget


def test_units_match_the_design(units):
    c = units.CORPORA
    # FINAL_DESIGN section 8 (and Table 6 of arXiv 2601.08620, English-original queries)
    assert (c["v3_hr"].pages, c["v3_hr"].queries) == (1110, 318)
    assert (c["v3_computer_science"].pages, c["v3_computer_science"].queries) == (1360, 215)
    assert (c["v3_pharmaceuticals"].pages, c["v3_pharmaceuticals"].queries) == (2313, 364)
    assert (c["v3_industrial"].pages, c["v3_industrial"].queries) == (5244, 283)
    assert (c["rmm_techslides"].pages, c["rmm_techslides"].queries) == (1963, 1354)
    assert c["beir_fiqa10k"].subsample == 10_000 and "beir_arguana" in units.SELF_MATCH_EXCLUDED
    assert all(x.scored for x in c.values()), "every registered corpus has >= 150 queries"
    assert list(units.DESIGN) == ["vidore/colpali-v1.3", "vidore/colqwen2-v1.0-hf", "vidore/colqwen2.5-v0.2",
                                  "vidore/colSmol-500M"]
    assert set(units.V2_MODELS) == {"vidore/colSmol-500M", "vidore/colqwen2-v1.0-hf"}
    assert units.TARGETS == (0.90, 0.95, 0.97, 0.98)
    assert units.s_of_n(500) == 0 and abs(units.s_of_n(2500) - 1) < 1e-12 and units.round_up(0.0151) == 0.0175


def test_heldout_candidates_are_outside_the_registry(units):
    from optivision.adapters import MODELS

    registry = {m.lower() for m in MODELS}
    assert len(units.HELDOUT_CANDIDATES) >= 2
    downloads = [c.downloads for c in units.HELDOUT_CANDIDATES]
    assert downloads == sorted(downloads, reverse=True), "candidates are tried in download order"
    for c in units.HELDOUT_CANDIDATES:
        assert c.model.lower() not in registry
    for name in units.HELDOUT_EXCLUDED:
        assert name.lower() not in registry


def test_every_batch_fits_one_session(units):
    batches = units.all_batches()
    assert {b.prereg_batch for b in batches} >= {"B1", "B2", "B3", "DT", "VLM"}
    for b in batches:
        lo, hi = units.batch_hours(b)
        assert 0 < lo <= hi <= units.DEADLINE_HOURS, (b.name, lo, hi)
    # B1-B3 together stay near the design's 23-31 h
    lo = sum(units.batch_hours(b)[0] for b in batches if b.prereg_batch in ("B1", "B2", "B3"))
    hi = sum(units.batch_hours(b)[1] for b in batches if b.prereg_batch in ("B1", "B2", "B3"))
    assert 20 <= lo <= hi <= 34, (lo, hi)


def test_dt_plan_meets_the_type_rule(units):
    dt = _mod("easy_validation.dt")
    for kind, datasets in dt.DT_TYPES.items():
        assert len(datasets) >= 2, kind
    assert len(dt.DT_MODELS) >= 4
    for b in dt.DT_BATCHES:
        for j in b.jobs:
            for c in j.corpora:
                assert units.corpus(c).queries >= units.MIN_QUERIES or c in dt.REPORTED_ONLY, c
    lo, hi = dt.budget()
    assert hi <= dt.GPU_BUDGET_HOURS


# --------------------------------------------------------------------------- notebooks


def test_notebooks_generate_deterministically(tmp_path, maker, units):
    a, b = tmp_path / "a", tmp_path / "b"
    maker.main(a)
    maker.main(b)
    names = sorted(p.name for p in a.glob("*.ipynb"))
    assert names == sorted(x.notebook for x in units.all_batches())
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


def test_committed_notebooks_are_current(maker, units):
    for b in units.all_batches():
        path = maker.OUT / b.notebook
        assert path.exists(), f"run python scripts/make_easy_validation.py ({b.notebook} missing)"
        assert path.read_text(encoding="utf-8") == maker.render(b), f"{b.notebook} is stale"


def _cells(maker, b):
    return ["".join(c["source"]) for c in maker.notebook(b)["cells"]]


def test_every_notebook_guards_before_installing_or_encoding(maker, units):
    for b in units.all_batches():
        cells = _cells(maker, b)
        guard = next(i for i, c in enumerate(cells) if "REFUSING TO RUN" in c)
        clone = next(i for i, c in enumerate(cells) if "git clone" in c)
        install = next(i for i, c in enumerate(cells) if "pip install" in c)
        encode = next(i for i, c in enumerate(cells) if "encode_unit.py" in c and "subprocess.run(cmd" in c)
        assert clone < guard < install < encode, b.name
        assert 'BRANCH = "easy-mode"' in cells[1] and "PREREG_EASY.frozen.json" in cells[guard]
        assert "REQUIRED_PARTS" in cells[1] and '"1"' in cells[1]


def _guard_env(maker, b, repo):
    cells = _cells(maker, b)
    config = {}
    exec(cells[1], config)  # noqa: S102 -- our own generated cell
    guard = next(c for c in cells if "REFUSING TO RUN" in c)
    return guard, {**config, "REPO": str(repo)}


def _frozen_tree(tmp_path, maker, b, units):
    repo = tmp_path / "repo"
    _, env = _guard_env(maker, b, repo)
    files = sorted(set(env["REQUIRED_FILES"]) | set(env["REQUIRED_FROZEN"]) | {"reports/easy/PREREG_EASY.md"})
    for rel in files:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {rel}\n", encoding="utf-8")
    freeze = _mod("easy_validation.freeze")
    entry = {"files": {rel: freeze.digest(repo / rel) for rel in files}}
    record = {"schema": 1, "parts": {str(p): entry for p in env["REQUIRED_PARTS"]}}
    (repo / "reports/easy/PREREG_EASY.frozen.json").write_text(json.dumps(record), encoding="utf-8")
    return repo


def test_guard_refuses_without_a_frozen_preregistration(tmp_path, maker, units):
    b = units.batch("e1a_easy_b1_design")
    guard, env = _guard_env(maker, b, tmp_path / "empty")
    with pytest.raises(SystemExit, match="not frozen"):
        exec(guard, env)  # noqa: S102


def test_guard_passes_on_a_frozen_tree_and_refuses_after_a_change(tmp_path, maker, units):
    for b in units.all_batches():
        repo = _frozen_tree(tmp_path / b.name, maker, b, units)
        guard, env = _guard_env(maker, b, repo)
        exec(guard, dict(env))  # noqa: S102 -- passes
        target = repo / "scripts/easy_validation/encode_unit.py"
        target.write_text(target.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="changed after it was frozen"):
            exec(guard, dict(env))  # noqa: S102


def test_guard_refuses_a_missing_part_or_input(tmp_path, maker, units):
    b = units.batch("e3b_easy_b3_heldout_text")
    repo = _frozen_tree(tmp_path, maker, b, units)
    (repo / units.HELDOUT_SELECTED).unlink()
    guard, env = _guard_env(maker, b, repo)
    with pytest.raises(SystemExit, match="heldout_selected.json is missing"):
        exec(guard, env)  # noqa: S102
    frozen = repo / "reports/easy/PREREG_EASY.frozen.json"
    record = json.loads(frozen.read_text(encoding="utf-8"))
    record["parts"].pop("2")
    frozen.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="part 2 is not frozen"):
        exec(guard, env)  # noqa: S102


def test_freeze_refuses_drafts_and_refreezing(tmp_path, monkeypatch):
    freeze = _mod("easy_validation.freeze")
    for rel in freeze.PARTS["3"]["required"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# PREREG part 3 (DRAFT - NOT FROZEN)\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "FROZEN", tmp_path / "reports/easy/PREREG_EASY.frozen.json")
    with pytest.raises(SystemExit, match="DRAFT"):
        freeze.freeze("3", tmp_path)
    (tmp_path / "reports/easy/PREREG_EASY_PART3.md").write_text("# PREREG part 3\n", encoding="utf-8")
    record = freeze.freeze("3", tmp_path)
    freeze.FROZEN.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="already frozen"):
        freeze.freeze("3", tmp_path)
    assert freeze.verify(["3"], tmp_path, freeze.FROZEN) == []


def test_drafts_are_marked_and_not_frozen():
    for name in ("PREREG_EASY.md", "PREREG_EASY_PART2.md", "PREREG_EASY_PART3.md"):
        text = (ROOT / "reports" / "easy" / name).read_text(encoding="utf-8")
        assert "DRAFT" in text.splitlines()[0] and "NOT FROZEN" in text.splitlines()[0], name
    assert not (ROOT / "reports/easy/PREREG_EASY.frozen.json").exists(), "the integrator freezes"


# --------------------------------------------------------------------------- encode side (CPU parts)


class _Src:
    """A text 'corpus' for encode_job: items are strings, encoded by a CallableAdapter."""


def test_encode_job_writes_shards_and_resumes(tmp_path, monkeypatch):
    eu = _mod("easy_validation.encode_unit")
    from optivision.adapters import CallableAdapter

    calls = []

    def enc(items):
        calls.append(len(items))
        rng = np.random.default_rng(len(calls))
        return [v / np.linalg.norm(v, axis=1, keepdims=True) for v in (rng.normal(size=(3, 8)) for _ in items)]

    adapter = CallableAdapter(enc, enc)
    texts = [f"document {i}" for i in range(45)]
    rows = [{"text": t} for t in texts]
    src = eu.Source("beir_fiqa10k", [f"d{i}" for i in range(45)], [{"row": i, "id": f"d{i}"} for i in range(45)],
                    [("q1", "a question")], {"q1": {"d3": 1}}, "text", rows, "text")
    monkeypatch.setattr(eu, "TEXT_SHARD_DOCS", 20)
    folder = tmp_path / "m__beir_fiqa10k"
    session = eu.Session(1.0)
    monkeypatch.setattr(eu, "RESERVE_SECONDS", 0)
    assert eu.encode_job(adapter, src, folder, session, False, []) == "complete"
    shards = sorted((folder / "docs_f16").glob("shard_*.npz"))
    assert [s.name for s in shards] == ["shard_000000.npz", "shard_000020.npz", "shard_000040.npz"]
    data = _mod("easy_validation.data")
    docs = data.read_shards(folder / "docs_f16")
    assert docs.ids == src.ids and docs.vectors.dtype == np.float16 and len(docs) == 45
    n = len(calls)
    (folder / "docs_f16" / "shard_000020.npz").unlink()
    assert eu.encode_job(adapter, src, folder, session, False, []) == "complete"
    assert calls[n:] == [20], "only the missing shard is encoded again"


def test_encode_job_stops_cleanly_at_the_deadline(tmp_path, monkeypatch):
    eu = _mod("easy_validation.encode_unit")
    from optivision.adapters import CallableAdapter

    adapter = CallableAdapter(lambda xs: [np.eye(2)[:1] for _ in xs], lambda xs: [np.eye(2)[:1] for _ in xs])
    src = eu.Source("beir_fiqa10k", ["a", "b"], [{"row": 0, "id": "a"}, {"row": 1, "id": "b"}], [], {}, "text",
                    [{"text": "x"}, {"text": "y"}], "text")
    session = eu.Session(0.0)
    log: list = []
    assert eu.encode_job(adapter, src, tmp_path / "u", session, False, log) == "partial" and log


def test_check_pairs_are_distinct_and_unambiguous():
    eu = _mod("easy_validation.encode_unit")
    ids = [f"p{i}" for i in range(6)]
    qrels = {"1": {"p0": 2, "p1": 1}, "2": {"p1": 2}, "3": {"p2": 1}, "4": {"p0": 1}, "5": {"p3": 1}}
    src = eu.Source("v3_hr", ids, [{"row": i, "id": p} for i, p in enumerate(ids)],
                    [(q, f"query {q}") for q in qrels], qrels)
    pairs = eu.check_pairs(src, n=12)
    pages = [ids[i] for i, _ in pairs]
    assert len(pages) == len(set(pages))
    assert pages[0] == "p0" and "p1" not in pages  # query 2's page is relevant to query 1


# --------------------------------------------------------------------------- ingestion


def test_ingest_splits_records_from_vectors(tmp_path):
    ingest = _mod("easy_validation.ingest")
    archive = tmp_path / "results_e9z_easy_test.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("results_e9z_easy_test/batch_summary.json", json.dumps([{"model": "m", "outcome": "complete"}]))
        z.writestr("results_e9z_easy_test/commit.txt", "abc\n")
        z.writestr("results_e9z_easy_test/m__v3_hr/run.json", json.dumps({"outcome": "complete"}))
        z.writestr("results_e9z_easy_test/m__v3_hr/docs_f16/shard_000000.npz", b"vectors")
        z.writestr("results_e9z_easy_test/_corpus/v3_hr/texts.json.gz", b"text")
        z.writestr("results_e9z_easy_test/_corpus/v3_hr/pages.json", "[]")
        z.writestr("results_e9z_easy_test/../escape.json", "{}")
    outcomes: dict = {}
    reports, data = tmp_path / "reports", tmp_path / "data"
    ingest.ingest(archive, outcomes, reports, data)
    r, d = reports / "e9z_easy_test", data / "e9z_easy_test"
    assert (r / "m__v3_hr/run.json").exists() and (r / "commit.txt").exists()
    assert not (r / "m__v3_hr/docs_f16").exists() and not (r / "_corpus/v3_hr/texts.json.gz").exists()
    assert not (r / "_corpus/v3_hr/pages.json").exists()
    assert (d / "m__v3_hr/docs_f16/shard_000000.npz").exists() and (d / "_corpus/v3_hr/pages.json").exists()
    assert not (tmp_path / "escape.json").exists() and not (reports / "escape.json").exists()
    assert outcomes["m @ - @ e9z_easy_test"]["outcome"] == "complete"


def test_campaign_ingest_routes_easy_batches_away():
    spec = importlib.util.spec_from_file_location("ingest_campaign", SCRIPTS / "ingest_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.is_easy_validation("e1a_easy_b1_design") and not module.is_easy_validation("f14a_easy_b0_colqwen")
    assert not module.is_followup("e1a_easy_b1_design")


# --------------------------------------------------------------------------- analyses


def test_rule_arms_and_rounding(units):
    rule = _mod("easy_validation.rule")
    api = _mod("easy_validation.api")
    fakes = _mod("easy_validation.fakes")
    b = fakes.FakeBackend()
    sel = b.select("vidore/colpali-v1.3", 0.95, 500, checked=True)
    pick, seq = rule.arm_choice(sel, 0.95, 500, "A", "page")
    assert pick.key == api.pick(sel).key
    assert api.rung(seq[-1]).name == "float16"
    # at 2,500 pages the size term asks more of merged and 1-bit rungs (arm A), arm C ignores it
    a, _ = rule.arm_choice(b.select("vidore/colpali-v1.3", 0.95, 2500, checked=True), 0.95, 2500, "A", "page")
    c, _ = rule.arm_choice(b.select("vidore/colpali-v1.3", 0.95, 2500, checked=True), 0.95, 2500, "C", "page")
    assert (c.ratio or 0) >= (a.ratio or 0)
    coef = rule.refit_coefficients({"codec": [(0.003, 1.0), (0.001, 0.43)], "binary": [(-0.01, 1.0)]})
    assert coef["codec"] == 0.005 and coef["binary"] == 0.0 and coef["merge>binary"] == 0.04


def test_decision_rules_propose_and_never_apply(units):
    decide = _mod("easy_validation.decide")
    results = {"V1b": {"passed": False}, "V1c": {"passed": True}, "V1d": {"passed": False, "per_target": {"0.9": False}},
               "V1-unchecked": {"passed": False}, "V2": {"passed": None}, "V3": {"passed": True},
               "V4": {"passed": False}, "V5a": {"passed": True}, "V5b": {"passed": False}, "V5c": {"passed": True},
               "V1u": {"passed": False, "int8_low_min": 0.985, "int4_mean_low_min": 0.94},
               "V1-arms": {"comparable": True, "arms": {"A": {"passes": {"V1b": False, "V1c": True}},
                                                        "B": {"passes": {"V1b": True, "V1c": True}, "severe_one_bit": 0},
                                                        "C": {"passes": {"V1b": False}}, "D": {"passes": {}}}}}
    ledger = (ROOT / "src/optivision/presets/easy_ledger.json").read_bytes()
    out = decide.decide(results, rerun=lambda arm, t: False)
    s = out["settings"]
    assert s["features.margins.codec_chains_only_at.0.97"] is True
    assert s["features.margins.lowest_target"] == 0.95
    assert s["features.margins.unchecked_merge_penalty"] == 0.01
    assert s["features.margins.one_bit_margin"] == 0.0
    assert s["features.unmeasured_rules.int8_at"] == 0.98 and "features.unmeasured_rules.int4_mean_at" not in s
    assert s["features.unmeasured_rules.float16_above"] == 0.95
    assert s["features.practice_check.mode"] == "warning"
    assert "features.margins.codecs_only_above" not in s, "an unscored prediction applies no rule"
    assert out["status"] == decide.IN_SAMPLE
    assert (ROOT / "src/optivision/presets/easy_ledger.json").read_bytes() == ledger


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    units = _mod("easy_validation.units")
    api = _mod("easy_validation.api")
    fakes = _mod("easy_validation.fakes")
    synthetic = _mod("easy_validation.synthetic")
    run_all = _mod("easy_validation.run_all")
    data = _mod("easy_validation.data")
    saved = {k: getattr(units, k) for k in ("MIN_QUERIES", "HARD_MIN", "DESIGN", "V1_CORPORA", "TEXT_MODELS")}
    units.MIN_QUERIES, units.HARD_MIN = 10, 10
    units.DESIGN = dict(list(units.DESIGN.items())[1:3])  # colqwen2-v1.0-hf (a V2 model) and colqwen2.5-v0.2
    units.V1_CORPORA = units.V1_CORPORA[:2]
    units.TEXT_MODELS = dict(list(units.TEXT_MODELS.items())[:1])
    try:
        root = tmp_path_factory.mktemp("easy")
        synthetic.write_tree(root, n_pages=36, n_queries=24)
        with api.use(fakes.FakeBackend(hard_min=10, boot=100)):
            info = run_all.run(data.batch_roots(root), root / "out", "fakes")
        yield info, root / "out"
    finally:
        for k, v in saved.items():
            setattr(units, k, v)


def test_analyses_run_end_to_end_on_a_synthetic_unit(synthetic_run):
    info, out = synthetic_run
    results = info["results"]
    expected = {"V1a", "V1b", "V1c", "V1d", "V1-unchecked", "V1-arms", "V1t", "V1u", "S", "arm S", "V2", "V3", "V4",
                "V5a", "V5b", "V5c", "V5d", "V6"}
    assert expected <= set(results)
    for k in expected:
        r = results[k]
        assert r["passed"] in (True, False, None) and r["prediction"] and r["pass_if"], k
    assert results["V1b"]["units"] and all(0 <= u["point"] <= 1.2 for u in results["V1b"]["units"])
    assert results["V1-arms"]["comparable"], "arm A recomputed from the rows equals select()"
    assert {u["pair"] for u in results["V3"]["units"]} >= {"sample fit vs full fit", "float16 vs float32 inputs"}
    assert results["V6"]["passed"] is True
    for name in ("predictions.json", "decisions.json", "ledger_proposal.json", "SUMMARY.md"):
        assert (out / name).exists()
    proposal = json.loads((out / "ledger_proposal.json").read_text(encoding="utf-8"))
    assert "proposed only" in proposal["note"]


def test_float16_layout_keeps_everything_and_faults_hurt(synthetic_run):
    info, _ = synthetic_run
    v5a = info["results"]["V5a"]["units"]
    assert v5a, "the planted faults were scored"
    random_rows = [r for r in v5a if r["fault"] == "random scores"]
    assert random_rows and max(r["real_point"] for r in random_rows) < 0.8
