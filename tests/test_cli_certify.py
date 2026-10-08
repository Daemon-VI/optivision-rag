"""``optivision certify`` and ``optivision verify`` through Typer's test runner, on small files.

Output is plain lines (never wrapped, never read as markup), so the assertions look for the words
that matter. ``--json`` output is checked as JSON on stdout alone. Exit codes: 0 for any outcome
of ``certify`` (abstaining is a result), 2 for invalid input, 1 for a failed verification.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from optivision.cli import app
from optivision.core.artifact import Certificate, sign_certificate
from optivision.core.artifact.record import content_digest
from optivision.core.identity import digest
from optivision.core.selection import spaces as spaces_module
from optivision.testing import make_corpus, make_queries

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()

try:
    import scipy.cluster.hierarchy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAVE_SCIPY, reason="the default candidate list needs SciPy")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli_certify")
    corpus = make_corpus(40, 16, tokens=(8, 12), seed=3)
    queries, qrels = make_queries(corpus, 120, noise=1.0, seed=4)
    corpus.save(d / "docs.npz")
    queries.save(d / "queries.npz")
    queries.select(range(8)).save(d / "few.npz")
    make_corpus(40, 16, tokens=(8, 12), seed=77).save(d / "other.npz")
    (d / "qrels.json").write_text(json.dumps({k: sorted(v) for k, v in qrels.items()}), encoding="utf-8")
    return types.SimpleNamespace(
        dir=d, docs=str(d / "docs.npz"), queries=str(d / "queries.npz"), few=str(d / "few.npz"),
        other=str(d / "other.npz"), qrels=str(d / "qrels.json"))


@pytest.fixture(scope="module")
def selected_run(files):
    """One real run of the default list: the JSON report and the artifact it wrote."""
    if not HAVE_SCIPY:
        pytest.skip("the default candidate list needs SciPy")
    out = files.dir / "selected-artifact"
    r = runner.invoke(app, ["certify", files.docs, files.queries, "--qrels", files.qrels,
                            "--target", "0.9", "--out", str(out), "--json"])
    assert r.exit_code == 0, r.output
    return types.SimpleNamespace(report=json.loads(r.stdout), artifact=out)


def invoke(*args):
    return runner.invoke(app, [str(a) for a in args])


def squash(text: str) -> str:
    """``text`` without whitespace and box-drawing characters: Typer draws usage errors in a
    panel that wraps long messages."""
    return re.sub(r"[\s\u2500-\u257f]", "", text)


# ------------------------------------------------------------------------- certify


@needs_scipy
def test_certify_prints_a_selection_plainly(files, tmp_path):
    r = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--target", "0.9",
               "--out", tmp_path / "artifact")
    assert r.exit_code == 0, r.output
    out = r.output
    assert "outcome: select (the rule rejected 'retention <= target' for this candidate" in out
    assert "at the stated error level, 0.05; claimed only under the assumptions" in out
    assert "outcome: select (certified)" not in out  # the headline claims no more than the rule
    assert "reference" in out and "labels" in out and "ndcg@5" in out
    assert ("40 candidates in 7 families; the default list registered for the development "
            "stores, not validated for other encoders") in out
    assert ("120 tested of 120 given (0 without a relevant document), "
            "order: seeded permutation (seed 0)") in out
    assert "compression" in out and "x vs float32 (a size, not a quality figure)" in out
    assert "(a property of the code, not a quality figure)" in out
    # the statement comes with its conditions, and square brackets in it are printed as written
    assert "what the rule states (claimed only if every assumption and condition below holds)" in out
    assert "conditional on" not in out
    assert "E_Q[metric of the candidate]" in out
    assert "This is a joint probability" in out and "not a statement about power" in out
    assert all(f"{k}{i}:" in out for k, n in (("A", 6), ("C", 4)) for i in range(1, n + 1))
    assert "certificate" in out and "artifact" in out
    written = {p.name for p in (tmp_path / "artifact").iterdir()}
    assert {"MANIFEST.json", "certificate.json", "codes.npz", "evidence.npz"} <= written
    assert written <= {"MANIFEST.json", "certificate.json", "codes.npz", "evidence.npz", "fitted.npz"}
    head = out.split("what the rule states")[0]  # no measured retention figure before the statement
    assert not [ln for ln in head.splitlines() if "retention" in ln and not ln.startswith("outcome:")]


def test_certify_json_is_the_report_alone_on_stdout(selected_run):
    rep = selected_run.report
    assert rep["outcome"] == "select" and rep["certified"] is True
    assert rep["construct"] == "labels" and rep["metric"] == "ndcg@5"
    assert rep["target"] == 0.9 and rep["error"] == 0.05
    assert rep["space"]["name"] == "default" and rep["space"]["n_candidates"] == 40
    assert rep["space"]["registered_default"] is True
    assert rep["selected"]["label"] and rep["compression_ratio"] > 1.0
    assert rep["basis_sensitive"] in (True, False)
    assert rep["artifact"] == str(selected_run.artifact)
    assert len(rep["assumptions"]) == 10 and rep["not_certified_note"] is None
    assert rep["n_exact_duplicate_queries"] == 0
    assert rep["query_order"] == {"kind": "seeded_permutation", "seed": 0, "budget": None}


@needs_scipy
def test_nothing_certified_is_a_normal_result_with_exit_status_zero(files, tmp_path):
    r = invoke("certify", files.docs, files.few, "--qrels", files.qrels, "--target", "0.999",
               "--out", tmp_path / "abstained")
    assert r.exit_code == 0, r.output
    out = r.output
    assert "outcome: abstain (nothing certified; use the uncompressed index)" in out
    assert "safe outcome, not an error" in out
    assert "new, independent queries" in out and "outside the statement of the rule" in out
    assert "a lower target" not in out
    assert not re.search(r"^\s+(pipeline|compression)\s", out, re.MULTILINE)  # nothing was compressed
    assert sorted(p.name for p in (tmp_path / "abstained").iterdir()) == [
        "MANIFEST.json", "certificate.json", "evidence.npz"]
    j = invoke("certify", files.docs, files.few, "--qrels", files.qrels, "--target", "0.999", "--json")
    assert j.exit_code == 0
    rep = json.loads(j.stdout)
    assert rep["outcome"] == "abstain" and rep["certified"] is False and rep["selected"] is None
    assert rep["compression_ratio"] == 1.0 and rep["basis_sensitive"] is None
    assert rep["not_certified_note"] and rep["artifact"] is None


@needs_scipy
def test_without_labels_the_reference_is_agreement_and_the_output_says_it_is_not_quality(files):
    r = invoke("certify", files.docs, files.queries, "--target", "0.9")
    assert r.exit_code == 0, r.output
    assert "agreement@1" in r.output
    assert "agreement with the uncompressed ranking, which is not retrieval quality" in r.output
    j = json.loads(invoke("certify", files.docs, files.queries, "--target", "0.9", "--json",
                          "--reference", "agreement").stdout)
    assert j["construct"] == "agreement@1"
    assert any("not retrieval quality" in c for c in j["caveats"])


@needs_scipy
def test_budget_and_order_seed_are_recorded(files):
    r = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--target", "0.9",
               "--budget", "60", "--order-seed", "5", "--json")
    assert r.exit_code == 0, r.output
    rep = json.loads(r.stdout)
    assert rep["n_queries"] == 60 and rep["n_queries_given"] == 120
    assert rep["query_order"] == {"kind": "seeded_permutation", "seed": 5, "budget": 60}
    text = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--budget", "60",
                  "--order-seed", "5").output
    assert "60 tested of 120 given" in text and "seeded permutation (seed 5), budget 60" in text


@needs_scipy
def test_the_order_is_a_seeded_random_one_unless_the_order_of_the_file_is_asked_for(files):
    default = json.loads(invoke("certify", files.docs, files.queries, "--qrels", files.qrels,
                                "--target", "0.9", "--json").stdout)
    assert default["query_order"] == {"kind": "seeded_permutation", "seed": 0, "budget": None}
    assert default["caveats"] == []  # the default seed asks nothing of the caller
    given = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--target", "0.9",
                   "--given-order", "--json")
    assert given.exit_code == 0, given.output
    rep = json.loads(given.stdout)
    assert rep["query_order"] == {"kind": "given", "seed": None, "budget": None}
    assert any("order supplied" in c and "unrelated to the outcomes" in c for c in rep["caveats"])
    text = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--target", "0.9",
                  "--given-order").output
    assert "order: given (the order of the file)" in text
    assert "The queries were tested in the order supplied" in text
    seven = json.loads(invoke("certify", files.docs, files.queries, "--qrels", files.qrels,
                              "--target", "0.9", "--order-seed", "7", "--json").stdout)
    assert any("order seed (7) was chosen by the caller" in c for c in seven["caveats"])


def test_graded_labels_are_refused_before_any_candidate_is_scored(files, tmp_path):
    graded = tmp_path / "graded.json"
    graded.write_text(json.dumps({"q00000": {"d000001": 1, "d000002": 0}}), encoding="utf-8")
    r = invoke("certify", files.docs, files.queries, "--qrels", graded)
    assert r.exit_code == 2
    assert squash("Graded labels are not supported") in squash(r.output)
    assert squash("grade 0 included") in squash(r.output)
    assert squash("graded labels are refused") in squash(invoke("certify", "--help").output)


def test_verify_refuses_graded_labels_too(selected_run, tmp_path):
    graded = tmp_path / "graded.json"
    graded.write_text(json.dumps({"q00000": {"d000001": 1, "d000002": 0}}), encoding="utf-8")
    v = invoke("verify", selected_run.artifact, "--qrels", graded)
    assert v.exit_code == 2 and squash("Graded labels are not supported") in squash(v.output)


def test_the_two_ways_of_choosing_the_order_cannot_be_combined(files):
    r = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--given-order",
               "--order-seed", "3")
    assert r.exit_code == 2
    assert squash("cannot be combined with --order-seed") in squash(r.output)


def test_the_help_says_what_the_order_requires():
    flat = " ".join(re.sub(r"[\s\u2500-\u257f]+", " ", invoke("certify", "--help").output).split())
    flat = flat.replace("| ", "").replace("\u2502", "")
    squashed = squash(flat)
    for sentence in ("seed of the random order in which the queries are tested (default 0",
                     "trying several seeds and reporting one is outside the statement",
                     "valid only if that order is unrelated to the outcomes",
                     "never sorted by difficulty, date, source or any query attribute"):
        assert squash(sentence) in squashed, sentence


@needs_scipy
def test_repeated_queries_are_refused_unless_the_flag_asserts_independence(files, tmp_path):
    from optivision.representation import MultiVectorCorpus

    queries = MultiVectorCorpus.load(files.queries)
    qrels = json.loads(Path(files.qrels).read_text(encoding="utf-8"))
    arrays = [queries.vectors[queries.offsets[i]:queries.offsets[i + 1]] for i in range(len(queries))]
    doubled = MultiVectorCorpus.from_arrays(arrays + arrays,
                                            ids=[*queries.ids, *[f"{q}_copy" for q in queries.ids]])
    doubled.save(tmp_path / "doubled.npz")
    labels = {**qrels, **{f"{q}_copy": docs for q, docs in qrels.items()}}
    (tmp_path / "doubled.json").write_text(json.dumps(labels), encoding="utf-8")
    refused = invoke("certify", files.docs, tmp_path / "doubled.npz", "--qrels",
                     tmp_path / "doubled.json", "--target", "0.9", "--json")
    assert refused.exit_code != 0
    assert "allow_duplicate_queries=True" in refused.output
    with pytest.warns(UserWarning, match="120 of the 240 queries tested repeat an earlier query"):
        r = invoke("certify", files.docs, tmp_path / "doubled.npz", "--qrels",
                   tmp_path / "doubled.json", "--target", "0.9", "--json",
                   "--allow-duplicate-queries")
    assert r.exit_code == 0, r.output
    rep = json.loads(r.stdout)
    assert rep["n_exact_duplicate_queries"] == 120 and rep["n_queries"] == 240
    assert any("120 of the 240 queries tested repeat an earlier query exactly" in c
               for c in rep["caveats"])
    with pytest.warns(UserWarning):
        text = invoke("certify", files.docs, tmp_path / "doubled.npz", "--qrels",
                      tmp_path / "doubled.json", "--target", "0.9",
                      "--allow-duplicate-queries").output
    assert "120 repeat an earlier query exactly" in text


@needs_scipy
def test_the_wide_space_is_an_explicit_choice_and_not_the_registered_list(files):
    r = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--target", "0.9",
               "--space", "wide", "--json")
    assert r.exit_code == 0, r.output
    space = json.loads(r.stdout)["space"]
    assert space["name"] == "wide" and space["registered_default"] is False
    assert space["n_candidates"] > 40 and space["n_families"] > 7


@pytest.mark.parametrize(("args", "message"), [
    (["--reference", "relevance"], "reference must be one of auto, labels, agreement"),
    (["--space", "huge"], "space must be one of default, wide"),
    (["--reference", "labels"], "reference=labels needs labels"),
    (["--reference", "agreement", "--qrels", "QRELS"], "not used by reference=agreement"),
])
def test_invalid_options_exit_with_status_two(files, args, message):
    args = [files.qrels if a == "QRELS" else a for a in args]
    r = invoke("certify", files.docs, files.queries, *args)
    assert r.exit_code == 2
    assert squash(message) in squash(r.output)


@pytest.mark.parametrize(("args", "message"), [
    (["--target", "1.5"], "target must be in (0, 1]"),
    (["--target", "0"], "target must be in (0, 1]"),
    (["--error", "0"], "error must be in (0, 1)"),
    (["--error", "1"], "error must be in (0, 1)"),
    (["--budget", "0"], "query_budget must be >= 1"),
    (["--budget", "500"], "exceeds the 120 queries given"),
    (["--metric", "map@10"], "metric"),
])
def test_values_the_library_refuses_exit_with_status_two_and_say_why(files, args, message):
    r = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, *args)
    assert r.exit_code == 2
    assert message in r.stderr and r.stdout == ""


def test_a_missing_file_is_a_usage_error(files, tmp_path):
    r = invoke("certify", tmp_path / "nothing.npz", files.queries)
    assert r.exit_code == 2 and squash("does not exist") in squash(r.output)
    r = invoke("certify", files.docs, files.queries, "--qrels", tmp_path / "nothing.json")
    assert r.exit_code == 2 and squash("does not exist") in squash(r.output)


def test_an_output_directory_that_is_in_use_is_refused_before_any_work(files, tmp_path):
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "notes.txt").write_text("keep", encoding="utf-8")
    r = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--out", taken)
    assert r.exit_code == 2 and squash("already exists and is not empty") in squash(r.output)
    assert (taken / "notes.txt").read_text(encoding="utf-8") == "keep"
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    assert invoke("certify", files.docs, files.queries, "--out", afile).exit_code == 2


def test_queries_of_another_dimension_are_refused_naming_both(files, tmp_path):
    other = make_corpus(10, 8, seed=1)
    path = other.save(tmp_path / "eight.npz")
    r = invoke("certify", files.docs, path)
    assert r.exit_code == 2
    assert "queries have dimension 8 but the vectors have dimension 16" in r.stderr


def test_without_scipy_the_cli_says_how_to_get_it_and_does_not_reduce_the_list(files, monkeypatch):
    monkeypatch.setattr(spaces_module, "_have_scipy", lambda: False)
    r = invoke("certify", files.docs, files.queries, "--qrels", files.qrels)
    assert r.exit_code == 1
    flat = " ".join(r.stderr.split())
    assert "optivision-rag[merge]" in flat and "never reduced silently" in flat
    assert r.stdout == ""


def test_allow_reduced_runs_the_reduced_list_without_scipy_and_records_it(files, monkeypatch):
    monkeypatch.setattr(spaces_module, "_have_scipy", lambda: False)
    refused = invoke("certify", files.docs, files.queries, "--qrels", files.qrels)
    assert refused.exit_code == 1 and "--allow-reduced" in refused.stderr
    r = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--target", "0.9",
               "--allow-reduced", "--json")
    assert r.exit_code == 0, r.output
    space = json.loads(r.stdout)["space"]
    assert space["name"] == "default-reduced" and space["registered_default"] is False
    wide = invoke("certify", files.docs, files.queries, "--qrels", files.qrels, "--space", "wide",
                  "--allow-reduced", "--target", "0.9", "--json")
    assert wide.exit_code == 0, wide.output


def test_the_existing_commands_and_the_two_new_ones_are_listed():
    r = invoke("--help")
    assert r.exit_code == 0
    for command in ("index", "search", "inspect", "compress", "calibrate", "benchmark", "compare",
                    "certify", "verify"):
        assert command in r.output
    for command, option in (("certify", "--trust-pickle"), ("certify", "--order-seed"),
                            ("verify", "--strict-code"), ("verify", "--key-file")):
        assert option in invoke(command, "--help").output


# ------------------------------------------------------------------------- verify


def test_verify_reports_every_check_and_passes_a_fresh_artifact(selected_run):
    r = invoke("verify", selected_run.artifact)
    assert r.exit_code == 0, r.output
    for name in ("artifact files", "schema", "digest", "consistency.decision", "replay.decision",
                 "binding.pipeline", "binding.codes"):
        assert name in r.output
    assert "[pass   ] digest" in r.output and "[skipped] binding.corpus" in r.output
    # corpus, queries and reference were not passed: the certificate is not bound to any data
    assert "OK: no check failed (level reached: replay)" in r.output
    assert "does not make the certificate true" in r.output


def test_verify_binds_the_certificate_to_the_files_it_is_given(selected_run, files):
    r = invoke("verify", selected_run.artifact, "--vectors", files.docs, "--queries", files.queries,
               "--qrels", files.qrels, "--json")
    assert r.exit_code == 0, r.output
    report = json.loads(r.stdout)
    status = {c["name"]: c["status"] for c in report["checks"]}
    assert report["ok"] is True and report["level"] == "binding"
    for name in ("binding.corpus", "binding.queries", "binding.reference", "binding.evidence_queries"):
        assert status[name] == "pass", name


def test_verify_exits_one_when_the_vectors_are_not_the_certified_ones(selected_run, files):
    r = invoke("verify", selected_run.artifact, "--vectors", files.other)
    assert r.exit_code == 1
    assert "[fail   ] binding.corpus" in r.output and "FAILED: 1 check(s) failed" in r.output
    j = invoke("verify", selected_run.artifact, "--vectors", files.other, "--json")
    assert j.exit_code == 1 and json.loads(j.stdout)["ok"] is False


def test_verify_exits_one_for_labels_the_certificate_was_not_issued_for(selected_run, files, tmp_path):
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"q00000": ["d000001"]}), encoding="utf-8")
    r = invoke("verify", selected_run.artifact, "--qrels", wrong)
    assert r.exit_code == 1 and "[fail   ] binding.reference" in r.output


def test_verify_exits_one_for_a_modified_artifact(selected_run, tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    for f in selected_run.artifact.iterdir():
        (copy / f.name).write_bytes(f.read_bytes())
    cert = copy / "certificate.json"
    cert.write_text(cert.read_text(encoding="utf-8").replace('"target": 0.9', '"target": 0.5'),
                    encoding="utf-8")
    r = invoke("verify", copy)
    assert r.exit_code == 1
    assert "[fail   ] artifact files" in r.output and "[fail   ] digest" in r.output
    assert "does not match its manifest digest" in r.output


def test_verify_takes_a_certificate_file_and_an_evidence_file(selected_run):
    cert = selected_run.artifact / "certificate.json"
    alone = json.loads(invoke("verify", cert, "--json").stdout)
    assert alone["ok"] is True and alone["level"] == "integrity"
    replayed = json.loads(invoke("verify", cert, "--evidence", selected_run.artifact / "evidence.npz",
                                 "--json").stdout)
    assert replayed["ok"] is True and replayed["level"] == "replay"


def test_verify_with_a_key_file_checks_the_signature(selected_run, tmp_path):
    key = bytes(range(32))
    signed = sign_certificate(Certificate.load(selected_run.artifact / "certificate.json"), key,
                              key_id="cli-test")
    path = signed.save(tmp_path / "signed.json")
    good, bad = tmp_path / "good.key", tmp_path / "bad.key"
    good.write_bytes(key)
    bad.write_bytes(bytes(32))
    ok = json.loads(invoke("verify", path, "--key-file", good, "--json").stdout)
    assert {c["name"]: c["status"] for c in ok["checks"]}["signature"] == "pass"
    unkeyed = json.loads(invoke("verify", path, "--json").stdout)
    assert {c["name"]: c["status"] for c in unkeyed["checks"]}["signature"] == "skipped"
    wrong = invoke("verify", path, "--key-file", bad)
    assert wrong.exit_code == 1 and "[fail   ] signature" in wrong.output


def test_verify_says_when_a_record_names_a_rule_that_is_not_the_librarys(tmp_path):
    from .test_core_certificate import DeclaresCertifying, issue
    from .test_core_collect_evidence import make_problem, small_space

    out = issue(make_problem(), small_space(), rule=DeclaresCertifying(), target=0.8)
    path = out.certificate.save(tmp_path / "custom.json")
    r = invoke("verify", path)
    assert r.exit_code == 0, r.output
    assert "[skipped] consistency.decision" in r.output and "[skipped] consistency.level" in r.output
    assert "note: this record names a rule" in r.output
    assert "nothing in this record is certified" in r.output
    notes = json.loads(invoke("verify", path, "--json").stdout)["notes"]
    assert notes and "nothing in this record is certified" in notes[0]
    forged = out.certificate.to_dict()  # claim certified for it: failed, never skipped
    forged["result"]["certified"] = True
    forged["rule"]["certifying"] = True
    forged["digest"] = content_digest(forged)
    bad = invoke("verify", Certificate.from_dict(forged).save(tmp_path / "forged.json"))
    assert bad.exit_code == 1
    assert "[fail   ] consistency.decision" in bad.output and "[fail   ] rule.text" in bad.output
    assert "note: this record says certified: true for a rule" in bad.output


def test_strict_code_turns_a_different_source_from_information_into_a_failure(selected_run, tmp_path):
    record = Certificate.load(selected_run.artifact / "certificate.json").to_dict()
    record["code"]["files"]["scoring.py"] = "0" * 64  # as if issued by another version of the code
    record["code"]["digest"] = digest(record["code"]["files"])
    record["digest"] = content_digest(record)
    path = Certificate.from_dict(record).save(tmp_path / "other-code.json")
    lenient = invoke("verify", path)
    assert lenient.exit_code == 0 and "[differs] code.files" in lenient.output
    strict = invoke("verify", path, "--strict-code")
    assert strict.exit_code == 1 and "[fail   ] code.files" in strict.output


def test_verify_of_something_that_is_not_there_or_not_a_certificate(tmp_path):
    missing = invoke("verify", tmp_path / "nothing")
    assert missing.exit_code == 2 and squash("does not exist") in squash(missing.output)
    junk = tmp_path / "junk.json"
    junk.write_text("not json", encoding="utf-8")
    r = invoke("verify", junk)
    assert r.exit_code == 1 and "cannot load" in r.stderr
    j = invoke("verify", junk, "--json")
    assert j.exit_code == 1 and json.loads(j.stdout)["ok"] is False
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert invoke("verify", empty_dir).exit_code == 1


def test_verify_exit_codes_are_documented_and_json_is_printed_for_codes_0_and_1(
        selected_run, tmp_path):
    help_text = squash(invoke("verify", "--help").output)
    for phrase in ("0=verified", "1=verificationfailed", "unreadableormalformed",
                   "2=usageerror", "paththatdoesnotexist"):
        assert phrase in help_text
    ok = invoke("verify", selected_run.artifact, "--json")
    assert ok.exit_code == 0 and json.loads(ok.stdout)["ok"] is True
    bad_evidence = tmp_path / "evidence.npz"
    bad_evidence.write_bytes(b"not an npz")  # exists, cannot be read: 1, with the JSON object
    r = invoke("verify", selected_run.artifact, "--evidence", bad_evidence, "--json")
    assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False
    assert invoke("verify", selected_run.artifact, "--evidence", bad_evidence).exit_code == 1
    assert invoke("verify", tmp_path / "nothing", "--json").exit_code == 2


# ------------------------------------------------------------ the real entry point


@needs_scipy
def test_python_dash_m_optivision_cli_certifies_and_verifies(files, tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    out = tmp_path / "artifact"
    done = subprocess.run(
        [sys.executable, "-m", "optivision.cli", "certify", files.docs, files.queries, "--qrels", files.qrels,
         "--target", "0.9", "--json", "--out", str(out)],
        capture_output=True, text=True, env=env, check=False, timeout=300)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["outcome"] == "select"
    check = subprocess.run(
        [sys.executable, "-m", "optivision.cli", "verify", str(out), "--vectors", files.docs, "--json"],
        capture_output=True, text=True, env=env, check=False, timeout=300)
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)["ok"] is True
