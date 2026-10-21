"""Easy mode, the practice check, the fixed sequence and the real-query gate
(``optivision.easy._veto``; FINAL_DESIGN 5.5.1, 5.5.2), and the strength table
(``scripts/make_veto_strength.py`` -> ``presets/veto_strength.json``).

Planted faults as in V5a (FINAL_DESIGN 8): random scores, ``project(16) > binary`` and
``adaptive_merge(0.6) > int4`` on synthetic vectors (``optivision.testing.synthetic``) behind a
``CallableAdapter``. Synthetic data exercise code paths; they say nothing about real encoders.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from optivision.adapters import CallableAdapter
from optivision.compose import Pipeline
from optivision.core.retrieve.exact import CompressedRetriever, ExactRetriever
from optivision.easy import _docs, _probes, _veto
from optivision.easy._probes import Probe
from optivision.easy._types import LayoutRow
from optivision.easy._veto import (
    CheckStat,
    bootstrap_upper,
    fixed_sequence,
    ndcg_at,
    practice_check,
    query_labels,
    real_query_check,
    strength,
    strength_line,
)
from optivision.testing import make_corpus, make_queries

from .easy_docs_fixtures import page_records, synthetic_texts, word_adapter
from .test_docs_examples import BANNED_PHRASES, BANNED_WORDS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_veto_strength.py"
EXTRA_BANNED = ("lossless", "universal", "guaranteed", "first", "sota", "state of the art", "veto")

ADAPTIVE = {"stage": "adaptive_merge", "radius": 0.6, "ratio": None, "max_vectors": None, "min_vectors": 1,
            "refine": 0, "representative": "mean", "seeding": "farthest", "n_importance_seeds": 0, "center": None}
FAULTS = {
    "project(16) > binary": [{"stage": "project", "dim": 16, "method": "pca"}, {"stage": "binary"}],
    "adaptive_merge(0.6) > int4": [ADAPTIVE, {"stage": "int4"}],
    "random scores": "random",
}
HEALTHY = {
    "float16": [{"stage": "float16"}],
    "int8(per_vector)": [{"stage": "int8", "scale": "per_vector"}],
}


def row(name: str, chain: str, spec=None, plain: str | None = None) -> LayoutRow:
    return LayoutRow(name=name, plain=plain or name, spec={"stages": spec if isinstance(spec, list) else []},
                     chain=chain, bytes_per_page=None, ratio_f16=None, ratio_f32=None, lowest_kept=None,
                     worst_point=None)


FLOAT16 = row("float16", "float", [{"stage": "float16"}], plain="float16 (uncompressed)")


def rank_with(corpus, queries, spec, k: int = 20, seed: int = 5) -> dict[str, list[str]]:
    """Each query's top ``k`` page ids under a layout (``"random"``: random scores)."""
    if spec == "random":
        rng = np.random.default_rng(seed)
        return {q: list(rng.permutation(corpus.ids)[:k]) for q in queries.ids}
    pipe = Pipeline.from_dict({"stages": spec})
    store = pipe.compress(corpus, fit=True)
    ranking = CompressedRetriever(store, pipe).search(queries, k)
    return {q: list(ranking.ids[i]) for i, q in enumerate(queries.ids)}


def assert_clean(text: str) -> None:
    low = text.lower()
    for phrase in BANNED_PHRASES + EXTRA_BANNED:
        assert phrase not in low, (phrase, text)
    assert BANNED_WORDS.search(low) is None, text
    assert text.isascii(), text


# ---------------------------------------------------------------------------- nDCG


def test_ndcg_at_five():
    assert ndcg_at(["a", "b", "c"], {"a"}) == 1.0
    assert ndcg_at(["b", "a"], {"a"}) == pytest.approx(1 / math.log2(3))
    assert ndcg_at(["b", "c", "d", "e", "f", "a"], {"a"}) == 0.0  # rank 6 is outside the top 5
    assert ndcg_at(["a", "x", "b"], {"a", "b"}) == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert ndcg_at([("b", 2.0), ("a", 1.0)], {"a"}) == pytest.approx(1 / math.log2(3))  # (id, score) pairs
    assert ndcg_at(["b", "b", "a"], {"a"}) == pytest.approx(1 / math.log2(3))  # a repeated id counts once
    assert ndcg_at(["a"], set()) == 0.0
    assert ndcg_at(["x", "a"], {"a"}, k=1) == 0.0


# ---------------------------------------------------------------------------- the check


def _synthetic_rankings(n: int = 120, depth: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    pages = [f"d/x.pdf::p{i}" for i in range(400)]
    ref, relevant, clusters = {}, {}, {}
    for i in range(n):
        q = f"P{i:05d}"
        order = list(rng.permutation(pages)[:depth])
        rank = int(rng.choice([0, 0, 0, 1, 2]))
        relevant[q] = frozenset([order[rank]])
        clusters[q] = order[rank]
        ref[q] = order
    return ref, relevant, clusters


def test_quiet_on_identical_rankings():
    ref, relevant, clusters = _synthetic_rankings()
    stat = practice_check(ref, ref, relevant, clusters, 0.97)
    assert (stat.R, stat.U, stat.n, stat.passed) == (1.0, 1.0, 120, True)
    assert stat.kind == "probe" and stat.B == 2000 and stat.seed == 0
    assert stat.n_clusters == len(set(clusters.values()))  # questions of one page share a cluster


def test_fires_on_a_collapse():
    ref, relevant, clusters = _synthetic_rankings()
    rng = np.random.default_rng(1)
    shuffled = {q: list(rng.permutation(r)) for q, r in ref.items()}
    stat = practice_check(shuffled, ref, relevant, clusters, 0.97)
    assert not stat.passed and stat.R < 0.6 and stat.U < 0.97


def test_bootstrap_is_seeded():
    ref, relevant, clusters = _synthetic_rankings(seed=3)
    rng = np.random.default_rng(2)
    cand = {q: (list(rng.permutation(r)) if rng.random() < 0.1 else r) for q, r in ref.items()}
    a = practice_check(cand, ref, relevant, clusters, 0.97)
    b = practice_check(cand, ref, relevant, clusters, 0.97)
    c = practice_check(cand, ref, relevant, clusters, 0.97, seed=1)
    assert a == b
    assert a.R == c.R and a.U != c.U and abs(a.U - c.U) < 0.02
    cs, bs = np.random.default_rng(4).random(50), np.ones(50)
    assert bootstrap_upper(cs, bs, seed=7) == bootstrap_upper(cs, bs, seed=7)


def test_clusters_are_resampled_whole():
    ref, relevant, _ = _synthetic_rankings(n=90)
    by_file = {q: f"file{i % 9}" for i, q in enumerate(sorted(ref))}
    stat = practice_check(ref, ref, relevant, by_file, 0.97)
    assert stat.n == 90 and stat.n_clusters == 9
    assert practice_check(ref, ref, relevant, None, 0.97).n_clusters == 90


def test_aliases_map_repeated_pages_to_their_first_copy():
    # the candidate lists the repeated copy "a2" where the reference lists "a"
    ref = {"P00000": ["a", "b", "c"]}
    cand = {"P00000": ["a2", "b", "c"]}
    relevant = {"P00000": {"a"}}
    assert practice_check(cand, ref, relevant, None, 0.97, B=10).R == 0.0
    assert practice_check(cand, ref, relevant, None, 0.97, B=10, aliases={"a2": "a"}).R == 1.0
    # both copies in the relevant set would raise the ideal DCG above what one stored copy can reach
    ref = {"P00000": ["a", "a2", "c"]}
    cand = {"P00000": ["a", "b", "c"]}
    relevant = {"P00000": {"a", "a2"}}
    assert practice_check(cand, ref, relevant, None, 0.97, B=10).R < 0.7
    assert practice_check(cand, ref, relevant, None, 0.97, B=10, aliases={"a2": "a"}).R == 1.0
    with pytest.raises(KeyError, match="no candidate ranking"):
        practice_check({}, ref, relevant, None, 0.97)


def test_no_full_precision_gain_passes_as_uninformative():
    stat = practice_check({"q": ["x"]}, {"q": ["y"]}, {"q": {"z"}}, None, 0.97)
    assert stat.passed and not stat.informative and math.isnan(stat.R)


# ---------------------------------------------------------------------------- fixed sequence


def test_fixed_sequence_picks_the_first_layout_that_passes():
    ref, relevant, clusters = _synthetic_rankings()
    rng = np.random.default_rng(1)
    broken = {q: list(rng.permutation(r)) for q, r in ref.items()}
    a, b, c = row("hm(0.25) > binary", "1-bit"), row("hm(0.25) > int8", "8-bit"), row("int8(per_vector)", "8-bit")
    outcome = {a.name: broken, b.name: broken, c.name: ref}
    called = []

    def rank_fn(r):
        called.append(r.name)
        return outcome[r.name]

    res = fixed_sequence([a, b, c, FLOAT16], rank_fn, ref, relevant, clusters, 0.97)
    assert res.chosen is c and called == [a.name, b.name, c.name]
    assert [r.name for r, _ in res.steps] == [a.name, b.name, c.name]
    assert [s.passed for _, s in res.steps] == [False, False, True]
    assert res.stepped_down and not res.floor and not res.exhausted
    assert res.failed == [a, b]

    called.clear()
    first = fixed_sequence([c, FLOAT16], rank_fn, ref, relevant, clusters, 0.97)
    assert first.chosen is c and not first.stepped_down and called == [c.name]


def test_fixed_sequence_floor_is_float16_and_never_tested():
    ref, relevant, clusters = _synthetic_rankings()
    broken = {q: list(reversed(r)) for q, r in ref.items()}
    a = row("int8(per_vector)", "8-bit")

    def rank_fn(r):
        assert r.chain != "float", "float16 is the reference and is never ranked"
        return broken

    res = fixed_sequence([a, FLOAT16], rank_fn, ref, relevant, clusters, 0.97)
    assert res.chosen is FLOAT16 and res.floor and len(res.steps) == 1
    only_float = fixed_sequence([FLOAT16], rank_fn, ref, relevant, clusters, 0.97)
    assert only_float.chosen is FLOAT16 and only_float.steps == [] and only_float.lines() == []
    stuck = fixed_sequence([a], rank_fn, ref, relevant, clusters, 0.97)
    assert stuck.exhausted and stuck.chosen is a
    text = "\n".join(ln.render() for ln in stuck.lines())
    assert "every layout offered (int8(per_vector)) fell below 97% on 120 hard questions" in text
    assert "none is safer, so int8(per_vector) is kept" in text and "safer int8" not in text
    assert_clean(text)
    with pytest.raises(ValueError):
        fixed_sequence([], rank_fn, ref, relevant, clusters, 0.97)


def test_sequence_lines_are_tagged_and_worded_as_designed():
    ref, relevant, clusters = _synthetic_rankings(n=137)
    broken = {q: list(reversed(r)) for q, r in ref.items()}
    merged = row("hm(0.25) > int8", "8-bit", plain="merged (1 in 4 vectors) + 8-bit")
    centred = row("int4(mean)", "centred 4-bit", plain="centred 4-bit")

    passed = fixed_sequence([centred, FLOAT16], lambda r: ref, ref, relevant, clusters, 0.97).lines()
    assert [ln.render() for ln in passed] == [
        "Practice   passed: 137 hard questions made from your PDFs' text  [questions from your PDFs]"]

    down = fixed_sequence([merged, centred, FLOAT16], lambda r: broken if r is merged else ref, ref, relevant,
                          clusters, 0.97).lines()
    assert [ln.render() for ln in down] == [
        ("Practice   merged (1 in 4 vectors) + 8-bit fell below 97% on 137 hard questions from your PDFs; "
         "we used the safer centred 4-bit  [questions from your PDFs]")]

    floor = fixed_sequence([merged, centred, FLOAT16], lambda r: broken, ref, relevant, clusters, 0.975).lines()
    text = "\n".join(ln.render() for ln in floor)
    assert "fell below 97.5% on 137 hard questions" in text and "we stored float16 (uncompressed)" in text
    assert "see report() and ov.doctor()" in text
    for ln in passed + down + floor:
        assert_clean(ln.render())


def test_hard_set_sequence_refuses_a_changed_set():
    probes = [Probe(f"P{i:05d}", f"t {i}", "span", f"d/x.pdf::p{i}", frozenset([f"d/x.pdf::p{i}"]))
              for i in range(70)]
    ref = {p.id: [(p.page_id, 2.0)] + [(f"o{j}", 1.0 - 0.01 * j) for j in range(19)] for p in probes}
    hs = _probes.hard_set(probes, ref)
    res = hs.sequence([row("int8(per_vector)", "8-bit"), FLOAT16], lambda r: hs.ref_ids(), 0.97)
    assert res.chosen.name == "int8(per_vector)"
    object.__setattr__(hs, "probes", hs.probes[1:])
    with pytest.raises(ValueError, match="changed after it was frozen"):
        hs.sequence([FLOAT16], lambda r: {}, 0.97)


# ---------------------------------------------------------------------------- planted faults (V5a style)


def _vector_unit(seed: int):
    """A synthetic collection behind a CallableAdapter: 240 pages, 150 practice questions and
    150 separate "real" questions with planted relevance."""
    corpus = make_corpus(240, 32, tokens=(20, 40), seed=seed, mean_shift=1.5)
    practice, practice_rel = make_queries(corpus, 150, tokens=(3, 6), noise=0.6, seed=seed + 100)
    real, real_rel = make_queries(corpus, 150, tokens=(3, 6), noise=0.6, seed=seed + 200)
    vectors = {pid: corpus.vectors[corpus.offsets[i]:corpus.offsets[i + 1]] for i, pid in enumerate(corpus.ids)}
    queries = {f"practice/{q}": practice.vectors[practice.offsets[i]:practice.offsets[i + 1]]
               for i, q in enumerate(practice.ids)}
    queries.update({f"real/{q}": real.vectors[real.offsets[i]:real.offsets[i + 1]] for i, q in enumerate(real.ids)})
    adapter = CallableAdapter(lambda ids: [vectors[i] for i in ids], lambda texts: [queries[t] for t in texts])
    probes = [Probe(q, f"practice/{q}", "span", next(iter(r)), frozenset(r)) for q, r in practice_rel.items()]
    reals = [Probe(f"R{q}", f"real/{q}", "span", next(iter(r)), frozenset(r)) for q, r in real_rel.items()]
    return adapter, corpus.ids, probes, reals


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_planted_faults_fire_and_healthy_layouts_stay_quiet(seed):
    adapter, ids, probes, reals = _vector_unit(seed)
    docs = adapter.encode_documents(ids, ids)
    pq = _probes.encode_probes(adapter, probes)
    rq = _probes.encode_probes(adapter, reals)
    ref_p = rank_with(docs, pq, [])
    ref_r = rank_with(docs, rq, [])
    rel_p = {p.id: p.relevant for p in probes}
    rel_r = {p.id: p.relevant for p in reals}
    clusters = {p.id: p.page_id for p in probes}
    fired = []
    for name, spec in {**FAULTS, **HEALTHY}.items():
        check = practice_check(rank_with(docs, pq, spec), ref_p, rel_p, clusters, 0.97)
        real = practice_check(rank_with(docs, rq, spec), ref_r, rel_r, None, 0.97)
        if name in FAULTS:
            assert real.R < 0.80, (name, real.R)  # the fault does real damage on this unit
            fired.append(not check.passed)
        else:
            assert check.passed and real.R >= 0.99, (name, check, real.R)
    assert all(fired)


def test_full_path_from_text_pages_steps_down_past_a_fault():
    """Text pages -> Tier A probes -> encode_queries -> float16 reference -> hard set -> fixed sequence."""
    pages = page_records(synthetic_texts(150, seed=0))
    profile = _docs.profile(pages)
    assert profile.checkable
    probes = _probes.make_probes(pages, profile=profile)
    adapter = word_adapter(shift=2.0, documents_allowed=True)
    docs = adapter.encode_documents([p.text for p in pages], [p.id for p in pages])
    queries = _probes.encode_probes(word_adapter(shift=2.0, documents_allowed=False), probes)
    ranking = ExactRetriever(docs).search(queries, 20)
    ref = {q: list(zip(ranking.ids[i], ranking.scores[i].tolist())) for i, q in enumerate(queries.ids)}
    hard = _probes.hard_set(probes, ref, aliases=profile.aliases)
    assert hard.checked and hard.n == 60 and hard.n_kept == len(probes)
    hard_queries = queries.select([queries.ids.index(pid) for pid in hard.ids])

    fault = row("adaptive_merge(0.6) > int4", "4-bit", FAULTS["adaptive_merge(0.6) > int4"])
    noise = row("random scores", "1-bit")
    int8 = row("int8(per_vector)", "8-bit", HEALTHY["int8(per_vector)"])

    def rank_fn(r):
        return rank_with(docs, hard_queries, "random" if r is noise else r.spec["stages"])

    res = hard.sequence([noise, fault, int8, FLOAT16], rank_fn, 0.97)
    assert res.chosen is int8
    assert [s.passed for _, s in res.steps] == [False, False, True]
    floor = hard.sequence([noise, fault, FLOAT16], rank_fn, 0.97)
    assert floor.chosen is FLOAT16 and floor.floor


# ---------------------------------------------------------------------------- real questions


def test_real_query_gate_without_answers_uses_agreement():
    ref, _, _ = _synthetic_rankings(n=80)
    relevant, clusters, kind = query_labels(ref)
    assert kind == "agreement" and all(relevant[q] == {ref[q][0]} for q in ref) and clusters["P00003"] == "P00003"
    stat = real_query_check(ref, ref, 0.97)
    assert stat.passed and stat.R == 1.0 and stat.kind == "agreement" and stat.n == 80
    line = stat.retention_line().render()
    assert line == ("Questions  on 80 of your questions, retention against the full-precision top page was "
                    "100.0%  [your questions: agreement with full precision, not relevance]")
    swapped = {q: [r[1], r[0]] + list(r[2:]) for q, r in ref.items()}  # every top page moves to rank 2
    low = real_query_check(swapped, ref, 0.97)
    assert not low.passed and low.R == pytest.approx(1 / math.log2(3))
    assert_clean(line)


def test_real_query_gate_with_answers_measures_search_quality():
    ref, _, _ = _synthetic_rankings(n=70)
    answers = {}
    for i, (q, r) in enumerate(sorted(ref.items())):
        doc, page = r[1].rsplit("::p", 1)
        answers[q] = (doc, int(page)) if i % 2 else {r[1], "d/x.pdf::p999"}
    relevant, _, kind = query_labels(ref, answers)
    assert kind == "search quality" and relevant["P00001"] == {ref["P00001"][1]}
    with pytest.raises(TypeError, match="answers must map"):
        query_labels(ref, list(answers.values()))
    stat = real_query_check(ref, ref, 0.97, answers=answers)
    assert stat.kind == "search quality" and stat.R == 1.0 and stat.passed
    assert stat.retention_line().render().endswith("[your questions: search quality]")
    res = fixed_sequence([row("int8(per_vector)", "8-bit"), FLOAT16], lambda r: ref, ref, relevant,
                         {q: q for q in relevant}, 0.97, kind="search quality")
    assert [ln.render() for ln in res.lines()] == [
        "Questions  passed: 70 of your questions  [your questions: search quality]"]


def test_real_query_gate_needs_sixty_questions():
    ref, _, _ = _synthetic_rankings(n=59)
    with pytest.raises(ValueError, match="at least 60"):
        real_query_check(ref, ref, 0.97)


# ---------------------------------------------------------------------------- strength


def _script():
    spec = importlib.util.spec_from_file_location("make_veto_strength", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_veto_strength"] = module
    spec.loader.exec_module(module)
    return module


def test_strength_table_was_generated_by_the_current_script():
    table = json.loads((ROOT / "src" / "optivision" / "presets" / "veto_strength.json").read_text(encoding="utf-8"))
    assert table == _veto.strength_table()
    script = _script()
    assert table["script_sha256"] == script._digest(SCRIPT)
    assert table["settings"]["B"] == _veto.B_DEFAULT and table["settings"]["level"] == _veto.LEVEL
    assert table["settings"]["n"] == [60, 100, 137, 200, 300]
    cell = next(c for c in table["cells"]
                if (c["model"], c["target"], c["retention"], c["n"]) == ("loss", 0.97, 0.9, 60))
    assert round(script.simulate_cell("loss", 0.97, 0.9, 60), 3) == cell["fire_rate"]


def test_strength_table_matches_the_design_table():
    """FINAL_DESIGN 5.5.1 at T = 0.97 (B = 1,000 there): false alarms of a few percent; a fall to
    0.90 caught in 62-72% of runs at n = 60 and 98-99% at n = 200 (simulation noise allowed)."""
    rows = {e["n"]: e for e in _veto.strength_table()["summary"] if e["target"] == 0.97}
    assert all(rows[n]["false_alarm"][1] <= 0.05 for n in rows)
    caught = {n: {f["retention"]: f["caught"] for f in rows[n]["falls"]} for n in rows}
    assert 0.55 <= caught[60][0.9][0] and caught[60][0.9][1] <= 0.78
    assert caught[200][0.9][0] >= 0.95
    lows = [caught[n][0.9][0] for n in sorted(caught)]
    assert lows == sorted(lows)


def test_script_bootstrap_is_the_one_the_check_uses():
    script = _script()
    rng = np.random.default_rng(9)
    c = (rng.random(80) < 0.9) / np.log2(rng.integers(1, 6, size=80) + 1.0)
    b = np.where(c > 0, c, 1.0)
    counts = script.resample_counts(np.random.default_rng(11), 80, 2000)
    via_script = float(script.upper_bounds(c[None, :], b[None, :], counts)[0])
    assert via_script == pytest.approx(bootstrap_upper(c, b, B=2000, seed=11), abs=1e-12)


def test_strength_line():
    line = strength_line(150, 0.97)
    entry = strength(150, 0.97)
    assert entry["n"] == 137 and entry["target"] == 0.97
    lo, hi = entry["falls"][-1]["caught"]
    a, b = math.floor(100 * lo + 1e-9), math.floor(100 * hi + 1e-9)
    caught = f"{a}%" if a == b else f"{a}-{b}%"
    assert line.render() == (f"Strength   at 137 questions a simulated fall to 90% of full-precision quality is caught "
                             f"in {caught} of runs; whether real losses show up in practice questions is untested (V5)"
                             f"  [estimate: simulation]")
    text = line.render()
    assert {c.kind for c in line.claims} == {"estimate"}
    assert_clean(text)
    assert strength_line(59, 0.97) is None
    # the largest simulated target at or below the user's: the rate shown can only understate
    assert strength(1000, 0.90)["n"] == 300 and strength(1000, 0.91)["target"] == 0.90
    assert strength(60, 0.96)["target"] == 0.95 and strength(60, 0.97)["target"] == 0.97
    assert strength(60, 0.99)["target"] == 0.98


def test_check_stat_lines_carry_one_kind_each():
    stat = CheckStat(R=0.981, U=0.995, n=137, passed=True, target=0.97)
    line = stat.retention_line("Practice")
    assert line.render() == ("Practice   on 137 hard questions made from your PDFs, retention against full precision "
                             "was 98.1%  [questions from your PDFs]")
    assert stat.as_dict()["level"] == 0.95
