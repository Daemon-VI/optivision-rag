"""Font-scale control for the one-bit cost.

E1 (tiled ColSmol, 11pt code) loses ~12 nDCG points to sign codes; the same
encoder with tiling off (blind to the code) loses ~3, and so does ColPali at
448px (E3). Two readings: (a) the codec cost follows how legible / concentrated
the discriminative evidence is, (b) it is a property of the encoder geometry.
Control: re-render the corpus with ONLY the unique code drawn 3x larger
(and, mirror, 0.4x smaller), everything else byte-identical, and re-encode with
the tiled ColSmol. (a) predicts 3x -> smaller loss, 0.4x -> loss collapses
toward the untiled ~3 pts with R@1 at the subject floor; (b) predicts ~12 both.

Runs on the CPU caches in seconds. Missing caches are skipped.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Rishi/optivision-rag/src")
from optivision.bench import EncodedCorpus  # noqa: E402
from optivision.metrics import ndcg_at_k  # noqa: E402

ROOT = Path("C:/Users/Rishi/optivision-rag")
SCR = Path("C:/Users/MPPSKA~1/AppData/Local/Temp/claude/C--Users-Rishi/2d9f7724-afc4-44a7-8036-3b2977a3e11b/scratchpad")
os.chdir(ROOT)
rng = np.random.default_rng(7)

def _run(label, tag, corpus):
    return (label, ROOT / f"data/cache/colsmol_{tag}.npz", ROOT / corpus, ROOT / f"reports/colsmol_{tag}/benchmark.json")


RUNS = [
    ("E1 tiled, code 1x (11pt)", ROOT / "data/cache/colsmol.npz", ROOT / "data/corpus", ROOT / "reports/colsmol/benchmark.json"),
    _run("tiled, code 3x (33pt)", "code3x", "data/corpus_code3x"),
    _run("tiled, code 0.4x (4.4pt)", "code04x", "data/corpus_code04x"),
    ("untiled, code 1x (blind)", SCR / "colsmol_notile.npz", ROOT / "data/corpus", None),
    # first-round variants, kept for what they show about layout sensitivity:
    # 3x drawn in the text flow pushed every line below it down 58 pt (~2.2
    # patch rows); 0.4x also shrank the field label.
    _run("3x in-flow, lines shifted", "code3x_shifted", "data/corpus_code3x_shifted"),
    _run("0.4x incl. label", "code04x_label", "data/corpus_code04x_label"),
]


def token_type(e):
    t = np.full(e.embeddings.shape[0], "grid", dtype="<U6")
    ti = e.text_token_index
    if ti.size == 0:
        return t
    runs = np.split(ti, np.flatnonzero(np.diff(ti) != 1) + 1)
    if not any(len(r) >= 64 for r in runs):  # untiled: no thumbnail, just the text tail
        t[ti] = "instr"
        return t
    for r in runs:
        t[r] = "thumb" if len(r) >= 64 else "marker"
    if len(runs[-1]) >= 64:
        t[runs[-1][64:]] = "instr"
    return t


def load(cache, corpus_dir):
    corpus = EncodedCorpus.load(cache)
    zq = np.load(cache.with_suffix(".queries.npz"), allow_pickle=True)
    texts = json.loads(str(zq["texts"]))
    queries = [np.asarray(zq[f"q_{i}"], np.float32) for i in range(len(texts))]
    rel = json.loads((corpus_dir / "queries.json").read_text())
    by_text = {r["query"]: r for r in rel}
    rel = [by_text[t] for t in texts]
    qrels = [set(r["relevant"]) for r in rel]
    family = np.array([r["type"] for r in rel])
    ids = [e.ref.page_id for e in corpus.encodings]
    pages = [e.embeddings for e in corpus.encodings]
    types = [token_type(e) for e in corpus.encodings]
    return corpus, queries, qrels, family, ids, pages, types


def per_query(pages, queries, ids, qrels, q_fn=lambda q: q):
    nd, r1 = [], []
    for qi, q in enumerate(queries):
        qq = q_fn(q)
        sc = np.array([float((qq @ d.T).max(1).sum()) for d in pages])
        ranked = [ids[i] for i in np.argsort(-sc)]
        nd.append(ndcg_at_k(ranked, qrels[qi], 5))
        r1.append(ranked[0] in qrels[qi])
    return np.array(nd), np.array(r1, float)


def boot(a, b, n=4000):
    d = b - a
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def int8(p):
    return (np.clip(np.round(p / 0.5 * 127), -127, 127) * 0.5 / 127).astype(np.float32)


def margin_stats(pages, queries, ids, qrels, family):
    """Precise queries: gold-page score minus best non-gold score (normalised by
    the gold score), and how many query tokens carry 80% of the positive part
    of that margin. Evidence concentration = few tokens carry it."""
    margins, n80, wins = [], [], []
    for qi, q in enumerate(queries):
        if family[qi] != "precise":
            continue
        per_tok = np.stack([(q @ d.T).max(1) for d in pages])  # pages x qtok
        sc = per_tok.sum(1)
        gold = [i for i, pid in enumerate(ids) if pid in qrels[qi]]
        g = max(gold, key=lambda i: sc[i])
        others = np.array([i for i in range(len(ids)) if i not in gold])
        o = others[sc[others].argmax()]
        margins.append((sc[g] - sc[o]) / sc[g])
        wins.append(sc[g] > sc[o])
        delta = per_tok[g] - per_tok[o]
        pos = np.sort(delta[delta > 0])[::-1]
        if pos.size:
            n80.append(int(np.searchsorted(np.cumsum(pos), 0.8 * pos.sum()) + 1))
    return float(np.mean(margins)), float(np.mean(wins)), float(np.mean(n80)) if n80 else float("nan")


def win_shares(pages, types, queries, family, which):
    cnt = {k: 0 for k in ("grid", "thumb", "marker", "instr")}
    for qi, q in enumerate(queries):
        if family[qi] != which:
            continue
        for d, t in zip(pages, types):
            am = (q @ d.T).argmax(1)
            for k in cnt:
                cnt[k] += int((t[am] == k).sum())
    tot = max(sum(cnt.values()), 1)
    return {k: v / tot for k, v in cnt.items()}


results = {}
for label, cache, corpus_dir, bench in RUNS:
    if not cache.exists() or not cache.with_suffix(".queries.npz").exists():
        print(f"[skip] {label}: {cache} not on disk")
        continue
    corpus, queries, qrels, family, ids, pages, types = load(cache, corpus_dir)
    allv = np.concatenate(pages)
    mu = allv.mean(0)
    p_pos = (allv > 0).mean(0)
    h = -(p_pos * np.log2(np.clip(p_pos, 1e-9, 1)) + (1 - p_pos) * np.log2(np.clip(1 - p_pos, 1e-9, 1)))
    ev = np.linalg.eigvalsh(np.cov((allv - mu).T))
    geo = dict(mu_norm=float(np.linalg.norm(mu)), dead_bits=int((h < 0.5).sum()), pr=float(ev.sum() ** 2 / (ev ** 2).sum()))

    rows = {
        "float32": per_query(pages, queries, ids, qrels),
        "sign(d)": per_query([np.sign(p) for p in pages], queries, ids, qrels),
        "sign(d-mu)": per_query([np.sign(p - mu) for p in pages], queries, ids, qrels, q_fn=lambda q: q),
        "int8": per_query([int8(p) for p in pages], queries, ids, qrels),
    }
    base_nd = rows["float32"][0]
    pm, tm = family == "precise", family == "topical"
    r = {"geometry": geo, "tokens_per_page": int(pages[0].shape[0]), "codecs": {}}
    r["_pq"] = {n: (nd, r1) for n, (nd, r1) in rows.items()}  # per-query, for the paired section
    r["_enc"] = corpus.encodings
    r["_ids"] = ids
    for name, (nd, r1) in rows.items():
        m, lo, hi = boot(base_nd, nd)
        r["codecs"][name] = dict(ndcg5=float(nd.mean()), retain=float(nd.mean() / base_nd.mean()), r1=float(r1.mean()),
                                 ndcg5_precise=float(nd[pm].mean()), r1_precise=float(r1[pm].mean()),
                                 ndcg5_topical=float(nd[tm].mean()), r1_topical=float(r1[tm].mean()),
                                 delta_vs_float=[float(m), float(lo), float(hi)])
    r["margin"] = {
        "float32": margin_stats(pages, queries, ids, qrels, family),
        "sign(d)": margin_stats([np.sign(p) for p in pages], queries, ids, qrels, family),
    }
    r["wins_precise"] = win_shares(pages, types, queries, family, "precise")
    r["wins_topical"] = win_shares(pages, types, queries, family, "topical")
    if bench and bench.exists():
        b = json.loads(bench.read_text())
        r["bench"] = {row["variant"]: dict(ndcg5=row["ndcg@5"], r1=row["recall@1"], compression=row["compression_ratio"]) for row in b["rows"]}
    results[label] = r

# ----------------------------------------------------------------- report
print("\n=== one-bit cost vs code legibility (tiled ColSmol unless noted; 60 pages, 72 queries) ===")
print(f"{'run':<28} {'tok':>4}  float nDCG@5  sign nDCG@5 (retain)   delta [95% CI]      R@1 float->sign   precise nDCG/R@1 float -> sign   topical R@1")
for label, r in results.items():
    f, s = r["codecs"]["float32"], r["codecs"]["sign(d)"]
    m, lo, hi = s["delta_vs_float"]
    print(f"{label:<28} {r['tokens_per_page']:>4}  {f['ndcg5']:.4f}        {s['ndcg5']:.4f} ({100*s['retain']:5.1f}%)   {m:+.3f} [{lo:+.3f},{hi:+.3f}]   "
          f"{f['r1']:.3f} -> {s['r1']:.3f}      {f['ndcg5_precise']:.3f}/{f['r1_precise']:.3f} -> {s['ndcg5_precise']:.3f}/{s['r1_precise']:.3f}      "
          f"{f['r1_topical']:.3f} -> {s['r1_topical']:.3f}")
print("subject-only floor: R@1 0.333 overall (precise 0.2), nDCG@5 0.658")

print("\n=== other codecs, retention of float nDCG@5 ===")
print(f"{'run':<28}" + "".join(f"{n:>14}" for n in ("sign(d)", "sign(d-mu)", "int8")))
for label, r in results.items():
    print(f"{label:<28}" + "".join(f"{100*r['codecs'][n]['retain']:13.1f}%" for n in ("sign(d)", "sign(d-mu)", "int8")))

print("\n=== geometry (should NOT move if the cost is about evidence, not the encoder) ===")
print(f"{'run':<28} ||mu||  dead bits  participation ratio")
for label, r in results.items():
    g = r["geometry"]
    print(f"{label:<28} {g['mu_norm']:.3f}   {g['dead_bits']:>4}       {g['pr']:.1f}")

print("\n=== precise queries: gold-vs-best-distractor margin (fraction of gold score), win rate, #query tokens carrying 80% of the margin ===")
print(f"{'run':<28} {'float margin':>13} {'win':>6} {'n80':>5}   {'sign margin':>12} {'win':>6} {'n80':>5}")
for label, r in results.items():
    (fm, fw, fn), (sm, sw, sn) = r["margin"]["float32"], r["margin"]["sign(d)"]
    print(f"{label:<28} {fm:>+13.4f} {fw:>6.3f} {fn:>5.1f}   {sm:>+12.4f} {sw:>6.3f} {sn:>5.1f}")

print("\n=== which token types win the MaxSim argmax (share of query-token wins, all pages) ===")
print(f"{'run':<28}  precise: grid  thumb marker instr   | topical: grid  thumb marker instr")
for label, r in results.items():
    p, t = r["wins_precise"], r["wins_topical"]
    print(f"{label:<28}  {100*p['grid']:13.1f} {100*p['thumb']:6.1f} {100*p['marker']:6.1f} {100*p['instr']:5.1f}   | "
          f"{100*t['grid']:13.1f} {100*t['thumb']:6.1f} {100*t['marker']:6.1f} {100*t['instr']:5.1f}")

have_bench = [(l, r) for l, r in results.items() if "bench" in r]
if have_bench:
    variants = ("baseline-float32", "binary-only", "int8-only", "optivision", "prune+int8", "keep-30pct", "keep-10pct")
    print("\n=== bench --sweep rows, nDCG@5 (retention vs that corpus' float baseline) ===")
    print(f"{'run':<28}" + "".join(f"{v:>18}" for v in variants))
    for label, r in have_bench:
        b = r["bench"]
        base = b["baseline-float32"]["ndcg5"]
        print(f"{label:<28}" + "".join(f"{b[v]['ndcg5']:.3f} ({100*b[v]['ndcg5']/base:5.1f}%)".rjust(18) if v in b else "-".rjust(18) for v in variants))

ref_label = "E1 tiled, code 1x (11pt)"
if ref_label in results:
    ref = results[ref_label]
    print("\n=== paired against E1, per query (same 72 queries, same page ids): mean delta [95% bootstrap CI] ===")
    print(f"{'run':<28} {'float nDCG@5':>22} {'sign nDCG@5':>22} {'one-bit loss':>22} {'precise float R@1':>22} {'precise sign R@1':>22}")
    for label, r in results.items():
        if label == ref_label or r["_ids"] != ref["_ids"] or r["tokens_per_page"] != ref["tokens_per_page"]:
            continue
        f0, s0 = ref["_pq"]["float32"], ref["_pq"]["sign(d)"]
        f1, s1 = r["_pq"]["float32"], r["_pq"]["sign(d)"]
        pm = family == "precise"
        cells = []
        for d in (f1[0] - f0[0], s1[0] - s0[0], (s1[0] - f1[0]) - (s0[0] - f0[0]), (f1[1] - f0[1])[pm], (s1[1] - s0[1])[pm]):
            m, lo, hi = boot(np.zeros_like(d), d)
            cells.append(f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]")
            r.setdefault("paired_vs_E1", []).append([float(m), float(lo), float(hi)])
        print(f"{label:<28} " + " ".join(f"{c:>22}" for c in cells))
    print("  (one-bit loss = sign - float; a positive delta means the codec costs LESS than on E1)")

    print("\n=== locality: mean cos(E1 token, variant token) by grid row, top to bottom (the code sits in rows 3-4 and the footer in row 31) ===")
    rows_n = np.asarray(ref["_enc"][0].grid.token_index).shape[0]
    print(f"{'run':<28} " + " ".join(f"{i:>4}" for i in range(rows_n)))
    for label, r in results.items():
        if label == ref_label or r["_ids"] != ref["_ids"] or r["tokens_per_page"] != ref["tokens_per_page"]:
            continue
        acc = np.zeros(rows_n)
        for e0, e1 in zip(ref["_enc"], r["_enc"]):
            ti = np.asarray(e0.grid.token_index)
            acc += (e0.embeddings[ti] * e1.embeddings[ti]).sum(-1).mean(1)
        acc /= len(ref["_enc"])
        r["row_cos_vs_E1"] = [float(v) for v in acc]
        print(f"{label:<28} " + " ".join(f"{v:4.2f}" for v in acc))
    e0, e1 = ref["_enc"][0], ref["_enc"][1]
    ti = np.asarray(e0.grid.token_index)
    print(f"  reference: cos between two DIFFERENT E1 pages at the same cell, mean {(e0.embeddings[ti] * e1.embeddings[ti]).sum(-1).mean():.2f}")

for r in results.values():
    for k in ("_pq", "_enc", "_ids"):
        r.pop(k, None)
out = ROOT / "reports/fontscale_summary.json"
out.write_text(json.dumps(results, indent=1))
print(f"\nwrote {out}")
