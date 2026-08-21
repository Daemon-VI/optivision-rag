"""Q2: what should 'did compression change the retriever's mind' be measured
with? Runs the real pipeline variants from the E1 cache and scores each against
the float baseline with: tau-b over the corpus (the fix), the superseded
shared-top-10 tau-b, tau_AP (Yilmaz et al. 2008), RBO (Webber et al. 2010),
top-1 agreement, and top-k overlap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Rishi/optivision-rag/src")
from optivision.bench import EncodedCorpus, default_variants, keep_ratio_sweep, run_variant  # noqa: E402
from optivision.config import Config  # noqa: E402
from optivision.corpus import load_queries  # noqa: E402
from optivision.metrics import rank_correlation, rank_correlation_shared  # noqa: E402

ROOT = Path("C:/Users/Rishi/optivision-rag")
import os
os.chdir(ROOT)


def rbo(base: list[str], other: list[str], p: float) -> float:
    """Exact RBO for two full permutations of the same set (tail agreement = 1)."""
    n = len(base)
    seen_b, seen_o = set(), set()
    acc, inter = 0.0, 0
    for d in range(1, n + 1):
        b, o = base[d - 1], other[d - 1]
        if b == o:
            inter += 1
        else:
            if b in seen_o:
                inter += 1
            if o in seen_b:
                inter += 1
        seen_b.add(b); seen_o.add(o)
        acc += p ** (d - 1) * inter / d
    return (1 - p) * acc + p**n


def tau_ap(base: list[str], other: list[str]) -> float:
    """AP correlation: base is the reference ranking."""
    pos = {x: i for i, x in enumerate(other)}
    n = len(base)
    total = 0.0
    for i in range(1, n):
        above_ref = base[:i]
        c = sum(1 for x in above_ref if pos[x] < pos[base[i]])
        total += c / i
    return 2 * total / (n - 1) - 1


def main() -> None:
    cfg = Config.load(ROOT / "configs/colsmol.yaml") if (ROOT / "configs/colsmol.yaml").exists() else Config()
    cfg.pruning.min_keep = 8  # matches reports/colsmol config
    corpus = EncodedCorpus.load(ROOT / "data/cache/colsmol.npz")
    qids, texts, qrels = load_queries(ROOT / "data/corpus/queries.json")
    zq = np.load(ROOT / "data/cache/colsmol.queries.npz", allow_pickle=True)
    qv = [zq[f"q_{i}"] for i in range(len(texts))]
    ids = [e.ref.page_id for e in corpus.encodings]

    variants = default_variants() + keep_ratio_sweep((0.5, 0.3, 0.1))
    out = {}
    for v in variants:
        r = run_variant(corpus, cfg, v, qv, qids, qrels, top_k=10, workdir=ROOT / "data/tmp/q2")
        out[v.name] = r
    base = out["baseline-float32"]
    bd, bs = base["deep"], base["run"]
    print(f"{'variant':<22} {'nDCG@5':>7} {'retain':>7} | {'tau60':>6} {'tau10sh':>7} {'tauAP':>6} {'RBO.9':>6} {'RBO.98':>6} | {'top1=':>5} {'ov@5':>5} {'ov@10':>5} | {'gold@1 kept':>11}")
    for name, r in out.items():
        d, s = r["deep"], r["run"]
        t60 = np.mean([rank_correlation(bd[q], d[q], pool=ids) for q in qids])
        t10 = np.mean([rank_correlation_shared(bs[q], s[q]) for q in qids])
        tap = np.mean([tau_ap(bd[q], d[q]) for q in qids])
        r9 = np.mean([rbo(bd[q], d[q], 0.9) for q in qids])
        r98 = np.mean([rbo(bd[q], d[q], 0.98) for q in qids])
        top1 = np.mean([bd[q][0] == d[q][0] for q in qids])
        ov5 = np.mean([len(set(bd[q][:5]) & set(d[q][:5])) / 5 for q in qids])
        ov10 = np.mean([len(set(bd[q][:10]) & set(d[q][:10])) / 10 for q in qids])
        # of the queries the baseline got right at rank 1, how many does the variant keep right
        right = [q for q in qids if bd[q][0] in qrels[q]]
        kept = np.mean([d[q][0] in qrels[q] for q in right]) if right else float("nan")
        row = r["row"]
        print(f"{name:<22} {row['ndcg@5']:7.4f} {100*row['ndcg@5']/base['row']['ndcg@5']:6.1f}% | {t60:6.3f} {t10:7.3f} {tap:6.3f} {r9:6.3f} {r98:6.3f} | {top1:5.2f} {ov5:5.2f} {ov10:5.2f} | {kept:11.2f}")

    # chance levels on a 60-item corpus vs a 500-item one
    rng = np.random.default_rng(0)
    for N in (60, 500):
        a = [str(i) for i in range(N)]
        vals = []
        for _ in range(200):
            b = list(rng.permutation(a))
            vals.append((rank_correlation(a, b, pool=a), rbo(a, b, 0.9), tau_ap(a, b), len(set(a[:5]) & set(b[:5])) / 5))
        v = np.array(vals).mean(0)
        print(f"chance level, N={N}: tau {v[0]:.3f}  RBO.9 {v[1]:.3f}  tauAP {v[2]:.3f}  ov@5 {v[3]:.3f}")


if __name__ == "__main__":
    main()
