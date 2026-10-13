"""Q5/Q6 step 1: correctness of the experimental scorers (PLAN.md §4), before any timing.

Writes reports/research/Q5Q6/correctness.json and exits 1 if an exact path fails the
ranking-equality criterion (the latency step must then not run).

    OPTIVISION_DATA=../optivision-rag-v2/data PYTHONPATH="src;scripts/research" \
        python scripts/research/q56_correctness.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import e1_common as E
import numpy as np
import q56_common as C

from optivision.evaluation import per_query_metrics, retention
from optivision.scoring import rank
from optivision.storage import ExactIndex

OUT = E.ROOT / "reports" / "research" / "Q5Q6"
U = 2.0 ** -24


def rank_compare(a, b, ks=(1, 5, 10, 50)):
    ra, rb = rank(a), rank(b)
    full = np.all(ra == rb, axis=1)
    out = {"queries": len(ra), "identical_full_ranking": int(full.sum()),
           "identical_full_ranking_all": bool(full.all())}
    for k in ks:
        out[f"identical_top{k}_ordered"] = int(np.all(ra[:, :k] == rb[:, :k], axis=1).sum())
        out[f"top{k}_set_overlap_mean"] = float(np.mean([len(set(ra[i, :k]) & set(rb[i, :k])) / k for i in range(len(ra))]))
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    fin = np.isfinite(d)
    out["max_abs_score_difference"] = float(d[fin].max()) if fin.any() else 0.0
    out["max_rel_score_difference"] = float((d[fin] / np.maximum(np.abs(a.astype(np.float64))[fin], 1e-30)).max()) if fin.any() else 0.0
    out["bitwise_equal_scores"] = bool(np.array_equal(a, b))
    return out, ra, rb, full


def quality(scores, base, rel):
    pq = per_query_metrics(scores, rel, ["ndcg@5"])["ndcg@5"]
    point, lo, hi = retention(pq, base, n_boot=1000)
    return pq, {"ndcg@5": float(pq.mean()), "retention": point, "retention_ci": [lo, hi]}


def paired_diff(pa, pb, base, seed=0, nb=1000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(base), (nb, len(base)))
    bm = base[idx].mean(1)
    d = pa[idx].mean(1) / bm - pb[idx].mean(1) / bm
    return [float(pa.mean() / base.mean() - pb.mean() / base.mean()), *np.quantile(d, [0.025, 0.975]).tolist()]


def int8_reference(queries, i8, qi, page):
    """float64 MaxSim of query qi against one page from the int8 codes: sum_t max_v c_v * sum_j x_j q_vj."""
    x = queries.vectors[queries.offsets[qi]:queries.offsets[qi + 1]].astype(np.float64)
    lo, hi = int(i8.offsets[page]), int(i8.offsets[page + 1])
    d = i8.q[lo:hi].astype(np.float64) * i8.c[lo:hi, None].astype(np.float64)
    prods = np.abs(x)[:, None, :] * np.abs(d)[None, :, :]
    s = x @ d.T
    j = s.argmax(axis=1)
    bound = float(sum(prods[t, j[t]].sum() for t in range(len(x))))  # sum |terms| along the max path
    return float(s.max(axis=1).sum()), bound


def diagnose(a, b, ra, rb, full, ref_fn, limit=200):
    out = []
    for qi in np.flatnonzero(~full)[:limit]:
        pos = int(np.flatnonzero(ra[qi] != rb[qi])[0])
        p1, p2 = int(ra[qi, pos]), int(rb[qi, pos])
        r1, b1 = ref_fn(qi, p1)
        r2, b2 = ref_fn(qi, p2)
        tol = 2 * C.DIM * U * max(b1, b2) * 1.01  # deterministic float32 dot-product error bound (gamma_n), both paths
        out.append({"query": int(qi), "first_differing_rank": pos + 1, "pages": [p1, p2],
                    "current_scores": [float(a[qi, p1]), float(a[qi, p2])],
                    "new_scores": [float(b[qi, p1]), float(b[qi, p2])],
                    "float64_reference": [r1, r2], "reference_gap": r1 - r2, "rounding_tolerance": tol,
                    "within_rounding": bool(abs(r1 - r2) <= tol)})
    return out


def main():
    t_start = time.time()
    ds = C.load()
    corpus, queries, rel = ds.corpus, ds.queries, ds.relevant()
    n_docs = len(corpus)
    hot, hotp, cold = C.build(corpus)
    i8 = C.Int8Direct(cold)
    res = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=E.ROOT, check=True).stdout.strip(),
           "n_docs": n_docs, "n_queries": len(queries), "query_tokens": int(queries.num_vectors),
           "vectors": int(corpus.num_vectors), "hot_vectors": int(hot.num_vectors), "candidates": C.CANDIDATES,
           "environment": E.environment()}
    base_scores = ExactIndex(corpus).score(queries)
    base = per_query_metrics(base_scores, rel, ["ndcg@5"])["ndcg@5"]
    _, res["F32"] = quality(base_scores, base, rel)
    store = E.Store(E.load_store("colqwen2_docvqa"), "labels")
    res["baseline_equals_E1_store"] = bool(np.array_equal(base, store.base))

    # ---- Experiment A: full int8 scan
    cur = ExactIndex(cold).score(queries)
    new = C.maxsim(queries.vectors, queries.offsets, i8.offsets, i8.sims, n_docs)
    cmp_, ra, rb, full = rank_compare(cur, new)
    pq_cur, q_cur = quality(cur, base, rel)
    pq_new, q_new = quality(new, base, rel)
    res["A_full_int8"] = {"comparison": cmp_, "I8-cur": q_cur, "I8-dir": q_new,
                          "ndcg_identical_per_query": bool(np.array_equal(pq_cur, pq_new)),
                          "diagnosis": diagnose(cur, new, ra, rb, full,
                                                lambda qi, p: int8_reference(queries, i8, qi, p))}

    # ---- Experiment A: two-tier rescoring with the current hot tier
    tier_scores, _ = C.current_two_tier(hot, cold, queries)
    hot_cur = ExactIndex(hot).score(queries)
    sl_cur = rank(hot_cur, C.CANDIDATES)
    pq_tier_cur, q_tier_cur = quality(tier_scores, base, rel)
    res["current_two_tier"] = q_tier_cur
    res["current_two_tier_matches_Q4"] = abs(q_tier_cur["retention"] - 0.9996815249137015) < 1e-12
    for mode, key in (("direct", "R-dir"), ("decoded", "R-bdec")):
        cold_s = C.rescore_grouped(queries, sl_cur, i8, mode)
        fin = C.merge_tiers(hot_cur, sl_cur, cold_s)
        c2, ra2, rb2, full2 = rank_compare(tier_scores, fin)
        pq2, q2 = quality(fin, base, rel)
        res[f"A_rescore_{key}"] = {"comparison": c2, "quality": q2,
                                   "ndcg_identical_per_query": bool(np.array_equal(pq_tier_cur, pq2)),
                                   "diagnosis": diagnose(tier_scores, fin, ra2, rb2, full2,
                                                         lambda qi, p: int8_reference(queries, i8, qi, p))}

    exact_ok = (res["A_full_int8"]["comparison"]["identical_full_ranking_all"]
                and res["A_rescore_R-dir"]["comparison"]["identical_full_ranking_all"])

    # ---- Experiment B1: exact LUT binary hot tier
    lut = C.BinaryLUT(hot)
    hot_b1 = C.maxsim(queries.vectors, queries.offsets, lut.offsets, lut.sims, n_docs)
    sl_b1 = rank(hot_b1, C.CANDIDATES)
    hc, _, _, _ = rank_compare(hot_cur, hot_b1)
    fin_b1 = C.merge_tiers(hot_b1, sl_b1, C.rescore_grouped(queries, sl_b1, i8, "direct"))
    cb1, _, _, _ = rank_compare(tier_scores, fin_b1)
    res["B1_exact_binary"] = {"hot_comparison": hc,
                              "shortlist_identical_ordered": int(np.all(sl_b1 == sl_cur, axis=1).sum()),
                              "shortlist_identical_set": int(sum(set(a) == set(b) for a, b in zip(sl_b1, sl_cur, strict=True))),
                              "two_tier_final_comparison": cb1, "two_tier_quality": quality(fin_b1, base, rel)[1]}
    b1_ok = res["B1_exact_binary"]["shortlist_identical_set"] == len(queries) and cb1["identical_full_ranking_all"]

    # ---- Experiment B2: binarized query (approximate)
    ham = C.BinaryHamming(hot)
    hot_b2 = C.maxsim(queries.vectors, queries.offsets, ham.offsets, ham.sims, n_docs)
    sl_b2 = rank(hot_b2, C.CANDIDATES)
    fin_b2 = C.merge_tiers(hot_b2, sl_b2, C.rescore_grouped(queries, sl_b2, i8, "direct"))
    pq_b2, q_b2 = quality(fin_b2, base, rel)
    target = np.array([int(r[0]) for r in rel])
    in_sl = lambda sl: float(np.mean([target[i] in set(sl[i]) for i in range(len(sl))]))
    cb2, _, _, _ = rank_compare(tier_scores, fin_b2)
    res["B2_binarized_query"] = {
        "label": "APPROXIMATE (binarized query; different similarity)",
        "shortlist_overlap_with_current_mean": float(np.mean([len(set(a) & set(b)) / C.CANDIDATES for a, b in zip(sl_b2, sl_cur, strict=True)])),
        "labelled_page_in_shortlist": {"current": in_sl(sl_cur), "B2": in_sl(sl_b2)},
        "hot_only_quality": quality(hot_b2, base, rel)[1], "hot_only_quality_current_binary": quality(hot_cur, base, rel)[1],
        "two_tier_final_vs_current": cb2, "two_tier_quality": q_b2,
        "retention_difference_vs_current_two_tier": paired_diff(pq_b2, pq_tier_cur, base),
        "candidates_passed_to_rescoring": C.CANDIDATES}

    # ---- secondary: plain binary hot tier (all vectors)
    tier_p, _ = C.current_two_tier(hotp, cold, queries)
    hot_p_cur = ExactIndex(hotp).score(queries)
    sec = {"current": quality(tier_p, base, rel)[1]}
    for key, scorer in (("B1", C.BinaryLUT(hotp)), ("B2", C.BinaryHamming(hotp))):
        h = C.maxsim(queries.vectors, queries.offsets, scorer.offsets, scorer.sims, n_docs)
        sl = rank(h, C.CANDIDATES)
        f = C.merge_tiers(h, sl, C.rescore_grouped(queries, sl, i8, "direct"))
        sec[key] = {"quality": quality(f, base, rel)[1], "final_vs_current": rank_compare(tier_p, f)[0],
                    "shortlist_identical_set_vs_current": int(sum(set(a) == set(b) for a, b in
                                                                 zip(sl, rank(hot_p_cur, C.CANDIDATES), strict=True)))}
    res["secondary_plain_binary_hot"] = sec

    res["exact_int8_criterion_passed"] = bool(exact_ok)
    res["exact_binary_B1_criterion_passed"] = bool(b1_ok)
    res["seconds"] = time.time() - t_start
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "correctness.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("exact_int8_criterion_passed", "exact_binary_B1_criterion_passed",
                                          "current_two_tier_matches_Q4", "baseline_equals_E1_store")}),
          f"{res['seconds']:.0f}s", flush=True)
    return 0 if exact_ok else 1


if __name__ == "__main__":
    sys.exit(main())
