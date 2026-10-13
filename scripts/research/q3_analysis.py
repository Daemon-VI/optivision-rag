"""Q3 observational analyses 3A-3F and the certification decomposition (reports/research/Q3/PLAN.md).

Reads the E1 store (read only) and reports/research/Q3/features/*.npz; writes
reports/research/Q3/observational.json. Bootstrap: 2,000 query resamples, seed 0.
"""

from __future__ import annotations

import json

import e1_common as E
import numpy as np

Q3 = E.ROOT / "reports" / "research" / "Q3"
VLM = {"ColPali": ("colpali_docvqa", "colpali_infovqa"),
       "ColQwen2": ("colqwen2_docvqa", "colqwen2_infovqa"),
       "ColQwen2.5": ("colqwen25_docvqa", "colqwen25_infovqa")}
REP = (1, 16, 27, 30)
N_CAL = 225
B = 2000


def boot(fn, *arrays, seed=0):
    """Point and 95% percentile interval of fn over query resamples (arrays share axis 0)."""
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    vals = [fn(*[a[idx] for a in arrays]) for idx in rng.integers(0, n, size=(B, n))]
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return [float(fn(*arrays)), float(lo), float(hi)]


def boot_diff(fn, xa, xb, seed=0):
    """fn(a) - fn(b) for two independent query sets (tuples of arrays), 95% interval."""
    rng = np.random.default_rng(seed)
    na, nb = len(xa[0]), len(xb[0])
    vals = []
    for _ in range(B):
        ia, ib = rng.integers(0, na, na), rng.integers(0, nb, nb)
        vals.append(fn(*[a[ia] for a in xa]) - fn(*[b[ib] for b in xb]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return [float(fn(*xa) - fn(*xb)), float(lo), float(hi)]


def quant(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), **{f"q{int(q * 100)}": float(np.quantile(x, q)) for q in qs}}


def safe_div(a, b):
    """None when the denominator is 0 (a lossless candidate has zero spread)."""
    return None if b == 0 else a / b


def ratio(c, b):
    return c.mean() / b.mean()


def loss_profile(b, c):
    d = b - c
    lose = d > 0
    drop = (b > 0) & (c == 0)
    shift = lose & ~drop
    mass = d[lose].sum()
    worst = np.sort(d[lose])[::-1]
    k = max(1, int(np.ceil(0.1 * lose.sum()))) if lose.any() else 0
    drop_share = float(d[drop].sum() / mass) if mass > 0 else 0.0
    top10 = float(worst[:k].sum() / mass) if mass > 0 else 0.0
    mode = ("B: few catastrophic" if drop_share >= 0.7 and top10 >= 0.5 else
            "A: widespread small" if drop_share < 0.3 else "C: mixture") if mass > 0 else "no loss"
    return {"unchanged": float((d == 0).mean()), "rank_shift": float(shift.mean()), "drop_out": float(drop.mean()),
            "gain": float((d < 0).mean()), "drop_out_share_of_loss_mass": drop_share,
            "worst10pct_share_of_loss_mass": top10, "loss_p95": float(np.quantile(d, 0.95)),
            "mean_loss": float(d.mean()), "failure_mode": mode}


def cert(b, c):
    R = ratio(c, b)
    num_sd = float(np.std(c - R * b, ddof=1))
    return {"R": float(R), "numerator_sd": num_sd, "mean_b": float(b.mean()),
            "se_n225": num_sd / (np.sqrt(N_CAL) * b.mean())}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def vlm_dataset(name):
    st = E.Store(E.load_store(name), "labels")
    f = dict(np.load(Q3 / "features" / f"{name}.npz"))
    b = st.base
    out = {"n_queries": len(b), "n_pages": len(f["page_vectors"])}
    rank = f["base_rank"]
    out["3A"] = {
        "b": quant(b), "b_mean_ci": boot(np.mean, b), "hit1": boot(np.mean, (rank == 1).astype(float)),
        "relevant_outside_top5": float((rank > 5).mean()), "rank": quant(rank),
        "margin_rel": quant(f["base_margin_rel"]), "margin_12": quant(f["base_margin_12"]),
        "near_ties": quant(f["base_near_ties"]), "query_tokens": quant(f["L"]),
        "s1_per_token": quant(f["base_s1"] / f["L"]),
    }
    miss1 = rank > 1
    out["3E"] = {"labelled_pages_per_query": 1, "unlabelled_pages": int(f["page_unlabelled"].sum()),
                 "top1_is_unlabelled_when_top1_wrong": float(f["base_top1_unlabelled"][miss1].mean()) if miss1.any() else None,
                 "top1_is_unlabelled_all_queries": float(f["base_top1_unlabelled"].mean())}
    out["3F"] = {"vectors_per_page": quant(f["page_vectors"]),
                 "page_nearest_mean_cosine": quant(f["page_nearest_mean_cosine"])}
    cands = []
    for k in range(st.K):
        c = st.pq[k]
        row = {"position": k, "label": st.labels[k], "compression": float(st.compression[k]),
               "R_ci": boot(ratio, c, b), **cert(b, c), "loss": loss_profile(b, c)}
        cands.append(row)
    out["3B_candidates"] = cands
    rep = {}
    for k in REP:
        c = st.pq[k]
        d = b - c
        pos = b > 0
        strata = {"rank1": rank == 1, "rank2_5": (rank >= 2) & (rank <= 5)}
        rep[k] = {
            "label": st.labels[k],
            "loss_rate_by_rank": {s: float((d[m] > 0).mean()) if m.any() else None for s, m in strata.items()},
            "dropout_rate_by_rank": {s: float(((c == 0) & m).sum() / m.sum()) if m.any() else None for s, m in strata.items()},
            "spearman_margin_vs_loss_b_pos": boot(spearman, f["base_margin_rel"][pos], d[pos]),
        }
        if k in (27, 30):
            # Binary codes decode to +-1 vectors (norm sqrt(dim)), so post-compression score
            # magnitudes are on another scale than the float baseline. Only scale-free
            # quantities are reported: whether the labelled page is still first.
            m0, m1 = f["base_margin_rel"], f[f"c{k}_margin_rel"]
            rep[k]["3D_sign_change_share"] = float((np.sign(m0) != np.sign(m1)).mean())
            rep[k]["3D_top1_lost_share_of_hits"] = float(((m0 > 0) & (m1 <= 0)).sum() / max(1, (m0 > 0).sum()))
    out["3C_3D"] = rep
    out["_margin"], out["_loss"] = f["base_margin_rel"], {k: b - st.pq[k] for k in REP}
    return out


def margin_quintiles(pair):
    """Loss rate by relevance-margin quintile, quintile edges pooled over both splits of a model."""
    allm = np.concatenate([p["_margin"] for p in pair])
    edges = np.quantile(allm, [0.2, 0.4, 0.6, 0.8])
    res = {}
    for split, p in zip(("DocVQA", "InfoVQA"), pair, strict=True):
        qi = np.searchsorted(edges, p["_margin"], side="right")
        res[split] = {"share_of_queries_per_quintile": [float((qi == i).mean()) for i in range(5)],
                      "loss_rate_per_quintile": {str(k): [float((p["_loss"][k][qi == i] > 0).mean()) if (qi == i).any() else None
                                                          for i in range(5)] for k in REP}}
    res["edges"] = edges.tolist()
    return res


def main():
    out = {"artifacts": {"E1_store_commit": "b18ae19", "features": "reports/research/Q3/features"}, "models": {}}
    for model, (doc, info) in VLM.items():
        d, i = vlm_dataset(doc), vlm_dataset(info)
        sd, si = E.Store(E.load_store(doc), "labels"), E.Store(E.load_store(info), "labels")
        fd, fi = dict(np.load(Q3 / "features" / f"{doc}.npz")), dict(np.load(Q3 / "features" / f"{info}.npz"))
        diffs = {
            "b_mean": boot_diff(np.mean, (sd.base,), (si.base,)),
            "hit1": boot_diff(lambda r: np.mean(r == 1), (fd["base_rank"],), (fi["base_rank"],)),
            "margin_rel_median": boot_diff(np.median, (fd["base_margin_rel"],), (fi["base_margin_rel"],)),
        }
        per_cand = []
        for k in range(sd.K):
            dR = boot_diff(ratio, (sd.pq[k], sd.base), (si.pq[k], si.base))
            per_cand.append({"position": k, "label": sd.labels[k], "R_doc_minus_info": dR,
                             "se_ratio_doc_over_info": safe_div(d["3B_candidates"][k]["se_n225"], i["3B_candidates"][k]["se_n225"]),
                             "numerator_sd_ratio": safe_div(d["3B_candidates"][k]["numerator_sd"], i["3B_candidates"][k]["numerator_sd"]),
                             "denominator_ratio_info_over_doc": i["3B_candidates"][k]["mean_b"] / d["3B_candidates"][k]["mean_b"]})
        mq = margin_quintiles((d, i))
        for x in (d, i):
            x.pop("_margin")
            x.pop("_loss")
        out["models"][model] = {"DocVQA": d, "InfoVQA": i, "doc_minus_info": diffs,
                                "per_candidate_doc_vs_info": per_cand, "3C_margin_quintiles": mq}
    # SciFact: store-only, separate (nDCG@10, several relevant documents)
    st = E.Store(E.load_store("scifact"), "labels")
    meta = E.load_store("scifact")["meta"]
    out["SciFact_separate"] = {
        "metric": st.metric, "relevant_per_query": meta["relevant_per_query"], "b": quant(st.base),
        "candidates": [{"position": k, "label": st.labels[k], "R_ci": boot(ratio, st.pq[k], st.base),
                        **cert(st.base, st.pq[k]), "loss": loss_profile(st.base, st.pq[k])} for k in range(st.K)],
        "note": "loss categories assume one relevant document; with several, 'drop_out' means nDCG@10 fell to 0",
    }
    (Q3 / "observational.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("written", Q3 / "observational.json")


if __name__ == "__main__":
    main()
