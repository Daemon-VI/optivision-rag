"""Q3G: margin-matched (and, as robustness, rank-matched) InfoVQA against DocVQA.

Reweights InfoVQA queries to DocVQA's distribution over the pooled relevance-margin
quintiles (reports/research/Q3/INTERVENTION.md), then measures pool retention, the
n = 225 certification SE and selection with the frozen E1 code (e1_1_study.run_cell).
Writes reports/research/Q3/intervention/<model>_<matching>.json.

    PYTHONPATH="src;scripts/research" python scripts/research/q3_intervention.py ColPali margin
"""

from __future__ import annotations

import json
import sys
import time

import e1_1_study as M
import e1_common as E
import numpy as np

Q3 = E.ROOT / "reports" / "research" / "Q3"
PAIRS = {"ColPali": ("colpali_docvqa", "colpali_infovqa"),
         "ColQwen2": ("colqwen2_docvqa", "colqwen2_infovqa"),
         "ColQwen2.5": ("colqwen25_docvqa", "colqwen25_infovqa")}
B = 2000


def rank_stratum(r):
    return np.where(r == 1, 0, np.where(r <= 5, 1, 2))


def strata(model, matching):
    doc, info = PAIRS[model]
    fd, fi = np.load(Q3 / "features" / f"{doc}.npz"), np.load(Q3 / "features" / f"{info}.npz")
    if matching == "margin":
        edges = np.quantile(np.concatenate([fd["base_margin_rel"], fi["base_margin_rel"]]), [0.2, 0.4, 0.6, 0.8])
        return (np.searchsorted(edges, fd["base_margin_rel"], side="right"),
                np.searchsorted(edges, fi["base_margin_rel"], side="right"), 5)
    return rank_stratum(fd["base_rank"]), rank_stratum(fi["base_rank"]), 3


def weights(sd, si, k):
    pd = np.bincount(sd, minlength=k) / len(sd)
    pi = np.bincount(si, minlength=k) / len(si)
    if (pi[pd > 0] == 0).any():
        raise SystemExit("a DocVQA stratum is empty on InfoVQA; matching impossible")
    return pd[si] / pi[si], pd, pi


def wstats(c, b, w):
    R = (w * c).sum() / (w * b).sum()
    mb = (w * b).sum() / w.sum()
    x = c - R * b
    mx = (w * x).sum() / w.sum()
    sd = np.sqrt((w * (x - mx) ** 2).sum() / w.sum())
    return float(R), float(sd), float(mb), float(sd / (np.sqrt(225) * mb))


def main(model, matching):
    t0 = time.time()
    doc, info = PAIRS[model]
    sd_, si_, k = strata(model, matching)
    w, pd, pi = weights(sd_, si_, k)
    D, I = E.Store(E.load_store(doc), "labels"), E.Store(E.load_store(info), "labels")
    p = w / w.sum()
    ones_d, ones_i = np.ones(len(D.base)), np.ones(len(I.base))
    rng = np.random.default_rng(0)
    idx_d = rng.integers(0, len(D.base), (B, len(D.base)))
    idx_w = rng.choice(len(I.base), (B, len(I.base)), p=p)
    cands = []
    for c in range(D.K):
        Rd, sdd, mbd, sed = wstats(D.pq[c], D.base, ones_d)
        Ri, sdi, mbi, sei = wstats(I.pq[c], I.base, ones_i)
        Rw, sdw, mbw, sew = wstats(I.pq[c], I.base, w)
        diff = D.pq[c][idx_d].sum(1) / D.base[idx_d].sum(1) - I.pq[c][idx_w].sum(1) / I.base[idx_w].sum(1)
        cands.append({"position": c, "label": D.labels[c], "R_doc": Rd, "R_info": Ri, "R_info_matched": Rw,
                      "R_doc_minus_matched_ci": [Rd - Rw, *np.quantile(diff, [0.025, 0.975]).tolist()],
                      "se_doc": sed, "se_info": sei, "se_info_matched": sew,
                      "numerator_sd": [sdd, sdi, sdw], "mean_b": [mbd, mbi, mbw]})

    def logratio(a, b):
        v = [np.log(x[a] / x[b]) for x in cands if x[a] > 0 and x[b] > 0]
        return float(np.median(v)), len(v)

    lu, nu = logratio("se_doc", "se_info")
    lm, nm = logratio("se_doc", "se_info_matched")
    truth = (I.pq * w).sum(axis=1) / (w * I.base).sum()
    prob = M.Problem(I.families, I.bytes, I.vectors, I.compression, truth)
    rows = []
    for n in M.NS:
        def draw(seed, n=n):
            idx = np.random.default_rng(seed).choice(len(I.base), size=n, p=p)
            return I.pq[:, idx], I.base[idx]
        rows += M.summarise(prob, M.run_cell(prob, draw, n, "library"), n)
    out = {"model": model, "matching": matching,
           "label": "MEASURED reweighting of MEASURED per-query results; selection rows are SIMULATED resampling",
           "strata_share_doc": pd.tolist(), "strata_share_info": pi.tolist(),
           "effective_sample_size_info": float(w.sum() ** 2 / (w ** 2).sum()), "n_info": len(w),
           "median_log_se_ratio_unmatched": lu, "median_log_se_ratio_matched": lm, "candidates_used": [nu, nm],
           "share_of_log_se_gap_closed": 1 - lm / lu if lu else None,
           "R_doc_below_matched_beyond_ci": sum(1 for x in cands if x["R_doc_minus_matched_ci"][2] < 0),
           "R_doc_above_matched_beyond_ci": sum(1 for x in cands if x["R_doc_minus_matched_ci"][1] > 0),
           "candidates": cands, "selection_rows": rows, "environment": E.environment(), "seconds": time.time() - t0}
    (Q3 / "intervention").mkdir(parents=True, exist_ok=True)
    (Q3 / "intervention" / f"{model}_{matching}.json").write_text(json.dumps(out, indent=1, default=float),
                                                                     encoding="utf-8")
    print(model, matching, "gap closed", out["share_of_log_se_gap_closed"], f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
