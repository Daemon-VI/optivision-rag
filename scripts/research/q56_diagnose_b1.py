"""Q5/Q6: diagnose the exact-binary (B1, LUT) ranking discordances against the current
decode-first binary scan with a float64 reference (PLAN.md section 4). Writes
reports/research/Q5Q6/b1_diagnosis.json.
"""

from __future__ import annotations

import json

import e1_common as E
import numpy as np
import q56_common as C

from optivision.scoring import rank
from optivision.storage import ExactIndex

U = 2.0 ** -24


def main():
    ds = C.load()
    corpus, queries = ds.corpus, ds.queries
    hot, _, _ = C.build(corpus)
    lut = C.BinaryLUT(hot)
    cur = ExactIndex(hot).score(queries)
    new = C.maxsim(queries.vectors, queries.offsets, lut.offsets, lut.sims, len(corpus))
    ra, rb = rank(cur), rank(new)
    bad = np.flatnonzero(~np.all(ra == rb, axis=1))
    signs = None
    out = []
    for qi in bad:
        pos = int(np.flatnonzero(ra[qi] != rb[qi])[0])
        p1, p2 = int(ra[qi, pos]), int(rb[qi, pos])
        x = queries.vectors[queries.offsets[qi]:queries.offsets[qi + 1]].astype(np.float64)
        refs, bounds = [], []
        for p in (p1, p2):
            lo, hi = int(hot.offsets[p]), int(hot.offsets[p + 1])
            signs = hot.quantizer.decode(hot.codes[lo:hi], C.DIM).astype(np.float64)
            s = x @ signs.T
            refs.append(float(s.max(axis=1).sum()))
            bounds.append(float(np.abs(x).sum()))  # sum |x_j * s_j| along any path = sum |x_j|
        tol = 2 * C.DIM * U * max(bounds) * 1.01
        out.append({"query": int(qi), "first_differing_rank": pos + 1, "pages": [p1, p2],
                    "current_scores": [float(cur[qi, p1]), float(cur[qi, p2])],
                    "new_scores": [float(new[qi, p1]), float(new[qi, p2])], "float64_reference": refs,
                    "reference_gap": refs[0] - refs[1], "rounding_tolerance": tol,
                    "within_rounding": bool(abs(refs[0] - refs[1]) <= tol)})
    sl_cur, sl_new = rank(cur, C.CANDIDATES), rank(new, C.CANDIDATES)
    res = {"discordant_queries": len(out), "all_within_rounding": all(d["within_rounding"] for d in out),
           "first_differing_rank_min": min((d["first_differing_rank"] for d in out), default=None),
           "shortlist_set_identical": int(sum(set(a) == set(b) for a, b in zip(sl_cur, sl_new, strict=True))),
           "float64_agrees_with_current": int(sum(d["float64_reference"][0] >= d["float64_reference"][1] for d in out)),
           "diagnosis": out}
    (E.ROOT / "reports" / "research" / "Q5Q6" / "b1_diagnosis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print({k: v for k, v in res.items() if k != "diagnosis"})


if __name__ == "__main__":
    main()
