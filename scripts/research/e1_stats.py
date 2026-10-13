"""Selection and certification procedures compared in E1.1.

Notation for one calibration sample of n queries: b_i is the float32 index's
nDCG@5 and c_ki the k-th candidate's, both in [0, 1]. A candidate's retention
under a query distribution Q is R_Q(k) = E_Q[c_k] / E_Q[b] (E_Q[b] > 0).

Key identity: R_Q(k) >= T  <=>  E_Q[c_k - T b] >= 0, and c_k - T b lies in
[-T, 1]. So "candidate k meets the target" is a statement about the mean of a
bounded variable. With Y = (c - T b + T) / (1 + T) in [0, 1] and m0 = T / (1 + T),

    H0_k:  E_Q[Y_k] <= m0   (equivalently R_Q(k) <= T)

and rejecting H0_k is the claim "R_Q(k) > T".

``betting_pvalues`` is a one-sided test of H0 by testing by betting
(Waudby-Smith & Ramdas, JRSSB 2023): wealth K_t = prod_s (1 + lam_s (Y_s - m0))
with a predictable lam_s in [0, c/m0]. Under H0 and i.i.d. sampling,
E[1 + lam_t (Y_t - m0) | past] = 1 + lam_t (E Y - m0) <= 1, so K_t is a
non-negative supermartingale and, by Ville's inequality,
P(max_t K_t >= 1/delta) <= delta. p = min(1, 1 / max_t K_t) is therefore a valid
p-value: P(p <= delta) <= delta under H0, for every n, with no normal
approximation. The bet is aGRAPA, lam_t = (mu_{t-1} - m0) / (var_{t-1} +
(mu_{t-1} - m0)^2) clipped to [0, c/m0], with c = 0.9 and the regularised running
mean and variance of WSR (prior mean 1/2, prior variance 1/4). Validity holds for
any c in (0, 1). c was set at the design stage from synthetic power checks only
(reports/research/E1/DESIGN.md), before any real-data E1.1 run: c = 0.5 could not
certify even a lossless candidate at T = 0.97 with 225 queries. It is not tuned
on the development datasets.

``lower_bound_all`` is a vectorised copy of evaluation.retention_lower_bound (the
bootstrap used by calibrate()); E1.1 checks it against the library.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

BET_CAP = 0.9


def to_y(c: np.ndarray, b: np.ndarray, target: float) -> np.ndarray:
    return (c - target * b + target) / (1.0 + target)


def betting_pvalues(y: np.ndarray, m0: float) -> np.ndarray:
    """One-sided p-values for H0: E[Y] <= m0, over the last axis (observation order)."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[-1]
    lead = y.shape[:-1]
    log_k = np.zeros(lead)
    log_kmax = np.zeros(lead)
    s1 = np.zeros(lead)  # running sum of Y
    s2 = np.zeros(lead)  # running sum of (Y - mu_hat)^2
    cap = BET_CAP / m0
    for t in range(n):
        mu = (0.5 + s1) / (t + 1)
        var = (0.25 + s2) / (t + 1)
        d = mu - m0
        lam = np.clip(d / (var + d * d), 0.0, cap)
        yt = y[..., t]
        log_k = log_k + np.log1p(lam * (yt - m0))
        log_kmax = np.maximum(log_kmax, log_k)
        s1 = s1 + yt
        mu_t = (0.5 + s1) / (t + 2)
        s2 = s2 + (yt - mu_t) ** 2
    return np.minimum(1.0, np.exp(-log_kmax))


def lower_bound_stats(pq: np.ndarray, base: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Point retention and bootstrap ratios' sd for every row of pq (K, m).

    Mirrors evaluation.retention_lower_bound: rng = default_rng(seed),
    idx = rng.integers(0, m, (n_boot, m)), ratio = mean(c[idx]) / mean(b[idx]).
    Inputs must be finite (true for every E1 array).
    """
    m = base.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    bm = base[idx].mean(axis=1)
    point = pq.mean(axis=1) / base.mean()
    cm = pq[:, idx].mean(axis=2)  # (K, n_boot)
    ratios = np.where(bm > 0, cm / np.where(bm > 0, bm, 1.0), np.nan)
    return point, ratios


def lower_bound_all(pq, base, confidence: float, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    point, ratios = lower_bound_stats(pq, base, n_boot, seed)
    if base.mean() == 0:
        return np.full(pq.shape[0], np.nan)
    if confidence <= 0.99:
        return np.nanquantile(ratios, 1.0 - confidence, axis=1)
    z = NormalDist().inv_cdf(confidence)
    return point - z * np.nanstd(ratios, axis=1)


def rank_key(bytes_, vectors, k, position):
    return (bytes_[k], vectors[k], position)


def walk_select(bounds: np.ndarray, families: list[str], bytes_, vectors, target: float):
    """calibrate()'s walk on precomputed per-candidate bounds (early stop per family)."""
    judged, stopped = [], set()
    for k, lo in enumerate(bounds):
        if families[k] in stopped:
            continue
        ok = bool(lo >= target)
        judged.append((k, ok))
        if not ok:
            stopped.add(families[k])
    feas = [(k, pos) for pos, (k, ok) in enumerate(judged) if ok]
    if not feas:
        return None
    return min(feas, key=lambda t: rank_key(bytes_, vectors, t[0], t[1]))[0]


def fixed_sequence_by_family(pvals: np.ndarray, families: list[str], level_per_family: float) -> list[int]:
    """Reject along each family's pre-specified order until the first non-rejection."""
    rejected, stopped = [], set()
    for k, p in enumerate(pvals):
        f = families[k]
        if f in stopped:
            continue
        if p <= level_per_family:
            rejected.append(k)
        else:
            stopped.add(f)
    return rejected


def fixed_sequence(pvals: np.ndarray, order: np.ndarray, level: float) -> list[int]:
    out = []
    for k in order:
        if pvals[k] <= level:
            out.append(int(k))
        else:
            break
    return out


def smallest(cands: list[int], bytes_, vectors):
    """Smallest stored size among cands (ties: fewer vectors, then earlier position)."""
    if not cands:
        return None
    return min(cands, key=lambda k: rank_key(bytes_, vectors, k, k))
