import json, numpy as np, collections, sys
from optivision.adapters import MODELS
m = json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
rng = np.random.default_rng(0)
mods = sorted(m)
info = {k: MODELS[k] for k in mods}
def band(d):
    return '<=96' if d <= 96 else '128' if d == 128 else '129-319' if d < 320 else '320-640' if d <= 640 else '641-2047' if d < 2048 else '>=2048'
print("models by dim/mod/family")
for k in mods:
    i = info[k]; print(k, i.dimension, i.modality, i.family, band(i.dimension), len(m[k]))
# units
labels = None
units = []  # (model, ds)
R = {}
for k in mods:
    for ds, d in m[k].items():
        for r in d['rows']:
            R[(k, ds, r['label'])] = r['retention']
        units.append((k, ds))
allab = collections.Counter(l for (_, _, l) in R)
labs = [l for l, c in allab.items() if c >= 0.8 * len(units)]
print(len(units), len(allab), len(labs))
labs = [l for l in labs if l not in ('float32',)]
L = len(labs)
ui = {u: j for j, u in enumerate(units)}
Y = np.full((len(units), L), np.nan)
for (k, ds, l), v in R.items():
    if l in labs: Y[ui[(k, ds)], labs.index(l)] = v
mi = [mods.index(u[0]) for u in units]
mi = np.array(mi)
mod = np.array([info[u[0]].modality for u in units])
dim = np.array([info[u[0]].dimension for u in units])
fam = np.array([info[u[0]].family for u in units])
org = np.array([u[0].split('/')[0] for u in units])
bd = np.array([band(d) for d in dim])
print(collections.Counter(zip(bd, mod)))

def brier_models(t, group, sib, a=1.0):
    """return per held-out model sum sq err and count, for group strategy"""
    P = (Y >= t).astype(float); V = ~np.isnan(Y)
    out = {}
    for h in range(len(mods)):
        te = mi == h
        # exclusion set of peers
        ex = mi == h
        if sib == 'fam': ex = ex | (fam == fam[te][0])
        elif sib == 'org': ex = ex | (org == org[te][0])
        tr = ~ex
        # global
        g_pass = np.nansum(np.where(V, P, 0)[tr], 0); g_n = V[tr].sum(0)
        gp = (g_pass + a * 0.5 * 2) / (g_n + 2 * a) if False else (g_pass + 1) / (g_n + 2)
        errs = []
        for u in np.where(te)[0]:
            if group == 'global': peers = tr
            else:
                key = group(u)
                peers = tr & key_arr[group_name][:, None].ravel() if False else tr & (keys == keys[u])
            # shrink to global prior
            ps = np.where(V, P, 0)[peers].sum(0); ns = V[peers].sum(0)
            pr = (ps + 2 * gp) / (ns + 2)
            ok = V[u]
            errs.append(((pr - P[u]) ** 2)[ok])
        out[h] = np.concatenate(errs)
    return out
def run(t, keyarr, sib):
    global keys
    keys = keyarr
    return brier_models(t, lambda u: 1, sib)
def boot(a, b, n=2000):
    hs = sorted(a)
    da = np.array([a[h].mean() - b[h].mean() for h in hs])  # per-model mean diff (model-weighted)
    idx = rng.integers(0, len(hs), (n, len(hs)))
    bs = da[idx].mean(1)
    return da.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), da
keysets = {
    'global': np.zeros(len(units), int).astype(str),
    'dimxmod': np.char.add(bd.astype(str), mod),
    'mod': mod, 'fam': fam, 'dim': bd,
    'famxmod': np.char.add(fam, mod),
}
if __name__ == '__main__':
    for sib in ['self', 'fam']:
        for t in [0.95, 0.97, 0.99]:
            res = {n: run(t, k, sib) for n, k in keysets.items()}
            print('sib', sib, 't', t, 'mean Brier model-weighted:', {n: round(np.mean([v.mean() for v in r.values()]), 4) for n, r in res.items()})
            for A, B in [('dimxmod', 'global'), ('fam', 'dimxmod'), ('famxmod', 'dimxmod'), ('mod', 'global'), ('dimxmod', 'mod'), ('dim', 'dimxmod')]:
                d, lo, hi, da = boot(res[A], res[B])
                print(f'  {A} - {B}: {d:+.4f} [{lo:+.4f},{hi:+.4f}]')
