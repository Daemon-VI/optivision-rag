import json, re, collections, sys, numpy as np
from optivision.adapters import MODELS
m = json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
rng = np.random.default_rng(1)
mods = sorted(m)
info = {k: MODELS[k] for k in mods}

def band(d):
    return '<=96' if d <= 96 else '128' if d == 128 else '129-319' if d < 320 else '320-640' if d <= 640 else '641-2047' if d < 2048 else '>=2048'

def norm(label, pipeline, dim):
    parts = []
    for s in pipeline['stages']:
        n = s['stage']
        if n == 'project':
            parts.append('pca%.3g%s' % (s['dim'] / dim, '' if s.get('method', 'pca') == 'pca' else 'R'))
        elif n == 'hierarchical_merge' or n == 'adaptive_merge':
            parts.append(label_stage(s))
        else:
            parts.append(label_stage(s))
    return '>'.join(parts) if parts else 'float32'

def label_stage(s):
    return s['stage'] + ''.join('%s=%s' % (a, b) for a, b in sorted(s.items()) if a != 'stage')

def stem(k):
    n = k.split('/')[1].lower()
    n = re.sub(r'-(hf|merged)$', '', n)
    n = re.sub(r'-v\d[\d.]*$', '', n)
    n = re.sub(r'-v0\.\d$', '', n)
    n = n.replace('colbertv2.0', 'colbertv2')
    return n

units = []
cells = {}
for k in mods:
    for ds, d in m[k].items():
        units.append((k, ds))
        for r in d['rows']:
            key = norm(r['label'], r['pipeline'], info[k].dimension)
            cells[(len(units) - 1, key)] = r['retention']
cnt = collections.Counter(key for (_, key) in cells)
MINU = int(sys.argv[1]) if len(sys.argv) > 1 else 10
keys = sorted(k for k, c in cnt.items() if c >= MINU and k != 'float32')
print('units', len(units), 'norm pipelines', len(cnt), 'used', len(keys))
Y = np.full((len(units), len(keys)), np.nan)
for (u, key), v in cells.items():
    if key in keys: Y[u, keys.index(key)] = v
mi = np.array([mods.index(u[0]) for u in units])
mod = np.array([info[u[0]].modality for u in units])
dim = np.array([info[u[0]].dimension for u in units])
fam = np.array([info[u[0]].family for u in units])
stm = np.array([stem(u[0]) for u in units])
bd = np.array([band(d) for d in dim])
V = ~np.isnan(Y)
print(collections.Counter(zip(bd, mod)))
print('stems with >1 model:', {s: sorted({mods[i] for i in np.unique(mi[stm == s])}) for s in set(stm) if len(set(mi[stm == s])) > 1})

# excluded pipelines at various trivial thresholds handled by 'keyset'
def predict(t, grp, sib, a=2.0):
    P = (np.where(V, Y, -1) >= t).astype(float)
    out = {}
    for h in range(len(mods)):
        te = np.where(mi == h)[0]
        ex = (mi == h)
        if sib == 'stem': ex = ex | (stm == stm[te[0]])
        tr = ~ex
        gp = ((P * V)[tr].sum(0) + 1) / (V[tr].sum(0) + 2)
        e = []; pp = []; yy = []
        for u in te:
            peers = tr & (grp == grp[u])
            pr = ((P * V)[peers].sum(0) + a * gp) / (V[peers].sum(0) + a)
            ok = V[u]
            e.append(((pr - P[u]) ** 2)[ok]); pp.append(pr[ok]); yy.append(P[u][ok])
        out[h] = (np.concatenate(e), np.concatenate(pp), np.concatenate(yy))
    return out

def bootdiff(A, B, n=2000):
    d = np.array([A[h][0].mean() - B[h][0].mean() for h in A])
    d_row = None
    idx = rng.integers(0, len(d), (n, len(d)))
    b = d[idx].mean(1)
    return d.mean(), *np.percentile(b, [2.5, 97.5]), d

groups = {
    'global': np.zeros(len(units), int).astype(str),
    'mod': mod,
    'dimxmod': np.char.add(bd, mod),
    'fam': fam,
    'famxmod': np.char.add(fam, mod),
    'dim': bd,
}
if __name__ == '__main__':
    for sib in ['self', 'stem']:
        for t in [0.95, 0.97, 0.99]:
            res = {n: predict(t, g, sib) for n, g in groups.items()}
            print('--- sib', sib, 't', t, {n: round(float(np.mean([v[0].mean() for v in r.values()])), 4) for n, r in res.items()})
            for A, B in [('dimxmod', 'global'), ('mod', 'global'), ('dimxmod', 'mod'), ('fam', 'dimxmod'), ('famxmod', 'dimxmod'), ('dim', 'dimxmod')]:
                d, lo, hi, da = bootdiff(res[A], res[B])
                print(f'  {A} - {B}: {d:+.4f} [{lo:+.4f},{hi:+.4f}]')
            if t == 0.95 and sib == 'stem':
                pr = np.concatenate([v[1] for v in res['dimxmod'].values()]); yy = np.concatenate([v[2] for v in res['dimxmod'].values()])
                for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .7), (.7, .9), (.9, 1.01)]:
                    s = (pr >= lo) & (pr < hi)
                    print('   rel', lo, hi, s.sum(), round(pr[s].mean(), 3) if s.sum() else None, round(yy[s].mean(), 3) if s.sum() else None)
