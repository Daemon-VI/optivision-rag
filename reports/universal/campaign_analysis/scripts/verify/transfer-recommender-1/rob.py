import sys
sys.argv = ['x', '10']
from an2 import *
groups['famxdim'] = np.char.add(np.char.add(fam, bd), mod)
fp16 = {mods.index(k) for k in mods if any('float16' in (d.get('note') or '') for d in m[k].values())}
print('float16-baseline models', len(fp16))
def summ(res, mask_models, A, B, n=2000):
    hs = [h for h in res[A] if h in mask_models]
    d = np.array([res[A][h][0].mean() - res[B][h][0].mean() for h in hs])
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return len(hs), d.mean(), *np.percentile(b, [2.5, 97.5])
allm = set(range(len(mods)))
img = {h for h in allm if info[mods[h]].modality == 'image'}
txt = allm - img
hband = {b: {h for h in allm if band(info[mods[h]].dimension) == b} for b in ['<=96', '128', '320-640', '>=2048']}
for t in [0.95, 0.97, 0.99]:
    res = {n: predict(t, g, 'stem') for n, g in groups.items()}
    print('=== t', t)
    for name, S in [('all', allm), ('image', img), ('text', txt), ('no fp16', allm - fp16), ('not>=2048', allm - hband['>=2048']),
                    ('band 128', hband['128']), ('band 320-640', hband['320-640']), ('band>=2048', hband['>=2048']), ('band<=96', hband['<=96'])]:
        for A, B in [('dimxmod', 'global'), ('dimxmod', 'mod'), ('fam', 'dimxmod'), ('famxdim', 'dimxmod')]:
            if len(S) < 2: continue
            print(f'  {name:14s} {A}-{B}: n={summ(res, S, A, B)[0]} {summ(res, S, A, B)[1]:+.4f} [{summ(res, S, A, B)[2]:+.4f},{summ(res, S, A, B)[3]:+.4f}]')
# image only: re-fit with image-only peers is same as dimxmod; global among image-only peers:
# leave-one-model-out removal of each model, effect of dimxmod-global
res = {n: predict(0.95, g, 'stem') for n, g in groups.items()}
d = np.array([res['dimxmod'][h][0].mean() - res['global'][h][0].mean() for h in range(len(mods))])
print('per-model diff dimxmod-global t=.95: min', d.min(), 'max', d.max(), 'share<0', (d < 0).mean())
order = np.argsort(d)
print([(mods[i], round(d[i], 3)) for i in order[:3]], [(mods[i], round(d[i], 3)) for i in order[-3:]])
# drop top-3 most negative models and recompute mean
print('mean excluding 5 most favourable', np.sort(d)[5:].mean())
# sign test
from scipy.stats import binomtest
print('sign test', binomtest(int((d < 0).sum()), int((d != 0).sum())).pvalue)
# dim-128 image held-out only: dimxmod == mod ?
print('identical on 128 image?', np.mean([abs(res['dimxmod'][h][0].mean() - res['mod'][h][0].mean()) for h in hband['128'] & img]))
