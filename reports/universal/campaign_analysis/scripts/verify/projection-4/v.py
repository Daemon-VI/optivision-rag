import json,numpy as np
from scipy.stats import mannwhitneyu
from optivision.adapters import MODELS
m=json.load(open('src/optivision/presets/measured.json'))['models']
def get(lab):
    out={}
    for mid,ds in m.items():
        v=[r['retention'] for d in ds.values() for r in d['rows'] if r['label']==lab]
        if v: out[mid]=np.mean(v)
    return out
rnd=get('project(32, random)');pca=get('project(32)')
mod={k:MODELS[k].modality if k in MODELS else '?' for k in m}
dim={k:MODELS[k].dimension if k in MODELS else None for k in m}
print(len(rnd),len(pca))
for mod_ in ('text','image'):
    ks=[k for k in rnd if mod[k]==mod_ and k in pca]
    print(mod_,len(ks),'rnd',np.mean([rnd[k] for k in ks]),'pca',np.mean([pca[k] for k in ks]),'gap',np.mean([pca[k]-rnd[k] for k in ks]))
tx=[k for k in rnd if mod[k]=='text' and k in pca]
for k in sorted(tx,key=lambda k:rnd[k]): print(k,dim[k],round(rnd[k],3),round(pca[k],3))
im=[k for k in rnd if mod[k]=='image' and k in pca]
print('image low:',sorted([(round(rnd[k],3),k,dim[k]) for k in im])[:6])
gt=np.array([pca[k]-rnd[k] for k in tx]);gi=np.array([pca[k]-rnd[k] for k in im])
print(mannwhitneyu(gt,gi,alternative='greater'))
rng=np.random.default_rng(0)
b=[rng.choice(gt,len(gt)).mean()-rng.choice(gi,len(gi)).mean() for _ in range(5000)]
print(np.percentile(b,[2.5,97.5]))
# dims
print('dim by modality',{mm:sorted(set(dim[k] for k in rnd if mod[k]==mm)) for mm in ('text','image')})
# k/d dependence in image: other random k
labs=sorted({r['label'] for ds in m.values() for d in ds.values() for r in d['rows'] if r['label'].startswith('project(') and 'random' in r['label'] and '>' not in r['label']})
print(labs)
for l in labs:
    g=get(l)
    print(l,{mm:(len([k for k in g if mod[k]==mm]),round(np.mean([g[k] for k in g if mod[k]==mm]),3)) for mm in ('text','image') if any(mod[k]==mm for k in g)})
# dedupe: drop LFM/ModernBERT siblings, and by-dim: text models with dim>=128
for k in im:
    pass
print('image dims small',sorted((dim[k],round(rnd[k],3)) for k in im))
