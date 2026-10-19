import json,numpy as np
from scipy.stats import spearmanr,rankdata
from optivision.adapters import MODELS
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
rows=[]
for m,ds in d.items():
    mi=MODELS.get(m)
    for k,v in ds.items():
        if 'docvqa' not in k and 'infovqa' not in k: continue
        bv=None;h={}
        for r in v['rows']:
            if r['label']=='float32': bv=r['vectors_per_doc']
            if r['label'].startswith('hierarchical_merge(') and '>' not in r['label']:
                h[float(r['label'][19:-1])]=1-r['retention']
        rows.append(dict(m=m,ds='doc' if 'docvqa' in k else 'info',bv=bv,h=h,mod=getattr(mi,'modality','?'),fam=getattr(mi,'family','?')))
print(len(rows),set(r['mod'] for r in rows))
img=[r for r in rows if r['mod']=='image']
print(len(img))
rng=np.random.default_rng(0)
def pooled(R,k):
    R=[r for r in R if k in r['h'] and r['bv']]
    x=[];y=[]
    for ds in ('doc','info'):
        S=[r for r in R if r['ds']==ds]
        if len(S)<3: continue
        x+=list(rankdata([r['bv'] for r in S])/len(S)-.5);y+=list(rankdata([r['h'][k] for r in S])/len(S)-.5)
    return spearmanr(x,y)[0],len(x)
for k in (0.5,0.33,0.25,0.1):
    out=[]
    for ds in ('doc','info'):
        S=[r for r in img if r['ds']==ds and k in r['h']]
        out.append((ds,len(S),spearmanr([r['bv'] for r in S],[r['h'][k] for r in S])))
    print(k,[(a,n,round(c[0],2),round(c[1],3)) for a,n,c in out],'pooled',pooled(img,k))
    # LOO ColNetra, and family bootstrap
    R=[r for r in img if 'ColNetra' not in r['m']]
    print('  noColNetra',pooled(R,k))
    fams=sorted(set(r['fam'] for r in img));bs=[]
    for _ in range(1000):
        pick=rng.choice(fams,len(fams));R2=[r for f in pick for r in img if r['fam']==f]
        try:bs.append(pooled(R2,k)[0])
        except: pass
    print('  fam-boot CI',np.nanpercentile(bs,[2.5,97.5]),len(fams))
    mods=sorted(set(r['m'] for r in img));bs=[]
    for _ in range(1000):
        pick=rng.choice(mods,len(mods));R2=[r for m in pick for r in img if r['m']==m]
        bs.append(pooled(R2,k)[0])
    print('  model-boot CI',np.nanpercentile(bs,[2.5,97.5]))
# family structure
from collections import defaultdict
g=defaultdict(list)
for r in img:
    if 0.1 in r['h']: g[r['fam']].append((r['bv'],r['h'][0.1]))
for f,v in g.items():print(f,len(v),round(np.mean([a for a,b in v])),round(np.mean([b for a,b in v]),4))
# within-family
for fam,v in g.items():
    if len(v)>=8:print(fam,spearmanr(*zip(*v)))
