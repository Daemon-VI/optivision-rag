import json,numpy as np
from scipy import stats
from optivision.adapters import MODELS
rng=np.random.default_rng(1)
m=json.load(open('src/optivision/presets/measured.json'))['models']
info={x.model_id:x for x in (MODELS.values() if isinstance(MODELS,dict) else MODELS)}
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled'
ids=[k for k in m if info[k].modality=='image' and info[k].dimension==128]
H='hierarchical_merge(0.25)'
labs=['int8(per_vector)','int4(mean)','binary','lloyd2(7)',H,H+' > int4(mean)',H+' > binary',H+' > lloyd2(7)']
R={}
for k in ids:
    for ds in (D,I):
        have={r['label']:r for r in m[k][ds]['rows']}
        for l in labs:
            if l not in have: print('missing',k,ds[:8],l)
            else: R[k,ds,l]=have[l]['retention']
# use available-label mean
def fr(k,ds,ls): 
    v=[R[k,ds,l] for l in ls if (k,ds,l) in R]; return np.mean(1-np.minimum(v,1)),len(v)
# core 4 labels all models
core=['int8(per_vector)','int4(mean)','binary','lloyd2(7)']
x=np.array([fr(k,D,core)[0] for k in ids]);y=np.array([fr(k,I,core)[0] for k in ids])
bs=[stats.spearmanr(x[i],y[i])[0] for i in (rng.integers(0,len(ids),len(ids)) for _ in range(4000))]
print('core4 n',len(ids),'rho %.3f CI'%stats.spearmanr(x,y)[0],np.nanpercentile(bs,[2.5,97.5]))
x=np.array([fr(k,D,labs)[0] for k in ids]);y=np.array([fr(k,I,labs)[0] for k in ids])
print('avail-labels n',len(ids),'rho %.3f'%stats.spearmanr(x,y)[0])
bs=[stats.spearmanr(x[i],y[i])[0] for i in (rng.integers(0,len(ids),len(ids)) for _ in range(4000))]
print(np.nanpercentile(bs,[2.5,97.5]))
f16=np.array(['float16' in (info[k].evidence or '').lower() for k in ids])
pool=(x+y)/2
print('f16 n',f16.sum(),'MWU pooled-avg p',stats.mannwhitneyu(pool[f16],pool[~f16]).pvalue,'means',pool[f16].mean(),pool[~f16].mean())
print('MWU D',stats.mannwhitneyu(x[f16],x[~f16]).pvalue,'I',stats.mannwhitneyu(y[f16],y[~f16]).pvalue)
# heterogeneity ratio: var of retention across models vs mean sampling var from CI
for l in labs+[H+' > project(128)']:
    for ds in (D,I):
        v=[];s=[]
        for k in ids:
            for r in m[k][ds]['rows']:
                if r['label']==l: v.append(r['retention']); lo,hi=r['retention_interval']; s.append(((hi-lo)/3.92)**2)
        if v: print(l[:38],ds[7:12],len(v),'ratio %.2f'%(np.var(v,ddof=1)/np.mean(s)))
