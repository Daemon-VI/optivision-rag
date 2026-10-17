import json,numpy as np
from scipy import stats
M=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
labs=['hierarchical_merge(0.5)','hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','adaptive_merge(0.6)']
rng=np.random.default_rng(1)
for l in labs:
    a={};b={}
    for m,v in M.items():
        for ds,t in (('docvqa',a),('infovqa',b)):
            for k in v:
                if ds in k:
                    for r in v[k]['rows']:
                        if r['label']==l: t[m]=1-r['retention']
    sh=sorted(set(a)&set(b));x=np.array([a[m] for m in sh]);y=np.array([b[m] for m in sh])
    bs=[]
    for _ in range(4000):
        i=rng.integers(0,len(sh),len(sh));bs.append(stats.spearmanr(x[i],y[i])[0])
    print(l,len(sh),'rho %.2f'%stats.spearmanr(x,y)[0],np.nanpercentile(bs,[2.5,97.5]).round(2),'pearson %.2f'%stats.pearsonr(x,y)[0],'log-rank-free kendall %.2f'%stats.kendalltau(x,y)[0])
